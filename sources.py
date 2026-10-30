# -*- coding: utf-8 -*-
"""Frame sources for the pushbroom simulator: HDF5 raw spectrograms and ENVI cubes.

Every source is turned into what a line-scan camera delivers: a C-contiguous stack of
2-D frames ``(n_frames, slit, spectral)`` in the file's own dtype (no float conversion).

* HDF5 (``*.h5``/``*.hdf5``) - the "Keep Spectrogram" files of the HSI project: dataset
  ``spectrogram`` ``(frame, slit, spectral)`` with raw sensor frames, root attributes
  holding the scan metadata (camera bin, exposure, gain, ``raw_inverted``, m/c ...).
  Stage ``'raw'``: dark, hot pixel, flip, Smile and binning still have to be applied.
* ENVI (``*.hdr`` + data file, BSQ/BIL/BIP) - cubes ``lines x samples x bands``. HSI writes
  ``lines = slit``, ``samples = scan``, ``bands = wavelength``; each scan position becomes
  one frame ``(slit, bands)`` (``scan_axis='samples'``; ``'lines'`` for the other convention).
  With a ``wavelength`` list the cube is already corrected and binned (stage ``'binned'``),
  without one it is treated as raw sensor frames.

Frames are materialised once in frame-major order so that reading frame ``i`` is one
contiguous copy (needed for >1000 frames/s): compressed HDF5 chunks are decoded in
parallel and BSQ/BIL cubes are transposed, either into RAM or into a ``.npy`` cache file
that is memory mapped and reused on the next start (``cache='disk'``). An uncompressed
contiguous HDF5 dataset or a BIP cube read along ``lines`` is mapped without any copy.

Recordings larger than the free disk/RAM are streamed (``cache='stream'``): a reader thread
reads the compressed chunks in order and a thread pool inflates them ahead of the camera
(:class:`H5FrameStream`), so memory stays at a few dozen frames. ``cache='auto'`` (default)
reuses an existing cache, else takes disk if it has room, RAM if 8 GB stay free, otherwise
streams - it never fills the disk. Streaming is fine for checking the processing but its own
decoding (~9 ms and a new 2.6 MB buffer per 1024 x 1280 frame) competes with the frame workers:
measure speed with frames in RAM or in the disk cache (see :attr:`FrameSource.storage`).
"""
from __future__ import annotations

import hashlib
import logging
import os
import queue
import re
import shutil
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

H5_SUFFIXES = ('.h5', '.hdf5')
STAGES = ('auto', 'raw', 'binned')
CACHES = ('auto', 'disk', 'ram', 'stream')
_DISK_MARGIN = 2 << 30          # bytes left free on the cache drive
_RAM_RESERVE = 8 << 30          # 'auto' keeps a file in RAM only if this much stays free
_ENVI_DTYPES = {1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32, 5: np.float64,
                12: np.uint16, 13: np.uint32}
_COPY_BLOCK_BYTES = 64 << 20


@dataclass
class FrameSource:
    """One file as a stack of camera frames.

    Attributes:
        path: The data file (HDF5 file or ENVI data file).
        frames: ``(n_frames, slit, spectral)`` C-contiguous array or read-only memory map.
        stage: ``'raw'`` (sensor frames before dark/hot pixel/flip/Smile/binning) or
            ``'binned'`` (already corrected and binned; only classification remains).
        wavelengths: nm of each spectral column for a ``'binned'`` source, else ``None``.
        metadata: HDF5 root attributes or ENVI header fields (plain Python values).
    """
    path: str
    frames: np.ndarray
    stage: str
    wavelengths: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def storage(self) -> str:
        """``'ram'``, ``'disk'`` (memory-mapped cache or file) or ``'stream'`` (decoded while playing)."""
        if isinstance(self.frames, H5FrameStream):
            return 'stream'
        return 'disk' if isinstance(self.frames, np.memmap) else 'ram'

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def frame_shape(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.frames.shape[1:])

    @property
    def dtype(self) -> np.dtype:
        return self.frames.dtype


def open_sources(path, pattern: str | None = None, *, stage: str = 'auto', scan_axis: str = 'samples',
                 dataset: str = 'spectrogram', cache: str = 'auto', cache_dir=None) -> list[FrameSource]:
    """Open a file, or every matching file of a folder, as a playlist of frame sources.

    Args:
        path: An HDF5 file, an ENVI header/data file, or a folder.
        pattern: Glob for a folder (default: all ``*.h5``, ``*.hdf5`` and ``*.hdr``).
        stage: ``'auto'``, ``'raw'`` or ``'binned'`` (see :class:`FrameSource`).
        scan_axis: ENVI axis that is the scan direction: ``'samples'`` (HSI) or ``'lines'``.
        dataset: HDF5 dataset name.
        cache: ``'auto'``, ``'disk'`` (decoded frames in a reusable ``.npy`` memory map), ``'ram'``
            or ``'stream'`` (decoded on the fly ahead of the camera; see the module docstring).
        cache_dir: Folder of the disk cache (default ``<folder of path>/.cache``).

    Returns:
        Sources sorted by file name, all with the same frame shape, dtype and stage.

    Raises:
        FileNotFoundError: nothing matches.
        ValueError: unsupported file, bad option, or files that cannot form one stream.
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    if scan_axis not in ('samples', 'lines'):
        raise ValueError(f"scan_axis must be 'samples' or 'lines', got {scan_axis!r}")
    if cache not in CACHES:
        raise ValueError(f"cache must be one of {CACHES}, got {cache!r}")
    path = Path(path)
    if path.is_dir():
        if pattern:
            files = sorted(path.glob(pattern))
        else:
            files = sorted(p for p in path.iterdir()
                           if p.suffix.lower() in H5_SUFFIXES + ('.hdr',))
        root = path
    elif path.exists():
        files, root = [path], path.parent
    else:
        raise FileNotFoundError(f"Input not found: {path}")
    if not files:
        raise FileNotFoundError(f"No HDF5/ENVI input in {path} (pattern={pattern!r})")
    cache_dir = Path(cache_dir) if cache_dir else root / '.cache'

    sources = []
    for file in files:
        if file.suffix.lower() in H5_SUFFIXES:
            sources.append(open_hdf5(file, dataset=dataset, stage=stage, cache=cache, cache_dir=cache_dir))
        else:
            sources.append(open_envi(file, stage=stage, scan_axis=scan_axis, cache=cache, cache_dir=cache_dir))
    first = sources[0]
    for other in sources[1:]:
        if (other.frame_shape, other.dtype, other.stage) != (first.frame_shape, first.dtype, first.stage):
            raise ValueError(
                f"{other.name} {other.frame_shape} {other.dtype} {other.stage} cannot follow "
                f"{first.name} {first.frame_shape} {first.dtype} {first.stage} in one stream")
        if (first.wavelengths is None) != (other.wavelengths is None) or (
                first.wavelengths is not None and not np.allclose(first.wavelengths, other.wavelengths)):
            raise ValueError(f"{other.name} has other wavelengths than {first.name}")
    return sources


# ----------------------------------------------------------------------------- HDF5 ---

def _plain(value):
    """HDF5 attribute -> plain Python value (str/int/float/bool/list)."""
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def open_hdf5(path, *, dataset: str = 'spectrogram', stage: str = 'auto',
              cache: str = 'auto', cache_dir=None) -> FrameSource:
    """Open an HDF5 spectrogram ``(frame, slit, spectral)`` (see :func:`open_sources`)."""
    import h5py
    path = Path(path)
    with h5py.File(path, 'r') as f:
        if dataset not in f:
            raise ValueError(f"{path.name}: no dataset {dataset!r} (found {list(f.keys())})")
        ds = f[dataset]
        if ds.ndim != 3:
            raise ValueError(f"{path.name}:{dataset} must be (frame, slit, spectral), got {ds.shape}")
        metadata = {k: _plain(v) for k, v in f.attrs.items()}
        metadata.update({f"{dataset}.{k}": _plain(v) for k, v in ds.attrs.items()})
        shape, dtype = tuple(ds.shape), ds.dtype
        offset = ds.id.get_offset() if ds.chunks is None and ds.compression is None else None
        wavelengths = metadata.get('wavelength_nm', metadata.get('wavelengths'))

    if stage == 'auto':
        stage = 'binned' if wavelengths is not None else 'raw'
    wavelengths = np.asarray(wavelengths, np.float32) if stage == 'binned' and wavelengths is not None else None

    if offset is not None and dtype.isnative:
        frames = np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=shape)
        logger.info("%s: %d frames %s %s mapped without copy", path.name, shape[0], shape[1:], dtype)
    else:
        frames = _materialise(path, f"h5:{dataset}", shape, dtype.newbyteorder('='), cache, cache_dir,
                              lambda out: _decode_hdf5(path, dataset, out),
                              stream=lambda: H5FrameStream(path, dataset))
    return FrameSource(str(path), frames, stage, wavelengths, metadata)


def _decode_hdf5(path, dataset, out):
    """Fill ``out`` with every frame; one-frame gzip chunks are inflated on all cores."""
    import h5py
    with h5py.File(path, 'r') as f:
        ds = f[dataset]
        plist = ds.id.get_create_plist()
        filters = [plist.get_filter(i)[0] for i in range(plist.get_nfilters())]
        one_frame_chunks = ds.chunks == (1, *ds.shape[1:])
        if one_frame_chunks and filters == [h5py.h5z.FILTER_DEFLATE]:
            dtype = ds.dtype
            frame_shape = ds.shape[1:]

            def inflate(i, raw):
                out[i] = np.frombuffer(zlib.decompress(raw), dtype=dtype).reshape(frame_shape)

            # HDF5 calls stay on this thread (h5py is not safe to call concurrently);
            # zlib releases the GIL, so only the inflating runs on the pool.
            workers = min(16, os.cpu_count() or 4)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending = []
                for i in range(ds.shape[0]):
                    mask, raw = ds.id.read_direct_chunk((i, 0, 0))
                    if mask:
                        raise ValueError("unexpected HDF5 filter mask")
                    pending.append(pool.submit(inflate, i, raw))
                    if len(pending) >= 4 * workers:
                        for job in pending:
                            job.result()
                        pending.clear()
                for job in pending:
                    job.result()
        else:
            step = max(1, _COPY_BLOCK_BYTES // max(1, out[0].nbytes))
            for i in range(0, ds.shape[0], step):
                out[i:i + step] = ds[i:i + step]


# ----------------------------------------------------------------------------- ENVI ---

def parse_envi_header(path) -> tuple[Path, dict]:
    """Find and parse the ENVI header of ``path`` (header or data file).

    Returns:
        ``(data_file, header)``; header keys are lower case, ``lines``/``samples``/``bands``/
        ``data type``/``byte order``/``header offset`` are ints, ``interleave`` is lower case and
        ``wavelength`` is a float32 array in nm (micrometres are converted) or ``None``.
    """
    path = Path(path)
    if path.suffix.lower() == '.hdr':
        hdr = path
        candidates = [path.with_suffix('')] + [path.with_suffix(s) for s in ('.bip', '.bil', '.bsq', '.img', '.raw', '.dat')]
        data = next((p for p in candidates if p.is_file()), None)
        if data is None:
            raise FileNotFoundError(f"No data file next to {hdr.name}")
    else:
        data = path
        hdr = next((p for p in (Path(str(path) + '.hdr'), path.with_suffix('.hdr')) if p.is_file()), None)
        if hdr is None:
            raise FileNotFoundError(f"No ENVI header for {path.name}")

    text = hdr.read_text(encoding='utf-8', errors='replace')
    if not text.lstrip().upper().startswith('ENVI'):
        raise ValueError(f"{hdr.name} is not an ENVI header")
    header = {}
    for key, value in re.findall(r'^\s*([^=\n]+?)\s*=\s*(\{[^}]*\}|[^\n]*)', text, re.MULTILINE):
        header[key.strip().lower()] = value.strip()
    for key, default in (('lines', None), ('samples', None), ('bands', None), ('data type', 4),
                         ('byte order', 0), ('header offset', 0)):
        if key not in header and default is None:
            raise ValueError(f"{hdr.name}: missing '{key}'")
        header[key] = int(header.get(key, default))
    header['interleave'] = header.get('interleave', 'bsq').lower()

    wavelengths = None
    if 'wavelength' in header:
        values = [v.strip() for v in header['wavelength'].strip('{}').split(',') if v.strip()]
        wavelengths = np.array([float(v) for v in values], dtype=np.float32)
        if wavelengths.size != header['bands']:
            raise ValueError(f"{hdr.name}: {wavelengths.size} wavelengths for {header['bands']} bands")
        units = header.get('wavelength units', 'nanometers').lower()
        if units.startswith('micro') or units in ('um', 'µm'):
            wavelengths = wavelengths * np.float32(1000.0)
    header['wavelength'] = wavelengths
    return data, header


def open_envi(path, *, stage: str = 'auto', scan_axis: str = 'samples',
              cache: str = 'auto', cache_dir=None) -> FrameSource:
    """Open an ENVI BSQ/BIL/BIP cube as frames ``(slit, bands)`` (see :func:`open_sources`)."""
    data_path, header = parse_envi_header(path)
    L, S, B = header['lines'], header['samples'], header['bands']
    code = header['data type']
    if code not in _ENVI_DTYPES:
        raise ValueError(f"{data_path.name}: unsupported ENVI data type {code}")
    dtype = np.dtype(_ENVI_DTYPES[code]).newbyteorder('>' if header['byte order'] == 1 else '<')
    interleave = header['interleave']
    file_axes = {'bsq': ('b', 'l', 's'), 'bil': ('l', 'b', 's'), 'bip': ('l', 's', 'b')}
    if interleave not in file_axes:
        raise ValueError(f"{data_path.name}: unsupported interleave {interleave!r}")
    sizes = {'l': L, 's': S, 'b': B}
    axes = file_axes[interleave]
    offset = header['header offset']
    needed = offset + L * S * B * dtype.itemsize
    if data_path.stat().st_size < needed:
        raise ValueError(f"{data_path.name}: {data_path.stat().st_size} bytes, header describes {needed}")
    raw = np.memmap(data_path, dtype=dtype, mode='r', offset=offset, shape=tuple(sizes[a] for a in axes))

    scan, slit = ('s', 'l') if scan_axis == 'samples' else ('l', 's')
    order = tuple(axes.index(a) for a in (scan, slit, 'b'))           # -> (frame, slit, bands)
    view = raw.transpose(order)

    wavelengths = header['wavelength']
    if stage == 'auto':
        stage = 'binned' if wavelengths is not None else 'raw'
    metadata = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in header.items()}
    metadata.update(envi_scan_axis=scan_axis, envi_interleave=interleave)

    if view.flags.c_contiguous and dtype.isnative:
        frames = view
        logger.info("%s: %d frames %s %s mapped without copy (%s)", data_path.name, view.shape[0],
                    view.shape[1:], dtype, interleave.upper())
    else:
        def fill(out):
            step = max(1, _COPY_BLOCK_BYTES // max(1, view[0].nbytes))
            for i in range(0, view.shape[0], step):
                out[i:i + step] = view[i:i + step]
        frames = _materialise(data_path, f"envi:{interleave}:{scan_axis}", view.shape,
                              dtype.newbyteorder('='), cache, cache_dir, fill,
                              stream=(lambda: view) if dtype.isnative else None)
    return FrameSource(str(data_path), frames, stage,
                       wavelengths if stage == 'binned' else None, metadata)


# --------------------------------------------------------------------------- caching ---

def _available_ram() -> int | None:
    """Free physical memory in bytes (``None`` if unknown)."""
    try:
        if os.name == 'nt':
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                            ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                            ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                            ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                            ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
            status = _Status()
            status.dwLength = ctypes.sizeof(_Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return int(status.ullAvailPhys)
        return int(os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE'))
    except Exception:
        return None


def _free_disk(folder: Path) -> int:
    folder = Path(folder)
    while not folder.exists() and folder.parent != folder:
        folder = folder.parent
    return shutil.disk_usage(folder).free


def _materialise(path: Path, kind: str, shape, dtype, cache, cache_dir, fill, stream=None):
    """Frame-major frames of a file: RAM copy, reusable ``.npy`` memory map, or a stream.

    ``cache='auto'`` reuses an existing cache file, else picks disk (enough room left),
    RAM (file at most a quarter of the free memory) or streaming. An explicit ``'disk'`` or
    ``'ram'`` without room raises instead of filling the drive or swapping.
    """
    shape = tuple(int(v) for v in shape)
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    st = path.stat()
    cache_dir = Path(cache_dir or path.parent / '.cache')
    key = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{kind}|{shape}|{np.dtype(dtype).str}"
                       .encode()).hexdigest()[:16]
    target = cache_dir / f"{path.stem}.{key}.npy"
    gb = nbytes / 2**30
    if cache == 'auto':
        free_disk, free_ram = _free_disk(cache_dir), _available_ram()
        if target.is_file():
            cache = 'disk'
        elif free_disk >= nbytes + _DISK_MARGIN:
            cache = 'disk'
        elif free_ram is not None and nbytes <= free_ram - _RAM_RESERVE:
            cache = 'ram'
        else:
            cache = 'stream' if stream is not None else 'ram'
        logger.info("%s: %.1f GB decoded; free disk %.1f GB, free RAM %s -> cache=%s", path.name, gb,
                    free_disk / 2**30, 'unknown' if free_ram is None else f"{free_ram / 2**30:.1f} GB", cache)
    if cache == 'stream':
        if stream is None:
            raise ValueError(f"{path.name}: streaming is not supported for this file; use cache 'ram' or 'disk'")
        return stream()
    if cache == 'disk' and not target.is_file() and _free_disk(cache_dir) < nbytes + _DISK_MARGIN:
        raise OSError(f"{path.name}: the cache needs {gb:.1f} GB but {cache_dir} has "
                      f"{_free_disk(cache_dir) / 2**30:.1f} GB free; use cache 'auto' or 'stream'")
    if cache == 'ram':
        free_ram = _available_ram()
        if free_ram is not None and nbytes > 0.8 * free_ram:
            raise MemoryError(f"{path.name}: {gb:.1f} GB does not fit the free RAM "
                              f"({free_ram / 2**30:.1f} GB); use cache 'auto' or 'stream'")
        out = np.empty(shape, dtype)
        fill(out)
        logger.info("%s: %d frames %s %s decoded into RAM (%.0f MB)", path.name, shape[0], shape[1:],
                    dtype, nbytes / 1e6)
        return out

    if target.is_file():
        try:
            frames = np.load(target, mmap_mode='r')
            if frames.shape == shape and frames.dtype == dtype:
                logger.info("%s: %d frames from cache %s", path.name, shape[0], target.name)
                return frames
        except (OSError, ValueError) as exc:
            logger.warning("Ignoring unreadable cache %s: %s", target, exc)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for old in cache_dir.glob(f"{path.stem}.*.npy"):        # earlier versions of this file
        try:
            old.unlink()
        except OSError:
            pass
    partial = target.with_suffix('.partial')
    out = np.lib.format.open_memmap(partial, mode='w+', dtype=dtype, shape=shape)
    try:
        fill(out)
        out.flush()
    finally:
        del out
    os.replace(partial, target)
    logger.info("%s: %d frames %s %s decoded into cache %s (%.0f MB)", path.name, shape[0], shape[1:],
                dtype, target.name, nbytes / 1e6)
    return np.load(target, mmap_mode='r')


# -------------------------------------------------------------------------- streaming ---

def _lower_thread_priority() -> None:
    """Below-normal OS priority for the calling thread (Windows; elsewhere a no-op)."""
    if os.name == 'nt':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetThreadPriority(kernel32.GetCurrentThread(), -1)     # THREAD_PRIORITY_BELOW_NORMAL
        except Exception:
            pass


class H5FrameStream:
    """Frames of a gzip HDF5 spectrogram, inflated ahead of the reader (``cache='stream'``).

    Behaves like a read-only ``(frames, slit, spectral)`` array for what the simulator needs:
    ``shape``, ``dtype``, ``ndim``, ``len()``, ``stream[i]`` and ``stream[a:b]``. Sequential
    access (the camera) is served from a bounded prefetch queue: one thread reads the compressed
    chunks in order and a pool inflates them; after the last frame it continues at frame 0, so a
    looping playlist never waits. A short skip ahead (lines lost to overrun) drops queued frames;
    any other index restarts the prefetch there, after the previous reader thread has ended -
    HDF5 is never called from two threads at once. Memory: ``depth`` decoded frames.
    """

    def __init__(self, path, dataset: str = 'spectrogram', depth: int = 32, workers: int | None = None):
        import h5py
        self.path, self.dataset = str(path), dataset
        with h5py.File(self.path, 'r') as f:
            ds = f[dataset]
            self.shape = tuple(int(v) for v in ds.shape)
            self.dtype = ds.dtype.newbyteorder('=')
            plist = ds.id.get_create_plist()
            filters = [plist.get_filter(i)[0] for i in range(plist.get_nfilters())]
            self._direct = ds.chunks == (1, *ds.shape[1:]) and filters == [h5py.h5z.FILTER_DEFLATE]
        self.ndim = 3
        self._depth = depth
        # Decoding is the simulator's own cost (a camera delivers frames uncompressed), so it runs
        # on a few low-priority threads: ~110 frames/s each for 1024 x 1280, and the frame
        # workers win the CPU whenever both want it.
        self._pool = ThreadPoolExecutor(max_workers=workers or min(4, os.cpu_count() or 4),
                                        thread_name_prefix='inflate', initializer=_lower_thread_priority)
        self._lock = threading.Lock()
        self._queue: queue.Queue | None = None
        self._stop: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._next = None
        logger.info("%s: %d frames %s %s streamed (%s, %d frames ahead)", Path(self.path).name, self.shape[0],
                    self.shape[1:], self.dtype, 'parallel inflate' if self._direct else 'sequential read', depth)

    def __len__(self) -> int:
        return self.shape[0]

    def _decode(self, raw) -> np.ndarray:
        return np.frombuffer(zlib.decompress(raw), dtype=self.dtype).reshape(self.shape[1:])

    def _reader(self, start: int, out: queue.Queue, stop: threading.Event) -> None:
        import h5py
        _lower_thread_priority()
        n = self.shape[0]
        try:
            with h5py.File(self.path, 'r') as f:
                ds = f[self.dataset]
                i = start
                while not stop.is_set():
                    if self._direct:
                        mask, raw = ds.id.read_direct_chunk((i, 0, 0))
                        item = self._pool.submit(self._decode, raw)
                    else:
                        item = ds[i]
                    while not stop.is_set():
                        try:
                            out.put((i, item), timeout=0.1)
                            break
                        except queue.Full:
                            continue
                    i = (i + 1) % n
        except Exception as exc:                          # surfaced to the consumer
            out.put((None, exc))

    def _end_reader(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        while self._thread.is_alive():          # it may be blocked on a full queue: drain it
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._thread.join(0.05)
        self._thread = None

    def _restart(self, start: int) -> None:
        self._end_reader()                      # only one thread may call HDF5
        self._stop = threading.Event()
        self._queue = queue.Queue(maxsize=self._depth)
        self._thread = threading.Thread(target=self._reader, args=(start, self._queue, self._stop),
                                        name=f"read-{Path(self.path).stem}", daemon=True)
        self._thread.start()
        self._next = start

    def _frame(self, i: int) -> np.ndarray:
        n = self.shape[0]
        if not -n <= i < n:
            raise IndexError(i)
        i %= n
        with self._lock:
            ahead = None if self._next is None else (i - self._next) % n
            if ahead is None or ahead > self._depth:
                self._restart(i)
            while True:                          # a short skip ahead: drop the frames in between
                index, item = self._queue.get()
                if index is None:
                    raise item
                if index == i:
                    break
            self._next = (i + 1) % n
        return item.result() if hasattr(item, 'result') else np.asarray(item)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self._frame(int(key))
        if isinstance(key, slice):
            indices = range(*key.indices(self.shape[0]))
            out = np.empty((len(indices), *self.shape[1:]), self.dtype)
            for j, i in enumerate(indices):
                out[j] = self._frame(i)
            return out
        raise TypeError("H5FrameStream supports int and slice indexing")

    def __array__(self, dtype=None):
        array = self[:]
        return array if dtype is None else array.astype(dtype)

    def close(self) -> None:
        with self._lock:
            self._end_reader()
        self._pool.shutdown(wait=False)
