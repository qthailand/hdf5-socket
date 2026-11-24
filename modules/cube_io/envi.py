"""ENVI header parsing and band reads extracted without changing layout semantics.

Data conventions:
    ENVI ``lines``/``samples``/``bands`` map to ``(slit, scan, wavelength)`` for
    cubes written by this package.  Band reads return float32 ``(lines, samples)``.
    The real-valued data types 1, 2, 3, 4, 5, 12, 13, 14 and 15 (:data:`ENVI_DATA_TYPES`)
    are decoded; complex (6, 9) and unknown codes raise ``ValueError``.  Band reads
    honour the header ``byte order`` and ``header offset`` and read BSQ, BIL and BIP
    files.  Wavelengths are returned in nm (micrometre headers are converted).

File handles:
    Band reads keep the memory maps of the last two data files open (re-mapping a
    BIP file for every band is about 40x slower), and :func:`read_envi_cube_bsq` keeps
    in-RAM BSQ copies of recently read cubes.  Call :func:`release_band_maps`
    before overwriting or deleting such a file: on Windows a mapped file cannot be
    replaced.  A file that changed on disk (size or modification time) is re-mapped.

Threading:
    Any thread; the map cache is protected by a lock.

Example:
    >>> import os, tempfile
    >>> from modules.cube_io.envi import parse_envi_header
    >>> d = tempfile.mkdtemp(); p = os.path.join(d, "x.img")
    >>> text = os.linesep.join(["ENVI", "lines = 2", "samples = 3", "bands = 4"])
    >>> open(p + ".hdr", "w").write(text) > 0
    True
    >>> m = parse_envi_header(p); (m["lines"], m["samples"], m["bands"], m["wavelength"])
    (2, 3, 4, None)
"""
import os
import re
import threading
from collections import OrderedDict

import numpy as np

#: ENVI ``data type`` code -> numpy dtype (byte order applied separately).
ENVI_DATA_TYPES = {
    1: np.dtype(np.uint8), 2: np.dtype(np.int16), 3: np.dtype(np.int32),
    4: np.dtype(np.float32), 5: np.dtype(np.float64), 12: np.dtype(np.uint16),
    13: np.dtype(np.uint32), 14: np.dtype(np.int64), 15: np.dtype(np.uint64),
}

#: ``wavelength units`` values (lower case) that mean micrometres.
_MICROMETRE_UNITS = ('micrometers', 'micrometres', 'micrometer', 'micrometre', 'microns', 'micron',
                     'um', 'µm', 'μm')


def envi_dtype(code):
    """The numpy dtype of an ENVI ``data type`` code.

    Raises:
        ValueError: complex (6, 9) or unknown codes.
    """
    try:
        return ENVI_DATA_TYPES[int(code)]
    except (KeyError, ValueError, TypeError):
        raise ValueError(f"Unsupported ENVI data type {code!r} (supported: "
                         f"{', '.join(str(c) for c in sorted(ENVI_DATA_TYPES))})") from None


def envi_data_type(dtype):
    """The ENVI ``(data type, byte order)`` of a numpy dtype (byte order 1 = big endian).

    Raises:
        ValueError: the dtype has no ENVI code (bool, int8, float16, complex, ...).
    """
    dtype = np.dtype(dtype)
    for code, known in ENVI_DATA_TYPES.items():
        if dtype.newbyteorder('=') == known:
            big = dtype.byteorder == '>' or (dtype.byteorder == '=' and not np.little_endian)
            return code, int(big)
    raise ValueError(f"dtype {dtype} cannot be written to ENVI (supported: "
                     f"{', '.join(str(d) for d in ENVI_DATA_TYPES.values())})")


_MAX_MAPS = 2
_maps = OrderedDict()            # (path, mtime, size, dtype, offset, shape) -> np.memmap
_maps_lock = threading.Lock()


def _band_map(filepath, dtype, offset, shape):
    """Read-only memory map of a data file, reused while the file is unchanged."""
    path = os.path.abspath(filepath)
    st = os.stat(path)
    key = (path, st.st_mtime_ns, st.st_size, dtype.str, offset, shape)
    with _maps_lock:
        data = _maps.get(key)
        if data is None:
            for old in [k for k in _maps if k[0] == path]:       # the file was rewritten
                del _maps[old]
            data = np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=shape)
            _maps[key] = data
            while len(_maps) > _MAX_MAPS:
                _maps.popitem(last=False)
        else:
            _maps.move_to_end(key)
        return data


def release_band_maps(filepath=None):
    """Close the cached memory maps of ``filepath`` (or of every file).

    Call it before overwriting or deleting a file that was read with
    :func:`_read_envi_band_float32`.
    """
    with _maps_lock:
        if filepath is None:
            _maps.clear()
            _cubes.clear()
        else:
            path = os.path.abspath(filepath)
            for cache in (_maps, _cubes):
                for old in [k for k in cache if k[0] == path]:
                    del cache[old]


#: Total size of the in-RAM BSQ copies kept by :func:`read_envi_cube_bsq` (file dtype).
CUBE_CACHE_BYTES = 4 << 30
_cubes = OrderedDict()           # (path, mtime, size, dtype, offset, shape, interleave) -> array


def read_envi_cube_bsq(filepath, meta, max_bytes=None):
    """A BIL/BIP/BSQ cube as a read-only in-RAM ``(bands, lines, samples)`` array.

    The array keeps the file's data type in native byte order, so ``cube[b].astype(np.float32)``
    equals :func:`_read_envi_band_float32` of band ``b``.  Converting an interleaved file costs
    one pass over it; the copy is then kept (largest :data:`CUBE_CACHE_BYTES` in total, least
    recently used dropped first, re-read when the file changes) so the next reads of the same
    cube are as fast as BSQ bands.  :func:`release_band_maps` drops it.

    Args:
        max_bytes: Largest cube to load; default :data:`CUBE_CACHE_BYTES`.  A cube that is
            not cached is also refused when it needs more than half of the free memory.

    Returns:
        numpy.ndarray or None: ``None`` when the cube is too large (read bands in blocks
        with :func:`read_envi_bands_float32` instead).
    """
    data, interleave = _envi_map(filepath, meta)
    path = os.path.abspath(filepath)
    st = os.stat(path)
    key = (path, st.st_mtime_ns, st.st_size, data.dtype.str, int(meta.get('header offset', 0)),
           data.shape, interleave)
    with _maps_lock:
        cube = _cubes.get(key)
        if cube is not None:
            _cubes.move_to_end(key)
            return cube
    limit = CUBE_CACHE_BYTES if max_bytes is None else int(max_bytes)
    if data.nbytes > limit or data.nbytes > _free_memory() // 2:
        return None
    n_bands, n_lines, n_samples = int(meta['bands']), int(meta['lines']), int(meta['samples'])
    cube = np.empty((n_bands, n_lines, n_samples), dtype=data.dtype.newbyteorder('='))
    _copy_bsq(data, interleave, 0, n_bands, cube)
    cube.flags.writeable = False
    with _maps_lock:
        for old in [k for k in _cubes if k[0] == path]:          # the file was rewritten
            del _cubes[old]
        _cubes[key] = cube
        while sum(c.nbytes for c in _cubes.values()) > CUBE_CACHE_BYTES and len(_cubes) > 1:
            _cubes.popitem(last=False)
    return cube


def _free_memory():
    """Available physical memory in bytes (unlimited if psutil is not installed)."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return 1 << 62


def parse_envi_header(filepath):
    """Parse the ENVI header that belongs to a data file.

    Args:
        filepath: Path of the data file; the header is looked up as
            ``filepath + '.hdr'`` and then as ``<filepath without extension>.hdr``.

    Returns:
        dict: Header fields with lower-cased keys and string values, plus
        defaults ``data_type=4``, ``byte_order=0``, ``interleave='bsq'``,
        ``description``; ``lines``/``samples``/``bands`` as ``int``;
        ``wavelength`` as a float32 array in nm or ``None``.  A ``wavelength units``
        of micrometres is converted (values x 1000, ``wavelength units`` becomes
        ``Nanometers``); other units are returned as written.

    Raises:
        ValueError: no header file found, or a wavelength entry is not a number.
        KeyError: the header lacks ``lines``, ``samples`` or ``bands``.
    """
    hdr_path = filepath + '.hdr'
    if not os.path.exists(hdr_path):
        hdr_path = os.path.splitext(filepath)[0] + '.hdr'
    if not os.path.exists(hdr_path):
        raise ValueError(f"Header file not found for {filepath}")

    meta = {'data_type': 4, 'byte_order': 0,
            'interleave': 'bsq', 'description': 'Processed Data'}
    with open(hdr_path, 'rb') as f:
        raw = f.read()
    try:
        txt = raw.decode('utf-8')
    except UnicodeDecodeError:
        txt = raw.decode('latin-1')          # Windows tools write e.g. 'µm' as cp1252
    txt = txt.replace('\r\n', '\n').replace('\r', '\n')

    m_desc = re.search(r'description\s*=\s*\{([^}]*)\}', txt, re.IGNORECASE | re.DOTALL)
    if m_desc:
        meta['description'] = m_desc.group(1).strip().replace('\n', ' ')
    else:
        m_desc_simple = re.search(
            r'^description\s*=\s*(.*)$', txt, re.IGNORECASE | re.MULTILINE)
        if m_desc_simple:
            meta['description'] = m_desc_simple.group(1).strip()

    for line in txt.split('\n'):
        if '=' in line and 'description' not in line.lower():
            k, v = line.split('=', 1)
            k = k.strip().lower()
            if k != 'wavelength':
                meta[k] = v.strip()

    m_wl = re.search(r'wavelength\s*=\s*\{([^}]*)\}', txt, re.IGNORECASE | re.DOTALL)
    if m_wl:
        vals = m_wl.group(1).replace('\n', ' ').split(',')
        meta['wavelength'] = np.array(
            [float(v.strip()) for v in vals if v.strip() != ''], dtype=np.float32)
        if str(meta.get('wavelength units', '')).strip().lower() in _MICROMETRE_UNITS:
            meta['wavelength'] = meta['wavelength'] * np.float32(1000.0)
            meta['wavelength units'] = 'Nanometers'
    else:
        meta['wavelength'] = None

    meta['lines'] = int(meta['lines'])
    meta['samples'] = int(meta['samples'])
    meta['bands'] = int(meta['bands'])
    return meta


def _read_envi_band_float32(filepath, meta, band_idx):
    """Read one band of an ENVI data file as float32 (public alias ``read_envi_band``).

    Args:
        filepath: Path of the data file (without header).
        meta: Dict from :func:`parse_envi_header` (uses ``lines``, ``samples``,
            ``bands``, ``data type``/``data_type`` and ``interleave``).
        band_idx: Zero-based band (wavelength) index.

    Returns:
        numpy.ndarray: New float32 array ``(lines, samples)``.  The file is memory
        mapped, so only the requested band is read (BSQ, BIL and BIP alike); the map
        stays cached (see :func:`release_band_maps`).  ``byte order`` (1 = big endian) and
        ``header offset`` are honoured.  32/64-bit integers beyond 2**24 lose
        precision in the float32 result.

    Raises:
        ValueError: unsupported data type or interleave, or the file is smaller than
            the header describes.
    """
    data, interleave = _envi_map(filepath, meta)
    if interleave == 'bsq':
        band = data[band_idx]
    elif interleave == 'bil':
        band = data[:, band_idx, :]
    else:
        band = data[:, :, band_idx]
    return np.array(band, dtype=np.float32)        # a copy, never a view of the map


def read_envi_bands_float32(filepath, meta, start, stop):
    """Read bands ``start .. stop-1`` at once as float32 ``(stop - start, lines, samples)``.

    Each band equals :func:`_read_envi_band_float32` of it.  For BIL and BIP files one
    block read passes over the file once instead of once per band, which is what makes
    band-by-band processing of interleaved cubes slow.

    Raises:
        ValueError: as :func:`_read_envi_band_float32`, or an empty / out-of-range block.
    """
    data, interleave = _envi_map(filepath, meta)
    start, stop = int(start), int(stop)
    if not 0 <= start < stop <= int(meta['bands']):
        raise ValueError(f"Band block {start}:{stop} outside 0:{meta['bands']}")
    out = np.empty((stop - start, int(meta['lines']), int(meta['samples'])), dtype=np.float32)
    _copy_bsq(data, interleave, start, stop, out)
    return out


def _copy_bsq(data, interleave, start, stop, out):
    """Copy bands ``start:stop`` of a mapped file into ``out`` ``(stop - start, lines, samples)``."""
    if interleave == 'bsq':
        out[...] = data[start:stop]
        return
    # a few lines at a time: transposing them in cache is ~3x faster than the whole block
    step = max(1, (1 << 20) // max(1, data[0].nbytes))
    for y in range(0, out.shape[1], step):
        lines = data[y:y + step]
        if interleave == 'bil':
            out[:, y:y + step] = lines[:, start:stop, :].transpose(1, 0, 2)
        else:
            out[:, y:y + step] = lines[:, :, start:stop].transpose(2, 0, 1)


def _envi_map(filepath, meta):
    """``(memmap, interleave)`` of an ENVI data file, shaped by its interleave."""
    L, S, B = meta['lines'], meta['samples'], meta['bands']
    dt = envi_dtype(meta.get('data type', meta.get('data_type', '4')))
    bpp = dt.itemsize
    dt = dt.newbyteorder('>' if int(meta.get('byte order', meta.get('byte_order', 0))) == 1 else '<')
    offset = int(meta.get('header offset', 0))

    interleave = str(meta.get('interleave', 'bsq')).lower().strip()
    shapes = {'bsq': (B, L, S), 'bil': (L, B, S), 'bip': (L, S, B)}
    if interleave not in shapes:
        raise ValueError(f"Unsupported interleave: {interleave}")
    expected = offset + L * S * B * bpp
    size = os.path.getsize(filepath)
    if size < expected:
        raise ValueError(f"{filepath}: file has {size} bytes, the header describes {expected}")

    return _band_map(filepath, dt, offset, shapes[interleave]), interleave


