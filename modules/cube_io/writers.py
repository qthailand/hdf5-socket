"""Qt-independent export services preserving existing file formats and metadata.

Data conventions:
    * ``HypercubeWriter`` / ``GeoTIFFWriter``: ``hypercube`` is band-first
      ``(wavelength, slit, scan)`` of any ENVI data type (uint8, int16, uint16,
      int32, uint32, int64, uint64, float32, float64); ``linear_spectrum`` is the 1-D
      band wavelengths in nm.  The ENVI output is BIP (``lines = slit``,
      ``samples = scan``, ``bands = wavelength``) whose ``data type`` and
      ``byte order`` follow the array's dtype; the GeoTIFF keeps the dtype too.
    * ``SpectrogramWriter``: ``spectrogram`` is ``(frames, height, width)``.  The
      binary file is a 16-byte little-endian header ``<fffHH`` =
      ``(m * camera_bin, c, pixel_size, height, width)`` followed by the raw
      frames (C order, array dtype, not recorded in the header).  There is no
      ``format_version`` field.
    * ``config`` is a dict with calibration ``m`` (nm/pixel at bin 1) and ``c`` (nm),
      required by ``SpectrogramWriter`` only; ``camera_bin`` is the hardware binning factor.

Threading:
    Each writer's ``run()`` is blocking and meant for a worker thread; ``progress``
    is called on that thread with an int percentage.  ``SpectrogramWriter.cancel()``
    is safe to call from another thread (sets a flag checked once per frame).

Example:
    >>> import numpy as np, os, tempfile
    >>> from modules.cube_io.writers import HypercubeWriter
    >>> from modules.cube_io.envi import parse_envi_header
    >>> path = os.path.join(tempfile.mkdtemp(), "cube.img")
    >>> cube = np.zeros((4, 2, 3), np.uint16)                 # (wavelength, slit, scan)
    >>> seconds = HypercubeWriter(save_path=path, config={},
    ...                           hypercube=cube,
    ...                           linear_spectrum=np.linspace(400, 410, 4)).run()
    >>> meta = parse_envi_header(path)
    >>> meta["lines"], meta["samples"], meta["bands"]
    (2, 3, 4)
"""
import os
import struct
import time
import traceback
from datetime import datetime, timezone
import numpy as np

from modules.cube_io.envi import envi_data_type

#: Scan temperatures (°C) written to the ENVI header when present in ``config``
#: (``ScanController.last_scan_temperatures``): ``(config key, header field)``.
TEMPERATURE_HEADER_FIELDS = (
    ('sensor_temperature_c_start', 'sensor temperature start'),
    ('sensor_temperature_c_end', 'sensor temperature end'),
    ('sensor_temperature_c_min', 'sensor temperature min'),
    ('sensor_temperature_c_max', 'sensor temperature max'),
    ('dark_temperature_c', 'dark temperature'),
    ('target_temperature_c', 'target temperature'),
)

#: Fields describing the raw counts (``camera_bit_info``); dropped once a cube is calibrated.
#: ``(config key, header field, kind)``.
COUNT_HEADER_FIELDS = (
    ('bit_depth', 'data bit depth', int),
    ('sensor_bit_depth', 'sensor bit depth', int),
    ('full_scale', 'data full scale', int),
)

#: Scan settings and the corrections the frames went through (``ScanController.last_scan_header``),
#: written when present in ``config``: ``(config key, header field, kind)``.  ``bool`` fields are
#: written ``1``/``0``; a missing key means unknown (e.g. a cube opened from a file without it).
SCAN_HEADER_FIELDS = (
    ('camera_model', 'camera model', str),
    ('camera_serial', 'camera serial number', str),
    ('exposure_ms', 'exposure time (ms)', float),
    ('gain', 'sensor gain', float),
    ('frame_rate_hz', 'frame rate (Hz)', float),
    ('camera_bin', 'camera bin', int),
    ('dark_corrected', 'dark corrected', bool),
    ('hot_pixel_corrected', 'hot pixel corrected', bool),
    ('hot_pixel_count', 'hot pixel count', int),
    ('dark_file', 'dark reference', str),          # its temperature: 'dark temperature' above
    ('dark_frames', 'dark reference frames', int),
    ('smile_corrected', 'smile corrected', bool),
    ('smile_model', 'smile model', str),
    ('smile_model_created', 'smile model created', str),
    ('wavelength_source', 'wavelength calibration source', str),
    ('wavelength_profile', 'wavelength calibration', str),
    ('wavelength_profile_saved', 'wavelength calibration saved', str),
    ('wavelength_rms_nm', 'wavelength calibration rms (nm)', float),
    ('wavelength_m_nm_per_px', 'wavelength slope (nm/px, bin 1)', float),
    ('wavelength_c_nm', 'wavelength offset (nm)', float),
    ('reflectance_reference', 'reflectance reference', str),
    ('reflectance_reference_cube', 'reflectance reference cube', str),
    ('reflectance_reference_rows', 'reflectance reference rows', str),
    ('calibration_warnings', 'calibration warnings', str),
    ('stage_type', 'stage type', str),
    ('speed_um_s', 'linear speed (um/s)', float),
    ('speed_um_s_achieved', 'linear speed achieved (um/s)', float),
    ('degree_per_second', 'rotation speed (deg/s)', float),
    ('degree_per_second_achieved', 'rotation speed achieved (deg/s)', float),
)


def _header_fields(counts=True):
    """Every metadata field in header order: ``(config key, header field, kind)``."""
    return ((COUNT_HEADER_FIELDS if counts else ())
            + tuple((key, name, 'temperature') for key, name in TEMPERATURE_HEADER_FIELDS)
            + SCAN_HEADER_FIELDS)


def _format_header_value(value, kind):
    if kind == 'temperature':
        return f'{float(value):.2f}'
    if kind is bool:
        return '1' if value else '0'
    if kind is int:
        return str(int(value))
    if kind is float:
        return f'{float(value):.10g}'
    text = str(value).replace('\n', ' ').replace('{', '(').replace('}', ')')
    # mismatch texts use these; plain ASCII keeps every header readable by older tools
    return text.replace('≠', '!=').replace('×', 'x').replace('°', ' deg ').replace('·', '-')


def metadata_header_lines(config, counts=True):
    """``'field = value'`` header lines of the metadata in ``config`` (see :data:`SCAN_HEADER_FIELDS`).

    Args:
        config: Dict with any of the keys of :data:`COUNT_HEADER_FIELDS`,
            :data:`TEMPERATURE_HEADER_FIELDS` and :data:`SCAN_HEADER_FIELDS`; ``None`` values
            and values that do not convert are left out.
        counts: Include the raw-count fields (bit depths, full scale); ``False`` for a
            calibrated (reflectance) cube.
    """
    lines = []
    for key, name, kind in _header_fields(counts):
        value = config.get(key)
        if value is None:
            continue
        try:
            lines.append(f'{name} = {_format_header_value(value, kind)}')
        except (TypeError, ValueError):
            continue
    return lines


def metadata_from_header(meta):
    """The inverse of :func:`metadata_header_lines`: ``config`` keys from a parsed ENVI header.

    Args:
        meta: Dict from :func:`modules.cube_io.envi.parse_envi_header` (lower-case keys).

    Returns:
        dict: The metadata keys present in the header (typed), plus ``started_at`` from
        ``acquisition time``; unparsable values are left out.
    """
    out = {}
    for key, name, kind in _header_fields():
        raw = meta.get(name.lower())
        if raw is None:
            continue
        text = str(raw).strip()
        try:
            if kind is bool:
                out[key] = text.lower() in ('1', 'true', 'yes')
            elif kind is int:
                out[key] = int(float(text))
            elif kind in (float, 'temperature'):
                out[key] = float(text)
            else:
                out[key] = text
        except ValueError:
            continue
    if meta.get('acquisition time'):
        out['started_at'] = str(meta['acquisition time']).strip()
    return out


def carried_header_lines(meta, calibrated, extra=None):
    """Header lines that a cube derived from the file of ``meta`` keeps (acquisition time,
    temperatures, scan settings, corrections; raw-count fields only when not ``calibrated``).

    ``extra``: metadata of the derived cube itself (e.g. ``reflectance_reference``) written
    with it; its ``calibration_warnings`` are appended to the carried ones (each once).  When it
    names a reflectance reference, the source's ``reflectance_reference*`` fields are dropped
    first (a calibrated cube calibrated again must not keep the old reference cube / rows).
    """
    config = metadata_from_header(meta)
    extra = {k: v for k, v in (extra or {}).items() if v is not None}
    if any(k.startswith('reflectance_reference') for k in extra):
        for key in [k for k in config if k.startswith('reflectance_reference')]:
            del config[key]
    for key, value in extra.items():
        if key == 'calibration_warnings' and config.get(key):
            carried = str(config[key]).split('; ')
            value = '; '.join(carried + [w for w in str(value).split('; ') if w not in carried])
        config[key] = value
    lines = [f"acquisition time = {config['started_at']}"] if 'started_at' in config else []
    return lines + metadata_header_lines(config, counts=not calibrated)


def acquisition_time_utc(config):
    """The ENVI ``acquisition time`` (ISO 8601, UTC, ``Z``): the scan's ``started_at``
    from ``config`` when present (an offset-less time is taken as local), else now."""
    when = None
    started = config.get('started_at')
    if started:
        try:
            text = str(started).strip()
            if text.endswith('Z'):                  # Python 3.10 fromisoformat has no 'Z'
                text = text[:-1] + '+00:00'
            when = started if isinstance(started, datetime) else datetime.fromisoformat(text)
        except ValueError:
            when = None
    if when is None:
        when = datetime.now(timezone.utc)
    elif when.tzinfo is None:
        when = when.astimezone()
    return when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class GeoTIFFWriter:
    """Write a band-first cube as a tiled, LZW-compressed multi-band GeoTIFF.

    The file carries no georeference (the pushbroom scan has none); the cube's dtype is
    kept.  Requires ``rasterio`` (imported when constructing).
    """

    def __init__(self, *, save_path: str, config: dict, hypercube: np.ndarray, linear_spectrum: np.ndarray, camera_bin: int = 1, progress=None):
        """Prepare the GeoTIFF profile (no file is written yet).

        Args:
            save_path: Output ``.tif`` path.
            config: Export settings (unused; accepted for API symmetry).
            hypercube: Band-first ``(wavelength, slit, scan)`` array; kept
                by reference, not copied.  Written in its own dtype (native byte order).
            linear_spectrum: 1-D band wavelengths (nm), one per band.
            camera_bin: Camera hardware bin (``0``/``None`` mean 1).
            progress: Optional ``callable(int percent)``.

        Raises:
            ValueError: ``hypercube`` is not 3-D (tuple unpacking of its shape).
            ImportError: ``rasterio`` missing.
        """
        self.progress = progress or (lambda value: None)
        import rasterio  # noqa: F401  (fail early when missing)
        self.save_path = save_path
        self.config = config
        self.camera_bin = int(camera_bin or 1)
        self.hypercube = hypercube
        self.linear_spectrum = linear_spectrum
        self.count, self.hight, self.width = self.hypercube.shape
        self.dtype = self.hypercube.dtype.newbyteorder('=')
        self.geo_meta = {
            'driver': 'GTiff',
            'width': self.width,
            'height': self.hight,
            'count': self.count,
            'dtype': self.dtype.name,
            'tiled': True,
            'compress': 'LZW',
            'predictor': 3 if self.dtype.kind == 'f' else 2,   # floating-point / horizontal differencing
        }


    def run(self):
        """Write the GeoTIFF band by band, with per-band wavelength tags.

        Returns:
            float: Elapsed seconds (rounded to 0.01).

        Raises:
            ValueError: ``hypercube`` is ``None`` or has no shape.
            Exception: rasterio/IO errors propagate.  Not cancellable; the file
                is written in place (no temporary file).
        """
        import warnings
        import rasterio
        from rasterio.errors import NotGeoreferencedWarning
        t0 = time.time()
        try:
            if self.hypercube is None or not hasattr(self.hypercube, "shape"):
                raise ValueError("spectrogram is None or not a numpy array")
            percents = 0

            with warnings.catch_warnings(), rasterio.open(self.save_path, "w", **self.geo_meta) as dst:
                warnings.simplefilter('ignore', NotGeoreferencedWarning)   # no georeference on purpose
                for i in range(self.count):
                    data = np.ascontiguousarray(self.hypercube[i], dtype=self.dtype)   # views welcome (no-op if contiguous)
                    wl = self.linear_spectrum[i]
                    dst.update_tags(bidx=i + 1, WAVELENGTH=f"{wl:.3f}", WAVELENGTH_UNIT="nm")
                    dst.set_band_description(i + 1, f"Band {i + 1} (λ={wl:.3f} nm)")
                    dst.write(data, indexes=i + 1)
                    percents = int(((i + 1) / self.count) * 100)
                    self.progress(percents)
            return float(np.round(time.time() - t0, 2))
        except Exception:
            raise


class HypercubeWriter:
    """Write a band-first cube as ENVI BIP ``save_path`` plus ``save_path + '.hdr'``."""
    chunk_size = 100 * 1024 * 1024
    def __init__(self, *, save_path: str, config: dict, fsync: bool = True, hypercube: np.ndarray, linear_spectrum: np.ndarray, camera_bin: int = 1, progress=None):
        """Store the export parameters (no file is written yet).

        Args:
            save_path: Data file path; the header goes to ``save_path + '.hdr'``.
            config: Export metadata dict (``m``/``c`` are not needed).
                Optional ``started_at`` (ISO 8601 scan start, see
                :func:`modules.acquisition.build_raw_metadata`) becomes the UTC
                ``acquisition time`` (default: the export time).  Optional ``bit_depth``, ``sensor_bit_depth`` and ``full_scale`` (from
                :func:`modules.acquisition.camera_bit_info`) are written to the header as
                ``data bit depth``, ``sensor bit depth`` and ``data full scale``; optional scan
                temperatures (°C, see :data:`TEMPERATURE_HEADER_FIELDS`) as ``sensor temperature
                start/end/min/max``, ``dark temperature`` and ``target temperature``; optional
                scan settings and corrections (:data:`SCAN_HEADER_FIELDS`: camera model and
                serial number, exposure, gain, frame
                rate, bin, dark / hot-pixel / smile corrected, smile model, stage speeds).
            fsync: Accepted for API symmetry; currently unused.
            hypercube: Band-first ``(wavelength, slit, scan)`` array, kept by
                reference.  The header's ``data type`` and ``byte order`` follow its dtype.
            linear_spectrum: 1-D band wavelengths in nm (written to the header).
            camera_bin: Camera hardware bin (``0``/``None`` mean 1).
            progress: Optional ``callable(int percent)``.
        """
        self.progress = progress or (lambda value: None)
        self.save_path = save_path
        self.config = config
        self.camera_bin = int(camera_bin or 1)
        self.hypercube = hypercube
        self.linear_spectrum = linear_spectrum
        self.fsync = fsync


    def run(self):
        """Write the BIP data (one slit line at a time) and the ENVI header.

        Returns:
            float: Elapsed seconds (rounded to 0.01).

        Raises:
            ValueError: ``hypercube`` is ``None``/not an array, not 3-D, or of a
                dtype ENVI cannot store (checked before any file is written).
            OSError: file errors propagate.  Not cancellable, not atomic.
        """
        t0 = time.time()
        try:
            if self.hypercube is None or not hasattr(self.hypercube, "shape"):
                raise ValueError("spectrogram is None or not a numpy array")

            bands, lines, samples = self.hypercube.shape
            data_type, byte_order = envi_data_type(self.hypercube.dtype)
            percents = 0

            bip = self.hypercube.transpose(1,2,0)
            with open(f'{self.save_path}', 'wb') as fb:
                for i in range(bip.shape[0]):
                    fb.write(bip[i, :, :].copy())
                    percents = int(((i + 1) / bip.shape[0]) * 100)
                    self.progress(percents)

            acquisition = acquisition_time_utc(self.config)

            with open(f'{self.save_path}.hdr', "w", encoding='utf-8', newline='\n') as file:
                file.write('ENVI\n')
                file.write('description = {Truth image generated by NARIT Feat. PSU}\n')
                file.write(f'data type = {data_type}\n')
                file.write(f'byte order = {byte_order}\n')
                file.write('interleave = BIP\n')
                file.write(f'acquisition time = {acquisition}\n')
                file.write(f'lines = {lines}\n')
                file.write(f'samples = {samples}\n')
                file.write(f'bands = {bands}\n')
                for line in metadata_header_lines(self.config):
                    file.write(line + '\n')
                file.write('wavelength units = Nanometers\n')
                file.write('wavelength = {%s}\n' % ', '.join(np.asarray(self.linear_spectrum).astype('str').tolist()))
            return float(np.round(time.time() - t0, 2))

        except Exception:
            raise


class SpectrogramWriter:
    """Write a raw spectrogram (header + frames) atomically and cancellably.

    Data goes to ``save_path + '.tmp'`` and is renamed to ``save_path`` on success;
    the temporary file is removed on error or cancellation.
    """
    def __init__(self, *, save_path: str, config: dict, spectrogram: np.ndarray, pixel_size: float, camera_bin: int = 1, fsync: bool = True, progress=None):
        """Store the export parameters (no file is written yet).

        Args:
            save_path: Output file path (parent directory is created).
            config: Calibration dict with ``m`` (nm/pixel at bin 1) and ``c`` (nm).
            spectrogram: ``(frames, height, width)`` array or memmap; iterated
                per frame, kept by reference.
            pixel_size: Spatial pixel size written to the header.
            camera_bin: Camera hardware bin; the stored slope is ``m * camera_bin``.
            fsync: ``os.fsync`` before the final rename.
            progress: Optional ``callable(int percent)``.
        """
        self.progress = progress or (lambda value: None)
        self.save_path = save_path
        self.config = config
        self.spectrogram = spectrogram
        self.pixel_size = float(pixel_size)
        self.camera_bin = int(camera_bin or 1)
        self.fsync = fsync
        self._cancelled = False


    def cancel(self):
        """Request cancellation; ``run`` stops before the next frame (thread-safe flag)."""
        self._cancelled = True


    def run(self):
        """Write the file.

        Returns:
            float: Elapsed seconds (rounded to 0.01); ``0.0`` for an empty
            spectrogram.  On cancellation the partial file is deleted,
            ``progress(0)`` is called and the elapsed time is returned.

        Raises:
            ValueError: ``spectrogram`` is ``None``/not an array or has fewer
                than 3 dimensions.
            TypeError: ``config['m']`` is missing/non-numeric.
            OSError: file errors propagate (the temporary file is removed).
        """
        t0 = time.time()
        try:
            if self.spectrogram is None or not hasattr(self.spectrogram, "shape"):
                raise ValueError("spectrogram is None or not a numpy array")

            shp = self.spectrogram.shape
            if len(shp) < 3:
                raise ValueError(f"Invalid spectrogram shape: {shp} (expected at least 3 dims: N,H,W)")
            number, height, width = shp[0], shp[1], shp[2]
            if number == 0:
                self.progress(100)
                return float(0.0)
                return

            bin_factor = int(getattr(self, 'camera_bin', 1) or 1)
            m = float(self.config.get('m')) * bin_factor
            c = float(self.config.get('c'))
            header_bin = struct.pack('<fffHH', m, c, self.pixel_size, height, width)

            tmp_path = self.save_path + ".tmp"
            os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)

            last_pct = -1
            with open(tmp_path, mode='wb', buffering=128 * 1024 * 1024) as f:
                f.write(header_bin)
                for i, frame in enumerate(self.spectrogram):
                    if self._cancelled:
                        try:
                            f.flush()
                            if self.fsync:
                                os.fsync(f.fileno())
                        except Exception:
                            pass
                        f.close()
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
                        self.progress(0)
                        return float(np.round(time.time() - t0, 2))
                        return
                    f.write(frame.tobytes())
                    pct = int((i + 1) * 100 / number)
                    if pct != last_pct:
                        self.progress(pct)
                        last_pct = pct

                f.flush()
                if self.fsync:
                    os.fsync(f.fileno())

            os.replace(tmp_path, self.save_path)
            return float(np.round(time.time() - t0, 2))

        except Exception:
            try:
                if os.path.exists(self.save_path + ".tmp"):
                    os.remove(self.save_path + ".tmp")
            except Exception:
                pass
            raise


