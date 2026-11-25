"""Offline reference calibration and cube processing, independent of Qt.

Formulas, interpolation, clipping and output layouts preserve the source GUI.
Callbacks receive integer progress values (band counts), not Qt signals.
This processing is never inserted into live camera acquisition.

Data conventions:
    * Inputs are ENVI files on disk (``.hdr`` + binary, any interleave that
      ``modules.cube_io.envi`` reads) described by the dict returned by
      ``parse_envi_header``: ``lines`` (first spatial axis, the slit direction),
      ``samples`` (second spatial axis, the scan direction), ``bands``,
      ``wavelength`` (float array in nm or ``None``) and ``data type``.
      Bands are read as float32 ``(lines, samples)`` images in raw sensor counts:
      one at a time from BSQ files; BIL/BIP cubes are converted to BSQ in RAM once
      (``read_envi_cube_bsq``, kept for the next step on the same cube) or, when too
      large, read in blocks of up to :data:`BAND_BLOCK_BYTES`.
    * A scan still in memory (``(lines, samples, bands)``, the scan's ``(slit, scan,
      wavelength)`` cube) needs no file: ``compute_rowwise_factors_array`` fits the same
      factors as the file version, ``apply_rowwise_factors`` applies them (after the scan,
      on the finished cube).
    * Calibration factors are multipliers applied to counts: ``rowwise`` factors
      are ``(lines, bands)`` float32 (one factor per slit row and band), global
      "fluorescence" factors are ``(bands,)`` (one factor per band).
    * Crop boxes are ``(x0, x1, y0, y1)`` half-open pixel ranges, ``x`` along
      ``samples`` and ``y`` along ``lines``.
    * Cubes written by ``create_datacube_logic`` / ``create_datacube_resampled_logic``
      are ENVI BIP files, ``(lines, samples, bands)`` little-endian, float32 when a
      calibration is applied, otherwise the input's dtype (integers rounded and
      clipped to its range), with a ``<save_path>.hdr`` text header (wavelengths in nm)
      that keeps the source's acquisition time, temperatures, scan settings and
      corrections (``carried_header_lines``; bit depth / full scale only uncalibrated).
    * The temporary cubes of ``build_raw_cube_tmp`` / ``build_cal_cube_tmp`` are
      header-less band-stacked memmaps of shape ``(lines * bands, samples)``:
      element ``[y + b * lines, x]`` is pixel ``(y, x)`` of band ``b``.

Threading:
    Every function is plain synchronous NumPy/SciPy code with no shared state and
    no Qt objects, so it may run on any thread. The GUI runs them on a worker
    thread through ``modules.radiometric.workers.Worker``. ``progress_callback``
    is called from the calling thread with an ``int``: ``0`` at start and then the
    number of bands processed so far (``1 .. n_bands``), not a percentage.

Example:
    >>> import numpy as np
    >>> from modules.cube_io.envi import parse_envi_header
    >>> from modules.radiometric import compute_fluorescence_factors
    >>> meta = parse_envi_header('reference.bsq')                # doctest: +SKIP
    >>> mask = np.zeros((meta['lines'], meta['samples']), bool)  # doctest: +SKIP
    >>> mask[10:50, 20:80] = True                                # doctest: +SKIP
    >>> compute_fluorescence_factors('reference.bsq', meta, mask)['factors'].shape  # doctest: +SKIP
    (bands,)
"""
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from modules.cube_io.envi import (parse_envi_header, _read_envi_band_float32, read_envi_bands_float32,
                                  read_envi_cube_bsq, envi_dtype)
from modules.cube_io.writers import carried_header_lines

#: Reflectance tables of the reference targets (Spectralon, PVC, ...): Excel sheets without
#: header, column 0 wavelength (nm), column 1 reflectance. Drop a new file here to offer it.
REFLECTANCE_PROFILE_DIR = Path(__file__).resolve().parent / 'reflectance_profiles'


def list_reflectance_profiles(folder=None):
    """Reflectance tables (``.xlsx``/``.xls``) in ``folder``, sorted by file name.

    Args:
        folder: Directory to list; defaults to :data:`REFLECTANCE_PROFILE_DIR`.

    Returns:
        list[pathlib.Path]: Absolute paths; empty if the folder does not exist. Excel
        lock files (``~$...``) are skipped.
    """
    folder = Path(folder) if folder is not None else REFLECTANCE_PROFILE_DIR
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in ('.xlsx', '.xls') and not p.name.startswith('~$')),
                  key=lambda p: p.name.lower())


def check_factor_bands(n_bands, factor_wavelengths, cube_meta):
    """Refuse calibration factors whose bands are not the cube's bands.

    Factors are applied by band index, so factors made from a cube with another band
    count or wavelength axis (for example another wavelength range) would be applied to
    the wrong bands or run out of range.

    Args:
        n_bands: Number of bands of the factors (last axis of row-wise or fluorescence factors).
        factor_wavelengths: Band centres (nm) of the cube the factors were made from, or
            ``None`` if unknown (then only the band count is checked).
        cube_meta: Header dict of the cube the factors are applied to (``bands``, ``wavelength``).

    Raises:
        ValueError: The band counts differ, or a band centre is further from the cube's than
            a quarter of the cube's median band step (at least 0.01 nm, since ENVI headers
            store 2 decimals).
    """
    n_bands, n_cube = int(n_bands), int(cube_meta['bands'])
    if n_bands != n_cube:
        raise ValueError(f"Calibration factors for {n_bands} bands do not fit this cube of {n_cube} bands: "
                         "use a reference or profile with the same bands.")
    cube_wl = cube_meta.get('wavelength')
    if factor_wavelengths is None or cube_wl is None:
        return
    fw = np.asarray(factor_wavelengths, dtype=np.float64).ravel()
    cw = np.asarray(cube_wl, dtype=np.float64).ravel()
    if fw.size != cw.size:
        raise ValueError(f"Calibration factors list {fw.size} wavelengths, this cube {cw.size}.")
    step = float(np.median(np.abs(np.diff(cw)))) if cw.size > 1 else 0.0
    tol = max(step / 4.0, 0.01)
    gap = np.abs(fw - cw)
    if not np.all(gap <= tol):
        raise ValueError(f"Calibration factors for {fw[0]:.2f}-{fw[-1]:.2f} nm do not fit this cube's "
                         f"{cw[0]:.2f}-{cw[-1]:.2f} nm (bands up to {np.max(gap):.3g} nm apart, tolerance "
                         f"{tol:.3g} nm): use a reference or profile with the same bands.")


def _check_params_factors(params, cube_meta):
    """:func:`check_factor_bands` for the factors a processing call applies (row-wise first, as applied)."""
    factors = params.get('rowwise_factors')
    if factors is None:
        factors = params.get('factors')
    if factors is not None:
        check_factor_bands(np.shape(factors)[-1], params.get('factor_wavelengths'), cube_meta)


def _wavelength_sidecar(profile_path):
    """``<name>.wavelength.npy`` next to the profile ``<name>.npy``."""
    return os.path.splitext(str(profile_path))[0] + '.wavelength.npy'


def save_factor_profile(path, factors, wavelengths=None):
    """Save factors as ``.npy`` and their band centres as ``<name>.wavelength.npy`` beside it.

    The profile file is what ``np.save(path, factors)`` writes. Without ``wavelengths`` a
    sidecar left by an earlier save to the same name is removed, so it never describes
    other factors.

    Args:
        path: Profile path; ``.npy`` is appended if missing, as ``np.save`` does.
        factors: Row-wise ``(lines, bands)`` or fluorescence ``(bands,)`` factors.
        wavelengths: Band centres (nm) of the cube the factors were made from, or ``None``.

    Returns:
        str: The profile path written.

    Raises:
        ValueError: ``wavelengths`` does not give one value per band.
    """
    path = str(path)
    if not path.endswith('.npy'):
        path += '.npy'
    sidecar = _wavelength_sidecar(path)
    if wavelengths is not None:
        wavelengths = np.asarray(wavelengths, dtype=np.float64).ravel()
        if wavelengths.size != np.shape(factors)[-1]:
            raise ValueError(f"{wavelengths.size} wavelengths for {np.shape(factors)[-1]} bands")
    np.save(path, factors)
    if wavelengths is not None:
        np.save(sidecar, wavelengths)
    elif os.path.exists(sidecar):
        os.remove(sidecar)
    return path


def load_factor_profile(path):
    """Load a ``.npy`` factor profile and, if present, its ``<name>.wavelength.npy`` sidecar.

    Returns:
        tuple: ``(factors, wavelengths)``; ``wavelengths`` is a float64 array (nm) or
        ``None`` for a profile saved without one (only its band count can be checked).

    Raises:
        ValueError: The sidecar does not give one wavelength per band.
    """
    factors = np.load(path)
    sidecar = _wavelength_sidecar(path)
    if not os.path.isfile(sidecar):
        return factors, None
    wavelengths = np.asarray(np.load(sidecar), dtype=np.float64)
    n_bands = np.shape(factors)[-1] if np.ndim(factors) else 0
    if wavelengths.ndim != 1 or wavelengths.size != n_bands:
        raise ValueError(f"{os.path.basename(sidecar)} lists {wavelengths.size} wavelengths "
                         f"for a profile of {n_bands} bands")
    return factors, wavelengths


#: Memory for a block of bands read at once from a BIL/BIP file (see :class:`_BandReader`).
BAND_BLOCK_BYTES = 256 * 1024 * 1024


class _BandReader:
    """``reader(i)``: band ``i`` of an ENVI file as a new float32 ``(lines, samples)`` array.

    BSQ bands are read one by one.  A BIL/BIP cube is converted to BSQ in RAM once and
    kept for the following steps on it (:func:`read_envi_cube_bsq`); a cube too large for
    that is read in blocks of up to :data:`BAND_BLOCK_BYTES`, starting at the requested band
    (or ending at it when reading backwards).  Values equal :func:`_read_envi_band_float32`.
    """

    def __init__(self, filepath, meta):
        self.filepath, self.meta = filepath, meta
        self.n_bands = int(meta['bands'])
        band_bytes = 4 * int(meta['lines']) * int(meta['samples'])
        bsq = str(meta.get('interleave', 'bsq')).lower().strip() == 'bsq'
        self.size = 1 if bsq else max(1, min(self.n_bands, BAND_BLOCK_BYTES // max(band_bytes, 1)))
        self.start, self.block = 0, None
        self.cube = None if bsq else read_envi_cube_bsq(filepath, meta)

    def __call__(self, i):
        i = int(i)
        if self.cube is not None:
            return self.cube[i].astype(np.float32)
        if self.size == 1:
            return _read_envi_band_float32(self.filepath, self.meta, i)
        if self.block is None or not self.start <= i < self.start + len(self.block):
            self.start = i if self.block is None or i >= self.start else max(0, i + 1 - self.size)
            self.block = read_envi_bands_float32(self.filepath, self.meta, self.start,
                                                 min(self.n_bands, self.start + self.size))
        return self.block[i - self.start].copy()


def _write_bip(bands, path):
    """Write band-stacked ``(bands, lines, samples)`` data to ``path`` as BIP.

    Transposes a few lines at a time, in cache: writing the transposed view at once
    (or band by band into a BIP array) is several times slower.
    """
    n_bands, n_lines, n_samples = bands.shape
    step = max(1, (4 << 20) // max(1, n_bands * n_samples * bands.itemsize))
    with open(path, 'wb') as fh:
        for y in range(0, n_lines, step):
            np.ascontiguousarray(bands[:, y:y + step].transpose(1, 2, 0)).tofile(fh)


def _apply_spatial_binning(img, n, to_uint16=False):
    """Average spatial blocks, dropping incomplete edge blocks as before."""
    if n <= 1:
        return img
    h, w = img.shape
    h_new, w_new = h // n, w // n
    img_cropped = img[:h_new * n, :w_new * n]
    reshaped = img_cropped.reshape(h_new, n, w_new, n)
    binned = reshaped.mean(axis=(1, 3))
    if to_uint16:
        return np.round(binned).astype(np.uint16)
    return binned.astype(np.float32)


def _output_type(raw_meta, is_cal):
    """``(ENVI data type, numpy dtype)`` of a crop export: float32 when calibrated, else the input's."""
    if is_cal:
        return 4, np.dtype(np.float32)
    in_dt = int(raw_meta.get('data type', 12))
    return in_dt, envi_dtype(in_dt)


def _cast_band(img, dtype):
    """``img`` as ``dtype``; for integer types rounded and clipped to the type's range."""
    if dtype.kind == 'f':
        return img.astype(dtype, copy=False)
    info = np.iinfo(dtype)
    return np.clip(np.round(img), info.min, info.max).astype(dtype)


def rowwise_calculation_heavy_logic(ref_filepath, ref_meta, ref_excel,
                                    avg_window_size, mask,
                                    progress_callback=None, window_fit='mean'):
    """Fit reference-spectrum factors per image row using the selected ROI.

    Each slit row's reference counts come from a window of ``avg_window_size`` ROI rows around it
    (one-sided at the ROI's ends; rows outside the ROI use the nearest ROI row). ``window_fit``:
    ``'mean'`` - the window's mean (the original behaviour); ``'linear'`` - a pixel-weighted
    line through the window's row means evaluated at the row, which stays unbiased where the
    vignetting falls off along the slit (with the mean, a window one-sided at the ROI end on a
    falloff of ``g`` per row is off by about ``g * window / 2``).

    Returns:
        dict: ``factors`` ``(lines, bands)`` float32, ``span`` (first, last ROI row),
        ``out_of_range_bands`` (bands outside the reference table: factor NaN, so their
        reflectance is NaN instead of an extrapolated guess), ``table_range_nm``,
        ``roi_mean_spectrum`` (float64 ``(bands,)``: mean counts of the valid ROI pixels) and
        ``reference`` (header metadata: ``reflectance_reference`` table file,
        ``reflectance_reference_cube``, ``reflectance_reference_rows``, plus
        ``calibration_warnings`` when bands fall outside the table).
    """
    readers = []                                # opened on the first band: after the ROI checks

    def read_box(i, r0, r1, c0, c1):
        if not readers:
            readers.append(_BandReader(ref_filepath, ref_meta))
        return readers[0](i)[r0:r1, c0:c1]

    return _rowwise_factors(read_box, int(ref_meta['bands']), ref_meta['lines'], ref_meta['samples'],
                            ref_meta['wavelength'], ref_excel, avg_window_size, mask, progress_callback,
                            os.path.basename(str(ref_filepath)), window_fit)


def compute_rowwise_factors_array(cube, wavelengths, ref_excel, avg_window_size, mask,
                                  progress_callback=None, source_name='in-memory cube', window_fit='mean'):
    """:func:`rowwise_calculation_heavy_logic` for a cube held in memory (no temporary file).

    The factors are bit-identical to the file version on the same counts: each band is read
    as float32, as ``_read_envi_band_float32`` reads a file band.

    Args:
        cube: ``(lines, samples, bands)`` array or memmap (the scan's ``(slit, scan, wavelength)``
            layout, any real dtype) in dark-subtracted counts.
        wavelengths: Band centres in nm, one per band.
        ref_excel: Reflectance table of the reference target (see :func:`list_reflectance_profiles`).
        avg_window_size: Number of ROI rows averaged for each slit row.
        mask: Boolean ROI ``(lines, samples)``.
        progress_callback: Called with ``0`` and then the number of bands done.
        source_name: Name recorded as ``reflectance_reference_cube`` in ``reference``.

    Returns:
        dict: As :func:`rowwise_calculation_heavy_logic`.

    Raises:
        ValueError: ``cube`` is not 3-D, ``wavelengths`` does not give one value per band, the
            mask has another shape or covers no pixel.
    """
    cube = np.asarray(cube) if not isinstance(cube, np.ndarray) else cube
    if cube.ndim != 3:
        raise ValueError(f'cube must be (lines, samples, bands), got shape {cube.shape}')
    wl = np.asarray(wavelengths, dtype=np.float64).ravel()
    if wl.size != cube.shape[2]:
        raise ValueError(f'{wl.size} wavelengths for a cube of {cube.shape[2]} bands')
    mask = np.asarray(mask, dtype=bool)
    return _rowwise_factors(lambda i, r0, r1, c0, c1: np.array(cube[r0:r1, c0:c1, i], dtype=np.float32),
                            cube.shape[2], cube.shape[0], cube.shape[1], wl, ref_excel,
                            avg_window_size, mask, progress_callback, str(source_name), window_fit)


def reflectance_at(ref_excel, wavelengths):
    """Reflectance of a reference table at ``wavelengths`` (nm), as the factor fit uses it.

    Cubic spline through the table, never extrapolated: wavelengths outside the table give NaN.

    Returns:
        tuple: ``(reflectance float32 (n,), (table min nm, table max nm))``.
    """
    data_ref = pd.read_excel(ref_excel, header=None).to_numpy()
    # no extrapolation: a cubic spline outside the table goes wild (PVC 400-999.5 nm gives
    # -885 at 1200 nm); bands outside it get NaN factors and are reported
    f_ref = CubicSpline(data_ref[:, 0], data_ref[:, 1], extrapolate=False)
    table_range = (float(np.min(data_ref[:, 0])), float(np.max(data_ref[:, 0])))
    return f_ref(np.asarray(wavelengths, dtype=np.float64)).astype(np.float32), table_range


def processed_cube_problem(meta) -> Optional[str]:
    """Why a cube may not be used as counts for reflectance factors, or ``None`` if it may.

    Factors (row-wise, flat-field or fluorescence) are ``R / counts`` of dark-subtracted counts.
    A cube already divided by a flat-field profile (``flatfield_profile``) or already
    reflectance (``reflectance_reference``) would be divided a second time: by ``F`` again at
    every slit row, or by the white counts on top of an earlier white. Reads both the export
    header labels as :func:`modules.cube_io.envi.parse_envi_header` gives them (``'flat-field
    profile'``) and the metadata keys (``'flatfield_profile'``).
    """
    meta = meta or {}

    def field(*names):
        for name in names:
            value = meta.get(name)
            if value not in (None, ''):
                return str(value).strip()
        return None

    reflectance = field('reflectance_reference', 'reflectance reference')
    if reflectance:
        return (f'it is already reflectance (calibrated with {reflectance}): calibrating it again divides '
                'by the reference counts twice. Use the scan in counts.')
    flat = field('flatfield_profile', 'flat-field profile')
    if flat:
        return (f'it is already flat-fielded (counts divided by {flat}): reflectance factors divide by the '
                'slit falloff again. Use the scan in counts.')
    return None


def _rowwise_factors(read_box, n_bands, H, W, wl, ref_excel, avg_window_size, mask,
                     progress_callback, source_name, window_fit='mean'):
    """Shared body of the row-wise factor fit.

    ``read_box(i, r0, r1, c0, c1)`` returns band ``i`` restricted to rows ``r0:r1`` and columns
    ``c0:c1`` (the ROI's bounding box) as float32: every value the fit uses lies inside it.
    ``window_fit``: ``'mean'`` (the window's mean counts) or ``'linear'`` (a line through the
    window's rows, evaluated at the row).
    """
    progress_callback = progress_callback or (lambda value: None)
    R_ref_per_band, table_range = reflectance_at(ref_excel, wl)

    mask = np.asarray(mask, dtype=bool)         # a 0/1 integer mask would index rows, not select pixels
    if mask.shape != (H, W):
        raise ValueError('ROI mask shape does not match reference image.')

    rows_has_roi = np.where(mask.any(axis=1))[0]
    if len(rows_has_roi) == 0:
        raise ValueError('Selected ROI does not cover any pixels.')
    top_roi, bot_roi = int(rows_has_roi.min()), int(rows_has_roi.max())

    def pick_rows_in_roi(y):
        nearest = rows_has_roi[np.argmin(np.abs(rows_has_roi - y))]
        idx = np.where(rows_has_roi == nearest)[0][0]
        left = max(0, idx - (avg_window_size // 2))
        right = min(len(rows_has_roi), left + avg_window_size)
        if right - left < avg_window_size:
            left = max(0, right - avg_window_size)
        return rows_has_roi[left:right]

    rowwise_factors = np.zeros((H, len(wl)), dtype=np.float32)
    out_of_range = int(np.count_nonzero(~np.isfinite(R_ref_per_band)))

    # The averaging window of a row depends only on the row: rows_has_roi[left:right].
    # Per band, all window means then come from cumulative ROI row sums and counts
    # (float64): the NaN-ignoring mean of the window's ROI pixels, as np.nanmean.
    left = np.zeros(H, dtype=np.intp)
    right = np.zeros(H, dtype=np.intp)
    for y in range(H):
        rows_window = pick_rows_in_roi(int(np.clip(y, top_roi, bot_roi)))
        if len(rows_window):
            left[y] = np.searchsorted(rows_has_roi, rows_window[0])
            right[y] = left[y] + len(rows_window)
    cols = np.where(mask.any(axis=0))[0]
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    r0, r1 = top_roi, bot_roi + 1
    roi_mask = mask[rows_has_roi, c0:c1]
    box_mask = mask[r0:r1, c0:c1]               # every ROI pixel: ``img[mask]`` of the full band
    box_rows = rows_has_roi - r0
    roi_mean = np.full(n_bands, np.nan, dtype=np.float64)
    if window_fit not in ('mean', 'linear'):
        raise ValueError(f"window_fit must be 'mean' or 'linear', got {window_fit!r}")
    linear = window_fit == 'linear'
    x_rows = (rows_has_roi - top_roi).astype(np.float64)                   # row positions, ROI-relative
    x_eval = (np.clip(np.arange(H), top_roi, bot_roi) - top_roi).astype(np.float64)

    def cumulative(values):
        return np.concatenate(([0], np.cumsum(values)))

    progress_callback(0)
    for i in range(n_bands):
        box = read_box(i, r0, r1, c0, c1)
        roi = box[box_rows]
        valid = roi_mask & ~np.isnan(roi)
        row_sum = np.where(valid, roi, 0).sum(axis=1, dtype=np.float64)
        n_valid = int(valid.sum())
        if n_valid:
            roi_mean[i] = row_sum.sum() / n_valid
        inf_row = ~np.isfinite(row_sum)          # a +-inf pixel makes its windows' mean not finite
        finite_sum = np.where(inf_row, 0.0, row_sum)
        sums = cumulative(finite_sum)
        n_row = valid.sum(axis=1)
        counts = cumulative(n_row)
        infs = cumulative(inf_row)
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_val = (sums[right] - sums[left]) / (counts[right] - counts[left])
            if linear:
                # pixel-weighted least-squares line through the window's ROI rows, evaluated at the
                # row itself (its nearest ROI row outside the span): unbiased on a falloff gradient,
                # where the plain mean of a window that is one-sided at the ROI ends is not
                n_f = n_row.astype(np.float64)
                sx = cumulative(n_f * x_rows)
                sxx = cumulative(n_f * x_rows * x_rows)
                sxs = cumulative(x_rows * finite_sum)
                rows_used = cumulative(n_row > 0)
                w, s = counts[right] - counts[left], sums[right] - sums[left]
                swx, swxx, swxs = sx[right] - sx[left], sxx[right] - sxx[left], sxs[right] - sxs[left]
                den = w * swxx - swx * swx
                slope = (w * swxs - swx * s) / den
                fitted = (s - slope * swx) / w + slope * x_eval
                use = (rows_used[right] - rows_used[left] >= 2) & np.isfinite(fitted) & (den > 0)
                mean_val = np.where(use, fitted, mean_val)
        bad = ~np.isfinite(mean_val) | (mean_val <= 0) | (infs[right] > infs[left])
        if bad.any():
            fallback = np.nanmedian(box[box_mask])
            if not np.isfinite(fallback) or fallback <= 0:
                fallback = 1.0
            mean_val[bad] = fallback
        rowwise_factors[:, i] = R_ref_per_band[i] / mean_val

        progress_callback(i + 1)
    reference = {
        'reflectance_reference': os.path.basename(str(ref_excel)),
        'reflectance_reference_cube': source_name,
        'reflectance_reference_rows': f'{top_roi}-{bot_roi}',
    }
    if out_of_range:
        reference['calibration_warnings'] = (
            f'{out_of_range} bands outside the reference table '
            f'({table_range[0]:g}-{table_range[1]:g} nm) have no reflectance (NaN)')
    return {'factors': rowwise_factors, 'span': (top_roi, bot_roi), 'out_of_range_bands': out_of_range,
            'table_range_nm': table_range, 'roi_mean_spectrum': roi_mean, 'window_fit': window_fit,
            'reference': reference}


def compute_flatfield_factors_array(cube, wavelengths, ref_excel, mask, flatfield,
                                    progress_callback=None, source_name='in-memory cube'):
    """Row-wise factors from a small reference target and a flat-field profile of the slit.

    The flat-field profile ``F(row, band)`` (``modules.flatfield.compute_flatfield``) carries how
    the response changes along the slit; the reference target only gives the scale per band, so
    its ROI may cover a few slit rows (a Spectralon disc). With ``c`` the ROI counts::

        scale[band]        = R_ref[band] / mean over the ROI of (c / F[row, band])
        factors[row, band] = scale[band] / F[row, band]

    Unlike :func:`compute_rowwise_factors_array`, rows outside the ROI are calibrated with their
    own measured response instead of the nearest ROI row's.

    Args:
        cube: ``(lines, samples, bands)`` scan of the reference target, dark-subtracted counts.
        wavelengths: Band centres in nm, one per band.
        ref_excel: Reflectance table of the reference target.
        mask: Boolean ROI ``(lines, samples)`` on the target.
        flatfield: Flat-field profile ``(lines, bands)``, finite and positive.
        progress_callback: Called with ``0`` and then the number of bands done.
        source_name: Name recorded as ``reflectance_reference_cube`` in ``reference``.

    Returns:
        dict: As :func:`rowwise_calculation_heavy_logic` (``window_fit`` is ``'flatfield'``), plus
        ``scale`` float64 ``(bands,)``. Bands outside the reflectance table, or without positive
        ROI counts, have NaN factors.

    Raises:
        ValueError: Wrong shapes, an empty ROI, or a profile that is not finite and positive.
    """
    progress_callback = progress_callback or (lambda value: None)
    if getattr(cube, 'ndim', None) != 3:
        raise ValueError(f'cube must be (lines, samples, bands), got shape {getattr(cube, "shape", None)}')
    H, W, n_bands = cube.shape
    wl = np.asarray(wavelengths, dtype=np.float64).ravel()
    if wl.size != n_bands:
        raise ValueError(f'{wl.size} wavelengths for a cube of {n_bands} bands')
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (H, W):
        raise ValueError('ROI mask shape does not match reference image.')
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) == 0:
        raise ValueError('Selected ROI does not cover any pixels.')
    flat = np.asarray(flatfield, dtype=np.float64)
    if flat.shape != (H, n_bands):
        raise ValueError(f'Flat-field profile {flat.shape} does not fit this cube ({H} rows, {n_bands} bands): '
                         'make one at the same camera setting.')
    if not np.all(np.isfinite(flat)) or np.any(flat <= 0):
        raise ValueError('The flat-field profile must be finite and positive everywhere.')
    R_ref, table_range = reflectance_at(ref_excel, wl)
    out_of_range = int(np.count_nonzero(~np.isfinite(R_ref)))

    cols = np.where(mask.any(axis=0))[0]
    r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    box_mask = mask[r0:r1, c0:c1]
    roi_mean = np.full(n_bands, np.nan, dtype=np.float64)
    scale = np.full(n_bands, np.nan, dtype=np.float64)
    no_signal = 0
    progress_callback(0)
    for i in range(n_bands):
        box = np.asarray(cube[r0:r1, c0:c1, i], dtype=np.float64)
        valid = box_mask & np.isfinite(box)
        n_valid = int(valid.sum())
        if n_valid:
            row_sum = np.where(valid, box, 0.0).sum(axis=1)
            roi_mean[i] = row_sum.sum() / n_valid
            evened = (row_sum / flat[r0:r1, i]).sum() / n_valid
            if evened > 0:
                scale[i] = float(R_ref[i]) / evened
        if not np.isfinite(scale[i]) and np.isfinite(R_ref[i]):
            no_signal += 1
        progress_callback(i + 1)
    factors = (scale[None, :] / flat).astype(np.float32)

    reference = {
        'reflectance_reference': os.path.basename(str(ref_excel)),
        'reflectance_reference_cube': source_name,
        'reflectance_reference_rows': f'{int(rows[0])}-{int(rows[-1])}',
    }
    warnings = []
    if out_of_range:
        warnings.append(f'{out_of_range} bands outside the reference table '
                        f'({table_range[0]:g}-{table_range[1]:g} nm) have no reflectance (NaN)')
    if no_signal:
        warnings.append(f'{no_signal} bands without positive reference counts have no reflectance (NaN)')
    if warnings:
        reference['calibration_warnings'] = '; '.join(warnings)
    return {'factors': factors, 'span': (int(rows[0]), int(rows[-1])), 'out_of_range_bands': out_of_range,
            'table_range_nm': table_range, 'roi_mean_spectrum': roi_mean, 'window_fit': 'flatfield',
            'scale': scale, 'reference': reference}


def apply_rowwise_factors(cube, factors, factor_wavelengths=None, wavelengths=None, rows_per_chunk=64,
                          progress_callback=None):
    """Calibrate a cube held in memory with row-wise factors: ``cube * factors[:, None, :]``.

    Gives the values :func:`create_datacube_logic` writes for the full, unbinned cube (each band
    as float32 times the factor in the factor's own dtype, rounded once to float32), without a
    temporary file. The cube is processed a few slit rows at a time, so no float64 copy of the
    whole cube is made.

    Args:
        cube: ``(lines, samples, bands)`` array or memmap, any real dtype (counts).
        factors: Row-wise factors ``(lines, bands)`` (:func:`compute_rowwise_factors_array`).
        factor_wavelengths: Band centres (nm) the factors were made for, or ``None``.
        wavelengths: Band centres (nm) of ``cube``, or ``None``; with both given they are
            compared by :func:`check_factor_bands`.
        rows_per_chunk: Slit rows multiplied per step.
        progress_callback: Optional ``callable(int)``, given the slit rows done after each step.

    Returns:
        numpy.ndarray: New float32 ``(lines, samples, bands)`` cube; NaN where a factor is NaN
        (bands outside the reflectance table).

    Raises:
        ValueError: The factors do not have the cube's slit rows or bands, or their band
            centres do not fit the cube's.
    """
    factors = np.asarray(factors)
    if not np.issubdtype(factors.dtype, np.floating):
        factors = factors.astype(np.float32)
    if cube.ndim != 3 or factors.ndim != 2:
        raise ValueError(f'expected a (lines, samples, bands) cube and (lines, bands) factors, '
                         f'got {cube.shape} and {factors.shape}')
    if factors.shape[0] != cube.shape[0]:
        raise ValueError(f'Factors for {factors.shape[0]} slit rows do not fit this cube of '
                         f'{cube.shape[0]} rows: make a reference at the same camera setting.')
    check_factor_bands(factors.shape[1], factor_wavelengths,
                       {'bands': cube.shape[2], 'wavelength': wavelengths})
    out = np.empty(cube.shape, dtype=np.float32)
    step = max(1, int(rows_per_chunk))
    for y in range(0, cube.shape[0], step):
        part = np.asarray(cube[y:y + step], dtype=np.float32)
        np.multiply(part, factors[y:y + step, None, :], out=out[y:y + step], casting='same_kind')
        if progress_callback is not None:
            progress_callback(min(y + step, cube.shape[0]))
    return out


def create_datacube_logic(params, progress_callback=None):
    """Crop, apply factors and spatially bin an ENVI cube; write BIP output."""
    progress_callback = progress_callback or (lambda value: None)
    raw_filepath = params['raw_filepath']
    raw_meta = params['raw_meta']
    save_path = params['save_path']
    x0, x1, y0, y1 = params['crop_coords']
    is_wavelength = params['is_wavelength']
    band_count = params['band_count']
    factors = params['factors']
    rowwise_factors = params['rowwise_factors']
    n_bin = params.get('binning_factor', 1)
    _check_params_factors(params, raw_meta)

    cubheight, cubwidth = (y1 - y0) // n_bin, (x1 - x0) // n_bin
    is_cal = (rowwise_factors is not None) or (factors is not None)
    out_dt, out_dtype = _output_type(raw_meta, is_cal)

    hdr_path = save_path + '.hdr'
    with open(hdr_path, 'w', encoding='utf-8', newline='\n') as hopen:
        hopen.write('ENVI\n')
        old_desc = raw_meta.get('description', 'Processed Data')
        bin_text = f"Spatial Binning = {n_bin}x{n_bin}" if n_bin > 1 else "Original Spatial Res"
        hopen.write(f"description = {{\n  {old_desc},\n  {bin_text}\n}}\n")
        hopen.write(f'samples = {cubwidth}\n')
        hopen.write(f'lines = {cubheight}\n')
        hopen.write(f'bands = {band_count}\n')
        hopen.write(f'data type = {out_dt}\n')
        hopen.write('interleave = bip\n')
        hopen.write('byte order = 0\n')
        for line in carried_header_lines(raw_meta, calibrated=is_cal,
                                         extra=params.get('reference_info') if is_cal else None):
            hopen.write(line + '\n')
        hopen.write('wavelength units = Nanometers\n')
        if is_wavelength and raw_meta.get('wavelength') is not None:
            wl_str = ','.join([str(item) for item in raw_meta['wavelength']])
            hopen.write(f'wavelength = {{{wl_str}}}\n')

    c = np.zeros((band_count, cubheight, cubwidth), dtype=out_dtype)
    reader = _BandReader(raw_filepath, raw_meta)
    progress_callback(0)

    for i in range(band_count):
        im = reader(i)
        roi_img = im[y0:y1, x0:x1]
        if rowwise_factors is not None:
            roi_img *= rowwise_factors[y0:y1, i].reshape(-1, 1)
        elif factors is not None:
            roi_img *= float(factors[i])
        out = _apply_spatial_binning(
            roi_img, n_bin, to_uint16=(out_dtype == np.uint16))
        c[i] = _cast_band(out, out_dtype)
        progress_callback(i + 1)

    _write_bip(c, save_path)
    return save_path


def create_datacube_resampled_logic(params, progress_callback=None):
    """Apply factors and linearly resample bands before spatial binning."""
    progress_callback = progress_callback or (lambda value: None)
    raw_filepath = params['raw_filepath']
    raw_meta = params['raw_meta']
    save_path = params['save_path']
    x0, x1, y0, y1 = params['crop_coords']
    is_wavelength = params['is_wavelength']
    factors = params['factors']
    rowwise_factors = params['rowwise_factors']
    lam_target = np.array(params['lam_target'], dtype=np.float32)
    n_bin = params.get('binning_factor', 1)
    _check_params_factors(params, raw_meta)

    wl_all = raw_meta['wavelength']
    idx_right = np.searchsorted(wl_all, lam_target, side='left')
    idx_left = np.clip(idx_right - 1, 0, len(wl_all) - 1)
    idx_right = np.clip(idx_right, 0, len(wl_all) - 1)
    denom = np.maximum(wl_all[idx_right] - wl_all[idx_left], 1e-6)
    alpha = (lam_target - wl_all[idx_left]) / denom

    cubheight, cubwidth = (y1 - y0) // n_bin, (x1 - x0) // n_bin
    is_cal = (rowwise_factors is not None) or (factors is not None)
    out_dt, out_dtype_np = _output_type(raw_meta, is_cal)

    hdr_path = save_path + '.hdr'
    with open(hdr_path, 'w', encoding='utf-8', newline='\n') as hopen:
        hopen.write('ENVI\n')
        old_desc = raw_meta.get('description', 'Processed Data')
        bin_text = f"Spatial Binning = {n_bin}x{n_bin}" if n_bin > 1 else "Original Spatial Res"
        hopen.write(f"description = {{\n  {old_desc},\n  Resampled = True,\n  {bin_text}\n}}\n")
        hopen.write(f'samples = {cubwidth}\n')
        hopen.write(f'lines = {cubheight}\n')
        hopen.write(f'bands = {len(lam_target)}\n')
        hopen.write(f'data type = {out_dt}\n')
        hopen.write('interleave = bip\n')
        hopen.write('byte order = 0\n')
        for line in carried_header_lines(raw_meta, calibrated=is_cal,
                                         extra=params.get('reference_info') if is_cal else None):
            hopen.write(line + '\n')
        hopen.write('wavelength units = Nanometers\n')
        if is_wavelength:
            wl_str = ','.join(f"{float(v):.2f}" for v in lam_target)
            hopen.write(f'wavelength = {{{wl_str}}}\n')

    c = np.zeros((len(lam_target), cubheight, cubwidth), dtype=out_dtype_np)
    reader = _BandReader(raw_filepath, raw_meta)
    progress_callback(0)

    for j in range(len(lam_target)):
        iL, iR, a = int(idx_left[j]), int(idx_right[j]), float(alpha[j])
        imL = reader(iL)
        imR = reader(iR)

        Wmin = min(imL.shape[1], imR.shape[1])
        roiL = imL[y0:y1, x0:min(x1, x0 + Wmin)]
        roiR = imR[y0:y1, x0:min(x1, x0 + Wmin)]

        if rowwise_factors is not None:
            roiL *= rowwise_factors[y0:y1, iL].reshape(-1, 1)
            roiR *= rowwise_factors[y0:y1, iR].reshape(-1, 1)
        elif factors is not None:
            roiL *= float(factors[iL])
            roiR *= float(factors[iR])

        roiJ = (1.0 - a) * roiL + a * roiR
        out = _apply_spatial_binning(
            roiJ, n_bin, to_uint16=(out_dtype_np == np.uint16))
        c[j] = _cast_band(out, out_dtype_np)
        progress_callback(j + 1)

    _write_bip(c, save_path)
    return save_path


def build_raw_cube_tmp(params, progress_callback=None):
    """Build the original uint16 cropped temporary band-stack memmap."""
    progress_callback = progress_callback or (lambda value: None)
    raw_filepath = params['folder']
    raw_meta = params['raw_meta']
    tmp_path = params['save_path']
    x0, x1, y0, y1 = params['crop']

    lines, samples = (y1 - y0), (x1 - x0)
    bands = raw_meta['bands']
    mm = np.memmap(tmp_path, mode='w+', dtype=np.uint16, shape=(lines * bands, samples))

    reader = _BandReader(raw_filepath, raw_meta)
    progress_callback(0)
    for i in range(bands):
        im = reader(i)
        roi = im[y0:y1, x0:x1]
        mm[lines * i: lines * (i + 1)] = np.clip(roi, 0, 65535).astype(np.uint16)
        progress_callback(i + 1)
    mm.flush()
    return {'path': tmp_path, 'lines': lines, 'samples': samples,
            'bands': bands, 'dtype': 'uint16'}


def build_cal_cube_tmp(params, progress_callback=None):
    """Build a float32 calibrated temporary band-stack memmap."""
    progress_callback = progress_callback or (lambda value: None)
    raw_filepath = params['raw_filepath']
    raw_meta = params['raw_meta']
    tmp_path = params['save_path']
    x0, x1, y0, y1 = params['crop']
    factors = params.get('factors', None)
    rowwise = params.get('rowwise_factors', None)
    _check_params_factors(params, raw_meta)

    lines, samples = (y1 - y0), (x1 - x0)
    bands = raw_meta['bands']
    mm = np.memmap(tmp_path, mode='w+', dtype=np.float32, shape=(lines * bands, samples))

    reader = _BandReader(raw_filepath, raw_meta)
    progress_callback(0)
    for i in range(bands):
        im = reader(i)
        roi = im[y0:y1, x0:x1]
        if rowwise is not None:
            fac = rowwise[y0:y1, i].reshape(-1, 1)
            roi = roi * fac
        elif factors is not None:
            roi = roi * float(factors[i])
        mm[lines * i: lines * (i + 1)] = roi.astype(np.float32)
        progress_callback(i + 1)
    mm.flush()
    return {'path': tmp_path, 'lines': lines, 'samples': samples,
            'bands': bands, 'dtype': 'float32'}


def fluo_calculation_heavy_logic(ref_filepath, ref_meta, mask, progress_callback=None):
    """Invert the smoothed ROI-average spectrum to form fluorescence factors."""
    progress_callback = progress_callback or (lambda value: None)
    num_bands = ref_meta['bands']
    avg_ref_spectrum = np.zeros(num_bands, dtype=np.float32)

    reader = _BandReader(ref_filepath, ref_meta)
    progress_callback(0)
    for i in range(num_bands):
        img = reader(i)
        roi_data = img[mask[:, :img.shape[1]]]
        avg_ref_spectrum[i] = np.nanmean(roi_data) if roi_data.size > 0 else 1.0
        progress_callback(i + 1)

    try:
        smoothed = savgol_filter(avg_ref_spectrum, window_length=11, polyorder=3)
        smoothed = np.where(smoothed <= 0, 1.0, smoothed)
    except Exception:
        smoothed = np.where(avg_ref_spectrum <= 0, 1.0, avg_ref_spectrum)

    return {'factors': 1.0 / smoothed}


