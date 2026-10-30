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
      Bands are read one at a time as float32 ``(lines, samples)`` images in raw
      sensor counts, so memory use stays at one or two bands.
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

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from modules.cube_io.envi import parse_envi_header, _read_envi_band_float32, envi_dtype
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
                                    progress_callback=None):
    """Fit reference-spectrum factors per image row using the selected ROI.

    Returns:
        dict: ``factors`` ``(lines, bands)`` float32, ``span`` (first, last ROI row),
        ``out_of_range_bands`` (bands outside the reference table: factor NaN, so their
        reflectance is NaN instead of an extrapolated guess), ``table_range_nm`` and
        ``reference`` (header metadata: ``reflectance_reference`` table file,
        ``reflectance_reference_cube``, ``reflectance_reference_rows``, plus
        ``calibration_warnings`` when bands fall outside the table).
    """
    progress_callback = progress_callback or (lambda value: None)
    data_ref = pd.read_excel(ref_excel, header=None).to_numpy()
    # no extrapolation: a cubic spline outside the table goes wild (PVC 400-999.5 nm gives
    # -885 at 1200 nm); bands outside it get NaN factors and are reported
    f_ref = CubicSpline(data_ref[:, 0], data_ref[:, 1], extrapolate=False)
    table_range = (float(np.min(data_ref[:, 0])), float(np.max(data_ref[:, 0])))
    wl = ref_meta['wavelength']

    H, W = ref_meta['lines'], ref_meta['samples']
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
    R_ref_per_band = f_ref(np.asarray(wl, dtype=np.float64)).astype(np.float32)
    out_of_range = int(np.count_nonzero(~np.isfinite(R_ref_per_band)))

    progress_callback(0)
    for i in range(ref_meta['bands']):
        ref_img = _read_envi_band_float32(ref_filepath, ref_meta, i)
        ref_roi = np.where(mask[:, :ref_img.shape[1]], ref_img[:, :ref_img.shape[1]], np.nan)
        fallback = np.nanmedian(ref_roi)
        if not np.isfinite(fallback) or fallback <= 0:
            fallback = 1.0

        for y in range(ref_roi.shape[0]):
            y_ref = int(np.clip(y, top_roi, bot_roi))
            rows_window = pick_rows_in_roi(y_ref)
            mean_val = np.nanmean(ref_roi[rows_window, :]) if len(rows_window) > 0 else fallback
            if not np.isfinite(mean_val) or mean_val <= 0:
                mean_val = fallback
            rowwise_factors[y, i] = R_ref_per_band[i] / mean_val

        progress_callback(i + 1)
    reference = {
        'reflectance_reference': os.path.basename(str(ref_excel)),
        'reflectance_reference_cube': os.path.basename(str(ref_filepath)),
        'reflectance_reference_rows': f'{top_roi}-{bot_roi}',
    }
    if out_of_range:
        reference['calibration_warnings'] = (
            f'{out_of_range} bands outside the reference table '
            f'({table_range[0]:g}-{table_range[1]:g} nm) have no reflectance (NaN)')
    return {'factors': rowwise_factors, 'span': (top_roi, bot_roi), 'out_of_range_bands': out_of_range,
            'table_range_nm': table_range, 'reference': reference}


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

    c = np.zeros((cubheight * band_count, cubwidth), dtype=out_dtype)
    progress_callback(0)

    for i in range(band_count):
        im = _read_envi_band_float32(raw_filepath, raw_meta, i)
        roi_img = im[y0:y1, x0:x1]
        if rowwise_factors is not None:
            roi_img *= rowwise_factors[y0:y1, i].reshape(-1, 1)
        elif factors is not None:
            roi_img *= float(factors[i])
        out = _apply_spatial_binning(
            roi_img, n_bin, to_uint16=(out_dtype == np.uint16))
        c[cubheight * i: cubheight * (i + 1)] = _cast_band(out, out_dtype)
        progress_callback(i + 1)

    bip = c.reshape((band_count, cubheight, cubwidth)).transpose(1, 2, 0)  # HWC
    bip.tofile(save_path)
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

    c = np.zeros((cubheight * len(lam_target), cubwidth), dtype=out_dtype_np)
    progress_callback(0)

    for j in range(len(lam_target)):
        iL, iR, a = int(idx_left[j]), int(idx_right[j]), float(alpha[j])
        imL = _read_envi_band_float32(raw_filepath, raw_meta, iL)
        imR = _read_envi_band_float32(raw_filepath, raw_meta, iR)

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
        c[cubheight * j: cubheight * (j + 1)] = _cast_band(out, out_dtype_np)
        progress_callback(j + 1)

    bip = c.reshape((len(lam_target), cubheight, cubwidth)).transpose(1, 2, 0)  # HWC
    bip.tofile(save_path)
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

    progress_callback(0)
    for i in range(bands):
        im = _read_envi_band_float32(raw_filepath, raw_meta, i)
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

    progress_callback(0)
    for i in range(bands):
        im = _read_envi_band_float32(raw_filepath, raw_meta, i)
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

    progress_callback(0)
    for i in range(num_bands):
        img = _read_envi_band_float32(ref_filepath, ref_meta, i)
        roi_data = img[mask[:, :img.shape[1]]]
        avg_ref_spectrum[i] = np.nanmean(roi_data) if roi_data.size > 0 else 1.0
        progress_callback(i + 1)

    try:
        smoothed = savgol_filter(avg_ref_spectrum, window_length=11, polyorder=3)
        smoothed = np.where(smoothed <= 0, 1.0, smoothed)
    except Exception:
        smoothed = np.where(avg_ref_spectrum <= 0, 1.0, avg_ref_spectrum)

    return {'factors': 1.0 / smoothed}


