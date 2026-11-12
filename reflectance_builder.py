# -*- coding: utf-8 -*-
"""Reflectance factors from a white-reference recording, through the same chain as the data.

In operation the camera delivers raw frames and the sorter calibrates them in real time::

    raw ─► dark+hot ─► flip ─► Smile ─► binning ─► × factor ─► model
                                                     ▲
    white reference (Spectralon) recorded raw ─► same steps ─► mean ─► factor = R_ref(λ) / white

so the white reference must pass the very same steps (same dark, flip, Smile, binning) before it
is averaged; only then ``counts × factor`` gives reflectance. This module does that with the
simulator's own plan and hsi's ``PreprocessingRuntime``, then computes the factors exactly like
hsi ``modules.radiometric`` (reference table through ``CubicSpline``; row-wise factors with the
ROI row window of ``rowwise_calculation_heavy_logic``) and saves them with hsi's
``save_factor_profile`` as ``reflectance/<data source>.npy`` (+ ``.wavelength.npy``).

An already binned white cube (ENVI with wavelengths) has no dark step in the chain, so its dark
must be given in the same form (a dark cube with the same bands) or explicitly waived.
"""
from __future__ import annotations

import datetime
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from modules.preprocessing import PreprocessingRuntime
from modules.radiometric import save_factor_profile

logger = logging.getLogger(__name__)


@dataclass
class WhiteReference:
    """Mean white-reference line after the chain, before reflectance."""
    mean: np.ndarray            # (slit rows, bands) float64
    wavelengths: np.ndarray     # (bands,) nm
    n_frames: int
    steps: dict                 # steps applied while averaging
    source: str


def white_mean(source, plan, frames: slice | None = None, dark_cube=None, backend: str = 'cpu',
               require_dark: bool = True) -> WhiteReference:
    """Average the white reference over its frames after the plan's steps.

    Args:
        source: ``FrameSource`` of the white recording (raw H5/ENVI or binned cube).
        plan: ``AcquisitionPlan`` built for ``source`` with the *data's* configuration
            (``sim_config.build_plan``), so dark/flip/Smile/binning are the data's.
        frames: Scan positions to use (e.g. only where the Spectralon is under the slit).
        dark_cube: For a binned white cube only: a ``FrameSource`` with the same frame shape
            whose mean is subtracted (the dark that the raw chain would have removed).
        backend: Correction backend for the raw chain.
        require_dark: ``False`` only for a placeholder profile (see ``save_profile``): the
            white is averaged as it comes, dark offset included.

    Raises:
        ValueError: a raw white reference without the dark step (would leave the dark offset
            in the white and bias every factor), or a dark cube of another shape.
    """
    frames = frames or slice(None)
    indices = range(*frames.indices(source.n_frames))
    if len(indices) == 0:
        raise ValueError("no frames selected")
    steps = {name: info['on'] for name, info in plan.steps.items()}
    total = None
    if plan.stage == 'raw':
        if not steps.get('dark') and require_dark:
            raise ValueError("the white reference must be dark-corrected like the data: configure "
                             "corrections.dark (auto or a file) for this recording")
        runtime = PreprocessingRuntime(backend)
        dark = plan.dark_processor(bool(steps.get('dark')))
        smile = plan.smile_model if steps.get('smile') else None
        for i in indices:                         # one frame at a time: no full stack in memory
            binned, _, _ = runtime.process_binned(np.ascontiguousarray(source.frames[i]), dark, plan.binner,
                                                  calibration=smile, inverted=bool(steps.get('flip')))
            total = binned.astype(np.float64) if total is None else total + binned
        mean = total / len(indices)
        wavelengths = plan.wavelengths
    else:
        mean = np.asarray(source.frames[frames], dtype=np.float64).mean(axis=0)
        if dark_cube is None and not require_dark:
            dark_cube = False
            steps['dark'] = 'not subtracted (placeholder)'
        elif dark_cube is None:
            raise ValueError("a binned white cube still contains the dark offset unless it was "
                             "dark-corrected before export: give its dark cube (same bands), or "
                             "state that it is already dark-corrected")
        if dark_cube is False:
            steps.setdefault('dark', None)
            if steps['dark'] != 'not subtracted (placeholder)':
                steps['dark'] = 'stated: already dark-corrected before export'
        else:
            if dark_cube.frame_shape != source.frame_shape:
                raise ValueError(f"dark cube frames {dark_cube.frame_shape} differ from white {source.frame_shape}")
            mean = mean - np.asarray(dark_cube.frames, dtype=np.float64).mean(axis=0)
            steps['dark'] = True
        wavelengths = source.wavelengths
    if wavelengths is None:
        raise ValueError("the white reference has no wavelength axis")
    return WhiteReference(mean, np.asarray(wavelengths, np.float64), len(indices), steps, source.name)


class SlitCropFrames:
    """Raw frames cropped to some slit rows, put back into full-height sensor frames.

    A white reference saved as a slit crop (e.g. only the rows of a Spectralon disc) cannot meet
    the full-frame dark and Smile model. Placed back at its sensor rows (zeros elsewhere), it goes
    through the unchanged chain; dark, flip and the horizontal-only Smile act per row, so the
    crop rows come out exactly as in the full frame (hot-pixel repair at the two crop edge rows
    may read padding - leave those rows out of the ROI, see :func:`crop_rows_after_chain`).
    """

    def __init__(self, frames, row_offset: int, full_rows: int) -> None:
        self._frames = frames
        n, rows, width = frames.shape
        if not (0 <= row_offset and row_offset + rows <= full_rows):
            raise ValueError(f"crop rows {row_offset}:{row_offset + rows} do not fit a {full_rows}-row frame")
        self.row_offset, self.rows = int(row_offset), int(rows)
        self.shape = (n, int(full_rows), width)
        self.dtype = frames.dtype
        self.ndim = 3

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, i):
        if not isinstance(i, (int, np.integer)):
            raise TypeError("SlitCropFrames supports single frames")
        out = np.zeros(self.shape[1:], self.dtype)
        out[self.row_offset:self.row_offset + self.rows] = self._frames[int(i)]
        return out


def embed_slit_crop(source, row_offset: int, full_rows: int):
    """``FrameSource`` of full-height frames made from a slit-cropped raw white reference."""
    from sources import FrameSource
    metadata = dict(source.metadata, frame_height=int(full_rows), slit_crop_offset=int(row_offset),
                    slit_crop_rows=int(source.frame_shape[0]))
    return FrameSource(source.path, SlitCropFrames(source.frames, row_offset, full_rows), source.stage,
                       source.wavelengths, metadata)


def crop_rows_after_chain(row_offset: int, rows: int, full_rows: int, flipped: bool, margin: int = 1):
    """Processed slit rows ``(first, last + 1)`` of the crop: after the flip, minus ``margin`` per edge."""
    first, last = row_offset + margin, row_offset + rows - margin      # raw sensor rows
    if flipped:                                                        # raw_inverted: row y -> full - 1 - y
        first, last = full_rows - last, full_rows - first
    return first, last


def find_slit_crop(white_frames, recording) -> tuple[int, int] | None:
    """``(frame, row)`` of ``recording`` (HDF5) where the white crop's first frame starts, by exact match."""
    import zlib
    import h5py
    from concurrent.futures import ThreadPoolExecutor
    first = np.asarray(white_frames[0])
    signature = first[0]
    with h5py.File(recording, 'r') as f, ThreadPoolExecutor(8) as pool:
        ds = f['spectrogram']
        n, height, width = ds.shape
        direct = ds.chunks == (1, height, width) and ds.compression == 'gzip' and not ds.shuffle

        def match(item):
            i, frame = item
            if not isinstance(frame, np.ndarray):
                frame = np.frombuffer(zlib.decompress(frame), ds.dtype).reshape(height, width)
            for row in np.flatnonzero((frame == signature).all(axis=1)):
                if row + first.shape[0] <= height and np.array_equal(frame[row:row + first.shape[0]], first):
                    return i, int(row)
            return None

        for start in range(0, n, 64):
            batch = [(i, ds.id.read_direct_chunk((i, 0, 0))[1] if direct else ds[i])
                     for i in range(start, min(n, start + 64))]
            for hit in pool.map(match, batch):
                if hit:
                    return hit
    return None


def reference_reflectance(table, wavelengths) -> np.ndarray:
    """Target reflectance at the band centres, as hsi: ``CubicSpline`` of the table (nm, R)."""
    import pandas as pd
    from scipy.interpolate import CubicSpline
    data = pd.read_excel(table, header=None).to_numpy() if str(table).lower().endswith(('.xlsx', '.xls')) \
        else np.loadtxt(table, delimiter=',')
    wl = np.asarray(wavelengths, np.float64)
    if wl.min() < data[:, 0].min() or wl.max() > data[:, 0].max():
        raise ValueError(f"{Path(table).name} covers {data[0, 0]:.0f}-{data[-1, 0]:.0f} nm, "
                         f"bands are {wl.min():.1f}-{wl.max():.1f} nm")
    return CubicSpline(data[:, 0], data[:, 1])(wl).astype(np.float32)


def factors_from_white(white: np.ndarray, reflectance: np.ndarray, kind: str = 'band',
                       rows: tuple[int, int] | None = None, avg_window: int = 20) -> np.ndarray:
    """hsi factors ``R_ref / white``.

    Args:
        white: Mean white ``(slit rows, bands)`` after dark (and the other steps).
        reflectance: ``(bands,)`` target reflectance.
        kind: ``'band'`` - one factor per band from the mean over the ROI rows, ``(bands,)``;
            ``'row'`` - one per slit row and band, ``(rows, bands)``: each row uses the mean of
            ``avg_window`` ROI rows nearest to it (rows outside the ROI take the nearest ROI
            rows), as hsi ``rowwise_calculation_heavy_logic``.
        rows: ROI slit rows ``(first, last + 1)`` where the target is; default all.
        avg_window: Row window for ``'row'`` (hsi default in the window: 20).
    """
    n_rows = white.shape[0]
    r0, r1 = rows or (0, n_rows)
    if not 0 <= r0 < r1 <= n_rows:
        raise ValueError(f"rows {r0}:{r1} outside 0:{n_rows}")
    roi = white[r0:r1]
    fallback = np.nanmedian(roi, axis=0)
    fallback = np.where(np.isfinite(fallback) & (fallback > 0), fallback, 1.0)
    if kind == 'band':
        mean = roi.mean(axis=0)
        mean = np.where(np.isfinite(mean) & (mean > 0), mean, fallback)
        return (reflectance / mean).astype(np.float32)
    if kind != 'row':
        raise ValueError("kind must be 'band' or 'row'")
    roi_rows = np.arange(r0, r1)
    factors = np.empty((n_rows, white.shape[1]), np.float32)
    for y in range(n_rows):
        y_ref = int(np.clip(y, r0, r1 - 1))
        idx = int(np.argmin(np.abs(roi_rows - y_ref)))
        left = max(0, idx - avg_window // 2)
        right = min(len(roi_rows), left + avg_window)
        left = max(0, right - avg_window) if right - left < avg_window else left
        mean = white[roi_rows[left:right]].mean(axis=0)
        mean = np.where(np.isfinite(mean) & (mean > 0), mean, fallback)
        factors[y] = reflectance / mean
    return factors


def save_profile(folder, data_names, factors: np.ndarray, wavelengths, info: dict | None = None) -> list[Path]:
    """``<folder>/<name>.npy`` + ``.wavelength.npy`` per data source name (hsi format).

    ``info`` is written as ``<name>.json`` beside them: where the factors came from, and
    ``"placeholder": true`` for factors that only stand in until a real white reference exists
    (the simulator then marks the step as placeholder everywhere it shows it).
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in data_names:
        path = Path(save_factor_profile(folder / f"{name}.npy", factors, wavelengths))
        record = dict(info or {}, created=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
                      data_source=name, factor_shape=list(np.shape(factors)))
        path.with_suffix('.json').write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
        paths.append(path)
    return paths


def profile_info(npy_path) -> dict:
    """The ``<name>.json`` written by :func:`save_profile` (``{}`` if absent or unreadable)."""
    path = Path(npy_path).with_suffix('.json')
    try:
        return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    except (OSError, ValueError):
        return {}


def quality(white: WhiteReference, factors: np.ndarray, reflectance: np.ndarray,
            full_scale: float | None = None, rows: tuple[int, int] | None = None) -> dict:
    """Numbers to judge the reference: low signal, saturation risk, spread across the slit."""
    r0, r1 = rows or (0, white.mean.shape[0])
    roi = white.mean[r0:r1]
    band_mean = roi.mean(axis=0)
    row_mean = roi.mean(axis=1)
    weak = np.flatnonzero(band_mean < 0.02 * band_mean.max())
    report = {
        'frames': white.n_frames, 'rows': [int(r0), int(r1)], 'bands': int(roi.shape[1]),
        'wavelength_nm': [float(white.wavelengths[0]), float(white.wavelengths[-1])],
        'white_counts_min_max': [float(band_mean.min()), float(band_mean.max())],
        'row_spread_pct': float(100 * row_mean.std() / row_mean.mean()) if row_mean.mean() else None,
        'reflectance_min_max': [float(reflectance.min()), float(reflectance.max())],
        'factor_min_max': [float(np.min(factors)), float(np.max(factors))],
        'weak_bands_nm': [float(white.wavelengths[i]) for i in weak],
        'steps': white.steps,
    }
    if full_scale:
        report['white_peak_pct_of_full_scale'] = float(100 * roi.max() / full_scale)
    return report


# ------------------------------------------------- white reference from a folder (config) ---

def bright_span(values, fraction: float = 0.5, margin: int = 0) -> tuple[int, int] | None:
    """Longest run ``(first, last + 1)`` of ``values`` above ``lo + fraction * (hi - lo)``.

    ``None`` when there is no clear target (contrast under 10 % of the peak): the white then
    fills everything, e.g. a tile across the whole slit. ``margin`` is cut from both ends of the
    run (edge rows/frames are partly on the target).
    """
    v = np.asarray(values, np.float64)
    if v.size == 0:
        return None
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    if not np.isfinite(hi) or hi <= 0 or hi - lo < 0.1 * hi:
        return None
    mask = np.append(v >= lo + fraction * (hi - lo), False)
    best, start = (0, 0), None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    first, last = best
    if last - first > 2 * margin + 1:
        first, last = first + margin, last - margin
    return first, last


def _white_metadata(path: Path) -> dict:
    """Recorded camera keys of a white file (HDF5 attributes; ``{}`` for other formats)."""
    if path.suffix.lower() not in ('.h5', '.hdf5'):
        return {}
    import h5py
    try:
        with h5py.File(path, 'r') as f:
            meta = dict(f.attrs)
            for ds in f.values():
                if hasattr(ds, 'attrs'):
                    meta.update({k: v for k, v in ds.attrs.items() if k not in meta})
    except OSError:
        return {}
    return meta


def find_white(config, plan) -> tuple[Path | None, str]:
    """Newest file of ``white_reference.path`` recorded with the data's camera backend and bin.

    Returns:
        ``(path, '')``, or ``(None, why)`` when the folder has no white for these frames.

    Raises:
        FileNotFoundError: ``white_reference.path`` does not exist.
    """
    wr = config.white_reference
    root = config.resolve(wr.path)
    if root is None or not root.exists():
        raise FileNotFoundError(f"white_reference.path not found: {root}")
    files = [p for p in root.glob(wr.pattern or '*.h5') if p.is_file()] if root.is_dir() else [root]
    where = f"{root.name}/{wr.pattern or '*.h5'}" if root.is_dir() else root.name
    if not files:
        return None, f"no white reference in {where}"
    from sim_config import camera_profile_name
    skipped = []
    for path in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True):
        meta = _white_metadata(path)
        white_id = camera_profile_name(meta.get('camera_model'), meta.get('camera_serial'))
        if plan.camera_id and white_id and white_id != plan.camera_id:
            skipped.append(f"{path.name}: camera {white_id}")
            continue
        if plan.stage == 'raw':
            backend, camera_bin = meta.get('camera_backend'), meta.get('camera_bin')
            if ((backend is not None and str(backend).lower() != str(plan.camera_backend).lower())
                    or (camera_bin is not None and int(camera_bin) != int(plan.camera_bin))):
                skipped.append(f"{path.name}: {backend} bin {camera_bin}")
                continue
        return path, ''
    return None, (f"no white reference for {plan.camera_backend} bin {plan.camera_bin} in {where} "
                  f"({'; '.join(skipped)})")


def _plain(value):
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return value


def _clean_steps(steps: dict) -> dict:
    return {k: (v if isinstance(v, str) or v is None else bool(v)) for k, v in steps.items()}


def ensure_white_profile(config, plan, name: str) -> tuple[Path | None, str]:
    """Factors for data source ``name`` from the configured white reference (``white_reference``).

    The newest matching white (:func:`find_white`) goes through the data's chain exactly as
    ``tools/make_reflectance.py`` does it: its own dark (by its exposure/gain), flip, Smile,
    binning; frames and slit rows with the target are found by brightness unless given; a white
    of another exposure than the data is scaled to the data's exposure (counts are linear in it).
    The result is cached as ``<reflectance_dir>/from_white/<name>.npy`` (+ ``.json`` with the
    signature) and rebuilt only when the white, the table, the settings or the data's bands change.

    Returns:
        ``(profile path, note)``, or ``(None, why)`` when no white fits these frames.

    Raises:
        ValueError, OSError: the white or the table cannot give factors.
    """
    from dataclasses import replace
    from sim_config import build_plan, reflectance_folder
    from sources import open_sources

    white_path, why = find_white(config, plan)
    if white_path is None:
        return None, why
    wr = config.white_reference
    table = config.resolve(wr.table)
    if table is None or not table.is_file():
        raise FileNotFoundError(f"white_reference.table not found: {table}")
    out = reflectance_folder(config) / 'from_white' / f"{name}.npy"
    st, tst = white_path.stat(), table.stat()
    wl = plan.wavelengths
    signature = {
        'white': str(white_path), 'white_size': int(st.st_size), 'white_mtime_ns': int(st.st_mtime_ns),
        'table': str(table), 'table_mtime_ns': int(tst.st_mtime_ns), 'kind': wr.kind,
        'rows': _plain(wr.rows), 'frames': _plain(wr.frames), 'avg_window': int(wr.avg_window),
        'slit_offset': wr.slit_offset, 'frame_shape': [int(v) for v in plan.frame_shape],
        'bands': int(plan.n_columns),
        'wavelength_nm': [round(float(wl[0]), 4), round(float(wl[-1]), 4)] if wl is not None else None,
        'exposure_ms': plan.exposure_ms, 'gain': plan.gain, 'camera_id': plan.camera_id,
        'steps': {k: bool(plan.steps.get(k, {}).get('on')) for k in ('dark', 'flip', 'smile')},
    }
    if out.is_file() and profile_info(out).get('signature') == signature:
        return out, f"from white {white_path.name} (cached)"

    logger.info("Building reflectance factors for %s from white reference %s", name, white_path)
    src = open_sources(white_path, stage=config.source.stage, scan_axis=config.source.envi_scan_axis,
                       dataset=config.source.dataset, cache='ram')[0]
    crop = None
    if src.stage == 'raw' and plan.stage == 'raw' and src.frame_shape[0] != plan.frame_shape[0]:
        if wr.slit_offset is None:
            raise ValueError(f"{white_path.name} has {src.frame_shape[0]} slit rows, the frames "
                             f"{plan.frame_shape[0]}: a slit crop - set white_reference.slit_offset")
        crop = (int(wr.slit_offset), int(src.frame_shape[0]), int(plan.frame_shape[0]))
        src = embed_slit_crop(src, crop[0], crop[2])
    white_config = replace(config, corrections=replace(config.corrections, reflectance='none'),
                           white_reference=replace(wr, path=None))
    white_plan = build_plan(white_config, src)

    if wr.frames == 'auto':
        means = [float(np.asarray(src.frames[i])[::4, ::4].mean()) for i in range(src.n_frames)]
        span = bright_span(means, margin=1)
        frames = slice(*span) if span else None
    else:
        frames = slice(*wr.frames) if wr.frames else None
    white = white_mean(src, white_plan, frames)

    lo, hi = 0, white.mean.shape[0]
    if crop:
        lo, hi = crop_rows_after_chain(crop[0], crop[1], crop[2], bool(white_plan.steps['flip']['on']))
    if wr.rows == 'auto':
        span = bright_span(white.mean[lo:hi].mean(axis=1), margin=2)
        rows = (lo + span[0], lo + span[1]) if span else ((lo, hi) if crop else None)
    else:
        rows = tuple(int(v) for v in wr.rows) if wr.rows else ((lo, hi) if crop else None)

    reflectance = reference_reflectance(table, white.wavelengths)
    factors = factors_from_white(white.mean, reflectance, wr.kind, rows, wr.avg_window)
    notes = []
    scale = 1.0
    if white_plan.exposure_ms and plan.exposure_ms and abs(white_plan.exposure_ms - plan.exposure_ms) > 1e-9:
        scale = float(white_plan.exposure_ms) / float(plan.exposure_ms)
        factors = (factors * scale).astype(np.float32)
        notes.append(f"white {white_plan.exposure_ms:g} ms scaled to {plan.exposure_ms:g} ms")
    if white_plan.gain is not None and plan.gain is not None and abs(white_plan.gain - plan.gain) > 1e-9:
        notes.append(f"WARNING gain {white_plan.gain:g} vs data {plan.gain:g} (not scaled)")
        logger.warning("White %s gain %s differs from the data's %s; factors are not scaled for gain",
                       white_path.name, white_plan.gain, plan.gain)
    full_scale = src.metadata.get('full_scale') or src.metadata.get('data full scale')
    report = quality(white, factors, reflectance, float(full_scale) if full_scale else None, rows)
    report['steps'] = _clean_steps(report['steps'])
    if report['weak_bands_nm']:
        logger.warning("White %s: %d bands under 2 %% of the peak signal; their factors amplify noise",
                       white_path.name, len(report['weak_bands_nm']))
    span = frames.indices(src.n_frames)[:2] if frames else None
    info = {'placeholder': False, 'white': white_path.name, 'white_reference': str(white_path),
            'table': table.name, 'kind': wr.kind, 'rows': list(rows) if rows else 'all',
            'frames': list(span) if span else 'all', 'frames_averaged': white.n_frames,
            'steps': _clean_steps(white.steps), 'exposure_scale': scale, 'notes': notes,
            'quality': report, 'signature': signature,
            # bound like a drawn calibration: the data's bin/gain; already scaled to its exposure
            'camera_bin': plan.camera_bin, 'gain': plan.gain, 'exposure_ms': plan.exposure_ms,
            'camera_id': plan.camera_id}
    if crop:
        info['slit_crop'] = {'raw_rows': [crop[0], crop[0] + crop[1]], 'sensor_rows': crop[2]}
    save_profile(out.parent, [name], factors, white.wavelengths, info)
    logger.info("Wrote %s (%s, rows %s, %d frames%s)", out, wr.kind, info['rows'], white.n_frames,
                ("; " + "; ".join(notes)) if notes else '')
    return out, f"from white {white_path.name}" + (f" ({'; '.join(notes)})" if notes else '')


# ------------------------------------------------------ white reference drawn on the belt map ---

def reference_values(reference, wavelengths) -> np.ndarray:
    """Target reflectance at the band centres: a table path (nm, R) or a constant (0..1)."""
    if isinstance(reference, (int, float)):
        if not 0 < float(reference) <= 1.5:
            raise ValueError(f"reflectance {reference} must be in (0, 1.5] (a fraction, 0.99 = 99 %)")
        return np.full(len(wavelengths), float(reference), np.float32)
    return reference_reflectance(reference, wavelengths)


def factors_from_lines(lines: np.ndarray, rows: tuple[int, int], reflectance: np.ndarray, wavelengths,
                       kind: str = 'band', avg_window: int = 20, steps: dict | None = None,
                       source: str = 'drawn on RAW'):
    """Factors from binned lines ``(n, slit, bands)`` (before reflectance) with the target in ``rows``.

    Returns:
        ``(factors, WhiteReference)``; the white is the mean over the lines, all slit rows.

    Raises:
        ValueError: no lines, rows outside the slit, or a target darker than its surroundings.
    """
    lines = np.asarray(lines)
    if lines.ndim != 3 or lines.shape[0] == 0:
        raise ValueError("no captured lines inside the drawn area")
    mean = lines.mean(axis=0, dtype=np.float64)
    r0, r1 = int(rows[0]), int(rows[1])
    if not 0 <= r0 < r1 <= mean.shape[0]:
        raise ValueError(f"rows {r0}:{r1} outside the slit 0:{mean.shape[0]}")
    if not np.isfinite(mean[r0:r1]).all() or mean[r0:r1].mean() <= 0:
        raise ValueError("the drawn area has no signal (dark step on? target under the slit?)")
    white = WhiteReference(mean, np.asarray(wavelengths, np.float64), int(lines.shape[0]),
                           dict(steps or {}), source)
    return factors_from_white(mean, reflectance, kind, (r0, r1), avg_window), white


def backup_profile(path) -> list[Path]:
    """Move ``<name>.npy`` (+ ``.wavelength.npy``, ``.json``) to ``<folder>/history/<name>_<time>.*``."""
    path = Path(path)
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    history = path.parent / 'history'
    moved = []
    for suffix in ('.npy', '.wavelength.npy', '.json'):
        old = path.with_name(path.stem + suffix)
        if old.is_file():
            history.mkdir(parents=True, exist_ok=True)
            target = history / f"{path.stem}_{stamp}{suffix}"
            old.replace(target)
            moved.append(target)
    return moved
