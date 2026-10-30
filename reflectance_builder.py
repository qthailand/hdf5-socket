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
