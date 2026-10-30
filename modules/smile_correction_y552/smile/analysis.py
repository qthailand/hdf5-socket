# smile/analysis.py
"""Full multi-line smile analysis pipeline."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .core import (
    Line,
    _traces_rms,
    auto_sat_level,
    detect_lines,
    estimate_noise,
    trace_line,
)
from .surface import Surface2D, cross_validate_lines, fit_surface_2d
from .calibration import correct_smile, crop_margin

__all__ = ["SmileAnalysis", "analyze_smile", "verify_on_frame"]


def _default_reference_row(height: int) -> int:
    """Return the zero-based reference row from the agreed center rule."""
    if height <= 0:
        raise ValueError("height must be positive")
    half_height = height / 2.0
    center = int(np.floor(half_height))
    if not half_height.is_integer():
        center -= 1
    return center


@dataclass
class SmileAnalysis:
    """Complete results of a smile correction analysis.

    Attributes:
        frame_orig: Original calibration image.
        frame_corr: Corrected image (zero-valued borders).
        all_lines: Every detected emission line.
        used_lines: Lines used in the surface fit.
        traces_orig: ``{column -> trace_array}`` for original image.
        traces_corr: ``{column -> trace_array}`` for corrected image.
        refs: ``{column -> sub-pixel reference}`` for each line.
        surface: Fitted 2-D displacement surface.
        fit_info: Surface fitting diagnostics.
        meas_y: Row coordinates of measurement points.
        meas_x: Sub-pixel reference column coordinates (adjusted).
        meas_v: Measured shift values.
        meas_id: Index into ``used_lines`` for each point.
        residuals: ``meas_v - surface(meas_y, meas_x)``.
        margin: ``(left, right)`` border pixels to crop.
        rms_before: Straightness RMS before correction.
        rms_after: Straightness RMS after correction (in-sample).
        cv_rms: Leave-one-line-out cross-validated RMS.
        interp: Interpolation method used.
        sat_level: Effective saturation threshold.
        noise: Estimated per-pixel noise level.
    """

    frame_orig: np.ndarray
    frame_corr: np.ndarray
    all_lines: List[Line]
    used_lines: List[Line]
    traces_orig: Dict[float, np.ndarray]
    traces_corr: Dict[float, np.ndarray]
    refs: Dict[float, float]
    surface: Surface2D
    fit_info: dict
    meas_y: np.ndarray
    meas_x: np.ndarray
    meas_v: np.ndarray
    meas_id: np.ndarray
    residuals: np.ndarray
    margin: Tuple[int, int]
    rms_before: float
    rms_after: float
    cv_rms: float
    interp: str
    sat_level: float
    noise: float
    y_ref: float


def analyze_smile(
    frame: np.ndarray,
    sat_level: Optional[float] = None,
    deg_y: int = 2,
    deg_x: int = 1,
    interp: str = "cubic",
    snr_min: float = 5.0,
    n_ref_iter: int = 2,
    y_ref: Optional[float] = None,
    verbose: bool = False,
) -> SmileAnalysis:
    """Run the full multi-line smile analysis pipeline.

    Args:
        frame: Calibration image ``(H, W)``.
        sat_level: Saturation threshold.  Inferred if ``None`` or <= 0.
        deg_y: Polynomial degree in the spatial (row) direction.
        deg_x: Polynomial degree in the spectral (column) direction.
        interp: Interpolation method.
        snr_min: Minimum peak-to-noise ratio for tracing.
        n_ref_iter: Number of reference-offset iterations.
        verbose: Print per-line report to stdout.
        y_ref: Row used as zero-displacement reference. Defaults to the
            geometric image center, with the lower-index center row selected.

    Returns:
        A :class:`SmileAnalysis` with all diagnostics.

    Raises:
        ValueError: If no usable emission lines are found.
    """
    frame = np.asarray(frame, dtype=np.float32)
    n_rows = frame.shape[0]
    if sat_level is None or sat_level <= 0:
        sat_level = auto_sat_level(frame)
    noise = estimate_noise(frame)

    # Step 1: Detect emission lines.
    lines = detect_lines(frame, sat_level)
    if not lines:
        raise ValueError(
            "No emission lines detected. Check image brightness."
        )

    # Step 2: Trace every line through all rows.
    traces_orig: Dict[float, np.ndarray] = {}
    for ln in lines:
        halfwin = max(3, int(round(1.2 * ln.fwhm)))
        traces_orig[ln.column] = trace_line(
            frame, ln.column, halfwin, sat_level, noise, snr_min
        )

    # Step 3: Collect measurement points.
    #   Reference = median of the trace (sub-pixel).
    ys, xs, vs, ids, used = [], [], [], [], []
    refs: Dict[float, float] = {}
    for ln in lines:
        if ln.saturated:
            continue
        centers = traces_orig[ln.column]
        rows = np.where(np.isfinite(centers))[0]
        if rows.size < 20 or rows.size / n_rows < 0.02:
            continue
        ref = float(np.median(centers[rows]))
        refs[ln.column] = ref
        ys.append(rows.astype(float))
        xs.append(np.full(rows.size, ref))
        vs.append(centers[rows] - ref)
        ids.append(np.full(rows.size, len(used)))
        used.append(ln)

    if not ys:
        raise ValueError(
            "All lines were filtered out "
            "(saturated, low coverage, or low SNR)."
        )

    meas_y = np.concatenate(ys)
    meas_x = np.concatenate(xs)
    meas_v = np.concatenate(vs)
    meas_id = np.concatenate(ids)

    # Preserve original x coordinates for cross-validation.
    meas_x_orig = meas_x.copy()

    # Validate deg_x against available lines.
    n_ux = np.unique(np.round(meas_x, 4)).size
    deg_x_eff = min(deg_x, n_ux - 1)
    if deg_x_eff < deg_x:
        warnings.warn(
            f"Only {n_ux} unique line positions; "
            f"reducing deg_x from {deg_x} to {deg_x_eff}."
        )

    # Step 4: Iterative reference adjustment + robust surface fit.
    for _ in range(max(0, n_ref_iter)):
        surf_tmp, _ = fit_surface_2d(
            meas_y, meas_x, meas_v, deg_y, deg_x_eff
        )
        res_tmp = meas_v - surf_tmp(meas_y, meas_x)
        for k, ln in enumerate(used):
            mk = meas_id == k
            off = float(np.median(res_tmp[mk]))
            meas_v[mk] -= off
            meas_x[mk] += off
            refs[ln.column] += off

    surface, fit_info = fit_surface_2d(
        meas_y, meas_x, meas_v, deg_y, deg_x_eff
    )
    residuals = meas_v - surface(meas_y, meas_x)
    cv_rms = cross_validate_lines(
        meas_y, meas_x_orig, meas_v, meas_id, deg_y, deg_x_eff
    )

    # Anchor the fitted displacement at the requested physical reference row.
    # The fit range must include that row so Surface2D does not clamp it.
    if y_ref is None:
        y_ref = float(_default_reference_row(n_rows))
    if not np.isfinite(y_ref) or not 0 <= y_ref < n_rows:
        raise ValueError(f"y_ref must be within image rows [0, {n_rows - 1}]")
    x_grid = np.arange(frame.shape[1], dtype=float)
    ref_shift = surface(np.full_like(x_grid, y_ref), x_grid)
    yy, xx = np.meshgrid(
        np.linspace(0, n_rows - 1, max(surface.deg_y + 2, 4)),
        np.linspace(0, frame.shape[1] - 1, max(surface.deg_x + 2, 4)),
        indexing="ij",
    )
    vals = surface(yy, xx) - surface(np.full_like(xx, y_ref), xx)
    surface, _ = fit_surface_2d(
        yy.ravel(), xx.ravel(), vals.ravel(), surface.deg_y, surface.deg_x,
        n_iter=0,
    )

    # Step 5: Remap (zero-valued borders).
    frame_corr, margin, _, _ = correct_smile(frame, surface, interp)

    # Step 6: Re-trace on the corrected image.
    traces_corr: Dict[float, np.ndarray] = {}
    for ln in used:
        halfwin = max(3, int(round(1.2 * ln.fwhm)))
        traces_corr[ln.column] = trace_line(
            frame_corr, refs[ln.column], halfwin, sat_level, noise, snr_min
        )

    # Step 7: Straightness metrics (reuse existing traces).
    rms_before = _traces_rms([traces_orig[ln.column] for ln in used])
    rms_after = _traces_rms([traces_corr[ln.column] for ln in used])

    if verbose:
        print(
            f"Lines used: {len(used)}/{len(lines)}  "
            f"deg={deg_y}x{deg_x_eff}  interp={interp}"
        )
        print(
            f"Straightness RMS: {rms_before:.4f} -> {rms_after:.4f} px "
            f"(in-sample)   LOO-CV: {cv_rms:.4f} px"
        )
        print(f"Border margin: left={margin[0]} right={margin[1]} px")

    return SmileAnalysis(
        frame_orig=frame,
        frame_corr=frame_corr,
        all_lines=lines,
        used_lines=used,
        traces_orig=traces_orig,
        traces_corr=traces_corr,
        refs=refs,
        surface=surface,
        fit_info=fit_info,
        meas_y=meas_y,
        meas_x=meas_x,
        meas_v=meas_v,
        meas_id=meas_id,
        residuals=residuals,
        margin=margin,
        rms_before=rms_before,
        rms_after=rms_after,
        cv_rms=cv_rms,
        interp=interp,
        sat_level=float(sat_level),
        noise=float(noise),
        y_ref=float(y_ref),
    )


def verify_on_frame(
    analysis: SmileAnalysis, other_frame: np.ndarray
) -> float:
    """Measure straightness RMS after correction on a *different* frame.

    Args:
        analysis: Completed :class:`SmileAnalysis`.
        other_frame: Second calibration image with the same shape.

    Returns:
        Straightness RMS in pixels after applying the correction.

    Raises:
        ValueError: If image shapes do not match.
    """
    other = np.asarray(other_frame, dtype=np.float32)
    if other.shape != analysis.frame_orig.shape:
        raise ValueError(
            f"Shape mismatch: {other.shape} vs {analysis.frame_orig.shape}"
        )
    corr, _, _, _ = correct_smile(other, analysis.surface, analysis.interp)
    noise = estimate_noise(other)
    traces = []
    for ln in analysis.used_lines:
        halfwin = max(3, int(round(1.2 * ln.fwhm)))
        traces.append(
            trace_line(
                corr, analysis.refs[ln.column], halfwin,
                analysis.sat_level, noise,
            )
        )
    return _traces_rms(traces)
