#!/usr/bin/env python3
"""Smile Correction Module for Imaging Spectrometers (v2.2).

Performs multi-line smile correction on spectral calibration frames
(e.g., Hg-Ar lamp images) by tracing emission lines, fitting a robust
2-D displacement surface ``shift(y, x)``, and remapping the image.

Changelog (v2.1 -> v2.2):
    - A   Added ``SmileCorrector``: sklearn-style ``fit`` / ``transform`` /
          ``fit_transform`` / ``inverse_transform`` / ``save`` / ``load`` API.
    - B   ``SmileCorrectionWidget`` uses ``SmileCorrector`` as backend.

Changelog (v2.0 -> v2.1):
    - C1  ``fit_surface_2d``: initial fit before IRLS loop (fixes NameError
          when ``n_iter=0``).
    - C2  ``_centroid_row``: clearer comment for ``min_peak`` threshold.
    - C3  ``analyze_smile``: preserve ``meas_x_orig`` for cross-validation
          to avoid contamination from iterative reference adjustment.
    - C4  ``SmileCalibration.summary``: display ``n/a`` for non-finite
          ``cv_rms`` (e.g., loaded from format v1).
    - I1  ``detect_lines``: subsample rows for percentile calculation.
    - I2  ``estimate_noise``: subsample rows to reduce memory allocation.
    - I3  ``build_maps``: ``map_y`` built via ``np.repeat`` (no ``meshgrid``).
    - I4  ``calibration_from_analysis``: removed unused ``deg_y``/``deg_x``
          parameters.
    - I5  ``trace_line``: emit ``warnings.warn`` when all rows in the
          initial strip are saturated.
    - I6  ``_plot_traces``: corrected-plot uses its own reference (not the
          original one).
    - T   Threading: lambda slot replaced with bound method; ``worker.wait()``
          before clearing ``self._worker``.
    - S   ``set_frame`` clears ``_result`` and disables save buttons.
    - D   Docstrings follow Google style / PEP 257; type hints completed;
          all exception messages in English; ``__all__`` added.

Usage:
    1. Standalone GUI::

           python smile_module.py [image_path]

       Falls back to CLI mode when PyQt5 is not installed.

    2. Embed as a PyQt5 widget::

           from smile_module import SmileCorrectionWidget
           widget = SmileCorrectionWidget(parent)
           layout.addWidget(widget)
           widget.analysisComplete.connect(handler)

    3. sklearn-style (no Qt dependency)::

           from smile_module import SmileCorrector
           corrector = SmileCorrector(deg_y=2, deg_x=1)
           corrector.fit(hgar_frame)
           corrector.save("model.npz")
           corrected = corrector.transform(new_frame, crop=True)

    4. Core-only functions::

           from smile_module import analyze_smile
           result = analyze_smile(frame)
           corrected = result.frame_corr

Dependencies:
    numpy, opencv-python, scipy
    (Widget requires PyQt5 and matplotlib)
"""

from __future__ import annotations
from matplotlib.figure import Figure

import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import find_peaks

__version__ = "2.2.0"

__all__ = [
    # Core
    "load_frame",
    "save_frame",
    "robust_sigma",
    "auto_sat_level",
    "estimate_noise",
    "Line",
    "detect_lines",
    "trace_line",
    "Surface2D",
    "fit_surface_2d",
    "cross_validate_lines",
    "correct_smile",
    "crop_margin",
    "straightness_rms",
    # Analysis
    "SmileAnalysis",
    "analyze_smile",
    "verify_on_frame",
    # Calibration
    "SmileCalibration",
    "calibration_from_analysis",
    "save_calibration",
    "load_calibration",
    # sklearn-style API
    "SmileCorrector",
    # Qt widget (optional)
    "PlotCanvas",
    "SmileCorrectionWidget",
]


# =====================================================================
#  SECTION 1 -- Core  (pure NumPy / OpenCV / SciPy, no Qt dependency)
# =====================================================================

# Interpolation name -> (cv2 flag, taps before floor, taps after floor).
_INTERP = {
    "linear": (cv2.INTER_LINEAR, 0, 1),
    "cubic": (cv2.INTER_CUBIC, 1, 2),
    "lanczos": (cv2.INTER_LANCZOS4, 3, 4),
}


def load_frame(path: str) -> np.ndarray:
    """Load a grayscale image as ``float32``.

    Supports 8-bit and 16-bit sources.  Colour images (including BGRA)
    are converted to grayscale automatically.

    Args:
        path: Filesystem path to the image file.

    Returns:
        Grayscale image array of shape ``(H, W)``, dtype ``float32``.

    Raises:
        FileNotFoundError: If the image cannot be read from *path*.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        img = cv2.cvtColor(img, code)
    return img.astype(np.float32)


def save_frame(path: str, img: np.ndarray, bits: Optional[int] = None) -> None:
    """Write *img* to disk, preserving bit depth when possible.

    Args:
        path: Destination file path.
        img: Image array to write.
        bits: Force 8 or 16 bit output.  ``None`` selects automatically
            based on the maximum pixel value.
    """
    img = np.nan_to_num(np.asarray(img, np.float32))
    if bits is None:
        bits = 8 if float(img.max()) <= 255 else 16
    if bits == 8:
        out = np.clip(img, 0, 255).astype(np.uint8)
    else:
        out = np.clip(img, 0, 65535).astype(np.uint16)
    cv2.imwrite(path, out)


def robust_sigma(x: np.ndarray) -> float:
    """Estimate standard deviation via the median absolute deviation.

    Args:
        x: Input data array.

    Returns:
        Robust standard deviation estimate (always > 0).
    """
    x = np.asarray(x, dtype=float)
    return 1.4826 * np.median(np.abs(x - np.median(x))) + 1e-9


def auto_sat_level(frame: np.ndarray) -> float:
    """Determine the sensor full-scale value from the image maximum.

    Tests common bit depths: 8, 10, 12, 14, and 16 bit.

    Note:
        A dark 16-bit image (max < 4095) may be mis-detected as 12-bit.
        Supply ``sat_level`` explicitly when the true value is known.
        If no standard level matches, returns ``inf`` (no saturation check).

    Args:
        frame: Input image array.

    Returns:
        Full-scale level (e.g. ``255.0``, ``4095.0``, ``65535.0``),
        or ``float('inf')`` if no standard level matches.
    """
    mx = float(np.max(frame))
    for level in (255.0, 1023.0, 4095.0, 16383.0, 65535.0):
        if mx <= level:
            return level
    return float("inf")


def estimate_noise(frame: np.ndarray) -> float:
    """Estimate per-pixel noise from adjacent-row differences.

    Uses the MAD of the row-to-row difference divided by ``sqrt(2)``,
    which is robust against emission lines (signal cancels in the diff).

    Args:
        frame: Input image ``(H, W)``.

    Returns:
        Estimated noise level (always > 0).
    """
    step = max(1, frame.shape[0] // 256)
    d = np.diff(frame[::step], axis=0).ravel()
    floor = 1e-4 * float(np.percentile(frame, 99.9))
    return max(robust_sigma(d) / np.sqrt(2.0), floor, 1e-9)


# -----------------------------------------------------------------
#  Line descriptor
# -----------------------------------------------------------------


@dataclass
class Line:
    """Describes a single detected emission line.

    Attributes:
        column: Spectral column index of the line centre (integer from
            ``find_peaks``).  Used as a dictionary key and initial trace
            position.  The sub-pixel reference is computed separately
            during :func:`analyze_smile`.
        fwhm: Full width at half maximum in pixels.
        height: Peak intensity above the local baseline.
        saturated: Whether the line reaches saturation in at least
            ``min_sat_rows`` rows.
    """

    column: float
    fwhm: float
    height: float
    saturated: bool


# -----------------------------------------------------------------
#  Multi-line detection
# -----------------------------------------------------------------


def _half_width(s: np.ndarray, p: int) -> float:
    """Compute the half-width at half-maximum for peak at index *p*.

    Args:
        s: 1-D spectrum array.
        p: Peak index.

    Returns:
        Half-width in pixels (minimum 1.0).
    """
    half = s[p] * 0.5
    left = p
    while left > 0 and s[left] > half:
        left -= 1
    right = p
    while right < len(s) - 1 and s[right] > half:
        right += 1
    return max(1.0, (right - left) / 2.0)


def detect_lines(
    frame: np.ndarray,
    sat_level: Optional[float] = None,
    sat_frac: float = 0.98,
    max_lines: int = 30,
    spec_percentile: float = 90.0,
    min_sat_rows: int = 3,
) -> List[Line]:
    """Detect emission lines in a spectral calibration frame.

    Lines are found on a percentile spectrum (robust to partial slit
    illumination), filtered for proximity, and saturation is assessed by
    counting full-scale rows (not single hot pixels).

    Args:
        frame: Grayscale calibration image ``(H, W)``.
        sat_level: Saturation threshold.  Inferred via
            :func:`auto_sat_level` if ``None`` or <= 0.
        sat_frac: Fraction of *sat_level* at which a pixel is saturated.
        max_lines: Maximum number of lines to return (brightest first).
        spec_percentile: Percentile used for the collapsed spectrum.
        min_sat_rows: Minimum number of saturated rows for a line to be
            flagged as saturated.

    Returns:
        List of :class:`Line` sorted by column position.
    """
    if sat_level is None or sat_level <= 0:
        sat_level = auto_sat_level(frame)

    n_rows, n_cols = frame.shape

    step = max(1, n_rows // 512)
    spec = np.percentile(frame[::step], spec_percentile, axis=0)

    noise = robust_sigma(spec - np.median(spec))
    s = np.maximum(spec - np.percentile(spec, 10), 0)
    if s.max() <= 0:
        return []

    peaks, _ = find_peaks(
        s, prominence=max(5.0 * noise, 0.01 * s.max()), distance=4
    )
    sat_thr = sat_frac * sat_level

    lines: List[Line] = []
    for p in peaks:
        lo = max(p - 2, 0)
        hi = min(p + 3, n_cols)
        n_sat = int(np.count_nonzero((frame[:, lo:hi] >= sat_thr).any(axis=1)))
        lines.append(
            Line(
                column=float(p),
                fwhm=2.0 * _half_width(s, p),
                height=float(s[p]),
                saturated=n_sat >= min_sat_rows,
            )
        )

    # Merge close pairs: keep only the brighter line to prevent
    # centroid blending between neighbours.
    lines.sort(key=lambda ln: ln.column)
    keep: List[Line] = []
    for ln in lines:
        if keep and (ln.column - keep[-1].column) < 1.2 * (ln.fwhm + keep[-1].fwhm):
            if ln.height > keep[-1].height:
                keep[-1] = ln
            continue
        keep.append(ln)

    # Saturated lines get lowest priority (preserving max_lines for usable ones).
    keep.sort(key=lambda ln: (ln.saturated, -ln.height))
    keep = keep[:max_lines]
    keep.sort(key=lambda ln: ln.column)
    return keep


# -----------------------------------------------------------------
#  Sub-pixel tracing (window-following)
# -----------------------------------------------------------------


def _centroid_row(
    row: np.ndarray,
    xc: float,
    halfwin: int,
    sat_thr: float,
    min_peak: float,
    n_iter: int = 3,
) -> Tuple[float, bool]:
    """Compute the centre-of-mass for a single image row.

    Iteratively re-centres the window until convergence (within 0.01 px)
    or *n_iter* iterations.

    Args:
        row: 1-D pixel row (spectrum).
        xc: Predicted line centre in column coordinates.
        halfwin: Half-width of the analysis window in pixels.
        sat_thr: Saturation threshold (absolute pixel value).
        min_peak: Minimum peak amplitude above the local baseline.
            This is ``snr_min * noise`` -- the line must be clearly
            distinguishable from background noise.
        n_iter: Maximum number of centroid iterations.

    Returns:
        Tuple ``(centre, valid)``.  *centre* is a sub-pixel position, or
        ``nan`` when *valid* is ``False`` (saturated, out of bounds, or
        below noise floor).
    """
    c = xc
    for _ in range(n_iter):
        lo = int(round(xc)) - halfwin
        hi = int(round(xc)) + halfwin + 1
        if lo < 0 or hi > len(row):
            return np.nan, False

        seg = row[lo:hi]
        if seg.max() >= sat_thr:
            return np.nan, False

        k = int(0.2 * len(seg))
        base = np.partition(seg, k)[k]
        w = np.maximum(seg - base, 0)
        total = w.sum()

        if total <= 0 or w.max() < min_peak:
            return np.nan, False

        c = float(np.sum(np.arange(lo, hi) * w) / total)
        if abs(c - xc) < 0.01:
            break
        xc = c
    return c, True


def trace_line(
    frame: np.ndarray,
    x0: float,
    halfwin: int,
    sat_level: Optional[float] = None,
    noise: Optional[float] = None,
    snr_min: float = 5.0,
    max_gap: int = 10,
    sat_frac: float = 0.98,
) -> np.ndarray:
    """Trace a single emission line through all spatial rows.

    Uses a window-following predictor with SNR gating, jump limiting,
    and gap termination.  Starts from the brightest non-saturated row.

    Args:
        frame: Calibration image ``(H, W)``.
        x0: Initial column position of the line.
        halfwin: Half-width of the centroid window in pixels.
        sat_level: Saturation threshold.  Inferred if ``None`` or <= 0.
        noise: Per-pixel noise level.  Estimated if ``None``.
        snr_min: Minimum peak-to-noise ratio for accepting a centroid.
        max_gap: Maximum consecutive failed rows before stopping.
        sat_frac: Fraction of *sat_level* for saturation detection.

    Returns:
        Array of shape ``(H,)`` with sub-pixel column centres;
        ``np.nan`` at rows where tracing fails.
    """
    if sat_level is None or sat_level <= 0:
        sat_level = auto_sat_level(frame)
    if noise is None:
        noise = estimate_noise(frame)
    sat_thr = sat_frac * sat_level
    min_peak = snr_min * noise
    max_jump = max(1.0, 0.5 * halfwin)

    n_rows, n_cols = frame.shape
    centers = np.full(n_rows, np.nan)

    col = int(round(x0))
    lo = max(col - 2, 0)
    hi = min(col + 3, n_cols)
    strip = frame[:, lo:hi]
    score = strip.sum(axis=1).astype(float)
    score[(strip >= sat_thr).any(axis=1)] = -np.inf

    if not np.isfinite(score.max()):
        warnings.warn(
            f"trace_line: all rows saturated near column {x0}; "
            f"returning NaN array."
        )
        return centers

    row0 = int(np.argmax(score))

    for step in (1, -1):
        xc = float(col)
        r, gap = row0, 0
        while 0 <= r < n_rows:
            c, ok = _centroid_row(frame[r], xc, halfwin, sat_thr, min_peak)
            if ok and abs(c - xc) <= max_jump:
                centers[r] = c
                xc = c
                gap = 0
            else:
                gap += 1
                if gap > max_gap:
                    break
            r += step
    return centers


# -----------------------------------------------------------------
#  Robust 2-D surface fit
# -----------------------------------------------------------------


class Surface2D:
    """Polynomial surface ``S(y, x)`` on normalised coordinates.

    The surface is::

        S(y, x) = sum_{i,j} a_{ij} * Y^i * X^j

    where ``Y, X`` are linearly mapped to ``[-1, 1]``.  When *clamp* is
    ``True`` (default), evaluation outside the fitting range is clamped
    to the boundary instead of extrapolating.

    Args:
        coef: Flat coefficient array of length ``(deg_y+1)*(deg_x+1)``.
        deg_y: Polynomial degree in the spatial (row) direction.
        deg_x: Polynomial degree in the spectral (column) direction.
        y_rng: ``(min, max)`` of *y* values used during fitting.
        x_rng: ``(min, max)`` of *x* values used during fitting.
        clamp: Whether to clamp normalised coordinates to ``[-1, 1]``.
    """

    def __init__(
        self,
        coef: np.ndarray,
        deg_y: int,
        deg_x: int,
        y_rng: Tuple[float, float],
        x_rng: Tuple[float, float],
        clamp: bool = True,
    ):
        self.coef = np.asarray(coef, dtype=float)
        self.deg_y = int(deg_y)
        self.deg_x = int(deg_x)
        self.y_rng = (float(y_rng[0]), float(y_rng[1]))
        self.x_rng = (float(x_rng[0]), float(x_rng[1]))
        self.clamp = clamp

    def __call__(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Evaluate the surface at arbitrary ``(y, x)`` positions.

        Args:
            y: Row coordinates (any shape).
            x: Column coordinates (broadcastable with *y*).

        Returns:
            Surface values with the broadcast shape of *y* and *x*.
        """
        y0, y1 = self.y_rng
        x0, x1 = self.x_rng
        Y = 2.0 * (np.asarray(y, dtype=float) - y0) / (y1 - y0) - 1.0
        X = 2.0 * (np.asarray(x, dtype=float) - x0) / (x1 - x0) - 1.0
        if self.clamp:
            Y = np.clip(Y, -1.0, 1.0)
            X = np.clip(X, -1.0, 1.0)
        Y, X = np.broadcast_arrays(Y, X)

        out = np.zeros(Y.shape, dtype=float)
        k = 0
        for i in range(self.deg_y + 1):
            for j in range(self.deg_x + 1):
                out += self.coef[k] * (Y**i) * (X**j)
                k += 1
        return out


def _rng(a: np.ndarray) -> Tuple[float, float]:
    """Return ``(min, max)`` of *a*, handling the degenerate case."""
    lo, hi = float(a.min()), float(a.max())
    return (lo, hi) if hi > lo else (lo - 1.0, lo + 1.0)


def fit_surface_2d(
    y: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    deg_y: int = 2,
    deg_x: int = 1,
    huber_k: float = 1.5,
    n_iter: int = 20,
) -> Tuple[Surface2D, dict]:
    """Fit a robust 2-D polynomial surface using IRLS with Huber weights.

    Outliers beyond ``4 * robust_sigma`` are hard-rejected each iteration.

    Args:
        y: Row coordinates of measurement points.
        x: Column coordinates of measurement points.
        v: Measured shift values at ``(y, x)``.
        deg_y: Polynomial degree in *y*.
        deg_x: Polynomial degree in *x*.
        huber_k: Huber loss threshold in residual-sigma units.
        n_iter: Maximum IRLS iterations (0 = single least-squares fit).

    Returns:
        Tuple ``(Surface2D, info_dict)``.  *info_dict* keys: ``rms``,
        ``n``, ``n_used``, ``dof``, ``deg_y``, ``deg_x``, ``coef``,
        ``stderr``.

    Raises:
        ValueError: If data points are insufficient for the requested
            polynomial degree.
    """
    y, x, v = (np.asarray(a, dtype=float) for a in (y, x, v))
    mask = np.isfinite(y) & np.isfinite(x) & np.isfinite(v)
    y, x, v = y[mask], x[mask], v[mask]

    n_coef = (deg_y + 1) * (deg_x + 1)
    if y.size < n_coef + 2:
        raise ValueError(
            f"Insufficient data ({y.size} points) for degree {deg_y}x{deg_x}"
        )
    n_ux = np.unique(np.round(x, 4)).size
    if n_ux <= deg_x:
        raise ValueError(
            f"Only {n_ux} unique x positions; need at least {deg_x + 1} "
            f"for deg_x={deg_x}"
        )

    y_rng, x_rng = _rng(y), _rng(x)
    Y = 2.0 * (y - y_rng[0]) / (y_rng[1] - y_rng[0]) - 1.0
    X = 2.0 * (x - x_rng[0]) / (x_rng[1] - x_rng[0]) - 1.0
    A = np.column_stack(
        [Y**i * X**j for i in range(deg_y + 1) for j in range(deg_x + 1)]
    )

    w = np.ones_like(v)
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], v * sw, rcond=None)
    res = v - A @ coef

    for _ in range(max(0, n_iter)):
        sig = robust_sigma(res)
        u = np.abs(res) / (huber_k * sig)
        w = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-9))
        w[np.abs(res) > 4.0 * sig] = 0.0
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], v * sw, rcond=None)
        res = v - A @ coef

    n_used = int(np.sum(w > 0))
    rms = float(np.sqrt(np.sum(w * res**2) / max(np.sum(w), 1.0)))
    dof = max(n_used - n_coef, 1)

    try:
        cov = (np.sum(w * res**2) / dof) * np.linalg.pinv((A * w[:, None]).T @ A)
        stderr = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        stderr = np.full(A.shape[1], np.nan)

    info = {
        "rms": rms,
        "n": int(y.size),
        "n_used": n_used,
        "dof": dof,
        "deg_y": deg_y,
        "deg_x": deg_x,
        "coef": coef,
        "stderr": stderr,
    }
    return Surface2D(coef, deg_y, deg_x, y_rng, x_rng), info


def cross_validate_lines(
    y: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    ids: np.ndarray,
    deg_y: int,
    deg_x: int,
) -> float:
    """Leave-one-line-out cross-validation.

    Fits the surface without line *k* and measures residuals on line *k*.
    Edge lines are extrapolated, making this a pessimistic estimate.

    Args:
        y: Row coordinates.
        x: Column coordinates.
        v: Measured shift values.
        ids: Integer line index for each point.
        deg_y: Polynomial degree in *y*.
        deg_x: Polynomial degree in *x*.

    Returns:
        Cross-validated RMS in pixels, or ``nan`` if fewer than
        ``deg_x + 2`` unique lines are available.
    """
    ids_u = np.unique(ids)
    if ids_u.size < deg_x + 2:
        return float("nan")

    sq: List[np.ndarray] = []
    for k in ids_u:
        tr = ids != k
        if np.unique(np.round(x[tr], 4)).size <= deg_x:
            continue
        try:
            s, _ = fit_surface_2d(y[tr], x[tr], v[tr], deg_y, deg_x)
        except ValueError:
            continue
        sq.append((v[~tr] - s(y[~tr], x[~tr])) ** 2)

    return float(np.sqrt(np.mean(np.concatenate(sq)))) if sq else float("nan")


# -----------------------------------------------------------------
#  Remap + margin
# -----------------------------------------------------------------


def build_maps(
    shape: Tuple[int, int], smile: Surface2D
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build remap arrays for :func:`cv2.remap`.

    Args:
        shape: Image shape ``(H, W)``.
        smile: Fitted displacement surface.

    Returns:
        Tuple ``(map_x, map_y, S)`` where ``map_x`` and ``map_y`` are
        ``float32`` arrays of shape ``(H, W)`` and ``S`` is the surface
        values.
    """
    n_rows, n_cols = shape
    X, Y = np.meshgrid(
        np.arange(n_cols, dtype=np.float32),
        np.arange(n_rows, dtype=np.float32),
    )
    S = smile(Y, X)
    map_x = np.ascontiguousarray(X + S, dtype=np.float32)
    map_y = np.ascontiguousarray(
        np.repeat(np.arange(n_rows, dtype=np.float32)[:, None], n_cols, axis=1)
    )
    return map_x, map_y, S


def margin_from_map(map_x: np.ndarray, interp: str = "cubic") -> Tuple[int, int]:
    """Compute the left/right margin where all source pixels exist.

    Accounts for the interpolation kernel taps (cubic uses 1 + 2 taps,
    lanczos4 uses 3 + 4) so that no zero-valued border leaks into the
    result.

    Args:
        map_x: Horizontal remap array.
        interp: Interpolation method name.

    Returns:
        Tuple ``(left, right)`` pixel counts to crop.

    Raises:
        ValueError: If no valid column exists after correction.
    """
    if interp not in _INTERP:
        raise ValueError(f"interp must be one of {list(_INTERP)}, got {interp!r}")
    _, pad_l, pad_r = _INTERP[interp]
    n_cols = map_x.shape[1]
    lo = float(pad_l)
    hi = n_cols - 1.0 if interp == "linear" else n_cols - pad_r - 1e-3

    valid = ((map_x >= lo) & (map_x <= hi)).all(axis=0)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        raise ValueError(
            "No valid columns after smile correction -- shift is too large. "
            "Check the fit or reduce polynomial degree."
        )
    if idx.size != idx[-1] - idx[0] + 1:
        warnings.warn("Valid columns are non-contiguous; margin is approximate.")
    return int(idx[0]), int(n_cols - 1 - idx[-1])


def correct_smile(
    frame: np.ndarray, smile: Surface2D, interp: str = "cubic"
) -> Tuple[np.ndarray, Tuple[int, int], np.ndarray, np.ndarray]:
    """Apply smile correction via sub-pixel remapping.

    Border regions without source data are set to zero (never replicated).

    Args:
        frame: Input image ``(H, W)``.
        smile: Fitted 2-D displacement surface.
        interp: Interpolation method.

    Returns:
        Tuple ``(corrected, margin, map_x, map_y)``.

    Raises:
        ValueError: If *interp* is not a recognised method.
    """
    if interp not in _INTERP:
        raise ValueError(f"interp must be one of {list(_INTERP)}, got {interp!r}")
    flag = _INTERP[interp][0]
    frame = np.ascontiguousarray(frame, dtype=np.float32)
    map_x, map_y, _ = build_maps(frame.shape, smile)
    corrected = cv2.remap(
        frame, map_x, map_y,
        interpolation=flag,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return corrected, margin_from_map(map_x, interp), map_x, map_y


def crop_margin(frame: np.ndarray, margin: Tuple[int, int]) -> np.ndarray:
    """Crop invalid (zero) border columns produced by :func:`correct_smile`.

    Args:
        frame: Corrected image with zero-valued borders.
        margin: ``(left, right)`` pixel counts to remove.

    Returns:
        Cropped image, or *frame* unchanged when ``margin == (0, 0)``.
    """
    left, right = margin
    if left + right > 0:
        return frame[:, left: frame.shape[1] - right]
    return frame


def _traces_rms(traces: List[np.ndarray]) -> float:
    """Compute RMS deviation from each trace's median position."""
    residuals = []
    for c in traces:
        c = c[np.isfinite(c)]
        if c.size > 10:
            residuals.append(c - np.median(c))
    r = np.concatenate(residuals) if residuals else np.array([0.0])
    return float(np.sqrt(np.mean(r**2)))


def straightness_rms(
    frame: np.ndarray,
    lines: List[Line],
    sat_level: Optional[float] = None,
    noise: Optional[float] = None,
) -> float:
    """Measure line straightness as RMS deviation from the median.

    Args:
        frame: Image to measure.
        lines: Emission lines to evaluate.
        sat_level: Saturation threshold.  Inferred if ``None`` or <= 0.
        noise: Noise level.  Estimated if ``None``.

    Returns:
        RMS deviation in pixels (lower = straighter).
    """
    if sat_level is None or sat_level <= 0:
        sat_level = auto_sat_level(frame)
    if noise is None:
        noise = estimate_noise(frame)
    traces = []
    for ln in lines:
        halfwin = max(3, int(round(1.2 * ln.fwhm)))
        traces.append(trace_line(frame, ln.column, halfwin, sat_level, noise))
    return _traces_rms(traces)


# =====================================================================
#  SECTION 2 -- Analysis  (result bundle)
# =====================================================================


@dataclass
class SmileAnalysis:
    """Complete results of a smile correction analysis.

    Attributes:
        frame_orig: Original calibration image.
        frame_corr: Corrected image (zero-valued borders where data absent).
        all_lines: Every detected emission line.
        used_lines: Lines that passed quality filters and were used
            in the surface fit.
        traces_orig: ``{column -> trace_array}`` for the original image.
        traces_corr: ``{column -> trace_array}`` for the corrected image.
        refs: ``{column -> sub-pixel reference}`` used for each line.
        surface: Fitted 2-D displacement surface.
        fit_info: Surface fitting diagnostics.
        meas_y: Row coordinates of all measurement points.
        meas_x: Sub-pixel reference column coordinates (after adjustment).
        meas_v: Measured shift values at each point.
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


def analyze_smile(
    frame: np.ndarray,
    sat_level: Optional[float] = None,
    deg_y: int = 2,
    deg_x: int = 1,
    interp: str = "cubic",
    snr_min: float = 5.0,
    n_ref_iter: int = 2,
    verbose: bool = False,
) -> SmileAnalysis:
    """Run the full multi-line smile analysis pipeline.

    Steps:
        1. Detect emission lines.
        2. Trace each line sub-pixel through all rows.
        3. Use per-line median as reference (sub-pixel).
        4. Iteratively adjust per-line offsets and fit robust 2-D surface.
        5. Remap with zero-valued borders.
        6. Re-trace and compute straightness metrics + cross-validation.

    Args:
        frame: Calibration image ``(H, W)``.
        sat_level: Saturation threshold.  Inferred if ``None`` or <= 0.
        deg_y: Polynomial degree in the spatial (row) direction.
        deg_x: Polynomial degree in the spectral (column) direction.
        interp: Interpolation method.
        snr_min: Minimum peak-to-noise ratio for tracing.
        n_ref_iter: Number of reference-offset iterations.
        verbose: Print per-line report to stdout.

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

    lines = detect_lines(frame, sat_level)
    if not lines:
        raise ValueError("No emission lines detected. Check image brightness.")

    traces_orig: Dict[float, np.ndarray] = {}
    for ln in lines:
        halfwin = max(3, int(round(1.2 * ln.fwhm)))
        traces_orig[ln.column] = trace_line(
            frame, ln.column, halfwin, sat_level, noise, snr_min
        )

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
            "All lines were filtered out (saturated, low coverage, or low SNR)."
        )

    meas_y = np.concatenate(ys)
    meas_x = np.concatenate(xs)
    meas_v = np.concatenate(vs)
    meas_id = np.concatenate(ids)

    meas_x_orig = meas_x.copy()

    n_ux = np.unique(np.round(meas_x, 4)).size
    deg_x_eff = min(deg_x, n_ux - 1)
    if deg_x_eff < deg_x:
        warnings.warn(
            f"Only {n_ux} unique line positions; "
            f"reducing deg_x from {deg_x} to {deg_x_eff}."
        )

    for _ in range(max(0, n_ref_iter)):
        surf_tmp, _ = fit_surface_2d(meas_y, meas_x, meas_v, deg_y, deg_x_eff)
        res_tmp = meas_v - surf_tmp(meas_y, meas_x)
        for k, ln in enumerate(used):
            mk = meas_id == k
            off = float(np.median(res_tmp[mk]))
            meas_v[mk] -= off
            meas_x[mk] += off
            refs[ln.column] += off

    surface, fit_info = fit_surface_2d(meas_y, meas_x, meas_v, deg_y, deg_x_eff)
    residuals = meas_v - surface(meas_y, meas_x)
    cv_rms = cross_validate_lines(
        meas_y, meas_x_orig, meas_v, meas_id, deg_y, deg_x_eff
    )

    frame_corr, margin, _, _ = correct_smile(frame, surface, interp)

    traces_corr: Dict[float, np.ndarray] = {}
    for ln in used:
        halfwin = max(3, int(round(1.2 * ln.fwhm)))
        traces_corr[ln.column] = trace_line(
            frame_corr, refs[ln.column], halfwin, sat_level, noise, snr_min
        )

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
    )


def verify_on_frame(analysis: SmileAnalysis, other_frame: np.ndarray) -> float:
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
    sat = analysis.sat_level
    noise = estimate_noise(other)
    traces = []
    for ln in analysis.used_lines:
        halfwin = max(3, int(round(1.2 * ln.fwhm)))
        traces.append(trace_line(corr, analysis.refs[ln.column], halfwin, sat, noise))
    return _traces_rms(traces)


# =====================================================================
#  SECTION 2b -- Calibration save / load
# =====================================================================


@dataclass
class SmileCalibration:
    """Persistable calibration for repeated application.

    Stores ``map_x`` only (``map_y`` is a trivial row-index array built
    lazily).

    Attributes:
        map_x: Horizontal remap array ``float32 (H, W)``.
        margin: ``(left, right)`` border pixels to crop.
        image_shape: ``(H, W)`` of the calibrated sensor.
        deg_y: Polynomial degree in the spatial direction.
        deg_x: Polynomial degree in the spectral direction.
        surface_coef: Flattened polynomial coefficients.
        y_rng: Row coordinate range used during fitting.
        x_rng: Column coordinate range used during fitting.
        interp: Interpolation method name.
        fit_rms: Residual RMS of the surface fit.
        cv_rms: Leave-one-line-out cross-validated RMS.
        rms_before: Straightness RMS before correction.
        rms_after: Straightness RMS after correction.
        n_lines_used: Number of lines used in the fit.
        n_lines_total: Number of lines detected.
        created: Timestamp string.
        notes: Free-form annotation.
    """

    map_x: np.ndarray
    margin: Tuple[int, int]
    image_shape: Tuple[int, int]
    deg_y: int
    deg_x: int
    surface_coef: np.ndarray
    y_rng: Tuple[float, float]
    x_rng: Tuple[float, float]
    interp: str
    fit_rms: float
    cv_rms: float
    rms_before: float
    rms_after: float
    n_lines_used: int
    n_lines_total: int
    created: str
    notes: str = ""
    _map_y: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def surface(self) -> Surface2D:
        """Reconstruct the :class:`Surface2D` from stored coefficients.

        Returns:
            The polynomial surface.

        Note:
            For format v1 files, ``y_rng``/``x_rng`` are approximate.
        """
        return Surface2D(
            self.surface_coef, self.deg_y, self.deg_x, self.y_rng, self.x_rng
        )

    def _get_map_y(self) -> np.ndarray:
        """Build (or return cached) ``map_y`` row-index array."""
        h, w = self.image_shape
        if self._map_y is None or self._map_y.shape != (h, w):
            self._map_y = np.ascontiguousarray(
                np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
            )
        return self._map_y

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply this calibration to a new frame.

        Args:
            frame: Image whose shape must match ``self.image_shape``.

        Returns:
            Corrected image with zero-valued borders.

        Raises:
            ValueError: If *frame* shape does not match.
        """
        if frame.shape != tuple(self.image_shape):
            raise ValueError(
                f"Frame shape mismatch: got {frame.shape}, "
                f"calibrated for {self.image_shape}. Re-calibration required."
            )
        frame = np.ascontiguousarray(frame, dtype=np.float32)
        return cv2.remap(
            frame,
            np.ascontiguousarray(self.map_x, dtype=np.float32),
            self._get_map_y(),
            interpolation=_INTERP[self.interp][0],
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def apply_cropped(self, frame: np.ndarray) -> np.ndarray:
        """Apply calibration and crop the invalid border.

        Args:
            frame: Image to correct.

        Returns:
            Corrected image with borders removed.
        """
        return crop_margin(self.apply(frame), self.margin)

    def summary(self) -> str:
        """Return a human-readable summary.

        Returns:
            Multi-line summary string.
        """
        left, right = self.margin
        cv_str = f"{self.cv_rms:.5f}" if np.isfinite(self.cv_rms) else "n/a"
        return (
            f"Smile Calibration [{self.created}]\n"
            f"  Sensor size  : {self.image_shape[1]}x{self.image_shape[0]} px\n"
            f"  Lines used   : {self.n_lines_used}/{self.n_lines_total}\n"
            f"  Fit          : deg_y={self.deg_y} deg_x={self.deg_x} "
            f"interp={self.interp}  RMS={self.fit_rms:.5f} px  "
            f"CV={cv_str} px\n"
            f"  Straightness : {self.rms_before:.4f} -> {self.rms_after:.4f} px\n"
            f"  Border crop  : left={left} right={right} px "
            f"({left + right} px total)\n"
            f"  Notes        : {self.notes}"
        )


def calibration_from_analysis(
    analysis: SmileAnalysis,
    notes: str = "",
) -> SmileCalibration:
    """Build a :class:`SmileCalibration` from analysis results.

    Args:
        analysis: Completed :class:`SmileAnalysis`.
        notes: Free-form annotation.

    Returns:
        A ready-to-save :class:`SmileCalibration`.
    """
    s = analysis.surface
    map_x, _, _ = build_maps(analysis.frame_orig.shape, s)
    return SmileCalibration(
        map_x=map_x,
        margin=tuple(analysis.margin),
        image_shape=tuple(analysis.frame_orig.shape),
        deg_y=s.deg_y,
        deg_x=s.deg_x,
        surface_coef=s.coef.copy(),
        y_rng=s.y_rng,
        x_rng=s.x_rng,
        interp=analysis.interp,
        fit_rms=float(analysis.fit_info["rms"]),
        cv_rms=float(analysis.cv_rms),
        rms_before=float(analysis.rms_before),
        rms_after=float(analysis.rms_after),
        n_lines_used=len(analysis.used_lines),
        n_lines_total=len(analysis.all_lines),
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        notes=notes,
    )


def save_calibration(path: str, cal: SmileCalibration) -> None:
    """Write a calibration to a compressed ``.npz`` file.

    Args:
        path: Destination file path.
        cal: Calibration to serialise.
    """
    meta = {
        "format_version": 2,
        "margin": [int(v) for v in cal.margin],
        "image_shape": [int(v) for v in cal.image_shape],
        "deg_y": cal.deg_y,
        "deg_x": cal.deg_x,
        "surface_coef": cal.surface_coef.tolist(),
        "y_rng": list(cal.y_rng),
        "x_rng": list(cal.x_rng),
        "interp": cal.interp,
        "fit_rms": cal.fit_rms,
        "cv_rms": cal.cv_rms,
        "rms_before": cal.rms_before,
        "rms_after": cal.rms_after,
        "n_lines_used": cal.n_lines_used,
        "n_lines_total": cal.n_lines_total,
        "created": cal.created,
        "notes": cal.notes,
    }
    np.savez_compressed(
        path,
        map_x=cal.map_x.astype(np.float32),
        metadata=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
    )


def load_calibration(path: str) -> SmileCalibration:
    """Read a calibration from ``.npz`` (supports format v1 and v2).

    Args:
        path: Path to the ``.npz`` file.

    Returns:
        The deserialised :class:`SmileCalibration`.

    Raises:
        ValueError: If the format version is newer than supported.
    """
    with np.load(path) as data:
        meta = json.loads(data["metadata"].tobytes().decode("utf-8"))
        map_x = data["map_x"].astype(np.float32)

    ver = meta.get("format_version", 1)
    if ver > 2:
        raise ValueError(f"format_version {ver} is newer than this module supports.")

    shape = tuple(meta["image_shape"])
    return SmileCalibration(
        map_x=map_x,
        margin=tuple(int(v) for v in meta["margin"]),
        image_shape=(int(shape[0]), int(shape[1])),
        deg_y=meta["deg_y"],
        deg_x=meta["deg_x"],
        surface_coef=np.array(meta["surface_coef"]),
        y_rng=tuple(meta.get("y_rng", (0.0, float(shape[0] - 1)))),
        x_rng=tuple(meta.get("x_rng", (0.0, float(shape[1] - 1)))),
        interp=meta.get("interp", "linear" if ver == 1 else "cubic"),
        fit_rms=meta["fit_rms"],
        cv_rms=meta.get("cv_rms", float("nan")),
        rms_before=meta["rms_before"],
        rms_after=meta["rms_after"],
        n_lines_used=meta["n_lines_used"],
        n_lines_total=meta["n_lines_total"],
        created=meta["created"],
        notes=meta.get("notes", ""),
    )


# =====================================================================
#  SECTION 2c -- SmileCorrector  (sklearn-style fit / transform API)
# =====================================================================


class SmileCorrector:
    """Sklearn-style smile correction: ``fit`` once, ``transform`` many.

    Usage::

        # Calibrate once
        corrector = SmileCorrector(deg_y=2, deg_x=1, interp="cubic")
        corrector.fit(hgar_frame)
        corrector.save("smile_model.npz")

        # Apply to many frames
        corrected = corrector.transform(raw_frame, crop=True)

        # Load later
        corrector = SmileCorrector.load("smile_model.npz")
        corrected = corrector.transform(raw_frame)

        # Undo
        restored = corrector.inverse_transform(corrected)

    Args:
        deg_y: Polynomial degree in the spatial (row) direction.
        deg_x: Polynomial degree in the spectral (column) direction.
        interp: Interpolation method ("linear", "cubic", "lanczos").
        sat_level: Sensor full-scale (None = auto-detect).
        snr_min: Minimum peak-to-noise ratio for tracing.
        n_ref_iter: Number of reference-offset iterations.
    """

    def __init__(
        self,
        deg_y: int = 2,
        deg_x: int = 1,
        interp: str = "cubic",
        sat_level: Optional[float] = None,
        snr_min: float = 5.0,
        n_ref_iter: int = 2,
    ):
        self.deg_y = deg_y
        self.deg_x = deg_x
        self.interp = interp
        self.sat_level = sat_level
        self.snr_min = snr_min
        self.n_ref_iter = n_ref_iter

        # Fitted state (set by fit() or load()).
        self._surface: Optional[Surface2D] = None
        self._map_x: Optional[np.ndarray] = None
        self._margin: Tuple[int, int] = (0, 0)
        self._image_shape: Optional[Tuple[int, int]] = None
        self._analysis: Optional[SmileAnalysis] = None

    # ─── Core API ───

    def fit(self, frame: np.ndarray, verbose: bool = False) -> SmileCorrector:
        """Learn the smile model from a calibration frame.

        Args:
            frame: Hg-Ar calibration image ``(H, W)``.
            verbose: Print analysis report.

        Returns:
            self (for method chaining).

        Raises:
            ValueError: If no usable emission lines are found.
        """
        self._analysis = analyze_smile(
            np.asarray(frame, dtype=np.float32),
            sat_level=self.sat_level,
            deg_y=self.deg_y,
            deg_x=self.deg_x,
            interp=self.interp,
            snr_min=self.snr_min,
            n_ref_iter=self.n_ref_iter,
            verbose=verbose,
        )
        self._surface = self._analysis.surface
        self._map_x, _, _ = build_maps(
            self._analysis.frame_orig.shape, self._surface
        )
        self._margin = self._analysis.margin
        self._image_shape = tuple(self._analysis.frame_orig.shape)
        return self

    def transform(self, frame: np.ndarray, crop: bool = True) -> np.ndarray:
        """Apply the fitted smile correction to a new frame.

        Args:
            frame: Image ``(H, W)`` matching the fitted sensor shape.
            crop: Remove invalid border columns (data is truly lost).

        Returns:
            Corrected image.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If *frame* shape does not match.
        """
        self._check_fitted()
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != self._image_shape:
            raise ValueError(
                f"Frame shape {frame.shape} != fitted {self._image_shape}"
            )
        corrected = cv2.remap(
            np.ascontiguousarray(frame, dtype=np.float32),
            np.ascontiguousarray(self._map_x, dtype=np.float32),
            self._get_map_y(),
            interpolation=_INTERP[self.interp][0],
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return crop_margin(corrected, self._margin) if crop else corrected

    def fit_transform(
        self, frame: np.ndarray, crop: bool = True, verbose: bool = False
    ) -> np.ndarray:
        """Fit on *frame*, then transform the same frame.

        Args:
            frame: Calibration image ``(H, W)``.
            crop: Remove invalid border columns.
            verbose: Print analysis report.

        Returns:
            Corrected image.
        """
        self.fit(frame, verbose=verbose)
        return self.transform(frame, crop=crop)

    def inverse_transform(self, frame: np.ndarray) -> np.ndarray:
        """Undo the smile correction (reverse displacement).

        Approximate: double interpolation introduces small round-trip
        error.  Useful for coordinate mapping and debugging.

        Args:
            frame: Corrected image ``(H, W)`` (uncropped).

        Returns:
            Image with smile restored.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        self._check_fitted()
        n_rows, n_cols = self._image_shape
        X, Y = np.meshgrid(
            np.arange(n_cols, dtype=np.float32),
            np.arange(n_rows, dtype=np.float32),
        )
        S = self._surface(Y, X)
        inv_map_x = np.ascontiguousarray(X - S, dtype=np.float32)
        inv_map_y = np.ascontiguousarray(
            np.repeat(np.arange(n_rows, dtype=np.float32)[:, None], n_cols, axis=1)
        )
        return cv2.remap(
            np.ascontiguousarray(frame, dtype=np.float32),
            inv_map_x,
            inv_map_y,
            interpolation=_INTERP[self.interp][0],
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    # ─── Persistence ───

    def save(self, path: str) -> None:
        """Save the fitted model to a compressed ``.npz`` file.

        Args:
            path: Destination file path.
        """
        self._check_fitted()
        cal = calibration_from_analysis(self._analysis) if self._analysis else None
        if cal is None:
            # Reconstruct from fitted state (loaded from file).
            cal = SmileCalibration(
                map_x=self._map_x,
                margin=self._margin,
                image_shape=self._image_shape,
                deg_y=self._surface.deg_y,
                deg_x=self._surface.deg_x,
                surface_coef=self._surface.coef.copy(),
                y_rng=self._surface.y_rng,
                x_rng=self._surface.x_rng,
                interp=self.interp,
                fit_rms=float("nan"),
                cv_rms=float("nan"),
                rms_before=float("nan"),
                rms_after=float("nan"),
                n_lines_used=0,
                n_lines_total=0,
                created=time.strftime("%Y-%m-%d %H:%M:%S"),
                notes="re-saved from loaded model",
            )
        save_calibration(path, cal)

    @classmethod
    def load(cls, path: str) -> SmileCorrector:
        """Load a fitted model from a ``.npz`` file.

        Args:
            path: Path to the saved file.

        Returns:
            A ready-to-use :class:`SmileCorrector`.
        """
        cal = load_calibration(path)
        obj = cls(
            deg_y=cal.deg_y,
            deg_x=cal.deg_x,
            interp=cal.interp,
        )
        obj._map_x = cal.map_x
        obj._margin = cal.margin
        obj._image_shape = tuple(cal.image_shape)
        obj._surface = cal.surface()
        return obj

    # ─── Parameters ───

    def get_params(self) -> dict:
        """Return hyperparameters as a dictionary.

        Returns:
            Dict of parameter names and values.
        """
        return {
            "deg_y": self.deg_y,
            "deg_x": self.deg_x,
            "interp": self.interp,
            "sat_level": self.sat_level,
            "snr_min": self.snr_min,
            "n_ref_iter": self.n_ref_iter,
        }

    def set_params(self, **params: object) -> SmileCorrector:
        """Update hyperparameters.

        Args:
            **params: Parameter name-value pairs.

        Returns:
            self.

        Raises:
            ValueError: If a parameter name is unknown.
        """
        for key, val in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown parameter: {key!r}")
            setattr(self, key, val)
        return self

    # ─── Introspection ───

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` or :meth:`load` has been called."""
        return self._surface is not None

    @property
    def margin(self) -> Tuple[int, int]:
        """Border pixels lost by correction ``(left, right)``."""
        return self._margin

    @property
    def analysis(self) -> Optional[SmileAnalysis]:
        """Full analysis result (after :meth:`fit`, not after :meth:`load`)."""
        return self._analysis

    def summary(self) -> str:
        """Return a human-readable summary of the fitted model.

        Returns:
            Multi-line summary string.
        """
        if not self.is_fitted:
            return "SmileCorrector (not fitted)"
        left, right = self._margin
        a = self._analysis
        if a is not None:
            cv_str = f"{a.cv_rms:.5f}" if np.isfinite(a.cv_rms) else "n/a"
            return (
                f"SmileCorrector (fitted)\n"
                f"  Sensor       : {self._image_shape[1]}x{self._image_shape[0]} px\n"
                f"  Lines used   : {len(a.used_lines)}/{len(a.all_lines)}\n"
                f"  Fit          : deg={self.deg_y}x{self.deg_x}  "
                f"interp={self.interp}\n"
                f"  Straightness : {a.rms_before:.4f} -> {a.rms_after:.4f} px\n"
                f"  CV (LOO)     : {cv_str} px\n"
                f"  Border loss  : L={left} R={right} ({left + right} px)"
            )
        return (
            f"SmileCorrector (loaded)\n"
            f"  Sensor       : {self._image_shape[1]}x{self._image_shape[0]} px\n"
            f"  Fit          : deg={self.deg_y}x{self.deg_x}  "
            f"interp={self.interp}\n"
            f"  Border loss  : L={left} R={right} ({left + right} px)"
        )

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"SmileCorrector(deg_y={self.deg_y}, deg_x={self.deg_x}, "
            f"interp={self.interp!r}, {status})"
        )

    # ─── Internal ───

    def _check_fitted(self) -> None:
        if self._surface is None:
            raise RuntimeError("Not fitted. Call fit() or load() first.")

    def _get_map_y(self) -> np.ndarray:
        h, w = self._image_shape
        return np.ascontiguousarray(
            np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
        )


# =====================================================================
#  SECTION 3 -- Qt widget  (optional: module importable without Qt)
# =====================================================================

try:
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtWidgets import (
        QCheckBox,
        QFileDialog,
        QFrame,
        QLabel,
        QPushButton,
        QSizePolicy,
        QSpinBox,
        QTabWidget,
        QHBoxLayout,
        QVBoxLayout,
        QWidget,
    )
    import matplotlib

    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

    HAVE_QT = True
except ImportError:
    HAVE_QT = False
    QWidget = QThread = FigureCanvasQTAgg = object
    pyqtSignal = lambda *a, **k: None  # noqa: E731


_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


class PlotCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas wrapper for Qt layouts.

    Args:
        parent: Parent widget.
        figsize: Figure size ``(width, height)`` in inches.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        figsize: Tuple[float, float] = (8, 5),
    ):
        self.fig = Figure(figsize=figsize, dpi=100)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def clear(self) -> None:
        """Remove all axes and artists from the figure."""
        self.fig.clear()

    def refresh(self) -> None:
        """Apply tight layout and request a canvas redraw."""
        try:
            self.fig.tight_layout()
        except Exception:
            pass
        self.draw_idle()


class _AnalyzeWorker(QThread):
    """Run :func:`analyze_smile` in a background thread.

    Signals:
        done: Emitted with the :class:`SmileAnalysis` on success.
        failed: Emitted with the error message string on failure.
    """

    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, frame: np.ndarray, kwargs: dict):
        super().__init__()
        self._frame = frame
        self._kwargs = kwargs

    def run(self) -> None:
        """Execute the analysis."""
        try:
            self.done.emit(analyze_smile(self._frame, **self._kwargs))
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            self.failed.emit(str(exc))


class SmileCorrectionWidget(QWidget):
    """Interactive smile correction widget for PyQt5 applications.

    Signals:
        analysisComplete: Emitted with :class:`SmileAnalysis` after
            successful analysis.
        statusChanged: Emitted with a status message string.

    Example::

        widget = SmileCorrectionWidget(parent)
        layout.addWidget(widget)
        widget.analysisComplete.connect(my_handler)
        widget.set_frame(raw_frame)
        widget.analyze()          # async -- wait for signal
    """

    analysisComplete = pyqtSignal(object)
    statusChanged = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        if not HAVE_QT:
            raise ImportError("PyQt5 and matplotlib are required for the widget.")
        super().__init__(parent)
        self._frame: Optional[np.ndarray] = None
        self._result: Optional[SmileAnalysis] = None
        self._calibration: Optional[SmileCalibration] = None
        self._corrector: Optional[SmileCorrector] = None
        self._worker: Optional[_AnalyzeWorker] = None
        self._build_ui()

    # -----------------------------------------------------------------
    #  UI construction
    # -----------------------------------------------------------------

    def _build_ui(self) -> None:
        """Construct the widget layout and controls."""
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        ctrl = QHBoxLayout()

        self.btn_load = QPushButton("📂 โหลดภาพ")
        self.btn_load.clicked.connect(self._on_load)
        ctrl.addWidget(self.btn_load)

        self.btn_analyze = QPushButton("🔍 วิเคราะห์ Smile")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.setStyleSheet("font-weight:bold;")
        self.btn_analyze.clicked.connect(self._on_analyze)
        ctrl.addWidget(self.btn_analyze)

        self.btn_save = QPushButton("💾 บันทึก")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._on_save)
        ctrl.addWidget(self.btn_save)

        self.btn_save_cal = QPushButton("📐 บันทึก Matrix")
        self.btn_save_cal.setEnabled(False)
        self.btn_save_cal.setToolTip("Save map_x for reuse without re-analysis.")
        self.btn_save_cal.clicked.connect(self._on_save_cal)
        ctrl.addWidget(self.btn_save_cal)

        self.btn_load_cal = QPushButton("📥 โหลด Matrix")
        self.btn_load_cal.setToolTip("Load a previously saved calibration matrix.")
        self.btn_load_cal.clicked.connect(self._on_load_cal)
        ctrl.addWidget(self.btn_load_cal)

        ctrl.addStretch()

        ctrl.addWidget(QLabel("deg_y:"))
        self.spin_dy = QSpinBox()
        self.spin_dy.setRange(1, 5)
        self.spin_dy.setValue(2)
        ctrl.addWidget(self.spin_dy)

        ctrl.addWidget(QLabel("deg_x:"))
        self.spin_dx = QSpinBox()
        self.spin_dx.setRange(0, 3)
        self.spin_dx.setValue(1)
        ctrl.addWidget(self.spin_dx)

        ctrl.addWidget(QLabel("sat:"))
        self.spin_sat = QSpinBox()
        self.spin_sat.setRange(0, 65535)
        self.spin_sat.setValue(0)
        self.spin_sat.setSpecialValueText("auto")
        self.spin_sat.setToolTip(
            "Sensor full-scale (e.g. 4095 for 12-bit). auto = detect from image."
        )
        ctrl.addWidget(self.spin_sat)

        self.chk_crop = QCheckBox("ตัดขอบ (หายจริง)")
        self.chk_crop.setChecked(True)
        ctrl.addWidget(self.chk_crop)

        root.addLayout(ctrl)

        self.tabs = QTabWidget()
        self._canvases: List[PlotCanvas] = []
        tab_names = ["ภาพ  Before / After", "เส้น  Traces", "Residual", "Shift Map"]
        for name in tab_names:
            canvas = PlotCanvas(self.tabs)
            self._canvases.append(canvas)
            self.tabs.addTab(canvas, name)

        (
            self.canvas_img,
            self.canvas_trace,
            self.canvas_resid,
            self.canvas_surf,
        ) = self._canvases
        root.addWidget(self.tabs, stretch=1)

        self.lbl_status = QLabel("พร้อมใช้งาน")
        self.lbl_status.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.lbl_status.setMinimumHeight(24)
        root.addWidget(self.lbl_status)

    # -----------------------------------------------------------------
    #  Public API
    # -----------------------------------------------------------------

    def set_frame(self, frame: Optional[np.ndarray]) -> None:
        """Set the calibration image from a NumPy array.

        Clears any previous analysis result to prevent saving stale data.

        Args:
            frame: Grayscale image ``(H, W)``, or ``None`` to clear.
        """
        self._result = None
        self._corrector = None
        self.btn_save.setEnabled(False)
        self.btn_save_cal.setEnabled(False)

        self._frame = frame.astype(np.float32) if frame is not None else None
        self.btn_analyze.setEnabled(self._frame is not None and self._worker is None)
        self._show_preview(self._frame)
        if self._frame is not None:
            h, w = self._frame.shape
            self._set_status(
                f"Size {w}x{h}  max={self._frame.max():.0f}  "
                f"sat_level(auto)={auto_sat_level(self._frame):.0f}"
            )

    def load_file(self, path: str) -> None:
        """Load a calibration image from disk.

        Args:
            path: Image file path.
        """
        self.set_frame(load_frame(path))
        h, w = self._frame.shape
        self._set_status(
            f"{os.path.basename(path)}  {w}x{h}  "
            f"max={self._frame.max():.0f}  "
            f"sat_level(auto)={auto_sat_level(self._frame):.0f}"
        )

    def analyze(self) -> None:
        """Trigger async analysis on the current frame."""
        self._on_analyze()

    def get_analysis(self) -> Optional[SmileAnalysis]:
        """Return the most recent analysis result, or ``None``."""
        return self._result

    def get_corrected(self) -> Optional[np.ndarray]:
        """Return the corrected frame, or ``None`` if not yet analysed."""
        return self._result.frame_corr if self._result else None

    def get_corrector(self) -> Optional[SmileCorrector]:
        """Return the fitted :class:`SmileCorrector`, or ``None``."""
        return self._corrector

    def set_params(
        self,
        deg_y: Optional[int] = None,
        deg_x: Optional[int] = None,
        crop: Optional[bool] = None,
        sat_level: Optional[int] = None,
    ) -> None:
        """Update fit parameters programmatically.

        Args:
            deg_y: Polynomial degree in the spatial direction.
            deg_x: Polynomial degree in the spectral direction.
            crop: Whether to crop invalid borders on save.
            sat_level: Sensor full-scale value (0 = auto-detect).
        """
        if deg_y is not None:
            self.spin_dy.setValue(deg_y)
        if deg_x is not None:
            self.spin_dx.setValue(deg_x)
        if crop is not None:
            self.chk_crop.setChecked(crop)
        if sat_level is not None:
            self.spin_sat.setValue(int(sat_level))

    def apply_calibration(self, frame: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply a previously loaded calibration matrix to a frame.

        Args:
            frame: Target image.  Falls back to ``self._frame`` if ``None``.

        Returns:
            Corrected and cropped image.

        Raises:
            ValueError: If no calibration is loaded or no frame is available.
        """
        if self._calibration is None:
            raise ValueError("No calibration matrix loaded.")
        src = frame if frame is not None else self._frame
        if src is None:
            raise ValueError("No frame available.")
        corrected = self._calibration.apply_cropped(src)
        self._set_status(
            f"Applied: {self._calibration.image_shape[1]}x"
            f"{self._calibration.image_shape[0]} -> "
            f"{corrected.shape[1]}x{corrected.shape[0]}"
        )
        return corrected

    # -----------------------------------------------------------------
    #  Internal helpers
    # -----------------------------------------------------------------

    def _set_status(self, msg: str) -> None:
        self.lbl_status.setText(msg)
        self.statusChanged.emit(msg)

    def _show_preview(self, frame: Optional[np.ndarray]) -> None:
        self.canvas_img.clear()
        ax = self.canvas_img.fig.add_subplot(111)
        if frame is not None:
            vmin, vmax = np.percentile(frame, [1, 99.5])
            ax.imshow(frame, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
            ax.set_title("Original -- waiting for analysis")
        self.canvas_img.refresh()

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "เลือกภาพ calibration",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All (*.*)",
        )
        if path:
            try:
                self.load_file(path)
            except Exception as exc:
                self._set_status(f"Error: {exc}")

    def _on_analyze(self) -> None:
        if self._frame is None or self._worker is not None:
            return
        kwargs = dict(
            sat_level=(self.spin_sat.value() or None),
            deg_y=self.spin_dy.value(),
            deg_x=self.spin_dx.value(),
            verbose=True,
        )
        self._set_status("Analyzing ...")
        self.btn_analyze.setEnabled(False)
        worker = _AnalyzeWorker(self._frame.copy(), kwargs)
        worker.done.connect(self._on_analysis_done)
        worker.failed.connect(self._on_analysis_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        worker.start()

    def _on_worker_finished(self) -> None:
        if self._worker is not None:
            self._worker.wait()
        self._worker = None
        self.btn_analyze.setEnabled(self._frame is not None)

    def _on_analysis_failed(self, msg: str) -> None:
        self._set_status(f"Error: {msg}")

    def _on_analysis_done(self, r: SmileAnalysis) -> None:
        self._result = r

        # Build a SmileCorrector from the analysis for easy reuse.
        self._corrector = SmileCorrector(
            deg_y=self.spin_dy.value(),
            deg_x=self.spin_dx.value(),
            interp=r.interp,
        )
        self._corrector._surface = r.surface
        self._corrector._map_x, _, _ = build_maps(r.frame_orig.shape, r.surface)
        self._corrector._margin = r.margin
        self._corrector._image_shape = tuple(r.frame_orig.shape)
        self._corrector._analysis = r

        cv_str = f"{r.cv_rms:.3f}" if np.isfinite(r.cv_rms) else "n/a"
        self._set_status(
            f"Lines {len(r.used_lines)}/{len(r.all_lines)}  |  "
            f"RMS {r.rms_before:.3f}->{r.rms_after:.3f} px "
            f"(CV {cv_str})  |  "
            f"margin L={r.margin[0]} R={r.margin[1]} px "
            f"({r.margin[0] + r.margin[1]} px lost)"
        )
        self.btn_save.setEnabled(True)
        self.btn_save_cal.setEnabled(True)
        self._update_all_plots(r)
        self.analysisComplete.emit(r)

    def _on_save(self) -> None:
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "บันทึกภาพที่แก้ smile แล้ว",
            "Corrected.png",
            "PNG (*.png);;TIFF (*.tif);;All (*.*)",
        )
        if not path:
            return
        img = self._result.frame_corr
        if self.chk_crop.isChecked():
            img = crop_margin(img, self._result.margin)
        bits = 8 if self._result.sat_level <= 255 else 16
        save_frame(path, img, bits=bits)
        self._set_status(
            f"Saved {path} ({img.shape[1]}x{img.shape[0]} px, {bits}-bit)"
        )

    def _on_save_cal(self) -> None:
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "บันทึก Smile Calibration Matrix",
            "smile_calibration.npz",
            "Smile Calibration (*.npz);;All (*.*)",
        )
        if not path:
            return
        if not path.endswith(".npz"):
            path += ".npz"
        if self._corrector is not None:
            self._corrector.save(path)
        else:
            cal = calibration_from_analysis(self._result)
            save_calibration(path, cal)
        self._set_status(f"Saved calibration: {path}")

    def _on_load_cal(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "โหลด Smile Calibration Matrix",
            "",
            "Smile Calibration (*.npz);;All (*.*)",
        )
        if not path:
            return
        try:
            self._calibration = load_calibration(path)
            self._corrector = SmileCorrector.load(path)
            self._set_status(self._calibration.summary().replace("\n", " | "))
        except Exception as exc:
            self._set_status(f"Load failed: {exc}")

    # -----------------------------------------------------------------
    #  Plotting
    # -----------------------------------------------------------------

    def _update_all_plots(self, r: SmileAnalysis) -> None:
        self._plot_image(r)
        self._plot_traces(r)
        self._plot_residual(r)
        self._plot_surface(r)

    def _plot_image(self, r: SmileAnalysis) -> None:
        self.canvas_img.clear()
        fig = self.canvas_img.fig

        vmin, vmax = np.percentile(r.frame_orig, [1, 99.5])
        n_cols = r.frame_orig.shape[1]

        ax1 = fig.add_subplot(211)
        ax1.imshow(r.frame_orig, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
        ax1.set_title(f"Before  ({n_cols} cols)", fontsize=10)
        ax1.set_ylabel("Row")

        ax2 = fig.add_subplot(212)
        img = r.frame_corr
        left, right = r.margin
        if self.chk_crop.isChecked() and (left + right > 0):
            img = crop_margin(img, (left, right))
            title = (
                f"After  ({img.shape[1]} cols -- cropped L={left} R={right}, "
                f"{left + right} px lost)"
            )
        else:
            title = (
                f"After  ({img.shape[1]} cols -- not cropped, "
                f"border zeros = no data)"
            )
        ax2.imshow(img, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
        ax2.set_title(title, fontsize=10)
        ax2.set_xlabel("Column (spectral)")
        ax2.set_ylabel("Row")
        self.canvas_img.refresh()

    def _plot_traces(self, r: SmileAnalysis) -> None:
        self.canvas_trace.clear()
        fig = self.canvas_trace.fig
        rows = np.arange(r.frame_orig.shape[0])

        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)

        for i, ln in enumerate(r.used_lines):
            color = _COLORS[i % len(_COLORS)]
            label = f"col {ln.column:.0f}"
            ref = r.refs[ln.column]

            c = r.traces_orig.get(ln.column)
            if c is not None:
                v = np.isfinite(c)
                ax1.plot(c[v] - ref, rows[v], ".", ms=0.5, color=color, label=label)

            c2 = r.traces_corr.get(ln.column)
            if c2 is not None:
                v2 = np.isfinite(c2)
                if v2.any():
                    ref2 = float(np.median(c2[v2]))
                    ax2.plot(
                        c2[v2] - ref2, rows[v2],
                        ".", ms=0.5, color=color, label=label,
                    )

        ax1.invert_yaxis()
        ax2.invert_yaxis()
        ax1.set_ylabel("Row")
        ax1.set_title(f"Before -- RMS {r.rms_before:.4f} px", fontsize=10)
        if len(r.used_lines) <= 12:
            ax1.legend(fontsize=5, loc="upper right", ncol=2)

        ax2.set_xlabel("Shift from per-line reference (px)")
        ax2.set_ylabel("Row")
        ax2.set_title(
            f"After -- RMS {r.rms_after:.4f} px  (target: near 0)", fontsize=10
        )
        ax2.axvline(0, color="black", lw=0.5)
        self.canvas_trace.refresh()

    def _plot_residual(self, r: SmileAnalysis) -> None:
        self.canvas_resid.clear()
        fig = self.canvas_resid.fig

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        sig = robust_sigma(r.residuals)

        for i, ln in enumerate(r.used_lines):
            color = _COLORS[i % len(_COLORS)]
            mask = r.meas_id == i
            if np.any(mask):
                ax1.scatter(
                    r.meas_y[mask], r.residuals[mask],
                    s=0.5, alpha=0.3, color=color,
                )

        ax1.axhline(0, color="black", lw=0.5)
        ax1.axhline(
            +2 * sig, color="red", lw=0.5, ls="--", label=f"+/-2 sigma = {2 * sig:.4f}"
        )
        ax1.axhline(-2 * sig, color="red", lw=0.5, ls="--")
        ax1.set_xlabel("Row (spatial)")
        ax1.set_ylabel("Residual (px)")
        cv_str = f"{r.cv_rms:.5f}" if np.isfinite(r.cv_rms) else "n/a"
        ax1.set_title(
            f"Residual vs Row  (RMS={r.fit_info['rms']:.5f}, "
            f"CV={cv_str}, n_used={r.fit_info['n_used']})",
            fontsize=9,
        )
        ax1.legend(fontsize=7)

        ax2.hist(
            r.residuals, bins=50, edgecolor="black", lw=0.4,
            color="steelblue", alpha=0.8,
        )
        ax2.axvline(0, color="black", lw=0.5)
        ax2.set_xlabel("Residual (px)")
        ax2.set_ylabel("Count")
        ax2.set_title("Distribution", fontsize=9)
        self.canvas_resid.refresh()

    def _plot_surface(self, r: SmileAnalysis) -> None:
        self.canvas_surf.clear()
        ax = self.canvas_surf.fig.add_subplot(111)

        n_rows, n_cols = r.frame_orig.shape
        yy = np.linspace(0, n_rows - 1, 100)
        xx = np.linspace(0, n_cols - 1, 100)
        XX, YY = np.meshgrid(xx, yy)
        S = r.surface(YY, XX)

        im = ax.pcolormesh(xx, yy, S, cmap="RdBu_r", shading="auto")
        self.canvas_surf.fig.colorbar(im, ax=ax, label="Shift (px)")

        ax.scatter(r.meas_x, r.meas_y, s=0.3, c="black", alpha=0.15)
        ax.set_xlabel("Column (spectral)")
        ax.set_ylabel("Row (spatial)")
        ax.set_title(
            f"Smile Shift Surface  "
            f"[deg_y={r.surface.deg_y}, deg_x={r.surface.deg_x}]  "
            f"(clamped outside line range)",
            fontsize=10,
        )
        self.canvas_surf.refresh()


# =====================================================================
#  SECTION 4 -- Standalone launcher
# =====================================================================

if __name__ == "__main__":
    path = (
        sys.argv[1]
        if len(sys.argv) > 1 and os.path.isfile(sys.argv[1])
        else None
    )

    # CLI mode (no Qt).
    if not HAVE_QT:
        if path is None:
            sys.exit(
                "PyQt5 not available. CLI mode requires an image path:\n"
                "  python smile_module.py image.png"
            )

        # Demonstrate SmileCorrector API in CLI mode.
        corrector = SmileCorrector(deg_y=2, deg_x=1, interp="cubic")
        corrector.fit(load_frame(path), verbose=True)
        print()
        print(corrector.summary())
        cal_path = os.path.splitext(path)[0] + "_smile_model.npz"
        corrector.save(cal_path)
        sys.exit(0)

    # GUI mode.
    from PyQt5.QtWidgets import QApplication, QMainWindow

    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle(f"Smile Correction v{__version__}")
    window.resize(1150, 780)

    widget = SmileCorrectionWidget()
    window.setCentralWidget(widget)
    if path:
        widget.load_file(path)

    widget.analysisComplete.connect(
        lambda r: window.setWindowTitle(
            f"Smile Correction -- RMS {r.rms_before:.3f} -> {r.rms_after:.3f} px "
            f"(CV {r.cv_rms:.3f})"
        )
    )

    window.show()
    sys.exit(app.exec_())
