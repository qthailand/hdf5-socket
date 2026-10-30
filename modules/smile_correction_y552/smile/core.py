# smile/core.py
"""Core utilities, line detection, and sub-pixel tracing."""
import numpy as np
import cv2
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, List
from scipy.signal import find_peaks

__all__ = [
    "load_frame", "save_frame",
    "robust_sigma", "auto_sat_level", "estimate_noise",
    "Line", "detect_lines", "trace_line", "straightness_rms",
]

# Interpolation registry
_INTERP = {
    "linear":  (cv2.INTER_LINEAR,   0, 1),
    "cubic":   (cv2.INTER_CUBIC,    1, 2),
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