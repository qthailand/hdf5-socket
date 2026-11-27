# -*- coding: utf-8 -*-
"""Regression results on the belt map: value codes, summary statistics, objects, colour map.

The pipeline's belt map is ``uint8`` (class ids; 255 = lost line). A regression model fills it
with *codes*: ``0 .. 253`` = the predicted value quantised over the model's valid range
(:func:`encode`), :data:`BACKGROUND_CODE` = no object (outside the object mask, or a pixel
without a valid spectrum) and 255 = lost line, as for classes. The ring, the lost lines, the
stride and the 256-bin counts of the pipeline work unchanged: the counts are a histogram of the
values. Exact statistics (sum, sum of squares) and the float values themselves travel beside it
(``Snapshot.value_stats`` / ``value_canvas``).

Pure numpy (no Qt): the GUI and the tests share it.
"""
from __future__ import annotations

import numpy as np

LEVELS = 254                          # codes 0 .. 253 carry a value
BACKGROUND_CODE = 254                 # no object here (the belt, an invalid spectrum)
LOST_CODE = 255                       # = pipeline.LOST_CLASS


def encode(values: np.ndarray, valid: np.ndarray, value_range: tuple) -> np.ndarray:
    """``uint8`` codes of ``values`` (``valid`` False -> :data:`BACKGROUND_CODE`); values outside
    ``value_range`` take the end codes 0 / 253 (the exact values stay in the value canvas)."""
    lo, hi = float(value_range[0]), float(value_range[1])
    scale = (LEVELS - 1) / (hi - lo)
    with np.errstate(invalid='ignore'):
        codes = np.rint((np.asarray(values, np.float32) - np.float32(lo)) * np.float32(scale))
    np.nan_to_num(codes, copy=False, nan=0.0, posinf=LEVELS - 1, neginf=0.0)
    np.clip(codes, 0, LEVELS - 1, out=codes)
    out = codes.astype(np.uint8)
    out[~np.asarray(valid, bool)] = BACKGROUND_CODE
    return out


def code_values(value_range: tuple) -> np.ndarray:
    """``(LEVELS,)`` value of every code (its bin centre)."""
    lo, hi = float(value_range[0]), float(value_range[1])
    return lo + np.arange(LEVELS, dtype=np.float64) * (hi - lo) / (LEVELS - 1)


def line_stats(values: np.ndarray, valid: np.ndarray, value_range: tuple) -> np.ndarray:
    """``[object pixels, sum, sum of squares, pixels outside value_range]`` of one line (float64)."""
    v = np.asarray(values, np.float64)[np.asarray(valid, bool)]
    if not v.size:
        return np.zeros(4)
    lo, hi = value_range
    return np.array([v.size, v.sum(), (v * v).sum(), np.count_nonzero((v < lo) | (v > hi))], np.float64)


def summary(stats: np.ndarray | None, counts: np.ndarray | None, value_range: tuple, background: int = 0) -> dict:
    """Summary of the scanned objects since the batch reset.

    Args:
        stats: ``line_stats`` summed over the lines (exact mean and SD).
        counts: ``(256,)`` pixels per code (median and percentiles, to the code's resolution).
        background: pixels of :data:`BACKGROUND_CODE` (for the object share).

    Returns:
        ``{'n', 'mean', 'sd', 'median', 'p10', 'p90', 'out_of_range', 'object_share'}`` (NaN where
        there is nothing yet).
    """
    out = {'n': 0, 'mean': np.nan, 'sd': np.nan, 'median': np.nan, 'p10': np.nan, 'p90': np.nan,
           'out_of_range': 0.0, 'object_share': 0.0}
    n = 0 if stats is None else float(stats[0])
    if n > 0:
        mean = stats[1] / n
        out.update(n=int(n), mean=float(mean),
                   sd=float(np.sqrt(max(stats[2] / n - mean * mean, 0.0))),
                   out_of_range=float(stats[3] / n))
    if counts is not None:
        hist = np.asarray(counts[:LEVELS], np.float64)
        total = hist.sum()
        if total > 0:
            out['median'], out['p10'], out['p90'] = (float(v) for v in
                                                      histogram_quantiles(hist, value_range, (0.5, 0.1, 0.9)))
        everything = total + background
        out['object_share'] = float(total / everything) if everything else 0.0
    return out


def histogram_quantiles(hist: np.ndarray, value_range: tuple, quantiles) -> np.ndarray:
    """Quantiles of the values a code histogram counts, interpolated inside the code bins."""
    hist = np.asarray(hist, np.float64)
    lo, hi = float(value_range[0]), float(value_range[1])
    step = (hi - lo) / (LEVELS - 1)
    edges = lo + (np.arange(LEVELS + 1) - 0.5) * step               # bin edges around the code values
    cumulative = np.concatenate([[0.0], np.cumsum(hist)])
    total = cumulative[-1]
    out = []
    for q in np.atleast_1d(quantiles):
        target = q * total
        k = int(np.clip(np.searchsorted(cumulative, target, side='left') - 1, 0, LEVELS - 1))
        inside = hist[k]
        frac = 0.5 if inside <= 0 else (target - cumulative[k]) / inside
        out.append(float(np.clip(edges[k] + frac * step, lo, hi)))
    return np.asarray(out)


def objects(values: np.ndarray, min_pixels: int = 20, max_objects: int = 50) -> list:
    """Connected object regions of a value map (NaN = no object), largest first.

    Returns:
        ``[{'pixels', 'mean', 'median', 'min', 'max', 'rows': (first, last), 'columns': (first, last)}]``
        for regions of at least ``min_pixels`` map pixels (8-connected).
    """
    from scipy import ndimage
    values = np.asarray(values, np.float32)
    present = np.isfinite(values)
    if not present.any():
        return []
    labels, count = ndimage.label(present, structure=np.ones((3, 3), bool))
    if count == 0:
        return []
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = [k for k in np.argsort(sizes[1:])[::-1] + 1 if sizes[k] >= min_pixels][:max_objects]
    slices = ndimage.find_objects(labels)
    found = []
    for k in keep:
        sl = slices[k - 1]
        inside = labels[sl] == k
        v = values[sl][inside].astype(np.float64)
        found.append({'pixels': int(v.size), 'mean': float(v.mean()), 'median': float(np.median(v)),
                      'min': float(v.min()), 'max': float(v.max()),
                      'rows': (int(sl[0].start), int(sl[0].stop - 1)),
                      'columns': (int(sl[1].start), int(sl[1].stop - 1))})
    return found


# ----------------------------------------------------------------------------- colours ---

#: perceptually ordered dry -> wet colour map (anchor colours, linearly interpolated):
#: from a sand brown through green to a deep blue
_ANCHORS = np.array([
    (0.00, (140, 81, 10)),
    (0.20, (216, 179, 101)),
    (0.40, (246, 232, 195)),
    (0.55, (199, 234, 229)),
    (0.75, (90, 180, 172)),
    (0.90, (1, 102, 94)),
    (1.00, (0, 60, 48)),
], dtype=object)


def colormap(t: np.ndarray) -> np.ndarray:
    """RGB ``uint8`` of positions ``t`` in [0, 1] (clipped) on the dry -> wet colour map."""
    t = np.clip(np.asarray(t, np.float64), 0.0, 1.0)
    stops = np.array([a for a, _ in _ANCHORS], np.float64)
    colours = np.array([c for _, c in _ANCHORS], np.float64)
    out = np.empty(t.shape + (3,), np.float64)
    for ch in range(3):
        out[..., ch] = np.interp(t, stops, colours[:, ch])
    return np.rint(out).astype(np.uint8)


def value_lut(value_range: tuple, display_range: tuple, background=(0, 0, 0), lost=(255, 0, 255)) -> np.ndarray:
    """``(256, 3)`` belt-map LUT: codes over ``value_range`` coloured over ``display_range``."""
    lo, hi = float(display_range[0]), float(display_range[1])
    t = (code_values(value_range) - lo) / max(hi - lo, 1e-12)
    lut = np.zeros((256, 3), np.uint8)
    lut[:LEVELS] = colormap(t)
    lut[BACKGROUND_CODE] = background
    lut[LOST_CODE] = lost
    return lut


def colour_hex(value: float, display_range: tuple) -> str:
    lo, hi = float(display_range[0]), float(display_range[1])
    r, g, b = colormap((value - lo) / max(hi - lo, 1e-12))
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"
