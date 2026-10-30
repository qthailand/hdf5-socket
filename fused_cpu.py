# -*- coding: utf-8 -*-
"""Fused CPU path: Smile + flip + lambda binning in one Numba pass, from the same calibration data.

hsi's CPU path (``modules.preprocessing.cpu``) remaps the whole frame with ``cv2.remap``
(``INTER_CUBIC``: a 4 x 4 kernel per pixel, all columns), writes it, and the binner reads it
again. The Smile models are horizontal only (``map_y`` is the row itself, a flip only reverses
it), so per output pixel only 4 horizontal taps matter - exactly what hsi's CuPy kernel
computes. This module does the same on the CPU and bins on the fly, for the columns the
binner uses only:

    dark + hot pixel  -> hsi code unchanged (cv2.subtract, DarkProcessor._apply_hot_pixel)
    flip + Smile      -> fixed-point maps from hsi ``prepare_maps`` (same map_x, same 1/32-pixel
                         grid) and OpenCV's own weight table (hsi ``horizontal_table``),
                         4 taps, border 0, rounded and clamped to uint16 like cv2.remap
    binning           -> the binner's own ``iL``/``iR``/``w`` and rounding ``(s + w//2) // w``

The recorded calibration (map_x, dark, wavelength fit) is used as stored, never modified.
:class:`FusedCPU` checks its first frame of every configuration against hsi's CPU path and
must agree within 1 count (the tolerance hsi applies to its CUDA backends); otherwise the
caller falls back to hsi's path.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from numba import njit, prange

from modules.preprocessing.maps import horizontal_table, prepare_maps

logger = logging.getLogger(__name__)


@njit(parallel=True, cache=True)
def _smile_bin(src, xy, frac, weights, iL, iR, w, out):
    """out[r, k] = binned( Smile(src)[r, iL[k]:iR[k]] ), Smile = 4/2-tap horizontal interpolation."""
    height, width = src.shape
    taps = weights.shape[1]
    off = 1 if taps == 4 else 0
    n_bins = iL.shape[0]
    for r in prange(height):
        for k in range(n_bins):
            s = 0
            for j in range(iL[k], iR[k]):
                x = np.int64(xy[r, j, 0])
                y = np.int64(xy[r, j, 1])
                f = np.int64(frac[r, j]) & 31
                v = np.float32(0.0)
                if 0 <= y < height:
                    for t in range(taps):
                        sx = x + t - off
                        if 0 <= sx < width:
                            v += np.float32(src[y, sx]) * weights[f, t]
                if v < 0.0:
                    v = np.float32(0.0)
                elif v > 65535.0:
                    v = np.float32(65535.0)
                s += np.int64(np.rint(v))
            ww = w[k]
            val = (s + (ww // 2)) // ww
            out[r, k] = 65535 if val > 65535 else val


@njit(parallel=True, cache=True)
def _flip_bin(src, rows, iL, iR, w, out):
    """Binning without Smile; output row ``r`` reads source row ``rows[r]`` (reversed = the flip).

    The row order is an array: a row index assigned under a condition inside ``prange`` is
    not reliably private to each iteration.
    """
    height = src.shape[0]
    n_bins = iL.shape[0]
    for r in prange(height):
        row = rows[r]
        for k in range(n_bins):
            s = 0
            for j in range(iL[k], iR[k]):
                s += np.int64(src[row, j])
            ww = w[k]
            val = (s + (ww // 2)) // ww
            out[r, k] = 65535 if val > 65535 else val


class FusedCPU:
    """Per-worker fused path (holds buffers; use from one thread)."""

    name = 'cpu_fused'

    def __init__(self) -> None:
        self._buffer = None
        self._maps_key = None
        self._maps = None
        self._checked: set = set()
        self.fallback_reason = ''

    @staticmethod
    def supports(frame, calibration) -> bool:
        return frame.dtype == np.uint16 and (calibration is None or calibration.interp in ('linear', 'cubic'))

    def _smile_maps(self, calibration, inverted):
        key = (id(calibration), bool(inverted))
        if key != self._maps_key:
            _, _, fixed, fractions, _ = prepare_maps(calibration, inverted)
            if fractions is not None and np.any(np.asarray(fractions) >> 5):
                raise ValueError("Smile map has vertical fractions: the fused path is horizontal-only")
            table = horizontal_table(calibration.interp).astype(np.float32)
            frac = np.ascontiguousarray(fractions if fractions is not None else np.zeros(fixed.shape[:2], np.uint16))
            self._maps = (np.ascontiguousarray(fixed), frac, np.ascontiguousarray(table))
            self._maps_key = key
        return self._maps

    def process_binned(self, frame, dark_processor, binner, calibration=None, inverted=False, profiler=None):
        """``(slit, n_bins)`` uint16, same contract as hsi ``PreprocessingRuntime.process_binned``."""
        reference = getattr(dark_processor, 'dark_reference', dark_processor)
        enabled = bool(getattr(dark_processor, '_dark_enabled', False))
        dark = getattr(reference, '_dark_mean', None)
        with _span(profiler, 'dark'):
            if enabled and dark is not None and dark.shape == frame.shape:
                if self._buffer is None or self._buffer.shape != frame.shape:
                    self._buffer = np.empty_like(frame)
                corrected = cv2.subtract(frame, dark, dst=self._buffer)
            else:
                corrected = frame
        with _span(profiler, 'hot_pixel'):
            if enabled and getattr(reference, '_hot_coords', None) is not None:
                if corrected is frame:
                    corrected = frame.copy()
                corrected = reference._apply_hot_pixel(corrected)
        out = np.empty((frame.shape[0], binner.n_bins), np.uint16)
        with _span(profiler, 'smile_bin'):
            if calibration is None:
                rows = np.arange(frame.shape[0], dtype=np.int64)
                _flip_bin(corrected, rows[::-1].copy() if inverted else rows, binner.iL, binner.iR, binner.w, out)
            else:
                xy, frac, table = self._smile_maps(calibration, inverted)
                _smile_bin(corrected, xy, frac, table, binner.iL, binner.iR, binner.w, out)
        return out

    def verify(self, frame, dark_processor, binner, calibration, inverted, runtime) -> int:
        """Max difference (counts) against hsi's CPU path for this configuration (checked once)."""
        key = (frame.shape, id(binner), id(calibration), bool(inverted),
               id(getattr(dark_processor, 'dark_reference', dark_processor)),
               bool(getattr(dark_processor, '_dark_enabled', False)))
        if key in self._checked:
            return 0
        ours = self.process_binned(frame, dark_processor, binner, calibration, inverted)
        theirs, _, _ = runtime.process_binned(frame, dark_processor, binner, calibration=calibration,
                                              inverted=inverted)
        error = int(np.max(np.abs(ours.astype(np.int32) - theirs.astype(np.int32)))) if ours.size else 0
        if error > 1:
            raise RuntimeError(f"fused CPU differs from hsi CPU by {error} counts (allowed 1)")
        self._checked.add(key)
        logger.info("cpu_fused verified against hsi CPU: maximum error=%d count", error)
        return error


class _NoSpan:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _span(profiler, name):
    return profiler.time(name) if profiler is not None else _NoSpan()


class FusedRuntime:
    """Drop-in for hsi ``PreprocessingRuntime`` that runs :class:`FusedCPU` on supported frames.

    Every new configuration (frame shape, dark, Smile model, flip, binner) is first computed by
    both paths; a difference above 1 count switches this runtime to hsi's CPU path for good
    (``fallback_reason`` says why). Unsupported input (not uint16, or Smile interpolation other
    than linear/cubic) uses hsi's CPU path for that frame.
    """

    def __init__(self) -> None:
        from modules.preprocessing import PreprocessingRuntime
        self._hsi = PreprocessingRuntime('cpu')
        self._fused = FusedCPU()
        self._failed = False
        self.fallback_reason = ''
        self.requested = 'cpu_fused'

    @property
    def backend_name(self) -> str:
        return 'cpu' if self._failed else 'cpu_fused'

    def process_binned(self, frame, dark_processor, binner, calibration=None, inverted=False,
                       snapshot=False, full=False, profiler=None):
        if self._failed or snapshot or full or not self._fused.supports(frame, calibration):
            return self._hsi.process_binned(frame, dark_processor, binner, calibration=calibration,
                                            inverted=inverted, snapshot=snapshot, full=full, profiler=profiler)
        try:
            self._fused.verify(frame, dark_processor, binner, calibration, inverted, self._hsi)
        except Exception as exc:
            self._failed = True
            self.fallback_reason = f"cpu_fused: {exc}"
            logger.warning("Fused CPU path disabled, using hsi CPU: %s", exc)
            return self._hsi.process_binned(frame, dark_processor, binner, calibration=calibration,
                                            inverted=inverted, profiler=profiler)
        with _span(profiler, 'preprocess'):
            binned = self._fused.process_binned(frame, dark_processor, binner, calibration, inverted, profiler)
        return binned, None, None
