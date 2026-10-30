"""Cached Numba wavelength binning; arithmetic and buffers preserve the legacy path.

Data conventions:
    Input spectrograms are ``(frames, spatial, spectral)`` (any dtype, converted
    to uint16); spectral pixel ``j`` has wavelength ``m * j + c`` nm.  Output is
    ``(frames, spatial, n_bins)`` uint16, each bin being the integer-rounded (and
    clamped to 65535) mean of the spectral pixels inside the bin edges.  Bin
    edges are ``step`` nm apart starting at ``c`` nm; empty bins are dropped.
    A wavelength range keeps a contiguous run of those bins (``range_bins`` and
    ``subset``): the kept bins are bit-identical to the same bins of the full
    binner, because the grid is never refitted for the range.

Threading:
    Pure computation, no Qt.  ``transform`` calls Numba's process-wide
    ``set_num_threads`` when ``threads`` is set, so concurrent binners with
    different thread counts influence each other.

Example:
    >>> import numpy as np
    >>> from modules.spectral_processing.binning import LambdaBinner
    >>> binner = LambdaBinner(m=0.29, c=400.0, step=2.0).fit(100)
    >>> out = binner.transform(np.ones((1, 4, 100), dtype=np.uint16))
    >>> out.dtype, out.shape[:2]
    (dtype('uint16'), (1, 4))
"""
# -*- coding: utf-8 -*-
import os
from typing import Optional, Tuple

import numpy as np
from numba import njit, prange, set_num_threads

#: Tolerance (nm) of :meth:`LambdaBinner.range_bins`: a bound this close to a pixel's
#: wavelength still includes that pixel.
RANGE_EPS_NM = 1e-6


@njit(parallel=True, fastmath=True, cache=True)
def _bin_average_uint16_numba(X, iL, iR, w):
    b, h, v = X.shape
    n_bins = iL.shape[0]
    Xf = X.reshape(b * h, v)
    out = np.zeros((b * h, n_bins), dtype=np.uint16)

    for r in prange(b * h):
        for k in range(n_bins):
            il = iL[k]; ir = iR[k]; ww = w[k]
            s = 0
            for j in range(il, ir):
                s += Xf[r, j]
            val = (s + (ww // 2)) // ww
            if val > 65535:
                val = 65535
            out[r, k] = val
    return out.reshape(b, h, n_bins)


class LambdaBinner:
    """Average spectral pixels into uniform wavelength bins.

    Call :meth:`fit` (or :meth:`fit_from`) once for a given spectral width, then
    :meth:`transform` any number of spectrograms of that width.

    Attributes:
        m: Slope of the pixel -> wavelength fit (nm per spectral pixel).
        c: Intercept (nm) - the wavelength of spectral pixel 0.
        step: Bin width in nm.
        threads: Numba thread count used by :meth:`transform`, or ``None``/<=0
            to leave the Numba setting untouched.
        v_: Spectral width the binner was fitted to (``None`` before fit).
        iL, iR: Per-bin first (inclusive) and last (exclusive) spectral pixel
            index, int32 (``None`` before fit).
        w: Per-bin pixel count, int32.
        source_bins: ``(k0, k1, n_full)`` for a binner made by :meth:`subset`
            (bins ``k0:k1`` of a fitted binner with ``n_full`` bins), else ``None``.

    Thread-safety:
        Not thread-safe while being re-fitted; read-only use after ``fit`` is fine.
        Never change a fitted binner in place: GPU backends cache its tables by
        object identity. :meth:`subset` returns a new object.
    """

    def __init__(self, m: float, c: float, step: float = 2.0,
                 threads: Optional[int] = None):
        """Store the calibration and binning parameters (no computation yet).

        Args:
            m: Wavelength slope in nm per spectral pixel (must be non-zero).
            c: Wavelength in nm of spectral pixel 0.
            step: Bin width in nm.
            threads: Numba threads for :meth:`transform`; ``None`` keeps the default.
        """
        self.m = float(m)
        self.c = float(c)
        self.step = float(step)
        self.threads = threads

        self.v_: Optional[int] = None
        self.iL: Optional[np.ndarray] = None
        self.iR: Optional[np.ndarray] = None
        self.w: Optional[np.ndarray] = None
        self._lambda_out: Optional[np.ndarray] = None
        self.source_bins: Optional[Tuple[int, int, int]] = None

    # ---------- precompute ----------
    def _precompute_bins(self, v: int):
        m, c, step = self.m, self.c, self.step
        lam_first = c
        lam_last = m * (v - 1) + c

        n_edges = int(np.round((lam_last - lam_first) / step)) + 1
        edges = np.linspace(lam_first, lam_last, n_edges, dtype=np.float32)

        if edges.size < 2:
            edges = np.array([lam_first, lam_last], dtype=np.float32)

        L = edges[:-1]
        R = edges[1:]

        iL = np.ceil((L - c) / m).astype(np.int32)
        iR = np.floor((R - c) / m).astype(np.int32) + 1

        iL = np.clip(iL, 0, v)
        iR = np.clip(iR, 0, v)

        w = (iR - iL).astype(np.int32)
        ok = w > 0

        return iL[ok], iR[ok], w[ok], ((L[ok] + R[ok]) * 0.5).astype(np.float32)

    # ---------- public API ----------
    def fit(self, v: int) -> "LambdaBinner":
        """Precompute the bin edges for a spectral width.

        Args:
            v: Number of spectral pixels (length of the last axis of the data).

        Returns:
            LambdaBinner: ``self``, to allow chaining.
        """
        self.v_ = int(v)
        self.iL, self.iR, self.w, self._lambda_out = self._precompute_bins(self.v_)
        return self

    def fit_from(self, spectrogram: np.ndarray) -> "LambdaBinner":
        """Fit to the last-axis length of ``spectrogram``.

        Args:
            spectrogram: Array whose last axis is the spectral axis.

        Returns:
            LambdaBinner: ``self``.
        """
        return self.fit(int(spectrogram.shape[-1]))

    @property
    def lambda_out(self) -> np.ndarray:
        """Bin-centre wavelengths in nm, float32, shape ``(n_bins,)``.

        Raises:
            AssertionError: if :meth:`fit` has not been called.
        """
        assert self._lambda_out is not None
        return self._lambda_out

    @property
    def n_bins(self) -> int:
        """Number of non-empty output bins.

        Raises:
            AssertionError: if :meth:`fit` has not been called.
        """
        assert self.iL is not None
        return int(self.iL.size)

    def range_bins(self, lo_nm: float, hi_nm: float) -> Tuple[int, int]:
        """The bins whose every spectral pixel lies inside ``[lo_nm, hi_nm]``.

        A pixel's wavelength is its calibrated centre ``m * j + c``; a bound within
        :data:`RANGE_EPS_NM` of a pixel's wavelength counts as including it. The test is
        float64 arithmetic on the integer pixel indices with Python-float operands, so it
        decides the same on NumPy 1.x and 2.x.

        Args:
            lo_nm: Shortest wavelength to keep (nm).
            hi_nm: Longest wavelength to keep (nm).

        Returns:
            tuple: ``(k0, k1)``: bins ``k0:k1`` of this binner, a contiguous run;
            ``(0, 0)`` when no bin fits (or ``m`` is not positive).

        Raises:
            AssertionError: if :meth:`fit` has not been called.
        """
        assert self.iL is not None
        if not self.m > 0 or self.iL.size == 0:
            return 0, 0
        lo = float(lo_nm) - RANGE_EPS_NM
        hi = float(hi_nm) + RANGE_EPS_NM
        first = self.m * self.iL.astype(np.float64) + self.c
        last = self.m * (self.iR.astype(np.float64) - 1.0) + self.c
        kept = np.flatnonzero((first >= lo) & (last <= hi))
        if kept.size == 0:
            return 0, 0
        k0, k1 = int(kept[0]), int(kept[-1]) + 1
        assert k1 - k0 == kept.size, "bins inside a wavelength range must be contiguous"
        return k0, k1

    def subset(self, k0: int, k1: int) -> "LambdaBinner":
        """A new binner that computes only bins ``k0:k1`` of this one.

        It keeps this binner's ``m``, ``c``, ``step``, ``threads`` and fitted width
        ``v_`` (it bins the same full-width frames), so its output equals
        ``self.transform(x)[..., k0:k1]`` bit for bit and its :attr:`lambda_out` is
        the same float32 values. The tables are int32 C-contiguous copies (the same
        Numba signature: no recompilation).

        Args:
            k0: First bin to keep.
            k1: One past the last bin to keep.

        Returns:
            LambdaBinner: New fitted binner with ``source_bins`` set.

        Raises:
            AssertionError: if :meth:`fit` has not been called.
            ValueError: if ``k0:k1`` is empty or outside ``0:n_bins``.
        """
        assert self.iL is not None
        n = self.n_bins
        k0, k1 = int(k0), int(k1)
        if not 0 <= k0 < k1 <= n:
            raise ValueError(f"bin slice {k0}:{k1} is empty or outside 0:{n}")
        part = LambdaBinner(self.m, self.c, self.step, self.threads)
        part.v_ = self.v_
        part.iL = np.array(self.iL[k0:k1], dtype=np.int32)
        part.iR = np.array(self.iR[k0:k1], dtype=np.int32)
        part.w = np.array(self.w[k0:k1], dtype=np.int32)
        part._lambda_out = np.array(self._lambda_out[k0:k1], dtype=np.float32)
        offset, total = (0, n) if self.source_bins is None else (self.source_bins[0], self.source_bins[2])
        part.source_bins = (offset + k0, offset + k1, total)
        return part

    def warmup(self) -> None:
        """Trigger Numba compilation/cache load with a tiny dummy input.

        Does nothing before :meth:`fit`.  Call it off the real-time path.
        """
        if self.v_ is None:
            return
        dummy = np.zeros((1, 8, self.v_), dtype=np.uint16)
        _ = _bin_average_uint16_numba(dummy, self.iL, self.iR, self.w)

    def transform(self, spectrogram: np.ndarray) -> np.ndarray:
        """Bin the spectral axis.

        Args:
            spectrogram: ``(frames, spatial, v)`` array with ``v`` equal to the
                fitted width; converted to a C-contiguous uint16 array (no copy
                if it already is one).

        Returns:
            numpy.ndarray: New uint16 array ``(frames, spatial, n_bins)`` owned
            by the caller.

        Raises:
            AssertionError: if not fitted or the last axis differs from the fit.
                Inputs that are not 3-D fail inside the Numba kernel.
        """
        assert self.v_ is not None
        assert spectrogram.shape[-1] == self.v_
        X = np.ascontiguousarray(spectrogram, dtype=np.uint16)
        if self.threads is not None and self.threads > 0:
            set_num_threads(int(self.threads))
        return _bin_average_uint16_numba(X, self.iL, self.iR, self.w)

    def transform_to_cube(self, spectrogram: np.ndarray):
        """Bin and return the result band-first.

        Args:
            spectrogram: As for :meth:`transform`.

        Returns:
            tuple: ``(cube, wavelengths)`` where ``cube`` is a *view* (no copy,
            not C-contiguous) of shape ``(n_bins, spatial, frames)`` (band-first,
            the layout the ENVI/GeoTIFF writers in ``modules.cube_io`` expect)
            and ``wavelengths`` is :attr:`lambda_out` (nm, float32).
        """
        out = self.transform(spectrogram)
        return out.transpose(2, 1, 0), self.lambda_out