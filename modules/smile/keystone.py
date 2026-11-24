"""Keystone (along-slit) correction joined with the smile into one 2-D geometry map.

The smile is measured from an emission-lamp frame (lines run along the slit); the keystone from a
*line target* frame: a halogen-lit target whose dark (or bright) bars cross the slit, so each bar
is one slit position and should be a horizontal line on the detector. Both are solved together
and applied as ONE remap (one interpolation):

    source_x = X + dx(X, Y)      smile:     dx(X, y_ref) = 0  (the wavelength axis of the reference row is kept)
    source_y = Y + dy(X, Y)      keystone:  dy(x_ref, Y) = 0  (every band is co-registered onto the slit
                                                                as seen at the reference column)

``(X, Y)`` are destination (corrected) pixels. Both shifts are low-order 2-D polynomials whose
basis enforces the anchors exactly (no ``y**0`` term in dx, no ``(x - x_ref)**0`` term in dy).
They are fitted jointly, alternating: a lamp point measured at raw row ``y`` belongs to the
destination row ``Y = y - dy(X, Y)``, a bar point measured at raw column ``x`` to the destination
column ``X = x - dx(X, Y)``. Outside the measured span the shifts hold the edge value along the
spectrum (as the smile-only model does) and continue linearly along the slit.

Accuracy notes:
    * Line and bar centres are intensity centroids in a window that never reaches a neighbour
      (robust to the broad, double-lobed lamp lines of some spectrographs).
    * Uneven illumination along the slit that changes with wavelength biases a bar centre
      differently per column, which looks like keystone. Pass a *white* frame (same light and
      distance, target removed) to divide it out; without one a slow trend along the slit is
      removed with a wide running mean.
    * The rotation of the target in its plane does not matter (every bar still crosses the slit
      at one point); its focus and working distance do.

Data conventions:
    Frames are 2-D ``(slit rows, spectral columns)``, zero-based pixel centres, the same
    orientation / hardware bin / size as the frames the model will be applied to (dark-subtracted
    when the live pipeline subtracts dark before the remap). Maps are OpenCV destination ->
    source ``float32`` arrays.

Threading:
    Pure NumPy / OpenCV, no shared state: any thread.

Example:
    >>> from modules.smile import SmileCorrector                          # doctest: +SKIP
    >>> c = SmileCorrector().fit(lamp)                                     # doctest: +SKIP
    >>> c.fit_keystone(target, white=white)                                # doctest: +SKIP
    >>> c.save('smile_model_bin2.npz', camera_setting)                     # doctest: +SKIP
"""
from __future__ import annotations

import copy
import math
import datetime as _dt
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks

from .core import _INTERP, _centroid_row, auto_sat_level, estimate_noise, robust_sigma, trace_line
from .debug_log import logged_analysis

__all__ = [
    "KEYSTONE_SCHEMA", "TargetLine", "GeometrySurface", "GeometryModel", "KeystoneAnalysis",
    "normalize_target", "detect_polarity", "frames_same_scene", "trace_target_lines", "bar_coverage",
    "coverage_warnings", "lamp_points_from_analysis", "fit_geometry", "SplineSurface", "TensorSurface",
    "recover_bar_gaps", "keystone_candidates", "select_keystone_fitter",
    "analyze_keystone", "analyze_keystone_from_smile", "row_margin_from_map", "valid_rect", "keystone_report",
]

KEYSTONE_SCHEMA = "hsi-smile-keystone/1"
#: Target bars must span at least this share of the slit (rows).
MIN_SLIT_COVERAGE = 0.5
#: A bar is kept when it is followed over at least this share of the columns (first to last traced
#: column). 0.25 keeps the slit-end bars of an optic with a curved focus, which are sharp over only
#: part of the spectrum; where they are missing the per-zone coverage warning still reports it.
MIN_BAR_COVERAGE = 0.25
#: A lamp line enters the joint fit when it is traced over this share of the rows (and spans y_ref).
MIN_LAMP_COVERAGE = 0.5
#: Re-traced straightness (px RMS of the median line / bar) above which the result carries a warning.
RETRACE_WARN_PX = 0.25
#: Highest keystone degree along the slit tried by ``dy_ydeg="auto"``.
MAX_DY_YDEG = 4
#: Keystone model: ``"auto"`` (default: the candidate fitter with the fewest held-out bar errors over
#: 1 px, see :func:`select_keystone_fitter`), a candidate name (:func:`keystone_candidates`),
#: ``"spline"`` (= ``"anchored_bspline"``) or ``"poly"`` (the earlier polynomial with its slit degree
#: chosen by leave-one-bar-out). A low-order polynomial cannot bend where the bars are lost (a weak /
#: defocused part of the spectrum) and extrapolates them along a straight line; on a real target
#: (bin 2, 1024 x 1280) that was 3-7 px off the gap-recovered bar centres, which the spline follows.
DEFAULT_DY_MODEL = "auto"
#: Automatic fitter choice: folds of whole bars held out, and the error counted (px).
SELECT_FOLDS = 3
SELECT_ERROR_PX = 1.0
#: Spline knot spacing (px) along the spectrum / the slit, and the most intervals per axis.
SPLINE_KNOT_PX = 64.0
SPLINE_MAX_INTERVALS = (20, 16)                 # (along the spectrum, along the slit)
#: Spline smoothness: weight of the second-difference penalty per 10 000 units of point weight.
SPLINE_REG = 0.04
#: Weight of a gap-recovered bar centre in the keystone fit (a tracer centre weighs 1).
RECOVERED_WEIGHT = 0.35
#: Smallest row spacing of the corrected map (d map_y / d Y) before a warning (a fold is <= 0).
ROW_JACOBIAN_WARN = 0.5


# ----------------------------------------------------------------------------------------------- basis
class GeometrySurface:
    """``f(X, Y) = sum_ij C[i, j] * yn**ypows[i] * xn**xpows[j]`` with ``xn = (X - x0) / sx``,
    ``yn = (Y - y0) / sy``.

    Evaluation outside the measured box: ``X`` is clamped to ``[xlo, xhi]`` (edge value held along
    the spectrum); ``Y`` outside ``[ylo, yhi]`` continues linearly from the edge (value + slope),
    so a magnification term keeps scaling the rows instead of a cubic blowing up
    (``y_rule="linear"``), or is clamped too (``y_rule="hold"``, the rule of a stored smile-only
    :class:`~modules.smile.surface.Surface2D`).
    """

    #: Robust weights in the keystone solvers: Huber with rejection (the earlier polynomial behaviour).
    robust = "huber"

    def __init__(self, xpows, ypows, x0, sx, y0, sy, coef=None, box=None, y_rule="linear"):
        self.xpows = [int(v) for v in xpows]
        self.ypows = [int(v) for v in ypows]
        self.x0, self.sx, self.y0, self.sy = float(x0), float(sx), float(y0), float(sy)
        n = (len(self.ypows), len(self.xpows))
        self.C = np.zeros(n) if coef is None else np.asarray(coef, float).reshape(n)
        self.box = None if box is None else [float(v) for v in box]     # [xlo, xhi, ylo, yhi]
        if y_rule not in ("linear", "hold"):
            raise ValueError("y_rule must be 'linear' or 'hold'")
        self.y_rule = y_rule

    # -- basis
    def _bx(self, X, deriv=False):
        xn = (np.asarray(X, float).ravel() - self.x0) / self.sx
        return self._pow(xn, self.xpows, deriv) / (self.sx if deriv else 1.0)

    def _by(self, Y, deriv=False):
        yn = (np.asarray(Y, float).ravel() - self.y0) / self.sy
        return self._pow(yn, self.ypows, deriv) / (self.sy if deriv else 1.0)

    @staticmethod
    def _pow(t, pows, deriv):
        out = np.empty((t.size, len(pows)))
        for k, p in enumerate(pows):
            if deriv:
                out[:, k] = p * t ** (p - 1) if p > 0 else 0.0
            else:
                out[:, k] = t ** p
        return out

    def design(self, X, Y):
        bx, by = self._bx(X), self._by(Y)
        return (by[:, :, None] * bx[:, None, :]).reshape(len(bx), -1)

    def raw(self, X, Y):
        """Polynomial value at points, no box rule."""
        X, Y = np.broadcast_arrays(np.asarray(X, float), np.asarray(Y, float))
        return np.sum((self._by(Y) @ self.C) * self._bx(X), axis=1).reshape(X.shape)

    def fit(self, X, Y, values, weights=None):
        A = self.design(X, Y)
        v = np.asarray(values, float).ravel()
        if weights is not None:
            w = np.sqrt(np.asarray(weights, float).ravel())
            A, v = A * w[:, None], v * w
        n = A.shape[1]
        AtA = A.T @ A
        coef = np.linalg.solve(AtA + 1e-12 * max(1.0, np.trace(AtA) / n) * np.eye(n), A.T @ v)
        self.C = coef.reshape(self.C.shape)
        return self

    # -- evaluation with the box rule
    def _clamped(self, X, Y):
        X = np.asarray(X, float); Y = np.asarray(Y, float)
        if self.box is None:
            return X, Y, np.zeros_like(Y)
        Xc = np.clip(X, self.box[0], self.box[1])
        Yc = np.clip(Y, self.box[2], self.box[3])
        return Xc, Yc, (Y - Yc) if self.y_rule == "linear" else np.zeros_like(Y)

    def at(self, X, Y):
        """Value at points with the box rule."""
        X, Y = np.broadcast_arrays(np.asarray(X, float), np.asarray(Y, float))
        Xc, Yc, dY = self._clamped(X.ravel(), Y.ravel())
        bx = self._bx(Xc)
        f = np.sum((self._by(Yc) @ self.C) * bx, axis=1)
        f = f + np.sum((self._by(Yc, True) @ self.C) * bx, axis=1) * dY
        return f.reshape(X.shape)

    def grid(self, xs, ys):
        """Values on the grid ``ys`` (rows) x ``xs`` (columns), box rule, separable."""
        Xc, Yc, dY = self._clamped(np.asarray(xs, float), np.asarray(ys, float))
        bx = self._bx(Xc)
        by = self._by(Yc) + self._by(Yc, True) * np.asarray(dY, float).ravel()[:, None]
        return by @ self.C @ bx.T

    def to_json(self) -> dict:
        return dict(xpows=self.xpows, ypows=self.ypows, x0=self.x0, sx=self.sx, y0=self.y0, sy=self.sy,
                    coef=self.C.tolist(), box=self.box, y_rule=self.y_rule)

    @classmethod
    def from_json(cls, d: dict) -> "GeometrySurface":
        return cls(d["xpows"], d["ypows"], d["x0"], d["sx"], d["y0"], d["sy"], d["coef"], d.get("box"),
                   d.get("y_rule", "linear"))

    @classmethod
    def from_smile(cls, s) -> "GeometrySurface":
        """The same function as a smile-only :class:`~modules.smile.surface.Surface2D` ``s(y, x)``.

        ``Surface2D`` normalises ``Y = 2 (y - y0) / (y1 - y0) - 1`` (the same for ``x``) and clamps
        to ``[-1, 1]``: centre / half-range here, and a box that holds the edge value on both axes.
        """
        y0, y1 = (float(v) for v in s.y_rng)
        x0, x1 = (float(v) for v in s.x_rng)
        coef = np.asarray(s.coef, float).reshape(s.deg_y + 1, s.deg_x + 1)
        box = [x0, x1, y0, y1] if getattr(s, "clamp", True) else None
        return cls(range(0, s.deg_x + 1), range(0, s.deg_y + 1), 0.5 * (x0 + x1), max(1e-9, 0.5 * (x1 - x0)),
                   0.5 * (y0 + y1), max(1e-9, 0.5 * (y1 - y0)), coef, box, y_rule="hold")


def _clamped_knots(lo: float, hi: float, n: int) -> np.ndarray:
    br = np.linspace(float(lo), float(hi), int(n) + 1)
    return np.r_[np.repeat(br[0], 3), br, np.repeat(br[-1], 3)]


def _bspline_basis(v, t: np.ndarray) -> sparse.csr_matrix:
    return BSpline.design_matrix(np.asarray(v, float).ravel(), t, 3, extrapolate=True)


class SplineSurface:
    """Keystone ``f(X, Y) = sum_jk C[j, k] * By_j(Y) * (Bx_k(X) - Bx_k(x_ref))``: cubic B-splines.

    The spectral basis is differenced against its value at ``x_ref``, so ``f(x_ref, Y) = 0`` for
    every ``Y`` exactly, as the polynomial keystone. The coefficients are fitted by weighted least
    squares with a penalty on their second differences along both axes (smooth where the data are
    sparse, local where they are dense). Same interface as :class:`GeometrySurface` (``fit``,
    ``raw``, ``at``, ``grid``, ``box``, ``to_json``), and the same box rule: ``X`` clamped to the
    measured columns, ``Y`` continued linearly beyond the outer bars.
    """

    kind = "bspline"
    robust = "soft"

    def __init__(self, x_ref, tx, ty, reg: float = SPLINE_REG, coef=None, box=None, y_rule: str = "linear"):
        self.x_ref = float(x_ref)
        self.tx = np.asarray(tx, float); self.ty = np.asarray(ty, float)
        self.nbx, self.nby = len(self.tx) - 4, len(self.ty) - 4
        self.reg = float(reg)
        n = (self.nby, self.nbx)
        self.C = np.zeros(n) if coef is None else np.asarray(coef, float).reshape(n)
        self.box = None if box is None else [float(v) for v in box]
        if y_rule not in ("linear", "hold"):
            raise ValueError("y_rule must be 'linear' or 'hold'")
        self.y_rule = y_rule
        self._pen = None

    @classmethod
    def for_frame(cls, shape, x_ref: float, period: Optional[float] = None, reg: float = SPLINE_REG) -> "SplineSurface":
        """Uniform knots over the frame: ~``SPLINE_KNOT_PX`` apart (at most ``SPLINE_MAX_INTERVALS``),
        and never closer than three bar periods along the slit."""
        h, w = (int(v) for v in shape)
        nx = int(np.clip(round((w - 1) / SPLINE_KNOT_PX), 4, SPLINE_MAX_INTERVALS[0]))
        ny = int(np.clip(round((h - 1) / SPLINE_KNOT_PX), 4, SPLINE_MAX_INTERVALS[1]))
        if period and period > 0:
            ny = max(4, min(ny, int((h - 1) // (3.0 * period))))
        return cls(x_ref, _clamped_knots(0, w - 1, nx), _clamped_knots(0, h - 1, ny), reg)

    # -- basis
    def _bx(self, X) -> np.ndarray:
        return _bspline_basis(X, self.tx).toarray() - _bspline_basis([self.x_ref], self.tx).toarray()[0]

    def _by(self, Y, deriv: bool = False) -> np.ndarray:
        if not deriv:
            return _bspline_basis(Y, self.ty).toarray()
        Y = np.asarray(Y, float).ravel()
        return (_bspline_basis(Y + 0.5, self.ty) - _bspline_basis(Y - 0.5, self.ty)).toarray()

    def design(self, X, Y) -> sparse.csr_matrix:
        X = np.asarray(X, float).ravel(); Y = np.asarray(Y, float).ravel()
        n = X.size
        bx = self._bx(X)
        by = _bspline_basis(Y, self.ty)
        if by.nnz != 4 * n:                             # always 4 per row for a cubic; dense fallback
            byd = by.toarray()
            return sparse.csr_matrix((byd[:, :, None] * bx[:, None, :]).reshape(n, -1))
        iy = by.indices.reshape(n, 4); vy = by.data.reshape(n, 4)
        cols = (iy[:, :, None] * self.nbx + np.arange(self.nbx)[None, None, :]).ravel()
        vals = (vy[:, :, None] * bx[:, None, :]).ravel()
        rows = np.repeat(np.arange(n), 4 * self.nbx)
        return sparse.csr_matrix((vals, (rows, cols)), shape=(n, self.nby * self.nbx))

    def _penalty(self) -> np.ndarray:
        if self._pen is None:
            def d2(m):
                return sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(max(1, m - 2), m))
            dx = sparse.kron(sparse.eye(self.nby), d2(self.nbx))
            dy = sparse.kron(d2(self.nby), sparse.eye(self.nbx))
            self._pen = (dx.T @ dx + dy.T @ dy).toarray() + 0.002 * np.eye(self.nby * self.nbx)
        return self._pen

    def raw(self, X, Y):
        """Spline value at points, no box rule."""
        X, Y = np.broadcast_arrays(np.asarray(X, float), np.asarray(Y, float))
        return (self.design(X, Y) @ self.C.ravel()).reshape(X.shape)

    def fit(self, X, Y, values, weights=None):
        A = self.design(X, Y)
        v = np.asarray(values, float).ravel()
        wt = np.ones(v.size) if weights is None else np.asarray(weights, float).ravel()
        Aw = sparse.diags(wt) @ A
        AtA = (A.T @ Aw).toarray()
        self.C = np.linalg.solve(AtA + self._reg_matrix(A, wt), Aw.T @ v).reshape(self.C.shape)
        return self

    def _reg_matrix(self, A, wt) -> np.ndarray:
        lam = self.reg * max(float(np.sum(wt)), 1.0) / 1e4
        return lam * self._penalty() + 1e-9 * np.eye(self.nby * self.nbx)

    # -- evaluation with the box rule
    def _clamped(self, X, Y):
        return GeometrySurface._clamped(self, X, Y)

    def at(self, X, Y):
        """Value at points with the box rule."""
        X, Y = np.broadcast_arrays(np.asarray(X, float), np.asarray(Y, float))
        Xc, Yc, dY = self._clamped(X.ravel(), Y.ravel())
        f = self.raw(Xc, Yc)
        if np.any(dY):
            f = f + (self.raw(Xc, Yc + 0.5) - self.raw(Xc, Yc - 0.5)) * dY
        return f.reshape(X.shape)

    def grid(self, xs, ys):
        """Values on the grid ``ys`` (rows) x ``xs`` (columns), box rule."""
        Xc, Yc, dY = self._clamped(np.asarray(xs, float), np.asarray(ys, float))
        by = self._by(Yc) + self._by(Yc, True) * np.asarray(dY, float).ravel()[:, None]
        return by @ self.C @ self._bx(Xc).T

    def to_json(self) -> dict:
        return dict(kind=self.kind, x_ref=self.x_ref, tx=self.tx.tolist(), ty=self.ty.tolist(), reg=self.reg,
                    coef=self.C.tolist(), box=self.box, y_rule=self.y_rule)

    @classmethod
    def from_json(cls, d: dict) -> "SplineSurface":
        return cls(d["x_ref"], d["tx"], d["ty"], d.get("reg", SPLINE_REG), d["coef"], d.get("box"),
                   d.get("y_rule", "linear"))


class TensorSurface:
    """Keystone ``f(X, Y) = sum_jk C[j, k] * Py_j(v) * (Px_k(u) - Px_k(u_ref))`` on ``u, v`` in [-1, 1].

    Families of the spectral basis ``Px``: ``"polynomial"`` (powers 1..x_degree), ``"chebyshev"``
    (T1..T_x_degree) and ``"piecewise"`` (powers 1..3 plus truncated cubics ``(u - k)_+^3`` at
    interior *knots*: C2 continuous). ``Py`` is powers (polynomial / piecewise) or Chebyshev
    T0..T_y_degree. The anchor ``f(x_ref, Y) = 0`` is exact. Fitted by weighted least squares with a
    ridge on every coefficient (``reg`` x the column's RMS, at least 0.03): high degrees cannot run
    away where the bars are sparse. Same interface and box rule as :class:`SplineSurface`.
    """

    kind = "tensor"
    robust = "soft"

    def __init__(self, family: str, x_degree: int, y_degree: int, x_ref: float, shape, knots=(), reg: float = 0.0,
                 coef=None, box=None, y_rule: str = "linear"):
        if family not in ("polynomial", "chebyshev", "piecewise"):
            raise ValueError("family must be 'polynomial', 'chebyshev' or 'piecewise'")
        self.family, self.x_degree, self.y_degree = family, int(x_degree), int(y_degree)
        self.x_ref = float(x_ref)
        self.shape = tuple(int(v) for v in shape)
        self.knots = [float(k) for k in knots]
        self.reg = float(reg)
        n = (self.y_degree + 1, self.x_degree + len(self.knots))
        self.C = np.zeros(n) if coef is None else np.asarray(coef, float).reshape(n)
        self.box = None if box is None else [float(v) for v in box]
        if y_rule not in ("linear", "hold"):
            raise ValueError("y_rule must be 'linear' or 'hold'")
        self.y_rule = y_rule

    def _u(self, X):
        return 2.0 * np.asarray(X, float).ravel() / max(self.shape[1] - 1, 1) - 1.0

    def _px(self, u):
        if self.family == "chebyshev":
            return np.polynomial.chebyshev.chebvander(u, self.x_degree)[:, 1:]
        cols = [u ** i for i in range(1, self.x_degree + 1)]
        for k in self.knots:
            kk = 2.0 * k / max(self.shape[1] - 1, 1) - 1.0
            cols.append(np.maximum(0.0, u - kk) ** 3)
        return np.column_stack(cols)

    def _bx(self, X):
        return self._px(self._u(X)) - self._px(self._u([self.x_ref]))[0]

    def _by(self, Y, deriv: bool = False):
        if deriv:
            Y = np.asarray(Y, float).ravel()
            return self._by(Y + 0.5) - self._by(Y - 0.5)
        v = 2.0 * np.asarray(Y, float).ravel() / max(self.shape[0] - 1, 1) - 1.0
        if self.family == "chebyshev":
            return np.polynomial.chebyshev.chebvander(v, self.y_degree)
        return np.column_stack([v ** i for i in range(self.y_degree + 1)])

    def design(self, X, Y):
        bx, by = self._bx(X), self._by(Y)
        return (by[:, :, None] * bx[:, None, :]).reshape(len(bx), -1)

    def raw(self, X, Y):
        X, Y = np.broadcast_arrays(np.asarray(X, float), np.asarray(Y, float))
        return np.sum((self._by(Y) @ self.C) * self._bx(X), axis=1).reshape(X.shape)

    def fit(self, X, Y, values, weights=None):
        A = self.design(X, Y)
        v = np.asarray(values, float).ravel()
        wt = np.ones(v.size) if weights is None else np.asarray(weights, float).ravel()
        self.C = np.linalg.solve(A.T @ (A * wt[:, None]) + self._reg_matrix(A, wt),
                                 A.T @ (wt * v)).reshape(self.C.shape)
        return self

    def _reg_matrix(self, A, wt) -> np.ndarray:
        ridge = np.maximum(np.sqrt(np.mean(np.asarray(A) ** 2, axis=0)), 0.03) ** 2 * self.reg
        return np.diag(ridge + 1e-9)

    _clamped = GeometrySurface._clamped
    at = SplineSurface.at
    grid = SplineSurface.grid

    def to_json(self) -> dict:
        return dict(kind=self.kind, family=self.family, x_degree=self.x_degree, y_degree=self.y_degree,
                    x_ref=self.x_ref, shape=list(self.shape), knots=self.knots, reg=self.reg,
                    coef=self.C.tolist(), box=self.box, y_rule=self.y_rule)

    @classmethod
    def from_json(cls, d: dict) -> "TensorSurface":
        return cls(d["family"], d["x_degree"], d["y_degree"], d["x_ref"], d["shape"], d.get("knots", ()),
                   d.get("reg", 0.0), d["coef"], d.get("box"), d.get("y_rule", "linear"))


def surface_from_json(d: dict):
    """A stored dx / dy surface: :class:`SplineSurface`, :class:`TensorSurface` or :class:`GeometrySurface`."""
    kind = d.get("kind")
    if kind == SplineSurface.kind:
        return SplineSurface.from_json(d)
    if kind == TensorSurface.kind:
        return TensorSurface.from_json(d)
    return GeometrySurface.from_json(d)


@dataclass
class GeometryModel:
    """The joint smile + keystone model of one frame layout.

    Attributes:
        shape: ``(H, W)`` of the frames.
        y_ref: Reference row (``dx = 0``).
        x_ref: Reference column (``dy = 0``).
        dx: Smile surface (columns), anchored at ``y_ref``.
        dy: Keystone surface (rows), anchored at ``x_ref``.
        lamp_X: Destination column of every lamp line used.
        target_Y: Destination row of every target bar used.
        dx_rows: ``"destination"``: ``dx(X, Y)`` of the joint fit (lamp + target in one solve);
            ``"source"``: ``dx(X, y)`` over the RAW row, as a stored smile-only model measures it
            (keystone fitted later onto a stored smile): ``x = X + dx(X, Y + dy(X, Y))``.
    """

    shape: Tuple[int, int]
    y_ref: float
    x_ref: float
    dx: GeometrySurface
    dy: GeometrySurface
    lamp_X: List[float] = field(default_factory=list)
    target_Y: List[float] = field(default_factory=list)
    dx_rows: str = "destination"

    def maps(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(map_x, map_y)`` float32 ``(H, W)`` destination -> source maps."""
        h, w = self.shape
        xs = np.arange(w, dtype=float); ys = np.arange(h, dtype=float)
        my = ys[:, None] + self.dy.grid(xs, ys)
        if self.dx_rows == "source":
            X = np.broadcast_to(xs[None, :], my.shape)
            mx = X + self.dx.at(X, my)                  # the smile at the row the pixel is read from
        else:
            mx = xs[None, :] + self.dx.grid(xs, ys)
        return np.ascontiguousarray(mx, np.float32), np.ascontiguousarray(my, np.float32)

    def source(self, X, Y):
        """Raw (source) coordinates of destination points: ``(x, y)``."""
        X = np.asarray(X, float); Y = np.asarray(Y, float)
        y = Y + self.dy.at(X, Y)
        return X + self.dx.at(X, y if self.dx_rows == "source" else Y), y

    def destination(self, x, y, iterations: int = 8):
        """Destination coordinates of raw points (fixed-point inverse of :meth:`source`)."""
        x = np.asarray(x, float); y = np.asarray(y, float)
        X, Y = x.copy(), y.copy()
        for _ in range(iterations):
            X = x - self.dx.at(X, y if self.dx_rows == "source" else Y)
            Y = y - self.dy.at(X, Y)
        return X, Y

    def to_json(self) -> dict:
        return dict(schema=KEYSTONE_SCHEMA, shape=[int(v) for v in self.shape], y_ref=float(self.y_ref),
                    x_ref=float(self.x_ref), dx=self.dx.to_json(), dy=self.dy.to_json(),
                    lamp_X=[float(v) for v in self.lamp_X], target_Y=[float(v) for v in self.target_Y],
                    dx_rows=self.dx_rows)

    @classmethod
    def from_json(cls, d: dict) -> "GeometryModel":
        if d.get("schema") != KEYSTONE_SCHEMA:
            raise ValueError(f"Not a smile + keystone geometry model ({d.get('schema')!r}).")
        return cls(shape=tuple(int(v) for v in d["shape"]), y_ref=float(d["y_ref"]), x_ref=float(d["x_ref"]),
                   dx=surface_from_json(d["dx"]), dy=surface_from_json(d["dy"]),
                   lamp_X=list(d.get("lamp_X", [])), target_Y=list(d.get("target_Y", [])),
                   dx_rows=d.get("dx_rows", "destination"))


# ----------------------------------------------------------------------------------------------- target
@dataclass
class TargetLine:
    """One bar of the line target, traced across the spectrum.

    Attributes:
        y_seed: Row of the bar at the seed column (px).
        x: Sampled column centres (px) where the bar was found.
        y: Bar centre (row, px) at each sampled column.
        coef: Robust cubic ``y(xn)``, ``xn = (x - xc) / sx`` (frame centre / half width), increasing powers.
        coverage: Share of the frame width followed.
        halfwin: Centroid half-window (rows) used for this bar (never reaches a neighbour).
        span: Half distance (rows) between the outer dips of a merged pair (0 for a single bar).
        recovered: Per point of ``x`` / ``y``: True for a centre added by :func:`recover_bar_gaps`
            (measured on the image, not predicted), ``None`` when all come from the tracer. ``coef``
            and ``coverage`` always describe the tracer's points.
    """

    y_seed: float
    x: np.ndarray
    y: np.ndarray
    coef: np.ndarray
    coverage: float
    halfwin: int = 0
    span: float = 0.0
    recovered: Optional[np.ndarray] = None

    def y_at(self, x, w: int) -> np.ndarray:
        xc, sx = (w - 1) / 2.0, max(1.0, (w - 1) / 2.0)
        return np.polynomial.polynomial.polyval((np.asarray(x, float) - xc) / sx, self.coef)


def normalize_target(target, white=None, dark=None, trend_rows: Optional[int] = None,
                     min_white_fraction: float = 0.08) -> np.ndarray:
    """Target frame with the illumination divided out (bright background ~ 1).

    Args:
        target: Line-target frame ``(H, W)``.
        white: Same setup with the target removed (optional). Divides out the light, vignetting
            and pixel response, which is the safest way to avoid a false keystone.
        dark: Dark frame subtracted from both (optional).
        trend_rows: Without *white*: window (rows) of the running mean that removes the slow trend
            along the slit; ``None`` uses ~1/8 of the height (at least 31 rows).
        min_white_fraction: Pixels whose white level is below this share of the white's 99th
            percentile (unlit) become 0.

    Returns:
        float32 ``(H, W)``; unlit pixels 0.
    """
    t = np.asarray(target, np.float32)
    if t.ndim != 2:
        raise ValueError("The target frame must be 2-D (slit rows, spectral columns).")
    d = None if dark is None else np.asarray(dark, np.float32)
    if d is not None:
        if d.shape != t.shape:
            raise ValueError(f"Dark {d.shape} differs from the target {t.shape}.")
        t = t - d
    if white is not None:
        wf = np.asarray(white, np.float32)
        if wf.shape != t.shape:
            raise ValueError(f"White {wf.shape} differs from the target {t.shape}.")
        if d is not None:
            wf = wf - d
        lit = wf > min_white_fraction * float(np.percentile(wf, 99))
        out = np.zeros_like(t)
        out[lit] = t[lit] / wf[lit]
        return out
    h = t.shape[0]
    n = int(trend_rows) if trend_rows else max(31, h // 8)
    n += 1 - n % 2
    trend = uniform_filter1d(t, size=n, axis=0, mode="nearest")
    lit = trend > min_white_fraction * float(np.percentile(trend, 99))
    out = np.zeros_like(t)
    out[lit] = t[lit] / trend[lit]
    return out


def _band_columns(img: np.ndarray, band: int) -> np.ndarray:
    """Mean over ``band`` neighbouring columns, centred (same width as *img*)."""
    return uniform_filter1d(img, size=max(1, int(band)), axis=1, mode="nearest")


def _robust_poly(x, y, deg, k=4.0, floor=0.05, iters=4):
    """Polynomial fit with MAD rejection; returns ``(coef, keep)``."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    keep = np.ones(x.size, bool)
    deg = int(min(deg, max(0, x.size - 2)))
    coef = np.polynomial.polynomial.polyfit(x, y, deg)
    for _ in range(iters):
        if keep.sum() <= deg + 2:
            break
        coef = np.polynomial.polynomial.polyfit(x[keep], y[keep], deg)
        r = y - np.polynomial.polynomial.polyval(x, coef)
        sig = max(floor, robust_sigma(r[keep]))
        new = np.abs(r) <= k * sig
        if new.sum() <= deg + 2 or np.array_equal(new, keep):
            break
        keep = new
    coef = np.polynomial.polynomial.polyfit(x[keep], y[keep], deg)
    return coef, keep


def _bar_profile_noise(prof: np.ndarray) -> float:
    return max(robust_sigma(np.diff(prof)) / math.sqrt(2.0), 1e-6)


def _column_noise(img: np.ndarray) -> np.ndarray:
    """Per-column noise of slit profiles (robust sigma of row-to-row differences / sqrt 2).

    Row differences also contain the bar edges, so the median absolute difference over-estimates
    the noise a little where the bars are sharp; it is a conservative detection floor.
    """
    d = np.diff(np.asarray(img, np.float64), axis=0)
    med = np.median(d, axis=0)
    sig = 1.4826 * np.median(np.abs(d - med), axis=0) / math.sqrt(2.0)
    return np.maximum(sig, 1e-6)


def _merge_close(peaks: np.ndarray, near: float = 0.6, spacing: Optional[float] = None):
    """Group dips closer than ``near`` x the median spacing into one feature.

    A pair of thin lines that the optics resolve in one part of the spectrum and blur together in
    another would make a centroid locked on one of them jump; the centroid of the whole pair
    stays put. Returns a list of ``(centre, half_span)`` (``half_span`` 0 for a single dip) and
    the list of merged groups (rows) for the report.

    *spacing* (the bar period) replaces the median spacing of *peaks*: a column where only a few,
    scattered bars are found has a median spacing far above the period, which would group two
    neighbouring bars as one pair.
    """
    peaks = np.sort(np.asarray(peaks, float))
    if spacing is None:
        spacing = float(np.median(np.diff(peaks))) if len(peaks) > 1 else 0.0
    groups = [[peaks[0]]] if len(peaks) else []
    for p in peaks[1:]:
        if p - groups[-1][-1] < near * spacing:
            groups[-1].append(p)
        else:
            groups.append([p])
    feats = [(0.5 * (g[0] + g[-1]), 0.5 * (g[-1] - g[0])) for g in groups]
    merged = [[float(v) for v in g] for g in groups if len(g) > 1]
    return feats, merged


def _feature_halfwins(feats, base: int) -> List[int]:
    """Centroid half-window per feature: its half span + ``base``, never past half-way to a neighbour's dip."""
    out = []
    for i, (c, s) in enumerate(feats):
        r = s + base
        for j in (i - 1, i + 1):
            if 0 <= j < len(feats):
                cj, sj = feats[j]
                d = abs(cj - c) - sj                    # distance to the neighbour's nearest dip
                r = min(r, 0.5 * (d + s))               # half-way between own edge dip and that one
        out.append(int(max(2, math.floor(r))))
    return out


def _neighbour_halfwins(rows: np.ndarray, halfwin: int) -> List[int]:
    """Centroid half-window per bar: ``halfwin``, but never past half-way to a neighbour."""
    return _feature_halfwins([(float(r), 0.0) for r in rows], halfwin)


#: Dark-area share within this distance of 50 % counts as ambiguous: both polarities are traced and compared.
POLARITY_AMBIGUOUS = 0.08


def detect_polarity(target, white=None, dark=None, normalized=None, band: int = 5, snr_min: float = 5.0
                    ) -> Tuple[str, dict]:
    """Are the target's bars darker or brighter than the background? Decided from the image.

    The bars are the narrow features: on a target of thin dark lines on white, the pixels darker
    than the middle level (between the 5th and 95th percentile of the lit, normalised middle 60 %
    of the spectrum) cover less than half of the area; on bright lines on black, more. When the
    share is within ``POLARITY_AMBIGUOUS`` of 50 % (bars and gaps about as wide), both polarities
    are traced and the one with more bars followed further and scattering less is taken; either
    then gives the same geometry within a few hundredths of a pixel.

    Args:
        target: Line-target frame ``(H, W)`` (raw counts).
        white, dark: See :func:`normalize_target`.
        normalized: Pre-computed :func:`normalize_target` output.
        band, snr_min: Passed to the tracing of the ambiguous case.

    Returns:
        ``(polarity, info)``: ``"dark"`` or ``"bright"``; ``info`` has ``dark_fraction``,
        ``method`` (``"area"`` or ``"trace"``) and, for ``"trace"``, the score of each polarity.
    """
    raw = np.asarray(target, np.float32)
    norm = normalize_target(raw, white, dark) if normalized is None else np.asarray(normalized, np.float32)
    h, w = norm.shape
    c0, c1 = int(0.2 * w), max(int(0.2 * w) + 1, int(0.8 * w))
    v = norm[:, c0:c1]
    v = v[v > 0]
    if v.size < 100:
        return "dark", dict(dark_fraction=None, method="default", reason="no lit pixels")
    lo, hi = np.percentile(v, (5, 95))
    dark_fraction = float(np.mean(v < 0.5 * (lo + hi)))
    info = dict(dark_fraction=dark_fraction, method="area")
    if abs(dark_fraction - 0.5) >= POLARITY_AMBIGUOUS:
        return ("dark" if dark_fraction < 0.5 else "bright"), info
    scores = {}
    for pol in ("dark", "bright"):
        try:
            bars, _ = trace_target_lines(raw, polarity=pol, band=band, snr_min=snr_min, normalized=norm)
        except ValueError:
            scores[pol] = 0.0
            continue
        xn = lambda b: (b.x - (w - 1) / 2.0) / max(1.0, (w - 1) / 2.0)
        scatter = float(np.median([np.sqrt(np.mean((b.y - np.polynomial.polynomial.polyval(xn(b), b.coef)) ** 2))
                                   for b in bars]))
        cover = float(np.median([b.coverage for b in bars]))
        scores[pol] = len(bars) * cover / (1.0 + scatter / 0.1)
    info.update(method="trace", scores=scores)
    return max(scores, key=scores.get) if any(scores.values()) else "dark", info


#: Most seed columns :func:`trace_target_lines` starts bars from.
MAX_SEED_COLUMNS = 4
#: An extra seed column is added only when it finds at least this many bars no earlier seed found.
MIN_SEED_GAIN = 2


def _seed_peaks(img: np.ndarray, lit: np.ndarray, noise_col: np.ndarray, col: int, snr_min: float) -> np.ndarray:
    """Rows of the bars (peaks of the signed profile) at column *col*, inside the lit slit."""
    sm = gaussian_filter1d(img[:, col], 1.0)
    peaks, props = find_peaks(sm, prominence=snr_min * float(noise_col[col]))
    if len(peaks) >= 2:
        prom = props["prominences"]
        peaks = peaks[prom >= 0.25 * float(np.percentile(prom, 90))]
    return peaks[lit[peaks, col]] if len(peaks) else peaks


def _extra_seeds(img, lit, lit_cols, noise_col, seed: int, peaks, period: float, snr_min: float,
                 max_seeds: int) -> List[int]:
    """The first seed plus, greedily, the columns that find the most bars no earlier seed found.

    Candidates are 19 columns spread over the lit spectrum; a candidate counts the bars it finds
    farther than half a period (by row) from every bar already found. Row is close enough: the
    keystone moves a bar by a few pixels across the spectrum, much less than half a period on a
    useful target. Stops at *max_seeds* or when no candidate adds :data:`MIN_SEED_GAIN` bars.
    """
    seeds = [int(seed)]
    if max_seeds <= 1:
        return seeds
    full = np.flatnonzero(lit_cols.mean(axis=0) > 0.5)
    if full.size < 2:
        return seeds
    lo, hi = int(full[0]), int(full[-1])
    cand = np.unique(np.linspace(lo + 0.03 * (hi - lo), hi - 0.03 * (hi - lo), 19).round().astype(int))
    found = {int(c): _seed_peaks(img, lit, noise_col, int(c), snr_min) for c in cand}
    known = np.asarray(peaks, float)
    while len(seeds) < max_seeds:
        best, gain = None, MIN_SEED_GAIN - 1
        for c, rows in found.items():
            if c in seeds or len(rows) == 0:
                continue
            if known.size:
                new = int(np.sum(np.min(np.abs(rows[:, None] - known[None, :]), axis=1) > 0.5 * period))
            else:
                new = len(rows)
            if new > gain:
                best, gain = c, new
        if best is None:
            break
        seeds.append(best)
        known = np.r_[known, found[best]]
    return seeds


def _track_near(tracks: List[dict], col: int, row: float, step: int, tol: float) -> bool:
    """Whether a traced bar passes within *tol* rows of ``(col, row)`` (at a point within two steps)."""
    for t in tracks:
        for c in (col, col - step, col + step, col - 2 * step, col + 2 * step):
            y = t["pts"].get(int(c))
            if y is not None and abs(y - row) <= tol:
                return True
    return False


def _end_fit(pts: Dict[int, float], side: str, n: int = 12):
    """Line through the *n* points at one end (``"lo"``/``"hi"``) of a track: ``(coef, end column)``."""
    xs = np.array(sorted(pts), float)
    sel = xs[:n] if side == "lo" else xs[-n:]
    ys = np.array([pts[int(c)] for c in sel])
    coef = np.polynomial.polynomial.polyfit(sel, ys, 1) if sel.size >= 3 else np.array([ys.mean(), 0.0])
    return coef, float(sel[0] if side == "lo" else sel[-1])


def _same_bar(a: dict, b: dict, period: float, w: int) -> bool:
    """Whether two traced point sets belong to one bar.

    Overlapping tracks: their rows agree (median over the shared columns within a quarter period).
    Tracks with a gap no wider than a quarter of the frame: each one's end line, extended to the
    middle of the gap, meets the other's within 0.3 periods.
    """
    pa, pb = a["pts"], b["pts"]
    shared = sorted(set(pa) & set(pb))
    if len(shared) >= 3:
        return float(np.median([abs(pa[c] - pb[c]) for c in shared])) < 0.25 * period
    if shared:
        return False
    (alo, ahi), (blo, bhi) = (min(pa), max(pa)), (min(pb), max(pb))
    if ahi < blo:
        left, right = pa, pb
    elif bhi < alo:
        left, right = pb, pa
    else:
        return False                                    # interleaved columns without shared ones: not one bar
    cl, xl = _end_fit(left, "hi")
    cr, xr = _end_fit(right, "lo")
    if xr - xl > 0.25 * w:
        return False
    xm = 0.5 * (xl + xr)
    yl = float(np.polynomial.polynomial.polyval(xm, cl))
    yr = float(np.polynomial.polynomial.polyval(xm, cr))
    return abs(yl - yr) < 0.3 * period


def _merge_tracks(tracks: List[dict], period: float, w: int) -> List[dict]:
    """Merge tracks of the same bar (found from different seeds); the earlier track's points win
    where both have one. A track that matches two different earlier tracks joins neither."""
    out: List[dict] = []
    for t in tracks:
        hits = [m for m in out if _same_bar(m, t, period, w)]
        if len(hits) == 1:
            m = hits[0]
            for c, y in t["pts"].items():
                m["pts"].setdefault(c, y)
            m["hw"] = min(m["hw"], t["hw"])
        elif not hits:
            out.append(dict(t, pts=dict(t["pts"])))
    return out


def trace_target_lines(target, polarity: str = "dark", white=None, dark=None, band: int = 5, step: Optional[int] = None,
                       snr_min: float = 5.0, sat_level: Optional[float] = None, seed_column: Optional[float] = None,
                       max_gap: Optional[int] = None, min_coverage: float = MIN_BAR_COVERAGE, normalized=None,
                       max_seeds: int = MAX_SEED_COLUMNS) -> Tuple[List[TargetLine], dict]:
    """Follow the bars of a line-target frame across the spectrum.

    Each bar is located per sampled column as the intensity centroid of the (inverted, for dark
    bars) slit profile in a window of ~0.45 bar periods, re-centred iteratively, starting from a
    seed column and walking outward with a jump limit.

    Bars are found first at the seed column with the most bar contrast. When the focus varies
    across the frame (field curvature with a focus that moves with wavelength), bars near the slit
    ends may be too blurred there to be found at all, although they are sharp elsewhere in the
    spectrum. Further seed columns are therefore added (up to *max_seeds*), each where the most
    bars appear that no earlier seed found; bars found from several seeds are merged into one.

    Args:
        target: Line-target frame ``(H, W)`` (raw counts).
        polarity: ``"dark"`` (bars darker than the background), ``"bright"``, or ``"auto"``
            (:func:`detect_polarity`; the choice is in ``info["polarity"]`` / ``info["polarity_auto"]``).
        white, dark: See :func:`normalize_target`.
        band: Columns averaged per sample (noise).
        step: Column step between samples; ``None`` = *band*.
        snr_min: Minimum bar depth over the profile noise.
        sat_level: Sensor full scale (``None``: inferred). Windows touching clipped pixels are skipped.
        seed_column: Column where bars are found first; ``None``: the brightest column of the middle 60 %.
        max_gap: Consecutive failed samples before a bar is given up in that direction; ``None``:
            about 4 % of the width (crosses a dim stripe of the target or the light).
        min_coverage: Minimum share of the width a bar must be followed over.
        normalized: Pre-computed :func:`normalize_target` output (skips normalisation).
        max_seeds: Most seed columns (1 = the single-seed tracing of earlier versions).

    Returns:
        ``(lines, info)``; lines sorted by row. ``info``: ``seed_column`` (the first seed; each
        line's ``y_seed`` is its row there, from its fit when it was traced from another seed),
        ``seed_columns`` (all seeds used), ``period``, ``halfwin``, ``noise``, ``polarity`` (the one
        used), ``polarity_auto`` (detection details or ``None``), ``rejected`` (bars dropped and
        why), ``merged`` (close pairs), ``from_extra_seeds`` (bars that only extra seeds found).

    Raises:
        ValueError: Fewer than three bars, or bars spanning less than ``MIN_SLIT_COVERAGE`` of the slit.
    """
    raw = np.asarray(target, np.float32)
    if raw.ndim != 2 or min(raw.shape) < 16:
        raise ValueError("Use a 2-D target frame of at least 16 x 16 pixels.")
    h, w = raw.shape
    if polarity not in ("dark", "bright", "auto"):
        raise ValueError("polarity must be 'dark', 'bright' or 'auto'")
    norm = normalize_target(raw, white, dark) if normalized is None else np.asarray(normalized, np.float32)
    polarity_auto = None
    if polarity == "auto":
        polarity, polarity_auto = detect_polarity(raw, normalized=norm, band=band, snr_min=snr_min)
    sign = -1.0 if polarity == "dark" else 1.0
    if sat_level is None or sat_level <= 0:
        sat_level = auto_sat_level(raw)
    clipped = raw >= 0.98 * float(sat_level)
    lit = norm > 0
    band = max(1, int(band))
    step = max(1, int(step or band))
    img = sign * _band_columns(norm, band)
    clip_cols = _band_columns(clipped.astype(np.float32), band) > 0
    lit_cols = _band_columns(lit.astype(np.float32), band) > 0.99

    if seed_column is None:
        # the column with the most bar contrast over the lit slit (where the most bars are present);
        # not the brightest one: after a white frame every column is equally bright
        lo, hi = int(0.2 * w), max(int(0.2 * w) + 1, int(0.8 * w))
        lit_any = lit.any(axis=1)
        sub = np.where(lit[lit_any], norm[lit_any], np.nan)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            contrast = np.nan_to_num(np.nanstd(sub, axis=0) * np.mean(lit[lit_any], axis=0))
        seed = int(lo + np.argmax(gaussian_filter1d(contrast[lo:hi], 7)))
    else:
        seed = int(np.clip(round(seed_column), 0, w - 1))
    noise_col = _column_noise(img)                      # bar contrast and noise change along the spectrum
    noise = float(noise_col[seed])
    peaks = _seed_peaks(img, lit, noise_col, seed, snr_min)
    if len(peaks) < 3:
        raise ValueError("Fewer than three target bars found at the seed column. Check the halogen light, the "
                         "bar polarity (dark / bright) and the exposure.")
    feats, merged = _merge_close(peaks)
    centres = np.array([c for c, _ in feats])
    period = float(np.median(np.diff(centres))) if len(centres) > 1 else float(np.median(np.diff(peaks)))
    halfwin = int(max(2, math.floor(0.45 * period)))
    if max_gap is None:
        max_gap = max(6, int(math.ceil(0.04 * w / step)))
    cols = np.arange(seed % step, w, step)
    cols = cols[(cols >= band // 2) & (cols <= w - 1 - band // 2)]
    seeds = _extra_seeds(img, lit, lit_cols, noise_col, seed, peaks, period, snr_min, max(1, int(max_seeds)))
    cols = np.unique(np.r_[cols, seeds])

    def follow(sd, c0, hw):
        """Points ``{column: row}`` of one bar followed both ways from seed column *sd*."""
        max_jump = max(1.0, 0.5 * hw)
        pts: Dict[int, float] = {int(sd): c0}
        for direction in (-1, 1):
            chosen = cols[cols < sd][::-1] if direction < 0 else cols[cols > sd]
            yc, gap = c0, 0
            for c in chosen:
                r0, r1 = int(round(yc)) - hw, int(round(yc)) + hw + 1
                bad = r0 < 0 or r1 > h or clip_cols[max(0, r0):r1, c].any() or not lit_cols[max(0, r0):r1, c].all()
                if not bad:
                    yy, ok = _centroid_row(img[:, c], yc, hw, np.inf, snr_min * float(noise_col[c]))
                    bad = not ok or abs(yy - yc) > max_jump
                if bad:
                    gap += 1
                    if gap > max_gap:
                        break
                    continue
                pts[int(c)] = yy
                yc, gap = yy, 0
        return pts

    rejected = []                                       # (row at the first seed, why)
    tracks: List[dict] = []                             # dict(pts, hw, span, seed, y0)
    for k, sd in enumerate(seeds):
        if k == 0:
            s_feats = feats
        else:
            # group with the first seed's period: this column may show only a few scattered bars
            s_feats, s_merged = _merge_close(_seed_peaks(img, lit, noise_col, sd, snr_min), spacing=period)
            merged += [g for g in s_merged if all(abs(g[0] - m[0]) > 0.5 * period for m in merged)]
        for (p, sp), hw in zip(s_feats, _feature_halfwins(s_feats, halfwin)):
            max_jump = max(1.0, 0.5 * hw)
            c0, ok = _centroid_row(img[:, sd], float(p), hw, np.inf, snr_min * float(noise_col[sd]))
            if not ok or abs(c0 - p) > max_jump:
                if k == 0:
                    rejected.append((float(p), "no centroid at the seed column"))
                continue
            if k > 0 and _track_near(tracks, sd, c0, step, max(1.0, 0.5 * hw)):
                continue                                # already followed through this column from an earlier seed
            tracks.append(dict(pts=follow(sd, c0, hw), hw=int(hw), span=float(sp), seed=int(sd), y0=float(c0)))
    tracks = _merge_tracks(tracks, period, w)

    lines: List[TargetLine] = []
    from_extra = 0
    for t in tracks:
        xs = np.array(sorted(t["pts"]), float)
        ys = np.array([t["pts"][int(c)] for c in xs], float)
        cov = float(np.ptp(xs) / max(1, w - 1)) if xs.size else 0.0
        if xs.size < 8 or cov < min_coverage:
            if t["seed"] == seed:
                rejected.append((t["y0"], "followed over %.0f %% of the width only" % (100 * cov)))
            continue
        xn = (xs - (w - 1) / 2.0) / max(1.0, (w - 1) / 2.0)
        coef, keep = _robust_poly(xn, ys, 3)
        if seed in t["pts"]:
            y_seed = float(t["pts"][seed])
        else:                                           # traced from another seed: its row at the first seed
            y_seed = float(np.polynomial.polynomial.polyval((seed - (w - 1) / 2.0) / max(1.0, (w - 1) / 2.0), coef))
        from_extra += t["seed"] != seed
        lines.append(TargetLine(y_seed=y_seed, x=xs[keep], y=ys[keep], coef=coef, coverage=cov,
                                halfwin=t["hw"], span=t["span"]))
    lines.sort(key=lambda t: t.y_seed)
    # drop bars that merged with a neighbour (same track)
    out: List[TargetLine] = []
    for t in lines:
        if out and abs(t.y_seed - out[-1].y_seed) < 2.0:
            rejected.append((t.y_seed, "duplicate of the bar at row %.1f" % out[-1].y_seed))
            continue
        out.append(t)
    # a bar rejected at the first seed but followed from another one is not rejected
    rejected = [(r, why) for r, why in rejected
                if why.startswith("duplicate") or all(abs(t.y_seed - r) > 0.5 * period for t in out)]
    if len(out) < 3:
        raise ValueError("Fewer than three target bars could be followed across the spectrum. Check the halogen "
                         "light, the focus and the bar polarity.")
    span = out[-1].y_seed - out[0].y_seed
    if span < MIN_SLIT_COVERAGE * (h - 1):
        raise ValueError("The target bars span only %.0f %% of the slit (minimum %.0f %%). Place the line target "
                         "across the whole slit." % (100 * span / max(1, h - 1), 100 * MIN_SLIT_COVERAGE))
    info = dict(seed_column=seed, seed_columns=[int(c) for c in seeds], period=period, halfwin=halfwin, noise=noise,
                band=band, step=step, polarity=polarity, polarity_auto=polarity_auto, rejected=rejected,
                merged=merged, from_extra_seeds=int(from_extra))
    return out, info


# ----------------------------------------------------------------------------------------------- gap recovery
#: Gap recovery: columns averaged per slit profile, slit smoothing (px), trough / crest prominence and
#: minimum trough depth (normalised units), search radius around the prediction (px).
GAP_AVG_COLUMNS = 21
GAP_SMOOTH_Y = 1.05
GAP_TROUGH_PROMINENCE = 0.014
GAP_CREST_PROMINENCE = 0.012
GAP_MIN_DEPTH = 0.035
GAP_SEARCH_RADIUS = 3.8


def _gap_features(prof_img: np.ndarray, xs: np.ndarray, period: float):
    """Per column: trough rows, their sub-pixel centres (3-point quadratic) and the crest rows."""
    h = prof_img.shape[0]
    dist = max(5, int(round(0.6 * period)))
    out = {}
    for x in xs:
        p = prof_img[:, int(x)]
        dips, _ = find_peaks(-p, distance=dist, prominence=GAP_TROUGH_PROMINENCE)
        crests, _ = find_peaks(p, distance=dist, prominence=GAP_CREST_PROMINENCE)
        dips = dips[(dips > 8) & (dips < h - 9)]
        cen = []
        for d in dips:
            v0, v1, v2 = p[d - 1], p[d], p[d + 1]
            den = v0 - 2 * v1 + v2
            off = 0.5 * (v0 - v2) / den if abs(den) > 1e-9 else 0.0
            cen.append(float(d + np.clip(off, -0.5, 0.5)))
        out[int(x)] = (dips, np.asarray(cen, float), crests)
    return out


def _gap_observe(x: int, pred: float, feats, prof_img, period: float):
    """The one trough near *pred* at column *x*, bounded by a crest on each side and deep enough."""
    dips, cen, crests = feats[x]
    if not len(dips):
        return None
    near = np.flatnonzero(np.abs(cen - pred) < GAP_SEARCH_RADIUS)
    if near.size != 1:                                  # none, or ambiguous: no measurement
        return None
    d = int(dips[int(near[0])]); y = float(cen[int(near[0])])
    left, right = crests[crests < d], crests[crests > d]
    if not len(left) or not len(right):
        return None
    lo, hi = int(left[-1]), int(right[0])
    if min(d - lo, hi - d) < 1.5 or hi - lo > 1.8 * period:
        return None
    depth = float(min(prof_img[lo, x] - prof_img[d, x], prof_img[hi, x] - prof_img[d, x]))
    return (y, depth) if depth >= GAP_MIN_DEPTH else None


def _gap_grow(obs: Dict[int, float], feats, prof_img, xs, period: float) -> Dict[int, tuple]:
    """Walk both ways along the sampled columns; where the bar has no centre, look for its trough
    at the position predicted from the last (up to 7) centres; never more than one candidate."""
    found: Dict[int, tuple] = {}
    for seq in (xs, xs[::-1]):
        hist: List[Tuple[int, float]] = []
        for x in seq:
            x = int(x)
            if x in obs:
                hist.append((x, obs[x])); continue
            if len(hist) < 2:
                continue
            last = hist[-7:]
            xx = np.array([p[0] for p in last], float); yy = np.array([p[1] for p in last], float)
            slope = float(np.median(np.diff(yy) / np.diff(xx))) if len(last) >= 3 else \
                (yy[-1] - yy[-2]) / (xx[-1] - xx[-2])
            slope = float(np.clip(slope, -0.10, 0.10))
            got = _gap_observe(x, float(yy[-1] + slope * (x - xx[-1])), feats, prof_img, period)
            if got is None or abs(got[0] - yy[-1]) > max(1.6, 0.20 * abs(x - xx[-1])):
                continue
            if x not in found or got[1] > found[x][1]:
                found[x] = got
            hist.append((x, got[0]))
    return found


def recover_bar_gaps(bars: List[TargetLine], info: dict, normalized: np.ndarray) -> Tuple[List[TargetLine], dict]:
    """Add MEASURED bar centres where :func:`trace_target_lines` lost a bar (one pass).

    Where the bar contrast fades (a weak or defocused part of the spectrum) the tracer's 5-column
    centroid stops, although the bars are still visible over a wider average. Starting from each
    traced bar (its identity and points are kept), the slit profile is averaged over
    ``GAP_AVG_COLUMNS`` columns and smoothed; at every sampled column the bar has no centre, the
    single trough near the position predicted from its last centres is taken when two crests bound
    it and it is deep enough (3-point quadratic sub-pixel centre). Candidates are then checked
    against the neighbouring bars (0.60-1.48 periods apart), against a local line through the
    nearby centres (spikes over 1.25 px), and by the tracer's robust cubic per bar. Nothing
    predicted is ever stored: a column without a qualifying trough stays empty.

    The recovered centres go into ``x`` / ``y`` with ``recovered`` set; the keystone fit weighs them
    ``RECOVERED_WEIGHT``. Recovery stays within the columns where the tracer found bars.

    Args:
        bars: Output of :func:`trace_target_lines`.
        info: Its ``info`` (``period``, ``step``, ``seed_column``, ``polarity``).
        normalized: :func:`normalize_target` output used for the tracing.

    Returns:
        ``(bars, stats)``: new :class:`TargetLine` objects (the input is not changed) sorted by row,
        and ``candidates`` / ``accepted`` / ``rejected_neighbour`` / ``rejected_spike`` /
        ``rejected_robust`` counts.
    """
    stats = dict(candidates=0, accepted=0, rejected_neighbour=0, rejected_spike=0, rejected_robust=0)
    bars = sorted(bars, key=lambda b: b.y_seed)
    if len(bars) < 2:
        return [copy.copy(b) for b in bars], stats
    norm = np.asarray(normalized, np.float32)
    h, w = norm.shape
    period, step = float(info["period"]), int(info["step"])
    # troughs = the bars: dark bars as they are, bright bars inverted
    sign = 1.0 if info.get("polarity", "dark") == "dark" else -1.0
    prof_img = gaussian_filter1d(uniform_filter1d(sign * norm, size=GAP_AVG_COLUMNS, axis=1, mode="nearest"),
                                 GAP_SMOOTH_Y, axis=0)
    lo = min(float(b.x.min()) for b in bars); hi = max(float(b.x.max()) for b in bars)
    xs = np.arange(int(info["seed_column"]) % step, w, step)
    xs = xs[(xs >= lo) & (xs <= hi)]
    if xs.size < 3:
        return [copy.copy(b) for b in bars], stats
    feats = _gap_features(prof_img, xs, period)
    obs = [{int(round(x)): float(y) for x, y in zip(b.x, b.y) if int(round(x)) in feats} for b in bars]
    recs = [_gap_grow(o, feats, prof_img, xs, period) for o in obs]
    # neighbour order: a recovered centre must sit 0.60-1.48 periods from the next bar up / down
    for x in xs:
        x = int(x)
        for i, rec in enumerate(recs):
            if x not in rec:
                continue
            y = rec[x][0]
            for j in (i - 1, i + 1):
                if 0 <= j < len(recs):
                    other = obs[j].get(x, recs[j].get(x, (None, 0))[0])
                    if other is not None:
                        delta = y - other if j < i else other - y
                        if not 0.60 * period < delta < 1.48 * period:
                            del rec[x]; stats["rejected_neighbour"] += 1
                            break
    # spikes against a local line through the centres two steps either side
    for o, rec in zip(obs, recs):
        for x in list(rec):
            near = []
            for u in (x - 2 * step, x - step, x + step, x + 2 * step):
                if u in o:
                    near.append((u, o[u]))
                elif u in rec:
                    near.append((u, rec[u][0]))
            if len(near) >= 2:
                xx = np.array([p[0] for p in near], float); yy = np.array([p[1] for p in near], float)
                if abs(np.polyval(np.polyfit(xx, yy, 1), x) - rec[x][0]) > 1.25:
                    del rec[x]; stats["rejected_spike"] += 1
    out: List[TargetLine] = []
    for b, rec in zip(bars, recs):
        stats["candidates"] += len(rec)
        nb = copy.copy(b)
        if rec:
            old = {int(round(x)): float(y) for x, y in zip(b.x, b.y)}
            new = {x: v[0] for x, v in rec.items() if x not in old}
            allx = np.array(sorted(set(old) | set(new)), float)
            ally = np.array([old.get(int(x), new.get(int(x))) for x in allx], float)
            xn = (allx - (w - 1) / 2.0) / max(1.0, (w - 1) / 2.0)
            _, keep = _robust_poly(xn, ally, 3)
            is_new = np.array([int(x) in new and int(x) not in old for x in allx])
            acc = is_new & keep
            stats["rejected_robust"] += int(np.sum(is_new & ~keep))
            if acc.any():
                x = np.r_[b.x, allx[acc]]; y = np.r_[b.y, ally[acc]]
                r = np.r_[np.zeros(b.x.size, bool) if b.recovered is None else b.recovered, np.ones(int(acc.sum()), bool)]
                o = np.argsort(x, kind="stable")
                nb.x, nb.y, nb.recovered = x[o], y[o], r[o]
                stats["accepted"] += int(acc.sum())
        out.append(nb)
    return out, stats


def _recover(bars, tinfo, norm, enabled: bool, notes: List[str]):
    """:func:`recover_bar_gaps` when *enabled*; returns ``(bars, stats or None)`` and adds a note."""
    if not enabled:
        return bars, None
    bars, gap = recover_bar_gaps(bars, tinfo, norm)
    n_trace = sum(int(b.x.size - (0 if b.recovered is None else b.recovered.sum())) for b in bars)
    notes.append("gap recovery: %d bar centres measured where the tracer lost the bars (%d traced; weight %.2f "
                 "in the fit)" % (gap["accepted"], n_trace, RECOVERED_WEIGHT))
    return bars, gap


def _point_weights(bars: List[TargetLine]) -> np.ndarray:
    """Base weight of every bar point in the keystone fit: 1, or ``RECOVERED_WEIGHT`` for a recovered centre."""
    return np.concatenate([np.where(b.recovered, RECOVERED_WEIGHT, 1.0) if b.recovered is not None
                           else np.ones(b.x.size) for b in bars])


# ----------------------------------------------------------------------------------------------- coverage
#: A spectral zone is covered when its bars span at least this share of the bars' full span along the slit.
ZONE_SPAN_MIN = 0.9
#: Lit slit rows beyond the outermost bars (at either end), as a share of the lit rows, above which a warning is given.
SLIT_END_MAX = 0.06


def bar_coverage(bars: List[TargetLine], shape, normalized=None, n_zones: int = 8, step: int = 5) -> dict:
    """Where along the slit the target bars were followed, zone by zone across the spectrum.

    The keystone's magnification at a wavelength comes from the outermost bars measured there. When
    the bars near the slit ends are lost toward one end of the spectrum (dim, vignetted, blurred
    corners), the model is extrapolated along the slit there although the bars' overall column span
    looks complete; a single QC median does not show it.

    Args:
        bars: Traced bars (:func:`trace_target_lines`).
        shape: ``(H, W)``.
        normalized: :func:`normalize_target` output (its lit area sets the zones and the slit ends).
        n_zones: Spectral zones across the lit columns.
        step: Column step of the tracing (sets how many samples a zone should have).

    Returns:
        dict: ``zones`` (list of ``{c0, c1, n_bars, rows: [lo, hi] or None, share}``), ``bar_rows``
        (rows of the outermost bars), ``lit_rows`` / ``lit_columns`` (``None`` without *normalized*).
    """
    h, w = (int(v) for v in shape)
    out = dict(zones=[], bar_rows=None, lit_rows=None, lit_columns=None)
    if not bars:
        return out
    lit_rows = lit_cols = None
    if normalized is not None:
        lit = np.asarray(normalized) > 0
        rows_lit = np.flatnonzero(lit[:, int(0.3 * w):max(int(0.3 * w) + 1, int(0.7 * w))].mean(axis=1) > 0.5)
        cols_lit = np.flatnonzero(lit[int(0.3 * h):max(int(0.3 * h) + 1, int(0.7 * h)), :].mean(axis=0) > 0.5)
        if rows_lit.size:
            lit_rows = [int(rows_lit[0]), int(rows_lit[-1])]
        if cols_lit.size:
            lit_cols = [int(cols_lit[0]), int(cols_lit[-1])]
    c_lo, c_hi = lit_cols if lit_cols else (0, w - 1)
    full = [float(min(b.y.min() for b in bars)), float(max(b.y.max() for b in bars))]
    span = max(1.0, full[1] - full[0])
    edges = np.linspace(c_lo, c_hi + 1, n_zones + 1)
    zones = []
    for c0, c1 in zip(edges[:-1], edges[1:]):
        need = max(3, int(0.3 * (c1 - c0) / max(1, step)))
        ys = [float(np.mean(b.y[(b.x >= c0) & (b.x < c1)])) for b in bars if np.count_nonzero((b.x >= c0) & (b.x < c1)) >= need]
        rows = [min(ys), max(ys)] if len(ys) >= 2 else None
        share = (rows[1] - rows[0]) / span if rows else 0.0
        zones.append(dict(c0=int(c0), c1=int(c1) - 1, n_bars=len(ys), rows=rows, share=float(share)))
    out.update(zones=zones, bar_rows=full, lit_rows=lit_rows, lit_columns=lit_cols)
    return out


def coverage_warnings(cov: dict, wavelength=None) -> List[str]:
    """Readable warnings for :func:`bar_coverage`: spectral zones whose outermost bars are missing,
    and lit slit ends without bars. ``wavelength(column)`` (optional) adds nm to the column ranges."""
    def cols(c0, c1):
        text = "columns %d-%d" % (c0, c1)
        if wavelength is not None:
            try:
                text += " (%.0f-%.0f nm)" % (float(wavelength(c0)), float(wavelength(c1)))
            except Exception:  # noqa: BLE001
                pass
        return text
    warn = []
    bad = [z for z in cov.get("zones", []) if z["share"] < ZONE_SPAN_MIN]
    groups = []
    for z in bad:
        if groups and z["c0"] <= groups[-1][-1]["c1"] + 1:
            groups[-1].append(z)
        else:
            groups.append([z])
    full = cov.get("bar_rows") or [0, 0]
    for g in groups:
        rows = [z["rows"] for z in g if z["rows"]]
        where = ("rows %.0f-%.0f only" % (min(r[0] for r in rows), max(r[1] for r in rows))) if rows else "no bars"
        warn.append("%s: the bars cover %s (all bars: rows %.0f-%.0f), so the keystone there is extrapolated along "
                    "the slit. Raise the exposure (the centre may come close to full scale), average frames of the "
                    "static target, or add a white reference, so the bars near the slit ends are followed there too."
                    % (cols(g[0]["c0"], g[-1]["c1"]).capitalize(), where, full[0], full[1]))
    lit = cov.get("lit_rows")
    if lit and cov.get("bar_rows"):
        n = max(1, lit[1] - lit[0])
        top, bottom = full[0] - lit[0], lit[1] - full[1]
        ends = []
        if top > SLIT_END_MAX * n:
            ends.append("rows %d-%.0f" % (lit[0], full[0]))
        if bottom > SLIT_END_MAX * n:
            ends.append("rows %.0f-%d" % (full[1], lit[1]))
        if ends:
            warn.append("The lit slit has no bars at %s: the keystone there is extrapolated. Use a target whose bars "
                        "reach the slit ends." % " and ".join(ends))
    return warn


# ----------------------------------------------------------------------------------------------- lamp points
@dataclass
class _LampLine:
    column: float          # detection column (id)
    y: np.ndarray          # raw rows
    x: np.ndarray          # raw centres
    coef: np.ndarray       # robust quadratic x(yn), yn = (y - y_ref) / sy


def lamp_points_from_analysis(analysis, min_coverage: float = MIN_LAMP_COVERAGE) -> Tuple[List[_LampLine], list]:
    """Lamp-line traces of a :class:`~modules.smile.analysis.SmileAnalysis` usable for the joint fit.

    Saturated lines are already excluded by the smile analysis; here lines traced over less than
    *min_coverage* of the rows, or not spanning the reference row, are left out too (their
    destination column would be extrapolated).

    Returns:
        ``(lines, rejected)``.
    """
    h = analysis.frame_orig.shape[0]
    y_ref = float(analysis.y_ref)
    sy = max(1.0, (h - 1) / 2.0)
    out, rejected = [], []
    for ln in analysis.used_lines:
        c = np.asarray(analysis.traces_orig[ln.column], float)
        rows = np.flatnonzero(np.isfinite(c))
        cov = rows.size / float(h)
        if cov < min_coverage or rows.size < 20 or not (rows.min() <= y_ref <= rows.max()):
            rejected.append((float(ln.column), "traced over %.0f %% of the slit" % (100 * cov)))
            continue
        y = rows.astype(float); x = c[rows]
        coef, keep = _robust_poly((y - y_ref) / sy, x, 2, floor=0.05)
        out.append(_LampLine(column=float(ln.column), y=y[keep], x=x[keep], coef=coef))
    out.sort(key=lambda t: t.coef[0])
    return out, rejected


# ----------------------------------------------------------------------------------------------- joint fit
def _huber_weights(r: np.ndarray, keep: np.ndarray, k: float = 1.5, reject: float = 4.0, floor: float = 0.02):
    sig = max(floor, robust_sigma(r[keep])) if keep.any() else floor
    u = np.abs(r) / (k * sig)
    w = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-9))
    w[np.abs(r) > reject * sig] = 0.0
    return w


def _soft_weights(r: np.ndarray, keep: np.ndarray, k: float = 2.5, floor: float = 0.12, low: float = 0.15):
    """Robust weights of the spline keystone: ``clip(k * scale / |r|, low, 1)``, ``scale`` the MAD
    sigma (at least *floor* px). Nothing is dropped and no bar as a whole is weighted down: with
    the spline a bar that scatters more is mostly a bar measured in a weak part of the spectrum,
    exactly where the keystone needs the data (block cross-validation on a real target)."""
    v = r[keep] if keep.any() else r
    scale = max(floor, 1.4826 * float(np.median(np.abs(v - np.median(v))))) if v.size else floor
    return np.clip(k * scale / np.maximum(np.abs(r), 1e-8), low, 1.0)


def _downweighted(r: np.ndarray, keep: np.ndarray) -> int:
    """Bar points the spline's soft weights cut below half (residual over ~5 robust sigma)."""
    return int(np.sum(_soft_weights(r, keep) < 0.5))


#: Floor (px) of a feature's scatter in the inverse-variance weights (no single line dominates).
FEATURE_SIGMA_FLOOR = 0.05


def _feature_weights(r: np.ndarray, keep: np.ndarray, idx: np.ndarray, n: int,
                     floor: float = FEATURE_SIGMA_FLOOR) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse-variance weight of every line / bar from its own scatter around the joint model.

    A target bar whose profile is asymmetric (a wide or two-level bar) has a centroid that moves
    with the wavelength-dependent blur: a smooth false keystone of its own. Such a bar scatters
    more around the common geometry and gets less weight, without a hard threshold.

    Returns:
        ``(weights, sigma)`` per feature; weights normalised to median 1 (0 for a feature with no
        kept point).
    """
    sig = np.full(n, np.nan)
    for i in range(n):
        m = keep & (idx == i)
        if m.any():
            sig[i] = _rms(r[m])
    w = np.where(np.isfinite(sig), 1.0 / np.maximum(np.nan_to_num(sig, nan=1.0), floor) ** 2, 0.0)
    pos = w[w > 0]
    if pos.size:
        w = w / float(np.median(pos))
    return w, sig


#: Candidate fitters of :func:`keystone_candidates`: name -> (family, x_degree, y_degree, reg, piecewise knots).
KEYSTONE_FITTERS = {
    "legacy_poly_x3_y1": ("polynomial", 3, 1, 0.003, 0),
    "legacy_poly_x3_y4": ("polynomial", 3, 4, 0.008, 0),
    "chebyshev_x3_y2": ("chebyshev", 3, 2, 0.004, 0),
    "chebyshev_x5_y3": ("chebyshev", 5, 3, 0.015, 0),
    "chebyshev_x6_y4": ("chebyshev", 6, 4, 0.025, 0),
    "piecewise_cubic_y2": ("piecewise", 3, 2, 0.02, 4),
    "piecewise_cubic_y4": ("piecewise", 3, 4, 0.04, 4),
    "anchored_bspline": ("spline", 0, 0, SPLINE_REG, 0),
}


def keystone_candidates(shape, x_ref: float, period: Optional[float] = None,
                        columns: Optional[Tuple[float, float]] = None) -> Dict[str, Tuple[object, int]]:
    """The keystone fitters :func:`select_keystone_fitter` chooses from: ``name -> (factory, complexity)``.

    Four families (polynomial, Chebyshev, C2 piecewise cubic along the spectrum, 2-D B-spline),
    each anchored exactly at *x_ref*. The piecewise knots are spread evenly over the measured
    *columns* (default: the frame), at 1/5 .. 4/5.
    """
    h, w = (int(v) for v in shape)
    lo, hi = columns if columns is not None else (0.0, w - 1.0)
    out: Dict[str, Tuple[object, int]] = {}
    for name, (fam, xd, yd, reg, nk) in KEYSTONE_FITTERS.items():
        if fam == "spline":
            out[name] = ((lambda: SplineSurface.for_frame((h, w), x_ref, period)), 999)
            continue
        knots = tuple(lo + (hi - lo) * np.arange(1, nk + 1) / (nk + 1)) if nk else ()
        out[name] = ((lambda fam=fam, xd=xd, yd=yd, reg=reg, knots=knots:
                      TensorSurface(fam, xd, yd, x_ref, (h, w), knots, reg)), (xd + len(knots)) * (yd + 1))
    return out


def _check_dy_model(dy_model: str) -> str:
    if dy_model == "spline":
        return "anchored_bspline"
    if dy_model in ("auto", "poly") or dy_model in KEYSTONE_FITTERS:
        return dy_model
    raise ValueError("dy_model must be 'auto', 'poly', 'spline' or one of: " + ", ".join(KEYSTONE_FITTERS))


def _fitter_note(qc: dict) -> str:
    """One line naming the keystone fitter and, for ``"auto"``, how it won."""
    name, rows = qc.get("dy_fitter"), [r for r in (qc.get("dy_selection") or []) if r.get("valid")]
    if not rows:
        return "keystone fitter: %s" % name
    r = next((r for r in rows if r["name"] == name), None)
    runner = sorted(rows, key=lambda r: (r["above"], r["max_px"]))
    other = next((o for o in runner if o["name"] != name), None)
    s = "keystone fitter: %s (auto, whole bars held out: %d of %d points over %.0f px, max %.2f px" % (
        name, r["above"], r["n"], SELECT_ERROR_PX, r["max_px"]) if r else "keystone fitter: %s (auto" % name
    if other:
        s += "; next %s: %d" % (other["name"], other["above"])
    return s + ")"


def _bar_rows(s, X, ty, idx: np.ndarray, w: Optional[np.ndarray], Ym: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Row of every bar at the reference column implied by the keystone surface *s*.

    With ``f(x_ref, Y) = 0`` a bar of destination row ``Y`` sits at raw ``ty = Y + f(X, Y)`` in
    every column, so ``Y = mean(ty - f(X, Y))`` over the bar's points (weights *w*; the median
    without). All of the bar's centres are used, not a cubic along the spectrum through them: a
    bar curved more than a cubic (mustache-type distortion) has a cubic that misses its row at
    ``x_ref`` (0.3-0.5 px on a test target), which then shifts the whole bar. Bars without points
    keep their value in *Ym*.
    """
    Ym = np.asarray(Ym, float).copy()
    n = Ym.size
    for _ in range(iterations):
        r = ty - s.raw(X, Ym[idx])
        if w is None:
            for k in np.unique(idx):
                Ym[k] = float(np.median(r[idx == k]))
        else:
            num = np.bincount(idx, w * r, n); den = np.bincount(idx, w, n)
            Ym = np.where(den > 0, num / np.maximum(den, 1e-12), Ym)
    return Ym


def _fit_rows(s, X, ty, idx: np.ndarray, w: np.ndarray, Ym: np.ndarray, passes: int = 2) -> np.ndarray:
    """Fit surface *s* and every bar's reference row in ONE weighted least squares; returns the rows.

    Model per bar point: ``ty = Y_k + f(X, Y_k)``, unknowns the surface coefficients and one
    ``Y_k`` per bar (the basis is evaluated at the previous ``Y_k``; *passes* re-linearise). Solving
    them together matters for a local basis (B-spline): its anchor holds only near ``x_ref``, so
    a bar offset and a dip of the surface at ``x_ref`` nearly trade places, and alternating
    between the two creeps along that valley (0.17 px left on a test target after 12 rounds);
    jointly, the centres near ``x_ref`` fix the rows at once. Bars without weight keep *Ym*.
    """
    Ym = np.asarray(Ym, float).copy()
    n = Ym.size
    idx = np.asarray(idx, int); w = np.asarray(w, float); ty = np.asarray(ty, float)
    E = sparse.csr_matrix((np.ones(idx.size), (np.arange(idx.size), idx)), shape=(idx.size, n))
    Wd = sparse.diags(w)
    den = np.bincount(idx, w, n)
    empty = den <= 0
    for _ in range(passes):
        A = s.design(X, Ym[idx])
        As = A if sparse.issparse(A) else sparse.csr_matrix(A)
        p = As.shape[1]
        M = np.zeros((p + n, p + n))
        M[:p, :p] = (As.T @ Wd @ As).toarray() + s._reg_matrix(A, w)
        AtWE = (As.T @ Wd @ E).toarray()
        M[:p, p:] = AtWE; M[p:, :p] = AtWE.T
        M[p:, p:] = np.diag(np.where(empty, 1.0, den))
        rhs = np.r_[As.T @ (w * ty), np.where(empty, Ym, np.bincount(idx, w * ty, n))]
        sol = np.linalg.solve(M, rhs)
        s.C = sol[:p].reshape(s.C.shape)
        Ym = sol[p:]
    return Ym


def _robust_fit_rows(s, X, ty, idx: np.ndarray, w0: np.ndarray, Ym: np.ndarray, iterations: int = 8):
    """:func:`_fit_rows` inside the soft x per-bar robust re-weighting; returns ``(weights, rows)``."""
    n = Ym.size
    wt = w0.copy()
    for it in range(iterations):
        new_ym = _fit_rows(s, X, ty, idx, wt, Ym)
        moved = float(np.max(np.abs(new_ym - Ym))); Ym = new_ym
        r = ty - (Ym[idx] + s.raw(X, Ym[idx]))
        sw = _soft_weights(r, wt > 0)
        ft, _ = _feature_weights(r, sw > 0, idx, n)
        new = sw * ft[idx] * w0
        if it >= 2 and moved < 1e-4 and np.allclose(new, wt, rtol=1e-3, atol=1e-6):
            wt = new
            break
        wt = new
    return wt, _fit_rows(s, X, ty, idx, wt, Ym)


def _robust_fit(s, X, Y, v, idx: np.ndarray, n: int, w0: np.ndarray, iterations: int = 8) -> np.ndarray:
    """Fit surface *s* with the solvers' robust weights (soft x per-bar x base, or Huber); returns the weights."""
    wt = w0.copy()
    for it in range(iterations):
        s.fit(X, Y, v, wt)
        r = v - s.raw(X, Y)
        pw = _soft_weights(r, wt > 0) if getattr(s, "robust", "huber") == "soft" else _huber_weights(r, wt > 0)
        ft, _ = _feature_weights(r, pw > 0, idx, n)
        new = pw * ft[idx] * w0
        if it >= 2 and np.array_equal(new > 0, wt > 0) and np.allclose(new, wt, rtol=1e-3, atol=1e-6):
            wt = new
            break
        wt = new
    s.fit(X, Y, v, wt)
    return wt


def _map_check(s, shape, x_ref: float, min_jac: float = 0.1, max_jac: float = 2.0) -> dict:
    """Coarse safety of a keystone surface: row Jacobian, largest shift, anchor."""
    h, w = shape
    xs = np.unique(np.r_[np.linspace(0, w - 1, 90), x_ref]); ys = np.linspace(0, h - 1, 85)
    g = s.grid(xs, ys)
    jac = 1.0 + np.diff(g, axis=0) / (ys[1] - ys[0])
    anchor = float(np.max(np.abs(s.at(np.full_like(ys, x_ref), ys))))
    shift = float(np.max(np.abs(g)))
    ok = bool(np.isfinite(g).all() and min_jac <= jac.min() and jac.max() <= max_jac
              and shift <= max(30.0, 0.03 * h) and anchor < 1e-5)
    return dict(valid=ok, jacobian_min=float(jac.min()), jacobian_max=float(jac.max()), max_shift_px=shift,
                anchor_px=anchor)


def _corrected_row_error(s, X, ty, Yt) -> np.ndarray:
    """|destination row of a raw bar point - its bar's row|, the surface inverted by fixed point (uncensored)."""
    est = ty.copy()
    for _ in range(10):
        est = ty - s.at(X, est)
    return np.abs(est - Yt)


def select_keystone_fitter(candidates: Dict[str, Tuple[object, int]], Xt, Yt, ty, tm, w0, shape, x_ref: float,
                           folds: int = SELECT_FOLDS, limit: float = SELECT_ERROR_PX) -> Tuple[Optional[str], List[dict]]:
    """Choose the keystone fitter by cross-validation over WHOLE bars (one pass, no re-tracing).

    The bars (sorted along the slit) are dealt round-robin into *folds*; each fold's bars are left
    out entirely, every candidate is fitted on the other bars with the same robust weights as the
    final fit (the training bars' rows estimated with its own surface, :func:`_bar_rows`), and
    the held-out bar points are corrected (the surface inverted) and compared with their bar's
    row, one value per held-out bar (the median): what is scored is how well the candidate
    predicts the SHAPE of a bar it never saw. Nothing is censored. A candidate whose map folds (row Jacobian outside 0.1-2),
    jumps more than 30 px or leaves the anchor is invalid. Ranking: the count of held-out errors
    over *limit* px, then the 95th percentile, the largest error, the complexity, the name. (The
    95th percentile before the largest error: when the counts tie, the largest error is one point,
    usually of one biased bar, and on a test target it picked a fitter 3.5x worse at the 95th
    percentile and 0.14 px off the true map, against 0.003-0.012 px for the others.)

    These are consistency errors on the same target (tracker / bar-identity errors included), not
    optical ground truth.

    Returns:
        ``(name or None, rows)``: ``None`` when fewer than ``3 * folds`` bars or no valid candidate.
    """
    tm = np.asarray(tm, int)
    ids = np.unique(tm)
    rows: List[dict] = []
    if ids.size < 3 * folds:
        return None, rows
    n = int(tm.max()) + 1
    ym0 = np.zeros(n)                                   # starting row of every bar (from the given Yt)
    ym0[tm] = Yt
    for name, (factory, complexity) in candidates.items():
        errs = []; kinds = []; checks = []
        try:
            for f in range(folds):
                held = np.isin(tm, ids[np.arange(ids.size) % folds == f])
                tr = ~held
                s = factory()
                if getattr(s, "robust", "huber") == "soft":     # training bars' rows solved with the surface
                    _, ym = _robust_fit_rows(s, Xt[tr], ty[tr], tm[tr], w0[tr], ym0.copy())
                else:
                    ym = ym0.copy()
                    _robust_fit(s, Xt[tr], ym[tm[tr]], ty[tr] - ym[tm[tr]], tm[tr], n, w0[tr])
                y_tr = ym[tm[tr]]
                s.box = [float(np.percentile(Xt[tr], 0.5)), float(np.percentile(Xt[tr], 99.5)),
                         float(y_tr.min()), float(y_tr.max())]
                chk = _map_check(s, shape, x_ref)
                checks.append(chk)
                if not chk["valid"]:
                    raise ValueError("unsafe map: jacobian %.2f-%.2f, shift %.1f px, anchor %.1e px"
                                     % (chk["jacobian_min"], chk["jacobian_max"], chk["max_shift_px"], chk["anchor_px"]))
                # held-out bars: one row each (median, no shape), so the error is the bar's SHAPE misfit
                ym_h = _bar_rows(s, Xt[held], ty[held], tm[held], None, ym)
                e = _corrected_row_error(s, Xt[held], ty[held], ym_h[tm[held]])
                if not np.all(np.isfinite(e)):
                    raise ValueError("non-finite held-out errors")
                errs.append(e); kinds.append(w0[held] < 1.0)
            e = np.concatenate(errs); rec = np.concatenate(kinds)
            rows.append(dict(name=name, valid=True, above=int(np.sum(e > limit)),
                             above_recovered=int(np.sum((e > limit) & rec)), above_2px=int(np.sum(e > 2.0)),
                             above_5px=int(np.sum(e > 5.0)), max_px=float(e.max()), p95_px=float(np.percentile(e, 95)),
                             median_px=float(np.median(e)), n=int(e.size), complexity=int(complexity),
                             jacobian_min=float(min(c["jacobian_min"] for c in checks))))
        except (ValueError, np.linalg.LinAlgError) as exc:
            rows.append(dict(name=name, valid=False, reason=str(exc)))
    valid = [r for r in rows if r["valid"]]
    if not valid:
        return None, rows
    best = min(valid, key=lambda r: (r["above"], round(r["p95_px"], 4), r["max_px"], r["complexity"], r["name"]))
    return best["name"], rows


def _surfaces(shape, y_ref, x_ref, dx_deg, dy_deg):
    h, w = shape
    sx, sy = max(1.0, (w - 1) / 2.0), max(1.0, (h - 1) / 2.0)
    xc, yc = (w - 1) / 2.0, (h - 1) / 2.0
    # dx: powers of (Y - y_ref) start at 1 -> dx(X, y_ref) = 0; any power of X (centred)
    sdx = GeometrySurface(range(0, dx_deg[1] + 1), range(1, dx_deg[0] + 1), xc, sx, y_ref, sy)
    # dy: powers of (X - x_ref) start at 1 -> dy(x_ref, Y) = 0; any power of Y (centred)
    sdy = GeometrySurface(range(1, dy_deg[1] + 1), range(0, dy_deg[0] + 1), x_ref, sx, yc, sy)
    return sdx, sdy


class _Data:
    def __init__(self, lamp: List[_LampLine], target: List[TargetLine], shape, y_ref):
        h, w = shape
        self.h, self.w = h, w
        self.y_ref = float(y_ref)
        self.sy = max(1.0, (h - 1) / 2.0)
        self.lamp, self.target = lamp, target
        self.ly = np.concatenate([t.y for t in lamp]); self.lx = np.concatenate([t.x for t in lamp])
        self.lk = np.concatenate([np.full(t.y.size, i) for i, t in enumerate(lamp)])
        self.tx = np.concatenate([t.x for t in target]); self.ty = np.concatenate([t.y for t in target])
        self.tm = np.concatenate([np.full(t.x.size, i) for i, t in enumerate(target)])
        self.tw0 = _point_weights(target)

    def lamp_x_at(self, k: int, y: float) -> float:
        return float(np.polynomial.polynomial.polyval((y - self.y_ref) / self.sy, self.lamp[k].coef))

    def bar_y_at(self, m: int, x: float) -> float:
        return float(self.target[m].y_at([x], self.w)[0])


def _solve(data: _Data, sdx: GeometrySurface, sdy: GeometrySurface, x_ref: float, iterations: int = 10):
    y_ref = data.y_ref
    Xk = np.array([data.lamp_x_at(k, y_ref) for k in range(len(data.lamp))])
    Ym = np.array([data.bar_y_at(m, x_ref) for m in range(len(data.target))])
    wl = np.ones(data.lx.size); wt = data.tw0.copy()
    for it in range(iterations):
        Xl = Xk[data.lk]
        Yl = data.ly.copy()
        for _ in range(3):
            Yl = data.ly - sdy.raw(Xl, Yl)
        sdx.fit(Xl, Yl, data.lx - Xl, wl)
        Yt = Ym[data.tm]
        Xt = data.tx.copy()
        for _ in range(3):
            Xt = data.tx - sdx.raw(Xt, Yt)
        soft = getattr(sdy, "robust", "huber") == "soft"
        if soft:                                        # surface and bar rows in one solve (see _fit_rows)
            newY = _fit_rows(sdy, Xt, data.ty, data.tm, wt, Ym)
        else:
            sdy.fit(Xt, Yt, data.ty - Yt, wt)
        newX = np.array([data.lamp_x_at(k, y_ref + float(sdy.raw(Xk[k], y_ref))) for k in range(len(data.lamp))])
        if not soft:                                    # the earlier polynomial: each bar's cubic at x_ref
            newY = np.array([data.bar_y_at(m, x_ref + float(sdx.raw(x_ref, Ym[m]))) for m in range(len(data.target))])
        moved = max(float(np.max(np.abs(newX - Xk))), float(np.max(np.abs(newY - Ym))))
        Xk, Ym = newX, newY
        if it >= 1:
            rl = data.lx - (Xl + sdx.raw(Xl, Yl))
            rt = data.ty - (Yt + sdy.raw(Xt, Yt))
            hl = _huber_weights(rl, wl > 0)
            fl, _ = _feature_weights(rl, hl > 0, data.lk, len(data.lamp))
            wl = hl * fl[data.lk]
            if getattr(sdy, "robust", "huber") == "soft":
                st_ = _soft_weights(rt, wt > 0)
                ft, _ = _feature_weights(rt, st_ > 0, data.tm, len(data.target))
                wt = st_ * ft[data.tm] * data.tw0
            else:
                ht = _huber_weights(rt, wt > 0)
                ft, _ = _feature_weights(rt, ht > 0, data.tm, len(data.target))
                wt = ht * ft[data.tm] * data.tw0
        if it >= 3 and moved < 1e-4:
            break
    # final destination coordinates and residuals with the final anchors
    Xl = Xk[data.lk]; Yl = data.ly.copy()
    for _ in range(3):
        Yl = data.ly - sdy.raw(Xl, Yl)
    Yt = Ym[data.tm]; Xt = data.tx.copy()
    for _ in range(3):
        Xt = data.tx - sdx.raw(Xt, Yt)
    rl = data.lx - (Xl + sdx.raw(Xl, Yl))
    rt = data.ty - (Yt + sdy.raw(Xt, Yt))
    fl, sl = _feature_weights(rl, wl > 0, data.lk, len(data.lamp))
    ft, st = _feature_weights(rt, wt > 0, data.tm, len(data.target))
    return dict(Xk=Xk, Ym=Ym, Xl=Xl, Yl=Yl, Xt=Xt, Yt=Yt, rl=rl, rt=rt, wl=wl, wt=wt,
                lamp_weight=fl, lamp_sigma=sl, bar_weight=ft, bar_sigma=st)


def _rms(v) -> float:
    v = np.asarray(v, float)
    return float(np.sqrt(np.mean(v * v))) if v.size else float("nan")


def _loo(data: _Data, sol: dict, make, which: str):
    """Leave-one-line-out RMS (px) of one surface on the solved destination coordinates."""
    if which == "x":
        n, idx, X, Y, v, w = len(data.lamp), data.lk, sol["Xl"], sol["Yl"], data.lx - sol["Xl"], sol["wl"]
    else:
        n, idx, X, Y, v, w = len(data.target), data.tm, sol["Xt"], sol["Yt"], data.ty - sol["Yt"], sol["wt"]
    if n < 3:
        return None, []
    per = []
    for i in range(n):
        s = make()
        m = (idx != i) & (w > 0)
        try:
            s.fit(X[m], Y[m], v[m], w[m])
        except np.linalg.LinAlgError:
            return None, []
        sel = (idx == i) & (w > 0)
        if sel.any():                       # a line / bar the robust weights dropped entirely has nothing to predict
            per.append(_rms(s.raw(X[sel], Y[sel]) - v[sel]))
    return (_rms(per) if per else None), per


def fit_geometry(lamp: List[_LampLine], target: List[TargetLine], shape, y_ref: float, x_ref: Optional[float] = None,
                 dx_deg: Tuple[int, int] = (2, 1), dy_xdeg: int = 3, dy_ydeg="auto", dy_model: str = DEFAULT_DY_MODEL,
                 period: Optional[float] = None) -> Tuple[GeometryModel, dict]:
    """Joint smile + keystone model from lamp-line and target-bar traces of one optical state.

    Args:
        lamp: Lamp lines (:func:`lamp_points_from_analysis`), at least two.
        target: Target bars (:func:`trace_target_lines`), at least three.
        shape: ``(H, W)``.
        y_ref: Reference row (dx = 0).
        x_ref: Reference column (dy = 0); ``None``: the frame centre, moved inside the bars' span.
        dx_deg: ``(deg_y, deg_x)`` of the smile (deg_y >= 1).
        dy_xdeg: Degree of the keystone along the spectrum (>= 1).
        dy_ydeg: Degree along the slit, or ``"auto"`` (0 to ``MAX_DY_YDEG`` by leave-one-bar-out:
            the lowest degree within 10 % of the best).
        dy_model: ``"auto"`` (:func:`select_keystone_fitter` over :func:`keystone_candidates`), a
            candidate name, ``"spline"`` or ``"poly"`` (the earlier polynomial: *dy_xdeg* / *dy_ydeg*;
            unused otherwise).
        period: Bar period (px), keeps the spline knots at least three periods apart along the slit.

    Returns:
        ``(model, qc)``; ``qc`` holds fit / leave-one-line-out residuals in px.
    """
    h, w = (int(v) for v in shape)
    if len(lamp) < 2:
        raise ValueError("At least two lamp lines covering the slit are needed for the joint geometry.")
    if len(target) < 3:
        raise ValueError("At least three target bars are needed for the keystone.")
    t_lo = float(np.median([t.x.min() for t in target])); t_hi = float(np.median([t.x.max() for t in target]))
    if x_ref is None:
        x_ref = float(np.clip((w - 1) / 2.0, t_lo, t_hi))
    x_ref = float(x_ref)
    dx_deg = (max(1, int(dx_deg[0])), max(0, int(dx_deg[1])))
    dx_deg = (dx_deg[0], min(dx_deg[1], len(lamp) - 1))
    dy_xdeg = max(1, int(dy_xdeg))
    data = _Data(lamp, target, (h, w), y_ref)
    Xk0 = [data.lamp_x_at(k, y_ref) for k in range(len(lamp))]
    Ym0 = [data.bar_y_at(m, x_ref) for m in range(len(target))]
    if np.any(np.diff(Xk0) < 1) or np.any(np.diff(sorted(Ym0)) < 1):
        raise ValueError("Lamp lines or target bars are duplicated / unresolved. Remove merged lines and rebuild.")
    mode = _check_dy_model(dy_model)
    fitter, selection = None, []
    if mode != "poly":
        cands = keystone_candidates((h, w), x_ref, period,
                                    (float(np.percentile(data.tx, 0.5)), float(np.percentile(data.tx, 99.5))))
        if mode == "auto":
            sdx0 = _surfaces((h, w), y_ref, x_ref, dx_deg, (0, 1))[0]
            sol0 = _solve(data, sdx0, cands["anchored_bspline"][0](), x_ref)
            fitter, selection = select_keystone_fitter(cands, sol0["Xt"], sol0["Yt"], data.ty, data.tm, data.tw0,
                                                       (h, w), x_ref)
            fitter = fitter or "anchored_bspline"
        else:
            fitter = mode
        factory = cands[fitter][0]

        def surfaces(d=None):
            return _surfaces((h, w), y_ref, x_ref, dx_deg, (0, 1))[0], factory()
        cand, d_y = [], None
    elif dy_ydeg == "auto":
        # odd powers along the slit are lens distortion of the slit image that changes with
        # wavelength (lateral colour); a real GigE spectrograph needed y**3..y**4
        cand = [d for d in range(0, MAX_DY_YDEG + 1) if d <= max(0, len(target) - 3)]
        d_y = min(1, cand[-1])                  # start of the leave-one-bar-out choice
    else:
        cand = [max(0, min(int(dy_ydeg), len(target) - 2))]
        d_y = cand[0]
    if fitter is None:
        def surfaces(d=None):
            return _surfaces((h, w), y_ref, x_ref, dx_deg, (d_y if d is None else d, dy_xdeg))
    sdx, sdy = surfaces()
    sol = _solve(data, sdx, sdy, x_ref)
    choice = {}
    if len(cand) > 1:
        res = {d: _loo(data, sol, lambda d=d: surfaces(d)[1], "y")[0] for d in cand}
        valid = {d: v for d, v in res.items() if v is not None and np.isfinite(v)}
        if valid:
            best = min(valid, key=valid.get)
            for d in sorted(valid):
                if valid[d] <= 1.1 * valid[best]:
                    best = d
                    break
            d_y = best
        choice = {str(k): v for k, v in res.items()}
    sdx, sdy = surfaces()
    sol = _solve(data, sdx, sdy, x_ref)
    # measured boxes (destination coordinates)
    sdx.box = [float(sol["Xk"].min()), float(sol["Xk"].max()), float(sol["Yl"].min()), float(sol["Yl"].max())]
    sdy.box = [float(np.percentile(sol["Xt"], 0.5)), float(np.percentile(sol["Xt"], 99.5)),
               float(sol["Ym"].min()), float(sol["Ym"].max())]
    loo_x, per_x = _loo(data, sol, lambda: surfaces()[0], "x")
    loo_y, per_y = _loo(data, sol, lambda: surfaces()[1], "y")
    kl, kt = sol["wl"] > 0, sol["wt"] > 0
    qc = dict(
        lamp_lines=len(lamp), target_bars=len(target),
        before_lamp_rms_px=_rms(data.lx - sol["Xl"]), before_target_rms_px=_rms(data.ty - sol["Yt"]),
        fit_lamp_rms_px=_rms(sol["rl"][kl]), fit_target_rms_px=_rms(sol["rt"][kt]),
        fit_lamp_max_px=float(np.max(np.abs(sol["rl"][kl]))) if kl.any() else float("nan"),
        fit_target_max_px=float(np.max(np.abs(sol["rt"][kt]))) if kt.any() else float("nan"),
        loo_lamp_rms_px=loo_x, loo_target_rms_px=loo_y, loo_lamp_per_line_px=per_x, loo_target_per_bar_px=per_y,
        rejected_points=int((~kl).sum() + (~kt).sum()),
        downweighted_points=_downweighted(sol["rt"], kt) if fitter else 0,
        degrees=dict(dx=list(dx_deg), dy=(fitter or [d_y, dy_xdeg])), dy_model=dy_model, dy_fitter=fitter or "poly",
        dy_selection=selection, dy_ydeg_loo=choice,
        fit_lamp_per_line_px=[float(v) for v in sol["lamp_sigma"]],
        fit_target_per_bar_px=[float(v) for v in sol["bar_sigma"]],
        lamp_weight=[float(v) for v in sol["lamp_weight"]], bar_weight=[float(v) for v in sol["bar_weight"]])
    model = GeometryModel(shape=(h, w), y_ref=float(y_ref), x_ref=x_ref, dx=sdx, dy=sdy,
                          lamp_X=[float(v) for v in sol["Xk"]], target_Y=[float(v) for v in sol["Ym"]])
    return model, qc


# ----------------------------------------------------------------------------------------------- checks
def row_margin_from_map(map_y: np.ndarray, interp: str = "cubic", columns: Optional[Tuple[int, int]] = None
                        ) -> Tuple[int, int]:
    """Rows at the top / bottom whose source lies outside the frame in some valid column.

    Args:
        map_y: Row map ``(H, W)``.
        interp: Interpolation name (its kernel needs ``pad`` rows inside the frame).
        columns: ``(first, last)`` columns to consider (the valid columns); ``None`` = all.

    Returns:
        ``(top, bottom)`` row counts.
    """
    _, pad_l, pad_r = _INTERP[interp]
    h = map_y.shape[0]
    sel = map_y if columns is None else map_y[:, int(columns[0]):int(columns[1]) + 1]
    hi = h - 1.0 if interp == "linear" else h - pad_r - 1e-3
    ok = ((sel >= float(pad_l)) & (sel <= hi)).all(axis=1)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        raise ValueError("No valid rows after keystone correction: the shift is too large.")
    return int(idx[0]), int(h - 1 - idx[-1])


def valid_rect(map_x: np.ndarray, map_y: np.ndarray, interp: str = "cubic"
               ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Crop of a joint (smile + keystone) remap: ``((left, right), (top, bottom))``.

    Both maps are joined first: a pixel is valid when its interpolation kernel lies inside the
    frame along both axes (``map_x`` within the columns, ``map_y`` within the rows, with the
    same bounds as :func:`~modules.smile.calibration.margin_from_map` and
    :func:`row_margin_from_map`). The rows come from the valid columns of ``map_x`` over every
    row, as before; the columns are then widened as far as the joint mask stays valid in the rows
    kept. The slit-end rows the row margin removes carry the largest smile, so they no longer
    narrow the spectral range. The rectangle is never smaller than the separate crop along either
    axis, and every pixel in it is valid.

    Args:
        map_x: Column map ``(H, W)``.
        map_y: Row map ``(H, W)``.
        interp: Interpolation name.

    Returns:
        ``((left, right), (top, bottom))`` column and row counts to remove.

    Raises:
        ValueError: No valid column or row.
    """
    from .calibration import margin_from_map
    h, w = map_x.shape
    left, right = margin_from_map(map_x, interp)
    top, bottom = row_margin_from_map(map_y, interp, (left, w - 1 - right))
    _, pad_l, pad_r = _INTERP[interp]

    def inside(m, n):
        hi = n - 1.0 if interp == "linear" else n - pad_r - 1e-3
        return (m >= float(pad_l)) & (m <= hi)

    rows = slice(top, h - bottom)
    ok = (inside(map_x[rows], w) & inside(map_y[rows], h)).all(axis=0)
    first, last = left, w - 1 - right            # all valid already (rows chosen inside them)
    while first > 0 and ok[first - 1]:
        first -= 1
    while last < w - 1 and ok[last + 1]:
        last += 1
    return (int(first), int(w - 1 - last)), (int(top), int(bottom))


def _remap(img, mx, my, interp="cubic"):
    return cv2.remap(np.ascontiguousarray(img, np.float32), mx, my, interpolation=_INTERP[interp][0],
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _retrace(model: GeometryModel, lamp_corr, target_norm_corr, valid, lamp_halfwins, sat_level, noise, snr_min,
             polarity, band, bar_halfwins, step):
    """Re-measure lines on the REMAPPED frames: lamp lines must be vertical at X, bars horizontal at Y."""
    h, w = model.shape
    out = dict(lamp=[], target=[])
    per = dict(lamp=[], target=[])
    lo, hi = (model.dx.box[2], model.dx.box[3]) if model.dx.box else (0, h - 1)
    for X, hw in zip(model.lamp_X, lamp_halfwins):
        c = trace_line(lamp_corr, X, hw, sat_level, noise, snr_min)
        rows = np.flatnonzero(np.isfinite(c))
        rows = rows[(rows >= lo) & (rows <= hi)]
        cx = int(round(X))
        rows = rows[valid[rows, min(max(cx, 0), w - 1)]]
        v = c[rows] - X
        out["lamp"].extend(v.tolist())
        per["lamp"].append(dict(X=float(X), rms_px=_rms(v), max_px=float(np.max(np.abs(v))) if v.size else float("nan"),
                                n=int(v.size)))
    sign = -1.0 if polarity == "dark" else 1.0
    img = sign * _band_columns(target_norm_corr, band)
    noise_col = _column_noise(img)
    c0, c1 = model.dy.box[0], model.dy.box[1]
    cols = np.arange(int(math.ceil(c0)), int(math.floor(c1)) + 1, max(1, step))
    for Y, hw in zip(model.target_Y, bar_halfwins):
        vals = []
        r = int(round(Y))
        hw = int(hw)
        max_jump = max(1.0, 0.5 * hw)
        for c in cols:
            if r - hw < 0 or r + hw >= h or not valid[r, c]:
                continue
            yy, ok = _centroid_row(img[:, c], Y, hw, np.inf, snr_min * float(noise_col[c]))
            if ok and abs(yy - Y) <= max_jump:       # same acceptance as the tracer: never a neighbour
                vals.append(yy - Y)
        v = np.asarray(vals)
        out["target"].extend(vals)
        per["target"].append(dict(Y=float(Y), rms_px=_rms(v), max_px=float(np.max(np.abs(v))) if v.size else float("nan"),
                                  n=int(v.size)))
    lam, tar = np.asarray(out["lamp"]), np.asarray(out["target"])
    p95 = lambda v: float(np.percentile(np.abs(v), 95)) if v.size else float("nan")
    return dict(lamp_rms_px=_rms(lam), lamp_max_px=float(np.max(np.abs(lam))) if lam.size else float("nan"),
                target_rms_px=_rms(tar), target_max_px=float(np.max(np.abs(tar))) if tar.size else float("nan"),
                lamp_p95_px=p95(lam), target_p95_px=p95(tar),
                lamp_per_line=per["lamp"], target_per_bar=per["target"])


def _summary(model: GeometryModel, map_x, map_y) -> dict:
    h, w = model.shape
    xs = np.arange(w)[None, :]; ys = np.arange(h)[:, None]
    dx = map_x - xs; dy = map_y - ys
    bx, by = model.dx.box, model.dy.box
    x0, x1 = int(math.ceil(bx[0])), int(math.floor(bx[1]))
    c0, c1 = int(math.ceil(by[0])), int(math.floor(by[1]))
    r0, r1 = int(math.ceil(by[2])), int(math.floor(by[3]))
    smile = float(np.max(np.abs(dx[:, x0:x1 + 1]))) if x1 >= x0 else 0.0
    keystone = float(np.max(np.abs(dy[r0:r1 + 1, c0:c1 + 1]))) if (r1 >= r0 and c1 >= c0) else 0.0
    mid = int(round((r0 + r1) / 2))
    span = max(1, c1 - c0)
    tilt = float(dy[mid, c1] - dy[mid, c0])
    top = float(dy[r0, c1] - dy[r0, c0]); bottom = float(dy[r1, c1] - dy[r1, c0])
    jac = float(np.min(np.diff(map_y[:, c0:c1 + 1], axis=0))) if c1 >= c0 and h > 1 else 1.0
    out = dict(smile_max_px=smile, keystone_max_px=keystone, tilt_px=tilt, row_jacobian_min=jac,
               tilt_deg=float(math.degrees(math.atan2(tilt, span))), keystone_top_bottom_px=float(top - bottom),
               lamp_columns=[bx[0], bx[1]], target_columns=[by[0], by[1]], target_rows=[by[2], by[3]],
               extrapolated_columns=float((max(0.0, bx[0]) + max(0.0, (w - 1) - bx[1])) / max(1, w)))
    if abs(tilt) >= 2.0 and abs(out["keystone_top_bottom_px"]) < 0.35 * abs(tilt):
        out["verdict"] = ("Mostly tilt: the slit image is rotated %.2f deg on the detector; rotating the camera by "
                          "that angle would remove most of it." % out["tilt_deg"])
    elif abs(out["keystone_top_bottom_px"]) >= 1.0:
        out["verdict"] = "Keystone: the top and bottom of the slit move differently across the spectrum."
    elif keystone >= 1.0:
        out["verdict"] = "Mostly a uniform shift / tilt of the slit across the spectrum."
    else:
        out["verdict"] = "Small keystone: under 1 px in the measured area."
    return out


# ----------------------------------------------------------------------------------------------- analysis
@dataclass
class KeystoneAnalysis:
    """Result of :func:`analyze_keystone` (the joint smile + keystone geometry).

    Attributes:
        model: The fitted :class:`GeometryModel`.
        map_x, map_y: Destination -> source maps ``float32 (H, W)``.
        margin: ``(left, right)`` columns without a full source (as the smile-only model).
        row_margin: ``(top, bottom)`` rows without a full source in the valid columns.
        target_lines: Bars used.
        qc: Fit / leave-one-out / re-trace numbers (px); see :func:`keystone_report`.
        summary: Smile / keystone / tilt at a glance.
        warnings: Human-readable problems (empty when the result looks sound).
        target_corr, lamp_corr: Remapped frames (for display), float32.
        target_norm: Normalised raw target (for display), float32.
    """

    model: GeometryModel
    map_x: np.ndarray
    map_y: np.ndarray
    margin: Tuple[int, int]
    row_margin: Tuple[int, int]
    target_lines: List[TargetLine]
    qc: dict
    summary: dict
    warnings: List[str]
    target_corr: np.ndarray
    lamp_corr: np.ndarray
    target_norm: np.ndarray
    interp: str = "cubic"
    polarity: str = "dark"
    created: str = ""
    inputs: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        """JSON-able description stored in the model file (no images)."""
        qc = copy.deepcopy(self.qc)
        return dict(model=self.model.to_json(), margin=[int(v) for v in self.margin],
                    row_margin=[int(v) for v in self.row_margin], qc=_jsonable(qc), summary=_jsonable(self.summary),
                    warnings=list(self.warnings), interp=self.interp, polarity=self.polarity, created=self.created,
                    inputs=_jsonable(self.inputs),
                    target_bars=[dict(y_seed=t.y_seed, coverage=t.coverage, n=int(t.x.size)) for t in self.target_lines])


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, np.floating):
        v = float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


@logged_analysis("keystone (joint smile + keystone)")
def analyze_keystone(analysis, target, white=None, dark=None, polarity: str = "auto", x_ref: Optional[float] = None,
                     dy_xdeg: int = 3, dy_ydeg="auto", band: int = 5, step: Optional[int] = None,
                     snr_min: Optional[float] = None, interp: Optional[str] = None, inputs: Optional[dict] = None,
                     wavelength=None, dy_model: str = DEFAULT_DY_MODEL, gap_recovery: bool = True) -> KeystoneAnalysis:
    """Joint smile + keystone geometry from a smile analysis and a line-target frame.

    Args:
        analysis: :class:`~modules.smile.analysis.SmileAnalysis` of the lamp frame (same camera setting).
        target: Line-target frame ``(H, W)``; several frames should be averaged beforehand.
        white: Optional white frame (target removed) for normalisation; strongly recommended.
        dark: Optional dark frame subtracted from target and white (not from the lamp, which the
            smile analysis already used as given).
        polarity: ``"dark"`` or ``"bright"`` bars, or ``"auto"`` (decided from the image).
        x_ref: Reference column (dy = 0); ``None``: frame centre within the bars' span.
        dy_xdeg, dy_ydeg: Keystone degrees (see :func:`fit_geometry`).
        band, step: Column averaging / step of the bar tracing.
        snr_min: Bar / line detection threshold; ``None``: the smile analysis value (5).
        interp: Remap interpolation for the check images; ``None``: the smile analysis's.
        inputs: Free description of the input files (stored in the model).
        wavelength: Optional ``callable(column) -> nm`` to name the spectral range in coverage warnings.
        dy_model: Keystone fitter, ``"auto"`` (default), a candidate name, ``"spline"`` or ``"poly"``
            (see :func:`fit_geometry`).
        gap_recovery: Add measured bar centres where the tracer lost a bar (:func:`recover_bar_gaps`).

    Returns:
        :class:`KeystoneAnalysis`.

    Raises:
        ValueError: Unusable frames (sizes differ, too few lines / bars, coverage too small).
    """
    _check_dy_model(dy_model)
    lamp = analysis.frame_orig
    h, w = lamp.shape
    t = np.asarray(target, np.float32)
    if t.shape != (h, w):
        raise ValueError(f"The target frame {t.shape} differs from the lamp frame {(h, w)}: the same camera "
                         "setting (orientation, bin, ROI) is needed.")
    interp = interp or analysis.interp
    snr = 5.0 if snr_min is None else float(snr_min)
    notes: List[str] = []
    lamp_lines, lamp_rej = lamp_points_from_analysis(analysis)
    for col, why in lamp_rej:
        notes.append("lamp line at column %.0f not used for the joint fit: %s" % (col, why))
    norm = normalize_target(t, white, dark)
    bars, tinfo = trace_target_lines(t, polarity=polarity, band=band, step=step, snr_min=snr,
                                     normalized=norm, sat_level=None)
    polarity = tinfo["polarity"]
    notes.append(_polarity_note(tinfo))
    for y, why in tinfo["rejected"]:
        notes.append("target bar near row %.0f not used: %s" % (y, why))
    if x_ref is None and bars:                          # from the tracer's span, as without recovery
        x_ref = float(np.clip((w - 1) / 2.0, np.median([b.x.min() for b in bars]), np.median([b.x.max() for b in bars])))
    bars, gap = _recover(bars, tinfo, norm, gap_recovery, notes)
    sm = analysis.surface
    model, qc = fit_geometry(lamp_lines, bars, (h, w), analysis.y_ref, x_ref=x_ref,
                             dx_deg=(max(1, sm.deg_y), sm.deg_x), dy_xdeg=dy_xdeg, dy_ydeg=dy_ydeg,
                             dy_model=dy_model, period=tinfo["period"])
    qc["gap_recovery"] = gap
    notes.append(_fitter_note(qc))
    mx, my = model.maps()
    margin, row_margin = valid_rect(mx, my, interp)
    lamp_corr = _remap(lamp, mx, my, interp)
    target_corr = _remap(norm, mx, my, interp)
    valid = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    by_col = {float(ln.column): ln for ln in analysis.used_lines}
    halfwins = []
    for L in lamp_lines:
        ln = by_col.get(L.column)
        halfwins.append(max(3, int(round(1.2 * ln.fwhm))) if ln is not None else 5)
    rt = _retrace(model, lamp_corr, target_corr, valid, halfwins, analysis.sat_level, analysis.noise, snr,
                  polarity, tinfo["band"], [t.halfwin or tinfo["halfwin"] for t in bars], tinfo["step"])
    qc.update(retrace_lamp_rms_px=rt["lamp_rms_px"], retrace_lamp_max_px=rt["lamp_max_px"],
              retrace_target_rms_px=rt["target_rms_px"], retrace_target_max_px=rt["target_max_px"],
              retrace_lamp_p95_px=rt["lamp_p95_px"], retrace_target_p95_px=rt["target_p95_px"],
              retrace_lamp_per_line=rt["lamp_per_line"], retrace_target_per_bar=rt["target_per_bar"],
              bar_period_px=tinfo["period"], bar_halfwin_px=tinfo["halfwin"], seed_column=tinfo["seed_column"],
              normalised_with_white=white is not None, notes=notes, merged_bar_pairs=tinfo.get("merged", []))
    # the median line / bar: robust to a few features whose own shape biases their centroid
    med = lambda per: (float(np.nanmedian([p["rms_px"] for p in per])) if per else float("nan"))
    qc["retrace_lamp_median_line_px"] = med(rt["lamp_per_line"])
    qc["retrace_target_median_bar_px"] = med(rt["target_per_bar"])
    for label, where_name, per in (("target bars", "rows", rt["target_per_bar"]),
                                   ("lamp lines", "columns", rt["lamp_per_line"])):
        weak = [p for p in per if np.isfinite(p["rms_px"]) and p["rms_px"] > 2 * RETRACE_WARN_PX]
        if weak:
            where = ", ".join("%.0f" % (p.get("Y", p.get("X"))) for p in weak[:12])
            notes.append("%d %s scatter over %.2f px RMS after correction (asymmetric profile or weak signal; "
                         "the fit weights them down), %s %s" % (len(weak), label, 2 * RETRACE_WARN_PX,
                                                               where_name, where))
    summ = _summary(model, mx, my)
    warn: List[str] = []
    if summ["row_jacobian_min"] < ROW_JACOBIAN_WARN:
        warn.append("The row map compresses the slit to %.2f of its spacing somewhere (folds below 0): the "
                    "keystone surface follows inconsistent bar centres there; check the target / focus."
                    % summ["row_jacobian_min"])
    for key, label in (("retrace_lamp_median_line_px", "lamp lines"), ("retrace_target_median_bar_px", "target bars")):
        v = qc.get(key)
        if v is None or not np.isfinite(v):
            warn.append("The %s could not be re-measured on the corrected frame." % label)
        elif v > RETRACE_WARN_PX:
            warn.append("The median of the %s is %.2f px RMS from straight after correction (limit %.2f px)."
                        % (label, v, RETRACE_WARN_PX))
    cov = bar_coverage(bars, (h, w), normalized=norm, step=tinfo["step"])
    qc["coverage"] = cov
    warn.extend(coverage_warnings(cov, wavelength))
    if white is None:
        warn.append("No white frame: the illumination was flattened with a running mean along the slit only.")
    if summ["extrapolated_columns"] > 0.25:
        warn.append("%.0f %% of the columns lie outside the lamp lines (the smile there is held at the edge "
                    "value)." % (100 * summ["extrapolated_columns"]))
    return KeystoneAnalysis(model=model, map_x=mx, map_y=my, margin=margin, row_margin=row_margin,
                            target_lines=bars, qc=qc, summary=summ, warnings=warn, target_corr=target_corr,
                            lamp_corr=lamp_corr, target_norm=norm, interp=interp, polarity=polarity,
                            created=_dt.datetime.now().isoformat(timespec="milliseconds"), inputs=dict(inputs or {}))


def _solve_dy(bars: List[TargetLine], dx: GeometrySurface, sdy: GeometrySurface, x_ref: float, w: int,
              iterations: int = 8) -> dict:
    """Keystone alone, with the smile known over RAW rows (``dx_rows="source"``).

    A bar point measured at raw ``(x, y)`` belongs to the destination column ``X = x - dx(X, y)``
    (the row is the raw row, so this does not involve dy); its destination row is the bar's ``Ym``,
    fixed by ``dy(x_ref, Ym) = 0``. It starts as ``bar_y(x_ref + dx(x_ref, Ym))`` (the bar's cubic);
    for the soft-weighted fitters (all but the earlier polynomial) it is then re-estimated from
    all the bar's points with the surface, alternating with the fit (:func:`_bar_rows`).
    """
    tx = np.concatenate([t.x for t in bars]); ty = np.concatenate([t.y for t in bars])
    tm = np.concatenate([np.full(t.x.size, i) for i, t in enumerate(bars)])
    Xt = tx.copy()
    for _ in range(6):
        Xt = tx - dx.at(Xt, ty)
    Ym = np.array([float(b.y_at([x_ref], w)[0]) for b in bars])
    for _ in range(6):
        Ym = np.array([float(b.y_at([x_ref + float(dx.at(x_ref, y))], w)[0]) for b, y in zip(bars, Ym)])
    Yt = Ym[tm]
    w0 = _point_weights(bars)
    wt = w0.copy()
    if getattr(sdy, "robust", "huber") == "soft":       # surface and bar rows solved together
        wt, Ym = _robust_fit_rows(sdy, Xt, ty, tm, w0, Ym, max(iterations, 12))
        Yt = Ym[tm]
    else:                                               # the earlier polynomial: rows from each bar's cubic
        for it in range(iterations):
            sdy.fit(Xt, Yt, ty - Yt, wt)
            rt = ty - (Yt + sdy.raw(Xt, Yt))
            ht = _huber_weights(rt, wt > 0)
            ft, _ = _feature_weights(rt, ht > 0, tm, len(bars))
            new = ht * ft[tm] * w0
            if it >= 2 and np.array_equal(new > 0, wt > 0) and np.allclose(new, wt, rtol=1e-3, atol=1e-6):
                wt = new
                break
            wt = new
        sdy.fit(Xt, Yt, ty - Yt, wt)
    rt = ty - (Yt + sdy.raw(Xt, Yt))
    ft, st = _feature_weights(rt, wt > 0, tm, len(bars))
    return dict(Ym=Ym, Xt=Xt, Yt=Yt, ty=ty, tm=tm, rt=rt, wt=wt, w0=w0, bar_weight=ft, bar_sigma=st)


def _loo_dy(sol: dict, make, n: int):
    """Leave-one-bar-out RMS (px) of the keystone surface."""
    if n < 3:
        return None, []
    per = []
    v = sol["ty"] - sol["Yt"]
    for i in range(n):
        s = make()
        m = (sol["tm"] != i) & (sol["wt"] > 0)
        try:
            s.fit(sol["Xt"][m], sol["Yt"][m], v[m], sol["wt"][m])
        except np.linalg.LinAlgError:
            return None, []
        sel = (sol["tm"] == i) & (sol["wt"] > 0)
        if sel.any():                       # a bar the robust weights dropped entirely has nothing to predict
            per.append(_rms(s.raw(sol["Xt"][sel], sol["Yt"][sel]) - v[sel]))
    return (_rms(per) if per else None), per


def _lamp_check(lamp_corr: np.ndarray, valid: np.ndarray, sat_level=None) -> dict:
    """Straightness of the lamp lines on a corrected lamp frame (deviation from each line's median column)."""
    from .core import detect_lines
    a = np.asarray(lamp_corr, np.float32)
    sat = auto_sat_level(a) if sat_level is None else float(sat_level)
    noise = estimate_noise(a)
    per, allv = [], []
    for ln in detect_lines(a, sat):
        if ln.saturated:
            continue
        c = trace_line(a, ln.column, max(3, int(round(1.2 * ln.fwhm))), sat, noise)
        rows = np.flatnonzero(np.isfinite(c))
        cx = int(np.clip(round(ln.column), 0, a.shape[1] - 1))
        rows = rows[valid[rows, cx]]
        if rows.size < 20:
            continue
        v = c[rows] - np.median(c[rows])
        allv.extend(v.tolist())
        per.append(dict(X=float(np.median(c[rows])), rms_px=_rms(v), max_px=float(np.max(np.abs(v))), n=int(v.size)))
    allv = np.asarray(allv)
    return dict(lamp_rms_px=_rms(allv), lamp_max_px=float(np.max(np.abs(allv))) if allv.size else float("nan"),
                lamp_p95_px=float(np.percentile(np.abs(allv), 95)) if allv.size else float("nan"),
                lamp_per_line=per)


@logged_analysis("keystone (onto the stored smile)")
def analyze_keystone_from_smile(smile, y_ref: float, target, white=None, dark=None, polarity: str = "auto",
                                x_ref: Optional[float] = None, dy_xdeg: int = 3, dy_ydeg="auto", band: int = 5,
                                step: Optional[int] = None, snr_min: float = 5.0, interp: str = "cubic",
                                inputs: Optional[dict] = None, smile_info: Optional[dict] = None,
                                lamp=None, wavelength=None, dy_model: str = DEFAULT_DY_MODEL,
                                gap_recovery: bool = True) -> KeystoneAnalysis:
    """Keystone from a line-target frame onto a STORED smile (no lamp frame needed).

    For a spectrograph whose fore-lens is taken off to measure the smile (lamp at the slit) and put
    back for the line target: the smile belongs to the spectrograph and detector, so the smile
    model already stored for this camera setting is reused, and only the keystone is fitted. The
    smile is defined over the raw row (as the lamp lines were measured), so the combined map reads
    the smile at the row the pixel comes from: ``src_y = Y + dy(X, Y)``, ``src_x = X + S(src_y, X)``.
    The smile stays exactly as stored (``S(y_ref, X) = 0``: the wavelength axis is unchanged).

    Args:
        smile: The stored smile-only :class:`~modules.smile.surface.Surface2D` ``S(y, x)``
            (``SmileCalibration.surface()``), anchored at *y_ref*.
        y_ref: Its reference row.
        target: Line-target frame ``(H, W)`` taken with the lens on (same camera setting).
        white, dark, polarity, x_ref, dy_xdeg, dy_ydeg, band, step, snr_min: As :func:`analyze_keystone`.
        interp: Remap interpolation of the model.
        inputs: Free description of the input files.
        smile_info: Where the smile came from (file, created, ...), stored in the result.
        lamp: Optional lamp frame of the same setting to CHECK the stored smile (not fitted).
        wavelength: Optional ``callable(column) -> nm`` to name the spectral range in coverage warnings.
        dy_model, gap_recovery: As :func:`analyze_keystone`.

    Returns:
        :class:`KeystoneAnalysis` with ``model.dx_rows == "source"`` (``lamp_corr`` only with *lamp*).

    Raises:
        ValueError: Unusable target, or a lamp frame of another size.
    """
    t = np.asarray(target, np.float32)
    if t.ndim != 2:
        raise ValueError("The target frame must be 2-D (slit rows, spectral columns).")
    h, w = t.shape
    y0, y1 = (float(v) for v in smile.y_rng)
    x0, x1 = (float(v) for v in smile.x_rng)
    if y1 > h - 1 + 0.5 or x1 > w - 1 + 0.5:
        raise ValueError(f"The stored smile was measured on a larger frame (rows to {y1:.0f}, columns to {x1:.0f}) "
                         f"than this target {t.shape}: use the smile model of this camera setting / bin.")
    notes: List[str] = []
    norm = normalize_target(t, white, dark)
    bars, tinfo = trace_target_lines(t, polarity=polarity, band=band, step=step, snr_min=snr_min,
                                     normalized=norm, sat_level=None)
    polarity = tinfo["polarity"]
    notes.append(_polarity_note(tinfo))
    for y, why in tinfo["rejected"]:
        notes.append("target bar near row %.0f not used: %s" % (y, why))
    t_lo = float(np.median([b.x.min() for b in bars])); t_hi = float(np.median([b.x.max() for b in bars]))
    x_ref = float(np.clip((w - 1) / 2.0, t_lo, t_hi)) if x_ref is None else float(x_ref)
    bars, gap = _recover(bars, tinfo, norm, gap_recovery, notes)
    dx = GeometrySurface.from_smile(smile)
    dy_xdeg = max(1, int(dy_xdeg))
    sx, sy = max(1.0, (w - 1) / 2.0), max(1.0, (h - 1) / 2.0)
    mode = _check_dy_model(dy_model)
    fitter, selection = None, []
    if mode != "poly":
        tx_all = np.concatenate([b.x for b in bars])
        cands = keystone_candidates((h, w), x_ref, tinfo["period"],
                                    (float(np.percentile(tx_all, 0.5)), float(np.percentile(tx_all, 99.5))))
        if mode == "auto":
            # the bar points' destination coordinates depend only on the stored smile: one solve gives them
            sol0 = _solve_dy(bars, dx, cands["anchored_bspline"][0](), x_ref, w)
            fitter, selection = select_keystone_fitter(cands, sol0["Xt"], sol0["Yt"], sol0["ty"], sol0["tm"],
                                                       sol0["w0"], (h, w), x_ref)
            fitter = fitter or "anchored_bspline"
        else:
            fitter = mode
        factory = cands[fitter][0]
        make = lambda d=None: factory()
        cand, d_y = [], None
    else:
        make = lambda d: GeometrySurface(range(1, dy_xdeg + 1), range(0, d + 1), x_ref, sx, (h - 1) / 2.0, sy)
        if dy_ydeg == "auto":
            cand = [d for d in range(0, MAX_DY_YDEG + 1) if d <= max(0, len(bars) - 3)]
            d_y = min(1, cand[-1])
        else:
            cand = [max(0, min(int(dy_ydeg), len(bars) - 2))]
            d_y = cand[0]
    sol = _solve_dy(bars, dx, make(d_y), x_ref, w)
    choice = {}
    if len(cand) > 1:
        res = {d: _loo_dy(sol, lambda d=d: make(d), len(bars))[0] for d in cand}
        valid_res = {d: v for d, v in res.items() if v is not None and np.isfinite(v)}
        if valid_res:
            best = min(valid_res, key=valid_res.get)
            for d in sorted(valid_res):
                if valid_res[d] <= 1.1 * valid_res[best]:
                    best = d
                    break
            d_y = best
        choice = {str(k): v for k, v in res.items()}
    sdy = make(d_y)
    sol = _solve_dy(bars, dx, sdy, x_ref, w)
    sdy.box = [float(np.percentile(sol["Xt"], 0.5)), float(np.percentile(sol["Xt"], 99.5)),
               float(sol["Ym"].min()), float(sol["Ym"].max())]
    loo_y, per_y = _loo_dy(sol, lambda: make(d_y), len(bars))
    kt = sol["wt"] > 0
    model = GeometryModel(shape=(h, w), y_ref=float(y_ref), x_ref=x_ref, dx=dx, dy=sdy, lamp_X=[],
                          target_Y=[float(v) for v in sol["Ym"]], dx_rows="source")
    mx, my = model.maps()
    margin, row_margin = valid_rect(mx, my, interp)
    target_corr = _remap(norm, mx, my, interp)
    valid = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    rt = _retrace(model, np.zeros((h, w), np.float32), target_corr, valid, [], None, None, snr_min, polarity,
                  tinfo["band"], [b.halfwin or tinfo["halfwin"] for b in bars], tinfo["step"])
    med = lambda per: (float(np.nanmedian([p["rms_px"] for p in per])) if per else float("nan"))
    notes.append(_fitter_note(dict(dy_fitter=fitter or "poly", dy_selection=selection)))
    qc = dict(
        smile_source="stored", lamp_lines=0, target_bars=len(bars),
        degrees=dict(dx=[smile.deg_y, smile.deg_x], dy=(fitter or [d_y, dy_xdeg])),
        dy_model=dy_model, dy_fitter=fitter or "poly", dy_selection=selection, gap_recovery=gap, dy_ydeg_loo=choice,
        before_lamp_rms_px=None, before_target_rms_px=_rms(sol["ty"] - sol["Yt"]),
        fit_lamp_rms_px=None, fit_target_rms_px=_rms(sol["rt"][kt]),
        fit_lamp_max_px=None, fit_target_max_px=float(np.max(np.abs(sol["rt"][kt]))) if kt.any() else float("nan"),
        loo_lamp_rms_px=None, loo_target_rms_px=loo_y, loo_lamp_per_line_px=[], loo_target_per_bar_px=per_y,
        rejected_points=int((~kt).sum()), downweighted_points=_downweighted(sol["rt"], kt) if fitter else 0,
        fit_lamp_per_line_px=[], fit_target_per_bar_px=[float(v) for v in sol["bar_sigma"]],
        lamp_weight=[], bar_weight=[float(v) for v in sol["bar_weight"]],
        retrace_lamp_rms_px=None, retrace_lamp_max_px=None, retrace_lamp_p95_px=None, retrace_lamp_per_line=[],
        retrace_lamp_median_line_px=None,
        retrace_target_rms_px=rt["target_rms_px"], retrace_target_max_px=rt["target_max_px"],
        retrace_target_p95_px=rt["target_p95_px"], retrace_target_per_bar=rt["target_per_bar"],
        retrace_target_median_bar_px=med(rt["target_per_bar"]),
        bar_period_px=tinfo["period"], bar_halfwin_px=tinfo["halfwin"], seed_column=tinfo["seed_column"],
        normalised_with_white=white is not None, notes=notes, merged_bar_pairs=tinfo.get("merged", []),
        smile_info=dict(smile_info or {}))
    lamp_corr = None
    if lamp is not None:
        la = np.asarray(lamp, np.float32)
        if la.shape != (h, w):
            raise ValueError(f"The check lamp frame {la.shape} differs from the target {(h, w)}.")
        lamp_corr = _remap(la, mx, my, interp)
        lc = _lamp_check(lamp_corr, valid)
        qc.update(retrace_lamp_rms_px=lc["lamp_rms_px"], retrace_lamp_max_px=lc["lamp_max_px"],
                  retrace_lamp_p95_px=lc["lamp_p95_px"], retrace_lamp_per_line=lc["lamp_per_line"],
                  retrace_lamp_median_line_px=med(lc["lamp_per_line"]), lamp_lines=len(lc["lamp_per_line"]))
    weak = [p for p in rt["target_per_bar"] if np.isfinite(p["rms_px"]) and p["rms_px"] > 2 * RETRACE_WARN_PX]
    if weak:
        notes.append("%d target bars scatter over %.2f px RMS after correction (asymmetric profile or weak signal; "
                     "the fit weights them down), rows %s" % (len(weak), 2 * RETRACE_WARN_PX,
                                                            ", ".join("%.0f" % p["Y"] for p in weak[:12])))
    summ = _summary(model, mx, my)
    warn: List[str] = []
    if summ["row_jacobian_min"] < ROW_JACOBIAN_WARN:
        warn.append("The row map compresses the slit to %.2f of its spacing somewhere (folds below 0): the "
                    "keystone surface follows inconsistent bar centres there; check the target / focus."
                    % summ["row_jacobian_min"])
    created = (smile_info or {}).get("created")
    warn.append("Smile taken from the stored model%s: it was not re-measured here. Check it with a lamp frame "
                "now and then (lens off is fine)." % (" of %s" % created if created else ""))
    v = qc["retrace_target_median_bar_px"]
    if not np.isfinite(v):
        warn.append("The target bars could not be re-measured on the corrected frame.")
    elif v > RETRACE_WARN_PX:
        warn.append("The median of the target bars is %.2f px RMS from straight after correction (limit %.2f px)."
                    % (v, RETRACE_WARN_PX))
    lv = qc.get("retrace_lamp_median_line_px")
    if lv is not None and np.isfinite(lv) and lv > RETRACE_WARN_PX:
        warn.append("The check lamp frame is %.2f px RMS from straight with the stored smile: measure the smile "
                    "again for this bin." % lv)
    cov = bar_coverage(bars, (h, w), normalized=norm, step=tinfo["step"])
    qc["coverage"] = cov
    warn.extend(coverage_warnings(cov, wavelength))
    if white is None:
        warn.append("No white frame: the illumination was flattened with a running mean along the slit only.")
    if summ["extrapolated_columns"] > 0.25:
        warn.append("%.0f %% of the columns lie outside the lamp lines of the stored smile (held at the edge "
                    "value)." % (100 * summ["extrapolated_columns"]))
    return KeystoneAnalysis(model=model, map_x=mx, map_y=my, margin=margin, row_margin=row_margin,
                            target_lines=bars, qc=qc, summary=summ, warnings=warn, target_corr=target_corr,
                            lamp_corr=lamp_corr, target_norm=norm, interp=interp, polarity=polarity,
                            created=_dt.datetime.now().isoformat(timespec="milliseconds"), inputs=dict(inputs or {}))


def _polarity_note(tinfo: dict) -> str:
    """One line saying which bar polarity was used and why."""
    pol, auto = tinfo.get("polarity"), tinfo.get("polarity_auto")
    if not auto:
        return "Bar polarity: %s (set)" % pol
    df = auto.get("dark_fraction")
    share = "" if df is None else ", dark area %.0f %% of the lit target" % (100 * df)
    how = {"area": "from the area of the narrow features", "trace": "both traced, the better kept",
           "default": "default"}.get(auto.get("method"), auto.get("method"))
    return "Bar polarity: %s (auto: %s%s)" % (pol, how, share)


def keystone_report(k) -> str:
    """Human-readable report of a :class:`KeystoneAnalysis` (or its :meth:`~KeystoneAnalysis.to_json`)."""
    d = k.to_json() if hasattr(k, "to_json") else k      # duck-typed: also under ``python -m``
    q, s = d["qc"], d["summary"]
    f = lambda v: "-" if v is None else "%.3f px" % v
    lines = [
        "Smile + keystone geometry (%s)" % d.get("created", ""),
        *(["Smile: stored model %s (not re-fitted; keystone fitted onto it)"
           % ((q.get("smile_info") or {}).get("created") or "")] if q.get("smile_source") == "stored" else []),
        "Lamp lines %s, target bars %d, degrees dx %s dy %s" % (
            ("%d (check frame)" % q["lamp_lines"] if q["lamp_lines"] else "- (stored smile)")
            if q.get("smile_source") == "stored" else "%d" % q["lamp_lines"], q["target_bars"],
                                                               q["degrees"]["dx"], q["degrees"]["dy"]),
        "Before: lamp %s, target %s" % (f(q["before_lamp_rms_px"]), f(q["before_target_rms_px"])),
        "Fit RMS: lamp %s, target %s" % (f(q["fit_lamp_rms_px"]), f(q["fit_target_rms_px"])),
        "Leave-one-line-out: lamp %s, target %s" % (f(q["loo_lamp_rms_px"]), f(q["loo_target_rms_px"])),
        "Re-traced after remap (median line / bar): lamp %s, target %s" % (
            f(q.get("retrace_lamp_median_line_px")), f(q.get("retrace_target_median_bar_px"))),
        "Re-traced after remap (all points): lamp %s (p95 %s, max %s), target %s (p95 %s, max %s)" % (
            f(q["retrace_lamp_rms_px"]), f(q.get("retrace_lamp_p95_px")), f(q["retrace_lamp_max_px"]),
            f(q["retrace_target_rms_px"]), f(q.get("retrace_target_p95_px")), f(q["retrace_target_max_px"])),
        "Smile max %.2f px, keystone max %.2f px, tilt %.3f deg - %s" % (
            s["smile_max_px"], s["keystone_max_px"], s["tilt_deg"], s.get("verdict", "")),
        "Border: columns %s, rows %s" % (tuple(d["margin"]), tuple(d["row_margin"])),
    ]
    lines += ["Warning: " + w for w in d.get("warnings", [])]
    lines += q.get("notes", []) or []
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------- CLI
#: Frames to be averaged must agree to this many rows (slit shift) ...
SAME_SCENE_MAX_SHIFT = 0.5
#: ... and this correlation of their slit profiles.
SAME_SCENE_MIN_CORR = 0.95


def frames_same_scene(frames: Sequence[np.ndarray], band: int = 41) -> Tuple[bool, List[dict]]:
    """Do several target (or white) frames show the same scene, so they may be averaged?

    Averaging frames of a target that moved, or of another part / distance of it, mixes two bar
    patterns into doubled, asymmetric bars (a real GigE pair: period 21 px vs 14 px; each frame alone
    re-traced to 0.06-0.07 px, their mean to 0.20 px). Each frame's slit profile (median of ``band``
    columns in the middle of the spectrum) is compared with the first: the shift along the slit
    (sub-pixel, from the cross-correlation peak) and the correlation.

    Returns:
        ``(ok, details)``: per frame after the first ``{"index", "shift_rows", "correlation"}``.
    """
    frames = [np.asarray(f, np.float32) for f in frames]
    if len(frames) < 2:
        return True, []
    h, w = frames[0].shape
    c0, c1 = max(0, w // 2 - band // 2), min(w, w // 2 + band // 2 + 1)

    def prof(f):
        p = np.median(f[:, c0:c1], axis=1).astype(np.float64)
        p = p - uniform_filter1d(p, size=max(31, h // 8), mode="nearest")       # slow light trend out
        return (p - p.mean()) / (p.std() + 1e-12)
    ref = prof(frames[0])
    details, ok = [], True
    for i, f in enumerate(frames[1:], start=1):
        if f.shape != (h, w):
            return False, details + [dict(index=i, shift_rows=None, correlation=None, reason="other size")]
        p = prof(f)
        cc = np.correlate(p, ref, "full") / len(p)
        k = int(np.argmax(cc))
        sub = 0.0
        if 0 < k < len(cc) - 1:
            a, b, c = cc[k - 1], cc[k], cc[k + 1]
            den = a - 2 * b + c
            sub = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
        shift = (k - (len(p) - 1)) + sub
        corr = float(np.corrcoef(p, ref)[0, 1])
        same = abs(shift) <= SAME_SCENE_MAX_SHIFT and corr >= SAME_SCENE_MIN_CORR
        ok = ok and bool(same)
        details.append(dict(index=i, shift_rows=float(shift), correlation=corr))
    return bool(ok), details


def _read(paths: Sequence[str]) -> np.ndarray:
    from .core import load_frame
    frames = [load_frame(p) for p in paths]
    if any(f.shape != frames[0].shape for f in frames):
        raise SystemExit("All frames of one kind must have the same size.")
    if len(frames) > 1:
        same, det = frames_same_scene(frames)
        if not same:
            bad = "; ".join("%s: shift %s rows, correlation %s" % (
                paths[d["index"]], "?" if d.get("shift_rows") is None else "%.2f" % d["shift_rows"],
                "?" if d.get("correlation") is None else "%.3f" % d["correlation"]) for d in det)
            raise SystemExit("These frames do not show the same scene and must not be averaged (the target moved, "
                             "or another part / distance of it was imaged): %s. Use one frame, or several frames "
                             "of a target that does not move." % bad)
    return np.mean(np.stack(frames), axis=0).astype(np.float32) if len(frames) > 1 else frames[0]


def main(argv=None) -> int:
    """``python -m modules.smile.keystone (--lamp L.tif | --smile-model M.npz) --target T1.tif T2.tif [--white W.tif] [--out model.npz]``

    With ``--lamp`` the smile and the keystone are fitted jointly; with ``--smile-model`` the
    keystone is fitted onto that stored smile (fore-lens off for the smile, on for the target).
    """
    import argparse
    from .corrector import SmileCorrector
    ap = argparse.ArgumentParser(description="Smile + keystone model from a lamp frame (or a stored smile model) "
                                             "and a line-target frame.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--lamp", nargs="+", help="lamp frame(s): fit smile and keystone jointly")
    src.add_argument("--smile-model", help="stored smile model (.npz) of this camera setting: fit the keystone onto it")
    ap.add_argument("--check-lamp", nargs="+", help="with --smile-model: lamp frame(s) to check the stored smile")
    ap.add_argument("--target", nargs="+", required=True)
    ap.add_argument("--white", nargs="+")
    ap.add_argument("--dark", nargs="+")
    ap.add_argument("--polarity", choices=("auto", "dark", "bright"), default="auto")
    ap.add_argument("--deg-y", type=int, default=2); ap.add_argument("--deg-x", type=int, default=1)
    ap.add_argument("--dy-xdeg", type=int, default=3)
    ap.add_argument("--interp", default="cubic", choices=sorted(_INTERP))
    ap.add_argument("--out", help="model .npz to write (no camera setting is stored)")
    ap.add_argument("--no-log", action="store_true", help="do not write a debug log to modules/smile/logs")
    a = ap.parse_args(argv)
    from . import debug_log
    if not a.no_log:
        debug_log.enable()
    target = _read(a.target)
    white = _read(a.white) if a.white else None
    dark = _read(a.dark) if a.dark else None
    inputs = dict(lamp=a.lamp, smile_model=a.smile_model, target=a.target, white=a.white, dark=a.dark)
    if a.smile_model:
        c = SmileCorrector.load(a.smile_model)
        if a.interp != c.interp:
            c.set_params(interp=a.interp)
        check = _read(a.check_lamp) if a.check_lamp else None
        if check is not None and dark is not None:
            check = check - dark
        k = c.fit_keystone(target, white=white, dark=dark, polarity=a.polarity, dy_xdeg=a.dy_xdeg,
                           inputs=inputs, lamp=check)
    else:
        lamp = _read(a.lamp)
        if dark is not None:
            lamp = lamp - dark
        c = SmileCorrector(deg_y=a.deg_y, deg_x=a.deg_x, interp=a.interp).fit(lamp)
        k = c.fit_keystone(target, white=white, dark=dark, polarity=a.polarity, dy_xdeg=a.dy_xdeg, inputs=inputs)
    print(keystone_report(k))
    if a.out:
        c.save(a.out)
        print("Written %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
