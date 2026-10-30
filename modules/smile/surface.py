# smile/surface.py
"""Robust 2-D polynomial surface fitting with IRLS / Huber weights."""
from __future__ import annotations

import numpy as np
from typing import List, Tuple

from .core import robust_sigma

__all__ = ["Surface2D", "fit_surface_2d", "cross_validate_lines"]


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
        """Args:
            coef: Flat coefficients (row-major over ``(i, j)``, power of ``y`` then of ``x``);
                copied into a float array.
            deg_y: Polynomial degree in the row direction.
            deg_x: Polynomial degree in the column direction.
            y_rng: ``(min, max)`` row range (px) used to normalise ``y`` to ``[-1, 1]``.
            x_rng: ``(min, max)`` column range (px) used to normalise ``x`` to ``[-1, 1]``.
            clamp: Clip normalised coordinates to ``[-1, 1]`` when evaluating.
        """
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

    def __repr__(self) -> str:
        """Short description with degrees and clamp flag."""
        return (
            f"Surface2D(deg_y={self.deg_y}, deg_x={self.deg_x}, "
            f"clamp={self.clamp})"
        )


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

    # Initial least-squares fit ensures *res* is defined even when n_iter == 0.
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
        cov = (np.sum(w * res**2) / dof) * np.linalg.pinv(
            (A * w[:, None]).T @ A
        )
        stderr = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        stderr = np.full(A.shape[1], np.nan)

    # NOTE: stderr underestimates true uncertainty because points within
    # one line are correlated.  Use cv_rms (leave-one-line-out) as the
    # primary reliability metric.
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