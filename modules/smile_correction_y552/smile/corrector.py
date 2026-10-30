# smile/corrector.py
"""Sklearn-style fit / transform API for smile correction."""
from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .core import _INTERP
from .surface import Surface2D
from .analysis import SmileAnalysis, analyze_smile
from .calibration import (
    SmileCalibration,
    build_maps,
    calibration_from_analysis,
    crop_margin,
    load_calibration,
    save_calibration,
)

__all__ = ["SmileCorrector"]


class SmileCorrector:
    """Sklearn-style smile correction: ``fit`` once, ``transform`` many.

    Usage::

        corrector = SmileCorrector(deg_y=2, deg_x=1, interp="cubic")
        corrector.fit(hgar_frame)
        corrector.save("smile_model.npz")

        corrected = corrector.transform(raw_frame, crop=True)

        corrector = SmileCorrector.load("smile_model.npz")
        corrected = corrector.transform(raw_frame)
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
        y_ref: Optional[float] = None,
    ):
        self.deg_y = deg_y
        self.deg_x = deg_x
        self.interp = interp
        self.sat_level = sat_level
        self.snr_min = snr_min
        self.n_ref_iter = n_ref_iter
        self.y_ref = y_ref

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
            y_ref=self.y_ref,
            verbose=verbose,
        )
        self.y_ref = self._analysis.y_ref
        self._surface = self._analysis.surface
        self._map_x, _, _ = build_maps(
            self._analysis.frame_orig.shape, self._surface
        )
        self._margin = self._analysis.margin
        self._image_shape = tuple(self._analysis.frame_orig.shape)
        return self

    def transform(
        self, frame: np.ndarray, crop: bool = True
    ) -> np.ndarray:
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
            np.repeat(
                np.arange(n_rows, dtype=np.float32)[:, None], n_cols, axis=1
            )
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
        if self._analysis is not None:
            cal = calibration_from_analysis(self._analysis)
        else:
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
                created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                notes="re-saved from loaded model",
                y_ref=self.y_ref,
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
            y_ref=cal.y_ref,
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
            "y_ref": self.y_ref,
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
        if "interp" in params and self.is_fitted:
            # The fitted map is interpolation-independent; only the remap
            # kernel changes. Recompute the valid crop for the new kernel.
            from .calibration import margin_from_map
            self._margin = margin_from_map(self._map_x, self.interp)
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
        """Full analysis result (after :meth:`fit`, not :meth:`load`)."""
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
            cv_str = (
                f"{a.cv_rms:.5f}" if np.isfinite(a.cv_rms) else "n/a"
            )
            return (
                f"SmileCorrector (fitted)\n"
                f"  Sensor       : "
                f"{self._image_shape[1]}x{self._image_shape[0]} px\n"
                f"  Lines used   : {len(a.used_lines)}/{len(a.all_lines)}\n"
                f"  Fit          : deg={self.deg_y}x{self.deg_x}  "
                f"interp={self.interp}\n"
                f"  Straightness : "
                f"{a.rms_before:.4f} -> {a.rms_after:.4f} px\n"
                f"  CV (LOO)     : {cv_str} px\n"
                f"  Border loss  : L={left} R={right} ({left + right} px)"
            )
        return (
            f"SmileCorrector (loaded)\n"
            f"  Sensor       : "
            f"{self._image_shape[1]}x{self._image_shape[0]} px\n"
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
            np.repeat(
                np.arange(h, dtype=np.float32)[:, None], w, axis=1
            )
        )
