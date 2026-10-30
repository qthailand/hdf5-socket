# -*- coding: utf-8 -*-
"""Per-line linear classifier (``argmax(x[bands] @ W + b)``) bound to the input's wavelengths.

The model artifact (``joblib``) holds ``W_fused`` ``(n_selected, n_classes)``, ``b_fused``,
``selected_bands`` (indices into the training cube's bands), ``classes`` and
``category_names``; ``band_wavelengths`` (nm of every training band) is optional.

Binning is not done here: the frame workers bin raw frames exactly like the HSI acquisition
(see ``sim_config.build_plan``) and pass ``(slit, n_bins)`` lines. :meth:`bind` maps the
model's bands onto the input's wavelengths, so a different calibration or a cut wavelength
range cannot silently feed the wrong bands to the model.
"""
from __future__ import annotations

import logging

import joblib
import numpy as np

logger = logging.getLogger(__name__)


class PushbroomLineClassifier:
    """Classify each slit pixel of a binned frame; read-only, so workers can share it."""

    def __init__(self, model_artifact_path: str) -> None:
        artifact = joblib.load(model_artifact_path)
        self.path = str(model_artifact_path)
        self.W_fused = np.ascontiguousarray(artifact['W_fused'], dtype=np.float32)
        self.b_fused = np.ascontiguousarray(artifact['b_fused'], dtype=np.float32)
        self.selected_bands = np.asarray(artifact['selected_bands'], dtype=np.intp)
        self.classes_ = np.asarray(artifact['classes'])
        self.category_names: dict = dict(artifact.get('category_names', {}))
        # 'counts' (dark-corrected DN, as dataset.bip) or 'reflectance' (after the reflectance step)
        self.input_units_declared = 'input_units' in artifact
        self.input_units = str(artifact.get('input_units', 'counts'))
        wl = artifact.get('band_wavelengths')
        self.band_wavelengths = None if wl is None else np.asarray(wl, dtype=np.float32)
        self.n_bands = artifact.get('n_bands', None if wl is None else len(wl))
        if self.W_fused.shape != (self.selected_bands.size, self.classes_.size):
            raise ValueError(f"W_fused {self.W_fused.shape} does not match {self.selected_bands.size} bands x "
                             f"{self.classes_.size} classes")
        if self.classes_.min() < 0 or self.classes_.max() > 254:
            raise ValueError("class ids must be 0..254 (255 marks lost lines)")
        self._class_ids = self.classes_.astype(np.uint8)

    @property
    def selected_wavelengths(self) -> np.ndarray | None:
        return None if self.band_wavelengths is None else self.band_wavelengths[self.selected_bands]

    def bind(self, wavelengths: np.ndarray | None, n_columns: int,
             tolerance_nm: float = 0.5) -> tuple[np.ndarray, str]:
        """Choose the input columns that feed the model (pass them to :meth:`predict`).

        Args:
            wavelengths: nm of every input column, or ``None`` when unknown.
            n_columns: Number of input columns.
            tolerance_nm: Largest accepted distance between a model band and its input column.

        Returns:
            ``(columns, description)``: input column of every selected model band, and a
            one-line description of how they were matched.

        Raises:
            ValueError: a model band has no input column within ``tolerance_nm``, or (without
                wavelengths) the input has another band count than the model was trained on.
        """
        wanted = self.selected_wavelengths
        if wanted is not None and wavelengths is not None:
            wavelengths = np.asarray(wavelengths, dtype=np.float64)
            columns = np.abs(wavelengths[None, :] - wanted[:, None]).argmin(axis=1)
            error = np.abs(wavelengths[columns] - wanted)
            if error.max() > tolerance_nm:
                worst = int(error.argmax())
                raise ValueError(
                    f"Model band {wanted[worst]:.2f} nm has no input band within {tolerance_nm} nm "
                    f"(nearest {wavelengths[columns[worst]]:.2f} nm). Input covers "
                    f"{wavelengths[0]:.2f}-{wavelengths[-1]:.2f} nm in {wavelengths.size} bands; check the "
                    f"wavelength calibration (m, c, camera bin, resolution) against the training data.")
            binding = (f"by wavelength: {wanted.size} bands {wanted.min():.1f}-{wanted.max():.1f} nm, "
                            f"max offset {error.max():.3f} nm")
        else:
            if self.n_bands is not None and n_columns != self.n_bands:
                raise ValueError(f"Input has {n_columns} bands, the model was trained on {self.n_bands}")
            if self.selected_bands.max() >= n_columns:
                raise ValueError(f"Model uses band index {self.selected_bands.max()}, input has {n_columns} bands")
            columns = self.selected_bands
            why = 'model has no band_wavelengths' if wanted is None else 'input has no wavelengths'
            binding = f"by index ({why}; wavelength alignment NOT verified)"
            logger.warning("Classifier bound %s", binding)
        return np.ascontiguousarray(columns, dtype=np.intp), binding

    def predict(self, binned: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Class id (uint8) of every slit pixel of one ``(slit, n_bins)`` frame."""
        x = binned[:, columns].astype(np.float32)
        scores = x @ self.W_fused
        scores += self.b_fused
        return self._class_ids[scores.argmax(axis=1)]


class SpeedTestClassifier(PushbroomLineClassifier):
    """Stand-in model with the same per-line work as a real one, for data without a model yet.

    Uses **every input band** by default (a full-spectrum linear model: ``bands x classes``
    weights), or ``n_selected`` bands spread evenly. Fixed random weights, ``n_classes`` classes:
    ``predict`` costs what a trained linear model of that size costs, the classes mean nothing.
    Used when ``model.path`` is ``"none"`` (e.g. a new camera before training).
    """

    def __init__(self, n_selected: int | None = None, n_classes: int = 3, seed: int = 0) -> None:
        self._n_selected, self._seed = n_selected, seed
        self.classes_ = np.arange(1, n_classes + 1)
        self.category_names = {int(c): f"Class {int(c)} (speed test)" for c in self.classes_}
        self.band_wavelengths = None
        self.n_bands = None
        self.input_units = 'any'
        self.input_units_declared = True
        self._class_ids = self.classes_.astype(np.uint8)
        self.W_fused = None
        self.b_fused = np.zeros(n_classes, np.float32)
        self.selected_bands = np.arange(0, dtype=np.intp)
        self.path = (f"speed-only model ({'all' if n_selected is None else n_selected} bands x {n_classes} classes, "
                     "no trained weights)")

    def bind(self, wavelengths, n_columns: int, tolerance_nm: float = 0.5):
        n = n_columns if self._n_selected is None else self._n_selected
        if n_columns < n:
            raise ValueError(f"speed-only model needs at least {n} bands, input has {n_columns}")
        if self.W_fused is None:                  # weights sized on the first stream's band count
            rng = np.random.default_rng(self._seed)
            self.W_fused = rng.standard_normal((n, self.classes_.size)).astype(np.float32)
            self.selected_bands = np.arange(n, dtype=np.intp)
        elif self.W_fused.shape[0] != n:
            raise ValueError(f"speed-only model was sized for {self.W_fused.shape[0]} bands, this input gives {n}")
        columns = np.linspace(0, n_columns - 1, n).round().astype(np.intp)
        which = 'all' if n == n_columns else f"{n} of"
        return columns, f"speed-only model: {which} {n_columns} bands x {self.classes_.size} classes, classes not meaningful"
