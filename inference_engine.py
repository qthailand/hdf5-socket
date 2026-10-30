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
