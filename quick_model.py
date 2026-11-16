# -*- coding: utf-8 -*-
"""Quick model: a linear SGD classifier trained in the running program, for system tests.

The GUI (``widgets/quick_model_dialog.py``) freezes the newest model-input lines, the operator
marks areas per class, and :func:`train_artifact` turns the marked spectra into the same artifact
``tools/train_sgd_coco.py`` writes (scaler folded into ``W_fused`` / ``b_fused``), kept in memory
only: ``PushbroomLineClassifier.from_artifact`` makes the classifier. Nothing here is saved; a
restart of the program starts without labels and without quick models.
"""
from __future__ import annotations

import numpy as np

from config import TAB10_RGB_COLORS

#: fewest spectra a class needs before a model is trained
MIN_SAMPLES = 10
#: most spectra kept per class (a random subset beyond that): bounds memory and training time
MAX_PER_CLASS = 20000
#: most spectra per class used for one training
TRAIN_PER_CLASS = 5000


class LabelSet:
    """Classes and their labelled spectra of one program session (memory only)."""

    def __init__(self) -> None:
        self.names: dict[int, str] = {}
        self.colors: dict[int, list[int]] = {}
        self._spectra: dict[int, np.ndarray] = {}
        self._rng = np.random.default_rng(0)

    @property
    def class_ids(self) -> list[int]:
        return sorted(self.names)

    def add_class(self, name: str | None = None) -> int:
        """A new class with the next free id (1..254; 0 stays the belt map's empty colour)."""
        cid = next(i for i in range(1, 255) if i not in self.names)
        self.names[cid] = name or f"class {cid}"
        self.colors[cid] = list(TAB10_RGB_COLORS[(cid - 1) % len(TAB10_RGB_COLORS)])
        return cid

    def remove_class(self, cid: int) -> None:
        for store in (self.names, self.colors, self._spectra):
            store.pop(cid, None)

    def add(self, cid: int, spectra: np.ndarray) -> None:
        """Add spectra ``(n, bands)`` to class ``cid``; other band counts replace what was kept."""
        spectra = np.asarray(spectra, np.float32).reshape(-1, np.shape(spectra)[-1])
        if cid not in self.names or not len(spectra):
            return
        kept = self._spectra.get(cid)
        if kept is not None and kept.shape[1] == spectra.shape[1]:
            spectra = np.concatenate([kept, spectra])
        if len(spectra) > MAX_PER_CLASS:
            spectra = spectra[self._rng.choice(len(spectra), MAX_PER_CLASS, replace=False)]
        self._spectra[cid] = spectra

    def count(self, cid: int) -> int:
        kept = self._spectra.get(cid)
        return 0 if kept is None else len(kept)

    def spectra(self, n_bands: int | None = None) -> dict[int, np.ndarray]:
        """Class id -> spectra; with ``n_bands`` only those of that band count (the camera bin changed)."""
        return {cid: s for cid, s in self._spectra.items() if n_bands is None or s.shape[1] == n_bands}


def spectral_angle(cube: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Angle (rad) between every spectrum of ``cube`` (..., bands) and ``seed``; pi where undefined."""
    cube = np.asarray(cube, np.float32)
    seed = np.asarray(seed, np.float32)
    norm = np.sqrt(np.einsum('...b,...b->...', cube, cube)) * float(np.linalg.norm(seed))
    with np.errstate(divide='ignore', invalid='ignore'):
        cosine = (cube @ seed) / norm
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    angle[~np.isfinite(angle)] = np.pi
    return angle


def wand_mask(cube: np.ndarray, line: int, row: int, tolerance: float) -> np.ndarray:
    """Pixels like the clicked one and touching it: the simple automatic selection.

    Args:
        cube: ``(lines, slit, bands)`` spectra.
        line, row: The clicked pixel.
        tolerance: Largest spectral angle (rad) to the clicked pixel's spectrum.

    Returns:
        Boolean mask ``(lines, slit)``: the connected area around the click whose spectra lie
        within ``tolerance`` (the angle ignores brightness, so shading does not cut an object).
    """
    from scipy import ndimage
    similar = spectral_angle(cube, cube[line, row]) <= tolerance
    similar[line, row] = True
    labels, _ = ndimage.label(similar)
    return labels == labels[line, row]


def train_artifact(spectra: dict[int, np.ndarray], names: dict[int, str], wavelengths, input_units: str,
                   colors: dict[int, list[int]] | None = None, seed: int = 0) -> tuple[dict, dict]:
    """Train the linear SGD model and fold its scaler into the weights.

    Args:
        spectra: Class id -> labelled spectra ``(n, bands)`` as the model will receive them.
        names: Class id -> name.
        wavelengths: nm of every band, or ``None`` when the input has none.
        input_units: ``'reflectance'`` or ``'counts'``: what the spectra are.
        colors: Class id -> ``[r, g, b]`` for the belt map.

    Returns:
        ``(artifact, report)``: the artifact of ``PushbroomLineClassifier`` (all bands selected)
        and ``{'accuracy': held-out 20 % or None, 'samples': {class id: spectra used}}``.

    Raises:
        ValueError: fewer than two classes with ``MIN_SAMPLES`` finite spectra each.
    """
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    x_parts, y_parts, used = [], [], {}
    for cid in sorted(spectra):
        x = np.asarray(spectra[cid], np.float32)
        x = x[np.isfinite(x).all(axis=1)]
        if len(x) > TRAIN_PER_CLASS:
            x = x[rng.choice(len(x), TRAIN_PER_CLASS, replace=False)]
        if len(x) >= MIN_SAMPLES:
            x_parts.append(x)
            y_parts.append(np.full(len(x), cid))
            used[cid] = len(x)
    if len(x_parts) < 2:
        raise ValueError(f"Label at least two classes with {MIN_SAMPLES} or more pixels each "
                         f"(now: {len(x_parts)} such class{'es' if len(x_parts) != 1 else ''}).")
    x, y = np.concatenate(x_parts), np.concatenate(y_parts)

    def fit(xs, ys):
        scaler = StandardScaler().fit(xs)
        model = SGDClassifier(loss='log_loss', class_weight='balanced', max_iter=1000, tol=1e-3,
                              random_state=seed).fit(scaler.transform(xs), ys)
        return scaler, model

    # held-out accuracy: pixels of the same marked areas, so it flatters; a sanity check only
    order = rng.permutation(len(y))
    test = np.zeros(len(y), bool)
    for cid in used:
        members = order[y[order] == cid]
        test[members[:len(members) // 5]] = True
    scaler, model = fit(x[~test], y[~test])
    accuracy = float((model.predict(scaler.transform(x[test])) == y[test]).mean()) if test.any() else None

    scaler, model = fit(x, y)                              # the model that is used: every labelled pixel
    coef, intercept = model.coef_.astype(np.float64), model.intercept_.astype(np.float64)
    if coef.shape[0] == 1:                                 # two classes: sklearn keeps one score
        coef, intercept = np.vstack([-coef, coef]), np.concatenate([-intercept, intercept])
    weights = coef / scaler.scale_
    n_bands = x.shape[1]
    artifact = {
        'W_fused': weights.T.astype(np.float32),
        'b_fused': (intercept - weights @ scaler.mean_).astype(np.float32),
        'selected_bands': np.arange(n_bands), 'classes': model.classes_.astype(np.int64),
        'category_names': {int(c): str(names.get(int(c), int(c))) for c in model.classes_},
        'class_colors': {int(c): list(colors[int(c)]) for c in model.classes_ if colors and int(c) in colors},
        'input_units': input_units, 'n_bands': n_bands,
        'band_wavelengths': None if wavelengths is None else np.asarray(wavelengths, np.float32),
    }
    return artifact, {'accuracy': accuracy, 'samples': used}
