# -*- coding: utf-8 -*-
"""Per-line linear classifier (``argmax(x[bands] @ W + b)``) bound to the input's wavelengths.

:meth:`PushbroomLineClassifier.predict_confidence` also returns the winning class's
probability per pixel (sklearn ``predict_proba`` of the ``log_loss`` model); only for offline
checks (``tools/check_calibration.py``): the pipeline calls :meth:`~PushbroomLineClassifier.predict`.

The model artifact (``joblib``) holds ``W_fused`` ``(n_selected, n_classes)``, ``b_fused``,
``selected_bands`` (indices into the training cube's bands), ``classes`` and
``category_names``; ``band_wavelengths`` (nm of every training band) is optional.

A trained-model package of HSI MLOps (``.hsipkg``, or a zip around one) is a second kind of
artifact: :class:`HsiPackageClassifier` runs its own chain (SNV or a Savitzky-Golay first
derivative, band selection, StandardScaler, Gaussian maximum likelihood) from the arrays
``hsi_package.load_package`` reads.
:func:`load_classifier` picks the class from the file.

Binning is not done here: the frame workers bin raw frames exactly like the HSI acquisition
(see ``sim_config.build_plan``) and pass ``(slit, n_bins)`` lines. :meth:`bind` maps the
model's bands onto the input's wavelengths, so a different calibration or a cut wavelength
range cannot silently feed the wrong bands to the model. When the input's bands lie beside the
model's (the camera's wavelength calibration was redone after the model was trained), the input
is resampled onto the model's bands (:class:`Resampled`): the model stays usable, and the
binding says so.
"""
from __future__ import annotations

import logging
import math

import joblib
import numpy as np
from numba import njit, prange

logger = logging.getLogger(__name__)

_MLC_ROWS = 16        # slit pixels per parallel task of the MLC kernel (neighbours share the first guess)


@njit(parallel=True, fastmath={'reassoc', 'contract'}, cache=True, nogil=True)
def _mlc_classify(x, columns, tap_count, tap_index, taps, snv, inv_scale, centers, whiten, constants, class_ids,
                  invalid, out):
    """Gaussian maximum-likelihood class of every row of ``x`` (see :class:`HsiPackageClassifier`).

    Per row: SNV over ``columns`` (population SD) into ``z``, features
    ``f[j] = sum_t taps[j, t] * z[tap_index[j, t]] * inv_scale[j]`` over ``tap_count[j]`` taps (one
    tap of 1 = band selection; the Savitzky-Golay window = first derivative of the selected band), then per
    class ``score = constants[k] - 0.5 |whiten[k] @ (features - centers[k])|^2``. ``whiten[k]`` is
    lower triangular, so the squared distance grows row by row: a class is dropped as soon as
    its score is already below the best complete one. The result is the arg max of the full
    scores (lowest class index on an exact tie), only cheaper: on a belt most classes lose early.
    The class of the previous pixel is tried first. ``reassoc``/``contract`` only (no ``nnan``):
    NaN and infinity keep their meaning.
    """
    n = x.shape[0]
    n_bands = columns.shape[0]
    n_features = tap_count.shape[0]
    n_classes = constants.shape[0]
    for task in prange((n + _MLC_ROWS - 1) // _MLC_ROWS):
        features = np.empty(n_features, np.float32)
        delta = np.empty(n_features, np.float32)
        guess = 0
        for r in range(task * _MLC_ROWS, min(n, (task + 1) * _MLC_ROWS)):
            valid = True
            mean = 0.0
            deviation = 1.0
            if snv:
                total = 0.0
                for b in range(n_bands):
                    total += x[r, columns[b]]
                mean = total / n_bands
                squares = 0.0
                for b in range(n_bands):
                    v = x[r, columns[b]] - mean
                    squares += v * v
                deviation = math.sqrt(squares / n_bands)
                valid = deviation > 0.0 and deviation != math.inf       # False for 0, NaN, infinity
            else:
                for b in range(n_bands):
                    if not math.isfinite(x[r, columns[b]]):
                        valid = False
            if valid:
                for j in range(n_features):
                    value = 0.0
                    for t in range(tap_count[j]):
                        value += taps[j, t] * (x[r, columns[tap_index[j, t]]] - mean)
                    features[j] = value / deviation * inv_scale[j]
            else:
                for j in range(n_features):
                    features[j] = 0.0
            best = -math.inf
            best_k = -1
            for turn in range(n_classes):
                k = (guess + turn) % n_classes
                for j in range(n_features):
                    delta[j] = features[j] - centers[k, j]
                score = constants[k]
                lost = False
                for j in range(n_features):
                    y = np.float32(0.0)
                    for i in range(j + 1):
                        y += whiten[k, j, i] * delta[i]
                    score -= 0.5 * y * y
                    if score < best:
                        lost = True
                        break
                if not lost and (score > best or best_k < 0 or (score == best and k < best_k)):
                    best = score
                    best_k = k
            if best_k < 0:                        # every score NaN: cannot happen for a valid pixel
                best_k = 0
            guess = best_k
            out[r] = class_ids[best_k] if valid or invalid < 0 else invalid


@njit(parallel=True, fastmath={'reassoc', 'contract', 'arcp', 'afn'}, cache=True, nogil=True)
def _svm_decisions(dots, f_sq, sv_sq, gamma, pair_coef, intercept, out):
    """One-vs-one decision values of an RBF SVM (see :class:`HsiSvmClassifier`).

    ``out[r, p] = intercept[p] + sum_k pair_coef[k, p] exp(-gamma |f_r - s_k|^2)`` with
    ``|f - s|^2 = f_sq[r] + sv_sq[k] - 2 dots[r, k]`` (``dots = f @ s.T`` from BLAS). The
    distance and exponential are float32 (the product's own rounding is larger), the sums
    float64. No ``nnan``/``ninf``: an overflowing pixel stays NaN/infinite, not a class.
    """
    n, n_vectors = dots.shape
    n_pairs = pair_coef.shape[1]
    zero = np.float32(0.0)
    two = np.float32(2.0)
    for r in prange(n):
        acc = np.zeros(n_pairs)
        f_r = f_sq[r]
        for k in range(n_vectors):
            d = f_r + sv_sq[k] - two * dots[r, k]
            kv = math.exp(-gamma * max(d, zero))        # max: rounding at a support vector itself
            for q in range(n_pairs):
                acc[q] += kv * pair_coef[k, q]
        for q in range(n_pairs):
            out[r, q] = acc[q] + intercept[q]


@njit(cache=True, nogil=True)
def _svm_probability(decisions, prob_a, prob_b, n_classes, out):
    """libsvm ``svm_predict_probability``: Platt's sigmoid per class pair (clipped to
    [1e-7, 1 - 1e-7]) coupled into class probabilities (Wu, Lin & Weng 2004, method 2) exactly as
    libsvm's ``multiclass_probability`` iterates."""
    k = n_classes
    r = np.empty((k, k))
    q_mat = np.empty((k, k))
    qp = np.empty(k)
    p = np.empty(k)
    eps = 0.005 / k
    max_iter = max(100, k)
    for row in range(decisions.shape[0]):
        pair = 0
        for i in range(k):
            for j in range(i + 1, k):
                fapb = decisions[row, pair] * prob_a[pair] + prob_b[pair]
                if fapb >= 0.0:
                    value = math.exp(-fapb) / (1.0 + math.exp(-fapb))
                else:
                    value = 1.0 / (1.0 + math.exp(fapb))
                value = min(max(value, 1e-7), 1.0 - 1e-7)
                r[i, j] = value
                r[j, i] = 1.0 - value
                pair += 1
        if k == 2:
            out[row, 0] = r[0, 1]
            out[row, 1] = r[1, 0]
            continue
        for t in range(k):
            p[t] = 1.0 / k
            q_mat[t, t] = 0.0
            for j in range(t):
                q_mat[t, t] += r[j, t] * r[j, t]
                q_mat[t, j] = q_mat[j, t]
            for j in range(t + 1, k):
                q_mat[t, t] += r[j, t] * r[j, t]
                q_mat[t, j] = -r[j, t] * r[t, j]
        for _ in range(max_iter):
            pqp = 0.0
            for t in range(k):
                qp[t] = 0.0
                for j in range(k):
                    qp[t] += q_mat[t, j] * p[j]
                pqp += p[t] * qp[t]
            max_error = 0.0
            for t in range(k):
                error = abs(qp[t] - pqp)
                if error > max_error:
                    max_error = error
            if max_error < eps:
                break
            for t in range(k):
                diff = (-qp[t] + pqp) / q_mat[t, t]
                p[t] += diff
                pqp = (pqp + diff * (diff * q_mat[t, t] + 2.0 * qp[t])) / (1.0 + diff) / (1.0 + diff)
                for j in range(k):
                    qp[j] = (qp[j] + diff * q_mat[t, j]) / (1.0 + diff)
                    p[j] /= (1.0 + diff)
        for t in range(k):
            out[row, t] = p[t]


@njit(parallel=True, cache=True, nogil=True)
def _resample(x, left, right, weight, out):
    """``out[r, b] = x[r, left[b]] * (1 - weight[b]) + x[r, right[b]] * weight[b]`` (float32).

    A row whose input values are all the same (a saturated or dead pixel) comes out exactly
    constant, not constant up to float32 rounding: the classifiers recognise such a pixel by a
    standard deviation of exactly 0.
    """
    for r in prange(x.shape[0]):
        lo = x[r, left[0]]
        hi = lo
        for b in range(left.shape[0]):
            w = weight[b]
            a = x[r, left[b]]
            c = x[r, right[b]]
            lo = min(lo, a, c)
            hi = max(hi, a, c)
            out[r, b] = a * (np.float32(1.0) - w) + c * w
        if lo == hi:
            for b in range(left.shape[0]):
                out[r, b] = lo


class Resampled:
    """Binding of a model whose bands lie between the input's: linear interpolation by wavelength.

    Returned by :meth:`PushbroomLineClassifier.bind` in place of the column indices and passed to
    ``predict`` the same way. Model band ``b`` (``wanted[b]`` nm) is read from the two input
    columns around it, ``left[b]`` and ``right[b]``, weighted by the distance; a band just outside
    the input's range (:meth:`PushbroomLineClassifier._resampling` limits how far) is extrapolated
    from the two edge columns. A band exactly on an input column reads that column only, so a bad
    value in its neighbour does not reach the model. Read-only: workers share it.

    Attributes:
        max_shift_nm: Largest distance between a model band and its nearest input band (what
            binding without resampling would have had to accept).
        linear_weights: For a linear model: ``(input columns used, classes)`` weights with the
            interpolation folded in, so a line costs what it costs without resampling.
    """

    def __init__(self, wavelengths: np.ndarray, wanted: np.ndarray) -> None:
        wavelengths = np.asarray(wavelengths, dtype=np.float64)
        wanted = np.asarray(wanted, dtype=np.float64)
        right = np.clip(np.searchsorted(wavelengths, wanted), 1, wavelengths.size - 1)
        left = right - 1
        weight = ((wanted - wavelengths[left]) / (wavelengths[right] - wavelengths[left])).astype(np.float32)
        # on an input column: read that one only (a weight of 0 must not let NaN/inf in from the other)
        left, right = np.where(weight == 1.0, right, left), np.where(weight == 0.0, left, right)
        self.left = np.ascontiguousarray(left, dtype=np.intp)
        self.right = np.ascontiguousarray(right, dtype=np.intp)
        self.weight = np.ascontiguousarray(weight, dtype=np.float32)
        self.n_bands = int(wanted.size)
        self.max_shift_nm = float(np.abs(wavelengths[None, :] - wanted[:, None]).min(axis=1).max())
        # the same interpolation as a matrix over the input columns it reads (for linear models)
        self.columns = np.unique(np.concatenate([self.left, self.right]))
        matrix = np.zeros((self.columns.size, self.n_bands), dtype=np.float64)
        rows = np.arange(self.n_bands)
        np.add.at(matrix, (np.searchsorted(self.columns, self.left), rows), 1.0 - weight.astype(np.float64))
        np.add.at(matrix, (np.searchsorted(self.columns, self.right), rows), weight.astype(np.float64))
        self.matrix = matrix
        self.linear_weights = None

    def apply(self, binned: np.ndarray) -> np.ndarray:
        """``(n, model bands)`` float32: ``binned`` ``(n, input bands)`` on the model's bands."""
        x = np.ascontiguousarray(binned, dtype=np.float32)
        out = np.empty((x.shape[0], self.n_bands), np.float32)
        _resample(x, self.left, self.right, self.weight, out)
        return out


class PushbroomLineClassifier:
    """Classify each slit pixel of a binned frame; read-only, so workers can share it."""

    #: True for a model trained in this session (never a file: gone when the program closes)
    session = False

    def __init__(self, model_artifact_path: str) -> None:
        self._load(joblib.load(model_artifact_path), str(model_artifact_path))

    @classmethod
    def from_artifact(cls, artifact: dict, name: str) -> 'PushbroomLineClassifier':
        """A classifier of an artifact that is in memory only (``quick_model.train_artifact``).

        ``name`` stands in for the file path wherever the model is named (log, selector tile).
        """
        self = cls.__new__(cls)
        self._load(artifact, str(name))
        self.session = True
        return self

    def _load(self, artifact: dict, path: str) -> None:
        self.path = path
        if artifact.get('class_colors'):                   # class id -> [r, g, b] for the belt map
            self.class_colors = {int(k): [int(c) for c in v] for k, v in artifact['class_colors'].items()}
        self.W_fused = np.ascontiguousarray(artifact['W_fused'], dtype=np.float32)
        self.b_fused = np.ascontiguousarray(artifact['b_fused'], dtype=np.float32)
        self.selected_bands = np.asarray(artifact['selected_bands'], dtype=np.intp)
        self.classes_ = np.asarray(artifact['classes'])
        self.category_names: dict = dict(artifact.get('category_names', {}))
        # Temperature scaling of the confidence (train_sgd_coco.py fits it on held-out pixels);
        # 1.0 = sklearn's predict_proba. Never changes the class.
        self.temperature = float(artifact.get('temperature', 1.0))
        if not self.temperature > 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
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
             tolerance_nm: float = 0.5, resample: bool = True) -> tuple[np.ndarray | Resampled, str]:
        """Choose the input columns that feed the model (pass them to :meth:`predict`).

        Args:
            wavelengths: nm of every input column, or ``None`` when unknown.
            n_columns: Number of input columns.
            tolerance_nm: Largest accepted distance between a model band and its input column.
            resample: Allow interpolating the input onto the model's bands when they are
                further apart than ``tolerance_nm`` (see :meth:`_resampling`).

        Returns:
            ``(columns, description)``: input column of every selected model band, or a
            :class:`Resampled` binding, and a one-line description of how they were matched.

        Raises:
            ValueError: a model band has no input column within ``tolerance_nm`` and the input
                cannot be resampled either, or (without wavelengths) the input has another band
                count than the model was trained on.
        """
        wanted = self.selected_wavelengths
        if wanted is not None and wavelengths is not None:
            wavelengths = np.asarray(wavelengths, dtype=np.float64)
            columns = np.abs(wavelengths[None, :] - wanted[:, None]).argmin(axis=1)
            error = np.abs(wavelengths[columns] - wanted)
            if error.max() > tolerance_nm:
                why_not = self._resampling(wavelengths, wanted, tolerance_nm) if resample else 'resampling is off'
                if why_not is None:
                    binding = Resampled(wavelengths, wanted)
                    if self.W_fused is not None:           # linear model: fold the interpolation in
                        binding.linear_weights = np.ascontiguousarray(binding.matrix @ self.W_fused, np.float32)
                    described = (f"resampled to the model's bands: {wanted.size} bands {wanted.min():.1f}-"
                                 f"{wanted.max():.1f} nm interpolated from the input's "
                                 f"{wavelengths[0]:.1f}-{wavelengths[-1]:.1f} nm, nearest input band up to "
                                 f"{binding.max_shift_nm:.2f} nm away (another wavelength calibration than the "
                                 "training data)")
                    logger.info("Classifier %s", described)
                    return binding, described
                worst = int(error.argmax())
                raise ValueError(
                    f"Model band {wanted[worst]:.2f} nm has no input band within {tolerance_nm} nm "
                    f"(nearest {wavelengths[columns[worst]]:.2f} nm). Input covers "
                    f"{wavelengths[0]:.2f}-{wavelengths[-1]:.2f} nm in {wavelengths.size} bands; check the "
                    f"wavelength calibration (m, c, camera bin, resolution) against the training data. "
                    f"Not resampled: {why_not}.")
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

    #: Resampling needs input bands about as wide as the model's: at most this many times the
    #: model's band spacing (wider bands have already averaged away what the model looks at) ...
    RESAMPLE_MAX_COARSER = 1.25
    #: ... and at least this fraction of it (much narrower bands are noisier than the training's).
    RESAMPLE_MIN_FINER = 0.8
    #: A recalibration moves the whole grid, so the model's first or last band then lies outside
    #: the input: up to this fraction of a band it is extrapolated from the two edge bands.
    RESAMPLE_MAX_OUTSIDE = 0.5

    def _resampling(self, wavelengths: np.ndarray, wanted: np.ndarray, tolerance_nm: float) -> str | None:
        """``None`` when the input can be interpolated onto the model's bands, else why not.

        Interpolation stands in for a redone wavelength calibration: the same spectrograph, bands
        of the same width, shifted by a fraction of a band. So the input must run in increasing
        wavelength, cover every model band (at the two ends a band may lie up to
        ``RESAMPLE_MAX_OUTSIDE`` of a band, or ``tolerance_nm``, outside) and have the training
        data's band spacing (``RESAMPLE_MIN_FINER`` .. ``RESAMPLE_MAX_COARSER`` times).
        """
        if wavelengths.size < 2 or not np.all(np.diff(wavelengths) > 0):
            return "the input's wavelengths do not increase band by band"
        if self.band_wavelengths is None or self.band_wavelengths.size < 2:
            return "the model does not give its band spacing"
        model_step = float(np.median(np.abs(np.diff(self.band_wavelengths.astype(np.float64)))))
        outside = max(tolerance_nm, self.RESAMPLE_MAX_OUTSIDE * model_step)
        if wanted.min() < wavelengths[0] - outside or wanted.max() > wavelengths[-1] + outside:
            return (f"the model's bands {wanted.min():.1f}-{wanted.max():.1f} nm are not all inside the input's "
                    f"{wavelengths[0]:.1f}-{wavelengths[-1]:.1f} nm (at most {outside:.1f} nm outside)")
        inside = (wavelengths >= wanted.min() - model_step) & (wavelengths <= wanted.max() + model_step)
        input_step = float(np.median(np.diff(wavelengths[inside]))) if inside.sum() > 1 else float('inf')
        if not self.RESAMPLE_MIN_FINER * model_step <= input_step <= self.RESAMPLE_MAX_COARSER * model_step:
            return (f"the input's bands are {input_step:.2f} nm apart, the model's {model_step:.2f} nm "
                    "(another spectral resolution, not a shifted calibration)")
        return None

    def _scores(self, binned: np.ndarray, columns) -> np.ndarray:
        if isinstance(columns, Resampled):                 # interpolation folded into the weights
            scores = binned[:, columns.columns].astype(np.float32) @ columns.linear_weights
        else:
            scores = binned[:, columns].astype(np.float32) @ self.W_fused
        scores += self.b_fused
        return scores

    def predict(self, binned: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Class id (uint8) of every slit pixel of one ``(slit, n_bins)`` frame."""
        return self._class_ids[self._scores(binned, columns).argmax(axis=1)]

    def predict_confidence(self, binned: np.ndarray, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(class ids, confidence)`` of every slit pixel of one ``(slit, n_bins)`` frame.

        Confidence (float32, 0..1) is the probability of the winning class exactly as sklearn's
        ``predict_proba`` gives it for a ``log_loss`` SGD model: one-vs-rest, the sigmoid of every
        class score normalised to sum 1, after dividing the scores by the artifact's
        ``temperature`` (1.0 when absent: sklearn's value). Check it against labelled pixels with
        ``tools/check_calibration.py`` before trusting a threshold. A pixel with a NaN band gets
        0 (uncertain), so one bad pixel cannot turn a class's sum into NaN.
        """
        scores = self._scores(binned, columns)
        winner = scores.argmax(axis=1)
        if self.temperature != 1.0:
            scores /= self.temperature                    # T > 0: the winner stays the same
        np.clip(scores, -80.0, 80.0, out=scores)          # exp stays finite in float32
        np.negative(scores, out=scores)
        np.exp(scores, out=scores)
        scores += 1.0
        np.reciprocal(scores, out=scores)                 # sigmoid; monotonic, so the max is the winner
        confidence = scores.max(axis=1)
        confidence /= scores.sum(axis=1)
        confidence[np.isnan(confidence)] = 0.0
        return self._class_ids[winner], confidence


def load_classifier(model_artifact_path, mlc_reg_covar: float = 0.0) -> PushbroomLineClassifier:
    """The classifier of a model file: an HSI MLOps package (a zip) or a joblib artifact.

    ``mlc_reg_covar`` (``model.mlc_reg_covar``) only concerns a package: see
    :class:`HsiPackageClassifier`.
    """
    from hsi_package import is_package, load_package
    if is_package(model_artifact_path):
        package = load_package(model_artifact_path)
        if package.svm is not None:
            return HsiSvmClassifier(model_artifact_path, mlc_reg_covar, package=package)
        return HsiPackageClassifier(model_artifact_path, mlc_reg_covar, package=package)
    return PushbroomLineClassifier(model_artifact_path)


class _HsiPackageBase(PushbroomLineClassifier):
    """What every HSI MLOps package model shares: classes, colours, binding to every training
    band, and the per-pixel preprocessing up to the features (see the subclasses)."""

    _CHUNK = 4096                                 # slit pixels per pass (bounds the scratch arrays)

    def _setup(self, model_artifact_path, mlc_reg_covar, package):
        """Set the shared attributes; returns ``to_features`` (F, B) (see below)."""
        from hsi_package import load_package
        mlc_reg_covar = float(mlc_reg_covar)
        if not 0.0 <= mlc_reg_covar < float('inf'):
            raise ValueError(f"mlc_reg_covar must be >= 0, got {mlc_reg_covar!r}")
        self.mlc_reg_covar = mlc_reg_covar
        if package is None:
            package = load_package(model_artifact_path)
        self.package = package
        self.path = str(model_artifact_path)
        self.classes_ = package.classes
        if self.classes_.min() < 0 or self.classes_.max() > 254:
            raise ValueError("class ids must be 0..254 (255 marks lost lines)")
        self._class_ids = self.classes_.astype(np.uint8)
        self.category_names = dict(package.labels)
        self.class_colors = dict(package.colors)               # class id -> [r, g, b] chosen at training
        self.temperature = 1.0
        self.input_units = package.input_units
        self.input_units_declared = True
        # every band of the training axis feeds SNV / the derivative: bind() matches all of them
        self.band_wavelengths = package.wavelengths.astype(np.float32)
        self.n_bands = int(package.wavelengths.size)
        self.selected_bands = np.arange(self.n_bands, dtype=np.intp)
        self.model_bands = package.selected.astype(np.intp)     # the bands the classifier itself uses
        self.W_fused = self.b_fused = None                     # not a linear model
        self._snv = package.snv
        self._invalid_class = np.uint8(package.background[0]) if package.background else None
        self._invalid_id = -1 if self._invalid_class is None else int(self._invalid_class)
        # features before the scaler, from the (SNV) spectrum z: (z @ to_features.T)[F]. Rows of the
        # identity = band selection; rows of the Savitzky-Golay matrix = first derivative at those bands
        return (np.eye(self.n_bands) if package.derivative is None else package.derivative)[package.selected]

    def bind(self, wavelengths: np.ndarray | None, n_columns: int,
             tolerance_nm: float = 0.5, resample: bool = True) -> tuple[np.ndarray | Resampled, str]:
        """One input column per **training** band (SNV needs them all); see the base class."""
        columns, binding = super().bind(wavelengths, n_columns, tolerance_nm, resample)
        binding = f"{binding}; {self.chain}"
        if isinstance(columns, Resampled):                 # every training band interpolated
            return columns, binding
        if np.unique(columns).size != columns.size:
            raise ValueError(f"The input's {n_columns} bands cannot supply the model's {self.n_bands} training "
                             "bands one to one (two model bands fall on the same input band): the input must "
                             "have the training data's spectral resolution")
        return columns, binding

    def _spectra(self, binned: np.ndarray, columns: np.ndarray, dtype) -> tuple[np.ndarray, np.ndarray]:
        """``(z (n, B), valid (n,))``: the training bands of every pixel after SNV (if the package
        uses it); an invalid pixel (see the subclasses) is all zero."""
        x = np.take(binned, columns, axis=1).astype(dtype, copy=False)
        with np.errstate(invalid='ignore', over='ignore'):       # a NaN/inf pixel is handled, not reported
            if self._snv:
                x -= x.mean(axis=1, keepdims=True)
                deviation = np.einsum('ij,ij->i', x, x)
                deviation /= x.shape[1]
                np.sqrt(deviation, out=deviation)                # population SD, as the training
                good = deviation > 0                             # False for 0 and NaN
                good &= np.isfinite(deviation)
                deviation[~good] = 1.0
                x /= deviation[:, None]
            else:
                good = np.isfinite(x).all(axis=1)
        if not good.all():
            x[~good] = 0.0
        return x, good


class HsiPackageClassifier(_HsiPackageBase):
    """A model trained in HSI MLOps (``.hsipkg``), run as its training software runs it.

    Per slit pixel: SNV over **all** bands of the training axis (mean 0, population SD 1) or the
    Savitzky-Golay first derivative (each selected band from its window of neighbours, see
    ``hsi_package.savgol_derivative_matrix``), the package's selected bands, its StandardScaler,
    then the Gaussian maximum-likelihood classifier
    (per class a mean and a full covariance C): the class with the largest
    ``log prior - 0.5 log|C| - 0.5 d C^-1 d``, ``d`` = features minus the class mean.

    The input must therefore supply every training band, not only the selected ones:
    :meth:`bind` returns one input column per training band. Read-only after construction, so
    workers can share it (each call allocates its own arrays).

    With SNV, a pixel whose spectrum is constant (SD 0: a dead or saturated pixel, the all-zero
    warm-up frame) has no SNV; with or without SNV, neither has a pixel that is not finite. Such a
    pixel has no features. It gets the package's background class if it has one,
    otherwise the class the model gives the all-zero feature vector; its confidence is 0.

    ``mlc_reg_covar`` (``model.mlc_reg_covar``, default 0 = exactly the trained model) is added to
    the diagonal of every class covariance (scaled feature units, on top of the training's own
    ``reg_covar``). A full covariance over many bands has directions with almost no training
    variance; input a little noisier than the training cubes is then far from every class and
    falls to the class with the widest covariance (the background). A larger value makes the
    model tolerate that noise, at the price of no longer being the model that was validated.
    """

    def __init__(self, model_artifact_path, mlc_reg_covar: float = 0.0, package=None) -> None:
        """``package``: the already loaded ``hsi_package.HsiPackage`` of the file (else it is loaded)."""
        to_features = self._setup(model_artifact_path, mlc_reg_covar, package)
        package = self.package
        if package.means is None:
            raise ValueError(f"{self.path}: {package.algorithm} package, not an MLC (use load_classifier)")

        # d C^-1 d = |d @ A|^2 with C = L L^T and A = L^-T. With features f = (z[selected] - m) / s
        # this is |z @ W_k - t_k|^2 per class k: one matrix product gives all classes.
        n_classes, n_features = package.means.shape
        self._n_features = n_features
        weights = np.zeros((self.n_bands, n_classes * n_features))
        offsets = np.empty(n_classes * n_features)
        constants = np.empty(n_classes)
        whitening = np.empty((n_classes, n_features, n_features))
        centers = package.scaler_mean / package.scaler_scale + package.means         # (K, F)
        extra = mlc_reg_covar * np.eye(n_features)
        for k in range(n_classes):
            try:
                lower = np.linalg.cholesky(package.covariances[k] + extra)
            except np.linalg.LinAlgError as exc:
                raise ValueError(f"{self.path}: the covariance of class {int(self.classes_[k])} is not positive "
                                 "definite") from exc
            whitening[k] = np.tril(np.linalg.inv(lower))                             # L^-1, lower triangular
            part = slice(k * n_features, (k + 1) * n_features)
            weights[:, part] = to_features.T @ (whitening[k].T / package.scaler_scale[:, None])
            offsets[part] = centers[k] @ whitening[k].T
            constants[k] = np.log(package.priors[k]) - np.log(np.diag(lower)).sum()    # -0.5 log|C|
        self._weights = np.ascontiguousarray(weights, dtype=np.float32)
        self._offsets = offsets.astype(np.float32)
        self._constants = constants.astype(np.float32)
        # the same numbers for the kernel that predict() runs (_mlc_classify): the non-zero
        # entries of every row of to_features as taps
        rows = [np.flatnonzero(row) for row in to_features]
        self._tap_count = np.array([row.size for row in rows], dtype=np.intp)
        self._tap_index = np.zeros((n_features, max(self._tap_count.max(), 1)), dtype=np.intp)
        self._taps = np.zeros(self._tap_index.shape, dtype=np.float64)
        for j, row in enumerate(rows):
            self._tap_index[j, :row.size] = row
            self._taps[j, :row.size] = to_features[j, row]
        self._inv_scale = (1.0 / package.scaler_scale).astype(np.float32)
        self._centers = np.ascontiguousarray(centers, dtype=np.float32)
        self._whitening = np.ascontiguousarray(whitening, dtype=np.float32)
        self._constants64 = constants
        # compile (or load from the Numba cache) now, not on the first camera line
        self.predict(np.zeros((1, self.n_bands), np.float32), self.selected_bands)
        _resample(np.zeros((1, 2), np.float32), np.zeros(1, np.intp), np.ones(1, np.intp), np.zeros(1, np.float32),
                  np.empty((1, 1), np.float32))

    @property
    def chain(self) -> str:
        """The package's chain, and what this classifier was told to change of it."""
        if not self.mlc_reg_covar:
            return self.package.chain
        return f"{self.package.chain} (covariance + {self.mlc_reg_covar:g}: model.mlc_reg_covar, not as trained)"

    def _scores(self, binned: np.ndarray, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(scores (n, classes) float32, valid (n,) bool)``; an invalid pixel is scored as z = 0."""
        if isinstance(columns, Resampled):
            binned, columns = columns.apply(binned), self.selected_bands
        n = binned.shape[0]
        n_classes = self._constants.size
        scores = np.empty((n, n_classes), np.float32)
        valid = np.empty(n, bool)
        for start in range(0, n, self._CHUNK):
            part = slice(start, start + self._CHUNK)
            x, good = self._spectra(binned[part], columns, np.float32)
            y = x @ self._weights
            y -= self._offsets
            np.square(y, out=y)
            distance = y.reshape(x.shape[0], n_classes, self._n_features).sum(axis=2)
            distance *= -0.5
            distance += self._constants
            scores[part] = distance
            valid[part] = good
        return scores, valid

    def predict(self, binned: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Class id (uint8) of every slit pixel of one ``(slit, n_bins)`` frame.

        Runs :func:`_mlc_classify`: the arg max of the scores :meth:`_scores` computes in full,
        without finishing the classes that have already lost (a test compares the two).
        """
        if isinstance(columns, Resampled):
            binned, columns = columns.apply(binned), self.selected_bands
        x = np.ascontiguousarray(binned, dtype=np.float32)       # one compiled signature; no copy if float32
        out = np.empty(x.shape[0], np.uint8)
        _mlc_classify(x, np.ascontiguousarray(columns, dtype=np.intp), self._tap_count, self._tap_index, self._taps,
                      self._snv, self._inv_scale, self._centers, self._whitening, self._constants64,
                      self._class_ids, self._invalid_id, out)
        return out

    def predict_confidence(self, binned: np.ndarray, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(class ids, confidence)``: confidence is the posterior of the winning class.

        The posterior under the fitted Gaussian class models and priors (what the package calls
        ``gaussian_model_posterior``): the package states it is **not** calibrated. Pixels without
        SNV (see the class docstring) get 0.
        """
        scores, valid = self._scores(binned, columns)
        winner = scores.argmax(axis=1)
        ids = self._class_ids[winner]
        relative = scores.astype(np.float64)
        relative -= relative.max(axis=1, keepdims=True)
        np.maximum(relative, -700.0, out=relative)                   # exp stays above the float64 underflow
        np.exp(relative, out=relative)                               # the winner's term is exp(0) = 1
        confidence = (1.0 / relative.sum(axis=1)).astype(np.float32)
        if not valid.all():
            confidence[~valid] = 0.0
            if self._invalid_class is not None:
                ids[~valid] = self._invalid_class
        return ids, confidence


class HsiSvmClassifier(_HsiPackageBase):
    """An RBF support vector machine trained in HSI MLOps (``sklearn.svm.SVC`` in a ``.hsipkg``).

    Per slit pixel the package's preprocessing (SNV or the Savitzky-Golay first derivative, see
    :class:`HsiPackageClassifier`), its selected bands and StandardScaler give the features
    ``f``; then, as ``SVC.predict`` (libsvm) does: per class pair ``(i, j)`` the decision
    ``sum_s c_s exp(-gamma |f - s|^2) + b`` over the support vectors of both classes, a vote for
    ``i`` when it is positive, else for ``j``, and the class with most votes (the lowest on a
    tie). ``hsi_package.SvmModel`` has the details.

    Cost: every pixel against every support vector (``|f - s|^2 = |f|^2 + |s|^2 - 2 f.s``, the
    last term one float32 BLAS product per chunk of pixels, then one exponential per pixel and
    support vector). Far more work per pixel than an MLC with as many features; the float32
    product changes a decision only within ~1e-4 of zero (a tie of the vote).

    ``mlc_reg_covar`` does not apply (no covariances); a value other than 0 is shown as ignored.
    Pixels without features (see :class:`HsiPackageClassifier`) get the package's background
    class (else the class of z = 0) and confidence 0. Read-only after construction (workers share it).
    """

    _CHUNK = 1024                       # pixels per pass: the (pixels, support vectors) scratch stays small

    def __init__(self, model_artifact_path, mlc_reg_covar: float = 0.0, package=None) -> None:
        """``package``: the already loaded ``hsi_package.HsiPackage`` of the file (else it is loaded)."""
        to_features = self._setup(model_artifact_path, mlc_reg_covar, package)
        package = self.package
        svm = package.svm
        if svm is None:
            raise ValueError(f"{self.path}: {package.algorithm} package, not an SVM (use load_classifier)")
        # f = z @ affine + shift: the selected bands (or their derivative) through the scaler
        self._affine = np.ascontiguousarray((to_features / package.scaler_scale[:, None]).T)          # (B, F)
        self._shift = -package.scaler_mean / package.scaler_scale
        vectors = svm.support_vectors.astype(np.float32)
        self._vectors_t = np.ascontiguousarray(vectors.T)                                            # (F, S)
        self._vector_sq = (vectors.astype(np.float64) ** 2).sum(axis=1).astype(np.float32)   # of the values used
        self._gamma = np.float32(svm.gamma)
        pairs = svm.pairs
        start = np.concatenate([[0], np.cumsum(svm.n_support)])
        coef = np.zeros((vectors.shape[0], len(pairs)))
        for p, (i, j) in enumerate(pairs):
            coef[start[i]:start[i + 1], p] = svm.dual_coef[j - 1, start[i]:start[i + 1]]
            coef[start[j]:start[j + 1], p] = svm.dual_coef[i, start[j]:start[j + 1]]
        self._pair_coef = coef.astype(np.float32)
        self._intercept = np.ascontiguousarray(svm.intercept, dtype=np.float64)
        self._pair_i = np.array([i for i, _ in pairs], dtype=np.intp)
        self._pair_j = np.array([j for _, j in pairs], dtype=np.intp)
        self.n_support_vectors = int(vectors.shape[0])
        # compile (or load from the Numba cache) now, not on the first camera line
        self.predict_confidence(np.zeros((1, self.n_bands), np.float32), self.selected_bands)
        _resample(np.zeros((1, 2), np.float32), np.zeros(1, np.intp), np.ones(1, np.intp), np.zeros(1, np.float32),
                  np.empty((1, 1), np.float32))

    @property
    def chain(self) -> str:
        if not self.mlc_reg_covar:
            return self.package.chain
        return f"{self.package.chain} (model.mlc_reg_covar {self.mlc_reg_covar:g} ignored: not an MLC)"

    def _decisions(self, binned: np.ndarray, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(decisions (n, class pairs) float64, valid (n,) bool)``; an invalid pixel is z = 0."""
        if isinstance(columns, Resampled):
            binned, columns = columns.apply(binned), self.selected_bands
        n = binned.shape[0]
        decisions = np.empty((n, self._intercept.size))
        valid = np.empty(n, bool)
        for start in range(0, n, self._CHUNK):
            part = slice(start, start + self._CHUNK)
            z, good = self._spectra(binned[part], columns, np.float64)
            with np.errstate(over='ignore', invalid='ignore'):
                features = z @ self._affine
                features += self._shift
                features = features.astype(np.float32)
                dots = features @ self._vectors_t
                f_sq = np.einsum('ij,ij->i', features, features)
            _svm_decisions(dots, f_sq, self._vector_sq, self._gamma, self._pair_coef, self._intercept,
                           decisions[part])
            valid[part] = good
        return decisions, valid

    def _vote(self, decisions: np.ndarray) -> np.ndarray:
        """Index (into ``classes_``) of the class with most one-vs-one votes, the lowest on a tie."""
        votes = np.zeros((decisions.shape[0], self.classes_.size), np.int32)
        rows = np.arange(decisions.shape[0])
        for p in range(decisions.shape[1]):
            winner = np.where(decisions[:, p] > 0, self._pair_i[p], self._pair_j[p])
            votes[rows, winner] += 1
        return votes.argmax(axis=1)

    def predict(self, binned: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Class id (uint8) of every slit pixel of one ``(slit, n_bins)`` frame (``SVC.predict``)."""
        decisions, valid = self._decisions(binned, columns)
        ids = self._class_ids[self._vote(decisions)]
        if self._invalid_class is not None and not valid.all():
            ids[~valid] = self._invalid_class
        return ids

    def predict_confidence(self, binned: np.ndarray, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(class ids, confidence)``: the ids of :meth:`predict`; the confidence is libsvm's
        probability of that class (``SVC.predict_proba``, Platt scaling fitted by cross-validation
        at training: may disagree with the vote near a boundary). Without Platt parameters
        (trained with ``probability=False``) it is the winner's share of the votes."""
        decisions, valid = self._decisions(binned, columns)
        winner = self._vote(decisions)
        ids = self._class_ids[winner]
        svm = self.package.svm
        if svm.prob_a is not None:
            probability = np.empty((decisions.shape[0], self.classes_.size))
            _svm_probability(decisions, svm.prob_a, svm.prob_b, self.classes_.size, probability)
            confidence = probability[np.arange(winner.size), winner].astype(np.float32)
        else:
            votes = np.zeros(winner.size, np.float32)
            for p in range(decisions.shape[1]):
                votes += np.where(decisions[:, p] > 0, self._pair_i[p], self._pair_j[p]) == winner
            confidence = votes / max(self.classes_.size - 1, 1)
        if not valid.all():
            confidence[~valid] = 0.0
            if self._invalid_class is not None:
                ids[~valid] = self._invalid_class
        return ids, confidence

    def probabilities(self, binned: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """``(n, classes)`` libsvm probabilities (``SVC.predict_proba``); offline checks only."""
        svm = self.package.svm
        if svm.prob_a is None:
            raise ValueError(f"{self.path}: the SVM was trained without probability estimates")
        decisions, _ = self._decisions(binned, columns)
        probability = np.empty((decisions.shape[0], self.classes_.size))
        _svm_probability(decisions, svm.prob_a, svm.prob_b, self.classes_.size, probability)
        return probability


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

    def bind(self, wavelengths, n_columns: int, tolerance_nm: float = 0.5, resample: bool = True):
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
