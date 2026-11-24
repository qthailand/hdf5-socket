# -*- coding: utf-8 -*-
"""Read a trained-model package of HSI MLOps (``.hsipkg``) without ``hsi_lab_core``.

The models of this project are trained elsewhere (HSI MLOps, ``hsi_lab_core``) and arrive as a
package: a zip named ``*.hsipkg`` (often inside another ``.zip``) holding ``manifest.json``,
``model.joblib``, ``scaler.joblib``, ``pipeline.joblib`` and
``training_spectral_reference.json``. :func:`load_package` turns one into plain numpy arrays, so
``inference_engine.HsiPackageClassifier`` can run the same chain as the training software:

    reflectance (159 bands) -> SNV per pixel -> band selection -> StandardScaler -> MLC
    reflectance (159 bands) -> Savitzky-Golay 1st derivative -> band selection -> StandardScaler -> MLC | SVM
    reflectance (159 bands) -> [SNV | 1st derivative] -> band selection -> [PCA] -> StandardScaler -> MLC
    reflectance (159 bands) -> [SNV | 1st derivative] -> band selection -> [PCA] -> StandardScaler -> SGD | PLS-DA

What is read, and how it is kept honest:

* ``manifest.json`` says which steps the package uses. Anything this reader does not implement
  (another preprocessing step, spectral indices, resampling, another algorithm, PCA before an
  SVM) is refused by name; nothing is skipped silently.
* The joblib files are pickles of ``hsi_lab_core`` objects (an SVM: of ``sklearn.svm.SVC``; an
  SGD: of ``sklearn.linear_model.SGDClassifier``; a PLS-DA: of ``sklearn.cross_decomposition.
  PLSRegression`` inside ``hsi_lab_core``'s ``PLSDAClassifier``; a PCA: of ``spectral_pipeline.SafePCA``).
  They are read with a restricted unpickler: numpy arrays are loaded, ``hsi_lab_core`` classes
  and those few sklearn / PCA classes become plain attribute holders, every other global is
  refused. No code of the package runs and neither ``hsi_lab_core`` nor ``sklearn`` is needed.
* An SGD or PLS-DA package is linear from the input bands on (PCA, scaler, the one-vs-rest
  scores or the PLS regression are matrix products): ``inference_engine.HsiLinearClassifier``
  folds it into one ``(bands, classes)`` matrix. A PCA's mean, the scaler's mean and variance of
  the PCA scores are checked against the training reference and the PCA's own
  ``explained_variance_``. A PLS-DA predicts one response per class (one-hot coded) and the class
  is the largest response; its stored training scores give back the one-hot responses it was
  fitted on, which are checked to be one-hot with the reference's class counts.
* ``artifact_sha256`` of the manifest is checked for every file that is read.
* ``training_spectral_reference.json`` (class means of the training spectra after the
  preprocessing) is the package's own evidence: the class means of the model must equal those
  spectra passed through this reader's band selection and scaler (they do, to 1e-13, for the
  packages this was written against), and the SNV convention (population SD over all bands) is
  checked on them. The first derivative is linear, so the class means of the raw training
  spectra, derived here, must give the preprocessed ones (``scipy.signal.savgol_filter`` with
  ``mode='interp'``: polynomial fits over the first and last windows at the ends). The reference
  also gives the count, mean and SD of every class, so the mean and variance the StandardScaler
  was fitted with must come out of it (they do, to 1e-14) - for an SVM, whose support vectors
  cannot be compared with class means, this and its RBF ``gamma`` ('scale' = 1 / (features x
  variance of the scaled training data)) are the evidence. A package that fails this is
  refused: its chain is not the one implemented here.

``python hsi_package.py <package>`` prints what a package contains and whether it passed.
"""
from __future__ import annotations

import bz2
import gzip
import hashlib
import importlib
import inspect
import io
import json
import logging
import lzma
import pickle
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_ALGORITHMS = ('MLC', 'SVM', 'SGD', 'PLS-DA')
# classes of other libraries a package may pickle, read as attribute holders like hsi_lab_core's
_SVM_HOLDERS = {('sklearn.svm._classes', 'SVC'), ('sklearn.svm.classes', 'SVC')}
_SGD_HOLDERS = {('sklearn.linear_model._stochastic_gradient', 'SGDClassifier'),
                ('sklearn.linear_model.stochastic_gradient', 'SGDClassifier')}
_SGD_LOSS_HOLDERS = ({(module, name) for module in ('sklearn.linear_model._sgd_fast', 'sklearn.linear_model.sgd_fast')
                      for name in ('Hinge', 'Log', 'LogLoss', 'ModifiedHuber', 'SquaredHinge')}
                     # log_loss since sklearn 1.6: the Cython loss of sklearn._loss (its module reads '_loss')
                     | {(module, 'CyHalfBinomialLoss') for module in ('_loss', 'sklearn._loss._loss')})
_PCA_HOLDERS = {('spectral_pipeline', 'SafePCA'), ('sklearn.decomposition._pca', 'PCA')}
_PLS_HOLDERS = {('sklearn.cross_decomposition._pls', 'PLSRegression'),
                ('sklearn.cross_decomposition.pls_', 'PLSRegression')}
_FOREIGN_HOLDERS = _SVM_HOLDERS | _SGD_HOLDERS | _SGD_LOSS_HOLDERS | _PCA_HOLDERS | _PLS_HOLDERS
# the 'loss' of a PLS-DA LinearModel: its scores are regression responses (confidence = their softmax)
PLS_RESPONSE = 'pls_response'
_PLS_TOLERANCE = 1e-6                 # PLS coef_ vs its rotations / loadings, recovered one-hot responses
# SGDClassifier losses whose decision is a class (the classifier runs every one the same way)
_SGD_LOSSES = ('hinge', 'log_loss', 'log', 'modified_huber', 'squared_hinge', 'perceptron')
_SCALER_TOLERANCE = 1e-8              # scaler mean / variance vs the ones the reference gives (relative)
_GAMMA_TOLERANCE = 1e-8               # SVM gamma 'scale' vs the one the reference gives (relative)
# preprocessing step lists of pipeline_audit this reader implements (in this order, nothing else)
SUPPORTED_STEPS = ('snv', 'first_derivative')
_SUPPORTED_CHAINS = ((), ('snv',), ('first_derivative',))
_DERIVATIVE_IMPLEMENTATION = 'hsi_lab_core.preprocessing.savgol_derivative'
_REFERENCE_DOMAIN = 'preprocessed_wavelength_selected'
_RAW_DOMAIN = 'raw_wavelength_selected'
_MEAN_TOLERANCE = 1e-6                # model class means vs the package's reference spectra (scaled units)
_DERIVATIVE_TOLERANCE = 1e-8          # derived raw class means vs the preprocessed ones (relative to their size)


# ------------------------------------------------------------------------------ package ---

@dataclass
class TrainingClass:
    """The training spectra of one class, summarised per band (``training_spectral_reference.json``)."""
    class_id: int
    label: str
    n_spectra: int
    mean: np.ndarray                 # (B,) on HsiPackage.wavelengths
    sd: np.ndarray                   # (B,)


@dataclass
class SvmModel:
    """An RBF ``sklearn.svm.SVC`` as libsvm evaluates it (float64, exactly as stored).

    Classes are numbered by their index in ``HsiPackage.classes``; class pairs ``(i, j)``, ``i <
    j``, run in the order (0, 1), (0, 2), ..., (1, 2), ... For pair ``p = (i, j)`` the decision
    is ``sum over the support vectors s of class i of dual_coef[j - 1, s] K(s, x) + sum over those
    of class j of dual_coef[i, s] K(s, x) + intercept[p]`` with ``K = exp(-gamma |s - x|^2)``;
    positive votes for ``i``, otherwise ``j``; the class with most votes wins (the lowest index
    on a tie). That is ``SVC.predict`` (libsvm ``svm_predict``) with ``break_ties=False``.
    """
    support_vectors: np.ndarray      # (S, F) in scaled feature units, grouped by class
    n_support: np.ndarray            # (K,) support vectors per class, in class order
    dual_coef: np.ndarray            # (K - 1, S) libsvm layout (SVC._dual_coef_)
    intercept: np.ndarray            # (P,) = -rho per class pair (SVC._intercept_)
    gamma: float                     # the fitted RBF gamma (SVC._gamma)
    gamma_setting: object            # SVC.gamma as trained: 'scale', 'auto' or a number
    prob_a: np.ndarray | None        # (P,) Platt sigmoid per pair (probability=True), else None
    prob_b: np.ndarray | None

    @property
    def pairs(self) -> list:
        k = self.n_support.size
        return [(i, j) for i in range(k) for j in range(i + 1, k)]


@dataclass
class LinearModel:
    """An ``sklearn.linear_model.SGDClassifier`` or a PLS-DA (float64, from the stored values).

    Scores ``f @ coef.T + intercept`` of the (scaled) features ``f``; with more than two classes
    (always for a PLS-DA) one row per class and the class with the largest score wins. An SGD
    with two classes has one row: a score > 0 gives the second class, otherwise the first
    (``SGDClassifier.predict``). For a PLS-DA the scores are ``PLSRegression.predict`` (the
    regression's own centring folded into ``intercept``) and ``loss`` is ``PLS_RESPONSE``.
    """
    coef: np.ndarray                 # (K, F), or (1, F) with two classes
    intercept: np.ndarray            # (K,), or (1,)
    loss: str                        # what it was trained with ('hinge' has no probabilities)
    components: int | None = None    # PLS-DA: latent components of the regression
    fit_counts: np.ndarray | None = None   # PLS-DA: rows per class of the fit (from its one-hot responses)
    fit_accuracy: float | None = None      # PLS-DA: share of the fitted rows the largest response gets right


@dataclass
class HsiPackage:
    """One trained model as plain arrays (float64, exactly as stored)."""
    path: str
    name: str                        # manifest name (free text of the training software)
    algorithm: str
    classes: np.ndarray              # (K,) class ids the model returns
    labels: dict                     # class id -> name
    colors: dict                     # class id -> [r, g, b] (classes the manifest gives a colour)
    background: tuple                # class ids the manifest marks as background
    wavelengths: np.ndarray          # (B,) nm: every band the input must supply (SNV runs over all)
    snv: bool
    selected: np.ndarray             # (F,) indices into ``wavelengths`` the model uses
    scaler_mean: np.ndarray          # (F,)
    scaler_scale: np.ndarray         # (F,)
    means: np.ndarray | None         # MLC: (K, F) in scaled units (None for an SVM)
    covariances: np.ndarray | None   # MLC: (K, F, F), regularisation included
    priors: np.ndarray | None        # MLC: (K,)
    input_units: str                 # 'reflectance' | 'counts'
    metric: str                      # the selection score the package reports
    exported_at: str
    lab_core_version: str
    checks: list = field(default_factory=list)   # what was verified while loading (text lines)
    # training spectra per class, for display: {'input': [TrainingClass], 'preprocessed': [TrainingClass]}
    # ('input' = what the model is given, 'preprocessed' = after the preprocessing); a list is empty
    # when the package lacks it
    training: dict = field(default_factory=dict)
    # Savitzky-Golay first derivative: z = derivative @ x per pixel, (B, B) float64; None = not used
    derivative: np.ndarray | None = None
    derivative_parameters: dict = field(default_factory=dict)    # window, polyorder, delta (nm)
    svm: SvmModel | None = None      # the model of an SVM package (None for an MLC)
    scaler_var: np.ndarray | None = None      # (F,) variance the scaler was fitted with, if stored
    scaler_samples: float | None = None       # rows the scaler was fitted on, if stored
    linear: LinearModel | None = None         # the model of an SGD package
    # PCA between the selected bands and the scaler (not before an SVM): p = (x[selected] - pca_mean)
    # @ pca_components.T, then the scaler works on p (scaler_mean / scaler_scale are then (P,))
    pca_mean: np.ndarray | None = None        # (F,)
    pca_components: np.ndarray | None = None  # (P, F)
    pca_variance: np.ndarray | None = None    # (P,) explained_variance_ (sample variance of the scores), if stored
    pca_samples: float | None = None          # rows the PCA was fitted on, if stored

    @property
    def model_text(self) -> str:
        if self.svm is not None:
            return (f"SVM {self.classes.size} classes (RBF, {self.svm.support_vectors.shape[0]} support vectors, "
                    "one-vs-one vote)")
        if self.linear is not None and self.linear.loss == PLS_RESPONSE:
            return f"PLS-DA {self.classes.size} classes ({self.linear.components} components, largest response)"
        if self.linear is not None:
            return f"SGD {self.classes.size} classes (linear, loss {self.linear.loss})"
        return f"{self.algorithm} {self.classes.size} classes"

    @property
    def chain(self) -> str:
        steps = [f"{self.input_units} {self.wavelengths.size} bands"]
        if self.snv:
            steps.append('SNV')
        if self.derivative is not None:
            p = self.derivative_parameters
            steps.append(f"1st derivative (Savitzky-Golay {p['window']}/{p['polyorder']})")
        if self.selected.size != self.wavelengths.size:
            steps.append(f"{self.selected.size} bands")
        if self.pca_components is not None:
            steps.append(f"PCA {self.pca_components.shape[0]}")
        steps += ['StandardScaler', self.model_text]
        return ' -> '.join(steps)

    def describe(self) -> str:
        wl = self.wavelengths
        lines = [f"{Path(self.path).name}: {self.algorithm} '{self.name}', hsi_lab_core {self.lab_core_version}, "
                 f"exported {self.exported_at}",
                 f"  chain    {self.chain}",
                 f"  bands    {wl.size} at {wl[0]:.2f}-{wl[-1]:.2f} nm; the model uses {self.selected.size}",
                 f"  classes  " + ', '.join(f"{int(c)} {self.labels[int(c)]}" for c in self.classes),
                 f"  score    {self.metric}"]
        lines += [f"  check    {text}" for text in self.checks]
        return '\n'.join(lines)


def is_package(path) -> bool:
    """``True`` for a zip file (a ``.hsipkg``, or a zip around one); a joblib artifact is not."""
    try:
        return zipfile.is_zipfile(path)
    except OSError:
        return False


def load_package(path) -> HsiPackage:
    """Read and verify a ``.hsipkg`` (or a ``.zip`` holding exactly one).

    Raises:
        ValueError: not a model package, a file does not match its checksum, the package uses a
            step this reader does not implement, or it fails the check against its own training
            reference.
    """
    path = Path(path)
    files = _read_files(path)
    try:
        manifest = json.loads(files['manifest.json'].decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name}: manifest.json is not readable JSON ({exc})") from exc
    checks = [_check_sha256(path, manifest, files, ('model.joblib', 'scaler.joblib', 'pca.joblib',
                                                    'training_spectral_reference.json'))]

    algorithm = str(manifest.get('algorithm'))
    if algorithm not in SUPPORTED_ALGORITHMS or manifest.get('task_type') != 'classification':
        raise ValueError(f"{path.name}: algorithm {algorithm!r} ({manifest.get('task_type')}) is not supported; "
                         f"this reader runs {', '.join(SUPPORTED_ALGORITHMS)} classification packages")
    wavelengths, snv, derivative_parameters, selected = _feature_chain(path, manifest)
    derivative = None if not derivative_parameters else savgol_derivative_matrix(
        wavelengths.size, derivative_parameters['window'], derivative_parameters['polyorder'],
        derivative_parameters['delta'])
    pca_mean = pca_components = pca_variance = pca_samples = None
    if (manifest.get('pipeline_audit') or {}).get('pca') is not None:
        if algorithm == 'SVM':                  # its gamma check needs the band features
            raise ValueError(f"{path.name}: the package uses pca before an SVM, which this reader does not "
                             "implement (pca before an MLC, SGD or PLS-DA model)")
        pca_mean, pca_components, pca_variance, pca_samples = _pca(path, manifest, files, selected.size)
    n_scaled = selected.size if pca_components is None else pca_components.shape[0]
    scaler_mean, scaler_scale, scaler_var, scaler_samples = _scaler(path, manifest, files, n_scaled)
    svm = means = covariances = priors = linear = None
    if algorithm == 'SVM':
        classes, svm = _svm(path, files, n_scaled)
    elif algorithm == 'SGD':
        classes, linear = _sgd(path, files, n_scaled)
    elif algorithm == 'PLS-DA':
        classes, linear = _plsda(path, files, n_scaled)
    else:
        classes, means, covariances, priors = _mlc(path, files, n_scaled)

    labels = [str(name) for name in manifest.get('labels', [])]
    if len(labels) != classes.size or not np.array_equal(classes, np.arange(classes.size)):
        raise ValueError(f"{path.name}: the model's classes {classes.tolist()} do not number the manifest's "
                         f"{len(labels)} labels 0..{len(labels) - 1}")
    names = {int(c): labels[int(c)] for c in classes}
    colors = {}
    for cid, name in names.items():
        rgb = _hex_rgb((manifest.get('class_colors') or {}).get(name))
        if rgb is not None:
            colors[cid] = rgb
    flags = manifest.get('class_background') or {}
    background = tuple(cid for cid, name in names.items() if flags.get(name) is True)

    contract = manifest.get('input_contract') or {}
    mode, units = contract.get('measurement_mode'), contract.get('units')
    if mode == 'reflectance' and units not in (None, 'fraction'):
        raise ValueError(f"{path.name}: the model expects reflectance in {units!r}; this pipeline gives a fraction")
    input_units = 'reflectance' if mode == 'reflectance' else 'counts'

    package = HsiPackage(
        path=str(path), name=str(manifest.get('name', '')), algorithm=algorithm, classes=classes, labels=names,
        colors=colors, background=background, wavelengths=wavelengths, snv=snv, selected=selected,
        scaler_mean=scaler_mean, scaler_scale=scaler_scale, means=means, covariances=covariances, priors=priors,
        input_units=input_units, metric=_metric_text(manifest), exported_at=str(manifest.get('exported_at', '?')),
        lab_core_version=str(manifest.get('hsi_lab_core_version', '?')), checks=checks,
        derivative=derivative, derivative_parameters=derivative_parameters, svm=svm, scaler_var=scaler_var,
        scaler_samples=scaler_samples, linear=linear, pca_mean=pca_mean, pca_components=pca_components,
        pca_variance=pca_variance, pca_samples=pca_samples)
    checks.append(_check_reference(path, package, files.get('training_spectral_reference.json')))
    package.training = _training_spectra(package, files.get('training_spectral_reference.json'))
    logger.info("Model package loaded\n%s", package.describe())
    return package


# -------------------------------------------------------------------------------- files ---

_MAX_MEMBER_BYTES = 512 * 1024 * 1024     # refuse absurd members instead of filling the RAM


def _read_files(path: Path) -> dict:
    """Member name -> bytes of the package (looking inside a wrapping zip)."""
    if not path.is_file():
        raise FileNotFoundError(f"Model package not found: {path}")
    if not zipfile.is_zipfile(path):
        raise ValueError(f"{path.name} is not a model package (.hsipkg is a zip file)")
    with zipfile.ZipFile(path) as outer:
        names = outer.namelist()
        if 'manifest.json' in names:
            return _members(outer, path.name)
        inner = [n for n in names if n.lower().endswith('.hsipkg')]
        if len(inner) != 1:
            raise ValueError(f"{path.name}: expected manifest.json or exactly one .hsipkg inside, found "
                             f"{inner or names[:5]}")
        _size_ok(outer.getinfo(inner[0]), path.name)
        blob = outer.read(inner[0])
    if not zipfile.is_zipfile(io.BytesIO(blob)):
        raise ValueError(f"{path.name}: {inner[0]} is not a zip file")
    with zipfile.ZipFile(io.BytesIO(blob)) as package:
        if 'manifest.json' not in package.namelist():
            raise ValueError(f"{path.name}: {inner[0]} has no manifest.json")
        return _members(package, inner[0])


def _size_ok(info: zipfile.ZipInfo, where: str) -> None:
    if info.file_size > _MAX_MEMBER_BYTES:
        raise ValueError(f"{where}: {info.filename} is {info.file_size / 1e6:.0f} MB, too large for a model package")


def _members(package: zipfile.ZipFile, where: str) -> dict:
    wanted = ('manifest.json', 'model.joblib', 'scaler.joblib', 'training_spectral_reference.json', 'pca.joblib')
    files = {}
    for name in wanted:
        if name in package.namelist():
            _size_ok(package.getinfo(name), where)
            files[name] = package.read(name)
    missing = [name for name in wanted[:3] if name not in files]
    if missing and 'manifest.json' in missing:
        raise ValueError(f"{where}: no manifest.json")
    return files


def _check_sha256(path: Path, manifest: dict, files: dict, names: tuple) -> str:
    expected = manifest.get('artifact_sha256') or {}
    checked = []
    for name in names:
        if name in files and name in expected:
            if hashlib.sha256(files[name]).hexdigest() != str(expected[name]).lower():
                raise ValueError(f"{path.name}: {name} does not match its sha256 in the manifest "
                                 "(the package is damaged or was edited)")
            checked.append(name)
    if not checked:
        return "sha256: the manifest lists no checksums (not verified)"
    return f"sha256 ok: {', '.join(checked)}"


# ---------------------------------------------------------------------------- unpickler ---

class _Holder:
    """Stands in for a ``hsi_lab_core`` object: keeps the pickled attributes, runs nothing."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __setstate__(self, state) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)


def _plain_dtype(*args) -> np.dtype:
    """``numpy.dtype`` for numbers, booleans and text only: an object dtype would carry pickles."""
    dtype = np.dtype(*args)
    if dtype.hasobject or dtype.kind not in 'biufcSU':
        raise pickle.UnpicklingError(f"refused dtype {dtype}: a model package holds plain numeric arrays")
    return dtype


def _numpy_multiarray():
    for candidate in ('numpy._core.multiarray', 'numpy.core.multiarray'):       # numpy >= 2, then numpy 1.x
        try:
            return importlib.import_module(candidate)
        except ImportError:
            continue
    raise ImportError('numpy multiarray module not found')


def _plain_scalar(dtype, data):
    """``numpy.core.multiarray.scalar`` for plain dtypes (it unpickles ``data`` for object ones)."""
    if not isinstance(dtype, np.dtype) or dtype.hasobject or not isinstance(data, bytes):
        raise pickle.UnpicklingError("refused a numpy scalar that is not a plain number")
    return _numpy_multiarray().scalar(dtype, data)


def _numpy_global(module: str, name: str):
    if (module, name) == ('numpy', 'ndarray'):
        return np.ndarray
    if (module, name) == ('numpy', 'dtype'):
        return _plain_dtype
    if module in ('numpy.core.multiarray', 'numpy._core.multiarray'):
        if name == '_reconstruct':
            return _numpy_multiarray()._reconstruct
        if name == 'scalar':
            return _plain_scalar
    return None


def _restricted_unpickler():
    """joblib's unpickler limited to numeric numpy arrays and attribute holders (built on first use)."""
    from joblib.numpy_pickle import NumpyArrayWrapper, NumpyUnpickler

    class PlainArrayWrapper(NumpyArrayWrapper):
        """joblib reads an object array with an unrestricted ``pickle.load``: refuse those."""

        def read(self, unpickler, *args, **kwargs):
            dtype = getattr(self, 'dtype', None)
            if not isinstance(dtype, np.dtype) or dtype.hasobject or dtype.kind not in 'biufc':
                raise pickle.UnpicklingError(f"refused an array of dtype {dtype}: a model package holds "
                                             "numeric arrays")
            if getattr(self, 'subclass', np.ndarray) is not np.ndarray:
                raise pickle.UnpicklingError("refused an array subclass")
            return super().read(unpickler, *args, **kwargs)

    class RestrictedUnpickler(NumpyUnpickler):
        def find_class(self, module, name):
            if (module, name) == ('joblib.numpy_pickle', 'NumpyArrayWrapper'):
                return PlainArrayWrapper
            found = _numpy_global(module, name)
            if found is not None:
                return found
            if module == 'hsi_lab_core' or module.startswith('hsi_lab_core.') or (module, name) in _FOREIGN_HOLDERS:
                return type(name, (_Holder,), {'__module__': module})
            raise pickle.UnpicklingError(f"refused {module}.{name}: a model package may hold numpy arrays, "
                                         "hsi_lab_core objects and the sklearn SVC / SGDClassifier / "
                                         "PLSRegression / PCA this reader runs only")

    return RestrictedUnpickler


def _decompressed(blob: bytes, what: str) -> bytes:
    """The pickle stream of a joblib file (joblib may wrap it in one of these compressors)."""
    if blob[:1] == b'\x80':
        return blob
    try:
        if blob[:2] == b'\x1f\x8b':
            return gzip.decompress(blob)
        if blob[:3] == b'BZh':
            return bz2.decompress(blob)
        if blob[:6] == b'\xfd7zXZ\x00' or blob[:3] == b'\x5d\x00\x00':
            return lzma.decompress(blob)
        if blob[:1] == b'\x78':
            return zlib.decompress(blob)
    except (OSError, EOFError, zlib.error, lzma.LZMAError) as exc:
        raise ValueError(f"{what}: cannot decompress ({exc})") from exc
    raise ValueError(f"{what}: not a joblib pickle (starts with {blob[:4]!r})")


def _load_joblib(blob: bytes, what: str):
    """Attributes of the object pickled in a joblib file, without importing its class."""
    unpickler_class = _restricted_unpickler()
    handle = io.BytesIO(_decompressed(blob, what))
    kwargs = {}
    if 'ensure_native_byte_order' in inspect.signature(unpickler_class.__init__).parameters:
        kwargs['ensure_native_byte_order'] = True              # joblib >= 1.5
    try:
        loaded = unpickler_class(what, handle, **kwargs).load()
    except pickle.UnpicklingError as exc:
        raise ValueError(f"{what}: {exc}") from exc
    except Exception as exc:                                    # truncated or foreign pickle
        raise ValueError(f"{what}: cannot be read as a joblib pickle ({type(exc).__name__}: {exc})") from exc
    if not isinstance(loaded, _Holder):
        raise ValueError(f"{what}: holds a {type(loaded).__name__}, not an hsi_lab_core object")
    return loaded


def _array(holder, name: str, what: str, shape: tuple | None = None) -> np.ndarray:
    value = getattr(holder, name, None)
    if value is None:
        raise ValueError(f"{what}: no {name}")
    try:
        value = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what}: {name} is not numeric ({exc})") from exc
    if shape is not None and value.shape != shape:
        raise ValueError(f"{what}: {name} has shape {value.shape}, expected {shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{what}: {name} has NaN or infinite values")
    return value


# ----------------------------------------------------------------------- manifest parts ---

def savgol_derivative_matrix(n_bands: int, window: int, polyorder: int, delta: float) -> np.ndarray:
    """``D`` (n_bands, n_bands) with ``D @ x`` = the Savitzky-Golay first derivative of ``x``.

    Each band gets the slope (per ``delta``) of the least-squares polynomial of degree
    ``polyorder`` over ``window`` bands centred on it; within half a window of either end the
    window stays on the first (last) ``window`` bands. That is ``scipy.signal.savgol_filter(x,
    window, polyorder, deriv=1, delta=delta, mode='interp')``, written without scipy.
    """
    half = window // 2
    matrix = np.zeros((n_bands, n_bands))
    for band in range(n_bands):
        start = min(max(band - half, 0), n_bands - window)
        offsets = np.arange(start, start + window, dtype=np.float64) - band
        vander = np.vander(offsets, polyorder + 1, increasing=True)        # columns u^0 .. u^polyorder
        matrix[band, start:start + window] = np.linalg.pinv(vander)[1] / delta   # the u^1 coefficient
    return matrix


def _whole(value) -> int | None:
    """``value`` as an int when it is a whole number (not a bool), else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) \
            or value != int(value):
        return None
    return int(value)


def _derivative_parameters(path: Path, step: dict, n_bands: int) -> dict:
    """``{window, polyorder, delta}`` of the ``first_derivative`` step, checked."""
    implementation = step.get('implementation')
    if implementation not in (None, _DERIVATIVE_IMPLEMENTATION):
        raise ValueError(f"{path.name}: first_derivative by {implementation!r} is not supported "
                         f"({_DERIVATIVE_IMPLEMENTATION} is)")
    parameters = step.get('parameters') or {}
    if not isinstance(parameters, dict):
        raise ValueError(f"{path.name}: the parameters of first_derivative are not a mapping ({parameters!r})")
    if parameters.get('deriv', 1) != 1 or parameters.get('mode', 'interp') != 'interp':
        raise ValueError(f"{path.name}: first_derivative with deriv {parameters.get('deriv')!r} / mode "
                         f"{parameters.get('mode')!r} is not supported (deriv 1, mode 'interp' is)")
    window, polyorder = _whole(parameters.get('window')), _whole(parameters.get('polyorder'))
    delta = parameters.get('delta')
    units = parameters.get('units', 'per_nm')
    if units != 'per_nm':
        raise ValueError(f"{path.name}: first_derivative in units {units!r} is not supported (per_nm is)")
    if window is None or polyorder is None or isinstance(delta, bool) or not isinstance(delta, (int, float)) \
            or not np.isfinite(delta) or delta <= 0:
        raise ValueError(f"{path.name}: first_derivative needs a whole window and polyorder and a delta > 0, "
                         f"got {parameters}")
    if window % 2 == 0 or not 3 <= window <= n_bands or not 1 <= polyorder < window:
        raise ValueError(f"{path.name}: first_derivative window {window} / polyorder {polyorder} is not an odd "
                         f"window of at most {n_bands} bands with 1 <= polyorder < window")
    return {'window': window, 'polyorder': polyorder, 'delta': float(delta)}


def _feature_chain(path: Path, manifest: dict) -> tuple:
    """``(wavelengths of every input band, SNV on?, first-derivative parameters ({} = none),
    selected band indices)`` from the manifest."""
    audit = manifest.get('pipeline_audit')
    if not isinstance(audit, dict):
        raise ValueError(f"{path.name}: the manifest has no pipeline_audit (which steps did the training use?)")
    step_list = [step if isinstance(step, dict) else {} for step in audit.get('steps') or []]
    steps = tuple(str(step.get('name')) for step in step_list)
    if steps not in _SUPPORTED_CHAINS:
        raise ValueError(f"{path.name}: preprocessing {list(steps)} is not supported (this reader implements "
                         f"{' or '.join(str(list(chain)) for chain in _SUPPORTED_CHAINS)}); the model cannot "
                         "be run faithfully")
    for key in ('spectral_indices', 'spectral_index_selection', 'spectral_index_discovery'):   # pca: load_package
        if audit.get(key) is not None:
            raise ValueError(f"{path.name}: the package uses {key}, which this reader does not implement")

    all_wavelengths = np.asarray(manifest.get('wavelengths') or [], dtype=np.float64)
    axis = audit.get('spectral_axis_transform') or manifest.get('spectral_axis') or {}
    if axis.get('spectral_resampling_mode', 'none') != 'none' or axis.get('spectral_axis_kind',
                                                                         'wavelength_nm') != 'wavelength_nm':
        raise ValueError(f"{path.name}: spectral axis {axis.get('spectral_axis_kind')!r} with resampling "
                         f"{axis.get('spectral_resampling_mode')!r} is not supported")
    source = axis.get('selected_source_band_indices')
    source = np.arange(all_wavelengths.size) if source is None else np.asarray(source, dtype=np.intp)
    if all_wavelengths.size == 0 or source.size == 0 or source.min() < 0 or source.max() >= all_wavelengths.size:
        raise ValueError(f"{path.name}: the manifest's wavelengths ({all_wavelengths.size}) and band indices "
                         "do not fit together")
    wavelengths = all_wavelengths[source]
    if not np.isfinite(wavelengths).all() or np.any(np.diff(wavelengths) <= 0):
        raise ValueError(f"{path.name}: the wavelengths are not increasing")

    selection = audit.get('wavelength_selection')
    if selection is None:
        selected = np.arange(wavelengths.size, dtype=np.intp)
    else:
        selected = np.asarray(selection.get('selected_band_indices') or [], dtype=np.intp)
        if selected.size == 0 or selected.min() < 0 or selected.max() >= wavelengths.size \
                or np.unique(selected).size != selected.size:
            raise ValueError(f"{path.name}: wavelength_selection.selected_band_indices is not a set of bands "
                             f"of the {wavelengths.size}-band axis")
        stated = selection.get('selected_wavelengths_nm')
        if stated is not None and not np.allclose(wavelengths[selected], np.asarray(stated, dtype=np.float64),
                                                  atol=1e-6):
            raise ValueError(f"{path.name}: the selected band indices and selected_wavelengths_nm disagree")
    derivative = {}
    if 'first_derivative' in steps:
        derivative = _derivative_parameters(path, step_list[steps.index('first_derivative')], wavelengths.size)
    return wavelengths, 'snv' in steps, derivative, selected


def _scaler(path: Path, manifest: dict, files: dict, n_features: int) -> tuple:
    """``(mean, scale, variance or None, samples or None)`` of the StandardScaler (the last two
    only for the check against the training reference)."""
    final = (manifest.get('pipeline_audit') or {}).get('final_scaler')
    if final is None:
        return np.zeros(n_features), np.ones(n_features), None, None
    if final.get('type') != 'StandardScaler':
        raise ValueError(f"{path.name}: final scaler {final.get('type')!r} is not supported (StandardScaler is)")
    stated = final.get('input_features')
    if stated is not None and stated != n_features:       # e.g. a scaler before the PCA, not after it
        raise ValueError(f"{path.name}: the final scaler takes {stated} features, the package's chain gives "
                         f"{n_features} at that point")
    if 'scaler.joblib' not in files:
        raise ValueError(f"{path.name}: the manifest names a StandardScaler but scaler.joblib is missing")
    what = f"{path.name}/scaler.joblib"
    holder = _load_joblib(files['scaler.joblib'], what)
    with_mean = bool(getattr(holder, 'with_mean', final.get('with_mean', True)))
    with_std = bool(getattr(holder, 'with_std', final.get('with_std', True)))
    mean = _array(holder, 'mean_', what, (n_features,)) if with_mean else np.zeros(n_features)
    scale = _array(holder, 'scale_', what, (n_features,)) if with_std else np.ones(n_features)
    if np.any(scale <= 0):
        raise ValueError(f"{what}: scale_ has values <= 0")
    variance = samples = None
    if with_mean and with_std and getattr(holder, 'var_', None) is not None:
        variance = _array(holder, 'var_', what, (n_features,))
        seen = np.asarray(getattr(holder, 'n_samples_seen_', np.nan), dtype=np.float64)
        samples = float(seen) if seen.ndim == 0 and np.isfinite(seen) and seen > 0 else None
    return mean, scale, variance, samples


def _mlc(path: Path, files: dict, n_features: int) -> tuple:
    if 'model.joblib' not in files:
        raise ValueError(f"{path.name}: model.joblib is missing")
    what = f"{path.name}/model.joblib"
    holder = _load_joblib(files['model.joblib'], what)
    kind = type(holder).__name__
    if kind != 'MaximumLikelihoodClassifier' or not type(holder).__module__.startswith('hsi_lab_core'):
        raise ValueError(f"{what}: holds a {kind}; this reader runs MaximumLikelihoodClassifier")
    covariance_type = getattr(holder, 'covariance_type_', getattr(holder, 'covariance_type', None))
    if covariance_type != 'full':
        raise ValueError(f"{what}: covariance_type {covariance_type!r} is not supported ('full' is)")
    classes = np.asarray(getattr(holder, 'classes_', []))
    if classes.ndim != 1 or classes.size < 2 or classes.dtype.kind not in 'iu':
        raise ValueError(f"{what}: classes_ {classes!r} is not a list of at least two integer class ids")
    n_classes = classes.size
    means = _array(holder, 'means_', what, (n_classes, n_features))
    covariances = _array(holder, 'covariances_', what, (n_classes, n_features, n_features))
    priors = _array(holder, 'priors_', what, (n_classes,))
    if np.any(priors <= 0) or abs(priors.sum() - 1.0) > 1e-6:
        raise ValueError(f"{what}: priors_ {priors.tolist()} are not positive probabilities summing to 1")
    if not np.allclose(covariances, covariances.transpose(0, 2, 1), rtol=1e-9, atol=1e-12):
        raise ValueError(f"{what}: a covariance matrix is not symmetric")
    stored = getattr(holder, 'log_determinants_', None)
    if stored is not None:                 # the stored covariances are the ones the training evaluates
        ours = np.array([np.linalg.slogdet(c)[1] for c in covariances])
        if not np.allclose(ours, np.asarray(stored, dtype=np.float64), rtol=1e-8, atol=1e-6):
            raise ValueError(f"{what}: log_determinants_ do not belong to covariances_")
    return classes.astype(np.int64), means, covariances, priors


def _svm_array(holder, names: tuple, what: str, shape: tuple) -> np.ndarray:
    for name in names:                       # sklearn renamed some attributes across versions
        if getattr(holder, name, None) is not None:
            return _array(holder, name, what, shape)
    raise ValueError(f"{what}: no {' / '.join(names)}")


def _svm(path: Path, files: dict, n_features: int) -> tuple:
    """``(classes, SvmModel)`` of a pickled ``sklearn.svm.SVC`` (read as attributes, never run)."""
    if 'model.joblib' not in files:
        raise ValueError(f"{path.name}: model.joblib is missing")
    what = f"{path.name}/model.joblib"
    holder = _load_joblib(files['model.joblib'], what)
    kind = f"{type(holder).__module__}.{type(holder).__name__}"
    if (type(holder).__module__, type(holder).__name__) not in _SVM_HOLDERS:
        raise ValueError(f"{what}: holds a {kind}; an SVM package must hold sklearn.svm.SVC")
    settings = {name: getattr(holder, name, None) for name in ('kernel', '_sparse', 'break_ties',
                                                               'decision_function_shape', 'fit_status_')}
    if settings['kernel'] != 'rbf':
        raise ValueError(f"{what}: SVC kernel {settings['kernel']!r} is not supported ('rbf' is)")
    if settings['_sparse'] is True:
        raise ValueError(f"{what}: an SVC fitted on sparse data is not supported")
    if settings['break_ties'] is True and settings['decision_function_shape'] == 'ovr':
        raise ValueError(f"{what}: break_ties=True (predict by the one-vs-rest decision) is not supported")
    if settings['fit_status_'] not in (None, 0):
        raise ValueError(f"{what}: the SVC did not converge (fit_status_ {settings['fit_status_']})")
    classes = np.asarray(getattr(holder, 'classes_', []))
    if classes.ndim != 1 or classes.size < 2 or classes.dtype.kind not in 'iu':
        raise ValueError(f"{what}: classes_ {classes!r} is not a list of at least two integer class ids")
    n_classes = classes.size
    n_pairs = n_classes * (n_classes - 1) // 2
    n_in = getattr(holder, 'n_features_in_', n_features)
    if n_in != n_features:
        raise ValueError(f"{what}: the SVC takes {n_in} features, the package's chain gives {n_features}")
    support_vectors = _array(holder, 'support_vectors_', what)
    if support_vectors.ndim != 2 or support_vectors.shape[1] != n_features:
        raise ValueError(f"{what}: support_vectors_ has shape {support_vectors.shape}, expected (n, {n_features})")
    n_vectors = support_vectors.shape[0]
    raw = getattr(holder, '_n_support', getattr(holder, 'n_support_', None))
    n_support = np.asarray(raw if raw is not None else [])
    if n_support.shape != (n_classes,) or n_support.dtype.kind not in 'iu' or np.any(n_support < 0) \
            or int(n_support.sum()) != n_vectors or n_vectors < 1:
        raise ValueError(f"{what}: support vectors per class {n_support.tolist()} do not add up to the "
                         f"{n_vectors} support vectors of {n_classes} classes")
    # libsvm's own coefficients: sklearn flips the public dual_coef_ / intercept_ of a binary SVC
    if getattr(holder, '_dual_coef_', None) is not None and getattr(holder, '_intercept_', None) is not None:
        dual_coef = _array(holder, '_dual_coef_', what, (n_classes - 1, n_vectors))
        intercept = _array(holder, '_intercept_', what, (n_pairs,))
    else:
        sign = -1.0 if n_classes == 2 else 1.0
        dual_coef = sign * _array(holder, 'dual_coef_', what, (n_classes - 1, n_vectors))
        intercept = sign * _array(holder, 'intercept_', what, (n_pairs,))
    gamma = float(_array(holder, '_gamma', what, ()))
    if not gamma > 0:
        raise ValueError(f"{what}: gamma {gamma} is not positive")
    prob_a = prob_b = None
    if getattr(holder, 'probability', False):
        prob_a = _svm_array(holder, ('_probA', 'probA_'), what, (n_pairs,))
        prob_b = _svm_array(holder, ('_probB', 'probB_'), what, (n_pairs,))
    model = SvmModel(support_vectors=support_vectors, n_support=n_support.astype(np.int64), dual_coef=dual_coef,
                     intercept=intercept, gamma=gamma, gamma_setting=getattr(holder, 'gamma', None),
                     prob_a=prob_a, prob_b=prob_b)
    return classes.astype(np.int64), model


def _pca(path: Path, manifest: dict, files: dict, n_features: int) -> tuple:
    """``(mean (F,), components (P, F))`` of the package's PCA (read as attributes, never run)."""
    spec = manifest['pipeline_audit']['pca']
    if not isinstance(spec, dict) or spec.get('type', 'PCA') != 'PCA':
        raise ValueError(f"{path.name}: pca {spec!r} is not supported (a PCA is)")
    if spec.get('whiten'):
        raise ValueError(f"{path.name}: a whitened pca is not supported")
    stated = spec.get('input_features')
    if stated is not None and stated != n_features:
        raise ValueError(f"{path.name}: the pca takes {stated} features, the package's chain gives {n_features}")
    if 'pca.joblib' not in files:
        raise ValueError(f"{path.name}: the manifest names a pca but pca.joblib is missing")
    what = f"{path.name}/pca.joblib"
    holder = _load_joblib(files['pca.joblib'], what)
    kind = (type(holder).__module__, type(holder).__name__)
    if kind not in _PCA_HOLDERS and not (kind[0].startswith('hsi_lab_core') and kind[1] in ('SafePCA', 'PCA')):
        raise ValueError(f"{what}: holds a {'.'.join(kind)}, not a PCA")
    if getattr(holder, 'whiten', False):
        raise ValueError(f"{what}: a whitened pca is not supported")
    components = _array(holder, 'components_', what)
    if components.ndim != 2 or components.shape[1] != n_features or not 1 <= components.shape[0] <= n_features:
        raise ValueError(f"{what}: components_ has shape {components.shape}, expected (n <= {n_features}, "
                         f"{n_features})")
    wanted = spec.get('n_components')
    if wanted is not None and wanted != components.shape[0]:
        raise ValueError(f"{what}: {components.shape[0]} components, the manifest says {wanted}")
    variance = samples = None                       # only for the check against the training reference
    if getattr(holder, 'explained_variance_', None) is not None:
        variance = _array(holder, 'explained_variance_', what, (components.shape[0],))
        seen = np.asarray(getattr(holder, 'n_samples_', np.nan), dtype=np.float64)
        samples = float(seen) if seen.ndim == 0 and np.isfinite(seen) and seen > 1 else None
    return _array(holder, 'mean_', what, (n_features,)), components, variance, samples


def _sgd(path: Path, files: dict, n_features: int) -> tuple:
    """``(classes, LinearModel)`` of a pickled ``sklearn.linear_model.SGDClassifier`` (never run)."""
    if 'model.joblib' not in files:
        raise ValueError(f"{path.name}: model.joblib is missing")
    what = f"{path.name}/model.joblib"
    holder = _load_joblib(files['model.joblib'], what)
    kind = (type(holder).__module__, type(holder).__name__)
    if kind not in _SGD_HOLDERS:
        raise ValueError(f"{what}: holds a {'.'.join(kind)}; an SGD package must hold "
                         "sklearn.linear_model.SGDClassifier")
    loss = getattr(holder, 'loss', None)
    if loss not in _SGD_LOSSES:
        raise ValueError(f"{what}: loss {loss!r} is not a classification loss of SGDClassifier")
    classes = np.asarray(getattr(holder, 'classes_', []))
    if classes.ndim != 1 or classes.size < 2 or classes.dtype.kind not in 'iu':
        raise ValueError(f"{what}: classes_ {classes!r} is not a list of at least two integer class ids")
    n_in = getattr(holder, 'n_features_in_', n_features)
    if n_in != n_features:
        raise ValueError(f"{what}: the SGDClassifier takes {n_in} features, the package's chain gives {n_features}")
    rows = 1 if classes.size == 2 else classes.size
    coef = _array(holder, 'coef_', what, (rows, n_features))
    intercept = _array(holder, 'intercept_', what, (rows,))
    return classes.astype(np.int64), LinearModel(coef=coef, intercept=intercept, loss=str(loss))


def _plsda(path: Path, files: dict, n_features: int) -> tuple:
    """``(classes, LinearModel)`` of ``hsi_lab_core``'s ``PLSDAClassifier`` around a pickled
    ``sklearn.cross_decomposition.PLSRegression`` (read as attributes, never run).

    ``PLSRegression.predict`` is ``(f - x_mean) @ coef.T + intercept`` (sklearn >= 1.1: ``coef_``
    already divided by the X standard deviation); the class is the largest response. Checked:
    ``coef_`` is the one the regression's rotations and loadings give, and the training scores
    it stores give back responses that are one-hot (one column per class, in class order).
    """
    if 'model.joblib' not in files:
        raise ValueError(f"{path.name}: model.joblib is missing")
    what = f"{path.name}/model.joblib"
    holder = _load_joblib(files['model.joblib'], what)
    kind = (type(holder).__module__, type(holder).__name__)
    if kind[1] != 'PLSDAClassifier' or not kind[0].startswith('hsi_lab_core'):
        raise ValueError(f"{what}: holds a {'.'.join(kind)}; a PLS-DA package must hold PLSDAClassifier")
    pls = getattr(holder, 'model_', None)
    if pls is None or (type(pls).__module__, type(pls).__name__) not in _PLS_HOLDERS:
        raise ValueError(f"{what}: the PLSDAClassifier holds no sklearn PLSRegression (model_)")
    classes = np.asarray(getattr(holder, 'classes_', []))
    if classes.ndim != 1 or classes.size < 2 or classes.dtype.kind not in 'iu':
        raise ValueError(f"{what}: classes_ {classes!r} is not a list of at least two integer class ids")
    k = classes.size
    n_in = getattr(pls, 'n_features_in_', n_features)
    if n_in != n_features:
        raise ValueError(f"{what}: the PLSRegression takes {n_in} features, the package's chain gives {n_features}")
    if getattr(pls, '_predict_1d', False):
        raise ValueError(f"{what}: a PLSRegression with one response is not a PLS-DA of {k} classes")
    coef = _array(pls, 'coef_', what, (k, n_features))
    intercept = _array(pls, 'intercept_', what, (k,))
    x_mean = _array(pls, '_x_mean', what, (n_features,))
    x_std = _array(pls, '_x_std', what, (n_features,))
    y_mean = _array(pls, '_y_mean', what, (k,))
    y_std = _array(pls, '_y_std', what, (k,))
    rotations = _array(pls, 'x_rotations_', what)
    if rotations.ndim != 2 or rotations.shape[0] != n_features or rotations.shape[1] < 1:
        raise ValueError(f"{what}: x_rotations_ has shape {rotations.shape}, expected ({n_features}, n)")
    components = rotations.shape[1]
    loadings = _array(pls, 'y_loadings_', what, (k, components))
    if np.any(x_std <= 0) or np.any(y_std <= 0):
        raise ValueError(f"{what}: a standard deviation of the PLS regression is <= 0")
    if not np.allclose(intercept, y_mean, rtol=0, atol=1e-12):
        raise ValueError(f"{what}: intercept_ is not the response mean (a PLSRegression of another sklearn version)")
    expected = (rotations @ loadings.T * y_std).T / x_std
    error = float(np.abs(coef - expected).max() / max(np.abs(expected).max(), 1e-300))
    if not error <= _PLS_TOLERANCE:
        raise ValueError(f"{what}: coef_ is not the one its rotations and loadings give (relative difference "
                         f"{error:.3g}): not the PLSRegression.predict implemented here")
    counts = accuracy = None
    x_name = next((n for n in ('_x_scores', 'x_scores_') if getattr(pls, n, None) is not None), None)
    y_name = next((n for n in ('_y_scores', 'y_scores_') if getattr(pls, n, None) is not None), None)
    if x_name and y_name and getattr(pls, 'y_weights_', None) is not None \
            and getattr(pls, 'deflation_mode', 'regression') == 'regression' \
            and not getattr(pls, '_norm_y_weights', False):
        x_scores, y_scores = _array(pls, x_name, what), _array(pls, y_name, what)
        weights = _array(pls, 'y_weights_', what, (k, components))
        if x_scores.shape != y_scores.shape or x_scores.ndim != 2 or x_scores.shape[1] != components:
            raise ValueError(f"{what}: the stored training scores {x_scores.shape} / {y_scores.shape} do not fit "
                             f"{components} components")
        # NIPALS, regression deflation: Y_i = Y_0 - sum_{j<i} t_j q_j^T and u_i = Y_i c_i / |c_i|^2, so
        # Y_0 c_i = u_i |c_i|^2 + sum_{j<i} t_j (q_j . c_i). Every c_i lies in the row space of Y_0:
        # the least-norm solution gives Y_0 back whenever the c_i span it.
        projected = np.empty_like(y_scores)
        for i in range(components):
            c = weights[:, i]
            projected[:, i] = y_scores[:, i] * (c @ c) + x_scores[:, :i] @ (loadings[:, :i].T @ c)
        responses = np.linalg.lstsq(weights.T, projected.T, rcond=None)[0].T * y_std + y_mean     # (n, K)
        hot = np.rint(responses)
        if np.abs(responses - hot).max() <= _PLS_TOLERANCE and np.all((hot == 0) | (hot == 1)) \
                and np.all(hot.sum(axis=1) == 1):
            truth = hot.argmax(axis=1)
            counts = np.bincount(truth, minlength=k).astype(np.float64)
            fitted = x_scores @ (loadings.T * y_std) + y_mean       # PLSRegression.predict of the training rows
            accuracy = float(np.mean(fitted.argmax(axis=1) == truth))
        elif np.linalg.matrix_rank(weights) >= k - 1:               # the c_i span every one-hot coding
            raise ValueError(f"{what}: the responses the PLS regression was fitted on are not one-hot per class: "
                             "not the PLS-DA implemented here (largest response = class)")
    return classes.astype(np.int64), LinearModel(
        coef=coef, intercept=intercept - x_mean @ coef.T, loss=PLS_RESPONSE, components=components,
        fit_counts=counts, fit_accuracy=accuracy)


def _hex_rgb(text) -> list | None:
    if not isinstance(text, str):
        return None
    digits = text.strip().lstrip('#')
    if len(digits) == 3:
        digits = ''.join(ch * 2 for ch in digits)
    if len(digits) != 6:
        return None
    try:
        return [int(digits[k:k + 2], 16) for k in (0, 2, 4)]
    except ValueError:
        return None


def _metric_text(manifest: dict) -> str:
    evidence = manifest.get('evaluation_evidence') or {}
    winner = (manifest.get('selection_manifest') or {}).get('winner') or {}
    name = evidence.get('selection_metric') or winner.get('selection_metric') or manifest.get('metric_name')
    value = evidence.get('selection_score', winner.get('selection_score', manifest.get('metric_value')))
    text = f"{name} = {value:.4f}" if isinstance(value, (int, float)) else 'not reported'
    if isinstance(evidence.get('test_score'), (int, float)):
        text += f"; locked test {evidence.get('test_metric_name')} = {evidence['test_score']:.4f}"
    else:
        text += '; no locked test'
    return text


# ------------------------------------------------------------------ training reference ---

def _check_reference(path: Path, package: HsiPackage, blob: bytes | None) -> str:
    """Compare the model with the package's own preprocessed training spectra (see module docstring)."""
    if blob is None:
        logger.warning("%s has no training_spectral_reference.json: the processing chain is not verified",
                       path.name)
        return "training reference: not in the package (chain NOT verified)"
    try:
        domain = json.loads(blob.decode('utf-8'))['domains'][_REFERENCE_DOMAIN]
        rows = {int(c['class_id']): (np.asarray(c['mean'], dtype=np.float64), np.asarray(c['sd'], dtype=np.float64))
                for c in domain['classes']}
        counts = {int(c['class_id']): c.get('n_spectra') for c in domain['classes']}
    except (KeyError, TypeError, ValueError, UnicodeDecodeError, AttributeError) as exc:
        logger.warning("%s: training_spectral_reference.json has no %s spectra (%s): chain not verified",
                       path.name, _REFERENCE_DOMAIN, type(exc).__name__)
        return f"training reference: no {_REFERENCE_DOMAIN} spectra (chain NOT verified)"
    n_bands = package.wavelengths.size
    derived = ''
    if package.derivative is not None:
        derived = _check_derivative(path, package, blob, rows)
    worst = 0.0
    for k, cid in enumerate(package.classes):
        if int(cid) not in rows or rows[int(cid)][0].shape != (n_bands,):
            raise ValueError(f"{path.name}: the training reference has no {n_bands}-band spectrum of class {cid}")
        mean, sd = rows[int(cid)]
        if package.snv:
            # SNV with the population SD over all bands: every spectrum has mean 0 and sum of squares
            # B, so the class mean sums to 0 and sum(mean^2 + sd^2) is B (B - 1 with the sample SD).
            total = float(np.sum(mean ** 2 + sd ** 2))
            if abs(mean.sum()) > 1e-6 * n_bands or abs(total - n_bands) > 0.005 * n_bands:
                raise ValueError(
                    f"{path.name}: class {cid} of the training reference is not SNV over {n_bands} bands with "
                    f"the population SD (sum of mean {mean.sum():.3g}, sum of squares {total:.2f}): the "
                    "package's preprocessing is not the one this reader implements")
        if package.means is not None:
            features = mean[package.selected]
            if package.pca_components is not None:
                features = (features - package.pca_mean) @ package.pca_components.T
            scaled = (features - package.scaler_mean) / package.scaler_scale
            worst = max(worst, float(np.abs(scaled - package.means[k]).max()))
    if worst > _MEAN_TOLERANCE:
        raise ValueError(
            f"{path.name}: the model's class means differ from the package's training reference by {worst:.3g} "
            "after this reader's band selection and scaler: the package's chain is not the one implemented here")
    scaler = _check_scaler(path, package, rows, counts)
    text = "training reference ok: " + (f"class means reproduced to {worst:.1e}, " if package.means is not None else "")
    return (text + scaler + (", SNV = population SD over all bands" if package.snv else "") + derived
            + _check_pls_counts(path, package, counts))


def _check_pls_counts(path: Path, package: HsiPackage, counts: dict) -> str:
    """A PLS-DA's one-hot responses must count the reference's spectra of every class."""
    linear = package.linear
    if linear is None or linear.loss != PLS_RESPONSE:
        return ''
    if linear.fit_counts is None:
        return ", PLS-DA responses NOT verified (the regression stores no training scores)"
    try:
        n = np.array([float(counts[int(cid)]) for cid in package.classes])
    except (KeyError, TypeError, ValueError):
        return ", PLS-DA responses one-hot (class counts NOT verified: the reference has none)"
    if not np.array_equal(n, linear.fit_counts):
        raise ValueError(f"{path.name}: the PLS-DA was fitted on {linear.fit_counts.astype(int).tolist()} rows per "
                         f"class, the training reference counts {n.astype(int).tolist()}: its responses are not the "
                         "classes in the package's order")
    return (f", PLS-DA responses one-hot with the reference's class counts, largest response right on "
            f"{linear.fit_accuracy:.4f} of its training rows")


def _check_scaler(path: Path, package: HsiPackage, rows: dict, counts: dict) -> str:
    """The scaler's mean and variance, and an SVM's gamma, from the reference's per-class count,
    mean and SD (sample SD) of the preprocessed training spectra.

    The reference covers the rows the final model was fitted on: the pooled mean and variance of
    those rows (on the selected bands) are what the StandardScaler stored. A gamma of 'scale' is
    ``1 / (features x variance of all scaled training values)`` (sklearn), which follows from them.
    """
    if package.pca_components is not None:
        return _check_pca(path, package, rows, counts)
    unverified = "scaler NOT verified" + (" (nor the SVM)" if package.svm is not None else "")
    if package.scaler_var is None or package.scaler_samples is None:
        return f"{unverified}: the scaler stores no variance / sample count"
    try:
        n = np.array([float(counts[int(cid)]) for cid in package.classes])
    except (KeyError, TypeError, ValueError):
        return f"{unverified}: the reference has no spectrum count per class"
    if not np.all(np.isfinite(n)) or np.any(n < 2) or abs(n.sum() - package.scaler_samples) > 0.5:
        return (f"{unverified}: the reference counts {n.sum():g} spectra, the scaler was fitted on "
                f"{package.scaler_samples:g}")
    means = np.array([rows[int(cid)][0][package.selected] for cid in package.classes])
    sds = np.array([rows[int(cid)][1][package.selected] for cid in package.classes])
    if sds.shape != means.shape or not np.isfinite(sds).all():
        return f"{unverified}: the reference has no SD per band"
    total = n.sum()
    pooled = (n[:, None] * means).sum(axis=0) / total
    variance = ((n[:, None] - 1.0) * sds ** 2 + n[:, None] * (means - pooled) ** 2).sum(axis=0) / total
    mean_error = float(np.abs(pooled - package.scaler_mean).max() / max(np.abs(package.scaler_mean).max(),
                                                                      package.scaler_scale.min(), 1e-300))
    var_error = float(np.abs(variance / np.maximum(package.scaler_var, 1e-300) - 1.0).max())
    if not (mean_error <= _SCALER_TOLERANCE and var_error <= _SCALER_TOLERANCE):
        raise ValueError(
            f"{path.name}: the scaler's mean / variance differ from those of the package's training reference by "
            f"{mean_error:.3g} / {var_error:.3g} (relative): the package's chain is not the one implemented here")
    text = f"scaler mean / variance reproduced to {max(mean_error, var_error):.1e}"
    svm = package.svm
    if svm is None:
        return text
    n_features = package.selected.size
    if svm.gamma_setting == 'scale':
        shift = (pooled - package.scaler_mean) / package.scaler_scale
        second = variance / package.scaler_scale ** 2 + shift ** 2
        expected = 1.0 / (n_features * (second.mean() - shift.mean() ** 2))
    elif svm.gamma_setting == 'auto':
        expected = 1.0 / n_features
    else:
        try:
            expected = float(svm.gamma_setting)
        except (TypeError, ValueError):
            return text + f", SVM gamma {svm.gamma:g} NOT verified (setting {svm.gamma_setting!r})"
    error = abs(svm.gamma / expected - 1.0)
    if not error <= _GAMMA_TOLERANCE:
        raise ValueError(f"{path.name}: the SVM's gamma {svm.gamma:.10g} is not the {svm.gamma_setting!r} gamma "
                         f"{expected:.10g} of the package's training reference: its features are not the ones "
                         "implemented here")
    return text + f", SVM gamma {svm.gamma_setting!r} = {svm.gamma:.6g} reproduced to {error:.1e}"


def _check_pca(path: Path, package: HsiPackage, rows: dict, counts: dict) -> str:
    """The PCA's mean, and the scaler's mean of the PCA scores, from the reference's per-class
    count and mean (both were fitted on the rows the reference covers).

    The variance of the scores is not in the reference (it would need the spectra's covariances),
    but the PCA stores it: ``explained_variance_`` is their sample variance, so the scaler fitted
    on the same rows holds it times ``(n - 1) / n``.
    """
    unverified = "PCA and scaler NOT verified"
    try:
        n = np.array([float(counts[int(cid)]) for cid in package.classes])
    except (KeyError, TypeError, ValueError):
        return f"{unverified}: the reference has no spectrum count per class"
    if not np.all(np.isfinite(n)) or np.any(n < 1):
        return f"{unverified}: the reference has no spectrum count per class"
    means = np.array([rows[int(cid)][0][package.selected] for cid in package.classes])
    pooled = (n[:, None] * means).sum(axis=0) / n.sum()
    pca_error = float(np.abs(pooled - package.pca_mean).max() / max(np.abs(package.pca_mean).max(), 1e-300))
    projected = (pooled - package.pca_mean) @ package.pca_components.T
    scaler_error = float(np.abs(projected - package.scaler_mean).max()
                         / max(np.abs(package.scaler_mean).max(), package.scaler_scale.min(), 1e-300))
    if not (pca_error <= _SCALER_TOLERANCE and scaler_error <= _SCALER_TOLERANCE):
        raise ValueError(
            f"{path.name}: the PCA mean / the scaler mean of the PCA scores differ from those of the package's "
            f"training reference by {pca_error:.3g} / {scaler_error:.3g} (relative): the package's chain is not the "
            "one implemented here")
    text = f"PCA mean and scaler mean reproduced to {max(pca_error, scaler_error):.1e}"
    m = package.pca_samples
    if package.pca_variance is None or m is None or package.scaler_var is None:
        return text + ", scaler variance NOT verified (the PCA or the scaler stores no variance / sample count)"
    if abs(m - n.sum()) > 0.5 or (package.scaler_samples is not None and abs(package.scaler_samples - m) > 0.5):
        return (text + f", scaler variance NOT verified (PCA fitted on {m:g} rows, scaler on "
                       f"{package.scaler_samples}, reference {n.sum():g})")
    var_error = float(np.abs(package.scaler_var / (package.pca_variance * (m - 1.0) / m) - 1.0).max())
    if not var_error <= _SCALER_TOLERANCE:
        raise ValueError(f"{path.name}: the scaler's variance is not the PCA scores' variance (relative difference "
                         f"{var_error:.3g}): the scaler does not follow the PCA as implemented here")
    return text + f", scaler variance = PCA variance to {var_error:.1e}"


def _check_derivative(path: Path, package: HsiPackage, blob: bytes, preprocessed: dict) -> str:
    """The first derivative of the raw training class means must be the preprocessed class means.

    The derivative is linear, so the mean of the derived spectra is the derivative of their mean:
    this checks the Savitzky-Golay window, order, spacing and end handling against the package.
    """
    try:
        domain = json.loads(blob.decode('utf-8'))['domains'][_RAW_DOMAIN]
        raw = {int(c['class_id']): np.asarray(c['mean'], dtype=np.float64) for c in domain['classes']}
    except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
        logger.warning("%s: training_spectral_reference.json has no %s spectra (%s): the first derivative is "
                       "not verified", path.name, _RAW_DOMAIN, type(exc).__name__)
        return f"; first derivative NOT verified (no {_RAW_DOMAIN} spectra)"
    n_bands = package.wavelengths.size
    worst = 0.0
    for cid in package.classes:
        mean = raw.get(int(cid))
        if mean is None or mean.shape != (n_bands,) or not np.isfinite(mean).all():
            raise ValueError(f"{path.name}: the training reference has no raw {n_bands}-band spectrum of class {cid}")
        if int(cid) not in preprocessed or preprocessed[int(cid)][0].shape != (n_bands,):
            raise ValueError(f"{path.name}: the training reference has no {n_bands}-band spectrum of class {cid}")
        target = preprocessed[int(cid)][0]
        size = float(np.abs(target).max())
        error = float(np.abs(package.derivative @ mean - target).max())
        if not error <= _DERIVATIVE_TOLERANCE * max(size, 1e-300):
            p = package.derivative_parameters
            raise ValueError(
                f"{path.name}: the Savitzky-Golay first derivative (window {p['window']}, order {p['polyorder']}, "
                f"delta {p['delta']:g} nm) of the raw training means differs from the preprocessed ones by "
                f"{error:.3g} (largest value {size:.3g}): the package's derivative is not the one implemented here")
        worst = max(worst, error / max(size, 1e-300))
    return f", 1st derivative of the raw means reproduced to {worst:.1e} (relative)"


_TRAINING_DOMAINS = {'input': 'raw_wavelength_analysis', 'preprocessed': _REFERENCE_DOMAIN}


def _training_spectra(package: HsiPackage, blob: bytes | None) -> dict:
    """Per-class mean and SD of the training spectra, for display only (never raises)."""
    out = {key: [] for key in _TRAINING_DOMAINS}
    if blob is None:
        return out
    n_bands = package.wavelengths.size
    for key, name in _TRAINING_DOMAINS.items():
        try:
            rows = []
            for c in json.loads(blob.decode('utf-8'))['domains'][name]['classes']:
                mean, sd = np.asarray(c['mean'], np.float64), np.asarray(c['sd'], np.float64)
                if mean.shape != (n_bands,) or sd.shape != (n_bands,):
                    raise ValueError('band count')
                cid = int(c['class_id'])
                rows.append(TrainingClass(cid, package.labels.get(cid, str(c.get('label', cid))),
                                          int(c.get('n_spectra') or 0), mean, sd))
            out[key] = rows
        except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
            logger.info("%s: no %s training spectra to display (%s)", Path(package.path).name, name,
                        type(exc).__name__)
    return out


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Show and verify an HSI MLOps model package (.hsipkg / .zip).")
    parser.add_argument('package', nargs='+')
    args = parser.parse_args(argv)
    status = 0
    for name in args.package:
        try:
            print(load_package(name).describe())
        except (ValueError, OSError) as exc:
            print(f"{name}: REFUSED - {exc}")
            status = 1
    return status


if __name__ == '__main__':
    raise SystemExit(main())
