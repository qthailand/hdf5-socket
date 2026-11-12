# -*- coding: utf-8 -*-
"""Read a trained-model package of HSI MLOps (``.hsipkg``) without ``hsi_lab_core``.

The models of this project are trained elsewhere (HSI MLOps, ``hsi_lab_core``) and arrive as a
package: a zip named ``*.hsipkg`` (often inside another ``.zip``) holding ``manifest.json``,
``model.joblib``, ``scaler.joblib``, ``pipeline.joblib`` and
``training_spectral_reference.json``. :func:`load_package` turns one into plain numpy arrays, so
``inference_engine.HsiPackageClassifier`` can run the same chain as the training software:

    reflectance (159 bands) -> SNV per pixel -> band selection -> StandardScaler -> MLC

What is read, and how it is kept honest:

* ``manifest.json`` says which steps the package uses. Anything this reader does not implement
  (another preprocessing step, PCA, spectral indices, resampling, another algorithm) is refused
  by name; nothing is skipped silently.
* The joblib files are pickles of ``hsi_lab_core`` objects. They are read with a restricted
  unpickler: numpy arrays are loaded, ``hsi_lab_core`` classes become plain attribute holders,
  every other global is refused. No code of the package runs and ``hsi_lab_core`` is not needed.
* ``artifact_sha256`` of the manifest is checked for every file that is read.
* ``training_spectral_reference.json`` (class means of the training spectra after the
  preprocessing) is the package's own evidence: the class means of the model must equal those
  spectra passed through this reader's band selection and scaler (they do, to 1e-13, for the
  packages this was written against), and the SNV convention (population SD over all bands) is
  checked on them. A package that fails this is refused: its chain is not the one implemented here.

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

SUPPORTED_ALGORITHMS = ('MLC',)
SUPPORTED_STEPS = ('snv',)            # preprocessing steps of pipeline_audit this reader implements
_REFERENCE_DOMAIN = 'preprocessed_wavelength_selected'
_MEAN_TOLERANCE = 1e-6                # model class means vs the package's reference spectra (scaled units)


# ------------------------------------------------------------------------------ package ---

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
    means: np.ndarray                # (K, F) in scaled units
    covariances: np.ndarray          # (K, F, F), regularisation included
    priors: np.ndarray               # (K,)
    input_units: str                 # 'reflectance' | 'counts'
    metric: str                      # the selection score the package reports
    exported_at: str
    lab_core_version: str
    checks: list = field(default_factory=list)   # what was verified while loading (text lines)

    @property
    def chain(self) -> str:
        steps = [f"{self.input_units} {self.wavelengths.size} bands"]
        if self.snv:
            steps.append('SNV')
        if self.selected.size != self.wavelengths.size:
            steps.append(f"{self.selected.size} bands")
        steps += ['StandardScaler', f"{self.algorithm} {self.classes.size} classes"]
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
    checks = [_check_sha256(path, manifest, files, ('model.joblib', 'scaler.joblib',
                                                    'training_spectral_reference.json'))]

    algorithm = str(manifest.get('algorithm'))
    if algorithm not in SUPPORTED_ALGORITHMS or manifest.get('task_type') != 'classification':
        raise ValueError(f"{path.name}: algorithm {algorithm!r} ({manifest.get('task_type')}) is not supported; "
                         f"this reader runs {', '.join(SUPPORTED_ALGORITHMS)} classification packages")
    wavelengths, snv, selected = _feature_chain(path, manifest)
    scaler_mean, scaler_scale = _scaler(path, manifest, files, selected.size)
    classes, means, covariances, priors = _mlc(path, files, selected.size)

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
        lab_core_version=str(manifest.get('hsi_lab_core_version', '?')), checks=checks)
    checks.append(_check_reference(path, package, files.get('training_spectral_reference.json')))
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
    wanted = ('manifest.json', 'model.joblib', 'scaler.joblib', 'training_spectral_reference.json')
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
            if module == 'hsi_lab_core' or module.startswith('hsi_lab_core.'):
                return type(name, (_Holder,), {'__module__': module})
            raise pickle.UnpicklingError(f"refused {module}.{name}: a model package may hold numpy arrays and "
                                         "hsi_lab_core objects only")

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

def _feature_chain(path: Path, manifest: dict) -> tuple:
    """``(wavelengths of every input band, SNV on?, selected band indices)`` from the manifest."""
    audit = manifest.get('pipeline_audit')
    if not isinstance(audit, dict):
        raise ValueError(f"{path.name}: the manifest has no pipeline_audit (which steps did the training use?)")
    steps = [str(step.get('name')) for step in audit.get('steps') or []]
    unknown = [step for step in steps if step not in SUPPORTED_STEPS]
    if unknown or len(steps) != len(set(steps)):
        raise ValueError(f"{path.name}: preprocessing {steps} is not supported (this reader implements "
                         f"{', '.join(SUPPORTED_STEPS)}); the model cannot be run faithfully")
    for key in ('pca', 'spectral_indices', 'spectral_index_selection', 'spectral_index_discovery'):
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
    return wavelengths, 'snv' in steps, selected


def _scaler(path: Path, manifest: dict, files: dict, n_features: int) -> tuple:
    final = (manifest.get('pipeline_audit') or {}).get('final_scaler')
    if final is None:
        return np.zeros(n_features), np.ones(n_features)
    if final.get('type') != 'StandardScaler':
        raise ValueError(f"{path.name}: final scaler {final.get('type')!r} is not supported (StandardScaler is)")
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
    return mean, scale


def _mlc(path: Path, files: dict, n_features: int) -> tuple:
    if 'model.joblib' not in files:
        raise ValueError(f"{path.name}: model.joblib is missing")
    what = f"{path.name}/model.joblib"
    holder = _load_joblib(files['model.joblib'], what)
    kind = type(holder).__name__
    if kind != 'MaximumLikelihoodClassifier':
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
    except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
        logger.warning("%s: training_spectral_reference.json has no %s spectra (%s): chain not verified",
                       path.name, _REFERENCE_DOMAIN, type(exc).__name__)
        return f"training reference: no {_REFERENCE_DOMAIN} spectra (chain NOT verified)"
    n_bands = package.wavelengths.size
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
        scaled = (mean[package.selected] - package.scaler_mean) / package.scaler_scale
        worst = max(worst, float(np.abs(scaled - package.means[k]).max()))
    if worst > _MEAN_TOLERANCE:
        raise ValueError(
            f"{path.name}: the model's class means differ from the package's training reference by {worst:.3g} "
            "after this reader's band selection and scaler: the package's chain is not the one implemented here")
    return (f"training reference ok: class means reproduced to {worst:.1e}"
            + (", SNV = population SD over all bands" if package.snv else ""))


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
