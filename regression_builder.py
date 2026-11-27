# -*- coding: utf-8 -*-
"""Build a regression ``.hsipkg`` (e.g. moisture by PLSR) from labelled spectra.

The classification packages of this project come from HSI MLOps (``hsi_lab_core``); this module
writes a regression package in the same layout, so ``hsi_package.load_package`` reads, verifies
and runs it like those (see ``hsi_package._load_regression`` for the manifest fields):

    spectra (n, B) + target (n,) -> [SNV | 1st derivative] -> [band selection] -> StandardScaler
                                 -> PLSRegression (components by cross-validation) | Ridge

``model.joblib`` holds the sklearn regression, ``scaler.joblib`` the sklearn ``StandardScaler``
(read back as attributes only, nothing of them runs at load time);
``training_spectral_reference.json`` the mean spectra of three target bins (display, scaler and
derivative checks), the fit statistics and a few training spectra with the predictions sklearn
made of them (the reader must reproduce those).

    python regression_builder.py samples.npz -o models/moisture_plsr.hsipkg --target moisture --units %
    python regression_builder.py samples.csv --preprocessing first_derivative --valid-range 0 80

``samples.npz``: ``spectra`` (n, B), ``target`` (n,), ``wavelengths`` (B,). ``samples.csv``: a header
row of wavelengths (nm) and a last column named after the target, one sample per row.
"""
from __future__ import annotations

import datetime
import hashlib
import io
import json
import logging
import zipfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DERIVATIVE_DEFAULTS = {'window': 11, 'polyorder': 2}
CHECK_SPECTRA = 8


def _snv(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    return x / np.sqrt((x ** 2).mean(axis=1, keepdims=True))


def preprocess(x: np.ndarray, preprocessing: str, wavelengths: np.ndarray, derivative: dict | None = None):
    """``(preprocessed spectra, derivative matrix or None, derivative parameters or {})``."""
    from hsi_package import savgol_derivative_matrix
    x = np.asarray(x, np.float64)
    if preprocessing in (None, '', 'none'):
        return x, None, {}
    if preprocessing == 'snv':
        return _snv(x), None, {}
    if preprocessing == 'first_derivative':
        p = dict(DERIVATIVE_DEFAULTS, **(derivative or {}))
        delta = float(np.median(np.diff(wavelengths)))
        matrix = savgol_derivative_matrix(wavelengths.size, int(p['window']), int(p['polyorder']), delta)
        return x @ matrix.T, matrix, {'window': int(p['window']), 'polyorder': int(p['polyorder']),
                                      'delta': delta, 'deriv': 1, 'mode': 'interp', 'units': 'per_nm'}
    raise ValueError(f"preprocessing must be none, snv or first_derivative, got {preprocessing!r}")


def _cv_rmse(features: np.ndarray, target: np.ndarray, make, folds: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    order = rng.permutation(target.size)
    errors = np.empty(target.size)
    for k in range(folds):
        test = order[k::folds]
        train = np.setdiff1d(order, test)
        model = make().fit(features[train], target[train])
        errors[test] = np.ravel(model.predict(features[test])) - target[test]
    return float(np.sqrt(np.mean(errors ** 2)))


def choose_components(features: np.ndarray, target: np.ndarray, max_components: int = 15, folds: int = 5,
                      seed: int = 0) -> tuple[int, list]:
    """PLS components by ``folds``-fold cross-validation: the fewest whose RMSECV is within one
    standard error of the best (fewer components = less overfitting to the training lots)."""
    from sklearn.cross_decomposition import PLSRegression
    limit = int(max(1, min(max_components, features.shape[1], target.size - target.size // folds - 1)))
    rmse = [_cv_rmse(features, target, lambda c=c: PLSRegression(n_components=c), folds, seed)
            for c in range(1, limit + 1)]
    best = int(np.argmin(rmse))
    tolerance = rmse[best] * (1.0 + 1.0 / np.sqrt(2.0 * target.size))
    chosen = next(c for c in range(1, best + 2) if rmse[c - 1] <= tolerance)
    return chosen, rmse


def _bins(target: np.ndarray, units: str) -> list:
    """Three target bins (tertiles): ``[(class id, label, row mask, mean target)]``."""
    edges = np.quantile(target, [0.0, 1 / 3, 2 / 3, 1.0])
    out = []
    for k, name in enumerate(('low', 'mid', 'high')):
        upper = target <= edges[k + 1] if k == 2 else target < edges[k + 1]
        rows = (target >= edges[k]) & upper
        if rows.sum() < 2:
            continue
        unit = f" {units}" if units else ''
        out.append((k, f"{name} {target[rows].min():.3g}–{target[rows].max():.3g}{unit}", rows,
                    float(target[rows].mean())))
    return out


def _domain(x: np.ndarray, bins: list) -> dict:
    return {'classes': [{'class_id': cid, 'label': label, 'n_spectra': int(rows.sum()), 'target_mean': mean,
                         'mean': x[rows].mean(axis=0).tolist(), 'sd': x[rows].std(axis=0, ddof=1).tolist()}
                        for cid, label, rows, mean in bins]}


def build_regression_package(spectra, target, wavelengths, path, *, target_name: str = 'moisture',
                             units: str = '%', valid_range=None, display_range=None, preprocessing: str = 'snv',
                             selected=None, algorithm: str = 'PLSR', components: int | None = None,
                             ridge_alpha: float = 1.0, input_units: str = 'reflectance',
                             object_min_mean: float | None = None, name: str | None = None,
                             derivative: dict | None = None, folds: int = 5, seed: int = 0) -> Path:
    """Fit and write a regression package; returns its path.

    Args:
        spectra: ``(n, B)`` model input spectra (reflectance as fractions, or counts).
        target: ``(n,)`` reference values (e.g. moisture from oven drying).
        wavelengths: ``(B,)`` nm, increasing.
        valid_range: ``(lo, hi)`` the model may be trusted for (default: training range + 10 %).
        selected: band indices the model uses after the preprocessing (default: all).
        algorithm: ``'PLSR'`` (``components`` or chosen by cross-validation) or ``'Ridge'``.
        object_min_mean: object mask stored in the package (background below this mean of the input).
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    import joblib

    x = np.asarray(spectra, np.float64)
    y = np.asarray(target, np.float64).reshape(-1)
    wl = np.asarray(wavelengths, np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.size or x.shape[1] != wl.size:
        raise ValueError(f"spectra {x.shape}, target {y.shape} and wavelengths {wl.shape} do not fit together")
    good = np.isfinite(x).all(axis=1) & np.isfinite(y)
    x, y = x[good], y[good]
    if y.size < 10:
        raise ValueError(f"{y.size} usable samples: at least 10 are needed")
    if np.any(np.diff(wl) <= 0):
        raise ValueError("wavelengths must increase")
    z, _, derivative_parameters = preprocess(x, preprocessing, wl, derivative)
    selected = np.arange(wl.size) if selected is None else np.asarray(selected, np.intp)
    features = z[:, selected]
    scaler = StandardScaler().fit(features)
    scaled = scaler.transform(features)
    rmsecv = None
    if algorithm == 'PLSR':
        cv_curve = []
        if components is None:
            components, cv_curve = choose_components(scaled, y, folds=folds, seed=seed)
        model = PLSRegression(n_components=int(components)).fit(scaled, y)
        if cv_curve:
            rmsecv = cv_curve[int(components) - 1]
        else:
            rmsecv = _cv_rmse(scaled, y, lambda: PLSRegression(n_components=int(components)), folds, seed)
    elif algorithm == 'Ridge':
        model = Ridge(alpha=float(ridge_alpha)).fit(scaled, y)
        rmsecv = _cv_rmse(scaled, y, lambda: Ridge(alpha=float(ridge_alpha)), folds, seed)
    else:
        raise ValueError(f"algorithm must be PLSR or Ridge, got {algorithm!r}")
    fitted = np.ravel(model.predict(scaled))
    residual = fitted - y
    rmsec = float(np.sqrt(np.mean(residual ** 2)))
    r2 = float(1.0 - np.sum(residual ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-300))
    if valid_range is None:
        pad = 0.1 * (y.max() - y.min())
        valid_range = (float(y.min() - pad), float(y.max() + pad))

    bins = _bins(y, units)
    stats = {'n': int(y.size), 'target_mean': float(y.mean()), 'target_sd': float(y.std(ddof=1)),
             'target_min': float(y.min()), 'target_max': float(y.max()), 'rmsec': rmsec, 'r2': r2}
    if rmsecv is not None:
        stats['rmsecv'] = float(rmsecv)
    if algorithm == 'PLSR':
        stats['components'] = int(components)
    pick = np.linspace(0, y.size - 1, min(CHECK_SPECTRA, y.size)).astype(int)
    reference = {
        'domains': {'raw_wavelength_analysis': _domain(x, bins),
                    'raw_wavelength_selected': _domain(x, bins),
                    'preprocessed_wavelength_selected': _domain(z, bins)},
        'regression': dict(stats, check={'spectra': x[pick].tolist(), 'predictions': fitted[pick].tolist()}),
    }
    steps = [] if not preprocessing or preprocessing == 'none' else [
        {'name': preprocessing, **({'parameters': derivative_parameters,
                                    'implementation': 'hsi_lab_core.preprocessing.savgol_derivative'}
                                   if preprocessing == 'first_derivative' else {})}]
    audit = {'steps': steps, 'final_scaler': {'type': 'StandardScaler', 'with_mean': True, 'with_std': True,
                                               'input_features': int(selected.size)}}
    if selected.size != wl.size:
        audit['wavelength_selection'] = {'selected_band_indices': selected.tolist(),
                                         'selected_wavelengths_nm': wl[selected].tolist()}

    def dump(obj) -> bytes:
        buffer = io.BytesIO()
        joblib.dump(obj, buffer)
        return buffer.getvalue()

    files = {'model.joblib': dump(model), 'scaler.joblib': dump(scaler),
             'training_spectral_reference.json': json.dumps(reference).encode('utf-8')}
    manifest = {
        'name': name or f"{algorithm} {target_name}",
        'task_type': 'regression', 'algorithm': algorithm,
        'target': {'name': target_name, 'units': units, 'valid_range': list(map(float, valid_range)),
                   'display_range': list(map(float, display_range or valid_range))},
        'wavelengths': wl.tolist(),
        'pipeline_audit': audit,
        'input_contract': {'measurement_mode': 'reflectance', 'units': 'fraction'} if input_units == 'reflectance'
        else {'measurement_mode': 'counts'},
        'evaluation_evidence': {'selection_metric': 'rmsecv', 'selection_score': stats.get('rmsecv', rmsec)},
        'exported_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'hsi_lab_core_version': 'hdf5-socket regression_builder',
        'artifact_sha256': {},
    }
    if object_min_mean is not None:
        manifest['object_mask'] = {'min_mean': float(object_min_mean)}
    manifest['artifact_sha256'] = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('manifest.json', json.dumps(manifest, indent=1))
        for member, blob in files.items():
            archive.writestr(member, blob)
    logger.info("Regression package %s: %s, n=%d, RMSEC %.4g, RMSECV %s, R² %.4f", path.name, algorithm, y.size,
                rmsec, f"{rmsecv:.4g}" if rmsecv is not None else 'n/a', r2)
    return path


def read_samples(path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str | None]:
    """``(spectra, target, wavelengths, target name or None)`` of an ``.npz`` or ``.csv`` file."""
    path = Path(path)
    if path.suffix.lower() == '.npz':
        with np.load(path) as data:
            return (np.asarray(data['spectra'], np.float64), np.asarray(data['target'], np.float64),
                    np.asarray(data['wavelengths'], np.float64), None)
    with open(path, encoding='utf-8-sig') as handle:
        header = handle.readline().strip().split(',')
    table = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
    wavelengths = np.array([float(h) for h in header[:-1]])
    return table[:, :-1], table[:, -1], wavelengths, header[-1].strip() or None


def main(argv=None) -> int:
    import argparse
    from hsi_package import load_package
    parser = argparse.ArgumentParser(description="Build a regression .hsipkg (e.g. moisture by PLSR).")
    parser.add_argument('samples', help=".npz (spectra, target, wavelengths) or .csv (wavelength header, target last)")
    parser.add_argument('-o', '--output', default=None, help="package path (default: models/<target>_<algorithm>.hsipkg)")
    parser.add_argument('--target', default=None, help="target name (default: the CSV's last column, or moisture)")
    parser.add_argument('--units', default='%')
    parser.add_argument('--valid-range', nargs=2, type=float, default=None)
    parser.add_argument('--display-range', nargs=2, type=float, default=None)
    parser.add_argument('--preprocessing', choices=('none', 'snv', 'first_derivative'), default='snv')
    parser.add_argument('--algorithm', choices=('PLSR', 'Ridge'), default='PLSR')
    parser.add_argument('--components', type=int, default=None, help="PLS components (default: cross-validation)")
    parser.add_argument('--alpha', type=float, default=1.0, help="Ridge alpha")
    parser.add_argument('--counts', action='store_true', help="the spectra are counts, not reflectance")
    parser.add_argument('--object-min-mean', type=float, default=None,
                        help="background below this mean of the input bands (stored in the package)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    spectra, target, wavelengths, column = read_samples(args.samples)
    name = args.target or column or 'moisture'
    output = args.output or Path(__file__).resolve().parent / 'models' / f"{name}_{args.algorithm}.hsipkg"
    path = build_regression_package(
        spectra, target, wavelengths, output, target_name=name, units=args.units, valid_range=args.valid_range,
        display_range=args.display_range, preprocessing=args.preprocessing, algorithm=args.algorithm,
        components=args.components, ridge_alpha=args.alpha, input_units='counts' if args.counts else 'reflectance',
        object_min_mean=args.object_min_mean)
    print(load_package(path).describe())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
