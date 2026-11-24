"""SGD packages (.hsipkg with sklearn SGDClassifier, optionally after a PCA): read, folded into one
linear map, and the same classes as sklearn's own pipeline.

The package is written like a real one (SGD_731919: reflectance -> PCA -> StandardScaler ->
SGDClassifier hinge); the expected classes come from the fitted sklearn objects themselves.
"""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hsi_package import load_package  # noqa: E402
from inference_engine import HsiLinearClassifier, load_classifier  # noqa: E402
from test_hsi_package import LABELS, SELECTED, WAVELENGTHS, dump_as, training_spectra  # noqa: E402


def fit_sgd(pca_components=4, selected=None, n_classes=3, loss='hinge'):
    """sklearn objects fitted on the synthetic spectra: (x, y, pca or None, scaler, sgd, selected)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    x, y = training_spectra(np.random.default_rng(7))
    keep = y < n_classes
    x, y = x[keep], y[keep]
    selected = np.arange(WAVELENGTHS.size) if selected is None else np.asarray(selected)
    f = x[:, selected]
    pca = PCA(n_components=pca_components).fit(f) if pca_components else None
    p = pca.transform(f) if pca is not None else f
    scaler = StandardScaler().fit(p)
    sgd = SGDClassifier(loss=loss, random_state=0, max_iter=2000, tol=1e-6).fit(scaler.transform(p), y)
    return x, y, pca, scaler, sgd, selected


def sklearn_predict(fitted, x):
    _, _, pca, scaler, sgd, selected = fitted
    f = np.asarray(x, np.float64)[:, selected]
    p = pca.transform(f) if pca is not None else f
    return sgd.predict(scaler.transform(p))


def build_sgd_package(folder, fitted, name='sgd', change=None):
    """Write ``<name>.hsipkg`` around the fitted objects (the PCA as ``spectral_pipeline.SafePCA``)."""
    import joblib
    x, y, pca, scaler, sgd, selected = fitted
    classes = np.unique(y)
    counts = np.bincount(y)
    buffer = io.BytesIO()
    joblib.dump(sgd, buffer)                               # the real sklearn pickle, as in a package
    files = {'model.joblib': buffer.getvalue(),
             'scaler.joblib': dump_as({'with_mean': True, 'with_std': True, 'mean_': scaler.mean_,
                                       'var_': scaler.var_, 'scale_': scaler.scale_,
                                       'n_features_in_': scaler.n_features_in_,
                                       'n_samples_seen_': np.float64(scaler.n_samples_seen_)},
                                      'hsi_lab_core.features.scaling', 'StandardFeatureScaler')}
    if pca is not None:
        files['pca.joblib'] = dump_as({'whiten': False, 'mean_': pca.mean_, 'components_': pca.components_,
                                       'explained_variance_': pca.explained_variance_,
                                       'n_samples_': pca.n_samples_, 'n_components_': pca.n_components_},
                                      'spectral_pipeline', 'SafePCA')
    rows = [{'class_id': int(k), 'label': LABELS[k], 'n_spectra': int(counts[k]),
             'mean': x[y == k].mean(axis=0).tolist(), 'sd': x[y == k].std(axis=0, ddof=1).tolist()} for k in classes]
    domain = {'axis_values': WAVELENGTHS.tolist(), 'classes': rows}
    files['training_spectral_reference.json'] = json.dumps({'domains': {
        'preprocessed_wavelength_selected': domain, 'raw_wavelength_analysis': domain,
        'raw_wavelength_selected': domain}}).encode()
    n_scaled = scaler.n_features_in_
    audit = {'steps': [], 'spectral_axis_transform': {'spectral_axis_kind': 'wavelength_nm',
                                                      'spectral_resampling_mode': 'none'},
             'pca': None if pca is None else {'type': 'PCA', 'n_components': int(pca.n_components_), 'whiten': False,
                                              'input_features': int(selected.size)},
             'spectral_indices': None, 'spectral_index_selection': None, 'spectral_index_discovery': None,
             'wavelength_selection': None if selected.size == WAVELENGTHS.size else {
                 'selected_band_indices': selected.tolist(), 'selected_wavelengths_nm': WAVELENGTHS[selected].tolist()},
             'final_scaler': {'type': 'StandardScaler', 'with_mean': True, 'with_std': True,
                              'input_features': int(n_scaled)}}
    manifest = {'name': 'sgd test', 'algorithm': 'SGD', 'task_type': 'classification',
                'labels': LABELS[:classes.size], 'hsi_lab_core_version': '1.17.0', 'exported_at': '2026-11-18T03:32:17',
                'wavelengths': WAVELENGTHS.tolist(), 'pipeline_audit': audit,
                'input_contract': {'measurement_mode': 'reflectance', 'units': 'fraction'},
                'class_background': {'Background': True}}
    if change is not None:
        change(manifest, files)
    manifest.setdefault('artifact_sha256', {k: hashlib.sha256(v).hexdigest() for k, v in files.items()})
    path = Path(folder) / f'{name}.hsipkg'
    with zipfile.ZipFile(path, 'w') as package:
        package.writestr('manifest.json', json.dumps(manifest))
        for member, data in files.items():
            package.writestr(member, data)
    return path


def test_sgd_package_with_pca_classifies_as_sklearn(tmp_path):
    fitted = fit_sgd(pca_components=4)
    path = build_sgd_package(tmp_path, fitted)
    package = load_package(path)
    assert package.algorithm == 'SGD' and package.pca_components.shape == (4, WAVELENGTHS.size)
    text = package.describe()
    assert 'PCA 4 -> StandardScaler -> SGD 3 classes (linear, loss hinge)' in text
    assert 'PCA mean and scaler mean reproduced' in text and 'scaler variance = PCA variance' in text
    model = load_classifier(path)
    assert isinstance(model, HsiLinearClassifier) and model.W_fused.shape == (WAVELENGTHS.size, 3)
    columns, _ = model.bind(WAVELENGTHS, WAVELENGTHS.size)
    x = fitted[0] * np.random.default_rng(1).uniform(0.8, 1.2, (fitted[0].shape[0], 1))   # beside the training
    expected = sklearn_predict(fitted, x)
    got = model.predict(x.astype(np.float32), columns)
    scores = model._package_scores(x, columns)[0]
    top = np.sort(scores, axis=1)
    clear = top[:, -1] - top[:, -2] > 1e-4                 # float32 sums may flip an exact tie only
    assert clear.mean() > 0.99
    np.testing.assert_array_equal(got[clear], expected[clear])
    np.testing.assert_array_equal(scores.argmax(axis=1), expected)                  # float64: all of them
    with pytest.raises(ValueError, match='no probabilities'):
        model.predict_confidence(x, columns)


def test_binary_sgd_on_selected_bands_without_pca(tmp_path):
    fitted = fit_sgd(pca_components=0, selected=SELECTED, n_classes=2, loss='log_loss')
    path = build_sgd_package(tmp_path, fitted)
    model = load_classifier(path)
    assert model.W_fused.shape == (WAVELENGTHS.size, 2) and not np.any(model.W_fused[:, 0])
    columns, _ = model.bind(WAVELENGTHS, WAVELENGTHS.size)
    x = fitted[0]
    np.testing.assert_array_equal(model.predict(x.astype(np.float32), columns), sklearn_predict(fitted, x))
    ids, confidence = model.predict_confidence(x, columns)
    _, _, _, scaler, sgd, selected = fitted
    proba = sgd.predict_proba(scaler.transform(x[:, selected]))
    np.testing.assert_allclose(confidence, proba.max(axis=1), rtol=1e-5)
    np.testing.assert_array_equal(ids, sgd.classes_[proba.argmax(axis=1)])


def test_sgd_pixel_with_a_bad_band_is_background_and_resampling_folds(tmp_path):
    fitted = fit_sgd(pca_components=4)
    model = load_classifier(build_sgd_package(tmp_path, fitted))
    columns, _ = model.bind(WAVELENGTHS, WAVELENGTHS.size)
    x = fitted[0][[500, 900, 450, 950]].astype(np.float32)           # Leaf, Wood rows
    x[2, 3], x[3, 11] = np.nan, np.inf
    ids = model.predict(x, columns)
    assert ids[2] == 0 and ids[3] == 0 and ids[0] != 0 and ids[1] != 0
    shifted, _ = model.bind(WAVELENGTHS + 1.0, WAVELENGTHS.size)       # recalibrated: interpolated, folded in
    good = fitted[0][::7].astype(np.float32)
    np.testing.assert_array_equal(model.predict(good, shifted), model.predict(shifted.apply(good), model.selected_bands))


def test_sgd_packages_that_do_not_fit_are_refused(tmp_path):
    fitted = fit_sgd(pca_components=4)

    def other_pca_mean(manifest, files):
        files['pca.joblib'] = dump_as({'whiten': False, 'mean_': fitted[2].mean_ + 0.01,
                                       'components_': fitted[2].components_}, 'spectral_pipeline', 'SafePCA')

    def scaler_before_pca(manifest, files):
        manifest['pipeline_audit']['final_scaler']['input_features'] = WAVELENGTHS.size

    def no_pca_file(manifest, files):
        del files['pca.joblib']

    def whitened(manifest, files):
        manifest['pipeline_audit']['pca']['whiten'] = True

    def not_sgd(manifest, files):
        files['model.joblib'] = dump_as({'loss': 'hinge'}, 'hsi_lab_core.models.custom', 'SGDClassifier')

    for name, change, message in (('mean', other_pca_mean, 'PCA mean'), ('order', scaler_before_pca, 'final scaler'),
                                  ('file', no_pca_file, 'pca.joblib is missing'), ('white', whitened, 'whitened'),
                                  ('model', not_sgd, 'must hold sklearn')):
        with pytest.raises(ValueError, match=message):
            load_package(build_sgd_package(tmp_path, fitted, name=name, change=change))
