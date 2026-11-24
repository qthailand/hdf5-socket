"""PLS-DA packages and an MLC after a PCA (.hsipkg): read, run, and the same classes as sklearn /
scipy on the pickled objects.

Written like the real packages (PLS-DA_617987, MLC_20261123: reflectance -> [1st derivative] ->
PCA -> StandardScaler -> PLS-DA | MLC); the PLS-DA model.joblib holds a real
``sklearn.cross_decomposition.PLSRegression`` inside a stand-in ``hsi_lab_core`` PLSDAClassifier.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hsi_package import PLS_RESPONSE, load_package  # noqa: E402
from inference_engine import HsiLinearClassifier, HsiPackageClassifier, load_classifier  # noqa: E402
from test_hsi_package import SELECTED, WAVELENGTHS, dump_as, training_spectra  # noqa: E402
from test_hsi_package_sgd import build_sgd_package  # noqa: E402


def fitted_chain(pca_components=4, selected=None, uneven=False):
    """(x, y, pca or None, scaler, None, selected) as test_hsi_package_sgd's builder takes them."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    x, y = training_spectra(np.random.default_rng(7))
    if uneven:                                     # other counts per class: a swapped coding shows
        keep = (y != 2) | (np.arange(y.size) % 4 != 0)
        x, y = x[keep], y[keep]
    selected = np.arange(WAVELENGTHS.size) if selected is None else np.asarray(selected)
    f = x[:, selected]
    pca = PCA(n_components=pca_components).fit(f) if pca_components else None
    scaler = StandardScaler().fit(pca.transform(f) if pca is not None else f)
    return x, y, pca, scaler, None, selected


def scaled_features(fitted, x):
    _, _, pca, scaler, _, selected = fitted
    f = np.asarray(x, np.float64)[:, selected]
    return scaler.transform(pca.transform(f) if pca is not None else f)


def fit_pls(fitted, components=3, coding=None):
    from sklearn.cross_decomposition import PLSRegression
    y = fitted[1]
    target = np.eye(3)[y] if coding is None else coding(y)
    return PLSRegression(n_components=components).fit(scaled_features(fitted, fitted[0]), target)


def plsda_change(pls, algorithm='PLS-DA'):
    def change(manifest, files):
        manifest['algorithm'] = algorithm
        files['model.joblib'] = dump_as({'n_components': pls.n_components, 'scale': True, 'classes_': np.arange(3),
                                         'n_components_': pls.n_components, 'model_': pls, 'n_features_in_':
                                         pls.n_features_in_}, 'hsi_lab_core.models.custom', 'PLSDAClassifier')
    return change


def test_plsda_with_pca_classifies_as_sklearn(tmp_path):
    fitted = fitted_chain(pca_components=4)
    pls = fit_pls(fitted)
    path = build_sgd_package(tmp_path, fitted, name='pls', change=plsda_change(pls))
    package = load_package(path)
    assert package.algorithm == 'PLS-DA' and package.linear.loss == PLS_RESPONSE
    text = package.describe()
    assert 'PCA 4 -> StandardScaler -> PLS-DA 3 classes (3 components, largest response)' in text
    assert "PLS-DA responses one-hot with the reference's class counts" in text
    model = load_classifier(path)
    assert isinstance(model, HsiLinearClassifier) and model.W_fused.shape == (WAVELENGTHS.size, 3)
    columns, _ = model.bind(WAVELENGTHS, WAVELENGTHS.size)
    x = fitted[0] * np.random.default_rng(1).uniform(0.8, 1.2, (fitted[0].shape[0], 1))   # beside the training
    response = pls.predict(scaled_features(fitted, x))
    expected = response.argmax(axis=1)
    scores = model._package_scores(x, columns)[0]
    np.testing.assert_allclose(scores, response, atol=1e-9)                       # float64: the responses
    top = np.sort(response, axis=1)
    clear = top[:, -1] - top[:, -2] > 1e-4                                         # float32 may flip a tie only
    assert clear.mean() > 0.99
    np.testing.assert_array_equal(model.predict(x.astype(np.float32), columns)[clear], expected[clear])
    ids, confidence = model.predict_confidence(x, columns)
    np.testing.assert_array_equal(ids, expected)
    softmax = np.exp(response - response.max(axis=1, keepdims=True))
    np.testing.assert_allclose(confidence, 1.0 / softmax.sum(axis=1), rtol=1e-5)


def test_plsda_on_selected_bands_without_pca(tmp_path):
    fitted = fitted_chain(pca_components=0, selected=SELECTED)
    pls = fit_pls(fitted, components=2)
    model = load_classifier(build_sgd_package(tmp_path, fitted, name='pls', change=plsda_change(pls)))
    columns, _ = model.bind(WAVELENGTHS, WAVELENGTHS.size)
    x = fitted[0]
    expected = pls.predict(scaled_features(fitted, x)).argmax(axis=1)
    np.testing.assert_array_equal(model._package_scores(x, columns)[0].argmax(axis=1), expected)
    bad = x[[500, 900]].astype(np.float32)
    bad[0, 3] = np.nan
    ids, confidence = model.predict_confidence(bad, columns)
    assert ids[0] == 0 and confidence[0] == 0.0 and ids[1] != 0                    # background, no confidence


def test_mlc_after_pca_classifies_as_scipy(tmp_path):
    from scipy.stats import multivariate_normal
    fitted = fitted_chain(pca_components=5)
    x, y = fitted[0], fitted[1]
    scaled = scaled_features(fitted, x)
    classes = np.arange(3)
    means = np.array([scaled[y == k].mean(axis=0) for k in classes])
    covariances = np.array([np.cov(scaled[y == k].T, bias=True) + 1e-6 * np.eye(5) for k in classes])
    priors = np.bincount(y) / y.size

    def mlc(manifest, files):
        manifest['algorithm'] = 'MLC'
        files['model.joblib'] = dump_as({'covariance_type': 'full', 'classes_': classes, 'means_': means,
                                         'covariances_': covariances, 'priors_': priors,
                                         'log_determinants_': np.array([np.linalg.slogdet(c)[1] for c in covariances]),
                                         'n_features_in_': 5}, 'hsi_lab_core.models.custom',
                                        'MaximumLikelihoodClassifier')

    path = build_sgd_package(tmp_path, fitted, name='mlc', change=mlc)
    text = load_package(path).describe()
    assert 'PCA 5 -> StandardScaler -> MLC 3 classes' in text and 'class means reproduced' in text
    model = load_classifier(path)
    assert isinstance(model, HsiPackageClassifier)
    columns, _ = model.bind(WAVELENGTHS, WAVELENGTHS.size)
    probe = x * np.random.default_rng(3).uniform(0.9, 1.1, (x.shape[0], 1))
    f = scaled_features(fitted, probe)
    reference = np.stack([multivariate_normal(means[k], covariances[k]).logpdf(f) + np.log(priors[k])
                          for k in classes], axis=1)
    expected = reference.argmax(axis=1)
    top = np.sort(reference, axis=1)
    clear = top[:, -1] - top[:, -2] > 1e-3
    assert clear.mean() > 0.99
    np.testing.assert_array_equal(model.predict(probe.astype(np.float32), columns)[clear], expected[clear])
    ids, _ = model.predict_confidence(probe.astype(np.float32), columns)
    np.testing.assert_array_equal(ids[clear], expected[clear])


def test_plsda_packages_that_do_not_fit_are_refused(tmp_path):
    fitted = fitted_chain(pca_components=4, uneven=True)
    pls = fit_pls(fitted)

    def swapped(y):                                          # responses in another class order
        return np.eye(3)[y][:, [2, 1, 0]]

    def plus_minus(y):                                       # +1 / -1 coding, not one-hot
        return 2.0 * np.eye(3)[y] - 1.0

    def other_coef(manifest, files):
        tampered = fit_pls(fitted)
        tampered.coef_ = tampered.coef_ * 1.01
        plsda_change(tampered)(manifest, files)

    def svm_after_pca(manifest, files):
        manifest['algorithm'] = 'SVM'

    for name, change, message in (
            ('order', plsda_change(fit_pls(fitted, coding=swapped)), 'not the classes in the package'),
            ('coding', plsda_change(fit_pls(fitted, coding=plus_minus)), 'not one-hot'),
            ('coef', other_coef, 'rotations and loadings'),
            ('svm', svm_after_pca, 'pca before an SVM')):
        with pytest.raises(ValueError, match=message):
            load_package(build_sgd_package(tmp_path, fitted, name=name, change=change))
    load_package(build_sgd_package(tmp_path, fitted, name='ok', change=plsda_change(pls)))     # the plain one passes
