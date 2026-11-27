"""HSI MLOps model packages (.hsipkg): reading without hsi_lab_core, and the same classes as its chain.

The package of the tests is written here with the layout of a real one (manifest keys, joblib
pickles of ``hsi_lab_core`` objects, training reference); the reference result is an independent
float64 implementation (SNV, band selection, StandardScaler, ``scipy`` Gaussian log-density).
"""
import hashlib
import io
import json
import pickle
import sys
import types
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsi_package import is_package, load_package  # noqa: E402
from inference_engine import HsiPackageClassifier, PushbroomLineClassifier, load_classifier  # noqa: E402
from sim_config import CameraConfig, CorrectionConfig, ModelConfig, ProcessingConfig, SimulationConfig, \
    SourceConfig, build_plan  # noqa: E402
from sources import FrameSource  # noqa: E402

WAVELENGTHS = 900.0 + 5.0 * np.arange(12)
SELECTED = [1, 3, 4, 7, 9, 10]
LABELS = ['Background', 'Leaf', 'Wood']
REAL_PACKAGES = Path('C:/Datacubes/DrJoModel')


# ------------------------------------------------------------------ a synthetic package ---

def training_spectra(rng, n=400):
    """Reflectance of three materials: a dark flat belt and two spectra of different shape."""
    x = np.linspace(0.0, 1.0, WAVELENGTHS.size)
    shapes = [0.08 + 0.02 * x, 0.45 + 0.25 * np.sin(3.0 * x), 0.55 - 0.3 * x + 0.1 * np.cos(5.0 * x)]
    spectra, labels = [], []
    for k, shape in enumerate(shapes):
        gain = 1.0 + 0.2 * rng.standard_normal((n, 1))                 # brightness: removed by SNV
        spectra.append(np.abs(shape * gain + 0.01 * rng.standard_normal((n, x.size))))
        labels += [k] * n
    return np.vstack(spectra), np.array(labels)


def snv(x, ddof=0):
    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, ddof=ddof, keepdims=True)


DERIVATIVE = {'window': 5, 'polyorder': 2, 'delta': 5.0, 'units': 'per_nm'}


def first_derivative(x, mode='interp', **parameters):
    """The first derivative as HSI MLOps takes it (independent of hsi_package: scipy)."""
    from scipy.signal import savgol_filter
    p = {**DERIVATIVE, **parameters}
    return savgol_filter(np.asarray(x, np.float64), p['window'], p['polyorder'], deriv=1, delta=p['delta'],
                         mode=mode, axis=-1)


def preprocess(x, steps, ddof=0, derivative_mode='interp'):
    z = np.asarray(x, np.float64)
    for step in steps:
        z = snv(z, ddof) if step == 'snv' else first_derivative(z, derivative_mode)
    return z


def dump_as(obj_state, module, name):
    """joblib pickle of an object whose class lives in ``module`` (a stand-in for hsi_lab_core)."""
    import joblib
    created = []
    parts = module.split('.')
    for k in range(1, len(parts) + 1):
        key = '.'.join(parts[:k])
        if key not in sys.modules:
            sys.modules[key] = types.ModuleType(key)
            created.append(key)
    try:
        cls = type(name, (), {'__module__': module})
        setattr(sys.modules[module], name, cls)
        obj = cls()
        obj.__dict__.update(obj_state)
        buffer = io.BytesIO()
        joblib.dump(obj, buffer)
        return buffer.getvalue()
    finally:
        for key in created:
            del sys.modules[key]


def build_package(folder, name='model', ddof=0, change=None, wrap=False, steps=('snv',), derivative_mode='interp'):
    """Write ``<name>.hsipkg`` (or a zip around it); ``change(manifest, files)`` edits it before the checksums."""
    rng = np.random.default_rng(7)
    x, y = training_spectra(rng)
    z = preprocess(x, steps, ddof, derivative_mode)
    features = z[:, SELECTED]
    mean, var = features.mean(axis=0), features.var(axis=0)
    scale = np.sqrt(var)
    scaled = (features - mean) / scale
    classes = np.arange(len(LABELS))
    means = np.array([scaled[y == k].mean(axis=0) for k in classes])
    covariances = [np.cov(scaled[y == k].T, bias=True) + 1e-6 * np.eye(len(SELECTED)) for k in classes]
    counts = np.bincount(y)
    files = {
        'model.joblib': dump_as({'covariance_type': 'full', 'reg_covar': 1e-6, 'priors': 'empirical',
                                 'classes_': classes, 'means_': means, 'covariances_': covariances,
                                 'precisions_': [np.linalg.inv(c) for c in covariances],
                                 'log_determinants_': np.array([np.linalg.slogdet(c)[1] for c in covariances]),
                                 'priors_': counts / counts.sum(), 'class_counts_': counts,
                                 'n_features_in_': len(SELECTED), 'covariance_type_': 'full'},
                                'hsi_lab_core.models.custom', 'MaximumLikelihoodClassifier'),
        'scaler.joblib': dump_as({'with_mean': True, 'with_std': True, 'copy': True, 'mean_': mean, 'var_': var,
                                  'scale_': scale, 'n_features_in_': len(SELECTED),
                                  'n_samples_seen_': np.float64(len(y))},
                                 'hsi_lab_core.features.scaling', 'StandardFeatureScaler'),
    }
    reference = {'domains': {'preprocessed_wavelength_selected': {
        'axis_values': WAVELENGTHS.tolist(),
        'classes': [{'class_id': int(k), 'label': LABELS[k], 'n_spectra': int(counts[k]),
                     'mean': z[y == k].mean(axis=0).tolist(), 'sd': z[y == k].std(axis=0, ddof=1).tolist()}
                    for k in classes]},
        'raw_wavelength_analysis': {
        'axis_values': WAVELENGTHS.tolist(),
        'classes': [{'class_id': int(k), 'label': LABELS[k], 'n_spectra': int(counts[k]),
                     'mean': x[y == k].mean(axis=0).tolist(), 'sd': x[y == k].std(axis=0, ddof=1).tolist()}
                    for k in classes]}}}
    reference['domains']['raw_wavelength_selected'] = reference['domains']['raw_wavelength_analysis']
    axis ={'spectral_axis_kind': 'wavelength_nm', 'spectral_resampling_mode': 'none',
            'selected_source_band_indices': list(range(WAVELENGTHS.size))}
    manifest = {
        'name': 'test model', 'algorithm': 'MLC', 'task_type': 'classification', 'labels': LABELS,
        'hsi_lab_core_version': '1.15.0', 'exported_at': '2026-11-10T08:56:38',
        'wavelengths': WAVELENGTHS.tolist(), 'n_bands': int(WAVELENGTHS.size),
        'pipeline_audit': {
            'steps': [{'name': step, 'parameters': dict(DERIVATIVE) if step == 'first_derivative' else {},
                       'implementation': 'hsi_lab_core.preprocessing.savgol_derivative'
                       if step == 'first_derivative' else 'hsi_lab_core.preprocessing.snv'} for step in steps],
            'spectral_axis_transform': axis, 'pca': None, 'spectral_indices': None,
            'spectral_index_selection': None, 'spectral_index_discovery': None,
            'wavelength_selection': {'method': 'cars', 'selected_band_indices': SELECTED,
                                     'selected_wavelengths_nm': WAVELENGTHS[SELECTED].tolist()},
            'final_scaler': {'type': 'StandardScaler', 'with_mean': True, 'with_std': True}},
        'input_contract': {'measurement_mode': 'reflectance', 'units': 'fraction'},
        'class_colors': {'Background': '#000000', 'Leaf': '#05f515', 'Wood': '#ffbb00'},
        'class_background': {'Background': True, 'Leaf': False, 'Wood': False},
        'evaluation_evidence': {'selection_metric': '3-Fold OOF Balanced Accuracy', 'selection_score': 0.98},
    }
    files['training_spectral_reference.json'] = json.dumps(reference).encode()
    if change is not None:
        change(manifest, files)
    manifest.setdefault('artifact_sha256', {k: hashlib.sha256(v).hexdigest() for k, v in files.items()})
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, 'w') as package:
        package.writestr('manifest.json', json.dumps(manifest))
        for member, data in files.items():
            package.writestr(member, data)
    path = Path(folder) / (f'{name}.zip' if wrap else f'{name}.hsipkg')
    if wrap:
        with zipfile.ZipFile(path, 'w') as outer:
            outer.writestr(f'{name}.hsipkg', blob.getvalue())
    else:
        path.write_bytes(blob.getvalue())
    return path


def reference_scores(package, x):
    """float64, written independently of the classifier: log prior + Gaussian log-density."""
    from scipy.stats import multivariate_normal
    z = snv(np.asarray(x, np.float64)) if package.snv else np.asarray(x, np.float64)
    if package.derivative is not None:
        z = first_derivative(z, **package.derivative_parameters)
    f = (z[:, package.selected] - package.scaler_mean) / package.scaler_scale
    return np.stack([np.log(package.priors[k]) + multivariate_normal(package.means[k], package.covariances[k]).logpdf(f)
                     for k in range(package.classes.size)], axis=1)


@pytest.fixture
def package_path(tmp_path):
    return build_package(tmp_path)


# ------------------------------------------------------------------------------ reading ---

@pytest.mark.parametrize('wrap', [False, True])
def test_package_is_read_without_hsi_lab_core(tmp_path, wrap):
    path = build_package(tmp_path, wrap=wrap)
    assert is_package(path)
    package = load_package(path)
    assert not any(name == 'hsi_lab_core' or name.startswith('hsi_lab_core.') for name in sys.modules)
    assert package.algorithm == 'MLC' and package.snv and package.input_units == 'reflectance'
    assert package.selected.tolist() == SELECTED and package.wavelengths.tolist() == WAVELENGTHS.tolist()
    assert package.labels == {0: 'Background', 1: 'Leaf', 2: 'Wood'} and package.background == (0,)
    assert package.colors == {0: [0, 0, 0], 1: [5, 245, 21], 2: [255, 187, 0]}
    text = package.describe()
    assert 'sha256 ok' in text and 'training reference ok' in text and 'SNV -> 6 bands' in text


def test_load_classifier_picks_the_class_from_the_file(tmp_path, package_path):
    import joblib
    assert isinstance(load_classifier(package_path), HsiPackageClassifier)
    joblib.dump({'W_fused': np.eye(2, dtype=np.float32), 'b_fused': np.zeros(2, np.float32),
                 'selected_bands': np.array([0, 1]), 'classes': np.array([1, 2])}, tmp_path / 'm.joblib')
    assert not is_package(tmp_path / 'm.joblib')
    linear = load_classifier(tmp_path / 'm.joblib')
    assert type(linear) is PushbroomLineClassifier
    with pytest.raises(FileNotFoundError):
        load_classifier(tmp_path / 'missing.hsipkg')


def test_package_gives_its_class_colours_to_the_display(package_path):
    from utils import generate_tab10_lut
    model = load_classifier(package_path)
    lut = generate_tab10_lut(model.class_colors)
    assert lut[1].tolist() == [5, 245, 21] and lut[2].tolist() == [255, 187, 0] and lut[0].tolist() == [0, 0, 0]
    assert model.category_names == {0: 'Background', 1: 'Leaf', 2: 'Wood'} and model.classes_.tolist() == [0, 1, 2]
    assert model.input_units == 'reflectance' and model.input_units_declared


# ---------------------------------------------------------------------------- inference ---

def test_package_classes_are_those_of_the_training_chain(package_path):
    """SNV -> selected bands -> scaler -> Gaussian ML gives the classes (and posterior) of a float64
    reference; the kernel that drops losing classes early returns the arg max of the full scores."""
    model = load_classifier(package_path)
    columns, how = model.bind(WAVELENGTHS, WAVELENGTHS.size)
    assert how.startswith('by wavelength: 12 bands') and 'SNV -> 6 bands -> StandardScaler -> MLC 3 classes' in how
    rng = np.random.default_rng(11)
    x, y = training_spectra(rng, n=1500)
    x = np.vstack([x, rng.uniform(0.02, 0.9, (1500, WAVELENGTHS.size))]).astype(np.float32)    # and far from them
    reference = reference_scores(model.package, x)
    ids = model.predict(x, columns)
    assert ids.dtype == np.uint8 and ids.shape == (x.shape[0],)
    np.testing.assert_array_equal(ids, reference.argmax(axis=1))
    assert (ids[:y.size] == y).mean() > 0.99                               # the materials it was trained on
    scores, valid = model._scores(x, columns)
    assert valid.all()
    np.testing.assert_array_equal(ids, scores.argmax(axis=1))
    np.testing.assert_allclose(scores + 0.0, reference + 0.5 * len(SELECTED) * np.log(2 * np.pi),
                               rtol=2e-4, atol=0.05)
    confidence_ids, confidence = model.predict_confidence(x, columns)
    np.testing.assert_array_equal(confidence_ids, ids)
    posterior = np.exp(reference - reference.max(axis=1, keepdims=True))
    np.testing.assert_allclose(confidence, posterior.max(axis=1) / posterior.sum(axis=1), atol=1e-3)
    # SNV: brightness and offset do not change the class; counts instead of a fraction neither
    np.testing.assert_array_equal(model.predict(x * 4000.0, columns), ids)
    np.testing.assert_array_equal(model.predict(x + 0.3, columns), ids)
    # any input layout: more columns than bands, another order
    wide = np.zeros((x.shape[0], 20), np.float32)
    order = rng.permutation(20)[:WAVELENGTHS.size]
    wide[:, order] = x
    np.testing.assert_array_equal(model.predict(wide, order.astype(np.intp)), ids)


def test_package_without_snv_or_selection(tmp_path):
    def all_bands(manifest, files):
        manifest['pipeline_audit']['wavelength_selection'] = None
    with pytest.raises(ValueError, match='shape'):                         # the model has 6 features, not 12
        load_package(build_package(tmp_path, name='all', change=all_bands))
    model = load_classifier(build_package(tmp_path, name='plain', steps=()))
    assert not model.package.snv and 'SNV' not in model.package.chain
    columns, _ = model.bind(WAVELENGTHS, WAVELENGTHS.size)
    x, y = training_spectra(np.random.default_rng(5), n=300)
    np.testing.assert_array_equal(model.predict(x.astype(np.float32), columns),
                                  reference_scores(model.package, x.astype(np.float32)).argmax(axis=1))


@pytest.mark.parametrize('n_bands, window, polyorder, delta', [(159, 11, 2, 4.9941), (12, 5, 2, 5.0), (30, 7, 3, 1.0),
                                                               (11, 11, 2, 2.0), (20, 9, 1, 0.5), (8, 3, 2, 3.0)])
def test_derivative_matrix_is_scipy_savgol_interp(n_bands, window, polyorder, delta):
    from hsi_package import savgol_derivative_matrix
    x = np.random.default_rng(n_bands).normal(size=(5, n_bands))
    matrix = savgol_derivative_matrix(n_bands, window, polyorder, delta)
    np.testing.assert_allclose(x @ matrix.T, first_derivative(x, window=window, polyorder=polyorder, delta=delta),
                               rtol=0, atol=1e-12 * np.abs(x).max() / delta)
    np.testing.assert_allclose(matrix.sum(axis=1), 0.0, atol=1e-12)          # a constant has no slope
    assert all(np.count_nonzero(row) <= window for row in matrix)


def test_derivative_package_classes_are_those_of_the_training_chain(tmp_path):
    """reflectance -> Savitzky-Golay 1st derivative -> selected bands -> scaler -> MLC: the classes
    and posterior of a float64 scipy reference, in both the kernel and the full scores."""
    path = build_package(tmp_path, name='d1', steps=('first_derivative',))
    package = load_package(path)
    assert not package.snv and package.derivative_parameters == {'window': 5, 'polyorder': 2, 'delta': 5.0}
    assert '1st derivative (Savitzky-Golay 5/2) -> 6 bands -> StandardScaler -> MLC 3 classes' in package.chain
    text = package.describe()
    assert 'training reference ok' in text and '1st derivative of the raw means reproduced' in text
    model = load_classifier(path)
    columns, how = model.bind(WAVELENGTHS, WAVELENGTHS.size)
    assert '1st derivative' in how
    rng = np.random.default_rng(12)
    x, y = training_spectra(rng, n=1500)
    x = np.vstack([x, rng.uniform(0.02, 0.9, (1500, WAVELENGTHS.size))]).astype(np.float32)
    reference = reference_scores(package, x)
    ids = model.predict(x, columns)
    np.testing.assert_array_equal(ids, reference.argmax(axis=1))
    assert (ids[:y.size] == y).mean() > 0.95
    scores, valid = model._scores(x, columns)
    assert valid.all()
    np.testing.assert_array_equal(ids, scores.argmax(axis=1))
    np.testing.assert_allclose(scores + 0.0, reference + 0.5 * len(SELECTED) * np.log(2 * np.pi),
                               rtol=2e-4, atol=0.05)
    confidence_ids, confidence = model.predict_confidence(x, columns)
    np.testing.assert_array_equal(confidence_ids, ids)
    posterior = np.exp(reference - reference.max(axis=1, keepdims=True))
    np.testing.assert_allclose(confidence, posterior.max(axis=1) / posterior.sum(axis=1), atol=1e-3)
    # an offset has no slope: same classes; the input layout does not matter
    np.testing.assert_array_equal(model.predict(x + 0.3, columns), reference_scores(package, x + 0.3).argmax(axis=1))
    wide = np.zeros((x.shape[0], 20), np.float32)
    order = rng.permutation(20)[:WAVELENGTHS.size]
    wide[:, order] = x
    np.testing.assert_array_equal(model.predict(wide, order.astype(np.intp)), ids)
    # resampled onto the training bands (another calibration): still the chain on the training bands
    binding, how = model.bind(WAVELENGTHS + 1.2, WAVELENGTHS.size)
    assert 'resampled' in how
    shifted = binding.apply(x)
    np.testing.assert_array_equal(model.predict(x, binding), reference_scores(package, shifted).argmax(axis=1))
    # a pixel without a finite spectrum is background, confidence 0
    bad = x[:3].copy()
    bad[0, 2], bad[1, 7] = np.nan, np.inf
    with np.errstate(all='raise'):
        ids_bad, confidence_bad = model.predict_confidence(bad, columns)
        assert model.predict(bad, columns)[:2].tolist() == [0, 0]
    assert ids_bad[:2].tolist() == [0, 0] and confidence_bad[:2].tolist() == [0.0, 0.0]


def test_derivative_package_that_is_not_the_implemented_one_is_refused(tmp_path):
    def parameters(**changes):
        def change(manifest, files):
            manifest['pipeline_audit']['steps'][0]['parameters'].update(changes)
        return change

    def implementation(manifest, files):
        manifest['pipeline_audit']['steps'][0]['implementation'] = 'other.derivative'

    def missing_class(manifest, files):
        reference = json.loads(files['training_spectral_reference.json'])
        del reference['domains']['preprocessed_wavelength_selected']['classes'][2]
        files['training_spectral_reference.json'] = json.dumps(reference).encode()

    def no_raw(manifest, files):
        reference = json.loads(files['training_spectral_reference.json'])
        del reference['domains']['raw_wavelength_selected']
        files['training_spectral_reference.json'] = json.dumps(reference).encode()

    cases = (('window', parameters(window=7), 'not the one implemented'),        # stated != trained
             ('delta', parameters(delta=1.0), 'not the one implemented'),
             ('even', parameters(window=4), 'odd window'),
             ('wide', parameters(window=13), 'odd window'),
             ('order', parameters(polyorder=5), 'polyorder'),
             ('order0', parameters(polyorder=0), 'polyorder'),
             ('float', parameters(window=5.5), 'whole window'),
             ('units', parameters(units='per_band'), 'per_band'),
             ('inf', parameters(window=float('inf')), 'whole window'),
             ('deriv2', parameters(deriv=2), 'deriv'),
             ('nearest', parameters(mode='nearest'), 'mode'),
             ('missing', missing_class, 'no 12-band spectrum of class 2'),
             ('impl', implementation, 'other.derivative'))
    for name, change, message in cases:
        with pytest.raises(ValueError, match=message):
            load_package(build_package(tmp_path, name=name, steps=('first_derivative',), change=change))
    with pytest.raises(ValueError, match='not the one implemented'):            # other end handling
        load_package(build_package(tmp_path, name='mirror', steps=('first_derivative',), derivative_mode='mirror'))
    for steps in (('snv', 'first_derivative'), ('first_derivative', 'snv'), ('first_derivative', 'first_derivative')):
        with pytest.raises(ValueError, match='not supported'):
            load_package(build_package(tmp_path, name='-'.join(steps), steps=steps))
    package = load_package(build_package(tmp_path, name='noraw', steps=('first_derivative',), change=no_raw))
    assert any('first derivative NOT verified' in text for text in package.checks)
    # no delta stated (HSI MLOps schema 2.0): the uniform grid's spacing, still checked by the reference
    package = load_package(build_package(tmp_path, name='nodelta', steps=('first_derivative',),
                                         change=parameters(delta=None)))
    assert package.derivative_parameters['delta'] == 5.0
    assert any('1st derivative of the raw means reproduced' in text for text in package.checks)


def test_package_binds_every_training_band(package_path):
    """SNV runs over all training bands, so the input must supply all of them, not the selected ones."""
    model = load_classifier(package_path)
    wider = 890.0 + 5.0 * np.arange(16)
    columns, how = model.bind(wider + 0.2, wider.size)
    assert columns.tolist() == list(range(2, 14)) and 'max offset 0.200 nm' in how
    with pytest.raises(ValueError, match='no input band'):                 # a cut wavelength range
        model.bind(WAVELENGTHS[:9], 9)
    with pytest.raises(ValueError, match='no input band'):                 # another calibration, refused
        model.bind(WAVELENGTHS + 1.2, WAVELENGTHS.size, resample=False)
    binding, how = model.bind(WAVELENGTHS + 1.2, WAVELENGTHS.size)         # ... or resampled (the default)
    assert 'resampled' in how and binding.n_bands == WAVELENGTHS.size and 'SNV' in how
    with pytest.raises(ValueError, match='one to one'):                    # half the spectral resolution
        model.bind(WAVELENGTHS[::2], 6, tolerance_nm=6.0)
    columns, how = model.bind(None, WAVELENGTHS.size)
    assert columns.tolist() == list(range(12)) and 'NOT verified' in how
    with pytest.raises(ValueError, match='trained on 12'):
        model.bind(None, 11)


def test_package_pixel_without_a_spectrum_is_background(package_path):
    """A constant, NaN or infinite spectrum has no SNV (the pipeline's warm-up frame is all zero)."""
    model = load_classifier(package_path)
    columns, _ = model.bind(WAVELENGTHS, WAVELENGTHS.size)
    x, _ = training_spectra(np.random.default_rng(3), n=4)
    x = x[[4, 5, 6, 7, 8, 9]].astype(np.float32)                           # Leaf and Wood pixels
    good = model.predict(x, columns)
    assert set(good.tolist()) <= {1, 2}
    bad = x.copy()
    bad[0, 2] = np.nan
    bad[1] = 0.5
    bad[2, 5] = np.inf
    with np.errstate(all='raise'):
        assert model.predict(np.zeros((5, WAVELENGTHS.size), np.uint16), columns).tolist() == [0] * 5
        ids = model.predict(bad, columns)
        confidence_ids, confidence = model.predict_confidence(bad, columns)
    assert ids.tolist() == [0, 0, 0] + good[3:].tolist()
    np.testing.assert_array_equal(confidence_ids, ids)
    assert confidence[:3].tolist() == [0.0, 0.0, 0.0] and np.all(confidence[3:] > 0.5)

    def no_background(manifest, files):
        manifest['class_background'] = {}
    plain = load_classifier(build_package(Path(model.path).parent, name='nobg', change=no_background))
    ids = plain.predict(bad, columns)                                      # then: the class of the zero vector
    assert ids[0] == ids[1] == ids[2] and ids[3:].tolist() == good[3:].tolist()
    assert ids[0] == plain._scores(bad, columns)[0][0].argmax()


def test_package_classifier_is_shared_by_threads(package_path):
    import threading
    model = load_classifier(package_path)
    columns, _ = model.bind(WAVELENGTHS, WAVELENGTHS.size)
    x, _ = training_spectra(np.random.default_rng(9), n=200)
    x = x.astype(np.float32)
    expected = model.predict(x, columns)
    wrong = []

    def work():
        for _ in range(50):
            if not np.array_equal(model.predict(x, columns), expected):
                wrong.append(1)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not wrong


# ------------------------------------------------------------------------------ refusals ---

def test_package_that_cannot_be_run_faithfully_is_refused(tmp_path):
    def other_step(manifest, files):
        manifest['pipeline_audit']['steps'].append({'name': 'msc'})

    def other_algorithm(manifest, files):
        manifest['algorithm'] = 'SVM'

    def with_pca(manifest, files):
        manifest['pipeline_audit']['pca'] = {'n_components': 3}

    def percent(manifest, files):
        manifest['input_contract']['units'] = 'percent'

    def wrong_checksum(manifest, files):
        manifest['artifact_sha256'] = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
        manifest['artifact_sha256']['model.joblib'] = '0' * 64

    def other_reference(manifest, files):
        reference = json.loads(files['training_spectral_reference.json'])
        row = reference['domains']['preprocessed_wavelength_selected']['classes'][1]
        row['mean'] = row['mean'][1:] + row['mean'][:1]                    # still sums to 0: another band order
        files['training_spectral_reference.json'] = json.dumps(reference).encode()

    for name, change, message in (('step', other_step, 'msc'), ('algo', other_algorithm, 'SVM'),
                                  ('pca', with_pca, 'pca'), ('percent', percent, 'percent'),
                                  ('sha', wrong_checksum, 'sha256'), ('ref', other_reference, 'training reference')):
        with pytest.raises(ValueError, match=message):
            load_package(build_package(tmp_path, name=name, change=change))
    with pytest.raises(ValueError, match='population SD'):                 # SNV with the sample SD: another chain
        load_package(build_package(tmp_path, name='ddof', ddof=1))
    (tmp_path / 'text.hsipkg').write_text('not a zip')
    with pytest.raises(ValueError, match='not a model package'):
        load_package(tmp_path / 'text.hsipkg')


def test_package_pickle_runs_no_code(tmp_path):
    """The joblib files are pickles: anything but numpy arrays and hsi_lab_core objects is refused."""
    marker = tmp_path / 'ran'

    class Payload:
        def __reduce__(self):
            return (Path.touch, (marker,))

    def poison(manifest, files):
        files['model.joblib'] = pickle.dumps(Payload(), protocol=4)

    with pytest.raises(ValueError, match='refused'):
        load_package(build_package(tmp_path, name='poison', change=poison))
    assert not marker.exists()

    def nested(manifest, files):                                           # inside an hsi_lab_core object too
        files['scaler.joblib'] = dump_as({'mean_': np.zeros(6), 'scale_': np.ones(6), 'hook': Payload()},
                                         'hsi_lab_core.features.scaling', 'StandardFeatureScaler')

    with pytest.raises(ValueError, match='refused'):
        load_package(build_package(tmp_path, name='nested', change=nested))
    assert not marker.exists()

    def object_array(manifest, files):         # joblib would read an object array with a plain pickle.load
        files['scaler.joblib'] = dump_as({'mean_': np.array([Payload()] * 6, dtype=object), 'scale_': np.ones(6)},
                                         'hsi_lab_core.features.scaling', 'StandardFeatureScaler')

    with pytest.raises(ValueError, match='refused'):
        load_package(build_package(tmp_path, name='objects', change=object_array))
    assert not marker.exists()


# ------------------------------------------------------------------------- in the pipeline ---

def test_pipeline_classifies_with_a_package(tmp_path, package_path):
    """model.path may be a package: the pipeline binds it by wavelength and counts its classes."""
    from pipeline import build_pipeline
    from test_simulation import run_to_end, write_envi
    rng = np.random.default_rng(13)
    x, _ = training_spectra(rng, n=40)
    cube = x.astype(np.float32).reshape(8, 15, WAVELENGTHS.size)           # 8 slit pixels, 15 lines
    write_envi(tmp_path, cube, 'bip', wavelengths=WAVELENGTHS)
    config = replace(SimulationConfig(base_dir=str(tmp_path)),
                     source=SourceConfig(path=str(tmp_path / 'cube.bip.hdr'), cache='ram'),
                     camera=CameraConfig(fps=0, overrun='block', buffer_frames=4, loop=False),
                     corrections=CorrectionConfig(backend='cpu', reflectance='none'),
                     processing=ProcessingConfig(workers=2, profile_every_s=0),
                     model=ModelConfig(path=str(package_path))).validate()
    pipeline = build_pipeline(config)
    assert isinstance(pipeline.classifier, HsiPackageClassifier)
    assert 'MLC 3 classes' in pipeline.plans[0].provenance['model bands'][0]
    s = run_to_end(pipeline)
    assert s.state == 'finished' and s.lines_processed == 15 and s.lines_lost == 0
    columns, _ = pipeline.classifier.bind(WAVELENGTHS, WAVELENGTHS.size)
    expected = np.bincount(pipeline.classifier.predict(cube.reshape(-1, WAVELENGTHS.size), columns), minlength=256)
    np.testing.assert_array_equal(s.counts, expected)
    assert expected[0] and expected[1] and expected[2]


def test_mlc_reg_covar_widens_the_covariances_and_is_off_by_default(tmp_path, package_path):
    """model.mlc_reg_covar adds to the diagonal of every class covariance; 0 is the trained model."""
    from pipeline import build_pipeline
    from test_simulation import write_envi
    trained = load_classifier(package_path)
    assert trained.mlc_reg_covar == 0.0 and trained.chain == trained.package.chain
    model = load_classifier(package_path, 0.25)
    columns, how = model.bind(WAVELENGTHS, WAVELENGTHS.size)
    assert 'covariance + 0.25' in how and 'not as trained' in how
    rng = np.random.default_rng(3)
    x, y = training_spectra(rng, n=600)
    x = (x + rng.normal(0.0, 0.01, x.shape)).astype(np.float32)             # noisier than the training
    widened = replace(model.package, covariances=model.package.covariances + 0.25 * np.eye(len(SELECTED)))
    reference = reference_scores(widened, x)
    np.testing.assert_array_equal(model.predict(x, columns), reference.argmax(axis=1))
    ids, confidence = model.predict_confidence(x, columns)
    np.testing.assert_array_equal(ids, reference.argmax(axis=1))
    posterior = np.exp(reference - reference.max(axis=1, keepdims=True))
    np.testing.assert_allclose(confidence, posterior.max(axis=1) / posterior.sum(axis=1), atol=1e-3)
    with pytest.raises(ValueError, match='mlc_reg_covar'):
        load_classifier(package_path, -0.1)
    # joblib artifacts ignore it; the configuration carries it to the pipeline's classifier
    for bad in (-1.0, 'big', True, float('nan')):
        with pytest.raises(ValueError, match='mlc_reg_covar'):
            replace(SimulationConfig(base_dir=str(tmp_path)), model=ModelConfig(mlc_reg_covar=bad)).validate()
    write_envi(tmp_path, x[:120].reshape(8, 15, WAVELENGTHS.size), 'bip', wavelengths=WAVELENGTHS)
    config = replace(SimulationConfig(base_dir=str(tmp_path)),
                     source=SourceConfig(path=str(tmp_path / 'cube.bip.hdr'), cache='ram'),
                     camera=CameraConfig(fps=0, overrun='block', buffer_frames=4, loop=False),
                     corrections=CorrectionConfig(backend='cpu', reflectance='none'),
                     processing=ProcessingConfig(workers=1, profile_every_s=0),
                     model=ModelConfig(path=str(package_path), mlc_reg_covar=0.25)).validate()
    pipeline = build_pipeline(config)
    assert pipeline.classifier.mlc_reg_covar == 0.25
    assert 'covariance + 0.25' in pipeline.plans[0].provenance['model bands'][0]


# ------------------------------------------------------------- the delivered packages ---

@pytest.mark.parametrize('path, classes, bands', [(REAL_PACKAGES / 'Kratom_adulteration_1794300998.zip', 4, 137),
                                                  (REAL_PACKAGES / 'MLC_874171_1794299876.zip', 5, 101),
                                                  (ROOT / 'models' / 'MLC_869291_1794818878.hsipkg', 3, 159)])
def test_delivered_package_loads_and_meets_the_camera_grid(tmp_path, path, classes, bands):
    """The models delivered for this project (skipped where the files are not present)."""
    if not path.is_file():
        pytest.skip(f"{path} is not on this machine")
    model = load_classifier(path)
    assert model.classes_.size == classes and model.package.selected.size == bands and model.n_bands == 159
    assert any(text.startswith('training reference ok') for text in model.package.checks)
    # the live GenICam grid (bin 2, 5 nm, 900-1700 nm): 159 bands at 904.65-1695.29 nm
    meta = {'camera_bin': 2, 'm_nm_per_px_bin1': 0.36385766843509226, 'c_nm': 842.1014795620235,
            'resolution_nm': 5.0, 'wavelength_range_bands': [12, 171, 186]}
    plan = build_plan(SimulationConfig(base_dir=str(tmp_path)).validate(),
                      FrameSource('g.h5', np.zeros((1, 4, 1280), np.uint16), 'raw', metadata=meta))
    offset = np.abs(plan.wavelengths - model.band_wavelengths).max()
    if offset <= 0.5:
        columns, how = model.bind(plan.wavelengths, plan.n_columns, 0.5)
        assert columns.tolist() == list(range(159))
    else:                                              # trained on another wavelength calibration
        with pytest.raises(ValueError, match='no input band'):                 # refused one to one ...
            model.bind(plan.wavelengths, plan.n_columns, 0.5, resample=False)
        inside = plan.wavelengths.min() <= model.band_wavelengths.min() \
            and model.band_wavelengths.max() <= plan.wavelengths.max()
        if inside:                                                             # ... or resampled
            binding, how = model.bind(plan.wavelengths, plan.n_columns, 0.5)
            assert 'resampled' in how and binding.n_bands == 159
