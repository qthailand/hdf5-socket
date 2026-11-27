"""Regression packages (e.g. moisture): build / read round trip, the three engines, the pipeline's
value map and statistics, the Setup tab block, and the GUI's switch between class and value display."""
import io
import json
import sys
import time
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import regression  # noqa: E402
from hsi_package import load_package, regression_predict  # noqa: E402
from inference_engine import HsiRegressor, load_classifier  # noqa: E402
from regression_builder import build_regression_package, read_samples  # noqa: E402
from regression_engines import CpuEngine, CupyEngine, OpenCVEngine, RegressionRuntime, self_test  # noqa: E402
from sim_config import CameraConfig, CorrectionConfig, ModelConfig, ProcessingConfig, SimulationConfig, \
    SourceConfig  # noqa: E402
from test_hsi_package import WAVELENGTHS, build_package  # noqa: E402
from test_simulation import run_to_end, write_envi  # noqa: E402

BELT_MEAN = 0.2              # object mask of the test packages: the belt (mean 0.08) is background


def moisture_spectra(rng, n=300, belt=0):
    """Reflectance whose water band (around 930 nm here) deepens with the moisture, plus ``belt``
    spectra of a dark flat belt (no object). Returns ``(spectra, moisture or NaN for the belt)``."""
    x = np.linspace(0.0, 1.0, WAVELENGTHS.size)
    moisture = rng.uniform(5.0, 60.0, n)
    water = np.exp(-0.5 * ((WAVELENGTHS - 930.0) / 12.0) ** 2)
    gain = 1.0 + 0.15 * rng.standard_normal((n, 1))                    # brightness: SNV removes it
    spectra = gain * (0.55 + 0.1 * np.sin(4 * x) - 0.004 * moisture[:, None] * water)
    spectra += 0.001 * rng.standard_normal(spectra.shape)
    if belt:
        flat = 0.08 + 0.02 * x + 0.002 * rng.standard_normal((belt, x.size))
        spectra = np.vstack([spectra, flat])
        moisture = np.concatenate([moisture, np.full(belt, np.nan)])
    return spectra, moisture


def sklearn_chain(preprocessing, x, y, components=None, selected=None, algorithm='PLSR'):
    """What the training software predicts (sklearn on the preprocessed spectra)."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from regression_builder import preprocess
    z, _, _ = preprocess(x, preprocessing, WAVELENGTHS, {'window': 5})
    z = z if selected is None else z[:, selected]
    scaler = StandardScaler().fit(z)
    model = (PLSRegression(n_components=components) if algorithm == 'PLSR' else Ridge(alpha=1.0))
    model.fit(scaler.transform(z), y)
    return lambda spectra: np.ravel(model.predict(scaler.transform(
        (preprocess(spectra, preprocessing, WAVELENGTHS, {'window': 5})[0])[:, slice(None) if selected is None
                                                                                else selected])))


def make_package(folder, name='moisture', preprocessing='snv', **kwargs):
    x, y = moisture_spectra(np.random.default_rng(3))
    kwargs.setdefault('object_min_mean', BELT_MEAN)
    path = build_regression_package(x, y, WAVELENGTHS, Path(folder) / f'{name}.hsipkg', preprocessing=preprocessing,
                                    valid_range=(0.0, 70.0), display_range=(5.0, 60.0),
                                    derivative={'window': 5}, **kwargs)
    return path, x, y


def edit_package(path, change):
    """Rewrite ``path`` after ``change(manifest, files)`` (checksums updated, like a re-export)."""
    import hashlib
    with zipfile.ZipFile(path) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(files.pop('manifest.json'))
    change(manifest, files)
    manifest['artifact_sha256'] = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('manifest.json', json.dumps(manifest))
        for name, blob in files.items():
            archive.writestr(name, blob)


# --------------------------------------------------------------------- package round trip ---

@pytest.mark.parametrize('preprocessing,selected,algorithm', [
    ('snv', None, 'PLSR'), ('first_derivative', None, 'PLSR'), ('none', [0, 2, 3, 5, 7, 8, 10, 11], 'PLSR'),
    ('snv', None, 'Ridge')])
def test_package_round_trip_predicts_as_sklearn(tmp_path, preprocessing, selected, algorithm):
    """Built, read back without running any pickled code, and folded into one weight per band: the
    predictions are sklearn's own (float32 kernel: to 1e-4 of the target range)."""
    x, y = moisture_spectra(np.random.default_rng(3))
    components = 3 if algorithm == 'PLSR' else None
    path = build_regression_package(x, y, WAVELENGTHS, tmp_path / 'm.hsipkg', preprocessing=preprocessing,
                                    selected=selected, algorithm=algorithm, components=components,
                                    derivative={'window': 5}, target_name='moisture', units='%')
    package = load_package(path)
    assert package.is_regression and package.task == 'regression' and package.algorithm == algorithm
    assert package.target.name == 'moisture' and package.target.units == '%'
    assert package.input_units == 'reflectance'
    assert any('check spectra predicted' in c for c in package.checks)
    assert any('scaler mean / variance reproduced' in c for c in package.checks)
    if preprocessing == 'first_derivative':
        assert any('1st derivative of the raw means reproduced' in c for c in package.checks)
    expected = sklearn_chain(preprocessing, x, y, components, selected, algorithm)(x)
    np.testing.assert_allclose(regression_predict(package, x), expected, rtol=0, atol=1e-9 * np.ptp(y))
    model = load_classifier(path)
    assert isinstance(model, HsiRegressor) and model.task == 'regression'
    columns, binding = model.bind(WAVELENGTHS, WAVELENGTHS.size)
    assert 'moisture' in binding
    values, valid = model.predict_values(x.astype(np.float32), columns)
    assert valid.all()
    np.testing.assert_allclose(values, expected, rtol=0, atol=1e-4 * np.ptp(y))
    assert 'R²' in package.metric and 'RMSECV' in package.metric
    assert 'moisture (%)' in package.describe()
    training = package.training['input']
    assert [t.label.split()[0] for t in training] == ['low', 'mid', 'high']
    assert training[0].target_mean < training[1].target_mean < training[2].target_mean


def test_cross_validation_chooses_few_components(tmp_path):
    path, x, y = make_package(tmp_path)
    package = load_package(path)
    assert 1 <= package.regression.components <= 6
    assert package.regression.stats['rmsecv'] < 0.1 * np.ptp(y)


def test_csv_samples_are_read(tmp_path):
    x, y = moisture_spectra(np.random.default_rng(1), n=20)
    csv = tmp_path / 's.csv'
    rows = [','.join(f"{w:g}" for w in WAVELENGTHS) + ',moisture']
    rows += [','.join(f"{v:.6f}" for v in row) + f",{t:.3f}" for row, t in zip(x, y)]
    csv.write_text('\n'.join(rows))
    spectra, target, wavelengths, name = read_samples(csv)
    assert name == 'moisture' and spectra.shape == x.shape
    np.testing.assert_allclose(wavelengths, WAVELENGTHS)
    np.testing.assert_allclose(target, y, atol=1e-3)


def test_regression_package_that_is_not_the_implemented_one_is_refused(tmp_path):
    path, _, _ = make_package(tmp_path)

    def broken(change):
        copy = tmp_path / f'broken{len(list(tmp_path.iterdir()))}.hsipkg'
        copy.write_bytes(path.read_bytes())
        edit_package(copy, change)
        return copy

    def wrong_check(manifest, files):
        reference = json.loads(files['training_spectral_reference.json'])
        reference['regression']['check']['predictions'][0] += 1.0
        files['training_spectral_reference.json'] = json.dumps(reference).encode()
    with pytest.raises(ValueError, match='check spectra'):
        load_package(broken(wrong_check))

    with pytest.raises(ValueError, match='target'):
        load_package(broken(lambda m, f: m.pop('target')))
    with pytest.raises(ValueError, match='not supported'):
        load_package(broken(lambda m, f: m.update(algorithm='GradientBoosting')))

    def two_responses(manifest, files):
        from sklearn.cross_decomposition import PLSRegression
        import joblib
        rng = np.random.default_rng(0)
        pls = PLSRegression(2).fit(rng.random((30, WAVELENGTHS.size)), rng.random((30, 2)))
        buffer = io.BytesIO()
        joblib.dump(pls, buffer)
        files['model.joblib'] = buffer.getvalue()
    with pytest.raises(ValueError, match='one response'):
        load_package(broken(two_responses))

    def foreign_object(manifest, files):
        from sklearn.ensemble import RandomForestRegressor
        import joblib
        buffer = io.BytesIO()
        joblib.dump(RandomForestRegressor(n_estimators=1).fit(np.ones((3, 2)), np.ones(3)), buffer)
        files['model.joblib'] = buffer.getvalue()
    with pytest.raises(ValueError, match='refused'):
        load_package(broken(foreign_object))


# ------------------------------------------------- HSI MLOps export (schema 2.0, PLS-DA / regression) ---

MLOPS_WAVELENGTHS = 854.17 + 5.0 * np.arange(40)
REAL_REGRESSION = Path('C:/Users/TSC/Downloads/PLS-DA_436001-regression_1786941443/'
                       'PLS-DA_436001-regression_1786941443.hsipkg')


def mlops_regression_package(folder, steps=('savitzky_golay', 'first_derivative'), wavelengths=MLOPS_WAVELENGTHS):
    """A package as HSI MLOps exports a regression: algorithm 'PLS-DA' with task_type 'regression',
    target_col "Weight Loss (%)", no target block, no input_contract, no training reference,
    derivative steps without a delta (the grid's spacing), sklearn StandardScaler + PLSRegression.
    Returns ``(path, spectra, targets, sklearn predictions)``."""
    import hashlib
    import joblib
    from scipy.signal import savgol_filter
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(11)
    samples = rng.uniform(12.0, 42.0, 20)                          # 20 samples x 25 pixels, as the real one
    y = np.repeat(samples, 25)
    band = np.exp(-0.5 * ((wavelengths - 1450.0) / 40.0) ** 2)
    x = (0.4 + 0.1 * np.sin(wavelengths / 120.0) + 0.003 * y[:, None] * band) * (1 + 0.1 * rng.random((y.size, 1)))
    x += 0.002 * rng.standard_normal(x.shape)

    def chain(spectra):
        z = np.asarray(spectra, np.float64)
        if 'savitzky_golay' in steps:
            z = savgol_filter(z, 11, 2, axis=1)
        return savgol_filter(z, 11, 2, deriv=1, delta=5.0, axis=1)
    scaler = StandardScaler().fit(chain(x))
    pls = PLSRegression(n_components=4).fit(scaler.transform(chain(x)), y)

    def dump(obj):
        buffer = io.BytesIO()
        joblib.dump(obj, buffer)
        return buffer.getvalue()
    files = {'model.joblib': dump(pls), 'scaler.joblib': dump(scaler)}
    manifest = {
        'name': 'Background', 'algorithm': 'PLS-DA', 'task_type': 'regression', 'labels': [],
        'pipeline_audit': {
            'schema_version': '2.0',
            'spectral_axis': {'units': 'nm', 'median_spacing_nm': 5.0, 'derivative_units': 'per_nm'},
            'steps': [{'name': name, 'parameters': {'window': 11, 'polyorder': 2}} for name in steps],
            'pca': None, 'final_scaler': {'type': 'StandardScaler', 'with_mean': True, 'with_std': True},
            'estimator': 'PLSRegression'},
        'wavelengths': list(map(float, wavelengths)), 'metric_name': '4-Fold R²', 'metric_value': 0.52,
        'target_col': 'Weight Loss (%)', 'exported_at': '2026-08-17T04:37:23',
        'artifact_sha256': {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}}
    path = Path(folder) / 'mlops_regression.hsipkg'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('manifest.json', json.dumps(manifest))
        for name, blob in files.items():
            archive.writestr(name, blob)
    return path, x, y, np.ravel(pls.predict(scaler.transform(chain(x))))


def test_hsi_mlops_regression_export_is_read_and_runs_as_sklearn(tmp_path):
    path, x, y, expected = mlops_regression_package(tmp_path)
    package = load_package(path)
    assert package.is_regression and package.algorithm == 'PLSR' and package.regression.components == 4
    assert package.target.name == 'Weight Loss' and package.target.units == '%'
    # no valid range in the manifest: the training targets the PLS scores give back, widened by 10 %
    stats = package.regression.stats
    assert stats['n'] == y.size and stats['target_values'] == 20
    assert stats['target_min'] == pytest.approx(y.min()) and stats['target_max'] == pytest.approx(y.max())
    assert package.target.display_range == pytest.approx((y.min(), y.max()))
    pad = 0.1 * np.ptp(y)
    assert package.target.valid_range == pytest.approx((y.min() - pad, y.max() + pad))
    assert package.input_units == 'reflectance' and not package.input_units_declared
    assert any('training targets recovered' in c for c in package.checks)
    assert any('NOT verified' in c for c in package.checks)              # no training reference
    assert '4-Fold R² = 0.52' in package.metric and 'fit R²' in package.metric
    assert 'SG smoothing 11/2 -> 1st derivative' in package.chain
    # smoothing, then the derivative of the smoothed spectrum: sklearn's predictions on scipy's chain
    np.testing.assert_allclose(regression_predict(package, x), expected, rtol=0, atol=1e-9 * np.ptp(y))
    model = load_classifier(path)
    assert not model.input_units_declared
    columns, _ = model.bind(MLOPS_WAVELENGTHS, MLOPS_WAVELENGTHS.size)
    values, valid = model.predict_values(x.astype(np.float32), columns)
    assert valid.all()
    np.testing.assert_allclose(values, expected, rtol=0, atol=1e-4 * np.ptp(y))
    # (a derivative-only reading of the two steps would miss sklearn's predictions above by far more
    # than 1e-9); an export with the derivative step alone is read as that
    (tmp_path / 'one').mkdir()
    alone, x1, _, expected1 = mlops_regression_package(tmp_path / 'one', ('first_derivative',))
    assert 'SG smoothing' not in load_package(alone).chain
    np.testing.assert_allclose(regression_predict(load_package(alone), x1), expected1, atol=1e-9 * np.ptp(y))


def test_derivative_without_delta_needs_a_uniform_grid(tmp_path):
    uneven = MLOPS_WAVELENGTHS + np.where(np.arange(MLOPS_WAVELENGTHS.size) > 20, 0.7, 0.0)
    path, *_ = mlops_regression_package(tmp_path, wavelengths=uneven)
    with pytest.raises(ValueError, match='uniform grid'):
        load_package(path)


@pytest.mark.skipif(not REAL_REGRESSION.is_file(), reason='the delivered example regression package is not here')
def test_delivered_regression_package_loads_and_binds_to_its_grid():
    package = load_package(REAL_REGRESSION)
    assert package.is_regression and package.algorithm == 'PLSR' and package.regression.components == 10
    assert package.target.name == 'Weight Loss' and package.target.units == '%'
    assert package.wavelengths.size == 185 and package.derivative_parameters['delta'] == 5.0
    assert package.derivative_parameters['smooth'] == {'window': 11, 'polyorder': 2}
    stats = package.regression.stats
    assert stats['n'] == 10000 and stats['target_values'] == 20
    assert stats['target_min'] == pytest.approx(12.651, abs=1e-3) and stats['target_max'] == pytest.approx(41.511, abs=1e-3)
    model = load_classifier(REAL_REGRESSION)
    columns, binding = model.bind(package.wavelengths, package.wavelengths.size)
    assert 'max offset 0.000 nm' in binding
    rng = np.random.default_rng(0)
    x = (0.35 + 0.05 * rng.random((256, 185))).astype(np.float32)
    values, valid = model.predict_values(x, columns)
    assert valid.all()
    np.testing.assert_allclose(values, regression_predict(package, x.astype(np.float64)), atol=1e-4)


def test_classification_packages_are_unchanged(tmp_path):
    package = load_package(build_package(tmp_path, name='classes'))
    assert package.task == 'classification' and not package.is_regression and package.regression is None
    assert load_classifier(package.path).task == 'classification'


# -------------------------------------------------------------------------------- engines ---

def reference_line(x, columns, weights, bias, snv):
    z = np.asarray(x, np.float64)[:, columns]
    means = z.mean(axis=1)
    if snv:
        z = z - means[:, None]
        with np.errstate(invalid='ignore', divide='ignore'):
            z = z / np.sqrt((z ** 2).mean(axis=1, keepdims=True))
    return z @ weights + bias, means


def engines():
    from cuda_emulation import fake_cupy, fake_cv2
    found = [('cpu', CpuEngine()), ('cupy (emulated)', CupyEngine(fake_cupy())),
             ('opencv_cuda (emulated)', OpenCVEngine(fake_cv2()))]
    from modules.compute_runtime import get_compute_capabilities
    caps = get_compute_capabilities()
    if caps.cupy_cuda.available:
        found.append(('cupy_cuda', CupyEngine()))
    if caps.opencv_cuda.available:
        try:
            engine = OpenCVEngine()
            self_test(engine)
            found.append(('opencv_cuda', engine))
        except Exception:                          # OpenCV built without cuBLAS: no gemm
            pass
    return found


@pytest.mark.parametrize('name,engine', engines(), ids=lambda v: v if isinstance(v, str) else '')
@pytest.mark.parametrize('snv', [False, True])
def test_engines_predict_as_the_float64_reference(name, engine, snv):
    rng = np.random.default_rng(5)
    x = (0.3 + 0.4 * rng.random((300, 40))).astype(np.float32)
    x[4, 7] = np.nan                                              # a bad band: no value
    x[9, :] = 0.5                                                 # constant: invalid with SNV only
    columns = np.sort(rng.choice(40, 33, replace=False)).astype(np.intp)
    weights = rng.standard_normal(33)
    values, means, valid = engine.predict(x, columns, weights, 12.5, snv)
    expected, expected_means = reference_line(x, columns, weights, 12.5, snv)
    good = np.isfinite(expected)
    np.testing.assert_array_equal(valid, good)
    assert not valid[4] and valid[9] != snv
    assert np.isnan(values[~valid]).all()
    spread = np.ptp(expected[good])
    np.testing.assert_allclose(values[valid], expected[valid], rtol=0, atol=2e-4 * spread)
    np.testing.assert_allclose(means[valid], expected_means[valid], rtol=1e-5)


def test_runtime_falls_back_in_hsi_order_and_after_a_failure():
    from cuda_emulation import fake_cupy, fake_cv2
    cpu = RegressionRuntime('auto')
    assert cpu.name == 'cpu' and cpu.fallback_reason == ''
    gpu = RegressionRuntime('cupy_cuda', modules={'cupy': fake_cupy(), 'cv2': fake_cv2()})
    assert gpu.name == 'cupy_cuda' and gpu.fallback_reason == ''
    # CuPy missing: the other GPU engine, then the CPU, each with its reason
    other = RegressionRuntime('cupy_cuda', modules={'cupy': object(), 'cv2': fake_cv2()})
    assert other.name == 'opencv_cuda' and 'cupy_cuda' in other.fallback_reason
    none = RegressionRuntime('opencv_cuda', modules={'cupy': object(), 'cv2': object()})
    assert none.name == 'cpu' and 'opencv_cuda' in none.fallback_reason and 'cupy_cuda' in none.fallback_reason

    class Breaks(CupyEngine):
        def predict(self, *args):
            raise RuntimeError('device lost')
    gpu.engine = Breaks(fake_cupy())
    x = np.random.default_rng(0).random((16, 8)).astype(np.float32)
    values, _, valid = gpu.predict(x, np.arange(8), np.ones(8), 0.0, False)
    assert valid.all() and gpu.name == 'cpu' and 'device lost' in gpu.fallback_reason
    np.testing.assert_allclose(values, x.sum(axis=1), rtol=1e-5)


def test_regressor_uses_the_engine_it_is_given_and_the_object_mask(tmp_path):
    from cuda_emulation import fake_cupy
    path, _, _ = make_package(tmp_path)
    model = load_classifier(path)
    rng = np.random.default_rng(8)
    x, y = moisture_spectra(rng, n=50, belt=30)
    x = x.astype(np.float32)
    columns, _ = model.bind(WAVELENGTHS, WAVELENGTHS.size)
    values, valid = model.predict_values(x, columns)
    assert valid[:50].all() and not valid[50:].any()             # the belt is below the package's mask
    assert np.isnan(values[50:]).all()
    assert np.sqrt(np.mean((values[:50] - y[:50]) ** 2)) < 2.0
    gpu = RegressionRuntime('cupy_cuda', modules={'cupy': fake_cupy()})
    gpu_values, gpu_valid = model.predict_values(x, columns, gpu)
    np.testing.assert_array_equal(gpu_valid, valid)
    np.testing.assert_allclose(gpu_values[valid], values[valid], atol=1e-3)
    codes = model.predict(x, columns)
    assert codes.dtype == np.uint8 and (codes[50:] == regression.BACKGROUND_CODE).all()
    model.set_object_mask(None)                                   # every valid pixel is an object
    assert model.predict_values(x, columns)[1].all()
    bulk, bulk_valid = model.predict_spectra(x.reshape(10, 8, -1), columns)
    assert bulk.shape == (10, 8) and bulk_valid.all()


# ------------------------------------------------------------------ codes, stats, objects ---

def test_codes_summary_and_objects():
    value_range = (0.0, 50.8)
    values = np.array([0.0, 25.6, 50.8, 70.0, -3.0, np.nan], np.float32)
    valid = np.array([True, True, True, True, True, False])
    codes = regression.encode(values, valid, value_range)
    assert codes.tolist() == [0, 127, 253, 253, 0, regression.BACKGROUND_CODE]
    centres = regression.code_values(value_range)
    assert centres[0] == 0.0 and centres[-1] == pytest.approx(50.8)

    rng = np.random.default_rng(2)
    v = rng.normal(30.0, 4.0, 5000)
    stats = regression.line_stats(v, np.ones(v.size, bool), (0.0, 50.8))
    counts = np.bincount(regression.encode(v, np.ones(v.size, bool), (0.0, 50.8)), minlength=256)
    counts[regression.BACKGROUND_CODE] = 5000
    s = regression.summary(stats, counts, (0.0, 50.8), background=5000)
    assert s['n'] == 5000 and s['mean'] == pytest.approx(v.mean()) and s['sd'] == pytest.approx(v.std())
    assert s['median'] == pytest.approx(np.median(v), abs=0.15)
    assert s['p10'] == pytest.approx(np.percentile(v, 10), abs=0.15)
    assert s['object_share'] == pytest.approx(0.5) and s['out_of_range'] == 0.0

    canvas = np.full((40, 60), np.nan, np.float32)
    canvas[2:12, 3:20] = 10.0                                     # a dry kernel
    canvas[20:35, 30:55] = np.linspace(40, 50, 25)[None, :]       # a wet one
    canvas[0, 59] = 99.0                                          # a speck: too small
    found = regression.objects(canvas, min_pixels=20)
    assert [o['pixels'] for o in found] == [15 * 25, 10 * 17]
    assert found[0]['mean'] == pytest.approx(45.0, abs=0.01) and found[1]['median'] == 10.0
    assert found[0]['min'] == pytest.approx(40.0) and found[0]['max'] == pytest.approx(50.0)

    lut = regression.value_lut((0.0, 50.0), (10.0, 40.0))
    assert lut.shape == (256, 3) and lut[regression.BACKGROUND_CODE].tolist() == [0, 0, 0]
    assert lut[regression.LOST_CODE].tolist() == [255, 0, 255]
    assert lut[0].tolist() == lut[40].tolist()                    # below the display range: its first colour


# ------------------------------------------------------------------------------- pipeline ---

def regression_setup(tmp_path, fps=0.0, loop=False, spatial_bin=1):
    """A reflectance cube of produce on a belt (12 bands), a models folder with a moisture
    package and a classification package, a config running the moisture model."""
    models = tmp_path / 'models'
    models.mkdir(exist_ok=True)
    moisture, _, _ = make_package(models)
    classes = build_package(models, name='classes')
    x, y = moisture_spectra(np.random.default_rng(13), n=60, belt=60)
    cube = x.astype(np.float32).reshape(8, 15, WAVELENGTHS.size)
    write_envi(tmp_path, cube, 'bip', wavelengths=WAVELENGTHS)
    config = replace(SimulationConfig(base_dir=str(tmp_path)),
                     source=SourceConfig(path=str(tmp_path / 'cube.bip.hdr'), cache='ram'),
                     camera=CameraConfig(fps=fps, overrun='block', buffer_frames=8, loop=loop),
                     corrections=CorrectionConfig(backend='cpu', reflectance='none'),
                     processing=ProcessingConfig(workers=2, profile_every_s=0, spatial_bin=spatial_bin),
                     model=ModelConfig(path=str(moisture), folders=('models',))).validate()
    return config, moisture, classes, cube, y.reshape(8, 15)


def test_pipeline_draws_values_and_counts_exact_statistics(tmp_path):
    from pipeline import build_pipeline
    config, _, _, cube, truth = regression_setup(tmp_path)
    pipeline = build_pipeline(config)
    model = pipeline.classifier
    assert isinstance(model, HsiRegressor)
    assert 'PLSR -> moisture (%)' in pipeline.plans[0].provenance['model bands'][0]
    s = run_to_end(pipeline)
    assert s.state == 'finished' and s.lines_processed == cube.shape[1] and s.error is None
    assert s.task == 'regression' and s.regression_engine == 'cpu'
    columns, _ = model.bind(WAVELENGTHS, WAVELENGTHS.size)
    expected, valid = model.predict_values(cube.reshape(-1, WAVELENGTHS.size), columns)
    # the cube is (slit pixels, lines): one map column per line, its slit pixels down the rows
    expected, valid = expected.reshape(cube.shape[:2]), valid.reshape(cube.shape[:2])
    np.testing.assert_array_equal(np.isfinite(s.value_canvas), valid)
    np.testing.assert_allclose(s.value_canvas[valid], expected[valid], rtol=1e-6)
    np.testing.assert_array_equal(s.canvas, model.encode(expected, valid))
    v = expected[valid].astype(np.float64)
    assert s.value_stats[0] == v.size and s.value_stats[1] == pytest.approx(v.sum(), rel=1e-6)
    assert s.counts[regression.BACKGROUND_CODE] == (~valid).sum()
    summary = regression.summary(s.value_stats, s.counts, model.value_range, int(s.counts[regression.BACKGROUND_CODE]))
    assert summary['mean'] == pytest.approx(v.mean(), rel=1e-6)
    assert abs(summary['mean'] - np.nanmean(np.where(valid, truth, np.nan))) < 2.0
    # count delay: the statistics leave out the raw-zone lines like the class counts do
    delayed = pipeline.snapshot(count_delay_lines=3)
    kept = valid[:, :-3]
    assert delayed.value_stats[0] == kept.sum()
    pipeline.reset_counts()
    assert pipeline.snapshot().value_stats[0] == 0


def test_pipeline_switches_between_classification_and_regression(tmp_path):
    from pipeline import build_pipeline
    from test_model_switch import wait_for
    config, moisture, classes, cube, _ = regression_setup(tmp_path, fps=2000.0, loop=True)
    pipeline = build_pipeline(replace(config, model=replace(config.model, path=str(classes))))
    pipeline.start()
    try:
        assert wait_for(lambda: pipeline.snapshot().lines_processed > 60)
        s = pipeline.snapshot()
        assert s.task == 'classification' and s.value_canvas is None and s.value_stats is None
        generation = pipeline.set_classifier(load_classifier(moisture))
        assert wait_for(lambda: pipeline.snapshot().model_generation == generation
                        and pipeline.snapshot().counted_lines > 30)
        s = pipeline.snapshot()
        assert s.task == 'regression' and s.value_stats[0] > 0 and s.value_canvas is not None
        assert s.counts[:regression.LEVELS].sum() == s.value_stats[0]       # the history started again
        back = pipeline.set_classifier(load_classifier(classes))
        assert wait_for(lambda: pipeline.snapshot().model_generation == back
                        and pipeline.snapshot().counted_lines > 30)
        s = pipeline.snapshot()
        assert s.task == 'classification' and s.value_stats is None
        assert set(np.flatnonzero(s.counts).tolist()) <= {0, 1, 2}
    finally:
        assert pipeline.stop()


def test_regression_with_2x2_bin_and_a_gpu_engine_request(tmp_path):
    """The block's value goes to its 4 pixels; an unavailable GPU engine falls back to the CPU."""
    from pipeline import build_pipeline
    config, _, _, cube, _ = regression_setup(tmp_path, spatial_bin=2)
    config = replace(config, model=replace(config.model, regression_engine='cupy_cuda'))
    pipeline = build_pipeline(config)
    s = run_to_end(pipeline)
    assert s.state == 'finished' and s.task == 'regression'
    from modules.compute_runtime import get_compute_capabilities
    if not get_compute_capabilities().cuda_available:
        assert s.regression_engine == 'cpu' and 'cupy_cuda' in s.regression_engine_note
    finite = np.isfinite(s.value_canvas)
    assert finite.any()
    np.testing.assert_array_equal(s.value_canvas[0::2][:4], s.value_canvas[1::2][:4])   # slit row pairs
    with pytest.raises(ValueError, match='regression_engine'):
        replace(config, model=replace(config.model, regression_engine='tpu')).validate()


def test_setup_tab_model_block_for_a_regression(tmp_path):
    from pipeline import build_pipeline
    from pipeline_overview import describe
    config, _, _, _, _ = regression_setup(tmp_path)
    pipeline = build_pipeline(config)
    s = run_to_end(pipeline)
    _, blocks = describe(pipeline, s)
    block = next(b for b in blocks if b.key == 'model')
    rows = dict(block.rows)
    assert block.line2 == 'moisture (%) · PLSR' and rows['Valid range'] == '0 – 70 %'
    assert rows['Object mask'] == f"mean input ≥ {BELT_MEAN:g}" and rows['Engine'] == 'cpu'
    assert block.state == 'error' and 'reflectance' in block.note                # input is counts here
    # with the units right, many objects outside the valid range warn
    pipeline.classifier.input_units = 'counts'
    s.value_stats = np.array([2000.0, 2000.0 * 80.0, 0.0, 400.0])
    block = next(b for b in describe(pipeline, s)[1] if b.key == 'model')
    assert block.state == 'warn' and '20 %' in block.note and 'outside 0 – 70' in block.note
    s.value_stats = np.array([2000.0, 2000.0 * 30.0, 0.0, 10.0])
    block = next(b for b in describe(pipeline, s)[1] if b.key == 'model')
    assert block.state == 'ok' and 'mean 30.0 %' in dict(block.rows)['Now']


def test_catalog_describes_a_regression_model(tmp_path):
    from model_catalog import scan_models
    from pipeline import build_pipeline
    config, _, _, _, _ = regression_setup(tmp_path)
    pipeline = build_pipeline(config)
    entries = {e.path.name: e for e in scan_models(config, pipeline.plans, pipeline.classifier)}
    assert entries['moisture.hsipkg'].usable and entries['moisture.hsipkg'].kind == 'PLSR · moisture (%)'
    assert 'valid 0 – 70' in entries['moisture.hsipkg'].detail
    assert entries['classes.hsipkg'].kind == 'MLC · 3 classes'


# ------------------------------------------------------------------------------------ GUI ---

@pytest.fixture
def window(tmp_path, monkeypatch):
    pytest.importorskip('PyQt5')
    from PyQt5.QtCore import QSettings
    from PyQt5.QtWidgets import QApplication
    from themes import apply_theme
    import gui
    app = QApplication.instance() or QApplication([])
    apply_theme(app, 'dark')
    settings = QSettings(str(tmp_path / 'layout.ini'), QSettings.IniFormat)
    monkeypatch.setattr(gui.PushbroomScannerMainWindow, '_settings', lambda self: settings)
    config, moisture, classes, cube, _ = regression_setup(tmp_path, fps=300.0, loop=True)
    w = gui.PushbroomScannerMainWindow(config)
    w.resize(1280, 760)
    w.show()
    app.processEvents()
    w._gui_timer.stop()
    yield app, w, moisture, classes
    w._pipeline.stop()                    # not close(): that saves the user's window layout
    w.hide()
    app.processEvents()


def refresh_until(app, w, condition, timeout=20.0):
    t = time.perf_counter()
    while time.perf_counter() - t < timeout:
        app.processEvents()
        w._refresh_gui_slot()
        if condition():
            return True
        time.sleep(0.02)
    return False


def test_gui_shows_a_moisture_map_and_summary(window):
    app, w, _, _ = window
    tabs = w._side_tabs
    assert w._regression is w._pipeline.classifier
    assert w.color_bar.isVisible() and w.color_bar.label == 'moisture (%)'
    assert w.color_bar.display_range == (5.0, 60.0)
    assert tabs.currentWidget() is w.regression_panel and tabs.tabText(tabs.indexOf(w.regression_panel)) == 'Moisture'
    assert not tabs.isTabVisible(tabs.indexOf(w.composition_widget))
    assert not tabs.isTabVisible(tabs.indexOf(w.class_legend_widget))
    assert w.model_selector.parent() is not None and w.regression_panel.isAncestorOf(w.model_selector)
    assert w._lut[regression.BACKGROUND_CODE].tolist() == [0, 0, 0]
    assert refresh_until(app, w, lambda: w.regression_panel.summary.get('n', 0) > 200)
    summary = w.regression_panel.summary
    assert 5.0 < summary['mean'] < 60.0 and summary['median'] == pytest.approx(summary['mean'], abs=8.0)
    assert w.regression_panel.lbl_mean.text().endswith(' %')
    assert 0.3 < summary['object_share'] < 0.7                    # half of the cube is belt
    assert refresh_until(app, w, lambda: len(w.regression_panel.objects) > 0)
    assert w.regression_panel.objects_table.rowCount() == len(w.regression_panel.objects)
    # colour range: the map's LUT follows it (the map holds value codes: history recolours too)
    before = w._lut[100].copy()
    w.regression_panel.set_display_range(20.0, 30.0, emit=True)
    assert w.color_bar.display_range == (20.0, 30.0) and not np.array_equal(w._lut[100], before)
    w._auto_display_range()
    lo, hi = w.color_bar.display_range
    assert 0.0 <= lo < hi <= 70.0
    # object mask off: the belt becomes objects with (meaningless) values
    w.regression_panel.mask_box.setChecked(False)
    assert w._pipeline.classifier.object_min_mean is None
    w.regression_panel.mask_box.setChecked(True)
    assert w._pipeline.classifier.object_min_mean == pytest.approx(BELT_MEAN)


def test_gui_probe_reads_the_predicted_value(window):
    app, w, _, _ = window
    w._spectrum_button.setChecked(True)
    w._sync_spectrum_history()
    assert refresh_until(app, w, lambda: w._belt is not None and w._belt.get('values') is not None
                         and np.isfinite(w._belt['values']).sum() > 20)
    values = w._belt['values']
    rows, cols = np.nonzero(np.isfinite(values))
    from PyQt5.QtCore import QPointF
    lines, drawn = w._belt_columns()
    label = w._canvas_label
    # the newest drawn columns are on the right: aim at a column whose value is known
    column = int(cols.max())
    x = (drawn - len(lines) + column + 0.5) / drawn * label.width()
    y = (rows[cols == column][0] * w._belt['step'][0] + 0.5) / w._pipeline.slit_rows * label.height()
    w._probe_spectrum(QPointF(x, y))
    note = w.spectrum_panel.note_label.text()
    points = w.spectrum_panel.points
    if points:
        assert points[-1].label.startswith('moisture ') and points[-1].label.endswith(' %')
    else:                                                       # the line left the kept history
        assert 'kept' in note or 'lost' in note


def test_gui_switches_back_to_classes_and_again(window):
    app, w, moisture, classes = window
    from model_catalog import probe_model
    tabs = w._side_tabs
    entry = probe_model(classes, w._pipeline.plans, w.config.model.band_tolerance_nm)
    w._on_model_chosen(entry)
    assert refresh_until(app, w, lambda: w._shown_generation == 1)
    assert w._regression is None and not w.color_bar.isVisible()
    assert tabs.isTabVisible(tabs.indexOf(w.composition_widget)) and tabs.isTabVisible(tabs.indexOf(w.class_legend_widget))
    assert not tabs.isTabVisible(tabs.indexOf(w.regression_panel))
    assert w.class_legend_widget.isAncestorOf(w.model_selector)
    assert w.class_legend_widget._names == ['Background', 'Leaf', 'Wood']
    assert refresh_until(app, w, lambda: w._pipeline.snapshot().counted_lines > 20)
    assert set(np.flatnonzero(w._pipeline.snapshot().counts).tolist()) <= {0, 1, 2}
    entry = probe_model(moisture, w._pipeline.plans, w.config.model.band_tolerance_nm)
    assert entry.usable and entry.kind == 'PLSR · moisture (%)'
    w._on_model_chosen(entry)
    assert refresh_until(app, w, lambda: w._shown_generation == 2)
    assert w._regression is entry.classifier and w.color_bar.isVisible()
    assert tabs.currentWidget() is w.regression_panel
    assert refresh_until(app, w, lambda: w.regression_panel.summary.get('n', 0) > 0)
