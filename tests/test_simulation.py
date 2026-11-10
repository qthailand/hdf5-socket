"""Frame sources, acquisition plan, classifier binding and the lossless pipeline."""
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py  # noqa: E402

from inference_engine import PushbroomLineClassifier  # noqa: E402
from modules.preprocessing import PreprocessingRuntime  # noqa: E402
from pipeline import LOST_CLASS, Pipeline  # noqa: E402
from sim_config import AcquisitionConfig, CameraConfig, CorrectionConfig, ProcessingConfig, \
    SimulationConfig, build_plan, load_config  # noqa: E402
from sources import FrameSource, open_envi, open_hdf5, open_sources  # noqa: E402


# ----------------------------------------------------------------------------- ENVI ---

def write_envi(folder, cube, interleave, wavelengths=None, byte_order=0, units='Nanometers', name='cube'):
    """``cube`` is (lines, samples, bands)."""
    L, S, B = cube.shape
    data = {'bsq': cube.transpose(2, 0, 1), 'bil': cube.transpose(0, 2, 1), 'bip': cube}[interleave]
    dtype = cube.dtype.newbyteorder('>' if byte_order else '<')
    path = Path(folder) / f'{name}.{interleave}'
    np.ascontiguousarray(data).astype(dtype).tofile(path)
    code = {np.dtype(np.uint16): 12, np.dtype(np.float32): 4}[cube.dtype]
    lines = ['ENVI', 'description = {test cube}', f'samples = {S}', f'lines = {L}', f'bands = {B}',
             f'data type = {code}', f'interleave = {interleave}', f'byte order = {byte_order}']
    if wavelengths is not None:
        lines += [f'wavelength units = {units}', 'wavelength = {' + ',\n '.join(f'{w:.4f}' for w in wavelengths) + '}']
    Path(str(path) + '.hdr').write_text('\n'.join(lines) + '\n')
    return path


@pytest.fixture
def cube():
    rng = np.random.default_rng(3)
    return rng.integers(0, 4000, (6, 9, 5), dtype=np.uint16)         # 6 slit, 9 scan, 5 bands


@pytest.mark.parametrize('interleave', ['bsq', 'bil', 'bip'])
@pytest.mark.parametrize('cache', ['ram', 'disk'])
def test_envi_interleaves_become_scan_frames(tmp_path, cube, interleave, cache):
    path = write_envi(tmp_path, cube, interleave, wavelengths=np.linspace(400, 480, 5))
    src = open_envi(path, cache=cache, cache_dir=tmp_path / 'c')
    assert src.frames.shape == (9, 6, 5) and src.stage == 'binned' and src.frames.flags.c_contiguous
    for i in range(9):
        np.testing.assert_array_equal(src.frames[i], cube[:, i, :])          # frame = (slit, bands)
    np.testing.assert_allclose(src.wavelengths, np.linspace(400, 480, 5), atol=1e-3)


@pytest.mark.parametrize('interleave', ['bsq', 'bil', 'bip'])
def test_envi_scan_along_lines_big_endian_micrometres(tmp_path, cube, interleave):
    path = write_envi(tmp_path, cube, interleave, wavelengths=np.linspace(0.4, 0.48, 5), byte_order=1,
                      units='Micrometers')
    src = open_envi(Path(str(path) + '.hdr'), scan_axis='lines', cache='ram')
    assert src.frames.shape == (6, 9, 5) and src.frames.dtype == np.dtype('uint16')
    np.testing.assert_array_equal(src.frames, cube)
    np.testing.assert_allclose(src.wavelengths, np.linspace(400, 480, 5), atol=1e-2)


def test_envi_without_wavelengths_is_raw_and_bip_lines_is_zero_copy(tmp_path, cube):
    path = write_envi(tmp_path, cube, 'bip')
    src = open_envi(path, scan_axis='lines')
    assert src.stage == 'raw' and src.wavelengths is None and isinstance(src.frames, np.memmap)


# ----------------------------------------------------------------------------- HDF5 ---

def write_h5(path, frames, attrs=None, compression='gzip'):
    with h5py.File(path, 'w') as f:
        kwargs = dict(compression='gzip', compression_opts=4, chunks=(1, *frames.shape[1:])) if compression else {}
        f.create_dataset('spectrogram', data=frames, **kwargs)
        for k, v in (attrs or {}).items():
            f.attrs[k] = v
    return path


def test_hdf5_gzip_decoded_once_then_cached(tmp_path):
    frames = np.random.default_rng(1).integers(0, 65535, (40, 8, 30), dtype=np.uint16)
    path = write_h5(tmp_path / 'a.h5', frames, {'camera_bin': 2})
    src = open_hdf5(path, cache_dir=tmp_path / 'cache')
    np.testing.assert_array_equal(src.frames, frames)
    assert src.stage == 'raw' and src.metadata['camera_bin'] == 2 and src.frames.dtype == np.uint16
    assert len(list((tmp_path / 'cache').glob('a.*.npy'))) == 1
    again = open_hdf5(path, cache_dir=tmp_path / 'cache')
    np.testing.assert_array_equal(again.frames, frames)
    ram = open_hdf5(path, cache='ram')
    assert not isinstance(ram.frames, np.memmap)
    np.testing.assert_array_equal(ram.frames, frames)


def test_hdf5_contiguous_is_mapped(tmp_path):
    frames = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
    src = open_hdf5(write_h5(tmp_path / 'b.h5', frames, compression=None))
    assert isinstance(src.frames, np.memmap)
    np.testing.assert_array_equal(src.frames, frames)


def test_playlist_rejects_mixed_shapes(tmp_path):
    write_h5(tmp_path / 'a.h5', np.zeros((2, 3, 4), np.uint16))
    write_h5(tmp_path / 'b.h5', np.zeros((2, 3, 5), np.uint16))
    with pytest.raises(ValueError, match='cannot follow'):
        open_sources(tmp_path)


# -------------------------------------------------------------------------- plan ---

def make_config(tmp_path, **sections):
    config = SimulationConfig(base_dir=str(tmp_path))
    return replace(config, **sections).validate()


def test_file_metadata_overrides_config(tmp_path):
    src = FrameSource('x.h5', np.zeros((1, 10, 804), np.uint16), 'raw', metadata={
        'camera_bin': 2, 'raw_inverted': True, 'm_nm_per_px_bin1': 0.2903, 'c_nm': 399.791, 'resolution_nm': 2.0})
    config = make_config(tmp_path, acquisition=AcquisitionConfig(camera_bin=4, raw_inverted=False, m_nm_per_px_bin1=1.0))
    plan = build_plan(config, src)
    assert plan.camera_bin == 2 and plan.raw_inverted is True
    assert plan.binner.n_bins == 233 and abs(plan.wavelengths[0] - 400.7915) < 1e-3
    assert plan.provenance['camera_bin'][1] == 'file:camera_bin'
    plan = build_plan(replace(config, acquisition=replace(config.acquisition, use_file_metadata=False)), src)
    assert plan.camera_bin == 4 and plan.raw_inverted is False


def calibration_tree(tmp_path, shape):
    dark = tmp_path / 'calibration' / 'dark_frames' / 'poa'
    dark.mkdir(parents=True)
    hot = np.zeros(shape, np.uint8)
    hot[2, 3] = 255
    np.savez(dark / 'dark_b2_e1.0_g1.0.npz', mean=np.full(shape, 7, np.uint16), hot_pixel_map=hot,
             bin=np.int32(2), exposure=np.float32(1), gain=np.float32(1), n_frames=np.int32(5))
    smile = tmp_path / 'calibration' / 'smile_maps' / 'poa'
    smile.mkdir(parents=True)
    # A real hsi model (any camera in calibration/) of another frame size than the test frames.
    import shutil
    models = sorted((ROOT / 'calibration' / 'smile_maps').glob('*/smile_model_bin*.npz'), key=lambda p: p.stat().st_size)
    if not models:
        pytest.skip('no hsi Smile model in calibration/smile_maps')
    shutil.copy(models[0], smile / 'smile_model_bin2.npz')


def test_auto_dark_from_metadata_and_mismatching_smile_bypassed(tmp_path):
    shape = (6, 40)
    calibration_tree(tmp_path, shape)
    src = FrameSource('x.h5', np.zeros((1, *shape), np.uint16), 'raw',
                      metadata={'camera_bin': 2, 'exposure_ms': 1.0, 'gain': 1.0})
    config = make_config(tmp_path, corrections=CorrectionConfig(dark='auto', smile='auto', backend='cpu'))
    plan = build_plan(config, src)
    assert plan.dark._dark_mean is not None and int(plan.dark._dark_mean[0, 0]) == 7
    assert 'dark_b2_e1.0_g1.0.npz' in plan.provenance['dark'][0] and '1 hot pixels' in plan.provenance['dark'][0]
    assert plan.smile is None and plan.provenance['smile'][0].startswith('mismatch')
    explicit = replace(config, corrections=replace(config.corrections, smile='calibration/smile_maps/poa/smile_model_bin2.npz'))
    with pytest.raises(ValueError, match='does not match'):
        build_plan(explicit, src)


def test_auto_dark_without_exposure_is_bypassed(tmp_path):
    calibration_tree(tmp_path, (6, 40))
    src = FrameSource('x.h5', np.zeros((1, 6, 40), np.uint16), 'raw')
    plan = build_plan(make_config(tmp_path, corrections=CorrectionConfig(dark='auto')), src)
    assert plan.dark._dark_enabled is False and plan.provenance['dark'][0].startswith('missing')


def test_unknown_config_key_is_an_error(tmp_path):
    path = tmp_path / 'sim.json'
    path.write_text(json.dumps({'camera': {'fsp': 100}}))
    with pytest.raises(ValueError, match='fsp'):
        load_config(path)


# -------------------------------------------------------------------- classifier ---

@pytest.fixture
def model(tmp_path):
    import joblib
    path = tmp_path / 'm.joblib'
    joblib.dump({'W_fused': np.eye(2, dtype=np.float32), 'b_fused': np.zeros(2, np.float32),
                 'selected_bands': np.array([1, 3]), 'classes': np.array([1, 2]),
                 'category_names': {1: 'A', 2: 'B'},
                 'band_wavelengths': np.array([400, 402, 404, 406, 408], np.float32)}, path)
    return PushbroomLineClassifier(path)


def test_classifier_binds_by_wavelength(model):
    columns, how = model.bind(np.array([398, 400, 402, 404, 406, 408, 410], np.float32), 7)
    assert columns.tolist() == [2, 4] and how.startswith('by wavelength')
    line = np.zeros((3, 7), np.uint16)
    line[0, 2] = 9                      # 402 nm high -> class 1
    line[1, 4] = 9                      # 406 nm high -> class 2
    assert model.predict(line, columns).tolist()[:2] == [1, 2]
    with pytest.raises(ValueError, match='no input band'):
        model.bind(np.array([500, 502, 504], np.float32), 3)


def test_classifier_index_fallback_checks_band_count(model):
    columns, how = model.bind(None, 5)
    assert columns.tolist() == [1, 3] and 'NOT verified' in how
    with pytest.raises(ValueError, match='trained on 5'):
        model.bind(None, 6)


def test_confidence_is_sklearn_predict_proba(tmp_path):
    """Fused weights (as train_sgd_coco.py stores them) give the probability of sklearn's own
    log_loss model (one-vs-rest) for the chosen class."""
    import joblib
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(1)
    y = rng.integers(1, 4, 600)
    x = rng.normal(100.0, 20.0, (600, 6)) + y[:, None] * np.array([5.0, -3.0, 0.0, 8.0, 1.0, -6.0])
    scaler = StandardScaler().fit(x)
    sk = SGDClassifier(loss='log_loss', random_state=0).fit(scaler.transform(x), y)
    joblib.dump({'W_fused': (sk.coef_ / scaler.scale_).T,
                 'b_fused': sk.intercept_ - sk.coef_ @ (scaler.mean_ / scaler.scale_),
                 'selected_bands': np.arange(6), 'classes': sk.classes_}, tmp_path / 'm.joblib')
    model = PushbroomLineClassifier(tmp_path / 'm.joblib')
    columns, _ = model.bind(None, 6)
    ids, confidence = model.predict_confidence(x, columns)
    np.testing.assert_array_equal(ids, model.predict(x, columns))
    assert (ids == sk.predict(scaler.transform(x))).mean() > 0.99          # float32 vs float64 near ties
    np.testing.assert_allclose(confidence, sk.predict_proba(scaler.transform(x)).max(axis=1), atol=1e-4)
    with np.errstate(all='raise'):                                          # far outside the training data
        _, extreme = model.predict_confidence(x * 1e6, columns)
    assert np.all((extreme > 0) & (extreme <= 1))
    x[0, 2] = np.nan                                                        # a bad band: uncertain, never NaN
    ids, confidence = model.predict_confidence(x, columns)
    assert confidence[0] == 0.0 and np.isfinite(confidence).all()
    np.testing.assert_array_equal(ids, model.predict(x, columns))


def test_temperature_scales_confidence_not_class(tmp_path):
    """The artifact's ``temperature`` divides the scores before the confidence (what
    train_sgd_coco.py fits), never changes the class, and is recovered from labels drawn at it."""
    import joblib
    from train_sgd_coco import fit_temperature, ovr_probabilities
    rng = np.random.default_rng(2)
    W = rng.normal(0.0, 1.0, (4, 3)).astype(np.float32)
    b = np.array([-1.0, 0.5, -0.2], np.float32)
    artifact = {'W_fused': W, 'b_fused': b, 'selected_bands': np.arange(4), 'classes': np.array([1, 2, 3])}
    joblib.dump(artifact, tmp_path / 'plain.joblib')
    joblib.dump({**artifact, 'temperature': 0.5}, tmp_path / 'sharp.joblib')
    plain, sharp = PushbroomLineClassifier(tmp_path / 'plain.joblib'), PushbroomLineClassifier(tmp_path / 'sharp.joblib')
    assert plain.temperature == 1.0                                         # older artifacts: sklearn's value
    x = rng.normal(0.0, 2.0, (2000, 4)).astype(np.float32)
    columns, _ = plain.bind(None, 4)
    ids, conf = plain.predict_confidence(x, columns)
    sharp_ids, sharp_conf = sharp.predict_confidence(x, columns)
    np.testing.assert_array_equal(sharp_ids, ids)
    np.testing.assert_array_equal(sharp.predict(x, columns), plain.predict(x, columns))
    scores = x.astype(np.float64) @ W + b
    np.testing.assert_allclose(conf, ovr_probabilities(scores).max(axis=1), atol=1e-5)
    np.testing.assert_allclose(sharp_conf, ovr_probabilities(scores, 0.5).max(axis=1), atol=1e-5)
    assert sharp_conf.mean() > conf.mean() + 0.05                          # T < 1 sharpens (on average:
                                                                            # one-vs-rest is not a softmax)

    big = rng.normal(0.0, 3.0, (20000, 3))                                 # labels drawn at T = 0.5
    p = ovr_probabilities(big, 0.5)
    y_index = (rng.random(big.shape[0])[:, None] > p.cumsum(axis=1)).sum(axis=1)
    assert fit_temperature(big, y_index) == pytest.approx(0.5, rel=0.1)

    joblib.dump({**artifact, 'temperature': 0.0}, tmp_path / 'bad.joblib')
    with pytest.raises(ValueError, match='temperature'):
        PushbroomLineClassifier(tmp_path / 'bad.joblib')


# ---------------------------------------------------------------------- pipeline ---

class StubPlan:
    stage = 'binned'
    columns = None
    correction_backend = 'cpu'


class StubClassifier:
    """Class id = first pixel of the frame (so the line order can be checked)."""
    category_names = {}

    def __init__(self, delay=0.0):
        self.delay = delay

    def predict(self, binned, columns):
        if self.delay:
            time.sleep(self.delay)
        return np.full(binned.shape[0], binned[0, 0], np.uint8)


def stub_pipeline(tmp_path, n, fps, overrun, buffers, workers, delay=0.0, stride=1):
    frames = np.zeros((n, 4, 3), np.uint16)
    frames[:, 0, 0] = np.arange(n) % 200 + 1
    src = FrameSource('s.h5', frames, 'binned')
    config = make_config(tmp_path, camera=CameraConfig(fps=fps, overrun=overrun, buffer_frames=buffers, loop=False),
                         processing=ProcessingConfig(workers=workers, profile_every_s=0))
    config = replace(config, display=replace(config.display, canvas_width=4096, canvas_line_stride=stride))
    return Pipeline(config, [src], [StubPlan()], StubClassifier(delay)), frames


def run_to_end(pipeline, timeout=20):
    pipeline.start()
    t = time.perf_counter()
    while pipeline.snapshot().state in ('starting', 'running') and time.perf_counter() - t < timeout:
        time.sleep(0.02)
    pipeline.stop()
    return pipeline.snapshot()


def test_block_mode_counts_every_line_in_order(tmp_path):
    pipeline, frames = stub_pipeline(tmp_path, 600, fps=0, overrun='block', buffers=4, workers=3)
    s = run_to_end(pipeline)
    assert s.state == 'finished' and s.lines_processed == 600 and s.lines_lost == 0
    assert s.counted_lines == 600 and s.counts.sum() == 600 * 4
    np.testing.assert_array_equal(s.canvas[0], frames[:, 0, 0])            # line order restored


def test_overrun_drops_are_counted_and_drawn(tmp_path):
    pipeline, frames = stub_pipeline(tmp_path, 300, fps=2000, overrun='drop', buffers=2, workers=1, delay=0.004)
    s = run_to_end(pipeline)
    assert s.state == 'finished'
    assert s.lines_lost > 0 and s.lines_processed + s.lines_lost == 300
    assert s.counts.sum() == s.lines_processed * 4
    drawn_lost = int((s.canvas[0] == LOST_CLASS).sum())
    assert 0 < drawn_lost <= s.lines_lost
    assert s.canvas.shape[1] <= 300


def test_reset_counts_starts_new_batch(tmp_path):
    pipeline, _ = stub_pipeline(tmp_path, 50, fps=0, overrun='block', buffers=4, workers=1)
    s = run_to_end(pipeline)
    assert s.counted_lines == 50
    pipeline.reset_counts()
    s = pipeline.snapshot()
    assert s.counted_lines == 0 and s.counts.sum() == 0 and s.lines_processed == 50


# --------------------------------------------------------- reproduce HSI acquisition ---

DATA = ROOT / 'data' / 'spectrogram02.h5'
CUBE = ROOT / 'dataset' / 'dataset.bip.hdr'


@pytest.mark.skipif(not (DATA.is_file() and CUBE.is_file()), reason='recorded data not present')
def test_raw_pipeline_reproduces_the_training_cube():
    """dataset.bip column j was produced by HSI from spectrogram02.h5 frame 2j+1 (bin 2, no flip,
    no dark/Smile): the simulator's raw path must give the identical binned spectra."""
    config = load_config(ROOT / 'simulation.json')
    raw = open_hdf5(DATA, cache='disk', cache_dir=ROOT / 'data' / '.cache')
    cube = open_envi(CUBE)
    plan = build_plan(config, raw)
    np.testing.assert_allclose(plan.wavelengths, cube.wavelengths, atol=1e-3)
    runtime = PreprocessingRuntime('cpu')
    for j in (0, 1, 100, 391):
        binned, _, _ = runtime.process_binned(raw.frames[2 * j + 1], plan.dark, plan.binner,
                                              calibration=plan.smile, inverted=plan.raw_inverted)
        np.testing.assert_array_equal(binned, cube.frames[j])


# ------------------------------------------------------------- same code as hsi ---

def test_processing_code_is_identical_to_hsi():
    """Dark, hot pixel, flip, Smile and binning must stay the hsi algorithm byte for byte."""
    sys.path.insert(0, str(ROOT / 'tools'))
    import sync_hsi
    if not (sync_hsi.DEFAULT_HSI / 'modules').is_dir():
        pytest.skip('hsi project not next to this one')
    assert sync_hsi.compare(sync_hsi.DEFAULT_HSI) == []


# ------------------------------------------------------------------ backend switch ---

def test_backend_switch_while_running_loses_no_line(tmp_path):
    frames = np.random.default_rng(5).integers(0, 4000, (400, 6, 804), dtype=np.uint16)
    frames[:, 0, 0] = 1
    src = FrameSource('r.h5', frames, 'raw')
    config = make_config(tmp_path, camera=CameraConfig(fps=0, overrun='block', buffer_frames=8, loop=True),
                         corrections=CorrectionConfig(backend='auto'),
                         processing=ProcessingConfig(workers=2, profile_every_s=0))
    plan = build_plan(config, src)
    plan.columns = None
    pipeline = Pipeline(config, [src], [plan], StubClassifier())
    with pytest.raises(ValueError):
        pipeline.set_backend('gpu')
    pipeline.start()
    try:
        t = time.perf_counter()
        while pipeline.snapshot().lines_processed < 200 and time.perf_counter() - t < 20:
            time.sleep(0.02)
        first = list(pipeline._runtimes)
        pipeline.set_backend('cpu')
        t = time.perf_counter()
        while (any(a is b for a, b in zip(pipeline._runtimes, first))
               and time.perf_counter() - t < 20):
            time.sleep(0.02)
        s = pipeline.snapshot()
        assert s.requested_backend == 'cpu' and s.backend == 'cpu' and s.backend_note == ''
        assert not any(a is b for a, b in zip(pipeline._runtimes, first))     # new runtime per worker
        from modules.compute_runtime import get_compute_capabilities
        if not get_compute_capabilities().cupy_cuda.available:
            pipeline.set_backend('cupy_cuda')
            t = time.perf_counter()
            while 'unavailable' not in pipeline.snapshot().backend_note and time.perf_counter() - t < 20:
                time.sleep(0.02)
            s = pipeline.snapshot()
            assert s.backend == 'cpu' and 'cupy_cuda unavailable' in s.backend_note
    finally:
        pipeline.stop()
    s = pipeline.snapshot()
    assert s.error is None and s.lines_lost == 0 and s.counted_lines == s.lines_processed


# --------------------------------------------------------------- step switches ---

class ValueClassifier:
    """Class id = binned value of the first pixel, so the applied corrections are visible."""
    category_names = {}

    def predict(self, binned, columns):
        return np.full(binned.shape[0], min(254, int(binned[0, 0])), np.uint8)


def dark_plan(tmp_path, **corrections):
    shape = (6, 40)
    calibration_tree(tmp_path, shape)
    frames = np.full((30, *shape), 100, np.uint16)
    src = FrameSource('x.h5', frames, 'raw', metadata={'camera_bin': 2, 'exposure_ms': 1.0, 'gain': 1.0})
    config = make_config(tmp_path, camera=CameraConfig(fps=0, overrun='block', buffer_frames=4, loop=False),
                         corrections=CorrectionConfig(backend='cpu', **corrections),
                         processing=ProcessingConfig(workers=2, profile_every_s=0))
    plan = build_plan(config, src)
    plan.columns = None
    return config, src, plan


def test_steps_dark_and_hot_pixel_are_one_switch(tmp_path):
    _, _, plan = dark_plan(tmp_path, dark='auto')
    assert set(plan.steps) == {'dark', 'flip', 'smile', 'binning', 'reflectance'}
    assert plan.steps['dark'] == {'available': True, 'on': True, 'why': plan.steps['dark']['why']}
    assert '1 hot pixels repaired' in plan.steps['dark']['why']
    on, off = plan.dark_processor(True), plan.dark_processor(False)
    assert on._dark_enabled and on._hot_coords is not None and not off._dark_enabled
    assert plan.steps['binning'] == {'available': False, 'on': True,
                                     'why': 'always on: the model needs wavelength bins'}


def test_reference_found_but_not_configured_is_offered_off(tmp_path):
    _, _, plan = dark_plan(tmp_path, dark='none')
    assert plan.steps['dark']['available'] and not plan.steps['dark']['on']
    assert plan.steps['dark']['why'].startswith('none (available: dark_b2_e1.0_g1.0.npz')
    assert not plan.dark._dark_enabled                         # configured "none" still means off


def test_binned_input_has_no_switchable_step():
    plan = build_plan(SimulationConfig(), FrameSource('c.bip', np.zeros((2, 3, 4), np.uint16), 'binned',
                                                      wavelengths=np.arange(4, dtype=np.float32)))
    assert all(not info['available'] for info in plan.steps.values())


@pytest.mark.parametrize('dark_on, expected', [(True, 93), (False, 100)])
def test_switching_dark_changes_what_workers_apply(tmp_path, dark_on, expected):
    config, src, plan = dark_plan(tmp_path, dark='auto')          # dark mean 7 -> 100 - 7 = 93
    pipeline = Pipeline(config, [src], [plan], ValueClassifier())
    pipeline.set_steps(dark=dark_on)
    assert pipeline.effective_steps()['dark'] is dark_on and pipeline.effective_steps()['binning']
    with pytest.raises(ValueError):
        pipeline.set_steps(binning=False)
    s = run_to_end(pipeline)
    assert s.state == 'finished' and s.lines_processed == 30
    assert set(np.unique(s.canvas[0]).tolist()) == {expected}


def test_unavailable_step_stays_off(tmp_path):
    config, src, plan = dark_plan(tmp_path, dark='auto')
    plan.steps['smile'] = {'available': False, 'on': False, 'why': 'no model'}
    pipeline = Pipeline(config, [src], [plan], ValueClassifier())
    pipeline.set_steps(smile=True, flip=True)
    eff = pipeline.effective_steps()
    assert eff['smile'] is False and eff['flip'] is True
    pipeline.set_steps(flip=None)                                  # back to the plan's choice
    assert pipeline.effective_steps()['flip'] is plan.steps['flip']['on']


# ------------------------------------------------------------------ reflectance ---

def reflectance_setup(tmp_path, factor_value=0.5, rowwise=True, reflectance='auto', name='x', bands=None):
    from modules.radiometric import save_factor_profile
    shape = (6, 40)
    calibration_tree(tmp_path, shape)
    src = FrameSource(f'{name}.h5', np.full((30, *shape), 100, np.uint16), 'raw',
                      metadata={'camera_bin': 2, 'exposure_ms': 1.0, 'gain': 1.0})
    config = make_config(tmp_path, camera=CameraConfig(fps=0, overrun='block', buffer_frames=4, loop=False),
                         corrections=CorrectionConfig(backend='cpu', dark='auto', reflectance=reflectance),
                         processing=ProcessingConfig(workers=2, profile_every_s=0))
    probe = build_plan(replace(config, corrections=replace(config.corrections, reflectance='none')), src)
    n = bands or probe.n_columns
    factors = np.full((shape[0], n) if rowwise else (n,), factor_value, np.float32)
    (tmp_path / 'reflectance').mkdir(exist_ok=True)
    save_factor_profile(tmp_path / 'reflectance' / 'x.npy', factors,
                        probe.wavelengths if bands is None else np.arange(n, dtype=float))
    return config, src


@pytest.mark.parametrize('rowwise', [True, False])
def test_reflectance_is_counts_times_factor_before_the_model(tmp_path, rowwise):
    config, src = reflectance_setup(tmp_path, 0.5, rowwise)
    plan = build_plan(config, src)
    assert plan.steps['reflectance']['on'] and plan.steps['reflectance']['available']
    assert 'bands checked by wavelength' in plan.steps['reflectance']['why']
    plan.columns = None
    pipeline = Pipeline(config, [src], [plan], ValueClassifier())
    s = run_to_end(pipeline)
    assert s.state == 'finished' and s.lines_processed == 30
    assert set(np.unique(s.canvas[0]).tolist()) == {46}            # (100 - 7 dark) * 0.5 = 46.5
    timed = set(s.stage_ms).union(*(st['avg'] for st in s.stage_stats.values()),
                                  *(prof.take() for prof in pipeline._profilers))
    assert 'reflectance' in timed                                   # its own bar in the performance chart


def test_reflectance_maps_by_source_name_and_is_offered_off(tmp_path):
    config, src = reflectance_setup(tmp_path, reflectance='none')
    plan = build_plan(config, src)
    assert plan.steps['reflectance']['available'] and not plan.steps['reflectance']['on']
    assert plan.steps['reflectance']['why'].startswith('none (available: x.npy')
    other = FrameSource('other.h5', src.frames, 'raw', metadata=src.metadata)
    plan = build_plan(config, other)                                 # no reflectance/other.npy
    assert not plan.steps['reflectance']['available'] and 'other.npy' in plan.steps['reflectance']['why']


def test_reflectance_with_other_bands_is_refused(tmp_path):
    config, src = reflectance_setup(tmp_path, bands=17)
    plan = build_plan(config, src)                                   # auto: logged, step unavailable
    assert not plan.steps['reflectance']['available'] and 'do not fit' in plan.steps['reflectance']['why']
    explicit = replace(config, corrections=replace(config.corrections, reflectance='reflectance/x.npy'))
    with pytest.raises(ValueError, match='do not fit'):
        build_plan(explicit, src)


def test_rowwise_reflectance_needs_one_row_per_slit_pixel(tmp_path):
    from modules.radiometric import save_factor_profile
    config, src = reflectance_setup(tmp_path)
    plan = build_plan(config, src)
    save_factor_profile(tmp_path / 'reflectance' / 'x.npy', np.ones((5, plan.n_columns), np.float32), plan.wavelengths)
    plan = build_plan(config, src)
    assert not plan.steps['reflectance']['available'] and 'slit rows' in plan.steps['reflectance']['why']


def test_envi_source_name_mapping(tmp_path):
    from sim_config import reflectance_path
    config = make_config(tmp_path)
    src = FrameSource(str(tmp_path / 'dataset.bip'), np.zeros((1, 2, 3), np.uint16), 'binned')
    assert reflectance_path(config, src) == tmp_path / 'reflectance' / 'dataset.npy'


# ------------------------------------------------------------ white reference ---

def white_setup(tmp_path, signal=1000.0, gradient=False):
    """Raw white frames = dark (7) + signal; the white is what the data chain should cancel."""
    shape = (6, 40)
    calibration_tree(tmp_path, shape)
    rows = np.linspace(0.8, 1.2, shape[0])[:, None] if gradient else 1.0
    frames = np.repeat((7 + signal * rows * np.ones(shape)).astype(np.uint16)[None], 12, axis=0)
    src = FrameSource('white.h5', frames, 'raw', metadata={'camera_bin': 2, 'exposure_ms': 1.0, 'gain': 1.0})
    config = make_config(tmp_path, corrections=CorrectionConfig(backend='cpu', dark='auto'))
    return config, src, build_plan(config, src)


@pytest.mark.parametrize('kind', ['band', 'row'])
def test_white_through_the_chain_gives_target_reflectance(tmp_path, kind):
    from reflectance_builder import factors_from_white, white_mean
    from sim_config import apply_reflectance
    config, src, plan = white_setup(tmp_path, gradient=(kind == 'row'))
    white = white_mean(src, plan)
    assert white.steps['dark'] and white.n_frames == 12
    target = np.linspace(0.40, 0.45, plan.n_columns).astype(np.float32)
    factors = factors_from_white(white.mean, target, kind, avg_window=1)
    binned, _, _ = PreprocessingRuntime('cpu').process_binned(src.frames[0], plan.dark_processor(True), plan.binner)
    reflectance = apply_reflectance(binned, factors)
    if kind == 'row':                    # every slit row is corrected on its own
        np.testing.assert_allclose(reflectance, np.broadcast_to(target, reflectance.shape), rtol=2e-3)
    else:
        np.testing.assert_allclose(reflectance.mean(axis=0), target, rtol=2e-3)


def test_row_factors_outside_roi_use_nearest_roi_rows():
    from reflectance_builder import factors_from_white
    white = np.array([[1.0], [2.0], [4.0], [8.0], [16.0]])
    factors = factors_from_white(white, np.array([1.0], np.float32), 'row', rows=(1, 3), avg_window=1)
    np.testing.assert_allclose(factors[:, 0], [0.5, 0.5, 0.25, 0.25, 0.25])   # rows 0 -> 1, 3/4 -> 2


def test_white_needs_the_dark_step(tmp_path):
    from reflectance_builder import white_mean
    config, src, plan = white_setup(tmp_path)
    plan.steps['dark']['on'] = False
    with pytest.raises(ValueError, match='dark-corrected'):
        white_mean(src, plan)
    cube = FrameSource('w.bip', np.full((3, 4, 5), 100, np.uint16), 'binned', wavelengths=np.arange(5.0))
    cube_plan = build_plan(config, cube)
    with pytest.raises(ValueError, match='dark offset'):
        white_mean(cube, cube_plan)
    dark = FrameSource('d.bip', np.full((2, 4, 5), 30, np.uint16), 'binned', wavelengths=np.arange(5.0))
    assert np.allclose(white_mean(cube, cube_plan, dark_cube=dark).mean, 70)


def test_make_reflectance_cli_writes_profile_for_each_data_source(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('make_reflectance', ROOT / 'tools' / 'make_reflectance.py')
    tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
    cube = np.full((3, 4, 5), 1000, np.uint16)
    path = write_envi(tmp_path, cube, 'bip', wavelengths=np.linspace(900, 1000, 5), name='white')
    table = tmp_path / 'ref.csv'
    table.write_text('\n'.join(f"{w},0.4" for w in range(800, 1101, 10)))
    out = tmp_path / 'refl'
    code = tool.main([str(path), '--for', 'run_A', 'run_B', '--table', str(table), '--already-dark-corrected',
                      '--out', str(out), '--config', str(tmp_path / 'none.json')])
    assert code == 0
    from modules.radiometric import load_factor_profile
    factors, wl = load_factor_profile(str(out / 'run_A.npy'))
    np.testing.assert_allclose(factors, 0.4 / 1000, rtol=1e-4)
    assert (out / 'run_B.npy').is_file() and wl.size == 5


def test_placeholder_profile_is_marked_everywhere(tmp_path):
    import json as _json
    from reflectance_builder import factors_from_white, save_profile, white_mean
    config, src, plan = white_setup(tmp_path)
    plan.steps['dark']['on'] = False                      # no real white/dark yet: placeholder only
    white = white_mean(src, plan, require_dark=False)
    factors = factors_from_white(white.mean, np.full(plan.n_columns, 0.41, np.float32))
    save_profile(tmp_path / 'reflectance', ['white'], factors, white.wavelengths,
                 {'placeholder': True, 'white': 'white.h5'})
    assert _json.loads((tmp_path / 'reflectance' / 'white.json').read_text())['placeholder'] is True
    marked = build_plan(replace(config, corrections=replace(config.corrections, reflectance='auto')), src)
    assert marked.reflectance_placeholder and marked.steps['reflectance']['why'].startswith('PLACEHOLDER')


# ------------------------------------------------------- streaming / disk guard ---

def test_stream_reads_in_order_wraps_jumps_and_slices(tmp_path):
    from sources import H5FrameStream
    frames = np.random.default_rng(2).integers(0, 65535, (20, 5, 7), dtype=np.uint16)
    stream = H5FrameStream(write_h5(tmp_path / 's.h5', frames), depth=4, workers=2)
    try:
        assert stream.shape == frames.shape and len(stream) == 20 and stream.dtype == np.uint16
        for i in list(range(20)) + [0, 1]:                        # sequential, then wrap-around
            np.testing.assert_array_equal(stream[i], frames[i])
        np.testing.assert_array_equal(stream[13], frames[13])     # jump
        np.testing.assert_array_equal(stream[2:6], frames[2:6])
        np.testing.assert_array_equal(np.asarray(stream), frames)
    finally:
        stream.close()


def test_auto_cache_never_fills_the_disk(tmp_path, monkeypatch):
    import sources
    frames = np.zeros((10, 4, 6), np.uint16)
    path = write_h5(tmp_path / 'big.h5', frames)
    monkeypatch.setattr(sources, '_free_disk', lambda folder: 1 << 20)      # 1 MB free
    monkeypatch.setattr(sources, '_available_ram', lambda: 100)           # tiny RAM
    src = open_hdf5(path, cache='auto', cache_dir=tmp_path / 'c')
    assert isinstance(src.frames, sources.H5FrameStream)
    assert not (tmp_path / 'c').exists() or not list((tmp_path / 'c').iterdir())
    src.frames.close()
    with pytest.raises(OSError, match='GB free'):
        open_hdf5(path, cache='disk', cache_dir=tmp_path / 'c')


# ----------------------------------------------- wavelength range / temperature ---

def test_recorded_wavelength_range_keeps_the_same_bins(tmp_path):
    meta = {'camera_bin': 2, 'm_nm_per_px_bin1': 0.36385766843509226, 'c_nm': 842.1014795620235,
            'resolution_nm': 5.0, 'wavelength_range_bands': [12, 171, 186]}
    src = FrameSource('g.h5', np.zeros((1, 4, 1280), np.uint16), 'raw', metadata=meta)
    plan = build_plan(make_config(tmp_path), src)
    assert plan.binner.n_bins == 159 and plan.binner.source_bins == (12, 171, 186)
    assert abs(plan.wavelengths[0] - 904.65) < 0.01 and abs(plan.wavelengths[-1] - 1695.29) < 0.01
    bad = FrameSource('g.h5', src.frames, 'raw', metadata=dict(meta, resolution_nm=4.0))
    with pytest.raises(ValueError, match='wavelength calibration differs'):
        build_plan(make_config(tmp_path), bad)
    cfg = make_config(tmp_path, acquisition=AcquisitionConfig(use_file_metadata=False, camera_bin=2,
                      m_nm_per_px_bin1=0.36385766843509226, c_nm=842.1014795620235, resolution_nm=5.0,
                      wavelength_range_nm=[900.0, 1700.0]))
    assert build_plan(cfg, src).binner.source_bins == (12, 171, 186)       # same bins from the config range


def test_dark_is_chosen_at_the_recordings_dark_temperature(tmp_path):
    folder = tmp_path / 'calibration' / 'dark_frames' / 'genicam'
    folder.mkdir(parents=True)
    for temp, level in ((21, 5), (35, 9)):
        np.savez(folder / f'dark_b2_e4.0_g1.0_t+{temp}.npz', mean=np.full((6, 40), level, np.uint16),
                 hot_pixel_map=np.zeros((6, 40), np.uint8), bin=np.int32(2), exposure=np.float32(4),
                 gain=np.float32(1), n_frames=np.int32(5), temperature=np.float64(temp))
    src = FrameSource('g.h5', np.zeros((1, 6, 40), np.uint16), 'raw', metadata={
        'camera_backend': 'genicam', 'camera_bin': 2, 'exposure_ms': 4.0, 'gain': 1.0, 'dark_temperature_c': 21.02})
    plan = build_plan(make_config(tmp_path, corrections=CorrectionConfig(dark='auto')), src)
    assert int(plan.dark_mean[0, 0]) == 5 and 't+21' in plan.steps['dark']['why']


def test_speed_only_model_binds_to_any_band_axis():
    from inference_engine import SpeedTestClassifier
    model = SpeedTestClassifier()                                   # default: every band
    columns, how = model.bind(np.linspace(900, 1700, 159), 159)
    assert columns.tolist() == list(range(159)) and model.W_fused.shape == (159, 3) and 'all 159' in how
    assert model.predict(np.ones((8, 159), np.uint16), columns).shape == (8,)
    with pytest.raises(ValueError, match='sized for 159'):
        model.bind(None, 120)
    some = SpeedTestClassifier(n_selected=30)
    columns, how = some.bind(None, 159)
    assert columns.size == 30 and columns[0] == 0 and columns[-1] == 158 and '30 of 159' in how


def test_stream_skip_ahead_does_not_restart_the_reader(tmp_path):
    from sources import H5FrameStream
    frames = np.arange(30 * 2 * 3, dtype=np.uint16).reshape(30, 2, 3)
    stream = H5FrameStream(write_h5(tmp_path / 's.h5', frames), depth=8, workers=2)
    try:
        np.testing.assert_array_equal(stream[0], frames[0])
        reader = stream._thread
        np.testing.assert_array_equal(stream[5], frames[5])          # 4 lost lines: skip, same reader
        assert stream._thread is reader
        np.testing.assert_array_equal(stream[25], frames[25])        # far jump: new reader, old one ended
        assert stream._thread is not reader and not reader.is_alive()
    finally:
        stream.close()


# ---------------------------------------------------------------- slit-cropped white ---

def test_slit_crop_rows_map_through_the_flip():
    from reflectance_builder import crop_rows_after_chain
    assert crop_rows_after_chain(453, 143, 1024, flipped=True) == (429, 570)     # spectralon-20.h5
    assert crop_rows_after_chain(453, 143, 1024, flipped=False) == (454, 595)
    assert crop_rows_after_chain(10, 5, 20, flipped=True, margin=0) == (5, 10)


def test_cropped_white_gives_the_same_rows_as_the_full_frame(tmp_path):
    """Put back at its rows, a slit crop through the chain equals those rows of the full frame."""
    from reflectance_builder import crop_rows_after_chain, embed_slit_crop, find_slit_crop, white_mean
    config, full_src, plan_full = white_setup(tmp_path, gradient=True)
    full_src.frames[:, :, ::3] += np.arange(full_src.frames.shape[1], dtype=np.uint16)[None, :, None]  # row texture
    plan_full = build_plan(config, full_src)
    crop = FrameSource('white_crop.h5', np.ascontiguousarray(full_src.frames[:, 2:5]), 'raw',
                       metadata=full_src.metadata)
    path = write_h5(tmp_path / 'rec.h5', np.asarray(full_src.frames))
    assert find_slit_crop(crop.frames, path) == (0, 2)
    embedded = embed_slit_crop(crop, 2, full_src.frame_shape[0])
    plan_crop = build_plan(config, embedded)
    assert plan_crop.steps['dark']['on'] and embedded.frame_shape == full_src.frame_shape
    rows = crop_rows_after_chain(2, 3, full_src.frame_shape[0], flipped=plan_full.steps['flip']['on'], margin=0)
    a = white_mean(full_src, plan_full).mean[slice(*rows)]
    b = white_mean(embedded, plan_crop).mean[slice(*rows)]
    np.testing.assert_allclose(a[1:-1], b[1:-1])           # inner rows identical (edges: hot-pixel neighbours)


# ------------------------------------------------------------------ live line rate ---

def test_line_rate_changes_while_running_and_max_is_lossless(tmp_path):
    pipeline, frames = stub_pipeline(tmp_path, 400, fps=100, overrun='drop', buffers=2, workers=1, delay=0.002)
    pipeline.config = replace(pipeline.config, camera=replace(pipeline.config.camera, loop=True))
    with pytest.raises(ValueError):
        pipeline.set_fps(-1)
    pipeline.start()
    try:
        t = time.perf_counter()
        while pipeline.snapshot().state != 'running' and time.perf_counter() - t < 10:
            time.sleep(0.01)
        time.sleep(1.2)
        assert 80 <= pipeline.snapshot().camera_fps <= 120          # 100 lines/s
        pipeline.set_fps(0)                                          # max: waits for buffers, never drops
        lost = pipeline.snapshot().lines_lost
        time.sleep(1.5)
        s = pipeline.snapshot()
        assert s.target_fps == 0 and s.lines_lost == lost and s.camera_fps > 150
        pipeline.set_fps(50)
        time.sleep(1.2)
        assert 40 <= pipeline.snapshot().camera_fps <= 60
    finally:
        pipeline.stop()
    assert pipeline.snapshot().error is None


# ------------------------------------------------------------------------ fused CPU ---

class _HorizontalSmile:
    """hsi-style Smile calibration (horizontal map_x, identity rows) for synthetic tests."""
    def __init__(self, shape, interp):
        self.image_shape, self.interp = shape, interp
        y, x = np.indices(shape, dtype=np.float32)
        self.map_x = np.ascontiguousarray(x + 1.3 * np.sin(y / 9.0) + 0.4 * np.cos(x / 7.0))
        self._y = np.ascontiguousarray(y)
    def _get_map_y(self):
        return self._y


@pytest.mark.parametrize('interp', ['linear', 'cubic'])
@pytest.mark.parametrize('inverted', [False, True])
@pytest.mark.parametrize('with_smile', [False, True])
def test_fused_cpu_matches_hsi_cpu(tmp_path, interp, inverted, with_smile):
    from fused_cpu import FusedCPU
    from modules.spectral_processing import LambdaBinner
    shape = (37, 211)
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 60000, shape, dtype=np.uint16)
    dark = rng.integers(0, 900, shape, dtype=np.uint16)
    hot = np.zeros(shape, np.uint8); hot[[0, 5, 36], [0, 100, 210]] = 255
    from modules.dark import DarkProcessor
    dp = DarkProcessor(); dp.set_dark_reference(dark, hot)
    binner = LambdaBinner(m=1.7, c=900.0, step=5.0).fit(shape[1]).subset(3, 60)
    smile = _HorizontalSmile(shape, interp) if with_smile else None
    hsi, _, _ = PreprocessingRuntime('cpu').process_binned(frame, dp, binner, calibration=smile, inverted=inverted)
    ours = FusedCPU().process_binned(frame, dp, binner, smile, inverted)
    assert np.max(np.abs(hsi.astype(int) - ours.astype(int))) <= 1


def test_fused_runtime_falls_back_to_hsi_on_mismatch(monkeypatch):
    import fused_cpu
    from modules.dark import DarkProcessor
    from modules.spectral_processing import LambdaBinner
    runtime = fused_cpu.FusedRuntime()
    frame = np.full((8, 60), 1000, np.uint16)
    binner = LambdaBinner(m=1.0, c=900.0, step=4.0).fit(60)
    dp = DarkProcessor(); dp.set_dark_enabled(False)
    out, _, _ = runtime.process_binned(frame, dp, binner)
    assert runtime.backend_name == 'cpu_fused' and np.all(out == 1000)
    monkeypatch.setattr(fused_cpu.FusedCPU, 'process_binned', lambda self, *a, **k: np.zeros((8, binner.n_bins), np.uint16))
    runtime._fused._checked.clear()
    out, _, _ = runtime.process_binned(frame, dp, binner)
    assert runtime.backend_name == 'cpu' and 'differs' in runtime.fallback_reason and np.all(out == 1000)


def test_snapshot_returns_only_the_newest_canvas_columns(tmp_path):
    pipeline, frames = stub_pipeline(tmp_path, 300, fps=0, overrun='block', buffers=4, workers=2)
    s = run_to_end(pipeline)
    tail = pipeline.snapshot(canvas_columns=50).canvas
    assert tail.shape[1] == 50 and np.array_equal(tail[0], frames[-50:, 0, 0])



def test_snapshot_raw_canvas_is_the_model_input_intensity_of_the_newest_lines(tmp_path):
    pipeline, frames = stub_pipeline(tmp_path, 300, fps=0, overrun='block', buffers=4, workers=2)
    run_to_end(pipeline)
    s = pipeline.snapshot(canvas_columns=50, raw_columns=20)
    assert s.raw_canvas.shape == (4, 20) and s.raw_canvas.dtype == np.float32
    np.testing.assert_allclose(s.raw_canvas[0], frames[-20:].mean(axis=2)[:, 0])    # same columns as the map
    assert pipeline.snapshot().raw_canvas.shape == (4, 0)                          # not asked: nothing copied
    d = pipeline.snapshot(canvas_columns=50, raw_columns=20, step=(2, 3))         # screen resolution
    np.testing.assert_array_equal(d.canvas, s.canvas[::2, ::-1][:, ::3][:, ::-1])  # newest column kept
    np.testing.assert_array_equal(d.raw_canvas, s.raw_canvas[::2, ::-1][:, ::3][:, ::-1])
    pipeline._draw_lost(1)
    assert np.isnan(pipeline.snapshot(raw_columns=1).raw_canvas).all()             # lost line: no intensity


def test_belt_image_raw_zone_grey_on_the_right_classes_on_the_left():
    from types import SimpleNamespace
    from gui import PushbroomScannerMainWindow as W
    from pipeline import LOST_CLASS
    lut = np.zeros((256, 3), np.uint8)
    lut[1] = (200, 0, 0)
    lut[LOST_CLASS] = (255, 0, 255)
    label = SimpleNamespace(width=lambda: 90, height=lambda: 4)
    win = SimpleNamespace(_lut=lut, _raw_range=None, _canvas_label=label, _canvas_columns=lambda: 90,
                          config=SimpleNamespace(display=SimpleNamespace(raw_zone_fraction=1 / 3)))
    for name in ('_raw_zone_columns', '_gray_into'):
        setattr(win, name, getattr(W, name).__get__(win))
    canvas = np.ones((4, 60), np.uint8)                   # 60 lines so far, all class 1
    raw = np.tile(np.linspace(0, 1, 30, dtype=np.float32), (4, 1))
    raw[:, 10] = np.nan                                   # a lost line in the raw zone
    image = W._belt_image(win, canvas, raw)
    assert image.shape == (4, 90, 3)
    assert (image[:, :30] == 0).all()                     # no history yet: black
    assert (image[:, 30:60] == (200, 0, 0)).all()         # older lines: class colours
    grey = image[:, 60:]
    ok = np.delete(grey, 10, axis=1)
    assert (ok[..., 0] == ok[..., 1]).all() and ok[0, -1, 0] > ok[0, 0, 0]          # newest: grey levels
    assert (grey[:, 10] == (255, 0, 255)).all()
    win.config.display.raw_zone_fraction = 0.0
    assert (W._belt_image(win, canvas, raw)[:, 60:] == (200, 0, 0)).all()            # zone off: all classes


def test_optics_summary_slit_in_pixels_and_bandwidth():
    from sim_config import OpticsConfig, optics_summary
    text = optics_summary(OpticsConfig(50.0, 5.0), 2, 0.36385766843509226, 5.0)
    assert 'slit image 10 px = 5 binned px' in text and '~3.64 nm' in text and '1.37 x' in text
    assert 'pixel 0.813 mm' in text and 'slit width 4.07 mm' in text and '500 mm field = 615 px' in text
    assert abs(OpticsConfig().belt_pixel(2) - 0.8130) < 1e-3 and OpticsConfig(belt_pixel_mm=0.5).belt_pixel(2) == 0.5
    assert optics_summary(OpticsConfig(None, 5.0), 2, 0.36, 5.0) == ''


# ------------------------------------------------- white reference folder (config input) ---

def white_folder_setup(tmp_path, attrs=None, rows=6, **white):
    """A white recording in ``white/`` like a Spectralon held under part of the slit for some frames.

    Frames 3..8 have the target (dark 7 + 1000) in slit rows 1..4; the rest is dark only.
    """
    from sim_config import WhiteReferenceConfig
    shape = (6, 40)
    calibration_tree(tmp_path, shape)
    frames = np.full((12, rows, shape[1]), 7, np.uint16)
    frames[3:9, 1:5] = 1007
    (tmp_path / 'white').mkdir()
    write_h5(tmp_path / 'white' / 'white.h5', frames,
             attrs or {'camera_backend': 'poa', 'camera_bin': 2, 'exposure_ms': 1.0, 'gain': 1.0})
    (tmp_path / 'ref.csv').write_text('300,0.5\n1500,0.5\n2600,0.5\n', encoding='utf-8')
    config = make_config(tmp_path, corrections=CorrectionConfig(backend='cpu', dark='auto', reflectance='auto'),
                         white_reference=WhiteReferenceConfig(path='white', table='ref.csv', **white))
    data = FrameSource('run.h5', np.repeat(frames[5:6, :6] if rows == 6 else np.full((1, *shape), 1007, np.uint16),
                                           3, axis=0), 'raw',
                       metadata={'camera_bin': 2, 'exposure_ms': 1.0, 'gain': 1.0})
    return config, data


def test_white_reference_folder_makes_factors_at_start(tmp_path):
    from sim_config import apply_reflectance
    config, data = white_folder_setup(tmp_path)
    plan = build_plan(config, data)
    step = plan.steps['reflectance']
    assert step['available'] and step['on'] and 'from white white.h5' in step['why']
    info = json.loads((tmp_path / 'reflectance' / 'from_white' / 'run.json').read_text(encoding='utf-8'))
    assert info['rows'] == [1, 5] and info['frames'] == [4, 8]          # found by brightness
    binned, _, _ = PreprocessingRuntime('cpu').process_binned(
        np.ascontiguousarray(data.frames[0]), plan.dark_processor(True), plan.binner)
    reflectance = apply_reflectance(binned, plan.reflectance_factors)
    np.testing.assert_allclose(reflectance[1:5].mean(axis=0), 0.5, rtol=1e-2)
    assert not (tmp_path / 'reflectance' / 'run.npy').exists()           # hand-made profiles untouched


def test_white_reference_is_cached_until_the_white_changes(tmp_path):
    config, data = white_folder_setup(tmp_path)
    build_plan(config, data)
    out = tmp_path / 'reflectance' / 'from_white' / 'run.npy'
    first = out.stat().st_mtime_ns
    assert 'cached' in build_plan(config, data).steps['reflectance']['why']
    assert out.stat().st_mtime_ns == first
    time.sleep(0.05)
    frames = np.full((12, 6, 40), 7, np.uint16)
    frames[:, :] = 2007                                                  # a new, brighter white
    write_h5(tmp_path / 'white' / 'white.h5', frames,
             {'camera_backend': 'poa', 'camera_bin': 2, 'exposure_ms': 1.0, 'gain': 1.0})
    plan = build_plan(config, data)
    assert 'cached' not in plan.steps['reflectance']['why']
    assert json.loads(out.with_suffix('.json').read_text())['rows'] == 'all'   # fills the slit: no target edge


def test_white_of_another_camera_falls_back_to_the_named_profile(tmp_path):
    config, data = white_folder_setup(tmp_path, attrs={'camera_backend': 'genicam', 'camera_bin': 2})
    step = build_plan(config, data).steps['reflectance']
    assert not step['available'] and 'no white reference for poa bin 2' in step['why']
    assert 'white.h5: genicam bin 2' in step['why']


def test_slit_cropped_white_needs_its_offset(tmp_path):
    config, data = white_folder_setup(tmp_path, rows=4)
    step = build_plan(config, data).steps['reflectance']
    assert not step['available'] and 'slit_offset' in step['why']
    config = replace(config, white_reference=replace(config.white_reference, slit_offset=1))
    step = build_plan(config, data).steps['reflectance']
    assert step['available'], step['why']
    info = json.loads((tmp_path / 'reflectance' / 'from_white' / 'run.json').read_text(encoding='utf-8'))
    assert info['slit_crop'] == {'raw_rows': [1, 5], 'sensor_rows': 6}


def test_white_reference_config_is_checked(tmp_path):
    from sim_config import WhiteReferenceConfig
    with pytest.raises(ValueError, match='table'):
        make_config(tmp_path, white_reference=WhiteReferenceConfig(path='white'))
    with pytest.raises(ValueError, match='rows'):
        make_config(tmp_path, white_reference=WhiteReferenceConfig(path='white', table='t.csv', rows=[5, 2]))


# ------------------------------------------------------- reflectance drawn on the belt map ---

def test_calibration_capture_keeps_binned_lines_before_reflectance(tmp_path):
    config, src = reflectance_setup(tmp_path, 0.5, rowwise=False)          # reflectance on: x 0.5
    plan = build_plan(config, src)
    plan.columns = None
    pipeline = Pipeline(config, [src], [plan], ValueClassifier())
    pipeline.start_calibration_capture(8)
    snap = run_to_end(pipeline)
    ids, lines = pipeline.calibration_capture()
    assert ids.tolist() == list(range(22, 30)) and lines.shape == (8, 6, plan.n_columns)
    np.testing.assert_allclose(lines, 93.0, rtol=1e-3)                    # 100 - dark 7, not x 0.5
    assert pipeline.snapshot(raw_columns=8).raw_lines.tolist() == list(range(22, 30))   # = belt map columns
    first, part = pipeline.calibration_capture(24, 25)
    assert first.tolist() == [24, 25] and part.shape[0] == 2
    pipeline.stop_calibration_capture()
    assert pipeline.calibration_capture()[0].size == 0 and pipeline.calibration_count() == 0


def test_drawn_factors_give_the_standard_reflectance_and_apply_live(tmp_path):
    from reflectance_builder import factors_from_lines, reference_values
    config, src, plan = dark_plan(tmp_path, dark='auto')
    lines = np.full((10, 6, plan.n_columns), 93.0, np.float32)
    lines[:, :2] = 5.0                                                    # rows 0-1 off the standard
    target = reference_values(0.99, plan.wavelengths)
    factors, white = factors_from_lines(lines, (2, 6), target, plan.wavelengths, 'band')
    np.testing.assert_allclose(93.0 * factors, 0.99, rtol=1e-6)
    with pytest.raises(ValueError, match='no signal'):
        factors_from_lines(np.zeros_like(lines), (2, 6), target, plan.wavelengths)
    with pytest.raises(ValueError):
        reference_values(99, plan.wavelengths)                            # percent, not a fraction
    pipeline = Pipeline(config, [src], [plan], ValueClassifier())
    pipeline.set_reflectance_factors(factors, 'drawn')
    assert pipeline.effective_steps()['reflectance'] and plan.steps['reflectance']['available']
    with pytest.raises(ValueError, match='do not fit'):
        pipeline.set_reflectance_factors(np.ones(3, np.float32), 'bad')


def test_profile_is_bound_to_the_camera_values_it_was_made_at(tmp_path):
    from reflectance_builder import backup_profile, save_profile
    config, src = reflectance_setup(tmp_path, 0.5, rowwise=False)
    probe = build_plan(config, src)
    folder = tmp_path / 'reflectance'
    moved = backup_profile(folder / 'x.npy')
    assert [p.name.split('_')[0] for p in moved] == ['x', 'x'] and not (folder / 'x.npy').exists()
    save_profile(folder, ['x'], np.full(probe.n_columns, 0.5, np.float32), probe.wavelengths,
                 {'exposure_ms': 2.0, 'gain': 4.0})                       # made at 2 ms, gain 4
    plan = build_plan(config, src)                                       # frames at 1 ms, gain 1
    np.testing.assert_allclose(plan.reflectance_factors, 1.0)            # half the exposure: x 2
    why = plan.steps['reflectance']['why']
    assert 'scaled from 2 ms to 1 ms' in why and 'WARNING made at gain 4' in why
