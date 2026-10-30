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
    # A real hsi model of another frame size (552 x 804) than the test frames.
    import shutil
    shutil.copy(ROOT / 'calibration' / 'smile_maps' / 'poa' / 'smile_model_bin2.npz', smile / 'smile_model_bin2.npz')


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
