"""Smile + keystone models in the live pipeline: the geometry window and the Keystone step.

hsi (``modules/acquisition/engine.py``) crops the keystone margin rows at acquisition, leaves out
the bins that read the Smile border columns, and switches the keystone off with ``smile_only()``.
Here the crop and the trim are fixed by the loaded model for the whole run, so switching Smile or
Keystone changes only the remap, never the size of the model input.
"""
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.smile import SmileCalibration  # noqa: E402
from modules.smile.calibration import margin_from_map, save_calibration_fast  # noqa: E402
from pipeline import Pipeline  # noqa: E402
from sim_config import CameraConfig, CorrectionConfig, ProcessingConfig, SimulationConfig, \
    build_plan, geometry_model, keystone_margin  # noqa: E402
from sources import FrameSource  # noqa: E402

SHAPE = (8, 40)
MARGIN = (2, 1)
ROWS = SHAPE[0] - sum(MARGIN)


def keystone_model(shape=SHAPE, margin=MARGIN, shift=0.0, keystone=True):
    """Smile (+ keystone) model: columns shifted by ``shift``; keystone margin rows read outside."""
    h, w = shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_y = ys.copy()
    map_y[:margin[0]] = -10.0
    map_y[h - margin[1]:] = h + 10.0
    map_x = xs + np.float32(shift)
    return SmileCalibration(
        map_x=map_x, margin=margin_from_map(map_x, 'linear'), image_shape=shape, deg_y=0, deg_x=0,
        surface_coef=np.array([shift]), y_rng=(0.0, h - 1.0), x_rng=(0.0, w - 1.0), interp='linear',
        fit_rms=0.0, cv_rms=0.0, rms_before=0.0, rms_after=0.0, n_lines_used=0, n_lines_total=0,
        created='test', map_y=map_y if keystone else None,
        keystone={'row_margin': list(margin)} if keystone else None)


class RowClassifier:
    """Class id = binned value of each slit row's first column (the applied geometry is visible)."""
    category_names = {}

    def predict(self, binned, columns):
        return np.minimum(254, np.asarray(binned[:, 0])).astype(np.uint8)


def keystone_plan(tmp_path, backend='cpu', model=None, frames=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / 'smile_model_bin2.npz'
    save_calibration_fast(str(path), model if model is not None else keystone_model())
    frames = np.full((30, *SHAPE), 100, np.uint16) if frames is None else frames
    src = FrameSource('x.h5', frames, 'raw', metadata={'camera_bin': 2, 'exposure_ms': 1.0, 'gain': 1.0})
    config = replace(SimulationConfig(base_dir=str(tmp_path)),
                     camera=CameraConfig(fps=0, overrun='block', buffer_frames=4, loop=False),
                     corrections=CorrectionConfig(backend=backend, smile=str(path)),
                     processing=ProcessingConfig(workers=2, profile_every_s=0)).validate()
    plan = build_plan(config, src)
    plan.columns = None
    return config, src, plan


def run_to_end(pipeline, timeout=20):
    pipeline.start()
    t = time.perf_counter()
    while pipeline.snapshot().state in ('starting', 'running') and time.perf_counter() - t < timeout:
        time.sleep(0.02)
    pipeline.stop()
    return pipeline.snapshot()


def test_plan_without_keystone_model_crops_nothing(tmp_path):
    src = FrameSource('x.h5', np.zeros((1, *SHAPE), np.uint16), 'raw')
    plan = build_plan(SimulationConfig(base_dir=str(tmp_path)).validate(), src)
    assert plan.steps['keystone']['available'] is False and plan.steps['keystone']['on'] is False
    assert plan.keystone_crop == (0, 0) and plan.n_slit == SHAPE[0]


def test_margin_follows_hsi_crop_rule():
    model = keystone_model()
    assert keystone_margin(model) == MARGIN
    assert keystone_margin(None) == (0, 0)
    assert keystone_margin(model.smile_only()) == (0, 0)
    assert keystone_margin(keystone_model(margin=(5, 3))) == (0, 0)      # would leave no rows


def test_plan_crops_keystone_rows(tmp_path):
    _, _, plan = keystone_plan(tmp_path)
    assert plan.steps['keystone'] == {'available': True, 'on': True, 'why': plan.steps['keystone']['why']}
    assert plan.keystone_crop == MARGIN and plan.n_slit == ROWS
    assert plan.slit_rows == slice(MARGIN[0], SHAPE[0] - MARGIN[1])
    assert 'cropped' in plan.steps['keystone']['why']


def test_geometry_model_switches(tmp_path):
    _, _, plan = keystone_plan(tmp_path)
    model = plan.smile_model
    assert geometry_model(plan, {'smile': False, 'keystone': True}) is None
    assert geometry_model(plan, {'smile': True, 'keystone': True}) is model
    view = geometry_model(plan, {'smile': True, 'keystone': False})
    assert not view.has_keystone and view.map_y is None
    assert geometry_model(plan, {'smile': True, 'keystone': False}) is view          # built once


@pytest.mark.parametrize('backend', ['cpu', 'cpu_fused'])
@pytest.mark.parametrize('steps', [dict(keystone=True), dict(keystone=False), dict(smile=False)])
def test_every_switch_keeps_the_cropped_size(tmp_path, backend, steps):
    config, src, plan = keystone_plan(tmp_path, backend)
    pipeline = Pipeline(config, [src], [plan], RowClassifier())
    pipeline.set_steps(**steps)
    assert pipeline.slit_rows == ROWS
    if steps.get('smile') is False:
        assert pipeline.effective_steps()['keystone'] is False
    s = run_to_end(pipeline)
    assert s.state == 'finished' and s.lines_processed == 30
    assert s.canvas.shape[0] == ROWS
    assert s.canvas[:, -1].tolist() == [100] * ROWS            # no zero-filled margin row left


def test_switching_while_running_does_not_break(tmp_path):
    frames = np.full((400, *SHAPE), 100, np.uint16)
    config, src, plan = keystone_plan(tmp_path, frames=frames)
    config = replace(config, camera=replace(config.camera, fps=400))
    pipeline = Pipeline(config, [src], [plan], RowClassifier())
    pipeline.start()
    for steps in (dict(keystone=False), dict(smile=False), dict(smile=True), dict(keystone=True)):
        time.sleep(0.15)
        pipeline.set_steps(**steps)
    t = time.perf_counter()
    while pipeline.snapshot().state in ('starting', 'running') and time.perf_counter() - t < 20:
        time.sleep(0.02)
    pipeline.stop()
    s = pipeline.snapshot()
    assert s.state == 'finished' and s.lines_processed == 400
    assert s.canvas.shape[0] == ROWS


def test_smile_border_bands_are_left_out(tmp_path):
    shifted = keystone_model(shift=3.0, keystone=False)            # last 3 columns read past the sensor
    assert shifted.margin == (0, 3)
    _, _, plain = keystone_plan(tmp_path / 'a', model=keystone_model(keystone=False))
    _, _, plan = keystone_plan(tmp_path / 'b', model=shifted)
    assert plan.smile_border == (0, 3) and plan.keystone_crop == (0, 0)
    assert plan.binner.n_bins < plain.binner.n_bins
    assert np.all(plan.binner.iR <= SHAPE[1] - 3)
    assert 'Smile border' in plan.provenance['binning'][0]
