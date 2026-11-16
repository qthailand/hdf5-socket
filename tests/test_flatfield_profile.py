"""Portable profile matching, automatic selection and the actual worker correction."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flatfield_profile import configure_flatfield, install_flatfield, load_flatfield
from pipeline import Pipeline, _effective
from sim_config import camera_profile_name, ensure_profile_structure, SimulationConfig
from test_simulation import dark_plan, run_to_end, ValueClassifier


def setup(tmp_path):
    config, src, plan = dark_plan(tmp_path, dark='auto')
    plan.camera_id = camera_profile_name('Test camera', '123')
    plan.profile_dir = tmp_path / 'profiles' / plan.camera_id
    return config, src, plan


def write_profile(plan, name='flat_20261113_100000', **changes):
    folder = plan.profile_dir / 'flatfield' / name
    folder.mkdir(parents=True)
    key = dict(camera_backend=plan.camera_backend, camera_model='Test camera', camera_serial='123',
               camera_bin=plan.camera_bin, lines=plan.frame_shape[0], gain=plan.gain,
               raw_inverted=plan.raw_inverted, resolution_nm=plan.resolution_nm,
               dark_corrected=True, smile_corrected=False, hot_pixel_corrected=plan.hot_map is not None,
               wavelengths=plan.wavelengths.tolist())
    key.update(changes)
    data = dict(format_version=1, created_at='2026-11-13T10:00:00+07:00', key=key)
    (folder / 'flatfield.json').write_text(json.dumps(data), encoding='utf-8')
    np.save(folder / 'profile.npy', np.full((plan.frame_shape[0], plan.n_columns), 2, np.float32))
    np.save(folder / 'profile.wavelength.npy', plan.wavelengths)
    return folder


def test_copied_folder_loads_without_hsi_or_roi_file(tmp_path):
    _, _, plan = setup(tmp_path)
    folder = write_profile(plan)
    profile = load_flatfield(folder, plan)
    np.testing.assert_array_equal(profile.values, 2)
    assert not profile.values.flags.writeable


@pytest.mark.parametrize('changes', [dict(camera_serial='other'), dict(camera_bin=99), dict(gain=99),
    dict(dark_corrected=False), dict(hot_pixel_corrected=False), dict(wavelengths=[400.0])])
def test_incompatible_profiles_are_refused(tmp_path, changes):
    _, _, plan = setup(tmp_path)
    with pytest.raises(ValueError):
        load_flatfield(write_profile(plan, **changes), plan)


@pytest.mark.parametrize('bad', [0, -1, np.nan, np.inf])
def test_corrupt_values_are_refused(tmp_path, bad):
    _, _, plan = setup(tmp_path)
    folder = write_profile(plan)
    np.save(folder / 'profile.npy', np.full((plan.frame_shape[0], plan.n_columns), bad))
    with pytest.raises(ValueError, match='positive, finite'):
        load_flatfield(folder, plan)


def test_auto_selects_newest_compatible_profile_and_skips_bad_one(tmp_path):
    config, _, plan = setup(tmp_path)
    older = write_profile(plan)
    write_profile(plan, 'flat_20261113_110000', gain=99)
    configure_flatfield(config, plan)
    assert plan.steps['flatfield']['available'] and not plan.steps['flatfield']['on']
    assert plan.flatfield_profile.label == str(older / 'flatfield.json')


def test_missing_profile_is_disabled(tmp_path):
    config, _, plan = setup(tmp_path)
    configure_flatfield(config, plan)
    assert not plan.steps['flatfield']['available']


def test_auto_loads_files_placed_directly_in_flatfield_folder(tmp_path):
    config, _, plan = setup(tmp_path)
    source = write_profile(plan)
    for path in source.iterdir():
        shutil.copy2(path, source.parent / path.name)
    configure_flatfield(config, plan)
    assert plan.steps['flatfield']['available']
    assert plan.flatfield_profile.values.shape == (plan.frame_shape[0], plan.n_columns)


def test_auto_loads_renamed_hsi_profile_subfolder(tmp_path):
    config, _, plan = setup(tmp_path)
    source = write_profile(plan)
    copied = source.parent / 'copied-from-HSI'
    copied.mkdir()
    for path in source.iterdir():
        shutil.copy2(path, copied / path.name)
    # A later timestamp chooses the renamed folder, without depending on its name.
    data = json.loads((copied / 'flatfield.json').read_text())
    data['created_at'] = '2026-11-13T11:00:00+07:00'
    (copied / 'flatfield.json').write_text(json.dumps(data))
    configure_flatfield(config, plan)
    assert plan.flatfield_profile.label == str(copied / 'flatfield.json')


def test_hsi_backend_camera_folder_uses_metadata_identity_not_folder_spelling(tmp_path):
    config, _, plan = setup(tmp_path)
    source = write_profile(plan)
    copied = tmp_path / 'profiles' / 'poa' / 'Test_camera_123' / 'flatfield'
    copied.mkdir(parents=True)
    for path in source.iterdir():
        shutil.copy2(path, copied / path.name)
    data = json.loads((copied / 'flatfield.json').read_text())
    data['created_at'] = '2026-11-13T11:00:00+07:00'
    (copied / 'flatfield.json').write_text(json.dumps(data))
    configure_flatfield(config, plan)
    assert plan.flatfield_profile.label == str(copied / 'flatfield.json')


def test_full_hsi_grid_can_supply_a_cropped_sorter_range(tmp_path):
    _, _, plan = setup(tmp_path)
    folder = write_profile(plan)
    wavelengths = plan.wavelengths.copy()
    values = np.arange(plan.frame_shape[0] * plan.n_columns, dtype=np.float32).reshape(plan.frame_shape[0], -1) + 1
    np.save(folder / 'profile.npy', values)
    plan.wavelengths = wavelengths[1:-1]
    profile = load_flatfield(folder, plan)
    np.testing.assert_array_equal(profile.values, values[:, 1:-1])


def test_rejection_reason_is_available_in_provenance(tmp_path):
    config, _, plan = setup(tmp_path)
    write_profile(plan, camera_serial='other')
    configure_flatfield(config, plan)
    assert 'camera identity' in plan.steps['flatfield']['why']
    assert 'camera identity' in plan.provenance['flatfield'][0]


@pytest.mark.parametrize('broken', ['metadata', 'array', 'sidecar'])
def test_auto_skips_corrupt_profiles_without_stopping_startup(tmp_path, broken):
    config, _, plan = setup(tmp_path)
    folder = write_profile(plan)
    if broken == 'metadata':
        (folder / 'flatfield.json').write_text('[]')
    elif broken == 'array':
        (folder / 'profile.npy').write_bytes(b'')
    else:
        (folder / 'profile.wavelength.npy').unlink()
    configure_flatfield(config, plan)
    assert not plan.steps['flatfield']['available']


def test_startup_creates_missing_structure_and_preserves_files(tmp_path):
    config = SimulationConfig(base_dir=str(tmp_path))
    ensure_profile_structure(config)
    root = tmp_path / 'profiles'
    assert root.is_dir()
    ensure_profile_structure(config, 'Test-123')
    marker = root / 'Test-123' / 'flatfield' / 'keep.txt'
    marker.write_text('keep')
    (root / 'Other-456').mkdir()
    ensure_profile_structure(config)
    assert (root / 'Other-456' / 'flatfield').is_dir()
    assert marker.read_text() == 'keep'


def test_correction_dependencies_and_rowwise_reflectance(tmp_path):
    _, _, plan = setup(tmp_path)
    install_flatfield(plan, load_flatfield(write_profile(plan), plan), True)
    assert _effective(plan, {})['flatfield']
    assert not _effective(plan, {'dark': False})['flatfield']
    plan.reflectance_factors = np.ones((plan.frame_shape[0], plan.n_columns), np.float32)
    plan.steps['reflectance'] = dict(available=True, on=True)
    assert not _effective(plan, {})['flatfield']
    assert _effective(plan, {'reflectance': False})['flatfield']
    plan.reflectance_factors = np.ones(plan.n_columns, np.float32)
    assert not _effective(plan, {})['flatfield']  # these factors were calibrated on pre-flat counts too


def test_workers_divide_only_when_checkbox_is_enabled(tmp_path):
    config, src, plan = setup(tmp_path)
    install_flatfield(plan, load_flatfield(write_profile(plan), plan))
    results = []
    for enabled in (False, True):
        pipeline = Pipeline(config, [src], [plan], ValueClassifier())
        pipeline.set_steps(flatfield=enabled)
        snapshot = run_to_end(pipeline)
        assert snapshot.error is None and snapshot.lines_lost == 0
        results.append(int(np.argmax(snapshot.counts)))
    assert results[0] > 0 and results[1] == results[0] // 2


def test_gui_has_only_checkbox_and_switches_off_rowwise_reflectance(tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QApplication, QPushButton
    from inference_engine import SpeedTestClassifier
    from themes import apply_theme
    import gui
    config, src, plan = setup(tmp_path)
    install_flatfield(plan, load_flatfield(write_profile(plan), plan))
    plan.reflectance_factors = np.ones((plan.frame_shape[0], plan.n_columns), np.float32)
    plan.steps['reflectance'] = dict(available=True, on=True, why='row-wise')
    classifier = SpeedTestClassifier()
    plan.columns, _ = classifier.bind(plan.wavelengths, plan.n_columns)
    pipeline = Pipeline(config, [src], [plan], classifier)
    monkeypatch.setattr(gui, 'build_pipeline', lambda config: pipeline)
    app = QApplication.instance() or QApplication([])
    apply_theme(app, 'dark')
    window = gui.PushbroomScannerMainWindow(config)
    try:
        window._gui_timer.stop()
        box = window._step_boxes['flatfield']
        assert box.isEnabled() and not box.isChecked()
        assert not any(b.text() == 'Flat-field…' for b in window.findChildren(QPushButton))
        box.click()
        assert box.isChecked() and pipeline.effective_steps()['flatfield']
        assert not window._step_boxes['reflectance'].isChecked()
        window._step_boxes['reflectance'].click()
        assert not box.isChecked() and not pipeline.effective_steps()['flatfield']
        box.click()
        window._step_boxes['dark'].click()
        assert not box.isChecked() and not pipeline.effective_steps()['flatfield']
    finally:
        pipeline.stop()
        window.close()
        app.processEvents()
