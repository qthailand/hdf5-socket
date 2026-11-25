# -*- coding: utf-8 -*-
"""hsi's camera_profiles folder copied in as is: dark, Smile, wavelength, radiometric, flat-field."""
import datetime
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsi_profiles import find_camera_profile, hsi_profile_name, scan_key  # noqa: E402
from modules.radiometric.reference import RadiometricReferenceStore  # noqa: E402
from pipeline import _effective  # noqa: E402
from sim_config import AcquisitionConfig, CorrectionConfig, SimulationConfig, build_plan, \
    calibration_folders  # noqa: E402
from sources import FrameSource  # noqa: E402

HSI = ROOT.parent / 'hsi'
SHAPE = (6, 40)
META = {'camera_backend': 'poa', 'camera_model': 'Apollo-M MAX', 'camera_serial': 'CAMP 7', 'camera_bin': 2,
        'raw_inverted': True, 'exposure_ms': 1.0, 'gain': 1.0, 'full_scale': 65520}


def make_config(tmp_path, **corrections):
    cor = CorrectionConfig(backend='cpu', dark='auto', smile='auto', reflectance='auto', **corrections)
    acq = AcquisitionConfig(camera_backend='poa', m_nm_per_px_bin1=5.0, c_nm=400.0, resolution_nm=10.0)
    return replace(SimulationConfig(base_dir=str(tmp_path)), corrections=cor, acquisition=acq).validate()


def source(**meta):
    return FrameSource('run.h5', np.full((2,) + SHAPE, 1000, np.uint16), 'raw', metadata={**META, **meta})


def camera_folder(tmp_path, backend='poa', name='Apollo-M_MAX_CAMP_7'):
    folder = tmp_path / 'camera_profiles' / backend / name
    folder.mkdir(parents=True)
    (folder / 'profile.json').write_text(json.dumps(
        {'format_version': 1, 'backend': 'poa', 'model': 'Apollo-M MAX', 'serial': 'CAMP 7'}), encoding='utf-8')
    return folder


def save_dark(folder, value, captured_at=None):
    folder.mkdir(parents=True, exist_ok=True)
    extra = {'captured_at': np.array(captured_at)} if captured_at else {}
    hot = np.zeros(SHAPE, np.uint8)
    hot[1, 2] = 255
    np.savez(folder / 'dark_b2_e1.0_g1.0.npz', mean=np.full(SHAPE, value, np.uint16), hot_pixel_map=hot,
             bin=np.int32(2), exposure=np.float32(1), gain=np.float32(1), n_frames=np.int32(5), **extra)


def save_reference(folder, plan, meta, factors=None, wavelengths=None, acquired=None, **key_changes):
    """An active hsi reference made for ``plan`` (hsi's own store writes it)."""
    store = RadiometricReferenceStore(folder / 'radiometric')
    key = replace(scan_key(plan, meta, plan.wavelengths if wavelengths is None else wavelengths), **key_changes)
    if factors is None:
        factors = np.full((key.lines, len(key.wavelengths)), 0.25, np.float32)
    now = acquired or datetime.datetime.now().astimezone()
    return store.save(factors, key, reflectance_profile='Spectralon_ref-040.xlsx', avg_window_size=4,
                      mask=np.ones((key.lines, 3), bool), acquired_at=now, created_at=now, activate=True)


def test_folder_name_is_hsi_rule():
    assert hsi_profile_name('Apollo-M MAX', 'CAMP7161A470A1209000') == 'Apollo-M_MAX_CAMP7161A470A1209000'
    assert hsi_profile_name('fxo992MXGE-T', '151498') == 'fxo992MXGE-T_151498'
    assert hsi_profile_name(' MQ022 HG/IM ', '12:34') == 'MQ022_HG_IM_12_34'
    assert hsi_profile_name('M', None) == 'M' and hsi_profile_name(None, '5') == 'SN5'
    assert hsi_profile_name(b'..', '') == 'unknown'


def test_camera_folder_found_by_name_or_profile_json(tmp_path):
    config = make_config(tmp_path)
    folder = camera_folder(tmp_path)
    assert find_camera_profile(config, 'POA', 'Apollo-M MAX', 'CAMP 7') == folder
    moved = tmp_path / 'camera_profiles' / 'other' / 'renamed'
    shutil.move(str(folder), str(moved))
    assert find_camera_profile(config, 'poa', 'Apollo-M MAX', 'CAMP 7') == moved       # by profile.json
    assert find_camera_profile(config, 'poa', 'Apollo-M MAX', None) is None             # no unit named
    assert find_camera_profile(config, 'poa', 'Apollo-M MAX', 'CAMP 8') is None


def test_hsi_camera_folder_comes_first_and_falls_back_per_file(tmp_path):
    folder = camera_folder(tmp_path)
    (folder / 'dark').mkdir()                                     # empty, as hsi's often are
    save_dark(tmp_path / 'calibration' / 'dark_frames' / 'poa', 7)
    config = make_config(tmp_path)
    plan = build_plan(config, source())
    assert plan.hsi_profile_dir == folder
    assert [label for _, label in calibration_folders(config, plan, 'dark')] == [
        'hsi camera_profiles/poa/Apollo-M_MAX_CAMP_7', 'calibration/dark_frames/poa']
    assert int(plan.dark_mean[0, 0]) == 7 and 'calibration/dark_frames/poa' in plan.provenance['dark'][0]
    save_dark(tmp_path / 'dark_frames' / 'poa', 9)                # hsi's own older folder, copied too
    assert int(build_plan(config, source()).dark_mean[0, 0]) == 9
    save_dark(folder / 'dark', 11)                                # the camera's own dark wins
    plan = build_plan(config, source())
    assert int(plan.dark_mean[0, 0]) == 11 and 'hsi camera_profiles/poa' in plan.provenance['dark'][0]


def test_wavelength_from_hsi_folder_and_its_camera_setting(tmp_path):
    folder = camera_folder(tmp_path)
    (folder / 'wavelength').mkdir()
    record = {'m': 8.0, 'c': 410.0, 'camera_bin': 2, 'frame_shape': list(SHAPE),
              'camera_setting': {'camera_backend': 'poa', 'camera_index': 3, 'camera_bin': 2,
                                 'raw_inverted': True, 'frame_shape': list(SHAPE)}}
    (folder / 'wavelength' / 'wavelength_bin2.json').write_text(json.dumps(record), encoding='utf-8')
    config = replace(make_config(tmp_path), acquisition=replace(make_config(tmp_path).acquisition,
                                                                wavelength_profile='auto'))
    plan = build_plan(config, source(m_nm_per_px_bin1=None, c_nm=None))
    assert plan.m_nm_per_px_bin1 == 4.0 and plan.c_nm == 410.0        # camera index is not compared
    assert 'hsi poa/Apollo-M_MAX_CAMP_7' in plan.provenance['wavelength m,c'][1]
    with pytest.raises(ValueError, match='raw_inverted'):
        build_plan(config, source(raw_inverted=False))


def test_active_hsi_reference_is_applied_as_hsi_applies_it(tmp_path):
    folder = camera_folder(tmp_path)
    save_dark(folder / 'dark', 7, captured_at='2026-11-20T10:00:00+07:00')
    config = make_config(tmp_path)
    meta = source().metadata
    plan = build_plan(config, source())
    assert plan.dark_reference == 'Apollo-M_MAX_CAMP_7/dark/dark_b2_e1.0_g1.0.npz @ 2026-11-20T10:00:00+07:00'
    factors = np.linspace(0.1, 0.9, plan.n_slit * plan.n_columns, dtype=np.float32).reshape(plan.n_slit, -1)
    ref = save_reference(folder, plan, meta, factors)
    plan = build_plan(config, source())
    step = plan.steps['reflectance']
    assert step['on'] and step['why'].startswith(f'hsi reference {ref.id}'), step['why']
    np.testing.assert_array_equal(plan.reflectance_factors, factors)          # not scaled, not copied wrong
    assert plan.reflectance_requires == {'dark': True, 'flip': True, 'smile': False, 'keystone': False}
    on = _effective(plan, {})
    assert on['reflectance'] and not _effective(plan, {'flip': False})['reflectance']   # hsi key: orientation
    assert not _effective(plan, {'dark': False})['reflectance']


@pytest.mark.parametrize('meta, reason', [({'exposure_ms': 2.0}, 'exposure: scan 2 ms, reference 1 ms'),
                                          ({'gain': 3.0}, 'gain'), ({'full_scale': 4095}, 'data full scale')])
def test_hsi_reference_of_another_setting_is_refused(tmp_path, meta, reason):
    folder = camera_folder(tmp_path)
    save_dark(folder / 'dark', 7)
    config = make_config(tmp_path)
    plan = build_plan(config, source())
    save_reference(folder, plan, source().metadata)
    # a dark of the other exposure / gain exists, so only the reference's key differs
    if 'exposure_ms' in meta or 'gain' in meta:
        name = 'dark_b2_e2.0_g1.0.npz' if 'exposure_ms' in meta else 'dark_b2_e1.0_g3.0.npz'
        shutil.copy(folder / 'dark' / 'dark_b2_e1.0_g1.0.npz', folder / 'dark' / name)
    step = build_plan(config, source(**meta)).steps['reflectance']
    assert not step['available'] and 'hsi radiometric' in step['why'] and reason in step['why'], step['why']


def test_hsi_reference_of_a_wider_band_grid_gives_the_streams_bands(tmp_path):
    folder = camera_folder(tmp_path)
    save_dark(folder / 'dark', 7)
    base = make_config(tmp_path)
    full = build_plan(base, source())
    factors = np.tile(np.arange(full.n_columns, dtype=np.float32) + 1, (full.n_slit, 1))
    save_reference(folder, full, source().metadata, factors)
    lo, hi = float(full.wavelengths[3]) - 1, float(full.wavelengths[9]) + 1
    config = replace(base, acquisition=replace(base.acquisition, wavelength_range_nm=(lo, hi)))
    plan = build_plan(config, source())
    assert plan.steps['reflectance']['on'], plan.steps['reflectance']['why']
    k0 = int(np.flatnonzero(np.isclose(full.wavelengths, plan.wavelengths[0]))[0])
    assert 0 < k0 and plan.n_columns < full.n_columns
    np.testing.assert_array_equal(plan.reflectance_factors[0], factors[0, k0:k0 + plan.n_columns])


def test_own_newer_calibration_wins_over_an_older_hsi_reference(tmp_path):
    from reflectance_builder import save_profile
    folder = camera_folder(tmp_path)
    save_dark(folder / 'dark', 7)
    config = make_config(tmp_path)
    plan = build_plan(config, source())
    save_reference(folder, plan, source().metadata,
                   acquired=datetime.datetime.now().astimezone() - datetime.timedelta(hours=2))
    own = tmp_path / 'profiles' / plan.camera_id / 'reflectance'
    save_profile(own, ['refl_b2_g1.0'], np.full(plan.n_columns, 0.5, np.float32), plan.wavelengths,
                 {'camera_bin': 2, 'gain': 1.0, 'exposure_ms': 1.0})
    step = build_plan(config, source()).steps['reflectance']
    assert step['why'].startswith('refl_b2_g1.0.npy') and 'is newer' in step['why']


def test_flatfield_of_the_hsi_camera_folder_is_offered(tmp_path):
    from modules.radiometric.core import save_factor_profile
    folder = camera_folder(tmp_path)
    save_dark(folder / 'dark', 7)
    config = make_config(tmp_path)
    plan = build_plan(config, source())
    key = scan_key(plan, source().metadata, plan.wavelengths).to_dict()
    flat = folder / 'flatfield' / 'flat_20261120_100000'
    flat.mkdir(parents=True)
    save_factor_profile(flat / 'profile.npy', np.ones((plan.n_slit, plan.n_columns), np.float32), plan.wavelengths)
    np.save(flat / 'roi_mask.npy', np.ones((plan.n_slit, 3), bool))
    (flat / 'flatfield.json').write_text(json.dumps({
        'format_version': 1, 'id': flat.name, 'created_at': '2026-11-20T10:00:00+07:00',
        'acquired_at': '2026-11-20T09:59:00+07:00', 'key': key, 'roi': {'span': [0, 5]}}), encoding='utf-8')
    plan = build_plan(config, source())
    assert plan.steps['flatfield']['available'], plan.steps['flatfield']['why']
    assert plan.flatfield_profile.label.endswith('flatfield.json')


@pytest.mark.skipif(not (HSI / 'camera_profiles' / 'poa').is_dir(), reason='hsi project not next to this one')
def test_real_hsi_camera_profiles_copied_as_is(tmp_path):
    """hsi's own camera_profiles (and its older dark_frames) copied unchanged: the POA camera gets
    its Smile model, wavelength calibration and active radiometric reference."""
    shutil.copytree(HSI / 'camera_profiles', tmp_path / 'camera_profiles')
    if (HSI / 'dark_frames').is_dir():
        shutil.copytree(HSI / 'dark_frames', tmp_path / 'dark_frames')
    folder = tmp_path / 'camera_profiles' / 'poa' / 'Apollo-M_MAX_CAMP7161A470A1209000'
    active = json.loads((folder / 'radiometric' / 'active.json').read_text(encoding='utf-8'))['active']
    if not active or not (folder / 'smile' / 'smile_model_bin2.npz').is_file():
        pytest.skip('hsi POA profile has no active reference / bin 2 Smile model')
    config = replace(SimulationConfig(base_dir=str(tmp_path)), corrections=CorrectionConfig(
        backend='cpu', dark='auto', smile='auto', reflectance='auto'), acquisition=AcquisitionConfig(
        camera_backend='poa', wavelength_profile='auto', resolution_nm=2.0)).validate()
    src = FrameSource('run.h5', np.full((2, 552, 804), 1000, np.uint16), 'raw', metadata={
        'camera_backend': 'poa', 'camera_model': 'Apollo-M MAX', 'camera_serial': 'CAMP7161A470A1209000',
        'camera_bin': 2, 'raw_inverted': True, 'exposure_ms': 1.0, 'gain': 1.0, 'full_scale': 65520})
    plan = build_plan(config, src)
    assert 'hsi poa/Apollo-M_MAX_CAMP7161A470A1209000' in plan.provenance['smile'][0]
    assert 'hsi poa/Apollo-M_MAX_CAMP7161A470A1209000' in plan.provenance['wavelength m,c'][1]
    if plan.dark_mean is None:
        pytest.skip('no hsi dark for bin 2, 1 ms, gain 1')
    step = plan.steps['reflectance']
    assert step['on'] and step['why'].startswith(f'hsi reference {active[0]}'), step['why']
    ref = np.load(folder / 'radiometric' / active[0] / 'factors.npy')
    k0 = int(np.argmin(np.abs(np.load(folder / 'radiometric' / active[0] / 'factors.wavelength.npy')
                              - plan.wavelengths[0])))
    np.testing.assert_array_equal(plan.reflectance_factors, ref[:, k0:k0 + plan.n_columns])


def test_full_scale_is_hsis_bit_rule_not_the_saturation_level():
    from hsi_profiles import scan_full_scale
    # live POA: 12-bit sensor in 16-bit frames, saturation level 65535 -> hsi key 65520
    assert scan_full_scale({'bit_depth': 16, 'sensor_bit_depth': 12, 'full_scale': 65535}) == 65520
    assert scan_full_scale({'bit_depth': 16, 'sensor_bit_depth': 10, 'full_scale': 65520}) == 65472  # fxo
    assert scan_full_scale({'full_scale': 65472}) == 65472 and scan_full_scale({}) is None


def test_offered_dark_is_switched_on_for_a_wanted_hsi_reference(tmp_path):
    folder = camera_folder(tmp_path)
    save_dark(folder / 'dark', 7)
    plan = build_plan(make_config(tmp_path), source())
    save_reference(folder, plan, source().metadata)
    config = make_config(tmp_path)
    config = replace(config, corrections=replace(config.corrections, dark='none'))   # dark offered off
    plan = build_plan(config, source())
    assert plan.steps['reflectance']['on'] and plan.steps['dark']['on'], plan.steps['reflectance']['why']
    assert 'Dark on' in plan.steps['reflectance']['why']
    assert _effective(plan, {})['reflectance'] and not _effective(plan, {'dark': False})['reflectance']


def test_unreadable_dark_in_the_camera_folder_is_not_replaced(tmp_path):
    folder = camera_folder(tmp_path)
    save_dark(tmp_path / 'calibration' / 'dark_frames' / 'poa', 7)
    (folder / 'dark').mkdir()
    (folder / 'dark' / 'dark_b2_e1.0_g1.0.npz').write_bytes(b'not a dark')          # interrupted copy
    plan = build_plan(make_config(tmp_path), source())
    assert plan.dark_mean is None and 'unreadable' in plan.provenance['dark'][0]


def test_newer_own_factors_that_do_not_fit_leave_the_hsi_reference(tmp_path):
    from reflectance_builder import save_profile
    folder = camera_folder(tmp_path)
    save_dark(folder / 'dark', 7)
    config = make_config(tmp_path)
    plan = build_plan(config, source())
    ref = save_reference(folder, plan, source().metadata,
                         acquired=datetime.datetime.now().astimezone() - datetime.timedelta(hours=2))
    own = tmp_path / 'profiles' / plan.camera_id / 'reflectance'
    save_profile(own, ['refl_b2_g1.0'], np.full(plan.n_columns + 1, 0.5, np.float32),   # other band axis
                 np.arange(plan.n_columns + 1, dtype=np.float64) * 10 + 400, {'camera_bin': 2, 'gain': 1.0})
    step = build_plan(config, source()).steps['reflectance']
    assert step['on'] and step['why'].startswith(f'hsi reference {ref.id}'), step['why']


def test_newest_reference_with_nan_factors_is_not_replaced_by_an_older_one(tmp_path):
    folder = camera_folder(tmp_path)
    save_dark(folder / 'dark', 7)
    config = make_config(tmp_path)
    plan = build_plan(config, source())
    now = datetime.datetime.now().astimezone()
    save_reference(folder, plan, source().metadata, acquired=now - datetime.timedelta(hours=3))
    nan = np.full((plan.n_slit, plan.n_columns), 0.25, np.float32)
    nan[:, 0] = np.nan
    store = RadiometricReferenceStore(folder / 'radiometric')
    key = scan_key(plan, source().metadata, plan.wavelengths)
    newest = store.save(nan, key, reflectance_profile='t.xlsx', avg_window_size=4, mask=np.ones((plan.n_slit, 3), bool),
                        acquired_at=now, created_at=now + datetime.timedelta(seconds=5))
    data = json.loads(store.active_path.read_text(encoding='utf-8'))
    store.active_path.write_text(json.dumps({**data, 'active': data['active'] + [newest.id]}), encoding='utf-8')
    step = build_plan(config, source()).steps['reflectance']
    assert not step['available'] and f'{newest.id} not applied' in step['why'] and 'NaN' in step['why']


def test_hsis_older_dark_folder_comes_before_this_projects_profile(tmp_path):
    folder = camera_folder(tmp_path)
    (folder / 'dark').mkdir()
    save_dark(tmp_path / 'dark_frames' / 'poa', 9)                   # what hsi uses
    config = make_config(tmp_path)
    plan = build_plan(config, source())
    save_dark(tmp_path / 'profiles' / plan.camera_id / 'dark', 3)    # an older copy of this project
    plan = build_plan(config, source())
    assert int(plan.dark_mean[0, 0]) == 9 and 'hsi dark_frames/poa' in plan.provenance['dark'][0]


def test_own_calibration_keeps_the_switches_it_was_made_with(tmp_path):
    from reflectance_builder import save_profile
    save_dark(tmp_path / 'calibration' / 'dark_frames' / 'poa', 7)
    config = make_config(tmp_path)
    plan = build_plan(config, source())
    own = tmp_path / 'profiles' / plan.camera_id / 'reflectance'
    save_profile(own, ['refl_b2_g1.0'], np.full(plan.n_columns, 0.5, np.float32), plan.wavelengths,
                 {'camera_bin': 2, 'gain': 1.0, 'exposure_ms': 1.0,
                  'made_with': {'dark': True, 'flip': True, 'smile': False, 'keystone': False}})
    plan = build_plan(config, source())
    assert plan.reflectance_requires == {'dark': True, 'flip': True, 'smile': False, 'keystone': False}
    assert _effective(plan, {})['reflectance'] and not _effective(plan, {'flip': False})['reflectance']


def test_ticking_reflectance_that_its_steps_keep_off_leaves_flatfield_on():
    from types import SimpleNamespace
    from pipeline import Pipeline
    plan = SimpleNamespace(steps={'dark': {'available': True, 'on': True}, 'flip': {'available': True, 'on': True},
                                  'smile': {'available': True, 'on': False}, 'binning': {'available': False, 'on': True},
                                  'flatfield': {'available': True, 'on': True},
                                  'reflectance': {'available': True, 'on': False}},
                           reflectance_requires={'dark': True, 'flip': True, 'smile': True, 'keystone': False},
                           flatfield_profile=None)
    fake = SimpleNamespace(_lock=__import__('threading').Lock(), _steps={}, plans=[plan],
                           _invalidate_spectra_locked=lambda: None)
    Pipeline.set_steps(fake, reflectance=True)                   # Smile is off: Reflectance stays off
    assert fake._steps.get('flatfield') is None
    Pipeline.set_steps(fake, smile=True, reflectance=True)
    assert fake._steps['flatfield'] is False


def test_older_hsi_reference_takes_its_flatfield_from_the_copied_folder():
    """A reference stored before hsi kept flatfield.npy: F comes from flatfield/<id>/profile.npy,
    accepted because the factors were fitted with it; the ROI rows come from roi_mask.npy."""
    import numpy as np
    from hsi_profiles import HsiReference
    from modules.radiometric.reference import RadiometricReferenceStore
    folder = ROOT / 'camera_profiles' / 'genicam' / 'fxo992MXGE-T_151498'
    if not (folder / 'radiometric' / 'ref_20261118_111526').is_dir():
        pytest.skip('copied fxo camera folder not present')
    ref = RadiometricReferenceStore(folder / 'radiometric').load('ref_20261118_111526')
    bands = np.arange(10, 50)
    rows = slice(2, -2)
    wrapped = HsiReference(ref, ref.factors()[rows][:, bands], bands, folder, {}, rows)
    flat, weights, where = wrapped.flatfield_parts()
    assert flat.shape == wrapped.factors.shape and 'flat-field folder' in where
    assert weights.shape == (flat.shape[0],) and weights.sum() > 0
