"""Calibration round trip and spectrum freshness after a processing change."""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reflectance_builder import factors_from_lines, quality
from pipeline import Pipeline, _Job
from test_simulation import dark_plan, ValueClassifier, run_to_end


def test_twenty_percent_standard_roundtrips_on_captured_roi():
    bands = 12
    white = np.linspace(1200, 6000, bands)[None, None, :] * np.ones((8, 10, 1))
    target = np.full(bands, .2)
    factors, reference = factors_from_lines(white, (2, 8), target, np.arange(bands), 'band')
    report = quality(reference, factors, target, rows=(2, 8))
    assert report['captured_roi_check']['calibrated_mean'] == pytest.approx(.2, rel=1e-6)
    assert report['captured_roi_check']['max_relative_error_pct'] < .001


def test_small_reference_with_flatfield_preserves_twenty_percent_on_whole_slit():
    response = np.linspace(.3, 1.5, 20)[:, None] * np.linspace(1000, 5000, 12)[None, :]
    flat = response / np.median(response, axis=0)
    lines = np.repeat((.2 * response)[None, :, :], 5, axis=0)
    target = np.full(12, .2)
    factors, reference = factors_from_lines(lines, (8, 12), target, np.arange(12), 'flatfield', flatfield=flat)
    np.testing.assert_allclose(lines * factors, .2, rtol=1e-6)
    report = quality(reference, factors, target, rows=(8, 12))
    assert report['captured_roi_check']['calibrated_mean'] == pytest.approx(.2, rel=1e-6)


def test_verify_lines_reports_agreement_offset_and_wavelength_error():
    from reflectance_builder import verify_lines
    bands, wl = 12, np.linspace(1000, 1600, 12)
    white = np.linspace(1200, 6000, bands)[None, None, :] * np.ones((8, 10, 1))
    target = np.linspace(.99, .95, bands)
    factors, _ = factors_from_lines(white, (2, 8), target, wl, 'band')
    good = verify_lines(white, (2, 8), factors, target, wl)
    np.testing.assert_allclose(good['measured'], target, rtol=1e-6)
    assert good['ok'] and good['max_abs_error_pct'] < .001 and good['pixels'] == 8 * 6
    dim = verify_lines(white * .9, (2, 8), factors, target, wl)          # lamp 10 % weaker
    assert not dim['ok'] and dim['mean_error_pct'] == pytest.approx(-10, abs=1e-3)
    assert dim['verdict'].startswith('Same offset')
    tilt = verify_lines(white * np.linspace(1, 1.2, bands), (2, 8), factors, target, wl)
    assert tilt['verdict'].startswith('Error changes with wavelength') and tilt['worst_nm'] == 1600
    rowwise = np.tile(factors, (10, 1))
    np.testing.assert_allclose(verify_lines(white, (2, 8), rowwise, target)['measured'], target, rtol=1e-6)
    with pytest.raises(ValueError, match='do not fit'):
        verify_lines(white, (2, 8), factors[:-1], target)


def test_clipped_raw_pixels_block_a_calibration_whose_binned_level_looks_fine():
    from reflectance_builder import signal_check
    rng = np.random.default_rng(0)
    lines = (30000 + 20 * rng.standard_normal((20, 12, 8))).astype(np.float32)   # binned mean: 46 % of range
    clipped = np.zeros((20, 12), np.uint16)
    clean = signal_check(lines, (2, 10), None, 65535.0, 320.0, clipped)
    assert clean['status'] == 'green' and clean['clipped_pixels'] == 0
    clipped[5, 0] = 9                                                     # outside the drawn rows: ignored
    assert signal_check(lines, (2, 10), None, 65535.0, 320.0, clipped)['status'] == 'green'
    clipped[5, 4], clipped[6, 4] = 3, 1
    hit = signal_check(lines, (2, 10), None, 65535.0, 320.0, clipped)
    assert hit['status'] == 'red' and hit['levels']['peak'] == 'red' and hit['clipped_pixels'] == 4
    assert 'Saturated: 4 raw pixels' in hit['issues'][0][1] and '2 of 160 line-rows' in hit['issues'][0][1]
    assert signal_check(lines, (2, 10), None, 65535.0, 320.0)['clipped_pixels'] is None   # not counted


def test_pipeline_counts_clipped_raw_pixels_per_slit_row_during_capture(tmp_path):
    config, source, plan = dark_plan(tmp_path, dark='auto')
    source.metadata['full_scale'] = 4095
    frames = np.array(source.frames)
    frames[:, 1, :3] = 4095                                               # raw row 1: three full pixels
    frames[:, 2, 0] = 3900                                                # 95.2 % of full scale: counted too
    source.frames = frames
    pipeline = Pipeline(config, [source], [plan], ValueClassifier())
    pipeline.start_calibration_capture(64)
    assert run_to_end(pipeline).error is None
    ids, lines, clipped = pipeline.calibration_capture(with_clipped=True)
    assert clipped.shape == lines.shape[:2] and len(ids) == clipped.shape[0] > 0
    expected = np.zeros(clipped.shape[1], np.uint16)
    expected[[1, 2]] = 3, 1
    if pipeline.effective_steps()['flip']:
        expected = expected[::-1]
    np.testing.assert_array_equal(clipped, np.tile(expected, (len(ids), 1)))
    assert len(pipeline.calibration_capture()) == 2                       # the old call is unchanged
    pipeline.stop_calibration_capture()
    assert pipeline.calibration_capture(with_clipped=True)[2] is None


def test_old_and_inflight_spectra_are_not_presented_after_recalibration(tmp_path):
    config, source, plan = dark_plan(tmp_path, dark='auto')
    pipeline = Pipeline(config, [source], [plan], ValueClassifier())
    pipeline.start_spectrum_history()
    ids = np.ones(source.frame_shape[0], np.uint8)
    spectra = np.full((len(ids), plan.n_columns), .06, np.float32)
    def job(n, generation):
        return _Job(n, n, 0, time.perf_counter(), 0, n, 0, time.perf_counter(), spectrum_generation=generation)
    before = pipeline._spectrum_generation
    pipeline._account(job(0, before), ids, 0, spec=spectra)
    assert pipeline.spectrum_at(0, 0)[0] == pytest.approx(.06)
    pipeline.set_reflectance_factors(np.ones(plan.n_columns, np.float32), 'new calibration')
    assert pipeline.spectrum_at(0, 0) is None
    pipeline._account(job(1, before), ids, 0, spec=spectra)
    assert pipeline.spectrum_at(1, 0) is None
    pipeline._account(job(2, pipeline._spectrum_generation), ids, 0, spec=np.full_like(spectra, .2))
    assert pipeline.spectrum_at(2, 0)[0] == pytest.approx(.2)


def test_reapplying_factors_while_reflectance_is_on_still_invalidates_history(tmp_path):
    config, source, plan = dark_plan(tmp_path, dark='auto')
    pipeline = Pipeline(config, [source], [plan], ValueClassifier())
    pipeline.set_reflectance_factors(np.ones(plan.n_columns, np.float32), 'first')
    first = pipeline._spectrum_generation
    pipeline.set_reflectance_factors(np.full(plan.n_columns, 2, np.float32), 'second')
    assert pipeline._spectrum_generation > first


def test_live_worker_returns_point_two_without_dividing_flatfield_twice(tmp_path):
    from flatfield_profile import FlatField, install_flatfield
    config, source, plan = dark_plan(tmp_path, dark='auto')
    # This divisor would turn .2 into .06 if an independent pre-flat calibration were
    # accidentally combined with standalone Flat-field.
    profile = FlatField(np.full((6, plan.n_columns), 10 / 3, np.float32), 'test', True, False, False)
    install_flatfield(plan, profile, True)
    pipeline = Pipeline(config, [source], [plan], ValueClassifier())
    pipeline.set_steps(flatfield=True)
    pipeline.set_reflectance_factors(np.full(plan.n_columns, .2 / 93, np.float32), '20% standard')
    assert pipeline.effective_steps()['reflectance'] and not pipeline.effective_steps()['flatfield']
    pipeline.start_spectrum_history()
    result = run_to_end(pipeline)
    assert result.error is None
    ids, spectra = pipeline.spectrum_snapshot()
    assert len(ids) == 30
    np.testing.assert_allclose(spectra, .2, rtol=1e-5)
