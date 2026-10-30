# smile/test_corrector.py
"""Tests for the SmileCorrector sklearn-style API."""
import numpy as np
import pytest
from smile.corrector import SmileCorrector

class TestSmileCorrector:
    def test_fit_anchors_displacement_at_reference_row(self, synthetic_frame):
        c = SmileCorrector(deg_y=2, deg_x=0, y_ref=49)
        c.fit(synthetic_frame)
        xs = np.arange(synthetic_frame.shape[1], dtype=float)
        shifts = c._surface(np.full_like(xs, 49.0), xs)
        np.testing.assert_allclose(shifts, 0.0, atol=1e-8)

    def test_even_height_defaults_to_ceil_half_height(self, synthetic_frame):
        frame = np.zeros((100, synthetic_frame.shape[1]), dtype=np.float32)
        yy = np.arange(100, dtype=float)
        for x0 in (40, 80, 120):
            center = x0 + 2.0 * ((yy - 49.0) / 49.0) ** 2
            for y, xc in enumerate(center):
                frame[y, int(round(xc))] = 100
        c = SmileCorrector(deg_y=2, deg_x=0, snr_min=1)
        c.fit(frame)
        assert c._analysis.y_ref == 50

    def test_reference_row_rule_for_even_and_odd_heights(self):
        from smile.analysis import _default_reference_row
        assert _default_reference_row(1104) == 552
        assert _default_reference_row(1105) == 551

    def test_fit_transform(self, synthetic_frame):
        c = SmileCorrector(deg_y=2, deg_x=0)
        corrected = c.fit_transform(synthetic_frame)
        assert corrected.shape[0] == synthetic_frame.shape[0]
        assert corrected.shape[1] <= synthetic_frame.shape[1]  # cropped

    def test_save_load_roundtrip(self, synthetic_frame, tmp_cal_path):
        c = SmileCorrector(deg_y=2, deg_x=0)
        c.fit(synthetic_frame)
        c.save(tmp_cal_path)

        c2 = SmileCorrector.load(tmp_cal_path)
        result1 = c.transform(synthetic_frame)
        result2 = c2.transform(synthetic_frame)
        np.testing.assert_allclose(result1, result2, atol=1e-3)

    def test_calibration_save_load_roundtrip(self, synthetic_frame, tmp_cal_path):
        c = SmileCorrector(deg_y=2, deg_x=0).fit(synthetic_frame)
        expected = c.transform(synthetic_frame, crop=False)
        c.save(tmp_cal_path)
        loaded = SmileCorrector.load(tmp_cal_path)
        actual = loaded.transform(synthetic_frame, crop=False)
        np.testing.assert_allclose(actual, expected, atol=1e-2)
        assert loaded.y_ref == c.y_ref

    def test_not_fitted_raises(self):
        c = SmileCorrector()
        with pytest.raises(RuntimeError):
            c.transform(np.zeros((10, 10), dtype=np.float32))

    def test_shape_mismatch_raises(self, synthetic_frame):
        c = SmileCorrector(deg_y=2, deg_x=0)
        c.fit(synthetic_frame)
        with pytest.raises(ValueError):
            c.transform(np.zeros((5, 5), dtype=np.float32))

    def test_get_set_params(self):
        c = SmileCorrector(deg_y=3)
        params = c.get_params()
        assert params["deg_y"] == 3
        c.set_params(deg_y=5)
        assert c.deg_y == 5

    def test_inverse_transform(self, synthetic_frame):
        c = SmileCorrector(deg_y=2, deg_x=0)
        corrected = c.fit_transform(synthetic_frame, crop=False)
        restored = c.inverse_transform(corrected)
        assert restored.shape == synthetic_frame.shape
