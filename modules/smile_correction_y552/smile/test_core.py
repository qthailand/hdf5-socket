# smile/test_core.py
"""Tests for core detection and tracing functions."""
import numpy as np
import pytest
from smile.core import (
    robust_sigma, auto_sat_level, estimate_noise,
    detect_lines, trace_line,
)

class TestRobustSigma:
    def test_normal_data(self):
        x = np.random.default_rng(42).normal(0, 2, 10000)
        assert 1.5 < robust_sigma(x) < 2.5

    def test_with_outliers(self):
        x = np.concatenate([np.random.default_rng(42).normal(0, 1, 9900),
                            np.full(100, 1000)])
        assert robust_sigma(x) < 5.0  # not fooled by outliers

class TestAutoSatLevel:
    @pytest.mark.parametrize("mx, expected", [
        (255, 255.0), (1023, 1023.0), (4095, 4095.0), (65535, 65535.0),
    ])
    def test_common_bit_depths(self, mx, expected):
        frame = np.array([[mx]], dtype=np.float32)
        assert auto_sat_level(frame) == expected

class TestDetectLines:
    def test_three_lines(self, synthetic_frame):
        lines = detect_lines(synthetic_frame, sat_level=255.0)
        assert 2 <= len(lines) <= 5  # 3 synthetic lines ± noise

class TestTraceLine:
    def test_traces_finite_rows(self, synthetic_frame):
        lines = detect_lines(synthetic_frame, sat_level=255.0)
        centers = trace_line(synthetic_frame, lines[0].column, 5, 255.0)
        assert np.sum(np.isfinite(centers)) > 100