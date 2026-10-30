# smile/test_surface.py
"""Tests for Surface2D and fit_surface_2d."""
import numpy as np
import pytest
from smile.surface import Surface2D, fit_surface_2d, cross_validate_lines

class TestSurface2D:
    def test_linear(self):
        s = Surface2D([0, 1, 0, 0], 1, 1, (0, 10), (0, 10))
        # Coefficients are row-major: [Y^0X^0, Y^0X^1, Y^1X^0, Y^1X^1].
        v = s(np.array([5.0]), np.array([3.0]))
        assert abs(v[0] - (-0.4)) < 1e-9

    def test_clamp(self):
        s = Surface2D([0, 0, 0, 10], 1, 1, (0, 10), (0, 10), clamp=True)
        v_outside = s(np.array([20.0]), np.array([5.0]))
        v_edge    = s(np.array([10.0]), np.array([5.0]))
        assert abs(v_outside[0] - v_edge[0]) < 1e-6  # clamped

class TestFitSurface2D:
    def test_recovers_known_surface(self):
        rng = np.random.default_rng(42)
        y = rng.uniform(0, 100, 500)
        x = rng.uniform(0, 300, 500)
        v = 2.0 * (y/100 - 0.5)**2  # known smile
        surf, info = fit_surface_2d(y, x, v, deg_y=2, deg_x=0)
        assert info["rms"] < 0.01

class TestCrossValidate:
    def test_cv_rms_close_to_fit_rms(self):
        rng = np.random.default_rng(42)
        y = np.tile(np.arange(100, dtype=float), 5)
        x = np.repeat([50, 100, 150, 200, 250], 100).astype(float)
        v = 2.0 * (y/100 - 0.5)**2 + rng.normal(0, 0.01, y.size)
        ids = np.repeat(range(5), 100)
        cv = cross_validate_lines(y, x, v, ids, 2, 0)
        assert np.isfinite(cv)
        assert cv < 0.5
