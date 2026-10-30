# smile/conftest.py
"""Shared test fixtures."""
import numpy as np
import pytest

@pytest.fixture
def synthetic_frame():
    """Synthetic Hg-Ar-like frame with known smile (parabolic shift)."""
    h, w = 200, 300
    frame = np.zeros((h, w), dtype=np.float32)
    # Three emission lines with known smile curvature
    for col in (50, 150, 250):
        for y in range(h):
            shift = 2.0 * ((y / h) - 0.5) ** 2  # known smile
            x_center = col + shift
            x_int = int(round(x_center))
            for dx in range(-3, 4):
                xi = x_int + dx
                if 0 <= xi < w:
                    frame[y, xi] += 200.0 * np.exp(-0.5 * (dx + x_center - x_int) ** 2 / 1.5)
    frame += np.random.default_rng(42).normal(0, 1, frame.shape).astype(np.float32)
    return frame

@pytest.fixture
def tmp_cal_path(tmp_path):
    return str(tmp_path / "test_cal.npz")