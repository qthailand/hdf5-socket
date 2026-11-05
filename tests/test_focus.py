"""Focus aid: edge contrast, hsi's sharpness score and the themed panel beside the preview."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from focus import edge_contrast  # noqa: E402
from modules.sharpness import sharpness_score  # noqa: E402


def _bar_frame(blur_rows: float, noise: float = 0.0, slit: int = 128, spectral: int = 256, seed: int = 0):
    """Raw-like frame: dark/bright bars along the slit (a focus target), defocused by ``blur_rows``
    (Gaussian sigma, slit pixels), then sensor noise - defocus happens before the noise."""
    import cv2
    profile = np.where((np.arange(slit) // 16) % 2 == 0, 3000.0, 9000.0).astype(np.float32)
    frame = np.repeat(profile[:, None], spectral, axis=1)
    if blur_rows > 0:
        frame = cv2.GaussianBlur(frame, (1, 0), sigmaX=0.1, sigmaY=blur_rows)
    if noise:
        frame = frame + np.random.default_rng(seed).normal(0, noise, frame.shape).astype(np.float32)
    return np.clip(frame, 0, 65535).astype(np.uint16)


def test_sharpness_code_is_identical_to_hsi():
    """The 'hsi score' line must use hsi's formula: the same code byte for byte."""
    sys.path.insert(0, str(ROOT / 'tools'))
    import sync_hsi
    assert 'sharpness/core.py' in sync_hsi.VENDORED
    if not (sync_hsi.DEFAULT_HSI / 'modules' / 'sharpness').is_dir():
        pytest.skip('hsi project not next to this one')
    assert sync_hsi.compare(sync_hsi.DEFAULT_HSI, only='sharpness') == []


def test_edge_contrast_falls_with_defocus_where_the_hsi_score_does_not():
    """Why the bar uses edge contrast: mean |gradient| across an edge is the edge height however
    wide it is blurred, so hsi's score stays flat near focus; the RMS step does not."""
    blurs = (0, 0.5, 1, 2, 4)
    contrast = [edge_contrast(_bar_frame(b), axis=0) for b in blurs]
    assert all(a > b for a, b in zip(contrast, contrast[1:])), contrast
    assert contrast[2] < 0.7 * contrast[0]                                   # 1 px blur: clearly lower
    hsi = [sharpness_score(_bar_frame(b), axis=0) for b in blurs[:4]]
    assert max(hsi) - min(hsi) < 0.01 * max(hsi), hsi                        # flat up to 2 px blur
    # with sensor noise added after the optics the order still holds (profile = mean of 256 columns)
    noisy = [edge_contrast(_bar_frame(b, noise=500, seed=k), axis=0) for k, b in enumerate(blurs)]
    assert all(a > b for a, b in zip(noisy, noisy[1:])), noisy


def test_edge_contrast_units_axes_and_exposure():
    frame = _bar_frame(0)
    # profile 3000/9000 bars of 16 rows: 7 steps of 6000 counts in 127, mean 6000 -> 100 % per step
    assert edge_contrast(frame, 0) == pytest.approx(100.0 * np.sqrt(7 / 127), rel=1e-6)
    assert edge_contrast(frame, 1) == 0.0                                   # nothing along wavelength
    assert edge_contrast(frame.T, 1) == pytest.approx(edge_contrast(frame, 0))
    brighter = (frame.astype(np.float32) * 2.5).astype(np.uint16)          # exposure / gain x 2.5
    assert edge_contrast(brighter, 0) == pytest.approx(edge_contrast(frame, 0), rel=1e-4)
    assert np.isnan(edge_contrast(np.zeros((8, 8), np.uint16), 0))           # dark frame: no score
    assert np.isnan(edge_contrast(np.ones((1, 8)), 0))
    with pytest.raises(ValueError):
        edge_contrast(frame, 2)
    with pytest.raises(ValueError):
        edge_contrast(frame[0], 0)


@pytest.fixture
def qt_app():
    pytest.importorskip('PyQt5')
    from PyQt5.QtWidgets import QApplication
    from themes import apply_theme
    app = QApplication.instance() or QApplication([])
    apply_theme(app, 'dark')
    return app


def test_panel_bar_is_score_over_a_slowly_decaying_peak(qt_app):
    from widgets.focus_panel import FocusPanel
    panel = FocusPanel()
    assert panel.ratio == 0.0 and panel.history == [] and panel.reference is None    # idle
    panel.set_value(2.0, reference=1400.0)
    assert panel.peak == 2.0 and panel.ratio == 1.0 and panel.reference == 1400.0
    assert panel.property('accent') == 'ok'
    panel.set_value(1.0)
    assert panel.peak == pytest.approx(2.0 * FocusPanel.PEAK_DECAY)      # as hsi: decay per value
    assert panel.ratio == pytest.approx(1.0 / panel.peak)
    assert panel.property('accent') == 'neutral' and panel.reference is None
    panel.set_value(1.9)
    assert panel.property('accent') == 'ok' and panel.history == [2.0, 1.0, 1.9]
    panel.set_value(float('nan'))                                       # empty frame: ignored
    assert len(panel.history) == 3
    for _ in range(FocusPanel.HISTORY + 10):
        panel.set_value(1.5)
    assert len(panel.history) == FocusPanel.HISTORY                      # 30 s window
    panel.reset()
    assert panel.history == [] and panel.peak == FocusPanel.PEAK_FLOOR and panel.ratio == 0.0


def test_panel_axis_switch_restarts_the_scale(qt_app):
    from widgets.focus_panel import FocusPanel
    panel = FocusPanel()
    seen = []
    panel.axis_changed.connect(seen.append)
    panel.set_value(5.0)
    panel._axis_group.button(1).click()                                 # the λ button
    assert panel.axis == 1 and seen == [1] and panel.history == []
    panel.set_axis(1)                                                   # no change: nothing happens
    assert seen == [1]
    panel.set_axis(0)
    assert panel.axis == 0 and panel._axis_group.button(0).isChecked() and seen == [1, 0]
    with pytest.raises(ValueError):
        panel.set_axis(2)


@pytest.mark.parametrize('theme', ['dark', 'light'])
def test_panel_paints_idle_and_live_at_any_height(qt_app, theme):
    from themes import apply_theme
    from widgets.focus_panel import FocusPanel, format_reference, format_score
    apply_theme(qt_app, theme)
    try:
        panel = FocusPanel()
        for height in (40, 90, 220):
            panel.resize(FocusPanel.WIDTH, height)
            assert not panel.grab().isNull()                            # idle
        for value in np.linspace(0.8, 1.4, 40):
            panel.set_value(value, reference=1000 * value)
        for height in (40, 90, 220):
            panel.resize(FocusPanel.WIDTH, height)
            assert not panel.grab().isNull()
    finally:
        apply_theme(qt_app, 'dark')
    assert format_score(1.4366) == '1.44' and format_score(12.34) == '12.3'
    assert format_reference(1406.58) == '1,407' and format_reference(274.4) == '274'


def test_gui_scores_each_new_frame_once_and_only_while_shown(qt_app):
    """``_update_focus``: edge contrast + hsi's score of the preview frame, along the panel's axis."""
    from gui import PushbroomScannerMainWindow as W
    from widgets.focus_panel import FocusPanel
    panel = FocusPanel()
    shown = {'on': True}
    win = SimpleNamespace(focus_panel=panel, _focus_frame=None, _focus_active=lambda: shown['on'])
    frame = _bar_frame(1, noise=50)
    W._update_focus(win, frame)
    assert panel.history == [pytest.approx(edge_contrast(frame, 0))]
    assert panel.reference == pytest.approx(sharpness_score(frame, axis=0))
    W._update_focus(win, frame)                                         # same preview frame: not again
    W._update_focus(win, None)
    assert len(panel.history) == 1
    panel.set_axis(1)
    W._update_focus(win, frame.copy())
    assert panel.history == [pytest.approx(edge_contrast(frame, 1))]
    assert panel.reference == pytest.approx(sharpness_score(frame, axis=1))
    shown['on'] = False                                                 # hidden / preview collapsed
    W._update_focus(win, frame.copy())
    assert len(panel.history) == 1
