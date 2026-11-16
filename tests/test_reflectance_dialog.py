"""Zoomed image ROI must select the same frozen spectra used for calibration."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip('PyQt5')
from PyQt5.QtCore import QPoint, QPointF, Qt
from PyQt5.QtGui import QWheelEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QDialog
from widgets.reflectance_dialog import ReflectanceDialog


@pytest.fixture
def dialog():
    app = QApplication.instance() or QApplication([])
    ids = np.arange(100, 180)
    cube = (np.arange(80 * 60 * 3).reshape(80, 60, 3) + 100).astype(np.float32)
    d = ReflectanceDialog(None, [], [500, 510, 520], 'Test camera', lambda: len(ids), 256,
                          lambda: (ids, cube))
    d.show()
    app.processEvents()
    yield app, d, ids, cube
    d._timer.stop()
    d.close()
    app.processEvents()


def draw(d, first=(20.25, 15.25), last=(40.25, 35.25)):
    start, end = (d.view.mapFromScene(QPointF(*p)) for p in (first, last))
    QTest.mousePress(d.view.viewport(), Qt.LeftButton, pos=start)
    QTest.mouseRelease(d.view.viewport(), Qt.LeftButton, pos=end)


def test_grab_freezes_data_and_regrab_clears_roi(dialog):
    app, d, ids, cube = dialog
    assert not d.apply_button.isEnabled()
    d.grab_button.click()
    draw(d)
    selected, rows, kept_ids = d.selection()
    x0, x1, y0, y1 = d.view.roi
    np.testing.assert_array_equal(selected, cube[x0:x1])
    np.testing.assert_array_equal(kept_ids, ids[x0:x1])
    assert rows == (y0, y1) and d.apply_button.isEnabled()
    frozen = selected.copy()
    cube[:] = 7
    np.testing.assert_array_equal(d.selection()[0], frozen)
    d.grab_button.click()
    assert d.view.roi is None and not d.apply_button.isEnabled()
    np.testing.assert_array_equal(d.lines, 7)


def test_check_spectrum_plots_drawn_area_with_factors_in_use(dialog):
    from widgets.spectrum_check import SpectrumCheckDialog
    app, d, ids, cube = dialog
    d.grab_image()
    draw(d)
    app.processEvents()
    assert not d.check_button.isEnabled()                 # no factors in use yet
    x0, x1, y0, y1 = d.view.roi
    factors = (0.5 / cube[x0:x1, y0:y1].mean(axis=(0, 1))).astype(np.float32)
    d._current = lambda: (factors, 'test factors')
    d._update_status()
    assert d.check_button.isEnabled()
    d.percent.setValue(50)
    report, label = d.check_report()
    assert label == 'test factors' and report['ok']
    np.testing.assert_allclose(report['measured'], .5, rtol=1e-5)
    d.percent.setValue(99)                                # wrong standard selected: same offset everywhere
    report, _ = d.check_report()
    assert not report['ok'] and report['mean_error_pct'] == pytest.approx(100 * (.5 / .99 - 1), abs=1e-2)
    view = SpectrumCheckDialog(d, [500, 510, 520], report, 'constant 99 %', label)
    view.show()
    app.processEvents()
    assert view.grab().width() > 0 and len(view.spectrum_plot.curves) == 2 and len(view.error_plot.levels) == 3
    view.close()


def noisy_dialog(app, level, context):
    rng = np.random.default_rng(1)
    ids = np.arange(80)
    cube = (level * np.linspace(.2, 1, 12) + 3 * rng.standard_normal((80, 60, 12))).astype(np.float32)
    d = ReflectanceDialog(None, [], np.linspace(400, 850, 12), 'Test camera', lambda: len(ids), 256,
                          lambda: (ids, cube), context=lambda: context)
    d.show()
    d.grab_image()
    draw(d)
    app.processEvents()
    return d


def test_signal_check_blocks_a_starved_white_and_passes_a_good_one():
    app = QApplication.instance() or QApplication([])
    context = {'full_scale': 65535.0, 'dark_level': 320.0}
    starved = noisy_dialog(app, 80, context)                # like 1 ms on a 20 % standard: 16-80 counts
    assert starved.signal['status'] == 'red' and not starved.apply_button.isEnabled()
    assert starved.signal['peak_pct'] < 1 and 'below the dark offset' in starved.signal_text.text()
    assert starved.verdict.text() == 'Do not calibrate' and starved.tiles['peak'].property('accent') == 'error'
    assert len(starved.signal_plot.curves) == 1
    good = noisy_dialog(app, 40000, context)
    assert good.signal['status'] == 'green' and good.apply_button.isEnabled()
    assert good.verdict.text() == 'Ready to calibrate' and good.tiles['dark'].value.text() == '320'
    assert 55 < good.signal['peak_pct'] < 65 and good.signal['median_snr'] > 1000
    good.grab_image()                                       # a new grab: the old verdict is gone
    assert good.signal is None and 'Grab an image' in good.signal_text.text()
    assert good.tiles['peak'].value.text() == '–'
    saturated = noisy_dialog(app, 65300, context)
    assert saturated.signal['status'] == 'red' and 'Saturated' in saturated.signal_text.text()
    for d in (starved, good, saturated):
        d._timer.stop()
        d.close()


def test_corrections_table_lists_every_step_with_its_file(dialog):
    app, d, ids, cube = dialog

    def rows():
        return {name: (state, accent, source) for name, state, accent, source in d.corrections_rows}

    assert rows() == {'Flat-field': ('NONE', 'neutral', 'no profile loaded')}
    steps = [('Dark + hot pixel', True, 'dark_b2_e1.0_g1.0.npz (auto from profile X)'), ('Smile', False, 'none')]
    d._context = lambda: {'steps': steps, 'flatfield_label': 'profiles/X/flatfield/flat_1', 'flatfield_matches': True}
    d._update_corrections()
    assert rows()['Dark + hot pixel'] == ('ON', 'ok', 'dark_b2_e1.0_g1.0.npz (auto from profile X)')
    assert rows()['Smile'][:2] == ('OFF', 'neutral')
    assert rows()['Flat-field'][:2] == ('NOT USED', 'warn') and 'profiles/X/flatfield/flat_1' in rows()['Flat-field'][2]
    app.processEvents()
    assert d._corrections_grid.count() == 3 * 3                          # pill, name, file per row
    for _ in range(20):                                                  # repeated updates reuse the row widgets
        d._update_corrections()
    assert d._corrections_grid.count() == 3 * 3
    d.kind.addItem('Flat-field + reference', 'flatfield')
    d.kind.setCurrentIndex(d.kind.findData('flatfield'))
    assert rows()['Flat-field'][:2] == ('IN FACTORS', 'ok')
    d._context = lambda: {'steps': steps, 'flatfield_label': 'flat_1', 'flatfield_matches': False}
    d._update_corrections()
    assert rows()['Flat-field'][:2] == ('MISMATCH', 'error')
    d._context = lambda: {'steps': [('Dark + hot pixel', False, 'none')]}
    d._update_corrections()
    assert rows()['Dark + hot pixel'][:2] == ('OFF', 'error')              # dark off: an error, not neutral


def test_zoom_pan_reverse_draw_and_fit_preserve_image_coordinates(dialog):
    app, d, ids, cube = dialog
    d.grab_image()
    baseline = d.view.transform().m11()
    center = QPointF(d.view.viewport().rect().center())
    event = QWheelEvent(center, center, QPoint(), QPoint(0, 120), Qt.NoButton, Qt.NoModifier,
                        Qt.NoScrollPhase, False)
    d.view.wheelEvent(event)
    assert d.view.transform().m11() > baseline
    # Move scrollbars as panning would, then select the same image rectangle backwards.
    d.view.horizontalScrollBar().setValue(d.view.horizontalScrollBar().maximum())
    d.view.verticalScrollBar().setValue(d.view.verticalScrollBar().maximum())
    draw(d, first=(40.25, 35.25), last=(20.25, 15.25))
    assert d.view.roi == (20, 41, 15, 36)
    np.testing.assert_array_equal(d.selection()[0], cube[20:41])
    assert d.selection()[1] == (15, 36)
    d.fit_button.click()
    assert d.view.roi == (20, 41, 15, 36)
    assert d.view.transform().m11() == pytest.approx(baseline)


def test_review_uses_roi_and_retains_dialog_on_save_failure(dialog, monkeypatch):
    import gui
    app, d, ids, cube = dialog
    d.grab_image()
    draw(d)
    d.percent.setValue(20)
    applied = []

    class Confirm:
        Question, AcceptRole, ActionRole, Cancel = range(4)
        def __init__(self, *args, **kwargs):
            self.buttons = []
        def addButton(self, *args):
            button = object()
            self.buttons.append(button)
            return button
        def exec_(self):
            pass
        def clickedButton(self):
            return self.buttons[0]

    monkeypatch.setattr(gui, 'QMessageBox', Confirm)
    host = NS(_pipeline=NS(plans=[NS(stage='raw', wavelengths=np.array([500, 510, 520]))],
                           sources=[NS(metadata={})], effective_steps=lambda: dict(dark=True, flatfield=True)),
              _camera_text=lambda: 'test', _camera_values=lambda: {}, _next_start_note=lambda: '',
              _apply_calibration=lambda f, info: applied.append((f, info)) or False)
    gui.PushbroomScannerMainWindow._review_reflectance(host, d)
    x0, x1, y0, y1 = d.view.roi
    np.testing.assert_allclose(applied[0][0], 0.2 / cube[x0:x1, y0:y1].mean(axis=(0, 1)), rtol=1e-6)
    assert applied[0][1]['steps']['flatfield'] is False
    assert applied[0][1]['lines'] == [int(ids[x0]), int(ids[x1 - 1])]
    assert d.result() != QDialog.Accepted
    host._apply_calibration = lambda *args: True
    gui.PushbroomScannerMainWindow._review_reflectance(host, d)
    assert d.result() == QDialog.Accepted


@pytest.mark.parametrize('failed', [False, True])
def test_capture_is_released_on_cancel_or_dialog_failure(tmp_path, monkeypatch, failed):
    import gui
    from sim_config import SimulationConfig
    state = {'capturing': False, 'deleted': False}

    class Dialog:
        def __init__(self, *args):
            self._timer = NS(stop=lambda: None)
            self.calibration_requested = NS(connect=lambda callback: None)
            self.lines = np.ones((2, 3, 4))
            state['dialog'] = self
        def exec_(self):
            assert state['capturing']
            if failed:
                raise RuntimeError('dialog failure')
            return QDialog.Rejected
        def deleteLater(self):
            state['deleted'] = True

    pipeline = NS(plans=[NS(wavelengths=np.array([500, 510]))],
                  start_calibration_capture=lambda _: state.update(capturing=True),
                  stop_calibration_capture=lambda: state.update(capturing=False),
                  calibration_count=lambda: 0, calibration_capture=lambda: None)
    host = NS(_reflectance_dialog=None, _pipeline=pipeline, config=SimulationConfig(base_dir=str(tmp_path)),
              _camera_text=lambda: 'test', _review_reflectance=lambda _: pytest.fail('cancel must not save'))
    monkeypatch.setattr(gui, 'ReflectanceDialog', Dialog)
    if failed:
        with pytest.raises(RuntimeError, match='dialog failure'):
            gui.PushbroomScannerMainWindow._open_reflectance_dialog(host)
    else:
        gui.PushbroomScannerMainWindow._open_reflectance_dialog(host)
    assert not state['capturing'] and state['deleted']
    assert state['dialog'].lines is None and host._reflectance_dialog is None
