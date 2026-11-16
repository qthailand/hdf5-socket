"""The operator view keeps data inspection independent of technical camera previews."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip('PyQt5')
from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QApplication
from themes import apply_theme
from test_model_switch import setup
import gui


@pytest.fixture
def window(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    apply_theme(app, 'dark')
    settings = QSettings(str(tmp_path / 'layout.ini'), QSettings.IniFormat)
    monkeypatch.setattr(gui.PushbroomScannerMainWindow, '_settings', lambda self: settings)
    config, *_ = setup(tmp_path, fps=100)
    w = gui.PushbroomScannerMainWindow(config)
    w.resize(1280, 760)
    w.show()
    app.processEvents()
    w._gui_timer.stop()
    yield app, w, settings
    w.close()
    app.processEvents()


def test_belt_spectrum_and_original_composition_are_default(window):
    app, w, _ = window
    assert w._workspace.currentWidget() is w._operation_page
    assert w._splitter.widget(0) is w._canvas_label
    assert w._splitter.widget(1) is w.spectrum_panel
    assert w._side_tabs.currentWidget() is w.composition_widget
    assert w._splitter.sizes()[0] > w._splitter.sizes()[1] >= 180
    assert not w._raw_label.isVisible() and not w.focus_panel.isVisible()
    assert w.spectrum_panel.training_button.isChecked()
    assert 'mean ± 1 SD' in w.spectrum_panel.reference_label.text()
    assert len(w.spectrum_panel.shown_training()[1]) == 3
    assert w._canvas_label.width() > 600
    assert not w.grab().isNull()


def test_spectrum_toggle_does_not_open_camera_preview(window):
    app, w, _ = window
    assert w._pipeline.spectrum_lines > 0
    w._spectrum_button.click()
    app.processEvents()
    assert w._pipeline.spectrum_lines == 0
    assert not w.spectrum_panel.isVisible()
    assert w._workspace.currentWidget() is w._operation_page
    w._spectrum_button.click()
    app.processEvents()
    assert w._pipeline.spectrum_lines > 0 and w.spectrum_panel.isVisible()
    assert not w._raw_label.isVisible()


def test_setup_and_diagnostics_keep_the_existing_tools(window):
    app, w, _ = window
    w._workspace.setCurrentIndex(1)
    app.processEvents()
    assert w._raw_label.isVisible() and w.focus_panel.isVisible()
    assert w._reflectance_button.isVisible()
    assert w._step_boxes['flatfield'].isVisible()
    assert not w.spectrum_panel.isVisible()
    assert w._pipeline.spectrum_lines > 0  # switching workspaces keeps probe history
    w._focus_button.click()
    app.processEvents()
    assert not w.focus_panel.isVisible() and w._raw_label.isVisible()
    w._workspace.setCurrentIndex(2)
    app.processEvents()
    assert w.performance_panel.isVisible() and w._status_label.isVisible()
    assert not w._focus_active()
    w._tick = 5  # the next refresh is the 5 Hz chart update at a 30 Hz GUI
    w._refresh_gui_slot()
    assert w.performance_panel.latency_chart._categories == ['Queue wait', 'p50', 'p99', 'max']
    w._workspace.setCurrentIndex(0)
    app.processEvents()
    assert w.spectrum_panel.isVisible() and w.composition_widget.isVisible()


def test_new_splitter_layout_restores_spectrum_visibility(window):
    app, w, settings = window
    w._spectrum_button.setChecked(False)
    w._save_layout()
    assert settings.value(w._SPECTRUM_KEY, type=bool) is False
    assert w._LAYOUT_KEY != 'main_splitter'
    w._spectrum_button.setChecked(True)
    w._restore_layout()
    app.processEvents()
    assert not w._spectrum_button.isChecked()
    assert w._splitter.sizes()[1] == 0
    assert w._pipeline.spectrum_lines == 0
