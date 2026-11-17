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


def test_flatfield_box_shows_it_is_active_inside_the_reference(window):
    import numpy as np
    from PyQt5.QtCore import Qt
    from flatfield_profile import FlatField, install_flatfield
    app, w, _ = window
    w._workspace.setCurrentIndex(1)
    pipeline = w._pipeline
    plan = pipeline.plans[0]
    slit, bands = plan.frame_shape[0], plan.n_columns
    on = pipeline.effective_steps()
    flat = FlatField(np.ones((slit, bands), np.float32), 'profiles/cam/flatfield/flat_A/flatfield.json',
                     on['dark'], on['flip'], on['smile'])
    install_flatfield(plan, flat, False)
    box, reference = w._step_boxes['flatfield'], w._step_boxes['reflectance']

    pipeline.set_reflectance_factors(np.ones(bands, np.float32), 'per band')          # factors without flat-field
    w._setup_step_switches()
    assert not box.isChecked() and not box.isEnabled() and 'Not in this reflectance calibration' in box.toolTip()
    assert 'No flat-field in this calibration' in reference.toolTip() and w._input_text() == 'Reflectance'

    pipeline.set_reflectance_factors(np.ones((slit, bands), np.float32), 'integrated', flatfield=flat.label)
    w._setup_step_switches()
    assert box.isChecked() and box.isEnabled() and box.text() == 'Flat-field 🔒 in reference'
    assert box.testAttribute(Qt.WA_TransparentForMouseEvents) and 'flat_A' in box.toolTip()
    assert not pipeline.effective_steps()['flatfield']                                 # never divided twice
    assert 'Includes flat-field: flat_A' in reference.toolTip() and reference.text() == 'Reflectance'
    assert w._input_text() == 'Reflectance + flat-field'

    other = FlatField(flat.values, 'profiles/cam/flatfield/flat_B/flatfield.json', flat.dark, flat.flip, flat.smile)
    install_flatfield(plan, other, False)                                              # another profile loaded
    w._setup_step_switches()
    assert reference.text() == 'Reflectance ⚠' and 'flat_B' in reference.toolTip()

    reference.setChecked(False)                                                        # the step is its own switch again
    assert box.text() == 'Flat-field' and box.isEnabled() and not box.isChecked()
    assert not box.testAttribute(Qt.WA_TransparentForMouseEvents) and w._input_text() == 'Counts'
    box.setChecked(True)
    assert pipeline.effective_steps()['flatfield'] and w._input_text() == 'Counts + flat-field'
    assert not reference.isChecked()


def test_reflectance_record_tells_which_flatfield_the_factors_contain():
    from sim_config import reflectance_flatfield
    assert reflectance_flatfield({'kind': 'band', 'flatfield_profile': None}) is None
    assert reflectance_flatfield({'kind': 'row'}) is None and reflectance_flatfield({}) is None
    assert reflectance_flatfield({'kind': 'flatfield', 'flatfield_profile': 'p/flat_A/flatfield.json'}) \
        == 'p/flat_A/flatfield.json'
    assert reflectance_flatfield({'kind': 'flatfield'}) == 'unknown profile'


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
