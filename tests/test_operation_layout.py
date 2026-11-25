"""The operator view keeps data inspection independent of technical camera previews."""
from pathlib import Path
import sys
import time

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
    w.pipeline_view.select('sensor')                  # the spectrogram and focus aid are the Sensor workspace
    app.processEvents()
    assert w._raw_label.isVisible() and w.focus_panel.isVisible()
    assert not w._step_boxes['flatfield'].isVisible()  # folded under Advanced
    w._advanced_button.click()
    app.processEvents()
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
    # the loaded profile is the one the factors were fitted with (by its values): it can be taken out
    assert box.isChecked() and box.isEnabled() and box.text() == 'Flat-field in reference'
    assert not box.testAttribute(Qt.WA_TransparentForMouseEvents) and 'flat_A' in box.toolTip()
    assert not pipeline.effective_steps()['flatfield']                                 # never divided twice
    assert 'Includes flat-field: flat_A' in reference.toolTip() and reference.text() == 'Reflectance'
    assert w._input_text() == 'Reflectance + flat-field'

    box.click()                                                                        # hsi: Flat-field off
    assert not pipeline.flatfield_in_reflectance and pipeline.effective_steps()['reflectance']
    assert not box.isChecked() and 'out of reference' in box.text() and w._input_text() == 'Reflectance'
    box.click()                                                                        # put back
    assert pipeline.flatfield_in_reflectance and box.isChecked() and w._input_text() == 'Reflectance + flat-field'

    rows = np.linspace(0.6, 1.0, slit, dtype=np.float32)[:, None] * np.ones((1, bands), np.float32)
    other = FlatField(rows, 'profiles/cam/flatfield/flat_B/flatfield.json', flat.dark, flat.flip, flat.smile)
    install_flatfield(plan, other, False)                       # another profile loaded: other values
    w._setup_step_switches()
    assert reference.text() == 'Reflectance ⚠' and 'flat_B' in reference.toolTip()
    assert box.testAttribute(Qt.WA_TransparentForMouseEvents)  # the profile inside is not known now

    same = FlatField(flat.values, 'profiles/cam/flatfield/flat_C/flatfield.json', flat.dark, flat.flip, flat.smile)
    install_flatfield(plan, same, False)                        # another name, the same values: it fits
    w._setup_step_switches()
    assert reference.text() == 'Reflectance'

    reference.setChecked(False)                                                        # the step is its own switch again
    assert box.text() == 'Flat-field' and box.isEnabled() and not box.isChecked()
    assert not box.testAttribute(Qt.WA_TransparentForMouseEvents) and w._input_text() == 'Counts'
    box.setChecked(True)
    assert pipeline.effective_steps()['flatfield'] and w._input_text() == 'Counts + flat-field'
    assert not reference.isChecked()


def test_live_camera_starts_with_the_steps_used_last_time(window):
    """Flat-field switched on in the morning is still on after the program is opened again."""
    import numpy as np
    from flatfield_profile import FlatField, install_flatfield
    app, w, settings = window
    pipeline = w._pipeline
    plan = pipeline.plans[0]
    on = pipeline.effective_steps()
    flat = FlatField(np.ones((plan.frame_shape[0], plan.n_columns), np.float32),
                     'profiles/cam/flatfield/flat_A/flatfield.json', on['dark'], on['flip'], on['smile'])
    install_flatfield(plan, flat, False)
    w._setup_step_switches()
    w._live = True
    w._step_boxes['flatfield'].setChecked(True)                    # operator: Flat-field on
    assert pipeline.effective_steps()['flatfield']
    assert settings.value(w._steps_key())

    pipeline.set_steps(flatfield=False, reflectance=None)          # a new start: config defaults
    assert not pipeline.effective_steps()['flatfield']
    w._restore_steps()
    w._setup_step_switches()
    assert pipeline.effective_steps()['flatfield'] and w._step_boxes['flatfield'].isChecked()

    w._step_boxes['flatfield'].setChecked(False)                   # switched off: off next time too
    pipeline.set_steps(flatfield=True)
    w._restore_steps()
    assert not pipeline.effective_steps()['flatfield']


def test_pipeline_view_shows_every_step_from_sensor_to_model(window):
    from pipeline_overview import describe
    app, w, _ = window
    w._workspace.setCurrentIndex(1)
    app.processEvents()
    w._last_snapshot = w._pipeline.snapshot()
    w._refresh_pipeline_view()
    headline, blocks = describe(w._pipeline, w._last_snapshot)
    assert [b.key for b in blocks] == ['sensor', 'dark', 'flip', 'smile', 'binning', 'flatfield',
                                       'reflectance', 'spatial_bin', 'model']
    assert all(b.state in ('ok', 'warn', 'error', 'off') for b in blocks)
    view = w.pipeline_view
    assert view.isVisible() and set(view._blocks) == {b.key for b in blocks}
    assert view.input_label.text() == headline == w._input_text()
    view.select('sensor')
    assert dict(view.detail_rows())['Camera'].endswith(w._pipeline.sources[0].name)
    view._blocks['model'].clicked.emit('model')
    assert view.selected == 'model' and 'Expects' in dict(view.detail_rows())
    assert not w.grab().isNull()


def test_pipeline_view_says_the_loaded_flatfield_is_not_in_the_reflectance(window):
    """Per-band factors and a loaded flat-field profile: Flat-field is 'not applied' with the way
    to apply it; factors made with it: 'in reference'. The buttons act on the pipeline."""
    import numpy as np
    from flatfield_profile import FlatField, install_flatfield
    from pipeline_overview import describe
    app, w, _ = window
    pipeline = w._pipeline
    plan = pipeline.plans[0]
    slit, bands = plan.frame_shape[0], plan.n_columns
    on = pipeline.effective_steps()
    flat = FlatField(np.ones((slit, bands), np.float32), 'profiles/cam/flatfield/flat_A/flatfield.json',
                     on['dark'], on['flip'], on['smile'])
    install_flatfield(plan, flat, False)

    pipeline.set_reflectance_factors(np.ones(bands, np.float32), 'per band')
    blocks = {b.key: b for b in describe(pipeline)[1]}
    assert blocks['flatfield'].state == 'warn' and blocks['flatfield'].line1 == 'not applied'
    assert ('reflectance', 'Calibrate with flat-field…') in blocks['flatfield'].actions
    assert blocks['reflectance'].line1 == 'per band'

    pipeline.set_reflectance_factors(np.ones((slit, bands), np.float32), 'integrated', flatfield=flat.label)
    blocks = {b.key: b for b in describe(pipeline)[1]}
    assert blocks['flatfield'].state == 'ok' and blocks['flatfield'].line1 == 'in reference'
    assert blocks['reflectance'].line1 == 'flat-field + ref'

    w._on_pipeline_action('toggle:reflectance')                     # Switch off: flat-field alone possible
    assert not pipeline.effective_steps()['reflectance']
    w._on_pipeline_action('toggle:flatfield')
    assert pipeline.effective_steps()['flatfield'] and w._step_boxes['flatfield'].isChecked()
    assert {b.key: b for b in describe(pipeline)[1]}['flatfield'].line1 == 'applied'
    assert w.pipeline_view.input_label.text() == 'Counts + flat-field'


def test_setup_workspace_follows_the_selected_block(window):
    """Flat-field shows the brightness along the slit (profile + after correction), Model its
    summary; the calibration file list and the status bar are filled."""
    import numpy as np
    from flatfield_profile import FlatField, install_flatfield
    app, w, _ = window
    w._workspace.setCurrentIndex(1)
    plan = w._pipeline.plans[0]
    slit, bands = plan.frame_shape[0], plan.n_columns
    on = w._pipeline.effective_steps()
    values = np.ones((slit, bands), np.float32) * np.linspace(0.6, 1.0, slit, dtype=np.float32)[:, None]
    install_flatfield(plan, FlatField(values, 'profiles/cam/flatfield/flat_20261118_111459/flatfield.json',
                                      on['dark'], on['flip'], on['smile']), False)
    w._setup_step_switches()
    w._last_snapshot = w._pipeline.snapshot()
    w.pipeline_view.select('flatfield')
    app.processEvents()
    assert w._work_stack.currentWidget() is w._profile_page
    labels = [s[0] for s in w.profile_plot.series]
    assert labels == ['optics (flat-field profile)', 'with the correction in use']
    assert 'optics 30 % darker' in w._profile_note.text()        # inner rows of 0.6 .. 1.0 over 8 rows
    w._on_pipeline_action('toggle:flatfield')
    assert max(w.profile_plot.series[1][2]) - min(w.profile_plot.series[1][2]) < 1e-6   # flat after correction
    t = time.time()
    while time.time() - t < 5 and w._pipeline.snapshot().lines_processed < 20:
        time.sleep(0.02)
    w._last_snapshot = w._pipeline.snapshot(raw_columns=10)                 # the GUI passes its raw zone
    w._check_with_white()
    assert w.profile_plot.series[-1][0] == 'model input now'
    files = {row[0]: row for row in w._files_shown}
    assert files['Flat-field'][1] == 'flat_20261118_111459' and files['Flat-field'][3] == 'ok'
    assert files['Flat-field'][2]                                                      # an age from the stamp
    w.pipeline_view.select('model')
    assert w._work_stack.currentWidget() is w._model_page and 'expects' in w._model_summary.text()
    assert w.band_bar.input_range is not None and w.band_bar.bands          # model bands on the input range
    w._refresh_pipeline_view()
    assert any('fps' in text for text, _ in [(l.text(), 0) for l in
               (w.pipeline_view._status.itemAt(i).widget() for i in range(w.pipeline_view._status.count()))])
    assert not w.grab().isNull()


def test_temperature_history_is_plotted_with_the_calibration_band(window):
    from types import SimpleNamespace
    app, w, _ = window
    w._workspace.setCurrentIndex(1)
    w.pipeline_view.select('sensor')
    plan = w._pipeline.plans[0]
    plan.reflectance_temperature_c, plan.reflectance_max_dtemp_c = 30.0, 5.0
    w._live = True
    for i, value in enumerate((31.0, 31.4, 32.0)):
        w._temperature_history.append((time.time() - 25 + 10 * i, value))
    w._record_temperature(SimpleNamespace(sensor_temperature_c=32.5))      # < 10 s after the last: not kept
    assert len(w._temperature_history) == 3
    w._refresh_workspace()
    label, xs, ys, _, _ = w.temperature_plot.series[0]
    assert ys == [31.0, 31.4, 32.0] and xs[0] < xs[-1] <= 0
    assert w.temperature_plot.band[:2] == (25.0, 35.0) and '30.0 °C' in w.temperature_plot.band[3]
    assert w._temperature_value.text() == '32.0 °C' and w._temperature_value.property('state') == 'ok'
    assert '+2.0 °C from the reflectance calibration' in w._temperature_note.text()
    assert 's ago' in w._temperature_note.text()


def test_camera_process_counters_carry_the_temperature():
    from camera_process import NO_TEMPERATURE, TEMPERATURE, SharedFrames
    from camera_source import TemperaturePoll
    shared = SharedFrames(2, (4, 4), 'uint16')
    try:
        assert shared.counters[TEMPERATURE] == NO_TEMPERATURE
    finally:
        shared.release()

    class Camera:
        def __init__(self, values):
            self.values = list(values)

        def get_temperature(self):
            return self.values.pop(0)

    poll = TemperaturePoll(Camera([33.25, -999.0]), period_s=10.0)
    assert poll.due(0.0) and poll.read(0.0) == 33.25 and not poll.due(5.0)
    assert poll.due(10.0) and poll.read(10.0) is None                      # -999: not reported
    silent = TemperaturePoll(Camera([-999.0] * 3), period_s=1.0)
    for t in range(3):
        silent.read(float(t))
    assert not silent.due(1e9)                                             # gave up after three


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


def test_next_steps_fold_the_model_into_reflectance():
    from pipeline_overview import Block, next_steps
    blocks = [Block('sensor', 'Sensor', 'warn', note='Peak signal is 1 % of full scale.', actions=[('sensor', 'Sensor settings…')]),
              Block('flatfield', 'Flat-field', 'off'),
              Block('reflectance', 'Reflectance', 'error', note='The model expects reflectance but receives counts.',
                    actions=[('reflectance', 'Calibrate…'), ('toggle:reflectance', 'Switch on')]),
              Block('model', 'Model', 'error', note='The model was trained on reflectance, it receives counts.',
                    actions=[('model', 'Choose model…')])]
    items = next_steps(blocks)
    assert [i[0] for i in items] == ['error', 'warn']                  # blocking first, the model folded in
    assert items[0][2:] == ('reflectance', 'Calibrate…') and 'clears the Model block' in items[0][1]
    assert items[1][1].startswith('Sensor: Peak signal')


def test_stored_reflectance_that_does_not_fit_is_named():
    from types import SimpleNamespace
    from pipeline_overview import stored_reflectance
    plan = SimpleNamespace(provenance={'reflectance': ('not usable: 224 bands', 'auto')},
                           steps={'reflectance': {'available': False, 'on': False,
                                                  'why': 'refl_b2_g1.0.npy: factor profile has 224 bands, data 221'}})
    assert stored_reflectance(plan) == ('refl_b2_g1.0.npy', 'factor profile has 224 bands, data 221')
    plan.provenance['reflectance'] = ('none', 'auto')
    assert stored_reflectance(plan) is None


def test_flatfield_taken_out_keeps_the_white_target_at_its_reflectance():
    """hsi ScanController._without_flatfield: factors x F / F_ROI."""
    import numpy as np
    from flatfield_profile import factors_without_flatfield, flatfield_fits
    slit, bands = 10, 4
    flat = np.linspace(0.7, 1.2, slit)[:, None] * np.linspace(0.9, 1.1, bands)[None, :]
    flat = flat / np.median(flat)
    reflectance = np.full(bands, 0.95)
    counts = 2000.0 * flat                                       # a uniform white seen through the optics
    rows = slice(3, 6)                                           # the drawn white target
    scale = reflectance / (counts / flat)[rows].mean(axis=0)
    factors = (scale[None, :] / flat).astype(np.float32)         # "Flat-field + reference"
    assert flatfield_fits(factors, flat) and not flatfield_fits(factors, np.ones_like(flat) + np.arange(slit)[:, None])
    weights = np.zeros(slit)
    weights[rows] = 1.0
    out = factors_without_flatfield(factors, flat, weights)
    on_target = (counts * out)[rows].mean(axis=0)
    assert np.allclose(on_target, reflectance, rtol=1e-5)        # the target stays at R
    plain = (counts * factors * flat)[rows].mean(axis=0)         # factors x F alone: R x F_ROI
    assert not np.allclose(plain, reflectance, rtol=1e-3)
    assert np.allclose((counts * out) / (counts * out)[rows].mean(axis=0), flat / flat[rows].mean(axis=0))
