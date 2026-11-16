"""Spectrum preview: the pipeline's spectrum history, the panel, and clicks on the belt map."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import build_pipeline  # noqa: E402
from test_hsi_package import WAVELENGTHS  # noqa: E402
from test_model_switch import setup, wait_for  # noqa: E402


def test_history_returns_the_model_input_of_a_line(tmp_path):
    """File played once: line n, slit pixel r is cube[r, n]; nothing is kept when switched off."""
    config, package, linear, cube = setup(tmp_path, loop=False, fps=0)
    pipeline = build_pipeline(config)
    assert pipeline.spectrum_lines == 0 and pipeline.spectrum_at(0, 0) is None
    assert pipeline.start_spectrum_history() >= 15
    pipeline.start()
    try:
        assert wait_for(lambda: pipeline.snapshot().state == 'finished')
    finally:
        pipeline.stop()
    for line, row in ((0, 0), (7, 3), (14, 7)):
        np.testing.assert_allclose(pipeline.spectrum_at(line, row), cube[row, line], rtol=1e-6)
    assert pipeline.spectrum_at(15, 0) is None and pipeline.spectrum_at(0, 8) is None
    assert pipeline.spectrum_at(-1, 0) is None
    pipeline.stop_spectrum_history()
    assert pipeline.spectrum_lines == 0 and pipeline.spectrum_at(0, 0) is None


def test_history_keeps_only_the_newest_lines(tmp_path):
    config, package, linear, cube = setup(tmp_path, loop=False, fps=0)
    pipeline = build_pipeline(config)
    per_line = cube.shape[0] * WAVELENGTHS.size * 4
    assert pipeline.start_spectrum_history(max_mb=4 * per_line / 1e6) == 4
    pipeline.start()
    try:
        assert wait_for(lambda: pipeline.snapshot().state == 'finished')
    finally:
        pipeline.stop()
    assert pipeline.spectrum_at(10, 2) is None
    np.testing.assert_allclose(pipeline.spectrum_at(11, 2), cube[2, 11], rtol=1e-6)
    np.testing.assert_allclose(pipeline.spectrum_at(14, 2), cube[2, 14], rtol=1e-6)


def test_panel_gives_each_point_its_own_colour_and_clear_removes_them():
    pytest.importorskip('PyQt5')
    from PyQt5.QtWidgets import QApplication
    from themes import apply_theme
    from widgets.spectrum_panel import COLOURS, SpectrumPanel
    app = QApplication.instance() or QApplication([])
    apply_theme(app, 'dark')
    panel = SpectrumPanel()
    cleared = []
    panel.cleared.connect(lambda: cleared.append(1))
    assert not panel.clear_button.isEnabled()
    values = np.linspace(0.1, 0.9, WAVELENGTHS.size)
    values[3] = np.nan                                  # a gap, not a crash
    for i in range(panel.MAX_POINTS + 2):
        panel.add_point(WAVELENGTHS, values * (i + 1), line=i, row=i, label='Leaf')
    points = panel.points
    assert len(points) == panel.MAX_POINTS
    assert [p.number for p in points] == list(range(3, panel.MAX_POINTS + 3))    # the oldest two went
    assert sorted(p.colour for p in points) == sorted(COLOURS)                   # no colour twice
    panel.resize(600, 160)
    assert not panel.grab().isNull()                                             # paints with curves
    with pytest.raises(ValueError):
        panel.add_point(WAVELENGTHS, values[:-1], line=0, row=0)
    panel.clear_button.click()
    assert panel.points == [] and cleared == [1] and not panel.clear_button.isEnabled()
    assert panel.add_point(WAVELENGTHS, values, line=0, row=0).number == 1
    assert not panel.grab().isNull()


def test_panel_draws_training_bands_and_the_snv_view():
    pytest.importorskip('PyQt5')
    from PyQt5.QtWidgets import QApplication
    from themes import apply_theme
    from widgets.spectrum_panel import SpectrumPanel, TrainingBand, snv
    app = QApplication.instance() or QApplication([])
    apply_theme(app, 'dark')
    panel = SpectrumPanel()
    panel.resize(600, 160)
    assert not panel.training_button.isEnabled() and not panel.snv_button.isEnabled()
    mean = np.linspace(0.2, 0.6, WAVELENGTHS.size)
    raw = [TrainingBand('Leaf', '#22AA44', mean, np.full(mean.size, 0.05), 400),
           TrainingBand('Belt', '#000000', mean * 0.1, np.full(mean.size, 0.01), 300)]   # dark: still visible
    after = [TrainingBand(b.label, b.colour, snv(b.mean), np.full(mean.size, 0.2), b.n_spectra) for b in raw]
    panel.set_training(WAVELENGTHS, raw, after)
    assert panel.training_button.isChecked() and panel.shown_training()[1] == raw    # shown without a click
    assert not panel.grab().isNull()                                               # bands alone paint
    # a curve with 3 extra bands outside the training range: SNV is taken over the training bands only
    step = WAVELENGTHS[1] - WAVELENGTHS[0]
    wide = np.concatenate([WAVELENGTHS, WAVELENGTHS[-1] + step * np.arange(1, 4)])
    values = np.concatenate([mean * 2.0, [9.0, 9.0, 9.0]])
    point = panel.add_point(wide, values, line=0, row=0)
    panel.snv_button.click()
    np.testing.assert_allclose(panel.shown_values(point)[:mean.size], snv(mean), atol=1e-12)   # gain is gone
    assert panel.shown_training()[1] == after and not panel.grab().isNull()
    assert np.isnan(snv(np.ones(5))).all()                                         # constant: no SNV
    panel.training_button.click()
    assert panel.shown_training()[1] == []
    panel.set_training(None)                                                        # a model without them
    assert not panel.snv_shown and not panel.training_button.isEnabled()
    np.testing.assert_array_equal(panel.shown_values(point), values)


def test_gui_click_on_the_belt_map_adds_spectra_and_clear_removes_them(tmp_path):
    pytest.importorskip('PyQt5')
    from PyQt5.QtCore import QPoint, Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication
    from themes import apply_theme
    import gui
    app = QApplication.instance() or QApplication([])
    apply_theme(app, 'dark')
    config, package, linear, cube = setup(tmp_path)
    window = gui.PushbroomScannerMainWindow(config)
    try:
        window._gui_timer.stop()                       # refreshed by hand below
        window._spectrum_button.setChecked(True)       # whatever the saved layout says
        window.resize(1400, 800)
        window.show()
        app.processEvents()
        label, panel, pipeline = window._canvas_label, window.spectrum_panel, window._pipeline
        assert pipeline.spectrum_lines > 0

        def filled():
            window._refresh_gui_slot()
            # The redesigned belt pane is taller. True pixel proportions can legitimately
            # show fewer than 21 columns for this synthetic eight-row camera.
            return window._belt is not None and (window._belt['lines'] >= 0).sum() >= min(21, window._canvas_columns())
        assert wait_for(filled)

        w, h = label.width(), label.height()
        clicks = [QPoint(w - 5, h // 4), QPoint(w - 12, h // 2), QPoint(w - 20, 3 * h // 4)]
        for n, pos in enumerate(clicks, 1):
            QTest.mouseClick(label, Qt.LeftButton, Qt.NoModifier, pos)
            assert len(panel.points) == n, panel.note_label.text()
        points = panel.points
        assert len({p.colour for p in points}) == 3 and [p.number for p in points] == [1, 2, 3]
        slit, lines = cube.shape[0], cube.shape[1]
        for point, pos in zip(points, clicks):
            assert point.row == pos.y() * slit // h
            np.testing.assert_array_equal(point.wavelengths, WAVELENGTHS)
            # the cube loops: line n is file line n % 15
            np.testing.assert_allclose(point.values, cube[point.row, point.line % lines], rtol=1e-6)
            assert point.label in ('Background', 'Leaf', 'Wood')
        assert panel.unit.startswith('model input') and '#3' in panel.note_label.text()
        assert len(window._spectrum_marks()) == 3          # drawn on the map in the same colours

        # the package's training spectra lie behind the curves, in the class colours of the map
        wl, bands = panel.shown_training()
        assert panel.training_button.isChecked() and [b.label for b in bands] == ['Background', 'Leaf', 'Wood']
        np.testing.assert_array_equal(wl, WAVELENGTHS)
        assert bands[1].colour.lower() == '#%02x%02x%02x' % tuple(window._lut[1])
        panel.snv_button.click()                           # the model's view: SNV of curve and bands
        shown = panel.shown_values(points[0])
        assert abs(shown.mean()) < 1e-9 and abs(shown.std() - 1) < 1e-9
        assert abs(panel.shown_training()[1][1].mean.sum()) < 1e-6
        app.processEvents()                                # repaints with bands and SNV curves
        window._show_training_spectra(object())            # what a model switch to a joblib model does
        assert not panel.training_button.isEnabled()
        assert panel.shown_training()[1] == [] and not panel.snv_shown    # a joblib model brings none
        assert 'No training reference' in panel.reference_label.text()

        panel.clear_button.click()
        assert panel.points == [] and window._spectrum_marks() == []
        window._spectrum_button.setChecked(False)          # hidden: no history, a click adds nothing
        assert pipeline.spectrum_lines == 0 and not panel.isVisible()
        QTest.mouseClick(label, Qt.LeftButton, Qt.NoModifier, clicks[0])
        assert panel.points == []
        window._spectrum_button.setChecked(True)
        assert pipeline.spectrum_lines > 0
    finally:
        assert window._pipeline.stop()                 # not close(): that saves the user's window layout
