"""Quick model: training in memory, the wand, the dialog, and the session tile of the selector."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference_engine import PushbroomLineClassifier  # noqa: E402
from pipeline import build_pipeline  # noqa: E402
from quick_model import MAX_PER_CLASS, LabelSet, train_artifact, wand_mask  # noqa: E402
from test_hsi_package import WAVELENGTHS  # noqa: E402
from test_model_switch import setup, wait_for  # noqa: E402

pytest.importorskip('sklearn')


def materials(rng, n=300, classes=3):
    """Class id (1..) -> spectra (n, bands): one spectral shape per class, brightness and noise vary."""
    axis = np.linspace(0.0, 1.0, WAVELENGTHS.size)
    shapes = [0.2 + 0.6 * axis, 0.8 - 0.6 * axis, 0.3 + 0.5 * np.sin(np.pi * axis), 0.5 + 0.0 * axis]
    return {k + 1: (shapes[k] * rng.uniform(0.7, 1.3, (n, 1)) + rng.normal(0, 0.01, (n, axis.size))).astype(np.float32)
            for k in range(classes)}


def scene(rng):
    """A view (30 lines, 16 slit pixels): class 1 everywhere, a block of class 2, a far block of class 2."""
    spectra = materials(rng, n=30 * 16, classes=2)
    cube = spectra[1].reshape(30, 16, -1).copy()
    other = spectra[2].reshape(30, 16, -1)
    cube[5:12, 3:9] = other[5:12, 3:9]
    cube[20:25, 10:14] = other[20:25, 10:14]
    return cube


@pytest.mark.parametrize('classes', [2, 3])
def test_trained_artifact_classifies_like_sklearn_would(classes):
    """Scaler folded into the weights; two classes get both scores; names, colours and units travel."""
    rng = np.random.default_rng(1)
    spectra = materials(rng, classes=classes)
    names = {1: 'PET', 2: 'PP', 3: 'belt'}
    artifact, report = train_artifact(spectra, names, WAVELENGTHS, 'reflectance', {1: [1, 2, 3]})
    assert report['accuracy'] > 0.95 and report['samples'] == {c: 300 for c in spectra}
    model = PushbroomLineClassifier.from_artifact(artifact, 'Quick 1')
    assert model.session and model.path == 'Quick 1' and model.input_units == 'reflectance'
    assert model.category_names == {c: names[c] for c in spectra} and model.class_colors == {1: [1, 2, 3]}
    columns, text = model.bind(WAVELENGTHS, WAVELENGTHS.size)
    assert 'by wavelength' in text
    fresh = materials(np.random.default_rng(2), classes=classes)
    for cid, x in fresh.items():
        assert (model.predict(x, columns) == cid).mean() > 0.95
        ids, confidence = model.predict_confidence(x, columns)
        assert ((confidence > 0) & (confidence <= 1)).all()
    with pytest.raises(ValueError, match='no input band'):               # another camera bin: refused as files are
        model.bind(WAVELENGTHS + 40.0, WAVELENGTHS.size)


def test_training_needs_two_classes_with_enough_finite_pixels():
    spectra = materials(np.random.default_rng(3), classes=2)
    with pytest.raises(ValueError, match='at least two classes'):
        train_artifact({1: spectra[1]}, {}, WAVELENGTHS, 'counts')
    with pytest.raises(ValueError, match='at least two classes'):
        train_artifact({1: spectra[1], 2: spectra[2][:5]}, {}, WAVELENGTHS, 'counts')
    holes = spectra[2].copy()
    holes[:, 0] = np.nan                                                  # no finite spectrum left
    with pytest.raises(ValueError, match='at least two classes'):
        train_artifact({1: spectra[1], 2: holes}, {}, WAVELENGTHS, 'counts')
    artifact, _ = train_artifact(spectra, {}, None, 'counts')            # input without wavelengths
    model = PushbroomLineClassifier.from_artifact(artifact, 'q')
    assert model.band_wavelengths is None and model.category_names == {1: '1', 2: '2'}
    assert 'by index' in model.bind(None, WAVELENGTHS.size)[1]


def test_label_set_keeps_a_bounded_number_of_spectra_per_class():
    labels = LabelSet()
    a, b = labels.add_class(), labels.add_class('belt')
    assert (a, b) == (1, 2) and labels.names == {1: 'class 1', 2: 'belt'} and labels.colors[1] != labels.colors[2]
    labels.add(a, np.ones((MAX_PER_CLASS, 4)))
    labels.add(a, np.ones((50, 4)))
    labels.add(9, np.ones((5, 4)))                                        # no such class: ignored
    assert labels.count(a) == MAX_PER_CLASS and labels.count(b) == 0 and labels.count(9) == 0
    labels.add(a, np.ones((7, 6)))                                        # another band count replaces
    assert labels.count(a) == 7 and list(labels.spectra(4)) == [] and list(labels.spectra(6)) == [a]
    labels.remove_class(a)
    assert labels.class_ids == [b] and labels.add_class() == 1            # the freed id is used again


def test_wand_selects_the_touching_area_of_the_same_material():
    cube = scene(np.random.default_rng(4))
    mask = wand_mask(cube, 8, 5, 0.06)
    expected = np.zeros((30, 16), bool)
    expected[5:12, 3:9] = True                                            # not the far block of the same material
    np.testing.assert_array_equal(mask, expected)
    assert wand_mask(cube, 0, 0, 0.06).sum() == 30 * 16 - 42 - 20         # the background around both blocks
    assert wand_mask(cube * 0, 3, 3, 0.06).sum() == 1                     # no spectrum: only the click


def test_snapshot_returns_the_newest_history_lines_in_order(tmp_path):
    config, package, linear, cube = setup(tmp_path, loop=False, fps=0)
    pipeline = build_pipeline(config)
    ids, lines = pipeline.spectrum_snapshot()
    assert ids.size == 0 and lines.shape == (0, 8, WAVELENGTHS.size)      # history off
    per_line = cube.shape[0] * WAVELENGTHS.size * 4
    assert pipeline.start_spectrum_history(max_mb=6 * per_line / 1e6) == 6
    pipeline.start()
    try:
        assert wait_for(lambda: pipeline.snapshot().state == 'finished')
    finally:
        pipeline.stop()
    ids, lines = pipeline.spectrum_snapshot()
    assert ids.tolist() == [9, 10, 11, 12, 13, 14]
    np.testing.assert_allclose(lines, cube[:, 9:15].transpose(1, 0, 2), rtol=1e-6)
    ids, lines = pipeline.spectrum_snapshot(max_mb=2 * per_line / 1e6)
    assert ids.tolist() == [13, 14] and lines.shape[0] == 2


def qt_app():
    pytest.importorskip('PyQt5')
    from PyQt5.QtWidgets import QApplication
    from themes import apply_theme
    app = QApplication.instance() or QApplication([])
    apply_theme(app, 'dark')
    return app


def test_dialog_marks_trains_and_keeps_the_labels_for_the_session():
    app = qt_app()
    from PyQt5.QtCore import QPoint, Qt
    from PyQt5.QtTest import QTest
    from widgets.quick_model_dialog import QuickModelDialog
    cube = scene(np.random.default_rng(5))
    views = [(np.arange(30), cube)]
    labels = LabelSet()
    dialog = QuickModelDialog(None, lambda: views[-1], WAVELENGTHS, 'reflectance', labels)
    ready = []
    dialog.model_ready.connect(ready.append)
    assert labels.class_ids == [1, 2] and dialog.current_class() == 1 and dialog.shape == (30, 16)
    assert not dialog.add_button.isEnabled() and not dialog.preview_box.isEnabled()
    assert not dialog.train() and 'at least two classes' in dialog.status.text()

    dialog.paint_rect(0, 3, 0, 15)                                        # class 1: background lines
    dialog.table.setCurrentCell(1, 1)
    dialog.wand_button.setChecked(True)
    dialog.wand(8, 5)                                                     # class 2: the block, found by the wand
    assert dialog.count(1) == 64 and dialog.count(2) == 42
    assert dialog.table.item(1, 2).text() == '42'
    dialog.tolerance.setValue(1)                                          # narrow: the last wand area shrinks...
    assert 1 <= dialog.count(2) < 42
    dialog.tolerance.setValue(6)                                          # ...and grows back
    assert dialog.count(2) == 42 and dialog.count(1) == 64
    dialog.paint_rect(0, 0, 0, 15, erase=True)
    assert dialog.count(1) == 48
    dialog.undo()
    assert dialog.count(1) == 64 and dialog.undo_button.isEnabled()

    dialog.table.item(1, 1).setText('PP')                                 # rename in the table
    assert labels.names[2] == 'PP'
    assert dialog.train() and dialog.report['accuracy'] > 0.9
    assert 'PP 42' in dialog.status.text() and dialog.preview_box.isChecked()
    truth = np.ones((30, 16), np.uint8)
    truth[5:12, 3:9] = truth[20:25, 10:14] = 2
    assert (dialog._preview == truth).mean() > 0.97                       # the far block too: never labelled
    dialog.resize(900, 500)
    assert not dialog.grab().isNull()                                     # paints view, preview and marks

    # the mouse: a rectangle drag marks, a right drag erases, a wand click marks
    dialog.show()
    app.processEvents()
    canvas = dialog.canvas
    w, h = canvas.width(), canvas.height()
    dialog.add_class('third')
    assert dialog.current_class() == 3
    dialog.rect_button.setChecked(True)
    QTest.mousePress(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(w * 27 // 30 + 2, 2))
    QTest.mouseRelease(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(w - 2, h - 2))
    assert (dialog.mask[27:30] == 3).all() and dialog.count(3) == 48
    QTest.mousePress(canvas, Qt.RightButton, Qt.NoModifier, QPoint(w * 29 // 30 + 2, 2))
    QTest.mouseRelease(canvas, Qt.RightButton, Qt.NoModifier, QPoint(w - 2, h - 2))
    assert dialog.count(3) == 32
    dialog.remove_class()                                                 # 'third' goes with its marks
    assert labels.class_ids == [1, 2] and not (dialog.mask == 3).any()

    dialog.add_to_models()
    assert len(ready) == 1 and ready[0] is dialog.artifact and not dialog.add_button.isEnabled()
    assert ready[0]['category_names'] == {1: 'class 1', 2: 'PP'} and ready[0]['input_units'] == 'reflectance'

    # another scene: the marks become labelled pixels, the trained model is shown on the new view
    views.append((np.arange(30, 60), scene(np.random.default_rng(6))))
    dialog.grab_button.click()
    assert labels.count(1) == 64 and labels.count(2) == 42 and not dialog.mask.any()
    assert dialog.count(2) == 42 and (dialog._preview == truth).mean() > 0.97
    dialog.paint_rect(20, 24, 10, 13)
    dialog.close_button.click()                                           # closing keeps them too
    assert labels.count(1) == 84 and not dialog.isVisible()

    again = QuickModelDialog(None, lambda: (np.zeros(0, np.int64), np.zeros((0, 16, WAVELENGTHS.size), np.float32)),
                             WAVELENGTHS, 'reflectance', labels)
    assert not again.has_view and 'Grab again' in again.status.text()     # history just started: no line yet
    assert again.count(2) == 42 and not again.grab().isNull()
    again.paint_rect(0, 1, 0, 1)                                          # nothing to mark: no crash
    assert again.train()                                                  # the session's labels are enough


def test_gui_adds_the_trained_model_as_a_session_tile_and_switches(tmp_path):
    app = qt_app()
    from PyQt5.QtCore import QTimer
    import gui
    config, package, linear, cube = setup(tmp_path)
    window = gui.PushbroomScannerMainWindow(config)
    try:
        window._gui_timer.stop()                       # refreshed by hand below
        window._spectrum_button.setChecked(False)      # hidden panel: the dialog keeps lines itself
        selector = window.model_selector
        assert wait_for(lambda: app.processEvents() or len(selector.tiles) == 2)
        assert window._pipeline.spectrum_lines == 0
        window._pipeline.start_spectrum_history()      # as a first Train… click that found no line yet
        assert wait_for(lambda: window._pipeline.spectrum_snapshot()[0].size >= 15)
        seen = {}

        def use_dialog():
            dialog = window._quick_dialog
            lines, slit = dialog.shape
            dialog.paint_rect(0, lines - 1, 0, slit // 2 - 1)             # upper half of the slit: class 1
            dialog.table.setCurrentCell(1, 1)
            dialog.paint_rect(0, lines - 1, slit // 2, slit - 1)          # lower half: class 2
            seen['trained'] = dialog.train()
            dialog.add_to_models()
            dialog.accept()
        QTimer.singleShot(0, use_dialog)
        window._train_button.click()                   # modal: returns when use_dialog closed it
        assert seen == {'trained': True} and window._quick_dialog is None
        assert window._pipeline.spectrum_lines == 0    # the Spectrum panel is hidden: history off again

        names = [t.name_label.text() for t in selector.tiles]
        assert names == ['linear', 'package', 'Quick 1']
        quick = selector.tiles[2]
        assert quick.property('session') is True and quick.kind_label.text() == 'linear · 2 classes · session'
        assert 'gone when the program closes' in quick.toolTip()
        assert [t.isChecked() for t in selector.tiles] == [False, False, True]     # switched to it
        assert window._pipeline.classifier.session
        assert Path(window.config.model.path).name == 'package.hsipkg'   # a restart loads a file, not this
        assert not list(tmp_path.rglob('Quick*'))                         # nothing written

        def shown():
            window._refresh_gui_slot()
            return window._shown_generation == 1
        assert wait_for(shown)
        assert window.class_legend_widget._names == ['class 1', 'class 2']
        assert window._lut[1].tolist() == window._quick_labels.colors[1]
        s = window._pipeline.snapshot()
        assert set(np.flatnonzero(s.counts).tolist()) <= {1, 2} and s.error is None

        selector.tiles[0].click()                      # back to a model file, and to the quick model again
        assert window._pipeline.classifier.category_names == {7: 'low', 9: 'high'}
        selector.tiles[2].click()
        assert window._pipeline.classifier.session

        selector.rescan_button.click()                 # a rescan keeps the session tile
        assert wait_for(lambda: app.processEvents() or len(selector.tiles) == 3)
        assert [t.name_label.text() for t in selector.tiles] == names
        assert [t.isChecked() for t in selector.tiles] == [False, False, True]

        selector.remove_requested.emit(selector.tiles[2].entry)
        assert [t.name_label.text() for t in selector.tiles] == ['linear', 'package']
        assert 'Quick 1' in selector._note.text()      # still running until another model is chosen
    finally:
        assert window._pipeline.stop()                 # not close(): that saves the user's window layout


def test_linear_predict_is_one_numba_call_with_the_classes_of_the_numpy_scores():
    """The linear predict runs in numba (no BLAS threads: OpenBLAS <= 0.3.26 threaded a quick
    model's 1024 x 159 @ 159 x 2 product on every core, per worker, on the Orin) and gives the
    classes of argmax(x @ W + b): index and resampled bindings, NaN / inf pixels, 2 and 5 classes."""
    from inference_engine import PushbroomLineClassifier, Resampled
    rng = np.random.default_rng(3)
    wl = np.linspace(900.0, 1700.0, 159)
    for n_classes in (2, 5):
        spectra = {c: (0.3 + 0.1 * c * np.sin(wl / (90 + 15 * c)) + rng.normal(0, 0.02, (300, wl.size)))
                   .astype(np.float32) for c in range(1, n_classes + 1)}
        art, _ = train_artifact(spectra, {c: f'c{c}' for c in spectra}, wl, 'reflectance')
        model = PushbroomLineClassifier.from_artifact(art, 'Quick')
        x = np.concatenate(list(spectra.values()))
        x[::37, 4] = np.nan
        x[::53, 90] = np.inf
        for axis in (wl, wl + 2.0):                         # by index, and resampled onto the model's bands
            columns, _ = model.bind(axis, axis.size)
            assert isinstance(columns, Resampled) == (axis is not wl)
            ids = model.predict(x, columns)
            assert ids.dtype == np.uint8 and ids.shape == (x.shape[0],)
            np.testing.assert_array_equal(ids, model._class_ids[model._scores(x, columns).argmax(axis=1)])
