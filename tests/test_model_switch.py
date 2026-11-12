"""Switching the model while the pipeline runs, the list of usable models, and the GUI selector."""
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference_engine import load_classifier  # noqa: E402
from model_catalog import model_files, model_folders, scan_models  # noqa: E402
from pipeline import build_pipeline  # noqa: E402
from sim_config import CameraConfig, CorrectionConfig, ModelConfig, ProcessingConfig, SimulationConfig, \
    SourceConfig  # noqa: E402
from test_hsi_package import WAVELENGTHS, build_package, training_spectra  # noqa: E402
from test_simulation import write_envi  # noqa: E402


def linear_model(folder, name='linear', wavelengths=WAVELENGTHS):
    """A joblib artifact with classes 7 and 9 (ids no package of the tests uses)."""
    import joblib
    path = Path(folder) / f'{name}.joblib'
    joblib.dump({'W_fused': np.array([[1.0, -1.0], [-1.0, 1.0]], np.float32), 'b_fused': np.zeros(2, np.float32),
                 'selected_bands': np.array([2, 8]), 'classes': np.array([7, 9]),
                 'category_names': {7: 'low', 9: 'high'}, 'input_units': 'reflectance',
                 'band_wavelengths': np.asarray(wavelengths, np.float32)}, path)
    return path


def setup(tmp_path, loop=True, fps=2000.0):
    """A looping reflectance cube (8 slit pixels, 15 lines, 12 bands), a models folder, a config."""
    models = tmp_path / 'models'
    models.mkdir()
    package = build_package(models, name='package')
    linear = linear_model(models)
    x, _ = training_spectra(np.random.default_rng(13), n=40)
    cube = x.astype(np.float32).reshape(8, 15, WAVELENGTHS.size)
    write_envi(tmp_path, cube, 'bip', wavelengths=WAVELENGTHS)
    config = replace(SimulationConfig(base_dir=str(tmp_path)),
                     source=SourceConfig(path=str(tmp_path / 'cube.bip.hdr'), cache='ram'),
                     camera=CameraConfig(fps=fps, overrun='block', buffer_frames=8, loop=loop),
                     corrections=CorrectionConfig(backend='cpu', reflectance='none'),
                     processing=ProcessingConfig(workers=2, profile_every_s=0),
                     model=ModelConfig(path=str(package), folders=('models',))).validate()
    return config, package, linear, cube


def wait_for(condition, timeout=20.0):
    t = time.perf_counter()
    while not condition() and time.perf_counter() - t < timeout:
        time.sleep(0.01)
    return condition()


# ----------------------------------------------------------------------------- pipeline ---

def test_model_is_switched_at_one_line_while_running(tmp_path):
    """No line is lost or classified by a mix of models; counts and class history start again."""
    config, package, linear, cube = setup(tmp_path)
    pipeline = build_pipeline(config)
    first = pipeline.classifier
    second = load_classifier(linear)
    pipeline.start()
    try:
        assert wait_for(lambda: pipeline.snapshot().lines_processed > 100)
        before = pipeline.snapshot()
        assert before.model_generation == 0 and set(np.flatnonzero(before.counts).tolist()) <= {0, 1, 2}
        assert pipeline.set_classifier(second) == 1
        assert pipeline.classifier is second
        assert 'linear.joblib' in pipeline.plans[0].provenance['model bands'][1]
        assert wait_for(lambda: pipeline.snapshot().model_generation == 1)
        assert wait_for(lambda: pipeline.snapshot().counted_lines > 100)
        after = pipeline.snapshot()
        # only the new model's classes are counted, every pixel of every counted line
        assert set(np.flatnonzero(after.counts).tolist()) <= {7, 9}
        assert after.counts.sum() == after.counted_lines * cube.shape[0]
        assert after.counted_lines < after.lines_processed          # the batch restarted at the switch
        assert set(np.unique(after.canvas).tolist()) <= {0, 7, 9}   # 0: history cleared, not the old classes
        assert after.lines_lost == 0 and after.state == 'running'
        # back again: generation 2, the package's classes
        assert pipeline.set_classifier(first) == 2
        assert wait_for(lambda: pipeline.snapshot().model_generation == 2)
        assert wait_for(lambda: pipeline.snapshot().counted_lines > 50)
        again = pipeline.snapshot()
        assert set(np.flatnonzero(again.counts).tolist()) <= {0, 1, 2} and again.lines_lost == 0
    finally:
        assert pipeline.stop()
    assert pipeline.snapshot().error is None


def test_model_that_does_not_fit_changes_nothing(tmp_path):
    config, package, linear, cube = setup(tmp_path)
    other = load_classifier(linear_model(tmp_path, 'other', WAVELENGTHS + 40.0))
    pipeline = build_pipeline(config)
    first, columns = pipeline.classifier, pipeline.plans[0].columns
    with pytest.raises(ValueError, match='no input band'):
        pipeline.set_classifier(other)
    assert pipeline.classifier is first and pipeline.plans[0].columns is columns
    assert 'package.hsipkg' in pipeline.plans[0].provenance['model bands'][1]
    pipeline.start()
    try:
        assert wait_for(lambda: pipeline.snapshot().lines_processed > 20)
        assert pipeline.snapshot().model_generation == 0
    finally:
        assert pipeline.stop()


def test_lines_follow_the_model_they_were_stamped_with(tmp_path):
    """Every counted line carries the classes of one model, in line order (file played once)."""
    config, package, linear, cube = setup(tmp_path, loop=False, fps=0)
    pipeline = build_pipeline(config)
    second = load_classifier(linear)
    pipeline.set_classifier(second)                    # before the first line: all 15 lines use it
    pipeline.start()
    try:
        assert wait_for(lambda: pipeline.snapshot().state == 'finished')
    finally:
        pipeline.stop()
    s = pipeline.snapshot()
    columns, _ = second.bind(WAVELENGTHS, WAVELENGTHS.size)
    expected = np.bincount(second.predict(cube.reshape(-1, WAVELENGTHS.size), columns), minlength=256)
    assert s.model_generation == 1 and s.lines_processed == 15
    np.testing.assert_array_equal(s.counts, expected)


# ------------------------------------------------------------------------------ catalog ---

def test_catalog_lists_usable_models_and_says_why_others_are_not(tmp_path):
    import joblib
    config, package, linear, cube = setup(tmp_path)
    models = tmp_path / 'models'
    build_package(models, name='wrapped', wrap=True)
    linear_model(models, 'elsewhere', WAVELENGTHS + 40.0)              # other bands: listed, not usable

    def with_pca(manifest, files):
        manifest['pipeline_audit']['pca'] = {'n_components': 3}
    build_package(models, name='refused', change=with_pca)             # a package this program cannot run
    joblib.dump({'scaler': 1}, models / 'not_a_model.joblib')          # some other joblib file: not listed
    (models / 'notes.zip').write_bytes(b'PK\x05\x06' + b'\0' * 18)     # an empty zip: not listed
    (models / 'readme.txt').write_text('x')

    assert model_folders(config) == [models]
    names = [p.name for p in model_files([models])]
    assert names == ['elsewhere.joblib', 'linear.joblib', 'not_a_model.joblib', 'package.hsipkg',
                     'refused.hsipkg', 'wrapped.zip']
    pipeline = build_pipeline(config)
    entries = {e.path.name: e for e in scan_models(config, pipeline.plans, pipeline.classifier)}
    assert sorted(entries) == ['elsewhere.joblib', 'linear.joblib', 'package.hsipkg', 'refused.hsipkg',
                               'wrapped.zip']
    assert [n for n, e in entries.items() if e.usable] == ['linear.joblib', 'package.hsipkg', 'wrapped.zip']
    assert entries['package.hsipkg'].classifier is pipeline.classifier  # the running model is not loaded twice
    assert 'MLC · 3 classes' in entries['wrapped.zip'].label and 'SNV' in entries['wrapped.zip'].detail
    assert 'linear · 2 classes' in entries['linear.joblib'].label
    assert 'does not fit' in entries['elsewhere.joblib'].label and 'no input band' in entries['elsewhere.joblib'].detail
    assert 'not usable' in entries['refused.hsipkg'].label and 'pca' in entries['refused.hsipkg'].detail
    assert entries['elsewhere.joblib'].classifier is None and entries['refused.hsipkg'].classifier is None
    # only model.folders is searched: a configured model from elsewhere is not listed
    lone = replace(config, model=replace(config.model, folders=('nowhere',)))
    assert model_folders(lone) == [] and scan_models(lone, pipeline.plans, pipeline.classifier) == []


# ---------------------------------------------------------------------------------- GUI ---

def test_gui_switches_the_model_from_the_selector(tmp_path):
    pytest.importorskip('PyQt5')
    from PyQt5.QtWidgets import QApplication
    from themes import apply_theme
    import gui
    app = QApplication.instance() or QApplication([])
    apply_theme(app, 'dark')
    config, package, linear, cube = setup(tmp_path)
    linear_model(tmp_path / 'models', 'elsewhere', WAVELENGTHS + 40.0)
    window = gui.PushbroomScannerMainWindow(config)
    try:
        window._gui_timer.stop()                       # refreshed by hand below
        selector = window.model_selector

        def scanned():
            app.processEvents()
            return len(selector.tiles) == 3
        assert wait_for(scanned)
        tiles = selector.tiles                                         # one tile per model of the folder
        assert [t.name_label.text() for t in tiles] == ['elsewhere', 'linear', 'package']
        assert tiles[0].kind_label.text() == 'does not fit this input' and 'no input band' in tiles[0].toolTip()
        assert tiles[1].kind_label.text() == 'linear · 2 classes' and 'MLC' in tiles[2].kind_label.text()
        assert not tiles[0].isEnabled() and tiles[1].isEnabled() and tiles[2].isEnabled()
        assert [t.isChecked() for t in tiles] == [False, False, True]  # the model in use is highlighted
        assert selector.rescan_button.isEnabled()
        assert window.class_legend_widget._names == ['Background', 'Leaf', 'Wood']
        assert window._lut[1].tolist() == [5, 245, 21]                # the package's colour

        tiles[1].click()                                               # 'linear'
        assert [t.isChecked() for t in tiles] == [False, True, False]
        assert window._pipeline.classifier.category_names == {7: 'low', 9: 'high'}
        assert Path(window.config.model.path).name == 'linear.joblib'  # a restart keeps the choice

        def shown():
            window._refresh_gui_slot()
            return window._shown_generation == 1
        assert wait_for(shown)
        assert window.class_legend_widget._names == ['low', 'high']
        assert [t.isChecked() for t in selector.tiles] == [False, True, False]
        assert window._lut[1].tolist() != [5, 245, 21]                # Tab10 again
        assert 'linear.joblib' in window._info_label.text()
        for _ in range(5):
            window._refresh_gui_slot()
        s = window._pipeline.snapshot()
        assert set(np.flatnonzero(s.counts).tolist()) <= {7, 9} and s.error is None

        tiles[0].click()                                               # not usable: nothing happens
        window._on_model_chosen(tiles[0].entry)
        tiles[1].click()                                               # the model in use: stays highlighted
        assert window._pipeline.classifier.category_names == {7: 'low', 9: 'high'}
        assert [t.isChecked() for t in tiles] == [False, True, False]
    finally:
        assert window._pipeline.stop()                 # not close(): that saves the user's window layout


def test_selector_tiles_stay_readable(tmp_path):
    """Same-named exports get their file names; a long name or note never widens the panel."""
    pytest.importorskip('PyQt5')
    from PyQt5.QtWidgets import QApplication
    from model_catalog import ModelEntry
    from widgets.model_selector import TILE_HEIGHT, ModelSelector
    app = QApplication.instance() or QApplication([])
    selector = ModelSelector()
    names = ['Kratom_adulteration_1794300998.hsipkg', 'Kratom_adulteration_1794399999.zip', 'MLC_874171_1794299876.hsipkg',
             'PET bottles clear and coloured sorting line number two.hsipkg']
    selector.set_entries([ModelEntry(tmp_path / n, n, True, n, object(), 'MLC · 3 classes') for n in names])
    assert [t.name_label.text() for t in selector.tiles] == [
        names[0].replace('_', ' '), names[1].replace('_', ' '), 'MLC 874171', 'PET bottles clear and coloured sorting line number two']
    narrow = selector.minimumSizeHint().width()

    class Elsewhere:
        path = 'speed-only model (all bands x 4 classes, no trained weights) ' * 3
    selector.set_current(Elsewhere())                      # not a tile: a long note, clipped
    assert 'not in models' in selector._note.text() and Elsewhere.path in selector._note.toolTip()
    assert selector.minimumSizeHint().width() == narrow
    assert not any(t.isChecked() for t in selector.tiles)
    selector.resize(240, 600)
    selector.show()
    app.processEvents()
    short, long = selector.tiles[2], selector.tiles[3]
    assert short.height() >= TILE_HEIGHT
    assert long.height() >= long.layout().heightForWidth(long.width()) > TILE_HEIGHT      # all lines shown
    selector.hide()


# --------------------------------------------------------------------------- resampling ---

#: the camera after a new wavelength calibration: the same 5 nm bands, 2 nm beside the models' bands
RECALIBRATED = 898.0 + 5.0 * np.arange(13)


def on_model_bands(x):
    """``x`` (n, 13) on ``RECALIBRATED`` interpolated onto the models' ``WAVELENGTHS`` (float64)."""
    return np.stack([np.interp(WAVELENGTHS, RECALIBRATED, row) for row in x])


def test_recalibrated_input_is_resampled_onto_the_model_bands(tmp_path):
    """A model trained before a wavelength recalibration classifies the interpolated input."""
    from inference_engine import Resampled
    x = np.random.default_rng(5).random((200, RECALIBRATED.size)).astype(np.float32) + 0.05
    expected = on_model_bands(x).astype(np.float32)
    for classifier in (load_classifier(build_package(tmp_path, name='package')),
                       load_classifier(linear_model(tmp_path))):
        exact, text = classifier.bind(WAVELENGTHS, WAVELENGTHS.size)
        assert isinstance(exact, np.ndarray) and 'resampled' not in text         # same bands: nothing changes
        binding, text = classifier.bind(RECALIBRATED, RECALIBRATED.size)
        assert isinstance(binding, Resampled) and 'resampled' in text and '2.00 nm' in text
        assert binding.max_shift_nm == pytest.approx(2.0)
        np.testing.assert_allclose(binding.apply(x), expected[:, classifier.selected_bands], rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(classifier.predict(x, binding), classifier.predict(expected, exact))
        ids, confidence = classifier.predict_confidence(x, binding)
        ids_exact, confidence_exact = classifier.predict_confidence(expected, exact)
        np.testing.assert_array_equal(ids, ids_exact)
        np.testing.assert_allclose(confidence, confidence_exact, rtol=1e-3, atol=1e-4)
        # refused: resampling switched off, another spectral resolution, bands outside the input
        with pytest.raises(ValueError, match='no input band.*resampling is off'):
            classifier.bind(RECALIBRATED, RECALIBRATED.size, resample=False)
        coarse = 890.0 + 12.0 * np.arange(7)                                      # 890-962 nm in 12 nm bands
        with pytest.raises(ValueError, match='another spectral resolution'):
            classifier.bind(coarse, coarse.size)
        with pytest.raises(ValueError, match='not all inside'):
            classifier.bind(RECALIBRATED + 40.0, RECALIBRATED.size)
        with pytest.raises(ValueError, match='do not increase'):
            classifier.bind(RECALIBRATED[::-1], RECALIBRATED.size)
        # the band spacing must be the training's: 0.8 .. 1.25 times, pinned from both sides
        # (the package reads all 12 bands, so on each of these grids some band is off by > 0.5 nm)
        lo, hi = WAVELENGTHS[0] - 1.8, WAVELENGTHS[-1] + 7.0
        for step, fits in ((2.5, False), (3.75, False), (4.25, True), (6.0, True), (6.5, False)):
            if classifier.W_fused is not None:
                break
            grid = np.arange(lo, hi, step)
            if fits:
                assert isinstance(classifier.bind(grid, grid.size)[0], Resampled), step
            else:
                with pytest.raises(ValueError, match='another spectral resolution'):
                    classifier.bind(grid, grid.size)
        # a recalibration moves the whole grid: the end band may lie up to half a band outside
        assert isinstance(classifier.bind(WAVELENGTHS + 2.0, WAVELENGTHS.size)[0], Resampled)
        assert isinstance(classifier.bind(WAVELENGTHS - 2.0, WAVELENGTHS.size)[0], Resampled)
        if classifier.W_fused is None:                     # the package reads the first band (900 nm) too
            with pytest.raises(ValueError, match='not all inside'):
                classifier.bind(WAVELENGTHS + 3.0, WAVELENGTHS.size)

    # calibration drift, not a constant shift: every band has its own weight (folded linear path too)
    model_wl = 900.0 + 4.994 * np.arange(12)
    drifted = 898.8 + 5.004 * np.arange(14)
    classifier = load_classifier(linear_model(tmp_path, 'drift', model_wl))
    binding, _ = classifier.bind(drifted, drifted.size)
    exact, _ = classifier.bind(model_wl, model_wl.size)
    assert isinstance(binding, Resampled) and np.unique(binding.weight).size > 1
    x = np.random.default_rng(6).random((200, drifted.size)).astype(np.float32) + 0.05
    model_axis = classifier.band_wavelengths.astype(np.float64)
    expected = np.stack([np.interp(model_axis, drifted, row) for row in x]).astype(np.float32)
    np.testing.assert_allclose(binding.apply(x), expected[:, classifier.selected_bands], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(classifier._scores(x, binding), classifier._scores(expected, exact), rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(classifier.predict(x, binding), classifier.predict(expected, exact))

    # a band outside the input is extrapolated from the two edge bands; one on an input band reads only that
    edge = Resampled(np.array([900.0, 905.0, 910.0]), np.array([898.0, 902.5, 910.0, 912.0]))
    np.testing.assert_allclose(edge.apply(np.array([[1.0, 3.0, 7.0]], np.float32)), [[0.2, 2.0, 7.0, 8.6]], rtol=1e-6)
    on_band = Resampled(np.array([900.0, 905.0, 910.0, 915.0]), np.array([905.0, 910.0]))
    assert on_band.columns.tolist() == [1, 2]
    np.testing.assert_array_equal(on_band.apply(np.array([[np.inf, 1.0, 2.0, np.nan]], np.float32)), [[1.0, 2.0]])


def test_constant_pixel_stays_invalid_through_resampling(tmp_path):
    """A saturated pixel (the same value in every band) has no SNV: background, confidence 0."""
    from inference_engine import Resampled
    classifier = load_classifier(build_package(tmp_path, name='package'))
    drifted = 898.8 + 5.02 * np.arange(13)                 # every band its own weight
    binding, _ = classifier.bind(drifted, drifted.size)
    assert isinstance(binding, Resampled) and np.unique(binding.weight).size > 1
    x = np.random.default_rng(8).random((6, drifted.size)).astype(np.float32) + 0.05
    x[1], x[4] = 4095.0, 16383.0
    assert np.unique(binding.apply(x)[1]).size == 1 and np.unique(binding.apply(x)[4]).size == 1
    ids, confidence = classifier.predict_confidence(x, binding)
    assert ids[[1, 4]].tolist() == [0, 0] and confidence[[1, 4]].tolist() == [0.0, 0.0]
    assert classifier.predict(x, binding)[[1, 4]].tolist() == [0, 0]
    assert (confidence[[0, 2, 3, 5]] > 0).all()


def test_resample_switch_must_be_a_boolean(tmp_path):
    config, *_ = setup(tmp_path)
    for text in ('False', 'off', 0):
        with pytest.raises(ValueError, match='model.resample'):
            replace(config, model=replace(config.model, resample=text)).validate()


def test_pipeline_runs_a_model_from_before_the_recalibration(tmp_path):
    """Start, switch and list models on an input whose bands lie 2 nm beside the models'."""
    config, package, linear, _ = setup(tmp_path, loop=False, fps=0)
    x = np.random.default_rng(21).random((8 * 15, RECALIBRATED.size)).astype(np.float32) + 0.05
    write_envi(tmp_path, x.reshape(8, 15, RECALIBRATED.size), 'bip', wavelengths=RECALIBRATED)
    with pytest.raises(ValueError, match='resampling is off'):
        build_pipeline(replace(config, model=replace(config.model, resample=False)))

    pipeline = build_pipeline(config)                       # model.path is the package: resampled at the start
    assert 'resampled' in pipeline.plans[0].provenance['model bands'][0]
    entries = {e.path.name: e for e in scan_models(config, pipeline.plans, pipeline.classifier)}
    assert all(e.usable and e.resampled_nm == pytest.approx(2.0) for e in entries.values())
    assert entries['package.hsipkg'].kind == 'MLC · 3 classes · resampled 2.0 nm'
    assert 'another wavelength calibration' in entries['linear.joblib'].detail
    off = replace(config, model=replace(config.model, resample=False))
    assert not any(e.usable for e in scan_models(off, pipeline.plans, pipeline.classifier)
                   if e.classifier is not pipeline.classifier)
    second = load_classifier(linear)
    pipeline.set_classifier(second)                         # switched before the first line
    pipeline.start()
    try:
        assert wait_for(lambda: pipeline.snapshot().state == 'finished')
    finally:
        pipeline.stop()
    s = pipeline.snapshot()
    exact, _ = second.bind(WAVELENGTHS, WAVELENGTHS.size)
    expected = np.bincount(second.predict(on_model_bands(x).astype(np.float32), exact), minlength=256)
    assert s.error is None and s.lines_processed == 15
    np.testing.assert_array_equal(s.counts, expected)


def test_tile_marks_a_resampled_model(tmp_path):
    pytest.importorskip('PyQt5')
    from PyQt5.QtWidgets import QApplication
    from model_catalog import ModelEntry
    from widgets.model_selector import ModelTile
    app = QApplication.instance() or QApplication([])      # noqa: F841 (widgets need the application)
    plain = ModelTile(ModelEntry(tmp_path / 'a.hsipkg', 'a', True, 'a', object(), 'MLC · 3 classes'))
    moved = ModelTile(ModelEntry(tmp_path / 'b.hsipkg', 'b', True, 'b', object(), 'MLC · 3 classes · resampled 1.2 nm',
                                 resampled_nm=1.17))
    assert plain.kind_label.property('resampled') is False and moved.kind_label.property('resampled') is True
    assert moved.isEnabled() and moved.kind_label.text().endswith('resampled 1.2 nm')
