"""Model input of 2 x 2 pixels (processing.spatial_bin = 2): pairing, the mean, what is reported."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import bin_2x2, build_pipeline  # noqa: E402
from test_model_switch import setup, wait_for  # noqa: E402


def block_means(cube: np.ndarray) -> np.ndarray:
    """``(slit, lines, bands)`` -> per pixel the mean of its 2 x 2 block (partial at odd ends)."""
    slit, lines = cube.shape[:2]
    out = np.empty(cube.shape, np.float64)
    for r in range(slit):
        for n in range(lines):
            r0, n0 = r // 2 * 2, n // 2 * 2
            out[r, n] = cube[r0:r0 + 2, n0:n0 + 2].astype(np.float64).mean(axis=(0, 1))
    return out


def test_bin_2x2_is_the_block_mean_with_partial_blocks():
    rng = np.random.default_rng(0)
    a, b = rng.random((7, 5)).astype(np.float32), rng.random((7, 5)).astype(np.float32)
    both = bin_2x2([a, b])
    assert both.shape == (4, 5) and both.dtype == np.float32
    np.testing.assert_allclose(both[:3], (a[0:6:2] + a[1:6:2] + b[0:6:2] + b[1:6:2]) / 4, rtol=1e-6)
    np.testing.assert_allclose(both[3], (a[6] + b[6]) / 2, rtol=1e-6)          # odd last row
    alone = bin_2x2([a.astype(np.uint16)])                                     # one line: rows only
    np.testing.assert_allclose(alone[:3], (a.astype(np.uint16)[0:6:2] + a.astype(np.uint16)[1:6:2]) / 2.0)


def test_pipeline_classifies_the_2x2_mean_and_reports_it_on_every_pixel(tmp_path):
    """15 lines of 8 slit pixels: lines (0,1) ... (12,13) paired, line 14 alone; the spectrum
    preview gives each pixel its block's model input, the class is the block's, nothing is lost."""
    config, package, linear, cube = setup(tmp_path, loop=False, fps=0)
    config = replace(config, processing=replace(config.processing, spatial_bin=2)).validate()
    pipeline = build_pipeline(config)
    assert pipeline.spatial_bin == 2
    pipeline.start_spectrum_history()
    pipeline.start()
    try:
        assert wait_for(lambda: pipeline.snapshot().state == 'finished')
    finally:
        pipeline.stop()
    snap = pipeline.snapshot()
    slit, lines, bands = cube.shape
    assert snap.lines_processed == lines and snap.lines_lost == 0
    assert snap.counts.sum() == slit * lines                       # every pixel counted once
    expected = block_means(cube)
    for n in range(lines):
        for r in range(slit):
            np.testing.assert_allclose(pipeline.spectrum_at(n, r), expected[r, n], rtol=1e-5)
    # the class of each line = the model's class of the block means, row pairs alike
    columns = pipeline.plans[0].columns
    classes = {}
    for n in range(lines):
        classes[n] = pipeline.classifier.predict(expected[:, n].astype(np.float32), columns)
        assert (classes[n][0::2] == classes[n][1::2]).all()
    canvas, drawn = snap.canvas, snap.canvas_lines
    for col in np.flatnonzero(drawn >= 0):
        np.testing.assert_array_equal(canvas[:, col], classes[int(drawn[col])])


def test_spatial_bin_switches_while_running_and_is_validated(tmp_path):
    config, package, linear, cube = setup(tmp_path)
    with pytest.raises(ValueError):
        replace(config, processing=replace(config.processing, spatial_bin=3)).validate()
    pipeline = build_pipeline(config)
    assert pipeline.spatial_bin == 1
    pipeline.start_spectrum_history()
    pipeline.start()
    try:
        assert wait_for(lambda: pipeline.snapshot().lines_processed > 40)
        pipeline.set_spatial_bin(2)
        with pytest.raises(ValueError):
            pipeline.set_spatial_bin(4)
        start = pipeline.snapshot().lines_processed
        assert wait_for(lambda: pipeline.snapshot().lines_processed > start + 60)
        line = pipeline.snapshot().lines_processed - 6             # a line of the binned run, still kept
        values = pipeline.spectrum_at(line, 2)
        assert values is not None
        np.testing.assert_array_equal(values, pipeline.spectrum_at(line, 3))      # one block: rows 2-3 ...
        # ... and this line with its neighbour (pairs start where the switch happened, not at even lines)
        assert any(np.array_equal(values, pipeline.spectrum_at(other, 2)) for other in (line - 1, line + 1))
        pipeline.set_spatial_bin(1)
        start = pipeline.snapshot().lines_processed
        assert wait_for(lambda: pipeline.snapshot().lines_processed > start + 40)
        assert pipeline.snapshot().lines_lost == 0
    finally:
        assert pipeline.stop()


def test_a_slow_camera_still_pairs_its_lines_and_a_paused_one_flushes(tmp_path):
    """The idle flush waits a few line periods, not one idle tick: lines 0.2 s apart (5 lines/s)
    still go in pairs; a line whose partner does not come goes alone after the limit."""
    import time
    from pipeline import Pipeline, _Job
    config, package, linear, cube = setup(tmp_path, loop=False, fps=0)
    config = replace(config, processing=replace(config.processing, spatial_bin=2)).validate()
    pipeline = build_pipeline(config)
    sent = []

    class Work:                                                       # see what goes to the workers
        put = staticmethod(sent.append)
    pipeline._work = Work()
    t = time.perf_counter()
    for n in range(4):                                                # 0.2 s apart, as 5 lines/s
        pipeline._dispatch(_Job(n, n, n, t + 0.2 * n, 0, n, 0, t + 0.2 * n, pipeline._model))
    assert [j.partner is not None for j in sent] == [True, True] and pipeline._waiting is None
    pipeline._dispatch(_Job(4, 4, 4, time.perf_counter(), 0, 4, 0, time.perf_counter(), pipeline._model))
    pipeline._flush_held()
    assert len(sent) == 2 and pipeline._waiting is not None           # waits for line 5 ...
    pipeline._waiting.t_queued -= Pipeline.FLUSH_MAX_S + 0.01
    pipeline._flush_held()                                            # ... until the camera paused
    assert len(sent) == 3 and sent[-1].partner is None and pipeline._waiting is None
