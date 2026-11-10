"""Live camera read in its own process (live.reader = "process") against the fake camera - no hardware.

The fake camera is built inside the camera process by :func:`make_fake` (a picklable factory), so
these tests cross the real process boundary: spawn, shared frame buffers, the two semaphores and
the control pipe.
"""
import os
import sys
import threading
import time
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from camera_process import open_camera_process  # noqa: E402
from pipeline import Pipeline  # noqa: E402
from sim_config import LiveConfig, build_plan  # noqa: E402
from test_live_camera import FakeLineCamera, ValueClassifier, check_line_order, live_config, run_until  # noqa: E402


def make_fake(kwargs, settings):
    """``create_backend`` stand-in, called in the camera process."""
    return FakeLineCamera(**kwargs)


def process_config(tmp_path, workers=2, buffers=64, **acquisition):
    config = live_config(tmp_path, workers=workers, buffers=buffers, **acquisition)
    return replace(config, live=LiveConfig(read_timeout_ms=50, reader='process')).validate()


def process_pipeline(tmp_path, delay=0.0, workers=2, buffers=64, **camera):
    config = process_config(tmp_path, workers=workers, buffers=buffers)
    cam, live = open_camera_process(config, backend_factory=partial(make_fake, camera))
    try:
        plan = build_plan(config, live)
        plan.columns = None
        return Pipeline(config, [live], [plan], ValueClassifier(delay), camera=cam)
    except BaseException:
        cam.close()
        raise


def test_frames_cross_the_process_in_order_and_the_process_ends(tmp_path):
    pipeline = process_pipeline(tmp_path, rate_hz=2000, into=True)
    process = pipeline.camera.process
    assert process.is_alive() and process.pid != os.getpid()
    s = run_until(pipeline, 400)
    assert s.error is None and s.source_kind == 'camera' and s.line_clock == 'arrival'
    assert s.lines_lost == 0 and s.lines_processed >= 400 and s.frames_delivered >= 400
    assert check_line_order(s) == 0 and s.canvas.shape[1] >= 400
    assert not process.is_alive() and process.exitcode == 0
    assert pipeline._shared is None                    # shared frame memory released


def test_frame_id_gaps_in_the_camera_process_are_lost_lines(tmp_path):
    s = run_until(process_pipeline(tmp_path, rate_hz=2000, skip=(10, 11, 12, 50)), 200)
    assert s.error is None
    assert s.lines_lost == 4 and s.lines_lost_camera == 4
    assert check_line_order(s) == 4


def test_overruns_and_timeouts_are_counted_in_the_camera_process(tmp_path):
    s = run_until(process_pipeline(tmp_path, delay=0.004, workers=1, buffers=2, rate_hz=2000, timeouts=5), 300)
    assert s.error is None and s.grab_timeouts == 5
    assert s.lines_lost > 0 and s.lines_lost_camera == 0      # no free buffer: overruns
    assert 0 < check_line_order(s) <= s.lines_lost


def test_camera_timestamps_work_across_the_process(tmp_path):
    """perf_counter is one clock for both processes, so the camera-clock latency stays right."""
    s = run_until(process_pipeline(tmp_path, rate_hz=1000, stamps=True, stamp_age_s=0.02), 800)
    assert s.error is None and s.line_clock == 'camera'
    assert 20.0 <= s.latency_ms['p50'] < 200.0


def test_open_errors_come_back_with_their_type(tmp_path):
    config = process_config(tmp_path)
    with pytest.raises(ValueError, match='does not support acquisition.camera_bin'):
        open_camera_process(config, backend_factory=partial(make_fake, {'bins': (1,)}))
    with pytest.raises(RuntimeError, match='refused bin 2'):
        open_camera_process(config, backend_factory=partial(make_fake, {'refuse_bin': 2}))


def test_a_read_error_in_the_camera_process_stops_the_pipeline(tmp_path):
    pipeline = process_pipeline(tmp_path, rate_hz=2000, resize_after=30)
    s = run_until(pipeline, 10_000, timeout=10)
    assert s.state == 'error' and 'RuntimeError' in s.error and 'differs' in s.error
    assert not pipeline.camera.process.is_alive()


def test_stop_without_start_ends_the_camera_process(tmp_path):
    pipeline = process_pipeline(tmp_path, rate_hz=1000)
    assert pipeline.stop()
    assert not pipeline.camera.process.is_alive() and pipeline._shared is None


def test_a_camera_process_stuck_in_the_sdk_is_terminated(tmp_path):
    pipeline = process_pipeline(tmp_path, rate_hz=1000, block_after=50, block_s=60.0)
    pipeline.camera.close_timeout_s = 0.5
    pipeline.start()
    t = time.perf_counter()
    while pipeline.snapshot().lines_processed < 40 and time.perf_counter() - t < 20:
        time.sleep(0.01)
    time.sleep(0.2)                                    # the camera process is now inside the blocked read
    t = time.perf_counter()
    assert pipeline.stop()                             # no thread left behind, unlike the thread reader
    assert time.perf_counter() - t < 5.0
    assert not pipeline.camera.process.is_alive() and pipeline.busy_threads() == []


def test_a_busy_gui_process_does_not_slow_the_camera(tmp_path):
    """Three pure-Python threads hog this process's GIL; the camera process still reads every frame."""
    previous = sys.getswitchinterval()
    sys.setswitchinterval(0.005)                       # Python's default: the worst case for a reader thread
    stop = threading.Event()

    def busy():
        x = 0
        while not stop.is_set():
            for i in range(1000):
                x += i * i

    pipeline = process_pipeline(tmp_path, rate_hz=1000, buffers=256)
    hogs = [threading.Thread(target=busy, daemon=True) for _ in range(3)]
    try:
        pipeline.start()
        t = time.perf_counter()
        while pipeline.snapshot().state == 'starting' and time.perf_counter() - t < 20:
            time.sleep(0.01)
        for hog in hogs:
            hog.start()
        time.sleep(0.3)
        n0, t0 = pipeline.camera.counters()['received'], time.perf_counter()
        time.sleep(1.5)
        n1, t1 = pipeline.camera.counters()['received'], time.perf_counter()
        s = pipeline.snapshot()
    finally:
        stop.set()
        for hog in hogs:
            if hog.is_alive():
                hog.join()
        assert pipeline.stop()
        sys.setswitchinterval(previous)
    assert s.error is None and s.lines_lost_camera == 0
    assert (n1 - n0) / (t1 - t0) > 850                 # the camera's 1000 frames/s, GIL or not
