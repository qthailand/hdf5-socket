"""Live camera source (source.kind = camera) against a fake hsi camera backend - no hardware."""
import json
import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from camera_source import CameraClock, CameraFrameGaps, open_camera, resolve_cti  # noqa: E402
from modules.camera import AbstractCameraBackend, CameraBackendInfo, factory  # noqa: E402
from pipeline import LOST_CLASS, Pipeline, build_pipeline  # noqa: E402
from sim_config import AcquisitionConfig, CameraConfig, CorrectionConfig, LiveConfig, ModelConfig, \
    ProcessingConfig, SimulationConfig, build_plan, load_config, parse_set_overrides  # noqa: E402

SENSOR = (12, 80)                    # (slit, spectral) at bin 1


class FakeLineCamera(AbstractCameraBackend):
    """Free-running camera: frame ``n`` (from 1) is filled with ``n % 200 + 1``.

    ``skip`` are frame numbers the camera exposes but the link loses (they never reach the
    reader); ``ids`` reports them as GenICam frame-id gaps, ``sdk_counter`` as POA's dropped
    counter. ``stamps`` gives GenICam-style camera timestamps on a clock ``offset_ns`` ahead of
    the host, ``stamp_age_s`` before the frame is returned. ``into`` writes into the buffer the
    reader provides (GenICam's zero-copy path).
    """

    def __init__(self, rate_hz=2000.0, ids=True, skip=(), sdk_counter=False, stamps=False, stamp_age_s=0.003,
                 offset_ns=7_000_000_000, into=False, timeouts=0, start_bin=1, bins=(1, 2), refuse_bin=None,
                 resize_after=None, stall_after=None, block_after=None, block_s=0.0, slow_start=False):
        super().__init__()
        self.period = 1.0 / rate_hz
        self.ids, self.skip, self.sdk_counter = ids, set(skip), sdk_counter
        self.stamps, self.stamp_age_s, self.offset_ns = stamps, stamp_age_s, offset_ns
        self.into, self.timeouts, self.resize_after = into, timeouts, resize_after
        self.stall_after, self.block_after, self.block_s = stall_after, block_after, block_s
        self.slow_start, self.starting, self.blocked = slow_start, threading.Event(), False
        self.bin, self.bins, self.refuse_bin = start_bin, tuple(bins), refuse_bin
        self.n = 0
        self.dropped = 0
        self.gain = self.exposure_ms = None
        self.closed = 0
        self.provided = 0
        self.stop_event = None
        self.threads = {}                 # call -> set of thread names
        self._next_t = 0.0

    def _note(self, call):
        self.threads.setdefault(call, set()).add(threading.current_thread().name)

    # hardware hooks
    def _open_device(self):
        self._note('open')

    def _close_device(self):
        self._note('close')
        self.closed += 1
        if self.stop_event is not None:
            self.stop_event.set()

    def _start_device(self):
        self._note('start')
        if self.slow_start:               # like GenICam's AcquisitionStart retries, cut short by a stop
            self.starting.set()
            while not (self.stop_event is not None and self.stop_event.is_set()):
                time.sleep(0.01)
            raise RuntimeError('GenICam start interrupted')
        self.dropped = 0
        self._next_t = time.perf_counter()

    def _stop_device(self):
        self._note('stop')

    def set_stop_event(self, event):
        self.stop_event = event

    def _produce(self):
        self._note('read')
        if self.stop_event is not None and self.stop_event.is_set():
            return None
        if self.timeouts:
            self.timeouts -= 1
            time.sleep(0.002)
            return None
        if self.block_after is not None and self.n >= self.block_after and not self.blocked:
            self.blocked = True           # stuck inside the SDK: the stop event is not looked at
            time.sleep(self.block_s)
            return None
        if self.stall_after is not None and self.n >= self.stall_after:
            time.sleep(0.005)             # the camera sends nothing (link down, trigger stopped)
            return None
        delay = self._next_t - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        self._next_t = max(self._next_t + self.period, time.perf_counter() - 0.05)
        self.n += 1
        while self.n in self.skip:        # exposed, then lost on the link
            self.dropped += 1
            self.n += 1
        if self.resize_after is not None and self.n > self.resize_after:
            self.bin = 1                  # e.g. someone changed the bin behind the pipeline's back
        stamp = time.perf_counter_ns() + self.offset_ns - int(self.stamp_age_s * 1e9) if self.stamps else 0
        self.last_frame_info = (stamp, self.n) if self.ids else None
        return np.full((self.info.height, self.info.width), self.n % 200 + 1, np.uint16)

    def _fetch_raw_frame(self, timeout_ms=1000):
        return self._produce()

    def read_frame_raw_into(self, provide, timeout_ms=1000):
        frame = self._produce()
        if frame is None or not self.into:
            return frame
        out = provide(frame.shape)
        if out is None:
            return frame
        self.provided += 1
        out[...] = frame
        return out

    def camera_clock_offset_ns(self):
        return self.offset_ns if self.stamps else None

    # parameters and diagnostics
    def set_params(self, gain, exposure_ms):
        self.gain, self.exposure_ms = float(gain), round(float(exposure_ms) * 1.02, 4)   # clamps a little

    def set_bin(self, bin_value):
        if bin_value not in self.bins:
            raise ValueError(f"bin {bin_value} not supported")
        if bin_value == self.refuse_bin:
            raise RuntimeError(f"camera refused bin {bin_value}")
        self.bin = bin_value

    @property
    def info(self):
        return CameraBackendInfo(camera_id=0, bit_depth=16, pixel_size=1.0, width=SENSOR[1] // self.bin,
                                 height=SENSOR[0] // self.bin, image_format='Mono12')

    exposure_range_us = property(lambda self: (10, 1_000_000))
    gain_range = property(lambda self: (0, 24))
    current_bin = property(lambda self: self.bin)

    def get_bins(self):
        return self.bins

    def dropped_images_count(self):
        return self.dropped if self.sdk_counter else 0

    def get_temperature(self):
        return 31.5

    def model_name(self):
        return 'FakeCam'

    def serial_number(self):
        return 'SN1'

    def sensor_bit_depth(self):
        return 12


class ValueClassifier:
    """Class id = first binned pixel (the frame number pattern), so the line order is visible."""
    category_names = {}

    def __init__(self, delay=0.0):
        self.delay = delay

    def predict(self, binned, columns):
        if self.delay:
            time.sleep(self.delay)
        return np.full(binned.shape[0], min(254, int(round(float(binned[0, 0])))), np.uint8)


@pytest.fixture
def install(monkeypatch):
    """``install(camera)``: ``create_backend`` returns ``camera``; the settings it got are returned."""
    def _install(camera):
        made = {}

        def create(settings):
            made['settings'] = settings
            return camera
        monkeypatch.setattr(factory, 'create_backend', create)
        return made
    return _install


def live_config(tmp_path, workers=2, buffers=64, corrections=None, **acquisition):
    acq = dict(camera_backend='poa', camera_bin=2, exposure_ms=1.0, gain=1.0, raw_inverted=False,
               m_nm_per_px_bin1=10.0, c_nm=400.0, resolution_nm=20.0)
    acq.update(acquisition)
    config = SimulationConfig(base_dir=str(tmp_path))
    return replace(config, source=replace(config.source, kind='camera'), acquisition=AcquisitionConfig(**acq),
                   corrections=corrections or CorrectionConfig(enabled=False, backend='cpu'),
                   processing=ProcessingConfig(workers=workers, profile_every_s=0, numba_threads=1),
                   camera=CameraConfig(buffer_frames=buffers), model=ModelConfig(path='none'),
                   live=LiveConfig(read_timeout_ms=50)).validate()


def run_until(pipeline, lines, timeout=20.0):
    """Start, wait until ``lines`` lines are counted (or the pipeline ends), stop; final snapshot."""
    pipeline.start()
    t = time.perf_counter()
    try:
        while time.perf_counter() - t < timeout:
            s = pipeline.snapshot()
            if s.lines_processed + s.lines_lost >= lines or s.state in ('error', 'stopped', 'finished'):
                break
            time.sleep(0.01)
    finally:
        assert pipeline.stop()
    return pipeline.snapshot()


def value_pipeline(tmp_path, camera, install, delay=0.0, workers=2, buffers=64):
    """A pipeline whose model reports the frame number, on the plan resolved from the camera."""
    install(camera)
    config = live_config(tmp_path, workers=workers, buffers=buffers)
    cam, live = open_camera(config)
    plan = build_plan(config, live)
    plan.columns = None
    return Pipeline(config, [live], [plan], ValueClassifier(delay), camera=cam)


def check_line_order(s):
    """Belt column j is line j: lost (magenta) or the frame number pattern of frame j + 1."""
    row = s.canvas[0].astype(int)
    expected = (np.arange(row.size) + 1) % 200 + 1
    assert np.all((row == LOST_CLASS) | (row == expected))
    return int(np.count_nonzero(row == LOST_CLASS))


# ------------------------------------------------------------------------ end to end ---

def test_live_camera_through_build_pipeline(tmp_path, install):
    camera = FakeLineCamera(rate_hz=1000)
    made = install(camera)
    pipeline = build_pipeline(live_config(tmp_path))
    settings = made['settings']
    assert (settings.camera_backend, settings.default_bin, settings.default_exposure_ms, settings.default_gain) == \
        ('poa', 2, 1.0, 1.0)
    # opened and configured off the main thread, before any stream; bin and params applied
    assert 'MainThread' not in camera.threads['open'] and camera.bin == 2
    assert camera.gain == 1.0 and camera.exposure_ms == 1.02 and 'start' not in camera.threads
    live = pipeline.sources[0]
    assert live.name == 'poa FakeCam (SN1)' and live.frame_shape == (6, 40) and live.stage == 'raw'
    plan = pipeline.plans[0]
    assert plan.camera_bin == 2 and plan.provenance['camera_bin'][1] == 'camera:camera_bin'
    assert plan.provenance['camera_backend'] == ('poa', 'camera:camera_backend')
    assert 'live camera' in plan.provenance['input'][0]
    s = run_until(pipeline, 800)                       # > 0.5 s: one rate measurement
    assert s.error is None and s.state == 'stopped' and s.source_kind == 'camera'
    assert s.lines_processed >= 800 and s.lines_lost == 0 and s.counted_lines == s.lines_processed
    assert s.position[0] == live.name and s.position[2] == 0
    assert 500 < s.target_fps < 1500 and s.line_clock == 'arrival'
    # the camera thread owned the stream and closed the camera exactly once
    assert camera.threads['start'] == {'camera'} and camera.threads['read'] == {'camera'}
    assert camera.closed == 1 and not camera._is_open
    with pytest.raises(RuntimeError, match='own line rate'):
        pipeline.set_fps(100)


def test_frames_keep_their_order_and_reach_the_buffers_directly(tmp_path, install):
    camera = FakeLineCamera(rate_hz=3000, into=True)
    s = run_until(value_pipeline(tmp_path, camera, install), 400)
    assert s.error is None and s.lines_lost == 0
    assert check_line_order(s) == 0 and s.canvas.shape[1] >= 400
    assert camera.provided >= 400                      # written into the pipeline's own buffers


def test_frame_id_gaps_are_lost_lines_drawn_magenta(tmp_path, install):
    camera = FakeLineCamera(rate_hz=2000, skip=(10, 11, 12, 50))
    s = run_until(value_pipeline(tmp_path, camera, install), 200)
    assert s.error is None
    assert s.lines_lost == 4 and s.lines_lost_camera == 4
    assert check_line_order(s) == 4


def test_poa_style_dropped_counter_counts_lost_frames(tmp_path, install):
    camera = FakeLineCamera(rate_hz=2000, ids=False, sdk_counter=True, skip=(20, 21, 22))
    s = run_until(value_pipeline(tmp_path, camera, install), 200)
    assert s.error is None and s.lines_lost_camera == 3 and s.lines_lost == 3
    assert check_line_order(s) == 3


def test_overrun_reads_and_drops_the_frame_and_counts_every_line(tmp_path, install):
    camera = FakeLineCamera(rate_hz=2000)
    pipeline = value_pipeline(tmp_path, camera, install, delay=0.004, workers=1, buffers=2)
    s = run_until(pipeline, 300)
    assert s.error is None
    assert s.lines_lost > 0 and s.lines_lost_camera == 0
    assert s.frames_delivered + s.lines_lost == camera.n       # every exposed line accounted for
    # lost lines are drawn when the next delivered line arrives (those after the last one are not)
    assert 0 < check_line_order(s) <= s.lines_lost


def test_read_timeouts_are_counted_and_retried(tmp_path, install):
    camera = FakeLineCamera(rate_hz=2000, timeouts=5)
    s = run_until(value_pipeline(tmp_path, camera, install), 100)
    assert s.error is None and s.grab_timeouts == 5 and s.lines_lost == 0 and s.lines_processed >= 100


def test_camera_timestamps_give_the_latency_from_exposure(tmp_path, install):
    camera = FakeLineCamera(rate_hz=1000, stamps=True, stamp_age_s=0.02)
    s = run_until(value_pipeline(tmp_path, camera, install), 800)
    assert s.error is None and s.line_clock == 'camera'
    assert s.latency_ms['p50'] >= 20.0                 # the 20 ms before the frame arrived count
    queue = next(iter(s.stage_stats.values()))['avg']['queue']
    assert queue < 10.0                                # ... but not as time spent waiting in the buffers


def test_a_changed_frame_size_stops_with_an_error_and_closes(tmp_path, install):
    camera = FakeLineCamera(rate_hz=2000, resize_after=30)
    pipeline = value_pipeline(tmp_path, camera, install)
    s = run_until(pipeline, 10_000, timeout=10)
    assert s.state == 'error' and 'differs' in s.error
    assert camera.closed == 1


# ------------------------------------------------------------------ open / close ---

def test_unsupported_bin_is_an_error_and_the_camera_is_closed(tmp_path, install):
    camera = FakeLineCamera(bins=(1, 3))
    install(camera)
    with pytest.raises(ValueError, match='camera_bin = 2'):
        build_pipeline(live_config(tmp_path))
    assert camera.closed == 1


def test_a_refused_bin_closes_the_camera(tmp_path, install):
    camera = FakeLineCamera(refuse_bin=2)
    install(camera)
    with pytest.raises(RuntimeError, match='refused'):
        build_pipeline(live_config(tmp_path))
    assert camera.closed == 1


def test_a_failure_after_opening_closes_the_camera(tmp_path, install):
    camera = FakeLineCamera()
    install(camera)
    config = live_config(tmp_path)
    config = replace(config, model=ModelConfig(path='missing_model.joblib'))
    with pytest.raises(Exception):
        build_pipeline(config)
    assert camera.closed == 1 and not camera._is_open


def test_stop_without_start_closes_the_camera(tmp_path, install):
    camera = FakeLineCamera()
    install(camera)
    pipeline = build_pipeline(live_config(tmp_path))
    assert camera._is_open
    assert pipeline.stop() and camera.closed == 1
    assert pipeline.stop() and camera.closed == 1     # idempotent


def test_dark_auto_is_chosen_with_the_values_the_camera_applied(tmp_path, install):
    shape = (6, 40)
    folder = tmp_path / 'calibration' / 'dark_frames' / 'poa'
    folder.mkdir(parents=True)
    np.savez(folder / 'dark_b2_e1.0_g1.0.npz', mean=np.full(shape, 7, np.uint16),
             hot_pixel_map=np.zeros(shape, np.uint8), bin=np.int32(2), exposure=np.float32(1),
             gain=np.float32(1), n_frames=np.int32(5))
    camera = FakeLineCamera()
    install(camera)
    config = live_config(tmp_path, corrections=CorrectionConfig(backend='cpu', dark='auto'),
                         use_file_metadata=False)     # a file switch: the camera's values still count
    pipeline = build_pipeline(config)
    try:
        plan = pipeline.plans[0]
        assert plan.steps['dark']['available'] and plan.steps['dark']['on']
        assert 'dark_b2_e1.0_g1.0.npz' in plan.provenance['dark'][0]
        assert '1.02 ms [camera:exposure_ms]' in plan.provenance['dark'][0]
        assert pipeline.sources[0].metadata['sensor_temperature_c_start'] == 31.5
    finally:
        pipeline.stop()


# ------------------------------------------------------------- frame gaps / clock ---

def test_frame_gaps_count_forward_jumps_only():
    camera = FakeLineCamera(ids=True)
    gaps = CameraFrameGaps(camera)
    gaps.start()
    seen = []
    for frame_id in (5, 6, 9, 10, 3, 4, 4, 7):        # 3: ids restarted / wrapped
        camera.last_frame_info = (0, frame_id)
        seen.append(gaps.missed())
    assert seen == [0, 0, 2, 0, 0, 0, 0, 2]


def test_frame_gaps_from_the_dropped_counter():
    camera = FakeLineCamera(ids=False, sdk_counter=True)
    camera.dropped = 4                                 # before the stream: not this stream's loss
    gaps = CameraFrameGaps(camera)
    gaps.start()
    seen = []
    for dropped in (4, 6, 6, 9, 0, 9, 11):             # 0: a failed SDK call (POA reports 0), not a reset
        camera.dropped = dropped
        seen.append(gaps.missed())
    assert seen == [0, 2, 0, 3, 0, 0, 2]
    camera.dropped = 1                                 # a new stream starts from its own counter
    gaps.start()
    camera.dropped = 3
    assert gaps.missed() == 2


class _StampCamera:
    def __init__(self, offset_ns):
        self.offset_ns, self.last_frame_info, self.latches = offset_ns, None, 0

    def camera_clock_offset_ns(self):
        self.latches += 1
        return self.offset_ns() if callable(self.offset_ns) else self.offset_ns


def _stamp(cam, t, offset_ns=None):
    return int(t * 1e9) + (cam.offset_ns if offset_ns is None else offset_ns)


def test_camera_clock_uses_plausible_stamps_relatches_then_gives_up():
    cam = _StampCamera(offset_ns=5_000_000_000)
    clock = CameraClock(cam, max_bad=3)
    now = time.perf_counter()
    cam.last_frame_info = (_stamp(cam, now - 0.004), 1)
    assert abs(clock.sensor_time(now) - (now - 0.004)) < 1e-6 and clock.source == 'camera'
    cam.last_frame_info = (0, 2)                       # producer without timestamps
    assert clock.sensor_time(now) == now
    assert cam.latches == 1                            # re-latched only after relatch_s
    for i in range(3):                                 # stamps in another unit: implausible
        cam.last_frame_info = (int((now - 0.004) * 1e12), 3 + i)
        assert clock.sensor_time(now) == now
    cam.last_frame_info = (_stamp(cam, now - 0.004), 6)    # after a fresh latch they fit again
    assert abs(clock.sensor_time(now) - (now - 0.004)) < 1e-6 and cam.latches == 2
    for i in range(3):
        cam.last_frame_info = (int((now - 0.004) * 1e12), 7 + i)
        clock.sensor_time(now)
    for i in range(3):                                 # still wrong after the next latch: given up
        cam.last_frame_info = (int((now - 0.004) * 1e12), 10 + i)
        clock.sensor_time(now)
    cam.last_frame_info = (_stamp(cam, now - 0.004), 13)
    assert clock.sensor_time(now) == now and clock.source == 'arrival' and cam.latches == 3


def test_camera_clock_retries_a_failed_first_latch_then_uses_the_arrival_time():
    answers = iter([None, 5_000_000_000])
    cam = _StampCamera(offset_ns=lambda: next(answers))
    clock = CameraClock(cam)
    now = time.perf_counter()
    cam.last_frame_info = (_stamp(cam, now - 0.003, 5_000_000_000), 1)
    assert clock.sensor_time(now) == now and clock.source == 'arrival'      # latch failed once
    assert clock.sensor_time(now + 0.5) == now + 0.5 and cam.latches == 1  # retried a second later
    cam.last_frame_info = (_stamp(cam, now + 1.0 - 0.003, 5_000_000_000), 2)
    assert abs(clock.sensor_time(now + 1.0) - (now + 1.0 - 0.003)) < 1e-6 and clock.source == 'camera'

    cam = _StampCamera(offset_ns=None)                 # a camera that cannot latch at all
    clock = CameraClock(cam)
    cam.last_frame_info = (123456789, 1)
    for k in range(5):
        assert clock.sensor_time(now + k) == now + k
    assert clock.source == 'arrival' and cam.latches == 3    # given up after three attempts


# ------------------------------------------------------------------- stalls / stop ---

def test_rates_fall_to_zero_when_the_camera_stops_delivering(tmp_path, install):
    camera = FakeLineCamera(rate_hz=1000, stall_after=700)
    pipeline = value_pipeline(tmp_path, camera, install)
    pipeline.start()
    try:
        t = time.perf_counter()
        while pipeline.snapshot().lines_processed < 700 and time.perf_counter() - t < 20:
            time.sleep(0.01)
        assert pipeline.snapshot().target_fps > 0
        t = time.perf_counter()
        while pipeline.snapshot().target_fps > 0 and time.perf_counter() - t < 5:
            time.sleep(0.05)
        s = pipeline.snapshot()
        assert s.state == 'running' and s.target_fps == 0 and s.camera_fps == 0 and s.grab_timeouts > 0
    finally:
        assert pipeline.stop()


def test_stop_waits_for_a_read_and_never_closes_the_camera_from_outside(tmp_path, install):
    camera = FakeLineCamera(rate_hz=1000, block_after=50, block_s=2.5)
    pipeline = value_pipeline(tmp_path, camera, install)
    pipeline.start()
    t = time.perf_counter()
    while camera.n < 50 and time.perf_counter() - t < 20:
        time.sleep(0.01)
    time.sleep(0.1)                                    # the camera thread is now inside the blocked read
    t = time.perf_counter()
    assert pipeline.stop(timeout=0.2) is False         # waits one read (50 ms) + 1 s, not 0.2 s
    assert 1.0 <= time.perf_counter() - t < 2.0
    assert camera.closed == 0 and pipeline.busy_threads() == ['camera']
    t = time.perf_counter()
    while camera.closed == 0 and time.perf_counter() - t < 5:
        time.sleep(0.02)
    assert camera.closed == 1 and camera.threads['close'] == {'camera'}   # its own thread closed it
    assert pipeline.stop() and pipeline.busy_threads() == []


def test_a_stop_during_the_camera_start_is_a_stop_not_an_error(tmp_path, install):
    camera = FakeLineCamera(slow_start=True)
    pipeline = value_pipeline(tmp_path, camera, install)
    pipeline.start()
    assert camera.starting.wait(20)
    assert pipeline.stop()
    s = pipeline.snapshot()
    assert s.state == 'stopped' and s.error is None and camera.closed == 1


# ------------------------------------------------------------------------ config ---

def test_config_needs_exposure_and_gain_and_a_known_backend(tmp_path):
    with pytest.raises(ValueError, match='exposure_ms and acquisition.gain'):
        live_config(tmp_path, exposure_ms=None)
    with pytest.raises(ValueError, match='camera_backend'):
        live_config(tmp_path, camera_backend='webcam')
    config = SimulationConfig(base_dir=str(tmp_path))
    with pytest.raises(ValueError, match='source.kind'):
        replace(config, source=replace(config.source, kind='usb')).validate()
    with pytest.raises(ValueError, match='read_timeout_ms'):
        replace(live_config(tmp_path), live=LiveConfig(read_timeout_ms=60000)).validate()
    # file playback does not need them
    assert replace(config, acquisition=AcquisitionConfig(exposure_ms=None)).validate().live_camera is False


def test_live_section_and_set_overrides_load_from_json(tmp_path):
    path = tmp_path / 'simulation.json'
    path.write_text('{"source": {"kind": "camera"}, "acquisition": {"camera_backend": "genicam", '
                    '"exposure_ms": 2.0, "gain": 1.0}, "live": {"camera_index": 1, "cti_path": "producer.cti"}}',
                    encoding='utf-8')
    explicit = parse_set_overrides(['live.read_timeout_ms=250', 'acquisition.camera_bin=1'])
    config = load_config(path, explicit=explicit)
    assert config.live_camera and config.live.camera_index == 1 and config.live.read_timeout_ms == 250
    assert config.acquisition.camera_bin == 1
    assert config.resolve(config.live.cti_path) == tmp_path / 'producer.cti'
    # --set ...=null clears a file value; a flag that was not given (None in overrides) keeps it
    config = load_config(path, explicit=parse_set_overrides(['live.cti_path=null']), live={'camera_index': None})
    assert config.live.cti_path is None and config.live.camera_index == 1
    with pytest.raises(ValueError, match='unknown key'):
        load_config(path, live={'fps': 10})
    with pytest.raises(ValueError, match='SECTION.KEY=VALUE'):
        parse_set_overrides(['live.read_timeout_ms'])


def test_cti_path_and_index_reach_the_backend(tmp_path, install):
    camera = FakeLineCamera()
    made = install(camera)
    (tmp_path / 'gentl').mkdir()
    (tmp_path / 'gentl' / 'producer.cti').write_bytes(b'')
    config = live_config(tmp_path, camera_backend='genicam')
    config = replace(config, live=LiveConfig(camera_index=3, cti_path='gentl/producer.cti', sensor_pixel_size=12))
    cam, live = open_camera(config)
    try:
        settings = made['settings']
        assert settings.camera_backend == 'genicam' and settings.camera_index == 3
        assert settings.cti_path == str(tmp_path / 'gentl' / 'producer.cti') and settings.sensor_pixel_size == 12
        assert live.path == 'live_genicam' and live.name.startswith('genicam #3')
    finally:
        cam.close()


def test_genicam_producer_is_looked_up_in_the_gentl_path(tmp_path, monkeypatch):
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.mkdir()
    b.mkdir()
    (a / 'other_vendor.cti').write_bytes(b'')
    (b / 'sv_gev_tl_x64.cti').write_bytes(b'')
    monkeypatch.setenv('GENICAM_GENTL64_PATH', os.pathsep.join([str(a), str(b)]))
    config = live_config(tmp_path, camera_backend='genicam')
    assert resolve_cti(config) == str(b / 'sv_gev_tl_x64.cti')
    monkeypatch.setenv('GENICAM_GENTL64_PATH', str(a))
    with pytest.raises(FileNotFoundError, match='set live.cti_path'):
        resolve_cti(config)
    with pytest.raises(FileNotFoundError, match='missing.cti'):
        resolve_cti(replace(config, live=LiveConfig(cti_path='missing.cti')))


def test_live_camera_needs_its_own_wavelength_profile(tmp_path, install):
    camera = FakeLineCamera()
    install(camera)
    with pytest.raises(ValueError, match='no wavelength profile'):
        build_pipeline(live_config(tmp_path, wavelength_profile='auto'))
    assert camera.closed == 1
    folder = tmp_path / 'calibration' / 'wavelength' / 'poa'
    folder.mkdir(parents=True)
    (folder / 'wavelength_bin2.json').write_text(
        json.dumps({'m': 20.0, 'c': 400.0, 'camera_bin': 2, 'frame_shape': [6, 40]}), encoding='utf-8')
    install(FakeLineCamera())
    pipeline = build_pipeline(live_config(tmp_path, wavelength_profile='auto'))
    try:
        assert pipeline.plans[0].provenance['wavelength m,c'][1] == 'calibration:wavelength_bin2.json'
    finally:
        pipeline.stop()


def test_simulate_live_report_and_exit_code(tmp_path, install, capsys):
    import simulate
    path = tmp_path / 'simulation.json'
    path.write_text(json.dumps({
        'source': {'kind': 'camera'},
        'acquisition': {'camera_backend': 'poa', 'camera_bin': 2, 'exposure_ms': 1.0, 'gain': 1.0,
                        'm_nm_per_px_bin1': 10.0, 'c_nm': 400.0, 'resolution_nm': 20.0},
        'corrections': {'enabled': False, 'backend': 'cpu'},
        'processing': {'workers': 1, 'profile_every_s': 0, 'numba_threads': 1},
        'model': {'path': 'none'}, 'live': {'read_timeout_ms': 50}}), encoding='utf-8')
    install(FakeLineCamera(rate_hz=500))
    assert simulate.main(['--config', str(path), '--live', '--seconds', '1', '--json']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['source'] == 'camera poa FakeCam (SN1)' and report['frames_delivered'] > 0
    assert 300 < report['target_fps'] < 700 and report['mm_per_line'] and 'schedule_slips' not in report
    assert report['lines_lost_camera'] == 0 and report['line_clock'] == 'arrival'
    install(FakeLineCamera(timeouts=10 ** 9))          # a camera that never delivers a frame
    assert simulate.main(['--config', str(path), '--seconds', '1', '--json']) == 1
    assert 'no frame was delivered' in capsys.readouterr().err


def test_camera_code_is_identical_to_hsi():
    """The camera backends must stay hsi's code byte for byte (the SDK folders only if installed)."""
    sys.path.insert(0, str(ROOT / 'tools'))
    import sync_hsi
    if not (sync_hsi.DEFAULT_HSI / 'modules' / 'camera').is_dir():
        pytest.skip('hsi project not next to this one')
    assert sync_hsi.compare(sync_hsi.DEFAULT_HSI, only='camera') == []
    sdk_problems, _ = sync_hsi.compare_sdk(sync_hsi.DEFAULT_HSI, only='camera')
    assert sdk_problems == []


# --------------------------------------------------------------------------- GUI ---

def test_gui_shows_the_live_camera(tmp_path, install):
    pytest.importorskip('PyQt5')
    from PyQt5.QtWidgets import QApplication
    from themes import apply_theme
    import gui
    app = QApplication.instance() or QApplication([])
    apply_theme(app, 'dark')
    camera = FakeLineCamera(rate_hz=1000)
    install(camera)
    window = gui.PushbroomScannerMainWindow(live_config(tmp_path))
    try:
        window._gui_timer.stop()                       # refreshed by hand below
        t = time.perf_counter()
        while window._pipeline.snapshot().target_fps == 0 and time.perf_counter() - t < 10:
            time.sleep(0.05)
        for _ in range(12):
            window._refresh_gui_slot()
        assert 'live camera' in window.windowTitle()
        text = window._status_label.text()
        assert 'poa FakeCam (SN1) frame' in text and 'cycle' not in text
        assert not window._fps_spin.isEnabled() and window._fps_spin.value() > 0
        assert window._info_label.text().startswith('live camera poa FakeCam (SN1)')
    finally:
        assert window._pipeline.stop()                 # not close(): that saves the user's window layout
    assert camera.closed == 1
