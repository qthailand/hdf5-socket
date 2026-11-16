# -*- coding: utf-8 -*-
"""Lossless high-rate pushbroom pipeline: camera -> parallel frame workers -> in-order sink.

A pushbroom frame is one line of the belt, so no frame may vanish unnoticed::

    Camera thread ──(frame buffers)──► FrameWorker × N ──(results)──► Aggregator ──► snapshot()
     file playback paced at camera.fps,  dark/hot/flip/Smile/bin          restores line order,     GUI pulls
     or a live camera (source.kind);     + classify (each worker owns     counts EVERY line, fills  at ~30 Hz
     each frame lands in a free buffer   its PreprocessingRuntime)        the belt map, latency

* Source: ``source.kind = "file"`` replays recordings (:meth:`Pipeline._playback_loop`);
  ``"camera"`` reads a live hsi camera backend (:meth:`Pipeline._live_loop`, ``camera_source.py``),
  by default in a camera process of its own (``live.reader``, :meth:`Pipeline._live_process_loop`).
  The workers, aggregator and GUI are the same for both.
* Buffers: ``camera.buffer_frames`` frame slots are allocated once. A line for which no slot
  is free is an *overrun*: with ``overrun='drop'`` it is lost exactly like on a real camera
  (counted, and shown as a magenta column on the belt map); with ``'block'`` the simulated
  camera waits (lossless throughput measurement; a live camera always drops). Frames a live
  camera loses itself (frame-id gaps, SDK counter) are lost lines too (``lines_lost_camera``).
* Timing: file playback has a fixed line clock ``camera.fps`` (no drift) and a frame's sensor
  time is its scheduled time, so latency includes any lag of the simulated camera itself. A live
  camera free-runs at its own rate; sensor time is the camera timestamp where it has one
  (GenICam), else the arrival time (see ``camera_source.CameraClock``).
* Statistics: class pixel counts are accumulated per line in the aggregator, in line order,
  independently of how often the GUI draws.

Qt-free; the GUI and ``simulate.py`` only call :meth:`Pipeline.start`, :meth:`snapshot`,
:meth:`reset_counts` and :meth:`stop`.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from modules.preprocessing import PreprocessingRuntime
from sim_config import STEP_NAMES, apply_reflectance

#: A raw pixel at or above this fraction of the camera's full scale counts as clipped. Below
#: 1.0 on purpose: a camera that averages its hardware bin (POA) reports less than full scale
#: when only some of the binned photosites are full.
CLIP_FRACTION = 0.95

logger = logging.getLogger(__name__)

LOST_CLASS = 255                     # class id drawn for lines lost to overrun
BACKENDS = ('auto', 'cpu_fused', 'cpu', 'cupy_cuda', 'opencv_cuda')
_LATENCY_WINDOW = 4096
_RATE_PERIOD_S = 0.5


# ------------------------------------------------------------------------------ helpers ---

def _effective(plan, overrides: dict) -> dict:
    """Steps a worker applies for ``plan``: the override if given, else the plan's own choice,
    and never a step the plan cannot do (no reference loaded, already binned input)."""
    result = {}
    steps = getattr(plan, 'steps', {})
    for name in STEP_NAMES:
        info = steps.get(name, {'available': False, 'on': False})
        wanted = overrides.get(name)
        if name == 'binning' or wanted is None:
            wanted = info['on']
        result[name] = bool(wanted) and (info['available'] or name == 'binning' and info['on'])
    flat = getattr(plan, 'flatfield_profile', None)
    if result.get('flatfield') and flat is not None:
        result['flatfield'] = (result['dark'] == flat.dark and result['flip'] == flat.flip
                               and result['smile'] == flat.smile
                               and not result['reflectance'])
    return result


class HighResTimer:
    """Windows: 1 ms scheduler tick while the simulation runs (default is 15.6 ms)."""

    def __init__(self) -> None:
        self._ok = False
        if os.name == 'nt':
            try:
                import ctypes
                self._winmm = ctypes.windll.winmm
                self._ok = self._winmm.timeBeginPeriod(1) == 0
            except Exception:
                self._ok = False

    def close(self) -> None:
        if self._ok:
            self._winmm.timeEndPeriod(1)
            self._ok = False


def _wait_until(deadline: float, stop: threading.Event) -> None:
    """Sleep most of the way, then yield (``sleep(0)`` releases the GIL to the workers)."""
    while not stop.is_set():
        remain = deadline - time.perf_counter()
        if remain <= 0:
            return
        time.sleep(remain - 0.0015 if remain > 0.002 else 0)


class StageProfiler:
    """Per-worker stage timings in ms; the interface ``modules.preprocessing`` expects."""

    class _Span:
        __slots__ = ('owner', 'name', 'start')

        def __init__(self, owner, name):
            self.owner, self.name = owner, name

        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *exc):
            self.owner._record(self.name, (time.perf_counter() - self.start) * 1000.0)
            return False

    def __init__(self, detail_every: int = 64) -> None:
        self._lock = threading.Lock()
        self._samples: dict[str, tuple[float, int]] = {}
        self._frames = 0
        self._detail_every = detail_every

    def time(self, name):
        return self._Span(self, name)

    def _record(self, name, milliseconds):
        with self._lock:
            total, count = self._samples.get(name, (0.0, 0))
            self._samples[name] = (total + milliseconds, count + 1)

    def want_detail(self) -> bool:
        """GPU per-stage events only on every N-th frame (they synchronise the device)."""
        self._frames += 1
        return self._frames % self._detail_every == 0

    def take(self) -> dict[str, tuple[float, int]]:
        with self._lock:
            samples, self._samples = self._samples, {}
        return samples


@dataclass(slots=True)
class _Job:
    seq: int            # delivered-frame number (contiguous)
    line: int           # sensor line number (includes lost lines)
    slot: int
    t_sensor: float     # perf_counter time the line was exposed (latency starts here)
    source: int
    index: int
    cycle: int
    t_queued: float     # perf_counter time the line entered the buffers (queue wait starts here)
    model: tuple = None  # (generation, classifier, columns per plan) this line is classified with
    spectrum_generation: int = 0  # processing/calibration version used by this worker


@dataclass
class Snapshot:
    """State copied for one GUI/console refresh (owned by the caller)."""
    state: str                       # starting | running | finished | stopped | error
    error: str | None
    elapsed_s: float
    target_fps: float                # file: the simulated line rate (0 = max); live camera: its measured rate
    camera_fps: float                # lines the camera delivered per second (recent)
    processed_fps: float             # lines classified per second (recent)
    frames_delivered: int
    lines_processed: int
    lines_lost: int                  # overrun: no free buffer when the line was exposed
    schedule_slips: int              # the simulated camera itself fell > 50 ms behind its clock
    buffers_in_use: int
    buffer_frames: int
    latency_ms: dict                 # p50 / p99 / max, sensor time -> classified
    stage_ms: dict                   # mean ms per stage over the last interval (~0.5 s)
    stage_stats: dict                # {backend: {'avg': {stage: ms}, 'peak': {stage: ms}}} since reset_perf
    workers: int
    steps: dict                      # step -> True/False override, or None = as the plan resolved it
    backend: str                     # backend(s) the workers actually run
    requested_backend: str           # what was asked for (config or set_backend)
    backend_note: str                # why the actual backend differs from the request ('' if not)
    counts: np.ndarray               # (256,) pixels per class id since the last reset
    counted_lines: int               # lines in ``counts``
    canvas: np.ndarray               # (slit, canvas_width) class ids, oldest -> newest
    raw_canvas: np.ndarray           # (slit, raw_columns) model-input intensity of the newest lines
                                     # (mean over bands, float32; NaN = lost line), oldest -> newest
    canvas_filled: int
    raw_frame: np.ndarray | None     # newest raw frame (display only)
    position: tuple                  # (source name, frame index, n frames, cycle); live: n frames = 0
    source_kind: str = 'file'        # file | camera
    lines_lost_camera: int = 0       # live: part of lines_lost the camera/link lost (never delivered)
    grab_timeouts: int = 0           # live: reads that returned no frame within live.read_timeout_ms
    line_clock: str = 'schedule'     # sensor time = file: 'schedule'; live: 'camera' timestamps or 'arrival'
    raw_lines: np.ndarray | None = None   # line number of each raw_canvas column (-1 = none yet)
    calibration_lines: int = 0       # lines held by the calibration capture (0 = not capturing)
    model_generation: int = 0        # counts and canvas belong to this model (see Pipeline.set_classifier)
    canvas_lines: np.ndarray | None = None   # line number of each canvas column (-1 = lost / none yet)


# ----------------------------------------------------------------------------- pipeline ---

def build_pipeline(config) -> 'Pipeline':
    """Open the configured input, resolve one acquisition plan per file, load and bind the model.

    With ``source.kind = "camera"`` the input is the live camera (``camera_source.open_camera``):
    it is opened and configured here and handed to the pipeline, which closes it when it stops
    (or here, if building the rest fails). With ``live.reader = "process"`` it is opened and read
    in a camera process of its own (``camera_process.py``).
    """
    from sim_config import ensure_profile_structure
    ensure_profile_structure(config)
    src_cfg = config.source
    camera = None
    if src_cfg.kind == 'camera':
        if config.live.reader == 'process':
            from camera_process import open_camera_process
            camera, live = open_camera_process(config)
        else:
            from camera_source import open_camera
            camera, live = open_camera(config)
        sources = [live]
    else:
        from sources import open_sources
        sources = open_sources(config.resolve(src_cfg.path), src_cfg.pattern, stage=src_cfg.stage,
                               scan_axis=src_cfg.envi_scan_axis, dataset=src_cfg.dataset, cache=src_cfg.cache,
                               cache_dir=config.resolve(src_cfg.cache_dir))
    try:
        return _bind_pipeline(config, sources, camera)
    except BaseException:
        if camera is not None:
            try:
                camera.close()
            except Exception:
                logger.exception("Closing the camera after a failed build failed too")
        raise


def _bind_pipeline(config, sources, camera) -> 'Pipeline':
    from inference_engine import SpeedTestClassifier, load_classifier
    from sim_config import build_plan

    src_cfg = config.source
    if config.model.path in (None, '', 'none'):
        classifier = SpeedTestClassifier()
        logger.warning("model.path = none: %s", classifier.path)
    else:
        classifier = load_classifier(config.resolve(config.model.path), config.model.mlc_reg_covar)
    plans = []
    for src in sources:
        plan = build_plan(config, src)
        plan.columns, binding = classifier.bind(plan.wavelengths, plan.n_columns, config.model.band_tolerance_nm,
                                                config.model.resample)
        if camera is not None:
            plan.provenance['input'] = (f"live camera {src.name} {src.frame_shape} {np.dtype(src.dtype).name}, "
                                        f"read timeout {config.live.read_timeout_ms} ms", 'source.kind = camera')
        else:
            plan.provenance['input storage'] = (
                src.storage + (" - timing includes the simulator's own HDF5 decoding; use frames in RAM or "
                               "the disk cache to measure speed" if src.storage == 'stream' else ''),
                f"cache={src_cfg.cache}")
        if src.storage == 'stream':
            logger.warning("%s is streamed: its decoding competes with the workers, timing is pessimistic", src.name)
        _describe_model(plan, classifier, binding, src.name)
        logger.info("%s: %s %s %s\n%s", src.name, _frames_text(src), src.frame_shape, src.dtype, plan.summary())
        plans.append(plan)
    return Pipeline(config, sources, plans, classifier, camera=camera)


def _describe_model(plan, classifier, binding: str, source_name: str) -> None:
    """Write the model lines of ``plan.provenance`` (band binding, input units against the plan)."""
    plan.provenance['model bands'] = (binding, os.path.basename(classifier.path))
    refl_on = plan.steps.get('reflectance', {}).get('on', False)
    units = classifier.input_units
    if units == 'any':
        plan.provenance['model input'] = ('any (speed-only model)', 'model.path = none')
        return
    plan.provenance['model input'] = (
        f"{units}" + ("" if (units == 'reflectance') == refl_on else
                      f" — WARNING: reflectance step is {'on' if refl_on else 'off'}"),
        'model artifact' if classifier.input_units_declared else 'assumed (artifact has no input_units)')
    if (units == 'reflectance') != refl_on:
        logger.warning("%s: model expects %s but the reflectance step is %s", source_name, units,
                       'on' if refl_on else 'off')


def _frames_text(src) -> str:
    return 'live camera' if src.storage == 'live' else f"{src.n_frames} frames"


class Pipeline:
    """Run one stream: a playlist of recordings, or a live camera.

    Args:
        config: ``SimulationConfig``.
        sources: Playlist from ``sources.open_sources`` (same frame shape/dtype/stage), or the
            one ``camera_source.LiveSource`` of ``camera``.
        plans: One ``AcquisitionPlan`` per source (``sim_config.build_plan``), with ``columns``.
        classifier: ``PushbroomLineClassifier``.
        camera: An opened, not streaming hsi camera backend (``camera_source.open_camera``), or
            ``None`` to play ``sources``. The pipeline owns it from here: its camera thread starts
            the stream, reads every frame and closes the camera when it ends; :meth:`stop` closes
            it too if the pipeline never started.
    """

    def __init__(self, config, sources, plans, classifier, camera=None) -> None:
        if len(plans) != len(sources):
            raise ValueError("one acquisition plan per source is required")
        if camera is not None and len(sources) != 1:
            raise ValueError("a live camera is exactly one source")
        self.config = config
        self.sources = sources
        self.plans = plans
        self.classifier = classifier
        # The model of a line travels with the line (the camera thread stamps it), so
        # set_classifier() changes it at one line: (generation, classifier, columns per plan).
        self._model = (0, classifier, [plan.columns for plan in plans])
        self._drawn_generation = 0               # generation of the lines in counts / canvas (lock)
        self.camera = camera
        cam, disp = config.camera, config.display
        # A live camera free-runs; its measured rate takes the place of the simulated line clock.
        self._fps = 0.0 if camera is not None else float(cam.fps)
        if camera is not None and cam.overrun == 'block':
            logger.warning("camera.overrun = block applies to file playback only: a live camera cannot wait, "
                           "a line with no free buffer is lost")
        self._fps_generation = 0                 # bumped by set_fps; the camera re-reads the rate
        self._n_workers = int(config.processing.workers)
        shape = sources[0].frame_shape
        self._slit = shape[0]

        self._n_buffers = int(cam.buffer_frames)
        # live.reader = "process": the buffers are shared memory the camera process writes into,
        # and a slot a worker puts back goes to that process (camera_process.SlotReturn).
        self._process_reader = bool(getattr(camera, 'separate_process', False))
        self._shared = None
        if self._process_reader:
            from camera_process import SharedFrames, SlotReturn
            self._shared = SharedFrames(cam.buffer_frames, shape, sources[0].dtype)
            self._buffers = self._shared.frames
            self._free = SlotReturn(camera, cam.buffer_frames)
        else:
            self._buffers = np.empty((cam.buffer_frames, *shape), dtype=sources[0].dtype)
            self._free = queue.SimpleQueue()
            for slot in range(cam.buffer_frames):
                self._free.put(slot)
        self._work: queue.SimpleQueue = queue.SimpleQueue()
        self._results: queue.SimpleQueue = queue.SimpleQueue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._profilers = [StageProfiler() for _ in range(self._n_workers)]
        self._runtimes: list[PreprocessingRuntime | None] = [None] * self._n_workers
        # Step switches (GUI): replaced as a whole, read once per frame by the workers.
        self._steps: dict = {name: None for name in STEP_NAMES}
        # Backend switch: workers compare the generation at each frame boundary.
        self._requested_backend = self.plans[0].correction_backend
        self._backend_generation = 0
        self._timer: HighResTimer | None = None

        # camera thread state
        self._delivered = 0
        self._lost = 0
        self._lost_camera = 0                    # live: frames the camera never delivered (in _lost too)
        self._grab_timeouts = 0                  # live: reads without a frame
        self._line_clock = 'schedule' if camera is None else 'arrival'
        self._slips = 0
        self._camera_fps = 0.0
        self._camera_done = False
        self._workers_released = False           # stop() sent the workers' end marker itself

        # aggregator state (guarded by _lock)
        self._counts = np.zeros(256, np.int64)
        self._counted_lines = 0
        self._processed = 0
        self._processed_fps = 0.0
        self._latency = np.zeros(_LATENCY_WINDOW, np.float32)
        self._latency_n = 0
        self._stride = disp.canvas_line_stride
        self._canvas = np.zeros((self._slit, disp.canvas_width), np.uint8)
        # Same ring, same columns: what the model saw (before classification), for the raw zone.
        self._raw_canvas = np.full((self._slit, disp.canvas_width), np.nan, np.float32)
        self._canvas_lines = np.full(disp.canvas_width, -1, np.int64)   # line number of each column
        # Reflectance calibration (GUI "Reflectance..."): the newest binned lines before the
        # reflectance step, kept only while a capture is open (see start_calibration_capture).
        self._cal_ring: np.ndarray | None = None
        self._cal_ids: np.ndarray | None = None
        self._cal_pos = 0
        self._cal_clip: np.ndarray | None = None       # (lines, slit) clipped raw pixels per slit row
        self._cal_clip_level: float | None = None      # raw counts; None: unknown full scale / binned input
        # Spectrum preview (GUI: click on the belt map): the newest lines as the model received
        # them, by line number, kept only while switched on (see start_spectrum_history).
        self._spec_ring: np.ndarray | None = None
        self._spec_ids: np.ndarray | None = None
        self._spectrum_generation = 0
        self._canvas_col = 0
        self._canvas_filled = 0
        self._canvas_line = 0
        # Per-line class counts of the newest lines (ring by line number): snapshot() can leave out
        # the lines still in the GUI's raw zone, so the statistics follow the classified zone.
        recent = int(np.ceil(disp.canvas_width * disp.canvas_line_stride * disp.raw_zone_fraction)) + 1
        self._recent_counts = np.zeros((recent, 256), np.int32)
        self._recent_counted = np.zeros(recent, bool)
        self._position = (sources[0].name, 0, sources[0].n_frames, 0)
        self._stage_ms: dict[str, float] = {}
        self._stage_totals: dict[str, dict[str, list]] = {}   # backend -> stage -> [ms sum, n]
        self._stage_peak: dict[str, dict[str, float]] = {}    # backend -> stage -> max interval mean

        self._display_interval = 1.0 / max(1.0, disp.gui_hz)
        self._display_t = 0.0
        self._raw_frame: np.ndarray | None = None
        self._state = 'starting'
        self._error: str | None = None
        self._t0 = 0.0
        self._workers_done = 0
        self._ready = [threading.Event() for _ in range(self._n_workers)]

    # -------------------------------------------------------------------------- control ---

    def start(self) -> None:
        self._timer = HighResTimer()
        self._t0 = time.perf_counter()
        targets = [('camera', self._camera_loop, ())]
        targets += [(f'worker-{k}', self._worker_loop, (k,)) for k in range(self._n_workers)]
        targets += [('aggregator', self._aggregate_loop, ())]
        for name, target, args in targets:
            thread = threading.Thread(target=target, args=args, name=name, daemon=True)
            self._threads.append(thread)
            thread.start()
        if self.camera is not None:
            logger.info("Pipeline started: live camera %s (free-running, read in its own %s), %d workers, "
                        "%d buffers", self.sources[0].name, 'process' if self._process_reader else 'thread',
                        self._n_workers, self._n_buffers)
        else:
            logger.info("Pipeline started: %s lines/s target, %d workers, %d buffers, overrun=%s",
                        'max' if self._fps == 0 else f"{self._fps:g}", self._n_workers,
                        self._n_buffers, self.config.camera.overrun)

    def stop(self, timeout: float = 3.0) -> bool:
        """Stop all threads; returns False if one is still busy after ``timeout``.

        A live camera is closed by its own thread on the way out; if the pipeline never started,
        it is closed here. ``timeout`` is at least one ``live.read_timeout_ms`` read plus 1 s, so a
        read in progress can end. A camera thread still blocked in the SDK after that keeps the
        camera (closing it from this thread could crash the SDK): call ``stop`` again. A camera
        process never blocks the stop: one stuck in the SDK is terminated (``close_timeout_s``).
        """
        if self._process_reader:
            timeout = max(timeout, self.camera.close_timeout_s + 5.0)
        elif self.camera is not None:
            timeout = max(timeout, self.config.live.read_timeout_ms / 1000.0 + 1.0)
        self._stop.set()
        deadline = time.perf_counter() + timeout
        for thread in self._threads:
            thread.join(max(0.0, deadline - time.perf_counter()))
            if thread.name == 'camera' and thread.is_alive() and not self._workers_released:
                # The camera thread is stuck in a read and has not sent the workers their end
                # marker yet: send it here, so that only the camera thread is left running.
                self._workers_released = True
                for _ in range(self._n_workers):
                    self._work.put(None)
                deadline = max(deadline, time.perf_counter() + 1.0)
        if self._timer is not None:
            self._timer.close()
            self._timer = None
        if self.camera is not None and not any(t.name == 'camera' and t.is_alive() for t in self._threads):
            self._close_camera()
        if self._shared is not None and not any(t.is_alive() for t in self._threads):
            self._buffers = None                 # the workers are done with the shared frames
            self._shared.release()
            self._shared = None
        if self._state in ('running', 'starting'):
            self._state = 'stopped'
        return not any(t.is_alive() for t in self._threads)

    def busy_threads(self) -> list[str]:
        """Names of the pipeline threads still running (after :meth:`stop`: the ones that did not end)."""
        return [t.name for t in self._threads if t.is_alive()]

    def _close_camera(self) -> None:
        """Stop the stream and release the live camera (idempotent, never raises)."""
        try:
            self.camera.close()
        except Exception:
            logger.exception("Closing the camera failed")

    def reset_perf(self) -> None:
        """Restart the per-stage averages and peaks (e.g. after changing a setting)."""
        with self._lock:
            self._stage_totals.clear()
            self._stage_peak.clear()

    def _held(self, lines: int) -> tuple[np.ndarray, int]:
        """(class counts, counted lines) of the newest ``lines`` belt lines. Lock held."""
        ring = len(self._recent_counted)
        hold = min(int(lines), ring, self._canvas_line)
        if hold <= 0:
            return np.zeros(256, np.int64), 0
        held = (self._canvas_line - 1 - np.arange(hold)) % ring
        return self._recent_counts[held].sum(axis=0, dtype=np.int64), int(self._recent_counted[held].sum())

    def reset_counts(self, pending_lines: int = 0) -> None:
        """Start a new batch: class counts restart from zero (throughput counters keep running).

        The newest ``pending_lines`` lines (the GUI's raw zone: not shown as counted yet) belong
        to the new batch; they are counted as they cross into the classified zone.
        """
        with self._lock:
            counts, lines = self._held(pending_lines)
            self._counts[:] = counts
            self._counted_lines = lines

    def set_classifier(self, classifier, tolerance_nm: float | None = None) -> int:
        """Classify with another model from the next camera line on, while running.

        The model is bound to every plan first (``classifier.bind``); if it does not fit one of
        them, ``ValueError`` is raised and nothing changes. Lines already in the buffers are still
        classified with the previous model. At the first line of the new model the class counts
        restart and the belt map's class history is cleared (class ids of two models do not mean
        the same): ``Snapshot.model_generation`` then takes the returned value, which is when a
        display should change its class names and colours.

        Returns:
            The generation of the new model.
        """
        if tolerance_nm is None:
            tolerance_nm = self.config.model.band_tolerance_nm
        bound = [classifier.bind(plan.wavelengths, plan.n_columns, tolerance_nm, self.config.model.resample)
                 for plan in self.plans]
        with self._lock:
            generation = self._model[0] + 1
            for plan, src, (columns, binding) in zip(self.plans, self.sources, bound):
                plan.columns = columns
                _describe_model(plan, classifier, binding, src.name)
            self.classifier = classifier
            self._model = (generation, classifier, [columns for columns, _ in bound])   # one reference swap
        logger.info("Model -> %s (generation %d)", classifier.path, generation)
        return generation

    def set_backend(self, name: str) -> None:
        """Switch the correction backend (``auto``, ``cpu``, ``cupy_cuda``, ``opencv_cuda``) while running.

        Each worker builds a new hsi ``PreprocessingRuntime`` in its own thread (CUDA contexts
        belong to the thread), warms it up on a blank frame and then continues with the next
        line; no line is skipped. The runtime applies hsi's rules: an unavailable or failing
        GPU backend falls back to the next candidate / CPU, see ``Snapshot.backend_note``.
        """
        if name not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {name!r}")
        with self._lock:
            if name == self._requested_backend:
                return
            self._requested_backend = name
            self._backend_generation += 1
        logger.info("Backend switch requested: %s", name)

    def set_fps(self, fps: float) -> None:
        """Change the simulated line rate while running (``0`` = as fast as the chain takes:
        the camera then waits for a free buffer, so it measures the lossless maximum).

        The camera picks it up at its next line and restarts its clock from there (no burst of
        "late" lines); buffers, counts and the line order are kept.
        """
        if self.camera is not None:
            raise RuntimeError("a live camera runs at its own line rate (exposure and readout); "
                               "set_fps applies to file playback")
        fps = float(fps)
        if fps < 0:
            raise ValueError("fps must be >= 0")
        if fps == self._fps:
            return
        self._fps = fps
        self._fps_generation += 1
        logger.info("Camera line rate -> %s", 'max' if fps == 0 else f"{fps:g} lines/s")

    def set_steps(self, **switches) -> None:
        """Turn processing steps on/off while running: ``dark`` (dark + hot pixel), ``flip``, ``smile``,
        ``reflectance``.

        ``True``/``False`` overrides the plan, ``None`` returns to what the plan resolved from
        file/config. A step that is not available in a plan (no reference loaded, binned input)
        stays off there whatever is asked; binning cannot be switched off. The change applies
        from the next line every worker takes; the hsi algorithms themselves are unchanged.
        """
        unknown = set(switches) - set(STEP_NAMES)
        if unknown or 'binning' in switches:
            raise ValueError(f"switchable steps are dark, flip, smile, flatfield, reflectance; got {sorted(switches)}")
        with self._lock:
            if switches.get('flatfield') is True and switches.get('reflectance') is True:
                raise ValueError('Use integrated flat-field reflectance factors instead of applying Flat-field twice')
            steps = dict(self._steps)
            steps.update(switches)
            if switches.get('flatfield') is True:
                steps['reflectance'] = False
            elif switches.get('reflectance') is True:
                steps['flatfield'] = False
            changed = steps != self._steps
            self._steps = steps
            if changed:
                self._invalidate_spectra_locked()
        logger.info("Processing steps: %s", ', '.join(f"{k}={'plan' if v is None else v}" for k, v in steps.items()))

    def effective_steps(self, plan_index: int = 0) -> dict:
        """``{step: bool}`` that the workers currently apply for ``plans[plan_index]``."""
        return _effective(self.plans[plan_index], self._steps)

    def start_calibration_capture(self, max_lines: int = 256) -> None:
        """Keep the newest ``max_lines`` binned lines *before* the reflectance step.

        For a reflectance calibration drawn on the belt map: :meth:`calibration_capture` returns
        them by line number (the numbers of ``Snapshot.raw_lines``). Costs one copy per line and
        ``max_lines x slit x bands x 4`` bytes, only until :meth:`stop_calibration_capture`.
        """
        plan = self.plans[0]
        shape = (int(max_lines), self._slit, plan.n_columns)
        meta = self.sources[0].metadata or {}
        full_scale = meta.get('full_scale') or meta.get('saturation_level')
        with self._lock:
            # Clipping is only visible in the raw frame: dark, Smile and the wavelength binning
            # all mix a full pixel with its neighbours.
            self._cal_clip_level = (CLIP_FRACTION * float(full_scale)
                                    if full_scale and plan.stage == 'raw' else None)
            self._cal_clip = np.zeros(shape[:2], np.uint16)
            self._cal_ring = np.zeros(shape, np.float32)
            self._cal_ids = np.full(int(max_lines), -1, np.int64)
            self._cal_pos = 0
        logger.info("Calibration capture: newest %d lines (%.0f MB)", max_lines, self._cal_ring.nbytes / 1e6)

    def calibration_capture(self, first_line: int | None = None, last_line: int | None = None,
                            with_clipped: bool = False):
        """``(line numbers, binned lines (n, slit, bands) float32)`` held now, oldest first.

        ``first_line`` / ``last_line`` (inclusive) select a range; nothing captured -> empty arrays.
        ``with_clipped`` adds a third item: ``(n, slit) uint16``, the raw pixels of each slit row at
        or above ``CLIP_FRACTION`` of the camera's full scale (counted before dark, Smile and
        binning), or ``None`` when the full scale is unknown or the input is already binned.
        """
        with self._lock:
            if self._cal_ring is None:
                empty = np.empty(0, np.int64), np.empty((0, self._slit, self.plans[0].n_columns), np.float32)
                return (*empty, None) if with_clipped else empty
            ids = self._cal_ids.copy()
            keep = ids >= 0
            if first_line is not None:
                keep &= ids >= int(first_line)
            if last_line is not None:
                keep &= ids <= int(last_line)
            index = np.flatnonzero(keep)
            index = index[np.argsort(ids[index])]
            if with_clipped:
                clipped = self._cal_clip[index].copy() if self._cal_clip_level is not None else None
                return ids[index], self._cal_ring[index].copy(), clipped
            return ids[index], self._cal_ring[index].copy()

    def calibration_count(self) -> int:
        """Lines the calibration capture holds now (0 = not capturing)."""
        ids = self._cal_ids
        return 0 if ids is None else int((ids >= 0).sum())

    def stop_calibration_capture(self) -> None:
        with self._lock:
            self._cal_ring = self._cal_ids = self._cal_clip = None

    def start_spectrum_history(self, max_mb: float | None = None) -> int:
        """Keep the newest lines as the model receives them (after every step that is on).

        For the GUI's spectrum preview: :meth:`spectrum_at` returns one pixel's spectrum by line
        number (the numbers of ``Snapshot.canvas_lines``). Costs one copy per line and at most
        ``max_mb`` MB (default ``display.spectrum_history_mb``), only until
        :meth:`stop_spectrum_history`. Returns the number of lines kept.
        """
        budget = self.config.display.spectrum_history_mb if max_mb is None else max_mb
        per_line = self._slit * self.plans[0].n_columns * 4
        lines = int(max(1, min(budget * 1e6 // per_line, self._canvas.shape[1] * self._stride)))
        with self._lock:
            self._spec_ids = np.full(lines, -1, np.int64)
            self._spec_ring = np.zeros((lines, self._slit, self.plans[0].n_columns), np.float32)
            self._invalidate_spectra_locked()
        logger.info("Spectrum history: newest %d lines (%.0f MB)", lines, lines * per_line / 1e6)
        return lines

    def stop_spectrum_history(self) -> None:
        with self._lock:
            self._spec_ring = self._spec_ids = None

    def _invalidate_spectra_locked(self) -> None:
        """Old inputs and in-flight results must not be shown as a new calibration."""
        self._spectrum_generation += 1
        if self._spec_ids is not None:
            self._spec_ids.fill(-1)

    @property
    def spectrum_lines(self) -> int:
        """Lines the spectrum history can hold (0 = off)."""
        ids = self._spec_ids
        return 0 if ids is None else len(ids)

    def spectrum_at(self, line: int, row: int) -> np.ndarray | None:
        """Model-input spectrum (bands,) float32 of slit pixel ``row`` in ``line``; ``None`` when
        that line is not kept (history off, lost line, or older than the history)."""
        with self._lock:
            if self._spec_ring is None or line < 0 or not 0 <= row < self._slit:
                return None
            k = int(line) % len(self._spec_ids)
            if self._spec_ids[k] != line:
                return None
            return self._spec_ring[k, int(row)].copy()

    def spectrum_snapshot(self, max_mb: float = 128.0) -> tuple[np.ndarray, np.ndarray]:
        """A copy of the newest lines of the spectrum history, oldest first.

        Returns:
            ``(line numbers (n,), spectra (n, slit, bands) float32)``; at most ``max_mb`` MB, the
            newest lines. Empty (n = 0) while the history is off or has no line yet.
        """
        with self._lock:
            if self._spec_ring is None:
                return np.zeros(0, np.int64), np.zeros((0, self._slit, self.plans[0].n_columns), np.float32)
            ids = self._spec_ids
            order = np.argsort(ids, kind='stable')
            order = order[ids[order] >= 0]
            per_line = self._spec_ring[0].nbytes
            order = order[-int(max(1, max_mb * 1e6 // per_line)):]
            return ids[order].copy(), self._spec_ring[order]       # fancy index: a copy

    def set_reflectance_factors(self, factors: np.ndarray, label: str, plan_index: int = 0) -> None:
        """Use new reflectance factors from the next line on and switch the step on."""
        plan = self.plans[plan_index]
        factors = np.asarray(factors, np.float32)
        if factors.shape[-1] != plan.n_columns or (factors.ndim == 2 and factors.shape[0] != self._slit):
            raise ValueError(f"factors {factors.shape} do not fit ({self._slit}, {plan.n_columns})")
        plan.reflectance_factors = factors          # one reference swap: a worker sees old or new
        plan.reflectance_placeholder = False
        plan.steps['reflectance'] = {'available': True, 'on': True, 'why': label}
        plan.provenance['reflectance'] = (label, 'drawn on RAW')
        self.set_steps(reflectance=True)
        with self._lock:
            self._invalidate_spectra_locked()

    def _backend_note(self) -> str:
        if self.plans[0].stage == 'binned':
            return ''
        requested = self._requested_backend
        actual = {rt.backend_name for rt in self._runtimes if rt is not None}
        if requested == 'auto' or actual == {requested} or not actual:
            return ''
        reasons = {rt.fallback_reason for rt in self._runtimes if rt is not None and rt.fallback_reason}
        if not reasons and requested != 'cpu':
            from modules.compute_runtime import get_compute_capabilities
            capability = getattr(get_compute_capabilities(), requested, None)
            if capability is not None and not capability.available:
                reasons = {f"{requested} unavailable: {capability.reason}"}
        return '; '.join(sorted(reasons)) or 'switching'

    @property
    def backend_name(self) -> str:
        names = {rt.backend_name for rt in self._runtimes if rt is not None}
        if self.plans[0].stage == 'binned':
            return 'none (binned input)'
        return '/'.join(sorted(names)) or 'uninitialized'

    def snapshot(self, canvas_columns: int | None = None, raw_columns: int = 0,
                 step: tuple[int, int] = (1, 1), count_delay_lines: int = 0) -> Snapshot:
        """State for one refresh.

        ``count_delay_lines`` leaves the newest N belt lines out of ``counts`` / ``counted_lines``:
        the GUI passes its raw zone, so a line is counted when it crosses into the classified zone.

        ``canvas_columns`` limits the belt map to the newest N columns; ``raw_columns`` returns
        the model-input intensity of the newest M of them (the GUI's pre-classification zone).
        ``step`` = (rows, columns) returns every k-th row / column only (counted from the newest
        column): the GUI asks for its screen resolution instead of copying the full map.
        """
        with self._lock:
            n = min(self._latency_n, _LATENCY_WINDOW)
            lat = self._latency[:n]
            latency = ({'p50': float(np.percentile(lat, 50)), 'p99': float(np.percentile(lat, 99)),
                        'max': float(lat.max())} if n else {'p50': 0.0, 'p99': 0.0, 'max': 0.0})
            width = self._canvas.shape[1]
            n = self._canvas_filled if canvas_columns is None else min(self._canvas_filled, int(canvas_columns))
            # the newest n columns, oldest first (the canvas is a ring; copy only what is asked)
            columns = (self._canvas_col - n + np.arange(n)) % width if self._canvas_filled == width \
                else np.arange(self._canvas_filled - n, self._canvas_filled)
            rs, cs = max(1, int(step[0])), max(1, int(step[1]))
            offsets = np.arange(n - 1, -1, -cs)[::-1]              # from the oldest; the newest is kept
            canvas = self._canvas[::rs, columns[offsets]]
            raw_columns_idx = columns[offsets[offsets >= n - int(raw_columns)]]
            raw_canvas = self._raw_canvas[::rs, raw_columns_idx]
            raw_lines = self._canvas_lines[raw_columns_idx]
            counts, counted_lines = self._counts.copy(), self._counted_lines
            if count_delay_lines > 0:       # leave out the newest lines (still in the raw zone)
                held_counts, held_lines = self._held(count_delay_lines)
                # never below zero when the zone was resized after a reset
                counts = np.maximum(counts - held_counts, 0)
                counted_lines = max(counted_lines - held_lines, 0)
            snap = Snapshot(
                state=self._state, error=self._error, elapsed_s=time.perf_counter() - self._t0,
                target_fps=self._fps, camera_fps=self._camera_fps, processed_fps=self._processed_fps,
                frames_delivered=self._delivered, lines_processed=self._processed, lines_lost=self._lost,
                schedule_slips=self._slips,
                buffers_in_use=self._n_buffers - self._free.qsize(), buffer_frames=self._n_buffers,
                latency_ms=latency, stage_ms=dict(self._stage_ms), backend=self.backend_name,
                stage_stats={b: {'avg': {n: t / c for n, (t, c) in st.items() if c},
                                 'peak': dict(self._stage_peak.get(b, {}))}
                             for b, st in self._stage_totals.items()},
                workers=self._n_workers, steps=dict(self._steps),
                requested_backend=self._requested_backend, backend_note=self._backend_note(),
                counts=counts, counted_lines=counted_lines,
                canvas=canvas, raw_canvas=raw_canvas, canvas_filled=self._canvas_filled,
                raw_frame=self._raw_frame, position=self._position,
                source_kind='file' if self.camera is None else 'camera',
                lines_lost_camera=self._lost_camera, grab_timeouts=self._grab_timeouts,
                line_clock=self._line_clock, raw_lines=raw_lines,
                canvas_lines=self._canvas_lines[columns[offsets]],
                calibration_lines=0 if self._cal_ids is None else int((self._cal_ids >= 0).sum()),
                model_generation=self._drawn_generation)
        return snap

    # --------------------------------------------------------------------------- camera ---

    def _camera_loop(self) -> None:
        if self._process_reader:
            self._live_process_loop()
        elif self.camera is not None:
            self._live_loop()
        else:
            self._playback_loop()

    def _wait_workers_ready(self) -> bool:
        """Wait until every worker has compiled/selected its kernels; False if stopped first.

        The line clock (or the live camera's stream) starts only then: a real camera is started
        after the processing chain is ready, too.
        """
        for ready in self._ready:
            while not ready.wait(0.05):
                if self._stop.is_set():
                    return False
        logger.info("Workers ready after %.2f s; camera started", time.perf_counter() - self._t0)
        return True

    def _playback_loop(self) -> None:
        cam = self.config.camera
        generation = self._fps_generation
        period = 1.0 / self._fps if self._fps > 0 else 0.0
        drop = cam.overrun == 'drop'
        line = seq = cycle = 0
        try:
            if not self._wait_workers_ready():
                return
            self._t0 = time.perf_counter()
            self._state = 'running'
            rate_t, rate_n = time.perf_counter(), 0
            next_t = time.perf_counter()
            while not self._stop.is_set():
                for src_i, src in enumerate(self.sources):
                    frames = src.frames
                    for i in range(src.n_frames):
                        if self._stop.is_set():
                            return
                        if generation != self._fps_generation:         # line rate changed (GUI)
                            generation = self._fps_generation
                            period = 1.0 / self._fps if self._fps > 0 else 0.0
                            next_t = time.perf_counter()
                            rate_t, rate_n = next_t, 0
                        if period:
                            _wait_until(next_t, self._stop)
                            t_sensor = next_t
                            next_t += period
                            if time.perf_counter() - next_t > 0.05:     # simulator itself fell behind
                                self._slips += 1
                                next_t = time.perf_counter()
                        else:
                            t_sensor = time.perf_counter()
                        try:
                            slot = self._free.get_nowait()
                        except queue.Empty:
                            # Overrun loses a line only with a line clock; without one (max) the
                            # camera simply waits: max = the chain's lossless maximum.
                            if drop and period:
                                self._lost += 1
                                line += 1
                                continue
                            slot = self._wait_slot()
                            if slot is None:
                                return
                        np.copyto(self._buffers[slot], frames[i])
                        self._work.put(_Job(seq, line, slot, t_sensor, src_i, i, cycle, t_sensor, self._model))
                        seq += 1
                        line += 1
                        self._delivered = seq
                        rate_n += 1
                        now = time.perf_counter()
                        if now - rate_t >= _RATE_PERIOD_S:
                            self._camera_fps = rate_n / (now - rate_t)
                            rate_t, rate_n = now, 0
                cycle += 1
                if not cam.loop:
                    logger.info("Camera: end of playlist after %d frames", seq)
                    return
        except Exception as exc:
            self._fail('camera', exc)
        finally:
            self._camera_done = True
            self._camera_fps = 0.0
            for _ in range(self._n_workers):
                self._work.put(None)

    def _live_loop(self) -> None:
        """Read the live camera into the frame buffers (``source.kind = "camera"``).

        This thread owns the camera: it starts the stream once the workers are ready, reads
        every frame and closes the camera at the end (stop, or an error). The camera free-runs;
        before each read the loop takes a free buffer so the backend can write the frame straight
        into it (``read_frame_raw_into``: GenICam converts into it, the other backends return a
        new array that is copied). With no free buffer the frame is still read - into a scratch
        buffer - and, if no buffer came free during the read either, dropped as an overrun: a
        camera cannot wait, and leaving frames in the SDK queue would only make the SDK lose them
        uncounted. Frames the camera or link lost (``camera_source.CameraFrameGaps``) are lost
        lines too. A read that returns nothing (timeout, SDK error) is retried; the stop event
        aborts a blocked read.

        ``target_fps`` becomes the rate at which the camera delivers frames (overruns included,
        frames it lost itself not, so an outage does not end in a spike), ``camera_fps`` the rate
        of frames that reached the buffers. Both fall to 0 while the camera delivers nothing.
        """
        from camera_source import CameraClock, CameraFrameGaps
        camera = self.camera
        timeout_ms = int(self.config.live.read_timeout_ms)
        shape = tuple(self._buffers.shape[1:])
        scratch = np.empty(shape, self._buffers.dtype)
        target = scratch
        line = seq = received = 0
        rate_t, rate_n, rate_received = time.perf_counter(), 0, 0

        def provide(frame_shape):
            # called by the backend inside its read; must not block (GenICam holds a buffer meanwhile)
            return target if tuple(frame_shape) == shape else None

        def update_rates(now):
            nonlocal rate_t, rate_n, rate_received
            if now - rate_t >= _RATE_PERIOD_S:
                self._camera_fps = rate_n / (now - rate_t)
                self._fps = (received - rate_received) / (now - rate_t)
                rate_t, rate_n, rate_received = now, 0, received

        try:
            if not self._wait_workers_ready():
                return
            camera.set_stop_event(self._stop)          # set() makes a blocked read return None
            try:
                camera.start_stream()
            except Exception:
                if self._stop.is_set():                # e.g. GenICam start retries cut short by a stop
                    logger.info("Camera start abandoned: stop requested")
                    return
                raise
            gaps, clock = CameraFrameGaps(camera), CameraClock(camera)
            gaps.start()
            self._t0 = time.perf_counter()
            self._state = 'running'
            rate_t = time.perf_counter()
            while not self._stop.is_set():
                try:
                    slot = self._free.get_nowait()
                except queue.Empty:
                    slot = None
                target = scratch if slot is None else self._buffers[slot]
                frame = camera.read_frame_raw_into(provide, timeout_ms=timeout_ms)
                t_arrival = time.perf_counter()
                if frame is None:
                    if slot is not None:
                        self._free.put(slot)
                    if not self._stop.is_set():
                        self._grab_timeouts += 1
                    update_rates(t_arrival)
                    continue
                if tuple(frame.shape) != shape:
                    raise RuntimeError(f"camera frame {tuple(frame.shape)} differs from the {shape} the "
                                       "calibration was resolved for (bin or ROI changed?)")
                received += 1
                missed = gaps.missed()
                if missed:
                    self._lost += missed
                    self._lost_camera += missed
                    line += missed
                t_sensor = clock.sensor_time(t_arrival)
                self._line_clock = clock.source
                if slot is None:                         # a buffer may have come free during the read
                    try:
                        slot = self._free.get_nowait()
                    except queue.Empty:
                        pass
                    else:
                        target = self._buffers[slot]
                if slot is None:                         # overrun: no free buffer for this line
                    self._lost += 1
                    line += 1
                else:
                    if frame is not target:
                        np.copyto(target, frame)
                    self._work.put(_Job(seq, line, slot, t_sensor, 0, line, 0, t_arrival, self._model))
                    seq += 1
                    line += 1
                    self._delivered = seq
                    rate_n += 1
                update_rates(time.perf_counter())
        except Exception as exc:
            self._fail('camera', exc)
        finally:
            self._camera_done = True
            self._camera_fps = 0.0
            self._close_camera()                         # stops the stream first
            logger.info("Camera closed after %d frames (%d lost in the camera, %d read timeouts)",
                        seq, self._lost_camera, self._grab_timeouts)
            for _ in range(self._n_workers):
                self._work.put(None)

    def _live_process_loop(self) -> None:
        """Take the frames a camera process wrote into the shared buffers (``live.reader = "process"``).

        The camera process (``camera_process.py``) reads the camera with :meth:`_live_loop`'s
        rules - a free buffer before each read, overruns and camera gaps counted as lost lines -
        and announces each frame with a job record. This thread turns the records into jobs and
        copies the process's counters into the statistics. Nothing here can slow the camera: if
        this process is busy, frames wait in the buffers, and only when all are full does the
        camera process count overruns.
        """
        camera = self.camera
        rate_t, rate_n, rate_received = time.perf_counter(), 0, 0
        received = 0

        def update_counters(now, rates=True):
            nonlocal rate_t, rate_n, rate_received, received
            c = camera.counters()
            received = c['received']
            self._lost, self._lost_camera = c['lost'], c['lost_camera']
            self._grab_timeouts, self._line_clock = c['timeouts'], c['line_clock']
            if rates and now - rate_t >= _RATE_PERIOD_S:
                self._camera_fps = rate_n / (now - rate_t)
                self._fps = (received - rate_received) / (now - rate_t)
                rate_t, rate_n, rate_received = now, 0, received

        try:
            if not self._wait_workers_ready():
                return
            if not camera.start_stream(self._shared, self._stop):
                logger.info("Camera start abandoned: stop requested")
                return
            self._t0 = time.perf_counter()
            self._state = 'running'
            rate_t = next_check = time.perf_counter()
            while not self._stop.is_set():
                job = camera.next_job(timeout=0.05)     # waits (gives the GIL away) only when idle
                now = time.perf_counter()
                if job is not None:
                    seq, line, slot, t_sensor, t_arrival = job
                    self._work.put(_Job(seq, line, slot, t_sensor, 0, line, 0, t_arrival, self._model))
                    self._delivered = seq + 1
                    rate_n += 1
                if job is None or now >= next_check:
                    next_check = now + 0.1
                    camera.check()                       # raises the camera process's error
                    update_counters(now)
        except Exception as exc:
            self._fail('camera', exc)
        finally:
            self._camera_done = True
            self._close_camera()                         # ends the camera process
            update_counters(time.perf_counter(), rates=False)
            self._camera_fps = 0.0
            logger.info("Camera process ended after %d frames (%d lost in the camera, %d read timeouts)",
                        received, self._lost_camera, self._grab_timeouts)
            for _ in range(self._n_workers):
                self._work.put(None)

    def _wait_slot(self):
        while not self._stop.is_set():
            try:
                return self._free.get(timeout=0.05)
            except queue.Empty:
                continue
        return None

    # -------------------------------------------------------------------------- workers ---

    def _worker_loop(self, k: int) -> None:
        prof = self._profilers[k]
        runtime = None
        raw = self.plans[0].stage == 'raw'
        try:
            generation = self._backend_generation
            if raw:
                runtime = self._new_runtime(k, self._requested_backend)
            self._warm_up(runtime, prof)
            self._ready[k].set()
            while True:
                job = self._work.get()
                if job is None:
                    return
                if raw and generation != self._backend_generation:      # switch at a frame boundary
                    generation = self._backend_generation
                    runtime = self._new_runtime(k, self._requested_backend)
                    self._warm_up(runtime, prof)
                    logger.info("worker-%d now on %s (requested %s)", k, runtime.backend_name,
                                self._requested_backend)
                plan = self.plans[job.source]
                frame = self._buffers[job.slot]
                # time the line waited in the buffers before a worker took it (for a live camera
                # with timestamps the latency also has exposure and transfer; the queue wait not)
                prof._record('queue', (time.perf_counter() - job.t_queued) * 1000.0)
                with prof.time('total'):
                    job.spectrum_generation = self._spectrum_generation
                    on = _effective(plan, self._steps)                # one consistent set per frame
                    clip_level, clipped = self._cal_clip_level, None
                    if clip_level is not None and self._cal_ring is not None and plan.stage == 'raw':
                        # raw pixels at full scale per slit row, in the row order after the flip
                        clipped = np.count_nonzero(frame >= clip_level, axis=1)
                        if on['flip']:
                            clipped = clipped[::-1]
                    if plan.stage == 'raw':
                        binned, _, _ = runtime.process_binned(
                            frame, plan.dark_processor(on['dark']), plan.binner,
                            calibration=plan.smile_model if on['smile'] else None,
                            inverted=on['flip'], profiler=prof)
                    else:
                        binned = frame
                    pre = binned.astype(np.float32, copy=True) if self._cal_ring is not None else None
                    flat = getattr(plan, 'flatfield_profile', None)
                    if on['flatfield'] and flat is not None:
                        with prof.time('flatfield'):
                            binned = np.divide(binned, flat.values, dtype=np.float32)
                    if on['reflectance']:                             # last step before the model
                        with prof.time('reflectance'):
                            binned = apply_reflectance(binned, plan.reflectance_factors)
                    # spectrum preview: what the model receives (a copy: the buffers are reused)
                    spec = binned.astype(np.float32, copy=True) if self._spec_ring is not None else None
                    with prof.time('predict'):
                        _, classifier, columns = job.model
                        ids = classifier.predict(binned, columns[job.source])
                # display only: brightness of what the model saw, one value per slit pixel
                intensity = binned.mean(axis=1, dtype=np.float32)
                now = time.perf_counter()
                if now - self._display_t >= self._display_interval:
                    self._display_t = now
                    self._raw_frame = frame.copy()
                self._free.put(job.slot)                       # buffer back to the camera
                self._results.put((job, ids, intensity, pre, spec, clipped))
        except Exception as exc:
            self._fail(f'worker-{k}', exc)
        finally:
            self._results.put(None)

    def _new_runtime(self, k: int, backend: str) -> PreprocessingRuntime:
        # Created in this thread: a CUDA backend's context belongs to the thread using it.
        threads = self.config.processing.numba_threads
        if threads:
            import numba                       # thread-local: binning/fused kernels of this worker
            numba.set_num_threads(max(1, min(int(threads), numba.config.NUMBA_NUM_THREADS)))
        if backend == 'auto':
            # hsi's order (CuPy, OpenCV CUDA, CPU) when a GPU is usable; on CPU the fused path,
            # which gives the same binned frames (checked) with about half the work.
            from modules.compute_runtime import get_compute_capabilities
            if not get_compute_capabilities().cuda_available:
                backend = 'cpu_fused'
        if backend == 'cpu_fused':
            from fused_cpu import FusedRuntime
            runtime = FusedRuntime()
        else:
            runtime = PreprocessingRuntime(backend)
        self._runtimes[k] = runtime
        return runtime

    def _warm_up(self, runtime, prof) -> None:
        """Select the backend, JIT-compile binning and touch the model before the clock starts."""
        frame = np.zeros(self._buffers.shape[1:], self._buffers.dtype)
        for plan in self.plans:
            if plan.stage == 'raw':
                binned, _, _ = runtime.process_binned(frame, plan.dark, plan.binner, calibration=plan.smile,
                                                      inverted=plan.raw_inverted)
            else:
                binned = frame
            self.classifier.predict(binned, plan.columns)
        prof.take()

    # ----------------------------------------------------------------------- aggregator ---

    def _aggregate_loop(self) -> None:
        pending: dict[int, tuple] = {}
        next_seq = 0
        last_line = -1
        rate_t, rate_n = time.perf_counter(), 0
        profile_every = self.config.processing.profile_every_s
        stage_t = log_t = time.perf_counter()
        stage_acc: dict[str, list] = {}
        log_acc: dict[str, list] = {}
        try:
            while self._workers_done < self._n_workers:
                if self._state == 'starting':
                    time.sleep(0.01)
                    if not self._stop.is_set():
                        continue
                try:
                    item = self._results.get(timeout=0.25)
                except queue.Empty:
                    item = False
                if item is None:
                    self._workers_done += 1
                elif item:
                    pending[item[0].seq] = item
                while next_seq in pending:
                    job, ids, intensity, pre, spec, clipped = pending.pop(next_seq)
                    next_seq += 1
                    self._account(job, ids, job.line - last_line - 1, intensity, pre, spec, clipped)
                    last_line = job.line
                    rate_n += 1

                now = time.perf_counter()
                if now - rate_t >= _RATE_PERIOD_S:
                    self._processed_fps = rate_n / (now - rate_t)
                    rate_t, rate_n = now, 0
                if now - stage_t >= 0.5:
                    for prof in self._profilers:
                        for name, (total, count) in prof.take().items():
                            for acc in (stage_acc, log_acc):
                                cell = acc.setdefault(name, [0.0, 0])
                                cell[0] += total
                                cell[1] += count
                    backend = self.backend_name
                    with self._lock:
                        self._stage_ms = {n: t / c for n, (t, c) in stage_acc.items() if c}
                        totals = self._stage_totals.setdefault(backend, {})
                        peaks = self._stage_peak.setdefault(backend, {})
                        for name, (total, count) in stage_acc.items():
                            if count:
                                cell = totals.setdefault(name, [0.0, 0])
                                cell[0] += total
                                cell[1] += count
                                peaks[name] = max(peaks.get(name, 0.0), total / count)
                    stage_acc.clear()
                    stage_t = now
                if profile_every and now - log_t >= profile_every:
                    snap = self.snapshot()
                    logger.info("[profile] backend=%s cam=%.0f/s proc=%.0f/s lost=%d buffers=%d/%d "
                                "latency p50=%.2f p99=%.2f ms | %s", snap.backend, snap.camera_fps,
                                snap.processed_fps, snap.lines_lost, snap.buffers_in_use, snap.buffer_frames,
                                snap.latency_ms['p50'], snap.latency_ms['p99'],
                                '  '.join(f"{n}={t / c:.3f}ms" for n, (t, c) in log_acc.items() if c))
                    log_acc.clear()
                    log_t = now
        except Exception as exc:
            self._fail('aggregator', exc)
        finally:
            self._processed_fps = 0.0
            if self._state in ('running', 'starting'):
                self._state = 'stopped' if self._stop.is_set() else 'finished'

    def _account(self, job: _Job, ids: np.ndarray, gap: int, intensity: np.ndarray | None = None,
                 pre: np.ndarray | None = None, spec: np.ndarray | None = None,
                 clipped: np.ndarray | None = None) -> None:
        """One line, in order. ``gap`` lines before it were lost to overrun."""
        counts = np.bincount(ids, minlength=256)
        t_done = time.perf_counter()
        with self._lock:
            if job.model is not None and job.model[0] != self._drawn_generation:
                # first line of another model: its class ids are new, so the history starts again
                self._drawn_generation = job.model[0]
                self._counts[:] = 0
                self._counted_lines = 0
                self._recent_counts[:] = 0
                self._recent_counted[:] = False
                self._canvas[:] = 0
            if gap > 0:
                self._draw_lost(gap)
            if pre is not None and self._cal_ring is not None and pre.shape == self._cal_ring.shape[1:]:
                k = self._cal_pos % self._cal_ring.shape[0]
                self._cal_ring[k] = pre
                if self._cal_clip is not None:
                    self._cal_clip[k] = 0 if clipped is None else np.minimum(clipped, 65535)
                self._cal_ids[k] = self._canvas_line
                self._cal_pos += 1
            if (spec is not None and self._spec_ring is not None and spec.shape == self._spec_ring.shape[1:]
                    and job.spectrum_generation == self._spectrum_generation):
                k = self._canvas_line % len(self._spec_ids)
                self._spec_ring[k] = spec
                self._spec_ids[k] = self._canvas_line
            self._counts += counts
            self._counted_lines += 1
            k = self._canvas_line % len(self._recent_counted)
            self._recent_counts[k] = counts
            self._recent_counted[k] = True
            self._processed += 1
            if self._canvas_line % self._stride == 0:
                self._canvas[:, self._canvas_col] = ids
                self._raw_canvas[:, self._canvas_col] = np.nan if intensity is None else intensity
                self._canvas_lines[self._canvas_col] = self._canvas_line
                self._advance_canvas(1)
            self._canvas_line += 1
            self._latency[self._latency_n % _LATENCY_WINDOW] = (t_done - job.t_sensor) * 1000.0
            self._latency_n += 1
            src = self.sources[job.source]
            self._position = (src.name, job.index, src.n_frames, job.cycle)

    def _draw_lost(self, gap: int) -> None:
        first = self._canvas_line
        ring = len(self._recent_counted)
        lost = np.arange(first, first + min(gap, ring)) % ring       # lost lines count nothing
        self._recent_counts[lost] = 0
        self._recent_counted[lost] = False
        self._canvas_line += gap
        columns = len(range(-(-first // self._stride) * self._stride, self._canvas_line, self._stride))
        width = self._canvas.shape[1]
        for _ in range(min(columns, width)):
            self._canvas[:, self._canvas_col] = LOST_CLASS
            self._raw_canvas[:, self._canvas_col] = np.nan
            self._canvas_lines[self._canvas_col] = -1
            self._advance_canvas(1)

    def _advance_canvas(self, n: int) -> None:
        width = self._canvas.shape[1]
        self._canvas_col = (self._canvas_col + n) % width
        self._canvas_filled = min(width, self._canvas_filled + n)

    def _fail(self, where: str, exc: Exception) -> None:
        logger.exception("%s failed", where)
        if self._error is None:
            self._error = f"{where}: {type(exc).__name__}: {exc}"
        self._state = 'error'
        self._stop.set()
