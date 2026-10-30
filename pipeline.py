# -*- coding: utf-8 -*-
"""Lossless high-rate pushbroom simulation: camera -> parallel frame workers -> in-order sink.

A pushbroom frame is one line of the belt, so no frame may vanish unnoticed::

    CameraSimulator ──(frame buffers)──► FrameWorker × N ──(results)──► Aggregator ──► snapshot()
     paces lines at camera.fps           dark/hot/flip/Smile/bin          restores line order,     GUI pulls
     copies each frame into a free       + classify (each worker owns     counts EVERY line, fills  at ~30 Hz
     buffer like a frame grabber         its PreprocessingRuntime)        the belt map, latency

* Buffers: ``camera.buffer_frames`` frame slots are allocated once. A line for which no slot
  is free is an *overrun*: with ``overrun='drop'`` it is lost exactly like on a real camera
  (counted, and shown as a magenta column on the belt map); with ``'block'`` the camera waits
  (lossless throughput measurement).
* Timing: ``camera.fps`` is a fixed line clock (no drift); a frame's sensor time is its
  scheduled time, so latency includes any lag of the simulated camera itself.
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

logger = logging.getLogger(__name__)

LOST_CLASS = 255                     # class id drawn for lines lost to overrun
BACKENDS = ('auto', 'cpu', 'cupy_cuda', 'opencv_cuda')
_LATENCY_WINDOW = 4096
_RATE_PERIOD_S = 0.5


# ------------------------------------------------------------------------------ helpers ---

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
    t_sensor: float     # perf_counter time the line was exposed
    source: int
    index: int
    cycle: int


@dataclass
class Snapshot:
    """State copied for one GUI/console refresh (owned by the caller)."""
    state: str                       # starting | running | finished | stopped | error
    error: str | None
    elapsed_s: float
    target_fps: float
    camera_fps: float                # lines the camera delivered per second (recent)
    processed_fps: float             # lines classified per second (recent)
    frames_delivered: int
    lines_processed: int
    lines_lost: int                  # overrun: no free buffer when the line was exposed
    schedule_slips: int              # the simulated camera itself fell > 50 ms behind its clock
    buffers_in_use: int
    buffer_frames: int
    latency_ms: dict                 # p50 / p99 / max, sensor time -> classified
    stage_ms: dict                   # mean ms per stage over the last interval
    backend: str                     # backend(s) the workers actually run
    requested_backend: str           # what was asked for (config or set_backend)
    backend_note: str                # why the actual backend differs from the request ('' if not)
    counts: np.ndarray               # (256,) pixels per class id since the last reset
    counted_lines: int               # lines in ``counts``
    canvas: np.ndarray               # (slit, canvas_width) class ids, oldest -> newest
    canvas_filled: int
    raw_frame: np.ndarray | None     # newest raw frame (display only)
    position: tuple                  # (source name, frame index, n frames, cycle)


# ----------------------------------------------------------------------------- pipeline ---

def build_pipeline(config) -> 'Pipeline':
    """Open the configured input, resolve one acquisition plan per file, load and bind the model."""
    from inference_engine import PushbroomLineClassifier
    from sim_config import build_plan
    from sources import open_sources

    src_cfg = config.source
    sources = open_sources(config.resolve(src_cfg.path), src_cfg.pattern, stage=src_cfg.stage,
                           scan_axis=src_cfg.envi_scan_axis, dataset=src_cfg.dataset, cache=src_cfg.cache,
                           cache_dir=config.resolve(src_cfg.cache_dir))
    classifier = PushbroomLineClassifier(config.resolve(config.model.path))
    plans = []
    for src in sources:
        plan = build_plan(config, src)
        plan.columns, binding = classifier.bind(plan.wavelengths, plan.n_columns, config.model.band_tolerance_nm)
        plan.provenance['model bands'] = (binding, os.path.basename(classifier.path))
        logger.info("%s: %d frames %s %s\n%s", src.name, src.n_frames, src.frame_shape, src.dtype, plan.summary())
        plans.append(plan)
    return Pipeline(config, sources, plans, classifier)


class Pipeline:
    """Run one simulation stream.

    Args:
        config: ``SimulationConfig``.
        sources: Playlist from ``sources.open_sources`` (same frame shape/dtype/stage).
        plans: One ``AcquisitionPlan`` per source (``sim_config.build_plan``), with ``columns``.
        classifier: ``PushbroomLineClassifier``.
    """

    def __init__(self, config, sources, plans, classifier) -> None:
        if len(plans) != len(sources):
            raise ValueError("one acquisition plan per source is required")
        self.config = config
        self.sources = sources
        self.plans = plans
        self.classifier = classifier
        cam, disp = config.camera, config.display
        self._fps = float(cam.fps)
        self._n_workers = int(config.processing.workers)
        shape = sources[0].frame_shape
        self._slit = shape[0]

        self._buffers = np.empty((cam.buffer_frames, *shape), dtype=sources[0].dtype)
        self._free: queue.SimpleQueue = queue.SimpleQueue()
        for slot in range(cam.buffer_frames):
            self._free.put(slot)
        self._work: queue.SimpleQueue = queue.SimpleQueue()
        self._results: queue.SimpleQueue = queue.SimpleQueue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._profilers = [StageProfiler() for _ in range(self._n_workers)]
        self._runtimes: list[PreprocessingRuntime | None] = [None] * self._n_workers
        # Backend switch: workers compare the generation at each frame boundary.
        self._requested_backend = self.plans[0].correction_backend
        self._backend_generation = 0
        self._timer: HighResTimer | None = None

        # camera thread state
        self._delivered = 0
        self._lost = 0
        self._slips = 0
        self._camera_fps = 0.0
        self._camera_done = False

        # aggregator state (guarded by _lock)
        self._counts = np.zeros(256, np.int64)
        self._counted_lines = 0
        self._processed = 0
        self._processed_fps = 0.0
        self._latency = np.zeros(_LATENCY_WINDOW, np.float32)
        self._latency_n = 0
        self._stride = disp.canvas_line_stride
        self._canvas = np.zeros((self._slit, disp.canvas_width), np.uint8)
        self._canvas_col = 0
        self._canvas_filled = 0
        self._canvas_line = 0
        self._position = (sources[0].name, 0, sources[0].n_frames, 0)
        self._stage_ms: dict[str, float] = {}

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
        logger.info("Pipeline started: %s lines/s target, %d workers, %d buffers, overrun=%s",
                    'max' if self._fps == 0 else f"{self._fps:g}", self._n_workers,
                    len(self._buffers), self.config.camera.overrun)

    def stop(self, timeout: float = 3.0) -> bool:
        """Stop all threads; returns False if one is still busy after ``timeout``."""
        self._stop.set()
        deadline = time.perf_counter() + timeout
        for thread in self._threads:
            thread.join(max(0.0, deadline - time.perf_counter()))
        if self._timer is not None:
            self._timer.close()
        if self._state in ('running', 'starting'):
            self._state = 'stopped'
        return not any(t.is_alive() for t in self._threads)

    def reset_counts(self) -> None:
        """Start a new batch: class counts restart from zero (throughput counters keep running)."""
        with self._lock:
            self._counts[:] = 0
            self._counted_lines = 0

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

    def snapshot(self) -> Snapshot:
        with self._lock:
            n = min(self._latency_n, _LATENCY_WINDOW)
            lat = self._latency[:n]
            latency = ({'p50': float(np.percentile(lat, 50)), 'p99': float(np.percentile(lat, 99)),
                        'max': float(lat.max())} if n else {'p50': 0.0, 'p99': 0.0, 'max': 0.0})
            if self._canvas_filled < self._canvas.shape[1]:
                canvas = self._canvas[:, :self._canvas_filled].copy()
            else:
                canvas = np.roll(self._canvas, -self._canvas_col, axis=1)
            snap = Snapshot(
                state=self._state, error=self._error, elapsed_s=time.perf_counter() - self._t0,
                target_fps=self._fps, camera_fps=self._camera_fps, processed_fps=self._processed_fps,
                frames_delivered=self._delivered, lines_processed=self._processed, lines_lost=self._lost,
                schedule_slips=self._slips,
                buffers_in_use=len(self._buffers) - self._free.qsize(), buffer_frames=len(self._buffers),
                latency_ms=latency, stage_ms=dict(self._stage_ms), backend=self.backend_name,
                requested_backend=self._requested_backend, backend_note=self._backend_note(),
                counts=self._counts.copy(), counted_lines=self._counted_lines,
                canvas=canvas, canvas_filled=self._canvas_filled,
                raw_frame=self._raw_frame, position=self._position)
        return snap

    # --------------------------------------------------------------------------- camera ---

    def _camera_loop(self) -> None:
        cam = self.config.camera
        period = 1.0 / self._fps if self._fps > 0 else 0.0
        drop = cam.overrun == 'drop'
        line = seq = cycle = 0
        try:
            # The clock starts once every worker has compiled/selected its kernels: a real
            # camera is started after the processing chain is ready, too.
            for ready in self._ready:
                while not ready.wait(0.05):
                    if self._stop.is_set():
                        return
            logger.info("Workers ready after %.2f s; camera clock started", time.perf_counter() - self._t0)
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
                            if drop:                                   # overrun: the line is lost
                                self._lost += 1
                                line += 1
                                continue
                            slot = self._wait_slot()
                            if slot is None:
                                return
                        np.copyto(self._buffers[slot], frames[i])
                        self._work.put(_Job(seq, line, slot, t_sensor, src_i, i, cycle))
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
        classifier = self.classifier
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
                with prof.time('total'):
                    if plan.stage == 'raw':
                        binned, _, _ = runtime.process_binned(
                            frame, plan.dark, plan.binner, calibration=plan.smile,
                            inverted=plan.raw_inverted, profiler=prof)
                    else:
                        binned = frame
                    with prof.time('predict'):
                        ids = classifier.predict(binned, plan.columns)
                now = time.perf_counter()
                if now - self._display_t >= self._display_interval:
                    self._display_t = now
                    self._raw_frame = frame.copy()
                self._free.put(job.slot)                       # buffer back to the camera
                self._results.put((job, ids, time.perf_counter()))
        except Exception as exc:
            self._fail(f'worker-{k}', exc)
        finally:
            self._results.put(None)

    def _new_runtime(self, k: int, backend: str) -> PreprocessingRuntime:
        # Created in this thread: a CUDA backend's context belongs to the thread using it.
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
                    job, ids, _ = pending.pop(next_seq)
                    next_seq += 1
                    self._account(job, ids, job.line - last_line - 1)
                    last_line = job.line
                    rate_n += 1

                now = time.perf_counter()
                if now - rate_t >= _RATE_PERIOD_S:
                    self._processed_fps = rate_n / (now - rate_t)
                    rate_t, rate_n = now, 0
                if now - stage_t >= 1.0:
                    for prof in self._profilers:
                        for name, (total, count) in prof.take().items():
                            for acc in (stage_acc, log_acc):
                                cell = acc.setdefault(name, [0.0, 0])
                                cell[0] += total
                                cell[1] += count
                    with self._lock:
                        self._stage_ms = {n: t / c for n, (t, c) in stage_acc.items() if c}
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

    def _account(self, job: _Job, ids: np.ndarray, gap: int) -> None:
        """One line, in order. ``gap`` lines before it were lost to overrun."""
        counts = np.bincount(ids, minlength=256)
        t_done = time.perf_counter()
        with self._lock:
            if gap > 0:
                self._draw_lost(gap)
            self._counts += counts
            self._counted_lines += 1
            self._processed += 1
            if self._canvas_line % self._stride == 0:
                self._canvas[:, self._canvas_col] = ids
                self._advance_canvas(1)
            self._canvas_line += 1
            self._latency[self._latency_n % _LATENCY_WINDOW] = (t_done - job.t_sensor) * 1000.0
            self._latency_n += 1
            src = self.sources[job.source]
            self._position = (src.name, job.index, src.n_frames, job.cycle)

    def _draw_lost(self, gap: int) -> None:
        first = self._canvas_line
        self._canvas_line += gap
        columns = len(range(-(-first // self._stride) * self._stride, self._canvas_line, self._stride))
        width = self._canvas.shape[1]
        for _ in range(min(columns, width)):
            self._canvas[:, self._canvas_col] = LOST_CLASS
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
