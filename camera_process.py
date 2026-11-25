# -*- coding: utf-8 -*-
"""Live camera read in a process of its own (``live.reader = "process"``, the default).

Why: in one process the camera thread shares the GIL with the GUI, the frame workers and the
aggregator. Every time a camera read waits for the SDK it gives the GIL away and may wait a whole
switch interval to get it back. On the 2026-11-07 live run that capped the reader at ~88 fps
against the camera's 121 fps: the GenTL buffer pool stayed full (0.72 s latency) and the producer
dropped 26 % of the frames. A child process has its own GIL, so nothing in the GUI process can
slow the camera down. A slow GUI process now only shows up as overruns (no free frame buffer),
counted as lost lines like before.

Layout::

    GUI / simulate process                             camera process (spawned)
    ----------------------                             ------------------------
    open_camera_process(config) -- spawn ------------> camera_source.open_camera(config)
                                <-- ('ready', LiveSource)   (the same open code as the thread reader)
    Pipeline creates SharedFrames
    camera thread: start_stream -- ('start', name) --> attach, start_stream, read every frame:
                                                         take a free slot, read straight into it,
    next_job() <-- job ring + jobs_ready semaphore ----  write a job record, release
    workers: return_slot() -- free ring + slots_free -> next read takes it (never blocks)
    close(): stop event ------------------------------> the read returns, the camera is closed

Shared memory (:class:`SharedFrames`): the frame buffers ``(n, slit, spectral)``, a ring of ``n``
job records, a ring of ``n`` free slot numbers and a few counters. Each ring has one writer, and
its semaphore counts the entries written: an entry is written before the ``release`` that
announces it and read after the ``acquire`` that takes it, so the semaphore also orders the
memory (frame data included). ``SemLock.acquire`` first tries without releasing the GIL, so the
GUI process's camera thread gives the GIL away only when it has to wait (once per batch of
frames), never once per frame. At most ``n`` slots exist, so neither ring can overflow.

The child process ignores Ctrl+C (the parent stops it) and is a daemon (it ends with the parent).
A child stuck in the SDK is terminated after ``close_timeout_s``; the operating system then
releases the camera.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from multiprocessing import shared_memory

import numpy as np

logger = logging.getLogger(__name__)

_CTX = mp.get_context('spawn')       # never fork: Qt, CUDA and SDK threads do not survive a fork
OPEN_TIMEOUT_S = 120.0               # spawn + imports + camera open (GenICam start retries)

JOB_DTYPE = np.dtype([('seq', '<i8'), ('line', '<i8'), ('slot', '<i8'),
                      ('t_sensor', '<f8'), ('t_arrival', '<f8')])
# Counters written by the camera process (one aligned int64 each; read at any time, monotonic).
RECEIVED, LOST, LOST_CAMERA, TIMEOUTS, CAMERA_CLOCK, TEMPERATURE = range(6)
#: ``counters[TEMPERATURE]``: sensor temperature in m°C, or this when the camera reports none
NO_TEMPERATURE = -(2 ** 62)
_N_COUNTERS = 8
_ALIGN = 64

# Exception types re-raised as themselves in the GUI process (others become RuntimeError).
_ERRORS = {cls.__name__: cls for cls in (ValueError, RuntimeError, OSError, FileNotFoundError,
                                         PermissionError, ImportError, ModuleNotFoundError,
                                         TimeoutError, KeyError, TypeError)}


class SharedFrames:
    """Frame buffers, job ring, free ring and counters in one shared memory block.

    Created by the GUI process (``name=None``), attached by the camera process (``name``).
    """

    def __init__(self, n: int, shape, dtype, name: str | None = None) -> None:
        self.n, self.shape, self.dtype = int(n), tuple(int(s) for s in shape), np.dtype(dtype)
        frames = self.n * int(np.prod(self.shape)) * self.dtype.itemsize
        jobs_at = _aligned(frames)
        free_at = _aligned(jobs_at + self.n * JOB_DTYPE.itemsize)
        counters_at = _aligned(free_at + self.n * 8)
        size = counters_at + _N_COUNTERS * 8
        if name is None:
            _check_room(size)
            self.shm = shared_memory.SharedMemory(create=True, size=size)
        else:
            self.shm = shared_memory.SharedMemory(name=name)
        buf = self.shm.buf
        self.frames = np.ndarray((self.n, *self.shape), self.dtype, buffer=buf)
        self.jobs = np.ndarray((self.n,), JOB_DTYPE, buffer=buf, offset=jobs_at)
        self.free = np.ndarray((self.n,), np.int64, buffer=buf, offset=free_at)
        self.counters = np.ndarray((_N_COUNTERS,), np.int64, buffer=buf, offset=counters_at)
        if name is None:
            self.counters[:] = 0
            self.counters[TEMPERATURE] = NO_TEMPERATURE
        self._owner = name is None

    @property
    def name(self) -> str:
        return self.shm.name

    def release(self) -> None:
        """Drop the views and the mapping; the creator also removes the block (POSIX)."""
        self.frames = self.jobs = self.free = self.counters = None
        try:
            self.shm.close()
        except BufferError:                  # a view is still alive somewhere: freed with it
            logger.debug("shared frame memory still referenced; left to the garbage collector")
        if self._owner:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass


class SlotReturn:
    """``Pipeline._free`` for the process reader: workers ``put`` slots back to the camera process."""

    def __init__(self, camera: 'CameraProcess', n: int) -> None:
        self._camera, self._n = camera, int(n)

    def put(self, slot: int) -> None:
        self._camera.return_slot(slot)

    def qsize(self) -> int:
        """Slots not held by the pipeline (free, or in the camera process)."""
        return self._n - self._camera.slots_in_pipeline()


class CameraProcess:
    """The camera process seen from the GUI process; ``Pipeline`` holds it as its ``camera``.

    Use :func:`open_camera_process` to make one. Methods are called from the pipeline's camera
    thread, except :meth:`return_slot` (any worker thread) and :meth:`close` (camera thread, or
    the pipeline's ``stop`` / ``build_pipeline`` when the stream never started).
    """

    separate_process = True

    def __init__(self, config, backend_factory=None) -> None:
        self.read_timeout_s = config.live.read_timeout_ms / 1000.0
        self.close_timeout_s = self.read_timeout_s + 5.0     # read + camera close; then terminated
        self._conn, child_conn = _CTX.Pipe()
        self.stop_event = _CTX.Event()
        self.jobs_ready = _CTX.Semaphore(0)
        self.slots_free = _CTX.Semaphore(0)
        self.process = _CTX.Process(
            target=_camera_main, name='camera',
            args=(config, child_conn, self.stop_event, self.jobs_ready, self.slots_free, backend_factory),
            daemon=True)
        self.process.start()
        child_conn.close()
        self.shared: SharedFrames | None = None
        self._free_lock = threading.Lock()
        self._free_w = 0                     # free-ring entries written (initial fill included)
        self._jobs_r = 0                     # job records read
        self._closed = False
        self._ended = False                  # the camera process said it has finished

    # ------------------------------------------------------------------ control messages ---

    def _message(self, timeout: float):
        """Next control message, or ``None`` after ``timeout``; raises if the process failed or died."""
        try:
            if self._conn.poll(timeout):
                message = self._conn.recv()
            elif self.process.is_alive():
                return None
            else:
                raise RuntimeError(f"camera process ended unexpectedly (exit code {self.process.exitcode})")
        except (EOFError, BrokenPipeError, ConnectionResetError):
            self.process.join(1.0)
            raise RuntimeError(f"camera process ended unexpectedly (exit code {self.process.exitcode})") from None
        if message[0] == 'error':
            _, kind, text = message
            raise _ERRORS.get(kind, RuntimeError)(text if kind in _ERRORS else f"{kind}: {text}")
        if message[0] == 'closed':
            self._ended = True
        return message

    def wait_ready(self, timeout: float = OPEN_TIMEOUT_S):
        """Wait for the camera process to open the camera; returns its ``LiveSource``."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            message = self._message(0.1)
            if message is not None and message[0] == 'ready':
                return message[1]
        raise TimeoutError(f"the camera process did not open the camera within {timeout:g} s")

    def start_stream(self, shared: SharedFrames, stop: threading.Event) -> bool:
        """Hand over the shared buffers and start the stream; ``False`` if ``stop`` cut the start short."""
        self.shared = shared
        for slot in range(shared.n):         # every buffer starts free
            self._put_free(slot)
        self._conn.send(('start', shared.name, shared.n, shared.shape, shared.dtype.str))
        while True:
            if stop.is_set():
                self.stop_event.set()        # e.g. GenICam start retries: the camera process gives up
            message = self._message(0.05)
            if message is None:
                continue
            if message[0] == 'running':
                return True
            if message[0] in ('abandoned', 'closed'):
                return False

    def check(self) -> None:
        """Raise if the camera process reported an error or ended while streaming."""
        message = self._message(0.0)
        if self._ended or (message is not None and message[0] == 'closed'):
            raise RuntimeError(f"camera process ended (exit code {self.process.exitcode})")

    # ---------------------------------------------------------------------------- frames ---

    def next_job(self, timeout: float):
        """``(seq, line, slot, t_sensor, t_arrival)`` of the next frame, or ``None`` after ``timeout``."""
        if not self.jobs_ready.acquire(timeout=timeout):
            return None
        record = self.shared.jobs[self._jobs_r % self.shared.n]
        self._jobs_r += 1
        return (int(record['seq']), int(record['line']), int(record['slot']),
                float(record['t_sensor']), float(record['t_arrival']))

    def return_slot(self, slot: int) -> None:
        """A worker is done with ``slot``: the camera process may fill it again."""
        self._put_free(slot)

    def _put_free(self, slot: int) -> None:
        with self._free_lock:
            self.shared.free[self._free_w % self.shared.n] = slot
            self._free_w += 1
        self.slots_free.release()

    def slots_in_pipeline(self) -> int:
        """Frames read by the pipeline whose slot has not been returned yet."""
        if self.shared is None:
            return 0
        return max(0, self._jobs_r - (self._free_w - self.shared.n))

    def counters(self) -> dict:
        """The camera process's running totals (zeros before the stream starts)."""
        if self.shared is None or self.shared.counters is None:
            return {'received': 0, 'lost': 0, 'lost_camera': 0, 'timeouts': 0, 'line_clock': 'arrival',
                    'temperature_c': None}
        c = self.shared.counters.copy()
        return {'received': int(c[RECEIVED]), 'lost': int(c[LOST]), 'lost_camera': int(c[LOST_CAMERA]),
                'timeouts': int(c[TIMEOUTS]), 'line_clock': 'camera' if c[CAMERA_CLOCK] else 'arrival',
                'temperature_c': None if c[TEMPERATURE] == NO_TEMPERATURE else c[TEMPERATURE] / 1000.0}

    # ----------------------------------------------------------------------------- close ---

    def close(self) -> None:
        """Stop the stream and end the camera process (idempotent, never raises).

        The process closes the camera itself; one that is still busy after ``close_timeout_s``
        (stuck in the SDK) is terminated, which releases the camera too.
        """
        if self._closed:
            return
        self._closed = True
        self.stop_event.set()
        try:
            self._conn.send(('close',))     # read only while it waits for 'start'
        except (OSError, ValueError):
            pass
        self.process.join(self.close_timeout_s)
        if self.process.is_alive():
            logger.warning("Camera process still busy %.1f s after the stop: terminated", self.close_timeout_s)
            self.process.terminate()
            self.process.join(2.0)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(2.0)
        try:
            self._conn.close()
        except OSError:
            pass


def open_camera_process(config, backend_factory=None):
    """:func:`camera_source.open_camera` in a new camera process.

    Args:
        config: The validated ``SimulationConfig`` (``source.kind = "camera"``); it is pickled to
            the camera process.
        backend_factory: Optional picklable ``settings -> backend`` used instead of
            ``modules.camera.factory.create_backend`` in the camera process (tests).

    Returns:
        ``(CameraProcess, LiveSource)``: the camera is open and not streaming.

    Raises:
        The open error of the camera process, as its own type for the usual built-in exceptions
        (``ValueError`` for an unsupported bin, ``FileNotFoundError`` for a missing producer, ...).
    """
    camera = CameraProcess(config, backend_factory)
    try:
        live = camera.wait_ready()
    except BaseException:
        camera.close()
        raise
    logger.info("Camera reader runs in process %d", camera.process.pid)
    return camera, live


# ------------------------------------------------------------------------ camera process ---

def _camera_main(config, conn, stop_event, jobs_ready, slots_free, backend_factory) -> None:
    """Entry point of the camera process: open, wait for 'start', read until the stop event."""
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)     # Ctrl+C is the parent's; it stops us
    except (ValueError, OSError):
        pass
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                            format='%(asctime)s [%(levelname)s] (camera-proc) %(message)s')
    camera = shared = None
    try:
        if backend_factory is not None:
            from modules.camera import factory
            factory.create_backend = backend_factory
        from camera_source import open_camera
        camera, live = open_camera(config)
        conn.send(('ready', live))
        command = conn.recv()
        if command[0] != 'start':
            return
        _, name, n, shape, dtype = command
        shared = SharedFrames(n, shape, dtype, name=name)
        _read_loop(camera, shared, conn, stop_event, jobs_ready, slots_free,
                   int(config.live.read_timeout_ms))
    except (EOFError, BrokenPipeError):
        pass                                             # the GUI process is gone
    except BaseException as exc:
        logger.exception("Camera process failed")
        _send(conn, ('error', type(exc).__name__, str(exc)))
    finally:
        if camera is not None:
            try:
                camera.close()
            except Exception:
                logger.exception("Closing the camera failed")
        if shared is not None:
            shared.release()
        _send(conn, ('closed',))


def _read_loop(camera, shared: SharedFrames, conn, stop_event, jobs_ready, slots_free, timeout_ms: int) -> None:
    """``Pipeline._live_loop`` on this side of the shared memory (same counting rules)."""
    from camera_source import CameraClock, CameraFrameGaps, TemperaturePoll
    n, shape = shared.n, shared.shape
    frames, jobs, free, counters = shared.frames, shared.jobs, shared.free, shared.counters
    scratch = np.empty(shape, shared.dtype)
    target = scratch
    held = None                          # a free slot taken for a read that returned nothing
    free_r = job_w = 0
    line = seq = received = lost = lost_camera = timeouts = 0

    def provide(frame_shape):
        return target if tuple(frame_shape) == shape else None

    def take_slot():
        nonlocal free_r, held
        if held is not None:
            slot, held = held, None
            return slot
        if not slots_free.acquire(False):
            return None
        slot = int(free[free_r % n])
        free_r += 1
        return slot

    camera.set_stop_event(stop_event)    # set() makes a blocked read return None
    try:
        camera.start_stream()
    except Exception:
        if stop_event.is_set():
            logger.info("Camera start abandoned: stop requested")
            _send(conn, ('abandoned',))
            return
        raise
    gaps, clock = CameraFrameGaps(camera), CameraClock(camera)
    temperature = TemperaturePoll(camera)
    gaps.start()
    conn.send(('running',))
    try:
        while not stop_event.is_set():
            slot = take_slot()
            target = scratch if slot is None else frames[slot]
            frame = camera.read_frame_raw_into(provide, timeout_ms=timeout_ms)
            t_arrival = time.perf_counter()
            if temperature.due(t_arrival):
                value = temperature.read(t_arrival)
                counters[TEMPERATURE] = NO_TEMPERATURE if value is None else int(round(value * 1000.0))
            if frame is None:
                held = slot
                if not stop_event.is_set():
                    timeouts += 1
                    counters[TIMEOUTS] = timeouts
                continue
            if tuple(frame.shape) != shape:
                raise RuntimeError(f"camera frame {tuple(frame.shape)} differs from the {shape} the "
                                   "calibration was resolved for (bin or ROI changed?)")
            received += 1
            missed = gaps.missed()
            if missed:
                lost += missed
                lost_camera += missed
                line += missed
            t_sensor = clock.sensor_time(t_arrival)
            if slot is None:                 # a buffer may have come free during the read
                slot = take_slot()
                if slot is not None:
                    target = frames[slot]
            if slot is None:                 # overrun: no free buffer for this line
                lost += 1
                line += 1
            else:
                if frame is not target:
                    np.copyto(target, frame)
                jobs[job_w % n] = (seq, line, slot, t_sensor, t_arrival)
                job_w += 1
                jobs_ready.release()         # after the record and the frame: orders the memory
                seq += 1
                line += 1
            counters[RECEIVED] = received
            counters[LOST] = lost
            counters[LOST_CAMERA] = lost_camera
            counters[CAMERA_CLOCK] = clock.source == 'camera'
    finally:
        logger.info("Camera process read %d frames (%d lost in the camera, %d read timeouts)",
                    received, lost_camera, timeouts)


def _send(conn, message) -> None:
    try:
        conn.send(message)
    except (OSError, ValueError):
        pass


def _aligned(offset: int) -> int:
    return -(-offset // _ALIGN) * _ALIGN


def _check_room(size: int) -> None:
    """Linux: fail clearly when ``/dev/shm`` is too small (Docker's default is 64 MB), instead of a
    SIGBUS in the middle of a run."""
    if not sys.platform.startswith('linux') or not os.path.isdir('/dev/shm'):
        return
    st = os.statvfs('/dev/shm')
    room = st.f_bavail * st.f_frsize
    if room < size:
        raise OSError(f"/dev/shm has {room / 2**20:.0f} MB free, the camera buffers need {size / 2**20:.0f} MB: "
                      "lower camera.buffer_frames, enlarge /dev/shm (Docker: --shm-size) or set "
                      "live.reader = \"thread\"")
