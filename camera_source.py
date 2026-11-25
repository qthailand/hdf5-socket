# -*- coding: utf-8 -*-
"""Live camera as a frame source: hsi ``modules.camera`` backends feeding the same pipeline.

``source.kind = "camera"`` replaces the recorded files with a camera. Everything after the frame
buffers stays as it is (dark / hot pixel / flip / Smile / binning, model, statistics), because a
camera delivers exactly what the HDF5 "Keep Spectrogram" recordings store: the raw ``uint16``
sensor frame ``(slit, spectral)`` at the hardware bin.

* :func:`open_camera` builds the backend named by ``acquisition.camera_backend`` with hsi's
  ``create_backend``, opens it, applies ``acquisition.camera_bin``, ``exposure_ms`` and ``gain``
  and the cooler target ``live.target_temperature_c`` (TEC cameras, as hsi), and describes it as
  a :class:`LiveSource`. Its ``metadata`` holds the keys hsi writes into a
  recording (``camera_backend``, ``camera_bin``, ``exposure_ms``, ``gain``, sensor temperature ...),
  read back from the opened camera, so ``sim_config.build_plan`` selects the dark, Smile and
  wavelength files exactly as it does for a recording.
* After that the pipeline's camera thread owns the backend (``Pipeline._live_loop``):
  ``start_stream``, every read and finally ``close``. Calls are never concurrent; the open runs
  on a short-lived helper thread so that harvesters (GenICam) does not install its Ctrl+C handler
  on the main thread (it would destroy the camera instead of raising ``KeyboardInterrupt``).
* :class:`CameraFrameGaps` counts frames the camera exposed but never delivered, and
  :class:`CameraClock` turns GenICam frame timestamps into ``time.perf_counter()`` time for the
  latency figures (other cameras: the arrival time).
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from modules.dark import valid_temperature

logger = logging.getLogger(__name__)

#: GenTL producers looked for in ``$GENICAM_GENTL64_PATH`` when ``live.cti_path`` is not set
#: (SVS-Vistek, the producer hsi uses; the second name is hsi's built-in default).
SVS_PRODUCERS = ('sv_gev_tl_x64.cti', 'libsv_gev_tl_x64.cti')


@dataclass
class LiveSource:
    """A live camera in the place of a :class:`sources.FrameSource`.

    It has what the pipeline, ``sim_config.build_plan`` and the GUI read from a source; there is
    no ``frames`` array - the pipeline's camera thread reads the camera instead.

    Attributes:
        path: ``live_<backend>``; with ``corrections.reflectance = "auto"`` the factors are
            ``<reflectance_dir>/live_<backend>.npy``.
        name: Display name, e.g. ``genicam #0 Model (serial)``.
        frame_shape: ``(slit, spectral)`` = ``(info.height, info.width)`` at the applied bin.
        metadata: hsi recording keys read back from the opened camera (see :func:`camera_metadata`).
    """
    path: str
    name: str
    frame_shape: tuple
    metadata: dict = field(default_factory=dict)
    cooling: dict = field(default_factory=dict)  # see :func:`apply_target_temperature`
    dtype: np.dtype = np.dtype(np.uint16)
    stage: str = 'raw'                   # a camera delivers raw sensor frames
    wavelengths: None = None
    storage: str = 'live'                # a file reports 'ram' / 'disk' / 'stream'
    n_frames: int = 0                    # unbounded: the status shows a frame counter, not "i / n"
    metadata_origin: str = 'camera'      # provenance prefix in build_plan ('file' for recordings)


def open_camera(config):
    """Create, open and configure the camera of ``config`` (``source.kind = "camera"``).

    The backend gets ``acquisition.gain`` / ``exposure_ms`` / ``camera_bin`` as its start-up values
    and ``live.camera_index`` / ``cti_path`` / ``sensor_pixel_size``; after ``open`` the bin is set
    explicitly when the camera picked another one (POA always starts at its own default).

    Returns:
        ``(camera, LiveSource)``. The camera is open and not streaming; the caller owns it and
        must ``close()`` it (``Pipeline`` does).

    Raises:
        ValueError: the camera does not support ``acquisition.camera_bin``.
        RuntimeError, ImportError, OSError: from the backend (no device, SDK not installed, ...).
            The camera is closed before the error is raised.
    """
    from modules.camera import CameraConfig, factory
    acq, live = config.acquisition, config.live
    backend = str(acq.camera_backend).lower()
    wanted_bin = int(acq.camera_bin)
    cti = resolve_cti(config) if backend == 'genicam' else None
    settings = CameraConfig(camera_backend=backend, camera_index=int(live.camera_index),
                            cti_path=cti, default_gain=acq.gain,
                            default_exposure_ms=float(acq.exposure_ms), default_bin=wanted_bin,
                            sensor_pixel_size=int(live.sensor_pixel_size))
    camera = factory.create_backend(settings)

    def configure():
        camera.open()
        if camera.current_bin != wanted_bin:
            bins = tuple(int(b) for b in camera.get_bins())
            if bins and wanted_bin not in bins:
                raise ValueError(f"{backend} camera does not support acquisition.camera_bin = {wanted_bin} "
                                 f"(bins {bins})")
            camera.set_bin(wanted_bin)
        camera.set_params(acq.gain, float(acq.exposure_ms))
        if camera.current_bin != wanted_bin:
            raise RuntimeError(f"{backend} camera stayed at bin {camera.current_bin}, "
                               f"acquisition.camera_bin is {wanted_bin}")
        cooling = apply_target_temperature(camera, getattr(live, 'target_temperature_c', None))
        metadata = camera_metadata(camera, backend, int(live.camera_index), acq)
        if cooling.get('target_c') is not None:
            metadata['target_temperature_c'] = float(cooling['target_c'])
        if cooling.get('state'):
            metadata['cooling_state'] = cooling['state']
        return metadata, cooling

    try:
        metadata, cooling = _run_on_helper_thread(configure, f"open-{backend}")
    except BaseException:
        _close_quietly(camera)
        raise
    shape = (int(metadata['frame_height']), int(metadata['frame_width']))
    label = backend if backend == 'poa' else f"{backend} #{int(live.camera_index)}"
    model, serial = metadata.get('camera_model'), metadata.get('camera_serial')
    name = label + (f" {model}" if model else '') + (f" ({serial})" if serial else '')
    source = LiveSource(path=f"live_{backend}", name=name, frame_shape=shape, metadata=metadata, cooling=cooling)
    logger.info("Camera %s open: %s uint16, bin %d, exposure %s ms, gain %s, sensor %s", name, shape,
                metadata['camera_bin'], metadata.get('exposure_ms'), metadata.get('gain'),
                'n/a' if metadata.get('sensor_temperature_c_start') is None
                else f"{metadata['sensor_temperature_c_start']:.1f} °C")
    return camera, source


def resolve_cti(config) -> str:
    """The GenTL producer (``.cti``) for the GenICam backend.

    ``live.cti_path`` if set (relative to the JSON file), else the first SVS producer
    (:data:`SVS_PRODUCERS`) in one of the ``;``-separated folders of ``$GENICAM_GENTL64_PATH``.
    hsi's own fallback joins that variable with a file name, which fails when it lists several
    folders, and harvesters then only reports "no device found".

    Raises:
        FileNotFoundError: the configured file does not exist, or none was found.
    """
    configured = config.resolve(config.live.cti_path)
    if configured is not None:
        if not configured.is_file():
            raise FileNotFoundError(f"GenTL producer not found: {configured} (live.cti_path)")
        return str(configured)
    folders = [f.strip().strip('"') for f in os.environ.get('GENICAM_GENTL64_PATH', '').split(os.pathsep)]
    folders = [f for f in folders if f]
    for name in SVS_PRODUCERS:
        for folder in folders:
            path = Path(folder) / name
            if path.is_file():
                logger.info("GenTL producer: %s (from GENICAM_GENTL64_PATH)", path)
                return str(path)
    raise FileNotFoundError(f"live.cti_path is not set and no {' / '.join(SVS_PRODUCERS)} is in a "
                            f"GENICAM_GENTL64_PATH folder ({'; '.join(folders) or 'variable not set'}); "
                            "set live.cti_path to the camera's GenTL producer")


def camera_metadata(camera, backend: str, camera_index: int, acq) -> dict:
    """hsi recording attributes for an opened camera (``acquisition/metadata.py`` key names).

    ``exposure_ms`` and ``gain`` are the values the backend applied (it may clamp them), the
    temperature is the sensor's at start (``build_plan`` picks the nearest dark with it). Keys
    whose value is unknown are left out, as hsi does.
    """
    info = camera.info
    values = {
        'camera_backend': backend,
        'camera_index': camera_index,
        'camera_model': camera.model_name(),
        'camera_serial': camera.serial_number(),
        'camera_bin': int(camera.current_bin),
        'frame_height': int(info.height),
        'frame_width': int(info.width),
        'bit_depth': int(info.bit_depth),
        'sensor_bit_depth': int(camera.sensor_bit_depth()),
        'full_scale': int(camera.saturation_level()),
        'exposure_ms': float(getattr(camera, 'exposure_ms', None) or acq.exposure_ms),
        'gain': float(acq.gain if getattr(camera, 'gain', None) is None else camera.gain),
        'sensor_temperature_c_start': valid_temperature(camera.get_temperature()),
    }
    return {key: value for key, value in values.items() if value is not None}


def apply_target_temperature(camera, target_c) -> dict:
    """Set the TEC cooler target of an opened camera, as hsi's ``ScanController._setup_cooling``.

    ``target_c`` ``None`` leaves the cooler as the camera has it (only read back). A camera without
    a settable target, or one that refuses the value, does not stop the start: the reason is
    logged and returned (the camera then runs uncooled or at its own target).

    Returns:
        ``{'supported', 'range', 'requested', 'target_c', 'state', 'error'}``: ``range`` the
        camera's ``(min, max)`` °C or ``None``, ``target_c`` the target read back after the
        write, ``state`` the regulation state text (GenICam), ``error`` why it was not applied.
    """
    result = {'supported': False, 'range': None, 'requested': None if target_c is None else float(target_c),
              'target_c': None, 'state': None, 'error': None}
    try:
        result['supported'] = bool(camera.cooling_supported())
        if result['supported']:
            value_range = camera.target_temperature_range()
            result['range'] = None if value_range is None else (float(value_range[0]), float(value_range[1]))
    except Exception as exc:
        logger.warning("Could not query the camera's cooling: %s", exc)
        result['supported'] = False
    if target_c is not None:
        if not result['supported']:
            result['error'] = 'the camera has no settable target temperature'
            logger.warning("live.target_temperature_c = %g °C ignored: %s", float(target_c), result['error'])
        else:
            try:
                camera.set_target_temperature(float(target_c))
            except Exception as exc:
                result['error'] = str(exc) or type(exc).__name__
                logger.warning("Target temperature %g °C not applied: %s", float(target_c), result['error'])
    if result['supported']:
        for key, read in (('target_c', camera.get_target_temperature), ('state', camera.cooling_state)):
            try:
                result[key] = read()
            except Exception:
                result[key] = None
        target = valid_temperature(result['target_c'])
        result['target_c'] = target
        if target_c is not None and result['error'] is None:
            logger.info("Cooler on, target %s °C (asked %g)%s", 'n/a' if target is None else f"{target:g}",
                        float(target_c), f", state {result['state']}" if result['state'] else '')
    return result


def _run_on_helper_thread(function, name: str):
    """Run ``function`` on a new thread, wait for it and return its result (or raise its error)."""
    result = {}

    def run():
        try:
            result['value'] = function()
        except BaseException as exc:           # handed to the caller
            result['error'] = exc

    thread = threading.Thread(target=run, name=name, daemon=True)
    thread.start()
    thread.join()
    if 'error' in result:
        raise result['error']
    return result.get('value')


def _close_quietly(camera) -> None:
    try:
        camera.close()
    except Exception:
        logger.exception("Closing the camera after a failed start failed too")


class CameraFrameGaps:
    """Frames the camera exposed but never delivered, counted before each received frame.

    * Frame ids (GenICam ``last_frame_info``): forward gaps, as hsi's acquisition engine counts
      them - ids restart at every AcquisitionStart and wrap, so a backward jump counts nothing.
    * Otherwise the increase of ``dropped_images_count()`` since :meth:`start` (POA's SDK counter,
      reset when the stream stops; XIMEA and GenICam always report 0). Within one stream the
      counter only grows, so a lower reading (POA reports 0 when the SDK call fails) is ignored
      instead of counting the earlier drops again.

    Call :meth:`start` right after ``start_stream`` and :meth:`missed` after every frame, on the
    camera thread.
    """

    def __init__(self, camera) -> None:
        self._camera = camera
        self._last_id: int | None = None
        self._dropped = 0

    def start(self) -> None:
        self._last_id = None
        self._dropped = self._dropped_count()

    def _dropped_count(self) -> int:
        try:
            return max(0, int(self._camera.dropped_images_count()))
        except Exception:
            return 0

    def missed(self) -> int:
        """Frames lost by the camera/link between the previous received frame and this one."""
        info = getattr(self._camera, 'last_frame_info', None)
        frame_id = info[1] if info else None
        if isinstance(frame_id, (int, np.integer)) and not isinstance(frame_id, bool):
            frame_id = int(frame_id)
            last, self._last_id = self._last_id, frame_id
            return frame_id - last - 1 if last is not None and frame_id > last else 0
        count = self._dropped_count()
        if count <= self._dropped:
            return 0
        previous, self._dropped = self._dropped, count
        return count - previous


class TemperaturePoll:
    """Sensor temperature read every ``period_s`` while the camera streams.

    Called by the reader between frames (one node / SDK read, a few ms; the frame buffers absorb
    it). ``value`` is ``None`` while the camera reports none; after three failed reads in a row
    it is not asked again.
    """

    def __init__(self, camera, period_s: float = 10.0) -> None:
        self._camera, self.period_s = camera, float(period_s)
        self._next = 0.0
        self._failures = 0
        self.value: float | None = None

    def due(self, now: float) -> bool:
        return now >= self._next

    def read(self, now: float) -> float | None:
        self._next = now + self.period_s
        try:
            self.value = valid_temperature(self._camera.get_temperature())
        except Exception:
            self.value = None
        self._failures = 0 if self.value is not None else self._failures + 1
        if self._failures >= 3:
            self._next = float('inf')
            logger.info("Sensor temperature not reported by the camera: not polled any more")
        return self.value


class CameraClock:
    """Sensor time of each live frame on the ``time.perf_counter()`` clock (for the latency).

    GenICam frames carry a camera timestamp (``last_frame_info[0]``, ns) and the camera clock can
    be latched against the host (``camera_clock_offset_ns``); then sensor time = camera time -
    offset, re-latched every ``relatch_s`` because the two clocks drift. A stamp that cannot be
    right - later than the frame's arrival, or more than ``max_age_s`` before it (wrong tick unit,
    a stale or skewed latch) - is replaced by the arrival time. After ``max_bad`` such stamps in a
    row the clock is latched again at once; if the stamps stay implausible after that, the camera
    clock is given up with a warning. A camera that cannot latch (POA, XIMEA: no timestamps
    either) uses the arrival time, which leaves exposure and transfer out of the latency; a first
    latch that fails is retried ``latch_attempts`` times, a second apart, before that.

    Use on the camera thread only (the latch talks to the camera).
    """

    def __init__(self, camera, relatch_s: float = 30.0, max_age_s: float = 5.0, max_bad: int = 20,
                 tolerance_s: float = 0.002, latch_attempts: int = 3) -> None:
        self._camera = camera
        self._relatch_s, self._max_age_s, self._max_bad, self._tolerance_s = relatch_s, max_age_s, max_bad, tolerance_s
        self._latch_attempts = latch_attempts
        self._offset_ns: int | None = None
        self._next_latch = 0.0
        self._failed_latches = 0
        self._bad = 0
        self._relatched_for_bad = False
        self._usable = True
        self.source = 'arrival'                # 'camera' while camera stamps are used

    def _latch(self, now: float) -> None:
        try:
            offset = self._camera.camera_clock_offset_ns()
        except Exception as exc:
            logger.warning("Camera clock latch failed: %s", exc)
            offset = None
        if offset is not None:
            self._offset_ns = int(offset)
            self._failed_latches = 0
            self._next_latch = now + self._relatch_s
        elif self._offset_ns is not None:          # keep the last offset until the next latch
            self._next_latch = now + self._relatch_s
        else:
            self._failed_latches += 1
            self._next_latch = now + 1.0
            if self._failed_latches >= self._latch_attempts:
                self._give_up("Camera clock not latched: latency is measured from the frame's arrival", info=True)

    def _give_up(self, message: str, *args, info: bool = False) -> None:
        self._usable = False
        self.source = 'arrival'
        (logger.info if info else logger.warning)(message, *args)

    def sensor_time(self, t_arrival: float) -> float:
        """Sensor time (``perf_counter`` seconds) of the frame just read, which arrived at ``t_arrival``."""
        if not self._usable:
            return t_arrival
        info = getattr(self._camera, 'last_frame_info', None)
        stamp = info[0] if info else None
        if not stamp:                              # None, or 0 (harvesters: producer has no timestamp)
            return t_arrival
        if t_arrival >= self._next_latch:
            self._latch(t_arrival)
        if not self._usable or self._offset_ns is None:
            return t_arrival
        t_sensor = (int(stamp) - self._offset_ns) / 1e9
        if t_arrival - self._max_age_s <= t_sensor <= t_arrival + self._tolerance_s:
            self._bad = 0
            self._relatched_for_bad = False
            self.source = 'camera'
            return t_sensor
        self._bad += 1
        if self._bad >= self._max_bad:
            if not self._relatched_for_bad:        # perhaps a stale or skewed latch: latch again first
                self._relatched_for_bad = True
                self._bad = 0
                self._next_latch = 0.0
            else:
                self._give_up("Camera timestamps do not match the host clock (last one %.3f s from its "
                              "arrival); latency is measured from the frame's arrival from now on",
                              t_sensor - t_arrival)
        return t_arrival
