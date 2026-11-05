# modules/camera/ximea_camera.py
"""XIMEA camera backend (xiAPI).

Follows the same contract as the POA and GenICam backends: the base class owns
open/close/stream state and the shared correction pipeline (dark, hot pixel);
this class only implements the hardware-specific ``_open_device``,
``_close_device``, ``_start_device``, ``_stop_device`` and ``_fetch_raw_frame``.
``read_frame`` is *not* overridden, so ``read_frame_raw`` returns the untouched
sensor frame and the shared preprocessing pipeline applies to XIMEA too.

Data conventions:
    Frames are ``uint16`` ``(height, width)`` arrays in ``XI_MONO16`` format at the
    current bin, not scaled (a 10-bit camera gives 0..1023); ``info.bit_depth`` is xiAPI's
    ``image_data_bit_depth`` (10 if it cannot be read, as the original SWIR app). Exposure is
    set in ms (clamped, sent to xiAPI as integer us), ranges are in us; gain is in xiAPI
    units (dB). Binning uses ``XI_BINNING`` downsampling (``XI_DWN_NxN``).

Threading:
    SDK calls are serialised by an internal lock, so ``get_temperature`` and
    ``dropped_images_count`` are safe from the GUI thread (they never block behind a fetch).
    Streaming calls still belong to one acquisition thread.
"""

from __future__ import annotations

import logging
import re
import threading

import numpy as np

from .ximea import xiapi
from .base import AbstractCameraBackend, CameraBackendInfo, device_text

logger = logging.getLogger(__name__)

_TEMPERATURE_UNAVAILABLE = -999.0
_LEGACY_BIT_DEPTH = 10      # the original SWIR app (QTNarit.py) used 10 bits


def _xi_bits(value) -> int:
    """Bits of an xiAPI bit-depth value: ``'XI_BPP_10'`` (what xiAPI returns) or ``10`` -> 10."""
    if isinstance(value, bytes):
        value = value.decode()
    match = re.search(r"(\d+)", str(value))
    if match is None:
        raise ValueError("not an xiAPI bit depth: %r" % (value,))
    return int(match.group(1))


class XIMEACameraBackend(AbstractCameraBackend):
    """Backend for XIMEA cameras via xiAPI."""

    def __init__(
        self,
        camera_id: int = 0,
        gain: float = 0,
        exposure_ms: float = 100,
        default_bin: int | None = None,
    ):
        """Create a closed backend.

        Args:
            camera_id: xiAPI device index.
            gain: Initial gain in xiAPI units, clamped at ``open``.
            exposure_ms: Initial exposure in milliseconds.
            default_bin: Bin applied at ``open``; failure is logged, not raised.
        """
        super().__init__()
        self.camera_id = int(camera_id)
        self.gain = gain
        self.exposure_ms = exposure_ms
        self.default_bin = default_bin

        self._camera = None
        self._image = None
        self._info = CameraBackendInfo(
            camera_id=self.camera_id,
            bit_depth=_LEGACY_BIT_DEPTH,
            pixel_size=1.0,
            width=0,
            height=0,
            image_format="XI_MONO16",
        )
        self._exposure_range_us: tuple = (0, 10_000_000)
        self._gain_range: tuple = (0, 0)
        self._bins: tuple = ()
        self._current_bin: int = 1
        self._last_temperature: float = _TEMPERATURE_UNAVAILABLE
        self._cooled: bool = False
        self._target_temp_range: tuple | None = None
        self._model_name: str | None = None
        self._serial_number: str | None = None
        self._stop_event: threading.Event | None = None
        # Serialises SDK calls; fetch holds it for at most its timeout.
        self._lock = threading.RLock()

    # ══════════════════════════════════════════════════════
    #  Properties
    # ══════════════════════════════════════════════════════

    @property
    def current_bin(self) -> int:
        """int: Bin currently applied."""
        return self._current_bin

    @property
    def exposure_range_us(self) -> tuple:
        """tuple: ``(min, max)`` exposure in us, read at ``open``."""
        return self._exposure_range_us

    @property
    def gain_range(self) -> tuple:
        """tuple: ``(min, max)`` gain in xiAPI units, read at ``open``."""
        return self._gain_range

    def get_bins(self) -> tuple:
        """Return the XI_BINNING factors between the camera's minimum and maximum."""
        return self._bins

    @property
    def info(self) -> CameraBackendInfo:
        """CameraBackendInfo: sensor bit depth, pixel size and current width/height."""
        return self._info

    def model_name(self) -> str | None:
        """xiAPI ``device_name`` read at ``open`` (``None`` before, or if unavailable)."""
        return self._model_name

    def serial_number(self) -> str | None:
        """xiAPI ``device_sn`` read at ``open`` (``None`` before, or if unavailable)."""
        return self._serial_number

    # ══════════════════════════════════════════════════════
    #  Internal helpers (call with self._lock held)
    # ══════════════════════════════════════════════════════

    def _pixel_size(self) -> float:
        try:
            return float(self._camera.get_pixel_size())
        except Exception:
            return 1.0

    def _refresh_info(self) -> None:
        cam = self._camera
        if cam is None:
            return
        try:
            bit_depth = _xi_bits(cam.get_image_data_bit_depth())    # bits in the XI_MONO16 buffer
        except Exception:
            bit_depth = self._info.bit_depth
        try:
            width, height = int(cam.get_width()), int(cam.get_height())
        except Exception:
            width, height = int(cam.get_aeag_roi_width()), int(cam.get_aeag_roi_height())
        self._info = CameraBackendInfo(
            camera_id=self.camera_id,
            bit_depth=bit_depth,
            pixel_size=self._pixel_size(),
            width=width,
            height=height,
            image_format="XI_MONO16",
        )

    def _collect_bins(self) -> tuple:
        cam = self._camera
        if cam is None:
            return ()
        try:
            if cam.get_downsampling_type() != "XI_BINNING":
                cam.set_downsampling_type("XI_BINNING")
            low = cam.get_downsampling_minimum()
            high = cam.get_downsampling_maximum()
        except Exception as e:
            logger.debug("Unable to query XIMEA binning support: %s", e)
            return ()

        def parse(name: str) -> int | None:
            try:
                return int(name.split("_")[-1].split("x")[0])
            except (IndexError, ValueError, AttributeError):
                return None

        lo, hi = parse(low), parse(high)
        if lo is None or hi is None or lo > hi:
            return ()
        return tuple(range(lo, hi + 1))

    def _read_cooling_support(self) -> None:
        """``iscooled`` and the ``target_temp`` range (fan-cooled models: not cooled)."""
        cam = self._camera
        self._cooled = False
        self._target_temp_range = None
        try:
            self._cooled = bool(cam.is_iscooled())
        except Exception as e:
            logger.debug("XIMEA iscooled unavailable: %s", e)
        if self._cooled:
            try:
                self._target_temp_range = (float(cam.get_target_temp_minimum()),
                                           float(cam.get_target_temp_maximum()))
            except Exception as e:
                logger.debug("XIMEA target_temp range unavailable: %s", e)
        logger.info("XIMEA cooled: %s (target range %s)", self._cooled, self._target_temp_range)

    def _apply_params(self, gain: float, exposure_ms: float) -> None:
        if self._camera is None:
            logger.warning("set_params called but camera is not initialized")
            return
        gain = float(max(self._gain_range[0], min(float(gain), self._gain_range[1])))
        exposure_us = int(max(
            self._exposure_range_us[0],
            min(float(exposure_ms) * 1000.0, self._exposure_range_us[1]),
        ))
        self._camera.set_gain(gain)
        self._camera.set_exposure(exposure_us)
        self.gain = gain
        self.exposure_ms = exposure_us / 1000.0
        logger.info("XIMEA params: gain=%s exposure_ms=%s", self.gain, self.exposure_ms)

    # ══════════════════════════════════════════════════════
    #  Params
    # ══════════════════════════════════════════════════════

    def set_params(self, gain: float, exposure_ms: float) -> None:
        """Clamp and apply gain and exposure (thread-safe).

        Args:
            gain: Gain in xiAPI units, clamped to ``gain_range``.
            exposure_ms: Exposure in ms, clamped to ``exposure_range_us`` and truncated to
                whole microseconds.
        """
        with self._lock:
            self._apply_params(gain, exposure_ms)

    def set_bin(self, bin_value: int) -> None:
        """Apply an N x N ``XI_BINNING`` downsampling and verify it.

        Does nothing if the camera is not open.

        Args:
            bin_value: One of ``get_bins()``.

        Raises:
            ValueError: the bin is not supported.
            RuntimeError: the camera reports a different bin afterwards.
        """
        with self._lock:
            if self._camera is None:
                return
            desired_bin = int(bin_value)
            if not self._bins:
                self._bins = self._collect_bins()
            if self._bins and desired_bin not in self._bins:
                raise ValueError(
                    "Unsupported bin value: %d. Supported: %s"
                    % (desired_bin, self._bins)
                )
            self._camera.set_downsampling_type("XI_BINNING")
            self._camera.set_downsampling("XI_DWN_%dx%d" % (desired_bin, desired_bin))
            self._refresh_info()
            try:
                reported = self._camera.get_downsampling()
                reported_type = self._camera.get_downsampling_type()
            except Exception:
                reported = reported_type = None
            logger.info(
                "XIMEA bin %d requested: camera reports %s / %s, frame %dx%d",
                desired_bin, reported_type, reported, self._info.width, self._info.height)
            try:
                applied = int(str(reported).split("_")[-1].split("x")[0])
            except (ValueError, IndexError):
                applied = None
            if applied is not None and applied != desired_bin:
                raise RuntimeError(
                    "XIMEA did not apply bin %d (camera reports %s)" % (desired_bin, reported))
            self._current_bin = desired_bin

    # ══════════════════════════════════════════════════════
    #  Hardware lifecycle — base class calls these
    # ══════════════════════════════════════════════════════

    def _open_device(self) -> None:
        """Open the device, select XI_MONO16, read ranges/bins and apply defaults.

        Raises:
            Exception: whatever xiAPI raises; the device is closed again first.
        """
        with self._lock:
            camera = xiapi.Camera(self.camera_id)
            camera.open_device()
            self._camera = camera
            try:
                camera.set_imgdataformat("XI_MONO16")
                self._image = xiapi.Image()
                try:
                    self._model_name = device_text(camera.get_device_name())
                except Exception as e:
                    logger.debug("XIMEA device_name unavailable: %s", e)
                try:
                    self._serial_number = device_text(camera.get_device_sn())
                except Exception as e:
                    logger.debug("XIMEA device_sn unavailable: %s", e)

                self._exposure_range_us = (
                    int(camera.get_exposure_minimum()),
                    int(camera.get_exposure_maximum()),
                )
                self._gain_range = (
                    float(camera.get_gain_minimum()),
                    float(camera.get_gain_maximum()),
                )
                self._bins = self._collect_bins()
                self._refresh_info()
                self._read_cooling_support()
                self._apply_params(self.gain, self.exposure_ms)

                if self.default_bin is not None:
                    try:
                        self.set_bin(int(self.default_bin))
                    except Exception as e:
                        logger.warning(
                            "Could not apply default_bin=%s: %s", self.default_bin, e)
            except Exception:
                self._close_device()
                raise

            logger.info(
                "XIMEA %s: exposure %s~%s us, gain %s~%s, %dx%d, %d-bit, bins=%s",
                self.camera_id, self._exposure_range_us[0], self._exposure_range_us[1],
                self._gain_range[0], self._gain_range[1],
                self._info.width, self._info.height, self._info.bit_depth, self._bins,
            )

    def _close_device(self) -> None:
        """Set the stop event and close the device; errors are swallowed."""
        if self._stop_event is not None:
            self._stop_event.set()
        with self._lock:
            camera, self._camera = self._camera, None
            self._image = None
            if camera is not None:
                try:
                    camera.close_device()
                except Exception:
                    pass

    def _start_device(self) -> None:
        """Start xiAPI acquisition.

        Raises:
            RuntimeError: the camera is not open.
        """
        with self._lock:
            if self._camera is None:
                raise RuntimeError("XIMEA camera is not opened")
            self._camera.start_acquisition()

    def _stop_device(self) -> None:
        """Stop xiAPI acquisition; errors are swallowed."""
        with self._lock:
            if self._camera is not None:
                try:
                    self._camera.stop_acquisition()
                except Exception:
                    pass

    # ══════════════════════════════════════════════════════
    #  Fetch — raw sensor frame only
    # ══════════════════════════════════════════════════════

    def _fetch_raw_frame(self, timeout_ms: int = 1000) -> np.ndarray | None:
        """Wait for the next image and copy it out of the SDK buffer.

        Args:
            timeout_ms: Maximum wait in milliseconds.

        Returns:
            ``uint16`` ``(height, width)`` copy owned by the caller, or ``None`` on timeout,
            error, closed camera or stop event. ``info`` is refreshed if the size changed.
        """
        if self._stop_event is not None and self._stop_event.is_set():
            return None
        with self._lock:
            camera, image = self._camera, self._image
            if camera is None or image is None:
                return None
            try:
                camera.get_image(image, int(timeout_ms))
                height, width = int(image.height), int(image.width)
                data = image.get_image_data_numpy()
                # Own the data: the SDK image buffer is reused for the next frame.
                frame = np.array(data, dtype=np.uint16, copy=True).reshape(height, width)
            except Exception as e:
                logger.debug("XIMEA fetch failed: %s", e)
                return None

            if (height, width) != (self._info.height, self._info.width):
                self._refresh_info()
            return frame

    # ══════════════════════════════════════════════════════
    #  Misc
    # ══════════════════════════════════════════════════════

    def set_stop_event(self, event: threading.Event) -> None:
        """Register an event that makes reads return ``None`` and is set on close."""
        self._stop_event = event

    def dropped_images_count(self) -> int:
        """Return the SDK dropped-frame count; 0 if unsupported or a fetch holds the lock."""
        if not self._lock.acquire(blocking=False):
            return 0
        try:
            camera = self._camera
            if camera is not None and hasattr(camera, "get_dropped_images_count"):
                try:
                    return int(camera.get_dropped_images_count())
                except Exception:
                    return 0
            return 0
        finally:
            self._lock.release()

    def get_temperature(self) -> float:
        """Return the chip temperature in degrees Celsius.

        Never blocks the GUI behind a frame fetch: returns the last value if busy.
        ``-999.0`` means unavailable (closed camera or no temperature getter).
        """
        if not self._lock.acquire(blocking=False):
            return self._last_temperature
        try:
            camera = self._camera
            if camera is None:
                return _TEMPERATURE_UNAVAILABLE
            for getter in ("get_chip_temp", "get_temp"):
                try:
                    value = float(getattr(camera, getter)())
                except Exception:
                    continue
                self._last_temperature = value
                return value
            return _TEMPERATURE_UNAVAILABLE
        finally:
            self._lock.release()

    def cooling_supported(self) -> bool:
        """True for cooled XIMEA models (``iscooled``); known after ``open``."""
        return self._cooled and self._camera is not None

    def target_temperature_range(self):
        """``(min, max)`` of ``target_temp`` in °C, or ``None``."""
        return self._target_temp_range if self.cooling_supported() else None

    def get_target_temperature(self):
        """``target_temp`` in °C, or ``None`` when not cooled, unreadable or the SDK is busy."""
        if not self.cooling_supported() or not self._lock.acquire(blocking=False):
            return None
        try:
            return float(self._camera.get_target_temp())
        except Exception:
            return None
        finally:
            self._lock.release()

    def set_target_temperature(self, temp_c: float) -> None:
        """Switch temperature control to automatic and set ``target_temp`` (thread-safe).

        Raises:
            RuntimeError: the camera is not cooled.
            Exception: whatever xiAPI raises for a refused value.
        """
        with self._lock:
            if not self.cooling_supported():
                raise RuntimeError("This XIMEA camera is not cooled")
            self._camera.set_cooling("XI_TEMP_CTRL_MODE_AUTO")
            self._camera.set_target_temp(float(temp_c))
            logger.info("XIMEA cooling on, target %.1f C", float(temp_c))
