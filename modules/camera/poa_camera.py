# modules/camera/poa_camera.py
"""POA (Player One) camera backend over the vendor ``PlayerOneCamera.dll`` wrapper.

Data conventions:
    Frames are ``uint16`` ``(height, width)`` arrays (RAW16, first channel if the SDK
    returns 3-D data) at the current bin. Exposure is set in ms, reported in us; gain is in
    POA units. ``bit_depth`` is reported as 16 for 10/12/16-bit sensors.

Threading:
    One acquisition thread for ``open``/``start_stream``/``read_frame*``/``set_params``/
    ``set_bin``; ``set_stop_event`` may be given an event set from another thread.

Quirks:
    Only the first enumerated camera (index 0) is opened; ``open`` forces RAW16, offset 20,
    USB bandwidth 90 and a default bin of 2 (else 1, else the first supported).
"""

from __future__ import annotations
import logging
import time
import threading
import numpy as np
from .poa import pyPOACamera
from .base import AbstractCameraBackend, CameraBackendInfo, device_text

logger = logging.getLogger(__name__)


class POACameraBackend(AbstractCameraBackend):
    """Backend for POA cameras; see the module docstring for conventions and quirks."""

    def __init__(self, gain: int = 0, exposure_ms: int = 100):
        """Create a closed backend.

        Args:
            gain: Initial gain in POA units, applied at ``open``.
            exposure_ms: Initial exposure in milliseconds, applied at ``open``.
        """
        super().__init__()
        self.gain = gain
        self.exposure_ms = exposure_ms
        self.camera_id = -1
        self.bit_depth = 8
        self.pixel_size = 1.0
        self.img_height = 0
        self.img_width = 0
        self.img_format = None
        self.buf_array = None
        self._exposure_range_us: tuple = (0, 10_000_000)
        self._gain_range: tuple = (0, 255)
        self._bins: tuple = ()
        self._sensor_size: tuple = (0, 0)   # (maxWidth, maxHeight) at bin 1
        self._current_bin: int = 1
        self._stop_event: threading.Event | None = None
        self._has_cooler: bool = False
        self._target_temp_range: tuple | None = None
        self._model_name: str | None = None
        self._serial_number: str | None = None

    # ══════════════════════════════════════════════════════
    #  Properties
    # ══════════════════════════════════════════════════════

    @property
    def current_bin(self) -> int:
        """int: Bin currently applied (1 before ``open``)."""
        return self._current_bin

    @property
    def exposure_range_us(self) -> tuple:
        """tuple: ``(min, max)`` exposure in us, read from the camera at ``open``."""
        return self._exposure_range_us

    @property
    def gain_range(self) -> tuple:
        """tuple: ``(min, max)`` gain in POA units, read at ``open``."""
        return self._gain_range

    def get_bins(self) -> tuple:
        """Return the bins the camera reports (empty before ``open``)."""
        return self._bins

    @property
    def info(self) -> CameraBackendInfo:
        """CameraBackendInfo: a snapshot built from the current attributes."""
        return CameraBackendInfo(
            camera_id=self.camera_id,
            bit_depth=self.bit_depth,
            pixel_size=self.pixel_size,
            width=self.img_width,
            height=self.img_height,
            image_format=self.img_format,
        )

    def sensor_bit_depth(self) -> int:
        """Bit depth the SDK reports for the sensor (e.g. 12), before RAW16 scales it to 16 bits."""
        return int(getattr(self, "_sensor_bits", self.bit_depth))

    def model_name(self) -> str | None:
        """``cameraModelName`` from the camera properties (``None`` before ``open``)."""
        return self._model_name

    def serial_number(self) -> str | None:
        """``SN`` from the camera properties (``None`` before ``open``)."""
        return self._serial_number

    # ══════════════════════════════════════════════════════
    #  Internal helpers
    # ══════════════════════════════════════════════════════

    def _allocate_image_buffer(self) -> None:
        img_size = pyPOACamera.ImageCalcSize(
            self.img_height, self.img_width, self.img_format)
        self.buf_array = np.zeros(img_size, dtype=np.uint8)

    def _refresh_image_state(self) -> None:
        _, self.img_width, self.img_height = pyPOACamera.GetImageSize(
            self.camera_id)
        _, self.img_format = pyPOACamera.GetImageFormat(self.camera_id)
        self._allocate_image_buffer()

    # ══════════════════════════════════════════════════════
    #  Params
    # ══════════════════════════════════════════════════════

    def set_params(self, gain: int, exposure_ms: int) -> None:
        """Store gain/exposure and, if open, send them to the camera.

        Args:
            gain: Gain in POA units (not clamped here).
            exposure_ms: Exposure in ms, sent to the SDK as microseconds (rounded).
        """
        self.gain = gain
        self.exposure_ms = exposure_ms
        if self.camera_id > -1:
            pyPOACamera.SetGain(self.camera_id, int(round(gain)), False)
            pyPOACamera.SetExp(
                self.camera_id, int(round(exposure_ms * 1000)), False)

    def set_bin(self, bin_value: int) -> None:
        """Apply a hardware bin and re-read the image size and buffer.

        Does nothing if the camera is not open.

        Args:
            bin_value: One of ``get_bins()``.

        Raises:
            ValueError: the bin is not supported by the camera.
        """
        if self.camera_id < 0:
            return
        desired_bin = int(bin_value)
        if self._bins and desired_bin not in self._bins:
            raise ValueError(f"Unsupported bin value: {desired_bin}")

        pyPOACamera.SetImageBin(self.camera_id, desired_bin)
        # The SDK keeps the *previous* ROI (rescaled and aligned), so the frame size would
        # drift with every bin change (e.g. 804 -> 800 -> 796 columns at bin 2). Always return
        # to the full sensor at the new bin so each bin has one fixed, reproducible size
        # (the one dark references, Smile models and wavelength profiles are made for).
        width, height = self._sensor_size
        if width and height:
            error = pyPOACamera.SetImageSize(self.camera_id, width // desired_bin, height // desired_bin)
            if error != pyPOACamera.POAErrors.POA_OK:
                logger.warning("SetImageSize for bin %d failed: %s", desired_bin, pyPOACamera.GetErrorString(error))
        self._current_bin = desired_bin
        self._refresh_image_state()

    # ══════════════════════════════════════════════════════
    #  Hardware lifecycle — base class calls these
    # ══════════════════════════════════════════════════════

    def _open_device(self) -> None:
        """Open camera 0, read ranges and bins, force RAW16 and apply the default bin.

        Raises:
            RuntimeError: no camera, or the SDK failed to open/initialise it.
        """
        camera_count = pyPOACamera.GetCameraCount()
        if camera_count <= 0:
            raise RuntimeError("No POA camera found")

        _, camera_properties = pyPOACamera.GetCameraProperties(0)

        self.camera_id = camera_properties.cameraID
        self._model_name = device_text(getattr(camera_properties, "cameraModelName", None))
        self._serial_number = device_text(getattr(camera_properties, "SN", None))
        self.bit_depth = camera_properties.bitDepth
        self._sensor_bits = int(camera_properties.bitDepth)
        if self.bit_depth in (10, 12, 16):
            self.bit_depth = 16
        self.pixel_size = camera_properties.pixelSize
        self._bins = tuple(int(v) for v in camera_properties.bins)
        self._sensor_size = (int(camera_properties.maxWidth), int(camera_properties.maxHeight))

        camera_error = pyPOACamera.OpenCamera(self.camera_id)
        if camera_error != pyPOACamera.POAErrors.POA_OK:
            raise RuntimeError(pyPOACamera.GetErrorString(camera_error))

        camera_error = pyPOACamera.InitCamera(self.camera_id)
        if camera_error != pyPOACamera.POAErrors.POA_OK:
            raise RuntimeError(pyPOACamera.GetErrorString(camera_error))

        # ranges
        _, exposure_attr = pyPOACamera.GetConfigAttributesByConfigID(
            self.camera_id, pyPOACamera.POAConfig.POA_EXPOSURE)
        self._exposure_range_us = (
            int(exposure_attr.minValue), int(exposure_attr.maxValue))

        _, gain_attr = pyPOACamera.GetConfigAttributesByConfigID(
            self.camera_id, pyPOACamera.POAConfig.POA_GAIN)
        self._gain_range = (int(gain_attr.minValue), int(gain_attr.maxValue))

        # cooled models (isHasCooler) regulate to POA_TARGET_TEMP; fan models only report a temperature
        self._has_cooler = bool(getattr(camera_properties, "isHasCooler", 0))
        self._target_temp_range = None
        if self._has_cooler:
            error, temp_attr = pyPOACamera.GetConfigAttributesByConfigID(
                self.camera_id, pyPOACamera.POAConfig.POA_TARGET_TEMP)
            if error == pyPOACamera.POAErrors.POA_OK:
                self._target_temp_range = (float(temp_attr.minValue), float(temp_attr.maxValue))
        logger.info("POA cooler: %s (target range %s)", self._has_cooler, self._target_temp_range)

        # defaults
        self.set_params(self.gain, self.exposure_ms)

        pyPOACamera.SetConfig(
            self.camera_id, pyPOACamera.POAConfig.POA_OFFSET, 20, False)
        pyPOACamera.SetConfig(
            self.camera_id, pyPOACamera.POAConfig.POA_USB_BANDWIDTH_LIMIT, 90, False)
        pyPOACamera.SetConfig(
            self.camera_id, pyPOACamera.POAConfig.POA_PIXEL_BIN_SUM, 0, False)

        pyPOACamera.SetImageFormat(
            self.camera_id, pyPOACamera.POAImgFormat.POA_RAW16)

        # default bin
        if 2 in self._bins:
            default_bin = 2
        elif 1 in self._bins:
            default_bin = 1
        elif self._bins:
            default_bin = self._bins[0]
        else:
            default_bin = 1
        self.set_bin(default_bin)

    def _close_device(self) -> None:
        """Set the stop event, close the camera and forget its id."""
        if self._stop_event is not None:
            self._stop_event.set()

        if self.camera_id > -1:
            try:
                pyPOACamera.CloseCamera(self.camera_id)
            except Exception:
                pass

        self.camera_id = -1

    def _start_device(self) -> None:
        """Start exposure (continuous mode).

        Raises:
            RuntimeError: the SDK refused to start.
        """
        camera_error = pyPOACamera.StartExposure(self.camera_id, False)
        if camera_error != pyPOACamera.POAErrors.POA_OK:
            raise RuntimeError(f"StartExposure failed: {camera_error}")

    def _stop_device(self) -> None:
        """Stop exposure; SDK errors are swallowed."""
        if self.camera_id > -1:
            try:
                pyPOACamera.StopExposure(self.camera_id)
            except Exception:
                pass

    # ══════════════════════════════════════════════════════
    #  Fetch
    # ══════════════════════════════════════════════════════

    def _fetch_raw_frame(self, timeout_ms: int = 1000) -> np.ndarray | None:
        """Poll until a frame is ready and return it.

        Args:
            timeout_ms: Maximum wait in milliseconds (polls every 2 ms).

        Returns:
            Contiguous ``uint16`` ``(height, width)`` copy owned by the caller, or ``None``
            on timeout, SDK error, closed camera or stop event.
        """
        if self.camera_id < 0:
            return None

        deadline = time.monotonic() + timeout_ms / 1000.0

        while time.monotonic() < deadline:
            if self._stop_event is not None and self._stop_event.is_set():
                return None

            try:
                _, p_is_ready = pyPOACamera.ImageReady(self.camera_id)
            except Exception:
                return None

            if p_is_ready:
                try:
                    _, live_w, live_h = pyPOACamera.GetImageSize(
                        self.camera_id)
                    if (live_w, live_h) != (self.img_width, self.img_height):
                        self.img_width, self.img_height = live_w, live_h
                        self._allocate_image_buffer()

                    pyPOACamera.GetImageData(
                        self.camera_id, self.buf_array, timeout_ms)

                    frame_nd = pyPOACamera.ImageDataConvert(
                        self.buf_array, self.img_height,
                        self.img_width, self.img_format,
                    )

                    if frame_nd is None:
                        return None

                    if frame_nd.ndim == 3:
                        frame = frame_nd[:, :, 0]
                    elif frame_nd.ndim == 2:
                        frame = frame_nd
                    else:
                        return None

                    self.img_height, self.img_width = frame.shape
                    return np.ascontiguousarray(frame)

                except Exception:
                    return None

            time.sleep(0.002)

        return None

    # ══════════════════════════════════════════════════════
    #  Misc
    # ══════════════════════════════════════════════════════

    def set_stop_event(self, event: threading.Event):
        """Register an event that makes reads return ``None`` and is set on close."""
        self._stop_event = event

    def dropped_images_count(self) -> int:
        """Return the SDK's dropped-frame counter (0 when closed)."""
        if self.camera_id < 0:
            return 0
        _, drop_count = pyPOACamera.GetDroppedImagesCount(self.camera_id)
        return drop_count

    def get_temperature(self) -> float:
        """Return the sensor temperature in degrees Celsius, ``-999.0`` if unavailable."""
        if self.camera_id < 0:
            return -999.0
        try:
            camera_error, temp = pyPOACamera.GetCameraTEMP(self.camera_id)
            if camera_error == pyPOACamera.POAErrors.POA_OK:
                return float(temp)
        except Exception:
            pass
        return -999.0

    def cooling_supported(self) -> bool:
        """True for POA models with a cooler (``isHasCooler``); known after ``open``."""
        return self._has_cooler and self.camera_id > -1

    def target_temperature_range(self):
        """``(min, max)`` of ``POA_TARGET_TEMP`` in °C, or ``None``."""
        return self._target_temp_range if self.cooling_supported() else None

    def get_target_temperature(self):
        """``POA_TARGET_TEMP`` in °C, or ``None`` without a cooler or on an SDK error."""
        if not self.cooling_supported():
            return None
        error, value, _ = pyPOACamera.GetConfig(self.camera_id, pyPOACamera.POAConfig.POA_TARGET_TEMP)
        return float(value) if error == pyPOACamera.POAErrors.POA_OK else None

    def set_target_temperature(self, temp_c: float) -> None:
        """Set ``POA_TARGET_TEMP`` (whole °C) and switch the cooler on.

        Raises:
            RuntimeError: no cooler, or the SDK refused a value.
        """
        if not self.cooling_supported():
            raise RuntimeError("This POA camera has no cooler")
        for config, value in ((pyPOACamera.POAConfig.POA_TARGET_TEMP, int(round(temp_c))),
                              (pyPOACamera.POAConfig.POA_COOLER, 1)):
            error = pyPOACamera.SetConfig(self.camera_id, config, value, False)
            if error != pyPOACamera.POAErrors.POA_OK:
                raise RuntimeError(f"{config.name}={value}: {pyPOACamera.GetErrorString(error)}")