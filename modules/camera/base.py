"""Base class for all camera backends (``modules/camera/base.py``).

Responsibility split:
    Subclass implements   : the ``_open_device`` / ``_close_device`` / ``_start_device`` /
                            ``_stop_device`` / ``_fetch_raw_frame`` hooks plus the
                            parameter and diagnostic methods below. No processing.
    Base class implements : open/close/stream state and the shared correction pipeline
                            (dark subtract, hot-pixel repair, saturation check) that
                            every backend gets for free.

Data conventions:
    A frame is a 2-D ``uint16`` array of shape ``(rows, columns)`` = ``(height, width)``
    = ``(slit position, spectral pixel)`` at the current hardware bin. Exposure is passed
    to ``set_params`` in milliseconds but ranges are reported in microseconds
    (``exposure_range_us``); gain is in the vendor's own units (``gain_range``).
    Temperatures are degrees Celsius, ``-999.0`` meaning unavailable.

Threading:
    The streaming calls (``open``/``close``/``start_stream``/``stop_stream``/
    ``read_frame*``/``set_params``/``set_bin``/``set_target_temperature``) belong to one
    acquisition thread.
    ``info``, ``current_bin`` and the range properties are plain reads.
    ``set_stop_event`` lets another thread abort a blocked read.

Example:
    >>> # with create_backend(config) as camera:           # doctest: +SKIP
    >>> #     camera.start_stream()                         # doctest: +SKIP
    >>> #     frame = camera.read_frame_raw(1000)           # doctest: +SKIP
"""

from abc import ABC, abstractmethod
from modules.dark import DarkProcessor
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraBackendInfo:
    """Static description of the opened camera and its current geometry.

    Attributes:
        camera_id: Vendor/device identifier (index or SDK id).
        bit_depth: Significant bits per pixel as reported by the sensor (8-16);
            frames are always returned as ``uint16``.
        pixel_size: Pixel pitch in micrometres when the SDK reports it, else ``1.0``.
        width: Frame width in pixels (columns) at the current bin.
        height: Frame height in pixels (rows) at the current bin.
        image_format: Vendor pixel-format token (enum value or string); informational.
    """

    camera_id: int
    bit_depth: int
    pixel_size: float
    width: int
    height: int
    image_format: object


def device_text(value) -> str | None:
    """Decode an SDK string (``bytes`` or ``str``) to stripped text, or ``None`` if empty."""
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).split(b"\0", 1)[0].decode("utf-8", errors="replace")
    text = str(value).strip() if value is not None else ""
    return text or None


class AbstractCameraBackend(DarkProcessor, ABC):
    """Contract every camera backend honours, plus the shared frame pipeline.

    Lifecycle: ``open()`` -> ``start_stream()`` -> ``read_frame*()`` ->
    ``stop_stream()`` -> ``close()``; also usable as a context manager. ``open`` and
    ``close`` are idempotent; ``close`` stops a running stream first. Subclasses
    implement the underscore hooks and the abstract methods; they must not override
    ``read_frame`` (the contract test checks this).

    Not thread-safe: use one acquisition thread for streaming calls.
    """

    def __init__(self):
        """Create a closed, non-streaming backend with no dark reference."""
        DarkProcessor.__init__(self)
        self._is_open: bool = False
        self._is_streaming: bool = False

    # ══════════════════════════════════════════════════════
    #  Context manager — ใช้ with ได้
    # ══════════════════════════════════════════════════════

    def __enter__(self):
        """Open the camera and return ``self``."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the camera; exceptions from the ``with`` body are not suppressed."""
        self.close()
        return False

    # ══════════════════════════════════════════════════════
    #  Public lifecycle — base class จัดการ state
    # ══════════════════════════════════════════════════════

    def open(self) -> None:
        """Connect and configure the device (no-op if already open).

        Raises:
            RuntimeError: backend specific, e.g. no camera found or the SDK refused to open.
        """
        if self._is_open:
            return
        self._open_device()
        self._is_open = True

    def close(self) -> None:
        """Stop streaming if needed and release the device (no-op if closed)."""
        if not self._is_open:
            return
        if self._is_streaming:
            self.stop_stream()
        self._close_device()
        self._is_open = False

    def start_stream(self) -> None:
        """Start hardware acquisition (no-op if already streaming).

        Raises:
            RuntimeError: the camera is not open, or the SDK failed to start.
        """
        if not self._is_open:
            raise RuntimeError("Camera not open — call open() first")
        if self._is_streaming:
            return
        self._start_device()
        self._is_streaming = True

    def stop_stream(self) -> None:
        """Stop hardware acquisition (no-op if not streaming)."""
        if not self._is_streaming:
            return
        self._stop_device()
        self._is_streaming = False

    # ══════════════════════════════════════════════════════
    #  Subclass implements — hardware-specific
    # ══════════════════════════════════════════════════════

    @abstractmethod
    def _open_device(self) -> None:
        """Hardware init: connect, configure, tune.

        Contract: after it returns, ``info``, ``exposure_range_us``, ``gain_range``,
        ``get_bins()`` and ``current_bin`` are valid, the pixel format is 16-bit
        mono, and the configured default gain/exposure have been applied.

        Raises:
            RuntimeError: no device or the SDK reported an error. A backend that fails
                half-way should release what it acquired.
        """
        pass

    @abstractmethod
    def _close_device(self) -> None:
        """Release hardware resources; must not raise and must be safe to call twice."""
        pass

    @abstractmethod
    def _start_device(self) -> None:
        """Start hardware acquisition.

        Raises:
            RuntimeError: the device is not open or the SDK refused to start.
        """
        pass

    @abstractmethod
    def _stop_device(self) -> None:
        """Stop hardware acquisition; must not raise."""
        pass

    @abstractmethod
    def _fetch_raw_frame(self, timeout_ms: int = 1000) -> np.ndarray | None:
        """Fetch one raw frame from hardware, with no processing.

        Args:
            timeout_ms: Maximum time to wait for a frame, in milliseconds.

        Returns:
            ``uint16`` array of shape ``(info.height, info.width)`` that the caller owns
            (a copy, never a view of an SDK buffer that is reused), or ``None`` on
            timeout, incomplete frame, geometry mismatch, SDK error, or when the
            stop event (see ``set_stop_event``) is set. Must not raise for those cases.
        """
        pass

    # ══════════════════════════════════════════════════════
    #  Subclass implements — params
    # ══════════════════════════════════════════════════════

    @abstractmethod
    def set_params(self, gain: int, exposure_ms: int) -> None:
        """Set gain and exposure.

        Args:
            gain: Gain in the vendor's units (see ``gain_range``).
            exposure_ms: Exposure time in milliseconds (``exposure_range_us`` is in
                microseconds). Backends may clamp to the supported range.
        """

    @abstractmethod
    def set_bin(self, bin_value: int) -> None:
        """Apply an N x N hardware bin and refresh ``info`` (width/height shrink by N).

        Args:
            bin_value: Bin factor; must be one of ``get_bins()``.

        Raises:
            ValueError: the bin is not supported.
            RuntimeError: backend specific, the camera did not apply the bin.
        """

    @property
    @abstractmethod
    def info(self) -> CameraBackendInfo:
        """CameraBackendInfo: geometry and bit depth, valid after ``open``."""

    @property
    @abstractmethod
    def exposure_range_us(self) -> tuple:
        """tuple: ``(min, max)`` exposure in microseconds."""

    @property
    @abstractmethod
    def gain_range(self) -> tuple:
        """tuple: ``(min, max)`` gain in the vendor's units."""

    @abstractmethod
    def get_bins(self) -> tuple:
        """Return the supported bin factors as a tuple of ints (empty if unknown)."""

    @property
    @abstractmethod
    def current_bin(self) -> int:
        """int: The bin factor currently applied."""

    @abstractmethod
    def dropped_images_count(self) -> int:
        """Return the number of frames the SDK reports as dropped (0 if unknown)."""

    @abstractmethod
    def get_temperature(self) -> float:
        """Return the sensor temperature in degrees Celsius, or ``-999.0`` if unavailable."""

    def model_name(self) -> str | None:
        """Return the camera model the SDK reports (e.g. ``"Uranus-C"``), or ``None`` if unknown.

        The default is ``None``; backends override it. The value is known after ``open``.
        """
        return None

    def serial_number(self) -> str | None:
        """Return the camera's serial number from the SDK, or ``None`` if unknown.

        With :meth:`model_name` it identifies the physical camera (dark references are kept per
        camera). The default is ``None``; backends override it. The value is known after ``open``.
        """
        return None

    def cooling_supported(self) -> bool:
        """Return True if the sensor temperature can be regulated to a target (TEC cooling).

        The default is False (fan-cooled or uncooled cameras); the value is known after ``open``.
        """
        return False

    def target_temperature_range(self):
        """Return ``(min, max)`` settable target temperature in °C, or ``None`` if unknown/unsupported."""
        return None

    def get_target_temperature(self):
        """Return the camera's target temperature in °C, or ``None`` if it has none."""
        return None

    def cooling_state(self):
        """Return the cooler's regulation state as short text (e.g. ``"stable"``), or ``None``.

        The default is ``None`` (no state reported); GenICam reads ``SensorTemperatureControlState``.
        Polled from the GUI thread like ``get_temperature``.
        """
        return None

    def set_target_temperature(self, temp_c: float) -> None:
        """Turn the cooler on and regulate the sensor to ``temp_c`` °C.

        Called from the acquisition thread (see ``AcquisitionEngine.request_target_temperature``).

        Raises:
            RuntimeError: the camera has no settable target temperature, or refused it.
        """
        raise RuntimeError("This camera has no settable target temperature")

    def frame_rate(self):
        """Return the camera's acquisition frame rate in Hz, or ``None`` if the SDK does not expose it.

        The default returns ``None``; backends override it (GenICam reads ``AcquisitionFrameRate``).
        """
        return None

    def saturation_level(self) -> int:
        """Return the largest value a raw frame can hold (a clipped pixel reads this value).

        The default is ``2**info.bit_depth - 1``; backends that shift fewer sensor bits into
        a 16-bit frame override it (GenICam Mono12 -> 4095 << 4 = 65520).
        """
        return (1 << int(self.info.bit_depth)) - 1

    def sensor_bit_depth(self) -> int:
        """Return how many bits of real data each pixel carries before any scaling to 16 bits.

        The default is ``info.bit_depth``; backends whose frames are scaled up override it
        (POA 12-bit sensor -> 12, GenICam 10-bit sensor in Mono12Packed -> 10).
        """
        return int(self.info.bit_depth)

    def set_stop_event(self, event) -> None:
        """Register a ``threading.Event`` that aborts blocking reads when set.

        The default implementation ignores it; backends that support it return
        ``None`` from ``_fetch_raw_frame`` while the event is set and also set it
        when the device is closed.

        Args:
            event: ``threading.Event`` shared with the acquisition worker.
        """

    # ══════════════════════════════════════════════════════
    #  Frame acquisition — shared pipeline
    # ══════════════════════════════════════════════════════

    def read_frame(self, timeout_ms: int = 1000) -> np.ndarray | None:
        """Fetch a frame and apply dark subtract + hot-pixel correction.

        Corrections are applied only while dark correction is enabled
        (``set_dark_enabled``); without a dark reference they leave the frame as is.

        Args:
            timeout_ms: Maximum wait for a frame, in milliseconds.

        Returns:
            ``uint16`` ``(height, width)`` array owned by the caller, or ``None`` if no
            frame was available (see ``_fetch_raw_frame``).
        """
        frame = self._fetch_raw_frame(timeout_ms)
        if frame is None:
            return None
        if self._dark_enabled:
            frame = self._apply_dark_subtract(frame)
            frame = self._apply_hot_pixel(frame)
        return frame

    def read_frame_profiled(self, profiler, timeout_ms: int = 1000):
        """Time acquisition and corrections separately, preserving read_frame.

        Backend-specific read_frame overrides retain their existing behavior.
        Their timing is acquisition-only in the current XIMEA implementation.

        Args:
            profiler: Object with ``time(name)`` (context manager) and ``_record(name,
                seconds)``; stages recorded are ``fetch``, ``dark`` and ``hot_pixel``.
            timeout_ms: Maximum wait for a frame, in milliseconds.

        Returns:
            Same as :meth:`read_frame`.
        """
        if type(self).read_frame is not AbstractCameraBackend.read_frame:
            with profiler.time("fetch"):
                return self.read_frame(timeout_ms=timeout_ms)
        with profiler.time("fetch"):
            frame = self._fetch_raw_frame(timeout_ms)
        if frame is None:
            return None
        if self._dark_enabled:
            with profiler.time("dark"):
                frame = self._apply_dark_subtract(frame)
            with profiler.time("hot_pixel"):
                frame = self._apply_hot_pixel(frame)
        else:
            # Record zero for skipped operations, avoiding stale active timings.
            profiler._record("dark", 0.0)
            profiler._record("hot_pixel", 0.0)
        return frame

    def read_frame_raw(self, timeout_ms: int = 1000) -> np.ndarray | None:
        """Fetch a frame with no correction applied.

        Args:
            timeout_ms: Maximum wait for a frame, in milliseconds.

        Returns:
            ``uint16`` ``(height, width)`` array owned by the caller, or ``None``.
        """
        return self._fetch_raw_frame(timeout_ms)

    #: ``(camera_time_ns, frame_id)`` of the last frame returned by a read, where the
    #: backend knows them (GenICam); ``None`` otherwise.
    last_frame_info = None

    def camera_clock_offset_ns(self):
        """``camera_time_ns - time.perf_counter_ns()``, or ``None`` if the camera cannot latch its clock.

        Backends with per-frame camera timestamps (GenICam) override this; call it from
        the acquisition thread.
        """
        return None

    def read_frame_raw_into(self, provide, timeout_ms: int = 1000) -> np.ndarray | None:
        """``read_frame_raw`` that may write into a caller-supplied buffer.

        A backend that converts the camera buffer anyway (GenICam) writes straight into
        ``provide(shape)`` so a GPU can read the frame in place.  This default ignores
        ``provide`` and returns ``read_frame_raw``'s new array.

        Args:
            provide: Callable ``(shape) -> numpy.ndarray | None`` giving a writable
                ``uint16`` buffer, or ``None`` for a new array.
            timeout_ms: Maximum wait for a frame, in milliseconds.

        Returns:
            The frame (possibly the provided buffer, overwritten by the next call), or ``None``.
        """
        return self.read_frame_raw(timeout_ms=timeout_ms)

    # ══════════════════════════════════════════════════════
    #  Dark / hot-pixel — shared
    # ══════════════════════════════════════════════════════






    # ══════════════════════════════════════════════════════
    #  Diagnostics — shared
    # ══════════════════════════════════════════════════════

    def detect_saturation(self, raw_frame):
        """Flag saturated pixels.

        Args:
            raw_frame: Frame array (any shape) in the sensor's native scale.

        Returns:
            Boolean array, same shape: ``raw_frame >= 2**info.bit_depth - 2``.
            Note: the threshold uses the reported bit depth, so a backend that
            scales data into the 16-bit range (GenICam) is compared on that scale.
        """
        bit_depth = self.info.bit_depth
        threshold = (1 << bit_depth) - 2
        return raw_frame >= threshold