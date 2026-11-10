# modules/camera/genicam.py
"""GenICam / GenTL (GigE Vision) camera backend built on ``harvesters``.

Requires the ``harvesters`` package and a GenTL producer ``.cti`` file (``cti_path``, or
``libsv_gev_tl_x64.cti`` under ``$GENICAM_GENTL64_PATH``).

Data conventions:
    Frames are ``uint16`` ``(height, width)`` copies, like every backend. ``open`` writes the
    vendor ``SensorPixelSize`` node (``SensorBpp8/10/12``, from ``sensor_pixel_size``; the
    original code always used 10). How many bits each pixel carries is taken from the
    transmitted ``PixelFormat`` (or the buffer's ``data_format``) when it names 8/10/12/14 bits;
    when it is ``Mono16``/unknown, from the ``SensorPixelSize`` read back; else 12. Buffers are
    ``Mono12Packed`` (the SVS fxo992's only format besides ``Mono8``) buffers are fetched raw and
    unpacked by :func:`unpack_mono12packed` in one pass instead of harvesters' expansion (7-10 ms
    per 1024 x 1280 frame, which capped the reader below the camera's 121.6 fps):

    * 8 bits  -> values unchanged (0..255), ``bit_depth`` 8;
    * 10/12/14 bits -> shifted left into the 16-bit range (10 ``<< 6``, 12 ``<< 4``,
      14 ``<< 2``), ``bit_depth`` 16;
    * 16 bits -> unchanged, ``bit_depth`` 16.

    ``bit_depth`` is the scale of the delivered values, so display and saturation maths match.
    Mono12 output is identical to the original code. Exposure is set in ms, ranges are
    reported in us; gain is in the camera's Gain node units.

Threading:
    One acquisition thread for ``open``/``start_stream``/``read_frame*``/``set_params``/
    ``set_bin``; ``set_stop_event`` may be given an event set from another thread.

Quirks:
    ``set_bin`` retries a refused bin with a minimal ROI and restores the full frame
    afterwards; a ``default_bin`` that cannot be applied at ``open`` only prints a warning.
"""
import logging
import os
import re
import threading
import numpy as np
import faulthandler
import time
from harvesters.core import Harvester
from .base import AbstractCameraBackend, CameraBackendInfo, device_text

try:
    from genicam.genapi import EAccessMode
    _WRITABLE_ACCESS = (int(EAccessMode.RW), int(EAccessMode.WO))
except Exception:                       # older genicam package: writability found by trying
    _WRITABLE_ACCESS = None

try:
    from harvesters.util.pfnc import dict_by_ints as _PFNC_NAMES   # PFNC value -> name, as harvesters
except Exception:
    _PFNC_NAMES = {}

#: Unpacked formats read in place from the raw buffer: name -> bytes per pixel.
_UNPACKED_BYTES = {"Mono8": 1, "Mono10": 2, "Mono12": 2, "Mono14": 2, "Mono16": 2}


def _unpack_mono12packed_numpy(src, out, shift):
    """Mono12Packed rows ``src`` (uint8, 3 bytes per 2 pixels) -> ``out`` (uint16) << ``shift``."""
    h, w = out.shape
    b = src[:, :3 * (w // 2)].reshape(h, w // 2, 3)
    mid = b[..., 1].astype(np.uint16)
    out[:, 0::2] = ((b[..., 0].astype(np.uint16) << 4) | (mid & 0xF)) << shift
    out[:, 1::2] = ((b[..., 2].astype(np.uint16) << 4) | (mid >> 4)) << shift


try:
    import numba

    @numba.njit(nogil=True, cache=True)
    def _unpack_mono12packed_numba(src, out, shift):
        h, w = out.shape
        for y in range(h):
            for i in range(w // 2):
                b0 = np.uint16(src[y, 3 * i])
                b1 = np.uint16(src[y, 3 * i + 1])
                b2 = np.uint16(src[y, 3 * i + 2])
                out[y, 2 * i] = ((b0 << 4) | (b1 & 15)) << shift
                out[y, 2 * i + 1] = ((b2 << 4) | (b1 >> 4)) << shift
except Exception:                       # no numba: numpy (about as slow as harvesters' expansion)
    _unpack_mono12packed_numba = None


def unpack_mono12packed(src, out, shift):
    """GigE Vision ``Mono12Packed`` -> ``uint16`` in one pass, values as harvesters' PFNC expansion.

    Two pixels in three bytes: ``b0`` = p0[11:4], ``b1`` = p1[3:0] << 4 | p0[3:0], ``b2`` = p1[11:4].
    harvesters expands it with about ten numpy passes (7-10 ms for 1024 x 1280: with it the reader
    of an fxo992 tops out near 99 of 121.6 fps); with numba this writes the 16-bit-scaled values
    (``<< shift``) straight into ``out`` in ~1.5 ms and releases the GIL meanwhile. Without numba a
    numpy version runs, about as slow as harvesters'.

    Args:
        src: ``(height, >= 3 * width / 2)`` ``uint8`` rows of the raw buffer (padding allowed).
        out: ``(height, width)`` C-contiguous ``uint16`` destination; ``width`` must be even.
        shift: Left shift applied to the 12-bit values (4 -> full 16-bit scale).
    """
    if _unpack_mono12packed_numba is not None:
        _unpack_mono12packed_numba(src, out, shift)
    else:
        _unpack_mono12packed_numpy(src, out, shift)

faulthandler.enable()

logger = logging.getLogger(__name__)

_DIGITS = re.compile(r"(\d+)")

#: Writable target-temperature (TEC set-point) nodes tried in order; vendor names differ
#: (SFNC has none). Names matching _TARGET_PATTERN are tried after these.
TARGET_TEMPERATURE_NODES = (
    "TECTargetTemperature", "DeviceTECTargetTemperature", "TargetTemperature",
    "DeviceTemperatureTarget", "SensorTemperatureTarget", "CoolingTargetTemperature",
    "TemperatureSetpoint", "TECTemperatureSetpoint", "TECSetpoint",
)
#: Nodes that switch the cooler on (bool, or an enumeration with an "On"-like entry).
#: SVS fxo992MXGE-T: ``SensorTemperatureControl`` = OFF / Cooling / CoolingandHeating.
COOLER_ENABLE_NODES = (
    "SensorTemperatureControl",
    "TECEnable", "DeviceTECEnable", "TECControl", "TECMode", "TECControlMode",
    "CoolingEnable", "CoolerEnable", "Cooling", "Cooler",
)
#: Read-only TEC regulation state (SVS: SensorTemp_low / _high / _stable / _alert).
COOLING_STATE_NODES = ("SensorTemperatureControlState", "TECStatus", "TECState", "CoolingState")
#: ``DeviceTemperatureSelector`` entry measured by ``get_temperature`` (the sensor, not the board),
#: when the camera offers it; otherwise the selector is left as the camera has it.
TEMPERATURE_SELECTOR_ENTRY = "Sensor"
_TARGET_PATTERN = re.compile(r"(?i)(tec|cool).*(target|set)|target.*temp|temp.*set")
_ENABLE_PATTERN = re.compile(r"(?i)^(device)?(tec|cool(ing|er)?)(enable|mode|control|controlmode|state)?$")
_COOLING_LOG_PATTERN = re.compile(r"(?i)temp|tec|cool|peltier|fan")
_ON_VALUES = ("Cooling", True, "On", "Active", "Enable", "Enabled", "Auto", "Continuous")
_STATE_PREFIX = re.compile(r"(?i)^(sensor_?temp(erature)?_?|tec_?|cooling_?)")


def pixel_format_bits(name):
    """Bits per pixel of a GenICam pixel-format name, or ``None`` if it has no number.

    ``"Mono8"`` -> 8, ``"Mono10p"``/``"Mono10Packed"`` -> 10, ``"Mono12"`` -> 12,
    ``"BayerRG12p"`` -> 12, ``"Bpp12"`` (``PixelSize`` node) -> 12.
    """
    if name is None:
        return None
    match = _DIGITS.search(str(name))
    return int(match.group(1)) if match else None


class GenICamBackend(AbstractCameraBackend):
    """Backend for any GenICam camera reachable through a GenTL producer."""

    def __init__(
        self,
        camera_index: int = 0,
        gain: int = 0,
        exposure_ms: int = 100,
        cti_path: str = None,
        default_bin: int = None,
        sensor_pixel_size: int = 10,
    ):
        """Create a closed backend.

        Args:
            camera_index: Index into the Harvester device list.
            gain: Initial gain in camera units, clamped to the range at ``open``.
            exposure_ms: Initial exposure in milliseconds.
            cti_path: Path of the GenTL producer ``.cti``; defaults to
                ``$GENICAM_GENTL64_PATH/libsv_gev_tl_x64.cti``.
            default_bin: Bin applied at ``open`` (``None`` leaves the camera as is).
            sensor_pixel_size: ADC depth written to ``SensorPixelSize`` at ``open``
                (8, 10 or 12; ``None`` leaves the camera as is). Default 10, as originally.
        """
        super().__init__()
        self.camera_index = camera_index
        self.gain = gain
        self.exposure_ms = exposure_ms
        self.default_bin = default_bin
        self.sensor_pixel_size = None if sensor_pixel_size is None else int(sensor_pixel_size)
        if self.sensor_pixel_size not in (None, 8, 10, 12):
            raise ValueError("sensor_pixel_size must be 8, 10 or 12, got %r" % (sensor_pixel_size,))
        self.CTI_PATH = cti_path or os.path.join(
            os.getenv("GENICAM_GENTL64_PATH", ""),
            "libsv_gev_tl_x64.cti",
        )

        # hardware handles
        self._h = None
        self._ia = None
        self._nm = None

        # state
        self._current_bin: int = 1
        self._stop_event: threading.Event | None = None
        # (cam_wait_ms, convert_ms) of the last frame returned by _fetch, None if it returned None.
        self.last_fetch_timing: tuple[float, float] | None = None
        # (camera_time_ns, frame_id) of that frame; either part None if the producer lacks it.
        self.last_frame_info: tuple | None = None
        self._tick_hz: float | None = None         # GevTimestampTickFrequency, read at open
        self._info = CameraBackendInfo(
            camera_id=self.camera_index,
            bit_depth=16,
            pixel_size=1.0,
            width=0,
            height=0,
            image_format=None,
        )
        self._exposure_range_us = (1, 10_000_000)
        self._gain_range = (0, 255)
        self._supported_bins = ()
        self._source_bits = 12              # transmitted bits per pixel (see pixel_format_bits)
        # buffer data_format -> bits, worked out on the first frame of each format so _fetch reads
        # no camera node per frame (a node read is a GigE round trip); emptied by _refresh_info
        self._bits_by_format: dict = {}
        self._target_temp_node: str | None = None    # writable TEC set-point found at open
        self._cooler_enable_node: str | None = None
        self._cooling_state_node: str | None = None
        self._model_name: str | None = None
        self._serial_number: str | None = None

        # GigE config
        self.BUFFER_COUNT = 64
        self.ASSUME_JUMBO = True
        self.PACKET_SAFE = 9000
        self.SCPD_SAFE = 0
        self.SCPD_FAST = 0

    # ══════════════════════════════════════════════════════
    #  Properties
    # ══════════════════════════════════════════════════════

    @property
    def current_bin(self) -> int:
        """int: Bin currently applied (1 before any ``set_bin``)."""
        return self._current_bin

    @property
    def exposure_range_us(self) -> tuple:
        """tuple: ``(min, max)`` exposure in us from the ExposureTime node."""
        return self._exposure_range_us

    @property
    def gain_range(self) -> tuple:
        """tuple: ``(min, max)`` from the Gain node, in camera units."""
        return self._gain_range

    def get_bins(self) -> tuple:
        """Return the bins derived from the Binning node's min/max/increment (may be empty)."""
        return self._supported_bins

    @property
    def info(self) -> CameraBackendInfo:
        """CameraBackendInfo: width/height re-read from the camera after bin/param changes."""
        return self._info

    def model_name(self) -> str | None:
        """The ``DeviceModelName`` node read at ``open`` (``None`` before, or if absent)."""
        return self._model_name

    def serial_number(self) -> str | None:
        """The ``DeviceSerialNumber`` node read at ``open`` (``None`` before, or if absent)."""
        return self._serial_number

    # ══════════════════════════════════════════════════════
    #  Node helpers
    # ══════════════════════════════════════════════════════

    def _has_node(self, name: str) -> bool:
        return self._nm is not None and hasattr(self._nm, name)

    def _node(self, name: str):
        return getattr(self._nm, name) if self._has_node(name) else None

    def _read_node(self, name: str):
        node = self._node(name)
        if node is None:
            return None
        try:
            return node.value
        except Exception:
            return None

    def _set_node_value(self, name: str, value) -> bool:
        node = self._node(name)
        if node is None:
            return False
        try:
            node.value = value
            _ = node.value
            return True
        except Exception:
            return False

    def _set_node_max(self, name: str) -> bool:
        node = self._node(name)
        if node is None:
            return False
        try:
            node.value = node.max
            _ = node.value
            return True
        except Exception:
            return False

    def _node_names(self) -> list:
        """Names of every node in the camera's node map (empty if it cannot be listed)."""
        try:
            nodes = self._nm.nodes
        except Exception:
            return []
        names = []
        for item in nodes:
            inner = getattr(item, "node", item)
            try:
                names.append(str(inner.name))
            except Exception:
                continue
        return names

    def _is_writable(self, name: str) -> bool:
        node = self._node(name)
        if node is None:
            return False
        if _WRITABLE_ACCESS is None:
            return True
        try:
            return int(node.get_access_mode()) in _WRITABLE_ACCESS
        except Exception:
            return False

    def _is_numeric(self, name: str) -> bool:
        """True for an Integer/Float node (one with a numeric min and max)."""
        node = self._node(name)
        try:
            float(node.min), float(node.max)
            return not isinstance(node.value, (bool, str))
        except Exception:
            return False

    def _find_cooling_nodes(self) -> None:
        """Find a writable TEC set-point and cooler switch; log every temperature-related node."""
        names = self._node_names()
        related = [name for name in names if _COOLING_LOG_PATTERN.search(name)]
        if related:
            logger.info("GenICam temperature/cooling nodes: %s", ", ".join(
                "%s=%s%s" % (name, self._read_node(name), "" if self._is_writable(name) else " (RO)")
                for name in related))
        target_names = list(TARGET_TEMPERATURE_NODES) + [n for n in names if _TARGET_PATTERN.search(n)]
        enable_names = list(COOLER_ENABLE_NODES) + [n for n in names if _ENABLE_PATTERN.match(n)]
        self._target_temp_node = next(
            (n for n in target_names if self._is_writable(n) and self._is_numeric(n)), None)
        self._cooler_enable_node = next(
            (n for n in enable_names if n != self._target_temp_node and self._is_writable(n)), None)
        self._cooling_state_node = next(
            (n for n in COOLING_STATE_NODES if self._read_node(n) is not None), None)
        logger.info("GenICam cooling: target node %s, cooler switch %s, state node %s",
                    self._target_temp_node, self._cooler_enable_node, self._cooling_state_node)
        self._select_sensor_temperature()

    def _select_sensor_temperature(self) -> None:
        """Point ``DeviceTemperatureSelector`` at the sensor so ``get_temperature`` reads the sensor."""
        if not self._has_node("DeviceTemperatureSelector"):
            return
        before = self._read_node("DeviceTemperatureSelector")
        if self._set_node_value("DeviceTemperatureSelector", TEMPERATURE_SELECTOR_ENTRY):
            logger.info("GenICam DeviceTemperatureSelector %s -> %s: DeviceTemperature = %s",
                        before, self._read_node("DeviceTemperatureSelector"),
                        self._read_node("DeviceTemperature"))
        else:
            logger.warning("GenICam DeviceTemperatureSelector has no writable %r entry; temperature "
                           "is measured at %s", TEMPERATURE_SELECTOR_ENTRY, before)

    def _exec_node(self, name: str) -> bool:
        node = self._node(name)
        if node is None:
            return False
        try:
            node.execute()
            return True
        except Exception:
            return False

    # ══════════════════════════════════════════════════════
    #  Refresh
    # ══════════════════════════════════════════════════════

    def _refresh_info(self) -> None:
        width = int(self._read_node("Width") or 0)
        height = int(self._read_node("Height") or 0)
        pixel_format = self._read_node("PixelFormat")
        self._source_bits = self._bits_for(pixel_format)
        self._bits_by_format = {}           # format or SensorPixelSize may have changed
        self._info = CameraBackendInfo(
            camera_id=self.camera_index,
            bit_depth=8 if self._source_bits <= 8 else 16,
            pixel_size=1.0,
            width=width,
            height=height,
            image_format=pixel_format,
        )

        self._supported_bins = ()
        horiz = self._node("BinningHorizontal")
        vert = self._node("BinningVertical")
        if horiz is not None or vert is not None:
            try:
                source = horiz if horiz is not None else vert
                min_bin = int(source.min)
                max_bin = int(source.max)
                step = max(1, int(getattr(source, "inc", 1) or 1))
                if max_bin >= min_bin:
                    self._supported_bins = tuple(range(min_bin, max_bin + 1, step))
            except Exception:
                self._supported_bins = ()

    def _refresh_ranges(self) -> None:
        if self._has_node("ExposureTime"):
            try:
                node = self._node("ExposureTime")
                self._exposure_range_us = (float(node.min), float(node.max))
            except Exception:
                pass

        if self._has_node("Gain"):
            try:
                node = self._node("Gain")
                self._gain_range = (float(node.min), float(node.max))
            except Exception:
                pass

    # ══════════════════════════════════════════════════════
    #  Tuning
    # ══════════════════════════════════════════════════════

    def _tune_gige(self):
        self._exec_node("GevSCPSFireTest")

        if self._has_node("GevSCPSPacketSize"):
            if self.ASSUME_JUMBO:
                self._set_node_max("GevSCPSPacketSize")
            else:
                self._set_node_value("GevSCPSPacketSize", int(self.PACKET_SAFE))

        if self._has_node("DeviceLinkThroughputLimit"):
            self._set_node_max("DeviceLinkThroughputLimit")

        delay = int(self.SCPD_FAST if self.ASSUME_JUMBO else self.SCPD_SAFE)
        self._set_node_value("GevSCPD", delay)

    def _tune_framerate(self):
        if self._has_node("AcquisitionFrameRate"):
            self._set_node_max("AcquisitionFrameRate")
        if self._has_node("AcquisitionFrameRateEnable"):
            if not self._set_node_value("AcquisitionFrameRateEnable", False):
                self._set_node_value("AcquisitionFrameRateEnable", "Off")

    def _tune_image(self):
        self._set_node_value("ExposureAuto", "Off")
        self._set_node_value("AcquisitionMode", "Continuous")
        self._set_node_value("TriggerMode", "Off")

    # ══════════════════════════════════════════════════════
    #  Params
    # ══════════════════════════════════════════════════════

    def _set_gain(self, gain):
        if self._nm is None:
            return
        node = self._node("Gain")
        if node is None:
            return
        try:
            node.value = float(gain)
        except Exception:
            try:
                g = float(gain)
                if hasattr(node, "min"):
                    g = max(g, float(node.min))
                if hasattr(node, "max"):
                    g = min(g, float(node.max))
                node.value = g
            except Exception:
                pass

    def _set_exposure(self, expo_us):
        if self._nm is None:
            return
        self._set_node_value("ExposureAuto", "Off")
        node = self._node("ExposureTime")
        if node is None:
            return
        try:
            node.value = float(expo_us)
        except Exception:
            try:
                e = float(expo_us)
                if hasattr(node, "min"):
                    e = max(e, float(node.min))
                if hasattr(node, "max"):
                    e = min(e, float(node.max))
                node.value = e
            except Exception:
                pass

    def set_params(self, gain: int, exposure_ms: int) -> None:
        """Clamp and apply gain and exposure, then refresh ``info``.

        Args:
            gain: Gain in camera units, clamped to ``gain_range``.
            exposure_ms: Exposure in ms, clamped to ``exposure_range_us`` / 1000.
        """
        self.gain = max(self.gain_range[0], min(float(gain), self.gain_range[1]))
        self.exposure_ms = max(
            self.exposure_range_us[0] / 1000.0,
            min(float(exposure_ms), self.exposure_range_us[1] / 1000.0),
        )

        if self._nm is not None:
            self._set_gain(self.gain)
            self._set_exposure(self.exposure_ms * 1000.0)
            self._refresh_info()

    def set_bin(self, bin_value: int) -> None:
        """Apply the same bin horizontally and vertically.

        If the camera refuses it, the ROI is shrunk to the minimum, the bin retried and the
        full frame restored. Does nothing if the camera is not open.

        Args:
            bin_value: One of ``get_bins()``.

        Raises:
            ValueError: the bin is unsupported or the camera has no BinningHorizontal node.
            RuntimeError: the camera could not apply the bin even with a reset ROI.
        """
        if self._nm is None:
            return

        desired_bin = int(bin_value)
        if self._supported_bins and desired_bin not in self._supported_bins:
            raise ValueError(
                f"Unsupported bin value: {desired_bin}. "
                f"Supported: {self._supported_bins}"
            )

        if not self._has_node("BinningHorizontal"):
            raise ValueError("GenICam backend does not support BinningHorizontal")

        # BinningSelector is deliberately left as the camera has it (as in the original code):
        # forcing "Sensor" binning made some cameras refuse AcquisitionStart at bin 3/4.
        try:
            self._write_binning(desired_bin)
        except Exception as first_error:
            # Some cameras refuse a bin while the current ROI exceeds the new
            # sensor size: shrink ROI, change bin, then restore the full frame.
            logger.warning("Bin %d rejected (%s); retrying with a reset ROI", desired_bin, first_error)
            try:
                self._reset_roi(full=False)
                self._write_binning(desired_bin)
            except Exception as second_error:
                self._reset_roi(full=True)   # never leave the camera on a minimal ROI
                self._refresh_info()
                raise RuntimeError(
                    "GenICam could not apply bin %d: %s" % (desired_bin, second_error)
                ) from second_error
            self._reset_roi(full=True)

        self._current_bin = desired_bin
        self._refresh_info()
        logger.info("GenICam bin %d applied: frame %dx%d",
                    desired_bin, self._info.width, self._info.height)

    def _write_binning(self, desired_bin: int) -> None:
        """Write both binning nodes and verify what the camera now reports."""
        for name in ("BinningHorizontal", "BinningVertical"):
            node = self._node(name)
            if node is not None:
                node.value = desired_bin
        for name in ("BinningHorizontal", "BinningVertical"):
            actual = self._read_node(name)
            if actual is not None and int(actual) != desired_bin:
                raise ValueError("%s is %s after setting %d" % (name, actual, desired_bin))

    def _reset_roi(self, full: bool) -> None:
        """Move the ROI to the origin and set it to the smallest or the largest size."""
        for name in ("OffsetX", "OffsetY"):
            self._set_node_value(name, 0)
        for name in ("Width", "Height"):
            node = self._node(name)
            if node is None:
                continue
            try:
                node.value = node.max if full else node.min
            except Exception as exc:
                logger.debug("Could not set %s: %s", name, exc)

    # ══════════════════════════════════════════════════════
    #  Hardware lifecycle — base class calls these
    # ══════════════════════════════════════════════════════

    def _open_device(self) -> None:
        """Load the producer, create the acquirer, tune GigE/frame-rate and apply defaults.

        Raises:
            RuntimeError: no device was found.
        """
        self._h = Harvester()
        self._h.add_file(self.CTI_PATH)
        self._h.update()

        if len(self._h.device_info_list) == 0:
            raise RuntimeError(
                "No device found. ตรวจสอบ IP/subnet/firewall"
            )

        self._ia = self._h.create(self.camera_index)
        self._ia.buffer_handling_mode = "OldestFirst"
        self._ia.num_buffers = self.BUFFER_COUNT
        self._nm = self._ia.remote_device.node_map
        self._model_name = device_text(self._read_node("DeviceModelName"))
        self._serial_number = device_text(self._read_node("DeviceSerialNumber"))
        if self.sensor_pixel_size is not None:
            wanted = "SensorBpp%d" % self.sensor_pixel_size
            if self._set_node_value("SensorPixelSize", wanted):
                logger.info("GenICam SensorPixelSize = %s", self._read_node("SensorPixelSize"))
            elif self._has_node("SensorPixelSize"):
                logger.warning("GenICam camera refused SensorPixelSize=%s (now %s)",
                               wanted, self._read_node("SensorPixelSize"))
            else:
                logger.warning("GenICam camera has no SensorPixelSize node; %s not applied", wanted)
        self._set_node_value("DefectPixelCorrectionEnable_Control", "On")
        self._set_node_value("DefectPixelCorrectionMark_Control", "Off")

        self._refresh_ranges()
        self._refresh_info()
        self._tune_gige()
        self._tune_framerate()
        self._tune_image()
        tick = self._read_node("GevTimestampTickFrequency") or self._read_node("TimestampTickFrequency")
        self._tick_hz = float(tick) if tick else None
        self._find_cooling_nodes()
        self.set_params(self.gain, self.exposure_ms)

        if self.default_bin is not None:
            try:
                self.set_bin(int(self.default_bin))
            except Exception as e:
                print(f"[GenICam] Warning: could not apply default_bin={self.default_bin}: {e}")
        # compile (or load from numba's cache) the Mono12Packed unpacker now, not on the first frame
        unpack_mono12packed(np.zeros((1, 3), np.uint8), np.zeros((1, 2), np.uint16), 4)

    def _close_device(self) -> None:
        """Set the stop event and destroy the acquirer and Harvester; errors are swallowed."""
        if self._stop_event is not None:
            self._stop_event.set()

        if self._ia is not None:
            try:
                self._ia.destroy()
            except Exception:
                pass
        self._ia = None
        self._nm = None

        if self._h is not None:
            try:
                self._h.reset()
            except Exception:
                pass
        self._h = None

    #: Waits (s) before each AcquisitionStart attempt. After a binning/ROI change some cameras
    #: are busy re-configuring the sensor and answer the start command with GC_ERR_IO.
    START_RETRY_DELAYS = (0.0, 0.2, 0.5, 1.0, 2.0)

    def _start_device(self) -> None:
        """Start the acquirer, retrying while the camera is still busy (e.g. right after a bin change).

        Raises:
            RuntimeError: the backend is not open.
            Exception: the last error from the camera when every attempt failed.
        """
        if self._ia is None:
            raise RuntimeError("GenICam backend is not opened")
        last_error = None
        for attempt, delay in enumerate(self.START_RETRY_DELAYS):
            if self._stop_event is not None and self._stop_event.is_set():
                break
            if delay:
                time.sleep(delay)
            try:
                self._ia.start()
                if attempt:
                    logger.warning("GenICam acquisition started after %d retries", attempt)
                return
            except Exception as exc:
                last_error = exc
                logger.warning("GenICam AcquisitionStart failed (attempt %d/%d): %s",
                               attempt + 1, len(self.START_RETRY_DELAYS), exc)
                try:
                    self._ia.stop()
                except Exception:
                    pass
        raise last_error if last_error is not None else RuntimeError("GenICam start interrupted")

    def _stop_device(self) -> None:
        """Stop the acquirer; errors are swallowed."""
        if self._ia is not None:
            try:
                self._ia.stop()
            except Exception:
                pass

    # ══════════════════════════════════════════════════════
    #  Fetch — fetch + convert by pixel format (8 -> uint8, 10/12/14 -> uint16 scaled)
    # ══════════════════════════════════════════════════════

    def _fetch_raw_frame(self, timeout_ms: int = 1000) -> np.ndarray | None:
        """Fetch one buffer and convert it to the output format (see the module docstring).

        Args:
            timeout_ms: Maximum wait in milliseconds.

        Returns:
            ``(height, width)`` copy owned by the caller - ``uint8`` for 8-bit formats, ``uint16``
            scaled to 16 bits otherwise - or ``None`` on timeout, incomplete buffer, size mismatch
            with ``info``, error or stop event.
        """
        return self._fetch(timeout_ms, None)

    def read_frame_raw_into(self, provide, timeout_ms: int = 1000) -> np.ndarray | None:
        """``read_frame_raw`` that writes the converted frame into a caller-supplied buffer.

        The camera buffer must be copied anyway (harvesters reuses it), so converting
        straight into ``provide((height, width))`` replaces the new array of
        ``read_frame_raw`` at no extra cost; a GPU that reads that buffer in place then
        needs no upload.  Values are identical to ``read_frame_raw``.

        Args:
            provide: Callable ``(shape) -> numpy.ndarray | None`` returning a writable
                C-contiguous ``uint16`` array of that shape, or ``None`` to get a new array.
            timeout_ms: Maximum wait in milliseconds.

        Returns:
            The provided buffer (overwritten by the next call), a new array when
            ``provide`` gave none, or ``None`` as for ``read_frame_raw``.
        """
        return self._fetch(timeout_ms, provide)

    def _fetch(self, timeout_ms, provide):
        """Fetch one complete buffer of the expected size and return it converted.

        Args:
            timeout_ms: Maximum wait in milliseconds.
            provide: ``read_frame_raw_into``'s callable, or ``None`` for a new array.

        Sets ``last_fetch_timing`` to ``(cam_wait_ms, convert_ms)``: the wait for the
        buffer, then the checks and conversion; ``None`` when no frame is returned.

        With ``PixelFormat`` ``Mono12Packed`` the buffer is fetched raw (:meth:`_fetch_raw`):
        harvesters' own expansion of that format costs 7-10 ms per 1024 x 1280 frame. Every
        other format goes through harvesters' payload (:meth:`_fetch_payload`). The values are
        the same either way.

        Camera nodes are read only when the frame size differs from ``info`` or on the first
        frame of a pixel format (its bits are then kept in ``_bits_by_format``), not per frame.
        """
        self.last_fetch_timing = None
        self.last_frame_info = None
        if self._ia is None:
            return None

        if self._stop_event is not None and self._stop_event.is_set():
            return None

        if self._info.image_format == "Mono12Packed":
            return self._fetch_raw(timeout_ms, provide)
        return self._fetch_payload(timeout_ms, provide)

    def _frame_size_ok(self, h, w) -> bool:
        """``(h, w)`` matches ``info`` (re-read once: info may be stale right after a bin change)."""
        if self._info.height > 0 and self._info.width > 0:
            if h != self._info.height or w != self._info.width:
                self._refresh_info()
                if h != self._info.height or w != self._info.width:
                    return False
        return True

    def _bits_of(self, data_format) -> int:
        """Bits per pixel of a buffer's format; nodes are read on its first frame only."""
        if data_format is None:
            return self._source_bits
        bits = self._bits_by_format.get(data_format)
        if bits is None:
            if data_format != self._info.image_format:
                self._refresh_info()               # format changed on the camera: follow it
            bits = self._bits_by_format[data_format] = self._bits_for(data_format)
        return bits

    def _fetch_payload(self, timeout_ms, provide):
        """harvesters' payload path: it expands the pixels, ``_convert``/``_convert_into`` scale them."""
        try:
            start = time.perf_counter()
            with self._ia.fetch(timeout=timeout_ms / 1000.0) as buffer:   # harvesters takes seconds
                received = time.perf_counter()
                if getattr(buffer, 'is_incomplete', False):
                    return None

                comp = buffer.payload.components[0]
                h, w = comp.height, comp.width
                if not self._frame_size_ok(h, w):
                    return None

                bits = self._bits_of(getattr(comp, "data_format", None))
                data = comp.data.reshape(h, w)
                frame = self._convert(data, bits) if provide is None else self._convert_into(data, bits, provide)
                self.last_fetch_timing = ((received - start) * 1000.0, (time.perf_counter() - received) * 1000.0)
                self.last_frame_info = (self._camera_ns(getattr(buffer, 'timestamp', None)),
                                        getattr(buffer, 'frame_id', None))
                return frame

        except Exception:
            return None

    def _fetch_raw(self, timeout_ms, provide):
        """The raw GenTL buffer (no harvesters payload), converted by :meth:`_convert_raw` and queued back."""
        try:
            start = time.perf_counter()
            raw = self._ia.fetch(timeout=timeout_ms / 1000.0, is_raw=True)   # harvesters takes seconds
        except Exception:
            return None
        if raw is None:
            return None
        received = time.perf_counter()
        try:
            if getattr(raw, 'is_incomplete', False):
                return None
            h, w = int(raw.height), int(raw.width)
            if not self._frame_size_ok(h, w):
                return None
            data_format = self._format_of(raw)
            bits = self._bits_of(data_format)
            frame = self._convert_raw(raw, h, w, data_format, bits, provide)
            if frame is None:
                return None
            self.last_fetch_timing = ((received - start) * 1000.0, (time.perf_counter() - received) * 1000.0)
            self.last_frame_info = (self._camera_ns(self._timestamp_of(raw)), getattr(raw, 'frame_id', None))
            return frame
        except Exception:
            return None
        finally:
            try:
                raw.parent.queue_buffer(raw)       # back to the acquisition pool
            except Exception:
                pass

    def _convert_raw(self, raw, h, w, data_format, bits, provide):
        """The converted frame of a raw GenTL buffer, or ``None`` if it holds too few bytes."""
        try:
            padding = int(raw.padding_x)
        except Exception:
            padding = 0
        if data_format == "Mono12Packed" and w % 2 == 0:
            line = w * 3 // 2 + padding
            data = np.frombuffer(raw.raw_buffer, dtype=np.uint8)
            if data.size < h * line:
                return None
            out = provide((h, w)) if provide is not None else None
            if out is None:
                out = np.empty((h, w), np.uint16)
            unpack_mono12packed(data[:h * line].reshape(h, line), out, 16 - int(bits) if 8 < bits < 16 else 0)
            return out
        size = _UNPACKED_BYTES.get(data_format)
        if size is not None:
            line = w * size + padding
            data = np.frombuffer(raw.raw_buffer, dtype=np.uint8)
            if data.size < h * line:
                return None
            pixels = data[:h * line].reshape(h, line)[:, :w * size]
            data2d = pixels if size == 1 else pixels.view(np.uint16)
        else:                                      # format changed under us: harvesters' own expansion
            from harvesters.core import Buffer
            payload = Buffer(module=raw, node_map=self._ia.remote_device.node_map, acquire=self._ia).payload
            comp = payload.components[0]
            data2d = comp.data.reshape(h, w)
        if provide is not None:
            return self._convert_into(data2d, bits, provide)
        return self._convert(data2d, bits)

    def _format_of(self, raw):
        """Pixel format name of a raw buffer (``PixelFormat`` from ``info`` if it has none)."""
        try:
            value = raw.pixel_format
        except Exception:
            return self._info.image_format
        return _PFNC_NAMES.get(value, self._info.image_format)

    @staticmethod
    def _timestamp_of(raw):
        """The value harvesters' ``Buffer.timestamp`` gives: ``timestamp_ns``, else ticks, else 0."""
        for name in ("timestamp_ns", "timestamp"):
            try:
                return getattr(raw, name)
            except Exception:
                continue
        return 0

    def _camera_ns(self, ticks):
        """Camera timestamp ticks -> nanoseconds (``None`` without a tick frequency or value)."""
        if ticks is None or not self._tick_hz:
            return None
        ticks = int(ticks)
        return ticks if self._tick_hz == 1e9 else int(ticks * 1e9 / self._tick_hz)

    def camera_clock_offset_ns(self):
        """Latch the camera clock and return ``camera_ns - host perf_counter_ns``.

        The host time is the midpoint of the latch round trip (about a millisecond over
        GigE), so the offset is good to about half of it.  Returns ``None`` if the camera
        has no latch node or the value cannot be read.  Acquisition thread only.
        """
        for command, value_node in (("GevTimestampControlLatch", "GevTimestampValue"),
                                    ("TimestampLatch", "TimestampLatchValue")):
            if not self._has_node(command):
                continue
            before = time.perf_counter_ns()
            if not self._exec_node(command):
                return None
            after = time.perf_counter_ns()
            camera_ns = self._camera_ns(self._read_node(value_node))
            return None if camera_ns is None else camera_ns - (before + after) // 2
        return None

    def _bits_for(self, pixel_format) -> int:
        """Bits carried per pixel: the pixel format if it says 8-14, else ``SensorPixelSize``, else 12."""
        bits = pixel_format_bits(pixel_format)
        if bits is None or bits >= 16:                 # Mono16 container / unknown: ask the sensor
            sensor_bits = pixel_format_bits(self._read_node("SensorPixelSize"))
            if sensor_bits is not None and sensor_bits < 16:
                return sensor_bits
        return bits if bits is not None else 12

    def saturation_level(self) -> int:
        """Full scale of the delivered frames: ``(2**bits - 1) << (16 - bits)`` for 10-14 bits."""
        bits = int(self._source_bits)
        if 8 < bits < 16:
            return ((1 << bits) - 1) << (16 - bits)
        return super().saturation_level()

    def sensor_bit_depth(self) -> int:
        """Real data bits: the pixel format's bits, or ``SensorPixelSize`` if fewer (10-bit in Mono12Packed -> 10)."""
        bits = int(self._source_bits)
        sensor_bits = pixel_format_bits(self._read_node("SensorPixelSize"))
        if sensor_bits is not None and sensor_bits < bits:
            return sensor_bits
        return bits

    @staticmethod
    def _convert(data, bits):
        """``uint16`` copy of ``data``: 8-bit values unchanged, 10-14 bits shifted to 16 bits."""
        frame = np.array(data, dtype=np.uint16, copy=True)
        if 8 < bits < 16:
            frame <<= 16 - int(bits)
        return frame

    @staticmethod
    def _convert_into(data, bits, provide):
        """``_convert`` written into ``provide(data.shape)`` in one pass (new array if it gives none)."""
        out = provide(data.shape) if data.dtype in (np.uint8, np.uint16) else None
        if out is None:
            return GenICamBackend._convert(data, bits)
        shift = 16 - int(bits) if 8 < bits < 16 else 0
        np.left_shift(data, np.uint16(shift), out=out, dtype=np.uint16)
        return out

    # ══════════════════════════════════════════════════════
    #  Misc
    # ══════════════════════════════════════════════════════

    def set_stop_event(self, event: threading.Event):
        """Register an event that makes reads return ``None`` and is set on close."""
        self._stop_event = event

    def dropped_images_count(self) -> int:
        """Always 0: this backend does not track dropped frames."""
        return 0

    def frame_rate(self):
        """Return ``AcquisitionFrameRate`` (or ``AcquisitionFrameRateAbs``) in Hz, ``None`` if absent."""
        if self._nm is None:
            return None
        for node_name in ("AcquisitionFrameRate", "AcquisitionFrameRateAbs"):
            val = self._read_node(node_name)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return None
        return None

    def get_temperature(self) -> float:
        """Return the first of DeviceTemperature/Temperature/TemperatureAbs in degrees Celsius,
        or ``-999.0`` if none is available.

        ``DeviceTemperature`` is measured where ``DeviceTemperatureSelector`` points; ``open``
        selects the sensor (:data:`TEMPERATURE_SELECTOR_ENTRY`) when the camera offers it.
        """
        if self._nm is None:
            return -999.0
        for node_name in ("DeviceTemperature", "Temperature", "TemperatureAbs"):
            val = self._read_node(node_name)
            if val is not None:
                return float(val)
        return -999.0

    def cooling_state(self):
        """TEC regulation state from the state node (``"stable"``, ``"low"``, ``"high"``,
        ``"alert"`` on SVS cameras), or ``None`` if the camera has none."""
        if self._nm is None or self._cooling_state_node is None:
            return None
        value = self._read_node(self._cooling_state_node)
        if value is None:
            return None
        return _STATE_PREFIX.sub("", str(value)) or str(value)

    def cooling_supported(self) -> bool:
        """True when ``open`` found a writable TEC set-point node (see ``TARGET_TEMPERATURE_NODES``)."""
        return self._nm is not None and self._target_temp_node is not None

    def target_temperature_range(self):
        """``(min, max)`` of the set-point node in °C, or ``None``."""
        node = self._node(self._target_temp_node) if self.cooling_supported() else None
        if node is None:
            return None
        try:
            return float(node.min), float(node.max)
        except Exception:
            return None

    def get_target_temperature(self):
        """Value of the set-point node in °C, or ``None``."""
        if not self.cooling_supported():
            return None
        value = self._read_node(self._target_temp_node)
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    def set_target_temperature(self, temp_c: float) -> None:
        """Switch the cooler on (if the camera has a switch node) and write the set-point.

        Raises:
            RuntimeError: no set-point node, or the camera refused the value.
        """
        if not self.cooling_supported():
            raise RuntimeError("This GenICam camera has no writable target-temperature node")
        if self._cooler_enable_node is not None:
            if not any(self._set_node_value(self._cooler_enable_node, value) for value in _ON_VALUES):
                logger.warning("GenICam: could not switch %s on (now %s)",
                               self._cooler_enable_node, self._read_node(self._cooler_enable_node))
        node = self._node(self._target_temp_node)
        value = float(temp_c)
        try:
            value = min(max(value, float(node.min)), float(node.max))
        except Exception:
            pass
        try:
            node.value = value
        except Exception:
            try:
                node.value = int(round(value))      # integer set-point node
            except Exception as exc:
                raise RuntimeError("%s=%s refused: %s" % (self._target_temp_node, value, exc)) from exc
        logger.info("GenICam %s = %s (cooler %s = %s)", self._target_temp_node,
                    self._read_node(self._target_temp_node), self._cooler_enable_node,
                    self._read_node(self._cooler_enable_node) if self._cooler_enable_node else None)