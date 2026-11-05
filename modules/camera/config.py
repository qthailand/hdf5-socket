"""Settings a camera needs, owned by the camera module.

The application aggregates these section dataclasses into its ``AppConfig`` and stores
them in ``config/camera.json``; a stand-alone user can build a :class:`CameraConfig`
and pass it straight to :func:`modules.camera.create_backend`.

Data conventions:
    Exposure is in milliseconds; gain in the camera's vendor units.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CameraConfig:
    """Which camera to open and with which start-up settings.

    Attributes:
        camera_backend: ``"poa"``, ``"genicam"`` or ``"ximea"``.
        camera_id: Legacy numeric id (kept for stored profiles' compatibility blocks).
        camera_index: Device index for GenICam and XIMEA (POA opens the first camera).
        cti_path: GenTL producer (``.cti``) for GenICam; ``None`` uses the environment default.
        default_gain: Gain applied at start-up.
        default_exposure_ms: Exposure (ms) applied at start-up.
        default_bin: Hardware bin applied at start-up when the camera supports it.
        sensor_pixel_size: GenICam ``SensorPixelSize`` (8, 10 or 12 bits) written at start-up.
            8 gives frames with values 0..255 (``bit_depth`` 8); 10/12 are scaled to 16 bits.
            Optional in ``camera.json``; 10 is the original behaviour.
        target_temperature_c: Cooler target (°C) applied at start-up on cameras with a settable
            target (TEC); ``None`` leaves the cooler as the camera has it. Optional in
            ``camera.json``; the application saves the last value the user set.
        camera_profile_folder: Root of the per-camera calibration folders
            (``<root>/<backend>/<model>_<serial>/`` with ``dark``, ``smile`` and ``wavelength``,
            see :class:`modules.contracts.CameraProfile`). Optional in ``camera.json``.

    ``default_gain``, ``default_exposure_ms``, ``default_bin`` and ``target_temperature_c`` are
    written back to ``camera.json`` when the user changes them, so the next start uses them.
    """

    camera_backend: str = "poa"
    camera_id: int = 0
    camera_index: int = 0
    cti_path: str | None = None
    default_gain: int = 100
    default_exposure_ms: float = 50.0
    default_bin: int = 1
    sensor_pixel_size: int = 10
    target_temperature_c: float | None = None
    camera_profile_folder: str = "camera_profiles"
