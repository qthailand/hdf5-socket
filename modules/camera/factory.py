"""Select and construct a camera backend by name."""
from __future__ import annotations

from .base import AbstractCameraBackend

BACKENDS = ("poa", "genicam", "ximea")


def create_backend(config) -> AbstractCameraBackend:
    """Build the backend named by ``config.camera_backend``.

    Backend modules (and their vendor SDKs) are imported on demand, so an SDK that is
    not installed matters only when that backend is selected, and then fails with its
    own error.

    Args:
        config: Any object with attributes ``camera_backend`` (``"poa"``,
            ``"genicam"`` or ``"ximea"``, case-insensitive), ``default_gain``,
            ``default_exposure_ms`` and ``default_bin``; ``camera_index``,
            ``cti_path`` and ``sensor_pixel_size`` (GenICam) are optional. ``AppConfig`` qualifies.

    Returns:
        A closed backend; call :meth:`AbstractCameraBackend.open` before use.

    Raises:
        ValueError: if the backend name is unknown.
        ImportError: if the selected backend's SDK is not installed.

    Note:
        ``camera_index`` selects the device for GenICam and XIMEA; POA always opens
        the first camera it enumerates.
    """
    name = str(getattr(config, "camera_backend", "poa")).lower()
    gain = config.default_gain
    exposure_ms = config.default_exposure_ms
    if name == "poa":
        from .poa_camera import POACameraBackend as backend_class
        return backend_class(gain=gain, exposure_ms=exposure_ms)
    if name == "genicam":
        from .genicam import GenICamBackend as backend_class
        return backend_class(
            camera_index=getattr(config, "camera_index", 0),
            gain=gain,
            exposure_ms=exposure_ms,
            cti_path=getattr(config, "cti_path", None),
            default_bin=config.default_bin,
            sensor_pixel_size=getattr(config, "sensor_pixel_size", 10),
        )
    if name == "ximea":
        from .ximea_camera import XIMEACameraBackend as backend_class
        return backend_class(
            camera_id=getattr(config, "camera_index", 0),
            gain=gain,
            exposure_ms=exposure_ms,
            default_bin=config.default_bin,
        )
    raise ValueError("Unknown camera_backend %r (supported: %s)" % (name, ", ".join(BACKENDS)))
