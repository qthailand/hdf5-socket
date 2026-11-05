"""Camera acquisition backends (POA, GenICam, XIMEA) copied unchanged from the HSI project.

``base``, ``config``, ``factory``, ``__main__`` and the three backends are byte-identical copies
of ``hsi/modules/camera`` (``tools/sync_hsi.py``); only this ``__init__`` is local: it exports
the backends but not hsi's Qt widgets (``CameraControls``, ``PixelPitchDialog``), which are not
copied. The vendor SDK folders ``poa/`` and ``ximea/`` are not in git; copy them with
``python tools/sync_hsi.py --update --only camera``. Each SDK is imported only when its backend
is created, so a missing SDK matters only for that camera.

The live frame source of the simulator (``camera_source.py``) builds a backend with
:func:`create_backend`; see ``hsi/modules/camera/README.md`` for the backend contract.

    python -m modules.camera --backend genicam --cti C:\\path\\producer.cti --bin 2   # hardware check
"""
import importlib

from .base import AbstractCameraBackend, CameraBackendInfo
from .config import CameraConfig
from .factory import BACKENDS, create_backend

__all__ = ["CameraConfig", "AbstractCameraBackend", "CameraBackendInfo", "create_backend", "BACKENDS",
           "POACameraBackend", "GenICamBackend", "XIMEACameraBackend"]

_LAZY = {
    "POACameraBackend": ".poa_camera",
    "GenICamBackend": ".genicam",
    "XIMEACameraBackend": ".ximea_camera",
}


def __getattr__(name):
    """Import a backend class (and its SDK) on first access."""
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
