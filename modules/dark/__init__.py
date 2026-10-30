"""Dark/hot-pixel correction copied unchanged from the HSI acquisition project.

``core``, ``library`` and ``selection`` are byte-identical copies of
``hsi/modules/dark``; only this ``__init__`` is local (no Qt widget, no config).
"""
from .core import DarkProcessor, DarkReference
from .library import DarkFrameLibrary, UNKNOWN_TEMPERATURE_C, valid_temperature
from .selection import DarkSelection, select_dark

__all__ = ["DarkProcessor", "DarkReference", "DarkFrameLibrary", "UNKNOWN_TEMPERATURE_C",
           "valid_temperature", "DarkSelection", "select_dark"]
