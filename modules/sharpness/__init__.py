"""Focus (sharpness) score copied unchanged from the HSI acquisition project.

``core`` is a byte-identical copy of ``hsi/modules/sharpness/core.py``, so the formula is exactly
hsi's (hsi applies it to its dark/Smile-corrected, binned frame; the GUI here applies it to the
raw frame, so the values differ from hsi's scan window). Only this ``__init__`` is local: hsi's
``SharpnessPanel`` (fixed dark colours) is not copied; the GUI uses the themed
``widgets.focus_panel.FocusPanel`` instead.

Data conventions:
    Images are ``(slit, wavelength)``; ``axis=0`` measures focus along the slit (pushbroom
    imaging focus), ``axis=1`` along the spectral axis (spectrograph focus on line features).
"""
from .core import ImageSharpness, sharpness_score

__all__ = ["ImageSharpness", "sharpness_score"]
