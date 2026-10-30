"""Reflectance factors copied unchanged from ``hsi/modules/radiometric``.

``core.py`` is a byte-identical copy; only this ``__init__`` is local (no calibration window).
The simulator uses ``load_factor_profile`` / ``check_factor_bands`` and hsi's rule that
reflectance = counts x factor (row-wise ``(lines, bands)`` or per band ``(bands,)``).
"""
from .core import check_factor_bands, load_factor_profile, save_factor_profile

__all__ = ["check_factor_bands", "load_factor_profile", "save_factor_profile"]
