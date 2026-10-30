"""Choose the stored dark reference that fits the camera's current frame.

Dark files are looked up by ``(bin, exposure, gain)`` and, among those, the one captured nearest
the current sensor temperature (see :meth:`DarkFrameLibrary.find`). A file may still be unusable:
it can come from another camera or bin whose frame has a different size, and applying it
would be wrong (its hot-pixel coordinates could even fall outside the frame). This helper
makes that decision once, in one place, without any GUI.

Data conventions:
    ``frame_shape`` is ``(height, width) = (slit rows, spectral columns)`` of the frames
    the camera currently delivers. Temperatures are °C; ``None``/``-999`` means unknown.

Threading:
    Reads one ``.npz`` file; call from the GUI/control thread, not from the frame loop.

Example:
    >>> # selection = select_dark(library, bin=2, exposure=1.0, gain=1.0, frame_shape=(552, 804))
    >>> # if selection.kind == 'loaded': camera.set_dark_reference(selection.result['mean'], selection.result['hot_pixel_map'])
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .library import valid_temperature


@dataclass(frozen=True)
class DarkSelection:
    """Result of :func:`select_dark`.

    Attributes:
        kind: ``"loaded"`` (usable), ``"missing"`` (no file for these parameters) or
            ``"mismatch"`` (a file exists but its frame size differs from the camera's).
        result: The dict returned by :meth:`DarkFrameLibrary.load` for ``"loaded"``, else ``None``.
        dark_shape: ``(height, width)`` of the stored dark, if one was found.
        frame_shape: ``(height, width)`` the camera delivers (as passed in).
        n_hot: Number of hot pixels in the stored map (``"loaded"`` only).
        dark_temperature: Mean sensor temperature (°C) of the stored dark, ``None`` if it has none.
        temperature: The camera temperature the selection was made for (``None`` if unknown).
    """

    kind: str
    result: Optional[dict] = None
    dark_shape: Optional[Tuple[int, int]] = None
    frame_shape: Optional[Tuple[int, int]] = None
    n_hot: int = 0
    dark_temperature: Optional[float] = None
    temperature: Optional[float] = None


def select_dark(library, bin: int, exposure: float, gain: float,
                frame_shape: Optional[Tuple[int, int]],
                temperature: Optional[float] = None) -> DarkSelection:
    """Look up the dark for ``(bin, exposure, gain)`` nearest ``temperature`` and check it fits.

    Args:
        library: A :class:`~modules.dark.DarkFrameLibrary`.
        bin: Hardware bin factor.
        exposure: Exposure in ms.
        gain: Camera gain.
        frame_shape: ``(height, width)`` of the camera frames; ``None``/zeros skip the size check.
        temperature: Current sensor temperature (°C); ``None``/``-999`` = unknown (the library
            then assumes :data:`~modules.dark.UNKNOWN_TEMPERATURE_C`).

    Returns:
        A :class:`DarkSelection`.

    Raises:
        Exception: whatever the library raises for a corrupt file (callers report it).
    """
    temperature = valid_temperature(temperature)
    result = library.load(bin=bin, exposure=exposure, gain=gain, temperature=temperature)
    if result is None:
        return DarkSelection("missing", frame_shape=frame_shape, temperature=temperature)
    dark_shape = tuple(int(v) for v in result["mean"].shape)
    dark_temperature = result.get("temperature")
    if frame_shape and all(frame_shape) and dark_shape != tuple(frame_shape):
        return DarkSelection("mismatch", dark_shape=dark_shape, frame_shape=tuple(frame_shape),
                             dark_temperature=dark_temperature, temperature=temperature)
    return DarkSelection("loaded", result, dark_shape, tuple(frame_shape) if frame_shape else None,
                         int(np.count_nonzero(result["hot_pixel_map"])),
                         dark_temperature=dark_temperature, temperature=temperature)
