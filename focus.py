# -*- coding: utf-8 -*-
"""Focus score for the live focus aid (``widgets/focus_panel.py``).

Edge contrast: the frame is averaged across the other axis first, then the RMS step between
neighbouring pixels of that profile is taken, relative to its mean::

    profile = frame.mean(axis=1)                       # axis 0: the slit image, all wavelengths
    edge_contrast = 100 * rms(diff(profile)) / mean(profile)      # % per pixel

Why not only hsi's ``sharpness_score`` (mean |Sobel|, ``modules/sharpness``): the mean *absolute*
gradient across an edge equals the edge height however wide the edge is blurred, so it hardly
changes near focus, and on raw frames the sensor noise (about 500 counts here) sets a high floor.
On three recorded frames in ``data/`` with defocus applied before the noise, at 1 px blur
(sigma) hsi's score kept 93.5-95.1 % of its in-focus value, edge contrast 71-96 % (on one
frame slightly less responsive than hsi's up to 2 px, ahead from 3 px; at 5 px 36-73 % against
hsi's 81-88 %). On a bar target (isolated edges) hsi's score stays flat up to 2 px, edge
contrast falls. Averaging the 1280 spectral columns first removes most of the noise (the slit
image is the same at every wavelength), the RMS makes a blurred edge score lower, the mean makes
the score independent of exposure and gain, and it costs about 1 ms instead of 7-9 ms per
1024 x 1280 frame. hsi's formula is still shown beside it, on the same raw frame (hsi itself
scores its corrected, binned frame, so hsi's own numbers differ).

Data conventions:
    ``frame`` is a raw ``(slit, wavelength)`` frame of any numeric dtype. ``axis=0`` scores focus
    along the slit (imaging focus, needs edges under the slit); ``axis=1`` along the wavelength
    axis (spectrograph focus, needs spectral lines). The result is in percent per pixel; scores
    compare only for the same scene and bin.

Threading:
    Pure function, any thread.
"""
from __future__ import annotations

import numpy as np


def edge_contrast(frame: np.ndarray, axis: int = 0) -> float:
    """RMS step between neighbouring pixels of the profile along ``axis``, % of its mean.

    Args:
        frame: 2-D ``(slit, wavelength)`` array.
        axis: 0 = along the slit (the profile is the mean over wavelengths), 1 = along the
            wavelength axis (the mean over the slit).

    Returns:
        Percent per pixel (larger = sharper); ``nan`` for a frame with fewer than 2 pixels along
        ``axis`` or a profile mean of 0 or less.
    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 (slit) or 1 (wavelength), got {axis}")
    frame = np.asarray(frame)
    if frame.ndim != 2:
        raise ValueError(f"expected a 2-D (slit, wavelength) frame, got shape {frame.shape}")
    profile = frame.mean(axis=1 - axis, dtype=np.float64)
    mean = float(profile.mean()) if profile.size else 0.0
    if profile.size < 2 or not mean > 0:
        return float('nan')
    step = np.diff(profile)
    return float(100.0 * np.sqrt(np.mean(step * step)) / mean)
