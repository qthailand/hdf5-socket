"""Image sharpness (focus) score for live focusing of a pushbroom spectrograph.

Data conventions:
    The input is a 2-D image ``(slit, wavelength)``; any dtype (converted to float32
    internally). For a pushbroom camera the focus that matters is along the spatial
    (slit) axis, so use ``axis=0`` (vertical edges).

Threading:
    Pure functions and a tiny stateful wrapper; no shared state. Call from any thread
    (the acquisition worker computes the score about five times per second).

Example:
    >>> import numpy as np
    >>> sharp = np.zeros((20, 20)); sharp[::2] = 100.0           # strong vertical edges
    >>> blurry = np.full((20, 20), 50.0)
    >>> sharpness_score(sharp, axis=0) > sharpness_score(blurry, axis=0)
    True
"""
from __future__ import annotations

import cv2
import numpy as np


def sharpness_score(image: np.ndarray, axis=None) -> float:
    """Return a focus score; larger means sharper.

    Args:
        image: 2-D array (any numeric dtype).
        axis: ``None`` for the variance of the Laplacian (all directions); ``0`` for
            the mean absolute vertical Sobel gradient (spatial/slit axis, use this
            for pushbroom focusing); ``1`` for the horizontal gradient (spectral axis).

    Returns:
        The score as a Python float. Scores are only comparable between images of
        the same size, exposure and scene.
    """
    data = np.asarray(image).astype(np.float32)
    if axis == 0:
        return float(np.mean(np.abs(cv2.Sobel(data, cv2.CV_32F, dx=0, dy=1, ksize=3))))
    if axis == 1:
        return float(np.mean(np.abs(cv2.Sobel(data, cv2.CV_32F, dx=1, dy=0, ksize=3))))
    return float(cv2.Laplacian(data, cv2.CV_32F).var())


class ImageSharpness:
    """Stateful convenience wrapper: ``ImageSharpness().LoadImage(a).SharpnessScore(axis=0)``.

    Kept for compatibility with existing callers; new code can call
    :func:`sharpness_score` directly.
    """

    img = None

    def LoadImage(self, data: np.ndarray) -> "ImageSharpness":
        """Remember ``data`` (no copy) and return ``self`` for chaining."""
        self.img = data
        return self

    def SharpnessScore(self, axis=None) -> float:
        """Score the loaded image; see :func:`sharpness_score` for ``axis``."""
        return sharpness_score(self.img, axis)
