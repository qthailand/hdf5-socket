"""Cached OpenCV-compatible horizontal Smile interpolation tables."""
import cv2
import numpy as np

FLAGS = {'nearest': cv2.INTER_NEAREST, 'linear': cv2.INTER_LINEAR, 'cubic': cv2.INTER_CUBIC, 'lanczos': cv2.INTER_LANCZOS4}

def prepare_maps(calibration, inverted):
    """Build OpenCV remap maps for a Smile calibration.

    Args:
        calibration: Smile calibration with ``image_shape`` ``(h, w)``,
            ``map_x``, ``_get_map_y()`` and ``interp`` (``'nearest'``, ``'linear'``,
            ``'cubic'`` or ``'lanczos'``).
        inverted: If true the slit axis is flipped by remapping ``y`` to
            ``h - 1 - y``.

    Returns:
        tuple: ``(x, y, fixed, fractions, flag)`` - float32 ``(h, w)`` maps in
        pixel units, the fixed-point ``CV_16SC2`` map and its interpolation
        fractions (``cv2.convertMaps`` output) and the ``cv2.INTER_*`` flag.
        Arrays are new (not views of the calibration) unless the calibration's
        maps are already contiguous float32.

    Raises:
        KeyError: unknown interpolation name.
    """
    h, w = calibration.image_shape
    x = np.ascontiguousarray(calibration.map_x, dtype=np.float32)
    y = calibration._get_map_y()
    if inverted:
        y = np.ascontiguousarray(h - 1 - y, dtype=np.float32)
    flag = FLAGS[calibration.interp]
    fixed, fractions = cv2.convertMaps(x, y, cv2.CV_16SC2, nninterpolation=flag == cv2.INTER_NEAREST)
    return x, y, fixed, fractions, flag

def horizontal_table(interp):
    """Use OpenCV's own float tables rather than a different cubic convention.

    Args:
        interp: ``'nearest'``, ``'linear'`` or ``'cubic'`` (anything else is
            treated as 4-tap cubic).

    Returns:
        numpy.ndarray: float32 weights ``(32, taps)`` (taps = 1, 2 or 4), one
        row per 1/32 sub-pixel fraction, as used by the CuPy kernel.
    """
    if interp == 'nearest':
        return np.ones((32, 1), dtype=np.float32)
    count = 2 if interp == 'linear' else 4
    origin = 0 if count == 2 else 1
    x = (origin + np.arange(32, dtype=np.float32) / 32).reshape(1, 32)
    y = np.zeros_like(x)
    weights = []
    for index in range(count):
        basis = np.zeros((1, count), dtype=np.float32)
        basis[0, index] = 1
        weights.append(cv2.remap(basis, x, y, FLAGS[interp], borderMode=cv2.BORDER_CONSTANT)[0])
    return np.ascontiguousarray(np.stack(weights, axis=1))
