"""CPU reference with reusable subtraction storage and fused orientation/remap."""
import cv2
import numpy as np
from .maps import prepare_maps

class CPUBackend:
    """Reference CPU preprocessing backend (OpenCV / NumPy).

    Keeps a reusable subtraction buffer and the last remap maps, so an instance
    must be used from one thread at a time.
    """
    name = 'cpu'
    def __init__(self):
        """Create an empty backend (buffers are allocated on first use)."""
        self._buffer = None
        self._map_key = None
        self._maps = None

    def process(self, frame, dark, hot, enabled, inverted, calibration, snapshot, profiler):
        """Dark-subtract, repair hot pixels, flip and Smile-remap one frame.

        Args:
            frame: 2-D ``(slit, wavelength)`` frame; not modified.
            dark: Dark mean of the same shape, or ``None``.
            hot: Dark-processor-like object with ``_hot_coords`` and
                ``_apply_hot_pixel``.
            enabled: Whether dark/hot correction is applied.
            inverted: Flip the slit axis.
            calibration: Smile calibration or ``None``.
            snapshot: Also return the pre-Smile image.
            profiler: Object with ``time(name)`` context manager.

        Returns:
            tuple: ``(output, grabbed)`` - new C-contiguous arrays (``grabbed`` is
            ``None`` unless ``snapshot``).  For linear/cubic/lanczos Smile the
            flip is folded into the remap, for ``'nearest'`` the calibration's
            own ``apply_native`` is used.
        """
        with profiler.time('dark'):
            if enabled and dark is not None and dark.shape == frame.shape:
                if self._buffer is None or self._buffer.shape != frame.shape or self._buffer.dtype != frame.dtype:
                    self._buffer = np.empty_like(frame)
                corrected = cv2.subtract(frame, dark, dst=self._buffer)
            else:
                corrected = frame
        with profiler.time('hot_pixel'):
            if enabled and hot._hot_coords is not None:
                if corrected is frame:
                    corrected = frame.copy()
                corrected = hot._apply_hot_pixel(corrected)
        oriented = corrected[::-1] if inverted else corrected
        grabbed = np.ascontiguousarray(oriented).copy() if snapshot else None
        with profiler.time('orient_smile'):
            if calibration is None:
                output = np.array(oriented, copy=True, order='C')
            elif calibration.interp == 'nearest':
                # Preserve the original fixed-map nearest behavior exactly.
                output = calibration.apply_native(np.ascontiguousarray(oriented))
            else:
                key = (id(calibration), inverted)
                if key != self._map_key:
                    _, _, fixed, fractions, flag = prepare_maps(calibration, inverted)
                    self._maps = (fixed, fractions, flag)
                    self._map_key = key
                    self._calibration_ref = calibration
                fixed, fractions, flag = self._maps
                output = cv2.remap(corrected, fixed, fractions, flag, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return output, grabbed
