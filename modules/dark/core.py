"""Cached dark-frame subtraction and hot-pixel repair (Qt-free).

Data conventions:
    Frames are 2-D ``(slit, wavelength)`` sensor images (raw camera frames),
    typically ``uint8``/``uint16``. The dark mean must have the same shape and
    dtype as the frames it is subtracted from. ``hot_pixel_map`` is a 2-D mask of
    the same shape where non-zero means "hot" (``DarkFrameLibrary`` writes
    ``uint8`` with 255). Subtraction saturates at 0 for unsigned integer
    dtypes (``cv2.subtract``).

    Hot-pixel repair mutates the supplied frame in place, as in the legacy
    backend: each hot pixel is replaced by the rounded mean of its non-hot
    8-neighbours that lie inside the image.

Threading:
    A ``DarkProcessor`` has no locks.  ``set_dark_reference`` builds a complete
    :class:`DarkReference` (dark mean and hot-pixel tables) and installs it with
    one assignment, so it may be called from another thread (the GUI) while
    frames are processed.  A frame loop reads ``dark_reference`` once per frame
    and uses only that object, so it sees either the old or the new reference,
    never a mix.  The ``_dark_enabled`` flag is a separate single attribute.

Example:
    >>> import numpy as np
    >>> from modules.dark import DarkProcessor
    >>> dark = np.full((4, 4), 10, np.uint16)
    >>> hot = np.zeros((4, 4), np.uint8); hot[1, 1] = 255
    >>> proc = DarkProcessor(); proc.set_dark_reference(dark, hot)
    >>> frame = np.full((4, 4), 110, np.uint16); frame[1, 1] = 65535
    >>> out = proc._apply_hot_pixel(proc._apply_dark_subtract(frame))
    >>> int(out[0, 0]), int(out[1, 1])
    (100, 100)
"""
import cv2
import numpy as np

_OFFSETS = np.array([
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
])


class DarkReference:
    """One dark mean with its hot-pixel tables; never modified after construction.

    Attribute names match the duck-typed interface the frame pipeline reads
    (``modules.preprocessing``), so a reference can be passed wherever a
    ``DarkProcessor`` was.

    Attributes:
        _dark_mean: 2-D dark mean frame (original dtype) or ``None``.
        _hot_pixel_map: 2-D hot-pixel mask or ``None``.
        _hot_coords: ``(n, 2)`` ``(y, x)`` coordinates of hot pixels, or ``None``
            when there are none.
        _hot_ys, _hot_xs: ``(n, 8)`` clipped neighbour coordinates.
        _hot_valid: ``(n, 8)`` mask of neighbours inside the image and not hot.
        _neighbor_count: ``(n, 1)`` number of valid neighbours (at least 1).
    """

    def __init__(self, dark_mean=None, hot_pixel_map=None):
        """Build the reference and its neighbour tables.

        Args:
            dark_mean: 2-D dark mean, or ``None``.  Stored by reference, not copied.
            hot_pixel_map: 2-D mask (non-zero = hot) of the same shape, or ``None``.
                ``None`` or an all-zero mask disables hot-pixel repair.
        """
        self._dark_mean = dark_mean
        self._hot_pixel_map = hot_pixel_map
        self._hot_coords = None
        self._hot_ys = None
        self._hot_xs = None
        self._hot_valid = None
        self._neighbor_count = None
        if hot_pixel_map is None:
            return
        coords = np.column_stack(np.nonzero(hot_pixel_map))
        if coords.size == 0:
            return
        # A neighbour is valid if it is inside the image and is not itself hot.
        h, w = hot_pixel_map.shape
        ys = coords[:, 0:1] + _OFFSETS[:, 0]
        xs = coords[:, 1:2] + _OFFSETS[:, 1]
        in_bounds = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
        ys_clipped = np.clip(ys, 0, h - 1)
        xs_clipped = np.clip(xs, 0, w - 1)
        neighbor_is_hot = hot_pixel_map[ys_clipped, xs_clipped].astype(bool)
        valid = in_bounds & ~neighbor_is_hot
        self._hot_coords = coords
        self._hot_ys = ys_clipped
        self._hot_xs = xs_clipped
        self._hot_valid = valid
        self._neighbor_count = np.maximum(valid.sum(axis=1, keepdims=True), 1)

    def _apply_dark_subtract(self, frame):
        """Subtract the dark mean from ``frame``.

        Args:
            frame: 2-D frame, same dtype as the dark mean.

        Returns:
            numpy.ndarray: new array ``frame - dark`` (saturating at 0 for unsigned
            dtypes); ``frame`` itself (not a copy) if no dark is set; a copy of
            ``frame`` if the dark shape differs.
        """
        if self._dark_mean is None:
            return frame
        if self._dark_mean.shape != frame.shape:
            return frame.copy()
        return cv2.subtract(frame, self._dark_mean)

    def _apply_hot_pixel(self, frame):
        """Replace hot pixels by the mean of their valid neighbours, in place.

        Args:
            frame: 2-D frame with the same shape as the hot-pixel map. Modified
                in place.

        Returns:
            numpy.ndarray: ``frame`` (the same object).
        """
        if self._hot_coords is None:
            return frame

        neighbor_vals = frame[self._hot_ys, self._hot_xs].astype(np.float32)
        neighbor_vals = np.where(self._hot_valid, neighbor_vals, 0.0)
        neighbor_avg = neighbor_vals.sum(axis=1) / self._neighbor_count.ravel()

        frame[self._hot_coords[:, 0], self._hot_coords[:, 1]] = np.round(
            neighbor_avg
        ).astype(frame.dtype)
        return frame


_EMPTY_REFERENCE = DarkReference()


def _reference_attribute(name):
    return property(lambda self: getattr(self._reference, name),
                    doc=f"``{name}`` of the current :class:`DarkReference` (read-only).")


class DarkProcessor:
    """Holds a dark-mean frame and hot-pixel map and applies them to frames.

    The per-hot-pixel neighbour indices are precomputed in ``set_dark_reference``
    so that repair is a few vectorised numpy operations per frame. Camera
    backends call the ``_apply_dark_subtract`` / ``_apply_hot_pixel`` steps
    directly from their frame loop.  The current reference is the immutable
    :attr:`dark_reference`; the ``_dark_mean``/``_hot_*`` attributes read it.
    """

    _dark_mean = _reference_attribute('_dark_mean')
    _hot_pixel_map = _reference_attribute('_hot_pixel_map')
    _hot_coords = _reference_attribute('_hot_coords')
    _hot_ys = _reference_attribute('_hot_ys')
    _hot_xs = _reference_attribute('_hot_xs')
    _hot_valid = _reference_attribute('_hot_valid')
    _neighbor_count = _reference_attribute('_neighbor_count')

    def __init__(self):
        """Create a processor with no dark reference (frames pass through unchanged)."""
        self._reference = _EMPTY_REFERENCE
        self._dark_enabled: bool = True

    @property
    def dark_reference(self) -> DarkReference:
        """DarkReference: The current reference.  Read it once per frame and use
        only that object, so a concurrent ``set_dark_reference`` cannot mix two
        references within one frame."""
        return self._reference

    def set_dark_enabled(self, enabled: bool) -> None:
        """Store the dark-enabled flag (``bool(enabled)``).

        Args:
            enabled: Whether dark correction is wanted by the caller.

        Note:
            The flag is only stored; the ``_apply_*`` methods do not read it, so
            callers decide whether to invoke them.
        """
        self._dark_enabled = bool(enabled)


    def set_dark_reference(self, dark_mean, hot_pixel_map):
        """Install a dark reference and precompute hot-pixel neighbour tables.

        The new :class:`DarkReference` is built completely before it replaces the
        current one in a single assignment.

        Args:
            dark_mean: 2-D ``(slit, wavelength)`` dark mean frame (original frame
                dtype), or ``None`` to disable subtraction. Stored by reference,
                not copied; do not modify it afterwards.
            hot_pixel_map: 2-D mask of the same shape (non-zero = hot pixel) or
                ``None``. Stored by reference. If it is ``None`` or has no
                non-zero entries, hot-pixel repair is disabled.

        Returns:
            None
        """
        self._reference = DarkReference(dark_mean, hot_pixel_map)


    def _apply_dark_subtract(self, frame):
        """Subtract the dark mean from ``frame`` (see :meth:`DarkReference._apply_dark_subtract`)."""
        return self._reference._apply_dark_subtract(frame)


    def _apply_hot_pixel(self, frame):
        """Repair hot pixels in place (see :meth:`DarkReference._apply_hot_pixel`)."""
        return self._reference._apply_hot_pixel(frame)
