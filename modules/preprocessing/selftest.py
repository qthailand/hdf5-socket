"""Start-up check that a GPU backend runs the whole path (correction and binning).

``PreprocessingRuntime`` runs :func:`self_test` once on each candidate GPU
backend before accepting it, so a provider with a missing feature (for example
OpenCV without cuBLAS) or a differing result is skipped for the next GPU
provider instead of being found on the first real frame.

Data conventions:
    Small synthetic uint16 frames with a dark, hot pixels (corners, edges,
    adjacent pairs), a horizontal cubic Smile that samples past both borders
    and lambda bins of odd and even widths.  Tolerance is the runtime's: one
    count for corrected and binned images, exact pre-Smile Grab.
"""
import numpy as np

_SHAPE = (7, 41)


class _HotPixels:
    """Dark-processor stand-in with the neighbour tables of ``modules.dark``."""

    def __init__(self, dark, hot_map):
        self._dark_enabled = True
        self._dark_mean = dark
        self._hot_coords = np.column_stack(np.nonzero(hot_map))
        offsets = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])
        h, w = hot_map.shape
        ys = self._hot_coords[:, 0:1] + offsets[:, 0]
        xs = self._hot_coords[:, 1:2] + offsets[:, 1]
        inside = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
        self._hot_ys = np.clip(ys, 0, h - 1)
        self._hot_xs = np.clip(xs, 0, w - 1)
        self._hot_valid = inside & ~hot_map[self._hot_ys, self._hot_xs].astype(bool)
        self._neighbor_count = np.maximum(self._hot_valid.sum(axis=1, keepdims=True), 1)

    def _apply_hot_pixel(self, frame):
        values = np.where(self._hot_valid, frame[self._hot_ys, self._hot_xs].astype(np.float32), 0.0)
        average = values.sum(axis=1) / self._neighbor_count.ravel()
        frame[self._hot_coords[:, 0], self._hot_coords[:, 1]] = np.round(average).astype(frame.dtype)
        return frame


class _Smile:
    """Horizontal-only Smile model with integer rows."""

    def __init__(self, shape):
        self.image_shape = shape
        self.interp = 'cubic'
        y, x = np.indices(shape, dtype=np.float32)
        self.map_x = np.ascontiguousarray(x + 2.3 * np.sin(y / 2 + x / 9) - 0.4, dtype=np.float32)
        self._map_y = np.ascontiguousarray(y)

    def _get_map_y(self):
        return self._map_y


class _Bins:
    """``LambdaBinner`` stand-in: contiguous bins of widths 1-4 pixels."""

    def __init__(self, width):
        widths, start = [], 0
        while start < width:
            widths.append(min(1 + len(widths) % 4, width - start))
            start += widths[-1]
        self.w = np.array(widths, dtype=np.int32)
        self.iR = np.cumsum(self.w).astype(np.int32)
        self.iL = (self.iR - self.w).astype(np.int32)
        self.v_ = width
        self.n_bins = int(self.w.size)

    def transform(self, spectrogram):
        """Same integer binning as ``LambdaBinner.transform``."""
        csum = np.cumsum(spectrogram, axis=-1, dtype=np.int64)
        csum = np.concatenate([np.zeros(csum.shape[:-1] + (1,), np.int64), csum], axis=-1)
        sums = csum[..., self.iR] - csum[..., self.iL]
        return np.minimum((sums + self.w // 2) // self.w, 65535).astype(np.uint16)


def _max_error(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise RuntimeError(f'self-test layout {actual.shape}/{actual.dtype} differs from CPU {expected.shape}/{expected.dtype}')
    return int(np.max(np.abs(actual.astype(np.int32) - expected.astype(np.int32))))


def self_test(backend, cpu, profiler, zero_copy=False):
    """Run ``backend`` against ``cpu`` on synthetic data.

    Args:
        backend: GPU backend with ``process_binned``.
        cpu: ``CPUBackend`` reference.
        profiler: No-op profiler passed to both backends.
        zero_copy: Pass each frame through ``backend.input_buffer`` (read in place
            by the GPU) and also check that the buffer is left unmodified, including
            hot-pixel repair without a dark.

    Raises:
        RuntimeError: a result differs by more than one count, the Grab differs, or
            the zero-copy input buffer was modified or not provided.
        Exception: whatever the backend raises for an unsupported operation.
    """
    rng = np.random.default_rng(1729)
    frame = rng.integers(0, 65536, _SHAPE, dtype=np.uint16)
    frame[2, 10:14] = 65535                       # saturated run: rounding/clamp at the top
    dark = rng.integers(0, 4000, _SHAPE, dtype=np.uint16)
    hot_map = np.zeros(_SHAPE, dtype=np.uint8)
    hot_map[[0, 0, 3, 3, 6, 4], [0, 1, 20, 21, 40, 7]] = 1
    hot = _HotPixels(dark, hot_map)
    bins = _Bins(_SHAPE[1])
    darks = (dark, None) if zero_copy else (dark,)    # no dark: hot pixels repaired from the input itself
    for calibration in (None, _Smile(_SHAPE)):
        for inverted in (False, True):
            for frame_dark in darks:
                source = frame
                if zero_copy:
                    source = backend.input_buffer(_SHAPE)
                    if source is None:
                        raise RuntimeError('zero-copy input buffer not provided')
                    source[...] = frame
                args = (source, frame_dark, hot, True, inverted, calibration, True)
                binned, output, grabbed = backend.process_binned(*args, bins, True, profiler)
                if zero_copy and not np.array_equal(source, frame):
                    raise RuntimeError('zero-copy input buffer was modified')
                reference, reference_grab = cpu.process(*args, profiler)
                error = max(_max_error(output, reference),
                            _max_error(binned, bins.transform(reference[None])[0]))
                if error > 1:
                    raise RuntimeError(f'self-test differs from CPU by {error} counts (allowed 1)')
                if not np.array_equal(grabbed, reference_grab):
                    raise RuntimeError('self-test Grab differs from CPU pre-Smile image')
