"""NumPy stand-ins for the ``cv2.cuda`` and CuPy calls used by the GPU preprocessing backends.

They let the GPU backends' Python logic (caching, argument order, arithmetic
order, rounding) run on a machine without CUDA.  Every operation uses IEEE
float32/float64 NumPy arithmetic and the same rounding (half to even, then
saturate) as the device code, so a match here checks the algorithm, not the
OpenCV bindings or the CUDA C kernels; those still need a real GPU run.
"""
from types import SimpleNamespace
import cv2
import numpy as np

_DEPTH = {cv2.CV_16U: np.uint16, cv2.CV_32F: np.float32, cv2.CV_64F: np.float64}


def _saturate(values, dtype):
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(values), info.min, info.max).astype(dtype)
    return values.astype(dtype)


class Stream:
    """``cv2.cuda_Stream`` stand-in (work is done immediately)."""
    waits = 0

    def waitForCompletion(self):
        Stream.waits += 1


class GpuMat:
    """``cv2.cuda_GpuMat`` stand-in holding a NumPy array."""

    def __init__(self, *args):
        if len(args) == 3:                                   # (rows, cols, type): a preallocated matrix
            rows, cols, mat_type = args
            self.data = np.zeros((rows, cols), _DEPTH[mat_type])
        else:
            self.data = args[0] if args else None

    def upload(self, array, stream=None):
        self.data = np.array(array, copy=True)

    def download(self, stream=None):
        return np.array(self.data, copy=True)

    def convertTo(self, rtype, stream=None, dst=None):
        return _store(dst, _saturate(self.data, _DEPTH[rtype]), stream)


def _store(dst, values, stream):
    """Write a result into the caller's preallocated ``dst`` (strict: no per-frame allocation)."""
    assert stream is not None, 'frame operation without the backend stream'
    assert dst is not None and dst.data is not None, 'frame operation without a preallocated dst'
    assert dst.data.shape == values.shape and dst.data.dtype == values.dtype,         ('dst would be reallocated', dst.data.shape, dst.data.dtype, values.shape, values.dtype)
    dst.data = values
    return dst


def _same(a, b):
    assert a.data.dtype == b.data.dtype and a.data.shape == b.data.shape, (a.data.dtype, b.data.dtype)


class _LinearFilter:
    def __init__(self, kernel):
        assert kernel.shape == (3, 3) and np.all(kernel == 1)

    def apply(self, src, dst=None, stream=None):
        padded = np.pad(src.data, 1)
        h, w = src.data.shape
        total = np.zeros_like(src.data)
        for dy in range(3):
            for dx in range(3):
                total = total + padded[dy:dy + h, dx:dx + w]
        return _store(dst, total, stream)


def _subtract(a, b, dst=None, stream=None):
    _same(a, b)
    values = (_saturate(a.data.astype(np.float64) - b.data, a.data.dtype)
              if np.issubdtype(a.data.dtype, np.integer) else a.data - b.data)
    return _store(dst, values, stream)


def _arith(op):
    def run(a, b, dst=None, stream=None):
        _same(a, b)
        return _store(dst, op(a.data, b.data), stream)
    return run


def _remap(src, map_x, map_y, interpolation, dst=None, borderMode=None, borderValue=None, stream=None):
    assert interpolation == cv2.INTER_NEAREST and borderMode == cv2.BORDER_CONSTANT and borderValue == 0
    x, y = map_x.data, map_y.data
    assert np.array_equal(x, np.round(x)) and np.array_equal(y, np.round(y))
    x, y = x.astype(np.int64), y.astype(np.int64)
    h, w = src.data.shape
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    out = src.data[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)]
    return _store(dst, np.where(inside, out, 0).astype(src.data.dtype), stream)


def _flip(src, code, dst=None, stream=None):
    return _store(dst, np.ascontiguousarray(src.data[::-1] if code == 0 else src.data[:, ::-1]), stream)


def fake_cv2():
    """Return a ``cv2`` proxy whose ``cuda`` namespace is emulated."""
    cuda = SimpleNamespace(
        setDevice=lambda device: None,
        getCudaEnabledDeviceCount=lambda: 1,
        getDevice=lambda: 0,
        subtract=_subtract,
        add=_arith(np.add),
        multiply=_arith(np.multiply),
        divide=_arith(np.divide),
        flip=_flip,
        remap=_remap,
        createLinearFilter=lambda src_type, dst_type, kernel, borderMode: _LinearFilter(kernel),
    )

    class Proxy:
        def __getattr__(self, name):
            if name == 'cuda':
                return cuda
            if name == 'cuda_GpuMat':
                return GpuMat
            if name == 'cuda_Stream':
                return Stream
            return getattr(cv2, name)
    return Proxy()


# ── CuPy ────────────────────────────────────────────────────────────────


class DeviceArray(np.ndarray):
    """``cupy.ndarray`` stand-in: a NumPy array with ``get``/``set``."""

    def get(self):
        return np.array(self, copy=True).view(np.ndarray)

    def set(self, array):
        self[...] = array


def _device(array):
    return np.array(array, copy=True).view(DeviceArray)


def _remap16(src, dst, xy, frac, weights, n, width, height, taps, inverted, identity):
    src, out = src.view(np.ndarray).ravel(), dst.view(np.ndarray).reshape(-1)
    rows, cols = np.divmod(np.arange(int(n)), int(width))
    if identity:
        out[:] = src[(height - 1 - rows if inverted else rows) * width + cols]
        return
    pairs = xy.view(np.ndarray).reshape(-1, 2).astype(np.int64)
    x, y = pairs[:, 0], pairs[:, 1]
    f = frac.view(np.ndarray).ravel() & 31
    table = weights.view(np.ndarray).reshape(-1, int(taps))
    value = np.zeros(int(n), np.float32)
    for k in range(int(taps)):
        sx = x + k - (1 if taps == 4 else 0)
        inside = (sx >= 0) & (sx < width) & (y >= 0) & (y < height)
        v = np.where(inside, src[np.clip(y, 0, height - 1) * width + np.clip(sx, 0, width - 1)], 0).astype(np.float32)
        value = (value + v * table[f, k]).astype(np.float32)
    out[:] = np.rint(np.clip(value, 0, 65535)).astype(np.uint16)


def _bin16(src, dst, left, right, count, rows, width, bins):
    image = src.view(np.ndarray).reshape(int(rows), int(width)).astype(np.uint64)
    csum = np.concatenate([np.zeros((int(rows), 1), np.uint64), np.cumsum(image, axis=1)], axis=1)
    left, right = left.view(np.ndarray), right.view(np.ndarray)
    w = count.view(np.ndarray).astype(np.uint64)
    sums = (csum[:, right] - csum[:, left]).astype(np.uint32)       # the kernel's uint32 sum
    value = (sums.astype(np.uint64) + w // 2) // w
    dst.view(np.ndarray).reshape(int(rows), int(bins))[:] = np.minimum(value, 65535).astype(np.uint16)


def _dark16(src, dark, dst, n):
    s, d = src.view(np.ndarray).reshape(-1), dark.view(np.ndarray).reshape(-1)
    dst.view(np.ndarray).reshape(-1)[:] = np.where(s > d, s - d, 0).astype(np.uint16)


def _hot16(img, target, neighbour, count, n):
    flat = img.view(np.ndarray).reshape(-1)                 # a view: writes reach ``img``
    index = neighbour.view(np.ndarray).reshape(-1, 8)
    values = np.where(index >= 0, flat[np.clip(index, 0, None)], 0).astype(np.float32)
    total = np.zeros(index.shape[0], np.float32)
    for k in range(8):                                      # the kernel's sequential float32 sum
        total = (total + values[:, k]).astype(np.float32)
    average = total.astype(np.float64) / count.view(np.ndarray).astype(np.float64)
    flat[target.view(np.ndarray)] = np.rint(average).astype(np.uint16)


_KERNELS = {'remap16': _remap16, 'bin16': _bin16, 'dark16': _dark16, 'hot16': _hot16}


class _RawKernel:
    def __init__(self, source, name):
        assert f'void {name}(' in source
        self.run = _KERNELS[name]

    def __call__(self, grid, block, args):
        self.run(*args)


class _Event:
    def record(self): pass
    def synchronize(self): pass


class _Device:
    def __init__(self, index): pass
    def use(self): pass
    def __enter__(self): return self
    def __exit__(self, *args): return False


class PinnedMemory:
    """``cupy.cuda.PinnedMemory`` stand-in: host bytes that the "device" views share."""
    allocations = 0

    def __init__(self, size, flags=0):
        assert flags == 2, 'zero-copy needs cudaHostAllocMapped'
        self.array = np.zeros(size, np.uint8)
        PinnedMemory.allocations += 1


class _UnownedMemory:
    def __init__(self, ptr, size, owner):
        assert ptr == owner.array.__array_interface__['data'][0] and size == owner.array.size
        self.owner = owner


def _device_view(shape, dtype, memptr):
    """``cupy.ndarray(shape, dtype, MemoryPointer(UnownedMemory(...), 0))``: same bytes as the host."""
    return memptr.owner.array.view(dtype).reshape(shape).view(DeviceArray)


def fake_cupy():
    """Return a ``cupy`` module stand-in covering the calls of ``CuPyCUDABackend``."""
    return SimpleNamespace(
        cuda=SimpleNamespace(Device=_Device, Event=_Event, get_elapsed_time=lambda start, end: 0.25,
                             PinnedMemory=PinnedMemory, PinnedMemoryPointer=lambda memory, offset: memory.array,
                             UnownedMemory=_UnownedMemory, MemoryPointer=lambda memory, offset: memory,
                             runtime=SimpleNamespace(hostAllocMapped=2)),
        RawKernel=_RawKernel,
        ndarray=_device_view,
        uint16=np.uint16, int16=np.int16, float32=np.float32, float64=np.float64,
        empty=lambda shape, dtype: np.empty(shape, dtype).view(DeviceArray),
        zeros=lambda shape, dtype: np.zeros(shape, dtype).view(DeviceArray),
        ones=lambda shape, dtype: np.ones(shape, dtype).view(DeviceArray),
        asarray=_device,
        maximum=np.maximum, subtract=np.subtract, where=np.where, rint=np.rint,
    )
