"""Process-wide, immutable compute capabilities detected once at startup.

Availability is distinct from the active processing backend. Detection does not
switch numerical kernels, create persistent GPU buffers, or run in the frame loop.

Data conventions:
    No array data.  A *capability* says whether a CUDA backend (OpenCV CUDA or
    CuPy) is usable on this machine; the *processing backend* is a string
    (``'cpu'``, ``'opencv_cuda'`` or ``'cupy_cuda'``) published by whoever
    actually runs the frame pipeline (``modules.preprocessing``).  Device ids
    are CUDA ordinals (always 0 here); names are plain ``str``.

Threading:
    :func:`initialize_compute` / :func:`get_compute_capabilities` are
    thread-safe (guarded by a lock) and probe at most once per process.  The
    returned dataclasses are frozen.  :func:`set_processing_backend` just
    rebinds a module global (atomic) and may be called from any thread.

Example:
    >>> from modules.compute_runtime import get_compute_capabilities
    >>> caps = get_compute_capabilities()           # probes once, CPU fallback is normal
    >>> isinstance(caps.cuda_available, bool) and caps.processing_backend in (
    ...     'cpu', 'opencv_cuda', 'cupy_cuda')
    True
"""
from dataclasses import dataclass
import logging
import threading

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class BackendCapability:
    """Immutable result of probing one CUDA backend.

    Attributes:
        available: True only if a smoke test on the device succeeded.
        device_count: Number of CUDA devices the library reports (0 if none).
        device_id: Ordinal of the device that was verified, ``None`` if unavailable.
        device_name: Device name, ``''`` if unavailable.
        reason: Human-readable outcome (why unavailable, or what was verified).
    """
    available: bool = False
    device_count: int = 0
    device_id: int | None = None
    device_name: str = ''
    reason: str = ''

@dataclass(frozen=True)
class ComputeCapabilities:
    """Immutable snapshot of the compute capabilities of this process.

    Attributes:
        opencv_cuda: Capability of OpenCV's CUDA module (uint16 ``remap`` verified).
        cupy_cuda: Capability of CuPy (allocation, kernel and download verified).
    """
    opencv_cuda: BackendCapability
    cupy_cuda: BackendCapability

    @property
    def cuda_available(self):
        """bool: True if either CUDA backend is available."""
        return self.opencv_cuda.available or self.cupy_cuda.available

    @property
    def processing_backend(self):
        """str: Currently published processing backend (``'cpu'`` until
        :func:`set_processing_backend` is called); read live, not frozen."""
        return _processing_backend

_processing_backend = 'cpu'
_capabilities = None
_lock = threading.Lock()

def _probe_opencv():
    count = 0
    try:
        import cv2
        import numpy as np
        count = int(cv2.cuda.getCudaEnabledDeviceCount())
        if count <= 0:
            return BackendCapability(reason='No CUDA device/build support' if count == 0 else 'CUDA driver unavailable/incompatible')
        if not callable(getattr(cv2.cuda, 'remap', None)):
            return BackendCapability(device_count=count, reason='OpenCV CUDA remap is unavailable')
        previous = cv2.cuda.getDevice()
        try:
            cv2.cuda.setDevice(0)
            source = np.arange(16, dtype=np.uint16).reshape(4, 4)
            y, x = np.indices(source.shape, dtype=np.float32)
            gpu_source, gpu_x, gpu_y = (cv2.cuda_GpuMat() for _ in range(3))
            gpu_source.upload(source); gpu_x.upload(x); gpu_y.upload(y)
            output = cv2.cuda.remap(gpu_source, gpu_x, gpu_y, interpolation=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0).download()
            if output.dtype != source.dtype or not np.array_equal(output, source):
                raise RuntimeError('uint16 remap smoke test failed')
            try:
                name = cv2.cuda.DeviceInfo(0).name()
            except Exception:
                name = 'CUDA device 0'
            return BackendCapability(True, count, 0, str(name), 'uint16 remap verified')
        finally:
            cv2.cuda.setDevice(previous)
    except Exception as exc:
        return BackendCapability(device_count=max(count, 0), reason=f'{type(exc).__name__}: {exc}')

def _probe_cupy():
    count = 0
    try:
        import cupy as cp
        count = int(cp.cuda.runtime.getDeviceCount())
        if count <= 0:
            return BackendCapability(reason='No CUDA device visible to CuPy')
        with cp.cuda.Device(0):
            source = cp.arange(4, dtype=cp.uint16)
            output = source + cp.uint16(1)
            cp.cuda.get_current_stream().synchronize()
            if output.get().tolist() != [1, 2, 3, 4]:
                raise RuntimeError('CuPy kernel smoke test failed')
            properties = cp.cuda.runtime.getDeviceProperties(0)
            name = properties.get('name', 'CUDA device 0')
            if isinstance(name, bytes):
                name = name.decode('utf-8', errors='replace')
            return BackendCapability(True, count, 0, str(name), 'Allocation/kernel/download verified')
    except Exception as exc:
        return BackendCapability(device_count=max(count, 0), reason=f'{type(exc).__name__}: {exc}')

def initialize_compute():
    """Detect once per process; missing libraries/devices are normal CPU fallback.

    Runs small smoke tests (device 0) for OpenCV CUDA and CuPy on the first call
    and caches the result; later calls return the cached object.  Probe failures
    never raise - they are recorded in ``BackendCapability.reason``.  This may
    take noticeable time (CUDA context creation), so call it at start-up, not in
    the frame loop.

    Returns:
        ComputeCapabilities: The shared, immutable snapshot.
    """
    global _capabilities
    with _lock:
        if _capabilities is None:
            _capabilities = ComputeCapabilities(_probe_opencv(), _probe_cupy())
            for name, capability in (('opencv', _capabilities.opencv_cuda), ('cupy', _capabilities.cupy_cuda)):
                logger.info('CUDA %s: available=%s device=%s reason=%s', name,
                            capability.available, capability.device_name or '-', capability.reason)
            logger.info('Initial processing backend: cpu (CUDA availability=%s)', _capabilities.cuda_available)
        return _capabilities

def get_compute_capabilities():
    """Return the shared snapshot; standalone callers initialize lazily once.

    Returns:
        ComputeCapabilities: Same object as :func:`initialize_compute`.
    """
    if _capabilities is not None:
        return _capabilities
    return initialize_compute()

def set_processing_backend(name):
    """Publish the active acquisition runtime's backend, distinct from capability.

    Args:
        name: One of ``'cpu'``, ``'opencv_cuda'``, ``'cupy_cuda'``.

    Raises:
        ValueError: if ``name`` is not one of the three allowed values.
    """
    global _processing_backend
    if name not in ('cpu', 'opencv_cuda', 'cupy_cuda'):
        raise ValueError(name)
    _processing_backend = name

__all__ = ['BackendCapability', 'ComputeCapabilities', 'initialize_compute', 'get_compute_capabilities',
           'set_processing_backend']
__requires__ = ()
