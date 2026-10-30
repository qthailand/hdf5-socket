"""Shared preprocessing API with CPU reference and optional CUDA backends.

Outputs are owned contiguous arrays. GPU failures latch CPU fallback for this
runtime. New GPU configurations are checked against the CPU reference once;
integer interpolation may differ by at most one uint16 count.

Data conventions:
    A sensor frame is a 2-D ``(slit, wavelength)`` uint16 array (CUDA backends
    require uint16; the CPU backend accepts any dtype).  The pipeline per frame
    is: dark subtraction (saturating at 0) -> hot-pixel repair -> optional slit
    flip (``inverted``) -> optional Smile remap.  ``process`` returns
    ``(output, grabbed)``: ``output`` is the corrected ``(slit, wavelength)``
    image, ``grabbed`` is the image after dark/hot/flip but *before* Smile (only
    when ``snapshot`` is true, else ``None``).  Both are new C-contiguous arrays
    owned by the caller.

    ``process_binned`` continues with lambda binning and returns
    ``(binned, output, grabbed)``: ``binned`` is the ``(slit, n_bins)`` uint16
    result of ``binner.transform``; ``output`` is the corrected image only when
    ``full`` is true (else ``None``).  On a GPU backend the whole path from
    upload to binning stays on the device; only ``binned`` (and ``output`` /
    ``grabbed`` when requested) is downloaded.

    The ``dark_processor`` and ``calibration`` arguments are duck-typed (no
    import): a ``modules.dark.DarkProcessor`` provides ``_dark_enabled``,
    ``_dark_mean`` (same shape as the frame), ``_hot_coords`` and
    ``_apply_hot_pixel``, read from its immutable ``dark_reference`` once per
    frame when it has one; a Smile calibration provides ``image_shape``,
    ``interp``, ``map_x``, ``_get_map_y()`` and ``apply_native``; a binner
    (``modules.spectral_processing.LambdaBinner``) provides ``v_``, ``iL``,
    ``iR``, ``w``, ``n_bins`` and ``transform``.

Threading:
    Create one :class:`PreprocessingRuntime` per acquisition worker thread and
    call ``process`` only from that thread: the backend is chosen lazily on the
    first call so that CUDA contexts belong to that thread.  Instances are not
    thread-safe.

Example:
    >>> import numpy as np
    >>> from types import SimpleNamespace
    >>> from modules.preprocessing import PreprocessingRuntime
    >>> rt = PreprocessingRuntime(backend='cpu')
    >>> idle = SimpleNamespace(_dark_enabled=False, _dark_mean=None, _hot_coords=None)
    >>> out, grab = rt.process(np.zeros((4, 6), np.uint16), idle, inverted=True)
    >>> out.shape, grab, rt.backend_name
    ((4, 6), None, 'cpu')
"""
import logging
import numpy as np
from modules.compute_runtime import get_compute_capabilities, set_processing_backend
from .cpu import CPUBackend
from . import cupy_cuda, opencv_cuda         # cupy itself is imported only when its backend is created
from .selftest import self_test

logger = logging.getLogger(__name__)

# Candidates tried in order; every GPU candidate runs the whole path itself.
_ORDER = {
    'auto': ('cupy_cuda', 'opencv_cuda', 'cpu'),
    'cupy_cuda': ('cupy_cuda', 'opencv_cuda', 'cpu'),
    'opencv_cuda': ('opencv_cuda', 'cupy_cuda', 'cpu'),
    'cpu': ('cpu',),
}

class PreprocessingRuntime:
    """Per-frame preprocessing with CPU reference and optional CUDA backends.

    Attributes:
        requested: Backend requested at construction.
        capabilities: ``ComputeCapabilities`` used to choose a backend.
        fallback_reason: Message of the last backend initialisation/validation
            failure that caused a fallback (``''`` if none).
    """

    def __init__(self, backend='auto', capabilities=None):
        """Create a runtime; no backend is initialised until the first frame.

        Args:
            backend: ``'auto'`` (CuPy, then OpenCV CUDA, then CPU), ``'cpu'``,
                ``'opencv_cuda'`` (then CuPy, then CPU) or ``'cupy_cuda'`` (then
                OpenCV CUDA, then CPU).  A GPU candidate is used only if its
                capability is available and it passes a start-up self-test of
                the whole path, binning included.
            capabilities: A ``ComputeCapabilities`` to use instead of the
                process-wide one from ``modules.compute_runtime``.

        Raises:
            ValueError: unknown ``backend`` name.
        """
        if backend not in _ORDER:
            raise ValueError(f'Unknown preprocessing backend: {backend}')
        self.requested = backend
        self.capabilities = capabilities or get_compute_capabilities()
        self._cpu = CPUBackend()
        self._backend = None
        self._validated = None
        self.fallback_reason = ''

    @property
    def backend_name(self):
        """str: Active backend (``'cpu'``, ``'opencv_cuda'``, ``'cupy_cuda'``) or
        ``'uninitialized'`` before the first ``process`` call."""
        return self._backend.name if self._backend is not None else 'uninitialized'

    def _select(self):
        for name in _ORDER[self.requested]:
            if name == 'cpu':
                self._backend = self._cpu
                break
            capability = getattr(self.capabilities, name)
            if not capability.available:
                continue
            try:
                backend_class = (cupy_cuda.CuPyCUDABackend if name == 'cupy_cuda'
                                 else opencv_cuda.OpenCVCUDABackend)
                backend = backend_class(capability.device_id)
                self_test(backend, self._cpu, _NoProfiler())
                self._backend = backend
                break
            except Exception as exc:
                logger.warning('Cannot use %s: %s', name, exc)
                self.fallback_reason = f'{name}: {exc}'
        if hasattr(self._backend, 'input_buffer'):
            try:
                self_test(self._backend, self._cpu, _NoProfiler(), zero_copy=True)
            except Exception as exc:
                # The backend itself passed: keep it, with ordinary uploads.
                self._backend.disable_zero_copy()
                logger.warning('%s zero-copy input disabled: %s', self.backend_name, exc)
        logger.info('Preprocessing backend: %s (requested=%s)', self.backend_name, self.requested)
        self._publish_backend()

    def _publish_backend(self):
        set_processing_backend(self.backend_name)

    def input_buffer(self, shape):
        """Reusable host buffer for the next frame that the active backend reads in place.

        Only the CuPy backend offers one: mapped pinned memory that the GPU reads with
        no upload (one DRAM on Jetson).  A frame written there and passed to
        :meth:`process_binned` skips the host-to-device copy.  The buffer is overwritten
        by the next frame of that shape, so a caller that keeps a frame must copy it.

        Args:
            shape: ``(height, width)`` of the frame.

        Returns:
            numpy.ndarray | None: Writable C-contiguous ``uint16`` array, or ``None``
            before a backend is selected, for CPU/OpenCV CUDA, or if zero-copy is disabled.
        """
        provide = getattr(self._backend, 'input_buffer', None)
        return provide(tuple(shape)) if provide is not None else None

    def _prepare(self, frame, dark_processor, calibration, inverted, snapshot):
        if frame.ndim != 2:
            raise ValueError('Preprocessing requires a 2D sensor frame')
        if calibration is not None and tuple(calibration.image_shape) != frame.shape:
            raise ValueError(f'Smile model {calibration.image_shape} does not match sensor frame {frame.shape}')
        if self._backend is None:
            self._select()  # Called by the acquisition worker, owning its CUDA context.
        enabled = bool(getattr(dark_processor, '_dark_enabled', False))
        # One snapshot per frame: the GUI thread may install a new reference at any
        # time, and GPU, CPU and validation must all see the same one.
        dark_processor = getattr(dark_processor, 'dark_reference', dark_processor)
        dark = getattr(dark_processor, '_dark_mean', None)
        if enabled and dark is not None and dark.shape != frame.shape:
            # Reference from another camera/bin: neither its dark nor its hot-pixel
            # coordinates (which would index outside this frame) may be applied.
            dark_processor, dark = _NO_REFERENCE, None
        return (frame, dark, dark_processor, enabled, bool(inverted), calibration, snapshot)

    @staticmethod
    def _key(args, *extra):
        frame, dark, hot, enabled, inverted, calibration, snapshot = args
        return (frame.shape, frame.dtype.str, id(dark), id(hot._hot_coords),
                enabled, inverted, id(calibration), bool(snapshot), *extra)

    @staticmethod
    def _check_gpu_input(frame, calibration):
        if frame.dtype != np.uint16:
            raise TypeError('CUDA preprocessing currently requires uint16 sensor frames')
        if calibration is not None and calibration.interp not in ('linear', 'cubic'):
            raise ValueError('CUDA preprocessing currently supports linear/cubic Smile')

    def _fall_back(self, exc):
        self.fallback_reason = str(exc)
        logger.warning('Preprocessing CUDA failure; switching to CPU: %s', exc)
        self._backend = self._cpu
        self._publish_backend()

    def process(self, frame, dark_processor, calibration=None, inverted=False, snapshot=False, profiler=None):
        """Correct one sensor frame.

        Args:
            frame: 2-D ``(slit, wavelength)`` sensor frame, uint16 for CUDA
                backends.  Not modified.
            dark_processor: Object exposing the dark/hot-pixel state (see the
                module docstring).  If its dark reference has a different shape
                than ``frame`` neither dark nor hot pixels are applied.
            calibration: Optional Smile calibration; its ``image_shape`` must
                equal ``frame.shape``.  ``None`` skips the remap.
            inverted: Flip the slit axis (rows) before the remap.
            snapshot: Also return the pre-Smile image.
            profiler: Optional object with ``time(name)`` returning a context
                manager; ``None`` disables timing.

        Returns:
            tuple: ``(output, grabbed)``; ``grabbed`` is ``None`` unless
            ``snapshot``.  Both are new arrays owned by the caller.

        Raises:
            ValueError: ``frame`` is not 2-D, or the calibration shape differs
                from the frame shape.

        A CUDA backend error or a CUDA/CPU mismatch (more than one count, checked
        once per configuration) does not raise: the runtime switches
        permanently to CPU, records ``fallback_reason`` and recomputes the frame.
        """
        if profiler is None:
            profiler = _NoProfiler()
        args = self._prepare(frame, dark_processor, calibration, inverted, snapshot)
        with profiler.time('preprocess'):
            if self._backend is not self._cpu:
                try:
                    self._check_gpu_input(frame, calibration)
                    output, grabbed = self._backend.process(*args, profiler)
                    key = self._key(args)
                    if key != self._validated:
                        reference, reference_grab = self._cpu.process(*args, _NoProfiler())
                        error = _max_error('output', output, reference)
                        _check_grab(snapshot, grabbed, reference_grab)
                        self._validated = key
                        logger.info('%s verified against CPU: maximum error=%d count', self.backend_name, error)
                    return output, grabbed
                except Exception as exc:
                    self._fall_back(exc)
            return self._cpu.process(*args, profiler)

    def process_binned(self, frame, dark_processor, binner, calibration=None, inverted=False,
                       snapshot=False, full=False, profiler=None):
        """Correct one sensor frame and lambda-bin it.

        Args:
            frame, dark_processor, calibration, inverted, snapshot, profiler:
                As for :meth:`process`.
            binner: Fitted ``LambdaBinner`` (``binner.v_ == frame.shape[1]``).
            full: Also return the corrected, unbinned image.

        Returns:
            tuple: ``(binned, output, grabbed)``: ``binned`` is the
            ``(slit, n_bins)`` uint16 result of ``binner.transform``; ``output``
            is ``None`` unless ``full``; ``grabbed`` is ``None`` unless
            ``snapshot``.  All are new arrays owned by the caller.

        Raises:
            ValueError: as for :meth:`process`, or the binner was fitted for
                another spectral width.

        Profiling: on the CPU the spans are ``preprocess`` (correction) and
        ``bin``; on a GPU binning runs inside ``preprocess``/``gpu_preprocess``
        and no ``bin`` span is recorded.  GPU errors and mismatches fall back
        to CPU exactly as in :meth:`process`; the first frame of each
        configuration is checked on ``binned`` (and ``output`` when ``full``).
        """
        if profiler is None:
            profiler = _NoProfiler()
        args = self._prepare(frame, dark_processor, calibration, inverted, snapshot)
        if binner.v_ != frame.shape[1]:
            raise ValueError(f'Binner fitted for {binner.v_} spectral pixels, frame has {frame.shape[1]}')
        if self._backend is not self._cpu:
            with profiler.time('preprocess'):
                try:
                    self._check_gpu_input(frame, calibration)
                    binned, output, grabbed = self._backend.process_binned(*args, binner, bool(full), profiler)
                    key = self._key(args, id(binner), bool(full))
                    if key != self._validated:
                        reference, reference_grab = self._cpu.process(*args, _NoProfiler())
                        error = _max_error('binned', binned, _bin(binner, reference))
                        if full:
                            error = max(error, _max_error('output', output, reference))
                        _check_grab(snapshot, grabbed, reference_grab)
                        self._validated = key
                        logger.info('%s verified against CPU: maximum error=%d count', self.backend_name, error)
                    return binned, output, grabbed
                except Exception as exc:
                    self._fall_back(exc)
        with profiler.time('preprocess'):
            output, grabbed = self._cpu.process(*args, profiler)
        with profiler.time('bin'):
            binned = _bin(binner, output)
        return binned, (output if full else None), grabbed

def _bin(binner, image):
    return binner.transform(image.reshape(1, *image.shape))[0]

def _max_error(name, actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise RuntimeError(f'CUDA {name} layout differs from CPU')
    error = int(np.max(np.abs(actual.astype(np.int32) - expected.astype(np.int32)))) if actual.size else 0
    if error > 1:
        raise RuntimeError(f'CUDA/CPU validation differs by {error} counts (allowed 1)')
    return error

def _check_grab(snapshot, grabbed, reference_grab):
    if snapshot and not np.array_equal(grabbed, reference_grab):
        raise RuntimeError('CUDA Grab differs from CPU pre-Smile image')

class _NoReference:
    """Stand-in processor with no dark or hot-pixel data."""
    _dark_enabled = False
    _dark_mean = None
    _hot_coords = None

_NO_REFERENCE = _NoReference()

class _NoProfiler:
    class Span:
        def __enter__(self): return self
        def __exit__(self, *args): return False
    def time(self, name): return self.Span()
    def _record(self, *args): pass

__all__ = ['PreprocessingRuntime']
__requires__ = ('compute_runtime',)
