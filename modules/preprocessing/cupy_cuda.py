"""Resident CuPy uint16 pipeline with cached maps, sparse hot-pixel repair and lambda binning."""
import numpy as np
from .maps import prepare_maps, horizontal_table
from .timing import CudaEventTimer, stage_timer

try:
    import cupy as cp
except Exception as exc:      # optional dependency: CPU-only machines have no (working) CuPy
    cp = None
    _CUPY_ERROR = exc
else:
    _CUPY_ERROR = None

_REMAP = r'''
extern "C" __global__ void remap16(const unsigned short* src, unsigned short* dst,
 const short* xy, const unsigned short* frac, const float* weights,
 int n, int width, int height, int taps, int inverted, int identity) {
 int i=blockDim.x*blockIdx.x+threadIdx.x; if(i>=n)return;
 int row=i/width, col=i%width;
 if(identity){dst[i]=src[(inverted?height-1-row:row)*width+col];return;}
 int x=xy[2*i], y=xy[2*i+1];
 if(taps==1){dst[i]=(x>=0&&x<width&&y>=0&&y<height)?src[y*width+x]:0;return;}
 int f=frac[i]&31; float value=0;
 for(int k=0;k<taps;k++){
   int sx=x+k-(taps==4?1:0);
   float v=(sx>=0&&sx<width&&y>=0&&y<height)?(float)src[y*width+sx]:0;
   value=__fadd_rn(value,__fmul_rn(v,weights[f*taps+k]));
 }
 value=fminf(65535.0f,fmaxf(0.0f,value));dst[i]=(unsigned short)__float2uint_rn(value);
}
'''

# Same integer arithmetic as LambdaBinner's Numba kernel: (sum + w/2) / w, clamped.
# A uint32 sum cannot overflow: at most 65535 * 65536 for a 65536-pixel row.
_BIN = r'''
extern "C" __global__ void bin16(const unsigned short* src, unsigned short* dst,
 const int* left, const int* right, const int* count, int rows, int width, int bins) {
 int i=blockDim.x*blockIdx.x+threadIdx.x; if(i>=rows*bins)return;
 int row=i/bins, k=i%bins; const unsigned short* line=src+(long long)row*width;
 unsigned int sum=0; for(int j=left[k];j<right[k];j++) sum+=line[j];
 unsigned int w=(unsigned int)count[k], value=(sum+w/2u)/w;
 dst[i]=(unsigned short)(value>65535u?65535u:value);
}
'''

# Saturating uint16 dark subtraction in one pass (cv2.subtract semantics).
_DARK = r'''
extern "C" __global__ void dark16(const unsigned short* src, const unsigned short* dark,
 unsigned short* dst, int n) {
 int i=blockDim.x*blockIdx.x+threadIdx.x; if(i>=n)return;
 unsigned short s=src[i], d=dark[i]; dst[i]=s>d?(unsigned short)(s-d):(unsigned short)0;
}
'''

# One thread per hot pixel: mean of its valid neighbours (flat indices, -1 = not valid),
# with the CPU's arithmetic - exact float32 sum, float64 division, round half to even.
# Threads read only non-hot pixels and write only hot ones, so there is no data race.
_HOT = r'''
extern "C" __global__ void hot16(unsigned short* img, const int* target, const int* neighbour,
 const int* count, int n) {
 int i=blockDim.x*blockIdx.x+threadIdx.x; if(i>=n)return;
 float sum=0.0f;
 for(int k=0;k<8;k++){ int j=neighbour[8*i+k]; if(j>=0) sum=__fadd_rn(sum,(float)img[j]); }
 img[target[i]]=(unsigned short)rint((double)sum/(double)count[i]);
}
'''

class CuPyCUDABackend:
    """Resident CuPy uint16 preprocessing backend (CUDA).

    Device buffers, Smile tables and bin tables are cached between frames and
    rebuilt when the frame shape, dark, hot-pixel set, calibration or binner
    object changes.  Use from the thread that constructed it (it owns the CUDA
    device context).
    """
    name = 'cupy_cuda'
    def __init__(self, device_id):
        """Select the CUDA device and compile the remap and binning kernels.

        Args:
            device_id: CUDA device ordinal.

        Raises:
            ImportError: CuPy is not installed (other CUDA errors also propagate).
        """
        if cp is None:
            raise ImportError(f'CuPy is not available: {_CUPY_ERROR}')
        self.cp = cp
        self.device_id = device_id
        cp.cuda.Device(device_id).use()
        self._kernel = cp.RawKernel(_REMAP, 'remap16')
        self._bin_kernel = cp.RawKernel(_BIN, 'bin16')
        self._dark_kernel = cp.RawKernel(_DARK, 'dark16')
        self._hot_kernel = cp.RawKernel(_HOT, 'hot16')
        self._shape = None
        self._dark_ref = None
        self._hot_ref = None
        self._hot_width = None
        self._map_ref = None
        self._binner_ref = None
        self._binned = None
        self._events = None
        self._zero_copy = True
        self._mapped_shape = None
        self._mapped_host = None

    def input_buffer(self, shape):
        """Host array the kernels read in place (mapped pinned memory: no upload).

        Jetson CPU and GPU share one DRAM, but the GPU cannot read ordinary
        (pageable) numpy memory; memory from ``cudaHostAlloc(..., Mapped)`` it can.
        The buffer is reused for every frame of ``shape`` and freed by reference
        counting when a new shape replaces it.

        Args:
            shape: ``(height, width)``.

        Returns:
            numpy.ndarray | None: C-contiguous ``uint16`` array of ``shape``, or
            ``None`` once :meth:`disable_zero_copy` was called.
        """
        if not self._zero_copy:
            return None
        if self._mapped_shape != shape:
            cp = self.cp
            count = int(np.prod(shape))
            memory = cp.cuda.PinnedMemory(count * 2, cp.cuda.runtime.hostAllocMapped)
            host = np.frombuffer(cp.cuda.PinnedMemoryPointer(memory, 0), dtype=np.uint16, count=count)
            # Unified addressing: the host pointer is the device pointer; ``memory`` stays alive.
            device_memory = cp.cuda.UnownedMemory(host.__array_interface__['data'][0], count * 2, memory)
            self._mapped_device = cp.ndarray(shape, cp.uint16, cp.cuda.MemoryPointer(device_memory, 0))
            self._mapped_host = host.reshape(shape)
            self._mapped_shape = shape
        return self._mapped_host

    def disable_zero_copy(self):
        """Stop offering :meth:`input_buffer`; frames are uploaded as before."""
        self._zero_copy = False
        self._mapped_shape = self._mapped_host = self._mapped_device = None

    def _is_mapped(self, frame):
        return (self._mapped_host is not None and frame.shape == self._mapped_shape and frame.dtype == np.uint16
                and frame.__array_interface__['data'][0] == self._mapped_host.__array_interface__['data'][0]
                and frame.flags.c_contiguous)

    def _timer(self, profiler, snapshot):
        return stage_timer(profiler, snapshot, self._event_timer)

    def _event_timer(self):
        if self._events is None:          # events are reused between sampled frames
            self._events = CudaEventTimer(self.cp)
        return self._events

    def process(self, frame, dark, hot, enabled, inverted, calibration, snapshot, profiler):
        """GPU equivalent of ``CPUBackend.process`` for uint16 frames.

        Args:
            frame: 2-D uint16 ``(slit, wavelength)`` frame; not modified.
            dark: Dark mean (same shape) or ``None``.
            hot: Object exposing the cached hot-pixel arrays (``_hot_coords``,
                ``_hot_ys``, ``_hot_xs``, ``_hot_valid``, ``_neighbor_count``).
            enabled: Whether dark/hot correction is applied.
            inverted: Flip the slit axis.
            calibration: Smile calibration or ``None``; only horizontal
                (spectral-axis) shifts with linear/cubic/nearest tables.
            snapshot: Also return the pre-Smile image.
            profiler: Object with ``time(name)`` context manager.

        Returns:
            tuple: ``(output, grabbed)`` host arrays owned by the caller
            (``grabbed`` is ``None`` unless ``snapshot``).

        Raises:
            ValueError: the Smile map has vertical (row) interpolation.
        """
        cp = self.cp
        with cp.cuda.Device(self.device_id), profiler.time('gpu_preprocess'):
            timer = self._timer(profiler, snapshot)
            grabbed = self._correct(frame, dark, hot, enabled, inverted, calibration, snapshot, timer)
            timer.mark(5)
            output = self._output.get()
            timer.mark(6)
            timer.report(profiler)
            return output, grabbed

    def process_binned(self, frame, dark, hot, enabled, inverted, calibration, snapshot, binner, full, profiler):
        """``process`` followed by lambda binning on the device.

        Args:
            frame, dark, hot, enabled, inverted, calibration, snapshot, profiler:
                As for :meth:`process`.
            binner: Fitted ``LambdaBinner`` (``iL``, ``iR``, ``w``, ``n_bins``).
            full: Also download the corrected, unbinned image.

        Returns:
            tuple: ``(binned, output, grabbed)`` host arrays owned by the caller;
            ``binned`` is ``(slit, n_bins)`` uint16, ``output`` is ``None``
            unless ``full``, ``grabbed`` is ``None`` unless ``snapshot``.
        """
        cp = self.cp
        with cp.cuda.Device(self.device_id), profiler.time('gpu_preprocess'):
            timer = self._timer(profiler, snapshot)
            grabbed = self._correct(frame, dark, hot, enabled, inverted, calibration, snapshot, timer)
            h, w = frame.shape
            if binner is not self._binner_ref or self._binned.shape[0] != h:
                self._left = cp.asarray(np.asarray(binner.iL, dtype=np.int32))
                self._right = cp.asarray(np.asarray(binner.iR, dtype=np.int32))
                self._count = cp.asarray(np.asarray(binner.w, dtype=np.int32))
                self._binned = cp.empty((h, int(binner.n_bins)), dtype=cp.uint16)
                self._binner_ref = binner
            bins = self._binned.shape[1]
            self._bin_kernel(((h * bins + 255)//256,), (256,),
                             (self._output, self._binned, self._left, self._right, self._count,
                              np.int32(h), np.int32(w), np.int32(bins)))
            timer.mark(5)
            binned = self._binned.get()
            output = self._output.get() if full else None
            timer.mark(6)
            timer.report(profiler)
            return binned, output, grabbed

    def _correct(self, frame, dark, hot, enabled, inverted, calibration, snapshot, timer):
        """Run dark/hot/flip/Smile into ``self._output``; return the host Grab or ``None``.

        ``timer`` marks the stage boundaries 0-4 (upload, dark, hot pixel, Smile).
        """
        cp = self.cp
        if self._shape != frame.shape:
            self._shape = frame.shape
            self._input = cp.empty(frame.shape, dtype=cp.uint16)
            self._work = cp.empty(frame.shape, dtype=cp.uint16)
            self._output = cp.empty(frame.shape, dtype=cp.uint16)
        timer.mark(0)
        mapped = self._is_mapped(frame)
        if mapped:
            source = self._mapped_device          # zero-copy: the kernels read the host buffer
        else:
            self._input.set(np.ascontiguousarray(frame))
            source = self._input
        timer.mark(1)
        corrected = source
        if enabled and dark is not None and dark.shape == frame.shape:
            if dark is not self._dark_ref:
                self._dark = cp.asarray(dark)
                self._dark_ref = dark
            self._dark_kernel(((frame.size + 255)//256,), (256,),
                              (source, self._dark, self._work, np.int32(frame.size)))
            corrected = self._work
        timer.mark(2)
        if enabled and hot._hot_coords is not None:
            if mapped and corrected is source:
                # Hot-pixel repair writes in place: never into the caller's frame.
                self._work[...] = corrected
                corrected = self._work
            width = frame.shape[1]
            if hot._hot_coords is not self._hot_ref or width != self._hot_width:
                coords = hot._hot_coords
                self._hot_target = cp.asarray((coords[:, 0] * width + coords[:, 1]).astype(np.int32))
                self._hot_neighbour = cp.asarray(
                    np.where(hot._hot_valid, hot._hot_ys * width + hot._hot_xs, -1).astype(np.int32))
                self._hot_count = cp.asarray(hot._neighbor_count.ravel().astype(np.int32))
                self._hot_ref = hot._hot_coords
                self._hot_width = width
            n = int(self._hot_target.size)
            self._hot_kernel(((n + 255)//256,), (256,),
                             (corrected, self._hot_target, self._hot_neighbour, self._hot_count, np.int32(n)))
        timer.mark(3)
        grabbed = None
        if snapshot:
            host = corrected.get()
            grabbed = np.array(host[::-1] if inverted else host, copy=True, order='C')
        identity = calibration is None
        key = (id(calibration), inverted, frame.shape)
        if key != self._map_ref:
            if identity:
                self._xy = cp.zeros((1, 2), dtype=cp.int16)
                self._frac = cp.zeros(1, dtype=cp.uint16)
                self._weights = cp.ones(1, dtype=cp.float32)
                self._taps = 1
            else:
                _, _, xy, frac, _ = prepare_maps(calibration, inverted)
                # Smile moves spectral columns only; no vertical interpolation.
                if frac is not None and np.any(frac >> 5):
                    raise ValueError('CuPy horizontal Smile requires integer row coordinates')
                self._xy = cp.asarray(xy)
                self._frac = cp.asarray(frac) if frac is not None else cp.zeros(1, dtype=cp.uint16)
                table = horizontal_table(calibration.interp)
                self._weights = cp.asarray(table)
                self._taps = table.shape[1]
            self._map_ref = key
            self._calibration_ref = calibration
        h, w = frame.shape
        self._kernel(((frame.size + 255)//256,), (256,),
                     (corrected, self._output, self._xy, self._frac, self._weights,
                      np.int32(frame.size), np.int32(w), np.int32(h), np.int32(self._taps),
                      np.int32(inverted), np.int32(identity)))
        timer.mark(4)
        return grabbed
