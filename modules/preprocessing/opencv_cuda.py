"""OpenCV CUDA uint16 pipeline: dark, hot pixels, flip, Smile and lambda binning on the GPU.

Every operation of a frame is queued on one ``cv2.cuda.Stream`` with a
preallocated output (no per-frame device allocation or implicit device
synchronisation); the host waits once, after the downloads.

Hot-pixel repair is a masked 3x3 neighbour sum divided by the cached valid
neighbour count.  Smile gathers each interpolation tap with an integer
nearest-neighbour remap and weights it with OpenCV's own 1/32 tables (the
tables the CPU and CuPy use), because ``cv2.cuda.remap`` interpolates with
float coordinates and its own cubic kernel.  Binning gathers the j-th column
of every bin with one nearest-neighbour remap per column offset and sums them
(no cuBLAS GEMM, whose handle OpenCV creates on every call), then rounds in
float64.  All sums of uint16 values stay below 2**24, so float32 holds them
exactly.
"""
import cv2
import numpy as np
from .maps import prepare_maps, horizontal_table
from .timing import HostClockTimer, stage_timer

# float32 sums of up to this many uint16 pixels are exact (256 * 65535 < 2**24).
MAX_BIN_WIDTH = 256

def _upload(array):
    mat = cv2.cuda_GpuMat()
    mat.upload(np.ascontiguousarray(array))
    return mat

class OpenCVCUDABackend:
    """OpenCV CUDA preprocessing and binning backend (uint16), resident on the GPU.

    Use from the thread that constructed it.
    """
    name = 'opencv_cuda'
    def __init__(self, device_id):
        """Select the CUDA device and create the stream and reusable GPU objects.

        Args:
            device_id: CUDA device ordinal.

        Raises:
            cv2.error: OpenCV was built without CUDA or the device is invalid.
        """
        cv2.cuda.setDevice(device_id)
        self._stream = cv2.cuda_Stream()
        self._buffers = {}
        self._dark = cv2.cuda_GpuMat()
        self._neighbour_sum = cv2.cuda.createLinearFilter(
            cv2.CV_32FC1, cv2.CV_32FC1, np.ones((3, 3), np.float32), borderMode=cv2.BORDER_CONSTANT)
        self._dark_ref = None
        self._hot_ref = None
        self._hot_shape = None
        self._map_ref = None
        self._binner_ref = None
        self._bin_rows = None

    def _buffer(self, name, rows, cols, mat_type):
        """Reusable device matrix ``name``; reallocated only when its size or type changes."""
        key = (rows, cols, mat_type)
        entry = self._buffers.get(name)
        if entry is None or entry[0] != key:
            entry = (key, cv2.cuda_GpuMat(rows, cols, mat_type))
            self._buffers[name] = entry
        return entry[1]

    def _timer(self, profiler, snapshot):
        # Stream work is asynchronous: a sampled frame waits at each stage boundary.
        return stage_timer(profiler, snapshot, lambda: HostClockTimer(self._stream.waitForCompletion))

    def process(self, frame, dark, hot, enabled, inverted, calibration, snapshot, profiler):
        """GPU equivalent of ``CPUBackend.process`` for uint16 frames.

        Args:
            frame: 2-D uint16 ``(slit, wavelength)`` frame; not modified.
            dark: Dark mean (same shape) or ``None``.
            hot: Object with ``_hot_coords`` and ``_neighbor_count``.
            enabled: Whether dark/hot correction is applied.
            inverted: Flip the slit axis.
            calibration: Smile calibration or ``None``; horizontal (spectral
                axis) linear/cubic only.
            snapshot: Also return the pre-Smile image.
            profiler: Object with ``time(name)`` context manager.

        Returns:
            tuple: ``(output, grabbed)`` host arrays owned by the caller
            (``grabbed`` is ``None`` unless ``snapshot``).

        Raises:
            ValueError: the Smile map has vertical (row) interpolation.
        """
        with profiler.time('gpu_preprocess'):
            timer = self._timer(profiler, snapshot)
            output, grabbed = self._correct(frame, dark, hot, enabled, inverted, calibration, snapshot, timer)
            timer.mark(5)
            host = output.download(self._stream)
            self._stream.waitForCompletion()
            timer.mark(6)
            timer.report(profiler)
            return host, grabbed

    def process_binned(self, frame, dark, hot, enabled, inverted, calibration, snapshot, binner, full, profiler):
        """``process`` followed by lambda binning on the device.

        Args:
            frame, dark, hot, enabled, inverted, calibration, snapshot, profiler:
                As for :meth:`process`.
            binner: Fitted ``LambdaBinner`` (``v_``, ``iL``, ``iR``, ``w``).
            full: Also download the corrected, unbinned image.

        Returns:
            tuple: ``(binned, output, grabbed)`` host arrays owned by the caller;
            ``binned`` is ``(slit, n_bins)`` uint16, ``output`` is ``None``
            unless ``full``, ``grabbed`` is ``None`` unless ``snapshot``.

        Raises:
            ValueError: a bin is wider than ``MAX_BIN_WIDTH`` pixels.
        """
        with profiler.time('gpu_preprocess'):
            timer = self._timer(profiler, snapshot)
            output, grabbed = self._correct(frame, dark, hot, enabled, inverted, calibration, snapshot, timer)
            binned = self._bin(output, binner, frame.shape)
            timer.mark(5)
            binned = binned.download(self._stream)
            full_output = output.download(self._stream) if full else None
            self._stream.waitForCompletion()
            timer.mark(6)
            timer.report(profiler)
            return binned, full_output, grabbed

    def _correct(self, frame, dark, hot, enabled, inverted, calibration, snapshot, timer):
        """Queue dark/hot/flip/Smile; return the corrected uint16 GpuMat and the host Grab (or ``None``).

        ``timer`` marks the stage boundaries 0-4 (upload, dark, hot pixel, Smile).
        """
        stream = self._stream
        h, w = frame.shape
        timer.mark(0)
        corrected = self._buffer('input', h, w, cv2.CV_16UC1)
        corrected.upload(np.ascontiguousarray(frame), stream)
        timer.mark(1)
        if enabled and dark is not None and dark.shape == frame.shape:
            if dark is not self._dark_ref:
                self._dark.upload(np.ascontiguousarray(dark))
                self._dark_ref = dark
            corrected = cv2.cuda.subtract(corrected, self._dark, dst=self._buffer('dark', h, w, cv2.CV_16UC1),
                                          stream=stream)
        timer.mark(2)
        if enabled and hot._hot_coords is not None:
            corrected = self._repair_hot_pixels(corrected, hot, frame.shape)
        timer.mark(3)
        grabbed = None
        if snapshot:
            host = corrected.download(stream)
            stream.waitForCompletion()
            grabbed = np.array(host[::-1] if inverted else host, copy=True, order='C')
        if calibration is None:
            output = (cv2.cuda.flip(corrected, 0, dst=self._buffer('flip', h, w, cv2.CV_16UC1), stream=stream)
                      if inverted else corrected)
        else:
            output = self._smile(corrected, calibration, inverted, frame.shape)
        timer.mark(4)
        return output, grabbed

    def _repair_hot_pixels(self, corrected, hot, shape):
        """Replace hot pixels by the rounded mean of their valid (in-image, non-hot) neighbours."""
        h, w = shape
        if hot._hot_coords is not self._hot_ref or self._hot_shape != shape:
            ys, xs = hot._hot_coords[:, 0], hot._hot_coords[:, 1]
            mask = np.zeros(shape, np.float32)
            mask[ys, xs] = 1
            count = np.ones(shape, np.float32)
            count[ys, xs] = hot._neighbor_count.ravel()
            self._hot_mask = _upload(mask)
            self._cold_mask = _upload(1 - mask)
            self._neighbour_count = _upload(count)
            self._hot_ref = hot._hot_coords
            self._hot_shape = shape
        stream = self._stream
        values = corrected.convertTo(cv2.CV_32F, stream, self._buffer('hot_values', h, w, cv2.CV_32FC1))
        cold = cv2.cuda.multiply(values, self._cold_mask, dst=self._buffer('hot_cold', h, w, cv2.CV_32FC1),
                                 stream=stream)
        sums = self._neighbour_sum.apply(cold, dst=self._buffer('hot_sums', h, w, cv2.CV_32FC1), stream=stream)
        average = cv2.cuda.divide(sums, self._neighbour_count, dst=values, stream=stream)
        # uint16 conversion rounds half to even, like np.round on the CPU.
        rounded = average.convertTo(cv2.CV_16U, stream, self._buffer('hot_rounded', h, w, cv2.CV_16UC1))
        rounded.convertTo(cv2.CV_32F, stream, values)
        cv2.cuda.multiply(values, self._hot_mask, dst=values, stream=stream)
        cv2.cuda.add(cold, values, dst=cold, stream=stream)
        return cold.convertTo(cv2.CV_16U, stream, self._buffer('hot_out', h, w, cv2.CV_16UC1))

    def _smile(self, corrected, calibration, inverted, shape):
        """Horizontal Smile remap (with the slit flip folded into the row map)."""
        h, w = shape
        key = (id(calibration), inverted, shape)
        if key != self._map_ref:
            _, _, xy, frac, _ = prepare_maps(calibration, inverted)
            if frac is None or np.any(frac >> 5):
                raise ValueError('OpenCV CUDA horizontal Smile requires integer row coordinates')
            table = horizontal_table(calibration.interp)
            taps = table.shape[1]
            x = xy[..., 0].astype(np.float32)
            fraction = frac & 31
            self._row_map = _upload(xy[..., 1].astype(np.float32))
            self._tap_maps = [(_upload(x + (k - (1 if taps == 4 else 0))), _upload(table[fraction, k]))
                              for k in range(taps)]
            self._map_ref = key
            self._calibration_ref = calibration
        stream = self._stream
        source = corrected.convertTo(cv2.CV_32F, stream, self._buffer('smile_source', h, w, cv2.CV_32FC1))
        value = self._buffer('smile_value', h, w, cv2.CV_32FC1)
        sample = self._buffer('smile_sample', h, w, cv2.CV_32FC1)
        # Same order of float32 operations as the CuPy kernel: ((v0*w0) + v1*w1) + ...
        for k, (column_map, weight) in enumerate(self._tap_maps):
            cv2.cuda.remap(source, column_map, self._row_map, cv2.INTER_NEAREST, dst=sample,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0, stream=stream)
            if k == 0:
                cv2.cuda.multiply(sample, weight, dst=value, stream=stream)
            else:
                cv2.cuda.multiply(sample, weight, dst=sample, stream=stream)
                cv2.cuda.add(value, sample, dst=value, stream=stream)
        return value.convertTo(cv2.CV_16U, stream, self._buffer('smile_out', h, w, cv2.CV_16UC1))

    def _bin(self, image, binner, shape):
        """Lambda binning: ``(sum + w//2) // w`` per bin, clamped to 65535."""
        h, v = shape
        if binner is not self._binner_ref or self._bin_rows != h:
            widths = np.asarray(binner.w, dtype=np.int64)
            if widths.size and int(widths.max()) > MAX_BIN_WIDTH:
                raise ValueError(f'OpenCV CUDA binning supports bins up to {MAX_BIN_WIDTH} pixels')
            left = np.asarray(binner.iL, dtype=np.int64)
            bins = int(widths.size)
            self._bin_row_map = _upload(np.repeat(np.arange(h, dtype=np.float32)[:, None], bins, axis=1))
            # Column j of each bin, or -1 (border value 0) once the bin is narrower than j + 1.
            self._bin_column_maps = [
                _upload(np.tile(np.where(j < widths, left + j, -1).astype(np.float32), (h, 1)))
                for j in range(int(widths.max()) if widths.size else 0)]
            # floor((s + w//2) / w) == round((s + offset) / w) with offset 0.5 for even w:
            # the quotient's fraction stays within 1/(2w) of the integer, never on a tie.
            offset = np.where(widths % 2 == 0, 0.5, 0.0)
            self._bin_offset = _upload(np.tile(offset, (h, 1)))
            self._bin_width = _upload(np.tile(widths.astype(np.float64), (h, 1)))
            self._binner_ref = binner
            self._bin_rows = h
            self._bin_count = bins
        stream = self._stream
        bins = self._bin_count
        source = image.convertTo(cv2.CV_32F, stream, self._buffer('bin_source', h, v, cv2.CV_32FC1))
        total = self._buffer('bin_total', h, bins, cv2.CV_32FC1)
        sample = self._buffer('bin_sample', h, bins, cv2.CV_32FC1)
        for j, column_map in enumerate(self._bin_column_maps):
            cv2.cuda.remap(source, column_map, self._bin_row_map, cv2.INTER_NEAREST,
                           dst=total if j == 0 else sample,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0, stream=stream)
            if j:
                cv2.cuda.add(total, sample, dst=total, stream=stream)
        totals = total.convertTo(cv2.CV_64F, stream, self._buffer('bin_totals', h, bins, cv2.CV_64FC1))
        cv2.cuda.add(totals, self._bin_offset, dst=totals, stream=stream)
        cv2.cuda.divide(totals, self._bin_width, dst=totals, stream=stream)
        return totals.convertTo(cv2.CV_16U, stream, self._buffer('bin_out', h, bins, cv2.CV_16UC1))
