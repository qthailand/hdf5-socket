# preprocessing

## Overview

Per-frame sensor preprocessing: dark subtraction, hot-pixel repair, slit flip and Smile remap, optionally followed by lambda binning (`process_binned`), with a CPU reference backend and optional OpenCV-CUDA and CuPy-CUDA backends. Each GPU backend runs the whole path from upload to binning on the device. `PreprocessingRuntime` chooses a backend lazily (a GPU backend must pass a start-up self-test, otherwise the other GPU backend is tried before CPU), validates it against the CPU once per configuration and falls back to CPU permanently on any later GPU failure.

## Requirements

* NumPy and OpenCV (CPU path and Smile maps). CuPy and OpenCV-with-CUDA are optional (cuBLAS is not needed).
* Other modules: `compute_runtime` (`__requires__ = ("compute_runtime",)`). The dark processor and Smile calibration are duck-typed (`modules.dark.DarkProcessor`, `modules.smile` calibration), not imported.
* No PyQt5.

## Stand-alone use

There is no widget and no `__main__.py`; use it from Python (see below).

## Embedding in an application

```python
from modules.preprocessing import PreprocessingRuntime
from modules.dark import DarkProcessor

processor = DarkProcessor()
processor.set_dark_reference(dark_mean, hot_pixel_map)
runtime = PreprocessingRuntime("auto")      # 'auto' | 'cpu' | 'cupy_cuda' | 'opencv_cuda'
# in the acquisition worker thread, for every frame:
output, snapshot = runtime.process(raw_frame, processor, calibration=smile_model,
                                   inverted=True, snapshot=False)
# or, with lambda binning on the same device (binner = fitted LambdaBinner):
binned, full, snapshot = runtime.process_binned(raw_frame, processor, binner, calibration=smile_model,
                                                inverted=True, full=False)
print(runtime.backend_name, runtime.fallback_reason)
```

## Data formats / conventions

* Input frame: 2-D `(slit, wavelength)` array; CUDA backends need uint16, the CPU backend any dtype. A non-2-D frame or a calibration whose `image_shape` differs from the frame raises `ValueError`.
* Pipeline order: dark subtraction (saturating at 0) -> hot-pixel repair -> flip of the slit axis (`inverted`) -> Smile remap.
* `process` returns `(output, grabbed)`: `output` is the corrected `(slit, wavelength)` image; `grabbed` is the post-dark / post-hot / post-flip, pre-Smile image, or `None` unless `snapshot=True`. Both are new C-contiguous arrays owned by the caller.
* A dark reference whose shape differs from the frame disables dark and hot-pixel correction for that frame.
* `process_binned` returns `(binned, output, grabbed)`: `binned` is `binner.transform` of the corrected image, `(slit, n_bins)` uint16; `output` is the corrected image only with `full=True`. The binner must be fitted for `frame.shape[1]`.
* GPU results must match the CPU within one uint16 count (checked once per configuration, on `binned` and on `output` when requested).
* Duck-typed inputs: dark processor attributes `_dark_enabled`, `_dark_mean`, `_hot_coords`, `_apply_hot_pixel` (and for CuPy `_hot_ys`, `_hot_xs`, `_hot_valid`, `_neighbor_count`); calibration attributes `image_shape`, `interp`, `map_x`, `_get_map_y()`, `apply_native`.
* Optional `profiler` objects need a `time(name)` context manager; spans used: `dark`, `hot_pixel`, `orient_smile` (CPU), `gpu_preprocess` (GPU), `preprocess` (total), `bin` (CPU binning in `process_binned`; on a GPU binning is inside `preprocess` and no `bin` span is recorded). GPU backends also record `gpu_upload`, `gpu_dark`, `gpu_hot_pixel`, `gpu_smile`, `gpu_bin`, `gpu_download` (see `timing.py`) on frames where the profiler's optional `want_detail()` is true (one in ten with `StageProfiler`) and no Grab is requested: CUDA events for CuPy; for OpenCV CUDA host time with a stream wait at each boundary (sampled frames only). Their sum is a little below `gpu_preprocess`; the rest is Python launch overhead.

## API summary

| Name | Purpose |
|------|---------|
| `PreprocessingRuntime(backend='auto', capabilities=None)` | runtime; `process(frame, dark_processor, calibration=None, inverted=False, snapshot=False, profiler=None)`, `process_binned(frame, dark_processor, binner, calibration=None, inverted=False, snapshot=False, full=False, profiler=None)`, `input_buffer(shape)`, `backend_name`, `requested`, `fallback_reason` |
| `modules.preprocessing.timing` | `STAGES`, `stage_timer` and the CUDA-event / host-clock stage timers |
| `modules.preprocessing.selftest.self_test` | start-up check of a GPU backend against the CPU on synthetic data |
| `modules.preprocessing.cpu.CPUBackend` | reference backend |
| `modules.preprocessing.cupy_cuda.CuPyCUDABackend` | CuPy backend (optional) |
| `modules.preprocessing.opencv_cuda.OpenCVCUDABackend` | OpenCV CUDA backend (optional) |
| `modules.preprocessing.maps` | `prepare_maps`, `horizontal_table`, `FLAGS` |

## Threading

Create one runtime per acquisition thread and call `process` only from it: the backend is selected on the first call so that CUDA contexts belong to that thread. Instances are not thread-safe. The selected backend is published through `modules.compute_runtime.set_processing_backend`.

## Tests

`tests/test_preprocessing.py` (CPU equivalence, GPU fallback and validation, CuPy interpolation maths, shape checks; real-CUDA tests are skipped without a GPU), `tests/test_preprocessing_binned.py` (`process_binned`, selection order, and both GPU backends' logic on NumPy-emulated `cv2.cuda`/CuPy from `tests/cuda_emulation.py`; the emulation checks the algorithms, not the CUDA kernels or OpenCV bindings); contract checks in `tests/test_module_contract.py`. Run `python -m pytest -q tests/test_preprocessing.py`.

## Design notes

Standalone numerical package with CPU, OpenCV CUDA and CuPy CUDA backends. No Qt dependency. Requires NumPy and OpenCV for CPU/reference maps; GPU providers are optional.

```python
from modules.preprocessing import PreprocessingRuntime
from modules.dark import DarkProcessor

processor = DarkProcessor()
processor.set_dark_reference(dark_mean, hot_pixel_map)
runtime = PreprocessingRuntime("auto")
output, snapshot = runtime.process(raw_frame, processor, calibration=smile_model,
                                   inverted=True, snapshot=False)
```

`output` is an owned contiguous 2D image in original sensor dtype. `snapshot` is None unless requested, then contains the owned post-Dark/post-hot/post-orientation, pre-Smile image. Keep the runtime in its acquisition thread. CUDA device/context and persistent buffers are created there, not shared with the GUI. A custom camera read_frame override is retained by HSI and its corrections are not applied twice.

## Selection

Add `"preprocessing_backend": "auto"` to config/processing.json if desired. Existing split settings without this key default to auto. Accepted values: auto, cpu, cupy_cuda, opencv_cuda. This overlay does not replace existing settings/calibration files.

Order tried: `auto` and `cupy_cuda`: CuPy, OpenCV CUDA, CPU; `opencv_cuda`: OpenCV CUDA, CuPy, CPU. A GPU provider is used only if its startup probe succeeded and it passes the self-test of the whole path (binning included); otherwise the next one is tried. This is availability selection, not a measured speed ranking. CUDA currently processes uint16 frames with linear/cubic Smile; unsupported types/interpolation fall back to CPU for the rest of this runtime. Restart or recreate the runtime to retry. CPU retains other existing Smile modes, including Lanczos. Orientation retains the current vertical flip, not a 180-degree rotation.

## Implementations

CPU: reuse a dark-subtraction buffer, repair sparse hot pixels with the existing kernel, fuse orientation into the full-frame Smile remap. Without Smile, produce an owned oriented output. All returned images remain stable when the next frame is processed.

CuPy: upload once, then four RawKernels: saturating dark subtraction (`dark16`), hot-pixel repair with one thread per hot pixel over cached flat neighbour indices and the CPU's arithmetic (exact float32 sum, float64 division, round half to even; `hot16`), horizontal Smile plus orientation (`remap16`) and lambda binning with the Numba binner's integer arithmetic (`bin16`); download only the binned image (plus the full image or Grab when requested). Dark, hot-pixel tables, maps, bin tables and device buffers are cached; a frame allocates nothing on the device. Zero-copy input: `input_buffer(shape)` returns mapped pinned host memory (`cudaHostAllocMapped`; Jetson shares one DRAM, but the GPU cannot read pageable numpy memory). A frame written there (the acquisition engine has GenICam convert straight into it) is read by the kernels in place, with no upload; it is never modified (hot-pixel repair without a dark first copies it to a device buffer). The buffer is reused per shape and overwritten by the next frame. At selection the zero-copy path is self-tested separately; if it fails, only zero-copy is disabled and frames are uploaded as before. Measured on the Orin (tools/bench_upload.py, GenICam bin 2): host-to-GPU path 1.42 ms -> 0.47 ms per frame. A requested Grab requires an additional download. The remap uses OpenCV-quantized coordinates and coefficient tables; the model must have integer row coordinates, as current Smile models do.

OpenCV CUDA: GPU upload, saturating dark subtraction, hot-pixel repair as a masked 3x3 neighbour sum divided by the cached valid-neighbour count, Smile as one integer nearest-neighbour gather per tap weighted with OpenCV's 1/32 tables (the same tables and float32 operation order as CPU/CuPy; `cv2.cuda.remap`'s own interpolation is not used because it takes unquantized coordinates and a different cubic kernel), then binning as one nearest-neighbour gather per column offset of the widest bin, summed in float32, with float64 rounding (no `cv2.cuda.gemm`: OpenCV creates a cuBLAS handle on every call). Float32 sums of uint16 values stay exact below 2**24, so bins are limited to 256 pixels. Every operation of a frame is queued on one `cv2.cuda.Stream` with preallocated outputs (no per-frame device allocation, no implicit synchronisation); the host waits once, after the downloads. Only the binned image is downloaded (plus the full image or Grab when requested). Unsupported operations or numerical differences trigger CPU fallback.

Raw dark capture bypasses this runtime completely. Binning results are unchanged (same integer arithmetic as the Numba LambdaBinner); recording, preview throttling, and cube export are unchanged.

## Validation and timing

For each new GPU configuration, process the first frame with the CPU reference as well. Require identical shape/dtype, maximum error at most one uint16 count, and exact Grab data when requested. Failure/exception latches CPU fallback and records the reason. This adds first-frame work after configuration changes; it is not run on every frame. GPU interpolation is not guaranteed bit-identical to CPU. Backend unit tests additionally check original CPU ordering, exact CPU numerical equivalence, snapshots, buffer ownership and input immutability.

`fetch` now measures raw camera acquisition for the shared camera pipeline. `preprocess` measures total preprocessing wall time. CPU detail fields are `dark`, `hot_pixel`, `orient_smile`; GPU uses `gpu_preprocess`, including upload/download and synchronization. These detail fields are contained within `preprocess`; do not sum them with the total. GPU initialization may fall outside the total; kernel JIT/cache uploads/CPU validation affect early frames. Existing stage averages are not a single simultaneous frame trace.

Validation here: 44 targeted tests passed, 2 real CUDA tests skipped because no GPU was available. CUDA kernels and OpenCV operations must still be tested on the target Orin/PC. No acquisition FPS or sub-millisecond latency claim is made. Check normal scan, raw dark capture, Grab, bin/model changes, and backend logs on the target device before using the new path for measurements.
