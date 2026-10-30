# compute_runtime

## Overview

Process-wide, immutable snapshot of which CUDA backends (OpenCV CUDA, CuPy) work on this machine, plus a published "active processing backend" string. Detection runs once; missing libraries or devices simply mean CPU operation.

## Requirements

* Python standard library only for the package itself; OpenCV (`cv2`) and CuPy are imported lazily inside the probes and are optional.
* Other modules: none (`__requires__ = ()`). No PyQt5.

## Stand-alone use

There is no widget and no `__main__.py`. From a shell:

```
python -c "from modules.compute_runtime import get_compute_capabilities as g; print(g())"
```

## Embedding in an application

```python
from modules.compute_runtime import get_compute_capabilities, set_processing_backend

caps = get_compute_capabilities()          # probes once per process, then cached
if caps.cupy_cuda.available:
    device_id = caps.cupy_cuda.device_id   # always device 0
print(caps.cuda_available, caps.processing_backend)   # processing_backend is 'cpu' until published
set_processing_backend("cpu")              # normally called by modules.preprocessing
```

## Data formats / conventions

* `BackendCapability` (frozen dataclass): `available`, `device_count`, `device_id` (int or `None`), `device_name`, `reason`.
* `ComputeCapabilities` (frozen dataclass): `opencv_cuda`, `cupy_cuda` and the properties `cuda_available` (either available) and `processing_backend` (live value, one of `'cpu'`, `'opencv_cuda'`, `'cupy_cuda'`).
* Probes test device 0 only (OpenCV: uint16 `remap` round trip; CuPy: allocation, kernel, download) and never raise; failures are stored in `reason`.

## API summary

| Name | Purpose |
|------|---------|
| `initialize_compute()` | probe once (thread-safe) and return `ComputeCapabilities` |
| `get_compute_capabilities()` | cached snapshot, initialising lazily |
| `set_processing_backend(name)` | publish the active backend; `ValueError` for an unknown name |
| `BackendCapability`, `ComputeCapabilities` | result types |

## Threading

`initialize_compute` is guarded by a lock and safe to call from any thread; the result is immutable. `set_processing_backend` rebinds a module global and may be called from any thread. Probing creates CUDA contexts and can take noticeable time: call it at start-up, never in the frame loop.

## Tests

`tests/test_compute_runtime.py` (cached initialisation, immutability, missing dependencies, probe failures); contract checks in `tests/test_module_contract.py`. Run `python -m pytest -q tests/test_compute_runtime.py`.

## Design notes

Process-wide immutable capability snapshot initialized at HSI startup. MainWindow also initializes it for embedded launches; repeated calls return the same object. No probe runs in the frame loop. Optional dependency or device failures retain CPU operation.

```python
from modules.compute_runtime import get_compute_capabilities

caps = get_compute_capabilities()
if caps.cupy_cuda.available:
    device_id = caps.cupy_cuda.device_id
if caps.opencv_cuda.available:
    device_id = caps.opencv_cuda.device_id
```

cuda_available is true when either supported provider passes its smoke test. It does not promise that arbitrary CUDA operations or the full correction pipeline are supported. processing_backend reports the active acquisition preprocessing backend (cpu, opencv_cuda or cupy_cuda). Capability alone does not select kernels; PreprocessingRuntime publishes the backend after initialization and on CPU fallback.

OpenCV: check device count and CUDA remap availability, then upload a small uint16 image, perform identity remap, download and validate exact results. CuPy: check device count, allocate a small array, execute a kernel, synchronize, download and validate. Each probe tests device 0 and records its provider-specific identifier/name or failure reason. A failing device 0 is not automatically replaced with another device. Consumers must select the device in their own worker thread before creating CUDA resources.

Cache lifetime is the Python process. Restart after changing drivers/libraries/devices; do not hot-reload this module while GPU workers are active. No persistent device buffers are shared between threads.

Validation: five unit tests covering cached concurrent initialization, immutable state, absent dependencies, device/driver/remap failures, and kernel failure. CPU-only startup was checked here. Real CUDA hardware and Jetson Orin were unavailable.

References: https://docs.opencv.org/4.2.0/d8/d40/group__cudacore__init.html and https://docs.cupy.dev/en/stable/reference/cuda.html
