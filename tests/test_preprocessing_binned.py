"""process_binned, backend selection order and the GPU backends' logic on emulated CUDA."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cuda_emulation import fake_cupy, fake_cv2, GpuMat
from test_preprocessing import Calibration, fixture, reference, CPU_CAPS
from modules.preprocessing import PreprocessingRuntime, cupy_cuda, opencv_cuda
from modules.preprocessing.cpu import CPUBackend
from modules.preprocessing.selftest import self_test
from modules.compute_runtime import BackendCapability, ComputeCapabilities
from modules.spectral_processing.binning import LambdaBinner

GPU_CAPS = ComputeCapabilities(BackendCapability(True, 1, 0, 'emulated', 'test'),
                               BackendCapability(True, 1, 0, 'emulated', 'test'))


class _Silent:
    class Span:
        def __enter__(self): return self
        def __exit__(self, *args): return False
    def time(self, name): return self.Span()


@pytest.fixture
def emulated(monkeypatch):
    monkeypatch.setattr(opencv_cuda, 'cv2', fake_cv2())
    monkeypatch.setattr(cupy_cuda, 'cp', fake_cupy())


def binner_for(width, m=0.29, step=2.0):
    return LambdaBinner(m=m, c=400.0, step=step).fit(width)


def expected_binned(frame, processor, model, inverted, binner):
    output, grab = reference(frame, processor, model, inverted)
    return binner.transform(output[None])[0], output, grab


@pytest.mark.parametrize('interp', [None, 'linear', 'cubic'])
@pytest.mark.parametrize('inverted', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('backend', ['cupy_cuda', 'opencv_cuda'])
def test_emulated_gpu_backend_matches_cpu_exactly(emulated, backend, interp, inverted, enabled):
    frame, processor = fixture(); processor.set_dark_enabled(enabled)
    frame[5, 20:30] = 65535
    model = Calibration(frame.shape, interp) if interp else None
    binner = binner_for(frame.shape[1], m=0.71)          # widths 2 and 3
    gpu = (cupy_cuda.CuPyCUDABackend if backend == 'cupy_cuda' else opencv_cuda.OpenCVCUDABackend)(0)
    args = (frame, processor._dark_mean, processor, enabled, inverted, model, True)
    for _ in range(2):                                   # second pass uses the cached tables
        binned, output, grabbed = gpu.process_binned(*args, binner, True, _Silent())
        want_binned, want_output, want_grab = expected_binned(frame, processor, model, inverted, binner)
        np.testing.assert_array_equal(output, want_output)
        np.testing.assert_array_equal(binned, want_binned)
        np.testing.assert_array_equal(grabbed, want_grab)
    plain, grab = gpu.process(*args, _Silent())
    np.testing.assert_array_equal(plain, want_output)
    _, output, grabbed = gpu.process_binned(*args[:-1], False, binner, False, _Silent())
    assert output is None and grabbed is None


@pytest.mark.parametrize('backend', ['cupy_cuda', 'opencv_cuda'])
def test_emulated_backends_pass_start_up_self_test(emulated, backend):
    gpu = (cupy_cuda.CuPyCUDABackend if backend == 'cupy_cuda' else opencv_cuda.OpenCVCUDABackend)(0)
    self_test(gpu, CPUBackend(), _Silent())


@pytest.mark.parametrize('width', [1, 2, 3, 4, 7, 8, 31, 32, 255, 256])
def test_opencv_binning_rounding_matches_numba(emulated, width):
    rng = np.random.default_rng(width)
    v = width * 5
    image = rng.integers(0, 65536, (64, v), dtype=np.uint16)
    image[0] = 65535; image[1] = 0
    image[2] = np.arange(v) % 2                            # sums right at the .5 boundaries
    image[3] = (np.arange(v) % width == 0) * 65535
    iL = np.arange(0, v, width, dtype=np.int32)
    binner = type('B', (), dict(iL=iL, iR=iL + width, w=np.full(iL.size, width, np.int32), v_=v, n_bins=iL.size))()
    gpu = opencv_cuda.OpenCVCUDABackend(0)
    mat = GpuMat(); mat.upload(image)
    got = gpu._bin(mat, binner, image.shape).download()
    sums = image.reshape(64, -1, width).astype(np.int64).sum(axis=2)
    np.testing.assert_array_equal(got, np.minimum((sums + width // 2) // width, 65535))


def test_opencv_rejects_bins_wider_than_exact_float32_range(emulated):
    gpu = opencv_cuda.OpenCVCUDABackend(0)
    binner = type('B', (), dict(iL=np.array([0]), iR=np.array([300]), w=np.array([300]), v_=300, n_bins=1))()
    mat = GpuMat(); mat.upload(np.zeros((2, 300), np.uint16))
    with pytest.raises(ValueError, match='256'):
        gpu._bin(mat, binner, (2, 300))


@pytest.mark.parametrize('full', [False, True])
def test_cpu_process_binned_is_process_then_binner(full):
    frame, processor = fixture(); model = Calibration(frame.shape, 'cubic')
    binner = binner_for(frame.shape[1])
    runtime = PreprocessingRuntime('cpu', CPU_CAPS)
    binned, output, grabbed = runtime.process_binned(frame, processor, binner, model, True, snapshot=True, full=full)
    want_binned, want_output, want_grab = expected_binned(frame, processor, model, True, binner)
    np.testing.assert_array_equal(binned, want_binned)
    np.testing.assert_array_equal(grabbed, want_grab)
    if full:
        np.testing.assert_array_equal(output, want_output)
    else:
        assert output is None
    with pytest.raises(ValueError, match='Binner fitted'):
        runtime.process_binned(frame, processor, binner_for(frame.shape[1] + 1))


@pytest.mark.parametrize('requested, broken, expected', [
    ('auto', (), 'cupy_cuda'),
    ('auto', ('cupy_cuda',), 'opencv_cuda'),
    ('opencv_cuda', (), 'opencv_cuda'),
    ('opencv_cuda', ('opencv_cuda',), 'cupy_cuda'),
    ('cupy_cuda', ('cupy_cuda',), 'opencv_cuda'),
    ('opencv_cuda', ('opencv_cuda', 'cupy_cuda'), 'cpu'),
    ('cpu', (), 'cpu'),
])
def test_selection_moves_to_the_other_gpu_before_cpu(emulated, monkeypatch, requested, broken, expected):
    for module, name in ((cupy_cuda, 'CuPyCUDABackend'), (opencv_cuda, 'OpenCVCUDABackend')):
        if getattr(module, name).name in broken:
            class Broken(getattr(module, name)):
                def process_binned(self, *args):
                    raise RuntimeError(f'{self.name} cannot bin')
            monkeypatch.setattr(module, name, Broken)
    frame, processor = fixture()
    runtime = PreprocessingRuntime(requested, GPU_CAPS)
    binner = binner_for(frame.shape[1])
    binned, _, _ = runtime.process_binned(frame, processor, binner, Calibration(frame.shape, 'cubic'), True)
    assert runtime.backend_name == expected
    np.testing.assert_array_equal(binned, expected_binned(frame, processor, Calibration(frame.shape, 'cubic'), True, binner)[0])
    if broken:
        assert 'cannot bin' in runtime.fallback_reason


def test_unavailable_capability_is_skipped(emulated):
    caps = ComputeCapabilities(BackendCapability(reason='no cuBLAS'), BackendCapability(True, 1, 0, 'emulated', 'test'))
    runtime = PreprocessingRuntime('opencv_cuda', caps)
    frame, processor = fixture()
    runtime.process_binned(frame, processor, binner_for(frame.shape[1]))
    assert runtime.backend_name == 'cupy_cuda'


def test_wrong_gpu_binning_falls_back_to_cpu(emulated, monkeypatch):
    frame, processor = fixture()
    runtime = PreprocessingRuntime('cupy_cuda', GPU_CAPS)
    binner = binner_for(frame.shape[1])
    runtime.process_binned(frame, processor, binner)              # selects and validates CuPy
    original = cupy_cuda.CuPyCUDABackend.process_binned
    def off_by_two(self, *args):
        binned, output, grabbed = original(self, *args)
        return binned + 2, output, grabbed
    monkeypatch.setattr(cupy_cuda.CuPyCUDABackend, 'process_binned', off_by_two)
    binned, _, _ = runtime.process_binned(frame, processor, binner, inverted=True)  # new key: re-validated
    assert runtime.backend_name == 'cpu' and 'validation differs' in runtime.fallback_reason
    np.testing.assert_array_equal(binned, expected_binned(frame, processor, None, True, binner)[0])


@pytest.mark.parametrize('backend', ['opencv_cuda', 'cupy_cuda'])
def test_real_cuda_process_binned_when_available(backend):
    from modules.compute_runtime import get_compute_capabilities
    caps = get_compute_capabilities()
    capability = caps.opencv_cuda if backend == 'opencv_cuda' else caps.cupy_cuda
    if not capability.available: pytest.skip(capability.reason)
    frame, processor = fixture(); model = Calibration(frame.shape, 'cubic')
    binner = binner_for(frame.shape[1], m=0.71)
    runtime = PreprocessingRuntime(backend, caps)
    binned, output, grabbed = runtime.process_binned(frame, processor, binner, model, True, snapshot=True, full=True)
    want_binned, want_output, want_grab = expected_binned(frame, processor, model, True, binner)
    assert runtime.backend_name == backend, runtime.fallback_reason
    assert np.max(np.abs(binned.astype(int) - want_binned.astype(int))) <= 1
    assert np.max(np.abs(output.astype(int) - want_output.astype(int))) <= 1
    np.testing.assert_array_equal(grabbed, want_grab)


class _Sampling(_Silent):
    def __init__(self, detail): self.detail = detail; self.records = {}
    def want_detail(self): return self.detail
    def _record(self, name, ms): self.records[name] = ms


@pytest.mark.parametrize('backend', ['cupy_cuda', 'opencv_cuda'])
def test_gpu_stage_breakdown_only_on_sampled_frames(emulated, backend):
    from modules.preprocessing.timing import STAGES
    frame, processor = fixture(); model = Calibration(frame.shape, 'cubic')
    gpu = (cupy_cuda.CuPyCUDABackend if backend == 'cupy_cuda' else opencv_cuda.OpenCVCUDABackend)(0)
    args = (frame, processor._dark_mean, processor, True, True, model, False)
    binner = binner_for(frame.shape[1])
    for detail, expected in ((True, set(STAGES)), (False, set())):
        prof = _Sampling(detail)
        gpu.process_binned(*args, binner, False, prof)
        assert set(prof.records) == expected
        assert all(ms >= 0 for ms in prof.records.values())
    prof = _Sampling(True)                                     # a Grab frame is not broken down
    gpu.process_binned(*args[:-1], True, binner, False, prof)
    assert not prof.records


@pytest.mark.parametrize('backend', ['cupy_cuda', 'opencv_cuda'])
def test_dark_reference_swapped_mid_frame_does_not_mix(emulated, monkeypatch, backend):
    """The GUI thread installing a new dark during a frame (seen on the Orin as a
    1332/3444-count validation failure right at 'Dark loaded ... hot=2')."""
    frame, processor = fixture()
    shape = frame.shape
    rng = np.random.default_rng(3)
    new_dark = rng.integers(0, 3000, shape, dtype=np.uint16)
    new_hot = np.zeros(shape, np.uint8); new_hot[10, 10] = new_hot[20, 30] = 1
    module, name = (cupy_cuda, 'CuPyCUDABackend') if backend == 'cupy_cuda' else (opencv_cuda, 'OpenCVCUDABackend')
    original = getattr(module, name).process_binned
    swaps = []
    def swap_then_process(self, *args):
        if not swaps:                                          # GUI thread acts mid-frame
            swaps.append(True)
            processor.set_dark_reference(new_dark, new_hot)
        return original(self, *args)
    runtime = PreprocessingRuntime(backend, GPU_CAPS)
    binner = binner_for(shape[1])
    from modules.dark import DarkProcessor
    idle = DarkProcessor(); idle.set_dark_enabled(False)
    runtime.process_binned(np.zeros_like(frame), idle, binner)          # select + self-test
    monkeypatch.setattr(getattr(module, name), 'process_binned', swap_then_process)
    old = processor.dark_reference
    binned, _, _ = runtime.process_binned(frame, processor, binner, inverted=True)
    assert swaps and runtime.backend_name == backend, runtime.fallback_reason
    expected = binner.transform(old._apply_hot_pixel(old._apply_dark_subtract(frame.copy()))[::-1][None])[0]
    np.testing.assert_array_equal(binned, expected)                # whole frame used the old reference
    binned, _, _ = runtime.process_binned(frame, processor, binner, inverted=True)
    new = processor.dark_reference
    expected = binner.transform(new._apply_hot_pixel(new._apply_dark_subtract(frame.copy()))[::-1][None])[0]
    np.testing.assert_array_equal(binned, expected)                # next frame uses the new one


def test_set_dark_reference_installs_a_new_immutable_snapshot():
    from modules.dark import DarkProcessor, DarkReference
    processor = DarkProcessor()
    empty = processor.dark_reference
    assert isinstance(empty, DarkReference) and processor._dark_mean is None and processor._hot_coords is None
    dark = np.zeros((4, 5), np.uint16); hot = np.zeros((4, 5), np.uint8); hot[1, 1] = 255
    processor.set_dark_reference(dark, hot)
    first = processor.dark_reference
    assert first is not empty and empty._hot_coords is None            # old snapshot untouched
    assert processor._dark_mean is dark and processor._hot_coords.tolist() == [[1, 1]]
    processor.set_dark_reference(None, None)
    assert first._hot_coords.tolist() == [[1, 1]] and processor._hot_coords is None


def test_opencv_frame_waits_on_its_stream_once_and_reuses_buffers(emulated):
    from cuda_emulation import Stream
    frame, processor = fixture(); model = Calibration(frame.shape, 'cubic')
    gpu = opencv_cuda.OpenCVCUDABackend(0)
    args = (frame, processor._dark_mean, processor, True, True, model, False)
    binner = binner_for(frame.shape[1])
    gpu.process_binned(*args, binner, False, _Silent())               # warm-up: buffers and tables
    buffers = {name: entry[1] for name, entry in gpu._buffers.items()}
    before = Stream.waits
    gpu.process_binned(*args, binner, False, _Silent())
    assert Stream.waits - before == 1                                  # one host wait per frame
    assert all(gpu._buffers[name][1] is mat for name, mat in buffers.items())   # no reallocation
