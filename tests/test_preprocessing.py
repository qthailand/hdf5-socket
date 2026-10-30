"""Compare preprocessing with the original CPU stage order and buffer ownership."""
import sys
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.preprocessing import PreprocessingRuntime
from modules.preprocessing.maps import prepare_maps, horizontal_table, FLAGS
from modules.dark import DarkProcessor
from modules.compute_runtime import BackendCapability, ComputeCapabilities, get_compute_capabilities

CPU_CAPS = ComputeCapabilities(BackendCapability(), BackendCapability())

class Calibration:
    def __init__(self, shape, interp):
        self.image_shape = shape; self.interp = interp
        y, x = np.indices(shape, dtype=np.float32)
        self.map_x = x + np.sin(y / 7) * 1.45 + np.cos(x / 5) * .7
        self.y = y
    def _get_map_y(self): return self.y
    def apply_native(self, frame):
        xy, fraction = cv2.convertMaps(self.map_x, self.y, cv2.CV_16SC2)
        return cv2.remap(frame, xy, fraction, FLAGS[self.interp], borderMode=cv2.BORDER_CONSTANT)

def fixture(shape=(37, 61)):
    rng = np.random.default_rng(46)
    frame = rng.integers(0, 65536, shape, dtype=np.uint16)
    dark = rng.integers(0, 2000, shape, dtype=np.uint16)
    mask = np.zeros(shape, dtype=np.uint8)
    for y, x in [(0, 0), (0, 1), (3, 4), (3, 5), (36, 60)]:
        if y < shape[0] and x < shape[1]: mask[y, x] = 1
    processor = DarkProcessor(); processor.set_dark_reference(dark, mask)
    return frame, processor

def reference(frame, processor, calibration, inverted):
    corrected = frame.copy()
    if processor._dark_enabled:
        corrected = processor._apply_dark_subtract(corrected)
        corrected = processor._apply_hot_pixel(corrected)
    oriented = np.ascontiguousarray(corrected[::-1] if inverted else corrected)
    output = calibration.apply_native(oriented) if calibration is not None else oriented
    return output, oriented

@pytest.mark.parametrize('interp', [None, 'nearest', 'linear', 'cubic', 'lanczos'])
@pytest.mark.parametrize('inverted', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
def test_cpu_matches_original_pipeline(interp, inverted, enabled):
    frame, processor = fixture(); processor.set_dark_enabled(enabled)
    model = Calibration(frame.shape, interp) if interp else None
    expected, snapshot = reference(frame, processor, model, inverted)
    runtime = PreprocessingRuntime('cpu', CPU_CAPS)
    original = frame.copy()
    output, grabbed = runtime.process(frame, processor, model, inverted, snapshot=True)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(grabbed, snapshot)
    np.testing.assert_array_equal(frame, original)
    preserved = output.copy(); saved_grab = grabbed.copy()
    runtime.process(np.zeros_like(frame), processor, model, inverted, snapshot=True)
    np.testing.assert_array_equal(output, preserved)
    np.testing.assert_array_equal(grabbed, saved_grab)
    assert output.flags.c_contiguous and not np.shares_memory(output, frame)

def test_gpu_failure_latches_safe_cpu_fallback():
    frame, processor = fixture(); runtime = PreprocessingRuntime('cpu', CPU_CAPS)
    class Broken:
        name = 'cupy_cuda'
        def process(self, *args): raise RuntimeError('GPU lost')
    runtime._backend = Broken()
    expected, _ = reference(frame, processor, None, True)
    output, _ = runtime.process(frame, processor, inverted=True)
    np.testing.assert_array_equal(output, expected)
    assert runtime.backend_name == 'cpu' and 'GPU lost' in runtime.fallback_reason

def test_wrong_gpu_result_is_rejected():
    frame, processor = fixture(); runtime = PreprocessingRuntime('cpu', CPU_CAPS)
    class Wrong:
        name = 'cupy_cuda'
        def process(self, *args): return np.zeros_like(frame), None
    runtime._backend = Wrong()
    expected, _ = reference(frame, processor, None, False)
    output, _ = runtime.process(frame, processor)
    np.testing.assert_array_equal(output, expected)
    assert runtime.backend_name == 'cpu' and 'validation differs' in runtime.fallback_reason

@pytest.mark.parametrize('interp', ['linear', 'cubic'])
def test_cupy_interpolation_math_matches_opencv(interp):
    frame, _ = fixture(); model = Calibration(frame.shape, interp)
    _, _, xy, frac, _ = prepare_maps(model, True)
    weights = horizontal_table(interp)
    taps = weights.shape[1]; result = np.zeros(frame.shape, np.float32)
    x, y = xy[:,:,0].astype(int), xy[:,:,1].astype(int)
    for k in range(taps):
        sx = x + k - (1 if taps == 4 else 0)
        valid = (sx >= 0) & (sx < frame.shape[1]) & (y >= 0) & (y < frame.shape[0])
        values = frame[np.clip(y,0,frame.shape[0]-1),np.clip(sx,0,frame.shape[1]-1)].astype(np.float32)
        result += np.where(valid,values,0) * weights[frac & 31,k]
    result = np.rint(np.clip(result,0,65535)).astype(np.uint16)
    expected = model.apply_native(np.ascontiguousarray(frame[::-1]))
    assert np.max(np.abs(result.astype(int) - expected.astype(int))) <= 1

def test_shape_mismatch_fails_before_processing():
    frame, processor = fixture()
    runtime = PreprocessingRuntime('cpu', CPU_CAPS)
    with pytest.raises(ValueError, match='does not match'):
        runtime.process(frame, processor, Calibration((10, 10), 'cubic'))

@pytest.mark.parametrize('backend', ['opencv_cuda', 'cupy_cuda'])
def test_real_cuda_backend_when_available(backend):
    caps = get_compute_capabilities()
    capability = caps.opencv_cuda if backend == 'opencv_cuda' else caps.cupy_cuda
    if not capability.available: pytest.skip(capability.reason)
    frame, processor = fixture(); model = Calibration(frame.shape, 'cubic')
    runtime = PreprocessingRuntime(backend, caps)
    output, grabbed = runtime.process(frame, processor, model, True, True)
    expected, expected_grab = reference(frame, processor, model, True)
    assert runtime.backend_name == backend, runtime.fallback_reason
    assert np.max(np.abs(output.astype(int)-expected.astype(int))) <= 1
    np.testing.assert_array_equal(grabbed, expected_grab)
