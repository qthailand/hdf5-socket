# -*- coding: utf-8 -*-
"""Per-pixel linear regression of one line on the three compute engines (CPU, CuPy, OpenCV CUDA).

A regression package (``hsi_package``, ``task_type`` "regression") folds into one weight per
input band: per slit pixel ``y = sum_b w_b z_b + bias`` with ``z`` the spectrum after SNV (or the
spectrum itself, the first derivative / band selection / PCA / scaler being folded into ``w``).
Every engine returns, per pixel, the prediction ``y``, the mean of its input bands (for the object
mask) and whether it is valid (all bands finite; with SNV a standard deviation > 0).

Engines, in hsi's fallback order:

* ``cpu``: one numba kernel (two passes per pixel, float64 sums, no GIL, no BLAS threads).
* ``cupy_cuda``: the line is uploaded, gathered, centred and reduced on the device; one download
  of ``(3, n)`` values.
* ``opencv_cuda``: ``cv2.cuda.gemm`` products for the row means, the centred line and the weighted
  sums, the per-pixel finish on the host (float64). Needs OpenCV built with cuBLAS.

A GPU engine must pass a self-test against the CPU kernel when it is made, and falls back to the
next candidate (then the CPU) with the reason; a GPU failure while running switches that runtime
to the CPU for good (:class:`RegressionRuntime`), as ``modules.preprocessing`` does.

Speed: the work per line is ``slit x bands`` multiply-adds (1024 x 159 = 0.16 M), about 0.1 ms on
one CPU core; uploading the same line costs more than that on a discrete GPU. So ``auto`` runs the
CPU kernel per line, and the GPU engines are there for a GPU-only device or bulk predictions
(:func:`predict_bulk`, e.g. a whole spectrum history).
"""
from __future__ import annotations

import importlib
import logging
import math

import numpy as np
from numba import njit

logger = logging.getLogger(__name__)

ENGINES = ('auto', 'cpu', 'cupy_cuda', 'opencv_cuda')
#: GPU result vs CPU kernel in the self-test: relative to the spread of the CPU predictions
SELFTEST_TOLERANCE = 1e-3
#: with SNV, a spectrum whose SD is at most this fraction of its mean is constant (saturated, dead,
#: the all-zero warm-up line): no value. Relative, so float32 rounding on a GPU decides alike.
SNV_MIN_RELATIVE_SD = 1e-4


@njit(cache=True, nogil=True)
def _regress_line(x, columns, weights, bias, snv, values, means, valid):
    """``values[r] = sum_b weights[b] * z[r, b] + bias`` with ``z = x[r, columns]`` (after SNV with
    the population SD when ``snv``); ``means[r]`` the mean of ``x[r, columns]``; ``valid[r]`` False
    for a non-finite band or (SNV) a constant spectrum - ``values[r]`` is then NaN."""
    n = x.shape[0]
    n_bands = columns.shape[0]
    for r in range(n):
        ok = True
        total = 0.0
        for b in range(n_bands):
            v = x[r, columns[b]]
            if not math.isfinite(v):
                ok = False
            total += v
        mean = total / n_bands
        means[r] = mean
        acc = 0.0
        if snv:
            squares = 0.0
            for b in range(n_bands):
                d = x[r, columns[b]] - mean
                squares += d * d
                acc += weights[b] * d
            deviation = math.sqrt(squares / n_bands)
            if not (deviation > 0.0 and deviation > SNV_MIN_RELATIVE_SD * abs(mean) and math.isfinite(deviation)):
                ok = False
            y = acc / deviation + bias if ok else 0.0
        else:
            for b in range(n_bands):
                acc += weights[b] * x[r, columns[b]]
            y = acc + bias
        if ok and math.isfinite(y):
            values[r] = y
            valid[r] = True
        else:
            values[r] = np.nan
            valid[r] = False


class CpuEngine:
    """The numba kernel (reference of the other engines)."""
    name = 'cpu'

    def predict(self, x, columns, weights, bias: float, snv: bool):
        x = np.ascontiguousarray(x, dtype=np.float32)
        n = x.shape[0]
        values = np.empty(n, np.float32)
        means = np.empty(n, np.float32)
        valid = np.empty(n, np.bool_)
        _regress_line(x, np.ascontiguousarray(columns, np.intp), np.ascontiguousarray(weights, np.float64),
                      float(bias), bool(snv), values, means, valid)
        return values, means, valid


class CupyEngine:
    """CuPy: upload, gather, centre and reduce on the device; one download per line."""
    name = 'cupy_cuda'

    def __init__(self, cupy=None) -> None:
        self._cp = cupy if cupy is not None else importlib.import_module('cupy')
        self._weights_host = self._columns_host = None
        self._weights = self._columns = None

    def _device_constants(self, columns, weights):
        cp = self._cp
        if weights is not self._weights_host:
            self._weights = cp.asarray(np.ascontiguousarray(weights, np.float32))
            self._weights_host = weights
        if columns is not self._columns_host:
            self._columns = cp.asarray(np.ascontiguousarray(columns, np.int64))
            self._columns_host = columns
        return self._columns, self._weights

    def predict(self, x, columns, weights, bias: float, snv: bool):
        cp = self._cp
        columns_d, weights_d = self._device_constants(columns, weights)
        line = cp.asarray(np.ascontiguousarray(x, dtype=np.float32))
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):    # NumPy stand-ins (tests)
            return self._predict(cp, line, columns_d, weights_d, bias, snv)

    @staticmethod
    def _predict(cp, line, columns_d, weights_d, bias, snv):
        z = line[:, columns_d]
        n_bands = float(z.shape[1])
        means = z.sum(axis=1) / np.float32(n_bands)
        if snv:
            z = z - means[:, None]
            deviation = cp.sqrt((z * z).sum(axis=1) / np.float32(n_bands))
            values = (z @ weights_d) / deviation + np.float32(bias)
            valid = cp.isfinite(values) & (deviation > 0) & (deviation > np.float32(SNV_MIN_RELATIVE_SD) * abs(means))
        else:
            values = z @ weights_d + np.float32(bias)
            valid = cp.isfinite(values) & cp.isfinite(means)
        out = cp.stack([values.astype(cp.float32), means.astype(cp.float32), valid.astype(cp.float32)]).get()
        good = out[2] > 0.5
        values = np.where(good, out[0], np.float32(np.nan)).astype(np.float32)
        return values, out[1].astype(np.float32), good


class OpenCVEngine:
    """OpenCV CUDA: row means, the centred line (SNV), its squares and the weighted sums as
    ``gemm`` products on the device (two passes, like the CPU kernel); three ``(n, 1)`` downloads,
    the per-pixel finish on the host."""
    name = 'opencv_cuda'

    def __init__(self, cv2=None) -> None:
        self._cv2 = cv2 if cv2 is not None else importlib.import_module('cv2')
        self._stream = self._cv2.cuda_Stream()
        self._weights_host = None
        self._constants: dict = {}
        self._buffers: dict = {}

    def _buffer(self, name, rows, cols):
        key = (name, rows, cols)
        mat = self._buffers.get(key)
        if mat is None:
            mat = self._buffers[key] = self._cv2.cuda_GpuMat(rows, cols, self._cv2.CV_32F)
        return mat

    def _constant(self, name, host):
        mat = self._cv2.cuda_GpuMat()
        mat.upload(np.ascontiguousarray(host, np.float32), self._stream)
        self._constants[name] = mat
        return mat

    def predict(self, x, columns, weights, bias: float, snv: bool):
        cv2, stream = self._cv2, self._stream
        z = np.ascontiguousarray(np.asarray(x, np.float32)[:, np.asarray(columns, np.intp)])
        n, n_bands = z.shape
        if weights is not self._weights_host or self._constants.get('bands') != n_bands:
            self._constant('weights', np.asarray(weights, np.float32).reshape(-1, 1))
            self._constant('mean', np.full((n_bands, 1), 1.0 / n_bands, np.float32))
            self._constant('ones_row', np.ones((1, n_bands), np.float32))
            self._constants['bands'] = n_bands
            self._weights_host = weights
        c = self._constants
        line = self._buffer('line', n, n_bands)
        line.upload(z, stream)
        means = self._buffer('means', n, 1)
        cv2.cuda.gemm(line, c['mean'], 1.0, None, 0.0, dst=means, stream=stream)
        source = line
        if snv:
            spread = self._buffer('spread', n, n_bands)
            cv2.cuda.gemm(means, c['ones_row'], 1.0, None, 0.0, dst=spread, stream=stream)
            centred = self._buffer('centred', n, n_bands)
            cv2.cuda.subtract(line, spread, dst=centred, stream=stream)
            squared = self._buffer('squared', n, n_bands)
            cv2.cuda.sqr(centred, dst=squared, stream=stream)
            variance = self._buffer('variance', n, 1)
            cv2.cuda.gemm(squared, c['mean'], 1.0, None, 0.0, dst=variance, stream=stream)
            source = centred
        dots = self._buffer('dots', n, 1)
        cv2.cuda.gemm(source, c['weights'], 1.0, None, 0.0, dst=dots, stream=stream)
        mean_host = means.download(stream)[:, 0].astype(np.float64)
        dot_host = dots.download(stream)[:, 0].astype(np.float64)
        variance_host = variance.download(stream)[:, 0].astype(np.float64) if snv else None
        stream.waitForCompletion()
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            if snv:
                deviation = np.sqrt(np.maximum(variance_host, 0.0))
                values = dot_host / deviation + bias
                valid = np.isfinite(values) & (deviation > 0) & (deviation > SNV_MIN_RELATIVE_SD * np.abs(mean_host))
            else:
                values = dot_host + bias
                valid = np.isfinite(values) & np.isfinite(mean_host)
        values = np.where(valid, values, np.nan).astype(np.float32)
        return values, mean_host.astype(np.float32), valid


_CLASSES = {'cpu': CpuEngine, 'cupy_cuda': CupyEngine, 'opencv_cuda': OpenCVEngine}


def self_test(engine, seed: int = 0) -> float:
    """Largest difference of ``engine`` from the CPU kernel on random lines (relative to the spread
    of the predictions), with and without SNV; raises ``RuntimeError`` above the tolerance."""
    rng = np.random.default_rng(seed)
    cpu = CpuEngine()
    worst = 0.0
    for snv in (False, True):
        x = (0.3 + 0.2 * rng.random((96, 48))).astype(np.float32)
        x[3, 5] = np.nan                                     # an invalid pixel must stay invalid
        x[7] = 0.4                                           # constant: invalid with SNV only
        columns = np.sort(rng.choice(48, 40, replace=False)).astype(np.intp)
        weights = rng.standard_normal(40)
        expected, expected_means, expected_valid = cpu.predict(x, columns, weights, 0.7, snv)
        values, means, valid = engine.predict(x, columns, weights, 0.7, snv)
        if not np.array_equal(valid, expected_valid):
            raise RuntimeError(f"{engine.name}: valid pixels differ from the CPU kernel (snv={snv})")
        spread = max(float(np.ptp(expected[expected_valid])), 1e-6)
        error = float(np.abs(values[valid] - expected[valid]).max()) / spread
        error = max(error, float(np.abs(means[valid] - expected_means[valid]).max()) / 0.5)
        if not error <= SELFTEST_TOLERANCE:
            raise RuntimeError(f"{engine.name}: differs from the CPU kernel by {error:.3g} (relative, snv={snv})")
        worst = max(worst, error)
    return worst


def _candidates(requested: str) -> list:
    if requested == 'cupy_cuda':
        return ['cupy_cuda', 'opencv_cuda', 'cpu']
    if requested == 'opencv_cuda':
        return ['opencv_cuda', 'cupy_cuda', 'cpu']
    return ['cpu']


def _available(name: str) -> str | None:
    """``None`` when the CUDA engine ``name`` can be tried here, else why not."""
    try:
        from modules.compute_runtime import get_compute_capabilities
        capability = getattr(get_compute_capabilities(), name, None)
    except Exception as exc:                               # no compute_runtime: try anyway
        logger.debug("compute capabilities unknown: %s", exc)
        return None
    if capability is not None and not capability.available:
        return capability.reason or 'not available'
    return None


class RegressionRuntime:
    """The engine one worker thread uses (CUDA contexts belong to the thread that made them).

    Args:
        requested: ``auto`` | ``cpu`` | ``cupy_cuda`` | ``opencv_cuda`` (``model.regression_engine``).
        modules: For tests: ``{'cupy': module, 'cv2': module}`` stand-ins (then the capability
            probe is skipped).
    """

    def __init__(self, requested: str = 'auto', modules: dict | None = None) -> None:
        if requested not in ENGINES:
            raise ValueError(f"regression engine must be one of {ENGINES}, got {requested!r}")
        self.requested = requested
        self.fallback_reason = ''
        self._cpu = CpuEngine()
        reasons = []
        self.engine = self._cpu
        for name in _candidates(requested):
            if name == 'cpu':
                break
            why = None if modules is not None else _available(name)
            if why is not None:
                reasons.append(f"{name}: {why}")
                continue
            try:
                kwargs = {}
                if modules is not None:
                    kwargs = {'cupy': modules.get('cupy')} if name == 'cupy_cuda' else {'cv2': modules.get('cv2')}
                engine = _CLASSES[name](**kwargs)
                self_test(engine)
            except Exception as exc:
                reasons.append(f"{name}: {type(exc).__name__}: {exc}")
                continue
            self.engine = engine
            break
        if requested not in ('auto', self.engine.name) or reasons:
            self.fallback_reason = '; '.join(reasons)
        if self.fallback_reason:
            logger.warning("Regression engine %s requested, running %s (%s)", requested, self.engine.name,
                           self.fallback_reason)

    @property
    def name(self) -> str:
        return self.engine.name

    def predict(self, x, columns, weights, bias: float, snv: bool):
        """``(values (n,) float32, means (n,) float32, valid (n,) bool)`` of one line."""
        if self.engine is self._cpu:
            return self._cpu.predict(x, columns, weights, bias, snv)
        try:
            return self.engine.predict(x, columns, weights, bias, snv)
        except Exception as exc:                            # a GPU failure: CPU from here on
            self.fallback_reason = f"{self.engine.name} failed while running ({type(exc).__name__}: {exc}); CPU"
            logger.exception("Regression engine %s failed: CPU from now on", self.engine.name)
            self.engine = self._cpu
            return self._cpu.predict(x, columns, weights, bias, snv)


def predict_bulk(runtime: RegressionRuntime, spectra, columns, weights, bias: float, snv: bool,
                 chunk: int = 65536):
    """:meth:`RegressionRuntime.predict` over many pixels ``(..., C)`` in chunks; results shaped
    like ``spectra[..., 0]``."""
    spectra = np.asarray(spectra)
    flat = spectra.reshape(-1, spectra.shape[-1])
    values = np.empty(flat.shape[0], np.float32)
    means = np.empty(flat.shape[0], np.float32)
    valid = np.empty(flat.shape[0], bool)
    for start in range(0, flat.shape[0], chunk):
        part = slice(start, start + chunk)
        values[part], means[part], valid[part] = runtime.predict(flat[part], columns, weights, bias, snv)
    shape = spectra.shape[:-1]
    return values.reshape(shape), means.reshape(shape), valid.reshape(shape)
