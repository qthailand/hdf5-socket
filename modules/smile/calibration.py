# smile/calibration.py
"""Remap utilities, margin computation, and calibration persistence."""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

from .core import _INTERP
from .surface import Surface2D

__all__ = [
    "build_maps",
    "margin_from_map",
    "correct_smile",
    "crop_margin",
    "SmileCalibration",
    "calibration_from_analysis",
    "save_calibration",
    "load_calibration",
    "save_calibration_fast",
    "load_calibration_fast",
]


# -----------------------------------------------------------------
#  Remap + margin
# -----------------------------------------------------------------


def build_maps(
    shape: Tuple[int, int], smile: Surface2D
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build remap arrays for :func:`cv2.remap`.

    Args:
        shape: Image shape ``(H, W)``.
        smile: Fitted displacement surface.

    Returns:
        Tuple ``(map_x, map_y, S)`` where ``map_x`` and ``map_y`` are
        ``float32`` arrays of shape ``(H, W)`` and ``S`` is the surface
        values.
    """
    n_rows, n_cols = shape
    X, Y = np.meshgrid(
        np.arange(n_cols, dtype=np.float32),
        np.arange(n_rows, dtype=np.float32),
    )
    S = smile(Y, X)
    map_x = np.ascontiguousarray(X + S, dtype=np.float32)
    map_y = np.ascontiguousarray(
        np.repeat(
            np.arange(n_rows, dtype=np.float32)[:, None], n_cols, axis=1
        )
    )
    return map_x, map_y, S


def margin_from_map(
    map_x: np.ndarray, interp: str = "cubic"
) -> Tuple[int, int]:
    """Compute the left/right margin where all source pixels exist.

    Args:
        map_x: Horizontal remap array.
        interp: Interpolation method name.

    Returns:
        Tuple ``(left, right)`` pixel counts to crop.

    Raises:
        ValueError: If no valid column exists after correction.
    """
    if interp not in _INTERP:
        raise ValueError(
            f"interp must be one of {list(_INTERP)}, got {interp!r}"
        )
    _, pad_l, pad_r = _INTERP[interp]
    n_cols = map_x.shape[1]
    lo = float(pad_l)
    hi = n_cols - 1.0 if interp == "linear" else n_cols - pad_r - 1e-3

    valid = ((map_x >= lo) & (map_x <= hi)).all(axis=0)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        raise ValueError(
            "No valid columns after smile correction -- shift is too large. "
            "Check the fit or reduce polynomial degree."
        )
    if idx.size != idx[-1] - idx[0] + 1:
        warnings.warn(
            "Valid columns are non-contiguous; margin is approximate."
        )
    return int(idx[0]), int(n_cols - 1 - idx[-1])


def correct_smile(
    frame: np.ndarray, smile: Surface2D, interp: str = "cubic"
) -> Tuple[np.ndarray, Tuple[int, int], np.ndarray, np.ndarray]:
    """Apply smile correction via sub-pixel remapping.

    Args:
        frame: Input image ``(H, W)``.
        smile: Fitted 2-D displacement surface.
        interp: Interpolation method.

    Returns:
        Tuple ``(corrected, margin, map_x, map_y)``.

    Raises:
        ValueError: If *interp* is not a recognised method.
    """
    if interp not in _INTERP:
        raise ValueError(
            f"interp must be one of {list(_INTERP)}, got {interp!r}"
        )
    flag = _INTERP[interp][0]
    frame = np.ascontiguousarray(frame, dtype=np.float32)
    map_x, map_y, _ = build_maps(frame.shape, smile)
    corrected = cv2.remap(
        frame, map_x, map_y,
        interpolation=flag,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return corrected, margin_from_map(map_x, interp), map_x, map_y


def crop_margin(frame: np.ndarray, margin: Tuple[int, int]) -> np.ndarray:
    """Crop invalid (zero) border columns produced by :func:`correct_smile`.

    Args:
        frame: Corrected image with zero-valued borders.
        margin: ``(left, right)`` pixel counts to remove.

    Returns:
        Cropped image, or *frame* unchanged when ``margin == (0, 0)``.
    """
    left, right = margin
    if left + right > 0:
        return frame[:, left : frame.shape[1] - right]
    return frame


# -----------------------------------------------------------------
#  Optimized remap engine  ★ ย้ายขึ้นมาก่อน SmileCalibration
# -----------------------------------------------------------------



class _RemapEngine:
    """Pre-computed remap engine with selectable GPU / CPU backend.

    Backends:
        - ``"cpu"``     : Fixed-point maps (fastest for < 4K images)
        - ``"opencl"``  : cv2.UMat (GPU via OpenCL, good for batch)
        - ``"cuda"``    : cv2.cuda (GPU via CUDA, fastest for > 4K)
        - ``"auto"``    : Pick the fastest available backend

    Example::

        engine = _RemapEngine(map_x, map_y, interp="cubic", backend="cpu")
        out = engine.remap(frame)
    """

    # Ordered from most to least preferred for single-frame throughput.
    _BACKEND_PRIORITY = ["cpu", "cuda", "opencl"]

    def __init__(
        self,
        map_x: np.ndarray,
        map_y: np.ndarray,
        interp: str = "cubic",
        backend: str = "auto",
    ) -> None:
        """Args:
            map_x: Horizontal source-column map ``float32 (H, W)`` (px).
            map_y: Source-row map ``float32 (H, W)`` (px).
            interp: ``"linear"``, ``"cubic"`` or ``"lanczos"``.
            backend: ``"auto"``, ``"cpu"``, ``"opencl"`` or ``"cuda"``.

        Raises:
            ValueError: Unknown ``backend``.
        """
        self.interp = interp
        self._flag = _INTERP[interp][0]
        self._map_x = np.ascontiguousarray(map_x, dtype=np.float32)
        self._map_y = np.ascontiguousarray(map_y, dtype=np.float32)

        # --- Detect available backends ---
        self._cpu_ok = True
        self._opencl_ok = self._detect_opencl()
        self._cuda_ok = self._detect_cuda()

        # --- Select backend ---
        if backend == "auto":
            self.backend = self._pick_auto()
        elif backend in ("cpu", "opencl", "cuda"):
            self.backend = backend
        else:
            raise ValueError(
                f"backend must be 'auto', 'cpu', 'opencl', or 'cuda', "
                f"got {backend!r}"
            )

        # --- Initialise selected backend ---
        self._init_cpu()
        if self.backend == "opencl":
            self._init_opencl()
        elif self.backend == "cuda":
            self._init_cuda()

    # ---- detection ----

    @staticmethod
    def _detect_opencl() -> bool:
        """True if an OpenCL context can be created."""
        try:
            _ = cv2.ocl.getContext()
            return True
        except Exception:
            return False

    @staticmethod
    def _detect_cuda() -> bool:
        """True if OpenCV was built with CUDA and a device is present."""
        try:
            import cv2.cuda  # noqa: F401
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except (ImportError, AttributeError):
            return False

    def _pick_auto(self) -> str:
        """Choose the best backend (CPU wins for single small frames)."""
        # CPU is fastest for < 4K single-frame (no PCIe transfer).
        # GPU wins for > 4K or when staying on GPU between operations.
        n_px = self._map_x.shape[0] * self._map_x.shape[1]
        if n_px > 8_000_000 and self._cuda_ok:
            return "cuda"
        return "cpu"

    # ---- CPU init ----

    def _init_cpu(self) -> None:
        """Prepare fixed-point CPU maps."""
        try:
            self._mx_fix, self._my_fix = cv2.convertMaps(
                self._map_x, self._map_y, cv2.CV_16SC2
            )
            self._cpu_fixed = True
        except cv2.error:
            self._cpu_fixed = False

    # ---- OpenCL init ----

    def _init_opencl(self) -> None:
        """Upload the maps as ``cv2.UMat``."""
        self._umx = cv2.UMat(self._map_x)
        self._umy = cv2.UMat(self._map_y)

    # ---- CUDA init ----

    def _init_cuda(self) -> None:
        """Upload the maps to GPU memory."""
        import cv2.cuda
        self._gpu_mx = cv2.cuda_GpuMat()
        self._gpu_mx.upload(self._map_x)
        self._gpu_my = cv2.cuda_GpuMat()
        self._gpu_my.upload(self._map_y)

    # ---- remap (dispatches to backend) ----

    def remap(self, frame: np.ndarray) -> np.ndarray:
        """Apply remap to a single frame.

        Args:
            frame: Input image ``(H, W)``, any dtype.

        Returns:
            Corrected image ``float32 (H, W)``.
        """
        src = (
            frame
            if frame.dtype == np.float32 and frame.flags.c_contiguous
            else np.ascontiguousarray(frame, dtype=np.float32)
        )

        if self.backend == "cuda":
            return self._remap_cuda(src)
        if self.backend == "opencl":
            return self._remap_opencl(src)
        return self._remap_cpu(src)

    def remap_batch(self, frames: list) -> list:
        """Apply remap to multiple frames.

        CUDA uploads maps once, then processes all frames in the loop
        — amortising transfer overhead.

        Args:
            frames: List of ``(H, W)`` images.

        Returns:
            List of corrected images.
        """
        srcs = [
            f if f.dtype == np.float32 and f.flags.c_contiguous
            else np.ascontiguousarray(f, dtype=np.float32)
            for f in frames
        ]

        if self.backend == "cuda":
            import cv2.cuda
            results = []
            for src in srcs:
                gpu_in = cv2.cuda_GpuMat()
                gpu_in.upload(src)
                gpu_out = cv2.cuda.remap(
                    gpu_in, self._gpu_mx, self._gpu_my,
                    interpolation=self._flag,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                results.append(gpu_out.download())
            return results

        if self.backend == "opencl":
            results = []
            for src in srcs:
                u_src = cv2.UMat(src)
                u_dst = cv2.remap(
                    u_src, self._umx, self._umy,
                    interpolation=self._flag,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                results.append(u_dst.get())
            return results

        return [self._remap_cpu(src) for src in srcs]

    # ---- backend implementations ----

    def _remap_cpu(self, src: np.ndarray) -> np.ndarray:
        """Remap ``src`` on the CPU (fixed-point maps when available); borders are filled with 0.
        Preserves the dtype of ``src`` for uint8/uint16/float32.
        """
        if self._cpu_fixed:
            return cv2.remap(
                src, self._mx_fix, self._my_fix,
                interpolation=self._flag,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        return cv2.remap(
            src, self._map_x, self._map_y,
            interpolation=self._flag,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def _remap_opencl(self, src: np.ndarray) -> np.ndarray:
        """Remap ``src`` through OpenCL; borders are filled with 0."""
        u_src = cv2.UMat(src)
        u_dst = cv2.remap(
            u_src, self._umx, self._umy,
            interpolation=self._flag,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return u_dst.get()

    def _remap_cuda(self, src: np.ndarray) -> np.ndarray:
        """Remap ``src`` on the GPU; borders are filled with 0."""
        import cv2.cuda
        gpu_in = cv2.cuda_GpuMat()
        gpu_in.upload(src)
        gpu_out = cv2.cuda.remap(
            gpu_in, self._gpu_mx, self._gpu_my,
            interpolation=self._flag,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return gpu_out.download()

    # ---- info ----

    @property
    def backend_info(self) -> dict:
        """Return backend availability and selection info."""
        return {
            "selected": self.backend,
            "available": {
                "cpu": self._cpu_ok,
                "opencl": self._opencl_ok,
                "cuda": self._cuda_ok,
            },
            "interp": self.interp,
            "map_shape": self._map_x.shape,
        }


# -----------------------------------------------------------------
#  Calibration dataclass
# -----------------------------------------------------------------


@dataclass
class SmileCalibration:
    """Persistable calibration for repeated application.

    Stores ``map_x`` only (``map_y`` is a trivial row-index array built
    lazily).  A :class:`_RemapEngine` is pre-built after loading so
    that ``apply()`` runs at peak speed from the first call.

    Attributes:
        map_x: Horizontal remap array ``float32 (H, W)``.
        margin: ``(left, right)`` border pixels to crop.
        image_shape: ``(H, W)`` of the calibrated sensor.
        deg_y: Polynomial degree in the spatial direction.
        deg_x: Polynomial degree in the spectral direction.
        surface_coef: Flattened polynomial coefficients.
        y_rng: Row coordinate range used during fitting.
        x_rng: Column coordinate range used during fitting.
        interp: Interpolation method name.
        fit_rms: Residual RMS of the surface fit.
        cv_rms: Leave-one-line-out cross-validated RMS.
        rms_before: Straightness RMS before correction.
        rms_after: Straightness RMS after correction.
        n_lines_used: Number of lines used in the fit.
        n_lines_total: Number of lines detected.
        created: Timestamp string.
        notes: Free-form annotation.
    """

    map_x: np.ndarray
    margin: Tuple[int, int]
    image_shape: Tuple[int, int]
    deg_y: int
    deg_x: int
    surface_coef: np.ndarray
    y_rng: Tuple[float, float]
    x_rng: Tuple[float, float]
    interp: str
    fit_rms: float
    cv_rms: float
    rms_before: float
    rms_after: float
    n_lines_used: int
    n_lines_total: int
    created: str
    notes: str = ""
    backend: str = "auto" 
    y_ref: Optional[float] = None
    _map_y: Optional[np.ndarray] = field(
        default=None, init=False, repr=False
    )
    _engine: Optional[_RemapEngine] = field(
        default=None, init=False, repr=False
    )

    def set_interpolation(self, interp: str) -> None:
        """Change interpolation and invalidate the cached remap engine."""
        if interp not in _INTERP:
            raise ValueError(
                f"interp must be one of {list(_INTERP)}, got {interp!r}"
            )
        self.interp = interp
        self._engine = None
        self.margin = margin_from_map(self.map_x, interp)

    def surface(self) -> Surface2D:
        """Reconstruct the :class:`Surface2D` from stored coefficients.

        Returns:
            The polynomial surface.

        Note:
            For format v1 files, ``y_rng``/``x_rng`` are approximate.
        """
        return Surface2D(
            self.surface_coef, self.deg_y, self.deg_x,
            self.y_rng, self.x_rng,
        )

    def _get_map_y(self) -> np.ndarray:
        """Build (or return cached) ``map_y`` row-index array."""
        h, w = self.image_shape
        if self._map_y is None or self._map_y.shape != (h, w):
            self._map_y = np.ascontiguousarray(
                np.repeat(
                    np.arange(h, dtype=np.float32)[:, None], w, axis=1
                )
            )
        return self._map_y

    def _get_engine(self) -> _RemapEngine:
        """Return (or lazily build) the optimized remap engine."""
        if self._engine is None:
            self._engine = _RemapEngine(
                self.map_x,
                self._get_map_y(),
                interp=self.interp,
                backend=self.backend,
            )
        return self._engine

    def set_backend(self, backend: str) -> None:
        """Switch remap backend at runtime.

        Args:
            backend: "auto", "cpu", "opencl", or "cuda".
        """
        self.backend = backend
        self._engine = None   # force rebuild

    @property
    def backend_info(self) -> dict:
        """Return backend availability and selection info."""
        return self._get_engine().backend_info

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply this calibration to a new frame (optimized).

        Uses a pre-built :class:`_RemapEngine` with fixed-point maps
        for maximum throughput.

        Args:
            frame: Image whose shape must match ``self.image_shape``.

        Returns:
            Corrected image with zero-valued borders.

        Raises:
            ValueError: If *frame* shape does not match.
        """
        if frame.shape != tuple(self.image_shape):
            raise ValueError(
                f"Frame shape mismatch: got {frame.shape}, "
                f"calibrated for {self.image_shape}. "
                f"Re-calibration required."
            )
        return self._get_engine().remap(frame)          # ★ ใช้ engine

    def apply_native(self, frame: np.ndarray) -> np.ndarray:
        """CPU remap preserving sensor dtype and full image dimensions.

        Intended for real-time camera pipelines. Unlike apply(), this method
        avoids the float32 conversion. Integer interpolation is rounded by
        OpenCV; scientific analysis retains the existing float32 API.
        """
        if frame.shape != tuple(self.image_shape):
            raise ValueError(f"Frame shape {frame.shape} != calibrated {self.image_shape}")
        if frame.dtype not in (np.dtype('uint8'), np.dtype('uint16'), np.dtype('float32')):
            raise TypeError("Native remap requires uint8, uint16 or float32")
        return self._get_engine()._remap_cpu(np.ascontiguousarray(frame))

    def apply_cropped(self, frame: np.ndarray) -> np.ndarray:
        """Apply calibration and crop the invalid border.

        Args:
            frame: Image to correct.

        Returns:
            Corrected image with borders removed.
        """
        return crop_margin(self.apply(frame), self.margin)

    def summary(self) -> str:
        """Return a human-readable summary.

        Returns:
            Multi-line summary string.
        """
        left, right = self.margin
        cv_str = (
            f"{self.cv_rms:.5f}" if np.isfinite(self.cv_rms) else "n/a"
        )
        return (
            f"Smile Calibration [{self.created}]\n"
            f"  Sensor size  : "
            f"{self.image_shape[1]}x{self.image_shape[0]} px\n"
            f"  Lines used   : "
            f"{self.n_lines_used}/{self.n_lines_total}\n"
            f"  Fit          : deg_y={self.deg_y} deg_x={self.deg_x} "
            f"interp={self.interp}  RMS={self.fit_rms:.5f} px  "
            f"CV={cv_str} px\n"
            f"  Straightness : "
            f"{self.rms_before:.4f} -> {self.rms_after:.4f} px\n"
            f"  Border crop  : left={left} right={right} px "
            f"({left + right} px total)\n"
            f"  Notes        : {self.notes}"
        )


# -----------------------------------------------------------------
#  Build from analysis
# -----------------------------------------------------------------


def calibration_from_analysis(analysis, notes: str = "") -> SmileCalibration:
    """Build a :class:`SmileCalibration` from analysis results.

    The polynomial degree is taken from the fitted surface automatically.

    Args:
        analysis: Completed :class:`~smile.analysis.SmileAnalysis`.
        notes: Free-form annotation.

    Returns:
        A ready-to-save :class:`SmileCalibration`.
    """
    s = analysis.surface
    map_x, _, _ = build_maps(analysis.frame_orig.shape, s)
    cal = SmileCalibration(
        map_x=map_x,
        margin=tuple(analysis.margin),
        image_shape=tuple(analysis.frame_orig.shape),
        deg_y=s.deg_y,
        deg_x=s.deg_x,
        surface_coef=s.coef.copy(),
        y_rng=s.y_rng,
        x_rng=s.x_rng,
        interp=analysis.interp,
        fit_rms=float(analysis.fit_info["rms"]),
        cv_rms=float(analysis.cv_rms),
        rms_before=float(analysis.rms_before),
        rms_after=float(analysis.rms_after),
        n_lines_used=len(analysis.used_lines),
        n_lines_total=len(analysis.all_lines),
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        notes=notes,
        y_ref=float(analysis.y_ref),
    )
    cal._get_engine()    # ★ pre-build engine
    return cal


# -----------------------------------------------------------------
#  Shared metadata helpers (avoids duplication)
# -----------------------------------------------------------------


def _cal_to_meta(cal: SmileCalibration, format_version: int) -> dict:
    """Build the metadata dict for serialisation.

    Args:
        cal: Calibration to serialise.
        format_version: File format version number.

    Returns:
        JSON-serialisable dict.
    """
    return {
        "format_version": format_version,
        "margin": [int(v) for v in cal.margin],
        "image_shape": [int(v) for v in cal.image_shape],
        "deg_y": cal.deg_y,
        "deg_x": cal.deg_x,
        "surface_coef": cal.surface_coef.tolist(),
        "y_rng": list(cal.y_rng),
        "x_rng": list(cal.x_rng),
        "interp": cal.interp,
        "backend": cal.backend, 
        "fit_rms": cal.fit_rms,
        "cv_rms": cal.cv_rms,
        "rms_before": cal.rms_before,
        "rms_after": cal.rms_after,
        "n_lines_used": cal.n_lines_used,
        "n_lines_total": cal.n_lines_total,
        "created": cal.created,
        "notes": cal.notes,
        "y_ref": cal.y_ref,
    }


def _meta_to_cal(meta: dict, map_x: np.ndarray) -> SmileCalibration:
    """Reconstruct a :class:`SmileCalibration` from metadata + map_x.

    Args:
        meta: Deserialised metadata dict.
        map_x: Horizontal remap array.

    Returns:
        The calibration with ``map_y`` and ``_RemapEngine`` pre-built.
    """
    ver = meta.get("format_version", 1)
    shape = tuple(meta["image_shape"])
    cal = SmileCalibration(
        map_x=map_x,
        margin=tuple(int(v) for v in meta["margin"]),
        image_shape=(int(shape[0]), int(shape[1])),
        deg_y=meta["deg_y"],
        deg_x=meta["deg_x"],
        surface_coef=np.array(meta["surface_coef"]),
        y_rng=tuple(meta.get("y_rng", (0.0, float(shape[0] - 1)))),
        x_rng=tuple(meta.get("x_rng", (0.0, float(shape[1] - 1)))),
        interp=meta.get("interp", "linear" if ver == 1 else "cubic"),
        backend=meta.get("backend", "auto"),
        fit_rms=meta["fit_rms"],
        cv_rms=meta.get("cv_rms", float("nan")),
        rms_before=meta["rms_before"],
        rms_after=meta["rms_after"],
        n_lines_used=meta["n_lines_used"],
        n_lines_total=meta["n_lines_total"],
        created=meta["created"],
        notes=meta.get("notes", ""),
        y_ref=(float(meta["y_ref"]) if meta.get("y_ref") is not None else None),
    )
    cal._get_engine()   # ★ pre-build engine (map_y + fixed-point maps)
    return cal


# -----------------------------------------------------------------
#  Save / load  (compressed — smaller file)
# -----------------------------------------------------------------


def save_calibration(path: str, cal: SmileCalibration) -> None:
    """Write a calibration to a compressed ``.npz`` file.

    Smaller file, slower load.  Use :func:`save_calibration_fast`
    for production pipelines that load repeatedly.

    Args:
        path: Destination file path.
        cal: Calibration to serialise.
    """
    meta = _cal_to_meta(cal, format_version=2)
    np.savez_compressed(
        path,
        map_x=cal.map_x.astype(np.float32),
        metadata=np.frombuffer(
            json.dumps(meta).encode("utf-8"), dtype=np.uint8
        ),
    )


def load_calibration(path: str) -> SmileCalibration:
    """Read a calibration from ``.npz`` (supports format v1–v3).

    Args:
        path: Path to the ``.npz`` file.

    Returns:
        The deserialised :class:`SmileCalibration` with engine pre-built.

    Raises:
        ValueError: If the format version is newer than supported.
    """
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(data["metadata"].tobytes().decode("utf-8"))
        map_x = data["map_x"].astype(np.float32)

    ver = meta.get("format_version", 1)
    if ver > 3:
        raise ValueError(
            f"format_version {ver} is newer than this module supports."
        )
    return _meta_to_cal(meta, map_x)


# -----------------------------------------------------------------
#  Save / load  (uncompressed — faster load)
# -----------------------------------------------------------------


def save_calibration_fast(path: str, cal: SmileCalibration) -> None:
    """Write a calibration to an *uncompressed* ``.npz`` file.

    Larger file (~3-5x), but loads 3-5x faster than
    :func:`save_calibration`.  Preferred for production pipelines.

    Args:
        path: Destination file path.
        cal: Calibration to serialise.
    """
    meta = _cal_to_meta(cal, format_version=3)
    np.savez(
        path,
        map_x=np.ascontiguousarray(cal.map_x, dtype=np.float32),
        metadata=np.frombuffer(
            json.dumps(meta).encode("utf-8"), dtype=np.uint8
        ),
    )


def load_calibration_fast(path: str) -> SmileCalibration:
    """Load a calibration from ``.npz`` (any format v1–v3).

    Same output as :func:`load_calibration`, but skips compressed-file
    overhead when the file was saved with :func:`save_calibration_fast`.

    Args:
        path: Path to the ``.npz`` file.

    Returns:
        The deserialised :class:`SmileCalibration` with engine pre-built.

    Raises:
        ValueError: If the format version is newer than supported.
    """
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(data["metadata"].tobytes().decode("utf-8"))
        map_x = np.ascontiguousarray(data["map_x"], dtype=np.float32)

    ver = meta.get("format_version", 1)
    if ver > 3:
        raise ValueError(
            f"format_version {ver} is newer than this module supports."
        )
    return _meta_to_cal(meta, map_x)
