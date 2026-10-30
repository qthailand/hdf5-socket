# smile/test_cmd.py
"""Command-line smoke test with performance timing.

Usage:
    python -m smile.test_cmd image.png
    python -m smile.test_cmd image.png --repeat 5
    python -m smile.test_cmd image.png --cal model.npz
    python -m smile.test_cmd image.png --cal model.npz --production
    python -m smile.test_cmd image.png --cal model.npz --production --interp linear
    python -m smile.test_cmd image.png --cal model.npz --gpu-compare
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Ensure package root is importable when run directly.
_pkg_root = str(Path(__file__).resolve().parent.parent)
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from smile import (
    SmileCorrector,
    analyze_smile,
    load_frame,
)
from smile.core import _INTERP
from smile.calibration import (
    load_calibration,
    load_calibration_fast,
)

# ------------------------------------------------------------------
#  Timing helpers
# ------------------------------------------------------------------


def _fmt_ms(seconds: float) -> str:
    """Format seconds as milliseconds with adaptive precision."""
    ms = seconds * 1000.0
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    if ms >= 10:
        return f"{ms:.1f} ms"
    return f"{ms:.3f} ms"


class Timer:
    """Context manager that records elapsed wall time.

    Usage::

        with Timer("load") as t:
            frame = load_frame(path)
        print(t)
    """

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.elapsed: float = 0.0
        self._t0: float = 0.0

    def __enter__(self) -> Timer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._t0

    def __repr__(self) -> str:
        return f"{self.label:>28s} : {_fmt_ms(self.elapsed)}"


def _stats(times: List[float]) -> str:
    """Return min / mean / max / std of a list of timings."""
    arr = np.array(times)
    return (
        f"min={_fmt_ms(arr.min())}  "
        f"mean={_fmt_ms(arr.mean())}  "
        f"max={_fmt_ms(arr.max())}  "
        f"std={_fmt_ms(arr.std())}"
    )


# ------------------------------------------------------------------
#  Single-run pipeline test
# ------------------------------------------------------------------


def cli_pipeline(
    image_path: str,
    cal_path: Optional[str] = None,
) -> Dict[str, float]:
    """Full pipeline test on a real image with timing.

    Measures wall-clock time for each stage:
        1. load      -- read image from disk
        2. fit       -- detect lines + trace + surface fit
        3. transform -- remap with fitted model
        4. (optional) apply -- remap using a pre-saved calibration

    Args:
        image_path: Path to the Hg-Ar calibration image.
        cal_path: Optional path to a saved ``.npz`` calibration.

    Returns:
        Dict of stage name -> elapsed seconds.
    """
    timings: Dict[str, float] = {}
    print(f"\n{'=' * 56}")
    print(f"  Image: {image_path}")
    print(f"{'=' * 56}")

    # --- 1. Load ---
    with Timer("load image") as t:
        frame = load_frame(image_path)
    timings["load"] = t.elapsed
    print(t)
    h, w = frame.shape
    print(
        f"{'':>28s} : {w}x{h} px  "
        f"max={frame.max():.0f}  dtype={frame.dtype}"
    )
    assert frame.ndim == 2, f"Expected 2-D image, got {frame.shape}"

    # --- 2. Fit (analyze_smile) ---
    with Timer("fit (analyze_smile)") as t:
        result = analyze_smile(frame, verbose=True)
    timings["fit"] = t.elapsed
    print(t)

    assert len(result.used_lines) > 0, "No lines used in fit"
    assert result.rms_after < result.rms_before, (
        f"Straightness did not improve: "
        f"{result.rms_before:.4f} -> {result.rms_after:.4f}"
    )

    cv_str = (
        f"{result.cv_rms:.4f}" if np.isfinite(result.cv_rms) else "n/a"
    )
    print(
        f"{'':>28s} : "
        f"lines={len(result.used_lines)}/{len(result.all_lines)}  "
        f"RMS={result.rms_before:.4f}->{result.rms_after:.4f}  "
        f"CV={cv_str}  "
        f"margin={result.margin}"
    )

    # --- 3. Transform (correct) ---
    corrector = SmileCorrector(deg_y=2, deg_x=1)
    with Timer("fit SmileCorrector") as t:
        corrector.fit(frame)
    timings["fit_corrector"] = t.elapsed
    print(t)

    with Timer("transform (crop=True)") as t:
        corrected = corrector.transform(frame, crop=True)
    timings["transform"] = t.elapsed
    print(t)
    print(
        f"{'':>28s} : "
        f"{w}x{h} -> {corrected.shape[1]}x{corrected.shape[0]}"
    )
    assert corrected.shape[1] < frame.shape[1], (
        f"Expected narrower output, got {corrected.shape[1]} >= {w}"
    )

    with Timer("transform (crop=False)") as t:
        corrected_full = corrector.transform(frame, crop=False)
    timings["transform_full"] = t.elapsed
    print(t)
    assert corrected_full.shape == frame.shape

    # --- 4. Optional: apply from saved calibration ---
    if cal_path is not None:
        with Timer("load calibration") as t:
            cal = load_calibration(cal_path)
        timings["load_cal"] = t.elapsed
        print(t)

        with Timer("apply_cropped") as t:
            applied = cal.apply_cropped(frame)
        timings["apply"] = t.elapsed
        print(t)
        print(
            f"{'':>28s} : "
            f"{w}x{h} -> {applied.shape[1]}x{applied.shape[0]}"
        )

    # --- Summary ---
    print(f"\n{'-' * 56}")
    print("  SUMMARY")
    print(f"{'-' * 56}")
    for stage, sec in timings.items():
        print(f"  {stage:>24s} : {_fmt_ms(sec)}")
    total = sum(timings.values())
    print(f"  {'TOTAL':>24s} : {_fmt_ms(total)}")
    print(f"{'=' * 56}\n")

    return timings


# ------------------------------------------------------------------
#  Multi-run benchmark (full pipeline)
# ------------------------------------------------------------------


def benchmark(
    image_path: str,
    repeat: int = 5,
    cal_path: Optional[str] = None,
) -> Dict[str, List[float]]:
    """Run the full pipeline multiple times and report timing statistics.

    Args:
        image_path: Path to the Hg-Ar calibration image.
        repeat: Number of repetitions.
        cal_path: Optional path to a saved ``.npz`` calibration.

    Returns:
        Dict of stage name -> list of elapsed seconds.
    """
    all_times: Dict[str, List[float]] = {}
    for i in range(repeat):
        print(f"\n--- Run {i + 1}/{repeat} ---")
        t = cli_pipeline(image_path, cal_path)
        for k, v in t.items():
            all_times.setdefault(k, []).append(v)

    print(f"\n{'=' * 56}")
    print(f"  BENCHMARK  ({repeat} runs)")
    print(f"{'=' * 56}")
    for stage, times in all_times.items():
        print(f"  {stage:>24s} : {_stats(times)}")
    print(f"{'=' * 56}\n")

    return all_times


# ------------------------------------------------------------------
#  Production benchmark (load once, transform N times)
# ------------------------------------------------------------------


def benchmark_production(
    image_path: str,
    cal_path: str,
    repeat: int = 100,
    interp: str = "cubic",
    backend: str = "auto",
) -> Dict[str, object]:
    """Production benchmark: load model once, transform N frames.

    This mirrors the real-world pattern:
        load model once  →  transform many frames

    Args:
        image_path: Path to the calibration image.
        cal_path: Path to saved ``.npz`` calibration.
        repeat: Number of transforms to measure.
        interp: Interpolation method ("linear", "cubic", "lanczos").
        backend: Remap backend ("auto", "cpu", "opencl", "cuda").

    Returns:
        Dict with timing statistics.
    """
    # --- Load model once ---
    with Timer("load model (once)") as t_load:
        cal = load_calibration_fast(cal_path)

    # Override interpolation / backend for speed test.
    cal.set_interpolation(interp)
    cal.set_backend(backend)

    frame = load_frame(image_path)
    h, w = frame.shape

    # Warm-up (JIT / cache / engine build).
    _ = cal.apply_cropped(frame)

    # --- Transform N times ---
    times: List[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        corrected = cal.apply_cropped(frame)
        times.append(time.perf_counter() - t0)

    arr = np.array(times)
    ms = arr * 1000

    info = cal.backend_info
    print(f"\n{'=' * 56}")
    print(f"  PRODUCTION BENCHMARK  ({repeat} transforms)")
    print(f"{'=' * 56}")
    print(f"  Backend        : {info['selected']}")
    print(f"  Interpolation  : {interp}")
    print(f"  Image          : {w}x{h} px")
    print(f"  Output         : "
          f"{corrected.shape[1]}x{corrected.shape[0]} px")
    print(f"  Load model     : {_fmt_ms(t_load.elapsed)}  (once)")
    print(f"  Transform      : {ms.mean():.3f} ms avg  "
          f"(min={ms.min():.3f}  max={ms.max():.3f}  std={ms.std():.3f})")
    print(f"  Throughput     : {1000.0 / ms.mean():.0f} fps")
    print(f"  Total ({repeat}x)  : "
          f"{_fmt_ms(t_load.elapsed + arr.sum())}")
    print(f"{'=' * 56}\n")

    return {
        "backend": info["selected"],
        "interp": interp,
        "transform_mean": float(ms.mean()),
        "transform_min": float(ms.min()),
        "transform_max": float(ms.max()),
        "throughput_fps": 1000.0 / ms.mean(),
    }


# ------------------------------------------------------------------
#  GPU vs CPU comparison
# ------------------------------------------------------------------


def benchmark_gpu_vs_cpu(
    image_path: str,
    cal_path: str,
    repeat: int = 200,
) -> Dict[str, np.ndarray]:
    """Compare all available remap backends on the same image.

    Tests: cpu (cubic), cpu (linear), opencl, cuda.

    Args:
        image_path: Path to the calibration image.
        cal_path: Path to saved ``.npz`` calibration.
        repeat: Number of transforms per backend.

    Returns:
        Dict of backend name -> array of elapsed seconds.
    """
    frame = load_frame(image_path)
    src = np.ascontiguousarray(frame, dtype=np.float32)
    h, w = frame.shape

    # Configurations to test: (label, interp, backend)
    configs = [
        ("cpu+cubic",  "cubic",  "cpu"),
        ("cpu+linear", "linear", "cpu"),
        ("opencl",     "cubic",  "opencl"),
        ("cuda",       "cubic",  "cuda"),
    ]

    results: Dict[str, np.ndarray] = {}

    print(f"\n{'=' * 56}")
    print(f"  BACKEND COMPARISON  ({repeat} transforms, {w}x{h})")
    print(f"{'=' * 56}")

    for label, interp, backend in configs:
        cal = load_calibration_fast(cal_path)
        cal.set_interpolation(interp)
        try:
            cal.set_backend(backend)
            info = cal.backend_info
            if not info["available"].get(backend, False):
                print(f"  {label:>12s} : NOT AVAILABLE")
                continue

            # Warm-up
            _ = cal.apply_cropped(src)

            times = []
            for _ in range(repeat):
                t0 = time.perf_counter()
                _ = cal.apply_cropped(src)
                times.append(time.perf_counter() - t0)

            arr = np.array(times)
            results[label] = arr
            ms = arr.mean() * 1000
            print(f"  {label:>12s} : {ms:.3f} ms  "
                  f"({1000 / ms:.0f} fps)")

        except Exception as exc:
            print(f"  {label:>12s} : FAILED ({exc})")

    # Winner
    if results:
        best = min(results, key=lambda k: results[k].mean())
        best_ms = results[best].mean() * 1000
        print(f"\n  Winner: {best}  ({best_ms:.3f} ms, "
              f"{1000 / best_ms:.0f} fps)")
    print(f"{'=' * 56}\n")

    return results


# ------------------------------------------------------------------
#  Entry point
# ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smile correction pipeline test with timing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m smile.test_cmd img.png\n"
            "  python -m smile.test_cmd img.png --cal model.npz --production\n"
            "  python -m smile.test_cmd img.png --cal model.npz --gpu-compare\n"
            "  python -m smile.test_cmd img.png --cal model.npz "
            "--production --interp linear --backend cuda\n"
        ),
    )
    parser.add_argument(
        "image",
        help="Path to Hg-Ar calibration image",
    )
    parser.add_argument(
        "--repeat", "-n",
        type=int,
        default=1,
        help="Number of benchmark runs (default: 1)",
    )
    parser.add_argument(
        "--cal",
        type=str,
        default=None,
        help="Path to saved .npz calibration matrix",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Production benchmark (load once, transform N times). "
             "Requires --cal.",
    )
    parser.add_argument(
        "--interp",
        type=str,
        default="cubic",
        choices=["linear", "cubic", "lanczos"],
        help="Interpolation method (default: cubic)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "cpu", "opencl", "cuda"],
        help="Remap backend (default: auto)",
    )
    parser.add_argument(
        "--gpu-compare",
        action="store_true",
        help="Compare all available backends (CPU/GPU). Requires --cal.",
    )
    args = parser.parse_args()

    if not Path(args.image).is_file():
        sys.exit(f"File not found: {args.image}")

    # --- GPU comparison mode ---
    if args.gpu_compare:
        if args.cal is None:
            sys.exit("--gpu-compare requires --cal model.npz")
        if not Path(args.cal).is_file():
            sys.exit(f"Calibration file not found: {args.cal}")
        benchmark_gpu_vs_cpu(
            args.image,
            args.cal,
            repeat=max(args.repeat, 100),
        )
        return

    # --- Production mode ---
    if args.production:
        if args.cal is None:
            sys.exit("--production requires --cal model.npz")
        if not Path(args.cal).is_file():
            sys.exit(f"Calibration file not found: {args.cal}")
        benchmark_production(
            args.image,
            args.cal,
            repeat=max(args.repeat, 100),
            interp=args.interp,
            backend=args.backend,
        )
        return

    # --- Standard mode ---
    if args.repeat <= 1:
        cli_pipeline(args.image, args.cal)
    else:
        benchmark(args.image, args.repeat, args.cal)


if __name__ == "__main__":
    main()
