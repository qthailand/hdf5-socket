# -*- coding: utf-8 -*-
"""Check and benchmark the CUDA backends on a machine with an NVIDIA GPU (PC or Jetson Orin).

Runs, in order, and writes everything to ``gpu_report.json``:

1. Environment: Python, NumPy, OpenCV, CuPy, device, compute capability, driver/runtime.
2. ``modules.compute_runtime`` probe (the same smoke tests hsi uses).
3. The real-hardware tests copied from hsi (``test_real_cuda_*``): GPU result vs CPU
   reference, at most 1 count difference.
4. Full chain (dark + hot pixel + flip + Smile + binning + model) at maximum rate, lossless
   (``overrun=block``), for CPU and each available CUDA backend and several worker counts.
   A run whose backend fell back to CPU is marked so.
5. Fixed line rates with ``overrun=drop`` on the best CUDA setting: lost lines and latency.

    python tools/gpu_check.py
    python tools/gpu_check.py --seconds 15 --rates 1000 1500 2000 3000 --workers 1 2 4

Send ``gpu_report.json`` back for evaluation.
"""
import argparse
import io
import json
import platform
import subprocess
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FULL_CHAIN = ['--set', 'corrections.dark=calibration/dark_frames/poa/dark_b2_e1.0_g1.0.npz',
              '--set', 'corrections.smile=auto', '--set', 'acquisition.raw_inverted=true']


def environment() -> dict:
    import numpy as np
    import cv2
    env = {'python': sys.version.split()[0], 'platform': platform.platform(), 'machine': platform.machine(),
           'numpy': np.__version__, 'opencv': cv2.__version__,
           'opencv_cuda_devices': _safe(lambda: int(cv2.cuda.getCudaEnabledDeviceCount()))}
    try:
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props.get('name', b'')
        env.update(cupy=cp.__version__,
                   device=name.decode(errors='replace') if isinstance(name, bytes) else str(name),
                   compute_capability=f"{props.get('major')}.{props.get('minor')}",
                   device_memory_gb=round(props.get('totalGlobalMem', 0) / 2**30, 1),
                   cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                   cuda_driver=cp.cuda.runtime.driverGetVersion())
    except Exception as exc:
        env['cupy'] = f"unavailable: {type(exc).__name__}: {exc}"
    return env


def _safe(fn):
    try:
        return fn()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def capabilities() -> dict:
    from modules.compute_runtime import initialize_compute
    caps = initialize_compute()
    return {name: {'available': c.available, 'device': c.device_name, 'reason': c.reason}
            for name, c in (('cupy_cuda', caps.cupy_cuda), ('opencv_cuda', caps.opencv_cuda))}


def hardware_tests() -> dict:
    cmd = [sys.executable, '-m', 'pytest', '-q', '-rs', '-p', 'no:cacheprovider', '-k', 'real_cuda',
           'tests/test_preprocessing.py', 'tests/test_preprocessing_binned.py']
    run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    lines = [l for l in run.stdout.splitlines() if l.strip()]
    return {'exit_code': run.returncode, 'summary': lines[-1] if lines else '', 'output_tail': lines[-15:]}


def simulate(args: list[str]) -> dict:
    import simulate as sim
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = sim.main(args + ['--json'])
    try:
        report = json.loads(out.getvalue())
    except json.JSONDecodeError:
        return {'error': f'exit {code}', 'log_tail': err.getvalue().splitlines()[-10:]}
    keep = ('state', 'error', 'backend', 'target_fps', 'mean_processed_fps', 'lines_processed', 'lines_lost',
            'latency_ms', 'stage_ms')
    return {k: report.get(k) for k in keep}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seconds', type=float, default=10.0)
    p.add_argument('--workers', type=int, nargs='+', default=[1, 2, 4])
    p.add_argument('--rates', type=float, nargs='+', default=[1000, 1500, 2000, 3000])
    p.add_argument('--out', default=str(ROOT / 'gpu_report.json'))
    args = p.parse_args(argv)

    report = {'environment': environment()}
    print(json.dumps(report['environment'], indent=2))
    report['capabilities'] = capabilities()
    print(json.dumps(report['capabilities'], indent=2))
    report['hardware_tests'] = hardware_tests()
    print('hardware tests:', report['hardware_tests']['summary'])

    backends = ['cpu'] + [name for name, c in report['capabilities'].items() if c['available']]
    report['max_rate_full_chain'] = []
    for backend in backends:
        for workers in args.workers:
            result = simulate(['--backend', backend, '--workers', str(workers), '--fps', '0',
                               '--overrun', 'block', '--seconds', str(args.seconds)] + FULL_CHAIN)
            result.update(requested_backend=backend, workers=workers,
                          fell_back_to_cpu=backend != 'cpu' and result.get('backend') == 'cpu')
            report['max_rate_full_chain'].append(result)
            print(f"max rate  {backend:<12} workers={workers}: {result.get('mean_processed_fps')} lines/s "
                  f"(ran on {result.get('backend')})")

    gpu_runs = [r for r in report['max_rate_full_chain']
                if r['requested_backend'] != 'cpu' and not r['fell_back_to_cpu'] and r.get('mean_processed_fps')]
    best = max(gpu_runs or report['max_rate_full_chain'], key=lambda r: r.get('mean_processed_fps') or 0)
    report['fixed_rate_full_chain'] = []
    for rate in args.rates:
        result = simulate(['--backend', best['requested_backend'], '--workers', str(best['workers']),
                           '--fps', str(rate), '--overrun', 'drop', '--seconds', str(args.seconds)] + FULL_CHAIN)
        result.update(requested_backend=best['requested_backend'], workers=best['workers'])
        report['fixed_rate_full_chain'].append(result)
        print(f"fixed {rate:>6.0f}/s {best['requested_backend']} workers={best['workers']}: "
              f"lost {result.get('lines_lost')} latency {result.get('latency_ms')}")

    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"report written to {args.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
