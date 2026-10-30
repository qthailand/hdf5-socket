# -*- coding: utf-8 -*-
"""Run the pushbroom simulation without GUI and report throughput, loss and latency.

    python simulate.py --seconds 10
    python simulate.py --fps 1000 --workers 4 --seconds 20
    python simulate.py --fps 0 --overrun block --seconds 10      # maximum lossless rate
    python simulate.py --source dataset/dataset.bip.hdr --fps 300
    python simulate.py --backend cpu --switch cupy_cuda@10 --seconds 20   # CPU, then CuPy after 10 s
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from pipeline import build_pipeline
from sim_config import load_config


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=None, help='simulation.json (default: beside this script)')
    p.add_argument('--source', default=None, help='file or folder to play (overrides source.path)')
    p.add_argument('--pattern', default=None)
    p.add_argument('--fps', type=float, default=None, help='line rate; 0 = as fast as possible')
    p.add_argument('--overrun', choices=('drop', 'block'), default=None)
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--backend', choices=('auto', 'cpu_fused', 'cpu', 'cupy_cuda', 'opencv_cuda'), default=None)
    p.add_argument('--seconds', type=float, default=10.0, help='run time (stops earlier at the end of a non-looping playlist)')
    p.add_argument('--no-loop', action='store_true', help='play the files once')
    p.add_argument('--json', action='store_true', help='print the final report as JSON')
    p.add_argument('--set', action='append', default=[], metavar='SECTION.KEY=VALUE',
                   help='override any simulation.json value, e.g. --set corrections.smile=auto (JSON values)')
    p.add_argument('--switch', action='append', default=[], metavar='BACKEND@SECONDS',
                   help='switch the correction backend while running, e.g. --switch cupy_cuda@10 --switch cpu@20')
    args = p.parse_args(argv)
    switches = []
    for item in args.switch:
        name, _, at = item.partition('@')
        try:
            switches.append((float(at), name))
        except ValueError:
            p.error(f"--switch expects BACKEND@SECONDS, got {item!r}")
        if name not in ('auto', 'cpu_fused', 'cpu', 'cupy_cuda', 'opencv_cuda'):
            p.error(f"--switch: unknown backend {name!r}")
    switches.sort()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s')
    overrides = {'camera': {'fps': args.fps, 'overrun': args.overrun, 'loop': False if args.no_loop else None},
                 'processing': {'workers': args.workers}, 'corrections': {'backend': args.backend}}
    for item in args.set:
        key, _, text = item.partition('=')
        section, _, name = key.partition('.')
        if not name or not _:
            p.error(f"--set expects SECTION.KEY=VALUE, got {item!r}")
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = text
        overrides.setdefault(section, {})[name] = value
    config = load_config(args.config, **overrides)
    if args.source:                     # a path typed on the command line is relative to the shell
        config = replace(config, source=replace(config.source, path=str(Path(args.source).resolve()),
                                                pattern=args.pattern))

    pipeline = build_pipeline(config)
    pipeline.start()
    t_end = time.perf_counter() + args.seconds
    try:
        while time.perf_counter() < t_end:
            time.sleep(1.0)
            s = pipeline.snapshot()
            while switches and s.state == 'running' and s.elapsed_s >= switches[0][0]:
                print(f"[{s.elapsed_s:6.1f}s] switching correction backend -> {switches[0][1]}",
                      file=sys.stderr if args.json else sys.stdout, flush=True)
                pipeline.set_backend(switches.pop(0)[1])
            print(f"[{s.elapsed_s:6.1f}s] {s.state:<8} cam {s.camera_fps:7.1f}/s  proc {s.processed_fps:7.1f}/s  "
                  f"lines {s.lines_processed:>8}  lost {s.lines_lost:>6}  buf {s.buffers_in_use:>3}/{s.buffer_frames}  "
                  f"lat p50 {s.latency_ms['p50']:6.2f} p99 {s.latency_ms['p99']:6.2f} ms  {s.backend}"
                  f"{' (' + s.backend_note + ')' if s.backend_note else ''}",
                  file=sys.stderr if args.json else sys.stdout, flush=True)
            if s.state in ('finished', 'error'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()

    s = pipeline.snapshot()
    names = pipeline.classifier.category_names
    total = int(s.counts.sum())
    report = {
        'state': s.state, 'error': s.error, 'elapsed_s': round(s.elapsed_s, 2),
        'target_fps': s.target_fps, 'frames_delivered': s.frames_delivered,
        'lines_processed': s.lines_processed, 'lines_lost': s.lines_lost, 'schedule_slips': s.schedule_slips,
        'mean_processed_fps': round(s.lines_processed / s.elapsed_s, 1) if s.elapsed_s else 0.0,
        'latency_ms': {k: round(v, 3) for k, v in s.latency_ms.items()},
        'stage_ms': {k: round(v, 4) for k, v in s.stage_ms.items()},
        'backend': s.backend, 'requested_backend': s.requested_backend, 'backend_note': s.backend_note,
        'class_pixels': {names.get(int(c), str(int(c))): int(s.counts[int(c)])
                         for c in np.flatnonzero(s.counts)},
        'class_share': {names.get(int(c), str(int(c))): round(float(s.counts[int(c)]) / total, 4)
                        for c in np.flatnonzero(s.counts)} if total else {},
        'mm_per_line': round(config.camera.belt_speed_m_s * 1000.0 / s.target_fps, 3) if s.target_fps else None,
    }
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        for key, value in report.items():
            print(f"{key:>18}: {value}")
    return 0 if s.state != 'error' else 1


if __name__ == '__main__':
    sys.exit(main())
