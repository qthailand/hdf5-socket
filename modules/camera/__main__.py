"""Command line: ``python -m modules.camera --backend poa --grab frame.npy``.

Opens the camera, optionally sets gain/exposure/bin, reads a few frames, prints the
device information and frame statistics, and can save the last raw frame as ``.npy``
(``--grab``) and/or as an image with cv2 (``--image frame.png``).
"""
from __future__ import annotations
# Allow running this file directly (IDE 'run file', runfile): make the project importable.
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "modules.camera"

import argparse
import time
from types import SimpleNamespace

import cv2
import numpy as np

from .factory import BACKENDS, create_backend


def main(argv=None) -> int:
    """Run the camera check; returns a process exit code (0 on success)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, default="poa")
    parser.add_argument("--index", type=int, default=0, help="device index (GenICam, XIMEA)")
    parser.add_argument("--cti", help="GenTL producer (.cti) for GenICam")
    parser.add_argument("--gain", type=float, default=1)
    parser.add_argument("--exposure-ms", type=float, default=10.0)
    parser.add_argument("--bin", type=int, default=1)
    parser.add_argument("--frames", type=int, default=20, help="frames to read for statistics")
    parser.add_argument("--grab", help="save the last raw frame to this .npy file")
    parser.add_argument("--image", help="save the last raw frame with cv2 (.png/.tif keep 16 bits, values unchanged)")
    args = parser.parse_args(argv)

    config = SimpleNamespace(camera_backend=args.backend, camera_index=args.index, cti_path=args.cti,
                             default_gain=args.gain, default_exposure_ms=args.exposure_ms, default_bin=args.bin)
    camera = create_backend(config)
    try:
        camera.open()
        if args.bin in camera.get_bins():
            camera.set_bin(args.bin)
        info = camera.info
        print(f"{args.backend}: {info.width}x{info.height}, {info.bit_depth}-bit, bin {camera.current_bin}, "
              f"bins {camera.get_bins()}, gain {camera.gain_range}, exposure_us {camera.exposure_range_us}")
        camera.set_params(args.gain, args.exposure_ms)
        camera.start_stream()
        frame, start, got = None, time.perf_counter(), 0
        for _ in range(args.frames):
            image = camera.read_frame_raw(timeout_ms=2000)
            if image is not None:
                frame, got = image, got + 1
        elapsed = time.perf_counter() - start
        if frame is None:
            print("no frame received")
            return 1
        print(f"{got}/{args.frames} frames, {got / elapsed:.1f} fps, shape {frame.shape}, dtype {frame.dtype}, "
              f"min {frame.min()} max {frame.max()} mean {frame.mean():.1f}, temperature {camera.get_temperature()}")
        if args.grab:
            np.save(args.grab, frame)
            print("saved", args.grab)
        if args.image:
            if not cv2.imwrite(args.image, frame):
                print("cv2 could not write", args.image)
                return 1
            print("saved", args.image)
        return 0
    finally:
        camera.close()


if __name__ == "__main__":
    raise SystemExit(main())
