# -*- coding: utf-8 -*-
"""Keep the processing code identical to the HSI acquisition project.

Dark, hot pixel, flip, Smile and binning must run the same algorithm as
``C:\\ClassificationPipeLine\\hsi``. These files are byte-identical copies of hsi; only the
small package ``__init__`` files are local.

    python tools/sync_hsi.py            # check: list files that differ from hsi (exit 1 if any)
    python tools/sync_hsi.py --update   # copy the current hsi files over the local copies
    python tools/sync_hsi.py --hsi D:\\path\\to\\hsi
"""
import argparse
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HSI = ROOT.parent / 'hsi'

#: Paths relative to ``modules/`` in both projects.
VENDORED = (
    'preprocessing/__init__.py', 'preprocessing/cpu.py', 'preprocessing/cupy_cuda.py',
    'preprocessing/opencv_cuda.py', 'preprocessing/maps.py', 'preprocessing/selftest.py',
    'preprocessing/timing.py',
    'compute_runtime/__init__.py',
    'dark/core.py', 'dark/library.py', 'dark/selection.py',
    'smile/calibration.py', 'smile/core.py', 'smile/surface.py', 'smile/runtime.py',
    'spectral_processing/binning.py',
    'radiometric/core.py', 'cube_io/envi.py', 'cube_io/writers.py',
)


def compare(hsi: Path) -> list[tuple[str, str]]:
    """``[(path, problem)]`` for every vendored file that is not identical to hsi."""
    problems = []
    for rel in VENDORED:
        theirs, ours = hsi / 'modules' / rel, ROOT / 'modules' / rel
        if not theirs.is_file():
            problems.append((rel, 'missing in hsi'))
        elif not ours.is_file():
            problems.append((rel, 'missing here'))
        elif not filecmp.cmp(theirs, ours, shallow=False):
            problems.append((rel, 'differs'))
    return problems


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--hsi', default=str(DEFAULT_HSI), help=f'hsi project folder (default {DEFAULT_HSI})')
    p.add_argument('--update', action='store_true', help='copy hsi files over the local copies')
    args = p.parse_args(argv)
    hsi = Path(args.hsi)
    if not (hsi / 'modules').is_dir():
        print(f"hsi project not found: {hsi}")
        return 2
    problems = compare(hsi)
    if args.update:
        for rel, problem in problems:
            if problem == 'missing in hsi':
                print(f"  skip {rel}: {problem}")
                continue
            (ROOT / 'modules' / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(hsi / 'modules' / rel, ROOT / 'modules' / rel)
            print(f"  updated {rel}")
        problems = compare(hsi)
    for rel, problem in problems:
        print(f"  {problem}: modules/{rel}")
    print(f"{len(VENDORED) - len(problems)}/{len(VENDORED)} files identical to {hsi}")
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
