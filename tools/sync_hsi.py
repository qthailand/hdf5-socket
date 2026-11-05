# -*- coding: utf-8 -*-
"""Keep the processing and camera code identical to the HSI acquisition project.

Dark, hot pixel, flip, Smile and binning must run the same algorithm as
``C:\\ClassificationPipeLine\\hsi``, and the live camera source drives the camera through hsi's
``modules/camera`` backends. These files are byte-identical copies of hsi; only the small
package ``__init__`` files are local.

The vendor SDK wrappers of the camera backends (``modules/camera/poa``, ``modules/camera/ximea``:
ctypes wrappers plus native DLLs, the XIMEA one 110 MB) are not in git here, as in hsi. ``--update``
copies them from hsi (without ``__pycache__`` and the 32-bit XIMEA libraries); a missing SDK is
only reported, it matters only when that camera backend is used.

    python tools/sync_hsi.py                          # check: list files that differ from hsi (exit 1 if any)
    python tools/sync_hsi.py --update                 # copy the current hsi files over the local copies
    python tools/sync_hsi.py --update --only camera   # only the paths under modules/camera
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
    # Live camera source (camera_source.py). Not copied: widget.py and pixel_pitch.py (PyQt5
    # widgets; pixel_pitch needs hsi's modules.contracts) - the local __init__ does not export them.
    'camera/base.py', 'camera/config.py', 'camera/factory.py', 'camera/__main__.py',
    'camera/genicam.py', 'camera/poa_camera.py', 'camera/ximea_camera.py',
    # Focus score of the GUI's focus panel (widgets/focus_panel.py); hsi's widget.py is not copied.
    'sharpness/core.py',
)

#: Vendor SDK folders (relative to ``modules/``): copied by ``--update``, not tracked by git.
SDK_FOLDERS = ('camera/poa', 'camera/ximea')
_SDK_SKIP = ('__pycache__', '32bit')        # folder names never copied (bytecode, 32-bit XIMEA DLLs)


def _selected(rel: str, only: str | None) -> bool:
    rel = rel.lower()
    return not only or rel == only or rel.startswith(only.rstrip('/') + '/')


def normalise_only(value: str | None) -> str | None:
    """``--only`` as a lower-case path under ``modules/`` (``modules\\Camera\\`` -> ``camera``)."""
    if not value:
        return None
    only = value.replace('\\', '/').strip('/').lower()
    return only[len('modules/'):] if only.startswith('modules/') else only


def only_matches(only: str | None) -> bool:
    """True if ``only`` selects at least one vendored file or SDK folder (or no filter is given)."""
    return (not only or any(_selected(rel, only) for rel in VENDORED)
            or any(_selected(folder, only) or only.startswith(folder + '/') for folder in SDK_FOLDERS))


def compare(hsi: Path, only: str | None = None) -> list[tuple[str, str]]:
    """``[(path, problem)]`` for every vendored file that is not identical to hsi.

    ``only`` limits the check to the paths under that folder of ``modules/`` (e.g. ``'camera'``).
    """
    problems = []
    for rel in VENDORED:
        if not _selected(rel, only):
            continue
        theirs, ours = hsi / 'modules' / rel, ROOT / 'modules' / rel
        if not theirs.is_file():
            problems.append((rel, 'missing in hsi'))
        elif not ours.is_file():
            problems.append((rel, 'missing here'))
        elif not filecmp.cmp(theirs, ours, shallow=False):
            problems.append((rel, 'differs'))
    return problems


def sdk_files(hsi: Path, only: str | None = None) -> list[str]:
    """Files of the SDK folders in hsi, relative to ``modules/`` (bytecode and 32-bit libraries left out)."""
    files = []
    for folder in SDK_FOLDERS:
        base = hsi / 'modules' / folder
        if not base.is_dir():
            continue
        for path in sorted(base.rglob('*')):
            rel = path.relative_to(hsi / 'modules').as_posix()
            if path.is_file() and not set(path.relative_to(base).parts) & set(_SDK_SKIP) and _selected(rel, only):
                files.append(rel)
    return files


def compare_sdk(hsi: Path, only: str | None = None) -> tuple[list[tuple[str, str]], list[str]]:
    """``(problems, missing)``: SDK files here that differ from hsi, and SDK folders not installed here.

    A missing SDK is not a problem (that camera backend is simply unavailable); a different one is.
    """
    problems, missing = [], []
    files = sdk_files(hsi, only)
    for folder in SDK_FOLDERS:
        if _selected(folder, only) or (only or '').startswith(folder + '/'):
            if not (ROOT / 'modules' / folder).is_dir():
                missing.append(folder)
    for rel in files:
        ours = ROOT / 'modules' / rel
        if ours.is_file() and not filecmp.cmp(hsi / 'modules' / rel, ours, shallow=False):
            problems.append((rel, 'differs'))
        elif not ours.is_file() and not any(rel.startswith(m + '/') for m in missing):
            problems.append((rel, 'missing here'))
    return problems, missing


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--hsi', default=str(DEFAULT_HSI), help=f'hsi project folder (default {DEFAULT_HSI})')
    p.add_argument('--update', action='store_true', help='copy hsi files over the local copies')
    p.add_argument('--only', default=None, metavar='FOLDER',
                   help="only the paths under modules/FOLDER (e.g. 'camera'), for check and update")
    args = p.parse_args(argv)
    hsi = Path(args.hsi)
    if not (hsi / 'modules').is_dir():
        print(f"hsi project not found: {hsi}")
        return 2
    only = normalise_only(args.only)
    if not only_matches(only):
        folders = sorted({rel.split('/')[0] for rel in VENDORED})
        print(f"--only {args.only!r} matches nothing under modules/ (folders: {', '.join(folders)}; "
              f"SDKs: {', '.join(SDK_FOLDERS)})")
        return 2
    problems = compare(hsi, only)
    sdk_problems, sdk_missing = compare_sdk(hsi, only)
    if args.update:
        copies = [rel for rel, problem in problems if problem != 'missing in hsi']
        copies += [rel for rel, _ in sdk_problems]
        copies += [rel for rel in sdk_files(hsi, only) if any(rel.startswith(m + '/') for m in sdk_missing)]
        for rel, problem in problems:
            if problem == 'missing in hsi':
                print(f"  skip {rel}: {problem}")
        for rel in copies:
            (ROOT / 'modules' / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(hsi / 'modules' / rel, ROOT / 'modules' / rel)
            print(f"  updated {rel}")
        problems = compare(hsi, only)
        sdk_problems, sdk_missing = compare_sdk(hsi, only)
    for rel, problem in problems + sdk_problems:
        print(f"  {problem}: modules/{rel}")
    for folder in sdk_missing:
        print(f"  not installed: modules/{folder} (camera SDK, not in git; copy it with --update --only {folder})")
    checked = [rel for rel in VENDORED if _selected(rel, only)]
    print(f"{len(checked) - len(problems)}/{len(checked)} files identical to {hsi}"
          + (f" (only modules/{only})" if only else ""))
    return 1 if problems or sdk_problems else 0


if __name__ == '__main__':
    sys.exit(main())
