# -*- coding: utf-8 -*-
"""Move one camera's calibration from the HSI layout into its profile ``profiles/<model>-<serial>/``.

    calibration/dark_frames/<backend>/*   -> profiles/<id>/dark/
    calibration/smile_maps/<backend>/*    -> profiles/<id>/smile/        (history/ too)
    calibration/wavelength/<backend>/*    -> profiles/<id>/wavelength/   (history/ too)
    reflectance/live_<backend>.npy (+ .wavelength.npy, .json)
                                          -> profiles/<id>/reflectance/refl_b<bin>_g<gain>.npy

The camera is named by a recording it made (its ``camera_model`` / ``camera_serial`` /
``camera_backend`` attributes) or by ``--backend --model --serial``. The calibration files carry
no serial, so only move them for the camera they were made with. The old reflectance profile is
bound to ``--bin`` / ``--gain`` / ``--exposure`` (default: the recording's values).

    python tools/migrate_profiles.py --from-recording data/poa.h5            # dry run: prints the plan
    python tools/migrate_profiles.py --from-recording data/poa.h5 --apply    # moves the files
"""
import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim_config import CALIBRATION_KINDS, camera_profile_name, gain_key, load_config, profiles_root  # noqa: E402


def recording_attrs(path) -> dict:
    import h5py
    with h5py.File(path, 'r') as f:
        attrs = {k: f.attrs[k] for k in f.attrs}
    return {k: (v.decode() if isinstance(v, bytes) else v.item() if getattr(v, 'size', 0) == 1 else v)
            for k, v in attrs.items()}


def plan_moves(config, backend: str, camera_id: str, bin_: int | None, gain: float | None) -> list[tuple[Path, Path]]:
    profile = profiles_root(config) / camera_id
    moves = []
    for kind, legacy in CALIBRATION_KINDS.items():
        folder = config.calibration_dir / legacy / backend
        if folder.is_dir():
            for item in sorted(folder.rglob('*')):
                if item.is_file():
                    moves.append((item, profile / kind / item.relative_to(folder)))
    refl = (config.resolve(config.corrections.reflectance_dir) or ROOT / 'reflectance') / f"live_{backend}.npy"
    if refl.is_file() and bin_ is not None and gain is not None:
        stem = f"refl_b{int(bin_)}_g{gain_key(gain)}"
        for suffix in ('.npy', '.wavelength.npy', '.json'):
            old = refl.with_name(refl.stem + suffix)
            if old.is_file():
                moves.append((old, profile / 'reflectance' / f"{stem}{suffix}"))
    return moves


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--from-recording', default=None, help='HDF5 recording of the camera (names it)')
    p.add_argument('--backend', default=None)
    p.add_argument('--model', default=None)
    p.add_argument('--serial', default=None)
    p.add_argument('--bin', type=int, default=None, help='bin the old live_<backend> reflectance was made at')
    p.add_argument('--gain', type=float, default=None)
    p.add_argument('--exposure', type=float, default=None)
    p.add_argument('--config', default=None)
    p.add_argument('--apply', action='store_true', help='move the files (default: only print the plan)')
    args = p.parse_args(argv)

    attrs = recording_attrs(args.from_recording) if args.from_recording else {}
    backend = args.backend or attrs.get('camera_backend')
    model = args.model or attrs.get('camera_model')
    serial = args.serial or attrs.get('camera_serial')
    camera_id = camera_profile_name(model, serial)
    if not backend or not camera_id:
        print("ERROR: give --from-recording, or --backend --model --serial", file=sys.stderr)
        return 2
    bin_ = args.bin if args.bin is not None else attrs.get('camera_bin')
    gain = args.gain if args.gain is not None else attrs.get('gain')
    exposure = args.exposure if args.exposure is not None else attrs.get('exposure_ms')
    config = load_config(args.config)
    moves = plan_moves(config, backend, camera_id, bin_, gain)
    profile = profiles_root(config) / camera_id
    print(f"camera {camera_id} ({backend}) -> {profile}")
    for old, new in moves:
        print(f"  {old.relative_to(config.base_dir)}  ->  {new.relative_to(config.base_dir)}")
    clash = [new for _, new in moves if new.exists()]
    if clash:
        print(f"ERROR: {len(clash)} target(s) exist already, e.g. {clash[0]}", file=sys.stderr)
        return 2
    if not args.apply:
        print("dry run: add --apply to move")
        return 0
    for old, new in moves:
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old), str(new))
    for npy in [new for _, new in moves if new.parent.name == 'reflectance' and new.name.endswith('.npy')
                and not new.name.endswith('.wavelength.npy')]:
        new = npy.with_suffix('.json')                 # bind it, also when the old one had no JSON
        info = json.loads(new.read_text(encoding='utf-8')) if new.is_file() else {}
        info.update(camera_bin=int(bin_), gain=float(gain), camera_id=camera_id, camera_backend=backend)
        if exposure is not None:
            info.setdefault('exposure_ms', float(exposure))
        info['migrated'] = f"from reflectance/live_{backend}.* by tools/migrate_profiles.py"
        new.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    record = profile / 'profile.json'
    if not record.exists():
        record.write_text(json.dumps({
            'camera_id': camera_id, 'camera_backend': backend, 'camera_model': model, 'camera_serial': serial,
            'created': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
            'migrated_from': f"calibration/*/{backend}/"}, indent=2, ensure_ascii=False), encoding='utf-8')
    for legacy in CALIBRATION_KINDS.values():         # drop the emptied backend folders
        folder = config.calibration_dir / legacy / backend
        for sub in sorted((d for d in folder.rglob('*') if d.is_dir()), reverse=True) if folder.is_dir() else []:
            if not any(sub.iterdir()):
                sub.rmdir()
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
    print(f"moved {len(moves)} file(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
