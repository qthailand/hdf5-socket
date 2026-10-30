# -*- coding: utf-8 -*-
"""Make reflectance factors for one or more data sources from a white-reference recording.

The white reference goes through the same steps as the data (dark + hot pixel, flip, Smile,
binning from simulation.json / the file's metadata), is averaged, and the factors
``R_ref(λ) / white`` are written to ``reflectance/<data source>.npy`` (+ ``.wavelength.npy``),
where the simulator applies them in real time as the last step before the model.

    # raw white recording (HDF5 from Keep Spectrogram), Spectralon only in slit rows 120..430
    python tools/make_reflectance.py data/white_40.h5 --for run_A run_B \\
        --table reflectance/Spectralon_ref-040.xlsx --rows 120:430 --kind row

    # binned white cube (ENVI) with its dark cube of the same bands
    python tools/make_reflectance.py reflectance/spectralon-40.bip.hdr --for run_A \\
        --table reflectance/Spectralon_ref-040.xlsx --dark-cube reflectance/dark-40.bip.hdr

    # placeholder until a real white exists: any recording, whole cube, no dark required
    python tools/make_reflectance.py data/spectrogram02.h5 --for spectrogram spectrogram02 \
        --table reflectance/Spectralon_ref-040.xlsx --placeholder

    --dry-run prints the report without writing anything.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reflectance_builder import (crop_rows_after_chain, embed_slit_crop, factors_from_white,  # noqa: E402
                                 find_slit_crop, quality, reference_reflectance, save_profile, white_mean)
from sim_config import build_plan, load_config  # noqa: E402
from sources import open_sources  # noqa: E402


def _range(text):
    if not text:
        return None
    a, _, b = text.partition(':')
    return int(a or 0), int(b)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('white', help='white-reference recording: HDF5 (raw) or ENVI header/data file')
    p.add_argument('--for', dest='targets', nargs='+', required=True, metavar='NAME',
                   help='data source name(s) the factors belong to (file name without extension)')
    p.add_argument('--table', required=True, help='reference reflectance table (.xlsx: nm, R) of the target')
    p.add_argument('--kind', choices=('band', 'row'), default='band',
                   help='band: (bands,) from the ROI mean; row: (slit rows, bands) as hsi row-wise')
    p.add_argument('--rows', default=None, help='slit rows FIRST:LAST+1 where the target is (default all)')
    p.add_argument('--frames', default=None, help='scan frames FIRST:LAST+1 to average (default all)')
    p.add_argument('--avg-window', type=int, default=20, help='row window for --kind row (hsi)')
    crop = p.add_argument_group('slit-cropped raw white (fewer slit rows than the sensor frame)')
    crop.add_argument('--slit-offset', type=int, default=None,
                      help='raw sensor row where the crop starts (before flip)')
    crop.add_argument('--find-in', default=None, metavar='RECORDING',
                      help='HDF5 recording the crop was cut from: finds --slit-offset (and the frames) by exact match')
    crop.add_argument('--full-rows', type=int, default=None,
                      help='sensor frame height (default: the --find-in recording, else the dark/Smile size)')
    dark = p.add_mutually_exclusive_group()
    dark.add_argument('--dark-cube', default=None, help='binned white only: dark cube with the same bands')
    dark.add_argument('--already-dark-corrected', action='store_true',
                      help='binned white only: the cube was dark-corrected before export')
    p.add_argument('--config', default=None, help='simulation.json of the data (default beside the tools)')
    p.add_argument('--set', action='append', default=[], metavar='SECTION.KEY=VALUE',
                   help='override the data configuration, e.g. --set corrections.dark=auto')
    p.add_argument('--out', default=None, help='output folder (default corrections.reflectance_dir)')
    p.add_argument('--placeholder', action='store_true',
                   help='stand-in factors (no dark required, marked placeholder in <name>.json and the GUI) '
                        'so the Reflectance step can run until a real white reference is recorded')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    overrides = {}
    for item in args.set:
        key, _, text = item.partition('=')
        section, _, name = key.partition('.')
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = text
        overrides.setdefault(section, {})[name] = value
    config = load_config(args.config, **overrides)

    white_src = open_sources(Path(args.white).resolve(), stage=config.source.stage,
                             scan_axis=config.source.envi_scan_axis)[0]
    crop_info = None
    if white_src.stage == 'raw' and (args.slit_offset is not None or args.find_in):
        offset, full_rows = args.slit_offset, args.full_rows
        if args.find_in:
            import h5py
            with h5py.File(args.find_in, 'r') as f:
                full_rows = full_rows or int(f['spectrogram'].shape[1])
            if offset is None:
                hit = find_slit_crop(white_src.frames, args.find_in)
                if hit is None:
                    print(f"ERROR: {white_src.name} was not found in {args.find_in}", file=sys.stderr)
                    return 2
                print(f"{white_src.name} = frames {hit[0]}:{hit[0] + white_src.n_frames}, "
                      f"slit rows {hit[1]}:{hit[1] + white_src.frame_shape[0]} of {Path(args.find_in).name}",
                      file=sys.stderr)
                offset = hit[1]
        if full_rows is None:
            print("ERROR: give --full-rows or --find-in for a slit crop", file=sys.stderr)
            return 2
        crop_rows = white_src.frame_shape[0]
        white_src = embed_slit_crop(white_src, offset, full_rows)
        crop_info = (offset, crop_rows, full_rows)
    plan = build_plan(config, white_src)
    if crop_info and args.rows is None:           # average over the crop's rows after the chain
        offset, crop_rows, full_rows = crop_info
        args.rows = '%d:%d' % crop_rows_after_chain(offset, crop_rows, full_rows, bool(plan.steps['flip']['on']))
        print(f"slit crop {offset}:{offset + crop_rows} of {full_rows} rows -> ROI rows {args.rows} after the chain",
              file=sys.stderr)
    dark_cube = False if args.already_dark_corrected else None
    if args.dark_cube:
        dark_cube = open_sources(Path(args.dark_cube).resolve(), scan_axis=config.source.envi_scan_axis)[0]
    rows, frames = _range(args.rows), _range(args.frames)
    try:
        white = white_mean(white_src, plan, slice(*frames) if frames else None, dark_cube=dark_cube,
                           require_dark=not args.placeholder)
        reflectance = reference_reflectance(args.table, white.wavelengths)
        factors = factors_from_white(white.mean, reflectance, args.kind, rows, args.avg_window)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    full_scale = white_src.metadata.get('full_scale') or white_src.metadata.get('data full scale')
    report = quality(white, factors, reflectance, float(full_scale) if full_scale else None, rows)
    report.update(white=white_src.name, table=Path(args.table).name, kind=args.kind,
                  factor_shape=list(factors.shape), targets=args.targets)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report['weak_bands_nm']:
        print(f"WARNING: {len(report['weak_bands_nm'])} bands have < 2 % of the peak white signal; "
              "their factors amplify noise", file=sys.stderr)
    if args.dry_run:
        return 0
    folder = Path(args.out) if args.out else (config.resolve(config.corrections.reflectance_dir)
                                             or ROOT / 'reflectance')
    info = {'placeholder': bool(args.placeholder), 'white': str(Path(args.white).name), 'table': Path(args.table).name,
            'kind': args.kind, 'rows': list(rows) if rows else 'all', 'frames': list(frames) if frames else 'all',
            'frames_averaged': white.n_frames, 'steps': {k: v for k, v in white.steps.items()}}
    if crop_info:
        info['slit_crop'] = {'raw_rows': [crop_info[0], crop_info[0] + crop_info[1]], 'sensor_rows': crop_info[2]}
    for path in save_profile(folder, args.targets, factors, white.wavelengths, info):
        print(f"wrote {path} (+ {path.stem}.wavelength.npy, {path.stem}.json)"
              + ("  [PLACEHOLDER]" if args.placeholder else ""))
    return 0


if __name__ == '__main__':
    sys.exit(main())
