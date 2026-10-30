# -*- coding: utf-8 -*-
"""Record the training cube's band wavelengths in a model artifact.

The simulator then feeds the model by wavelength instead of by band index, and refuses an
input whose calibration does not match the training data.

    python tools/add_band_wavelengths.py rf_coco_model.joblib dataset/dataset.bip.hdr

The weights are not touched; ``band_wavelengths``, ``n_bands`` and ``training_header`` are
added and the artifact is written back (``--out`` writes a new file instead).
"""
import argparse
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sources import parse_envi_header  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model')
    p.add_argument('header', help='ENVI header of the cube the model was trained on')
    p.add_argument('--out', default=None)
    args = p.parse_args(argv)

    _, header = parse_envi_header(args.header)
    wavelengths = header['wavelength']
    if wavelengths is None:
        sys.exit(f"{args.header} has no wavelength list")
    artifact = joblib.load(args.model)
    selected = artifact['selected_bands']
    if max(selected) >= wavelengths.size:
        sys.exit(f"model uses band {max(selected)}, header has {wavelengths.size} bands")
    artifact['band_wavelengths'] = wavelengths
    artifact['n_bands'] = int(wavelengths.size)
    artifact['training_header'] = Path(args.header).name
    out = args.out or args.model
    joblib.dump(artifact, out)
    sel = wavelengths[selected]
    print(f"{out}: {wavelengths.size} band wavelengths {wavelengths[0]:.2f}-{wavelengths[-1]:.2f} nm; "
          f"model uses {len(selected)} bands {sel.min():.2f}-{sel.max():.2f} nm")


if __name__ == '__main__':
    main()
