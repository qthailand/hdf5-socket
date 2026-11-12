# -*- coding: utf-8 -*-
"""Check that the model's per-pixel confidence is calibrated, on labelled pixels.

Loads the artifact with ``inference_engine.PushbroomLineClassifier`` (the simulator's loader and
wavelength binding; ``predict_confidence`` incl. the artifact's ``temperature``), the
labelled ENVI cube and its COCO polygons, and prints a reliability table: per confidence bin the
pixel count, mean confidence and how many of those pixels are actually right, plus the ECE
(expected calibration error) and per-class accuracy.

    python tools/check_calibration.py                                   # models/sgd_first_reflectance.joblib
    python tools/check_calibration.py rf_model.joblib --hdr dataset/first.hdr --labels dataset/first.bip.json
    python tools/check_calibration.py --temperature 1.0                 # the uncalibrated model

By default only the held-out pixels are used: ``train_sgd_coco.py`` splits the labelled pixels
with ``train_test_split(test_size=0.2, random_state=42, stratify=y)``, which depends only on the
labels, so the same split is reproduced here (``--split train`` / ``all`` for the others). The
held-out pixels come from the same recording as the training pixels (neighbours of them), so the
numbers are an upper bound for a new recording; ``--hdr`` / ``--labels`` of another labelled
recording check all of its pixels.
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import spectral as spy
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference_engine import PushbroomLineClassifier  # noqa: E402
from tools.train_sgd_coco import load_coco_masks  # noqa: E402


def reliability(confidence, correct, edges):
    """``(rows, ece)``: per non-empty bin ``(lo, hi, n, mean confidence, accuracy)``."""
    which = np.clip(np.searchsorted(edges, confidence, side='right') - 1, 0, edges.size - 2)
    rows, ece = [], 0.0
    for b in range(edges.size - 1):
        sel = which == b
        n = int(sel.sum())
        if n:
            conf, acc = float(confidence[sel].mean()), float(correct[sel].mean())
            rows.append((edges[b], edges[b + 1], n, conf, acc))
            ece += n / confidence.size * abs(acc - conf)
    return rows, ece


def _existing(path):
    for p in (Path(path), ROOT / path):
        if p.exists():
            return p
    sys.exit(f"{path} not found")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model', nargs='?', default='models/sgd_first_reflectance.joblib')
    p.add_argument('--hdr', help="labelled cube (default: the artifact's training_header)")
    p.add_argument('--labels', help='COCO polygons of that cube (default: <cube data file>.json)')
    p.add_argument('--split', choices=('test', 'train', 'all'), default='test',
                   help='pixels to check: the training script held-out 20%% (default), its 80%%, or all')
    p.add_argument('--temperature', type=float, help="override the artifact's temperature (1.0 = uncalibrated)")
    p.add_argument('--min-confidence', type=float, default=0.5,
                   help='report the accuracy below and at or above this confidence')
    p.add_argument('--bin', type=float, default=0.1, help='width of the confidence bins')
    args = p.parse_args(argv)

    model_path = _existing(args.model)
    clf = PushbroomLineClassifier(model_path)
    if args.temperature is not None:
        if not args.temperature > 0:
            sys.exit('--temperature must be > 0')
        clf.temperature = args.temperature
    training_header = joblib.load(model_path).get('training_header')
    if (args.hdr or training_header) is None:
        sys.exit('the artifact has no training_header: pass --hdr')
    hdr = _existing(args.hdr or training_header)
    own_cube = training_header is None or Path(training_header).name == hdr.name
    img = spy.open_image(str(hdr))
    data_file = Path(img.filename)
    labels = _existing(args.labels or data_file.with_name(data_file.name + '.json'))
    mask, names = load_coco_masks(str(labels), img.nrows, img.ncols)
    names = {**names, **clf.category_names}

    # Same pixel order and split as train_sgd_coco.py (another cube: every pixel is unseen).
    y_all = mask.reshape(-1)
    valid = np.where(y_all > 0)[0]
    y = y_all[valid]
    if own_cube and args.split != 'all':
        idx_train, idx_test = train_test_split(valid, test_size=0.2, random_state=42, stratify=y)
        valid = np.sort(idx_test if args.split == 'test' else idx_train)
        y = y_all[valid]

    cube = img.open_memmap()                                    # (lines, samples, bands)
    wavelengths = getattr(img.bands, 'centers', None)
    columns, binding = clf.bind(wavelengths, img.nbands)
    pred = np.empty(valid.size, np.uint8)
    conf = np.empty(valid.size, np.float32)
    rows, cols = np.divmod(valid, img.ncols)
    for s in range(0, valid.size, 100_000):
        part = slice(s, s + 100_000)
        pixels = np.asarray(cube[rows[part], cols[part]], dtype=np.float32)
        pred[part], conf[part] = clf.predict_confidence(pixels, columns)
    correct = pred == y

    split = {'test': "held-out 20 % (not used to fit the weights" +
                     ("; the temperature was fitted on them)" if clf.temperature != 1.0 else ")"),
             'train': "TRAINING pixels (the model has seen them: optimistic)",
             'all': "all labelled pixels (80 % of them are training pixels: optimistic)"}[args.split]
    if not own_cube:
        split = f"all labelled pixels of a cube the model was not trained on ({training_header})"
    print(f"Model       {model_path} (temperature {clf.temperature:g})")
    print(f"Cube        {hdr}  labels {labels}")
    print(f"Binding     {binding}")
    print(f"Pixels      {valid.size} {split}")
    print(f"Accuracy    {correct.mean():.4f}   mean confidence {conf.mean():.4f}   "
          f"({'over' if conf.mean() > correct.mean() else 'under'}-confident by "
          f"{abs(conf.mean() - correct.mean()):.4f})")

    edges = np.round(np.arange(0.0, 1.0 + args.bin / 2, args.bin), 6)
    table, ece = reliability(conf, correct, edges)
    print(f"\n{'confidence':>11} {'pixels':>9} {'share':>7} {'mean conf':>10} {'accuracy':>9} {'gap':>8}")
    for lo, hi, n, c, a in table:
        print(f"{lo:5.2f}-{hi:4.2f} {n:9d} {n / valid.size:7.1%} {c:10.4f} {a:9.4f} {a - c:+8.4f}")
    print(f"ECE {ece:.4f}   (gap = accuracy - mean confidence: > 0 under-confident, < 0 over-confident)")

    low = conf < args.min_confidence
    print(f"\nBelow min_confidence {args.min_confidence:g}: {low.mean():.1%} of pixels, "
          f"accuracy {correct[low].mean() if low.any() else float('nan'):.4f}; "
          f"at or above: accuracy {correct[~low].mean() if (~low).any() else float('nan'):.4f}")

    print(f"\n{'class':>12} {'labelled':>9} {'accuracy':>9} {'assigned':>9} {'precision':>10} {'mean conf':>10}")
    for c in np.unique(np.concatenate([y, pred])):
        true_c, pred_c = y == c, pred == c
        recall = correct[true_c].mean() if true_c.any() else float('nan')
        precision = correct[pred_c].mean() if pred_c.any() else float('nan')
        mean_conf = conf[pred_c].mean() if pred_c.any() else float('nan')
        print(f"{names.get(int(c), c)!s:>12} {int(true_c.sum()):9d} {recall:9.4f} {int(pred_c.sum()):9d} "
              f"{precision:10.4f} {mean_conf:10.4f}")
    print("accuracy = right / labelled as the class; precision = right / assigned to the class, "
          "compare it with the mean confidence")


if __name__ == '__main__':
    main()
