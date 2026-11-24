# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:55:00 2026

Smart Feature Selection (SelectKBest + Scaler + SGDClassifier) for High Accuracy.

By default it trains on EVERY labelled cube in ``dataset/``: each ENVI header ``<name>.hdr`` that
has COCO polygons next to it (``<name>.json`` or ``<name>.bip.json``). The labelled pixels of all
cubes are pooled and split once (80 % fit / 20 % held out). Class ids are matched by name across
the cubes, so every cube may label only some of the classes. The cubes must have the same bands
(wavelengths) and the same units (counts or reflectance).

    python tools/train_sgd_coco.py                               # every labelled cube in dataset/
    python tools/train_sgd_coco.py --dataset other_folder --out models/sgd_other.joblib
    python tools/train_sgd_coco.py --hdr dataset/first.hdr       # one cube only
    python tools/train_sgd_coco.py --no-report                   # numbers only, no figures

Unless ``--no-report``, the figures of ``tools/sgd_report.py`` (confusion matrix, per-class metrics,
spectra + kept bands, weights, confidence calibration, LDA scatter, whole-cube maps) and an
``index.html`` are written to ``<out>_report/`` (``--report`` picks another folder).

@author: TSC
"""
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import cv2
import numpy as np
import spectral as spy
from scipy.optimize import minimize_scalar
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = 'dataset'
DEFAULT_OUT = 'models/sgd_dataset.joblib'
WAVELENGTH_TOLERANCE_NM = 0.01


def load_coco_masks(json_path: str, height: int, width: int) -> tuple[np.ndarray, dict[int, str]]:
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    label_mask = np.zeros((height, width), dtype=np.int32)
    category_names = {cat['id']: cat['name'] for cat in coco_data['categories']}

    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        for seg in ann['segmentation']:
            pts = np.array(seg, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(label_mask, [pts], color=cat_id)

    return label_mask, category_names


def ovr_probabilities(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """sklearn's ``predict_proba`` of a one-vs-rest ``log_loss`` model, from ``scores / temperature``."""
    p = 1.0 / (1.0 + np.exp(-np.clip(scores / temperature, -80.0, 80.0)))
    return p / p.sum(axis=1, keepdims=True)


def fit_temperature(scores: np.ndarray, y_index: np.ndarray) -> float:
    """Temperature minimising the log loss of ``ovr_probabilities(scores, T)`` on held-out pixels.

    ``y_index`` is the column of each pixel's true class. T < 1 sharpens an under-confident
    model, T > 1 softens an over-confident one; the winning class never changes.
    """
    rows = np.arange(y_index.size)

    def log_loss(log_t):
        p = ovr_probabilities(scores, np.exp(log_t))[rows, y_index]
        return -np.log(np.maximum(p, 1e-12)).mean()

    result = minimize_scalar(log_loss, bounds=(np.log(0.01), np.log(100.0)), method='bounded')
    return float(np.exp(result.x))


def _resolve(path) -> Path:
    """``path`` as given, else beside the project (so the script also runs from another folder)."""
    p = Path(path)
    return p if p.exists() else ROOT / p


def label_path_for(hdr: Path) -> Path | None:
    """The COCO file of the cube ``hdr``: ``<name>.json``, ``<name>.bip.json`` (``first.hdr`` ->
    ``first.bip.json``) or, for ``<name>.bip.hdr``, ``<name>.json``; ``None`` if there is none."""
    stem = hdr.name[:-len('.hdr')] if hdr.name.lower().endswith('.hdr') else hdr.stem
    base = stem[:-len('.bip')] if stem.lower().endswith('.bip') else stem
    for name in dict.fromkeys((stem + '.json', base + '.json', base + '.bip.json')):
        if (hdr.parent / name).is_file():
            return hdr.parent / name
    return None


def find_cubes(dataset_dir) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """``(cubes, skipped)``: every ``*.hdr`` of the folder with its COCO file, and the headers without."""
    folder = _resolve(dataset_dir)
    if not folder.is_dir():
        raise ValueError(f"dataset folder {dataset_dir} not found")
    cubes, skipped = [], []
    for hdr in sorted(folder.glob('*.hdr')):
        labels = label_path_for(hdr)
        if labels is None:
            skipped.append(hdr)
        else:
            cubes.append((hdr, labels))
    return cubes, skipped


def _coco_image_sizes(json_path: Path) -> list[tuple[int, int]]:
    with open(json_path, 'r', encoding='utf-8') as f:
        images = json.load(f).get('images', [])
    return [(int(i['width']), int(i['height'])) for i in images if i.get('width') and i.get('height')]


def merge_categories(per_cube: list[dict[int, str]]) -> tuple[dict[int, str], list[np.ndarray]]:
    """One class table for all cubes, matched by name (case and surrounding spaces ignored).

    ``(category_names, luts)``: ``luts[i][local id]`` is the merged id of cube ``i`` (0 for an
    unknown id). The ids of the cubes are kept when they agree (same id = same name everywhere);
    otherwise the classes are numbered 1, 2, ... in the order they first appear.
    """
    key = lambda name: str(name).strip().casefold()          # noqa: E731
    id_of_key, key_of_id, display, keep = {}, {}, {}, True
    for names in per_cube:
        for cid, name in names.items():
            k = key(name)
            display.setdefault(k, str(name).strip())
            if id_of_key.setdefault(k, cid) != cid or key_of_id.setdefault(cid, k) != k:
                keep = False
    if keep:
        merged = dict(id_of_key)
    else:
        merged = {k: i + 1 for i, k in enumerate(display)}
    luts = []
    for names in per_cube:
        lut = np.zeros(max([0, *names]) + 1, np.int32)
        for cid, name in names.items():
            lut[cid] = merged[key(name)]
        luts.append(lut)
    return {merged[k]: display[k] for k in sorted(display, key=lambda k: merged[k])}, luts


@dataclass
class TrainingSet:
    X: np.ndarray                    # (pixels, bands) float32: the labelled pixels of every cube
    y: np.ndarray                    # (pixels,) merged class id
    source: np.ndarray               # (pixels,) index into ``cubes``
    category_names: dict[int, str]
    wavelengths: np.ndarray
    input_units: str                 # 'counts' or 'reflectance'
    cubes: list                      # [(hdr, labels)] as given
    position: np.ndarray | None = None   # (pixels,) flat index (line * samples + sample) in its cube


def load_training_set(cubes: list[tuple]) -> TrainingSet:
    """Labelled pixels of every ``(hdr, labels)`` cube, pooled, with one merged class table.

    Raises ``ValueError`` (naming the files) when a cube cannot be opened, its COCO image size
    is not the cube's, or the cubes differ in bands, wavelengths or units.
    """
    if not cubes:
        raise ValueError("no labelled cube to train on")
    opened, masks, names = [], [], []
    for hdr, labels in cubes:
        try:
            img = spy.open_image(str(hdr))
        except Exception as exc:                              # spectral raises several kinds
            raise ValueError(f"cannot open cube {hdr}: {exc}") from exc
        for width, height in _coco_image_sizes(Path(labels)):
            if (width, height) != (img.ncols, img.nrows):
                raise ValueError(f"{labels}: labels are drawn on {width} x {height} pixels but {hdr} is "
                                 f"{img.ncols} x {img.nrows} (samples x lines)")
        mask, cat = load_coco_masks(str(labels), img.nrows, img.ncols)
        opened.append(img)
        masks.append(mask)
        names.append(cat)
    category_names, luts = merge_categories(names)

    first = opened[0]
    wavelengths = getattr(first.bands, 'centers', None)
    if wavelengths is None:
        raise ValueError(f"{cubes[0][0]} has no wavelength list: the model could not be fed by wavelength")
    wavelengths = np.asarray(wavelengths, np.float32)
    parts_x, parts_y, parts_s, parts_p, units = [], [], [], [], []
    for i, ((hdr, _), img, mask, lut) in enumerate(zip(cubes, opened, masks, luts)):
        centers = getattr(img.bands, 'centers', None)
        if img.nbands != first.nbands or centers is None or not np.allclose(
                np.asarray(centers, np.float32), wavelengths, atol=WAVELENGTH_TOLERANCE_NM):
            raise ValueError(f"{hdr} has other bands than {cubes[0][0]} ({img.nbands} vs {first.nbands} "
                             f"bands, or other wavelengths): cubes must share the same bands")
        cube = img.open_memmap()                              # (lines, samples, bands)
        merged = lut[np.minimum(mask, lut.size - 1)] * (mask < lut.size)
        valid = np.flatnonzero(merged.reshape(-1))
        rows, cols = np.divmod(valid, img.ncols)
        parts_x.append(np.asarray(cube[rows, cols], dtype=np.float32))
        parts_y.append(merged.reshape(-1)[valid])
        parts_s.append(np.full(valid.size, i, np.int32))
        parts_p.append(valid.astype(np.int64))
        # Reflectance cubes (hsi radiometric output) are float32 with values around 0..1; raw/binned
        # cubes are counts. The simulator compares this with its Reflectance step.
        sample = np.asarray(cube[::8, ::8], dtype=np.float64)
        is_float = np.issubdtype(cube.dtype, np.floating)
        units.append('reflectance' if is_float and np.percentile(sample, 99) < 5 else 'counts')
    if len(set(units)) > 1:
        raise ValueError("cubes mix units: " + ", ".join(f"{Path(h).name}={u}" for (h, _), u in zip(cubes, units)))
    return TrainingSet(np.concatenate(parts_x), np.concatenate(parts_y).astype(np.int32),
                       np.concatenate(parts_s), category_names, wavelengths, units[0], list(cubes),
                       np.concatenate(parts_p))


def default_report_dir(model_output) -> Path:
    """``models/sgd_dataset.joblib`` -> ``models/sgd_dataset_report``."""
    out = Path(model_output)
    return out.with_name(out.stem + '_report')


def train_sgd_on_cubes(cubes: list[tuple], num_selected_bands: int = 30,
                       model_output: str = DEFAULT_OUT, report_dir=None) -> dict:
    """Train on the labelled pixels of all ``cubes``, write ``model_output`` and return the artifact.

    ``report_dir``: also write the figures of ``tools/sgd_report.py`` there (``None``: no report).
    """
    print(f"Loading {len(cubes)} labelled cube(s)...")
    data = load_training_set(cubes)
    category_names, y, num_bands = data.category_names, data.y, data.X.shape[1]
    if np.unique(y).size < 3:
        # A 2-class SGD keeps one weight vector (coef_ has a single row), which the fused
        # (bands x classes) artifact and the classifier cannot use.
        raise ValueError(f"need labelled pixels of at least 3 classes, found {np.unique(y).size} "
                         f"({', '.join(category_names[int(c)] for c in np.unique(y))}): label more "
                         f"classes or add more cubes to the folder")
    print(f"{'cube':<42}" + "".join(f"{category_names[c]:>10}" for c in category_names))
    for i, (hdr, _) in enumerate(data.cubes):
        counts = np.bincount(y[data.source == i], minlength=max(category_names) + 1)
        print(f"{Path(hdr).name:<42}" + "".join(f"{counts[c]:>10d}" for c in category_names))
    print(f"{'total':<42}" + "".join(f"{int((y == c).sum()):>10d}" for c in category_names))

    # Train / Test Split (one split of the pooled pixels; indices so the split can be reproduced
    # without the pixel values: tools/check_calibration.py)
    index = np.arange(y.size)
    idx_train, idx_test = train_test_split(index, test_size=0.2, random_state=42, stratify=y)
    X_train_raw, X_test_raw = data.X[idx_train], data.X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    # 1. Feature Selection ด้วย SelectKBest (ANOVA F-value) เลือกกระจายแบนด์เด่นจริง
    print(f"\nSelecting Best {num_selected_bands} Discriminative Bands using SelectKBest...")
    selector = SelectKBest(score_func=f_classif, k=num_selected_bands)
    X_train_selected = selector.fit_transform(X_train_raw, y_train)
    X_test_selected = selector.transform(X_test_raw)

    selected_bands = np.where(selector.get_support())[0]
    print(f"Selected Band Indices ({len(selected_bands)} bands): {selected_bands.tolist()}")

    # 2. Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    # 3. Train SGDClassifier
    print("\nTraining High-Speed SGDClassifier Model...")
    model = SGDClassifier(
        loss='log_loss',
        penalty='l2',
        alpha=1e-4,
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    target_names = [category_names[cid]
                    for cid in sorted(category_names.keys()) if cid in np.unique(y_test)]

    print("\nClassification Evaluation Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=target_names))
    if len(data.cubes) > 1:
        print("Held-out accuracy per cube:")
        for i, (hdr, _) in enumerate(data.cubes):
            of_cube = data.source[idx_test] == i
            if of_cube.any():
                print(f"  {Path(hdr).name:<42} {np.mean(y_pred[of_cube] == y_test[of_cube]):.4f} "
                      f"({int(of_cube.sum())} pixels)")

    # Temperature scaling: one scalar, fitted on the held-out pixels, so that the confidence (the
    # winning class's predict_proba) matches how often the class is right. The class is unchanged.
    test_scores = model.decision_function(X_test_scaled)
    temperature = fit_temperature(test_scores, np.searchsorted(model.classes_, y_test))
    print(f"Confidence temperature {temperature:.3f}: held-out accuracy {np.mean(y_pred == y_test):.4f}, "
          f"mean confidence {ovr_probabilities(test_scores).max(axis=1).mean():.4f} -> "
          f"{ovr_probabilities(test_scores, temperature).max(axis=1).mean():.4f} "
          f"(reliability table: tools/check_calibration.py)")

    # Save Artifact (คำนวณ Weights รวม Scaler สำหรับ Direct Matrix Mult)
    # W_scaled = W / scale, b_scaled = b - sum(W * mean / scale)
    W_orig = model.coef_                               # (n_classes, k_bands)
    b_orig = model.intercept_                          # (n_classes,)
    scale = scaler.scale_                              # (k_bands,)
    mean = scaler.mean_                                # (k_bands,)

    W_fused = (W_orig / scale).T                       # (k_bands, n_classes)
    b_fused = b_orig - np.dot(W_orig, (mean / scale))  # (n_classes,)

    artifact = {
        'model': model,
        'W_fused': np.ascontiguousarray(W_fused, dtype=np.float32),
        'b_fused': np.ascontiguousarray(b_fused, dtype=np.float32),
        'selected_bands': selected_bands,
        'category_names': category_names,
        'classes': model.classes_,
        # Lets the simulator feed the model by wavelength and reject a mismatching calibration.
        'band_wavelengths': data.wavelengths,
        'n_bands': int(num_bands),
        # 'counts' or 'reflectance', from the training cubes (the simulator compares this with its
        # Reflectance step: a counts model gets nonsense from reflectance input and vice versa).
        'input_units': data.input_units,
        # The first cube (older readers) and all of them with their labels (check_calibration.py
        # rebuilds the pooled split from these).
        'training_header': Path(data.cubes[0][0]).as_posix(),
        'training_headers': [Path(h).as_posix() for h, _ in data.cubes],
        'training_labels': [Path(label).as_posix() for _, label in data.cubes],
        # Divides the scores before the confidence (inference_engine.predict_confidence).
        'temperature': temperature,
    }
    out = Path(model_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out)
    print(f"Model input units: {data.input_units} (the simulator's Reflectance step must match)")
    print(f"\nSuccessfully trained and saved optimized model artifact to '{model_output}'")
    if report_dir is not None:
        if str(ROOT) not in sys.path:                     # run as ``python tools/train_sgd_coco.py``
            sys.path.insert(0, str(ROOT))
        from tools.sgd_report import ReportInputs, write_report
        print("Writing the visual report (whole-cube maps take a moment)...")
        index = write_report(ReportInputs(data, idx_train, idx_test, selector, scaler, model, temperature,
                                          artifact), report_dir, title=f"SGD training report - {out.name}")
        print(f"Report: {index.resolve()}")
    return artifact


def train_smart_bands_sgd(num_selected_bands: int = 30, hdr_path: str | None = None,
                          json_path: str | None = None, model_output: str = DEFAULT_OUT,
                          dataset_dir: str = DEFAULT_DATASET, report_dir=None) -> dict:
    """One cube (``hdr_path``, labels ``json_path`` or the file next to it), else every labelled cube
    of ``dataset_dir``."""
    if hdr_path is not None:
        hdr = _resolve(hdr_path)
        labels = _resolve(json_path) if json_path else label_path_for(hdr)
        if labels is None:
            raise ValueError(f"no COCO labels next to {hdr}: pass --labels")
        cubes = [(hdr, labels)]
    else:
        cubes, skipped = find_cubes(dataset_dir)
        for hdr in skipped:
            print(f"skipped {hdr.name}: no COCO labels (<name>.json) next to it")
        if not cubes:
            raise ValueError(f"no labelled cube in {dataset_dir}: need <name>.hdr with <name>.json")
    return train_sgd_on_cubes(cubes, num_selected_bands, model_output, report_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Train the linear (SGD) model on the labelled ENVI cubes of a folder (default: every '
                    'labelled cube in dataset/), or on one cube with --hdr.')
    parser.add_argument('--dataset', default=DEFAULT_DATASET,
                        help='folder with <name>.hdr + <name>.json pairs (default: %(default)s)')
    parser.add_argument('--hdr', default=None, help='train on this one cube only instead of the folder')
    parser.add_argument('--labels', default=None,
                        help='COCO polygons of --hdr (default: the .json next to it)')
    parser.add_argument('--out', default=DEFAULT_OUT, help='model artifact to write (default: %(default)s)')
    parser.add_argument('--bands', type=int, default=30, help='bands kept by SelectKBest')
    parser.add_argument('--report', default=None,
                        help='folder for the figures + index.html (default: <out>_report beside the model)')
    parser.add_argument('--no-report', action='store_true', help='numbers only, no figures')
    args = parser.parse_args()
    if args.labels and not args.hdr:
        parser.error('--labels needs --hdr')
    try:
        report = None if args.no_report else (args.report or default_report_dir(args.out))
        train_smart_bands_sgd(args.bands, args.hdr, args.labels, args.out, args.dataset, report)
    except ValueError as exc:
        sys.exit(f"error: {exc}")
