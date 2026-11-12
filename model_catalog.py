# -*- coding: utf-8 -*-
"""The models a running pipeline can switch to (GUI model selector).

:func:`scan_models` looks into the folders of ``model.folders`` (default: ``models``) and nowhere
else for model files (joblib artifacts, HSI MLOps packages ``.hsipkg`` or the ``.zip`` around
one), loads each with ``inference_engine.load_classifier`` and binds it to
every acquisition plan of the pipeline. A model is **usable** when all of that works: it can
then be handed to ``Pipeline.set_classifier`` without stopping the camera. One that does not
fit (other bands, another wavelength calibration, a package this program cannot run
faithfully) is listed too, with the reason, so that a delivered model never silently goes
missing from the list.
"""
from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_SUFFIXES = ('.joblib', '.hsipkg', '.zip')


@dataclass
class ModelEntry:
    """One model file and whether the running pipeline can use it."""
    path: Path
    label: str                      # one line for the selector
    usable: bool
    detail: str                     # tooltip: what it is and how it binds, or why it cannot be used
    classifier: object = None       # loaded and ready for Pipeline.set_classifier (usable entries)
    kind: str = ''                  # short second line of a selector tile: what it is, or why not usable
    resampled_nm: float | None = None   # usable through interpolation: largest distance model band - input band

    def same_file(self, other) -> bool:
        return other is not None and _key(self.path) == _key(other)


def _key(path) -> str:
    try:
        return str(Path(path).resolve()).casefold()
    except OSError:
        return str(path).casefold()


def model_folders(config) -> list[Path]:
    """Folders searched for models: the existing ones of ``model.folders``, nothing else."""
    folders = []
    for name in getattr(config.model, 'folders', ()) or ():
        path = config.resolve(name)
        if path is not None:
            folders.append(Path(path))
    unique, seen = [], set()
    for folder in folders:
        if folder.is_dir() and _key(folder) not in seen:
            seen.add(_key(folder))
            unique.append(folder)
    return unique


def _holds_package(path: Path) -> bool:
    """A zip that is, or wraps, a model package (cheap: reads the member names only)."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except (OSError, zipfile.BadZipFile):
        return False
    return 'manifest.json' in names or any(name.lower().endswith('.hsipkg') for name in names)


def model_files(folders) -> list[Path]:
    """Model files of ``folders`` (not recursive), by name; a ``.zip`` only if it holds a package."""
    files, seen = [], set()
    for folder in folders:
        try:
            members = sorted(Path(folder).iterdir(), key=lambda p: p.name.casefold())
        except OSError:
            continue
        for path in members:
            suffix = path.suffix.lower()
            if not path.is_file() or suffix not in MODEL_SUFFIXES or _key(path) in seen:
                continue
            if suffix == '.zip' and not _holds_package(path):
                continue
            seen.add(_key(path))
            files.append(path)
    return files


def _describe(classifier) -> tuple[str, str]:
    """``(kind for the label, lines for the tooltip)`` of a loaded classifier."""
    names = [str(classifier.category_names.get(int(c), int(c))) for c in classifier.classes_]
    package = getattr(classifier, 'package', None)
    if package is not None:
        kind = f"{package.algorithm} · {len(names)} classes"
        lines = [package.chain, f"score: {package.metric}", f"hsi_lab_core {package.lab_core_version}, "
                 f"exported {package.exported_at}"]
    else:
        kind = f"linear · {len(names)} classes"
        lines = [f"linear model on {classifier.selected_bands.size} bands"]
    lines.append(f"classes: {', '.join(names)}")
    lines.append(f"input: {classifier.input_units}"
                 + ('' if classifier.input_units_declared else ' (assumed: the artifact does not say)'))
    return kind, '\n'.join(lines)


def probe_model(path, plans, tolerance_nm: float, classifier=None, resample: bool = True) -> ModelEntry | None:
    """Load ``path`` and bind it to every plan.

    Args:
        resample: ``model.resample``: a model whose bands lie beside the input's is usable, fed
            the input interpolated onto its bands; its entry says so (``resampled_nm``).
        classifier: The already loaded classifier of ``path`` (the pipeline's current model), to
            save loading it again.

    Returns:
        The entry, or ``None`` for a joblib file that is not a model artifact at all (those
        folders hold other joblib files too).
    """
    from inference_engine import Resampled, load_classifier
    path = Path(path)
    try:
        if classifier is None:
            classifier = load_classifier(path)
    except (KeyError, TypeError, AttributeError) as exc:
        if path.suffix.lower() == '.joblib':             # some other joblib file, not a model artifact
            logger.debug("%s is not a model artifact (%s: %s)", path.name, type(exc).__name__, exc)
            return None
        return ModelEntry(path, f"{path.stem}  — not usable", False, f"{path}\n{type(exc).__name__}: {exc}",
                          kind='not usable')
    except Exception as exc:                             # refused package, unreadable file, ...
        logger.info("Model %s cannot be used: %s", path.name, exc)
        return ModelEntry(path, f"{path.stem}  — not usable", False, f"{path}\n{exc}", kind='not usable')
    kind, about = _describe(classifier)
    try:
        bound = [classifier.bind(plan.wavelengths, plan.n_columns, tolerance_nm, resample) for plan in plans]
    except ValueError as exc:
        logger.info("Model %s does not fit the input: %s", path.name, exc)
        return ModelEntry(path, f"{path.stem}  ({kind})  — does not fit this input", False,
                          f"{path}\n{about}\n\nDoes not fit the input:\n{exc}", kind='does not fit this input')
    shifts = [columns.max_shift_nm for columns, _ in bound if isinstance(columns, Resampled)]
    texts = list(dict.fromkeys(text for _, text in bound))          # one line per different binding
    detail = f"{path}\n{about}\nbands: " + ('\n       '.join(texts) if texts else 'not bound')
    if not shifts:
        return ModelEntry(path, f"{path.stem}  ({kind})", True, detail, classifier, kind)
    shift = max(shifts)
    detail += ("\n\nResampled: this model was trained with another wavelength calibration than the camera has "
               f"now (bands up to {shift:.2f} nm apart). The input is interpolated onto the model's bands, so the "
               "model can be used; check its result on known material, and retrain it when it matters.")
    return ModelEntry(path, f"{path.stem}  ({kind}, resampled {shift:.2f} nm)", True, detail, classifier,
                      f"{kind} · resampled {shift:.1f} nm", resampled_nm=shift)


def scan_models(config, plans, current=None) -> list[ModelEntry]:
    """Every model file of :func:`model_folders`, probed against ``plans``.

    Args:
        current: The pipeline's classifier; when its file is one of the list it is not loaded a
            second time. A current model from elsewhere is not listed.
    """
    tolerance = config.model.band_tolerance_nm
    files = model_files(model_folders(config))
    current_path = getattr(current, 'path', None)
    entries = []
    for path in files:
        own = current if current_path and _key(path) == _key(current_path) else None
        entry = probe_model(path, plans, tolerance, own, getattr(config.model, 'resample', True))
        if entry is not None:
            entries.append(entry)
    logger.info("Models: %d usable of %d found in %s", sum(e.usable for e in entries), len(entries),
                ', '.join(str(f) for f in model_folders(config)) or '(no folder)')
    return entries
