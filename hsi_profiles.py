# -*- coding: utf-8 -*-
"""hsi's camera profiles, read in place: copy hsi's ``camera_profiles`` folder into this project.

hsi keeps everything one camera was calibrated with in one folder per camera
(``hsi/modules/contracts/camera_profile.py``)::

    camera_profiles/<backend>/<model>_<serial>/
        profile.json      backend, model, serial
        dark/             dark_b<bin>_e<exp>_g<gain>[_t<temp>].npz
        smile/            smile_model_bin<N>.npz           (Smile, or Smile + keystone)
        wavelength/       wavelength_bin<N>.json
        radiometric/      active.json + ref_<date>_<time>/  (factors.npy, reference.json, ...)
        flatfield/        flat_<date>_<time>/               (flatfield.json, profile.npy, ...)

Copy the whole ``camera_profiles`` folder of hsi next to ``simulation.json`` (or point
``corrections.camera_profiles_dir`` at it) and nothing has to be renamed or converted: the
camera's folder is found from the model and serial the camera (or the recording) reports, by
hsi's own naming rule, or else by the identity in its ``profile.json``.

What is taken from it, with hsi's rules:

* dark, Smile, wavelength (``sim_config.calibration_folders``): the camera folder's file
  first; when it has none for the setting, the next folder (this project's ``profiles/`` and
  ``calibration/``), as hsi falls back to its older per-backend folders. A file in the camera
  folder that does not fit the frames is not replaced by another camera's file.
* reflectance (:func:`find_reference`): the newest *active* reference whose key fits the
  stream, chosen with hsi's ``ReferenceKey`` rules (``modules/radiometric/reference.py``, a
  byte-identical copy): camera, bin, orientation, slit rows, exposure, gain, bands, wavelength
  slope / offset, bin width, data full scale, Smile model and its creation time, keystone,
  dark and hot-pixel correction, and the dark frame when both sides know it. The factors are
  applied as hsi applies them: ``counts x factors`` after the keystone row crop, never scaled.
  Not used from hsi's rules: the stage type (a sorter has no scan stage) and the 4 h age and
  2 C temperature limits of one hsi scan (a sorter runs all day): age and temperature are
  flagged with ``corrections.reflectance_max_age_h`` / ``reflectance_max_dtemp_c`` instead.
* flat-field (``flatfield_profile.configure_flatfield``): ``flatfield/flat_*`` profiles.

Threading:
    Plain file reads on the calling thread (``build_plan``).

Example:
    >>> hsi_profile_name('Apollo-M MAX', 'CAMP7161A470A1209000')
    'Apollo-M_MAX_CAMP7161A470A1209000'
    >>> hsi_profile_name(' MQ022 HG/IM ', '12:34'), hsi_profile_name(None, '151498')
    ('MQ022_HG_IM_12_34', 'SN151498')
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import re
from dataclasses import replace
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["hsi_profile_name", "camera_profiles_root", "find_camera_profile", "dark_reference_text",
           "scan_full_scale", "find_reference", "HsiReference"]

#: hsi ``camera_profile.py``: characters outside these become ``_`` in a folder name.
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
#: Folder names hsi gives a camera that reports neither model nor serial (not used here: it
#: names no unit, so it could hold another camera's calibration).
UNKNOWN_CAMERA = "unknown"
#: hsi's older per-backend folders (``<folder>/<backend>``, next to ``camera_profiles``): hsi reads
#: a file there when the camera folder has none; copied along, they are read the same way.
HSI_LEGACY_FOLDERS = {'dark': 'dark_frames', 'smile': 'smile_maps', 'wavelength': 'calibration/wavelength'}
#: Largest difference (nm) between a stream band and the reference band it is taken from.
BAND_TOLERANCE_NM = 0.01


def _text(value) -> str | None:
    """Model / serial as text (bytes decoded, blanks -> ``None``), as hsi's ``device_text``."""
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.split(b"\0", 1)[0].decode(errors="replace")
    text = str(value).strip()
    return text or None


def _name_part(text) -> str:
    return _UNSAFE_NAME_CHARS.sub("_", str(text or "").strip()).strip("._")


def hsi_profile_name(model, serial) -> str:
    """hsi ``profile_name``: ``<model>_<serial>``, ``<model>``, ``SN<serial>`` or ``unknown``."""
    model, serial = _name_part(_text(model)), _name_part(_text(serial))
    if model and serial:
        return f"{model}_{serial}"
    if model:
        return model
    if serial:
        return f"SN{serial}"
    return UNKNOWN_CAMERA


def camera_profiles_root(config) -> Path:
    """``corrections.camera_profiles_dir`` as a path (default ``<project>/camera_profiles``)."""
    choice = getattr(config.corrections, 'camera_profiles_dir', 'camera_profiles') or 'camera_profiles'
    return config.resolve(choice) or Path(config.base_dir) / 'camera_profiles'


def _read_profile_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def find_camera_profile(config, backend, model, serial) -> Path | None:
    """The hsi camera folder of this camera in :func:`camera_profiles_root`, or ``None``.

    ``<root>/<backend>/<hsi name>`` when it exists, else the one folder ``<root>/*/*`` whose
    ``profile.json`` names this model and serial (a folder copied under another backend name).
    A camera that reports no model or no serial has no folder here: hsi would name it
    ``<model>`` / ``SN<serial>`` / ``unknown``, which another unit of the same model can share.
    """
    model, serial = _text(model), _text(serial)
    if model is None or serial is None:
        return None
    root = camera_profiles_root(config)
    if not root.is_dir():
        return None
    direct = root / str(backend or '').strip().lower() / hsi_profile_name(model, serial)
    if direct.is_dir():
        return direct
    found = []
    for record in sorted(root.glob('*/*/profile.json')):
        data = _read_profile_json(record)
        same = (_text(data.get('model')) == model and _text(data.get('serial')) == serial)
        other_backend = (_text(data.get('backend')) or '').lower() not in ('', str(backend or '').lower())
        if same and not other_backend:
            found.append(record.parent)
    if len(found) > 1:
        logger.warning("Several hsi camera folders name %s %s: %s; none used", model, serial,
                       ', '.join(str(p) for p in found))
        return None
    return found[0] if found else None


def profile_label(folder: Path) -> str:
    """``camera_profiles/<backend>/<name>`` for provenance texts."""
    return f"{folder.parent.parent.name}/{folder.parent.name}/{folder.name}"




def dark_reference_text(result: dict, camera_folder: Path | None, camera_name: str | None = None) -> str | None:
    """hsi's ``dark reference`` key text ``<file label> @ <capture time>`` of a loaded dark.

    The file label is hsi's ``_dark_file_label``: ``<camera>/dark/<file>`` inside the camera
    folder (``<camera>`` = ``camera_name``, hsi's folder name, whatever the copied folder is
    called), ``<folder>/<backend>/<file>`` for an older per-backend dark. ``None`` for a dark
    without a recorded capture time (``captured_at``): hsi would use the file's modification
    time, which copying the folder changes, so the dark is then not compared (as hsi does for
    references made before the dark was recorded).
    """
    captured = _text(result.get('captured_at'))
    path = result.get('path')
    if not captured or not path:
        return None
    path = os.path.abspath(path)
    if camera_folder is not None:
        root = os.path.abspath(str(camera_folder))
        if os.path.normcase(path).startswith(os.path.normcase(root) + os.sep):
            inner = os.path.relpath(path, root).replace(os.sep, '/')
            return f"{camera_name or os.path.basename(root)}/{inner} @ {captured}"
    base = os.path.dirname(os.path.dirname(os.path.dirname(path)))
    return f"{os.path.relpath(path, base).replace(os.sep, '/')} @ {captured}"


def scan_full_scale(meta: dict) -> int | None:
    """Largest value the frames can hold, as hsi records it (``acquisition/metadata.py``
    ``camera_bit_info``): ``((1 << sensor bits) - 1) << (bit depth - sensor bits)`` from
    ``bit_depth`` / ``sensor_bit_depth`` (12 bit in 16 -> 65520, 10 bit in 16 -> 65472), else the
    recorded ``full_scale`` / ``data_full_scale``, else ``None``. A live camera's ``full_scale``
    is its saturation level (65535 for the POA), which hsi does not use for the key."""
    try:
        bits = int(meta['bit_depth'])
        sensor = max(1, min(int(meta.get('sensor_bit_depth', bits)), bits))
        return ((1 << sensor) - 1) << (bits - sensor)
    except (KeyError, TypeError, ValueError):
        pass
    value = meta.get('full_scale', meta.get('data_full_scale'))
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


class HsiReference:
    """An active hsi reference chosen for a stream (:func:`find_reference`)."""

    def __init__(self, reference, factors: np.ndarray, bands: np.ndarray, camera_folder: Path, steps: dict,
                 rows=None):
        self.reference = reference             # modules.radiometric.reference.RadiometricReference
        self.factors = factors                 # float32 (slit rows, stream bands)
        self.bands = bands                     # indices of the reference's bands the stream uses
        self.camera_folder = camera_folder
        #: Dark / Flip / Smile / Keystone as the reference was made: the stream must run so
        self.steps = steps
        self.rows = rows                       # slit rows of the reference the stream keeps (None: all)

    def flatfield_parts(self) -> tuple:
        """``(F, ROI pixels per slit row, note)`` of a "Flat-field + reference" reference, cut to
        the stream like :attr:`factors`; ``(None, weights, note)`` when ``F`` is not known.

        ``F`` is the copy hsi keeps with the reference (``flatfield.npy``), else, for a reference
        stored before that, the profile of that id in the copied camera folder - each accepted only
        when the factors were fitted with it (hsi ``check_flatfield_fits``)."""
        flat_id = self.reference.info.get('flatfield')
        weights = None
        try:
            weights = np.asarray(self.reference.roi_mask(), bool).sum(axis=1).astype(np.float64)
        except Exception as exc:                                   # missing / unreadable mask
            logger.warning("hsi %s: ROI mask not readable: %s", self.id, exc)
        if weights is not None and self.rows is not None:
            weights = weights[self.rows]
        if not flat_id:
            return None, weights, ''
        from modules.radiometric.reference import check_flatfield_fits
        try:
            flat = self.reference.flatfield()
            source = 'kept with the reference'
            if flat is None:
                path = self.camera_folder / 'flatfield' / str(flat_id) / 'profile.npy'
                if not path.is_file():
                    return None, weights, f"{flat_id} is not in the copied camera folder"
                flat = check_flatfield_fits(self.reference.factors(), np.load(path, allow_pickle=False))
                source = 'from the copied flat-field folder'
        except (ValueError, OSError) as exc:
            return None, weights, str(exc)
        flat = np.asarray(flat, np.float32)
        if self.rows is not None:
            flat = flat[self.rows]
        return np.ascontiguousarray(flat[:, self.bands]), weights, source

    @property
    def id(self) -> str:
        return self.reference.id

    @property
    def path(self) -> Path:
        return self.reference.folder / 'factors.npy'

    @property
    def flatfield(self) -> str | None:
        """Label of the flat-field profile inside the factors ("Flat-field + reference"), or
        ``None`` (hsi ``info.flatfield``: the profile id). The GUI compares its folder name."""
        flat = self.reference.info.get('flatfield')
        if not flat:
            return None
        return str(self.camera_folder / 'flatfield' / str(flat) / 'flatfield.json')

    def info(self) -> dict:
        """The fields ``sim_config._reflectance_warnings`` reads (``profile_info`` names)."""
        key = self.reference.key
        return {'created': self.reference.acquired_at, 'temperature_c': self.reference.sensor_temperature_c,
                'steps': {'dark': bool(key.dark_corrected)}, 'camera_bin': key.camera_bin, 'gain': key.gain,
                'exposure_ms': key.exposure_ms, 'method': f'hsi radiometric reference {self.reference.id}'}

    def label(self) -> str:
        ref = self.reference
        table = f", {ref.reflectance_profile}" if ref.reflectance_profile else ''
        subset = (f", bands {int(self.bands[0])}:{int(self.bands[-1]) + 1} of {len(ref.key.wavelengths)}"
                  if self.bands.size != len(ref.key.wavelengths) else '')
        return (f"hsi reference {ref.id} (active, {profile_label(self.camera_folder)}; row-wise "
                f"{self.factors.shape}{subset}, acquired {ref.acquired_at}{table})")


def _band_subset(ref_wavelengths, wavelengths) -> np.ndarray | None:
    """Indices of ``ref_wavelengths`` equal (within :data:`BAND_TOLERANCE_NM`) to ``wavelengths``,
    in order and contiguous (a wavelength range / Smile trim of the same grid), else ``None``."""
    ref = np.asarray(ref_wavelengths, dtype=np.float64)
    want = np.asarray(wavelengths, dtype=np.float64)
    if ref.size == 0 or want.size == 0 or want.size > ref.size:
        return None
    start = int(np.argmin(np.abs(ref - want[0])))
    idx = np.arange(start, start + want.size)
    if idx[-1] >= ref.size or not np.all(np.abs(ref[idx] - want) <= BAND_TOLERANCE_NM):
        return None
    return idx


def default_state(plan) -> tuple:
    """``(dark, smile, keystone)`` as the plan starts (its steps' ``on``)."""
    steps = plan.steps
    smile = bool(steps.get('smile', {}).get('on')) and plan.smile_model is not None
    keystone = (smile and bool(getattr(plan.smile_model, 'has_keystone', False))
                and bool(steps.get('keystone', {}).get('on', True)))
    return bool(steps.get('dark', {}).get('on')) and plan.dark_mean is not None, smile, keystone


def reference_state(plan, ref_key) -> tuple:
    """``(dark, smile, keystone)`` in which the stream can take a reference of ``ref_key``: its own
    setting of each step where the plan has what the step needs (a dark, a Smile model, its
    keystone), else off (the key comparison then says why it does not fit)."""
    dark = bool(ref_key.dark_corrected) and plan.dark_mean is not None
    smile = bool(ref_key.smile_corrected) and plan.smile_model is not None
    keystone = smile and bool(ref_key.keystone_corrected) and bool(getattr(plan.smile_model, 'has_keystone', False))
    return dark, smile, keystone


def scan_key(plan, meta: dict, wavelengths, state: tuple | None = None):
    """hsi ``ReferenceKey`` of the stream with Dark / Smile / Keystone in ``state`` (default: as
    the plan starts). ``lines`` follow hsi's ``crop_keystone_rows``: the keystone margin is cut
    only with the keystone remap on; otherwise every frame row is a slit row."""
    from modules.radiometric.reference import ReferenceKey
    dark_on, smile_on, keystone = default_state(plan) if state is None else state
    model = plan.smile_model if smile_on else None
    smile_path = getattr(plan, 'smile_path', None)
    lines = int(plan.n_slit) if keystone else int(plan.frame_shape[0])
    return ReferenceKey(
        camera_backend=_text(plan.camera_backend), camera_model=_text(meta.get('camera_model')),
        camera_serial=_text(meta.get('camera_serial')), camera_bin=int(plan.camera_bin),
        raw_inverted=bool(plan.raw_inverted), lines=lines,
        exposure_ms=None if plan.exposure_ms is None else float(plan.exposure_ms),
        gain=None if plan.gain is None else float(plan.gain),
        wavelengths=tuple(float(v) for v in np.asarray(wavelengths, dtype=np.float64).ravel()),
        resolution_nm=float(plan.resolution_nm), wavelength_m_nm_per_px=float(plan.m_nm_per_px_bin1),
        wavelength_c_nm=float(plan.c_nm), full_scale=scan_full_scale(meta),
        smile_corrected=bool(smile_on),
        smile_model=Path(smile_path).name if smile_on and smile_path else None,
        smile_model_created=_text(getattr(model, 'created', None)) if smile_on else None,
        keystone_corrected=True if keystone else None,
        dark_corrected=bool(dark_on), hot_pixel_corrected=bool(dark_on and plan.hot_map is not None),
        stage_type=None, dark_reference=getattr(plan, 'dark_reference', None) if dark_on else None)


def find_reference(plan, meta: dict, now: datetime.datetime | None = None):
    """``(HsiReference or None, why)``: the active reference of the plan's hsi camera folder the
    stream takes, and the reason when it takes none.

    As hsi's ``RadiometricReferenceStore.find`` (without the age / temperature limits and the
    stage type, see the module docstring): the active references are tried newest first and
    the first whose key fits is taken. Each is compared in the Dark / Smile / Keystone setting it
    was made with, where the plan can run it; one made in the setting the plan starts with is
    preferred. As in hsi, factors with NaN or non-positive values in the stream's bands are not
    applied, and no older reference is taken instead.
    """
    from modules.radiometric.reference import UNREADABLE, ReferenceMatch, RadiometricReferenceStore, \
        reference_problems
    folder = getattr(plan, 'hsi_profile_dir', None)
    if folder is None:
        return None, 'no hsi camera folder'
    store_folder = folder / 'radiometric'
    if not store_folder.is_dir():
        return None, f"no radiometric/ in {profile_label(folder)}"
    store = RadiometricReferenceStore(store_folder)
    rejected, fitting = {}, []
    # as RadiometricReferenceStore.find: unreadable active ids are named, then the readable
    # active references are tried newest first
    for ref_id in store.active_ids():
        try:
            store.load(ref_id)
        except (OSError, ValueError) as exc:
            rejected[ref_id] = (f"{UNREADABLE}: {exc}",)
    start = default_state(plan)
    for ref in store.active():
        state = reference_state(plan, ref.key)
        full = scan_key(plan, meta, plan.wavelengths, state)
        bands = _band_subset(ref.key.wavelengths, plan.wavelengths)
        key = replace(full, stage_type=ref.key.stage_type)         # a sorter has no scan stage
        if bands is not None:
            # the stream keeps these bands of the reference's grid; its slope, offset and bin
            # width are compared exactly below, so they are the same bins: compare the
            # reference's own band list and use the factors of the kept bands
            key = replace(key, wavelengths=tuple(ref.key.wavelengths))
        problems = reference_problems(ref, key, now=now, max_age_h=None, temperature_tolerance_c=None)
        if bands is None and not any(p.startswith('bands') for p in problems):
            problems.append(f"bands: the stream's {len(full.wavelengths)} bands are not a range of the "
                            f"reference's {len(ref.key.wavelengths)}")
        if problems:
            rejected[ref.id] = tuple(problems)
            continue
        try:
            factors = ref.factors()
        except Exception as exc:                    # missing, truncated, wrong shape ... (as hsi)
            rejected[ref.id] = (f"{UNREADABLE}: factors: {exc or type(exc).__name__}",)
            continue
        rows = None
        if not state[2] and factors.shape[0] != plan.n_slit:
            rows = plan.slit_rows                   # hsi kept every row; the plan crops the keystone margin
            factors = factors[rows]
        fitting.append((ref, state, factors[:, bands], bands, rows))
    if not fitting:
        return None, ReferenceMatch(None, rejected).reason()
    ref, state, factors, bands, rows = next((f for f in fitting if f[1] == start), fitting[0])
    if not np.isfinite(factors).all() or (factors <= 0).any():
        # hsi records such a scan in counts and calibrates it afterwards with this reference;
        # a live model cannot wait, and an older reference is not what hsi would use
        return None, (f"{ref.id} not applied: factors have NaN or non-positive values in the stream's "
                      "bands (bands outside the reflectance table)")
    steps = {'dark': state[0], 'flip': bool(ref.key.raw_inverted), 'smile': state[1], 'keystone': state[2]}
    return (HsiReference(ref, np.ascontiguousarray(factors, dtype=np.float32), bands, folder, steps, rows),
            f"calibrated with {ref.id}")
