"""Portable HSI flat-field profiles; no dependency on the acquisition project."""
from __future__ import annotations

import json
import datetime
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FlatField:
    values: np.ndarray
    label: str
    # Processing state used when this profile was measured.
    dark: bool
    flip: bool
    smile: bool
    keystone: bool = False        # measured with a smile + keystone model (rows cropped like here)


def load_flatfield(path, plan) -> FlatField:
    """Read a copied flat_<date>_<time> folder or its flatfield.json, validating its key."""
    path = Path(path)
    if path.is_dir():
        path = path / 'flatfield.json'
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict) or not isinstance(data.get('key'), dict):
        raise ValueError('Flat-field metadata must contain a key object')
    if data.get('format_version') != 1:
        raise ValueError('Unsupported flat-field format_version (expected 1)')
    key = data['key']
    expected = {
        'camera_backend': plan.camera_backend,
        'camera_bin': plan.camera_bin,
        'lines': plan.n_slit,                      # slit rows after the keystone crop (as hsi records)
        'gain': plan.gain,
        'raw_inverted': plan.raw_inverted,
        'resolution_nm': plan.resolution_nm,
    }
    for name, value in expected.items():
        actual = key.get(name)
        if actual is None or value is None:
            raise ValueError(f'Flat-field: {name} must be known in both profile and input')
        equal = (str(actual).lower() == str(value).lower() if name == 'camera_backend'
                 else np.isclose(actual, value, rtol=1e-6, atol=1e-6))
        if not equal:
            raise ValueError(f'Flat-field {name}: profile {actual}, input {value}')
    from sim_config import camera_profile_name
    identity = camera_profile_name(key.get('camera_model'), key.get('camera_serial'))
    if not identity or identity != plan.camera_id:
        raise ValueError(f'Flat-field camera identity: profile {identity}, input {plan.camera_id}')
    if key.get('dark_corrected') is not True:
        raise ValueError('Flat-field profile must have dark subtraction enabled')
    for name in ('smile_corrected', 'hot_pixel_corrected'):
        if not isinstance(key.get(name), bool):
            raise ValueError(f'Flat-field {name} is unknown')
    hot = bool(plan.hot_map is not None)
    if key['hot_pixel_corrected'] != hot:
        raise ValueError('Flat-field hot-pixel correction differs from this input')
    if plan.stage != 'raw':
        raise ValueError('Flat-field currently requires raw input with known dark/Smile settings')
    if plan.dark_mean is None or (key['smile_corrected'] and plan.smile_model is None):
        raise ValueError('Flat-field requires the matching dark and Smile references to be loaded')
    keystone = bool(key.get('keystone_corrected'))        # absent (older hsi) = no keystone
    if keystone != bool(getattr(plan.smile_model, 'has_keystone', False)):
        raise ValueError('Flat-field geometry: ' + ('measured with keystone correction, this input has no '
                         'smile + keystone model' if keystone else 'measured without keystone correction, '
                         'this input has a smile + keystone model - make a new one with it'))
    if key['smile_corrected'] and key.get('smile_model_created'):
        if key['smile_model_created'] != getattr(plan.smile_model, 'created', None):
            raise ValueError('Flat-field was measured with a different Smile calibration')
    wl = np.asarray(key['wavelengths'], dtype=np.float64)
    current = np.asarray(plan.wavelengths, dtype=np.float64)
    if wl.ndim != 1 or not len(wl) or not np.isfinite(wl).all() or np.any(np.diff(wl) <= 0):
        raise ValueError('Flat-field wavelengths must be finite and strictly increasing')
    # HSI may save the full calibrated grid while the sorter uses a cropped range.
    # Select identical bands; never interpolate a spatial correction onto another grid.
    columns = np.abs(wl[:, None] - current[None, :]).argmin(axis=0)
    if len(np.unique(columns)) != len(current) or np.any(np.abs(wl[columns] - current) > 0.01):
        raise ValueError('Flat-field wavelengths do not cover the input grid (tolerance 0.01 nm)')
    try:
        values = np.load(path.parent / 'profile.npy', allow_pickle=False)
        sidecar = np.load(path.parent / 'profile.wavelength.npy', allow_pickle=False)
    except (ValueError, OSError, EOFError) as exc:
        raise ValueError(f'Cannot read flat-field arrays: {exc}') from exc
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (plan.n_slit, len(wl)) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Flat-field profile must be positive, finite and match (slit rows, bands)')
    if sidecar.shape != wl.shape or not np.allclose(sidecar, wl, rtol=0, atol=0.01):
        raise ValueError('Flat-field wavelength sidecar differs from its metadata')
    values = np.ascontiguousarray(values[:, columns])
    values.setflags(write=False)
    return FlatField(values, str(path), True, bool(key['raw_inverted']), key['smile_corrected'],
                     keystone and bool(key['smile_corrected']))


def install_flatfield(plan, profile, enabled=False):
    plan.flatfield_profile = profile
    plan.steps['flatfield'] = {'available': True, 'on': enabled,
                              'why': profile.label + '; requires matching Dark/Flip/Smile/Keystone; '
                              'Reflectance uses counts directly; choose Flat-field + reference when calibrating'}
    plan.provenance['flatfield'] = (profile.label, 'copied HSI profile')


def configure_flatfield(config, plan):
    """Offer the newest compatible profile: of this camera's local folder, of the copied hsi
    camera folder (``camera_profiles/<backend>/<model>_<serial>/flatfield``) or copied under
    ``profiles/``, matched by the identity in each profile's key."""
    choice = config.corrections.flatfield
    folder = plan.profile_dir / 'flatfield' if plan.profile_dir else None
    reason = f'No matching flat-field profile in {folder}' if folder else 'Camera identity is unknown'
    plan.steps['flatfield'] = {'available': False, 'on': False, 'why': reason}
    plan.provenance['flatfield'] = (reason, 'local profiles')
    from hsi_profiles import camera_profiles_root
    from sim_config import profiles_root
    roots = (profiles_root(config), camera_profiles_root(config))
    if choice not in (None, '', 'none', 'auto'):
        path = config.resolve(choice).resolve()
        if not any(_inside(path, r) for r in roots):
            raise ValueError(f"corrections.flatfield {path} is not under {roots[0]} or {roots[1]}")
        install_flatfield(plan, load_flatfield(path, plan), config.corrections.flatfield_enabled)
        return
    if folder is None:
        return
    candidates = []
    root = roots[0]
    # HSI's camera folder names include backend/underscores, whereas the sorter normally
    # uses <model>-<serial>. Match the metadata identity instead of forcing a folder rename.
    folders = {folder} | set(root.glob('*/flatfield')) | set(root.glob('*/*/flatfield'))
    hsi = getattr(plan, 'hsi_profile_dir', None)
    if hsi is not None and (hsi / 'flatfield').is_dir():
        folders.add(hsi / 'flatfield')

    def rel(path):
        for base in roots:
            if _inside(path, base):
                return path.resolve().relative_to(base.resolve())
        return path
    paths = set()
    for candidate_folder in folders:
        paths.update(candidate_folder.glob('flatfield.json'))
        paths.update(candidate_folder.glob('*/flatfield.json'))
        paths.update(candidate_folder.glob('*/*/flatfield.json'))
    rejected = []
    for path in sorted(paths):
        try:
            data = json.loads(path.read_text(encoding='utf-8-sig'))
            created = datetime.datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
            if created.tzinfo is None:
                raise ValueError('created_at must include timezone')
            candidates.append((created, path))
        except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
            rejected.append(f'{rel(path)}: {exc}')
            logging.getLogger(__name__).warning('Flat-field %s skipped: %s', path, exc)
    for _, path in sorted(candidates, key=lambda item: (item[0], str(item[1])), reverse=True):
        try:
            profile = load_flatfield(path, plan)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            rejected.append(f'{rel(path)}: {exc}')
            logging.getLogger(__name__).warning('Flat-field %s skipped: %s', path, exc)
            continue
        install_flatfield(plan, profile, choice == 'auto' and config.corrections.flatfield_enabled)
        return
    if rejected:
        reason = 'No compatible flat-field profile:\n' + '\n'.join(rejected)
    elif folder is not None:
        reason += ' (expected flatfield.json + profile.npy + profile.wavelength.npy, directly or in a profile subfolder)'
    plan.steps['flatfield']['why'] = reason
    plan.provenance['flatfield'] = (reason, 'local profiles')


def _inside(path: Path, root: Path) -> bool:
    """True when ``path`` lies under ``root``."""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False
