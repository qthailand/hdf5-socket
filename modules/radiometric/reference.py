"""Radiometric references kept per camera, and the rule that decides when one may be applied.

A *radiometric reference* is the set of row-wise factors fitted on one scan of a reflectance
target (Spectralon), stored so that later scans at the same camera setting can be calibrated
without opening the calibration window each time. Row-wise factors depend on the slit row and
the band only, so the scan length and the stage speed may differ; everything that changes the
counts per pixel (camera, bin, exposure, gain, wavelength axis, corrections, stage type) must
be the same. That setting is a :class:`ReferenceKey`; :func:`reference_problems` also limits
the reference's age and the sensor-temperature difference.

Layout of a store folder (in the application: the camera profile's ``radiometric/``)::

    <folder>/
        ref_<YYYYMMDD>_<HHMMSS>[_<n>]/   one reference, never modified after it is written
            reference.json             format_version, id, created_at, acquired_at, key, profile, ROI, info
            factors.npy                (lines, bands) float32   (save_factor_profile)
            factors.wavelength.npy     (bands,) float64 band centres, nm
            roi_mask.npy               bool (lines, samples) ROI of the reference scan
            roi_mean_spectrum.npy      (bands,) float64 mean ROI counts (when given)
        active.json                    {"format_version": 1, "active": ["ref_...", ...]}

At most one active reference per key: activating one deactivates the others with the same key.

Data conventions:
    Factors multiply dark-subtracted counts of a ``(lines, samples, bands)`` cube (see
    :func:`modules.radiometric.core.apply_rowwise_factors`). Exposure is in ms, wavelengths and
    bin widths in nm, temperatures in degrees C, ages in hours. ``created_at`` (save time) and
    ``acquired_at`` (when the reference target was scanned; the age limit counts from it) are
    ISO 8601 times with UTC offset (``2026-11-04T10:12:00+07:00``). Mismatch texts are plain
    ASCII, since they can end up in an ENVI header (``calibration warnings``).

Threading:
    Plain file I/O on the calling thread; use one store from one thread at a time (the GUI
    thread). ``active.json`` is replaced atomically and a reference folder appears only once
    complete, so a crash never leaves a half-written reference that looks valid.

Example:
    >>> import numpy as np
    >>> from modules.radiometric.reference import ReferenceKey, RadiometricReferenceStore
    >>> meta = {'camera_backend': 'poa', 'camera_model': 'Apollo-M MAX', 'camera_serial': 'CAMP7161',
    ...         'camera_bin': 2, 'raw_inverted': True, 'exposure_ms': 10.0, 'gain': 100.0,
    ...         'resolution_nm': 2.0, 'dark_corrected': True, 'hot_pixel_corrected': True,
    ...         'smile_corrected': False, 'stage_type': 'linear'}
    >>> key = ReferenceKey.from_scan(meta, wavelengths=[400.0, 402.0, 404.0], lines=4)
    >>> key.mismatch(ReferenceKey.from_scan({**meta, 'exposure_ms': 12.0}, [400.0, 402.0, 404.0], 4))
    ['exposure: scan 12 ms, reference 10 ms']
    >>> store = RadiometricReferenceStore('camera_profiles/poa/Apollo-M_MAX_CAMP7161/radiometric')  # doctest: +SKIP
    >>> ref = store.save(np.ones((4, 3), np.float32), key, reflectance_profile='Spectralon_ref-040.xlsx',
    ...                  avg_window_size=20, mask=np.ones((4, 6), bool), activate=True)   # doctest: +SKIP
    >>> store.find(key).reference.id == ref.id                                             # doctest: +SKIP
    True
"""
from __future__ import annotations

import datetime
import json
import logging
import math
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import check_factor_bands, load_factor_profile, save_factor_profile

logger = logging.getLogger(__name__)

__all__ = ["ReferenceKey", "RadiometricReference", "RadiometricReferenceStore", "ReferenceMatch",
           "reference_problems", "scan_temperature_c", "REFERENCE_FORMAT_VERSION", "UNREADABLE",
           "DEFAULT_MAX_AGE_H", "DEFAULT_TEMPERATURE_TOLERANCE_C"]

#: ``format_version`` of ``reference.json`` and ``active.json``.
REFERENCE_FORMAT_VERSION = 1
#: A reference older than this (hours) is not applied: lamps drift (to be confirmed by test T4).
DEFAULT_MAX_AGE_H = 4.0
#: Largest sensor-temperature difference (degrees C) between reference and scan (test T5).
DEFAULT_TEMPERATURE_TOLERANCE_C = 2.0

_REFERENCE_ID = re.compile(r"^ref_\d{8}_\d{6}(?:_\d+)?$")
_REFERENCE_FILE = "reference.json"
_FACTORS_FILE = "factors.npy"
_MASK_FILE = "roi_mask.npy"
_MEAN_FILE = "roi_mean_spectrum.npy"
_ACTIVE_FILE = "active.json"
#: Temperatures at or below this are a camera's "no reading" value (-999), not a reading.
_NO_TEMPERATURE_C = -273.15
#: Start of the reason given for an active reference whose files cannot be read.
UNREADABLE = "cannot be read"


# ── small typed readers (metadata can come from a live scan or a parsed ENVI header) ──

def _text(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int(value) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None if value is None else int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if math.isfinite(number) and number == int(number) else None


def _float(value) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bool(value) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
        return None
    return bool(value)


def _temperature(value) -> Optional[float]:
    number = _float(value)
    return number if number is not None and number > _NO_TEMPERATURE_C else None


def _first(meta: dict, *names):
    for name in names:
        value = meta.get(name)
        if value is not None:
            return value
    return None


def _fmt(value, unit: str = "") -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:g}{unit}"
    return f"{value}{unit}"


def scan_temperature_c(metadata: dict) -> Optional[float]:
    """Mean sensor temperature of a scan (degrees C), or ``None`` when the camera gave none.

    Uses ``sensor_temperature_c_start`` / ``_end`` (``ScanController.last_scan_temperatures``),
    else ``_min`` / ``_max``; camera "no reading" values (``-999``) are ignored.
    """
    for pair in (("sensor_temperature_c_start", "sensor_temperature_c_end"),
                 ("sensor_temperature_c_min", "sensor_temperature_c_max")):
        values = [t for t in (_temperature(metadata.get(name)) for name in pair) if t is not None]
        if values:
            return float(sum(values) / len(values))
    return None


# ── the camera setting a reference belongs to ──────────────────────────────

@dataclass(frozen=True)
class ReferenceKey:
    """Everything that must be equal between a reference scan and a scan it calibrates.

    ``None`` means unknown. A value unknown on one side only is a mismatch; one unknown on
    both sides is not compared, except the values :meth:`missing` requires.

    Attributes:
        camera_backend: ``"poa"``, ``"genicam"``, ``"ximea"`` (compared case-insensitively).
        camera_model: Model reported by the SDK.
        camera_serial: Serial number reported by the SDK.
        camera_bin: Hardware bin.
        raw_inverted: Whether raw frames are flipped before processing.
        lines: Slit rows of the cube (rows of the factors).
        exposure_ms: Exposure time actually applied, ms.
        gain: Sensor gain actually applied.
        wavelengths: Band centres of the cube, nm (compared as :func:`check_factor_bands` does).
        resolution_nm: Spectral bin width, nm.
        wavelength_m_nm_per_px: Wavelength slope used (bin 1), nm per pixel.
        wavelength_c_nm: Wavelength offset used, nm.
        full_scale: Largest value the frames can hold (``camera_bit_info``: bit depth and the
            shift of the sensor bits, e.g. 65520 for 12 -> 16 bit, 255 for 8 bit).
        smile_corrected: Whether the Smile correction was applied.
        smile_model: Smile model file (compared when Smile correction is on).
        smile_model_created: Creation time of that model: a re-fitted model keeps its file name.
        keystone_corrected: Whether the Smile model also corrected keystone (rows moved along
            the slit, :mod:`modules.smile.keystone`). ``None`` and ``False`` both mean "no
            keystone" (keys written before keystone existed have no value); a reference made
            with one geometry does not fit scans of the other.
        dark_corrected: Whether dark subtraction was applied (required: factors need it).
        hot_pixel_corrected: Whether hot pixels were repaired.
        stage_type: ``"linear"``, ``"rotation"`` or ``"free"`` (no stage).
        dark_reference: The dark frame the counts were corrected with: ``<dark file> @ <capture
            time>`` (header ``dark reference`` + ``dark reference captured``). Compared only when
            both sides know it (references made before it was recorded have none): factors fitted
            on counts of one dark do not fit counts of another (a dark that caught light removes
            lines from the counts, and the reflectance then shows false peaks and dips).
    """

    camera_backend: Optional[str] = None
    camera_model: Optional[str] = None
    camera_serial: Optional[str] = None
    camera_bin: Optional[int] = None
    raw_inverted: Optional[bool] = None
    lines: Optional[int] = None
    exposure_ms: Optional[float] = None
    gain: Optional[float] = None
    wavelengths: Tuple[float, ...] = ()
    resolution_nm: Optional[float] = None
    wavelength_m_nm_per_px: Optional[float] = None
    wavelength_c_nm: Optional[float] = None
    full_scale: Optional[int] = None
    smile_corrected: Optional[bool] = None
    smile_model: Optional[str] = None
    smile_model_created: Optional[str] = None
    keystone_corrected: Optional[bool] = None
    dark_corrected: Optional[bool] = None
    hot_pixel_corrected: Optional[bool] = None
    stage_type: Optional[str] = None
    dark_reference: Optional[str] = None

    @classmethod
    def from_scan(cls, metadata: dict, wavelengths: Optional[Sequence[float]], lines: Optional[int]) -> "ReferenceKey":
        """Key of a scan from its metadata.

        Args:
            metadata: The scan's raw metadata (``build_raw_metadata``) merged with its export-header
                metadata (``ScanController.last_scan_header``), or ``metadata_from_header`` of an
                ENVI file. ``wavelength_m_nm_per_px`` / ``wavelength_c_nm`` are preferred over the
                scan-start ``m_nm_per_px_bin1`` / ``c_nm``.
            wavelengths: Band centres of the cube (nm).
            lines: Slit rows of the cube.
        """
        wl = () if wavelengths is None else tuple(float(v) for v in np.asarray(wavelengths, dtype=np.float64).ravel())
        return cls(
            camera_backend=_text(metadata.get("camera_backend")),
            camera_model=_text(metadata.get("camera_model")),
            camera_serial=_text(metadata.get("camera_serial")),
            camera_bin=_int(metadata.get("camera_bin")),
            raw_inverted=_bool(metadata.get("raw_inverted")),
            lines=_int(lines),
            exposure_ms=_float(metadata.get("exposure_ms")),
            gain=_float(metadata.get("gain")),
            wavelengths=wl,
            resolution_nm=_float(metadata.get("resolution_nm")),
            wavelength_m_nm_per_px=_float(_first(metadata, "wavelength_m_nm_per_px", "m_nm_per_px_bin1")),
            wavelength_c_nm=_float(_first(metadata, "wavelength_c_nm", "c_nm")),
            full_scale=_int(metadata.get("full_scale")),
            smile_corrected=_bool(metadata.get("smile_corrected")),
            smile_model=_text(metadata.get("smile_model")),
            smile_model_created=_text(metadata.get("smile_model_created")),
            keystone_corrected=_bool(metadata.get("keystone_corrected")),
            dark_corrected=_bool(metadata.get("dark_corrected")),
            hot_pixel_corrected=_bool(metadata.get("hot_pixel_corrected")),
            stage_type=_text(metadata.get("stage_type")),
            dark_reference=_dark_reference(metadata),
        )

    def to_dict(self) -> dict:
        """JSON-serialisable dict (``wavelengths`` as a list)."""
        data = asdict(self)
        data["wavelengths"] = list(self.wavelengths)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ReferenceKey":
        """Inverse of :meth:`to_dict`; unknown keys are ignored, missing ones are ``None``."""
        data = dict(data or {})
        wl = data.get("wavelengths") or ()
        return cls(
            camera_backend=_text(data.get("camera_backend")),
            camera_model=_text(data.get("camera_model")),
            camera_serial=_text(data.get("camera_serial")),
            camera_bin=_int(data.get("camera_bin")),
            raw_inverted=_bool(data.get("raw_inverted")),
            lines=_int(data.get("lines")),
            exposure_ms=_float(data.get("exposure_ms")),
            gain=_float(data.get("gain")),
            wavelengths=tuple(float(v) for v in wl),
            resolution_nm=_float(data.get("resolution_nm")),
            wavelength_m_nm_per_px=_float(data.get("wavelength_m_nm_per_px")),
            wavelength_c_nm=_float(data.get("wavelength_c_nm")),
            full_scale=_int(data.get("full_scale")),
            smile_corrected=_bool(data.get("smile_corrected")),
            smile_model=_text(data.get("smile_model")),
            smile_model_created=_text(data.get("smile_model_created")),
            keystone_corrected=_bool(data.get("keystone_corrected")),
            dark_corrected=_bool(data.get("dark_corrected")),
            hot_pixel_corrected=_bool(data.get("hot_pixel_corrected")),
            stage_type=_text(data.get("stage_type")),
            dark_reference=_text(data.get("dark_reference")),
        )

    def missing(self) -> List[str]:
        """Why this key cannot carry a reference (empty = usable).

        Exposure, gain, bin, slit rows and band centres must be known, and dark subtraction must
        have been on: the factors are ``R / counts`` of dark-subtracted counts.
        """
        problems = [f"{label} unknown" for label, value in (
            ("exposure", self.exposure_ms), ("gain", self.gain), ("camera bin", self.camera_bin),
            ("slit rows", self.lines)) if value is None]
        if not self.wavelengths:
            problems.append("band wavelengths unknown")
        if self.dark_corrected is not True:
            problems.append("dark subtraction was " + ("off" if self.dark_corrected is False else "unknown")
                            + ": factors need dark-subtracted counts")
        return problems

    def label(self) -> str:
        """Short description, e.g. ``Apollo-M MAX #CAMP7161, bin 2, 10 ms, gain 100, 3 bands``."""
        camera = " ".join(v for v in (self.camera_model or self.camera_backend or "camera",
                                      f"#{self.camera_serial}" if self.camera_serial else "") if v)
        return (f"{camera}, bin {_fmt(self.camera_bin)}, {_fmt(self.exposure_ms, ' ms')}, "
                f"gain {_fmt(self.gain)}, {len(self.wavelengths)} bands")

    def mismatch(self, scan: "ReferenceKey") -> List[str]:
        """Differences that forbid applying a reference of this key to ``scan`` (empty = same setting).

        ``self`` is the reference's key. Texts read ``"<what>: scan <value>, reference <value>"``.
        """
        problems: List[str] = []

        def compare(label, ref_value, scan_value, equal=None, unit=""):
            if ref_value is None and scan_value is None:
                return
            same = (ref_value is not None and scan_value is not None
                    and (equal(ref_value, scan_value) if equal else ref_value == scan_value))
            if not same:
                problems.append(f"{label}: scan {_fmt(scan_value, unit)}, reference {_fmt(ref_value, unit)}")

        def close(rel, abs_):
            return lambda a, b: math.isclose(a, b, rel_tol=rel, abs_tol=abs_)

        compare("camera backend", self.camera_backend, scan.camera_backend, lambda a, b: a.lower() == b.lower())
        compare("camera model", self.camera_model, scan.camera_model)
        compare("camera serial", self.camera_serial, scan.camera_serial)
        compare("camera bin", self.camera_bin, scan.camera_bin)
        compare("orientation (raw inverted)", self.raw_inverted, scan.raw_inverted)
        compare("slit rows", self.lines, scan.lines)
        compare("exposure", self.exposure_ms, scan.exposure_ms, close(1e-6, 1e-6), " ms")
        compare("gain", self.gain, scan.gain, close(1e-6, 1e-6))
        problems.extend(self._band_mismatch(scan))
        compare("bin width", self.resolution_nm, scan.resolution_nm, close(1e-6, 1e-6), " nm")
        compare("wavelength slope", self.wavelength_m_nm_per_px, scan.wavelength_m_nm_per_px,
                close(1e-9, 1e-12), " nm/px")
        compare("wavelength offset", self.wavelength_c_nm, scan.wavelength_c_nm, close(1e-9, 1e-9), " nm")
        compare("data full scale", self.full_scale, scan.full_scale)
        compare("smile correction", self.smile_corrected, scan.smile_corrected)
        if self.smile_corrected and scan.smile_corrected:
            compare("smile model", self.smile_model, scan.smile_model)
            compare("smile model created", self.smile_model_created, scan.smile_model_created)
        problems.extend(self._keystone_mismatch(scan))
        compare("dark correction", self.dark_corrected, scan.dark_corrected)
        compare("hot-pixel correction", self.hot_pixel_corrected, scan.hot_pixel_corrected)
        compare("stage type", self.stage_type, scan.stage_type, lambda a, b: a.lower() == b.lower())
        if self.dark_reference is not None and scan.dark_reference is not None                 and self.dark_reference != scan.dark_reference:
            problems.append(f"dark reference: scan {scan.dark_reference}, reference {self.dark_reference}"
                            " - make a new reference with the current dark")
        return problems

    def _keystone_mismatch(self, scan: "ReferenceKey") -> List[str]:
        """Keystone correction moves pixels along the slit, so row-wise factors (and flat-field
        profiles) made with one geometry do not fit scans of the other. ``None`` = ``False``.
        The texts say "a new one" since flat-field profiles use the same comparison."""
        ref_ks, scan_ks = bool(self.keystone_corrected), bool(scan.keystone_corrected)
        if ref_ks == scan_ks:
            return []
        if scan_ks:
            return ["geometry: scan has keystone correction, reference was made without it"
                    " - make a new one with the smile + keystone model"]
        return ["geometry: scan has no keystone correction, reference was made with it"
                " - load the smile + keystone model or make a new one"]

    def _band_mismatch(self, scan: "ReferenceKey") -> List[str]:
        if not self.wavelengths and not scan.wavelengths:
            return []
        if not self.wavelengths or not scan.wavelengths:
            return [f"bands: scan {len(scan.wavelengths) or 'unknown'}, "
                    f"reference {len(self.wavelengths) or 'unknown'}"]
        try:
            check_factor_bands(len(self.wavelengths), self.wavelengths,
                               {"bands": len(scan.wavelengths), "wavelength": np.asarray(scan.wavelengths)})
        except ValueError as exc:
            return [f"bands: {exc}"]
        return []


def _dark_reference(metadata: dict) -> Optional[str]:
    """``<dark file> @ <capture time>`` of a scan's metadata, or ``None`` when either is unknown or
    dark subtraction was off."""
    if _bool(metadata.get("dark_corrected")) is False:
        return None
    file, captured = _text(metadata.get("dark_file")), _text(metadata.get("dark_captured"))
    if not file or not captured:
        return None
    return f"{file} @ {captured}"


# ── a stored reference ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class RadiometricReference:
    """One stored reference (read from its ``reference.json``).

    Attributes:
        id: Folder name, ``ref_<YYYYMMDD>_<HHMMSS>[_<n>]`` (of the save time).
        folder: The reference's folder.
        created_at: When the reference was saved, local ISO 8601 time with UTC offset.
        acquired_at: When the reference target was scanned (the age limit counts from here).
        key: Camera setting of the reference scan.
        reflectance_profile: File name of the reflectance table used.
        avg_window_size: ROI rows averaged per slit row.
        span: First and last slit row covered by the ROI.
        roi_pixels: Pixels in the ROI.
        sensor_temperature_c: Mean sensor temperature of the reference scan, or ``None``.
        info: Further JSON data given when it was saved (QC, table range, source, ...).
    """

    id: str
    folder: Path
    created_at: str
    acquired_at: str
    key: ReferenceKey
    reflectance_profile: str
    avg_window_size: int
    span: Tuple[int, int]
    roi_pixels: int
    sensor_temperature_c: Optional[float] = None
    info: dict = field(default_factory=dict)

    @property
    def created(self) -> datetime.datetime:
        """``created_at`` as an aware datetime (a time without offset is taken as local time)."""
        return _aware(self.created_at)

    @property
    def acquired(self) -> datetime.datetime:
        """``acquired_at`` as an aware datetime."""
        return _aware(self.acquired_at)

    def age_hours(self, now: Optional[datetime.datetime] = None) -> float:
        """Hours from the reference scan (``acquired_at``) to ``now`` (default: the current time);
        negative when ``now`` is earlier."""
        now = now or datetime.datetime.now().astimezone()
        if now.tzinfo is None:
            now = now.astimezone()
        return (now - self.acquired).total_seconds() / 3600.0

    def factors(self) -> np.ndarray:
        """The row-wise factors ``(lines, bands)`` float32.

        Raises:
            ValueError: The stored file does not have the key's slit rows and bands any more.
        """
        factors, wavelengths = load_factor_profile(self.folder / _FACTORS_FILE)
        factors = np.asarray(factors, dtype=np.float32)
        expected = (self.key.lines, len(self.key.wavelengths))
        if factors.shape != expected:
            raise ValueError(f"{self.id}: factors {factors.shape} do not match its key {expected}")
        if wavelengths is not None:
            check_factor_bands(factors.shape[1], wavelengths,
                               {"bands": len(self.key.wavelengths), "wavelength": np.asarray(self.key.wavelengths)})
        return factors

    def roi_mask(self) -> np.ndarray:
        """Boolean ROI ``(lines, samples)`` of the reference scan."""
        return np.load(self.folder / _MASK_FILE).astype(bool)

    def roi_mean_spectrum(self) -> Optional[np.ndarray]:
        """Mean ROI counts per band, or ``None`` when none was stored."""
        path = self.folder / _MEAN_FILE
        return np.load(path) if path.is_file() else None

    def header_fields(self) -> dict:
        """Export-header metadata of a cube calibrated with this reference (``SCAN_HEADER_FIELDS`` keys)."""
        source = self.info.get("source")
        flat = self.info.get("flatfield")                  # the flat-field profile the factors hold
        notes = [str(v) for v in (source, f"flat-field {flat}" if flat else None) if v]
        return {"reflectance_reference": self.reflectance_profile,
                "reflectance_reference_cube": f"{self.id} ({', '.join(notes)})" if notes else self.id,
                "reflectance_reference_rows": f"{self.span[0]}-{self.span[1]}"}

    @classmethod
    def _read(cls, folder: Path) -> "RadiometricReference":
        """Parse and validate ``reference.json``; every malformed value raises ``ValueError`` here,
        so a bad file never fails later (sorting, age) where it would stop the whole store."""
        data = json.loads((folder / _REFERENCE_FILE).read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError(f"{_REFERENCE_FILE} is not a JSON object")
        version = data.get("format_version")
        if version != REFERENCE_FORMAT_VERSION:
            raise ValueError(f"format_version {version!r} is not {REFERENCE_FORMAT_VERSION}")
        if data.get("id") not in (None, folder.name):
            raise ValueError(f"{_REFERENCE_FILE} names {data.get('id')!r}, not its folder {folder.name!r}"
                             " (copied or renamed by hand?)")
        created_at = str(data["created_at"])
        acquired_at = str(data.get("acquired_at") or created_at)
        _aware(created_at), _aware(acquired_at)          # malformed times fail now
        roi = data.get("roi")
        if not isinstance(roi, dict):
            raise ValueError("'roi' is not an object")
        span = roi.get("span")
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ValueError("'roi.span' is not [first, last]")
        key = data.get("key")
        if not isinstance(key, dict):
            raise ValueError("'key' is not an object")
        info = data.get("info") or {}
        if not isinstance(info, dict):
            raise ValueError("'info' is not an object")
        return cls(id=folder.name, folder=folder, created_at=created_at, acquired_at=acquired_at,
                   key=ReferenceKey.from_dict(key),
                   reflectance_profile=str(data.get("reflectance_profile") or ""),
                   avg_window_size=int(data.get("avg_window_size") or 0),
                   span=(int(span[0]), int(span[1])), roi_pixels=int(roi.get("pixels") or 0),
                   sensor_temperature_c=_temperature(data.get("sensor_temperature_c")), info=dict(info))


def _sync_files(folder: Path) -> None:
    """``fsync`` every file in ``folder``, so a power cut cannot leave a complete-looking reference
    with empty data files."""
    for path in folder.iterdir():
        if path.is_file():
            with open(path, "rb+") as handle:
                os.fsync(handle.fileno())


def _aware(text: str) -> datetime.datetime:
    """ISO 8601 text as an aware datetime (a time without offset is taken as local time)."""
    moment = datetime.datetime.fromisoformat(str(text))
    return moment if moment.tzinfo is not None else moment.astimezone()


def reference_problems(reference: RadiometricReference, scan_key: ReferenceKey, *,
                       scan_temperature_c: Optional[float] = None,
                       now: Optional[datetime.datetime] = None,
                       max_age_h: Optional[float] = DEFAULT_MAX_AGE_H,
                       temperature_tolerance_c: Optional[float] = DEFAULT_TEMPERATURE_TOLERANCE_C) -> List[str]:
    """Why ``reference`` must not calibrate the scan of ``scan_key`` (empty = it may).

    Args:
        reference: A stored reference.
        scan_key: :meth:`ReferenceKey.from_scan` of the scan.
        scan_temperature_c: :func:`scan_temperature_c` of the scan, or ``None`` if unknown.
        now: Time of the scan (its ``started_at``; default: now). The reference scan
            (``acquired_at``) must be at most ``max_age_h`` before it, and not after it.
        max_age_h: Age limit in hours; ``None`` = no limit.
        temperature_tolerance_c: Largest temperature difference; ``None`` = not compared.
    """
    problems = list(reference.key.missing()) + reference.key.mismatch(scan_key)
    if max_age_h is not None:
        age = reference.age_hours(now)
        if age > float(max_age_h):
            problems.append(f"reference is {age:.1f} h old (limit {float(max_age_h):g} h)")
        elif age < -1.0 / 60.0:
            problems.append(f"reference was scanned {-age * 60:.0f} min after this scan")
    if temperature_tolerance_c is not None:
        ref_t, scan_t = reference.sensor_temperature_c, _temperature(scan_temperature_c)
        if ref_t is not None and scan_t is not None:
            if abs(scan_t - ref_t) > float(temperature_tolerance_c):
                problems.append(f"sensor temperature: scan {scan_t:.1f} C, reference {ref_t:.1f} C "
                                f"(tolerance {float(temperature_tolerance_c):g} C)")
        elif ref_t is not None or scan_t is not None:
            problems.append(f"sensor temperature: scan {_fmt(scan_t, ' C')}, reference {_fmt(ref_t, ' C')}")
    return problems


@dataclass(frozen=True)
class ReferenceMatch:
    """Result of :meth:`RadiometricReferenceStore.find`.

    Attributes:
        reference: The active reference that may calibrate the scan, or ``None``.
        rejected: ``{reference id: reasons}`` for the active references that may not.
    """

    reference: Optional[RadiometricReference]
    rejected: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when a reference was found."""
        return self.reference is not None

    def reason(self) -> str:
        """One line for the user: the reference used, why none applies, or that none is active.

        When none applies, the closest readable active reference is named (fewest reasons; the
        newest among equals), since its differences are the ones the user can most likely fix;
        an unreadable reference is named only when no readable one was rejected.
        """
        if self.reference is not None:
            return f"calibrated with {self.reference.id}"
        if not self.rejected:
            return "no active radiometric reference"
        def closeness(item):
            reasons = item[1]
            unreadable = bool(reasons) and reasons[0].startswith(UNREADABLE)
            return unreadable, len(reasons)

        ref_id, reasons = min(self.rejected.items(), key=closeness)
        return f"{ref_id} not applied: " + "; ".join(reasons)


# ── the store ──────────────────────────────────────────────────────────────

class RadiometricReferenceStore:
    """References of one camera in one folder (see the module docstring for the layout).

    Args:
        folder: Store folder; created on the first save. The application passes the camera
            profile's ``radiometric_folder``.
    """

    def __init__(self, folder):
        self.folder = Path(folder)

    @property
    def active_path(self) -> Path:
        """``active.json``: the ids of the active references."""
        return self.folder / _ACTIVE_FILE

    # ── writing ──
    def save(self, factors, key: ReferenceKey, *, reflectance_profile: str, avg_window_size: int, mask,
             roi_mean_spectrum=None, sensor_temperature_c: Optional[float] = None,
             info: Optional[dict] = None, acquired_at=None,
             created_at: Optional[datetime.datetime] = None, activate: bool = False) -> RadiometricReference:
        """Store a new reference (and activate it when ``activate``).

        Args:
            factors: Row-wise factors ``(lines, bands)`` (:func:`compute_rowwise_factors_array`).
            key: Camera setting of the reference scan; it must be complete (:meth:`ReferenceKey.missing`).
            reflectance_profile: Reflectance table used (its file name is stored).
            avg_window_size: ROI rows averaged per slit row.
            mask: Boolean ROI ``(lines, samples)`` the factors were fitted on.
            roi_mean_spectrum: Optional mean ROI counts per band.
            sensor_temperature_c: Mean sensor temperature of the reference scan.
            info: Further JSON-serialisable data (QC numbers, table range, source cube, ...).
            acquired_at: When the reference target was scanned: the scan's ``started_at`` (ISO text
                or datetime). The age limit counts from it. Default: ``created_at``.
            created_at: Save time (default: now, local time); names the folder.
            activate: Make it the active reference of its key.

        Returns:
            The stored :class:`RadiometricReference`.

        Raises:
            ValueError: The key is incomplete, or factors / mask / spectrum do not fit the key,
                or no factor is a finite positive number.
            OSError: The folder cannot be written.
        """
        problems = key.missing()
        if problems:
            raise ValueError("Cannot store this radiometric reference: " + "; ".join(problems))
        factors = np.asarray(factors, dtype=np.float32)
        n_bands = len(key.wavelengths)
        if factors.shape != (key.lines, n_bands):
            raise ValueError(f"factors {factors.shape} do not fit the key ({key.lines} rows, {n_bands} bands)")
        finite = np.isfinite(factors)
        if not finite.any() or np.any(factors[finite] <= 0):
            raise ValueError("factors must be finite positive numbers (NaN only for bands outside the table)")
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 2 or mask.shape[0] != key.lines or not mask.any():
            raise ValueError(f"ROI mask {mask.shape} must be (lines={key.lines}, samples) and select pixels")
        if roi_mean_spectrum is not None:
            roi_mean_spectrum = np.asarray(roi_mean_spectrum, dtype=np.float64).ravel()
            if roi_mean_spectrum.size != n_bands:
                raise ValueError(f"ROI mean spectrum has {roi_mean_spectrum.size} values for {n_bands} bands")
        created = created_at or datetime.datetime.now().astimezone()
        if created.tzinfo is None:
            created = created.astimezone()
        if acquired_at is None:
            acquired = created
        elif isinstance(acquired_at, datetime.datetime):
            acquired = acquired_at if acquired_at.tzinfo is not None else acquired_at.astimezone()
        else:
            acquired = _aware(str(acquired_at).replace("Z", "+00:00"))
        info = dict(info or {})
        json.dumps(info)                                   # fail before writing anything
        rows = np.where(mask.any(axis=1))[0]
        record = {
            "format_version": REFERENCE_FORMAT_VERSION,
            "created_at": created.isoformat(timespec="seconds"),
            "acquired_at": acquired.isoformat(timespec="seconds"),
            "key": key.to_dict(),
            "reflectance_profile": os.path.basename(str(reflectance_profile)),
            "avg_window_size": int(avg_window_size),
            "roi": {"span": [int(rows.min()), int(rows.max())], "pixels": int(mask.sum()),
                    "rows": int(rows.size), "samples": int(mask.shape[1])},
            "sensor_temperature_c": _temperature(sensor_temperature_c),
            "info": info,
        }
        self.folder.mkdir(parents=True, exist_ok=True)
        base = "ref_" + created.strftime("%Y%m%d_%H%M%S")
        staging = Path(tempfile.mkdtemp(prefix=f".{base}.", dir=self.folder))
        try:
            save_factor_profile(staging / _FACTORS_FILE, factors, np.asarray(key.wavelengths))
            np.save(staging / _MASK_FILE, mask)
            if roi_mean_spectrum is not None:
                np.save(staging / _MEAN_FILE, roi_mean_spectrum)
            for n in range(1, 1000):
                ref_id = base if n == 1 else f"{base}_{n}"
                target = self.folder / ref_id
                if target.exists():
                    continue
                record["id"] = ref_id
                (staging / _REFERENCE_FILE).write_text(json.dumps(record, indent=2, ensure_ascii=False),
                                                       encoding="utf-8")
                _sync_files(staging)                # on disk before the folder becomes visible
                try:
                    staging.rename(target)
                except FileExistsError:
                    continue
                break
            else:
                raise OSError(f"no free reference name for {base} in {self.folder}")
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        reference = self.load(ref_id)
        logger.info("Radiometric reference %s stored for %s", ref_id, key.label())
        if activate:
            self.activate(ref_id)
        return reference

    def activate(self, ref_id: str) -> List[str]:
        """Make ``ref_id`` the active reference of its key.

        Only readable references with the same key are replaced; one that cannot be read now
        (locked by another program, written by a newer version) stays listed.

        Returns:
            The ids deactivated because they have the same key.

        Raises:
            ValueError: ``ref_id`` is not a readable reference of this store, or ``active.json``
                exists but cannot be parsed (it is left as it is, not overwritten).
            OSError: A file cannot be read or written.
        """
        reference = self.load(ref_id)
        kept, replaced = [], []
        for other_id in self._read_active(strict=True):
            if other_id == reference.id:
                continue
            try:
                other = self.load(other_id)
            except (OSError, ValueError):
                kept.append(other_id)
                continue
            (replaced if not reference.key.mismatch(other.key) else kept).append(other_id)
        self._write_active(kept + [reference.id])
        if replaced:
            logger.info("Radiometric reference %s replaces %s", reference.id, ", ".join(replaced))
        return replaced

    def deactivate(self, ref_id: str) -> None:
        """Remove ``ref_id`` from the active references (the reference itself is kept).

        Raises:
            ValueError: ``active.json`` exists but cannot be parsed (it is left as it is).
        """
        ids = self._read_active(strict=True)
        if ref_id in ids:
            self._write_active([i for i in ids if i != ref_id])

    def _write_active(self, ids: List[str]) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        tmp = self.active_path.with_name(self.active_path.name + ".tmp")
        tmp.write_text(json.dumps({"format_version": REFERENCE_FORMAT_VERSION, "active": ids}, indent=2),
                       encoding="utf-8")
        tmp.replace(self.active_path)

    # ── reading ──
    def _folder_of(self, ref_id: str) -> Path:
        if not isinstance(ref_id, str) or not _REFERENCE_ID.match(ref_id):
            raise ValueError(f"{ref_id!r} is not a radiometric reference id")
        return self.folder / ref_id

    def load(self, ref_id: str) -> RadiometricReference:
        """Read one reference.

        Raises:
            ValueError: ``ref_id`` is malformed, or its ``reference.json`` is not a reference.
            OSError: It cannot be read (e.g. does not exist).
        """
        folder = self._folder_of(ref_id)
        try:
            return RadiometricReference._read(folder)
        except OSError:
            raise
        except KeyError as exc:
            raise ValueError(f"{ref_id}: {_REFERENCE_FILE} lacks {exc}") from None
        except Exception as exc:                        # any malformed value: one bad file, not a crash
            raise ValueError(f"{ref_id}: {exc}") from None

    def references(self) -> List[RadiometricReference]:
        """Every readable reference, newest first (unreadable folders are logged and skipped)."""
        found = []
        if not self.folder.is_dir():
            return found
        for path in self.folder.iterdir():
            if not path.is_dir() or not _REFERENCE_ID.match(path.name):
                continue
            try:
                found.append(self.load(path.name))
            except (OSError, ValueError) as exc:
                logger.warning("Skipping radiometric reference %s: %s", path.name, exc)
        return sorted(found, key=lambda r: (r.created, r.id), reverse=True)

    def active_ids(self) -> List[str]:
        """Ids listed in ``active.json`` (empty when it is missing or unreadable)."""
        try:
            return self._read_active(strict=True)
        except (OSError, ValueError) as exc:
            logger.warning("Cannot read %s: %s; no radiometric reference is active", self.active_path, exc)
            return []

    def _read_active(self, strict: bool) -> List[str]:
        """``active.json``'s valid ids; ``[]`` when missing. With ``strict`` a file that exists but
        cannot be parsed raises ``ValueError`` (so it is not silently overwritten)."""
        try:
            data = json.loads(self.active_path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            return []
        except ValueError as exc:
            if strict:
                raise ValueError(f"{self.active_path} is not valid JSON ({exc}); fix or delete it") from None
            return []
        ids = data.get("active") if isinstance(data, dict) else None
        if not isinstance(ids, list):
            if strict:
                raise ValueError(f"{self.active_path} has no 'active' list; fix or delete it")
            return []
        return [i for i in ids if isinstance(i, str) and _REFERENCE_ID.match(i)]

    def active(self) -> List[RadiometricReference]:
        """The active references that can be read, newest first."""
        found = []
        for ref_id in self.active_ids():
            try:
                found.append(self.load(ref_id))
            except (OSError, ValueError) as exc:
                logger.warning("Active radiometric reference %s cannot be read: %s", ref_id, exc)
        return sorted(found, key=lambda r: (r.created, r.id), reverse=True)

    def status(self, now: Optional[datetime.datetime] = None,
               max_age_h: Optional[float] = DEFAULT_MAX_AGE_H) -> Tuple[str, str]:
        """One-line state of automatic calibration for a status badge.

        Returns:
            ``(state, text)``: ``state`` ``"off"`` (no active reference), ``"expired"`` (the newest
            active reference is older than ``max_age_h``) or ``"on"``; ``text`` names the newest
            active reference's setting and age, e.g. ``Auto radiometric: on - Apollo-M MAX #CAMP7161,
            bin 2, 10 ms, gain 100, 300 bands - scanned 1.2 h ago``.
        """
        active = self.active()
        if not active:
            return "off", "Auto radiometric: off (no active reference)"
        newest = max(active, key=lambda r: (r.acquired, r.id))
        age = newest.age_hours(now)
        more = f" (+{len(active) - 1} more)" if len(active) > 1 else ""
        if max_age_h is not None and age > float(max_age_h):
            return "expired", (f"Auto radiometric: reference expired - {newest.key.label()} - scanned "
                               f"{age:.1f} h ago (limit {float(max_age_h):g} h){more}")
        return "on", f"Auto radiometric: on - {newest.key.label()} - scanned {age:.1f} h ago{more}"

    def find(self, scan_key: ReferenceKey, *, scan_temperature_c: Optional[float] = None,
             now: Optional[datetime.datetime] = None, max_age_h: Optional[float] = DEFAULT_MAX_AGE_H,
             temperature_tolerance_c: Optional[float] = DEFAULT_TEMPERATURE_TOLERANCE_C) -> ReferenceMatch:
        """The newest active reference that may calibrate the scan of ``scan_key``.

        Arguments after ``scan_key`` as in :func:`reference_problems`. An active reference whose
        files cannot be read (``reference.json``, or factors that no longer fit its key) is
        reported in ``rejected`` too, so the returned reference's :meth:`~RadiometricReference.factors`
        has just been read successfully.
        """
        rejected: Dict[str, Tuple[str, ...]] = {}
        for ref_id in self.active_ids():
            try:
                self.load(ref_id)
            except (OSError, ValueError) as exc:
                rejected[ref_id] = (f"{UNREADABLE}: {exc}",)
        for reference in self.active():
            problems = reference_problems(reference, scan_key, scan_temperature_c=scan_temperature_c, now=now,
                                          max_age_h=max_age_h, temperature_tolerance_c=temperature_tolerance_c)
            if not problems:
                try:
                    reference.factors()
                except Exception as exc:            # missing, truncated (EOFError), wrong shape ...
                    rejected[reference.id] = (f"{UNREADABLE}: factors: {exc or type(exc).__name__}",)
                    continue
                return ReferenceMatch(reference, rejected)
            rejected[reference.id] = tuple(problems)
        return ReferenceMatch(None, rejected)
