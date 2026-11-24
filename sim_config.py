# -*- coding: utf-8 -*-
"""Simulation settings (``simulation.json``) and the per-stream acquisition plan.

The simulator re-runs the HSI acquisition pipeline on recorded frames::

    raw sensor frame -> dark subtraction -> hot-pixel repair -> slit flip (raw_inverted)
                     -> Smile remap -> lambda binning (m * camera_bin, c, resolution) -> model

exactly as ``hsi/modules/acquisition/engine.py`` does. Calibration belongs to one physical
camera, so it lives in a profile per camera, named from the model and serial the camera (or
the recording's metadata) reports::

    profiles/<model>-<serial>/dark/dark_b<bin>_e<exp>_g<gain>[_t<temp>].npz
    profiles/<model>-<serial>/smile/smile_model_bin<bin>.npz
    profiles/<model>-<serial>/wavelength/wavelength_bin<bin>.json
    profiles/<model>-<serial>/reflectance/refl_b<bin>_g<gain>.npy (+ .wavelength.npy, .json)

A camera without model/serial, or a profile without that kind of file, uses the HSI layout
per backend (``calibration/dark_frames|smile_maps|wavelength/<backend>/``).

hsi's own profile folders are read in place, first: copy hsi's ``camera_profiles`` folder next
to this file and the camera's ``camera_profiles/<backend>/<model>_<serial>/`` supplies dark,
Smile (+ keystone), wavelength, the active radiometric reference and flat-field profiles with
hsi's rules (``hsi_profiles``).

Every acquisition value is taken from the recorded file's metadata (HDF5 attributes written
by HSI) when ``use_file_metadata`` is on and the file has it, else from this config. Paths in
the JSON are relative to the JSON file.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path

import numpy as np

from modules.dark import DarkFrameLibrary, DarkProcessor
from modules.radiometric import check_factor_bands, load_factor_profile
from modules.smile import SmileRuntime
from modules.spectral_processing import LambdaBinner

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / 'simulation.json'


#: ``source.kind``: replay recorded files, or read a live camera (``camera_source.py``).
SOURCE_KINDS = ('file', 'camera')


@dataclass(frozen=True)
class SourceConfig:
    kind: str = 'file'                   # file (path below) | camera (live: see the 'live' section)
    path: str = 'data'
    pattern: str | None = None           # glob in a folder; None = all *.h5/*.hdf5/*.hdr
    stage: str = 'auto'                  # auto | raw | binned
    envi_scan_axis: str = 'samples'      # ENVI axis that is the scan direction
    dataset: str = 'spectrogram'         # HDF5 dataset name
    cache: str = 'auto'                  # auto | disk (.npy memory map) | ram | stream (decode ahead, no copy)
    cache_dir: str | None = None


@dataclass(frozen=True)
class CameraConfig:
    fps: float = 500.0                   # simulated line rate; 0 = as fast as the pipeline takes
                                         # (file source only: a live camera runs at its own rate)
    overrun: str = 'drop'                # drop: a frame with no free buffer is lost (counted)
                                         # block: the camera waits (lossless throughput test; file
                                         # source only - a live camera cannot wait, it always drops)
    buffer_frames: int = 256             # frame buffers between camera and processing
    loop: bool = True                    # restart the playlist at its end (file source only)
    belt_speed_m_s: float = 0.4          # 24 m/min; only for the mm/line readout


@dataclass(frozen=True)
class LiveConfig:
    """The live camera (``source.kind = "camera"``): which device and how it is read.

    The values that select the calibration files - ``acquisition.camera_backend``, ``camera_bin``,
    ``exposure_ms``, ``gain`` and ``raw_inverted`` - are applied to the camera at start, so the
    dark/Smile/wavelength files match the frames it delivers (see ``camera_source.py``).
    """
    camera_index: int = 0                # device index for GenICam / XIMEA (POA opens its first camera)
    cti_path: str | None = None          # GenICam GenTL producer (.cti); None = the SVS producer found in
                                         # $GENICAM_GENTL64_PATH (sv_gev_tl_x64.cti / libsv_gev_tl_x64.cti)
    sensor_pixel_size: int = 10          # GenICam SensorPixelSize 8/10/12 bits: must be what the darks used
    read_timeout_ms: int = 100           # longest wait for one frame (1..10000); a stop waits for one read
    reader: str = 'process'              # 'process': the camera is read in a process of its own (own GIL,
                                         # camera_process.py); 'thread': in a thread of this process


@dataclass(frozen=True)
class AcquisitionConfig:
    use_file_metadata: bool = True       # recorded HSI attributes override the values below
    camera_backend: str = 'poa'
    camera_bin: int = 2
    raw_inverted: bool = False
    m_nm_per_px_bin1: float = 0.2903
    c_nm: float = 399.791
    resolution_nm: float = 2.0
    wavelength_profile: str | None = None    # None | "auto" | path (HSI wavelength JSON)
    wavelength_range_nm: list | None = None  # [lo, hi]: keep only the bins inside (hsi LambdaBinner.subset)
    exposure_ms: float | None = None     # for dark "auto"
    gain: float | None = None
    temperature_c: float | None = None


@dataclass(frozen=True)
class CorrectionConfig:
    enabled: bool = True                 # False: frames go straight to binning
    backend: str = 'auto'                # auto (GPU if usable, else cpu_fused) | cpu_fused | cpu | cupy_cuda | opencv_cuda
    dark: str = 'none'                   # none | auto | path to .npz/.npy
    hot_pixel: bool = True               # repair the dark file's hot pixels
    smile: str = 'none'                  # none | auto | path to smile_model_bin<N>.npz
    reflectance: str = 'none'            # none | auto | path: counts x factor, last step before the model
    flatfield: str = 'auto'              # newest matching copied profile in profiles/<camera>/flatfield/
    flatfield_enabled: bool = False      # initial checkbox state
    reflectance_dir: str = 'reflectance' # <dir>/<source name>.npy (+ .wavelength.npy), see reflectance/README.md
    profiles_dir: str = 'profiles'       # <dir>/<model>-<serial>/{dark,smile,wavelength,reflectance}/
    camera_profiles_dir: str = 'camera_profiles'  # hsi's camera_profiles folder, copied as is (hsi_profiles.py)
    reflectance_max_age_h: float = 24.0  # older factors are flagged (lamp drift); 0 = never
    reflectance_max_dtemp_c: float = 5.0 # sensor temperature this far from the calibration's is flagged


@dataclass(frozen=True)
class WhiteReferenceConfig:
    """The reflectance standard (Spectralon) the factors are made from at start, like ``source``.

    With ``path`` set and ``corrections.reflectance = "auto"``, ``build_plan`` takes the newest
    file of ``path`` (a file or a folder + ``pattern``) recorded with the data's camera and bin,
    runs it through the data's chain (``reflectance_builder.ensure_white_profile``) and uses the
    factors, cached as ``<reflectance_dir>/from_white/<source name>.npy`` until the white, the
    table or the data's bands change. No matching white: the hand-made ``<source name>.npy``.
    """
    path: str | None = None              # white recording(s): file or folder; None = off
    pattern: str | None = '*.h5'         # glob in a folder
    table: str | None = None             # reference reflectance of the target (.xlsx/.csv: nm, R)
    kind: str = 'band'                   # band: (bands,) from the ROI mean | row: (slit rows, bands)
    rows: object = 'auto'                # slit rows [first, last+1] after the chain | 'auto' (bright rows) | None = all
    frames: object = 'auto'              # scan frames [first, last+1] | 'auto' (frames with the target) | None = all
    avg_window: int = 20                 # row window for kind = row (hsi)
    slit_offset: int | None = None       # white saved as a slit crop: raw sensor row where the crop starts


@dataclass(frozen=True)
class ProcessingConfig:
    workers: int = 4                     # parallel frame workers (order is restored after)
    numba_threads: int | None = 1        # Numba threads per worker (binning/fused kernels); None = all cores
    profile_every_s: float = 5.0         # log stage timings this often (0 = never)
    spatial_bin: int = 1                 # 2 = model input averaged over 2x2 pixels (2 slit rows x 2
                                         # consecutive lines) after reflectance, as HSI-SCAN's offline
                                         # reflectance export ('spatial_bin 2'); 1 = every pixel


@dataclass(frozen=True)
class DisplayConfig:
    gui_hz: float = 30.0
    canvas_width: int = 4096             # belt-map history (lines / stride) kept for display
    canvas_line_stride: int = 1          # draw every N-th line (display only; all lines are counted)
    line_aspect: float = 1.0             # width of one scan line in slit pixels (1 = square: a round
                                         # target stays round; measure: target width in lines / height in rows)
    raw_zone_fraction: float = 1 / 3    # right part of the belt map showing the newest lines before
                                         # classification (grey: model-input intensity); 0 = off
    spectrum_history_mb: float = 512.0   # spectrum preview (click on the belt map): memory for the
                                         # newest model-input lines, used only while the panel is shown
    theme: str = 'dark'                  # themes/<name>.json or a path to a theme JSON
    hidden_classes: tuple = ('bg', 'background')   # Materials chart: hidden until ticked
                                         # (class names, case-insensitive)


@dataclass(frozen=True)
class OpticsConfig:
    """Spectrograph geometry (documentation and derived figures; no processing step uses it yet)."""
    slit_width_um: float | None = 50.0   # entrance slit width
    pixel_size_um: float | None = 5.0    # sensor pixel pitch (square), unbinned
    spectrograph_magnification: float = 1.0   # slit -> sensor (1:1 for a typical imaging spectrograph)
    # Belt installation (production): fore-optics magnification sensor/object and the belt width the
    # slit covers. Pixel on the belt = pixel_size x bin / lens_magnification (unless belt_pixel_mm).
    lens_magnification: float | None = 0.0123
    slit_fov_mm: float | None = 500.0    # slit length on the belt
    belt_pixel_mm: float | None = None   # override; None = from lens_magnification
    recording_pixel_mm: float | None = 0.1264  # lab recording (240 mm): 1" Spectralon = 201 px

    def belt_pixel(self, camera_bin: int) -> float | None:
        """One slit pixel on the belt (mm) at ``camera_bin``."""
        if self.belt_pixel_mm:
            return float(self.belt_pixel_mm)
        if self.lens_magnification and self.pixel_size_um:
            return self.pixel_size_um * camera_bin / 1000.0 / self.lens_magnification
        return None


@dataclass(frozen=True)
class ModelConfig:
    # a joblib artifact or an HSI MLOps package (.hsipkg / the .zip around it);
    # "none": speed-only stand-in model (no trained weights)
    path: str = 'models/rf_coco_model.joblib'
    # folders whose models the GUI offers for switching while running (only these)
    folders: tuple = ('models',)
    band_tolerance_nm: float = 0.5       # max distance between a model band and an input band
    # A model whose bands are further away than that (the wavelength calibration was redone after
    # it was trained) is fed the input interpolated onto its bands; false: such a model is refused.
    resample: bool = True
    # .hsipkg (MLC) only: added to the diagonal of every class covariance. 0 = the model exactly as
    # trained. Raise it (0.03 .. 1) when live data falls to Background: the input is noisier than
    # the training cubes. Not the validated model any more: check the result on known material.
    mlc_reg_covar: float = 0.0


@dataclass(frozen=True)
class SimulationConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    corrections: CorrectionConfig = field(default_factory=CorrectionConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optics: OpticsConfig = field(default_factory=OpticsConfig)
    live: LiveConfig = field(default_factory=LiveConfig)
    white_reference: WhiteReferenceConfig = field(default_factory=WhiteReferenceConfig)
    base_dir: str = str(ROOT)            # folder relative paths resolve against

    def resolve(self, value: str | None) -> Path | None:
        if value in (None, '', 'none', 'auto'):
            return None
        path = Path(value)
        return path if path.is_absolute() else Path(self.base_dir) / path

    @property
    def calibration_dir(self) -> Path:
        return Path(self.base_dir) / 'calibration'

    @property
    def live_camera(self) -> bool:
        """True when the frames come from a live camera (``source.kind = "camera"``)."""
        return self.source.kind == 'camera'

    def validate(self) -> 'SimulationConfig':
        if self.source.kind not in SOURCE_KINDS:
            raise ValueError(f"source.kind must be one of {SOURCE_KINDS}, got {self.source.kind!r}")
        if self.live_camera:
            from modules.camera.factory import BACKENDS as camera_backends
            acq = self.acquisition
            if str(acq.camera_backend).lower() not in camera_backends:
                raise ValueError(f"source.kind = camera: acquisition.camera_backend must be one of "
                                 f"{camera_backends}, got {acq.camera_backend!r}")
            if acq.exposure_ms is None or acq.gain is None:
                raise ValueError("source.kind = camera: set acquisition.exposure_ms and acquisition.gain "
                                 "(they are applied to the camera and select the dark frame)")
            if not acq.exposure_ms > 0:
                raise ValueError("acquisition.exposure_ms must be > 0")
            if self.live.sensor_pixel_size not in (8, 10, 12):
                raise ValueError("live.sensor_pixel_size must be 8, 10 or 12")
            if not 1 <= self.live.read_timeout_ms <= 10000:
                raise ValueError("live.read_timeout_ms must be 1..10000 (a stop waits for one read)")
            if self.live.reader not in ('process', 'thread'):
                raise ValueError(f"live.reader must be 'process' or 'thread', got {self.live.reader!r}")
            if self.live.camera_index < 0:
                raise ValueError("live.camera_index must be >= 0")
        if self.source.stage not in ('auto', 'raw', 'binned'):
            raise ValueError(f"source.stage: {self.source.stage!r}")
        if self.camera.fps < 0:
            raise ValueError("camera.fps must be >= 0")
        if self.camera.overrun not in ('drop', 'block'):
            raise ValueError(f"camera.overrun must be 'drop' or 'block', got {self.camera.overrun!r}")
        if self.camera.buffer_frames < 2:
            raise ValueError("camera.buffer_frames must be >= 2")
        if self.acquisition.camera_bin < 1:
            raise ValueError("acquisition.camera_bin must be >= 1")
        if not isinstance(self.model.resample, bool):       # "False"/"off" are text, and text is true
            raise ValueError(f"model.resample must be true or false, got {self.model.resample!r}")
        reg = self.model.mlc_reg_covar
        if isinstance(reg, bool) or not isinstance(reg, (int, float)) or not 0 <= reg < float('inf'):
            raise ValueError(f"model.mlc_reg_covar must be a number >= 0, got {reg!r}")
        if self.acquisition.resolution_nm <= 0 or self.acquisition.m_nm_per_px_bin1 <= 0:
            raise ValueError("acquisition.m_nm_per_px_bin1 and resolution_nm must be > 0")
        if self.corrections.backend not in ('auto', 'cpu_fused', 'cpu', 'cupy_cuda', 'opencv_cuda'):
            raise ValueError(f"corrections.backend: {self.corrections.backend!r}")
        if not 1 <= self.processing.workers <= 32:
            raise ValueError("processing.workers must be 1..32")
        if self.processing.spatial_bin not in (1, 2):
            raise ValueError("processing.spatial_bin must be 1 (every pixel) or 2 (2x2 mean)")
        if self.display.canvas_line_stride < 1 or self.display.canvas_width < 16:
            raise ValueError("display.canvas_line_stride >= 1 and canvas_width >= 16")
        if not self.display.line_aspect > 0:
            raise ValueError("display.line_aspect must be > 0")
        if not self.display.spectrum_history_mb > 0:
            raise ValueError("display.spectrum_history_mb must be > 0")
        if not 0.0 <= self.display.raw_zone_fraction < 1.0:
            raise ValueError("display.raw_zone_fraction must be in [0, 1)")
        white = self.white_reference
        if white.path not in (None, ''):
            if not white.table:
                raise ValueError("white_reference.table is required with white_reference.path")
            if white.kind not in ('band', 'row'):
                raise ValueError(f"white_reference.kind must be 'band' or 'row', got {white.kind!r}")
            for key in ('rows', 'frames'):
                value = getattr(white, key)
                if not (value in (None, 'auto') or (isinstance(value, (list, tuple)) and len(value) == 2
                                                    and 0 <= int(value[0]) < int(value[1]))):
                    raise ValueError(f"white_reference.{key} must be 'auto', null or [first, last+1]")
            if white.avg_window < 1:
                raise ValueError("white_reference.avg_window must be >= 1")
        return self


_SECTION_CLASSES = {'source': SourceConfig, 'camera': CameraConfig, 'acquisition': AcquisitionConfig,
                    'corrections': CorrectionConfig, 'processing': ProcessingConfig,
                    'display': DisplayConfig, 'model': ModelConfig, 'optics': OpticsConfig,
                    'live': LiveConfig, 'white_reference': WhiteReferenceConfig}


def parse_set_overrides(items) -> dict:
    """``--set SECTION.KEY=VALUE`` strings -> ``{section: {key: value}}``, the ``explicit`` of :func:`load_config`.

    Values are read as JSON (``2``, ``true``, ``null``, ``[900, 1700]``), else kept as text;
    ``null`` sets the key to ``None`` (e.g. ``--set live.cti_path=null``).

    Raises:
        ValueError: an item is not ``SECTION.KEY=VALUE``.
    """
    overrides = {}
    for item in items:
        key, eq, text = item.partition('=')
        section, dot, name = key.partition('.')
        if not eq or not dot or not section or not name:
            raise ValueError(f"--set expects SECTION.KEY=VALUE, got {item!r}")
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = text
        overrides.setdefault(section, {})[name] = value
    return overrides


def load_config(path=None, explicit: dict | None = None, **overrides) -> SimulationConfig:
    """Read ``simulation.json``; unknown sections/keys are errors (typos must not pass silently).

    Args:
        path: JSON file (default ``simulation.json`` beside this module).
        explicit: ``{section: {key: value}}`` applied last and as given, ``None`` included
            (``--set``, see :func:`parse_set_overrides`).
        overrides: ``section={key: value}`` dicts applied on top of the file (command-line
            flags); a ``None`` value means "flag not given" and keeps the file's value.
    """
    path = Path(path or DEFAULT_CONFIG_PATH).resolve()
    data = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    if not path.is_file():
        logger.warning("No %s; using built-in defaults", path)
    for name, values in overrides.items():
        data.setdefault(name, {}).update({k: v for k, v in values.items() if v is not None})
    for name, values in (explicit or {}).items():
        data.setdefault(name, {}).update(values)
    unknown = set(data) - set(_SECTION_CLASSES)
    if unknown:
        raise ValueError(f"{path.name}: unknown section(s) {sorted(unknown)}")
    sections = {}
    for name, cls in _SECTION_CLASSES.items():
        values = data.get(name, {})
        allowed = {f.name for f in fields(cls)}
        bad = set(values) - allowed
        if bad:
            raise ValueError(f"{path.name}: unknown key(s) {sorted(bad)} in '{name}' (allowed: {sorted(allowed)})")
        sections[name] = cls(**values)
    config = SimulationConfig(**sections, base_dir=str(path.parent)).validate()
    ensure_profile_structure(config)
    return config


# --------------------------------------------------------------------- acquisition plan ---

#: Processing steps in hsi order; the GUI shows one switch per step. ``dark`` is dark
#: subtraction + hot-pixel repair together, like hsi's ``DarkProcessor.set_dark_enabled``
#: (whether hot pixels are repaired is ``corrections.hot_pixel``). ``reflectance`` is the
#: last step before inference (hsi ``modules.radiometric``: counts x factor). ``keystone``
#: exists only with a smile + keystone model: off = the model's smile alone (hsi
#: ``AcquisitionEngine.set_geometry``); it never applies while ``smile`` is off. The switches
#: change only the remap: the slit rows and bands kept (see ``AcquisitionPlan.keystone_crop``,
#: ``smile_border``) are fixed by the loaded model, so the model input keeps its size.
STEP_NAMES = ('dark', 'flip', 'smile', 'keystone', 'binning', 'flatfield', 'reflectance')


@dataclass
class AcquisitionPlan:
    """Everything the frame workers need, resolved once per stream.

    ``provenance`` maps each setting to where it came from (``file``, ``config``,
    ``calibration/...``), so the log and GUI can show why a value was used.
    """
    stage: str
    frame_shape: tuple
    camera_backend: str = ''
    camera_bin: int = 1
    raw_inverted: bool = False
    m_nm_per_px_bin1: float = 0.0
    c_nm: float = 0.0
    resolution_nm: float = 0.0
    correction_backend: str = 'cpu'
    corrections_enabled: bool = False
    dark: DarkProcessor | None = None             # as configured (dark + hot pixel)
    smile: object = None                         # as configured
    dark_mean: np.ndarray | None = None          # loaded references, also when offered but off
    hot_map: np.ndarray | None = None
    smile_model: object = None
    #: ``(model, model.smile_only())`` built on first use with the keystone step off (see geometry_model)
    smile_only_view: tuple = field(default=(None, None), repr=False)
    #: ``(top, bottom)`` slit rows left out after the chain: the keystone margin of the loaded
    #: smile + keystone model (hsi crops them at acquisition). Fixed for the plan, whatever the
    #: Smile / Keystone switches: the model input, canvas and factors keep one size.
    keystone_crop: tuple = (0, 0)
    #: ``(left, right)`` spectral columns the loaded Smile model zero-fills at the slit ends; the
    #: bins touching them are left out of ``binner`` (hsi ``_smile_trim``)
    smile_border: tuple = (0, 0)
    reflectance_factors: np.ndarray | None = None   # (slit, bands) or (bands,) float32, also when offered off
    reflectance_placeholder: bool = False           # factors marked placeholder in reflectance/<name>.json
    # Flat-field profile (its label) the reflectance factors already contain ("Flat-field +
    # reference" calibration): the Flat-field step must not run again, but it is active.
    reflectance_flatfield: str | None = None
    flatfield_profile: object = None               # portable HSI profile; immutable per-frame snapshot
    exposure_ms: float | None = None             # of the frames: a white of another exposure is scaled to it
    gain: float | None = None
    temperature_c: float | None = None           # sensor temperature (selects the dark; flags reflectance)
    camera_id: str | None = None                 # '<model>-<serial>' (None: not reported)
    profile_dir: Path | None = None              # profiles/<camera_id> (may not exist yet)
    hsi_profile_dir: Path | None = None          # camera_profiles/<backend>/<model>_<serial> (hsi), if copied
    smile_path: Path | None = None               # file of smile_model (hsi key: its name)
    dark_reference: str | None = None            # hsi key '<dark file> @ <captured>' of dark_mean, if known
    #: switches the reflectance factors were made with (an hsi reference): another setting of
    #: Dark / Flip / Smile / Keystone turns Reflectance off, as hsi refuses the reference
    reflectance_requires: dict = field(default_factory=dict)
    reflectance_max_age_h: float = 0.0           # from corrections (0 = no age check)
    reflectance_max_dtemp_c: float = 0.0
    #: per step: {'available': bool, 'on': bool, 'why': str} for dark (+hot pixel), flip, smile, binning
    steps: dict = field(default_factory=dict)
    binner: LambdaBinner | None = None
    wavelengths: np.ndarray | None = None        # nm of each column the model receives
    columns: np.ndarray | None = None            # model band -> input column (set by the pipeline)
    provenance: dict = field(default_factory=dict)

    def dark_processor(self, on: bool) -> DarkProcessor:
        """hsi ``DarkProcessor`` with dark + hot pixel enabled (``on``) or disabled."""
        cache = self.__dict__.setdefault('_dark_variants', {})
        on = bool(on and self.dark_mean is not None)
        if on not in cache:
            processor = DarkProcessor()
            if on:
                processor.set_dark_reference(self.dark_mean, self.hot_map)
            else:
                processor.set_dark_enabled(False)
            cache[on] = processor
        return cache[on]

    @property
    def n_columns(self) -> int:
        return self.binner.n_bins if self.binner is not None else int(self.frame_shape[1])

    @property
    def slit_rows(self) -> slice:
        """Slit rows of a processed frame the model receives (all but :attr:`keystone_crop`)."""
        top, bottom = self.keystone_crop
        return slice(top, int(self.frame_shape[0]) - bottom)

    @property
    def n_slit(self) -> int:
        """Slit rows the model receives per line."""
        top, bottom = self.keystone_crop
        return int(self.frame_shape[0]) - top - bottom

    def summary(self) -> str:
        lines = [f"stage={self.stage} frame={self.frame_shape}"]
        for key, (value, origin) in self.provenance.items():
            lines.append(f"  {key:<18} {value}  [{origin}]")
        return "\n".join(lines)


def _pick(meta: dict, keys, fallback, use_meta: bool, cast=lambda v: v, origin: str = 'file'):
    """``(value, origin)`` from the first metadata key present, else ``(fallback, 'config')``.

    ``origin`` names where the metadata came from: ``file`` (recorded attributes) or ``camera``
    (what a live camera reports after its settings were applied).
    """
    if use_meta:
        for key in keys:
            if key in meta and meta[key] is not None:
                return cast(meta[key]), f"{origin}:{key}"
    return fallback, 'config'


def build_plan(config: SimulationConfig, source) -> AcquisitionPlan:
    """Resolve calibration and corrections for a stream whose first file is ``source``.

    Raises:
        ValueError / FileNotFoundError: an explicitly configured file is missing or does not
            match the frames (an ``auto`` lookup that finds nothing only logs and bypasses).
    """
    shape = source.frame_shape
    plan = AcquisitionPlan(stage=source.stage, frame_shape=shape)
    plan.reflectance_max_age_h = float(config.corrections.reflectance_max_age_h or 0)
    plan.reflectance_max_dtemp_c = float(config.corrections.reflectance_max_dtemp_c or 0)
    prov = plan.provenance
    if source.stage == 'binned':
        plan.wavelengths = source.wavelengths
        prov['input'] = ('already corrected and binned: no dark/hot pixel/flip/Smile/binning', source.name)
        plan.steps = {name: {'available': False, 'on': False, 'why': 'input is already corrected and binned'}
                      for name in STEP_NAMES}
        _reflectance_step(config, plan, source)
        if source.wavelengths is not None:
            prov['wavelength'] = (f"{source.wavelengths[0]:.2f}-{source.wavelengths[-1]:.2f} nm, "
                                  f"{source.wavelengths.size} bands", 'file')
        return plan

    acq, cor = config.acquisition, config.corrections
    # A live camera's metadata is what the camera reports with the configured settings applied:
    # always the truth for its frames, whatever use_file_metadata says (that switch is for files).
    origin = getattr(source, 'metadata_origin', 'file')
    meta, use = source.metadata, acq.use_file_metadata or origin == 'camera'

    def pick(keys, fallback, cast=lambda v: v):
        return _pick(meta, keys, fallback, use, cast, origin)

    plan.camera_backend, o = pick(('camera_backend',), acq.camera_backend, str)
    prov['camera_backend'] = (plan.camera_backend, o)
    # identity, not a setting: always from the metadata (use_file_metadata does not apply)
    plan.camera_id = camera_profile_name(meta.get('camera_model'), meta.get('camera_serial'))
    plan.profile_dir = profiles_root(config) / plan.camera_id if plan.camera_id else None
    ensure_profile_structure(config, plan.camera_id)
    prov['camera profile'] = (
        (f"{plan.camera_id}" + ('' if plan.profile_dir.is_dir() else ' (no profile folder yet)'))
        if plan.camera_id else 'none: no camera_model/camera_serial (calibration/<backend>/ is used)', origin)
    from hsi_profiles import camera_profiles_root, find_camera_profile, profile_label
    plan.hsi_profile_dir = find_camera_profile(config, plan.camera_backend, meta.get('camera_model'),
                                               meta.get('camera_serial'))
    if plan.hsi_profile_dir is not None:
        prov['hsi camera profile'] = (profile_label(plan.hsi_profile_dir), 'copied hsi camera_profiles')
    elif plan.camera_id and camera_profiles_root(config).is_dir():
        prov['hsi camera profile'] = (f"none for {meta.get('camera_model')} {meta.get('camera_serial')} in "
                                      f"{camera_profiles_root(config).name}/", 'copied hsi camera_profiles')
    plan.camera_bin, o = pick(('camera_bin',), acq.camera_bin, int)
    prov['camera_bin'] = (plan.camera_bin, o)
    plan.raw_inverted, o = pick(('raw_inverted',), acq.raw_inverted, bool)
    prov['raw_inverted'] = (plan.raw_inverted, o)
    exposure, o_e = pick(('exposure_ms',), acq.exposure_ms, float)
    gain, o_g = pick(('gain',), acq.gain, float)
    plan.exposure_ms, plan.gain = exposure, gain
    # the dark the recording used was captured at dark_temperature_c (HSI picks the nearest one)
    temperature, _ = pick(('temperature_c', 'dark_temperature_c', 'sensor_temperature_c_start', 'temperature'),
                          acq.temperature_c, float)
    plan.temperature_c = temperature

    # Wavelength axis: recorded values > wavelength profile > config numbers.
    m1, c, o_m = acq.m_nm_per_px_bin1, acq.c_nm, 'config'
    profile = _wavelength_profile_path(config, plan)
    if profile is None and origin == 'camera' and acq.wavelength_profile == 'auto':
        # A recording carries its own m/c; a live camera does not. The config numbers belong to
        # one camera, so a missing profile must not silently put another camera's axis under the model.
        folder, _ = calibration_folder(config, plan, 'wavelength')
        raise ValueError(
            f"no wavelength profile {folder}/wavelength_bin{plan.camera_bin}.json "
            f"for the live {plan.camera_backend} camera at bin {plan.camera_bin}: add it (hsi wavelength "
            "calibration), set acquisition.wavelength_profile to a file, or to \"none\" to use "
            "acquisition.m_nm_per_px_bin1 / c_nm on purpose")
    if profile is not None:
        m1, c, o_m = _load_wavelength_profile(profile, plan.camera_bin, shape, plan)
    if use and meta.get('m_nm_per_px_bin1') is not None and meta.get('c_nm') is not None:
        m1, c, o_m = float(meta['m_nm_per_px_bin1']), float(meta['c_nm']), 'file:m_nm_per_px_bin1,c_nm'
    plan.m_nm_per_px_bin1, plan.c_nm = float(m1), float(c)
    prov['wavelength m,c'] = (f"m={m1:.6g} nm/px (bin 1) x bin {plan.camera_bin}, c={c:.6g} nm", o_m)
    plan.resolution_nm, o = pick(('resolution_nm',), acq.resolution_nm, float)
    prov['resolution_nm'] = (plan.resolution_nm, o)

    full = LambdaBinner(m=plan.m_nm_per_px_bin1 * plan.camera_bin, c=plan.c_nm,
                        step=plan.resolution_nm).fit(shape[1])
    # Wavelength range, as hsi's engine: the full grid is fitted, the bins inside the range are
    # kept (LambdaBinner.subset, bit-identical to the same bins of the full grid).
    recorded = meta.get('wavelength_range_bands') if use else None
    if recorded:
        k0, k1, n_full = (int(v) for v in recorded)
        if full.n_bins != n_full:
            raise ValueError(f"the recording kept bins {k0}:{k1} of a {n_full}-bin grid, but its m/c/resolution "
                             f"give {full.n_bins} bins: wavelength calibration differs from the recording")
        plan.binner, origin = full.subset(k0, k1), 'file:wavelength_range_bands'
        requested = meta.get('wavelength_range_requested_nm')
        range_text = f"bins {k0}:{k1} of {n_full}" + (f" (requested {requested[0]:g}-{requested[1]:g} nm)"
                                                      if requested else '')
    elif acq.wavelength_range_nm:
        lo, hi = (float(v) for v in acq.wavelength_range_nm)
        k0, k1 = full.range_bins(lo, hi)
        if k1 - k0 < 1:
            raise ValueError(f"no bin lies inside wavelength_range_nm {lo:g}-{hi:g} nm")
        plan.binner, origin = full.subset(k0, k1), 'config'
        range_text = f"bins {k0}:{k1} of {full.n_bins} ({lo:g}-{hi:g} nm)"
    else:
        k0, k1 = 0, full.n_bins
        plan.binner, origin, range_text = full, 'computed', 'full grid'
    plan.wavelengths = plan.binner.lambda_out.astype(np.float32)
    prov['binning'] = (f"{shape[1]} px -> {plan.binner.n_bins} bins "
                       f"{plan.wavelengths[0]:.2f}-{plan.wavelengths[-1]:.2f} nm, {range_text}", origin)

    plan.corrections_enabled = cor.enabled
    plan.correction_backend = cor.backend
    flip_why = f"raw_inverted={plan.raw_inverted} [{prov['raw_inverted'][1]}]"
    plan.steps = {'flip': {'available': True, 'on': plan.raw_inverted, 'why': flip_why},
                  'binning': {'available': False, 'on': True,
                              'why': 'always on: the model needs wavelength bins'}}
    if not cor.enabled:
        prov['corrections'] = ('disabled (frames go straight to binning)', 'config')
        for name in ('dark', 'smile', 'flatfield', 'reflectance'):
            plan.steps[name] = {'available': False, 'on': False, 'why': 'corrections.enabled = false'}
        return plan

    # Dark / hot pixel: the configured reference, or (with "none") an auto match offered as off.
    mean, hot, label, record = _dark_reference(config, plan, exposure, gain, temperature, o_e, o_g)
    dark_on = mean is not None
    if mean is None and cor.dark in (None, '', 'none'):
        offered_mean, offered_hot, offered, offered_record = _dark_reference(
            config, plan, exposure, gain, temperature, o_e, o_g, choice='auto', quiet=True)
        if offered_mean is not None:
            mean, hot, label, record = offered_mean, offered_hot, f"none (available: {offered})", offered_record
    from hsi_profiles import dark_reference_text, hsi_profile_name
    plan.dark_reference = dark_reference_text(record, plan.hsi_profile_dir, hsi_profile_name(
        meta.get('camera_model'), meta.get('camera_serial'))) if record else None
    hot_ok = hot is not None and np.count_nonzero(hot) > 0
    n_hot = int(np.count_nonzero(hot)) if hot_ok else 0
    plan.dark_mean = mean
    plan.hot_map = hot if hot_ok and cor.hot_pixel else None
    plan.dark = plan.dark_processor(dark_on)
    suffix = f" ({n_hot} hot pixels{'' if cor.hot_pixel else ', repair off'})" if dark_on else ''
    prov['dark'] = (label + suffix, cor.dark)
    no_dark = 'no dark reference loaded (corrections.dark: auto or a file; auto needs exposure/gain)'
    hot_text = (f"{n_hot} hot pixels repaired" if hot_ok and cor.hot_pixel else
                'hot-pixel repair off (corrections.hot_pixel)' if hot_ok else 'no hot-pixel map in the file')
    plan.steps['dark'] = {'available': mean is not None, 'on': dark_on,
                          'why': f"{label}; {hot_text}" if mean is not None else no_dark}

    # Smile: the configured model, or (with "none") smile_model_bin<N>.npz offered as off.
    plan.smile, label = _smile_model(config, plan)
    smile_on = plan.smile is not None
    model = plan.smile
    if model is None and cor.smile in (None, '', 'none'):
        offered_model, offered = _smile_model(config, plan, choice='auto', quiet=True)
        if offered_model is not None:
            model, label = offered_model, f"none (available: {offered})"
    plan.smile_model = model
    prov['smile'] = (label, cor.smile)
    plan.steps['smile'] = {'available': model is not None, 'on': smile_on,
                           'why': label if model is not None else
                           f"no Smile model for bin {plan.camera_bin} with frame size {shape}"}
    # Geometry window of the loaded model (also when offered off): fixed for the plan, so the
    # Smile / Keystone switches never change the model input's size.
    _trim_smile_border(plan, model, full, k0, k1, range_text, origin)
    has_keystone = bool(getattr(model, 'has_keystone', False))
    plan.keystone_crop = keystone_margin(model)
    if has_keystone:
        top, bottom = plan.keystone_crop
        prov['slit rows'] = (f"{plan.n_slit} of {shape[0]}: keystone margin {top} top / {bottom} bottom cropped",
                             'smile model')
        keystone_why = (f"Smile + keystone model: rows remapped too; margin rows {top} top / {bottom} bottom "
                        f"cropped ({plan.n_slit} of {shape[0]} slit rows). Off = the model's smile alone; "
                        "the crop stays, so the map and factors keep their size")
    else:
        keystone_why = ('the Smile model has no keystone (smile only)' if model is not None
                        else 'no Smile model')
    plan.steps['keystone'] = {'available': has_keystone, 'on': has_keystone, 'why': keystone_why}
    _reflectance_step(config, plan, source)
    from flatfield_profile import configure_flatfield
    configure_flatfield(config, plan)
    prov['correction backend'] = (cor.backend, 'config')
    optics = optics_summary(config.optics, plan.camera_bin, plan.m_nm_per_px_bin1, plan.resolution_nm)
    if optics:
        prov['optics'] = (optics, 'config:optics')
    return plan


def optics_summary(optics, camera_bin: int, m_nm_per_px_bin1: float, resolution_nm: float) -> str:
    """Slit image in sensor pixels and the slit-limited spectral bandwidth, for the info panel.

    The slit image on the sensor is ``slit_width x magnification``; across the dispersion it covers
    ``slit / pixel`` sensor pixels, i.e. ``/ camera_bin`` binned pixels, and spans
    ``slit / pixel x m_bin1`` nm (the bandwidth before optical blur). Bins much narrower than that
    only oversample; much wider ones lose resolution.
    """
    if not optics.slit_width_um or not optics.pixel_size_um:
        return ''
    belt = ''
    pixel = optics.belt_pixel(camera_bin)
    if pixel:
        belt = f"; on the belt: pixel {pixel:.3g} mm"
        if optics.lens_magnification:
            belt += (f", slit width {optics.slit_width_um / 1000.0 / optics.lens_magnification:.3g} mm "
                     f"along the belt (M {optics.lens_magnification:g})")
        if optics.slit_fov_mm:
            belt += f", {optics.slit_fov_mm:g} mm field = {optics.slit_fov_mm / pixel:.0f} px"
    slit_px = optics.slit_width_um * optics.spectrograph_magnification / optics.pixel_size_um
    bandwidth = slit_px * m_nm_per_px_bin1
    return (f"slit {optics.slit_width_um:g} µm, pixel {optics.pixel_size_um:g} µm "
            f"({optics.pixel_size_um * camera_bin:g} µm at bin {camera_bin}): slit image {slit_px:g} px "
            f"= {slit_px / camera_bin:g} binned px -> slit-limited bandwidth ~{bandwidth:.2f} nm; "
            f"bins {resolution_nm:g} nm = {resolution_nm / bandwidth:.2f} x that{belt}")


#: calibration kind in a profile -> its folder in the HSI layout ``calibration/<folder>/<backend>``
CALIBRATION_KINDS = {'dark': 'dark_frames', 'smile': 'smile_maps', 'wavelength': 'wavelength'}


def camera_profile_name(model, serial) -> str | None:
    """``<model>-<serial>`` as a folder name (spaces and other unsafe characters become ``-``)."""
    import re
    if isinstance(model, bytes):
        model = model.decode(errors='replace')
    if isinstance(serial, bytes):
        serial = serial.decode(errors='replace')
    if model is None or serial is None or not re.search(r'[A-Za-z0-9]', str(model)) \
            or not re.search(r'[A-Za-z0-9]', str(serial)):
        return None                                   # both are needed: a blank one names no unit
    name = re.sub(r'[^A-Za-z0-9._]+', '-', f"{str(model).strip()}-{str(serial).strip()}").strip('-.')
    return name or None


def profiles_root(config) -> Path:
    """``corrections.profiles_dir`` as a path."""
    return config.resolve(config.corrections.profiles_dir) or Path(config.base_dir) / 'profiles'


def ensure_profile_structure(config, camera_id=None) -> None:
    """Create portable flat-field folders on startup, without altering calibration lookup."""
    root = profiles_root(config)
    root.mkdir(parents=True, exist_ok=True)
    folders = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')]
    if camera_id:
        folders.append(root / camera_id)
    for folder in folders:
        (folder / 'flatfield').mkdir(parents=True, exist_ok=True)


def calibration_folders(config, plan, kind: str) -> list[tuple[Path, str]]:
    """Folders of one calibration ``kind`` (dark, smile, wavelength) for the plan's camera, in the
    order they are searched; a file is taken from the first folder that has it.

    1. hsi's camera folder ``camera_profiles/<backend>/<model>_<serial>/<kind>`` (copied from hsi);
       like hsi, a file it does not have is looked up in the next folders, first in hsi's older
       per-backend folder (``dark_frames|smile_maps|calibration/wavelength/<backend>`` next to
       ``camera_profiles``) when it was copied too.
    2. This project's profile ``profiles/<model>-<serial>/<kind>``. When it exists, this
       project's per-backend folder (3) is not searched (never another camera's files).
    3. The HSI layout per backend ``calibration/<folder>/<backend>`` (camera unknown).
    """
    found = []
    hsi = getattr(plan, 'hsi_profile_dir', None)
    if hsi is not None and (hsi / kind).is_dir():
        from hsi_profiles import profile_label
        found.append((hsi / kind, f"hsi {profile_label(hsi)}"))
    if hsi is not None:
        # hsi's own fallback: its older per-backend folder, when copied next to camera_profiles
        # (hsi searches exactly these two, so they come before this project's folders)
        from hsi_profiles import HSI_LEGACY_FOLDERS
        older = hsi.parent.parent.parent / HSI_LEGACY_FOLDERS[kind] / str(plan.camera_backend).lower()
        if older.is_dir():
            found.append((older, f"hsi {HSI_LEGACY_FOLDERS[kind]}/{str(plan.camera_backend).lower()}"))
    if plan.profile_dir is not None and (plan.profile_dir / kind).is_dir():
        found.append((plan.profile_dir / kind, f"profile {plan.camera_id}"))
        return found
    legacy = CALIBRATION_KINDS[kind]
    folder = config.calibration_dir / legacy / plan.camera_backend
    if plan.profile_dir is None and not folder.is_dir():
        # No model/serial (old recording, camera without a serial) and the backend's files were
        # moved into a profile: the only profile of this backend is the camera they came from.
        only = backend_profile(config, plan.camera_backend)
        if only is not None and (only / kind).is_dir():
            found.append((only / kind, f"profile {only.name} (only {plan.camera_backend} profile; no serial)"))
            return found
    if all(folder.resolve() != f.resolve() for f, _ in found):
        found.append((folder, f"calibration/{legacy}/{plan.camera_backend}"))
    return found


def calibration_folder(config, plan, kind: str) -> tuple[Path, str]:
    """The first folder of :func:`calibration_folders` (where a new file of ``kind`` belongs)."""
    return calibration_folders(config, plan, kind)[0]


def _first_file(config, plan, kind: str, name: str):
    """``(path, label)`` of ``name`` in the first of :func:`calibration_folders` that has it, or
    ``(None, labels of the folders searched)``. A fallback past hsi's camera folder is logged, as
    hsi warns of an older file of unknown camera."""
    folders = calibration_folders(config, plan, kind)
    for i, (folder, label) in enumerate(folders):
        if (folder / name).is_file():
            if i and folders[0][1].startswith('hsi '):
                logger.warning("%s: %s has no %s; using %s", kind, folders[0][1], name, label)
            return folder / name, label
    return None, ', '.join(label for _, label in folders)


def backend_profile(config, backend) -> Path | None:
    """The one profile whose ``profile.json`` names ``backend`` (None: none, or several)."""
    root = profiles_root(config)
    found = []
    for record in sorted(root.glob('*/profile.json')) if root.is_dir() else []:
        try:
            if str(json.loads(record.read_text(encoding='utf-8')).get('camera_backend', '')).lower() == \
                    str(backend).lower():
                found.append(record.parent)
        except (OSError, ValueError):
            continue
    return found[0] if len(found) == 1 else None


def _where(path) -> str:
    """``hsi <backend>/<camera>``, ``profile <id>`` or ``calibration`` for a provenance label."""
    parts = Path(path).parts
    for i, part in enumerate(parts[:-2]):
        if part == 'profiles' and i + 1 < len(parts):
            return f"profile {parts[i + 1]}"
        if part == 'camera_profiles' and i + 2 < len(parts) - 1:
            return f"hsi {parts[i + 1]}/{parts[i + 2]}"
    return 'calibration'


def gain_key(gain) -> str:
    """Gain in a file name: 1 -> '1.0', 1.2 -> '1.2', 1.25 -> '1.25' (the Sensor dialog's 2 decimals)."""
    text = f"{round(float(gain), 2):.2f}".rstrip('0')
    return text + '0' if text.endswith('.') else text


def profile_reflectance_path(plan) -> Path | None:
    """``profiles/<camera>/reflectance/refl_b<bin>_g<gain>.npy`` for the plan's camera values."""
    if plan.profile_dir is None or plan.gain is None:
        return None
    return plan.profile_dir / 'reflectance' / f"refl_b{int(plan.camera_bin)}_g{gain_key(plan.gain)}.npy"


def reflectance_save_path(config, plan, source) -> Path:
    """Where a new calibration of ``source`` goes: the camera profile, else ``reflectance/<source>.npy``."""
    return profile_reflectance_path(plan) or reflectance_folder(config) / f"{source_stem(source)}.npy"


def source_stem(source) -> str:
    """Data source name the factors belong to: the file name without its extension."""
    name = Path(source.path).name
    for suffix in ('.hdr', '.h5', '.hdf5', '.bip', '.bil', '.bsq', '.img', '.raw', '.dat'):
        if name.lower().endswith(suffix):
            name = name[:-len(suffix)]
    return name


def reflectance_folder(config) -> Path:
    """``corrections.reflectance_dir`` as a path."""
    return config.resolve(config.corrections.reflectance_dir) or Path(config.base_dir) / 'reflectance'


def reflectance_path(config, source) -> Path | None:
    """The factor profile of ``source``: ``<reflectance_dir>/<source name>.npy``, or the configured file."""
    choice = config.corrections.reflectance
    if choice not in (None, '', 'none', 'auto'):
        return config.resolve(choice)
    return reflectance_folder(config) / f"{source_stem(source)}.npy"


def _reflectance_step(config, plan, source) -> None:
    """Load the source's reflectance factors and check them against the bands the model gets.

    ``auto``/a path turns the step on; ``none`` offers a matching file as off. With
    ``white_reference.path`` (and no explicit factor file) the factors are first made from the
    newest matching white recording (``reflectance_builder.ensure_white_profile``). Factors are
    applied by band index (hsi), so ``check_factor_bands`` refuses another band axis; row-wise
    factors must also have one row per slit pixel of the frame.
    """
    choice = config.corrections.reflectance
    path = reflectance_path(config, source)
    wanted = choice not in (None, '', 'none')
    n_bands = plan.n_columns
    white_note = ''
    plan.reflectance_requires = {}
    profile_path = profile_reflectance_path(plan) if choice in (None, '', 'none', 'auto') else None
    if profile_path is not None and (profile_path.is_file() or not path.is_file()):
        path = profile_path                   # the camera's own factors for this bin/gain
    if choice in (None, '', 'none', 'auto') and _hsi_reflectance(config, plan, source, profile_path, wanted):
        return
    # why the copied hsi reference was not used, ahead of what happened instead
    hsi_note = getattr(plan, '_hsi_reflectance_note', '')
    hsi_prefix = f"{hsi_note}; " if hsi_note else ''
    own = profile_path is not None and profile_path.is_file()
    if not own and path is not None and path.is_file():
        # A calibration the operator drew in the GUI (saved by data source name: camera without a
        # serial) is newer knowledge than any recorded white: it must survive the next start.
        from reflectance_builder import profile_info
        own = str(profile_info(path).get('method', '')).startswith('drawn in Reflectance dialog')
    if own and config.white_reference.path not in (None, ''):
        white_note = ('camera profile used' if path == profile_path else 'calibration drawn in the GUI used') \
            + '; white_reference folder skipped'
    if config.white_reference.path not in (None, '') and not own and config.live_camera:
        # No default for a live camera: factors hold only for the light, distance and optics the
        # white was recorded with. Without its own calibration the step stays off and says so.
        white_note = 'not calibrated (the recorded white in white_reference is not used for a live camera)'
    elif config.white_reference.path not in (None, '') and choice in (None, '', 'none', 'auto') and not own:
        from reflectance_builder import ensure_white_profile
        try:
            built, white_note = ensure_white_profile(config, plan, source_stem(source))
        except (ValueError, OSError) as exc:
            logger.warning("White reference for %s not usable: %s", source.name, exc)
            plan.steps['reflectance'] = {'available': False, 'on': False,
                                         'why': f"{hsi_prefix}white reference: {exc}"}
            plan.provenance['reflectance'] = (f"white reference not usable: {exc}", choice)
            return
        if built is not None:
            path = built
        else:
            logger.info("%s; reflectance from %s if present", white_note, path.name)
    if path is None or not path.is_file():
        if wanted and choice != 'auto':
            raise FileNotFoundError(f"Reflectance profile not found: {path}")
        why = (f"no {path.name} in profile {plan.camera_id} (bin {plan.camera_bin}, gain {gain_key(plan.gain)}): "
               "calibrate with Reflectance... in the GUI" if path == profile_path else
               f"no {path.name} in {path.parent.name}/ (name = data source)")
        plan.steps['reflectance'] = {'available': False, 'on': False,
                                     'why': hsi_prefix + (f"{white_note}; {why}" if white_note else why)}
        plan.provenance['reflectance'] = ('none', choice)
        return
    try:
        factors, wavelengths = load_factor_profile(str(path))
        factors = np.asarray(factors, dtype=np.float32)
        check_factor_bands(factors.shape[-1], wavelengths, {'bands': n_bands, 'wavelength': plan.wavelengths})
        if factors.ndim == 2 and factors.shape[0] != plan.n_slit:
            crop = (f" ({plan.frame_shape[0]} minus the keystone margin {list(plan.keystone_crop)})"
                    if any(plan.keystone_crop) else '')
            raise ValueError(f"row-wise factors have {factors.shape[0]} rows, the model input has "
                             f"{plan.n_slit} slit rows{crop}")
        if factors.ndim not in (1, 2) or not np.isfinite(factors).all():
            raise ValueError("factors must be finite (bands,) or (slit rows, bands)")
        _check_binding(plan, path)
    except (ValueError, OSError) as exc:
        if wanted and choice != 'auto':
            raise ValueError(f"{path.name}: {exc}") from exc
        logger.warning("Reflectance %s not usable for %s: %s", path.name, source.name, exc)
        plan.steps['reflectance'] = {'available': False, 'on': False, 'why': f"{hsi_prefix}{path.name}: {exc}"}
        plan.provenance['reflectance'] = (f"not usable: {exc}", choice)
        return
    kind = 'row-wise' if factors.ndim == 2 else 'per band'
    checked = 'bands checked by wavelength' if wavelengths is not None else 'bands checked by count only (no sidecar)'
    label = f"{path.name} ({kind} {factors.shape}, {checked})"
    if white_note:
        label += f" - {white_note}"
    if hsi_note:
        label += f" ({hsi_note})"
    from reflectance_builder import profile_info
    info = profile_info(path)
    # Factors bound to the camera values they were made at (drawn on RAW / white folder): counts
    # are linear in exposure, so another exposure is scaled; another gain is only flagged.
    made_exp, made_gain = info.get('exposure_ms'), info.get('gain')
    if made_exp and plan.exposure_ms and abs(float(made_exp) - float(plan.exposure_ms)) > 1e-9:
        factors = factors * np.float32(float(made_exp) / float(plan.exposure_ms))
        label += f", scaled from {float(made_exp):g} ms to {float(plan.exposure_ms):g} ms"
    for note in _reflectance_warnings(plan, info):
        label += f", WARNING {note}"
        logger.warning("Reflectance %s: %s", path.name, note)
    plan.reflectance_flatfield = reflectance_flatfield(info)
    if plan.reflectance_flatfield:
        label += f", includes flat-field {Path(plan.reflectance_flatfield).parent.name or plan.reflectance_flatfield}"
    made_with = info.get('made_with') if isinstance(info.get('made_with'), dict) else {}
    if made_with:
        # drawn in the Reflectance dialog with these Dark / Flip / Smile / Keystone switches (as an
        # hsi reference, the factors hold only for them): start so, and gate Reflectance on them
        plan.reflectance_requires = {k: bool(made_with[k]) for k in ('dark', 'flip', 'smile', 'keystone')
                                     if k in made_with}
        if wanted:
            label += _start_steps_as(plan, plan.reflectance_requires, f"calibration {path.name}")
    plan.reflectance_placeholder = bool(info.get('placeholder'))
    if plan.reflectance_placeholder:
        label = (f"PLACEHOLDER {label} - stand-in from {info.get('white', '?')}, not a real white reference; "
                 "speed is real, reflectance values are not")
    plan.reflectance_factors = factors
    plan.steps['reflectance'] = {'available': True, 'on': wanted,
                                 'why': label if wanted else f"none (available: {label})"}
    plan.provenance['reflectance'] = (label if wanted else f"none (available: {label})", choice)


def _hsi_reflectance(config, plan, source, profile_path, wanted) -> bool:
    """Install the active hsi radiometric reference that fits the stream (``hsi_profiles``) and
    return True; False when there is none or this project's own factors for this bin / gain
    (``profile_path``) are newer. The reason an hsi reference was not used is kept in
    ``plan._hsi_reflectance_note`` for the step's text."""
    plan._hsi_reflectance_note = ''
    if plan.stage != 'raw' or getattr(plan, 'hsi_profile_dir', None) is None:
        return False
    if not (plan.hsi_profile_dir / 'radiometric').is_dir():
        return False
    from hsi_profiles import find_reference
    reference, why = find_reference(plan, source.metadata)
    if reference is None:
        plan._hsi_reflectance_note = f"hsi radiometric: {why}"
        logger.warning("hsi radiometric reference not used: %s", why)
        return False
    if profile_path is not None and profile_path.is_file() and _own_factors_fit(plan, profile_path):
        from reflectance_builder import profile_info
        import datetime
        own = profile_info(profile_path).get('created')
        try:
            own_time = datetime.datetime.fromisoformat(str(own).replace('Z', '+00:00')) if own else None
            if own_time is not None and own_time.tzinfo is None:
                own_time = own_time.astimezone()
        except ValueError:
            own_time = None
        if own_time is not None and own_time > reference.reference.acquired:
            plan._hsi_reflectance_note = (f"hsi {reference.id} not used: {profile_path.name} of this project "
                                          f"is newer ({own})")
            return False
    label = reference.label()
    # hsi refuses the reference for another setting of these steps: so does the pipeline
    plan.reflectance_requires = dict(reference.steps)
    if wanted:
        label += _start_steps_as(plan, reference.steps, f"hsi reference {reference.id}")
    for note in _reflectance_warnings(plan, reference.info()):
        label += f", WARNING {note}"
        logger.warning("Reflectance %s: %s", reference.id, note)
    plan.reflectance_factors = reference.factors
    plan.reflectance_placeholder = False
    plan.reflectance_flatfield = reference.flatfield
    if plan.reflectance_flatfield:
        label += f", includes flat-field {Path(plan.reflectance_flatfield).parent.name}"
    label += f"; made with {_steps_text(reference.steps)}"
    plan.steps['reflectance'] = {'available': True, 'on': wanted,
                                 'why': label if wanted else f"none (available: {label})"}
    plan.provenance['reflectance'] = (label if wanted else f"none (available: {label})",
                                      config.corrections.reflectance)
    logger.info("Reflectance: %s", label)
    return True


def _steps_text(steps: dict) -> str:
    """``Dark on, Flip on, Smile on, Keystone off`` for the steps factors were made with."""
    return ', '.join(f"{name.capitalize()} {'on' if value else 'off'}" for name, value in steps.items())


def _start_steps_as(plan, required: dict, what: str) -> str:
    """Start the available Dark / Smile / Keystone steps as ``required`` (the setting reflectance
    factors were made with) and return a note for the Reflectance text ('' when nothing changed)."""
    switched = []
    for name in ('dark', 'smile', 'keystone'):
        step = plan.steps.get(name)
        if name in required and step and step.get('available') and bool(step.get('on')) != bool(required[name]):
            step['on'] = bool(required[name])
            state = 'on' if step['on'] else 'off'
            switched.append(f"{name.capitalize()} {state}")
            step['why'] = f"{step['why']} - switched {state} for {what}"
            if name in plan.provenance:
                text, origin = plan.provenance[name]
                plan.provenance[name] = (f"{text} - switched {state} for {what}", origin)
    if not switched:
        return ''
    plan.dark = plan.dark_processor(plan.steps['dark']['on'])
    plan.smile = plan.smile_model if plan.steps['smile']['on'] else None
    logger.info("%s: %s", what, ', '.join(switched))
    return f" ({', '.join(switched)}: as the factors were made)"


def _own_factors_fit(plan, path) -> bool:
    """True when this project's factor file ``path`` has the stream's bands and slit rows."""
    try:
        factors, wavelengths = load_factor_profile(str(path))
        check_factor_bands(np.shape(factors)[-1], wavelengths, {'bands': plan.n_columns,
                                                               'wavelength': plan.wavelengths})
        return np.ndim(factors) == 1 or np.shape(factors)[0] == plan.n_slit
    except (ValueError, OSError):
        return False


def reflectance_flatfield(info: dict) -> str | None:
    """Label of the flat-field profile inside factors with this ``<name>.json`` record, else ``None``.

    Only a "Flat-field + reference" calibration (``kind: flatfield``) contains one; per-band and
    row-wise factors, and records from before this was written, contain none.
    """
    if info.get('kind') != 'flatfield':
        return None
    return str(info.get('flatfield_profile') or 'unknown profile')


def _check_binding(plan, path) -> None:
    """Refuse factors made at another bin or gain (counts per pixel differ; gain is not scaled).

    Factors in a camera's profile are keyed by bin/gain already. A name-based file is checked
    against the bin/gain its JSON records; for an identified camera it must record them (a
    per-band profile of another bin has the same bands and would pass every other check).
    """
    from reflectance_builder import profile_info
    if plan.stage != 'raw':
        return
    info = profile_info(path)
    made_bin, made_gain = info.get('camera_bin'), info.get('gain')
    if made_bin is not None and int(made_bin) != int(plan.camera_bin):
        raise ValueError(f"made at bin {int(made_bin)}, frames are bin {plan.camera_bin}")
    if made_gain is not None and plan.gain is not None and gain_key(made_gain) != gain_key(plan.gain):
        raise ValueError(f"made at gain {float(made_gain):g}, camera is at gain {float(plan.gain):g} "
                         "(gain is not scaled: calibrate again at this gain)")
    if plan.camera_id and 'signature' not in info and (made_bin is None or made_gain is None):
        raise ValueError(f"does not record the bin/gain it was made at; camera {plan.camera_id} needs factors "
                         "bound to its bin/gain (calibrate with Reflectance... or move them into its profile "
                         "with tools/migrate_profiles.py)")


def _reflectance_warnings(plan, info: dict) -> list[str]:
    """Conditions that do not stop the factors but make them doubtful: age, temperature, dark."""
    import datetime
    notes = []
    created = info.get('created')
    if created:
        try:
            text = str(created).strip()
            if text[-1:] in ('Z', 'z'):
                text = text[:-1] + '+00:00'
            made = datetime.datetime.fromisoformat(text)
            now = datetime.datetime.now(made.tzinfo) if made.tzinfo else datetime.datetime.now()
            hours = (now - made).total_seconds() / 3600.0
            if plan.reflectance_max_age_h and hours > plan.reflectance_max_age_h:
                notes.append(f"made {hours:.0f} h ago (lamp may have drifted)")
        except ValueError:
            notes.append(f"unreadable 'created' {created!r}: age not checked")
    made_t = info.get('temperature_c')
    if made_t is not None and plan.temperature_c is not None and plan.reflectance_max_dtemp_c and \
            abs(float(made_t) - float(plan.temperature_c)) > plan.reflectance_max_dtemp_c:
        notes.append(f"made at {float(made_t):.1f} °C, sensor at {float(plan.temperature_c):.1f} °C")
    steps = info.get('steps') or {}
    if steps.get('dark') is True and plan.steps.get('dark', {}).get('on') is False:
        notes.append("made with the dark step on, dark is off now")
    return notes


def apply_reflectance(binned: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """hsi ``modules.radiometric``: reflectance = counts x factor, float32.

    Row-wise factors ``(slit, bands)`` multiply each slit row and band, per-band factors
    ``(bands,)`` every row (as ``create_datacube_logic`` does band by band).
    """
    out = binned.astype(np.float32)
    out *= factors
    return out


#: Fewest bands a Smile border trim may leave (hsi ``MIN_RANGE_BANDS``); otherwise all are kept.
MIN_SMILE_TRIM_BANDS = 3


def _trim_smile_border(plan, model, full, k0: int, k1: int, range_text: str, origin) -> None:
    """Leave out the bins of ``full[k0:k1]`` that read the columns ``model`` zero-fills at the slit
    ends (hsi ``AcquisitionEngine._smile_trim``): such a bin is darker in the slit rows where the
    smile reaches past the sensor, which no flat-field or row-wise factor corrects."""
    if model is None or plan.stage != 'raw':
        return
    try:
        if int(model.image_shape[1]) != int(full.v_):
            return
        left, right = (max(0, int(v)) for v in model.margin)
    except (AttributeError, TypeError, ValueError, IndexError):
        return
    plan.smile_border = (left, right)
    if not (left or right):
        return
    keep = np.where((full.iL >= left) & (full.iR <= full.v_ - right))[0]
    if keep.size < MIN_SMILE_TRIM_BANDS:
        logger.warning("Smile border %s leaves %d of %d bands; all bands kept", (left, right), keep.size,
                       full.n_bins)
        return
    s0, s1 = max(k0, int(keep[0])), min(k1, int(keep[-1]) + 1)
    if (s0, s1) == (k0, k1):
        return
    if s1 - s0 < 1:
        raise ValueError(f"the Smile border (columns {left} left / {right} right) leaves no bin of "
                         f"{range_text}")
    plan.binner = full.subset(s0, s1)
    plan.wavelengths = plan.binner.lambda_out.astype(np.float32)
    plan.provenance['binning'] = (
        f"{full.v_} px -> {plan.binner.n_bins} bins {plan.wavelengths[0]:.2f}-{plan.wavelengths[-1]:.2f} nm, "
        f"bins {s0}:{s1} of {full.n_bins} ({range_text}; Smile border columns {left} left / {right} right "
        f"left out)", origin)
    logger.info("Smile border: columns %d left / %d right are zero-filled at the slit ends; bands %d:%d of %d "
                "kept (%.1f-%.1f nm)", left, right, s0, s1, full.n_bins, plan.wavelengths[0], plan.wavelengths[-1])


def geometry_model(plan, on: dict):
    """The Smile model the workers apply for the steps ``on`` (hsi
    ``AcquisitionEngine._active_calibration``): ``None`` with the smile step off, the model's
    smile-only view with the keystone step off (built once per model), else the model itself."""
    model = plan.smile_model if on.get('smile') else None
    if model is None or on.get('keystone', True) or not getattr(model, 'has_keystone', False):
        return model
    cached, view = plan.smile_only_view
    if cached is not model:
        view = model.smile_only()
        plan.smile_only_view = (model, view)
    return view


def keystone_margin(model) -> tuple:
    """``(top, bottom)`` slit rows the keystone remap of ``model`` filled from outside the sensor
    (0 in some bands), by hsi's ``crop_keystone_rows`` rule: ``(0, 0)`` without a model, without
    keystone, or when the margin would leave no rows. The plan crops them (``keystone_crop``)."""
    if model is None or not getattr(model, 'has_keystone', False):
        return (0, 0)
    try:
        top, bottom = (max(0, int(v)) for v in getattr(model, 'row_margin', (0, 0)))
    except (TypeError, ValueError):
        return (0, 0)
    h = int(model.image_shape[0])
    if top + bottom <= 0 or top + bottom >= h:
        return (0, 0)
    return (top, bottom)


def _wavelength_profile_path(config, plan):
    choice = config.acquisition.wavelength_profile
    if choice in (None, '', 'none'):
        return None
    if choice == 'auto':
        name = f'wavelength_bin{plan.camera_bin}.json'
        path, where = _first_file(config, plan, 'wavelength', name)
        if path is None:
            logger.warning("No wavelength profile %s in %s; using configured m/c", name, where)
        return path
    path = config.resolve(choice)
    if not path.is_file():
        raise FileNotFoundError(f"Wavelength profile not found: {path}")
    return path


def setting_mismatch(saved, plan) -> list:
    """hsi ``CameraSetting.mismatch`` of a calibration's ``camera_setting`` against the plan:
    backend, bin, orientation and frame size (each when both sides know it). The camera index is
    not compared: it numbers cameras on one computer, the profile folder names the unit."""
    if not saved:
        return []
    problems = []
    backend = saved.get('camera_backend')
    if backend is not None and plan.camera_backend and str(backend).lower() != str(plan.camera_backend).lower():
        problems.append(f"camera backend {backend}, frames {plan.camera_backend}")
    if saved.get('camera_bin') is not None and int(saved['camera_bin']) != int(plan.camera_bin):
        problems.append(f"bin {int(saved['camera_bin'])}, frames bin {plan.camera_bin}")
    if saved.get('raw_inverted') is not None and bool(saved['raw_inverted']) != bool(plan.raw_inverted):
        problems.append(f"raw_inverted {bool(saved['raw_inverted'])}, frames {bool(plan.raw_inverted)}")
    shape = saved.get('frame_shape')
    if shape and tuple(int(v) for v in shape) != tuple(plan.frame_shape):
        problems.append(f"frames {tuple(int(v) for v in shape)}, input {tuple(plan.frame_shape)}")
    return problems


def _load_wavelength_profile(path, camera_bin, shape, plan=None):
    """HSI profile: ``m`` is nm/px at the profile's own ``camera_bin`` (base m = m / bin)."""
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    profile_bin = int(data.get('camera_bin', 1))
    if profile_bin != camera_bin:
        raise ValueError(f"{Path(path).name} is for bin {profile_bin}, frames are bin {camera_bin}")
    frame_shape = data.get('frame_shape')
    if frame_shape and tuple(frame_shape) != tuple(shape):
        raise ValueError(f"{Path(path).name} was fitted on frames {tuple(frame_shape)}, input is {shape}")
    problems = setting_mismatch(data.get('camera_setting'), plan) if plan is not None else []
    if problems:
        raise ValueError(f"{Path(path).name} was made at another camera setting: {'; '.join(problems)}")
    return float(data['m']) / profile_bin, float(data['c']), f"{_where(path)}:{Path(path).name}"


def _dark_reference(config, plan, exposure, gain, temperature, o_e, o_g, choice=None, quiet=False):
    """``(mean, hot_pixel_map, label, record)``; ``mean`` is None when no dark is applied.

    ``auto`` searches :func:`calibration_folders` in order and takes the first folder that has a
    dark of this (bin, exposure, gain): hsi's ``DarkFrameLibrary`` rule with its fallback folder
    (within a folder, the file nearest the sensor temperature). ``record`` has the dark's
    ``path`` and ``captured_at`` (``None`` when no dark was found).
    """
    choice = config.corrections.dark if choice is None else choice
    warn = logger.debug if quiet else logger.warning
    shape = plan.frame_shape
    result = None
    if choice in (None, '', 'none'):
        return None, None, 'none', None
    if choice == 'auto':
        if exposure is None or gain is None:
            warn("Dark auto: exposure/gain unknown (no file metadata, not configured); dark bypassed")
            return None, None, 'missing: exposure/gain unknown', None
        folders = calibration_folders(config, plan, 'dark')
        where = folders[0][1]
        for i, (folder, label) in enumerate(folders):
            if not folder.is_dir():
                continue
            library = DarkFrameLibrary(str(folder))
            # hsi: the next folder only when this one has no file for the setting; a file here
            # that cannot be read means no dark (DarkFrameLibrary with its fallback_folder)
            if library.find(plan.camera_bin, exposure, gain, temperature) is None:
                continue
            result = library.load(bin=plan.camera_bin, exposure=exposure, gain=gain, temperature=temperature)
            if result is None:
                warn("Dark auto: the dark for bin %d, %g ms, gain %g in %s cannot be read; dark bypassed",
                     plan.camera_bin, exposure, gain, label)
                return None, None, f"unreadable: bin {plan.camera_bin} exp {exposure} gain {gain} in {label}", None
            if result is not None:
                if i and folders[0][1].startswith('hsi '):
                    warn("Dark auto: %s has no dark for bin %d, %g ms, gain %g; using %s", folders[0][1],
                         plan.camera_bin, exposure, gain, label)
                where = label
                break
        if result is None:
            warn("Dark auto: no file for bin %d, %.1f ms, gain %.1f in %s; dark bypassed",
                 plan.camera_bin, exposure, gain, ', '.join(label for _, label in folders))
            return None, None, f"missing: bin {plan.camera_bin} exp {exposure} gain {gain}", None
        mean, hot = result['mean'], result.get('hot_pixel_map')
        label = (f"{result['filename']} (auto from {where}: bin {plan.camera_bin}, {exposure:g} ms [{o_e}], "
                 f"gain {gain:g} [{o_g}])")
        if mean.shape != shape:
            warn("Dark auto: reference %s does not match frames %s; dark bypassed", mean.shape, shape)
            return None, None, f"mismatch: {mean.shape} vs frame {shape}", None
    else:
        path = config.resolve(choice)
        if not path.is_file():
            raise FileNotFoundError(f"Dark reference not found: {path}")
        captured = None
        if path.suffix.lower() == '.npy':
            mean, hot = np.load(path, allow_pickle=False), None
        else:
            with np.load(path, allow_pickle=False) as data:
                mean = np.array(data['mean'])
                hot = np.array(data['hot_pixel_map']) if 'hot_pixel_map' in data.files else None
                captured = str(data['captured_at']) if 'captured_at' in data.files else None
        result = {'path': str(path), 'captured_at': captured}
        if mean.shape != shape:
            raise ValueError(f"Dark reference {path.name} {mean.shape} does not match frames {shape}")
        label = path.name
    if hot is not None and hot.shape != mean.shape:
        raise ValueError("Dark hot-pixel map shape differs from its mean")
    return np.ascontiguousarray(mean, dtype=np.uint16), hot, label, result


def _smile_model(config, plan, choice=None, quiet=False):
    """``(model or None, label)``; also sets ``plan.smile_path`` to the file of a returned model."""
    choice = config.corrections.smile if choice is None else choice
    warn = logger.debug if quiet else logger.warning
    if choice in (None, '', 'none'):
        return None, 'none'
    if choice == 'auto':
        name = f'smile_model_bin{plan.camera_bin}.npz'
        path, where = _first_file(config, plan, 'smile', name)
        if path is None:
            warn("Smile auto: no %s in %s; Smile bypassed", name, where)
            return None, f"missing: {name}"
    else:
        path = config.resolve(choice)
        if not path.is_file():
            raise FileNotFoundError(f"Smile model not found: {path}")
    # Loaded exactly as hsi's SmileRuntime does (load_calibration + CPU fixed-point remap).
    runtime = SmileRuntime(path.parent)
    runtime.load(path, plan.camera_bin)
    model = runtime.calibration
    problems = []
    if tuple(model.image_shape) != tuple(plan.frame_shape):
        problems.append(f"{path.name} is {tuple(model.image_shape)}, frames are {tuple(plan.frame_shape)}")
    # hsi AcquisitionEngine._check_smile_setting: a model measured at another setting is not applied
    problems += setting_mismatch(getattr(model, 'camera_setting', None), plan)
    if problems:
        if choice == 'auto':
            warn("Smile auto: %s does not fit: %s; Smile bypassed", path.name, '; '.join(problems))
            return None, f"mismatch: {path.name}: {'; '.join(problems)}"
        raise ValueError(f"Smile model {path.name} does not match the frames: {'; '.join(problems)}")
    plan.smile_path = path
    return model, f"{path.name} ({model.interp}) [{_where(path)}]"


def config_as_dict(config: SimulationConfig) -> dict:
    data = asdict(config)
    data.pop('base_dir', None)
    return data
