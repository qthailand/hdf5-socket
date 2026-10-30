# -*- coding: utf-8 -*-
"""Simulation settings (``simulation.json``) and the per-stream acquisition plan.

The simulator re-runs the HSI acquisition pipeline on recorded frames::

    raw sensor frame -> dark subtraction -> hot-pixel repair -> slit flip (raw_inverted)
                     -> Smile remap -> lambda binning (m * camera_bin, c, resolution) -> model

exactly as ``hsi/modules/acquisition/engine.py`` does, with the calibration files copied
into ``calibration/`` (same layout as the HSI project, no link to it at run time)::

    calibration/dark_frames/<backend>/dark_b<bin>_e<exp>_g<gain>[_t<temp>].npz
    calibration/smile_maps/<backend>/smile_model_bin<bin>.npz
    calibration/wavelength/<backend>/wavelength_bin<bin>.json

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
from modules.smile import SmileRuntime
from modules.spectral_processing import LambdaBinner

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / 'simulation.json'


@dataclass(frozen=True)
class SourceConfig:
    path: str = 'data'
    pattern: str | None = None           # glob in a folder; None = all *.h5/*.hdf5/*.hdr
    stage: str = 'auto'                  # auto | raw | binned
    envi_scan_axis: str = 'samples'      # ENVI axis that is the scan direction
    dataset: str = 'spectrogram'         # HDF5 dataset name
    cache: str = 'disk'                  # disk (.npy memory map) | ram
    cache_dir: str | None = None


@dataclass(frozen=True)
class CameraConfig:
    fps: float = 500.0                   # simulated line rate; 0 = as fast as the pipeline takes
    overrun: str = 'drop'                # drop: a frame with no free buffer is lost (counted)
                                         # block: the camera waits (lossless throughput test)
    buffer_frames: int = 256             # frame buffers between camera and processing
    loop: bool = True                    # restart the playlist at its end
    belt_speed_m_s: float = 24.0         # only for the mm/line readout


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
    exposure_ms: float | None = None     # for dark "auto"
    gain: float | None = None
    temperature_c: float | None = None


@dataclass(frozen=True)
class CorrectionConfig:
    enabled: bool = True                 # False: frames go straight to binning
    backend: str = 'auto'                # auto | cpu | cupy_cuda | opencv_cuda
    dark: str = 'none'                   # none | auto | path to .npz/.npy
    hot_pixel: bool = True               # repair the dark file's hot pixels
    smile: str = 'none'                  # none | auto | path to smile_model_bin<N>.npz


@dataclass(frozen=True)
class ProcessingConfig:
    workers: int = 4                     # parallel frame workers (order is restored after)
    profile_every_s: float = 5.0         # log stage timings this often (0 = never)


@dataclass(frozen=True)
class DisplayConfig:
    gui_hz: float = 30.0
    canvas_width: int = 1024
    canvas_line_stride: int = 2          # draw every N-th line (aspect only; all lines are counted)
    theme: str = 'dark'                  # themes/<name>.json or a path to a theme JSON


@dataclass(frozen=True)
class ModelConfig:
    path: str = 'rf_coco_model.joblib'
    band_tolerance_nm: float = 0.5       # max distance between a model band and an input band


@dataclass(frozen=True)
class SimulationConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    corrections: CorrectionConfig = field(default_factory=CorrectionConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    base_dir: str = str(ROOT)            # folder relative paths resolve against

    def resolve(self, value: str | None) -> Path | None:
        if value in (None, '', 'none', 'auto'):
            return None
        path = Path(value)
        return path if path.is_absolute() else Path(self.base_dir) / path

    @property
    def calibration_dir(self) -> Path:
        return Path(self.base_dir) / 'calibration'

    def validate(self) -> 'SimulationConfig':
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
        if self.acquisition.resolution_nm <= 0 or self.acquisition.m_nm_per_px_bin1 <= 0:
            raise ValueError("acquisition.m_nm_per_px_bin1 and resolution_nm must be > 0")
        if self.corrections.backend not in ('auto', 'cpu', 'cupy_cuda', 'opencv_cuda'):
            raise ValueError(f"corrections.backend: {self.corrections.backend!r}")
        if not 1 <= self.processing.workers <= 32:
            raise ValueError("processing.workers must be 1..32")
        if self.display.canvas_line_stride < 1 or self.display.canvas_width < 16:
            raise ValueError("display.canvas_line_stride >= 1 and canvas_width >= 16")
        return self


_SECTION_CLASSES = {'source': SourceConfig, 'camera': CameraConfig, 'acquisition': AcquisitionConfig,
                    'corrections': CorrectionConfig, 'processing': ProcessingConfig,
                    'display': DisplayConfig, 'model': ModelConfig}


def load_config(path=None, **overrides) -> SimulationConfig:
    """Read ``simulation.json``; unknown sections/keys are errors (typos must not pass silently).

    Args:
        path: JSON file (default ``simulation.json`` beside this module).
        overrides: ``section={key: value}`` dicts applied on top of the file (command line).
    """
    path = Path(path or DEFAULT_CONFIG_PATH).resolve()
    data = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    if not path.is_file():
        logger.warning("No %s; using built-in defaults", path)
    for name, values in overrides.items():
        data.setdefault(name, {}).update({k: v for k, v in values.items() if v is not None})
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
    return SimulationConfig(**sections, base_dir=str(path.parent)).validate()


# --------------------------------------------------------------------- acquisition plan ---

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
    dark: DarkProcessor | None = None
    smile: object = None
    binner: LambdaBinner | None = None
    wavelengths: np.ndarray | None = None        # nm of each column the model receives
    columns: np.ndarray | None = None            # model band -> input column (set by the pipeline)
    provenance: dict = field(default_factory=dict)

    @property
    def n_columns(self) -> int:
        return self.binner.n_bins if self.binner is not None else int(self.frame_shape[1])

    def summary(self) -> str:
        lines = [f"stage={self.stage} frame={self.frame_shape}"]
        for key, (value, origin) in self.provenance.items():
            lines.append(f"  {key:<18} {value}  [{origin}]")
        return "\n".join(lines)


def _pick(meta: dict, keys, fallback, use_meta: bool, cast=lambda v: v):
    """``(value, origin)`` from the first metadata key present, else ``(fallback, 'config')``."""
    if use_meta:
        for key in keys:
            if key in meta and meta[key] is not None:
                return cast(meta[key]), f"file:{key}"
    return fallback, 'config'


def build_plan(config: SimulationConfig, source) -> AcquisitionPlan:
    """Resolve calibration and corrections for a stream whose first file is ``source``.

    Raises:
        ValueError / FileNotFoundError: an explicitly configured file is missing or does not
            match the frames (an ``auto`` lookup that finds nothing only logs and bypasses).
    """
    shape = source.frame_shape
    plan = AcquisitionPlan(stage=source.stage, frame_shape=shape)
    prov = plan.provenance
    if source.stage == 'binned':
        plan.wavelengths = source.wavelengths
        prov['input'] = ('already corrected and binned: no dark/hot pixel/flip/Smile/binning', source.name)
        if source.wavelengths is not None:
            prov['wavelength'] = (f"{source.wavelengths[0]:.2f}-{source.wavelengths[-1]:.2f} nm, "
                                  f"{source.wavelengths.size} bands", 'file')
        return plan

    acq, cor = config.acquisition, config.corrections
    meta, use = source.metadata, acq.use_file_metadata
    plan.camera_backend, o = _pick(meta, ('camera_backend',), acq.camera_backend, use, str)
    prov['camera_backend'] = (plan.camera_backend, o)
    plan.camera_bin, o = _pick(meta, ('camera_bin',), acq.camera_bin, use, int)
    prov['camera_bin'] = (plan.camera_bin, o)
    plan.raw_inverted, o = _pick(meta, ('raw_inverted',), acq.raw_inverted, use, bool)
    prov['raw_inverted'] = (plan.raw_inverted, o)
    exposure, o_e = _pick(meta, ('exposure_ms',), acq.exposure_ms, use, float)
    gain, o_g = _pick(meta, ('gain',), acq.gain, use, float)
    temperature, _ = _pick(meta, ('temperature_c', 'temperature'), acq.temperature_c, use, float)

    # Wavelength axis: recorded values > wavelength profile > config numbers.
    m1, c, o_m = acq.m_nm_per_px_bin1, acq.c_nm, 'config'
    profile = _wavelength_profile_path(config, plan.camera_backend, plan.camera_bin)
    if profile is not None:
        m1, c, o_m = _load_wavelength_profile(profile, plan.camera_bin, shape)
    if use and meta.get('m_nm_per_px_bin1') is not None and meta.get('c_nm') is not None:
        m1, c, o_m = float(meta['m_nm_per_px_bin1']), float(meta['c_nm']), 'file:m_nm_per_px_bin1,c_nm'
    plan.m_nm_per_px_bin1, plan.c_nm = float(m1), float(c)
    prov['wavelength m,c'] = (f"m={m1:.6g} nm/px (bin 1) x bin {plan.camera_bin}, c={c:.6g} nm", o_m)
    plan.resolution_nm, o = _pick(meta, ('resolution_nm',), acq.resolution_nm, use, float)
    prov['resolution_nm'] = (plan.resolution_nm, o)

    plan.binner = LambdaBinner(m=plan.m_nm_per_px_bin1 * plan.camera_bin, c=plan.c_nm,
                               step=plan.resolution_nm).fit(shape[1])
    plan.wavelengths = plan.binner.lambda_out.astype(np.float32)
    prov['binning'] = (f"{shape[1]} px -> {plan.binner.n_bins} bins "
                       f"{plan.wavelengths[0]:.2f}-{plan.wavelengths[-1]:.2f} nm", 'computed')

    plan.corrections_enabled = cor.enabled
    plan.correction_backend = cor.backend
    if not cor.enabled:
        prov['corrections'] = ('disabled (frames go straight to binning)', 'config')
        return plan

    plan.dark = DarkProcessor()
    mean, hot, label = _dark_reference(config, plan, exposure, gain, temperature, o_e, o_g)
    if mean is not None:
        plan.dark.set_dark_reference(mean, hot if cor.hot_pixel else None)
        n_hot = 0 if hot is None or not cor.hot_pixel else int(np.count_nonzero(hot))
        prov['dark'] = (f"{label} ({n_hot} hot pixels{'' if cor.hot_pixel else ', repair off'})", cor.dark)
    else:
        plan.dark.set_dark_enabled(False)
        prov['dark'] = (label, cor.dark)

    plan.smile, label = _smile_model(config, plan)
    prov['smile'] = (label, cor.smile)
    prov['correction backend'] = (cor.backend, 'config')
    return plan


def _wavelength_profile_path(config, backend, camera_bin):
    choice = config.acquisition.wavelength_profile
    if choice in (None, '', 'none'):
        return None
    if choice == 'auto':
        path = config.calibration_dir / 'wavelength' / backend / f'wavelength_bin{camera_bin}.json'
        if not path.is_file():
            logger.warning("No wavelength profile %s; using configured m/c", path)
            return None
        return path
    path = config.resolve(choice)
    if not path.is_file():
        raise FileNotFoundError(f"Wavelength profile not found: {path}")
    return path


def _load_wavelength_profile(path, camera_bin, shape):
    """HSI profile: ``m`` is nm/px at the profile's own ``camera_bin`` (base m = m / bin)."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    profile_bin = int(data.get('camera_bin', 1))
    if profile_bin != camera_bin:
        raise ValueError(f"{Path(path).name} is for bin {profile_bin}, frames are bin {camera_bin}")
    frame_shape = data.get('frame_shape')
    if frame_shape and tuple(frame_shape) != tuple(shape):
        raise ValueError(f"{Path(path).name} was fitted on frames {tuple(frame_shape)}, input is {shape}")
    return float(data['m']) / profile_bin, float(data['c']), f"calibration:{Path(path).name}"


def _dark_reference(config, plan, exposure, gain, temperature, o_e, o_g):
    """``(mean, hot_pixel_map, label)``; ``mean`` is None when no dark is applied."""
    choice = config.corrections.dark
    shape = plan.frame_shape
    if choice in (None, '', 'none'):
        return None, None, 'none'
    if choice == 'auto':
        if exposure is None or gain is None:
            logger.warning("Dark auto: exposure/gain unknown (no file metadata, not configured); dark bypassed")
            return None, None, 'missing: exposure/gain unknown'
        folder = config.calibration_dir / 'dark_frames' / plan.camera_backend
        result = DarkFrameLibrary(str(folder)).load(bin=plan.camera_bin, exposure=exposure, gain=gain,
                                                    temperature=temperature)
        if result is None:
            logger.warning("Dark auto: no file for bin %d, %.1f ms, gain %.1f in %s; dark bypassed",
                           plan.camera_bin, exposure, gain, folder)
            return None, None, f"missing: bin {plan.camera_bin} exp {exposure} gain {gain}"
        mean, hot = result['mean'], result.get('hot_pixel_map')
        label = (f"{result['filename']} (auto: bin {plan.camera_bin}, {exposure:g} ms [{o_e}], "
                 f"gain {gain:g} [{o_g}])")
        if mean.shape != shape:
            logger.warning("Dark auto: reference %s does not match frames %s; dark bypassed", mean.shape, shape)
            return None, None, f"mismatch: {mean.shape} vs frame {shape}"
    else:
        path = config.resolve(choice)
        if not path.is_file():
            raise FileNotFoundError(f"Dark reference not found: {path}")
        if path.suffix.lower() == '.npy':
            mean, hot = np.load(path, allow_pickle=False), None
        else:
            with np.load(path, allow_pickle=False) as data:
                mean = np.array(data['mean'])
                hot = np.array(data['hot_pixel_map']) if 'hot_pixel_map' in data.files else None
        if mean.shape != shape:
            raise ValueError(f"Dark reference {path.name} {mean.shape} does not match frames {shape}")
        label = path.name
    if hot is not None and hot.shape != mean.shape:
        raise ValueError("Dark hot-pixel map shape differs from its mean")
    return np.ascontiguousarray(mean, dtype=np.uint16), hot, label


def _smile_model(config, plan):
    choice = config.corrections.smile
    if choice in (None, '', 'none'):
        return None, 'none'
    if choice == 'auto':
        path = config.calibration_dir / 'smile_maps' / plan.camera_backend / f'smile_model_bin{plan.camera_bin}.npz'
        if not path.is_file():
            logger.warning("Smile auto: no %s; Smile bypassed", path)
            return None, f"missing: {path.name}"
    else:
        path = config.resolve(choice)
        if not path.is_file():
            raise FileNotFoundError(f"Smile model not found: {path}")
    # Loaded exactly as hsi's SmileRuntime does (load_calibration + CPU fixed-point remap).
    runtime = SmileRuntime(path.parent)
    runtime.load(path, plan.camera_bin)
    model = runtime.calibration
    if tuple(model.image_shape) != tuple(plan.frame_shape):
        if choice == 'auto':
            logger.warning("Smile auto: %s is %s, frames are %s; Smile bypassed", path.name,
                           model.image_shape, plan.frame_shape)
            return None, f"mismatch: {path.name} {model.image_shape}"
        raise ValueError(f"Smile model {path.name} {model.image_shape} does not match frames {plan.frame_shape}")
    return model, f"{path.name} ({model.interp})"


def config_as_dict(config: SimulationConfig) -> dict:
    data = asdict(config)
    data.pop('base_dir', None)
    return data
