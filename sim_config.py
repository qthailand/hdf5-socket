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
from modules.radiometric import check_factor_bands, load_factor_profile
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
    cache: str = 'auto'                  # auto | disk (.npy memory map) | ram | stream (decode ahead, no copy)
    cache_dir: str | None = None


@dataclass(frozen=True)
class CameraConfig:
    fps: float = 500.0                   # simulated line rate; 0 = as fast as the pipeline takes
    overrun: str = 'drop'                # drop: a frame with no free buffer is lost (counted)
                                         # block: the camera waits (lossless throughput test)
    buffer_frames: int = 256             # frame buffers between camera and processing
    loop: bool = True                    # restart the playlist at its end
    belt_speed_m_s: float = 0.4          # 24 m/min; only for the mm/line readout


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
    reflectance_dir: str = 'reflectance' # <dir>/<source name>.npy (+ .wavelength.npy), see reflectance/README.md


@dataclass(frozen=True)
class ProcessingConfig:
    workers: int = 4                     # parallel frame workers (order is restored after)
    numba_threads: int | None = 1        # Numba threads per worker (binning/fused kernels); None = all cores
    profile_every_s: float = 5.0         # log stage timings this often (0 = never)


@dataclass(frozen=True)
class DisplayConfig:
    gui_hz: float = 30.0
    canvas_width: int = 4096             # belt-map history (lines / stride) kept for display
    canvas_line_stride: int = 1          # draw every N-th line (display only; all lines are counted)
    line_aspect: float = 1.0             # width of one scan line in slit pixels (1 = square: a round
                                         # target stays round; measure: target width in lines / height in rows)
    raw_zone_fraction: float = 1 / 3    # right part of the belt map showing the newest lines before
                                         # classification (grey: model-input intensity); 0 = off
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
    path: str = 'rf_coco_model.joblib'  # "none": speed-only stand-in model (no trained weights)
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
    optics: OpticsConfig = field(default_factory=OpticsConfig)
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
        if self.corrections.backend not in ('auto', 'cpu_fused', 'cpu', 'cupy_cuda', 'opencv_cuda'):
            raise ValueError(f"corrections.backend: {self.corrections.backend!r}")
        if not 1 <= self.processing.workers <= 32:
            raise ValueError("processing.workers must be 1..32")
        if self.display.canvas_line_stride < 1 or self.display.canvas_width < 16:
            raise ValueError("display.canvas_line_stride >= 1 and canvas_width >= 16")
        if not self.display.line_aspect > 0:
            raise ValueError("display.line_aspect must be > 0")
        if not 0.0 <= self.display.raw_zone_fraction < 1.0:
            raise ValueError("display.raw_zone_fraction must be in [0, 1)")
        return self


_SECTION_CLASSES = {'source': SourceConfig, 'camera': CameraConfig, 'acquisition': AcquisitionConfig,
                    'corrections': CorrectionConfig, 'processing': ProcessingConfig,
                    'display': DisplayConfig, 'model': ModelConfig, 'optics': OpticsConfig}


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

#: Processing steps in hsi order; the GUI shows one switch per step. ``dark`` is dark
#: subtraction + hot-pixel repair together, like hsi's ``DarkProcessor.set_dark_enabled``
#: (whether hot pixels are repaired is ``corrections.hot_pixel``). ``reflectance`` is the
#: last step before inference (hsi ``modules.radiometric``: counts x factor).
STEP_NAMES = ('dark', 'flip', 'smile', 'binning', 'reflectance')


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
    reflectance_factors: np.ndarray | None = None   # (slit, bands) or (bands,) float32, also when offered off
    reflectance_placeholder: bool = False           # factors marked placeholder in reflectance/<name>.json
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
        plan.steps = {name: {'available': False, 'on': False, 'why': 'input is already corrected and binned'}
                      for name in STEP_NAMES}
        _reflectance_step(config, plan, source)
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
    # the dark the recording used was captured at dark_temperature_c (HSI picks the nearest one)
    temperature, _ = _pick(meta, ('temperature_c', 'dark_temperature_c', 'sensor_temperature_c_start',
                                  'temperature'), acq.temperature_c, use, float)

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
        for name in ('dark', 'smile', 'reflectance'):
            plan.steps[name] = {'available': False, 'on': False, 'why': 'corrections.enabled = false'}
        return plan

    # Dark / hot pixel: the configured reference, or (with "none") an auto match offered as off.
    mean, hot, label = _dark_reference(config, plan, exposure, gain, temperature, o_e, o_g)
    dark_on = mean is not None
    if mean is None and cor.dark in (None, '', 'none'):
        offered_mean, offered_hot, offered = _dark_reference(config, plan, exposure, gain, temperature,
                                                             o_e, o_g, choice='auto', quiet=True)
        if offered_mean is not None:
            mean, hot, label = offered_mean, offered_hot, f"none (available: {offered})"
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
    _reflectance_step(config, plan, source)
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


def reflectance_path(config, source) -> Path | None:
    """The factor profile of ``source``: ``<reflectance_dir>/<source name>.npy``, or the configured file."""
    choice = config.corrections.reflectance
    if choice not in (None, '', 'none', 'auto'):
        return config.resolve(choice)
    folder = config.resolve(config.corrections.reflectance_dir) or Path(config.base_dir) / 'reflectance'
    name = Path(source.path).name
    for suffix in ('.hdr', '.h5', '.hdf5', '.bip', '.bil', '.bsq', '.img', '.raw', '.dat'):
        if name.lower().endswith(suffix):
            name = name[:-len(suffix)]
    return folder / f"{name}.npy"


def _reflectance_step(config, plan, source) -> None:
    """Load the source's reflectance factors and check them against the bands the model gets.

    ``auto``/a path turns the step on; ``none`` offers a matching file as off. Factors are
    applied by band index (hsi), so ``check_factor_bands`` refuses another band axis; row-wise
    factors must also have one row per slit pixel of the frame.
    """
    choice = config.corrections.reflectance
    path = reflectance_path(config, source)
    wanted = choice not in (None, '', 'none')
    n_bands = plan.n_columns
    if path is None or not path.is_file():
        if wanted and choice != 'auto':
            raise FileNotFoundError(f"Reflectance profile not found: {path}")
        plan.steps['reflectance'] = {'available': False, 'on': False,
                                     'why': f"no {path.name} in {path.parent.name}/ (name = data source)"}
        plan.provenance['reflectance'] = ('none', choice)
        return
    try:
        factors, wavelengths = load_factor_profile(str(path))
        factors = np.asarray(factors, dtype=np.float32)
        check_factor_bands(factors.shape[-1], wavelengths, {'bands': n_bands, 'wavelength': plan.wavelengths})
        if factors.ndim == 2 and factors.shape[0] != plan.frame_shape[0]:
            raise ValueError(f"row-wise factors have {factors.shape[0]} rows, frames have "
                             f"{plan.frame_shape[0]} slit rows")
        if factors.ndim not in (1, 2) or not np.isfinite(factors).all():
            raise ValueError("factors must be finite (bands,) or (slit rows, bands)")
    except (ValueError, OSError) as exc:
        if wanted and choice != 'auto':
            raise ValueError(f"{path.name}: {exc}") from exc
        logger.warning("Reflectance %s not usable for %s: %s", path.name, source.name, exc)
        plan.steps['reflectance'] = {'available': False, 'on': False, 'why': f"{path.name}: {exc}"}
        plan.provenance['reflectance'] = (f"not usable: {exc}", choice)
        return
    kind = 'row-wise' if factors.ndim == 2 else 'per band'
    checked = 'bands checked by wavelength' if wavelengths is not None else 'bands checked by count only (no sidecar)'
    label = f"{path.name} ({kind} {factors.shape}, {checked})"
    from reflectance_builder import profile_info
    info = profile_info(path)
    plan.reflectance_placeholder = bool(info.get('placeholder'))
    if plan.reflectance_placeholder:
        label = (f"PLACEHOLDER {label} - stand-in from {info.get('white', '?')}, not a real white reference; "
                 "speed is real, reflectance values are not")
    plan.reflectance_factors = factors
    plan.steps['reflectance'] = {'available': True, 'on': wanted,
                                 'why': label if wanted else f"none (available: {label})"}
    plan.provenance['reflectance'] = (label if wanted else f"none (available: {label})", choice)


def apply_reflectance(binned: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """hsi ``modules.radiometric``: reflectance = counts x factor, float32.

    Row-wise factors ``(slit, bands)`` multiply each slit row and band, per-band factors
    ``(bands,)`` every row (as ``create_datacube_logic`` does band by band).
    """
    out = binned.astype(np.float32)
    out *= factors
    return out


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


def _dark_reference(config, plan, exposure, gain, temperature, o_e, o_g, choice=None, quiet=False):
    """``(mean, hot_pixel_map, label)``; ``mean`` is None when no dark is applied."""
    choice = config.corrections.dark if choice is None else choice
    warn = logger.debug if quiet else logger.warning
    shape = plan.frame_shape
    if choice in (None, '', 'none'):
        return None, None, 'none'
    if choice == 'auto':
        if exposure is None or gain is None:
            warn("Dark auto: exposure/gain unknown (no file metadata, not configured); dark bypassed")
            return None, None, 'missing: exposure/gain unknown'
        folder = config.calibration_dir / 'dark_frames' / plan.camera_backend
        result = DarkFrameLibrary(str(folder)).load(bin=plan.camera_bin, exposure=exposure, gain=gain,
                                                    temperature=temperature)
        if result is None:
            warn("Dark auto: no file for bin %d, %.1f ms, gain %.1f in %s; dark bypassed",
                           plan.camera_bin, exposure, gain, folder)
            return None, None, f"missing: bin {plan.camera_bin} exp {exposure} gain {gain}"
        mean, hot = result['mean'], result.get('hot_pixel_map')
        label = (f"{result['filename']} (auto: bin {plan.camera_bin}, {exposure:g} ms [{o_e}], "
                 f"gain {gain:g} [{o_g}])")
        if mean.shape != shape:
            warn("Dark auto: reference %s does not match frames %s; dark bypassed", mean.shape, shape)
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


def _smile_model(config, plan, choice=None, quiet=False):
    choice = config.corrections.smile if choice is None else choice
    warn = logger.debug if quiet else logger.warning
    if choice in (None, '', 'none'):
        return None, 'none'
    if choice == 'auto':
        path = config.calibration_dir / 'smile_maps' / plan.camera_backend / f'smile_model_bin{plan.camera_bin}.npz'
        if not path.is_file():
            warn("Smile auto: no %s; Smile bypassed", path)
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
            warn("Smile auto: %s is %s, frames are %s; Smile bypassed", path.name,
                           model.image_shape, plan.frame_shape)
            return None, f"mismatch: {path.name} {model.image_shape}"
        raise ValueError(f"Smile model {path.name} {model.image_shape} does not match frames {plan.frame_shape}")
    return model, f"{path.name} ({model.interp})"


def config_as_dict(config: SimulationConfig) -> dict:
    data = asdict(config)
    data.pop('base_dir', None)
    return data
