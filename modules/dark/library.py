"""
DarkFrameLibrary — captures, computes, stores, and loads dark frame references.

Data conventions:
    Frames are 2-D ``(slit, wavelength)`` sensor images; exposure is in ms; temperatures
    are degrees Celsius, and a value below ``-900`` (``get_temperature()`` returns
    ``-999.0``) means unavailable.
    All frames of one capture must share a shape (a changed shape restarts
    accumulation). ``hot_pixel_map`` is ``uint8`` (255 = hot).

Threading:
    Not thread-safe; call ``start_capture``/``add_frame``/``cancel`` from one
    thread (normally the camera worker). File I/O happens synchronously inside
    the ``add_frame`` call that completes the capture.

Example:
    >>> import tempfile, numpy as np
    >>> from modules.dark import DarkFrameLibrary
    >>> lib = DarkFrameLibrary(tempfile.mkdtemp(), n_frames=3)
    >>> lib.start_capture(bin=1, exposure=100.0, gain=5.0)
    >>> [lib.add_frame(np.full((4, 4), 7, np.uint16), temperature=24.8) for _ in range(3)]
    [False, False, True]
    >>> result = lib.load(1, 100.0, 5.0, temperature=30.0)
    >>> int(result['mean'][0, 0]), result['temperature'], result['filename']
    (7, 24.8, 'dark_b1_e100.0_g5.0_t+25.npz')

Storage layout (one .npz per (bin, exposure, gain, temperature) combination; the
temperature in the name is the capture's mean rounded to whole degrees):
    dark_frames/
        dark_b1_e100.0_g5.0_t+25.npz
        dark_b1_e100.0_g5.0_t+31.npz
        dark_b2_e50.0_g1.0.npz          (camera reported no temperature, or an older file)
        ...

Each .npz contains:
    mean          — pre-computed mean dark frame (original dtype, ready to subtract)
    hot_pixel_map — uint8 mask, 255 = hot pixel
    bin           — scalar int
    exposure      — scalar float (ms)
    gain          — scalar float
    n_frames      — scalar int (number of frames averaged)
    temperature, temperature_min, temperature_max
                  — scalar float (°C) over the captured frames; absent when unknown

Selection:
    ``load``/``find`` pick, among the files of a (bin, exposure, gain), the one whose
    temperature is nearest the given temperature (:data:`UNKNOWN_TEMPERATURE_C` when the
    camera reports none). A file without a temperature is used only when no file of that
    combination has one.
"""

import logging
import os
import re
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

#: Temperature (°C) assumed when the camera reports none: pick the dark taken nearest to it.
UNKNOWN_TEMPERATURE_C = 60.0

_TEMPERATURE_SUFFIX = re.compile(r"_t([+-]\d+)\.npz$")


def valid_temperature(value) -> Optional[float]:
    """``float(value)``, or ``None`` for ``None``, NaN or the ``-999`` "unavailable" marker."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or value < -900.0:
        return None
    return value


class DarkFrameLibrary:
    """
    Manages dark frame capture, storage, and retrieval.

    Usage — capture:
        lib = DarkFrameLibrary(folder='dark_frames', n_frames=50)
        lib.start_capture(bin=1, exposure=100.0, gain=5.0)
        for frame in camera_frames:
            done = lib.add_frame(frame, temperature=camera.get_temperature())
            if done:
                break
        result = lib.load(bin=1, exposure=100.0, gain=5.0, temperature=25.0)

    Usage — load existing:
        result = lib.load(bin=1, exposure=100.0, gain=5.0, temperature=25.0)
        if result:
            dark_mean = result['mean']
            hot_map   = result['hot_pixel_map']
    """

    def __init__(self, folder: str, n_frames: int = 50):
        """Open (and create if needed) the library folder.

        Args:
            folder: Directory holding the ``.npz`` files; created if missing.
            n_frames: Number of frames averaged per capture.
        """
        self.folder = folder
        self.n_frames = n_frames
        os.makedirs(folder, exist_ok=True)

        # Capture state (reset by start_capture)
        self._collecting: bool = False
        self._stack: List[np.ndarray] = []
        self._temperatures: List[float] = []
        self._bin: int = 1
        self._exposure: float = 0.0
        self._gain: float = 0.0

    # ------------------------------------------------------------------
    #  Capture
    # ------------------------------------------------------------------

    def start_capture(self, bin: int, exposure: float, gain: float) -> None:
        """Begin a new accumulation session for the given camera parameters.

        Args:
            bin: Hardware bin factor.
            exposure: Exposure time in ms.
            gain: Camera gain.
        """
        self._collecting = True
        self._stack = []
        self._temperatures = []
        self._bin = int(bin)
        self._exposure = float(exposure)
        self._gain = float(gain)
        logger.info(
            "DarkFrameLibrary: starting capture — bin=%d exposure=%.1f gain=%.1f target=%d",
            self._bin, self._exposure, self._gain, self.n_frames,
        )

    def add_frame(self, frame: np.ndarray, temperature: Optional[float] = None) -> bool:
        """
        Add one frame to the accumulation buffer.

        Args:
            frame: 2-D ``(slit, wavelength)`` frame; copied, the caller keeps
                ownership of the original.
            temperature: Sensor temperature (°C) when the frame was taken; ``None`` or
                ``-999`` when unknown. The saved file records the mean/min/max of the
                known values and is named after their mean.

        Returns:
            bool: True when ``n_frames`` have been collected, the mean has been
            computed, and the result has been saved to disk (capture then stops).
            False otherwise, including when no capture is active.

        Raw frames are discarded immediately after the mean is computed.
        """
        if not self._collecting:
            return False

        if self._stack and frame.shape != self._stack[0].shape:
            # ★ stale frame จาก queue — reset ด้วย frame ใหม่
            logger.warning(
                "DarkFrameLibrary: shape changed %s → %s, resetting",
                self._stack[0].shape, frame.shape,
            )
            self._stack.clear()
            self._temperatures.clear()

        self._stack.append(frame.copy())
        temperature = valid_temperature(temperature)
        if temperature is not None:
            self._temperatures.append(temperature)

        if len(self._stack) >= self.n_frames:
            self._compute_and_save()
            self._collecting = False
            return True

        return False

    def cancel(self) -> None:
        """Abort current capture without saving anything."""
        self._collecting = False
        self._stack.clear()
        self._temperatures.clear()
        logger.info("DarkFrameLibrary: capture cancelled")

    @property
    def progress(self) -> tuple:
        """tuple: ``(collected_frames, target_frames)``."""
        return len(self._stack), self.n_frames

    @property
    def is_collecting(self) -> bool:
        """bool: True while a capture started by ``start_capture`` is in progress."""
        return self._collecting

    # ------------------------------------------------------------------
    #  Load / Query
    # ------------------------------------------------------------------

    def find(self, bin: int, exposure: float, gain: float,
             temperature: Optional[float] = None) -> Optional[str]:
        """Path of the stored dark that :meth:`load` would return, without reading it.

        Only lists the folder, so it is cheap enough to call whenever the temperature changes.

        Args:
            bin: Hardware bin factor.
            exposure: Exposure time in ms (matched at 0.1 ms resolution).
            gain: Camera gain (matched at 0.1 resolution).
            temperature: Current sensor temperature (°C); ``None``/``-999`` means unknown
                and :data:`UNKNOWN_TEMPERATURE_C` is used.

        Returns:
            The file whose temperature is nearest (the newest one on a tie); a file
            without a temperature only when the combination has no other; ``None`` if
            nothing is stored.
        """
        legacy, by_temperature = self._candidates(bin, exposure, gain)
        if by_temperature:
            target = valid_temperature(temperature)
            if target is None:
                target = UNKNOWN_TEMPERATURE_C
            best = min(by_temperature, key=lambda item: (abs(item[0] - target), -self._mtime(item[1])))
            return best[1]
        return legacy

    def load(self, bin: int, exposure: float, gain: float,
             temperature: Optional[float] = None) -> Optional[Dict]:
        """
        Load the stored dark frame for the given parameters (see :meth:`find`).

        Args:
            bin: Hardware bin factor.
            exposure: Exposure time in ms (matched at 0.1 ms resolution).
            gain: Camera gain (matched at 0.1 resolution).
            temperature: Current sensor temperature (°C), or ``None``/``-999`` if unknown.

        Returns a dict:
            {
                'mean':            np.ndarray,   # dark mean frame, original dtype
                'hot_pixel_map':   np.ndarray,   # uint8 mask (255 = hot)
                'bin':             int,
                'exposure':        float,
                'gain':            float,
                'n_frames':        int,
                'temperature':     float | None, # °C, mean over the capture
                'temperature_min': float | None,
                'temperature_max': float | None,
                'filename':        str,
            }
        Returns None if no matching dark frame exists or it cannot be read.
        """
        path = self.find(bin, exposure, gain, temperature)
        if path is None:
            return None
        try:
            data = np.load(path)
            return {
                'mean':          data['mean'],
                'hot_pixel_map': data['hot_pixel_map'],
                'bin':           int(data['bin']),
                'exposure':      float(data['exposure']),
                'gain':          float(data['gain']),
                'n_frames':      int(data['n_frames']),
                **self._temperature_fields(data),
                'filename':      os.path.basename(path),
            }
        except Exception:
            logger.exception("DarkFrameLibrary: failed to load %s", path)
            return None

    def exists(self, bin: int, exposure: float, gain: float) -> bool:
        """Return True if any dark frame is stored for this combination.

        Args:
            bin: Hardware bin factor.
            exposure: Exposure time in ms.
            gain: Camera gain.
        """
        legacy, by_temperature = self._candidates(bin, exposure, gain)
        return legacy is not None or bool(by_temperature)

    def delete(self, bin: int, exposure: float, gain: float,
               temperature: Optional[float] = None) -> None:
        """Delete stored dark frames (no error if none exist).

        Args:
            bin: Hardware bin factor.
            exposure: Exposure time in ms.
            gain: Camera gain.
            temperature: Delete only the file of this temperature (rounded to whole
                degrees); ``None`` deletes every file of the combination.
        """
        legacy, by_temperature = self._candidates(bin, exposure, gain)
        if temperature is None:
            paths = [path for _, path in by_temperature] + ([legacy] if legacy else [])
        else:
            paths = [self._filepath(bin, exposure, gain, temperature)]
        for path in paths:
            if os.path.isfile(path):
                os.remove(path)
                logger.info("DarkFrameLibrary: deleted %s", path)

    def list_keys(self) -> List[Dict]:
        """
        Return metadata for all stored dark frames.

        Each entry: {'bin', 'exposure', 'gain', 'n_frames', 'temperature', 'filename'}
        (``temperature`` is ``None`` when the file has none).
        """
        keys = []
        for name in sorted(os.listdir(self.folder)):
            if not name.endswith('.npz'):
                continue
            path = os.path.join(self.folder, name)
            try:
                data = np.load(path)
                keys.append({
                    'bin':         int(data['bin']),
                    'exposure':    float(data['exposure']),
                    'gain':        float(data['gain']),
                    'n_frames':    int(data['n_frames']),
                    'temperature': self._temperature_fields(data)['temperature'],
                    'filename':    name,
                })
            except Exception:
                logger.warning("DarkFrameLibrary: cannot read metadata from %s", name)
        return keys

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _key(self, bin: int, exposure: float, gain: float, temperature: Optional[float] = None) -> str:
        key = f"dark_b{int(bin)}_e{float(exposure):.1f}_g{float(gain):.1f}"
        temperature = valid_temperature(temperature)
        if temperature is not None:
            key += f"_t{int(round(temperature)):+d}"
        return key

    def _filepath(self, bin: int, exposure: float, gain: float, temperature: Optional[float] = None) -> str:
        return os.path.join(self.folder, self._key(bin, exposure, gain, temperature) + '.npz')

    def _candidates(self, bin: int, exposure: float, gain: float):
        """``(legacy path or None, [(temperature from the name, path), ...])`` of one combination."""
        prefix = self._key(bin, exposure, gain)
        legacy = None
        by_temperature = []
        try:
            names = os.listdir(self.folder)
        except OSError:
            return None, []
        for name in names:
            if name == prefix + '.npz':
                legacy = os.path.join(self.folder, name)
            elif name.startswith(prefix + '_t'):
                match = _TEMPERATURE_SUFFIX.search(name)
                if match is not None and match.start() == len(prefix):
                    by_temperature.append((int(match.group(1)), os.path.join(self.folder, name)))
        return legacy, by_temperature

    @staticmethod
    def _mtime(path: str) -> float:
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0

    @staticmethod
    def _temperature_fields(data) -> Dict:
        fields = {}
        for name in ('temperature', 'temperature_min', 'temperature_max'):
            fields[name] = float(data[name]) if name in data.files else None
        return fields

    def _compute_and_save(self) -> None:
        if not self._stack:
            return

        stack = np.stack(self._stack, axis=0).astype(np.float32)
        original_dtype = self._stack[0].dtype
        mean = np.round(np.mean(stack, axis=0)).astype(original_dtype)

        hot_pixel_map = self._detect_hot_pixels(mean)

        temperature = {}
        if self._temperatures:
            temps = np.asarray(self._temperatures, dtype=np.float64)
            temperature = {
                'temperature':     np.array(round(float(temps.mean()), 2), dtype=np.float64),
                'temperature_min': np.array(float(temps.min()), dtype=np.float64),
                'temperature_max': np.array(float(temps.max()), dtype=np.float64),
            }
        mean_temperature = float(temperature['temperature']) if temperature else None

        path = self._filepath(self._bin, self._exposure, self._gain, mean_temperature)
        np.savez_compressed(
            path,
            mean=mean,
            hot_pixel_map=hot_pixel_map,
            bin=np.array(self._bin,     dtype=np.int32),
            exposure=np.array(self._exposure, dtype=np.float32),
            gain=np.array(self._gain,    dtype=np.float32),
            n_frames=np.array(len(self._stack), dtype=np.int32),
            **temperature,
        )
        logger.info(
            "DarkFrameLibrary: saved %s (%d frames, %d hot pixels, temperature %s)",
            path, len(self._stack), int(hot_pixel_map.sum() > 0 and np.count_nonzero(hot_pixel_map)),
            "unknown" if mean_temperature is None else f"{mean_temperature:.1f} C",
        )
        self._stack.clear()
        self._temperatures.clear()

    def _detect_hot_pixels(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect hot pixels using a normalised threshold of mean + 4σ
        (or 99.9th percentile, whichever is higher).
        Returns a uint8 mask where 255 = hot pixel.
        """
        if np.issubdtype(frame.dtype, np.integer):
            max_val = float(np.iinfo(frame.dtype).max)
        else:
            max_val = float(np.max(frame)) or 1.0

        img_norm = frame.astype(np.float64) / max_val
        mean_val = np.mean(img_norm)
        std_val  = np.std(img_norm)
        perc_val = np.percentile(img_norm, 99.9)
        threshold = max(mean_val + 4.0 * std_val, perc_val)

        mask = img_norm > threshold
        if not mask.any() and std_val > 0:
            fallback = max(mean_val + 3.0 * std_val, np.percentile(img_norm, 99.5))
            mask = img_norm > fallback

        hot_map = np.zeros_like(frame, dtype=np.uint8)
        hot_map[mask] = 255
        return hot_map
