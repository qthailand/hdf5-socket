"""HSI runtime adapter for sensor-coordinate models, before wavelength binning.

No crop or float conversion is applied: uint16 input remains uint16 and the
spectral pixel origin stays unchanged. Invalid border columns are zero-filled.
Models must be calibrated after the same orientation and hardware binning.
"""
import logging
from pathlib import Path
from .calibration import load_calibration

logger = logging.getLogger(__name__)

class SmileRuntime:
    """Per-bin smile model holder for the HSI frame loop.

    Looks for ``smile_model_bin<N>.npz`` (see :func:`modules.smile.save_calibration`) in a
    folder, keeps the loaded :class:`SmileCalibration` and applies it to sensor frames
    without changing their dtype or size.

    Data conventions: frames are 2-D ``(slit rows, spectral columns)`` in sensor
    coordinates after orientation, at the hardware bin the model was calibrated for;
    uint8, uint16 or float32.

    Threading: not thread-safe; call ``set_bin``/``load`` and ``apply`` from the same
    thread (or serialise them externally).
    """
    def __init__(self, folder):
        """Args:
            folder: Directory containing ``smile_model_bin<N>.npz`` files (str or ``os.PathLike``).
        """
        self.folder = Path(folder)
        self.calibration = None
        self.bin_value = None
        self.path = None

    def load(self, path, bin_value):
        """Load a model file and select the CPU fixed-point remap backend.

        Args:
            path: Path of a smile model ``.npz``.
            bin_value: Hardware bin the model belongs to (stored as ``int``).

        Raises:
            ValueError: The file's ``format_version`` is newer than supported.
            OSError, KeyError: The file cannot be read or lacks required entries.
        """
        cal = load_calibration(str(path))
        # CPU fixed-point remap preserves dtype and avoids transfers to GPU.
        cal.set_backend('cpu')
        _ = cal.backend_info
        self.calibration = cal
        self.bin_value = int(bin_value)
        self.path = str(path)

    def set_bin(self, bin_value):
        """Switch to the model of a hardware bin, disabling correction if there is none.

        Args:
            bin_value: Hardware bin; the file ``smile_model_bin<bin_value>.npz`` is looked up in the folder.

        Returns:
            bool: True if a model was found and loaded, False if correction is now disabled.
        """
        self.calibration = None
        self.bin_value = int(bin_value)
        self.path = None
        path = self.folder / f'smile_model_bin{bin_value}.npz'
        if path.exists():
            self.load(path, bin_value)
            logger.info('Smile model loaded: %s', path)
            return True
        logger.info('No smile model for bin%d; correction disabled', bin_value)
        return False

    def apply(self, frame):
        """Correct one frame, or return it unchanged when no model is loaded.

        Args:
            frame: 2-D sensor frame ``(slit rows, spectral columns)``, uint8/uint16/float32.

        Returns:
            numpy.ndarray: The remapped frame with the same shape and dtype (invalid border
            columns are zero-filled), or ``frame`` itself when no model is loaded.

        Raises:
            ValueError: The frame shape differs from the model's ``image_shape``; the model is
                discarded so correction stays disabled until ``set_bin``/``load`` is called again.
            TypeError: The dtype is not uint8, uint16 or float32.
        """
        cal = self.calibration
        if cal is None:
            return frame
        if frame.shape != tuple(cal.image_shape):
            self.calibration = None
            raise ValueError(f'Smile model {cal.image_shape} does not match oriented sensor frame {frame.shape}; correction disabled')
        return cal.apply_native(frame)
