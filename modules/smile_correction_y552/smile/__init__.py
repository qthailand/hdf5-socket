# smile/__init__.py
"""Smile Correction Module for Imaging Spectrometers.

Public API:
    Core:        load_frame, save_frame, analyze_smile
    Sklearn:     SmileCorrector
    Calibration: SmileCalibration, save_calibration, load_calibration
    Widget:      SmileCorrectionWidget (requires PyQt5)

Usage::

    from smile import SmileCorrector

    corrector = SmileCorrector(deg_y=2, deg_x=1)
    corrector.fit(hgar_frame)
    corrector.save("model.npz")
    corrected = corrector.transform(raw_frame, crop=True)
"""

from .core import (
    Line,
    auto_sat_level,
    detect_lines,
    estimate_noise,
    load_frame,
    robust_sigma,
    save_frame,
    straightness_rms,
    trace_line,
)
from .surface import (
    Surface2D,
    cross_validate_lines,
    fit_surface_2d,
)
from .calibration import (
    SmileCalibration,
    build_maps,
    calibration_from_analysis,
    correct_smile,
    crop_margin,
    load_calibration,
    margin_from_map,
    save_calibration,
)
from .analysis import (
    SmileAnalysis,
    analyze_smile,
    verify_on_frame,
)
from .corrector import SmileCorrector

# Qt widget (optional).
try:
    from .widget import HAVE_QT, PlotCanvas, SmileCorrectionWidget
except ImportError:
    HAVE_QT = False

__version__ = "2.2.0"

__all__ = [
    # Core
    "load_frame", "save_frame", "robust_sigma", "auto_sat_level",
    "estimate_noise", "Line", "detect_lines", "trace_line",
    "straightness_rms",
    # Surface
    "Surface2D", "fit_surface_2d", "cross_validate_lines",
    # Remap
    "build_maps", "margin_from_map", "correct_smile", "crop_margin",
    # Analysis
    "SmileAnalysis", "analyze_smile", "verify_on_frame",
    # Calibration
    "SmileCalibration", "calibration_from_analysis",
    "save_calibration", "load_calibration",
    # Sklearn API
    "SmileCorrector",
    # Qt (optional)
    "SmileCorrectionWidget", "PlotCanvas", "HAVE_QT",
]