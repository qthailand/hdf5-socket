# -*- coding: utf-8 -*-
"""Desktop view of the pushbroom sorting simulation (recorded files) or of a live camera.

The window only draws: ``pipeline.Pipeline`` runs camera, workers and statistics in its own
threads and the GUI pulls a :class:`pipeline.Snapshot` ``display.gui_hz`` times per second.
A slow GUI therefore shows fewer refreshes but never loses or double-counts a line.

    python gui.py
    python gui.py --config simulation.json --source dataset/dataset.bip.hdr --fps 300
    python gui.py --theme light                 # themes/<name>.json, or a path to a theme JSON
    python gui.py --live                        # live camera (source.kind = camera), acquisition.* applied
    python gui.py --live --set acquisition.exposure_ms=2 --set acquisition.gain=1

Colours and fonts live in ``themes/`` (``app.qss`` + one token JSON per theme); this module
only sets object names and the panel/accent properties.
"""
import argparse
import logging
import threading
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QEvent, QPointF, QSettings, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QScrollArea, QSpinBox, QTabWidget, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
                             QSizePolicy, QSplitter, QVBoxLayout, QWidget)

from focus import edge_contrast
from modules.compute_runtime import get_compute_capabilities, initialize_compute
from modules.sharpness import sharpness_score
from model_catalog import scan_models, session_entry
from pipeline import LOST_CLASS, build_pipeline
from quick_model import LabelSet
from sim_config import load_config, parse_set_overrides
from themes import apply_theme, make_panel, set_accent, use_dark_title_bar
from utils import generate_tab10_lut
from widgets.class_legend_widget import ClassLegendWidget
from widgets.composition_widget import CompositionWidget
from widgets.focus_panel import FocusPanel
from widgets.model_selector import ModelSelector
from widgets.performance_panel import PerformancePanel, ThroughputStrip
from widgets.reflectance_dialog import ReflectanceDialog
from widgets.sensor_dialog import SensorDialog
from widgets.spectrum_panel import SpectrumPanel, TrainingBand

#: binned lines kept before Flat-field/Reflectance while the calibration dialog is open
CALIBRATION_LINES = 256
#: most memory of the frozen view of the quick model dialog (the newest model-input lines)
QUICK_MODEL_MB = 256.0

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] (%(threadName)-10s) %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class PushbroomScannerMainWindow(QMainWindow):
    """Raw frame with focus aid, belt classification map, throughput/latency status and class statistics."""

    #: (entries, scan number): the model scan runs in a thread and reports here
    _models_scanned = pyqtSignal(object, int)

    def __init__(self, config, config_path=None) -> None:
        super().__init__()
        initialize_compute()
        self.config = config
        self._config_path = config_path
        self._reflectance_dialog = None             # calibration window (the belt remains live)
        self._step_connected: set[str] = set()
        # Quick models (Train…): labels and models of this session, in memory only.
        self._quick_labels = LabelSet()
        self._session_models: list = []            # classifiers trained in this session
        self._quick_number = 0
        self._scanned_entries: list = []           # the model files of the last scan
        self._live = config.live_camera
        self.setWindowTitle("HSI Industrial Scanner — " +
                            ("live camera" if self._live else "pushbroom simulation"))
        self.resize(1440, 900)
        self._init_ui()

        try:
            self._pipeline = build_pipeline(config)
        except Exception as exc:
            logger.critical("Initialization failed: %s", exc, exc_info=True)
            QMessageBox.critical(self, "Initialization Error",
                                 f"Core Initialization Failed:\n{exc}")
            raise

        try:
            self._start(config)
        except Exception as exc:
            # Never leave silently: main() exits on this, and the camera process and the shared
            # frame memory of the pipeline that was already built must be given back first.
            logger.critical("Start failed: %s", exc, exc_info=True)
            self._pipeline.stop()
            QMessageBox.critical(self, "Initialization Error", f"Start Failed:\n{exc}")
            raise

    def _start(self, config) -> None:
        """Bind the controls to the pipeline that was just built and start it."""
        self._model_scan = 0
        self._models_scanned.connect(self._show_models)
        self.model_selector.chosen.connect(self._on_model_chosen)
        self.model_selector.rescan_requested.connect(self._scan_models)
        self.model_selector.remove_requested.connect(self._remove_session_model)
        self._show_model_classes(0)
        self._scan_models()
        self._sync_spectrum_history()
        self.class_legend_widget.color_changed.connect(self._on_class_color_changed)
        # through lambdas: the pipeline is replaced when the sensor settings change
        self.class_legend_widget.reset_requested.connect(
            lambda: self._pipeline.reset_counts(getattr(self, '_count_delay', 0)))
        self.class_legend_widget.reset_requested.connect(self._forget_previous_counts)
        self.performance_panel.reset_requested.connect(lambda: self._pipeline.reset_perf())
        self._setup_backend_selector()
        self._setup_step_switches()
        self._setup_fps_control()
        self._forget_previous_counts()
        self._info_label.setText(self._plan_text())

        self._pipeline.start()
        self._gui_timer = QTimer(self)
        self._gui_timer.setInterval(int(1000 / max(1.0, config.display.gui_hz)))
        self._gui_timer.timeout.connect(self._refresh_gui_slot)
        self._gui_timer.start()

    def _init_ui(self) -> None:
        self._raw_label = QLabel("Raw sensor frame")
        self._raw_label.setObjectName("imagePanel")
        self._raw_label.setAlignment(Qt.AlignCenter)
        self._raw_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._raw_label.setToolTip(f"Newest raw frame from the {'live' if self._live else 'simulated'} camera "
                                   "(slit × spectral), display only")
        # Focus aid beside the preview (as in hsi's scan window): sharpness of the same raw frames.
        self.focus_panel = FocusPanel()
        self._focus_frame = None                   # last frame scored (each frame is scored once)
        self._preview_row = QWidget()
        self._preview_row.setObjectName("previewRow")
        preview_row = QHBoxLayout(self._preview_row)
        preview_row.setContentsMargins(0, 0, 0, 0)
        preview_row.setSpacing(8)
        # Spectrum preview: spectra of the pixels clicked on the belt map (see _probe_spectrum).
        self.spectrum_panel = SpectrumPanel()
        self._belt = None                          # what the belt map shows now (click -> line, row)
        preview_row.addWidget(self._raw_label, stretch=1)
        preview_row.addWidget(self.focus_panel)

        self._canvas_label = QLabel("Belt classification map")
        self._canvas_label.setObjectName("imagePanel")
        self._canvas_label.setAlignment(Qt.AlignCenter)
        self._canvas_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._canvas_label.setToolTip("Belt map at true proportions (display.line_aspect, canvas_line_stride); "
                                      "the belt moves right to left: new lines enter the RAW zone on the right "
                                      "(grey: what the model receives, before classification) and continue "
                                      "into the CLASSIFIED zone (class colours); magenta = lines lost to overrun")
        self._raw_range: tuple[float, float] | None = None   # grey scale of the raw zone (smoothed)

        self._status_label = QLabel("Starting…")
        self._status_label.setObjectName("statusLabel")
        make_panel(self._status_label, accent='ok')
        self._status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._info_label = QLabel("")
        self._info_label.setObjectName("infoLabel")
        make_panel(self._info_label, accent='info')
        self._info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        # Technical controls are assembled together for the Setup & Calibration workspace.
        self._preview_button = QPushButton("Spectrogram")
        self._preview_button.setObjectName("toggleButton")
        self._preview_button.setCheckable(True)
        self._preview_button.setChecked(True)
        self._preview_button.setToolTip("Show / hide the real-time raw spectrogram preview "
                                        "(hidden = not rendered, saves GUI time)")
        self._preview_button.toggled.connect(self._set_preview_visible)

        self._focus_button = QPushButton("Focus")
        self._focus_button.setObjectName("toggleButton")
        self._focus_button.setCheckable(True)
        self._focus_button.setChecked(True)
        self._focus_button.setToolTip("Show / hide the focus aid beside the spectrogram preview "
                                      "(hidden = no sharpness computed)")
        self._focus_button.toggled.connect(self._set_focus_visible)

        self._spectrum_button = QPushButton("Spectrum")
        self._spectrum_button.setObjectName("toggleButton")
        self._spectrum_button.setCheckable(True)
        self._spectrum_button.setChecked(True)
        self._spectrum_button.setToolTip("Show / hide the spectrum preview: click a pixel on the belt map to draw the "
                                         "spectrum the model received\n(hidden = no lines kept: no memory, no copy per line)")
        self._spectrum_button.toggled.connect(self._set_spectrum_visible)
        self.spectrum_panel.cleared.connect(self._repaint_belt)

        self._side_button = QPushButton("Analytics")
        self._side_button.setObjectName("toggleButton")
        self._side_button.setCheckable(True)
        self._side_button.setChecked(True)
        self._side_button.setToolTip(
            "Show / hide Composition and Materials (or drag the vertical handle)")
        self._side_button.toggled.connect(self._set_side_visible)

        self._reflectance_button = QPushButton("Reflectance…")
        self._reflectance_button.setObjectName("toggleButton")
        self._reflectance_button.setToolTip("Calibrate reflectance: grab an image, zoom and draw the white standard "
                                            "in the settings window (bound to the current camera values)")
        self._reflectance_button.clicked.connect(self._open_reflectance_dialog)
        self._train_button = QPushButton("Train…")
        self._train_button.setObjectName("toggleButton")
        self._train_button.setToolTip("Quick model for system tests: mark areas of the newest lines per class, "
                                      "train a linear SGD model and switch to it.\nNothing is saved: the model and "
                                      "its labels are gone when the program closes.")
        self._train_button.clicked.connect(self._open_quick_model_dialog)
        self._sensor_button = QPushButton("Sensor…")
        self._sensor_button.setObjectName("toggleButton")
        self._sensor_button.setToolTip("Exposure, gain and bin of the camera (restarts the pipeline; the dark, "
                                       "Smile and wavelength files follow the new values)")
        self._sensor_button.clicked.connect(self._open_sensor_dialog)
        self._sensor_button.setVisible(self._live)

        backend_row = QHBoxLayout()
        backend_row.addWidget(self._preview_button)
        backend_row.addWidget(self._focus_button)
        backend_row.addWidget(self._sensor_button)
        backend_row.addWidget(self._reflectance_button)
        backend_row.addWidget(self._train_button)
        caption = QLabel("Correction backend")
        caption.setObjectName("metricCaption")
        backend_row.addWidget(caption)
        self._backend_combo = QComboBox()
        self._backend_combo.setToolTip("Dark / hot pixel / flip / Smile / binning backend (hsi modules.preprocessing).\n"
                                       "Switches at the next line in every worker; nothing is skipped.")
        backend_row.addWidget(self._backend_combo)
        self._backend_label = QLabel("")
        self._backend_label.setObjectName("backendLabel")
        make_panel(self._backend_label, accent='neutral')
        backend_row.addWidget(self._backend_label, stretch=1)

        self._info_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Ignored)
        self._info_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._bottom_pane = QWidget()
        self._bottom_pane.setObjectName("bottomPane")
        bottom = QVBoxLayout(self._bottom_pane)
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setSpacing(8)
        self.throughput_strip = ThroughputStrip()        # FPS / throughput, always visible

        # Simulated camera line rate, changed while running (0 = as fast as the chain takes).
        camera_row = QHBoxLayout()
        camera_row.setSpacing(6)
        fps_caption = QLabel("Camera line rate")
        fps_caption.setObjectName("metricCaption")
        camera_row.addWidget(fps_caption)
        self._fps_spin = QSpinBox()
        self._fps_spin.setObjectName("fpsSpin")
        self._fps_spin.setRange(0, 20000)
        self._fps_spin.setSingleStep(10)
        self._fps_spin.setSuffix(" lines/s")
        self._fps_spin.setSpecialValueText("max (lossless)")
        self._fps_spin.setKeyboardTracking(False)          # apply when typing is finished
        self._fps_spin.setToolTip("Line rate of the simulated camera; applies at the next line.\n"
                                  "0 = max: the next line as soon as a buffer is free (the chain's lossless maximum).")
        camera_row.addWidget(self._fps_spin)
        self._fps_presets = QHBoxLayout()
        self._fps_presets.setSpacing(4)
        camera_row.addLayout(self._fps_presets)
        self._belt_label = QLabel("")
        self._belt_label.setObjectName("metricCaption")
        camera_row.addWidget(self._belt_label)
        camera_row.addStretch()
        bottom.addLayout(camera_row)
        bottom.addLayout(backend_row)

        # Processing steps in hsi order; unchecked = skipped (e.g. data already corrected).
        steps_row = QHBoxLayout()
        steps_row.setSpacing(6)
        steps_caption = QLabel("Processing steps")
        steps_caption.setObjectName("metricCaption")
        steps_row.addWidget(steps_caption)
        self._step_boxes = {}
        for i, (name, text) in enumerate(self._STEP_LABELS):
            if i:
                arrow = QLabel("→")
                arrow.setObjectName("metricCaption")
                steps_row.addWidget(arrow)
            box = QCheckBox(text)
            box.setObjectName("stepCheck")
            self._step_boxes[name] = box
            steps_row.addWidget(box)
        # The model was trained on counts or on reflectance; feeding it the other gives wrong classes.
        self._units_warning = QLabel("")
        self._units_warning.setObjectName("stepWarning")
        self._units_warning.setVisible(False)
        steps_row.addWidget(self._units_warning)
        steps_row.addStretch()
        bottom.addLayout(steps_row)
        bottom.addWidget(self._info_label, stretch=1)

        # Operation keeps the existing belt map above the model-reference spectrum.
        # Technical previews and settings live on their own workspace.
        self._splitter = QSplitter(Qt.Vertical)
        self._splitter.setObjectName("mainSplitter")
        self._splitter.setChildrenCollapsible(False)
        for widget in (self._canvas_label, self.spectrum_panel):
            self._splitter.addWidget(widget)
        self._splitter.setCollapsible(1, True)
        for index, stretch in enumerate((2, 1)):
            self._splitter.setStretchFactor(index, stretch)
        self._splitter.setHandleWidth(8)
        self._splitter.splitterMoved.connect(lambda *_: self._sync_spectrum_button())
        self._raw_label.setMinimumHeight(40)
        self._canvas_label.setMinimumHeight(160)
        self.spectrum_panel.setMinimumHeight(180)
        self._canvas_label.installEventFilter(self)
        self._spectrum_height = None

        left = QVBoxLayout()
        left.addWidget(self._splitter)

        # Keep the existing statistics; performance charts have their own Diagnostics workspace.
        self.class_legend_widget = ClassLegendWidget()
        # Model tiles below the class key (Materials tab): one click switches the running model.
        self.model_selector = ModelSelector()
        self.class_legend_widget.add_section(self.model_selector)
        self.composition_widget = CompositionWidget()
        self.class_legend_widget.shares_changed.connect(self.composition_widget.set_shares)
        self.performance_panel = PerformancePanel()
        self._side_tabs = QTabWidget()
        self._side_tabs.setObjectName("sideTabs")
        self._side_tabs.tabBar().setElideMode(Qt.ElideNone)
        self._side_tabs.tabBar().setExpanding(False)
        self._side_tabs.addTab(self.composition_widget, "Composition")
        self._side_tabs.addTab(self.class_legend_widget, "Materials")
        self._side_tabs.setMinimumWidth(300)
        left.setContentsMargins(0, 0, 0, 0)
        left_pane = QWidget()
        left_pane.setLayout(left)
        left_pane.setMinimumWidth(320)

        # Second splitter, side by side: [left (images + bottom pane) | Material Analytics].
        # Dragging the handle right can hide the analytics panel completely.
        self._side_splitter = QSplitter(Qt.Horizontal)
        self._side_splitter.setObjectName("sideSplitter")
        self._side_splitter.setChildrenCollapsible(False)
        self._side_splitter.addWidget(left_pane)
        self._side_splitter.addWidget(self._side_tabs)
        self._side_splitter.setCollapsible(1, True)
        self._side_splitter.setStretchFactor(0, 1)           # extra width goes to the images
        self._side_splitter.setStretchFactor(1, 0)
        self._side_splitter.setHandleWidth(8)
        self._side_splitter.splitterMoved.connect(lambda *_: self._sync_side_button())
        self._side_width = None                    # analytics width before the last hide

        self._operation_page = QWidget()
        operation = QVBoxLayout(self._operation_page)
        operation.setContentsMargins(0, 0, 0, 0)
        operation.setSpacing(8)
        heading = QHBoxLayout()
        self._run_label = QLabel("Starting…")
        self._run_label.setObjectName("operationState")
        make_panel(self._run_label, accent='info')
        self._context_label = QLabel("")
        self._context_label.setObjectName("operationContext")
        self._context_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        heading.addWidget(self._run_label)
        heading.addWidget(self._context_label, stretch=1)
        heading.addWidget(self._spectrum_button)
        heading.addWidget(self._side_button)
        operation.addLayout(heading)
        self._units_warning.setWordWrap(True)
        operation.addWidget(self._units_warning)
        operation.addWidget(self._side_splitter, stretch=1)
        operation.addWidget(self.throughput_strip)

        self._setup_page = QWidget()
        setup = QVBoxLayout(self._setup_page)
        setup.setContentsMargins(0, 0, 0, 0)
        setup.addWidget(self._preview_row, stretch=2)
        setup_scroll = QScrollArea()
        setup_scroll.setWidgetResizable(True)
        setup_scroll.setWidget(self._bottom_pane)
        setup.addWidget(setup_scroll, stretch=1)

        diagnostics = QWidget()
        diagnostic_layout = QVBoxLayout(diagnostics)
        diagnostic_layout.setContentsMargins(0, 0, 0, 0)
        diagnostic_layout.addWidget(self.performance_panel, stretch=1)
        self._status_label.setWordWrap(True)
        diagnostic_layout.addWidget(self._status_label)
        diagnostic_scroll = QScrollArea()
        diagnostic_scroll.setWidgetResizable(True)
        diagnostic_scroll.setWidget(diagnostics)

        self._workspace = QTabWidget()
        self._workspace.setObjectName("workspaceTabs")
        self._workspace.addTab(self._operation_page, "Operation")
        self._workspace.addTab(self._setup_page, "Setup && Calibration")
        self._workspace.addTab(diagnostic_scroll, "Diagnostics")

        main = QVBoxLayout()
        main.setContentsMargins(10, 10, 10, 10)
        main.addWidget(self._workspace)
        container = QWidget()
        container.setLayout(main)
        self.setCentralWidget(container)
        self._restore_layout()

    # -------------------------------------------------------------------------- splitter ---

    _LAYOUT_KEY = "operation_splitter_v2"
    _SIDE_LAYOUT_KEY = "operation_side_v2"
    _FOCUS_KEY = "focus_panel_visible"
    _SPECTRUM_KEY = "operation_spectrum_visible_v2"

    def _preview_collapsed(self) -> bool:
        """Technical camera preview is rendered only on its visible workspace."""
        return not self._raw_label.isVisible()

    def _set_preview_visible(self, visible: bool) -> None:
        self._raw_label.setVisible(visible)
        self._preview_row.setVisible(visible or self._focus_button.isChecked())
        if not visible:
            self._raw_label.clear()

    def _set_focus_visible(self, visible: bool) -> None:
        self.focus_panel.setVisible(visible)
        self._preview_row.setVisible(visible or self._preview_button.isChecked())
        if visible:
            self.focus_panel.reset()
        self._focus_frame = None

    def _set_spectrum_visible(self, visible: bool) -> None:
        """The spectrum has its own pane; hiding it returns space to the belt only."""
        canvas, spectrum = self._splitter.sizes()
        if not visible and spectrum:
            self._spectrum_height = spectrum
        self.spectrum_panel.setVisible(visible)
        if visible:
            total = canvas + spectrum
            height = self._spectrum_height or max(self.spectrum_panel.minimumHeight(), total // 3)
            self._splitter.setSizes([max(self._canvas_label.minimumHeight(), total - height), height])
        else:
            self._splitter.setSizes([canvas + spectrum, 0])
            self.spectrum_panel.clear()
        if hasattr(self, '_pipeline'):
            self._sync_spectrum_history()
            self._repaint_belt()

    def _sync_spectrum_history(self) -> None:
        """Keep the newest model-input lines in the pipeline only while the preview is switched on."""
        on = self._spectrum_button.isChecked()
        if on and not self._pipeline.spectrum_lines:
            self._pipeline.start_spectrum_history()
        elif not on and self._pipeline.spectrum_lines:
            self._pipeline.stop_spectrum_history()
        self._update_canvas_cursor()

    def _update_canvas_cursor(self) -> None:
        if self._spectrum_button.isChecked():
            self._canvas_label.setCursor(Qt.CrossCursor)
        else:
            self._canvas_label.unsetCursor()

    def _focus_active(self) -> bool:
        return self._focus_button.isChecked() and self.focus_panel.isVisible()

    def _update_focus(self, frame) -> None:
        """Score each new preview frame once (hsi's 5 Hz), only while the focus aid is on screen."""
        if frame is None or frame is self._focus_frame or not self._focus_active():
            return
        self._focus_frame = frame
        axis = self.focus_panel.axis
        self.focus_panel.set_value(edge_contrast(frame, axis),
                                   reference=sharpness_score(frame, axis=axis))

    def _sync_spectrum_button(self) -> None:
        """Dragging the spectrum closed also releases its history buffer."""
        visible = self._splitter.sizes()[1] > 0 and not self.spectrum_panel.isHidden()
        if self._spectrum_button.isChecked() != visible:
            self._spectrum_button.blockSignals(True)
            self._spectrum_button.setChecked(visible)
            self._spectrum_button.blockSignals(False)
            if not visible:
                self.spectrum_panel.clear()
            if hasattr(self, '_pipeline'):
                self._sync_spectrum_history()
                self._repaint_belt()

    def _set_side_visible(self, visible: bool) -> None:
        """Hide / reopen the analytics panel; its width goes to / comes from the images."""
        main, side = self._side_splitter.sizes()
        if visible and side == 0:
            width = self._side_width or max(self._side_tabs.minimumWidth(), (main + side) // 4)
            self._side_splitter.setSizes([main - width, width])
        elif not visible and side > 0:
            self._side_width = side
            self._side_splitter.setSizes([main + side, 0])

    def _sync_side_button(self) -> None:
        visible = self._side_splitter.sizes()[1] > 0
        if self._side_button.isChecked() != visible:
            self._side_button.blockSignals(True)
            self._side_button.setChecked(visible)
            self._side_button.blockSignals(False)

    def _settings(self) -> QSettings:
        return QSettings("NARIT-TSC", "PushbroomSortingSimulation")

    def _restore_layout(self) -> None:
        state = self._settings().value(self._LAYOUT_KEY)
        restored = state is not None and self._splitter.restoreState(state)
        side_state = self._settings().value(self._SIDE_LAYOUT_KEY)
        side_restored = side_state is not None and self._side_splitter.restoreState(side_state)
        focus = self._settings().value(self._FOCUS_KEY, True)
        self._focus_button.setChecked(focus not in (False, 'false', '0', 0))
        spectrum = self._settings().value(self._SPECTRUM_KEY, True)
        self._spectrum_button.setChecked(spectrum not in (False, 'false', '0', 0))
        # Sizes are only real once the window is laid out; set the defaults and sync then.
        QTimer.singleShot(0, lambda: self._finish_layout(restored, side_restored))

    def _finish_layout(self, restored: bool, side_restored: bool = True) -> None:
        total = sum(self._splitter.sizes())
        width = sum(self._side_splitter.sizes())
        if total == 0 or width == 0:                # not laid out yet: try again shortly
            QTimer.singleShot(50, lambda: self._finish_layout(restored, side_restored))
            return
        if not side_restored:                       # analytics panel a quarter of the width
            side = max(self._side_tabs.minimumWidth(), width // 4)
            self._side_splitter.setSizes([width - side, side])
        self._sync_side_button()
        if not restored:
            spectrum = max(self.spectrum_panel.minimumHeight(), total // 3)
            self._splitter.setSizes([max(self._canvas_label.minimumHeight(), total - spectrum), spectrum])
        if not self._spectrum_button.isChecked():
            self.spectrum_panel.hide()
            self._splitter.setSizes([total, 0])
        self._sync_spectrum_button()

    def _save_layout(self) -> None:
        settings = self._settings()
        settings.setValue(self._LAYOUT_KEY, self._splitter.saveState())
        settings.setValue(self._SIDE_LAYOUT_KEY, self._side_splitter.saveState())
        settings.setValue(self._FOCUS_KEY, self._focus_button.isChecked())
        settings.setValue(self._SPECTRUM_KEY, self._spectrum_button.isChecked())

    def _plan_text(self) -> str:
        """Where every correction/calibration value came from (first file of the playlist)."""
        plan = self._pipeline.plans[0]
        src = self._pipeline.sources[0]
        if self._live:
            parts = [f"live camera {src.name} {src.frame_shape} {src.dtype}, stage={plan.stage}"]
        else:
            parts = [f"{len(self._pipeline.sources)} file(s), first {src.name} {src.frame_shape} {src.dtype}, "
                     f"stage={plan.stage}"]
        parts += [f"{key}: {value}  [{origin}]" for key, (value, origin) in plan.provenance.items()]
        return "\n".join(parts)

    def _setup_fps_control(self) -> None:
        """Presets for the simulated line rate. The recording's exposure only selects the
        preprocessing files (dark etc.); the simulation speed is set here, independently.

        A live camera free-runs at the rate its exposure and readout allow: the box then shows
        the measured line rate, read-only (refreshed with the status line)."""
        if self._live:
            self._fps_spin.setEnabled(False)
            self._fps_spin.setSpecialValueText("measuring…")
            self._fps_spin.setValue(0)
            self._fps_spin.setToolTip("Measured line rate of the live camera. It free-runs at the rate its exposure "
                                      "and readout allow\n(acquisition.exposure_ms); file playback sets the rate here.")
            square = self._square_pixel_fps()
            if square:
                self._belt_label.setToolTip(f"Square pixels on the belt need {square:g} lines/s at "
                                            f"{self.config.camera.belt_speed_m_s * 60:g} m/min")
            return
        square = self._square_pixel_fps()
        values = sorted({100, 250, 500, 1000, 2000} | ({square} if square else set())) + [0]
        for value in values:
            button = QPushButton('Max' if value == 0 else f"{value:g}")
            button.setObjectName("presetButton")
            if square and value == square:
                button.setText(f"{value:g} ■")
                button.setToolTip(f"Square pixels at {self.config.camera.belt_speed_m_s * 60:g} m/min: belt speed / "
                                  f"{self._belt_pixel_mm():.3g} mm per pixel on the belt (optics)")
            if value == 0:
                button.setToolTip("Max: the camera delivers the next line as soon as a buffer is free - "
                                  "the pipeline's lossless maximum throughput")
            button.clicked.connect(lambda _, v=value: self._fps_spin.setValue(v))
            self._fps_presets.addWidget(button)
        self._fps_spin.setValue(int(round(self.config.camera.fps)))
        self._fps_spin.valueChanged.connect(self._on_fps_changed)
        self._update_fps_labels(self._fps_spin.value())

    def _on_fps_changed(self, value: int) -> None:
        self._pipeline.set_fps(value)
        self.performance_panel.reset_requested.emit()     # stage averages restart at the new rate
        self._update_fps_labels(value)

    def _belt_pixel_mm(self) -> float | None:
        """One slit pixel on the production belt (mm), from the optics section."""
        return self.config.optics.belt_pixel(self._pipeline.plans[0].camera_bin or self.config.acquisition.camera_bin)

    def _square_pixel_fps(self) -> int | None:
        """Line rate at which one line covers as much belt as one slit pixel (square pixels)."""
        pixel = self._belt_pixel_mm()
        speed = self.config.camera.belt_speed_m_s
        return int(round(speed * 1000.0 / pixel)) if pixel and speed else None

    def _show_live_rate(self, fps: float, stalled: bool = False) -> None:
        """Live camera: the measured line rate in the (read-only) rate box and the belt readout."""
        if not fps:
            if stalled:                                # no stale rate while the camera sends nothing
                self._fps_spin.setSpecialValueText("no frames")
                self._fps_spin.setValue(0)
                self._belt_label.setText("no frames from the camera")
            return
        self._fps_spin.setValue(int(round(fps)))
        self._update_fps_labels(fps)

    def _update_fps_labels(self, fps: float) -> None:
        speed = self.config.camera.belt_speed_m_s
        if not fps:
            self._belt_label.setText("lossless maximum")
            return
        mm = speed * 1000.0 / fps
        text = f"{mm:.3g} mm/line at {speed * 60:g} m/min"
        pixel = self._belt_pixel_mm()
        if pixel:
            text += f" = {mm / pixel:.2g} × the {pixel:.3g} mm pixel across the belt"
        self._belt_label.setText(text)

    _STEP_LABELS = (('dark', 'Dark + Hot pixel'), ('flip', 'Flip'), ('smile', 'Smile'), ('binning', 'Binning'),
                    ('flatfield', 'Flat-field'), ('reflectance', 'Reflectance'))

    def _setup_step_switches(self) -> None:
        """One checkbox per step; a step the input cannot have (no reference, binned input) is
        disabled with the reason as tooltip. Changes reach the workers at the next line."""
        plans = self._pipeline.plans
        for name, box in self._step_boxes.items():
            box.blockSignals(True)
            infos = [plan.steps.get(
                name, {'available': False, 'on': False, 'why': ''}) for plan in plans]
            box.setChecked(self._pipeline.effective_steps().get(name, False))
            available = any(info['available'] for info in infos)
            box.setEnabled(available)
            tip = infos[0]['why']
            if len(plans) > 1 and len({info['why'] for info in infos}) > 1:
                n_have = sum(i['available'] for i in infos)
                tip += f"\n(first file; {n_have}/{len(plans)} files have this step)"
            if name == 'binning' and infos[0]['on']:
                # Always applied to raw input: shown checked and active (theme colour), but locked,
                # so it does not look like a step that is off.
                plan = plans[0]
                box.setEnabled(True)
                box.setAttribute(Qt.WA_TransparentForMouseEvents, True)
                box.setFocusPolicy(Qt.NoFocus)
                box.setText("Binning 🔒")
                tip = (f"Always on (the model needs wavelength bins): {plan.provenance['binning'][0]}"
                       if 'binning' in plan.provenance else tip)
            box.setToolTip(tip)
            box.blockSignals(False)
            if available and name != 'binning' and name not in self._step_connected:
                box.toggled.connect(lambda checked, n=name: self._on_step_toggled(n, checked))
                self._step_connected.add(name)
        self._show_flatfield_state()
        self._update_units_warning()

    def _reflectance_flatfield_state(self) -> tuple[str, str, str] | None:
        """How the flat-field relates to the reflectance factors in use.

        ``None`` while Reflectance is off. Else ``(state, profile in the factors, loaded profile)``
        with state ``'in'`` (the factors contain it: it is active though its own step is off),
        ``'mismatch'`` (they contain another profile than the one loaded) or ``'absent'``.
        """
        pipeline = self._pipeline
        if not pipeline.effective_steps().get('reflectance'):
            return None
        plan = pipeline.plans[0]
        inside = getattr(plan, 'reflectance_flatfield', None)
        if not inside:
            return 'absent', '', ''

        def short(label: str) -> str:
            return Path(label).parent.name or str(label)

        loaded = getattr(plan, 'flatfield_profile', None)
        if loaded is not None and str(loaded.label) != str(inside):
            return 'mismatch', short(inside), short(loaded.label)
        return 'in', short(inside), ''

    def _input_text(self) -> str:
        """What the model receives now, for the heading: counts or reflectance, with or without flat-field."""
        steps = self._pipeline.effective_steps()
        state = self._reflectance_flatfield_state()
        if state is None:
            return 'Counts + flat-field' if steps.get('flatfield') else 'Counts'
        return 'Reflectance + flat-field' if state[0] in ('in', 'mismatch') else 'Reflectance'

    def _show_flatfield_state(self) -> None:
        """Make the Flat-field box show what happens to the data, not only its own switch.

        Factors of a "Flat-field + reference" calibration contain the flat-field, so its step is
        off while Reflectance is on (it must not divide twice) - but it is active: the box is then
        shown checked and locked. Factors without it: unchecked and locked, with the way out.
        """
        labels = dict(self._STEP_LABELS)
        plans = self._pipeline.plans
        box, reflectance_box = self._step_boxes['flatfield'], self._step_boxes['reflectance']
        infos = [plan.steps.get('flatfield', {'available': False, 'why': ''}) for plan in plans]
        reflectance_tip = plans[0].steps.get('reflectance', {}).get('why', '')
        state = self._reflectance_flatfield_state()
        box.blockSignals(True)
        box.setAttribute(Qt.WA_TransparentForMouseEvents, state is not None and state[0] != 'absent')
        box.setFocusPolicy(Qt.NoFocus if state is not None else Qt.StrongFocus)
        reflectance_box.setText(labels['reflectance'])
        if state is None:                                  # Reflectance off: the step is its own switch
            box.setText(labels['flatfield'])
            box.setEnabled(any(info['available'] for info in infos))
            box.setChecked(bool(self._pipeline.effective_steps().get('flatfield')))
            box.setToolTip(infos[0]['why'])
        elif state[0] == 'absent':
            box.setText(labels['flatfield'])
            box.setEnabled(False)
            box.setChecked(False)
            box.setToolTip("Not in this reflectance calibration (per-band or row-wise factors).\n"
                           "To use it with Reflectance: calibrate again with 'Flat-field + reference'.\n"
                           "To use the Flat-field step alone: untick Reflectance first.")
            reflectance_tip += "\nNo flat-field in this calibration"
        else:
            box.setText(f"{labels['flatfield']} 🔒 in reference")
            box.setEnabled(True)
            box.setChecked(True)
            box.setToolTip(f"Active inside the reflectance factors: {state[1]}.\n"
                           "It is not applied a second time; untick Reflectance to switch both off.")
            reflectance_tip += f"\nIncludes flat-field: {state[1]}"
            if state[0] == 'mismatch':
                reflectance_box.setText(f"{labels['reflectance']} ⚠")
                reflectance_tip += (f"\n⚠ The loaded flat-field profile is {state[2]}: the factors were made with "
                                    "another one - calibrate again")
        box.blockSignals(False)
        reflectance_box.setToolTip(reflectance_tip)

    def _on_step_toggled(self, name: str, checked: bool) -> None:
        if checked and name == 'flatfield':
            for plan in self._pipeline.plans:
                flat = plan.flatfield_profile
                if flat is not None:
                    self._pipeline.set_steps(dark=flat.dark, flip=flat.flip, smile=flat.smile)
        self._pipeline.set_steps(**{name: checked})
        self.spectrum_panel.clear()
        self.spectrum_panel.set_note('Processing changed · click a newly acquired line to inspect the current input')
        for step, box in self._step_boxes.items():
            box.blockSignals(True)
            box.setChecked(self._pipeline.effective_steps().get(step, False))
            box.blockSignals(False)
        self._show_flatfield_state()
        self.performance_panel.reset_requested.emit()     # stage averages restart for the new chain
        self._update_units_warning()

    def _update_units_warning(self) -> None:
        classifier = self._pipeline.classifier
        reflectance = self._pipeline.effective_steps()['reflectance']
        wrong = classifier.input_units != 'any' and (
            classifier.input_units == 'reflectance') != reflectance
        placeholder = reflectance and any(
            getattr(p, 'reflectance_placeholder', False) for p in self._pipeline.plans)
        streamed = [src.name for src in self._pipeline.sources if src.storage == 'stream']
        self._units_warning.setVisible(wrong or placeholder or bool(streamed))
        if streamed and not (wrong or placeholder):
            self._units_warning.setText("⚠ streamed from HDF5: timing includes decoding")
            self._units_warning.setToolTip(
                f"{', '.join(streamed)}: frames are decompressed while playing (the simulator's own cost, a real "
                "camera has none).\nFor speed measurements keep the frames in RAM or the disk cache "
                "(source.cache = ram/disk, enough free memory/disk), or play one file at a time.")
        elif placeholder and not wrong:
            self._units_warning.setText("⚠ reflectance factors are a placeholder")
            self._units_warning.setToolTip("reflectance/<source>.json marks them as stand-in factors.\n"
                                           "Speed is real; replace them with tools/make_reflectance.py on a real "
                                           "white reference.")
        elif wrong:
            given = 'reflectance' if reflectance else 'counts'
            self._units_warning.setText(f"⚠ model expects {classifier.input_units}, gets {given}"
                                        + (" · placeholder factors" if placeholder else ""))
            source = 'declared in the artifact' if classifier.input_units_declared else \
                'assumed: the artifact has no input_units'
            self._units_warning.setToolTip(
                f"{classifier.path}: input units {classifier.input_units} ({source}).\n"
                "Speed measurements stay valid; the classes are not meaningful.")

    # --------------------------------------------------------------------------- models ---

    def _scan_models(self) -> None:
        """List the models this input can use (loaded and bound in a thread; the pipeline keeps running)."""
        self._model_scan += 1
        number, config, pipeline = self._model_scan, self.config, self._pipeline
        self.model_selector.set_scanning()

        def work():
            try:
                entries = scan_models(config, pipeline.plans, pipeline.classifier)
            except Exception:
                logger.exception("Looking for models failed")
                entries = []
            self._models_scanned.emit(entries, number)

        threading.Thread(target=work, name='model-scan', daemon=True).start()

    def _show_models(self, entries, number: int) -> None:
        if number != self._model_scan:                 # an older scan (pipeline restarted meanwhile)
            return
        self._scanned_entries = list(entries)
        self._show_model_tiles()

    def _show_model_tiles(self) -> None:
        """The model files of the last scan, then this session's quick models (bound to the input now)."""
        session = [session_entry(c, self.config, self._pipeline.plans) for c in self._session_models]
        self.model_selector.set_entries(self._scanned_entries + session)
        self._select_current_model()

    def _open_quick_model_dialog(self) -> None:
        """Train…: label the newest lines, train a session model (see widgets/quick_model_dialog.py)."""
        if self._reflectance_dialog is not None:
            return
        from widgets.quick_model_dialog import QuickModelDialog
        pipeline = self._pipeline
        if not pipeline.spectrum_lines:                # the Spectrum panel is hidden: keep lines meanwhile
            pipeline.start_spectrum_history()
        units = 'reflectance' if pipeline.effective_steps().get('reflectance') else 'counts'
        dialog = QuickModelDialog(self, lambda: pipeline.spectrum_snapshot(QUICK_MODEL_MB), pipeline.plans[0].wavelengths, units,
                                  self._quick_labels)
        dialog.model_ready.connect(self._add_session_model)
        self._quick_dialog = dialog                    # reachable while it runs (tests)
        try:
            dialog.exec_()
        finally:
            self._quick_dialog = None
            self._sync_spectrum_history()

    def _add_session_model(self, artifact: dict) -> None:
        """A quick model was trained: a tile beside the model files, and the pipeline switches to it."""
        from inference_engine import PushbroomLineClassifier
        self._quick_number += 1
        classifier = PushbroomLineClassifier.from_artifact(artifact, f"Quick {self._quick_number}")
        self._session_models.append(classifier)
        self._show_model_tiles()
        self._on_model_chosen(session_entry(classifier, self.config, self._pipeline.plans))

    def _remove_session_model(self, entry) -> None:
        """Forget a quick model; one that is running keeps running until another model is chosen."""
        name = str(entry.path)
        self._session_models = [c for c in self._session_models if c.path != name]
        self._show_model_tiles()

    def _select_current_model(self) -> None:
        """Highlight the pipeline's model in the selector."""
        self.model_selector.set_current(self._pipeline.classifier)

    def _on_model_chosen(self, entry) -> None:
        """Switch to the model of the tile the operator clicked (``entry``: a ``ModelEntry``)."""
        if entry is None or not entry.usable or entry.classifier is self._pipeline.classifier:
            self._select_current_model()
            return
        try:
            generation = self._pipeline.set_classifier(entry.classifier)
        except ValueError as exc:                      # the input changed since the scan
            logger.warning("Model %s not used: %s", entry.path, exc)
            QMessageBox.warning(self, "Model", f"{entry.path.name} does not fit the current input:\n{exc}")
            self._select_current_model()
            return
        self._generation_models[generation] = entry.classifier
        # a later pipeline restart (sensor settings) keeps the chosen model; a session model is
        # not a file, so the restart goes back to the last model file
        if not entry.session:
            self.config = replace(self.config, model=replace(self.config.model, path=str(entry.path)))
        self._select_current_model()
        self._update_units_warning()
        self._info_label.setText(self._plan_text())

    def _show_model_classes(self, generation: int) -> None:
        """Class names and colours of the model whose lines the pipeline now reports."""
        if generation == 0:
            self._generation_models = {0: self._pipeline.classifier}
        classifier = self._generation_models.get(generation, self._pipeline.classifier)
        self._generation_models = {g: c for g, c in self._generation_models.items() if g >= generation}
        self._shown_generation = generation
        # a model package brings the class colours chosen at training (ids without one keep Tab10)
        self._lut = generate_tab10_lut(getattr(classifier, 'class_colors', None))
        self._show_training_spectra(classifier)
        self.class_legend_widget.setup_classes_from_model(
            category_names=classifier.category_names, lut=self._lut, classes=classifier.classes_,
            hidden=self.config.display.hidden_classes)
        self._prev_counts = np.zeros(256, np.int64)
        self._prev_counted_lines = 0

    _BACKEND_CHOICES = (('auto', 'Auto (GPU if usable, else CPU fused)'), ('cpu_fused', 'CPU fused (fast, checked vs hsi)'),
                        ('cpu', 'CPU (hsi reference)'),
                        ('cupy_cuda', 'CuPy CUDA'), ('opencv_cuda', 'OpenCV CUDA'))

    def _setup_backend_selector(self) -> None:
        caps = get_compute_capabilities()
        combo = self._backend_combo
        if combo.count():                       # pipeline replaced: only the selection follows it
            self._pipeline.set_backend(combo.currentData())
            return
        for key, text in self._BACKEND_CHOICES:
            capability = getattr(caps, key, None)
            if capability is not None and not capability.available:
                text += "  — not available"
            combo.addItem(text, key)
            if capability is not None:
                combo.setItemData(combo.count() - 1,
                                  capability.reason or capability.device_name, Qt.ToolTipRole)
                if not capability.available:
                    combo.model().item(combo.count() - 1).setEnabled(False)
        combo.setCurrentIndex(max(0, combo.findData(self.config.corrections.backend)))
        if self._pipeline.plans[0].stage == 'binned':
            combo.setEnabled(False)
            combo.setToolTip("Input is already corrected and binned: no correction backend runs")
        combo.currentIndexChanged.connect(lambda _: self._pipeline.set_backend(combo.currentData()))

    # ------------------------------------------------------------------------- refresh ---

    def _forget_previous_counts(self) -> None:
        # Reset Batch zeroes the pipeline's counts in this thread, so the next real-time delta
        # starts from zero even when the batch is already as long again by the next refresh.
        self._prev_counts = np.zeros(256, np.int64)
        self._prev_counted_lines = 0

    def _refresh_gui_slot(self) -> None:
        columns, step = self._canvas_columns(), self._belt_step()
        raw_columns = self._raw_zone_columns(columns)
        # statistics count a line when it leaves the raw zone, in step with the belt map
        self._count_delay = raw_columns * self.config.display.canvas_line_stride
        s = self._pipeline.snapshot(canvas_columns=columns, raw_columns=raw_columns, step=step,
                                    count_delay_lines=self._count_delay)
        if s.model_generation != self._shown_generation:      # first lines of a newly chosen model
            self._show_model_classes(s.model_generation)
        # Images at gui_hz; the class statistics at ~10 Hz and the performance charts at ~5 Hz
        # (their stage numbers change every 0.5 s): less GUI time holding the GIL.
        self._tick = getattr(self, '_tick', 0) + 1
        hz = max(1.0, self.config.display.gui_hz)
        legend_due = self._tick % max(1, round(hz / 10)) == 0
        if legend_due:
            self.throughput_strip.update_snapshot(s)
        charts_due = self._tick % max(1, round(hz / 5)) == 0
        focus_due = self._tick % max(1, round(hz * FocusPanel.INTERVAL_S)) == 0

        if s.raw_frame is not None and not self._preview_collapsed():
            self._set_pixmap(self._raw_label, _to_gray8(
                s.raw_frame, self._raw_label.size()), Qt.KeepAspectRatio)
        if focus_due:
            self._update_focus(s.raw_frame)
        if s.canvas.shape[1] > 0:
            self._belt = {'canvas': s.canvas, 'raw': s.raw_canvas, 'lines': s.canvas_lines, 'step': step}
            self._show_belt(s.canvas, s.raw_canvas, step[1])

        # Counts accumulate per line in the pipeline; "real-time" is every line since the
        # previous refresh. A reset makes the totals smaller than the previous values.
        if legend_due:
            if s.counted_lines < self._prev_counted_lines:
                self._prev_counts[:] = 0
            recent = s.counts - self._prev_counts
            self._prev_counts, self._prev_counted_lines = s.counts, s.counted_lines
            self.class_legend_widget.update_counts(recent, s.counts, s.counted_lines)
        if charts_due and self.performance_panel.isVisible():
            self.performance_panel.update_snapshot(s)

        name, index, n_frames, cycle = s.position
        total = s.frames_delivered + s.lines_lost
        lost_pct = 100.0 * s.lines_lost / total if total else 0.0
        mm = (f" · {self.config.camera.belt_speed_m_s * 1000.0 / s.target_fps:.1f} mm/line"
              if s.target_fps else "")
        stage_total = s.stage_ms.get('total', 0.0)
        state = s.state.upper() if s.state != 'running' else 'RUN'
        stalled = False
        if s.source_kind == 'camera':
            # free-running camera: target_fps is its measured frame rate (0 = nothing delivered lately)
            stalled = s.state == 'running' and s.elapsed_s > 1.0 and s.target_fps == 0
            target = f"{s.target_fps:,.0f}" if s.target_fps else (
                '0 (no frames)' if stalled else '…')
            where = f"{name} frame {index + 1:,}"
            lost_detail = (f", camera {s.lines_lost_camera:,}" if s.lines_lost_camera else "") + \
                (f", read timeouts {s.grab_timeouts:,}" if s.grab_timeouts else "")
            if legend_due:
                self._show_live_rate(s.target_fps, stalled)
        else:
            target = 'max' if s.target_fps == 0 else f"{s.target_fps:g}"
            where = f"{name} [{index + 1}/{n_frames}] cycle {cycle}"
            lost_detail = ""
        text = (f"{state} | {where} | "
                f"line rate {target}/s{mm} | camera {s.camera_fps:,.0f}/s · processed {s.processed_fps:,.0f}/s\n"
                f"lines {s.lines_processed:,} · lost {s.lines_lost:,} ({lost_pct:.2f}%{lost_detail}) · "
                f"buffers {s.buffers_in_use}/{s.buffer_frames} · latency p50 {s.latency_ms['p50']:.2f} "
                f"p99 {s.latency_ms['p99']:.2f} max {s.latency_ms['max']:.1f} ms · "
                f"{s.backend} {stage_total:.2f} ms/line/worker")
        if s.error:
            text += f"\nERROR: {s.error}"
        self._status_label.setText(text)
        run_text = ('ERROR' if s.error else 'NO FRAMES' if stalled else state)
        source_text = 'LIVE CAMERA' if s.source_kind == 'camera' else 'PLAYBACK'
        self._run_label.setText(f"{run_text} · {source_text}")
        model_name = Path(str(getattr(self._pipeline.classifier, 'path', 'model'))).name
        context = f"{name}  ·  Model: {model_name}  ·  Input: {self._input_text()}"
        if s.error:
            context = f"{s.error}  ·  {context}"
        self._context_label.setText(context)
        self._context_label.setToolTip(context)
        set_accent(self._run_label, 'error' if s.error else 'warn' if s.lines_lost or stalled else 'ok')
        note = f" — {s.backend_note}" if s.backend_note else ""
        self._backend_label.setText(
            f"running: {s.backend}  (requested {s.requested_backend}){note}")
        set_accent(self._backend_label, 'warn' if s.backend_note else 'ok')
        # ok / warn / error: stripe and text colours come from the theme
        set_accent(self._status_label,
                   'error' if s.error else 'warn' if s.lines_lost or stalled else 'ok')

    def _canvas_columns(self) -> int:
        """Belt-map columns that fill the label at the true pixel proportions.

        One slit pixel is the unit; a drawn column is ``canvas_line_stride`` lines of
        ``line_aspect`` slit pixels each, so the label shows ``rows * (w / h)`` slit-pixel widths.
        """
        disp = self.config.display
        w, h = max(1, self._canvas_label.width()), max(1, self._canvas_label.height())
        rows = self._pipeline.sources[0].frame_shape[0]
        return max(1, int(round(rows * w / h / (disp.line_aspect * disp.canvas_line_stride))))

    def _raw_zone_columns(self, columns: int) -> int:
        """Columns of the pre-classification (raw) zone at the right end of the belt map."""
        return int(round(columns * self.config.display.raw_zone_fraction))

    def _belt_step(self) -> tuple[int, int]:
        """(rows, columns) decimation of the belt map down to the label's resolution."""
        rows = self._pipeline.sources[0].frame_shape[0]
        h, w = max(1, self._canvas_label.height()), max(1, self._canvas_label.width())
        return max(1, rows // h), max(1, self._canvas_columns() // w)

    def _belt_image(self, canvas: np.ndarray, raw: np.ndarray | None = None, column_step: int = 1) -> np.ndarray:
        """RGB belt map at true proportions, newest line on the right, empty (black) before the start.

        The newest ``display.raw_zone_fraction`` of the columns show ``raw`` (model-input intensity)
        in grey; the older ones the class colours. They are one continuous strip of belt: a line
        is drawn grey first and in colour once it has moved into the classified zone.
        ``canvas`` / ``raw`` hold every ``column_step``-th column, counted from the newest.
        """
        needed = self._canvas_columns()
        # drawn columns (offsets from the oldest)
        grid = np.arange(needed - 1, -1, -column_step)[::-1]
        if canvas.shape[1] < len(grid):         # not enough history yet: pad the left side
            pad = np.zeros((canvas.shape[0], len(grid) - canvas.shape[1]), canvas.dtype)
            canvas = np.concatenate([pad, canvas], axis=1)
        image = np.take(self._lut, canvas, axis=0)
        zone = self._raw_zone_columns(needed)
        if raw is None or zone == 0:
            return image
        first = int(np.searchsorted(grid, needed - zone))    # first drawn column of the raw zone
        # raw zone before any line reached it: black
        image[:, first:image.shape[1] - raw.shape[1]] = 0
        if raw.shape[1]:
            self._gray_into(raw, image[:, image.shape[1] - raw.shape[1]:])
        return image

    def _gray_into(self, raw: np.ndarray, out: np.ndarray) -> None:
        """Intensity -> grey RGB in ``out``, smoothed 1-99 % range; NaN (lost line) in the lost colour."""
        finite = np.isfinite(raw)
        sample = raw[::4, ::4][finite[::4, ::4]]
        if sample.size:
            lo, hi = np.percentile(sample, (1.0, 99.0))
            if self._raw_range is None:
                self._raw_range = (float(lo), float(hi))
            else:                               # steady grey levels: follow the data slowly
                a = 0.1
                self._raw_range = tuple((1 - a) * old + a * new for old,
                                        new in zip(self._raw_range, (lo, hi)))
        lo, hi = self._raw_range or (0.0, 1.0)
        gray = np.where(finite, raw, np.float32(lo))
        gray -= lo
        gray *= np.float32(255.0 / max(hi - lo, 1e-6))
        np.clip(gray, 0, 255, out=gray)
        out[...] = gray.astype(np.uint8)[..., None]
        if not finite.all():
            out[~finite] = self._lut[LOST_CLASS]

    def _show_belt(self, canvas: np.ndarray, raw: np.ndarray | None, column_step: int = 1) -> None:
        """Belt map with the zone divider and captions painted at screen resolution."""
        self._set_pixmap(self._canvas_label, self._belt_image(
            canvas, raw, column_step), Qt.IgnoreAspectRatio)
        needed = self._canvas_columns()
        zone = self._raw_zone_columns(needed)
        pixmap = self._canvas_label.pixmap()
        marks = self._spectrum_marks()
        if (zone == 0 and not marks) or pixmap is None or pixmap.isNull():
            return
        tokens = QApplication.instance().property('theme_tokens')
        pixmap = pixmap.copy()
        w, h = pixmap.width(), pixmap.height()
        painter = QPainter(pixmap)
        font = QFont(tokens['font_family'])
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        if zone:
            x = int(round(w * (needed - zone) / needed))
            painter.setPen(QPen(QColor(tokens['accent']), 2, Qt.DashLine))
            painter.drawLine(x, 0, x, h)
            for text, left in (("◀ CLASSIFIED  (inference zone)", 6), ("RAW  (before classification)", x + 6)):
                box = metrics.boundingRect(text).adjusted(-5, -2, 5, 2)
                box.moveTo(left, 6)
                painter.fillRect(box, QColor(tokens['image_overlay']))   # #AARRGGBB
                painter.setPen(QColor(tokens['image_overlay_text']))
                painter.drawText(box, Qt.AlignCenter, text)
        # clicked pixels (spectrum preview): they travel with the belt until they leave the map
        painter.setRenderHint(QPainter.Antialiasing)
        for point, fx, fy in marks:
            cx, cy = fx * w, fy * h
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(Qt.black, 4))           # halo: the ring shows on any class colour
            painter.drawEllipse(QPointF(cx, cy), 6.0, 6.0)
            painter.setPen(QPen(QColor(point.colour), 2))
            painter.drawEllipse(QPointF(cx, cy), 6.0, 6.0)
            text = str(point.number)
            box = metrics.boundingRect(text).adjusted(-3, -1, 3, 1)
            box.moveTo(int(cx + 9) if cx + 9 + box.width() < w else int(cx - 9 - box.width()),
                       int(min(max(cy - box.height() / 2, 0), h - box.height())))
            painter.fillRect(box, QColor(tokens['image_overlay']))
            painter.setPen(QColor(point.colour))
            painter.drawText(box, Qt.AlignCenter, text)
        painter.end()
        self._canvas_label.setPixmap(pixmap)

    def _repaint_belt(self) -> None:
        """Draw the belt map again from the last snapshot (marks added or cleared)."""
        if self._belt is not None:
            self._show_belt(self._belt['canvas'], self._belt['raw'], self._belt['step'][1])

    # --------------------------------------------------------------- spectrum preview (click) ---

    def _belt_columns(self) -> tuple[np.ndarray, int]:
        """``(line number of each drawn column, drawn columns)``; the lines fill the right end."""
        lines = self._belt['lines']
        drawn = len(range(self._canvas_columns() - 1, -1, -self._belt['step'][1]))
        return lines, max(drawn, len(lines))

    def _spectrum_marks(self) -> list:
        """``(point, x, y)`` of the clicked pixels still on the map, as fractions of the label."""
        points = self.spectrum_panel.points if self._spectrum_button.isChecked() else []
        if not points or self._belt is None or self._belt['lines'] is None or not len(self._belt['lines']):
            return []
        lines, drawn = self._belt_columns()
        valid = np.flatnonzero(lines >= 0)
        if not valid.size:
            return []
        slit = self._pipeline.sources[0].frame_shape[0]
        per_column = self.config.display.canvas_line_stride * self._belt['step'][1]
        marks = []
        for point in points:
            if not lines[valid[0]] - per_column < point.line <= lines[valid[-1]]:
                continue                                    # scrolled off the map
            k = int(np.searchsorted(lines[valid], point.line))
            column = int(valid[min(k, valid.size - 1)])
            marks.append((point, (drawn - len(lines) + column + 0.5) / drawn, (point.row + 0.5) / slit))
        return marks

    def _show_training_spectra(self, classifier) -> None:
        """Give the spectrum preview the training spectra of ``classifier`` (a model package has them)."""
        package = getattr(classifier, 'package', None)
        training = getattr(package, 'training', None) or {}

        def bands(rows):
            return [TrainingBand(row.label, '#%02X%02X%02X' % tuple(int(v) for v in self._lut[row.class_id]),
                                 row.mean, row.sd, row.n_spectra) for row in rows]

        if not training.get('input'):
            self.spectrum_panel.set_training(None)
            return
        preprocessed = bands(training.get('preprocessed', ()))
        derivative = getattr(package, 'derivative', None)
        selected = getattr(package, 'selected', None)
        self.spectrum_panel.set_training(
            package.wavelengths, bands(training['input']),
            snv_bands=preprocessed if package.snv else (),
            derivative_bands=preprocessed if derivative is not None and not package.snv else (),
            derivative=derivative,
            selected=None if selected is None else package.wavelengths[np.asarray(selected, int)])

    def _probe_spectrum(self, pos) -> None:
        """Add the spectrum of the belt-map pixel under ``pos`` (label pixels) to the preview."""
        panel = self.spectrum_panel
        if self._belt is None or self._belt['lines'] is None:
            return
        w, h = max(1, self._canvas_label.width()), max(1, self._canvas_label.height())
        slit = self._pipeline.sources[0].frame_shape[0]
        row = int(np.clip(pos.y() / h * slit, 0, slit - 1))
        # the map keeps every step[0]-th slit row: take the row it shows, so the spectrum and the
        # class read from the map belong to the same pixel
        row -= row % max(1, int(self._belt['step'][0]))
        lines, drawn = self._belt_columns()
        column = int(np.clip(pos.x() / w * drawn, 0, drawn - 1)) - (drawn - len(lines))
        if column < 0:
            panel.set_note("no line there yet")
            return
        line = int(lines[column])
        if line < 0:
            panel.set_note("that line was lost: no spectrum")
            return
        values = self._pipeline.spectrum_at(line, row)
        if values is None:
            panel.set_note(f"line {line:,} is no longer kept (only the newest {self._pipeline.spectrum_lines:,} "
                           "lines: display.spectrum_history_mb)")
            return
        plan = self._pipeline.plans[0]
        canvas = self._belt['canvas']
        class_id = int(canvas[min(row // self._belt['step'][0], canvas.shape[0] - 1), column])
        classifier = self._generation_models.get(self._shown_generation, self._pipeline.classifier)
        names = classifier.category_names
        label = str(names.get(class_id, class_id) if hasattr(names, 'get') else class_id)
        on = self._pipeline.effective_steps()
        panel.set_unit("model input: reflectance" if on.get('reflectance') else
                       "model input: counts" if plan.stage == 'raw' else "model input")
        point = panel.add_point(plan.wavelengths, values, line=line, row=row, label=label)
        finite = values[np.isfinite(values)]
        span = f"{finite.min():.4g} – {finite.max():.4g}" if finite.size else "no finite value"
        panel.set_note(f"#{point.number}: {label}, slit row {row}, line {line:,}, values {span}")
        self._repaint_belt()

    @staticmethod
    def _set_pixmap(label: QLabel, image: np.ndarray, aspect) -> None:
        image = np.ascontiguousarray(image)
        h, w = image.shape[:2]
        if image.ndim == 2:
            qimg = QImage(image.tobytes(), w, h, w, QImage.Format_Grayscale8)
        else:
            qimg = QImage(image.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(label.size(), aspect, Qt.FastTransformation))

    # ------------------------------------------------------------- reflectance (zoomable dialog ROI) ---

    def _open_reflectance_dialog(self) -> None:
        if self._reflectance_dialog is not None:
            return
        from sim_config import reflectance_folder
        pipeline = self._pipeline
        plan = pipeline.plans[0]
        folder = reflectance_folder(self.config)
        tables = sorted(p for p in folder.glob('*') if p.suffix.lower() in ('.xlsx', '.xls', '.csv'))
        pipeline.start_calibration_capture(CALIBRATION_LINES)
        dialog = None
        try:
            dialog = ReflectanceDialog(self, tables, plan.wavelengths, self._camera_text(), pipeline.calibration_count,
                                       CALIBRATION_LINES,
                                       lambda: pipeline.calibration_capture(with_clipped=True),
                                       lambda: self._current_factors(), lambda: self._calibration_context())
            if getattr(plan, 'flatfield_profile', None) is not None:
                dialog.kind.addItem('Flat-field + reference (whole slit from a small target)', 'flatfield')
                if pipeline.effective_steps().get('flatfield'):
                    dialog.kind.setCurrentIndex(dialog.kind.findData('flatfield'))
            self._reflectance_dialog = dialog
            dialog.calibration_requested.connect(lambda: self._review_reflectance(dialog))
            dialog.exec_()
        finally:
            if dialog is not None:
                dialog._timer.stop()
                dialog.lines = None
                dialog.deleteLater()
            pipeline.stop_calibration_capture()
            self._reflectance_dialog = None

    def _current_factors(self):
        """``(factors, label)`` the reflectance step holds now, or ``None`` (for *Check spectrum*)."""
        plan = self._pipeline.plans[0]
        if plan.reflectance_factors is None:
            return None
        label = str(plan.steps.get('reflectance', {}).get('why') or 'factors in use')
        return plan.reflectance_factors, label + (' (placeholder)' if plan.reflectance_placeholder else '')

    def _dark_level(self, dark_mean) -> float:
        """Median of the dark frame, computed once per dark (asked at every drawn area)."""
        cached = getattr(self, '_dark_level_cache', None)
        if cached is None or cached[0] is not dark_mean:
            cached = self._dark_level_cache = (dark_mean, float(np.median(dark_mean)))
        return cached[1]

    def _calibration_context(self) -> dict:
        """Full scale, dark and flat-field state for the Reflectance dialog's signal check."""
        pipeline = self._pipeline
        plan, meta = pipeline.plans[0], pipeline.sources[0].metadata or {}
        steps = pipeline.effective_steps()
        full_scale = meta.get('full_scale') or meta.get('saturation_level')
        dark_on = bool(steps.get('dark')) and plan.dark_mean is not None
        flat = getattr(plan, 'flatfield_profile', None)
        def source(name: str) -> str:               # file and where it was found, as the plan recorded it
            return str(plan.provenance.get(name, (plan.steps.get(name, {}).get('why', ''),))[0])

        flat_label = None
        if flat is not None:
            folder = Path(flat.label).parent
            try:
                flat_label = str(folder.relative_to(Path(self.config.base_dir).resolve()))
            except ValueError:
                flat_label = str(folder)
        return {'full_scale': float(full_scale) if full_scale else None,
                'dark_level': self._dark_level(plan.dark_mean) if dark_on else None,
                'steps': [('Dark + hot pixel', dark_on, source('dark')),
                          ('Flip', bool(steps.get('flip')), str(plan.steps.get('flip', {}).get('why', ''))),
                          ('Smile', bool(steps.get('smile')), source('smile')),
                          ('Binning', True, source('binning'))],
                'flatfield_label': flat_label,
                'flatfield_matches': flat is not None and all(steps.get(k) == getattr(flat, k)
                                                              for k in ('dark', 'flip', 'smile'))}

    def _camera_text(self) -> str:
        """The camera values a calibration is bound to (also written into its JSON)."""
        v = self._camera_values()
        temp = f", sensor {v['temperature_c']:.1f} °C" if v.get('temperature_c') is not None else ''
        return (f"{v['camera']}: bin {v['camera_bin']}, exposure {v['exposure_ms']} ms, gain {v['gain']}{temp}")

    def _camera_values(self) -> dict:
        plan, src = self._pipeline.plans[0], self._pipeline.sources[0]
        meta = src.metadata or {}
        temperature = next((meta[k] for k in ('sensor_temperature_c', 'temperature_c', 'sensor_temperature_c_start')
                            if meta.get(k) is not None), None)
        return {'camera': src.name, 'camera_id': plan.camera_id, 'camera_backend': plan.camera_backend,
                'camera_bin': plan.camera_bin,
                'exposure_ms': plan.exposure_ms, 'gain': plan.gain,
                'temperature_c': None if temperature is None else float(temperature)}

    def eventFilter(self, obj, event) -> bool:
        if (obj is self._canvas_label and event.type() == QEvent.MouseButtonPress
                and event.button() == Qt.LeftButton and self._spectrum_button.isChecked()):
            self._probe_spectrum(event.pos())
            return True
        return super().eventFilter(obj, event)

    def _review_reflectance(self, dialog: ReflectanceDialog) -> None:
        from reflectance_builder import _clean_steps, factors_from_lines, quality
        plan = self._pipeline.plans[0]
        steps = self._pipeline.effective_steps()
        try:
            if plan.stage == 'raw' and not steps['dark']:
                raise ValueError("the dark step is off: the white would keep the dark offset. "
                                 "Turn 'Dark + Hot pixel' on (a dark file for these camera values is needed)")
            from reflectance_builder import reference_values
            lines, rows, line_ids = dialog.selection()
            choice = dialog.choice()
            choice['values'] = reference_values(choice['reference'], plan.wavelengths)
            # Capture is deliberately before Flat-field and Reflectance.
            capture_steps = {k: bool(v) for k, v in steps.items() if k != 'reflectance'}
            capture_steps['flatfield'] = False
            flat = getattr(plan, 'flatfield_profile', None) if choice['kind'] == 'flatfield' else None
            if choice['kind'] == 'flatfield':
                if flat is None or any(steps.get(k) != getattr(flat, k) for k in ('dark', 'flip', 'smile')):
                    raise ValueError('Flat-field requires the same Dark/Flip/Smile settings as its profile')
            factors, white = factors_from_lines(lines, rows, choice['values'], plan.wavelengths,
                                                choice['kind'], steps=capture_steps,
                                                source='drawn in Reflectance dialog',
                                                flatfield=flat.values if flat is not None else None)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(dialog, "Reflectance", str(exc))
            return
        meta = self._pipeline.sources[0].metadata or {}
        full_scale = meta.get('full_scale') or meta.get('saturation_level')
        report = quality(white, factors, choice['values'], float(full_scale) if full_scale else None, rows)
        report['steps'] = _clean_steps(report['steps'])
        check = report['captured_roi_check']
        text = (f"White standard: {choice['label']}  ({choice['kind']} factors {factors.shape})\n"
                f"Area: slit rows {rows[0]}-{rows[1] - 1}, {line_ids.size} lines ({int(line_ids[0])}-{int(line_ids[-1])})\n"
                f"White counts {report['white_counts_min_max'][0]:.0f}-{report['white_counts_min_max'][1]:.0f}"
                + (f", peak {report['white_peak_pct_of_full_scale']:.0f} % of full scale"
                   if 'white_peak_pct_of_full_scale' in report else '')
                + f"\nSpread along the drawn rows: {report['row_spread_pct'] or 0:.1f} %\n"
                f"Factors {report['factor_min_max'][0]:.3g}-{report['factor_min_max'][1]:.3g}"
                + f"\nCaptured ROI check: target {check['target_mean']:.4f}, calibrated {check['calibrated_mean']:.4f}"
                + f" (max band-mean error {check['max_relative_error_pct']:.2f} %)"
                + "\nThis checks the captured ROI, not an independent live measurement."
                + (f"\n⚠ {len(report['weak_bands_nm'])} weak bands (< 2 % of the peak): noisy factors"
                   if report['weak_bands_nm'] else '')
                + f"\nCamera: {self._camera_text()}"
                + self._next_start_note()
                + "\n\nApply now and save (the previous profile goes to history/)?")
        box = QMessageBox(QMessageBox.Question, "Reflectance calibration", text, parent=dialog)
        apply_button = box.addButton("Apply && Save", QMessageBox.AcceptRole)
        again = box.addButton("Draw again", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Cancel)
        box.exec_()
        if box.clickedButton() is again:
            return
        if box.clickedButton() is apply_button:
            info = {'placeholder': False, 'method': 'drawn in Reflectance dialog (GUI)', 'reference': choice['label'],
                    'kind': choice['kind'], 'rows': list(rows),
                    'lines': [int(line_ids[0]), int(line_ids[-1])], 'frames_averaged': int(line_ids.size),
                    'steps': report['steps'], 'quality': report, **self._camera_values()}
            info['factor_input_space'] = 'pre_flatfield_counts'
            info['flatfield_profile'] = flat.label if flat is not None else None
            if self._apply_calibration(factors, info):
                dialog.accept()

    def _next_start_note(self) -> str:
        """Warning when the saved calibration will not be the one loaded at the next start."""
        cor = self.config.corrections
        if cor.reflectance not in (None, '', 'none', 'auto'):
            return (f"\n⚠ corrections.reflectance is the file {cor.reflectance}: the next start loads that, "
                    "not this calibration")
        return ''

    def _apply_calibration(self, factors: np.ndarray, info: dict) -> bool:
        from reflectance_builder import backup_profile, save_profile
        from sim_config import reflectance_save_path
        plan, src = self._pipeline.plans[0], self._pipeline.sources[0]
        path = reflectance_save_path(self.config, plan, src)    # camera profile: refl_b<bin>_g<gain>.npy
        name = path.stem
        from sim_config import reflectance_flatfield
        inside = reflectance_flatfield(info)               # the flat-field profile these factors contain
        kind = 'flat-field + reference' if inside else 'row-wise' if factors.ndim == 2 else 'per band'
        label = f"{path.name} ({kind} {factors.shape}, drawn in dialog, {info['reference']})"
        if inside:
            label += f", includes flat-field {Path(inside).parent.name or inside}"
        moved = []
        try:
            moved = backup_profile(path)
            save_profile(path.parent, [name], factors, plan.wavelengths, info)
        except Exception as exc:
            for backup in moved:                         # put the previous profile back
                suffix = backup.name[len(path.stem) + 1 + len('YYYYmmdd-HHMMSS'):]
                try:
                    backup.replace(path.with_name(path.stem + suffix))
                except OSError:
                    logger.exception("Could not restore %s", backup)
            logger.exception("Reflectance calibration not saved")
            QMessageBox.critical(self, "Reflectance", f"Not saved, nothing changed:\n{exc}")
            return False
        try:
            self._pipeline.set_reflectance_factors(factors, label, flatfield=inside)
        except Exception as exc:
            logger.exception("Reflectance calibration saved but not applied")
            QMessageBox.critical(self, "Reflectance", f"Saved to {path.name} but not applied:\n{exc}")
            return False
        logger.info("Reflectance calibrated on RAW: %s%s", path,
                    f" (previous moved to {moved[0].parent})" if moved else '')
        self._setup_step_switches()
        self.spectrum_panel.clear()
        self.spectrum_panel.set_note('Calibration applied · old spectra cleared; click a newly acquired line to verify')
        self._info_label.setText(self._plan_text())
        self.performance_panel.reset_requested.emit()
        return True

    # ------------------------------------------------------------------- sensor settings ---

    def _open_sensor_dialog(self) -> None:
        if self._reflectance_dialog is not None:
            return
        plan = self._pipeline.plans[0]
        dialog = SensorDialog(self, self.config, plan, self._config_path)
        if not dialog.exec_():
            return
        values = dialog.values()
        acquisition = replace(self.config.acquisition, exposure_ms=values['exposure_ms'], gain=values['gain'],
                              camera_bin=values['camera_bin'])
        new_config = replace(self.config, acquisition=acquisition).validate()
        if self._restart_pipeline(new_config) and values['save']:      # never save values that failed
            try:
                dialog.save(values)
            except Exception as exc:
                QMessageBox.warning(self, "Sensor", f"Settings not saved to the config file:\n{exc}")

    def _restart_pipeline(self, config) -> bool:
        """Stop the pipeline (closes the camera) and build it again with ``config``."""
        self._gui_timer.stop()
        old_config = self.config
        self._status_label.setText("Restarting with the new sensor settings…")
        QApplication.processEvents()
        if not self._pipeline.stop():
            QMessageBox.warning(self, "Sensor", "The pipeline did not stop (camera busy); try again.")
            self._gui_timer.start()
            return False
        ok = True
        try:
            pipeline = build_pipeline(config)
        except Exception as exc:
            logger.exception("Restart with the new sensor settings failed")
            QMessageBox.critical(self, "Sensor", f"The camera could not start with these settings:\n{exc}\n\n"
                                                 "Going back to the previous settings.")
            config, ok = old_config, False
            try:
                pipeline = build_pipeline(config)
            except Exception as exc2:
                logger.exception("Restart with the previous sensor settings failed too")
                self._status_label.setText(f"STOPPED: the camera could not be restarted ({exc2}). "
                                           "Close the window and start again.")
                set_accent(self._status_label, 'error')
                QMessageBox.critical(self, "Sensor", f"The previous settings failed too:\n{exc2}\n\n"
                                                     "Close the window and start the program again.")
                return False
        self.config = config
        self._pipeline = pipeline
        self._show_model_classes(0)                    # the new pipeline starts with config.model.path
        self._scan_models()                            # other bands after a bin change: other models fit
        self._setup_backend_selector()
        self._setup_step_switches()
        self._forget_previous_counts()
        self._info_label.setText(self._plan_text())
        self._raw_range = None
        self._belt = None
        self.spectrum_panel.clear()                    # the new pipeline counts its lines from 0
        self._sync_spectrum_history()
        self.performance_panel.reset_requested.emit()
        pipeline.start()
        self._gui_timer.start()
        return ok

    # ------------------------------------------------------------------------- events ---

    def _on_class_color_changed(self, class_id: int, new_rgb: list[int]) -> None:
        if 0 <= class_id < 255:
            self._lut[class_id] = new_rgb          # the map stores class ids: history recolours too
            logger.info("Class %d colour -> RGB %s", class_id, new_rgb)

    def closeEvent(self, event) -> None:
        logger.info("Shutdown sequence initiated...")
        self._gui_timer.stop()
        self._save_layout()
        if not self._pipeline.stop():
            busy = self._pipeline.busy_threads()
            if busy == ['camera'] and getattr(self, '_close_refused', False):
                # Second try and only the camera thread is left, blocked inside the SDK: quit anyway
                # (closing the camera from this thread could crash the SDK; the OS releases it).
                logger.warning(
                    "Closing with the camera thread still blocked in a read; the camera was not closed")
                event.accept()
                return
            # Keep the window (and the threads' owners) alive while a worker is still busy,
            # e.g. a CUDA backend finishing its initialisation; close again to retry.
            self._close_refused = True
            logger.warning("Shutdown pending; pipeline thread(s) still running: %s",
                           ', '.join(busy))
            self._status_label.setText(f"Shutdown pending: waiting for {', '.join(busy)} — close again")
            self._gui_timer.start()                    # keep the window alive meanwhile
            event.ignore()
            return
        logger.info("All pipeline threads stopped. Closing application.")
        event.accept()


def _to_gray8(frame: np.ndarray, size=None) -> np.ndarray:
    """Percentile stretch (1-99 % of a subsample) to 8 bit, for display only.

    The frame is first decimated to about the label size: a 1024 x 1280 frame converted in full
    30 times a second holds the GIL long enough to slow the frame workers down.
    """
    if size is not None and size.width() > 0 and size.height() > 0:
        # the image keeps its aspect ratio, so the smaller scale decides the displayed size
        scale = min(size.height() / frame.shape[0], size.width() / frame.shape[1])
        frame = frame[::max(1, int(1.0 / max(scale, 1e-6))), ::max(1, int(1.0 / max(scale, 1e-6)))]
    sample = frame[::8, ::8]
    lo, hi = np.percentile(sample, (1, 99))
    if hi <= lo:
        hi = lo + 1
    scaled = (frame.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=None, help='simulation.json (default: beside gui.py)')
    parser.add_argument('--source', default=None,
                        help='HDF5/ENVI file or folder (overrides source.path)')
    parser.add_argument('--pattern', default=None)
    parser.add_argument('--fps', type=float, default=None,
                        help='line rate; 0 = as fast as possible')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--theme', default=None,
                        help='theme name in themes/ or a theme JSON (overrides display.theme)')
    parser.add_argument('--live', action='store_true',
                        help='live camera instead of files (source.kind = camera; see the live section)')
    parser.add_argument('--set', action='append', default=[], metavar='SECTION.KEY=VALUE',
                        help='override any simulation.json value, e.g. --set acquisition.exposure_ms=2 (JSON values)')
    args = parser.parse_args()

    overrides = {'camera': {'fps': args.fps}, 'processing': {'workers': args.workers},
                 'display': {'theme': args.theme}, 'source': {'kind': 'camera' if args.live else None}}
    try:
        explicit = parse_set_overrides(args.set)
    except ValueError as exc:
        parser.error(str(exc))
    config = load_config(args.config, explicit=explicit, **overrides)
    if args.source:
        config = replace(config, source=replace(config.source, path=str(Path(args.source).resolve()),
                                                pattern=args.pattern))
    app = QApplication(sys.argv[:1])
    tokens = apply_theme(app, config.display.theme)
    try:
        window = PushbroomScannerMainWindow(config, Path(args.config).resolve() if args.config else None)
    except Exception:
        sys.exit(1)
    use_dark_title_bar(window, bool(tokens.get('dark')))
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
