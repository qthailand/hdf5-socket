# -*- coding: utf-8 -*-
"""Desktop view of the pushbroom sorting simulation.

The window only draws: ``pipeline.Pipeline`` runs camera, workers and statistics in its own
threads and the GUI pulls a :class:`pipeline.Snapshot` ``display.gui_hz`` times per second.
A slow GUI therefore shows fewer refreshes but never loses or double-counts a line.

    python gui.py
    python gui.py --config simulation.json --source dataset/dataset.bip.hdr --fps 300
    python gui.py --theme light                 # themes/<name>.json, or a path to a theme JSON

Colours and fonts live in ``themes/`` (``app.qss`` + one token JSON per theme); this module
only sets object names and the panel/accent properties.
"""
import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QSettings, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QSpinBox, QTabWidget, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
                             QSizePolicy, QSplitter, QVBoxLayout, QWidget)

from modules.compute_runtime import get_compute_capabilities, initialize_compute
from pipeline import LOST_CLASS, build_pipeline
from sim_config import load_config
from themes import apply_theme, make_panel, set_accent, use_dark_title_bar
from utils import generate_tab10_lut
from widgets.class_legend_widget import ClassLegendWidget
from widgets.performance_panel import PerformancePanel, ThroughputStrip

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] (%(threadName)-10s) %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class PushbroomScannerMainWindow(QMainWindow):
    """Raw frame, belt classification map, throughput/latency status and class statistics."""

    def __init__(self, config) -> None:
        super().__init__()
        initialize_compute()
        self.config = config
        self.setWindowTitle("HSI Industrial Scanner — pushbroom simulation")
        self.resize(1280, 760)
        self._init_ui()

        try:
            self._pipeline = build_pipeline(config)
        except Exception as exc:
            logger.critical("Initialization failed: %s", exc, exc_info=True)
            QMessageBox.critical(self, "Initialization Error", f"Core Initialization Failed:\n{exc}")
            raise

        classifier = self._pipeline.classifier
        self._lut = generate_tab10_lut()
        self.class_legend_widget.setup_classes_from_model(
            category_names=classifier.category_names, lut=self._lut, classes=classifier.classes_,
            hidden=self.config.display.hidden_classes)
        self.class_legend_widget.color_changed.connect(self._on_class_color_changed)
        self.class_legend_widget.reset_requested.connect(self._pipeline.reset_counts)
        self.performance_panel.reset_requested.connect(self._pipeline.reset_perf)
        self._setup_backend_selector()
        self._setup_step_switches()
        self._setup_fps_control()
        self._prev_counts = np.zeros(256, np.int64)
        self._prev_counted_lines = 0
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
        self._raw_label.setToolTip("Newest raw frame from the simulated camera (slit × spectral), display only")

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

        # Bottom pane: backend row, status line, provenance info. Dragging the lower handle
        # down shrinks the info panel (down to nothing); the backend row and status stay visible.
        self._preview_button = QPushButton("Spectrogram")
        self._preview_button.setObjectName("toggleButton")
        self._preview_button.setCheckable(True)
        self._preview_button.setChecked(True)
        self._preview_button.setToolTip("Show / hide the real-time raw spectrogram preview "
                                        "(hidden = not rendered, saves GUI time)")
        self._preview_button.toggled.connect(self._set_preview_visible)

        self._side_button = QPushButton("Analytics")
        self._side_button.setObjectName("toggleButton")
        self._side_button.setCheckable(True)
        self._side_button.setChecked(True)
        self._side_button.setToolTip("Show / hide the side panel: Performance and Materials (or drag the vertical handle)")
        self._side_button.toggled.connect(self._set_side_visible)

        backend_row = QHBoxLayout()
        backend_row.addWidget(self._preview_button)
        backend_row.addWidget(self._side_button)
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
        bottom.addWidget(self.throughput_strip)

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
        bottom.addWidget(self._status_label)
        bottom.addWidget(self._info_label, stretch=1)

        # One vertical splitter: [raw spectrogram | belt map | bottom pane]. The preview can be
        # collapsed completely (handle or button); the belt map and the status cannot.
        self._splitter = QSplitter(Qt.Vertical)
        self._splitter.setObjectName("mainSplitter")
        self._splitter.setChildrenCollapsible(False)
        for widget in (self._raw_label, self._canvas_label, self._bottom_pane):
            self._splitter.addWidget(widget)
        self._splitter.setCollapsible(0, True)
        for index, stretch in enumerate((1, 2, 0)):          # extra height goes to the images
            self._splitter.setStretchFactor(index, stretch)
        self._splitter.setHandleWidth(8)
        self._splitter.splitterMoved.connect(lambda *_: self._sync_preview_button())
        self._raw_label.setMinimumHeight(40)
        self._canvas_label.setMinimumHeight(80)
        self._preview_height = None                # preview height before the last collapse

        left = QVBoxLayout()
        left.addWidget(self._splitter)

        # Side panel: real-time pipeline performance (speed) and material statistics.
        self.class_legend_widget = ClassLegendWidget()
        self.performance_panel = PerformancePanel()
        self._side_tabs = QTabWidget()
        self._side_tabs.setObjectName("sideTabs")
        self._side_tabs.tabBar().setElideMode(Qt.ElideNone)
        self._side_tabs.tabBar().setExpanding(False)
        self._side_tabs.addTab(self.performance_panel, "Performance")
        self._side_tabs.addTab(self.class_legend_widget, "Materials")
        self._side_tabs.setMinimumWidth(260)
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

        main = QHBoxLayout()
        main.setContentsMargins(10, 10, 10, 10)
        main.addWidget(self._side_splitter)
        container = QWidget()
        container.setLayout(main)
        self.setCentralWidget(container)
        self._restore_layout()

    # -------------------------------------------------------------------------- splitter ---

    _LAYOUT_KEY = "main_splitter"
    _SIDE_LAYOUT_KEY = "side_splitter"

    def _preview_collapsed(self) -> bool:
        return self._splitter.sizes()[0] == 0

    def _set_preview_visible(self, visible: bool) -> None:
        """Collapse / reopen the preview; the height moves to / comes from the belt map only."""
        raw, canvas, bottom = self._splitter.sizes()
        if visible and raw == 0:
            height = self._preview_height or (raw + canvas) // 3
            height = min(height, max(0, canvas - self._canvas_label.minimumHeight()))
            self._splitter.setSizes([height, canvas - height, bottom])
        elif not visible and raw > 0:
            self._preview_height = raw
            self._splitter.setSizes([0, raw + canvas, bottom])
            self._raw_label.clear()

    def _sync_preview_button(self) -> None:
        """Keep the button in step when the preview is collapsed/opened by dragging the handle."""
        visible = not self._preview_collapsed()
        if self._preview_button.isChecked() != visible:
            self._preview_button.blockSignals(True)
            self._preview_button.setChecked(visible)
            self._preview_button.blockSignals(False)

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
        if not restored:                            # bottom at its natural height, images 1:2
            # the info label may shrink to nothing (Ignored policy), so add its text height
            bottom = self._bottom_pane.sizeHint().height() + self._info_label.sizeHint().height()
            bottom = min(bottom, total // 3)
            images = max(0, total - bottom)
            self._splitter.setSizes([images // 3, images - images // 3, bottom])
        self._sync_preview_button()

    def _save_layout(self) -> None:
        settings = self._settings()
        settings.setValue(self._LAYOUT_KEY, self._splitter.saveState())
        settings.setValue(self._SIDE_LAYOUT_KEY, self._side_splitter.saveState())

    def _plan_text(self) -> str:
        """Where every correction/calibration value came from (first file of the playlist)."""
        plan = self._pipeline.plans[0]
        src = self._pipeline.sources[0]
        parts = [f"{len(self._pipeline.sources)} file(s), first {src.name} {src.frame_shape} {src.dtype}, stage={plan.stage}"]
        parts += [f"{key}: {value}  [{origin}]" for key, (value, origin) in plan.provenance.items()]
        return "\n".join(parts)

    def _setup_fps_control(self) -> None:
        """Presets for the simulated line rate. The recording's exposure only selects the
        preprocessing files (dark etc.); the simulation speed is set here, independently."""
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
                    ('reflectance', 'Reflectance'))

    def _setup_step_switches(self) -> None:
        """One checkbox per step; a step the input cannot have (no reference, binned input) is
        disabled with the reason as tooltip. Changes reach the workers at the next line."""
        plans = self._pipeline.plans
        for name, box in self._step_boxes.items():
            infos = [plan.steps.get(name, {'available': False, 'on': False, 'why': ''}) for plan in plans]
            box.setChecked(infos[0]['on'])
            available = any(info['available'] for info in infos)
            box.setEnabled(available)
            tip = infos[0]['why']
            if len(plans) > 1 and len({info['why'] for info in infos}) > 1:
                tip += f"\n(first file; {sum(i['available'] for i in infos)}/{len(plans)} files have this step)"
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
            if available and name != 'binning':
                box.toggled.connect(lambda checked, n=name: self._on_step_toggled(n, checked))
        self._update_units_warning()

    def _on_step_toggled(self, name: str, checked: bool) -> None:
        self._pipeline.set_steps(**{name: checked})
        self.performance_panel.reset_requested.emit()     # stage averages restart for the new chain
        self._update_units_warning()

    def _update_units_warning(self) -> None:
        classifier = self._pipeline.classifier
        reflectance = self._pipeline.effective_steps()['reflectance']
        wrong = classifier.input_units != 'any' and (classifier.input_units == 'reflectance') != reflectance
        placeholder = reflectance and any(getattr(p, 'reflectance_placeholder', False) for p in self._pipeline.plans)
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

    _BACKEND_CHOICES = (('auto', 'Auto (GPU if usable, else CPU fused)'), ('cpu_fused', 'CPU fused (fast, checked vs hsi)'),
                        ('cpu', 'CPU (hsi reference)'),
                        ('cupy_cuda', 'CuPy CUDA'), ('opencv_cuda', 'OpenCV CUDA'))

    def _setup_backend_selector(self) -> None:
        caps = get_compute_capabilities()
        combo = self._backend_combo
        for key, text in self._BACKEND_CHOICES:
            capability = getattr(caps, key, None)
            if capability is not None and not capability.available:
                text += "  — not available"
            combo.addItem(text, key)
            if capability is not None:
                combo.setItemData(combo.count() - 1, capability.reason or capability.device_name, Qt.ToolTipRole)
                if not capability.available:
                    combo.model().item(combo.count() - 1).setEnabled(False)
        combo.setCurrentIndex(max(0, combo.findData(self.config.corrections.backend)))
        if self._pipeline.plans[0].stage == 'binned':
            combo.setEnabled(False)
            combo.setToolTip("Input is already corrected and binned: no correction backend runs")
        combo.currentIndexChanged.connect(lambda _: self._pipeline.set_backend(combo.currentData()))

    # ------------------------------------------------------------------------- refresh ---

    def _refresh_gui_slot(self) -> None:
        columns, step = self._canvas_columns(), self._belt_step()
        s = self._pipeline.snapshot(canvas_columns=columns, raw_columns=self._raw_zone_columns(columns), step=step)
        # Images at gui_hz; the class statistics at ~10 Hz and the performance charts at ~5 Hz
        # (their stage numbers change every 0.5 s): less GUI time holding the GIL.
        self._tick = getattr(self, '_tick', 0) + 1
        hz = max(1.0, self.config.display.gui_hz)
        legend_due = self._tick % max(1, round(hz / 10)) == 0
        if legend_due:
            self.throughput_strip.update_snapshot(s)
        charts_due = self._tick % max(1, round(hz / 5)) == 0

        if s.raw_frame is not None and not self._preview_collapsed():
            self._set_pixmap(self._raw_label, _to_gray8(s.raw_frame, self._raw_label.size()), Qt.KeepAspectRatio)
        if s.canvas.shape[1] > 0:
            self._show_belt(s.canvas, s.raw_canvas, step[1])

        # Counts accumulate per line in the pipeline; "real-time" is every line since the
        # previous refresh. A reset makes the totals smaller than the previous values.
        if legend_due:
            if s.counted_lines < self._prev_counted_lines:
                self._prev_counts[:] = 0
            recent = s.counts - self._prev_counts
            self._prev_counts, self._prev_counted_lines = s.counts, s.counted_lines
            self.class_legend_widget.update_counts(recent, s.counts, s.counted_lines)
        if charts_due and self._side_tabs.currentWidget() is self.performance_panel and self._side_tabs.width() > 0:
            self.performance_panel.update_snapshot(s)

        name, index, n_frames, cycle = s.position
        total = s.frames_delivered + s.lines_lost
        lost_pct = 100.0 * s.lines_lost / total if total else 0.0
        target = 'max' if s.target_fps == 0 else f"{s.target_fps:g}"
        mm = (f" · {self.config.camera.belt_speed_m_s * 1000.0 / s.target_fps:.1f} mm/line"
              if s.target_fps else "")
        stage_total = s.stage_ms.get('total', 0.0)
        state = s.state.upper() if s.state != 'running' else 'RUN'
        text = (f"{state} | {name} [{index + 1}/{n_frames}] cycle {cycle} | "
                f"line rate {target}/s{mm} | camera {s.camera_fps:,.0f}/s · processed {s.processed_fps:,.0f}/s\n"
                f"lines {s.lines_processed:,} · lost {s.lines_lost:,} ({lost_pct:.2f}%) · "
                f"buffers {s.buffers_in_use}/{s.buffer_frames} · latency p50 {s.latency_ms['p50']:.2f} "
                f"p99 {s.latency_ms['p99']:.2f} max {s.latency_ms['max']:.1f} ms · "
                f"{s.backend} {stage_total:.2f} ms/line/worker")
        if s.error:
            text += f"\nERROR: {s.error}"
        self._status_label.setText(text)
        note = f" — {s.backend_note}" if s.backend_note else ""
        self._backend_label.setText(f"running: {s.backend}  (requested {s.requested_backend}){note}")
        set_accent(self._backend_label, 'warn' if s.backend_note else 'ok')
        # ok / warn / error: stripe and text colours come from the theme
        set_accent(self._status_label, 'error' if s.error else 'warn' if s.lines_lost else 'ok')

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
        grid = np.arange(needed - 1, -1, -column_step)[::-1]   # drawn columns (offsets from the oldest)
        if canvas.shape[1] < len(grid):         # not enough history yet: pad the left side
            pad = np.zeros((canvas.shape[0], len(grid) - canvas.shape[1]), canvas.dtype)
            canvas = np.concatenate([pad, canvas], axis=1)
        image = np.take(self._lut, canvas, axis=0)
        zone = self._raw_zone_columns(needed)
        if raw is None or zone == 0:
            return image
        first = int(np.searchsorted(grid, needed - zone))    # first drawn column of the raw zone
        image[:, first:image.shape[1] - raw.shape[1]] = 0     # raw zone before any line reached it: black
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
                self._raw_range = tuple((1 - a) * old + a * new for old, new in zip(self._raw_range, (lo, hi)))
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
        self._set_pixmap(self._canvas_label, self._belt_image(canvas, raw, column_step), Qt.IgnoreAspectRatio)
        needed = self._canvas_columns()
        zone = self._raw_zone_columns(needed)
        pixmap = self._canvas_label.pixmap()
        if zone == 0 or pixmap is None or pixmap.isNull():
            return
        tokens = QApplication.instance().property('theme_tokens')
        pixmap = pixmap.copy()
        w, h = pixmap.width(), pixmap.height()
        x = int(round(w * (needed - zone) / needed))
        painter = QPainter(pixmap)
        painter.setPen(QPen(QColor(tokens['accent']), 2, Qt.DashLine))
        painter.drawLine(x, 0, x, h)
        font = QFont(tokens['font_family'])
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        for text, left in (("◀ CLASSIFIED  (inference zone)", 6), ("RAW  (before classification)", x + 6)):
            box = metrics.boundingRect(text).adjusted(-5, -2, 5, 2)
            box.moveTo(left, 6)
            painter.fillRect(box, QColor(tokens['image_overlay']))   # #AARRGGBB
            painter.setPen(QColor(tokens['image_overlay_text']))
            painter.drawText(box, Qt.AlignCenter, text)
        painter.end()
        self._canvas_label.setPixmap(pixmap)

    @staticmethod
    def _set_pixmap(label: QLabel, image: np.ndarray, aspect) -> None:
        image = np.ascontiguousarray(image)
        h, w = image.shape[:2]
        if image.ndim == 2:
            qimg = QImage(image.tobytes(), w, h, w, QImage.Format_Grayscale8)
        else:
            qimg = QImage(image.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(label.size(), aspect, Qt.FastTransformation))

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
            # Keep the window (and the threads' owners) alive while a worker is still busy,
            # e.g. a CUDA backend finishing its initialisation; close again to retry.
            logger.warning("Shutdown pending; a pipeline thread is still running")
            self._status_label.setText("Shutdown pending: waiting for a pipeline thread — close again")
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=None, help='simulation.json (default: beside gui.py)')
    parser.add_argument('--source', default=None, help='HDF5/ENVI file or folder (overrides source.path)')
    parser.add_argument('--pattern', default=None)
    parser.add_argument('--fps', type=float, default=None, help='line rate; 0 = as fast as possible')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--theme', default=None, help='theme name in themes/ or a theme JSON (overrides display.theme)')
    args = parser.parse_args()

    config = load_config(args.config, camera={'fps': args.fps}, processing={'workers': args.workers},
                         display={'theme': args.theme})
    if args.source:
        config = replace(config, source=replace(config.source, path=str(Path(args.source).resolve()),
                                                pattern=args.pattern))
    app = QApplication(sys.argv[:1])
    tokens = apply_theme(app, config.display.theme)
    try:
        window = PushbroomScannerMainWindow(config)
    except Exception:
        sys.exit(1)
    use_dark_title_bar(window, bool(tokens.get('dark')))
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
