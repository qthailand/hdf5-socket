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
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QSizePolicy,
                             QVBoxLayout, QWidget)

from modules.compute_runtime import get_compute_capabilities, initialize_compute
from pipeline import build_pipeline
from sim_config import load_config
from themes import apply_theme, make_panel, set_accent, use_dark_title_bar
from utils import generate_tab10_lut
from widgets.class_legend_widget import ClassLegendWidget

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
            category_names=classifier.category_names, lut=self._lut, classes=classifier.classes_)
        self.class_legend_widget.color_changed.connect(self._on_class_color_changed)
        self.class_legend_widget.reset_requested.connect(self._pipeline.reset_counts)
        self._setup_backend_selector()
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
        self._canvas_label.setToolTip("Every N-th line (display.canvas_line_stride); magenta = lines lost to overrun")

        self._status_label = QLabel("Starting…")
        self._status_label.setObjectName("statusLabel")
        make_panel(self._status_label, accent='ok')
        self._status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._info_label = QLabel("")
        self._info_label.setObjectName("infoLabel")
        make_panel(self._info_label, accent='info')
        self._info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        left = QVBoxLayout()
        left.setSpacing(8)
        left.addWidget(self._raw_label, stretch=1)
        left.addWidget(self._canvas_label, stretch=2)
        backend_row = QHBoxLayout()
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
        left.addLayout(backend_row)
        left.addWidget(self._status_label)
        left.addWidget(self._info_label)

        self.class_legend_widget = ClassLegendWidget()
        side = QVBoxLayout()
        side.addWidget(self.class_legend_widget)

        main = QHBoxLayout()
        main.setContentsMargins(10, 10, 10, 10)
        main.addLayout(left, stretch=3)
        main.addLayout(side, stretch=1)
        container = QWidget()
        container.setLayout(main)
        self.setCentralWidget(container)

    def _plan_text(self) -> str:
        """Where every correction/calibration value came from (first file of the playlist)."""
        plan = self._pipeline.plans[0]
        src = self._pipeline.sources[0]
        parts = [f"{len(self._pipeline.sources)} file(s), first {src.name} {src.frame_shape} {src.dtype}, stage={plan.stage}"]
        parts += [f"{key}: {value}  [{origin}]" for key, (value, origin) in plan.provenance.items()]
        return "\n".join(parts)

    _BACKEND_CHOICES = (('auto', 'Auto (CuPy → OpenCV CUDA → CPU)'), ('cpu', 'CPU'),
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
        s = self._pipeline.snapshot()

        if s.raw_frame is not None:
            self._set_pixmap(self._raw_label, _to_gray8(s.raw_frame), Qt.KeepAspectRatio)
        if s.canvas.shape[1] > 0:
            self._set_pixmap(self._canvas_label, self._lut[s.canvas], Qt.IgnoreAspectRatio)

        # Counts accumulate per line in the pipeline; "real-time" is every line since the
        # previous refresh. A reset makes the totals smaller than the previous values.
        if s.counted_lines < self._prev_counted_lines:
            self._prev_counts[:] = 0
        recent = s.counts - self._prev_counts
        self._prev_counts, self._prev_counted_lines = s.counts, s.counted_lines
        self.class_legend_widget.update_counts(recent, s.counts, s.counted_lines)

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
        if not self._pipeline.stop():
            # Keep the window (and the threads' owners) alive while a worker is still busy,
            # e.g. a CUDA backend finishing its initialisation; close again to retry.
            logger.warning("Shutdown pending; a pipeline thread is still running")
            self._status_label.setText("Shutdown pending: waiting for a pipeline thread — close again")
            event.ignore()
            return
        logger.info("All pipeline threads stopped. Closing application.")
        event.accept()


def _to_gray8(frame: np.ndarray) -> np.ndarray:
    """Percentile stretch (1-99 % of a subsample) to 8 bit, for display only."""
    sample = frame[::4, ::4]
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
