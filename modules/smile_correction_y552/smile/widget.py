# smile/widget.py
"""PyQt5 widget for interactive smile correction."""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

# --- Qt imports (guarded) ---
try:
    from PyQt5.QtCore import QThread, Qt, pyqtSignal
    from PyQt5.QtWidgets import (
        QCheckBox,
        QFileDialog,
        QFrame,
        QLabel,
        QPushButton,
        QSizePolicy,
        QSpinBox,
        QTabWidget,
        QHBoxLayout,
        QVBoxLayout,
        QWidget,
    )
    import matplotlib

    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    HAVE_QT = True
except ImportError:
    HAVE_QT = False

    # Stubs so class definitions below do not raise NameError.
    QWidget = QThread = FigureCanvasQTAgg = object

    def pyqtSignal(*args, **kwargs):  # noqa: D103
        return None

    class Figure:  # noqa: D101
        pass

# --- Project imports (no Qt dependency) ---
from .core import (
    _INTERP,
    auto_sat_level,
    load_frame,
    robust_sigma,
    save_frame,
)
from .analysis import SmileAnalysis, analyze_smile
from .calibration import (
    SmileCalibration,
    build_maps,
    calibration_from_analysis,
    crop_margin,
    load_calibration,
    save_calibration,
)
from .corrector import SmileCorrector

__all__ = ["PlotCanvas", "SmileCorrectionWidget", "HAVE_QT"]

_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


class PlotCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas wrapper for Qt layouts.

    Args:
        parent: Parent widget.
        figsize: Figure size ``(width, height)`` in inches.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        figsize: Tuple[float, float] = (8, 5),
    ):
        self.fig = Figure(figsize=figsize, dpi=100)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def clear(self) -> None:
        """Remove all axes and artists from the figure."""
        self.fig.clear()

    def refresh(self) -> None:
        """Apply tight layout and request a canvas redraw."""
        try:
            self.fig.tight_layout()
        except Exception:
            pass
        self.draw_idle()


class _AnalyzeWorker(QThread):
    """Run :func:`analyze_smile` in a background thread.

    Signals:
        done: Emitted with the :class:`SmileAnalysis` on success.
        failed: Emitted with the error message string on failure.
    """

    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, frame: np.ndarray, kwargs: dict):
        super().__init__()
        self._frame = frame
        self._kwargs = kwargs

    def run(self) -> None:
        """Execute the analysis."""
        try:
            self.done.emit(analyze_smile(self._frame, **self._kwargs))
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            self.failed.emit(str(exc))


class SmileCorrectionWidget(QWidget):
    """Interactive smile correction widget for PyQt5 applications.

    Provides file loading, analysis, visualisation (4 tabs), and
    calibration save / load.  Can be embedded as a sub-widget or used
    standalone.

    Signals:
        analysisComplete: Emitted with :class:`SmileAnalysis` after
            successful analysis.
        statusChanged: Emitted with a status message string.

    Example::

        widget = SmileCorrectionWidget(parent)
        layout.addWidget(widget)
        widget.analysisComplete.connect(my_handler)
        widget.set_frame(raw_frame)
        widget.analyze()          # async -- wait for signal
    """

    analysisComplete = pyqtSignal(object)
    statusChanged = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        if not HAVE_QT:
            raise ImportError(
                "PyQt5 and matplotlib are required for the widget."
            )
        super().__init__(parent)
        self._frame: Optional[np.ndarray] = None
        self._result: Optional[SmileAnalysis] = None
        self._calibration: Optional[SmileCalibration] = None
        self._corrector: Optional[SmileCorrector] = None
        self._worker: Optional[_AnalyzeWorker] = None
        self._build_ui()

    # -----------------------------------------------------------------
    #  UI construction
    # -----------------------------------------------------------------

    def _build_ui(self) -> None:
        """Construct the widget layout and controls."""
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # -- control row --
        ctrl = QHBoxLayout()

        self.btn_load = QPushButton("📂 โหลดภาพ")
        self.btn_load.clicked.connect(self._on_load)
        ctrl.addWidget(self.btn_load)

        self.btn_analyze = QPushButton("🔍 วิเคราะห์ Smile")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.setStyleSheet("font-weight:bold;")
        self.btn_analyze.clicked.connect(self._on_analyze)
        ctrl.addWidget(self.btn_analyze)

        self.btn_save = QPushButton("💾 บันทึก")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._on_save)
        ctrl.addWidget(self.btn_save)

        self.btn_save_cal = QPushButton("📐 บันทึก Matrix")
        self.btn_save_cal.setEnabled(False)
        self.btn_save_cal.setToolTip(
            "Save map_x for reuse without re-analysis."
        )
        self.btn_save_cal.clicked.connect(self._on_save_cal)
        ctrl.addWidget(self.btn_save_cal)

        self.btn_load_cal = QPushButton("📥 โหลด Matrix")
        self.btn_load_cal.setToolTip(
            "Load a previously saved calibration matrix."
        )
        self.btn_load_cal.clicked.connect(self._on_load_cal)
        ctrl.addWidget(self.btn_load_cal)

        ctrl.addStretch()

        # -- fit parameters --
        ctrl.addWidget(QLabel("deg_y:"))
        self.spin_dy = QSpinBox()
        self.spin_dy.setRange(1, 5)
        self.spin_dy.setValue(2)
        ctrl.addWidget(self.spin_dy)

        ctrl.addWidget(QLabel("deg_x:"))
        self.spin_dx = QSpinBox()
        self.spin_dx.setRange(0, 3)
        self.spin_dx.setValue(1)
        ctrl.addWidget(self.spin_dx)

        ctrl.addWidget(QLabel("sat:"))
        self.spin_sat = QSpinBox()
        self.spin_sat.setRange(0, 65535)
        self.spin_sat.setValue(0)
        self.spin_sat.setSpecialValueText("auto")
        self.spin_sat.setToolTip(
            "Sensor full-scale (e.g. 4095 for 12-bit). "
            "auto = detect from image."
        )
        ctrl.addWidget(self.spin_sat)

        self.chk_crop = QCheckBox("ตัดขอบ (หายจริง)")
        self.chk_crop.setChecked(True)
        ctrl.addWidget(self.chk_crop)

        root.addLayout(ctrl)

        # -- tabbed plot area --
        self.tabs = QTabWidget()
        self._canvases: List[PlotCanvas] = []
        tab_names = [
            "ภาพ  Before / After",
            "เส้น  Traces",
            "Residual",
            "Shift Map",
        ]
        for name in tab_names:
            canvas = PlotCanvas(self.tabs)
            self._canvases.append(canvas)
            self.tabs.addTab(canvas, name)

        (
            self.canvas_img,
            self.canvas_trace,
            self.canvas_resid,
            self.canvas_surf,
        ) = self._canvases
        root.addWidget(self.tabs, stretch=1)

        # -- status bar --
        self.lbl_status = QLabel("พร้อมใช้งาน")
        self.lbl_status.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.lbl_status.setMinimumHeight(24)
        root.addWidget(self.lbl_status)

    # -----------------------------------------------------------------
    #  Public API
    # -----------------------------------------------------------------

    def set_frame(self, frame: Optional[np.ndarray]) -> None:
        """Set the calibration image from a NumPy array.

        Clears any previous analysis result to prevent saving stale data.

        Args:
            frame: Grayscale image ``(H, W)``, or ``None`` to clear.
        """
        self._result = None
        self._corrector = None
        self.btn_save.setEnabled(False)
        self.btn_save_cal.setEnabled(False)

        self._frame = (
            frame.astype(np.float32) if frame is not None else None
        )
        self.btn_analyze.setEnabled(
            self._frame is not None and self._worker is None
        )
        self._show_preview(self._frame)
        if self._frame is not None:
            h, w = self._frame.shape
            self._set_status(
                f"Size {w}x{h}  max={self._frame.max():.0f}  "
                f"sat_level(auto)={auto_sat_level(self._frame):.0f}"
            )

    def load_file(self, path: str) -> None:
        """Load a calibration image from disk.

        Args:
            path: Image file path.
        """
        self.set_frame(load_frame(path))
        h, w = self._frame.shape
        self._set_status(
            f"{os.path.basename(path)}  {w}x{h}  "
            f"max={self._frame.max():.0f}  "
            f"sat_level(auto)={auto_sat_level(self._frame):.0f}"
        )

    def analyze(self) -> None:
        """Trigger async analysis on the current frame.

        Connect to ``analysisComplete`` to receive the result.
        """
        self._on_analyze()

    def get_analysis(self) -> Optional[SmileAnalysis]:
        """Return the most recent analysis result, or ``None``."""
        return self._result

    def get_corrected(self) -> Optional[np.ndarray]:
        """Return the corrected frame, or ``None`` if not yet analysed."""
        return self._result.frame_corr if self._result else None

    def get_corrector(self) -> Optional[SmileCorrector]:
        """Return the fitted :class:`SmileCorrector`, or ``None``."""
        return self._corrector

    def set_params(
        self,
        deg_y: Optional[int] = None,
        deg_x: Optional[int] = None,
        crop: Optional[bool] = None,
        sat_level: Optional[int] = None,
    ) -> None:
        """Update fit parameters programmatically.

        Args:
            deg_y: Polynomial degree in the spatial direction.
            deg_x: Polynomial degree in the spectral direction.
            crop: Whether to crop invalid borders on save.
            sat_level: Sensor full-scale value (0 = auto-detect).
        """
        if deg_y is not None:
            self.spin_dy.setValue(deg_y)
        if deg_x is not None:
            self.spin_dx.setValue(deg_x)
        if crop is not None:
            self.chk_crop.setChecked(crop)
        if sat_level is not None:
            self.spin_sat.setValue(int(sat_level))

    def apply_calibration(
        self, frame: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply a previously loaded calibration matrix to a frame.

        Args:
            frame: Target image.  Falls back to ``self._frame`` if ``None``.

        Returns:
            Corrected and cropped image.

        Raises:
            ValueError: If no calibration is loaded or no frame available.
        """
        if self._calibration is None:
            raise ValueError("No calibration matrix loaded.")
        src = frame if frame is not None else self._frame
        if src is None:
            raise ValueError("No frame available.")
        corrected = self._calibration.apply_cropped(src)
        self._set_status(
            f"Applied: {self._calibration.image_shape[1]}x"
            f"{self._calibration.image_shape[0]} -> "
            f"{corrected.shape[1]}x{corrected.shape[0]}"
        )
        return corrected

    # -----------------------------------------------------------------
    #  Internal helpers
    # -----------------------------------------------------------------

    def _set_status(self, msg: str) -> None:
        """Update the status label and emit ``statusChanged``."""
        self.lbl_status.setText(msg)
        self.statusChanged.emit(msg)

    def _show_preview(self, frame: Optional[np.ndarray]) -> None:
        """Draw the initial preview in the image tab."""
        self.canvas_img.clear()
        ax = self.canvas_img.fig.add_subplot(111)
        if frame is not None:
            vmin, vmax = np.percentile(frame, [1, 99.5])
            ax.imshow(
                frame, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax
            )
            ax.set_title("Original -- waiting for analysis")
        self.canvas_img.refresh()

    def _on_load(self) -> None:
        """Handle the 'Load Image' button click."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "เลือกภาพ calibration",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All (*.*)",
        )
        if path:
            try:
                self.load_file(path)
            except Exception as exc:
                self._set_status(f"Error: {exc}")

    def _on_analyze(self) -> None:
        """Handle the 'Analyze' button click (async via QThread)."""
        if self._frame is None or self._worker is not None:
            return
        kwargs = dict(
            sat_level=(self.spin_sat.value() or None),
            deg_y=self.spin_dy.value(),
            deg_x=self.spin_dx.value(),
            verbose=True,
        )
        self._set_status("Analyzing ...")
        self.btn_analyze.setEnabled(False)
        worker = _AnalyzeWorker(self._frame.copy(), kwargs)
        worker.done.connect(self._on_analysis_done)
        worker.failed.connect(self._on_analysis_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        worker.start()

    def _on_worker_finished(self) -> None:
        """Clean up after the analysis thread completes."""
        if self._worker is not None:
            self._worker.wait()
        self._worker = None
        self.btn_analyze.setEnabled(self._frame is not None)

    def _on_analysis_failed(self, msg: str) -> None:
        """Handle analysis failure (safe: queued to main thread)."""
        self._set_status(f"Error: {msg}")

    def _on_analysis_done(self, r: SmileAnalysis) -> None:
        """Handle successful analysis result."""
        self._result = r

        # Build a SmileCorrector from the analysis for easy reuse.
        self._corrector = SmileCorrector(
            deg_y=self.spin_dy.value(),
            deg_x=self.spin_dx.value(),
            interp=r.interp,
        )
        self._corrector._surface = r.surface
        self._corrector._map_x, _, _ = build_maps(
            r.frame_orig.shape, r.surface
        )
        self._corrector._margin = r.margin
        self._corrector._image_shape = tuple(r.frame_orig.shape)
        self._corrector._analysis = r

        cv_str = (
            f"{r.cv_rms:.3f}" if np.isfinite(r.cv_rms) else "n/a"
        )
        self._set_status(
            f"Lines {len(r.used_lines)}/{len(r.all_lines)}  |  "
            f"RMS {r.rms_before:.3f}->{r.rms_after:.3f} px "
            f"(CV {cv_str})  |  "
            f"margin L={r.margin[0]} R={r.margin[1]} px "
            f"({r.margin[0] + r.margin[1]} px lost)"
        )
        self.btn_save.setEnabled(True)
        self.btn_save_cal.setEnabled(True)
        self._update_all_plots(r)
        self.analysisComplete.emit(r)

    def _on_save(self) -> None:
        """Handle the 'Save Image' button click."""
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "บันทึกภาพที่แก้ smile แล้ว",
            "Corrected.png",
            "PNG (*.png);;TIFF (*.tif);;All (*.*)",
        )
        if not path:
            return
        img = self._result.frame_corr
        if self.chk_crop.isChecked():
            img = crop_margin(img, self._result.margin)
        bits = 8 if self._result.sat_level <= 255 else 16
        save_frame(path, img, bits=bits)
        self._set_status(
            f"Saved {path} ({img.shape[1]}x{img.shape[0]} px, {bits}-bit)"
        )

    def _on_save_cal(self) -> None:
        """Handle the 'Save Matrix' button click."""
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "บันทึก Smile Calibration Matrix",
            "smile_calibration.npz",
            "Smile Calibration (*.npz);;All (*.*)",
        )
        if not path:
            return
        if not path.endswith(".npz"):
            path += ".npz"
        if self._corrector is not None:
            self._corrector.save(path)
        else:
            cal = calibration_from_analysis(self._result)
            save_calibration(path, cal)
        self._set_status(f"Saved calibration: {path}")

    def _on_load_cal(self) -> None:
        """Handle the 'Load Matrix' button click."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "โหลด Smile Calibration Matrix",
            "",
            "Smile Calibration (*.npz);;All (*.*)",
        )
        if not path:
            return
        try:
            self._calibration = load_calibration(path)
            self._corrector = SmileCorrector.load(path)
            self._set_status(
                self._calibration.summary().replace("\n", " | ")
            )
        except Exception as exc:
            self._set_status(f"Load failed: {exc}")

    # -----------------------------------------------------------------
    #  Plotting
    # -----------------------------------------------------------------

    def _update_all_plots(self, r: SmileAnalysis) -> None:
        """Redraw all four analysis tabs."""
        self._plot_image(r)
        self._plot_traces(r)
        self._plot_residual(r)
        self._plot_surface(r)

    def _plot_image(self, r: SmileAnalysis) -> None:
        """Render the Before / After image comparison tab."""
        self.canvas_img.clear()
        fig = self.canvas_img.fig

        vmin, vmax = np.percentile(r.frame_orig, [1, 99.5])
        n_cols = r.frame_orig.shape[1]

        ax1 = fig.add_subplot(211)
        ax1.imshow(
            r.frame_orig, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax
        )
        ax1.set_title(f"Before  ({n_cols} cols)", fontsize=10)
        ax1.set_ylabel("Row")

        ax2 = fig.add_subplot(212)
        img = r.frame_corr
        left, right = r.margin
        if self.chk_crop.isChecked() and (left + right > 0):
            img = crop_margin(img, (left, right))
            title = (
                f"After  ({img.shape[1]} cols -- cropped "
                f"L={left} R={right}, {left + right} px lost)"
            )
        else:
            title = (
                f"After  ({img.shape[1]} cols -- not cropped, "
                f"border zeros = no data)"
            )
        ax2.imshow(img, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
        ax2.set_title(title, fontsize=10)
        ax2.set_xlabel("Column (spectral)")
        ax2.set_ylabel("Row")
        self.canvas_img.refresh()

    def _plot_traces(self, r: SmileAnalysis) -> None:
        """Render the Traces tab (before / after comparison)."""
        self.canvas_trace.clear()
        fig = self.canvas_trace.fig
        rows = np.arange(r.frame_orig.shape[0])

        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)

        for i, ln in enumerate(r.used_lines):
            color = _COLORS[i % len(_COLORS)]
            label = f"col {ln.column:.0f}"
            ref = r.refs[ln.column]

            # Before: offset from per-line reference.
            c = r.traces_orig.get(ln.column)
            if c is not None:
                v = np.isfinite(c)
                ax1.plot(
                    c[v] - ref, rows[v],
                    ".", ms=0.5, color=color, label=label,
                )

            # After: use the corrected trace's own reference.
            c2 = r.traces_corr.get(ln.column)
            if c2 is not None:
                v2 = np.isfinite(c2)
                if v2.any():
                    ref2 = float(np.median(c2[v2]))
                    ax2.plot(
                        c2[v2] - ref2, rows[v2],
                        ".", ms=0.5, color=color, label=label,
                    )

        ax1.invert_yaxis()
        ax2.invert_yaxis()
        ax1.set_ylabel("Row")
        ax1.set_title(
            f"Before -- RMS {r.rms_before:.4f} px", fontsize=10
        )
        if len(r.used_lines) <= 12:
            ax1.legend(fontsize=5, loc="upper right", ncol=2)

        ax2.set_xlabel("Shift from per-line reference (px)")
        ax2.set_ylabel("Row")
        ax2.set_title(
            f"After -- RMS {r.rms_after:.4f} px  (target: near 0)",
            fontsize=10,
        )
        ax2.axvline(0, color="black", lw=0.5)
        self.canvas_trace.refresh()

    def _plot_residual(self, r: SmileAnalysis) -> None:
        """Render the Residual tab (scatter + histogram)."""
        self.canvas_resid.clear()
        fig = self.canvas_resid.fig

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        sig = robust_sigma(r.residuals)

        for i, ln in enumerate(r.used_lines):
            color = _COLORS[i % len(_COLORS)]
            mask = r.meas_id == i
            if np.any(mask):
                ax1.scatter(
                    r.meas_y[mask], r.residuals[mask],
                    s=0.5, alpha=0.3, color=color,
                )

        ax1.axhline(0, color="black", lw=0.5)
        ax1.axhline(
            +2 * sig,
            color="red", lw=0.5, ls="--",
            label=f"+/-2 sigma = {2 * sig:.4f}",
        )
        ax1.axhline(-2 * sig, color="red", lw=0.5, ls="--")
        ax1.set_xlabel("Row (spatial)")
        ax1.set_ylabel("Residual (px)")
        cv_str = (
            f"{r.cv_rms:.5f}" if np.isfinite(r.cv_rms) else "n/a"
        )
        ax1.set_title(
            f"Residual vs Row  (RMS={r.fit_info['rms']:.5f}, "
            f"CV={cv_str}, n_used={r.fit_info['n_used']})",
            fontsize=9,
        )
        ax1.legend(fontsize=7)

        ax2.hist(
            r.residuals, bins=50,
            edgecolor="black", lw=0.4,
            color="steelblue", alpha=0.8,
        )
        ax2.axvline(0, color="black", lw=0.5)
        ax2.set_xlabel("Residual (px)")
        ax2.set_ylabel("Count")
        ax2.set_title("Distribution", fontsize=9)
        self.canvas_resid.refresh()

    def _plot_surface(self, r: SmileAnalysis) -> None:
        """Render the Shift Map tab (2-D displacement surface)."""
        self.canvas_surf.clear()
        ax = self.canvas_surf.fig.add_subplot(111)

        n_rows, n_cols = r.frame_orig.shape
        yy = np.linspace(0, n_rows - 1, 100)
        xx = np.linspace(0, n_cols - 1, 100)
        XX, YY = np.meshgrid(xx, yy)
        S = r.surface(YY, XX)

        im = ax.pcolormesh(xx, yy, S, cmap="RdBu_r", shading="auto")
        self.canvas_surf.fig.colorbar(im, ax=ax, label="Shift (px)")

        ax.scatter(r.meas_x, r.meas_y, s=0.3, c="black", alpha=0.15)
        ax.set_xlabel("Column (spectral)")
        ax.set_ylabel("Row (spatial)")
        ax.set_title(
            f"Smile Shift Surface  "
            f"[deg_y={r.surface.deg_y}, deg_x={r.surface.deg_x}]  "
            f"(clamped outside line range)",
            fontsize=10,
        )
        self.canvas_surf.refresh()