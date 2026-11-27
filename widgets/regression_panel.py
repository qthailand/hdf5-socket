# -*- coding: utf-8 -*-
"""Regression (e.g. moisture) display: colour bar, summary values, histogram and objects.

Shown instead of the class charts when the running model is a regression package
(``inference_engine.HsiRegressor``): :class:`ColorBar` sits above the belt map,
:class:`RegressionPanel` takes the side tab. Values come from ``Snapshot.value_stats`` /
``counts`` (the code histogram) / ``value_canvas`` through ``regression.summary`` and
``regression.objects``; the colour map is ``regression.colormap``. Pure display: the panel only
emits what the operator asks for (colour range, object mask, batch reset).
"""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QRectF, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt5.QtWidgets import (QAbstractItemView, QCheckBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout,
                             QHeaderView, QLabel, QPushButton, QScrollArea, QSizePolicy, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from regression import LEVELS, code_values, colormap
from themes import make_panel
from widgets.line_plot import _theme_colour


def _gradient(width: int, height: int = 1) -> QImage:
    rgb = np.ascontiguousarray(np.repeat(colormap(np.linspace(0.0, 1.0, max(2, width)))[None], height, axis=0))
    image = QImage(rgb.tobytes(), rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QImage.Format_RGB888)
    return image.copy()                     # owns its bytes


class ColorBar(QWidget):
    """Horizontal colour scale of the belt map: the display range, its units, ticks."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.label = ''
        self.display_range = (0.0, 1.0)
        self.valid_range = (0.0, 1.0)
        self.setFixedHeight(40)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setToolTip("Colour scale of the belt map (Moisture tab: colour range). Black = no object, "
                        "magenta = lost line")

    def set_scale(self, label: str, display_range: tuple, valid_range: tuple) -> None:
        self.label, self.display_range, self.valid_range = label, tuple(display_range), tuple(valid_range)
        self.update()

    def ticks(self) -> list:
        lo, hi = self.display_range
        span = hi - lo
        step = 10 ** np.floor(np.log10(max(span, 1e-12) / 5))
        for factor in (1, 2, 5, 10):
            if span / (step * factor) <= 6:
                step *= factor
                break
        first = np.ceil(lo / step) * step
        return [float(v) for v in np.arange(first, hi + step * 1e-6, step)]

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        font = QFont(self.font())
        font.setPixelSize(10)
        p.setFont(font)
        metrics = p.fontMetrics()
        caption = self.label
        left = metrics.horizontalAdvance(caption) + 12
        bar = QRectF(left, 4, max(10, self.width() - left - 8), 14)
        p.setPen(_theme_colour('text_muted'))
        p.drawText(QRectF(0, 0, left - 6, bar.bottom() + 2), Qt.AlignRight | Qt.AlignVCenter, caption)
        p.drawImage(bar, _gradient(256))
        p.setPen(QPen(_theme_colour('border'), 1))
        p.drawRect(bar)
        lo, hi = self.display_range
        p.setPen(_theme_colour('text_muted'))
        for value in self.ticks():
            x = bar.left() + (value - lo) / max(hi - lo, 1e-12) * bar.width()
            p.drawLine(int(x), int(bar.bottom()), int(x), int(bar.bottom() + 3))
            text = f"{value:g}"
            w = metrics.horizontalAdvance(text)
            p.drawText(int(min(max(x - w / 2, bar.left()), bar.right() - w)), int(bar.bottom() + 15), text)
        p.end()


class ValueHistogram(QWidget):
    """Histogram of the predicted values (code bins merged to ~60 bars), coloured like the map."""

    BARS = 60

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._hist = np.zeros(0)
        self._value_range = (0.0, 1.0)
        self._display_range = (0.0, 1.0)
        self._marks: list = []                 # [(value, label)]
        self.setMinimumHeight(110)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def sizeHint(self) -> QSize:
        return QSize(280, 130)

    def set_data(self, counts, value_range, display_range, marks=()) -> None:
        hist = np.asarray(counts[:LEVELS], np.float64) if counts is not None else np.zeros(LEVELS)
        edges = np.linspace(0, LEVELS, self.BARS + 1).astype(int)
        self._hist = np.add.reduceat(hist, edges[:-1])
        self._value_range, self._display_range = tuple(value_range), tuple(display_range)
        self._marks = list(marks)
        self.update()

    def bars(self) -> np.ndarray:
        return self._hist

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        font = QFont(self.font())
        font.setPixelSize(10)
        p.setFont(font)
        area = QRectF(4, 4, self.width() - 8, self.height() - 22)
        lo, hi = self._value_range
        p.setPen(_theme_colour('text_muted'))
        p.drawText(QRectF(0, area.bottom() + 2, self.width(), 16), Qt.AlignLeft, f" {lo:g}")
        p.drawText(QRectF(0, area.bottom() + 2, self.width() - 4, 16), Qt.AlignRight, f"{hi:g}")
        total = self._hist.sum() if self._hist.size else 0.0
        if total <= 0:
            p.drawText(area, Qt.AlignCenter, 'no object pixels yet')
            p.end()
            return
        peak = self._hist.max()
        width = area.width() / self._hist.size
        centres = code_values(self._value_range)
        per_bar = np.array_split(centres, self._hist.size)
        dlo, dhi = self._display_range
        for k, count in enumerate(self._hist):
            if count <= 0:
                continue
            value = float(np.mean(per_bar[k]))
            r, g, b = colormap((value - dlo) / max(dhi - dlo, 1e-12))
            h = area.height() * count / peak
            p.fillRect(QRectF(area.left() + k * width, area.bottom() - h, max(1.0, width - 1), h),
                       QColor(int(r), int(g), int(b)))
        for value, label in self._marks:
            if not np.isfinite(value):
                continue
            x = area.left() + (value - lo) / max(hi - lo, 1e-12) * area.width()
            p.setPen(QPen(_theme_colour('text'), 1, Qt.DashLine))
            p.drawLine(int(x), int(area.top()), int(x), int(area.bottom()))
            p.drawText(int(x) + 3, int(area.top()) + 10, label)
        p.end()


class RegressionPanel(QWidget):
    """Side tab of a regression model: summary, colour range, object mask, histogram, objects."""

    display_range_changed = pyqtSignal(float, float)
    object_mask_changed = pyqtSignal(object)          # float, or None = every valid pixel
    reset_requested = pyqtSignal()
    auto_range_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.target = None
        self.value_range = (0.0, 1.0)
        self.model_display_range = (0.0, 1.0)
        self._summary = {}
        self._objects: list = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        scroll.setWidget(body)
        outer.addWidget(scroll)

        header = QHBoxLayout()
        self.title = QLabel('Prediction')
        self.title.setObjectName('panelTitle')
        header.addWidget(self.title)
        self.reset_btn = QPushButton('Reset Batch')
        self.reset_btn.setObjectName('dangerButton')
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        self.reset_btn.clicked.connect(self.reset_requested.emit)
        header.addWidget(self.reset_btn, alignment=Qt.AlignRight)
        layout.addLayout(header)

        card = QFrame()
        card.setObjectName('summaryCard')
        make_panel(card, accent='info', alt=True)
        grid = QGridLayout(card)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(2)
        self.mean_caption = QLabel('MEAN OF OBJECTS (since reset)')
        self.mean_caption.setObjectName('caption')
        self.lbl_mean = QLabel('—')
        self.lbl_mean.setObjectName('bigValue')
        grid.addWidget(self.mean_caption, 0, 0, 1, 2)
        grid.addWidget(self.lbl_mean, 1, 0, 1, 2)
        self._values = {}
        for r, (key, text) in enumerate((('median', 'Median'), ('sd', 'SD'), ('p10_p90', 'P10 – P90'),
                                         ('n', 'Object pixels'), ('object_share', 'Object share of belt'),
                                         ('out_of_range', 'Outside valid range'))):
            caption = QLabel(text)
            caption.setObjectName('fieldLabel')
            value = QLabel('—')
            value.setObjectName('fieldValue')
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            grid.addWidget(caption, 2 + r, 0)
            grid.addWidget(value, 2 + r, 1)
            self._values[key] = value
        layout.addWidget(card)

        caption = QLabel('COLOUR RANGE OF THE MAP')
        caption.setObjectName('workspaceTitle')
        layout.addWidget(caption)
        row = QHBoxLayout()
        self.lo_spin = QDoubleSpinBox()
        self.hi_spin = QDoubleSpinBox()
        for spin in (self.lo_spin, self.hi_spin):
            spin.setKeyboardTracking(False)
            spin.setDecimals(2)
            spin.setRange(-1e6, 1e6)
            spin.valueChanged.connect(self._on_range_edited)
            row.addWidget(spin)
        self.auto_btn = QPushButton('Auto')
        self.auto_btn.setObjectName('quietButton')
        self.auto_btn.setToolTip('Colour range from the objects scanned now (2nd - 98th percentile)')
        self.auto_btn.clicked.connect(self.auto_range_requested.emit)
        self.model_btn = QPushButton('Model')
        self.model_btn.setObjectName('quietButton')
        self.model_btn.setToolTip("The package's display range")
        self.model_btn.clicked.connect(lambda: self.set_display_range(*self.model_display_range, emit=True))
        row.addWidget(self.auto_btn)
        row.addWidget(self.model_btn)
        layout.addLayout(row)

        row = QHBoxLayout()
        self.mask_box = QCheckBox('Object mask: mean input ≥')
        self.mask_box.setToolTip('Pixels whose mean of the model input bands is below this value are belt '
                                 '(background), not objects: they get no value and are not counted.\n'
                                 'Off: every pixel with a valid spectrum is an object.')
        self.mask_spin = QDoubleSpinBox()
        self.mask_spin.setDecimals(4)
        self.mask_spin.setRange(-1e6, 1e6)
        self.mask_spin.setSingleStep(0.01)
        self.mask_spin.setKeyboardTracking(False)
        self.mask_box.toggled.connect(self._on_mask_edited)
        self.mask_spin.valueChanged.connect(self._on_mask_edited)
        row.addWidget(self.mask_box)
        row.addWidget(self.mask_spin)
        row.addStretch()
        layout.addLayout(row)

        caption = QLabel('DISTRIBUTION OF OBJECT PIXELS')
        caption.setObjectName('workspaceTitle')
        layout.addWidget(caption)
        self.histogram = ValueHistogram()
        layout.addWidget(self.histogram)

        caption = QLabel('OBJECTS ON THE MAP (classified zone, largest first)')
        caption.setObjectName('workspaceTitle')
        layout.addWidget(caption)
        self.objects_table = QTableWidget(0, 5)
        self.objects_table.setHorizontalHeaderLabels(['#', 'map px', 'mean', 'median', 'min – max'])
        self.objects_table.setToolTip('Connected object regions of the predicted zone as the map shows it '
                                      '(map pixels: decimated to the screen). Touching objects count as one.')
        self.objects_table.verticalHeader().setVisible(False)
        self.objects_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.objects_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.objects_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.objects_table.setMinimumHeight(140)
        layout.addWidget(self.objects_table)
        self._sections = QVBoxLayout()
        layout.addLayout(self._sections)
        layout.addStretch()

    # ----------------------------------------------------------------------------- setup ---

    def add_section(self, widget: QWidget) -> None:
        """Put ``widget`` at the end of the panel (e.g. the model selector while this tab is shown)."""
        self._sections.addWidget(widget)

    def setup(self, target, value_range, display_range, object_min_mean) -> None:
        """A new regression model: its target, ranges and object mask."""
        self.target = target
        self.value_range = tuple(value_range)
        self.model_display_range = tuple(target.display_range) if target is not None else tuple(display_range)
        name = target.name if target is not None else 'value'
        self.title.setText(f"{name[:1].upper()}{name[1:]} prediction")
        self.mean_caption.setText(f"MEAN {name.upper()} OF OBJECTS (since reset)")
        units = f" {target.units}" if target is not None and target.units else ''
        decimals = target.decimals if target is not None else 2
        span = self.value_range[1] - self.value_range[0]
        for spin in (self.lo_spin, self.hi_spin):
            spin.blockSignals(True)
            spin.setSuffix(units)
            spin.setDecimals(max(decimals, 1))
            spin.setRange(self.value_range[0] - span, self.value_range[1] + span)
            spin.setSingleStep(10 ** -max(decimals - 1, 0))
            spin.blockSignals(False)
        self.set_display_range(*display_range)
        self.mask_box.blockSignals(True)
        self.mask_spin.blockSignals(True)
        self.mask_box.setChecked(object_min_mean is not None)
        if object_min_mean is not None:
            self.mask_spin.setValue(float(object_min_mean))
        self.mask_spin.setEnabled(object_min_mean is not None)
        self.mask_box.blockSignals(False)
        self.mask_spin.blockSignals(False)
        self._objects = []
        self.update_values({}, None, [])

    def display_range(self) -> tuple:
        return float(self.lo_spin.value()), float(self.hi_spin.value())

    def set_display_range(self, lo: float, hi: float, emit: bool = False) -> None:
        for spin, value in ((self.lo_spin, lo), (self.hi_spin, hi)):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)
        if emit:
            self.display_range_changed.emit(*self.display_range())

    def _on_range_edited(self, _=None) -> None:
        lo, hi = self.display_range()
        if hi <= lo:                                  # keep a usable range: move the other end
            if self.sender() is self.lo_spin:
                self.hi_spin.blockSignals(True)
                self.hi_spin.setValue(lo + 10 ** -self.hi_spin.decimals())
                self.hi_spin.blockSignals(False)
            else:
                self.lo_spin.blockSignals(True)
                self.lo_spin.setValue(hi - 10 ** -self.lo_spin.decimals())
                self.lo_spin.blockSignals(False)
        self.display_range_changed.emit(*self.display_range())

    def object_mask(self):
        return float(self.mask_spin.value()) if self.mask_box.isChecked() else None

    def _on_mask_edited(self, _=None) -> None:
        self.mask_spin.setEnabled(self.mask_box.isChecked())
        self.object_mask_changed.emit(self.object_mask())

    # ---------------------------------------------------------------------------- values ---

    def _fmt(self, value) -> str:
        if self.target is not None:
            return self.target.format(value)
        return '—' if value is None or not np.isfinite(value) else f"{value:.3g}"

    def _number(self, value) -> str:
        decimals = self.target.decimals if self.target is not None else 2
        return '—' if value is None or not np.isfinite(value) else f"{value:.{decimals}f}"

    def update_values(self, summary: dict, counts, objects: list | None = None) -> None:
        """New statistics (``regression.summary``), the code histogram and (optionally) the objects."""
        self._summary = dict(summary)
        s = self._summary
        self.lbl_mean.setText(self._fmt(s.get('mean', np.nan)))
        self._values['median'].setText(self._fmt(s.get('median', np.nan)))
        sd = s.get('sd', np.nan)
        self._values['sd'].setText('—' if not np.isfinite(sd) else self._fmt(sd))
        p10, p90 = s.get('p10', np.nan), s.get('p90', np.nan)
        self._values['p10_p90'].setText('—' if not np.isfinite(p10) else f"{self._fmt(p10)} – {self._fmt(p90)}")
        self._values['n'].setText(f"{int(s.get('n', 0)):,}")
        self._values['object_share'].setText(f"{100.0 * s.get('object_share', 0.0):.1f} %")
        out = s.get('out_of_range', 0.0)
        lo, hi = self.value_range
        self._values['out_of_range'].setText(f"{100.0 * out:.1f} %  (valid {lo:g} – {hi:g})")
        self._values['out_of_range'].setProperty('state', 'warn' if out > 0.05 else '')
        marks = [(s.get('mean', np.nan), 'mean')] if s else []
        self.histogram.set_data(counts, self.value_range, self.display_range(), marks)
        if objects is not None:
            self.set_objects(objects)

    def set_objects(self, objects: list) -> None:
        self._objects = list(objects)
        table = self.objects_table
        table.setRowCount(len(objects))
        for r, obj in enumerate(objects):
            cells = [str(r + 1), f"{obj['pixels']:,}", self._fmt(obj['mean']), self._fmt(obj['median']),
                     f"{self._number(obj['min'])} – {self._number(obj['max'])}"]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter if c else Qt.AlignCenter)
                table.setItem(r, c, item)

    @property
    def summary(self) -> dict:
        return dict(self._summary)

    @property
    def objects(self) -> list:
        return list(self._objects)
