# -*- coding: utf-8 -*-
"""Spectrum preview: the spectra of the pixels the operator clicked on the belt map.

Each click adds one curve in its own colour (the same colour marks the pixel on the map);
``Clear`` removes them all. The values are what the model received for that pixel (after every
processing step that is on), so a saturated, noisy or mis-scaled input shows at a glance.

At most :attr:`SpectrumPanel.MAX_POINTS` curves are kept; one more click drops the oldest.
Curve colours are data colours (:data:`COLOURS`), everything else comes from the theme tokens.
GUI thread only.

    panel = SpectrumPanel()
    point = panel.add_point(wavelengths, values, line=1200, row=310, label='Kratom')
    panel.clear()
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from themes import make_panel

#: One colour per curve, told apart on a dark and on a light panel and on the grey belt map.
COLOURS = ('#F97316', '#22D3EE', '#E879F9', '#FACC15', '#4ADE80', '#F87171', '#A78BFA', '#94A3B8')

EMPTY_TEXT = "Click on the belt map: the spectrum of that pixel is drawn here"


@dataclass
class SpectrumPoint:
    """One clicked pixel."""
    number: int                 # 1, 2, ... in click order (shown on the map and in the key)
    colour: str                 # '#RRGGBB'
    line: int                   # belt line number (Snapshot.canvas_lines)
    row: int                    # slit pixel
    wavelengths: np.ndarray     # (bands,) nm
    values: np.ndarray          # (bands,) what the model received
    label: str = ''             # class the model gave this pixel


def nice_ticks(lo: float, hi: float, count: int = 4) -> list[float]:
    """About ``count`` round tick values inside ``lo .. hi``."""
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        return [lo]
    raw = span / max(1, count)
    power = 10.0 ** np.floor(np.log10(raw))
    step = next(m for m in (1.0, 2.0, 2.5, 5.0, 10.0) if m * power >= raw) * power
    first = np.ceil(lo / step - 1e-9) * step
    return [float(v) for v in np.arange(first, hi + step * 1e-6, step)]


def format_tick(value: float, step: float) -> str:
    decimals = 0 if step >= 1 else int(min(4, np.ceil(-np.log10(step))))
    return f"{value:,.{decimals}f}"


class SpectrumPanel(QFrame):
    """Card: header (caption, note, Clear) above the painted curves and their key.

    Signals:
        cleared(): the operator pressed ``Clear`` (the points are already gone).
    """

    cleared = pyqtSignal()

    MAX_POINTS = len(COLOURS)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("spectrumPanel")
        make_panel(self, accent='neutral', alt=True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setToolTip("Spectrum preview: click a pixel on the belt map (either zone) to draw the spectrum the\n"
                        "model received for it. Every click adds a curve in its own colour; Clear removes them.\n"
                        f"The newest {self.MAX_POINTS} points are kept. Only lines still in the spectrum history\n"
                        "(display.spectrum_history_mb) can be clicked.")
        self._points: list[SpectrumPoint] = []
        self._number = 0
        self._unit = ''

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 8, 6)
        layout.setSpacing(4)
        header = QHBoxLayout()
        header.setSpacing(6)
        caption = QLabel("SPECTRUM")
        caption.setObjectName("caption")
        header.addWidget(caption)
        self.note_label = QLabel("")
        self.note_label.setObjectName("metricCaption")
        self.note_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)   # never widens the panel
        header.addWidget(self.note_label, stretch=1)
        self.clear_button = QPushButton("Clear")
        self.clear_button.setObjectName("presetButton")
        self.clear_button.setToolTip("Remove every spectrum and its mark on the belt map")
        self.clear_button.setFocusPolicy(Qt.NoFocus)
        self.clear_button.setEnabled(False)
        self.clear_button.clicked.connect(self._on_clear)
        header.addWidget(self.clear_button)
        layout.addLayout(header)
        self.plot = _SpectrumPlot(self)
        layout.addWidget(self.plot, stretch=1)

    # ------------------------------------------------------------------------ public ---

    @property
    def points(self) -> list[SpectrumPoint]:
        return list(self._points)

    @property
    def unit(self) -> str:
        return self._unit

    def set_unit(self, unit: str) -> None:
        """What the values are (y axis caption), e.g. ``reflectance`` or ``counts``."""
        if unit != self._unit:
            self._unit = unit
            self.plot.update()

    def set_note(self, text: str) -> None:
        """One short line beside the caption (why a click gave no spectrum, or what was added)."""
        self.note_label.setText(text)

    def add_point(self, wavelengths, values, line: int, row: int, label: str = '') -> SpectrumPoint:
        """Add one curve; the oldest goes when there are more than :attr:`MAX_POINTS`."""
        wavelengths = np.asarray(wavelengths, np.float64).ravel()
        values = np.asarray(values, np.float64).ravel()
        if wavelengths.size != values.size or not values.size:
            raise ValueError(f"{values.size} values for {wavelengths.size} wavelengths")
        if len(self._points) >= self.MAX_POINTS:
            self._points.pop(0)
        used = {p.colour for p in self._points}
        colour = next(c for c in COLOURS if c not in used)
        self._number += 1
        point = SpectrumPoint(self._number, colour, int(line), int(row), wavelengths, values, label)
        self._points.append(point)
        self.clear_button.setEnabled(True)
        self.plot.update()
        return point

    def clear(self) -> None:
        """Remove every curve; numbering starts again at 1."""
        self._points.clear()
        self._number = 0
        self.clear_button.setEnabled(False)
        self.set_note('')
        self.plot.update()

    def _on_clear(self) -> None:
        self.clear()
        self.cleared.emit()


class _SpectrumPlot(QWidget):
    """Painted part of :class:`SpectrumPanel`: axes and curves on the left, the key on the right."""

    KEY_WIDTH = 150
    KEY_ROW = 15

    def __init__(self, panel: SpectrumPanel) -> None:
        super().__init__(panel)
        self._panel = panel
        self._tokens: dict = {}
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def sizeHint(self):
        return QSize(420, 140)

    def minimumSizeHint(self):
        return QSize(160, 24)

    def _font(self, pixels: int, bold: bool = False) -> QFont:
        font = QFont(self.font())
        font.setPixelSize(pixels)
        font.setBold(bold)
        return font

    def _colour(self, name: str) -> QColor:
        return QColor(self._tokens.get(name) or self._tokens.get('text') or 'gray')

    def paintEvent(self, event) -> None:
        self._tokens = QApplication.instance().property('theme_tokens') or {}
        points = self._panel.points
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        if not points:
            p.setPen(self._colour('text_faint'))
            p.setFont(self._font(11))
            p.drawText(QRectF(0, 0, w, h), Qt.AlignCenter | Qt.TextWordWrap, EMPTY_TEXT)
            p.end()
            return

        key = self.KEY_WIDTH if w >= 2.5 * self.KEY_WIDTH else 0
        left, bottom = 44, 15 if h >= 60 else 0
        area = QRectF(left, 3, max(10.0, w - left - key - 6), max(8.0, h - bottom - 5))

        finite = [(pt.wavelengths[np.isfinite(pt.values)], pt.values[np.isfinite(pt.values)]) for pt in points]
        xs = np.concatenate([pt.wavelengths for pt in points])
        ys = np.concatenate([v for _, v in finite]) if any(v.size for _, v in finite) else np.zeros(1)
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = min(0.0, float(ys.min())), float(ys.max())
        if x1 <= x0:
            x1 = x0 + 1.0
        if y1 <= y0:
            y1 = y0 + 1.0
        y1 += (y1 - y0) * 0.06

        def at(x: float, y: float) -> QPointF:
            return QPointF(area.left() + (x - x0) / (x1 - x0) * area.width(),
                           area.bottom() - (y - y0) / (y1 - y0) * area.height())

        # grid and tick labels
        p.setFont(self._font(10))
        y_ticks = nice_ticks(y0, y1, 3 if area.height() < 90 else 5)
        y_step = y_ticks[1] - y_ticks[0] if len(y_ticks) > 1 else 1.0
        for value in y_ticks:
            y = at(x0, value).y()
            p.setPen(QPen(self._colour('chart_grid'), 1))
            p.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
            p.setPen(self._colour('text_faint'))
            p.drawText(QRectF(0, y - 7, left - 5, 14), Qt.AlignRight | Qt.AlignVCenter, format_tick(value, y_step))
        if bottom:
            x_ticks = nice_ticks(x0, x1, max(2, int(area.width() // 90)))
            x_step = x_ticks[1] - x_ticks[0] if len(x_ticks) > 1 else 1.0
            for value in x_ticks:
                x = at(value, y0).x()
                p.setPen(QPen(self._colour('chart_grid'), 1))
                p.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()))
                p.setPen(self._colour('text_faint'))
                p.drawText(QRectF(x - 30, area.bottom() + 1, 60, 14), Qt.AlignHCenter | Qt.AlignVCenter,
                           format_tick(value, x_step))
            p.drawText(QRectF(area.right() - 60, area.bottom() + 1, 60, 14), Qt.AlignRight | Qt.AlignVCenter, "nm")
        p.setPen(QPen(self._colour('border_strong'), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(area)
        if self._panel.unit:
            p.setPen(self._colour('text_muted'))
            p.drawText(QRectF(area.left() + 5, area.top() + 2, area.width() - 10, 14),
                       Qt.AlignLeft | Qt.AlignVCenter, self._panel.unit)

        # curves: a gap where a value is NaN / inf
        p.setClipRect(area.adjusted(-1, -1, 1, 1))
        for pt in points:
            path, pen_down = QPainterPath(), False
            for x, y in zip(pt.wavelengths, pt.values):
                if not np.isfinite(y):
                    pen_down = False
                    continue
                if pen_down:
                    path.lineTo(at(x, y))
                else:
                    path.moveTo(at(x, y))
                    pen_down = True
            p.setPen(QPen(QColor(pt.colour), 1.6))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
        p.setClipping(False)

        # key: number, class, slit row
        if key:
            x = w - key + 6
            for i, pt in enumerate(points):
                y = 3 + i * self.KEY_ROW
                if y + self.KEY_ROW > h:
                    break
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(pt.colour))
                p.drawEllipse(QPointF(x + 4, y + self.KEY_ROW / 2), 4.0, 4.0)
                p.setPen(self._colour('text'))
                p.setFont(self._font(11, bold=True))
                number = str(pt.number)
                p.drawText(QRectF(x + 13, y, 20, self.KEY_ROW), Qt.AlignLeft | Qt.AlignVCenter, number)
                offset = 13 + p.fontMetrics().horizontalAdvance(number) + 5
                p.setFont(self._font(11))
                p.setPen(self._colour('text_muted'))
                text = f"{pt.label} · row {pt.row}" if pt.label else f"row {pt.row}"
                text = p.fontMetrics().elidedText(text, Qt.ElideRight, int(key - offset - 8))
                p.drawText(QRectF(x + offset, y, key - offset - 6, self.KEY_ROW), Qt.AlignLeft | Qt.AlignVCenter, text)
        p.end()
