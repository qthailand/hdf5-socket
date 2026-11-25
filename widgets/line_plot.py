"""Small themed line plot: a few series, an optional tolerance band, a message when empty.

Used on the Setup tab for the sensor temperature over time, the brightness along the slit and the
model input against the training spectra; :class:`RangeBar` shows the model bands inside the input range.
"""
from __future__ import annotations

import math

from PyQt5.QtCore import QPointF, QRectF, QSize, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import QApplication, QSizePolicy, QWidget

_FALLBACK = {'text': '#E6E8EB', 'text_muted': '#9AA3AF', 'text_faint': '#6B7280', 'border': '#2A2F3A',
             'chart_grid': '#232834', 'accent': '#3B82F6', 'success': '#34D399', 'warning': '#F59E0B',
             'danger': '#EF4444'}


def _theme_colour(key: str) -> QColor:
    """A theme token (``accent``, ``warning`` ...) or a literal ``#RRGGBB``."""
    if str(key).startswith('#'):
        return QColor(key)
    tokens = QApplication.instance().property('theme_tokens') or {}
    return QColor(tokens.get(key) or _FALLBACK.get(key, '#888888'))


class LinePlot(QWidget):
    """``set_series([(label, xs, ys, colour token, dashed)])``; x labels by ``x_format``."""

    def __init__(self, unit: str = '', x_format=None, parent=None) -> None:
        super().__init__(parent)
        self.unit = unit
        self.x_format = x_format or (lambda x: f"{x:g}")
        self.series: list = []
        self.band: tuple | None = None             # (lo, hi, colour token, label)
        self.message = 'no data'
        self.y_min_span = 0.0                      # at least this y range (e.g. 2 °C)
        self.setMinimumHeight(90)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def sizeHint(self) -> QSize:
        return QSize(320, 130)

    def set_series(self, series: list, band: tuple | None = None, message: str = 'no data') -> None:
        self.series = [s for s in series if len(s[1]) >= 2]
        self.band, self.message = band, message
        self.update()

    def _colour(self, key: str) -> QColor:
        return _theme_colour(key)

    def _font(self, pixels: int) -> QFont:
        font = QFont(self.font())
        font.setPixelSize(pixels)
        return font

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setFont(self._font(10))
        w, h = self.width(), self.height()
        if not self.series:
            p.setPen(self._colour('text_faint'))
            p.drawText(QRectF(0, 0, w, h), Qt.AlignCenter | Qt.TextWordWrap, self.message)
            p.end()
            return
        xs = [x for s in self.series for x in s[1]]
        ys = [y for s in self.series for y in s[2] if y is not None and math.isfinite(y)]
        if self.band is not None:
            ys += [self.band[0], self.band[1]]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        if y1 - y0 < self.y_min_span:
            mid = (y0 + y1) / 2
            y0, y1 = mid - self.y_min_span / 2, mid + self.y_min_span / 2
        pad = (y1 - y0) * 0.08 or 1.0
        y0, y1 = y0 - pad, y1 + pad
        if x1 == x0:
            x1 = x0 + 1
        metrics = p.fontMetrics()
        left = max(metrics.horizontalAdvance(f"{v:.1f} {self.unit}") for v in (y0, y1)) + 6
        legend_h = 14
        plot = QRectF(left, 4, w - left - 4, h - 4 - 14 - legend_h)

        def point(x, y):
            return QPointF(plot.left() + (x - x0) / (x1 - x0) * plot.width(),
                           plot.bottom() - (y - y0) / (y1 - y0) * plot.height())

        p.setPen(QPen(self._colour('chart_grid'), 1))
        p.drawRect(plot)
        if self.band is not None:
            lo, hi, colour, _ = self.band
            band = QColor(self._colour(colour))
            band.setAlpha(40)
            p.fillRect(QRectF(point(x0, hi), point(x1, lo)), band)
        p.setPen(self._colour('text_faint'))
        for value, y in ((y1 - pad, plot.top()), (y0 + pad, plot.bottom() - 10)):
            p.drawText(QRectF(0, y, left - 4, 12), Qt.AlignRight, f"{value:.1f} {self.unit}".strip())
        p.drawText(QRectF(plot.left(), plot.bottom() + 1, plot.width() / 2, 12), Qt.AlignLeft, self.x_format(x0))
        p.drawText(QRectF(plot.center().x(), plot.bottom() + 1, plot.width() / 2, 12), Qt.AlignRight,
                   self.x_format(x1))
        for label, sx, sy, colour, dashed in self.series:
            path = None
            for x, y in zip(sx, sy):
                if y is None or not math.isfinite(y):
                    path = path if path is None else path      # gaps: keep drawing from the next value
                    continue
                if path is None:
                    path = QPainterPath(point(x, y))
                else:
                    path.lineTo(point(x, y))
            if path is not None:
                p.setPen(QPen(self._colour(colour), 1.6, Qt.DashLine if dashed else Qt.SolidLine))
                p.setBrush(Qt.NoBrush)
                p.drawPath(path)
        x = plot.left()
        legend_y = h - legend_h
        items = [(s[0], s[3]) for s in self.series] + ([(self.band[3], self.band[2])] if self.band and self.band[3] else [])
        for label, colour in items:
            p.setPen(QPen(self._colour(colour), 2))
            p.drawLine(QPointF(x, legend_y + 7), QPointF(x + 12, legend_y + 7))
            p.setPen(self._colour('text_muted'))
            p.drawText(QRectF(x + 16, legend_y, w, 14), Qt.AlignLeft | Qt.AlignVCenter, label)
            x += 16 + metrics.horizontalAdvance(label) + 14
        p.end()


class RangeBar(QWidget):
    """The input's wavelength range as a bar, the model's bands marked on it (one tick per band)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.input_range: tuple | None = None       # (lo, hi) nm of the model input
        self.bands: list = []                        # nm the model reads
        self.state = 'accent'                        # colour token of the model bands (warning: outside)
        self.setFixedHeight(46)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_range(self, input_range, bands, state: str = 'accent') -> None:
        self.input_range = None if input_range is None else (float(input_range[0]), float(input_range[1]))
        self.bands = [float(b) for b in bands if b is not None and math.isfinite(b)]
        self.state = state
        self.update()

    def _font(self, pixels: int) -> QFont:
        font = QFont(self.font())
        font.setPixelSize(pixels)
        return font

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setFont(self._font(10))
        w = self.width()
        if self.input_range is None:
            p.setPen(_theme_colour('text_faint'))
            p.drawText(QRectF(0, 0, w, self.height()), Qt.AlignCenter, 'no wavelengths')
            p.end()
            return
        lo, hi = self.input_range
        if self.bands:
            lo, hi = min(lo, min(self.bands)), max(hi, max(self.bands))
        span = (hi - lo) or 1.0
        left, right, top = 4.0, w - 4.0, 10.0

        def x(nm):
            return left + (nm - lo) / span * (right - left)

        bar = QRectF(x(self.input_range[0]), top, x(self.input_range[1]) - x(self.input_range[0]), 12)
        p.setPen(QPen(_theme_colour('border'), 1))
        p.setBrush(_theme_colour('chart_grid'))
        p.drawRoundedRect(bar, 3, 3)
        colour = _theme_colour(self.state)
        if self.bands:
            fill = QColor(colour)
            fill.setAlpha(70)
            p.setPen(QPen(colour, 1))
            p.setBrush(fill)
            p.drawRoundedRect(QRectF(x(min(self.bands)), top - 4, max(2.0, x(max(self.bands)) - x(min(self.bands))),
                                     20), 3, 3)
            p.setPen(QPen(colour, 1))
            for nm in self.bands:
                p.drawLine(QPointF(x(nm), top - 2), QPointF(x(nm), top + 14))
        p.setPen(_theme_colour('text_muted'))
        label_y = top + 20
        p.drawText(QRectF(bar.left(), label_y, 80, 14), Qt.AlignLeft, f"{self.input_range[0]:.0f}")
        p.drawText(QRectF(bar.right() - 80, label_y, 80, 14), Qt.AlignRight, f"{self.input_range[1]:.0f} nm")
        if self.bands:
            text = f"model {len(self.bands)} bands {min(self.bands):.0f}–{max(self.bands):.0f} nm"
            centre = (x(min(self.bands)) + x(max(self.bands))) / 2
            width = p.fontMetrics().horizontalAdvance(text) + 8
            cx = min(max(centre - width / 2, bar.left() + 30), bar.right() - 40 - width)
            p.setPen(colour)
            p.drawText(QRectF(cx, label_y, width, 14), Qt.AlignCenter, text)
        p.end()
