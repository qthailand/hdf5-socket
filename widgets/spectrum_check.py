# -*- coding: utf-8 -*-
"""Plot that checks a reflectance calibration: measured spectrum of a standard against its
certified curve, and the error per band.

    report = reflectance_builder.verify_lines(lines, rows, factors, reference, wavelengths)
    SpectrumCheckDialog(parent, wavelengths, report, "Spectralon 99 %", "refl_b2_g1.npy").exec_()

GUI thread only.
"""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, QSize, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import QApplication, QDialog, QDialogButtonBox, QLabel, QSizePolicy, QVBoxLayout, QWidget

from widgets.spectrum_panel import format_tick, nice_ticks

MEASURED, REFERENCE, ERROR, LIMIT = '#F97316', '#22D3EE', '#E879F9', '#94A3B8'


class CheckPlot(QWidget):
    """Curves over wavelength: ``add_curve`` / ``add_band`` / ``add_level`` then it paints itself."""

    def __init__(self, unit: str, parent=None, include_zero: bool = False) -> None:
        super().__init__(parent)
        self.unit, self._include_zero = unit, include_zero
        self.curves: list[tuple[np.ndarray, np.ndarray, str, bool, str]] = []   # x, y, colour, dashed, name
        self.bands: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []   # x, lower, upper, colour
        self.levels: list[tuple[float, str]] = []                               # y, colour
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def sizeHint(self):
        return QSize(640, 220)

    def clear(self) -> None:
        self.curves, self.bands, self.levels = [], [], []
        self.update()

    def add_curve(self, x, y, colour: str, name: str, dashed: bool = False) -> None:
        self.curves.append((np.asarray(x, np.float64), np.asarray(y, np.float64), colour, dashed, name))
        self.update()

    def add_band(self, x, lower, upper, colour: str) -> None:
        self.bands.append((np.asarray(x, np.float64), np.asarray(lower, np.float64),
                           np.asarray(upper, np.float64), colour))
        self.update()

    def add_level(self, y: float, colour: str = LIMIT) -> None:
        self.levels.append((float(y), colour))
        self.update()

    def y_range(self) -> tuple[float, float]:
        ys = np.concatenate([y for _, y, *_ in self.curves] + [v for _, lo, hi, _ in self.bands for v in (lo, hi)]
                            + [np.array([y for y, _ in self.levels])] + [np.zeros(1 if self._include_zero else 0)])
        ys = ys[np.isfinite(ys)]
        if not ys.size:
            return 0.0, 1.0
        y0, y1 = float(ys.min()), float(ys.max())
        pad = (y1 - y0) * 0.08 or max(abs(y1), 1.0) * 0.05
        return y0 - pad, y1 + pad

    def paintEvent(self, event) -> None:
        tokens = QApplication.instance().property('theme_tokens') or {}

        def colour(name: str) -> QColor:
            return QColor(tokens.get(name) or tokens.get('text') or 'gray')

        def font(pixels: int) -> QFont:
            f = QFont(self.font())
            f.setPixelSize(pixels)
            return f

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        if not self.curves:
            p.end()
            return
        left, bottom = 52, 16
        area = QRectF(left, 4, max(10.0, w - left - 8), max(8.0, h - bottom - 6))
        xs = np.concatenate([x for x, *_ in self.curves])
        x0, x1 = float(xs.min()), float(xs.max())
        x1 = x1 if x1 > x0 else x0 + 1.0
        y0, y1 = self.y_range()

        def at(x: float, y: float) -> QPointF:
            return QPointF(area.left() + (x - x0) / (x1 - x0) * area.width(),
                           area.bottom() - (y - y0) / (y1 - y0) * area.height())

        p.setFont(font(10))
        y_ticks = nice_ticks(y0, y1, 5)
        y_step = y_ticks[1] - y_ticks[0] if len(y_ticks) > 1 else 1.0
        for value in y_ticks:
            y = at(x0, value).y()
            p.setPen(QPen(colour('chart_grid'), 1))
            p.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
            p.setPen(colour('text_faint'))
            p.drawText(QRectF(0, y - 7, left - 5, 14), Qt.AlignRight | Qt.AlignVCenter, format_tick(value, y_step))
        x_ticks = nice_ticks(x0, x1, max(2, int(area.width() // 90)))
        x_step = x_ticks[1] - x_ticks[0] if len(x_ticks) > 1 else 1.0
        for value in x_ticks:
            x = at(value, y0).x()
            p.setPen(QPen(colour('chart_grid'), 1))
            p.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()))
            p.setPen(colour('text_faint'))
            if x < area.right() - 50:                      # the last label would run into "nm"
                p.drawText(QRectF(x - 30, area.bottom() + 1, 60, 14), Qt.AlignHCenter | Qt.AlignVCenter,
                           format_tick(value, x_step).replace(',', ''))
        p.drawText(QRectF(area.right() - 60, area.bottom() + 1, 60, 14), Qt.AlignRight | Qt.AlignVCenter, "nm")
        p.setPen(QPen(colour('border_strong'), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(area)

        p.setClipRect(area.adjusted(-1, -1, 1, 1))
        for x, lower, upper, band_colour in self.bands:
            outline = QPainterPath()
            points = [at(a, b) for a, b in zip(x, upper)] + [at(a, b) for a, b in zip(x[::-1], lower[::-1])]
            outline.moveTo(points[0])
            for q in points[1:]:
                outline.lineTo(q)
            outline.closeSubpath()
            fill = QColor(band_colour)
            fill.setAlpha(50)
            p.setPen(Qt.NoPen)
            p.setBrush(fill)
            p.drawPath(outline)
        p.setBrush(Qt.NoBrush)
        for level, level_colour in self.levels:
            p.setPen(QPen(QColor(level_colour), 1, Qt.DotLine))
            p.drawLine(at(x0, level), at(x1, level))
        for x, y, curve_colour, dashed, _ in self.curves:
            path, pen_down = QPainterPath(), False
            for a, b in zip(x, y):
                if not np.isfinite(b):
                    pen_down = False
                    continue
                path.lineTo(at(a, b)) if pen_down else path.moveTo(at(a, b))
                pen_down = True
            p.setPen(QPen(QColor(curve_colour), 1.6, Qt.DashLine if dashed else Qt.SolidLine))
            p.drawPath(path)
        p.setClipping(False)

        # unit (top left) and the key (top right)
        p.setFont(font(11))
        p.setPen(colour('text_muted'))
        p.drawText(QRectF(area.left() + 6, area.top() + 2, area.width() / 2, 14), Qt.AlignLeft | Qt.AlignVCenter,
                   self.unit)
        x = area.right() - 6
        for _, _, curve_colour, dashed, name in reversed(self.curves):
            width = p.fontMetrics().horizontalAdvance(name)
            p.setPen(colour('text_muted'))
            p.drawText(QRectF(x - width, area.top() + 2, width, 14), Qt.AlignLeft | Qt.AlignVCenter, name)
            p.setPen(QPen(QColor(curve_colour), 2, Qt.DashLine if dashed else Qt.SolidLine))
            p.drawLine(QPointF(x - width - 22, area.top() + 9), QPointF(x - width - 5, area.top() + 9))
            x -= width + 34
        p.end()


class SpectrumCheckDialog(QDialog):
    """Measured (mean ± pixel SD) against the certified reflectance, and the error per band."""

    def __init__(self, parent, wavelengths, report: dict, standard: str, factors_label: str) -> None:
        super().__init__(parent)
        self.setWindowTitle("Reflectance check: measured vs. reference")
        self.report = report
        wl = np.asarray(wavelengths, np.float64)
        tolerance = report['tolerance_pct']

        self.spectrum_plot = CheckPlot("reflectance", self)
        self.spectrum_plot.add_band(wl, report['measured'] - report['sd'], report['measured'] + report['sd'], MEASURED)
        self.spectrum_plot.add_curve(wl, report['reference'], REFERENCE, f"reference ({standard})", dashed=True)
        self.spectrum_plot.add_curve(wl, report['measured'], MEASURED, "measured mean ± SD")
        self.error_plot = CheckPlot("error %  (measured − reference) / reference", self, include_zero=True)
        for level in (-tolerance, 0.0, tolerance):
            self.error_plot.add_level(level)
        self.error_plot.add_curve(wl, report['error_pct'], ERROR, f"error (limit ±{tolerance:g} %)")

        worst = f" at {report['worst_nm']:.0f} nm" if report['worst_nm'] is not None else ''
        self.summary = QLabel(
            f"<b>{'✔' if report['ok'] else '⚠'} {report['verdict']}</b><br>"
            f"Error: mean {report['mean_error_pct']:+.2f} %, RMS {report['rms_error_pct']:.2f} %, "
            f"max {report['max_abs_error_pct']:.2f} %{worst}<br>"
            f"Area: slit rows {report['rows'][0]}-{report['rows'][1] - 1}, {report['lines']} lines "
            f"({report['pixels']:,} pixels) · Factors: {factors_label}<br>"
            "The grabbed lines are multiplied by the factors in use. Grab the standard again after the "
            "calibration (or use a second standard, e.g. 20 % grey) for an independent check.")
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.RichText)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.spectrum_plot, stretch=3)
        layout.addWidget(self.error_plot, stretch=2)
        layout.addWidget(self.summary)
        layout.addWidget(buttons)
        self.resize(820, 620)
