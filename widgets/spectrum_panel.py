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
    model_values: np.ndarray | None = None   # (training bands,) what the model read: its bands of ``values``


@dataclass
class TrainingBand:
    """The training spectra of one class: drawn as a band mean ± SD behind the clicked curves."""
    label: str
    colour: str                 # '#RRGGBB' (the class colour of the belt map)
    mean: np.ndarray            # (bands,)
    sd: np.ndarray              # (bands,)
    n_spectra: int = 0


def snv(values: np.ndarray, use: np.ndarray | None = None) -> np.ndarray:
    """Standard normal variate as the model package applies it: mean 0, population SD 1.

    ``use`` (bool mask) picks the bands the mean and SD are taken over (the training range).
    All NaN when those bands are not finite or constant.
    """
    values = np.asarray(values, np.float64)
    part = values if use is None else values[use]
    if not part.size or not np.isfinite(part).all() or part.std() == 0:
        return np.full(values.shape, np.nan)
    return (values - part.mean()) / part.std()


def resample(wavelengths: np.ndarray, values: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """``values`` on the ``wanted`` nm as ``inference_engine.Resampled`` reads them: linear between
    the two input bands around each wanted band, only that band when one lies exactly on it, linear
    extrapolation from the two edge bands up to half a band outside the input; further out NaN.
    A NaN / inf input band reaches only the wanted bands that read it."""
    order = np.argsort(wavelengths)
    x, y = np.asarray(wavelengths, np.float64)[order], np.asarray(values, np.float64)[order]
    wanted = np.asarray(wanted, np.float64)
    if x.size < 2:
        return np.full(wanted.shape, np.nan)
    right = np.clip(np.searchsorted(x, wanted), 1, x.size - 1)
    left = right - 1
    weight = (wanted - x[left]) / (x[right] - x[left])
    left, right = np.where(weight == 1.0, right, left), np.where(weight == 0.0, left, right)
    with np.errstate(invalid='ignore'):
        z = np.where(left == right, y[left], (1.0 - weight) * y[left] + weight * y[right])
    half = 0.5 * float(np.median(np.diff(wanted))) if wanted.size > 1 else 0.0
    z[(wanted < x[0] - half - 1e-6) | (wanted > x[-1] + half + 1e-6)] = np.nan
    return z


def derive(matrix: np.ndarray, z: np.ndarray) -> np.ndarray:
    """``matrix @ z`` where only the rows whose window (non-zero coefficients) holds a NaN / inf
    band of ``z`` are NaN: one bad band does not blank the whole derivative."""
    bad = ~np.isfinite(z)
    out = matrix @ np.where(bad, 0.0, z)
    if bad.any():
        out[(matrix[:, bad] != 0).any(axis=1)] = np.nan
    return out


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
        self._training_wavelengths = np.zeros(0)
        self._training: dict = {'input': [], 'snv': [], 'derivative': []}    # view -> [TrainingBand]
        self._derivative: np.ndarray | None = None          # (B, B): D @ x = SG 1st derivative on the training axis
        self._selected = np.zeros(0)                        # nm of the bands the model uses (marked in a model view)
        self.training_button = QPushButton("Model reference")
        self.training_button.setToolTip(
            "Draw the model's training spectra behind the clicked ones: per class a band mean ± 1 SD\n"
            "(from the model package). Use these to compare shape, scale and noise.\n"
            "A point outside mean ± 1 SD alone does not prove a wrong prediction.")
        self.snv_button = QPushButton("SNV")
        self.snv_button.setToolTip(
            "Show the spectra after SNV (mean 0, SD 1 over the training bands): what the model compares.\n"
            "Brightness differences vanish here; shape and noise remain.")
        self.derivative_button = QPushButton("1st deriv")
        self.derivative_button.setToolTip(
            "Show the Savitzky-Golay first derivative (the model package's window / order, per nm):\n"
            "what the model compares. Ticks at the bottom mark the bands the model uses.\n"
            "Offsets vanish here; a gain (brightness x factor) still scales the curve, noise is amplified.")
        for button in (self.training_button, self.snv_button, self.derivative_button):
            button.setObjectName("toggleButton")
            button.setCheckable(True)
            button.setFocusPolicy(Qt.NoFocus)
            button.setEnabled(False)
            button.toggled.connect(lambda _on: self.plot.update())
            header.addWidget(button)
        # one model view at a time
        self.snv_button.toggled.connect(lambda on: on and self.derivative_button.setChecked(False))
        self.derivative_button.toggled.connect(lambda on: on and self.snv_button.setChecked(False))
        self.clear_button = QPushButton("Clear")
        self.clear_button.setObjectName("presetButton")
        self.clear_button.setToolTip("Remove every spectrum and its mark on the belt map")
        self.clear_button.setFocusPolicy(Qt.NoFocus)
        self.clear_button.setEnabled(False)
        self.clear_button.clicked.connect(self._on_clear)
        header.addWidget(self.clear_button)
        layout.addLayout(header)
        self.plot = _SpectrumPlot(self)
        self.reference_label = QLabel("No training reference supplied by this model")
        self.reference_label.setObjectName("metricCaption")
        self.reference_label.setWordWrap(True)
        layout.addWidget(self.reference_label)
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

    def set_training(self, wavelengths=None, bands=(), snv_bands=(), derivative_bands=(), derivative=None,
                     selected=None) -> None:
        """The training spectra of the model in use (``None`` / empty: the model brings none).

        Args:
            wavelengths: (bands,) nm of the training axis.
            bands: :class:`TrainingBand` per class in the units of the model input.
            snv_bands: the same after SNV; empty when the model does not use SNV (no SNV view).
            derivative_bands: the same after the first derivative; empty when the model does not use it.
            derivative: (B, B) matrix, ``D @ x`` = that derivative of a spectrum on the training axis.
            selected: nm of the bands the model's features use (marked in the SNV / derivative view).
        """
        had = bool(self._training['input'])
        self._training_wavelengths = np.zeros(0) if wavelengths is None else np.asarray(wavelengths, np.float64)
        has_derivative = derivative is not None and bool(derivative_bands)
        self._derivative = np.asarray(derivative, np.float64) if has_derivative else None
        self._training = {'input': list(bands), 'snv': list(snv_bands),
                          'derivative': list(derivative_bands) if has_derivative else []}
        self._selected = np.zeros(0) if selected is None else np.asarray(selected, np.float64).ravel()
        self.reference_label.setText(
            f"Model reference: {len(bands)} classes · shaded mean ± 1 SD · dashed mean · solid clicked spectra"
            if bands else "No training reference supplied by this model · captured spectra remain available")
        self.training_button.setEnabled(bool(bands))
        self.snv_button.setEnabled(bool(snv_bands))
        if not snv_bands:
            self.snv_button.setChecked(False)
        self.derivative_button.setEnabled(has_derivative)
        if not has_derivative:
            self.derivative_button.setChecked(False)
        if not bands:
            self.training_button.setChecked(False)
        elif not had:
            self.training_button.setChecked(True)          # a model with training spectra: show them
        self.plot.update()

    @property
    def snv_shown(self) -> bool:
        return self.snv_button.isChecked()

    @property
    def derivative_shown(self) -> bool:
        return self.derivative_button.isChecked() and self._derivative is not None

    @property
    def view(self) -> str:
        """``'input'``, ``'snv'`` or ``'derivative'``."""
        return 'derivative' if self.derivative_shown else 'snv' if self.snv_shown else 'input'

    @property
    def selected_wavelengths(self) -> np.ndarray:
        """nm of the model's selected bands, marked in a model view (empty in the input view)."""
        return self._selected if self.view != 'input' else np.zeros(0)

    def shown_training(self) -> tuple[np.ndarray, list[TrainingBand]]:
        """``(wavelengths, bands)`` drawn now (no bands when ``Training`` is off)."""
        if not self.training_button.isChecked():
            return self._training_wavelengths, []
        return self._training_wavelengths, self._training[self.view]

    def shown_curve(self, point: SpectrumPoint) -> tuple[np.ndarray, np.ndarray]:
        """``(wavelengths, values)`` of ``point`` in the current view.

        The derivative view applies the package's derivative to the model's own bands of the
        point (:attr:`SpectrumPoint.model_values`, read through the model's binding: the same
        columns, interpolation and edge extrapolation). A point without them is resampled here the
        way the model's binding resamples (:func:`resample`). Only the
        derivative bands whose window holds a NaN / inf input band are NaN.
        """
        if not self.derivative_shown:
            return point.wavelengths, self.shown_values(point)
        wl, D = self._training_wavelengths, self._derivative
        z = point.model_values
        if z is None or z.size != D.shape[1]:
            z = resample(point.wavelengths, point.values, wl)
        return wl, derive(D, z)

    def shown_values(self, point: SpectrumPoint) -> np.ndarray:
        """The curve of ``point`` on its own wavelengths (as received, or after SNV)."""
        if not self.snv_shown:
            return point.values
        wl = self._training_wavelengths
        if not wl.size:
            return snv(point.values)
        half = (wl[-1] - wl[0]) / max(1, wl.size - 1) / 2
        use = (point.wavelengths >= wl[0] - half) & (point.wavelengths <= wl[-1] + half)
        return snv(point.values, use)

    def add_point(self, wavelengths, values, line: int, row: int, label: str = '',
                  model_values=None) -> SpectrumPoint:
        """Add one curve; the oldest goes when there are more than :attr:`MAX_POINTS`.

        ``model_values``: the point on the model's training bands as the model read it (used by
        the derivative view); ``None`` when not known.
        """
        wavelengths = np.asarray(wavelengths, np.float64).ravel()
        values = np.asarray(values, np.float64).ravel()
        if wavelengths.size != values.size or not values.size:
            raise ValueError(f"{values.size} values for {wavelengths.size} wavelengths")
        if len(self._points) >= self.MAX_POINTS:
            self._points.pop(0)
        used = {p.colour for p in self._points}
        colour = next(c for c in COLOURS if c not in used)
        self._number += 1
        if model_values is not None:
            model_values = np.asarray(model_values, np.float64).ravel()
        point = SpectrumPoint(self._number, colour, int(line), int(row), wavelengths, values, label, model_values)
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
        train_wl, bands = self._panel.shown_training()
        shown = [self._panel.shown_curve(pt) for pt in points]
        curves = [values for _wl, values in shown]
        selected = self._panel.selected_wavelengths
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        if not points and not bands:
            p.setPen(self._colour('text_faint'))
            p.setFont(self._font(11))
            p.drawText(QRectF(0, 0, w, h), Qt.AlignCenter | Qt.TextWordWrap, EMPTY_TEXT)
            p.end()
            return

        key = self.KEY_WIDTH if w >= 2.5 * self.KEY_WIDTH else 0
        left, bottom = 44, 15 if h >= 60 else 0
        area = QRectF(left, 3, max(10.0, w - left - key - 6), max(8.0, h - bottom - 5))

        xs = np.concatenate([wl for wl, _v in shown] + ([train_wl] if bands else []))
        ys = np.concatenate([v[np.isfinite(v)] for v in curves]
                            + [np.concatenate((b.mean - b.sd, b.mean + b.sd)) for b in bands] + [np.zeros(0)])
        if not ys.size:
            ys = np.zeros(1)
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
        unit = {'snv': "SNV of the model input",
                'derivative': "1st derivative of the model input (per nm)"}.get(self._panel.view, self._panel.unit)
        if unit:
            p.setPen(self._colour('text_muted'))
            p.drawText(QRectF(area.left() + 5, area.top() + 2, area.width() - 10, 14),
                       Qt.AlignLeft | Qt.AlignVCenter, unit)

        p.setClipRect(area.adjusted(-1, -1, 1, 1))
        # the bands the model uses: short ticks on the bottom edge, and the zero line of a model view
        if self._panel.view == 'derivative' and y0 < 0 < y1:
            p.setPen(QPen(self._colour('border_strong'), 1, Qt.DotLine))
            p.drawLine(at(x0, 0.0), at(x1, 0.0))
        if selected.size:
            p.setPen(QPen(self._colour('text_muted'), 1.4))
            for value in selected:
                x = at(float(value), y0).x()
                p.drawLine(QPointF(x, area.bottom()), QPointF(x, area.bottom() - 7))
        # training spectra: band mean ± SD and the mean (dashed), behind the clicked curves
        for band in bands:
            colour = self._band_colour(band)
            upper = [at(x, y) for x, y in zip(train_wl, band.mean + band.sd)]
            lower = [at(x, y) for x, y in zip(train_wl, band.mean - band.sd)]
            outline = QPainterPath()
            outline.moveTo(upper[0])
            for q in upper[1:] + lower[::-1]:
                outline.lineTo(q)
            outline.closeSubpath()
            fill = QColor(colour)
            fill.setAlpha(46)
            p.setPen(Qt.NoPen)
            p.setBrush(fill)
            p.drawPath(outline)
            centre = QPainterPath()
            centre.moveTo(at(train_wl[0], band.mean[0]))
            for x, y in zip(train_wl[1:], band.mean[1:]):
                centre.lineTo(at(x, y))
            p.setPen(QPen(colour, 1.2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawPath(centre)

        # curves: a gap where a value is NaN / inf
        for pt, (wavelengths, values) in zip(points, shown):
            path, pen_down = QPainterPath(), False
            for x, y in zip(wavelengths, values):
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
            # ... then the training bands (a bar, not a dot)
            p.setFont(self._font(11))
            for i, band in enumerate(bands, start=len(points)):
                y = 3 + i * self.KEY_ROW
                if y + self.KEY_ROW > h:
                    break
                fill = self._band_colour(band)
                fill.setAlpha(150)
                p.setPen(Qt.NoPen)
                p.setBrush(fill)
                p.drawRect(QRectF(x, y + self.KEY_ROW / 2 - 3, 9, 6))
                p.setPen(self._colour('text_muted'))
                text = f"{band.label} train" + (f" · n {band.n_spectra:,}" if band.n_spectra else "")
                text = p.fontMetrics().elidedText(text, Qt.ElideRight, int(key - 21))
                p.drawText(QRectF(x + 13, y, key - 19, self.KEY_ROW), Qt.AlignLeft | Qt.AlignVCenter, text)
        p.end()

    def _band_colour(self, band: TrainingBand) -> QColor:
        """The class colour; one too close to the panel background becomes the muted text colour."""
        colour, back = QColor(band.colour), self._colour('surface_alt')
        if abs(colour.lightness() - back.lightness()) < 40 and colour.saturation() < 60:
            return self._colour('text_muted')
        return colour
