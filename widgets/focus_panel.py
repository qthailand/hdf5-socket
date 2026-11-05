# -*- coding: utf-8 -*-
"""Focus aid beside the spectrogram preview, after hsi's sharpness sidebar, in the app's theme.

The operator turns the lens focus ring slowly and watches the number: it is highest when the
image on the slit is sharpest.

* **Score** (big number, bar, trend): :func:`focus.edge_contrast` of the raw frame along the
  chosen axis, in % per pixel - ``Slit`` (axis 0, default: imaging focus, needs edges in the
  scene, e.g. a bar target) or ``λ`` (axis 1: spectrograph focus, needs spectral lines, e.g. a
  line lamp). ``focus.py`` explains why it, and not hsi's score, drives the bar.
* **hsi score** (small line): hsi's :func:`modules.sharpness.sharpness_score` formula on the same
  raw frame, for comparison (hsi's scan window scores its corrected, binned frame, so its values
  differ).
* **Bar / percentage**: ``score / peak``, where the peak is the running maximum that decays by
  0.9995 per score (floor :attr:`FocusPanel.PEAK_FLOOR`), as in hsi's ``SharpnessPanel``;
  ``↺`` forgets it.
* **Trend**: the last :attr:`FocusPanel.HISTORY` scores (30 s at hsi's 5 Hz; ~50 s at the ~3 per second measured at 250 lines/s), scaled to their own
  range so small changes show; the dot marks the best one in the window - turn back towards it.

Colours are theme tokens (``themes/<name>.json``); nothing is hard-coded here. GUI thread only.

    panel = FocusPanel()
    panel.set_value(edge_contrast(frame, panel.axis), reference=sharpness_score(frame, panel.axis))
"""
from __future__ import annotations

from collections import deque

from PyQt5.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (QApplication, QButtonGroup, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                             QVBoxLayout, QWidget)

from themes import make_panel, set_accent

#: Ratio thresholds (score / peak) and their colours, as in hsi's SharpnessPanel.
_LEVELS = ((0.95, 'success', 'ok'), (0.80, 'chart_2', 'info'), (0.50, 'text_muted', 'neutral'),
           (0.0, 'text_faint', 'neutral'))

AXES = ((0, 'Slit', "Imaging focus: edge contrast along the slit (mean over all wavelengths).\n"
                    "Needs edges in the scene under the slit, e.g. a black/white bar target."),
        (1, 'λ', "Spectral focus: contrast along the wavelength axis (mean over the slit).\n"
                 "Needs sharp spectral features, e.g. a line lamp; for focusing the spectrograph."))

UNIT = "%/px"


def format_score(score: float) -> str:
    """Edge contrast text: two decimals below 10, one above."""
    return f"{score:.2f}" if score < 10 else f"{score:.1f}"


def format_reference(score: float) -> str:
    """hsi score text: thousands separators from 1000, one decimal below 100 (as hsi)."""
    if score >= 1000:
        return f"{score:,.0f}"
    if score >= 100:
        return f"{score:.0f}"
    return f"{score:.1f}"


def level(ratio: float) -> tuple[str, str]:
    """``(colour token, accent)`` for a ``score / peak`` ratio."""
    return next((colour, accent) for threshold, colour, accent in _LEVELS if ratio >= threshold)


class FocusPanel(QFrame):
    """Narrow card: header (axis switch, reset) above a painted gauge (bar, number, trend).

    Signals:
        axis_changed(int): the operator picked another axis (0 slit, 1 spectral); the panel has
            already reset itself, the next :meth:`set_value` should use the new axis.
    """

    axis_changed = pyqtSignal(int)

    INTERVAL_S = 0.2          # hsi SHARPNESS_INTERVAL: a score every 0.2 s
    PEAK_DECAY = 0.9995       # hsi SharpnessPanel: per set_value (~5 Hz -> half-life ~2.3 min)
    PEAK_FLOOR = 1e-6         # % per pixel (hsi's floor of 1.0 is for its own, larger scale)
    HISTORY = 150             # scores in the trend line (30 s at 5 Hz)
    WIDTH = 200

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("focusPanel")
        make_panel(self, accent='neutral', alt=True)
        self.setFixedWidth(self.WIDTH)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setToolTip("Focus aid: turn the lens focus ring slowly; the score is highest in focus.\n"
                        f"Score = edge contrast of the raw frame ({UNIT}): RMS step between neighbouring pixels "
                        "of the\nprofile (mean over the other axis), relative to its mean; every 6 GUI refreshes.\n"
                        "Bar and % = score / running peak (the peak decays slowly, as in hsi).\n"
                        "hsi score = hsi's sharpness_score formula on this raw frame (hsi itself scores the\n"
                        "corrected, binned frame, so the values in hsi's scan window differ).\n"
                        "Focus on a still target under the slit: on a moving belt the score follows the scene.\n"
                        "Scores compare only for the same scene and bin.")
        self._axis = 0
        self._score = 0.0
        self._reference: float | None = None
        self._peak = self.PEAK_FLOOR
        self._history: deque[float] = deque(maxlen=self.HISTORY)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 8, 8)
        layout.setSpacing(4)
        header = QHBoxLayout()
        header.setSpacing(4)
        caption = QLabel("FOCUS")
        caption.setObjectName("caption")
        header.addWidget(caption)
        header.addSpacing(4)
        self._axis_group = QButtonGroup(self)
        self._axis_group.setExclusive(True)
        for axis, text, tip in AXES:
            button = QPushButton(text)
            button.setObjectName("segmentButton")
            button.setCheckable(True)
            button.setChecked(axis == self._axis)
            button.setToolTip(tip)
            button.setFocusPolicy(Qt.NoFocus)
            self._axis_group.addButton(button, axis)
            header.addWidget(button)
        self._axis_group.buttonClicked.connect(lambda b: self.set_axis(self._axis_group.id(b)))
        header.addStretch()
        self._reset_button = QPushButton("↺")
        self._reset_button.setObjectName("presetButton")
        self._reset_button.setToolTip("Forget the peak and the trend (e.g. after changing the scene or exposure)")
        self._reset_button.setFocusPolicy(Qt.NoFocus)
        self._reset_button.clicked.connect(self.reset)
        header.addWidget(self._reset_button)
        layout.addLayout(header)
        self.gauge = _FocusGauge(self)
        layout.addWidget(self.gauge, stretch=1)

    # ------------------------------------------------------------------------ public ---

    @property
    def axis(self) -> int:
        """0 = along the slit (imaging focus), 1 = along the wavelength axis (spectral focus)."""
        return self._axis

    @property
    def score(self) -> float:
        return self._score

    @property
    def reference(self) -> float | None:
        """hsi's score of the last frame, or ``None``."""
        return self._reference

    @property
    def peak(self) -> float:
        return self._peak

    @property
    def ratio(self) -> float:
        """``score / peak`` in [0, 1] (what the bar and the percentage show)."""
        return min(1.0, self._score / max(self._peak, self.PEAK_FLOOR)) if self._history else 0.0

    @property
    def history(self) -> list[float]:
        return list(self._history)

    def set_value(self, score: float, reference: float | None = None) -> None:
        """Show a new score (and hsi's score of the same frame); a score above the peak raises
        it, otherwise the peak decays a little. A NaN score (empty frame) is ignored."""
        score = float(score)
        if score != score:
            return
        self._score = score
        self._reference = None if reference is None else float(reference)
        self._history.append(score)
        self._peak = score if score > self._peak else max(self._peak * self.PEAK_DECAY, self.PEAK_FLOOR)
        set_accent(self, level(self.ratio)[1])
        self.gauge.update()

    def set_axis(self, axis: int) -> None:
        """Score along another axis from now on; the scales differ, so the peak and trend restart."""
        axis = int(axis)
        if axis not in (0, 1):
            raise ValueError(f"axis must be 0 (slit) or 1 (spectral), got {axis}")
        button = self._axis_group.button(axis)
        if not button.isChecked():
            button.setChecked(True)
        if axis == self._axis:
            return
        self._axis = axis
        self.reset()
        self.axis_changed.emit(axis)

    def reset(self) -> None:
        """Forget the score, the peak and the trend (idle display until the next score)."""
        self._score = 0.0
        self._reference = None
        self._peak = self.PEAK_FLOOR
        self._history.clear()
        set_accent(self, 'neutral')
        self.gauge.update()


class _FocusGauge(QWidget):
    """Painted part of :class:`FocusPanel`: vertical bar on the left, numbers and trend on the right."""

    BAR = 14

    def __init__(self, panel: FocusPanel) -> None:
        super().__init__(panel)
        self._panel = panel
        self._tokens: dict = {}
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def sizeHint(self):
        return QSize(FocusPanel.WIDTH - 20, 140)

    def minimumSizeHint(self):
        return QSize(FocusPanel.WIDTH - 20, 24)

    def _font(self, pixels: int, bold: bool = False, mono: bool = False) -> QFont:
        font = QFont(self.font())
        if mono:
            font.setFamily(self._tokens.get('mono_family') or font.family())
        font.setPixelSize(pixels)
        font.setBold(bold)
        return font

    def _colour(self, name: str) -> QColor:
        return QColor(self._tokens.get(name) or self._tokens.get('text') or 'gray')

    def _text(self, p: QPainter, rect: QRectF, text: str, colour: str, pixels: int, bold: bool = False,
              align=Qt.AlignLeft) -> float:
        """Draw one line; return its width."""
        p.setPen(self._colour(colour))
        p.setFont(self._font(pixels, bold))
        p.drawText(rect, align | Qt.AlignVCenter, text)
        return p.fontMetrics().horizontalAdvance(text)

    def paintEvent(self, event) -> None:
        self._tokens = QApplication.instance().property('theme_tokens') or {}
        panel = self._panel
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        has_data = bool(panel.history)
        ratio = panel.ratio
        colour = level(ratio)[0]

        # vertical bar: score / peak; the dashed line at the top is the peak
        bar = QRectF(0.5, 0.5, self.BAR, h - 1.0)
        p.setPen(QPen(self._colour('border'), 1))
        p.setBrush(self._colour('chart_grid'))
        p.drawRoundedRect(bar, 3, 3)
        if has_data and ratio > 0:
            inner = bar.height() - 4
            p.setPen(Qt.NoPen)
            p.setBrush(self._colour(colour))
            p.drawRoundedRect(QRectF(bar.left() + 2, bar.bottom() - 2 - inner * ratio, bar.width() - 4,
                                     inner * ratio), 2, 2)
        p.setPen(QPen(self._colour('text_faint'), 1, Qt.DashLine))
        p.drawLine(QPointF(bar.left() + 1, bar.top() + 2), QPointF(bar.right() - 1, bar.top() + 2))

        x0 = self.BAR + 10
        width = w - x0 - 2
        if not has_data:
            self._text(p, QRectF(x0, 0, width, 30), "—", 'text_faint', 22, bold=True)
            if h >= 48:
                self._text(p, QRectF(x0, 30, width, 16), "waiting for frames", 'text_faint', 11)
            p.end()
            return

        # number with its unit, then "% of peak", then hsi's score
        big = 22 if h >= 56 else 16
        p.setPen(self._colour('text'))
        p.setFont(self._font(big, bold=True, mono=True))
        number = format_score(panel.score)
        p.drawText(QRectF(x0, 0, width, big + 6), Qt.AlignLeft | Qt.AlignVCenter, number)
        offset = p.fontMetrics().horizontalAdvance(number) + 4
        self._text(p, QRectF(x0 + offset, 4, width - offset, big + 6), UNIT, 'text_muted', 11)
        y = big + 6
        if h - y >= 14:
            percent = f"{100.0 * ratio:.0f} %"
            offset = self._text(p, QRectF(x0, y, width, 16), percent, colour, 11, bold=True) + 4
            self._text(p, QRectF(x0 + offset, y, width - offset, 16), f"of peak {format_score(panel.peak)}",
                       'text_muted', 11)
            y += 17
        if h - y >= 44 and panel.reference is not None:
            self._text(p, QRectF(x0, y, width, 14), f"hsi score {format_reference(panel.reference)}", 'text_faint', 10)
            y += 15

        # trend: the last scores in their own range; the dot is the best one in the window
        trend = QRectF(x0, y + 3, width, h - y - 18)
        history = panel.history
        if trend.height() >= 18 and len(history) >= 2:
            p.setPen(QPen(self._colour('chart_grid'), 1))
            p.setBrush(Qt.NoBrush)
            p.drawRect(trend)
            lo, hi = min(history), max(history)
            pad = max((hi - lo) * 0.1, abs(hi) * 1e-3, 1e-9)
            lo, hi = lo - pad, hi + pad
            start = FocusPanel.HISTORY - len(history)       # fixed time axis: new scores enter on the right

            def point(i: int, value: float) -> QPointF:
                fx = (start + i) / (FocusPanel.HISTORY - 1)
                return QPointF(trend.left() + fx * trend.width(),
                               trend.bottom() - (value - lo) / (hi - lo) * trend.height())

            path = QPainterPath(point(0, history[0]))
            for i, value in enumerate(history[1:], start=1):
                path.lineTo(point(i, value))
            p.setPen(QPen(self._colour('accent'), 1.5))
            p.drawPath(path)
            best = max(range(len(history)), key=history.__getitem__)
            p.setPen(Qt.NoPen)
            p.setBrush(self._colour('success'))
            p.drawEllipse(point(best, history[best]), 3.0, 3.0)
            footer = QRectF(x0, trend.bottom() + 2, width, 14)
            self._text(p, footer, f"last {FocusPanel.HISTORY} scores", 'text_faint', 10)
            self._text(p, footer, f"● best {format_score(history[best])}", 'success', 10, align=Qt.AlignRight)
        p.end()
