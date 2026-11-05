# -*- coding: utf-8 -*-
"""
Composition statistics: what percentage of the material each class makes up.
Author: TSC Pipeline Core Team

The numbers come from :class:`widgets.class_legend_widget.ClassLegendWidget` (signal
``shares_changed``), so they always match the Materials chart: same shown classes, same
basis (``% of pixels`` or ``% of shown classes``).

* a 100 % stacked bar of the accumulated composition (since the batch reset);
* per class, largest first: the accumulated share in large type, the real-time share
  (lines since the previous refresh) and the pixel count behind the share;
* a donut (pie) chart of the same shares, accumulated or real-time (switch above it);
  hovering a slice shows its class, share and pixels;
* *Copy* puts the table on the clipboard as tab-separated text (pastes into Excel).
"""

import math

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout, QPushButton,
    QScrollArea, QSizePolicy, QToolTip, QButtonGroup
)
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QFont, QFontMetrics
from PyQt5.QtCore import Qt, QRectF, QPointF

from themes import make_panel


def _theme_colour(token: str) -> QColor:
    tokens = QApplication.instance().property('theme_tokens') or {}
    return QColor(tokens.get(token) or 'gray')


def _ink_on(colour: QColor) -> QColor:
    """Black or white text, whichever reads better on ``colour``."""
    luminance = 0.299 * colour.red() + 0.587 * colour.green() + 0.114 * colour.blue()
    return QColor(Qt.black) if luminance > 150 else QColor(Qt.white)


class StackedShareBar(QWidget):
    """One horizontal bar split into the class shares (they add up to 100 %)."""

    HEIGHT = 26

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._segments = []          # [(name, QColor, percent)]
        self.setMinimumHeight(self.HEIGHT)
        self.setMaximumHeight(self.HEIGHT)

    def set_segments(self, segments) -> None:
        self._segments = [(n, QColor(c), float(p)) for n, c, p in segments]
        self.setToolTip("\n".join(f"{n}: {p:.1f}%" for n, _, p in self._segments))
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(Qt.NoPen)
        p.setBrush(_theme_colour('chart_grid'))
        p.drawRoundedRect(rect, 5, 5)

        total = sum(pct for _, _, pct in self._segments)
        if total > 0:
            font = QFont(self.font())
            font.setPointSizeF(8.5)
            font.setBold(True)
            p.setFont(font)
            metrics = QFontMetrics(font)
            x = rect.left()
            p.save()
            path = QPainterPath()
            path.addRoundedRect(rect, 5, 5)
            p.setClipPath(path)
            for _, colour, pct in self._segments:
                width = rect.width() * pct / total
                if width <= 0:
                    continue
                seg = QRectF(x, rect.top(), width, rect.height())
                p.fillRect(seg, colour)
                text = f"{pct:.1f}%"
                if metrics.horizontalAdvance(text) + 6 <= width:
                    p.setPen(_ink_on(colour))
                    p.drawText(seg, Qt.AlignCenter, text)
                    p.setPen(Qt.NoPen)
                x += width
            p.restore()
        p.end()


class DonutChart(QWidget):
    """Pie chart with a hole: one slice per class, clockwise from 12 o'clock, in the given order.

    The share is written on every slice wide enough for it; the centre shows the first
    (largest) slice, or the hovered one. Hovering outlines a slice and shows a tooltip.
    """

    HOLE = 0.55          # inner radius / outer radius
    MIN_LABEL_DEG = 16   # narrower slices get no label (hover still tells)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._slices = []    # [(name, QColor, percent, pixels)]
        self._hover = None
        self.setMouseTracking(True)
        self.setMinimumSize(180, 180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_slices(self, slices) -> None:
        self._slices = [(n, QColor(c), float(pct), int(px)) for n, c, pct, px in slices if pct > 0]
        if self._hover is not None and self._hover >= len(self._slices):
            self._hover = None
        self.update()

    def slices(self) -> list[tuple]:
        """``[(name, percent, pixels)]`` as drawn."""
        return [(n, pct, px) for n, _, pct, px in self._slices]

    def _geometry(self):
        side = min(self.width(), self.height()) - 8
        centre = QPointF(self.width() / 2, self.height() / 2)
        return centre, max(side, 0) / 2

    def _spans(self):
        """``(start, span)`` in degrees, clockwise from 12 o'clock, per slice."""
        total = sum(pct for _, _, pct, _ in self._slices)
        start, spans = 0.0, []
        for _, _, pct, _ in self._slices:
            span = 360.0 * pct / total if total else 0.0
            spans.append((start, span))
            start += span
        return spans

    def slice_at(self, pos) -> int | None:
        """Index of the slice under ``pos`` (widget coordinates), or None."""
        centre, radius = self._geometry()
        dx, dy = pos.x() - centre.x(), pos.y() - centre.y()
        dist = math.hypot(dx, dy)
        if not self._slices or dist > radius or dist < radius * self.HOLE:
            return None
        angle = math.degrees(math.atan2(dx, -dy)) % 360.0   # clockwise from 12 o'clock
        for i, (start, span) in enumerate(self._spans()):
            if start <= angle < start + span:
                return i
        return len(self._slices) - 1

    def mouseMoveEvent(self, event) -> None:
        hover = self.slice_at(event.pos())
        if hover != self._hover:
            self._hover = hover
            self.update()
        if hover is None:
            QToolTip.hideText()
        else:
            name, _, pct, px = self._slices[hover]
            QToolTip.showText(event.globalPos(), f"{name}: {pct:.1f}%  ({px:,} px)", self)

    def leaveEvent(self, event) -> None:
        self._hover = None
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        centre, radius = self._geometry()
        outer = QRectF(centre.x() - radius, centre.y() - radius, 2 * radius, 2 * radius)
        inner_r = radius * self.HOLE
        inner = QRectF(centre.x() - inner_r, centre.y() - inner_r, 2 * inner_r, 2 * inner_r)

        if not self._slices:
            ring = QPainterPath()
            ring.addEllipse(outer)
            ring.addEllipse(inner)
            p.fillPath(ring, _theme_colour('chart_grid'))
            p.setPen(_theme_colour('text_faint'))
            p.drawText(outer, Qt.AlignCenter, "No data")
            p.end()
            return

        font = QFont(self.font())
        font.setBold(True)
        font.setPointSizeF(8.5)
        p.setFont(font)
        gap = _theme_colour('surface_alt')
        for i, ((_, colour, pct, _), (start, span)) in enumerate(zip(self._slices, self._spans())):
            # Qt angles: counter-clockwise from 3 o'clock.
            qt_start, qt_span = 90.0 - start, -span
            path = QPainterPath()
            path.arcMoveTo(outer, qt_start)
            path.arcTo(outer, qt_start, qt_span)
            path.arcTo(inner, qt_start + qt_span, -qt_span)
            path.closeSubpath()
            p.setPen(gap if len(self._slices) > 1 else Qt.NoPen)
            p.setBrush(colour.lighter(125) if i == self._hover else colour)
            p.drawPath(path)
            if span >= self.MIN_LABEL_DEG:
                mid = math.radians(start + span / 2)
                r = (radius + inner_r) / 2
                at = QPointF(centre.x() + r * math.sin(mid), centre.y() - r * math.cos(mid))
                p.setPen(_ink_on(colour))
                p.drawText(QRectF(at.x() - 30, at.y() - 9, 60, 18), Qt.AlignCenter, f"{pct:.1f}%")

        # Centre: the hovered slice, else the first (largest) one.
        name, _, pct, _ = self._slices[self._hover if self._hover is not None else 0]
        big = QFont(self.font())
        big.setBold(True)
        big.setPointSizeF(max(9.0, inner_r / 4.5))
        p.setFont(big)
        p.setPen(_theme_colour('text'))
        p.drawText(QRectF(inner.left(), centre.y() - inner_r * 0.5, inner.width(), inner_r * 0.6),
                   Qt.AlignHCenter | Qt.AlignBottom, f"{pct:.1f}%")
        small = QFont(self.font())
        small.setPointSizeF(max(7.5, inner_r / 8.0))
        p.setFont(small)
        p.setPen(_theme_colour('text_muted'))
        text = QFontMetrics(small).elidedText(name, Qt.ElideRight, int(inner.width() * 0.8))
        p.drawText(QRectF(inner.left(), centre.y() + inner_r * 0.12, inner.width(), inner_r * 0.4),
                   Qt.AlignHCenter | Qt.AlignTop, text)
        p.end()


class CompositionWidget(QWidget):
    """Percentage of each class in the scanned material (accumulated and real-time)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rows = []       # [(name, QColor, realtime %, accumulated %, pixels)] as shown
        self._widgets = []    # per table row: (swatch, name, share, realtime, pixels)
        self.init_ui()

    def init_ui(self) -> None:
        main = QVBoxLayout(self)
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("Composition")
        title.setObjectName("panelTitle")
        header.addWidget(title)
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setCursor(Qt.PointingHandCursor)
        self.copy_btn.setToolTip("Copy the table (tab-separated) to the clipboard, e.g. to paste into Excel")
        self.copy_btn.clicked.connect(self.copy_to_clipboard)
        header.addWidget(self.copy_btn, alignment=Qt.AlignRight)
        main.addLayout(header)

        summary = QFrame()
        make_panel(summary, accent='info', alt=True)
        s_layout = QVBoxLayout(summary)
        s_layout.setContentsMargins(12, 10, 12, 10)
        s_layout.setSpacing(6)
        self.lbl_basis = QLabel("ACCUMULATED · % OF PIXELS")
        self.lbl_basis.setObjectName("caption")
        self.bar = StackedShareBar()
        self.lbl_total = QLabel("0 px")
        self.lbl_total.setObjectName("metricCaption")
        s_layout.addWidget(self.lbl_basis)
        s_layout.addWidget(self.bar)
        s_layout.addWidget(self.lbl_total)
        main.addWidget(summary)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        table = QFrame()
        table.setObjectName("compositionTable")
        make_panel(table, alt=True)
        self.grid = QGridLayout(table)
        self.grid.setContentsMargins(10, 8, 10, 8)
        self.grid.setHorizontalSpacing(10)
        self.grid.setVerticalSpacing(6)
        self.grid.setColumnStretch(1, 1)
        for col, (text, tip) in enumerate((
                ("", ""),
                ("CLASS", ""),
                ("ACCUM.", "Share since the batch reset"),
                ("REAL-TIME", "Share of the lines since the previous refresh"),
                ("PIXELS", "Accumulated pixels behind the share"))):
            if not text:
                continue
            lbl = QLabel(text)
            lbl.setObjectName("caption")
            lbl.setToolTip(tip)
            if col >= 2:
                lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.grid.addWidget(lbl, 0, col)
        container = QWidget()
        c_layout = QVBoxLayout(container)
        c_layout.setContentsMargins(0, 0, 0, 0)
        c_layout.setSpacing(12)
        c_layout.addWidget(table)

        pie_panel = QFrame()
        pie_panel.setObjectName("compositionPie")
        make_panel(pie_panel, alt=True)
        p_layout = QVBoxLayout(pie_panel)
        p_layout.setContentsMargins(10, 8, 10, 10)
        p_layout.setSpacing(6)
        pie_header = QHBoxLayout()
        pie_title = QLabel("SHARE")
        pie_title.setObjectName("caption")
        pie_header.addWidget(pie_title)
        pie_header.addStretch()
        self._pie_mode = QButtonGroup(self)
        for i, (text, tip) in enumerate((("Accumulated", "Shares since the batch reset"),
                                         ("Real-time", "Shares of the lines since the previous refresh"))):
            btn = QPushButton(text)
            btn.setObjectName("segmentButton")
            btn.setCheckable(True)
            btn.setChecked(i == 0)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(tip)
            self._pie_mode.addButton(btn, i)
            pie_header.addWidget(btn)
        self._pie_mode.buttonClicked.connect(lambda _: self._update_pie())
        p_layout.addLayout(pie_header)
        self.pie = DonutChart()
        p_layout.addWidget(self.pie, stretch=1)
        c_layout.addWidget(pie_panel, stretch=1)
        scroll.setWidget(container)
        main.addWidget(scroll)

    # ------------------------------------------------------------------ data
    def set_shares(self, unit: str, rows) -> None:
        """``rows``: ``[(name, QColor, realtime %, accumulated %, accumulated pixels)]``.

        Shown largest accumulated share first.
        """
        rows = [(n, QColor(c), float(rt), float(acc), int(px)) for n, c, rt, acc, px in rows]
        rows.sort(key=lambda r: -r[3])
        self._rows = rows
        self._ensure_rows(len(rows))

        total_px = sum(r[4] for r in rows)
        self.lbl_basis.setText(f"ACCUMULATED · {unit.upper()}")
        self.lbl_total.setText(f"{total_px:,} px counted since the batch reset")
        self.bar.set_segments([(n, c, acc) for n, c, _, acc, _ in rows])
        self._update_pie()

        for (name, colour, rt, acc, px), (swatch, name_lbl, share, realtime, pixels) in zip(rows, self._widgets):
            swatch.setStyleSheet(f"background-color: rgb({colour.red()}, {colour.green()}, {colour.blue()});")
            name_lbl.setText(name)
            share.setText(f"{acc:.1f}%")
            realtime.setText(f"{rt:.1f}%")
            pixels.setText(f"{px:,}")

    def pie_realtime(self) -> bool:
        """True while the pie shows real-time shares (else accumulated)."""
        return self._pie_mode.checkedId() == 1

    def _update_pie(self) -> None:
        """Slices largest first in the chosen mode (pixels: accumulated, as in the table)."""
        if self.pie_realtime():
            slices = sorted(((n, c, rt, px) for n, c, rt, _, px in self._rows), key=lambda r: -r[2])
        else:
            slices = [(n, c, acc, px) for n, c, _, acc, px in self._rows]
        self.pie.set_slices(slices)

    def rows(self) -> list[tuple]:
        """``[(name, realtime %, accumulated %, pixels)]`` as displayed (largest first)."""
        return [(n, rt, acc, px) for n, _, rt, acc, px in self._rows]

    def as_text(self) -> str:
        lines = ["Class\tAccumulated %\tReal-time %\tPixels"]
        lines += [f"{n}\t{acc:.2f}\t{rt:.2f}\t{px}" for n, rt, acc, px in self.rows()]
        return "\n".join(lines)

    def copy_to_clipboard(self) -> None:
        QApplication.clipboard().setText(self.as_text())

    def _ensure_rows(self, n: int) -> None:
        while len(self._widgets) > n:
            for w in self._widgets.pop():
                w.setParent(None)
        while len(self._widgets) < n:
            r = len(self._widgets) + 1
            swatch = QLabel()
            swatch.setObjectName("classSwatch")
            swatch.setFixedSize(12, 12)
            name = QLabel()
            name.setObjectName("className")
            share = QLabel()
            share.setObjectName("shareValue")
            realtime = QLabel()
            realtime.setObjectName("metricRealtime")
            pixels = QLabel()
            pixels.setObjectName("metricCaption")
            for col, w in enumerate((swatch, name, share, realtime, pixels)):
                if col >= 2:
                    w.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.grid.addWidget(w, r, col, alignment=Qt.AlignVCenter if col == 0 else Qt.Alignment())
            self._widgets.append((swatch, name, share, realtime, pixels))
