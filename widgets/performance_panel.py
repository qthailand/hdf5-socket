# -*- coding: utf-8 -*-
"""Real-time pipeline performance: horizontal grouped bar charts of speed, not accuracy.

What a sorting system must guarantee is that every line is processed in time, so the panel
answers three questions while the simulation runs:

* **Time per line, by stage** (ms per line per worker): dark + hot pixel,
  flip + Smile, binning, model, total — "now" (last ~0.5 s), the average for every backend
  that ran since the last reset (CPU vs CuPy side by side) and the peak. The red marker is the
  **budget**: ``workers × 1000 / line rate``; a total above it means the chain cannot keep up.
* **Throughput** (lines/s): target, camera, processed and the **capacity** the measured
  stage times allow (``workers × 1000 / total``).
* **Latency** (ms, sensor time → line counted): queue wait (time a line sat in the buffers
  before a worker took it), p50, p99, max. Queue wait growing = the chain falls behind.

Colours are theme tokens (``chart_*`` in ``themes/<name>.json``); nothing is hard-coded here.
"""
from __future__ import annotations

import math

from PyQt5.QtCore import QLineF, QRectF, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PyQt5.QtWidgets import (QApplication, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
                             QSizePolicy, QVBoxLayout, QWidget)

from themes import make_panel, set_accent

#: Display order and names of the profiler stages (others are appended as they come).
STAGES = (('dark_hot', 'Dark + Hot pixel'),
          ('orient_smile', 'Flip + Smile'), ('gpu_preprocess', 'GPU correction'), ('bin', 'Binning'),
          ('reflectance', 'Reflectance'), ('predict', 'Model'), ('total', 'Total / line'))
_HIDDEN = {'dark', 'hot_pixel',     # shown together as one step, like the step switch
           'preprocess',            # sum of dark/hot/flip/Smile on CPU; shown only without them
           'queue'}                 # waiting, not processing: shown in the latency chart


_TOKENS: dict = {}


def _token(name: str, fallback: str = 'text') -> QColor:
    """Theme colour; the token table is read once per paint (see ``HBarChart.paintEvent``)."""
    tokens = _TOKENS or QApplication.instance().property('theme_tokens') or {}
    return QColor(tokens.get(name) or tokens.get(fallback) or 'gray')


def _series_colour(colour, category: int | None) -> QColor:
    """A series' colour: a theme token, or per-category colours (legend: neutral, same alpha)."""
    if isinstance(colour, str):
        return _token(colour, 'accent')
    if category is not None and category < len(colour):
        return QColor(colour[category])
    swatch = _token('text_muted')
    swatch.setAlpha(QColor(colour[0]).alpha() if colour else 255)
    return swatch


def _nice_step(span: float, ticks: int = 4) -> float:
    raw = span / max(1, ticks)
    magnitude = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    for factor in (1, 2, 2.5, 5, 10):
        if raw <= factor * magnitude:
            return factor * magnitude
    return 10 * magnitude


def _fmt(value: float) -> str:
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


class HBarChart(QWidget):
    """Grouped horizontal bar chart (categories down, one bar per series), painted directly.

    ``set_data(categories, series, markers)``:
        categories: row labels.
        series: ``[(name, colour, [value or None per category])]``; ``colour`` is a theme token,
            or a list of ``QColor`` (one per category: data colours such as class colours).
        markers: ``[(label, value, colour_token)]`` vertical lines (e.g. the time budget).
    The axis eases towards the new maximum, so it does not jump on every refresh.
    ``label_all`` writes the value of every series (default: the first only), ``suffix`` is
    appended to the values and ``axis_cap`` limits the axis (e.g. 100 for percentages).
    """

    BAR = 8
    BAR_GAP = 2
    ROW_GAP = 9

    def __init__(self, title: str, unit: str, parent=None, label_all: bool = False, suffix: str = '',
                 axis_cap: float | None = None) -> None:
        super().__init__(parent)
        self.title, self.unit = title, unit
        self.label_all, self.suffix, self.axis_cap = label_all, suffix, axis_cap
        self._categories: list[str] = []
        self._series: list = []
        self._markers: list = []
        self._axis_max = 0.0
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_data(self, categories, series, markers=()) -> None:
        changed = len(categories) != len(self._categories) or len(series) != len(self._series)
        self._categories, self._series, self._markers = list(categories), list(series), list(markers)
        peak = max([v for _, _, values in series for v in values if v is not None]
                   + [v for _, v, _ in markers if v is not None] + [1e-6])
        target = peak * 1.12 if self.axis_cap is None else min(peak * 1.12, self.axis_cap)
        if self._axis_max <= 0 or target > self._axis_max or target < self._axis_max * 0.6:
            self._axis_max = target if self._axis_max <= 0 or target > self._axis_max \
                else self._axis_max * 0.85 + target * 0.15
        if changed:
            self.updateGeometry()
        self.update()

    def _font(self, size: int, bold: bool = False) -> QFont:
        font = QFont(self.font())
        font.setPixelSize(size)
        font.setBold(bold)
        return font

    def _bar(self) -> int:
        return self.BAR if len(self._series) > 1 else self.BAR + 4      # single series: thicker

    def _row_height(self) -> int:
        n = max(1, len(self._series))
        return n * self._bar() + (n - 1) * self.BAR_GAP + self.ROW_GAP

    def sizeHint(self):
        legend = 18 if len(self._series) > 1 or self._markers else 0
        return QSize(260, 24 + len(self._categories) * self._row_height() + 20 + legend)

    def minimumSizeHint(self):
        return self.sizeHint()

    def paintEvent(self, event) -> None:
        _TOKENS.clear()
        _TOKENS.update(QApplication.instance().property('theme_tokens') or {})
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        text, faint, grid = _token('text'), _token('text_faint'), _token('chart_grid', 'border')
        w = self.width()

        p.setFont(self._font(12, True))
        p.setPen(text)
        p.drawText(0, 0, w, 18, Qt.AlignLeft | Qt.AlignVCenter, self.title)
        p.setFont(self._font(10))
        p.setPen(faint)
        p.drawText(0, 0, w, 18, Qt.AlignRight | Qt.AlignVCenter, self.unit)

        label_font = self._font(11)
        fm = QFontMetrics(label_font)
        label_w = max([fm.horizontalAdvance(c) for c in self._categories] + [40]) + 8
        value_w = QFontMetrics(self._font(10)).horizontalAdvance("8,888" + self.suffix) + 6
        x0, x1 = label_w, max(label_w + 40, w - value_w)
        top = 24
        row_h = self._row_height()
        bottom = top + len(self._categories) * row_h
        vmax = self._axis_max or 1.0

        def x_of(value):
            return x0 + (x1 - x0) * min(value, vmax) / vmax

        # grid + axis labels
        step = _nice_step(vmax)
        decimals = 0 if step >= 1 else (1 if step >= 0.1 else 2)
        p.setFont(self._font(10))
        tick = 0.0
        while tick <= vmax + 1e-9:
            x = x_of(tick)
            p.setPen(QPen(grid, 1))
            p.drawLine(int(x), top - 2, int(x), bottom)
            p.setPen(faint)
            p.drawText(QRectF(x - 30, bottom + 2, 60, 14), Qt.AlignHCenter | Qt.AlignTop,
                       f"{tick:,.{decimals}f}")
            tick += step

        # bars
        for i, category in enumerate(self._categories):
            y = top + i * row_h
            p.setFont(label_font)
            p.setPen(text)
            group_h = row_h - self.ROW_GAP
            p.drawText(QRectF(0, y, label_w - 8, group_h), Qt.AlignRight | Qt.AlignVCenter, category)
            bar = self._bar()
            for j, (_, colour, values) in enumerate(self._series):
                value = values[i] if i < len(values) else None
                by = y + j * (bar + self.BAR_GAP)
                if value is None:
                    continue
                p.setPen(Qt.NoPen)
                p.setBrush(_series_colour(colour, i))
                p.drawRoundedRect(QRectF(x0, by, max(1.0, x_of(value) - x0), bar), 2, 2)
                if j == 0 or self.label_all:                  # value of the first series by default
                    p.setFont(self._font(10))
                    p.setPen(faint)
                    p.drawText(QRectF(x_of(value) + 4, by - 3, value_w + 20, bar + 6),
                               Qt.AlignLeft | Qt.AlignVCenter, _fmt(value) + self.suffix)

        # markers (budget / target)
        for label, value, colour in self._markers:
            if value is None or value <= 0:
                continue
            x = x_of(value)
            p.setPen(QPen(_token(colour, 'danger'), 1.5, Qt.DashLine))
            p.drawLine(QLineF(x, top - 4, x, bottom))

        # legend
        if len(self._series) > 1 or self._markers:
            p.setFont(self._font(10))
            x, y = 0.0, bottom + 18
            items = [(name, colour, False) for name, colour, _ in self._series]
            items += [(f"{label} {_fmt(value)}", colour, True) for label, value, colour in self._markers
                      if value]
            for name, colour, dashed in items:
                if dashed:
                    p.setPen(QPen(_token(colour, 'danger'), 1.5, Qt.DashLine))
                    p.drawLine(int(x), int(y + 7), int(x + 10), int(y + 7))
                else:
                    p.setPen(Qt.NoPen)
                    p.setBrush(_series_colour(colour, None))
                    p.drawRect(QRectF(x, y + 3, 9, 9))
                p.setPen(faint)
                width = QFontMetrics(p.font()).horizontalAdvance(name)
                p.drawText(QRectF(x + 13, y, width + 4, 16), Qt.AlignLeft | Qt.AlignVCenter, name)
                x += 13 + width + 12
        p.end()


class _Kpi(QFrame):
    """Small value card with the theme's accent stripe."""

    def __init__(self, caption: str, tooltip: str = '') -> None:
        super().__init__()
        make_panel(self, accent='neutral', alt=True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 8, 6)
        layout.setSpacing(0)
        title = QLabel(caption.upper())
        title.setObjectName("caption")
        self.value = QLabel("–")
        self.value.setObjectName("kpiValue")
        layout.addWidget(title)
        layout.addWidget(self.value)
        if tooltip:
            self.setToolTip(tooltip)

    def show_value(self, text: str, accent: str) -> None:
        self.value.setText(text)
        set_accent(self, accent)


KpiCard = _Kpi


class ThroughputStrip(QWidget):
    """Always-visible throughput row: camera/processed lines per second, capacity, lost, latency.

    Shown under the belt map whatever side tab is open, refreshed with :meth:`update_snapshot`.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self.camera = KpiCard("Camera FPS", "Lines per second the simulated camera delivered (target in brackets)")
        self.processed = KpiCard("Processed FPS", "Lines per second classified, every line counted in order")
        self.capacity = KpiCard("Capacity (est.)", "workers x 1000 / measured ms per line: what the chain could take;\n"
                                                   "optimistic on CPU (workers share cores) - the measured maximum is "
                                                   "simulate.py --fps 0 --overrun block")
        self.lost = KpiCard("Lost lines", "Lines the camera exposed with no free buffer (overrun)")
        self.latency = KpiCard("Latency p50 / p99", "Sensor time to line counted, ms")
        for card in (self.camera, self.processed, self.capacity, self.lost, self.latency):
            row.addWidget(card, stretch=1)

    def update_snapshot(self, s) -> None:
        target = s.target_fps
        self.camera.show_value(f"{s.camera_fps:,.0f}" + (f"  / {target:g}" if target else "  (max)"),
                               'ok' if not target or s.camera_fps >= 0.98 * target else 'warn')
        behind = target and s.processed_fps < 0.98 * target and s.state == 'running'
        self.processed.show_value(f"{s.processed_fps:,.0f}", 'warn' if behind else 'ok')
        total = s.stage_ms.get('total')
        if total:
            capacity = s.workers * 1000.0 / total
            headroom = capacity / target if target else None
            self.capacity.show_value(f"{capacity:,.0f}" + (f"  ({headroom:.1f}×)" if headroom else ""),
                                     'error' if headroom and headroom < 1 else
                                     'warn' if headroom and headroom < 1.3 else 'info')
        delivered = s.frames_delivered + s.lines_lost
        pct = 100.0 * s.lines_lost / delivered if delivered else 0.0
        self.lost.show_value(f"{s.lines_lost:,}  ({pct:.2f} %)", 'error' if s.lines_lost else 'ok')
        lat = s.latency_ms
        self.latency.show_value(f"{lat['p50']:.1f} / {lat['p99']:.1f} ms", 'info')


class PerformancePanel(QWidget):
    """Side panel: KPIs + three charts, fed with :meth:`update_snapshot` from the GUI timer."""

    reset_requested = pyqtSignal()
    #: series colour tokens: now, then one per backend average, then peak
    _NOW, _AVG, _PEAK = 'chart_1', ('chart_2', 'chart_4', 'chart_5'), 'chart_3'

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("Pipeline Performance")
        title.setObjectName("panelTitle")
        header.addWidget(title)
        reset = QPushButton("Reset")
        reset.setToolTip("Restart averages and peaks (e.g. after changing backend or workers)")
        reset.clicked.connect(self.reset_requested.emit)
        header.addWidget(reset, alignment=Qt.AlignRight)
        outer.addLayout(header)

        kpis = QGridLayout()
        kpis.setSpacing(8)
        self.kpi_util = _Kpi("Utilisation")
        self.kpi_headroom = _Kpi("Headroom")
        self.kpi_lost = _Kpi("Lost lines")
        self.kpi_p99 = _Kpi("Latency p99")
        for i, kpi in enumerate((self.kpi_util, self.kpi_headroom, self.kpi_lost, self.kpi_p99)):
            kpis.addWidget(kpi, i // 2, i % 2)
        outer.addLayout(kpis)

        body = QWidget()
        charts = QVBoxLayout(body)
        charts.setContentsMargins(0, 0, 4, 0)
        charts.setSpacing(14)
        self.stage_chart = HBarChart("Time per line", "ms / worker")
        self.rate_chart = HBarChart("Throughput", "lines / s")
        self.rate_chart.setToolTip("Capacity (est.) = workers × 1000 / measured total ms per line.\n"
                                   "Optimistic on CPU: workers share cores, so it does not scale linearly;\n"
                                   "measure the real maximum with: simulate.py --fps 0 --overrun block")
        self.latency_chart = HBarChart("Latency", "ms, sensor → counted")
        for chart in (self.stage_chart, self.rate_chart, self.latency_chart):
            charts.addWidget(chart)
        charts.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        outer.addWidget(scroll, stretch=1)

    @staticmethod
    def _merge(stages: dict) -> dict:
        """Dark and hot pixel are one processing step: show their sum as ``dark_hot``."""
        if 'dark' not in stages and 'hot_pixel' not in stages:
            return stages
        merged = dict(stages)
        merged['dark_hot'] = stages.get('dark', 0.0) + stages.get('hot_pixel', 0.0)
        return merged

    def update_snapshot(self, s) -> None:
        now = self._merge(s.stage_ms)
        stats = {b: {'avg': self._merge(st['avg']), 'peak': self._merge(st['peak'])}
                 for b, st in s.stage_stats.items()}
        names = [key for key, _ in STAGES if key in now or any(key in st['avg'] for st in stats.values())]
        extra = sorted({k for st in [now] + [v['avg'] for v in stats.values()] for k in st}
                       - {key for key, _ in STAGES} - _HIDDEN)
        if 'dark_hot' not in names and 'gpu_preprocess' not in names and ('preprocess' in now or any('preprocess' in v['avg'] for v in stats.values())):
            names.insert(1, 'preprocess')
        names += extra
        label = dict(STAGES, preprocess='Correction')
        categories = [label.get(n, n) for n in names]

        series = [('Now', self._NOW, [now.get(n) for n in names])]
        for i, (backend, st) in enumerate(stats.items()):
            series.append((f"Avg {backend}", self._AVG[i % len(self._AVG)], [st['avg'].get(n) for n in names]))
        current = stats.get(s.backend)
        if current:
            series.append((f"Peak {s.backend}", self._PEAK, [current['peak'].get(n) for n in names]))
        budget = s.workers * 1000.0 / s.target_fps if s.target_fps else None
        self.stage_chart.set_data(categories, series,
                                  [('budget', budget, 'chart_budget')] if budget else [])

        total = now.get('total')
        capacity = s.workers * 1000.0 / total if total else None
        rates = [s.target_fps or None, s.camera_fps, s.processed_fps, capacity]
        self.rate_chart.set_data(['Target', 'Camera', 'Processed', 'Capacity (est.)'],
                                 [('lines/s', self._NOW, rates)],
                                 [('target', s.target_fps, 'chart_budget')] if s.target_fps else [])

        lat = s.latency_ms
        self.latency_chart.set_data(['Queue wait', 'p50', 'p99', 'max'],
                                    [('ms', self._NOW, [now.get('queue'), lat['p50'], lat['p99'], lat['max']])])

        if capacity and s.processed_fps:
            util = s.processed_fps / capacity
            self.kpi_util.show_value(f"{util * 100:.0f} %", 'error' if util > 0.95 else 'warn' if util > 0.75 else 'ok')
        if capacity and s.target_fps:
            headroom = capacity / s.target_fps
            self.kpi_headroom.show_value(f"{headroom:.2f} ×",
                                         'error' if headroom < 1.0 else 'warn' if headroom < 1.3 else 'ok')
        elif capacity:
            self.kpi_headroom.show_value(f"{capacity:,.0f}/s", 'info')
        total_lines = s.frames_delivered + s.lines_lost
        lost_pct = 100.0 * s.lines_lost / total_lines if total_lines else 0.0
        self.kpi_lost.show_value(f"{s.lines_lost:,} ({lost_pct:.2f} %)", 'error' if s.lines_lost else 'ok')
        self.kpi_p99.show_value(f"{lat['p99']:.1f} ms", 'info')
