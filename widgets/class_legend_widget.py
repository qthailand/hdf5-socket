# -*- coding: utf-8 -*-
"""
Material analytics for the industrial optical sorter: class shares as a horizontal bar chart.
Author: TSC Pipeline Core Team

One chart compares every class at a glance: per class a real-time bar (lines classified since
the previous refresh) and an accumulated bar (since the batch reset), both in the class colour.
Below it a compact key gives the pixel counts, the colour picker and a show/hide switch per
class. Shares are computed over the shown classes, so hiding the background (hidden by
default, see ``HIDDEN_BY_DEFAULT``) compares the materials with each other.
Appearance comes from the application stylesheet (``themes/app.qss``) via object names; only
the class colours are set here, because they are data, not theme.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QScrollArea,
    QGridLayout, QPushButton, QColorDialog, QCheckBox
)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np

from themes import make_panel
from widgets.performance_panel import HBarChart

#: Opacity of the accumulated bar (the real-time bar is the full class colour).
ACCUMULATED_ALPHA = 110
#: Class names (case-insensitive) hidden from the chart until the operator shows them.
HIDDEN_BY_DEFAULT = ('bg', 'background')


class ClassLegendWidget(QWidget):
    """Modern Stat Cards Dashboard Widget displaying class metrics and interactive palette controls."""

    # Signal emitted when class color changes: (class_id, [R, G, B])
    color_changed = pyqtSignal(int, list)
    # The operator started a new batch; the owner resets the pipeline's counters.
    reset_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.cards_map = {}  # {class_id: (badge_btn, chart row, accumulated label)}
        self._rows = {}      # {chart row: (name label, accumulated label)}
        self._current_lut = None
        self.init_ui()

    def init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # -------------------------------------------------------------
        # 1. Header & Reset Button
        # -------------------------------------------------------------
        header_layout = QHBoxLayout()
        title = QLabel("Material Analytics")
        title.setObjectName("panelTitle")
        header_layout.addWidget(title)

        self.reset_btn = QPushButton("Reset Batch")
        self.reset_btn.setObjectName("dangerButton")
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        self.reset_btn.clicked.connect(self._on_reset_clicked)
        header_layout.addWidget(self.reset_btn, alignment=Qt.AlignRight)
        main_layout.addLayout(header_layout)

        # -------------------------------------------------------------
        # 2. Top Summary Stat Cards (Total Scanned)
        # -------------------------------------------------------------
        summary_card = QFrame()
        summary_card.setObjectName("summaryCard")
        make_panel(summary_card, accent='info', alt=True)
        summary_layout = QVBoxLayout(summary_card)
        summary_layout.setContentsMargins(12, 10, 12, 10)

        sum_title = QLabel("TOTAL SCANNED VOLUME")
        sum_title.setObjectName("caption")

        self.lbl_total_pixels = QLabel("0 px")
        self.lbl_total_pixels.setObjectName("bigValue")

        summary_layout.addWidget(sum_title)
        summary_layout.addWidget(self.lbl_total_pixels)
        main_layout.addWidget(summary_card)

        # -------------------------------------------------------------
        # 3. Class share chart + key (badge, name, accumulated pixels)
        # -------------------------------------------------------------
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)

        self.container_widget = QWidget()
        content = QVBoxLayout(self.container_widget)
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(12)

        self.chart = HBarChart("Class share", "% of pixels", label_all=True, suffix='%', axis_cap=100.0)
        self.chart.BAR, self.chart.ROW_GAP = 12, 12          # few rows: thicker bars than the stage charts
        self.chart.setToolTip("Per class: real-time share (lines since the previous refresh, solid) and "
                              "accumulated share since the batch reset (faded)")
        content.addWidget(self.chart)

        key_panel = QFrame()
        key_panel.setObjectName("classKey")
        make_panel(key_panel, alt=True)
        self.grid_layout = QGridLayout(key_panel)
        self.grid_layout.setContentsMargins(10, 8, 10, 8)
        self.grid_layout.setHorizontalSpacing(8)
        self.grid_layout.setVerticalSpacing(6)
        self.grid_layout.setColumnStretch(2, 1)
        content.addWidget(key_panel)
        content.addStretch()

        self.scroll_area.setWidget(self.container_widget)
        main_layout.addWidget(self.scroll_area)

    def setup_classes_from_model(self, category_names: dict[int, str], lut: np.ndarray, classes: np.ndarray,
                                 hidden=HIDDEN_BY_DEFAULT) -> None:
        """Builds the chart rows and the key from the model classes.

        ``hidden``: class names (case-insensitive) shown unticked, i.e. left out of the chart.
        """
        hidden = {str(name).casefold() for name in hidden}
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        self.cards_map.clear()
        self._rows.clear()
        self._current_lut = lut.copy()
        self._names = []
        self._recent_px = np.zeros(len(classes))      # pixels per class since the previous refresh
        self._total_px = np.zeros(len(classes))       # ... and since the batch reset
        self._shown = np.ones(len(classes), bool)

        for idx, class_id in enumerate(classes):
            class_id = int(class_id)
            class_name = category_names.get(class_id, f"Class {class_id}")
            self._names.append(class_name)

            badge_btn = QPushButton()
            badge_btn.setObjectName("classBadge")
            badge_btn.setFixedSize(14, 14)
            badge_btn.setCursor(Qt.PointingHandCursor)
            badge_btn.setToolTip("Click to change color palette")
            self._update_badge_style(badge_btn, self._current_lut[class_id])
            badge_btn.clicked.connect(lambda _, cid=class_id,
                                      btn=badge_btn: self._pick_new_color(cid, btn))

            name_lbl = QLabel(class_name)
            name_lbl.setObjectName("className")
            accum_val = QLabel("0 px")
            accum_val.setObjectName("metricTotal")
            accum_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            show = QCheckBox()
            show.setObjectName("stepCheck")
            show.setCursor(Qt.PointingHandCursor)
            show.setToolTip(f"Show {class_name} in the chart (shares are over the shown classes)")
            show.setChecked(class_name.casefold() not in hidden)
            show.toggled.connect(lambda on, i=idx: self._set_shown(i, on))

            self.grid_layout.addWidget(show, idx, 0)
            self.grid_layout.addWidget(badge_btn, idx, 1)
            self.grid_layout.addWidget(name_lbl, idx, 2)
            self.grid_layout.addWidget(accum_val, idx, 3)
            self.cards_map[class_id] = (badge_btn, idx, accum_val)
            self._rows[idx] = (name_lbl, accum_val)
            self._set_shown(idx, show.isChecked(), redraw=False)
        self._redraw_chart()

    def _set_shown(self, idx: int, shown: bool, redraw: bool = True) -> None:
        self._shown[idx] = shown
        for label in self._rows[idx]:
            label.setEnabled(shown)                  # hidden: dimmed in the key
        if redraw:
            self._redraw_chart()

    def shares(self) -> tuple[list[str], np.ndarray, np.ndarray]:
        """``(names, real-time %, accumulated %)`` of the shown classes, over the shown classes."""
        def percent(px):
            px = px[self._shown]
            return px / px.sum() * 100.0 if px.sum() else np.zeros_like(px)
        names = [n for n, on in zip(self._names, self._shown) if on]
        return names, percent(self._recent_px), percent(self._total_px)

    def _redraw_chart(self) -> None:
        names, realtime, accumulated = self.shares()
        colours = [QColor(*map(int, self._current_lut[cid]))
                   for cid, on in zip(self.cards_map, self._shown) if on]
        faded = [QColor(c) for c in colours]
        for c in faded:
            c.setAlpha(ACCUMULATED_ALPHA)
        self.chart.unit = "% of pixels" if self._shown.all() else "% of shown classes"
        self.chart.set_data(names, [("Real-time", colours, list(realtime)),
                                    ("Accumulated", faded, list(accumulated))])

    @staticmethod
    def _update_badge_style(button: QPushButton, rgb: list[int] | np.ndarray) -> None:
        """Only the class colour; border/hover come from ``#classBadge`` in the theme."""
        button.setStyleSheet(f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]});")

    def _pick_new_color(self, class_id: int, button: QPushButton) -> None:
        current_rgb = self._current_lut[class_id]
        initial_color = QColor(current_rgb[0], current_rgb[1], current_rgb[2])

        selected_color = QColorDialog.getColor(
            initial_color, self, f"Select Color for Class ID {class_id}")
        if selected_color.isValid():
            new_rgb = [selected_color.red(), selected_color.green(), selected_color.blue()]
            self._current_lut[class_id] = new_rgb
            self._update_badge_style(button, new_rgb)
            self._redraw_chart()

            self.color_changed.emit(class_id, new_rgb)

    def update_counts(self, recent_counts: np.ndarray, total_counts: np.ndarray, total_lines: int) -> None:
        """Show class shares.

        Args:
            recent_counts: ``(256,)`` pixels per class id since the previous refresh (every line
                classified in that interval, not just the newest one).
            total_counts: ``(256,)`` pixels per class id since the last batch reset.
            total_lines: Lines included in ``total_counts``.
        """
        if not self.cards_map:
            return
        recent_total = int(recent_counts.sum())
        total = int(total_counts.sum())
        self.lbl_total_pixels.setText(f"{total:,} px · {total_lines:,} lines")
        for class_id, (_, idx, accum_val) in self.cards_map.items():
            if recent_total:          # keep the last value while no line arrives
                self._recent_px[idx] = recent_counts[class_id]
            self._total_px[idx] = total_counts[class_id]
            accum_val.setText(f"{int(total_counts[class_id]):,} px")
        self._redraw_chart()

    def _on_reset_clicked(self) -> None:
        self.reset_requested.emit()
        self.reset_accumulated_stats()

    def reset_accumulated_stats(self) -> None:
        """Clears the displayed metrics (the counts themselves live in the pipeline)."""
        self.lbl_total_pixels.setText("0 px")
        for _, (_, idx, accum_val) in self.cards_map.items():
            accum_val.setText("0 px")
        if self.cards_map:
            self._recent_px[:] = 0.0
            self._total_px[:] = 0.0
            self._redraw_chart()
