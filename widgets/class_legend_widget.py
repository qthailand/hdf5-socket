# -*- coding: utf-8 -*-
"""
Modern Dashboard Stat Cards Widget for Industrial Optical Sorter.
Author: TSC Pipeline Core Team

Appearance comes from the application stylesheet (``themes/app.qss``) via object names;
only the class colours (badge, bar) are set here, because they are data, not theme.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QScrollArea,
    QGridLayout, QPushButton, QColorDialog, QProgressBar
)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np

from themes import make_panel, set_accent_color


class ClassLegendWidget(QWidget):
    """Modern Stat Cards Dashboard Widget displaying class metrics and interactive palette controls."""

    # Signal emitted when class color changes: (class_id, [R, G, B])
    color_changed = pyqtSignal(int, list)
    # The operator started a new batch; the owner resets the pipeline's counters.
    reset_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.cards_map = {}  # {class_id: (badge_btn, rt_lbl, accum_lbl, pbar)}
        self._cards = {}     # {class_id: card frame}
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
        # 3. Scrollable Grid for Class Material Cards
        # -------------------------------------------------------------
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)

        self.container_widget = QWidget()
        self.grid_layout = QGridLayout(self.container_widget)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(10)

        self.scroll_area.setWidget(self.container_widget)
        main_layout.addWidget(self.scroll_area)

    def setup_classes_from_model(self, category_names: dict[int, str], lut: np.ndarray, classes: np.ndarray) -> None:
        """Dynamically builds material Cards from model classes."""
        # Clear existing cards
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        self.cards_map.clear()
        self._cards.clear()
        self._current_lut = lut.copy()

        for idx, class_id in enumerate(classes):
            class_id = int(class_id)
            class_name = category_names.get(class_id, f"Class {class_id}")
            rgb = self._current_lut[class_id]

            # --- Create Individual Class Card ---
            card = QFrame()
            card.setObjectName("classCard")
            make_panel(card)
            set_accent_color(card, rgb)              # stripe = class colour
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 10, 10, 10)
            card_layout.setSpacing(6)

            # Card Header: Color Badge + Name
            c_header = QHBoxLayout()
            badge_btn = QPushButton()
            badge_btn.setObjectName("classBadge")
            badge_btn.setFixedSize(16, 16)
            badge_btn.setCursor(Qt.PointingHandCursor)
            badge_btn.setToolTip("Click to change color palette")
            self._update_badge_style(badge_btn, rgb)
            badge_btn.clicked.connect(lambda _, cid=class_id,
                                      btn=badge_btn: self._pick_new_color(cid, btn))

            name_lbl = QLabel(class_name)
            name_lbl.setObjectName("className")

            c_header.addWidget(badge_btn)
            c_header.addWidget(name_lbl)
            c_header.addStretch()
            card_layout.addLayout(c_header)

            # Real-time Bar / Metrics
            rt_layout = QHBoxLayout()
            rt_title = QLabel("Real-time:")
            rt_title.setObjectName("metricCaption")
            rt_val = QLabel("0.0%")
            rt_val.setObjectName("metricRealtime")
            rt_layout.addWidget(rt_title)
            rt_layout.addStretch()
            rt_layout.addWidget(rt_val)
            card_layout.addLayout(rt_layout)

            # Real-time Progress Bar
            pbar = QProgressBar()
            pbar.setObjectName("classBar")
            pbar.setFixedHeight(4)
            pbar.setTextVisible(False)
            self._update_bar_style(pbar, rgb)
            pbar.setValue(0)
            card_layout.addWidget(pbar)

            # Accumulated Metric
            accum_layout = QHBoxLayout()
            accum_title = QLabel("Accumulated:")
            accum_title.setObjectName("metricCaption")
            accum_val = QLabel("0 px (0.0%)")
            accum_val.setObjectName("metricTotal")
            accum_layout.addWidget(accum_title)
            accum_layout.addStretch()
            accum_layout.addWidget(accum_val)
            card_layout.addLayout(accum_layout)

            self.grid_layout.addWidget(card, idx, 0)
            self.cards_map[class_id] = (badge_btn, rt_val, accum_val, pbar)
            self._cards[class_id] = card

    @staticmethod
    def _update_badge_style(button: QPushButton, rgb: list[int] | np.ndarray) -> None:
        """Only the class colour; border/hover come from ``#classBadge`` in the theme."""
        button.setStyleSheet(f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]});")

    @staticmethod
    def _update_bar_style(bar: QProgressBar, rgb: list[int] | np.ndarray) -> None:
        """Only the chunk colour; the track comes from ``#classBar`` in the theme."""
        bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); "
                          f"border-radius: 2px; }}")

    def _pick_new_color(self, class_id: int, button: QPushButton) -> None:
        current_rgb = self._current_lut[class_id]
        initial_color = QColor(current_rgb[0], current_rgb[1], current_rgb[2])

        selected_color = QColorDialog.getColor(
            initial_color, self, f"Select Color for Class ID {class_id}")
        if selected_color.isValid():
            new_rgb = [selected_color.red(), selected_color.green(), selected_color.blue()]
            self._current_lut[class_id] = new_rgb
            self._update_badge_style(button, new_rgb)

            # Update progress bar chunk color
            if class_id in self.cards_map:
                _, _, _, pbar = self.cards_map[class_id]
                self._update_bar_style(pbar, new_rgb)
                set_accent_color(self._cards[class_id], new_rgb)

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
        for class_id, (_, rt_val, accum_val, pbar) in self.cards_map.items():
            if recent_total:          # keep the last value while no line arrives
                rt_ratio = recent_counts[class_id] / recent_total * 100.0
                rt_val.setText(f"{rt_ratio:.1f}%")
                pbar.setValue(int(round(rt_ratio)))
            accum_count = int(total_counts[class_id])
            accum_ratio = accum_count / total * 100.0 if total else 0.0
            accum_val.setText(f"{accum_count:,} ({accum_ratio:.1f}%)")

    def _on_reset_clicked(self) -> None:
        self.reset_requested.emit()
        self.reset_accumulated_stats()

    def reset_accumulated_stats(self) -> None:
        """Clears the displayed metrics (the counts themselves live in the pipeline)."""
        self.lbl_total_pixels.setText("0 px")
        for _, (_, rt_val, accum_val, pbar) in self.cards_map.items():
            rt_val.setText("0.0%")
            accum_val.setText("0 (0.0%)")
            pbar.setValue(0)
