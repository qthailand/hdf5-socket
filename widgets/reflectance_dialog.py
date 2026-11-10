# -*- coding: utf-8 -*-
"""Reflectance calibration dialog: choose the white standard, then draw it on the RAW zone.

The dialog only collects the choice; the main window keeps the newest binned lines meanwhile
(``Pipeline.start_calibration_capture``), freezes the belt map on "Draw on RAW" and turns the
drawn rectangle (slit rows x lines) into factors (``reflectance_builder.factors_from_lines``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel,
                             QVBoxLayout)

CONSTANT = '__constant__'


class ReflectanceDialog(QDialog):
    """Reference (table or constant %), factor kind, and the camera values the result is bound to."""

    def __init__(self, parent, tables: list[Path], wavelengths, camera_text: str, captured,
                 capacity: int) -> None:
        super().__init__(parent)
        self.setWindowTitle("Reflectance calibration")
        self.setModal(True)
        self._wavelengths = None if wavelengths is None else np.asarray(wavelengths, np.float64)
        self._captured, self._capacity = captured, capacity

        self.reference = QComboBox()
        for table in tables:
            self.reference.addItem(table.name, str(table))
        self.reference.addItem("Constant reflectance (%)", CONSTANT)
        self.reference.setToolTip("Certified reflectance of the white standard (table: nm, R), "
                                  "or one value for every band")
        self.percent = QDoubleSpinBox()
        self.percent.setRange(1.0, 150.0)
        self.percent.setDecimals(1)
        self.percent.setSuffix(" %")
        self.percent.setValue(99.0)
        self.percent.setToolTip("Reflectance of the standard in percent (e.g. 99 % white, 20 % grey)")
        self.table_note = QLabel("")
        self.table_note.setObjectName("metricCaption")
        self.kind = QComboBox()
        self.kind.addItem("Per band (mean over the drawn rows)", 'band')
        self.kind.addItem("Row-wise (each slit row; target across the whole slit)", 'row')
        self.kind.setToolTip("Per band: one factor per band, for a target under part of the slit.\n"
                             "Row-wise: also corrects the light along the slit; rows outside the drawn "
                             "area use the nearest drawn rows.")
        camera = QLabel(camera_text)
        camera.setObjectName("metricCaption")
        camera.setWordWrap(True)
        self.status = QLabel("")
        self.status.setObjectName("metricCaption")

        form = QFormLayout()
        form.addRow("White standard", self.reference)
        form.addRow("Reflectance", self.percent)
        form.addRow("", self.table_note)
        form.addRow("Factors", self.kind)
        form.addRow("Camera (bound)", camera)
        hint = QLabel("Put the standard under the slit, then <b>Draw on RAW</b>: the belt map freezes and you "
                      "drag a rectangle over the standard in the RAW zone. Only the newest "
                      f"{capacity} lines are kept for drawing (brighter part of the frozen view). "
                      "The dark step must be on.")
        hint.setWordWrap(True)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.draw_button = buttons.addButton("Draw on RAW", QDialogButtonBox.AcceptRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addWidget(self.status)
        layout.addWidget(buttons)
        self.setMinimumWidth(460)

        self.reference.currentIndexChanged.connect(self._sync)
        for i in range(self.reference.count()):          # Spectralon first if there is one
            if 'spectralon' in self.reference.itemText(i).lower():
                self.reference.setCurrentIndex(i)
                break
        self._sync()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_status)
        self._timer.start(200)
        self._update_status()

    def _sync(self) -> None:
        data = self.reference.currentData()
        self.percent.setEnabled(data == CONSTANT)
        if data == CONSTANT or self._wavelengths is None:
            self.table_note.setText("")
            return
        try:
            from reflectance_builder import reference_reflectance
            values = reference_reflectance(data, self._wavelengths)
            self.table_note.setText(f"table at the bands: {100 * values.min():.1f}-{100 * values.max():.1f} % "
                                    f"(mean {100 * values.mean():.1f} %)")
        except Exception as exc:                          # wrong range / unreadable table
            self.table_note.setText(f"⚠ {exc}")

    def _update_status(self) -> None:
        n = int(self._captured())
        self.status.setText(f"Lines kept for drawing: {n} / {self._capacity}")
        self.draw_button.setEnabled(n > 0)

    def choice(self) -> dict:
        """``{'reference': table path or fraction, 'label': str, 'kind': 'band'|'row'}``."""
        data = self.reference.currentData()
        if data == CONSTANT:
            pct = float(self.percent.value())
            return {'reference': pct / 100.0, 'label': f"constant {pct:g} %", 'kind': self.kind.currentData()}
        return {'reference': data, 'label': Path(data).name, 'kind': self.kind.currentData()}
