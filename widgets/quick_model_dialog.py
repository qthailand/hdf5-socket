# -*- coding: utf-8 -*-
"""Quick model dialog: mark areas of a frozen view per class, train a linear SGD model, try it.

For system tests: the model lives in memory only (``quick_model``), as do the labels (the main
window's ``quick_model.LabelSet``), so closing the program forgets both. The dialog works on its
own copy of the newest model-input lines (``Pipeline.spectrum_snapshot``); the pipeline keeps
running behind it.

    left drag   mark a rectangle as the selected class (tool Rectangle)
    left click  mark the area that looks like the clicked pixel (tool Wand, see Tolerance)
    right drag  erase marks
"""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QDialog,
                             QHBoxLayout, QHeaderView, QLabel, QPushButton, QSizePolicy, QSlider, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from quick_model import LabelSet, spectral_angle, train_artifact
from utils import pick_color

#: lines classified at a time for the preview (bounds the temporary copy)
PREVIEW_CHUNK = 64
UNDO_STEPS = 10


class _LabelCanvas(QWidget):
    """The frozen view (x: lines, oldest left; y: slit) with the marks; reports the mouse to the dialog."""

    def __init__(self, dialog: 'QuickModelDialog') -> None:
        super().__init__(dialog)
        self._dialog = dialog
        self._start: QPoint | None = None
        self._rect: QRect | None = None
        self._erase = False
        self.setMinimumSize(480, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.CrossCursor)

    def pixel(self, pos) -> tuple[int, int]:
        """``(line index, slit row)`` of the view under ``pos``."""
        lines, slit = self._dialog.shape
        line = int(np.clip(pos.x() * lines // max(1, self.width()), 0, lines - 1))
        row = int(np.clip(pos.y() * slit // max(1, self.height()), 0, slit - 1))
        return line, row

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        image = self._dialog.image()
        if image is None:
            painter.fillRect(self.rect(), QColor(20, 20, 20))
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignCenter, "No lines yet - press Grab again")
            return
        painter.drawImage(self.rect(), image)
        if self._rect is not None:
            painter.setPen(QPen(QColor(255, 255, 255), 1, Qt.DashLine if self._erase else Qt.SolidLine))
            painter.drawRect(self._rect)

    def mousePressEvent(self, event) -> None:
        if not self._dialog.has_view or event.button() not in (Qt.LeftButton, Qt.RightButton):
            return
        self._erase = event.button() == Qt.RightButton
        if not self._erase and self._dialog.tool == 'wand':
            self._dialog.wand(*self.pixel(event.pos()))
            return
        self._start = event.pos()
        self._rect = QRect(self._start, self._start)
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._start is not None:
            self._rect = QRect(self._start, event.pos()).normalized()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._start is None:
            return
        (l0, r0), (l1, r1) = self.pixel(self._start), self.pixel(event.pos())
        self._start = self._rect = None
        self._dialog.paint_rect(min(l0, l1), max(l0, l1), min(r0, r1), max(r0, r1), erase=self._erase)
        self.update()


class QuickModelDialog(QDialog):
    """Label, train and hand over a session model (``model_ready``: the artifact)."""

    model_ready = pyqtSignal(object)

    def __init__(self, parent, grab, wavelengths, input_units: str, labels: LabelSet) -> None:
        """
        Args:
            grab: ``() -> (line numbers, spectra (lines, slit, bands))``: the newest model input.
            wavelengths: nm of every band, or ``None``.
            input_units: ``'reflectance'`` or ``'counts'``: what the model input is now.
            labels: The session's classes and labelled spectra (kept by the caller).
        """
        super().__init__(parent)
        self.setWindowTitle("Quick model (this session only)")
        self.setModal(True)
        self._grab, self._wavelengths, self._units, self.labels = grab, wavelengths, input_units, labels
        self._cube: np.ndarray | None = None
        self._mask = np.zeros((0, 0), np.uint8)            # (lines, slit): class id marked, 0 = none
        self._gray = np.zeros((0, 0), np.uint8)
        self._undo: list[np.ndarray] = []
        self._wand: tuple | None = None                    # (angles, mask before the click, class id)
        self._preview: np.ndarray | None = None
        self._image: QImage | None = None
        self._filling = False
        self.artifact: dict | None = None
        self.report: dict | None = None
        self.added = 0

        def button(text, tip, checkable=False):
            b = QPushButton(text)
            b.setObjectName("toggleButton")
            b.setCheckable(checkable)
            b.setToolTip(tip)
            return b

        self.grab_button = button("Grab again", "Take the newest lines again (another scene or material). The marks "
                                                "of this view are kept as labelled pixels first.")
        self.rect_button = button("Rectangle", "Drag a rectangle: its pixels get the selected class", True)
        self.wand_button = button("Wand", "Click a pixel: the connected area with a similar spectrum gets the "
                                          "selected class (Tolerance widens or narrows it)", True)
        self.rect_button.setChecked(True)
        tools = QButtonGroup(self)
        tools.addButton(self.rect_button)
        tools.addButton(self.wand_button)
        self.tolerance = QSlider(Qt.Horizontal)
        self.tolerance.setRange(1, 40)                     # hundredths of a radian
        self.tolerance.setValue(6)
        self.tolerance.setFixedWidth(120)
        self.tolerance.setToolTip("Wand: largest spectral angle to the clicked pixel; moving it adjusts the last "
                                  "Wand area")
        self._tolerance_label = QLabel("")
        self._tolerance_label.setObjectName("metricCaption")
        self.undo_button = button("Undo", "Take back the last mark")
        top = QHBoxLayout()
        for widget in (self.grab_button, self.rect_button, self.wand_button, self.tolerance, self._tolerance_label,
                       self.undo_button):
            top.addWidget(widget)
        top.addStretch()
        hint = QLabel("left: mark · right drag: erase")
        hint.setObjectName("metricCaption")
        top.addWidget(hint)

        self.canvas = _LabelCanvas(self)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["", "Class", "Pixels"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.setColumnWidth(0, 26)
        self.table.setToolTip("Select the class to mark. Double-click a name to change it; click the colour to "
                              "choose another.")
        self.add_class_button = button("+ Class", "Add a class")
        self.remove_class_button = button("Remove", "Remove the selected class and its labelled pixels")
        class_buttons = QHBoxLayout()
        class_buttons.addWidget(self.add_class_button)
        class_buttons.addWidget(self.remove_class_button)
        caption = QLabel("CLASSES")
        caption.setObjectName("caption")
        side = QVBoxLayout()
        side.addWidget(caption)
        side.addWidget(self.table, stretch=1)
        side.addLayout(class_buttons)
        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setFixedWidth(250)
        side.setContentsMargins(0, 0, 0, 0)
        middle = QHBoxLayout()
        middle.addWidget(self.canvas, stretch=1)
        middle.addWidget(side_widget)

        self.status = QLabel("")
        self.status.setObjectName("metricCaption")
        self.status.setWordWrap(True)
        self.preview_box = QCheckBox("Preview")
        self.preview_box.setToolTip("Colour the view with the trained model's classes")
        self.preview_box.setEnabled(False)
        self.train_button = button("Train", "Train a linear SGD model on the labelled pixels (well under a second)")
        self.add_button = button("Add to models", "Put the trained model beside the other models (Materials tab) "
                                                  "and switch to it. It is not saved: gone when the program closes.")
        self.add_button.setEnabled(False)
        self.close_button = button("Close", "Close; the labelled pixels stay until the program closes")
        bottom = QHBoxLayout()
        bottom.addWidget(self.status, stretch=1)
        for widget in (self.preview_box, self.train_button, self.add_button, self.close_button):
            bottom.addWidget(widget)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(middle, stretch=1)
        layout.addLayout(bottom)
        self.resize(1040, 560)

        self.grab_button.clicked.connect(self.grab_view)
        self.tolerance.valueChanged.connect(self._on_tolerance)
        self.undo_button.clicked.connect(self.undo)
        self.add_class_button.clicked.connect(lambda: self.add_class())
        self.remove_class_button.clicked.connect(self.remove_class)
        self.table.itemChanged.connect(self._on_item_changed)
        self.preview_box.toggled.connect(lambda _: self._changed())
        self.train_button.clicked.connect(self.train)
        self.add_button.clicked.connect(self.add_to_models)
        self.close_button.clicked.connect(self.accept)
        self._on_tolerance()
        if not self.labels.class_ids:                      # two classes to start with: the least a model needs
            self.labels.add_class()
            self.labels.add_class()
        self._fill_table(select=self.labels.class_ids[0])
        self.grab_view()

    # ------------------------------------------------------------------------------ view ---

    @property
    def has_view(self) -> bool:
        return self._cube is not None and self._cube.shape[0] > 0

    @property
    def shape(self) -> tuple[int, int]:
        """``(lines, slit)`` of the frozen view."""
        return self._mask.shape

    @property
    def tool(self) -> str:
        return 'wand' if self.wand_button.isChecked() else 'rect'

    @property
    def mask(self) -> np.ndarray:
        """Class id marked on every pixel ``(lines, slit)`` of the view (0: none)."""
        return self._mask

    def grab_view(self) -> None:
        """Freeze the newest lines; what was marked on the previous view is kept as labelled spectra."""
        self.commit()
        _, cube = self._grab()
        self._cube = cube if cube.shape[0] else None
        self._undo, self._wand, self._preview = [], None, None
        if self._cube is None:
            self._mask = np.zeros((0, 0), np.uint8)
        else:
            self._mask = np.zeros(cube.shape[:2], np.uint8)
            with np.errstate(invalid='ignore'):
                level = np.nan_to_num(np.nanmean(cube, axis=2), nan=0.0, posinf=0.0, neginf=0.0)
            lo, hi = np.percentile(level, (1, 99))
            self._gray = (np.clip((level - lo) / max(hi - lo, 1e-12), 0, 1) * 255).astype(np.uint8).T.copy()
            if self.artifact is not None:                  # the trained model on the new scene
                self._preview = self._predict(self.artifact)
        self._changed()

    def commit(self) -> None:
        """Move the spectra under the marks into the session's labels and clear the marks."""
        if self.has_view:
            for cid in np.unique(self._mask):
                if cid:
                    self.labels.add(int(cid), self._cube[self._mask == cid])
            self._mask[:] = 0
        self._undo, self._wand = [], None

    def image(self) -> QImage | None:
        """The view as drawn: grey level, the model's classes (Preview), the marks on top."""
        if not self.has_view:
            return None
        if self._image is None:
            rgb = np.repeat(self._gray[:, :, None], 3, axis=2).astype(np.float32)
            lut = np.zeros((256, 3), np.float32)
            for cid, colour in self.labels.colors.items():
                lut[cid] = colour
            if self._preview is not None and self.preview_box.isChecked():
                rgb = 0.35 * rgb + 0.65 * lut[self._preview.T]
            marked = self._mask.T > 0
            rgb[marked] = 0.3 * rgb[marked] + 0.7 * lut[self._mask.T[marked]]
            data = np.ascontiguousarray(rgb.astype(np.uint8))
            h, w = data.shape[:2]
            self._image = QImage(data.tobytes(), w, h, 3 * w, QImage.Format_RGB888).copy()
        return self._image

    def _changed(self) -> None:
        """Marks, colours or the preview changed: redraw, recount."""
        self._image = None
        self.canvas.update()
        self._fill_counts()
        self.undo_button.setEnabled(bool(self._undo))
        self.preview_box.setEnabled(self._preview is not None)
        if not self.has_view:
            self.status.setText("No lines kept yet (the pipeline has just started keeping them): press Grab again.")
        elif self.report is None:
            lines, slit = self.shape
            self.status.setText(f"{lines} lines × {slit} slit pixels × {self._cube.shape[2]} bands · "
                                f"model input: {self._units}")

    # ---------------------------------------------------------------------------- classes ---

    def current_class(self) -> int | None:
        row = self.table.currentRow()
        item = self.table.item(row, 1) if row >= 0 else None
        return None if item is None else int(item.data(Qt.UserRole))

    def count(self, cid: int) -> int:
        """Labelled pixels of a class: kept from earlier views and marked on this one."""
        return self.labels.count(cid) + int((self._mask == cid).sum())

    def add_class(self, name: str | None = None) -> int:
        cid = self.labels.add_class(name)
        self._fill_table(select=cid)
        return cid

    def remove_class(self) -> None:
        cid = self.current_class()
        if cid is None:
            return
        self.labels.remove_class(cid)
        self._mask[self._mask == cid] = 0
        self._undo, self._wand = [], None
        ids = self.labels.class_ids
        self._fill_table(select=ids[0] if ids else None)
        self._changed()

    def _fill_table(self, select: int | None = None) -> None:
        self._filling = True
        ids = self.labels.class_ids
        self.table.setRowCount(len(ids))
        for row, cid in enumerate(ids):
            swatch = QTableWidgetItem("")
            swatch.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            # a button, not the item's background: the row selection would paint over that
            colour = QPushButton()
            colour.setCursor(Qt.PointingHandCursor)
            colour.setStyleSheet("border: none; border-radius: 0; background-color: rgb(%d, %d, %d);"
                                 % tuple(self.labels.colors[cid]))
            colour.clicked.connect(lambda _=False, r=row: self._on_cell_clicked(r, 0))
            self.table.setCellWidget(row, 0, colour)
            name = QTableWidgetItem(self.labels.names[cid])
            name.setData(Qt.UserRole, cid)
            count = QTableWidgetItem("")
            count.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            count.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            for column, item in enumerate((swatch, name, count)):
                self.table.setItem(row, column, item)
            if cid == select:
                self.table.setCurrentCell(row, 1)
        self._filling = False
        self._fill_counts()

    def _fill_counts(self) -> None:
        self._filling = True
        for row in range(self.table.rowCount()):
            cid = int(self.table.item(row, 1).data(Qt.UserRole))
            self.table.item(row, 2).setText(f"{self.count(cid):,}")
        self._filling = False

    def _on_item_changed(self, item) -> None:
        if self._filling or item.column() != 1:
            return
        cid = int(item.data(Qt.UserRole))
        self.labels.names[cid] = item.text().strip() or f"class {cid}"

    def _on_cell_clicked(self, row: int, column: int) -> None:
        if column != 0:
            return
        cid = int(self.table.item(row, 1).data(Qt.UserRole))
        colour = pick_color(QColor(*self.labels.colors[cid]), self, "Class colour")
        if colour.isValid():
            self.labels.colors[cid] = [colour.red(), colour.green(), colour.blue()]
            self._fill_table(select=cid)
            self._changed()

    # ------------------------------------------------------------------------------ marks ---

    def _remember(self) -> None:
        self._undo = (self._undo + [self._mask.copy()])[-UNDO_STEPS:]

    def paint_rect(self, line0: int, line1: int, row0: int, row1: int, erase: bool = False) -> None:
        """Mark lines ``line0..line1`` × slit rows ``row0..row1`` (inclusive) as the selected class."""
        cid = 0 if erase else self.current_class()
        if cid is None or not self.has_view:
            return
        self._remember()
        self._mask[line0:line1 + 1, row0:row1 + 1] = cid
        self._wand = None
        self._changed()

    def wand(self, line: int, row: int) -> None:
        """Mark the connected area that looks like pixel ``(line, row)`` as the selected class."""
        cid = self.current_class()
        if cid is None or not self.has_view:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            angles = spectral_angle(self._cube, self._cube[line, row])
        finally:
            QApplication.restoreOverrideCursor()
        self._remember()
        self._wand = (angles, self._mask.copy(), cid, (line, row))
        self._apply_wand()

    def _apply_wand(self) -> None:
        from scipy import ndimage
        angles, before, cid, (line, row) = self._wand
        similar = angles <= self.tolerance.value() / 100.0
        similar[line, row] = True
        areas, _ = ndimage.label(similar)
        self._mask = before.copy()
        self._mask[areas == areas[line, row]] = cid
        self._changed()

    def _on_tolerance(self) -> None:
        self._tolerance_label.setText(f"Tolerance {self.tolerance.value() / 100:.2f} rad")
        if self._wand is not None:
            self._apply_wand()

    def undo(self) -> None:
        if self._undo:
            self._mask = self._undo.pop()
            self._wand = None
            self._changed()

    # ------------------------------------------------------------------------------ model ---

    def samples(self) -> dict[int, np.ndarray]:
        """Class id -> every labelled spectrum: kept from earlier views and marked on this one."""
        n_bands = self._cube.shape[2] if self.has_view else None
        out = dict(self.labels.spectra(n_bands))
        if self.has_view:
            for cid in np.unique(self._mask):
                if cid:
                    here = self._cube[self._mask == cid]
                    out[int(cid)] = np.concatenate([out[int(cid)], here]) if int(cid) in out else here
        return out

    def _predict(self, artifact: dict) -> np.ndarray | None:
        """Class id of every pixel of the view ``(lines, slit)``; ``None`` if the model does not fit it."""
        from inference_engine import PushbroomLineClassifier
        classifier = PushbroomLineClassifier.from_artifact(artifact, 'preview')
        lines, slit, bands = self._cube.shape
        try:
            columns, _ = classifier.bind(self._wavelengths, bands)
        except ValueError:
            return None
        out = np.zeros((lines, slit), np.uint8)
        for first in range(0, lines, PREVIEW_CHUNK):
            part = self._cube[first:first + PREVIEW_CHUNK]
            out[first:first + len(part)] = classifier.predict(part.reshape(-1, bands), columns).reshape(-1, slit)
        return out

    def train(self) -> bool:
        """Train on the labelled pixels; the status line says what came out. ``True`` when a model exists."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            artifact, report = train_artifact(self.samples(), self.labels.names, self._wavelengths, self._units,
                                              self.labels.colors)
        except Exception as exc:                           # too few labels, sklearn missing, ...
            self.status.setText(f"⚠ Not trained: {exc}")
            return False
        finally:
            QApplication.restoreOverrideCursor()
        self.artifact, self.report = artifact, report
        self._preview = self._predict(artifact) if self.has_view else None
        self.preview_box.setEnabled(self._preview is not None)
        self.preview_box.setChecked(self._preview is not None)
        self.add_button.setEnabled(True)
        used = ', '.join(f"{self.labels.names[c]} {n:,}" for c, n in report['samples'].items())
        accuracy = '' if report['accuracy'] is None else f"{100 * report['accuracy']:.1f} % on held-out pixels " \
                                                         "of the same areas · "
        self.status.setText(f"Trained: {accuracy}{used} · input: {self._units}")
        self._changed()
        return True

    def add_to_models(self) -> None:
        if self.artifact is None:
            return
        self.added += 1
        self.model_ready.emit(self.artifact)
        self.add_button.setEnabled(False)                  # the same model twice helps nobody
        self.status.setText(self.status.text() + " · added to the models")

    def done(self, result: int) -> None:
        self.commit()                                      # the marks become labelled pixels of the session
        super().done(result)
