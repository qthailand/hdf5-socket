# -*- coding: utf-8 -*-
"""Reflectance settings and a frozen, zoomable view for selecting the reference ROI."""
from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPen, QPixmap
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QDoubleSpinBox, QFrame, QGraphicsScene, QGraphicsView,
                             QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout)

from themes import make_panel, repolish, set_accent

CONSTANT = '__constant__'
SIGNAL_EMPTY = "Grab an image and draw the standard."
STATE = {'green': 'ok', 'yellow': 'warn', 'red': 'error', 'grey': 'neutral'}    # check level -> theme accent
VERDICT = {'green': "Ready to calibrate", 'yellow': "Usable, with limits", 'red': "Do not calibrate"}
SIDE_WIDTH = 380


def _label(text: str, name: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName(name)
    return label


def _card(title: str, accent: str | None = None):
    """``(frame, layout)``: a themed panel with a small caption on top."""
    frame = make_panel(QFrame(), accent)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 10, 14, 12)
    layout.setSpacing(6)
    layout.addWidget(_label(title, 'caption'))
    return frame, layout


def _set_state(widget, state: str) -> None:
    if widget.property('state') != state:
        widget.setProperty('state', state)
        repolish(widget)


class MetricTile(QFrame):
    """One number of the signal check: caption, value, a line of detail; the stripe shows its level."""

    def __init__(self, caption: str) -> None:
        super().__init__()
        make_panel(self, 'neutral', alt=True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 8, 7)
        layout.setSpacing(0)
        layout.addWidget(_label(caption, 'caption'))
        self.value = _label('–', 'bigValue')
        self.detail = _label('', 'metricCaption')
        layout.addWidget(self.value)
        layout.addWidget(self.detail)

    def show_value(self, value: str, detail: str, level: str = 'grey') -> None:
        self.value.setText(value)
        self.detail.setText(detail)
        set_accent(self, STATE[level])


class ReferenceView(QGraphicsView):
    """Image coordinates are scan positions (x) and slit rows (y), independent of zoom/pan."""
    roi_changed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setMinimumSize(460, 260)
        self.setTransformationAnchor(self.AnchorUnderMouse)
        self.setResizeAnchor(self.AnchorViewCenter)
        self.setCursor(Qt.CrossCursor)
        self._image = None
        self._selection = None
        self._start = self._pan = None
        self._fit = True
        self.roi = None

    def set_image(self, gray):
        self.scene().clear()
        self._selection = None
        self._start = self._pan = None
        self.roi = None
        h, w = gray.shape
        gray = np.ascontiguousarray(gray, dtype=np.uint8)
        image = QImage(gray.data, w, h, gray.strides[0], QImage.Format_Grayscale8).copy()
        self._image = self.scene().addPixmap(QPixmap.fromImage(image))
        self.setSceneRect(QRectF(0, 0, w, h))
        self.fit_image()
        self.roi_changed.emit(None)

    def fit_image(self):
        if self._image is not None:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self._fit = True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit:
            self.fit_image()

    def wheelEvent(self, event):
        if self._image is None:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        if 0.02 <= self.transform().m11() * factor <= 128:
            self.scale(factor, factor)
            self._fit = False
        event.accept()

    def mousePressEvent(self, event):
        if self._image is None:
            return
        if event.button() in (Qt.RightButton, Qt.MiddleButton):
            self._pan = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
            point = self.mapToScene(event.pos())
            if not self.sceneRect().contains(point):
                return
            self._start = point
            self.roi = None
            self.roi_changed.emit(None)
            if self._selection is not None:
                self.scene().removeItem(self._selection)
            pen = QPen(QColor('#FFB547'), 2)
            pen.setCosmetic(True)
            self._selection = self.scene().addRect(QRectF(point, point), pen)

    def mouseMoveEvent(self, event):
        if self._pan is not None:
            delta = event.pos() - self._pan
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._pan = event.pos()
        elif self._start is not None:
            rect = QRectF(self._start, self.mapToScene(event.pos())).normalized().intersected(self.sceneRect())
            self._selection.setRect(rect)

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.RightButton, Qt.MiddleButton):
            self._pan = None
            self.setCursor(Qt.CrossCursor)
        elif event.button() == Qt.LeftButton and self._start is not None:
            rect = QRectF(self._start, self.mapToScene(event.pos())).normalized().intersected(self.sceneRect())
            self._start = None
            if rect.width() > 0 and rect.height() > 0:
                x0, x1 = max(0, int(np.floor(rect.left()))), min(int(self.sceneRect().width()), int(np.ceil(rect.right())))
                y0, y1 = max(0, int(np.floor(rect.top()))), min(int(self.sceneRect().height()), int(np.ceil(rect.bottom())))
                self.roi = (x0, x1, y0, y1)
                self._selection.setRect(QRectF(x0, y0, x1 - x0, y1 - y0))
                self.roi_changed.emit(self.roi)


class ReflectanceDialog(QDialog):
    """Reference (table or constant %), factor kind, and the camera values the result is bound to."""

    calibration_requested = pyqtSignal()

    def __init__(self, parent, tables: list[Path], wavelengths, camera_text: str, captured,
                 capacity: int, grab, current=None, context=None) -> None:
        """``current()`` -> ``(factors, label)`` in use now, or ``None`` (enables *Check spectrum*).

        ``context()`` -> what the signal check and the corrections table need: ``full_scale``,
        ``dark_level`` (counts, ``None`` when the dark step is off), ``steps`` as
        ``[(name, on, source file / origin)]``, ``flatfield_label`` (``None``: no profile loaded)
        and ``flatfield_matches`` (the profile's Dark/Flip/Smile are the ones in use).
        """
        super().__init__(parent)
        self._current = current or (lambda: None)
        self._context = context or (lambda: {})
        self.signal = None
        self.setWindowTitle("Reflectance calibration")
        self.setModal(True)
        self._wavelengths = None if wavelengths is None else np.asarray(wavelengths, np.float64)
        self._captured, self._capacity = captured, capacity
        self._grab = grab
        self.line_ids = np.empty(0, np.int64)
        self.lines = None
        self.clipped = None                 # (lines, slit) raw pixels at full scale, when the grab gives it

        self.setMinimumSize(1040, 680)

        # ------------------------------------------------------------ header
        header = QVBoxLayout()
        header.setSpacing(1)
        header.addWidget(_label("Reflectance calibration", 'dialogTitle'))
        camera = _label(camera_text, 'metricCaption')
        camera.setToolTip("The calibration is bound to these camera values")
        header.addWidget(camera)

        # ------------------------------------------------------------ left: image and its spectrum
        image_card, image_box = _card("STANDARD UNDER THE SLIT")
        self.view = ReferenceView(self)
        self.view.setObjectName('referenceView')
        self.grab_button = QPushButton('Grab image')
        self.fit_button = QPushButton('Fit')
        self.grab_button.setToolTip("Freeze the newest lines; then drag a rectangle over the standard")
        self.fit_button.setToolTip("Show the whole image")
        toolbar = QHBoxLayout()
        toolbar.addWidget(self.grab_button)
        toolbar.addWidget(self.fit_button)
        toolbar.addStretch()
        toolbar.addWidget(_label("drag: select area  ·  wheel: zoom  ·  right drag: pan", 'metricCaption'))
        self.status = _label("", 'metricCaption')
        image_box.addLayout(toolbar)
        image_box.addWidget(self.view, stretch=1)
        image_box.addWidget(self.status)

        from widgets.spectrum_check import CheckPlot
        plot_card, plot_box = _card("SIGNAL IN THE DRAWN AREA")
        self.signal_plot = CheckPlot("counts after dark", self, include_zero=True)
        self.signal_plot.setMinimumHeight(150)
        plot_box.addWidget(self.signal_plot, stretch=1)

        left = QVBoxLayout()
        left.setSpacing(10)
        left.addWidget(image_card, stretch=5)
        left.addWidget(plot_card, stretch=2)

        # ------------------------------------------------------------ right: standard, corrections, check
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
        self.table_note = _label("", 'metricCaption')
        self.table_note.setWordWrap(True)
        self.kind = QComboBox()
        self.kind.addItem("Per band (mean over the drawn rows)", 'band')
        self.kind.addItem("Row-wise (each slit row; target across the whole slit)", 'row')
        self.kind.setToolTip("Per band: one factor per band, for a target under part of the slit.\n"
                             "Row-wise: also corrects the light along the slit; rows outside the drawn "
                             "area use the nearest drawn rows.\n"
                             "Flat-field + reference (when a profile is loaded): the profile supplies spatial "
                             "response; the drawn target supplies the reflectance scale. Correction is applied once.")
        standard_card, standard_box = _card("REFERENCE")
        standard_box.addWidget(_label("White standard", 'fieldLabel'))
        standard_box.addWidget(self.reference)
        standard_box.addWidget(self.percent)
        standard_box.addWidget(self.table_note)
        standard_box.addSpacing(4)
        standard_box.addWidget(_label("Factors", 'fieldLabel'))
        standard_box.addWidget(self.kind)

        corrections_card, corrections_box = _card("CORRECTIONS IN THIS CALIBRATION")
        self._corrections_grid = QGridLayout()
        self._corrections_grid.setHorizontalSpacing(10)
        self._corrections_grid.setVerticalSpacing(1)
        self._corrections_grid.setColumnStretch(1, 1)
        corrections_box.addLayout(self._corrections_grid)
        self.corrections_rows: list[tuple[str, str, str, str]] = []      # name, state text, accent, source
        self._corrections_widgets: list[tuple[QLabel, QLabel, QLabel]] = []

        self.signal_card, signal_box = _card("SIGNAL CHECK", 'neutral')
        self.verdict = _label("Waiting for an area", 'verdict')
        signal_box.addWidget(self.verdict)
        self.tiles = {'peak': MetricTile("PEAK LEVEL"), 'snr': MetricTile("SNR · 1 PIXEL"),
                      'even': MetricTile("EVENNESS"), 'dark': MetricTile("DARK OFFSET")}
        tiles = QGridLayout()
        tiles.setSpacing(6)
        for i, tile in enumerate(self.tiles.values()):
            tiles.addWidget(tile, i // 2, i % 2)
        signal_box.addLayout(tiles)
        self.signal_text = _label(SIGNAL_EMPTY, 'issueList')
        self.signal_text.setWordWrap(True)
        self.signal_text.setTextFormat(Qt.RichText)
        signal_box.addWidget(self.signal_text)

        right = QVBoxLayout()
        right.setSpacing(10)
        for card in (standard_card, corrections_card, self.signal_card):
            card.setFixedWidth(SIDE_WIDTH)
            right.addWidget(card)
        right.addStretch()

        # ------------------------------------------------------------ footer
        self.check_button = QPushButton("Check spectrum…")
        self.check_button.setObjectName('quietButton')
        self.check_button.setToolTip("Plot the drawn area calibrated with the factors in use against the selected\n"
                                     "standard (and the error per band). Changes nothing.")
        self.apply_button = QPushButton("Review calibration…")
        self.apply_button.setObjectName('primaryButton')
        self.apply_button.setDefault(True)
        cancel = QPushButton("Cancel")
        cancel.setObjectName('quietButton')
        footer = QHBoxLayout()
        footer.addWidget(self.check_button)
        footer.addStretch()
        footer.addWidget(cancel)
        footer.addWidget(self.apply_button)

        body = QHBoxLayout()
        body.setSpacing(10)
        body.addLayout(left, stretch=1)
        body.addLayout(right)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        layout.addLayout(header)
        layout.addLayout(body, stretch=1)
        layout.addLayout(footer)
        self.resize(1220, 800)

        cancel.clicked.connect(self.reject)
        self.check_button.clicked.connect(self.check_spectrum)
        self.apply_button.clicked.connect(self.calibration_requested.emit)
        self.grab_button.clicked.connect(self.grab_image)
        self.fit_button.clicked.connect(self.view.fit_image)
        self.view.roi_changed.connect(self._update_status)
        self.view.roi_changed.connect(self._update_signal)
        self.kind.currentIndexChanged.connect(self._update_corrections)
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
        self._update_corrections()

    def _update_corrections(self, *_) -> None:
        """Every correction the white goes through: a state pill, and the file it comes from."""
        context = self._context()
        rows = [(name, 'ON' if on else 'OFF', 'ok' if on else 'error' if name.startswith('Dark') else 'neutral', source)
                for name, on, source in context.get('steps', ())]
        label, kind = context.get('flatfield_label'), self.kind.currentData()
        if kind == 'flatfield' and label and context.get('flatfield_matches'):
            rows.append(('Flat-field', 'IN FACTORS', 'ok', label))
        elif kind == 'flatfield':
            rows.append(('Flat-field', 'MISMATCH', 'error', f"{label or 'no profile'}: Dark/Flip/Smile differ from it"))
        elif label:
            rows.append(('Flat-field', 'NOT USED', 'warn', f"{label} (loaded; pick Flat-field + reference)"))
        else:
            rows.append(('Flat-field', 'NONE', 'neutral', 'no profile loaded'))
        self.corrections_rows = rows
        grid = self._corrections_grid
        while len(self._corrections_widgets) < len(rows):          # rows are reused, never rebuilt
            i = len(self._corrections_widgets)
            chip = _label('', 'chip')
            chip.setAlignment(Qt.AlignCenter)
            chip.setFixedSize(86, 20)
            widgets = (chip, _label('', 'rowName'), _label('', 'rowSource'))
            grid.addWidget(chip, 2 * i, 0, 2, 1, Qt.AlignVCenter)
            grid.addWidget(widgets[1], 2 * i, 1)
            grid.addWidget(widgets[2], 2 * i + 1, 1)
            grid.setRowMinimumHeight(2 * i + 1, 20)
            self._corrections_widgets.append(widgets)
        for i, (chip, name_label, file_label) in enumerate(self._corrections_widgets):
            for widget in (chip, name_label, file_label):
                widget.setVisible(i < len(rows))
            if i >= len(rows):
                continue
            name, state, accent, source = rows[i]
            chip.setText(state)
            _set_state(chip, accent)
            name_label.setText(name)
            file_label.setText(file_label.fontMetrics().elidedText(source, Qt.ElideRight, SIDE_WIDTH - 140))
            file_label.setToolTip(source)

    def _update_signal(self, *_) -> None:
        """Recompute the signal check for the drawn area (``self.signal``; ``None`` without an area)."""
        from reflectance_builder import signal_check
        self.signal = None
        self.signal_plot.clear()
        report = None
        message = SIGNAL_EMPTY
        if self.lines is not None and self.view.roi is not None:
            context = self._context()
            try:
                lines, rows, _ = self.selection()
                x0, x1 = self.view.roi[:2]
                report = signal_check(lines, rows, self._wavelengths, context.get('full_scale'),
                                      context.get('dark_level'),
                                      None if self.clipped is None else self.clipped[x0:x1])
            except ValueError as exc:
                message = f"⚠ {escape(str(exc))}"
        if report is None:
            self.verdict.setText("Waiting for an area")
            _set_state(self.verdict, '')
            set_accent(self.signal_card, 'neutral')
            for tile in self.tiles.values():
                tile.show_value('–', '')
            self.signal_text.setText(message)
            self.signal_text.setVisible(True)
            return
        self.signal = report
        x = self._wavelengths if self._wavelengths is not None else np.arange(report['mean'].size, dtype=np.float64)
        noise = np.nan_to_num(report['noise'])
        self.signal_plot.add_band(x, report['mean'] - noise, report['mean'] + noise, '#F97316')
        self.signal_plot.add_curve(x, report['mean'], '#F97316', "mean ± 1-pixel noise")
        self.verdict.setText(VERDICT[report['status']])
        _set_state(self.verdict, STATE[report['status']])
        set_accent(self.signal_card, STATE[report['status']])
        levels, pct = report['levels'], report['peak_pct']
        self.tiles['peak'].show_value('–' if pct is None else f"{pct:.{2 if pct < 1 else 1}f} %",
                                      f"{report['peak']:,.0f} counts" + ('' if pct is not None else ' · range unknown')
                                      + (f" · {report['clipped_pixels']:,} clipped" if report['clipped_pixels'] else ''),
                                      levels['peak'])
        self.tiles['snr'].show_value("> 1000" if report['median_snr'] > 1000 else f"{report['median_snr']:.0f}",
                                     f"median · worst {min(report['min_snr'], 9999):.0f}", levels['snr'])
        self.tiles['even'].show_value(f"±{max(report['slit_spread_pct'], report['scan_spread_pct']):.1f} %",
                                      f"slit ±{report['slit_spread_pct']:.1f} · scan ±{report['scan_spread_pct']:.1f}",
                                      levels['even'])
        dark = report['dark_level']
        self.tiles['dark'].show_value(f"{dark:,.0f}" if dark else 'OFF', "counts subtracted" if dark else "no dark",
                                      levels['dark'])
        tokens = QApplication.instance().property('theme_tokens') or {}
        colour = {'red': tokens.get('danger', '#EF4444'), 'yellow': tokens.get('warning', '#F59E0B')}
        self.signal_text.setText('<br>'.join(f"<span style='color:{colour[level]}'>●</span>&nbsp; {escape(text)}"
                                             for level, text in report['issues']))
        self.signal_text.setVisible(bool(report['issues']))
        self._update_status()

    def grab_image(self):
        ids, lines, *rest = self._grab()
        if not len(ids):
            return
        self.line_ids, self.lines = ids.copy(), lines.copy()
        self.clipped = rest[0].copy() if rest and rest[0] is not None else None
        brightness = self.lines.mean(axis=2, dtype=np.float32).T
        finite = brightness[np.isfinite(brightness)]
        lo, hi = np.percentile(finite, (1, 99)) if finite.size else (0, 1)
        gray = np.nan_to_num((brightness - lo) * (255 / max(float(hi - lo), 1e-6)), nan=0, posinf=255, neginf=0)
        self.view.set_image(np.clip(gray, 0, 255).astype(np.uint8))
        self._update_status()

    def selection(self):
        """Frozen spectra, slit range and exact line IDs from the drawn image (half-open ROI)."""
        if self.lines is None or self.view.roi is None:
            raise ValueError('Grab an image and draw a rectangle over the reference first')
        x0, x1, y0, y1 = self.view.roi
        return self.lines[x0:x1], (y0, y1), self.line_ids[x0:x1]

    def check_report(self) -> tuple[dict, str]:
        """``(verify_lines report, factors label)`` of the drawn area with the factors in use."""
        from reflectance_builder import reference_values, verify_lines
        current = self._current()
        if current is None:
            raise ValueError('No reflectance factors in use yet: calibrate first')
        if self._wavelengths is None:
            raise ValueError('The data has no wavelength axis')
        lines, rows, _ = self.selection()
        target = reference_values(self.choice()['reference'], self._wavelengths)
        return verify_lines(lines, rows, current[0], target, self._wavelengths), current[1]

    def check_spectrum(self) -> None:
        from widgets.spectrum_check import SpectrumCheckDialog
        try:
            report, label = self.check_report()
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Reflectance check", str(exc))
            return
        SpectrumCheckDialog(self, self._wavelengths, report, self.choice()['label'], label).exec_()

    def _sync(self) -> None:
        data = self.reference.currentData()
        self.percent.setEnabled(data == CONSTANT)
        self.percent.setVisible(data == CONSTANT)          # a table: no stray "99 %"
        self.table_note.setVisible(data != CONSTANT)
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

    def _update_status(self, *_) -> None:
        n = int(self._captured())
        frozen = f'Frozen: {len(self.line_ids)} lines' if self.lines is not None else 'Nothing grabbed yet'
        if self.view.roi is not None:
            x0, x1, y0, y1 = self.view.roi
            frozen += f'  ·  area {x1 - x0} lines × {y1 - y0} slit rows'
        self.status.setText(f'{frozen}  ·  buffer {n}/{self._capacity}')
        self.grab_button.setEnabled(n > 0)
        self.fit_button.setEnabled(self.lines is not None)
        drawn = self.lines is not None and self.view.roi is not None
        blocked = self.signal is not None and self.signal['status'] == 'red'
        self.apply_button.setEnabled(drawn and not blocked)
        self.apply_button.setToolTip("The signal check is red: fix what it lists, then grab and draw again"
                                     if blocked else "")
        self.check_button.setEnabled(drawn and self._current() is not None)

    def choice(self) -> dict:
        """``{'reference': table path or fraction, 'label': str, 'kind': 'band'|'row'}``."""
        data = self.reference.currentData()
        if data == CONSTANT:
            pct = float(self.percent.value())
            return {'reference': pct / 100.0, 'label': f"constant {pct:g} %", 'kind': self.kind.currentData()}
        return {'reference': data, 'label': Path(data).name, 'kind': self.kind.currentData()}
