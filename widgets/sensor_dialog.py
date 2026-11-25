# -*- coding: utf-8 -*-
"""Sensor settings of the live camera: exposure, gain and bin.

The camera takes them when the pipeline is rebuilt (``camera_source.open_camera`` applies
``acquisition.exposure_ms`` / ``gain`` / ``camera_bin``), so the dark, Smile and wavelength files
are selected again for the new values. The dialog shows which dark file the values will use:
without one the Dark step is off, and the white standard / reflectance would be wrong.
"""
from __future__ import annotations

import json
from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                             QGridLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from modules.dark import DarkFrameLibrary
from sim_config import DEFAULT_CONFIG_PATH


class SensorForm(QWidget):
    """Exposure, gain, bin and what they select (dark, reflectance), with an Apply button.

    ``compact``: the Setup tab's layout (values in one row); else the dialog's form.
    ``apply_requested`` is emitted by Apply; :meth:`values` / :meth:`save` as in the dialog.
    """

    apply_requested = pyqtSignal()

    def __init__(self, config, plan, config_path=None, compact: bool = False, cooling=None, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._cooling = dict(cooling or {})              # LiveSource.cooling of the running camera
        self._backend = plan.camera_backend or config.acquisition.camera_backend
        self._config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        from sim_config import calibration_folders
        self._plan = plan
        # the camera's dark folder, falling back to the next one per setting (as build_plan does)
        folders = [(f, label) for f, label in calibration_folders(config, plan, 'dark') if f.is_dir()]
        self._dark_where = ' > '.join(label for _, label in folders) or 'no dark folder'
        self._libraries = [DarkFrameLibrary(str(f)) for f, _ in folders]
        self._library = self._libraries[0] if self._libraries else None
        seen, self._darks = set(), []
        for library in self._libraries:                  # a setting is taken from the first folder that has it
            items = library.list_keys()
            keys = {(item.get('bin'), item.get('exposure'), item.get('gain')) for item in items}
            self._darks += [item for item in items if (item.get('bin'), item.get('exposure'), item.get('gain'))
                            not in seen]
            seen |= keys

        self.exposure = QDoubleSpinBox()
        self.exposure.setRange(0.01, 60000.0)
        self.exposure.setDecimals(3)
        self.exposure.setSuffix(" ms")
        self.exposure.setValue(float(plan.exposure_ms or config.acquisition.exposure_ms or 1.0))
        self.exposure.setToolTip("Exposure time; the line rate of a free-running camera is at most 1000 / exposure")
        self.gain = QDoubleSpinBox()
        self.gain.setRange(0.0, 10000.0)
        self.gain.setDecimals(2)
        self.gain.setValue(float(plan.gain if plan.gain is not None else config.acquisition.gain or 1.0))
        self.gain.setToolTip("Gain in the camera's own units")
        self.camera_bin = QSpinBox()
        self.camera_bin.setRange(1, 8)
        self.camera_bin.setValue(int(plan.camera_bin or config.acquisition.camera_bin))
        self.camera_bin.setToolTip("Hardware bin; Smile and wavelength files exist per bin")
        self._build_cooler(config)
        self.preset = QComboBox()
        self.preset.addItem("— pick a stored dark —", None)
        for key in sorted(self._darks, key=lambda k: (k['bin'], k['exposure'], k['gain'])):
            self.preset.addItem(f"bin {key['bin']} · {key['exposure']:g} ms · gain {key['gain']:g}  ({key['filename']})",
                                (key['bin'], key['exposure'], key['gain']))
        self.preset.setToolTip(f"Values for which a dark frame is stored ({self._dark_where})")
        self.preset.currentIndexChanged.connect(self._use_preset)
        self.dark_note = QLabel("")
        self.dark_note.setWordWrap(True)
        self.refl_note = QLabel("")
        self.refl_note.setWordWrap(True)
        self.rate_note = QLabel("")
        self.rate_note.setObjectName("metricCaption")
        self.save_box = QCheckBox(f"Save to {self._config_path.name}")
        self.save_box.setToolTip("Write acquisition.exposure_ms / gain / camera_bin (and "
                                 "live.target_temperature_c) into the config file (otherwise only for this run)")
        note = QLabel("Apply restarts the pipeline (camera closed and opened with the new values); the dark, "
                      "Smile and wavelength files follow them. Reflectance factors made at another exposure "
                      "are scaled to the new one.")
        note.setWordWrap(True)
        note.setObjectName("metricCaption")
        self.apply_button = QPushButton("Apply (restart)")
        self.apply_button.setToolTip(note.text())
        self.apply_button.clicked.connect(self.apply_requested.emit)
        for label in (self.dark_note, self.refl_note):
            label.setObjectName("rowSource")

        if compact:
            grid = QGridLayout(self)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(8)
            grid.setVerticalSpacing(4)
            for column, (text, widget) in enumerate((("Exposure", self.exposure), ("Gain", self.gain),
                                                     ("Bin", self.camera_bin), ("Cooler target", self.cooler_row))):
                caption = QLabel(text)
                caption.setObjectName("fieldLabel")
                caption.setVisible(self._cooler_supported or widget is not self.cooler_row)
                grid.addWidget(caption, 0, column)
                grid.addWidget(widget, 1, column)
            grid.addWidget(self.apply_button, 1, 4)
            stored = QLabel("Stored darks")
            stored.setObjectName("fieldLabel")
            grid.addWidget(stored, 0, 5)
            grid.addWidget(self.preset, 1, 5)
            grid.setColumnStretch(5, 1)
            grid.addWidget(self.dark_note, 2, 0, 1, 6)
            grid.addWidget(self.refl_note, 3, 0, 1, 6)
            grid.addWidget(self.cooler_note, 4, 0, 1, 6)
            grid.addWidget(self.rate_note, 5, 0, 1, 4)
            grid.addWidget(self.save_box, 5, 4, 1, 2)
        else:
            form = QFormLayout()
            form.addRow("Stored darks", self.preset)
            form.addRow("Exposure", self.exposure)
            form.addRow("Gain", self.gain)
            form.addRow("Bin", self.camera_bin)
            if self._cooler_supported:
                form.addRow("Cooler target", self.cooler_row)
            form.addRow("", self.cooler_note)
            form.addRow("Dark frame", self.dark_note)
            form.addRow("Reflectance", self.refl_note)
            form.addRow("", self.rate_note)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addLayout(form)
            layout.addWidget(note)
            layout.addWidget(self.save_box)
            self.apply_button.setVisible(False)          # the dialog has its own buttons
        for widget in (self.exposure, self.gain):
            widget.valueChanged.connect(self._check)
        self.camera_bin.valueChanged.connect(self._check)
        self._check()

    def _build_cooler(self, config) -> None:
        """Cooler on/off + target °C, shown only for a camera with a settable target (TEC).

        Off = ``live.target_temperature_c`` null: the cooler is left as the camera has it (it is
        not switched off), as hsi does without a configured target."""
        cooling = self._cooling
        self._cooler_supported = bool(cooling.get('supported'))
        configured = getattr(getattr(config, 'live', None), 'target_temperature_c', None)
        self.cooler_on = QCheckBox("On")
        self.cooler_on.setToolTip("Set the TEC cooler target at every camera start (live.target_temperature_c). "
                                  "Off: the cooler is left as the camera has it (not switched off)")
        self.cooler_target = QDoubleSpinBox()
        self.cooler_target.setDecimals(1)
        self.cooler_target.setSuffix(" °C")
        low, high = cooling.get('range') or (-100.0, 60.0)
        self.cooler_target.setRange(float(low), float(high))
        start = configured if configured is not None else cooling.get('target_c')
        self.cooler_target.setValue(float(start) if start is not None else min(max(0.0, low), high))
        self.cooler_target.setToolTip("Sensor temperature the cooler regulates to; darks are matched by "
                                      "sensor temperature, so keep it where the darks were recorded")
        self.cooler_on.setChecked(configured is not None)
        self.cooler_target.setEnabled(configured is not None)
        self.cooler_on.toggled.connect(self.cooler_target.setEnabled)
        self.cooler_row = QWidget(self)
        row = QHBoxLayout(self.cooler_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.cooler_on)
        row.addWidget(self.cooler_target, 1)
        self.cooler_row.setVisible(self._cooler_supported)
        self.cooler_note = QLabel(self._cooler_text())
        self.cooler_note.setObjectName("rowSource")
        self.cooler_note.setWordWrap(True)
        self.cooler_note.setVisible(bool(self.cooler_note.text()))

    def _cooler_text(self) -> str:
        cooling = self._cooling
        if cooling.get('error'):
            return f"⚠ cooler target not applied: {cooling['error']}"
        if not cooling.get('supported'):
            return ''
        parts = []
        if cooling.get('target_c') is not None:
            parts.append(f"camera target {cooling['target_c']:.1f} °C")
        if cooling.get('state'):
            parts.append(f"state {cooling['state']}")
        if cooling.get('requested') is None:
            parts.append("not set by this program (live.target_temperature_c is null)")
        return "Cooler: " + ", ".join(parts) if parts else ''

    def _use_preset(self) -> None:
        data = self.preset.currentData()
        if data:
            self.camera_bin.setValue(int(data[0]))
            self.exposure.setValue(float(data[1]))
            self.gain.setValue(float(data[2]))

    def _check(self) -> None:
        values = self.values()
        found = next((path for path in (library.find(values['camera_bin'], values['exposure_ms'], values['gain'])
                                        for library in self._libraries) if path), None)
        if found:
            self.dark_note.setText(f"✓ {Path(found).name}")
        else:
            self.dark_note.setText(f"⚠ no dark for bin {values['camera_bin']}, {values['exposure_ms']:g} ms, gain "
                                   f"{values['gain']:g}: the Dark step will be off and reflectance will be wrong. "
                                   "Record a dark for these values or pick a stored one.")
        self.refl_note.setText(self._reflectance_text(values))
        self.rate_note.setText(f"max line rate by exposure: {1000.0 / values['exposure_ms']:,.0f} lines/s")

    def _reflectance_text(self, values) -> str:
        from dataclasses import replace
        from sim_config import profile_reflectance_path
        if self._plan.camera_id is None:
            return "camera has no model/serial: reflectance/<source>.npy is used"
        hsi = getattr(self._plan, 'hsi_profile_dir', None)
        if hsi is not None and (hsi / 'radiometric').is_dir():
            return ("hsi radiometric reference of the copied camera folder: used when one of its active "
                    "references was made at exactly this bin, exposure and gain (not scaled); see "
                    "Reflectance in the processing steps after the restart")
        path = profile_reflectance_path(replace(self._plan, camera_bin=values['camera_bin'], gain=values['gain']))
        if path is not None and path.is_file():
            misfit = self._band_misfit(path, values)
            if misfit:
                return f"⚠ {path.name} is there but does not fit: {misfit}. Calibrate again (Reflectance...)"
            return f"✓ {path.name} (scaled to the exposure)"
        return (f"⚠ no {path.name if path else 'factors'} for bin {values['camera_bin']}, gain {values['gain']:g}: "
                "Reflectance will be off until you calibrate (Reflectance...)")

    def _band_misfit(self, path: Path, values) -> str:
        """Why stored factors do not fit the bands the model gets now ('' = they fit or unknown).

        Only checked at the running bin: another bin has other bands, known after the restart."""
        if int(values['camera_bin']) != int(self._plan.camera_bin or 0) or self._plan.wavelengths is None:
            return ''
        sidecar = path.with_name(path.stem + '.wavelength.npy')
        try:
            import numpy as np
            factors = np.load(path, mmap_mode='r')
            stored = np.load(sidecar) if sidecar.is_file() else None
        except (OSError, ValueError):
            return ''
        now = np.asarray(self._plan.wavelengths, np.float64)
        if factors.shape[-1] == now.size and (stored is None or np.allclose(stored, now, atol=0.5)):
            return ''
        made = (f"{factors.shape[-1]} bands" + (f" {stored[0]:.0f}–{stored[-1]:.0f} nm" if stored is not None
                                                 and stored.size else ''))
        return f"{made}, the input has {now.size} bands {now[0]:.0f}–{now[-1]:.0f} nm"

    def values(self) -> dict:
        """The form's values; ``target_temperature_c`` is only there for a camera with a cooler
        (``None`` = leave the cooler as the camera has it)."""
        values = {'exposure_ms': float(self.exposure.value()), 'gain': float(self.gain.value()),
                  'camera_bin': int(self.camera_bin.value()), 'save': self.save_box.isChecked()}
        if self._cooler_supported:
            values['target_temperature_c'] = (float(self.cooler_target.value()) if self.cooler_on.isChecked()
                                              else None)
        return values

    def save(self, values: dict) -> None:
        """Write the values into ``acquisition`` (and ``live``) of the config file, other keys untouched."""
        path = self._config_path
        data = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
        acquisition = data.setdefault('acquisition', {})
        acquisition.update(exposure_ms=values['exposure_ms'], gain=values['gain'], camera_bin=values['camera_bin'])
        if 'target_temperature_c' in values:
            data.setdefault('live', {})['target_temperature_c'] = values['target_temperature_c']
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding='utf-8')


class SensorDialog(QDialog):
    """:class:`SensorForm` in a modal window with Apply (restart) / Cancel."""

    def __init__(self, parent, config, plan, config_path=None, cooling=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sensor settings")
        self.setModal(True)
        self.form = SensorForm(config, plan, config_path, cooling=cooling)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttons.addButton("Apply (restart)", QDialogButtonBox.AcceptRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.form)
        layout.addWidget(buttons)
        self.setMinimumWidth(480)

    def values(self) -> dict:
        return self.form.values()

    def save(self, values: dict) -> None:
        self.form.save(values)
