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

from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                             QLabel, QSpinBox, QVBoxLayout)

from modules.dark import DarkFrameLibrary
from sim_config import DEFAULT_CONFIG_PATH


class SensorDialog(QDialog):
    def __init__(self, parent, config, plan, config_path=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sensor settings")
        self.setModal(True)
        self._config = config
        self._backend = plan.camera_backend or config.acquisition.camera_backend
        self._config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        folder = config.calibration_dir / 'dark_frames' / self._backend
        self._library = DarkFrameLibrary(str(folder)) if folder.is_dir() else None
        self._darks = self._library.list_keys() if self._library else []

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
        self.preset = QComboBox()
        self.preset.addItem("— pick a stored dark —", None)
        for key in sorted(self._darks, key=lambda k: (k['bin'], k['exposure'], k['gain'])):
            self.preset.addItem(f"bin {key['bin']} · {key['exposure']:g} ms · gain {key['gain']:g}  ({key['filename']})",
                                (key['bin'], key['exposure'], key['gain']))
        self.preset.setToolTip("Values for which a dark frame is stored (calibration/dark_frames/"
                               f"{self._backend}/)")
        self.preset.currentIndexChanged.connect(self._use_preset)
        self.dark_note = QLabel("")
        self.dark_note.setWordWrap(True)
        self.rate_note = QLabel("")
        self.rate_note.setObjectName("metricCaption")
        self.save_box = QCheckBox(f"Save to {self._config_path.name}")
        self.save_box.setToolTip("Write acquisition.exposure_ms / gain / camera_bin into the config file "
                                 "(otherwise only for this run)")
        note = QLabel("Apply restarts the pipeline (camera closed and opened with the new values); the dark, "
                      "Smile and wavelength files follow them. Reflectance factors made at another exposure "
                      "are scaled to the new one.")
        note.setWordWrap(True)
        note.setObjectName("metricCaption")

        form = QFormLayout()
        form.addRow("Stored darks", self.preset)
        form.addRow("Exposure", self.exposure)
        form.addRow("Gain", self.gain)
        form.addRow("Bin", self.camera_bin)
        form.addRow("Dark frame", self.dark_note)
        form.addRow("", self.rate_note)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttons.addButton("Apply (restart)", QDialogButtonBox.AcceptRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(self.save_box)
        layout.addWidget(buttons)
        self.setMinimumWidth(480)
        for widget in (self.exposure, self.gain):
            widget.valueChanged.connect(self._check)
        self.camera_bin.valueChanged.connect(self._check)
        self._check()

    def _use_preset(self) -> None:
        data = self.preset.currentData()
        if data:
            self.camera_bin.setValue(int(data[0]))
            self.exposure.setValue(float(data[1]))
            self.gain.setValue(float(data[2]))

    def _check(self) -> None:
        values = self.values()
        found = self._library.find(values['camera_bin'], values['exposure_ms'], values['gain']) \
            if self._library else None
        if found:
            self.dark_note.setText(f"✓ {Path(found).name}")
        else:
            self.dark_note.setText(f"⚠ no dark for bin {values['camera_bin']}, {values['exposure_ms']:g} ms, gain "
                                   f"{values['gain']:g}: the Dark step will be off and reflectance will be wrong. "
                                   "Record a dark for these values or pick a stored one.")
        self.rate_note.setText(f"max line rate by exposure: {1000.0 / values['exposure_ms']:,.0f} lines/s")

    def values(self) -> dict:
        return {'exposure_ms': float(self.exposure.value()), 'gain': float(self.gain.value()),
                'camera_bin': int(self.camera_bin.value()), 'save': self.save_box.isChecked()}

    def save(self, values: dict) -> None:
        """Write the values into ``acquisition`` of the config file, other keys untouched."""
        path = self._config_path
        data = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
        acquisition = data.setdefault('acquisition', {})
        acquisition.update(exposure_ms=values['exposure_ms'], gain=values['gain'], camera_bin=values['camera_bin'])
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding='utf-8')
