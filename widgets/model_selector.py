# -*- coding: utf-8 -*-
"""Model selector for the operator: one large tile per model of the ``models`` folder.

Clicking a tile asks the owner (``chosen``) to switch the running pipeline to that model; the
tile of the model in use is highlighted. A model that does not fit the current input is shown
too, greyed out, with the reason in its tooltip. Appearance comes from ``themes/app.qss``
(``#modelTile`` ...).
"""
from __future__ import annotations

import re
from pathlib import Path

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtWidgets import (QButtonGroup, QGridLayout, QHBoxLayout, QLabel, QMenu, QPushButton, QSizePolicy,
                             QVBoxLayout, QWidget)

#: Tiles per row (the side panel is at least 260 px wide: two tiles keep the names readable).
COLUMNS = 2
TILE_HEIGHT = 66          # smallest tile; a long name makes its row taller

SWITCH_NOTE = ("Click to switch to this model at the next camera line: nothing stops, the class "
               "statistics and the belt map's classes start again.")


def display_name(path) -> str:
    """A model file's name for a tile: no extension, no trailing export number, spaces for ``_``."""
    stem = Path(str(path)).stem
    stem = re.sub(r'[_-]\d{6,}$', '', stem) or stem
    return stem.replace('_', ' ').strip() or stem


class ModelTile(QPushButton):
    """One model: name, kind (algorithm and classes, or why it cannot be used)."""

    def __init__(self, entry, parent=None) -> None:
        super().__init__(parent)
        self.entry = entry
        self.setObjectName("modelTile")
        self.setProperty('session', bool(getattr(entry, 'session', False)))   # dashed border: not a file
        self.setCheckable(True)
        self.setMinimumHeight(TILE_HEIGHT)
        policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        policy.setHeightForWidth(True)                 # a button ignores its layout: see heightForWidth
        self.setSizePolicy(policy)
        self.setEnabled(bool(entry.usable))
        self.setCursor(Qt.PointingHandCursor if entry.usable else Qt.ArrowCursor)
        self.setToolTip(entry.detail + (f"\n\n{SWITCH_NOTE}" if entry.usable else ''))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(2)
        self.name_label = QLabel(display_name(entry.path))
        self.name_label.setObjectName("modelTileName")
        self.name_label.setWordWrap(True)
        self.kind_label = QLabel(getattr(entry, 'kind', '') or '')
        self.kind_label.setObjectName("modelTileKind")
        # usable, but through interpolation onto the model's bands: the tile says so in the warning colour
        self.kind_label.setProperty('resampled', getattr(entry, 'resampled_nm', None) is not None)
        self.kind_label.setWordWrap(True)
        for label in (self.name_label, self.kind_label):
            label.setAttribute(Qt.WA_TransparentForMouseEvents)      # the click belongs to the tile
            layout.addWidget(label)
        layout.addStretch()
        self.setAccessibleName(f"Model {self.name_label.text()}")

    def set_name(self, name: str) -> None:
        self.name_label.setText(name)
        self.setAccessibleName(f"Model {name}")
        self.updateGeometry()

    # QPushButton sizes itself from its own (empty) text; the labels decide the height here.
    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return max(TILE_HEIGHT, self.layout().heightForWidth(width))

    def sizeHint(self) -> QSize:
        return QSize(120, TILE_HEIGHT)

    def minimumSizeHint(self) -> QSize:
        return QSize(60, TILE_HEIGHT)


class ModelSelector(QWidget):
    """Tiles of the models found, the one in use highlighted."""

    #: the entry (``model_catalog.ModelEntry``) of the tile the operator clicked
    chosen = pyqtSignal(object)
    rescan_requested = pyqtSignal()
    #: the entry of a session model the operator wants gone (right click on its tile)
    remove_requested = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.tiles: list[ModelTile] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        header = QHBoxLayout()
        title = QLabel("MODEL")
        title.setObjectName("caption")
        header.addWidget(title)
        self._note = QLabel("")                            # scanning / none found / model from elsewhere
        self._note.setObjectName("metricCaption")
        # a long note is clipped (full text in its tooltip): it must not widen the whole tab
        self._note.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        header.addWidget(self._note, stretch=1)
        self.rescan_button = QPushButton("↻")
        self.rescan_button.setObjectName("toggleButton")
        self.rescan_button.setCursor(Qt.PointingHandCursor)
        self.rescan_button.setToolTip("Look for models again (the folder models)")
        self.rescan_button.clicked.connect(self.rescan_requested)
        header.addWidget(self.rescan_button)
        layout.addLayout(header)
        self._grid = QGridLayout()
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(8)
        for column in range(COLUMNS):
            self._grid.setColumnStretch(column, 1)
        layout.addLayout(self._grid)
        self._group = QButtonGroup(self)
        self._group.setExclusive(False)                    # the highlight is set by set_current only

    def _clear(self) -> None:
        for tile in self.tiles:
            self._group.removeButton(tile)
            self._grid.removeWidget(tile)
            tile.setParent(None)
            tile.deleteLater()
        self.tiles = []

    def set_scanning(self) -> None:
        """While the models are loaded and bound in the background."""
        self._clear()
        self._note.setText("looking for models…")
        self.rescan_button.setEnabled(False)

    def set_entries(self, entries) -> None:
        self._clear()
        entries = list(entries)
        names = [display_name(entry.path) for entry in entries]
        for index, entry in enumerate(entries):
            tile = ModelTile(entry)
            if names.count(names[index]) > 1:          # two exports of one model, x.zip beside x.hsipkg
                tile.set_name(Path(str(entry.path)).name.replace('_', ' '))   # full name, can wrap
            tile.clicked.connect(lambda _=False, t=tile: self._on_clicked(t))
            if getattr(entry, 'session', False):
                tile.setContextMenuPolicy(Qt.CustomContextMenu)
                tile.customContextMenuRequested.connect(lambda pos, t=tile: self._session_menu(t, pos))
            self._group.addButton(tile)
            self._grid.addWidget(tile, index // COLUMNS, index % COLUMNS)
            self.tiles.append(tile)
        self._note.setText('')
        self._note.setToolTip('')
        self.rescan_button.setEnabled(True)

    def set_current(self, classifier) -> None:
        """Highlight the tile of the pipeline's model; name it when it is not one of the tiles."""
        if not self.rescan_button.isEnabled():             # still scanning: the note says so
            return
        found = False
        for tile in self.tiles:
            current = tile.entry.classifier is not None and tile.entry.classifier is classifier
            tile.setChecked(current)
            found = found or current
        if found:
            self._note.setText('')
            self._note.setToolTip('')
        else:
            running = f"running: {display_name(getattr(classifier, 'path', ''))}"
            self._note.setText(f"{running} (not in models)" if self.tiles else f"{running} · folder models is empty")
            self._note.setToolTip(f"{self._note.text()}\n{getattr(classifier, 'path', '')}")

    def _session_menu(self, tile: ModelTile, pos) -> None:
        menu = QMenu(self)
        remove = menu.addAction("Remove this session model")
        if menu.exec_(tile.mapToGlobal(pos)) is remove:
            self.remove_requested.emit(tile.entry)

    def _on_clicked(self, tile: ModelTile) -> None:
        # the owner decides (and calls set_current): a click alone does not move the highlight
        tile.setChecked(not tile.isChecked())
        self.chosen.emit(tile.entry)
