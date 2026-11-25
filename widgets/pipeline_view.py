"""Pipeline view of the Setup tab: one block per step from the sensor to the model.

Each block shows its state (coloured border and dot), its time per line and one short line (the
second goes in the tooltip);
a click shows the block's detail rows, the note on what to do and its action buttons below.
The content comes from ``pipeline_overview.describe``; an action button emits
:attr:`PipelineView.action_requested` with the action id (``toggle:<step>``, ``sensor``,
``reflectance``, ``spatial_bin``, ``model``), the window does the work.
"""
from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout,
                             QWidget)

from themes import repolish

_STATE_TEXT = {'ok': 'ok', 'warn': 'check', 'error': 'blocks the model', 'off': 'off'}


def drop_widget(widget) -> None:
    """Remove a widget from view now; ``deleteLater`` alone leaves it drawn until the next pass."""
    widget.hide()
    widget.setParent(None)
    widget.deleteLater()


_drop = drop_widget


def _set_state(widget, state: str) -> None:
    if widget.property('state') != state:
        widget.setProperty('state', state)
        repolish(widget)


def _elided(label: QLabel, text: str) -> None:
    """Text in a fixed-width block: cut with … and the whole text as tooltip."""
    label.setProperty('full', text)
    metrics = label.fontMetrics()
    width = max(20, label.width() or 120)
    label.setText(metrics.elidedText(text, Qt.ElideRight, width))
    label.setToolTip(text if metrics.horizontalAdvance(text) > width else '')


class _BlockWidget(QFrame):
    clicked = pyqtSignal(str)

    def __init__(self, key: str) -> None:
        super().__init__()
        self.key = key
        self.setObjectName('pipelineBlock')            # full border in the state's colour (app.qss)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumWidth(80)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # Compact: title, time and dot on one row, one short line below; the second line and the
        # time's source go in the tooltip (the detail panel holds everything).
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 7, 4)
        layout.setSpacing(0)
        head = QHBoxLayout()
        head.setSpacing(5)
        self.title = QLabel()
        self.title.setObjectName('blockTitle')
        self.dot = QLabel('●')
        self.dot.setObjectName('blockDot')
        self.time = QLabel()
        self.time.setObjectName('blockTime')
        head.addWidget(self.title, stretch=1)
        head.addWidget(self.time)
        head.addWidget(self.dot)
        layout.addLayout(head)
        self.line1, self.line2 = QLabel(), QLabel()
        for label in (self.line1, self.line2):
            label.setObjectName('blockLine')
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout.addWidget(self.line1)
        self.line2.setVisible(False)                   # kept for its text (tooltip, tests)

    def show_block(self, block, selected: bool) -> None:
        self.title.setText(block.title)
        _set_state(self.dot, block.state)
        self.dot.setToolTip(_STATE_TEXT[block.state])
        _elided(self.line1, block.line1)
        self.line2.setText(block.line2)
        self.time.setText(f"{block.ms:.2f} ms" if block.ms is not None else '')
        tip = [t for t in (block.line1, block.line2, 'always on' if block.locked else '') if t]
        self.setToolTip(f"{block.title}: " + ' · '.join(tip) if tip else block.title)
        _set_state(self, block.state)
        if self.property('selected') != selected:
            self.setProperty('selected', selected)
            repolish(self)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.line1.property('full') is not None:
            _elided(self.line1, self.line1.property('full'))

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.key)
        super().mousePressEvent(event)


class PipelineView(QWidget):
    """Status bar and blocks Sensor → … → Model; :attr:`detail` (the selected block) is placed
    by the window where it wants it."""

    action_requested = pyqtSignal(str)
    selected_changed = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName('pipelineView')
        self._blocks: dict[str, _BlockWidget] = {}
        self._data: dict = {}
        self._order: list[str] = []
        self.selected: str | None = None
        self._shown_detail = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        head = QHBoxLayout()
        caption = QLabel('Model receives')
        caption.setObjectName('metricCaption')
        self.input_label = QLabel()
        self.input_label.setObjectName('chip')
        self.input_label.setContentsMargins(10, 2, 10, 2)
        self.summary_label = QLabel()
        self.summary_label.setObjectName('pipelineSummary')
        self._status = QHBoxLayout()
        self._status.setSpacing(14)
        self.status_bar = QFrame()
        self.status_bar.setObjectName('setupPanel')
        head.setContentsMargins(10, 5, 10, 5)
        head.setSpacing(10)
        self.status_bar.setLayout(head)
        head.addWidget(caption)
        head.addWidget(self.input_label)
        head.addSpacing(8)
        head.addLayout(self._status)
        head.addStretch()
        head.addWidget(self.summary_label)
        layout.addWidget(self.status_bar)

        self._row = QHBoxLayout()
        self._row.setSpacing(4)
        layout.addLayout(self._row)

        self.detail = QFrame()
        self.detail.setObjectName('pipelineDetail')    # full border in the state's colour (app.qss)
        self.detail.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        detail = QVBoxLayout(self.detail)
        detail.setContentsMargins(12, 8, 12, 10)
        detail.setSpacing(6)
        title_row = QHBoxLayout()
        self.detail_title = QLabel()
        self.detail_title.setObjectName('detailTitle')
        self.detail_state = QLabel()
        self.detail_state.setObjectName('detailState')
        title_row.addWidget(self.detail_title)
        title_row.addStretch()
        title_row.addWidget(self.detail_state)
        detail.addLayout(title_row)
        self._grid = QGridLayout()
        self._grid.setHorizontalSpacing(16)
        self._grid.setVerticalSpacing(3)
        self._grid.setColumnStretch(1, 1)
        detail.addLayout(self._grid)
        self.note_label = QLabel()
        self.note_label.setObjectName('pipelineNote')
        self.note_label.setWordWrap(True)
        detail.addWidget(self.note_label)
        self._actions = QHBoxLayout()
        self._actions.setSpacing(6)
        detail.addLayout(self._actions)

    # ----------------------------------------------------------------------------------- data ---

    def set_blocks(self, headline: str, blocks: list) -> None:
        """Show ``pipeline_overview.describe``'s result; the selection is kept."""
        keys = [b.key for b in blocks]
        if keys != self._order:
            self._build_row(keys)
        self._data = {b.key: b for b in blocks}
        first = self.selected not in self._data
        if first:
            problem = next((b for b in blocks if b.state == 'error'), None) or \
                next((b for b in blocks if b.state == 'warn'), None)
            self.selected = (problem or blocks[0]).key
        for b in blocks:
            self._blocks[b.key].show_block(b, b.key == self.selected)
        receives_counts = not headline.startswith('Reflectance')
        model = self._data.get('model')
        self.input_label.setText(headline)
        _set_state(self.input_label, 'error' if model is not None and model.state == 'error' else
                   'warn' if receives_counts else 'ok')
        n_error = sum(b.state == 'error' for b in blocks)
        n_warn = sum(b.state == 'warn' for b in blocks)
        parts = [f"{n_error} blocking" if n_error else '', f"{n_warn} to check" if n_warn else '']
        self.summary_label.setText(' · '.join(p for p in parts if p) or 'all steps ok')
        _set_state(self.summary_label, 'error' if n_error else 'warn' if n_warn else 'ok')
        self._show_detail()
        if first:
            self.selected_changed.emit(self.selected)

    def set_status(self, items: list) -> None:
        """``[(text, state)]`` beside the headline: camera, line rate, lost lines, temperature."""
        while self._status.count() > len(items):
            item = self._status.takeAt(self._status.count() - 1)
            if item.widget() is not None:
                _drop(item.widget())
        while self._status.count() < len(items):
            label = QLabel()
            label.setObjectName('pipelineStatusItem')
            self._status.addWidget(label)
        for i, (text, state) in enumerate(items):
            label = self._status.itemAt(i).widget()
            label.setText(text)
            _set_state(label, state)

    def select(self, key: str) -> None:
        if key in self._data and key != self.selected:
            self.selected = key
            for k, w in self._blocks.items():
                w.show_block(self._data[k], k == key)
            self._show_detail()
            self.selected_changed.emit(key)

    def _build_row(self, keys: list) -> None:
        while self._row.count():
            item = self._row.takeAt(0)
            if item.widget() is not None:
                _drop(item.widget())
        self._blocks = {}
        for i, key in enumerate(keys):
            if i:
                arrow = QLabel('›')
                arrow.setObjectName('blockArrow')
                self._row.addWidget(arrow)
            block = _BlockWidget(key)
            block.clicked.connect(self.select)
            self._blocks[key] = block
            self._row.addWidget(block, stretch=1)
        self._order = list(keys)

    def _show_detail(self) -> None:
        block = self._data.get(self.selected)
        if block is None:
            return
        shown = (block.key, block.state, tuple(block.rows), block.note, tuple(block.actions))
        if shown == self._shown_detail:            # refreshed twice a second: rebuild only on change
            return
        self._shown_detail = shown
        self.detail_title.setText(block.title)
        self.detail_state.setText('● ' + _STATE_TEXT[block.state])
        _set_state(self.detail_state, block.state)
        _set_state(self.detail, block.state)
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget() is not None:
                _drop(item.widget())
        for row, (name, value) in enumerate(block.rows):
            key = QLabel(str(name))
            key.setObjectName('fieldLabel')
            text = QLabel(str(value))
            text.setObjectName('rowSource')
            text.setWordWrap(True)
            text.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._grid.addWidget(key, row, 0, Qt.AlignTop)
            self._grid.addWidget(text, row, 1)
        self.note_label.setText(block.note)
        self.note_label.setVisible(bool(block.note))
        _set_state(self.note_label, block.state)
        while self._actions.count():
            item = self._actions.takeAt(0)
            if item.widget() is not None:
                _drop(item.widget())
        for action, text in block.actions:
            button = QPushButton(text)
            button.setObjectName('quietButton')
            button.clicked.connect(lambda _, a=action: self.action_requested.emit(a))
            self._actions.addWidget(button)
        self._actions.addStretch()

    def detail_rows(self) -> list:
        """``[(label, value)]`` shown now (tests)."""
        block = self._data.get(self.selected)
        return list(block.rows) if block is not None else []
