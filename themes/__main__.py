# -*- coding: utf-8 -*-
"""Theme gallery: every reusable style on one page, to design or check a theme.

    python -m themes              # dark
    python -m themes light
    python -m themes path/to/mytheme.json

The theme box switches live; "Reload" re-reads app.qss and the JSON from disk, so a theme
can be edited and checked without restarting.
"""
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QProgressBar,
                             QPushButton, QVBoxLayout, QWidget)

from themes import ACCENTS, THEME_DIR, apply_theme, make_panel, set_accent_color, use_dark_title_bar


def accent_card(title: str, text: str, accent=None, alt=False) -> QFrame:
    card = make_panel(QFrame(), accent=accent, alt=alt)
    layout = QVBoxLayout(card)
    layout.setContentsMargins(12, 10, 12, 10)
    caption = QLabel(title.upper())
    caption.setObjectName("caption")
    value = QLabel(text)
    value.setObjectName("bigValue")
    layout.addWidget(caption)
    layout.addWidget(value)
    return card


class Gallery(QWidget):
    def __init__(self, app: QApplication, theme: str) -> None:
        super().__init__()
        self.app = app
        self.setWindowTitle("Theme gallery")
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        bar = QHBoxLayout()
        title = QLabel("Theme gallery")
        title.setObjectName("panelTitle")
        bar.addWidget(title)
        bar.addStretch()
        self.combo = QComboBox()
        for path in sorted(THEME_DIR.glob('*.json')):
            self.combo.addItem(path.stem, str(path))
        if self.combo.findText(theme) < 0:
            self.combo.addItem(theme, theme)
        self.combo.setCurrentText(theme)
        self.combo.currentIndexChanged.connect(lambda _: self.reload())
        bar.addWidget(self.combo)
        reload_button = QPushButton("Reload")
        reload_button.clicked.connect(self.reload)
        bar.addWidget(reload_button)
        danger = QPushButton("Danger button")
        danger.setObjectName("dangerButton")
        bar.addWidget(danger)
        root.addLayout(bar)

        # Accent stripes: one card per accent + a data colour.
        grid = QGridLayout()
        grid.setSpacing(10)
        for i, accent in enumerate(ACCENTS):
            grid.addWidget(accent_card(f'accent="{accent}"', accent.upper(), accent), 0, i)
        data = accent_card('set_accent_color', 'class colour')
        data.setObjectName("galleryData")
        set_accent_color(data, (255, 127, 14))
        grid.addWidget(data, 1, 0)
        grid.addWidget(accent_card('panel alt', 'no accent', alt=True), 1, 1)
        grid.addWidget(accent_card('panel alt', 'info', 'info', alt=True), 1, 2)
        root.addLayout(grid)

        # Labels with the main-window roles.
        for accent, text in (('ok', 'RUN | status line, accent="ok"'),
                             ('warn', 'lines lost — status line, accent="warn"'),
                             ('error', 'ERROR — status line, accent="error"')):
            label = make_panel(QLabel(text), accent=accent)
            label.setObjectName("statusLabel")
            root.addWidget(label)
        info = make_panel(QLabel("infoLabel: monospace provenance text\nkey: value  [origin]"), accent='info')
        info.setObjectName("infoLabel")
        root.addWidget(info)

        row = QHBoxLayout()
        for name, text in (('metricCaption', 'metricCaption'), ('metricRealtime', '42.0%'),
                           ('metricTotal', '1,234 (5.6%)'), ('className', 'className')):
            label = QLabel(text)
            label.setObjectName(name)
            row.addWidget(label)
        bar_widget = QProgressBar()
        bar_widget.setObjectName("classBar")
        bar_widget.setFixedHeight(4)
        bar_widget.setTextVisible(False)
        bar_widget.setValue(60)
        row.addWidget(bar_widget, stretch=1)
        root.addLayout(row)

        image = QLabel("imagePanel")
        image.setObjectName("imagePanel")
        image.setAlignment(Qt.AlignCenter)
        image.setMinimumHeight(80)
        root.addWidget(image)
        self.reload()

    def reload(self) -> None:
        tokens = apply_theme(self.app, self.combo.currentData())
        use_dark_title_bar(self, bool(tokens.get('dark')))


def main() -> None:
    app = QApplication(sys.argv[:1])
    gallery = Gallery(app, sys.argv[1] if len(sys.argv) > 1 else 'dark')
    gallery.resize(1000, 640)
    gallery.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
