# -*- coding: utf-8 -*-
"""GUI themes: one stylesheet template (``app.qss``) plus a colour-token JSON per theme.

``app.qss`` refers to tokens as ``@name``; ``<theme>.json`` gives their values. A new theme is
a copy of ``dark.json`` with other colours; layout/shape changes go into ``app.qss`` only.

    from themes import apply_theme
    theme = apply_theme(app, 'dark')          # or a path to a JSON file
    theme['danger']                           # tokens are available to Python code too

Reusable look for any QFrame/QLabel (see "panel + accent stripe" in ``app.qss``)::

    make_panel(label, accent='ok')            # card with a coloured stripe on the left
    set_accent(label, 'error')                # ok | warn | error | info | neutral | None
    set_accent_color(card, (214, 39, 40))     # stripe in a data colour (e.g. a class colour)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

THEME_DIR = Path(__file__).resolve().parent
STYLESHEET = THEME_DIR / 'app.qss'
_TOKEN = re.compile(r'@([A-Za-z_][A-Za-z0-9_]*)')


def theme_path(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.suffix.lower() != '.json':
        path = THEME_DIR / f'{name_or_path}.json'
    if not path.is_file():
        available = sorted(p.stem for p in THEME_DIR.glob('*.json'))
        raise FileNotFoundError(f"Theme {name_or_path!r} not found (available: {available})")
    return path


def load_theme(name_or_path: str = 'dark', stylesheet: Path = STYLESHEET) -> tuple[dict, str]:
    """``(tokens, qss)``: the theme's tokens and the stylesheet with every ``@token`` filled in.

    Raises:
        FileNotFoundError: no such theme.
        KeyError: the stylesheet uses a token the theme does not define.
    """
    tokens = json.loads(theme_path(name_or_path).read_text(encoding='utf-8'))
    template = Path(stylesheet).read_text(encoding='utf-8')
    template = re.sub(r'/\*.*?\*/', '', template, flags=re.DOTALL)     # comments may mention @names
    missing = sorted({m for m in _TOKEN.findall(template) if m not in tokens})
    if missing:
        raise KeyError(f"Theme {name_or_path!r} lacks token(s) {missing} used in {Path(stylesheet).name}")
    return tokens, _TOKEN.sub(lambda m: str(tokens[m.group(1)]), template)


def apply_theme(app, name_or_path: str = 'dark') -> dict:
    """Style the whole application (Fusion + palette for native dialogs + stylesheet)."""
    from PyQt5.QtGui import QColor, QPalette
    from PyQt5.QtWidgets import QStyleFactory

    tokens, qss = load_theme(name_or_path)
    app.setStyle(QStyleFactory.create('Fusion'))
    palette = QPalette()
    for role, key in ((QPalette.Window, 'bg'), (QPalette.Base, 'surface'), (QPalette.AlternateBase, 'surface_alt'),
                      (QPalette.Button, 'surface_alt'), (QPalette.ToolTipBase, 'surface_alt'),
                      (QPalette.WindowText, 'text'), (QPalette.Text, 'text'), (QPalette.ButtonText, 'text'),
                      (QPalette.ToolTipText, 'text'), (QPalette.PlaceholderText, 'text_faint'),
                      (QPalette.Highlight, 'selection'), (QPalette.HighlightedText, 'on_accent'),
                      (QPalette.Link, 'accent')):
        palette.setColor(role, QColor(tokens[key]))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(tokens['text_faint']))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(tokens['text_faint']))
    app.setPalette(palette)
    app.setStyleSheet(qss)
    app.setProperty('theme_tokens', tokens)
    return tokens


def use_dark_title_bar(window, dark: bool) -> None:
    """Windows 10/11: dark window frame to match a dark theme (no effect elsewhere)."""
    import sys
    if sys.platform != 'win32':
        return
    try:
        import ctypes
        value = ctypes.c_int(1 if dark else 0)
        for attribute in (20, 19):          # DWMWA_USE_IMMERSIVE_DARK_MODE (new, old builds)
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(int(window.winId()), attribute,
                                                          ctypes.byref(value), ctypes.sizeof(value)) == 0:
                break
    except Exception:
        pass


def repolish(widget) -> None:
    """Re-apply the stylesheet after a dynamic property (e.g. ``state``) changed."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)


ACCENTS = ('ok', 'warn', 'error', 'info', 'neutral')


def make_panel(widget, accent: str | None = None, alt: bool = False):
    """Give a QFrame/QLabel the theme's panel look (surface, border, rounded corners)."""
    widget.setProperty('panel', 'alt' if alt else True)
    set_accent(widget, accent)
    return widget


def set_accent(widget, accent: str | None) -> None:
    """Coloured stripe on the left edge: ``ok``/``warn``/``error``/``info``/``neutral``, or None."""
    if accent is not None and accent not in ACCENTS:
        raise ValueError(f"accent must be one of {ACCENTS} or None, got {accent!r}")
    value = accent or ''
    if widget.property('accent') != value:
        widget.setProperty('accent', value)
        repolish(widget)


def set_accent_color(widget, rgb) -> None:
    """Stripe in an arbitrary colour (data such as a class colour, not a theme colour).

    Uses a selector on the widget's own object name, so child widgets are not affected.
    """
    name = widget.objectName()
    if not name:
        raise ValueError("set_accent_color needs a widget with an objectName")
    r, g, b = (int(v) for v in rgb[:3])
    widget.setStyleSheet(f"#{name} {{ border-left: 3px solid rgb({r}, {g}, {b}); }}")
