"""Every theme defines every token the stylesheet uses; Python code sets no colours."""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from themes import THEME_DIR, load_theme  # noqa: E402


@pytest.mark.parametrize('theme', sorted(p.stem for p in THEME_DIR.glob('*.json')))
def test_theme_fills_every_token(theme):
    tokens, qss = load_theme(theme)
    assert '@' not in re.sub(r'/\*.*?\*/', '', qss, flags=re.DOTALL)
    assert 'QFrame[accent="error"]' in qss and tokens['danger'] in qss


def test_missing_token_is_reported(tmp_path):
    (tmp_path / 'broken.json').write_text('{"bg": "#000000"}')
    with pytest.raises(KeyError, match='lacks token'):
        load_theme(str(tmp_path / 'broken.json'))


@pytest.mark.parametrize('module', ['gui.py', 'widgets/class_legend_widget.py'])
def test_no_hard_coded_colours_in_widgets(module):
    source = (ROOT / module).read_text(encoding='utf-8')
    assert not re.findall(r'#[0-9A-Fa-f]{6}\b', source), "colours belong in themes/*.json"
