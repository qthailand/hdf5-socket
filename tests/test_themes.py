"""Every theme defines every token the stylesheet uses; Python code sets no colours."""
import numpy as np
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


@pytest.mark.parametrize('theme', sorted(p.stem for p in THEME_DIR.glob('*.json')))
def test_theme_has_chart_tokens(theme):
    tokens, _ = load_theme(theme)
    for key in ('chart_1', 'chart_2', 'chart_3', 'chart_4', 'chart_5', 'chart_grid', 'chart_budget'):
        assert key in tokens, f"{theme}.json lacks {key} (used by widgets/performance_panel.py)"


@pytest.mark.parametrize('module', ['gui.py', 'widgets/class_legend_widget.py', 'widgets/performance_panel.py'])
def test_no_hard_coded_colours_in_widgets(module):
    source = (ROOT / module).read_text(encoding='utf-8')
    assert not re.findall(r'#[0-9A-Fa-f]{6}\b', source), "colours belong in themes/*.json"


def test_performance_panel_reads_a_snapshot():
    from types import SimpleNamespace
    from PyQt5.QtWidgets import QApplication
    from themes import apply_theme
    from widgets.performance_panel import PerformancePanel
    app = QApplication.instance() or QApplication([])
    apply_theme(app, 'dark')
    panel = PerformancePanel()
    snap = SimpleNamespace(
        stage_ms={'queue': 0.3, 'dark': 0.1, 'hot_pixel': 0.1, 'orient_smile': 1.2, 'preprocess': 1.4,
                  'bin': 0.3, 'predict': 0.2, 'total': 2.0},
        stage_stats={'cpu': {'avg': {'total': 2.1, 'dark': 0.1}, 'peak': {'total': 2.6}},
                     'cupy_cuda': {'avg': {'total': 0.4}, 'peak': {'total': 0.5}}},
        backend='cpu', workers=4, target_fps=1000.0, camera_fps=1000.0, processed_fps=1000.0,
        latency_ms={'p50': 1.0, 'p99': 3.0, 'max': 5.0}, frames_delivered=1000, lines_lost=0)
    panel.update_snapshot(snap)
    chart = panel.stage_chart
    assert 'Queue wait' not in chart._categories and chart._categories[-1] == 'Total / line'
    assert chart._categories[0] == 'Dark + Hot pixel' and abs(chart._series[0][2][0] - 0.2) < 1e-9
    assert [name for name, _, _ in chart._series] == ['Now', 'Avg cpu', 'Avg cupy_cuda', 'Peak cpu']
    assert chart._markers[0][1] == 4.0                         # budget = 4 workers x 1000 / 1000 lines/s
    assert panel.rate_chart._series[0][2][3] == 2000.0         # capacity = 4 x 1000 / 2.0 ms
    assert panel.kpi_headroom.value.text() == '2.00 ×'
    assert panel.latency_chart._categories[0] == 'Queue wait'
    panel.resize(320, 900)
    panel.grab()                                               # paints without error


def test_throughput_strip_shows_fps_capacity_and_loss():
    from types import SimpleNamespace
    from PyQt5.QtWidgets import QApplication
    from themes import apply_theme
    from widgets.performance_panel import ThroughputStrip
    app = QApplication.instance() or QApplication([])
    apply_theme(app, 'dark')
    strip = ThroughputStrip()
    snap = SimpleNamespace(state='running', target_fps=250.0, camera_fps=250.0, processed_fps=200.0, workers=4,
                           stage_ms={'total': 2.5}, frames_delivered=990, lines_lost=10,
                           latency_ms={'p50': 3.5, 'p99': 5.5, 'max': 9.0})
    strip.update_snapshot(snap)
    assert strip.camera.value.text() == '250  / 250' and strip.processed.value.text() == '200'
    assert strip.processed.property('accent') == 'warn'                   # behind the line rate
    assert strip.capacity.value.text() == '1,600  (6.4×)'                 # 4 x 1000 / 2.5 ms
    assert strip.lost.value.text() == '10  (1.00 %)' and strip.lost.property('accent') == 'error'


def test_materials_tab_shows_class_shares_as_a_bar_chart():
    from PyQt5.QtWidgets import QApplication
    from themes import apply_theme
    from utils import generate_tab10_lut
    from widgets.class_legend_widget import ClassLegendWidget
    app = QApplication.instance() or QApplication([])
    apply_theme(app, 'dark')
    legend = ClassLegendWidget()
    legend.setup_classes_from_model({0: 'mango', 1: 'pradoo'}, generate_tab10_lut(), np.array([0, 1]))
    recent, total = np.zeros(256, np.int64), np.zeros(256, np.int64)
    recent[:2], total[:2] = (1, 3), (30, 10)
    legend.update_counts(recent, total, 4)
    names, series = legend.chart._categories, legend.chart._series
    assert names == ['mango', 'pradoo'] and [s[0] for s in series] == ['Real-time', 'Accumulated']
    np.testing.assert_allclose(series[0][2], [25.0, 75.0])
    np.testing.assert_allclose(series[1][2], [75.0, 25.0])
    assert series[0][1][0].alpha() == 255 and series[1][1][0].alpha() < 255     # accumulated: faded
    assert legend.cards_map[0][2].text() == '30 px'
    legend.resize(300, 500)
    legend.grab()                                                                # paints without error
    legend.reset_accumulated_stats()
    assert list(legend.chart._series[1][2]) == [0.0, 0.0]


def test_materials_background_hidden_by_default_and_shares_over_shown_classes():
    from PyQt5.QtWidgets import QApplication, QCheckBox
    from themes import apply_theme
    from utils import generate_tab10_lut
    from widgets.class_legend_widget import ClassLegendWidget
    app = QApplication.instance() or QApplication([])
    apply_theme(app, 'dark')
    legend = ClassLegendWidget()
    legend.setup_classes_from_model({0: 'mango', 1: 'katom', 2: 'BG'}, generate_tab10_lut(), np.array([0, 1, 2]))
    counts = np.zeros(256, np.int64)
    counts[:3] = (10, 30, 60)
    legend.update_counts(counts, counts, 1)
    names, realtime, accumulated = legend.shares()
    assert names == ['mango', 'katom']                                  # 'BG' hidden (case-insensitive)
    np.testing.assert_allclose(accumulated, [25.0, 75.0])               # over the shown classes
    assert legend.chart.unit == '% of shown classes' and legend.cards_map[2][2].text() == '60 px'
    switches = legend.findChildren(QCheckBox)
    assert [c.isChecked() for c in switches] == [True, True, False]
    switches[2].setChecked(True)                                        # operator shows it again
    names, _, accumulated = legend.shares()
    assert names == ['mango', 'katom', 'BG'] and legend.chart.unit == '% of pixels'
    np.testing.assert_allclose(accumulated, [10.0, 30.0, 60.0])
    legend.setup_classes_from_model({0: 'a', 1: 'b'}, generate_tab10_lut(), np.array([0, 1]), hidden=['A'])
    assert legend.shares()[0] == ['b']
