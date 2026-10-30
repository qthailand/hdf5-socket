# smile/test_widget.py
"""Tests for SmileCorrectionWidget (requires PyQt5 + pytest-qt)."""
import numpy as np
import pytest
pytest.importorskip("PyQt5")
pytest.importorskip("pytest-qt")

from smile.widget import SmileCorrectionWidget

class TestWidget:
    def test_set_frame_clears_result(self, qtbot, synthetic_frame):
        w = SmileCorrectionWidget()
        qtbot.addWidget(w)

        w.set_frame(synthetic_frame)
        w.analyze()  # async
        # ... wait for signal ...

        w.set_frame(np.zeros((10, 10), dtype=np.float32))
        assert w.get_analysis() is None          # ★ stale result cleared
        assert w.btn_save.isEnabled() == False   # ★ save disabled

    def test_analysis_signal(self, qtbot, synthetic_frame):
        w = SmileCorrectionWidget()
        qtbot.addWidget(w)
        w.set_frame(synthetic_frame)

        with qtbot.waitSignal(w.analysisComplete, timeout=30000):
            w.analyze()

        assert w.get_analysis() is not None
        assert w.get_corrector() is not None     # ★ SmileCorrector available