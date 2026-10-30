# smile/test_gui.py
"""Visual/manual GUI test -- run standalone to inspect interactively.

Usage:
    cd c:\\modules
    python -m smile.test_gui [image_path]
"""
import sys

from smile.widget import HAVE_QT, SmileCorrectionWidget


def main() -> None:
    """Launch the widget in a standalone window."""
    if not HAVE_QT:
        sys.exit(
            "PyQt5 not available. Install with:\n"
            "  pip install PyQt5 matplotlib"
        )

    from PyQt5.QtWidgets import QApplication, QMainWindow

    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("Smile Module -- GUI Test")
    window.resize(1200, 800)

    widget = SmileCorrectionWidget()
    window.setCentralWidget(widget)

    if len(sys.argv) > 1:
        widget.load_file(sys.argv[1])

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()