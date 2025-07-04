"""Multimodal-IAD PyQt6 GUI application."""

import sys
import warnings

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication


def main() -> None:
    """Run the main GUI application."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    app = QApplication(sys.argv)
    app.setApplicationName("Multimodal-IAD")
    app.setStyle("Fusion")

    font = QFont("Arial", 10)
    app.setFont(font)

    from multimodal_iad.gui.main_window import MainWindow  # noqa: PLC0415

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
