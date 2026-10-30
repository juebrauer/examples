"""Launch the desktop experiment: python main.py"""
import sys

import torch
from PySide6.QtWidgets import QApplication

from ui import ExperimentWindow


def main():
    # Small CPU batches are much faster without a large BLAS thread pool.
    torch.set_num_threads(1)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ExperimentWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
