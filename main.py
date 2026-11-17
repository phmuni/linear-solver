"""Linear Programming Solver: graphical method, tableau simplex and Big M (PyQt5 + pyqtgraph).

Run with:
    python main.py

Requirements: PyQt5, pyqtgraph and numpy (see requirements.txt).
"""

import sys

from PyQt5.QtWidgets import QApplication

from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
