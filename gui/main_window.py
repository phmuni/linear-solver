"""Main window of the Linear Solver application."""

import math
import os
import tempfile
from typing import Optional, TypeVar, cast

import pyqtgraph as pg

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListView,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QShortcut,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from solver.expression import parse_objective
from solver.formatting import (
    fmt_num,
    format_constraint,
    format_objective,
    format_objective_n,
)
from solver.graphical import solve_graphical
from solver.problem import Problem, ProblemN
from solver.simplex import format_tableau, requires_big_m, solve_simplex

SIGNS = [("<=", "≤"), (">=", "≥"), ("==", "=")]

TAB_CHART, TAB_VERTICES, TAB_STEPS, TAB_TABLEAUX = range(4)
METHOD_GRAPHICAL, METHOD_SIMPLEX, METHOD_BIG_M = range(3)

OP_WIDTH, SIGN_WIDTH, REMOVE_WIDTH = 44, 80, 44
MIN_NUMBER_WIDTH = 84
MAX_VARIABLES = 10
OPERATORS = [(1.0, "+"), (-1.0, "−")]

ALIGN_RIGHT = cast(
    Qt.AlignmentFlag,
    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
)
ALIGN_CENTER = cast(Qt.AlignmentFlag, Qt.AlignmentFlag.AlignCenter)

POPUP_FLAGS = cast(
    Qt.WindowType,
    Qt.WindowType.Popup
    | Qt.WindowType.FramelessWindowHint
    | Qt.WindowType.NoDropShadowWindowHint,
)

EXAMPLE_Z = "3x1 + 5x2"
EXAMPLE_CONSTRAINTS = [
    (1, 0, "<=", 4),
    (0, 2, "<=", 12),
    (3, 2, "<=", 18),
]

T = TypeVar("T")


def _req(obj: Optional[T]) -> T:
    """Ensure that an object PyQt types as optional is not None."""
    if obj is None:
        raise RuntimeError("Expected widget was not created by Qt.")
    return obj


def _mono_font(points: float = 10.0) -> QFont:
    """Monospaced font for the tableaux and steps (columns must line up).

    QSS does not understand the generic 'monospace' family, so the choice is
    made here: the first known fixed-width font installed on the system.
    """
    db = QFontDatabase()
    installed = set(db.families())
    for family in ("Menlo", "Monaco", "Consolas", "DejaVu Sans Mono",
                   "Liberation Mono", "Courier New", "Courier"):
        if family in installed:
            font = QFont(family)
            break
    else:
        font = QFont("Monospace")
        font.setStyleHint(QFont.StyleHint.Monospace)
    font.setFixedPitch(True)
    font.setPointSizeF(points)
    return font


def _make_arrow(color="#7a6e65"):
    """Draw the dropdown arrow into a temporary PNG and return its path.

    Qt only accepts an image in ::down-arrow, so the arrow is generated once
    at startup instead of shipping an extra file with the project.
    """
    pix = QPixmap(16, 16)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color), 2)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.drawLine(3, 6, 8, 11)
    painter.drawLine(8, 11, 13, 6)
    painter.end()
    path = os.path.join(tempfile.gettempdir(), "linear_solver_arrow.png")
    pix.save(path, "PNG")
    return path.replace("\\", "/")


class _OperatorButton(QPushButton):
    """+ / − button between two terms; each click flips the sign."""

    def __init__(self):
        super().__init__()
        self.setObjectName("operator")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click to toggle between + and −.")
        self.clicked.connect(lambda: self.setCurrentIndex(1 - self._index))
        self.setCurrentIndex(0)

    def setCurrentIndex(self, index):
        self._index = index
        self.setText(OPERATORS[index][1])

    def currentData(self):
        return OPERATORS[self._index][0]


class _ConstraintTable(QTableWidget):
    """Table that reports when its visible area is resized."""

    def __init__(self, on_resize):
        super().__init__(0, 0)
        self._on_resize = on_resize

    def viewportEvent(self, event):
        if event is not None and event.type() == QEvent.Type.Resize:
            self._on_resize()
        return super().viewportEvent(event)


STYLE = """
    QWidget { color: #2b2622; font-size: 10pt; }
    QMainWindow, QWidget#centralWidget { background: #f7f4f1; }

    QGroupBox { background: #ffffff; font-weight: 700; color: #1a1511;
                border: 1px solid #e4dcd4; border-radius: 10px;
                margin-top: 20px; padding: 14px 12px 12px 12px; }
    QGroupBox::title { subcontrol-origin: margin; left: 14px; top: 0px;
                       padding: 0 8px; min-height: 22px; }
    QLabel { color: #2b2622; background: transparent; font-weight: 400; }
    QLabel#subtitle { color: #7a6e65; font-weight: 600; }
    QLabel#hint { color: #7a6e65; font-size: 9pt; }
    QLabel#preview { color: #c4572f; font-size: 10.5pt; padding: 2px 0; }

    QPushButton { background: #c4572f; color: #ffffff; border: 1px solid #c4572f;
                  border-radius: 6px; padding: 7px 18px; font-weight: 600;
                  min-height: 20px; }
    QPushButton:hover { background: #e97451; border-color: #e97451; }
    QPushButton:pressed { background: #a8461f; border-color: #a8461f; }
    QPushButton:disabled { background: #efe9e3; color: #b8aca2; border-color: #e4dcd4; }
    QPushButton#secondary { background: #ffffff; border: 1px solid #d9cfc5;
                            color: #2b2622; }
    QPushButton#secondary:hover { background: #fcebe3; border-color: #e97451;
                                  color: #a8461f; }
    QPushButton#secondary:pressed { background: #f9d9ca; }
    QPushButton#secondary:disabled { color: #b8aca2; background: #faf7f4; }
    QPushButton#primary { font-size: 11pt; padding: 9px 28px; }
    QPushButton#stepper { background: #ffffff; border: 1px solid #d9cfc5;
                          color: #2b2622; padding: 0; min-height: 0;
                          font-size: 15pt; font-weight: 400; }
    QPushButton#stepper:hover { background: #fcebe3; border-color: #e97451;
                                color: #a8461f; }
    QPushButton#stepper:pressed { background: #f9d9ca; }
    QPushButton#stepper:disabled { color: #b8aca2; background: #faf7f4; }
    QPushButton#remove { background: transparent; border: 1px solid transparent;
                         color: #b8aca2; padding: 2px 6px; min-height: 0;
                         font-size: 11pt; }
    QPushButton#remove:hover { color: #c4572f; border-color: #e97451;
                               background: #fcebe3; }

    QLineEdit, QComboBox, QTableWidget, QPlainTextEdit {
        background: #ffffff; color: #2b2622;
        border: 1px solid #d9cfc5; border-radius: 8px; padding: 5px;
        selection-background-color: #f9d9ca; selection-color: #1a1511;
    }
    QLineEdit:focus, QComboBox:focus { border: 1px solid #e97451; }

    QComboBox { padding: 5px 6px 5px 12px; min-height: 20px;
                combobox-popup: 0; }
    QComboBox:hover { border: 1px solid #b8aca2; }
    QComboBox:on { border: 1px solid #e97451; }
    QComboBox::drop-down { subcontrol-origin: padding;
                           subcontrol-position: center right;
                           width: 28px; border: 0; background: transparent; }
    QComboBox::down-arrow { image: url(__ARROW__); width: 12px; height: 12px; }
    QComboBox QAbstractItemView {
        background: #ffffff; color: #2b2622;
        border: 1px solid #e4dcd4; border-radius: 8px;
        padding: 4px; outline: 0;
        selection-background-color: #fcebe3; selection-color: #8f3a1b;
    }
    QComboBox QAbstractItemView::item {
        min-height: 26px; padding: 4px 10px; border-radius: 5px;
    }
    QComboBox QAbstractItemView::item:hover,
    QComboBox QAbstractItemView::item:selected {
        background: #fcebe3; color: #8f3a1b;
    }
    QTableWidget { gridline-color: #efe9e3; }
    QTableWidget QLineEdit { margin: 4px 4px; padding: 2px 6px; }
    QTableWidget QComboBox { margin: 3px 4px; padding: 2px 2px 2px 8px;
                             min-height: 18px; }
    QTableWidget QComboBox::drop-down { width: 20px; }
    QPushButton#operator { background: #ffffff; color: #2b2622; border: 1px solid #d9cfc5;
                           border-radius: 8px; padding: 0; min-height: 0; margin: 4px 4px;
                           font-size: 13pt; font-weight: 600; }
    QPushButton#operator:hover { background: #fcebe3; border-color: #e97451;
                                 color: #a8461f; }
    QTableWidget QPushButton#remove { margin: 4px 4px; padding: 0; }
    QHeaderView::section {
        background: #f3eee9; color: #1a1511; font-weight: 700;
        border: 0; border-bottom: 1px solid #e4dcd4; padding: 7px 5px;
    }
    QTableWidget::item { color: #2b2622; padding: 5px; }
    QTableWidget::item:selected { background: #fcebe3; color: #8f3a1b; }

    QLabel#result { background: #ffffff; border: 1px solid #e4dcd4;
                    border-left: 4px solid #b8aca2; border-radius: 6px;
                    padding: 9px 12px; font-weight: 600; }
    QLabel#result[state="optimal"] { border-left-color: #5aa24a; }
    QLabel#result[state="unbounded"] { border-left-color: #e0a030; }
    QLabel#result[state="infeasible"] { border-left-color: #d1432f; }

    QTabWidget::pane { background: #ffffff; border: 1px solid #e4dcd4;
                       border-radius: 10px; border-top-left-radius: 0;
                       top: -1px; }
    QTabWidget::tab-bar { left: 0px; }
    QTabBar { qproperty-drawBase: 0; }
    QTabBar::tab {
        background: transparent; color: #7a6e65; font-weight: 700;
        padding: 10px 22px; margin-right: 4px; min-width: 80px;
        border: 0; border-bottom: 3px solid transparent;
        border-top-left-radius: 8px; border-top-right-radius: 8px;
    }
    QTabBar::tab:hover { background: #f3eee9; color: #2b2622; }
    QTabBar::tab:selected { background: #ffffff; color: #c4572f;
                            font-weight: 700; border-bottom: 3px solid #e97451; }
    QTabBar::tab:disabled { background: transparent; color: #b8aca2;
                            border-bottom: 3px solid transparent; }
    QPlainTextEdit#textArea { border: 0; border-radius: 8px; padding: 10px; }

    QSpinBox { background: #ffffff; border: 1px solid #d9cfc5; border-radius: 8px;
               padding: 4px 6px; min-height: 20px; }
    QSpinBox:disabled { color: #b8aca2; background: #faf7f4; }
    QSpinBox:focus { border: 1px solid #e97451; }

    QCheckBox { color: #2b2622; spacing: 7px; }
    QCheckBox::indicator { width: 16px; height: 16px; }
    QSplitter::handle { background: transparent; }
    QStatusBar { background: #ffffff; color: #7a6e65;
                 border-top: 1px solid #e4dcd4; }
    QToolTip { background: #ffffff; color: #2b2622; border: 1px solid #e4dcd4; }
"""


class MainWindow(QMainWindow):
    """Problem input, chart, vertices and step-by-step walkthrough."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Linear Programming Solver")
        self.resize(1280, 860)
        self.setMinimumSize(1040, 720)
        self.setStyleSheet(STYLE.replace("__ARROW__", _make_arrow()))
        self.result = None
        self.problem = None
        self.n_vars = 2
        self._build_ui()
        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(self._solve)
        QShortcut(QKeySequence("Ctrl+Enter"), self).activated.connect(self._solve)
        self._clear()
        self._on_method_changed()
        self._notify("Ready. Fill in the model and click Solve (Ctrl+Enter), "
                     "or use Example to load a sample problem.")

    def _notify(self, text):
        """Message in the status bar."""
        _req(self.statusBar()).showMessage(text)

    def _new_combo(self):
        """Dropdown with a rounded, stylable options list."""
        combo = QComboBox()
        combo.setView(QListView())
        combo.setCursor(Qt.CursorShape.PointingHandCursor)
        popup = _req(_req(combo.view()).window())
        popup.setWindowFlags(POPUP_FLAGS)
        popup.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        return combo

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("centralWidget")
        box = QVBoxLayout(central)
        box.setContentsMargins(14, 14, 14, 10)
        box.setSpacing(10)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)
        splitter.addWidget(self._problem_panel())
        splitter.addWidget(self._result_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([620, 600])
        box.addWidget(splitter)
        self.setCentralWidget(central)

    def _problem_panel(self):
        group = QGroupBox("Problem model")
        group.setMinimumWidth(560)
        group.setMaximumWidth(760)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(8)

        self.combo_method = self._new_combo()
        self.combo_method.addItems(["Graphical method", "Simplex (tableau)", "Big M"])
        self.combo_method.setToolTip("Choose the algorithm used to solve the model.")
        layout.addWidget(self._subtitle("1. Method"))
        layout.addWidget(self.combo_method)

        self.spin_vars = QSpinBox()
        self.spin_vars.setRange(2, MAX_VARIABLES)
        self.spin_vars.setValue(2)
        self.spin_vars.setToolTip("Number of decision variables (x1, x2, x3...). "
                                  "The graphical method always uses 2.")
        vars_row = QHBoxLayout()
        vars_row.setSpacing(8)
        vars_row.addWidget(QLabel("Variables:"))
        self.spin_vars.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.spin_vars.setAlignment(ALIGN_CENTER)
        self.spin_vars.setFixedSize(56, 34)
        self.bt_minus = QPushButton("−")
        self.bt_plus = QPushButton("+")
        for bt, step in ((self.bt_minus, -1), (self.bt_plus, 1)):
            bt.setObjectName("stepper")
            bt.setFixedSize(38, 34)
            bt.setCursor(Qt.CursorShape.PointingHandCursor)
            bt.clicked.connect(lambda _c=False, d=step: self.spin_vars.stepBy(d))
        vars_row.addWidget(self.bt_minus)
        vars_row.addWidget(self.spin_vars)
        vars_row.addWidget(self.bt_plus)
        vars_row.addStretch(1)
        layout.addLayout(vars_row)
        layout.addSpacing(6)

        self.combo_sense = self._new_combo()
        self.combo_sense.addItems(["Maximize", "Minimize"])
        self.combo_sense.setToolTip("Choose whether to maximize or minimize Z.")

        self.field_z = QLineEdit()
        self.field_z.setMinimumWidth(56)
        self.field_z.setPlaceholderText("3x1 + 5x2")
        self.field_z.setToolTip("Linear objective function in x1, x2, x3..., with an optional "
                                "constant term. E.g.: 3x1 + 5x2, 2x1 - x2 + x3 + 10.")

        z_row = QHBoxLayout()
        z_row.setSpacing(6)
        z_row.addWidget(QLabel("Z ="))
        z_row.addWidget(self.field_z, 1)

        self.label_preview = QLabel("")
        self.label_preview.setObjectName("preview")
        self.label_preview.setFont(_mono_font(10.5))
        self.label_preview.setWordWrap(True)

        layout.addWidget(self._subtitle("2. Objective"))
        layout.addWidget(self.combo_sense)
        layout.addLayout(z_row)
        layout.addWidget(self.label_preview)
        layout.addSpacing(6)
        self.combo_method.currentIndexChanged.connect(self._on_method_changed)
        self.spin_vars.valueChanged.connect(self._set_n)
        self.combo_sense.currentIndexChanged.connect(self._update_preview)
        self.field_z.textChanged.connect(self._update_preview)
        self.field_z.returnPressed.connect(self._solve)

        hint = QLabel("Each row: variable coefficients (with + or − between them), "
                      "sign and right-hand side")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(self._subtitle("3. Subject to"))
        layout.addWidget(hint)

        self.constraints_table = _ConstraintTable(self._adjust_widths)
        _req(self.constraints_table.verticalHeader()).setVisible(False)
        self.constraints_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.constraints_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.constraints_table.setHorizontalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel)
        header = _req(self.constraints_table.horizontalHeader())
        header.setDefaultAlignment(ALIGN_CENTER)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self._configure_columns()
        self.constraints_table.setMinimumHeight(120)
        layout.addWidget(self.constraints_table, 1)

        bt_add = QPushButton("+ Constraint")
        bt_add.setObjectName("secondary")
        bt_add.setToolTip("Add a new constraint row.")
        bt_add.clicked.connect(lambda: self._add_constraint())
        layout.addWidget(bt_add)

        self.check_non_negative = QCheckBox("x1, x2 ≥ 0")
        self.check_non_negative.setToolTip("Non-negative variables.")
        self.check_non_negative.setChecked(True)
        layout.addWidget(self.check_non_negative)
        layout.addSpacing(6)

        self.bt_solve = QPushButton("Solve")
        self.bt_solve.setObjectName("primary")
        self.bt_solve.setDefault(True)
        self.bt_solve.setToolTip("Solve the model with the chosen method (Ctrl+Enter).")
        self.bt_solve.clicked.connect(self._solve)

        bt_example = QPushButton("Example")
        bt_example.setObjectName("secondary")
        bt_example.setToolTip("Fill the form with a sample problem.")
        bt_example.clicked.connect(self._load_example)

        bt_clear = QPushButton("Clear")
        bt_clear.setObjectName("secondary")
        bt_clear.setToolTip("Erase the model and the results.")
        bt_clear.clicked.connect(self._clear)

        layout.addWidget(self.bt_solve)
        layout.addWidget(bt_example)
        layout.addWidget(bt_clear)
        return group

    def _result_panel(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.label_result = QLabel("")
        self.label_result.setObjectName("result")
        self.label_result.setWordWrap(True)
        self.label_result.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.label_result.hide()
        layout.addWidget(self.label_result)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(False)
        tab_font = QFont(self.tabs.font())
        tab_font.setBold(True)
        _req(self.tabs.tabBar()).setFont(tab_font)
        self.tabs.addTab(self._chart_tab(), "Chart")
        self.tabs.addTab(self._vertices_tab(), "Vertices")
        self.tabs.addTab(self._steps_tab(), "Step by step")
        self.tableau_text = QPlainTextEdit()
        self.tableau_text.setFont(_mono_font())
        self.tableau_text.setObjectName("textArea")
        self.tableau_text.setReadOnly(True)
        self.tableau_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.tabs.addTab(self.tableau_text, "Tableaux")
        layout.addWidget(self.tabs, 1)
        return container

    def _subtitle(self, text):
        label = QLabel(text)
        label.setObjectName("subtitle")
        return label

    def _chart_tab(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        self.chart = pg.PlotWidget()
        self.chart.setBackground("#ffffff")
        self.chart.showGrid(x=True, y=True, alpha=0.25)
        self.chart.setLabel("bottom", "x1", color="#2b2622")
        self.chart.setLabel("left", "x2", color="#2b2622")
        self.chart.getAxis("bottom").setPen(pg.mkPen("#cfc5bc"))
        self.chart.getAxis("left").setPen(pg.mkPen("#cfc5bc"))
        self.chart.getAxis("bottom").setTextPen(pg.mkPen("#7a6e65"))
        self.chart.getAxis("left").setTextPen(pg.mkPen("#7a6e65"))
        self.chart.setMenuEnabled(True)
        self.chart.setMouseEnabled(x=True, y=True)
        layout.addWidget(self.chart, 1)
        return container

    def _vertices_tab(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        self.vertices_table = QTableWidget()
        self.vertices_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.vertices_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.vertices_table.setColumnCount(5)
        self.vertices_table.setHorizontalHeaderLabels(
            ["Vertex", "Origin (intersection)", "Coordinates", "Status", "Z"])
        _req(self.vertices_table.verticalHeader()).setVisible(False)
        header = _req(self.vertices_table.horizontalHeader())
        header.setDefaultAlignment(ALIGN_CENTER)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.vertices_table)
        return container

    def _steps_tab(self):
        self.steps_text = QPlainTextEdit()
        self.steps_text.setFont(_mono_font())
        self.steps_text.setObjectName("textArea")
        self.steps_text.setReadOnly(True)
        return self.steps_text

    def _col_coef(self, i):
        return 2 * i

    def _col_op(self, i):
        """Operator between the coefficients of x(i+1) and x(i+2)."""
        return 2 * i + 1

    @property
    def _col_sign(self):
        return 2 * self.n_vars - 1

    @property
    def _col_b(self):
        return 2 * self.n_vars

    @property
    def _col_remove(self):
        return 2 * self.n_vars + 1

    def _configure_columns(self):
        """Headers and widths of the table for the current number of variables."""
        n = self.n_vars
        table = self.constraints_table
        table.setColumnCount(2 * n + 2)
        labels = []
        for i in range(n):
            labels.append(f"Coef. x{i + 1}")
            if i < n - 1:
                labels.append("")
        labels += ["Sign", "RHS b", ""]
        table.setHorizontalHeaderLabels(labels)
        self._adjust_widths()

    def _adjust_widths(self):
        """Split the free width among the numeric fields; scroll only if they don't fit."""
        n = self.n_vars
        table = self.constraints_table
        if table.columnCount() != 2 * n + 2:
            return
        fixed = (n - 1) * OP_WIDTH + SIGN_WIDTH + REMOVE_WIDTH
        free = _req(table.viewport()).width() - fixed
        width = max(MIN_NUMBER_WIDTH, free // (n + 1))
        leftover = max(0, free - width * (n + 1))
        for i in range(n):
            table.setColumnWidth(self._col_coef(i), width)
            if i < n - 1:
                table.setColumnWidth(self._col_op(i), OP_WIDTH)
        table.setColumnWidth(self._col_sign, SIGN_WIDTH)
        table.setColumnWidth(self._col_b, width + leftover)
        table.setColumnWidth(self._col_remove, REMOVE_WIDTH)

    def _constraint_field(self, row, column):
        """Text field of a cell in the constraints table."""
        widget = self.constraints_table.cellWidget(row, column)
        if not isinstance(widget, QLineEdit):
            raise RuntimeError(f"Cell ({row}, {column}) has no text field.")
        return widget

    def _sign_combo(self, row):
        """Sign selector of a row in the constraints table."""
        widget = self.constraints_table.cellWidget(row, self._col_sign)
        if not isinstance(widget, QComboBox):
            raise RuntimeError(f"Row {row} has no sign selector.")
        return widget

    def _operator_button(self, row, i):
        """+ / − selector between the coefficients of x(i+1) and x(i+2)."""
        widget = self.constraints_table.cellWidget(row, self._col_op(i))
        if not isinstance(widget, _OperatorButton):
            raise RuntimeError(f"Row {row} has no operator selector.")
        return widget

    def _add_constraint(self, values=None):
        """New row. values = (coefficients, sign, b), as numbers or strings."""
        n = self.n_vars
        row = self.constraints_table.rowCount()
        self.constraints_table.insertRow(row)
        fields = []
        for col in [self._col_coef(i) for i in range(n)] + [self._col_b]:
            field = QLineEdit()
            field.setAlignment(ALIGN_RIGHT)
            field.setPlaceholderText("0")
            field.returnPressed.connect(self._solve)
            self.constraints_table.setCellWidget(row, col, field)
            fields.append(field)
        op_buttons = []
        for i in range(n - 1):
            op_button = _OperatorButton()
            self.constraints_table.setCellWidget(row, self._col_op(i), op_button)
            op_buttons.append(op_button)
        combo = self._new_combo()
        for value, text in SIGNS:
            combo.addItem(text, value)
        self.constraints_table.setCellWidget(row, self._col_sign, combo)

        bt = QPushButton("✕")
        bt.setObjectName("remove")
        bt.setToolTip("Remove this constraint.")
        bt.setCursor(Qt.CursorShape.PointingHandCursor)
        bt.clicked.connect(lambda _checked=False, b=bt: self._remove_row(b))
        self.constraints_table.setCellWidget(row, self._col_remove, bt)

        self.constraints_table.setRowHeight(row, 38)
        if values:
            coefs, sign, b = values
            for i in range(n):
                value = coefs[i] if i < len(coefs) else ""
                if isinstance(value, str):
                    negative = i > 0 and value.strip().startswith("-")
                    fields[i].setText(value.strip().lstrip("-") if negative else value)
                    if i > 0:
                        op_buttons[i - 1].setCurrentIndex(1 if negative else 0)
                    continue
                fields[i].setText(fmt_num(value if i == 0 else abs(value)))
                if i > 0:
                    op_buttons[i - 1].setCurrentIndex(1 if value < 0 else 0)
            fields[n].setText(b if isinstance(b, str) else fmt_num(b))
            combo.setCurrentIndex([s[0] for s in SIGNS].index(sign))
        self.constraints_table.scrollToBottom()
        fields[0].setFocus()

    def _remove_row(self, button):
        """Remove the row of the clicked button; the last row is only emptied."""
        for i in range(self.constraints_table.rowCount()):
            if self.constraints_table.cellWidget(i, self._col_remove) is button:
                if self.constraints_table.rowCount() > 1:
                    self.constraints_table.removeRow(i)
                else:
                    for col in [self._col_coef(k) for k in range(self.n_vars)] + [self._col_b]:
                        self._constraint_field(i, col).clear()
                    self._sign_combo(i).setCurrentIndex(0)
                    for k in range(self.n_vars - 1):
                        self._operator_button(i, k).setCurrentIndex(0)
                return

    def _on_method_changed(self):
        """The graphical method only works with 2 variables; the others accept n."""
        graphical = self.combo_method.currentIndex() == METHOD_GRAPHICAL
        for w in (self.spin_vars, self.bt_minus, self.bt_plus):
            w.setEnabled(not graphical)
        if graphical:
            self.spin_vars.setValue(2)

    def _set_n(self, n):
        """Change the number of variables, keeping what has already been typed."""
        if n == self.n_vars:
            return
        old = self.n_vars
        saved = []
        for row in range(self.constraints_table.rowCount()):
            coefs = []
            for i in range(old):
                text = self._constraint_field(row, self._col_coef(i)).text()
                op_sign = self._operator_button(row, i - 1).currentData() if i else 1.0
                try:
                    coefs.append(float(text.replace(",", ".")) * op_sign)
                except ValueError:
                    coefs.append(text)
            saved.append((coefs,
                          self._sign_combo(row).currentData(),
                          self._constraint_field(row, self._col_b).text()))
        self.n_vars = n
        self.constraints_table.setRowCount(0)
        self._configure_columns()
        for coefs, sign, b in saved:
            self._add_constraint((coefs, sign, b))
        if not saved:
            self._add_constraint()
        self.check_non_negative.setText(", ".join(f"x{i + 1}" for i in range(n)) + " ≥ 0")
        self._update_preview()

    def _load_example(self):
        """Fill the form with a ready-made problem to try out."""
        self.spin_vars.setValue(2)
        self._clear()
        self.constraints_table.setRowCount(0)
        self.combo_sense.setCurrentIndex(0)
        self.field_z.setText(EXAMPLE_Z)
        for a1, a2, sign, b in EXAMPLE_CONSTRAINTS:
            self._add_constraint(([a1, a2], sign, b))
        self.bt_solve.setFocus()
        self._notify("Example loaded. Click Solve to see the result.")

    def _value(self, text, context):
        text = text.strip()
        if text in ("", "-"):
            raise ValueError(f"Please fill in {context}.")
        try:
            return float(text.replace(",", "."))
        except ValueError:
            raise ValueError(f"Invalid value in {context}: '{text}'.")

    def _read_problem(self):
        n = self.n_vars
        sense = "maximize" if self.combo_sense.currentIndex() == 0 else "minimize"
        c, c0 = parse_objective(self.field_z.text(), n)
        constraints = []
        for i in range(self.constraints_table.rowCount()):
            texts = [self._constraint_field(i, self._col_coef(k)).text() for k in range(n)]
            b_txt = self._constraint_field(i, self._col_b).text()
            sign = self._sign_combo(i).currentData()
            if all(t.strip() == "" for t in texts) and b_txt.strip() == "":
                continue
            coefs = []
            for k, text in enumerate(texts):
                value = self._value(text, f"the coefficient of x{k + 1} in constraint {i + 1}")
                if k > 0:
                    value *= self._operator_button(i, k - 1).currentData()
                coefs.append(value)
            b = self._value(b_txt, f"the right-hand side of constraint {i + 1}")
            constraints.append((coefs, sign, b))
        if not constraints:
            raise ValueError("Enter at least one complete constraint.")
        return ProblemN(sense=sense, c=c, constraints=constraints,
                        non_negative=self.check_non_negative.isChecked(), c0=c0)

    def _update_preview(self):
        """Show the objective function as it will appear in the model, if valid."""
        try:
            c, c0 = parse_objective(self.field_z.text(), self.n_vars)
        except ValueError:
            self.label_preview.setText("")
            return
        verb = "max" if self.combo_sense.currentIndex() == 0 else "min"
        self.label_preview.setText(f"→  {verb} Z = {format_objective_n(c, c0)}")

    def _show_result(self, text, status=None):
        """Show the result banner with the color matching the status."""
        if not text:
            self.label_result.hide()
            self.label_result.setText("")
            return
        state = status if status in ("optimal", "unbounded", "infeasible") else "neutral"
        self.label_result.setProperty("state", state)
        style = _req(self.label_result.style())
        style.unpolish(self.label_result)
        style.polish(self.label_result)
        self.label_result.setText(text)
        self.label_result.show()

    def _solve(self):
        graphical_res = None
        simplex_res = None
        try:
            problem = self._read_problem()
            method = self.combo_method.currentIndex()
            if method == METHOD_GRAPHICAL:
                problem = problem.to_two_var()
                graphical_res = solve_graphical(problem)
            else:
                if method == METHOD_SIMPLEX and requires_big_m(problem):
                    raise ValueError(
                        "The plain tableau simplex only starts from ≤ constraints with b ≥ 0, "
                        "where the slacks form the initial basis. This model has "
                        "≥ or = constraints (or a negative b) and needs artificial "
                        "variables: choose the Big M method.")
                simplex_res = solve_simplex(problem)
        except ValueError as error:
            QMessageBox.warning(self, "Invalid input", str(error))
            return
        self.problem = problem
        if graphical_res is not None:
            self.result = graphical_res
            self._show_graphical(problem, graphical_res)
        elif simplex_res is not None:
            self.result = simplex_res
            self._show_simplex(simplex_res)

    def _show_graphical(self, problem, res):
        self._show_result(res.status_text, res.status)
        self._fill_vertices(res)
        self.steps_text.setPlainText(self._steps_text(problem, res))
        self.tableau_text.clear()
        self._draw(problem, res)
        self._enable_tabs(graphical=True)
        self.tabs.setCurrentIndex(TAB_CHART)
        self._notify(f"{len(res.vertices)} vertex candidates evaluated.")

    def _show_simplex(self, res):
        self._show_result(res.status_text, res.status)
        self.vertices_table.setRowCount(0)
        self.chart.clear()
        self.steps_text.setPlainText("\n".join(res.steps))
        self.tableau_text.setPlainText(
            "\n\n".join(f"TABLEAU {it.number}\n{format_tableau(it)}"
                        for it in res.iterations))
        self._enable_tabs(graphical=False)
        self.tabs.setCurrentIndex(TAB_TABLEAUX)
        self._notify(f"{len(res.iterations) - 1} iterations performed.")

    def _enable_tabs(self, graphical):
        """Disable the tabs that do not apply to the method used."""
        self.tabs.setTabEnabled(TAB_CHART, graphical)
        self.tabs.setTabEnabled(TAB_VERTICES, graphical)
        self.tabs.setTabEnabled(TAB_STEPS, True)
        self.tabs.setTabEnabled(TAB_TABLEAUX, not graphical)

    def _fill_vertices(self, res):
        self.vertices_table.setRowCount(0)
        for i, v in enumerate(res.vertices):
            n = self.vertices_table.rowCount()
            self.vertices_table.insertRow(n)
            status = "feasible" if v.feasible else f"infeasible ({v.reason})"
            if res.best is not None and v is res.best:
                status += " (optimal)"
            z = fmt_num(v.z) if v.z is not None else ""
            cells = [
                f"V{i + 1}",
                v.origin,
                f"({fmt_num(v.x)}, {fmt_num(v.y)})",
                status,
                z,
            ]
            for j, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if j in (0, 2, 4):
                    item.setTextAlignment(ALIGN_CENTER)
                self.vertices_table.setItem(n, j, item)
            self.vertices_table.setRowHeight(n, 30)

    def _segment_in_window(self, r, xlim, ylim):
        """Two points of the line inside the viewing window."""
        a1, a2, b = r["a1"], r["a2"], r["b"]
        if abs(a2) < 1e-12:
            x = b / a1
            return [(x, ylim[0]), (x, ylim[1])]
        if abs(a1) < 1e-12:
            y = b / a2
            return [(xlim[0], y), (xlim[1], y)]
        points = []
        for x in xlim:
            y = (b - a1 * x) / a2
            if ylim[0] - 1e-9 <= y <= ylim[1] + 1e-9:
                points.append((x, y))
        for y in ylim:
            x = (b - a2 * y) / a1
            if xlim[0] - 1e-9 <= x <= xlim[1] + 1e-9:
                points.append((x, y))
        if len(points) < 2:
            return None
        return points[:2]

    def _centroid(self, polygon, feasible):
        if polygon:
            n = len(polygon)
            return (sum(p[0] for p in polygon) / n,
                    sum(p[1] for p in polygon) / n)
        if feasible:
            n = len(feasible)
            return (sum(v.x for v in feasible) / n,
                    sum(v.y for v in feasible) / n)
        return None

    def _draw(self, p: Problem, res):
        self.chart.clear()
        self.chart.setBackground("#ffffff")
        self.chart.showGrid(x=True, y=True, alpha=0.25)
        self.chart.setLabel("bottom", "x1", color="#2b2622")
        self.chart.setLabel("left", "x2", color="#2b2622")
        B = res.box
        if p.non_negative:
            margin = 0.04 * B
            xlim = (-margin, B)
            ylim = (-margin, B)
        else:
            xlim = (-B, B)
            ylim = (-B, B)

        if res.polygon:
            xs = [pt[0] for pt in res.polygon]
            ys = [pt[1] for pt in res.polygon]
            outline = pg.PlotDataItem(xs + xs[:1], ys + ys[:1],
                                      pen=pg.mkPen("#e97451", width=1.2))
            fill = pg.FillBetweenItem(
                pg.PlotDataItem(xs, ys),
                pg.PlotDataItem(xs, [0] * len(ys)),
                brush=pg.mkBrush(233, 116, 81, 70),
            )
            self.chart.addItem(fill)
            self.chart.addItem(outline)

        for r in res.lines:
            seg = self._segment_in_window(r, xlim, ylim)
            if seg is None:
                continue
            (x1, y1), (x2, y2) = seg
            self.chart.plot([x1, x2], [y1, y2],
                            pen=pg.mkPen("#6f625a", width=1.6),
                            name=r["name"])
            text = pg.TextItem(" " + r["name"], color="#2b2622", anchor=(0, 0.5))
            text.setPos(x2, y2)
            self.chart.addItem(text)

        infeasible = [v for v in res.vertices if not v.feasible
                      and xlim[0] <= v.x <= xlim[1] and ylim[0] <= v.y <= ylim[1]]
        if infeasible:
            self.chart.plot([v.x for v in infeasible], [v.y for v in infeasible],
                            pen=None, symbol="x", symbolPen="#b0a398",
                            symbolBrush="#b0a398", symbolSize=10)

        feasible = [v for v in res.vertices if v.feasible]
        if feasible:
            self.chart.plot([v.x for v in feasible], [v.y for v in feasible],
                            pen=None, symbol="o", symbolPen="#e97451",
                            symbolBrush="#ffffff", symbolSize=9)

        if res.status == "optimal" and res.best is not None:
            m = res.best
            self.chart.plot([m.x], [m.y], pen=None, symbol="star",
                            symbolPen="#a8461f", symbolBrush="#a8461f",
                            symbolSize=16)
            text = pg.TextItem(
                f"Optimum ({fmt_num(m.x)}, {fmt_num(m.y)})\nZ = {fmt_num(m.z)}",
                color="#1a1511", anchor=(0, 1))
            text.setPos(m.x, m.y)
            self.chart.addItem(text)

        if res.status == "unbounded" and res.direction is not None:
            origin = self._centroid(res.polygon, feasible)
            if origin is not None:
                ox, oy = origin
                dx, dy = res.direction
                L = 0.3 * B
                arrow = pg.ArrowItem(
                    angle=-180 * math.atan2(dy, dx) / math.pi,
                    brush="#c77d0a", pen="#c77d0a")
                arrow.setPos(ox + dx * L, oy + dy * L)
                self.chart.addItem(arrow)
                text = pg.TextItem("Z improves\nin this direction",
                                   color="#c77d0a", anchor=(0, 0.5))
                text.setPos(ox + dx * L, oy + dy * L)
                self.chart.addItem(text)

        self.chart.getViewBox().setRange(
            xRange=(xlim[0], xlim[1]),
            yRange=(ylim[0], ylim[1]),
            padding=0,
        )
        verb = "max" if p.sense == "maximize" else "min"
        self.chart.setTitle(f"{verb} Z = {format_objective(*p.c, p.c0)}",
                            color="#1a1511", size="12pt")

    def _empty_chart(self):
        """Initial chart state, with an instruction instead of a blank screen."""
        self.chart.clear()
        self.chart.setTitle("")
        text = pg.TextItem("Fill in the model\nand click Solve",
                           color="#9a8d82", anchor=(0.5, 0.5))
        text.setPos(5, 5)
        self.chart.addItem(text)
        self.chart.getViewBox().setRange(xRange=(0, 10), yRange=(0, 10), padding=0)

    def _steps_text(self, p: Problem, res):
        lines = []
        verb = "Maximize" if p.sense == "maximize" else "Minimize"
        lines.append("PROBLEM")
        lines.append(f"{verb} Z = {format_objective(*p.c, p.c0)}")
        lines.append("subject to:")
        for (a1, a2, sign, b) in p.constraints:
            lines.append(f"    {format_constraint(a1, a2, sign, b)}")
        if p.non_negative:
            lines.append("    x1 ≥ 0, x2 ≥ 0")
        else:
            lines.append("    x1 and x2 free")
        lines.append("")
        for step in res.steps:
            lines.append(step.title.upper())
            for text in step.texts:
                lines.append(text)
            lines.append("")
        return "\n".join(lines)

    def _clear(self):
        self.combo_sense.setCurrentIndex(0)
        self.field_z.clear()
        self.constraints_table.setRowCount(0)
        for _ in range(2):
            self._add_constraint()
        self.check_non_negative.setChecked(True)
        self.result = None
        self.problem = None
        self._show_result("")
        self.vertices_table.setRowCount(0)
        self.steps_text.clear()
        self.tableau_text.clear()
        self._empty_chart()
        for i in range(self.tabs.count()):
            self.tabs.setTabEnabled(i, True)
        self.tabs.setCurrentIndex(TAB_CHART)
        self.field_z.setFocus()
        self._notify("Fields cleared.")
