# Linear Programming Solver

A desktop app for solving linear programming (LP) problems. It shows every step, the way a textbook would. You can choose one of three methods:

- **Graphical method** (2 variables): draws the constraint lines, the feasible region, the vertices and the optimum. It also explains how each vertex was found, using Cramer's rule and a feasibility test.
- **Simplex (tableau)**: for problems where every constraint is `≤` with `b ≥ 0`. It shows each tableau, the entering and leaving variables, the minimum ratio test and the row operations.
- **Big M**: for `≥` and `=` constraints, negative right-hand sides and free variables. Artificial variables are penalized, and Big M values are written symbolically (for example `-3M + 18`).

The solver detects optimal, unbounded and infeasible problems, multiple optima and degenerate solutions. It switches to Bland's rule after several degenerate pivots in a row, to avoid cycling. For 2-variable problems, the simplex result is cross-checked against the graphical method.

## Requirements

- Python 3.8+
- PyQt5, pyqtgraph, numpy

## Installation

```bash
git clone https://github.com/phmuni/linear-solver.git
cd linear-solver
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

1. Pick a method and the number of variables.
2. Choose maximize or minimize, and type the objective, e.g. `3x1 + 5x2` or `Z = 2x1 - x2 + 10`.
   Fractions (`1/2x1`), decimal commas and the aliases `x`/`y` are also accepted.
3. Fill in one row per constraint: coefficients, sign (`≤`, `≥`, `=`) and right-hand side.
4. Click **Solve** (or press `Ctrl+Enter`). Click **Example** to load a sample problem.

Results appear in the **Chart**, **Vertices**, **Step by step** and **Tableaux** tabs.

### Using the solver without the GUI

The `solver` package only needs the standard library:

```python
from solver import ProblemN, solve_simplex

p = ProblemN(
    sense="maximize",
    c=[3, 5],
    constraints=[([1, 0], "<=", 4), ([0, 2], "<=", 12), ([3, 2], "<=", 18)],
)
result = solve_simplex(p)
print(result.status, result.x, result.z)   # optimal (2.0, 6.0) 36.0
print("\n\n".join(result.steps))           # full step-by-step explanation
```

## Project structure

```
linear-solver/
├── main.py              # application entry point
├── gui/
│   └── main_window.py   # PyQt5 window (input form, chart, tabs)
├── solver/
│   ├── problem.py       # Problem / ProblemN data classes
│   ├── expression.py    # objective function parser
│   ├── graphical.py     # graphical (vertex) method for 2 variables
│   ├── simplex.py       # tableau simplex with Big M
│   └── formatting.py    # number and text formatting for the explanations
└── tests/
    └── test_solver.py
```

## Tests

```bash
python -m unittest discover tests
```

## License

[MIT](LICENSE)
