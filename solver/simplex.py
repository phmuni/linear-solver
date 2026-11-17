"""Tableau simplex with Big M penalty for LP with n variables.

Besides the solution, the result includes a complete step-by-step walkthrough:
standard form, initial tableau, and for each iteration the choice of the
entering variable, the minimum ratio test, the pivot and the elementary row
operations used.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .formatting import (
    constraint_analysis_n,
    fmt_big_m,
    fmt_num,
    format_constraint_n,
    format_objective_n,
    substitution_n,
    terms_text,
    text_table,
)
from .graphical import solve_graphical
from .problem import ProblemN

EPS = 1e-8
BIG_M = 1_000_000.0
MAX_ITERATIONS = 100


@dataclass
class Iteration:
    number: int
    basis: List[str]
    columns: List[str]
    table: List[List[float]]
    pivot: Optional[Tuple[int, int]] = None
    entering: str = ""
    leaving: str = ""
    next_pivot: Optional[Tuple[int, int]] = None
    ratios: Optional[List[Optional[float]]] = None
    z_label: str = "Z"
    unbounded_column: Optional[int] = None


@dataclass
class SimplexResult:
    status: str
    status_text: str
    x: Optional[Tuple[float, ...]]
    z: Optional[float]
    iterations: List[Iteration] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    variables: Dict[str, float] = field(default_factory=dict)


def _is_artificial(name: str) -> bool:
    return name.startswith("a")


def _normalize(p: ProblemN):
    """Ensure b >= 0; return (coefs, sign, b, was_multiplied_by_-1)."""
    rows = []
    for coefs, sign, b in p.constraints:
        coefs = [float(a) for a in coefs]
        flipped = b < 0
        if flipped:
            coefs, b = [-a for a in coefs], -b
            sign = {"<=": ">=", ">=": "<=", "==": "=="}.get(sign, sign)
        rows.append((coefs, sign, float(b), flipped))
    return rows


def _decision_names(n: int, non_negative: bool) -> List[str]:
    if non_negative:
        return [f"x{i + 1}" for i in range(n)]
    return [f"x{i + 1}{sign}" for i in range(n) for sign in ("+", "-")]


def _expand(values: List[float], non_negative: bool) -> List[float]:
    """Coefficients per variable -> coefficients per decision column (x+ and x-)."""
    if non_negative:
        return list(values)
    return [v for a in values for v in (a, -a)]


def requires_big_m(p: ProblemN) -> bool:
    """True if some constraint needs an artificial variable (≥ or = after b ≥ 0)."""
    return any(sign != "<=" for _coefs, sign, _b, _flip in _normalize(p))


def _standard_form_text(p: ProblemN, rows, plan, names, maximize) -> str:
    cs, c0 = [float(c) for c in p.c], float(p.c0)
    vars_txt = ", ".join(f"x{i + 1}" for i in range(p.n))
    out = ["1. STANDARD FORM",
           "The simplex works with a MAXIMIZATION problem, equality constraints "
           "and right-hand sides b ≥ 0. To get there:",
           ""]
    if maximize:
        out.append(f"  • Objective: Maximize Z = {format_objective_n(cs, c0)} "
                   "(already a maximization).")
    else:
        out.append(f"  • Objective: Minimize Z = {format_objective_n(cs, c0)} is equivalent to "
                   f"Maximize W = -Z = {format_objective_n([-c for c in cs], -c0)}. "
                   "At the end, Z = -W.")
    if abs(c0) > EPS:
        out.append(f"  • The constant {fmt_num(c0)} does not change the optimal point: the tableau "
                   "optimizes only the variable part, and the constant is added to the final value of Z.")
    if p.non_negative:
        out.append(f"  • Variables: {vars_txt} ≥ 0 (already non-negative).")
    else:
        subst = ", ".join(f"x{i + 1} = x{i + 1}⁺ - x{i + 1}⁻" for i in range(p.n))
        out.append(f"  • Free variables: substitute {subst}, with all "
                   f"⁺ and ⁻ parts ≥ 0 (columns: {', '.join(names)}).")
    out += ["  • Constraints (each inequality becomes an equality with auxiliary variables):", ""]
    legend = []
    for i, ((coefs, sign, b, flipped), (s, sc, a)) in enumerate(zip(rows, plan)):
        original = format_constraint_n(*p.constraints[i])
        coef = list(zip(_expand(coefs, p.non_negative), names))
        aux = []
        if s:
            aux.append((sc, s))
        if a:
            aux.append((1.0, a))
        standard = f"{terms_text(coef + aux)} = {fmt_num(b)}"
        line = f"      C{i + 1}: {original}"
        if flipped:
            line += (f"   → b < 0: multiply by -1: "
                     f"{format_constraint_n(coefs, sign, b)}")
        out.append(line)
        out.append(f"            standard form: {standard}")
        if s and sc > 0:
            legend.append(f"{s} = SLACK variable of C{i + 1} "
                          "(how far the constraint is from being tight)")
        if s and sc < 0:
            legend.append(f"{s} = SURPLUS variable of C{i + 1} "
                          "(how much the left-hand side exceeds b)")
        if a:
            legend.append(f"{a} = ARTIFICIAL variable of C{i + 1} "
                          "(only used to form the initial basis)")
    out += ["", "  Meaning of the auxiliary variables:"]
    out += [f"      {t}" for t in legend] if legend else ["      (none)"]
    artificials = [a for (_s, _c, a) in plan if a]
    if artificials:
        pen = " ".join(f"- M·{a}" for a in artificials)
        out += ["",
                f"  • Big M method: the artificials enter the objective with penalty "
                f"M = {fmt_num(BIG_M, 0)} (very large):",
                f"      Maximize {'Z' if maximize else 'W'} = "
                f"{format_objective_n(cs if maximize else [-c for c in cs])} {pen}",
                "    This way, any solution that keeps an artificial positive is "
                "'expensive' and the simplex tends to drive them to zero; if it cannot, "
                "the problem is infeasible."]
    else:
        out += ["", "  • All constraints are ≤ with b ≥ 0: no artificial variables "
                    "are needed (the initial basis is made of the slacks)."]
    return "\n".join(out)


def _initial_tableau_text(basis, matrix, columns, z_label, artificials_in_basis) -> str:
    out = ["2. INITIAL TABLEAU",
           "Each row needs a basic variable with an 'identity' column "
           "(1 in its own row and 0 in the others). These variables form the initial "
           "basis; all other variables (including the decision variables) start outside "
           "the basis, equal to 0.",
           ""]
    sol = ", ".join(f"{name} = {fmt_num(matrix[i][1])}" for i, name in enumerate(basis))
    out.append(f"  Initial basis: {', '.join(basis)}")
    out.append(f"  Initial basic solution: {sol}; other variables = 0.")
    out.append(f"  The {z_label} row holds the objective coefficients with the sign flipped "
               f"({z_label} - c·x = 0).")
    if artificials_in_basis:
        ops = "; ".join(f"{z_label} ← {z_label} - M·R{i + 1}  ({a})"
                        for i, a in artificials_in_basis)
        out.append("  Since there are artificials in the basis, their M coefficient in the "
                   f"{z_label} row must be zeroed (basic variables must have reduced cost 0): "
                   + ops + ".")
        out.append(f"  That is why the {z_label} row shows terms in M (M = "
                   f"{fmt_num(BIG_M, 0)}).")
    out.append("  The full tableau is in the 'Tableaux' tab (Tableau 0).")
    return "\n".join(out)


def _row_name(i: int) -> str:
    return f"R{i + 1}"


def _iteration_text(round_no, columns, basis, objective, candidates, column, row_ratios,
                    row, pivot, factors, z_factor, z_label, ties, bland=False) -> str:
    entering, leaving = columns[column], basis[row]
    out = [f"ITERATION {round_no}"]
    costs = ", ".join(f"{columns[j]} = {fmt_big_m(objective[j])}"
                      for j in range(len(columns)) if not _is_artificial(columns[j]))
    out.append(f"  {z_label} row (reduced costs): {costs}")
    neg_names = ", ".join(f"{columns[j]} ({fmt_big_m(objective[j])})" for j in candidates)
    out.append(f"  Optimality test: there are negative coefficients in the {z_label} row "
               f"→ {neg_names}. The current solution can still improve.")
    if bland:
        out.append(f"  ENTERING variable: {entering} (Bland's rule: the lowest index "
                   "among the candidates, used after consecutive degenerate pivots "
                   "to avoid cycling).")
    else:
        out.append(f"  ENTERING variable: {entering} (most negative coefficient, "
                   f"{fmt_big_m(objective[column])}; largest gain per unit rule).")
    out.append(f"  Minimum ratio test (b ÷ column of {entering}, positive entries only):")
    for i, (a, r) in enumerate(row_ratios):
        mark = "   ← minimum" if i == row else ""
        if r is None:
            out.append(f"      {_row_name(i)} ({basis[i]}): column = {fmt_num(a)} ≤ 0 → ignored")
        else:
            out.append(f"      {_row_name(i)} ({basis[i]}): "
                       f"{fmt_num(a * r)} ÷ {fmt_num(a)} = {fmt_num(r)}{mark}")
    if ties:
        out.append("      Note: tie in the minimum ratio → the next solution will be "
                   "degenerate (some basic variable will be 0).")
    out.append(f"  LEAVING variable: {leaving}  (row {_row_name(row)}).")
    out.append(f"  Pivot element: {fmt_num(pivot)}  (row {_row_name(row)}, column {entering}).")
    out.append("  Elementary row operations:")
    if abs(pivot - 1.0) > 1e-12:
        out.append(f"      {_row_name(row)} ← {_row_name(row)} ÷ {fmt_num(pivot)}   "
                   "(pivot becomes 1)")
    else:
        out.append(f"      {_row_name(row)} unchanged (pivot is already 1)")
    for i, f in factors:
        sign = "-" if f >= 0 else "+"
        out.append(f"      {_row_name(i)} ← {_row_name(i)} {sign} "
                   f"{fmt_num(abs(f))}·{_row_name(row)}")
    if abs(z_factor) > EPS:
        sign = "-" if z_factor >= 0 else "+"
        fz = fmt_big_m(abs(z_factor))
        fz = f"({fz})" if " " in fz else fz
        out.append(f"      {z_label} ← {z_label} {sign} {fz}·{_row_name(row)}")
    out.append(f"  Result: {entering} replaces {leaving} in the basis. New basis: "
               + ", ".join(basis[:row] + [entering] + basis[row + 1:]) + ".")
    out.append(f"  Tableau {round_no} is in the 'Tableaux' tab.")
    return "\n".join(out)


def solve_simplex(p: ProblemN) -> SimplexResult:
    """Solve by tableau; >= and = constraints get artificial variables."""
    geometric = solve_graphical(p.to_two_var()) if p.n == 2 else None
    rows = _normalize(p)
    maximize = p.sense == "maximize"
    z_label = "Z" if maximize else "W"
    names = _decision_names(p.n, p.non_negative)
    n_decision = len(names)

    extras: List[str] = []
    plan = []
    n_s = n_a = 0
    for _coefs, sign, _b, _flip in rows:
        if sign == "<=":
            n_s += 1
            s = f"s{n_s}"
            extras.append(s)
            plan.append((s, 1.0, None))
        elif sign == ">=":
            n_s += 1
            n_a += 1
            s, a = f"s{n_s}", f"a{n_a}"
            extras += [s, a]
            plan.append((s, -1.0, a))
        else:
            n_a += 1
            a = f"a{n_a}"
            extras.append(a)
            plan.append((None, 0.0, a))
    columns = names + extras
    n_cols = len(columns)
    artificials = [a for (_s, _c, a) in plan if a]

    matrix = []
    basis: List[str] = []
    for (coefs, _sign, b, _flip), (s, sc, a) in zip(rows, plan):
        row = [0.0] * n_cols
        row[:n_decision] = _expand(coefs, p.non_negative)
        if s:
            row[columns.index(s)] = sc
        if a:
            row[columns.index(a)] = 1.0
        basis.append(a if a else s)
        matrix.append([row, b])

    cs, c0 = [float(c) for c in p.c], float(p.c0)
    direction = 1.0 if maximize else -1.0
    objective = [0.0] * n_cols + [0.0]
    objective[:n_decision] = [-direction * c for c in _expand(cs, p.non_negative)]
    for name in artificials:
        objective[columns.index(name)] = BIG_M
    artificials_in_basis = []
    for i, basis_name in enumerate(basis):
        if _is_artificial(basis_name):
            artificials_in_basis.append((i, basis_name))
            objective = [a - BIG_M * b for a, b in zip(objective, matrix[i][0] + [matrix[i][1]])]

    def copy_table():
        return [r[0] + [r[1]] for r in matrix] + [objective[:]]

    iterations = [Iteration(0, basis[:], columns[:], copy_table(), z_label=z_label)]
    steps = [
        _standard_form_text(p, rows, plan, names, maximize),
        _initial_tableau_text(basis, matrix, columns, z_label, artificials_in_basis),
    ]

    def result(status, text, x=None, z=None, variables=None):
        return SimplexResult(status, text, x, z, iterations, steps, variables or {})

    degenerate = 0
    for round_no in range(1, MAX_ITERATIONS + 1):
        candidates = [
            j for j, value in enumerate(objective[:-1])
            if value < -EPS and not _is_artificial(columns[j])
        ]
        if not candidates:
            break
        bland = degenerate >= 3
        column = min(candidates) if bland else min(candidates, key=lambda j: objective[j])
        row_ratios = []
        for i in range(len(matrix)):
            a = matrix[i][0][column]
            row_ratios.append((a, matrix[i][1] / a if a > EPS else None))
        valid = [(r, i) for i, (a, r) in enumerate(row_ratios) if r is not None]
        iterations[-1].ratios = [r for (_a, r) in row_ratios]
        if not valid:
            iterations[-1].next_pivot = None
            iterations[-1].unbounded_column = column
            reason = [
                f"ITERATION {round_no}",
                f"  {columns[column]} enters (coefficient {fmt_big_m(objective[column])} "
                f"in the {z_label} row), "
                "but its column has no positive entry: there is no minimum "
                "ratio, i.e. the variable can grow indefinitely without "
                "violating any constraint."]
            if geometric is not None and geometric.status == "infeasible":
                reason.append("  Since the feasible region is empty (confirmed by the graphical "
                              "method), the correct conclusion is infeasibility.")
                steps.append("\n".join(reason))
                return result("infeasible", "Infeasible: the feasible region is empty.")
            if any(_is_artificial(name) and matrix[i][1] > EPS
                   for i, name in enumerate(basis)):
                reason.append("  There is a positive artificial in the basis: the problem is infeasible.")
                steps.append("\n".join(reason))
                return result("infeasible",
                              "Infeasible: an artificial variable remained positive.")
            reason.append("  Conclusion: the problem is UNBOUNDED (Z improves without limit).")
            steps.append("\n".join(reason))
            return result("unbounded",
                          "Unbounded problem: the objective function can improve without limit.")
        rmin, row = min(valid)
        if bland:
            row = min((i for r, i in valid if r - rmin < 1e-9),
                      key=lambda i: columns.index(basis[i]))
        degenerate = degenerate + 1 if rmin < 1e-9 else 0
        ties = sum(1 for r, _i in valid if abs(r - rmin) < 1e-9) > 1
        iterations[-1].next_pivot = (row, column)

        pivot = matrix[row][0][column]
        leaving = basis[row]
        matrix[row][0] = [v / pivot for v in matrix[row][0]]
        matrix[row][1] /= pivot
        factors = []
        for i in range(len(matrix)):
            if i == row:
                continue
            factor = matrix[i][0][column]
            if abs(factor) > EPS:
                factors.append((i, factor))
            matrix[i][0] = [a - factor * b for a, b in zip(matrix[i][0], matrix[row][0])]
            matrix[i][1] -= factor * matrix[row][1]
        z_factor = objective[column]
        objective = ([a - z_factor * b for a, b in zip(objective[:-1], matrix[row][0])]
                     + [objective[-1] - z_factor * matrix[row][1]])
        steps.append(_iteration_text(round_no, columns, basis, iterations[-1].table[-1],
                                     candidates, column, row_ratios, row, pivot,
                                     factors, z_factor, z_label, ties, bland))
        basis[row] = columns[column]
        iterations.append(Iteration(round_no, basis[:], columns[:], copy_table(),
                                    (row, column), columns[column], leaving, z_label=z_label))
    else:
        steps.append(f"The limit of {MAX_ITERATIONS} iterations was reached without convergence.")
        return result("iteration_limit", f"No convergence: limit of {MAX_ITERATIONS} "
                                         "iterations reached.")

    basic_artificials = [(i, name) for i, name in enumerate(basis)
                         if _is_artificial(name) and matrix[i][1] > EPS]
    check = geometric
    last = iterations[-1]

    end = ["STOPPING TEST"]
    end.append(f"  No negative coefficient is left in the {z_label} row (among the "
               "non-artificial variables) → the optimality criterion has been met.")

    geo_status = check.status if check is not None else None
    if geo_status == "infeasible" or basic_artificials:
        if basic_artificials:
            listing = ", ".join(f"{name} = {fmt_num(matrix[i][1])}" for i, name in basic_artificials)
            end.append(f"  Artificial(s) still positive in the basis: {listing}. The M penalty "
                       "cannot drive it/them to zero without violating some constraint → INFEASIBLE.")
        if geo_status == "infeasible":
            end.append("  The graphical method confirms: the intersection of the half-planes is empty.")
        steps.append("\n".join(end))
        return result("infeasible", "Infeasible: the feasible region is empty."
                      if geo_status == "infeasible"
                      else "Infeasible: an artificial variable remained positive.")
    if geo_status == "unbounded":
        end.append("  The graphical method finds an escape direction that improves Z without "
                   "leaving the feasible region → UNBOUNDED problem.")
        steps.append("\n".join(end))
        return result("unbounded",
                      "Unbounded problem: the objective function can improve without limit.")
    steps.append("\n".join(end))

    values = {name: 0.0 for name in columns}
    for i, name in enumerate(basis):
        values[name] = matrix[i][1]
    if p.non_negative:
        x = tuple(values[f"x{i + 1}"] for i in range(p.n))
    else:
        x = tuple(values[f"x{i + 1}+"] - values[f"x{i + 1}-"] for i in range(p.n))
    z = sum(c * xi for c, xi in zip(cs, x)) + c0

    check_note = None
    if (check is not None and check.status == "optimal" and check.best is not None
            and check.z_opt is not None):
        z_geo = check.z_opt
        if abs(z - z_geo) <= 1e-6 * max(1.0, abs(z_geo)):
            check_note = (f"  Cross-check: the graphical method also finds Z* = {fmt_num(z_geo)} "
                          "✓ (both methods agree).")
            if check.multiple_optima:
                check_note += ("\n  Note: there are multiple optimal solutions; the simplex returns "
                               "one of the optimal vertices.")
        else:
            check_note = (f"  Cross-check: the tableau value (Z = {fmt_num(z)}) differs from the "
                          f"graphical method (Z = {fmt_num(z_geo)}) due to Big M numerical error; "
                          "the graphical method value is used.")
            x = (check.best.x, check.best.y)
            z = z_geo

    res = ["READING THE SOLUTION",
           "  Non-basic variables are 0; basic variables take the value in the b column "
           "of the final tableau.",
           ""]
    basics = ", ".join(f"{name} = {fmt_num(matrix[i][1])}" for i, name in enumerate(basis))
    non_basics = [n for n in columns if n not in basis and not _is_artificial(n)]
    res.append(f"  Basic:     {basics}")
    if non_basics:
        res.append(f"  Non-basic: {', '.join(n + ' = 0' for n in non_basics)}")
    if not p.non_negative:
        for i in range(p.n):
            res.append(f"  Free variable: x{i + 1} = x{i + 1}+ - x{i + 1}- = "
                       f"{fmt_num(values[f'x{i + 1}+'])} - {fmt_num(values[f'x{i + 1}-'])} "
                       f"= {fmt_num(x[i])}")
    x_names = ", ".join(f"x{i + 1}" for i in range(p.n))
    res.append(f"  Solution: ({x_names}) = ({', '.join(fmt_num(v) for v in x)})")
    res.append(f"  Optimal value: Z = {substitution_n(cs, x, c0)} = {fmt_num(z)}"
               + ("" if maximize else f"   (W = -Z = {fmt_num(-z)})"))
    if any(abs(matrix[i][1]) < EPS for i in range(len(basis))):
        res.append("  Note: some basic variable is 0 → DEGENERATE basic solution.")
    partner = {}
    if not p.non_negative:
        for i in range(p.n):
            partner[f"x{i + 1}+"], partner[f"x{i + 1}-"] = f"x{i + 1}-", f"x{i + 1}+"
    alternatives = [columns[j] for j in range(n_cols)
                    if columns[j] not in basis and not _is_artificial(columns[j])
                    and partner.get(columns[j]) not in basis
                    and abs(last.table[-1][j]) < EPS]
    if alternatives:
        res.append("  Note: zero reduced cost on non-basic variable(s) ("
                   + ", ".join(alternatives) + ") → there may be alternative optimal solutions.")
    if check_note:
        res.append("")
        res.append(check_note)
    res += ["", "  Check against the original constraints:"]
    analysis_lines, binding = constraint_analysis_n(p.constraints, x)
    res += ["  " + line for line in analysis_lines]
    if p.non_negative:
        res.append("      " + "   ".join(f"x{i + 1} ≥ 0: {fmt_num(v)}"
                                        for i, v in enumerate(x)))
    if binding:
        res.append(f"  Binding constraints (slack 0): {', '.join(binding)}.")
    steps.append("\n".join(res))
    final_values = "\n".join(f"  x{i + 1}* = {fmt_num(v)}" for i, v in enumerate(x))
    steps.append("FINAL ANSWER\n"
                 f"{final_values}\n  Z* = {fmt_num(z)}\n"
                 f"  Iterations performed: {len(iterations) - 1}")
    return result(
        "optimal",
        f"Optimum found: Z* = {fmt_num(z)} at ({', '.join(fmt_num(v) for v in x)}).",
        x, z, values)


def format_tableau(iteration: Iteration) -> str:
    """Tableau in aligned columns, with the pivot in brackets and the ratios."""
    header = ["Basis"] + iteration.columns + ["b"]
    ratios = iteration.ratios
    has_ratio = ratios is not None
    if has_ratio:
        header.append("b/col")
    body = []
    n_basis = len(iteration.basis)
    for i, (name, row) in enumerate(zip(iteration.basis + [iteration.z_label], iteration.table)):
        is_z = i == n_basis
        fmt = fmt_big_m if is_z else (lambda v: fmt_num(v, 4))
        cells = [name]
        for j, v in enumerate(row[:-1]):
            text = fmt(v)
            if iteration.next_pivot == (i, j):
                text = f"[{text}]"
            cells.append(text)
        cells.append(fmt(row[-1]))
        if has_ratio:
            r = None if is_z or ratios is None else ratios[i]
            cells.append(fmt_num(r) if r is not None else "-")
        body.append(cells)
    lines = []
    if iteration.pivot is not None and iteration.entering:
        lines.append(f"Obtained by pivoting: {iteration.entering} enters, {iteration.leaving} leaves "
                     f"(R{iteration.pivot[0] + 1}, column {iteration.entering}).")
    lines += text_table(header, body)
    if iteration.next_pivot is not None:
        ri, co = iteration.next_pivot
        lines.append(f"Next pivot: [{fmt_big_m(iteration.table[ri][co])}] at R{ri + 1}, "
                     f"column {iteration.columns[co]} (minimum ratio).")
    elif iteration.unbounded_column is not None:
        name = iteration.columns[iteration.unbounded_column]
        lines.append(f"Final tableau: {name} could enter the basis, but its column has no "
                     "positive entries (no minimum ratio) → unbounded problem.")
    else:
        lines.append("Final tableau: no variable can enter the basis.")
    return "\n".join(lines)
