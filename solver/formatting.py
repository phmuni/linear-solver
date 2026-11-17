"""Numeric and text formatting shared by the solver and the interface.

Makes numbers look the way they do in textbooks: clean coefficients
("2x1 + x2" instead of "2x1 + 1x2") and rounded values without trailing zeros.
"""

from math import gcd


def fmt_num(value, places=4):
    """Format a number without trailing zeros (13.333333 -> '13.3333')."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(value) < 1e-9:
        return "0"
    text = f"{value:.{places}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in ("", "-", "-0", "0.0"):
        text = "0"
    return text


def paren(value):
    """fmt_num with parentheses around negatives: -2 -> '(-2)'."""
    text = fmt_num(value)
    return f"({text})" if text.startswith("-") else text


def fmt_big_m(value, big_m=1_000_000.0):
    """Show Big M tableau values symbolically.

    A number v = (n/d)·M + r (small d, small r) is written as 'nM/d + r':
    -3000000 -> '-3M', -2999982 -> '-3M + 18', 2500000 -> '5M/2'.
    Values with no M component (small |v| or no plausible decomposition) are
    formatted normally. Use only in tableaux where M is actually involved.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(value) < 1e3:
        return fmt_num(value)
    found = None
    for d in range(1, 13):
        n = int(round(value * d / big_m))
        if n == 0:
            continue
        rest = value - n * big_m / d
        if abs(rest) < 1e3:
            found = (n, d, rest)
            break
    if found is None:
        return fmt_num(value)
    n, d, rest = found
    g = gcd(abs(n), d)
    n, d = n // g, d // g
    num = "" if abs(n) == 1 else str(abs(n))
    m_part = ("-" if n < 0 else "") + num + "M" + (f"/{d}" if d > 1 else "")
    if abs(rest) < 1e-6:
        return m_part
    sign = "-" if rest < 0 else "+"
    return f"{m_part} {sign} {fmt_num(abs(rest), 3)}"


def sign_text(sign):
    """'<=' -> '≤', '>=' -> '≥', '==' -> '=' (display form)."""
    return {"<=": "≤", ">=": "≥", "==": "="}.get(sign, sign)


def terms_text(terms):
    """Build '2x + y - 3z' from [(2, 'x'), (1, 'y'), (-3, 'z')]."""
    parts = []
    for coef, var in terms:
        if abs(coef) < 1e-12:
            continue
        negative = coef < 0
        magnitude = abs(coef)
        coef_text = "" if (var and abs(magnitude - 1) < 1e-12) else fmt_num(magnitude)
        if not parts:
            parts.append(("-" if negative else "") + coef_text + var)
        else:
            parts.append((" - " if negative else " + ") + coef_text + var)
    return "".join(parts) if parts else "0"


def format_objective(c1, c2, c0=0.0):
    """(3, 5) -> '3x1 + 5x2'; (3, 5, -2) -> '3x1 + 5x2 - 2'."""
    return terms_text([(float(c1), "x1"), (float(c2), "x2"), (float(c0), "")])


def format_objective_n(coefs, c0=0.0):
    """([3, 5, 2], -1) -> '3x1 + 5x2 + 2x3 - 1' (any number of variables)."""
    return terms_text([(float(c), f"x{i + 1}") for i, c in enumerate(coefs)]
                      + [(float(c0), "")])


def format_constraint_n(coefs, sign, b):
    """([2, 1, 3], '<=', 10) -> '2x1 + x2 + 3x3 ≤ 10'."""
    terms = terms_text([(float(c), f"x{i + 1}") for i, c in enumerate(coefs)])
    return f"{terms} {sign_text(sign)} {fmt_num(b)}"


def format_constraint(a1, a2, sign, b):
    """(2, 1, '<=', 10) -> '2x1 + x2 ≤ 10'."""
    return f"{terms_text([(float(a1), 'x1'), (float(a2), 'x2')])} {sign_text(sign)} {fmt_num(b)}"


def z_expression(c1, c2, x, y, c0=0.0):
    """Substitute a point into the objective function:
    (3, 5, 2, 8) -> 'Z = 3·2 + 5·8 = 46'.
    """
    text = substitution(c1, c2, x, y, c0)
    if text == "0":
        return "Z = 0"
    return (f"Z = {text} = "
            f"{fmt_num(float(c1) * float(x) + float(c2) * float(y) + float(c0))}")


def substitution(a1, a2, x, y, c0=0.0):
    """(3, 2, 2, 6) -> '3·2 + 2·6' (zero terms are omitted).

    When c0 != 0, the constant is added at the end: '3·2 + 2·6 - 5'.
    """
    return substitution_n((a1, a2), (x, y), c0)


def substitution_n(coefs, xs, c0=0.0):
    """([3, 2, 1], [2, 6, 0], 0) -> '3·2 + 2·6 + 1·0' (only zero-coefficient terms are dropped)."""
    parts = []
    for coef, value in zip(coefs, xs):
        coef = float(coef)
        if abs(coef) < 1e-12:
            continue
        term = fmt_num(abs(coef)) + "·" + paren(float(value))
        if not parts:
            parts.append(("-" if coef < 0 else "") + term)
        else:
            parts.append((" - " if coef < 0 else " + ") + term)
    if abs(float(c0)) >= 1e-12:
        negative = float(c0) < 0
        prefix = (" - " if negative else " + ") if parts else ("-" if negative else "")
        parts.append(prefix + fmt_num(abs(float(c0))))
    return "".join(parts) if parts else "0"


def evaluate_constraint_n(coefs, sign, b, xs, tol=1e-6):
    """Return (satisfied, left_hand_side_value) for the point xs."""
    v = sum(a * x for a, x in zip(coefs, xs))
    t = tol * max(1.0, abs(b), abs(v))
    if sign == "<=":
        return v <= b + t, v
    if sign == ">=":
        return v >= b - t, v
    return abs(v - b) <= t, v


def evaluate_constraint(a1, a2, sign, b, x, y, tol=1e-6):
    """Return (satisfied, left_hand_side_value) for the point (x, y)."""
    return evaluate_constraint_n((a1, a2), sign, b, (x, y), tol)


def constraint_analysis_n(constraints, xs):
    """Text lines substituting the point xs into each constraint.

    constraints: list of (coefficients, sign, b). Shows the left-hand side
    value, the slack/surplus and whether the constraint is binding or not at
    the point. Returns (lines, names_of_binding_constraints).
    """
    lines, binding = [], []
    for i, (coefs, sign, b) in enumerate(constraints):
        ok, v = evaluate_constraint_n(coefs, sign, b, xs)
        slack = (b - v) if sign == "<=" else (v - b) if sign == ">=" else 0.0
        tight = abs(slack) <= 1e-6 * max(1.0, abs(b), abs(v))
        if sign == "==":
            note = ("equality satisfied" if ok
                    else f"equality violated (difference of {fmt_num(abs(v - b))})")
        elif not ok:
            note = f"VIOLATED (by {fmt_num(abs(slack))})"
        elif tight:
            note = "BINDING (slack 0: the constraint is 'tight')"
        else:
            label = "slack" if sign == "<=" else "surplus"
            note = f"non-binding ({label} of {fmt_num(abs(slack))})"
        mark = "✓" if ok else "✗"
        lines.append(
            f"    C{i + 1}: {substitution_n(coefs, xs)} = {fmt_num(v)} "
            f"{sign_text(sign)} {fmt_num(b)}  {mark}  {note}")
        if ok and (tight or sign == "=="):
            binding.append(f"C{i + 1}")
    return lines, binding


def constraint_analysis(p, x, y):
    """constraint_analysis_n for a two-variable Problem at the point (x, y)."""
    return constraint_analysis_n([((a1, a2), s, b) for a1, a2, s, b in p.constraints], (x, y))


def text_table(header, rows, left=(0,)):
    """Plain-text table with aligned columns (first column left-aligned)."""
    head = [str(c) for c in header]
    body = [[str(c) for c in row] for row in rows]
    widths = [max(len(col[k]) for col in [head] + body) for k in range(len(head))]

    def build(row):
        cells = [(c.ljust(widths[k]) if k in left else c.rjust(widths[k]))
                 for k, c in enumerate(row)]
        return " | ".join(cells)

    ruler = "-+-".join("-" * w for w in widths)
    return [build(head), ruler] + [build(row) for row in body]
