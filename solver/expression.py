"""Parsing of the objective function typed as a linear expression in x1..xn."""

import re
from typing import List, Tuple

_TERM = re.compile(
    r"([+-]?)\s*"
    r"(?:(\d+(?:[.,]\d+)?(?:\s*/\s*\d+(?:[.,]\d+)?)?)\s*[*·]?\s*)?"
    r"(x\d+|[xy])?",
    re.IGNORECASE,
)


def _number(text: str) -> float:
    if "/" in text:
        num, den = (float(t.strip().replace(",", ".")) for t in text.split("/"))
        if den == 0:
            raise ValueError("Division by zero in the objective function.")
        return num / den
    return float(text.replace(",", "."))


def parse_objective(text: str, n: int = 2) -> Tuple[List[float], float]:
    """'3x1 + 5x2 - 2' -> ([3.0, 5.0], -2.0), i.e. (coefficients, c0).

    Variables are x1..xn (x and y are also accepted as aliases for x1 and x2).
    Accepts a 'Z =' prefix, implicit coefficients (x1, -x2), '*' or '·' between
    coefficient and variable, decimal comma and simple fractions (1/2x).
    Repeated terms of the same variable are added together.
    """
    s = text.strip()
    s = re.sub(r"^\s*[zZ]\s*=", "", s).strip()
    if not s:
        raise ValueError("Please fill in the objective function.")
    coef = [0.0] * n
    c0 = 0.0
    pos = 0
    first = True
    while pos < len(s):
        m = _TERM.match(s, pos)
        if m is None or m.end() == pos:
            break
        sign, num, var = m.group(1), m.group(2), (m.group(3) or "").lower()
        if num is None and not var:
            break
        if not sign and not first:
            raise ValueError(f"Missing + or - before '{s[pos:].strip()}' in the objective function.")
        value = 1.0 if num is None else _number(num)
        value = -value if sign == "-" else value
        if not var:
            c0 += value
        else:
            index = {"x": 1, "y": 2}.get(var) or int(var[1:])
            if not 1 <= index <= n:
                raise ValueError(f"Variable {var} does not exist: the model has {n} variables "
                                 "(adjust the 'Variables' field).")
            coef[index - 1] += value
        pos = m.end()
        first = False
        while pos < len(s) and s[pos].isspace():
            pos += 1
    if pos < len(s) or first:
        raise ValueError(f"Could not understand the objective function: '{text.strip()}'. "
                         "Use something like 3x1 + 5x2 or 2x1 - x2 + 10.")
    return coef, c0
