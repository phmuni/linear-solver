"""Linear programming problem definitions."""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Problem:
    """Linear programming problem with two variables (x1, x2).

    sense: "maximize" or "minimize".
    c: coefficients of x1 and x2 in the objective function Z.
    constraints: list of tuples (a1, a2, sign, b), where sign is one of
        "<=", ">=" and "==".
    non_negative: True when x1 >= 0 and x2 >= 0.
    c0: constant term of Z (Z = c1·x1 + c2·x2 + c0); it does not change the
        optimal point, only the value of Z.
    """

    sense: str
    c: Tuple[float, float]
    constraints: List[Tuple[float, float, str, float]]
    non_negative: bool = True
    c0: float = 0.0


@dataclass
class ProblemN:
    """Linear programming problem with n variables (x1..xn), used by the simplex.

    c: coefficients of Z; constraints: tuples (coefficients, sign, b), with one
    coefficient per variable. The other fields mean the same as in Problem.
    """

    sense: str
    c: List[float]
    constraints: List[Tuple[List[float], str, float]]
    non_negative: bool = True
    c0: float = 0.0

    @property
    def n(self) -> int:
        return len(self.c)

    def to_two_var(self) -> Problem:
        """Two-variable version (only valid when n == 2), for the graphical method."""
        return Problem(self.sense, (self.c[0], self.c[1]),
                       [(a[0], a[1], s, b) for a, s, b in self.constraints],
                       self.non_negative, self.c0)
