"""Graphical method for linear programming with two variables.

Implemented steps:
  1. draw the line of each constraint (a1·x1 + a2·x2 = b) and find the valid
     half-plane using a test point;
  2. compute the pairwise intersections (constraints and axes), which are the
     vertex candidates, and test the feasibility of each one;
  3. evaluate Z at the feasible vertices (by the fundamental theorem of LP, if
     an optimum exists it lies at a vertex);
  4. conclude: optimal, unbounded, infeasible or region without vertices.

Unboundedness is checked with the recession cone of the feasible region: if
there is a ray starting at a feasible point that never leaves the region and
improves Z, the problem is unbounded. The feasible region polygon (half-plane
intersection clipped to a viewing box) is also computed for drawing in the
interface.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .formatting import (
    constraint_analysis,
    evaluate_constraint,
    fmt_num,
    format_constraint,
    format_objective,
    paren,
    sign_text,
    substitution,
    z_expression,
)
from .problem import Problem

TOL = 1e-6


@dataclass
class Vertex:
    x: float
    y: float
    feasible: bool
    z: Optional[float] = None
    origin: str = ""
    reason: str = ""
    also: List[str] = field(default_factory=list)
    pair: Optional[tuple] = None


@dataclass
class Step:
    title: str
    texts: List[str] = field(default_factory=list)


@dataclass
class GraphicalResult:
    vertices: List[Vertex]
    best: Optional[Vertex]
    z_opt: Optional[float]
    steps: List[Step]
    lines: List[dict]
    polygon: List[Tuple[float, float]]
    box: float
    truncated: bool
    multiple_optima: bool
    status: str
    direction: Optional[Tuple[float, float]] = None
    status_text: str = ""
    parallels: List[tuple] = field(default_factory=list)


def _z_of(v: Vertex) -> float:
    """Z of a feasible vertex (only feasible vertices have Z computed)."""
    if v.z is None:
        raise ValueError("An infeasible vertex has no Z value.")
    return v.z


def _feasible_point(x: float, y: float, p: Problem):
    """Test a point against every constraint; return (feasible, reason)."""
    if p.non_negative:
        if x < -TOL:
            return False, "violates x1 ≥ 0"
        if y < -TOL:
            return False, "violates x2 ≥ 0"
    for i, (a1, a2, sign, b) in enumerate(p.constraints):
        v = a1 * x + a2 * y
        tol = TOL * max(1.0, abs(b), abs(v))
        if sign == "<=" and v > b + tol:
            return False, f"violates C{i + 1}"
        if sign == ">=" and v < b - tol:
            return False, f"violates C{i + 1}"
        if sign == "==" and abs(v - b) > tol:
            return False, f"violates C{i + 1}"
    return True, ""


def _intersection(L1: dict, L2: dict) -> Optional[Tuple[float, float]]:
    """Intersection of the lines a1·x1 + a2·x2 = b (None if parallel)."""
    p, q, r = L1["a1"], L1["a2"], L1["b"]
    u, v, w = L2["a1"], L2["a2"], L2["b"]
    det = p * v - q * u
    if abs(det) < 1e-12:
        return None
    return ((r * v - q * w) / det, (p * w - r * u) / det)


def _coincident(L1: dict, L2: dict) -> bool:
    """True when two parallel lines are the same line."""
    p, q, r = L1["a1"], L1["a2"], L1["b"]
    u, v, w = L2["a1"], L2["a2"], L2["b"]
    return abs(p * w - u * r) < 1e-9 and abs(q * w - v * r) < 1e-9


def _clip_polygon(pol, a, b, c):
    """Sutherland-Hodgman: keep the part of the polygon with a*x + b*y <= c."""
    if not pol:
        return []

    def inside(pt):
        return a * pt[0] + b * pt[1] <= c + 1e-9

    def crossing(P, Q):
        fp = a * P[0] + b * P[1] - c
        fq = a * Q[0] + b * Q[1] - c
        t = fp / (fp - fq)
        return (P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1]))

    out = []
    n = len(pol)
    for i in range(n):
        current, nxt = pol[i], pol[(i + 1) % n]
        in_current, in_next = inside(current), inside(nxt)
        if in_current:
            out.append(current)
        if in_current != in_next:
            out.append(crossing(current, nxt))
    return out


def _compute_polygon(p: Problem, lines: List[dict], feasible: List[Vertex]):
    """Feasible region clipped to a box; return (polygon, B, truncated)."""
    references = []
    for v in feasible:
        references += [abs(v.x), abs(v.y)]
    for r in lines:
        if abs(r["a1"]) > 1e-12:
            references.append(abs(r["b"] / r["a1"]))
        if abs(r["a2"]) > 1e-12:
            references.append(abs(r["b"] / r["a2"]))
    B = 1.35 * max(references) if references else 10.0
    B = max(B, 10.0)

    if p.non_negative:
        pol: List[Tuple[float, float]] = [(0.0, 0.0), (B, 0.0), (B, B), (0.0, B)]
    else:
        pol = [(-B, -B), (B, -B), (B, B), (-B, B)]

    for (a1, a2, sign, b) in p.constraints:
        if abs(a1) < 1e-12 and abs(a2) < 1e-12:
            if sign == "<=" and 0 > b + TOL:
                pol = []
            elif sign == ">=" and 0 < b - TOL:
                pol = []
            elif sign == "==" and abs(b) > TOL:
                pol = []
            if not pol:
                break
            continue
        if sign == "<=":
            pol = _clip_polygon(pol, a1, a2, b)
        elif sign == ">=":
            pol = _clip_polygon(pol, -a1, -a2, -b)
        else:
            pol = _clip_polygon(_clip_polygon(pol, a1, a2, b), -a1, -a2, -b)
        if not pol:
            break

    truncated = any(abs(x) >= B - 1e-6 or abs(y) >= B - 1e-6 for (x, y) in pol)
    return pol, B, truncated


def _cone_rays(p: Problem) -> List[Tuple[float, float]]:
    """Unit directions of the feasible region's recession cone (extreme ray candidates).

    The recession cone of the region is made of the directions d that keep a
    point inside the region for any positive displacement:
    a1*dx + a2*dy <= 0 for '<=' constraints, >= 0 for '>=', = 0 for '='
    (plus dx >= 0 and dy >= 0 when the variables are non-negative).
    The extreme rays of the cone lie on the lines a1*dx + a2*dy = 0, so we
    test both directions of each of them, the axes and the gradient of Z.
    """
    g1, g2 = float(p.c[0]), float(p.c[1])

    def in_cone(dx, dy):
        for (a1, a2, sign, _b) in p.constraints:
            if abs(a1) < 1e-12 and abs(a2) < 1e-12:
                continue
            v = a1 * dx + a2 * dy
            if sign == "<=" and v > 1e-9:
                return False
            if sign == ">=" and v < -1e-9:
                return False
            if sign == "==" and abs(v) > 1e-9:
                return False
        if p.non_negative and (dx < -1e-9 or dy < -1e-9):
            return False
        return True

    candidates: List[Tuple[float, float]] = []
    for (a1, a2, _sign, _b) in p.constraints:
        if abs(a1) < 1e-12 and abs(a2) < 1e-12:
            continue
        candidates.append((-a2, a1))
        candidates.append((a2, -a1))
    if p.non_negative:
        candidates += [(0.0, 1.0), (1.0, 0.0)]
    candidates += [(g1, g2), (-g1, -g2)]

    rays = []
    for (dx, dy) in candidates:
        norm = (dx * dx + dy * dy) ** 0.5
        if norm < 1e-12:
            continue
        dx, dy = dx / norm, dy / norm
        if in_cone(dx, dy):
            rays.append((dx, dy))
    return rays


def _escape_direction(p: Problem) -> Tuple[bool, Optional[Tuple[float, float]]]:
    """Look for a recession cone ray that improves Z (unbounded problem)."""
    g1, g2 = float(p.c[0]), float(p.c[1])
    maximize = p.sense == "maximize"
    for (dx, dy) in _cone_rays(p):
        value = g1 * dx + g2 * dy
        if (value > 1e-9) if maximize else (value < -1e-9):
            return True, (dx, dy)
    return False, None


def _optimal_infinite_edge(p: Problem) -> bool:
    """True if a ray of the feasible region keeps Z constant (optimum along a half-line)."""
    g1, g2 = float(p.c[0]), float(p.c[1])
    return any(abs(g1 * dx + g2 * dy) < 1e-9 for (dx, dy) in _cone_rays(p))


def _lines_of(p: Problem) -> List[dict]:
    """List the constraint lines (used when solving and in the steps)."""
    lines = []
    for i, (a1, a2, sign, b) in enumerate(p.constraints):
        if abs(a1) < 1e-12 and abs(a2) < 1e-12:
            continue
        lines.append({"name": f"C{i + 1}", "a1": float(a1), "a2": float(a2),
                      "b": float(b), "sign": sign, "index": i})
    return lines


def solve_graphical(p: Problem) -> GraphicalResult:
    c1, c2, c0 = float(p.c[0]), float(p.c[1]), float(p.c0)
    maximize = p.sense == "maximize"

    lines = _lines_of(p)
    all_lines = list(lines)
    if p.non_negative:
        all_lines += [
            {"name": "x1 = 0", "a1": 1.0, "a2": 0.0, "b": 0.0, "sign": ">=", "index": -1},
            {"name": "x2 = 0", "a1": 0.0, "a2": 1.0, "b": 0.0, "sign": ">=", "index": -1},
        ]

    by_key = {}
    vertices: List[Vertex] = []
    parallels: List[tuple] = []
    for i in range(len(all_lines)):
        for j in range(i + 1, len(all_lines)):
            point = _intersection(all_lines[i], all_lines[j])
            pair_name = f"{all_lines[i]['name']} ∩ {all_lines[j]['name']}"
            if point is None:
                parallels.append((all_lines[i]["name"], all_lines[j]["name"],
                                  _coincident(all_lines[i], all_lines[j])))
                continue
            x, y = point
            key = (round(x, 6), round(y, 6))
            if key in by_key:
                by_key[key].also.append(pair_name)
                continue
            feasible, reason = _feasible_point(x, y, p)
            z = c1 * x + c2 * y + c0 if feasible else None
            v = Vertex(x, y, feasible, z, pair_name, reason)
            v.pair = (all_lines[i], all_lines[j])
            by_key[key] = v
            vertices.append(v)
    vertices.sort(key=lambda v: (v.x, v.y))

    feasible_vs = [v for v in vertices if v.feasible]
    polygon, box, truncated = _compute_polygon(p, lines, feasible_vs)
    unbounded, direction = _escape_direction(p)

    best: Optional[Vertex] = None
    multiple_optima = False
    if not polygon:
        status = "infeasible"
    elif unbounded:
        status = "unbounded"
    elif feasible_vs:
        status = "optimal"
        best = (max(feasible_vs, key=_z_of) if maximize
                else min(feasible_vs, key=_z_of))
        z_best = _z_of(best)
        tol_z = 1e-7 * max(1.0, abs(z_best))
        multiple_optima = (
            any(abs(_z_of(v) - z_best) < tol_z for v in feasible_vs if v is not best)
            or _optimal_infinite_edge(p))
    else:
        status = "no_vertices"

    if status == "optimal" and best is not None:
        extra = " (multiple optimal solutions)" if multiple_optima else ""
        status_text = (f"Optimal: Z* = {fmt_num(best.z)} at vertex "
                       f"({fmt_num(best.x)}, {fmt_num(best.y)}){extra}.")
    elif status == "unbounded":
        status_text = ("Unbounded: Z improves without limit along the escape "
                       "direction of the feasible region.")
    elif status == "no_vertices":
        status_text = ("Feasible region has no vertices: the vertex method "
                       "cannot locate the optimum.")
    else:
        status_text = "Infeasible: the feasible region is empty."

    steps = _build_steps(p, vertices, best, truncated, multiple_optima,
                         status, direction, parallels)
    return GraphicalResult(
        vertices=vertices,
        best=best,
        z_opt=best.z if best is not None else None,
        steps=steps,
        lines=lines,
        polygon=polygon,
        box=box,
        truncated=truncated,
        multiple_optima=multiple_optima,
        status=status,
        direction=direction,
        status_text=status_text,
        parallels=parallels,
    )


def _half_plane_test(a1, a2, sign, b):
    """Pick a test point and return (point, value, satisfied)."""
    point = (0.0, 0.0) if abs(b) > 1e-12 else (1.0, 1.0)
    ok, v = evaluate_constraint(a1, a2, sign, b, point[0], point[1])
    return point, v, ok


def _describe_line(r: dict) -> List[str]:
    a1, a2, b = r["a1"], r["a2"], r["b"]
    out = [f"    {r['name']}: {format_constraint(a1, a2, r['sign'], b)}",
           f"        Boundary line: {format_constraint(a1, a2, '==', b)}"]
    if abs(a2) < 1e-12:
        out.append(f"        Vertical line x1 = {fmt_num(b / a1)} "
                   "(parallel to the x2 axis).")
    elif abs(a1) < 1e-12:
        out.append(f"        Horizontal line x2 = {fmt_num(b / a2)} "
                   "(parallel to the x1 axis).")
    if abs(b) < 1e-12:
        out.append("        The line passes through the origin (b = 0).")
    else:
        points = []
        if abs(a1) > 1e-12:
            points.append(f"({fmt_num(b / a1)}, 0) setting x2 = 0")
        if abs(a2) > 1e-12:
            points.append(f"(0, {fmt_num(b / a2)}) setting x1 = 0")
        out.append("        Intercepts: " + "; ".join(points) + ".")
    if r["sign"] == "==":
        out.append("        Since it is an equality, the solution must lie "
                   "exactly on the line (there is no half-plane).")
        return out
    point, v, ok = _half_plane_test(a1, a2, r["sign"], b)
    point_name = "the origin (0, 0)" if point == (0.0, 0.0) else "the point (1, 1)"
    out.append(
        f"        Test with {point_name}: {substitution(a1, a2, *point)} = "
        f"{fmt_num(v)} {sign_text(r['sign'])} {fmt_num(b)} → "
        f"{'true' if ok else 'false'}.")
    if ok:
        out.append(f"        Valid half-plane: the side containing {point_name}.")
    else:
        out.append(f"        Valid half-plane: the side opposite to {point_name}.")
    return out


def _cramer_lines(L1: dict, L2: dict, x: float, y: float) -> List[str]:
    p, q, r = L1["a1"], L1["a2"], L1["b"]
    u, v, w = L2["a1"], L2["a2"], L2["b"]
    det = p * v - q * u
    return [
        "        System:   " + format_constraint(p, q, "==", r),
        "                  " + format_constraint(u, v, "==", w),
        f"        Determinant: det = {paren(p)}·{paren(v)} - {paren(q)}·{paren(u)} = {fmt_num(det)}",
        f"        x1 = ({paren(r)}·{paren(v)} - {paren(q)}·{paren(w)}) / {paren(det)} = {fmt_num(x)}",
        f"        x2 = ({paren(p)}·{paren(w)} - {paren(r)}·{paren(u)}) / {paren(det)} = {fmt_num(y)}",
    ]


def _feasibility_test_lines(p: Problem, x: float, y: float) -> List[str]:
    out = []
    for i, (a1, a2, sign, b) in enumerate(p.constraints):
        ok, v = evaluate_constraint(a1, a2, sign, b, x, y)
        out.append(f"          C{i + 1}: {substitution(a1, a2, x, y)} = {fmt_num(v)} "
                   f"{sign_text(sign)} {fmt_num(b)}  "
                   f"{'✓' if ok else '✗ violated'}")
    if p.non_negative:
        out.append(f"          x1 ≥ 0: {fmt_num(x)} ≥ 0  "
                   f"{'✓' if x >= -TOL else '✗ violated'}")
        out.append(f"          x2 ≥ 0: {fmt_num(y)} ≥ 0  "
                   f"{'✓' if y >= -TOL else '✗ violated'}")
    return out


def _sort_polygon(points):
    """Sort points counter-clockwise around the centroid."""
    cx = sum(pt[0] for pt in points) / len(points)
    cy = sum(pt[1] for pt in points) / len(points)
    return sorted(points, key=lambda pt: math.atan2(pt[1] - cy, pt[0] - cx))


def _area(points) -> float:
    n = len(points)
    s = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _infeasibility_diagnosis(p: Problem) -> List[str]:
    """Find the constraints (alone or in pairs) that make the region empty."""
    def empty(constraints):
        sub = Problem(p.sense, p.c, list(constraints), p.non_negative)
        pol, _, _ = _compute_polygon(sub, _lines_of(sub), [])
        return not pol

    base = p.constraints
    context = " (together with x1 ≥ 0 and x2 ≥ 0)" if p.non_negative else ""
    for i, r in enumerate(base):
        if empty([r]):
            return [f"C{i + 1} alone{context} already admits no point: "
                    f"{format_constraint(*r)}."]
    findings = []
    for i in range(len(base)):
        for j in range(i + 1, len(base)):
            if empty([base[i], base[j]]):
                findings.append(
                    f"C{i + 1} and C{j + 1} are incompatible with each other{context}: "
                    f"{format_constraint(*base[i])}  and  "
                    f"{format_constraint(*base[j])}.")
    if findings:
        return findings
    return ["No single constraint or pair of constraints is enough to cause "
            "the infeasibility: the conflict comes from the combination of three "
            "or more constraints."]


def _build_steps(p: Problem, vertices, best, truncated, multiple_optima,
                 status, direction, parallels=None) -> List[Step]:
    c1, c2, c0 = float(p.c[0]), float(p.c[1]), float(p.c0)
    maximize = p.sense == "maximize"
    verb = "Maximize" if maximize else "Minimize"
    parallels = parallels or []
    steps: List[Step] = []
    label = {id(v): f"V{i + 1}" for i, v in enumerate(vertices)}
    feasible_vs = [v for v in vertices if v.feasible]
    n_infeasible = len(vertices) - len(feasible_vs)
    lines = _lines_of(p)

    t1 = [
        "Each constraint a1·x1 + a2·x2 (sign) b is drawn on the x1x2 plane as the "
        "line a1·x1 + a2·x2 = b, which splits the plane into two half-planes. The "
        "valid half-plane holds the points that satisfy the inequality; it is "
        "identified by testing a simple point (the origin; if the line passes "
        "through the origin, the point (1, 1)).",
        "",
        f"Objective function: {verb} Z = {format_objective(c1, c2, c0)}.",
        f"Gradient ∇Z = ({fmt_num(c1)}, {fmt_num(c2)}): the direction in which Z "
        f"grows fastest. To {'maximize' if maximize else 'minimize'}, the level "
        f"line {format_objective(c1, c2)} = k is moved "
        f"{'along the gradient' if maximize else 'against the gradient'} "
        "until its last point of contact with the feasible region.",
        "",
    ]
    for r in lines:
        t1 += _describe_line(r)
        t1.append("")
    if p.non_negative:
        t1.append("    x1 ≥ 0 and x2 ≥ 0: non-negativity constraints; they "
                  "restrict the solution to the first quadrant (the axes x1 = 0 and "
                  "x2 = 0 are also boundaries of the region).")
    else:
        t1.append("    x1 and x2 free: there is no sign constraint; the axes do not "
                  "bound the region and vertices only come from intersecting "
                  "constraints.")
    steps.append(Step("1. Constraint lines", t1))

    t2 = [
        "The lines, taken two at a time"
        + (" (including the axes x1 = 0 and x2 = 0)" if p.non_negative else "")
        + ", intersect at points that are the vertex candidates. Each point is "
        "obtained by solving the 2×2 system with Cramer's rule and then "
        "substituted into every original constraint: if any of them fails, the "
        "point is discarded.",
        "",
        f"{len(vertices)} distinct candidates were found: "
        f"{len(feasible_vs)} feasible and {n_infeasible} infeasible.",
        "",
    ]
    for v in vertices:
        situation = "FEASIBLE" if v.feasible else f"INFEASIBLE ({v.reason})"
        t2.append(f"    {label[id(v)]} = {v.origin}  →  "
                  f"({fmt_num(v.x)}, {fmt_num(v.y)})  [{situation}]")
        if v.pair is not None:
            t2 += _cramer_lines(v.pair[0], v.pair[1], v.x, v.y)
        if v.also:
            t2.append("        Same point also obtained from: " + "; ".join(v.also)
                      + " → degenerate vertex (more than two lines meet here).")
        t2.append("        Feasibility test:")
        t2 += _feasibility_test_lines(p, v.x, v.y)
        t2.append("")
    if parallels:
        t2.append("    Pairs with no intersection (parallel lines):")
        for a, b, same in parallels:
            t2.append(f"        {a} ∥ {b}"
                      + (" (coincident lines)" if same else " (distinct)"))
        t2.append("")
    steps.append(Step("2. Vertex candidates and feasibility", t2))

    t3 = [
        "By the fundamental theorem of linear programming, if a linear problem "
        "has an optimal solution, it occurs at at least one vertex of the "
        "feasible region. So it is enough to evaluate the objective function "
        f"Z = {format_objective(c1, c2, c0)} at the feasible vertices and compare:",
        "",
    ]
    for v in feasible_vs:
        mark = "   ← best" if (best is not None and v is best) else ""
        t3.append(f"    {label[id(v)]} ({fmt_num(v.x)}, {fmt_num(v.y)}):  "
                  f"{z_expression(c1, c2, v.x, v.y, c0)}{mark}")
    if not feasible_vs:
        t3.append("    no feasible vertex was found.")
    elif len(feasible_vs) > 1:
        order = sorted(feasible_vs, key=lambda v: v.z, reverse=maximize)
        t3 += ["", f"Ranking (from {'highest' if maximize else 'lowest'} to "
                   f"{'lowest' if maximize else 'highest'} Z):"]
        t3.append("    " + ("  >  " if maximize else "  <  ").join(
            f"{label[id(v)]} (Z = {fmt_num(v.z)})" for v in order))
    steps.append(Step("3. Evaluating Z at the feasible vertices", t3))

    t4: List[str] = []
    if status == "infeasible":
        t4.append("The intersection of all half-planes is empty: there is no "
                  "feasible region.")
    else:
        if len(feasible_vs) >= 1:
            t4.append("Vertices of the feasible region: " + ", ".join(
                f"{label[id(v)]} ({fmt_num(v.x)}, {fmt_num(v.y)})"
                for v in feasible_vs) + ".")
        if truncated:
            t4.append("Type: UNBOUNDED region (open in at least one "
                      "direction); the drawing is cut off by the viewing window.")
        elif len(feasible_vs) >= 3:
            ordered = _sort_polygon([(v.x, v.y) for v in feasible_vs])
            t4.append(f"Type: BOUNDED region (convex polygon with "
                      f"{len(ordered)} vertices).")
            t4.append(f"Area of the feasible region: {fmt_num(_area(ordered))}.")
        elif len(feasible_vs) == 2:
            t4.append("Type: degenerate region (a line segment).")
        elif len(feasible_vs) == 1:
            t4.append("Type: degenerate region (a single point).")
    steps.append(Step("4. Feasible region", t4))

    t5: List[str] = []
    answer: List[str] = []
    if status == "optimal":
        assert best is not None
        comparative = "highest" if maximize else "lowest"
        t5.append(
            f"The {comparative} value of Z among the feasible vertices is "
            f"Z* = {fmt_num(best.z)}, at vertex {label[id(best)]} = "
            f"({fmt_num(best.x)}, {fmt_num(best.y)}). Since Z is linear, "
            "no point in the interior or on the edges of the region can beat "
            "this value: this vertex is the optimal solution of the problem.")
        others = sorted((v for v in feasible_vs
                         if v is not best and abs(v.z - best.z) >= 1e-7 * max(1.0, abs(best.z))),
                        key=lambda v: v.z, reverse=maximize)
        if others:
            second = others[0]
            t5.append(f"The second-best vertex is {label[id(second)]} with "
                      f"Z = {fmt_num(second.z)}; the optimum beats it by "
                      f"{fmt_num(abs(best.z - second.z))}.")
        t5 += ["", "Checking the optimal point against the constraints:"]
        analysis_lines, binding = constraint_analysis(p, best.x, best.y)
        t5 += analysis_lines
        if p.non_negative:
            t5.append(f"    x1 ≥ 0: x1 = {fmt_num(best.x)}"
                      f"{'  (BINDING)' if abs(best.x) < 1e-9 else '  (non-binding)'}")
            t5.append(f"    x2 ≥ 0: x2 = {fmt_num(best.y)}"
                      f"{'  (BINDING)' if abs(best.y) < 1e-9 else '  (non-binding)'}")
            if abs(best.x) < 1e-9:
                binding.append("x1 = 0")
            if abs(best.y) < 1e-9:
                binding.append("x2 = 0")
        if binding:
            t5.append("")
            t5.append("Binding constraints at the optimum (resource fully "
                      "used / boundary reached): " + ", ".join(binding) + ".")
            t5.append("Binding constraints are the ones that limit the gain in Z; "
                      "non-binding ones have slack and do not prevent improving the result.")
        if multiple_optima:
            tied = [v for v in feasible_vs
                    if abs(v.z - best.z) < 1e-7 * max(1.0, abs(best.z))]
            names = ", ".join(label[id(v)] for v in tied)
            if len(tied) == 1:
                t5 += ["", f"Note: the region is open and the level line of Z is "
                           f"parallel to an infinite edge starting at {names}: "
                           f"every point on that half-line has Z = {fmt_num(best.z)}, "
                           "so the solution is NOT unique."]
            else:
                t5 += ["", f"Note: vertices {names} have the same optimal value "
                           f"Z = {fmt_num(best.z)}, so the solution is NOT unique. The "
                           "whole edge joining these vertices is made of equally "
                           "optimal points."]
            if len(tied) == 2:
                a, b = tied
                t5.append(f"    Optimal points: (x1, x2) = (1 - t)·({fmt_num(a.x)}, "
                          f"{fmt_num(a.y)}) + t·({fmt_num(b.x)}, {fmt_num(b.y)}), "
                          "with 0 ≤ t ≤ 1.")
                t5.append("    (the level line of Z is parallel to this edge.)")
        answer = [f"    x1* = {fmt_num(best.x)}",
                  f"    x2* = {fmt_num(best.y)}",
                  f"    Z* = {fmt_num(best.z)}"
                  + ("   (one of infinitely many optimal solutions)" if multiple_optima else "")]
    elif status == "unbounded":
        t5.append(
            "The feasible region is open: there is an escape direction starting "
            "at a feasible point that never leaves the region. Along it Z "
            "improves at every step, so Z has no finite optimal value: "
            "the problem is unbounded.")
        if direction is not None:
            dx, dy = direction
            t5 += ["", f"Escape direction d = ({fmt_num(dx)}, {fmt_num(dy)}) "
                       "(shown by the arrow on the chart). Checking that d "
                       "does not leave the region (a·d must respect each constraint):"]
            for i, (a1, a2, sign, _b) in enumerate(p.constraints):
                v = a1 * dx + a2 * dy
                cond = {"<=": "≤ 0", ">=": "≥ 0", "==": "= 0"}[sign]
                t5.append(f"    C{i + 1}: {substitution(a1, a2, dx, dy)} = {fmt_num(v)} {cond}")
            if p.non_negative:
                t5.append(f"    d1 = {fmt_num(dx)} ≥ 0 and d2 = {fmt_num(dy)} ≥ 0")
            gain = c1 * dx + c2 * dy
            t5.append(f"    ∇Z·d = {substitution(c1, c2, dx, dy)} = {fmt_num(gain)} "
                      f"{'> 0' if gain > 0 else '< 0'}: Z "
                      f"{'increases' if gain > 0 else 'decreases'} at every step along d.")
            if feasible_vs:
                b0 = feasible_vs[0]
                t5 += ["", f"Example starting from {label[id(b0)]} = "
                           f"({fmt_num(b0.x)}, {fmt_num(b0.y)}):"]
                for t in (1, 10, 100, 1000):
                    px, py = b0.x + t * dx, b0.y + t * dy
                    t5.append(f"    t = {t}: point ({fmt_num(px)}, {fmt_num(py)}), "
                              f"Z = {fmt_num(c1 * px + c2 * py + c0)}")
        answer = ["    There is no finite optimal solution "
                  f"(Z → {'+∞' if maximize else '-∞'})."]
    elif status == "no_vertices":
        t5.append(
            "The feasible region has no vertices (it is open and has no 'corners'). "
            "The vertex method cannot locate the optimum in this case: additional "
            "constraints would be needed to close the region and create "
            "vertices.")
        answer = ["    The vertex method does not apply to this region."]
    else:
        t5.append(
            "No vertex candidate satisfies all constraints at the same time and "
            "the half-planes do not intersect: the feasible region "
            "is empty and the problem has no solution.")
        t5 += ["", "Conflict diagnosis:"]
        t5 += ["    " + line for line in _infeasibility_diagnosis(p)]
        answer = ["    Infeasible problem: no point satisfies all constraints."]
    if truncated and status == "optimal":
        t5 += ["", "Note: the feasible region is not bounded and the drawing is "
                   "cut off by the viewing window. Since Z does not improve "
                   "in the direction in which the region opens, the optimum found "
                   "at the vertices is still valid."]
    t5 += ["", "FINAL ANSWER"] + answer
    steps.append(Step("5. Conclusion", t5))
    return steps
