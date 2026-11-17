"""Solver tests. Run with: python -m unittest discover tests"""

import unittest

from solver import ProblemN, parse_objective, requires_big_m, solve_graphical, solve_simplex


def problem(sense, c, constraints, non_negative=True, c0=0.0):
    return ProblemN(sense, c, constraints, non_negative, c0)


class ParseObjectiveTest(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(parse_objective("3x1 + 5x2"), ([3.0, 5.0], 0.0))

    def test_prefix_aliases_and_constant(self):
        self.assertEqual(parse_objective("Z = 2x - y + 10"), ([2.0, -1.0], 10.0))

    def test_fractions_and_decimal_comma(self):
        self.assertEqual(parse_objective("1/2x1 - 3,5x2"), ([0.5, -3.5], 0.0))

    def test_unknown_variable(self):
        with self.assertRaises(ValueError):
            parse_objective("x1 + x3", 2)

    def test_missing_operator(self):
        with self.assertRaises(ValueError):
            parse_objective("3x1 5x2")


class SolverTest(unittest.TestCase):
    def assertOptimal(self, res, x, z):
        self.assertEqual(res.status, "optimal")
        for got, expected in zip(res.x, x):
            self.assertAlmostEqual(got, expected, places=6)
        self.assertAlmostEqual(res.z, z, places=6)

    def test_classic_max(self):
        p = problem("maximize", [3, 5],
                    [([1, 0], "<=", 4), ([0, 2], "<=", 12), ([3, 2], "<=", 18)])
        self.assertFalse(requires_big_m(p))
        self.assertOptimal(solve_simplex(p), (2, 6), 36)
        g = solve_graphical(p.to_two_var())
        self.assertEqual(g.status, "optimal")
        self.assertAlmostEqual(g.z_opt, 36)

    def test_min_with_big_m_and_constant(self):
        p = problem("minimize", [2, 3], [([1, 1], ">=", 4), ([1, 3], ">=", 6)], c0=5)
        self.assertTrue(requires_big_m(p))
        self.assertOptimal(solve_simplex(p), (3, 1), 14)

    def test_unbounded(self):
        p = problem("maximize", [1, 1], [([1, -1], "<=", 2)])
        self.assertEqual(solve_simplex(p).status, "unbounded")
        self.assertEqual(solve_graphical(p.to_two_var()).status, "unbounded")

    def test_infeasible(self):
        p = problem("maximize", [1, 1], [([1, 1], "<=", 2), ([1, 1], ">=", 5)])
        self.assertEqual(solve_simplex(p).status, "infeasible")
        self.assertEqual(solve_graphical(p.to_two_var()).status, "infeasible")

    def test_multiple_optima(self):
        p = problem("maximize", [2, 4], [([1, 2], "<=", 8), ([1, 0], "<=", 6)])
        g = solve_graphical(p.to_two_var())
        self.assertTrue(g.multiple_optima)
        self.assertAlmostEqual(solve_simplex(p).z, 16)

    def test_free_variables_and_negative_rhs(self):
        p = problem("minimize", [1, -2], [([1, 1], "==", 3), ([-1, 1], "<=", -1)],
                    non_negative=False)
        self.assertOptimal(solve_simplex(p), (2, 1), 0)

    def test_three_variables(self):
        p = problem("maximize", [3, 2, 4],
                    [([1, 1, 2], "<=", 4), ([2, 0, 3], "<=", 5), ([2, 1, 3], "<=", 7)])
        self.assertOptimal(solve_simplex(p), (2.5, 1.5, 0), 10.5)

    def test_three_variables_big_m(self):
        p = problem("minimize", [2, 3, 1], [([1, 1, 1], ">=", 10), ([2, 1, 0], "==", 8)])
        self.assertOptimal(solve_simplex(p), (4, 0, 6), 14)


if __name__ == "__main__":
    unittest.main()
