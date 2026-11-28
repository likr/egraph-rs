from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import math
import unittest
from typing import Any

import egraph as eg
import numpy as np
import numpy.typing as npt


def _make_grid(rows: int, cols: int) -> eg.CsrGraph:
    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            if c + 1 < cols:
                edges.append((u, u + 1))
            if r + 1 < rows:
                edges.append((u, u + cols))
    return eg.CsrGraph.from_edges(rows * cols, edges, directed=False, sanitize=True)


def _make_star(n: int) -> eg.CsrGraph:
    edges = [(0, i) for i in range(1, n)]
    return eg.CsrGraph.from_edges(n, edges, directed=False, sanitize=True)


def _make_disconnected_with_isolated() -> eg.CsrGraph:
    # Component 0: triangle (0,1,2); Component 1: edge (3,4); Isolated: vertex 5.
    return eg.CsrGraph.from_edges(
        6,
        [(0, 1), (1, 2), (0, 2), (3, 4)],
        directed=False,
        sanitize=True,
    )


def _make_directed_reciprocal() -> eg.CsrGraph:
    # Directed graph with reciprocal edges (0,1)+(1,0) and (1,2)+(2,1) plus one-way (2,3).
    return eg.CsrGraph.from_edges(
        4,
        [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3)],
        directed=True,
        sanitize=False,
    )


def _seeded_coords(n: int, seed: int = 42, scale: float = 1.0) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=(n, 2)).astype(np.float64)


def _collinear_coords(n: int) -> npt.NDArray[np.float64]:
    return np.column_stack(
        [np.linspace(-1.0, 1.0, n, dtype=np.float64), np.zeros(n, dtype=np.float64)]
    )


def _to_float_array(
    coords: npt.NDArray[np.float64], dtype_str: str
) -> npt.NDArray[np.floating[Any]]:
    if dtype_str == "float32":
        return coords.astype(np.float32)
    return coords.astype(np.float64)


class TestMaxentStressSolver(unittest.TestCase):
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
        fn: Callable[[], Any],
    ) -> None:
        with self.assertRaises(expected) as ctx:
            fn()
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

    def test_maxent_stress_solver(self) -> None:
        grid = _make_grid(4, 4)
        star = _make_star(10)
        base_grid = _seeded_coords(grid.num_nodes, seed=29, scale=1.0)

        for dtype_str in ("float64", "float32"):
            for use_bh in (True, False):
                l_run = eg.Layout2d.from_coords(_to_float_array(base_grid, dtype_str))
                l_step = eg.Layout2d.from_coords(_to_float_array(base_grid, dtype_str))
                s_run = eg.MaxentStressSolver(
                    grid,
                    q=0.5,
                    stages=3,
                    steps_per_stage=6,
                    use_barnes_hut=use_bh,
                    seed=19,
                    dtype=dtype_str,
                )
                s_step = eg.MaxentStressSolver(
                    grid,
                    q=0.5,
                    stages=3,
                    steps_per_stage=6,
                    use_barnes_hut=use_bh,
                    seed=19,
                    dtype=dtype_str,
                )
                self.assertAlmostEqual(s_run.effective_q, 0.5, places=12)
                self.assertAlmostEqual(s_run.current_alpha, 1.0, places=12)
                self.assertEqual(s_run.total_iterations, 18)
                self.assertIn("MaxentStressSolver", repr(s_run))

                s_run.run(l_run)
                s_step.step(l_step)
                s_step.run_steps(l_step, 7)
                s_step.run(l_step)
                self.assertTrue(s_run.is_finished)
                self.assertLess(s_run.current_alpha, 1.0)
                self.assertTrue(l_run.is_all_finite())
                np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))

        # Auto q detection: 4x4 grid has 4/16 = 25% <= 30% degree-1 -> q=0.0; star has 90% > 30% -> q=0.8.
        self.assertAlmostEqual(eg.MaxentStressSolver(grid).effective_q, 0.0, places=12)
        self.assertAlmostEqual(eg.MaxentStressSolver(star).effective_q, 0.8, places=12)

        # Degenerate graphs, collinear layout, isolated node preservation, stages=0
        s_zero = eg.MaxentStressSolver(grid, stages=0, steps_per_stage=5)
        self.assertTrue(s_zero.is_finished)
        self.assertEqual(s_zero.total_iterations, 0)

        g_disc = _make_disconnected_with_isolated()
        c_disc = _collinear_coords(g_disc.num_nodes)
        c_disc[5] = [12.5, -3.5]
        l_disc = eg.Layout2d.from_coords(c_disc)
        eg.MaxentStressSolver(g_disc, stages=2, steps_per_stage=4, seed=42).run(l_disc)
        self.assertTrue(l_disc.is_all_finite())
        np.testing.assert_array_equal(l_disc.coords(copy=False)[5], np.array([12.5, -3.5]))

        for g_deg in (
            eg.CsrGraph.from_edges(0, [], directed=False),
            eg.CsrGraph.from_edges(1, [], directed=False),
            eg.CsrGraph.from_edges(5, [], directed=False),
            _make_directed_reciprocal(),
        ):
            m = g_deg.num_nodes
            lay = eg.Layout2d.from_coords(_seeded_coords(m, seed=5) if m > 0 else np.zeros((0, 2)))
            eg.MaxentStressSolver(g_deg, stages=2, steps_per_stage=3).run(lay)
            self.assertTrue(lay.is_all_finite())

        # Validation errors
        self.assert_no_panic_raises((ValueError,), lambda: eg.MaxentStressSolver(grid, q=-2.0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.MaxentStressSolver(grid, q=math.nan))
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.MaxentStressSolver(grid, alpha_init=-1.0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.MaxentStressSolver(grid, alpha_decay=0.0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.MaxentStressSolver(grid, alpha_decay=1.5)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.MaxentStressSolver(grid, alpha_min=-0.1)
        )
        self.assert_no_panic_raises((ValueError,), lambda: eg.MaxentStressSolver(grid, tol=-0.1))
        self.assert_no_panic_raises((ValueError,), lambda: eg.MaxentStressSolver(grid, cg_tol=-0.1))
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.MaxentStressSolver(grid, cg_max_iters=0)
        )
        self.assert_no_panic_raises((ValueError,), lambda: eg.MaxentStressSolver(grid, theta=-0.5))
        self.assert_no_panic_raises((ValueError,), lambda: eg.MaxentStressSolver(grid, eps_sq=0.0))
        bad_lay = eg.Layout2d.zeros(grid.num_nodes)
        bad_lay.coords(copy=False)[0, 0] = math.inf
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.MaxentStressSolver(grid, stages=1, steps_per_stage=2).run(bad_lay)
        )
        wrong_len = eg.Layout2d.zeros(3)
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.MaxentStressSolver(grid, stages=1, steps_per_stage=2).run(wrong_len)
        )

    def test_baseline_solvers_gil_release_concurrent(self) -> None:
        graph = _make_grid(4, 4)
        base = _seeded_coords(graph.num_nodes, seed=61, scale=1.0)

        def run_hde() -> bool:
            return eg.hde(graph, dimensions=8, seed=42).is_all_finite()

        def run_ortmann() -> bool:
            lay = eg.Layout2d.from_coords(base)
            eg.SgdSolver.ortmann_sparse_stress(graph, num_pivots=4, iterations=10, seed=42).run(lay)
            return lay.is_all_finite()

        def run_maxent() -> bool:
            lay = eg.Layout2d.from_coords(base)
            eg.MaxentStressSolver(graph, stages=2, steps_per_stage=5, seed=42).run(lay)
            return lay.is_all_finite()

        def run_grip() -> bool:
            lay = eg.Layout2d.from_coords(base)
            eg.GripSolver(graph, rounds_min=2, rounds_max=5, seed=42).run(lay)
            return lay.is_all_finite()

        def run_sel() -> bool:
            lay = eg.Layout2d.from_coords(base)
            eg.SelLayoutSolver(graph, iterations=10, seed=42).run(lay)
            return lay.is_all_finite()

        def run_sfc() -> bool:
            lay = eg.SfcLayout(graph, seed=42).generate()
            return lay.is_all_finite()

        with ThreadPoolExecutor(max_workers=2) as pool:
            futs = [
                pool.submit(fn)
                for fn in (run_hde, run_ortmann, run_maxent, run_grip, run_sel, run_sfc)
            ]
            for fut in futs:
                self.assertTrue(fut.result())


if __name__ == "__main__":
    unittest.main()
