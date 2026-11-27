from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import inspect
import math
import unittest
from typing import Any, Callable

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


def _seeded_coords(n: int, seed: int = 42, scale: float = 1.0) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=(n, 2)).astype(np.float64)


def _to_float_array(
    coords: npt.NDArray[np.float64], dtype_str: str
) -> npt.NDArray[np.floating[Any]]:
    if dtype_str == "float32":
        return coords.astype(np.float32)
    return coords.astype(np.float64)


class TestForceBundlingAndValidation(unittest.TestCase):
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
        fn: Callable[[], Any],
    ) -> None:
        with self.assertRaises(expected) as ctx:
            fn()
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

    def test_force_atlas2_and_yifan_hu(self) -> None:
        graph = _make_grid(5, 5)
        n = graph.num_nodes
        base = _seeded_coords(n, seed=41, scale=1.0)

        sig_fa2 = inspect.signature(eg.ForceAtlas2Solver)
        self.assertIsNone(sig_fa2.parameters["max_displacement"].default)
        self.assertIs(sig_fa2.parameters["outbound_attraction_distribution"].default, False)

        for dtype_str in ("float64", "float32"):
            for use_bh in (True, False):
                l_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
                l_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
                fa1 = eg.ForceAtlas2Solver(
                    graph,
                    iterations=15,
                    use_barnes_hut=use_bh,
                    linlog_mode=True,
                    strong_gravity=not use_bh,
                    outbound_attraction_distribution=True,
                    seed=7,
                    dtype=dtype_str,
                )
                fa2 = eg.ForceAtlas2Solver(
                    graph,
                    iterations=15,
                    use_barnes_hut=use_bh,
                    linlog_mode=True,
                    strong_gravity=not use_bh,
                    outbound_attraction_distribution=True,
                    seed=7,
                    dtype=dtype_str,
                )
                self.assertEqual(fa1.num_nodes, n)
                self.assertEqual(fa1.total_iterations, 15)
                self.assertEqual(fa1.dtype, dtype_str)
                self.assertIn("ForceAtlas2Solver", repr(fa1))
                fa1.run(l_run)
                fa2.step(l_step)
                fa2.run_steps(l_step, 5)
                fa2.run(l_step)
                np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))

            # max_displacement caps large displacements and diverges from None while keeping step == run
            star = eg.CsrGraph.from_edges(5, [(0, 1), (0, 2), (0, 3), (0, 4)], directed=False, sanitize=True)
            wide = _to_float_array(_seeded_coords(5, seed=11, scale=200.0), dtype_str)
            l_unbounded = eg.Layout2d.from_coords(wide)
            l_cap_run = eg.Layout2d.from_coords(wide)
            l_cap_step = eg.Layout2d.from_coords(wide)
            fa_unbounded = eg.ForceAtlas2Solver(star, iterations=5, max_displacement=None, seed=7, dtype=dtype_str)
            fa_cap_r = eg.ForceAtlas2Solver(star, iterations=5, max_displacement=0.2, seed=7, dtype=dtype_str)
            fa_cap_s = eg.ForceAtlas2Solver(star, iterations=5, max_displacement=0.2, seed=7, dtype=dtype_str)
            fa_unbounded.run(l_unbounded)
            fa_cap_r.run(l_cap_run)
            fa_cap_s.step(l_cap_step)
            fa_cap_s.run(l_cap_step)
            np.testing.assert_array_equal(l_cap_run.coords(copy=True), l_cap_step.coords(copy=True))
            self.assertFalse(np.allclose(l_unbounded.coords(copy=True), l_cap_run.coords(copy=True)))
            max_disp_step = float(np.max(np.linalg.norm(l_cap_run.coords(copy=True) - wide, axis=1)))
            self.assertLessEqual(max_disp_step, 5.0 * 0.2 + 1e-4)

        # YifanHu multilevel (force >= 2 levels with min_coarsen_size=6 on 25-node grid)
        for dtype_str in ("float64", "float32"):
            l_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            l_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            yh1 = eg.YifanHuSolver(
                graph,
                max_iterations_per_level=15,
                min_coarsen_size=6,
                min_coarsen_ratio=0.9,
                seed=11,
                dtype=dtype_str,
            )
            yh2 = eg.YifanHuSolver(
                graph,
                max_iterations_per_level=15,
                min_coarsen_size=6,
                min_coarsen_ratio=0.9,
                seed=11,
                dtype=dtype_str,
            )
            self.assertGreaterEqual(yh1.num_levels, 2)
            self.assertEqual(yh1.current_level, yh1.num_levels - 1)
            self.assertEqual(yh1.num_nodes, n)
            self.assertEqual(yh1.dtype, dtype_str)
            self.assertIn("YifanHuSolver", repr(yh1))

            yh1.run(l_run)
            yh2.step(l_step)
            yh2.run_steps(l_step, 5)
            yh2.run(l_step)
            self.assertTrue(yh1.is_finished)
            self.assertEqual(yh1.current_level, 0)
            np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))

    def test_fdeb_solver_step_vs_run_and_polylines(self) -> None:
        graph = _make_grid(3, 3)
        base = np.array(
            [[float(c), float(r)] for r in range(3) for c in range(3)],
            dtype=np.float64,
        )
        for dtype_str in ("float64", "float32"):
            layout = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            f_run = eg.FdebSolver(graph, cycles=3, steps_per_cycle=6, dtype=dtype_str)
            f_step = eg.FdebSolver(graph, cycles=3, steps_per_cycle=6, dtype=dtype_str)

            self.assertEqual(f_run.dtype, dtype_str)
            self.assertEqual(f_run.num_edges, graph.num_edges)
            self.assertEqual(f_run.status, "initialized")
            self.assertIn("FdebSolver", repr(f_run))

            status = f_run.run(layout)
            self.assertEqual(status, "completed")
            self.assertTrue(f_run.is_finished)

            f_step.step(layout)
            f_step.run_steps(layout, 4)
            f_step.run(layout)
            self.assertTrue(f_step.is_finished)

            offsets = f_run.edge_offsets()
            pts_run = f_run.points()
            pts_step = f_step.points()
            self.assertEqual(offsets.dtype, np.uint32)
            self.assertEqual(offsets.shape, (graph.num_edges + 1,))
            self.assertEqual(int(offsets[0]), 0)
            self.assertEqual(int(offsets[-1]), pts_run.shape[0])
            self.assertEqual(pts_run.ndim, 2)
            self.assertEqual(pts_run.shape[1], 2)
            self.assertEqual(pts_run.dtype, np.float32 if dtype_str == "float32" else np.float64)
            self.assertTrue(np.all(np.isfinite(pts_run)))
            np.testing.assert_array_equal(offsets, f_step.edge_offsets())
            np.testing.assert_array_equal(pts_run, pts_step)

    def test_error_handling_and_gil_release(self) -> None:
        graph = _make_grid(3, 3)
        undir = graph
        layout = eg.Layout2d.zeros(9)
        bad_layout = eg.Layout2d.zeros(5)

        # No panics on bad inputs / domain errors
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.TFdpSolver(graph, backend="invalid")
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.TFdpSolver(graph, momentum=1.0)
        )
        self.assert_no_panic_raises(
            (TypeError,), lambda: eg.TFdpSolver(graph, iterations=True)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.TFdpSolver(graph).step(bad_layout)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.IbUmapSolver(graph, tau_scale=0.0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SgdSolver.snap_t_fdp(graph, rho_power=0.0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SgdSolver.snap_t_fdp(graph, degree_beta=-0.1)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SgdSolver.snap_t_fdp(graph, degree_beta=float("nan"))
        )
        self.assert_no_panic_raises(
            (TypeError,), lambda: eg.SgdSolver.snap_t_fdp(graph, degree_beta=True)
        )
        self.assert_no_panic_raises(
            (TypeError,), lambda: eg.SgdSolver.snap_t_fdp(graph, bounded_attraction=1)  # type: ignore[arg-type]
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SgdSolver.pacmap_graph(graph, n_neighbors=0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SgdSolver.sgd2(graph, batch_size=0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.ForceAtlas2Solver(graph, jitter_tolerance=0.0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.ForceAtlas2Solver(graph, max_displacement=0.0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.ForceAtlas2Solver(graph, max_displacement=float("nan"))
        )
        self.assert_no_panic_raises(
            (TypeError,), lambda: eg.ForceAtlas2Solver(graph, max_displacement=True)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.YifanHuSolver(graph, step_ratio=1.0)
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.PrismOverlapRemoval([1.0, 1.0], [1.0]),
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.PrismOverlapRemoval([1.0], [-0.5]),
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SeparationConstraint(0, 1, math.nan)
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.project_layout_axis(layout, 2, []),
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.project_1d(
                np.array([0.0, 1.0], dtype=np.float64),
                [eg.SeparationConstraint(0, 1, 1.0), eg.SeparationConstraint(1, 0, 1.0)],
            ),
        )
        self.assert_no_panic_raises((ValueError,), lambda: eg.cycle_edges(undir))
        self.assert_no_panic_raises((ValueError,), lambda: eg.remove_cycles(undir))
        self.assert_no_panic_raises((ValueError,), lambda: eg.assign_layers(undir))
        dag = eg.CsrGraph.from_edges(3, [(0, 1), (1, 2)], directed=True)
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.assign_layers(dag, method="unknown")
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.FdebSolver(graph).step(bad_layout)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.FdebSolver(graph, cycles=30)
        )

        # GIL release concurrency smoke test
        g_big = _make_grid(6, 6)
        coords_big = _seeded_coords(36, seed=99)

        def run_worker(seed_val: int) -> bool:
            s = eg.ForceAtlas2Solver(g_big, iterations=10, seed=seed_val)
            lay = eg.Layout2d.from_coords(coords_big)
            s.run(lay)
            return lay.is_all_finite()

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(run_worker, [1, 2]))
        self.assertEqual(results, [True, True])


if __name__ == "__main__":
    unittest.main()
