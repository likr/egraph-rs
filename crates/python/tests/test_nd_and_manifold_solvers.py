from __future__ import annotations

import inspect
import unittest
from collections.abc import Callable
from typing import Any

import egraph as eg
import numpy as np


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


class TestNdAndManifoldSolvers(unittest.TestCase):
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
        fn: Callable[[], Any],
    ) -> None:
        with self.assertRaises(expected) as ctx:
            fn()
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

    def test_all_11_nd_solvers_and_fdeb_nd(self) -> None:
        g = _make_grid(3, 3)
        n = g.num_nodes
        for dtype_str, np_dtype in (("float64", np.float64), ("float32", np.float32)):
            for dims in (2, 3, 4):
                init_nd = eg.LayoutNd.init(n, dims, seed=7, scale=1.0, dtype=dtype_str)
                factories_nd: list[Callable[[], Any]] = [
                    lambda: eg.SgdSolver.full(g, iterations=4, seed=42, dtype=dtype_str),
                    lambda: eg.StressMajorizationSolver(g, max_iterations=4, dtype=dtype_str),
                    lambda: eg.KamadaKawaiSolver(g, max_iterations=4, dtype=dtype_str),
                    lambda: eg.MaxentStressSolver(
                        g,
                        stages=2,
                        steps_per_stage=2,
                        use_barnes_hut=(dims <= 3),
                        seed=42,
                        dtype=dtype_str,
                    ),
                    lambda: eg.FruchtermanReingoldSolver(g, iterations=4, seed=42, dtype=dtype_str),
                    lambda: eg.LinLogSolver(
                        g, iterations=4, use_barnes_hut=(dims <= 3), seed=42, dtype=dtype_str
                    ),
                    lambda: eg.YifanHuSolver(g, max_iterations_per_level=4, seed=42, dtype=dtype_str),
                    lambda: eg.ForceAtlas2Solver(
                        g, iterations=4, use_barnes_hut=(dims <= 3), seed=42, dtype=dtype_str
                    ),
                    lambda: eg.TsNetSolver.pure_2hop(
                        g,
                        backend="auto",
                        iterations_stage1=2,
                        iterations_stage2=2,
                        iterations_stage3=2,
                        seed=42,
                        dtype=dtype_str,
                    ),
                    lambda: eg.GuMapSolver.from_graph_dog(
                        g, max_2hop_per_node=16, num_landmarks=4, iterations=4, seed=42, dtype=dtype_str
                    ),
                ]
                for make_s in factories_nd:
                    sr = make_s()
                    ss = make_s()
                    l_run = init_nd.copy()
                    l_step = init_nd.copy()
                    sr.run(l_run)
                    ss.step(l_step)
                    ss.run_steps(l_step, 1)
                    ss.run(l_step)
                    self.assertTrue(l_run.is_all_finite())
                    self.assertEqual(l_run.dims, dims)
                    self.assertEqual(l_step.coords(copy=False).dtype, np_dtype)
                    np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))

            # FdebSolver (#11 N-D solver) on 3D LayoutNd: point_dims, points(), points_axis(axis), points_z()
            lay3 = eg.LayoutNd.init(n, 3, seed=11, scale=2.0, dtype=dtype_str)
            fdeb3 = eg.FdebSolver(g, cycles=2, steps_per_cycle=4, dtype=dtype_str)
            self.assertEqual(fdeb3.run(lay3), "completed")
            self.assertEqual(fdeb3.point_dims, 3)
            pts3 = fdeb3.points()
            self.assertEqual(pts3.ndim, 2)
            self.assertEqual(pts3.shape[1], 3)
            np.testing.assert_allclose(fdeb3.points_axis(2), pts3[:, 2])
            np.testing.assert_allclose(fdeb3.points_z(), pts3[:, 2])

    def test_all_6_manifold_solvers_step_eq_run(self) -> None:
        g = _make_grid(3, 3)
        n = g.num_nodes
        for kind in ("hyperbolic", "hyperbolic_lorentz", "spherical", "torus"):
            for dtype_str in ("float64", "float32"):
                init_m = eg.ManifoldLayout2d.init(kind, n, scale=0.25, seed=13, dtype=dtype_str)
                factories: list[Callable[[], Any]] = [
                    lambda: eg.SgdSolver.full(g, iterations=4, seed=42, dtype=dtype_str),
                    lambda: eg.StressMajorizationSolver(g, max_iterations=4, dtype=dtype_str),
                    lambda: eg.KamadaKawaiSolver(g, max_iterations=4, dtype=dtype_str),
                    lambda: eg.FruchtermanReingoldSolver(g, iterations=4, seed=42, dtype=dtype_str),
                    lambda: eg.TsNetSolver.from_graph_row_normalized(
                        g,
                        backend="auto",
                        iterations_stage1=2,
                        iterations_stage2=2,
                        iterations_stage3=2,
                        seed=42,
                        dtype=dtype_str,
                    ),
                    lambda: eg.GuMapSolver.from_graph_dog(
                        g, max_2hop_per_node=16, num_landmarks=4, iterations=4, seed=42, dtype=dtype_str
                    ),
                ]
                for make_s in factories:
                    sr = make_s()
                    ss = make_s()
                    lr = init_m.copy()
                    ls = init_m.copy()
                    sr.run(lr)
                    ss.step(ls)
                    ss.run_steps(ls, 2)
                    ss.run(ls)
                    self.assertTrue(lr.is_all_finite())
                    np.testing.assert_array_equal(lr.coords(copy=True), ls.coords(copy=True))

    def test_sgd_packed_window_tsnet_stages_and_auto_backend(self) -> None:
        g = _make_grid(4, 4)
        s_pack = eg.SgdSolver.from_packed_symmetric(
            3,
            [0, 1, 0],
            [1, 2, 2],
            [1.0, 1.0, 2.0],
            [1.0, 1.0, 0.25],
            [False, False, True],
            iterations=8,
            max_mu=0.5,
            seed=42,
        )
        self.assertTrue(s_pack.supports_manifold)
        self.assertEqual(s_pack.num_springs, 2)
        self.assertEqual(s_pack.num_hinges, 1)

        l2 = eg.Layout2d.init(3, seed=5)
        self.assertEqual(s_pack.run_schedule_window(l2, 0, 3), "running")
        self.assertEqual(s_pack.iteration, 3)
        s_pack.reset_schedule(iterations=6, schedule="linear", max_mu=0.8, seed=99)
        self.assertEqual(s_pack.total_iterations, 6)
        self.assertEqual(s_pack.run(l2), "completed")
        self.assertTrue(l2.is_all_finite())

        # TsNetStage, from_probabilities_with_stages, learning_rate_divisor, auto backend, two_step_random_walk
        knn, _, _ = eg.build_pure_2hop_knn(g)
        probs = eg.compute_knn_probabilities(knn, 3.0)
        stages = [
            eg.TsNetStage(3, exaggeration=4.0, lambda_kl=1.0, lambda_c=0.1, lambda_r=0.0),
            eg.TsNetStage(4, exaggeration=1.0, lambda_kl=1.0, lambda_c=0.01, lambda_r=0.05),
        ]
        self.assertEqual(stages[0].iterations, 3)
        self.assertAlmostEqual(stages[1].lambda_r, 0.05)
        self.assertIn("TsNetStage(", repr(stages[0]))

        ts = eg.TsNetSolver.from_probabilities_with_stages(
            probs, stages, backend="auto", learning_rate_divisor=4.0, seed=42
        )
        self.assertIn(ts.resolved_backend, ("exact", "ibfft", "barnes_hut"))
        self.assertEqual(ts.total_iterations, 7)
        lay = eg.Layout2d.init(16, seed=7)
        self.assertEqual(ts.run(lay), "completed")

        ts_rw = eg.TsNetSolver.from_graph_row_normalized(
            g,
            mode="two_step_random_walk",
            max_bridge_degree=12,
            backend="auto",
            learning_rate_divisor=3.0,
            iterations_stage1=2,
            iterations_stage2=2,
            iterations_stage3=2,
        )
        self.assertEqual(ts_rw.run(eg.Layout2d.init(16, seed=7)), "completed")

    def test_exported_signatures_and_docstrings(self) -> None:
        for name in eg.__all__:
            obj = getattr(eg, name)
            self.assertTrue((obj.__doc__ or "").strip(), f"missing docstring on {name}")
            if callable(obj):
                sig = inspect.signature(obj)
                self.assertIsNotNone(sig)


if __name__ == "__main__":
    unittest.main()
