from __future__ import annotations

from collections.abc import Callable
import inspect
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


def _make_barbell(clique_size: int, bridge_len: int = 1) -> eg.CsrGraph:
    n = 2 * clique_size + bridge_len - 1
    edges: list[tuple[int, int]] = []
    for i in range(clique_size):
        for j in range(i + 1, clique_size):
            edges.append((i, j))
    offset = clique_size + bridge_len - 1
    for i in range(clique_size):
        for j in range(i + 1, clique_size):
            edges.append((offset + i, offset + j))
    prev = clique_size - 1
    for b in range(bridge_len - 1):
        cur = clique_size + b
        edges.append((prev, cur))
        prev = cur
    edges.append((prev, offset))
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


class TestHdeAndSgdPresets(unittest.TestCase):
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
        fn: Callable[[], Any],
    ) -> None:
        with self.assertRaises(expected) as ctx:
            fn()
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

    def test_baseline_solver_exports_docstrings_and_signatures(self) -> None:
        baseline_symbols = [
            "GemSolver",
            "GripSolver",
            "MaxentStressSolver",
            "PfsPrimeOverlapRemoval",
            "SelLayoutSolver",
            "SfcLayout",
            "hde",
            "hde_nd",
        ]
        for sym in baseline_symbols:
            self.assertIn(sym, eg.__all__, f"{sym} missing from __all__")
            obj = getattr(eg, sym)
            self.assertTrue((obj.__doc__ or "").strip(), f"{sym} missing docstring")

        # Verify keyword-only parameters and defaults on baseline layout solvers.
        sig_hde = inspect.signature(eg.hde)
        self.assertEqual(sig_hde.parameters["dimensions"].default, 50)
        self.assertEqual(sig_hde.parameters["dimensions"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(sig_hde.parameters["pivots"].default)
        self.assertEqual(sig_hde.parameters["dtype"].default, "float64")

        sig_hde_nd = inspect.signature(eg.hde_nd)
        self.assertEqual(sig_hde_nd.parameters["output_dims"].default, 2)
        self.assertEqual(sig_hde_nd.parameters["output_dims"].kind, inspect.Parameter.KEYWORD_ONLY)

        sig_ort = inspect.signature(eg.SgdSolver.ortmann_sparse_stress)
        self.assertEqual(sig_ort.parameters["num_pivots"].default, 100)
        self.assertEqual(sig_ort.parameters["pivot_strategy"].default, "kmeans_sp")
        self.assertEqual(sig_ort.parameters["eps"].default, 0.1)

        sig_lv = inspect.signature(eg.LargeVisSolver)
        self.assertEqual(sig_lv.parameters["gamma"].default, 7.0)
        self.assertEqual(sig_lv.parameters["degree_exponent"].default, 0.75)

        sig_dr = inspect.signature(eg.DrGraphSolver)
        self.assertEqual(sig_dr.parameters["k_order"].default, 1)
        self.assertEqual(sig_dr.parameters["coarsen_stop_ratio"].default, 0.8)

        sig_cne = inspect.signature(eg.GraphCneSolver)
        self.assertEqual(sig_cne.parameters["objective"].default, "infonce")
        self.assertIs(sig_cne.parameters["row_normalize"].default, True)

        sig_maxent = inspect.signature(eg.MaxentStressSolver)
        self.assertIsNone(sig_maxent.parameters["q"].default)
        self.assertEqual(sig_maxent.parameters["alpha_min"].default, 0.008)
        self.assertEqual(sig_maxent.parameters["backend"].default, "auto")
        self.assertIsNone(sig_maxent.parameters["use_barnes_hut"].default)

        sig_grip = inspect.signature(eg.GripSolver)
        self.assertEqual(sig_grip.parameters["rounds_min"].default, 5)
        self.assertEqual(sig_grip.parameters["rounds_max"].default, 30)

        sig_gem = inspect.signature(eg.GemSolver)
        self.assertEqual(sig_gem.parameters["edge_length"].default, 128.0)
        self.assertEqual(sig_gem.parameters["gravity"].default, 1.0 / 16.0)
        self.assertEqual(sig_gem.parameters["oscillation_angle"].default, math.pi)
        self.assertEqual(sig_gem.parameters["rotation_angle"].default, math.pi / 3.0)
        self.assertEqual(sig_gem.parameters["oscillation_sensitivity"].default, 1.0 / 3.0)
        self.assertIsNone(sig_gem.parameters["max_iterations"].default)
        self.assertIsNone(sig_gem.parameters["rotation_sensitivity"].default)

        sig_sel = inspect.signature(eg.SelLayoutSolver)
        self.assertEqual(sig_sel.parameters["mapping"].default, "sigmoid")
        self.assertEqual(sig_sel.parameters["layout_backend"].default, "fr")

        sig_pfs = inspect.signature(eg.PfsPrimeOverlapRemoval)
        self.assertEqual(sig_pfs.parameters["padding"].default, 0.0)
        sig_pfs_r = inspect.signature(eg.PfsPrimeOverlapRemoval.from_radii)
        self.assertEqual(sig_pfs_r.parameters["padding"].default, 0.0)

        sig_sfc = inspect.signature(eg.SfcLayout)
        self.assertEqual(sig_sfc.parameters["curve"].default, "hilbert")
        self.assertEqual(sig_sfc.parameters["scale"].default, 100.0)
        self.assertIsNone(sig_sfc.parameters["resolution_order"].default)

        # Verify clarified docstrings.
        self.assertIn("S_RNG", eg.drng_preservation.__doc__ or "")
        self.assertIn("Eades", eg.drng_preservation.__doc__ or "")
        self.assertIn("S_GG", eg.dgg_preservation.__doc__ or "")
        self.assertIn("Eades", eg.dgg_preservation.__doc__ or "")
        self.assertIn("S_RNG", eg.evaluate_metrics.__doc__ or "")
        self.assertIn("S_GG", eg.evaluate_metrics.__doc__ or "")
        self.assertIn("Ahmed", eg.Sgd2Solver.__doc__ or "")
        self.assertIn("ForceAtlas2", eg.ForceAtlas2Solver.__doc__ or "")
        self.assertIn("linlog_mode", inspect.signature(eg.ForceAtlas2Solver).parameters)
        self.assertIn("IPSep-CoLa", eg.project_clustered_rectangle_no_overlap_2d.__doc__ or "")
        self.assertIn("MISSING_PDF", eg.SfcLayout.__doc__ or "")
        self.assertIn("muelder2008_space_filling", eg.SfcLayout.__doc__ or "")

    def test_hde_and_hde_nd(self) -> None:
        graph = _make_grid(4, 4)
        n = graph.num_nodes
        weighted = eg.CsrGraph.from_edges(
            4, [(0, 1), (1, 2), (2, 3), (3, 0)], weights=[1.0, 2.0, 1.5, 2.5], directed=False
        )

        for dtype_str, np_dtype in (("float64", np.float64), ("float32", np.float32)):
            l1 = eg.hde(graph, dimensions=8, seed=7, dtype=dtype_str)
            l2 = eg.hde(graph, dimensions=8, seed=7, dtype=dtype_str)
            self.assertEqual(l1.dtype, dtype_str)
            self.assertEqual(len(l1), n)
            self.assertTrue(l1.is_all_finite())
            c1 = l1.coords(copy=False)
            self.assertEqual(c1.dtype, np_dtype)
            np.testing.assert_array_equal(c1, l2.coords(copy=True))

            # normalize=True vs normalize=False on weighted graph
            lw_norm = eg.hde(weighted, dimensions=4, normalize=True, seed=11, dtype=dtype_str)
            lw_raw = eg.hde(weighted, dimensions=4, normalize=False, seed=11, dtype=dtype_str)
            self.assertTrue(lw_norm.is_all_finite() and lw_raw.is_all_finite())
            self.assertGreater(float(np.std(lw_norm.coords(copy=False))), 0.1)

            # Explicit pivots (with duplicate)
            lp = eg.hde(graph, pivots=[0, 5, 10, 5, 15], seed=7, dtype=dtype_str)
            self.assertEqual(len(lp), n)
            self.assertTrue(lp.is_all_finite())

            # hde_nd for output_dims=2, 3, 5
            for out_dim in (2, 3, 5):
                nd1 = eg.hde_nd(graph, dimensions=8, output_dims=out_dim, seed=13, dtype=dtype_str)
                nd2 = eg.hde_nd(
                    graph,
                    dimensions=8,
                    output_dims=out_dim,
                    pivots=np.array([0, 3, 12, 15], dtype=np.uint32),
                    seed=13,
                    dtype=dtype_str,
                )
                self.assertEqual(nd1.dims, out_dim)
                self.assertEqual(nd1.coords(copy=False).shape, (n, out_dim))
                self.assertEqual(nd1.coords(copy=False).dtype, np_dtype)
                self.assertTrue(nd1.is_all_finite())
                self.assertEqual(nd2.coords(copy=False).shape, (n, out_dim))
                self.assertTrue(nd2.is_all_finite())

        # Degenerate topologies
        empty = eg.CsrGraph.from_edges(0, [], directed=False)
        self.assertEqual(len(eg.hde(empty)), 0)
        self.assertEqual(eg.hde_nd(empty, output_dims=3).coords(copy=False).shape, (0, 3))

        singleton = eg.CsrGraph.from_edges(1, [], directed=False)
        np.testing.assert_array_equal(eg.hde(singleton).coords(copy=False), np.zeros((1, 2)))
        np.testing.assert_array_equal(
            eg.hde_nd(singleton, output_dims=3).coords(copy=False), np.zeros((1, 3))
        )

        edgeless = eg.CsrGraph.from_edges(5, [], directed=False)
        self.assertTrue(eg.hde(edgeless).is_all_finite())
        self.assertTrue(eg.hde(_make_disconnected_with_isolated()).is_all_finite())
        self.assertTrue(eg.hde(_make_directed_reciprocal()).is_all_finite())

        # Validation errors (no PanicException)
        self.assert_no_panic_raises((ValueError,), lambda: eg.hde(graph, dimensions=0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.hde_nd(graph, output_dims=0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.hde(graph, tol=0.0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.hde(graph, tol=math.nan))
        self.assert_no_panic_raises((ValueError,), lambda: eg.hde(graph, tol=-1.0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.hde(graph, max_sweeps=0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.hde(graph, pivots=[]))
        self.assert_no_panic_raises((IndexError,), lambda: eg.hde(graph, pivots=[n]))
        neg_w = eg.CsrGraph.from_edges(3, [(0, 1), (1, 2)], weights=[1.0, -0.5], sanitize=False)
        self.assert_no_panic_raises((ValueError,), lambda: eg.hde(neg_w))
        self.assert_no_panic_raises((TypeError,), lambda: eg.hde(graph, seed=True))
        self.assert_no_panic_raises((TypeError,), lambda: eg.hde(graph, dimensions=True))

    def test_sgd_presets(self) -> None:
        graph = _make_barbell(5, bridge_len=2)
        n = graph.num_nodes
        base = _seeded_coords(n, seed=17, scale=1.0)

        for dtype_str in ("float64", "float32"):
            # Ortmann Sparse Stress across all 3 pivot strategies
            for strat in ("kmeans_sp", "maxmin_random_sp", "maxmin_sp"):
                l_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
                l_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
                s_run = eg.SgdSolver.ortmann_sparse_stress(
                    graph,
                    num_pivots=4,
                    pivot_strategy=strat,
                    iterations=12,
                    eta_max=1.5,
                    eta_min=0.05,
                    seed=21,
                    dtype=dtype_str,
                )
                s_step = eg.SgdSolver.ortmann_sparse_stress(
                    graph,
                    num_pivots=4,
                    pivot_strategy=strat,
                    iterations=12,
                    eta_max=1.5,
                    eta_min=0.05,
                    seed=21,
                    dtype=dtype_str,
                )
                self.assertEqual(s_run.num_nodes, n)
                self.assertEqual(s_run.total_iterations, 12)
                self.assertEqual(s_run.dtype, dtype_str)
                self.assertIn("SgdSolver", repr(s_run))
                self.assertEqual(s_run.run(l_run), "completed")
                self.assertTrue(s_run.is_finished)
                s_step.step(l_step)
                s_step.run_steps(l_step, 4)
                s_step.run(l_step)
                self.assertGreaterEqual(s_step.current_eta, 0.0)
                np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))
                self.assertTrue(l_run.is_all_finite())

            # LargeVis, DRGraph, Graph-CNE
            constructors: list[tuple[str, Callable[[], Any]]] = [
                (
                    "largevis",
                    lambda: eg.LargeVisSolver(
                        graph,
                        iterations=10,
                        negative_samples=3,
                        gamma=5.0,
                        seed=23,
                        dtype=dtype_str,
                    ),
                ),
                (
                    "drgraph_k1",
                    lambda: eg.DrGraphSolver(
                        graph,
                        k_order=1,
                        negative_samples=3,
                        coarsen_stop_ratio=0.7,
                        iterations=12,
                        seed=23,
                        dtype=dtype_str,
                    ),
                ),
                (
                    "drgraph_k2",
                    lambda: eg.DrGraphSolver(
                        graph,
                        k_order=2,
                        negative_samples=3,
                        iterations=12,
                        seed=23,
                        dtype=dtype_str,
                    ),
                ),
                (
                    "graph_cne_infonce",
                    lambda: eg.GraphCneSolver(
                        graph,
                        objective="infonce",
                        negative_samples=3,
                        iterations=12,
                        row_normalize=True,
                        seed=23,
                        dtype=dtype_str,
                    ),
                ),
                (
                    "graph_cne_neg_tsne",
                    lambda: eg.GraphCneSolver(
                        graph,
                        objective="neg_tsne",
                        negative_samples=3,
                        iterations=12,
                        row_normalize=False,
                        seed=23,
                        dtype=dtype_str,
                    ),
                ),
            ]
            for label, make_solver in constructors:
                l_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
                l_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
                s_run = make_solver()
                s_step = make_solver()
                self.assertEqual(s_run.status, "initialized", label)
                self.assertEqual(s_run.iteration, 0, label)
                s_run.run(l_run)
                s_step.step(l_step)
                s_step.run_steps(l_step, 3)
                s_step.run(l_step)
                self.assertTrue(s_run.is_finished, label)
                self.assertTrue(s_step.is_finished, label)
                self.assertTrue(l_run.is_all_finite(), label)
                np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))

        # Degenerate graphs and collinear layouts
        for g_deg in (
            eg.CsrGraph.from_edges(0, [], directed=False),
            eg.CsrGraph.from_edges(1, [], directed=False),
            eg.CsrGraph.from_edges(5, [], directed=False),
            _make_disconnected_with_isolated(),
            _make_directed_reciprocal(),
        ):
            m = g_deg.num_nodes
            init = _collinear_coords(m) if m > 0 else np.zeros((0, 2), dtype=np.float64)
            deg_solvers: list[Any] = [
                eg.SgdSolver.ortmann_sparse_stress(g_deg, num_pivots=3, iterations=5),
                eg.LargeVisSolver(g_deg, iterations=5),
                eg.DrGraphSolver(g_deg, iterations=5),
                eg.GraphCneSolver(g_deg, iterations=5),
            ]
            for s_deg in deg_solvers:
                lay = eg.Layout2d.from_coords(init)
                s_deg.run(lay)
                self.assertTrue(lay.is_all_finite())

        # Validation guards
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SgdSolver.ortmann_sparse_stress(graph, pivot_strategy="bad")
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SgdSolver.ortmann_sparse_stress(graph, num_pivots=0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SgdSolver.ortmann_sparse_stress(graph, eps=0.0)
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.SgdSolver.ortmann_sparse_stress(graph, eta_min=2.0, eta_max=1.0),
        )
        self.assert_no_panic_raises((ValueError,), lambda: eg.LargeVisSolver(graph, gamma=-1.0))
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.LargeVisSolver(graph, initial_lr=0.0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.LargeVisSolver(graph, initial_lr=0.1, min_lr=0.5)
        )
        self.assert_no_panic_raises((ValueError,), lambda: eg.DrGraphSolver(graph, k_order=0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.DrGraphSolver(graph, b=0.0))
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.DrGraphSolver(graph, coarsen_stop_ratio=0.0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.DrGraphSolver(graph, coarsen_stop_ratio=1.5)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.GraphCneSolver(graph, objective="bad")
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.GraphCneSolver(graph, repulsion_weight=-0.5)
        )
        self.assert_no_panic_raises((TypeError,), lambda: eg.LargeVisSolver(graph, seed=True))
        bad_lay = eg.Layout2d.zeros(n)
        bad_lay.coords(copy=False)[0, 0] = math.nan
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.LargeVisSolver(graph, iterations=3).run(bad_lay)
        )
        wrong_lay = eg.Layout2d.zeros(n + 1)
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.LargeVisSolver(graph, iterations=3).run(wrong_lay)
        )
        # Cross-dtype conversion between float64 solver and float32 Layout2d succeeds cleanly.
        f32_lay = eg.Layout2d.from_coords(base.astype(np.float32))
        eg.LargeVisSolver(graph, iterations=3, dtype="float64").run(f32_lay)
        self.assertEqual(f32_lay.dtype, "float32")
        self.assertTrue(f32_lay.is_all_finite())


if __name__ == "__main__":
    unittest.main()
