from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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


class TestV6Wave2Baselines(unittest.TestCase):
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
        fn: Callable[[], Any],
    ) -> None:
        with self.assertRaises(expected) as ctx:
            fn()
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

    def test_w2_exports_docstrings_and_signatures(self) -> None:
        w2_symbols = [
            "GemSolver",
            "GripSolver",
            "MaxentStressSolver",
            "PfsPrimeOverlapRemoval",
            "SelLayoutSolver",
            "SfcLayoutSolver",
            "hde",
            "hde_nd",
        ]
        for sym in w2_symbols:
            self.assertIn(sym, eg.__all__, f"{sym} missing from __all__")
            obj = getattr(eg, sym)
            self.assertTrue((obj.__doc__ or "").strip(), f"{sym} missing docstring")

        # Verify keyword-only parameters and defaults on W2 symbols.
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

        sig_lv = inspect.signature(eg.SgdSolver.largevis)
        self.assertEqual(sig_lv.parameters["gamma"].default, 7.0)
        self.assertEqual(sig_lv.parameters["degree_exponent"].default, 0.75)

        sig_dr = inspect.signature(eg.SgdSolver.drgraph)
        self.assertEqual(sig_dr.parameters["k_order"].default, 1)
        self.assertEqual(sig_dr.parameters["coarsen_stop_ratio"].default, 0.8)

        sig_cne = inspect.signature(eg.SgdSolver.graph_cne)
        self.assertEqual(sig_cne.parameters["objective"].default, "infonce")
        self.assertIs(sig_cne.parameters["row_normalize"].default, True)

        sig_maxent = inspect.signature(eg.MaxentStressSolver)
        self.assertIsNone(sig_maxent.parameters["q"].default)
        self.assertEqual(sig_maxent.parameters["alpha_min"].default, 0.008)
        self.assertIs(sig_maxent.parameters["use_barnes_hut"].default, True)

        sig_grip = inspect.signature(eg.GripSolver)
        self.assertEqual(sig_grip.parameters["rounds_min"].default, 5)
        self.assertEqual(sig_grip.parameters["rounds_max"].default, 30)

        sig_gem = inspect.signature(eg.GemSolver)
        self.assertEqual(sig_gem.parameters["edge_length"].default, 128.0)
        self.assertEqual(sig_gem.parameters["gravity"].default, 1.0 / 16.0)
        self.assertEqual(sig_gem.parameters["oscillation_angle"].default, math.pi)
        self.assertEqual(sig_gem.parameters["rotation_angle"].default, math.pi / 3.0)
        self.assertEqual(sig_gem.parameters["oscillation_sensitivity"].default, 1.0 / 3.0)
        self.assertIsNone(sig_gem.parameters["max_rounds"].default)
        self.assertIsNone(sig_gem.parameters["rotation_sensitivity"].default)

        sig_sel = inspect.signature(eg.SelLayoutSolver)
        self.assertEqual(sig_sel.parameters["mapping"].default, "sigmoid")
        self.assertEqual(sig_sel.parameters["layout_backend"].default, "fr")

        sig_pfs = inspect.signature(eg.PfsPrimeOverlapRemoval)
        self.assertEqual(sig_pfs.parameters["padding"].default, 0.0)
        sig_pfs_r = inspect.signature(eg.PfsPrimeOverlapRemoval.from_radii)
        self.assertEqual(sig_pfs_r.parameters["padding"].default, 0.0)

        sig_sfc = inspect.signature(eg.SfcLayoutSolver)
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
        self.assertIn("Ahmed", eg.SgdSolver.sgd2.__doc__ or "")
        self.assertIn("linlog_mode", eg.ForceAtlas2Solver.__doc__ or "")
        self.assertIn("Noack", eg.ForceAtlas2Solver.__doc__ or "")
        self.assertIn("IPSep-CoLa", eg.project_clustered_rectangle_no_overlap_2d.__doc__ or "")
        self.assertIn("MISSING_PDF", eg.SfcLayoutSolver.__doc__ or "")
        self.assertIn("muelder2008_space_filling", eg.SfcLayoutSolver.__doc__ or "")

    def test_w2_1a_hde_and_hde_nd(self) -> None:
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
                self.assertEqual(nd1.dim, out_dim)
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

    def test_w2_1b_w2_3_sgd_presets(self) -> None:
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
            constructors: list[tuple[str, Callable[[], eg.SgdSolver]]] = [
                (
                    "largevis",
                    lambda: eg.SgdSolver.largevis(
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
                    lambda: eg.SgdSolver.drgraph(
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
                    lambda: eg.SgdSolver.drgraph(
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
                    lambda: eg.SgdSolver.graph_cne(
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
                    lambda: eg.SgdSolver.graph_cne(
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
            deg_solvers: list[eg.SgdSolver] = [
                eg.SgdSolver.ortmann_sparse_stress(g_deg, num_pivots=3, iterations=5),
                eg.SgdSolver.largevis(g_deg, iterations=5),
                eg.SgdSolver.drgraph(g_deg, iterations=5),
                eg.SgdSolver.graph_cne(g_deg, iterations=5),
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
        self.assert_no_panic_raises((ValueError,), lambda: eg.SgdSolver.largevis(graph, gamma=-1.0))
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SgdSolver.largevis(graph, initial_lr=0.0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SgdSolver.largevis(graph, initial_lr=0.1, min_lr=0.5)
        )
        self.assert_no_panic_raises((ValueError,), lambda: eg.SgdSolver.drgraph(graph, k_order=0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.SgdSolver.drgraph(graph, b=0.0))
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SgdSolver.drgraph(graph, coarsen_stop_ratio=0.0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SgdSolver.drgraph(graph, coarsen_stop_ratio=1.5)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SgdSolver.graph_cne(graph, objective="bad")
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SgdSolver.graph_cne(graph, repulsion_weight=-0.5)
        )
        self.assert_no_panic_raises((TypeError,), lambda: eg.SgdSolver.largevis(graph, seed=True))
        bad_lay = eg.Layout2d.zeros(n)
        bad_lay.coords(copy=False)[0, 0] = math.nan
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SgdSolver.largevis(graph, iterations=3).run(bad_lay)
        )
        wrong_lay = eg.Layout2d.zeros(n + 1)
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SgdSolver.largevis(graph, iterations=3).run(wrong_lay)
        )
        # Cross-dtype conversion between float64 solver and float32 Layout2d succeeds cleanly.
        f32_lay = eg.Layout2d.from_coords(base.astype(np.float32))
        eg.SgdSolver.largevis(graph, iterations=3, dtype="float64").run(f32_lay)
        self.assertEqual(f32_lay.dtype, "float32")
        self.assertTrue(f32_lay.is_all_finite())

    def test_w2_2_maxent_stress_solver(self) -> None:
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

    def test_w2_4_grip_gem_sel_pfs_prime_sfc(self) -> None:
        graph = _make_barbell(5, bridge_len=2)
        n = graph.num_nodes
        base = _seeded_coords(n, seed=37, scale=2.0)

        # 1. GripSolver
        for dtype_str in ("float64", "float32"):
            l_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            l_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            g_run = eg.GripSolver(
                graph, rounds_min=3, rounds_max=8, seed=11, dtype=dtype_str
            )
            g_step = eg.GripSolver(
                graph, rounds_min=3, rounds_max=8, seed=11, dtype=dtype_str
            )
            self.assertGreaterEqual(g_run.num_levels, 1)
            self.assertIn("GripSolver", repr(g_run))
            g_run.run(l_run)
            g_step.step(l_step)
            g_step.run_steps(l_step, 2)
            g_step.run(l_step)
            self.assertTrue(g_run.is_finished)
            self.assertTrue(l_run.is_all_finite())
            np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))

        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.GripSolver(graph, rounds_min=10, rounds_max=5)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.GripSolver(graph, local_neighborhood_size=0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.GripSolver(graph, repulsion_scale=-0.1)
        )
        self.assert_no_panic_raises((ValueError,), lambda: eg.GripSolver(graph, edge_length=0.0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.GripSolver(graph, temp_r=-0.1))
        self.assert_no_panic_raises((ValueError,), lambda: eg.GripSolver(graph, temp_s=-1.0))

        # 2. GemSolver
        for dtype_str in ("float64", "float32"):
            l_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            l_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            m_run = eg.GemSolver(graph, max_rounds=12, seed=13, dtype=dtype_str)
            m_step = eg.GemSolver(graph, max_rounds=12, seed=13, dtype=dtype_str)
            self.assertIn("GemSolver", repr(m_run))
            m_run.run(l_run)
            m_step.step(l_step)
            m_step.run_steps(l_step, 3)
            m_step.run(l_step)
            self.assertTrue(m_run.is_finished)
            self.assertTrue(l_run.is_all_finite())
            np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))

        m_zero = eg.GemSolver(graph, max_rounds=0)
        self.assertTrue(m_zero.is_finished)
        self.assert_no_panic_raises((ValueError,), lambda: eg.GemSolver(graph, edge_length=0.0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.GemSolver(graph, gravity=-1.0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.GemSolver(graph, t_init=-1.0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.GemSolver(graph, t_max=0.0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.GemSolver(graph, t_min=-1.0))
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.GemSolver(graph, oscillation_angle=0.0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.GemSolver(graph, oscillation_angle=4.0)
        )
        self.assert_no_panic_raises((ValueError,), lambda: eg.GemSolver(graph, rotation_angle=0.0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.GemSolver(graph, rotation_angle=4.0))
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.GemSolver(graph, oscillation_sensitivity=-0.1)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.GemSolver(graph, rotation_sensitivity=-0.1)
        )
        self.assert_no_panic_raises((ValueError,), lambda: eg.GemSolver(graph, perturbation=-1.0))

        # 3. SelLayoutSolver (5 mappings x 2 backends x 2 dtypes)
        for dtype_str in ("float64", "float32"):
            for mapping in ("sigmoid", "linear", "tanh", "custom_logistic", "binary"):
                for backend in ("fr", "t_fdp"):
                    l_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
                    l_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
                    s_run = eg.SelLayoutSolver(
                        graph,
                        mapping=mapping,
                        layout_backend=backend,
                        iterations=8,
                        seed=41,
                        dtype=dtype_str,
                    )
                    s_step = eg.SelLayoutSolver(
                        graph,
                        mapping=mapping,
                        layout_backend=backend,
                        iterations=8,
                        seed=41,
                        dtype=dtype_str,
                    )
                    weights = s_run.modulated_weights()
                    self.assertEqual(weights.dtype, np.float64)
                    self.assertEqual(weights.ndim, 1)
                    self.assertTrue(np.all(np.isfinite(weights)))
                    self.assertTrue(math.isfinite(s_run.se_target))
                    self.assertIn("SelLayoutSolver", repr(s_run))
                    s_run.run(l_run)
                    s_step.step(l_step)
                    s_step.run_steps(l_step, 2)
                    s_step.run(l_step)
                    self.assertTrue(s_run.is_finished)
                    self.assertTrue(l_run.is_all_finite())
                    np.testing.assert_array_equal(
                        l_run.coords(copy=True), l_step.coords(copy=True)
                    )

        self.assert_no_panic_raises((ValueError,), lambda: eg.SelLayoutSolver(graph, tree_depth=0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.SelLayoutSolver(graph, pq_max_pops=0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.SelLayoutSolver(graph, mapping="bad"))
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SelLayoutSolver(graph, layout_backend="bad")
        )

        # 4. PfsPrimeOverlapRemoval (new + from_radii, zero overlap, strict oo_nni == 0.0)
        hw = np.full(n, 0.8, dtype=np.float64)
        hh = np.full(n, 0.5, dtype=np.float64)
        radii = np.full(n, 0.6, dtype=np.float64)
        coincident = _seeded_coords(n, seed=51, scale=0.2)
        coincident[1] = coincident[0]

        for dtype_str in ("float64", "float32"):
            l_run = eg.Layout2d.from_coords(_to_float_array(coincident, dtype_str))
            l_step = eg.Layout2d.from_coords(_to_float_array(coincident, dtype_str))
            p_run = eg.PfsPrimeOverlapRemoval(hw, hh, padding=0.05, seed=7, dtype=dtype_str)
            p_step = eg.PfsPrimeOverlapRemoval(hw, hh, padding=0.05, seed=7, dtype=dtype_str)
            self.assertEqual(p_run.total_iterations, 1)
            self.assertIn("PfsPrimeOverlapRemoval", repr(p_run))
            p_run.run(l_run)
            p_step.step(l_step)
            self.assertTrue(p_run.is_finished)
            self.assertEqual(p_run.last_overlap_count, 0)
            np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))
            self.assertEqual(
                len(eg.rectangle_overlaps(l_run, hw, hh)),
                0,
            )

            # Collinear 1D layout with from_radii and NOR check
            l_orig = eg.Layout2d.from_coords(_to_float_array(_collinear_coords(n), dtype_str))
            l_adj = eg.Layout2d.from_coords(_to_float_array(_collinear_coords(n), dtype_str))
            p_rad = eg.PfsPrimeOverlapRemoval.from_radii(radii, padding=0.0, seed=7, dtype=dtype_str)
            p_rad.run(l_adj)
            self.assertEqual(p_rad.last_overlap_count, 0)
            nor = eg.nor_metrics(l_orig, l_adj, radii, radii, graph=graph)
            self.assertAlmostEqual(float(nor["oo_nni"]), 0.0, places=12)

        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.PfsPrimeOverlapRemoval([1.0, 2.0], [1.0])
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.PfsPrimeOverlapRemoval([-1.0], [1.0])
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.PfsPrimeOverlapRemoval([math.nan], [1.0])
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.PfsPrimeOverlapRemoval([1.0], [1.0], padding=-0.1)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.PfsPrimeOverlapRemoval.from_radii([math.inf])
        )

        # 5. SfcLayoutSolver (hilbert & morton, zero co-location, permutation)
        for dtype_str in ("float64", "float32"):
            for curve in ("hilbert", "morton"):
                l_run = eg.Layout2d.zeros(n, dtype=dtype_str)
                l_step = eg.Layout2d.zeros(n, dtype=dtype_str)
                sfc_run = eg.SfcLayoutSolver(graph, curve=curve, scale=80.0, seed=9, dtype=dtype_str)
                sfc_step = eg.SfcLayoutSolver(graph, curve=curve, scale=80.0, seed=9, dtype=dtype_str)
                self.assertEqual(sfc_run.total_iterations, 1)
                self.assertIn("SfcLayoutSolver", repr(sfc_run))
                order = sfc_run.vertex_order()
                self.assertEqual(order.dtype, np.uint32)
                self.assertEqual(sorted(order.tolist()), list(range(n)))
                sfc_run.run(l_run)
                sfc_step.step(l_step)
                self.assertTrue(sfc_run.is_finished)
                np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))
                coords = l_run.coords(copy=False)
                self.assertTrue(np.all(np.abs(coords) <= 40.0 + 1e-4))
                degen = eg.layout_degeneracy(graph, l_run)
                self.assertEqual(degen["duplicate_nodes_f32"], 0)

        self.assert_no_panic_raises((ValueError,), lambda: eg.SfcLayoutSolver(graph, scale=0.0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.SfcLayoutSolver(graph, scale=math.nan))
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SfcLayoutSolver(graph, resolution_order=0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SfcLayoutSolver(graph, resolution_order=16)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SfcLayoutSolver(graph, resolution_order=1)
        )
        self.assert_no_panic_raises((ValueError,), lambda: eg.SfcLayoutSolver(graph, curve="peano"))

        # Shared degenerate topologies across Grip, Gem, Sel, PfsPrime, Sfc
        for g_deg in (
            eg.CsrGraph.from_edges(0, [], directed=False),
            eg.CsrGraph.from_edges(1, [], directed=False),
            eg.CsrGraph.from_edges(2, [(0, 1)], directed=False),
            eg.CsrGraph.from_edges(5, [], directed=False),
            _make_disconnected_with_isolated(),
            _make_directed_reciprocal(),
        ):
            m = g_deg.num_nodes
            init = _collinear_coords(m) if m > 0 else np.zeros((0, 2), dtype=np.float64)
            w2_deg_solvers: list[
                eg.GripSolver
                | eg.GemSolver
                | eg.SelLayoutSolver
                | eg.SfcLayoutSolver
                | eg.PfsPrimeOverlapRemoval
            ] = [
                eg.GripSolver(g_deg, rounds_min=2, rounds_max=4),
                eg.GemSolver(g_deg, max_rounds=4),
                eg.SelLayoutSolver(g_deg, iterations=4),
                eg.SfcLayoutSolver(g_deg),
                eg.PfsPrimeOverlapRemoval.from_radii(np.full(m, 0.25, dtype=np.float64)),
            ]
            for w2_solver in w2_deg_solvers:
                lay = eg.Layout2d.from_coords(init)
                w2_solver.run(lay)
                self.assertTrue(lay.is_all_finite())

        # Bad layouts rejected without panics
        bad_lay = eg.Layout2d.zeros(n)
        bad_lay.coords(copy=False)[0, 0] = math.nan
        wrong_lay = eg.Layout2d.zeros(2)
        bad_solvers: list[
            eg.GripSolver
            | eg.GemSolver
            | eg.SelLayoutSolver
            | eg.SfcLayoutSolver
            | eg.PfsPrimeOverlapRemoval
        ] = [
            eg.GripSolver(graph, rounds_min=1, rounds_max=2),
            eg.GemSolver(graph, max_rounds=2),
            eg.SelLayoutSolver(graph, iterations=2),
            eg.SfcLayoutSolver(graph),
            eg.PfsPrimeOverlapRemoval(hw, hh),
        ]
        for bad_solver in bad_solvers:

            def _run_bad(
                s: eg.GripSolver
                | eg.GemSolver
                | eg.SelLayoutSolver
                | eg.SfcLayoutSolver
                | eg.PfsPrimeOverlapRemoval = bad_solver,
            ) -> str:
                return s.run(bad_lay)

            def _run_wrong(
                s: eg.GripSolver
                | eg.GemSolver
                | eg.SelLayoutSolver
                | eg.SfcLayoutSolver
                | eg.PfsPrimeOverlapRemoval = bad_solver,
            ) -> str:
                return s.run(wrong_lay)

            self.assert_no_panic_raises((ValueError,), _run_bad)
            self.assert_no_panic_raises((ValueError,), _run_wrong)

    def test_w2_gil_release_concurrent(self) -> None:
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
            lay = eg.Layout2d.zeros(graph.num_nodes)
            eg.SfcLayoutSolver(graph, seed=42).run(lay)
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
