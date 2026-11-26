from __future__ import annotations

import inspect
import unittest
from typing import Any, Callable

import egraph as eg
import numpy as np
import numpy.typing as npt


def _make_grid(rows: int, cols: int, *, periodic: bool = False) -> eg.CsrGraph:
    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            if c + 1 < cols:
                edges.append((u, u + 1))
            elif periodic and cols > 2:
                edges.append((u, r * cols))
            if r + 1 < rows:
                edges.append((u, u + cols))
            elif periodic and rows > 2:
                edges.append((u, c))
    return eg.CsrGraph.from_edges(rows * cols, edges, directed=False, sanitize=True)


def _dense_symmetric_normalized_laplacian(
    graph: eg.CsrGraph,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    n = graph.num_nodes
    indptr = graph.indptr()
    indices = graph.indices()
    weights = graph.weights()
    adj = np.zeros((n, n), dtype=np.float64)
    for u in range(n):
        start = int(indptr[u])
        end = int(indptr[u + 1])
        for pos in range(start, end):
            v = int(indices[pos])
            if v == u:
                continue
            w = float(weights[pos]) if weights is not None else 1.0
            if w > 0.0:
                adj[u, v] = max(adj[u, v], w)
                adj[v, u] = max(adj[v, u], w)
    deg = adj.sum(axis=1)
    inv_sqrt = np.where(deg > 0.0, 1.0 / np.sqrt(np.maximum(deg, 1e-300)), 0.0)
    l_sym = np.eye(n, dtype=np.float64) - (inv_sqrt[:, None] * adj) * inv_sqrt[None, :]
    return l_sym, inv_sqrt


class TestAmgAndEffectiveResistance(unittest.TestCase):
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
        fn: Callable[[], Any],
    ) -> None:
        with self.assertRaises(expected) as ctx:
            fn()
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

    def test_amg_pcg_solver_hardened_and_cooperative(self) -> None:
        # 1. SPD regularized L_sym across gammas
        g = _make_grid(10, 10)
        n = g.num_nodes
        rng = np.random.default_rng(42)
        b = rng.normal(size=(n,))
        l_sym, _ = _dense_symmetric_normalized_laplacian(g)
        for gamma in (0.01, 1.0, 100.0, 10000.0):
            solver = eg.AmgPcgSolver.from_graph_regularized_laplacian(g, gamma=gamma)
            self.assertEqual(solver.num_nodes, n)
            self.assertGreaterEqual(solver.num_levels, 2)
            self.assertLessEqual(solver.coarsest_size, 64)
            x, info = solver.solve(b)
            self.assertTrue(info["converged"])
            self.assertLessEqual(info["relative_residual"], 1e-8)
            a_mat = np.eye(n) + gamma * l_sym
            true_rel = float(np.linalg.norm(b - a_mat @ x) / np.linalg.norm(b))
            self.assertLessEqual(true_rel, 1e-7)

        # 2. Raw CSR constructor + batched RHS + cooperative stepping equivalence
        indptr = np.array([0, 2, 5, 7], dtype=np.uint32)
        indices = np.array([0, 1, 0, 1, 2, 1, 2], dtype=np.uint32)
        values = np.array([2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0], dtype=np.float64)
        s_csr = eg.AmgPcgSolver.from_csr(3, indptr, indices, values)
        s_new = eg.AmgPcgSolver(3, indptr, indices, values)
        b3 = np.array([1.0, 0.0, 1.0], dtype=np.float64)
        x_sync, info_sync = s_csr.solve(b3)
        self.assertTrue(info_sync["converged"])

        status = s_new.start_solve(b3)
        self.assertIn(status, ("initialized", "converged"))
        if not s_new.is_finished:
            s_new.step()
            s_new.run()
        np.testing.assert_allclose(s_new.solution(), x_sync, atol=1e-12)

        b_mat = np.column_stack([b, 2.0 * b - 1.0])
        s_grid = eg.AmgPcgSolver.from_graph_laplacian(g, kind="combinatorial", shift=1e-3)
        x_mat, b_info = s_grid.solve_batch(b_mat)
        self.assertTrue(b_info["all_converged"])
        self.assertEqual(x_mat.shape, (n, 2))
        x_c0, _ = s_grid.solve(b_mat[:, 0])
        np.testing.assert_allclose(x_mat[:, 0], x_c0, atol=1e-12)

        # 3. Hub-dominated graph + singular disconnected + isolated + zero RHS + tiny/huge scale
        hub_edges = [(0, i) for i in range(1, 500)] + [(500, 501), (501, 502)]
        g_hub = eg.CsrGraph.from_edges(505, hub_edges, directed=False, sanitize=True)
        s_hub = eg.AmgPcgSolver.from_graph_laplacian(g_hub, kind="combinatorial", shift=0.0)
        self.assertLess(s_hub.operator_complexity, 2.0)
        b_hub = np.zeros(505, dtype=np.float64)
        b_hub[0] = 1.0
        b_hub[1] = -1.0
        b_hub[500] = 0.5
        b_hub[502] = -0.5
        x_hub, info_hub = s_hub.solve(b_hub)
        self.assertTrue(info_hub["converged"])
        self.assertEqual(float(x_hub[503]), 0.0)
        self.assertEqual(float(x_hub[504]), 0.0)

        x_zero, info_zero = s_hub.solve(np.zeros(505))
        self.assertTrue(info_zero["converged"])
        self.assertEqual(info_zero["iterations"], 0)
        np.testing.assert_array_equal(x_zero, np.zeros(505))

        x_tiny, info_tiny = s_grid.solve(b * 1e-12)
        self.assertTrue(info_tiny["converged"])
        self.assertGreater(info_tiny["iterations"], 0)
        self.assertTrue(np.all(np.isfinite(x_tiny)))

        # 4. Adversarial inputs must raise clean exceptions without PanicException
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.AmgPcgSolver(2, [0, 2, 4], [0, 1, 0, 1], [2.0, -1.0, -3.0, 2.0]),
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.AmgPcgSolver(2, [0, 2, 4], [0, 1, 0, 1], [-1.0, 0.0, 0.0, 2.0]),
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.AmgPcgSolver.from_graph_laplacian(g, kind="random_walk"),
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: s_grid.solve(np.full(n, float("nan")))
        )
        self.assert_no_panic_raises((ValueError,), lambda: s_grid.solve(np.ones(n - 1)))

    def test_linalg_and_neighborhood_exports_signatures_and_docstrings(self) -> None:
        exported_classes = (
            "AmgPcgSolver",
            "CsrMatrix",
            "EffectiveResistanceResult",
            "ResistanceCurvatureFlowResult",
            "KnnGraph",
            "SparseProbabilities",
            "Sparse12HopPairs",
            "DogVaradhanLandmarks",
        )
        exported_functions = (
            "anisotropic_laplacian",
            "compute_effective_resistances",
            "resistance_curvature_flow",
            "select_maxmin_sp_pivots",
            "pivot_mds_init",
            "koren_spectral_init",
            "normalize_layout_2d",
            "robust_normalize_layout_2d",
            "smallest_laplacian_eigenpairs",
            "spectral_moment_stats",
            "chebyshev_heat_spmv",
            "chebyshev_heat_spmv_matrix",
            "chebyshev_dog_jl_embedding",
            "chebyshev_dog_jl_embedding_matrix",
            "extract_1hop_2hop_csr",
            "extract_sparse_1hop_2hop_pairs",
            "build_pure_2hop_knn",
            "build_capped_12hop_knn",
            "compute_2hop_neighborhood_sizes",
            "compute_nodewise_2hop_perplexities",
            "compute_median_2hop_perplexity",
            "build_hybrid_adaptive_knn",
            "partial_bfs_knn",
            "compute_knn_probabilities",
            "graph_row_normalized_probabilities",
            "graph_12hop_row_normalized_probabilities",
            "compute_fuzzy_simplicial_set",
            "find_ab_params",
            "compute_horvitz_thompson_gamma",
            "compute_regularized_dog_varadhan_landmarks",
            "compute_regularized_dog_varadhan_landmarks_from_matrix",
        )
        for name in exported_classes + exported_functions:
            obj = getattr(eg, name)
            self.assertIs(obj, getattr(eg._core, name))
            self.assertIn(name, eg.__all__)
            self.assertTrue((obj.__doc__ or "").strip(), f"missing docstring on {name}")

        sig_pmds = inspect.signature(eg.pivot_mds_init)
        self.assertEqual(sig_pmds.parameters["num_pivots"].default, 50)
        self.assertEqual(sig_pmds.parameters["target_std"].default, 1.0)
        self.assertEqual(sig_pmds.parameters["dtype"].default, "float64")

        sig_hybrid = inspect.signature(eg.build_hybrid_adaptive_knn)
        self.assertEqual(sig_hybrid.parameters["k_factor"].default, 3.0)
        self.assertIsNone(sig_hybrid.parameters["k"].default)
        self.assertEqual(sig_hybrid.parameters["margin_gap"].default, 0.1)
        self.assertEqual(sig_hybrid.parameters["coarse_dim"].default, 10)
        self.assertEqual(sig_hybrid.parameters["rerank_factor"].default, 16)
        self.assertIsNone(sig_hybrid.parameters["max_row_k"].default)

        sig_dog = inspect.signature(eg.compute_regularized_dog_varadhan_landmarks)
        self.assertIsNone(sig_dog.parameters["degree"].default)
        self.assertEqual(sig_dog.parameters["num_landmarks"].default, 64)
        self.assertEqual(sig_dog.parameters["t_coarse"].default, 48.0)

        sig_pairs = inspect.signature(eg.extract_sparse_1hop_2hop_pairs)
        self.assertEqual(sig_pairs.parameters["max_2hop_per_node"].default, 32)
        self.assertEqual(sig_pairs.parameters["seed"].default, 42)

    def test_anisotropic_laplacian_dynamic_spectrum_and_amg_extreme_scale(self) -> None:
        # Star graph K_{1, 30}: L_comb has lambda_max = 31.0 >> 2.0
        n_star = 31
        star_edges = [(0, i) for i in range(1, n_star)]
        g_star = eg.CsrGraph.from_edges(n_star, star_edges, directed=False, sanitize=True)

        l_comb, ns_comb = eg.anisotropic_laplacian(
            g_star, alpha=0.0, beta=0.0, reweighted_degree_norm=False
        )
        self.assertEqual(len(l_comb), n_star)
        self.assertEqual(l_comb.num_nodes, n_star)
        self.assertTrue(l_comb.is_symmetric())
        self.assertAlmostEqual(l_comb.trace(), 60.0, places=12)
        self.assertAlmostEqual(l_comb.gershgorin_spectral_bound(), 60.0, places=12)
        lmax_pi = l_comb.estimate_lambda_max(num_iters=10, safety_factor=1.02, seed=42)
        lmax_lanc = l_comb.estimate_lambda_max_lanczos(num_steps=16, safety_factor=1.05, seed=42)
        self.assertGreaterEqual(lmax_pi, 30.0)
        self.assertGreaterEqual(lmax_lanc, 31.0)
        self.assertLessEqual(lmax_lanc, 31.0 * 1.05 + 1e-9)
        np.testing.assert_allclose(l_comb.spmv(ns_comb), np.zeros(n_star), atol=1e-12)

        # CL_sym at alpha=0.5, beta=-0.5, reweighted=True has spectrum in [0, 2]
        cl_sym, ns_cl = eg.anisotropic_laplacian(
            g_star, alpha=0.5, beta=-0.5, reweighted_degree_norm=True
        )
        evals_cl = np.linalg.eigvalsh(cl_sym.to_dense())
        self.assertTrue(np.all((evals_cl >= -1e-12) & (evals_cl <= 2.0 + 1e-12)))
        np.testing.assert_allclose(cl_sym.spmv(ns_cl), np.zeros(n_star), atol=1e-12)

        # Dynamic heat SpMV matches exact dense expm on L_comb (lambda_max = 31 >> 2)
        v = np.sin(0.37 * np.arange(n_star, dtype=np.float64))
        t = 0.2
        heat_dyn = eg.chebyshev_heat_spmv_matrix(l_comb, v, t)
        evals_comb, evecs_comb = np.linalg.eigh(l_comb.to_dense())
        heat_exact = evecs_comb @ (np.exp(-t * evals_comb) * (evecs_comb.T @ v))
        np.testing.assert_allclose(heat_dyn, heat_exact, atol=1e-8)

        # Dynamic DoG JL embedding and Varadhan landmarks from matrix (t_coarse=0.0)
        jl_dyn = eg.chebyshev_dog_jl_embedding_matrix(l_comb, jl_dim=8, seed=42)
        self.assertEqual(jl_dyn.shape, (n_star, 8))
        self.assertTrue(np.all(np.isfinite(jl_dyn)))

        lm_mat = eg.compute_regularized_dog_varadhan_landmarks_from_matrix(
            g_star, l_comb, num_landmarks=8, t_coarse=0.0, seed=42
        )
        self.assertFalse(lm_mat.used_coarse_scaffold)
        self.assertIsNone(lm_mat.coarse_table())
        self.assertGreaterEqual(lm_mat.degree, 40)
        self.assertTrue(np.all(np.isfinite(lm_mat.table())))

        # Extreme-scale AmgPcgSolver.from_csr_matrix invariance (1e-200 .. 1e200)
        reg_mat = l_comb.scale_and_shift(1.0, 1.0)
        amg = eg.AmgPcgSolver.from_csr_matrix(reg_mat)
        x_ref, info_ref = amg.solve(v)
        self.assertTrue(info_ref["converged"])
        for scale in (1e-160, 1e160, 1e-220, 1e220):
            x_sc, info_sc = amg.solve(v * scale)
            self.assertTrue(info_sc["converged"])
            self.assertEqual(info_sc["iterations"], info_ref["iterations"])
            np.testing.assert_allclose(x_sc / scale, x_ref, rtol=1e-7, atol=1e-9)

        # Zero matrix fast path
        g_zero = eg.CsrGraph.from_edges(4, [])
        l_zero, _ = eg.anisotropic_laplacian(g_zero, alpha=0.0, beta=0.0)
        self.assertEqual(l_zero.estimate_lambda_max_lanczos(), 0.0)
        v_z = np.array([1.0, -2.0, 3.0, -4.0])
        np.testing.assert_array_equal(eg.chebyshev_heat_spmv_matrix(l_zero, v_z, 1.0), v_z)
        lm_z = eg.compute_regularized_dog_varadhan_landmarks_from_matrix(
            g_zero, l_zero, t_coarse=0.0
        )
        self.assertTrue(np.all(np.isfinite(lm_z.table())))

    def test_effective_resistance_and_curvature_flow(self) -> None:
        # 1. Complete graph K_8 analytical exact check + Foster's theorem
        k8_edges = [(u, v) for u in range(8) for v in range(u + 1, 8)]
        g_k8 = eg.CsrGraph.from_edges(8, k8_edges, directed=False, sanitize=True)
        er_k8 = eg.compute_effective_resistances(g_k8)
        self.assertTrue(er_k8.used_exact)
        self.assertEqual(er_k8.num_components, 1)
        self.assertLess(er_k8.foster_rel_err, 1e-11)
        np.testing.assert_allclose(er_k8.effective_resistances(), np.full(28, 0.25), atol=1e-10)
        self.assertEqual(len(er_k8), 28)
        self.assertIn("EffectiveResistanceResult", repr(er_k8))
        self.assertAlmostEqual(er_k8.to_dict()["expected_foster"], 7.0)

        # 2. JL + AmgPcgSolver on 10x10 grid (N=100) + isolated nodes + Foster check
        grid10 = _make_grid(10, 10)
        er_jl = eg.compute_effective_resistances(grid10, jl_dim=64, seed=42, exact=False)
        self.assertFalse(er_jl.used_exact)
        self.assertTrue(er_jl.amg_all_converged)
        self.assertGreater(er_jl.amg_total_iterations, 0)
        self.assertAlmostEqual(er_jl.expected_foster, 99.0)
        self.assertLess(er_jl.foster_rel_err, 0.05)

        # 3. Barbell RCF + ManifoldLayout2d integration + Torus classification
        bb_edges = (
            [(u, v) for u in range(6) for v in range(u + 1, 6)]
            + [(u + 6, v + 6) for u in range(6) for v in range(u + 1, 6)]
            + [(5, 6)]
        )
        g_bb = eg.CsrGraph.from_edges(12, bb_edges, directed=False, sanitize=True)
        flow_bb = eg.resistance_curvature_flow(g_bb, steps=3, eta=0.2, kappa_max=2.0)
        self.assertEqual(len(flow_bb.step_history()), 4)
        self.assertAlmostEqual(float(np.mean(flow_bb.edge_lengths())), 1.0, places=12)
        u_arr = flow_bb.edge_u()
        v_arr = flow_bb.edge_v()
        lens = flow_bb.edge_lengths()
        bridge_mask = (u_arr == 5) & (v_arr == 6)
        self.assertEqual(int(np.sum(bridge_mask)), 1)
        self.assertGreater(float(lens[bridge_mask][0]), 1.0)
        self.assertLess(float(np.mean(lens[~bridge_mask])), 1.0)

        lg = flow_bb.length_graph()
        cg = flow_bb.conductance_graph()
        self.assertTrue(lg.is_weighted and cg.is_weighted)
        lg_w = lg.weights()
        cg_w = cg.weights()
        assert lg_w is not None and cg_w is not None
        np.testing.assert_allclose(lg_w * cg_w, 1.0, rtol=1e-12)
        ml = eg.ManifoldLayout2d.from_layout_scaled(
            "hyperbolic", eg.pivot_mds_init(lg, num_pivots=6, min_degree=1)
        )
        sgd = eg.SgdSolver.full(lg, t_max=10, seed=42)
        sgd.run_manifold(ml)
        self.assertTrue(ml.is_all_finite())

        # Torus T_{8x8}
        t_edges = []
        for r in range(8):
            for c in range(8):
                u = r * 8 + c
                t_edges.append((u, r * 8 + ((c + 1) % 8)))
                t_edges.append((u, ((r + 1) % 8) * 8 + c))
        g_t8 = eg.CsrGraph.from_edges(64, t_edges, directed=False, sanitize=True)
        flow_t8 = eg.resistance_curvature_flow(g_t8, steps=0)
        self.assertEqual(flow_t8.suggested_geometry, "torus")
        self.assertEqual(flow_t8.suggested_manifold, "torus")

        # 4. Adversarial checks without PanicException
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.compute_effective_resistances(grid10, jl_dim=0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.resistance_curvature_flow(grid10, eta=-0.1)
        )
        g_513 = eg.CsrGraph.from_edges(513, [(0, 1)], directed=False, sanitize=True)
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.compute_effective_resistances(g_513, exact=True)
        )


if __name__ == "__main__":
    unittest.main()
