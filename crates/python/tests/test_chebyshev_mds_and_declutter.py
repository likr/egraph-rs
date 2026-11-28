from __future__ import annotations

import math
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


class TestChebyshevMdsAndDeclutter(unittest.TestCase):
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
        fn: Callable[[], Any],
    ) -> None:
        with self.assertRaises(expected) as ctx:
            fn()
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

    def test_laplacians_and_chebyshev_kernel_primitives(self) -> None:
        g = _make_grid(4, 4)
        l_comb = eg.combinatorial_laplacian(g)
        l_sym, inv_sqrt = eg.symmetric_normalized_laplacian(g)
        l_rw = eg.random_walk_laplacian(g)
        l_shift = eg.shifted_normalized_laplacian(l_sym, lambda_max=2.0)
        l_disp = eg.laplacian(g, kind="combinatorial")
        np.testing.assert_allclose(l_comb.to_dense(), l_disp.to_dense(), atol=1e-12)
        self.assertEqual(len(inv_sqrt), 16)
        self.assertTrue(l_sym.is_symmetric())
        self.assertFalse(l_rw.is_symmetric())
        np.testing.assert_allclose(l_shift.to_dense(), l_sym.to_dense() - np.eye(16), atol=1e-12)

        deg_auto = eg.adaptive_chebyshev_degree(40.0)
        deg_scaled = eg.adaptive_chebyshev_degree_scaled(40.0, lambda_max=2.0)
        self.assertGreater(deg_auto, 0)
        self.assertEqual(deg_auto, deg_scaled)

        terms = eg.build_dog_terms(1.0, 20.0, 0.5, beta_floor=1e-4)
        reg_terms = eg.build_regularized_dog_terms(2.0, beta_floor=1e-4)
        self.assertGreaterEqual(len(terms), 2)
        self.assertGreaterEqual(len(reg_terms), 2)

        coeffs = eg.compute_chebyshev_bandpass_coefficients(terms, lambda_max=2.0, degree=20)
        self.assertEqual(coeffs.shape, (21,))
        proj = eg.project_psd_spectral_filter(terms, lambda_max=2.0, degree=20, eps_psd=1e-4)
        self.assertGreaterEqual(len(proj.terms), len(terms))
        self.assertGreaterEqual(proj.delta_psd, 0.0)
        self.assertTrue(math.isfinite(proj.h_min))

        ker = eg.ChebyshevDogKernel(g, degree=20, n_landmarks=8, jl_dim=8, seed=42)
        ker_lm = eg.ChebyshevDogKernel.build_with_landmarks(
            g, [0, 3, 12, 15], degree=20, jl_dim=8, seed=42
        )
        self.assertEqual(ker.num_nodes, 16)
        self.assertEqual(ker.num_landmarks, 8)
        self.assertEqual(ker.jl_dim, 8)
        self.assertGreater(ker.degree, 0)
        self.assertGreater(ker.t2, ker.t1)
        self.assertGreaterEqual(ker.alpha, 0.0)
        self.assertGreaterEqual(ker.delta_psd, 0.0)
        self.assertGreater(ker.beta_eff, 0.0)
        self.assertAlmostEqual(ker.beta_eff, ker.beta_auto, places=12)
        self.assertIn("ChebyshevDogKernel(", repr(ker))
        self.assertEqual(ker.landmarks().shape, (8,))
        self.assertEqual(ker.landmark_kernel_matrix().shape, (16, 8))
        self.assertEqual(ker.diag_estimate().shape, (16,))
        self.assertEqual(ker.jl_embedding().shape, (16, 8))
        self.assertEqual(ker.varadhan_landmark_table().shape, (16, 8))
        self.assertGreater(len(ker.terms_psd()), 0)
        self.assertIn("sigma_lambda", ker.moment_stats())
        applied = ker.apply_batch(l_sym, np.ones((16, 2), dtype=np.float64))
        self.assertEqual(applied.shape, (16, 2))
        self.assertEqual(ker_lm.num_landmarks, 4)

    def test_mds_nd_rhgh_and_layout_primitives(self) -> None:
        g = _make_grid(4, 4)
        dmat = eg.all_pairs_bfs(g)

        for dtype_str, np_dtype in (("float64", np.float64), ("float32", np.float32)):
            c_nd = eg.classical_mds(g, dims=3, dtype=dtype_str)
            c_from_d = eg.classical_mds_from_distances(dmat, dims=3, dtype=dtype_str)
            self.assertEqual(c_nd.dims, 3)
            self.assertEqual(c_nd.dtype, dtype_str)
            np.testing.assert_allclose(c_nd.coords(copy=True), c_from_d.coords(copy=True), atol=1e-5)

            p_lay = eg.pivot_mds(g, num_pivots=6, min_degree=1, seed=42, dtype=dtype_str)
            pivots = eg.select_maxmin_sp_pivots(g, 6, seed=42, min_degree=1)
            p_from_d = eg.pivot_mds_from_distances(dmat[:, pivots], seed=42, dtype=dtype_str)
            self.assertEqual(p_lay.num_nodes, 16)
            self.assertEqual(p_from_d.num_nodes, 16)
            self.assertEqual(p_from_d.dtype, dtype_str)

            r_true = eg.rd_mds(g, dims=3, degree_normalize=True, seed=42, dtype=dtype_str)
            r_false = eg.rd_mds(g, dims=3, degree_normalize=False, seed=42, dtype=dtype_str)
            self.assertEqual(r_true.dims, 3)
            self.assertEqual(r_false.dims, 3)

            # Layout2d scale & copy; LayoutNd init, copy, get, set, normalize_std
            l2 = eg.Layout2d.init(8, seed=7, scale=3.0, dtype=dtype_str)
            l2_cp = l2.copy()
            l2_cp.set_x(0, 99.0)
            self.assertNotAlmostEqual(l2.x(0), 99.0)

            lnd = eg.LayoutNd.init(6, 4, seed=7, scale=2.0, dtype=dtype_str)
            self.assertEqual(lnd.dims, 4)
            self.assertEqual(len(lnd), 6)
            lnd_cp = lnd.copy()
            lnd_cp.set(0, 2, 12.5)
            self.assertAlmostEqual(lnd_cp.get(0, 2), 12.5, places=5)
            self.assertNotAlmostEqual(lnd.get(0, 2), 12.5)
            lnd_cp.normalize_std(target_std=1.5)
            self.assertTrue(lnd_cp.is_all_finite())

            self.assert_no_panic_raises((IndexError,), lambda: lnd.get(99, 0))
            self.assert_no_panic_raises((IndexError,), lambda: lnd.set(0, 99, 1.0))

            # RhghLandmarks & SparseProbabilities.from_csr
            rhgh_res = eg.compute_pivoted_rhgh_landmarks(g, num_landmarks=6, seed=42)
            rhgh = rhgh_res.landmarks
            rhgh_stats = rhgh_res.stats
            self.assertIsInstance(rhgh, eg.RhghLandmarks)
            self.assertIn("n_landmarks", rhgh_stats)
            self.assertEqual(len(rhgh), 16)
            self.assertEqual(rhgh.num_landmarks, 6)
            self.assertGreater(rhgh.gamma, 0.0)
            self.assertIn("RhghLandmarks(", repr(rhgh))
            self.assertEqual(rhgh.table().shape, (16, 6))
            dog_from_rhgh = rhgh.to_dog_landmarks()
            self.assertEqual(dog_from_rhgh.num_landmarks, 6)
            _ = np_dtype  # silence unused loop var

        sp = eg.SparseProbabilities.from_csr(
            3,
            np.array([0, 2, 4, 6], dtype=np.uint32),
            np.array([1, 2, 0, 2, 0, 1], dtype=np.uint32),
            np.array([0.2, 0.1, 0.2, 0.2, 0.1, 0.2], dtype=np.float64),
        )
        self.assertEqual(sp.indptr().dtype, np.uint32)
        self.assertEqual(sp.nnz, 6)

    def test_declutter_transitive_reduction_and_cluster_result_classes(self) -> None:
        # TwinDeclutterer & declutter_twins_2d
        twin_coords = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [10.0, 0.0],
                [0.0, 10.0],
            ],
            dtype=np.float64,
        )
        for dtype_str, np_dtype in (("float64", np.float64), ("float32", np.float32)):
            l_one = eg.Layout2d.from_coords(twin_coords.astype(np_dtype))
            l_run = eg.Layout2d.from_coords(twin_coords.astype(np_dtype))
            l_fn_cmp = eg.Layout2d.from_coords(twin_coords.astype(np_dtype))
            td_one = eg.TwinDeclutterer(iterations=1, dtype=dtype_str)
            td_one.apply(l_one)
            self.assertGreater(td_one.last_cluster_count, 0)
            self.assertGreater(td_one.last_moved_count, 0)
            td_run = eg.TwinDeclutterer(iterations=4, dtype=dtype_str)
            self.assertEqual(td_run.dtype, dtype_str)
            self.assertIn("TwinDeclutterer(", repr(td_run))
            td_run.apply(l_run)
            self.assertGreater(td_run.last_pass_count, 0)
            moved_cmp = eg.declutter_twins_2d(l_fn_cmp)
            self.assertGreater(moved_cmp, 0)
            np.testing.assert_array_equal(l_run.coords(copy=True), l_fn_cmp.coords(copy=True))

        l_fn = eg.Layout2d.from_coords(twin_coords)
        moved = eg.declutter_twins_2d(l_fn)
        self.assertGreater(moved, 0)

        # Transitive reduction of a chain 0->1->2 with shortcut 0->2 removes 0->2
        dag = eg.CsrGraph.from_edges(3, [(0, 1), (1, 2), (0, 2)], directed=True, sanitize=True)
        tr = eg.transitive_reduction(dag)
        self.assertEqual(tr.num_nodes, 3)
        self.assertEqual(tr.num_edges, 2)
        self.assertFalse(tr.has_edge(0, 2))
        self.assertTrue(tr.has_edge(0, 1) and tr.has_edge(1, 2))

        # Verify all 5 typed cluster Result classes
        barbell, labels_true = eg.generate_sbm_graph([6, 6], p_intra=1.0, p_inter=0.05, seed=42)
        louv = eg.louvain_communities(barbell, seed=42)
        leid = eg.leiden_communities(barbell, seed=42)
        lp = eg.label_propagation(barbell, max_iterations=100, seed=42)
        spec = eg.spectral_clustering(barbell, k=2, max_eigen_iterations=200, kmeans_max_iterations=50, seed=42)
        km = eg.kmeans_plus_plus(np.vstack([np.zeros((4, 2)), np.full((4, 2), 10.0)]), k=2, seed=42)

        self.assertIsInstance(louv, eg.LouvainResult)
        self.assertIsInstance(leid, eg.LeidenResult)
        self.assertIsInstance(lp, eg.LabelPropagationResult)
        self.assertIsInstance(spec, eg.SpectralClusteringResult)
        self.assertIsInstance(km, eg.KMeansResult)
        self.assertEqual(spec.num_clusters, 2)
        self.assertGreaterEqual(spec.eigen_iterations, 0)
        self.assertEqual(km.k, 2)
        self.assertEqual(km.dims, 2)
        self.assertEqual(len(labels_true), 12)

        # TwinDeclutterer preserves finiteness on large-coordinate coincident and multi-cluster layouts
        l_large_coincident = eg.Layout2d.from_coords(
            np.full((4, 2), 1e308, dtype=np.float64)
        )
        td_large = eg.TwinDeclutterer(iterations=4, dtype="float64")
        td_large.apply(l_large_coincident)
        self.assertTrue(l_large_coincident.is_all_finite())

        l_large_multi = eg.Layout2d.from_coords(
            np.array(
                [[-1e308, -1e308], [-1e308, -1e308], [1e308, 1e308], [1e308, 1e308], [0.0, 0.0]],
                dtype=np.float64,
            )
        )
        td_multi = eg.TwinDeclutterer(iterations=4, dtype="float64")
        td_multi.apply(l_large_multi)
        self.assertTrue(l_large_multi.is_all_finite())


if __name__ == "__main__":
    unittest.main()
