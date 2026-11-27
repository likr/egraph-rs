from __future__ import annotations

import unittest
from typing import Any, Callable

import egraph as eg
import numpy as np


def _make_cycle(n: int) -> eg.CsrGraph:
    edges = [(i, (i + 1) % n) for i in range(n)]
    return eg.CsrGraph.from_edges(n, edges, directed=False, sanitize=False)


class TestNeighborhoodKnnAndLandmarks(unittest.TestCase):
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
        fn: Callable[[], Any],
    ) -> None:
        with self.assertRaises(expected) as ctx:
            fn()
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

    def test_knn_graph_class_and_validation(self) -> None:
        indptr = np.array([0, 2, 4, 5], dtype=np.int64)
        indices = np.array([1, 2, 0, 2, 1], dtype=np.int32)
        distances = np.array([0.5, 1.5, 0.5, 2.0, 2.0], dtype=np.float32)

        knn = eg.KnnGraph.from_csr(3, indptr, indices, distances)
        self.assertEqual(knn.num_nodes, 3)
        self.assertEqual(len(knn), 3)
        self.assertEqual(knn.nnz, 5)
        self.assertIn("KnnGraph(", repr(knn))

        out_indptr = knn.indptr()
        out_indices = knn.indices()
        out_distances = knn.distances()
        self.assertEqual(out_indptr.dtype, np.dtype(np.uint32))
        self.assertEqual(out_indices.dtype, np.uint32)
        self.assertEqual(out_distances.dtype, np.float64)
        np.testing.assert_array_equal(out_indptr, np.array([0, 2, 4, 5], dtype=np.uint32))
        np.testing.assert_array_equal(out_indices, np.array([1, 2, 0, 2, 1], dtype=np.uint32))
        np.testing.assert_allclose(out_distances, [0.5, 1.5, 0.5, 2.0, 2.0])

        # Empty and singleton KnnGraph
        knn0 = eg.KnnGraph.from_csr(0, [0], [], [])
        self.assertEqual(knn0.num_nodes, 0)
        self.assertEqual(knn0.nnz, 0)
        knn1 = eg.KnnGraph.from_csr(1, [0, 0], [], [])
        self.assertEqual(knn1.num_nodes, 1)
        self.assertEqual(knn1.nnz, 0)

        # Validation errors
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.KnnGraph.from_csr(2, [0, 1], [1], [1.0])
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.KnnGraph.from_csr(2, [1, 1, 1], [], [])
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.KnnGraph.from_csr(2, [0, 2, 1], [1], [1.0])
        )
        self.assert_no_panic_raises(
            (IndexError,), lambda: eg.KnnGraph.from_csr(2, [0, 1, 1], [5], [1.0])
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.KnnGraph.from_csr(2, [0, 1, 1], [1], [-0.1])
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.KnnGraph.from_csr(2, [0, 1, 1], [1], [float("nan")])
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.KnnGraph.from_csr(2, [0, 1, 1], [1], [float("inf")])
        )
        self.assert_no_panic_raises(
            (TypeError,), lambda: eg.KnnGraph.from_csr(True, [0, 1], [0], [1.0])
        )
        self.assert_no_panic_raises(
            (TypeError,), lambda: eg.KnnGraph.from_csr(2, [0, 1, 1], [1], [True])
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.KnnGraph.from_csr(2, np.zeros((1, 3), dtype=np.int64), [1], [1.0]),
        )

    def test_1hop_2hop_perplexities_and_knn_builders(self) -> None:
        # Star hub 0 connected to 1..=8, chain 8-9-10-11, isolated 12 (N=13)
        edges = [(0, i) for i in range(1, 9)] + [(8, 9), (9, 10), (10, 11)]
        g = eg.CsrGraph.from_edges(13, edges, directed=False, sanitize=True)

        hop1, hop2 = eg.extract_1hop_2hop_csr(g)
        self.assertEqual(hop1.num_nodes, 13)
        self.assertEqual(hop2.num_nodes, 13)
        # Hub 0 has 1-hop {1..8} and pure 2-hop {9}
        np.testing.assert_array_equal(
            hop1.indices()[hop1.indptr()[0] : hop1.indptr()[1]], np.arange(1, 9, dtype=np.uint32)
        )
        np.testing.assert_array_equal(
            hop2.indices()[hop2.indptr()[0] : hop2.indptr()[1]], np.array([9], dtype=np.uint32)
        )

        knn2, k2_deg, stats2 = eg.build_pure_2hop_knn(g)
        sizes = eg.compute_2hop_neighborhood_sizes(g)
        self.assertEqual(k2_deg.dtype, np.uint32)
        self.assertEqual(sizes.dtype, np.uint32)
        np.testing.assert_array_equal(k2_deg, sizes)
        self.assertEqual(int(k2_deg[0]), 9)
        self.assertEqual(int(k2_deg[12]), 0)
        self.assertEqual(stats2["nnz"], knn2.nnz)
        self.assertEqual(stats2["n_1hop"], int(hop1.indptr()[-1]))
        self.assertEqual(stats2["n_2hop"], int(hop2.indptr()[-1]))

        # Sparse 1+2-hop pairs extraction
        pairs = eg.extract_sparse_1hop_2hop_pairs(g, max_2hop_per_node=32, seed=42)
        pairs_rep = eg.extract_sparse_1hop_2hop_pairs(g, max_2hop_per_node=32, seed=42)
        self.assertEqual(pairs.num_1hop, len(edges))
        self.assertGreater(pairs.num_2hop, 0)
        self.assertIn("Sparse12HopPairs", repr(pairs))
        for arr in (pairs.u1(), pairs.v1(), pairs.u2(), pairs.v2()):
            self.assertEqual(arr.dtype, np.uint32)
        self.assertTrue(np.all(pairs.u1() < pairs.v1()))
        self.assertTrue(np.all(pairs.u2() < pairs.v2()))
        np.testing.assert_array_equal(pairs.u1(), pairs_rep.u1())
        np.testing.assert_array_equal(pairs.v1(), pairs_rep.v1())
        np.testing.assert_array_equal(pairs.u2(), pairs_rep.u2())
        np.testing.assert_array_equal(pairs.v2(), pairs_rep.v2())

        pairs_1hop_only = eg.extract_sparse_1hop_2hop_pairs(g, max_2hop_per_node=0, seed=42)
        self.assertEqual(pairs_1hop_only.num_1hop, len(edges))
        self.assertEqual(pairs_1hop_only.num_2hop, 0)

        empty_pairs = eg.extract_sparse_1hop_2hop_pairs(eg.CsrGraph.from_edges(0, []))
        self.assertEqual(empty_pairs.num_1hop, 0)
        self.assertEqual(empty_pairs.num_2hop, 0)
        single_pairs = eg.extract_sparse_1hop_2hop_pairs(eg.CsrGraph.from_edges(1, []))
        self.assertEqual(single_pairs.num_1hop, 0)
        self.assertEqual(single_pairs.num_2hop, 0)

        # Row 0 in pure 2-hop knn has 8 distances == 1.0 and 1 distance == 2.0
        r0_idx = knn2.indices()[knn2.indptr()[0] : knn2.indptr()[1]]
        r0_dist = knn2.distances()[knn2.indptr()[0] : knn2.indptr()[1]]
        self.assertEqual(len(r0_idx), 9)
        self.assertEqual(int(np.sum(r0_dist == 1.0)), 8)
        self.assertEqual(int(np.sum(r0_dist == 2.0)), 1)

        # Capped 1+2-hop k-NN constructor & 1+2-hop degree-decayed row-normalized probabilities
        g_star = eg.CsrGraph.from_edges(50, [(0, i) for i in range(1, 50)], directed=False, sanitize=True)
        cknn, ck2, cstats = eg.build_capped_12hop_knn(g_star, max_2hop_per_node=4, seed=42)
        self.assertEqual(len(ck2), 50)
        self.assertEqual(cstats["n_1hop"], 98)
        self.assertLessEqual(cstats["n_2hop"], 2 * 50 * 4)
        self.assertGreater(cstats["n_truncated_2hop"], 0)
        for u in range(50):
            row_cols = cknn.indices()[cknn.indptr()[u] : cknn.indptr()[u + 1]]
            self.assertTrue(np.all(row_cols[:-1] < row_cols[1:]))

        p12_uncap = eg.graph_12hop_row_normalized_probabilities(g, w2=0.5, max_2hop_per_node=None, degree_beta=0.35, seed=42)
        p12_cap = eg.graph_12hop_row_normalized_probabilities(g, w2=0.5, max_2hop_per_node=4, degree_beta=0.35, seed=42)
        # g has 12 non-isolated nodes and 1 isolated node (node 12), so sum(P) == 12/13.
        self.assertAlmostEqual(float(np.sum(p12_uncap.values())), 12.0 / 13.0, places=12)
        self.assertAlmostEqual(float(np.sum(p12_cap.values())), 12.0 / 13.0, places=12)
        p12_star = eg.graph_12hop_row_normalized_probabilities(g_star, w2=0.5, max_2hop_per_node=4, degree_beta=0.35, seed=42)
        self.assertAlmostEqual(float(np.sum(p12_star.values())), 1.0, places=12)
        p12_extreme = eg.graph_12hop_row_normalized_probabilities(g, w2=0.5, max_2hop_per_node=4, degree_beta=200.0, seed=42)
        self.assertTrue(np.all(np.isfinite(p12_extreme.values())))
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.graph_12hop_row_normalized_probabilities(g, degree_beta=-0.1),
        )

        # Adaptive perplexities
        node_perps = eg.compute_nodewise_2hop_perplexities(
            g, c=0.9, min_perplexity=5.0, max_perplexity=1200.0
        )
        med_perp = eg.compute_median_2hop_perplexity(
            g, c=1.5, min_perplexity=15.0, max_perplexity=1200.0
        )
        upper = 12.0 / 3.0  # (N-1)/3 = 4.0
        self.assertEqual(node_perps.dtype, np.float64)
        np.testing.assert_allclose(node_perps, np.full(13, upper))
        self.assertAlmostEqual(med_perp, upper, places=12)

        # Hybrid adaptive k-NN using spectral JL embedding
        coords = eg.chebyshev_dog_jl_embedding(g, jl_dim=14, seed=42)
        perps = np.full(13, 3.0, dtype=np.float64)
        perps[0] = 1.0  # target k=3 < hub's 9 1+2-hop neighbors -> must NOT truncate!
        h_knn, h_stats = eg.build_hybrid_adaptive_knn(
            g, coords, perps, k_factor=3.0, margin_gap=0.15, coarse_dim=10, rerank_factor=4
        )
        self.assertEqual(h_stats["n_truncated_1hop"], 0)
        self.assertEqual(h_stats["n_truncated_2hop"], 0)
        h_indptr = h_knn.indptr()
        h_dist = h_knn.distances()
        self.assertEqual(int(h_indptr[1] - h_indptr[0]), 9)
        # Isolated node 12 gets round(3.0 * 3.0) = 9 residual neighbors
        self.assertEqual(int(h_indptr[13] - h_indptr[12]), 9)
        # Verify strict margin_gap separation on rows with residual fill
        for i in range(13):
            deg12 = int(k2_deg[i])
            row_d = h_dist[h_indptr[i] : h_indptr[i + 1]]
            if deg12 > 0 and len(row_d) > deg12:
                self.assertLess(float(np.max(row_d[:deg12])), float(np.min(row_d[deg12:])))

        # Scalar broadcast perplexity in build_hybrid_adaptive_knn and k override / max_row_k
        h_knn_ov, _ = eg.build_hybrid_adaptive_knn(g, coords, 3.0, k=4, max_row_k=6)
        for i in range(13):
            row_len = int(h_knn_ov.indptr()[i + 1] - h_knn_ov.indptr()[i])
            self.assertLessEqual(row_len, 6)

        # Layout2d and LayoutNd inputs accepted directly by build_hybrid_adaptive_knn and build_rerank_knn
        coords_2d = eg.Layout2d.from_coords(coords[:, :2])
        coords_nd = eg.LayoutNd.from_coords(coords)
        h_knn_2d, _ = eg.build_hybrid_adaptive_knn(g, coords_2d, 3.0, k=4, max_row_k=6)
        h_knn_nd, _ = eg.build_hybrid_adaptive_knn(g, coords_nd, perps, k=4, max_row_k=6)
        self.assertEqual(h_knn_2d.num_nodes, 13)
        self.assertEqual(h_knn_nd.num_nodes, 13)
        np.testing.assert_array_equal(h_knn_nd.indices(), h_knn_ov.indices())
        r_knn_2d = eg.build_rerank_knn(coords_2d, k=4)
        r_knn_nd = eg.build_rerank_knn(coords_nd, k=4)
        r_knn_arr = eg.build_rerank_knn(coords, k=4)
        self.assertEqual(r_knn_2d.num_nodes, 13)
        self.assertEqual(r_knn_nd.num_nodes, 13)
        np.testing.assert_array_equal(r_knn_nd.indices(), r_knn_arr.indices())

        # Partial BFS k-NN
        p_knn1 = eg.partial_bfs_knn(g, k=5, seed=42)
        p_knn2 = eg.partial_bfs_knn(g, k=5, seed=42)
        np.testing.assert_array_equal(p_knn1.indices(), p_knn2.indices())
        np.testing.assert_array_equal(p_knn1.distances(), p_knn2.distances())
        self.assertEqual(int(p_knn1.indptr()[1] - p_knn1.indptr()[0]), 5)
        self.assertEqual(int(p_knn1.indptr()[13] - p_knn1.indptr()[12]), 0)

        # Error checks
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.compute_nodewise_2hop_perplexities(g, c=0.0)
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.compute_median_2hop_perplexity(g, min_perplexity=20.0, max_perplexity=10.0),
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.build_hybrid_adaptive_knn(g, coords[:-1], perps)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.build_hybrid_adaptive_knn(g, coords, perps[:-1])
        )
        bad_coords = coords.copy()
        bad_coords[0, 0] = float("nan")
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.build_hybrid_adaptive_knn(g, bad_coords, perps)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.build_hybrid_adaptive_knn(g, coords, perps, max_row_k=0)
        )

    def test_probabilities_fuzzy_ab_gamma_and_dog_varadhan_landmarks(self) -> None:
        g = _make_cycle(16)
        knn = eg.partial_bfs_knn(g, k=6, seed=42)

        # SparseProbabilities from array vs scalar broadcast perplexity
        p_arr = eg.compute_knn_probabilities(
            knn, np.full(16, 3.0, dtype=np.float64), power=2.0, sigma_iters=50, sigma_tolerance=1e-6
        )
        p_scalar = eg.compute_knn_probabilities(
            knn, 3.0, power=2.0, sigma_iters=50, sigma_tolerance=1e-6
        )
        self.assertEqual(p_arr.num_nodes, 16)
        self.assertEqual(len(p_arr), 16)
        self.assertIn("SparseProbabilities(", repr(p_arr))
        self.assertEqual(p_arr.indptr().dtype, np.dtype(np.uint32))
        self.assertEqual(p_arr.indices().dtype, np.uint32)
        self.assertEqual(p_arr.values().dtype, np.float64)
        np.testing.assert_array_equal(p_arr.indptr(), p_scalar.indptr())
        np.testing.assert_array_equal(p_arr.indices(), p_scalar.indices())
        np.testing.assert_array_equal(p_arr.values(), p_scalar.values())
        self.assertAlmostEqual(float(np.sum(p_arr.values())), 1.0, places=12)

        p_rn = eg.compute_knn_probabilities(knn, 3.0, row_normalize=True)
        self.assertAlmostEqual(float(np.sum(p_rn.values())), 1.0, places=12)

        # Fuzzy simplicial set
        fuzzy = eg.compute_fuzzy_simplicial_set(knn, sigma_iters=64, sigma_tolerance=1e-5)
        self.assertEqual(fuzzy.num_nodes, 16)
        self.assertGreater(fuzzy.nnz, 0)
        vals = fuzzy.values()
        self.assertTrue(np.all((vals > 1e-7) & (vals <= 1.0)))

        # Verify symmetry of SparseProbabilities
        f_indptr = fuzzy.indptr()
        f_idx = fuzzy.indices()
        dense_p = np.zeros((16, 16), dtype=np.float64)
        for i in range(16):
            for pos in range(int(f_indptr[i]), int(f_indptr[i + 1])):
                dense_p[i, int(f_idx[pos])] = vals[pos]
        np.testing.assert_array_equal(dense_p, dense_p.T)

        # find_ab_params against UMAP canonical (0.1, 1.0)
        a, b = eg.find_ab_params(0.1, 1.0)
        self.assertAlmostEqual(a, 1.577, delta=0.015)
        self.assertAlmostEqual(b, 0.895, delta=0.015)

        # compute_horvitz_thompson_gamma exact formula
        self.assertAlmostEqual(eg.compute_horvitz_thompson_gamma(1.0, 1600, 16), 10.0, places=12)
        self.assertEqual(eg.compute_horvitz_thompson_gamma(1.0, 0, 16), 0.0)
        self.assertAlmostEqual(eg.compute_horvitz_thompson_gamma(0.5, 9, 0), 1.5, places=12)

        # compute_regularized_dog_varadhan_landmarks
        lm = eg.compute_regularized_dog_varadhan_landmarks(
            g, degree=25, num_landmarks=8, seed=42, degree_normalize=True, p_exp=0.5, eps_psd=1e-4, t_coarse=48.0
        )
        self.assertEqual(lm.num_nodes, 16)
        self.assertEqual(len(lm), 16)
        self.assertEqual(lm.num_landmarks, 8)
        self.assertIn("DogVaradhanLandmarks(", repr(lm))
        self.assertTrue(lm.used_coarse_scaffold)
        self.assertGreater(lm.t1, 0.0)
        self.assertGreater(lm.t2, lm.t1)
        self.assertGreaterEqual(lm.alpha, 0.0)
        self.assertGreaterEqual(lm.delta_psd, 0.0)
        self.assertGreaterEqual(lm.beta_eff, 1e-4)
        self.assertLessEqual(lm.beta_eff, 0.5)
        self.assertAlmostEqual(lm.beta_eff, lm.beta_auto, places=12)

        lm_ids = lm.landmarks()
        tbl = lm.table()
        ctbl = lm.coarse_table()
        self.assertEqual(lm_ids.dtype, np.uint32)
        self.assertEqual(lm_ids.shape, (8,))
        self.assertEqual(tbl.shape, (16, 8))
        self.assertTrue(np.all(np.isfinite(tbl) & (tbl >= 0.0)))
        assert ctbl is not None
        self.assertEqual(ctbl.shape, (16, 8))
        self.assertTrue(np.all(np.isfinite(ctbl) & (ctbl >= 0.0)))

        alt = lm.alt_distances([0, 0, 4], [0, 8, 12])
        self.assertEqual(alt.dtype, np.float64)
        self.assertEqual(alt[0], 0.0)
        self.assertGreater(alt[1], 0.0)
        self.assertGreater(alt[2], 0.0)

        # t_coarse=0.0 disables coarse scaffold -> coarse_table() is None
        lm_no_coarse = eg.compute_regularized_dog_varadhan_landmarks(
            g, num_landmarks=8, t_coarse=0.0, beta_floor=0.05
        )
        self.assertFalse(lm_no_coarse.used_coarse_scaffold)
        self.assertIsNone(lm_no_coarse.coarse_table())
        self.assertAlmostEqual(lm_no_coarse.beta_eff, 0.05, places=12)

        # Empty and singleton landmarks
        g0 = eg.CsrGraph.from_edges(0, [])
        lm0 = eg.compute_regularized_dog_varadhan_landmarks(g0)
        self.assertEqual(lm0.num_nodes, 0)
        self.assertEqual(lm0.table().shape, (0, 0))
        self.assertEqual(len(lm0.alt_distances([], [])), 0)

        g1 = eg.CsrGraph.from_edges(1, [])
        lm1 = eg.compute_regularized_dog_varadhan_landmarks(g1)
        self.assertEqual(lm1.num_nodes, 1)
        self.assertEqual(lm1.num_landmarks, 1)
        self.assertEqual(lm1.table().shape, (1, 1))
        np.testing.assert_allclose(lm1.alt_distances([0], [0]), [0.0])

        # 0-D NumPy scalar perplexities across compute_knn_probabilities, build_hybrid_adaptive_knn, TsNetSolver.from_knn
        p_0d_ref = eg.compute_knn_probabilities(knn, 3.0)
        p_0d_f = eg.compute_knn_probabilities(knn, np.array(3.0, dtype=np.float64))
        p_0d_i = eg.compute_knn_probabilities(knn, np.array(3, dtype=np.int32))
        np.testing.assert_array_equal(p_0d_f.values(), p_0d_ref.values())
        np.testing.assert_array_equal(p_0d_i.values(), p_0d_ref.values())

        g13 = eg.CsrGraph.from_edges(13, [(0, i) for i in range(1, 9)] + [(8, 9), (9, 10), (10, 11)])
        coords13 = eg.chebyshev_dog_jl_embedding(g13, jl_dim=14, seed=42)
        h_0d, _ = eg.build_hybrid_adaptive_knn(g13, coords13, np.array(3.0), k=4, max_row_k=6)
        self.assertEqual(h_0d.num_nodes, 13)
        ts_0d = eg.TsNetSolver.from_knn(
            knn, np.array(3.0), backend="bh", stage_iterations=(1, 1, 1)
        )
        self.assertEqual(ts_0d.num_nodes, 16)

        for bad_0d_val in (float("nan"), -1.0, 0.0):
            def _call_bad_0d(v: float = bad_0d_val) -> object:
                return eg.compute_knn_probabilities(knn, np.array(v))

            self.assert_no_panic_raises((ValueError,), _call_bad_0d)
        self.assert_no_panic_raises(
            (TypeError,), lambda: eg.compute_knn_probabilities(knn, np.array(True))
        )
        self.assert_no_panic_raises(
            (TypeError,), lambda: eg.compute_knn_probabilities(knn, np.bool_(True))  # type: ignore[arg-type]
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.compute_knn_probabilities(knn, np.array([[3.0]]))
        )

        # Error checks
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.compute_knn_probabilities(knn, [1.0, 2.0])
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.compute_knn_probabilities(knn, float("nan"))
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.compute_knn_probabilities(knn, -1.0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.compute_fuzzy_simplicial_set(knn, sigma_iters=0)
        )
        self.assert_no_panic_raises((ValueError,), lambda: eg.find_ab_params(-0.1, 1.0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.find_ab_params(0.1, 0.0))
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.compute_horvitz_thompson_gamma(-1.0, 10, 4)
        )
        self.assert_no_panic_raises((IndexError,), lambda: lm.alt_distances([99], [0]))
        self.assert_no_panic_raises((ValueError,), lambda: lm.alt_distances([0, 1], [0]))
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.compute_regularized_dog_varadhan_landmarks(g, beta_floor=1.0),
        )


if __name__ == "__main__":
    unittest.main()
