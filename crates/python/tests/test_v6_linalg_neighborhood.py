from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib
import inspect
import math
import threading
import unittest
from typing import Any, Callable

import egraph as eg
import numpy as np
import numpy.typing as npt

_sp_linalg: Any | None
try:
    _sp_linalg = importlib.import_module("scipy.linalg")
except ImportError:
    _sp_linalg = None


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


def _make_cycle(n: int) -> eg.CsrGraph:
    edges = [(i, (i + 1) % n) for i in range(n)]
    return eg.CsrGraph.from_edges(n, edges, directed=False, sanitize=False)


def _make_barbell(clique_size: int, bridge_len: int) -> eg.CsrGraph:
    n = 2 * clique_size + bridge_len
    edges: list[tuple[int, int]] = []
    for i in range(clique_size):
        for j in range(i + 1, clique_size):
            edges.append((i, j))
    offset = clique_size + bridge_len
    for i in range(clique_size):
        for j in range(i + 1, clique_size):
            edges.append((offset + i, offset + j))
    prev = clique_size - 1
    for b in range(bridge_len):
        cur = clique_size + b
        edges.append((prev, cur))
        prev = cur
    edges.append((prev, offset))
    return eg.CsrGraph.from_edges(n, edges, directed=False, sanitize=False)


def _dense_symmetric_normalized_laplacian(graph: eg.CsrGraph) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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


class TestV6LinalgAndNeighborhood(unittest.TestCase):
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
        self.assertEqual(out_indptr.dtype, np.dtype(np.uintp))
        self.assertEqual(out_indices.dtype, np.uint32)
        self.assertEqual(out_distances.dtype, np.float64)
        np.testing.assert_array_equal(out_indptr, np.array([0, 2, 4, 5], dtype=np.uintp))
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

    def test_pivots_pivot_mds_and_koren_init_f64_and_f32(self) -> None:
        # Core path 0..4 with leaves 5..8 attached to ends + isolated node 9
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (0, 6), (4, 7), (4, 8)]
        g_core = eg.CsrGraph.from_edges(10, edges, directed=False, sanitize=True)

        p1 = eg.select_maxmin_sp_pivots(g_core, 5, seed=42, min_degree=2)
        p2 = eg.select_maxmin_sp_pivots(g_core, 5, seed=42, min_degree=2)
        self.assertEqual(p1.dtype, np.uint32)
        np.testing.assert_array_equal(p1, p2)
        self.assertEqual(len(p1), 5)
        self.assertTrue(np.all(p1 <= 4))

        g_grid = _make_grid(5, 5)
        for dtype_str, np_dtype in (("float64", np.float64), ("float32", np.float32)):
            l_pmds = eg.pivot_mds_init(
                g_grid, num_pivots=16, seed=42, min_degree=2, target_std=1.0, dtype=dtype_str
            )
            self.assertEqual(len(l_pmds), 25)
            self.assertEqual(l_pmds.dtype, dtype_str)
            self.assertTrue(l_pmds.is_all_finite())
            coords = l_pmds.coords(copy=True)
            self.assertEqual(coords.dtype, np_dtype)
            self.assertAlmostEqual(float(np.std(coords)), 1.0, places=5)

            l_koren = eg.koren_spectral_init(
                g_grid, target_std=2.5, tol=1e-5, max_iters=300, seed=42, dtype=dtype_str
            )
            self.assertEqual(len(l_koren), 25)
            self.assertEqual(l_koren.dtype, dtype_str)
            self.assertTrue(l_koren.is_all_finite())
            k_coords = l_koren.coords(copy=True)
            self.assertAlmostEqual(float(np.std(k_coords)), 2.5, places=4)

        # target_std=None disables normalization
        l_unnorm = eg.pivot_mds_init(g_grid, num_pivots=12, target_std=None)
        self.assertTrue(l_unnorm.is_all_finite())
        self.assertNotAlmostEqual(float(np.std(l_unnorm.coords())), 1.0, places=2)

        l_koren_unnorm = eg.koren_spectral_init(g_grid, target_std=None)
        self.assertTrue(l_koren_unnorm.is_all_finite())

        # Empty, singleton, and disconnected graphs
        g0 = eg.CsrGraph.from_edges(0, [])
        g1 = eg.CsrGraph.from_edges(1, [])
        g_disc = eg.CsrGraph.from_edges(6, [(0, 1), (1, 2), (3, 4)])
        for g_edge, exp_n in ((g0, 0), (g1, 1), (g_disc, 6)):
            piv = eg.select_maxmin_sp_pivots(g_edge, 10)
            self.assertEqual(len(piv), min(exp_n, 10))
            lp = eg.pivot_mds_init(g_edge, num_pivots=4, dtype="float32")
            lk = eg.koren_spectral_init(g_edge, dtype="float64")
            self.assertEqual(len(lp), exp_n)
            self.assertEqual(len(lk), exp_n)
            self.assertTrue(lp.is_all_finite())
            self.assertTrue(lk.is_all_finite())

        # Error checks
        self.assert_no_panic_raises((ValueError,), lambda: eg.pivot_mds_init(g_grid, num_pivots=0))
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.pivot_mds_init(g_grid, target_std=0.0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.pivot_mds_init(g_grid, target_std=float("nan"))
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.pivot_mds_init(g_grid, dtype="int32")
        )
        self.assert_no_panic_raises((ValueError,), lambda: eg.koren_spectral_init(g_grid, tol=0.0))
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.koren_spectral_init(g_grid, max_iters=0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.koren_spectral_init(g0, tol=float("nan"))
        )

    def test_robust_normalize_layout_2d_outliers_and_edge_cases(self) -> None:
        rng = np.random.default_rng(123)
        core = rng.normal(size=(96, 2))
        outliers = np.full((4, 2), 500.0)
        pts = np.vstack([core, outliers])
        for dtype_str in ("float64", "float32"):
            lay = eg.Layout2d.from_coords(pts.astype(np.float32 if dtype_str == "float32" else np.float64))
            eg.robust_normalize_layout_2d(lay, target_std=1.0, quantile=0.95, seed=42)
            self.assertTrue(lay.is_all_finite())
            out_coords = lay.coords(copy=True)
            core_std = float(np.std(out_coords[:96]))
            self.assertAlmostEqual(core_std, 1.0, delta=0.35)

        # Validation errors
        lay_err = eg.Layout2d.init(5, seed=1)
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.robust_normalize_layout_2d(lay_err, target_std=0.0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.robust_normalize_layout_2d(lay_err, quantile=1.0)
        )

    def test_smallest_laplacian_eigenpairs_and_moment_stats(self) -> None:
        # Small graph -> dense Jacobi fallback
        g_small = _make_barbell(4, 2)
        l_sym_s, inv_s_ref = _dense_symmetric_normalized_laplacian(g_small)
        evals_ref, _ = np.linalg.eigh(l_sym_s)

        lambdas, vecs, inv_sqrt_deg, used_dense = eg.smallest_laplacian_eigenpairs(
            g_small, n_modes=4, tol=1e-8, max_iters=300, seed=42
        )
        self.assertTrue(used_dense)
        self.assertEqual(lambdas.shape, (4,))
        self.assertEqual(vecs.shape, (10, 4))
        self.assertEqual(inv_sqrt_deg.shape, (10,))
        np.testing.assert_allclose(inv_sqrt_deg, inv_s_ref, atol=1e-12)
        np.testing.assert_allclose(lambdas, evals_ref[1:5], atol=1e-10)
        np.testing.assert_allclose(vecs.T @ vecs, np.eye(4), atol=1e-10)
        np.testing.assert_allclose(l_sym_s @ vecs, vecs * lambdas[None, :], atol=1e-10)

        # Larger periodic 10x10 torus -> iterative block thick-restart Lanczos path (multiplicity 4!)
        g_torus = _make_grid(10, 10, periodic=True)
        l_sym_t, _ = _dense_symmetric_normalized_laplacian(g_torus)
        if _sp_linalg is not None:
            evals_t_ref = _sp_linalg.eigh(l_sym_t, eigvals_only=True)
        else:
            evals_t_ref = np.linalg.eigvalsh(l_sym_t)

        lam_t, vec_t, inv_t, used_dense_t = eg.smallest_laplacian_eigenpairs(
            g_torus, n_modes=6, tol=1e-6, max_iters=400, seed=42
        )
        self.assertFalse(used_dense_t)
        self.assertEqual(lam_t.shape, (6,))
        self.assertEqual(vec_t.shape, (100, 6))
        self.assertEqual(inv_t.shape, (100,))
        np.testing.assert_allclose(lam_t, evals_t_ref[1:7], atol=1e-5)
        np.testing.assert_allclose(vec_t.T @ vec_t, np.eye(6), atol=1e-6)
        res = np.max(np.abs(l_sym_t @ vec_t - vec_t * lam_t[None, :]))
        self.assertLess(float(res), 1e-4)

        # spectral_moment_stats exact Frobenius trace formula
        stats = eg.spectral_moment_stats(g_small)
        expected_sigma_sq = float(np.sum(l_sym_s**2) / g_small.num_nodes - 1.0)
        self.assertAlmostEqual(stats["sigma_lambda_sq"], expected_sigma_sq, places=12)
        self.assertAlmostEqual(
            stats["sigma_lambda"], math.sqrt(max(expected_sigma_sq, 1e-6)), places=12
        )
        self.assertGreater(stats["mean_degree"], 2.0)
        self.assertGreaterEqual(stats["deg_cv"], 0.0)
        self.assertEqual(stats["leaf_fraction"], 0.0)

        # Empty and singleton
        g0 = eg.CsrGraph.from_edges(0, [])
        l0, v0, d0, ud0 = eg.smallest_laplacian_eigenpairs(g0, 4)
        self.assertEqual(l0.shape, (0,))
        self.assertEqual(v0.shape, (0, 0))
        self.assertEqual(d0.shape, (0,))
        self.assertTrue(ud0)
        s0 = eg.spectral_moment_stats(g0)
        self.assertEqual(s0["sigma_lambda_sq"], 0.0)

        g1 = eg.CsrGraph.from_edges(1, [])
        l1, v1, d1, ud1 = eg.smallest_laplacian_eigenpairs(g1, 4)
        self.assertEqual(l1.shape, (0,))
        self.assertEqual(v1.shape, (1, 0))
        self.assertEqual(d1.shape, (1,))
        self.assertTrue(ud1)

        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.smallest_laplacian_eigenpairs(g_small, 0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.smallest_laplacian_eigenpairs(g_small, 2, tol=0.0)
        )

    def test_chebyshev_heat_spmv_and_dog_jl_embedding(self) -> None:
        g = _make_barbell(5, 2)
        n = g.num_nodes
        l_sym, inv_sqrt = _dense_symmetric_normalized_laplacian(g)

        rng = np.random.default_rng(7)
        v1d = rng.normal(size=(n,))
        v2d = np.asfortranarray(rng.normal(size=(n, 3)))

        for t in (0.0, 0.5, 4.0, 16.0):
            if _sp_linalg is not None:
                k_exact = _sp_linalg.expm(-t * l_sym)
            else:
                evals, evecs = np.linalg.eigh(l_sym)
                k_exact = (evecs * np.exp(-t * evals)[None, :]) @ evecs.T

            out1d = eg.chebyshev_heat_spmv(g, v1d, t)
            self.assertEqual(out1d.shape, (n,))
            self.assertEqual(out1d.dtype, np.float64)
            np.testing.assert_allclose(out1d, k_exact @ v1d, atol=1e-9)

            out2d = eg.chebyshev_heat_spmv(g, v2d, t)
            self.assertEqual(out2d.shape, (n, 3))
            self.assertEqual(out2d.dtype, np.float64)
            np.testing.assert_allclose(out2d, k_exact @ v2d, atol=1e-9)

        # (N, 0) empty column batch preserves shape
        out_empty_cols = eg.chebyshev_heat_spmv(g, np.empty((n, 0), dtype=np.float64), 1.0)
        self.assertEqual(out_empty_cols.shape, (n, 0))

        # Chebyshev DoG JL embedding: determinism and degree_normalize relation
        emb_norm = eg.chebyshev_dog_jl_embedding(
            g, degree=30, jl_dim=32, seed=42, degree_normalize=True, eps_psd=1e-4
        )
        emb_plain = eg.chebyshev_dog_jl_embedding(
            g, degree=30, jl_dim=32, seed=42, degree_normalize=False, eps_psd=1e-4
        )
        emb_repeat = eg.chebyshev_dog_jl_embedding(
            g, degree=30, jl_dim=32, seed=42, degree_normalize=True, eps_psd=1e-4
        )
        self.assertEqual(emb_norm.shape, (n, 32))
        self.assertEqual(emb_norm.dtype, np.float64)
        np.testing.assert_array_equal(emb_norm, emb_repeat)
        np.testing.assert_allclose(emb_norm, emb_plain * inv_sqrt[:, None], atol=1e-12)

        # Empty and singleton
        g0 = eg.CsrGraph.from_edges(0, [])
        self.assertEqual(eg.chebyshev_heat_spmv(g0, [], 1.0).shape, (0,))
        self.assertEqual(eg.chebyshev_dog_jl_embedding(g0, jl_dim=8).shape, (0, 8))

        g1 = eg.CsrGraph.from_edges(1, [])
        np.testing.assert_allclose(
            eg.chebyshev_heat_spmv(g1, [3.0], 1.0), [3.0 * math.exp(-1.0)], atol=1e-12
        )
        self.assertEqual(eg.chebyshev_dog_jl_embedding(g1, jl_dim=8).shape, (1, 8))

        # Errors
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.chebyshev_heat_spmv(g, np.ones(n - 1), 1.0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.chebyshev_heat_spmv(g, np.array([float("nan")] * n), 1.0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.chebyshev_heat_spmv(g, v1d, -0.5)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.chebyshev_heat_spmv(g, v1d, 1.0, lambda_max=0.0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.chebyshev_dog_jl_embedding(g, degree=0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.chebyshev_dog_jl_embedding(g, jl_dim=0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.chebyshev_dog_jl_embedding(g, t1=float("nan"))
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
        self.assertEqual(p_arr.indptr().dtype, np.dtype(np.uintp))
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

        # compute_horvitz_thompson_gamma exact R3 formula
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

    def test_gil_release_concurrent_execution(self) -> None:
        g = _make_grid(20, 20)
        ref_layout = eg.pivot_mds_init(g, num_pivots=24, seed=42).coords(copy=True)
        ref_lm = eg.compute_regularized_dog_varadhan_landmarks(
            g, degree=25, num_landmarks=16, seed=42
        ).table()
        ref_emb = eg.chebyshev_dog_jl_embedding(g, degree=25, jl_dim=16, seed=42)

        barrier = threading.Barrier(4)

        def worker(idx: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            barrier.wait(timeout=10.0)
            l = eg.pivot_mds_init(g, num_pivots=24, seed=42).coords(copy=True)
            t = eg.compute_regularized_dog_varadhan_landmarks(
                g, degree=25, num_landmarks=16, seed=42
            ).table()
            e = eg.chebyshev_dog_jl_embedding(g, degree=25, jl_dim=16, seed=42)
            _ = eg.smallest_laplacian_eigenpairs(g, n_modes=4, tol=1e-4, max_iters=100, seed=42 + idx)
            return l, t, e

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(worker, range(4)))

        for l_arr, t_arr, e_arr in results:
            np.testing.assert_array_equal(l_arr, ref_layout)
            np.testing.assert_array_equal(t_arr, ref_lm)
            np.testing.assert_array_equal(e_arr, ref_emb)

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

    def test_exports_signatures_and_docstrings(self) -> None:
        m2_classes = (
            "AmgPcgSolver",
            "CsrMatrix",
            "EffectiveResistanceResult",
            "ResistanceCurvatureFlowResult",
            "KnnGraph",
            "SparseProbabilities",
            "Sparse12HopPairs",
            "DogVaradhanLandmarks",
        )
        m2_functions = (
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
        for name in m2_classes + m2_functions:
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
