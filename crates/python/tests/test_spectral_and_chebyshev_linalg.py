from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib
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


class TestSpectralAndChebyshevLinalg(unittest.TestCase):
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
        fn: Callable[[], Any],
    ) -> None:
        with self.assertRaises(expected) as ctx:
            fn()
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

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
                g_grid, target_std=2.5, tol=1e-5, max_iterations=300, seed=42, dtype=dtype_str
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
            (ValueError,), lambda: eg.koren_spectral_init(g_grid, max_iterations=0)
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
            g_small, n_modes=4, tol=1e-8, max_iterations=300, seed=42
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
            g_torus, n_modes=6, tol=1e-6, max_iterations=400, seed=42
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

    def test_symmetric_eigen_jacobi_accuracy_and_boundaries(self) -> None:
        g_small = _make_barbell(4, 2)
        l_sym, _ = eg.symmetric_normalized_laplacian(g_small)
        dense = l_sym.to_dense()
        evals, evecs = eg.symmetric_eigen_jacobi(dense)
        ref_vals, _ = np.linalg.eigh(dense)
        self.assertEqual(evals.shape, (10,))
        self.assertEqual(evecs.shape, (10, 10))
        self.assertEqual(evals.dtype, np.float64)
        self.assertEqual(evecs.dtype, np.float64)
        np.testing.assert_allclose(evals, ref_vals, atol=1e-11)
        np.testing.assert_allclose(evecs.T @ evecs, np.eye(10), atol=1e-11)
        np.testing.assert_allclose(dense @ evecs, evecs * evals[None, :], atol=1e-11)

        # Empty and singleton
        ev0, vec0 = eg.symmetric_eigen_jacobi(np.empty((0, 0), dtype=np.float64))
        self.assertEqual(ev0.shape, (0,))
        self.assertEqual(vec0.shape, (0, 0))
        ev1, vec1 = eg.symmetric_eigen_jacobi([[3.5]])
        np.testing.assert_allclose(ev1, [3.5])
        np.testing.assert_allclose(vec1, [[1.0]])

        # Error checks without panics
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.symmetric_eigen_jacobi(np.zeros((2, 3), dtype=np.float64))
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.symmetric_eigen_jacobi(np.array([[1.0, float("nan")], [0.0, 1.0]]))
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.symmetric_eigen_jacobi([[1.0, 0.0], [0.0, 1.0]], tol=0.0)
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
            _ = eg.smallest_laplacian_eigenpairs(g, n_modes=4, tol=1e-4, max_iterations=100, seed=42 + idx)
            return l, t, e

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(worker, range(4)))

        for l_arr, t_arr, e_arr in results:
            np.testing.assert_array_equal(l_arr, ref_layout)
            np.testing.assert_array_equal(t_arr, ref_lm)
            np.testing.assert_array_equal(e_arr, ref_emb)


if __name__ == "__main__":
    unittest.main()
