"""Unit tests for adaptive CSR k-NN injection (`run_with_knn_adaptive`) on BhTsNet, FitTsNet, and LTsNet."""

import math
import unittest

import egraph as eg
import numpy as np


def make_graph(n):
    graph = eg.Graph()
    for i in range(n):
        graph.add_node(i)
    return graph


def make_two_cluster_knn(n_per_cluster=5):
    n = 2 * n_per_cluster
    coords = np.zeros((n, 2), dtype=np.float64)
    for i in range(n_per_cluster):
        angle = 2.0 * math.pi * i / n_per_cluster
        coords[i] = [math.cos(angle) * 0.2, math.sin(angle) * 0.2]
        coords[n_per_cluster + i] = [
            10.0 + math.cos(angle) * 0.2,
            math.sin(angle) * 0.2,
        ]
    diff = coords[:, None, :] - coords[None, :, :]
    dist_mat = np.sqrt(np.sum(diff * diff, axis=2))
    k = n - 1
    knn_indices = np.zeros((n, k), dtype=np.int64)
    knn_distances = np.zeros((n, k), dtype=np.float64)
    for i in range(n):
        order = [j for j in np.argsort(dist_mat[i]) if j != i][:k]
        knn_indices[i] = order
        knn_distances[i] = dist_mat[i, order]
    return n, knn_indices, knn_distances, coords


def build_algorithms(perplexity=3.0):
    return [
        (
            "BhTsNet",
            eg.BhTsNetBuilder()
            .perplexity(perplexity)
            .theta(0.5)
            .learning_rate(10.0)
            .iterations_stage1(20)
            .iterations_stage2(20)
            .iterations_stage3(20)
            .build(),
        ),
        (
            "FitTsNet",
            eg.FitTsNetBuilder()
            .intervals(10)
            .interpolation_points(3)
            .perplexity(perplexity)
            .theta(0.5)
            .learning_rate(10.0)
            .iterations_stage1(20)
            .iterations_stage2(20)
            .iterations_stage3(20)
            .build(),
        ),
        (
            "LTsNet",
            eg.LTsNetBuilder()
            .intervals(10)
            .interpolation_points(3)
            .perplexity(perplexity)
            .learning_rate(10.0)
            .iterations_stage1(20)
            .iterations_stage2(20)
            .iterations_stage3(20)
            .build(),
        ),
    ]


def dense_knn_to_csr(knn_indices, knn_distances):
    n, k = knn_indices.shape
    indptr = np.arange(0, (n + 1) * k, k, dtype=np.int64)
    indices = knn_indices.reshape(-1).astype(np.int64, copy=True)
    distances = knn_distances.reshape(-1).astype(np.float64, copy=True)
    return indptr, indices, distances


class TestTsNetKnnAdaptive(unittest.TestCase):
    def test_run_with_knn_adaptive_matches_run_with_knn_equivalence(self):
        n, knn_indices, knn_distances, _ = make_two_cluster_knn(n_per_cluster=5)
        graph = make_graph(n)
        indptr, indices, distances = dense_knn_to_csr(knn_indices, knn_distances)
        perp = 3.0
        perplexities = np.full(n, perp, dtype=np.float64)

        for name, algo in build_algorithms(perplexity=perp):
            with self.subTest(algorithm=name, fixture="two_cluster"):
                d_knn = eg.DrawingEuclidean2d.initial_placement(graph)
                d_adap = eg.DrawingEuclidean2d.initial_placement(graph)
                algo.run_with_knn(d_knn, knn_indices, knn_distances)
                algo.run_with_knn_adaptive(
                    d_adap, indptr, indices, distances, perplexities
                )
                pts_knn = np.array([[d_knn.x(u), d_knn.y(u)] for u in range(n)])
                pts_adap = np.array([[d_adap.x(u), d_adap.y(u)] for u in range(n)])
                np.testing.assert_allclose(pts_adap, pts_knn, atol=1e-12, rtol=0.0)

        # Cycle C_6 equivalence
        n6 = 6
        g6 = make_graph(n6)
        for i in range(n6):
            g6.add_edge(i, (i + 1) % n6, None)
        idx6 = np.array(
            [
                [(i + 1) % n6, (i - 1) % n6, (i + 2) % n6, (i - 2) % n6]
                for i in range(n6)
            ],
            dtype=np.int64,
        )
        dist6 = np.tile(np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float64), (n6, 1))
        indptr6, indices6, distances6 = dense_knn_to_csr(idx6, dist6)
        perps6 = np.full(n6, 2.5, dtype=np.float64)
        for name, algo in build_algorithms(perplexity=2.5):
            with self.subTest(algorithm=name, fixture="cycle6"):
                d_knn = eg.DrawingEuclidean2d.initial_placement(g6)
                d_adap = eg.DrawingEuclidean2d.initial_placement(g6)
                algo.run_with_knn(d_knn, idx6, dist6)
                algo.run_with_knn_adaptive(
                    d_adap, indptr6, indices6, distances6, perps6
                )
                pts_knn = np.array([[d_knn.x(u), d_knn.y(u)] for u in range(n6)])
                pts_adap = np.array([[d_adap.x(u), d_adap.y(u)] for u in range(n6)])
                np.testing.assert_allclose(pts_adap, pts_knn, atol=1e-12, rtol=0.0)

    def test_run_with_knn_adaptive_variable_rows_and_perplexities(self):
        n = 8
        graph = make_graph(n)
        # Rows with variable lengths (5, 3, 2, 1, 0, 2, 2, 2), self-loops, duplicates, zero dist
        rows = [
            [(0, 0.0), (1, 0.2), (1, 0.5), (2, 0.4), (3, 0.8), (4, 1.2), (5, 2.0)],
            [(0, 0.2), (2, 0.5), (3, 1.0)],
            [(0, 0.4), (1, 0.5)],
            [(0, 0.8)],
            [],
            [(0, 2.0), (6, 0.0)],
            [(5, 0.0), (7, 0.3)],
            [(6, 0.3), (0, 2.5)],
        ]
        indptr_list = [0]
        idx_list = []
        dist_list = []
        for row in rows:
            for j, d in row:
                idx_list.append(j)
                dist_list.append(d)
            indptr_list.append(len(idx_list))

        indptr = np.array(indptr_list, dtype=np.int64)
        indices = np.array(idx_list, dtype=np.int64)
        distances = np.array(dist_list, dtype=np.float64)
        perplexities = np.array(
            [0.5, 100.0, 1.5, 2.0, 3.0, 1.2, 5.0, 2.5], dtype=np.float64
        )

        for name, algo in build_algorithms(perplexity=2.0):
            with self.subTest(algorithm=name):
                drawing = eg.DrawingEuclidean2d.initial_placement(graph)
                algo.run_with_knn_adaptive(
                    drawing, indptr, indices, distances, perplexities
                )
                pts = np.array([[drawing.x(u), drawing.y(u)] for u in range(n)])
                self.assertTrue(np.all(np.isfinite(pts)))
                self.assertAlmostEqual(float(pts[:, 0].mean()), 0.0, places=5)
                self.assertAlmostEqual(float(pts[:, 1].mean()), 0.0, places=5)

    def test_run_with_knn_adaptive_index_dtypes_and_non_contiguous(self):
        n, knn_indices, knn_distances, _ = make_two_cluster_knn(n_per_cluster=4)
        graph = make_graph(n)
        indptr_base, indices_base, distances_base = dense_knn_to_csr(
            knn_indices, knn_distances
        )
        perps_base = np.full(n, 2.5, dtype=np.float64)

        dtypes = (np.int64, np.int32, np.uintp, np.intp, np.uint64, np.uint32)
        for dtype in dtypes:
            indptr_d = indptr_base.astype(dtype)
            indices_d = indices_base.astype(dtype)
            for name, algo in build_algorithms(perplexity=2.5):
                with self.subTest(algorithm=name, dtype=str(dtype)):
                    drawing = eg.DrawingEuclidean2d.initial_placement(graph)
                    algo.run_with_knn_adaptive(
                        drawing, indptr_d, indices_d, distances_base, perps_base
                    )
                    pts = np.array([[drawing.x(u), drawing.y(u)] for u in range(n)])
                    self.assertTrue(np.all(np.isfinite(pts)))

        # Mixed dtypes (indptr int64, indices int32) + non-contiguous strided slices (::2)
        indptr_strided = np.repeat(indptr_base, 2)[::2]
        indices_strided = np.repeat(indices_base.astype(np.int32), 2)[::2]
        distances_strided = np.repeat(distances_base, 2)[::2]
        perps_strided = np.repeat(perps_base, 2)[::2]
        for arr in (indptr_strided, indices_strided, distances_strided, perps_strided):
            self.assertFalse(arr.flags["C_CONTIGUOUS"])

        for name, algo in build_algorithms(perplexity=2.5):
            with self.subTest(algorithm=name, mode="mixed_strided"):
                d_ref = eg.DrawingEuclidean2d.initial_placement(graph)
                d_str = eg.DrawingEuclidean2d.initial_placement(graph)
                algo.run_with_knn_adaptive(
                    d_ref, indptr_base, indices_base, distances_base, perps_base
                )
                algo.run_with_knn_adaptive(
                    d_str,
                    indptr_strided,
                    indices_strided,
                    distances_strided,
                    perps_strided,
                )
                pts_ref = np.array([[d_ref.x(u), d_ref.y(u)] for u in range(n)])
                pts_str = np.array([[d_str.x(u), d_str.y(u)] for u in range(n)])
                np.testing.assert_allclose(pts_str, pts_ref, atol=1e-12, rtol=0.0)

    def test_run_with_knn_adaptive_edge_cases_n0_n1(self):
        g0 = make_graph(0)
        d0 = eg.DrawingEuclidean2d.initial_placement(g0)
        g1 = make_graph(1)
        d1 = eg.DrawingEuclidean2d.initial_placement(g1)

        empty_i64 = np.empty(0, dtype=np.int64)
        empty_f64 = np.empty(0, dtype=np.float64)

        for name, algo in build_algorithms(perplexity=2.0):
            with self.subTest(algorithm=name):
                algo.run_with_knn_adaptive(
                    d0, np.array([0], dtype=np.int64), empty_i64, empty_f64, empty_f64
                )
                with self.assertRaises(ValueError):
                    algo.run_with_knn_adaptive(
                        d0, empty_i64, empty_i64, empty_f64, empty_f64
                    )

                algo.run_with_knn_adaptive(
                    d1,
                    np.array([0, 0], dtype=np.int64),
                    empty_i64,
                    empty_f64,
                    np.array([2.0], dtype=np.float64),
                )
                with self.assertRaises(ValueError):
                    algo.run_with_knn_adaptive(
                        d1,
                        np.array([0, 0], dtype=np.int64),
                        empty_i64,
                        empty_f64,
                        np.array([0.0], dtype=np.float64),
                    )

    def test_run_with_knn_adaptive_validation_errors(self):
        n = 4
        graph = make_graph(n)
        indptr = np.array([0, 2, 4, 5, 6], dtype=np.int64)
        indices = np.array([1, 2, 0, 2, 1, 2], dtype=np.int64)
        distances = np.array([0.2, 0.5, 0.2, 0.4, 0.4, 0.8], dtype=np.float64)
        perps = np.array([1.5, 2.0, 1.5, 1.5], dtype=np.float64)

        for name, algo in build_algorithms(perplexity=2.0):
            with self.subTest(algorithm=name):
                d = eg.DrawingEuclidean2d.initial_placement(graph)

                # Wrong indptr lengths
                for bad_ptr in (indptr[:-1], np.append(indptr, 6)):
                    with self.assertRaises(ValueError):
                        algo.run_with_knn_adaptive(
                            d, bad_ptr, indices, distances, perps
                        )

                # indptr[0] != 0
                bad_ptr0 = indptr.copy()
                bad_ptr0[0] = 1
                with self.assertRaises(ValueError):
                    algo.run_with_knn_adaptive(d, bad_ptr0, indices, distances, perps)

                # Non-monotonic indptr
                bad_mono = np.array([0, 4, 2, 5, 6], dtype=np.int64)
                with self.assertRaises(ValueError):
                    algo.run_with_knn_adaptive(d, bad_mono, indices, distances, perps)

                # Negative indptr / negative indices
                bad_neg_ptr = np.array([0, -1, 4, 5, 6], dtype=np.int64)
                with self.assertRaises(ValueError):
                    algo.run_with_knn_adaptive(
                        d, bad_neg_ptr, indices, distances, perps
                    )
                bad_neg_idx = indices.copy()
                bad_neg_idx[1] = -1
                with self.assertRaises(ValueError):
                    algo.run_with_knn_adaptive(d, indptr, bad_neg_idx, distances, perps)

                # Length mismatches & huge indptr[n] without OOM/panic
                with self.assertRaises(ValueError):
                    algo.run_with_knn_adaptive(
                        d, indptr, indices[:-1], distances, perps
                    )
                with self.assertRaises(ValueError):
                    algo.run_with_knn_adaptive(
                        d, indptr, indices, distances[:-1], perps
                    )
                huge_ptr = np.array([0, 0, 0, 0, 10**12], dtype=np.int64)
                with self.assertRaises(ValueError):
                    algo.run_with_knn_adaptive(
                        d,
                        huge_ptr,
                        np.empty(0, dtype=np.int64),
                        np.empty(0, dtype=np.float64),
                        perps,
                    )

                # Out-of-range index >= n
                bad_oor = indices.copy()
                bad_oor[0] = n
                with self.assertRaises(ValueError):
                    algo.run_with_knn_adaptive(d, indptr, bad_oor, distances, perps)

                # Negative / NaN / Inf distances
                for bad_d in (-0.1, np.nan, np.inf, -np.inf):
                    bad_dists = distances.copy()
                    bad_dists[0] = bad_d
                    with self.assertRaises(ValueError):
                        algo.run_with_knn_adaptive(
                            d, indptr, indices, bad_dists, perps
                        )

                # Perplexities wrong length / <= 0 / NaN / Inf
                for bad_perps in (perps[:-1], np.append(perps, 1.0)):
                    with self.assertRaises(ValueError):
                        algo.run_with_knn_adaptive(
                            d, indptr, indices, distances, bad_perps
                        )
                for bad_p in (0.0, -0.0, -1.0, np.nan, np.inf, -np.inf):
                    bp = perps.copy()
                    bp[2] = bad_p
                    with self.assertRaises(ValueError):
                        algo.run_with_knn_adaptive(d, indptr, indices, distances, bp)

                # 2D arrays passed to 1D arguments
                with self.assertRaises(ValueError):
                    algo.run_with_knn_adaptive(
                        d, indptr[:, None], indices, distances, perps
                    )
                with self.assertRaises(ValueError):
                    algo.run_with_knn_adaptive(
                        d, indptr, indices[:, None], distances, perps
                    )
                with self.assertRaises(ValueError):
                    algo.run_with_knn_adaptive(
                        d, indptr, indices, distances[:, None], perps
                    )
                with self.assertRaises(ValueError):
                    algo.run_with_knn_adaptive(
                        d, indptr, indices, distances, perps[:, None]
                    )

                # Wrong dtypes (TypeError matching run_with_knn)
                with self.assertRaises(TypeError):
                    algo.run_with_knn_adaptive(
                        d, indptr.astype(np.float64), indices, distances, perps
                    )
                with self.assertRaises(TypeError):
                    algo.run_with_knn_adaptive(
                        d, indptr, indices.astype(np.float64), distances, perps
                    )
                with self.assertRaises(TypeError):
                    algo.run_with_knn_adaptive(
                        d, indptr, indices, distances.astype(np.float32), perps
                    )
                with self.assertRaises(TypeError):
                    algo.run_with_knn_adaptive(
                        d, indptr, indices, distances, perps.astype(np.float32)
                    )


if __name__ == "__main__":
    unittest.main()
