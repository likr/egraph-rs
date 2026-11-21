"""Unit tests for external k-NN injection (`run_with_knn`) on BhTsNet, FitTsNet, and LTsNet."""

import math
import unittest

import egraph as eg
import numpy as np

try:
    from scipy.spatial import cKDTree

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


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


class TestTsNetKnnInjection(unittest.TestCase):
    def test_run_with_knn_int64_float64_all_variants(self):
        n, knn_indices, knn_distances, _ = make_two_cluster_knn(n_per_cluster=5)
        graph = make_graph(n)

        for name, algo in build_algorithms(perplexity=3.0):
            with self.subTest(algorithm=name):
                drawing = eg.DrawingEuclidean2d.initial_placement(graph)
                algo.run_with_knn(drawing, knn_indices, knn_distances)

                xs = np.array([drawing.x(u) for u in range(n)], dtype=np.float64)
                ys = np.array([drawing.y(u) for u in range(n)], dtype=np.float64)
                self.assertTrue(np.all(np.isfinite(xs)))
                self.assertTrue(np.all(np.isfinite(ys)))
                self.assertAlmostEqual(float(xs.mean()), 0.0, places=5)
                self.assertAlmostEqual(float(ys.mean()), 0.0, places=5)

    def test_run_with_knn_index_dtypes(self):
        n, knn_indices_i64, knn_distances, _ = make_two_cluster_knn(n_per_cluster=4)
        graph = make_graph(n)

        for dtype in (np.int64, np.int32, np.uintp, np.uint64, np.uint32):
            indices = knn_indices_i64.astype(dtype)
            for name, algo in build_algorithms(perplexity=2.5):
                with self.subTest(algorithm=name, dtype=str(dtype)):
                    drawing = eg.DrawingEuclidean2d.initial_placement(graph)
                    algo.run_with_knn(drawing, indices, knn_distances)
                    for u in range(n):
                        self.assertTrue(math.isfinite(drawing.x(u)))
                        self.assertTrue(math.isfinite(drawing.y(u)))

    def test_run_with_knn_non_contiguous_slices_and_self_loops(self):
        n, knn_indices, knn_distances, coords = make_two_cluster_knn(n_per_cluster=4)
        graph = make_graph(n)

        # Simulate cKDTree.query(..., k=k+1) where column 0 is self with distance 0.0
        self_col = np.arange(n, dtype=np.int64)[:, None]
        self_dist = np.zeros((n, 1), dtype=np.float64)
        full_idx = np.hstack([self_col, knn_indices])
        full_dist = np.hstack([self_dist, knn_distances])

        # Non-contiguous slice view [:, 1:]
        sliced_idx = full_idx[:, 1:]
        sliced_dist = full_dist[:, 1:]
        self.assertFalse(sliced_idx.flags["C_CONTIGUOUS"])
        self.assertFalse(sliced_dist.flags["C_CONTIGUOUS"])

        for name, algo in build_algorithms(perplexity=2.5):
            with self.subTest(algorithm=name, mode="sliced"):
                drawing = eg.DrawingEuclidean2d.initial_placement(graph)
                algo.run_with_knn(drawing, sliced_idx, sliced_dist)
                for u in range(n):
                    self.assertTrue(math.isfinite(drawing.x(u)))
                    self.assertTrue(math.isfinite(drawing.y(u)))

            with self.subTest(algorithm=name, mode="with_self_loop_and_duplicates"):
                dup_idx = np.hstack([full_idx, knn_indices[:, :1]])
                dup_dist = np.hstack([full_dist, knn_distances[:, :1] + 1.0])
                drawing = eg.DrawingEuclidean2d.initial_placement(graph)
                algo.run_with_knn(drawing, dup_idx, dup_dist)
                for u in range(n):
                    self.assertTrue(math.isfinite(drawing.x(u)))
                    self.assertTrue(math.isfinite(drawing.y(u)))

        if HAS_SCIPY:
            tree = cKDTree(coords)
            dists, idxs = tree.query(coords, k=5)
            knn_d = dists[:, 1:]
            knn_i = idxs[:, 1:]
            for name, algo in build_algorithms(perplexity=2.5):
                with self.subTest(algorithm=name, mode="ckdtree"):
                    drawing = eg.DrawingEuclidean2d.initial_placement(graph)
                    algo.run_with_knn(drawing, knn_i, knn_d)
                    for u in range(n):
                        self.assertTrue(math.isfinite(drawing.x(u)))
                        self.assertTrue(math.isfinite(drawing.y(u)))

    def test_run_with_knn_matches_cycle_run(self):
        n = 6
        graph = make_graph(n)
        for i in range(n):
            graph.add_edge(i, (i + 1) % n, None)

        knn_indices = np.array(
            [
                [(i + 1) % n, (i - 1) % n, (i + 2) % n, (i - 2) % n]
                for i in range(n)
            ],
            dtype=np.int64,
        )
        knn_distances = np.tile(np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float64), (n, 1))

        bh = (
            eg.BhTsNetBuilder()
            .perplexity(2.5)
            .k(4)
            .learning_rate(2.0)
            .iterations_stage1(10)
            .iterations_stage2(10)
            .iterations_stage3(10)
            .build()
        )
        d_run = eg.DrawingEuclidean2d.initial_placement(graph)
        d_knn = eg.DrawingEuclidean2d.initial_placement(graph)
        rng = eg.Rng.seed_from(42)
        bh.run(d_run, graph, rng)
        bh.run_with_knn(d_knn, knn_indices, knn_distances)

        for u in range(n):
            self.assertAlmostEqual(d_run.x(u), d_knn.x(u), places=6)
            self.assertAlmostEqual(d_run.y(u), d_knn.y(u), places=6)

    def test_run_with_knn_edge_cases_and_validation(self):
        n = 4
        graph = make_graph(n)
        valid_idx = np.array([[1, 2], [0, 2], [1, 3], [0, 2]], dtype=np.int64)
        valid_dist = np.ones((n, 2), dtype=np.float64)

        # Empty k (n, 0) and single-node graph should not error
        empty_idx = np.zeros((n, 0), dtype=np.int64)
        empty_dist = np.zeros((n, 0), dtype=np.float64)
        g1 = make_graph(1)
        d1 = eg.DrawingEuclidean2d.initial_placement(g1)

        for name, algo in build_algorithms(perplexity=2.0):
            with self.subTest(algorithm=name):
                d_empty = eg.DrawingEuclidean2d.initial_placement(graph)
                algo.run_with_knn(d_empty, empty_idx, empty_dist)
                algo.run_with_knn(
                    d1,
                    np.zeros((1, 0), dtype=np.int64),
                    np.zeros((1, 0), dtype=np.float64),
                )

                d = eg.DrawingEuclidean2d.initial_placement(graph)
                # Wrong row count
                with self.assertRaises(ValueError):
                    algo.run_with_knn(d, valid_idx[:3], valid_dist[:3])
                # Mismatched distance shape
                with self.assertRaises(ValueError):
                    algo.run_with_knn(d, valid_idx, np.ones((n, 3), dtype=np.float64))
                # 1D array
                with self.assertRaises(ValueError):
                    algo.run_with_knn(d, valid_idx.ravel(), valid_dist)
                with self.assertRaises(ValueError):
                    algo.run_with_knn(d, valid_idx, valid_dist.ravel())
                # Negative index
                bad_neg = valid_idx.copy()
                bad_neg[0, 0] = -1
                with self.assertRaises(ValueError):
                    algo.run_with_knn(d, bad_neg, valid_dist)
                # Out-of-range index >= n
                bad_oor = valid_idx.copy()
                bad_oor[0, 0] = n
                with self.assertRaises(ValueError):
                    algo.run_with_knn(d, bad_oor, valid_dist)
                # Negative / NaN / Inf distances
                bad_dist_neg = valid_dist.copy()
                bad_dist_neg[1, 0] = -0.1
                with self.assertRaises(ValueError):
                    algo.run_with_knn(d, valid_idx, bad_dist_neg)
                bad_dist_nan = valid_dist.copy()
                bad_dist_nan[0, 1] = np.nan
                with self.assertRaises(ValueError):
                    algo.run_with_knn(d, valid_idx, bad_dist_nan)
                bad_dist_inf = valid_dist.copy()
                bad_dist_inf[0, 1] = np.inf
                with self.assertRaises(ValueError):
                    algo.run_with_knn(d, valid_idx, bad_dist_inf)
                # Wrong dtype
                with self.assertRaises(TypeError):
                    algo.run_with_knn(d, valid_idx.astype(np.float64), valid_dist)
                with self.assertRaises(TypeError):
                    algo.run_with_knn(d, valid_idx, valid_dist.astype(np.float32))

    def test_default_tsnet_variants_grid_and_cycle_pivot_mds_init(self):
        # grid_5x5 and cycle_30 with both initial_placement and PivotMds initialization
        grid = eg.Graph()
        grid_nodes = [grid.add_node(i) for i in range(25)]
        grid_edges = []
        for r in range(5):
            for c in range(5):
                u = r * 5 + c
                if c + 1 < 5:
                    grid.add_edge(grid_nodes[u], grid_nodes[u + 1], 1.0)
                    grid_edges.append((u, u + 1))
                if r + 1 < 5:
                    grid.add_edge(grid_nodes[u], grid_nodes[u + 5], 1.0)
                    grid_edges.append((u, u + 5))

        cycle = eg.Graph()
        cycle_nodes = [cycle.add_node(i) for i in range(30)]
        cycle_edges = [(i, (i + 1) % 30) for i in range(30)]
        for u, v in cycle_edges:
            cycle.add_edge(cycle_nodes[u], cycle_nodes[v], 1.0)

        for gname, graph, n, edges, perp in [
            ("grid_5x5", grid, 25, grid_edges, 8.0),
            ("cycle_30", cycle, 30, cycle_edges, 10.0),
        ]:
            for init_mode in ("initial_placement", "pivot_mds"):
                for cls_name, cls in [
                    ("BhTsNet", eg.BhTsNet),
                    ("FitTsNet", eg.FitTsNet),
                    ("LTsNet", eg.LTsNet),
                ]:
                    with self.subTest(graph=gname, init=init_mode, algo=cls_name):
                        if init_mode == "initial_placement":
                            drawing = eg.DrawingEuclidean2d.initial_placement(graph)
                        else:
                            drawing = eg.PivotMds(
                                graph, lambda _: 1.0, list(range(n))
                            ).run_2d()
                        algo = cls.builder().perplexity(perp).build()
                        algo.run(drawing, graph, eg.Rng.seed_from(42))
                        pts = np.array(
                            [[drawing.x(i), drawing.y(i)] for i in range(n)],
                            dtype=np.float64,
                        )
                        self.assertTrue(np.all(np.isfinite(pts)))
                        max_abs = float(np.max(np.abs(pts)))
                        mean_edge = float(
                            np.mean(
                                [np.linalg.norm(pts[u] - pts[v]) for u, v in edges]
                            )
                        )
                        diff = pts[:, None, :] - pts[None, :, :]
                        mean_all = float(
                            np.sum(np.linalg.norm(diff, axis=-1)) / (n * (n - 1))
                        )
                        self.assertLess(max_abs, 100.0)
                        self.assertLess(mean_edge, 0.6 * mean_all)


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


class TestTsNetAdversarial(unittest.TestCase):
    def test_builders_reject_nan_and_inf(self):
        for builder_cls in (
            eg.TsNetBuilder,
            eg.BhTsNetBuilder,
            eg.FitTsNetBuilder,
            eg.LTsNetBuilder,
        ):
            with self.assertRaises(ValueError):
                b = builder_cls()
                b.perplexity(float("nan"))
                b.build()
            with self.assertRaises(ValueError):
                b = builder_cls()
                b.learning_rate(float("inf"))
                b.build()
            with self.assertRaises(ValueError):
                b = builder_cls()
                b.momentum(float("nan"))
                b.build()

    def test_tsnet_nonfinite_drawing_and_knn_validation(self):
        rng = eg.Rng.seed_from(42)
        g = make_graph(4)
        for i in range(3):
            g.add_edge(i, i + 1, i)
        g_other = make_graph(2)
        d_other = eg.DrawingEuclidean2d.initial_placement(g_other)

        for builder_cls in (eg.BhTsNetBuilder, eg.FitTsNetBuilder, eg.LTsNetBuilder):
            algo = (
                builder_cls()
                .perplexity(2.0)
                .iterations_stage1(2)
                .iterations_stage2(2)
                .iterations_stage3(2)
                .build()
            )
            d_nan = eg.DrawingEuclidean2d.initial_placement(g)
            d_nan.set_x(0, float("nan"))
            with self.assertRaises(ValueError):
                algo.run(d_nan, g, rng)
            with self.assertRaises(ValueError):
                algo.run(d_other, g, rng)

            indptr = np.array([0, 1, 2, 3, 4], dtype=np.int64)
            bad_indices = np.array([99, 0, 1, 2], dtype=np.int64)
            good_indices = np.array([1, 0, 1, 2], dtype=np.int64)
            dists = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
            bad_dists = np.array([-0.5, 1.0, 1.0, 1.0], dtype=np.float64)
            perps = np.array([1.5, 1.5, 1.5, 1.5], dtype=np.float64)

            d_clean = eg.DrawingEuclidean2d.initial_placement(g)
            with self.assertRaises(ValueError):
                algo.run_with_knn_adaptive(d_clean, indptr, bad_indices, dists, perps)
            with self.assertRaises(ValueError):
                algo.run_with_knn_adaptive(d_clean, indptr, good_indices, bad_dists, perps)
            with self.assertRaises(ValueError):
                algo.run_with_knn_adaptive(d_nan, indptr, good_indices, dists, perps)

        ts_net = eg.TsNetBuilder().perplexity(2.0).n_epochs(5).build()
        dist = eg.all_sources_bfs(g, 1.0)
        d_nan = eg.DrawingEuclidean2d.initial_placement(g)
        d_nan.set_y(1, float("inf"))
        with self.assertRaises(ValueError):
            ts_net.run(d_nan, dist)
        with self.assertRaises(ValueError):
            ts_net.run(d_other, dist)


if __name__ == "__main__":
    unittest.main()
