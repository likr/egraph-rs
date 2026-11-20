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


if __name__ == "__main__":
    unittest.main()
