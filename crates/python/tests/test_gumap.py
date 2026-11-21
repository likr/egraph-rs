import math
import unittest
import networkx as nx
import numpy as np
import egraph as eg

try:
    from scipy.spatial import cKDTree

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def draw(nx_graph):
    graph = eg.Graph()
    indices = {}
    for u in nx_graph.nodes:
        indices[u] = graph.add_node(u)
    for u, v in nx_graph.edges:
        graph.add_edge(indices[u], indices[v], (u, v))
    return graph


class TestGuMap(unittest.TestCase):
    def test_gumap_builder_defaults(self):
        builder = eg.GuMapBuilder()
        gumap = builder.build()
        self.assertIsNotNone(gumap)

    def test_gumap_run(self):
        nx_graph = nx.cycle_graph(6)
        graph = draw(nx_graph)

        drawing = eg.DrawingEuclidean2d.initial_placement(graph)
        distance_matrix = eg.all_sources_dijkstra(graph, lambda _: 1.0)
        rng = eg.Rng.seed_from(42)

        # Initialize GUMAP with builder
        gumap = (
            eg.GuMapBuilder()
            .k(3)
            .min_dist(0.1)
            .spread(1.0)
            .n_epochs(50)
            .negative_sample_rate(5)
            .learning_rate(1.0)
            .build()
        )

        gumap.run(drawing, distance_matrix, rng)

        for u in graph.node_indices():
            self.assertTrue(math.isfinite(drawing.x(u)))
            self.assertTrue(math.isfinite(drawing.y(u)))

    def test_gumap_via_default_constructor(self):
        nx_graph = nx.cycle_graph(4)
        graph = draw(nx_graph)

        drawing = eg.DrawingEuclidean2d.initial_placement(graph)
        distance_matrix = eg.all_sources_dijkstra(graph, lambda _: 1.0)
        rng = eg.Rng.seed_from(42)

        gumap = eg.GuMap()
        gumap.run(drawing, distance_matrix, rng)

        for u in graph.node_indices():
            self.assertTrue(math.isfinite(drawing.x(u)))
            self.assertTrue(math.isfinite(drawing.y(u)))

    def test_gumap_validation_error(self):
        builder = eg.GuMapBuilder().k(0)
        with self.assertRaises(ValueError):
            builder.build()

        builder_epochs = eg.GuMapBuilder().n_epochs(0)
        with self.assertRaises(ValueError):
            builder_epochs.build()

    def test_gumap_run_with_knn_basic_and_matches_csr(self):
        n = 6
        graph = draw(nx.cycle_graph(n))
        knn_indices = np.array(
            [
                [(i + 1) % n, (i - 1) % n, (i + 2) % n, (i - 2) % n]
                for i in range(n)
            ],
            dtype=np.int64,
        )
        knn_distances = np.tile(np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float64), (n, 1))
        indptr = np.arange(0, (n + 1) * 4, 4, dtype=np.int64)
        indices = knn_indices.reshape(-1)
        distances = knn_distances.reshape(-1)

        gumap = eg.GuMapBuilder().k(4).n_epochs(40).build()
        dm = eg.all_sources_dijkstra(graph, lambda _: 1.0)

        d_run = eg.DrawingEuclidean2d.initial_placement(graph)
        d_knn = eg.DrawingEuclidean2d.initial_placement(graph)
        d_csr = eg.DrawingEuclidean2d.initial_placement(graph)
        for u in range(n):
            d_knn.set_x(u, d_run.x(u))
            d_knn.set_y(u, d_run.y(u))
            d_csr.set_x(u, d_run.x(u))
            d_csr.set_y(u, d_run.y(u))

        gumap.run(d_run, dm, eg.Rng.seed_from(42))
        gumap.run_with_knn(d_knn, knn_indices, knn_distances, eg.Rng.seed_from(42))
        gumap.run_with_knn_csr(d_csr, indptr, indices, distances, eg.Rng.seed_from(42))

        for u in range(n):
            self.assertTrue(math.isfinite(d_knn.x(u)))
            self.assertTrue(math.isfinite(d_knn.y(u)))
            self.assertEqual(d_run.x(u), d_knn.x(u))
            self.assertEqual(d_run.y(u), d_knn.y(u))
            self.assertEqual(d_run.x(u), d_csr.x(u))
            self.assertEqual(d_run.y(u), d_csr.y(u))

    def test_gumap_run_with_knn_index_dtypes_and_non_contiguous(self):
        n = 10
        graph = eg.Graph()
        for i in range(n):
            graph.add_node(i)
        coords = np.zeros((n, 2), dtype=np.float64)
        for i in range(5):
            ang = 2.0 * math.pi * i / 5
            coords[i] = [math.cos(ang) * 0.2, math.sin(ang) * 0.2]
            coords[5 + i] = [10.0 + math.cos(ang) * 0.2, math.sin(ang) * 0.2]
        diff = coords[:, None, :] - coords[None, :, :]
        dist_mat = np.sqrt(np.sum(diff * diff, axis=2))
        knn_i64 = np.zeros((n, 5), dtype=np.int64)
        knn_d = np.zeros((n, 5), dtype=np.float64)
        for i in range(n):
            order = [j for j in np.argsort(dist_mat[i]) if j != i][:5]
            knn_i64[i] = order
            knn_d[i] = dist_mat[i, order]

        gumap = eg.GuMapBuilder().k(5).n_epochs(40).build()
        for dtype in (np.int64, np.int32, np.uintp, np.uint64, np.uint32):
            d = eg.DrawingEuclidean2d.initial_placement(graph)
            gumap.run_with_knn(d, knn_i64.astype(dtype), knn_d, eg.Rng.seed_from(42))
            for u in range(n):
                self.assertTrue(math.isfinite(d.x(u)))
                self.assertTrue(math.isfinite(d.y(u)))

        # Non-contiguous [:, 1:] slice with self-loop column 0 + duplicates
        self_col = np.arange(n, dtype=np.int64)[:, None]
        self_dist = np.zeros((n, 1), dtype=np.float64)
        full_i = np.hstack([self_col, knn_i64])
        full_d = np.hstack([self_dist, knn_d])
        sliced_i = full_i[:, 1:]
        sliced_d = full_d[:, 1:]
        self.assertFalse(sliced_i.flags["C_CONTIGUOUS"])
        d_slice = eg.DrawingEuclidean2d.initial_placement(graph)
        gumap.run_with_knn(d_slice, sliced_i, sliced_d, eg.Rng.seed_from(42))

        dup_i = np.hstack([full_i, knn_i64[:, :1]])
        dup_d = np.hstack([full_d, knn_d[:, :1] + 1.0])
        d_dup = eg.DrawingEuclidean2d.initial_placement(graph)
        gumap.run_with_knn(d_dup, dup_i, dup_d, eg.Rng.seed_from(42))
        for u in range(n):
            self.assertTrue(math.isfinite(d_slice.x(u)))
            self.assertTrue(math.isfinite(d_dup.x(u)))

        # Strided CSR view
        indptr = np.arange(0, (n + 1) * 5, 5, dtype=np.int64)
        indices = knn_i64.reshape(-1)
        distances = knn_d.reshape(-1)
        d_str = eg.DrawingEuclidean2d.initial_placement(graph)
        gumap.run_with_knn_csr(
            d_str,
            np.repeat(indptr, 2)[::2],
            np.repeat(indices, 2)[::2],
            np.repeat(distances, 2)[::2],
            eg.Rng.seed_from(42),
        )

        if HAS_SCIPY:
            tree = cKDTree(coords)
            dists, idxs = tree.query(coords, k=5)
            d_tree = eg.DrawingEuclidean2d.initial_placement(graph)
            gumap.run_with_knn(d_tree, idxs[:, 1:], dists[:, 1:], eg.Rng.seed_from(42))
            for u in range(n):
                self.assertTrue(math.isfinite(d_tree.x(u)))

    def test_gumap_run_with_knn_edge_cases_and_validation(self):
        gumap = eg.GuMapBuilder().k(2).n_epochs(20).build()
        rng = eg.Rng.seed_from(42)

        # n=0, n=1, (n, 0) empty k-NN
        g0 = eg.Graph()
        d0 = eg.DrawingEuclidean2d.initial_placement(g0)
        gumap.run_with_knn(d0, np.zeros((0, 0), dtype=np.int64), np.zeros((0, 0), dtype=np.float64), rng)
        gumap.run_with_knn_csr(
            d0, np.array([0], dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), rng
        )

        g1 = eg.Graph()
        g1.add_node(0)
        d1 = eg.DrawingEuclidean2d.initial_placement(g1)
        gumap.run_with_knn(d1, np.zeros((1, 0), dtype=np.int64), np.zeros((1, 0), dtype=np.float64), rng)

        n = 4
        g4 = eg.Graph()
        for i in range(n):
            g4.add_node(i)
        d4 = eg.DrawingEuclidean2d.initial_placement(g4)
        valid_i = np.array([[1, 2], [0, 2], [1, 3], [0, 2]], dtype=np.int64)
        valid_d = np.ones((n, 2), dtype=np.float64)

        with self.assertRaises(ValueError):
            gumap.run_with_knn(d4, valid_i[:3], valid_d[:3], rng)
        with self.assertRaises(ValueError):
            gumap.run_with_knn(d4, valid_i, np.ones((n, 3), dtype=np.float64), rng)
        with self.assertRaises(ValueError):
            gumap.run_with_knn(d4, valid_i.ravel(), valid_d, rng)
        with self.assertRaises(ValueError):
            gumap.run_with_knn(d4, np.where(valid_i == 1, -1, valid_i), valid_d, rng)
        with self.assertRaises(ValueError):
            gumap.run_with_knn(d4, np.where(valid_i == 1, n, valid_i), valid_d, rng)
        for bad in (-0.1, np.nan, np.inf):
            bad_d = valid_d.copy()
            bad_d[0, 0] = bad
            with self.assertRaises(ValueError):
                gumap.run_with_knn(d4, valid_i, bad_d, rng)
        with self.assertRaises(TypeError):
            gumap.run_with_knn(d4, valid_i.astype(np.float64), valid_d, rng)
        with self.assertRaises(TypeError):
            gumap.run_with_knn(d4, valid_i, valid_d.astype(np.float32), rng)

        # CSR validation errors
        indptr = np.array([0, 2, 4, 5, 6], dtype=np.int64)
        indices = np.array([1, 2, 0, 2, 1, 2], dtype=np.int64)
        distances = np.ones(6, dtype=np.float64)
        with self.assertRaises(ValueError):
            gumap.run_with_knn_csr(d4, indptr[:-1], indices, distances, rng)
        with self.assertRaises(ValueError):
            gumap.run_with_knn_csr(d4, np.array([1, 2, 4, 5, 6], dtype=np.int64), indices, distances, rng)
        with self.assertRaises(ValueError):
            gumap.run_with_knn_csr(d4, np.array([0, 4, 2, 5, 6], dtype=np.int64), indices, distances, rng)
        with self.assertRaises(ValueError):
            gumap.run_with_knn_csr(d4, indptr, indices[:-1], distances, rng)
        with self.assertRaises(TypeError):
            gumap.run_with_knn_csr(d4, indptr.astype(np.float64), indices, distances, rng)
        with self.assertRaises(TypeError):
            gumap.run_with_knn_csr(d4, indptr, indices, distances.astype(np.float32), rng)

    def test_gumap_rejects_nan_inf_builder_and_nonfinite_drawing(self):
        for setter in ("learning_rate", "min_dist", "spread", "a", "b"):
            with self.assertRaises(ValueError):
                builder = eg.GuMapBuilder()
                getattr(builder, setter)(float("nan"))
                builder.build()
            with self.assertRaises(ValueError):
                builder = eg.GuMapBuilder()
                getattr(builder, setter)(float("inf"))
                builder.build()

        g = draw(nx.path_graph(4))
        dist = eg.all_sources_bfs(g, 1.0)
        gumap = eg.GuMapBuilder().n_epochs(5).build()
        rng = eg.Rng.seed_from(5)

        d_nan = eg.DrawingEuclidean2d.initial_placement(g)
        d_nan.set_x(0, float("nan"))
        with self.assertRaises(ValueError):
            gumap.run(d_nan, dist, rng)

        d_small = eg.DrawingEuclidean2d.initial_placement(draw(nx.path_graph(2)))
        with self.assertRaises(ValueError):
            gumap.run(d_small, dist, rng)


if __name__ == "__main__":
    unittest.main()
