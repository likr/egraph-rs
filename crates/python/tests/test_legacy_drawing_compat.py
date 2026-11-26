from __future__ import annotations

import unittest

import egraph as eg
import numpy as np


class TestLegacyDrawingCompat(unittest.TestCase):
    def test_legacy_graph_drawing_and_metrics_patterns(self) -> None:
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]], dtype=np.uintp)
        g = eg.Graph.from_edges_numpy(6, edges)
        self.assertEqual(g.node_count(), 6)
        self.assertEqual(g.edge_count(), 6)

        dg = eg.DiGraph()
        u0 = dg.add_node(0)
        u1 = dg.add_node(1)
        dg.add_edge(u0, u1, None)
        self.assertEqual(dg.node_count(), 2)

        d = eg.DrawingEuclidean2d.initial_placement(g)
        arr = d.to_numpy()
        self.assertEqual(arr.shape, (6, 2))
        d2 = eg.DrawingEuclidean2d.from_numpy(g, arr)
        d3 = eg.DrawingEuclidean2d.from_array2(g, eg.Array2.from_numpy(arr))
        d3.set_x(0, 0.25)
        d3.set_y(0, -0.25)
        x0 = d3.x(0)
        y0 = d3.y(0)
        assert x0 is not None and y0 is not None
        self.assertAlmostEqual(x0, 0.25)
        self.assertAlmostEqual(y0, -0.25)

        dm = eg.all_sources_dijkstra(g, lambda _e: 1.0)
        np1 = eg.neighborhood_preservation(g, d2)
        np2 = eg.neighborhood_preservation_2hop(g, d2)
        cr = eg.crossing_number(g, d2)
        st = eg.stress(d2, dm)
        self.assertTrue(all(np.isfinite([np1, np2, cr, st])))

    def test_legacy_tsnet_variants_gumap_and_sgd_patterns(self) -> None:
        n = 10
        edges = np.array(
            [[i, (i + 1) % n] for i in range(n)] + [[i, (i + 2) % n] for i in range(n)],
            dtype=np.uintp,
        )
        g = eg.Graph.from_edges_numpy(n, edges)
        rng = eg.Rng.seed_from(42)

        lap = eg.Laplacian.symmetric_normalized(g)
        self.assertIsInstance(lap, eg.Laplacian)

        pivots = [0, 2, 5, 8]
        kernel = eg.PivotedBandPassDiffusionKernel.dog(
            lap, 1.0, 4.0, 0.5, 1e-4, 15, pivots, rng=rng, num_hutchinson=16
        )
        builder = eg.PivotedNegLogSimDistanceBuilder()
        dist_mat = builder.build(g, kernel)
        self.assertEqual(dist_mat.shape()[0], n)

        d_piv = eg.DistanceMatrix.from_pivoted_numpy(
            g, pivots, np.ones((len(pivots), n), dtype=np.float64)
        )
        self.assertEqual(d_piv.shape(), (len(pivots), n))

        pmds = eg.PivotMds(g, (lambda _e: 1.0), pivots)
        d = pmds.run_2d()
        self.assertEqual(d.to_numpy().shape, (n, 2))

        indptr = np.arange(0, n * 3 + 1, 3, dtype=np.uintp)
        knn_indices = np.array(
            [[(i + 1) % n, (i + 2) % n, (i - 1) % n] for i in range(n)], dtype=np.uintp
        )
        knn_distances = np.full_like(knn_indices, 1.0, dtype=np.float64)

        fit = (
            eg.FitTsNetBuilder()
            .iterations_stage1(2)
            .iterations_stage2(2)
            .iterations_stage3(2)
            .build()
        )
        fit.run_with_knn(eg.DrawingEuclidean2d.initial_placement(g), knn_indices, knn_distances)
        fit.run_with_knn_adaptive(
            eg.DrawingEuclidean2d.initial_placement(g),
            indptr,
            knn_indices.ravel(),
            knn_distances.ravel(),
            np.full(n, 3.0, dtype=np.float64),
        )

        bh = (
            eg.BhTsNetBuilder()
            .iterations_stage1(2)
            .iterations_stage2(2)
            .iterations_stage3(2)
            .build()
        )
        bh.run_with_knn(eg.DrawingEuclidean2d.initial_placement(g), knn_indices, knn_distances)

        lts = (
            eg.LTsNetBuilder()
            .iterations_stage1(2)
            .iterations_stage2(2)
            .iterations_stage3(2)
            .build()
        )
        lts.run_with_knn(eg.DrawingEuclidean2d.initial_placement(g), knn_indices, knn_distances)

        gumap = eg.GuMap.builder().n_epochs(3).build()
        gumap.run_with_knn_csr(
            eg.DrawingEuclidean2d.initial_placement(g),
            indptr,
            knn_indices.ravel(),
            knn_distances.ravel(),
            eg.Rng.seed_from(42),
        )

    def test_legacy_sgd_barrier_stress_majorization_patterns(self) -> None:
        g = eg.Graph.from_edges_numpy(
            6, np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]], dtype=np.uintp)
        )
        d = eg.DrawingEuclidean2d.initial_placement(g)
        dm = eg.DistanceMatrix.from_numpy(g, np.ones((6, 6), dtype=np.float64) - np.eye(6))
        rng = eg.Rng.seed_from(42)

        full_sgd = eg.FullSgd().build_with_distance_matrix(dm)
        sched = full_sgd.scheduler(5, 0.1)
        for _ in range(5):
            full_sgd.shuffle(rng)
            sched.step(lambda eta: full_sgd.apply(d, eta))

        rp_sgd = eg.RandomPairSparseSgd().build(g, dm, rng)
        rp_sgd.shuffle(rng)
        rp_sgd.apply(d, 0.1)

        pair_sgd = eg.Sgd.from_numpy_pairs(
            np.array([0, 1], dtype=np.uintp),
            np.array([1, 2], dtype=np.uintp),
            np.array([1.0, 1.0], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
            n_nodes=6,
        )
        hinge_sgd = eg.Sgd.from_numpy_pairs_with_hinges(
            np.array([0], dtype=np.uintp),
            np.array([1], dtype=np.uintp),
            np.array([1.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            np.array([0], dtype=np.uintp),
            np.array([3], dtype=np.uintp),
            np.array([2.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            n_nodes=6,
        )
        exp_sched = eg.SchedulerExponential(5)
        exp_sched.run(lambda eta: pair_sgd.apply(d, eta))
        hinge_sgd.apply(d, 0.1)

        d_ms = np.full((6, 2), 2.0, dtype=np.float64)
        d_ms[0, 0] = 0.0
        d_ms[3, 1] = 0.0
        barrier_sgd, stats = eg.SparseSgd.barrier(h=2, max_2hop_per_node=8).build_barrier_from_landmark_tables(
            g, d_ms, pivots=np.array([0, 3], dtype=np.int64), neg_per_node=2
        )
        barrier_sgd.apply(d, 0.1)
        self.assertIn("n_pairs_hinge", stats)
        self.assertIn("n_pairs_1hop", stats)

        sm = eg.StressMajorization(g, d, lambda _e: 1.0)
        sm.run(d)
        self.assertTrue(np.all(np.isfinite(d.to_numpy())))


if __name__ == "__main__":
    unittest.main()
