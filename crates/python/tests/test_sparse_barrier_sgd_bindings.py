"""Unit tests for native Sgd hinge repulsion and SparseSgd barrier PyO3 bindings."""

import math
import unittest

import egraph as eg
import numpy as np


def make_graph(n: int, edges: list[tuple[int, int]]) -> eg.Graph:
    graph = eg.Graph()
    nodes = [graph.add_node(i) for i in range(n)]
    for u, v in edges:
        graph.add_edge(nodes[u], nodes[v], (u, v))
    return graph


class TestPySgdHingePairs(unittest.TestCase):
    def test_from_numpy_hinge_pairs_zero_force_when_separated_and_repels_close(self):
        graph = make_graph(2, [(0, 1)])
        src = np.array([0], dtype=np.int64)
        tgt = np.array([1], dtype=np.int64)
        dist = np.array([2.0], dtype=np.float64)
        wt = np.array([1.0], dtype=np.float64)

        sgd_hinge = eg.Sgd.from_numpy_hinge_pairs(src, tgt, dist, wt, n_nodes=2)
        self.assertEqual(sgd_hinge.node_pair_count(), 0)
        self.assertEqual(sgd_hinge.hinge_pair_count(), 1)

        # When ||x_0 - x_1|| = 3.0 >= 2.0, hinge exerts exact 0.0 force
        d_far = eg.DrawingEuclidean2d.initial_placement(graph)
        d_far.set_x(0, 0.0)
        d_far.set_y(0, 0.0)
        d_far.set_x(1, 3.0)
        d_far.set_y(1, 0.0)
        sgd_hinge.apply(d_far, 0.5)
        self.assertEqual(d_far.x(0), 0.0)
        self.assertEqual(d_far.y(0), 0.0)
        self.assertEqual(d_far.x(1), 3.0)
        self.assertEqual(d_far.y(1), 0.0)

        # Contrast with two-sided spring which pulls 3.0 inward toward 2.0
        d_two = eg.DrawingEuclidean2d.initial_placement(graph)
        d_two.set_x(0, 0.0)
        d_two.set_y(0, 0.0)
        d_two.set_x(1, 3.0)
        d_two.set_y(1, 0.0)
        sgd_two = eg.Sgd.from_numpy_pairs(src, tgt, dist, wt)
        sgd_two.apply(d_two, 0.5)
        self.assertGreater(d_two.x(0), 0.0)
        self.assertLess(d_two.x(1), 3.0)

        # When ||x_0 - x_1|| = 0.5 < 2.0, hinge repels outward
        d_close = eg.DrawingEuclidean2d.initial_placement(graph)
        d_close.set_x(0, 0.0)
        d_close.set_y(0, 0.0)
        d_close.set_x(1, 0.5)
        d_close.set_y(1, 0.0)
        sgd_hinge.apply(d_close, 0.5)
        self.assertLess(d_close.x(0), 0.0)
        self.assertGreater(d_close.x(1), 0.5)

    def test_from_numpy_pairs_with_hinges_and_add_hinge_pairs(self):
        graph = make_graph(3, [(0, 1)])
        s_src = np.array([0], dtype=np.int32)
        s_tgt = np.array([1], dtype=np.int32)
        s_dist = np.array([1.0], dtype=np.float64)
        s_wt = np.array([1.0], dtype=np.float64)

        h_src = np.array([0], dtype=np.uintp)
        h_tgt = np.array([2], dtype=np.uintp)
        h_dist = np.array([2.0], dtype=np.float64)
        h_wt = np.array([0.25], dtype=np.float64)

        sgd = eg.Sgd.from_numpy_pairs_with_hinges(
            s_src, s_tgt, s_dist, s_wt, h_src, h_tgt, h_dist, h_wt, n_nodes=3
        )
        self.assertEqual(sgd.node_pair_count(), 1)
        self.assertEqual(sgd.hinge_pair_count(), 1)

        d = eg.DrawingEuclidean2d.initial_placement(graph)
        d.set_x(0, 0.0)
        d.set_y(0, 0.0)
        d.set_x(1, 4.0)
        d.set_y(1, 0.0)
        d.set_x(2, 0.0)
        d.set_y(2, 5.0)
        sgd.apply(d, 0.5)
        self.assertGreater(d.x(0), 0.0)
        self.assertLess(d.x(1), 4.0)
        self.assertEqual(d.x(2), 0.0)
        self.assertEqual(d.y(2), 5.0)

        # Append via add_hinge_pairs_from_numpy and verify empty arrays are allowed
        sgd2 = eg.Sgd.from_numpy_pairs(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
        sgd2.add_hinge_pairs_from_numpy(h_src, h_tgt, h_dist, h_wt, n_nodes=3)
        self.assertEqual(sgd2.hinge_pair_count(), 1)

    def test_hinge_pair_validation_errors(self):
        ok_u = np.array([0], dtype=np.int64)
        ok_v = np.array([1], dtype=np.int64)
        ok_d = np.array([1.0], dtype=np.float64)
        ok_w = np.array([1.0], dtype=np.float64)

        # Shape mismatch
        with self.assertRaises(ValueError):
            eg.Sgd.from_numpy_hinge_pairs(ok_u, np.array([1, 2], dtype=np.int64), ok_d, ok_w)
        # 2D array
        with self.assertRaises(ValueError):
            eg.Sgd.from_numpy_hinge_pairs(ok_u[:, None], ok_v, ok_d, ok_w)
        # Negative index
        with self.assertRaises(ValueError):
            eg.Sgd.from_numpy_hinge_pairs(np.array([-1], dtype=np.int64), ok_v, ok_d, ok_w)
        # Out-of-bounds for n_nodes
        with self.assertRaises(ValueError):
            eg.Sgd.from_numpy_hinge_pairs(ok_u, np.array([2], dtype=np.int64), ok_d, ok_w, n_nodes=2)
        # Self-loop
        with self.assertRaises(ValueError):
            eg.Sgd.from_numpy_hinge_pairs(ok_u, ok_u, ok_d, ok_w, n_nodes=2)
        # Non-positive / NaN / Inf distances and weights
        for bad in (0.0, -1.0, np.nan, np.inf):
            with self.assertRaises(ValueError):
                eg.Sgd.from_numpy_hinge_pairs(ok_u, ok_v, np.array([bad], dtype=np.float64), ok_w)
            with self.assertRaises(ValueError):
                eg.Sgd.from_numpy_hinge_pairs(ok_u, ok_v, ok_d, np.array([bad], dtype=np.float64))
        # Wrong dtypes -> TypeError
        with self.assertRaises(TypeError):
            eg.Sgd.from_numpy_hinge_pairs(ok_u.astype(np.float64), ok_v, ok_d, ok_w)
        with self.assertRaises(TypeError):
            eg.Sgd.from_numpy_hinge_pairs(ok_u, ok_v, ok_d.astype(np.float32), ok_w)
        with self.assertRaises(TypeError):
            eg.Sgd.from_numpy_hinge_pairs(ok_u, ok_v, ok_d, ok_w.astype(np.float32))


class TestPySparseSgdBarrier(unittest.TestCase):
    def test_sparse_sgd_barrier_with_pivoted_distances_cycle_and_star(self):
        # Cycle(12) with PivotedBandPassDiffusionKernelBuilder.multiscale_heat + PivotedNegLogSimDistance
        n_c = 12
        cycle = make_graph(n_c, [(i, (i + 1) % n_c) for i in range(n_c)])
        lap = eg.Laplacian.symmetric_normalized(cycle)
        pivots = [0, 4, 8]
        ms_terms = [(0.5, 0.5), (0.3, 2.0), (0.2, 8.0)]
        kernel = (
            eg.PivotedBandPassDiffusionKernelBuilder.multiscale_heat(
                ms_terms, degree=20, pivots=pivots
            )
            .num_hutchinson(16)
            .build(lap, rng=eg.Rng.seed_from(42))
        )
        dist = eg.PivotedNegLogSimDistance(cycle, kernel)

        builder = eg.SparseSgd.barrier(h=3, max_2hop_per_node=4, delta_rel=0.25)
        sgd = builder.build_with_pivoted_distance(cycle, dist)
        self.assertGreater(sgd.node_pair_count(), n_c)
        self.assertGreater(sgd.hinge_pair_count(), 0)

        drawing = eg.DrawingEuclidean2d.initial_placement(cycle)
        rng = eg.Rng.seed_from(42)
        sched = sgd.scheduler_exponential(40, 0.01)
        sched.run(lambda eta: (sgd.shuffle(rng), sgd.apply(drawing, eta)))
        for i in range(n_c):
            self.assertTrue(math.isfinite(drawing.x(i)))
            self.assertTrue(math.isfinite(drawing.y(i)))

        # Star(1000) with DistanceMatrix.from_pivoted_numpy -> hub-safe O(N*cap) pairs
        n_s = 1000
        star = make_graph(n_s, [(0, i) for i in range(1, n_s)])
        piv_mat = np.zeros((2, n_s), dtype=np.float64)
        piv_mat[0, 1:] = 1.0
        piv_mat[1, 0] = 1.0
        piv_mat[1, 2:] = 2.0
        dm = eg.DistanceMatrix.from_pivoted_numpy(star, [0, 1], piv_mat)
        sgd_star = eg.SparseSgd.barrier(h=2, max_2hop_per_node=16).build_with_pivoted_distance(star, dm)
        self.assertLessEqual(sgd_star.node_pair_count(), (n_s - 1) + n_s * 16)
        self.assertEqual(sgd_star.hinge_pair_count(), 0)

    def test_build_barrier_from_landmark_tables_and_validation(self):
        # 5-node path 0-1-2-3-4
        path = make_graph(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
        d_ms = np.arange(5, dtype=np.float64)[:, None]
        builder = eg.SparseSgd.barrier(
            max_2hop_per_node=32,
            delta_rel=0.5,
            barrier_sym="sym_max",
            omega_1=2.0,
            omega_2=0.5,
            gamma_non2=3.0,
            min_ratio=0.5,
        )
        sgd, stats = builder.build_barrier_from_landmark_tables(
            path, d_ms, pivots=np.array([0, 4], dtype=np.int64), neg_per_node=8
        )
        self.assertEqual(stats["n_pairs_1hop"], 4)
        self.assertEqual(stats["n_pairs_2hop"], 3)
        self.assertEqual(stats["n_pairs_non2"], 3)
        self.assertEqual(stats["n_pairs_two_sided"], 7)
        self.assertEqual(stats["n_pairs_hinge"], 3)
        self.assertEqual(sgd.node_pair_count(), 7)
        self.assertEqual(sgd.hinge_pair_count(), 3)

        # Validation errors on Python builder
        with self.assertRaises(ValueError):
            eg.SparseSgd.barrier(barrier_sym="invalid")
        with self.assertRaises(ValueError):
            eg.SparseSgd.barrier(delta_rel=-0.1)
        with self.assertRaises(ValueError):
            eg.SparseSgd.barrier(eps_dist=0.0)
        with self.assertRaises(ValueError):
            builder.build_barrier_from_landmark_tables(path, np.zeros((3, 1), dtype=np.float64))
        with self.assertRaises(TypeError):
            builder.build_barrier_from_landmark_tables(path, np.zeros((5, 1), dtype=np.float32))


class TestAdversarialSgdAndSparseSgd(unittest.TestCase):
    def test_sparse_sgd_disconnected_and_zero_weight_no_panic(self):
        rng = eg.Rng.seed_from(7)
        g_disc = make_graph(4, [(0, 1), (2, 3)])
        sgd_disc = eg.SparseSgd().h(3).build(g_disc, lambda _: 1.0, rng)
        d_disc = eg.DrawingEuclidean2d.initial_placement(g_disc)
        sgd_disc.scheduler(5, 0.1).run(lambda eta: (sgd_disc.shuffle(rng), sgd_disc.apply(d_disc, eta)))
        for i in range(4):
            self.assertTrue(math.isfinite(d_disc.x(i)))
            self.assertTrue(math.isfinite(d_disc.y(i)))

        g_path = make_graph(4, [(0, 1), (1, 2), (2, 3)])
        sgd_zero = eg.SparseSgd().h(3).build(g_path, lambda _: 0.0, rng)
        d_zero = eg.DrawingEuclidean2d.initial_placement(g_path)
        sgd_zero.scheduler(5, 0.1).run(lambda eta: (sgd_zero.shuffle(rng), sgd_zero.apply(d_zero, eta)))
        for i in range(4):
            self.assertTrue(math.isfinite(d_zero.x(i)))
            self.assertTrue(math.isfinite(d_zero.y(i)))

    def test_sparse_sgd_duplicate_pivots_and_barrier_n0_n1_validation(self):
        rng = eg.Rng.seed_from(11)
        path = make_graph(4, [(0, 1), (1, 2), (2, 3)])
        dist = eg.all_sources_bfs(path, 1.0)
        sgd_dup = eg.SparseSgd().h(2).build_with_pivot_and_distance_matrix(path, [0, 0, 2], dist)
        d = eg.DrawingEuclidean2d.initial_placement(path)
        sgd_dup.scheduler(5, 0.1).run(lambda eta: (sgd_dup.shuffle(rng), sgd_dup.apply(d, eta)))
        for i in range(4):
            self.assertTrue(math.isfinite(d.x(i)))
            self.assertTrue(math.isfinite(d.y(i)))

        barrier = eg.SparseSgd.barrier(h=2, delta_rel=0.35)
        g0 = make_graph(0, [])
        sgd0 = barrier.build_barrier_from_landmark_tables(g0, np.zeros((0, 0), dtype=np.float64))
        self.assertIsNotNone(sgd0)
        with self.assertRaises(ValueError):
            barrier.build_barrier_from_landmark_tables(g0, np.zeros((3, 1), dtype=np.float64))

        g1 = make_graph(1, [])
        sgd1 = barrier.build_barrier_from_landmark_tables(g1, np.zeros((1, 1), dtype=np.float64))
        self.assertIsNotNone(sgd1)
        with self.assertRaises(ValueError):
            barrier.build_barrier_from_landmark_tables(g1, np.zeros((1, 0), dtype=np.float64))
        with self.assertRaises(ValueError):
            barrier.build_barrier_from_landmark_tables(
                g1, np.array([[float("nan")]], dtype=np.float64)
            )

    def test_sparse_sgd_mismatched_pivoted_distance_raises(self):
        rng = eg.Rng.seed_from(13)
        g6 = make_graph(6, [(i, i + 1) for i in range(5)])
        g10 = make_graph(10, [(i, i + 1) for i in range(9)])
        lap6 = eg.Laplacian.symmetric_normalized(g6)
        pk6 = eg.PivotedDiffusionKernel(lap6, 0.5, 8, [0, 2], rng)
        dist_sim_6 = eg.PivotedNegLogSimDistanceBuilder().build(g6, pk6)
        with self.assertRaises(ValueError):
            eg.PivotedNegLogSimDistanceBuilder().build(g10, pk6)
        with self.assertRaises(ValueError):
            eg.PivotedNegLogDistanceBuilder().build(g10, pk6)
        with self.assertRaises(ValueError):
            eg.SparseSgd().h(2).build_with_pivoted_distance(g10, dist_sim_6)

    def test_sgd_empty_small_tmax_schedulers_and_oob_updates(self):
        rng = eg.Rng.seed_from(19)
        sgd_empty = eg.Sgd.from_numpy_pairs(
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
        for sched in (
            sgd_empty.scheduler(10, 0.1),
            sgd_empty.scheduler_linear(10, 0.1),
            sgd_empty.scheduler_quadratic(10, 0.1),
            sgd_empty.scheduler_reciprocal(10, 0.1),
            eg.SchedulerExponential(1, 1.0, 0.1),
            eg.SchedulerLinear(0, 1.0, 0.1),
            eg.SchedulerQuadratic(1, 1.0, 0.1),
            eg.SchedulerReciprocal(1, 1.0, 0.1),
        ):
            etas = []
            sched.run(lambda eta: etas.append(eta))
            self.assertTrue(all(math.isfinite(e) and e >= 0.0 for e in etas))

        g2 = make_graph(2, [(0, 1)])
        d2 = eg.DrawingEuclidean2d.initial_placement(g2)
        d2.set_x(0, 0.0)
        d2.set_y(0, 0.0)
        d2.set_x(1, 1.0)
        d2.set_y(1, 0.0)
        sgd = eg.Sgd.from_numpy_pairs(
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([2.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        )
        sgd.apply(d2, float("nan"))
        sgd.apply(d2, 0.0)
        self.assertEqual(d2.x(0), 0.0)
        self.assertEqual(d2.x(1), 1.0)

        g1_small = make_graph(1, [])
        d1_small = eg.DrawingEuclidean2d.initial_placement(g1_small)
        with self.assertRaises(ValueError):
            sgd.apply(d1_small, 0.1)
        with self.assertRaises(TypeError):
            sgd.update_distance("not_callable")
        with self.assertRaises(TypeError):
            sgd.update_weight("not_callable")
        with self.assertRaises(ZeroDivisionError):
            sgd.update_distance(lambda i, j, d, w: 1.0 / 0.0)
        with self.assertRaises(ZeroDivisionError):
            sgd.update_weight(lambda i, j, d, w: 1.0 / 0.0)
        with self.assertRaises(ValueError):
            sgd.update_weight_from_numpy(np.zeros((1, 1), dtype=np.float64))
        with self.assertRaises(ValueError):
            sgd.update_distance_from_numpy(np.zeros((1, 1), dtype=np.float64))
        with self.assertRaises(ValueError):
            eg.Sgd.from_numpy_pairs(
                np.array([[0]], dtype=np.int64),
                np.array([1], dtype=np.int64),
                np.array([1.0], dtype=np.float64),
                np.array([1.0], dtype=np.float64),
            )
        with self.assertRaises(ValueError):
            eg.Sgd.from_numpy_pairs(
                np.array([0], dtype=np.int64),
                np.array([1], dtype=np.int64),
                np.array([float("nan")], dtype=np.float64),
                np.array([1.0], dtype=np.float64),
            )


if __name__ == "__main__":
    unittest.main()
