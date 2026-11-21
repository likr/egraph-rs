import math
import unittest

import egraph as eg
import numpy as np


class TestResidualAdversarialHardening(unittest.TestCase):
    def _path3(self):
        g = eg.Graph()
        nodes = [g.add_node(i) for i in range(3)]
        g.add_edge(nodes[0], nodes[1], 1.0)
        g.add_edge(nodes[1], nodes[2], 1.0)
        return g

    def _cycle10(self):
        g = eg.Graph()
        nodes = [g.add_node(i) for i in range(10)]
        for i in range(10):
            g.add_edge(nodes[i], nodes[(i + 1) % 10], 1.0)
        return g

    # R1: Edge-length validation, callback error propagation & DiGraph TypeError
    def test_r1_sparse_and_full_sgd_reject_negative_nan_bad_callbacks_and_digraph(self):
        g = self._path3()
        rng = eg.Rng.seed_from(1)
        for bad_len in [-1.0, float("nan"), float("inf")]:
            with self.assertRaises((ValueError, TypeError)):
                eg.SparseSgd().build(g, bad_len, rng)
            with self.assertRaises((ValueError, TypeError)):
                eg.SparseSgd().build(g, lambda _, v=bad_len: v, rng)
            with self.assertRaises((ValueError, TypeError)):
                eg.SparseSgd.barrier().new_with_pivot(g, bad_len, [0])
            with self.assertRaises((ValueError, TypeError)):
                eg.FullSgd().build(g, lambda _, v=bad_len: v)
            with self.assertRaises((ValueError, TypeError)):
                eg.FullSgd().build(g, bad_len)
            with self.assertRaises((ValueError, TypeError)):
                eg.all_sources_dijkstra(g, lambda _, v=bad_len: v)

        with self.assertRaises(TypeError):
            eg.SparseSgd().build(g, lambda _: "not_a_float", rng)
        with self.assertRaises(TypeError):
            eg.FullSgd().build(g, lambda _: "not_a_float")
        with self.assertRaises(ZeroDivisionError):
            eg.SparseSgd().build(g, lambda _: 1.0 / 0.0, rng)
        with self.assertRaises(ZeroDivisionError):
            eg.FullSgd().build(g, lambda _: 1.0 / 0.0)

        dg = eg.DiGraph()
        dg.add_node(0)
        with self.assertRaises(TypeError):
            eg.FullSgd().build(dg)
        d_dg = eg.DrawingEuclidean2d.initial_placement(dg)
        with self.assertRaises(TypeError):
            eg.StressMajorization(dg, d_dg)

    # R2: DistanceMatrix OOB, EmbeddingKernel/NegLog* mismatch, FullSgd/MDS pivoted rejection
    def test_r2_distance_matrix_and_kernel_bounds_and_shape_validation(self):
        g3 = self._path3()
        g10 = self._cycle10()
        dm_full = eg.DistanceMatrix.from_numpy(g3, np.ones((3, 3)))
        dm_piv = eg.DistanceMatrix.from_pivoted_numpy(g3, [0], np.ones((1, 3)))
        with self.assertRaises(IndexError):
            dm_full.get_by_index(0, 99)
        with self.assertRaises(IndexError):
            dm_piv.get_by_index(99, 0)

        with self.assertRaises(ValueError):
            eg.EmbeddingKernel(g10, eg.Array2(np.zeros((2, 2))))
        with self.assertRaises(ValueError):
            eg.EmbeddingKernelBuilder().build(g10, eg.Array2(np.zeros((2, 2))))

        lap4 = eg.Laplacian.symmetric_normalized(
            eg.Graph.from_edges_numpy(4, np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64))
        )
        dk4 = eg.DiffusionKernel(lap4, 1.0, 10, eg.Rng.seed_from(1))
        with self.assertRaises(ValueError):
            eg.NegLogSimDistanceBuilder().build(g10, dk4)
        with self.assertRaises(ValueError):
            eg.NegLogDistanceBuilder().build(g10, dk4)
        with self.assertRaises(ValueError):
            eg.NegLogSimDistanceBuilder().alpha(float("nan")).build(
                eg.Graph.from_edges_numpy(4, np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)),
                dk4,
            )

        lap10 = eg.Laplacian.symmetric_normalized(g10)
        pk10 = eg.PivotedBandPassDiffusionKernel.dog(
            lap10, 0.5, 2.0, 0.5, 0.05, 10, [0, 5], rng=eg.Rng.seed_from(1)
        )
        pdist10 = eg.PivotedNegLogSimDistance(g10, pk10)
        pnd10 = eg.PivotedNegLogDistanceBuilder().build(g10, pk10)
        dm_piv10 = eg.DistanceMatrix.from_pivoted_numpy(g10, [0, 5], np.ones((2, 10)))

        with self.assertRaises((ValueError, TypeError)):
            eg.FullSgd().build_with_distance_matrix(pdist10)
        with self.assertRaises((ValueError, TypeError)):
            eg.FullSgd().build_with_distance_matrix(dm_piv10)
        with self.assertRaises((ValueError, TypeError)):
            eg.FullSgd().build_with_distance_matrix(pnd10)
        with self.assertRaises((ValueError, TypeError)):
            eg.ClassicalMds.new_with_distance_matrix(dm_piv10)
        with self.assertRaises((ValueError, TypeError)):
            eg.ClassicalMds.new_with_distance_matrix(pdist10)
        with self.assertRaises((ValueError, TypeError)):
            eg.ClassicalMds.new_with_distance_matrix(pnd10)
        with self.assertRaises((ValueError, TypeError)):
            eg.PivotMds.new_with_distance_matrix(pdist10)
        with self.assertRaises((ValueError, TypeError)):
            eg.PivotMds.new_with_distance_matrix(pnd10)

        # Valid (h, N) PivotedDistanceMatrix still works with PivotMds
        pmds_ok = eg.PivotMds.new_with_distance_matrix(dm_piv10).run_2d()
        self.assertEqual(pmds_ok.len(), 10)

    # R3: Non-bandpass kernel parameter validation
    def test_r3_non_bandpass_kernels_reject_invalid_t_and_alpha(self):
        g10 = self._cycle10()
        lap10 = eg.Laplacian.symmetric_normalized(g10)
        rng = eg.Rng.seed_from(1)
        evals = eg.Array1(np.array([0.0, 1.0], dtype=np.float64))
        evecs = eg.Array2(np.zeros((10, 2), dtype=np.float64))

        for bad_t in [-1.0, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                eg.DiffusionKernel(lap10, bad_t, 10, rng)
            with self.assertRaises(ValueError):
                eg.PivotedDiffusionKernel(lap10, bad_t, 10, [0], rng)
            with self.assertRaises(ValueError):
                eg.LowRankDiffusionKernel(bad_t, evals, evecs)
            with self.assertRaises(ValueError):
                eg.LowRankDiffusionKernelBuilder().t(bad_t).build_unnormalized_laplacian(g10)
            with self.assertRaises(ValueError):
                eg.LowRankDiffusionKernelBuilder().t(bad_t).build_symmetric_normalized_laplacian(g10)

        for bad_alpha in [-0.1, 1.0, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                eg.MultiscaleDiffusionKernel(lap10, bad_alpha)
            with self.assertRaises(ValueError):
                eg.PivotedMultiscaleDiffusionKernel(lap10, bad_alpha, [0])

    # R4: Quality metrics edge cases
    def test_r4_quality_metrics_degenerate_finite(self):
        g0 = eg.Graph()
        d0 = eg.DrawingEuclidean2d.initial_placement(g0)
        self.assertEqual(eg.aspect_ratio(d0), 0.0)

        g1 = eg.Graph()
        g1.add_node(0)
        d1 = eg.DrawingEuclidean2d.initial_placement(g1)
        self.assertEqual(eg.aspect_ratio(d1), 0.0)

        g3 = self._path3()
        d_zero = eg.DrawingEuclidean2d.initial_placement(g3)
        for i in range(3):
            d_zero.set_x(i, 0.0)
            d_zero.set_y(i, 0.0)
        ar_zero = eg.aspect_ratio(d_zero)
        self.assertTrue(math.isfinite(ar_zero))
        self.assertEqual(ar_zero, 0.0)

        d_nan = eg.DrawingEuclidean2d.initial_placement(g3)
        d_nan.set_x(0, float("nan"))
        self.assertEqual(eg.aspect_ratio(d_nan), 0.0)

        d3 = eg.DrawingEuclidean2d.initial_placement(g3)
        dm_zero = eg.DistanceMatrix.from_numpy(g3, np.zeros((3, 3), dtype=np.float64))
        dm_nan = eg.DistanceMatrix.from_numpy(g3, np.full((3, 3), float("nan"), dtype=np.float64))
        self.assertTrue(math.isfinite(eg.stress(d3, dm_zero)))
        self.assertTrue(math.isfinite(eg.stress(d3, dm_nan)))
        self.assertTrue(math.isfinite(eg.ideal_edge_lengths(g3, d3, dm_zero)))
        self.assertTrue(math.isfinite(eg.ideal_edge_lengths(g3, d3, dm_nan)))

    # R5: StressMajorization empty/singleton, mismatched shapes, pivoted matrix, bad callbacks, NaN guards
    def test_r5_stress_majorization_no_panics_on_empty_mismatch_pivoted_or_bad_callback(self):
        g0 = eg.Graph()
        d0 = eg.DrawingEuclidean2d.initial_placement(g0)
        dm0 = eg.DistanceMatrix.from_numpy(g0, np.zeros((0, 0), dtype=np.float64))
        sm0 = eg.StressMajorization(g0, d0)
        self.assertEqual(sm0.apply(d0), 0.0)
        sm0.run(d0)
        sm0_dm = eg.StressMajorization.with_distance_matrix(d0, dm0)
        sm0_dm.run(d0)

        g1 = eg.Graph()
        g1.add_node(0)
        d1 = eg.DrawingEuclidean2d.initial_placement(g1)
        dm1 = eg.DistanceMatrix.from_numpy(g1, np.zeros((1, 1), dtype=np.float64))
        sm1 = eg.StressMajorization(g1, d1)
        self.assertEqual(sm1.apply(d1), 0.0)
        sm1.run(d1)
        sm1_dm = eg.StressMajorization.with_distance_matrix(d1, dm1)
        sm1_dm.run(d1)

        g2 = eg.Graph()
        n2 = [g2.add_node(i) for i in range(2)]
        g2.add_edge(n2[0], n2[1], 1.0)
        d2 = eg.DrawingEuclidean2d.initial_placement(g2)
        dm2 = eg.DistanceMatrix.from_numpy(g2, np.ones((2, 2), dtype=np.float64))

        g3 = self._path3()
        d3 = eg.DrawingEuclidean2d.initial_placement(g3)
        dm_piv3 = eg.DistanceMatrix.from_pivoted_numpy(g3, [0], np.ones((1, 3), dtype=np.float64))

        with self.assertRaises(ValueError):
            eg.StressMajorization(g3, d2)
        with self.assertRaises(ValueError):
            eg.StressMajorization.with_distance_matrix(d3, dm2)
        with self.assertRaises((TypeError, ValueError)):
            eg.StressMajorization.with_distance_matrix(d3, dm_piv3)

        d3_nan = eg.DrawingEuclidean2d.initial_placement(g3)
        d3_nan.set_x(0, float("nan"))
        with self.assertRaises(ValueError):
            eg.StressMajorization(g3, d3_nan)

        sm3 = eg.StressMajorization(g3, d3)
        with self.assertRaises(ValueError):
            sm3.apply(d2)
        with self.assertRaises(ValueError):
            sm3.run(d2)
        with self.assertRaises(ValueError):
            sm3.apply(d3_nan)
        with self.assertRaises(ValueError):
            sm3.run(d3_nan)
        with self.assertRaises(TypeError):
            sm3.update_weight("not_callable")
        with self.assertRaises(TypeError):
            sm3.update_weight(lambda i, j, d, w: "bad")
        with self.assertRaises(TypeError):
            sm3.update_weight(lambda i, j, d, w: True)
        with self.assertRaises(ZeroDivisionError):
            sm3.update_weight(lambda i, j, d, w: 1.0 / 0.0)
        for bad_w in [-1.0, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                sm3.update_weight(lambda i, j, d, w, v=bad_w: v)
            w_arr = np.ones((3, 3), dtype=np.float64)
            w_arr[0, 1] = bad_w
            with self.assertRaises(ValueError):
                sm3.update_weight_from_numpy(w_arr)
        for bad_eps in [0.0, -1e-4, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                sm3.epsilon = bad_eps

        # Disconnected graph keeps finite coordinates and diff
        g_disc = eg.Graph()
        dn = [g_disc.add_node(i) for i in range(4)]
        g_disc.add_edge(dn[0], dn[1], 1.0)
        g_disc.add_edge(dn[2], dn[3], 1.0)
        d_disc = eg.DrawingEuclidean2d.initial_placement(g_disc)
        sm_disc = eg.StressMajorization(g_disc, d_disc)
        self.assertTrue(math.isfinite(sm_disc.apply(d_disc)))
        sm_disc.run(d_disc)
        for u in range(4):
            self.assertTrue(math.isfinite(d_disc.x(u)))
            self.assertTrue(math.isfinite(d_disc.y(u)))

    # R6: LowRank*DiffusionKernelBuilder shift & tolerance validation
    def test_r6_low_rank_builders_reject_invalid_shift_and_tolerances(self):
        g3 = self._path3()
        for bad_val in [float("nan"), float("inf"), 0.0, -1.0]:
            with self.assertRaises(ValueError):
                eg.LowRankDiffusionKernelBuilder().shift(bad_val).build_unnormalized_laplacian(g3)
            with self.assertRaises(ValueError):
                eg.LowRankDiffusionKernelBuilder().shift(bad_val).build_symmetric_normalized_laplacian(g3)
            with self.assertRaises(ValueError):
                eg.LowRankMultiscaleDiffusionKernelBuilder().shift(bad_val).build_unnormalized_laplacian(g3)
            with self.assertRaises(ValueError):
                eg.LowRankMultiscaleDiffusionKernelBuilder().shift(bad_val).build_symmetric_normalized_laplacian(g3)
            with self.assertRaises(ValueError):
                eg.LowRankDiffusionKernelBuilder().eigenvalue_tolerance(bad_val).build_unnormalized_laplacian(g3)
            with self.assertRaises(ValueError):
                eg.LowRankDiffusionKernelBuilder().cg_tolerance(bad_val).build_symmetric_normalized_laplacian(g3)
            with self.assertRaises(ValueError):
                eg.LowRankMultiscaleDiffusionKernelBuilder().eigenvalue_tolerance(bad_val).build_unnormalized_laplacian(g3)
            with self.assertRaises(ValueError):
                eg.LowRankMultiscaleDiffusionKernelBuilder().cg_tolerance(bad_val).build_symmetric_normalized_laplacian(g3)

    # R7: GaussianKernel, ExponentialKernel, TKernel, KernelDistance & LowRank array validation
    def test_r7_remaining_kernels_and_kernel_distance_reject_invalid_params(self):
        g3 = self._path3()
        lap3 = eg.Laplacian.symmetric_normalized(g3)
        dk3 = eg.DiffusionKernel(lap3, 1.0, 10, eg.Rng.seed_from(1))

        # Valid boundary gamma=0.0 and min_dist=0.0 remain accepted
        self.assertTrue(math.isfinite(eg.GaussianKernel(dk3, 0.0).get(0, 1)))
        self.assertTrue(math.isfinite(eg.ExponentialKernel(dk3, 0.0).get(0, 1)))
        self.assertIsNotNone(eg.KernelDistance(dk3, 0.0))

        for bad_gamma in [-1.0, float("nan"), float("inf"), float("-inf")]:
            with self.assertRaises(ValueError):
                eg.GaussianKernel(dk3, bad_gamma)
            with self.assertRaises(ValueError):
                eg.GaussianKernelBuilder(bad_gamma).build(dk3)
            with self.assertRaises(ValueError):
                eg.ExponentialKernel(dk3, bad_gamma)
            with self.assertRaises(ValueError):
                eg.ExponentialKernelBuilder(bad_gamma).build(dk3)

        for bad_alpha in [0.0, -1.0, float("nan"), float("inf"), float("-inf")]:
            with self.assertRaises(ValueError):
                eg.TKernel(dk3, bad_alpha)
            with self.assertRaises(ValueError):
                eg.TKernelBuilder(bad_alpha).build(dk3)

        for bad_min_dist in [-1.0, float("nan"), float("inf"), float("-inf")]:
            with self.assertRaises(ValueError):
                eg.KernelDistance(dk3, bad_min_dist)
            with self.assertRaises(ValueError):
                eg.KernelDistanceBuilder(bad_min_dist).build(dk3)

        nan_evals = eg.Array1(np.array([float("nan")], dtype=np.float64))
        inf_evals = eg.Array1(np.array([float("inf")], dtype=np.float64))
        ok_evecs = eg.Array2(np.ones((3, 1), dtype=np.float64))
        ok_evals = eg.Array1(np.array([0.5], dtype=np.float64))
        nan_evecs = eg.Array2(np.full((3, 1), float("nan"), dtype=np.float64))
        inf_evecs = eg.Array2(np.full((3, 1), float("inf"), dtype=np.float64))
        for evals, evecs in [
            (nan_evals, ok_evecs),
            (inf_evals, ok_evecs),
            (ok_evals, nan_evecs),
            (ok_evals, inf_evecs),
        ]:
            with self.assertRaises(ValueError):
                eg.LowRankDiffusionKernel(1.0, evals, evecs)
            with self.assertRaises(ValueError):
                eg.LowRankMultiscaleDiffusionKernel(0.5, evals, evecs)

    # R8: WeightedEdgeLength OOB & self-looped graph safety
    def test_r8_weighted_edge_length_oob_and_self_loop_no_panic(self):
        g3 = self._path3()
        wel = eg.WeightedEdgeLength(g3)
        with self.assertRaises(IndexError):
            wel(99)

        g_loop = eg.Graph.from_edges_numpy(
            3, np.array([[0, 0], [0, 1], [1, 2]], dtype=np.int64)
        )
        wel_loop = eg.WeightedEdgeLength(g_loop)
        self.assertEqual(wel_loop(0), 0)
        self.assertEqual(wel_loop(1), 3)
        self.assertEqual(wel_loop(2), 3)
        with self.assertRaises(IndexError):
            wel_loop(99)
        dm = eg.all_sources_dijkstra(g_loop, wel_loop)
        self.assertTrue(math.isfinite(dm.get_by_index(0, 2)))

        g_rem = self._path3()
        g_rem.remove_edge(0)
        wel_rem = eg.WeightedEdgeLength(g_rem)
        self.assertEqual(wel_rem(0), 2)
        with self.assertRaises(IndexError):
            wel_rem(1)


if __name__ == "__main__":
    unittest.main()
