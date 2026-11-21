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


if __name__ == "__main__":
    unittest.main()
