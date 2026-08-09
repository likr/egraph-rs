import unittest
import egraph as eg


class TestDiffusionKernel(unittest.TestCase):
    def setUp(self):
        """Set up a simple graph for testing"""
        self.graph = eg.Graph()
        self.n0 = self.graph.add_node(0)
        self.n1 = self.graph.add_node(1)
        self.n2 = self.graph.add_node(2)
        self.graph.add_edge(self.n0, self.n1, None)
        self.graph.add_edge(self.n1, self.n2, None)

        self.rng = eg.Rng.seed_from(42)

    def test_basic_construction(self):
        """Test basic DiffusionKernel construction with StandardLaplacian"""
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        dk = eg.DiffusionKernel(
            laplacian,
            1000.0,  # t
            10,  # degree
            self.rng,
        )

        self.assertEqual(dk.n(), 3)

    def test_construction_with_lambda_max(self):
        """Test DiffusionKernel construction with external lambda_max"""
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        dk = eg.DiffusionKernel.new_with_lambda_max(
            laplacian,
            1000.0,  # t
            10,  # degree
            2.0,  # lambda_max
            self.rng,
        )

        self.assertEqual(dk.n(), 3)

    def test_element_access(self):
        """Test querying kernel matrix elements"""
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        dk = eg.DiffusionKernel(laplacian, 1000.0, 10, self.rng)

        k_00 = dk.get(0, 0)
        k_11 = dk.get(1, 1)
        k_22 = dk.get(2, 2)

        self.assertGreater(k_00, 0.0)
        self.assertGreater(k_11, 0.0)
        self.assertGreater(k_22, 0.0)

        k_01 = dk.get(0, 1)
        k_12 = dk.get(1, 2)

        self.assertGreater(k_01, 0.0)
        self.assertGreater(k_12, 0.0)

    def test_symmetry(self):
        """Test that K[i,j] == K[j,i] (symmetry)"""
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        dk = eg.DiffusionKernel(
            laplacian,
            1000.0,
            10,
            self.rng,
        )

        k_01 = dk.get(0, 1)
        k_10 = dk.get(1, 0)
        self.assertAlmostEqual(k_01, k_10, places=10)


class TestMultiscaleDiffusionKernel(unittest.TestCase):
    def setUp(self):
        self.graph = eg.Graph()
        self.n0 = self.graph.add_node(0)
        self.n1 = self.graph.add_node(1)
        self.n2 = self.graph.add_node(2)
        self.graph.add_edge(self.n0, self.n1, None)
        self.graph.add_edge(self.n1, self.n2, None)

    def test_basic_construction(self):
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        mk = eg.MultiscaleDiffusionKernel(laplacian, 0.85)
        self.assertEqual(mk.n(), 3)
        self.assertGreater(mk.get(0, 0), 0.0)


class TestLowRankDiffusionKernel(unittest.TestCase):
    def setUp(self):
        self.graph = eg.Graph()
        self.n0 = self.graph.add_node(0)
        self.n1 = self.graph.add_node(1)
        self.n2 = self.graph.add_node(2)
        self.graph.add_edge(self.n0, self.n1, None)
        self.graph.add_edge(self.n1, self.n2, None)

        self.rng = eg.Rng.seed_from(42)

    def test_basic_construction_and_properties(self):
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        import numpy as np
        evals = eg.Array1(np.array([0.0, 1.0], dtype=np.float64))
        evecs = eg.Array2(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64))
        lr = eg.LowRankDiffusionKernel(1.0, evals, evecs)
        self.assertEqual(lr.n(), 3)

    def test_element_access_and_symmetry(self):
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        import numpy as np
        evals = eg.Array1(np.array([0.0, 1.0], dtype=np.float64))
        evecs = eg.Array2(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64))
        lr = eg.LowRankDiffusionKernel(1.0, evals, evecs)

        k_00 = lr.get(0, 0)
        k_11 = lr.get(1, 1)
        self.assertGreater(k_00, 0.0)
        self.assertGreater(k_11, 0.0)

        k_01 = lr.get(0, 1)
        k_10 = lr.get(1, 0)
        self.assertAlmostEqual(k_01, k_10, places=10)

    def test_new_from_eigen(self):
        import numpy as np
        evals = eg.Array1(np.array([0.0, 1.0, 2.0], dtype=np.float64))
        evecs = eg.Array2(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64))
        lr = eg.LowRankDiffusionKernel(1.0, evals, evecs)
        self.assertEqual(lr.n(), 3)
        self.assertAlmostEqual(lr.get(0, 0), 1.3333333333333333, places=5)


class TestLowRankMultiscaleDiffusionKernel(unittest.TestCase):
    def setUp(self):
        self.graph = eg.Graph()
        self.n0 = self.graph.add_node(0)
        self.n1 = self.graph.add_node(1)
        self.n2 = self.graph.add_node(2)
        self.graph.add_edge(self.n0, self.n1, None)
        self.graph.add_edge(self.n1, self.n2, None)

        self.rng = eg.Rng.seed_from(42)

    def test_construction_and_from_eigen(self):
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        import numpy as np
        evals = eg.Array1(np.array([0.0, 1.0], dtype=np.float64))
        evecs = eg.Array2(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64))

        lrm = eg.LowRankMultiscaleDiffusionKernel(0.85, evals, evecs)
        self.assertEqual(lrm.n(), 3)

        import numpy as np
        evals = eg.Array1(np.array([0.0, 1.0, 2.0], dtype=np.float64))
        evecs = eg.Array2(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64))
        lrm_custom = eg.LowRankMultiscaleDiffusionKernel(0.85, evals, evecs)
        self.assertEqual(lrm_custom.n(), 3)


class TestPivotedKernelsAndDistances(unittest.TestCase):
    def setUp(self):
        self.graph = eg.Graph()
        self.n0 = self.graph.add_node(0)
        self.n1 = self.graph.add_node(1)
        self.n2 = self.graph.add_node(2)
        self.graph.add_edge(self.n0, self.n1, None)
        self.graph.add_edge(self.n1, self.n2, None)

        self.rng = eg.Rng.seed_from(42)

    def test_pivoted_diffusion_kernel(self):
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        pk = eg.PivotedDiffusionKernel(laplacian, 1.0, 10, [0, 1], self.rng)
        self.assertEqual(pk.pivots(), [0, 1])
        self.assertGreater(pk.get_from_pivot(0, 0), 0.0)

        neg_log_dist = eg.PivotedNegLogDistanceBuilder().alpha(1.0).beta(0.0).p(0.5).build(self.graph, pk)
        self.assertGreaterEqual(neg_log_dist.get(0, 1), 0.0)

    def test_pivoted_multiscale_diffusion_kernel(self):
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        pmk = eg.PivotedMultiscaleDiffusionKernel(laplacian, 0.85, [0])
        self.assertEqual(pmk.pivots(), [0])
        self.assertGreater(pmk.get_from_pivot(0, 0), 0.0)

        neg_log_dist = eg.PivotedNegLogDistanceBuilder().alpha(1.0).beta(0.0).p(0.5).build(self.graph, pmk)
        self.assertGreaterEqual(neg_log_dist.get(0, 1), 0.0)

    def test_sparse_sgd_with_pivoted_neg_log_distance(self):
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        pk = eg.PivotedDiffusionKernel(laplacian, 1.0, 10, [0, 1], self.rng)
        pnd = eg.PivotedNegLogDistanceBuilder().alpha(1.0).beta(0.0).p(0.5).build(self.graph, pk)

        sparse_sgd = eg.SparseSgd()
        sgd = sparse_sgd.build_with_pivoted_distance(self.graph, lambda i: 1.0, pnd)
        drawing = eg.DrawingEuclidean2d.initial_placement(self.graph)
        scheduler = eg.SchedulerConstant(10, 0.1)
        scheduler.run(lambda eta: sgd.apply(drawing, eta))

    def test_neg_log_distance_with_all_kernels(self):
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        
        # Test full DiffusionKernel
        dk = eg.DiffusionKernel(laplacian, 1.0, 10, self.rng)
        nld_dk = eg.NegLogDistanceBuilder().alpha(1.0).beta(0.0).p(0.5).build(self.graph, dk)
        self.assertGreaterEqual(nld_dk.get(0, 1), 0.0)

        # Test LowRankDiffusionKernel
        import numpy as np
        evals = eg.Array1(np.array([0.0, 1.0], dtype=np.float64))
        evecs = eg.Array2(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64))
        lr = eg.LowRankDiffusionKernel(1.0, evals, evecs)
        nld_lr = eg.NegLogDistanceBuilder().alpha(1.0).beta(0.0).p(0.5).build(self.graph, lr)
        self.assertGreaterEqual(nld_lr.get(0, 1), 0.0)

        # Test MultiscaleDiffusionKernel
        mk = eg.MultiscaleDiffusionKernel(laplacian, 0.85)
        nld_mk = eg.NegLogDistanceBuilder().alpha(1.0).beta(0.0).p(0.5).build(self.graph, mk)
        self.assertGreaterEqual(nld_mk.get(0, 1), 0.0)

        # Test LowRankMultiscaleDiffusionKernel
        lrm = eg.LowRankMultiscaleDiffusionKernel(0.85, evals, evecs)
        nld_lrm = eg.NegLogDistanceBuilder().alpha(1.0).beta(0.0).p(0.5).build(self.graph, lrm)
        self.assertGreaterEqual(nld_lrm.get(0, 1), 0.0)


if __name__ == "__main__":
    unittest.main()
