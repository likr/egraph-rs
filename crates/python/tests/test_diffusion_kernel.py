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
            50,  # num_vectors
            self.rng,
        )

        # Check that we can query the size
        self.assertEqual(dk.n(), 3)

    def test_construction_with_symmetric_normalized(self):
        """Test DiffusionKernel construction with SymmetricNormalizedLaplacian"""
        laplacian = eg.SymmetricNormalizedLaplacian.build(self.graph, lambda i: 1.0)
        dk = eg.DiffusionKernel(
            laplacian,
            1000.0,  # t
            10,  # degree
            50,  # num_vectors
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
            50,  # num_vectors
            self.rng,
        )

        self.assertEqual(dk.n(), 3)

    def test_element_access(self):
        """Test querying kernel matrix elements"""
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        dk = eg.DiffusionKernel(laplacian, 1000.0, 10, 50, self.rng)

        # Get diagonal elements
        k_00 = dk.get(0, 0)
        k_11 = dk.get(1, 1)
        k_22 = dk.get(2, 2)

        # Diagonal elements should be positive
        self.assertGreater(k_00, 0.0)
        self.assertGreater(k_11, 0.0)
        self.assertGreater(k_22, 0.0)

        # Get off-diagonal elements
        k_01 = dk.get(0, 1)
        k_12 = dk.get(1, 2)

        # Off-diagonal elements should be positive for connected graph
        self.assertGreater(k_01, 0.0)
        self.assertGreater(k_12, 0.0)

    def test_symmetry(self):
        """Test that K[i,j] == K[j,i] (symmetry)"""
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        dk = eg.DiffusionKernel(
            laplacian,
            1000.0,
            10,
            100,  # Use more vectors for better accuracy
            self.rng,
        )

        # Check symmetry for several pairs
        k_01 = dk.get(0, 1)
        k_10 = dk.get(1, 0)
        self.assertAlmostEqual(k_01, k_10, places=10)

        k_12 = dk.get(1, 2)
        k_21 = dk.get(2, 1)
        self.assertAlmostEqual(k_12, k_21, places=10)

    def test_custom_parameters(self):
        """Test with custom diffusion parameters"""
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        dk = eg.DiffusionKernel(
            laplacian,
            500.0,  # Different t
            20,  # Different degree
            100,  # Different num_vectors
            self.rng,
        )

        self.assertEqual(dk.n(), 3)

        # Should still get positive values
        k_00 = dk.get(0, 0)
        self.assertGreater(k_00, 0.0)

    def test_weighted_edges(self):
        """Test with weighted edges"""
        # Create graph with weights
        graph = eg.Graph()
        n0 = graph.add_node(0)
        n1 = graph.add_node(1)
        n2 = graph.add_node(2)
        e0 = graph.add_edge(n0, n1, None)
        e1 = graph.add_edge(n1, n2, None)

        rng = eg.Rng.seed_from(42)

        # Use different weights
        edge_weights = {0: 1.0, 1: 2.0}

        laplacian = eg.StandardLaplacian.build(
            graph, lambda i: edge_weights.get(i, 1.0)
        )
        dk = eg.DiffusionKernel(laplacian, 1000.0, 10, 50, rng)

        self.assertEqual(dk.n(), 3)
        k_01 = dk.get(0, 1)
        self.assertGreater(k_01, 0.0)

    def test_distance_computation(self):
        """Test computing distances from kernel elements"""
        import math

        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        dk = eg.DiffusionKernel(laplacian, 1000.0, 10, 50, self.rng)

        # Compute distance using kernel elements
        # distance(i,j) = sqrt(K[i,i] + K[j,j] - 2*K[i,j])
        k_00 = dk.get(0, 0)
        k_11 = dk.get(1, 1)
        k_01 = dk.get(0, 1)

        distance = math.sqrt(max(0.0, k_00 + k_11 - 2.0 * k_01))

        # Distance should be non-negative
        self.assertGreaterEqual(distance, 0.0)

    def test_single_source_and_pivot_distance(self):
        """Test single_source_heat_vector and pivot_distance_vector"""
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        heat_vec = eg.DiffusionKernel.single_source_heat_vector(laplacian, 0.5, 10, 0)
        self.assertEqual(len(heat_vec), 3)
        self.assertGreater(heat_vec[0], 0.0)

        dk = eg.DiffusionKernel(laplacian, 0.5, 10, 50, self.rng)
        dist_vec = dk.pivot_distance_vector(laplacian, 0.5, 10, 0)
        self.assertEqual(len(dist_vec), 3)
        self.assertEqual(dist_vec[0], 0.0)
        self.assertGreater(dist_vec[1], 0.0)

    def test_pivot_diffusion_sgd(self):
        """Test PivotDiffusionSgd builder"""
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        sgd = eg.PivotDiffusionSgd().t(0.5).degree(10).num_vectors(16).h(2).build(self.graph, laplacian, lambda i: 1.0, self.rng)
        self.assertIsNotNone(sgd)


class TestLowRankDiffusionKernel(unittest.TestCase):
    def setUp(self):
        """Set up a simple graph for testing LowRankDiffusionKernel"""
        self.graph = eg.Graph()
        self.n0 = self.graph.add_node(0)
        self.n1 = self.graph.add_node(1)
        self.n2 = self.graph.add_node(2)
        self.graph.add_edge(self.n0, self.n1, None)
        self.graph.add_edge(self.n1, self.n2, None)

        self.rng = eg.Rng.seed_from(42)

    def test_basic_construction_and_properties(self):
        """Test LowRankDiffusionKernel basic construction and properties"""
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        lr = eg.LowRankDiffusionKernel(laplacian, 1.0, 2, self.rng)

        self.assertEqual(lr.n(), 3)
        self.assertEqual(lr.t(), 1.0)
        self.assertEqual(lr.rank(), 2)
        self.assertEqual(lr.eta(), 0.0)

        eigenvalues = lr.eigenvalues()
        self.assertGreaterEqual(len(eigenvalues), 2)
        self.assertGreaterEqual(eigenvalues[0], 0.0)

    def test_eta_parameter(self):
        """Test LowRankDiffusionKernel with custom eta parameter"""
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        eta = 0.05
        lr = eg.LowRankDiffusionKernel(laplacian, 1.0, 2, self.rng, eta)
        self.assertEqual(lr.eta(), eta)

        import math
        k_00 = lr.get(0, 0)
        k_11 = lr.get(1, 1)
        k_01 = lr.get(0, 1)

        ratio = (k_01 + eta) / math.sqrt((k_00 + eta) * (k_11 + eta))
        expected_dist = math.sqrt(max(0.0, -4.0 * math.log(max(1e-15, min(1.0, ratio)))))
        self.assertAlmostEqual(lr.distance(0, 1), expected_dist, places=10)

    def test_element_access_and_symmetry(self):
        """Test LowRankDiffusionKernel matrix element queries and symmetry"""
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        lr = eg.LowRankDiffusionKernel(laplacian, 1.0, 2, self.rng)

        k_00 = lr.get(0, 0)
        k_11 = lr.get(1, 1)
        self.assertGreater(k_00, 0.0)
        self.assertGreater(k_11, 0.0)

        k_01 = lr.get(0, 1)
        k_10 = lr.get(1, 0)
        self.assertAlmostEqual(k_01, k_10, places=10)

    def test_distance_and_pivot_vector(self):
        """Test distance and pivot_distance_vector queries"""
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        lr = eg.LowRankDiffusionKernel(laplacian, 1.0, 2, self.rng)

        self.assertEqual(lr.distance(0, 0), 0.0)
        self.assertGreater(lr.distance(0, 1), 0.0)

        dist_vec = lr.pivot_distance_vector(0)
        self.assertEqual(len(dist_vec), 3)
        self.assertEqual(dist_vec[0], 0.0)
        self.assertGreater(dist_vec[1], 0.0)

    def test_low_rank_diffusion_distance_matrix(self):
        """Test HeatGeodesicDistanceMatrix wrapping and querying"""
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        lr = eg.LowRankDiffusionKernel(laplacian, 1.0, 2, self.rng)

        dm = eg.HeatGeodesicDistanceMatrix(self.graph, lr, 0.1)
        self.assertEqual(dm.get(0, 0), 0.0)
        self.assertGreaterEqual(dm.get(0, 1), 0.1)

        dm_pivots = eg.HeatGeodesicDistanceMatrix.new_with_pivots(self.graph, lr, [0], 0.1)
        self.assertEqual(dm_pivots.get(0, 0), 0.0)
        self.assertGreaterEqual(dm_pivots.get(0, 1), 0.1)


class TestExactDiffusionKernel(unittest.TestCase):
    def setUp(self):
        """Set up a simple graph for testing ExactDiffusionKernel"""
        self.graph = eg.Graph()
        self.n0 = self.graph.add_node(0)
        self.n1 = self.graph.add_node(1)
        self.n2 = self.graph.add_node(2)
        self.graph.add_edge(self.n0, self.n1, None)
        self.graph.add_edge(self.n1, self.n2, None)

    def test_basic_construction_and_properties(self):
        """Test ExactDiffusionKernel basic construction and properties"""
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        exact = eg.ExactDiffusionKernel(laplacian, 1.0, 20)

        self.assertEqual(exact.n(), 3)
        self.assertEqual(exact.t(), 1.0)

        self.assertGreater(exact.get(0, 0), 0.0)
        self.assertAlmostEqual(exact.get(0, 1), exact.get(1, 0), places=10)

        self.assertEqual(exact.distance(0, 0), 0.0)
        self.assertGreater(exact.distance(0, 1), 0.0)

        dist_vec = exact.pivot_distance_vector(0)
        self.assertEqual(len(dist_vec), 3)
        self.assertEqual(dist_vec[0], 0.0)
        self.assertGreater(dist_vec[1], 0.0)

    def test_heat_geodesic_distance_matrix(self):
        """Test HeatGeodesicDistanceMatrix with ExactDiffusionKernel"""
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        exact = eg.ExactDiffusionKernel(laplacian, 1.0, 20)

        dm = eg.HeatGeodesicDistanceMatrix(self.graph, exact, 0.1)
        self.assertEqual(dm.get(0, 0), 0.0)
        self.assertGreaterEqual(dm.get(0, 1), 0.1)


if __name__ == "__main__":
    unittest.main()

