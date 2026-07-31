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


if __name__ == "__main__":
    unittest.main()
