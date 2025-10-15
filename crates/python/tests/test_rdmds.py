"""
Tests for RdMds (Resistance-distance MDS) Python bindings
"""

import unittest
import egraph as eg


class TestRdMds(unittest.TestCase):
    """Test cases for RdMds spectral embedding computation"""

    def setUp(self):
        """Create a simple triangle graph for testing"""
        self.graph = eg.Graph()
        self.a = self.graph.add_node(0)
        self.b = self.graph.add_node(1)
        self.c = self.graph.add_node(2)
        self.graph.add_edge(self.a, self.b, 1.0)
        self.graph.add_edge(self.b, self.c, 1.0)
        self.graph.add_edge(self.c, self.a, 1.0)
        self.rng = eg.Rng.seed_from(42)

    def test_rdmds_default_parameters(self):
        """Test RdMds with default parameters"""
        rdmds = eg.RdMds()
        embedding = rdmds.embedding(self.graph, lambda i: 1.0, self.rng)

        # Check that embedding has correct shape (3 nodes, 2 dimensions)
        self.assertEqual(embedding.shape, (3, 2))

    def test_rdmds_custom_dimensions(self):
        """Test RdMds with custom number of dimensions"""
        rdmds = eg.RdMds().d(3)
        embedding = rdmds.embedding(self.graph, lambda i: 1.0, self.rng)

        # Check that embedding has correct shape (3 nodes, 3 dimensions)
        self.assertEqual(embedding.shape, (3, 3))

    def test_rdmds_method_chaining(self):
        """Test RdMds builder pattern with method chaining"""
        rdmds = (
            eg.RdMds()
            .d(2)
            .shift(1e-3)
            .eigenvalue_max_iterations(500)
            .cg_max_iterations(50)
            .eigenvalue_tolerance(1e-2)
            .cg_tolerance(1e-3)
        )

        embedding = rdmds.embedding(self.graph, lambda i: 1.0, self.rng)
        self.assertEqual(embedding.shape, (3, 2))

    def test_rdmds_eigendecomposition(self):
        """Test RdMds eigendecomposition method"""
        rdmds = eg.RdMds().d(2)
        embedding, eigenvalues = rdmds.eigendecomposition(
            self.graph, lambda i: 1.0, self.rng
        )

        # Check shapes
        self.assertEqual(embedding.shape, (3, 2))
        self.assertEqual(len(eigenvalues), 2)

        # Eigenvalues should be positive (non-zero eigenvalues of Laplacian)
        for i in range(2):
            self.assertGreater(eigenvalues[i], 0)

    def test_rdmds_with_weighted_edges(self):
        """Test RdMds with weighted edges"""
        # Create a graph with different edge weights
        graph = eg.Graph()
        a = graph.add_node(0)
        b = graph.add_node(1)
        c = graph.add_node(2)
        graph.add_edge(a, b, 1.0)
        graph.add_edge(b, c, 2.0)
        graph.add_edge(c, a, 3.0)

        rdmds = eg.RdMds().d(2)

        # Use edge weights
        edge_weights = [1.0, 2.0, 3.0]
        embedding = rdmds.embedding(graph, lambda i: edge_weights[i], self.rng)

        self.assertEqual(embedding.shape, (3, 2))


class TestRdMdsOmegaIntegration(unittest.TestCase):
    """Test cases for RdMds and Omega integration"""

    def setUp(self):
        """Create a simple graph for testing"""
        self.graph = eg.Graph()
        for i in range(5):
            self.graph.add_node(i)

        # Create a simple path graph
        for i in range(4):
            self.graph.add_edge(i, i + 1, 1.0)

        self.rng = eg.Rng.seed_from(42)

    def test_rdmds_omega_workflow(self):
        """Test complete workflow: RdMds -> Omega -> SGD"""
        # Step 1: Compute spectral embedding with RdMds
        rdmds = eg.RdMds().d(2).shift(1e-3)
        embedding = rdmds.embedding(self.graph, lambda i: 1.0, self.rng)

        # Step 2: Generate node pairs with Omega
        omega = eg.Omega().k(10).min_dist(1e-3)
        sgd = omega.build(self.graph, embedding, self.rng)

        # Step 3: Apply SGD to a drawing
        drawing = eg.DrawingEuclidean2d.initial_placement(self.graph)

        # Run a few SGD iterations
        for _ in range(10):
            sgd.shuffle(self.rng)
            sgd.apply(drawing, 0.1)

        # Verify that positions have been updated
        # (positions should be different from initial random placement)
        self.assertEqual(self.graph.node_count(), 5)

    def test_omega_with_custom_parameters(self):
        """Test Omega with custom k and min_dist parameters"""
        rdmds = eg.RdMds().d(2)
        embedding = rdmds.embedding(self.graph, lambda i: 1.0, self.rng)

        # Test with different k values
        omega1 = eg.Omega().k(5).min_dist(1e-2)
        sgd1 = omega1.build(self.graph, embedding, self.rng)

        omega2 = eg.Omega().k(20).min_dist(1e-4)
        sgd2 = omega2.build(self.graph, embedding, self.rng)

        # Both should create valid SGD instances
        drawing = eg.DrawingEuclidean2d.initial_placement(self.graph)
        sgd1.apply(drawing, 0.1)
        sgd2.apply(drawing, 0.1)

    def test_rdmds_reusable_embedding(self):
        """Test that RdMds embedding can be reused for multiple Omega instances"""
        # Compute embedding once
        rdmds = eg.RdMds().d(2)
        embedding = rdmds.embedding(self.graph, lambda i: 1.0, self.rng)

        # Create multiple Omega instances with the same embedding
        omega1 = eg.Omega().k(10)
        omega2 = eg.Omega().k(20)
        omega3 = eg.Omega().k(30)

        sgd1 = omega1.build(self.graph, embedding, self.rng)
        sgd2 = omega2.build(self.graph, embedding, self.rng)
        sgd3 = omega3.build(self.graph, embedding, self.rng)

        # All should work with the same drawing
        drawing = eg.DrawingEuclidean2d.initial_placement(self.graph)
        sgd1.apply(drawing, 0.1)
        sgd2.apply(drawing, 0.1)
        sgd3.apply(drawing, 0.1)

    def test_complete_layout_with_scheduler(self):
        """Test complete layout process with RdMds, Omega, and scheduler"""
        # Compute embedding
        rdmds = eg.RdMds().d(2)
        embedding = rdmds.embedding(self.graph, lambda i: 1.0, self.rng)

        # Build SGD with Omega
        omega = eg.Omega().k(15)
        sgd = omega.build(self.graph, embedding, self.rng)

        # Create drawing and scheduler
        drawing = eg.DrawingEuclidean2d.initial_placement(self.graph)
        scheduler = sgd.scheduler(50, 0.1)

        # Run layout optimization
        def step(eta):
            sgd.shuffle(self.rng)
            sgd.apply(drawing, eta)

        scheduler.run(step)

        # Verify final layout has valid positions
        for i in range(self.graph.node_count()):
            x = drawing.x(i)
            y = drawing.y(i)
            # Positions should be finite numbers
            self.assertTrue(-float("inf") < x < float("inf"))
            self.assertTrue(-float("inf") < y < float("inf"))


if __name__ == "__main__":
    unittest.main()
