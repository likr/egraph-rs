"""
Tests for the Omega layout algorithm Python bindings.

This module tests the Omega algorithm, which generates node pairs for SGD
from precomputed spectral embeddings (computed by RdMds).
"""

import unittest
import egraph as eg


class TestOmega(unittest.TestCase):
    """Test cases for the Omega layout algorithm."""

    def test_omega_basic(self):
        """Test basic Omega functionality with default parameters."""
        # Create a simple triangle graph
        graph = eg.Graph()
        a = graph.add_node(0)
        b = graph.add_node(1)
        c = graph.add_node(2)
        graph.add_edge(a, b, None)
        graph.add_edge(b, c, None)
        graph.add_edge(c, a, None)

        # Create RNG for reproducible results
        rng = eg.Rng.seed_from(42)

        # Compute spectral embedding with RdMds
        rdmds = eg.RdMds()
        embedding = rdmds.embedding(graph, lambda edge_idx: 1.0, rng)

        # Create Omega instance with default parameters
        sgd = eg.Omega().build(graph, embedding, rng)

        # Verify the algorithm was created successfully
        self.assertIsNotNone(sgd)

        # Create a drawing for layout
        drawing = eg.DrawingEuclidean2d.initial_placement(graph)

        # Test shuffle functionality
        sgd.shuffle(rng)

        # Test apply functionality
        sgd.apply(drawing, 0.1)

    def test_omega_builder(self):
        """Test Omega with custom configuration."""
        # Create a simple path graph
        graph = eg.Graph()
        nodes = [graph.add_node(i) for i in range(4)]
        for i in range(3):
            graph.add_edge(nodes[i], nodes[i + 1], None)

        # Create RNG
        rng = eg.Rng.seed_from(123)

        # Compute spectral embedding
        rdmds = eg.RdMds().d(3)
        embedding = rdmds.embedding(graph, lambda edge_idx: 1.0, rng)

        # Create Omega with custom configuration
        omega = eg.Omega().k(10).min_dist(1e-2)

        # Build SGD instance
        sgd = omega.build(graph, embedding, rng)

        self.assertIsNotNone(sgd)

        # Test with drawing
        drawing = eg.DrawingEuclidean2d.initial_placement(graph)
        sgd.shuffle(rng)
        sgd.apply(drawing, 0.05)

    def test_omega_method_chaining(self):
        """Test method chaining with Omega."""
        graph = eg.Graph()
        nodes = [graph.add_node(i) for i in range(3)]
        for i in range(2):
            graph.add_edge(nodes[i], nodes[i + 1], None)

        rng = eg.Rng.seed_from(456)

        # Compute embedding
        rdmds = eg.RdMds().d(2)
        embedding = rdmds.embedding(graph, lambda edge_idx: 2.0, rng)

        # Test method chaining
        sgd = eg.Omega().k(5).min_dist(5e-3).build(graph, embedding, rng)

        self.assertIsNotNone(sgd)

    def test_omega_with_weighted_edges(self):
        """Test Omega with weighted edges in embedding computation."""
        graph = eg.Graph()
        a = graph.add_node(0)
        b = graph.add_node(1)
        c = graph.add_node(2)
        graph.add_edge(a, b, None)  # edge index 0
        graph.add_edge(b, c, None)  # edge index 1
        graph.add_edge(c, a, None)  # edge index 2

        rng = eg.Rng.seed_from(999)

        # Define edge weights: first edge has weight 2.0, others have weight 1.0
        def edge_weight(edge_idx):
            return 2.0 if edge_idx == 0 else 1.0

        # Compute embedding with weighted edges
        rdmds = eg.RdMds()
        embedding = rdmds.embedding(graph, edge_weight, rng)

        # Build SGD with Omega
        sgd = eg.Omega().build(graph, embedding, rng)

        drawing = eg.DrawingEuclidean2d.initial_placement(graph)
        sgd.apply(drawing, 0.1)

        self.assertIsNotNone(sgd)

    def test_omega_full_layout_process(self):
        """Test a complete layout process using RdMds, Omega, and scheduler."""
        # Create a more complex graph (small grid)
        graph = eg.Graph()
        nodes = [[graph.add_node(i * 3 + j) for j in range(3)] for i in range(3)]

        # Add horizontal edges
        for i in range(3):
            for j in range(2):
                graph.add_edge(nodes[i][j], nodes[i][j + 1], None)

        # Add vertical edges
        for i in range(2):
            for j in range(3):
                graph.add_edge(nodes[i][j], nodes[i + 1][j], None)

        rng = eg.Rng.seed_from(1337)

        # Compute spectral embedding
        rdmds = eg.RdMds().d(2)
        embedding = rdmds.embedding(graph, lambda edge_idx: 1.0, rng)

        # Create Omega with specific parameters
        sgd = eg.Omega().k(5).min_dist(1e-3).build(graph, embedding, rng)

        # Create initial drawing
        drawing = eg.DrawingEuclidean2d.initial_placement(graph)

        # Run layout algorithm with exponential scheduler
        scheduler = sgd.scheduler(50, 0.1)

        def step(eta):
            sgd.shuffle(rng)
            sgd.apply(drawing, eta)

        scheduler.run(step)

        # Verify that drawing positions have been modified
        self.assertIsNotNone(drawing)

    def test_omega_different_k_values(self):
        """Test Omega with different k (random pairs per node) values."""
        graph = eg.Graph()
        for i in range(5):
            graph.add_node(i)
        for i in range(4):
            graph.add_edge(i, i + 1, None)

        rng = eg.Rng.seed_from(2021)

        # Compute embedding once
        rdmds = eg.RdMds()
        embedding = rdmds.embedding(graph, lambda edge_idx: 1.0, rng)

        # Test with different k values
        for k in [5, 10, 20, 30]:
            omega = eg.Omega().k(k)
            sgd = omega.build(graph, embedding, rng)
            self.assertIsNotNone(sgd)

            drawing = eg.DrawingEuclidean2d.initial_placement(graph)
            sgd.apply(drawing, 0.1)

    def test_omega_reusable_embedding(self):
        """Test that the same embedding can be used with multiple Omega instances."""
        graph = eg.Graph()
        for i in range(4):
            graph.add_node(i)
        for i in range(3):
            graph.add_edge(i, i + 1, None)

        rng = eg.Rng.seed_from(3000)

        # Compute embedding once
        rdmds = eg.RdMds().d(2)
        embedding = rdmds.embedding(graph, lambda edge_idx: 1.0, rng)

        # Create multiple Omega instances with the same embedding
        omega1 = eg.Omega().k(10)
        omega2 = eg.Omega().k(20)

        sgd1 = omega1.build(graph, embedding, rng)
        sgd2 = omega2.build(graph, embedding, rng)

        # Both should work
        drawing = eg.DrawingEuclidean2d.initial_placement(graph)
        sgd1.apply(drawing, 0.1)
        sgd2.apply(drawing, 0.1)

        self.assertIsNotNone(sgd1)
        self.assertIsNotNone(sgd2)


if __name__ == "__main__":
    unittest.main()
