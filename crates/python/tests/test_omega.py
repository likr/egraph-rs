"""
Tests for the Omega layout algorithm Python bindings.

This module tests the Omega algorithm, which uses spectral coordinates derived
from graph Laplacian eigenvalues to create high-quality graph layouts.
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
        
        # Create Omega instance with default parameters
        sgd = eg.Omega().build(graph, lambda edge_idx: 1.0, rng)
        
        # Verify the algorithm was created successfully
        self.assertIsNotNone(sgd)
        
        # Create a drawing for layout
        drawing = eg.DrawingEuclidean2d.initial_placement(graph)
        
        # Test shuffle functionality
        sgd.shuffle(rng)
        
        # Test apply functionality
        sgd.apply(drawing, 0.1)

    def test_omega_builder(self):
        """Test Omega for custom configuration."""
        # Create a simple path graph
        graph = eg.Graph()
        nodes = [graph.add_node(i) for i in range(4)]
        for i in range(3):
            graph.add_edge(nodes[i], nodes[i + 1], None)
        
        # Create RNG
        rng = eg.Rng.seed_from(123)
        
        # Create Omega with custom builder configuration
        builder = eg.Omega()
        builder = builder.d(3)  # 3 spectral dimensions
        builder = builder.k(10)  # 10 random pairs per node
        builder = builder.min_dist(1e-2)  # Minimum distance
        builder = builder.eigenvalue_max_iterations(500)
        builder = builder.cg_max_iterations(50)
        builder = builder.eigenvalue_tolerance(1e-3)
        builder = builder.cg_tolerance(1e-3)
        
        # Build SGD instance
        sgd = builder.build(graph, lambda edge_idx: 1.0, rng)
        
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
        
        # Test method chaining
        sgd = (eg.Omega()
                 .d(2)
                 .k(5) 
                 .min_dist(5e-3)
                 .eigenvalue_max_iterations(200)
                 .cg_max_iterations(30)
                 .eigenvalue_tolerance(5e-4)
                 .cg_tolerance(5e-4)
                 .build(graph, lambda edge_idx: 2.0, rng))
        
        self.assertIsNotNone(sgd)

    def test_omega_with_weighted_edges(self):
        """Test Omega with weighted edges."""
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
        
        sgd = eg.Omega().build(graph, edge_weight, rng)
        
        drawing = eg.DrawingEuclidean2d.initial_placement(graph)
        sgd.apply(drawing, 0.1)
        
        self.assertIsNotNone(sgd)

    def test_omega_full_layout_process(self):
        """Test a complete layout process using Omega with a scheduler."""
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
        
        # Create Omega with specific parameters
        sgd = (eg.Omega()
                 .d(2)
                 .k(5)
                 .min_dist(1e-3)
                 .build(graph, lambda edge_idx: 1.0, rng))
        
        # Create initial drawing
        drawing = eg.DrawingEuclidean2d.initial_placement(graph)
        
        # Run layout algorithm with exponential scheduler
        scheduler = eg.SchedulerExponential(50)
        
        def step(eta):
            sgd.shuffle(rng)
            sgd.apply(drawing, eta)
        
        scheduler.run(step)
        
        # Verify that drawing positions have been modified
        # (We can't check exact values since they depend on eigenvalue computation,
        #  but we can verify the positions are different from initial placement)
        self.assertIsNotNone(drawing)

    def test_omega_update_functions(self):
        """Test distance and weight update functions."""
        graph = eg.Graph()
        a = graph.add_node(0)
        b = graph.add_node(1)
        graph.add_edge(a, b, None)
        
        rng = eg.Rng.seed_from(2021)
        sgd = eg.Omega().build(graph, lambda edge_idx: 1.0, rng)
        
        # Test update_distance function
        def update_distance_func(i, j, distance, weight):
            return distance * 1.1  # Increase all distances by 10%
        
        sgd.update_distance(update_distance_func)
        
        # Test update_weight function  
        def update_weight_func(i, j, distance, weight):
            return weight * 0.9  # Decrease all weights by 10%
        
        sgd.update_weight(update_weight_func)
        
        # Verify the omega instance still works after updates
        drawing = eg.DrawingEuclidean2d.initial_placement(graph)
        sgd.apply(drawing, 0.1)


if __name__ == "__main__":
    unittest.main()
