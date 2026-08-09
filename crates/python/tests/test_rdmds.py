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
        result = rdmds.eigendecomposition(self.graph, lambda i: 1.0, eg.Ic0CgSolver(), self.rng)

        # Check that embedding has correct shape (3 nodes, 2 dimensions)
        self.assertEqual(result.eigenvectors.shape, (3, 2))

    def test_rdmds_custom_dimensions(self):
        """Test RdMds with custom number of dimensions"""
        rdmds = eg.RdMds().d(3)
        result = rdmds.eigendecomposition(self.graph, lambda i: 1.0, eg.Ic0CgSolver(), self.rng)

        # Check that embedding has correct shape (3 nodes, 3 dimensions)
        self.assertEqual(result.eigenvectors.shape, (3, 3))

    def test_rdmds_method_chaining(self):
        """Test RdMds builder pattern with method chaining"""
        rdmds = (
            eg.RdMds()
            .d(2)
            .shift(1e-3)
            .eigenvalue_max_iterations(500)
            .eigenvalue_tolerance(1e-2)
        )

        result = rdmds.eigendecomposition(self.graph, lambda i: 1.0, eg.Ic0CgSolver(), self.rng)
        self.assertEqual(result.eigenvectors.shape, (3, 2))

    def test_rdmds_eigendecomposition(self):
        """Test RdMds eigendecomposition method"""
        rdmds = eg.RdMds().d(2)
        result = rdmds.eigendecomposition(
            self.graph, lambda i: 1.0, eg.Ic0CgSolver(), self.rng
        )

        self.assertEqual(result.eigenvectors.shape, (3, 2))
        self.assertEqual(len(result.eigenvalues), 2)
        self.assertEqual(len(result.cg_iterations), 2)
        self.assertEqual(len(result.power_iterations), 2)

    def test_rdmds_with_weighted_edges(self):
        """Test RdMds with weighted edges"""
        edge_weights = {0: 1.0, 1: 5.0, 2: 10.0}
        rdmds = eg.RdMds()
        result = rdmds.eigendecomposition(self.graph, lambda i: edge_weights[i], eg.Ic0CgSolver(), self.rng)
        
        self.assertEqual(result.eigenvectors.shape, (3, 2))


class TestRdMdsOmegaIntegration(unittest.TestCase):
    """Test integration between RdMds and Omega algorithms"""

    def setUp(self):
        self.graph = eg.Graph()
        self.nodes = [self.graph.add_node(i) for i in range(5)]
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        for u, v in edges:
            self.graph.add_edge(self.nodes[u], self.nodes[v], 1.0)
        self.rng = eg.Rng.seed_from(42)

    def test_rdmds_omega_workflow(self):
        """Test complete workflow: RdMds -> Omega -> SGD"""
        rdmds = eg.RdMds().d(2)
        result = rdmds.eigendecomposition(self.graph, lambda i: 1.0, eg.Ic0CgSolver(), self.rng)
        embedding = result.eigenvectors

        omega = eg.Omega()
        sgd = omega.build(self.graph, embedding, self.rng)
        
        drawing = eg.DrawingEuclidean2d.initial_placement(self.graph)
        sgd.apply(drawing, 0.1)

    def test_omega_with_custom_parameters(self):
        """Test Omega with custom k and min_dist parameters"""
        rdmds = eg.RdMds().d(3)
        result = rdmds.eigendecomposition(self.graph, lambda i: 1.0, eg.Ic0CgSolver(), self.rng)
        embedding = result.eigenvectors

        omega = (
            eg.Omega()
            .k(2)
            .min_dist(0.5)
        )
        sgd = omega.build(self.graph, embedding, self.rng)
        
        drawing = eg.DrawingEuclidean2d.initial_placement(self.graph)
        sgd.apply(drawing, 0.1)

    def test_rdmds_reusable_embedding(self):
        """Test that RdMds embedding can be reused for multiple Omega instances"""
        rdmds = eg.RdMds().d(2)
        result = rdmds.eigendecomposition(self.graph, lambda i: 1.0, eg.Ic0CgSolver(), self.rng)
        embedding = result.eigenvectors

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
        result = rdmds.eigendecomposition(self.graph, lambda i: 1.0, eg.Ic0CgSolver(), self.rng)
        embedding = result.eigenvectors

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
