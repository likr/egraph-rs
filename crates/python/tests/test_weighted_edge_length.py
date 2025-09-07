import unittest
from egraph import Graph, WeightedEdgeLength


class TestWeightedEdgeLength(unittest.TestCase):
    """Test WeightedEdgeLength implementation."""

    def test_simple_graph(self):
        """Test WeightedEdgeLength with a simple graph."""
        graph = Graph()

        # Add nodes
        n0 = graph.add_node(0)
        n1 = graph.add_node(1)
        n2 = graph.add_node(2)

        # Add edges
        e01 = graph.add_edge(n0, n1, ())
        e12 = graph.add_edge(n1, n2, ())

        # Create WeightedEdgeLength calculator
        weight_calc = WeightedEdgeLength(graph)

        # Test edge weights
        # Edge 0: between n0 and n1 (degrees 1 and 2, no common neighbors)
        # weight = 1 + 2 - 2 * 0 = 3
        self.assertEqual(weight_calc(0), 3)

        # Edge 1: between n1 and n2 (degrees 2 and 1, no common neighbors)
        # weight = 2 + 1 - 2 * 0 = 3
        self.assertEqual(weight_calc(1), 3)

    def test_triangle_graph(self):
        """Test WeightedEdgeLength with a triangle graph."""
        graph = Graph()

        # Add nodes
        n0 = graph.add_node(0)
        n1 = graph.add_node(1)
        n2 = graph.add_node(2)

        # Add edges to form a triangle
        e01 = graph.add_edge(n0, n1, ())
        e12 = graph.add_edge(n1, n2, ())
        e02 = graph.add_edge(n0, n2, ())

        # Create WeightedEdgeLength calculator
        weight_calc = WeightedEdgeLength(graph)

        # All edges in triangle have the same weight
        # Each node has degree 2, each edge has 1 common neighbor
        # weight = 2 + 2 - 2 * 1 = 2
        self.assertEqual(weight_calc(0), 2)  # n0-n1
        self.assertEqual(weight_calc(1), 2)  # n0-n2
        self.assertEqual(weight_calc(2), 2)  # n1-n2

    def test_usage_with_sgd(self):
        """Test that WeightedEdgeLength can be used with SGD like the original Python implementation."""
        import egraph as eg

        graph = Graph()

        # Create a small test graph
        n0 = graph.add_node(0)
        n1 = graph.add_node(1)
        n2 = graph.add_node(2)
        n3 = graph.add_node(3)

        graph.add_edge(n0, n1, ())
        graph.add_edge(n1, n2, ())
        graph.add_edge(n2, n3, ())
        graph.add_edge(n3, n0, ())

        # This should work like the original Python usage:
        # sgd = eg.FullSgd().build(graph, WeightedEdgeLength(graph))
        weight_calc = WeightedEdgeLength(graph)
        sgd = eg.FullSgd().build(graph, weight_calc)

        # Basic check that SGD was created successfully
        self.assertIsNotNone(sgd)

    def test_callable_interface(self):
        """Test that WeightedEdgeLength works as a callable object."""
        graph = Graph()

        # Create a path graph: 0-1-2
        n0 = graph.add_node(0)
        n1 = graph.add_node(1)
        n2 = graph.add_node(2)

        graph.add_edge(n0, n1, ())
        graph.add_edge(n1, n2, ())

        weight_calc = WeightedEdgeLength(graph)

        # Test that it can be called directly like a function
        self.assertTrue(callable(weight_calc))
        self.assertEqual(weight_calc(0), 3)
        self.assertEqual(weight_calc(1), 3)


if __name__ == '__main__':
    unittest.main()
