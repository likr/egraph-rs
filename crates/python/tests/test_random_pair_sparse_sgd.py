"""
Tests for the RandomPairSparseSgd layout algorithm Python bindings.
"""

import unittest
import egraph as eg


class TestRandomPairSparseSgd(unittest.TestCase):
    """Test cases for the RandomPairSparseSgd layout algorithm."""

    def test_random_pair_sparse_sgd_basic(self):
        """Test basic RandomPairSparseSgd functionality."""
        graph = eg.Graph()
        a = graph.add_node(0)
        b = graph.add_node(1)
        c = graph.add_node(2)
        graph.add_edge(a, b, None)
        graph.add_edge(b, c, None)

        rng = eg.Rng.seed_from(42)

        # Compute distance matrix
        distance_matrix = eg.all_sources_dijkstra(graph, lambda edge_idx: 1.0)

        # Build SGD instance with RandomPairSparseSgd
        sgd = eg.RandomPairSparseSgd().k(5).build(graph, distance_matrix, rng)

        self.assertIsNotNone(sgd)

        # Create drawing and apply forces
        drawing = eg.DrawingEuclidean2d.initial_placement(graph)
        sgd.shuffle(rng)
        sgd.apply(drawing, 0.1)

    def test_random_pair_sparse_sgd_with_embedding(self):
        """Test RandomPairSparseSgd with EmbeddingDistanceMatrix."""
        graph = eg.Graph()
        a = graph.add_node(0)
        b = graph.add_node(1)
        c = graph.add_node(2)
        graph.add_edge(a, b, None)
        graph.add_edge(b, c, None)

        rng = eg.Rng.seed_from(123)

        # Compute spectral embedding with RdMds
        rdmds = eg.RdMds().d(2)
        embedding = rdmds.embedding(graph, lambda edge_idx: 1.0, rng)

        # Create EmbeddingDistanceMatrix
        distance_matrix = eg.EmbeddingDistanceMatrix(graph, embedding, 1e-3)

        # Build SGD
        sgd = eg.RandomPairSparseSgd().k(10).build(graph, distance_matrix, rng)

        self.assertIsNotNone(sgd)

        drawing = eg.DrawingEuclidean2d.initial_placement(graph)
        sgd.shuffle(rng)
        sgd.apply(drawing, 0.1)


if __name__ == "__main__":
    unittest.main()
