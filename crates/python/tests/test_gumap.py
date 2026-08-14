import math
import unittest
import networkx as nx
import egraph as eg


def draw(nx_graph):
    graph = eg.Graph()
    indices = {}
    for u in nx_graph.nodes:
        indices[u] = graph.add_node(u)
    for u, v in nx_graph.edges:
        graph.add_edge(indices[u], indices[v], (u, v))
    return graph


class TestGuMap(unittest.TestCase):
    def test_gumap_builder_defaults(self):
        builder = eg.GuMapBuilder()
        gumap = builder.build()
        self.assertIsNotNone(gumap)

    def test_gumap_run(self):
        nx_graph = nx.cycle_graph(6)
        graph = draw(nx_graph)

        drawing = eg.DrawingEuclidean2d.initial_placement(graph)
        distance_matrix = eg.all_sources_dijkstra(graph, lambda _: 1.0)
        rng = eg.Rng.seed_from(42)

        # Initialize GUMAP with builder
        gumap = (
            eg.GuMapBuilder()
            .k(3)
            .min_dist(0.1)
            .spread(1.0)
            .n_epochs(50)
            .negative_sample_rate(5)
            .learning_rate(1.0)
            .build()
        )

        gumap.run(drawing, distance_matrix, rng)

        for u in graph.node_indices():
            self.assertTrue(math.isfinite(drawing.x(u)))
            self.assertTrue(math.isfinite(drawing.y(u)))

    def test_gumap_via_default_constructor(self):
        nx_graph = nx.cycle_graph(4)
        graph = draw(nx_graph)

        drawing = eg.DrawingEuclidean2d.initial_placement(graph)
        distance_matrix = eg.all_sources_dijkstra(graph, lambda _: 1.0)
        rng = eg.Rng.seed_from(42)

        gumap = eg.GuMap()
        gumap.run(drawing, distance_matrix, rng)

        for u in graph.node_indices():
            self.assertTrue(math.isfinite(drawing.x(u)))
            self.assertTrue(math.isfinite(drawing.y(u)))

    def test_gumap_validation_error(self):
        builder = eg.GuMapBuilder().k(0)
        with self.assertRaises(ValueError):
            builder.build()

        builder_epochs = eg.GuMapBuilder().n_epochs(0)
        with self.assertRaises(ValueError):
            builder_epochs.build()


if __name__ == "__main__":
    unittest.main()
