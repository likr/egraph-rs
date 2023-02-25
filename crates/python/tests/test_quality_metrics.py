import unittest
import networkx as nx
from egraph import Graph, Coordinates, Rng, SparseSgd, all_sources_bfs, angular_resolution, aspect_ratio, crossing_angle, crossing_number, gabriel_graph_property, ideal_edge_lengths, neighborhood_preservation, node_resolution, stress, crossing_edges


def draw(nx_graph):
    graph = Graph()
    indices = {}
    for u in nx_graph.nodes:
        indices[u] = graph.add_node(u)
    for u, v in nx_graph.edges:
        graph.add_edge(indices[u], indices[v], (u, v))

    drawing = Coordinates.initial_placement(graph)
    rng = Rng.seed_from(0)  # random seed
    sgd = SparseSgd(
        graph,
        lambda _: 30,  # edge length
        50,  # number of pivots
        rng,
    )
    scheduler = sgd.scheduler(
        100,  # number of iterations
        0.1,  # eps: eta_min = eps * min d[i, j] ^ 2
    )

    def step(eta):
        sgd.shuffle(rng)
        sgd.apply(drawing, eta)
    scheduler.run(step)

    return (graph, drawing, all_sources_bfs(graph, 1))


class TestQualityMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._graphs = [
            draw(nx.les_miserables_graph())
        ]

    def test_angular_resolution(self):
        for (graph, drawing, _) in self._graphs:
            angular_resolution(graph, drawing)

    def test_aspect_ratio(self):
        for (_, drawing, _) in self._graphs:
            aspect_ratio(drawing)

    def test_crossing_angle(self):
        for (graph, drawing, _) in self._graphs:
            crossing_angle(graph, drawing)

    def test_crossing_angle_with_crossing_edges(self):
        for (graph, drawing, _) in self._graphs:
            crossings = crossing_edges(graph, drawing)
            crossing_angle(graph, drawing, crossings)

    def test_crossing_number(self):
        for (graph, drawing, _) in self._graphs:
            crossing_number(graph, drawing)

    def test_crossing_number_with_crossing_edges(self):
        for (graph, drawing, _) in self._graphs:
            crossings = crossing_edges(graph, drawing)
            crossing_number(graph, drawing, crossings)

    def test_gabriel_graph_property(self):
        for (graph, drawing, _) in self._graphs:
            gabriel_graph_property(graph, drawing)

    def test_ideal_edge_lengths(self):
        for (graph, drawing, distance_matrix) in self._graphs:
            ideal_edge_lengths(graph, drawing, distance_matrix)

    def test_neighborhood_preservation(self):
        for (graph, drawing, _) in self._graphs:
            neighborhood_preservation(graph, drawing)

    def test_node_resolution(self):
        for (graph, drawing, _) in self._graphs:
            node_resolution(graph, drawing)

    def test_stress(self):
        for (_, drawing, distance_matrix) in self._graphs:
            stress(drawing, distance_matrix)


if __name__ == '__main__':
    unittest.main()
