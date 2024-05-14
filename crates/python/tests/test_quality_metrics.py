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

    d = eg.all_sources_bfs(graph, 30)
    drawing = eg.DrawingEuclidean2d.initial_placement(graph)
    rng = eg.Rng.seed_from(0)
    sgd = eg.FullSgd.new_with_distance_matrix(d)
    scheduler = sgd.scheduler(100, 0.1)

    def step(eta):
        sgd.shuffle(rng)
        sgd.apply(drawing, eta)
    scheduler.run(step)

    return (graph, drawing, d)


def draw_torus_2d(nx_graph):
    graph = eg.Graph()
    indices = {}
    for u in nx_graph.nodes:
        indices[u] = graph.add_node(u)
    for u, v in nx_graph.edges:
        graph.add_edge(indices[u], indices[v], (u, v))

    d = eg.all_sources_bfs(graph, 1 / 30)
    drawing = eg.DrawingTorus2d.initial_placement(graph)
    rng = eg.Rng.seed_from(0)
    sgd = eg.FullSgd.new_with_distance_matrix(d)
    scheduler = sgd.scheduler(100, 0.1)

    def step(eta):
        sgd.shuffle(rng)
        sgd.apply(drawing, eta)
    scheduler.run(step)

    return (graph, drawing, d)


class TestQualityMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._graphs = [
            draw(nx.les_miserables_graph())
        ]
        cls._torus_graphs = [
            draw_torus_2d(nx.les_miserables_graph())
        ]

    def test_angular_resolution(self):
        for (graph, drawing, _) in self._graphs:
            assert (math.isfinite(eg.angular_resolution(graph, drawing)))

    def test_aspect_ratio(self):
        for (_, drawing, _) in self._graphs:
            assert (math.isfinite(eg.aspect_ratio(drawing)))

    def test_crossing_angle(self):
        for (graph, drawing, _) in self._graphs:
            assert (math.isfinite(eg.crossing_angle(graph, drawing)))

    def test_crossing_angle_with_crossing_edges(self):
        for (graph, drawing, _) in self._graphs:
            crossings = eg.crossing_edges(graph, drawing)
            assert (math.isfinite(eg.crossing_angle_with_crossing_edges(crossings)))
        for (graph, drawing, _) in self._torus_graphs:
            crossings = eg.crossing_edges(graph, drawing)
            assert (math.isfinite(eg.crossing_angle_with_crossing_edges(crossings)))

    def test_crossing_number(self):
        for (graph, drawing, _) in self._graphs:
            assert (math.isfinite(eg.crossing_number(graph, drawing)))

    def test_crossing_number_with_crossing_edges(self):
        for (graph, drawing, _) in self._graphs:
            crossings = eg.crossing_edges(graph, drawing)
            assert (math.isfinite(
                eg.crossing_number_with_crossing_edges(crossings)))
        for (graph, drawing, _) in self._torus_graphs:
            crossings = eg.crossing_edges(graph, drawing)
            assert (math.isfinite(
                eg.crossing_number_with_crossing_edges(crossings)))

    def test_gabriel_graph_property(self):
        for (graph, drawing, _) in self._graphs:
            assert (math.isfinite(eg.gabriel_graph_property(graph, drawing)))

    def test_ideal_edge_lengths(self):
        for (graph, drawing, distance_matrix) in self._graphs:
            assert (math.isfinite(eg.ideal_edge_lengths(
                graph, drawing, distance_matrix)))

    def test_neighborhood_preservation(self):
        for (graph, drawing, _) in self._graphs:
            assert (math.isfinite(eg.neighborhood_preservation(graph, drawing)))

    def test_node_resolution(self):
        for (_, drawing, _) in self._graphs:
            assert (math.isfinite(eg.node_resolution(drawing)))
        for (_, drawing, _) in self._torus_graphs:
            assert (math.isfinite(eg.node_resolution(drawing)))

    def test_stress(self):
        for (_, drawing, distance_matrix) in self._graphs:
            assert (math.isfinite(eg.stress(drawing, distance_matrix)))
        for (_, drawing, _) in self._torus_graphs:
            assert (math.isfinite(eg.stress(drawing, distance_matrix)))


if __name__ == '__main__':
    unittest.main()
