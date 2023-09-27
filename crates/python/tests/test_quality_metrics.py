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

    drawing = eg.Drawing.initial_placement(graph)
    rng = eg.Rng.seed_from(0)  # random seed
    sgd = eg.SparseSgd(
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

    return (graph, drawing, eg.all_sources_bfs(graph, 1))


class TestQualityMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._graphs = [
            draw(nx.les_miserables_graph())
        ]

    def test_angular_resolution(self):
        for (graph, drawing, _) in self._graphs:
            assert(math.isfinite(eg.angular_resolution(graph, drawing)))

    def test_aspect_ratio(self):
        for (_, drawing, _) in self._graphs:
            assert(math.isfinite(eg.aspect_ratio(drawing)))

    def test_crossing_angle(self):
        for (graph, drawing, _) in self._graphs:
            assert(math.isfinite(eg.crossing_angle(graph, drawing)))

    def test_crossing_angle_with_crossing_edges(self):
        for (graph, drawing, _) in self._graphs:
            crossings = eg.crossing_edges(graph, drawing)
            assert(math.isfinite(eg.crossing_angle(graph, drawing, crossings)))

    def test_crossing_number(self):
        for (graph, drawing, _) in self._graphs:
            assert(math.isfinite(eg.crossing_number(graph, drawing)))

    def test_crossing_number_with_crossing_edges(self):
        for (graph, drawing, _) in self._graphs:
            crossings = eg.crossing_edges(graph, drawing)
            assert(math.isfinite(eg.crossing_number(graph, drawing, crossings)))

    def test_gabriel_graph_property(self):
        for (graph, drawing, _) in self._graphs:
            assert(math.isfinite(eg.gabriel_graph_property(graph, drawing)))

    def test_ideal_edge_lengths(self):
        for (graph, drawing, distance_matrix) in self._graphs:
            assert(math.isfinite(eg.ideal_edge_lengths(graph, drawing, distance_matrix)))

    def test_neighborhood_preservation(self):
        for (graph, drawing, _) in self._graphs:
            assert(math.isfinite(eg.neighborhood_preservation(graph, drawing)))

    def test_node_resolution(self):
        for (graph, drawing, _) in self._graphs:
            assert(math.isfinite(eg.node_resolution(drawing)))

    def test_stress(self):
        for (_, drawing, distance_matrix) in self._graphs:
            assert(math.isfinite(eg.stress(drawing, distance_matrix)))


if __name__ == '__main__':
    unittest.main()
