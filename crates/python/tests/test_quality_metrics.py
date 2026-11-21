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
    sgd = eg.FullSgd().build_with_distance_matrix(d)
    scheduler = eg.SchedulerExponential(100)

    def step(eta):
        sgd.shuffle(eg.Ic0CgSolver(), rng)
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
    sgd = eg.FullSgd().build_with_distance_matrix(d)
    scheduler = eg.SchedulerExponential(100)

    def step(eta):
        sgd.shuffle(eg.Ic0CgSolver(), rng)
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
        for (graph, drawing, distance_matrix) in self._torus_graphs:
            assert (math.isfinite(eg.ideal_edge_lengths(
                graph, drawing, distance_matrix)))

    def test_neighborhood_preservation(self):
        for (graph, drawing, _) in self._graphs:
            assert (math.isfinite(eg.neighborhood_preservation(graph, drawing)))

    def test_neighborhood_preservation_2hop(self):
        for (graph, drawing, _) in self._graphs:
            val = eg.neighborhood_preservation_2hop(graph, drawing)
            assert (math.isfinite(val))
            assert (0.0 <= val <= 1.0)


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

    def test_metrics_edge_cases_edgeless_and_mismatch(self):
        g0 = eg.Graph()
        d0 = eg.DrawingEuclidean2d.initial_placement(g0)
        self.assertEqual(eg.neighborhood_preservation(g0, d0), 0.0)

        g_edgeless = eg.Graph()
        for i in range(5):
            g_edgeless.add_node(i)
        d_edgeless = eg.DrawingEuclidean2d.initial_placement(g_edgeless)
        np_score = eg.neighborhood_preservation(g_edgeless, d_edgeless)
        self.assertTrue(math.isfinite(np_score))
        self.assertEqual(np_score, 0.0)
        self.assertEqual(eg.neighborhood_preservation_2hop(g_edgeless, d_edgeless), 1.0)

        g4 = eg.Graph()
        nodes = [g4.add_node(i) for i in range(4)]
        g4.add_edge(nodes[0], nodes[1], 0)
        g4.add_edge(nodes[2], nodes[3], 1)

        d_collinear = eg.DrawingEuclidean2d.initial_placement(g4)
        for i in range(4):
            d_collinear.set_x(i, float(i))
            d_collinear.set_y(i, 0.0)
        self.assertEqual(eg.crossing_number(g4, d_collinear), 0.0)

        d_nan = eg.DrawingEuclidean2d.initial_placement(g4)
        d_nan.set_x(0, float("nan"))
        self.assertEqual(eg.crossing_number(g4, d_nan), 0.0)

        g2 = eg.Graph()
        g2.add_node(0)
        g2.add_node(1)
        d2 = eg.DrawingEuclidean2d.initial_placement(g2)
        dist4 = eg.all_sources_bfs(g4, 1.0)
        for fn in (
            eg.crossing_number,
            eg.crossing_angle,
            eg.crossing_edges,
            eg.angular_resolution,
            eg.gabriel_graph_property,
            eg.neighborhood_preservation,
            eg.neighborhood_preservation_2hop,
        ):
            with self.assertRaises(ValueError):
                fn(g4, d2)
        with self.assertRaises(ValueError):
            eg.ideal_edge_lengths(g4, d2, dist4)
        with self.assertRaises(ValueError):
            eg.stress(d2, dist4)


if __name__ == '__main__':
    unittest.main()
