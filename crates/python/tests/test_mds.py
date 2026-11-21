import math
import unittest
import networkx as nx
import egraph as eg


def convert_graph(nx_graph):
    graph = eg.Graph()
    indices = {}
    for u in nx_graph.nodes:
        indices[u] = graph.add_node(u)
    for u, v in nx_graph.edges:
        graph.add_edge(indices[u], indices[v], (u, v))
    return graph


def check_drawing_2d(graph, drawing):
    for u in graph.node_indices():
        assert math.isfinite(drawing.x(u))
        assert math.isfinite(drawing.y(u))


def check_drawing_3d(graph, drawing):
    for u in graph.node_indices():
        assert math.isfinite(drawing.get(u, 0))
        assert math.isfinite(drawing.get(u, 1))
        assert math.isfinite(drawing.get(u, 2))


class TestQualityMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._graphs = [
            convert_graph(nx.les_miserables_graph())
        ]

    def test_classical_mds_2d(self):
        for graph in self._graphs:
            mds = eg.ClassicalMds(graph, lambda _: 30)
            drawing = mds.run_2d()
            check_drawing_2d(graph, drawing)

    def test_pivot_mds_2d(self):
        for graph in self._graphs:
            pivot = graph.node_indices()[:10]
            mds = eg.PivotMds(graph, lambda _: 30, pivot)
            drawing = mds.run_2d()
            check_drawing_2d(graph, drawing)

    def test_classical_mds_2d_with_distance_matrix(self):
        for graph in self._graphs:
            d = eg.all_sources_dijkstra(graph, lambda _: 30)
            mds = eg.ClassicalMds.new_with_distance_matrix(d)
            drawing = mds.run_2d()
            check_drawing_2d(graph, drawing)

    def test_pivot_mds_2d_with_distance_matrix(self):
        for graph in self._graphs:
            d = eg.all_sources_dijkstra(graph, lambda _: 30)
            mds = eg.PivotMds.new_with_distance_matrix(d)
            drawing = mds.run_2d()
            check_drawing_2d(graph, drawing)

    def test_pivot_mds_2d_with_kernel_distance(self):
        for graph in self._graphs:
            rng = eg.Rng.seed_from(42)
            rdmds = eg.RdMds().d(2)
            embedding = rdmds.embedding(graph, lambda e: 1.0, eg.Ic0CgSolver(), rng)
            kernel = eg.EmbeddingKernelBuilder().build(graph, embedding)
            d = eg.KernelDistanceBuilder().build(kernel)
            mds = eg.PivotMds.new_with_distance_matrix(d)
            drawing = mds.run_2d()
            check_drawing_2d(graph, drawing)

    def test_classical_mds_3d(self):
        for graph in self._graphs:
            mds = eg.ClassicalMds(graph, lambda _: 30)
            drawing = mds.run(3)
            check_drawing_3d(graph, drawing)

    def test_pivot_mds_3d(self):
        for graph in self._graphs:
            pivot = graph.node_indices()[:10]
            mds = eg.PivotMds(graph, lambda _: 30, pivot)
            drawing = mds.run(3)
            check_drawing_3d(graph, drawing)

    def test_mds_error_handling_and_disconnected(self):
        dg = eg.DiGraph()
        n0 = dg.add_node(0)
        n1 = dg.add_node(1)
        dg.add_edge(n0, n1, 0)
        with self.assertRaises(TypeError):
            eg.ClassicalMds(dg, lambda _: 1.0)
        with self.assertRaises(TypeError):
            eg.PivotMds(dg, lambda _: 1.0, [0])

        g_disc = eg.Graph()
        nodes = [g_disc.add_node(i) for i in range(4)]
        g_disc.add_edge(nodes[0], nodes[1], 0)
        g_disc.add_edge(nodes[2], nodes[3], 1)

        with self.assertRaises(TypeError):
            eg.PivotMds(g_disc, "not_callable", [0])
        with self.assertRaises(ValueError):
            eg.PivotMds(g_disc, lambda _: -1.0, [0])
        with self.assertRaises(ValueError):
            eg.ClassicalMds(g_disc, lambda _: float("nan"))
        with self.assertRaises(IndexError):
            eg.PivotMds(g_disc, lambda _: 1.0, [99])

        mds_disc = eg.PivotMds(g_disc, lambda _: 1.0, [0])
        with self.assertRaises(ValueError):
            mds_disc.eps = float("nan")
        drawing_2d = mds_disc.run_2d()
        check_drawing_2d(g_disc, drawing_2d)

        cmds_disc = eg.ClassicalMds(g_disc, lambda _: 1.0)
        with self.assertRaises(ValueError):
            cmds_disc.eps = 0.0
        check_drawing_2d(g_disc, cmds_disc.run_2d())


if __name__ == '__main__':
    unittest.main()
