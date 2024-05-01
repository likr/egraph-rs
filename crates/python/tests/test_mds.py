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
        assert (math.isfinite(drawing.x(u)))
        assert (math.isfinite(drawing.y(u)))


def check_drawing_3d(graph, drawing):
    for u in graph.node_indices():
        assert (math.isfinite(drawing.x(u)))
        assert (math.isfinite(drawing.y(u)))


class TestQualityMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._graphs = [
            convert_graph(nx.les_miserables_graph())
        ]

    def test_classical_mds(self):
        for graph in self._graphs:
            mds = eg.ClassicalMds(graph, lambda _: 30)
            drawing = mds.run_2d()
            check_drawing_2d(graph, drawing)

    def test_pivot_mds(self):
        for graph in self._graphs:
            pivot = graph.node_indices()[:10]
            mds = eg.PivotMds(graph, lambda _: 30, pivot)
            drawing = mds.run_2d()
            check_drawing_2d(graph, drawing)


if __name__ == '__main__':
    unittest.main()
