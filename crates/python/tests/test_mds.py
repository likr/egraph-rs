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

def check_drawing(graph, drawing):
    for u in graph.node_indices():
        assert(math.isfinite(drawing.x(u)))
        assert(math.isfinite(drawing.y(u)))

class TestQualityMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._graphs = [
            draw(nx.les_miserables_graph())
        ]

    def test_classical_mds(self):
        mds = eg.ClassicalMds()
        for graph in self._graphs:
            drawing = mds.run(graph, lambda _: 30)
            check_drawing(graph, drawing)

    def test_pivot_mds(self):
        mds = eg.PivotMds()
        for graph in self._graphs:
            pivot = graph.node_indices()[:10]
            drawing = mds.run(graph, lambda _: 30, pivot)
            check_drawing(graph, drawing)


if __name__ == '__main__':
    unittest.main()
