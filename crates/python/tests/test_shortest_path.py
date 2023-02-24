import unittest
import networkx as nx
from egraph import Graph, all_sources_bfs, all_sources_dijkstra, warshall_floyd


def create_graph(nx_graph):
    graph = Graph()
    indices = {}
    for u in nx_graph.nodes:
        indices[u] = graph.add_node(u)
    for u, v in nx_graph.edges:
        graph.add_edge(indices[u], indices[v], (u, v))
    return graph


class TestQualityMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._graphs = [
            create_graph(nx.les_miserables_graph())
        ]

    def test_all_sources_bfs(self):
        for graph in self._graphs:
            all_sources_bfs(graph, 1)

    def test_all_sources_dijkstra(self):
        for graph in self._graphs:
            all_sources_dijkstra(graph, lambda _: 1)

    def test_warshall_floyd(self):
        for graph in self._graphs:
            warshall_floyd(graph, lambda _: 1)


if __name__ == '__main__':
    unittest.main()
