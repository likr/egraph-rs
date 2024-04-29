import egraph as eg
import networkx as nx
import unittest


def create_graph(nx_graph):
    graph = eg.DiGraph() if nx.is_directed(nx_graph) else eg.Graph()
    indices = {}
    for u in nx_graph.nodes:
        indices[u] = graph.add_node(u)
    for u, v in nx_graph.edges:
        graph.add_edge(indices[u], indices[v], (u, v))
    return (nx_graph, graph)


class TestShortestPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._graphs = [
            create_graph(nx.les_miserables_graph().to_undirected()),
        ]
        cls._digraphs = [
            create_graph(nx.les_miserables_graph().to_directed()),
            create_graph(nx.gn_graph(100, seed=0))
        ]

    def check(self, nx_graph, d_actual):
        d_expected = nx.floyd_warshall_numpy(nx_graph, weight='dummy')
        n = nx_graph.number_of_nodes()
        for i in range(n):
            for j in range(n):
                self.assertEqual(
                    d_actual.get(i, j),
                    d_expected[i, j],
                    f'({i},{j})'
                )

    def test_all_sources_bfs(self):
        for nx_graph, graph in self._graphs:
            self.check(nx_graph, eg.all_sources_bfs(graph, 1))

    def test_all_sources_bfs_directed(self):
        for nx_graph, graph in self._digraphs:
            self.check(nx_graph, eg.all_sources_bfs(graph, 1))

    def test_all_sources_dijkstra(self):
        for nx_graph, graph in self._graphs:
            self.check(nx_graph, eg.all_sources_dijkstra(graph, lambda _: 1))

    def test_all_sources_dijkstra_directed(self):
        for nx_graph, graph in self._digraphs:
            self.check(nx_graph, eg.all_sources_dijkstra(graph, lambda _: 1))

    def test_warshall_floyd(self):
        for nx_graph, graph in self._graphs:
            self.check(nx_graph, eg.warshall_floyd(graph, lambda _: 1))

    def test_warshall_floyd_directed(self):
        for nx_graph, graph in self._digraphs:
            self.check(nx_graph, eg.warshall_floyd(graph, lambda _: 1))


if __name__ == '__main__':
    unittest.main()
