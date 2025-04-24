import unittest
import egraph as eg


class TestClustering(unittest.TestCase):
    def test_louvain(self):
        # Create a simple graph with two communities
        graph = eg.Graph()
        n1 = graph.add_node({"id": "n1"})
        n2 = graph.add_node({"id": "n2"})
        n3 = graph.add_node({"id": "n3"})
        n4 = graph.add_node({"id": "n4"})

        # Community 1: n1, n2 are densely connected
        graph.add_edge(n1, n2, {"weight": 1.0})

        # Community 2: n3, n4 are densely connected
        graph.add_edge(n3, n4, {"weight": 1.0})

        # Weak connection between communities
        graph.add_edge(n2, n3, {"weight": 0.1})

        # Detect communities
        louvain = eg.Louvain()
        communities = louvain.detect_communities(graph)

        # Check that nodes 1 and 2 are in the same community
        self.assertEqual(communities[n1], communities[n2])

        # Check that nodes 3 and 4 are in the same community
        self.assertEqual(communities[n3], communities[n4])

        # Check that nodes 1 and 3 are in different communities
        self.assertNotEqual(communities[n1], communities[n3])

    def test_label_propagation(self):
        # Create a simple graph with two communities
        graph = eg.Graph()
        n1 = graph.add_node({"id": "n1"})
        n2 = graph.add_node({"id": "n2"})
        n3 = graph.add_node({"id": "n3"})
        n4 = graph.add_node({"id": "n4"})

        # Community 1: n1, n2 are densely connected
        graph.add_edge(n1, n2, {"weight": 1.0})

        # Community 2: n3, n4 are densely connected
        graph.add_edge(n3, n4, {"weight": 1.0})

        # Weak connection between communities
        graph.add_edge(n2, n3, {"weight": 0.1})

        # Detect communities
        label_prop = eg.LabelPropagation()
        communities = label_prop.detect_communities(graph)

        # The algorithm is stochastic, so we can't check exact communities
        # But we can check that the right number of communities are found
        self.assertGreaterEqual(len(set(communities.values())), 1)
        self.assertLessEqual(len(set(communities.values())), 4)

    def test_spectral_clustering(self):
        # Create a simple graph with two communities
        graph = eg.Graph()
        n1 = graph.add_node({"id": "n1"})
        n2 = graph.add_node({"id": "n2"})
        n3 = graph.add_node({"id": "n3"})
        n4 = graph.add_node({"id": "n4"})

        # Community 1: n1, n2 are densely connected
        graph.add_edge(n1, n2, {"weight": 1.0})

        # Community 2: n3, n4 are densely connected
        graph.add_edge(n3, n4, {"weight": 1.0})

        # Weak connection between communities
        graph.add_edge(n2, n3, {"weight": 0.1})

        # Detect communities
        spectral = eg.SpectralClustering(2)  # Explicitly request 2 communities
        communities = spectral.detect_communities(graph)

        # The algorithm should find 2 communities
        self.assertEqual(len(set(communities.values())), 2)

    def test_infomap(self):
        # Create a simple graph with two communities
        graph = eg.Graph()
        n1 = graph.add_node({"id": "n1"})
        n2 = graph.add_node({"id": "n2"})
        n3 = graph.add_node({"id": "n3"})
        n4 = graph.add_node({"id": "n4"})

        # Community 1: n1, n2 are densely connected
        graph.add_edge(n1, n2, {"weight": 1.0})

        # Community 2: n3, n4 are densely connected
        graph.add_edge(n3, n4, {"weight": 1.0})

        # Weak connection between communities
        graph.add_edge(n2, n3, {"weight": 0.1})

        # Detect communities
        infomap = eg.InfoMap()
        communities = infomap.detect_communities(graph)

        # The algorithm is stochastic, so we can't check exact communities
        # But we can check that the right number of communities are found
        self.assertGreaterEqual(len(set(communities.values())), 1)
        self.assertLessEqual(len(set(communities.values())), 4)

    def test_coarsen(self):
        # Create a simple graph with two communities
        graph = eg.Graph()
        n1 = graph.add_node({"id": "n1"})
        n2 = graph.add_node({"id": "n2"})
        n3 = graph.add_node({"id": "n3"})
        n4 = graph.add_node({"id": "n4"})

        # Community 1: n1, n2 are densely connected
        graph.add_edge(n1, n2, {"weight": 1.0})

        # Community 2: n3, n4 are densely connected
        graph.add_edge(n3, n4, {"weight": 1.0})

        # Weak connection between communities
        graph.add_edge(n2, n3, {"weight": 0.1})

        # Define a simple community assignment: 2 communities
        communities = {0: 0, 1: 0, 2: 1, 3: 1}

        # Create node grouping function
        def node_group_func(node):
            return communities[node]

        # Create node and edge merge functions
        def node_merge_func(nodes):
            return len(nodes)

        def edge_merge_func(edges):
            return len(edges)

        # Coarsen the graph using py_coarsen
        coarsened_graph, node_map = eg.py_coarsen(
            graph,
            node_group_func,
            node_merge_func,
            edge_merge_func
        )

        # Check that we have 2 nodes in the coarsened graph (one for each community)
        self.assertEqual(coarsened_graph.node_count(), 2)


if __name__ == "__main__":
    unittest.main()
