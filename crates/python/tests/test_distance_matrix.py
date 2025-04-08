import unittest
import math
import networkx as nx
import egraph as eg


def create_line_graph(size=3):
    """
    Create a line graph with the specified number of nodes
    """
    graph = eg.Graph()
    nodes = []
    for i in range(size):
        nodes.append(graph.add_node(i))
    for i in range(size - 1):
        graph.add_edge(nodes[i], nodes[i + 1], (i, i + 1))
    return graph, nodes


def create_cycle_graph(size=3):
    """
    Create a cycle graph with the specified number of nodes
    """
    graph = eg.Graph()
    nodes = []
    for i in range(size):
        nodes.append(graph.add_node(i))
    for i in range(size):
        graph.add_edge(nodes[i], nodes[(i + 1) % size], (i, (i + 1) % size))
    return graph, nodes


def create_complete_graph(size=3):
    """
    Create a complete graph with the specified number of nodes
    """
    graph = eg.Graph()
    nodes = []
    for i in range(size):
        nodes.append(graph.add_node(i))
    for i in range(size):
        for j in range(i + 1, size):
            graph.add_edge(nodes[i], nodes[j], (i, j))
    return graph, nodes


def create_directed_line_graph(size=3):
    """
    Create a directed line graph with the specified number of nodes
    """
    graph = eg.DiGraph()
    nodes = []
    for i in range(size):
        nodes.append(graph.add_node(i))
    for i in range(size - 1):
        graph.add_edge(nodes[i], nodes[i + 1], (i, i + 1))
    return graph, nodes


def create_directed_cycle_graph(size=3):
    """
    Create a directed cycle graph with the specified number of nodes
    """
    graph = eg.DiGraph()
    nodes = []
    for i in range(size):
        nodes.append(graph.add_node(i))
    for i in range(size):
        graph.add_edge(nodes[i], nodes[(i + 1) % size], (i, (i + 1) % size))
    return graph, nodes


def create_disconnected_graph():
    """
    Create a disconnected graph with two components
    """
    graph = eg.Graph()
    nodes = []
    # First component
    for i in range(3):
        nodes.append(graph.add_node(i))
    graph.add_edge(nodes[0], nodes[1], (0, 1))
    graph.add_edge(nodes[1], nodes[2], (1, 2))

    # Second component
    for i in range(3, 6):
        nodes.append(graph.add_node(i))
    graph.add_edge(nodes[3], nodes[4], (3, 4))
    graph.add_edge(nodes[4], nodes[5], (4, 5))

    return graph, nodes


def draw(nx_graph):
    """
    Convert a NetworkX graph to an egraph Graph
    """
    graph = eg.Graph()
    indices = {}
    for u in nx_graph.nodes:
        indices[u] = graph.add_node(u)
    for u, v in nx_graph.edges:
        graph.add_edge(indices[u], indices[v], (u, v))
    return graph


class TestDistanceMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create some test graphs
        cls.line_graph, cls.line_nodes = create_line_graph(5)
        cls.cycle_graph, cls.cycle_nodes = create_cycle_graph(6)
        cls.complete_graph, cls.complete_nodes = create_complete_graph(4)
        cls.directed_line_graph, cls.directed_line_nodes = create_directed_line_graph(
            5)
        cls.directed_cycle_graph, cls.directed_cycle_nodes = create_directed_cycle_graph(
            6)
        cls.disconnected_graph, cls.disconnected_nodes = create_disconnected_graph()
        cls.les_mis_graph = draw(nx.les_miserables_graph())

    def test_constructor_undirected(self):
        """
        Test creating a distance matrix from an undirected graph
        """
        # Create a distance matrix from a line graph using all_sources_dijkstra
        distance_matrix = eg.all_sources_dijkstra(
            self.line_graph, lambda e: 1.0)

        # Verify that the instance exists
        self.assertIsInstance(distance_matrix, eg.DistanceMatrix)

        # Check distances between adjacent nodes
        for i in range(4):
            self.assertEqual(distance_matrix.get(i, i+1), 1.0)
            # Symmetric for undirected
            self.assertEqual(distance_matrix.get(i+1, i), 1.0)

        # Check distances between non-adjacent nodes
        self.assertEqual(distance_matrix.get(0, 2), 2.0)
        self.assertEqual(distance_matrix.get(0, 3), 3.0)
        self.assertEqual(distance_matrix.get(0, 4), 4.0)
        self.assertEqual(distance_matrix.get(1, 3), 2.0)
        self.assertEqual(distance_matrix.get(1, 4), 3.0)
        self.assertEqual(distance_matrix.get(2, 4), 2.0)

        # Check self-distances
        for i in range(5):
            self.assertEqual(distance_matrix.get(i, i), 0.0)

    def test_constructor_directed(self):
        """
        Test creating a distance matrix from a directed graph
        """
        # Create a distance matrix from a directed line graph using all_sources_dijkstra
        distance_matrix = eg.all_sources_dijkstra(
            self.directed_line_graph, lambda e: 1.0)

        # Verify that the instance exists
        self.assertIsInstance(distance_matrix, eg.DistanceMatrix)

        # Check distances in forward direction
        for i in range(4):
            self.assertEqual(distance_matrix.get(i, i+1), 1.0)

        # Check distances in backward direction (should be infinity)
        for i in range(1, 5):
            self.assertTrue(math.isinf(distance_matrix.get(i, i-1)))

        # Check distances between non-adjacent nodes in forward direction
        self.assertEqual(distance_matrix.get(0, 2), 2.0)
        self.assertEqual(distance_matrix.get(0, 3), 3.0)
        self.assertEqual(distance_matrix.get(0, 4), 4.0)
        self.assertEqual(distance_matrix.get(1, 3), 2.0)
        self.assertEqual(distance_matrix.get(1, 4), 3.0)
        self.assertEqual(distance_matrix.get(2, 4), 2.0)

        # Check self-distances
        for i in range(5):
            self.assertEqual(distance_matrix.get(i, i), 0.0)

    def test_constructor_cycle(self):
        """
        Test creating a distance matrix from a cycle graph
        """
        # Create a distance matrix from a cycle graph using all_sources_dijkstra
        distance_matrix = eg.all_sources_dijkstra(
            self.cycle_graph, lambda e: 1.0)

        # Verify that the instance exists
        self.assertIsInstance(distance_matrix, eg.DistanceMatrix)

        # Check distances between adjacent nodes
        for i in range(6):
            self.assertEqual(distance_matrix.get(i, (i+1) % 6), 1.0)
            self.assertEqual(distance_matrix.get((i+1) % 6, i),
                             1.0)  # Symmetric for undirected

        # Check distances between non-adjacent nodes
        # In a cycle of 6 nodes, the maximum distance is 3
        self.assertEqual(distance_matrix.get(0, 3), 3.0)
        self.assertEqual(distance_matrix.get(1, 4), 3.0)
        self.assertEqual(distance_matrix.get(2, 5), 3.0)

        # Check some other distances
        self.assertEqual(distance_matrix.get(0, 2), 2.0)
        self.assertEqual(distance_matrix.get(0, 4), 2.0)
        self.assertEqual(distance_matrix.get(1, 3), 2.0)
        self.assertEqual(distance_matrix.get(1, 5), 2.0)

        # Check self-distances
        for i in range(6):
            self.assertEqual(distance_matrix.get(i, i), 0.0)

    def test_constructor_complete(self):
        """
        Test creating a distance matrix from a complete graph
        """
        # Create a distance matrix from a complete graph using all_sources_dijkstra
        distance_matrix = eg.all_sources_dijkstra(
            self.complete_graph, lambda e: 1.0)

        # Verify that the instance exists
        self.assertIsInstance(distance_matrix, eg.DistanceMatrix)

        # In a complete graph, all nodes are adjacent
        for i in range(4):
            for j in range(4):
                if i != j:
                    self.assertEqual(distance_matrix.get(i, j), 1.0)
                else:
                    self.assertEqual(distance_matrix.get(i, j), 0.0)

    def test_constructor_disconnected(self):
        """
        Test creating a distance matrix from a disconnected graph
        """
        # Create a distance matrix from a disconnected graph using all_sources_dijkstra
        distance_matrix = eg.all_sources_dijkstra(
            self.disconnected_graph, lambda e: 1.0)

        # Verify that the instance exists
        self.assertIsInstance(distance_matrix, eg.DistanceMatrix)

        # Check distances within the first component
        self.assertEqual(distance_matrix.get(0, 1), 1.0)
        self.assertEqual(distance_matrix.get(0, 2), 2.0)
        self.assertEqual(distance_matrix.get(1, 2), 1.0)

        # Check distances within the second component
        self.assertEqual(distance_matrix.get(3, 4), 1.0)
        self.assertEqual(distance_matrix.get(3, 5), 2.0)
        self.assertEqual(distance_matrix.get(4, 5), 1.0)

        # Check distances between components (should be infinity)
        for i in range(3):
            for j in range(3, 6):
                self.assertTrue(math.isinf(distance_matrix.get(i, j)))
                self.assertTrue(math.isinf(distance_matrix.get(j, i)))

    def test_constructor_empty(self):
        """
        Test creating a distance matrix from an empty graph
        """
        # Create an empty graph
        graph = eg.Graph()

        # Create a distance matrix using all_sources_dijkstra
        distance_matrix = eg.all_sources_dijkstra(graph, lambda e: 1.0)

        # Verify that the instance exists
        self.assertIsInstance(distance_matrix, eg.DistanceMatrix)

    def test_constructor_single_node(self):
        """
        Test creating a distance matrix from a single-node graph
        """
        # Create a single-node graph
        graph = eg.Graph()
        node = graph.add_node(0)

        # Create a distance matrix using all_sources_dijkstra
        distance_matrix = eg.all_sources_dijkstra(graph, lambda e: 1.0)

        # Verify that the instance exists
        self.assertIsInstance(distance_matrix, eg.DistanceMatrix)

        # Check self-distance
        self.assertEqual(distance_matrix.get(0, 0), 0.0)

    def test_set_get(self):
        """
        Test setting and getting distances
        """
        # Create a distance matrix from a line graph using all_sources_dijkstra
        distance_matrix = eg.all_sources_dijkstra(
            self.line_graph, lambda e: 1.0)

        # Get the initial distances
        initial_0_1 = distance_matrix.get(0, 1)
        initial_1_2 = distance_matrix.get(1, 2)
        initial_0_4 = distance_matrix.get(0, 4)
        initial_2_3 = distance_matrix.get(2, 3)
        initial_3_4 = distance_matrix.get(3, 4)

        # Verify initial distances
        self.assertEqual(initial_0_1, 1.0)
        self.assertEqual(initial_1_2, 1.0)
        self.assertEqual(initial_0_4, 4.0)
        self.assertEqual(initial_2_3, 1.0)
        self.assertEqual(initial_3_4, 1.0)

        # Modify some distances
        distance_matrix.set(0, 1, 2.5)
        distance_matrix.set(1, 2, 3.5)
        distance_matrix.set(0, 4, 10.0)

        # Check the modified distances
        self.assertEqual(distance_matrix.get(0, 1), 2.5)
        self.assertEqual(distance_matrix.get(1, 2), 3.5)
        self.assertEqual(distance_matrix.get(0, 4), 10.0)

        # Note: Setting (i,j) does not automatically set (j,i), even for undirected graphs
        # We need to set both directions explicitly
        distance_matrix.set(1, 0, 2.5)
        distance_matrix.set(2, 1, 3.5)
        distance_matrix.set(4, 0, 10.0)

        # Now check that both directions are set
        self.assertEqual(distance_matrix.get(1, 0), 2.5)
        self.assertEqual(distance_matrix.get(2, 1), 3.5)
        self.assertEqual(distance_matrix.get(4, 0), 10.0)

        # Other distances should remain unchanged
        self.assertEqual(distance_matrix.get(2, 3), initial_2_3)
        self.assertEqual(distance_matrix.get(3, 4), initial_3_4)

    def test_set_get_directed(self):
        """
        Test setting and getting distances in a directed graph
        """
        # Create a distance matrix from a directed line graph using all_sources_dijkstra
        distance_matrix = eg.all_sources_dijkstra(
            self.directed_line_graph, lambda e: 1.0)

        # Get the initial distances
        initial_0_1 = distance_matrix.get(0, 1)
        initial_1_2 = distance_matrix.get(1, 2)
        initial_0_4 = distance_matrix.get(0, 4)
        initial_2_3 = distance_matrix.get(2, 3)
        initial_3_4 = distance_matrix.get(3, 4)

        # Verify initial distances
        self.assertEqual(initial_0_1, 1.0)
        self.assertEqual(initial_1_2, 1.0)
        self.assertEqual(initial_0_4, 4.0)
        self.assertEqual(initial_2_3, 1.0)
        self.assertEqual(initial_3_4, 1.0)

        # Check backward edges (should be infinity)
        self.assertTrue(math.isinf(distance_matrix.get(1, 0)))
        self.assertTrue(math.isinf(distance_matrix.get(2, 1)))
        self.assertTrue(math.isinf(distance_matrix.get(4, 0)))

        # Modify some distances
        distance_matrix.set(0, 1, 2.5)
        distance_matrix.set(1, 2, 3.5)
        distance_matrix.set(0, 4, 10.0)

        # Check the modified distances
        self.assertEqual(distance_matrix.get(0, 1), 2.5)
        self.assertEqual(distance_matrix.get(1, 2), 3.5)
        self.assertEqual(distance_matrix.get(0, 4), 10.0)

        # For directed graphs, setting (i,j) should not affect (j,i)
        self.assertTrue(math.isinf(distance_matrix.get(1, 0)))
        self.assertTrue(math.isinf(distance_matrix.get(2, 1)))
        self.assertTrue(math.isinf(distance_matrix.get(4, 0)))

        # Other distances should remain unchanged
        self.assertEqual(distance_matrix.get(2, 3), initial_2_3)
        self.assertEqual(distance_matrix.get(3, 4), initial_3_4)

    def test_set_nonexistent(self):
        """
        Test setting distances for nonexistent node pairs
        """
        # Create a distance matrix from a line graph using all_sources_dijkstra
        distance_matrix = eg.all_sources_dijkstra(
            self.line_graph, lambda e: 1.0)

        # Try to set distances for nonexistent nodes
        result = distance_matrix.set(10, 11, 5.0)

        # Should return None to indicate failure
        self.assertIsNone(result)

        # Existing distances should remain unchanged
        self.assertEqual(distance_matrix.get(0, 1), 1.0)
        self.assertEqual(distance_matrix.get(1, 2), 1.0)

    def test_integration_stress_majorization(self):
        """
        Test integration with StressMajorization
        """
        # Create a drawing
        drawing = eg.DrawingEuclidean2d.initial_placement(self.line_graph)

        # Create a distance matrix using all_sources_dijkstra
        distance_matrix = eg.all_sources_dijkstra(
            self.line_graph, lambda e: 1.0)

        # Create a StressMajorization instance with the distance matrix
        sm = eg.StressMajorization.with_distance_matrix(
            drawing, distance_matrix)

        # Verify that the instance exists
        self.assertIsInstance(sm, eg.StressMajorization)

        # Apply a single iteration
        stress = sm.apply(drawing)

        # Verify that the stress value is a finite number
        self.assertTrue(math.isfinite(stress))

        # Run the algorithm
        sm.run(drawing)

        # Verify that all coordinates are finite
        for u in self.line_graph.node_indices():
            self.assertTrue(math.isfinite(drawing.x(u)))
            self.assertTrue(math.isfinite(drawing.y(u)))

    def test_with_large_graph(self):
        """
        Test with a larger graph (Les Miserables)
        """
        # Create a distance matrix from the Les Miserables graph using all_sources_dijkstra
        distance_matrix = eg.all_sources_dijkstra(
            self.les_mis_graph, lambda e: 1.0)

        # Verify that the instance exists
        self.assertIsInstance(distance_matrix, eg.DistanceMatrix)

        # Check a few distances
        # Since we don't know the exact distances, just verify they are finite
        for i in range(10):
            for j in range(10):
                if i != j:
                    distance = distance_matrix.get(i, j)
                    if distance is not None:
                        self.assertTrue(math.isfinite(distance))
                else:
                    self.assertEqual(distance_matrix.get(i, i), 0.0)

    def test_all_sources_dijkstra(self):
        """
        Test creating a distance matrix using all_sources_dijkstra
        """
        # Create a distance matrix using all_sources_dijkstra
        distance_matrix = eg.all_sources_dijkstra(
            self.line_graph, lambda e: 1.0)

        # Verify that the instance exists
        self.assertIsInstance(distance_matrix, eg.DistanceMatrix)

        # Check distances between adjacent nodes
        for i in range(4):
            self.assertEqual(distance_matrix.get(i, i+1), 1.0)
            # Symmetric for undirected
            self.assertEqual(distance_matrix.get(i+1, i), 1.0)

        # Check distances between non-adjacent nodes
        self.assertEqual(distance_matrix.get(0, 2), 2.0)
        self.assertEqual(distance_matrix.get(0, 3), 3.0)
        self.assertEqual(distance_matrix.get(0, 4), 4.0)
        self.assertEqual(distance_matrix.get(1, 3), 2.0)
        self.assertEqual(distance_matrix.get(1, 4), 3.0)
        self.assertEqual(distance_matrix.get(2, 4), 2.0)

        # Check self-distances
        for i in range(5):
            self.assertEqual(distance_matrix.get(i, i), 0.0)

    def test_all_sources_dijkstra_with_weights(self):
        """
        Test creating a distance matrix using all_sources_dijkstra with custom weights
        """
        # Create a dictionary to store custom weights
        weights = {}
        for e in self.line_graph.edge_indices():
            edge_data = self.line_graph.edge_weight(e)
            i, j = edge_data
            # Custom weight based on node indices
            weights[e] = abs(i - j) * 2.0

        # Create a distance matrix using all_sources_dijkstra with custom weights
        distance_matrix = eg.all_sources_dijkstra(
            self.line_graph, lambda e: weights[e])

        # Verify that the instance exists
        self.assertIsInstance(distance_matrix, eg.DistanceMatrix)

        # Check distances between adjacent nodes
        # The weight of edge (i, i+1) is 2.0
        for i in range(4):
            self.assertEqual(distance_matrix.get(i, i+1), 2.0)
            # Symmetric for undirected
            self.assertEqual(distance_matrix.get(i+1, i), 2.0)

        # Check distances between non-adjacent nodes
        self.assertEqual(distance_matrix.get(0, 2), 4.0)  # 2.0 + 2.0
        self.assertEqual(distance_matrix.get(0, 3), 6.0)  # 2.0 + 2.0 + 2.0
        self.assertEqual(distance_matrix.get(
            0, 4), 8.0)  # 2.0 + 2.0 + 2.0 + 2.0
        self.assertEqual(distance_matrix.get(1, 3), 4.0)  # 2.0 + 2.0
        self.assertEqual(distance_matrix.get(1, 4), 6.0)  # 2.0 + 2.0 + 2.0
        self.assertEqual(distance_matrix.get(2, 4), 4.0)  # 2.0 + 2.0

        # Check self-distances
        for i in range(5):
            self.assertEqual(distance_matrix.get(i, i), 0.0)


if __name__ == "__main__":
    unittest.main()
