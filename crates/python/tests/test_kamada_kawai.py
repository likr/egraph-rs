import math
import unittest
import networkx as nx
import numpy as np
import egraph as eg


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


def check_drawing_2d(graph, drawing):
    """
    Verify that all coordinates in a 2D drawing are finite
    """
    for u in graph.node_indices():
        assert math.isfinite(drawing.x(u))
        assert math.isfinite(drawing.y(u))


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


def record_positions(drawing, graph):
    """
    Record the positions of all nodes in a drawing
    """
    positions = {}
    for u in graph.node_indices():
        positions[u] = (drawing.x(u), drawing.y(u))
    return positions


def positions_changed(drawing, graph, initial_positions):
    """
    Check if any node positions have changed
    """
    for u in graph.node_indices():
        if (drawing.x(u), drawing.y(u)) != initial_positions[u]:
            return True
    return False


def calculate_energy(graph, drawing, distance_func=None):
    """
    Calculate the energy of a drawing according to the Kamada-Kawai model
    """
    if distance_func is None:
        def distance_func(e): return 1.0

    energy = 0.0
    for i in graph.node_indices():
        for j in graph.node_indices():
            if i < j:  # Only consider each pair once
                # Calculate Euclidean distance in the drawing
                dx = drawing.x(i) - drawing.x(j)
                dy = drawing.y(i) - drawing.y(j)
                actual_distance = math.sqrt(dx * dx + dy * dy)

                # Calculate ideal distance (graph-theoretic)
                # For simplicity, we'll use 1.0 for all edges
                ideal_distance = 1.0

                # Calculate spring constant (typically 1/d^2)
                spring_constant = 1.0 / (ideal_distance * ideal_distance)

                # Add to energy
                diff = actual_distance - ideal_distance
                energy += spring_constant * diff * diff

    return energy


class TestKamadaKawai(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create some test graphs
        cls.line_graph, cls.line_nodes = create_line_graph(5)
        cls.cycle_graph, cls.cycle_nodes = create_cycle_graph(6)
        cls.complete_graph, cls.complete_nodes = create_complete_graph(4)
        cls.les_mis_graph = draw(nx.les_miserables_graph())

    def test_constructor(self):
        """
        Test creating a KamadaKawai instance from a graph
        """
        # Create a KamadaKawai instance
        kk = eg.KamadaKawai(self.line_graph, lambda e: 1.0)

        # Verify that the instance exists
        self.assertIsInstance(kk, eg.KamadaKawai)

    def test_eps_parameter(self):
        """
        Test the eps parameter getter and setter
        """
        # Create a KamadaKawai instance
        kk = eg.KamadaKawai(self.line_graph, lambda e: 1.0)

        # Check default value
        self.assertIsInstance(kk.eps, float)
        self.assertTrue(kk.eps > 0)

        # Test setter
        new_eps = 0.005
        kk.eps = new_eps

        # Verify value was updated - use approximate comparison for floating point
        self.assertAlmostEqual(kk.eps, new_eps, delta=1e-8)

    def test_select_node(self):
        """
        Test node selection functionality
        """
        # Create a drawing
        drawing = eg.DrawingEuclidean2d.initial_placement(self.line_graph)

        # Create a KamadaKawai instance
        kk = eg.KamadaKawai(self.line_graph, lambda e: 1.0)

        # Test node selection
        selected_node = kk.select_node(drawing)

        # Verify that a valid node is selected or None
        if selected_node is not None:
            self.assertIsInstance(selected_node, int)
            self.assertTrue(0 <= selected_node < len(self.line_nodes))

    def test_apply_to_node(self):
        """
        Test applying the algorithm to a single node
        """
        # Create a drawing
        drawing = eg.DrawingEuclidean2d.initial_placement(self.line_graph)

        # Set specific positions for testing
        drawing.set_x(0, 0.0)
        drawing.set_y(0, 0.0)
        drawing.set_x(1, 1.0)
        drawing.set_y(1, 0.0)
        drawing.set_x(2, 0.5)
        drawing.set_y(2, 1.0)
        drawing.set_x(3, 1.5)
        drawing.set_y(3, 1.0)
        drawing.set_x(4, 2.0)
        drawing.set_y(4, 0.0)

        # Record initial position of node 2
        initial_x = drawing.x(2)
        initial_y = drawing.y(2)

        # Create a KamadaKawai instance
        kk = eg.KamadaKawai(self.line_graph, lambda e: 1.0)

        # Apply the algorithm to node 2
        kk.apply_to_node(2, drawing)

        # Verify that the position of node 2 has changed
        self.assertTrue(
            drawing.x(2) != initial_x or drawing.y(2) != initial_y,
            "Node position should change after applying the algorithm"
        )

        # Verify that all coordinates are finite
        check_drawing_2d(self.line_graph, drawing)

    def test_run(self):
        """
        Test running the complete Kamada-Kawai algorithm
        """
        # Create a drawing
        drawing = eg.DrawingEuclidean2d.initial_placement(self.cycle_graph)

        # Record initial positions
        initial_positions = record_positions(drawing, self.cycle_graph)

        # Calculate initial energy
        initial_energy = calculate_energy(self.cycle_graph, drawing)

        # Create a KamadaKawai instance
        kk = eg.KamadaKawai(self.cycle_graph, lambda e: 1.0)

        # Run the algorithm
        kk.run(drawing)

        # Verify that positions have changed
        self.assertTrue(positions_changed(
            drawing, self.cycle_graph, initial_positions))

        # Verify that all coordinates are finite
        check_drawing_2d(self.cycle_graph, drawing)

        # Calculate final energy
        final_energy = calculate_energy(self.cycle_graph, drawing)

        # Verify that energy has decreased
        self.assertLess(final_energy, initial_energy)

        # For a cycle graph with uniform edge lengths, the layout should
        # approximate a regular polygon. Check that nodes are roughly
        # equidistant from the center.
        center_x = sum(drawing.x(u)
                       for u in self.cycle_graph.node_indices()) / len(self.cycle_nodes)
        center_y = sum(drawing.y(u)
                       for u in self.cycle_graph.node_indices()) / len(self.cycle_nodes)

        distances = []
        for u in self.cycle_graph.node_indices():
            dx = drawing.x(u) - center_x
            dy = drawing.y(u) - center_y
            distances.append(math.sqrt(dx * dx + dy * dy))

        # Calculate standard deviation of distances
        avg_distance = sum(distances) / len(distances)
        variance = sum((d - avg_distance) **
                       2 for d in distances) / len(distances)
        std_dev = math.sqrt(variance)

        # Check that the standard deviation is small relative to the average distance
        self.assertLess(std_dev / avg_distance, 0.2)

    def test_with_large_graph(self):
        """
        Test with a larger graph (Les Miserables)
        """
        # Create a drawing
        drawing = eg.DrawingEuclidean2d.initial_placement(self.les_mis_graph)

        # Record initial positions
        initial_positions = record_positions(drawing, self.les_mis_graph)

        # Create a KamadaKawai instance
        kk = eg.KamadaKawai(self.les_mis_graph, lambda e: 1.0)

        # Set a larger epsilon for faster convergence in tests
        kk.eps = 0.1

        # Run the algorithm
        kk.run(drawing)

        # Verify that positions have changed
        self.assertTrue(positions_changed(
            drawing, self.les_mis_graph, initial_positions))

        # Verify that all coordinates are finite
        check_drawing_2d(self.les_mis_graph, drawing)

    def test_with_custom_distance(self):
        """
        Test with a custom distance function
        """
        # Create a drawing
        drawing = eg.DrawingEuclidean2d.initial_placement(self.complete_graph)

        # Create a dictionary to store custom distances
        distances = {}
        for e in self.complete_graph.edge_indices():
            edge_data = self.complete_graph.edge_weight(e)
            i, j = edge_data
            distances[e] = abs(i - j)  # Distance based on node indices

        # Create a KamadaKawai instance with the custom distance function
        kk = eg.KamadaKawai(
            self.complete_graph, lambda e: distances[e])

        # Set a larger epsilon for faster convergence in tests
        kk.eps = 0.01

        # Run the algorithm
        kk.run(drawing)

        # Verify that all coordinates are finite
        check_drawing_2d(self.complete_graph, drawing)

        # Verify that nodes with smaller index differences are positioned closer together
        for i in range(len(self.complete_nodes)):
            for j in range(i + 1, len(self.complete_nodes)):
                for k in range(j + 1, len(self.complete_nodes)):
                    # Check if nodes i and j are closer than nodes i and k
                    # (since |i-j| < |i-k|)
                    dx_ij = drawing.x(i) - drawing.x(j)
                    dy_ij = drawing.y(i) - drawing.y(j)
                    dist_ij = math.sqrt(dx_ij * dx_ij + dy_ij * dy_ij)

                    dx_ik = drawing.x(i) - drawing.x(k)
                    dy_ik = drawing.y(i) - drawing.y(k)
                    dist_ik = math.sqrt(dx_ik * dx_ik + dy_ik * dy_ik)

                    self.assertLessEqual(dist_ij, dist_ik)


if __name__ == "__main__":
    unittest.main()
