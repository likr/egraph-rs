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


def calculate_stress(graph, drawing, distance_func=None):
    """
    Calculate the stress of a drawing
    """
    if distance_func is None:
        def distance_func(e): return 1.0

    stress = 0.0
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

                # Calculate weight (typically 1/d^2)
                weight = 1.0 / (ideal_distance * ideal_distance)

                # Add to stress
                diff = actual_distance - ideal_distance
                stress += weight * diff * diff

    return stress


class TestStressMajorization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create some test graphs
        cls.line_graph, cls.line_nodes = create_line_graph(5)
        cls.cycle_graph, cls.cycle_nodes = create_cycle_graph(6)
        cls.complete_graph, cls.complete_nodes = create_complete_graph(4)
        cls.les_mis_graph = draw(nx.les_miserables_graph())

    def test_constructor(self):
        """
        Test creating a StressMajorization instance from a graph
        """
        # Create a drawing
        drawing = eg.DrawingEuclidean2d.initial_placement(self.line_graph)

        # Create a StressMajorization instance
        sm = eg.StressMajorization(self.line_graph, drawing, lambda e: 1.0)

        # Verify that the instance exists
        self.assertIsInstance(sm, eg.StressMajorization)

    def test_constructor_with_distance_matrix(self):
        """
        Test creating a StressMajorization instance from a distance matrix
        """
        # Create a drawing
        drawing = eg.DrawingEuclidean2d.initial_placement(self.line_graph)

        # Create a distance matrix using all_sources_dijkstra
        distance_matrix = eg.all_sources_dijkstra(
            self.line_graph, lambda e: 1.0)

        # Create a StressMajorization instance
        sm = eg.StressMajorization.with_distance_matrix(
            drawing, distance_matrix)

        # Verify that the instance exists
        self.assertIsInstance(sm, eg.StressMajorization)

    def test_apply(self):
        """
        Test applying a single iteration of the stress majorization algorithm
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

        # Record initial positions
        initial_positions = record_positions(drawing, self.line_graph)

        # Create a StressMajorization instance
        sm = eg.StressMajorization(self.line_graph, drawing, lambda e: 1.0)

        # Apply a single iteration
        stress = sm.apply(drawing)

        # Verify that the stress value is a finite number
        self.assertTrue(math.isfinite(stress))

        # Verify that positions have changed
        self.assertTrue(positions_changed(
            drawing, self.line_graph, initial_positions))

        # Verify that all coordinates are finite
        check_drawing_2d(self.line_graph, drawing)

    def test_run(self):
        """
        Test running the complete stress majorization algorithm
        """
        # Create a drawing
        drawing = eg.DrawingEuclidean2d.initial_placement(self.cycle_graph)

        # Record initial positions
        initial_positions = record_positions(drawing, self.cycle_graph)

        # Calculate initial stress
        initial_stress = calculate_stress(self.cycle_graph, drawing)

        # Create a StressMajorization instance
        sm = eg.StressMajorization(self.cycle_graph, drawing, lambda e: 1.0)

        # Run the algorithm
        sm.run(drawing)

        # Verify that positions have changed
        self.assertTrue(positions_changed(
            drawing, self.cycle_graph, initial_positions))

        # Verify that all coordinates are finite
        check_drawing_2d(self.cycle_graph, drawing)

        # Calculate final stress
        final_stress = calculate_stress(self.cycle_graph, drawing)

        # Verify that stress has decreased
        self.assertLess(final_stress, initial_stress)

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

    def test_update_weight(self):
        """
        Test updating the weight matrix
        """
        # Create a drawing
        drawing = eg.DrawingEuclidean2d.initial_placement(self.line_graph)

        # Create a StressMajorization instance
        sm = eg.StressMajorization(self.line_graph, drawing, lambda e: 1.0)

        # Update the weight matrix
        sm.update_weight(lambda i, j, dij, wij: 2.0 * wij)

        # Apply the algorithm and verify it still works
        stress = sm.apply(drawing)
        self.assertTrue(math.isfinite(stress))

        # Verify that all coordinates are finite
        check_drawing_2d(self.line_graph, drawing)

    def test_with_large_graph(self):
        """
        Test with a larger graph (Les Miserables)
        """
        # Create a drawing
        drawing = eg.DrawingEuclidean2d.initial_placement(self.les_mis_graph)

        # Record initial positions
        initial_positions = record_positions(drawing, self.les_mis_graph)

        # Create a StressMajorization instance
        sm = eg.StressMajorization(self.les_mis_graph, drawing, lambda e: 1.0)

        # Apply a single iteration
        stress = sm.apply(drawing)

        # Verify that the stress value is a finite number
        self.assertTrue(math.isfinite(stress))

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

        # Create a StressMajorization instance with the custom distance function
        sm = eg.StressMajorization(
            self.complete_graph, drawing, lambda e: distances[e])

        # Run the algorithm
        sm.run(drawing)

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

    def test_parameters(self):
        """
        Test the epsilon and max_iterations parameters
        """
        # Create a drawing
        drawing = eg.DrawingEuclidean2d.initial_placement(self.line_graph)

        # Create a StressMajorization instance
        sm = eg.StressMajorization(self.line_graph, drawing, lambda e: 1.0)

        # Check default values
        self.assertIsInstance(sm.epsilon, float)
        self.assertIsInstance(sm.max_iterations, int)

        # Default values should be finite numbers
        self.assertTrue(math.isfinite(sm.epsilon))
        self.assertTrue(sm.max_iterations > 0)

        # Test setters
        new_epsilon = 1e-6
        new_max_iterations = 200

        sm.epsilon = new_epsilon
        sm.max_iterations = new_max_iterations

        # Verify values were updated - use approximate comparison for floating point
        self.assertAlmostEqual(sm.epsilon, new_epsilon, delta=1e-10)
        self.assertEqual(sm.max_iterations, new_max_iterations)

        # Test that the algorithm respects these parameters
        # First, set a very high epsilon to ensure quick convergence
        sm.epsilon = 0.5  # Very high value
        sm.max_iterations = 1000

        # Record initial positions
        initial_positions = record_positions(drawing, self.line_graph)

        # Run the algorithm
        sm.run(drawing)

        # Verify that positions have changed
        self.assertTrue(positions_changed(
            drawing, self.line_graph, initial_positions))

        # Now, set a very low epsilon and a low max_iterations
        # to ensure the algorithm stops due to max_iterations
        drawing = eg.DrawingEuclidean2d.initial_placement(self.line_graph)
        sm.epsilon = 1e-10  # Very low value
        sm.max_iterations = 1  # Only one iteration

        # Record initial positions
        initial_positions = record_positions(drawing, self.line_graph)

        # Run the algorithm
        sm.run(drawing)

        # Verify that positions have changed
        self.assertTrue(positions_changed(
            drawing, self.line_graph, initial_positions))

        # Verify that all coordinates are finite
        check_drawing_2d(self.line_graph, drawing)


if __name__ == "__main__":
    unittest.main()
