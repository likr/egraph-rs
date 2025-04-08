import math
import unittest
import networkx as nx
import numpy as np
import egraph as eg
from tests.test_helpers import (
    create_line_graph,
    create_cycle_graph,
    create_complete_graph,
    create_star_graph,
    draw_networkx,
    check_drawing_2d,
    record_positions_2d,
    positions_changed_2d,
)


def calculate_overlap(graph, drawing, radius_func):
    """
    Calculate the total overlap between nodes in a drawing.

    Parameters:
        graph: An egraph Graph
        drawing: A DrawingEuclidean2d instance
        radius_func: A function that takes a node index and returns its radius

    Returns:
        The total overlap (sum of overlaps between all pairs of nodes)
    """
    total_overlap = 0.0
    for i in graph.node_indices():
        ri = radius_func(i)
        for j in graph.node_indices():
            if i < j:  # Only consider each pair once
                rj = radius_func(j)
                # Calculate Euclidean distance between centers
                dx = drawing.x(i) - drawing.x(j)
                dy = drawing.y(i) - drawing.y(j)
                distance = math.sqrt(dx * dx + dy * dy)

                # Calculate overlap (if any)
                min_distance = ri + rj
                if distance < min_distance:
                    overlap = min_distance - distance
                    total_overlap += overlap

    return total_overlap


class TestOverwrapRemoval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create some test graphs
        cls.line_graph, cls.line_nodes = create_line_graph(5)
        cls.cycle_graph, cls.cycle_nodes = create_cycle_graph(6)
        cls.complete_graph, cls.complete_nodes = create_complete_graph(4)
        cls.star_graph, cls.star_nodes = create_star_graph(8)
        cls.les_mis_graph = draw_networkx(nx.les_miserables_graph())

        # Define radius functions
        def uniform_radius(u, *args):
            return 0.1
        cls.uniform_radius = uniform_radius

        def index_based_radius(u, *args):
            return 0.05 + (u * 0.01)
        cls.index_based_radius = index_based_radius

        def central_node_larger(u, *args):
            return 0.2 if u == 0 else 0.1
        cls.central_node_larger = central_node_larger

    def test_constructor(self):
        """
        Test creating an OverwrapRemoval instance
        """
        # Create an OverwrapRemoval instance with uniform radius
        overwrap = eg.OverwrapRemoval(self.line_graph, self.uniform_radius)

        # Verify that the instance exists
        self.assertIsInstance(overwrap, eg.OverwrapRemoval)

    def test_parameters(self):
        """
        Test the parameters of the OverwrapRemoval algorithm
        """
        # Create an OverwrapRemoval instance
        overwrap = eg.OverwrapRemoval(self.line_graph, self.uniform_radius)

        # Check default values
        self.assertIsInstance(overwrap.strength, float)
        self.assertIsInstance(overwrap.iterations, int)
        self.assertIsInstance(overwrap.min_distance, float)

        # Default values should be finite numbers
        self.assertTrue(math.isfinite(overwrap.strength))
        self.assertTrue(overwrap.iterations > 0)
        self.assertTrue(math.isfinite(overwrap.min_distance))

        # Test setters
        new_strength = 0.5
        new_iterations = 10
        new_min_distance = 0.01

        overwrap.strength = new_strength
        overwrap.iterations = new_iterations
        overwrap.min_distance = new_min_distance

        # Verify values were updated
        self.assertAlmostEqual(overwrap.strength,
                               new_strength, delta=1e-6)
        self.assertEqual(overwrap.iterations, new_iterations)
        self.assertAlmostEqual(overwrap.min_distance,
                               new_min_distance, delta=1e-6)

    def test_apply_with_drawing_euclidean_2d(self):
        """
        Test applying the OverwrapRemoval algorithm to a 2D Euclidean drawing
        """
        # Create a drawing with overlapping nodes
        drawing = eg.DrawingEuclidean2d.initial_placement(self.line_graph)

        # Place nodes close together to ensure overlap
        for i in range(len(self.line_nodes)):
            drawing.set_x(i, i * 0.15)  # Nodes with radius 0.1 will overlap
            drawing.set_y(i, 0.0)

        # Record initial positions
        initial_positions = record_positions_2d(drawing, self.line_graph)

        # Calculate initial overlap
        initial_overlap = calculate_overlap(
            self.line_graph, drawing, self.uniform_radius)
        self.assertGreater(initial_overlap, 0.0,
                           "Initial setup should have overlapping nodes")

        # Create an OverwrapRemoval instance
        overwrap = eg.OverwrapRemoval(self.line_graph, self.uniform_radius)
        overwrap.iterations = 5  # Increase iterations for better results

        # Apply the algorithm
        overwrap.apply_with_drawing_euclidean_2d(drawing)

        # Verify that positions have changed
        self.assertTrue(positions_changed_2d(
            drawing, self.line_graph, initial_positions))

        # Verify that all coordinates are finite
        check_drawing_2d(self.line_graph, drawing)

        # Calculate final overlap
        final_overlap = calculate_overlap(
            self.line_graph, drawing, self.uniform_radius)

        # Verify that overlap has been reduced
        self.assertLess(final_overlap, initial_overlap,
                        "Overlap should be reduced after applying the algorithm")

    def test_apply_with_varying_radii(self):
        """
        Test applying the OverwrapRemoval algorithm with varying node radii
        """
        # Create a drawing with the star graph
        drawing = eg.DrawingEuclidean2d.initial_placement(self.star_graph)

        # Place nodes in a star pattern with overlaps
        drawing.set_x(0, 0.0)  # Center node
        drawing.set_y(0, 0.0)

        for i in range(1, len(self.star_nodes)):
            angle = 2 * math.pi * (i - 1) / (len(self.star_nodes) - 1)
            # Place nodes close to center to ensure overlap
            drawing.set_x(i, 0.15 * math.cos(angle))
            drawing.set_y(i, 0.15 * math.sin(angle))

        # Record initial positions
        initial_positions = record_positions_2d(drawing, self.star_graph)

        # Calculate initial overlap with central node larger
        initial_overlap = calculate_overlap(
            self.star_graph, drawing, self.central_node_larger)
        self.assertGreater(initial_overlap, 0.0,
                           "Initial setup should have overlapping nodes")

        # Create an OverwrapRemoval instance with central node larger
        overwrap = eg.OverwrapRemoval(
            self.star_graph, self.central_node_larger)
        overwrap.iterations = 5  # Increase iterations for better results

        # Apply the algorithm
        overwrap.apply_with_drawing_euclidean_2d(drawing)

        # Verify that positions have changed
        self.assertTrue(positions_changed_2d(
            drawing, self.star_graph, initial_positions))

        # Verify that all coordinates are finite
        check_drawing_2d(self.star_graph, drawing)

        # Calculate final overlap
        final_overlap = calculate_overlap(
            self.star_graph, drawing, self.central_node_larger)

        # Verify that overlap has been reduced
        self.assertLess(final_overlap, initial_overlap,
                        "Overlap should be reduced after applying the algorithm")

        # Verify that the central node is still at the center
        # (it should move less due to its larger radius)
        self.assertAlmostEqual(drawing.x(0), 0.0, delta=0.3)
        self.assertAlmostEqual(drawing.y(0), 0.0, delta=0.3)

    def test_apply_with_complete_graph(self):
        """
        Test applying the OverwrapRemoval algorithm to a complete graph
        """
        # Create a drawing with the complete graph
        drawing = eg.DrawingEuclidean2d.initial_placement(self.complete_graph)

        # Place nodes in a tight cluster to ensure overlap
        positions = [
            (0.0, 0.0),
            (0.1, 0.1),
            (0.0, 0.1),
            (0.1, 0.0)
        ]

        for i, (x, y) in enumerate(positions):
            drawing.set_x(i, x)
            drawing.set_y(i, y)

        # Record initial positions
        initial_positions = record_positions_2d(drawing, self.complete_graph)

        # Calculate initial overlap
        initial_overlap = calculate_overlap(
            self.complete_graph, drawing, self.uniform_radius)
        self.assertGreater(initial_overlap, 0.0,
                           "Initial setup should have overlapping nodes")

        # Create an OverwrapRemoval instance
        overwrap = eg.OverwrapRemoval(self.complete_graph, self.uniform_radius)
        overwrap.iterations = 10  # Increase iterations for better results

        # Apply the algorithm
        overwrap.apply_with_drawing_euclidean_2d(drawing)

        # Verify that positions have changed
        self.assertTrue(positions_changed_2d(
            drawing, self.complete_graph, initial_positions))

        # Verify that all coordinates are finite
        check_drawing_2d(self.complete_graph, drawing)

        # Calculate final overlap
        final_overlap = calculate_overlap(
            self.complete_graph, drawing, self.uniform_radius)

        # Verify that overlap has been reduced
        self.assertLess(final_overlap, initial_overlap,
                        "Overlap should be reduced after applying the algorithm")

    def test_with_large_graph(self):
        """
        Test with a larger graph (Les Miserables)
        """
        # Create a drawing
        drawing = eg.DrawingEuclidean2d.initial_placement(self.les_mis_graph)

        # Place nodes in a tight cluster to ensure overlap
        # This is necessary because the initial placement might not have overlaps
        center_x = 0.0
        center_y = 0.0
        node_indices = list(self.les_mis_graph.node_indices())
        num_nodes = len(node_indices)
        for i, u in enumerate(node_indices):
            # Place all nodes within a small radius around the center
            angle = 2 * math.pi * (i / num_nodes)
            drawing.set_x(u, center_x + 0.1 * math.cos(angle))
            drawing.set_y(u, center_y + 0.1 * math.sin(angle))

        # Record initial positions
        initial_positions = record_positions_2d(drawing, self.les_mis_graph)

        # Create an OverwrapRemoval instance
        overwrap = eg.OverwrapRemoval(self.les_mis_graph, self.uniform_radius)
        overwrap.iterations = 10  # Increase iterations for better results
        overwrap.strength = 1.0  # Increase strength for more movement

        # Apply the algorithm
        overwrap.apply_with_drawing_euclidean_2d(drawing)

        # Verify that positions have changed
        self.assertTrue(positions_changed_2d(
            drawing, self.les_mis_graph, initial_positions))

        # Verify that all coordinates are finite
        check_drawing_2d(self.les_mis_graph, drawing)

    def test_strength_parameter(self):
        """
        Test the effect of the strength parameter
        """
        # Create two identical drawings
        drawing1 = eg.DrawingEuclidean2d.initial_placement(self.line_graph)
        drawing2 = eg.DrawingEuclidean2d.initial_placement(self.line_graph)

        # Place nodes close together to ensure overlap
        for i in range(len(self.line_nodes)):
            x = i * 0.15  # Nodes with radius 0.1 will overlap
            y = 0.0
            drawing1.set_x(i, x)
            drawing1.set_y(i, y)
            drawing2.set_x(i, x)
            drawing2.set_y(i, y)

        # Create two OverwrapRemoval instances with different strengths
        overwrap1 = eg.OverwrapRemoval(self.line_graph, self.uniform_radius)
        overwrap1.strength = 0.1  # Low strength

        overwrap2 = eg.OverwrapRemoval(self.line_graph, self.uniform_radius)
        overwrap2.strength = 1.0  # High strength

        # Apply the algorithms
        overwrap1.apply_with_drawing_euclidean_2d(drawing1)
        overwrap2.apply_with_drawing_euclidean_2d(drawing2)

        # Calculate the total displacement for each drawing
        displacement1 = 0.0
        displacement2 = 0.0

        for i in range(len(self.line_nodes)):
            original_x = i * 0.15
            original_y = 0.0

            dx1 = drawing1.x(i) - original_x
            dy1 = drawing1.y(i) - original_y
            displacement1 += math.sqrt(dx1 * dx1 + dy1 * dy1)

            dx2 = drawing2.x(i) - original_x
            dy2 = drawing2.y(i) - original_y
            displacement2 += math.sqrt(dx2 * dx2 + dy2 * dy2)

        # Verify that higher strength causes more displacement
        self.assertLess(displacement1, displacement2,
                        "Higher strength should cause more displacement")

    def test_iterations_parameter(self):
        """
        Test the effect of the iterations parameter
        """
        # Create two identical drawings
        drawing1 = eg.DrawingEuclidean2d.initial_placement(self.line_graph)
        drawing2 = eg.DrawingEuclidean2d.initial_placement(self.line_graph)

        # Place nodes close together to ensure overlap
        for i in range(len(self.line_nodes)):
            x = i * 0.15  # Nodes with radius 0.1 will overlap
            y = 0.0
            drawing1.set_x(i, x)
            drawing1.set_y(i, y)
            drawing2.set_x(i, x)
            drawing2.set_y(i, y)

        # Create two OverwrapRemoval instances with different iteration counts
        overwrap1 = eg.OverwrapRemoval(self.line_graph, self.uniform_radius)
        overwrap1.iterations = 1  # Few iterations

        overwrap2 = eg.OverwrapRemoval(self.line_graph, self.uniform_radius)
        overwrap2.iterations = 5  # More iterations

        # Apply the algorithms
        overwrap1.apply_with_drawing_euclidean_2d(drawing1)
        overwrap2.apply_with_drawing_euclidean_2d(drawing2)

        # Calculate final overlap for each drawing
        overlap1 = calculate_overlap(
            self.line_graph, drawing1, self.uniform_radius)
        overlap2 = calculate_overlap(
            self.line_graph, drawing2, self.uniform_radius)

        # Verify that more iterations result in less overlap
        self.assertLessEqual(overlap2, overlap1,
                             "More iterations should result in less overlap")

    def test_min_distance_parameter(self):
        """
        Test the effect of the min_distance parameter
        """
        # Create two identical drawings
        drawing1 = eg.DrawingEuclidean2d.initial_placement(self.line_graph)
        drawing2 = eg.DrawingEuclidean2d.initial_placement(self.line_graph)

        # Place nodes close together to ensure overlap
        for i in range(len(self.line_nodes)):
            x = i * 0.15  # Nodes with radius 0.1 will overlap
            y = 0.0
            drawing1.set_x(i, x)
            drawing1.set_y(i, y)
            drawing2.set_x(i, x)
            drawing2.set_y(i, y)

        # Create two OverwrapRemoval instances with different min_distance values
        overwrap1 = eg.OverwrapRemoval(self.line_graph, self.uniform_radius)
        overwrap1.min_distance = 0.001  # Default value

        overwrap2 = eg.OverwrapRemoval(self.line_graph, self.uniform_radius)
        overwrap2.min_distance = 0.1  # Larger value

        # Apply the algorithms
        overwrap1.apply_with_drawing_euclidean_2d(drawing1)
        overwrap2.apply_with_drawing_euclidean_2d(drawing2)

        # Calculate the average distance between adjacent nodes
        avg_distance1 = 0.0
        avg_distance2 = 0.0

        for i in range(len(self.line_nodes) - 1):
            dx1 = drawing1.x(i) - drawing1.x(i + 1)
            dy1 = drawing1.y(i) - drawing1.y(i + 1)
            avg_distance1 += math.sqrt(dx1 * dx1 + dy1 * dy1)

            dx2 = drawing2.x(i) - drawing2.x(i + 1)
            dy2 = drawing2.y(i) - drawing2.y(i + 1)
            avg_distance2 += math.sqrt(dx2 * dx2 + dy2 * dy2)

        avg_distance1 /= (len(self.line_nodes) - 1)
        avg_distance2 /= (len(self.line_nodes) - 1)

        # Verify that larger min_distance results in greater spacing
        self.assertLessEqual(avg_distance1, avg_distance2,
                             "Larger min_distance should result in greater spacing between nodes")


if __name__ == "__main__":
    unittest.main()
