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
    create_grid_graph,
    check_drawing_nd,
    record_positions_nd,
    positions_changed_nd,
    verify_node_positions,
    verify_layout_quality
)


class TestDrawingEuclidean(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create some test graphs
        cls.line_graph, cls.line_nodes = create_line_graph(5)
        cls.cycle_graph, cls.cycle_nodes = create_cycle_graph(6)
        cls.complete_graph, cls.complete_nodes = create_complete_graph(4)
        cls.star_graph, cls.star_nodes = create_star_graph(5)
        cls.grid_graph, cls.grid_nodes = create_grid_graph(3, 3)

        # Create Les Miserables graph
        les_mis_graph = nx.les_miserables_graph()
        cls.les_mis = eg.Graph()
        cls.les_mis_indices = {}
        for u in les_mis_graph.nodes:
            cls.les_mis_indices[u] = cls.les_mis.add_node(u)
        for u, v in les_mis_graph.edges:
            cls.les_mis.add_edge(
                cls.les_mis_indices[u], cls.les_mis_indices[v], (u, v))

    def test_constructor_3d(self):
        """
        Test creating a 3D DrawingEuclidean instance using ClassicalMds
        """
        # Create a 3D drawing using ClassicalMds
        mds = eg.ClassicalMds(self.line_graph, lambda e: 1.0)
        drawing = mds.run(3)  # 3 dimensions

        # Verify that the instance exists and is of the correct type
        self.assertIsInstance(drawing, eg.DrawingEuclidean)

        # Verify initial state - check that all nodes have valid coordinates
        check_drawing_nd(self.line_graph, drawing, 3)

    def test_constructor_higher_dimensions(self):
        """
        Test creating DrawingEuclidean instances with higher dimensions
        """
        dimensions_to_test = [4, 5, 10]  # Test 4D, 5D, and 10D

        for dimensions in dimensions_to_test:
            # Create a drawing with the specified dimensions
            mds = eg.ClassicalMds(self.line_graph, lambda e: 1.0)
            drawing = mds.run(dimensions)

            # Verify that the instance exists and is of the correct type
            self.assertIsInstance(drawing, eg.DrawingEuclidean)

            # Verify initial state - check that all nodes have valid coordinates
            check_drawing_nd(self.line_graph, drawing, dimensions)

    def test_node_coordinates(self):
        """
        Test node coordinate operations (get/set) in different dimensions
        """
        # Create a 3D drawing
        mds = eg.ClassicalMds(self.line_graph, lambda e: 1.0)
        drawing = mds.run(3)  # 3 dimensions

        # Verify initial coordinates are finite numbers
        check_drawing_nd(self.line_graph, drawing, 3)

        # Test setting coordinates for the first node in each dimension
        new_coords = [10.5, -5.25, 7.75]
        for d in range(3):
            drawing.set(self.line_nodes[0], d, new_coords[d])

        # Verify coordinates were set correctly
        for d in range(3):
            self.assertAlmostEqual(drawing.get(self.line_nodes[0], d), new_coords[d], delta=1e-8,
                                   msg=f"Coordinate in dimension {d} should be updated")

        # Test setting coordinates for another node
        new_coords2 = [-3.5, 8.25, -12.75]
        for d in range(3):
            drawing.set(self.line_nodes[1], d, new_coords2[d])

        # Verify coordinates were set correctly
        for d in range(3):
            self.assertAlmostEqual(drawing.get(self.line_nodes[1], d), new_coords2[d], delta=1e-8,
                                   msg=f"Coordinate in dimension {d} should be updated")

        # Test getting coordinates for non-existent node
        # In the current implementation, get() returns None for non-existent nodes
        self.assertIsNone(drawing.get(999, 0),
                          "Coordinate for non-existent node should be None")
        self.assertIsNone(drawing.get(999, 1),
                          "Coordinate for non-existent node should be None")
        self.assertIsNone(drawing.get(999, 2),
                          "Coordinate for non-existent node should be None")

        # Test getting coordinates for non-existent dimension
        # In the current implementation, get() returns None for non-existent dimensions
        self.assertIsNone(drawing.get(self.line_nodes[0], 999),
                          "Coordinate for non-existent dimension should be None")

    def test_higher_dimensions(self):
        """
        Test with higher dimensions (5D and 10D)
        """
        dimensions_to_test = [5, 10]

        for dimensions in dimensions_to_test:
            # Create a drawing with the specified dimensions
            mds = eg.ClassicalMds(self.line_graph, lambda e: 1.0)
            drawing = mds.run(dimensions)

            # Verify all coordinates are finite
            check_drawing_nd(self.line_graph, drawing, dimensions)

            # Test setting and getting coordinates in all dimensions
            for d in range(dimensions):
                value = float(d) * 1.5  # Different value for each dimension
                drawing.set(self.line_nodes[0], d, value)
                self.assertAlmostEqual(drawing.get(self.line_nodes[0], d), value, delta=1e-8,
                                       msg=f"Coordinate in dimension {d} should be updated")

    def test_drawing_with_graph(self):
        """
        Test integration with Graph class
        """
        # Create a more complex graph with a specific structure
        graph = eg.Graph()
        nodes = []

        # Create nodes
        for i in range(5):
            nodes.append(graph.add_node(i))

        # Create a simple graph structure
        #    0
        #   / \
        #  1---2
        #  |   |
        #  3---4
        graph.add_edge(nodes[0], nodes[1], (0, 1))
        graph.add_edge(nodes[0], nodes[2], (0, 2))
        graph.add_edge(nodes[1], nodes[2], (1, 2))
        graph.add_edge(nodes[1], nodes[3], (1, 3))
        graph.add_edge(nodes[2], nodes[4], (2, 4))
        graph.add_edge(nodes[3], nodes[4], (3, 4))

        # Create a 3D drawing
        mds = eg.ClassicalMds(graph, lambda e: 1.0)
        drawing = mds.run(3)

        # Verify that all nodes have valid coordinates
        check_drawing_nd(graph, drawing, 3)

        # Note: In the current implementation, adding a node to the graph after creating
        # the drawing doesn't automatically add the node to the drawing's internal data structure.
        # This is expected behavior, as the drawing is created based on the graph's state at creation time.

        # Instead, let's test modifying coordinates of existing nodes
        for d in range(3):
            new_value = float(d) * 2.5
            drawing.set(nodes[0], d, new_value)
            self.assertAlmostEqual(drawing.get(nodes[0], d), new_value, delta=1e-8,
                                   msg=f"Node coordinate in dimension {d} should match what we set")

    def test_with_classical_mds(self):
        """
        Test with ClassicalMds layout algorithm
        """
        dimensions_to_test = [3, 4, 5]

        for dimensions in dimensions_to_test:
            # Create a drawing with ClassicalMds
            mds = eg.ClassicalMds(self.cycle_graph, lambda e: 1.0)
            drawing = mds.run(dimensions)

            # Verify that all nodes have valid coordinates
            check_drawing_nd(self.cycle_graph, drawing, dimensions)

            # For a cycle graph with uniform edge lengths, nodes should be roughly
            # equidistant from the center in the first 2 dimensions
            # (This is a property of MDS for cycle graphs)
            center = [0.0] * dimensions
            for u in self.cycle_graph.node_indices():
                for d in range(dimensions):
                    center[d] += drawing.get(u, d)

            for d in range(dimensions):
                center[d] /= len(self.cycle_nodes)

            distances = []
            for u in self.cycle_graph.node_indices():
                squared_dist = 0.0
                for d in range(dimensions):
                    diff = drawing.get(u, d) - center[d]
                    squared_dist += diff * diff
                distances.append(math.sqrt(squared_dist))

            # Calculate standard deviation of distances
            avg_distance = sum(distances) / len(distances)
            variance = sum((d - avg_distance) **
                           2 for d in distances) / len(distances)
            std_dev = math.sqrt(variance)

            # Check that the standard deviation is small relative to the average distance
            # This is less strict for higher dimensions
            self.assertLess(std_dev / avg_distance, 0.3,
                            f"Nodes should be roughly equidistant from center (dimensions={dimensions})")

    def test_with_large_graph(self):
        """
        Test with a larger graph (Les Miserables)
        """
        dimensions_to_test = [3, 5]

        for dimensions in dimensions_to_test:
            # Create a drawing with ClassicalMds
            mds = eg.ClassicalMds(self.les_mis, lambda e: 1.0)
            drawing = mds.run(dimensions)

            # Verify that all nodes have valid coordinates
            check_drawing_nd(self.les_mis, drawing, dimensions)

            # Record positions
            initial_positions = record_positions_nd(
                drawing, self.les_mis, dimensions)

            # Apply a small random perturbation to the positions
            for u in self.les_mis.node_indices():
                for d in range(dimensions):
                    current = drawing.get(u, d)
                    drawing.set(u, d, current + 0.01 *
                                (hash(f"{u}_{d}") % 100))

            # Verify that positions have changed
            self.assertTrue(positions_changed_nd(drawing, self.les_mis, initial_positions, dimensions),
                            "Positions should change after perturbation")

    def test_coordinate_validation(self):
        """
        Test coordinate validation
        """
        # Create a 3D drawing
        mds = eg.ClassicalMds(self.line_graph, lambda e: 1.0)
        drawing = mds.run(3)

        # Test setting coordinates to extreme values
        extreme_values = [
            float('inf'),  # Infinity
            float('-inf'),  # Negative infinity
            float('nan'),  # Not a number
            1e30,  # Very large number
            -1e30,  # Very large negative number
            0.0,  # Zero
            1e-30  # Very small number
        ]

        for value in extreme_values:
            # Skip NaN as it can't be compared with assertAlmostEqual
            if math.isnan(value):
                continue

            drawing.set(self.line_nodes[0], 0, value)

            # The implementation should accept any float value
            if math.isfinite(value):
                self.assertAlmostEqual(drawing.get(self.line_nodes[0], 0), value, delta=abs(value) * 1e-6,
                                       msg=f"Coordinate should be updated to {value}")
            else:
                # For infinite values, we can only check if they're the same infinity
                self.assertEqual(math.isinf(drawing.get(self.line_nodes[0], 0)), math.isinf(value),
                                 msg=f"Infinity status should match for {value}")
                if math.isinf(value):
                    self.assertEqual(drawing.get(self.line_nodes[0], 0) > 0, value > 0,
                                     msg=f"Sign of infinity should match for {value}")


if __name__ == "__main__":
    unittest.main()
