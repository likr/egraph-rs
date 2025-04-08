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
    check_drawing_2d,
    record_positions_2d,
    positions_changed_2d,
    verify_node_positions,
    verify_connected_nodes_closer,
    verify_layout_quality
)


class TestDrawingTorus2d(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create some test graphs
        cls.line_graph, cls.line_nodes = create_line_graph(5)
        cls.cycle_graph, cls.cycle_nodes = create_cycle_graph(6)
        cls.complete_graph, cls.complete_nodes = create_complete_graph(4)
        cls.star_graph, cls.star_nodes = create_star_graph(5)
        cls.grid_graph, cls.grid_nodes = create_grid_graph(3, 3)

    def test_constructor(self):
        """
        Test creating a DrawingTorus2d instance with initial placement
        """
        # Create a drawing with initial placement
        drawing = eg.DrawingTorus2d.initial_placement(self.line_graph)

        # Verify that the instance exists
        self.assertIsInstance(drawing, eg.DrawingTorus2d)

        # Verify initial state
        self.assertEqual(drawing.len(), len(self.line_nodes),
                         "Drawing should have the same number of nodes as the graph")

        # Verify all coordinates are finite
        check_drawing_2d(self.line_graph, drawing)

    def test_node_coordinates(self):
        """
        Test node coordinate operations (get/set x,y)
        """
        # Create a drawing with initial placement
        drawing = eg.DrawingTorus2d.initial_placement(self.line_graph)

        # Verify initial coordinates are finite numbers
        check_drawing_2d(self.line_graph, drawing)

        # Test setting coordinates for the first node
        new_x1 = 0.5
        new_y1 = 0.25
        drawing.set_x(self.line_nodes[0], new_x1)
        drawing.set_y(self.line_nodes[0], new_y1)

        # Verify coordinates were set correctly
        self.assertAlmostEqual(drawing.x(self.line_nodes[0]), new_x1, delta=1e-8,
                               msg="X coordinate should be updated")
        self.assertAlmostEqual(drawing.y(self.line_nodes[0]), new_y1, delta=1e-8,
                               msg="Y coordinate should be updated")

        # Test setting coordinates for another node
        new_x2 = 0.75
        new_y2 = 0.75
        drawing.set_x(self.line_nodes[1], new_x2)
        drawing.set_y(self.line_nodes[1], new_y2)

        # Verify coordinates were set correctly
        self.assertAlmostEqual(drawing.x(self.line_nodes[1]), new_x2, delta=1e-8,
                               msg="X coordinate should be updated")
        self.assertAlmostEqual(drawing.y(self.line_nodes[1]), new_y2, delta=1e-8,
                               msg="Y coordinate should be updated")

        # Test getting coordinates for non-existent node
        # In the current implementation, x() and y() return None for non-existent nodes
        self.assertIsNone(drawing.x(999),
                          "X coordinate for non-existent node should be None")
        self.assertIsNone(drawing.y(999),
                          "Y coordinate for non-existent node should be None")

    def test_torus_wrapping(self):
        """
        Test torus wrapping behavior (coordinates wrapping around)
        """
        # Create a drawing with initial placement
        drawing = eg.DrawingTorus2d.initial_placement(self.line_graph)

        # Test wrapping behavior for x-coordinate
        # Set x-coordinate to a value > 1
        drawing.set_x(self.line_nodes[0], 1.25)
        # On a torus, 1.25 should wrap to 0.25
        self.assertAlmostEqual(drawing.x(self.line_nodes[0]), 0.25, delta=1e-8,
                               msg="X-coordinate > 1 should wrap around to [0,1] range")

        # Set x-coordinate to a negative value
        drawing.set_x(self.line_nodes[0], -0.25)
        # On a torus, -0.25 should wrap to 0.75
        self.assertAlmostEqual(drawing.x(self.line_nodes[0]), 0.75, delta=1e-8,
                               msg="Negative x-coordinate should wrap around to [0,1] range")

        # Test wrapping behavior for y-coordinate
        # Set y-coordinate to a value > 1
        drawing.set_y(self.line_nodes[0], 1.75)
        # On a torus, 1.75 should wrap to 0.75
        self.assertAlmostEqual(drawing.y(self.line_nodes[0]), 0.75, delta=1e-8,
                               msg="Y-coordinate > 1 should wrap around to [0,1] range")

        # Set y-coordinate to a negative value
        drawing.set_y(self.line_nodes[0], -0.5)
        # On a torus, -0.5 should wrap to 0.5
        self.assertAlmostEqual(drawing.y(self.line_nodes[0]), 0.5, delta=1e-8,
                               msg="Negative y-coordinate should wrap around to [0,1] range")

        # Test with multiple wraps
        drawing.set_x(self.line_nodes[0], 3.25)
        # 3.25 should wrap to 0.25 (3.25 % 1 = 0.25)
        self.assertAlmostEqual(drawing.x(self.line_nodes[0]), 0.25, delta=1e-8,
                               msg="X-coordinate with multiple wraps should normalize correctly")

        drawing.set_y(self.line_nodes[0], -2.75)
        # -2.75 should wrap to 0.25 (-2.75 % 1 = -0.75, then 1 - 0.75 = 0.25)
        self.assertAlmostEqual(drawing.y(self.line_nodes[0]), 0.25, delta=1e-8,
                               msg="Negative y-coordinate with multiple wraps should normalize correctly")

    def test_edge_segments(self):
        """
        Test edge segment representation
        """
        # Create a drawing with initial placement
        drawing = eg.DrawingTorus2d.initial_placement(self.line_graph)

        # Set specific coordinates for testing
        # Place nodes at positions that don't cross the boundary
        drawing.set_x(self.line_nodes[0], 0.25)
        drawing.set_y(self.line_nodes[0], 0.25)
        drawing.set_x(self.line_nodes[1], 0.75)
        drawing.set_y(self.line_nodes[1], 0.75)

        # Test edge segments
        segments = drawing.edge_segments(
            self.line_nodes[0], self.line_nodes[1])

        # Verify segments exist
        self.assertIsNotNone(segments, "Edge segments should exist")
        self.assertGreater(
            len(segments), 0, "There should be at least one segment for the edge")

        # For nodes that don't cross the boundary, there should be exactly one segment
        self.assertEqual(len(segments), 1,
                         "There should be exactly one segment for nodes that don't cross the boundary")

        # Verify segment structure
        segment = segments[0]
        self.assertIsInstance(segment, tuple, "Segment should be a tuple")
        self.assertEqual(
            len(segment), 2, "Segment should have two points (start and end)")

        # Verify points structure
        start_point, end_point = segment
        self.assertIsInstance(start_point, tuple,
                              "Start point should be a tuple")
        self.assertIsInstance(end_point, tuple, "End point should be a tuple")
        self.assertEqual(len(start_point), 2,
                         "Start point should have x,y coordinates")
        self.assertEqual(len(end_point), 2,
                         "End point should have x,y coordinates")

        # Verify coordinates match node positions
        # The segment might be in either direction
        if abs(start_point[0] - drawing.x(self.line_nodes[0])) < 1e-8 and abs(start_point[1] - drawing.y(self.line_nodes[0])) < 1e-8:
            # First point matches first node
            self.assertAlmostEqual(start_point[0], drawing.x(self.line_nodes[0]), delta=1e-8,
                                   msg="Start point x should match node1 x")
            self.assertAlmostEqual(start_point[1], drawing.y(self.line_nodes[0]), delta=1e-8,
                                   msg="Start point y should match node1 y")
            self.assertAlmostEqual(end_point[0], drawing.x(self.line_nodes[1]), delta=1e-8,
                                   msg="End point x should match node2 x")
            self.assertAlmostEqual(end_point[1], drawing.y(self.line_nodes[1]), delta=1e-8,
                                   msg="End point y should match node2 y")
        else:
            # First point matches second node
            self.assertAlmostEqual(start_point[0], drawing.x(self.line_nodes[1]), delta=1e-8,
                                   msg="Start point x should match node2 x")
            self.assertAlmostEqual(start_point[1], drawing.y(self.line_nodes[1]), delta=1e-8,
                                   msg="Start point y should match node2 y")
            self.assertAlmostEqual(end_point[0], drawing.x(self.line_nodes[0]), delta=1e-8,
                                   msg="End point x should match node1 x")
            self.assertAlmostEqual(end_point[1], drawing.y(self.line_nodes[0]), delta=1e-8,
                                   msg="End point y should match node1 y")

        # Now test with nodes that cross the boundary
        # Place nodes on opposite sides of the torus
        drawing.set_x(self.line_nodes[0], 0.05)
        drawing.set_y(self.line_nodes[0], 0.05)
        drawing.set_x(self.line_nodes[1], 0.95)
        drawing.set_y(self.line_nodes[1], 0.95)

        # Get edge segments
        wrapping_segments = drawing.edge_segments(
            self.line_nodes[0], self.line_nodes[1])

        # Verify segments exist
        self.assertIsNotNone(
            wrapping_segments, "Edge segments should exist for wrapping edge")
        self.assertGreater(
            len(wrapping_segments), 0, "There should be at least one segment for the wrapping edge")

        # For nodes that cross the boundary on a torus, there may be multiple segments
        # The exact number depends on the implementation, but we can verify they exist
        self.assertGreaterEqual(
            len(wrapping_segments), 1, "There should be at least one segment for the wrapping edge")

        # Verify each segment is properly formatted
        for segment in wrapping_segments:
            self.assertIsInstance(
                segment, tuple, "Each segment should be a tuple")
            self.assertEqual(
                len(segment), 2, "Each segment should have two points")
            self.assertIsInstance(
                segment[0], tuple, "First point of each segment should be a tuple")
            self.assertIsInstance(
                segment[1], tuple, "Second point of each segment should be a tuple")
            self.assertEqual(
                len(segment[0]), 2, "First point should have x,y coordinates")
            self.assertEqual(
                len(segment[1]), 2, "Second point should have x,y coordinates")

        # Test edge segments for non-existent edge
        # In the current implementation, edge_segments() returns None for non-existent edges
        non_existent_segments = drawing.edge_segments(self.line_nodes[0], 999)
        self.assertIsNone(non_existent_segments,
                          "Edge segments for non-existent edge should be None")

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

        # Create a drawing with initial placement
        drawing = eg.DrawingTorus2d.initial_placement(graph)

        # Verify that the drawing has the correct number of nodes
        self.assertEqual(drawing.len(), len(nodes),
                         "Drawing should have the same number of nodes as the graph")

        # Verify all nodes have valid coordinates
        check_drawing_2d(graph, drawing)

        # Test edge segments for all edges
        for e in graph.edge_indices():
            u, v = graph.edge_endpoints(e)
            segments = drawing.edge_segments(u, v)
            self.assertIsNotNone(segments, f"Edge {e} should have segments")
            self.assertGreater(
                len(segments), 0, f"Edge {e} should have at least one segment")

        # Record initial positions
        initial_positions = record_positions_2d(drawing, graph)

        # Note: KamadaKawai layout doesn't support torus drawings directly
        # Instead, we'll manually modify some node positions to simulate a layout change

        # Modify node positions manually
        drawing.set_x(nodes[0], 0.3)
        drawing.set_y(nodes[0], 0.2)
        drawing.set_x(nodes[1], 0.7)
        drawing.set_y(nodes[1], 0.1)
        drawing.set_x(nodes[2], 0.5)
        drawing.set_y(nodes[2], 0.8)

        # Verify that positions have changed
        self.assertTrue(positions_changed_2d(drawing, graph, initial_positions),
                        "Node positions should change after manual modification")

        # Verify all nodes have valid coordinates after modification
        check_drawing_2d(graph, drawing)

        # Verify all nodes are still within the valid range [0,1]
        for u in graph.node_indices():
            x = drawing.x(u)
            y = drawing.y(u)
            self.assertGreaterEqual(
                x, 0, f"Node {u} x-coordinate should be >= 0")
            self.assertLessEqual(x, 1, f"Node {u} x-coordinate should be <= 1")
            self.assertGreaterEqual(
                y, 0, f"Node {u} y-coordinate should be >= 0")
            self.assertLessEqual(y, 1, f"Node {u} y-coordinate should be <= 1")

        # Add a new node after creating the drawing
        new_node = graph.add_node(5)

        # Verify the drawing doesn't have the new node yet
        self.assertIsNone(drawing.x(new_node),
                          "New node should not have coordinates in the drawing yet")

        # Create a new drawing with the updated graph
        updated_drawing = eg.DrawingTorus2d.initial_placement(graph)

        # Verify the new drawing includes the new node
        self.assertEqual(updated_drawing.len(), graph.node_count(),
                         "Updated drawing should include the new node")
        self.assertIsNotNone(updated_drawing.x(new_node),
                             "New node should have a valid x coordinate in the updated drawing")

    def test_coordinate_validation(self):
        """
        Test coordinate validation and normalization
        """
        # Create a drawing with initial placement
        drawing = eg.DrawingTorus2d.initial_placement(self.line_graph)

        # Test valid values within range
        valid_x = 0.5
        valid_y = 0.75

        drawing.set_x(self.line_nodes[0], valid_x)
        drawing.set_y(self.line_nodes[0], valid_y)

        # Verify coordinates were set correctly
        self.assertAlmostEqual(drawing.x(self.line_nodes[0]), valid_x, delta=1e-8,
                               msg="Valid x-coordinate should be set correctly")
        self.assertAlmostEqual(drawing.y(self.line_nodes[0]), valid_y, delta=1e-8,
                               msg="Valid y-coordinate should be set correctly")

        # Test boundary values
        drawing.set_x(self.line_nodes[0], 0)
        drawing.set_y(self.line_nodes[0], 0)

        # Verify coordinates were set correctly
        self.assertAlmostEqual(drawing.x(self.line_nodes[0]), 0, delta=1e-8,
                               msg="X-coordinate at lower boundary should be set correctly")
        self.assertAlmostEqual(drawing.y(self.line_nodes[0]), 0, delta=1e-8,
                               msg="Y-coordinate at lower boundary should be set correctly")

        drawing.set_x(self.line_nodes[0], 1)
        drawing.set_y(self.line_nodes[0], 1)

        # Verify coordinates were set correctly
        # Note: On a torus, 1 might be normalized to 0 since they're equivalent
        x_result = drawing.x(self.line_nodes[0])
        y_result = drawing.y(self.line_nodes[0])
        self.assertTrue(abs(x_result - 1) < 1e-8 or abs(x_result) < 1e-8,
                        "X-coordinate at upper boundary should be set correctly or normalized")
        self.assertTrue(abs(y_result - 1) < 1e-8 or abs(y_result) < 1e-8,
                        "Y-coordinate at upper boundary should be set correctly or normalized")

        # Test setting coordinates for multiple nodes
        drawing.set_x(self.line_nodes[0], 0.125)  # 1/8
        drawing.set_y(self.line_nodes[0], 0.25)   # 1/4
        drawing.set_x(self.line_nodes[1], 0.375)  # 3/8
        drawing.set_y(self.line_nodes[1], 0.5)    # 1/2

        # Verify coordinates were set correctly
        self.assertAlmostEqual(drawing.x(self.line_nodes[0]), 0.125, delta=1e-8,
                               msg="Node1 x-coordinate should be set correctly")
        self.assertAlmostEqual(drawing.y(self.line_nodes[0]), 0.25, delta=1e-8,
                               msg="Node1 y-coordinate should be set correctly")
        self.assertAlmostEqual(drawing.x(self.line_nodes[1]), 0.375, delta=1e-8,
                               msg="Node2 x-coordinate should be set correctly")
        self.assertAlmostEqual(drawing.y(self.line_nodes[1]), 0.5, delta=1e-8,
                               msg="Node2 y-coordinate should be set correctly")

        # Test that setting one node's coordinates doesn't affect others
        drawing.set_x(self.line_nodes[0], 0.875)  # 7/8

        # Verify only the intended node's coordinate was changed
        self.assertAlmostEqual(drawing.x(self.line_nodes[0]), 0.875, delta=1e-8,
                               msg="Node1 x-coordinate should be updated")
        self.assertAlmostEqual(drawing.x(self.line_nodes[1]), 0.375, delta=1e-8,
                               msg="Node2 x-coordinate should remain unchanged")

    def test_with_large_graph(self):
        """
        Test with a larger graph (Les Miserables)
        """
        # Create Les Miserables graph
        les_mis_graph = nx.les_miserables_graph()
        graph = eg.Graph()
        indices = {}
        for u in les_mis_graph.nodes:
            indices[u] = graph.add_node(u)
        for u, v in les_mis_graph.edges:
            graph.add_edge(indices[u], indices[v], (u, v))

        # Create a drawing with initial placement
        drawing = eg.DrawingTorus2d.initial_placement(graph)

        # Verify that the drawing has the correct number of nodes
        self.assertEqual(drawing.len(), len(les_mis_graph.nodes),
                         "Drawing should have the same number of nodes as the graph")

        # Verify all nodes have valid coordinates
        check_drawing_2d(graph, drawing)

        # Record initial positions
        initial_positions = record_positions_2d(drawing, graph)

        # Note: KamadaKawai layout doesn't support torus drawings directly
        # Instead, we'll manually modify some node positions to simulate a layout change

        # Modify a few node positions manually (just enough to verify the test logic)
        node_sample = list(graph.node_indices())[:5]  # Take first 5 nodes
        for i, node in enumerate(node_sample):
            # Set to different positions within the unit square
            drawing.set_x(node, (i * 0.1) + 0.3)
            drawing.set_y(node, (i % 3 * 0.1) + 0.4)

        # Verify that positions have changed
        self.assertTrue(positions_changed_2d(drawing, graph, initial_positions),
                        "Node positions should change after manual modification")

        # Verify all nodes have valid coordinates after modification
        check_drawing_2d(graph, drawing)

        # Verify that all coordinates are within the valid range [0,1]
        for u in graph.node_indices():
            x = drawing.x(u)
            y = drawing.y(u)
            self.assertGreaterEqual(
                x, 0, f"Node {u} x-coordinate should be >= 0")
            self.assertLessEqual(x, 1, f"Node {u} x-coordinate should be <= 1")
            self.assertGreaterEqual(
                y, 0, f"Node {u} y-coordinate should be >= 0")
            self.assertLessEqual(y, 1, f"Node {u} y-coordinate should be <= 1")

        # Verify that connected nodes are positioned closer together
        verify_connected_nodes_closer(graph, drawing)


if __name__ == "__main__":
    unittest.main()
