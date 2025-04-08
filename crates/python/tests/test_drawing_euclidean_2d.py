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


class TestDrawingEuclidean2d(unittest.TestCase):
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
        Test creating a DrawingEuclidean2d instance with initial placement
        """
        # Create a drawing with initial placement
        drawing = eg.DrawingEuclidean2d.initial_placement(self.line_graph)

        # Verify that the instance exists
        self.assertIsInstance(drawing, eg.DrawingEuclidean2d)

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
        drawing = eg.DrawingEuclidean2d.initial_placement(self.line_graph)

        # Verify initial coordinates are finite numbers
        check_drawing_2d(self.line_graph, drawing)

        # Test setting coordinates for the first node
        new_x1 = 10.5
        new_y1 = 20.5
        drawing.set_x(self.line_nodes[0], new_x1)
        drawing.set_y(self.line_nodes[0], new_y1)

        # Verify coordinates were set correctly
        self.assertAlmostEqual(drawing.x(self.line_nodes[0]), new_x1, delta=1e-8,
                               msg="X coordinate should be updated")
        self.assertAlmostEqual(drawing.y(self.line_nodes[0]), new_y1, delta=1e-8,
                               msg="Y coordinate should be updated")

        # Test setting coordinates for another node
        new_x2 = -5.5
        new_y2 = -15.5
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

    def test_drawing_manipulation(self):
        """
        Test drawing manipulation (centralize, clamp_region)
        """
        # Create a drawing with initial placement
        drawing = eg.DrawingEuclidean2d.initial_placement(self.line_graph)

        # Set specific coordinates for testing
        drawing.set_x(self.line_nodes[0], 10)
        drawing.set_y(self.line_nodes[0], 20)
        drawing.set_x(self.line_nodes[1], 30)
        drawing.set_y(self.line_nodes[1], 40)

        # Record positions before centralization
        before_x0 = drawing.x(self.line_nodes[0])
        before_y0 = drawing.y(self.line_nodes[0])
        before_x1 = drawing.x(self.line_nodes[1])
        before_y1 = drawing.y(self.line_nodes[1])

        # Calculate the center of mass before centralization
        before_center_x = (before_x0 + before_x1) / 2
        before_center_y = (before_y0 + before_y1) / 2

        # Test centralize
        drawing.centralize()

        # Record positions after centralization
        after_x0 = drawing.x(self.line_nodes[0])
        after_y0 = drawing.y(self.line_nodes[0])
        after_x1 = drawing.x(self.line_nodes[1])
        after_y1 = drawing.y(self.line_nodes[1])

        # Calculate the center of mass after centralization
        after_center_x = (after_x0 + after_x1) / 2
        after_center_y = (after_y0 + after_y1) / 2

        # Calculate the center of mass and positions after centralization

        # Verify that the center of mass has changed after centralization
        # The exact position may vary depending on the implementation
        self.assertNotEqual(before_center_x, after_center_x,
                            msg="Center X should change after centralization")
        self.assertNotEqual(before_center_y, after_center_y,
                            msg="Center Y should change after centralization")

        # Verify that the relative positions are preserved
        # The distance between nodes should be the same before and after
        before_dx = before_x1 - before_x0
        before_dy = before_y1 - before_y0
        after_dx = after_x1 - after_x0
        after_dy = after_y1 - after_y0

        self.assertAlmostEqual(after_dx, before_dx, delta=1e-6,
                               msg="Relative X distance should be preserved")
        self.assertAlmostEqual(after_dy, before_dy, delta=1e-6,
                               msg="Relative Y distance should be preserved")

        # Test clamp_region
        # Set coordinates outside the clamping region
        drawing.set_x(self.line_nodes[0], -100)
        drawing.set_y(self.line_nodes[0], -200)
        drawing.set_x(self.line_nodes[1], 100)
        drawing.set_y(self.line_nodes[1], 200)

        # Clamp to region [-50, -50, 50, 50]
        drawing.clamp_region(-50, -50, 50, 50)

        # Verify coordinates are clamped
        self.assertAlmostEqual(drawing.x(self.line_nodes[0]), -50, delta=1e-8,
                               msg="X coordinate should be clamped to minimum")
        self.assertAlmostEqual(drawing.y(self.line_nodes[0]), -50, delta=1e-8,
                               msg="Y coordinate should be clamped to minimum")
        self.assertAlmostEqual(drawing.x(self.line_nodes[1]), 50, delta=1e-8,
                               msg="X coordinate should be clamped to maximum")
        self.assertAlmostEqual(drawing.y(self.line_nodes[1]), 50, delta=1e-8,
                               msg="Y coordinate should be clamped to maximum")

    def test_edge_segments(self):
        """
        Test edge segment representation
        """
        # Create a drawing with initial placement
        drawing = eg.DrawingEuclidean2d.initial_placement(self.line_graph)

        # Set specific coordinates for testing
        drawing.set_x(self.line_nodes[0], 0)
        drawing.set_y(self.line_nodes[0], 0)
        drawing.set_x(self.line_nodes[1], 10)
        drawing.set_y(self.line_nodes[1], 10)

        # Test edge segments
        segments = drawing.edge_segments(
            self.line_nodes[0], self.line_nodes[1])

        # Verify segments exist
        self.assertIsNotNone(segments, "Edge segments should exist")
        self.assertGreater(
            len(segments), 0, "There should be at least one segment for the edge")

        # For Euclidean 2D, there should be one straight line segment
        self.assertEqual(len(segments), 1,
                         "There should be exactly one segment for a straight line in Euclidean 2D")

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
        self.assertAlmostEqual(start_point[0], drawing.x(self.line_nodes[0]), delta=1e-8,
                               msg="Start point x should match node1 x")
        self.assertAlmostEqual(start_point[1], drawing.y(self.line_nodes[0]), delta=1e-8,
                               msg="Start point y should match node1 y")
        self.assertAlmostEqual(end_point[0], drawing.x(self.line_nodes[1]), delta=1e-8,
                               msg="End point x should match node2 x")
        self.assertAlmostEqual(end_point[1], drawing.y(self.line_nodes[1]), delta=1e-8,
                               msg="End point y should match node2 y")

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
        drawing = eg.DrawingEuclidean2d.initial_placement(graph)

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

        # Test with a layout algorithm
        initial_positions = record_positions_2d(drawing, graph)

        # Apply Kamada-Kawai layout
        layout = eg.KamadaKawai(graph, lambda e: 1.0)
        layout.run(drawing)

        # Verify that positions have changed
        self.assertTrue(positions_changed_2d(drawing, graph, initial_positions),
                        "Layout algorithm should change node positions")

        # Verify layout quality
        verify_layout_quality(graph, drawing)

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
        drawing = eg.DrawingEuclidean2d.initial_placement(graph)

        # Verify that the drawing has the correct number of nodes
        self.assertEqual(drawing.len(), len(les_mis_graph.nodes),
                         "Drawing should have the same number of nodes as the graph")

        # Verify all nodes have valid coordinates
        check_drawing_2d(graph, drawing)

        # Record initial positions
        initial_positions = record_positions_2d(drawing, graph)

        # Apply Kamada-Kawai layout with a larger epsilon for faster convergence
        layout = eg.KamadaKawai(graph, lambda e: 1.0)
        layout.eps = 0.1
        layout.run(drawing)

        # Verify that positions have changed
        self.assertTrue(positions_changed_2d(drawing, graph, initial_positions),
                        "Layout algorithm should change node positions")

        # Verify all nodes have valid coordinates after layout
        check_drawing_2d(graph, drawing)

        # Verify that connected nodes are positioned closer together
        verify_connected_nodes_closer(graph, drawing)


if __name__ == "__main__":
    unittest.main()
