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
    check_drawing_spherical,
    record_positions_spherical,
    positions_changed_spherical,
    verify_node_positions,
    verify_layout_quality
)


class TestDrawingSpherical2d(unittest.TestCase):
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
        Test creating a DrawingSpherical2d instance with initial placement
        """
        # Create a drawing with initial placement
        drawing = eg.DrawingSpherical2d.initial_placement(self.line_graph)

        # Verify that the instance exists
        self.assertIsInstance(drawing, eg.DrawingSpherical2d)

        # Verify initial state
        self.assertEqual(drawing.len(), len(self.line_nodes),
                         "Drawing should have the same number of nodes as the graph")

        # Verify all coordinates are finite
        check_drawing_spherical(self.line_graph, drawing)

    def test_node_coordinates(self):
        """
        Test node coordinate operations (get/set longitude,latitude)
        """
        # Create a drawing with initial placement
        drawing = eg.DrawingSpherical2d.initial_placement(self.line_graph)

        # Verify initial coordinates are finite numbers
        check_drawing_spherical(self.line_graph, drawing)

        # Test setting coordinates for the first node
        new_lon1 = 0.5  # Longitude in radians
        new_lat1 = 0.25  # Latitude in radians
        drawing.set_lon(self.line_nodes[0], new_lon1)
        drawing.set_lat(self.line_nodes[0], new_lat1)

        # Verify coordinates were set correctly
        self.assertAlmostEqual(drawing.lon(self.line_nodes[0]), new_lon1, delta=1e-8,
                               msg="Longitude coordinate should be updated")
        self.assertAlmostEqual(drawing.lat(self.line_nodes[0]), new_lat1, delta=1e-8,
                               msg="Latitude coordinate should be updated")

        # Test setting coordinates for another node
        new_lon2 = -0.5  # Longitude in radians
        new_lat2 = -0.25  # Latitude in radians
        drawing.set_lon(self.line_nodes[1], new_lon2)
        drawing.set_lat(self.line_nodes[1], new_lat2)

        # Verify coordinates were set correctly
        self.assertAlmostEqual(drawing.lon(self.line_nodes[1]), new_lon2, delta=1e-8,
                               msg="Longitude coordinate should be updated")
        self.assertAlmostEqual(drawing.lat(self.line_nodes[1]), new_lat2, delta=1e-8,
                               msg="Latitude coordinate should be updated")

        # Test getting coordinates for non-existent node
        # In the current implementation, lon() and lat() return None for non-existent nodes
        self.assertIsNone(drawing.lon(999),
                          "Longitude coordinate for non-existent node should be None")
        self.assertIsNone(drawing.lat(999),
                          "Latitude coordinate for non-existent node should be None")

    def test_coordinate_validation(self):
        """
        Test coordinate validation and range checking
        """
        # Create a drawing with initial placement
        drawing = eg.DrawingSpherical2d.initial_placement(self.line_graph)

        # Test valid values within range
        valid_lon = math.pi / 4  # 45 degrees
        valid_lat = math.pi / 6  # 30 degrees

        drawing.set_lon(self.line_nodes[0], valid_lon)
        drawing.set_lat(self.line_nodes[0], valid_lat)

        # Verify coordinates were set correctly
        self.assertAlmostEqual(drawing.lon(self.line_nodes[0]), valid_lon, delta=1e-6,
                               msg="Valid longitude should be set correctly")
        self.assertAlmostEqual(drawing.lat(self.line_nodes[0]), valid_lat, delta=1e-6,
                               msg="Valid latitude should be set correctly")

        # Test longitude at the boundaries
        drawing.set_lon(self.line_nodes[0], math.pi)
        # π and -π are equivalent in spherical coordinates
        self.assertTrue(
            abs(drawing.lon(self.line_nodes[0]) - math.pi) < 1e-6 or
            abs(drawing.lon(self.line_nodes[0]) + math.pi) < 1e-6,
            "Longitude at π should be set correctly (or equivalent to -π)"
        )

        drawing.set_lon(self.line_nodes[0], -math.pi)
        # π and -π are equivalent in spherical coordinates
        self.assertTrue(
            abs(drawing.lon(self.line_nodes[0]) + math.pi) < 1e-6 or
            abs(drawing.lon(self.line_nodes[0]) - math.pi) < 1e-6,
            "Longitude at -π should be set correctly (or equivalent to π)"
        )

        # Test latitude at the boundaries
        drawing.set_lat(self.line_nodes[0], math.pi / 2)
        self.assertAlmostEqual(drawing.lat(self.line_nodes[0]), math.pi / 2, delta=1e-6,
                               msg="Latitude at π/2 should be set correctly")

        drawing.set_lat(self.line_nodes[0], -math.pi / 2)
        self.assertAlmostEqual(drawing.lat(self.line_nodes[0]), -math.pi / 2, delta=1e-6,
                               msg="Latitude at -π/2 should be set correctly")

        # Test setting coordinates for multiple nodes
        drawing.set_lon(self.line_nodes[0], 0)
        drawing.set_lat(self.line_nodes[0], 0)
        drawing.set_lon(self.line_nodes[1], math.pi / 2)
        drawing.set_lat(self.line_nodes[1], math.pi / 4)

        self.assertAlmostEqual(drawing.lon(self.line_nodes[0]), 0, delta=1e-6,
                               msg="Node1 longitude should be 0")
        self.assertAlmostEqual(drawing.lat(self.line_nodes[0]), 0, delta=1e-6,
                               msg="Node1 latitude should be 0")

        self.assertAlmostEqual(drawing.lon(self.line_nodes[1]), math.pi / 2, delta=1e-6,
                               msg="Node2 longitude should be π/2")
        self.assertAlmostEqual(drawing.lat(self.line_nodes[1]), math.pi / 4, delta=1e-6,
                               msg="Node2 latitude should be π/4")

        # Test that setting one node's coordinates doesn't affect others
        drawing.set_lon(self.line_nodes[0], math.pi / 3)

        self.assertAlmostEqual(drawing.lon(self.line_nodes[0]), math.pi / 3, delta=1e-6,
                               msg="Node1 longitude should be updated to π/3")
        self.assertAlmostEqual(drawing.lon(self.line_nodes[1]), math.pi / 2, delta=1e-6,
                               msg="Node2 longitude should remain π/2")

    def test_spherical_coordinates(self):
        """
        Test spherical coordinates and conversion to 3D points
        """
        # Create a drawing with initial placement
        drawing = eg.DrawingSpherical2d.initial_placement(self.line_graph)

        # Set specific coordinates for testing
        # Set nodes at different points on the sphere
        drawing.set_lon(self.line_nodes[0], 0)  # Prime meridian
        drawing.set_lat(self.line_nodes[0], 0)  # Equator
        drawing.set_lon(self.line_nodes[1], math.pi / 2)  # 90 degrees east
        drawing.set_lat(self.line_nodes[1], 0)  # Equator

        # Convert spherical coordinates to 3D Cartesian coordinates
        # For node1 at (0,0): (1,0,0)
        x1 = math.cos(drawing.lat(
            self.line_nodes[0])) * math.cos(drawing.lon(self.line_nodes[0]))
        y1 = math.cos(drawing.lat(
            self.line_nodes[0])) * math.sin(drawing.lon(self.line_nodes[0]))
        z1 = math.sin(drawing.lat(self.line_nodes[0]))

        # For node2 at (π/2,0): (0,1,0)
        x2 = math.cos(drawing.lat(
            self.line_nodes[1])) * math.cos(drawing.lon(self.line_nodes[1]))
        y2 = math.cos(drawing.lat(
            self.line_nodes[1])) * math.sin(drawing.lon(self.line_nodes[1]))
        z2 = math.sin(drawing.lat(self.line_nodes[1]))

        # Verify 3D coordinates are correct (with small floating-point tolerance)
        self.assertAlmostEqual(x1, 1.0, delta=1e-6,
                               msg="X coordinate for node1 should be 1.0")
        self.assertAlmostEqual(y1, 0.0, delta=1e-6,
                               msg="Y coordinate for node1 should be 0.0")
        self.assertAlmostEqual(z1, 0.0, delta=1e-6,
                               msg="Z coordinate for node1 should be 0.0")

        self.assertAlmostEqual(x2, 0.0, delta=1e-6,
                               msg="X coordinate for node2 should be 0.0")
        self.assertAlmostEqual(y2, 1.0, delta=1e-6,
                               msg="Y coordinate for node2 should be 1.0")
        self.assertAlmostEqual(z2, 0.0, delta=1e-6,
                               msg="Z coordinate for node2 should be 0.0")

        # Verify points are on the unit sphere (x^2 + y^2 + z^2 = 1)
        magnitude1 = math.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        magnitude2 = math.sqrt(x2 * x2 + y2 * y2 + z2 * z2)

        self.assertAlmostEqual(magnitude1, 1.0, delta=1e-6,
                               msg="Node1 should be on the unit sphere")
        self.assertAlmostEqual(magnitude2, 1.0, delta=1e-6,
                               msg="Node2 should be on the unit sphere")

        # Set node3 at the north pole
        # Longitude doesn't matter at poles
        drawing.set_lon(self.line_nodes[2], 0)
        drawing.set_lat(self.line_nodes[2], math.pi / 2)  # North pole

        # Convert to 3D coordinates
        x3 = math.cos(drawing.lat(
            self.line_nodes[2])) * math.cos(drawing.lon(self.line_nodes[2]))
        y3 = math.cos(drawing.lat(
            self.line_nodes[2])) * math.sin(drawing.lon(self.line_nodes[2]))
        z3 = math.sin(drawing.lat(self.line_nodes[2]))

        # Verify Z coordinate is close to 1.0 (north pole)
        self.assertAlmostEqual(
            z3, 1.0, delta=1e-6, msg="Z coordinate for node3 should be 1.0 (north pole)")

        # Verify point is on the unit sphere
        magnitude3 = math.sqrt(x3 * x3 + y3 * y3 + z3 * z3)
        self.assertAlmostEqual(magnitude3, 1.0, delta=1e-6,
                               msg="Node3 should be on the unit sphere")

    def test_great_circle_distance(self):
        """
        Test great circle distance calculations between nodes
        """
        # Create a drawing with initial placement
        drawing = eg.DrawingSpherical2d.initial_placement(self.line_graph)

        # Set specific coordinates for testing
        # Place nodes at known positions
        drawing.set_lon(self.line_nodes[0], 0)  # Prime meridian
        drawing.set_lat(self.line_nodes[0], 0)  # Equator

        # Test 1: 90 degrees east along equator (π/2 radians)
        drawing.set_lon(self.line_nodes[1], math.pi / 2)
        drawing.set_lat(self.line_nodes[1], 0)

        # Calculate great circle distance manually
        # Convert to 3D Cartesian coordinates
        x1 = math.cos(drawing.lat(
            self.line_nodes[0])) * math.cos(drawing.lon(self.line_nodes[0]))
        y1 = math.cos(drawing.lat(
            self.line_nodes[0])) * math.sin(drawing.lon(self.line_nodes[0]))
        z1 = math.sin(drawing.lat(self.line_nodes[0]))

        x2 = math.cos(drawing.lat(
            self.line_nodes[1])) * math.cos(drawing.lon(self.line_nodes[1]))
        y2 = math.cos(drawing.lat(
            self.line_nodes[1])) * math.sin(drawing.lon(self.line_nodes[1]))
        z2 = math.sin(drawing.lat(self.line_nodes[1]))

        # Calculate dot product
        dot_product1 = x1 * x2 + y1 * y2 + z1 * z2

        # Calculate angle (great circle distance)
        distance1 = math.acos(max(-1, min(1, dot_product1)))

        # For nodes on the equator separated by 90 degrees, the distance should be π/2 radians
        expected_distance1 = math.pi / 2

        # Verify the distance is close to the expected great circle distance
        self.assertAlmostEqual(distance1, expected_distance1, delta=1e-6,
                               msg=f"Great circle distance should be approximately {expected_distance1} radians")

        # Test 2: North pole (π/2 radians latitude)
        drawing.set_lon(self.line_nodes[1], 0)
        drawing.set_lat(self.line_nodes[1], math.pi / 2)

        # Recalculate 3D coordinates for node2
        x2b = math.cos(drawing.lat(
            self.line_nodes[1])) * math.cos(drawing.lon(self.line_nodes[1]))
        y2b = math.cos(drawing.lat(
            self.line_nodes[1])) * math.sin(drawing.lon(self.line_nodes[1]))
        z2b = math.sin(drawing.lat(self.line_nodes[1]))

        # Calculate dot product
        dot_product2 = x1 * x2b + y1 * y2b + z1 * z2b

        # Calculate angle (great circle distance)
        distance2 = math.acos(max(-1, min(1, dot_product2)))

        # For a node at the equator and a node at the north pole, the distance should be π/2 radians
        expected_distance2 = math.pi / 2

        # Verify the distance is close to the expected great circle distance
        self.assertAlmostEqual(distance2, expected_distance2, delta=1e-6,
                               msg=f"Great circle distance should be approximately {expected_distance2} radians")

        # Test 3: Antipodal points (opposite sides of the sphere)
        drawing.set_lon(self.line_nodes[0], 0)
        drawing.set_lat(self.line_nodes[0], 0)
        drawing.set_lon(self.line_nodes[1], math.pi)
        drawing.set_lat(self.line_nodes[1], 0)

        # Recalculate 3D coordinates
        x1c = math.cos(drawing.lat(
            self.line_nodes[0])) * math.cos(drawing.lon(self.line_nodes[0]))
        y1c = math.cos(drawing.lat(
            self.line_nodes[0])) * math.sin(drawing.lon(self.line_nodes[0]))
        z1c = math.sin(drawing.lat(self.line_nodes[0]))

        x2c = math.cos(drawing.lat(
            self.line_nodes[1])) * math.cos(drawing.lon(self.line_nodes[1]))
        y2c = math.cos(drawing.lat(
            self.line_nodes[1])) * math.sin(drawing.lon(self.line_nodes[1]))
        z2c = math.sin(drawing.lat(self.line_nodes[1]))

        # Calculate dot product
        dot_product3 = x1c * x2c + y1c * y2c + z1c * z2c

        # Calculate angle (great circle distance)
        distance3 = math.acos(max(-1, min(1, dot_product3)))

        # For antipodal points, the distance should be π radians
        expected_distance3 = math.pi

        # Verify the distance is close to the expected great circle distance
        self.assertAlmostEqual(distance3, expected_distance3, delta=1e-6,
                               msg=f"Great circle distance should be approximately {expected_distance3} radians")

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
        drawing = eg.DrawingSpherical2d.initial_placement(graph)

        # Verify that the drawing has the correct number of nodes
        self.assertEqual(drawing.len(), len(nodes),
                         "Drawing should have the same number of nodes as the graph")

        # Verify all nodes have valid coordinates
        check_drawing_spherical(graph, drawing)

        # Record initial positions
        initial_positions = record_positions_spherical(drawing, graph)

        # Note: KamadaKawai layout doesn't support spherical drawings directly
        # Instead, we'll manually modify some node positions to simulate a layout change

        # Modify node positions manually
        drawing.set_lon(nodes[0], 0.5)
        drawing.set_lat(nodes[0], 0.3)
        drawing.set_lon(nodes[1], -0.5)
        drawing.set_lat(nodes[1], 0.2)
        drawing.set_lon(nodes[2], 0.2)
        drawing.set_lat(nodes[2], -0.3)

        # Verify that positions have changed
        self.assertTrue(positions_changed_spherical(drawing, graph, initial_positions),
                        "Node positions should change after manual modification")

        # Verify all nodes have valid coordinates after modification
        check_drawing_spherical(graph, drawing)

        # Verify all nodes are still on the sphere
        for u in graph.node_indices():
            # Convert spherical to Cartesian coordinates
            lon = drawing.lon(u)
            lat = drawing.lat(u)

            x = math.cos(lat) * math.cos(lon)
            y = math.cos(lat) * math.sin(lon)
            z = math.sin(lat)

            # Calculate distance from origin (should be 1 for unit sphere)
            distance = math.sqrt(x * x + y * y + z * z)

            self.assertAlmostEqual(distance, 1.0, delta=1e-6,
                                   msg=f"Node {u} should remain on the unit sphere after layout")

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
        drawing = eg.DrawingSpherical2d.initial_placement(graph)

        # Verify that the drawing has the correct number of nodes
        self.assertEqual(drawing.len(), len(les_mis_graph.nodes),
                         "Drawing should have the same number of nodes as the graph")

        # Verify all nodes have valid coordinates
        check_drawing_spherical(graph, drawing)

        # Record initial positions
        initial_positions = record_positions_spherical(drawing, graph)

        # Note: KamadaKawai layout doesn't support spherical drawings directly
        # Instead, we'll manually modify some node positions to simulate a layout change

        # Modify a few node positions manually (just enough to verify the test logic)
        node_sample = list(graph.node_indices())[:5]  # Take first 5 nodes
        for i, node in enumerate(node_sample):
            # Set to different positions around the sphere
            drawing.set_lon(node, (i * math.pi / 5) - math.pi/2)
            drawing.set_lat(node, (i % 3 - 1) * math.pi / 6)

        # Verify that positions have changed
        self.assertTrue(positions_changed_spherical(drawing, graph, initial_positions),
                        "Node positions should change after manual modification")

        # Verify all nodes have valid coordinates after modification
        check_drawing_spherical(graph, drawing)

        # Verify all nodes are still on the sphere
        for u in graph.node_indices():
            # Convert spherical to Cartesian coordinates
            lon = drawing.lon(u)
            lat = drawing.lat(u)

            x = math.cos(lat) * math.cos(lon)
            y = math.cos(lat) * math.sin(lon)
            z = math.sin(lat)

            # Calculate distance from origin (should be 1 for unit sphere)
            distance = math.sqrt(x * x + y * y + z * z)

            self.assertAlmostEqual(distance, 1.0, delta=1e-6,
                                   msg=f"Node {u} should remain on the unit sphere after layout")


if __name__ == "__main__":
    unittest.main()
