import unittest
import egraph as eg


class TestTriangulation(unittest.TestCase):
    def test_triangulation_square(self):
        # Create a graph with 4 nodes in a square formation
        graph = eg.Graph()
        n1 = graph.add_node(None)
        n2 = graph.add_node(None)
        n3 = graph.add_node(None)
        n4 = graph.add_node(None)

        # Create a drawing with the nodes positioned in a square
        drawing = eg.DrawingEuclidean2d.initial_placement(graph)
        drawing.set_x(n1, 0.0)
        drawing.set_y(n1, 0.0)
        drawing.set_x(n2, 1.0)
        drawing.set_y(n2, 0.0)
        drawing.set_x(n3, 0.0)
        drawing.set_y(n3, 1.0)
        drawing.set_x(n4, 1.0)
        drawing.set_y(n4, 1.0)

        # Compute the Delaunay triangulation
        triangulated_graph = eg.triangulation(drawing)

        # The triangulated graph should have 4 nodes and 5 edges
        # (4 edges around the square and 1 diagonal)
        self.assertEqual(triangulated_graph.node_count(), 4)
        self.assertEqual(triangulated_graph.edge_count(), 5)

    def test_triangulation_triangle(self):
        # Create a graph with 3 nodes in a triangle formation
        graph = eg.Graph()
        n1 = graph.add_node(None)
        n2 = graph.add_node(None)
        n3 = graph.add_node(None)

        # Create a drawing with the nodes positioned in a triangle
        drawing = eg.DrawingEuclidean2d.initial_placement(graph)
        drawing.set_x(n1, 0.0)
        drawing.set_y(n1, 0.0)
        drawing.set_x(n2, 1.0)
        drawing.set_y(n2, 0.0)
        drawing.set_x(n3, 0.5)
        drawing.set_y(n3, 0.866)  # Approximately sqrt(3)/2

        # Compute the Delaunay triangulation
        triangulated_graph = eg.triangulation(drawing)

        # The triangulated graph should have 3 nodes and 3 edges
        self.assertEqual(triangulated_graph.node_count(), 3)
        self.assertEqual(triangulated_graph.edge_count(), 3)

    def test_triangulation_collinear_points(self):
        # Create a graph with 3 collinear nodes
        graph = eg.Graph()
        n1 = graph.add_node(None)
        n2 = graph.add_node(None)
        n3 = graph.add_node(None)

        # Create a drawing with the nodes positioned in a line
        drawing = eg.DrawingEuclidean2d.initial_placement(graph)
        drawing.set_x(n1, 0.0)
        drawing.set_y(n1, 0.0)
        drawing.set_x(n2, 1.0)
        drawing.set_y(n2, 0.0)
        drawing.set_x(n3, 2.0)
        drawing.set_y(n3, 0.0)

        # Compute the Delaunay triangulation
        triangulated_graph = eg.triangulation(drawing)

        # The triangulated graph should have 3 nodes and 2 edges
        self.assertEqual(triangulated_graph.node_count(), 3)
        self.assertEqual(triangulated_graph.edge_count(), 2)


if __name__ == "__main__":
    unittest.main()
