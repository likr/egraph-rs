import unittest
import egraph as eg


class TestSeparationConstraints(unittest.TestCase):
    def test_constraint_creation(self):
        """Test creating a separation constraint"""
        constraint = eg.Constraint(0, 1, 5.0)
        self.assertEqual(constraint.left, 0)
        self.assertEqual(constraint.right, 1)
        self.assertEqual(constraint.gap, 5.0)

    def test_project_1d(self):
        """Test applying separation constraints in one dimension"""
        # Create a graph with 2 nodes
        graph = eg.Graph()
        n1 = graph.add_node(None)
        n2 = graph.add_node(None)

        # Create a drawing with the nodes positioned closely
        drawing = eg.DrawingEuclidean2d.initial_placement(graph)
        drawing.set_x(n1, 0.0)
        drawing.set_y(n1, 0.0)
        drawing.set_x(n2, 1.0)
        drawing.set_y(n2, 0.0)

        # Create a constraint to separate the nodes in x dimension
        # n1 should be at least 5.0 units left of n2
        constraint = eg.Constraint(0, 1, 5.0)
        eg.project_1d(drawing, 0, [constraint])  # Apply in x dimension

        # Verify the constraint is now satisfied
        self.assertLess(drawing.x(n1), drawing.x(n2) - 5.0 + 1e-5)

    def test_generate_rectangle_no_overlap_constraints_2d(self):
        """Test generating rectangle non-overlap constraints"""
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
        drawing.set_x(n2, 2.0)
        drawing.set_y(n2, 0.0)
        drawing.set_x(n3, 0.0)
        drawing.set_y(n3, 2.0)
        drawing.set_x(n4, 2.0)
        drawing.set_y(n4, 2.0)

        # Each node is size 1.0 in both dimensions
        node_size = 1.0
        def size_fn(node, dim): return node_size

        # Generate constraints for the x dimension
        constraints_x = eg.generate_rectangle_no_overlap_constraints_2d(
            drawing, size_fn, 0
        )

        # There should be constraints between potentially overlapping rectangles
        self.assertGreater(len(constraints_x), 0)

        # Apply the constraints
        eg.project_1d(drawing, 0, constraints_x)

        # Verify no overlapping rectangles
        for i in range(4):
            for j in range(i + 1, 4):
                # Check if rectangles are separated in x-dimension
                centers_distance_x = abs(drawing.x(i) - drawing.x(j))
                min_distance_x = node_size

                if centers_distance_x < min_distance_x - 1e-5:
                    # If not separated in x, they must be separated in y
                    centers_distance_y = abs(drawing.y(i) - drawing.y(j))
                    min_distance_y = node_size
                    self.assertGreaterEqual(
                        centers_distance_y, min_distance_y - 1e-5)

    def test_project_rectangle_no_overlap_constraints_2d(self):
        """Test applying rectangle non-overlap constraints in a single call"""
        # Create a graph with 2 overlapping nodes
        graph = eg.Graph()
        n1 = graph.add_node(None)
        n2 = graph.add_node(None)

        # Create a drawing with the nodes positioned with overlap
        drawing = eg.DrawingEuclidean2d.initial_placement(graph)
        drawing.set_x(n1, 0.0)
        drawing.set_y(n1, 0.0)
        # Overlapping with n1 if size is 1.0
        drawing.set_x(n2, 0.5)
        drawing.set_y(n2, 0.5)

        # Save original positions
        orig_x1 = drawing.x(n1)
        orig_y1 = drawing.y(n1)
        orig_x2 = drawing.x(n2)
        orig_y2 = drawing.y(n2)

        # Apply constraints to remove overlaps in x dimension
        eg.project_rectangle_no_overlap_constraints_2d(
            drawing, lambda node, dim: 1.0, 0
        )

        # Verify positions have changed to resolve x-overlap
        self.assertNotEqual(drawing.x(n1), orig_x1,
                            "Position of node 1 should change")
        self.assertNotEqual(drawing.x(n2), orig_x2,
                            "Position of node 2 should change")

        # Verify the rectangles no longer overlap in x dimension
        dx = abs(drawing.x(n1) - drawing.x(n2))
        self.assertGreaterEqual(
            dx, 1.0 - 1e-5, "Rectangles should be separated in x dimension")

    def test_generate_layered_constraints(self):
        """Test generating layered constraints for a directed graph"""
        # Create a directed graph
        graph = eg.DiGraph()
        n1 = graph.add_node(None)
        n2 = graph.add_node(None)
        n3 = graph.add_node(None)

        # Add edges to form a path: n1 -> n2 -> n3
        graph.add_edge(n1, n2, None)
        graph.add_edge(n2, n3, None)

        # Generate constraints for layered layout
        constraints = eg.generate_layered_constraints(graph, 2.0)

        # Should have 2 constraints: n1 -> n2 and n2 -> n3
        self.assertEqual(len(constraints), 2)

        # Create a drawing to test constraint application
        drawing = eg.DrawingEuclidean2d.initial_placement(graph)
        for i in range(3):
            # Initial positions in a horizontal line
            drawing.set_x(i, i * 1.0)
            drawing.set_y(i, 0.0)

        # Apply constraints to the y-dimension (vertical)
        eg.project_1d(drawing, 1, constraints)

        # Verify that nodes are properly layered vertically
        self.assertGreaterEqual(drawing.y(n2) - drawing.y(n1), 2.0 - 1e-5)
        self.assertGreaterEqual(drawing.y(n3) - drawing.y(n2), 2.0 - 1e-5)

    def test_project_clustered_rectangle_no_overlap_constraints(self):
        """Test applying cluster overlap constraints"""
        # Create a graph with 4 nodes in 2 clusters
        graph = eg.Graph()
        n1 = graph.add_node(None)
        n2 = graph.add_node(None)
        n3 = graph.add_node(None)
        n4 = graph.add_node(None)

        # Create a drawing
        drawing = eg.DrawingEuclidean2d.initial_placement(graph)

        # Position nodes in two clusters
        drawing.set_x(n1, 0.0)
        drawing.set_y(n1, 0.0)
        drawing.set_x(n2, 1.0)
        drawing.set_y(n2, 1.0)
        # Cluster 2 starts close to cluster 1
        drawing.set_x(n3, 3.0)
        drawing.set_y(n3, 1.0)
        drawing.set_x(n4, 4.0)
        drawing.set_y(n4, 1.0)

        # Define a cluster ID function
        def get_cluster(node):
            # Nodes 0,1 in cluster 0; nodes 2,3 in cluster 1
            return 0 if node < 2 else 1

        # Apply cluster-based separation in x dimension
        eg.project_clustered_rectangle_no_overlap_constraints(
            graph,
            drawing,
            0,  # x dimension
            get_cluster,
            lambda node, dim: 1.0  # Size function
        )

        # Verify clusters are separated
        # Get the rightmost node of cluster 1 and leftmost node of cluster 2
        cluster1_right = max(drawing.x(n1), drawing.x(n2))
        cluster2_left = min(drawing.x(n3), drawing.x(n4))

        # Should have sufficient gap between clusters
        self.assertGreater(cluster2_left - cluster1_right,
                           0, "Clusters should be separated")


if __name__ == "__main__":
    unittest.main()
