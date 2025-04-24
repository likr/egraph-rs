import unittest
import egraph as eg


class TestLayering(unittest.TestCase):
    def test_longest_path(self):
        # Create a simple directed acyclic graph
        graph = eg.DiGraph()
        n1 = graph.add_node({"id": "n1"})
        n2 = graph.add_node({"id": "n2"})
        n3 = graph.add_node({"id": "n3"})
        n4 = graph.add_node({"id": "n4"})

        # Add edges creating a simple hierarchy
        graph.add_edge(n1, n2, {"weight": 1.0})
        graph.add_edge(n1, n3, {"weight": 1.0})
        graph.add_edge(n2, n4, {"weight": 1.0})
        graph.add_edge(n3, n4, {"weight": 1.0})

        # Assign layers using the LongestPath algorithm
        longest_path = eg.LongestPath()
        layers = longest_path.assign_layers(graph)

        # Check the layering
        self.assertEqual(layers[n1], 0)  # Root node should be at layer 0
        self.assertEqual(layers[n2], 1)  # Second level nodes
        self.assertEqual(layers[n3], 1)  # Second level nodes
        self.assertEqual(layers[n4], 2)  # Leaf node should be at deepest layer

    def test_cycle_detection(self):
        # Create a directed graph with a cycle
        graph = eg.DiGraph()
        n1 = graph.add_node({"id": "n1"})
        n2 = graph.add_node({"id": "n2"})
        n3 = graph.add_node({"id": "n3"})

        # Add edges creating a cycle: n1 -> n2 -> n3 -> n1
        e1 = graph.add_edge(n1, n2, {"weight": 1.0})
        e2 = graph.add_edge(n2, n3, {"weight": 1.0})
        e3 = graph.add_edge(n3, n1, {"weight": 1.0})

        # Detect cycle edges
        cycle_edges = eg.cycle_edges(graph)

        # At least one edge from the cycle should be detected
        self.assertGreaterEqual(len(cycle_edges), 1)

        # Check that the detected edges are part of our cycle
        for edge in cycle_edges:
            source, target = edge
            self.assertTrue(
                (source == n1 and target == n2) or
                (source == n2 and target == n3) or
                (source == n3 and target == n1)
            )

    def test_cycle_removal(self):
        # Create a directed graph with a cycle
        graph = eg.DiGraph()
        n1 = graph.add_node({"id": "n1"})
        n2 = graph.add_node({"id": "n2"})
        n3 = graph.add_node({"id": "n3"})

        # Add edges creating a cycle: n1 -> n2 -> n3 -> n1
        graph.add_edge(n1, n2, {"weight": 1.0})
        graph.add_edge(n2, n3, {"weight": 1.0})
        graph.add_edge(n3, n1, {"weight": 1.0})

        # Remove cycles
        eg.remove_cycle(graph)

        # After cycle removal, the graph should be acyclic
        # We can test this by checking if there are any cycle edges left
        cycle_edges = eg.cycle_edges(graph)
        self.assertEqual(len(cycle_edges), 0)

    def test_layering_with_cycle_removal(self):
        # Create a directed graph with a cycle
        graph = eg.DiGraph()
        n1 = graph.add_node({"id": "n1"})
        n2 = graph.add_node({"id": "n2"})
        n3 = graph.add_node({"id": "n3"})
        n4 = graph.add_node({"id": "n4"})

        # Add edges creating a hierarchy with a cycle
        graph.add_edge(n1, n2, {"weight": 1.0})
        graph.add_edge(n2, n3, {"weight": 1.0})
        graph.add_edge(n3, n1, {"weight": 1.0})  # Creates a cycle
        graph.add_edge(n2, n4, {"weight": 1.0})

        # Remove cycles
        eg.remove_cycle(graph)

        # Now we should be able to assign layers
        longest_path = eg.LongestPath()
        layers = longest_path.assign_layers(graph)

        # Check that all nodes have layer assignments
        for i in range(4):
            self.assertIn(i, layers)

        # Check that if there's an edge (u,v), then layer(v) > layer(u)
        # for all remaining edges in the graph
        for e in range(graph.edge_count()):
            source, target = graph.edge_endpoints(e)
            self.assertLess(layers[source], layers[target])


if __name__ == "__main__":
    unittest.main()
