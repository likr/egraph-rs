import math
import unittest
import networkx as nx
import egraph as eg


def draw(nx_graph):
    graph = eg.Graph()
    indices = {}
    for u in nx_graph.nodes:
        indices[u] = graph.add_node(u)
    for u, v in nx_graph.edges:
        graph.add_edge(indices[u], indices[v], (u, v))
    return graph


class TestBhTsNet(unittest.TestCase):
    def test_bh_ts_net_builder_defaults(self):
        builder = eg.BhTsNetBuilder()
        bh_ts_net = builder.build()
        self.assertIsNotNone(bh_ts_net)

    def test_bh_ts_net_run(self):
        # Create a simple cycle graph of size 6
        nx_graph = nx.cycle_graph(6)
        graph = draw(nx_graph)

        drawing = eg.DrawingEuclidean2d.initial_placement(graph)
        rng = eg.Rng.seed_from(42)

        # Initialize BH-tsNET with builder
        bh_ts_net = (
            eg.BhTsNetBuilder()
            .perplexity(2.0)
            .k(4)
            .theta(0.5)
            .learning_rate(2.0)
            .iterations_stage1(10)
            .iterations_stage2(10)
            .iterations_stage3(10)
            .lambda_c_stage2(1.2)
            .lambda_c_stage3(0.01)
            .lambda_r_stage3(0.6)
            .build()
        )

        bh_ts_net.run(drawing, graph, rng)

        for u in graph.node_indices():
            self.assertTrue(math.isfinite(drawing.x(u)))
            self.assertTrue(math.isfinite(drawing.y(u)))

    def test_bh_ts_net_via_default_constructor(self):
        nx_graph = nx.cycle_graph(4)
        graph = draw(nx_graph)

        drawing = eg.DrawingEuclidean2d.initial_placement(graph)
        rng = eg.Rng.seed_from(42)

        bh_ts_net = eg.BhTsNet()
        bh_ts_net.run(drawing, graph, rng)

        for u in graph.node_indices():
            self.assertTrue(math.isfinite(drawing.x(u)))
            self.assertTrue(math.isfinite(drawing.y(u)))

    def test_bh_ts_net_validation_error(self):
        builder = eg.BhTsNetBuilder().perplexity(-1.0)
        with self.assertRaises(ValueError):
            builder.build()

        builder_theta = eg.BhTsNetBuilder().theta(0.0)
        with self.assertRaises(ValueError):
            builder_theta.build()


if __name__ == "__main__":
    unittest.main()
