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


class TestTsNet(unittest.TestCase):
    def test_ts_net_builder_defaults(self):
        builder = eg.TsNetBuilder()
        ts_net = builder.build()
        self.assertIsNotNone(ts_net)

    def test_ts_net_with_distance_matrix(self):
        # Create a simple cycle graph of size 4
        nx_graph = nx.cycle_graph(4)
        graph = draw(nx_graph)

        drawing = eg.DrawingEuclidean2d.initial_placement(graph)
        distance_matrix = eg.all_sources_dijkstra(graph, lambda _: 1.0)

        # Initialize tsNET with builder
        ts_net = (
            eg.TsNetBuilder()
            .learning_rate(2.0)
            .iterations_stage1(0)
            .iterations_stage2(10)
            .iterations_stage3(10)
            .lambda_c_stage2(1.2)
            .lambda_c_stage3(0.01)
            .lambda_r_stage3(0.6)
            .build()
        )

        ts_net.run(drawing, distance_matrix)

        for u in graph.node_indices():
            self.assertTrue(math.isfinite(drawing.x(u)))
            self.assertTrue(math.isfinite(drawing.y(u)))

    def test_ts_net_with_embedding_distance(self):
        nx_graph = nx.cycle_graph(4)
        graph = draw(nx_graph)

        drawing = eg.DrawingEuclidean2d.initial_placement(graph)

        # Create dummy embedding: 4 nodes, 3 dimensions
        import numpy as np

        emb_data = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        embedding = eg.Array2(emb_data)
        kernel = eg.EmbeddingKernel(graph, embedding)
        edm = eg.KernelDistance(kernel, 1e-3)

        ts_net = (
            eg.TsNetBuilder()
            .learning_rate(2.0)
            .iterations_stage2(10)
            .iterations_stage3(10)
            .build()
        )

        ts_net.run(drawing, edm)

        for u in graph.node_indices():
            self.assertTrue(math.isfinite(drawing.x(u)))
            self.assertTrue(math.isfinite(drawing.y(u)))

    def test_ts_net_with_kernel_distance(self):
        nx_graph = nx.cycle_graph(4)
        graph = draw(nx_graph)

        drawing = eg.DrawingEuclidean2d.initial_placement(graph)

        laplacian = eg.StandardLaplacian.build(graph, lambda _: 1.0)
        rng = eg.Rng.seed_from(123)
        kernel = eg.DiffusionKernel(laplacian, 1.0, 10, rng)
        kd = eg.KernelDistance(kernel, 0.5)

        ts_net = (
            eg.TsNet.builder()
            .learning_rate(2.0)
            .iterations_stage2(10)
            .iterations_stage3(10)
            .build()
        )

        ts_net.run(drawing, kd)

        for u in graph.node_indices():
            self.assertTrue(math.isfinite(drawing.x(u)))
            self.assertTrue(math.isfinite(drawing.y(u)))

    def test_ts_net_validation_error(self):
        builder = eg.TsNetBuilder().perplexity(-1.0)
        with self.assertRaises(ValueError):
            builder.build()


if __name__ == "__main__":
    unittest.main()
