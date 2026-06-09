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
    def test_ts_net_with_distance_matrix(self):
        # Create a simple cycle graph of size 4
        nx_graph = nx.cycle_graph(4)
        graph = draw(nx_graph)

        drawing = eg.DrawingEuclidean2d.initial_placement(graph)
        distance_matrix = eg.all_sources_dijkstra(graph, lambda _: 1.0)

        # Initialize tsNET with a low learning rate to be stable on small graph
        ts_net = eg.TsNet()
        ts_net.learning_rate(2.0)
        ts_net.iterations_stage2(10)
        ts_net.iterations_stage3(10)

        ts_net.run(drawing, distance_matrix)

        for u in graph.node_indices():
            self.assertTrue(math.isfinite(drawing.x(u)))
            self.assertTrue(math.isfinite(drawing.y(u)))

    def test_ts_net_with_diffusion_distance(self):
        nx_graph = nx.cycle_graph(4)
        graph = draw(nx_graph)

        drawing = eg.DrawingEuclidean2d.initial_placement(graph)
        rng = eg.Rng.seed_from(42)

        # Create DiffusionKernel and DiffusionDistanceMatrix
        dk = eg.DiffusionKernel(graph, lambda _: 1.0, 1000.0, 10, 50, rng)
        ddm = eg.DiffusionDistanceMatrix(graph, dk, 1e-3)

        ts_net = eg.TsNet()
        ts_net.learning_rate(2.0)
        ts_net.iterations_stage2(10)
        ts_net.iterations_stage3(10)

        ts_net.run(drawing, ddm)

        for u in graph.node_indices():
            self.assertTrue(math.isfinite(drawing.x(u)))
            self.assertTrue(math.isfinite(drawing.y(u)))

    def test_ts_net_with_embedding_distance(self):
        nx_graph = nx.cycle_graph(4)
        graph = draw(nx_graph)

        drawing = eg.DrawingEuclidean2d.initial_placement(graph)

        # Create dummy embedding: 4 nodes, 3 dimensions
        import numpy as np
        emb_data = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0]
        ], dtype=np.float64)
        embedding = eg.Array2(emb_data)
        edm = eg.EmbeddingDistanceMatrix(graph, embedding, 1e-3)

        ts_net = eg.TsNet()
        ts_net.learning_rate(2.0)
        ts_net.iterations_stage2(10)
        ts_net.iterations_stage3(10)

        ts_net.run(drawing, edm)

        for u in graph.node_indices():
            self.assertTrue(math.isfinite(drawing.x(u)))
            self.assertTrue(math.isfinite(drawing.y(u)))

    def test_ts_net_with_kernel_distance(self):
        nx_graph = nx.cycle_graph(4)
        graph = draw(nx_graph)

        drawing = eg.DrawingEuclidean2d.initial_placement(graph)
        distance_matrix = eg.DistanceMatrix(graph)

        # Wrap distance matrix in KernelDistance with gamma = 0.5
        kd = eg.KernelDistance(distance_matrix, 0.5)

        ts_net = eg.TsNet()
        ts_net.learning_rate(2.0)
        ts_net.iterations_stage2(10)
        ts_net.iterations_stage3(10)

        ts_net.run(drawing, kd)

        for u in graph.node_indices():
            self.assertTrue(math.isfinite(drawing.x(u)))
            self.assertTrue(math.isfinite(drawing.y(u)))


if __name__ == "__main__":
    unittest.main()
