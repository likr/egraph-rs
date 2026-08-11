import unittest
import egraph as eg


class TestKernelBuilders(unittest.TestCase):
    def test_builders(self):
        graph = eg.Graph()
        a = graph.add_node(0)
        b = graph.add_node(1)
        c = graph.add_node(2)
        graph.add_edge(a, b, None)
        graph.add_edge(b, c, None)

        rng = eg.Rng.seed_from(42)
        rdmds = eg.RdMds().d(2)
        embedding = rdmds.embedding(graph, lambda e: 1.0, eg.Ic0CgSolver(), rng)

        # 1. EmbeddingKernelBuilder
        emb_kernel = eg.EmbeddingKernelBuilder().build(graph, embedding)
        self.assertIsNotNone(emb_kernel)

        # 2. GaussianKernelBuilder
        g_kernel = eg.GaussianKernelBuilder().gamma(0.5).build(emb_kernel)
        self.assertIsNotNone(g_kernel)

        # 3. ExponentialKernelBuilder
        exp_kernel = eg.ExponentialKernelBuilder().gamma(0.5).build(emb_kernel)
        self.assertIsNotNone(exp_kernel)

        # 4. TKernelBuilder
        t_kernel = eg.TKernelBuilder().alpha(2.0).build(emb_kernel)
        self.assertIsNotNone(t_kernel)

        # 5. KernelDistanceBuilder
        k_dist = eg.KernelDistanceBuilder().min_dist(1e-3).build(t_kernel)
        self.assertIsNotNone(k_dist)


if __name__ == "__main__":
    unittest.main()
