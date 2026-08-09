import unittest
import egraph as eg


class TestMultiscaleDiffusionKernel(unittest.TestCase):
    def setUp(self):
        """Set up a simple 4-node path graph: 0 - 1 - 2 - 3"""
        self.graph = eg.Graph()
        self.n0 = self.graph.add_node(0)
        self.n1 = self.graph.add_node(1)
        self.n2 = self.graph.add_node(2)
        self.n3 = self.graph.add_node(3)
        self.graph.add_edge(self.n0, self.n1, None)
        self.graph.add_edge(self.n1, self.n2, None)
        self.graph.add_edge(self.n2, self.n3, None)

        self.rng = eg.Rng.seed_from(42)

    def test_basic_construction(self):
        """Test basic MultiscaleDiffusionKernel construction and node count"""
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        mdk = eg.MultiscaleDiffusionKernel(laplacian, 0.85)

        self.assertEqual(mdk.n(), 4)






if __name__ == "__main__":
    unittest.main()
