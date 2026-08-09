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
        mdk = eg.MultiscaleDiffusionKernel(
            laplacian,
            0.85,  # alpha
            32,  # num_samples
            1e-7,  # tol
            100,  # max_iter
            self.rng,
        )

        self.assertEqual(mdk.n(), 4)

    def test_construction_with_symmetric_normalized(self):
        """Test MultiscaleDiffusionKernel construction with SymmetricNormalizedLaplacian"""
        laplacian = eg.SymmetricNormalizedLaplacian.build(self.graph, lambda i: 1.0)
        mdk = eg.MultiscaleDiffusionKernel(
            laplacian,
            0.85,
            32,
            1e-7,
            100,
            self.rng,
        )

        self.assertEqual(mdk.n(), 4)

    def test_distance_queries(self):
        """Test querying distances with get and sample_distance"""
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        mdk = eg.MultiscaleDiffusionKernel(
            laplacian,
            0.85,
            32,
            1e-7,
            100,
            self.rng,
        )

        d00 = mdk.get(0, 0)
        d01 = mdk.get(0, 1)
        d03 = mdk.get(0, 3)

        self.assertAlmostEqual(d00, 0.0, places=10)
        self.assertGreater(d01, 0.0)
        self.assertGreater(d03, d01)

        # sample_distance should return the same result
        d01_sample = mdk.sample_distance(0, 1)
        self.assertAlmostEqual(d01, d01_sample, places=10)

    def test_symmetry(self):
        """Test distance symmetry d(i, j) == d(j, i)"""
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        mdk = eg.MultiscaleDiffusionKernel(
            laplacian,
            0.85,
            64,
            1e-7,
            100,
            self.rng,
        )

        d01 = mdk.get(0, 1)
        d10 = mdk.get(1, 0)
        self.assertAlmostEqual(d01, d10, places=10)

        d03 = mdk.get(0, 3)
        d30 = mdk.get(3, 0)
        self.assertAlmostEqual(d03, d30, places=10)

    def test_rebuild_index(self):
        """Test explicitly re-building index with build_index"""
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        mdk = eg.MultiscaleDiffusionKernel(
            laplacian,
            0.85,
            32,
            1e-7,
            100,
            self.rng,
        )

        d01_before = mdk.get(0, 1)

        rng2 = eg.Rng.seed_from(12345)
        mdk.build_index(rng2)

        d01_after = mdk.get(0, 1)
        self.assertGreater(d01_after, 0.0)

    def test_distance_matrix_integration(self):
        """Test MultiscaleDiffusionDistanceMatrix integration"""
        laplacian = eg.StandardLaplacian.build(self.graph, lambda i: 1.0)
        mdk = eg.MultiscaleDiffusionKernel(
            laplacian,
            0.85,
            32,
            1e-7,
            100,
            self.rng,
        )

        dist_matrix = eg.MultiscaleDiffusionDistanceMatrix(self.graph, mdk, 1e-4)

        d00 = dist_matrix.get(0, 0)
        d01 = dist_matrix.get(0, 1)

        self.assertEqual(d00, 0.0)
        self.assertGreater(d01, 0.0)

        # Test using MultiscaleDiffusionDistanceMatrix with SGD
        sparse_sgd = eg.RandomPairSparseSgd()
        sgd_rng = eg.Rng.seed_from(100)
        sgd_layout = sparse_sgd.build(self.graph, dist_matrix, sgd_eg.Ic0CgSolver(), rng)
        self.assertIsNotNone(sgd_layout)


if __name__ == "__main__":
    unittest.main()
