import unittest
import networkx as nx
import egraph as eg


class TestRng(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a sample graph for testing
        cls._graph = eg.Graph()
        for i in range(10):
            cls._graph.add_node(i)
        for i in range(9):
            cls._graph.add_edge(i, i + 1, i)

    def test_constructor(self):
        """Test the default constructor that uses system entropy"""
        rng = eg.Rng()
        # We can't test the actual randomness, but we can ensure it doesn't crash
        self.assertIsNotNone(rng)

    def test_seed_constructor(self):
        """Test the seed-based constructor"""
        rng = eg.Rng.seed_from(42)
        self.assertIsNotNone(rng)

    def test_deterministic_results(self):
        """Test that the same seed produces deterministic results"""
        # Create two RNGs with the same seed
        rng1 = eg.Rng.seed_from(42)
        rng2 = eg.Rng.seed_from(42)

        # Use them with SGD to verify deterministic behavior
        drawing1 = eg.DrawingEuclidean2d.initial_placement(self._graph)
        drawing2 = eg.DrawingEuclidean2d.initial_placement(self._graph)

        # Record initial positions to verify they're the same
        initial_positions1 = {u: (drawing1.x(u), drawing1.y(u))
                              for u in self._graph.node_indices()}
        initial_positions2 = {u: (drawing2.x(u), drawing2.y(u))
                              for u in self._graph.node_indices()}

        # Verify initial positions are the same
        for u in self._graph.node_indices():
            self.assertEqual(initial_positions1[u], initial_positions2[u])

        # Create SGD instances with the same parameters
        sgd1 = eg.FullSgd(self._graph, lambda _: 1.0)
        sgd2 = eg.FullSgd(self._graph, lambda _: 1.0)

        # Apply SGD with the two RNGs
        sgd1.shuffle(rng1)
        sgd1.apply(drawing1, 0.1)

        sgd2.shuffle(rng2)
        sgd2.apply(drawing2, 0.1)

        # Verify positions after SGD are the same
        for u in self._graph.node_indices():
            self.assertAlmostEqual(drawing1.x(u), drawing2.x(u), places=6)
            self.assertAlmostEqual(drawing1.y(u), drawing2.y(u), places=6)

    def test_different_seeds(self):
        """Test that different seeds produce different results"""
        # Create two RNGs with different seeds
        rng1 = eg.Rng.seed_from(42)
        rng2 = eg.Rng.seed_from(43)

        # Use them with SGD to verify different behavior
        drawing1 = eg.DrawingEuclidean2d.initial_placement(self._graph)
        drawing2 = eg.DrawingEuclidean2d.initial_placement(self._graph)

        # Record initial positions
        initial_positions1 = {u: (drawing1.x(u), drawing1.y(u))
                              for u in self._graph.node_indices()}
        initial_positions2 = {u: (drawing2.x(u), drawing2.y(u))
                              for u in self._graph.node_indices()}

        # Verify initial positions are the same
        for u in self._graph.node_indices():
            self.assertEqual(initial_positions1[u], initial_positions2[u])

        # Create SGD instances with the same parameters
        sgd1 = eg.FullSgd(self._graph, lambda _: 1.0)
        sgd2 = eg.FullSgd(self._graph, lambda _: 1.0)

        # Apply SGD with the two RNGs
        sgd1.shuffle(rng1)
        sgd1.apply(drawing1, 0.1)

        sgd2.shuffle(rng2)
        sgd2.apply(drawing2, 0.1)

        # Verify at least some positions after SGD are different
        different_positions = False
        for u in self._graph.node_indices():
            if (abs(drawing1.x(u) - drawing2.x(u)) > 1e-6 or
                    abs(drawing1.y(u) - drawing2.y(u)) > 1e-6):
                different_positions = True
                break

        self.assertTrue(different_positions,
                        "Different seeds should produce different layouts")

    def test_multiple_shuffles(self):
        """Test that multiple shuffles with the same RNG produce different results"""
        rng = eg.Rng.seed_from(42)
        drawing1 = eg.DrawingEuclidean2d.initial_placement(self._graph)
        drawing2 = eg.DrawingEuclidean2d.initial_placement(self._graph)

        # Create SGD instances with the same parameters
        sgd1 = eg.FullSgd(self._graph, lambda _: 1.0)
        sgd2 = eg.FullSgd(self._graph, lambda _: 1.0)

        # Apply SGD with first shuffle
        sgd1.shuffle(rng)
        sgd1.apply(drawing1, 0.1)

        # Apply SGD with second shuffle of the same RNG
        sgd2.shuffle(rng)
        sgd2.apply(drawing2, 0.1)

        # Verify at least some positions after SGD are different
        different_positions = False
        for u in self._graph.node_indices():
            if (abs(drawing1.x(u) - drawing2.x(u)) > 1e-6 or
                    abs(drawing1.y(u) - drawing2.y(u)) > 1e-6):
                different_positions = True
                break

        self.assertTrue(different_positions,
                        "Multiple shuffles should produce different layouts")

    def test_with_mds(self):
        """Test RNG integration with MDS algorithm"""
        # Create two RNGs with the same seed
        rng1 = eg.Rng.seed_from(42)
        rng2 = eg.Rng.seed_from(42)

        # Use them with MDS to verify deterministic behavior
        # First, create distance matrices
        d1 = eg.all_sources_bfs(self._graph, 1.0)
        d2 = eg.all_sources_bfs(self._graph, 1.0)

        # Create MDS instances
        mds1 = eg.ClassicalMds.new_with_distance_matrix(d1)
        mds2 = eg.ClassicalMds.new_with_distance_matrix(d2)

        # Run MDS to get drawings
        drawing1 = mds1.run_2d()
        drawing2 = mds2.run_2d()

        # Apply SGD with the two RNGs for refinement
        sgd1 = eg.FullSgd.new_with_distance_matrix(d1)
        sgd2 = eg.FullSgd.new_with_distance_matrix(d2)

        sgd1.shuffle(rng1)
        sgd1.apply(drawing1, 0.1)

        sgd2.shuffle(rng2)
        sgd2.apply(drawing2, 0.1)

        # Verify positions after SGD are the same
        for u in self._graph.node_indices():
            self.assertAlmostEqual(drawing1.x(u), drawing2.x(u), places=6)
            self.assertAlmostEqual(drawing1.y(u), drawing2.y(u), places=6)

    def test_extreme_seeds(self):
        """Test with extreme seed values"""
        # Test with minimum seed value
        rng_min = eg.Rng.seed_from(0)
        self.assertIsNotNone(rng_min)

        # Test with maximum seed value
        rng_max = eg.Rng.seed_from(2**64 - 1)
        self.assertIsNotNone(rng_max)

        # Verify they produce different results
        drawing_min = eg.DrawingEuclidean2d.initial_placement(self._graph)
        drawing_max = eg.DrawingEuclidean2d.initial_placement(self._graph)

        sgd_min = eg.FullSgd(self._graph, lambda _: 1.0)
        sgd_max = eg.FullSgd(self._graph, lambda _: 1.0)

        sgd_min.shuffle(rng_min)
        sgd_min.apply(drawing_min, 0.1)

        sgd_max.shuffle(rng_max)
        sgd_max.apply(drawing_max, 0.1)

        # Verify at least some positions after SGD are different
        different_positions = False
        for u in self._graph.node_indices():
            if (abs(drawing_min.x(u) - drawing_max.x(u)) > 1e-6 or
                    abs(drawing_min.y(u) - drawing_max.y(u)) > 1e-6):
                different_positions = True
                break

        self.assertTrue(
            different_positions, "Different extreme seeds should produce different layouts")


if __name__ == "__main__":
    unittest.main()
