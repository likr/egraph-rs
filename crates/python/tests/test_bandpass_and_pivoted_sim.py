import math
import unittest

import egraph as eg
import numpy as np


HAS_BANDPASS = hasattr(eg, "BandPassDiffusionKernel") and hasattr(
    eg, "PivotedNegLogSimDistance"
)


def build_cycle_graph(n: int) -> eg.Graph:
    graph = eg.Graph()
    nodes = [graph.add_node(i) for i in range(n)]
    for i in range(n):
        graph.add_edge(nodes[i], nodes[(i + 1) % n], i)
    return graph


def exact_cycle_normalized_laplacian(n: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        j = (i + 1) % n
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    return np.eye(n) - 0.5 * adj


def exact_bandpass_matrix(lap_dense: np.ndarray, terms: list[tuple[float, float]]) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(lap_dense)
    weights = np.zeros_like(eigvals)
    for coeff, t in terms:
        weights += coeff * np.exp(-t * eigvals)
    return (eigvecs * weights) @ eigvecs.T


@unittest.skipUnless(HAS_BANDPASS, "requires rebuilt egraph with feat/bandpass-pivoted-kernels")
class TestBandPassAndPivotedSim(unittest.TestCase):
    def test_laplacians_and_optional_length(self):
        n = 8
        g = build_cycle_graph(n)

        l_std_none = eg.StandardLaplacian.build(g)
        l_std_explicit = eg.Laplacian.standard(g, length=None)
        l_std_fn = eg.StandardLaplacian.build(g, lambda _e: 1.0)
        self.assertEqual(l_std_none.n(), n)
        self.assertEqual(l_std_explicit.n(), n)
        self.assertEqual(l_std_fn.n(), n)

        l_sym_none = eg.SymmetricNormalizedLaplacian.build(g)
        l_sym_explicit = eg.Laplacian.symmetric_normalized(g, length=None)
        l_sym_const = eg.Laplacian.symmetric_normalized(g, 1.0)
        self.assertEqual(l_sym_none.n(), n)
        self.assertEqual(l_sym_explicit.n(), n)
        self.assertEqual(l_sym_const.n(), n)

        rng = eg.Rng.seed_from(11)
        k1 = eg.DiffusionKernel(l_sym_none, 1.5, 20, rng)
        k2 = eg.DiffusionKernel(l_sym_explicit, 1.5, 20, eg.Rng.seed_from(11))
        for i in range(n):
            self.assertAlmostEqual(k1.get(0, i), k2.get(0, i), places=10)

        with self.assertRaises(TypeError):
            eg.Laplacian.symmetric_normalized(g, True)

    def test_low_rank_optional_length(self):
        n = 8
        g = build_cycle_graph(n)

        lr_builder = eg.LowRankDiffusionKernelBuilder().t(1.0).rank(4)
        k_none = lr_builder.build_symmetric_normalized_laplacian(g, rng=eg.Rng.seed_from(7))
        k_fn = lr_builder.build_symmetric_normalized_laplacian(
            g, lambda _e: 1.0, eg.Rng.seed_from(7)
        )
        for i in range(n):
            self.assertTrue(math.isfinite(k_none.get(0, i)))
            self.assertAlmostEqual(k_none.get(0, i), k_fn.get(0, i), places=10)

        lrms_builder = eg.LowRankMultiscaleDiffusionKernelBuilder().alpha(0.85).rank(4)
        k_ms = lrms_builder.build_unnormalized_laplacian(g, None, eg.Rng.seed_from(7))
        self.assertTrue(math.isfinite(k_ms.get(0, 1)))

        for bad_alpha in (1.0, 1.2, -0.1, float("nan")):
            with self.assertRaises(ValueError):
                eg.LowRankMultiscaleDiffusionKernelBuilder().alpha(
                    bad_alpha
                ).rank(4).build_unnormalized_laplacian(g, None, eg.Rng.seed_from(7))
            with self.assertRaises(ValueError):
                eg.LowRankMultiscaleDiffusionKernelBuilder().alpha(
                    bad_alpha
                ).rank(4).build_symmetric_normalized_laplacian(g, None, eg.Rng.seed_from(7))
            with self.assertRaises(ValueError):
                eg.LowRankMultiscaleDiffusionKernel(
                    bad_alpha,
                    eg.Array1(np.array([0.0, 1.0], dtype=np.float64)),
                    eg.Array2(np.eye(2, dtype=np.float64)),
                )

    def test_bandpass_diffusion_kernel_new_dog_and_builder(self):
        n = 8
        g = build_cycle_graph(n)
        lap = eg.Laplacian.symmetric_normalized(g)
        lap_dense = exact_cycle_normalized_laplacian(n)

        terms = [(1.2, 0.4), (-0.4, 2.5), (0.2, 0.0)]
        expected = exact_bandpass_matrix(lap_dense, terms)

        k_bp = eg.BandPassDiffusionKernel(lap, terms, 25, rng=eg.Rng.seed_from(42))
        self.assertEqual(k_bp.n(), n)
        for i in range(n):
            for j in range(n):
                self.assertAlmostEqual(k_bp.get(i, j), expected[i, j], delta=1e-3)

        k_bp_lmax = eg.BandPassDiffusionKernel.new_with_lambda_max(
            lap, terms, 25, 2.0, rng=eg.Rng.seed_from(42)
        )
        self.assertAlmostEqual(k_bp_lmax.get(0, 1), expected[0, 1], delta=1e-3)

        t1, t2, alpha, beta_floor = 0.5, 4.0, 0.5, 0.05
        dog_terms = [(1.0 - beta_floor, t1), (-alpha * (1.0 - beta_floor), t2)]
        expected_dog = exact_bandpass_matrix(lap_dense, dog_terms) + beta_floor

        k_dog = eg.BandPassDiffusionKernel.dog(
            lap, t1, t2, alpha, beta_floor, 25, rng=eg.Rng.seed_from(42)
        )
        k_dog_built = (
            eg.BandPassDiffusionKernelBuilder.dog(t1, t2, alpha, beta_floor, degree=25)
            .lambda_max(2.0)
            .build(lap, rng=eg.Rng.seed_from(42))
        )
        for i in range(n):
            for j in range(n):
                self.assertAlmostEqual(k_dog.get(i, j), expected_dog[i, j], delta=1e-3)
                self.assertAlmostEqual(k_dog_built.get(i, j), expected_dog[i, j], delta=1e-3)

        # Zero-DC DoG (alpha = 1.0) is supported without division-by-(1 - alpha) singularity
        k_dog_alpha1 = eg.BandPassDiffusionKernel.dog(
            lap, t1, t2, 1.0, beta_floor, 25, rng=eg.Rng.seed_from(42)
        )
        expected_dog_alpha1 = (
            exact_bandpass_matrix(
                lap_dense, [(1.0 - beta_floor, t1), (-(1.0 - beta_floor), t2)]
            )
            + beta_floor
        )
        for i in range(n):
            for j in range(n):
                self.assertAlmostEqual(
                    k_dog_alpha1.get(i, j), expected_dog_alpha1[i, j], delta=1e-3
                )

        # off_diagonal_floor lifts off-diagonals uniformly without a zero-time identity spike
        k_floored = (
            eg.BandPassDiffusionKernelBuilder(
                [(0.95, t1), (-0.475, t2)], degree=25, lambda_max=2.0
            )
            .off_diagonal_floor(beta_floor)
            .build(lap, rng=eg.Rng.seed_from(42))
        )
        self.assertAlmostEqual(k_floored.get(0, 3), expected_dog[0, 3], delta=1e-3)

        # Positive 2-scale and 3-scale multiscale_heat constructors
        for ms_terms in (
            [(0.6, 0.5), (0.4, 2.0)],
            [(0.5, 0.5), (0.3, 2.0), (0.2, 8.0)],
        ):
            k_ms = eg.BandPassDiffusionKernel.multiscale_heat(
                lap, ms_terms, 25, rng=eg.Rng.seed_from(42)
            )
            k_ms_built = (
                eg.BandPassDiffusionKernelBuilder.multiscale_heat(ms_terms, degree=25)
                .lambda_max(2.0)
                .build(lap, rng=eg.Rng.seed_from(42))
            )
            expected_ms = exact_bandpass_matrix(lap_dense, ms_terms)
            for i in range(n):
                for j in range(n):
                    self.assertGreater(k_ms.get(i, j), 0.0)
                    self.assertAlmostEqual(k_ms.get(i, j), expected_ms[i, j], delta=1e-3)
                    self.assertAlmostEqual(k_ms_built.get(i, j), expected_ms[i, j], delta=1e-3)

        # Invalid parameters raise ValueError
        for bad_args in (
            (lap, 0.0, t2, alpha, beta_floor, 20),
            (lap, t1, -1.0, alpha, beta_floor, 20),
            (lap, t1, t2, -0.1, beta_floor, 20),
            (lap, t1, t2, alpha, -0.05, 20),
            (lap, t1, t2, alpha, 1.0, 20),
        ):
            with self.assertRaises(ValueError):
                eg.BandPassDiffusionKernel.dog(*bad_args)
        with self.assertRaises(ValueError):
            eg.BandPassDiffusionKernel.multiscale_heat(lap, [(-0.1, 0.5)], 20)

    def test_pivoted_bandpass_and_hutchinson_diagonal(self):
        n = 8
        pivots = [0, 4]
        g = build_cycle_graph(n)
        lap = eg.Laplacian.symmetric_normalized(g)

        pk_diff_none = eg.PivotedDiffusionKernel(
            lap, 1.5, 20, pivots, eg.Rng.seed_from(5), num_hutchinson=None
        )
        self.assertIsNone(pk_diff_none.diagonal(0))

        pk_diff_hutch = eg.PivotedDiffusionKernel(
            lap, 1.5, 20, pivots, eg.Rng.seed_from(5), num_hutchinson=32
        )
        full_diff = eg.DiffusionKernel(lap, 1.5, 20, eg.Rng.seed_from(5))
        for j in range(n):
            diag = pk_diff_hutch.diagonal(j)
            self.assertIsNotNone(diag)
            self.assertTrue(math.isfinite(diag))
            self.assertAlmostEqual(diag, full_diff.get(j, j), delta=0.25)

        t1, t2, alpha, beta_floor = 0.5, 4.0, 0.5, 0.05
        full_bp = eg.BandPassDiffusionKernel.dog(
            lap, t1, t2, alpha, beta_floor, 25, rng=eg.Rng.seed_from(9)
        )
        pivoted_bp = eg.PivotedBandPassDiffusionKernel.dog(
            lap, t1, t2, alpha, beta_floor, 25, pivots, rng=eg.Rng.seed_from(9)
        )
        pivoted_bp_builder = (
            eg.PivotedBandPassDiffusionKernelBuilder.dog(
                t1, t2, alpha, beta_floor, degree=25, pivots=pivots
            )
            .num_hutchinson(32)
            .build(lap, rng=eg.Rng.seed_from(9))
        )
        self.assertEqual(pivoted_bp.pivots(), pivots)
        self.assertEqual(pivoted_bp.n(), n)
        for p_idx, p in enumerate(pivots):
            for j in range(n):
                self.assertAlmostEqual(
                    pivoted_bp.get_from_pivot(p_idx, j), full_bp.get(p, j), delta=1e-3
                )
                self.assertAlmostEqual(
                    pivoted_bp_builder.get_from_pivot(p_idx, j), full_bp.get(p, j), delta=1e-3
                )
        self.assertIsNotNone(pivoted_bp.diagonal(0))

        pivoted_bp_no_hutch = eg.PivotedBandPassDiffusionKernel.dog(
            lap,
            t1,
            t2,
            alpha,
            beta_floor,
            25,
            pivots,
            rng=eg.Rng.seed_from(9),
            num_hutchinson=None,
        )
        self.assertIsNone(pivoted_bp_no_hutch.diagonal(0))

        # Pivoted multiscale_heat + off_diagonal_floor parity
        ms_terms = [(0.5, 0.5), (0.3, 2.0), (0.2, 8.0)]
        full_ms = eg.BandPassDiffusionKernel.multiscale_heat(
            lap, ms_terms, 25, rng=eg.Rng.seed_from(9), off_diagonal_floor=0.05
        )
        piv_ms = eg.PivotedBandPassDiffusionKernel.multiscale_heat(
            lap, ms_terms, 25, pivots, rng=eg.Rng.seed_from(9), off_diagonal_floor=0.05
        )
        piv_ms_builder = (
            eg.PivotedBandPassDiffusionKernelBuilder.multiscale_heat(
                ms_terms, degree=25, pivots=pivots
            )
            .off_diagonal_floor(0.05)
            .build(lap, rng=eg.Rng.seed_from(9))
        )
        for p_idx, p in enumerate(pivots):
            for j in range(n):
                self.assertAlmostEqual(
                    piv_ms.get_from_pivot(p_idx, j), full_ms.get(p, j), delta=1e-3
                )
                self.assertAlmostEqual(
                    piv_ms_builder.get_from_pivot(p_idx, j), full_ms.get(p, j), delta=1e-3
                )

        with self.assertRaises(IndexError):
            pivoted_bp.get_from_pivot(0, n)

    def test_pivoted_neg_log_sim_distance_and_sparse_sgd(self):
        n = 8
        pivots = [0, 4]
        g = build_cycle_graph(n)
        lap = eg.Laplacian.symmetric_normalized(g)

        pk_bp = eg.PivotedBandPassDiffusionKernel.dog(
            lap,
            0.5,
            4.0,
            0.5,
            0.05,
            25,
            pivots,
            rng=eg.Rng.seed_from(17),
            num_hutchinson=64,
        )
        dist = eg.PivotedNegLogSimDistance(g, pk_bp)
        self.assertEqual(dist.shape(), (n, len(pivots)))
        self.assertEqual(dist.pivots(), pivots)

        dist_scaled = (
            eg.PivotedNegLogSimDistanceBuilder()
            .epsilon(1e-12)
            .scale(2.0)
            .power(0.5)
            .build(g, pk_bp)
        )

        for p_idx, p in enumerate(pivots):
            self.assertEqual(dist.get(p, p), 0.0)
            self.assertEqual(dist.get_by_index(p, p_idx), 0.0)
            for j in range(n):
                d_row_col = dist.get(j, p)
                d_sym = dist.get(p, j)
                d_idx = dist.get_by_index(j, p_idx)
                self.assertIsNotNone(d_row_col)
                self.assertAlmostEqual(d_row_col, d_idx, places=12)
                self.assertAlmostEqual(d_sym, d_idx, places=12)
                self.assertAlmostEqual(dist_scaled.get(j, p), 2.0 * d_idx, places=12)
        self.assertIsNone(dist.get(1, 2))

        rng = eg.Rng.seed_from(23)
        sgd_direct = eg.SparseSgd().h(2).build(g, rng=rng)
        sgd_shorthand = eg.SparseSgd().h(2).build(g, rng)
        sgd_piv = eg.SparseSgd().h(2).build_with_pivoted_distance(g, d=dist)
        sgd_piv_pos = eg.SparseSgd().h(2).build_with_pivoted_distance(g, None, dist)
        sgd_piv_short = eg.SparseSgd().h(2).build_with_pivoted_distance(g, dist)
        sgd_piv_explicit = (
            eg.SparseSgd().h(2).build_with_pivot_and_distance_matrix(g, None, pivots, dist)
        )
        sgd_np = eg.SparseSgd().h(2).new_with_pivot(g, pivot=pivots)

        pk_diff = eg.PivotedDiffusionKernel(
            lap, 1.5, 25, pivots, eg.Rng.seed_from(29), num_hutchinson=32
        )
        dist_diff = eg.PivotedNegLogSimDistance(g, pk_diff)
        sgd_diff = eg.SparseSgd().h(2).build_with_pivoted_distance(g, d=dist_diff)

        for sgd in (
            sgd_direct,
            sgd_shorthand,
            sgd_piv,
            sgd_piv_pos,
            sgd_piv_short,
            sgd_piv_explicit,
            sgd_np,
            sgd_diff,
        ):
            drawing = eg.DrawingEuclidean2d.initial_placement(g)
            sched = sgd.scheduler(10, 0.1)
            sched.run(lambda eta: (sgd.shuffle(rng), sgd.apply(drawing, eta)))
            coords = [(drawing.x(i), drawing.y(i)) for i in range(n)]
            for x, y in coords:
                self.assertTrue(math.isfinite(x))
                self.assertTrue(math.isfinite(y))
            spread = sum(x * x + y * y for x, y in coords)
            self.assertGreater(spread, 1e-6)


if __name__ == "__main__":
    unittest.main()
