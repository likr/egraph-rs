from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import inspect
import math
import threading
import unittest
from typing import Any, Callable

import egraph as eg
import numpy as np
import numpy.typing as npt


def _make_grid(rows: int, cols: int) -> eg.CsrGraph:
    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            if c + 1 < cols:
                edges.append((u, u + 1))
            if r + 1 < rows:
                edges.append((u, u + cols))
    return eg.CsrGraph.from_edges(rows * cols, edges, directed=False, sanitize=True)


def _make_barbell(clique_size: int, bridge_len: int = 1) -> eg.CsrGraph:
    n = 2 * clique_size + bridge_len - 1
    edges: list[tuple[int, int]] = []
    for i in range(clique_size):
        for j in range(i + 1, clique_size):
            edges.append((i, j))
    offset = clique_size + bridge_len - 1
    for i in range(clique_size):
        for j in range(i + 1, clique_size):
            edges.append((offset + i, offset + j))
    prev = clique_size - 1
    for b in range(bridge_len - 1):
        cur = clique_size + b
        edges.append((prev, cur))
        prev = cur
    edges.append((prev, offset))
    return eg.CsrGraph.from_edges(n, edges, directed=False, sanitize=True)


def _seeded_coords(n: int, seed: int = 42, scale: float = 1.0) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=(n, 2)).astype(np.float64)


def _to_float_array(coords: npt.NDArray[np.float64], dtype_str: str) -> npt.NDArray[np.floating[Any]]:
    if dtype_str == "float32":
        return coords.astype(np.float32)
    return coords.astype(np.float64)


class TestV6Clustering(unittest.TestCase):
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
        fn: Callable[[], Any],
    ) -> None:
        with self.assertRaises(expected) as ctx:
            fn()
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

    def test_kmeans_plus_plus_determinism_and_validation(self) -> None:
        pts = np.array(
            [
                [10.0, 10.0],
                [10.2, 9.8],
                [9.8, 10.1],
                [-10.0, -10.0],
                [-9.9, -10.2],
                [-10.1, -9.8],
            ],
            dtype=np.float64,
        )
        r1 = eg.kmeans_plus_plus(pts, k=2, max_iters=100, tol=1e-4, n_init=3, seed=42)
        r2 = eg.kmeans_plus_plus(pts.astype(np.float32), k=2, n_init=3, seed=42)
        labels1 = r1["labels"]
        centroids1 = r1["centroids"]
        self.assertEqual(labels1.dtype, np.uint32)
        self.assertEqual(centroids1.dtype, np.float64)
        self.assertEqual(centroids1.shape, (2, 2))
        np.testing.assert_array_equal(labels1, np.array([0, 0, 0, 1, 1, 1], dtype=np.uint32))
        np.testing.assert_array_equal(labels1, r2["labels"])
        self.assertTrue(bool(r1["converged"]))
        self.assertLess(float(r1["inertia"]), 1.0)
        self.assertGreaterEqual(int(r1["iterations"]), 1)

        # Python list input and k=1 / k=n boundary
        r_list = eg.kmeans_plus_plus([[0.0, 0.0], [1.0, 0.0]], k=1, seed=42)
        np.testing.assert_array_equal(r_list["labels"], np.array([0, 0], dtype=np.uint32))
        r_kn = eg.kmeans_plus_plus([[0.0, 0.0], [1.0, 0.0]], k=2, seed=42)
        np.testing.assert_array_equal(r_kn["labels"], np.array([0, 1], dtype=np.uint32))

        # Validation errors
        self.assert_no_panic_raises((ValueError,), lambda: eg.kmeans_plus_plus(pts, k=0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.kmeans_plus_plus(pts, k=10))
        self.assert_no_panic_raises((ValueError,), lambda: eg.kmeans_plus_plus(pts, max_iters=0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.kmeans_plus_plus(pts, n_init=0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.kmeans_plus_plus(pts, tol=-0.1))
        self.assert_no_panic_raises((ValueError,), lambda: eg.kmeans_plus_plus(np.empty((0, 2)), k=1))
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.kmeans_plus_plus(np.array([[0.0, float("nan")]]), k=1)
        )
        self.assert_no_panic_raises((TypeError,), lambda: eg.kmeans_plus_plus(pts, k=True))
        self.assert_no_panic_raises((TypeError,), lambda: eg.kmeans_plus_plus("bad"))  # type: ignore[arg-type]

    def test_spectral_clustering_barbell_and_directed_rejection(self) -> None:
        barbell = _make_barbell(6, bridge_len=1)
        l1 = eg.spectral_clustering(barbell, k=2, seed=42)
        l2 = eg.spectral_clustering(barbell, k=2, seed=42)
        self.assertEqual(l1.dtype, np.uint32)
        self.assertEqual(l1.shape, (12,))
        np.testing.assert_array_equal(l1, l2)
        np.testing.assert_array_equal(l1[:6], np.zeros(6, dtype=np.uint32))
        np.testing.assert_array_equal(l1[6:], np.ones(6, dtype=np.uint32))

        # Larger barbell (> 48 nodes) exercises Lanczos path
        large_barbell = _make_barbell(30, bridge_len=1)
        l_large = eg.spectral_clustering(large_barbell, k=2, seed=42)
        self.assertEqual(l_large.shape, (60,))
        np.testing.assert_array_equal(l_large[:30], np.zeros(30, dtype=np.uint32))
        np.testing.assert_array_equal(l_large[30:], np.ones(30, dtype=np.uint32))

        # Directed graph raises ValueError cleanly
        g_dir = eg.CsrGraph.from_edges(4, [(0, 1), (1, 2), (2, 3)], directed=True, sanitize=True)
        self.assert_no_panic_raises((ValueError,), lambda: eg.spectral_clustering(g_dir, k=2))
        self.assert_no_panic_raises((ValueError,), lambda: eg.spectral_clustering(barbell, k=0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.spectral_clustering(barbell, eigen_tol=0.0))
        self.assert_no_panic_raises((TypeError,), lambda: eg.spectral_clustering(barbell, k=True))

    def test_louvain_and_label_propagation(self) -> None:
        barbell = _make_barbell(6, bridge_len=1)
        louv1 = eg.louvain_communities(barbell, max_passes=100, resolution=1.0, min_modularity_gain=1e-7, seed=42)
        louv2 = eg.louvain_communities(barbell, seed=42)
        self.assertEqual(louv1["num_communities"], 2)
        self.assertGreater(float(louv1["modularity"]), 0.4)
        self.assertGreaterEqual(int(louv1["passes"]), 1)
        self.assertEqual(louv1["labels"].dtype, np.uint32)
        np.testing.assert_array_equal(louv1["labels"], np.array([0] * 6 + [1] * 6, dtype=np.uint32))
        np.testing.assert_array_equal(louv1["labels"], louv2["labels"])

        lp1 = eg.label_propagation(barbell, max_iters=100, seed=42)
        lp2 = eg.label_propagation(barbell, max_iters=100, seed=42)
        self.assertEqual(lp1.dtype, np.uint32)
        np.testing.assert_array_equal(lp1, lp2)
        np.testing.assert_array_equal(lp1[:6], np.zeros(6, dtype=np.uint32))
        np.testing.assert_array_equal(lp1[6:], np.ones(6, dtype=np.uint32))

        g_dir = eg.CsrGraph.from_edges(3, [(0, 1), (1, 2)], directed=True, sanitize=True)
        self.assert_no_panic_raises((ValueError,), lambda: eg.louvain_communities(g_dir))
        self.assert_no_panic_raises((ValueError,), lambda: eg.label_propagation(g_dir))
        self.assert_no_panic_raises((ValueError,), lambda: eg.louvain_communities(barbell, max_passes=0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.louvain_communities(barbell, resolution=-1.0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.label_propagation(barbell, max_iters=0))

    def test_coarsen_and_project_clustered_layout_f64_and_f32(self) -> None:
        edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (0, 3), (2, 4)]
        g = eg.CsrGraph.from_edges(6, edges, directed=False, sanitize=True)
        # Non-contiguous labels are compacted canonically
        coarse_g, n2c, sizes, num_clusters = eg.coarsen_graph(g, [100, 100, 100, 999, 999, 999])
        self.assertEqual(num_clusters, 2)
        self.assertEqual(coarse_g.num_nodes, 2)
        self.assertEqual(coarse_g.num_edges, 1)
        w = coarse_g.weights()
        assert w is not None
        np.testing.assert_allclose(w, [2.0, 2.0])
        self.assertEqual(n2c.dtype, np.uint32)
        self.assertEqual(sizes.dtype, np.uint32)
        np.testing.assert_array_equal(n2c, np.array([0, 0, 0, 1, 1, 1], dtype=np.uint32))
        np.testing.assert_array_equal(sizes, np.array([3, 3], dtype=np.uint32))

        base_coarse = np.array([[-5.0, 1.0], [5.0, -1.0]], dtype=np.float64)
        for dtype_str, np_dtype in (("float64", np.float64), ("float32", np.float32)):
            coarse_coords = _to_float_array(base_coarse, dtype_str)
            cl = eg.Layout2d.from_coords(coarse_coords)
            f1 = eg.project_clustered_layout(cl, n2c, spread_radius=0.5, seed=42)
            f2 = eg.project_clustered_layout(cl, n2c, spread_radius=0.5, seed=42)
            self.assertEqual(f1.dtype, dtype_str)
            self.assertEqual(len(f1), 6)
            self.assertTrue(f1.is_all_finite())
            c1 = f1.coords(copy=True)
            self.assertEqual(c1.dtype, np_dtype)
            np.testing.assert_array_equal(c1, f2.coords(copy=True))
            for i in range(6):
                cid = int(n2c[i])
                dist = math.hypot(float(c1[i, 0] - coarse_coords[cid, 0]), float(c1[i, 1] - coarse_coords[cid, 1]))
                self.assertLessEqual(dist, 0.5 + 1e-5)

            exact = eg.project_clustered_layout(cl, n2c, spread_radius=0.0, seed=42)
            np.testing.assert_allclose(exact.coords(copy=True)[:3], np.tile(coarse_coords[0], (3, 1)))
            np.testing.assert_allclose(exact.coords(copy=True)[3:], np.tile(coarse_coords[1], (3, 1)))

        # Errors
        g_dir = eg.CsrGraph.from_edges(2, [(0, 1)], directed=True, sanitize=True)
        self.assert_no_panic_raises((ValueError,), lambda: eg.coarsen_graph(g_dir, [0, 1]))
        self.assert_no_panic_raises((ValueError,), lambda: eg.coarsen_graph(g, [0, 1]))
        cl2 = eg.Layout2d.from_coords([[0.0, 0.0], [1.0, 1.0]])
        self.assert_no_panic_raises((IndexError,), lambda: eg.project_clustered_layout(cl2, [0, 2]))
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.project_clustered_layout(cl2, [0, 1], spread_radius=-0.1)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.project_clustered_layout(cl2, [0, 1], spread_radius=float("nan"))
        )


class TestV6ManifoldAndSolvers(unittest.TestCase):
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
        fn: Callable[[], Any],
    ) -> None:
        with self.assertRaises(expected) as ctx:
            fn()
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

    def test_manifold_layout2d_all_manifolds_f64_and_f32(self) -> None:
        for raw_kind, canonical in (
            ("hyperbolic", "hyperbolic"),
            ("hyperbolic_poincare", "hyperbolic"),
            ("poincare", "hyperbolic"),
            ("spherical", "spherical"),
            ("sphere", "spherical"),
            ("torus", "torus"),
            ("torus-2d", "torus"),
        ):
            mz = eg.ManifoldLayout2d.zeros(raw_kind, 4, dtype="float32")
            self.assertEqual(mz.kind, canonical)
            self.assertEqual(mz.dtype, "float32")
            self.assertEqual(mz.len(), 4)
            self.assertEqual(len(mz), 4)
            self.assertFalse(mz.is_empty())
            self.assertIn("ManifoldLayout2d(", repr(mz))

        # Hyperbolic Poincaré clamping + geodesic distance
        mh = eg.ManifoldLayout2d.from_coords("hyperbolic", [[0.0, 0.0], [3.0, 4.0]])
        ch = mh.coords()
        self.assertEqual(ch.dtype, np.float64)
        self.assertLess(math.hypot(float(ch[1, 0]), float(ch[1, 1])), 1.0)
        mh2 = eg.ManifoldLayout2d.from_coords("hyperbolic", np.array([[0.0, 0.0], [0.5, 0.0]], dtype=np.float32))
        self.assertEqual(mh2.dtype, "float32")
        self.assertAlmostEqual(mh2.geodesic_distance(0, 1), 2.0 * math.atanh(0.5), places=5)

        # Spherical projection + equator quarter-circle distance
        ms = eg.ManifoldLayout2d.from_coords("spherical", [[-0.5, -0.25], [math.pi / 2.0, math.pi / 2.0]])
        cs = ms.coords()
        self.assertTrue(0.0 <= float(cs[0, 0]) < 2.0 * math.pi)
        self.assertTrue(0.0 <= float(cs[0, 1]) <= math.pi)
        ms_eq = eg.ManifoldLayout2d.from_coords(
            "spherical", [[0.2, math.pi / 2.0], [0.2 + math.pi / 2.0, math.pi / 2.0]]
        )
        self.assertAlmostEqual(ms_eq.geodesic_distance(0, 1), math.pi / 2.0, places=10)

        # Torus periodic wrapping + geodesic distance
        l_torus_in = eg.Layout2d.from_coords([[1.95, -0.95], [0.05, 0.95]])
        mt = eg.ManifoldLayout2d.from_layout("torus", l_torus_in)
        mt.project_all()
        ct = mt.coords()
        self.assertTrue(np.all((ct >= 0.0) & (ct < 1.0)))
        self.assertAlmostEqual(mt.geodesic_distance(0, 1), math.hypot(0.1, 0.1), places=10)
        back = mt.to_layout()
        self.assertEqual(back.dtype, "float64")
        np.testing.assert_allclose(back.coords(copy=True), ct)

        # Errors
        self.assert_no_panic_raises((ValueError,), lambda: eg.ManifoldLayout2d.zeros("euclidean", 3))
        self.assert_no_panic_raises((TypeError,), lambda: eg.ManifoldLayout2d.zeros(123, 3))  # type: ignore[arg-type]
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.ManifoldLayout2d.from_coords("torus", [[0.0, float("nan")]])
        )
        self.assert_no_panic_raises((IndexError,), lambda: mt.geodesic_distance(0, 99))
        self.assert_no_panic_raises((TypeError,), lambda: eg.ManifoldLayout2d.zeros("torus", True))

    def test_tsnet_solver_all_backends_step_vs_run_and_zero_copy(self) -> None:
        g = _make_grid(4, 4)
        init_coords = _seeded_coords(16, seed=11, scale=1.0)

        for backend in ("fft", "linear_fft", "barnes_hut", "exact"):
            for dtype_str, np_dtype in (("float64", np.float64), ("float32", np.float32)):
                kwargs: dict[str, Any] = dict(
                    backend=backend,
                    iterations_stage1=4,
                    iterations_stage2=4,
                    iterations_stage3=4,
                    seed=42,
                    dtype=dtype_str,
                )
                s_run = eg.TsNetSolver.pure_2hop(g, **kwargs)
                s_step = eg.TsNetSolver.pure_2hop(g, **kwargs)
                self.assertEqual(s_run.status, "initialized")
                self.assertEqual(s_run.iteration, 0)
                self.assertEqual(s_run.total_iterations, 12)
                self.assertFalse(s_run.is_finished)
                self.assertEqual(s_run.dtype, dtype_str)

                l_run = eg.Layout2d.from_coords(init_coords.astype(np_dtype))
                l_step = eg.Layout2d.from_coords(init_coords.astype(np_dtype))
                view_run = l_run.coords(copy=False)

                st = s_run.run(l_run)
                self.assertEqual(st, "completed")
                self.assertTrue(s_run.is_finished)
                self.assertEqual(s_run.iteration, 12)
                if backend in ("fft", "linear_fft"):
                    self.assertGreater(s_run.total_fft_2d_calls, 0)
                else:
                    self.assertEqual(s_run.total_fft_2d_calls, 0)

                self.assertEqual(s_step.run_steps(l_step, 5), "running")
                self.assertEqual(s_step.iteration, 5)
                while not s_step.is_finished:
                    s_step.step(l_step)
                self.assertEqual(s_step.status, "completed")
                self.assertEqual(s_step.iteration, 12)

                # Zero-copy view must reflect mutated coordinates without reallocation
                self.assertIs(view_run.base, l_run)
                np.testing.assert_array_equal(view_run, l_run.coords(copy=True))
                np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))

        # Verify default learning_rate=None across all 5 TsNetSolver constructors
        for ctor in (
            eg.TsNetSolver.pure_2hop,
            eg.TsNetSolver.from_knn,
            eg.TsNetSolver.from_probabilities,
            eg.TsNetSolver.from_graph_partial_bfs,
            eg.TsNetSolver.from_graph_row_normalized,
        ):
            self.assertIsNone(inspect.signature(ctor).parameters["learning_rate"].default)

        # from_knn (per-node & scalar perplexity), from_probabilities, and from_graph_row_normalized
        knn, _, _ = eg.build_pure_2hop_knn(g)
        perps = eg.compute_nodewise_2hop_perplexities(g)
        s_knn_vec = eg.TsNetSolver.from_knn(knn, perps, backend="fit", iterations_stage1=2, iterations_stage2=2, iterations_stage3=2)
        s_knn_scalar = eg.TsNetSolver.from_knn(knn, 8.0, backend="bh", iterations_stage1=2, iterations_stage2=2, iterations_stage3=2)
        probs = eg.compute_knn_probabilities(knn, perps)
        s_probs = eg.TsNetSolver.from_probabilities(probs, backend="l", iterations_stage1=2, iterations_stage2=2, iterations_stage3=2)

        for solver in (s_knn_vec, s_knn_scalar, s_probs):
            lay = eg.Layout2d.from_coords(init_coords)
            self.assertEqual(solver.run(lay), "completed")
            self.assertTrue(lay.is_all_finite())

        for dtype_str, np_dtype in (("float64", np.float64), ("float32", np.float32)):
            for max_2hop in (None, 4):
                rn_run = eg.TsNetSolver.from_graph_row_normalized(
                    g,
                    w2=0.5,
                    max_2hop_per_node=max_2hop,
                    backend="fft",
                    iterations_stage1=2,
                    iterations_stage2=2,
                    iterations_stage3=2,
                    seed=42,
                    dtype=dtype_str,
                )
                rn_step = eg.TsNetSolver.from_graph_row_normalized(
                    g,
                    w2=0.5,
                    max_2hop_per_node=max_2hop,
                    backend="fft",
                    iterations_stage1=2,
                    iterations_stage2=2,
                    iterations_stage3=2,
                    seed=42,
                    dtype=dtype_str,
                )
                l_rn_run = eg.Layout2d.from_coords(init_coords.astype(np_dtype))
                l_rn_step = eg.Layout2d.from_coords(init_coords.astype(np_dtype))
                self.assertEqual(rn_run.run(l_rn_run), "completed")
                while not rn_step.is_finished:
                    rn_step.step(l_rn_step)
                np.testing.assert_array_equal(l_rn_run.coords(copy=True), l_rn_step.coords(copy=True))
                # Stage 3 LR cap <= 2 * N for N=16 (< 100)
                self.assertLessEqual(rn_run.current_stage_lr, 2.0 * g.num_nodes)

        # Validation errors
        self.assert_no_panic_raises((ValueError,), lambda: eg.TsNetSolver.pure_2hop(g, backend="bad"))
        self.assert_no_panic_raises((TypeError,), lambda: eg.TsNetSolver.pure_2hop(g, backend=123))  # type: ignore[arg-type]
        self.assert_no_panic_raises((ValueError,), lambda: eg.TsNetSolver.pure_2hop(g, interpolation_order=2))
        self.assert_no_panic_raises((TypeError,), lambda: eg.TsNetSolver.pure_2hop(g, learning_rate=True))
        self.assert_no_panic_raises((ValueError,), lambda: eg.TsNetSolver.from_graph_row_normalized(g, w2=-0.1))
        self.assert_no_panic_raises((ValueError,), lambda: s_probs.step(eg.Layout2d.zeros(3)))

    def test_sgd_solver_all_constructors_schedules_and_manifolds(self) -> None:
        g = _make_grid(4, 4)
        n = g.num_nodes
        init_coords = _seeded_coords(n, seed=17, scale=1.5)

        # 1. from_pairs & from_pairs_with_hinges + schedule families + bit-exact f64 & f32
        c4_base = _seeded_coords(4, seed=9, scale=0.5)
        for sched in ("exponential", "linear", "quadratic", "reciprocal", "constant"):
            for dtype_str in ("float64", "float32"):
                s_run = eg.SgdSolver.from_pairs(
                    4, [0, 1, 2], [1, 2, 3], [1.0, 1.0, 1.0], t_max=10, schedule=sched, seed=42, dtype=dtype_str
                )
                s_step = eg.SgdSolver.from_pairs(
                    4, [0, 1, 2], [1, 2, 3], [1.0, 1.0, 1.0], t_max=10, schedule=sched, seed=42, dtype=dtype_str
                )
                c4 = _to_float_array(c4_base, dtype_str)
                l_run = eg.Layout2d.from_coords(c4)
                l_step = eg.Layout2d.from_coords(c4)
                self.assertEqual(s_run.num_springs, 3)
                self.assertEqual(s_run.num_hinges, 0)
                self.assertGreater(s_run.current_eta, 0.0)
                self.assertEqual(s_run.run(l_run), "completed")
                self.assertEqual(s_run.current_eta, 0.0)
                s_step.run_steps(l_step, 4)
                while not s_step.is_finished:
                    s_step.step(l_step)
                np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))

        sh = eg.SgdSolver.from_pairs_with_hinges(
            3,
            [0],
            [1],
            [1.0],
            None,
            None,
            None,
            [0],
            [2],
            [2.0],
            t_max=8,
            seed=42,
        )
        self.assertEqual(sh.num_springs, 1)
        self.assertEqual(sh.num_hinges, 1)
        lh = eg.Layout2d.from_coords([[0.0, 0.0], [3.0, 0.0], [0.0, 0.2]])
        self.assertEqual(sh.run(lh), "completed")
        self.assertTrue(lh.is_all_finite())

        # 2. full, sparse_barrier_dog, sparse_barrier_from_landmarks, random_pair, omega, streaming_negatives
        lm = eg.compute_regularized_dog_varadhan_landmarks(g, degree=20, num_landmarks=8, seed=42)
        emb = eg.chebyshev_dog_jl_embedding(g, degree=20, jl_dim=8, seed=42)

        s_full = eg.SgdSolver.full(g, t_max=15, seed=42)
        s_dog = eg.SgdSolver.sparse_barrier_dog(
            g, degree=20, num_landmarks=8, barrier_sym="sym_max", t_max=15, seed=42
        )
        s_lm = eg.SgdSolver.sparse_barrier_from_landmarks(
            g, lm, pivots=[0, 5, 10, 15], barrier_sym="sym_avg", t_max=15, seed=42
        )
        s_rp = eg.SgdSolver.random_pair(g, pairs_per_node=4, t_max=15, seed=42)
        s_om = eg.SgdSolver.omega(g, emb, k=4, min_dist=1e-3, t_max=15, seed=42)
        s_st = eg.SgdSolver.streaming_negatives(g, negatives_per_node=3, t_max=15, seed=42)

        self.assertIsNone(s_full.stats())
        dog_stats = s_dog.stats()
        lm_stats = s_lm.stats()
        assert dog_stats is not None and lm_stats is not None
        self.assertGreater(dog_stats["n_pairs_1hop"], 0)
        self.assertGreater(dog_stats["n_pairs_hinge"], 0)
        self.assertGreater(lm_stats["gamma_eff"], 0.0)
        self.assertIsNotNone(s_dog.dog_stats())

        for solver in (s_full, s_dog, s_lm, s_rp, s_om, s_st):
            lay = eg.Layout2d.from_coords(init_coords)
            self.assertEqual(solver.run(lay), "completed")
            self.assertTrue(lay.is_all_finite())

        # 3. Non-Euclidean manifold stepping across all 3 manifolds & both dtypes
        for kind in ("hyperbolic", "spherical", "torus"):
            for dtype_str, np_dtype in (("float64", np.float64), ("float32", np.float32)):
                ms_run = eg.SgdSolver.full(g, t_max=10, seed=42, dtype=dtype_str)
                ms_step = eg.SgdSolver.full(g, t_max=10, seed=42, dtype=dtype_str)
                ml_run = eg.ManifoldLayout2d.from_coords(kind, (init_coords * 0.25).astype(np_dtype))
                ml_step = eg.ManifoldLayout2d.from_coords(kind, (init_coords * 0.25).astype(np_dtype))

                self.assertEqual(ms_run.run_manifold(ml_run), "completed")
                self.assertEqual(ms_step.run_steps_manifold(ml_step, 4), "running")
                while not ms_step.is_finished:
                    ms_step.step_manifold(ml_step)
                self.assertEqual(ms_step.status, "completed")
                np.testing.assert_array_equal(ml_run.coords(), ml_step.coords())

                c_out = ml_run.coords()
                self.assertTrue(np.all(np.isfinite(c_out)))
                if kind == "hyperbolic":
                    self.assertTrue(np.all(c_out[:, 0] ** 2 + c_out[:, 1] ** 2 < 1.0))
                elif kind == "spherical":
                    self.assertTrue(np.all((c_out[:, 0] >= 0.0) & (c_out[:, 0] < 2.0 * math.pi)))
                    self.assertTrue(np.all((c_out[:, 1] >= 0.0) & (c_out[:, 1] <= math.pi)))
                else:
                    self.assertTrue(np.all((c_out >= 0.0) & (c_out < 1.0)))

        # Validation errors
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SgdSolver.full(g, schedule="cosine")
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SgdSolver.sparse_barrier_dog(g, barrier_sym="sym_min")
        )
        self.assert_no_panic_raises(
            (TypeError,), lambda: eg.SgdSolver.full(g, t_max=True)
        )
        self.assert_no_panic_raises(
            (IndexError,), lambda: eg.SgdSolver.from_pairs(2, [0], [5], [1.0])
        )

    def test_gumap_solver_all_constructors_and_step_vs_run(self) -> None:
        barbell = _make_barbell(6, bridge_len=1)
        n = barbell.num_nodes
        init_coords = _seeded_coords(n, seed=19, scale=1.0)

        for dtype_str, np_dtype in (("float64", np.float64), ("float32", np.float32)):
            s_run = eg.GuMapSolver.from_graph_dog(
                barbell, max_2hop_per_node=32, degree=20, num_landmarks=8, n_epochs=25, seed=42, dtype=dtype_str
            )
            s_step = eg.GuMapSolver.from_graph_dog(
                barbell, max_2hop_per_node=32, degree=20, num_landmarks=8, n_epochs=25, seed=42, dtype=dtype_str
            )
            self.assertGreater(s_run.num_edges, 0)
            a, b = s_run.ab
            self.assertGreater(a, 0.0)
            self.assertGreater(b, 0.0)

            l_run = eg.Layout2d.from_coords(init_coords.astype(np_dtype))
            l_step = eg.Layout2d.from_coords(init_coords.astype(np_dtype))
            self.assertEqual(s_run.run(l_run), "completed")
            self.assertEqual(s_step.run_steps(l_step, 7), "running")
            while not s_step.is_finished:
                s_step.step(l_step)
            self.assertEqual(s_step.status, "completed")
            np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))

        # Barbell cluster separation from all-zeros initial layout
        s_sep = eg.GuMapSolver.from_graph_dog(barbell, n_epochs=120, seed=42)
        l_zero = eg.Layout2d.zeros(n)
        self.assertEqual(s_sep.run(l_zero), "completed")
        coords = l_zero.coords(copy=True)
        c0 = np.mean(coords[:6], axis=0)
        c1 = np.mean(coords[6:], axis=0)
        self.assertGreater(float(np.linalg.norm(c0 - c1)), 1.0)

        # from_knn and from_fuzzy_set
        knn, _, _ = eg.build_pure_2hop_knn(barbell)
        fuzzy = eg.compute_fuzzy_simplicial_set(knn)
        s_knn = eg.GuMapSolver.from_knn(knn, n_epochs=15, seed=42)
        s_fuz = eg.GuMapSolver.from_fuzzy_set(fuzzy, n_epochs=15, seed=42)
        for s in (s_knn, s_fuz):
            lay = eg.Layout2d.zeros(n)
            self.assertEqual(s.run(lay), "completed")
            self.assertTrue(lay.is_all_finite())

        # from_landmarks parity with from_graph_dog
        lm = eg.compute_regularized_dog_varadhan_landmarks(barbell, degree=20, num_landmarks=8, seed=42)
        for dtype_str, np_dtype in (("float64", np.float64), ("float32", np.float32)):
            s_dog = eg.GuMapSolver.from_graph_dog(
                barbell, max_2hop_per_node=32, degree=20, num_landmarks=8, n_epochs=15, seed=42, dtype=dtype_str
            )
            s_lm_run = eg.GuMapSolver.from_landmarks(
                barbell, lm, max_2hop_per_node=32, n_epochs=15, seed=42, dtype=dtype_str
            )
            s_lm_step = eg.GuMapSolver.from_landmarks(
                barbell, lm, max_2hop_per_node=32, n_epochs=15, seed=42, dtype=dtype_str
            )
            l_dog = eg.Layout2d.from_coords(init_coords.astype(np_dtype))
            l_lm_run = eg.Layout2d.from_coords(init_coords.astype(np_dtype))
            l_lm_step = eg.Layout2d.from_coords(init_coords.astype(np_dtype))
            s_dog.run(l_dog)
            s_lm_run.run(l_lm_run)
            s_lm_step.run_steps(l_lm_step, 5)
            while not s_lm_step.is_finished:
                s_lm_step.step(l_lm_step)
            np.testing.assert_array_equal(l_lm_run.coords(copy=True), l_lm_step.coords(copy=True))
            np.testing.assert_array_equal(l_lm_run.coords(copy=True), l_dog.coords(copy=True))

        lm_other = eg.compute_regularized_dog_varadhan_landmarks(_make_grid(2, 2), num_landmarks=2, seed=42)
        self.assert_no_panic_raises((ValueError,), lambda: eg.GuMapSolver.from_landmarks(barbell, lm_other))
        self.assert_no_panic_raises((ValueError,), lambda: eg.GuMapSolver.from_knn(knn, min_dist=-0.1))
        self.assert_no_panic_raises((TypeError,), lambda: eg.GuMapSolver.from_knn(knn, n_epochs=True))

    def test_spatial_intruder_refiner(self) -> None:
        g = _make_grid(3, 3)
        clean = np.array([[float(c), float(r)] for r in range(3) for c in range(3)], dtype=np.float64)

        # Clean layout skips cleanly and converges early
        ref_clean = eg.SpatialIntruderRefiner(g, rounds=4, k_cap=32, seed=42)
        l_clean = eg.Layout2d.from_coords(clean)
        self.assertEqual(ref_clean.run(l_clean), "converged")
        st_clean = ref_clean.stats()
        self.assertEqual(st_clean["n_fp_total"], 0)
        self.assertTrue(st_clean["round_stats"][0]["skipped"])
        np.testing.assert_array_equal(l_clean.coords(copy=True), clean)

        # Folded intruder (far corner node 8 placed next to node 0) is pushed apart
        folded = clean.copy()
        folded[8] = np.array([0.08, 0.08], dtype=np.float64)
        dist_before = float(np.hypot(folded[0, 0] - folded[8, 0], folded[0, 1] - folded[8, 1]))

        for dtype_str in ("float64", "float32"):
            r_run = eg.SpatialIntruderRefiner(g, rounds=4, k_cap=32, seed=42, dtype=dtype_str)
            r_step = eg.SpatialIntruderRefiner(g, rounds=4, k_cap=32, seed=42, dtype=dtype_str)
            folded_typed = _to_float_array(folded, dtype_str)
            lf_run = eg.Layout2d.from_coords(folded_typed)
            lf_step = eg.Layout2d.from_coords(folded_typed)
            st = r_run.run(lf_run)
            self.assertIn(st, ("converged", "completed"))
            r_step.run_steps(lf_step, 1)
            while not r_step.is_finished:
                r_step.step(lf_step)
            np.testing.assert_array_equal(lf_run.coords(copy=True), lf_step.coords(copy=True))
            after = lf_run.coords(copy=True)
            dist_after = float(np.hypot(after[0, 0] - after[8, 0], after[0, 1] - after[8, 1]))
            self.assertGreater(dist_after, dist_before * 2.0)
            self.assertGreater(r_run.stats()["n_fp_total"], 0)

        # Folded 2x8 strip layout improves 2-hop neighborhood preservation
        g_strip = _make_grid(2, 8)
        strip_folded = np.array(
            [
                [float(c), float(r)] if c < 4 else [float(7 - c) + 0.12, float(r) + 0.12]
                for r in range(2)
                for c in range(8)
            ],
            dtype=np.float64,
        )
        strip_np2_before = eg.neighborhood_preservation_2hop_v6(g_strip, eg.Layout2d.from_coords(strip_folded))
        for dtype_str in ("float64", "float32"):
            rs_run = eg.SpatialIntruderRefiner(g_strip, rounds=4, k_cap=32, seed=42, dtype=dtype_str)
            rs_step = eg.SpatialIntruderRefiner(g_strip, rounds=4, k_cap=32, seed=42, dtype=dtype_str)
            strip_typed = _to_float_array(strip_folded, dtype_str)
            ls_run = eg.Layout2d.from_coords(strip_typed)
            ls_step = eg.Layout2d.from_coords(strip_typed)
            st_strip = rs_run.run(ls_run)
            self.assertIn(st_strip, ("converged", "completed"))
            rs_step.run_steps(ls_step, 1)
            while not rs_step.is_finished:
                rs_step.step(ls_step)
            np.testing.assert_array_equal(ls_run.coords(copy=True), ls_step.coords(copy=True))
            self.assertGreater(rs_run.stats()["n_fp_total"], 0)
            np2_after = eg.neighborhood_preservation_2hop_v6(g_strip, ls_run)
            self.assertGreater(np2_after, strip_np2_before + 0.1)

        # Dynamic k_cap & w_anchor2
        ref_dyn = eg.SpatialIntruderRefiner(
            g_strip,
            rounds=4,
            k_cap=8,
            seed=42,
            w_anchor2=0.5,
            k_cap_decay=0.5,
            k_cap_min=2,
        )
        self.assertEqual(ref_dyn.current_k_cap, 8)
        l_dyn = eg.Layout2d.from_coords(strip_folded)
        ref_dyn.step(l_dyn)
        self.assertEqual(ref_dyn.current_k_cap, 4)
        ref_dyn.set_k_cap(6)
        self.assertEqual(ref_dyn.current_k_cap, 6)
        ref_dyn.run(l_dyn)
        self.assertTrue(l_dyn.is_all_finite())
        self.assert_no_panic_raises((ValueError,), lambda: ref_dyn.set_k_cap(0))

        self.assert_no_panic_raises((ValueError,), lambda: eg.SpatialIntruderRefiner(g, k_cap=0))
        self.assert_no_panic_raises((TypeError,), lambda: eg.SpatialIntruderRefiner(g, hinge_fp=1))  # type: ignore[arg-type]

    def test_stress_majorization_and_kamada_kawai_solvers(self) -> None:
        g = _make_grid(3, 3)
        init_coords = _seeded_coords(9, seed=21, scale=1.5)

        for dtype_str, np_dtype in (("float64", np.float64), ("float32", np.float32)):
            sm_run = eg.StressMajorizationSolver.from_graph(g, max_iterations=30, epsilon=1e-6, dtype=dtype_str)
            sm_step = eg.StressMajorizationSolver.from_graph(g, max_iterations=30, epsilon=1e-6, dtype=dtype_str)
            l_run = eg.Layout2d.from_coords(init_coords.astype(np_dtype))
            l_step = eg.Layout2d.from_coords(init_coords.astype(np_dtype))

            self.assertIsNone(sm_run.last_stress)
            s_before = sm_run.compute_stress(l_run)
            st_sm = sm_run.run(l_run)
            self.assertIn(st_sm, ("converged", "completed"))
            s_after = sm_run.compute_stress(l_run)
            self.assertIsNotNone(sm_run.last_stress)
            self.assertLess(s_after, s_before * 0.25)

            sm_step.run_steps(l_step, 4)
            while not sm_step.is_finished:
                sm_step.step(l_step)
            np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))

            kk_run = eg.KamadaKawaiSolver.from_graph(g, dtype=dtype_str)
            kk_step = eg.KamadaKawaiSolver.from_graph(g, dtype=dtype_str)
            lk_run = eg.Layout2d.from_coords(init_coords.astype(np_dtype))
            lk_step = eg.Layout2d.from_coords(init_coords.astype(np_dtype))
            self.assertIn(kk_run.run(lk_run), ("converged", "completed"))
            kk_step.run_steps(lk_step, 5)
            while not kk_step.is_finished:
                kk_step.step(lk_step)
            np.testing.assert_array_equal(lk_run.coords(copy=True), lk_step.coords(copy=True))

        # from_distance_matrix with +inf unreachable pairs + collinear det == 0 guard
        phi_inv = (math.sqrt(5.0) - 1.0) * 0.5
        dmat_collinear = np.array(
            [
                [0.0, 1.0, phi_inv],
                [1.0, 0.0, 1.0],
                [phi_inv, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        kk_sing = eg.KamadaKawaiSolver.from_distance_matrix(dmat_collinear, eps=1e-3, max_outer_iters=60)
        l_col = eg.Layout2d.from_coords([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
        self.assertIn(kk_sing.run(l_col), ("converged", "completed"))
        self.assertTrue(l_col.is_all_finite())

        dmat_disc = np.array(
            [
                [0.0, 1.0, float("inf"), float("inf")],
                [1.0, 0.0, float("inf"), float("inf")],
                [float("inf"), float("inf"), 0.0, 1.0],
                [float("inf"), float("inf"), 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        sm_disc = eg.StressMajorizationSolver.from_distance_matrix(dmat_disc)
        l_disc = eg.Layout2d.from_coords([[0.0, 0.0], [0.2, 0.0], [5.0, 1.0], [8.0, 1.0]])
        self.assertIn(sm_disc.run(l_disc), ("converged", "completed"))
        self.assertTrue(l_disc.is_all_finite())

        # Validation errors on distance matrices
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.StressMajorizationSolver.from_distance_matrix(np.zeros((2, 3))),
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.KamadaKawaiSolver.from_distance_matrix([[0.0, float("nan")], [1.0, 0.0]]),
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.StressMajorizationSolver.from_distance_matrix([[0.0, -1.0], [-1.0, 0.0]]),
        )

    def test_gil_release_concurrent_m3_execution(self) -> None:
        g = _make_grid(5, 5)
        init = _seeded_coords(25, seed=31, scale=1.0)
        ref_solver = eg.SgdSolver.sparse_barrier_dog(g, degree=15, num_landmarks=6, t_max=12, seed=42)
        ref_layout = eg.Layout2d.from_coords(init)
        ref_solver.run(ref_layout)
        ref_coords = ref_layout.coords(copy=True)

        barrier = threading.Barrier(4)

        def worker(_idx: int) -> npt.NDArray[np.float64]:
            barrier.wait(timeout=10.0)
            s = eg.SgdSolver.sparse_barrier_dog(g, degree=15, num_landmarks=6, t_max=12, seed=42)
            lay = eg.Layout2d.from_coords(init)
            s.run(lay)
            _ = eg.louvain_communities(g, seed=42)
            _ = eg.spectral_clustering(g, k=2, seed=42)
            return lay.coords(copy=True)

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(worker, range(4)))
        for arr in results:
            np.testing.assert_array_equal(arr, ref_coords)

    def test_m3_exports_docstrings_and_signatures(self) -> None:
        m3_names = (
            "ManifoldLayout2d",
            "TsNetSolver",
            "SgdSolver",
            "GuMapSolver",
            "SpatialIntruderRefiner",
            "StressMajorizationSolver",
            "KamadaKawaiSolver",
            "kmeans_plus_plus",
            "spectral_clustering",
            "louvain_communities",
            "label_propagation",
            "coarsen_graph",
            "project_clustered_layout",
        )
        for name in m3_names:
            obj = getattr(eg, name)
            self.assertIs(obj, getattr(eg._core, name))
            self.assertIn(name, eg.__all__)
            self.assertTrue((obj.__doc__ or "").strip(), f"missing docstring on {name}")

        sig_km = inspect.signature(eg.kmeans_plus_plus)
        self.assertEqual(sig_km.parameters["k"].default, 2)
        self.assertEqual(sig_km.parameters["seed"].default, 42)
        sig_ref = inspect.signature(eg.SpatialIntruderRefiner)
        self.assertEqual(sig_ref.parameters["rounds"].default, 4)
        self.assertEqual(sig_ref.parameters["hinge_fp"].default, True)


if __name__ == "__main__":
    unittest.main()
