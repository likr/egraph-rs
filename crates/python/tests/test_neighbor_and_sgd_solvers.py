from __future__ import annotations

import inspect
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


def _to_float_array(
    coords: npt.NDArray[np.float64], dtype_str: str
) -> npt.NDArray[np.floating[Any]]:
    if dtype_str == "float32":
        return coords.astype(np.float32)
    return coords.astype(np.float64)


class TestNeighborAndSgdSolvers(unittest.TestCase):
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
        fn: Callable[[], Any],
    ) -> None:
        with self.assertRaises(expected) as ctx:
            fn()
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

    def test_t_fdp_backends_step_vs_run_and_from_knn(self) -> None:
        graph = _make_barbell(5, bridge_len=2)
        n = graph.num_nodes
        base = _seeded_coords(n, seed=11, scale=0.5)

        for dtype_str in ("float64", "float32"):
            for backend in ("ibfft", "barnes_hut", "exact"):
                l_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
                l_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))

                s_run = eg.TFdpSolver(
                    graph,
                    backend=backend,
                    iterations=12,
                    rho_start=1.5,
                    rho_end=0.2,
                    rho_power=1.0,
                    seed=7,
                    dtype=dtype_str,
                )
                s_step = eg.TFdpSolver(
                    graph,
                    backend=backend,
                    iterations=12,
                    rho_start=1.5,
                    rho_end=0.2,
                    rho_power=1.0,
                    seed=7,
                    dtype=dtype_str,
                )
                self.assertEqual(s_run.status, "initialized")
                self.assertEqual(s_run.num_nodes, n)
                self.assertEqual(s_run.total_iterations, 12)
                self.assertEqual(s_run.dtype, dtype_str)
                self.assertAlmostEqual(s_run.current_rho, 1.5, places=6)
                self.assertIn("TFdpSolver", repr(s_run))

                status_run = s_run.run(l_run)
                self.assertEqual(status_run, "completed")
                self.assertTrue(s_run.is_finished)
                self.assertAlmostEqual(s_run.current_rho, 0.2, places=6)

                rhos = [s_step.current_rho]
                for _ in range(5):
                    s_step.step(l_step)
                    rhos.append(s_step.current_rho)
                self.assertEqual(s_step.status, "running")
                self.assertEqual(s_step.iteration, 5)
                s_step.run_steps(l_step, 7)
                self.assertTrue(s_step.is_finished)
                self.assertTrue(all(rhos[i] >= rhos[i + 1] for i in range(len(rhos) - 1)))

                np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))
                self.assertTrue(l_run.is_all_finite())

        # Both from_knn(KnnGraph) and from_knn(indptr, indices, weights=None)
        knn, _, _ = eg.build_pure_2hop_knn(graph)
        s_knn_obj = eg.TFdpSolver.from_knn(knn, backend="exact", iterations=8, seed=42)
        s_knn_raw = eg.TFdpSolver.from_knn(
            knn.indptr(),
            knn.indices(),
            knn.distances(),
            backend="exact",
            iterations=8,
            seed=42,
        )
        l1 = eg.Layout2d.from_coords(base)
        l2 = eg.Layout2d.from_coords(base)
        s_knn_obj.run(l1)
        s_knn_raw.run(l2)
        np.testing.assert_array_equal(l1.coords(copy=True), l2.coords(copy=True))

        # from_knn with weights=None
        s_knn_now = eg.TFdpSolver.from_knn(
            graph.indptr(), graph.indices(), None, backend="barnes_hut", iterations=6
        )
        l3 = eg.Layout2d.from_coords(base)
        s_knn_now.run(l3)
        self.assertTrue(l3.is_all_finite())

    def test_ib_umap_constructors_and_step_vs_run(self) -> None:
        graph = _make_grid(4, 4)
        n = graph.num_nodes
        base = _seeded_coords(n, seed=19, scale=0.8)

        for dtype_str in ("float64", "float32"):
            for backend in ("ibfft", "barnes_hut", "exact"):
                l_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
                l_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))

                s_run = eg.IbUmapSolver(
                    graph,
                    max_2hop_per_node=20,
                    backend=backend,
                    iterations=10,
                    seed=123,
                    dtype=dtype_str,
                )
                s_step = eg.IbUmapSolver(
                    graph,
                    max_2hop_per_node=20,
                    backend=backend,
                    iterations=10,
                    seed=123,
                    dtype=dtype_str,
                )
                self.assertEqual(s_run.num_nodes, n)
                self.assertEqual(s_run.total_iterations, 10)
                self.assertEqual(s_run.dtype, dtype_str)
                self.assertAlmostEqual(s_run.current_rho, 1.0, places=6)
                self.assertIn("IbUmapSolver", repr(s_run))

                s_run.run(l_run)
                s_step.step(l_step)
                s_step.run_steps(l_step, 4)
                s_step.run(l_step)
                self.assertTrue(s_run.is_finished)
                self.assertTrue(s_step.is_finished)
                np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))

        knn, _, _ = eg.build_pure_2hop_knn(graph)
        probs = eg.compute_fuzzy_simplicial_set(knn)
        s_knn = eg.IbUmapSolver.from_knn(knn, backend="exact", iterations=6, seed=42)
        s_fset = eg.IbUmapSolver.from_fuzzy_set(probs, backend="exact", iterations=6, seed=42)
        l_a = eg.Layout2d.from_coords(base)
        l_b = eg.Layout2d.from_coords(base)
        s_knn.run(l_a)
        s_fset.run(l_b)
        np.testing.assert_allclose(l_a.coords(copy=True), l_b.coords(copy=True), rtol=1e-12, atol=1e-12)

        # from_landmarks (default max_2hop_per_node=32) & parity with IbUmapSolver.__new__
        self.assertEqual(inspect.signature(eg.IbUmapSolver.from_landmarks).parameters["max_2hop_per_node"].default, 32)
        lm = eg.compute_regularized_dog_varadhan_landmarks(graph, num_landmarks=8, seed=42)
        for dtype_str, np_dtype in (("float64", np.float64), ("float32", np.float32)):
            s_direct = eg.IbUmapSolver(
                graph, max_2hop_per_node=32, num_landmarks=8, backend="exact", iterations=8, seed=42, dtype=dtype_str
            )
            s_lm_run = eg.IbUmapSolver.from_landmarks(
                graph, lm, max_2hop_per_node=32, backend="exact", iterations=8, seed=42, dtype=dtype_str
            )
            s_lm_step = eg.IbUmapSolver.from_landmarks(
                graph, lm, max_2hop_per_node=32, backend="exact", iterations=8, seed=42, dtype=dtype_str
            )
            l_dir = eg.Layout2d.from_coords(base.astype(np_dtype))
            l_run = eg.Layout2d.from_coords(base.astype(np_dtype))
            l_step = eg.Layout2d.from_coords(base.astype(np_dtype))
            s_direct.run(l_dir)
            s_lm_run.run(l_run)
            s_lm_step.step(l_step)
            s_lm_step.run(l_step)
            np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))
            np.testing.assert_array_equal(l_run.coords(copy=True), l_dir.coords(copy=True))

        lm_bad = eg.compute_regularized_dog_varadhan_landmarks(_make_grid(2, 2), num_landmarks=2, seed=42)
        self.assert_no_panic_raises((ValueError,), lambda: eg.IbUmapSolver.from_landmarks(graph, lm_bad))

    def test_sgd_solvers_and_squared_schedule(self) -> None:
        graph = _make_barbell(5, bridge_len=2)
        n = graph.num_nodes
        base = _seeded_coords(n, seed=31, scale=1.0)

        sig_snap = inspect.signature(eg.SgdSolver.snap_t_fdp)
        self.assertEqual(sig_snap.parameters["degree_beta"].default, 0.0)
        self.assertIs(sig_snap.parameters["bounded_attraction"].default, True)

        for dtype_str in ("float64", "float32"):
            # 1. snap_t_fdp
            l_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            l_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            s1 = eg.SgdSolver.snap_t_fdp(graph, iterations=12, negatives_per_edge=3, seed=5, dtype=dtype_str)
            s2 = eg.SgdSolver.snap_t_fdp(graph, iterations=12, negatives_per_edge=3, seed=5, dtype=dtype_str)
            self.assertEqual(s1.total_iterations, 12)
            self.assertEqual(s1.dtype, dtype_str)
            self.assertIsNone(s1.stats())
            self.assertIsNone(s1.dog_stats())
            s1.run(l_run)
            s2.step(l_step)
            s2.run_steps(l_step, 5)
            s2.run(l_step)
            np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))

            # 1b. snap_t_fdp degree_beta + bounded_attraction=False diverges from defaults and preserves step == run
            l_mod_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            l_mod_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            sm1 = eg.SgdSolver.snap_t_fdp(
                graph,
                iterations=12,
                negatives_per_edge=3,
                degree_beta=0.5,
                bounded_attraction=False,
                seed=5,
                dtype=dtype_str,
            )
            sm2 = eg.SgdSolver.snap_t_fdp(
                graph,
                iterations=12,
                negatives_per_edge=3,
                degree_beta=0.5,
                bounded_attraction=False,
                seed=5,
                dtype=dtype_str,
            )
            sm1.run(l_mod_run)
            sm2.step(l_mod_step)
            sm2.run_steps(l_mod_step, 5)
            sm2.run(l_mod_step)
            np.testing.assert_array_equal(l_mod_run.coords(copy=True), l_mod_step.coords(copy=True))
            self.assertFalse(np.allclose(l_run.coords(copy=True), l_mod_run.coords(copy=True)))

            # 2. pacmap_graph
            l_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            l_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            p1 = eg.SgdSolver.pacmap_graph(
                graph,
                n_neighbors=6,
                phase_iterations=(4, 4, 6),
                seed=9,
                dtype=dtype_str,
            )
            p2 = eg.SgdSolver.pacmap_graph(
                graph,
                n_neighbors=6,
                phase_iterations=(4, 4, 6),
                seed=9,
                dtype=dtype_str,
            )
            self.assertEqual(p1.total_iterations, 14)
            self.assertEqual(p1.dtype, dtype_str)
            p1.run(l_run)
            p2.step(l_step)
            p2.run_steps(l_step, 6)
            p2.run(l_step)
            np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))

            # 2b. pacmap_graph keep_all_1hop=True retains all hub 1-hop edges beyond n_neighbors
            star = eg.CsrGraph.from_edges(9, [(0, i) for i in range(1, 9)], directed=False, sanitize=True)
            pm_keep = eg.SgdSolver.pacmap_graph(
                star, n_neighbors=2, phase_iterations=(2, 2, 2), keep_all_1hop=True, seed=42, dtype=dtype_str
            )
            pm_keep_step = eg.SgdSolver.pacmap_graph(
                star, n_neighbors=2, phase_iterations=(2, 2, 2), keep_all_1hop=True, seed=42, dtype=dtype_str
            )
            self.assertGreaterEqual(pm_keep.num_springs, 8)
            star_init = _to_float_array(_seeded_coords(9, seed=13, scale=1.0), dtype_str)
            ls_r = eg.Layout2d.from_coords(star_init)
            ls_s = eg.Layout2d.from_coords(star_init)
            pm_keep.run(ls_r)
            pm_keep_step.step(ls_s)
            pm_keep_step.run(ls_s)
            np.testing.assert_array_equal(ls_r.coords(copy=True), ls_s.coords(copy=True))

            # 2c. max_1hop_per_node, degree_weight_beta, and external landmarks (DogVaradhanLandmarks + 2D array)
            pm_cap = eg.SgdSolver.pacmap_graph(
                star,
                n_neighbors=2,
                keep_all_1hop=True,
                max_1hop_per_node=3,
                degree_weight_beta=0.5,
                phase_iterations=(2, 2, 2),
                seed=42,
                dtype=dtype_str,
            )
            self.assertLessEqual(pm_cap.num_springs, 24)
            dog_lm = eg.compute_regularized_dog_varadhan_landmarks(graph, num_landmarks=6, seed=7)
            pm_lm = eg.SgdSolver.pacmap_graph_with_landmarks(
                graph,
                dog_lm,
                n_neighbors=6,
                phase_iterations=(4, 4, 6),
                degree_weight_beta=0.25,
                seed=9,
                dtype=dtype_str,
            )
            pm_mat = eg.SgdSolver.pacmap_graph(
                graph,
                n_neighbors=6,
                phase_iterations=(4, 4, 6),
                degree_weight_beta=0.25,
                landmarks=dog_lm.table(),
                seed=9,
                dtype=dtype_str,
            )
            l_lm = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            l_mat = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            pm_lm.run(l_lm)
            pm_mat.run(l_mat)
            np.testing.assert_array_equal(l_lm.coords(copy=True), l_mat.coords(copy=True))

            # 3. sgd2
            l_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            l_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            q1 = eg.SgdSolver.sgd2(graph, iterations=10, batch_size=32, seed=13, dtype=dtype_str)
            q2 = eg.SgdSolver.sgd2(graph, iterations=10, batch_size=32, seed=13, dtype=dtype_str)
            self.assertEqual(q1.total_iterations, 10)
            self.assertEqual(q1.dtype, dtype_str)
            q1.run(l_run)
            q2.step(l_step)
            q2.run_steps(l_step, 4)
            q2.run(l_step)
            np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))

            # 4. "squared" schedule kind on SgdSolver.full
            l_sq1 = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            l_sq2 = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            sq1 = eg.SgdSolver.full(graph, iterations=10, schedule="squared", seed=17, dtype=dtype_str)
            sq2 = eg.SgdSolver.full(graph, iterations=10, schedule="squared", seed=17, dtype=dtype_str)
            eta0 = sq1.current_eta
            sq1.step(l_sq1)
            eta1 = sq1.current_eta
            self.assertGreater(eta0, eta1)
            sq1.run(l_sq1)
            sq2.run(l_sq2)
            np.testing.assert_array_equal(l_sq1.coords(copy=True), l_sq2.coords(copy=True))


if __name__ == "__main__":
    unittest.main()
