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


class TestGumapRefinerAndStressSolvers(unittest.TestCase):
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
        fn: Callable[[], Any],
    ) -> None:
        with self.assertRaises(expected) as ctx:
            fn()
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

    def test_gumap_solver_all_constructors_and_step_vs_run(self) -> None:
        for ctor in (
            eg.GuMapSolver.from_fuzzy_set,
            eg.GuMapSolver.from_knn,
            eg.GuMapSolver.from_graph_dog,
            eg.GuMapSolver.from_landmarks,
        ):
            sig = inspect.signature(ctor)
            self.assertIs(sig.parameters["symmetric_negative"].default, True)

        barbell = _make_barbell(6, bridge_len=1)
        n = barbell.num_nodes
        init_coords = _seeded_coords(n, seed=19, scale=1.0)

        for dtype_str, np_dtype in (("float64", np.float64), ("float32", np.float32)):
            s_run = eg.GuMapSolver.from_graph_dog(
                barbell, max_2hop_per_node=32, degree=20, num_landmarks=8, iterations=25, seed=42, dtype=dtype_str
            )
            s_step = eg.GuMapSolver.from_graph_dog(
                barbell, max_2hop_per_node=32, degree=20, num_landmarks=8, iterations=25, seed=42, dtype=dtype_str
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
        s_sep = eg.GuMapSolver.from_graph_dog(barbell, iterations=120, seed=42)
        l_zero = eg.Layout2d.zeros(n)
        self.assertEqual(s_sep.run(l_zero), "completed")
        coords = l_zero.coords(copy=True)
        c0 = np.mean(coords[:6], axis=0)
        c1 = np.mean(coords[6:], axis=0)
        self.assertGreater(float(np.linalg.norm(c0 - c1)), 1.0)

        # from_knn and from_fuzzy_set
        knn, _, _ = eg.build_pure_2hop_knn(barbell)
        fuzzy = eg.compute_fuzzy_simplicial_set(knn)
        s_knn = eg.GuMapSolver.from_knn(knn, iterations=15, seed=42)
        s_fuz = eg.GuMapSolver.from_fuzzy_set(fuzzy, iterations=15, seed=42)
        for s in (s_knn, s_fuz):
            lay = eg.Layout2d.zeros(n)
            self.assertEqual(s.run(lay), "completed")
            self.assertTrue(lay.is_all_finite())

        # from_landmarks parity with from_graph_dog
        lm = eg.compute_regularized_dog_varadhan_landmarks(barbell, degree=20, num_landmarks=8, seed=42)
        for dtype_str, np_dtype in (("float64", np.float64), ("float32", np.float32)):
            s_dog = eg.GuMapSolver.from_graph_dog(
                barbell, max_2hop_per_node=32, degree=20, num_landmarks=8, iterations=15, seed=42, dtype=dtype_str
            )
            s_lm_run = eg.GuMapSolver.from_landmarks(
                barbell, lm, max_2hop_per_node=32, iterations=15, seed=42, dtype=dtype_str
            )
            s_lm_step = eg.GuMapSolver.from_landmarks(
                barbell, lm, max_2hop_per_node=32, iterations=15, seed=42, dtype=dtype_str
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
        self.assert_no_panic_raises((TypeError,), lambda: eg.GuMapSolver.from_knn(knn, iterations=True))
        self.assert_no_panic_raises(
            (TypeError,), lambda: eg.GuMapSolver.from_knn(knn, symmetric_negative=1)  # type: ignore[arg-type]
        )

        # symmetric_negative=False diverges deterministically from True across all 4 constructors
        # while preserving step() == run() bit-exactness for both float64 and float32.
        def _build_gumap(label: str, sym: bool, dt: str) -> eg.GuMapSolver:
            if label == "from_fuzzy_set":
                return eg.GuMapSolver.from_fuzzy_set(
                    fuzzy, iterations=10, symmetric_negative=sym, seed=42, dtype=dt
                )
            if label == "from_knn":
                return eg.GuMapSolver.from_knn(
                    knn, iterations=10, symmetric_negative=sym, seed=42, dtype=dt
                )
            if label == "from_graph_dog":
                return eg.GuMapSolver.from_graph_dog(
                    barbell,
                    max_2hop_per_node=32,
                    degree=20,
                    num_landmarks=8,
                    iterations=10,
                    symmetric_negative=sym,
                    seed=42,
                    dtype=dt,
                )
            return eg.GuMapSolver.from_landmarks(
                barbell, lm, max_2hop_per_node=32, iterations=10, symmetric_negative=sym, seed=42, dtype=dt
            )

        for dtype_str, np_dtype in (("float64", np.float64), ("float32", np.float32)):
            for label in ("from_fuzzy_set", "from_knn", "from_graph_dog", "from_landmarks"):
                s_true = _build_gumap(label, True, dtype_str)
                s_false_run = _build_gumap(label, False, dtype_str)
                s_false_step = _build_gumap(label, False, dtype_str)
                l_true = eg.Layout2d.from_coords(init_coords.astype(np_dtype))
                l_false_run = eg.Layout2d.from_coords(init_coords.astype(np_dtype))
                l_false_step = eg.Layout2d.from_coords(init_coords.astype(np_dtype))
                s_true.run(l_true)
                s_false_run.run(l_false_run)
                s_false_step.run_steps(l_false_step, 4)
                while not s_false_step.is_finished:
                    s_false_step.step(l_false_step)
                self.assertTrue(l_false_run.is_all_finite(), label)
                np.testing.assert_array_equal(l_false_run.coords(copy=True), l_false_step.coords(copy=True))
                self.assertFalse(
                    np.allclose(l_true.coords(copy=True), l_false_run.coords(copy=True)),
                    f"{label}/{dtype_str}: symmetric_negative=False matched True",
                )

    def test_spatial_intruder_refiner(self) -> None:
        g = _make_grid(3, 3)
        clean = np.array([[float(c), float(r)] for r in range(3) for c in range(3)], dtype=np.float64)

        # Clean layout skips cleanly and converges early
        ref_clean = eg.SpatialIntruderRefiner(g, iterations=4, k_cap=32, seed=42)
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
            r_run = eg.SpatialIntruderRefiner(g, iterations=4, k_cap=32, seed=42, dtype=dtype_str)
            r_step = eg.SpatialIntruderRefiner(g, iterations=4, k_cap=32, seed=42, dtype=dtype_str)
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
        strip_np2_before = eg.neighborhood_preservation_2hop(g_strip, eg.Layout2d.from_coords(strip_folded))
        for dtype_str in ("float64", "float32"):
            rs_run = eg.SpatialIntruderRefiner(g_strip, iterations=4, k_cap=32, seed=42, dtype=dtype_str)
            rs_step = eg.SpatialIntruderRefiner(g_strip, iterations=4, k_cap=32, seed=42, dtype=dtype_str)
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
            np2_after = eg.neighborhood_preservation_2hop(g_strip, ls_run)
            self.assertGreater(np2_after, strip_np2_before + 0.1)

        # Dynamic k_cap & w_anchor2
        ref_dyn = eg.SpatialIntruderRefiner(
            g_strip,
            iterations=4,
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
        self.assert_no_panic_raises((ValueError,), lambda: eg.SpatialIntruderRefiner(g, min_sep_ratio=-0.1))
        self.assert_no_panic_raises((ValueError,), lambda: eg.SpatialIntruderRefiner(g, w_sep=-0.1))
        self.assert_no_panic_raises((TypeError,), lambda: eg.SpatialIntruderRefiner(g, hinge_fp=1))  # type: ignore[arg-type]

        # Opt-in separation hinge eliminates closed-twin collapse while preserving clean grids
        twin_edges = list(g.to_edges() if hasattr(g, "to_edges") else [])
        if not twin_edges:
            twin_edges = [
                (r * 3 + c, r * 3 + c + 1) for r in range(3) for c in range(2)
            ] + [
                (r * 3 + c, (r + 1) * 3 + c) for r in range(2) for c in range(3)
            ]
        twin_edges.extend([(0, 9), (0, 10), (9, 10)])
        g_twin = eg.CsrGraph.from_edges(11, twin_edges, directed=False, sanitize=True)
        twin_init = np.vstack([clean, np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)])
        for dtype_str in ("float64", "float32"):
            twin_typed = _to_float_array(twin_init, dtype_str)
            lt_run = eg.Layout2d.from_coords(twin_typed)
            lt_step = eg.Layout2d.from_coords(twin_typed)
            rt_run = eg.SpatialIntruderRefiner(
                g_twin, min_sep_ratio=0.15, w_sep=1.0, seed=42, dtype=dtype_str
            )
            rt_step = eg.SpatialIntruderRefiner(
                g_twin, min_sep_ratio=0.15, w_sep=1.0, seed=42, dtype=dtype_str
            )
            rt_run.run(lt_run)
            rt_step.step(lt_step)
            rt_step.run(lt_step)
            np.testing.assert_array_equal(lt_run.coords(copy=True), lt_step.coords(copy=True))
            st_twin = rt_run.stats()
            self.assertGreater(st_twin["n_sep_total"], 0)
            self.assertGreater(st_twin["round_stats"][0]["n_sep"], 0)
            coords_out = lt_run.coords(copy=True)
            d_9_10 = float(np.hypot(coords_out[9, 0] - coords_out[10, 0], coords_out[9, 1] - coords_out[10, 1]))
            self.assertGreater(d_9_10, 0.05)

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
        kk_sing = eg.KamadaKawaiSolver.from_distance_matrix(dmat_collinear, eps=1e-3, max_iterations=60)
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

    def test_gil_release_concurrent_solver_execution(self) -> None:
        g = _make_grid(5, 5)
        init = _seeded_coords(25, seed=31, scale=1.0)
        ref_solver = eg.SgdSolver.sparse_barrier_dog(g, degree=15, num_landmarks=6, iterations=12, seed=42)
        ref_layout = eg.Layout2d.from_coords(init)
        ref_solver.run(ref_layout)
        ref_coords = ref_layout.coords(copy=True)

        barrier = threading.Barrier(4)

        def worker(_idx: int) -> npt.NDArray[np.float64]:
            barrier.wait(timeout=10.0)
            s = eg.SgdSolver.sparse_barrier_dog(g, degree=15, num_landmarks=6, iterations=12, seed=42)
            lay = eg.Layout2d.from_coords(init)
            s.run(lay)
            _ = eg.louvain_communities(g, seed=42)
            _ = eg.spectral_clustering(g, k=2, seed=42)
            return lay.coords(copy=True)

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(worker, range(4)))
        for arr in results:
            np.testing.assert_array_equal(arr, ref_coords)


if __name__ == "__main__":
    unittest.main()
