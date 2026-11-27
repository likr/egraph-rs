from __future__ import annotations

import inspect
import math
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


def _seeded_coords(n: int, seed: int = 42, scale: float = 1.0) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=(n, 2)).astype(np.float64)


def _to_float_array(coords: npt.NDArray[np.float64], dtype_str: str) -> npt.NDArray[np.floating[Any]]:
    if dtype_str == "float32":
        return coords.astype(np.float32)
    return coords.astype(np.float64)


class TestTsNetAndSgdSolvers(unittest.TestCase):
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
        fn: Callable[[], Any],
    ) -> None:
        with self.assertRaises(expected) as ctx:
            fn()
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

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

        # Verify default learning_rate=None and jacobi_precondition=True across all 5 TsNetSolver constructors
        for ctor in (
            eg.TsNetSolver.pure_2hop,
            eg.TsNetSolver.from_knn,
            eg.TsNetSolver.from_probabilities,
            eg.TsNetSolver.from_graph_partial_bfs,
            eg.TsNetSolver.from_graph_row_normalized,
        ):
            self.assertIsNone(inspect.signature(ctor).parameters["learning_rate"].default)
            self.assertIs(inspect.signature(ctor).parameters["jacobi_precondition"].default, True)

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
        self.assert_no_panic_raises((ValueError,), lambda: eg.TsNetSolver.from_graph_row_normalized(g, degree_beta=-0.1))
        rn_beta = eg.TsNetSolver.from_graph_row_normalized(
            g, w2=0.5, max_2hop_per_node=4, degree_beta=0.35, backend="fft", iterations_stage1=2, iterations_stage2=2, iterations_stage3=2, seed=42
        )
        l_rn_beta = eg.Layout2d.from_coords(init_coords)
        self.assertEqual(rn_beta.run(l_rn_beta), "completed")
        self.assertTrue(l_rn_beta.is_all_finite())
        self.assert_no_panic_raises((ValueError,), lambda: s_probs.step(eg.Layout2d.zeros(3)))

    def test_tsnet_solver_jacobi_precondition_star_hub(self) -> None:
        n = 20
        star = eg.CsrGraph.from_edges(n, [(0, v) for v in range(1, n)], directed=False, sanitize=True)
        init_coords = _seeded_coords(n, seed=19, scale=0.5)
        init_coords[0] = [1.0, 0.0]
        init_coords -= init_coords.mean(axis=0, keepdims=True)

        knn, _, _ = eg.build_pure_2hop_knn(star)
        perps = eg.compute_nodewise_2hop_perplexities(star)
        probs = eg.graph_row_normalized_probabilities(star)

        for dtype_str in ("float64", "float32"):
            common: dict[str, Any] = {
                "backend": "exact",
                "iterations_stage1": 4,
                "iterations_stage2": 2,
                "iterations_stage3": 2,
                "seed": 42,
                "dtype": dtype_str,
            }

            def _mk_pure(jp: bool, cfg: dict[str, Any] = common) -> eg.TsNetSolver:
                return eg.TsNetSolver.pure_2hop(star, jacobi_precondition=jp, **cfg)

            def _mk_knn(jp: bool, cfg: dict[str, Any] = common) -> eg.TsNetSolver:
                return eg.TsNetSolver.from_knn(knn, perps, jacobi_precondition=jp, **cfg)

            def _mk_probs(jp: bool, cfg: dict[str, Any] = common) -> eg.TsNetSolver:
                return eg.TsNetSolver.from_probabilities(probs, jacobi_precondition=jp, **cfg)

            def _mk_bfs(jp: bool, cfg: dict[str, Any] = common) -> eg.TsNetSolver:
                return eg.TsNetSolver.from_graph_partial_bfs(
                    star, 8, 6.0, jacobi_precondition=jp, **cfg
                )

            def _mk_rn(jp: bool, cfg: dict[str, Any] = common) -> eg.TsNetSolver:
                return eg.TsNetSolver.from_graph_row_normalized(
                    star, w2=0.0, jacobi_precondition=jp, **cfg
                )

            ctors: list[tuple[str, Callable[[bool], eg.TsNetSolver]]] = [
                ("pure_2hop", _mk_pure),
                ("from_knn", _mk_knn),
                ("from_probabilities", _mk_probs),
                ("from_graph_partial_bfs", _mk_bfs),
                ("from_graph_row_normalized", _mk_rn),
            ]
            for name, make_solver in ctors:
                s_on = make_solver(True)
                s_off = make_solver(False)
                l_on = eg.Layout2d.from_coords(_to_float_array(init_coords, dtype_str))
                l_off = eg.Layout2d.from_coords(_to_float_array(init_coords, dtype_str))
                self.assertEqual(s_on.run(l_on), "completed", msg=name)
                self.assertEqual(s_off.run(l_off), "completed", msg=name)
                self.assertTrue(l_on.is_all_finite(), msg=name)
                self.assertTrue(l_off.is_all_finite(), msg=name)
                coords_on = l_on.coords(copy=True)
                coords_off = l_off.coords(copy=True)
                self.assertFalse(
                    np.allclose(coords_on, coords_off, atol=1e-6, rtol=1e-6),
                    msg=f"{name} ({dtype_str}): expected jacobi_precondition=True vs False to differ on star hub",
                )
                r_hub_on = float(np.hypot(coords_on[0, 0], coords_on[0, 1]))
                r_hub_off = float(np.hypot(coords_off[0, 0], coords_off[0, 1]))
                self.assertLess(
                    r_hub_on,
                    r_hub_off,
                    msg=f"{name} ({dtype_str}): expected jacobi_precondition=True to reduce hub overshoot ({r_hub_on} >= {r_hub_off})",
                )

        # Non-bool values must raise TypeError cleanly without PanicException
        bad_values: tuple[Any, ...] = (1, 0, "true", None)
        for bad_val in bad_values:

            def _call_pure(val: Any = bad_val) -> eg.TsNetSolver:
                return eg.TsNetSolver.pure_2hop(star, jacobi_precondition=val)

            def _call_rn(val: Any = bad_val) -> eg.TsNetSolver:
                return eg.TsNetSolver.from_graph_row_normalized(star, jacobi_precondition=val)

            self.assert_no_panic_raises((TypeError,), _call_pure)
            self.assert_no_panic_raises((TypeError,), _call_rn)

    def test_sgd_solver_all_constructors_schedules_and_manifolds(self) -> None:
        g = _make_grid(4, 4)
        n = g.num_nodes
        init_coords = _seeded_coords(n, seed=17, scale=1.5)

        # 1. from_pairs & from_pairs_with_hinges + schedule families + bit-exact f64 & f32
        c4_base = _seeded_coords(4, seed=9, scale=0.5)
        for sched in ("exponential", "linear", "quadratic", "reciprocal", "constant"):
            for dtype_str in ("float64", "float32"):
                s_run = eg.SgdSolver.from_pairs(
                    4, [0, 1, 2], [1, 2, 3], [1.0, 1.0, 1.0], iterations=10, schedule=sched, seed=42, dtype=dtype_str
                )
                s_step = eg.SgdSolver.from_pairs(
                    4, [0, 1, 2], [1, 2, 3], [1.0, 1.0, 1.0], iterations=10, schedule=sched, seed=42, dtype=dtype_str
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
            [0],
            [2],
            [2.0],
            iterations=8,
            seed=42,
        )
        self.assertEqual(sh.num_springs, 1)
        self.assertEqual(sh.num_hinges, 1)
        lh = eg.Layout2d.from_coords([[0.0, 0.0], [3.0, 0.0], [0.0, 0.2]])
        self.assertEqual(sh.run(lh), "completed")
        self.assertTrue(lh.is_all_finite())

        sh_kw = eg.SgdSolver.from_pairs_with_hinges(
            3,
            [0],
            [1],
            [1.0],
            [0],
            [2],
            [2.0],
            spring_dji=[1.0],
            spring_wij=[1.0],
            spring_wji=[1.0],
            hinge_dji=[2.0],
            hinge_wij=[0.25],
            hinge_wji=[0.25],
            iterations=8,
            seed=42,
        )
        self.assertEqual(sh_kw.num_springs, 1)
        self.assertEqual(sh_kw.num_hinges, 1)

        # 2. full, sparse_barrier_dog, sparse_barrier_from_landmarks, random_pair, omega, streaming_negatives
        lm = eg.compute_regularized_dog_varadhan_landmarks(g, degree=20, num_landmarks=8, seed=42)
        emb = eg.chebyshev_dog_jl_embedding(g, degree=20, jl_dim=8, seed=42)

        s_full = eg.SgdSolver.full(g, iterations=15, seed=42)
        s_dog = eg.SgdSolver.sparse_barrier_dog(
            g, degree=20, num_landmarks=8, barrier_sym="sym_max", iterations=15, seed=42
        )
        s_lm = eg.SgdSolver.sparse_barrier_from_landmarks(
            g, lm, pivots=[0, 5, 10, 15], barrier_sym="sym_avg", iterations=15, seed=42
        )
        s_rp = eg.SgdSolver.random_pair(g, pairs_per_node=4, iterations=15, seed=42)
        s_om = eg.SgdSolver.omega(g, emb, k=4, min_dist=1e-3, iterations=15, seed=42)
        s_st = eg.SgdSolver.streaming_negatives(g, negatives_per_node=3, iterations=15, seed=42)

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
                ms_run = eg.SgdSolver.full(g, iterations=10, seed=42, dtype=dtype_str)
                ms_step = eg.SgdSolver.full(g, iterations=10, seed=42, dtype=dtype_str)
                ml_run = eg.ManifoldLayout2d.from_coords(kind, (init_coords * 0.25).astype(np_dtype))
                ml_step = eg.ManifoldLayout2d.from_coords(kind, (init_coords * 0.25).astype(np_dtype))

                self.assertEqual(ms_run.run(ml_run), "completed")
                self.assertEqual(ms_step.run_steps(ml_step, 4), "running")
                while not ms_step.is_finished:
                    ms_step.step(ml_step)
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
            (TypeError,), lambda: eg.SgdSolver.full(g, iterations=True)
        )
        self.assert_no_panic_raises(
            (IndexError,), lambda: eg.SgdSolver.from_pairs(2, [0], [5], [1.0])
        )


if __name__ == "__main__":
    unittest.main()
