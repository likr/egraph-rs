from __future__ import annotations

import math
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

import egraph as eg
import egraph._core as eg_core
import egraph._core_core as eg_core_core
import numpy as np
import numpy.typing as npt
from egraph._core_types import BackendKind, TargetKind


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


def _circle_coords(n: int, radius: float = 1.0) -> npt.NDArray[np.float64]:
    angles = np.linspace(
        0.0, 2.0 * math.pi, num=max(n, 1), endpoint=False, dtype=np.float64
    )
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def _typed_coords(coords: npt.NDArray[np.float64], dtype_str: str) -> npt.NDArray[Any]:
    return (
        coords.astype(np.float32)
        if dtype_str == "float32"
        else coords.astype(np.float64)
    )


class TestSolverSeamsE2E(unittest.TestCase):
    @contextmanager
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
    ) -> Iterator[None]:
        with self.assertRaises(expected) as ctx:
            yield
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

    def test_error_hierarchy_and_exports_for_backend_and_device_errors(self) -> None:
        for name in ("BackendUnavailableError", "DeviceLimitExceededError"):
            self.assertIn(name, eg.__all__)
            cls_top = getattr(eg, name)
            cls_core = getattr(eg_core, name)
            cls_mod = getattr(eg_core_core, name)
            self.assertIs(cls_top, cls_core)
            self.assertIs(cls_top, cls_mod)
            self.assertTrue(issubclass(cls_top, eg.EgraphError))
            self.assertTrue(issubclass(cls_top, ValueError))
            self.assertFalse(issubclass(cls_top, IndexError))
            self.assertFalse(issubclass(cls_top, eg.UnsupportedOperationError))
            self.assertFalse(issubclass(cls_top, eg.InvalidInputError))

            err = cls_top("simulated failure")
            self.assertIsInstance(err, eg.EgraphError)
            self.assertIsInstance(err, ValueError)
            self.assertNotIsInstance(err, IndexError)
            try:
                raise err
            except eg.InvalidInputError:
                self.fail(f"{name} must not match InvalidInputError")
            except eg.EgraphError as caught:
                self.assertEqual(str(caught), "simulated failure")

    def test_ib_umap_all_backends_constructors_and_telemetry_f64_f32(self) -> None:
        g = _make_grid(4, 4)
        n = g.num_nodes
        base = _circle_coords(n, radius=1.2)
        knn = eg.build_pure_2hop_knn(g).knn
        probs = eg.compute_fuzzy_simplicial_set(knn)
        lm = eg.compute_regularized_dog_varadhan_landmarks(g, num_landmarks=6, seed=42)

        dtypes: tuple[Literal["float64", "float32"], ...] = ("float64", "float32")
        backends: tuple[BackendKind, ...] = ("ib_fft", "barnes_hut", "exact")
        for dtype_str in dtypes:
            for backend in backends:
                l_run = eg.Layout2d.from_coords(_typed_coords(base, dtype_str))
                l_step = eg.Layout2d.from_coords(_typed_coords(base, dtype_str))
                s_run = eg.IbUmapSolver(
                    g,
                    backend=backend,
                    iterations=8,
                    num_landmarks=6,
                    seed=42,
                    dtype=dtype_str,
                )
                s_step = eg.IbUmapSolver(
                    g,
                    backend=backend,
                    iterations=8,
                    num_landmarks=6,
                    seed=42,
                    dtype=dtype_str,
                )
                self.assertGreater(s_run.a, 0.0)
                self.assertGreater(s_run.b, 0.0)
                self.assertEqual(s_run.ab, (s_run.a, s_run.b))

                self.assertEqual(s_run.run(l_run), "completed")
                s_step.step(l_step)
                s_step.run_steps(l_step, 3)
                s_step.run(l_step)
                np.testing.assert_array_equal(
                    l_run.coords(copy=True), l_step.coords(copy=True)
                )
                self.assertTrue(l_run.is_all_finite())

                if backend == "ib_fft":
                    self.assertGreater(s_run.total_fft_2d_calls, 0)
                    fft_stats = s_run.last_fft_stats()
                    self.assertIsNotNone(fft_stats)
                    assert fft_stats is not None
                    self.assertGreater(fft_stats.total_fft_2d_calls, 0)
                    self.assertGreater(fft_stats.nx, 0)
                    self.assertGreater(fft_stats.ny, 0)
                else:
                    self.assertEqual(s_run.total_fft_2d_calls, 0)
                    self.assertIsNone(s_run.last_fft_stats())

            l_knn = eg.Layout2d.from_coords(_typed_coords(base, dtype_str))
            l_fset = eg.Layout2d.from_coords(_typed_coords(base, dtype_str))
            l_lm = eg.Layout2d.from_coords(_typed_coords(base, dtype_str))
            eg.IbUmapSolver.from_knn(
                knn, backend="exact", iterations=6, seed=42, dtype=dtype_str
            ).run(l_knn)
            eg.IbUmapSolver.from_fuzzy_set(
                probs,
                degrees=[4] * n,
                backend="exact",
                iterations=6,
                seed=42,
                dtype=dtype_str,
            ).run(l_fset)
            eg.IbUmapSolver.from_landmarks(
                g,
                lm,
                max_2hop_per_node=32,
                backend="exact",
                iterations=6,
                seed=42,
                dtype=dtype_str,
            ).run(l_lm)
            self.assertTrue(l_knn.is_all_finite())
            self.assertTrue(l_fset.is_all_finite())
            self.assertTrue(l_lm.is_all_finite())

    def test_ib_umap_target_config_and_layout_validation_guards(self) -> None:
        g = _make_grid(3, 3)
        knn = eg.build_pure_2hop_knn(g).knn
        probs = eg.compute_fuzzy_simplicial_set(knn)
        lm = eg.compute_regularized_dog_varadhan_landmarks(g, num_landmarks=4, seed=42)

        bad_targets: tuple[TargetKind, ...] = ("layout3d", "layout_nd", "manifold2d")
        for bad_target in bad_targets:
            with self.assert_no_panic_raises((eg.UnsupportedOperationError,)):
                eg.IbUmapSolver(g, target=bad_target)
            with self.assert_no_panic_raises((eg.UnsupportedOperationError,)):
                eg.IbUmapSolver.from_knn(knn, target=bad_target)
            with self.assert_no_panic_raises((eg.UnsupportedOperationError,)):
                eg.IbUmapSolver.from_fuzzy_set(probs, target=bad_target)
            with self.assert_no_panic_raises((eg.UnsupportedOperationError,)):
                eg.IbUmapSolver.from_landmarks(g, lm, target=bad_target)

        with self.assert_no_panic_raises((eg.UnsupportedOperationError,)):
            eg.IbUmapSolver(g, target="unknown")  # type: ignore[arg-type]
        with self.assert_no_panic_raises((eg.UnsupportedOperationError,)):
            eg.IbUmapSolver(g, backend="unknown_backend")  # type: ignore[arg-type]

        bad_float_params: tuple[dict[str, float | int | BackendKind], ...] = (
            {"backend": "barnes_hut", "theta": -0.1},
            {"backend": "ib_fft", "intervals": 0},
            {"learning_rate": 0.0},
            {"c_k": -1.0},
            {"tau_scale": 0.0},
            {"min_dist": -0.1},
            {"spread": 0.0},
        )
        for bad_kw in bad_float_params:
            with self.assert_no_panic_raises((eg.InvalidInputError,)):
                eg.IbUmapSolver(g, **bad_kw)  # type: ignore[arg-type]

        with self.assert_no_panic_raises((eg.NonFiniteValueError,)):
            eg.IbUmapSolver(g, learning_rate=math.nan)
        with self.assert_no_panic_raises(
            (eg.InvalidInputError, eg.NonFiniteValueError)
        ):
            eg.IbUmapSolver(g, backend="barnes_hut", theta=math.nan)
        with self.assert_no_panic_raises((eg.NonFiniteValueError,)):
            eg.IbUmapSolver(g, c_k=math.nan)

        with self.assert_no_panic_raises((eg.DimensionMismatchError,)):
            eg.IbUmapSolver.from_fuzzy_set(probs, degrees=[4, 4])

        g_small = _make_grid(2, 2)
        lm_small = eg.compute_regularized_dog_varadhan_landmarks(
            g_small, num_landmarks=2, seed=42
        )
        with self.assert_no_panic_raises((eg.DimensionMismatchError,)):
            eg.IbUmapSolver.from_landmarks(g, lm_small)

        # Dual-dtype solver intentionally coerces float64 <-> float32 Layout2d, while
        # rejecting mismatched node counts, wrong layout target types, and NaN/Inf coords.
        solver_f32 = eg.IbUmapSolver(g, backend="exact", iterations=4, dtype="float32")
        coerced_layout = eg.Layout2d.init(9, seed=5, dtype="float64")
        self.assertEqual(solver_f32.step(coerced_layout), "running")
        self.assertTrue(coerced_layout.is_all_finite())

        with self.assert_no_panic_raises((TypeError,)):
            solver_f32.step(eg.Layout3d.zeros(9, dtype="float32"))  # type: ignore[arg-type]
        bad_len_layout = eg.Layout2d.zeros(4, dtype="float32")
        with self.assert_no_panic_raises((eg.DimensionMismatchError,)):
            solver_f32.step(bad_len_layout)

        for bad_val in (math.nan, math.inf, -math.inf):
            bad_arr = np.zeros((9, 2), dtype=np.float64)
            bad_arr[0, 0] = bad_val
            with self.assert_no_panic_raises((eg.NonFiniteValueError,)):
                eg.Layout2d.from_coords(bad_arr)

            nf_lay = eg.Layout2d.zeros(9, dtype="float32")
            nf_view = nf_lay.coords(copy=False)
            nf_view[0, 0] = bad_val
            iter_before = solver_f32.iteration
            with self.assert_no_panic_raises((eg.NonFiniteValueError,)):
                solver_f32.validate_target(nf_lay)
            with self.assert_no_panic_raises((eg.NonFiniteValueError,)):
                solver_f32.run_steps(nf_lay, 0)
            with self.assert_no_panic_raises((eg.NonFiniteValueError,)):
                solver_f32.step(nf_lay)
            self.assertEqual(solver_f32.iteration, iter_before)

    def test_ib_umap_boundary_empty_singleton_zero_iters_and_zero_jitter(self) -> None:
        backends: tuple[BackendKind, ...] = ("ib_fft", "barnes_hut", "exact")
        for n in (0, 1):
            g_small = eg.CsrGraph.from_edges(n, [], directed=False, sanitize=False)
            for backend in backends:
                s = eg.IbUmapSolver(
                    g_small, backend=backend, iterations=4, dtype="float32"
                )
                lay = eg.Layout2d.zeros(n, dtype="float32")
                status = s.run(lay)
                self.assertIn(status, ("converged", "completed"))
                self.assertTrue(lay.is_all_finite())

        g9 = _make_grid(3, 3)
        s0 = eg.IbUmapSolver(g9, backend="exact", iterations=0, dtype="float32")
        self.assertTrue(s0.is_finished)
        l_zero_iter = eg.Layout2d.init(9, seed=3, dtype="float32")
        snap = l_zero_iter.coords(copy=True)
        self.assertEqual(s0.step(l_zero_iter), "completed")
        np.testing.assert_array_equal(l_zero_iter.coords(copy=True), snap)

        l_zero = eg.Layout2d.zeros(9, dtype="float32")
        s_jit = eg.IbUmapSolver(
            g9, backend="exact", iterations=6, seed=77, dtype="float32"
        )
        self.assertEqual(s_jit.run(l_zero), "completed")
        self.assertTrue(l_zero.is_all_finite())
        self.assertGreater(float(np.abs(l_zero.coords(copy=True)).max()), 1e-6)

    def test_end_to_end_grid_quality_workflow_f32_parity_baseline(self) -> None:
        g = _make_grid(4, 4)
        init_coords = _circle_coords(16, radius=1.0).astype(np.float32)
        results: list[npt.NDArray[np.float32]] = []
        backends: tuple[BackendKind, ...] = ("ib_fft", "barnes_hut", "exact")
        for backend in backends:
            lay = eg.Layout2d.from_coords(init_coords.copy())
            solver = eg.IbUmapSolver(
                g,
                backend=backend,
                iterations=20,
                num_landmarks=8,
                seed=42,
                dtype="float32",
            )
            self.assertEqual(solver.run(lay), "completed")
            report = eg.evaluate_metrics(
                g,
                lay,
                ["neighborhood_preservation_2hop", "scale_normalized_stress"],
            )
            self.assertIsNotNone(report.neighborhood_preservation_2hop)
            self.assertIsNotNone(report.scale_normalized_stress)
            assert report.neighborhood_preservation_2hop is not None
            assert report.scale_normalized_stress is not None
            self.assertGreater(report.neighborhood_preservation_2hop, 0.3)
            self.assertTrue(math.isfinite(report.scale_normalized_stress))
            results.append(lay.coords(copy=True))

        self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
