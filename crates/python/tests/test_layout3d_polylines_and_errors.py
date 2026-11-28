from __future__ import annotations

import math
import unittest

import egraph as eg
from egraph._core_types import (
    IterativeSolver,
    LayoutGenerator,
    LayoutStorage,
    LayoutTransform,
)
import numpy as np
import numpy.typing as npt


def _to_float_array(
    arr: npt.NDArray[np.float64], dtype_str: str
) -> npt.NDArray[np.float64] | npt.NDArray[np.float32]:
    return arr.astype(np.float32) if dtype_str == "float32" else arr.astype(np.float64)


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


class TestLayout3dPolylinesAndErrors(unittest.TestCase):
    def test_exception_hierarchy_and_typed_errors(self) -> None:
        self.assertTrue(issubclass(eg.EgraphError, ValueError))
        for exc_cls in (
            eg.InvalidInputError,
            eg.DimensionMismatchError,
            eg.IndexOutOfBoundsError,
            eg.InvalidCsrError,
            eg.NonFiniteValueError,
            eg.UnsupportedOperationError,
            eg.NumericalBreakdownError,
        ):
            self.assertTrue(issubclass(exc_cls, eg.EgraphError))
            self.assertTrue(issubclass(exc_cls, ValueError))
        self.assertTrue(issubclass(eg.IndexOutOfBoundsError, IndexError))

        g = _make_grid(3, 3)

        with self.assertRaises(eg.InvalidInputError) as ctx_inp:
            eg.SgdSolver.full(g, eps=0.0)
        self.assertIsInstance(ctx_inp.exception, ValueError)

        with self.assertRaises(eg.DimensionMismatchError):
            eg.SgdSolver.full(g, target="layout2d").step(eg.Layout2d.zeros(4))

        lay2 = eg.Layout2d.zeros(4)
        lay3 = eg.Layout3d.zeros(4)
        laynd = eg.LayoutNd.zeros(4, 3)
        with self.assertRaises(eg.IndexOutOfBoundsError) as ctx_idx:
            lay2.x(99)
        self.assertIsInstance(ctx_idx.exception, IndexError)
        self.assertIsInstance(ctx_idx.exception, ValueError)
        with self.assertRaises(eg.IndexOutOfBoundsError):
            lay3.z(99)
        with self.assertRaises(eg.IndexOutOfBoundsError):
            laynd.get(99, 0)

        with self.assertRaises(eg.InvalidCsrError):
            eg.CsrGraph.from_csr([0, 2, 1], [0, 1])

        with self.assertRaises(eg.NonFiniteValueError):
            eg.Layout3d.from_coords([[0.0, math.nan, 0.0], [1.0, 1.0, 1.0]])

        with self.assertRaises(eg.UnsupportedOperationError):
            eg.SgdSolver.full(g, target="unknown_target")  # type: ignore[arg-type]

    def test_layout3d_lifecycle_solvers_generators_and_metrics(self) -> None:
        g = _make_grid(3, 3)
        n = g.num_nodes
        self.assertEqual(len(eg.Layout3d.zeros(n)), n)
        self.assertTrue(eg.Layout3d.zeros(0).is_empty())

        for dtype_str in ("float64", "float32"):
            np_dtype = np.float64 if dtype_str == "float64" else np.float32
            lay = eg.Layout3d.init(n, seed=42, scale=2.0, dtype=dtype_str)
            self.assertEqual(lay.dtype, dtype_str)
            self.assertEqual(lay.num_nodes, n)
            self.assertTrue(lay.is_all_finite())
            self.assertIsInstance(lay, LayoutStorage)

            view = lay.coords(copy=False)
            self.assertEqual(view.shape, (n, 3))
            self.assertEqual(view.dtype, np_dtype)
            orig_x0 = lay.x(0)
            view[0, 2] = 7.5
            self.assertAlmostEqual(lay.z(0), 7.5, places=5)
            lay.set_x(0, 3.25)
            lay.set_y(0, -1.5)
            lay.set_z(0, 2.5)
            self.assertAlmostEqual(lay.x(0), 3.25, places=5)
            self.assertAlmostEqual(lay.y(0), -1.5, places=5)
            self.assertAlmostEqual(lay.z(0), 2.5, places=5)
            self.assertNotAlmostEqual(orig_x0, 999.0)

            copied = lay.copy()
            copied.set_z(1, 42.0)
            self.assertNotAlmostEqual(lay.z(1), 42.0)
            copied.normalize_std(1.5)
            self.assertTrue(copied.is_all_finite())

            base_seq = np.arange(n, dtype=np.float64)
            xs = _to_float_array(base_seq, dtype_str)
            ys = _to_float_array(base_seq * 0.5, dtype_str)
            zs = _to_float_array(-base_seq * 0.25, dtype_str)
            from_xyz = eg.Layout3d.from_xyz(xs, ys, zs, dtype=dtype_str)
            self.assertEqual(from_xyz.dtype, dtype_str)
            np.testing.assert_allclose(from_xyz.coords(copy=True)[:, 2], zs, atol=1e-5)

            for gen in (
                eg.ClassicalMds(g, target="layout3d", dtype=dtype_str),
                eg.PivotMds(g, target="layout3d", num_pivots=4, min_degree=1, dtype=dtype_str),
                eg.Hde(g, target="layout3d", dimensions=6, dtype=dtype_str),
                eg.RdMds(g, target="layout3d", dtype=dtype_str),
                eg.KorenSpectral(g, target="layout3d", dtype=dtype_str),
            ):
                self.assertIsInstance(gen, LayoutGenerator)
                out_l3 = gen.generate()
                self.assertIsInstance(out_l3, eg.Layout3d)
                self.assertEqual(out_l3.dtype, dtype_str)
                self.assertTrue(out_l3.is_all_finite())

            for solver in (
                eg.SgdSolver.full(g, target="layout3d", iterations=5, seed=7, dtype=dtype_str),
                eg.StressMajorizationSolver(g, target="layout3d", max_iterations=5, dtype=dtype_str),
                eg.KamadaKawaiSolver(g, target="layout3d", max_iterations=5, dtype=dtype_str),
                eg.MaxentStressSolver(
                    g, target="layout3d", backend="barnes_hut", stages=2, steps_per_stage=2, dtype=dtype_str
                ),
                eg.FruchtermanReingoldSolver(
                    g, target="layout3d", backend="barnes_hut", iterations=5, dtype=dtype_str
                ),
                eg.LinLogSolver(
                    g, target="layout3d", backend="barnes_hut", iterations=5, dtype=dtype_str
                ),
                eg.ForceAtlas2Solver(
                    g, target="layout3d", backend="barnes_hut", iterations=5, dtype=dtype_str
                ),
                eg.YifanHuSolver(
                    g, target="layout3d", max_iterations_per_level=4, dtype=dtype_str
                ),
                eg.TsNetSolver.pure_2hop(
                    g, target="layout3d", backend="barnes_hut", stage_iterations=(2, 2, 2), dtype=dtype_str
                ),
                eg.GuMapSolver.from_graph_dog(
                    g, max_2hop_per_node=16, target="layout3d", num_landmarks=4, iterations=5, dtype=dtype_str
                ),
                eg.TFdpSolver(
                    g, target="layout3d", backend="barnes_hut", iterations=5, dtype=dtype_str
                ),
            ):
                self.assertIsInstance(solver, IterativeSolver)
                l3_run = eg.Layout3d.init(n, seed=17, dtype=dtype_str)
                l3_step = l3_run.copy()
                self.assertEqual(solver.run(l3_run), "completed")
                self.assertTrue(l3_run.is_all_finite())
                self.assertGreater(eg.neighborhood_preservation_2hop(g, l3_run), 0.0)
                self.assertTrue(math.isfinite(eg.scale_normalized_stress(g, l3_run)))
                rep = eg.evaluate_metrics(
                    g, l3_run, ["neighborhood_preservation_1hop", "neighborhood_preservation_2hop"]
                )
                self.assertIsNotNone(rep.neighborhood_preservation_2hop)
                self.assertTrue(l3_step.is_all_finite())

    def test_uaf_safe_edge_polylines_and_fdeb_targets(self) -> None:
        g = _make_grid(3, 3)
        n = g.num_nodes
        for dtype_str in ("float64", "float32"):
            np_dtype = np.float64 if dtype_str == "float64" else np.float32
            for target_lay in (
                eg.Layout2d.init(n, seed=9, scale=3.0, dtype=dtype_str),
                eg.Layout3d.init(n, seed=9, scale=3.0, dtype=dtype_str),
                eg.LayoutNd.init(n, 4, seed=9, scale=3.0, dtype=dtype_str),
            ):
                poly_run = eg.EdgePolylines.straight_line(g, target_lay)
                poly_from_graph = eg.EdgePolylines.straight_line_from_graph(g, target_lay)
                np.testing.assert_array_equal(
                    poly_run.edge_offsets(copy=True), poly_from_graph.edge_offsets(copy=True)
                )
                np.testing.assert_array_equal(
                    poly_run.points(copy=True), poly_from_graph.points(copy=True)
                )
                poly_step = poly_run.copy()
                offs_before = poly_run.edge_offsets(copy=False)
                pts_before = poly_run.points(copy=False)
                offs_snapshot = offs_before.copy()
                old_snapshot = pts_before.copy()
                self.assertEqual(int(offs_snapshot[-1]), 2 * g.num_edges)
                self.assertEqual(poly_run.num_edges, g.num_edges)
                self.assertEqual(poly_run.dtype, dtype_str)

                f_run = eg.FdebSolver(poly_run, cycles=3, steps_per_cycle=4)
                f_step = eg.FdebSolver(poly_step, cycles=3, steps_per_cycle=4)
                self.assertEqual(f_run.run(poly_run), "completed")
                while not f_step.is_finished:
                    f_step.step(poly_step)

                # Retained (edge_offsets, points) snapshot pair must remain intact and synchronized.
                np.testing.assert_array_equal(offs_before, offs_snapshot)
                np.testing.assert_array_equal(pts_before, old_snapshot)
                self.assertEqual(int(offs_before[-1]), pts_before.shape[0])

                pts_after = poly_run.points(copy=False)
                offs_after = poly_run.edge_offsets(copy=False)
                self.assertEqual(pts_after.dtype, np_dtype)
                self.assertGreater(poly_run.total_points, 2 * g.num_edges)
                self.assertEqual(int(offs_after[-1]), pts_after.shape[0])
                self.assertTrue(poly_run.is_all_finite())
                np.testing.assert_array_equal(pts_after, poly_step.points(copy=True))

    def test_option1_backend_matrix_and_manifold_kind_lock(self) -> None:
        g = _make_grid(3, 3)
        illegal_pairs: list[tuple[str, object]] = [
            ("TsNet layout_nd + fit_fft", lambda: eg.TsNetSolver.pure_2hop(g, target="layout_nd", backend="fit_fft")),
            ("TsNet layout3d + ib_fft", lambda: eg.TsNetSolver.pure_2hop(g, target="layout3d", backend="ib_fft")),
            ("TFdp layout_nd + grid", lambda: eg.TFdpSolver(g, target="layout_nd", backend="grid")),
            ("IbUmap layout3d + ib_fft", lambda: eg.IbUmapSolver(g, target="layout3d", backend="ib_fft")),
            ("IbUmap layout3d + barnes_hut", lambda: eg.IbUmapSolver(g, target="layout3d", backend="barnes_hut")),
            ("Maxent layout_nd + barnes_hut", lambda: eg.MaxentStressSolver(g, target="layout_nd", backend="barnes_hut")),
            ("FR manifold2d + barnes_hut", lambda: eg.FruchtermanReingoldSolver(g, target="manifold2d", backend="barnes_hut")),
            ("LinLog layout_nd + barnes_hut", lambda: eg.LinLogSolver(g, target="layout_nd", backend="barnes_hut")),
            ("FA2 layout_nd + barnes_hut", lambda: eg.ForceAtlas2Solver(g, target="layout_nd", backend="barnes_hut")),
        ]
        for label, fn in illegal_pairs:
            with self.assertRaises(eg.UnsupportedOperationError, msg=label):
                fn()  # type: ignore[operator]

        # ManifoldKind locking: first layout locks the manifold kind; mixed manifold raises.
        m_hyp = eg.ManifoldLayout2d.init("hyperbolic", 9, scale=0.2, seed=1)
        m_sph = eg.ManifoldLayout2d.init("spherical", 9, scale=0.2, seed=2)
        for make_m in (
            lambda: eg.SgdSolver.full(g, target="manifold2d", iterations=4),
            lambda: eg.TsNetSolver.pure_2hop(g, target="manifold2d", backend="exact", stage_iterations=(1, 1, 1)),
            lambda: eg.GuMapSolver.from_graph_dog(g, target="manifold2d", num_landmarks=4, iterations=4),
        ):
            s = make_m()
            s.step(m_hyp)
            with self.assertRaises((eg.UnsupportedOperationError, eg.DimensionMismatchError)):
                s.step(m_sph)

        # Keyword-only '*' guards & Runtime Protocols
        with self.assertRaises(TypeError):
            eg.SgdSolver.full(g, "layout2d")  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            eg.SpatialIntruderRefiner(g, 4)  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            eg.TsNetSolver.from_graph_partial_bfs(g, 8)  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            eg.ClassicalMds(g, "layout2d")  # type: ignore[call-arg]

        pfs = eg.PfsPrimeOverlapRemoval.from_radii([0.2] * 9)
        self.assertIsInstance(pfs, LayoutTransform)


if __name__ == "__main__":
    unittest.main()
