from __future__ import annotations

from collections.abc import Callable
import math
import unittest
from typing import Any

import egraph as eg
import numpy as np
import numpy.typing as npt


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


def _make_disconnected_with_isolated() -> eg.CsrGraph:
    # Component 0: triangle (0,1,2); Component 1: edge (3,4); Isolated: vertex 5.
    return eg.CsrGraph.from_edges(
        6,
        [(0, 1), (1, 2), (0, 2), (3, 4)],
        directed=False,
        sanitize=True,
    )


def _make_directed_reciprocal() -> eg.CsrGraph:
    # Directed graph with reciprocal edges (0,1)+(1,0) and (1,2)+(2,1) plus one-way (2,3).
    return eg.CsrGraph.from_edges(
        4,
        [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3)],
        directed=True,
        sanitize=False,
    )


def _seeded_coords(n: int, seed: int = 42, scale: float = 1.0) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=(n, 2)).astype(np.float64)


def _collinear_coords(n: int) -> npt.NDArray[np.float64]:
    return np.column_stack(
        [np.linspace(-1.0, 1.0, n, dtype=np.float64), np.zeros(n, dtype=np.float64)]
    )


def _to_float_array(
    coords: npt.NDArray[np.float64], dtype_str: str
) -> npt.NDArray[np.floating[Any]]:
    if dtype_str == "float32":
        return coords.astype(np.float32)
    return coords.astype(np.float64)


class TestGripGemSelPfsSfc(unittest.TestCase):
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
        fn: Callable[[], Any],
    ) -> None:
        with self.assertRaises(expected) as ctx:
            fn()
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

    def test_grip_gem_sel_pfs_prime_sfc(self) -> None:
        graph = _make_barbell(5, bridge_len=2)
        n = graph.num_nodes
        base = _seeded_coords(n, seed=37, scale=2.0)

        # 1. GripSolver
        for dtype_str in ("float64", "float32"):
            l_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            l_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            g_run = eg.GripSolver(
                graph, rounds_min=3, rounds_max=8, seed=11, dtype=dtype_str
            )
            g_step = eg.GripSolver(
                graph, rounds_min=3, rounds_max=8, seed=11, dtype=dtype_str
            )
            self.assertGreaterEqual(g_run.num_levels, 1)
            self.assertIn("GripSolver", repr(g_run))
            g_run.run(l_run)
            g_step.step(l_step)
            g_step.run_steps(l_step, 2)
            g_step.run(l_step)
            self.assertTrue(g_run.is_finished)
            self.assertTrue(l_run.is_all_finite())
            np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))

        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.GripSolver(graph, rounds_min=10, rounds_max=5)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.GripSolver(graph, local_neighborhood_size=0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.GripSolver(graph, repulsion_scale=-0.1)
        )
        self.assert_no_panic_raises((ValueError,), lambda: eg.GripSolver(graph, edge_length=0.0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.GripSolver(graph, temp_r=-0.1))
        self.assert_no_panic_raises((ValueError,), lambda: eg.GripSolver(graph, temp_s=-1.0))

        # 2. GemSolver
        for dtype_str in ("float64", "float32"):
            l_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            l_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            m_run = eg.GemSolver(graph, max_iterations=12, seed=13, dtype=dtype_str)
            m_step = eg.GemSolver(graph, max_iterations=12, seed=13, dtype=dtype_str)
            self.assertIn("GemSolver", repr(m_run))
            m_run.run(l_run)
            m_step.step(l_step)
            m_step.run_steps(l_step, 3)
            m_step.run(l_step)
            self.assertTrue(m_run.is_finished)
            self.assertTrue(l_run.is_all_finite())
            np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))

        m_zero = eg.GemSolver(graph, max_iterations=0)
        self.assertTrue(m_zero.is_finished)
        self.assert_no_panic_raises((ValueError,), lambda: eg.GemSolver(graph, edge_length=0.0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.GemSolver(graph, gravity=-1.0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.GemSolver(graph, t_init=-1.0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.GemSolver(graph, t_max=0.0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.GemSolver(graph, t_min=-1.0))
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.GemSolver(graph, oscillation_angle=0.0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.GemSolver(graph, oscillation_angle=4.0)
        )
        self.assert_no_panic_raises((ValueError,), lambda: eg.GemSolver(graph, rotation_angle=0.0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.GemSolver(graph, rotation_angle=4.0))
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.GemSolver(graph, oscillation_sensitivity=-0.1)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.GemSolver(graph, rotation_sensitivity=-0.1)
        )
        self.assert_no_panic_raises((ValueError,), lambda: eg.GemSolver(graph, perturbation=-1.0))

        # 3. SelLayoutSolver (5 mappings x 2 backends x 2 dtypes)
        for dtype_str in ("float64", "float32"):
            for mapping in ("sigmoid", "linear", "tanh", "custom_logistic", "binary"):
                for backend in ("fr", "t_fdp"):
                    l_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
                    l_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
                    s_run = eg.SelLayoutSolver(
                        graph,
                        mapping=mapping,
                        layout_backend=backend,
                        iterations=8,
                        seed=41,
                        dtype=dtype_str,
                    )
                    s_step = eg.SelLayoutSolver(
                        graph,
                        mapping=mapping,
                        layout_backend=backend,
                        iterations=8,
                        seed=41,
                        dtype=dtype_str,
                    )
                    weights = s_run.modulated_weights()
                    self.assertEqual(weights.dtype, np.float64)
                    self.assertEqual(weights.ndim, 1)
                    self.assertTrue(np.all(np.isfinite(weights)))
                    self.assertTrue(math.isfinite(s_run.se_target))
                    self.assertIn("SelLayoutSolver", repr(s_run))
                    s_run.run(l_run)
                    s_step.step(l_step)
                    s_step.run_steps(l_step, 2)
                    s_step.run(l_step)
                    self.assertTrue(s_run.is_finished)
                    self.assertTrue(l_run.is_all_finite())
                    np.testing.assert_array_equal(
                        l_run.coords(copy=True), l_step.coords(copy=True)
                    )

        self.assert_no_panic_raises((ValueError,), lambda: eg.SelLayoutSolver(graph, tree_depth=0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.SelLayoutSolver(graph, pq_max_pops=0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.SelLayoutSolver(graph, mapping="bad"))
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SelLayoutSolver(graph, layout_backend="bad")
        )
        # Nested t_fdp_config must inherit valid IbFftConfig::default() grid settings
        for dtype_str in ("float64", "float32"):
            l_cfg = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            s_cfg = eg.SelLayoutSolver(
                graph,
                layout_backend="t_fdp",
                t_fdp_config={"alpha": 1.2, "iterations": 6},
                dtype=dtype_str,
            )
            s_cfg.run(l_cfg)
            self.assertTrue(s_cfg.is_finished)
            self.assertTrue(l_cfg.is_all_finite())
        l_custom = eg.Layout2d.from_coords(_to_float_array(base, "float64"))
        s_custom = eg.SelLayoutSolver(
            graph,
            layout_backend="t_fdp",
            t_fdp_config={
                "backend": "ib_fft",
                "iterations": 4,
                "alpha": 1.2,
                "interpolation_order": 3,
                "intervals": 16,
                "min_grid_dim": 32,
                "max_grid_dim": 128,
            },
        )
        s_custom.run(l_custom)
        self.assertTrue(s_custom.is_finished)
        self.assertTrue(l_custom.is_all_finite())
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.SelLayoutSolver(
                graph, layout_backend="t_fdp", t_fdp_config={"min_grid_dim": 50}
            ),
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.SelLayoutSolver(
                graph, layout_backend="t_fdp", t_fdp_config={"unknown_key": 1}
            ),
        )
        self.assert_no_panic_raises(
            (eg.InvalidInputError,),
            lambda: eg.SelLayoutSolver(
                graph, layout_backend="fr", fr_config={"theta": -1.0}
            ),
        )
        # Nested t_fdp_config Barnes-Hut must default theta to NeighborBarnesHut::default().theta (0.5)
        for dtype_str in ("float64", "float32"):
            l_bh_def = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            l_bh_05 = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            l_bh_12 = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            eg.SelLayoutSolver(
                graph,
                layout_backend="t_fdp",
                t_fdp_config={"backend": "barnes_hut", "iterations": 12, "seed": 42},
                dtype=dtype_str,
            ).run(l_bh_def)
            eg.SelLayoutSolver(
                graph,
                layout_backend="t_fdp",
                t_fdp_config={
                    "backend": "barnes_hut",
                    "theta": 0.5,
                    "iterations": 12,
                    "seed": 42,
                },
                dtype=dtype_str,
            ).run(l_bh_05)
            eg.SelLayoutSolver(
                graph,
                layout_backend="t_fdp",
                t_fdp_config={
                    "backend": "barnes_hut",
                    "theta": 1.2,
                    "iterations": 12,
                    "seed": 42,
                },
                dtype=dtype_str,
            ).run(l_bh_12)
            np.testing.assert_array_equal(
                l_bh_def.coords(copy=True), l_bh_05.coords(copy=True)
            )
            self.assertFalse(
                np.allclose(l_bh_def.coords(copy=True), l_bh_12.coords(copy=True))
            )

        # 4. PfsPrimeOverlapRemoval (new + from_radii, zero overlap, strict oo_nni == 0.0)
        hw = np.full(n, 0.8, dtype=np.float64)
        hh = np.full(n, 0.5, dtype=np.float64)
        radii = np.full(n, 0.6, dtype=np.float64)
        coincident = _seeded_coords(n, seed=51, scale=0.2)
        coincident[1] = coincident[0]

        for dtype_str in ("float64", "float32"):
            l_run = eg.Layout2d.from_coords(_to_float_array(coincident, dtype_str))
            l_cmp = eg.Layout2d.from_coords(_to_float_array(coincident, dtype_str))
            p_run = eg.PfsPrimeOverlapRemoval(hw, hh, padding=0.05, seed=7, dtype=dtype_str)
            p_cmp = eg.PfsPrimeOverlapRemoval(hw, hh, padding=0.05, seed=7, dtype=dtype_str)
            self.assertIn("PfsPrimeOverlapRemoval", repr(p_run))
            p_run.apply(l_run)
            p_cmp.apply(l_cmp)
            self.assertEqual(p_run.last_overlap_count, 0)
            np.testing.assert_array_equal(l_run.coords(copy=True), l_cmp.coords(copy=True))
            self.assertEqual(
                len(eg.rectangle_overlaps(l_run, hw, hh)),
                0,
            )

            # Collinear 1D layout with from_radii and NOR check
            l_orig = eg.Layout2d.from_coords(_to_float_array(_collinear_coords(n), dtype_str))
            l_adj = eg.Layout2d.from_coords(_to_float_array(_collinear_coords(n), dtype_str))
            p_rad = eg.PfsPrimeOverlapRemoval.from_radii(radii, padding=0.0, seed=7, dtype=dtype_str)
            p_rad.apply(l_adj)
            self.assertEqual(p_rad.last_overlap_count, 0)
            nor = eg.nor_metrics(l_orig, l_adj, radii, radii, graph=graph)
            self.assertAlmostEqual(float(nor["oo_nni"]), 0.0, places=12)

        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.PfsPrimeOverlapRemoval([1.0, 2.0], [1.0])
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.PfsPrimeOverlapRemoval([-1.0], [1.0])
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.PfsPrimeOverlapRemoval([math.nan], [1.0])
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.PfsPrimeOverlapRemoval([1.0], [1.0], padding=-0.1)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.PfsPrimeOverlapRemoval.from_radii([math.inf])
        )

        # 5. SfcLayout (hilbert & morton, zero co-location, permutation)
        for dtype_str in ("float64", "float32"):
            for curve in ("hilbert", "morton"):
                sfc1 = eg.SfcLayout(graph, curve=curve, scale=80.0, seed=9, dtype=dtype_str)
                sfc2 = eg.SfcLayout(graph, curve=curve, scale=80.0, seed=9, dtype=dtype_str)
                self.assertIn("SfcLayout", repr(sfc1))
                l_run = sfc1.generate()
                l_cmp = sfc2.generate()
                order = sfc1.vertex_order()
                self.assertEqual(order.dtype, np.uint32)
                self.assertEqual(sorted(order.tolist()), list(range(n)))
                np.testing.assert_array_equal(l_run.coords(copy=True), l_cmp.coords(copy=True))
                coords = l_run.coords(copy=False)
                self.assertTrue(np.all(np.abs(coords) <= 40.0 + 1e-4))
                degen = eg.layout_degeneracy(graph, l_run)
                self.assertEqual(degen["duplicate_nodes_f32"], 0)

        self.assert_no_panic_raises((ValueError,), lambda: eg.SfcLayout(graph, scale=0.0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.SfcLayout(graph, scale=math.nan))
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SfcLayout(graph, resolution_order=0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SfcLayout(graph, resolution_order=16)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SfcLayout(graph, resolution_order=1)
        )
        self.assert_no_panic_raises((ValueError,), lambda: eg.SfcLayout(graph, curve="peano"))

        # Shared degenerate topologies across Grip, Gem, Sel, PfsPrime, Sfc
        for g_deg in (
            eg.CsrGraph.from_edges(0, [], directed=False),
            eg.CsrGraph.from_edges(1, [], directed=False),
            eg.CsrGraph.from_edges(2, [(0, 1)], directed=False),
            eg.CsrGraph.from_edges(5, [], directed=False),
            _make_disconnected_with_isolated(),
            _make_directed_reciprocal(),
        ):
            m = g_deg.num_nodes
            init = _collinear_coords(m) if m > 0 else np.zeros((0, 2), dtype=np.float64)
            degenerate_solvers: list[
                eg.GripSolver
                | eg.GemSolver
                | eg.SelLayoutSolver
            ] = [
                eg.GripSolver(g_deg, rounds_min=2, rounds_max=4),
                eg.GemSolver(g_deg, max_iterations=4),
                eg.SelLayoutSolver(g_deg, iterations=4),
            ]
            for deg_solver in degenerate_solvers:
                lay = eg.Layout2d.from_coords(init)
                deg_solver.run(lay)
                self.assertTrue(lay.is_all_finite())
            lay_sfc = eg.SfcLayout(g_deg).generate()
            self.assertTrue(lay_sfc.is_all_finite())
            lay_pfs = eg.Layout2d.from_coords(init)
            eg.PfsPrimeOverlapRemoval.from_radii(
                np.full(m, 0.25, dtype=np.float64)
            ).apply(lay_pfs)
            self.assertTrue(lay_pfs.is_all_finite())

        # Bad layouts rejected without panics
        bad_lay = eg.Layout2d.zeros(n)
        bad_lay.coords(copy=False)[0, 0] = math.nan
        wrong_lay = eg.Layout2d.zeros(2)
        bad_solvers: list[
            eg.GripSolver
            | eg.GemSolver
            | eg.SelLayoutSolver
        ] = [
            eg.GripSolver(graph, rounds_min=1, rounds_max=2),
            eg.GemSolver(graph, max_iterations=2),
            eg.SelLayoutSolver(graph, iterations=2),
        ]
        for bad_solver in bad_solvers:

            def _run_bad(
                s: eg.GripSolver
                | eg.GemSolver
                | eg.SelLayoutSolver = bad_solver,
            ) -> str:
                return s.run(bad_lay)

            def _run_wrong(
                s: eg.GripSolver
                | eg.GemSolver
                | eg.SelLayoutSolver = bad_solver,
            ) -> str:
                return s.run(wrong_lay)

            self.assert_no_panic_raises((ValueError,), _run_bad)
            self.assert_no_panic_raises((ValueError,), _run_wrong)

        pfs_bad = eg.PfsPrimeOverlapRemoval(hw, hh)
        self.assert_no_panic_raises((ValueError,), lambda: pfs_bad.apply(bad_lay))
        self.assert_no_panic_raises((ValueError,), lambda: pfs_bad.apply(wrong_lay))


if __name__ == "__main__":
    unittest.main()
