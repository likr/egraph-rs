from __future__ import annotations

import inspect
import unittest
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

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


class TestForceAndOverlapSolvers(unittest.TestCase):
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
        fn: Callable[[], Any],
    ) -> None:
        with self.assertRaises(expected) as ctx:
            fn()
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

    def test_linlog_and_fr_solvers(self) -> None:
        sig_ll = inspect.signature(eg.LinLogSolver)
        self.assertEqual(sig_ll.parameters["repulsion_model"].default, "edge_repulsion")
        self.assertEqual(sig_ll.parameters["repulsion_exponent"].default, -1.0)
        self.assertEqual(sig_ll.parameters["dtype"].default, "float64")

        sig_fr = inspect.signature(eg.FruchtermanReingoldSolver)
        self.assertEqual(sig_fr.parameters["backend"].default, "exact")
        self.assertEqual(sig_fr.parameters["theta"].default, 1.2)

        graph = _make_barbell(5, bridge_len=2)
        n = graph.num_nodes
        base = _seeded_coords(n, seed=13, scale=2.0)

        for dtype_str in ("float64", "float32"):
            for rep_model in ("edge_repulsion", "node_repulsion"):
                l_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
                l_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
                ll1 = eg.LinLogSolver(
                    graph,
                    iterations=15,
                    repulsion_model=rep_model,
                    use_barnes_hut=(rep_model == "edge_repulsion"),
                    seed=7,
                    dtype=dtype_str,
                )
                ll2 = eg.LinLogSolver(
                    graph,
                    iterations=15,
                    repulsion_model=rep_model,
                    use_barnes_hut=(rep_model == "edge_repulsion"),
                    seed=7,
                    dtype=dtype_str,
                )
                self.assertEqual(ll1.num_nodes, n)
                self.assertEqual(ll1.dtype, dtype_str)
                assert ll1.max_displacement is not None
                self.assertGreater(ll1.max_displacement, 0.0)
                self.assertIn("LinLogSolver", repr(ll1))
                e_before = ll1.energy(l_run)
                ll1.run(l_run)
                e_after = ll1.energy(l_run)
                self.assertTrue(np.isfinite(e_before) and np.isfinite(e_after))
                self.assertLess(e_after, e_before)

                ll2.step(l_step)
                ll2.run_steps(l_step, 4)
                ll2.run(l_step)
                self.assertTrue(ll2.is_finished)
                np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))

            for backend in ("exact", "barnes_hut", "grid"):
                l_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
                l_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
                fr1 = eg.FruchtermanReingoldSolver(
                    graph, iterations=12, backend=backend, seed=11, dtype=dtype_str
                )
                fr2 = eg.FruchtermanReingoldSolver(
                    graph, iterations=12, backend=backend, seed=11, dtype=dtype_str
                )
                self.assertGreater(fr1.optimal_distance, 0.0)
                self.assertGreater(fr1.current_temperature, 0.0)
                self.assertIn("FruchtermanReingoldSolver", repr(fr1))
                fr1.run(l_run)
                fr2.step(l_step)
                fr2.run_steps(l_step, 3)
                fr2.run(l_step)
                self.assertEqual(fr1.current_temperature, 0.0)
                np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))

        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.LinLogSolver(graph, repulsion_model="bad")
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.LinLogSolver(graph, theta=-1.0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.FruchtermanReingoldSolver(graph, backend="bad")
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.FruchtermanReingoldSolver(graph, k=0.0)
        )

    def test_force_atlas2_prevent_overlap(self) -> None:
        sig_fa2 = inspect.signature(eg.ForceAtlas2Solver)
        self.assertIs(sig_fa2.parameters["prevent_overlap"].default, False)
        self.assertEqual(sig_fa2.parameters["default_node_radius"].default, 1.0)
        self.assertIsNone(sig_fa2.parameters["node_radii"].default)

        g = _make_grid(3, 3)
        n = g.num_nodes
        radii = np.full(n, 2.0, dtype=np.float64)
        base = _seeded_coords(n, seed=5, scale=0.2)

        for dtype_str in ("float64", "float32"):
            l_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            l_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            fa1 = eg.ForceAtlas2Solver(
                g,
                iterations=30,
                prevent_overlap=True,
                default_node_radius=1.5,
                node_radii=radii,
                seed=42,
                dtype=dtype_str,
            )
            fa2 = eg.ForceAtlas2Solver(
                g,
                iterations=30,
                prevent_overlap=True,
                default_node_radius=1.5,
                node_radii=radii,
                seed=42,
                dtype=dtype_str,
            )
            self.assertTrue(fa1.prevent_overlap)
            self.assertEqual(fa1.default_node_radius, 1.5)
            np.testing.assert_allclose(fa1.node_radii, radii)

            fa1.run(l_run)
            fa2.step(l_step)
            fa2.run_steps(l_step, 10)
            fa2.run(l_step)
            np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))
            self.assertTrue(l_run.is_all_finite())

        fa_def = eg.ForceAtlas2Solver(
            g, iterations=10, prevent_overlap=True, default_node_radius=2.5, seed=42
        )
        self.assertTrue(fa_def.prevent_overlap)
        self.assertEqual(fa_def.default_node_radius, 2.5)
        np.testing.assert_allclose(fa_def.node_radii, np.full(n, 2.5))
        l_def = eg.Layout2d.from_coords(base)
        fa_def.run(l_def)
        self.assertTrue(l_def.is_all_finite())

        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.ForceAtlas2Solver(g, default_node_radius=-0.5)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.ForceAtlas2Solver(g, node_radii=[1.0, 1.0])
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.ForceAtlas2Solver(g, node_radii=[-1.0] * n)
        )

    def test_gtree_and_forbid_overlap_removal(self) -> None:
        sig_fb = inspect.signature(eg.ForbidOverlapRemoval)
        self.assertEqual(sig_fb.parameters["variant"].default, "sparse_forbid")
        sig_fb_rad = inspect.signature(eg.ForbidOverlapRemoval.from_radii)
        self.assertEqual(sig_fb_rad.parameters["variant"].default, "sparse_forbid")

        n = 12
        rng = np.random.default_rng(77)
        coords = rng.uniform(-0.3, 0.3, size=(n, 2)).astype(np.float64)
        coords[1] = coords[0]  # coincident pair
        hw = np.full(n, 0.5, dtype=np.float64)
        hh = np.full(n, 0.35, dtype=np.float64)

        for dtype_str in ("float64", "float32"):
            for use_radii in (False, True):
                l_run = eg.Layout2d.from_coords(_to_float_array(coords, dtype_str))
                l_step = eg.Layout2d.from_coords(_to_float_array(coords, dtype_str))
                if use_radii:
                    gt1 = eg.GTreeOverlapRemoval.from_radii(hw, seed=42, dtype=dtype_str)
                    gt2 = eg.GTreeOverlapRemoval.from_radii(hw, seed=42, dtype=dtype_str)
                    check_hh = hw
                else:
                    gt1 = eg.GTreeOverlapRemoval(hw, hh, seed=42, dtype=dtype_str)
                    gt2 = eg.GTreeOverlapRemoval(hw, hh, seed=42, dtype=dtype_str)
                    check_hh = hh
                self.assertIn("GTreeOverlapRemoval", repr(gt1))
                gt1.run(l_run)
                gt2.step(l_step)
                gt2.run(l_step)
                np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))
                self.assertEqual(gt1.last_overlap_count, 0)
                self.assertEqual(len(eg.rectangle_overlaps(l_run, hw, check_hh)), 0)

            for variant in ("forbid", "forbid_prime", "sparse_forbid"):
                l_run = eg.Layout2d.from_coords(_to_float_array(coords, dtype_str))
                l_step = eg.Layout2d.from_coords(_to_float_array(coords, dtype_str))
                fb1 = eg.ForbidOverlapRemoval(
                    hw, hh, variant=variant, max_scale_iters=25, sgd_iterations=40, seed=42, dtype=dtype_str
                )
                fb2 = eg.ForbidOverlapRemoval(
                    hw, hh, variant=variant, max_scale_iters=25, sgd_iterations=40, seed=42, dtype=dtype_str
                )
                self.assertIn("ForbidOverlapRemoval", repr(fb1))
                fb1.run(l_run)
                fb2.step(l_step)
                fb2.run(l_step)
                np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))
                self.assertEqual(fb1.last_overlap_count, 0)
                self.assertEqual(len(eg.rectangle_overlaps(l_run, hw, hh)), 0)

            fb_rad = eg.ForbidOverlapRemoval.from_radii(hw, variant="forbid", seed=42, dtype=dtype_str)
            l_rad = eg.Layout2d.from_coords(_to_float_array(coords, dtype_str))
            fb_rad.run(l_rad)
            self.assertEqual(fb_rad.last_overlap_count, 0)

        # Verify N=80 > sparse_k=30 default behavior matches sparse_forbid and diverges from forbid
        coords80 = np.random.default_rng(123).uniform(-0.3, 0.3, size=(80, 2)).astype(np.float64)
        hw80 = np.full(80, 0.25, dtype=np.float64)
        hh80 = np.full(80, 0.25, dtype=np.float64)
        for dtype_str in ("float64", "float32"):
            c80 = _to_float_array(coords80, dtype_str)
            l_def = eg.Layout2d.from_coords(c80)
            l_sp = eg.Layout2d.from_coords(c80)
            l_fb = eg.Layout2d.from_coords(c80)
            eg.ForbidOverlapRemoval(hw80, hh80, dtype=dtype_str).step(l_def)
            eg.ForbidOverlapRemoval(hw80, hh80, variant="sparse_forbid", dtype=dtype_str).step(l_sp)
            eg.ForbidOverlapRemoval(hw80, hh80, variant="forbid", dtype=dtype_str).step(l_fb)
            np.testing.assert_array_equal(l_def.coords(copy=True), l_sp.coords(copy=True))
            self.assertFalse(np.allclose(l_def.coords(copy=True), l_fb.coords(copy=True)))

            lr_def = eg.Layout2d.from_coords(c80)
            lr_sp = eg.Layout2d.from_coords(c80)
            lr_fb = eg.Layout2d.from_coords(c80)
            eg.ForbidOverlapRemoval.from_radii(hw80, dtype=dtype_str).step(lr_def)
            eg.ForbidOverlapRemoval.from_radii(hw80, variant="sparse_forbid", dtype=dtype_str).step(lr_sp)
            eg.ForbidOverlapRemoval.from_radii(hw80, variant="forbid", dtype=dtype_str).step(lr_fb)
            np.testing.assert_array_equal(lr_def.coords(copy=True), lr_sp.coords(copy=True))
            self.assertFalse(np.allclose(lr_def.coords(copy=True), lr_fb.coords(copy=True)))

        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.GTreeOverlapRemoval([1.0, 1.0], [1.0])
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.GTreeOverlapRemoval.from_radii([-1.0])
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.GTreeOverlapRemoval([1.0], [1.0], compact_t_max=1.0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.ForbidOverlapRemoval([1.0], [1.0], variant="unknown")
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.ForbidOverlapRemoval([1.0], [1.0], scale_step=0.0)
        )

    def test_force_solvers_gil_release_concurrent(self) -> None:
        g = _make_grid(4, 4)
        base = _seeded_coords(g.num_nodes, seed=99, scale=1.0)

        def worker(seed_val: int) -> bool:
            ll = eg.LinLogSolver(g, iterations=8, seed=seed_val)
            lay = eg.Layout2d.from_coords(base)
            ll.run(lay)
            return lay.is_all_finite()

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(worker, [1, 2]))
        self.assertEqual(results, [True, True])


if __name__ == "__main__":
    unittest.main()
