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


class TestV6Wave1Baselines(unittest.TestCase):
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
        fn: Callable[[], Any],
    ) -> None:
        with self.assertRaises(expected) as ctx:
            fn()
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

    def test_w1_exports_and_docstrings(self) -> None:
        w1_symbols = [
            "BackboneSolver",
            "ClusterRefineSolver",
            "ForbidOverlapRemoval",
            "FruchtermanReingoldSolver",
            "GTreeOverlapRemoval",
            "GroupInABoxSolver",
            "LinLogSolver",
            "calinski_harabasz_score",
            "class_density_measure",
            "cq_hac",
            "cq_kmeans",
            "davies_bouldin_score",
            "distance_consistency",
            "distribution_consistency",
            "extract_quadrilateral_backbone",
            "leiden_communities",
            "neighborhood_hit",
            "nor_metrics",
            "pairwise_distance_overlap_error",
            "partition_agreement",
            "refine_cluster_layout",
            "sepme_gab",
            "sepme_gong",
            "sepme_knng",
            "sepme_rng",
            "silhouette_score",
            "sprawlter_nn",
            "squarified_treemap",
        ]
        for sym in w1_symbols:
            self.assertIn(sym, eg.__all__, f"{sym} missing from __all__")
            obj = getattr(eg, sym)
            self.assertTrue((obj.__doc__ or "").strip(), f"{sym} missing docstring")

    def test_w1_1_leiden_communities(self) -> None:
        sig = inspect.signature(eg.leiden_communities)
        self.assertEqual(sig.parameters["objective"].default, "modularity")
        self.assertEqual(sig.parameters["resolution"].default, 1.0)
        self.assertEqual(sig.parameters["seed"].default, 42)

        barbell = _make_barbell(6, bridge_len=1)
        res_mod = eg.leiden_communities(barbell, objective="modularity", seed=42)
        res_mod2 = eg.leiden_communities(barbell, objective="mod", seed=42)
        labels = res_mod["labels"]
        self.assertEqual(labels.dtype, np.uint32)
        self.assertEqual(labels.shape, (12,))
        np.testing.assert_array_equal(labels, res_mod2["labels"])
        self.assertEqual(res_mod["num_communities"], 2)
        self.assertGreater(float(res_mod["modularity"]), 0.35)
        self.assertAlmostEqual(float(res_mod["quality"]), float(res_mod["modularity"]), places=12)
        self.assertGreaterEqual(int(res_mod["passes"]), 1)

        res_cpm = eg.leiden_communities(barbell, objective="cpm", resolution=0.3, seed=42)
        self.assertEqual(res_cpm["labels"].dtype, np.uint32)
        self.assertGreaterEqual(int(res_cpm["num_communities"]), 2)

        # Degenerate empty & edgeless graphs
        empty = eg.CsrGraph.from_edges(0, [], directed=False)
        res_empty = eg.leiden_communities(empty)
        self.assertEqual(res_empty["num_communities"], 0)
        self.assertEqual(len(res_empty["labels"]), 0)

        # Validation guards
        directed = eg.CsrGraph.from_edges(3, [(0, 1), (1, 2)], directed=True)
        self.assert_no_panic_raises((ValueError,), lambda: eg.leiden_communities(directed))
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.leiden_communities(barbell, objective="unknown")
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.leiden_communities(barbell, max_passes=0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.leiden_communities(barbell, randomness=0.0)
        )
        self.assert_no_panic_raises(
            (TypeError,), lambda: eg.leiden_communities(barbell, seed=True)
        )

    def test_w1_2_linlog_and_fr_solvers(self) -> None:
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

    def test_w1_2_force_atlas2_prevent_overlap(self) -> None:
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

    def test_w1_3_quadrilateral_backbone(self) -> None:
        g = _make_barbell(5, bridge_len=1)
        bb = eg.extract_quadrilateral_backbone(g, retain_ratio=0.3)
        self.assertEqual(bb["umst_edges"].dtype, np.uint32)
        self.assertEqual(bb["backbone_edges"].dtype, np.uint32)
        self.assertEqual(bb["umst_edges"].ndim, 2)
        self.assertEqual(bb["backbone_edges"].ndim, 2)
        self.assertEqual(bb["simmelian_weights"].dtype, np.float64)
        self.assertEqual(bb["edge_weights"].dtype, np.float64)
        self.assertIsInstance(bb["backbone_csr"], eg.CsrGraph)
        self.assertGreaterEqual(len(bb["backbone_edges"]), len(bb["umst_edges"]))

        base = _seeded_coords(g.num_nodes, seed=21, scale=1.0)
        for dtype_str in ("float64", "float32"):
            for mode in ("stress", "sgd"):
                l_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
                l_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
                s1 = eg.BackboneSolver(
                    g,
                    retain_ratio=0.3,
                    layout_mode=mode,
                    iterations=15,
                    seed=9,
                    dtype=dtype_str,
                )
                s2 = eg.BackboneSolver(
                    g,
                    retain_ratio=0.3,
                    layout_mode=mode,
                    iterations=15,
                    seed=9,
                    dtype=dtype_str,
                )
                self.assertIn("BackboneSolver", repr(s1))
                res_dict = s1.backbone_result()
                self.assertEqual(len(res_dict["backbone_edges"]), len(bb["backbone_edges"]))
                s1.run(l_run)
                s2.step(l_step)
                s2.run(l_step)
                np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))

        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.extract_quadrilateral_backbone(g, retain_ratio=1.5)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.BackboneSolver(g, layout_mode="invalid")
        )

    def test_w1_4_cluster_refine_and_group_in_a_box(self) -> None:
        # Two clusters with two swapped points so ClusterRefine moves them back
        pts = np.array(
            [
                [0.0, 0.0],
                [0.2, 0.1],
                [0.1, -0.2],
                [10.0, 0.0],  # label 0 but placed in cluster 1
                [10.0, 0.2],
                [10.2, 0.1],
                [10.1, -0.2],
                [0.0, 0.2],   # label 1 but placed in cluster 0
            ],
            dtype=np.float64,
        )
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint32)
        g_small = _make_grid(2, 4)

        for method in ("kmeans", "hac"):
            for dtype_str in ("float64", "float32"):
                l_run = eg.Layout2d.from_coords(_to_float_array(pts, dtype_str))
                l_step = eg.Layout2d.from_coords(_to_float_array(pts, dtype_str))
                s1 = eg.ClusterRefineSolver(labels, method=method, max_rounds=5, seed=42, dtype=dtype_str)
                s2 = eg.ClusterRefineSolver.from_graph(
                    g_small, labels, method=method, max_rounds=5, seed=42, dtype=dtype_str
                )
                self.assertEqual(s1.num_clusters, 2)
                self.assertEqual(s1.num_nodes, 8)
                self.assertIn("ClusterRefineSolver", repr(s1))

                dsc_before = eg.distance_consistency(l_run, labels)
                s1.run(l_run)
                dsc_after = eg.distance_consistency(l_run, labels)
                self.assertGreater(dsc_after, dsc_before)

                s2.step(l_step)
                self.assertGreaterEqual(s2.last_moved_count, 0)
                s2.run(l_step)
                self.assertTrue(s2.is_finished)

                l_fn = eg.Layout2d.from_coords(_to_float_array(pts, dtype_str))
                moved = eg.refine_cluster_layout(l_fn, labels, method=method, max_rounds=5, seed=42)
                self.assertGreater(moved, 0)
                self.assertGreater(eg.distance_consistency(l_fn, labels), dsc_before)

        # Squarified treemap
        rects = eg.squarified_treemap([6.0, 3.0, 1.0], width=100.0, height=50.0, x=10.0, y=20.0)
        self.assertEqual(rects.shape, (3, 4))
        self.assertEqual(rects.dtype, np.float64)
        areas = (rects[:, 1] - rects[:, 0]) * (rects[:, 3] - rects[:, 2])
        np.testing.assert_allclose(areas, np.array([3000.0, 1500.0, 500.0]), atol=1e-9)

        # GroupInABoxSolver
        barbell = _make_barbell(4, bridge_len=1)
        gib_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint32)
        base = _seeded_coords(barbell.num_nodes, seed=31, scale=1.0)
        for box_mode in ("treemap", "force_directed"):
            for intra_mode in ("fr", "sgd_full", "yifan_hu"):
                for dtype_str in ("float64", "float32"):
                    l_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
                    l_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
                    cw = [2.0, 1.0] if box_mode == "treemap" else None
                    g1 = eg.GroupInABoxSolver(
                        barbell,
                        gib_labels,
                        box_mode=box_mode,
                        intra_mode=intra_mode,
                        intra_iterations=8,
                        fd_iterations=8,
                        cluster_weights=cw,
                        seed=42,
                        dtype=dtype_str,
                    )
                    g2 = eg.GroupInABoxSolver(
                        barbell,
                        gib_labels,
                        box_mode=box_mode,
                        intra_mode=intra_mode,
                        intra_iterations=8,
                        fd_iterations=8,
                        cluster_weights=cw,
                        seed=42,
                        dtype=dtype_str,
                    )
                    boxes = g1.boxes()
                    self.assertEqual(len(boxes), 2)
                    self.assertEqual(
                        set(boxes[0].keys()),
                        {"cluster_id", "x_min", "x_max", "y_min", "y_max", "node_count", "weight"},
                    )
                    self.assertIn("GroupInABoxSolver", repr(g1))
                    g1.run(l_run)
                    g2.step(l_step)
                    g2.run(l_step)
                    np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))

        self.assert_no_panic_raises((ValueError,), lambda: eg.squarified_treemap([]))
        self.assert_no_panic_raises((ValueError,), lambda: eg.squarified_treemap([1.0, -1.0]))
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.GroupInABoxSolver(barbell, gib_labels, box_mode="bad")
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.GroupInABoxSolver(barbell, gib_labels, intra_mode="bad")
        )

    def test_w1_5_gtree_and_forbid_overlap_removal(self) -> None:
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

    def test_w1_gil_release_concurrent(self) -> None:
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
