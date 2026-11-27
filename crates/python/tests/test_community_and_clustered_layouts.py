from __future__ import annotations

import inspect
import unittest
from collections.abc import Callable
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


class TestCommunityAndClusteredLayouts(unittest.TestCase):
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
        fn: Callable[[], Any],
    ) -> None:
        with self.assertRaises(expected) as ctx:
            fn()
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

    def test_community_cluster_and_overlap_exports(self) -> None:
        exported_symbols = [
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
        for sym in exported_symbols:
            self.assertIn(sym, eg.__all__, f"{sym} missing from __all__")
            obj = getattr(eg, sym)
            self.assertTrue((obj.__doc__ or "").strip(), f"{sym} missing docstring")

    def test_leiden_communities(self) -> None:
        sig = inspect.signature(eg.leiden_communities)
        self.assertEqual(sig.parameters["objective"].default, "modularity")
        self.assertEqual(sig.parameters["resolution"].default, 1.0)
        self.assertEqual(sig.parameters["seed"].default, 42)

        barbell = _make_barbell(6, bridge_len=1)
        res_mod = eg.leiden_communities(barbell, objective="modularity", seed=42)
        res_mod2 = eg.leiden_communities(barbell, objective="mod", seed=42)
        labels = res_mod.labels
        self.assertEqual(labels.dtype, np.uint32)
        self.assertEqual(labels.shape, (12,))
        np.testing.assert_array_equal(labels, res_mod2.labels)
        self.assertEqual(res_mod.num_communities, 2)
        self.assertGreater(float(res_mod.modularity), 0.35)
        self.assertAlmostEqual(float(res_mod.quality), float(res_mod.modularity), places=12)
        self.assertGreaterEqual(int(res_mod.passes), 1)

        res_cpm = eg.leiden_communities(barbell, objective="cpm", resolution=0.3, seed=42)
        self.assertEqual(res_cpm.labels.dtype, np.uint32)
        self.assertGreaterEqual(int(res_cpm.num_communities), 2)

        # CPM modularity field must reflect standard gamma=1.0 modularity, never resolution-inflated
        karate = eg.load_dataset("karate")
        res_cpm_k = eg.leiden_communities(karate, objective="cpm", resolution=0.1, seed=42)
        labels_k = res_cpm_k.labels
        adj_ind = karate.indptr()
        adj_idx = karate.indices()
        two_m = float(adj_ind[-1])
        q1_expected = 0.0
        for c in np.unique(labels_k):
            nodes_c = np.where(labels_k == c)[0]
            deg_sum = sum(int(adj_ind[u + 1] - adj_ind[u]) for u in nodes_c)
            c_set = set(int(u) for u in nodes_c)
            in_edges = sum(
                1 for u in nodes_c for v in adj_idx[adj_ind[u] : adj_ind[u + 1]] if int(v) in c_set
            )
            q1_expected += in_edges / two_m - (deg_sum / two_m) ** 2
        self.assertAlmostEqual(float(res_cpm_k.modularity), q1_expected, places=12)

        # Degenerate empty & edgeless graphs
        empty = eg.CsrGraph.from_edges(0, [], directed=False)
        res_empty = eg.leiden_communities(empty)
        self.assertEqual(res_empty.num_communities, 0)
        self.assertEqual(len(res_empty.labels), 0)

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

    def test_quadrilateral_backbone(self) -> None:
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
        dg = eg.CsrGraph.from_edges(4, [(0, 1), (1, 2), (2, 3), (0, 3)], directed=True)
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.extract_quadrilateral_backbone(dg)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.BackboneSolver(dg)
        )

    def test_cluster_refine_and_group_in_a_box(self) -> None:
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
                s1 = eg.ClusterRefineSolver(labels, method=method, max_iterations=5, seed=42, dtype=dtype_str)
                s2 = eg.ClusterRefineSolver.from_graph(
                    g_small, labels, method=method, max_iterations=5, seed=42, dtype=dtype_str
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
                moved = eg.refine_cluster_layout(l_fn, labels, method=method, max_iterations=5, seed=42)
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


if __name__ == "__main__":
    unittest.main()
