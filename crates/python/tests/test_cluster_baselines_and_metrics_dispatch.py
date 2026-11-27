from __future__ import annotations

import unittest
from typing import Any

import egraph as eg
import numpy as np


def _make_grid(rows: int, cols: int) -> tuple[eg.CsrGraph, eg.Layout2d]:
    edges: list[tuple[int, int]] = []
    coords: list[list[float]] = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            coords.append([float(c), float(r)])
            if c + 1 < cols:
                edges.append((idx, idx + 1))
            if r + 1 < rows:
                edges.append((idx, idx + cols))
    g = eg.CsrGraph.from_edges(rows * cols, edges, directed=False, sanitize=False)
    return g, eg.Layout2d.from_coords(np.array(coords, dtype=np.float64))


class TestClusterBaselinesAndMetricsDispatch(unittest.TestCase):
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
        fn: Any,
    ) -> None:
        with self.assertRaises(expected) as ctx:
            fn()
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

    def test_selective_evaluate_metrics(self) -> None:
        g, layout = _make_grid(4, 4)

        empty_report = eg.evaluate_metrics(g, layout, [])
        self.assertEqual(len(empty_report), 0)
        self.assertEqual(empty_report.to_dict(), {})

        subset = eg.evaluate_metrics(g, layout, ["sns", "np2", "drng", "dgg", "np2"])
        self.assertEqual(
            set(subset.keys()),
            {
                "scale_normalized_stress",
                "neighborhood_preservation_2hop",
                "relative_neighborhood_preservation",
                "gabriel_graph_preservation",
            },
        )
        self.assertAlmostEqual(
            subset["scale_normalized_stress"],
            eg.scale_normalized_stress(g, layout),
            places=12,
        )
        self.assertAlmostEqual(
            subset["neighborhood_preservation_2hop"],
            eg.neighborhood_preservation_2hop(g, layout),
            places=12,
        )
        self.assertAlmostEqual(
            subset["relative_neighborhood_preservation"],
            eg.drng_preservation(g, layout),
            places=12,
        )
        self.assertAlmostEqual(
            subset["gabriel_graph_preservation"],
            eg.dgg_preservation(g, layout),
            places=12,
        )

        # Evaluate all 16 targets using mixed alias conventions.
        all_targets = [
            "ScaleNormalizedStress",
            "sampled_sns",
            "crossing-number",
            "sweep_crossing_angle",
            "sampled_crossing_number",
            "np1",
            "neighborhood_preservation_2hop",
            "sampled-np1",
            "sampled_np2",
            "gabriel_graph_property",
            "node_resolution",
            "drng_preservation",
            "dgg_preservation",
            "aspect_ratio",
            "angular_resolution",
            "ideal_edge_lengths",
        ]
        full = eg.evaluate_metrics(g, layout, all_targets, num_pivots=6, num_samples=200, seed=9)
        self.assertEqual(len(full), 16)
        self.assertAlmostEqual(
            full["sampled_scale_normalized_stress"],
            eg.sampled_scale_normalized_stress(g, layout, num_pivots=6, seed=9),
            places=12,
        )
        self.assertEqual(full["crossing_number"], eg.sweep_crossing_number(g, layout))
        self.assertAlmostEqual(full["crossing_angle"], eg.sweep_crossing_angle(g, layout), places=12)
        self.assertAlmostEqual(
            full["sampled_crossing_number"],
            eg.sampled_crossing_number(g, layout, num_samples=200, seed=9),
            places=12,
        )
        self.assertAlmostEqual(
            full["neighborhood_preservation_1hop"],
            eg.neighborhood_preservation_1hop(g, layout),
            places=12,
        )
        self.assertAlmostEqual(
            full["sampled_np1"],
            eg.sampled_np1(g, layout, num_samples=200, seed=9),
            places=12,
        )
        self.assertAlmostEqual(
            full["sampled_np2"],
            eg.sampled_np2(g, layout, num_samples=200, seed=9),
            places=12,
        )
        self.assertAlmostEqual(
            full["gabriel_graph_property"],
            eg.gabriel_graph_property(g, layout),
            places=12,
        )
        self.assertAlmostEqual(full["node_resolution"], eg.node_resolution(layout), places=12)
        self.assertAlmostEqual(full["aspect_ratio"], eg.aspect_ratio(layout), places=12)
        self.assertAlmostEqual(
            full["angular_resolution"], eg.angular_resolution(g, layout), places=12
        )
        self.assertAlmostEqual(
            full["ideal_edge_lengths"], eg.ideal_edge_lengths(g, layout), places=12
        )

        # Invalid sampled parameter only errors when the corresponding target is requested.
        ok = eg.evaluate_metrics(g, layout, ["aspect_ratio"], num_pivots=0)
        self.assertIn("aspect_ratio", ok)
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.evaluate_metrics(g, layout, ["sampled_sns"], num_pivots=0),
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.evaluate_metrics(g, layout, ["sampled_np1"], num_samples=0),
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.evaluate_metrics(g, layout, ["sampled_np2"], num_samples=0),
        )

    def test_cluster_baselines_and_nor(self) -> None:
        # Well-separated 2-cluster 2D layout (4 points per cluster)
        coords = np.array(
            [
                [0.0, 0.0],
                [0.2, 0.0],
                [0.0, 0.2],
                [0.2, 0.2],
                [10.0, 0.0],
                [10.2, 0.0],
                [10.0, 0.2],
                [10.2, 0.2],
            ],
            dtype=np.float64,
        )
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint32)
        pred_perfect = np.array([10, 10, 10, 10, 20, 20, 20, 20], dtype=np.uint32)
        pred_partial = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint32)

        pa_perf = eg.partition_agreement(labels, pred_perfect)
        self.assertEqual(
            set(pa_perf.keys()),
            {"ari", "ami", "homogeneity", "completeness", "v_measure", "fmi"},
        )
        for key in ("ari", "ami", "homogeneity", "completeness", "v_measure", "fmi"):
            self.assertAlmostEqual(pa_perf[key], 1.0, places=12)

        # Cross-check ARI / FMI / Homogeneity / Completeness / V-measure on partial agreement
        pa_part = eg.partition_agreement(labels, pred_partial)
        self.assertLess(pa_part["ari"], 0.6)
        self.assertGreater(pa_part["fmi"], 0.4)

        for dtype in (np.float64, np.float32):
            lay = eg.Layout2d.from_coords(coords.astype(dtype))
            km = eg.cq_kmeans(lay, labels, n_init=5, max_iterations=100, seed=42)
            hac = eg.cq_hac(lay, labels)
            self.assertAlmostEqual(km["ari"], 1.0, places=12)
            self.assertAlmostEqual(km["ami"], 1.0, places=12)
            self.assertAlmostEqual(hac["homogeneity"], 1.0, places=12)
            self.assertAlmostEqual(hac["ami"], 1.0, places=12)

            self.assertAlmostEqual(eg.distance_consistency(lay, labels), 1.0, places=12)
            self.assertAlmostEqual(eg.distribution_consistency(lay, labels, grid_size=20), 1.0, places=12)
            cdm, hdm = eg.class_density_measure(lay, labels, grid_size=32)
            self.assertGreater(cdm, 0.9)
            self.assertAlmostEqual(hdm, 1.0, places=12)

            self.assertAlmostEqual(eg.sepme_knng(lay, labels, k=2), 1.0, places=12)
            self.assertGreater(eg.sepme_rng(lay, labels), 0.7)
            self.assertGreater(eg.sepme_gab(lay, labels), 0.5)
            self.assertGreater(eg.sepme_gong(lay, labels, gamma=0.35), 0.7)
            mixed_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint32)
            self.assertGreater(eg.sepme_rng(lay, labels), eg.sepme_rng(lay, mixed_labels))
            self.assertGreater(eg.sepme_gab(lay, labels), eg.sepme_gab(lay, mixed_labels))
            self.assertGreater(
                eg.sepme_gong(lay, labels, gamma=0.35),
                eg.sepme_gong(lay, mixed_labels, gamma=0.35),
            )
            self.assertAlmostEqual(eg.neighborhood_hit(lay, labels, k=3), 1.0, places=12)

            sil = eg.silhouette_score(lay, labels)
            ch = eg.calinski_harabasz_score(lay, labels)
            db = eg.davies_bouldin_score(lay, labels)
            self.assertGreater(sil, 0.95)
            self.assertGreater(ch, 1000.0)
            self.assertLess(db, 0.1)

            pdoe = eg.pairwise_distance_overlap_error(lay, labels, num_bins=50)
            self.assertGreaterEqual(pdoe, 0.0)
            self.assertLessEqual(pdoe, 1.0)

            hw = np.full(8, 0.25, dtype=np.float64)
            hh = np.full(8, 0.25, dtype=np.float64)
            sp_rect = eg.sprawlter_nn(lay, hw, hh)
            sp_rad = eg.sprawlter_nn(lay, hw)
            self.assertAlmostEqual(sp_rect, sp_rad, places=12)
            self.assertGreater(sp_rect, 0.0)

        # NOR metrics across (f64, f64), (f32, f32), (f64, f32), (f32, f64)
        orig_pts = np.array([[0.0, 0.0], [0.2, 0.0], [2.0, 0.0], [2.0, 2.0]], dtype=np.float64)
        adj_pts = np.array([[0.0, 0.0], [1.2, 0.0], [2.5, 0.0], [2.5, 2.0]], dtype=np.float64)
        hw4 = [0.5, 0.5, 0.5, 0.5]
        hh4 = [0.5, 0.5, 0.5, 0.5]
        g4 = eg.CsrGraph.from_edges(4, [(0, 1), (1, 2), (2, 3)], directed=False, sanitize=True)

        l_orig_64 = eg.Layout2d.from_coords(orig_pts)
        l_adj_64 = eg.Layout2d.from_coords(adj_pts)
        l_orig_32 = eg.Layout2d.from_coords(orig_pts.astype(np.float32))
        l_adj_32 = eg.Layout2d.from_coords(adj_pts.astype(np.float32))

        nor_no_g = eg.nor_metrics(l_orig_64, l_adj_64, hw4, hh4)
        self.assertEqual(
            set(nor_no_g.keys()),
            {"oo_nni", "sp_ch_a", "gs_bb_iar", "nm_dm_imse", "el_rsd"},
        )
        self.assertEqual(nor_no_g["oo_nni"], 0.0)
        self.assertEqual(nor_no_g["el_rsd"], 0.0)

        nor_g = eg.nor_metrics(l_orig_64, l_adj_64, hw4, hh4, graph=g4)
        self.assertGreater(nor_g["el_rsd"], 0.0)
        self.assertGreater(nor_g["sp_ch_a"], 0.0)
        self.assertGreaterEqual(nor_g["gs_bb_iar"], 1.0)
        self.assertGreater(nor_g["nm_dm_imse"], 0.0)

        for lo, la in ((l_orig_32, l_adj_32), (l_orig_64, l_adj_32), (l_orig_32, l_adj_64)):
            nor_mix = eg.nor_metrics(lo, la, hw4, hh4, graph=g4)
            self.assertAlmostEqual(nor_mix["oo_nni"], nor_g["oo_nni"], places=6)
            self.assertAlmostEqual(nor_mix["sp_ch_a"], nor_g["sp_ch_a"], places=5)

        # Degenerate empty / singleton boundaries (zero PanicException)
        empty_lay = eg.Layout2d.zeros(0)
        self.assertEqual(eg.distance_consistency(empty_lay, []), 0.0)
        self.assertEqual(eg.sepme_rng(empty_lay, []), 0.0)
        single_lay = eg.Layout2d.zeros(1)
        self.assertEqual(eg.sepme_gab(single_lay, [0]), 1.0)
        self.assertEqual(eg.silhouette_score(single_lay, [0]), 0.0)

        # Validation errors
        lay64 = eg.Layout2d.from_coords(coords)
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.partition_agreement([0, 1], [0])
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.sepme_knng(lay64, labels, k=0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.sepme_gong(lay64, labels, gamma=1.5)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.distribution_consistency(lay64, labels, grid_size=0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.sprawlter_nn(lay64, [-1.0] * 8)
        )

    def test_cluster_baselines_evaluate_metrics_targets(self) -> None:
        coords = np.array(
            [
                [0.0, 0.0],
                [0.2, 0.0],
                [0.0, 0.2],
                [0.2, 0.2],
                [8.0, 0.0],
                [8.2, 0.0],
                [8.0, 0.2],
                [8.2, 0.2],
            ],
            dtype=np.float64,
        )
        g = eg.CsrGraph.from_edges(
            8, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)], directed=False
        )
        lay = eg.Layout2d.from_coords(coords)
        labels = [0, 0, 0, 0, 1, 1, 1, 1]

        all_cluster_baseline_targets = [
            "dsc",
            "dc",
            "cdm",
            "hdm",
            "sepme_knng",
            "sepme_rng",
            "sepme_gab",
            "sepme_gong",
            "neighborhood_hit",
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
            "cq_kmeans_ari",
            "cq_kmeans_ami",
            "cq_hac_hom",
            "cq_hac_ami",
            "pairwise_distance_overlap_error",
            "sprawlter_nn",
        ]
        res = eg.evaluate_metrics(
            g,
            lay,
            all_cluster_baseline_targets,
            labels=labels,
            sepme_knng_k=2,
            neighborhood_hit_k=3,
            dc_grid_size=20,
            cdm_grid_size=32,
            cdm_bandwidth=0.05,
            num_bins=50,
            gong_gamma=0.35,
            kmeans_n_init=5,
            kmeans_max_iterations=100,
            sprawlter_node_radius=0.5,
            seed=42,
        )
        expected_canonical_keys = {
            "distance_consistency",
            "distribution_consistency",
            "class_density_measure",
            "histogram_density_measure",
            "sepme_knng",
            "sepme_rng",
            "sepme_gab",
            "sepme_gong",
            "neighborhood_hit",
            "silhouette_score",
            "calinski_harabasz_score",
            "davies_bouldin_score",
            "cq_kmeans_ari",
            "cq_kmeans_ami",
            "cq_hac_hom",
            "cq_hac_ami",
            "pairwise_distance_overlap_error",
            "sprawlter_nn",
        }
        self.assertEqual(set(res.keys()), expected_canonical_keys)
        self.assertAlmostEqual(
            res["distance_consistency"], eg.distance_consistency(lay, labels), places=12
        )
        self.assertAlmostEqual(
            res["neighborhood_hit"], eg.neighborhood_hit(lay, labels, k=3), places=12
        )
        self.assertAlmostEqual(
            res["sprawlter_nn"], eg.sprawlter_nn(lay, [0.5] * 8), places=12
        )

        # Alias checks for sepme_knng2 / sepme-knng2 / sepme-knng
        for alias in ("sepme_knng", "sepme-knng", "sepme_knng2", "sepme-knng2"):
            r_alias = eg.evaluate_metrics(g, lay, [alias], labels=labels, sepme_knng_k=2)
            self.assertAlmostEqual(r_alias["sepme_knng"], res["sepme_knng"], places=12)

        # Missing labels error for community-dependent cluster baseline targets
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.evaluate_metrics(g, lay, ["dsc"])
        )
        # Lazy validation ignores invalid baseline kwargs when unrelated targets are requested
        ok = eg.evaluate_metrics(g, lay, ["aspect_ratio"], sepme_knng_k=0)
        self.assertIn("aspect_ratio", ok)
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.evaluate_metrics(
                g, lay, ["sepme_knng"], labels=labels, sepme_knng_k=0
            ),
        )


if __name__ == "__main__":
    unittest.main()
