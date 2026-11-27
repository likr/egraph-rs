from __future__ import annotations

import math
import unittest
from collections.abc import Callable
from typing import Any

import egraph as eg
import numpy as np


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


class TestManifoldCrossingsAndMetricsReport(unittest.TestCase):
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
        fn: Callable[[], Any],
    ) -> None:
        with self.assertRaises(expected) as ctx:
            fn()
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

    def test_manifold_lorentz_scale_aspect_torus_and_crossings(self) -> None:
        # Hyperbolic Lorentz + conversion to Poincare + Lorentz/Poincare functions
        mlor = eg.ManifoldLayout2d.hyperbolic_lorentz(4, dtype="float64")
        self.assertEqual(mlor.kind, "hyperbolic_lorentz")
        mlor.set_point(0, 0.0, 0.0)
        mlor.set_point(1, 0.4, 0.2)
        view = mlor.coords(copy=False)
        self.assertEqual(view.shape, (4, 2))
        mpoin = mlor.to_poincare_layout()
        self.assertEqual(mpoin.kind, "hyperbolic")

        px, py = eg.lorentz_to_poincare_point(0.4, 0.2)
        self.assertLess(math.hypot(px, py), 1.0)
        self.assertGreater(eg.lorentz_distance(0.0, 0.0, 0.4, 0.2), 0.0)
        ex, ey = eg.lorentz_exp_map(0.0, 0.0, 0.3, -0.2)
        lx, ly = eg.lorentz_log_map(0.0, 0.0, ex, ey)
        self.assertAlmostEqual(lx, 0.3, places=6)
        self.assertAlmostEqual(ly, -0.2, places=6)
        mx, my = eg.lorentz_einstein_midpoint([0.1, -0.1], [0.2, -0.2])
        self.assertAlmostEqual(math.hypot(mx, my), 0.0, places=6)
        pmx, pmy = eg.poincare_einstein_midpoint([0.1, -0.1], [0.2, -0.2])
        self.assertAlmostEqual(math.hypot(pmx, pmy), 0.0, places=6)

        # Scale, aspect, metric(), rescale_geodesic, set_torus_extents
        m_tor = eg.ManifoldLayout2d.torus_2d(4)
        m_tor.set_scale(2.0)
        m_tor.set_aspect(1.5)
        self.assertAlmostEqual(m_tor.scale, 2.0)
        self.assertAlmostEqual(m_tor.aspect, 1.5)
        self.assertEqual(m_tor.metric(), (2.0, 1.5))
        m_tor.set_torus_extents(1.0, 1.0)
        m_tor.rescale_geodesic(3.0)
        self.assertAlmostEqual(m_tor.scale, 3.0)

        # Torus spectral init, auto pan, segments & extended crossing fields
        g_t = eg.generate_torus_grid(3, 3)
        m_t = eg.ManifoldLayout2d.torus_spectral_phase_init(g_t, seed=42)
        m_pan = m_t.torus_auto_pan(g_t)
        self.assertEqual(len(m_pan), 9)
        segs = eg.torus_edge_segments(0.9, 0.1, 0.1, 0.9)
        self.assertEqual(segs.ndim, 2)
        self.assertEqual(segs.shape[1], 4)
        lsegs = eg.torus_layout_edge_segments(g_t, m_pan)
        self.assertGreaterEqual(len(lsegs), g_t.num_edges)

        g_x = eg.CsrGraph.from_edges(4, [(0, 2), (1, 3)], directed=False)
        m_tx = eg.ManifoldLayout2d.from_coords("torus", [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]])
        self.assertEqual(eg.torus_crossing_number(g_x, m_tx), 1)
        x_tor = eg.crossing_edges_torus(g_x, m_tx)
        self.assertEqual(len(x_tor), 1)
        for key in ("edge_a", "edge_b", "u1", "v1", "u2", "v2", "angle_rad", "cos_sq"):
            self.assertIn(key, x_tor[0])

        # Spherical & hyperbolic crossing metrics
        m_sx = eg.ManifoldLayout2d.from_coords(
            "spherical",
            [[0.0, 1.2], [1.0, 1.2], [0.5, 0.5], [0.5, 2.0]],
        )
        self.assertGreaterEqual(eg.spherical_crossing_number(g_x, m_sx), 0)
        self.assertIsInstance(eg.crossing_edges_spherical(g_x, m_sx), list)

        m_hx = eg.ManifoldLayout2d.from_coords(
            "hyperbolic",
            [[-0.4, -0.4], [0.4, -0.4], [0.4, 0.4], [-0.4, 0.4]],
        )
        self.assertEqual(eg.hyperbolic_crossing_number(g_x, m_hx), 1)
        self.assertEqual(len(eg.crossing_edges_hyperbolic(g_x, m_hx)), 1)

    def test_all_49_metric_report_targets_details_and_c8_rta(self) -> None:
        g, labels = eg.generate_sbm_graph([4, 4], p_intra=0.9, p_inter=0.1, seed=42)
        l2 = eg.pivot_mds_init(g, num_pivots=6, min_degree=1, seed=42)
        l2_adj = l2.copy()
        eg.SgdSolver.full(g, iterations=10, seed=42).run(l2_adj)
        hw = np.full(8, 0.1, dtype=np.float64)
        hh = np.full(8, 0.1, dtype=np.float64)
        m_tor = eg.ManifoldLayout2d.from_layout("torus", l2_adj)

        all_targets = [
            "scale_normalized_stress",
            "sampled_scale_normalized_stress",
            "crossing_number",
            "crossing_edges",
            "crossing_angle",
            "sampled_crossing_number",
            "crosslessness",
            "sampled_crosslessness",
            "torus_crossing_number",
            "crossing_edges_torus",
            "neighborhood_preservation_1hop",
            "neighborhood_preservation_2hop",
            "sampled_np1",
            "sampled_np2",
            "gabriel_graph_property",
            "node_resolution",
            "relative_neighborhood_preservation",
            "gabriel_graph_preservation",
            "aspect_ratio",
            "angular_resolution",
            "minimum_angle",
            "minimum_angle_a1",
            "layout_degeneracy",
            "ideal_edge_lengths",
            "prism_distortion",
            "adjusted_patch_nig",
            "cluster_area_faithfulness",
            "landmark_stratified_rta",
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
            "nor_metrics",
        ]
        rep = eg.evaluate_metrics(
            g,
            l2_adj,
            all_targets,
            labels=labels,
            num_landmarks=4,
            min_degree=1,
            original_layout=l2,
            sprawlter_half_widths=hw,
            sprawlter_half_heights=hh,
            sepme_knng_k=2,
            neighborhood_hit_k=3,
        )
        self.assertIsInstance(rep, eg.MetricReport)
        self.assertGreaterEqual(len(rep), len(all_targets))
        for key in rep.keys():
            self.assertIn(key, rep)
            self.assertIsNotNone(getattr(rep, key))
        self.assertIsNotNone(rep.prism_sigma_dist)
        self.assertIsNotNone(rep.prism_sigma_disp)
        self.assertIsNotNone(rep.nor_oo_nni)
        self.assertIsNone(rep.geodesic_distortion)

        rep_m = eg.evaluate_metrics(g, m_tor, ["geodesic_distortion", "torus_crossing_number"])
        self.assertIsNotNone(rep_m.geodesic_distortion)
        self.assertAlmostEqual(rep_m["geodesic_distortion"], eg.geodesic_distortion(g, m_tor), places=12)

        # Standalone details & N-D / Manifold metrics
        self.assertIn("excess_duplicate_ratio", eg.layout_degeneracy(g, l2_adj))
        self.assertIn("near_collision_ratio", eg.layout_degeneracy(g, l2_adj))
        self.assertIn("nig", eg.adjusted_patch_nig_detail(l2_adj, labels))
        self.assertTrue(math.isfinite(eg.histogram_density_measure(l2_adj, labels, grid_size=32)))
        self.assertTrue(math.isfinite(eg.sprawlter_nn_radii(l2_adj, np.full(8, 0.1))))
        self.assertIn("normalized_stress", eg.scale_normalized_stress_detail(g, l2_adj))
        self.assertIn("normalized_stress", eg.sampled_scale_normalized_stress_detail(g, l2_adj, num_pivots=4))
        dmat = eg.all_pairs_bfs(g)
        self.assertIn("normalized_stress", eg.scale_normalized_stress_from_distances_detail(l2_adj, dmat))
        self.assertIn("normalized_stress", eg.scale_normalized_stress_from_pairs(l2_adj, [(0, 1, 1.0, 1.0)]))
        self.assertTrue(math.isfinite(eg.ideal_edge_lengths_from_distances(g, l2_adj, dmat)))
        self.assertTrue(math.isfinite(eg.ideal_edge_lengths_scaled(g, l2_adj, 1.0)))
        c_max = eg.max_crossing_pairs(g)
        self.assertGreater(c_max, 0)
        self.assertAlmostEqual(eg.crosslessness_from_count(g, 0.0), 1.0)

        # N-D and Manifold standalone metrics + C8 landmark_stratified_rta_from_landmarks
        lnd = eg.LayoutNd.init(8, 3, seed=7)
        self.assertTrue(math.isfinite(eg.scale_normalized_stress(g, lnd)))
        self.assertTrue(math.isfinite(eg.scale_normalized_stress(g, m_tor)))
        lm_ids = np.array([0, 4], dtype=np.uint32)
        lm_dists_f64 = dmat[:, lm_ids].astype(np.float64)
        lm_dists_f64[0, 1] = float("inf")
        for lay_any in (l2_adj, lnd, m_tor):
            r_c8 = eg.landmark_stratified_rta_from_landmarks(
                lay_any, lm_ids, lm_dists_f64, num_triplets=100, seed=42
            )
            self.assertGreaterEqual(r_c8["rta"], 0.0)

        # HyperbolicLorentz extreme-radius distance, geodesic_distortion, and Poincare boundary projection
        d160 = eg.lorentz_distance(1e160, 0.0, 1.0, 0.0)
        d200 = eg.lorentz_distance(1e200, 0.0, 1.0, 0.0)
        self.assertTrue(math.isfinite(d160) and d160 > 300.0)
        self.assertTrue(math.isfinite(d200) and d200 > 400.0)
        px, py = eg.lorentz_to_poincare_point(1.5e308, 1.5e308)
        self.assertGreater(px, 0.7)
        self.assertGreater(py, 0.7)
        self.assertAlmostEqual(math.hypot(px, py), 1.0 - 1e-5, places=10)

        g_pair = eg.CsrGraph.from_edges(2, [(0, 1)], directed=False)
        ml_extreme = eg.ManifoldLayout2d.from_coords(
            "hyperbolic_lorentz",
            np.array([[1e160, 0.0], [1.0, 0.0]], dtype=np.float64),
        )
        gd_extreme = eg.evaluate_metrics(g_pair, ml_extreme, ["geodesic_distortion"]).geodesic_distortion
        assert gd_extreme is not None
        self.assertTrue(math.isfinite(gd_extreme))
        self.assertAlmostEqual(gd_extreme, d160 - 1.0, places=9)

        ml_max = eg.ManifoldLayout2d.from_coords(
            "hyperbolic_lorentz",
            np.array([[1.5e308, 1.5e308], [-1.5e308, 0.0]], dtype=np.float64),
        )
        poinc_max = ml_max.to_poincare_layout()
        coords_max = poinc_max.coords(copy=True)
        self.assertAlmostEqual(float(np.hypot(coords_max[0, 0], coords_max[0, 1])), 1.0 - 1e-5, places=10)
        self.assertAlmostEqual(float(coords_max[1, 0]), -(1.0 - 1e-5), places=10)

        # Extreme-radius lorentz_einstein_midpoint and rescale_geodesic distance preservation
        em_x, em_y = eg.lorentz_einstein_midpoint([1e160, 0.0], [0.0, 1e160])
        self.assertAlmostEqual(em_x, math.sqrt(0.5), places=12)
        self.assertAlmostEqual(em_y, math.sqrt(0.5), places=12)
        em_max_x, em_max_y = eg.lorentz_einstein_midpoint([1.5e308, 0.0], [1.5e308, 0.0])
        self.assertTrue(math.isfinite(em_max_x) and math.isfinite(em_max_y))
        self.assertAlmostEqual(em_max_x / 1e153, 7.282376575609851, places=10)
        self.assertAlmostEqual(em_max_y / 1e153, 7.282376575609851, places=10)

        ml_off = eg.ManifoldLayout2d.from_coords(
            "hyperbolic_lorentz",
            np.array([[1e160, 0.0], [1e160, 1e155]], dtype=np.float64),
        )
        d_before = ml_off.geodesic_distance(0, 1)
        self.assertAlmostEqual(d_before, 713.8013788281291, places=8)
        ml_off.rescale_geodesic(100.0)
        d_after = ml_off.geodesic_distance(0, 1)
        self.assertAlmostEqual(d_after, d_before, places=6)

        # Extreme-radius lorentz_log_map & rescale_geodesic invariance regression checks
        vx_r, vy_r = eg.lorentz_log_map(1e20, 0.0, 1e41, 0.0)
        self.assertAlmostEqual(math.hypot(vx_r, vy_r), 48.35428695287496, places=10)
        self.assertGreater(vx_r, 48.0)
        for coords, new_scale, expected_d in (
            ([[1.5e308, 1.5e308], [0.0, 0.0]], 10.0, 710.6413945211141),
            ([[1e200, 0.0], [1e200, 1.0]], 3.0, 0.9624236501192069),
            ([[1e200, 5e49], [1e200, -5e49]], 3.0, 230.25850929940458),
            ([[1e200, 5e35], [1e200, -5e35]], 3.0, 165.78612669557128),
        ):
            ml_ext = eg.ManifoldLayout2d.from_coords(
                "hyperbolic_lorentz", np.array(coords, dtype=np.float64)
            )
            d0 = ml_ext.geodesic_distance(0, 1)
            self.assertAlmostEqual(d0, expected_d, places=7)
            ml_ext.rescale_geodesic(new_scale)
            d1 = ml_ext.geodesic_distance(0, 1)
            self.assertAlmostEqual(d1, d0, places=7)


if __name__ == "__main__":
    unittest.main()
