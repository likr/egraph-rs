from __future__ import annotations

import inspect
import unittest
from typing import Any

import egraph as eg
import numpy as np


def _make_path(n: int) -> tuple[eg.CsrGraph, eg.Layout2d]:
    edges = [(i, i + 1) for i in range(n - 1)]
    g = eg.CsrGraph.from_edges(n, edges, directed=False, sanitize=False)
    coords = np.array([[float(i), 0.0] for i in range(n)], dtype=np.float64)
    return g, eg.Layout2d.from_coords(coords)


class TestRtaCrosslessnessAndMetricErrors(unittest.TestCase):
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
        fn: Any,
    ) -> None:
        with self.assertRaises(expected) as ctx:
            fn()
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

    def test_adversarial_errors_and_signatures(self) -> None:
        g, layout = _make_path(4)
        short_layout = eg.Layout2d.zeros(3)

        # NaN injected via zero-copy NumPy view must raise ValueError cleanly.
        nan_layout = eg.Layout2d.from_coords(layout.coords(copy=True))
        nan_layout.coords(copy=False)[1, 0] = float("nan")

        for fn in (
            eg.scale_normalized_stress,
            eg.sampled_scale_normalized_stress,
            eg.sweep_crossing_number,
            eg.sweep_crossing_angle,
            eg.sampled_crossing_number,
            eg.torus_crossing_number,
            eg.neighborhood_preservation_1hop,
            eg.neighborhood_preservation_2hop,
            eg.drng_preservation,
            eg.dgg_preservation,
            eg.gabriel_graph_property,
            eg.angular_resolution,
            eg.ideal_edge_lengths,
            eg.crossing_number,
            eg.crossing_angle,
        ):
            self.assert_no_panic_raises((ValueError,), lambda f=fn: f(g, nan_layout))
            self.assert_no_panic_raises((ValueError,), lambda f=fn: f(g, short_layout))

        for ufn in (
            eg.node_resolution,
            eg.aspect_ratio,
            eg.delaunay_triangulation,
        ):
            self.assert_no_panic_raises((ValueError,), lambda f=ufn: f(nan_layout))

        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.prism_distortion(layout, nan_layout)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.prism_distortion(layout, short_layout)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.evaluate_metrics(g, nan_layout, ["sns"])
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.evaluate_metrics(g, short_layout, ["sns"])
        )

        # Invalid target inputs.
        self.assert_no_panic_raises((TypeError,), lambda: eg.evaluate_metrics(g, layout, "sns"))
        self.assert_no_panic_raises(
            (TypeError,), lambda: eg.evaluate_metrics(g, layout, [123])  # type: ignore[list-item]
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.evaluate_metrics(g, layout, ["unknown_metric"])
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.evaluate_metrics(g, layout, [""])
        )

        # Invalid sampling parameters.
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.sampled_scale_normalized_stress(g, layout, num_pivots=0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.sampled_scale_normalized_stress(g, layout, num_pivots=-1)
        )
        self.assert_no_panic_raises(
            (TypeError,), lambda: eg.sampled_scale_normalized_stress(g, layout, num_pivots=True)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.sampled_scale_normalized_stress(g, layout, seed=-1)
        )
        self.assert_no_panic_raises(
            (TypeError,), lambda: eg.sampled_scale_normalized_stress(g, layout, seed=True)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.sampled_crossing_number(g, layout, num_samples=0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.sampled_crossing_number(g, layout, num_samples=-5)
        )

        # Type mismatches for CSR/Layout2d inputs.
        self.assert_no_panic_raises((TypeError,), lambda: eg.crossing_number(g, "bad"))  # type: ignore[arg-type]
        self.assert_no_panic_raises((TypeError,), lambda: eg.crossing_number("bad", layout))  # type: ignore[arg-type]
        self.assert_no_panic_raises((TypeError,), lambda: eg.ideal_edge_lengths(g, "bad"))  # type: ignore[arg-type]
        self.assert_no_panic_raises((TypeError,), lambda: eg.ideal_edge_lengths("bad", layout))  # type: ignore[arg-type]
        self.assert_no_panic_raises((TypeError,), lambda: eg.aspect_ratio("bad"))  # type: ignore[arg-type]
        self.assert_no_panic_raises((TypeError,), lambda: eg.node_resolution("bad"))  # type: ignore[arg-type]
        self.assert_no_panic_raises((TypeError,), lambda: eg.edge_angle(True, 0.0, 1.0, 0.0))
        self.assert_no_panic_raises((TypeError,), lambda: eg.crossing_angle_from_pairs([123]))  # type: ignore[list-item]
        self.assert_no_panic_raises((TypeError,), lambda: eg.crossing_number_from_pairs("bad"))  # type: ignore[arg-type]

        # Check inspect.signature defaults and non-empty docstrings.
        sig_sns = inspect.signature(eg.sampled_scale_normalized_stress)
        self.assertEqual(sig_sns.parameters["num_pivots"].default, 64)
        self.assertEqual(sig_sns.parameters["seed"].default, 42)
        sig_cr = inspect.signature(eg.sampled_crossing_number)
        self.assertEqual(sig_cr.parameters["num_samples"].default, 4096)
        self.assertEqual(sig_cr.parameters["seed"].default, 42)
        sig_ev = inspect.signature(eg.evaluate_metrics)
        self.assertEqual(sig_ev.parameters["num_pivots"].default, 64)
        self.assertEqual(sig_ev.parameters["num_samples"].default, 4096)
        self.assertEqual(sig_ev.parameters["seed"].default, 42)

        for name in (
            "scale_normalized_stress",
            "sampled_scale_normalized_stress",
            "sweep_crossing_number",
            "sweep_crossing_angle",
            "sampled_crossing_number",
            "crosslessness",
            "sampled_crosslessness",
            "torus_crossing_number",
            "neighborhood_preservation_1hop",
            "neighborhood_preservation_2hop",
            "edge_angle",
            "prism_distortion",
            "drng_preservation",
            "dgg_preservation",
            "evaluate_metrics",
            "delaunay_triangulation",
            "gabriel_graph_property",
            "node_resolution",
            "aspect_ratio",
            "angular_resolution",
            "crossing_number",
            "crossing_angle",
            "crossing_edges",
            "crossing_number_from_pairs",
            "crossing_angle_from_pairs",
            "minimum_angle",
            "minimum_angle_summary",
            "layout_degeneracy",
            "adjusted_patch_nig",
            "adjusted_patch_nig_multiscale",
            "cluster_area_faithfulness",
            "cluster_area_faithfulness_detail",
            "landmark_stratified_rta",
            "landmark_stratified_rta_detail",
            "landmark_stratified_rta_from_landmarks",
            "ideal_edge_lengths",
        ):
            fn_obj = getattr(eg, name)
            self.assertTrue((fn_obj.__doc__ or "").strip(), f"missing docstring on {name}")

    def test_crosslessness_angle_degeneracy_nig_and_rta(self) -> None:
        # K4 square with crossing diagonals: c = 1, c_max = 3.
        k4 = eg.CsrGraph.from_edges(
            4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)], directed=False, sanitize=True
        )
        sq = eg.Layout2d.from_coords([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        expected_cl = 1.0 - (1.0 / 3.0) ** 0.5
        self.assertAlmostEqual(eg.crosslessness(k4, sq), expected_cl, places=12)
        self.assertAlmostEqual(
            eg.sampled_crosslessness(k4, sq, num_samples=100, seed=42),
            expected_cl,
            places=12,
        )

        # Minimum angle A1/A2 with zero-length edge divergence.
        star = eg.CsrGraph.from_edges(4, [(0, 1), (0, 2), (0, 3)], directed=False, sanitize=True)
        zero_l = eg.Layout2d.from_coords([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        ma = eg.minimum_angle_summary(star, zero_l)
        self.assertGreater(ma["a1"], ma["a2"])
        self.assertAlmostEqual(eg.minimum_angle(star, zero_l), ma["a2"], places=12)

        # Layout degeneracy: -0.0 canonicalization, f32 vs f64 duplicates, zero-length edge.
        l_deg = eg.Layout2d.from_coords(
            [[0.0, -0.0], [-0.0, 0.0], [1.0, 1.0], [1.0 + 1e-12, 1.0]]
        )
        deg = eg.layout_degeneracy(star, l_deg)
        self.assertEqual(deg["num_nodes"], 4)
        self.assertEqual(deg["num_edges"], 3)
        self.assertEqual(deg["duplicate_nodes_f32"], 4)
        self.assertEqual(deg["duplicate_nodes_f64"], 2)
        self.assertEqual(deg["zero_length_edges"], 1)
        self.assertFalse(deg["is_bbox_degenerate"])

        # Adjusted Patch-NIG single-scale & multiscale.
        coords_c = []
        labels_c = []
        for c in range(4):
            cx = float(c * 10)
            for k in range(10):
                coords_c.append([cx + 0.1 * k, 0.05 * (k % 3)])
                labels_c.append(c)
        l_comm = eg.Layout2d.from_coords(coords_c)
        nig = eg.adjusted_patch_nig(l_comm, labels_c, patch_radius=16, canvas_size=2048, seed=42)
        self.assertGreater(nig, 0.9)
        nig_ms = eg.adjusted_patch_nig_multiscale(
            l_comm, labels_c, radii=[8, 16], canvas_size=2048, seed=42
        )
        self.assertAlmostEqual(nig_ms["patch_nig_r16"], nig, places=12)

        # Landmark-Stratified RTA on a 1D path + precomputed landmark table parity.
        path = eg.CsrGraph.from_edges(
            8, [(i, i + 1) for i in range(7)], directed=False, sanitize=True
        )
        path_l = eg.Layout2d.from_coords([[float(i), 0.0] for i in range(8)])
        rta_det = eg.landmark_stratified_rta_detail(
            path, path_l, num_landmarks=4, num_triplets=10000, min_degree=1, seed=42
        )
        self.assertTrue(rta_det["is_exact"])
        self.assertAlmostEqual(rta_det["rta"], 1.0, places=12)
        self.assertAlmostEqual(
            eg.landmark_stratified_rta(
                path, path_l, num_landmarks=4, num_triplets=10000, min_degree=1, seed=42
            ),
            1.0,
            places=12,
        )
        lm_ids = eg.select_maxmin_sp_pivots(path, num_pivots=4, seed=42, min_degree=1)
        apsp = eg.all_pairs_bfs(path, 1.0)
        tbl = np.where(np.isfinite(apsp[lm_ids]), apsp[lm_ids], np.iinfo(np.uint32).max).astype(
            np.uint32
        )
        rta_tbl = eg.landmark_stratified_rta_from_landmarks(
            path_l, lm_ids, tbl, num_triplets=10000, seed=42
        )
        self.assertEqual(rta_tbl, rta_det)

        # Direct float64/float32/nested-sequence tables with np.inf (disconnected components)
        disc_g = eg.CsrGraph.from_edges(4, [(0, 1), (2, 3)], directed=False, sanitize=True)
        disc_l = eg.Layout2d.from_coords([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]])
        disc_apsp = eg.all_pairs_bfs(disc_g, 1.0)
        disc_u32 = np.where(np.isfinite(disc_apsp[[0, 2]]), disc_apsp[[0, 2]], np.iinfo(np.uint32).max).astype(np.uint32)
        expected_disc = eg.landmark_stratified_rta_from_landmarks(disc_l, [0, 2], disc_u32, num_triplets=500, seed=7)
        for cand in (
            disc_apsp[[0, 2]],
            disc_apsp[[0, 2]].astype(np.float32),
            [[0.0, 1.0, float("inf"), float("inf")], [float("inf"), float("inf"), 0, 1]],
        ):
            self.assertEqual(
                eg.landmark_stratified_rta_from_landmarks(disc_l, [0, 2], cand, num_triplets=500, seed=7),
                expected_disc,
            )
        for bad_dists in (
            np.array([[0.0, float("nan")], [1.0, 0.0]]),
            np.array([[0.0, -1.0], [1.0, 0.0]]),
            np.array([[0.0, 1.5], [1.0, 0.0]]),
            np.array([[0.0, float("-inf")], [1.0, 0.0]]),
        ):
            self.assert_no_panic_raises(
                (ValueError,),
                lambda bd=bad_dists: eg.landmark_stratified_rta_from_landmarks(
                    eg.Layout2d.zeros(2), [0, 1], bd
                ),
            )
        self.assert_no_panic_raises(
            (TypeError,),
            lambda: eg.landmark_stratified_rta_from_landmarks(
                eg.Layout2d.zeros(2), [0, 1], np.array([[True, False], [False, True]])
            ),
        )
        self.assert_no_panic_raises(
            (TypeError,),
            lambda: eg.landmark_stratified_rta_from_landmarks(
                eg.Layout2d.zeros(2), [0, 1], [[True, 0], [0, 1]]
            ),
        )

        # Selective evaluate_metrics with new targets + labels error guard.
        with self.assertRaises(ValueError):
            eg.evaluate_metrics(k4, sq, ["adjusted_patch_nig"])

        rep = eg.evaluate_metrics(
            k4,
            sq,
            [
                "crosslessness",
                "sampled_crosslessness",
                "minimum_angle",
                "minimum_angle_a1",
                "layout_degeneracy",
                "adjusted_patch_nig",
                "landmark_stratified_rta",
            ],
            labels=[0, 0, 1, 1],
            num_landmarks=4,
            min_degree=1,
        )
        self.assertAlmostEqual(rep["crosslessness"], expected_cl, places=12)
        self.assertAlmostEqual(rep["minimum_angle"], eg.minimum_angle(k4, sq), places=12)
        self.assertAlmostEqual(rep["minimum_angle_a1"], eg.minimum_angle_summary(k4, sq)["a1"], places=12)
        self.assertEqual(rep["layout_degeneracy"], 0.0)
        self.assertAlmostEqual(
            rep["adjusted_patch_nig"],
            eg.adjusted_patch_nig(sq, [0, 0, 1, 1], patch_radius=8, canvas_size=2048, seed=42),
            places=12,
        )
        self.assertGreaterEqual(rep["landmark_stratified_rta"], 0.0)

        g_comm = eg.CsrGraph.from_edges(
            40, [(i, i + 1) for i in range(39)], directed=False, sanitize=True
        )
        rep_comm = eg.evaluate_metrics(
            g_comm,
            l_comm,
            ["adjusted_patch_nig"],
            labels=labels_c,
            patch_radius=16,
        )
        self.assertAlmostEqual(rep_comm["adjusted_patch_nig"], nig, places=12)

    def test_crosslessness_directed_reciprocal_regression(self) -> None:
        # 1. Single reciprocal pair: c_max == 0 -> crosslessness == 1.0.
        g_pair = eg.CsrGraph.from_edges(2, [(0, 1), (1, 0)], directed=True, sanitize=True)
        l_pair = eg.Layout2d.from_coords([[0.0, 0.0], [1.0, 0.0]])
        self.assertEqual(eg.crosslessness(g_pair, l_pair), 1.0)
        self.assertEqual(eg.sampled_crosslessness(g_pair, l_pair, num_samples=32, seed=42), 1.0)

        # 2. Reciprocal 3-leaf star: adj_pairs > total_pairs, yet c_max == 0 -> crosslessness == 1.0.
        g_star = eg.CsrGraph.from_edges(
            4,
            [(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0)],
            directed=True,
            sanitize=True,
        )
        l_star = eg.Layout2d.from_coords([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        self.assertEqual(eg.crosslessness(g_star, l_star), 1.0)
        self.assertEqual(eg.sampled_crosslessness(g_star, l_star, num_samples=64, seed=42), 1.0)

        # 3. 4-node 6-edge directed graph: adj_pairs (16) > total_pairs (15), c_max = 1, c = 1 -> 0.0.
        g_dir4 = eg.CsrGraph.from_edges(
            4,
            [(0, 1), (0, 3), (1, 3), (3, 0), (3, 1), (3, 2)],
            directed=True,
            sanitize=True,
        )
        l_dir4 = eg.Layout2d.from_coords([[0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0]])
        self.assertEqual(eg.sweep_crossing_number(g_dir4, l_dir4), 1.0)
        self.assertEqual(eg.crosslessness(g_dir4, l_dir4), 0.0)
        self.assertEqual(eg.sampled_crosslessness(g_dir4, l_dir4, num_samples=100, seed=42), 0.0)
        rep_dir4 = eg.evaluate_metrics(
            g_dir4,
            l_dir4,
            ["crossing_number", "crosslessness", "sampled_crosslessness"],
            num_samples=100,
            seed=42,
        )
        self.assertEqual(rep_dir4["crossing_number"], 1.0)
        self.assertEqual(rep_dir4["crosslessness"], 0.0)
        self.assertEqual(rep_dir4["sampled_crosslessness"], 0.0)


if __name__ == "__main__":
    unittest.main()
