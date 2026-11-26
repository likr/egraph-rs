from __future__ import annotations

import inspect
import math
import unittest
from typing import Any

import egraph as eg
import numpy as np


class TestClusterAreaFaithfulness(unittest.TestCase):
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
        fn: Any,
    ) -> None:
        with self.assertRaises(expected) as ctx:
            fn()
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

    def test_cluster_area_faithfulness_and_detail(self) -> None:
        self.assertIn("cluster_area_faithfulness", eg.__all__)
        self.assertIn("cluster_area_faithfulness_detail", eg.__all__)

        # Check inspect.signature defaults across standalone and evaluate_metrics APIs.
        sig_caf = inspect.signature(eg.cluster_area_faithfulness)
        self.assertEqual(sig_caf.parameters["g_max"].default, 128)
        self.assertEqual(sig_caf.parameters["sigma_cells"].default, 0.85)
        self.assertEqual(sig_caf.parameters["kappa"].default, 10.0)
        self.assertEqual(sig_caf.parameters["tau_frac"].default, 0.18)
        self.assertEqual(sig_caf.parameters["boundary_r95"].default, 2.0)
        self.assertEqual(sig_caf.parameters["stevens_gamma"].default, 1.0)

        sig_det = inspect.signature(eg.cluster_area_faithfulness_detail)
        self.assertEqual(sig_det.parameters["g_max"].default, 128)
        self.assertEqual(sig_det.parameters["return_per_cluster"].default, False)

        sig_ev = inspect.signature(eg.evaluate_metrics)
        self.assertEqual(sig_ev.parameters["g_max"].default, 128)
        self.assertEqual(sig_ev.parameters["sigma_cells"].default, 0.85)
        self.assertEqual(sig_ev.parameters["kappa"].default, 10.0)
        self.assertEqual(sig_ev.parameters["tau_frac"].default, 0.18)
        self.assertEqual(sig_ev.parameters["boundary_r95"].default, 2.0)
        self.assertEqual(sig_ev.parameters["stevens_gamma"].default, 1.0)

        # Separated two-cluster fixture (100 nodes) in f64 and f32 + multiple label dtypes.
        rng = np.random.default_rng(7)
        coords = np.array(
            [[float(c) + (0.0 if c < 5 else 2.0), float(r)] for r in range(10) for c in range(10)],
            dtype=np.float64,
        )
        labels_list = [0 if c < 5 else 1 for _r in range(10) for c in range(10)]
        layout64 = eg.Layout2d.from_coords(coords)
        layout32 = eg.Layout2d.from_coords(coords.astype(np.float32))

        caf64 = eg.cluster_area_faithfulness(layout64, labels_list)
        self.assertGreater(caf64, 0.7)
        self.assertLessEqual(caf64, 1.0)

        caf32 = eg.cluster_area_faithfulness(layout32, labels_list)
        self.assertAlmostEqual(caf32, caf64, places=5)

        int_dtypes: tuple[type[np.integer[Any]], ...] = (np.uint32, np.int64, np.int32, np.uint8)
        for dtype in int_dtypes:
            arr = np.array(labels_list, dtype=dtype)
            self.assertAlmostEqual(eg.cluster_area_faithfulness(layout64, arr), caf64, places=12)

        # Detail without and with per-cluster vectors.
        base_keys = {
            "caf",
            "purity_adj",
            "prop_vis",
            "res_density",
            "res_deconv",
            "res_combined",
            "spatial_coherence",
            "in_frac",
            "effective_boundary_r95",
            "g_eff",
            "num_clusters",
            "num_nodes",
        }
        per_cluster_keys = {
            "cluster_ids",
            "counts",
            "p",
            "q",
            "purity",
            "resolution",
            "res_density_per_cluster",
            "res_deconv_per_cluster",
            "caf_per_cluster",
            "coherence",
        }
        det_summary = eg.cluster_area_faithfulness_detail(
            layout64, labels_list, return_per_cluster=False
        )
        self.assertEqual(set(det_summary.keys()), base_keys)
        self.assertEqual(det_summary["caf"], caf64)
        self.assertEqual(det_summary["num_clusters"], 2)
        self.assertEqual(det_summary["num_nodes"], 100)
        self.assertEqual(det_summary["effective_boundary_r95"], 2.0)
        self.assertEqual(det_summary["g_eff"], 12)
        for key in (
            "caf",
            "purity_adj",
            "prop_vis",
            "res_density",
            "res_deconv",
            "res_combined",
            "spatial_coherence",
            "in_frac",
        ):
            val = float(det_summary[key])
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

        det_full = eg.cluster_area_faithfulness_detail(
            layout64, labels_list, return_per_cluster=True
        )
        self.assertEqual(set(det_full.keys()), base_keys | per_cluster_keys)
        self.assertEqual(det_full["caf"], caf64)
        for key in per_cluster_keys:
            vec = det_full[key]
            self.assertIsInstance(vec, np.ndarray)
            self.assertEqual(vec.shape, (2,))
        np.testing.assert_array_equal(det_full["cluster_ids"], np.array([0, 1], dtype=np.uint32))
        np.testing.assert_array_equal(det_full["counts"], np.array([50, 50], dtype=np.uintp))
        self.assertAlmostEqual(float(np.sum(det_full["p"])), 1.0, places=12)
        self.assertAlmostEqual(float(np.sum(det_full["q"])), 1.0, places=12)

        # Exact grid-isometry invariances: translation + uniform scaling, 90-deg rotation, reflection.
        coords_scaled_shifted = coords * 7.5 + np.array([150.0, -42.0])
        caf_scaled_shifted = eg.cluster_area_faithfulness(
            eg.Layout2d.from_coords(coords_scaled_shifted), labels_list
        )
        self.assertAlmostEqual(caf_scaled_shifted, caf64, places=12)

        coords_rot90 = np.column_stack([-coords[:, 1], coords[:, 0]])
        caf_rot90 = eg.cluster_area_faithfulness(
            eg.Layout2d.from_coords(coords_rot90), labels_list
        )
        self.assertAlmostEqual(caf_rot90, caf64, places=12)

        # Oblique rotation preserves G-CAF up to raster grid discretization (~1e-4).
        theta = 0.65
        rot = np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
            dtype=np.float64,
        )
        coords_sim = (coords @ rot.T) * 7.5 + np.array([150.0, -42.0])
        caf_sim = eg.cluster_area_faithfulness(eg.Layout2d.from_coords(coords_sim), labels_list)
        self.assertAlmostEqual(caf_sim, caf64, places=3)

        coords_refl = coords.copy()
        coords_refl[:, 1] = -coords_refl[:, 1]
        caf_refl = eg.cluster_area_faithfulness(eg.Layout2d.from_coords(coords_refl), labels_list)
        self.assertAlmostEqual(caf_refl, caf64, places=12)

        perm = rng.permutation(len(labels_list))
        coords_perm = coords[perm]
        labels_perm = [labels_list[i] for i in perm]
        caf_perm = eg.cluster_area_faithfulness(eg.Layout2d.from_coords(coords_perm), labels_perm)
        self.assertAlmostEqual(caf_perm, caf64, places=12)

        relabeled = [100 if lbl == 0 else 200 for lbl in labels_list]
        det_relabeled = eg.cluster_area_faithfulness_detail(
            layout64, relabeled, return_per_cluster=True
        )
        self.assertAlmostEqual(float(det_relabeled["caf"]), caf64, places=12)
        np.testing.assert_array_equal(
            det_relabeled["cluster_ids"], np.array([100, 200], dtype=np.uint32)
        )

    def test_cluster_area_faithfulness_boundaries_and_evaluate_metrics(self) -> None:
        # Degenerate and boundary layouts: N=0, N=1, K=1, collapsed coordinates.
        empty_det = eg.cluster_area_faithfulness_detail(
            eg.Layout2d.zeros(0), [], return_per_cluster=True
        )
        self.assertEqual(empty_det["caf"], 0.0)
        self.assertEqual(empty_det["effective_boundary_r95"], 0.0)
        self.assertEqual(empty_det["g_eff"], 0)
        self.assertEqual(empty_det["num_clusters"], 0)
        self.assertEqual(empty_det["num_nodes"], 0)
        self.assertEqual(len(empty_det["cluster_ids"]), 0)

        one_det = eg.cluster_area_faithfulness_detail(
            eg.Layout2d.from_coords([[1.0, 2.0]]), [5], return_per_cluster=True
        )
        self.assertEqual(one_det["caf"], 0.0)
        self.assertEqual(one_det["effective_boundary_r95"], 0.0)
        self.assertEqual(one_det["g_eff"], 0)
        self.assertEqual(one_det["num_clusters"], 1)
        self.assertEqual(one_det["num_nodes"], 1)

        l4 = eg.Layout2d.from_coords([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        k1_det = eg.cluster_area_faithfulness_detail(l4, [3, 3, 3, 3], return_per_cluster=True)
        self.assertEqual(k1_det["caf"], 0.0)
        self.assertEqual(k1_det["effective_boundary_r95"], 0.0)
        self.assertEqual(k1_det["g_eff"], 0)
        self.assertEqual(k1_det["num_clusters"], 1)
        self.assertEqual(k1_det["num_nodes"], 4)

        collapsed = eg.Layout2d.from_coords([[2.0, -1.0]] * 6)
        self.assertEqual(eg.cluster_area_faithfulness(collapsed, [0, 0, 0, 1, 1, 1]), 0.0)

        # Imbalanced clusters: stevens_gamma and boundary_r95=inf.
        coords_imb = np.vstack(
            [
                np.column_stack([np.linspace(-2.0, -1.5, 20), np.zeros(20)]),
                np.column_stack([np.linspace(1.5, 2.0, 4), np.zeros(4)]),
            ]
        )
        coords_imb[:, 1] = np.sin(np.arange(24) * 0.7) * 0.2
        labels_imb = [0] * 20 + [1] * 4
        l_imb = eg.Layout2d.from_coords(coords_imb)

        det_g05 = eg.cluster_area_faithfulness_detail(
            l_imb, labels_imb, stevens_gamma=0.5, return_per_cluster=True
        )
        det_g20 = eg.cluster_area_faithfulness_detail(
            l_imb, labels_imb, stevens_gamma=2.0, return_per_cluster=True
        )
        self.assertAlmostEqual(float(np.sum(det_g05["p"])), 1.0, places=12)
        self.assertAlmostEqual(float(np.sum(det_g20["p"])), 1.0, places=12)
        self.assertLess(float(det_g05["p"][0]), float(det_g20["p"][0]))

        det_inf = eg.cluster_area_faithfulness_detail(
            l_imb, labels_imb, boundary_r95=float("inf"), return_per_cluster=False
        )
        self.assertAlmostEqual(float(det_inf["in_frac"]), 1.0, places=12)
        self.assertGreaterEqual(float(det_inf["caf"]), 0.0)
        self.assertLessEqual(float(det_inf["caf"]), 1.0)

        # evaluate_metrics integration ("cluster_area_faithfulness" & "gcaf") + lazy validation.
        g_imb = eg.CsrGraph.from_edges(
            24, [(i, i + 1) for i in range(23)], directed=False, sanitize=True
        )
        rep = eg.evaluate_metrics(
            g_imb,
            l_imb,
            ["cluster_area_faithfulness"],
            community_labels=labels_imb,
            g_max=64,
            sigma_cells=0.9,
            kappa=8.0,
            tau_frac=0.2,
            boundary_r95=2.5,
            stevens_gamma=0.7,
        )
        expected_caf = eg.cluster_area_faithfulness(
            l_imb,
            labels_imb,
            g_max=64,
            sigma_cells=0.9,
            kappa=8.0,
            tau_frac=0.2,
            boundary_r95=2.5,
            stevens_gamma=0.7,
        )
        self.assertAlmostEqual(rep["cluster_area_faithfulness"], expected_caf, places=12)

        rep_alias = eg.evaluate_metrics(g_imb, l_imb, ["gcaf"], community_labels=labels_imb)
        self.assertAlmostEqual(
            rep_alias["cluster_area_faithfulness"],
            eg.cluster_area_faithfulness(l_imb, labels_imb),
            places=12,
        )

        # Invalid G-CAF config is lazily ignored unless G-CAF target is requested.
        ok = eg.evaluate_metrics(g_imb, l_imb, ["aspect_ratio"], g_max=4)
        self.assertIn("aspect_ratio", ok)
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.evaluate_metrics(
                g_imb, l_imb, ["cluster_area_faithfulness"], community_labels=labels_imb, g_max=4
            ),
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.evaluate_metrics(g_imb, l_imb, ["cluster_area_faithfulness"]),
        )

        # Adversarial error handling (no panics).
        nan_layout = eg.Layout2d.from_coords(coords_imb.copy())
        nan_layout.coords(copy=False)[0, 0] = float("nan")
        inf_layout = eg.Layout2d.from_coords(coords_imb.copy())
        inf_layout.coords(copy=False)[0, 1] = float("inf")

        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.cluster_area_faithfulness(nan_layout, labels_imb)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.cluster_area_faithfulness(inf_layout, labels_imb)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.cluster_area_faithfulness(l_imb, labels_imb[:-1])
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.cluster_area_faithfulness(l_imb, labels_imb, g_max=11)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.cluster_area_faithfulness(l_imb, labels_imb, g_max=-1)
        )
        self.assert_no_panic_raises(
            (TypeError,), lambda: eg.cluster_area_faithfulness(l_imb, labels_imb, g_max=True)
        )
        for bad_sigma in (0.0, -0.5, float("nan"), float("inf"), 400.0):
            self.assert_no_panic_raises(
                (ValueError,),
                lambda s=bad_sigma: eg.cluster_area_faithfulness(
                    l_imb, labels_imb, sigma_cells=s
                ),
            )
        for bad_pos in (0.0, -1.0, float("nan"), float("inf")):
            self.assert_no_panic_raises(
                (ValueError,),
                lambda v=bad_pos: eg.cluster_area_faithfulness(l_imb, labels_imb, kappa=v),
            )
            self.assert_no_panic_raises(
                (ValueError,),
                lambda v=bad_pos: eg.cluster_area_faithfulness(l_imb, labels_imb, tau_frac=v),
            )
            self.assert_no_panic_raises(
                (ValueError,),
                lambda v=bad_pos: eg.cluster_area_faithfulness(l_imb, labels_imb, stevens_gamma=v),
            )
        for bad_b in (0.0, -1.0, float("nan"), float("-inf")):
            self.assert_no_panic_raises(
                (ValueError,),
                lambda b=bad_b: eg.cluster_area_faithfulness(l_imb, labels_imb, boundary_r95=b),
            )
        self.assert_no_panic_raises(
            (TypeError,),
            lambda: eg.cluster_area_faithfulness(l_imb, labels_imb, sigma_cells=True),
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.cluster_area_faithfulness(l_imb, [-1] + labels_imb[1:]),
        )
        self.assert_no_panic_raises(
            (TypeError,),
            lambda: eg.cluster_area_faithfulness(l_imb, [True] * len(labels_imb)),
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.cluster_area_faithfulness(
                l_imb, np.zeros((len(labels_imb), 1), dtype=np.uint32)
            ),
        )
        self.assert_no_panic_raises(
            (TypeError,),
            lambda: eg.cluster_area_faithfulness(l_imb, "01" * 12),  # type: ignore[arg-type]
        )


if __name__ == "__main__":
    unittest.main()
