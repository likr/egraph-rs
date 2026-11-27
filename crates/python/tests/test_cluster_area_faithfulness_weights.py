from __future__ import annotations

import unittest
from typing import Any

import egraph as eg
import numpy as np


class TestClusterAreaFaithfulnessWeights(unittest.TestCase):
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
        fn: Any,
    ) -> None:
        with self.assertRaises(expected) as ctx:
            fn()
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

    def test_cluster_area_faithfulness_label_coercions_and_lazy_validation(self) -> None:
        # Integer/strided/tuple coercions, rejections,
        # bool kwargs, and lazy invalid G-CAF args when G-CAF is not requested.
        coords = np.array(
            [
                [0.0, 0.0],
                [0.2, 0.0],
                [0.0, 0.2],
                [0.2, 0.2],
                [4.0, 0.0],
                [4.2, 0.0],
                [4.0, 0.2],
                [4.2, 0.2],
            ],
            dtype=np.float64,
        )
        layout = eg.Layout2d.from_coords(coords)
        base_labels = [0, 0, 0, 0, 1, 1, 1, 1]
        base_caf = eg.cluster_area_faithfulness(layout, base_labels)
        self.assertGreater(base_caf, 0.2)

        # Valid dtypes, non-contiguous strided slice, and tuple
        for arr in (
            np.array(base_labels, dtype=np.uint16),
            np.array(base_labels, dtype=np.uint64),
            np.array(base_labels, dtype=np.int16),
            np.array(base_labels, dtype=np.int8),
            np.array([0, 99, 0, 99, 0, 99, 0, 99, 1, 99, 1, 99, 1, 99, 1, 99], dtype=np.uint32)[
                ::2
            ],
            tuple(base_labels),
        ):
            val = eg.cluster_area_faithfulness(layout, arr)
            self.assertAlmostEqual(val, base_caf, places=12)

        # ValueError for negative int16 and u32::MAX + 1 in uint64
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.cluster_area_faithfulness(
                layout, np.array([-1, 0, 0, 0, 1, 1, 1, 1], dtype=np.int16)
            ),
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.cluster_area_faithfulness(
                layout, np.array([2**32, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint64)
            ),
        )

        # TypeError for bool ndarray, bytes, and bool kwargs
        self.assert_no_panic_raises(
            (TypeError,),
            lambda: eg.cluster_area_faithfulness(
                layout, np.array([False, False, False, False, True, True, True, True], dtype=bool)
            ),
        )
        self.assert_no_panic_raises(
            (TypeError,),
            lambda: eg.cluster_area_faithfulness(layout, b"\x00\x00\x00\x00\x01\x01\x01\x01"),
        )
        self.assert_no_panic_raises(
            (TypeError,),
            lambda: eg.cluster_area_faithfulness(layout, base_labels, g_max=True),
        )
        self.assert_no_panic_raises(
            (TypeError,),
            lambda: eg.cluster_area_faithfulness(layout, base_labels, sigma_cells=False),
        )
        self.assert_no_panic_raises(
            (TypeError,),
            lambda: eg.cluster_area_faithfulness_detail(
                layout, base_labels, return_per_cluster=1  # type: ignore[arg-type]
            ),
        )

        # Lazy non-GCAF evaluation ignores mismatched labels & invalid sigma_cells
        g = eg.CsrGraph.from_edges(
            8, [(i, (i + 1) % 8) for i in range(8)], directed=False, sanitize=True
        )
        sns_only = eg.evaluate_metrics(
            g, layout, ["sns"], labels=[0, 1], sigma_cells=-1.0
        )
        self.assertIn("scale_normalized_stress", sns_only)
        self.assertNotIn("cluster_area_faithfulness", sns_only)

    def test_cluster_area_faithfulness_boundary_clamping_and_weights(self) -> None:
        # Finite boundary_r95 clamps g_eff first while preserving effective_boundary_r95.
        rng = np.random.default_rng(123)
        n = 50_000
        coords_big = np.vstack(
            [
                rng.normal(loc=[-1.0, 0.0], scale=0.15, size=(n // 2, 2)),
                rng.normal(loc=[1.0, 0.0], scale=0.15, size=(n - n // 2, 2)),
            ]
        )
        labels_big = [0] * (n // 2) + [1] * (n - n // 2)
        det_2048 = eg.cluster_area_faithfulness_detail(
            eg.Layout2d.from_coords(coords_big),
            labels_big,
            g_max=2048,
            sigma_cells=3.0,
            boundary_r95=2.0,
        )
        self.assertEqual(det_2048["g_eff"], 506)
        self.assertEqual(det_2048["effective_boundary_r95"], 2.0)
        self.assertGreater(float(det_2048["in_frac"]), 0.99)

        # node_weights vs cluster_weights parity, integer/float arrays, stevens_gamma, and errors.
        coords = np.array(
            [
                [0.0, 0.0],
                [0.2, 0.0],
                [0.0, 0.2],
                [0.2, 0.2],
                [4.0, 0.0],
                [4.2, 0.0],
                [4.0, 0.2],
                [4.2, 0.2],
            ],
            dtype=np.float64,
        )
        layout = eg.Layout2d.from_coords(coords)
        labels = [20, 20, 20, 20, 10, 10, 10, 10]  # sorted cluster_ids = [10, 20]
        g = eg.CsrGraph.from_edges(
            8, [(i, (i + 1) % 8) for i in range(8)], directed=False, sanitize=True
        )
        nw = [1.0, 2.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]  # cluster 10 sum=12, cluster 20 sum=6
        cw = [12.0, 6.0]

        caf_nw = eg.cluster_area_faithfulness(layout, labels, node_weights=nw)
        caf_cw = eg.cluster_area_faithfulness(layout, labels, cluster_weights=cw)
        self.assertAlmostEqual(caf_nw, caf_cw, places=12)

        det_nw = eg.cluster_area_faithfulness_detail(
            layout, labels, node_weights=nw, stevens_gamma=2.0, return_per_cluster=True
        )
        det_cw = eg.cluster_area_faithfulness_detail(
            layout, labels, cluster_weights=cw, stevens_gamma=2.0, return_per_cluster=True
        )
        self.assertAlmostEqual(float(det_nw["caf"]), float(det_cw["caf"]), places=12)
        np.testing.assert_allclose(det_nw["p"], np.array([0.8, 0.2]), atol=1e-12)
        np.testing.assert_allclose(det_cw["p"], np.array([0.8, 0.2]), atol=1e-12)

        for w_arr in (
            np.array([12, 6], dtype=np.uint8),
            np.array([12, 6], dtype=np.uint16),
            np.array([12, 6], dtype=np.int8),
            np.array([12, 6], dtype=np.int16),
            np.array([12.0, 6.0], dtype=np.float32),
            np.array([12.0, 6.0], dtype=np.float64),
        ):
            self.assertAlmostEqual(
                eg.cluster_area_faithfulness(layout, labels, cluster_weights=w_arr),
                caf_cw,
                places=6,
            )

        ev_nw = eg.evaluate_metrics(g, layout, ["gcaf"], labels=labels, node_weights=nw)
        ev_cw = eg.evaluate_metrics(g, layout, ["gcaf"], labels=labels, cluster_weights=cw)
        self.assertAlmostEqual(ev_nw["cluster_area_faithfulness"], caf_nw, places=12)
        self.assertAlmostEqual(ev_cw["cluster_area_faithfulness"], caf_cw, places=12)

        # Lazy validation in evaluate_metrics ignores invalid weight values if gcaf not requested.
        ev_lazy = eg.evaluate_metrics(
            g, layout, ["aspect_ratio"], labels=labels, node_weights=[-1.0]
        )
        self.assertIn("aspect_ratio", ev_lazy)

        # ValueError guards: mutual exclusion, mismatch, non-finite, <= 0.
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.cluster_area_faithfulness(
                layout, labels, node_weights=nw, cluster_weights=cw
            ),
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.cluster_area_faithfulness(layout, labels, node_weights=nw[:-1]),
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.cluster_area_faithfulness(layout, labels, cluster_weights=[1.0]),
        )
        for bad_val in (0.0, -1.0, float("nan"), float("inf")):
            self.assert_no_panic_raises(
                (ValueError,),
                lambda v=bad_val: eg.cluster_area_faithfulness(
                    layout, labels, cluster_weights=[v, 1.0]
                ),
            )
            self.assert_no_panic_raises(
                (ValueError,),
                lambda v=bad_val: eg.cluster_area_faithfulness(
                    layout, labels, node_weights=[v] + nw[1:]
                ),
            )

        # TypeError guards: bool, complex, str, dict.
        bad_typed_inputs: tuple[Any, ...] = (
            np.array([True, False], dtype=bool),
            np.array([1 + 0j, 2 + 0j], dtype=np.complex128),
            [True, False],
            "12",
            {0: 1.0, 1: 2.0},
        )
        for bad_typed in bad_typed_inputs:
            self.assert_no_panic_raises(
                (TypeError,),
                lambda bw=bad_typed: eg.cluster_area_faithfulness(
                    layout, labels, cluster_weights=bw
                ),
            )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.cluster_area_faithfulness(
                layout, labels, cluster_weights=np.ones((2, 1), dtype=np.float64)
            ),
        )


if __name__ == "__main__":
    unittest.main()
