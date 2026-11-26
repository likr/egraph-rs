"""Python checks for G-CAF (Cluster Area Faithfulness)."""

from __future__ import annotations

import math
import time
from typing import Any

from parity.common import assert_non_panic_error, bind_call
from parity.fixtures_gcaf import gcaf_small_n_regression_cases


def run_gcaf_python_checks(eg: Any) -> list[dict[str, Any]]:
    """Verify G-CAF scalar/detail/dispatcher parity, invariances, boundaries, and adversarial guards."""
    import numpy as np

    t0 = time.perf_counter()
    # Exact-in-f32 dyadic integer 100-node 2-cluster grid
    pts = []
    labels = []
    for c in range(10):
        for r in range(10):
            pts.append([float(c + (0 if c < 5 else 2)), float(r)])
            labels.append(0 if c < 5 else 1)
    coords64 = np.array(pts, dtype=np.float64)
    coords32 = coords64.astype(np.float32)
    l64 = eg.Layout2d.from_coords(coords64)
    l32 = eg.Layout2d.from_coords(coords32)

    caf64 = float(eg.cluster_area_faithfulness(l64, labels))
    caf32 = float(eg.cluster_area_faithfulness(l32, labels))
    assert abs(caf64 - caf32) < 1e-12, f"G-CAF f64 vs f32 dyadic drift: {caf64} vs {caf32}"

    det64 = eg.cluster_area_faithfulness_detail(l64, labels, return_per_cluster=True)
    det32 = eg.cluster_area_faithfulness_detail(l32, labels, return_per_cluster=True)
    scalar_keys = (
        "caf",
        "purity_adj",
        "prop_vis",
        "res_density",
        "res_deconv",
        "res_combined",
        "spatial_coherence",
        "in_frac",
    )
    for k in scalar_keys:
        assert abs(float(det64[k]) - float(det32[k])) < 1e-12, f"G-CAF scalar {k} drift"
    assert abs(float(det64["effective_boundary_r95"]) - float(det32["effective_boundary_r95"])) < 1e-12
    assert det64["effective_boundary_r95"] > 0.0
    assert det64["g_eff"] == det32["g_eff"] >= 12
    assert det64["num_clusters"] == det32["num_clusters"] == 2
    assert det64["num_nodes"] == det32["num_nodes"] == 100
    vec_keys = (
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
    )
    for k in vec_keys:
        assert np.max(np.abs(np.asarray(det64[k], dtype=np.float64) - np.asarray(det32[k], dtype=np.float64))) < 1e-12

    g100 = eg.CsrGraph.from_edges(100, [(i, i + 1) for i in range(99)], directed=False, sanitize=True)
    for alias in ("cluster_area_faithfulness", "gcaf", "clusterAreaFaithfulness"):
        rep = eg.evaluate_metrics(g100, l64, [alias], community_labels=labels)
        assert abs(float(rep["cluster_area_faithfulness"]) - caf64) < 1e-12

    # Imbalanced 24-node fixture: stevens_gamma, boundary_r95=inf, custom config
    coords_imb = np.array(
        [
            [
                (-2.0 + 0.5 * (i / 19.0)) if i < 20 else (1.5 + 0.5 * ((i - 20) / 3.0)),
                math.sin(i * 0.7) * 0.2,
            ]
            for i in range(24)
        ],
        dtype=np.float64,
    )
    labels_imb = [0] * 20 + [1] * 4
    l_imb = eg.Layout2d.from_coords(coords_imb)
    g_imb = eg.CsrGraph.from_edges(24, [(i, i + 1) for i in range(23)], directed=False, sanitize=True)

    det_g05 = eg.cluster_area_faithfulness_detail(l_imb, labels_imb, stevens_gamma=0.5, return_per_cluster=True)
    det_g20 = eg.cluster_area_faithfulness_detail(l_imb, labels_imb, stevens_gamma=2.0, return_per_cluster=True)
    assert abs(float(np.sum(det_g05["p"])) - 1.0) < 1e-12
    assert abs(float(np.sum(det_g20["p"])) - 1.0) < 1e-12
    assert float(det_g05["p"][0]) < float(det_g20["p"][0])

    det_inf = eg.cluster_area_faithfulness_detail(l_imb, labels_imb, boundary_r95=float("inf"))
    assert abs(float(det_inf["in_frac"]) - 1.0) < 1e-12
    assert 0.0 <= float(det_inf["caf"]) <= 1.0

    caf_custom = float(
        eg.cluster_area_faithfulness(
            l_imb,
            labels_imb,
            g_max=64,
            sigma_cells=0.9,
            kappa=8.0,
            tau_frac=0.2,
            boundary_r95=2.5,
            stevens_gamma=0.7,
        )
    )
    eval_custom = eg.evaluate_metrics(
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
    assert abs(float(eval_custom["cluster_area_faithfulness"]) - caf_custom) < 1e-12

    # Degenerate boundaries: N=0, N=1, K=1, collapsed
    d0 = eg.cluster_area_faithfulness_detail(eg.Layout2d.zeros(0), [], return_per_cluster=True)
    assert d0["caf"] == 0.0 and d0["effective_boundary_r95"] == 0.0 and d0["g_eff"] == 0
    assert d0["num_clusters"] == 0 and d0["num_nodes"] == 0
    d1 = eg.cluster_area_faithfulness_detail(eg.Layout2d.from_coords([[1.0, 2.0]]), [5], return_per_cluster=True)
    assert d1["caf"] == 0.0 and d1["effective_boundary_r95"] == 0.0 and d1["g_eff"] == 0
    assert d1["num_clusters"] == 1 and d1["num_nodes"] == 1
    dk1 = eg.cluster_area_faithfulness_detail(
        eg.Layout2d.from_coords([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        [3, 3, 3, 3],
        return_per_cluster=True,
    )
    assert dk1["caf"] == 0.0 and dk1["effective_boundary_r95"] == 0.0 and dk1["g_eff"] == 0
    assert dk1["num_clusters"] == 1 and dk1["num_nodes"] == 4
    l_col = eg.Layout2d.from_coords([[2.0, -1.0]] * 6)
    assert float(eg.cluster_area_faithfulness(l_col, [0, 0, 0, 1, 1, 1])) == 0.0

    # Target weights positive parity + adversarial checks
    imb_nw = [1.5 if lbl == 0 else 3.0 for lbl in labels_imb]
    imb_cw = [30.0, 12.0]
    caf_nw = float(eg.cluster_area_faithfulness(l_imb, labels_imb, node_weights=imb_nw, stevens_gamma=2.0))
    caf_cw = float(eg.cluster_area_faithfulness(l_imb, labels_imb, cluster_weights=imb_cw, stevens_gamma=2.0))
    assert abs(caf_nw - caf_cw) < 1e-12
    ev_cw = eg.evaluate_metrics(
        g_imb, l_imb, ["gcaf"], community_labels=labels_imb, cluster_weights=imb_cw, stevens_gamma=2.0
    )
    assert abs(float(ev_cw["cluster_area_faithfulness"]) - caf_cw) < 1e-12
    assert_non_panic_error(
        "gcaf/weights_mutually_exclusive",
        lambda: eg.cluster_area_faithfulness(l_imb, labels_imb, node_weights=imb_nw, cluster_weights=imb_cw),
        expected_types=(ValueError,),
    )
    assert_non_panic_error(
        "gcaf/weights_zero",
        lambda: eg.cluster_area_faithfulness(l_imb, labels_imb, cluster_weights=[0.0, 1.0]),
        expected_types=(ValueError,),
    )
    assert_non_panic_error(
        "gcaf/weights_bool",
        lambda: eg.cluster_area_faithfulness(l_imb, labels_imb, cluster_weights=[True, False]),
        expected_types=(TypeError,),
    )

    # Adversarial non-panic error checks
    l_nan = eg.Layout2d.from_coords(coords_imb.copy())
    l_nan.coords(copy=False)[0, 0] = float("nan")
    assert_non_panic_error(
        "gcaf/label_mismatch",
        lambda: eg.cluster_area_faithfulness(l_imb, labels_imb[:-1]),
        expected_types=(ValueError,),
    )
    assert_non_panic_error(
        "gcaf/nan_layout",
        lambda: eg.cluster_area_faithfulness(l_nan, labels_imb),
        expected_types=(ValueError,),
    )
    for bad_g in (11, -1):
        assert_non_panic_error(
            f"gcaf/g_max_{bad_g}",
            bind_call(eg.cluster_area_faithfulness, l_imb, labels_imb, g_max=bad_g),
            expected_types=(ValueError,),
        )
    for bad_s in (0.0, -0.5, float("nan"), float("inf"), 400.0):
        assert_non_panic_error(
            f"gcaf/sigma_{bad_s}",
            bind_call(eg.cluster_area_faithfulness, l_imb, labels_imb, sigma_cells=bad_s),
            expected_types=(ValueError,),
        )
    for bad_v in (0.0, -1.0, float("nan"), float("inf")):
        assert_non_panic_error(
            f"gcaf/kappa_{bad_v}",
            bind_call(eg.cluster_area_faithfulness, l_imb, labels_imb, kappa=bad_v),
            expected_types=(ValueError,),
        )
        assert_non_panic_error(
            f"gcaf/tau_frac_{bad_v}",
            bind_call(eg.cluster_area_faithfulness, l_imb, labels_imb, tau_frac=bad_v),
            expected_types=(ValueError,),
        )
        assert_non_panic_error(
            f"gcaf/stevens_gamma_{bad_v}",
            bind_call(eg.cluster_area_faithfulness, l_imb, labels_imb, stevens_gamma=bad_v),
            expected_types=(ValueError,),
        )
    for bad_r in (0.0, -1.0, float("nan"), float("-inf")):
        assert_non_panic_error(
            f"gcaf/boundary_r95_{bad_r}",
            bind_call(eg.cluster_area_faithfulness, l_imb, labels_imb, boundary_r95=bad_r),
            expected_types=(ValueError,),
        )
    for missing_target in ("cluster_area_faithfulness", "gcaf"):
        assert_non_panic_error(
            f"evaluate_metrics/{missing_target}_missing_labels",
            bind_call(eg.evaluate_metrics, g_imb, l_imb, [missing_target]),
            expected_types=(ValueError,),
        )
    lazy_ok = eg.evaluate_metrics(g_imb, l_imb, ["aspect_ratio"], g_max=4)
    assert "aspect_ratio" in lazy_ok and "cluster_area_faithfulness" not in lazy_ok
    assert_non_panic_error(
        "evaluate_metrics/gcaf_lazy_gmax4",
        lambda: eg.evaluate_metrics(g_imb, l_imb, ["gcaf"], community_labels=labels_imb, g_max=4),
        expected_types=(ValueError,),
    )

    # Small-N (N <= 41) MEC quantile tie & boundary_r95=1.0 regression cases
    for case in gcaf_small_n_regression_cases():
        coords_arr = np.asarray(case["coords"], dtype=np.float32).reshape(-1, 2)
        n_case = coords_arr.shape[0]
        l_case = eg.Layout2d.from_coords(coords_arr)
        g_case = eg.CsrGraph.from_edges(n_case, [(i, i + 1) for i in range(n_case - 1)], directed=False, sanitize=True)
        kwargs: dict[str, Any] = {}
        if case["g_max"] is not None:
            kwargs["g_max"] = case["g_max"]
        if case["sigma_cells"] is not None:
            kwargs["sigma_cells"] = case["sigma_cells"]
        if case["boundary_r95"] is not None:
            kwargs["boundary_r95"] = case["boundary_r95"]
        if case["stevens_gamma"] is not None:
            kwargs["stevens_gamma"] = case["stevens_gamma"]
        caf_s = float(eg.cluster_area_faithfulness(l_case, case["labels"], **kwargs))
        det_s = eg.cluster_area_faithfulness_detail(l_case, case["labels"], return_per_cluster=True, **kwargs)
        eval_s = eg.evaluate_metrics(g_case, l_case, ["gcaf"], community_labels=case["labels"], **kwargs)
        assert 0.0 <= caf_s <= 1.0 and math.isfinite(caf_s)
        assert abs(caf_s - float(det_s["caf"])) < 1e-12
        assert abs(caf_s - float(eval_s["cluster_area_faithfulness"])) < 1e-12
        if case["name"] == "idx445_pure_tail":
            assert float(det_s["purity"][3]) == 1.0, f"expected exact 1.0 purity[3], got {det_s['purity'][3]}"

    return [{
        "case": "cluster_area_faithfulness_checks",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
    }]
