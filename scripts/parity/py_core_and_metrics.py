"""Core CsrGraph, Layout2d zero-copy, and geometry/spatial metrics Python checks."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

from parity.common import (
    assert_non_panic_error,
    make_adjusted_coords,
    make_fixture_coords,
    make_fixtures,
    make_similarity_coords,
)


def run_core_csr_layout_checks(eg: Any) -> list[dict[str, Any]]:
    """Run deterministic, zero-copy & adversarial checks on CsrGraph and Layout2d."""
    import numpy as np

    results: list[dict[str, Any]] = []
    fixtures = make_fixtures()

    def is_directed(g: Any) -> bool:
        attr = g.is_directed
        return bool(attr() if callable(attr) else attr)

    # 1. CsrGraph.from_csr (unweighted undirected, weighted directed, sanitize=True/False)
    t0 = time.perf_counter()
    indptr = np.array([0, 1, 3, 5, 6], dtype=np.uint32)
    indices = np.array([1, 0, 2, 1, 3, 2], dtype=np.uint32)
    g = eg.CsrGraph.from_csr(indptr, indices, directed=False, sanitize=False)
    assert g.num_nodes == 4
    assert g.num_edges == 3
    assert not is_directed(g)
    assert list(g.indptr()) == [0, 1, 3, 5, 6]
    assert list(g.indices()) == [1, 0, 2, 1, 3, 2]
    assert g.weights() is None

    dg = eg.CsrGraph.from_csr(
        np.array([0, 2, 2, 3], dtype=np.uint32),
        np.array([1, 2, 0], dtype=np.uint32),
        np.array([1.5, 2.5, 3.5], dtype=np.float64),
        directed=True,
        sanitize=False,
    )
    assert is_directed(dg)
    assert dg.num_nodes == 3
    assert dg.num_edges == 3
    assert np.allclose(dg.weights(), [1.5, 2.5, 3.5])

    dirty_indptr = np.array([0, 4, 5, 5], dtype=np.uint32)
    dirty_indices = np.array([0, 2, 1, 2, 0], dtype=np.uint32)
    dirty_weights = np.array([9.0, 2.0, 1.5, 4.0, 1.5], dtype=np.float64)
    assert_non_panic_error(
        "CsrGraph.from_csr/dirty_unsanitized",
        lambda: eg.CsrGraph.from_csr(
            dirty_indptr, dirty_indices, dirty_weights, directed=False, sanitize=False
        ),
        expected_types=(ValueError,),
    )
    clean_g = eg.CsrGraph.from_csr(
        dirty_indptr, dirty_indices, dirty_weights, directed=False, sanitize=True
    )
    assert clean_g.num_nodes == 3
    assert clean_g.num_edges == 2
    assert np.allclose(clean_g.weights(), [1.5, 4.0, 1.5, 4.0])
    results.append({
        "case": "csr_from_csr",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
    })

    # 2. CsrGraph.from_edges across fixtures & largest_connected_component
    t0 = time.perf_counter()
    for name, spec in fixtures.items():
        fg = eg.CsrGraph.from_edges(spec["n"], spec["edges"], directed=False, sanitize=True)
        assert fg.num_nodes == spec["n"], f"{name}: num_nodes mismatch"
        assert fg.num_edges == len(spec["edges"]), f"{name}: num_edges mismatch"

    g_asym = eg.CsrGraph.from_edges(
        6,
        [(0, 2), (1, 3), (3, 4), (4, 5)],
        weights=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        directed=False,
        sanitize=True,
    )
    lcc, orig_ids = g_asym.largest_connected_component()
    assert list(orig_ids) == [1, 3, 4, 5]
    assert lcc.num_nodes == 4
    assert lcc.num_edges == 3

    g_disc = eg.CsrGraph.from_edges(
        fixtures["disconnected_10"]["n"],
        fixtures["disconnected_10"]["edges"],
        directed=False,
        sanitize=True,
    )
    lcc_disc, orig_disc = g_disc.largest_connected_component()
    assert list(orig_disc) == [0, 1, 2]
    assert lcc_disc.num_nodes == 3
    results.append({
        "case": "csr_from_edges_and_lcc",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
    })

    # 3. Layout2d.zeros (float64 & float32) zero-copy coords(copy=False) round-trip
    for dtype_name, np_dtype in (("float64", np.float64), ("float32", np.float32)):
        t0 = time.perf_counter()
        layout = eg.Layout2d.zeros(5, dtype=dtype_name)
        assert len(layout) == 5
        assert layout.is_all_finite() is True

        view = layout.coords(copy=False)
        assert view.shape == (5, 2)
        assert view.dtype == np_dtype
        assert view.flags.writeable

        view[2, 0] = 3.5
        view[2, 1] = -1.25
        view_again = layout.coords(copy=False)
        assert math.isclose(float(view_again[2, 0]), 3.5, rel_tol=1e-6)
        assert math.isclose(float(view_again[2, 1]), -1.25, rel_tol=1e-6)

        copied = layout.coords(copy=True)
        assert math.isclose(float(copied[2, 0]), 3.5, rel_tol=1e-6)
        copied[0, 0] = 999.0
        assert float(layout.coords(copy=False)[0, 0]) == 0.0

        init_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np_dtype)
        l2 = eg.Layout2d.from_coords(init_arr)
        assert len(l2) == 2
        assert np.allclose(l2.coords(copy=False), init_arr)
        results.append({
            "case": f"layout2d_zerocopy_{dtype_name}",
            "status": "pass",
            "elapsed_sec": round(time.perf_counter() - t0, 4),
        })

    # 4. Adversarial core inputs (must raise ValueError/TypeError/IndexError, never PanicException)
    t0 = time.perf_counter()
    adversarial_cases: list[tuple[str, Callable[[], Any]]] = [
        (
            "CsrGraph.from_csr/len_mismatch",
            lambda: eg.CsrGraph.from_csr(
                np.array([0, 1], dtype=np.uint32),
                np.array([0, 1], dtype=np.uint32),
            ),
        ),
        (
            "CsrGraph.from_csr/non_monotonic",
            lambda: eg.CsrGraph.from_csr(
                np.array([0, 2, 1], dtype=np.uint32),
                np.array([0], dtype=np.uint32),
            ),
        ),
        (
            "CsrGraph.from_csr/oob_index",
            lambda: eg.CsrGraph.from_csr(
                np.array([0, 1, 1], dtype=np.uint32),
                np.array([5], dtype=np.uint32),
            ),
        ),
        ("CsrGraph.from_edges/oob_index", lambda: eg.CsrGraph.from_edges(2, [(0, 5)])),
        (
            "CsrGraph.from_csr/nan_weight",
            lambda: eg.CsrGraph.from_csr(
                np.array([0, 1, 2], dtype=np.uint32),
                np.array([1, 0], dtype=np.uint32),
                np.array([float("nan"), 1.0], dtype=np.float64),
            ),
        ),
        (
            "CsrGraph.from_edges/inf_weight",
            lambda: eg.CsrGraph.from_edges(
                2, [(0, 1)], np.array([float("inf")], dtype=np.float64)
            ),
        ),
        ("Layout2d.zeros/invalid_dtype", lambda: eg.Layout2d.zeros(4, dtype="int32")),
        (
            "Layout2d.from_coords/bad_shape",
            lambda: eg.Layout2d.from_coords(np.zeros((3, 3), dtype=np.float64)),
        ),
    ]
    for label, fn in adversarial_cases:
        assert_non_panic_error(label, fn)

    try:
        l_nan = eg.Layout2d.from_coords(np.array([[float("nan"), 0.0]], dtype=np.float64))
        assert l_nan.is_all_finite() is False
    except (ValueError, TypeError, IndexError) as exc:
        if "PanicException" in type(exc).__name__:
            raise AssertionError(f"Layout2d.from_coords/nan raised PanicException: {exc}") from exc

    results.append({
        "case": "core_adversarial_non_panic",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
    })

    return results


def run_geometry_metrics_spatial_checks(eg: Any) -> list[dict[str, Any]]:
    """Run geometry/spatial metrics (`float64` SNS scale invariance, PRISM, Delaunay) checks."""
    import numpy as np

    if not hasattr(eg, "scale_normalized_stress"):
        return []

    results: list[dict[str, Any]] = []
    fixtures = make_fixtures()

    t0 = time.perf_counter()
    metric_targets = [
        "sns",
        "crossing_number",
        "np1",
        "np2",
        "drng",
        "dgg",
        "aspect_ratio",
    ]
    for name, spec in fixtures.items():
        n = spec["n"]
        fg = eg.CsrGraph.from_edges(n, spec["edges"], directed=False, sanitize=True)
        coords_base = np.array(make_fixture_coords(name, n), dtype=np.float64)
        coords_sim = np.array(
            make_similarity_coords(make_fixture_coords(name, n)), dtype=np.float64
        )
        coords_adj = np.array(
            make_adjusted_coords(make_fixture_coords(name, n)), dtype=np.float64
        )

        l64 = eg.Layout2d.from_coords(coords_base)
        l64_001 = eg.Layout2d.from_coords(coords_base * 0.01)
        l64_100 = eg.Layout2d.from_coords(coords_base * 100.0)
        l64_sim = eg.Layout2d.from_coords(coords_sim)
        l64_adj = eg.Layout2d.from_coords(coords_adj)
        l32 = eg.Layout2d.from_coords(coords_base.astype(np.float32))

        sns1 = float(eg.scale_normalized_stress(fg, l64))
        sns001 = float(eg.scale_normalized_stress(fg, l64_001))
        sns100 = float(eg.scale_normalized_stress(fg, l64_100))
        assert math.isfinite(sns1) and 0.0 <= sns1 <= 1.0, f"{name}: invalid SNS {sns1}"
        assert abs(sns001 - sns1) < 1e-10, f"{name}: SNS 0.01x drift ({sns001} vs {sns1})"
        assert abs(sns100 - sns1) < 1e-10, f"{name}: SNS 100x drift ({sns100} vs {sns1})"

        sns_all = float(
            eg.sampled_scale_normalized_stress(fg, l64, num_pivots=n, seed=42)
        )
        assert abs(sns_all - sns1) < 1e-10, f"{name}: sampled SNS(n) drift"

        cr_dual = float(eg.crossing_number(fg, l64))
        cr_sweep = float(eg.sweep_crossing_number(fg, l64))
        assert cr_dual == cr_sweep, f"{name}: dual-dispatch crossing_number != sweep"
        assert float(eg.sweep_crossing_number(fg, l32)) == cr_sweep

        np1 = float(eg.neighborhood_preservation_1hop(fg, l64))
        np2 = float(eg.neighborhood_preservation_2hop(fg, l64))
        assert 0.0 <= np1 <= 1.0 and 0.0 <= np2 <= 1.0

        sig_dist_sim, sig_disp_sim = eg.prism_distortion(l64, l64_sim)
        assert sig_dist_sim < 1e-10 and sig_disp_sim < 1e-10, (
            f"{name}: similarity prism_distortion nonzero: {(sig_dist_sim, sig_disp_sim)}"
        )
        sig_dist_adj, sig_disp_adj = eg.prism_distortion(l64, l64_adj)
        assert math.isfinite(sig_dist_adj) and sig_dist_adj >= 0.0
        assert math.isfinite(sig_disp_adj) and 0.0 <= sig_disp_adj <= 1.0

        report = eg.evaluate_metrics(
            fg, l64, metric_targets, num_pivots=4, num_samples=200, seed=42
        )
        expected_keys = {
            "scale_normalized_stress",
            "crossing_number",
            "neighborhood_preservation_1hop",
            "neighborhood_preservation_2hop",
            "relative_neighborhood_preservation",
            "gabriel_graph_preservation",
            "aspect_ratio",
        }
        assert set(report.keys()) == expected_keys, f"{name}: unexpected keys {report.keys()}"
        assert abs(report["scale_normalized_stress"] - sns1) < 1e-12
        assert report["crossing_number"] == cr_sweep
        assert abs(report["neighborhood_preservation_1hop"] - np1) < 1e-12
        assert abs(report["neighborhood_preservation_2hop"] - np2) < 1e-12
        assert abs(report["relative_neighborhood_preservation"] - float(eg.drng_preservation(fg, l64))) < 1e-12
        assert abs(report["gabriel_graph_preservation"] - float(eg.dgg_preservation(fg, l64))) < 1e-12
        assert abs(report["aspect_ratio"] - float(eg.aspect_ratio(l64))) < 1e-12

        dt_edges, dt_tris = eg.delaunay_triangulation(l64)
        assert dt_edges.dtype == np.uint32 and dt_edges.ndim == 2 and dt_edges.shape[1] == 2
        assert dt_tris.dtype == np.uint32 and dt_tris.ndim == 2 and dt_tris.shape[1] == 3
        assert dt_edges.shape[0] > 0 and dt_tris.shape[0] > 0
        assert int(dt_edges.max()) < n and int(dt_tris.max()) < n
        dt_edges32, dt_tris32 = eg.delaunay_triangulation(l32)
        assert dt_edges32.shape[0] > 0 and dt_tris32.shape[0] > 0

    results.append({
        "case": "geometry_metrics_and_spatial",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
    })

    # Geometry/spatial adversarial error checks (must raise clean exceptions, never PanicException)
    t0 = time.perf_counter()
    g_p = eg.CsrGraph.from_edges(4, [(0, 1), (1, 2), (2, 3)], directed=False, sanitize=True)
    l_ok = eg.Layout2d.from_coords([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    l_short = eg.Layout2d.zeros(3)
    l_nan = eg.Layout2d.from_coords(l_ok.coords(copy=True))
    l_nan.coords(copy=False)[1, 0] = float("nan")

    def _bind_metric_call(
        metric_fn: Callable[[Any, Any], Any], graph_arg: Any, layout_arg: Any
    ) -> Callable[[], Any]:
        return lambda: metric_fn(graph_arg, layout_arg)

    metric_fns: tuple[tuple[str, Callable[[Any, Any], Any]], ...] = (
        ("scale_normalized_stress", eg.scale_normalized_stress),
        ("sweep_crossing_number", eg.sweep_crossing_number),
        ("neighborhood_preservation_1hop", eg.neighborhood_preservation_1hop),
        ("neighborhood_preservation_2hop", eg.neighborhood_preservation_2hop),
    )
    for fn_name, metric_fn in metric_fns:
        assert_non_panic_error(
            f"{fn_name}/dim_mismatch", _bind_metric_call(metric_fn, g_p, l_short), expected_types=(ValueError,)
        )
        assert_non_panic_error(
            f"{fn_name}/nan", _bind_metric_call(metric_fn, g_p, l_nan), expected_types=(ValueError,)
        )

    assert_non_panic_error(
        "prism_distortion/dim_mismatch",
        lambda: eg.prism_distortion(l_ok, l_short),
        expected_types=(ValueError,),
    )
    assert_non_panic_error(
        "prism_distortion/nan",
        lambda: eg.prism_distortion(l_ok, l_nan),
        expected_types=(ValueError,),
    )
    assert_non_panic_error(
        "delaunay_triangulation/nan",
        lambda: eg.delaunay_triangulation(l_nan),
        expected_types=(ValueError,),
    )
    assert_non_panic_error(
        "evaluate_metrics/nan",
        lambda: eg.evaluate_metrics(g_p, l_nan, ["sns"]),
        expected_types=(ValueError,),
    )
    assert_non_panic_error(
        "evaluate_metrics/non_list_targets",
        lambda: eg.evaluate_metrics(g_p, l_ok, "sns"),
        expected_types=(TypeError,),
    )
    assert_non_panic_error(
        "evaluate_metrics/non_str_target",
        lambda: eg.evaluate_metrics(g_p, l_ok, [123]),
        expected_types=(TypeError,),
    )
    assert_non_panic_error(
        "evaluate_metrics/unknown_target",
        lambda: eg.evaluate_metrics(g_p, l_ok, ["unknown_metric"]),
        expected_types=(ValueError,),
    )
    assert_non_panic_error(
        "sampled_scale_normalized_stress/zero_pivots",
        lambda: eg.sampled_scale_normalized_stress(g_p, l_ok, num_pivots=0),
        expected_types=(ValueError,),
    )
    assert_non_panic_error(
        "sampled_scale_normalized_stress/neg_seed",
        lambda: eg.sampled_scale_normalized_stress(g_p, l_ok, seed=-1),
        expected_types=(ValueError,),
    )
    assert_non_panic_error(
        "sampled_crossing_number/zero_samples",
        lambda: eg.sampled_crossing_number(g_p, l_ok, num_samples=0),
        expected_types=(ValueError,),
    )
    results.append({
        "case": "geometry_metrics_adversarial_non_panic",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
    })

    return results
