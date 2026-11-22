#!/usr/bin/env python3
"""Cross-target golden fixture & adversarial contract harness for egraph.

Supports both current legacy v6-alpha Python bindings (`Graph` / `DrawingEuclidean2d`)
and v6 redesigned APIs (`CsrGraph` / `Layout2d` / `scale_normalized_stress`), plus
optional Node.js WASM cross-checking via `--wasm-pkg`.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys
import time
from typing import Any, Callable


def make_fixtures() -> dict[str, dict[str, Any]]:
    """Build deterministic canonical graph fixtures."""
    barbell_edges = [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
        (3, 4), (4, 5),
        (5, 6), (5, 7), (5, 8), (6, 7), (6, 8), (7, 8),
    ]
    return {
        "path_10": {
            "n": 10,
            "edges": [(i, i + 1) for i in range(9)],
        },
        "cycle_12": {
            "n": 12,
            "edges": [(i, (i + 1) % 12) for i in range(12)],
        },
        "barbell_2comm_9": {
            "n": 9,
            "edges": barbell_edges,
        },
        "star_11": {
            "n": 11,
            "edges": [(0, i) for i in range(1, 11)],
        },
        "disconnected_10": {
            "n": 10,
            "edges": [(0, 1), (1, 2), (2, 0), (4, 5), (5, 6), (7, 8)],
        },
    }


def make_fixture_coords(name: str, n: int) -> list[tuple[float, float]]:
    """Build deterministic, finite, distinct 2D coordinates for a canonical fixture."""
    if name == "path_10":
        return [(float(i), 0.25 * math.sin(0.7 * float(i))) for i in range(n)]
    if name == "cycle_12":
        return [
            (math.cos(2.0 * math.pi * i / n), math.sin(2.0 * math.pi * i / n))
            for i in range(n)
        ]
    if name == "barbell_2comm_9":
        coords: list[tuple[float, float]] = []
        for i in range(n):
            if i < 4:
                ang = i * math.pi / 2.0
                coords.append((-2.5 + 0.6 * math.cos(ang), 0.6 * math.sin(ang)))
            elif i == 4:
                coords.append((0.0, 0.25))
            else:
                ang = (i - 5) * math.pi / 2.0 + 0.35
                coords.append((2.5 + 0.6 * math.cos(ang), 0.6 * math.sin(ang)))
        return coords
    if name == "star_11":
        coords = [(0.0, 0.0)]
        for i in range(1, n):
            ang = 2.0 * math.pi * (i - 1) / max(n - 1, 1)
            coords.append((math.cos(ang), math.sin(ang)))
        return coords
    if name == "disconnected_10":
        return [
            (
                1.5 * float(i % 4) + 0.2 * math.cos(float(i)),
                1.5 * float(i // 4) + 0.2 * math.sin(float(i)),
            )
            for i in range(n)
        ]
    raise ValueError(f"Unknown fixture name: {name}")


def make_similarity_coords(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Apply a deterministic similarity transform (rotation + uniform scale + translation)."""
    theta = 0.7
    scale = 2.5
    c = math.cos(theta)
    s = math.sin(theta)
    return [
        (scale * (c * x - s * y) + 4.0, scale * (s * x + c * y) - 3.0)
        for x, y in coords
    ]


def make_adjusted_coords(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Apply a deterministic similarity transform plus non-rigid perturbation for PRISM checks."""
    theta = 0.4
    scale = 1.35
    c = math.cos(theta)
    s = math.sin(theta)
    return [
        (
            scale * (c * x - s * y) + 2.0 + 0.08 * math.sin(1.3 * float(idx)),
            scale * (s * x + c * y) - 1.5 - 0.06 * math.cos(0.9 * float(idx)),
        )
        for idx, (x, y) in enumerate(coords)
    ]


def _extract_coords_legacy(drawing: Any, n: int) -> list[tuple[float, float]]:
    return [(float(drawing.x(i)), float(drawing.y(i))) for i in range(n)]


def _assert_finite_coords(coords: list[tuple[float, float]], label: str) -> None:
    for idx, (x, y) in enumerate(coords):
        if not (math.isfinite(x) and math.isfinite(y)):
            raise AssertionError(f"{label}: non-finite coord at node {idx}: ({x}, {y})")


def _assert_non_panic_error(
    label: str,
    fn: Callable[[], Any],
    expected_types: tuple[type[BaseException], ...] = (ValueError, TypeError, IndexError),
) -> None:
    caught = False
    try:
        fn()
    except expected_types as exc:
        if "PanicException" in type(exc).__name__:
            raise AssertionError(f"{label} raised PanicException: {exc}") from exc
        caught = True
    except BaseException as exc:
        if "PanicException" in type(exc).__name__:
            raise AssertionError(f"{label} raised PanicException: {exc}") from exc
        raise AssertionError(f"{label} raised unexpected {type(exc).__name__}: {exc}") from exc
    assert caught, f"{label} silently accepted invalid input"


def run_legacy_python_suite(eg: Any) -> list[dict[str, Any]]:
    """Run deterministic & adversarial checks against the current v6-alpha bindings."""
    results: list[dict[str, Any]] = []
    fixtures = make_fixtures()

    def build_graph(spec: dict[str, Any]) -> Any:
        g = eg.Graph()
        nodes = [g.add_node(i) for i in range(spec["n"])]
        for u, v in spec["edges"]:
            g.add_edge(nodes[u], nodes[v], 1.0)
        return g

    # 1. Deterministic PivotMds, FullSgd, SparseSgd, FitTsNet, GuMap + metrics
    for name, spec in fixtures.items():
        t0 = time.perf_counter()
        g = build_graph(spec)
        n = spec["n"]

        # PivotMds determinism & finite coordinates (including disconnected_10!)
        pivots = list(range(min(n, 4)))
        p1 = eg.PivotMds(g, lambda _: 1.0, pivots).run_2d()
        p2 = eg.PivotMds(g, lambda _: 1.0, pivots).run_2d()
        c1 = _extract_coords_legacy(p1, n)
        c2 = _extract_coords_legacy(p2, n)
        _assert_finite_coords(c1, f"PivotMds/{name}")
        assert c1 == c2, f"PivotMds non-deterministic on {name}"

        # FullSgd determinism
        d_sgd1 = eg.DrawingEuclidean2d.initial_placement(g)
        d_sgd2 = eg.DrawingEuclidean2d.initial_placement(g)
        s1 = eg.FullSgd().build(g, lambda _: 1.0)
        s2 = eg.FullSgd().build(g, lambda _: 1.0)
        r1 = eg.Rng.seed_from(42)
        r2 = eg.Rng.seed_from(42)
        eg.SchedulerExponential(20, 0.1).run(lambda eta: (s1.shuffle(r1), s1.apply(d_sgd1, eta)))
        eg.SchedulerExponential(20, 0.1).run(lambda eta: (s2.shuffle(r2), s2.apply(d_sgd2, eta)))
        cs1 = _extract_coords_legacy(d_sgd1, n)
        cs2 = _extract_coords_legacy(d_sgd2, n)
        _assert_finite_coords(cs1, f"FullSgd/{name}")
        assert cs1 == cs2, f"FullSgd non-deterministic on {name}"

        # Quality metrics finite check
        np1 = float(eg.neighborhood_preservation(g, d_sgd1))
        np2 = float(eg.neighborhood_preservation_2hop(g, d_sgd1))
        cr = float(eg.crossing_number(g, d_sgd1))
        assert math.isfinite(np1) and 0.0 <= np1 <= 1.0, f"Invalid NP1 on {name}: {np1}"
        assert math.isfinite(np2) and 0.0 <= np2 <= 1.0, f"Invalid NP2 on {name}: {np2}"
        assert math.isfinite(cr) and cr >= 0.0, f"Invalid crossings on {name}: {cr}"

        results.append({
            "case": f"legacy_core/{name}",
            "status": "pass",
            "elapsed_sec": round(time.perf_counter() - t0, 4),
            "np1": round(np1, 4),
            "np2": round(np2, 4),
        })

    # 2. FitTsNet & GuMap deterministic smoke on barbell_2comm_9
    t0 = time.perf_counter()
    g_bb = build_graph(fixtures["barbell_2comm_9"])
    dm_bb = eg.all_sources_bfs(g_bb, 1.0)
    d_fit = eg.DrawingEuclidean2d.initial_placement(g_bb)
    fit = (
        eg.FitTsNetBuilder()
        .perplexity(2.0)
        .iterations_stage1(10)
        .iterations_stage2(10)
        .iterations_stage3(10)
        .build()
    )
    fit.run(d_fit, g_bb, eg.Rng.seed_from(42))
    _assert_finite_coords(_extract_coords_legacy(d_fit, 9), "FitTsNet/barbell")

    d_umap = eg.DrawingEuclidean2d.initial_placement(g_bb)
    gumap = eg.GuMapBuilder().k(4).n_epochs(40).build()
    gumap.run(d_umap, dm_bb, eg.Rng.seed_from(42))
    _assert_finite_coords(_extract_coords_legacy(d_umap, 9), "GuMap/barbell")
    results.append({
        "case": "legacy_tsnet_gumap/barbell_2comm_9",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
    })

    # 3. Adversarial input checks (must raise ValueError/TypeError/IndexError, never PanicException)
    t0 = time.perf_counter()
    d_bad = eg.DrawingEuclidean2d.initial_placement(g_bb)
    d_bad.set_x(0, float("nan"))
    try:
        eg.crossing_number(g_bb, d_bad)
    except Exception as exc:
        if "PanicException" in type(exc).__name__:
            raise AssertionError(f"crossing_number panicked on NaN input: {exc}") from exc

    for label, fn in (
        ("PivotMds/nan_edge_length", lambda: eg.PivotMds(g_bb, lambda _: float("nan"), [0])),
        ("PivotMds/oob_pivot", lambda: eg.PivotMds(g_bb, lambda _: 1.0, [999])),
    ):
        _assert_non_panic_error(label, fn)

    results.append({
        "case": "adversarial_nan_guard",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
    })

    return results


def run_v6_python_suite(eg: Any) -> list[dict[str, Any]]:
    """Run deterministic, zero-copy & adversarial checks on v6 CsrGraph/Layout2d."""
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
    _assert_non_panic_error(
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
        "case": "v6_csr_from_csr",
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
        "case": "v6_csr_from_edges_and_lcc",
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
            "case": f"v6_layout2d_zerocopy_{dtype_name}",
            "status": "pass",
            "elapsed_sec": round(time.perf_counter() - t0, 4),
        })

    # 4. Adversarial v6 inputs (must raise ValueError/TypeError/IndexError, never PanicException)
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
        _assert_non_panic_error(label, fn)

    try:
        l_nan = eg.Layout2d.from_coords(np.array([[float("nan"), 0.0]], dtype=np.float64))
        assert l_nan.is_all_finite() is False
    except (ValueError, TypeError, IndexError) as exc:
        if "PanicException" in type(exc).__name__:
            raise AssertionError(f"Layout2d.from_coords/nan raised PanicException: {exc}") from exc

    results.append({
        "case": "v6_adversarial_non_panic",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
    })

    # 5. M1 metrics & spatial checks (`float64` SNS scale invariance, evaluate_metrics, PRISM, Delaunay)
    if hasattr(eg, "scale_normalized_stress"):
        t0 = time.perf_counter()
        m1_targets = [
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
            assert abs(np1 - float(eg.neighborhood_preservation(fg, l64))) < 1e-12
            np2 = float(eg.neighborhood_preservation_2hop_v6(fg, l64))
            assert abs(np2 - float(eg.neighborhood_preservation_2hop(fg, l64))) < 1e-12
            assert 0.0 <= np1 <= 1.0 and 0.0 <= np2 <= 1.0

            sig_dist_sim, sig_disp_sim = eg.prism_distortion(l64, l64_sim)
            assert sig_dist_sim < 1e-10 and sig_disp_sim < 1e-10, (
                f"{name}: similarity prism_distortion nonzero: {(sig_dist_sim, sig_disp_sim)}"
            )
            sig_dist_adj, sig_disp_adj = eg.prism_distortion(l64, l64_adj)
            assert math.isfinite(sig_dist_adj) and sig_dist_adj >= 0.0
            assert math.isfinite(sig_disp_adj) and 0.0 <= sig_disp_adj <= 1.0

            report = eg.evaluate_metrics(
                fg, l64, m1_targets, num_pivots=4, num_samples=200, seed=42
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
            assert abs(report["aspect_ratio"] - float(eg.aspect_ratio_v6(l64))) < 1e-12

            dt_edges, dt_tris = eg.delaunay_triangulation(l64)
            assert dt_edges.dtype == np.uint32 and dt_edges.ndim == 2 and dt_edges.shape[1] == 2
            assert dt_tris.dtype == np.uint32 and dt_tris.ndim == 2 and dt_tris.shape[1] == 3
            assert dt_edges.shape[0] > 0 and dt_tris.shape[0] > 0
            assert int(dt_edges.max()) < n and int(dt_tris.max()) < n
            dt_edges32, dt_tris32 = eg.delaunay_triangulation(l32)
            assert dt_edges32.shape[0] > 0 and dt_tris32.shape[0] > 0

        results.append({
            "case": "v6_m1_metrics_and_spatial",
            "status": "pass",
            "elapsed_sec": round(time.perf_counter() - t0, 4),
        })

        # M1 adversarial error checks (must raise clean exceptions, never PanicException)
        t0 = time.perf_counter()
        g_p = eg.CsrGraph.from_edges(4, [(0, 1), (1, 2), (2, 3)], directed=False, sanitize=True)
        l_ok = eg.Layout2d.from_coords([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        l_short = eg.Layout2d.zeros(3)
        l_nan = eg.Layout2d.from_coords(l_ok.coords(copy=True))
        l_nan.coords(copy=False)[1, 0] = float("nan")

        def _bind_call(
            metric_fn: Callable[[Any, Any], Any], graph_arg: Any, layout_arg: Any
        ) -> Callable[[], Any]:
            return lambda: metric_fn(graph_arg, layout_arg)

        m1_fns: tuple[tuple[str, Callable[[Any, Any], Any]], ...] = (
            ("scale_normalized_stress", eg.scale_normalized_stress),
            ("sweep_crossing_number", eg.sweep_crossing_number),
            ("neighborhood_preservation_1hop", eg.neighborhood_preservation_1hop),
            ("neighborhood_preservation_2hop_v6", eg.neighborhood_preservation_2hop_v6),
        )
        for fn_name, m1_fn in m1_fns:
            _assert_non_panic_error(
                f"{fn_name}/dim_mismatch", _bind_call(m1_fn, g_p, l_short), expected_types=(ValueError,)
            )
            _assert_non_panic_error(
                f"{fn_name}/nan", _bind_call(m1_fn, g_p, l_nan), expected_types=(ValueError,)
            )

        _assert_non_panic_error(
            "prism_distortion/dim_mismatch",
            lambda: eg.prism_distortion(l_ok, l_short),
            expected_types=(ValueError,),
        )
        _assert_non_panic_error(
            "prism_distortion/nan",
            lambda: eg.prism_distortion(l_ok, l_nan),
            expected_types=(ValueError,),
        )
        _assert_non_panic_error(
            "delaunay_triangulation/nan",
            lambda: eg.delaunay_triangulation(l_nan),
            expected_types=(ValueError,),
        )
        _assert_non_panic_error(
            "evaluate_metrics/nan",
            lambda: eg.evaluate_metrics(g_p, l_nan, ["sns"]),
            expected_types=(ValueError,),
        )
        _assert_non_panic_error(
            "evaluate_metrics/non_list_targets",
            lambda: eg.evaluate_metrics(g_p, l_ok, "sns"),
            expected_types=(TypeError,),
        )
        _assert_non_panic_error(
            "evaluate_metrics/non_str_target",
            lambda: eg.evaluate_metrics(g_p, l_ok, [123]),
            expected_types=(TypeError,),
        )
        _assert_non_panic_error(
            "evaluate_metrics/unknown_target",
            lambda: eg.evaluate_metrics(g_p, l_ok, ["unknown_metric"]),
            expected_types=(ValueError,),
        )
        _assert_non_panic_error(
            "sampled_scale_normalized_stress/zero_pivots",
            lambda: eg.sampled_scale_normalized_stress(g_p, l_ok, num_pivots=0),
            expected_types=(ValueError,),
        )
        _assert_non_panic_error(
            "sampled_scale_normalized_stress/neg_seed",
            lambda: eg.sampled_scale_normalized_stress(g_p, l_ok, seed=-1),
            expected_types=(ValueError,),
        )
        _assert_non_panic_error(
            "sampled_crossing_number/zero_samples",
            lambda: eg.sampled_crossing_number(g_p, l_ok, num_samples=0),
            expected_types=(ValueError,),
        )
        results.append({
            "case": "v6_m1_adversarial_non_panic",
            "status": "pass",
            "elapsed_sec": round(time.perf_counter() - t0, 4),
        })

    return results


def run_wasm_node_parity(wasm_pkg: pathlib.Path) -> list[dict[str, Any]]:
    """Verify Node.js ESM import, CsrGraph/Layout2d coordsView(), and JsError guard."""
    t0 = time.perf_counter()
    node_entry = wasm_pkg / "egraph_wasm.node.js"
    index_entry = wasm_pkg / "index.js"
    if not index_entry.exists():
        return [{
            "case": "wasm_node_esm_parity",
            "status": "skipped_no_v6_index_js",
            "elapsed_sec": 0.0,
        }]

    node_url = node_entry.resolve().as_uri() if node_entry.exists() else index_entry.resolve().as_uri()
    index_url = index_entry.resolve().as_uri()
    wasm_path = (wasm_pkg / "egraph_wasm_bg.wasm").resolve().as_posix()

    node_script = f"""
import {{ readFile }} from 'node:fs/promises';
import assert from 'node:assert/strict';

const mod = await import({json.dumps(node_url)});
const indexMod = await import({json.dumps(index_url)});
if (typeof mod.default === 'function') {{
  try {{
    // Try direct instantiation first; fallback to explicit bytes if needed.
    const testGraph = mod.CsrGraph.fromCsr(new Uint32Array([0, 1, 2]), new Uint32Array([1, 0]));
    testGraph.free?.();
  }} catch (_e) {{
    const bytes = await readFile({json.dumps(wasm_path)});
    await mod.default(bytes);
  }}
}}

assert.equal(typeof indexMod.CsrGraph, 'function', 'index.js must export CsrGraph');
assert.equal(typeof indexMod.Layout2d, 'function', 'index.js must export Layout2d');
assert.equal(typeof indexMod.Layout2d.prototype.coordsView, 'function', 'Layout2d must expose coordsView()');

const {{ CsrGraph, Layout2d }} = mod;
const g = CsrGraph.fromCsr(new Uint32Array([0, 1, 3, 4]), new Uint32Array([1, 0, 2, 1]));
const numNodes = typeof g.numNodes === 'function' ? g.numNodes() : g.numNodes;
const numEdges = typeof g.numEdges === 'function' ? g.numEdges() : g.numEdges;
const isDirected = typeof g.isDirected === 'function' ? g.isDirected() : g.isDirected;
assert.equal(numNodes, 3);
assert.equal(numEdges, 2);
assert.equal(isDirected, false);
g.free?.();

const layout = Layout2d.zeros(4);
assert.equal(typeof layout.coordsView, 'function');
const view = layout.coordsView();
assert.ok(view instanceof Float32Array, 'coordsView() must return Float32Array');
assert.equal(view.length, 8);
view[2] = 3.25;
view[3] = -1.5;
assert.ok(Math.abs(layout.x(1) - 3.25) < 1e-6);
assert.ok(Math.abs(layout.y(1) - (-1.5)) < 1e-6);
layout.setX(0, 7.5);
assert.ok(Math.abs(layout.coordsView()[0] - 7.5) < 1e-6);
assert.equal(layout.isAllFinite(), true);

let threwCsr = false;
try {{
  CsrGraph.fromCsr(new Uint32Array([0, 1]), new Uint32Array([99]));
}} catch (err) {{
  assert.ok(!(err instanceof WebAssembly.RuntimeError), 'CSR error must map to JsError, not RuntimeError');
  threwCsr = true;
}}
assert.ok(threwCsr, 'CsrGraph.fromCsr must reject out-of-bounds vertex index');

let threwLayout = false;
try {{
  layout.x(99);
}} catch (err) {{
  assert.ok(!(err instanceof WebAssembly.RuntimeError), 'Layout2d error must map to JsError, not RuntimeError');
  threwLayout = true;
}}
assert.ok(threwLayout, 'Layout2d.x must reject out-of-bounds index');
layout.free?.();

if (typeof mod.scaleNormalizedStress === 'function') {{
  assert.equal(typeof indexMod.scaleNormalizedStress, 'function');
  assert.equal(typeof indexMod.sweepCrossingNumber, 'function');
  assert.equal(typeof indexMod.neighborhoodPreservation1Hop, 'function');
  assert.equal(typeof indexMod.neighborhoodPreservation2HopV6, 'function');
  assert.equal(typeof indexMod.prismDistortion, 'function');
  assert.equal(typeof indexMod.evaluateMetrics, 'function');
  assert.equal(typeof indexMod.MetricReport, 'function');
  assert.equal(typeof indexMod.PrismDistortion, 'function');

  const bbEdges = new Uint32Array([
    0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3, 3, 4, 4, 5,
    5, 6, 5, 7, 5, 8, 6, 7, 6, 8, 7, 8,
  ]);
  const bbCoords = new Float32Array([
    -1.9, 0.0, -2.5, 0.6, -3.1, 0.0, -2.5, -0.6,
    0.0, 0.25,
    3.063, 0.206, 2.294, 0.563, 1.937, -0.206, 2.706, -0.563,
  ]);
  const gBb = CsrGraph.fromEdgeList(9, bbEdges);
  const lBb = Layout2d.fromCoords(bbCoords);
  const lBbScaled = Layout2d.fromCoords(bbCoords.map((v) => v * 100.0));
  const lBbSmall = Layout2d.fromCoords(bbCoords.map((v) => v * 0.01));

  const sns1 = mod.scaleNormalizedStress(gBb, lBb);
  const sns100 = mod.scaleNormalizedStress(gBb, lBbScaled);
  const sns001 = mod.scaleNormalizedStress(gBb, lBbSmall);
  assert.ok(Number.isFinite(sns1) && sns1 >= 0.0 && sns1 <= 1.0);
  assert.ok(Math.abs(sns100 - sns1) < 1e-5, 'WASM SNS must be scale-invariant (100x)');
  assert.ok(Math.abs(sns001 - sns1) < 1e-5, 'WASM SNS must be scale-invariant (0.01x)');

  const cr = mod.sweepCrossingNumber(gBb, lBb);
  const np1 = mod.neighborhoodPreservation1Hop(gBb, lBb);
  const np2 = mod.neighborhoodPreservation2HopV6(gBb, lBb);
  assert.ok(Number.isInteger(cr) && cr >= 0);
  assert.ok(Number.isFinite(np1) && np1 >= 0.0 && np1 <= 1.0);
  assert.ok(Number.isFinite(np2) && np2 >= 0.0 && np2 <= 1.0);

  const pd = mod.prismDistortion(lBb, lBbScaled);
  assert.ok(pd.sigmaDist < 1e-5 && pd.sigmaDisp < 1e-5);
  pd.free?.();

  const rep = mod.evaluateMetrics(gBb, lBb, ['sns', 'crossingNumber', 'np1', 'np2'], 4, 200, 42);
  const rec = rep.toRecord();
  assert.deepEqual(
    Object.keys(rec).sort(),
    ['crossingNumber', 'neighborhoodPreservation1Hop', 'neighborhoodPreservation2Hop', 'scaleNormalizedStress'],
  );
  assert.ok(Math.abs(rec.scaleNormalizedStress - sns1) < 1e-12);
  assert.equal(rec.crossingNumber, cr);
  assert.ok(Math.abs(rec.neighborhoodPreservation1Hop - np1) < 1e-12);
  assert.ok(Math.abs(rec.neighborhoodPreservation2Hop - np2) < 1e-12);
  assert.equal(rep.aspectRatio, undefined);
  rep.free?.();

  if (typeof mod.delaunayTriangulation === 'function') {{
    const dt = mod.delaunayTriangulation(lBb);
    assert.ok(dt.edges() instanceof Uint32Array && dt.edges().length > 0);
    assert.ok(dt.triangles() instanceof Uint32Array && dt.triangles().length > 0);
    dt.free?.();
  }}

  const lShort = Layout2d.zeros(3);
  const badCoords = new Float32Array(bbCoords);
  badCoords[0] = Number.NaN;
  const lNan = Layout2d.fromCoords(badCoords);
  const assertJsErr = (fn, label) => {{
    let threw = false;
    try {{
      fn();
    }} catch (err) {{
      assert.ok(!(err instanceof WebAssembly.RuntimeError), `${{label}} trapped with RuntimeError`);
      assert.ok(err instanceof Error, `${{label}} must throw Error`);
      threw = true;
    }}
    assert.ok(threw, `${{label}} must throw`);
  }};
  assertJsErr(() => mod.scaleNormalizedStress(gBb, lShort), 'm1 sns dim mismatch');
  assertJsErr(() => mod.sweepCrossingNumber(gBb, lNan), 'm1 sweep nan');
  assertJsErr(() => mod.evaluateMetrics(gBb, lBb, 'sns'), 'm1 bad targets');
  assertJsErr(() => mod.sampledScaleNormalizedStress(gBb, lBb, 0, 42), 'm1 zero pivots');
  assertJsErr(() => mod.sampledScaleNormalizedStress(gBb, lBb, 4, -1), 'm1 neg seed');

  lShort.free?.();
  lNan.free?.();
  gBb.free?.();
  lBb.free?.();
  lBbScaled.free?.();
  lBbSmall.free?.();
}}

console.log('WASM_NODE_PARITY_OK');
"""
    subprocess.check_call(["node", "--input-type=module", "-e", node_script])
    return [{
        "case": "wasm_node_esm_parity",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
    }]


def run_cross_target_m1_parity(eg: Any, wasm_pkg: pathlib.Path) -> list[dict[str, Any]]:
    """Verify Python float32 <-> WASM f32 M1 metric numerical parity (< 1e-4) across canonical fixtures."""
    import numpy as np

    if not hasattr(eg, "scale_normalized_stress"):
        return []
    node_entry = wasm_pkg / "egraph_wasm.node.js"
    index_entry = wasm_pkg / "index.js"
    if not index_entry.exists():
        return []

    t0 = time.perf_counter()
    fixtures = make_fixtures()
    payload: list[dict[str, Any]] = []
    py_results: dict[str, dict[str, Any]] = {}
    targets = [
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
        coords_f32 = np.array(make_fixture_coords(name, n), dtype=np.float32)
        adj_f32 = np.array(
            make_adjusted_coords(make_fixture_coords(name, n)), dtype=np.float32
        )
        l32 = eg.Layout2d.from_coords(coords_f32)
        l32_adj = eg.Layout2d.from_coords(adj_f32)

        cr_dual = float(eg.crossing_number(fg, l32))
        cr_sweep = float(eg.sweep_crossing_number(fg, l32))
        assert cr_dual == cr_sweep, f"{name}: Python crossing_number != sweep_crossing_number"

        sig_dist, sig_disp = eg.prism_distortion(l32, l32_adj)
        report = eg.evaluate_metrics(
            fg, l32, targets, num_pivots=4, num_samples=200, seed=42
        )
        dt_edges, dt_tris = eg.delaunay_triangulation(l32)

        py_results[name] = {
            "scale_normalized_stress": float(eg.scale_normalized_stress(fg, l32)),
            "crossing_number": cr_dual,
            "sweep_crossing_number": cr_sweep,
            "neighborhood_preservation_1hop": float(eg.neighborhood_preservation_1hop(fg, l32)),
            "neighborhood_preservation_2hop_v6": float(eg.neighborhood_preservation_2hop_v6(fg, l32)),
            "prism_sigma_dist": float(sig_dist),
            "prism_sigma_disp": float(sig_disp),
            "evaluate_metrics": {k: float(v) for k, v in report.items()},
            "delaunay_edge_count": int(dt_edges.shape[0]),
            "delaunay_tri_count": int(dt_tris.shape[0]),
        }
        payload.append({
            "name": name,
            "n": n,
            "edges": [int(v) for edge in spec["edges"] for v in edge],
            "coords": [float(v) for v in coords_f32.ravel()],
            "adj_coords": [float(v) for v in adj_f32.ravel()],
        })

    node_url = node_entry.resolve().as_uri() if node_entry.exists() else index_entry.resolve().as_uri()
    wasm_path = (wasm_pkg / "egraph_wasm_bg.wasm").resolve().as_posix()

    node_script = """
import { readFile } from 'node:fs/promises';

const nodeUrl = process.argv[1];
const wasmPath = process.argv[2];
const cases = JSON.parse(process.argv[3]);

const mod = await import(nodeUrl);
if (typeof mod.default === 'function') {
  try {
    const probe = mod.CsrGraph.fromCsr(new Uint32Array([0, 1, 2]), new Uint32Array([1, 0]));
    probe.free?.();
  } catch (_e) {
    await mod.default(await readFile(wasmPath));
  }
}

if (typeof mod.scaleNormalizedStress !== 'function') {
  console.log(JSON.stringify({ m1_available: false, cases: {} }));
  process.exit(0);
}

const out = {};
for (const c of cases) {
  const g = mod.CsrGraph.fromEdgeList(c.n, new Uint32Array(c.edges));
  const l = mod.Layout2d.fromCoords(new Float32Array(c.coords));
  const lAdj = mod.Layout2d.fromCoords(new Float32Array(c.adj_coords));

  const pd = mod.prismDistortion(l, lAdj);
  const rep = mod.evaluateMetrics(
    g,
    l,
    ['sns', 'crossing_number', 'np1', 'np2', 'drng', 'dgg', 'aspect_ratio'],
    4,
    200,
    42,
  );
  const rec = rep.toRecord();

  let delaunayCounts = null;
  if (typeof mod.delaunayTriangulation === 'function') {
    const dt = mod.delaunayTriangulation(l);
    delaunayCounts = {
      edges: dt.edges().length / 2,
      tris: dt.triangles().length / 3,
    };
    dt.free?.();
  }

  out[c.name] = {
    scale_normalized_stress: mod.scaleNormalizedStress(g, l),
    sweep_crossing_number: mod.sweepCrossingNumber(g, l),
    neighborhood_preservation_1hop: mod.neighborhoodPreservation1Hop(g, l),
    neighborhood_preservation_2hop_v6: mod.neighborhoodPreservation2HopV6(g, l),
    prism_sigma_dist: pd.sigmaDist,
    prism_sigma_disp: pd.sigmaDisp,
    evaluate_metrics: {
      scale_normalized_stress: rec.scaleNormalizedStress,
      crossing_number: rec.crossingNumber,
      neighborhood_preservation_1hop: rec.neighborhoodPreservation1Hop,
      neighborhood_preservation_2hop: rec.neighborhoodPreservation2Hop,
      relative_neighborhood_preservation: rec.relativeNeighborhoodPreservation,
      gabriel_graph_preservation: rec.gabrielGraphPreservation,
      aspect_ratio: rec.aspectRatio,
    },
    delaunay_counts: delaunayCounts,
  };

  pd.free?.();
  rep.free?.();
  g.free?.();
  l.free?.();
  lAdj.free?.();
}

console.log(JSON.stringify({ m1_available: true, cases: out }));
"""
    raw_out = subprocess.check_output(
        ["node", "--input-type=module", "-e", node_script, node_url, wasm_path, json.dumps(payload)],
        text=True,
    )
    wasm_payload = json.loads(raw_out.strip().splitlines()[-1])
    if not wasm_payload.get("m1_available", False):
        return [{
            "case": "cross_target_m1_py_vs_wasm",
            "status": "skipped_no_wasm_m1",
            "elapsed_sec": round(time.perf_counter() - t0, 4),
        }]

    wasm_cases = wasm_payload["cases"]
    tol = 1e-4
    max_diff = 0.0
    for name in fixtures:
        py_r = py_results[name]
        ws_r = wasm_cases[name]

        for py_key, ws_key in (
            ("scale_normalized_stress", "scale_normalized_stress"),
            ("crossing_number", "sweep_crossing_number"),
            ("sweep_crossing_number", "sweep_crossing_number"),
            ("neighborhood_preservation_1hop", "neighborhood_preservation_1hop"),
            ("neighborhood_preservation_2hop_v6", "neighborhood_preservation_2hop_v6"),
            ("prism_sigma_dist", "prism_sigma_dist"),
            ("prism_sigma_disp", "prism_sigma_disp"),
        ):
            diff = abs(float(py_r[py_key]) - float(ws_r[ws_key]))
            max_diff = max(max_diff, diff)
            assert diff < tol, (
                f"{name}: cross-target {py_key} mismatch py={py_r[py_key]} wasm={ws_r[ws_key]} diff={diff}"
            )

        for mk, py_mv in py_r["evaluate_metrics"].items():
            ws_mv = float(ws_r["evaluate_metrics"][mk])
            diff = abs(float(py_mv) - ws_mv)
            max_diff = max(max_diff, diff)
            assert diff < tol, (
                f"{name}: evaluate_metrics[{mk}] mismatch py={py_mv} wasm={ws_mv} diff={diff}"
            )

        if ws_r["delaunay_counts"] is not None:
            assert py_r["delaunay_edge_count"] == ws_r["delaunay_counts"]["edges"], (
                f"{name}: Delaunay edge count mismatch py={py_r['delaunay_edge_count']} wasm={ws_r['delaunay_counts']['edges']}"
            )
            assert py_r["delaunay_tri_count"] == ws_r["delaunay_counts"]["tris"], (
                f"{name}: Delaunay tri count mismatch py={py_r['delaunay_tri_count']} wasm={ws_r['delaunay_counts']['tris']}"
            )

    return [{
        "case": "cross_target_m1_py_vs_wasm",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "max_diff": round(max_diff, 12),
    }]


def main() -> int:
    parser = argparse.ArgumentParser(description="egraph cross-target parity harness")
    parser.add_argument("--wasm-pkg", type=pathlib.Path, default=None, help="Path to built v6 WASM pkg/")
    parser.add_argument("--wasm-only", action="store_true", help="Skip Python checks and run only WASM Node checks")
    parser.add_argument("--json", type=pathlib.Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    eg_mod: Any = None
    if not args.wasm_only:
        import egraph as eg

        eg_mod = eg
        if hasattr(eg, "CsrGraph") and hasattr(eg, "Layout2d"):
            results.extend(run_v6_python_suite(eg))
        if hasattr(eg, "Graph") and hasattr(eg, "PivotMds"):
            results.extend(run_legacy_python_suite(eg))

    if args.wasm_pkg is not None:
        results.extend(run_wasm_node_parity(args.wasm_pkg))
        if eg_mod is not None:
            results.extend(run_cross_target_m1_parity(eg_mod, args.wasm_pkg))

    print(f"{'CASE':<40} {'STATUS':<24} {'ELAPSED(s)':>10}")
    print("-" * 78)
    for row in results:
        print(f"{row['case']:<40} {row['status']:<24} {row['elapsed_sec']:>10.4f}")

    if args.json is not None:
        args.json.write_text(json.dumps({"results": results}, indent=2) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
