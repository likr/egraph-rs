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

    # 6. M2 linalg & neighborhood checks
    if hasattr(eg, "select_maxmin_sp_pivots"):
        results.extend(run_v6_m2_python_checks(eg))

    # 7. M3 clustering, ManifoldLayout2d & layout solver checks
    if hasattr(eg, "TsNetSolver"):
        results.extend(run_v6_m3_python_checks(eg))

    # 8. M4 unified force models, constraints, layering & FDEB checks
    if hasattr(eg, "TFdpSolver"):
        results.extend(run_v6_m4_python_checks(eg))

    # 9. G-CAF (Cluster Area Faithfulness) Python checks
    if hasattr(eg, "cluster_area_faithfulness"):
        results.extend(_run_v6_gcaf_python_checks(eg))

    assert not hasattr(eg, "InfoMap"), "InfoMap must not be exposed on egraph v6"
    assert "InfoMap" not in getattr(eg, "__all__", []), "InfoMap must not be in egraph.__all__"

    return results


def _gcaf_small_n_regression_cases() -> list[dict[str, Any]]:
    """Return exact f32 N <= 40 cross-target G-CAF parity regression fixtures."""
    n30_k3_coords = [
        -0.2815477252006531, 2.700129985809326, 1.5668566226959229, 0.07951676100492477,
        -1.7507059574127197, 3.2445662021636963, -0.7271009683609009, -1.1585856676101685,
        -0.9049389362335205, -0.01628471538424492, 0.6466772556304932, 0.3806455433368683,
        -0.6412783861160278, 1.3978006839752197, -0.4455091059207916, -0.49050599336624146,
        2.5104165077209473, 1.2259844541549683, 0.43586039543151855, -1.875908374786377,
        -2.004007339477539, 1.1233396530151367, 1.0720340013504028, 0.9364057779312134,
        0.1474851369857788, -0.5821631550788879, -1.3126808404922485, 2.892348527908325,
        1.1164329051971436, -2.457968235015869, -0.7849741578102112, -4.539268970489502,
        -1.1620724201202393, -0.7121404409408569, -0.565313458442688, -0.7346477508544922,
        -2.0997090339660645, -0.09313655644655228, 1.1614314317703247, 0.3544427454471588,
        0.4652692377567291, 1.7718443870544434, -0.44344475865364075, 1.4617308378219604,
        -1.6758195161819458, 1.5955342054367065, -3.2307612895965576, -0.6647124886512756,
        0.012851242907345295, 1.9697237014770508, 0.2668476700782776, 3.0945630073547363,
        -0.838649570941925, 0.16661247611045837, 2.785959482192993, -2.0786988735198975,
        -3.658062219619751, 2.8940680027008057, 4.266763687133789, 0.9111690521240234,
    ]
    n30_k3_labels = [2, 1, 2, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1]
    n39_k5_coords = [
        2.3179290294647217, -2.3773701190948486, -2.518195390701294, 2.2936649322509766,
        1.515089750289917, -0.1626361757516861, -0.5416049957275391, 1.4072234630584717,
        -2.659836530685425, 1.2367346286773682, 0.230801060795784, 2.8906409740448,
        0.14955002069473267, -2.741164445877075, 2.9515316486358643, 0.5958318710327148,
        -2.25287127494812, -1.281233549118042, -4.208866119384766, -2.057319402694702,
        1.240214228630066, -0.645514965057373, -3.716273784637451, -1.508569359779358,
        0.5517861247062683, 0.43332743644714355, -2.085688829421997, 0.824012041091919,
        1.1661124229431152, 0.5285953283309937, -1.7103712558746338, 0.9139032363891602,
        -1.189815878868103, -2.2106411457061768, 2.106105089187622, 1.6581815481185913,
        1.8313655853271484, -5.026828289031982, 0.15985800325870514, 2.3602962493896484,
        1.1456916332244873, -0.6920322775840759, -1.9657926559448242, -1.7649825811386108,
        -2.4530982971191406, -2.5909652709960938, 0.3976938724517822, 0.16761969029903412,
        -3.0827178955078125, -0.32256436347961426, -3.298811674118042, 0.3545989990234375,
        -2.273881196975708, 0.9519124031066895, -0.3883000612258911, -1.580034613609314,
        -0.7317273616790771, 2.549612522125244, 0.18937981128692627, 2.016829013824463,
        -0.643775999546051, -1.1126575469970703, -1.6523631811141968, -1.4371283054351807,
        1.3997529745101929, -1.374435544013977, 0.34608787298202515, -0.20996540784835815,
        0.8923608064651489, -0.3378030061721802, 0.48425641655921936, -1.9588091373443604,
        2.5424654483795166, 0.02001159079372883, 1.4568390846252441, -3.306309938430786,
        -2.286395311355591, -0.16748563945293427,
    ]
    n39_k5_labels = [3, 0, 0, 1, 4, 3, 0, 3, 1, 3, 3, 2, 2, 0, 0, 3, 4, 2, 0, 3, 0, 1, 1, 4, 1, 1, 3, 1, 0, 3, 1, 2, 2, 3, 2, 3, 3, 2, 1]
    n40_b1_coords = [
        -2.9233508110046387, 1.3258304595947266, -1.270039677619934, 0.05569106712937355,
        -3.5743446350097656, 0.08754363656044006, 2.8365554809570312, -1.2860935926437378,
        0.6736948490142822, 2.1501173973083496, 0.02794797532260418, -0.5190643072128296,
        1.8806356191635132, 0.3827267289161682, 0.594303548336029, -5.774633884429932,
        1.3010855913162231, 0.15634697675704956, -0.1997925490140915, 2.864908218383789,
        -0.4877355992794037, -0.3527091443538666, 1.1604030132293701, 1.5458732843399048,
        -3.2697393894195557, -1.7934491634368896, -1.9938316345214844, 0.7835420966148376,
        2.9356517791748047, -1.301081895828247, 2.102522373199463, 3.261350393295288,
        -3.1106839179992676, -3.077613353729248, 4.413053035736084, 1.5375747680664062,
        1.0798331499099731, -0.5608461499214172, -0.24963510036468506, 0.9973931312561035,
        2.0195343494415283, -1.7061171531677246, -3.577021360397339, 3.366581916809082,
        1.3103059530258179, -1.3890607357025146, 0.760463535785675, 0.3916463255882263,
        1.9436023235321045, 1.0128681659698486, -0.8362287878990173, 3.4891979694366455,
        -2.9951605796813965, 2.4798998832702637, -0.43555063009262085, 1.482376217842102,
        3.8454504013061523, -0.4614332914352417, -4.027052879333496, -0.38336920738220215,
        -2.1162023544311523, 2.383676290512085, 2.38696551322937, -1.7980482578277588,
        1.8000229597091675, 1.4273918867111206, -2.7043612003326416, -2.191859006881714,
        1.9361793994903564, 1.2998770475387573, -0.2770219147205353, 1.5509105920791626,
        1.5726526975631714, -3.3546550273895264, -0.31485745310783386, 2.2452876567840576,
        1.2533442974090576, -2.2307775020599365, -0.25232869386672974, 0.9945555329322815,
    ]
    n40_b1_labels = [3, 1, 3, 0, 1, 2, 2, 2, 1, 0, 0, 0, 2, 3, 2, 1, 0, 1, 2, 1, 0, 3, 1, 1, 1, 0, 0, 2, 1, 3, 0, 3, 2, 2, 0, 0, 1, 1, 0, 1]
    n6_b1_coords = [
        0.09900733083486557, 3.489287853240967, -2.256587505340576, 1.0310328006744385,
        -0.25905266404151917, 0.7601389288902283, 0.6031523942947388, 1.1567120552062988,
        4.331787109375, 2.105045795440674, -2.2524311542510986, 3.557722568511963,
    ]
    n6_b1_labels = [3, 0, 5, 1, 3, 5]
    idx445_pure_tail_coords = [
        -0.933991551399231, -0.08828876167535782, 0.2382437139749527, -1.6165368556976318,
        0.31952330470085144, -1.328585147857666, -0.29861539602279663, 1.6585638523101807,
        -1.2817468643188477, -0.23555676639080048, 0.9802638292312622, -0.26757410168647766,
        2.0682170391082764, 0.8867510557174683, 0.4469839334487915, 1.5724736452102661,
        0.4012336730957031, -1.8822963237762451,
    ]
    idx445_pure_tail_labels = [1, 4, 4, 0, 3, 3, 3, 3, 4]
    idx451_sparse_tail_coords = [
        -0.2518731951713562, 0.07975521683692932, 1.37264883518219, 0.8500950932502747,
        -1.343191385269165, 0.06236932426691055, 0.3976910710334778, 1.9115973711013794,
        -0.9139043092727661, -0.8880517482757568, 1.6514384746551514, -0.22949932515621185,
        -1.1391746997833252, -0.9518010020256042, 0.2050648033618927, -1.6683542728424072,
        -0.25118568539619446, -0.6805249452590942, 0.30096200108528137, 0.10765163600444794,
        -0.11161975562572479, -0.9943410754203796, 0.18235112726688385, -0.6725447177886963,
        -0.5753703713417053, 1.0845589637756348, -0.15376493334770203, 0.6685572862625122,
        -0.7996044158935547, -0.7258471250534058, 1.8715933561325073, -0.4190787076950073,
        -0.7483075857162476, 0.6918076872825623, 0.7984699606895447, 1.896586298942566,
        0.3965696394443512, -0.7518982887268066, -1.6543024778366089, -0.009605116210877895,
        -1.536963939666748, -0.49072691798210144, -1.4884775876998901, 0.6713424921035767,
        0.05031104385852814, 0.05718688666820526, -0.5325454473495483, -0.8697065711021423,
        1.642391324043274, -1.7892390489578247, 1.8272528648376465, 0.6015019416809082,
        0.2155684232711792, -0.2579333782196045, 1.403604507446289, 0.28627896308898926,
        -0.39399221539497375, 0.44464102387428284, 0.3066869080066681, -0.12467768788337708,
    ]
    idx451_sparse_tail_labels = [
        0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1,
    ]
    s500_292_coords = [
        -0.8589298129081726, -0.8472998738288879, -0.010375149548053741, -1.7285854816436768,
        0.001789810019545257, 2.976900577545166, -0.9761431217193604, 1.2507176399230957,
        0.7966066002845764, -0.43772125244140625, -1.9091739654541016, 0.8813920617103577,
        -1.4938889741897583, 0.055759936571121216, -0.35337790846824646, -1.470625877380371,
        0.6511914134025574, -0.35015806555747986, 0.903068482875824, -1.5927823781967163,
        -0.3389033079147339, -1.0626435279846191, -0.25786250829696655, 0.23115427792072296,
        -1.486572504043579, 0.38411444425582886, 0.5594135522842407, -0.8532565236091614,
        -1.2615015506744385, -0.5342092514038086, -0.9444082975387573, 0.579115092754364,
        -0.6351454854011536, -0.967582106590271, -0.5154885053634644, 0.9100424647331238,
        0.9587221741676331, 0.26757296919822693, -0.877332866191864, -1.196655511856079,
        0.7086302042007446, 1.4324191808700562, -1.5779736042022705, -0.42056649923324585,
        -0.9108667969703674, 0.9537004828453064, 0.7653982639312744, 0.9929231405258179,
        -1.488357663154602, 1.0644866228103638, -0.7887485027313232, 0.2259710282087326,
        -0.8795187473297119, -0.9363757967948914, 0.8706657886505127, 0.7357091903686523,
        0.9534659385681152, -1.8786526918411255, 0.2655912935733795, 1.7928541898727417,
        -0.10050357133150101, -0.9006339311599731, 1.3498365879058838, 0.5915183424949646,
        1.129069209098816, -0.5772221088409424, 1.0819532871246338, -0.8615137338638306,
        0.395887166261673, 0.6019364595413208, 0.42082059383392334, -1.8643262386322021,
        0.8065900802612305, 0.916283130645752,
    ]
    s500_292_k5_labels = [2, 4, 2, 4, 3, 0, 4, 1, 4, 1, 1, 0, 1, 0, 0, 3, 2, 0, 1, 3, 1, 3, 3, 0, 4, 4, 0, 1, 2, 4, 4, 0, 3, 3, 4, 3, 4]
    s500_292_k6_labels = [3, 5, 3, 4, 4, 0, 5, 1, 5, 1, 2, 0, 1, 0, 0, 4, 2, 0, 2, 3, 1, 4, 4, 1, 5, 5, 0, 2, 3, 5, 5, 0, 4, 4, 5, 4, 5]
    return [
        {"name": "n30_k3_default", "coords": n30_k3_coords, "labels": n30_k3_labels, "g_max": None, "sigma_cells": None, "boundary_r95": None, "stevens_gamma": None},
        {"name": "n30_k3_custom", "coords": n30_k3_coords, "labels": n30_k3_labels, "g_max": 12, "sigma_cells": 0.15, "boundary_r95": 0.8, "stevens_gamma": None},
        {"name": "n39_k5_default", "coords": n39_k5_coords, "labels": n39_k5_labels, "g_max": None, "sigma_cells": None, "boundary_r95": None, "stevens_gamma": None},
        {"name": "n39_k5_custom", "coords": n39_k5_coords, "labels": n39_k5_labels, "g_max": 32, "sigma_cells": 1.8, "boundary_r95": 5.0, "stevens_gamma": None},
        {"name": "n40_b1_default", "coords": n40_b1_coords, "labels": n40_b1_labels, "g_max": None, "sigma_cells": None, "boundary_r95": None, "stevens_gamma": None},
        {"name": "n40_b1_r1", "coords": n40_b1_coords, "labels": n40_b1_labels, "g_max": None, "sigma_cells": None, "boundary_r95": 1.0, "stevens_gamma": None},
        {"name": "n6_b1_default", "coords": n6_b1_coords, "labels": n6_b1_labels, "g_max": None, "sigma_cells": None, "boundary_r95": None, "stevens_gamma": None},
        {"name": "n6_b1_r1", "coords": n6_b1_coords, "labels": n6_b1_labels, "g_max": None, "sigma_cells": None, "boundary_r95": 1.0, "stevens_gamma": None},
        {"name": "idx445_pure_tail", "coords": idx445_pure_tail_coords, "labels": idx445_pure_tail_labels, "g_max": 12, "sigma_cells": 0.15, "boundary_r95": 1.0, "stevens_gamma": 0.3},
        {"name": "idx451_sparse_tail", "coords": idx451_sparse_tail_coords, "labels": idx451_sparse_tail_labels, "g_max": 32, "sigma_cells": 0.15, "boundary_r95": float("inf"), "stevens_gamma": 0.3},
        {"name": "s500_292_k5", "coords": s500_292_coords, "labels": s500_292_k5_labels, "g_max": 12, "sigma_cells": 0.15, "boundary_r95": float("inf"), "stevens_gamma": 2.5},
        {"name": "s500_292_k6", "coords": s500_292_coords, "labels": s500_292_k6_labels, "g_max": 12, "sigma_cells": 0.15, "boundary_r95": float("inf"), "stevens_gamma": 2.5},
    ]


def _run_v6_gcaf_python_checks(eg: Any) -> list[dict[str, Any]]:
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

    # G6 target weights positive parity + adversarial checks
    imb_nw = [1.5 if lbl == 0 else 3.0 for lbl in labels_imb]
    imb_cw = [30.0, 12.0]
    caf_nw = float(eg.cluster_area_faithfulness(l_imb, labels_imb, node_weights=imb_nw, stevens_gamma=2.0))
    caf_cw = float(eg.cluster_area_faithfulness(l_imb, labels_imb, cluster_weights=imb_cw, stevens_gamma=2.0))
    assert abs(caf_nw - caf_cw) < 1e-12
    ev_cw = eg.evaluate_metrics(
        g_imb, l_imb, ["gcaf"], community_labels=labels_imb, cluster_weights=imb_cw, stevens_gamma=2.0
    )
    assert abs(float(ev_cw["cluster_area_faithfulness"]) - caf_cw) < 1e-12
    _assert_non_panic_error(
        "gcaf/weights_mutually_exclusive",
        lambda: eg.cluster_area_faithfulness(l_imb, labels_imb, node_weights=imb_nw, cluster_weights=imb_cw),
        expected_types=(ValueError,),
    )
    _assert_non_panic_error(
        "gcaf/weights_zero",
        lambda: eg.cluster_area_faithfulness(l_imb, labels_imb, cluster_weights=[0.0, 1.0]),
        expected_types=(ValueError,),
    )
    _assert_non_panic_error(
        "gcaf/weights_bool",
        lambda: eg.cluster_area_faithfulness(l_imb, labels_imb, cluster_weights=[True, False]),
        expected_types=(TypeError,),
    )

    # Adversarial non-panic error checks
    l_nan = eg.Layout2d.from_coords(coords_imb.copy())
    l_nan.coords(copy=False)[0, 0] = float("nan")
    _assert_non_panic_error(
        "gcaf/label_mismatch",
        lambda: eg.cluster_area_faithfulness(l_imb, labels_imb[:-1]),
        expected_types=(ValueError,),
    )
    _assert_non_panic_error(
        "gcaf/nan_layout",
        lambda: eg.cluster_area_faithfulness(l_nan, labels_imb),
        expected_types=(ValueError,),
    )
    for bad_g in (11, -1):
        _assert_non_panic_error(
            f"gcaf/g_max_{bad_g}",
            lambda g=bad_g: eg.cluster_area_faithfulness(l_imb, labels_imb, g_max=g),
            expected_types=(ValueError,),
        )
    for bad_s in (0.0, -0.5, float("nan"), float("inf"), 400.0):
        _assert_non_panic_error(
            f"gcaf/sigma_{bad_s}",
            lambda s=bad_s: eg.cluster_area_faithfulness(l_imb, labels_imb, sigma_cells=s),
            expected_types=(ValueError,),
        )
    for bad_v in (0.0, -1.0, float("nan"), float("inf")):
        _assert_non_panic_error(
            f"gcaf/kappa_{bad_v}",
            lambda v=bad_v: eg.cluster_area_faithfulness(l_imb, labels_imb, kappa=v),
            expected_types=(ValueError,),
        )
        _assert_non_panic_error(
            f"gcaf/tau_frac_{bad_v}",
            lambda v=bad_v: eg.cluster_area_faithfulness(l_imb, labels_imb, tau_frac=v),
            expected_types=(ValueError,),
        )
        _assert_non_panic_error(
            f"gcaf/stevens_gamma_{bad_v}",
            lambda v=bad_v: eg.cluster_area_faithfulness(l_imb, labels_imb, stevens_gamma=v),
            expected_types=(ValueError,),
        )
    for bad_r in (0.0, -1.0, float("nan"), float("-inf")):
        _assert_non_panic_error(
            f"gcaf/boundary_r95_{bad_r}",
            lambda r=bad_r: eg.cluster_area_faithfulness(l_imb, labels_imb, boundary_r95=r),
            expected_types=(ValueError,),
        )
    for missing_target in ("cluster_area_faithfulness", "gcaf"):
        _assert_non_panic_error(
            f"evaluate_metrics/{missing_target}_missing_labels",
            lambda t=missing_target: eg.evaluate_metrics(g_imb, l_imb, [t]),
            expected_types=(ValueError,),
        )
    lazy_ok = eg.evaluate_metrics(g_imb, l_imb, ["aspect_ratio"], g_max=4)
    assert "aspect_ratio" in lazy_ok and "cluster_area_faithfulness" not in lazy_ok
    _assert_non_panic_error(
        "evaluate_metrics/gcaf_lazy_gmax4",
        lambda: eg.evaluate_metrics(g_imb, l_imb, ["gcaf"], community_labels=labels_imb, g_max=4),
        expected_types=(ValueError,),
    )

    # Small-N (N <= 41) MEC quantile tie & boundary_r95=1.0 regression cases
    for case in _gcaf_small_n_regression_cases():
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
        "case": "v6_gcaf_python_checks",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
    }]


def _make_m2_canonical_graph_spec() -> tuple[int, list[tuple[int, int]]]:
    """Deterministic 70-node multi-topology graph (8x8 grid + bridge + 4-leaf star + isolated)."""
    edges: list[tuple[int, int]] = []
    for r in range(8):
        for c in range(8):
            u = r * 8 + c
            if c + 1 < 8:
                edges.append((u, u + 1))
            if r + 1 < 8:
                edges.append((u, u + 8))
    edges.append((63, 64))
    for leaf in (65, 66, 67, 68):
        edges.append((64, leaf))
    return 70, edges


def run_v6_m2_python_checks(eg: Any) -> list[dict[str, Any]]:
    """Run deterministic, f32/f64, edge-case & adversarial checks for M2 Python APIs."""
    import numpy as np

    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    n, edges = _make_m2_canonical_graph_spec()
    g = eg.CsrGraph.from_edges(n, edges, directed=False, sanitize=True)

    # 1. Pivots, PivotMDS (f64/f32, target_std=1.0/None), Koren (f64/f32, target_std=1.0/None)
    piv1 = eg.select_maxmin_sp_pivots(g, 12, seed=42, min_degree=2)
    piv2 = eg.select_maxmin_sp_pivots(g, 12, seed=42, min_degree=2)
    assert piv1.dtype == np.uint32 and piv1.shape == (12,)
    assert np.array_equal(piv1, piv2) and int(piv1.max()) < n

    for dtype_name, np_dtype in (("float64", np.float64), ("float32", np.float32)):
        lp = eg.pivot_mds_init(
            g, num_pivots=12, seed=42, min_degree=2, target_std=1.0, dtype=dtype_name
        )
        assert lp.dtype == dtype_name and lp.is_all_finite()
        cp = lp.coords(copy=True)
        assert cp.dtype == np_dtype and cp.shape == (n, 2)
        assert abs(float(np.std(cp)) - 1.0) < 1e-4

        lk = eg.koren_spectral_init(
            g, target_std=1.0, tol=1e-4, max_iters=300, seed=42, dtype=dtype_name
        )
        assert lk.dtype == dtype_name and lk.is_all_finite()
        ck = lk.coords(copy=True)
        assert ck.dtype == np_dtype and ck.shape == (n, 2)
        assert abs(float(np.std(ck)) - 1.0) < 1e-4

    assert eg.pivot_mds_init(g, num_pivots=12, target_std=None).is_all_finite()
    assert eg.koren_spectral_init(g, target_std=None).is_all_finite()

    # Empty & singleton graph edge cases
    for empty_n in (0, 1):
        ge = eg.CsrGraph.from_edges(empty_n, [])
        assert len(eg.select_maxmin_sp_pivots(ge, 4)) == empty_n
        assert len(eg.pivot_mds_init(ge, num_pivots=4, dtype="float32")) == empty_n
        assert len(eg.koren_spectral_init(ge, dtype="float64")) == empty_n
        lm_e = eg.compute_regularized_dog_varadhan_landmarks(ge)
        assert lm_e.num_nodes == empty_n
        assert len(lm_e.alt_distances([0] if empty_n == 1 else [], [0] if empty_n == 1 else [])) == empty_n

    # 2. Eigenpairs (both Lanczos path on n=70 and dense fallback on barbell_2comm_9) & moment stats
    lams, vecs, inv_d, used_dense = eg.smallest_laplacian_eigenpairs(
        g, n_modes=4, tol=1e-4, max_iters=300, seed=42
    )
    assert used_dense is False
    assert lams.shape == (4,) and vecs.shape == (n, 4) and inv_d.shape == (n,)
    assert np.allclose(vecs.T @ vecs, np.eye(4), atol=1e-6)
    assert float(inv_d[69]) == 0.0

    bb_spec = make_fixtures()["barbell_2comm_9"]
    g_bb = eg.CsrGraph.from_edges(bb_spec["n"], bb_spec["edges"], directed=False, sanitize=True)
    _, _, _, bb_dense = eg.smallest_laplacian_eigenpairs(g_bb, n_modes=4)
    assert bb_dense is True

    stats = eg.spectral_moment_stats(g)
    assert stats["sigma_lambda_sq"] > 0.0 and stats["sigma_lambda"] > 0.0
    assert stats["leaf_fraction"] > 0.0 and stats["mean_degree"] > 0.0

    # 3. Chebyshev heat SpMV (1D & 2D) and DoG JL embedding
    deg_vec = np.where(inv_d > 0.0, 1.0 / np.maximum(inv_d, 1e-300), 0.0)
    heat1d = eg.chebyshev_heat_spmv(g, deg_vec, 2.5)
    assert heat1d.shape == (n,) and np.allclose(heat1d, deg_vec, atol=1e-9)
    mat2d = np.column_stack([deg_vec, np.arange(n, dtype=np.float64) / float(n)])
    heat2d = eg.chebyshev_heat_spmv(g, mat2d, 2.5, degree=20, lambda_max=2.0)
    assert heat2d.shape == (n, 2) and np.all(np.isfinite(heat2d))

    jl_emb = eg.chebyshev_dog_jl_embedding(
        g, degree=25, jl_dim=12, seed=42, degree_normalize=True, eps_psd=1e-4
    )
    jl_plain = eg.chebyshev_dog_jl_embedding(
        g, degree=25, jl_dim=12, seed=42, degree_normalize=False, eps_psd=1e-4
    )
    assert jl_emb.shape == (n, 12)
    assert np.allclose(jl_emb, jl_plain * inv_d[:, None], atol=1e-12)

    # 4. 1+2-hop CSR, pure 2-hop kNN, perplexities, hybrid kNN, partial BFS kNN
    hop1, hop2 = eg.extract_1hop_2hop_csr(g)
    assert hop1.num_nodes == n and hop2.num_nodes == n
    assert hop1.num_edges == g.num_edges

    pure_knn, k2_deg, pure_stats = eg.build_pure_2hop_knn(g)
    sizes = eg.compute_2hop_neighborhood_sizes(g)
    assert np.array_equal(k2_deg, sizes)
    assert int(k2_deg[69]) == 0 and pure_stats["nnz"] == pure_knn.nnz
    knn_rt = eg.KnnGraph.from_csr(
        pure_knn.num_nodes, pure_knn.indptr(), pure_knn.indices(), pure_knn.distances()
    )
    assert len(knn_rt) == n and knn_rt.nnz == pure_knn.nnz
    assert np.array_equal(knn_rt.indices(), pure_knn.indices())
    assert np.allclose(knn_rt.distances(), pure_knn.distances())

    node_perps = eg.compute_nodewise_2hop_perplexities(g)
    med_perp = eg.compute_median_2hop_perplexity(g)
    assert node_perps.shape == (n,) and np.all(node_perps > 0.0) and med_perp > 0.0

    hybrid_knn, hybrid_stats = eg.build_hybrid_adaptive_knn(g, jl_emb, node_perps)
    assert hybrid_knn.num_nodes == n
    assert hybrid_stats["n_truncated_1hop"] == 0
    assert hybrid_stats["n_filled_residual"] + hybrid_stats["n_pure_2hop_rows"] == n

    bfs_knn = eg.partial_bfs_knn(g, k=6, seed=42)
    assert bfs_knn.num_nodes == n and bfs_knn.nnz > 0

    # 5. Probabilities, fuzzy simplicial set, find_ab_params, Horvitz-Thompson, DoG Varadhan
    probs = eg.compute_knn_probabilities(hybrid_knn, node_perps)
    probs_scalar = eg.compute_knn_probabilities(hybrid_knn, float(med_perp))
    assert probs.num_nodes == n and abs(float(np.sum(probs.values())) - 1.0) < 1e-10
    assert abs(float(np.sum(probs_scalar.values())) - 1.0) < 1e-10

    fuzzy = eg.compute_fuzzy_simplicial_set(hybrid_knn)
    fvals = fuzzy.values()
    assert fuzzy.num_nodes == n and np.all((fvals > 1e-7) & (fvals <= 1.0))

    a, b = eg.find_ab_params(0.1, 1.0)
    assert abs(a - 1.577) < 0.02 and abs(b - 0.895) < 0.02
    assert abs(eg.compute_horvitz_thompson_gamma(1.0, 1600, 16) - 10.0) < 1e-12

    lm = eg.compute_regularized_dog_varadhan_landmarks(
        g, degree=25, num_landmarks=16, seed=42, t_coarse=48.0
    )
    assert lm.num_nodes == n and lm.num_landmarks == 16 and lm.used_coarse_scaffold is True
    assert lm.table().shape == (n, 16)
    coarse_tbl = lm.coarse_table()
    assert coarse_tbl is not None and coarse_tbl.shape == (n, 16)
    alt = lm.alt_distances([0, 63, 64, 69], [63, 0, 68, 69])
    assert alt.shape == (4,) and alt[3] == 0.0 and abs(alt[0] - alt[1]) < 1e-12

    lm_no_coarse = eg.compute_regularized_dog_varadhan_landmarks(
        g, degree=25, num_landmarks=8, seed=42, t_coarse=0.0
    )
    assert lm_no_coarse.used_coarse_scaffold is False and lm_no_coarse.coarse_table() is None

    # 5b. AmgPcgSolver (SPD, singular/isolated, cooperative stepping, batch)
    amg_reg = eg.AmgPcgSolver.from_graph_regularized_laplacian(g, gamma=10.0)
    b_amg = np.cos(0.17 * np.arange(n, dtype=np.float64))
    x_amg, amg_info = amg_reg.solve(b_amg)
    assert amg_info["converged"] and amg_info["relative_residual"] <= 1e-8
    amg_step = eg.AmgPcgSolver.from_graph_regularized_laplacian(g, gamma=10.0)
    amg_step.start_solve(b_amg)
    if not amg_step.is_finished:
        amg_step.step()
        amg_step.run()
    assert np.array_equal(amg_step.solution(), x_amg)

    b_amg_batch = np.column_stack([b_amg, -2.0 * b_amg])
    x_amg_batch, batch_info = amg_reg.solve_batch(b_amg_batch)
    assert batch_info["all_converged"] and np.allclose(x_amg_batch[:, 0], x_amg, atol=1e-12)

    amg_sing = eg.AmgPcgSolver.from_graph_laplacian(g, kind="combinatorial", shift=0.0)
    b_sing = np.zeros(n, dtype=np.float64)
    b_sing[0] = 1.0
    b_sing[64] = -1.0
    x_sing, sing_info = amg_sing.solve(b_sing)
    assert sing_info["converged"] and float(x_sing[69]) == 0.0

    results.append({
        "case": "v6_m2_linalg_and_neighborhood",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
    })

    # 6. Adversarial M2 Python checks (must raise ValueError/TypeError/IndexError, never PanicException)
    t1 = time.perf_counter()
    bad_coords = jl_emb.copy()
    bad_coords[0, 0] = float("nan")
    m2_bad_cases: list[tuple[str, Callable[[], Any]]] = [
        ("KnnGraph.from_csr/bad_indptr", lambda: eg.KnnGraph.from_csr(2, [0, 1], [1], [1.0])),
        ("KnnGraph.from_csr/neg_dist", lambda: eg.KnnGraph.from_csr(2, [0, 1, 1], [1], [-1.0])),
        ("KnnGraph.from_csr/nan_dist", lambda: eg.KnnGraph.from_csr(2, [0, 1, 1], [1], [float("nan")])),
        ("KnnGraph.from_csr/oob_idx", lambda: eg.KnnGraph.from_csr(2, [0, 1, 1], [9], [1.0])),
        ("KnnGraph.from_csr/bool_nodes", lambda: eg.KnnGraph.from_csr(True, [0, 1], [0], [1.0])),
        ("pivot_mds_init/zero_pivots", lambda: eg.pivot_mds_init(g, num_pivots=0)),
        ("pivot_mds_init/bad_dtype", lambda: eg.pivot_mds_init(g, dtype="int32")),
        ("koren_spectral_init/zero_tol", lambda: eg.koren_spectral_init(g, tol=0.0)),
        ("koren_spectral_init/zero_iters", lambda: eg.koren_spectral_init(g, max_iters=0)),
        ("smallest_laplacian_eigenpairs/zero_modes", lambda: eg.smallest_laplacian_eigenpairs(g, 0)),
        ("chebyshev_heat_spmv/len_mismatch", lambda: eg.chebyshev_heat_spmv(g, np.ones(n - 1), 1.0)),
        ("chebyshev_heat_spmv/nan_vec", lambda: eg.chebyshev_heat_spmv(g, np.full(n, float("nan")), 1.0)),
        ("chebyshev_heat_spmv/neg_t", lambda: eg.chebyshev_heat_spmv(g, deg_vec, -1.0)),
        ("chebyshev_dog_jl_embedding/zero_deg", lambda: eg.chebyshev_dog_jl_embedding(g, degree=0)),
        ("chebyshev_dog_jl_embedding/zero_dim", lambda: eg.chebyshev_dog_jl_embedding(g, jl_dim=0)),
        ("compute_nodewise_2hop_perplexities/bad_c", lambda: eg.compute_nodewise_2hop_perplexities(g, c=0.0)),
        (
            "compute_median_2hop_perplexity/min_gt_max",
            lambda: eg.compute_median_2hop_perplexity(g, min_perplexity=30.0, max_perplexity=10.0),
        ),
        ("build_hybrid_adaptive_knn/nan_coords", lambda: eg.build_hybrid_adaptive_knn(g, bad_coords, node_perps)),
        ("build_hybrid_adaptive_knn/len_perps", lambda: eg.build_hybrid_adaptive_knn(g, jl_emb, node_perps[:-1])),
        ("compute_knn_probabilities/nan_perp", lambda: eg.compute_knn_probabilities(hybrid_knn, float("nan"))),
        ("compute_fuzzy_simplicial_set/zero_iters", lambda: eg.compute_fuzzy_simplicial_set(hybrid_knn, sigma_iters=0)),
        ("find_ab_params/neg_min_dist", lambda: eg.find_ab_params(-0.1, 1.0)),
        ("compute_horvitz_thompson_gamma/neg_gamma", lambda: eg.compute_horvitz_thompson_gamma(-1.0, 10, 4)),
        ("DogVaradhanLandmarks.alt_distances/oob", lambda: lm.alt_distances([n + 5], [0])),
        ("DogVaradhanLandmarks.alt_distances/mismatch", lambda: lm.alt_distances([0, 1], [0])),
        ("compute_regularized_dog_varadhan_landmarks/bad_beta", lambda: eg.compute_regularized_dog_varadhan_landmarks(g, beta_floor=1.0)),
        ("AmgPcgSolver/asym", lambda: eg.AmgPcgSolver(2, [0, 2, 4], [0, 1, 0, 1], [2.0, -1.0, -2.0, 2.0])),
        ("AmgPcgSolver/neg_diag", lambda: eg.AmgPcgSolver(2, [0, 2, 4], [0, 1, 0, 1], [-1.0, 0.0, 0.0, 2.0])),
        ("AmgPcgSolver/solve_nan", lambda: amg_reg.solve(np.full(n, float("nan")))),
        ("AmgPcgSolver/solve_dim", lambda: amg_reg.solve(np.ones(n - 1))),
    ]
    for label, fn in m2_bad_cases:
        _assert_non_panic_error(label, fn)

    results.append({
        "case": "v6_m2_adversarial_non_panic",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t1, 4),
    })
    return results


def _make_m3_asymmetric_coords(n: int, scale: float = 1.0) -> list[tuple[float, float]]:
    """Generate deterministic non-symmetric 2D coordinates without gradient ties."""
    return [
        (
            scale * (0.7 * math.cos(0.83 * float(i) + 0.15) + 0.11 * float(i) / max(n, 1)),
            scale * (0.6 * math.sin(1.13 * float(i) - 0.21) - 0.09 * float(i) / max(n, 1)),
        )
        for i in range(n)
    ]


def run_v6_m3_python_checks(eg: Any) -> list[dict[str, Any]]:
    """Run deterministic, f32/f64, step()==run(), manifold & adversarial checks for M3 Python APIs."""
    import numpy as np

    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    # 1. Clustering & coarsening (KMeans++, Spectral, Louvain, LPA, Coarsen, Project)
    pts_km = np.array(
        [
            [10.0, 10.0],
            [10.2, 9.8],
            [9.8, 10.1],
            [-10.0, -10.0],
            [-9.9, -10.2],
            [-10.1, -9.8],
        ],
        dtype=np.float64,
    )
    km64 = eg.kmeans_plus_plus(pts_km, k=2, max_iters=100, tol=1e-4, n_init=3, seed=42)
    km32 = eg.kmeans_plus_plus(pts_km.astype(np.float32), k=2, max_iters=100, tol=1e-4, n_init=3, seed=42)
    assert np.array_equal(km64["labels"], np.array([0, 0, 0, 1, 1, 1], dtype=np.uint32))
    assert np.array_equal(km64["labels"], km32["labels"])
    assert bool(km64["converged"]) and float(km64["inertia"]) < 1.0

    # 12-node 2-clique barbell for clustering separation checks
    bb12_edges = [
        (i, j) for off in (0, 6) for i in range(off, off + 6) for j in range(i + 1, off + 6)
    ] + [(5, 6)]
    g_bb12 = eg.CsrGraph.from_edges(12, bb12_edges, directed=False, sanitize=True)

    sp1 = eg.spectral_clustering(g_bb12, k=2, seed=42)
    sp2 = eg.spectral_clustering(g_bb12, k=2, seed=42)
    assert np.array_equal(sp1, sp2) and np.array_equal(sp1, np.array([0] * 6 + [1] * 6, dtype=np.uint32))

    lv = eg.louvain_communities(g_bb12, max_passes=100, resolution=1.0, min_modularity_gain=1e-7, seed=42)
    assert lv["num_communities"] == 2 and float(lv["modularity"]) > 0.35
    assert np.array_equal(lv["labels"], np.array([0] * 6 + [1] * 6, dtype=np.uint32))

    lpa = eg.label_propagation(g_bb12, max_iters=100, seed=42)
    assert np.array_equal(lpa, np.array([0] * 6 + [1] * 6, dtype=np.uint32))

    coarse_g, n2c, sizes, num_clusters = eg.coarsen_graph(g_bb12, [100] * 6 + [999] * 6)
    assert num_clusters == 2 and coarse_g.num_nodes == 2 and coarse_g.num_edges == 1
    assert np.array_equal(n2c, np.array([0] * 6 + [1] * 6, dtype=np.uint32))
    assert np.array_equal(sizes, np.array([6, 6], dtype=np.uint32))

    coarse_base = np.array([[-2.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    for dtype_name, np_dtype in (("float64", np.float64), ("float32", np.float32)):
        cl = eg.Layout2d.from_coords(coarse_base.astype(np_dtype))
        proj1 = eg.project_clustered_layout(cl, n2c, spread_radius=0.2, seed=42)
        proj2 = eg.project_clustered_layout(cl, n2c, spread_radius=0.2, seed=42)
        assert proj1.dtype == dtype_name and len(proj1) == 12 and proj1.is_all_finite()
        assert np.array_equal(proj1.coords(copy=True), proj2.coords(copy=True))

    # 2. ManifoldLayout2d across all 3 geometries in float64 & float32
    manifold_coords = {
        "hyperbolic": np.array([[0.0, 0.0], [0.5, 0.0], [3.0, 4.0]], dtype=np.float64),
        "spherical": np.array([[-0.3, -0.2], [0.2, math.pi / 2.0], [0.2 + math.pi / 2.0, math.pi / 2.0]], dtype=np.float64),
        "torus": np.array([[1.95, -0.95], [0.05, 0.95], [0.5, 0.5]], dtype=np.float64),
    }
    for kind, base_arr in manifold_coords.items():
        for dtype_name, np_dtype in (("float64", np.float64), ("float32", np.float32)):
            ml = eg.ManifoldLayout2d.from_coords(kind, base_arr.astype(np_dtype))
            assert ml.kind == kind and ml.dtype == dtype_name and ml.len() == 3
            ml.project_all()
            c = ml.coords()
            assert np.all(np.isfinite(c))
            if kind == "hyperbolic":
                assert np.all(c[:, 0] ** 2 + c[:, 1] ** 2 < 1.0)
                assert abs(ml.geodesic_distance(0, 1) - 2.0 * math.atanh(0.5)) < 1e-5
            elif kind == "spherical":
                assert np.all((c[:, 0] >= 0.0) & (c[:, 0] < 2.0 * math.pi))
                assert np.all((c[:, 1] >= 0.0) & (c[:, 1] <= math.pi))
                assert abs(ml.geodesic_distance(1, 2) - math.pi / 2.0) < 1e-5
            else:
                assert np.all((c >= 0.0) & (c < 1.0))
                assert abs(ml.geodesic_distance(0, 1) - math.hypot(0.1, 0.1)) < 1e-5
            back = ml.to_layout()
            ml2 = eg.ManifoldLayout2d.from_layout(kind, back)
            assert np.allclose(ml2.coords(), c)

    # 3. All 6 Euclidean solvers + SgdSolver manifold stepping (step == run bit-exact in f64 & f32)
    g16_edges = [
        (r * 4 + c, r * 4 + c + 1) for r in range(4) for c in range(3)
    ] + [
        (r * 4 + c, (r + 1) * 4 + c) for r in range(3) for c in range(4)
    ]
    g16 = eg.CsrGraph.from_edges(16, g16_edges, directed=False, sanitize=True)
    init16 = np.array(_make_m3_asymmetric_coords(16, scale=0.8), dtype=np.float64)
    emb16 = eg.chebyshev_dog_jl_embedding(g16, degree=15, jl_dim=8, seed=42)

    for dtype_name, np_dtype in (("float64", np.float64), ("float32", np.float32)):
        init_typed = init16.astype(np_dtype)

        # TsNetSolver.pure_2hop (all 4 backends)
        for backend in ("fft", "linear_fft", "barnes_hut", "exact"):
            ts_r = eg.TsNetSolver.pure_2hop(
                g16, backend=backend, iterations_stage1=4, iterations_stage2=4, iterations_stage3=4, seed=42, dtype=dtype_name
            )
            ts_s = eg.TsNetSolver.pure_2hop(
                g16, backend=backend, iterations_stage1=4, iterations_stage2=4, iterations_stage3=4, seed=42, dtype=dtype_name
            )
            lr = eg.Layout2d.from_coords(init_typed)
            ls = eg.Layout2d.from_coords(init_typed)
            assert ts_r.run(lr) == "completed"
            assert ts_s.run_steps(ls, 5) == "running"
            while not ts_s.is_finished:
                ts_s.step(ls)
            assert ts_s.status == "completed"
            assert np.array_equal(lr.coords(copy=True), ls.coords(copy=True))

        # SgdSolver variants
        sgd_builders: list[tuple[str, Callable[[], Any]]] = [
            ("full", lambda dt=dtype_name: eg.SgdSolver.full(g16, t_max=10, seed=42, dtype=dt)),
            ("sparse_barrier_dog", lambda dt=dtype_name: eg.SgdSolver.sparse_barrier_dog(g16, degree=15, num_landmarks=8, t_max=10, seed=42, dtype=dt)),
            ("random_pair", lambda dt=dtype_name: eg.SgdSolver.random_pair(g16, pairs_per_node=4, t_max=10, seed=42, dtype=dt)),
            ("omega", lambda dt=dtype_name: eg.SgdSolver.omega(g16, emb16, k=4, t_max=10, seed=42, dtype=dt)),
            ("streaming_negatives", lambda dt=dtype_name: eg.SgdSolver.streaming_negatives(g16, negatives_per_node=3, t_max=10, seed=42, dtype=dt)),
            (
                "from_pairs_with_hinges",
                lambda dt=dtype_name: eg.SgdSolver.from_pairs_with_hinges(
                    16,
                    [e[0] for e in g16_edges],
                    [e[1] for e in g16_edges],
                    [1.0] * len(g16_edges),
                    [1.0] * len(g16_edges),
                    [1.0] * len(g16_edges),
                    [1.0] * len(g16_edges),
                    [0, 1],
                    [15, 14],
                    [3.0, 3.0],
                    [3.0, 3.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                    t_max=10,
                    seed=42,
                    dtype=dt,
                ),
            ),
        ]
        for s_name, mk_solver in sgd_builders:
            sr = mk_solver()
            ss = mk_solver()
            lr = eg.Layout2d.from_coords(init_typed)
            ls = eg.Layout2d.from_coords(init_typed)
            assert sr.run(lr) == "completed", f"SgdSolver.{s_name} did not complete"
            assert ss.run_steps(ls, 4) == "running"
            while not ss.is_finished:
                ss.step(ls)
            assert np.array_equal(lr.coords(copy=True), ls.coords(copy=True)), f"SgdSolver.{s_name} step != run"

        # SgdSolver manifold stepping (all 3 manifolds)
        for kind in ("hyperbolic", "spherical", "torus"):
            mr = eg.SgdSolver.full(g16, t_max=8, seed=42, dtype=dtype_name)
            ms = eg.SgdSolver.full(g16, t_max=8, seed=42, dtype=dtype_name)
            mlr = eg.ManifoldLayout2d.from_coords(kind, (init16 * 0.25).astype(np_dtype))
            mls = eg.ManifoldLayout2d.from_coords(kind, (init16 * 0.25).astype(np_dtype))
            assert mr.run_manifold(mlr) == "completed"
            assert ms.run_steps_manifold(mls, 3) == "running"
            while not ms.is_finished:
                ms.step_manifold(mls)
            assert np.array_equal(mlr.coords(), mls.coords())

        # GuMapSolver.from_graph_dog
        gmr = eg.GuMapSolver.from_graph_dog(g16, max_2hop_per_node=16, degree=15, num_landmarks=8, n_epochs=15, seed=42, dtype=dtype_name)
        gms = eg.GuMapSolver.from_graph_dog(g16, max_2hop_per_node=16, degree=15, num_landmarks=8, n_epochs=15, seed=42, dtype=dtype_name)
        glr = eg.Layout2d.from_coords(init_typed)
        gls = eg.Layout2d.from_coords(init_typed)
        assert gmr.run(glr) == "completed"
        assert gms.run_steps(gls, 5) == "running"
        while not gms.is_finished:
            gms.step(gls)
        assert np.array_equal(glr.coords(copy=True), gls.coords(copy=True))

        # SpatialIntruderRefiner on folded 3x3 grid
        g9 = eg.CsrGraph.from_edges(
            9,
            [(r * 3 + c, r * 3 + c + 1) for r in range(3) for c in range(2)]
            + [(r * 3 + c, (r + 1) * 3 + c) for r in range(2) for c in range(3)],
            directed=False,
            sanitize=True,
        )
        folded9 = np.array([[float(c), float(r)] for r in range(3) for c in range(3)], dtype=np.float64)
        folded9[8] = np.array([0.08, 0.08], dtype=np.float64)
        rf_r = eg.SpatialIntruderRefiner(g9, rounds=3, k_cap=32, t_max=10, seed=42, dtype=dtype_name)
        rf_s = eg.SpatialIntruderRefiner(g9, rounds=3, k_cap=32, t_max=10, seed=42, dtype=dtype_name)
        rlr = eg.Layout2d.from_coords(folded9.astype(np_dtype))
        rls = eg.Layout2d.from_coords(folded9.astype(np_dtype))
        assert rf_r.run(rlr) in ("converged", "completed")
        rf_s.step(rls)
        rf_s.run(rls)
        assert np.array_equal(rlr.coords(copy=True), rls.coords(copy=True))
        assert rf_r.stats()["n_fp_total"] > 0

        # StressMajorizationSolver.from_graph
        sm_r = eg.StressMajorizationSolver.from_graph(g16, max_iterations=15, epsilon=1e-6, dtype=dtype_name)
        sm_s = eg.StressMajorizationSolver.from_graph(g16, max_iterations=15, epsilon=1e-6, dtype=dtype_name)
        slr = eg.Layout2d.from_coords(init_typed)
        sls = eg.Layout2d.from_coords(init_typed)
        assert sm_r.run(slr) in ("converged", "completed")
        sm_s.run_steps(sls, 4)
        while not sm_s.is_finished:
            sm_s.step(sls)
        assert np.array_equal(slr.coords(copy=True), sls.coords(copy=True))

        # KamadaKawaiSolver.from_graph
        kk_r = eg.KamadaKawaiSolver.from_graph(g9, eps=1e-3, max_inner_iters=40, max_outer_iters=20, dtype=dtype_name)
        kk_s = eg.KamadaKawaiSolver.from_graph(g9, eps=1e-3, max_inner_iters=40, max_outer_iters=20, dtype=dtype_name)
        init9 = np.array(_make_m3_asymmetric_coords(9, scale=1.2), dtype=np.float64).astype(np_dtype)
        klr = eg.Layout2d.from_coords(init9)
        kls = eg.Layout2d.from_coords(init9)
        assert kk_r.run(klr) in ("converged", "completed")
        kk_s.run_steps(kls, 4)
        while not kk_s.is_finished:
            kk_s.step(kls)
        assert np.array_equal(klr.coords(copy=True), kls.coords(copy=True))

    results.append({
        "case": "v6_m3_layout_and_clustering",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
    })

    # 4. Adversarial M3 Python checks (never PanicException)
    t1 = time.perf_counter()
    g_dir = eg.CsrGraph.from_edges(3, [(0, 1), (1, 2)], directed=True, sanitize=True)
    l_nan = eg.Layout2d.zeros(12)
    l_nan.coords(copy=False)[0, 0] = float("nan")
    sm_chk = eg.StressMajorizationSolver.from_graph(g_bb12, max_iterations=5)
    kk_chk = eg.KamadaKawaiSolver.from_graph(g_bb12, max_outer_iters=5)
    m3_bad: list[tuple[str, Callable[[], Any]]] = [
        ("kmeans_plus_plus/k_zero", lambda: eg.kmeans_plus_plus(pts_km, k=0)),
        ("kmeans_plus_plus/k_gt_n", lambda: eg.kmeans_plus_plus(pts_km, k=10)),
        ("spectral_clustering/directed", lambda: eg.spectral_clustering(g_dir, k=2)),
        ("louvain_communities/directed", lambda: eg.louvain_communities(g_dir)),
        ("label_propagation/directed", lambda: eg.label_propagation(g_dir)),
        ("coarsen_graph/directed", lambda: eg.coarsen_graph(g_dir, [0, 1, 1])),
        ("project_clustered_layout/oob", lambda: eg.project_clustered_layout(eg.Layout2d.zeros(2), [0, 2])),
        ("ManifoldLayout2d/bad_kind", lambda: eg.ManifoldLayout2d.zeros("euclidean", 3)),
        ("ManifoldLayout2d/nan", lambda: eg.ManifoldLayout2d.from_coords("torus", [[float("nan"), 0.0]])),
        ("TsNetSolver/bad_backend", lambda: eg.TsNetSolver.pure_2hop(g16, backend="bad")),
        ("SgdSolver/bad_schedule", lambda: eg.SgdSolver.full(g16, schedule="cosine")),
        ("SgdSolver/oob_pair", lambda: eg.SgdSolver.from_pairs(2, [0], [5], [1.0])),
        ("GuMapSolver/neg_min_dist", lambda: eg.GuMapSolver.from_graph_dog(g16, min_dist=-0.1)),
        ("SpatialIntruderRefiner/k_cap_zero", lambda: eg.SpatialIntruderRefiner(g16, k_cap=0)),
        ("StressMajorizationSolver/nan_layout", lambda: sm_chk.step(l_nan)),
        ("KamadaKawaiSolver/dim_mismatch", lambda: kk_chk.step(eg.Layout2d.zeros(3))),
    ]
    for label, fn in m3_bad:
        _assert_non_panic_error(label, fn)

    results.append({
        "case": "v6_m3_adversarial_non_panic",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t1, 4),
    })
    return results


def _count_rect_overlaps_py(
    coords: Any, hw: Any, hh: Any, tol: float = 1e-6
) -> int:
    n = len(coords)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(float(coords[i, 0]) - float(coords[j, 0]))
            dy = abs(float(coords[i, 1]) - float(coords[j, 1]))
            if dx < float(hw[i]) + float(hw[j]) - tol and dy < float(hh[i]) + float(hh[j]) - tol:
                count += 1
    return count


def run_v6_m4_python_checks(eg: Any) -> list[dict[str, Any]]:
    """Run deterministic, f32/f64, step()==run() & adversarial checks for M4 Python APIs."""
    import numpy as np

    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    g16_edges = [
        (r * 4 + c, r * 4 + c + 1) for r in range(4) for c in range(3)
    ] + [
        (r * 4 + c, (r + 1) * 4 + c) for r in range(3) for c in range(4)
    ]
    g16 = eg.CsrGraph.from_edges(16, g16_edges, directed=False, sanitize=True)
    init16 = np.array(_make_m3_asymmetric_coords(16, scale=0.8), dtype=np.float64)

    g25_edges = [
        (r * 5 + c, r * 5 + c + 1) for r in range(5) for c in range(4)
    ] + [
        (r * 5 + c, (r + 1) * 5 + c) for r in range(4) for c in range(5)
    ]
    g25 = eg.CsrGraph.from_edges(25, g25_edges, directed=False, sanitize=True)
    init25 = np.array(_make_m3_asymmetric_coords(25, scale=1.0), dtype=np.float64)

    knn16, _, _ = eg.build_pure_2hop_knn(g16)
    fuzzy16 = eg.compute_fuzzy_simplicial_set(knn16)
    deg16 = np.diff(g16.indptr()).astype(np.uint32)

    for dtype_name, np_dtype in (("float64", np.float64), ("float32", np.float32)):
        init16_typed = init16.astype(np_dtype)
        init25_typed = init25.astype(np_dtype)

        # 1. TFdpSolver across all 3 backends + from_knn
        for backend in ("ibfft", "barnes_hut", "exact"):
            sr = eg.TFdpSolver(g16, backend=backend, total_iterations=8, seed=42, dtype=dtype_name)
            ss = eg.TFdpSolver(g16, backend=backend, total_iterations=8, seed=42, dtype=dtype_name)
            lr = eg.Layout2d.from_coords(init16_typed)
            ls = eg.Layout2d.from_coords(init16_typed)
            rho0 = sr.current_rho
            assert sr.run(lr) == "completed"
            assert sr.current_rho <= rho0
            assert ss.run_steps(ls, 3) == "running"
            while not ss.is_finished:
                ss.step(ls)
            assert ss.status == "completed"
            assert lr.is_all_finite()
            assert np.array_equal(lr.coords(copy=True), ls.coords(copy=True)), (
                f"TFdpSolver/{backend}/{dtype_name} step != run"
            )

        tk1 = eg.TFdpSolver.from_knn(knn16, backend="exact", total_iterations=6, seed=42, dtype=dtype_name)
        tk2 = eg.TFdpSolver.from_knn(
            knn16.indptr(), knn16.indices(), knn16.distances(),
            backend="exact", total_iterations=6, seed=42, dtype=dtype_name,
        )
        lk1 = eg.Layout2d.from_coords(init16_typed)
        lk2 = eg.Layout2d.from_coords(init16_typed)
        assert tk1.run(lk1) == "completed"
        assert tk2.run(lk2) == "completed"
        assert np.array_equal(lk1.coords(copy=True), lk2.coords(copy=True))

        # 2. IbUmapSolver across all 3 backends + from_knn + from_fuzzy_set
        for backend in ("ibfft", "barnes_hut", "exact"):
            ir = eg.IbUmapSolver(g16, max_2hop_per_node=20, backend=backend, n_epochs=6, seed=42, dtype=dtype_name)
            is_ = eg.IbUmapSolver(g16, max_2hop_per_node=20, backend=backend, n_epochs=6, seed=42, dtype=dtype_name)
            lr = eg.Layout2d.from_coords(init16_typed)
            ls = eg.Layout2d.from_coords(init16_typed)
            assert ir.run(lr) == "completed"
            assert math.isclose(ir.current_rho, 1.0, rel_tol=1e-9)
            assert is_.run_steps(ls, 2) == "running"
            while not is_.is_finished:
                is_.step(ls)
            assert lr.is_all_finite()
            assert np.array_equal(lr.coords(copy=True), ls.coords(copy=True)), (
                f"IbUmapSolver/{backend}/{dtype_name} step != run"
            )

        ik = eg.IbUmapSolver.from_knn(knn16, backend="exact", n_epochs=6, seed=42, dtype=dtype_name)
        ifs = eg.IbUmapSolver.from_fuzzy_set(fuzzy16, degrees=deg16, backend="exact", n_epochs=6, seed=42, dtype=dtype_name)
        lik = eg.Layout2d.from_coords(init16_typed)
        lif = eg.Layout2d.from_coords(init16_typed)
        assert ik.run(lik) == "completed" and lik.is_all_finite()
        assert ifs.run(lif) == "completed" and lif.is_all_finite()

        # 3. SgdSolver M4 constructors (snap_t_fdp, pacmap_graph, sgd2, schedule="squared")
        m4_sgd_builders: list[tuple[str, Callable[[], Any]]] = [
            ("snap_t_fdp", lambda dt=dtype_name: eg.SgdSolver.snap_t_fdp(g16, iterations=8, negatives_per_edge=3, seed=42, dtype=dt)),
            ("pacmap_graph", lambda dt=dtype_name: eg.SgdSolver.pacmap_graph(g16, n_neighbors=5, phase1_iters=3, phase2_iters=3, phase3_iters=4, seed=42, dtype=dt)),
            ("sgd2", lambda dt=dtype_name: eg.SgdSolver.sgd2(g16, t_max=8, batch_size=16, seed=42, dtype=dt)),
            ("squared", lambda dt=dtype_name: eg.SgdSolver.full(g16, t_max=8, schedule="squared", seed=42, dtype=dt)),
        ]
        for s_name, mk_s in m4_sgd_builders:
            sr = mk_s()
            ss = mk_s()
            lr = eg.Layout2d.from_coords(init16_typed)
            ls = eg.Layout2d.from_coords(init16_typed)
            assert sr.run(lr) == "completed", f"SgdSolver.{s_name} did not complete"
            assert ss.run_steps(ls, 3) == "running"
            while not ss.is_finished:
                ss.step(ls)
            assert lr.is_all_finite()
            assert np.array_equal(lr.coords(copy=True), ls.coords(copy=True)), (
                f"SgdSolver.{s_name}/{dtype_name} step != run"
            )

        # 4. ForceAtlas2Solver (Barnes-Hut & Exact)
        for use_bh in (True, False):
            fr = eg.ForceAtlas2Solver(g16, iterations=10, use_barnes_hut=use_bh, seed=42, dtype=dtype_name)
            fs = eg.ForceAtlas2Solver(g16, iterations=10, use_barnes_hut=use_bh, seed=42, dtype=dtype_name)
            lr = eg.Layout2d.from_coords(init16_typed)
            ls = eg.Layout2d.from_coords(init16_typed)
            assert fr.run(lr) == "completed"
            assert fs.run_steps(ls, 4) == "running"
            while not fs.is_finished:
                fs.step(ls)
            assert lr.is_all_finite()
            assert np.array_equal(lr.coords(copy=True), ls.coords(copy=True)), (
                f"ForceAtlas2Solver/bh={use_bh}/{dtype_name} step != run"
            )

        # 5. YifanHuSolver (multilevel)
        yr = eg.YifanHuSolver(g25, max_iterations_per_level=10, min_coarsen_size=6, min_coarsen_ratio=0.9, seed=42, dtype=dtype_name)
        ys = eg.YifanHuSolver(g25, max_iterations_per_level=10, min_coarsen_size=6, min_coarsen_ratio=0.9, seed=42, dtype=dtype_name)
        assert yr.num_levels >= 2
        lyr = eg.Layout2d.from_coords(init25_typed)
        lys = eg.Layout2d.from_coords(init25_typed)
        assert yr.run(lyr) in ("converged", "completed")
        assert yr.current_level == 0
        ys.step(lys)
        ys.run_steps(lys, 4)
        ys.run(lys)
        assert lyr.is_all_finite()
        assert np.array_equal(lyr.coords(copy=True), lys.coords(copy=True)), (
            f"YifanHuSolver/{dtype_name} step != run"
        )

        # 6. PrismOverlapRemoval (new + from_radii)
        prism_init = (init16 * 0.15).astype(np_dtype)
        hw = np.full(16, 0.5, dtype=np.float64)
        hh = np.full(16, 0.4, dtype=np.float64)
        pr = eg.PrismOverlapRemoval(hw, hh, max_iterations=50, padding=0.05, seed=42, dtype=dtype_name)
        ps = eg.PrismOverlapRemoval(hw, hh, max_iterations=50, padding=0.05, seed=42, dtype=dtype_name)
        lpr = eg.Layout2d.from_coords(prism_init)
        lps = eg.Layout2d.from_coords(prism_init)
        assert pr.run(lpr) == "converged"
        assert pr.last_overlap_count == 0
        ps.step(lps)
        ps.run_steps(lps, 2)
        ps.run(lps)
        assert np.array_equal(lpr.coords(copy=True), lps.coords(copy=True))
        tol_p = 1e-4 if dtype_name == "float32" else 1e-8
        assert _count_rect_overlaps_py(lpr.coords(copy=True), hw + 0.025, hh + 0.025, tol=tol_p) == 0

        pr_rad = eg.PrismOverlapRemoval.from_radii(np.full(16, 0.4, dtype=np.float64), max_iterations=50, seed=42, dtype=dtype_name)
        l_rad = eg.Layout2d.from_coords(prism_init)
        assert pr_rad.run(l_rad) == "converged"
        assert pr_rad.last_overlap_count == 0

        # 7. SeparationConstraint + projection helpers
        arr_1d = np.array([2.0, 0.5, 0.0], dtype=np_dtype)
        out_1d = eg.project_1d(arr_1d, [eg.SeparationConstraint(0, 1, 1.5), eg.SeparationConstraint(1, 2, 2.0)])
        assert out_1d.dtype == np_dtype
        assert float(out_1d[1] - out_1d[0]) >= 1.5 - 1e-5
        assert float(out_1d[2] - out_1d[1]) >= 2.0 - 1e-5

        rect_l = eg.Layout2d.from_coords(np.array([[0.0, 0.0], [0.2, 0.1], [0.1, -0.1], [-0.1, 0.2]], dtype=np_dtype))
        hw4 = np.full(4, 0.5, dtype=np_dtype)
        hh4 = np.full(4, 0.5, dtype=np_dtype)
        eg.project_rectangle_no_overlap_2d(rect_l, hw4, hh4)
        assert _count_rect_overlaps_py(rect_l.coords(copy=True), hw4, hh4, tol=1e-4) == 0

        cl_l = eg.Layout2d.from_coords(np.array([[0.0, 0.0], [0.5, 0.0], [0.3, 0.2], [0.8, 0.2]], dtype=np_dtype))
        eg.project_clustered_rectangle_no_overlap_2d(cl_l, np.array([0, 0, 1, 1], dtype=np.uint32), hw4, hh4)
        cl_c = cl_l.coords(copy=True)
        assert np.allclose(cl_c[1] - cl_c[0], [0.5, 0.0], atol=1e-5)
        assert np.allclose(cl_c[3] - cl_c[2], [0.5, 0.0], atol=1e-5)

        # 8. FdebSolver (step == run, input layout preserved)
        fdeb_base = np.array([[float(c), float(r)] for r in range(4) for c in range(4)], dtype=np_dtype)
        l_fdeb = eg.Layout2d.from_coords(fdeb_base)
        fr_fdeb = eg.FdebSolver(g16, l_fdeb, cycles=2, i0=6, dtype=dtype_name)
        fs_fdeb = eg.FdebSolver(g16, l_fdeb, cycles=2, i0=6, dtype=None)
        assert fs_fdeb.dtype == dtype_name
        assert fr_fdeb.run() == "completed"
        fs_fdeb.step()
        fs_fdeb.run_steps(3)
        fs_fdeb.run()
        off_r = fr_fdeb.edge_offsets()
        pts_r = fr_fdeb.points()
        assert off_r.dtype == np.uint32 and off_r.shape == (g16.num_edges + 1,)
        assert int(off_r[0]) == 0 and int(off_r[-1]) == pts_r.shape[0]
        assert pts_r.dtype == np_dtype and pts_r.shape[1] == 2 and np.all(np.isfinite(pts_r))
        assert np.array_equal(off_r, fs_fdeb.edge_offsets())
        assert np.array_equal(pts_r, fs_fdeb.points())
        assert np.array_equal(l_fdeb.coords(copy=True), fdeb_base)

    # 9. Layering (cycle_edges, remove_cycles, assign_layers) & generate_layered_constraints
    cyc = eg.CsrGraph.from_edges(4, [(0, 1), (1, 2), (2, 0), (1, 3)], directed=True, sanitize=True)
    ce = eg.cycle_edges(cyc)
    assert ce.dtype == np.uint32 and ce.ndim == 2 and ce.shape[1] == 2 and ce.shape[0] >= 1
    dag = eg.remove_cycles(cyc)
    assert dag.is_directed and eg.cycle_edges(dag).shape == (0, 2)

    skip_edges = [(0, 1), (1, 2), (2, 3), (4, 3)]
    skip_dag = eg.CsrGraph.from_edges(5, skip_edges, directed=True, sanitize=True)
    lp = eg.assign_layers(skip_dag, method="longest_path")
    ns = eg.assign_layers(skip_dag, method="network_simplex")
    assert lp.dtype == np.uint32 and ns.dtype == np.uint32
    for u, v in skip_edges:
        assert int(lp[u]) < int(lp[v]) and int(ns[u]) < int(ns[v])
    lp_span = sum(int(lp[v]) - int(lp[u]) for u, v in skip_edges)
    ns_span = sum(int(ns[v]) - int(ns[u]) for u, v in skip_edges)
    assert ns_span < lp_span
    layered_cs = eg.generate_layered_constraints(skip_dag, ns, min_layer_distance=2.0)
    l_lay = eg.Layout2d.zeros(5, dtype="float64")
    eg.project_layout_axis(l_lay, 1, layered_cs)
    ys_lay = l_lay.coords(copy=True)[:, 1]
    for u, v in skip_edges:
        assert float(ys_lay[v] - ys_lay[u]) >= 2.0 - 1e-6

    results.append({
        "case": "v6_m4_solvers_and_postproc",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
    })

    # 10. Adversarial M4 Python checks (never PanicException)
    t1 = time.perf_counter()
    bad_l = eg.Layout2d.zeros(5)
    l16_nan = eg.Layout2d.zeros(16)
    l16_nan.coords(copy=False)[0, 0] = float("nan")
    m4_bad: list[tuple[str, Callable[[], Any]]] = [
        ("TFdpSolver/bad_backend", lambda: eg.TFdpSolver(g16, backend="bad")),
        ("TFdpSolver/momentum_1", lambda: eg.TFdpSolver(g16, momentum=1.0)),
        ("TFdpSolver/bool_iters", lambda: eg.TFdpSolver(g16, total_iterations=True)),
        ("TFdpSolver/dim_mismatch", lambda: eg.TFdpSolver(g16).step(bad_l)),
        ("TFdpSolver/nan_layout", lambda: eg.TFdpSolver(g16).step(l16_nan)),
        ("TFdpSolver.from_knn/bad_indptr", lambda: eg.TFdpSolver.from_knn([], [])),
        ("IbUmapSolver/zero_tau", lambda: eg.IbUmapSolver(g16, tau_scale=0.0)),
        ("IbUmapSolver/bad_backend", lambda: eg.IbUmapSolver(g16, backend="bad")),
        ("SgdSolver.snap_t_fdp/zero_rho_power", lambda: eg.SgdSolver.snap_t_fdp(g16, rho_power=0.0)),
        ("SgdSolver.pacmap_graph/zero_neighbors", lambda: eg.SgdSolver.pacmap_graph(g16, n_neighbors=0)),
        ("SgdSolver.sgd2/zero_batch", lambda: eg.SgdSolver.sgd2(g16, batch_size=0)),
        ("ForceAtlas2Solver/zero_jitter", lambda: eg.ForceAtlas2Solver(g16, jitter_tolerance=0.0)),
        ("ForceAtlas2Solver/dim_mismatch", lambda: eg.ForceAtlas2Solver(g16).step(bad_l)),
        ("YifanHuSolver/step_ratio_1", lambda: eg.YifanHuSolver(g16, step_ratio=1.0)),
        ("YifanHuSolver/dim_mismatch", lambda: eg.YifanHuSolver(g16).step(bad_l)),
        ("PrismOverlapRemoval/len_mismatch", lambda: eg.PrismOverlapRemoval([1.0, 1.0], [1.0])),
        ("PrismOverlapRemoval/neg_half", lambda: eg.PrismOverlapRemoval([1.0], [-0.5])),
        ("PrismOverlapRemoval/nan_layout", lambda: eg.PrismOverlapRemoval.from_radii([0.5] * 16).step(l16_nan)),
        ("SeparationConstraint/nan_gap", lambda: eg.SeparationConstraint(0, 1, float("nan"))),
        ("project_1d/cycle", lambda: eg.project_1d([0.0, 1.0], [eg.SeparationConstraint(0, 1, 1.0), eg.SeparationConstraint(1, 0, 1.0)])),
        ("project_layout_axis/bad_axis", lambda: eg.project_layout_axis(bad_l, 2, [])),
        ("generate_layered_constraints/len_mismatch", lambda: eg.generate_layered_constraints(dag, [0])),
        ("generate_layered_constraints/neg_dist", lambda: eg.generate_layered_constraints(dag, [0, 1, 2, 3], -1.0)),
        ("project_rectangle_no_overlap_2d/len_mismatch", lambda: eg.project_rectangle_no_overlap_2d(bad_l, [1.0], [1.0])),
        ("project_clustered_rectangle_no_overlap_2d/len_mismatch", lambda: eg.project_clustered_rectangle_no_overlap_2d(bad_l, [0], [1.0] * 5, [1.0] * 5)),
        ("cycle_edges/undirected", lambda: eg.cycle_edges(g16)),
        ("remove_cycles/undirected", lambda: eg.remove_cycles(g16)),
        ("assign_layers/undirected", lambda: eg.assign_layers(g16)),
        ("assign_layers/bad_method", lambda: eg.assign_layers(dag, method="unknown")),
        ("FdebSolver/dim_mismatch", lambda: eg.FdebSolver(g16, bad_l)),
        ("FdebSolver/cycles_too_large", lambda: eg.FdebSolver(g16, eg.Layout2d.zeros(16), cycles=30)),
        ("FdebSolver/nan_layout", lambda: eg.FdebSolver(g16, l16_nan)),
    ]
    for label, fn in m4_bad:
        _assert_non_panic_error(label, fn)

    results.append({
        "case": "v6_m4_adversarial_non_panic",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t1, 4),
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

if (typeof mod.selectMaxminSpPivots === 'function') {{
  const m2Symbols = [
    'KnnGraph',
    'Sparse12HopPairs',
    'SparseProbabilities',
    'DogVaradhanLandmarks',
    'EigenResult',
    'SpectralMomentStats',
    'TwoHopCsr',
    'TwoHopKnnResult',
    'HybridKnnResult',
    'selectMaxminSpPivots',
    'pivotMdsInit',
    'korenSpectralInit',
    'smallestLaplacianEigenpairs',
    'spectralMomentStats',
    'chebyshevHeatSpmv',
    'chebyshevDogJlEmbedding',
    'extract1hop2hopCsr',
    'extractSparse1hop2hopPairs',
    'buildPure2hopKnn',
    'compute2hopNeighborhoodSizes',
    'computeNodewise2hopPerplexities',
    'computeMedian2hopPerplexity',
    'buildHybridAdaptiveKnn',
    'partialBfsKnn',
    'computeKnnProbabilities',
    'computeFuzzySimplicialSet',
    'findAbParams',
    'computeHorvitzThompsonGamma',
    'computeRegularizedDogVaradhanLandmarks',
  ];
  for (const sym of m2Symbols) {{
    assert.equal(typeof indexMod[sym], 'function', `index.js must export ${{sym}}`);
    assert.equal(typeof mod[sym], 'function', `node entry must export ${{sym}}`);
  }}

  const gEdges = [];
  for (let r = 0; r < 4; r += 1) {{
    for (let c = 0; c < 4; c += 1) {{
      const u = r * 4 + c;
      if (c + 1 < 4) gEdges.push(u, u + 1);
      if (r + 1 < 4) gEdges.push(u, u + 4);
    }}
  }}
  const g16 = CsrGraph.fromEdgeList(16, new Uint32Array(gEdges));

  const piv = mod.selectMaxminSpPivots(g16, 6, 42, 2);
  assert.ok(piv instanceof Uint32Array && piv.length === 6);

  const lPmds = mod.pivotMdsInit(g16, 6, 42, 2, 1.0);
  const lPmdsUnnorm = mod.pivotMdsInit(g16, 6, 42, 2, null);
  const lKoren = mod.korenSpectralInit(g16, 1.0, 1e-4, 300, 42);
  assert.equal(lPmds.len(), 16);
  assert.equal(lPmdsUnnorm.len(), 16);
  assert.equal(lKoren.len(), 16);
  assert.equal(lPmds.isAllFinite(), true);
  assert.equal(lKoren.isAllFinite(), true);

  const eig = mod.smallestLaplacianEigenpairs(g16, 4, 1e-4, 300, 42);
  assert.equal(eig.numNodes, 16);
  assert.equal(eig.numModes, 4);
  const invD = eig.invSqrtDeg();
  assert.equal(eig.lambdas().length, 4);
  assert.equal(eig.vectors().length, 64);

  const mStats = mod.spectralMomentStats(g16);
  assert.ok(mStats.sigmaLambdaSq > 0 && Math.abs(mStats.meanDegree - 3.0) < 1e-12);

  const statVec = new Float64Array(16);
  for (let i = 0; i < 16; i += 1) statVec[i] = 1.0 / invD[i];
  const heat = mod.chebyshevHeatSpmv(g16, statVec, 1, 2.0);
  for (let i = 0; i < 16; i += 1) assert.ok(Math.abs(heat[i] - statVec[i]) < 1e-9);

  const jl = mod.chebyshevDogJlEmbedding(g16, 12, 25, 42, true, 1e-4);
  assert.ok(jl instanceof Float64Array && jl.length === 16 * 12);

  const twoHop = mod.extract1hop2hopCsr(g16);
  const h1 = twoHop.hop1();
  const h2 = twoHop.hop2();
  assert.equal(h1.numEdges(), 24);
  assert.ok(h2.numEdges() > 0);

  const pureRes = mod.buildPure2hopKnn(g16);
  const pureKnn = pureRes.knn();
  assert.deepEqual(Array.from(pureRes.k2Deg()), Array.from(mod.compute2hopNeighborhoodSizes(g16)));
  const knnRt = mod.KnnGraph.fromCsr(pureKnn.numNodes, pureKnn.indptr(), pureKnn.indices(), pureKnn.distances());
  assert.equal(knnRt.nnz, pureKnn.nnz);

  const perps = mod.computeNodewise2hopPerplexities(g16);
  const medPerp = mod.computeMedian2hopPerplexity(g16);
  assert.ok(perps.length === 16 && Number.isFinite(medPerp));

  const hybRes = mod.buildHybridAdaptiveKnn(g16, jl, 12, perps);
  const hybKnn = hybRes.knn();
  assert.equal(hybRes.nFilledResidual + hybRes.nPure2hopRows, 16);

  const bfsKnn = mod.partialBfsKnn(g16, 5, 42);
  assert.equal(bfsKnn.nnz, 16 * 5);

  const probs = mod.computeKnnProbabilities(hybKnn, perps);
  const pSum = probs.values().reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(pSum - 1.0) < 1e-10);

  const fuzzy = mod.computeFuzzySimplicialSet(hybKnn);
  assert.ok(fuzzy.values().every((v) => v > 1e-7 && v <= 1.0));

  const ab = mod.findAbParams(0.1, 1.0);
  assert.ok(Math.abs(ab[0] - 1.577) < 0.02 && Math.abs(ab[1] - 0.895) < 0.02);
  assert.ok(Math.abs(mod.computeHorvitzThompsonGamma(1.0, 1600, 16) - 10.0) < 1e-12);

  const lm = mod.computeRegularizedDogVaradhanLandmarks(g16, 25, 16, 42);
  assert.equal(lm.numNodes, 16);
  assert.ok(lm.coarseTable() instanceof Float64Array);
  const alt = lm.altDistances(new Uint32Array([0, 15]), new Uint32Array([15, 0]));
  assert.ok(Math.abs(alt[0] - alt[1]) < 1e-12);

  const assertM2JsErr = (fn, label) => {{
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
  assertM2JsErr(() => mod.selectMaxminSpPivots(g16, 4, 1.5), 'm2 bad seed');
  assertM2JsErr(() => mod.pivotMdsInit(g16, 4, 42, 2, -1), 'm2 neg targetStd');
  assertM2JsErr(() => mod.chebyshevHeatSpmv(g16, new Float64Array([Number.NaN]), 1, 1.0), 'm2 nan vec');
  assertM2JsErr(() => mod.KnnGraph.fromCsr(2, new Uint32Array([0, 1, 1]), new Uint32Array([1]), new Float64Array([-1.0])), 'm2 neg dist');
  assertM2JsErr(() => lm.altDistances(new Uint32Array([0]), new Uint32Array([99])), 'm2 alt oob');
  assertM2JsErr(() => mod.computeHorvitzThompsonGamma(-1.0, 10, 4), 'm2 neg gamma');
  assertM2JsErr(() => mod.findAbParams(-0.1, 1.0), 'm2 neg minDist');
  assertM2JsErr(() => mod.computeFuzzySimplicialSet(hybKnn, 0), 'm2 zero sigmaIters');

  lm.free?.();
  fuzzy.free?.();
  probs.free?.();
  bfsKnn.free?.();
  hybKnn.free?.();
  hybRes.free?.();
  knnRt.free?.();
  pureKnn.free?.();
  pureRes.free?.();
  h1.free?.();
  h2.free?.();
  twoHop.free?.();
  mStats.free?.();
  eig.free?.();
  lKoren.free?.();
  lPmdsUnnorm.free?.();
  lPmds.free?.();
  g16.free?.();
}}

if (typeof mod.TsNetSolver === 'function') {{
  const m3Symbols = [
    'CoarsenedGraphResult',
    'GuMapSolver',
    'KMeansResult',
    'KamadaKawaiSolver',
    'LouvainResult',
    'ManifoldLayout2d',
    'SgdSolver',
    'SparseBarrierStats',
    'SpatialIntruderRefiner',
    'SpatialRefinerRoundStats',
    'SpatialRefinerStats',
    'StressMajorizationSolver',
    'TsNetSolver',
    'coarsenGraph',
    'kmeansPlusPlus',
    'labelPropagation',
    'louvainCommunities',
    'projectClusteredLayout',
    'spectralClustering',
  ];
  for (const sym of m3Symbols) {{
    assert.equal(typeof indexMod[sym], 'function', `index.js must export ${{sym}}`);
    assert.equal(typeof mod[sym], 'function', `node entry must export ${{sym}}`);
  }}

  const bbEdges = [];
  for (const off of [0, 6]) {{
    for (let i = 0; i < 6; i += 1) {{
      for (let j = i + 1; j < 6; j += 1) bbEdges.push(off + i, off + j);
    }}
  }}
  bbEdges.push(5, 6);
  const bb = CsrGraph.fromEdgeList(12, new Uint32Array(bbEdges));
  const km = mod.kmeansPlusPlus([10, 10, 10.2, 9.8, 9.8, 10.1, -10, -10, -9.9, -10.2, -10.1, -9.8], 6, 2, 2, 100, 1e-4, 3, 42);
  assert.deepEqual(Array.from(km.labels()), [0, 0, 0, 1, 1, 1]);
  assert.deepEqual(Array.from(mod.spectralClustering(bb, 2, 300, 1e-4, 100, 1e-4, 42)), [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]);
  const lv = mod.louvainCommunities(bb, 100, 1.0, 1e-7, 42);
  assert.equal(lv.numCommunities, 2);
  assert.deepEqual(Array.from(mod.labelPropagation(bb, 100, 42)), [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]);
  const coarse = mod.coarsenGraph(bb, [100, 100, 100, 100, 100, 100, 999, 999, 999, 999, 999, 999]);
  assert.equal(coarse.numClusters, 2);
  const cl = Layout2d.fromCoords(new Float32Array([-2, 0, 2, 0]));
  const proj = mod.projectClusteredLayout(cl, coarse.nodeToCoarse(), 0.2, 42);
  assert.equal(proj.len(), 12);
  proj.free?.();
  cl.free?.();
  coarse.free?.();
  lv.free?.();
  km.free?.();

  const g16Edges = [];
  for (let r = 0; r < 4; r += 1) {{
    for (let c = 0; c < 4; c += 1) {{
      const u = r * 4 + c;
      if (c + 1 < 4) g16Edges.push(u, u + 1);
      if (r + 1 < 4) g16Edges.push(u, u + 4);
    }}
  }}
  const g16 = CsrGraph.fromEdgeList(16, new Uint32Array(g16Edges));
  const initArr = new Float32Array(32);
  for (let i = 0; i < 16; i += 1) {{
    initArr[2 * i] = 0.7 * Math.cos(0.83 * i + 0.15) + 0.11 * i / 16.0;
    initArr[2 * i + 1] = 0.6 * Math.sin(1.13 * i - 0.21) - 0.09 * i / 16.0;
  }}

  const tsR = mod.TsNetSolver.pure2Hop(g16, {{ backend: 'fft', iterationsStage1: 4, iterationsStage2: 4, iterationsStage3: 4, seed: 42 }});
  const tsS = mod.TsNetSolver.pure2Hop(g16, {{ backend: 'fft', iterationsStage1: 4, iterationsStage2: 4, iterationsStage3: 4, seed: 42 }});
  const lR = Layout2d.fromCoords(initArr);
  const lS = Layout2d.fromCoords(initArr);
  assert.equal(tsR.run(lR), 'completed');
  tsS.runSteps(lS, 5);
  while (!tsS.isFinished) tsS.step(lS);
  assert.deepEqual(Array.from(lR.coordsView()), Array.from(lS.coordsView()));
  tsR.free?.();
  tsS.free?.();
  lR.free?.();
  lS.free?.();

  const ml = mod.ManifoldLayout2d.fromCoords('torus', new Float32Array([1.2, -0.1, 0.9, 0.2]));
  assert.equal(ml.kind, 'torus');
  ml.projectAll();
  assert.ok(ml.geodesicDistance(0, 1) >= 0);
  ml.free?.();

  const sgdR = mod.SgdSolver.sparseBarrierDog(g16, {{ degree: 15, numLandmarks: 8, tMax: 8, seed: 42 }});
  const sgdL = Layout2d.fromCoords(initArr);
  assert.equal(sgdR.run(sgdL), 'completed');
  assert.ok(sgdL.isAllFinite());
  sgdR.free?.();
  sgdL.free?.();

  const dirG = CsrGraph.fromEdgeList(3, new Uint32Array([0, 1, 1, 2]), null, true, true);
  let threwDir = false;
  try {{
    mod.spectralClustering(dirG, 2);
  }} catch (err) {{
    assert.ok(!(err instanceof WebAssembly.RuntimeError));
    threwDir = true;
  }}
  assert.ok(threwDir, 'spectralClustering must reject directed graphs');
  dirG.free?.();
  g16.free?.();
  bb.free?.();
}}

if (typeof mod.TFdpSolver === 'function') {{
  const m4Symbols = [
    'FdebSolver',
    'ForceAtlas2Solver',
    'IbUmapSolver',
    'PrismOverlapRemoval',
    'SeparationConstraint',
    'TFdpSolver',
    'YifanHuSolver',
    'assignLayers',
    'cycleEdges',
    'generateLayeredConstraints',
    'project1d',
    'projectClusteredRectangleNoOverlap2d',
    'projectLayoutAxis',
    'projectRectangleNoOverlap2d',
    'removeCycles',
  ];
  for (const sym of m4Symbols) {{
    assert.equal(typeof indexMod[sym], 'function', `index.js must export ${{sym}}`);
    assert.equal(typeof mod[sym], 'function', `node entry must export ${{sym}}`);
  }}
  assert.equal(typeof indexMod.runProgressive, 'function', 'index.js must export runProgressive');
  assert.equal(typeof indexMod.LayoutNd, 'function', 'index.js must export LayoutNd');
  assert.equal(typeof indexMod.LayoutNd.prototype.coordsView, 'function', 'LayoutNd must expose coordsView()');

  const g16Edges = [];
  for (let r = 0; r < 4; r += 1) {{
    for (let c = 0; c < 4; c += 1) {{
      const u = r * 4 + c;
      if (c + 1 < 4) g16Edges.push(u, u + 1);
      if (r + 1 < 4) g16Edges.push(u, u + 4);
    }}
  }}
  const g16 = CsrGraph.fromEdgeList(16, new Uint32Array(g16Edges));
  const init16 = new Float32Array(32);
  for (let i = 0; i < 16; i += 1) {{
    init16[2 * i] = 0.7 * Math.cos(0.83 * i + 0.15) + 0.11 * i / 16.0;
    init16[2 * i + 1] = 0.6 * Math.sin(1.13 * i - 0.21) - 0.09 * i / 16.0;
  }}

  const checkStepEqRun = (mkSolver, initArr, stepChunk, label) => {{
    const sr = mkSolver();
    const ss = mkSolver();
    const lr = Layout2d.fromCoords(initArr);
    const ls = Layout2d.fromCoords(initArr);
    const stR = sr.run(lr);
    assert.ok(stR === 'completed' || stR === 'converged', `${{label}} run status=${{stR}}`);
    ss.runSteps(ls, stepChunk);
    while (!ss.isFinished) ss.step(ls);
    assert.deepEqual(Array.from(lr.coordsView()), Array.from(ls.coordsView()), `${{label}} step != run`);
    assert.equal(lr.isAllFinite(), true, `${{label}} non-finite`);
    sr.free?.();
    ss.free?.();
    lr.free?.();
    ls.free?.();
  }};

  for (const bk of ['ibfft', 'barnes_hut', 'exact']) {{
    checkStepEqRun(() => new mod.TFdpSolver(g16, {{ backend: bk, totalIterations: 8, seed: 42 }}), init16, 3, `TFdp/${{bk}}`);
    checkStepEqRun(() => new mod.IbUmapSolver(g16, {{ backend: bk, nEpochs: 6, seed: 42 }}), init16, 2, `IbUmap/${{bk}}`);
  }}
  checkStepEqRun(() => mod.SgdSolver.snapTFdp(g16, {{ iterations: 8, negativesPerEdge: 3, seed: 42 }}), init16, 3, 'Sgd.snapTFdp');
  checkStepEqRun(() => mod.SgdSolver.pacmapGraph(g16, {{ nNeighbors: 5, phase1Iters: 3, phase2Iters: 3, phase3Iters: 4, seed: 42 }}), init16, 4, 'Sgd.pacmapGraph');
  checkStepEqRun(() => mod.SgdSolver.sgd2(g16, {{ tMax: 8, batchSize: 16, seed: 42 }}), init16, 3, 'Sgd.sgd2');
  checkStepEqRun(() => mod.SgdSolver.full(g16, {{ tMax: 8, kind: 'squared', seed: 42 }}), init16, 3, 'Sgd.squared');
  checkStepEqRun(() => new mod.ForceAtlas2Solver(g16, {{ iterations: 8, useBarnesHut: true, seed: 42 }}), init16, 3, 'FA2/bh');
  checkStepEqRun(() => new mod.ForceAtlas2Solver(g16, {{ iterations: 8, useBarnesHut: false, seed: 42 }}), init16, 3, 'FA2/exact');
  checkStepEqRun(() => new mod.YifanHuSolver(g16, {{ maxIterationsPerLevel: 8, minCoarsenSize: 6, minCoarsenRatio: 0.9, seed: 42 }}), init16, 3, 'YifanHu');

  const prismInit = init16.map((v) => v * 0.15);
  checkStepEqRun(() => mod.PrismOverlapRemoval.fromRadii(new Float32Array(16).fill(0.4), {{ maxIterations: 40, seed: 42 }}), prismInit, 2, 'PRISM');

  // FdebSolver step() == run() and input layout reuse (no value-drop bug)
  const fdebInit = new Float32Array(32);
  for (let r = 0; r < 4; r += 1) {{
    for (let c = 0; c < 4; c += 1) {{
      fdebInit[2 * (r * 4 + c)] = c;
      fdebInit[2 * (r * 4 + c) + 1] = r;
    }}
  }}
  const lFdeb = Layout2d.fromCoords(fdebInit);
  const fdebR = new mod.FdebSolver(g16, lFdeb, {{ cycles: 2, i0: 6 }});
  const fdebS = new mod.FdebSolver(g16, lFdeb, {{ cycles: 2, i0: 6 }});
  assert.equal(lFdeb.len(), 16, 'FdebSolver must not drop input Layout2d');
  assert.equal(fdebR.run(), 'completed');
  fdebS.step();
  fdebS.runSteps(3);
  while (!fdebS.isFinished) fdebS.step();
  assert.deepEqual(Array.from(fdebR.edgeOffsets()), Array.from(fdebS.edgeOffsets()));
  assert.deepEqual(Array.from(fdebR.points()), Array.from(fdebS.points()));
  assert.deepEqual(Array.from(lFdeb.coordsView()), Array.from(fdebInit));
  fdebR.free?.();
  fdebS.free?.();
  lFdeb.free?.();

  // runProgressive parity vs run()
  const pRun = new mod.ForceAtlas2Solver(g16, {{ iterations: 6, useBarnesHut: false, seed: 42 }});
  const pProg = new mod.ForceAtlas2Solver(g16, {{ iterations: 6, useBarnesHut: false, seed: 42 }});
  const lpRun = Layout2d.fromCoords(init16);
  const lpProg = Layout2d.fromCoords(init16);
  pRun.run(lpRun);
  const frames = [];
  const pCtx = await indexMod.runProgressive(pProg, lpProg, (ctx) => frames.push(ctx.iteration), {{ frameBudgetMs: 10, stepsPerYield: 1 }});
  assert.equal(pCtx.status, 'completed');
  assert.ok(frames.length >= 1);
  assert.deepEqual(Array.from(lpRun.coordsView()), Array.from(lpProg.coordsView()));
  pRun.free?.(); pProg.free?.(); lpRun.free?.(); lpProg.free?.();

  // Constraints & layering smoke + adversarial JsError checks
  const c0 = new mod.SeparationConstraint(0, 1, 2.0);
  const arr1d = new Float32Array([1.0, 0.0]);
  mod.project1d(arr1d, [c0]);
  assert.ok(arr1d[1] - arr1d[0] >= 2.0 - 1e-5);
  c0.free?.();

  const cyc = CsrGraph.fromEdgeList(4, new Uint32Array([0, 1, 1, 2, 2, 0, 1, 3]), null, true, true);
  assert.ok(mod.cycleEdges(cyc).length >= 2);
  const dag = mod.removeCycles(cyc);
  assert.equal(mod.cycleEdges(dag).length, 0);
  const layersNs = mod.assignLayers(dag, 'networkSimplex');
  assert.equal(layersNs.length, 4);
  const lCs = mod.generateLayeredConstraints(dag, layersNs, 2.0);
  const lDag = Layout2d.zeros(4);
  mod.projectLayoutAxis(lDag, 1, lCs);
  for (const c of lCs) c.free?.();
  lDag.free?.();
  dag.free?.();
  cyc.free?.();

  const assertM4JsErr = (fn, label) => {{
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
  const badL = Layout2d.zeros(3);
  assertM4JsErr(() => new mod.TFdpSolver(g16, {{ backend: 'bad' }}), 'm4 tfdp bad backend');
  assertM4JsErr(() => new mod.TFdpSolver(g16).step(badL), 'm4 tfdp dim mismatch');
  assertM4JsErr(() => new mod.IbUmapSolver(g16, {{ tauScale: 0 }}), 'm4 ibumap zero tau');
  assertM4JsErr(() => mod.SgdSolver.snapTFdp(g16, {{ rhoPower: 0 }}), 'm4 snapTFdp bad rhoPower');
  assertM4JsErr(() => mod.SgdSolver.pacmapGraph(g16, {{ nNeighbors: 0 }}), 'm4 pacmapGraph bad nNeighbors');
  assertM4JsErr(() => mod.SgdSolver.sgd2(g16, {{ batchSize: 0 }}), 'm4 sgd2 bad batchSize');
  assertM4JsErr(() => new mod.ForceAtlas2Solver(g16, {{ jitterTolerance: 0 }}), 'm4 fa2 bad jitter');
  assertM4JsErr(() => new mod.YifanHuSolver(g16, {{ stepRatio: 1.0 }}), 'm4 yh bad stepRatio');
  assertM4JsErr(() => new mod.PrismOverlapRemoval(new Float32Array([1, 1]), new Float32Array([1])), 'm4 prism len mismatch');
  assertM4JsErr(() => new mod.SeparationConstraint(0, 1, Number.NaN), 'm4 constraint nan');
  assertM4JsErr(() => mod.assignLayers(g16, 'longestPath'), 'm4 assignLayers undir');
  assertM4JsErr(() => new mod.FdebSolver(g16, badL), 'm4 fdeb dim mismatch');
  badL.free?.();
  g16.free?.();
}}

if (typeof mod.clusterAreaFaithfulness === 'function') {{
  assert.equal(typeof mod.ClusterAreaReport, 'function', 'mod missing ClusterAreaReport');
  assert.equal(typeof mod.clusterAreaFaithfulnessDetail, 'function', 'mod missing clusterAreaFaithfulnessDetail');
  assert.equal(typeof indexMod.ClusterAreaReport, 'function', 'indexMod missing ClusterAreaReport');
  assert.equal(typeof indexMod.clusterAreaFaithfulness, 'function', 'indexMod missing clusterAreaFaithfulness');
  assert.equal(typeof indexMod.clusterAreaFaithfulnessDetail, 'function', 'indexMod missing clusterAreaFaithfulnessDetail');

  const imbCoords = new Float32Array(48);
  const imbLabels = new Uint32Array(24);
  for (let i = 0; i < 24; i++) {{
    imbCoords[2 * i] = i < 20 ? -2.0 + 0.5 * (i / 19) : 1.5 + 0.5 * ((i - 20) / 3);
    imbCoords[2 * i + 1] = Math.sin(i * 0.7) * 0.2;
    imbLabels[i] = i < 20 ? 0 : 1;
  }}
  const lImb = mod.Layout2d.fromCoords(imbCoords);
  const imbEdges = [];
  for (let i = 0; i < 23; i++) imbEdges.push(i, i + 1);
  const gImb = mod.CsrGraph.fromEdgeList(24, new Uint32Array(imbEdges));

  const cafDef = mod.clusterAreaFaithfulness(lImb, imbLabels);
  const detDef = mod.clusterAreaFaithfulnessDetail(lImb, imbLabels);
  assert.ok(detDef instanceof mod.ClusterAreaReport);
  assert.ok(Math.abs(detDef.caf - cafDef) < 1e-12);
  const recDef = detDef.toRecord();
  assert.ok(Math.abs(recDef.caf - cafDef) < 1e-12);
  for (const k of ['caf', 'purityAdj', 'propVis', 'resDensity', 'resDeconv', 'resCombined', 'spatialCoherence', 'inFrac']) {{
    assert.ok(Number.isFinite(detDef[k]) && detDef[k] >= 0 && detDef[k] <= 1, `bad ${{k}}`);
    assert.ok(Math.abs(recDef[k] - detDef[k]) < 1e-12, `record mismatch ${{k}}`);
  }}
  assert.ok(Number.isFinite(detDef.effectiveBoundaryR95) && detDef.effectiveBoundaryR95 > 0);
  assert.ok(Math.abs(recDef.effectiveBoundaryR95 - detDef.effectiveBoundaryR95) < 1e-12);
  assert.ok(Number.isInteger(detDef.gEff) && detDef.gEff >= 12);
  assert.equal(recDef.gEff, detDef.gEff);
  assert.equal(detDef.numClusters, 2);
  assert.equal(detDef.numNodes, 24);
  if (typeof detDef.clusterIds === 'function') {{
    assert.equal(detDef.clusterIds().length, 2);
    assert.equal(detDef.counts().length, 2);
    const pSum = detDef.p().reduce((a, b) => a + b, 0);
    const qSum = detDef.q().reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(pSum - 1.0) < 1e-12);
    assert.ok(Math.abs(qSum - 1.0) < 1e-12);
    for (const m of ['purity', 'resolution', 'resDensityPerCluster', 'resDeconvPerCluster', 'cafPerCluster', 'coherence']) {{
      assert.equal(detDef[m]().length, 2, `bad len ${{m}}`);
    }}
  }}
  detDef.free?.();

  const cafCustom = mod.clusterAreaFaithfulness(lImb, imbLabels, 64, 0.9, 8.0, 0.2, 2.5, 0.7);
  const l5 = mod.Layout2d.fromCoords(new Float32Array([0, 0, 1, 0, 0, 1, 3, 3, 4, 3]));
  const l5Arr = [100, 5, 100, 4294967295, 5];
  assert.equal(mod.clusterAreaFaithfulness(l5, new Uint32Array(l5Arr)), mod.clusterAreaFaithfulness(l5, l5Arr));
  l5.free?.();
  for (const t of ['clusterAreaFaithfulness', 'cluster_area_faithfulness', 'gcaf']) {{
    const rep = mod.evaluateMetrics(gImb, lImb, [t], null, null, null, null, imbLabels, null, null, null, null, null, null, 64, 0.9, 8.0, 0.2, 2.5, 0.7);
    assert.ok(Math.abs(rep.clusterAreaFaithfulness - cafCustom) < 1e-12);
    assert.ok(Math.abs(rep.toRecord().clusterAreaFaithfulness - cafCustom) < 1e-12);
    rep.free?.();
  }}

  const detInf = mod.clusterAreaFaithfulnessDetail(lImb, imbLabels, null, null, null, null, Number.POSITIVE_INFINITY, null);
  assert.ok(Math.abs(detInf.inFrac - 1.0) < 1e-12);
  detInf.free?.();
  const detG05 = mod.clusterAreaFaithfulnessDetail(lImb, imbLabels, null, null, null, null, null, 0.5);
  const detG20 = mod.clusterAreaFaithfulnessDetail(lImb, imbLabels, null, null, null, null, null, 2.0);
  if (typeof detG05.p === 'function') assert.ok(detG05.p()[0] < detG20.p()[0]);
  detG05.free?.(); detG20.free?.();

  // Degenerate boundaries: N=0, N=1, K=1, collapsed
  const l0 = mod.Layout2d.zeros(0);
  const d0 = mod.clusterAreaFaithfulnessDetail(l0, []);
  assert.equal(d0.caf, 0); assert.equal(d0.effectiveBoundaryR95, 0); assert.equal(d0.gEff, 0);
  assert.equal(d0.numClusters, 0); assert.equal(d0.numNodes, 0);
  d0.free?.(); l0.free?.();
  const l1 = mod.Layout2d.fromCoords(new Float32Array([1, 2]));
  assert.equal(mod.clusterAreaFaithfulness(l1, [5]), 0);
  l1.free?.();
  const l4 = mod.Layout2d.fromCoords(new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]));
  assert.equal(mod.clusterAreaFaithfulness(l4, [3, 3, 3, 3]), 0);
  l4.free?.();
  const lCol = mod.Layout2d.fromCoords(new Float32Array([2, -1, 2, -1, 2, -1, 2, -1, 2, -1, 2, -1]));
  assert.equal(mod.clusterAreaFaithfulness(lCol, [0, 0, 0, 1, 1, 1]), 0);
  lCol.free?.();

  // Lazy gMax=4 passes without gcaf target
  const lazyOk = mod.evaluateMetrics(gImb, lImb, ['aspectRatio'], null, null, null, null, null, null, null, null, null, null, null, 4);
  assert.ok(Number.isFinite(lazyOk.aspectRatio));
  assert.equal(lazyOk.clusterAreaFaithfulness, undefined);
  lazyOk.free?.();

  const assertGcafErr = (fn, label) => {{
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
  assertGcafErr(() => mod.clusterAreaFaithfulness(lImb, [0, 1]), 'gcaf label len mismatch');
  const restLabels = Array.from(imbLabels.slice(1));
  const invalidCommunityLabels = [
    ['neg', [-1, ...restLabels]],
    ['float', [1.5, ...restLabels]],
    ['nan', [Number.NaN, ...restLabels]],
    ['inf', [Number.POSITIVE_INFINITY, ...restLabels]],
    ['overflow', [4294967296, ...restLabels]],
    ['strElem', ['bad', ...restLabels]],
    ['nullElem', [null, ...restLabels]],
    ['undefElem', [undefined, ...restLabels]],
    ['int32Array', new Int32Array([-1, ...restLabels])],
    ['float64Array', new Float64Array([1.5, ...restLabels])],
    ['arrayLikeObj', {{ length: 24 }}],
    ['string', '000000000000000000001111'],
  ];
  for (const [tag, badLabels] of invalidCommunityLabels) {{
    assertGcafErr(() => mod.clusterAreaFaithfulness(lImb, badLabels), `gcaf bad labels (${{tag}})`);
    assertGcafErr(() => mod.clusterAreaFaithfulnessDetail(lImb, badLabels), `gcafDetail bad labels (${{tag}})`);
    assertGcafErr(
      () => mod.evaluateMetrics(gImb, lImb, ['gcaf'], null, null, null, null, badLabels),
      `gcaf eval bad labels (${{tag}})`,
    );
  }}
  for (const [tag, nullish] of [['null', null], ['undefined', undefined]]) {{
    assertGcafErr(() => mod.clusterAreaFaithfulness(lImb, nullish), `gcaf nullish labels (${{tag}})`);
    assertGcafErr(() => mod.clusterAreaFaithfulnessDetail(lImb, nullish), `gcafDetail nullish labels (${{tag}})`);
    assertGcafErr(
      () => mod.evaluateMetrics(gImb, lImb, ['gcaf'], null, null, null, null, nullish),
      `gcaf eval nullish labels (${{tag}})`,
    );
  }}
  const nanCoords = new Float32Array(imbCoords); nanCoords[0] = Number.NaN;
  const lNan = mod.Layout2d.fromCoords(nanCoords);
  assertGcafErr(() => mod.clusterAreaFaithfulness(lNan, imbLabels), 'gcaf nan layout');
  lNan.free?.();
  for (const bg of [11, -1, 1.5]) assertGcafErr(() => mod.clusterAreaFaithfulness(lImb, imbLabels, bg), `gcaf gMax=${{bg}}`);
  for (const bs of [0, -0.5, Number.NaN, Number.POSITIVE_INFINITY, 400]) assertGcafErr(() => mod.clusterAreaFaithfulness(lImb, imbLabels, null, bs), `gcaf sigma=${{bs}}`);
  for (const bv of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {{
    assertGcafErr(() => mod.clusterAreaFaithfulness(lImb, imbLabels, null, null, bv), `gcaf kappa=${{bv}}`);
    assertGcafErr(() => mod.clusterAreaFaithfulness(lImb, imbLabels, null, null, null, bv), `gcaf tau=${{bv}}`);
    assertGcafErr(() => mod.clusterAreaFaithfulness(lImb, imbLabels, null, null, null, null, null, bv), `gcaf gamma=${{bv}}`);
  }}
  for (const br of [0, -1, Number.NaN, Number.NEGATIVE_INFINITY]) assertGcafErr(() => mod.clusterAreaFaithfulness(lImb, imbLabels, null, null, null, null, br), `gcaf r95=${{br}}`);
  assertGcafErr(() => mod.evaluateMetrics(gImb, lImb, ['gcaf']), 'gcaf eval missing labels');
  assertGcafErr(() => mod.evaluateMetrics(gImb, lImb, ['clusterAreaFaithfulness'], null, null, null, null, imbLabels, null, null, null, null, null, null, 4), 'gcaf eval lazy gMax=4');

  // G6 nodeWeights & clusterWeights parity + errors
  const imbNw = new Float64Array(24);
  for (let i = 0; i < 24; i++) imbNw[i] = i < 20 ? 1.5 : 3.0;
  const imbCw = new Float64Array([30.0, 12.0]);
  const cafNw = mod.clusterAreaFaithfulness(lImb, imbLabels, null, null, null, null, null, 2.0, imbNw, null);
  const cafCw = mod.clusterAreaFaithfulness(lImb, imbLabels, null, null, null, null, null, 2.0, null, imbCw);
  assert.ok(Math.abs(cafNw - cafCw) < 1e-12);
  const evalCw = mod.evaluateMetrics(gImb, lImb, ['gcaf'], null, null, null, null, imbLabels, null, null, null, null, null, null, null, null, null, null, null, 2.0, null, imbCw);
  assert.ok(Math.abs(evalCw.clusterAreaFaithfulness - cafCw) < 1e-12);
  evalCw.free?.();
  assertGcafErr(() => mod.clusterAreaFaithfulness(lImb, imbLabels, null, null, null, null, null, null, imbNw, imbCw), 'gcaf both weights');
  assertGcafErr(() => mod.clusterAreaFaithfulness(lImb, imbLabels, null, null, null, null, null, null, null, [0.0, 1.0]), 'gcaf zero weight');
  assertGcafErr(() => mod.clusterAreaFaithfulness(lImb, imbLabels, null, null, null, null, null, null, null, [true, false]), 'gcaf bool weight');

  lImb.free?.();
  gImb.free?.();
}}

assert.equal(typeof mod.InfoMap, 'undefined');
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
        "sampled_np1",
        "sampled_np2",
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
            fg, l32, targets, num_pivots=4, num_samples=5, seed=42
        )
        dt_edges, dt_tris = eg.delaunay_triangulation(l32)

        py_results[name] = {
            "scale_normalized_stress": float(eg.scale_normalized_stress(fg, l32)),
            "crossing_number": cr_dual,
            "sweep_crossing_number": cr_sweep,
            "neighborhood_preservation_1hop": float(eg.neighborhood_preservation_1hop(fg, l32)),
            "neighborhood_preservation_2hop_v6": float(eg.neighborhood_preservation_2hop_v6(fg, l32)),
            "sampled_np1_sub": float(eg.sampled_np1(fg, l32, num_samples=5, seed=42)),
            "sampled_np2_sub": float(eg.sampled_np2(fg, l32, num_samples=5, seed=42)),
            "sampled_np1_full": float(eg.sampled_np1(fg, l32, num_samples=n, seed=42)),
            "sampled_np2_full": float(eg.sampled_np2(fg, l32, num_samples=n, seed=42)),
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
    ['sns', 'crossing_number', 'np1', 'np2', 'sampledNp1', 'sampledNp2', 'drng', 'dgg', 'aspect_ratio'],
    4,
    5,
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
    sampled_np1_sub: mod.sampledNp1(g, l, 5, 42),
    sampled_np2_sub: mod.sampledNp2(g, l, 5, 42),
    sampled_np1_full: mod.sampledNp1(g, l, c.n, 42),
    sampled_np2_full: mod.sampledNp2(g, l, c.n, 42),
    prism_sigma_dist: pd.sigmaDist,
    prism_sigma_disp: pd.sigmaDisp,
    evaluate_metrics: {
      scale_normalized_stress: rec.scaleNormalizedStress,
      crossing_number: rec.crossingNumber,
      neighborhood_preservation_1hop: rec.neighborhoodPreservation1Hop,
      neighborhood_preservation_2hop: rec.neighborhoodPreservation2Hop,
      sampled_np1: rec.sampledNp1,
      sampled_np2: rec.sampledNp2,
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
            ("sampled_np1_sub", "sampled_np1_sub"),
            ("sampled_np2_sub", "sampled_np2_sub"),
            ("sampled_np1_full", "sampled_np1_full"),
            ("sampled_np2_full", "sampled_np2_full"),
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


def run_cross_target_m2_parity(eg: Any, wasm_pkg: pathlib.Path) -> list[dict[str, Any]]:
    """Verify Python <-> WASM numerical and structural parity across all 19 M2 APIs."""
    import numpy as np

    if not hasattr(eg, "select_maxmin_sp_pivots"):
        return []
    node_entry = wasm_pkg / "egraph_wasm.node.js"
    index_entry = wasm_pkg / "index.js"
    if not index_entry.exists():
        return []

    t0 = time.perf_counter()
    n, edges = _make_m2_canonical_graph_spec()
    g = eg.CsrGraph.from_edges(n, edges, directed=False, sanitize=True)

    pivots = eg.select_maxmin_sp_pivots(g, 12, seed=42, min_degree=2)
    pmds_f32 = eg.pivot_mds_init(
        g, num_pivots=12, seed=42, min_degree=2, target_std=1.0, dtype="float32"
    ).coords(copy=True)
    pmds_unnorm = eg.pivot_mds_init(
        g, num_pivots=12, seed=42, min_degree=2, target_std=None, dtype="float32"
    ).coords(copy=True)
    koren_f32 = eg.koren_spectral_init(
        g, target_std=1.0, tol=1e-4, max_iters=300, seed=42, dtype="float32"
    ).coords(copy=True)
    lams, evecs, inv_d, used_dense = eg.smallest_laplacian_eigenpairs(
        g, n_modes=4, tol=1e-4, max_iters=300, seed=42
    )
    mstats = eg.spectral_moment_stats(g)

    heat_in = np.column_stack([
        np.cos(0.17 * np.arange(n, dtype=np.float64)),
        np.sin(0.29 * np.arange(n, dtype=np.float64)),
    ])
    heat_out = eg.chebyshev_heat_spmv(g, heat_in, 2.5, degree=None, lambda_max=2.0)
    jl_emb = eg.chebyshev_dog_jl_embedding(
        g, degree=25, jl_dim=12, seed=42, degree_normalize=True, eps_psd=1e-4
    )

    hop1, hop2 = eg.extract_1hop_2hop_csr(g)
    sparse_pairs = eg.extract_sparse_1hop_2hop_pairs(g, max_2hop_per_node=8, seed=42)
    sparse_pairs_zero = eg.extract_sparse_1hop_2hop_pairs(g, max_2hop_per_node=0, seed=42)
    pure_knn, k2_deg, pure_stats = eg.build_pure_2hop_knn(g)
    sizes_2hop = eg.compute_2hop_neighborhood_sizes(g)
    node_perps = eg.compute_nodewise_2hop_perplexities(g)
    med_perp = float(eg.compute_median_2hop_perplexity(g))
    hybrid_knn, hybrid_stats = eg.build_hybrid_adaptive_knn(g, jl_emb, node_perps)
    bfs_knn = eg.partial_bfs_knn(g, k=6, seed=42)
    probs = eg.compute_knn_probabilities(hybrid_knn, node_perps)
    probs_rn = eg.compute_knn_probabilities(hybrid_knn, node_perps, row_normalize=True)
    fuzzy = eg.compute_fuzzy_simplicial_set(hybrid_knn)
    ab = eg.find_ab_params(0.1, 1.0)
    ht_gamma = float(eg.compute_horvitz_thompson_gamma(1.0, 1600, 16))
    lm = eg.compute_regularized_dog_varadhan_landmarks(
        g, degree=25, num_landmarks=16, seed=42, degree_normalize=True, p_exp=0.5, eps_psd=1e-4, t_coarse=48.0
    )
    alt_u = [0, 63, 64, 69]
    alt_v = [63, 68, 69, 0]
    alt_d = lm.alt_distances(alt_u, alt_v)

    amg_reg_py = eg.AmgPcgSolver.from_graph_regularized_laplacian(g, gamma=10.0)
    b_amg_py = heat_in[:, 0]
    x_amg_py, info_amg_py = amg_reg_py.solve(b_amg_py)
    x_amg_batch_py, info_amg_batch_py = amg_reg_py.solve_batch(heat_in)
    amg_step_py = eg.AmgPcgSolver.from_graph_regularized_laplacian(g, gamma=10.0)
    amg_step_py.start_solve(b_amg_py)
    if not amg_step_py.is_finished:
        amg_step_py.step()
        amg_step_py.run()
    x_amg_step_py = amg_step_py.solution()

    amg_sing_py = eg.AmgPcgSolver.from_graph_laplacian(g, kind="combinatorial", shift=0.0)
    b_sing_py = np.zeros(n, dtype=np.float64)
    b_sing_py[0] = 1.0
    b_sing_py[64] = -1.0
    x_sing_py, info_sing_py = amg_sing_py.solve(b_sing_py)

    aniso_py, aniso_ns_py = eg.anisotropic_laplacian(
        g, alpha=0.5, beta=0.0, degree_reg=0.0, degree_cap=None, reweighted_degree_norm=True
    )
    aniso_trace_py = float(aniso_py.trace())
    aniso_gersh_py = float(aniso_py.gershgorin_spectral_bound())
    aniso_pi_py = float(aniso_py.estimate_lambda_max(num_iters=10, safety_factor=1.02, seed=42))
    aniso_lanc_py = float(aniso_py.estimate_lambda_max_lanczos(num_steps=16, safety_factor=1.05, seed=42))
    aniso_heat_py = eg.chebyshev_heat_spmv_matrix(aniso_py, heat_in, 0.5)
    aniso_jl_py = eg.chebyshev_dog_jl_embedding_matrix(aniso_py, jl_dim=12, seed=42)
    aniso_lm_py = eg.compute_regularized_dog_varadhan_landmarks_from_matrix(
        g, aniso_py, num_landmarks=16, seed=42, t_coarse=0.0
    )

    er_jl_py = eg.compute_effective_resistances(g, jl_dim=64, seed=42, exact=False)
    er_ex_py = eg.compute_effective_resistances(g, seed=42, exact=True)
    rcf_py = eg.resistance_curvature_flow(g, steps=2, eta=0.2, jl_dim=64, seed=42)
    rcf_len_py = rcf_py.length_graph()
    rcf_cond_py = rcf_py.conductance_graph()

    node_url = node_entry.resolve().as_uri() if node_entry.exists() else index_entry.resolve().as_uri()
    wasm_path = (wasm_pkg / "egraph_wasm_bg.wasm").resolve().as_posix()
    spec_payload = {
        "n": n,
        "edges": [int(v) for edge in edges for v in edge],
        "heat_in": [float(v) for v in heat_in.ravel()],
        "b_sing": [float(v) for v in b_sing_py],
        "alt_u": alt_u,
        "alt_v": alt_v,
    }

    node_script = """
import { readFile } from 'node:fs/promises';

const nodeUrl = process.argv[1];
const wasmPath = process.argv[2];
const spec = JSON.parse(process.argv[3]);

const mod = await import(nodeUrl);
if (typeof mod.default === 'function') {
  try {
    const probe = mod.CsrGraph.fromCsr(new Uint32Array([0, 1, 2]), new Uint32Array([1, 0]));
    probe.free?.();
  } catch (_e) {
    await mod.default(await readFile(wasmPath));
  }
}

if (typeof mod.selectMaxminSpPivots !== 'function') {
  console.log(JSON.stringify({ m2_available: false }));
  process.exit(0);
}

const g = mod.CsrGraph.fromEdgeList(spec.n, new Uint32Array(spec.edges));
const pivots = Array.from(mod.selectMaxminSpPivots(g, 12, 42, 2));
const lPmds = mod.pivotMdsInit(g, 12, 42, 2, 1.0);
const pmdsCoords = Array.from(lPmds.coordsView());
const lPmdsUnnorm = mod.pivotMdsInit(g, 12, 42, 2, null);
const pmdsUnnormCoords = Array.from(lPmdsUnnorm.coordsView());
const lKoren = mod.korenSpectralInit(g, 1.0, 1e-4, 300, 42);
const korenCoords = Array.from(lKoren.coordsView());

const eig = mod.smallestLaplacianEigenpairs(g, 4, 1e-4, 300, 42);
const mstats = mod.spectralMomentStats(g);
const heatOut = Array.from(mod.chebyshevHeatSpmv(g, new Float64Array(spec.heat_in), 2, 2.5));
const jlEmb = mod.chebyshevDogJlEmbedding(g, 12, 25, 42, true, 1e-4);

const twoHop = mod.extract1hop2hopCsr(g);
const h1 = twoHop.hop1();
const h2 = twoHop.hop2();
const sparsePairs = mod.extractSparse1hop2hopPairs(g, 8, 42);
const sparsePairsZero = mod.extractSparse1hop2hopPairs(g, 0, 42);
const pureRes = mod.buildPure2hopKnn(g);
const pureKnn = pureRes.knn();
const pureRt = mod.KnnGraph.fromCsr(pureKnn.numNodes, pureKnn.indptr(), pureKnn.indices(), pureKnn.distances());
const sizes2hop = Array.from(mod.compute2hopNeighborhoodSizes(g));
const nodePerps = mod.computeNodewise2hopPerplexities(g);
const medPerp = mod.computeMedian2hopPerplexity(g);
const hybRes = mod.buildHybridAdaptiveKnn(g, jlEmb, 12, nodePerps);
const hybKnn = hybRes.knn();
const bfsKnn = mod.partialBfsKnn(g, 6, 42);
const probs = mod.computeKnnProbabilities(hybKnn, nodePerps);
const probsRn = mod.computeKnnProbabilities(hybKnn, nodePerps, null, null, null, true);
const fuzzy = mod.computeFuzzySimplicialSet(hybKnn);
const ab = Array.from(mod.findAbParams(0.1, 1.0));
const htGamma = mod.computeHorvitzThompsonGamma(1.0, 1600, 16);
const lm = mod.computeRegularizedDogVaradhanLandmarks(g, 25, 16, 42, true, 0.5, 1e-4, 48.0);
const altD = Array.from(lm.altDistances(new Uint32Array(spec.alt_u), new Uint32Array(spec.alt_v)));

const amgReg = mod.AmgPcgSolver.fromGraphRegularizedLaplacian(g, 10.0);
const bAmg = Float64Array.from({ length: spec.n }, (_, i) => spec.heat_in[2 * i]);
const amgRes = amgReg.solve(bAmg);
const amgBatchRes = amgReg.solveBatch(new Float64Array(spec.heat_in), 2);
const amgStep = mod.AmgPcgSolver.fromGraphRegularizedLaplacian(g, 10.0);
amgStep.startSolve(bAmg);
if (!amgStep.isFinished) {
  amgStep.step();
  amgStep.run();
}
const amgSing = mod.AmgPcgSolver.fromGraphLaplacian(g, "combinatorial", 0.0);
const amgSingRes = amgSing.solve(new Float64Array(spec.b_sing));

const aniso = mod.anisotropicLaplacian(g, 0.5, 0.0, 0.0, null, true);
const anisoMat = aniso.matrix();
const anisoNs = Array.from(aniso.nullspace());
const anisoTrace = anisoMat.trace();
const anisoGersh = anisoMat.gershgorinSpectralBound();
const anisoPi = anisoMat.estimateLambdaMax(10, 1.02, 42);
const anisoLanc = anisoMat.estimateLambdaMaxLanczos(16, 1.05, 42);
const anisoHeat = Array.from(mod.chebyshevHeatSpmvMatrix(anisoMat, new Float64Array(spec.heat_in), 2, 0.5));
const anisoJl = Array.from(mod.chebyshevDogJlEmbeddingMatrix(anisoMat, 12, null, 42));
const anisoLm = mod.computeRegularizedDogVaradhanLandmarksFromMatrix(
  g, anisoMat, null, null, 16, 2, 64, 42, true, 0.5, 1e-4, 0.0
);

const erJl = mod.computeEffectiveResistances(g, { jlDim: 64, seed: 42, exact: false });
const erEx = mod.computeEffectiveResistances(g, { seed: 42, exact: true });
const rcf = mod.resistanceCurvatureFlow(g, { steps: 2, eta: 0.2, jlDim: 64, seed: 42 });
const rcfLenG = rcf.lengthGraph();
const rcfCondG = rcf.conductanceGraph();

const out = {
  m2_available: true,
  pivots,
  pmdsCoords,
  pmdsUnnormCoords,
  korenCoords,
  eig: {
    usedDenseFallback: eig.usedDenseFallback,
    lambdas: Array.from(eig.lambdas()),
    vectors: Array.from(eig.vectors()),
    invSqrtDeg: Array.from(eig.invSqrtDeg()),
  },
  mstats: {
    sigma_lambda_sq: mstats.sigmaLambdaSq,
    sigma_lambda: mstats.sigmaLambda,
    deg_cv: mstats.degCv,
    mean_degree: mstats.meanDegree,
    leaf_fraction: mstats.leafFraction,
  },
  heatOut,
  jlEmb: Array.from(jlEmb),
  hop1: { indptr: Array.from(h1.indptr()), indices: Array.from(h1.indices()) },
  hop2: { indptr: Array.from(h2.indptr()), indices: Array.from(h2.indices()) },
  pure: {
    indptr: Array.from(pureRt.indptr()),
    indices: Array.from(pureRt.indices()),
    distances: Array.from(pureRt.distances()),
    k2Deg: Array.from(pureRes.k2Deg()),
    avgDeg1: pureRes.avgDeg1,
    avgDeg2: pureRes.avgDeg2,
    nnz: pureRes.nnz,
  },
  sizes2hop,
  nodePerps: Array.from(nodePerps),
  medPerp,
  hybrid: {
    indptr: Array.from(hybKnn.indptr()),
    indices: Array.from(hybKnn.indices()),
    distances: Array.from(hybKnn.distances()),
    avgDeg1: hybRes.avgDeg1,
    avgDeg2: hybRes.avgDeg2,
    avgRowK: hybRes.avgRowK,
    nnz: hybRes.nnz,
    nFilledResidual: hybRes.nFilledResidual,
    nPure2hopRows: hybRes.nPure2hopRows,
  },
  bfsKnn: {
    indptr: Array.from(bfsKnn.indptr()),
    indices: Array.from(bfsKnn.indices()),
    distances: Array.from(bfsKnn.distances()),
  },
  probs: {
    indptr: Array.from(probs.indptr()),
    indices: Array.from(probs.indices()),
    values: Array.from(probs.values()),
  },
  probsRn: {
    indptr: Array.from(probsRn.indptr()),
    indices: Array.from(probsRn.indices()),
    values: Array.from(probsRn.values()),
  },
  sparsePairs: {
    u1: Array.from(sparsePairs.u1),
    v1: Array.from(sparsePairs.v1),
    u2: Array.from(sparsePairs.u2),
    v2: Array.from(sparsePairs.v2),
  },
  sparsePairsZero: {
    u1: Array.from(sparsePairsZero.u1),
    v1: Array.from(sparsePairsZero.v1),
    u2: Array.from(sparsePairsZero.u2),
    v2: Array.from(sparsePairsZero.v2),
  },
  fuzzy: {
    indptr: Array.from(fuzzy.indptr()),
    indices: Array.from(fuzzy.indices()),
    values: Array.from(fuzzy.values()),
  },
  ab,
  htGamma,
  lm: {
    landmarks: Array.from(lm.landmarks()),
    table: Array.from(lm.table()),
    coarseTable: Array.from(lm.coarseTable()),
    t1: lm.t1,
    t2: lm.t2,
    alpha: lm.alpha,
    deltaPsd: lm.deltaPsd,
    betaEff: lm.betaEff,
    betaAuto: lm.betaAuto,
    usedCoarseScaffold: lm.usedCoarseScaffold,
    altD,
  },
  amg: {
    numLevels: amgReg.numLevels,
    coarsestSize: amgReg.coarsestSize,
    iterations: amgRes.iterations,
    converged: amgRes.converged,
    relRes: amgRes.relativeResidual,
    x: Array.from(amgRes.x()),
    xStep: Array.from(amgStep.solution()),
    batchAllConverged: amgBatchRes.allConverged,
    batchConvergedPerCol: Array.from(amgBatchRes.convergedPerCol()),
    batchX: Array.from(amgBatchRes.x()),
    singIterations: amgSingRes.iterations,
    singConverged: amgSingRes.converged,
    singX: Array.from(amgSingRes.x()),
  },
  aniso: {
    values: Array.from(anisoMat.values()),
    nullspace: anisoNs,
    trace: anisoTrace,
    gersh: anisoGersh,
    pi: anisoPi,
    lanc: anisoLanc,
    heat: anisoHeat,
    jl: anisoJl,
    lmTable: Array.from(anisoLm.table()),
    lmDegree: anisoLm.degree,
  },
  rcf: {
    erJlRes: Array.from(erJl.effectiveResistances()),
    erJlFosterErr: erJl.fosterRelErr,
    erJlAllConv: erJl.amgAllConverged,
    erExRes: Array.from(erEx.effectiveResistances()),
    erExFosterErr: erEx.fosterRelErr,
    numComp: rcf.numComponents,
    edgeU: Array.from(rcf.edgeU()),
    edgeV: Array.from(rcf.edgeV()),
    edgeLengths: Array.from(rcf.edgeLengths()),
    conductances: Array.from(rcf.conductances()),
    effectiveResistances: Array.from(rcf.effectiveResistances()),
    vertexCurvatures: Array.from(rcf.vertexCurvatures()),
    linkCurvatures: Array.from(rcf.linkCurvatures()),
    fosterSum: rcf.fosterSum,
    expectedFoster: rcf.expectedFoster,
    meanLink: rcf.meanLinkCurvature,
    suggestedGeom: rcf.suggestedGeometry,
    suggestedManifold: rcf.suggestedManifold ?? null,
    stepsLen: rcf.stepHistory().length,
    lenWeights: Array.from(rcfLenG.weights()),
    condWeights: Array.from(rcfCondG.weights()),
  },
};

rcfCondG.free?.();
rcfLenG.free?.();
rcf.free?.();
erEx.free?.();
erJl.free?.();
anisoLm.free?.();
anisoMat.free?.();
aniso.free?.();
amgSingRes.free?.();
amgSing.free?.();
amgStep.free?.();
amgBatchRes.free?.();
amgRes.free?.();
amgReg.free?.();
lm.free?.();
fuzzy.free?.();
probsRn.free?.();
probs.free?.();
sparsePairsZero.free?.();
sparsePairs.free?.();
bfsKnn.free?.();
hybKnn.free?.();
hybRes.free?.();
pureRt.free?.();
pureKnn.free?.();
pureRes.free?.();
h1.free?.();
h2.free?.();
twoHop.free?.();
mstats.free?.();
eig.free?.();
lKoren.free?.();
lPmdsUnnorm.free?.();
lPmds.free?.();
g.free?.();

console.log(JSON.stringify(out));
"""
    raw_out = subprocess.check_output(
        ["node", "--input-type=module", "-e", node_script, node_url, wasm_path, json.dumps(spec_payload)],
        text=True,
    )
    ws = json.loads(raw_out.strip().splitlines()[-1])
    if not ws.get("m2_available", False):
        return [{
            "case": "cross_target_m2_py_vs_wasm",
            "status": "skipped_no_wasm_m2",
            "elapsed_sec": round(time.perf_counter() - t0, 4),
        }]

    max_f64_diff = 0.0
    max_f32_diff = 0.0

    def check_f64(label: str, a: Any, b: Any, tol: float = 1e-12) -> None:
        nonlocal max_f64_diff
        arr_a = np.asarray(a, dtype=np.float64).ravel()
        arr_b = np.asarray(b, dtype=np.float64).ravel()
        assert arr_a.shape == arr_b.shape, f"{label}: shape mismatch {arr_a.shape} vs {arr_b.shape}"
        diff = float(np.max(np.abs(arr_a - arr_b))) if arr_a.size else 0.0
        max_f64_diff = max(max_f64_diff, diff)
        assert diff < tol, f"{label}: diff={diff} >= {tol}"

    def check_f32(label: str, a: Any, b: Any, tol: float = 1e-5) -> None:
        nonlocal max_f32_diff
        arr_a = np.asarray(a, dtype=np.float64).ravel()
        arr_b = np.asarray(b, dtype=np.float64).ravel()
        assert arr_a.shape == arr_b.shape, f"{label}: shape mismatch {arr_a.shape} vs {arr_b.shape}"
        diff = float(np.max(np.abs(arr_a - arr_b))) if arr_a.size else 0.0
        max_f32_diff = max(max_f32_diff, diff)
        assert diff < tol, f"{label}: diff={diff} >= {tol}"

    assert list(pivots) == ws["pivots"], "select_maxmin_sp_pivots mismatch"
    check_f32("pivot_mds_init", pmds_f32, ws["pmdsCoords"])
    check_f32("pivot_mds_init_unnorm", pmds_unnorm, ws["pmdsUnnormCoords"])
    check_f32("koren_spectral_init", koren_f32, ws["korenCoords"])

    assert bool(used_dense) == bool(ws["eig"]["usedDenseFallback"])
    check_f64("eigen/lambdas", lams, ws["eig"]["lambdas"])
    check_f64("eigen/vectors", evecs, ws["eig"]["vectors"])
    check_f64("eigen/inv_sqrt_deg", inv_d, ws["eig"]["invSqrtDeg"])

    for k, v in mstats.items():
        check_f64(f"spectral_moment_stats/{k}", [v], [ws["mstats"][k]])

    check_f64("chebyshev_heat_spmv", heat_out, ws["heatOut"])
    check_f64("chebyshev_dog_jl_embedding", jl_emb, ws["jlEmb"])

    assert list(hop1.indptr()) == ws["hop1"]["indptr"] and list(hop1.indices()) == ws["hop1"]["indices"]
    assert list(hop2.indptr()) == ws["hop2"]["indptr"] and list(hop2.indices()) == ws["hop2"]["indices"]
    assert list(pure_knn.indptr()) == ws["pure"]["indptr"] and list(pure_knn.indices()) == ws["pure"]["indices"]
    check_f64("build_pure_2hop_knn/distances", pure_knn.distances(), ws["pure"]["distances"])
    assert list(k2_deg) == ws["pure"]["k2Deg"] == ws["sizes2hop"] == list(sizes_2hop)
    assert pure_stats["nnz"] == ws["pure"]["nnz"]
    check_f64("build_pure_2hop_knn/avg_deg1", [pure_stats["avg_deg1"]], [ws["pure"]["avgDeg1"]])
    check_f64("build_pure_2hop_knn/avg_deg2", [pure_stats["avg_deg2"]], [ws["pure"]["avgDeg2"]])

    check_f64("compute_nodewise_2hop_perplexities", node_perps, ws["nodePerps"])
    check_f64("compute_median_2hop_perplexity", [med_perp], [ws["medPerp"]])

    hyb_ptr = list(hybrid_knn.indptr())
    assert hyb_ptr == ws["hybrid"]["indptr"]
    hyb_py_idx = list(hybrid_knn.indices())
    hyb_py_dst = list(hybrid_knn.distances())
    hyb_ws_idx = ws["hybrid"]["indices"]
    hyb_ws_dst = ws["hybrid"]["distances"]
    for row_i in range(n):
        s, e = hyb_ptr[row_i], hyb_ptr[row_i + 1]
        py_pairs = sorted(zip(hyb_py_idx[s:e], hyb_py_dst[s:e]))
        ws_pairs = sorted(zip(hyb_ws_idx[s:e], hyb_ws_dst[s:e]))
        assert [c for c, _ in py_pairs] == [c for c, _ in ws_pairs], (
            f"build_hybrid_adaptive_knn/row_{row_i}_indices mismatch"
        )
        check_f64(
            f"build_hybrid_adaptive_knn/row_{row_i}_distances",
            [d for _, d in py_pairs],
            [d for _, d in ws_pairs],
        )
    assert hybrid_stats["nnz"] == ws["hybrid"]["nnz"]
    assert hybrid_stats["n_filled_residual"] == ws["hybrid"]["nFilledResidual"]
    assert hybrid_stats["n_pure_2hop_rows"] == ws["hybrid"]["nPure2hopRows"]
    check_f64("build_hybrid_adaptive_knn/avg_row_k", [hybrid_stats["avg_row_k"]], [ws["hybrid"]["avgRowK"]])

    assert list(bfs_knn.indptr()) == ws["bfsKnn"]["indptr"] and list(bfs_knn.indices()) == ws["bfsKnn"]["indices"]
    check_f64("partial_bfs_knn/distances", bfs_knn.distances(), ws["bfsKnn"]["distances"])

    assert list(probs.indptr()) == ws["probs"]["indptr"] and list(probs.indices()) == ws["probs"]["indices"]
    check_f64("compute_knn_probabilities/values", probs.values(), ws["probs"]["values"])
    assert list(probs_rn.indptr()) == ws["probsRn"]["indptr"] and list(probs_rn.indices()) == ws["probsRn"]["indices"]
    check_f64("compute_knn_probabilities_rn/values", probs_rn.values(), ws["probsRn"]["values"])

    assert list(sparse_pairs.u1()) == ws["sparsePairs"]["u1"]
    assert list(sparse_pairs.v1()) == ws["sparsePairs"]["v1"]
    assert list(sparse_pairs.u2()) == ws["sparsePairs"]["u2"]
    assert list(sparse_pairs.v2()) == ws["sparsePairs"]["v2"]
    assert list(sparse_pairs_zero.u1()) == ws["sparsePairsZero"]["u1"]
    assert list(sparse_pairs_zero.v1()) == ws["sparsePairsZero"]["v1"]
    assert list(sparse_pairs_zero.u2()) == ws["sparsePairsZero"]["u2"] == []
    assert list(sparse_pairs_zero.v2()) == ws["sparsePairsZero"]["v2"] == []

    assert list(fuzzy.indptr()) == ws["fuzzy"]["indptr"] and list(fuzzy.indices()) == ws["fuzzy"]["indices"]
    check_f64("compute_fuzzy_simplicial_set/values", fuzzy.values(), ws["fuzzy"]["values"])

    check_f64("find_ab_params", ab, ws["ab"])
    check_f64("compute_horvitz_thompson_gamma", [ht_gamma], [ws["htGamma"]])

    assert list(lm.landmarks()) == ws["lm"]["landmarks"]
    assert bool(lm.used_coarse_scaffold) == bool(ws["lm"]["usedCoarseScaffold"])
    check_f64("dog_varadhan/table", lm.table(), ws["lm"]["table"])
    check_f64("dog_varadhan/coarse_table", lm.coarse_table(), ws["lm"]["coarseTable"])
    check_f64("dog_varadhan/alt_distances", alt_d, ws["lm"]["altD"])
    for py_attr, ws_key in (
        ("t1", "t1"),
        ("t2", "t2"),
        ("alpha", "alpha"),
        ("delta_psd", "deltaPsd"),
        ("beta_eff", "betaEff"),
        ("beta_auto", "betaAuto"),
    ):
        check_f64(f"dog_varadhan/{py_attr}", [getattr(lm, py_attr)], [ws["lm"][ws_key]])

    assert amg_reg_py.num_levels == ws["amg"]["numLevels"]
    assert amg_reg_py.coarsest_size == ws["amg"]["coarsestSize"]
    assert info_amg_py["iterations"] == ws["amg"]["iterations"]
    assert bool(info_amg_py["converged"]) == bool(ws["amg"]["converged"])
    check_f64("amg_pcg/x", x_amg_py, ws["amg"]["x"], tol=1e-10)
    check_f64("amg_pcg/x_step", x_amg_step_py, ws["amg"]["xStep"], tol=1e-10)
    assert bool(info_amg_batch_py["all_converged"]) == bool(ws["amg"]["batchAllConverged"])
    assert [int(v) for v in info_amg_batch_py["converged_per_col"]] == ws["amg"]["batchConvergedPerCol"]
    check_f64("amg_pcg/batch_x", x_amg_batch_py.ravel(), ws["amg"]["batchX"], tol=1e-10)
    assert info_sing_py["iterations"] == ws["amg"]["singIterations"]
    assert bool(info_sing_py["converged"]) == bool(ws["amg"]["singConverged"])
    check_f64("amg_pcg/sing_x", x_sing_py, ws["amg"]["singX"], tol=1e-10)

    check_f64("aniso/values", aniso_py.values(), ws["aniso"]["values"])
    check_f64("aniso/nullspace", aniso_ns_py, ws["aniso"]["nullspace"])
    check_f64("aniso/trace", [aniso_trace_py], [ws["aniso"]["trace"]])
    check_f64("aniso/gersh", [aniso_gersh_py], [ws["aniso"]["gersh"]])
    check_f64("aniso/pi", [aniso_pi_py], [ws["aniso"]["pi"]])
    check_f64("aniso/lanc", [aniso_lanc_py], [ws["aniso"]["lanc"]])
    check_f64("aniso/heat", aniso_heat_py, ws["aniso"]["heat"])
    check_f64("aniso/jl", aniso_jl_py, ws["aniso"]["jl"])
    check_f64("aniso/lm_table", aniso_lm_py.table(), ws["aniso"]["lmTable"])
    assert aniso_lm_py.degree == ws["aniso"]["lmDegree"]

    assert er_jl_py.amg_all_converged == ws["rcf"]["erJlAllConv"]
    check_f64("er_jl/resistances", er_jl_py.effective_resistances(), ws["rcf"]["erJlRes"], tol=1e-10)
    check_f64("er_jl/foster_rel_err", [er_jl_py.foster_rel_err], [ws["rcf"]["erJlFosterErr"]], tol=1e-10)
    check_f64("er_ex/resistances", er_ex_py.effective_resistances(), ws["rcf"]["erExRes"], tol=1e-10)
    check_f64("er_ex/foster_rel_err", [er_ex_py.foster_rel_err], [ws["rcf"]["erExFosterErr"]], tol=1e-10)
    assert rcf_py.num_components == ws["rcf"]["numComp"]
    assert list(rcf_py.edge_u()) == ws["rcf"]["edgeU"]
    assert list(rcf_py.edge_v()) == ws["rcf"]["edgeV"]
    check_f64("rcf/edge_lengths", rcf_py.edge_lengths(), ws["rcf"]["edgeLengths"], tol=1e-10)
    check_f64("rcf/conductances", rcf_py.conductances(), ws["rcf"]["conductances"], tol=1e-10)
    check_f64("rcf/effective_resistances", rcf_py.effective_resistances(), ws["rcf"]["effectiveResistances"], tol=1e-10)
    check_f64("rcf/vertex_curvatures", rcf_py.vertex_curvatures(), ws["rcf"]["vertexCurvatures"], tol=1e-10)
    check_f64("rcf/link_curvatures", rcf_py.link_curvatures(), ws["rcf"]["linkCurvatures"], tol=1e-10)
    check_f64("rcf/foster_sum", [rcf_py.foster_sum], [ws["rcf"]["fosterSum"]], tol=1e-10)
    check_f64("rcf/expected_foster", [rcf_py.expected_foster], [ws["rcf"]["expectedFoster"]], tol=1e-10)
    check_f64("rcf/mean_link_curvature", [rcf_py.mean_link_curvature], [ws["rcf"]["meanLink"]], tol=1e-10)
    assert rcf_py.suggested_geometry == ws["rcf"]["suggestedGeom"]
    assert rcf_py.suggested_manifold == ws["rcf"]["suggestedManifold"]
    assert len(rcf_py.step_history()) == ws["rcf"]["stepsLen"]
    check_f32("rcf/length_graph_weights", rcf_len_py.weights(), ws["rcf"]["lenWeights"], tol=1e-5)
    check_f32("rcf/conductance_graph_weights", rcf_cond_py.weights(), ws["rcf"]["condWeights"], tol=1e-5)

    return [{
        "case": "cross_target_m2_py_vs_wasm",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "max_f64_diff": round(max_f64_diff, 15),
        "max_f32_diff": round(max_f32_diff, 12),
    }]


def run_cross_target_m3_parity(eg: Any, wasm_pkg: pathlib.Path) -> list[dict[str, Any]]:
    """Verify Python float32/float64 <-> WASM f32 parity (< 1e-4) across M3 APIs."""
    import numpy as np

    if not hasattr(eg, "TsNetSolver"):
        return []
    node_entry = wasm_pkg / "egraph_wasm.node.js"
    index_entry = wasm_pkg / "index.js"
    if not index_entry.exists():
        return []

    t0 = time.perf_counter()

    # 1. Clustering & coarsening fixtures (12-node barbell)
    pts_km = np.array(
        [
            [10.0, 10.0],
            [10.2, 9.8],
            [9.8, 10.1],
            [-10.0, -10.0],
            [-9.9, -10.2],
            [-10.1, -9.8],
        ],
        dtype=np.float64,
    )
    km_py = eg.kmeans_plus_plus(pts_km, k=2, max_iters=100, tol=1e-4, n_init=3, seed=42)

    bb_edges = [
        (i, j) for off in (0, 6) for i in range(off, off + 6) for j in range(i + 1, off + 6)
    ] + [(5, 6)]
    g_bb = eg.CsrGraph.from_edges(12, bb_edges, directed=False, sanitize=True)
    spec_py = eg.spectral_clustering(g_bb, k=2, max_eigen_iters=300, eigen_tol=1e-4, kmeans_max_iters=100, kmeans_tol=1e-4, seed=42)
    louv_py = eg.louvain_communities(g_bb, max_passes=100, resolution=1.0, min_modularity_gain=1e-7, seed=42)
    lpa_py = eg.label_propagation(g_bb, max_iters=100, seed=42)
    cg_py, n2c_py, sizes_py, nclust_py = eg.coarsen_graph(g_bb, [100] * 6 + [999] * 6)
    coarse_coords_f32 = np.array([[-2.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    proj_py32 = eg.project_clustered_layout(
        eg.Layout2d.from_coords(coarse_coords_f32), n2c_py, spread_radius=0.2, seed=42
    ).coords(copy=True)
    proj_py64 = eg.project_clustered_layout(
        eg.Layout2d.from_coords(coarse_coords_f32.astype(np.float64)), n2c_py, spread_radius=0.2, seed=42
    ).coords(copy=True)

    # 2. ManifoldLayout2d & SgdSolver manifold stepping (all 3 manifolds)
    g16_edges = [
        (r * 4 + c, r * 4 + c + 1) for r in range(4) for c in range(3)
    ] + [
        (r * 4 + c, (r + 1) * 4 + c) for r in range(3) for c in range(4)
    ]
    g16 = eg.CsrGraph.from_edges(16, g16_edges, directed=False, sanitize=True)
    init16_f32 = np.array(_make_m3_asymmetric_coords(16, scale=0.8), dtype=np.float32)
    init16_f64 = init16_f32.astype(np.float64)
    gumap_init16_f32 = (init16_f32 * 5.0).astype(np.float32)
    gumap_init16_f64 = gumap_init16_f32.astype(np.float64)
    emb16 = eg.chebyshev_dog_jl_embedding(g16, degree=15, jl_dim=8, seed=42)

    man_init_f32 = (init16_f32 * 0.25).astype(np.float32)
    man_sph_init_f32 = (man_init_f32 + 1.5).astype(np.float32)
    man_py32: dict[str, dict[str, Any]] = {}
    man_py64: dict[str, dict[str, Any]] = {}
    for kind in ("hyperbolic", "spherical", "torus"):
        # Unit [0, 1)^2 torus has geodesic diameter sqrt(0.5) ~ 0.707, and spherical
        # coordinates near colat=0 hit a pole atan2 singularity; use equator-centered init
        # and scaled steps for spherical/torus against unscaled g16 shortest-path targets.
        m_step_scale = 0.02 if kind == "torus" else (0.2 if kind == "spherical" else 1.0)
        m_init = man_sph_init_f32 if kind == "spherical" else man_init_f32
        ml32 = eg.ManifoldLayout2d.from_coords(kind, m_init)
        sgd_m32 = eg.SgdSolver.full(g16, t_max=10, eps=0.01, schedule="exponential", step_scale=m_step_scale, seed=42, dtype="float32")
        sgd_m32.run_manifold(ml32)
        man_py32[kind] = {
            "coords": ml32.coords(),
            "dist": float(ml32.geodesic_distance(0, 15)),
        }
        ml64 = eg.ManifoldLayout2d.from_coords(kind, m_init.astype(np.float64))
        sgd_m64 = eg.SgdSolver.full(g16, t_max=10, eps=0.01, schedule="exponential", step_scale=m_step_scale, seed=42, dtype="float64")
        sgd_m64.run_manifold(ml64)
        man_py64[kind] = {
            "coords": ml64.coords(),
            "dist": float(ml64.geodesic_distance(0, 15)),
        }

    # 3. All 6 Euclidean solvers in Python float32 & float64
    g9_edges = [
        (r * 3 + c, r * 3 + c + 1) for r in range(3) for c in range(2)
    ] + [
        (r * 3 + c, (r + 1) * 3 + c) for r in range(2) for c in range(3)
    ]
    g9 = eg.CsrGraph.from_edges(9, g9_edges, directed=False, sanitize=True)
    folded9_f32 = np.array([[float(c), float(r)] for r in range(3) for c in range(3)], dtype=np.float32)
    folded9_f32[8] = np.array([0.08, 0.08], dtype=np.float32)
    init9_f32 = np.array(_make_m3_asymmetric_coords(9, scale=1.2), dtype=np.float32)

    g11_edges = list(g9_edges) + [(0, 9), (0, 10), (9, 10)]
    g11_twin = eg.CsrGraph.from_edges(11, g11_edges, directed=False, sanitize=True)
    twin11_f32 = np.vstack([
        np.array([[float(c), float(r)] for r in range(3) for c in range(3)], dtype=np.float32),
        np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
    ])

    hinge_s_src = [e[0] for e in g16_edges]
    hinge_s_tgt = [e[1] for e in g16_edges]
    hinge_s_ones = [1.0] * len(g16_edges)
    lm16 = eg.compute_regularized_dog_varadhan_landmarks(g16, degree=15, num_landmarks=8, seed=42)

    def run_py_solvers(dtype_name: str) -> dict[str, Any]:
        c16 = init16_f32 if dtype_name == "float32" else init16_f64
        c9_fold = folded9_f32 if dtype_name == "float32" else folded9_f32.astype(np.float64)
        c9_init = init9_f32 if dtype_name == "float32" else init9_f32.astype(np.float64)
        c11_twin = twin11_f32 if dtype_name == "float32" else twin11_f32.astype(np.float64)

        out: dict[str, Any] = {}
        for bk in ("fft", "linear_fft", "barnes_hut", "exact"):
            l = eg.Layout2d.from_coords(c16)
            sv = eg.TsNetSolver.pure_2hop(
                g16, backend=bk, iterations_stage1=4, iterations_stage2=4, iterations_stage3=4, seed=42, dtype=dtype_name
            )
            sv.run(l)
            out[f"tsnet_{bk}"] = l.coords(copy=True)
            out[f"tsnet_{bk}_ffts"] = int(sv.total_fft_2d_calls)

        for tag, w2_val, cap_val, bk in (
            ("tsnet_rn_uncapped", 0.5, None, "fft"),
            ("tsnet_rn_capped", 0.35, 8, "exact"),
        ):
            l_rn = eg.Layout2d.from_coords(c16)
            sv_rn = eg.TsNetSolver.from_graph_row_normalized(
                g16,
                w2=w2_val,
                max_2hop_per_node=cap_val,
                backend=bk,
                iterations_stage1=4,
                iterations_stage2=4,
                iterations_stage3=4,
                seed=42,
                dtype=dtype_name,
            )
            sv_rn.run(l_rn)
            out[tag] = l_rn.coords(copy=True)
            out[f"{tag}_ffts"] = int(sv_rn.total_fft_2d_calls)

        sgd_map = {
            "sgd_full": eg.SgdSolver.full(g16, t_max=10, eps=0.01, schedule="exponential", step_scale=1.0, seed=42, dtype=dtype_name),
            "sgd_dog": eg.SgdSolver.sparse_barrier_dog(g16, degree=15, num_landmarks=8, t_max=10, eps=0.01, seed=42, dtype=dtype_name),
            "sgd_rp": eg.SgdSolver.random_pair(g16, pairs_per_node=4, t_max=10, eps=0.01, seed=42, dtype=dtype_name),
            "sgd_omega": eg.SgdSolver.omega(g16, emb16, k=4, min_dist=1e-3, t_max=10, eps=0.01, seed=42, dtype=dtype_name),
            "sgd_stream": eg.SgdSolver.streaming_negatives(g16, negatives_per_node=3, target_distance=2.0, weight=0.25, hinge_only=True, t_max=10, eps=0.01, seed=42, dtype=dtype_name),
            "sgd_hinges": eg.SgdSolver.from_pairs_with_hinges(
                16,
                hinge_s_src,
                hinge_s_tgt,
                hinge_s_ones,
                hinge_s_ones,
                hinge_s_ones,
                hinge_s_ones,
                [0, 1],
                [15, 14],
                [3.0, 3.0],
                [3.0, 3.0],
                [1.0, 1.0],
                [1.0, 1.0],
                t_max=10,
                eps=0.01,
                seed=42,
                dtype=dtype_name,
            ),
        }
        for k, sv in sgd_map.items():
            l = eg.Layout2d.from_coords(c16)
            sv.run(l)
            out[k] = l.coords(copy=True)
            out[f"{k}_springs"] = int(sv.num_springs)
            out[f"{k}_hinges"] = int(sv.num_hinges)

        # Use UMAP-scale initialization and moderate lr=0.2 so native/WASM f32 stay
        # away from the d^2 -> 0 repulsive gradient clamp discontinuity across 15 epochs.
        l_gm = eg.Layout2d.from_coords(gumap_init16_f32 if dtype_name == "float32" else gumap_init16_f64)
        gm = eg.GuMapSolver.from_graph_dog(
            g16,
            max_2hop_per_node=16,
            degree=15,
            num_landmarks=8,
            n_epochs=15,
            learning_rate=0.2,
            seed=42,
            dtype=dtype_name,
        )
        gm.run(l_gm)
        out["gumap"] = l_gm.coords(copy=True)
        out["gumap_edges"] = int(gm.num_edges)

        l_gm_lm = eg.Layout2d.from_coords(gumap_init16_f32 if dtype_name == "float32" else gumap_init16_f64)
        gm_lm = eg.GuMapSolver.from_landmarks(
            g16,
            lm16,
            max_2hop_per_node=16,
            n_epochs=15,
            learning_rate=0.2,
            seed=42,
            dtype=dtype_name,
        )
        gm_lm.run(l_gm_lm)
        out["gumap_lm"] = l_gm_lm.coords(copy=True)
        out["gumap_lm_edges"] = int(gm_lm.num_edges)

        l_gm_asym = eg.Layout2d.from_coords(gumap_init16_f32 if dtype_name == "float32" else gumap_init16_f64)
        gm_asym = eg.GuMapSolver.from_graph_dog(
            g16,
            max_2hop_per_node=16,
            degree=15,
            num_landmarks=8,
            n_epochs=15,
            learning_rate=0.2,
            symmetric_negative=False,
            seed=42,
            dtype=dtype_name,
        )
        gm_asym.run(l_gm_asym)
        out["gumap_asym"] = l_gm_asym.coords(copy=True)

        l_rf = eg.Layout2d.from_coords(c9_fold)
        rf = eg.SpatialIntruderRefiner(g9, rounds=3, k_cap=32, t_max=10, seed=42, dtype=dtype_name)
        rf.run(l_rf)
        out["refiner"] = l_rf.coords(copy=True)
        out["refiner_fp"] = int(rf.stats()["n_fp_total"])

        l_rf_dyn = eg.Layout2d.from_coords(c9_fold)
        rf_dyn = eg.SpatialIntruderRefiner(
            g9,
            rounds=4,
            k_cap=8,
            t_max=10,
            seed=42,
            w_anchor2=0.5,
            k_cap_decay=0.5,
            k_cap_min=2,
            dtype=dtype_name,
        )
        rf_dyn.step(l_rf_dyn)
        assert rf_dyn.current_k_cap == 4
        rf_dyn.set_k_cap(6)
        assert rf_dyn.current_k_cap == 6
        rf_dyn.run(l_rf_dyn)
        out["refiner_dyn"] = l_rf_dyn.coords(copy=True)
        out["refiner_dyn_fp"] = int(rf_dyn.stats()["n_fp_total"])

        l_rf_sep = eg.Layout2d.from_coords(c11_twin)
        rf_sep = eg.SpatialIntruderRefiner(
            g11_twin,
            rounds=3,
            k_cap=16,
            t_max=10,
            min_sep_ratio=0.15,
            w_sep=1.0,
            seed=42,
            dtype=dtype_name,
        )
        rf_sep.run(l_rf_sep)
        out["refiner_sep"] = l_rf_sep.coords(copy=True)
        out["refiner_sep_total"] = int(rf_sep.stats()["n_sep_total"])

        l_sm = eg.Layout2d.from_coords(c16)
        sm = eg.StressMajorizationSolver.from_graph(g16, max_iterations=15, epsilon=1e-6, dtype=dtype_name)
        sm.run(l_sm)
        out["sm"] = l_sm.coords(copy=True)
        out["sm_stress"] = float(sm.last_stress or 0.0)

        l_kk = eg.Layout2d.from_coords(c9_init)
        kk = eg.KamadaKawaiSolver.from_graph(g9, eps=1e-3, max_inner_iters=40, max_outer_iters=20, dtype=dtype_name)
        kk.run(l_kk)
        out["kk"] = l_kk.coords(copy=True)
        return out

    py32 = run_py_solvers("float32")
    py64 = run_py_solvers("float64")

    node_url = node_entry.resolve().as_uri() if node_entry.exists() else index_entry.resolve().as_uri()
    wasm_path = (wasm_pkg / "egraph_wasm_bg.wasm").resolve().as_posix()
    spec_payload = {
        "pts_km": [float(v) for v in pts_km.ravel()],
        "bb_edges": [int(v) for edge in bb_edges for v in edge],
        "g16_edges": [int(v) for edge in g16_edges for v in edge],
        "g9_edges": [int(v) for edge in g9_edges for v in edge],
        "g11_edges": [int(v) for edge in g11_edges for v in edge],
        "coarse_coords": [float(v) for v in coarse_coords_f32.ravel()],
        "init16": [float(v) for v in init16_f32.ravel()],
        "gumap_init16": [float(v) for v in gumap_init16_f32.ravel()],
        "man_init16": [float(v) for v in man_init_f32.ravel()],
        "man_sph_init16": [float(v) for v in man_sph_init_f32.ravel()],
        "emb16": [float(v) for v in emb16.ravel()],
        "folded9": [float(v) for v in folded9_f32.ravel()],
        "twin11": [float(v) for v in twin11_f32.ravel()],
        "init9": [float(v) for v in init9_f32.ravel()],
        "hinge_src": hinge_s_src,
        "hinge_tgt": hinge_s_tgt,
        "hinge_ones": hinge_s_ones,
    }

    node_script = """
import { readFile } from 'node:fs/promises';

const nodeUrl = process.argv[1];
const wasmPath = process.argv[2];
const spec = JSON.parse(process.argv[3]);

const mod = await import(nodeUrl);
if (typeof mod.default === 'function') {
  try {
    const probe = mod.CsrGraph.fromCsr(new Uint32Array([0, 1, 2]), new Uint32Array([1, 0]));
    probe.free?.();
  } catch (_e) {
    await mod.default(await readFile(wasmPath));
  }
}

if (typeof mod.TsNetSolver !== 'function') {
  console.log(JSON.stringify({ m3_available: false }));
  process.exit(0);
}

const km = mod.kmeansPlusPlus(new Float64Array(spec.pts_km), 6, 2, 2, 100, 1e-4, 3, 42);
const bb = mod.CsrGraph.fromEdgeList(12, new Uint32Array(spec.bb_edges));
const specLabels = Array.from(mod.spectralClustering(bb, 2, 300, 1e-4, 100, 1e-4, 42));
const louv = mod.louvainCommunities(bb, 100, 1.0, 1e-7, 42);
const lpaLabels = Array.from(mod.labelPropagation(bb, 100, 42));
const coarse = mod.coarsenGraph(bb, [100, 100, 100, 100, 100, 100, 999, 999, 999, 999, 999, 999]);
const cg = coarse.graph();
const cl = mod.Layout2d.fromCoords(new Float32Array(spec.coarse_coords));
const proj = mod.projectClusteredLayout(cl, coarse.nodeToCoarse(), 0.2, 42);
const projCoords = Array.from(proj.coordsView());

const g16 = mod.CsrGraph.fromEdgeList(16, new Uint32Array(spec.g16_edges));
const g9 = mod.CsrGraph.fromEdgeList(9, new Uint32Array(spec.g9_edges));
const g11 = mod.CsrGraph.fromEdgeList(11, new Uint32Array(spec.g11_edges));
const init16 = new Float32Array(spec.init16);
const gumapInit16 = new Float32Array(spec.gumap_init16);
const manInit = new Float32Array(spec.man_init16);
const manSphInit = new Float32Array(spec.man_sph_init16);
const emb16 = new Float64Array(spec.emb16);
const folded9 = new Float32Array(spec.folded9);
const twin11 = new Float32Array(spec.twin11);
const init9 = new Float32Array(spec.init9);

const manifolds = {};
for (const kind of ['hyperbolic', 'spherical', 'torus']) {
  const mStepScale = kind === 'torus' ? 0.02 : (kind === 'spherical' ? 0.2 : 1.0);
  const mInit = kind === 'spherical' ? manSphInit : manInit;
  const mlRun = mod.ManifoldLayout2d.fromCoords(kind, mInit);
  const mlStep = mod.ManifoldLayout2d.fromCoords(kind, mInit);
  const sRun = mod.SgdSolver.full(g16, { tMax: 10, eps: 0.01, kind: 'exponential', stepScale: mStepScale, seed: 42 });
  const sStep = mod.SgdSolver.full(g16, { tMax: 10, eps: 0.01, kind: 'exponential', stepScale: mStepScale, seed: 42 });
  sRun.runManifold(mlRun);
  sStep.runStepsManifold(mlStep, 4);
  while (!sStep.isFinished) sStep.stepManifold(mlStep);
  const cRun = Array.from(mlRun.coords());
  const cStep = Array.from(mlStep.coords());
  if (JSON.stringify(cRun) !== JSON.stringify(cStep)) {
    throw new Error(`WASM SgdSolver.stepManifold != runManifold for ${kind}`);
  }
  manifolds[kind] = { coords: cRun, dist: mlRun.geodesicDistance(0, 15) };
  mlRun.free?.();
  mlStep.free?.();
  sRun.free?.();
  sStep.free?.();
}

const runPairAndCheckStep = (mkSolver, initCoords, stepChunk) => {
  const sR = mkSolver();
  const sS = mkSolver();
  const lR = mod.Layout2d.fromCoords(initCoords);
  const lS = mod.Layout2d.fromCoords(initCoords);
  sR.run(lR);
  sS.runSteps(lS, stepChunk);
  while (!sS.isFinished) sS.step(lS);
  const cR = Array.from(lR.coordsView());
  const cS = Array.from(lS.coordsView());
  if (JSON.stringify(cR) !== JSON.stringify(cS)) {
    throw new Error('WASM step() != run()');
  }
  lR.free?.();
  lS.free?.();
  return { coords: cR, solverRun: sR, solverStep: sS };
};

const solvers = {};
for (const bk of ['fft', 'linear_fft', 'barnes_hut', 'exact']) {
  const res = runPairAndCheckStep(
    () => mod.TsNetSolver.pure2Hop(g16, { backend: bk, iterationsStage1: 4, iterationsStage2: 4, iterationsStage3: 4, seed: 42 }),
    init16,
    5,
  );
  solvers[`tsnet_${bk}`] = res.coords;
  solvers[`tsnet_${bk}_ffts`] = res.solverRun.totalFft2dCalls;
  res.solverRun.free?.();
  res.solverStep.free?.();
}

for (const [tag, w2Val, capVal, bk] of [
  ['tsnet_rn_uncapped', 0.5, null, 'fft'],
  ['tsnet_rn_capped', 0.35, 8, 'exact'],
]) {
  const res = runPairAndCheckStep(
    () => mod.TsNetSolver.fromGraphRowNormalized(g16, { w2: w2Val, max2hopPerNode: capVal, backend: bk, iterationsStage1: 4, iterationsStage2: 4, iterationsStage3: 4, seed: 42 }),
    init16,
    5,
  );
  solvers[tag] = res.coords;
  solvers[`${tag}_ffts`] = res.solverRun.totalFft2dCalls;
  res.solverRun.free?.();
  res.solverStep.free?.();
}

const sgdSpecs = [
  ['sgd_full', () => mod.SgdSolver.full(g16, { tMax: 10, eps: 0.01, kind: 'exponential', stepScale: 1.0, seed: 42 })],
  ['sgd_dog', () => mod.SgdSolver.sparseBarrierDog(g16, { degree: 15, numLandmarks: 8, tMax: 10, eps: 0.01, seed: 42 })],
  ['sgd_rp', () => mod.SgdSolver.randomPair(g16, { pairsPerNode: 4, tMax: 10, eps: 0.01, seed: 42 })],
  ['sgd_omega', () => mod.SgdSolver.omega(g16, emb16, 8, { k: 4, minDist: 1e-3, tMax: 10, eps: 0.01, seed: 42 })],
  ['sgd_stream', () => mod.SgdSolver.streamingNegatives(g16, { negativesPerNode: 3, targetDistance: 2.0, weight: 0.25, hingeOnly: true, tMax: 10, eps: 0.01, seed: 42 })],
  [
    'sgd_hinges',
    () =>
      mod.SgdSolver.fromPairsWithHinges(
        16,
        new Uint32Array(spec.hinge_src),
        new Uint32Array(spec.hinge_tgt),
        new Float32Array(spec.hinge_ones),
        new Float32Array(spec.hinge_ones),
        new Float32Array(spec.hinge_ones),
        new Float32Array(spec.hinge_ones),
        new Uint32Array([0, 1]),
        new Uint32Array([15, 14]),
        new Float32Array([3.0, 3.0]),
        new Float32Array([3.0, 3.0]),
        new Float32Array([1.0, 1.0]),
        new Float32Array([1.0, 1.0]),
        { tMax: 10, eps: 0.01, seed: 42 },
      ),
  ],
];
for (const [k, mk] of sgdSpecs) {
  const res = runPairAndCheckStep(mk, init16, 4);
  solvers[k] = res.coords;
  solvers[`${k}_springs`] = res.solverRun.numSprings;
  solvers[`${k}_hinges`] = res.solverRun.numHinges;
  res.solverRun.free?.();
  res.solverStep.free?.();
}

{
  const res = runPairAndCheckStep(
    () => mod.GuMapSolver.fromGraphDog(g16, { max2hopPerNode: 16, degree: 15, numLandmarks: 8, nEpochs: 15, learningRate: 0.2, seed: 42 }),
    gumapInit16,
    5,
  );
  solvers.gumap = res.coords;
  solvers.gumap_edges = res.solverRun.numEdges;
  res.solverRun.free?.();
  res.solverStep.free?.();
}

{
  const lm16 = mod.computeRegularizedDogVaradhanLandmarks(g16, 15, 8, 42);
  const res = runPairAndCheckStep(
    () => mod.GuMapSolver.fromLandmarks(g16, lm16, 16, { nEpochs: 15, learningRate: 0.2, seed: 42 }),
    gumapInit16,
    5,
  );
  solvers.gumap_lm = res.coords;
  solvers.gumap_lm_edges = res.solverRun.numEdges;
  res.solverRun.free?.();
  res.solverStep.free?.();
  lm16.free?.();
}

{
  const res = runPairAndCheckStep(
    () => mod.GuMapSolver.fromGraphDog(g16, { max2hopPerNode: 16, degree: 15, numLandmarks: 8, nEpochs: 15, learningRate: 0.2, symmetricNegative: false, seed: 42 }),
    gumapInit16,
    5,
  );
  solvers.gumap_asym = res.coords;
  res.solverRun.free?.();
  res.solverStep.free?.();
}

{
  const res = runPairAndCheckStep(
    () => new mod.SpatialIntruderRefiner(g9, { rounds: 3, kCap: 32, tMax: 10, seed: 42 }),
    folded9,
    1,
  );
  solvers.refiner = res.coords;
  const st = res.solverRun.stats();
  solvers.refiner_fp = st.nFpTotal;
  st.free?.();
  res.solverRun.free?.();
  res.solverStep.free?.();
}

{
  const rfDyn = new mod.SpatialIntruderRefiner(g9, { rounds: 4, kCap: 8, wAnchor2: 0.5, kCapDecay: 0.5, kCapMin: 2, tMax: 10, seed: 42 });
  const lDyn = mod.Layout2d.fromCoords(folded9);
  rfDyn.step(lDyn);
  assert.equal(rfDyn.currentKCap, 4);
  rfDyn.setKCap(6);
  assert.equal(rfDyn.currentKCap, 6);
  rfDyn.run(lDyn);
  solvers.refiner_dyn = Array.from(lDyn.coordsView());
  const stDyn = rfDyn.stats();
  solvers.refiner_dyn_fp = stDyn.nFpTotal;
  stDyn.free?.();
  lDyn.free?.();
  rfDyn.free?.();
}

{
  const res = runPairAndCheckStep(
    () => new mod.SpatialIntruderRefiner(g11, { rounds: 3, kCap: 16, tMax: 10, minSepRatio: 0.15, wSep: 1.0, seed: 42 }),
    twin11,
    1,
  );
  solvers.refiner_sep = res.coords;
  const stSep = res.solverRun.stats();
  solvers.refiner_sep_total = stSep.nSepTotal;
  stSep.free?.();
  res.solverRun.free?.();
  res.solverStep.free?.();
}

{
  const res = runPairAndCheckStep(
    () => mod.StressMajorizationSolver.fromGraph(g16, { maxIterations: 15, epsilon: 1e-6 }),
    init16,
    4,
  );
  solvers.sm = res.coords;
  solvers.sm_stress = res.solverRun.lastStress ?? 0.0;
  res.solverRun.free?.();
  res.solverStep.free?.();
}

{
  const res = runPairAndCheckStep(
    () => mod.KamadaKawaiSolver.fromGraph(g9, { eps: 1e-3, maxInnerIters: 40, maxOuterIters: 20 }),
    init9,
    4,
  );
  solvers.kk = res.coords;
  res.solverRun.free?.();
  res.solverStep.free?.();
}

const out = {
  m3_available: true,
  km: {
    labels: Array.from(km.labels()),
    centroids: Array.from(km.centroids()),
    inertia: km.inertia,
    iterations: km.iterations,
    converged: km.converged,
  },
  spectral: specLabels,
  louvain: {
    labels: Array.from(louv.labels()),
    numCommunities: louv.numCommunities,
    modularity: louv.modularity,
    passes: louv.passes,
  },
  lpa: lpaLabels,
  coarsen: {
    numNodes: cg.numNodes(),
    numEdges: cg.numEdges(),
    weights: Array.from(cg.weights() ?? []),
    nodeToCoarse: Array.from(coarse.nodeToCoarse()),
    coarseSizes: Array.from(coarse.coarseSizes()),
    numClusters: coarse.numClusters,
  },
  projCoords,
  manifolds,
  solvers,
};

g9.free?.();
g11.free?.();
g16.free?.();
proj.free?.();
cl.free?.();
cg.free?.();
coarse.free?.();
louv.free?.();
km.free?.();
bb.free?.();

console.log(JSON.stringify(out));
"""
    raw_out = subprocess.check_output(
        ["node", "--input-type=module", "-e", node_script, node_url, wasm_path, json.dumps(spec_payload)],
        text=True,
    )
    ws = json.loads(raw_out.strip().splitlines()[-1])
    if not ws.get("m3_available", False):
        return [{
            "case": "cross_target_m3_py_vs_wasm",
            "status": "skipped_no_wasm_m3",
            "elapsed_sec": round(time.perf_counter() - t0, 4),
        }]

    max_f32_diff = 0.0
    max_f64_diff = 0.0

    def check_close(label: str, a: Any, b: Any, tol: float, is_f32: bool = True) -> None:
        nonlocal max_f32_diff, max_f64_diff
        arr_a = np.asarray(a, dtype=np.float64).ravel()
        arr_b = np.asarray(b, dtype=np.float64).ravel()
        assert arr_a.shape == arr_b.shape, f"{label}: shape mismatch {arr_a.shape} vs {arr_b.shape}"
        diff = float(np.max(np.abs(arr_a - arr_b))) if arr_a.size else 0.0
        if is_f32:
            max_f32_diff = max(max_f32_diff, diff)
        else:
            max_f64_diff = max(max_f64_diff, diff)
        assert diff < tol, f"{label}: diff={diff} >= {tol}"

    # Clustering & coarsen exact/f64 checks
    assert list(km_py["labels"]) == ws["km"]["labels"]
    assert bool(km_py["converged"]) == bool(ws["km"]["converged"])
    assert int(km_py["iterations"]) == int(ws["km"]["iterations"])
    check_close("kmeans/centroids", km_py["centroids"], ws["km"]["centroids"], 1e-12, is_f32=False)
    check_close("kmeans/inertia", [km_py["inertia"]], [ws["km"]["inertia"]], 1e-12, is_f32=False)
    assert list(spec_py) == ws["spectral"]
    assert list(louv_py["labels"]) == ws["louvain"]["labels"]
    assert int(louv_py["num_communities"]) == int(ws["louvain"]["numCommunities"])
    assert int(louv_py["passes"]) == int(ws["louvain"]["passes"])
    check_close("louvain/modularity", [louv_py["modularity"]], [ws["louvain"]["modularity"]], 1e-12, is_f32=False)
    assert list(lpa_py) == ws["lpa"]
    assert cg_py.num_nodes == ws["coarsen"]["numNodes"]
    assert cg_py.num_edges == ws["coarsen"]["numEdges"]
    assert list(n2c_py) == ws["coarsen"]["nodeToCoarse"]
    assert list(sizes_py) == ws["coarsen"]["coarseSizes"]
    assert nclust_py == ws["coarsen"]["numClusters"]
    check_close("project_clustered_layout/f32", proj_py32, ws["projCoords"], 1e-4)
    check_close("project_clustered_layout/f64_vs_f32", proj_py64, ws["projCoords"], 1e-3, is_f32=False)

    # ManifoldLayout2d & SgdSolver manifold parity
    for kind in ("hyperbolic", "spherical", "torus"):
        check_close(f"manifold_{kind}/coords_f32", man_py32[kind]["coords"], ws["manifolds"][kind]["coords"], 1e-4)
        check_close(f"manifold_{kind}/dist_f32", [man_py32[kind]["dist"]], [ws["manifolds"][kind]["dist"]], 1e-4)
        check_close(f"manifold_{kind}/coords_f64_vs_f32", man_py64[kind]["coords"], ws["manifolds"][kind]["coords"], 1e-3, is_f32=False)

    # Euclidean solver parity: Python float32 vs WASM f32 (< 1e-4) and Python float64 sanity
    ws_sol = ws["solvers"]
    for key in (
        "tsnet_fft",
        "tsnet_linear_fft",
        "tsnet_barnes_hut",
        "tsnet_exact",
        "tsnet_rn_uncapped",
        "tsnet_rn_capped",
        "sgd_full",
        "sgd_dog",
        "sgd_rp",
        "sgd_omega",
        "sgd_stream",
        "sgd_hinges",
        "gumap",
        "gumap_lm",
        "gumap_asym",
        "refiner",
        "refiner_dyn",
        "refiner_sep",
        "sm",
        "kk",
    ):
        check_close(f"{key}/f32_vs_wasm", py32[key], ws_sol[key], 1e-4)
        # Verify Python float64 stays finite and matches float32 within scale-relative tolerance
        arr64 = np.asarray(py64[key], dtype=np.float64).ravel()
        arr32 = np.asarray(py32[key], dtype=np.float64).ravel()
        assert np.all(np.isfinite(arr64)), f"{key}/f64_vs_f32: non-finite float64 output"
        rel_scale = max(1.0, float(np.max(np.abs(arr64))))
        rel_diff = float(np.max(np.abs(arr64 - arr32))) / rel_scale if arr64.size else 0.0
        max_f64_diff = max(max_f64_diff, rel_diff)
        assert rel_diff < 5e-3, (
            f"{key}/f64_vs_f32: rel_diff={rel_diff} >= 5e-3 "
            f"(abs={float(np.max(np.abs(arr64 - arr32)))}, scale={rel_scale})"
        )

    for bk in ("fft", "linear_fft", "barnes_hut", "exact"):
        assert py32[f"tsnet_{bk}_ffts"] == ws_sol[f"tsnet_{bk}_ffts"]
    for tag in ("tsnet_rn_uncapped", "tsnet_rn_capped"):
        assert py32[f"{tag}_ffts"] == ws_sol[f"{tag}_ffts"]
    for sk in ("sgd_full", "sgd_dog", "sgd_rp", "sgd_omega", "sgd_stream", "sgd_hinges"):
        assert py32[f"{sk}_springs"] == ws_sol[f"{sk}_springs"]
        assert py32[f"{sk}_hinges"] == ws_sol[f"{sk}_hinges"]
    assert py32["gumap_edges"] == ws_sol["gumap_edges"]
    assert py32["gumap_lm_edges"] == ws_sol["gumap_lm_edges"]
    assert py32["refiner_fp"] == ws_sol["refiner_fp"]
    assert py32["refiner_dyn_fp"] == ws_sol["refiner_dyn_fp"]
    assert py32["refiner_sep_total"] == ws_sol["refiner_sep_total"] > 0
    check_close("sm_stress/f32_vs_wasm", [py32["sm_stress"]], [ws_sol["sm_stress"]], 1e-4)

    return [{
        "case": "cross_target_m3_py_vs_wasm",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "max_f32_diff": round(max_f32_diff, 12),
        "max_f64_diff": round(max_f64_diff, 12),
    }]


def run_cross_target_m4_parity(eg: Any, wasm_pkg: pathlib.Path) -> list[dict[str, Any]]:
    """Verify Python float32/float64 <-> WASM f32 parity across M4 solvers and post-processing."""
    import numpy as np

    if not hasattr(eg, "TFdpSolver"):
        return []
    node_entry = wasm_pkg / "egraph_wasm.node.js"
    index_entry = wasm_pkg / "index.js"
    if not index_entry.exists():
        return []

    t0 = time.perf_counter()

    g16_edges = [
        (r * 4 + c, r * 4 + c + 1) for r in range(4) for c in range(3)
    ] + [
        (r * 4 + c, (r + 1) * 4 + c) for r in range(3) for c in range(4)
    ]
    g16 = eg.CsrGraph.from_edges(16, g16_edges, directed=False, sanitize=True)
    init16_f32 = np.array(_make_m3_asymmetric_coords(16, scale=0.8), dtype=np.float32)
    init16_f64 = init16_f32.astype(np.float64)

    g25_edges = [
        (r * 5 + c, r * 5 + c + 1) for r in range(5) for c in range(4)
    ] + [
        (r * 5 + c, (r + 1) * 5 + c) for r in range(4) for c in range(5)
    ]
    g25 = eg.CsrGraph.from_edges(25, g25_edges, directed=False, sanitize=True)
    init25_f32 = np.array(_make_m3_asymmetric_coords(25, scale=1.0), dtype=np.float32)
    init25_f64 = init25_f32.astype(np.float64)

    knn16, _, _ = eg.build_pure_2hop_knn(g16)
    fuzzy16 = eg.compute_fuzzy_simplicial_set(knn16)
    deg16 = np.diff(g16.indptr()).astype(np.uint32)

    prism_init_f32 = (init16_f32 * 0.15).astype(np.float32)
    prism_init_f64 = prism_init_f32.astype(np.float64)
    hw16_f32 = np.full(16, 0.5, dtype=np.float32)
    hh16_f32 = np.full(16, 0.4, dtype=np.float32)
    rad16_f32 = np.full(16, 0.4, dtype=np.float32)

    c1d_init_f32 = np.array([2.0, 0.5, 0.0, 1.5], dtype=np.float32)
    rect_init_f32 = np.array([[0.0, 0.0], [0.2, 0.1], [0.1, -0.1], [-0.1, 0.2]], dtype=np.float32)
    hw4_f32 = np.full(4, 0.5, dtype=np.float32)
    hh4_f32 = np.full(4, 0.5, dtype=np.float32)
    cl_init_f32 = np.array([[0.0, 0.0], [0.5, 0.0], [0.3, 0.2], [0.8, 0.2]], dtype=np.float32)
    cl_ids = [0, 0, 1, 1]

    fdeb_init_f32 = np.array(
        [[float(c), float(r)] for r in range(4) for c in range(4)], dtype=np.float32
    )

    cyc_edges_list = [(0, 1), (1, 2), (2, 0), (1, 3)]
    skip_edges_list = [(0, 1), (1, 2), (2, 3), (4, 3)]
    g_cyc = eg.CsrGraph.from_edges(4, cyc_edges_list, directed=True, sanitize=True)
    g_skip = eg.CsrGraph.from_edges(5, skip_edges_list, directed=True, sanitize=True)

    ce_py = eg.cycle_edges(g_cyc)
    dag_py = eg.remove_cycles(g_cyc)
    layers_lp_py = eg.assign_layers(g_skip, method="longest_path")
    layers_ns_py = eg.assign_layers(g_skip, method="network_simplex")
    lm16_m4 = eg.compute_regularized_dog_varadhan_landmarks(g16, num_landmarks=8, seed=42)

    def run_py_m4(dtype_name: str) -> dict[str, Any]:
        c16 = init16_f32 if dtype_name == "float32" else init16_f64
        c25 = init25_f32 if dtype_name == "float32" else init25_f64
        cp = prism_init_f32 if dtype_name == "float32" else prism_init_f64
        np_dt = np.float32 if dtype_name == "float32" else np.float64

        out: dict[str, Any] = {}
        for bk in ("ibfft", "barnes_hut", "exact"):
            l_tf = eg.Layout2d.from_coords(c16)
            s_tf = eg.TFdpSolver(g16, backend=bk, total_iterations=8, seed=42, dtype=dtype_name)
            s_tf.run(l_tf)
            out[f"tfdp_{bk}"] = l_tf.coords(copy=True)

            l_ib = eg.Layout2d.from_coords(c16)
            s_ib = eg.IbUmapSolver(g16, backend=bk, n_epochs=6, seed=42, dtype=dtype_name)
            s_ib.run(l_ib)
            out[f"ibumap_{bk}"] = l_ib.coords(copy=True)

        l_tfk = eg.Layout2d.from_coords(c16)
        eg.TFdpSolver.from_knn(knn16, backend="exact", total_iterations=6, seed=42, dtype=dtype_name).run(l_tfk)
        out["tfdp_knn"] = l_tfk.coords(copy=True)

        l_ibk = eg.Layout2d.from_coords(c16)
        eg.IbUmapSolver.from_knn(knn16, backend="exact", n_epochs=6, seed=42, dtype=dtype_name).run(l_ibk)
        out["ibumap_knn"] = l_ibk.coords(copy=True)

        l_ibf = eg.Layout2d.from_coords(c16)
        eg.IbUmapSolver.from_fuzzy_set(fuzzy16, degrees=deg16, backend="exact", n_epochs=6, seed=42, dtype=dtype_name).run(l_ibf)
        out["ibumap_fset"] = l_ibf.coords(copy=True)

        for bk in ("exact", "ibfft"):
            l_ibl = eg.Layout2d.from_coords(c16)
            s_ibl = eg.IbUmapSolver.from_landmarks(
                g16, lm16_m4, max_2hop_per_node=16, backend=bk, n_epochs=6, seed=42, dtype=dtype_name
            )
            s_ibl.run(l_ibl)
            out[f"ibumap_lm_{bk}"] = l_ibl.coords(copy=True)
            out[f"ibumap_lm_{bk}_edges"] = int(s_ibl.num_edges)

        l_pmk = eg.Layout2d.from_coords(c16)
        s_pmk = eg.SgdSolver.pacmap_graph(
            g16,
            n_neighbors=2,
            phase1_iters=3,
            phase2_iters=3,
            phase3_iters=4,
            keep_all_1hop=True,
            seed=42,
            dtype=dtype_name,
        )
        s_pmk.run(l_pmk)
        out["sgd_pacmap_keep"] = l_pmk.coords(copy=True)
        out["sgd_pacmap_keep_springs"] = int(s_pmk.num_springs)

        l_pm_lm = eg.Layout2d.from_coords(c16)
        s_pm_lm = eg.SgdSolver.pacmap_graph_with_landmarks(
            g16,
            lm16_m4,
            n_neighbors=4,
            phase1_iters=3,
            phase2_iters=3,
            phase3_iters=4,
            keep_all_1hop=True,
            max_1hop_per_node=2,
            degree_weight_beta=0.5,
            seed=42,
            dtype=dtype_name,
        )
        s_pm_lm.run(l_pm_lm)
        out["sgd_pacmap_cap_beta_lm"] = l_pm_lm.coords(copy=True)
        out["sgd_pacmap_cap_beta_lm_springs"] = int(s_pm_lm.num_springs)

        sgd_specs = {
            "sgd_snap": eg.SgdSolver.snap_t_fdp(g16, iterations=8, negatives_per_edge=3, seed=42, dtype=dtype_name),
            "sgd_snap_beta": eg.SgdSolver.snap_t_fdp(g16, iterations=8, negatives_per_edge=3, degree_beta=0.5, bounded_attraction=False, seed=42, dtype=dtype_name),
            "sgd_pacmap": eg.SgdSolver.pacmap_graph(g16, n_neighbors=5, phase1_iters=3, phase2_iters=3, phase3_iters=4, seed=42, dtype=dtype_name),
            "sgd_sgd2": eg.SgdSolver.sgd2(g16, t_max=8, batch_size=16, seed=42, dtype=dtype_name),
            "sgd_squared": eg.SgdSolver.full(g16, t_max=8, schedule="squared", seed=42, dtype=dtype_name),
        }
        for k, sv in sgd_specs.items():
            l = eg.Layout2d.from_coords(c16)
            sv.run(l)
            out[k] = l.coords(copy=True)

        for tag, use_bh in (("fa2_bh", True), ("fa2_exact", False)):
            l_fa = eg.Layout2d.from_coords(c16)
            eg.ForceAtlas2Solver(g16, iterations=10, use_barnes_hut=use_bh, seed=42, dtype=dtype_name).run(l_fa)
            out[tag] = l_fa.coords(copy=True)

        l_fa_cap = eg.Layout2d.from_coords(c16)
        eg.ForceAtlas2Solver(g16, iterations=10, use_barnes_hut=True, max_displacement=0.25, seed=42, dtype=dtype_name).run(l_fa_cap)
        out["fa2_cap"] = l_fa_cap.coords(copy=True)

        l_yh = eg.Layout2d.from_coords(c25)
        yh = eg.YifanHuSolver(
            g25,
            max_iterations_per_level=10,
            min_coarsen_size=6,
            min_coarsen_ratio=0.9,
            convergence_tolerance=1e-6,
            seed=42,
            dtype=dtype_name,
        )
        yh.run(l_yh)
        out["yh"] = l_yh.coords(copy=True)
        out["yh_levels"] = int(yh.num_levels)

        l_pr = eg.Layout2d.from_coords(cp)
        pr = eg.PrismOverlapRemoval(
            hw16_f32.astype(np_dt), hh16_f32.astype(np_dt), max_iterations=50, padding=0.05, seed=42, dtype=dtype_name
        )
        pr.run(l_pr)
        out["prism_box"] = l_pr.coords(copy=True)
        out["prism_box_ov"] = int(pr.last_overlap_count)

        l_pr_rad = eg.Layout2d.from_coords(cp)
        pr_rad = eg.PrismOverlapRemoval.from_radii(
            rad16_f32.astype(np_dt), max_iterations=50, seed=42, dtype=dtype_name
        )
        pr_rad.run(l_pr_rad)
        out["prism_rad"] = l_pr_rad.coords(copy=True)
        out["prism_rad_ov"] = int(pr_rad.last_overlap_count)

        out["proj_1d"] = eg.project_1d(
            c1d_init_f32.astype(np_dt),
            [
                eg.SeparationConstraint(0, 1, 1.5),
                eg.SeparationConstraint(1, 2, 2.0),
                eg.SeparationConstraint(2, 3, 1.0),
            ],
        )

        layered_cs = eg.generate_layered_constraints(g_skip, layers_ns_py, min_layer_distance=2.0)
        l_ax = eg.Layout2d.zeros(5, dtype=dtype_name)
        eg.project_layout_axis(l_ax, 1, layered_cs)
        out["proj_axis"] = l_ax.coords(copy=True)

        l_rc = eg.Layout2d.from_coords(rect_init_f32.astype(np_dt))
        eg.project_rectangle_no_overlap_2d(l_rc, hw4_f32.astype(np_dt), hh4_f32.astype(np_dt))
        out["proj_rect"] = l_rc.coords(copy=True)

        l_crc = eg.Layout2d.from_coords(cl_init_f32.astype(np_dt))
        eg.project_clustered_rectangle_no_overlap_2d(l_crc, cl_ids, hw4_f32.astype(np_dt), hh4_f32.astype(np_dt))
        out["proj_cl_rect"] = l_crc.coords(copy=True)

        l_fd = eg.Layout2d.from_coords(fdeb_init_f32.astype(np_dt))
        fd = eg.FdebSolver(g16, l_fd, cycles=2, i0=6, dtype=dtype_name)
        fd.run()
        out["fdeb_offsets"] = fd.edge_offsets()
        out["fdeb_points"] = fd.points()
        return out

    py32 = run_py_m4("float32")
    py64 = run_py_m4("float64")

    node_url = node_entry.resolve().as_uri() if node_entry.exists() else index_entry.resolve().as_uri()
    wasm_path = (wasm_pkg / "egraph_wasm_bg.wasm").resolve().as_posix()
    spec_payload = {
        "g16_edges": [int(v) for edge in g16_edges for v in edge],
        "g25_edges": [int(v) for edge in g25_edges for v in edge],
        "cyc_edges": [int(v) for edge in cyc_edges_list for v in edge],
        "skip_edges": [int(v) for edge in skip_edges_list for v in edge],
        "init16": [float(v) for v in init16_f32.ravel()],
        "init25": [float(v) for v in init25_f32.ravel()],
        "prism_init": [float(v) for v in prism_init_f32.ravel()],
        "hw16": [float(v) for v in hw16_f32],
        "hh16": [float(v) for v in hh16_f32],
        "rad16": [float(v) for v in rad16_f32],
        "c1d_init": [float(v) for v in c1d_init_f32],
        "rect_init": [float(v) for v in rect_init_f32.ravel()],
        "hw4": [float(v) for v in hw4_f32],
        "hh4": [float(v) for v in hh4_f32],
        "cl_init": [float(v) for v in cl_init_f32.ravel()],
        "cl_ids": cl_ids,
        "fdeb_init": [float(v) for v in fdeb_init_f32.ravel()],
        "knn_indptr": [int(v) for v in knn16.indptr()],
        "knn_indices": [int(v) for v in knn16.indices()],
        "knn_dists": [float(v) for v in knn16.distances()],
        "deg16": [int(v) for v in deg16],
    }

    node_script = """
import { readFile } from 'node:fs/promises';

const nodeUrl = process.argv[1];
const wasmPath = process.argv[2];
const spec = JSON.parse(process.argv[3]);

const mod = await import(nodeUrl);
if (typeof mod.default === 'function') {
  try {
    const probe = mod.CsrGraph.fromCsr(new Uint32Array([0, 1, 2]), new Uint32Array([1, 0]));
    probe.free?.();
  } catch (_e) {
    await mod.default(await readFile(wasmPath));
  }
}

if (typeof mod.TFdpSolver !== 'function') {
  console.log(JSON.stringify({ m4_available: false }));
  process.exit(0);
}

const g16 = mod.CsrGraph.fromEdgeList(16, new Uint32Array(spec.g16_edges));
const g25 = mod.CsrGraph.fromEdgeList(25, new Uint32Array(spec.g25_edges));
const gCyc = mod.CsrGraph.fromEdgeList(4, new Uint32Array(spec.cyc_edges), null, true, true);
const gSkip = mod.CsrGraph.fromEdgeList(5, new Uint32Array(spec.skip_edges), null, true, true);

const init16 = new Float32Array(spec.init16);
const init25 = new Float32Array(spec.init25);
const prismInit = new Float32Array(spec.prism_init);

const knn16 = mod.KnnGraph.fromCsr(
  16,
  new Uint32Array(spec.knn_indptr),
  new Uint32Array(spec.knn_indices),
  new Float64Array(spec.knn_dists),
);
const fuzzy16 = mod.computeFuzzySimplicialSet(knn16);

const runPairAndCheckStep = (mkSolver, initCoords, stepChunk) => {
  const sR = mkSolver();
  const sS = mkSolver();
  const lR = mod.Layout2d.fromCoords(initCoords);
  const lS = mod.Layout2d.fromCoords(initCoords);
  sR.run(lR);
  sS.runSteps(lS, stepChunk);
  while (!sS.isFinished) sS.step(lS);
  const cR = Array.from(lR.coordsView());
  const cS = Array.from(lS.coordsView());
  if (JSON.stringify(cR) !== JSON.stringify(cS)) {
    throw new Error('WASM M4 step() != run()');
  }
  lR.free?.();
  lS.free?.();
  return { coords: cR, sR, sS };
};

const solvers = {};
for (const bk of ['ibfft', 'barnes_hut', 'exact']) {
  {
    const r = runPairAndCheckStep(() => new mod.TFdpSolver(g16, { backend: bk, totalIterations: 8, seed: 42 }), init16, 3);
    solvers[`tfdp_${bk}`] = r.coords;
    r.sR.free?.(); r.sS.free?.();
  }
  {
    const r = runPairAndCheckStep(() => new mod.IbUmapSolver(g16, { backend: bk, nEpochs: 6, seed: 42 }), init16, 2);
    solvers[`ibumap_${bk}`] = r.coords;
    r.sR.free?.(); r.sS.free?.();
  }
}

{
  const r = runPairAndCheckStep(
    () => mod.TFdpSolver.fromKnn(
      new Uint32Array(spec.knn_indptr),
      new Uint32Array(spec.knn_indices),
      new Float64Array(spec.knn_dists),
      { backend: 'exact', totalIterations: 6, seed: 42 },
    ),
    init16,
    2,
  );
  solvers.tfdp_knn = r.coords;
  r.sR.free?.(); r.sS.free?.();
}

{
  const r = runPairAndCheckStep(
    () => mod.IbUmapSolver.fromKnn(knn16, { backend: 'exact', nEpochs: 6, seed: 42 }),
    init16,
    2,
  );
  solvers.ibumap_knn = r.coords;
  r.sR.free?.(); r.sS.free?.();
}

{
  const r = runPairAndCheckStep(
    () => mod.IbUmapSolver.fromFuzzySet(fuzzy16, new Uint32Array(spec.deg16), { backend: 'exact', nEpochs: 6, seed: 42 }),
    init16,
    2,
  );
  solvers.ibumap_fset = r.coords;
  r.sR.free?.(); r.sS.free?.();
}

{
  const lm16 = mod.computeRegularizedDogVaradhanLandmarks(g16, null, 8, 42);
  for (const bk of ['exact', 'ibfft']) {
    const r = runPairAndCheckStep(
      () => mod.IbUmapSolver.fromLandmarks(g16, lm16, 16, { backend: bk, nEpochs: 6, seed: 42 }),
      init16,
      2,
    );
    solvers[`ibumap_lm_${bk}`] = r.coords;
    solvers[`ibumap_lm_${bk}_edges`] = r.sR.numEdges;
    r.sR.free?.(); r.sS.free?.();
  }
  lm16.free?.();
}

{
  const r = runPairAndCheckStep(
    () => mod.SgdSolver.pacmapGraph(g16, { nNeighbors: 2, phase1Iters: 3, phase2Iters: 3, phase3Iters: 4, keepAll1hop: true, seed: 42 }),
    init16,
    3,
  );
  solvers.sgd_pacmap_keep = r.coords;
  solvers.sgd_pacmap_keep_springs = r.sR.numSprings;
  r.sR.free?.(); r.sS.free?.();
}

{
  const lm16Pm = mod.computeRegularizedDogVaradhanLandmarks(g16, null, 8, 42);
  const r = runPairAndCheckStep(
    () => mod.SgdSolver.pacmapGraphWithLandmarks(g16, lm16Pm, { nNeighbors: 4, phase1Iters: 3, phase2Iters: 3, phase3Iters: 4, keepAll1hop: true, max1hopPerNode: 2, degreeWeightBeta: 0.5, seed: 42 }),
    init16,
    3,
  );
  solvers.sgd_pacmap_cap_beta_lm = r.coords;
  solvers.sgd_pacmap_cap_beta_lm_springs = r.sR.numSprings;
  r.sR.free?.(); r.sS.free?.();
  lm16Pm.free?.();
}

for (const [k, mk] of [
  ['sgd_snap', () => mod.SgdSolver.snapTFdp(g16, { iterations: 8, negativesPerEdge: 3, seed: 42 })],
  ['sgd_snap_beta', () => mod.SgdSolver.snapTFdp(g16, { iterations: 8, negativesPerEdge: 3, degreeBeta: 0.5, boundedAttraction: false, seed: 42 })],
  ['sgd_pacmap', () => mod.SgdSolver.pacmapGraph(g16, { nNeighbors: 5, phase1Iters: 3, phase2Iters: 3, phase3Iters: 4, seed: 42 })],
  ['sgd_sgd2', () => mod.SgdSolver.sgd2(g16, { tMax: 8, batchSize: 16, seed: 42 })],
  ['sgd_squared', () => mod.SgdSolver.full(g16, { tMax: 8, kind: 'squared', seed: 42 })],
  ['fa2_bh', () => new mod.ForceAtlas2Solver(g16, { iterations: 10, useBarnesHut: true, seed: 42 })],
  ['fa2_exact', () => new mod.ForceAtlas2Solver(g16, { iterations: 10, useBarnesHut: false, seed: 42 })],
  ['fa2_cap', () => new mod.ForceAtlas2Solver(g16, { iterations: 10, useBarnesHut: true, maxDisplacement: 0.25, seed: 42 })],
]) {
  const r = runPairAndCheckStep(mk, init16, 3);
  solvers[k] = r.coords;
  r.sR.free?.(); r.sS.free?.();
}

{
  const r = runPairAndCheckStep(
    () => new mod.YifanHuSolver(g25, { maxIterationsPerLevel: 10, minCoarsenSize: 6, minCoarsenRatio: 0.9, convergenceTolerance: 1e-6, seed: 42 }),
    init25,
    4,
  );
  solvers.yh = r.coords;
  solvers.yh_levels = r.sR.numLevels;
  r.sR.free?.(); r.sS.free?.();
}

{
  const hw = new Float32Array(spec.hw16);
  const hh = new Float32Array(spec.hh16);
  const r = runPairAndCheckStep(
    () => new mod.PrismOverlapRemoval(hw, hh, { maxIterations: 50, padding: 0.05, seed: 42 }),
    prismInit,
    2,
  );
  solvers.prism_box = r.coords;
  solvers.prism_box_ov = r.sR.lastOverlapCount;
  r.sR.free?.(); r.sS.free?.();
}

{
  const rad = new Float32Array(spec.rad16);
  const r = runPairAndCheckStep(
    () => mod.PrismOverlapRemoval.fromRadii(rad, { maxIterations: 50, seed: 42 }),
    prismInit,
    2,
  );
  solvers.prism_rad = r.coords;
  solvers.prism_rad_ov = r.sR.lastOverlapCount;
  r.sR.free?.(); r.sS.free?.();
}

const c1d = new Float32Array(spec.c1d_init);
const cs1d = [
  new mod.SeparationConstraint(0, 1, 1.5),
  new mod.SeparationConstraint(1, 2, 2.0),
  new mod.SeparationConstraint(2, 3, 1.0),
];
const proj1d = Array.from(mod.project1d(c1d, cs1d));
for (const c of cs1d) c.free?.();

const ce = Array.from(mod.cycleEdges(gCyc));
const dag = mod.removeCycles(gCyc);
const dagCe = Array.from(mod.cycleEdges(dag));
const layersLp = Array.from(mod.assignLayers(gSkip, 'longestPath'));
const layersNs = Array.from(mod.assignLayers(gSkip, 'networkSimplex'));
const layeredCs = mod.generateLayeredConstraints(gSkip, new Uint32Array(layersNs), 2.0);
const lAx = mod.Layout2d.zeros(5);
mod.projectLayoutAxis(lAx, 1, layeredCs);
const projAxis = Array.from(lAx.coordsView());
for (const c of layeredCs) c.free?.();

const lRc = mod.Layout2d.fromCoords(new Float32Array(spec.rect_init));
mod.projectRectangleNoOverlap2d(lRc, new Float32Array(spec.hw4), new Float32Array(spec.hh4));
const projRect = Array.from(lRc.coordsView());

const lCrc = mod.Layout2d.fromCoords(new Float32Array(spec.cl_init));
mod.projectClusteredRectangleNoOverlap2d(lCrc, new Uint32Array(spec.cl_ids), new Float32Array(spec.hw4), new Float32Array(spec.hh4));
const projClRect = Array.from(lCrc.coordsView());

const lFd = mod.Layout2d.fromCoords(new Float32Array(spec.fdeb_init));
const fdR = new mod.FdebSolver(g16, lFd, { cycles: 2, i0: 6 });
const fdS = new mod.FdebSolver(g16, lFd, { cycles: 2, i0: 6 });
fdR.run();
fdS.step();
fdS.runSteps(3);
while (!fdS.isFinished) fdS.step();
const fdebOff = Array.from(fdR.edgeOffsets());
const fdebPts = Array.from(fdR.points());
if (JSON.stringify(fdebPts) !== JSON.stringify(Array.from(fdS.points()))) {
  throw new Error('WASM FdebSolver step() != run()');
}

const out = {
  m4_available: true,
  solvers,
  proj1d,
  projAxis,
  projRect,
  projClRect,
  ce,
  dagNumEdges: dag.numEdges(),
  dagCeLen: dagCe.length,
  layersLp,
  layersNs,
  fdebOff,
  fdebPts,
};

fdR.free?.(); fdS.free?.(); lFd.free?.();
lCrc.free?.(); lRc.free?.(); lAx.free?.();
dag.free?.(); fuzzy16.free?.(); knn16.free?.();
gSkip.free?.(); gCyc.free?.(); g25.free?.(); g16.free?.();

console.log(JSON.stringify(out));
"""
    raw_out = subprocess.check_output(
        ["node", "--input-type=module", "-e", node_script, node_url, wasm_path, json.dumps(spec_payload)],
        text=True,
    )
    ws = json.loads(raw_out.strip().splitlines()[-1])
    if not ws.get("m4_available", False):
        return [{
            "case": "cross_target_m4_py_vs_wasm",
            "status": "skipped_no_wasm_m4",
            "elapsed_sec": round(time.perf_counter() - t0, 4),
        }]

    max_f32_diff = 0.0
    max_f64_rel = 0.0

    def check_f32_wasm(label: str, a: Any, b: Any, tol: float = 1e-4) -> None:
        nonlocal max_f32_diff
        arr_a = np.asarray(a, dtype=np.float64).ravel()
        arr_b = np.asarray(b, dtype=np.float64).ravel()
        assert arr_a.shape == arr_b.shape, f"{label}: shape mismatch {arr_a.shape} vs {arr_b.shape}"
        diff = float(np.max(np.abs(arr_a - arr_b))) if arr_a.size else 0.0
        max_f32_diff = max(max_f32_diff, diff)
        assert diff < tol, f"{label}: f32_vs_wasm diff={diff} >= {tol}"

    def check_f64_vs_f32(label: str, a64: Any, a32: Any, tol: float = 5e-3) -> None:
        nonlocal max_f64_rel
        arr64 = np.asarray(a64, dtype=np.float64).ravel()
        arr32 = np.asarray(a32, dtype=np.float64).ravel()
        assert np.all(np.isfinite(arr64)), f"{label}: non-finite float64 output"
        rel_scale = max(1.0, float(np.max(np.abs(arr64))))
        rel_diff = float(np.max(np.abs(arr64 - arr32))) / rel_scale if arr64.size else 0.0
        max_f64_rel = max(max_f64_rel, rel_diff)
        assert rel_diff < tol, (
            f"{label}: f64_vs_f32 rel_diff={rel_diff} >= {tol} "
            f"(abs={float(np.max(np.abs(arr64 - arr32)))}, scale={rel_scale})"
        )

    ws_sol = ws["solvers"]
    solver_keys = (
        "tfdp_ibfft",
        "tfdp_barnes_hut",
        "tfdp_exact",
        "tfdp_knn",
        "ibumap_ibfft",
        "ibumap_barnes_hut",
        "ibumap_exact",
        "ibumap_knn",
        "ibumap_fset",
        "ibumap_lm_exact",
        "ibumap_lm_ibfft",
        "sgd_snap",
        "sgd_snap_beta",
        "sgd_pacmap",
        "sgd_pacmap_keep",
        "sgd_pacmap_cap_beta_lm",
        "sgd_sgd2",
        "sgd_squared",
        "fa2_bh",
        "fa2_exact",
        "fa2_cap",
        "yh",
        "prism_box",
        "prism_rad",
    )
    for key in solver_keys:
        check_f32_wasm(key, py32[key], ws_sol[key], 1e-4)
        check_f64_vs_f32(key, py64[key], py32[key], 5e-3)

    assert py32["yh_levels"] == ws_sol["yh_levels"]
    assert py32["ibumap_lm_exact_edges"] == ws_sol["ibumap_lm_exact_edges"]
    assert py32["ibumap_lm_ibfft_edges"] == ws_sol["ibumap_lm_ibfft_edges"]
    assert py32["sgd_pacmap_keep_springs"] == ws_sol["sgd_pacmap_keep_springs"]
    assert py32["sgd_pacmap_cap_beta_lm_springs"] == ws_sol["sgd_pacmap_cap_beta_lm_springs"]
    assert py32["prism_box_ov"] == ws_sol["prism_box_ov"] == 0
    assert py32["prism_rad_ov"] == ws_sol["prism_rad_ov"] == 0

    for pp_key, ws_pp_key in (
        ("proj_1d", "proj1d"),
        ("proj_axis", "projAxis"),
        ("proj_rect", "projRect"),
        ("proj_cl_rect", "projClRect"),
        ("fdeb_points", "fdebPts"),
    ):
        check_f32_wasm(pp_key, py32[pp_key], ws[ws_pp_key], 1e-4)
        check_f64_vs_f32(pp_key, py64[pp_key], py32[pp_key], 5e-3)

    assert list(ce_py.ravel()) == ws["ce"]
    assert dag_py.num_edges == ws["dagNumEdges"]
    assert ws["dagCeLen"] == 0
    assert list(layers_lp_py) == ws["layersLp"]
    assert list(layers_ns_py) == ws["layersNs"]
    assert list(py32["fdeb_offsets"]) == ws["fdebOff"]

    return [{
        "case": "cross_target_m4_py_vs_wasm",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "max_f32_diff": round(max_f32_diff, 12),
        "max_f64_diff": round(max_f64_rel, 12),
    }]


def run_cross_target_m5_parity(eg: Any, wasm_pkg: pathlib.Path) -> list[dict[str, Any]]:
    """Verify Python <-> WASM M5 API parity across core, dataset, spatial, metrics, linalg, and solvers."""
    import numpy as np

    assert hasattr(eg, "LayoutNd"), "Python missing M5 LayoutNd"
    assert hasattr(eg, "load_dataset"), "Python missing M5 load_dataset"
    index_entry = wasm_pkg / "index.js"
    node_entry = wasm_pkg / "egraph_wasm.node.js"
    assert index_entry.exists(), f"Missing WASM index.js at {index_entry}"

    t0 = time.perf_counter()

    # 1. CsrGraph M5 methods & LCC + dataset/CSV
    g_disc = eg.parse_csr_csv("6,5\n0,1\n2,3\n3,4\n4,5\n5,2\n", directed=False, sanitize=True)
    assert g_disc.nnz == 10
    assert g_disc.is_weighted is False
    assert g_disc.degree(2) == 2
    assert list(g_disc.neighbors(2)) == [3, 5]
    sub_py = g_disc.induced_subgraph([2, 3, 4])
    lcc_py, lcc_ids_py = g_disc.largest_connected_component()
    karate_py = eg.load_dataset("karate")

    # 2. Layout2d M5 methods & LayoutNd
    l2_empty = eg.Layout2d.zeros(0, dtype="float32")
    assert l2_empty.is_empty() is True
    l2_py = eg.Layout2d.zeros(4, dtype="float32")
    assert l2_py.num_nodes == 4
    assert l2_py.is_empty() is False
    l2_py.set_x(0, 1.0)
    l2_py.set_y(0, -1.0)
    l2_py.set_x(1, 3.5)
    l2_py.set_y(1, 2.5)
    l2_x1, l2_y1 = l2_py.x(1), l2_py.y(1)
    assert math.isclose(l2_x1, 3.5, rel_tol=1e-6)
    assert math.isclose(l2_y1, 2.5, rel_tol=1e-6)
    l2_py.normalize_std(target_std=1.5)
    l2_norm_coords = l2_py.copy_coords()

    l2_init_py = eg.Layout2d.init(5, seed=42, dtype="float32")
    assert l2_init_py.num_nodes == 5
    l2_init_coords_py = l2_init_py.copy_coords()

    nd_zero = eg.LayoutNd.zeros(3, 4, dtype="float32")
    assert nd_zero.num_nodes == 3 and nd_zero.dim == 4 and nd_zero.is_all_finite() and not nd_zero.is_empty()
    nd_arr_f32 = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    nd_py = eg.LayoutNd.from_coords(nd_arr_f32)
    assert nd_py.num_nodes == 3 and nd_py.dim == 4 and nd_py.is_all_finite()

    # 3. ManifoldLayout2d M5 methods & geodesic helpers
    ml_hyp_py = eg.ManifoldLayout2d.hyperbolic_poincare(2, dtype="float32")
    ml_sph_py = eg.ManifoldLayout2d.spherical(2, dtype="float32")
    ml_tor_py = eg.ManifoldLayout2d.torus_2d(2, dtype="float32")
    assert ml_hyp_py.manifold == "hyperbolic" and ml_hyp_py.num_nodes == 2
    assert ml_sph_py.manifold == "spherical" and ml_sph_py.num_nodes == 2
    assert ml_tor_py.manifold == "torus" and ml_tor_py.num_nodes == 2

    ml_py = eg.ManifoldLayout2d.zeros("hyperbolic", 2, dtype="float32")
    ml_py.set_point(0, 0.1, 0.2)
    ml_py.set_point(1, -0.1, 0.05)
    ml_x0, ml_y0 = ml_py.x(0), ml_py.y(0)
    td_vx, td_vy = ml_py.tangent_delta(0, 1)
    ml_py.apply_tangent_step(0, td_vx * 0.1, td_vy * 0.1)
    ml_coords_after = ml_py.coords()
    ml_rand_py = eg.ManifoldLayout2d.random_init("hyperbolic", 4, scale=0.25, seed=42, dtype="float32")
    ml_rand_py.center_poincare_centroid()
    ml_rand_cent_py = ml_rand_py.poincare_centroid()
    ml_scaled_py = eg.ManifoldLayout2d.from_layout_scaled(
        "hyperbolic", eg.Layout2d.from_coords(np.array([[10.0, -5.0], [20.0, 5.0], [30.0, 15.0]], dtype=np.float32)), max_radius=0.7
    )
    ml_scaled_py.recenter_poincare(0.1, -0.1)
    ml_scaled_py.recenter_poincare_at(1)
    geo_dists_py = [
        float(eg.poincare_distance(0.0, 0.0, 0.3, 0.4)),
        float(eg.spherical_distance(0.0, 0.0, 0.0, math.pi / 2.0)),
        float(eg.torus_distance(0.05, 0.05, 0.95, 0.05)),
        *[float(v) for v in eg.wrap_torus_point(1.25, -0.25)],
        *[float(v) for v in eg.poincare_mobius_add(0.2, -0.1, -0.15, 0.25)],
    ]

    # 4. Spatial & Metrics M5
    g_grid = eg.CsrGraph.from_edge_list(
        9,
        [(r * 3 + c, r * 3 + c + 1) for r in range(3) for c in range(2)]
        + [(r * 3 + c, (r + 1) * 3 + c) for r in range(2) for c in range(3)],
        directed=False,
        sanitize=True,
    )
    grid_coords_f32 = np.array(
        [[float(c) + 0.03 * float(r), float(r) - 0.02 * float(c)] for r in range(3) for c in range(3)],
        dtype=np.float32,
    )
    l_grid = eg.Layout2d.from_coords(grid_coords_f32)
    del_csr_py = eg.delaunay_to_csr(l_grid)
    rect_coords_f32 = np.array([[0.0, 0.0], [0.5, 0.0], [5.0, 0.0]], dtype=np.float32)
    rect_hw = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    rect_ov_py = eg.rectangle_overlaps(eg.Layout2d.from_coords(rect_coords_f32), rect_hw, rect_hw)

    g_x = eg.CsrGraph.from_edges(4, [(0, 3), (1, 2)], directed=False, sanitize=True)
    x_coords_f32 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    l_x = eg.Layout2d.from_coords(x_coords_f32)
    ce_py = eg.crossing_edges(g_x, l_x)
    cet_py = eg.crossing_edges_torus(g_x, l_x)
    cs_py = eg.crossing_summary(g_x, l_x)
    gg_py = eg.gabriel_graph_edges(l_grid)
    rng_py = eg.relative_neighborhood_edges(l_grid)
    apsp_py = eg.all_pairs_bfs(g_grid, unit_edge_length=1.0)
    sssp_py = eg.bfs_single_source(g_grid, 0, unit_edge_length=2.0)
    sns_dist_py = float(eg.scale_normalized_stress_from_distances(l_grid, apsp_py))
    iel_unit_py = float(eg.ideal_edge_lengths_with_unit(g_grid, l_grid, 1.5))
    eval_iel_py = eg.evaluate_metrics(
        g_grid, l_grid, ["sns", "ideal_edge_lengths"], num_pivots=4, num_samples=64, seed=42, ideal_edge_length=1.5
    )
    eval_default_py = eg.evaluate_metrics(
        g_grid, l_grid, ["sampled_sns", "sampled_crossing_number"]
    )
    sampled_sns_default_py = float(eg.sampled_scale_normalized_stress(g_grid, l_grid))
    sampled_cr_default_py = float(eg.sampled_crossing_number(g_grid, l_grid))
    l_grid_adj = eg.Layout2d.from_coords(grid_coords_f32 + np.array([0.05, -0.03], dtype=np.float32))
    pd_edges_py = eg.prism_distortion_with_edges(l_grid, l_grid_adj, gg_py)

    # 4b. W3 metrics & __version__
    import importlib.metadata as _imd

    py_pkg_ver = _imd.version("egraph")
    assert eg.__version__ == eg._core.__version__ == py_pkg_ver, (
        f"__version__ mismatch: {eg.__version__!r} vs {eg._core.__version__!r} vs {py_pkg_ver!r}"
    )
    w3_comm_labels = [0, 0, 0, 0, 0, 1, 1, 1, 1]
    cl_py = float(eg.crosslessness(g_grid, l_grid))
    scl_py = float(eg.sampled_crosslessness(g_grid, l_grid, num_samples=100, seed=42))
    ma_py = float(eg.minimum_angle(g_grid, l_grid))
    ma_sum_py = eg.minimum_angle_summary(g_grid, l_grid)
    deg_py = eg.layout_degeneracy(g_grid, l_grid)
    nig_py = float(eg.adjusted_patch_nig(l_grid, w3_comm_labels, patch_radius=16, canvas_size=2048, pad_frac=0.04, seed=42))
    nig_ms_py = eg.adjusted_patch_nig_multiscale(l_grid, w3_comm_labels, radii=[16, 32], canvas_size=2048, pad_frac=0.04, seed=42)
    rta_py = float(eg.landmark_stratified_rta(g_grid, l_grid, num_landmarks=4, num_triplets=50, min_degree=1, seed=42))
    rta_det_py = eg.landmark_stratified_rta_detail(g_grid, l_grid, num_landmarks=4, num_triplets=50, min_degree=1, seed=42)
    w3_lm_ids = eg.select_maxmin_sp_pivots(g_grid, num_pivots=3, seed=42, min_degree=1)
    w3_apsp = eg.all_pairs_bfs(g_grid, 1.0)
    w3_tbl = np.where(np.isfinite(w3_apsp[w3_lm_ids]), w3_apsp[w3_lm_ids], np.iinfo(np.uint32).max).astype(np.uint32)
    rta_tbl_py = eg.landmark_stratified_rta_from_landmarks(l_grid, w3_lm_ids, w3_tbl, num_triplets=1000, seed=42)
    w3_eval_py = eg.evaluate_metrics(
        g_grid,
        l_grid,
        [
            "crosslessness",
            "sampled_crosslessness",
            "minimum_angle",
            "minimum_angle_a1",
            "layout_degeneracy",
            "adjusted_patch_nig",
            "cluster_area_faithfulness",
            "landmark_stratified_rta",
        ],
        num_samples=100,
        seed=42,
        community_labels=w3_comm_labels,
        num_landmarks=4,
        num_triplets=50,
        min_degree=1,
        patch_radius=16,
        canvas_size=2048,
        pad_frac=0.04,
    )
    gcaf_grid_py = float(eg.cluster_area_faithfulness(l_grid, w3_comm_labels))
    gcaf_grid_det_py = eg.cluster_area_faithfulness_detail(l_grid, w3_comm_labels, return_per_cluster=True)
    imb24_coords_f32 = np.array(
        [
            [
                (-2.0 + 0.5 * (i / 19.0)) if i < 20 else (1.5 + 0.5 * ((i - 20) / 3.0)),
                math.sin(i * 0.7) * 0.2,
            ]
            for i in range(24)
        ],
        dtype=np.float32,
    )
    imb24_labels = [0] * 20 + [1] * 4
    l_imb24_py = eg.Layout2d.from_coords(imb24_coords_f32)
    g_imb24_py = eg.CsrGraph.from_edges(24, [(i, i + 1) for i in range(23)], directed=False, sanitize=True)
    gcaf_imb_def_py = float(eg.cluster_area_faithfulness(l_imb24_py, imb24_labels))
    gcaf_imb_custom_py = float(
        eg.cluster_area_faithfulness(
            l_imb24_py,
            imb24_labels,
            g_max=64,
            sigma_cells=0.9,
            kappa=8.0,
            tau_frac=0.2,
            boundary_r95=2.5,
            stevens_gamma=0.7,
        )
    )
    gcaf_imb_custom_det_py = eg.cluster_area_faithfulness_detail(
        l_imb24_py,
        imb24_labels,
        g_max=64,
        sigma_cells=0.9,
        kappa=8.0,
        tau_frac=0.2,
        boundary_r95=2.5,
        stevens_gamma=0.7,
        return_per_cluster=True,
    )
    gcaf_imb_inf_det_py = eg.cluster_area_faithfulness_detail(
        l_imb24_py, imb24_labels, boundary_r95=float("inf"), return_per_cluster=True
    )
    gcaf_imb_g05_det_py = eg.cluster_area_faithfulness_detail(
        l_imb24_py, imb24_labels, stevens_gamma=0.5, return_per_cluster=True
    )
    gcaf_imb_g20_det_py = eg.cluster_area_faithfulness_detail(
        l_imb24_py, imb24_labels, stevens_gamma=2.0, return_per_cluster=True
    )
    imb24_nw = [1.5 if lbl == 0 else 3.0 for lbl in imb24_labels]
    imb24_cw = [30.0, 12.0]
    gcaf_imb_nw_det_py = eg.cluster_area_faithfulness_detail(
        l_imb24_py, imb24_labels, stevens_gamma=2.0, node_weights=imb24_nw, return_per_cluster=True
    )
    gcaf_imb_cw_det_py = eg.cluster_area_faithfulness_detail(
        l_imb24_py, imb24_labels, stevens_gamma=2.0, cluster_weights=imb24_cw, return_per_cluster=True
    )
    gcaf_imb_eval_py = eg.evaluate_metrics(
        g_imb24_py,
        l_imb24_py,
        ["gcaf"],
        community_labels=imb24_labels,
        g_max=64,
        sigma_cells=0.9,
        kappa=8.0,
        tau_frac=0.2,
        boundary_r95=2.5,
        stevens_gamma=0.7,
    )
    int20_coords_f64 = np.array(
        [[float(c + (0 if c < 2 else 4)), float(r)] for c in range(4) for r in range(5)],
        dtype=np.float64,
    )
    int20_labels = [0 if c < 2 else 1 for c in range(4) for _r in range(5)]
    gcaf_int20_f64_py = eg.cluster_area_faithfulness_detail(
        eg.Layout2d.from_coords(int20_coords_f64), int20_labels, return_per_cluster=True
    )
    gcaf_int20_f32_py = eg.cluster_area_faithfulness_detail(
        eg.Layout2d.from_coords(int20_coords_f64.astype(np.float32)), int20_labels, return_per_cluster=True
    )
    gcaf_b0_py = eg.cluster_area_faithfulness_detail(eg.Layout2d.zeros(0), [], return_per_cluster=True)
    gcaf_b1_py = eg.cluster_area_faithfulness_detail(eg.Layout2d.from_coords(np.array([[1.0, 2.0]], dtype=np.float32)), [5], return_per_cluster=True)
    gcaf_bk1_py = eg.cluster_area_faithfulness_detail(eg.Layout2d.from_coords(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)), [3, 3, 3, 3], return_per_cluster=True)
    gcaf_bcol_py = float(eg.cluster_area_faithfulness(eg.Layout2d.from_coords(np.array([[2.0, -1.0]] * 6, dtype=np.float32)), [0, 0, 0, 1, 1, 1]))
    gcaf_small_n_cases = _gcaf_small_n_regression_cases()
    gcaf_small_n_py: dict[str, dict[str, Any]] = {}
    for case in gcaf_small_n_cases:
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
        gcaf_small_n_py[case["name"]] = {
            "caf": float(eg.cluster_area_faithfulness(l_case, case["labels"], **kwargs)),
            "det": eg.cluster_area_faithfulness_detail(l_case, case["labels"], return_per_cluster=True, **kwargs),
            "eval": eg.evaluate_metrics(g_case, l_case, ["gcaf"], community_labels=case["labels"], **kwargs),
        }
    g_dir4_py = eg.CsrGraph.from_edges(
        4,
        [(0, 1), (0, 3), (1, 3), (3, 0), (3, 1), (3, 2)],
        directed=True,
        sanitize=True,
    )
    l_dir4_py = eg.Layout2d.from_coords(
        np.array([[0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0]], dtype=np.float32)
    )
    cl_dir4_py = float(eg.crosslessness(g_dir4_py, l_dir4_py))
    scl_dir4_py = float(eg.sampled_crosslessness(g_dir4_py, l_dir4_py, num_samples=100, seed=42))
    assert cl_dir4_py == 0.0, f"expected dir4 crosslessness == 0.0, got {cl_dir4_py}"
    assert scl_dir4_py == 0.0, f"expected dir4 sampled_crosslessness == 0.0, got {scl_dir4_py}"
    g_rstar_py = eg.CsrGraph.from_edges(
        4,
        [(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0)],
        directed=True,
        sanitize=True,
    )
    l_rstar_py = eg.Layout2d.from_coords(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    )
    cl_rstar_py = float(eg.crosslessness(g_rstar_py, l_rstar_py))
    assert cl_rstar_py == 1.0, f"expected reciprocal star crosslessness == 1.0, got {cl_rstar_py}"

    # 5. Linalg & Neighborhood M5
    g16 = eg.CsrGraph.from_edges(
        16,
        [(r * 4 + c, r * 4 + c + 1) for r in range(4) for c in range(3)]
        + [(r * 4 + c, (r + 1) * 4 + c) for r in range(3) for c in range(4)],
        directed=False,
        sanitize=True,
    )
    cmds_py = eg.classical_mds_2d(g16, dtype="float32")
    eg.normalize_layout_2d(cmds_py, target_std=1.0, seed=42)
    rob_py = eg.classical_mds_2d(g16, dtype="float32")
    eg.robust_normalize_layout_2d(rob_py, target_std=1.5, quantile=0.95, seed=42)
    rd_py = eg.rd_mds(g16, dims=3, seed=42, dtype="float32")
    res_coords_py, res_lams_py = eg.compute_resistance_embedding(g16, dim=3, seed=42)
    diff_coords_py, diff_lams_py, diff_t_py = eg.compute_diffusion_embedding(g16, dim=3, p=0.5, seed=42)
    rep_in = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]], dtype=np.float64)
    rep_py = eg.representative_mds(rep_in, dims=2, dtype="float32")
    rerank_py = eg.build_rerank_knn(diff_coords_py, k=4, coarse_dim=2, rerank_factor=2)
    capped_knn_py, capped_k2_py, _ = eg.build_capped_12hop_knn(g16, max_2hop_per_node=4, seed=42)
    row_probs_py = eg.graph_row_normalized_probabilities(g16)
    row_probs12_py = eg.graph_12hop_row_normalized_probabilities(g16, w2=0.5, max_2hop_per_node=4, degree_beta=0.35, seed=42)
    sampled_sns_strat_py = float(eg.sampled_scale_normalized_stress(g16, cmds_py, num_pivots=4, seed=42))
    sym_probs_py = eg.symmetrize_row_normalized(3, [0, 1, 2, 3], [1, 2, 0], [1.0, 1.0, 1.0])
    prio_py = eg.build_gumap_12hop_priority_csr(3, [0], [1], [1.0], [0, 1], [2, 2], [2.0, 2.0], tier2_margin=1.1)
    f1_py, f2_py, f3_py, floor_py = eg.apply_relative_distance_floor([1e-6, 1.0], [2.0], [4.0], eps_dist=1e-3, min_ratio=0.1)

    # 6. Solvers M5 (TsNet partial_bfs + last_fft_stats, TFdp.from_knn(KnnGraph), IbUmap DoG/ibfft + ab, Sgd.dog_stats, Fdeb, runProgressive)
    init16_f32 = np.array(_make_m3_asymmetric_coords(16, scale=0.8), dtype=np.float32)
    l_ts_py = eg.Layout2d.from_coords(init16_f32)
    ts_partial_py = eg.TsNetSolver.from_graph_partial_bfs(
        g16,
        k=8,
        uniform_perplexity=6.0,
        backend="fft",
        iterations_stage1=2,
        iterations_stage2=2,
        iterations_stage3=2,
        intervals=1,
        interpolation_order=1,
        min_grid_dim=16,
        max_grid_dim=64,
        seed=42,
        dtype="float32",
    )
    assert ts_partial_py.last_fft_stats() is None
    ts_partial_py.run(l_ts_py)
    ts_fft_stats_py = ts_partial_py.last_fft_stats()
    assert ts_fft_stats_py is not None

    l_ts_rn_py = eg.Layout2d.from_coords(init16_f32)
    ts_rn_py = eg.TsNetSolver.from_graph_row_normalized(
        g16,
        w2=0.5,
        max_2hop_per_node=4,
        degree_beta=0.35,
        backend="fft",
        iterations_stage1=2,
        iterations_stage2=2,
        iterations_stage3=2,
        intervals=1,
        interpolation_order=1,
        min_grid_dim=16,
        max_grid_dim=64,
        seed=42,
        dtype="float32",
    )
    ts_rn_py.run(l_ts_rn_py)

    l_tfdp_knn_py = eg.Layout2d.from_coords(init16_f32)
    tfdp_knn_py = eg.TFdpSolver.from_knn(
        rerank_py, backend="exact", total_iterations=4, ibfft_order=1, ibfft_intervals=16, ibfft_min_grid_dim=32, ibfft_max_grid_dim=128, seed=42, dtype="float32"
    )
    tfdp_knn_py.run(l_tfdp_knn_py)

    l_ib_py = eg.Layout2d.from_coords(init16_f32)
    ib_py = eg.IbUmapSolver(
        g16,
        num_landmarks=8,
        min_degree=1,
        jl_dim=8,
        degree_normalize=True,
        p_exp=0.5,
        eps_psd=1e-4,
        t_coarse=32.0,
        backend="ibfft",
        n_epochs=5,
        ibfft_order=1,
        ibfft_intervals=16,
        ibfft_min_grid_dim=32,
        ibfft_max_grid_dim=128,
        seed=42,
        dtype="float32",
    )
    ib_py.run(l_ib_py)

    lm_dog_default_py = eg.compute_regularized_dog_varadhan_landmarks(g16)
    sgd_dog_py = eg.SgdSolver.sparse_barrier_dog(g16, t_max=4, seed=42, min_degree=1, jl_dim=8, dtype="float32")
    dog_stats_py = sgd_dog_py.dog_stats()
    assert dog_stats_py is not None
    l_sgd_dog_py = eg.Layout2d.from_coords(init16_f32)
    sgd_dog_py.run(l_sgd_dog_py)

    gm_dog_py = eg.GuMapSolver.from_graph_dog(g16, min_degree=1, jl_dim=8, n_epochs=4, seed=42, dtype="float32")

    l_fdeb_py = eg.Layout2d.from_coords(init16_f32)
    fdeb_py = eg.FdebSolver(g16, l_fdeb_py, cycles=2, i0=4, dtype="float32")
    fdeb_py.run()

    hier_py = eg.louvain_coarsen(g16, max_coarse_nodes=4, seed=42)
    rhgh_lm_py, rhgh_st_py = eg.compute_pivoted_rhgh_landmarks(
        g16, num_landmarks=8, min_degree=1, num_probes=16, seed=42
    )
    g16_bb_edges = [
        (r * 4 + c, r * 4 + c + 1) for r in range(4) for c in range(3)
    ] + [
        (r * 4 + c, (r + 1) * 4 + c) for r in range(3) for c in range(4)
    ]
    g16_bb_weights = [1.0 + 0.05 * i for i in range(len(g16_bb_edges))]
    g16_bb_py = eg.CsrGraph.from_edges(16, g16_bb_edges, weights=g16_bb_weights, directed=False, sanitize=True)
    bb_py = eg.coarse_backbone_init(
        g16_bb_py,
        max_coarse_nodes=4,
        edge_length_mode="resistance_curvature_flow",
        distance_backend="rhgh",
        coarse_t_max=20,
        seed=42,
        dtype="float32",
    )
    bb_aligned_py = eg.Layout2d.from_coords(bb_py["fine_layout"].coords(copy=True))
    eg.align_clustered_centroids(bb_aligned_py, bb_py["coarse_layout"], bb_py["node_to_coarse"], gamma_tether=0.5)

    sp_field_py = eg.ScreenedPoissonField2d(
        kernel="screened_poisson",
        grid_size=64,
        padding_factor=2,
        sigma=0.2,
        kappa=1.0,
        rho_max=0.1,
        exact_adjoint=True,
        dtype="float32",
    )
    l_sp_init_py = eg.Layout2d.from_coords(init16_f32)
    sp_eval_py = sp_field_py.evaluate_in_bounds(l_sp_init_py, [-2.0, 2.0, -2.0, 2.0])
    l_sp_ref_py = eg.Layout2d.from_coords(init16_f32)
    sp_ref_py = eg.ScreenedPoissonRefiner.from_graph(
        g16,
        grid_size=64,
        padding_factor=2,
        sigma=0.2,
        kappa=1.0,
        rho_max=0.1,
        iterations=4,
        anchor_weight=0.1,
        seed=42,
        dtype="float32",
    )
    sp_ref_py.run(l_sp_ref_py)

    node_url = node_entry.resolve().as_uri() if node_entry.exists() else index_entry.resolve().as_uri()
    index_url = index_entry.resolve().as_uri()
    wasm_path = (wasm_pkg / "egraph_wasm_bg.wasm").resolve().as_posix()
    spec_payload = {
        "grid_coords": [float(v) for v in grid_coords_f32.ravel()],
        "g9_edges": [int(v) for r in range(3) for c in range(3) for v in ((r, c + 1) if c + 1 < 3 else ())],
        "rect_coords": [float(v) for v in rect_coords_f32.ravel()],
        "rect_hw": [float(v) for v in rect_hw],
        "x_coords": [float(v) for v in x_coords_f32.ravel()],
        "nd_coords": [float(v) for v in nd_arr_f32.ravel()],
        "g16_edges": [
            int(v)
            for edge in g16_bb_edges
            for v in edge
        ],
        "g16_bb_weights": g16_bb_weights,
        "init16": [float(v) for v in init16_f32.ravel()],
        "diff_coords": [float(v) for v in diff_coords_py.ravel()],
        "diff_dim": int(diff_coords_py.shape[1]),
        "rerank_indptr": [int(v) for v in rerank_py.indptr()],
        "rerank_indices": [int(v) for v in rerank_py.indices()],
        "rerank_dists": [float(v) for v in rerank_py.distances()],
        "imb24_coords": [float(v) for v in imb24_coords_f32.ravel()],
        "int20_coords": [float(v) for v in int20_coords_f64.astype(np.float32).ravel()],
        "int20_labels": int20_labels,
        "gcaf_small_n": [
            {
                **c,
                "boundary_r95": (
                    "Infinity"
                    if (c["boundary_r95"] is not None and math.isinf(c["boundary_r95"]))
                    else c["boundary_r95"]
                ),
            }
            for c in gcaf_small_n_cases
        ],
    }

    node_script = """
import { readFile } from 'node:fs/promises';
import assert from 'node:assert/strict';

const nodeUrl = process.argv[1];
const indexUrl = process.argv[2];
const wasmPath = process.argv[3];
const spec = JSON.parse(process.argv[4]);

const mod = await import(nodeUrl);
const indexMod = await import(indexUrl);
if (typeof mod.default === 'function') {
  try {
    const probe = mod.CsrGraph.fromCsr(new Uint32Array([0, 1, 2]), new Uint32Array([1, 0]));
    probe.free?.();
  } catch (_e) {
    await mod.default(await readFile(wasmPath));
  }
}

assert.equal(typeof mod.LayoutNd, 'function', 'WASM missing M5 LayoutNd');
assert.equal(typeof mod.loadDataset, 'undefined', 'WASM must not expose embedded loadDataset');
assert.equal(typeof indexMod.runProgressive, 'function', 'WASM index.js missing runProgressive');

const gDisc = mod.parseCsrCsv("6,5\\n0,1\\n2,3\\n3,4\\n4,5\\n5,2\\n", false, true);
const sub = gDisc.inducedSubgraph(new Uint32Array([2, 3, 4]));
const lcc = gDisc.largestConnectedComponent();
const lccG = lcc.graph();

const l2Empty = mod.Layout2d.zeros(0);
assert.equal(l2Empty.isEmpty(), true);
const l2 = mod.Layout2d.zeros(4);
assert.equal(l2.numNodes, 4);
assert.equal(l2.isEmpty(), false);
l2.setX(0, 1.0); l2.setY(0, -1.0);
l2.setX(1, 3.5); l2.setY(1, 2.5);
const l2x1 = l2.x(1);
const l2y1 = l2.y(1);
l2.normalizeStd(1.5);
const l2NormCoords = Array.from(l2.copyCoords());
const l2Init = mod.Layout2d.init(5, 42);
assert.equal(l2Init.numNodes, 5);
const l2InitCoords = Array.from(l2Init.copyCoords());

const ndZero = mod.LayoutNd.zeros(3, 4);
assert.equal(ndZero.numNodes, 3);
assert.equal(ndZero.dim, 4);
assert.equal(ndZero.isAllFinite(), true);
assert.equal(ndZero.isEmpty(), false);
const lNd = mod.LayoutNd.fromCoords(3, 4, new Float32Array(spec.nd_coords));
const ndCoords = Array.from(lNd.copyCoords());

const mlHyp = mod.ManifoldLayout2d.hyperbolicPoincare(2);
const mlSph = mod.ManifoldLayout2d.spherical(2);
const mlTor = mod.ManifoldLayout2d.torus2d(2);
assert.equal(mlHyp.manifold, 'hyperbolic');
assert.equal(mlHyp.numNodes, 2);
assert.equal(mlSph.manifold, 'spherical');
assert.equal(mlSph.numNodes, 2);
assert.equal(mlTor.manifold, 'torus');
assert.equal(mlTor.numNodes, 2);
const ml = mod.ManifoldLayout2d.hyperbolicPoincare(2);
ml.setPoint(0, 0.1, 0.2);
ml.setPoint(1, -0.1, 0.05);
const mlx0 = ml.x(0);
const mly0 = ml.y(0);
const td = ml.tangentDelta(0, 1);
ml.applyTangentStep(0, td[0] * 0.1, td[1] * 0.1);
const mlCoords = Array.from(ml.coords());
const mlRand = mod.ManifoldLayout2d.randomInit('hyperbolic', 4, 0.25, 42);
mlRand.centerPoincareCentroid();
const mlRandCent = Array.from(mlRand.poincareCentroid());
const lScaleIn = mod.Layout2d.fromCoords(new Float32Array([10, -5, 20, 5, 30, 15]));
const mlScaled = mod.ManifoldLayout2d.fromLayoutScaled('hyperbolic', lScaleIn, 0.7);
mlScaled.recenterPoincare(0.1, -0.1);
mlScaled.recenterPoincareAt(1);
const mlScaledCoords = Array.from(mlScaled.coords());
const mlRandCoords = Array.from(mlRand.coords());
const geoDists = [
  mod.poincareDistance(0, 0, 0.3, 0.4),
  mod.sphericalDistance(0, 0, 0, Math.PI / 2),
  mod.torusDistance(0.05, 0.05, 0.95, 0.05),
  ...Array.from(mod.wrapTorusPoint(1.25, -0.25)),
  ...Array.from(mod.poincareMobiusAdd(0.2, -0.1, -0.15, 0.25)),
];

const g9Edges = [];
for (let r = 0; r < 3; r++) {
  for (let c = 0; c < 3; c++) {
    const u = r * 3 + c;
    if (c + 1 < 3) g9Edges.push(u, u + 1);
    if (r + 1 < 3) g9Edges.push(u, u + 3);
  }
}
const gGrid = mod.CsrGraph.fromEdgeList(9, new Uint32Array(g9Edges));
const lGrid = mod.Layout2d.fromCoords(new Float32Array(spec.grid_coords));
const delCsr = mod.delaunayToCsr(lGrid);
const lRect = mod.Layout2d.fromCoords(new Float32Array(spec.rect_coords));
const rectOv = Array.from(mod.rectangleOverlaps(lRect, new Float32Array(spec.rect_hw), new Float32Array(spec.rect_hw)));

const gX = mod.CsrGraph.fromEdgeList(4, new Uint32Array([0, 3, 1, 2]));
const lX = mod.Layout2d.fromCoords(new Float32Array(spec.x_coords));
const ce = mod.crossingEdges(gX, lX);
const cet = mod.crossingEdgesTorus(gX, lX);
const cs = mod.crossingSummary(gX, lX);
const gg = Array.from(mod.gabrielGraphEdges(lGrid));
const rngE = Array.from(mod.relativeNeighborhoodEdges(lGrid));
const apsp = mod.allPairsBfs(gGrid, 1.0);
const sssp = Array.from(mod.bfsSingleSource(gGrid, 0, 2.0));
const snsDist = mod.scaleNormalizedStressFromDistances(lGrid, apsp);
const ielUnit = mod.idealEdgeLengthsWithUnit(gGrid, lGrid, 1.5);
const evalRep = mod.evaluateMetrics(gGrid, lGrid, ['sns', 'idealEdgeLengths'], 4, 64, 42, 1.5);
const evalDefRep = mod.evaluateMetrics(gGrid, lGrid, ['sampled_sns', 'sampled_crossing_number']);
const sampledSnsDef = mod.sampledScaleNormalizedStress(gGrid, lGrid);
const sampledCrDef = mod.sampledCrossingNumber(gGrid, lGrid);
const lGridAdj = mod.Layout2d.fromCoords(new Float32Array(spec.grid_coords.map((v, i) => v + (i % 2 === 0 ? 0.05 : -0.03))));
const pdEdges = mod.prismDistortionWithEdges(lGrid, lGridAdj, new Uint32Array(gg));

const w3Comm = new Uint32Array([0, 0, 0, 0, 0, 1, 1, 1, 1]);
const w3Cl = mod.crosslessness(gGrid, lGrid);
const w3Scl = mod.sampledCrosslessness(gGrid, lGrid, 100, 42);
const w3Ma = mod.minimumAngle(gGrid, lGrid);
const w3MaSum = mod.minimumAngleSummary(gGrid, lGrid);
const w3Deg = mod.layoutDegeneracy(gGrid, lGrid);
const w3Nig = mod.adjustedPatchNig(lGrid, w3Comm, 16, 2048, 0.04, 42);
const w3NigMs = Array.from(mod.adjustedPatchNigMultiscale(lGrid, w3Comm, [16, 32], 2048, 0.04, 42));
const w3Rta = mod.landmarkStratifiedRta(gGrid, lGrid, 4, 50, 1, 42);
const w3RtaDet = mod.landmarkStratifiedRtaDetail(gGrid, lGrid, 4, 50, 1, 42);
const w3LmIds = mod.selectMaxminSpPivots(gGrid, 3, 42, 1);
const w3TblFlat = [];
for (let r = 0; r < w3LmIds.length; r++) {
  const rowOff = w3LmIds[r] * 9;
  for (let c = 0; c < 9; c++) {
    const dv = apsp[rowOff + c];
    w3TblFlat.push(Number.isFinite(dv) ? Math.round(dv) : 0xffffffff);
  }
}
const w3RtaTbl = mod.landmarkStratifiedRtaFromLandmarks(lGrid, w3LmIds, new Uint32Array(w3TblFlat), 1000, 42);
const w3Eval = mod.evaluateMetrics(
  gGrid,
  lGrid,
  ['crosslessness', 'sampledCrosslessness', 'minimumAngle', 'minimumAngleA1', 'layoutDegeneracy', 'adjustedPatchNig', 'clusterAreaFaithfulness', 'landmarkStratifiedRta'],
  null,
  100,
  42,
  null,
  w3Comm,
  4,
  50,
  1,
  16,
  2048,
  0.04,
);
const gcafGrid = mod.clusterAreaFaithfulness(lGrid, w3Comm);
const gcafGridDet = mod.clusterAreaFaithfulnessDetail(lGrid, w3Comm);
const imb24Labels = new Uint32Array(Array.from({ length: 24 }, (_, i) => (i < 20 ? 0 : 1)));
const lImb24 = mod.Layout2d.fromCoords(new Float32Array(spec.imb24_coords));
const imb24Edges = [];
for (let i = 0; i < 23; i++) imb24Edges.push(i, i + 1);
const gImb24 = mod.CsrGraph.fromEdgeList(24, new Uint32Array(imb24Edges));
const gcafImbDef = mod.clusterAreaFaithfulness(lImb24, imb24Labels);
const gcafImbCustom = mod.clusterAreaFaithfulness(lImb24, imb24Labels, 64, 0.9, 8.0, 0.2, 2.5, 0.7);
const gcafImbCustomDet = mod.clusterAreaFaithfulnessDetail(lImb24, imb24Labels, 64, 0.9, 8.0, 0.2, 2.5, 0.7);
const gcafImbInfDet = mod.clusterAreaFaithfulnessDetail(lImb24, imb24Labels, null, null, null, null, Number.POSITIVE_INFINITY, null);
const gcafImbG05Det = mod.clusterAreaFaithfulnessDetail(lImb24, imb24Labels, null, null, null, null, null, 0.5);
const gcafImbG20Det = mod.clusterAreaFaithfulnessDetail(lImb24, imb24Labels, null, null, null, null, null, 2.0);
const imb24Nw = new Float64Array(24);
for (let i = 0; i < 24; i++) imb24Nw[i] = i < 20 ? 1.5 : 3.0;
const imb24Cw = new Float64Array([30.0, 12.0]);
const gcafImbNwDet = mod.clusterAreaFaithfulnessDetail(lImb24, imb24Labels, null, null, null, null, null, 2.0, imb24Nw, null);
const gcafImbCwDet = mod.clusterAreaFaithfulnessDetail(lImb24, imb24Labels, null, null, null, null, null, 2.0, null, imb24Cw);
const gcafImbEval = mod.evaluateMetrics(gImb24, lImb24, ['gcaf'], null, null, null, null, imb24Labels, null, null, null, null, null, null, 64, 0.9, 8.0, 0.2, 2.5, 0.7);
const lInt20 = mod.Layout2d.fromCoords(new Float32Array(spec.int20_coords));
const gcafInt20Det = mod.clusterAreaFaithfulnessDetail(lInt20, new Uint32Array(spec.int20_labels));
const lGcafB0 = mod.Layout2d.zeros(0);
const gcafB0 = mod.clusterAreaFaithfulnessDetail(lGcafB0, []);
const lGcafB1 = mod.Layout2d.fromCoords(new Float32Array([1, 2]));
const gcafB1 = mod.clusterAreaFaithfulnessDetail(lGcafB1, [5]);
const lGcafBk1 = mod.Layout2d.fromCoords(new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]));
const gcafBk1 = mod.clusterAreaFaithfulnessDetail(lGcafBk1, [3, 3, 3, 3]);
const lGcafCol = mod.Layout2d.fromCoords(new Float32Array([2, -1, 2, -1, 2, -1, 2, -1, 2, -1, 2, -1]));
const gcafBCol = mod.clusterAreaFaithfulness(lGcafCol, [0, 0, 0, 1, 1, 1]);
const readGcafDet = (d) => {
  const rec = d.toRecord();
  return {
    caf: d.caf,
    purityAdj: d.purityAdj,
    propVis: d.propVis,
    resDensity: d.resDensity,
    resDeconv: d.resDeconv,
    resCombined: d.resCombined,
    spatialCoherence: d.spatialCoherence,
    inFrac: d.inFrac,
    effectiveBoundaryR95: d.effectiveBoundaryR95,
    gEff: d.gEff,
    numClusters: d.numClusters,
    numNodes: d.numNodes,
    clusterIds: Array.from(d.clusterIds()),
    counts: Array.from(d.counts()),
    p: Array.from(d.p()),
    q: Array.from(d.q()),
    purity: Array.from(d.purity()),
    resolution: Array.from(d.resolution()),
    resDensityPerCluster: Array.from(d.resDensityPerCluster()),
    resDeconvPerCluster: Array.from(d.resDeconvPerCluster()),
    cafPerCluster: Array.from(d.cafPerCluster()),
    coherence: Array.from(d.coherence()),
    rec: {
      caf: rec.caf,
      purityAdj: rec.purityAdj,
      propVis: rec.propVis,
      resDensity: rec.resDensity,
      resDeconv: rec.resDeconv,
      resCombined: rec.resCombined,
      spatialCoherence: rec.spatialCoherence,
      inFrac: rec.inFrac,
      effectiveBoundaryR95: rec.effectiveBoundaryR95,
      gEff: rec.gEff,
      numClusters: rec.numClusters,
      numNodes: rec.numNodes,
      clusterIds: Array.from(rec.clusterIds),
      counts: Array.from(rec.counts),
      p: Array.from(rec.p),
      q: Array.from(rec.q),
      purity: Array.from(rec.purity),
      resolution: Array.from(rec.resolution),
      resDensityPerCluster: Array.from(rec.resDensityPerCluster),
      resDeconvPerCluster: Array.from(rec.resDeconvPerCluster),
      cafPerCluster: Array.from(rec.cafPerCluster),
      coherence: Array.from(rec.coherence),
    },
  };
};
const gcafSmallN = {};
for (const c of spec.gcaf_small_n) {
  const nCase = c.labels.length;
  const lCase = mod.Layout2d.fromCoords(new Float32Array(c.coords));
  const cEdges = [];
  for (let i = 0; i + 1 < nCase; i++) cEdges.push(i, i + 1);
  const gCase = mod.CsrGraph.fromEdgeList(nCase, new Uint32Array(cEdges));
  const cLabels = new Uint32Array(c.labels);
  const bR95 = c.boundary_r95 === 'Infinity' ? Infinity : c.boundary_r95;
  const cafVal = mod.clusterAreaFaithfulness(lCase, cLabels, c.g_max, c.sigma_cells, null, null, bR95, c.stevens_gamma);
  const detHandle = mod.clusterAreaFaithfulnessDetail(lCase, cLabels, c.g_max, c.sigma_cells, null, null, bR95, c.stevens_gamma);
  const evalHandle = mod.evaluateMetrics(gCase, lCase, ['gcaf'], null, null, null, null, cLabels, null, null, null, null, null, null, c.g_max, c.sigma_cells, null, null, bR95, c.stevens_gamma);
  const evalRec = evalHandle.toRecord();
  gcafSmallN[c.name] = {
    caf: cafVal,
    det: readGcafDet(detHandle),
    evalCaf: evalHandle.clusterAreaFaithfulness,
    evalRecCaf: evalRec.clusterAreaFaithfulness,
  };
  detHandle.free?.();
  evalHandle.free?.();
  gCase.free?.();
  lCase.free?.();
}
const gDir4 = mod.CsrGraph.fromEdgeList(4, new Uint32Array([0, 1, 0, 3, 1, 3, 3, 0, 3, 1, 3, 2]), undefined, true, true);
const lDir4 = mod.Layout2d.fromCoords(new Float32Array([0, 0, 2, 2, 0, 2, 2, 0]));
const clDir4 = mod.crosslessness(gDir4, lDir4);
const sclDir4 = mod.sampledCrosslessness(gDir4, lDir4, 100, 42);
const gRstar = mod.CsrGraph.fromEdgeList(4, new Uint32Array([0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 3, 0]), undefined, true, true);
const lRstar = mod.Layout2d.fromCoords(new Float32Array([0, 0, 1, 0, 0, 1, -1, 0]));
const clRstar = mod.crosslessness(gRstar, lRstar);

const g16 = mod.CsrGraph.fromEdgeList(16, new Uint32Array(spec.g16_edges));
const cmds = mod.classicalMds2d(g16);
mod.normalizeLayout2d(cmds, 1.0, 42);
const cmdsCoords = Array.from(cmds.coordsView());
const rob = mod.classicalMds2d(g16);
mod.robustNormalizeLayout2d(rob, 1.5, 0.95, 42);
const robCoords = Array.from(rob.coordsView());
const rd = mod.rdMds(g16, 3, 1e-10, 1e-6, 1000, 42);
const rdCoords = Array.from(rd.copyCoords());
const resEmb = mod.computeResistanceEmbedding(g16, 3, 1e-4, 300, 42, 1e-10);
const diffEmb = mod.computeDiffusionEmbedding(g16, 3, 0.5, null, 1e-4, 300, 42);
const repNd = mod.representativeMds(new Float64Array([0, 3, 1, 2, 2, 1, 3, 0]), 4, 2, 2);
const rerank = mod.buildRerankKnn(new Float64Array(spec.diff_coords), 16, spec.diff_dim, 4, 2, 2);
const cappedKnn = mod.buildCapped12hopKnn(g16, 4, 42);
const rowProbs = mod.graphRowNormalizedProbabilities(g16);
const rowProbs12 = mod.graph12hopRowNormalizedProbabilities(g16, 0.5, 4, 0.35, 42);
const sampledSnsStrat = mod.sampledScaleNormalizedStress(g16, cmds, 4, 42);
const symProbs = mod.symmetrizeRowNormalized(3, new Uint32Array([0, 1, 2, 3]), new Uint32Array([1, 2, 0]), new Float64Array([1, 1, 1]));
const prio = mod.buildGumap12hopPriorityCsr(3, new Uint32Array([0]), new Uint32Array([1]), new Float64Array([1]), new Uint32Array([0, 1]), new Uint32Array([2, 2]), new Float64Array([2, 2]), 1.1);
const d1 = new Float64Array([1e-6, 1.0]);
const d2 = new Float64Array([2.0]);
const d3 = new Float64Array([4.0]);
const floorVal = mod.applyRelativeDistanceFloor(d1, d2, d3, 1e-3, 0.1);

const init16 = new Float32Array(spec.init16);
const lTs = mod.Layout2d.fromCoords(init16);
const lTsProg = mod.Layout2d.fromCoords(init16);
const tsOpts = {
  backend: 'fft',
  iterationsStage1: 2,
  iterationsStage2: 2,
  iterationsStage3: 2,
  intervals: 1,
  interpolationOrder: 1,
  minGridDim: 16,
  maxGridDim: 64,
  seed: 42,
};
const tsPartial = mod.TsNetSolver.fromGraphPartialBfs(g16, 8, 6.0, tsOpts);
const tsPartialProg = mod.TsNetSolver.fromGraphPartialBfs(g16, 8, 6.0, tsOpts);
const tsRn = mod.TsNetSolver.fromGraphRowNormalized(g16, { w2: 0.5, max2hopPerNode: 4, degreeBeta: 0.35, ...tsOpts });
const lTsRn = mod.Layout2d.fromCoords(init16);
tsRn.run(lTsRn);
assert.equal(tsPartial.lastFftStats(), undefined);
tsPartial.run(lTs);
const fftStats = tsPartial.lastFftStats();
assert.ok(fftStats && fftStats.totalFft2dCalls > 0);
const progFrames = [];
await indexMod.runProgressive(tsPartialProg, lTsProg, (ctx) => progFrames.push(ctx.iteration), { frameBudgetMs: 10, stepsPerYield: 1 });
assert.deepEqual(Array.from(lTs.coordsView()), Array.from(lTsProg.coordsView()));

const knnObj = mod.KnnGraph.fromCsr(16, new Uint32Array(spec.rerank_indptr), new Uint32Array(spec.rerank_indices), new Float64Array(spec.rerank_dists));
const lTfdpKnn = mod.Layout2d.fromCoords(init16);
const tfdpKnn = mod.TFdpSolver.fromKnn(knnObj, { backend: 'exact', totalIterations: 4, ibfftOrder: 1, ibfftIntervals: 16, ibfftMinGridDim: 32, ibfftMaxGridDim: 128, seed: 42 });
tfdpKnn.run(lTfdpKnn);

const lIb = mod.Layout2d.fromCoords(init16);
const ib = new mod.IbUmapSolver(g16, {
  numLandmarks: 8,
  minDegree: 1,
  jlDim: 8,
  degreeNormalize: true,
  pExp: 0.5,
  epsPsd: 1e-4,
  tCoarse: 32.0,
  backend: 'ibfft',
  nEpochs: 5,
  ibfftOrder: 1,
  ibfftIntervals: 16,
  ibfftMinGridDim: 32,
  ibfftMaxGridDim: 128,
  seed: 42,
});
ib.run(lIb);

const lmDogDef = mod.computeRegularizedDogVaradhanLandmarks(g16);
const sgdDog = mod.SgdSolver.sparseBarrierDog(g16, { tMax: 4, seed: 42, minDegree: 1, jlDim: 8 });
const dogStats = sgdDog.dogStats();
assert.ok(dogStats && dogStats.degree >= 40);
const lSgdDog = mod.Layout2d.fromCoords(init16);
sgdDog.run(lSgdDog);

const gmDog = mod.GuMapSolver.fromGraphDog(g16, { minDegree: 1, jlDim: 8, nEpochs: 4, seed: 42 });

const lFdeb = mod.Layout2d.fromCoords(init16);
const fdeb = new mod.FdebSolver(g16, lFdeb, { cycles: 2, i0: 4 });
fdeb.run();

const hier = mod.louvainCoarsen(g16, { maxCoarseNodes: 4, seed: 42 });
const rhghRes = mod.computePivotedRhghLandmarks(g16, { numLandmarks: 8, minDegree: 1, numProbes: 16, seed: 42 });
const rhghLm = rhghRes.landmarks();
const g16Bb = mod.CsrGraph.fromEdgeList(16, new Uint32Array(spec.g16_edges), new Float32Array(spec.g16_bb_weights));
const bb = mod.coarseBackboneInit(g16Bb, {
  maxCoarseNodes: 4,
  edgeLengthMode: "resistanceCurvatureFlow",
  distanceBackend: "rhgh",
  coarseTMax: 20,
  seed: 42,
});
const bbAligned = mod.Layout2d.fromCoords(bb.fineLayout().copyCoords());
const bbCoarse = bb.coarseLayout();
mod.alignClusteredCentroids(bbAligned, bbCoarse, bb.nodeToCoarse(), 0.5);

const spField = new mod.ScreenedPoissonField2d({
  kernel: "screenedPoisson",
  gridSize: 64,
  paddingFactor: 2,
  sigma: 0.2,
  kappa: 1.0,
  rhoMax: 0.1,
  exactAdjoint: true,
});
const lSpInit = mod.Layout2d.fromCoords(init16);
const spEval = spField.evaluateInBounds(lSpInit, [-2.0, 2.0, -2.0, 2.0]);
const lSpRef = mod.Layout2d.fromCoords(init16);
const spRef = mod.ScreenedPoissonRefiner.fromGraph(g16, {
  gridSize: 64,
  paddingFactor: 2,
  sigma: 0.2,
  kappa: 1.0,
  rhoMax: 0.1,
  iterations: 4,
  anchorWeight: 0.1,
  seed: 42,
});
spRef.run(lSpRef);

const out = {
  disc: { nnz: gDisc.nnz(), isW: gDisc.isWeighted(), deg2: gDisc.degree(2), nbr2: Array.from(gDisc.neighbors(2)), subN: sub.numNodes(), subE: sub.numEdges(), lccN: lccG.numNodes(), lccIds: Array.from(lcc.originalNodeIds()) },
  l2: { x1: l2x1, y1: l2y1, normCoords: l2NormCoords, initCoords: l2InitCoords },
  ndCoords,
  ml: { x0: mlx0, y0: mly0, td: Array.from(td), coords: mlCoords, geoDists, randCoords: mlRandCoords, randCent: mlRandCent, scaledCoords: mlScaledCoords },
  delCsr: { n: delCsr.numNodes(), e: delCsr.numEdges() },
  rectOv,
  ceLen: ce.length,
  ce0: [ce[0].edgeA, ce[0].edgeB],
  cetLen: cet.length,
  cs: { count: cs.count, minAngle: cs.minAngleRad, cosSq: cs.angleCosSqSum },
  gg,
  rngE,
  apsp: Array.from(apsp),
  sssp,
  snsDist,
  ielUnit,
  evalIel: evalRep.idealEdgeLengths,
  evalSns: evalRep.scaleNormalizedStress,
  evalDef: { sns: evalDefRep.sampledScaleNormalizedStress, cr: evalDefRep.sampledCrossingNumber, snsStandalone: sampledSnsDef, crStandalone: sampledCrDef },
  pdEdges: [pdEdges.sigmaDist, pdEdges.sigmaDisp],
  cmdsCoords,
  robCoords,
  rdCoords,
  res: { lams: Array.from(resEmb.lambdas()), coords: Array.from(resEmb.coords()) },
  diff: { tUsed: diffEmb.tUsed, lams: Array.from(diffEmb.lambdas()), coords: Array.from(diffEmb.coords()) },
  repCoords: Array.from(repNd.copyCoords()),
  rerank: { indptr: Array.from(rerank.indptr()), indices: Array.from(rerank.indices()), dists: Array.from(rerank.distances()) },
  cappedKnn: { k2Deg: Array.from(cappedKnn.k2Deg()), indptr: Array.from(cappedKnn.knn().indptr()), indices: Array.from(cappedKnn.knn().indices()), dists: Array.from(cappedKnn.knn().distances()) },
  rowProbs: { indptr: Array.from(rowProbs.indptr()), indices: Array.from(rowProbs.indices()), values: Array.from(rowProbs.values()) },
  rowProbs12: { indptr: Array.from(rowProbs12.indptr()), indices: Array.from(rowProbs12.indices()), values: Array.from(rowProbs12.values()) },
  symProbs: { indptr: Array.from(symProbs.indptr()), indices: Array.from(symProbs.indices()), values: Array.from(symProbs.values()) },
  prio: { indptr: Array.from(prio.indptr()), indices: Array.from(prio.indices()), dists: Array.from(prio.distances()) },
  floor: { d1: Array.from(d1), d2: Array.from(d2), d3: Array.from(d3), val: floorVal },
  tsPartial: { coords: Array.from(lTs.coordsView()), ffts: fftStats.totalFft2dCalls, pFfts: fftStats.particleForwardFfts, progFrames: progFrames.length },
  tsRnCoords: Array.from(lTsRn.coordsView()),
  tfdpKnnCoords: Array.from(lTfdpKnn.coordsView()),
  ib: { a: ib.a, b: ib.b, ab: Array.from(ib.ab), numEdges: ib.numEdges, ffts: ib.totalFft2dCalls, coords: Array.from(lIb.coordsView()) },
  lmDogDef: { table: Array.from(lmDogDef.table()), t1: lmDogDef.t1, t2: lmDogDef.t2, alpha: lmDogDef.alpha },
  dogStats: { degree: dogStats.degree, numLandmarks: dogStats.numLandmarks, t1: dogStats.t1, t2: dogStats.t2, alpha: dogStats.alpha },
  sgdDogCoords: Array.from(lSgdDog.coordsView()),
  gmAb: Array.from(gmDog.ab),
  fdeb: { offsets: Array.from(fdeb.edgeOffsets()), points: Array.from(fdeb.points()) },
  m51: {
    hierClusters: hier.numClusters,
    hierN2c: Array.from(hier.nodeToCoarse()),
    rhghGamma: rhghRes.gamma,
    rhghTable: Array.from(rhghLm.table()),
    bbFine: Array.from(bb.fineLayout().coordsView()),
    bbAligned: Array.from(bbAligned.coordsView()),
  },
  m52: {
    energy: spEval.energy,
    fftCalls: spEval.fft2dCalls,
    fx: Array.from(spEval.forceX()),
    fy: Array.from(spEval.forceY()),
    refCoords: Array.from(lSpRef.coordsView()),
    refLastEnergy: spRef.lastEnergy,
  },
  sampledSnsStrat,
  w3: {
    cl: w3Cl,
    scl: w3Scl,
    ma: w3Ma,
    maSum: w3MaSum.toRecord(),
    deg: w3Deg.toRecord(),
    nig: w3Nig,
    nigMs: w3NigMs,
    rta: w3Rta,
    rtaDet: w3RtaDet.toRecord(),
    rtaTbl: w3RtaTbl.toRecord(),
    eval: w3Eval.toRecord(),
    gcaf: {
      grid: gcafGrid,
      gridDet: readGcafDet(gcafGridDet),
      imbDef: gcafImbDef,
      imbCustom: gcafImbCustom,
      imbCustomDet: readGcafDet(gcafImbCustomDet),
      imbInfDet: readGcafDet(gcafImbInfDet),
      imbG05Det: readGcafDet(gcafImbG05Det),
      imbG20Det: readGcafDet(gcafImbG20Det),
      imbNwDet: readGcafDet(gcafImbNwDet),
      imbCwDet: readGcafDet(gcafImbCwDet),
      imbEval: gcafImbEval.toRecord(),
      int20Det: readGcafDet(gcafInt20Det),
      b0: readGcafDet(gcafB0),
      b1: readGcafDet(gcafB1),
      bk1: readGcafDet(gcafBk1),
      bCol: gcafBCol,
      smallN: gcafSmallN,
    },
    clDir4: clDir4,
    sclDir4: sclDir4,
    clRstar: clRstar,
  },
};

gcafGridDet.free?.(); gcafImbCustomDet.free?.(); gcafImbInfDet.free?.(); gcafImbG05Det.free?.(); gcafImbG20Det.free?.(); gcafImbNwDet.free?.(); gcafImbCwDet.free?.(); gcafImbEval.free?.();
gcafInt20Det.free?.(); gcafB0.free?.(); gcafB1.free?.(); gcafBk1.free?.();
lImb24.free?.(); gImb24.free?.(); lInt20.free?.(); lGcafB0.free?.(); lGcafB1.free?.(); lGcafBk1.free?.(); lGcafCol.free?.();
w3MaSum.free?.(); w3Deg.free?.(); w3RtaDet.free?.(); w3RtaTbl.free?.(); w3Eval.free?.();
gDir4.free?.(); lDir4.free?.(); gRstar.free?.(); lRstar.free?.();

for (const c of ce) c.free?.();
for (const tc of cet) tc.free?.();
cs.free?.(); evalRep.free?.(); evalDefRep.free?.(); pdEdges.free?.();
fdeb.free?.(); lFdeb.free?.(); gmDog.free?.(); lmDogDef.free?.(); lSgdDog.free?.(); dogStats.free?.(); sgdDog.free?.();
bbAligned.free?.(); bbCoarse.free?.(); bb.free?.(); g16Bb.free?.(); rhghLm.free?.(); rhghRes.free?.(); hier.free?.();
ib.free?.(); lIb.free?.(); tfdpKnn.free?.(); lTfdpKnn.free?.(); knnObj.free?.();
fftStats.free?.(); tsPartial.free?.(); tsPartialProg.free?.(); lTs.free?.(); lTsProg.free?.();
prio.free?.(); symProbs.free?.(); rowProbs.free?.(); rerank.free?.(); repNd.free?.();
diffEmb.free?.(); resEmb.free?.(); rd.free?.(); cmds.free?.(); g16.free?.();
lGridAdj.free?.(); lX.free?.(); gX.free?.(); lRect.free?.(); delCsr.free?.(); lGrid.free?.(); gGrid.free?.();
ml.free?.(); mlHyp.free?.(); mlSph.free?.(); mlTor.free?.(); lNd.free?.(); ndZero.free?.(); l2Init.free?.(); l2.free?.(); l2Empty.free?.();
lccG.free?.(); lcc.free?.(); sub.free?.(); gDisc.free?.();

console.log(JSON.stringify(out));
"""
    raw_out = subprocess.check_output(
        ["node", "--input-type=module", "-e", node_script, node_url, index_url, wasm_path, json.dumps(spec_payload)],
        text=True,
    )
    ws = json.loads(raw_out.strip().splitlines()[-1])

    max_f32_diff = 0.0
    max_f64_diff = 0.0

    def check_close(label: str, a: Any, b: Any, tol: float, is_f32: bool = True) -> None:
        nonlocal max_f32_diff, max_f64_diff
        arr_a = np.asarray(a, dtype=np.float64).ravel()
        arr_b = np.asarray(b, dtype=np.float64).ravel()
        assert arr_a.shape == arr_b.shape, f"{label}: shape mismatch {arr_a.shape} vs {arr_b.shape}"
        diff = float(np.max(np.abs(arr_a - arr_b))) if arr_a.size else 0.0
        if is_f32:
            max_f32_diff = max(max_f32_diff, diff)
        else:
            max_f64_diff = max(max_f64_diff, diff)
        assert diff < tol, f"{label}: diff={diff} >= {tol}"

    assert ws["disc"]["nnz"] == g_disc.nnz
    assert ws["disc"]["isW"] == g_disc.is_weighted
    assert ws["disc"]["deg2"] == g_disc.degree(2)
    assert ws["disc"]["nbr2"] == list(g_disc.neighbors(2))
    assert ws["disc"]["subN"] == sub_py.num_nodes and ws["disc"]["subE"] == sub_py.num_edges
    assert ws["disc"]["lccN"] == lcc_py.num_nodes and ws["disc"]["lccIds"] == list(lcc_ids_py)
    assert karate_py.num_nodes == 34 and karate_py.num_edges >= 78

    check_close("Layout2d.x/y", [l2_x1, l2_y1], [ws["l2"]["x1"], ws["l2"]["y1"]], 1e-6)
    check_close("Layout2d.normalize_std", l2_norm_coords, ws["l2"]["normCoords"], 1e-5)
    check_close("Layout2d.init", l2_init_coords_py, ws["l2"]["initCoords"], 1e-6)
    check_close("LayoutNd.from_coords", nd_py.coords(copy=True), ws["ndCoords"], 1e-6)

    check_close("ManifoldLayout2d.x/y", [ml_x0, ml_y0], [ws["ml"]["x0"], ws["ml"]["y0"]], 1e-6)
    check_close("ManifoldLayout2d.tangent_delta", [td_vx, td_vy], ws["ml"]["td"], 1e-5)
    check_close("ManifoldLayout2d.apply_tangent_step", ml_coords_after, ws["ml"]["coords"], 1e-5)
    check_close("ManifoldLayout2d.random_init+center_poincare_centroid", ml_rand_py.coords(), ws["ml"]["randCoords"], 1e-5)
    check_close("ManifoldLayout2d.poincare_centroid", ml_rand_cent_py, ws["ml"]["randCent"], 1e-5)
    check_close("ManifoldLayout2d.from_layout_scaled+recenter", ml_scaled_py.coords(), ws["ml"]["scaledCoords"], 1e-5)
    check_close("geodesic_distances", geo_dists_py, ws["ml"]["geoDists"], 1e-12, is_f32=False)

    assert ws["delCsr"]["n"] == del_csr_py.num_nodes and ws["delCsr"]["e"] == del_csr_py.num_edges
    assert list(rect_ov_py.ravel()) == ws["rectOv"]
    assert len(ce_py) == ws["ceLen"] == 1
    assert [ce_py[0]["edge_a"], ce_py[0]["edge_b"]] == ws["ce0"]
    assert len(cet_py) == ws["cetLen"]
    assert cs_py["count"] == ws["cs"]["count"]
    check_close("crossing_summary/min_angle", [cs_py["min_angle_rad"]], [ws["cs"]["minAngle"]], 1e-5)
    check_close("crossing_summary/cos_sq", [cs_py["angle_cos_sq_sum"]], [ws["cs"]["cosSq"]], 1e-5)
    assert list(gg_py.ravel()) == ws["gg"]
    assert list(rng_py.ravel()) == ws["rngE"]
    check_close("all_pairs_bfs", apsp_py, ws["apsp"], 1e-12, is_f32=False)
    check_close("bfs_single_source", sssp_py, ws["sssp"], 1e-12, is_f32=False)
    check_close("scale_normalized_stress_from_distances", [sns_dist_py], [ws["snsDist"]], 1e-5)
    check_close("ideal_edge_lengths_with_unit", [iel_unit_py], [ws["ielUnit"]], 1e-5)
    check_close("evaluate_metrics/ideal_edge_lengths", [eval_iel_py["ideal_edge_lengths"]], [ws["evalIel"]], 1e-5)
    check_close("evaluate_metrics/sns", [eval_iel_py["scale_normalized_stress"]], [ws["evalSns"]], 1e-5)
    check_close("evaluate_metrics/default_sampled_sns", [eval_default_py["sampled_scale_normalized_stress"]], [ws["evalDef"]["sns"]], 1e-12, is_f32=False)
    check_close("evaluate_metrics/default_sampled_crossing", [eval_default_py["sampled_crossing_number"]], [ws["evalDef"]["cr"]], 1e-12, is_f32=False)
    check_close("sampled_scale_normalized_stress/default", [sampled_sns_default_py], [ws["evalDef"]["snsStandalone"]], 1e-12, is_f32=False)
    check_close("sampled_crossing_number/default", [sampled_cr_default_py], [ws["evalDef"]["crStandalone"]], 1e-12, is_f32=False)
    check_close("prism_distortion_with_edges", pd_edges_py, ws["pdEdges"], 1e-5)
    check_close("sampled_scale_normalized_stress/degree_stratified_k4", [sampled_sns_strat_py], [ws["sampledSnsStrat"]], 1e-6)
    check_close("crosslessness", [cl_py], [ws["w3"]["cl"]], 1e-12, is_f32=False)
    check_close("sampled_crosslessness", [scl_py], [ws["w3"]["scl"]], 1e-12, is_f32=False)
    check_close("minimum_angle", [ma_py], [ws["w3"]["ma"]], 1e-12, is_f32=False)
    check_close("minimum_angle_summary", [ma_sum_py["a1"], ma_sum_py["a2"]], [ws["w3"]["maSum"]["a1"], ws["w3"]["maSum"]["a2"]], 1e-12, is_f32=False)
    assert deg_py["num_nodes"] == ws["w3"]["deg"]["numNodes"]
    assert deg_py["num_edges"] == ws["w3"]["deg"]["numEdges"]
    assert deg_py["duplicate_nodes_f32"] == ws["w3"]["deg"]["duplicateNodesF32"]
    assert deg_py["duplicate_nodes_f64"] == ws["w3"]["deg"]["duplicateNodesF64"]
    assert deg_py["zero_length_edges"] == ws["w3"]["deg"]["zeroLengthEdges"]
    assert deg_py["is_bbox_degenerate"] == ws["w3"]["deg"]["isBboxDegenerate"]
    check_close("layout_degeneracy/ratios_bbox", [deg_py["duplicate_ratio_f32"], deg_py["duplicate_ratio_f64"], deg_py["zero_length_edge_ratio"], deg_py["bbox_width"], deg_py["bbox_height"]], [ws["w3"]["deg"]["duplicateRatioF32"], ws["w3"]["deg"]["duplicateRatioF64"], ws["w3"]["deg"]["zeroLengthEdgeRatio"], ws["w3"]["deg"]["bboxWidth"], ws["w3"]["deg"]["bboxHeight"]], 1e-12, is_f32=False)
    check_close("adjusted_patch_nig", [nig_py], [ws["w3"]["nig"]], 1e-12, is_f32=False)
    check_close("adjusted_patch_nig_multiscale", [nig_ms_py["patch_nig_r16"], nig_ms_py["patch_nig_r32"]], ws["w3"]["nigMs"], 1e-12, is_f32=False)
    check_close("landmark_stratified_rta", [rta_py], [ws["w3"]["rta"]], 1e-12, is_f32=False)
    assert rta_det_py["num_landmarks"] == ws["w3"]["rtaDet"]["numLandmarks"]
    assert rta_det_py["num_triplets"] == ws["w3"]["rtaDet"]["numTriplets"]
    assert rta_det_py["num_strata"] == ws["w3"]["rtaDet"]["numStrata"]
    assert rta_det_py["is_exact"] == ws["w3"]["rtaDet"]["isExact"]
    check_close("landmark_stratified_rta_detail", [rta_det_py["rta"]], [ws["w3"]["rtaDet"]["rta"]], 1e-12, is_f32=False)
    assert rta_tbl_py["num_triplets"] == ws["w3"]["rtaTbl"]["numTriplets"]
    assert rta_tbl_py["is_exact"] == ws["w3"]["rtaTbl"]["isExact"]
    check_close("landmark_stratified_rta_from_landmarks", [rta_tbl_py["rta"]], [ws["w3"]["rtaTbl"]["rta"]], 1e-12, is_f32=False)
    check_close("evaluate_metrics/w3", [w3_eval_py["crosslessness"], w3_eval_py["sampled_crosslessness"], w3_eval_py["minimum_angle"], w3_eval_py["minimum_angle_a1"], w3_eval_py["layout_degeneracy"], w3_eval_py["adjusted_patch_nig"], w3_eval_py["cluster_area_faithfulness"], w3_eval_py["landmark_stratified_rta"]], [ws["w3"]["eval"]["crosslessness"], ws["w3"]["eval"]["sampledCrosslessness"], ws["w3"]["eval"]["minimumAngle"], ws["w3"]["eval"]["minimumAngleA1"], ws["w3"]["eval"]["layoutDegeneracy"], ws["w3"]["eval"]["adjustedPatchNig"], ws["w3"]["eval"]["clusterAreaFaithfulness"], ws["w3"]["eval"]["landmarkStratifiedRta"]], 1e-12, is_f32=False)
    def check_gcaf_detail(label: str, py_det: dict[str, Any], ws_det: dict[str, Any]) -> None:
        scalar_map = (
            ("caf", "caf"),
            ("purity_adj", "purityAdj"),
            ("prop_vis", "propVis"),
            ("res_density", "resDensity"),
            ("res_deconv", "resDeconv"),
            ("res_combined", "resCombined"),
            ("spatial_coherence", "spatialCoherence"),
            ("in_frac", "inFrac"),
            ("effective_boundary_r95", "effectiveBoundaryR95"),
        )
        vec_map = (
            ("p", "p"),
            ("q", "q"),
            ("purity", "purity"),
            ("resolution", "resolution"),
            ("res_density_per_cluster", "resDensityPerCluster"),
            ("res_deconv_per_cluster", "resDeconvPerCluster"),
            ("caf_per_cluster", "cafPerCluster"),
            ("coherence", "coherence"),
        )
        for src_obj in (ws_det, ws_det["rec"]):
            assert py_det["g_eff"] == src_obj["gEff"], f"{label}: g_eff mismatch"
            assert py_det["num_clusters"] == src_obj["numClusters"], f"{label}: num_clusters mismatch"
            assert py_det["num_nodes"] == src_obj["numNodes"], f"{label}: num_nodes mismatch"
            assert list(py_det["cluster_ids"]) == src_obj["clusterIds"], f"{label}: cluster_ids mismatch"
            assert list(py_det["counts"]) == src_obj["counts"], f"{label}: counts mismatch"
            for py_k, ws_k in scalar_map:
                check_close(f"{label}/{py_k}", [py_det[py_k]], [src_obj[ws_k]], 1e-12, is_f32=False)
            for py_k, ws_k in vec_map:
                check_close(f"{label}/{py_k}", py_det[py_k], src_obj[ws_k], 1e-12, is_f32=False)

    gcaf_ws = ws["w3"]["gcaf"]
    check_close("gcaf/grid", [gcaf_grid_py], [gcaf_ws["grid"]], 1e-12, is_f32=False)
    check_gcaf_detail("gcaf/grid_det", gcaf_grid_det_py, gcaf_ws["gridDet"])
    check_close("gcaf/imb_def", [gcaf_imb_def_py], [gcaf_ws["imbDef"]], 1e-12, is_f32=False)
    check_close("gcaf/imb_custom", [gcaf_imb_custom_py, gcaf_imb_eval_py["cluster_area_faithfulness"]], [gcaf_ws["imbCustom"], gcaf_ws["imbEval"]["clusterAreaFaithfulness"]], 1e-12, is_f32=False)
    check_gcaf_detail("gcaf/imb_custom_det", gcaf_imb_custom_det_py, gcaf_ws["imbCustomDet"])
    check_gcaf_detail("gcaf/imb_inf_det", gcaf_imb_inf_det_py, gcaf_ws["imbInfDet"])
    check_gcaf_detail("gcaf/imb_g05_det", gcaf_imb_g05_det_py, gcaf_ws["imbG05Det"])
    check_gcaf_detail("gcaf/imb_g20_det", gcaf_imb_g20_det_py, gcaf_ws["imbG20Det"])
    check_gcaf_detail("gcaf/imb_nw_det", gcaf_imb_nw_det_py, gcaf_ws["imbNwDet"])
    check_gcaf_detail("gcaf/imb_cw_det", gcaf_imb_cw_det_py, gcaf_ws["imbCwDet"])
    check_gcaf_detail("gcaf/imb_nw_vs_cw", gcaf_imb_nw_det_py, gcaf_ws["imbCwDet"])
    check_gcaf_detail("gcaf/int20_f32", gcaf_int20_f32_py, gcaf_ws["int20Det"])
    check_gcaf_detail("gcaf/int20_f64", gcaf_int20_f64_py, gcaf_ws["int20Det"])
    check_gcaf_detail("gcaf/b0", gcaf_b0_py, gcaf_ws["b0"])
    check_gcaf_detail("gcaf/b1", gcaf_b1_py, gcaf_ws["b1"])
    check_gcaf_detail("gcaf/bk1", gcaf_bk1_py, gcaf_ws["bk1"])
    check_close("gcaf/bcol", [gcaf_bcol_py], [gcaf_ws["bCol"]], 1e-12, is_f32=False)
    for case in gcaf_small_n_cases:
        cname = case["name"]
        py_c = gcaf_small_n_py[cname]
        ws_c = gcaf_ws["smallN"][cname]
        check_close(
            f"gcaf/small_n/{cname}/scalar_and_eval",
            [py_c["caf"], py_c["eval"]["cluster_area_faithfulness"], py_c["eval"]["cluster_area_faithfulness"]],
            [ws_c["caf"], ws_c["evalCaf"], ws_c["evalRecCaf"]],
            1e-12,
            is_f32=False,
        )
        check_gcaf_detail(f"gcaf/small_n/{cname}/det", py_c["det"], ws_c["det"])
    check_close("crosslessness/dir4_reciprocal_regression", [cl_dir4_py, scl_dir4_py, cl_rstar_py], [ws["w3"]["clDir4"], ws["w3"]["sclDir4"], ws["w3"]["clRstar"]], 1e-12, is_f32=False)

    check_close("classical_mds_2d", cmds_py.coords(copy=True), ws["cmdsCoords"], 1e-5)
    check_close("robust_normalize_layout_2d", rob_py.coords(copy=True), ws["robCoords"], 1e-5)
    check_close("rd_mds", rd_py.coords(copy=True), ws["rdCoords"], 1e-4)
    check_close("compute_resistance_embedding/lambdas", res_lams_py, ws["res"]["lams"], 1e-10, is_f32=False)
    check_close("compute_resistance_embedding/coords", res_coords_py, ws["res"]["coords"], 1e-10, is_f32=False)
    check_close("compute_diffusion_embedding/t_used", [diff_t_py], [ws["diff"]["tUsed"]], 1e-12, is_f32=False)
    check_close("compute_diffusion_embedding/lambdas", diff_lams_py, ws["diff"]["lams"], 1e-10, is_f32=False)
    check_close("compute_diffusion_embedding/coords", diff_coords_py, ws["diff"]["coords"], 1e-10, is_f32=False)
    check_close("representative_mds", rep_py.coords(copy=True), ws["repCoords"], 1e-5)

    rr_ptr = list(rerank_py.indptr())
    assert rr_ptr == ws["rerank"]["indptr"]
    for row_i in range(16):
        s, e = rr_ptr[row_i], rr_ptr[row_i + 1]
        py_pairs = sorted(zip(list(rerank_py.indices())[s:e], list(rerank_py.distances())[s:e]))
        ws_pairs = sorted(zip(ws["rerank"]["indices"][s:e], ws["rerank"]["dists"][s:e]))
        assert [c for c, _ in py_pairs] == [c for c, _ in ws_pairs], f"build_rerank_knn/row_{row_i} mismatch"
        check_close(f"build_rerank_knn/row_{row_i}_dists", [d for _, d in py_pairs], [d for _, d in ws_pairs], 1e-12, is_f32=False)

    assert list(row_probs_py.indptr()) == ws["rowProbs"]["indptr"]
    assert list(row_probs_py.indices()) == ws["rowProbs"]["indices"]
    check_close("graph_row_normalized_probabilities", row_probs_py.values(), ws["rowProbs"]["values"], 1e-12, is_f32=False)
    assert list(capped_k2_py) == ws["cappedKnn"]["k2Deg"]
    assert list(capped_knn_py.indptr()) == ws["cappedKnn"]["indptr"]
    assert list(capped_knn_py.indices()) == ws["cappedKnn"]["indices"]
    check_close("build_capped_12hop_knn", capped_knn_py.distances(), ws["cappedKnn"]["dists"], 1e-12, is_f32=False)
    assert list(row_probs12_py.indptr()) == ws["rowProbs12"]["indptr"]
    assert list(row_probs12_py.indices()) == ws["rowProbs12"]["indices"]
    check_close("graph_12hop_row_normalized_probabilities", row_probs12_py.values(), ws["rowProbs12"]["values"], 1e-12, is_f32=False)
    assert list(sym_probs_py.indptr()) == ws["symProbs"]["indptr"]
    assert list(sym_probs_py.indices()) == ws["symProbs"]["indices"]
    check_close("symmetrize_row_normalized", sym_probs_py.values(), ws["symProbs"]["values"], 1e-12, is_f32=False)
    assert list(prio_py.indptr()) == ws["prio"]["indptr"]
    assert list(prio_py.indices()) == ws["prio"]["indices"]
    check_close("build_gumap_12hop_priority_csr", prio_py.distances(), ws["prio"]["dists"], 1e-12, is_f32=False)
    check_close("apply_relative_distance_floor/d1", f1_py, ws["floor"]["d1"], 1e-12, is_f32=False)
    check_close("apply_relative_distance_floor/d2", f2_py, ws["floor"]["d2"], 1e-12, is_f32=False)
    check_close("apply_relative_distance_floor/d3", f3_py, ws["floor"]["d3"], 1e-12, is_f32=False)
    check_close("apply_relative_distance_floor/val", [floor_py], [ws["floor"]["val"]], 1e-12, is_f32=False)

    assert ws["tsPartial"]["progFrames"] >= 1
    assert ts_fft_stats_py["total_fft_2d_calls"] == ws["tsPartial"]["ffts"]
    assert ts_fft_stats_py["particle_forward_ffts"] == ws["tsPartial"]["pFfts"]
    check_close("TsNetSolver.from_graph_partial_bfs", l_ts_py.coords(copy=True), ws["tsPartial"]["coords"], 1e-4)
    check_close("TsNetSolver.from_graph_row_normalized(degree_beta=0.35)", l_ts_rn_py.coords(copy=True), ws["tsRnCoords"], 1e-4)
    check_close("TFdpSolver.from_knn(KnnGraph)", l_tfdp_knn_py.coords(copy=True), ws["tfdpKnnCoords"], 1e-4)
    assert ib_py.num_edges == ws["ib"]["numEdges"]
    assert ib_py.total_fft_2d_calls == ws["ib"]["ffts"]
    check_close("IbUmapSolver.ab", ib_py.ab, ws["ib"]["ab"], 1e-12, is_f32=False)
    check_close("IbUmapSolver.coords", l_ib_py.coords(copy=True), ws["ib"]["coords"], 1e-4)
    check_close("compute_regularized_dog_varadhan_landmarks/default_table", lm_dog_default_py.table(), ws["lmDogDef"]["table"], 1e-12, is_f32=False)
    check_close("compute_regularized_dog_varadhan_landmarks/default_scalars", [lm_dog_default_py.t1, lm_dog_default_py.t2, lm_dog_default_py.alpha], [ws["lmDogDef"]["t1"], ws["lmDogDef"]["t2"], ws["lmDogDef"]["alpha"]], 1e-12, is_f32=False)
    assert dog_stats_py["degree"] == ws["dogStats"]["degree"] >= 40
    assert dog_stats_py["n_landmarks"] == ws["dogStats"]["numLandmarks"]
    check_close("SgdSolver.dog_stats", [dog_stats_py["t1"], dog_stats_py["t2"], dog_stats_py["alpha"]], [ws["dogStats"]["t1"], ws["dogStats"]["t2"], ws["dogStats"]["alpha"]], 1e-12, is_f32=False)
    check_close("SgdSolver.sparse_barrier_dog/default_coords", l_sgd_dog_py.coords(copy=True), ws["sgdDogCoords"], 1e-4)
    check_close("GuMapSolver.from_graph_dog/default_ab", gm_dog_py.ab, ws["gmAb"], 1e-12, is_f32=False)
    assert list(fdeb_py.edge_offsets()) == ws["fdeb"]["offsets"]
    check_close("FdebSolver.points", fdeb_py.points(), ws["fdeb"]["points"], 1e-4)
    assert hier_py["num_clusters"] == ws["m51"]["hierClusters"]
    assert list(hier_py["node_to_coarse"]) == ws["m51"]["hierN2c"]
    check_close("compute_pivoted_rhgh_landmarks/gamma", [rhgh_st_py["gamma"]], [ws["m51"]["rhghGamma"]], 1e-12, is_f32=False)
    check_close("compute_pivoted_rhgh_landmarks/table", rhgh_lm_py.table(), ws["m51"]["rhghTable"], 1e-12, is_f32=False)
    check_close("coarse_backbone_init/fine_layout", bb_py["fine_layout"].coords(copy=True), ws["m51"]["bbFine"], 1e-4)
    check_close("align_clustered_centroids", bb_aligned_py.coords(copy=True), ws["m51"]["bbAligned"], 1e-4)
    assert sp_eval_py["fft_2d_calls"] == ws["m52"]["fftCalls"]
    check_close("ScreenedPoissonField2d.energy", [sp_eval_py["energy"]], [ws["m52"]["energy"]], 1e-10, is_f32=False)
    check_close("ScreenedPoissonField2d.force_x", sp_eval_py["force_x"], ws["m52"]["fx"], 1e-5)
    check_close("ScreenedPoissonField2d.force_y", sp_eval_py["force_y"], ws["m52"]["fy"], 1e-5)
    check_close("ScreenedPoissonRefiner.run", l_sp_ref_py.coords(copy=True), ws["m52"]["refCoords"], 1e-4)
    check_close("ScreenedPoissonRefiner.last_energy", [sp_ref_py.last_energy], [ws["m52"]["refLastEnergy"]], 1e-10, is_f32=False)

    return [{
        "case": "cross_target_m5_py_vs_wasm",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "max_f32_diff": round(max_f32_diff, 12),
        "max_f64_diff": round(max_f64_diff, 12),
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
            results.extend(run_cross_target_m2_parity(eg_mod, args.wasm_pkg))
            results.extend(run_cross_target_m3_parity(eg_mod, args.wasm_pkg))
            results.extend(run_cross_target_m4_parity(eg_mod, args.wasm_pkg))
            results.extend(run_cross_target_m5_parity(eg_mod, args.wasm_pkg))

    print(f"{'CASE':<40} {'STATUS':<24} {'ELAPSED(s)':>10}")
    print("-" * 78)
    for row in results:
        print(f"{row['case']:<40} {row['status']:<24} {row['elapsed_sec']:>10.4f}")

    if args.json is not None:
        args.json.write_text(json.dumps({"results": results}, indent=2) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
