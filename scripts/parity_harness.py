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

    return results


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
    ]
    for label, fn in m2_bad_cases:
        _assert_non_panic_error(label, fn)

    results.append({
        "case": "v6_m2_adversarial_non_panic",
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
    pure_knn, k2_deg, pure_stats = eg.build_pure_2hop_knn(g)
    sizes_2hop = eg.compute_2hop_neighborhood_sizes(g)
    node_perps = eg.compute_nodewise_2hop_perplexities(g)
    med_perp = float(eg.compute_median_2hop_perplexity(g))
    hybrid_knn, hybrid_stats = eg.build_hybrid_adaptive_knn(g, jl_emb, node_perps)
    bfs_knn = eg.partial_bfs_knn(g, k=6, seed=42)
    probs = eg.compute_knn_probabilities(hybrid_knn, node_perps)
    fuzzy = eg.compute_fuzzy_simplicial_set(hybrid_knn)
    ab = eg.find_ab_params(0.1, 1.0)
    ht_gamma = float(eg.compute_horvitz_thompson_gamma(1.0, 1600, 16))
    lm = eg.compute_regularized_dog_varadhan_landmarks(
        g, degree=25, num_landmarks=16, seed=42, degree_normalize=True, p_exp=0.5, eps_psd=1e-4, t_coarse=48.0
    )
    alt_u = [0, 63, 64, 69]
    alt_v = [63, 68, 69, 0]
    alt_d = lm.alt_distances(alt_u, alt_v)

    node_url = node_entry.resolve().as_uri() if node_entry.exists() else index_entry.resolve().as_uri()
    wasm_path = (wasm_pkg / "egraph_wasm_bg.wasm").resolve().as_posix()
    spec_payload = {
        "n": n,
        "edges": [int(v) for edge in edges for v in edge],
        "heat_in": [float(v) for v in heat_in.ravel()],
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
const fuzzy = mod.computeFuzzySimplicialSet(hybKnn);
const ab = Array.from(mod.findAbParams(0.1, 1.0));
const htGamma = mod.computeHorvitzThompsonGamma(1.0, 1600, 16);
const lm = mod.computeRegularizedDogVaradhanLandmarks(g, 25, 16, 42, true, 0.5, 1e-4, 48.0);
const altD = Array.from(lm.altDistances(new Uint32Array(spec.alt_u), new Uint32Array(spec.alt_v)));

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
};

lm.free?.();
fuzzy.free?.();
probs.free?.();
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

    return [{
        "case": "cross_target_m2_py_vs_wasm",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "max_f64_diff": round(max_f64_diff, 15),
        "max_f32_diff": round(max_f32_diff, 12),
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

    print(f"{'CASE':<40} {'STATUS':<24} {'ELAPSED(s)':>10}")
    print("-" * 78)
    for row in results:
        print(f"{row['case']:<40} {row['status']:<24} {row['elapsed_sec']:>10.4f}")

    if args.json is not None:
        args.json.write_text(json.dumps({"results": results}, indent=2) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
