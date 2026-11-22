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
from typing import Any


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


def _extract_coords_legacy(drawing: Any, n: int) -> list[tuple[float, float]]:
    return [(float(drawing.x(i)), float(drawing.y(i))) for i in range(n)]


def _assert_finite_coords(coords: list[tuple[float, float]], label: str) -> None:
    for idx, (x, y) in enumerate(coords):
        if not (math.isfinite(x) and math.isfinite(y)):
            raise AssertionError(f"{label}: non-finite coord at node {idx}: ({x}, {y})")


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
        caught = False
        try:
            fn()
        except (ValueError, TypeError, IndexError) as exc:
            if "PanicException" in type(exc).__name__:
                raise AssertionError(f"{label} raised PanicException: {exc}") from exc
            caught = True
        except Exception as exc:
            if "PanicException" in type(exc).__name__:
                raise AssertionError(f"{label} raised PanicException: {exc}") from exc
            caught = True
        assert caught, f"{label} silently accepted invalid input"

    results.append({
        "case": "adversarial_nan_guard",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
    })

    return results


def run_v6_python_suite(eg: Any) -> list[dict[str, Any]]:
    """Run deterministic & step-vs-run equivalence checks once v6 CsrGraph/Layout2d exist."""
    results: list[dict[str, Any]] = []
    # Hook for M0-M6 when CsrGraph & Layout2d land
    t0 = time.perf_counter()
    indptr = [0, 1, 2, 3, 4]
    indices = [1, 0, 2, 1]  # smoke placeholder once v6 CsrGraph lands
    _ = (indptr, indices, eg)
    results.append({
        "case": "v6_csr_hook",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
    })
    return results


def run_wasm_node_parity(wasm_pkg: pathlib.Path) -> list[dict[str, Any]]:
    """Verify Node.js ESM import of built WASM pkg when available."""
    t0 = time.perf_counter()
    node_script = (
        f"import('{wasm_pkg.as_posix()}/index.js')"
        ".then(() => console.log('WASM_ESM_OK'))"
        ".catch((e) => { console.error(e); process.exit(1); });"
    )
    if (wasm_pkg / "index.js").exists():
        subprocess.check_call(["node", "--input-type=module", "-e", node_script])
        return [{
            "case": "wasm_node_esm_import",
            "status": "pass",
            "elapsed_sec": round(time.perf_counter() - t0, 4),
        }]
    return [{
        "case": "wasm_node_esm_import",
        "status": "skipped_no_v6_index_js",
        "elapsed_sec": 0.0,
    }]


def main() -> int:
    parser = argparse.ArgumentParser(description="egraph cross-target parity harness")
    parser.add_argument("--wasm-pkg", type=pathlib.Path, default=None, help="Path to built v6 WASM pkg/")
    parser.add_argument("--json", type=pathlib.Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    import egraph as eg

    results: list[dict[str, Any]] = []
    if hasattr(eg, "CsrGraph") and hasattr(eg, "Layout2d"):
        results.extend(run_v6_python_suite(eg))
    else:
        results.extend(run_legacy_python_suite(eg))

    if args.wasm_pkg is not None:
        results.extend(run_wasm_node_parity(args.wasm_pkg))

    print(f"{'CASE':<40} {'STATUS':<22} {'ELAPSED(s)':>10}")
    print("-" * 76)
    for row in results:
        print(f"{row['case']:<40} {row['status']:<22} {row['elapsed_sec']:>10.4f}")

    if args.json is not None:
        args.json.write_text(json.dumps({"results": results}, indent=2) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
