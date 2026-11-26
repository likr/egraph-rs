"""Cross-target Python <-> WASM parity checks for geometry and spatial metrics."""

from __future__ import annotations

import pathlib
import time
from typing import Any

from parity.common import (
    make_adjusted_coords,
    make_fixture_coords,
    make_fixtures,
    resolve_wasm_entries,
    run_node_json,
)


def run_cross_target_metrics_spatial_parity(
    eg: Any, wasm_pkg: pathlib.Path
) -> list[dict[str, Any]]:
    """Verify Python float32 <-> WASM f32 metric numerical parity (< 1e-4) across canonical fixtures."""
    import numpy as np

    if not hasattr(eg, "scale_normalized_stress"):
        return []
    if resolve_wasm_entries(wasm_pkg) is None:
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

    node_body = """
if (typeof mod.scaleNormalizedStress !== 'function') {
  console.log(JSON.stringify({ metrics_available: false, cases: {} }));
  process.exit(0);
}

const out = {};
for (const c of spec) {
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

console.log(JSON.stringify({ metrics_available: true, cases: out }));
"""
    wasm_payload = run_node_json(wasm_pkg, node_body, payload)
    if not wasm_payload.get("metrics_available", False):
        return [{
            "case": "cross_target_geometry_spatial_py_vs_wasm",
            "status": "skipped_no_wasm_geometry_spatial",
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
        "case": "cross_target_geometry_spatial_py_vs_wasm",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "max_diff": round(max_diff, 12),
    }]
