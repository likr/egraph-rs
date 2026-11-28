"""Cross-target Python <-> WASM parity checks for PRISM, constraints, layering, and FDEB."""

from __future__ import annotations

import pathlib
import time
from typing import Any

from parity.common import (
    check_array_close,
    check_f64_vs_f32_rel,
    make_asymmetric_coords,
    make_grid_edges,
    resolve_wasm_entries,
    run_node_json,
)


def run_cross_target_overlap_layering_parity(
    eg: Any, wasm_pkg: pathlib.Path
) -> list[dict[str, Any]]:
    """Verify Python float32/float64 <-> WASM f32 parity across PRISM, projections, layering, and FDEB."""
    import numpy as np

    if not hasattr(eg, "PrismOverlapRemoval"):
        return []
    if resolve_wasm_entries(wasm_pkg) is None:
        return []

    t0 = time.perf_counter()

    g16_edges = make_grid_edges(4, 4)
    g16 = eg.CsrGraph.from_edges(16, g16_edges, directed=False, sanitize=True)
    init16_f32 = np.array(make_asymmetric_coords(16, scale=0.8), dtype=np.float32)

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

    def run_py_postproc(dtype_name: str) -> dict[str, Any]:
        cp = prism_init_f32 if dtype_name == "float32" else prism_init_f64
        np_dt = np.float32 if dtype_name == "float32" else np.float64

        out: dict[str, Any] = {}
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

        arr_1d = c1d_init_f32.astype(np_dt)
        assert eg.project_1d(
            arr_1d,
            [
                eg.SeparationConstraint(0, 1, 1.5),
                eg.SeparationConstraint(1, 2, 2.0),
                eg.SeparationConstraint(2, 3, 1.0),
            ],
        ) is None
        out["proj_1d"] = arr_1d

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
        poly_fd = eg.EdgePolylines.straight_line_from_graph(g16, l_fd)
        fd = eg.FdebSolver(poly_fd, cycles=2, steps_per_cycle=6)
        fd.run(poly_fd)
        out["fdeb_offsets"] = poly_fd.edge_offsets()
        out["fdeb_points"] = poly_fd.points()
        return out

    py32 = run_py_postproc("float32")
    py64 = run_py_postproc("float64")

    spec_payload = {
        "g16_edges": [int(v) for edge in g16_edges for v in edge],
        "cyc_edges": [int(v) for edge in cyc_edges_list for v in edge],
        "skip_edges": [int(v) for edge in skip_edges_list for v in edge],
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
    }

    node_body = """
if (typeof mod.PrismOverlapRemoval !== 'function') {
  console.log(JSON.stringify({ overlap_available: false }));
  process.exit(0);
}

const g16 = mod.CsrGraph.fromEdgeList(16, new Uint32Array(spec.g16_edges));
const gCyc = mod.CsrGraph.fromEdgeList(4, new Uint32Array(spec.cyc_edges), null, true, true);
const gSkip = mod.CsrGraph.fromEdgeList(5, new Uint32Array(spec.skip_edges), null, true, true);
const prismInit = new Float32Array(spec.prism_init);

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
    throw new Error('WASM overlap step() != run()');
  }
  lR.free?.(); lS.free?.();
  return { coords: cR, sR, sS };
};

const solvers = {};
{
  const hw = new Float32Array(spec.hw16);
  const hh = new Float32Array(spec.hh16);
  const r = runPairAndCheckStep(() => new mod.PrismOverlapRemoval(hw, hh, { maxIterations: 50, padding: 0.05, seed: 42 }), prismInit, 2);
  solvers.prism_box = r.coords;
  solvers.prism_box_ov = r.sR.lastOverlapCount;
  r.sR.free?.(); r.sS.free?.();
}
{
  const rad = new Float32Array(spec.rad16);
  const r = runPairAndCheckStep(() => mod.PrismOverlapRemoval.fromRadii(rad, { maxIterations: 50, seed: 42 }), prismInit, 2);
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
mod.project1d(c1d, cs1d);
const proj1d = Array.from(c1d);
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
const polyR = mod.EdgePolylines.straightLineFromGraph(g16, lFd);
const polyS = mod.EdgePolylines.straightLineFromGraph(g16, lFd);
const fdR = new mod.FdebSolver(polyR, { cycles: 2, stepsPerCycle: 6 });
const fdS = new mod.FdebSolver(polyS, { cycles: 2, stepsPerCycle: 6 });
fdR.run(polyR);
fdS.step(polyS);
fdS.runSteps(polyS, 3);
while (!fdS.isFinished) fdS.step(polyS);
const fdebOff = Array.from(polyR.edgeOffsetsView());
const fdebPts = Array.from(polyR.pointsView());
if (JSON.stringify(fdebPts) !== JSON.stringify(Array.from(polyS.pointsView()))) {
  throw new Error('WASM FdebSolver step() != run()');
}
polyR.free?.(); polyS.free?.(); lFd.free?.();

const out = {
  overlap_available: true,
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

fdR.free?.(); fdS.free?.(); lCrc.free?.(); lRc.free?.(); lAx.free?.();
dag.free?.(); gSkip.free?.(); gCyc.free?.(); g16.free?.();

console.log(JSON.stringify(out));
"""
    ws = run_node_json(wasm_pkg, node_body, spec_payload)
    if not ws.get("overlap_available", False):
        return [{
            "case": "cross_target_overlap_layering_py_vs_wasm",
            "status": "skipped_no_wasm_overlap_layering",
            "elapsed_sec": round(time.perf_counter() - t0, 4),
        }]

    max_f32_diff = 0.0
    max_f64_rel = 0.0
    ws_sol = ws["solvers"]
    for key in ("prism_box", "prism_rad"):
        max_f32_diff = max(max_f32_diff, check_array_close(key, py32[key], ws_sol[key], 1e-4))
        max_f64_rel = max(max_f64_rel, check_f64_vs_f32_rel(key, py64[key], py32[key], 5e-3))

    assert py32["prism_box_ov"] == ws_sol["prism_box_ov"] == 0
    assert py32["prism_rad_ov"] == ws_sol["prism_rad_ov"] == 0

    for pp_key, ws_pp_key in (
        ("proj_1d", "proj1d"),
        ("proj_axis", "projAxis"),
        ("proj_rect", "projRect"),
        ("proj_cl_rect", "projClRect"),
        ("fdeb_points", "fdebPts"),
    ):
        max_f32_diff = max(max_f32_diff, check_array_close(pp_key, py32[pp_key], ws[ws_pp_key], 1e-4))
        max_f64_rel = max(max_f64_rel, check_f64_vs_f32_rel(pp_key, py64[pp_key], py32[pp_key], 5e-3))

    assert list(ce_py.ravel()) == ws["ce"]
    assert dag_py.num_edges == ws["dagNumEdges"]
    assert ws["dagCeLen"] == 0
    assert list(layers_lp_py) == ws["layersLp"]
    assert list(layers_ns_py) == ws["layersNs"]
    assert list(py32["fdeb_offsets"]) == ws["fdebOff"]

    return [{
        "case": "cross_target_overlap_layering_py_vs_wasm",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "max_f32_diff": round(max_f32_diff, 12),
        "max_f64_diff": round(max_f64_rel, 12),
    }]
