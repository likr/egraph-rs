"""Cross-target Python <-> WASM parity checks for HDE, HDE-ND, and SGD embedding presets."""

from __future__ import annotations

import pathlib
import time
from collections.abc import Callable
from typing import Any

from parity.common import (
    check_array_close,
    make_asymmetric_coords,
    make_barbell_edges,
    make_grid_star_graph_spec,
    resolve_wasm_entries,
    run_node_json,
)


def run_cross_target_hde_sgd_presets_parity(
    eg: Any, wasm_pkg: pathlib.Path
) -> list[dict[str, Any]]:
    """Verify Python float32 <-> WASM f32 parity across HDE, HDE-ND, and SGD presets."""
    import numpy as np

    if not hasattr(eg, "hde"):
        return []
    if resolve_wasm_entries(wasm_pkg) is None:
        return []

    t0 = time.perf_counter()

    n_asym, asym_edges = make_grid_star_graph_spec()
    g_asym = eg.CsrGraph.from_edges(n_asym, asym_edges, directed=False, sanitize=True)
    bb_edges = make_barbell_edges(5)
    g_bb = eg.CsrGraph.from_edges(10, bb_edges, directed=False, sanitize=True)

    init10_f32 = np.array(make_asymmetric_coords(10, scale=1.2), dtype=np.float32)

    hde_py = eg.hde(g_asym, dimensions=16, normalize=True, seed=42, dtype="float32")
    hde_piv_py = eg.hde(g_asym, pivots=[0, 5, 10, 15], normalize=True, seed=42, dtype="float32")
    hde_nd_py = eg.hde_nd(g_asym, dimensions=16, output_dims=3, normalize=True, seed=42, dtype="float32")

    sgd_factories: tuple[tuple[str, Callable[[], Any]], ...] = (
        ("ort_kmeans", lambda: eg.SgdSolver.ortmann_sparse_stress(g_bb, num_pivots=4, pivot_strategy="kmeans_sp", iterations=10, seed=42, dtype="float32")),
        ("ort_maxmin_rand", lambda: eg.SgdSolver.ortmann_sparse_stress(g_bb, num_pivots=4, pivot_strategy="maxmin_random_sp", iterations=10, seed=42, dtype="float32")),
        ("ort_maxmin", lambda: eg.SgdSolver.ortmann_sparse_stress(g_bb, num_pivots=4, pivot_strategy="maxmin_sp", iterations=10, seed=42, dtype="float32")),
        ("largevis", lambda: eg.LargeVisSolver(g_bb, iterations=10, negative_samples=3, seed=42, dtype="float32")),
        ("drgraph", lambda: eg.DrGraphSolver(g_bb, k_order=1, negative_samples=3, iterations=10, seed=42, dtype="float32")),
        ("cne_infonce", lambda: eg.GraphCneSolver(g_bb, objective="infonce", negative_samples=3, iterations=10, seed=42, dtype="float32")),
        ("cne_neg_tsne", lambda: eg.GraphCneSolver(g_bb, objective="neg_tsne", negative_samples=3, iterations=10, seed=42, dtype="float32")),
    )
    sgd_py: dict[str, Any] = {}
    for key, mk in sgd_factories:
        s = mk()
        lay = eg.Layout2d.from_coords(init10_f32)
        s.run(lay)
        sgd_py[key] = lay.coords(copy=True)

    spec_payload = {
        "n_asym": n_asym,
        "asym_edges": [int(v) for edge in asym_edges for v in edge],
        "bb_edges": [int(v) for edge in bb_edges for v in edge],
        "init10": [float(v) for v in init10_f32.ravel()],
    }

    node_body = """
if (typeof mod.hde !== 'function') {
  console.log(JSON.stringify({ hde_available: false }));
  process.exit(0);
}

const gAsym = mod.CsrGraph.fromEdgeList(spec.n_asym, new Uint32Array(spec.asym_edges), null, false, true);
const gBb = mod.CsrGraph.fromEdgeList(10, new Uint32Array(spec.bb_edges), null, false, true);
const init10 = new Float32Array(spec.init10);

const lHde = mod.hde(gAsym, { dimensions: 16, normalize: true, seed: 42 });
const lHdePiv = mod.hde(gAsym, { pivots: [0, 5, 10, 15], normalize: true, seed: 42 });
const lHdeNd = mod.hdeNd(gAsym, { dimensions: 16, outputDims: 3, normalize: true, seed: 42 });
const hdeOut = {
  coords: Array.from(lHde.coordsView()), numNodes: lHde.numNodes,
  pivCoords: Array.from(lHdePiv.coordsView()),
  ndCoords: Array.from(lHdeNd.coordsView()), ndNumNodes: lHdeNd.numNodes, ndDim: lHdeNd.dims,
};
lHde.free?.(); lHdePiv.free?.(); lHdeNd.free?.();

const sgdConfigs = [
  ['ort_kmeans', () => mod.SgdSolver.ortmannSparseStress(gBb, { numPivots: 4, pivotStrategy: 'kmeansSp', iterations: 10, seed: 42 })],
  ['ort_maxmin_rand', () => mod.SgdSolver.ortmannSparseStress(gBb, { numPivots: 4, pivotStrategy: 'maxminRandomSp', iterations: 10, seed: 42 })],
  ['ort_maxmin', () => mod.SgdSolver.ortmannSparseStress(gBb, { numPivots: 4, pivotStrategy: 'maxminSp', iterations: 10, seed: 42 })],
  ['largevis', () => new mod.LargeVisSolver(gBb, { iterations: 10, negativeSamples: 3, seed: 42 })],
  ['drgraph', () => new mod.DrGraphSolver(gBb, { kOrder: 1, negativeSamples: 3, iterations: 10, seed: 42 })],
  ['cne_infonce', () => new mod.GraphCneSolver(gBb, { objective: 'infoNce', negativeSamples: 3, iterations: 10, seed: 42 })],
  ['cne_neg_tsne', () => new mod.GraphCneSolver(gBb, { objective: 'negTSne', negativeSamples: 3, iterations: 10, seed: 42 })],
];
const sgdOut = {};
for (const [k, mk] of sgdConfigs) {
  const s1 = mk();
  const s2 = mk();
  const l1 = mod.Layout2d.fromCoords(init10);
  const l2 = mod.Layout2d.fromCoords(init10);
  s1.run(l1); s2.step(l2); s2.runSteps(l2, 3); s2.run(l2);
  const c1 = Array.from(l1.coordsView());
  const c2 = Array.from(l2.coordsView());
  if (!arrEq(c1, c2)) throw new Error(`WASM step()!=run() for ${k}`);
  sgdOut[k] = c1;
  l1.free?.(); l2.free?.(); s1.free?.(); s2.free?.();
}

const gEmpty = mod.CsrGraph.fromEdgeList(0, new Uint32Array([]), null, false, true);
const gSingle = mod.CsrGraph.fromEdgeList(1, new Uint32Array([]), null, false, true);
const gDisc = mod.CsrGraph.fromEdgeList(6, new Uint32Array([0, 1, 1, 2, 0, 2, 3, 4]), null, false, true);
const gDirRec = mod.CsrGraph.fromEdgeList(4, new Uint32Array([0, 1, 1, 0, 1, 2, 2, 1, 2, 3]), null, true, false);
for (const gDeg of [gEmpty, gSingle, gDisc, gDirRec]) {
  const m = gDeg.numNodes();
  const h1 = mod.hde(gDeg, { dimensions: 4, seed: 42 });
  if (!h1.isAllFinite()) throw new Error('WASM hde non-finite on degenerate graph');
  h1.free?.();
  const h2 = mod.hdeNd(gDeg, { dimensions: 4, outputDims: 3, seed: 42 });
  if (!h2.isAllFinite()) throw new Error('WASM hdeNd non-finite on degenerate graph');
  h2.free?.();
  for (const s of [
    mod.SgdSolver.ortmannSparseStress(gDeg, { numPivots: 3, iterations: 4, seed: 42 }),
    new mod.LargeVisSolver(gDeg, { iterations: 4, seed: 42 }),
    new mod.DrGraphSolver(gDeg, { iterations: 4, seed: 42 }),
    new mod.GraphCneSolver(gDeg, { iterations: 4, seed: 42 }),
  ]) {
    const ld = mod.Layout2d.zeros(m);
    s.run(ld);
    if (!ld.isAllFinite()) throw new Error('WASM SGD preset non-finite on degenerate graph');
    ld.free?.(); s.free?.();
  }
}
gEmpty.free?.(); gSingle.free?.(); gDisc.free?.(); gDirRec.free?.();

assertNonPanicError('wasm/hde/dimensions_0', () => mod.hde(gBb, { dimensions: 0 }));
assertNonPanicError('wasm/hdeNd/outputDims_0', () => mod.hdeNd(gBb, { outputDims: 0 }));
assertNonPanicError('wasm/ortmann/bad_strat', () => mod.SgdSolver.ortmannSparseStress(gBb, { pivotStrategy: 'bad' }));
assertNonPanicError('wasm/largevis/neg_gamma', () => new mod.LargeVisSolver(gBb, { gamma: -1 }));
assertNonPanicError('wasm/drgraph/k0', () => new mod.DrGraphSolver(gBb, { kOrder: 0 }));
assertNonPanicError('wasm/cne/bad_obj', () => new mod.GraphCneSolver(gBb, { objective: 'bad' }));

gAsym.free?.(); gBb.free?.();
console.log(JSON.stringify({ hde_available: true, hdeOut, sgdOut }));
"""
    ws = run_node_json(wasm_pkg, node_body, spec_payload)
    if not ws.get("hde_available", False):
        return [{
            "case": "cross_target_hde_sgd_presets_py_vs_wasm",
            "status": "skipped_no_wasm_hde",
            "elapsed_sec": round(time.perf_counter() - t0, 4),
        }]

    max_f32_diff = 0.0
    assert hde_py.num_nodes == ws["hdeOut"]["numNodes"] == n_asym
    assert hde_nd_py.num_nodes == ws["hdeOut"]["ndNumNodes"] == n_asym
    assert hde_nd_py.dims == ws["hdeOut"]["ndDim"] == 3
    max_f32_diff = max(max_f32_diff, check_array_close("hde/asym", hde_py.coords(copy=True), ws["hdeOut"]["coords"], 1e-4))
    max_f32_diff = max(max_f32_diff, check_array_close("hde/pivots", hde_piv_py.coords(copy=True), ws["hdeOut"]["pivCoords"], 1e-4))
    max_f32_diff = max(max_f32_diff, check_array_close("hde_nd/asym", hde_nd_py.coords(copy=True), ws["hdeOut"]["ndCoords"], 1e-4))

    for key, py_coords in sgd_py.items():
        max_f32_diff = max(max_f32_diff, check_array_close(f"sgd_presets/{key}", py_coords, ws["sgdOut"][key], 1e-4))

    return [{
        "case": "cross_target_hde_sgd_presets_py_vs_wasm",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "max_f32_diff": round(max_f32_diff, 12),
    }]
