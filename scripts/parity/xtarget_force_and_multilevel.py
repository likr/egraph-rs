"""Cross-target Python <-> WASM parity checks for TFdp, IbUmap, Sgd force presets, FA2, and YifanHu."""

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


def run_cross_target_force_multilevel_parity(
    eg: Any, wasm_pkg: pathlib.Path
) -> list[dict[str, Any]]:
    """Verify Python float32/float64 <-> WASM f32 parity across TFdp, IbUmap, FA2, and YifanHu."""
    import numpy as np

    if not hasattr(eg, "TFdpSolver"):
        return []
    if resolve_wasm_entries(wasm_pkg) is None:
        return []

    t0 = time.perf_counter()

    g16_edges = make_grid_edges(4, 4)
    g16 = eg.CsrGraph.from_edges(16, g16_edges, directed=False, sanitize=True)
    init16_f32 = np.array(make_asymmetric_coords(16, scale=0.8), dtype=np.float32)
    init16_f64 = init16_f32.astype(np.float64)

    g25_edges = make_grid_edges(5, 5)
    g25 = eg.CsrGraph.from_edges(25, g25_edges, directed=False, sanitize=True)
    init25_f32 = np.array(make_asymmetric_coords(25, scale=1.0), dtype=np.float32)
    init25_f64 = init25_f32.astype(np.float64)

    knn16, _, _ = eg.build_pure_2hop_knn(g16)
    deg16 = np.diff(g16.indptr()).astype(np.uint32)
    lm16 = eg.compute_regularized_dog_varadhan_landmarks(g16, num_landmarks=8, seed=42)
    fuzzy16 = eg.compute_fuzzy_simplicial_set(knn16)

    def run_py_force(dtype_name: str) -> dict[str, Any]:
        c16 = init16_f32 if dtype_name == "float32" else init16_f64
        c25 = init25_f32 if dtype_name == "float32" else init25_f64

        out: dict[str, Any] = {}
        for bk in ("ibfft", "barnes_hut", "exact"):
            l_tf = eg.Layout2d.from_coords(c16)
            eg.TFdpSolver(g16, backend=bk, iterations=8, seed=42, dtype=dtype_name).run(l_tf)
            out[f"tfdp_{bk}"] = l_tf.coords(copy=True)

            l_ib = eg.Layout2d.from_coords(c16)
            eg.IbUmapSolver(g16, backend=bk, iterations=6, seed=42, dtype=dtype_name).run(l_ib)
            out[f"ibumap_{bk}"] = l_ib.coords(copy=True)

        l_tfk = eg.Layout2d.from_coords(c16)
        eg.TFdpSolver.from_knn(knn16, backend="exact", iterations=6, seed=42, dtype=dtype_name).run(l_tfk)
        out["tfdp_knn"] = l_tfk.coords(copy=True)

        l_ibk = eg.Layout2d.from_coords(c16)
        eg.IbUmapSolver.from_knn(knn16, backend="exact", iterations=6, seed=42, dtype=dtype_name).run(l_ibk)
        out["ibumap_knn"] = l_ibk.coords(copy=True)

        l_ibf = eg.Layout2d.from_coords(c16)
        eg.IbUmapSolver.from_fuzzy_set(fuzzy16, degrees=deg16, backend="exact", iterations=6, seed=42, dtype=dtype_name).run(l_ibf)
        out["ibumap_fset"] = l_ibf.coords(copy=True)

        for bk in ("exact", "ibfft"):
            l_ibl = eg.Layout2d.from_coords(c16)
            s_ibl = eg.IbUmapSolver.from_landmarks(
                g16, lm16, max_2hop_per_node=16, backend=bk, iterations=6, seed=42, dtype=dtype_name
            )
            s_ibl.run(l_ibl)
            out[f"ibumap_lm_{bk}"] = l_ibl.coords(copy=True)
            out[f"ibumap_lm_{bk}_edges"] = int(s_ibl.num_edges)

        l_pmk = eg.Layout2d.from_coords(c16)
        s_pmk = eg.SgdSolver.pacmap_graph(
            g16, n_neighbors=2, phase_iterations=(3, 3, 4), keep_all_1hop=True, seed=42, dtype=dtype_name
        )
        s_pmk.run(l_pmk)
        out["sgd_pacmap_keep"] = l_pmk.coords(copy=True)
        out["sgd_pacmap_keep_springs"] = int(s_pmk.num_springs)

        l_pm_lm = eg.Layout2d.from_coords(c16)
        s_pm_lm = eg.SgdSolver.pacmap_graph_with_landmarks(
            g16, lm16, n_neighbors=4, phase_iterations=(3, 3, 4),
            keep_all_1hop=True, max_1hop_per_node=2, degree_weight_beta=0.5, seed=42, dtype=dtype_name,
        )
        s_pm_lm.run(l_pm_lm)
        out["sgd_pacmap_cap_beta_lm"] = l_pm_lm.coords(copy=True)
        out["sgd_pacmap_cap_beta_lm_springs"] = int(s_pm_lm.num_springs)

        sgd_specs = {
            "sgd_snap": eg.SgdSolver.snap_t_fdp(g16, iterations=8, negatives_per_edge=3, seed=42, dtype=dtype_name),
            "sgd_snap_beta": eg.SgdSolver.snap_t_fdp(g16, iterations=8, negatives_per_edge=3, degree_beta=0.5, bounded_attraction=False, seed=42, dtype=dtype_name),
            "sgd_pacmap": eg.SgdSolver.pacmap_graph(g16, n_neighbors=5, phase_iterations=(3, 3, 4), seed=42, dtype=dtype_name),
            "sgd_sgd2": eg.SgdSolver.sgd2(g16, iterations=8, batch_size=16, seed=42, dtype=dtype_name),
            "sgd_squared": eg.SgdSolver.full(g16, iterations=8, schedule="squared", seed=42, dtype=dtype_name),
        }
        for k, sv in sgd_specs.items():
            lay = eg.Layout2d.from_coords(c16)
            sv.run(lay)
            out[k] = lay.coords(copy=True)

        for tag, use_bh in (("fa2_bh", True), ("fa2_exact", False)):
            l_fa = eg.Layout2d.from_coords(c16)
            eg.ForceAtlas2Solver(g16, iterations=10, use_barnes_hut=use_bh, seed=42, dtype=dtype_name).run(l_fa)
            out[tag] = l_fa.coords(copy=True)

        l_fa_cap = eg.Layout2d.from_coords(c16)
        eg.ForceAtlas2Solver(g16, iterations=10, use_barnes_hut=True, max_displacement=0.25, seed=42, dtype=dtype_name).run(l_fa_cap)
        out["fa2_cap"] = l_fa_cap.coords(copy=True)

        l_yh = eg.Layout2d.from_coords(c25)
        yh = eg.YifanHuSolver(
            g25, max_iterations_per_level=10, min_coarsen_size=6, min_coarsen_ratio=0.9,
            convergence_tolerance=1e-6, seed=42, dtype=dtype_name,
        )
        yh.run(l_yh)
        out["yh"] = l_yh.coords(copy=True)
        out["yh_levels"] = int(yh.num_levels)
        return out

    py32 = run_py_force("float32")
    py64 = run_py_force("float64")

    spec_payload = {
        "g16_edges": [int(v) for edge in g16_edges for v in edge],
        "g25_edges": [int(v) for edge in g25_edges for v in edge],
        "init16": [float(v) for v in init16_f32.ravel()],
        "init25": [float(v) for v in init25_f32.ravel()],
        "knn_indptr": [int(v) for v in knn16.indptr()],
        "knn_indices": [int(v) for v in knn16.indices()],
        "knn_dists": [float(v) for v in knn16.distances()],
        "deg16": [int(v) for v in deg16],
    }

    node_body = """
if (typeof mod.TFdpSolver !== 'function') {
  console.log(JSON.stringify({ force_available: false }));
  process.exit(0);
}

const g16 = mod.CsrGraph.fromEdgeList(16, new Uint32Array(spec.g16_edges));
const g25 = mod.CsrGraph.fromEdgeList(25, new Uint32Array(spec.g25_edges));
const init16 = new Float32Array(spec.init16);
const init25 = new Float32Array(spec.init25);
const knn16 = mod.KnnGraph.fromCsr(16, new Uint32Array(spec.knn_indptr), new Uint32Array(spec.knn_indices), new Float64Array(spec.knn_dists));
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
    throw new Error('WASM force/multilevel step() != run()');
  }
  lR.free?.(); lS.free?.();
  return { coords: cR, sR, sS };
};

const solvers = {};
for (const bk of ['ibfft', 'barnes_hut', 'exact']) {
  {
    const r = runPairAndCheckStep(() => new mod.TFdpSolver(g16, { backend: bk, iterations: 8, seed: 42 }), init16, 3);
    solvers[`tfdp_${bk}`] = r.coords;
    r.sR.free?.(); r.sS.free?.();
  }
  {
    const r = runPairAndCheckStep(() => new mod.IbUmapSolver(g16, { backend: bk, iterations: 6, seed: 42 }), init16, 2);
    solvers[`ibumap_${bk}`] = r.coords;
    r.sR.free?.(); r.sS.free?.();
  }
}

{
  const r = runPairAndCheckStep(
    () => mod.TFdpSolver.fromKnn(
      new Uint32Array(spec.knn_indptr), new Uint32Array(spec.knn_indices), new Float64Array(spec.knn_dists),
      { backend: 'exact', iterations: 6, seed: 42 },
    ),
    init16, 2,
  );
  solvers.tfdp_knn = r.coords;
  r.sR.free?.(); r.sS.free?.();
}
{
  const r = runPairAndCheckStep(() => mod.IbUmapSolver.fromKnn(knn16, { backend: 'exact', iterations: 6, seed: 42 }), init16, 2);
  solvers.ibumap_knn = r.coords;
  r.sR.free?.(); r.sS.free?.();
}
{
  const r = runPairAndCheckStep(() => mod.IbUmapSolver.fromFuzzySet(fuzzy16, new Uint32Array(spec.deg16), { backend: 'exact', iterations: 6, seed: 42 }), init16, 2);
  solvers.ibumap_fset = r.coords;
  r.sR.free?.(); r.sS.free?.();
}
{
  const lm16 = mod.computeRegularizedDogVaradhanLandmarks(g16, null, 8, 42);
  for (const bk of ['exact', 'ibfft']) {
    const r = runPairAndCheckStep(() => mod.IbUmapSolver.fromLandmarks(g16, lm16, 16, { backend: bk, iterations: 6, seed: 42 }), init16, 2);
    solvers[`ibumap_lm_${bk}`] = r.coords;
    solvers[`ibumap_lm_${bk}_edges`] = r.sR.numEdges;
    r.sR.free?.(); r.sS.free?.();
  }
  lm16.free?.();
}
{
  const r = runPairAndCheckStep(
    () => mod.SgdSolver.pacmapGraph(g16, { nNeighbors: 2, phaseIterations: [3, 3, 4], keepAll1hop: true, seed: 42 }),
    init16, 3,
  );
  solvers.sgd_pacmap_keep = r.coords;
  solvers.sgd_pacmap_keep_springs = r.sR.numSprings;
  r.sR.free?.(); r.sS.free?.();
}
{
  const lm16Pm = mod.computeRegularizedDogVaradhanLandmarks(g16, null, 8, 42);
  const r = runPairAndCheckStep(
    () => mod.SgdSolver.pacmapGraphWithLandmarks(g16, lm16Pm, { nNeighbors: 4, phaseIterations: [3, 3, 4], keepAll1hop: true, max1hopPerNode: 2, degreeWeightBeta: 0.5, seed: 42 }),
    init16, 3,
  );
  solvers.sgd_pacmap_cap_beta_lm = r.coords;
  solvers.sgd_pacmap_cap_beta_lm_springs = r.sR.numSprings;
  r.sR.free?.(); r.sS.free?.(); lm16Pm.free?.();
}

for (const [k, mk] of [
  ['sgd_snap', () => mod.SgdSolver.snapTFdp(g16, { iterations: 8, negativesPerEdge: 3, seed: 42 })],
  ['sgd_snap_beta', () => mod.SgdSolver.snapTFdp(g16, { iterations: 8, negativesPerEdge: 3, degreeBeta: 0.5, boundedAttraction: false, seed: 42 })],
  ['sgd_pacmap', () => mod.SgdSolver.pacmapGraph(g16, { nNeighbors: 5, phaseIterations: [3, 3, 4], seed: 42 })],
  ['sgd_sgd2', () => mod.SgdSolver.sgd2(g16, { iterations: 8, batchSize: 16, seed: 42 })],
  ['sgd_squared', () => mod.SgdSolver.full(g16, { iterations: 8, kind: 'squared', seed: 42 })],
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
    init25, 4,
  );
  solvers.yh = r.coords;
  solvers.yh_levels = r.sR.numLevels;
  r.sR.free?.(); r.sS.free?.();
}

fuzzy16.free?.(); knn16.free?.(); g25.free?.(); g16.free?.();
console.log(JSON.stringify({ force_available: true, solvers }));
"""
    ws = run_node_json(wasm_pkg, node_body, spec_payload)
    if not ws.get("force_available", False):
        return [{
            "case": "cross_target_force_multilevel_py_vs_wasm",
            "status": "skipped_no_wasm_force_multilevel",
            "elapsed_sec": round(time.perf_counter() - t0, 4),
        }]

    max_f32_diff = 0.0
    max_f64_rel = 0.0
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
    )
    for key in solver_keys:
        max_f32_diff = max(max_f32_diff, check_array_close(key, py32[key], ws_sol[key], 1e-4))
        max_f64_rel = max(max_f64_rel, check_f64_vs_f32_rel(key, py64[key], py32[key], 5e-3))

    assert py32["yh_levels"] == ws_sol["yh_levels"]
    assert py32["ibumap_lm_exact_edges"] == ws_sol["ibumap_lm_exact_edges"]
    assert py32["ibumap_lm_ibfft_edges"] == ws_sol["ibumap_lm_ibfft_edges"]
    assert py32["sgd_pacmap_keep_springs"] == ws_sol["sgd_pacmap_keep_springs"]
    assert py32["sgd_pacmap_cap_beta_lm_springs"] == ws_sol["sgd_pacmap_cap_beta_lm_springs"]

    return [{
        "case": "cross_target_force_multilevel_py_vs_wasm",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "max_f32_diff": round(max_f32_diff, 12),
        "max_f64_diff": round(max_f64_rel, 12),
    }]
