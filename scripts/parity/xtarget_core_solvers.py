"""Cross-target Python <-> WASM parity checks for TsNet, Sgd, GuMap, Refiner, SM, and KK solvers."""

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


def run_cross_target_core_solvers_parity(
    eg: Any, wasm_pkg: pathlib.Path
) -> list[dict[str, Any]]:
    """Verify Python float32/float64 <-> WASM f32 parity across Euclidean core solvers."""
    import numpy as np

    if not hasattr(eg, "TsNetSolver"):
        return []
    if resolve_wasm_entries(wasm_pkg) is None:
        return []

    t0 = time.perf_counter()

    g16_edges = make_grid_edges(4, 4)
    g16 = eg.CsrGraph.from_edges(16, g16_edges, directed=False, sanitize=True)
    init16_f32 = np.array(make_asymmetric_coords(16, scale=0.8), dtype=np.float32)
    init16_f64 = init16_f32.astype(np.float64)
    gumap_init16_f32 = (init16_f32 * 5.0).astype(np.float32)
    gumap_init16_f64 = gumap_init16_f32.astype(np.float64)
    emb16 = eg.chebyshev_dog_jl_embedding(g16, degree=15, jl_dim=8, seed=42)

    g9_edges = make_grid_edges(3, 3)
    g9 = eg.CsrGraph.from_edges(9, g9_edges, directed=False, sanitize=True)
    folded9_f32 = np.array([[float(c), float(r)] for r in range(3) for c in range(3)], dtype=np.float32)
    folded9_f32[8] = np.array([0.08, 0.08], dtype=np.float32)
    init9_f32 = np.array(make_asymmetric_coords(9, scale=1.2), dtype=np.float32)

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
            lay = eg.Layout2d.from_coords(c16)
            sv = eg.TsNetSolver.pure_2hop(
                g16, backend=bk, iterations_stage1=4, iterations_stage2=4, iterations_stage3=4, seed=42, dtype=dtype_name
            )
            sv.run(lay)
            out[f"tsnet_{bk}"] = lay.coords(copy=True)
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
            lay = eg.Layout2d.from_coords(c16)
            sv.run(lay)
            out[k] = lay.coords(copy=True)
            out[f"{k}_springs"] = int(sv.num_springs)
            out[f"{k}_hinges"] = int(sv.num_hinges)

        l_gm = eg.Layout2d.from_coords(gumap_init16_f32 if dtype_name == "float32" else gumap_init16_f64)
        gm = eg.GuMapSolver.from_graph_dog(
            g16, max_2hop_per_node=16, degree=15, num_landmarks=8, n_epochs=15, learning_rate=0.2, seed=42, dtype=dtype_name
        )
        gm.run(l_gm)
        out["gumap"] = l_gm.coords(copy=True)
        out["gumap_edges"] = int(gm.num_edges)

        l_gm_lm = eg.Layout2d.from_coords(gumap_init16_f32 if dtype_name == "float32" else gumap_init16_f64)
        gm_lm = eg.GuMapSolver.from_landmarks(
            g16, lm16, max_2hop_per_node=16, n_epochs=15, learning_rate=0.2, seed=42, dtype=dtype_name
        )
        gm_lm.run(l_gm_lm)
        out["gumap_lm"] = l_gm_lm.coords(copy=True)
        out["gumap_lm_edges"] = int(gm_lm.num_edges)

        l_gm_asym = eg.Layout2d.from_coords(gumap_init16_f32 if dtype_name == "float32" else gumap_init16_f64)
        gm_asym = eg.GuMapSolver.from_graph_dog(
            g16, max_2hop_per_node=16, degree=15, num_landmarks=8, n_epochs=15, learning_rate=0.2, symmetric_negative=False, seed=42, dtype=dtype_name
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
            g9, rounds=4, k_cap=8, t_max=10, seed=42, w_anchor2=0.5, k_cap_decay=0.5, k_cap_min=2, dtype=dtype_name
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
            g11_twin, rounds=3, k_cap=16, t_max=10, min_sep_ratio=0.15, w_sep=1.0, seed=42, dtype=dtype_name
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

    spec_payload = {
        "g16_edges": [int(v) for edge in g16_edges for v in edge],
        "g9_edges": [int(v) for edge in g9_edges for v in edge],
        "g11_edges": [int(v) for edge in g11_edges for v in edge],
        "init16": [float(v) for v in init16_f32.ravel()],
        "gumap_init16": [float(v) for v in gumap_init16_f32.ravel()],
        "emb16": [float(v) for v in emb16.ravel()],
        "folded9": [float(v) for v in folded9_f32.ravel()],
        "twin11": [float(v) for v in twin11_f32.ravel()],
        "init9": [float(v) for v in init9_f32.ravel()],
        "hinge_src": hinge_s_src,
        "hinge_tgt": hinge_s_tgt,
        "hinge_ones": hinge_s_ones,
    }

    node_body = """
if (typeof mod.TsNetSolver !== 'function') {
  console.log(JSON.stringify({ solvers_available: false }));
  process.exit(0);
}

const g16 = mod.CsrGraph.fromEdgeList(16, new Uint32Array(spec.g16_edges));
const g9 = mod.CsrGraph.fromEdgeList(9, new Uint32Array(spec.g9_edges));
const g11 = mod.CsrGraph.fromEdgeList(11, new Uint32Array(spec.g11_edges));
const init16 = new Float32Array(spec.init16);
const gumapInit16 = new Float32Array(spec.gumap_init16);
const emb16 = new Float64Array(spec.emb16);
const folded9 = new Float32Array(spec.folded9);
const twin11 = new Float32Array(spec.twin11);
const init9 = new Float32Array(spec.init9);

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
    init16, 5,
  );
  solvers[`tsnet_${bk}`] = res.coords;
  solvers[`tsnet_${bk}_ffts`] = res.solverRun.totalFft2dCalls;
  res.solverRun.free?.(); res.solverStep.free?.();
}

for (const [tag, w2Val, capVal, bk] of [
  ['tsnet_rn_uncapped', 0.5, null, 'fft'],
  ['tsnet_rn_capped', 0.35, 8, 'exact'],
]) {
  const res = runPairAndCheckStep(
    () => mod.TsNetSolver.fromGraphRowNormalized(g16, { w2: w2Val, max2hopPerNode: capVal, backend: bk, iterationsStage1: 4, iterationsStage2: 4, iterationsStage3: 4, seed: 42 }),
    init16, 5,
  );
  solvers[tag] = res.coords;
  solvers[`${tag}_ffts`] = res.solverRun.totalFft2dCalls;
  res.solverRun.free?.(); res.solverStep.free?.();
}

const sgdSpecs = [
  ['sgd_full', () => mod.SgdSolver.full(g16, { tMax: 10, eps: 0.01, kind: 'exponential', stepScale: 1.0, seed: 42 })],
  ['sgd_dog', () => mod.SgdSolver.sparseBarrierDog(g16, { degree: 15, numLandmarks: 8, tMax: 10, eps: 0.01, seed: 42 })],
  ['sgd_rp', () => mod.SgdSolver.randomPair(g16, { pairsPerNode: 4, tMax: 10, eps: 0.01, seed: 42 })],
  ['sgd_omega', () => mod.SgdSolver.omega(g16, emb16, 8, { k: 4, minDist: 1e-3, tMax: 10, eps: 0.01, seed: 42 })],
  ['sgd_stream', () => mod.SgdSolver.streamingNegatives(g16, { negativesPerNode: 3, targetDistance: 2.0, weight: 0.25, hingeOnly: true, tMax: 10, eps: 0.01, seed: 42 })],
  [
    'sgd_hinges',
    () => mod.SgdSolver.fromPairsWithHinges(
      16,
      new Uint32Array(spec.hinge_src), new Uint32Array(spec.hinge_tgt),
      new Float32Array(spec.hinge_ones), new Float32Array(spec.hinge_ones),
      new Float32Array(spec.hinge_ones), new Float32Array(spec.hinge_ones),
      new Uint32Array([0, 1]), new Uint32Array([15, 14]),
      new Float32Array([3.0, 3.0]), new Float32Array([3.0, 3.0]),
      new Float32Array([1.0, 1.0]), new Float32Array([1.0, 1.0]),
      { tMax: 10, eps: 0.01, seed: 42 },
    ),
  ],
];
for (const [k, mk] of sgdSpecs) {
  const res = runPairAndCheckStep(mk, init16, 4);
  solvers[k] = res.coords;
  solvers[`${k}_springs`] = res.solverRun.numSprings;
  solvers[`${k}_hinges`] = res.solverRun.numHinges;
  res.solverRun.free?.(); res.solverStep.free?.();
}

{
  const res = runPairAndCheckStep(
    () => mod.GuMapSolver.fromGraphDog(g16, { max2hopPerNode: 16, degree: 15, numLandmarks: 8, nEpochs: 15, learningRate: 0.2, seed: 42 }),
    gumapInit16, 5,
  );
  solvers.gumap = res.coords;
  solvers.gumap_edges = res.solverRun.numEdges;
  res.solverRun.free?.(); res.solverStep.free?.();
}
{
  const lm16 = mod.computeRegularizedDogVaradhanLandmarks(g16, 15, 8, 42);
  const res = runPairAndCheckStep(
    () => mod.GuMapSolver.fromLandmarks(g16, lm16, 16, { nEpochs: 15, learningRate: 0.2, seed: 42 }),
    gumapInit16, 5,
  );
  solvers.gumap_lm = res.coords;
  solvers.gumap_lm_edges = res.solverRun.numEdges;
  res.solverRun.free?.(); res.solverStep.free?.(); lm16.free?.();
}
{
  const res = runPairAndCheckStep(
    () => mod.GuMapSolver.fromGraphDog(g16, { max2hopPerNode: 16, degree: 15, numLandmarks: 8, nEpochs: 15, learningRate: 0.2, symmetricNegative: false, seed: 42 }),
    gumapInit16, 5,
  );
  solvers.gumap_asym = res.coords;
  res.solverRun.free?.(); res.solverStep.free?.();
}
{
  const res = runPairAndCheckStep(() => new mod.SpatialIntruderRefiner(g9, { rounds: 3, kCap: 32, tMax: 10, seed: 42 }), folded9, 1);
  solvers.refiner = res.coords;
  const st = res.solverRun.stats();
  solvers.refiner_fp = st.nFpTotal;
  st.free?.(); res.solverRun.free?.(); res.solverStep.free?.();
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
  stDyn.free?.(); lDyn.free?.(); rfDyn.free?.();
}
{
  const res = runPairAndCheckStep(() => new mod.SpatialIntruderRefiner(g11, { rounds: 3, kCap: 16, tMax: 10, minSepRatio: 0.15, wSep: 1.0, seed: 42 }), twin11, 1);
  solvers.refiner_sep = res.coords;
  const stSep = res.solverRun.stats();
  solvers.refiner_sep_total = stSep.nSepTotal;
  stSep.free?.(); res.solverRun.free?.(); res.solverStep.free?.();
}
{
  const res = runPairAndCheckStep(() => mod.StressMajorizationSolver.fromGraph(g16, { maxIterations: 15, epsilon: 1e-6 }), init16, 4);
  solvers.sm = res.coords;
  solvers.sm_stress = res.solverRun.lastStress ?? 0.0;
  res.solverRun.free?.(); res.solverStep.free?.();
}
{
  const res = runPairAndCheckStep(() => mod.KamadaKawaiSolver.fromGraph(g9, { eps: 1e-3, maxInnerIters: 40, maxOuterIters: 20 }), init9, 4);
  solvers.kk = res.coords;
  res.solverRun.free?.(); res.solverStep.free?.();
}

g9.free?.(); g11.free?.(); g16.free?.();
console.log(JSON.stringify({ solvers_available: true, solvers }));
"""
    ws = run_node_json(wasm_pkg, node_body, spec_payload)
    if not ws.get("solvers_available", False):
        return [{
            "case": "cross_target_core_solvers_py_vs_wasm",
            "status": "skipped_no_wasm_core_solvers",
            "elapsed_sec": round(time.perf_counter() - t0, 4),
        }]

    max_f32_diff = 0.0
    max_f64_diff = 0.0
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
        max_f32_diff = max(max_f32_diff, check_array_close(f"{key}/f32_vs_wasm", py32[key], ws_sol[key], 1e-4))
        max_f64_diff = max(max_f64_diff, check_f64_vs_f32_rel(f"{key}/f64_vs_f32", py64[key], py32[key], 5e-3))

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
    max_f32_diff = max(max_f32_diff, check_array_close("sm_stress/f32_vs_wasm", [py32["sm_stress"]], [ws_sol["sm_stress"]], 1e-4))

    return [{
        "case": "cross_target_core_solvers_py_vs_wasm",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "max_f32_diff": round(max_f32_diff, 12),
        "max_f64_diff": round(max_f64_diff, 12),
    }]
