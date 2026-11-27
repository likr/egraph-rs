"""Cross-target Python <-> WASM parity checks for TsNet partial-BFS, IbUmap DoG, backbone, and Poisson."""

from __future__ import annotations

import pathlib
import time
from typing import Any

from parity.common import (
    check_array_close,
    make_asymmetric_coords,
    make_grid_edges,
    resolve_wasm_entries,
    run_node_json,
)


def run_cross_target_solvers_backbone_poisson_parity(
    eg: Any, wasm_pkg: pathlib.Path
) -> list[dict[str, Any]]:
    """Verify Python <-> WASM parity across TsNet partial-BFS, IbUmap DoG, backbone, and ScreenedPoisson."""
    import numpy as np

    if not hasattr(eg, "ScreenedPoissonField2d"):
        return []
    if resolve_wasm_entries(wasm_pkg) is None:
        return []

    t0 = time.perf_counter()

    g16_edges = make_grid_edges(4, 4)
    g16 = eg.CsrGraph.from_edges(16, g16_edges, directed=False, sanitize=True)
    diff_coords_py, _, _ = eg.compute_diffusion_embedding(g16, dim=3, p=0.5, seed=42)
    rerank_py = eg.build_rerank_knn(diff_coords_py, k=4, coarse_dim=2, rerank_factor=2)

    init16_f32 = np.array(make_asymmetric_coords(16, scale=0.8), dtype=np.float32)
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
        rerank_py, backend="exact", iterations=4, interpolation_order=1, intervals=16,
        min_grid_dim=32, max_grid_dim=128, seed=42, dtype="float32",
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
        iterations=5,
        interpolation_order=1,
        intervals=16,
        min_grid_dim=32,
        max_grid_dim=128,
        seed=42,
        dtype="float32",
    )
    ib_py.run(l_ib_py)

    lm_dog_default_py = eg.compute_regularized_dog_varadhan_landmarks(g16)
    sgd_dog_py = eg.SgdSolver.sparse_barrier_dog(g16, iterations=4, seed=42, min_degree=1, jl_dim=8, dtype="float32")
    dog_stats_py = sgd_dog_py.dog_stats()
    assert dog_stats_py is not None
    l_sgd_dog_py = eg.Layout2d.from_coords(init16_f32)
    sgd_dog_py.run(l_sgd_dog_py)

    gm_dog_py = eg.GuMapSolver.from_graph_dog(g16, min_degree=1, jl_dim=8, iterations=4, seed=42, dtype="float32")

    l_fdeb_py = eg.Layout2d.from_coords(init16_f32)
    fdeb_py = eg.FdebSolver(g16, cycles=2, steps_per_cycle=4, dtype="float32")
    fdeb_py.run(l_fdeb_py)

    hier_py = eg.louvain_coarsen(g16, max_coarse_nodes=4, seed=42)
    rhgh_lm_py, rhgh_st_py = eg.compute_pivoted_rhgh_landmarks(
        g16, num_landmarks=8, min_degree=1, num_probes=16, seed=42
    )
    g16_bb_weights = [1.0 + 0.05 * i for i in range(len(g16_edges))]
    g16_bb_py = eg.CsrGraph.from_edges(16, g16_edges, weights=g16_bb_weights, directed=False, sanitize=True)
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

    sp_def_py = eg.ScreenedPoissonField2d(dtype="float32")
    assert sp_def_py.grid_size == 512
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
    sp_eval_py = sp_field_py.evaluate_in_bounds(l_sp_init_py, -2.0, -2.0, 2.0, 2.0)
    sp_dens_py = sp_field_py.density_grid()
    assert sp_dens_py.shape == (sp_field_py.density_grid_dim, sp_field_py.density_grid_dim)
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

    spec_payload = {
        "g16_edges": [int(v) for edge in g16_edges for v in edge],
        "g16_bb_weights": g16_bb_weights,
        "init16": [float(v) for v in init16_f32.ravel()],
        "rerank_indptr": [int(v) for v in rerank_py.indptr()],
        "rerank_indices": [int(v) for v in rerank_py.indices()],
        "rerank_dists": [float(v) for v in rerank_py.distances()],
    }

    node_body = """
const g16 = mod.CsrGraph.fromEdgeList(16, new Uint32Array(spec.g16_edges));
const init16 = new Float32Array(spec.init16);
const lTs = mod.Layout2d.fromCoords(init16);
const lTsProg = mod.Layout2d.fromCoords(init16);
const tsOpts = {
  backend: 'fft',
  stageIterations: [2, 2, 2],
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
const tfdpKnn = mod.TFdpSolver.fromKnn(knnObj, { backend: 'exact', iterations: 4, interpolationOrder: 1, intervals: 16, minGridDim: 32, maxGridDim: 128, seed: 42 });
tfdpKnn.run(lTfdpKnn);

const lIb = mod.Layout2d.fromCoords(init16);
const ib = new mod.IbUmapSolver(g16, {
  numLandmarks: 8, minDegree: 1, jlDim: 8, degreeNormalize: true, pExp: 0.5, epsPsd: 1e-4, tCoarse: 32.0,
  backend: 'ibfft', iterations: 5, interpolationOrder: 1, intervals: 16, minGridDim: 32, maxGridDim: 128, seed: 42,
});
ib.run(lIb);

const lmDogDef = mod.computeRegularizedDogVaradhanLandmarks(g16);
const sgdDog = mod.SgdSolver.sparseBarrierDog(g16, { iterations: 4, seed: 42, minDegree: 1, jlDim: 8 });
const dogStats = sgdDog.dogStats();
assert.ok(dogStats && dogStats.degree >= 40);
const lSgdDog = mod.Layout2d.fromCoords(init16);
sgdDog.run(lSgdDog);

const gmDog = mod.GuMapSolver.fromGraphDog(g16, { minDegree: 1, jlDim: 8, iterations: 4, seed: 42 });
const lFdeb = mod.Layout2d.fromCoords(init16);
const fdeb = new mod.FdebSolver(g16, { cycles: 2, stepsPerCycle: 4 });
fdeb.run(lFdeb);

const hier = mod.louvainCoarsen(g16, { maxCoarseNodes: 4, seed: 42 });
const rhghRes = mod.computePivotedRhghLandmarks(g16, { numLandmarks: 8, minDegree: 1, numProbes: 16, seed: 42 });
const rhghLm = rhghRes.landmarks();
const g16Bb = mod.CsrGraph.fromEdgeList(16, new Uint32Array(spec.g16_edges), new Float32Array(spec.g16_bb_weights));
const bb = mod.coarseBackboneInit(g16Bb, {
  maxCoarseNodes: 4, edgeLengthMode: "resistanceCurvatureFlow", distanceBackend: "rhgh", coarseTMax: 20, seed: 42,
});
const bbAligned = mod.Layout2d.fromCoords(bb.fineLayout().copyCoords());
const bbCoarse = bb.coarseLayout();
mod.alignClusteredCentroids(bbAligned, bbCoarse, bb.nodeToCoarse(), 0.5);

const spDef = new mod.ScreenedPoissonField2d();
assert.equal(spDef.gridSize, 512);
spDef.free?.();
const spField = new mod.ScreenedPoissonField2d({
  kernel: "screenedPoisson", gridSize: 64, paddingFactor: 2, sigma: 0.2, kappa: 1.0, rhoMax: 0.1, exactAdjoint: true,
});
const lSpInit = mod.Layout2d.fromCoords(init16);
const spEval = spField.evaluateInBounds(lSpInit, -2.0, -2.0, 2.0, 2.0);
const spDens = spField.densityGrid();
const lSpRef = mod.Layout2d.fromCoords(init16);
const spRef = mod.ScreenedPoissonRefiner.fromGraph(g16, {
  gridSize: 64, paddingFactor: 2, sigma: 0.2, kappa: 1.0, rhoMax: 0.1, iterations: 4, anchorWeight: 0.1, seed: 42,
});
spRef.run(lSpRef);

const out = {
  tsPartial: { coords: Array.from(lTs.coordsView()), ffts: fftStats.totalFft2dCalls, pFfts: fftStats.particleForwardFfts, progFrames: progFrames.length },
  tsRnCoords: Array.from(lTsRn.coordsView()),
  tfdpKnnCoords: Array.from(lTfdpKnn.coordsView()),
  ib: { a: ib.a, b: ib.b, ab: Array.from(ib.ab), numEdges: ib.numEdges, ffts: ib.totalFft2dCalls, coords: Array.from(lIb.coordsView()) },
  lmDogDef: { table: Array.from(lmDogDef.table()), t1: lmDogDef.t1, t2: lmDogDef.t2, alpha: lmDogDef.alpha },
  dogStats: { degree: dogStats.degree, numLandmarks: dogStats.numLandmarks, t1: dogStats.t1, t2: dogStats.t2, alpha: dogStats.alpha },
  sgdDogCoords: Array.from(lSgdDog.coordsView()),
  gmAb: Array.from(gmDog.ab),
  fdeb: { offsets: Array.from(fdeb.edgeOffsets()), points: Array.from(fdeb.points()) },
  backbone_bundle: {
    hierClusters: hier.numClusters,
    hierN2c: Array.from(hier.nodeToCoarse()),
    rhghGamma: rhghRes.gamma,
    rhghTable: Array.from(rhghLm.table()),
    bbFine: Array.from(bb.fineLayout().coordsView()),
    bbAligned: Array.from(bbAligned.coordsView()),
  },
  screened_poisson: {
    energy: spEval.energy,
    fftCalls: spEval.fft2dCalls,
    fx: Array.from(spEval.forceX()),
    fy: Array.from(spEval.forceY()),
    dens: Array.from(spDens),
    refCoords: Array.from(lSpRef.coordsView()),
    refLastEnergy: spRef.lastEnergy,
  },
};

spRef.free?.(); lSpRef.free?.(); spEval.free?.(); lSpInit.free?.(); spField.free?.();
fdeb.free?.(); lFdeb.free?.(); gmDog.free?.(); lmDogDef.free?.(); lSgdDog.free?.(); dogStats.free?.(); sgdDog.free?.();
bbAligned.free?.(); bbCoarse.free?.(); bb.free?.(); g16Bb.free?.(); rhghLm.free?.(); rhghRes.free?.(); hier.free?.();
ib.free?.(); lIb.free?.(); tfdpKnn.free?.(); lTfdpKnn.free?.(); knnObj.free?.();
fftStats.free?.(); lTsRn.free?.(); tsRn.free?.(); tsPartial.free?.(); tsPartialProg.free?.(); lTs.free?.(); lTsProg.free?.(); g16.free?.();

console.log(JSON.stringify(out));
"""
    ws = run_node_json(wasm_pkg, node_body, spec_payload)

    max_f32_diff = 0.0
    max_f64_diff = 0.0

    def check_close(label: str, a: Any, b: Any, tol: float, is_f32: bool = True) -> None:
        nonlocal max_f32_diff, max_f64_diff
        diff = check_array_close(label, a, b, tol)
        if is_f32:
            max_f32_diff = max(max_f32_diff, diff)
        else:
            max_f64_diff = max(max_f64_diff, diff)

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

    bb_ws = ws["backbone_bundle"]
    assert hier_py["num_clusters"] == bb_ws["hierClusters"]
    assert list(hier_py["node_to_coarse"]) == bb_ws["hierN2c"]
    check_close("compute_pivoted_rhgh_landmarks/gamma", [rhgh_st_py["gamma"]], [bb_ws["rhghGamma"]], 1e-12, is_f32=False)
    check_close("compute_pivoted_rhgh_landmarks/table", rhgh_lm_py.table(), bb_ws["rhghTable"], 1e-12, is_f32=False)
    check_close("coarse_backbone_init/fine_layout", bb_py["fine_layout"].coords(copy=True), bb_ws["bbFine"], 1e-4)
    check_close("align_clustered_centroids", bb_aligned_py.coords(copy=True), bb_ws["bbAligned"], 1e-4)

    sp_ws = ws["screened_poisson"]
    assert sp_eval_py["fft_2d_calls"] == sp_ws["fftCalls"]
    check_close("ScreenedPoissonField2d.energy", [sp_eval_py["energy"]], [sp_ws["energy"]], 1e-10, is_f32=False)
    check_close("ScreenedPoissonField2d.force_x", sp_eval_py["force_x"], sp_ws["fx"], 1e-5)
    check_close("ScreenedPoissonField2d.force_y", sp_eval_py["force_y"], sp_ws["fy"], 1e-5)
    check_close("ScreenedPoissonField2d.density_grid", sp_dens_py.ravel(), sp_ws["dens"], 1e-5)
    check_close("ScreenedPoissonRefiner.run", l_sp_ref_py.coords(copy=True), sp_ws["refCoords"], 1e-4)
    check_close("ScreenedPoissonRefiner.last_energy", [sp_ref_py.last_energy], [sp_ws["refLastEnergy"]], 1e-10, is_f32=False)

    return [{
        "case": "cross_target_solvers_backbone_poisson_py_vs_wasm",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "max_f32_diff": round(max_f32_diff, 12),
        "max_f64_diff": round(max_f64_diff, 12),
    }]
