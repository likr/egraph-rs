"""Cross-target Python <-> WASM parity checks for linalg, spectral, AMG-PCG, and curvature."""

from __future__ import annotations

import pathlib
import time
from typing import Any

from parity.common import (
    check_array_close,
    make_grid_star_graph_spec,
    resolve_wasm_entries,
    run_node_json,
)


def run_cross_target_linalg_spectral_parity(
    eg: Any, wasm_pkg: pathlib.Path
) -> list[dict[str, Any]]:
    """Verify Python <-> WASM parity across pivots, spectral, Chebyshev, DoG, AMG, and curvature."""
    import numpy as np

    if not hasattr(eg, "select_maxmin_sp_pivots"):
        return []
    if resolve_wasm_entries(wasm_pkg) is None:
        return []

    t0 = time.perf_counter()
    n, edges = make_grid_star_graph_spec()
    g = eg.CsrGraph.from_edges(n, edges, directed=False, sanitize=True)

    pivots = eg.select_maxmin_sp_pivots(g, 12, seed=42, min_degree=2)
    pmds_f32 = eg.pivot_mds_init(
        g, num_pivots=12, seed=42, min_degree=2, target_std=1.0, dtype="float32"
    ).coords(copy=True)
    pmds_unnorm = eg.pivot_mds_init(
        g, num_pivots=12, seed=42, min_degree=2, target_std=None, dtype="float32"
    ).coords(copy=True)
    koren_f32 = eg.koren_spectral_init(
        g, target_std=1.0, tol=1e-4, max_iterations=300, seed=42, dtype="float32"
    ).coords(copy=True)
    lams, evecs, inv_d, used_dense = eg.smallest_laplacian_eigenpairs(
        g, n_modes=4, tol=1e-4, max_iterations=300, seed=42
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

    spec_payload = {
        "n": n,
        "edges": [int(v) for edge in edges for v in edge],
        "heat_in": [float(v) for v in heat_in.ravel()],
        "b_sing": [float(v) for v in b_sing_py],
        "alt_u": alt_u,
        "alt_v": alt_v,
    }

    node_body = """
if (typeof mod.selectMaxminSpPivots !== 'function') {
  console.log(JSON.stringify({ linalg_available: false }));
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
  linalg_available: true,
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

rcfCondG.free?.(); rcfLenG.free?.(); rcf.free?.(); erEx.free?.(); erJl.free?.();
anisoLm.free?.(); anisoMat.free?.(); aniso.free?.();
amgSingRes.free?.(); amgSing.free?.(); amgStep.free?.(); amgBatchRes.free?.(); amgRes.free?.(); amgReg.free?.();
lm.free?.(); mstats.free?.(); eig.free?.(); lKoren.free?.(); lPmdsUnnorm.free?.(); lPmds.free?.(); g.free?.();

console.log(JSON.stringify(out));
"""
    ws = run_node_json(wasm_pkg, node_body, spec_payload)
    if not ws.get("linalg_available", False):
        return [{
            "case": "cross_target_linalg_spectral_py_vs_wasm",
            "status": "skipped_no_wasm_linalg",
            "elapsed_sec": round(time.perf_counter() - t0, 4),
        }]

    max_f64_diff = 0.0
    max_f32_diff = 0.0

    def check_f64(label: str, a: Any, b: Any, tol: float = 1e-12) -> None:
        nonlocal max_f64_diff
        max_f64_diff = max(max_f64_diff, check_array_close(label, a, b, tol))

    def check_f32(label: str, a: Any, b: Any, tol: float = 1e-5) -> None:
        nonlocal max_f32_diff
        max_f32_diff = max(max_f32_diff, check_array_close(label, a, b, tol))

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
        "case": "cross_target_linalg_spectral_py_vs_wasm",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "max_f64_diff": round(max_f64_diff, 15),
        "max_f32_diff": round(max_f32_diff, 12),
    }]
