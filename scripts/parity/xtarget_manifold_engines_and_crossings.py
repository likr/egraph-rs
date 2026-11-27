"""Cross-target Python <-> WASM parity for Manifold engines, Lorentz/Einstein maps, and crossings."""

from __future__ import annotations

import math
import time
from typing import Any

from parity.common import (
    assert_non_panic_error,
    check_array_close,
    make_asymmetric_coords,
    make_grid_edges,
    resolve_wasm_entries,
    run_node_json,
)


def run_cross_target_manifold_engines_crossings_parity(
    eg: Any, wasm_pkg: Any
) -> list[dict[str, Any]]:
    """Verify Python <-> WASM parity across 6 Manifold solvers, 17 SgdSolver modes, Lorentz/metric, and crossings."""
    import numpy as np

    if not hasattr(eg, "ManifoldLayout2d") or not hasattr(eg, "lorentz_distance"):
        return []
    if resolve_wasm_entries(wasm_pkg) is None:
        return []

    t0 = time.perf_counter()
    g16_edges = make_grid_edges(4, 4)
    g16 = eg.CsrGraph.from_edges(16, g16_edges, directed=False, sanitize=True)
    g4_x_edges = [(0, 2), (1, 3), (0, 1), (1, 2), (2, 3), (3, 0)]
    g4_x = eg.CsrGraph.from_edges(4, g4_x_edges, directed=False, sanitize=True)

    c16 = np.array(make_asymmetric_coords(16, scale=0.4), dtype=np.float32)
    poin_init = eg.ManifoldLayout2d.from_layout_scaled("hyperbolic_poincare", eg.Layout2d.from_coords(c16), max_radius=0.5)
    lor_init = eg.ManifoldLayout2d.init("hyperbolic_lorentz", 16, scale=0.35, seed=42, dtype="float32")
    sph_init = eg.ManifoldLayout2d.init("spherical", 16, scale=0.5, seed=42, dtype="float32")
    tor_init = eg.ManifoldLayout2d.torus_spectral_phase_init(g16, seed=42, dtype="float32")

    poin_c = poin_init.coords(copy=True)
    lor_c = lor_init.coords(copy=True)
    sph_c = sph_init.coords(copy=True)
    tor_c = tor_init.coords(copy=True)

    # Metric + Einstein midpoint radial rescale_geodesic exact cross-target parity
    m_res = eg.ManifoldLayout2d.from_coords("hyperbolic_poincare", poin_c)
    m_res.set_scale(1.5)
    m_res.set_aspect(1.25)
    assert m_res.metric() == (1.5, 1.25)
    m_res.rescale_geodesic(2.4)
    rescaled_poin_py = m_res.coords(copy=True).ravel()

    # Lorentz -> Poincare + geodesic helpers
    lor_to_poin_py = lor_init.to_poincare_layout().coords(copy=True).ravel()
    l_exp = eg.lorentz_exp_map(0.15, -0.1, 0.08, 0.12)
    l_log = eg.lorentz_log_map(0.15, -0.1, l_exp[0], l_exp[1])
    l_dist = eg.lorentz_distance(0.15, -0.1, -0.2, 0.25)
    l_pt = eg.lorentz_to_poincare_point(0.3, -0.4)
    l_mid = eg.lorentz_einstein_midpoint([0.1, -0.2, 0.15], [0.2, 0.1, -0.05])
    p_mid = eg.poincare_einstein_midpoint([0.1, -0.2, 0.15], [0.2, 0.1, -0.05])

    # Build 17 SgdSolver modes and verify supports_manifold == True + ManifoldLayout2d run
    lm16 = eg.compute_regularized_dog_varadhan_landmarks(g16, degree=15, num_landmarks=8, seed=42)
    emb16 = eg.chebyshev_dog_jl_embedding(g16, degree=15, jl_dim=8, seed=42)
    packed_src = [u for u, _ in g16_edges]
    packed_tgt = [v for _, v in g16_edges]
    packed_d = [1.0] * len(packed_src)
    packed_w = [1.0] * len(packed_src)
    packed_h = [False] * len(packed_src)

    def make_17_py() -> list[tuple[str, Any]]:
        return [
            ("full", eg.SgdSolver.full(g16, iterations=4, seed=42, dtype="float32")),
            ("from_pairs", eg.SgdSolver.from_pairs(16, packed_src, packed_tgt, packed_d, iterations=4, seed=42, dtype="float32")),
            ("from_pairs_with_hinges", eg.SgdSolver.from_pairs_with_hinges(16, packed_src, packed_tgt, packed_d, [0], [15], [2.5], iterations=4, seed=42, dtype="float32")),
            ("from_packed_symmetric", eg.SgdSolver.from_packed_symmetric(16, packed_src, packed_tgt, packed_d, packed_w, packed_h, iterations=4, seed=42, dtype="float32")),
            ("sparse_barrier_from_landmarks", eg.SgdSolver.sparse_barrier_from_landmarks(g16, lm16, iterations=4, seed=42, dtype="float32")),
            ("sparse_barrier_dog", eg.SgdSolver.sparse_barrier_dog(g16, degree=15, num_landmarks=8, iterations=4, seed=42, dtype="float32")),
            ("streaming_negatives", eg.SgdSolver.streaming_negatives(g16, negatives_per_node=2, iterations=4, seed=42, dtype="float32")),
            ("random_pair", eg.SgdSolver.random_pair(g16, pairs_per_node=3, iterations=4, seed=42, dtype="float32")),
            ("omega", eg.SgdSolver.omega(g16, emb16, k=3, iterations=4, seed=42, dtype="float32")),
            ("sgd2", eg.SgdSolver.sgd2(g16, iterations=4, batch_size=8, seed=42, dtype="float32")),
            ("pacmap_graph", eg.SgdSolver.pacmap_graph(g16, n_neighbors=4, phase_iterations=(2, 2, 2), seed=42, dtype="float32")),
            ("pacmap_graph_with_landmarks", eg.SgdSolver.pacmap_graph_with_landmarks(g16, lm16, n_neighbors=4, phase_iterations=(2, 2, 2), seed=42, dtype="float32")),
            ("snap_t_fdp", eg.SgdSolver.snap_t_fdp(g16, iterations=4, negatives_per_edge=2, seed=42, dtype="float32")),
            ("ortmann_sparse_stress", eg.SgdSolver.ortmann_sparse_stress(g16, num_pivots=4, iterations=4, seed=42, dtype="float32")),
            ("largevis", eg.SgdSolver.largevis(g16, iterations=4, negative_samples=2, seed=42, dtype="float32")),
            ("drgraph", eg.SgdSolver.drgraph(g16, iterations=4, negative_samples=2, seed=42, dtype="float32")),
            ("graph_cne", eg.SgdSolver.graph_cne(g16, iterations=4, negative_samples=2, seed=42, dtype="float32")),
        ]

    sgd17_py: dict[str, Any] = {}
    for mode_name, sv in make_17_py():
        assert sv.supports_manifold, f"{mode_name} must support manifold"
        ml_m = eg.ManifoldLayout2d.from_coords("hyperbolic_poincare", poin_c)
        sv.run(ml_m)
        assert ml_m.is_all_finite()
        sgd17_py[mode_name] = ml_m.coords(copy=True).ravel()

    # Remaining 5 Manifold solvers + reset_schedule / run_schedule_window + max_mu check
    def run_mani_py(mk: Any, kind: str, base: Any, chunk: int) -> Any:
        sr, ss = mk(), mk()
        lr = eg.ManifoldLayout2d.from_coords(kind, base)
        ls = eg.ManifoldLayout2d.from_coords(kind, base)
        sr.run(lr)
        ss.run_steps(ls, chunk)
        while not ss.is_finished:
            ss.step(ls)
        assert np.array_equal(lr.coords(copy=True), ls.coords(copy=True))
        return lr.coords(copy=True).ravel()

    mani_py = {
        "tsnet_sph": run_mani_py(lambda: eg.TsNetSolver.pure_2hop(g16, backend="exact", iterations_stage1=2, iterations_stage2=2, iterations_stage3=2, seed=42, dtype="float32"), "spherical", sph_c, 2),
        "gumap_tor": run_mani_py(lambda: eg.GuMapSolver.from_graph_dog(g16, max_2hop_per_node=16, degree=15, num_landmarks=8, iterations=5, learning_rate=0.2, seed=42, dtype="float32"), "torus_2d", tor_c, 2),
        "sm_lor": run_mani_py(lambda: eg.StressMajorizationSolver.from_graph(g16, max_iterations=5, dtype="float32"), "hyperbolic_lorentz", lor_c, 2),
        "kk_poin": run_mani_py(lambda: eg.KamadaKawaiSolver.from_graph(g16, max_iterations=5, eps=1e-3, max_inner_iters=20, dtype="float32"), "hyperbolic_poincare", poin_c, 2),
        "fr_sph": run_mani_py(lambda: eg.FruchtermanReingoldSolver(g16, iterations=5, backend="exact", seed=42, dtype="float32"), "spherical", sph_c, 2),
    }

    # Schedule window + max_mu parity on Layout2d and ManifoldLayout2d
    sv_w2 = eg.SgdSolver.full(g16, iterations=6, max_mu=None, seed=42, dtype="float32")
    l_w2 = eg.Layout2d.from_coords(c16)
    sv_w2.run_schedule_window(l_w2, 0, 3)
    sv_w2.reset_schedule(iterations=4, max_mu=0.05, seed=99)
    sv_w2.run_schedule_window(l_w2, 0, 2)
    win_2d_py = l_w2.coords(copy=True).ravel()

    sv_wm = eg.SgdSolver.full(g16, iterations=6, max_mu=None, seed=42, dtype="float32")
    l_wm = eg.ManifoldLayout2d.from_coords("hyperbolic_poincare", poin_c)
    sv_wm.run_schedule_window(l_wm, 0, 3)
    sv_wm.reset_schedule(iterations=4, max_mu=0.05, seed=99)
    sv_wm.run_schedule_window(l_wm, 0, 2)
    win_mani_py = l_wm.coords(copy=True).ravel()

    # Crossings, torus rectangular extents, auto_pan, geodesic_distortion
    ml_poin_x = eg.ManifoldLayout2d.from_coords("hyperbolic_poincare", np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]], dtype=np.float32))
    ml_sph_x = eg.ManifoldLayout2d.from_coords("spherical", np.array([[0.0, 1.2], [0.6, 1.2], [0.6, 1.9], [0.0, 1.9]], dtype=np.float32))
    ml_tor_x = eg.ManifoldLayout2d.from_coords("torus_2d", np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]], dtype=np.float32))
    ml_tor_x.set_torus_extents(2.0, 0.5)

    sph_cn_py = int(eg.spherical_crossing_number(g4_x, ml_sph_x))
    sph_ce_py = eg.crossing_edges_spherical(g4_x, ml_sph_x)
    hyp_cn_py = int(eg.hyperbolic_crossing_number(g4_x, ml_poin_x))
    hyp_ce_py = eg.crossing_edges_hyperbolic(g4_x, ml_poin_x)
    tor_cn_py = int(eg.torus_crossing_number(g4_x, ml_tor_x))
    tor_ce_py = eg.crossing_edges_torus(g4_x, ml_tor_x)
    assert tor_cn_py == len(tor_ce_py) >= 1
    tor_segs_edge_py = eg.torus_edge_segments(0.9, 0.2, 0.1, 0.8).ravel()
    tor_segs_lay_py = np.concatenate([s.ravel() for s in eg.torus_layout_edge_segments(g4_x, ml_tor_x)])
    tor_pan_py = tor_init.torus_auto_pan(g16).coords(copy=True).ravel()
    gd_poin_py = float(eg.geodesic_distortion(g16, poin_init))
    ev_gd_py = float(eg.evaluate_metrics(g16, poin_init, ["geodesic_distortion"]).geodesic_distortion or 0.0)

    # Adversarial antipodal & Poincare boundary-clamp non-panic checks
    ml_anti = eg.ManifoldLayout2d.from_coords("spherical", np.array([[0.0, math.pi * 0.5], [math.pi, math.pi * 0.5]], dtype=np.float32))
    assert math.isfinite(ml_anti.geodesic_distance(0, 1))
    ml_clamp = eg.ManifoldLayout2d.from_coords("hyperbolic_poincare", np.array([[0.99999, 0.0], [0.0, 0.99999]], dtype=np.float32))
    ml_clamp.project_all()
    assert ml_clamp.is_all_finite()
    assert_non_panic_error("py/manifold/bad_scale", lambda: m_res.set_scale(0.0))

    spec_payload = {
        "g16_edges": [int(v) for e in g16_edges for v in e],
        "g4_edges": [int(v) for e in g4_x_edges for v in e],
        "c16": [float(v) for v in c16.ravel()],
        "poin": [float(v) for v in poin_c.ravel()],
        "lor": [float(v) for v in lor_c.ravel()],
        "sph": [float(v) for v in sph_c.ravel()],
        "tor": [float(v) for v in tor_c.ravel()],
        "lm_table": [float(v) for v in lm16.table().ravel()],
        "lm_ids": [int(v) for v in lm16.landmarks()],
        "emb16": [float(v) for v in emb16.ravel()],
        "packed_src": packed_src,
        "packed_tgt": packed_tgt,
        "packed_d": packed_d,
        "packed_w": packed_w,
        "packed_h": [1 if x else 0 for x in packed_h],
    }

    node_body = """
const g16 = mod.CsrGraph.fromEdgeList(16, new Uint32Array(spec.g16_edges));
const g4 = mod.CsrGraph.fromEdgeList(4, new Uint32Array(spec.g4_edges));
const c16 = new Float32Array(spec.c16);
const poinC = new Float32Array(spec.poin);
const lorC = new Float32Array(spec.lor);
const sphC = new Float32Array(spec.sph);
const torC = new Float32Array(spec.tor);

const mRes = mod.ManifoldLayout2d.fromCoords('hyperbolic_poincare', poinC);
mRes.setScale(1.5);
mRes.setAspect(1.25);
assert.equal(mRes.scale, 1.5);
assert.equal(mRes.aspect, 1.25);
assert.deepEqual(mRes.metric(), { scale: 1.5, aspect: 1.25 });
mRes.rescaleGeodesic(2.4);
const rescaledPoin = Array.from(mRes.coordsView());

const mLor = mod.ManifoldLayout2d.fromCoords('hyperbolic_lorentz', lorC);
const lorToPoin = Array.from(mLor.toPoincareLayout().coordsView());
const lExp = Array.from(mod.lorentzExpMap(0.15, -0.1, 0.08, 0.12));
const lLog = Array.from(mod.lorentzLogMap(0.15, -0.1, lExp[0], lExp[1]));
const lDist = mod.lorentzDistance(0.15, -0.1, -0.2, 0.25);
const lPt = Array.from(mod.lorentzToPoincarePoint(0.3, -0.4));
const lMid = Array.from(mod.lorentzEinsteinMidpoint(new Float64Array([0.1, -0.2, 0.15]), new Float64Array([0.2, 0.1, -0.05])));
const pMid = Array.from(mod.poincareEinsteinMidpoint(new Float64Array([0.1, -0.2, 0.15]), new Float64Array([0.2, 0.1, -0.05])));

const lm16 = mod.computeRegularizedDogVaradhanLandmarks(g16, 15, 8, 42);
const emb16 = new Float64Array(spec.emb16);
const pSrc = new Uint32Array(spec.packed_src);
const pTgt = new Uint32Array(spec.packed_tgt);
const pD = new Float32Array(spec.packed_d);
const pW = new Float32Array(spec.packed_w);
const pH = new Uint8Array(spec.packed_h);

const sgdMakers = [
  ['full', () => mod.SgdSolver.full(g16, { iterations: 4, seed: 42 })],
  ['from_pairs', () => mod.SgdSolver.fromPairs(16, pSrc, pTgt, pD, null, null, null, { iterations: 4, seed: 42 })],
  ['from_pairs_with_hinges', () => mod.SgdSolver.fromPairsWithHinges(16, pSrc, pTgt, pD, null, null, null, new Uint32Array([0]), new Uint32Array([15]), new Float32Array([2.5]), null, null, null, { iterations: 4, seed: 42 })],
  ['from_packed_symmetric', () => mod.SgdSolver.fromPackedSymmetric(16, pSrc, pTgt, pD, pW, pH, { iterations: 4, seed: 42 })],
  ['sparse_barrier_from_landmarks', () => mod.SgdSolver.sparseBarrierFromLandmarks(g16, lm16, null, { iterations: 4, seed: 42 })],
  ['sparse_barrier_dog', () => mod.SgdSolver.sparseBarrierDog(g16, { degree: 15, numLandmarks: 8, iterations: 4, seed: 42 })],
  ['streaming_negatives', () => mod.SgdSolver.streamingNegatives(g16, { negativesPerNode: 2, iterations: 4, seed: 42 })],
  ['random_pair', () => mod.SgdSolver.randomPair(g16, { pairsPerNode: 3, iterations: 4, seed: 42 })],
  ['omega', () => mod.SgdSolver.omega(g16, emb16, 8, { k: 3, iterations: 4, seed: 42 })],
  ['sgd2', () => mod.SgdSolver.sgd2(g16, { iterations: 4, batchSize: 8, seed: 42 })],
  ['pacmap_graph', () => mod.SgdSolver.pacmapGraph(g16, { nNeighbors: 4, phaseIterations: [2, 2, 2], seed: 42 })],
  ['pacmap_graph_with_landmarks', () => mod.SgdSolver.pacmapGraphWithLandmarks(g16, lm16, { nNeighbors: 4, phaseIterations: [2, 2, 2], seed: 42 })],
  ['snap_t_fdp', () => mod.SgdSolver.snapTFdp(g16, { iterations: 4, negativesPerEdge: 2, seed: 42 })],
  ['ortmann_sparse_stress', () => mod.SgdSolver.ortmannSparseStress(g16, { numPivots: 4, iterations: 4, seed: 42 })],
  ['largevis', () => mod.SgdSolver.largevis(g16, { iterations: 4, negativeSamples: 2, seed: 42 })],
  ['drgraph', () => mod.SgdSolver.drgraph(g16, { iterations: 4, negativeSamples: 2, seed: 42 })],
  ['graph_cne', () => mod.SgdSolver.graphCne(g16, { iterations: 4, negativeSamples: 2, seed: 42 })],
];
const sgd17 = {};
for (const [k, mk] of sgdMakers) {
  const s = mk();
  assert.equal(s.supportsManifold, true);
  const ml = mod.ManifoldLayout2d.fromCoords('hyperbolic_poincare', poinC);
  s.run(ml);
  assert.ok(ml.isAllFinite());
  sgd17[k] = Array.from(ml.coordsView());
  ml.free?.(); s.free?.();
}

const runMani = (mk, kind, base, chunk) => {
  const sr = mk();
  const ss = mk();
  const lr = mod.ManifoldLayout2d.fromCoords(kind, base);
  const ls = mod.ManifoldLayout2d.fromCoords(kind, base);
  sr.run(lr);
  ss.runSteps(ls, chunk);
  while (!ss.isFinished) ss.step(ls);
  const cr = Array.from(lr.coordsView());
  assert.deepEqual(cr, Array.from(ls.coordsView()));
  lr.free?.(); ls.free?.(); sr.free?.(); ss.free?.();
  return cr;
};
const mani = {
  tsnet_sph: runMani(() => mod.TsNetSolver.pure2Hop(g16, { backend: 'exact', stageIterations: [2, 2, 2], seed: 42 }), 'spherical', sphC, 2),
  gumap_tor: runMani(() => mod.GuMapSolver.fromGraphDog(g16, { max2hopPerNode: 16, degree: 15, numLandmarks: 8, iterations: 5, learningRate: 0.2, seed: 42 }), 'torus_2d', torC, 2),
  sm_lor: runMani(() => mod.StressMajorizationSolver.fromGraph(g16, { maxIterations: 5 }), 'hyperbolic_lorentz', lorC, 2),
  kk_poin: runMani(() => mod.KamadaKawaiSolver.fromGraph(g16, { maxIterations: 5, eps: 1e-3, maxInnerIters: 20 }), 'hyperbolic_poincare', poinC, 2),
  fr_sph: runMani(() => new mod.FruchtermanReingoldSolver(g16, { iterations: 5, backend: 'exact', seed: 42 }), 'spherical', sphC, 2),
};

const svW2 = mod.SgdSolver.full(g16, { iterations: 6, seed: 42 });
const lW2 = mod.Layout2d.fromCoords(c16);
svW2.runScheduleWindow(lW2, 0, 3);
svW2.resetSchedule({ iterations: 4, maxMu: 0.05, seed: 99 });
svW2.runScheduleWindow(lW2, 0, 2);
const win2d = Array.from(lW2.coordsView());
lW2.free?.(); svW2.free?.();

const svWm = mod.SgdSolver.full(g16, { iterations: 6, seed: 42 });
const lWm = mod.ManifoldLayout2d.fromCoords('hyperbolic_poincare', poinC);
svWm.runScheduleWindow(lWm, 0, 3);
svWm.resetSchedule({ iterations: 4, maxMu: 0.05, seed: 99 });
svWm.runScheduleWindow(lWm, 0, 2);
const winMani = Array.from(lWm.coordsView());
lWm.free?.(); svWm.free?.();

const mlPoinX = mod.ManifoldLayout2d.fromCoords('hyperbolic_poincare', new Float32Array([0, 0, 0.5, 0, 0.5, 0.5, 0, 0.5]));
const mlSphX = mod.ManifoldLayout2d.fromCoords('spherical', new Float32Array([0, 1.2, 0.6, 1.2, 0.6, 1.9, 0, 1.9]));
const mlTorX = mod.ManifoldLayout2d.fromCoords('torus_2d', new Float32Array([0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9]));
mlTorX.setTorusExtents(2.0, 0.5);

const sphCn = mod.sphericalCrossingNumber(g4, mlSphX);
const sphCe = mod.crossingEdgesSpherical(g4, mlSphX).map((p) => [p.edgeA, p.edgeB]);
const hypCn = mod.hyperbolicCrossingNumber(g4, mlPoinX);
const hypCe = mod.crossingEdgesHyperbolic(g4, mlPoinX).map((p) => [p.edgeA, p.edgeB]);
const torCn = mod.torusCrossingNumber(g4, mlTorX);
const torCeObjs = mod.crossingEdgesTorus(g4, mlTorX);
const torCe0 = torCeObjs.length > 0 ? {
  edgeA: torCeObjs[0].edgeA, edgeB: torCeObjs[0].edgeB, u1: torCeObjs[0].u1, v1: torCeObjs[0].v1,
  u2: torCeObjs[0].u2, v2: torCeObjs[0].v2, cosSq: torCeObjs[0].cosSq, angleRad: torCeObjs[0].angleRad,
  segA: Array.from(torCeObjs[0].segA()), segB: Array.from(torCeObjs[0].segB()),
} : null;
const torSegsEdge = Array.from(mod.torusEdgeSegments(0.9, 0.2, 0.1, 0.8));
const torSegsLay = Array.from(mod.torusLayoutEdgeSegments(g4, mlTorX));
const mTorSpec = mod.torusSpectralPhaseInit(g16, 42);
const torPan = Array.from(mTorSpec.torusAutoPan(g16).coordsView());
const mPoin = mod.ManifoldLayout2d.fromCoords('hyperbolic_poincare', poinC);
const gdPoin = mod.geodesicDistortion(g16, mPoin);
const evGd = mod.evaluateMetrics(g16, mPoin, ['geodesic_distortion']).geodesicDistortion;

const mlAnti = mod.ManifoldLayout2d.fromCoords('spherical', new Float32Array([0, Math.PI * 0.5, Math.PI, Math.PI * 0.5]));
assert.ok(Number.isFinite(mlAnti.geodesicDistance(0, 1)));
const mlClamp = mod.ManifoldLayout2d.fromCoords('hyperbolic_poincare', new Float32Array([0.99999, 0, 0, 0.99999]));
mlClamp.projectAll();
assert.ok(mlClamp.isAllFinite());
assertNonPanicError('wasm/manifold/bad_scale', () => mRes.setScale(0.0));

mlClamp.free?.(); mlAnti.free?.(); mPoin.free?.(); mTorSpec.free?.();
mlTorX.free?.(); mlSphX.free?.(); mlPoinX.free?.(); lm16.free?.(); mLor.free?.(); mRes.free?.(); g4.free?.(); g16.free?.();
console.log(JSON.stringify({
  rescaledPoin, lorToPoin, lExp, lLog, lDist, lPt, lMid, pMid,
  sgd17, mani, win2d, winMani,
  sphCn, sphCe, hypCn, hypCe, torCn, torCe0, torSegsEdge, torSegsLay,
  torInit: Array.from(mTorSpec.coords ? [] : torC),
  torPan, gdPoin, evGd,
}));
"""
    ws = run_node_json(wasm_pkg, node_body, spec_payload)

    max_f32_diff = 0.0
    max_f64_diff = 0.0
    max_f32_diff = max(max_f32_diff, check_array_close("mani/rescale_geodesic", rescaled_poin_py, ws["rescaledPoin"], 1e-5))
    max_f32_diff = max(max_f32_diff, check_array_close("mani/lor_to_poin", lor_to_poin_py, ws["lorToPoin"], 1e-5))
    max_f64_diff = max(max_f64_diff, check_array_close("mani/l_exp", l_exp, ws["lExp"], 1e-12))
    max_f64_diff = max(max_f64_diff, check_array_close("mani/l_log", l_log, ws["lLog"], 1e-12))
    max_f64_diff = max(max_f64_diff, check_array_close("mani/l_dist", [l_dist], [ws["lDist"]], 1e-12))
    max_f64_diff = max(max_f64_diff, check_array_close("mani/l_pt", l_pt, ws["lPt"], 1e-12))
    max_f64_diff = max(max_f64_diff, check_array_close("mani/l_mid", l_mid, ws["lMid"], 1e-12))
    max_f64_diff = max(max_f64_diff, check_array_close("mani/p_mid", p_mid, ws["pMid"], 1e-12))

    for k, arr in sgd17_py.items():
        max_f32_diff = max(max_f32_diff, check_array_close(f"mani/sgd17/{k}", arr, ws["sgd17"][k], 1e-4))
    for k, arr in mani_py.items():
        max_f32_diff = max(max_f32_diff, check_array_close(f"mani/solver/{k}", arr, ws["mani"][k], 1e-4))

    max_f32_diff = max(max_f32_diff, check_array_close("mani/win_2d", win_2d_py, ws["win2d"], 1e-5))
    max_f32_diff = max(max_f32_diff, check_array_close("mani/win_mani", win_mani_py, ws["winMani"], 1e-5))

    assert sph_cn_py == ws["sphCn"]
    assert [[p["edge_a"], p["edge_b"]] for p in sph_ce_py] == ws["sphCe"]
    assert hyp_cn_py == ws["hypCn"]
    assert [[p["edge_a"], p["edge_b"]] for p in hyp_ce_py] == ws["hypCe"]
    assert tor_cn_py == ws["torCn"]
    tc_py = tor_ce_py[0]
    tc_ws = ws["torCe0"]
    assert (tc_py["edge_a"], tc_py["edge_b"], tc_py["u1"], tc_py["v1"], tc_py["u2"], tc_py["v2"]) == (
        tc_ws["edgeA"], tc_ws["edgeB"], tc_ws["u1"], tc_ws["v1"], tc_ws["u2"], tc_ws["v2"]
    )
    max_f64_diff = max(max_f64_diff, check_array_close("mani/tor_cos_sq", [tc_py["cos_sq"]], [tc_ws["cosSq"]], 1e-12))
    max_f64_diff = max(max_f64_diff, check_array_close("mani/tor_seg_a", tc_py["seg_a"], tc_ws["segA"], 1e-6))
    max_f64_diff = max(max_f64_diff, check_array_close("mani/tor_seg_b", tc_py["seg_b"], tc_ws["segB"], 1e-6))
    max_f64_diff = max(max_f64_diff, check_array_close("mani/tor_segs_edge", tor_segs_edge_py, ws["torSegsEdge"], 1e-12))
    max_f64_diff = max(max_f64_diff, check_array_close("mani/tor_segs_lay", tor_segs_lay_py, ws["torSegsLay"], 1e-6))
    max_f32_diff = max(max_f32_diff, check_array_close("mani/tor_pan", tor_pan_py, ws["torPan"], 1e-5))
    max_f64_diff = max(max_f64_diff, check_array_close("mani/gd_poin", [gd_poin_py], [ws["gdPoin"]], 1e-10))
    max_f64_diff = max(max_f64_diff, check_array_close("mani/eval_gd", [ev_gd_py], [ws["evGd"]], 1e-10))

    return [{
        "case": "cross_target_manifold_engines_crossings_py_vs_wasm",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "max_f32_diff": round(max_f32_diff, 12),
        "max_f64_diff": round(max_f64_diff, 12),
    }]
