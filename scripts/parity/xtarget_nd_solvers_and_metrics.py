"""Cross-target Python <-> WASM parity for N-D solvers, spatial trees, and N-D metrics."""

from __future__ import annotations

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


def run_cross_target_nd_solvers_metrics_parity(
    eg: Any, wasm_pkg: Any
) -> list[dict[str, Any]]:
    """Verify Python float32 <-> WASM f32 parity across 11 N-D solvers, ArenaOctree3d, DynamicKdTree, and N-D metrics."""
    import numpy as np

    if not hasattr(eg, "LayoutNd") or not hasattr(eg, "ArenaOctree3d"):
        return []
    if resolve_wasm_entries(wasm_pkg) is None:
        return []

    t0 = time.perf_counter()
    g16_edges = make_grid_edges(4, 4)
    g16 = eg.CsrGraph.from_edges(16, g16_edges, directed=False, sanitize=True)

    # Deterministic 3D and 4D initial coordinates in float32.
    c2 = np.array(make_asymmetric_coords(16, scale=1.1), dtype=np.float32)
    z3 = np.array([0.15 * np.sin(0.7 * i + 0.3) for i in range(16)], dtype=np.float32)
    w4 = np.array([0.12 * np.cos(0.5 * i - 0.2) for i in range(16)], dtype=np.float32)
    init3_f32 = np.column_stack([c2[:, 0], c2[:, 1], z3]).astype(np.float32)
    init4_f32 = np.column_stack([c2[:, 0], c2[:, 1], z3, w4]).astype(np.float32)
    masses16 = [1.0 + 0.1 * (i % 5) for i in range(16)]
    labels16 = [0] * 8 + [1] * 8

    # LayoutNd lifecycle checks in Python.
    nd0 = eg.LayoutNd.zeros(4, 3, dtype="float32")
    assert nd0.dims == 3 and nd0.num_nodes == 4 and len(nd0) == 4
    nd0.set(1, 2, 0.75)
    assert abs(nd0.get(1, 2) - 0.75) < 1e-6
    nd_init = eg.LayoutNd.init(16, 3, seed=42, scale=0.8, dtype="float32")
    nd_cp = nd_init.copy()
    nd_cp.normalize_std(1.0)
    assert nd_cp.is_all_finite()

    # 11 N-D solvers (dims=3) + dims=4 subset in Python.
    def run_py_pair(mk: Any, base: Any, step_chunk: int) -> tuple[Any, Any]:
        sr = mk()
        ss = mk()
        lr = eg.LayoutNd.from_coords(base)
        ls = eg.LayoutNd.from_coords(base)
        sr.run(lr)
        ss.run_steps(ls, step_chunk)
        while not ss.is_finished:
            ss.step(ls)
        cr = lr.coords(copy=True)
        cs = ls.coords(copy=True)
        assert np.array_equal(cr, cs), "Python N-D step() != run()"
        return sr, cr

    py_solvers: dict[str, Any] = {}
    solver_makers_3d: list[tuple[str, Any, int]] = [
        ("sgd", lambda: eg.SgdSolver.full(g16, iterations=6, seed=42, dtype="float32"), 2),
        (
            "tsnet",
            lambda: eg.TsNetSolver.pure_2hop(
                g16, backend="exact", stage_iterations=(3, 3, 3), seed=42, dtype="float32"
            ),
            3,
        ),
        (
            "gumap",
            lambda: eg.GuMapSolver.from_graph_dog(
                g16, max_2hop_per_node=16, degree=15, num_landmarks=8, iterations=8, learning_rate=0.2, seed=42, dtype="float32"
            ),
            3,
        ),
        ("sm", lambda: eg.StressMajorizationSolver.from_graph(g16, max_iterations=10, dtype="float32"), 3),
        ("kk", lambda: eg.KamadaKawaiSolver.from_graph(g16, max_iterations=10, eps=1e-3, max_inner_iters=30, dtype="float32"), 3),
        ("maxent", lambda: eg.MaxentStressSolver(g16, stages=2, steps_per_stage=3, use_barnes_hut=True, seed=42, dtype="float32"), 2),
        ("fr", lambda: eg.FruchtermanReingoldSolver(g16, iterations=6, backend="barnes_hut", seed=42, dtype="float32"), 2),
        ("linlog", lambda: eg.LinLogSolver(g16, iterations=6, use_barnes_hut=True, seed=42, dtype="float32"), 2),
        ("fa2", lambda: eg.ForceAtlas2Solver(g16, iterations=6, use_barnes_hut=True, seed=42, dtype="float32"), 2),
        ("yh", lambda: eg.YifanHuSolver(g16, max_iterations_per_level=6, min_coarsen_size=6, seed=42, dtype="float32"), 3),
    ]
    for name, mk, chunk in solver_makers_3d:
        _, coords = run_py_pair(mk, init3_f32, chunk)
        py_solvers[name] = coords.ravel()

    fd3_r = eg.FdebSolver(g16, cycles=1, steps_per_cycle=4, dtype="float32")
    fd3_s = eg.FdebSolver(g16, cycles=1, steps_per_cycle=4, dtype="float32")
    l_fd3 = eg.LayoutNd.from_coords(init3_f32)
    fd3_r.run(l_fd3)
    fd3_s.step(l_fd3)
    fd3_s.run(l_fd3)
    assert fd3_r.point_dims == 3
    assert np.array_equal(fd3_r.points(), fd3_s.points())
    py_solvers["fdeb3_pts"] = fd3_r.points().ravel()
    py_solvers["fdeb3_z"] = fd3_r.points_z()
    py_solvers["fdeb3_ax1"] = fd3_r.points_axis(1)

    # 4D checks (SgdSolver, StressMajorizationSolver, FdebSolver)
    _, sgd4_c = run_py_pair(lambda: eg.SgdSolver.full(g16, iterations=5, seed=42, dtype="float32"), init4_f32, 2)
    _, sm4_c = run_py_pair(lambda: eg.StressMajorizationSolver.from_graph(g16, max_iterations=6, dtype="float32"), init4_f32, 2)
    fd4 = eg.FdebSolver(g16, cycles=1, steps_per_cycle=3, dtype="float32")
    fd4.run(eg.LayoutNd.from_coords(init4_f32))
    assert fd4.point_dims == 4
    py_solvers["sgd4"] = sgd4_c.ravel()
    py_solvers["sm4"] = sm4_c.ravel()
    py_solvers["fdeb4_pts"] = fd4.points().ravel()
    py_solvers["fdeb4_ax3"] = fd4.points_axis(3)

    # ArenaOctree3d (exact theta=0.0 vs BH theta=0.5, unit & weighted)
    l_oct = eg.LayoutNd.from_coords(init3_f32)
    oct_u = eg.ArenaOctree3d(dtype="float32")
    oct_u.build(l_oct)
    oct_w = eg.ArenaOctree3d(dtype="float32")
    oct_w.build(l_oct, masses=masses16)
    py_oct = {
        "coul_ex": oct_u.compute_coulomb_repulsion(theta=0.0, eps_sq=1e-6).ravel(),
        "coul_bh": oct_u.compute_coulomb_repulsion(theta=0.5, eps_sq=1e-6).ravel(),
        "coul_w": oct_w.compute_coulomb_repulsion(theta=0.5, eps_sq=1e-6).ravel(),
        "maxent": oct_u.compute_maxent_repulsion(1.0, theta=0.5, eps_sq=1e-6).ravel(),
        "kl": oct_u.compute_kl_repulsion(theta=0.5)[0].ravel(),
        "entropy": oct_u.compute_entropy_repulsion(theta=0.5, epsilon_r=0.05).ravel(),
    }

    # DynamicKdTree (3D and 4D)
    kd3 = eg.DynamicKdTree(l_oct)
    kd4 = eg.DynamicKdTree(eg.LayoutNd.from_coords(init4_f32))
    assert kd3.dims == 3 and kd3.num_nodes == 16 and len(kd3) == 16
    assert kd4.dims == 4 and kd4.num_nodes == 16
    q3 = [0.1, -0.2, 0.05]
    knn3_idx, knn3_d2 = kd3.k_nearest(q3, 4)
    kex_idx, kex_d2 = kd3.k_nearest_excluding_self(3, 3)
    near3 = kd3.nearest_excluding_self(3)
    assert near3 is not None
    rad_idx, rad_d2 = kd3.within_radius(q3, 0.9)
    rsq_idx, rsq_d2 = kd3.within_radius_sq(q3, 0.81)
    knn4_idx, knn4_d2 = kd4.k_nearest([0.1, -0.2, 0.05, 0.0], 3)

    # N-D evaluate_metrics + standalone metrics
    l3_eval = eg.LayoutNd.from_coords(init3_f32)
    nd_targets = [
        "sns", "sampled_sns", "np1", "np2", "node_resolution", "drng", "dgg",
        "ideal_edge_lengths", "layout_degeneracy", "landmark_stratified_rta",
        "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score",
        "distance_consistency", "neighborhood_hit",
    ]
    ev3_py = eg.evaluate_metrics(
        g16, l3_eval, nd_targets,
        num_pivots=4, num_samples=64, seed=42, labels=labels16,
        num_landmarks=4, num_triplets=50, min_degree=1, neighborhood_hit_k=3,
    ).to_dict()
    gg3_py = eg.gabriel_graph_edges(l3_eval).ravel()
    rng3_py = eg.relative_neighborhood_edges(l3_eval).ravel()
    deg3_py = eg.layout_degeneracy(g16, l3_eval)

    # Invalid-dim adversarial non-panic checks in Python
    assert_non_panic_error("py/nd/dims_0", lambda: eg.LayoutNd.zeros(4, 0, dtype="float32"))
    assert_non_panic_error("py/octree/4d", lambda: oct_u.build(eg.LayoutNd.from_coords(init4_f32)))
    assert_non_panic_error("py/kdtree/bad_q", lambda: kd3.k_nearest([0.0, 1.0], 2))

    spec_payload = {
        "g16_edges": [int(v) for e in g16_edges for v in e],
        "init3": [float(v) for v in init3_f32.ravel()],
        "init4": [float(v) for v in init4_f32.ravel()],
        "masses16": masses16,
        "labels16": labels16,
    }

    node_body = """
const g16 = mod.CsrGraph.fromEdgeList(16, new Uint32Array(spec.g16_edges));
const init3 = new Float32Array(spec.init3);
const init4 = new Float32Array(spec.init4);

const l0 = mod.LayoutNd.zeros(4, 3);
assert.equal(l0.dims, 3);
assert.equal(l0.numNodes, 4);
l0.set(1, 2, 0.75);
assert.ok(Math.abs(l0.get(1, 2) - 0.75) < 1e-6);
const lInit = mod.LayoutNd.init(16, 3, 42, 0.8);
lInit.normalizeStd(1.0);
assert.ok(lInit.isAllFinite());
lInit.free?.(); l0.free?.();

const runPair = (mk, base, dims, chunk) => {
  const sr = mk();
  const ss = mk();
  const lr = mod.LayoutNd.fromCoords(16, dims, base);
  const ls = mod.LayoutNd.fromCoords(16, dims, base);
  sr.run(lr);
  ss.runSteps(ls, chunk);
  while (!ss.isFinished) ss.step(ls);
  const cr = Array.from(lr.coordsView());
  const cs = Array.from(ls.coordsView());
  assert.deepEqual(cr, cs);
  lr.free?.(); ls.free?.(); sr.free?.(); ss.free?.();
  return cr;
};

const solvers = {
  sgd: runPair(() => mod.SgdSolver.full(g16, { iterations: 6, seed: 42 }), init3, 3, 2),
  tsnet: runPair(() => mod.TsNetSolver.pure2Hop(g16, { backend: 'exact', stageIterations: [3, 3, 3], seed: 42 }), init3, 3, 3),
  gumap: runPair(() => mod.GuMapSolver.fromGraphDog(g16, { max2hopPerNode: 16, degree: 15, numLandmarks: 8, iterations: 8, learningRate: 0.2, seed: 42 }), init3, 3, 3),
  sm: runPair(() => mod.StressMajorizationSolver.fromGraph(g16, { maxIterations: 10 }), init3, 3, 3),
  kk: runPair(() => mod.KamadaKawaiSolver.fromGraph(g16, { maxIterations: 10, eps: 1e-3, maxInnerIters: 30 }), init3, 3, 3),
  maxent: runPair(() => new mod.MaxentStressSolver(g16, { stages: 2, stepsPerStage: 3, useBarnesHut: true, seed: 42 }), init3, 3, 2),
  fr: runPair(() => new mod.FruchtermanReingoldSolver(g16, { iterations: 6, backend: 'barnes_hut', seed: 42 }), init3, 3, 2),
  linlog: runPair(() => new mod.LinLogSolver(g16, { iterations: 6, useBarnesHut: true, seed: 42 }), init3, 3, 2),
  fa2: runPair(() => new mod.ForceAtlas2Solver(g16, { iterations: 6, useBarnesHut: true, seed: 42 }), init3, 3, 2),
  yh: runPair(() => new mod.YifanHuSolver(g16, { maxIterationsPerLevel: 6, minCoarsenSize: 6, seed: 42 }), init3, 3, 3),
  sgd4: runPair(() => mod.SgdSolver.full(g16, { iterations: 5, seed: 42 }), init4, 4, 2),
  sm4: runPair(() => mod.StressMajorizationSolver.fromGraph(g16, { maxIterations: 6 }), init4, 4, 2),
};

{
  const fd3R = new mod.FdebSolver(g16, { cycles: 1, stepsPerCycle: 4 });
  const fd3S = new mod.FdebSolver(g16, { cycles: 1, stepsPerCycle: 4 });
  const l3 = mod.LayoutNd.fromCoords(16, 3, init3);
  fd3R.run(l3); fd3S.step(l3); fd3S.run(l3);
  assert.equal(fd3R.pointDims, 3);
  solvers.fdeb3_pts = Array.from(fd3R.toInterleavedPointsNd());
  assert.deepEqual(solvers.fdeb3_pts, Array.from(fd3R.pointsNd()));
  solvers.fdeb3_z = Array.from(fd3R.pointsZ());
  solvers.fdeb3_ax1 = Array.from(fd3R.pointsAxis(1));
  l3.free?.(); fd3R.free?.(); fd3S.free?.();
}
{
  const fd4 = new mod.FdebSolver(g16, { cycles: 1, stepsPerCycle: 3 });
  const l4 = mod.LayoutNd.fromCoords(16, 4, init4);
  fd4.run(l4);
  assert.equal(fd4.pointDims, 4);
  solvers.fdeb4_pts = Array.from(fd4.toInterleavedPointsNd());
  solvers.fdeb4_ax3 = Array.from(fd4.pointsAxis(3));
  l4.free?.(); fd4.free?.();
}

const lOct = mod.LayoutNd.fromCoords(16, 3, init3);
const l4 = mod.LayoutNd.fromCoords(16, 4, init4);
const octU = new mod.ArenaOctree3d();
octU.build(lOct);
const octW = new mod.ArenaOctree3d();
octW.buildWeighted(lOct, new Float64Array(spec.masses16));
const oct = {
  coul_ex: Array.from(octU.computeCoulombRepulsion(0.0, 1e-6)),
  coul_bh: Array.from(octU.computeCoulombRepulsion(0.5, 1e-6)),
  coul_w: Array.from(octW.computeCoulombRepulsion(0.5, 1e-6)),
  maxent: Array.from(octU.computeMaxentRepulsion(0.5, 1.0, 1e-6)),
  kl: Array.from(octU.computeKlRepulsion(0.5)),
  entropy: Array.from(octU.computeEntropyRepulsion(0.5, 0.05)),
};

const kd3 = new mod.DynamicKdTree(lOct);
const kd4 = new mod.DynamicKdTree(l4);
assert.equal(kd3.dims, 3);
assert.equal(kd3.numNodes, 16);
assert.equal(kd4.dims, 4);
const q3 = new Float32Array([0.1, -0.2, 0.05]);
const knn3 = kd3.kNearest(q3, 4);
const kex3 = kd3.kNearestExcludingSelf(3, 3);
const near3 = kd3.nearestExcludingSelf(3);
const rad3 = kd3.withinRadius(q3, 0.9);
const rsq3 = kd3.withinRadiusSq(q3, 0.81);
const knn4 = kd4.kNearest(new Float32Array([0.1, -0.2, 0.05, 0.0]), 3);

const labels16 = new Uint32Array(spec.labels16);
const ev3 = mod.evaluateMetrics(
  g16, lOct,
  [
    'sns', 'sampled_sns', 'np1', 'np2', 'node_resolution', 'drng', 'dgg',
    'ideal_edge_lengths', 'layout_degeneracy', 'landmark_stratified_rta',
    'silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score',
    'distance_consistency', 'neighborhood_hit',
  ],
  { numPivots: 4, numSamples: 64, seed: 42, labels: labels16, numLandmarks: 4, numTriplets: 50, minDegree: 1, neighborhoodHitK: 3 },
).toRecord();
const gg3 = Array.from(mod.gabrielGraphEdges(lOct));
const rng3 = Array.from(mod.relativeNeighborhoodEdges(lOct));
const deg3 = mod.layoutDegeneracy(g16, lOct).toRecord();

assertNonPanicError('wasm/nd/dims_0', () => mod.LayoutNd.zeros(4, 0));
assertNonPanicError('wasm/octree/4d', () => octU.build(l4));
assertNonPanicError('wasm/kdtree/bad_q', () => kd3.kNearest(new Float32Array([0, 1]), 2));

kd4.free?.(); kd3.free?.(); octW.free?.(); octU.free?.(); l4.free?.(); lOct.free?.(); g16.free?.();
console.log(JSON.stringify({
  solvers, oct,
  kd: {
    pt3: Array.from(kd3.point ? [] : []),
    knn3Idx: knn3.map((h) => h.index), knn3D2: knn3.map((h) => h.distSq),
    kexIdx: kex3.map((h) => h.index), kexD2: kex3.map((h) => h.distSq),
    near3,
    radIdx: rad3.map((h) => h.index), radD2: rad3.map((h) => h.distSq),
    rsqIdx: rsq3.map((h) => h.index), rsqD2: rsq3.map((h) => h.distSq),
    knn4Idx: knn4.map((h) => h.index), knn4D2: knn4.map((h) => h.distSq),
  },
  ev3, gg3, rng3, deg3,
}));
"""
    # Query pt3 before kd3 is freed by adjusting node_body slightly:
    node_body = node_body.replace(
        "pt3: Array.from(kd3.point ? [] : []),",
        "pt3: Array.from(pt3Arr),",
    ).replace(
        "const near3 = kd3.nearestExcludingSelf(3);",
        "const pt3Arr = kd3.point(3);\nconst near3 = kd3.nearestExcludingSelf(3);",
    )
    ws = run_node_json(wasm_pkg, node_body, spec_payload)

    max_f32_diff = 0.0
    max_f64_diff = 0.0
    for name, arr in py_solvers.items():
        max_f32_diff = max(max_f32_diff, check_array_close(f"nd/solver/{name}", arr, ws["solvers"][name], 1e-4))
    for name, arr in py_oct.items():
        max_f32_diff = max(max_f32_diff, check_array_close(f"nd/octree/{name}", arr, ws["oct"][name], 1e-4))

    kd_ws = ws["kd"]
    max_f64_diff = max(max_f64_diff, check_array_close("nd/kd/point", kd3.point(3), kd_ws["pt3"], 1e-6))
    assert list(knn3_idx) == kd_ws["knn3Idx"]
    max_f64_diff = max(max_f64_diff, check_array_close("nd/kd/knn3_d2", knn3_d2, kd_ws["knn3D2"], 1e-6))
    assert list(kex_idx) == kd_ws["kexIdx"]
    max_f64_diff = max(max_f64_diff, check_array_close("nd/kd/kex_d2", kex_d2, kd_ws["kexD2"], 1e-12))
    assert near3[0] == kd_ws["near3"]["index"]
    assert abs(near3[1] - kd_ws["near3"]["distSq"]) < 1e-12
    assert list(rad_idx) == kd_ws["radIdx"]
    assert list(rsq_idx) == kd_ws["rsqIdx"]
    assert list(knn4_idx) == kd_ws["knn4Idx"]
    max_f64_diff = max(max_f64_diff, check_array_close("nd/kd/rad_d2", rad_d2, kd_ws["radD2"], 1e-6))
    max_f64_diff = max(max_f64_diff, check_array_close("nd/kd/rsq_d2", rsq_d2, kd_ws["rsqD2"], 1e-6))
    max_f64_diff = max(max_f64_diff, check_array_close("nd/kd/knn4_d2", knn4_d2, kd_ws["knn4D2"], 1e-6))

    assert list(gg3_py) == ws["gg3"]
    assert list(rng3_py) == ws["rng3"]
    assert abs(float(deg3_py["excess_duplicate_ratio"]) - ws["deg3"]["excessDuplicateRatio"]) < 1e-12
    ev_pairs = (
        ("scale_normalized_stress", "scaleNormalizedStress"),
        ("sampled_scale_normalized_stress", "sampledScaleNormalizedStress"),
        ("neighborhood_preservation_1hop", "neighborhoodPreservation1Hop"),
        ("neighborhood_preservation_2hop", "neighborhoodPreservation2Hop"),
        ("node_resolution", "nodeResolution"),
        ("relative_neighborhood_preservation", "relativeNeighborhoodPreservation"),
        ("gabriel_graph_preservation", "gabrielGraphPreservation"),
        ("ideal_edge_lengths", "idealEdgeLengths"),
        ("layout_degeneracy", "layoutDegeneracy"),
        ("landmark_stratified_rta", "landmarkStratifiedRta"),
        ("silhouette_score", "silhouetteScore"),
        ("calinski_harabasz_score", "calinskiHarabaszScore"),
        ("davies_bouldin_score", "daviesBouldinScore"),
        ("distance_consistency", "distanceConsistency"),
        ("neighborhood_hit", "neighborhoodHit"),
    )
    for pk, wk in ev_pairs:
        max_f64_diff = max(max_f64_diff, check_array_close(f"nd/eval/{pk}", [ev3_py[pk]], [ws["ev3"][wk]], 1e-10))

    return [{
        "case": "cross_target_nd_solvers_metrics_py_vs_wasm",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "max_f32_diff": round(max_f32_diff, 12),
        "max_f64_diff": round(max_f64_diff, 12),
    }]
