"""Cross-target Python <-> WASM parity for RN-tsNET, TwinDeclutterer, Dijkstra, datasets, and CsrGraph helpers."""

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


def run_cross_target_rn_tsnet_declutter_dijkstra_parity(
    eg: Any, wasm_pkg: Any
) -> list[dict[str, Any]]:
    """Verify Python <-> WASM parity for TsNet Auto/stages/RW, TwinDeclutterer, Dijkstra, datasets, and helpers."""
    import numpy as np

    if not hasattr(eg, "TwinDeclutterer") or not hasattr(eg, "dijkstra_single_source"):
        return []
    if resolve_wasm_entries(wasm_pkg) is None:
        return []

    t0 = time.perf_counter()

    # 1. TsNet Auto (N <= 256 -> exact vs N > 256 -> linear_fft), stages, two_step_random_walk + max_bridge_degree
    g16_edges = make_grid_edges(4, 4)
    g16 = eg.CsrGraph.from_edges(16, g16_edges, directed=False, sanitize=True)
    g260_edges = make_grid_edges(13, 20)
    g260 = eg.CsrGraph.from_edges(260, g260_edges, directed=False, sanitize=True)
    init16_f32 = np.array(make_asymmetric_coords(16, scale=0.6), dtype=np.float32)
    init260_f32 = np.array(make_asymmetric_coords(260, scale=0.6), dtype=np.float32)

    ts_auto_small = eg.TsNetSolver.pure_2hop(
        g16, backend="auto", iterations_stage1=2, iterations_stage2=2, iterations_stage3=2, seed=42, dtype="float32"
    )
    assert ts_auto_small.resolved_backend == "exact"
    l_auto_sm = eg.Layout2d.from_coords(init16_f32)
    ts_auto_small.run(l_auto_sm)
    assert ts_auto_small.total_fft_2d_calls == 0

    ts_auto_big = eg.TsNetSolver.pure_2hop(
        g260, backend="auto", iterations_stage1=1, iterations_stage2=1, iterations_stage3=1,
        learning_rate_divisor=3.0, seed=42, dtype="float32",
    )
    assert ts_auto_big.resolved_backend == "linear_fft"
    l_auto_bg = eg.Layout2d.from_coords(init260_f32)
    ts_auto_big.run(l_auto_bg)
    assert ts_auto_big.total_fft_2d_calls > 0

    # TsNetStage + from_probabilities_with_stages + SparseProbabilities.from_csr
    probs_rw = eg.graph_12hop_row_normalized_probabilities(
        g16, w2=0.4, max_2hop_per_node=8, degree_beta=0.25, mode="two_step_random_walk", max_bridge_degree=3, seed=42
    )
    probs_csr = eg.SparseProbabilities.from_csr(16, probs_rw.indptr(), probs_rw.indices(), probs_rw.values())
    assert list(probs_csr.indptr()) == list(probs_rw.indptr())
    stages_py = [
        eg.TsNetStage(2, exaggeration=4.0, lambda_kl=1.0, lambda_c=0.1, lambda_r=0.0),
        eg.TsNetStage(2, exaggeration=1.0, lambda_kl=1.0, lambda_c=0.0, lambda_r=0.0),
    ]
    l_st_py = eg.Layout2d.from_coords(init16_f32)
    ts_st_py = eg.TsNetSolver.from_probabilities_with_stages(probs_csr, stages_py, backend="exact", learning_rate_divisor=3.0, seed=42, dtype="float32")
    ts_st_py.run(l_st_py)

    pairs_rw = eg.extract_sparse_1hop_2hop_pairs(g16, max_2hop_per_node=8, mode="two_step_random_walk", max_bridge_degree=3, seed=42)
    l_rn_rw = eg.Layout2d.from_coords(init16_f32)
    ts_rn_rw = eg.TsNetSolver.from_graph_row_normalized(
        g16, w2=0.4, max_2hop_per_node=8, degree_beta=0.25, mode="two_step_random_walk", max_bridge_degree=3,
        backend="exact", iterations_stage1=2, iterations_stage2=2, iterations_stage3=2, seed=42, dtype="float32",
    )
    ts_rn_rw.run(l_rn_rw)

    # __new__ constructors (SgdSolver, TsNetSolver, GuMapSolver) & TsNet stage_iterations
    knn16, _, _ = eg.build_pure_2hop_knn(g16)
    perps16 = eg.compute_nodewise_2hop_perplexities(g16)
    probs16 = eg.compute_knn_probabilities(knn16, perps16)

    l_sgd_new = eg.Layout2d.from_coords(init16_f32)
    s_sgd_new = eg.SgdSolver(g16, dtype="float32")
    assert s_sgd_new.total_iterations == 60
    s_sgd_new.run(l_sgd_new)

    gumap_init16_f32 = (init16_f32 * 5.0).astype(np.float32)
    l_gm_new = eg.Layout2d.from_coords(gumap_init16_f32)
    l_gm_dog = eg.Layout2d.from_coords(gumap_init16_f32)
    s_gm_new = eg.GuMapSolver(g16, dtype="float32")
    eg.GuMapSolver.from_graph_dog(g16, dtype="float32").step(l_gm_dog)
    s_gm_new.step(l_gm_new)
    assert np.array_equal(l_gm_new.coords(copy=True), l_gm_dog.coords(copy=True))
    l_gm_lr = eg.Layout2d.from_coords(gumap_init16_f32)
    eg.GuMapSolver(g16, learning_rate=0.2, dtype="float32").step(l_gm_lr)

    si_ctors: dict[str, Any] = {
        "new": eg.TsNetSolver(g16, backend="exact", stage_iterations=(2, 3, 2), dtype="float32"),
        "pure": eg.TsNetSolver.pure_2hop(g16, backend="exact", stage_iterations=(2, 3, 2), dtype="float32"),
        "knn": eg.TsNetSolver.from_knn(knn16, perps16, backend="exact", stage_iterations=(2, 3, 2), dtype="float32"),
        "probs": eg.TsNetSolver.from_probabilities(probs16, backend="exact", stage_iterations=(2, 3, 2), dtype="float32"),
        "bfs": eg.TsNetSolver.from_graph_partial_bfs(g16, 15, 15.0, backend="exact", stage_iterations=(2, 3, 2), dtype="float32"),
        "rn": eg.TsNetSolver.from_graph_row_normalized(g16, backend="exact", stage_iterations=(2, 3, 2), dtype="float32"),
    }
    si_coords_py: dict[str, Any] = {}
    for k_si, sv_si in si_ctors.items():
        assert sv_si.total_iterations == 7
        lay_si = eg.Layout2d.from_coords(init16_f32)
        sv_si.run(lay_si)
        si_coords_py[k_si] = lay_si.coords(copy=True).ravel()

    assert_non_panic_error("py/tsnet/bad_stage_iterations_len", lambda: eg.TsNetSolver(g16, stage_iterations=(2, 2)))
    assert_non_panic_error("py/tsnet/bad_stage_iterations_zero", lambda: eg.TsNetSolver(g16, stage_iterations=(0, 2, 2)))
    assert_non_panic_error("py/tsnet/bad_stage_iterations_bool", lambda: eg.TsNetSolver(g16, stage_iterations=(True, 2, 2)))

    # 2. TwinDeclutterer / declutter_twins_2d (f32/f64 + degenerate cases)
    twin_coords_f32 = np.array(
        [[0.0, 0.0], [0.0, 0.0], [1e-7, -1e-7], [1.0, 0.0], [2.0, 1.0], [0.0, 2.0]],
        dtype=np.float32,
    )
    td_r = eg.TwinDeclutterer(iterations=4, dtype="float32")
    td_s = eg.TwinDeclutterer(iterations=4, dtype="float32")
    l_td_r = eg.Layout2d.from_coords(twin_coords_f32)
    l_td_s = eg.Layout2d.from_coords(twin_coords_f32)
    td_r.run(l_td_r)
    td_s.step(l_td_s)
    step1_moved = td_s.last_moved_count
    assert step1_moved >= 2 and td_s.last_cluster_count >= 1
    td_s.run(l_td_s)
    assert np.array_equal(l_td_r.coords(copy=True), l_td_s.coords(copy=True))
    l_td_64 = eg.Layout2d.from_coords(twin_coords_f32.astype(np.float64))
    passes_64 = eg.declutter_twins_2d(l_td_64, iterations=4)
    l_td_32 = eg.Layout2d.from_coords(twin_coords_f32)
    passes_32 = eg.declutter_twins_2d(l_td_32, iterations=4)
    assert passes_64 == passes_32 >= 1
    assert np.max(np.abs(l_td_64.coords(copy=True) - l_td_r.coords(copy=True))) < 1e-4

    # Degenerate twin cases: n <= 1 and collapsed diag <= 1e-12
    deg_out_py: list[Any] = []
    for deg_c in (np.zeros((0, 2), dtype=np.float32), np.array([[1.0, 2.0]], dtype=np.float32), np.zeros((4, 2), dtype=np.float32)):
        l_d = eg.Layout2d.from_coords(deg_c)
        assert eg.declutter_twins_2d(l_d) >= 0 and l_d.is_all_finite()
        deg_out_py.append(l_d.coords(copy=True).ravel())
    assert np.array_equal(deg_out_py[1], np.array([1.0, 2.0], dtype=np.float32))

    # 3. transitive_reduction
    dag5 = eg.CsrGraph.from_edges(5, [(0, 1), (1, 2), (0, 2), (2, 3), (0, 3), (3, 4)], directed=True, sanitize=True)
    tr5 = eg.transitive_reduction(dag5)
    assert tr5.num_edges == 4

    # 4. DijkstraWorkspace / dijkstra_single_source / all_pairs_dijkstra (weighted + disconnected)
    w6_edges = [(0, 1), (1, 2), (0, 2), (2, 3), (4, 5)]
    w6_weights = [1.0, 1.5, 4.0, 2.0, 0.7]
    g6_w = eg.CsrGraph.from_edges(6, w6_edges, weights=w6_weights, directed=False, sanitize=True, weight_dtype="float32")
    dws = eg.DijkstraWorkspace(6)
    d_ws0 = dws.dijkstra_from(g6_w, 0)
    d_ss0 = eg.dijkstra_single_source(g6_w, 0)
    d_apsp = eg.all_pairs_dijkstra(g6_w)
    assert np.isinf(d_ss0[4]) and np.array_equal(d_ws0, d_ss0)

    # 5. adjusted_patch_nig_detail & LayoutDegeneracy ratios
    l_nig = eg.Layout2d.from_coords(init16_f32)
    nig_det_py = eg.adjusted_patch_nig_detail(l_nig, [0] * 8 + [1] * 8, patch_radius=16, canvas_size=1024, pad_frac=0.05, seed=42)
    l_deg = eg.Layout2d.from_coords(twin_coords_f32)
    deg6_py = eg.layout_degeneracy(eg.CsrGraph.from_edges(6, [(0, 1), (1, 2)], directed=False, sanitize=True), l_deg)

    # 6. Datasets & synthetic generators + CsrGraph helpers
    ds_names = eg.list_datasets()
    cat = eg.dataset_catalog()
    ds_first = eg.dataset_info(ds_names[0])
    assert len(cat) == len(ds_names) >= 1 and ds_first.name == ds_names[0]
    gw = eg.generate_weighted_graph(12, avg_degree=4, min_weight=0.2, max_weight=1.5, seed=42)
    gdag = eg.generate_layered_dag(3, 4, edge_prob=0.4, seed=42)
    gsbm, sbm_lbl = eg.generate_sbm_graph([5, 5], p_intra=0.6, p_inter=0.1, seed=42)
    gtor = eg.generate_torus_grid(3, 4, triangulated=True)
    gico = eg.generate_spherical_icosphere(1)
    gsph = eg.generate_spherical_grid(4, 6)
    ghyp = eg.generate_hyperbolic_tree(2, 3)

    assert len(g6_w) == 6 and g6_w.has_edge(0, 1) and not g6_w.has_edge(0, 4)
    assert g6_w.edge_weight(0, 4) is None and abs(float(g6_w.edge_weight(0, 1) or 0.0) - 1.0) < 1e-6
    assert not g6_w.is_connected() and g16.is_connected() and g6_w.weight_dtype == "float32"

    assert_non_panic_error("py/dijkstra/oob", lambda: eg.dijkstra_single_source(g6_w, 99))

    spec_payload = {
        "g16_edges": [int(v) for e in g16_edges for v in e],
        "g260_edges": [int(v) for e in g260_edges for v in e],
        "init16": [float(v) for v in init16_f32.ravel()],
        "init260": [float(v) for v in init260_f32.ravel()],
        "twin_coords": [float(v) for v in twin_coords_f32.ravel()],
        "w6_edges": [int(v) for e in w6_edges for v in e],
        "w6_weights": w6_weights,
    }

    node_body = """
const g16 = mod.CsrGraph.fromEdgeList(16, new Uint32Array(spec.g16_edges));
const g260 = mod.CsrGraph.fromEdgeList(260, new Uint32Array(spec.g260_edges));
const init16 = new Float32Array(spec.init16);
const init260 = new Float32Array(spec.init260);

const tsSm = mod.TsNetSolver.pure2Hop(g16, { backend: 'auto', iterationsStage1: 2, iterationsStage2: 2, iterationsStage3: 2, seed: 42 });
assert.equal(tsSm.resolvedBackend, 'exact');
const lSm = mod.Layout2d.fromCoords(init16);
tsSm.run(lSm);
assert.equal(tsSm.totalFft2dCalls, 0);

const tsBg = mod.TsNetSolver.pure2Hop(g260, { backend: 'auto', iterationsStage1: 1, iterationsStage2: 1, iterationsStage3: 1, learningRateDivisor: 3.0, seed: 42 });
assert.equal(tsBg.resolvedBackend, 'linear_fft');
const lBg = mod.Layout2d.fromCoords(init260);
tsBg.run(lBg);
assert.ok(tsBg.totalFft2dCalls > 0);

const probsRw = mod.graph12hopRowNormalizedProbabilitiesWithOptions(g16, 0.4, 8, 0.25, 'two_step_random_walk', 3, 42);
const probsCsr = mod.SparseProbabilities.fromCsr(16, probsRw.indptr(), probsRw.indices(), probsRw.values());
const stages = [new mod.TsNetStage(2, 4.0, 1.0, 0.1, 0.0), new mod.TsNetStage(2, 1.0, 1.0, 0.0, 0.0)];
const tsSt = mod.TsNetSolver.fromProbabilitiesWithStages(probsCsr, stages, { backend: 'exact', learningRateDivisor: 3.0, seed: 42 });
const lSt = mod.Layout2d.fromCoords(init16);
tsSt.run(lSt);

const pairsRw = mod.extractSparse1hop2hopPairsWithMode(g16, 8, 'two_step_random_walk', 3, 42);
const tsRnRw = mod.TsNetSolver.fromGraphRowNormalizedWithOptions(g16, {
  w2: 0.4, max2hopPerNode: 8, degreeBeta: 0.25, mode: 'two_step_random_walk', maxBridgeDegree: 3,
  backend: 'exact', iterationsStage1: 2, iterationsStage2: 2, iterationsStage3: 2, seed: 42,
});
const lRnRw = mod.Layout2d.fromCoords(init16);
tsRnRw.run(lRnRw);

const sgdNew = new mod.SgdSolver(g16);
assert.equal(sgdNew.totalIterations, 60);
const lSgdNew = mod.Layout2d.fromCoords(init16);
sgdNew.run(lSgdNew);

const gumapInit16 = new Float32Array(spec.init16.map((v) => v * 5.0));
const gmNew = new mod.GuMapSolver(g16);
const lGmNew = mod.Layout2d.fromCoords(gumapInit16);
const lGmDog = mod.Layout2d.fromCoords(gumapInit16);
gmNew.step(lGmNew);
mod.GuMapSolver.fromGraphDog(g16).step(lGmDog);
assert.deepEqual(Array.from(lGmNew.coordsView()), Array.from(lGmDog.coordsView()));
const lGmLr = mod.Layout2d.fromCoords(gumapInit16);
new mod.GuMapSolver(g16, { learningRate: 0.2 }).step(lGmLr);

const pureRes16 = mod.buildPure2hopKnn(g16);
const knn16 = pureRes16.knn();
const perps16 = mod.computeNodewise2hopPerplexities(g16);
const probs16 = mod.computeKnnProbabilities(knn16, perps16);
const siOpts = { backend: 'exact', stageIterations: [2, 3, 2] };
const siSolvers = {
  new: new mod.TsNetSolver(g16, siOpts),
  pure: mod.TsNetSolver.pure2Hop(g16, siOpts),
  knn: mod.TsNetSolver.fromKnn(knn16, Array.from(perps16), 15.0, siOpts),
  probs: mod.TsNetSolver.fromProbabilities(probs16, siOpts),
  bfs: mod.TsNetSolver.fromGraphPartialBfs(g16, 15, 15.0, siOpts),
  rn: mod.TsNetSolver.fromGraphRowNormalized(g16, siOpts),
};
const siCoords = {};
for (const [kSi, svSi] of Object.entries(siSolvers)) {
  assert.equal(svSi.totalIterations, 7);
  const lSi = mod.Layout2d.fromCoords(init16);
  svSi.run(lSi);
  siCoords[kSi] = Array.from(lSi.coordsView());
  lSi.free?.();
  svSi.free?.();
}
probs16.free?.(); knn16.free?.(); pureRes16.free?.();

const twinC = new Float32Array(spec.twin_coords);
const tdR = new mod.TwinDeclutterer({ iterations: 4 });
const tdS = new mod.TwinDeclutterer({ iterations: 4 });
const lTdR = mod.Layout2d.fromCoords(twinC);
const lTdS = mod.Layout2d.fromCoords(twinC);
tdR.run(lTdR);
tdS.step(lTdS);
tdS.run(lTdS);
assert.deepEqual(Array.from(lTdR.coordsView()), Array.from(lTdS.coordsView()));
const lTdFn = mod.Layout2d.fromCoords(twinC);
const passesFn = mod.declutterTwins2d(lTdFn, { iterations: 4 });
assert.deepEqual(Array.from(lTdFn.coordsView()), Array.from(lTdR.coordsView()));
const degOut = [];
for (const flatDeg of [[], [1, 2], [0, 0, 0, 0, 0, 0, 0, 0]]) {
  const arr = new Float32Array(flatDeg);
  const ld = mod.Layout2d.fromCoords(arr);
  assert.ok(mod.declutterTwins2d(ld) >= 0);
  assert.ok(ld.isAllFinite());
  degOut.push(Array.from(ld.coordsView()));
  ld.free?.();
}

const dag5 = mod.CsrGraph.fromEdgeList(5, new Uint32Array([0, 1, 1, 2, 0, 2, 2, 3, 0, 3, 3, 4]), null, true, true);
const tr5 = mod.transitiveReduction(dag5);

const g6W = mod.CsrGraph.fromEdgeList(6, new Uint32Array(spec.w6_edges), new Float32Array(spec.w6_weights), false, true);
const dws = new mod.DijkstraWorkspace(6);
dws.ensureCapacity(8);
const dWs0 = Array.from(dws.dijkstraFrom(g6W, 0), (v) => (Number.isFinite(v) ? v : null));
const dSs0 = Array.from(mod.dijkstraSingleSource(g6W, 0), (v) => (Number.isFinite(v) ? v : null));
const dApsp = Array.from(mod.allPairsDijkstra(g6W), (v) => (Number.isFinite(v) ? v : null));

const nigLbl = new Uint32Array([0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1]);
const nigDet = mod.adjustedPatchNigDetail(mod.Layout2d.fromCoords(init16), nigLbl, 16, 1024, 0.05, 42).toRecord();
const gDeg2 = mod.CsrGraph.fromEdgeList(6, new Uint32Array([0, 1, 1, 2]));
const deg6 = mod.layoutDegeneracy(gDeg2, mod.Layout2d.fromCoords(twinC)).toRecord();

const dsNames = mod.listDatasets();
const catLen = mod.datasetCatalog().length;
const ds0 = mod.datasetInfo(dsNames[0]).toRecord();
const gw = mod.generateWeightedGraph(12, 4, 0.2, 1.5, 42);
const gdag = mod.generateLayeredDag(3, 4, 0.4, 42);
const gsbm = mod.generateSbmGraph(new Uint32Array([5, 5]), 0.6, 0.1, 42);
const gtor = mod.generateTorusGrid(3, 4, true);
const gico = mod.generateSphericalIcosphere(1);
const gsph = mod.generateSphericalGrid(4, 6);
const ghyp = mod.generateHyperbolicTree(2, 3);

assert.equal(g6W.len(), 6);
assert.equal(g6W.hasEdge(0, 1), true);
assert.equal(g6W.hasEdge(0, 4), false);
assert.equal(g6W.edgeWeight(0, 4), undefined);
assert.ok(Math.abs(g6W.edgeWeight(0, 1) - 1.0) < 1e-6);
assert.equal(g6W.isConnected(), false);
assert.equal(g16.isConnected(), true);
assert.equal(g6W.weightDtype, 'float32');
assertNonPanicError('wasm/dijkstra/oob', () => mod.dijkstraSingleSource(g6W, 99));

const out = {
  autoSm: Array.from(lSm.coordsView()), autoBg: Array.from(lBg.coordsView()),
  autoSmFfts: tsSm.totalFft2dCalls, autoBgFfts: tsBg.totalFft2dCalls,
  stCoords: Array.from(lSt.coordsView()), rnRwCoords: Array.from(lRnRw.coordsView()),
  sgdNewCoords: Array.from(lSgdNew.coordsView()), gmNewCoords: Array.from(lGmNew.coordsView()), gmLrCoords: Array.from(lGmLr.coordsView()),
  siCoords,
  pairsRwLen: pairsRw.num1hop + pairsRw.num2hop,
  tdCoords: Array.from(lTdR.coordsView()), tdClusters: tdR.lastClusterCount, tdMoved: tdR.lastMovedCount, passesFn, degOut,
  tr5Edges: Array.from(tr5.edges()), tr5NumEdges: tr5.numEdges(),
  dWs0, dSs0, dApsp, nigDet, deg6,
  dsNames, catLen, ds0Name: ds0.name,
  synCounts: {
    gw: [gw.numNodes(), gw.numEdges()], gdag: [gdag.numNodes(), gdag.numEdges()],
    gsbm: [gsbm.graph.numNodes(), gsbm.graph.numEdges(), Array.from(gsbm.labels)],
    gtor: [gtor.numNodes(), gtor.numEdges()], gico: [gico.numNodes(), gico.numEdges()],
    gsph: [gsph.numNodes(), gsph.numEdges()], ghyp: [ghyp.numNodes(), ghyp.numEdges()],
  },
  g6Edges: Array.from(g6W.edges()),
};
ghyp.free?.(); gsph.free?.(); gico.free?.(); gtor.free?.(); gsbm.graph.free?.(); gdag.free?.(); gw.free?.();
gDeg2.free?.(); dws.free?.(); g6W.free?.(); tr5.free?.(); dag5.free?.();
lTdFn.free?.(); lTdS.free?.(); lTdR.free?.(); tdS.free?.(); tdR.free?.();
lRnRw.free?.(); tsRnRw.free?.(); lGmLr.free?.(); lGmDog.free?.(); lGmNew.free?.(); gmNew.free?.(); lSgdNew.free?.(); sgdNew.free?.(); pairsRw.free?.(); lSt.free?.(); tsSt.free?.();
probsCsr.free?.(); probsRw.free?.(); lBg.free?.(); tsBg.free?.(); lSm.free?.(); tsSm.free?.();
g260.free?.(); g16.free?.();
console.log(JSON.stringify(out));
"""
    ws = run_node_json(wasm_pkg, node_body, spec_payload)

    max_f32_diff = 0.0
    max_f64_diff = 0.0
    max_f32_diff = max(max_f32_diff, check_array_close("rn/auto_small", l_auto_sm.coords(copy=True).ravel(), ws["autoSm"], 1e-4))
    max_f32_diff = max(max_f32_diff, check_array_close("rn/auto_big", l_auto_bg.coords(copy=True).ravel(), ws["autoBg"], 1e-4))
    assert ts_auto_small.total_fft_2d_calls == ws["autoSmFfts"] == 0
    assert ts_auto_big.total_fft_2d_calls == ws["autoBgFfts"] > 0
    max_f32_diff = max(max_f32_diff, check_array_close("rn/stages", l_st_py.coords(copy=True).ravel(), ws["stCoords"], 1e-4))
    max_f32_diff = max(max_f32_diff, check_array_close("rn/two_step_rw", l_rn_rw.coords(copy=True).ravel(), ws["rnRwCoords"], 1e-4))
    max_f32_diff = max(max_f32_diff, check_array_close("ctor/sgd_new_default", l_sgd_new.coords(copy=True).ravel(), ws["sgdNewCoords"], 1e-4))
    max_f32_diff = max(max_f32_diff, check_array_close("ctor/gumap_new_default", l_gm_new.coords(copy=True).ravel(), ws["gmNewCoords"], 5e-4))
    max_f32_diff = max(max_f32_diff, check_array_close("ctor/gumap_new_lr02", l_gm_lr.coords(copy=True).ravel(), ws["gmLrCoords"], 1e-4))
    for k_si, coords_si in si_coords_py.items():
        max_f32_diff = max(max_f32_diff, check_array_close(f"tsnet/stage_iterations/{k_si}", coords_si, ws["siCoords"][k_si], 1e-4))
    assert len(pairs_rw.u1()) + len(pairs_rw.u2()) == ws["pairsRwLen"]
    max_f32_diff = max(max_f32_diff, check_array_close("declutter/coords", l_td_r.coords(copy=True).ravel(), ws["tdCoords"], 1e-5))
    assert td_r.last_cluster_count == ws["tdClusters"]
    assert td_r.last_moved_count == ws["tdMoved"]
    assert passes_32 == ws["passesFn"]
    for idx_deg, arr_deg in enumerate(deg_out_py):
        max_f32_diff = max(max_f32_diff, check_array_close(f"declutter/deg/{idx_deg}", arr_deg, ws["degOut"][idx_deg], 1e-5))
    assert tr5.num_edges == ws["tr5NumEdges"] == 4
    assert list(tr5.edges().ravel()) == ws["tr5Edges"]

    py_d_ss0 = [float(v) if np.isfinite(v) else None for v in d_ss0]
    py_apsp = [float(v) if np.isfinite(v) else None for v in d_apsp.ravel()]
    assert [v is None for v in py_d_ss0] == [v is None for v in ws["dSs0"]]
    assert [v is None for v in py_apsp] == [v is None for v in ws["dApsp"]]
    max_f64_diff = max(max_f64_diff, check_array_close(
        "dijkstra/ss", [v for v in py_d_ss0 if v is not None], [v for v in ws["dSs0"] if v is not None], 1e-6
    ))
    max_f64_diff = max(max_f64_diff, check_array_close(
        "dijkstra/apsp", [v for v in py_apsp if v is not None], [v for v in ws["dApsp"] if v is not None], 1e-6
    ))

    for pk, wk in (("nig", "nig"), ("h_obs", "hObs"), ("h_null", "hNull"), ("singleton_patch_ratio", "singletonPatchRatio")):
        max_f64_diff = max(max_f64_diff, check_array_close(f"nig/{pk}", [nig_det_py[pk]], [ws["nigDet"][wk]], 1e-10))
    assert nig_det_py["num_nonempty_patches"] == ws["nigDet"]["numNonemptyPatches"]
    assert nig_det_py["is_degenerate"] == ws["nigDet"]["isDegenerate"]
    max_f64_diff = max(max_f64_diff, check_array_close("deg/excess_dup", [deg6_py["excess_duplicate_ratio"]], [ws["deg6"]["excessDuplicateRatio"]], 1e-12))
    max_f64_diff = max(max_f64_diff, check_array_close("deg/near_col", [deg6_py["near_collision_ratio"]], [ws["deg6"]["nearCollisionRatio"]], 1e-12))

    assert ds_names == ws["dsNames"] and len(cat) == ws["catLen"] and ds_first.name == ws["ds0Name"]
    assert (gw.num_nodes, gw.num_edges) == tuple(ws["synCounts"]["gw"])
    assert (gdag.num_nodes, gdag.num_edges) == tuple(ws["synCounts"]["gdag"])
    assert (gsbm.num_nodes, gsbm.num_edges, list(sbm_lbl)) == tuple(ws["synCounts"]["gsbm"])
    assert (gtor.num_nodes, gtor.num_edges) == tuple(ws["synCounts"]["gtor"])
    assert (gico.num_nodes, gico.num_edges) == tuple(ws["synCounts"]["gico"])
    assert (gsph.num_nodes, gsph.num_edges) == tuple(ws["synCounts"]["gsph"])
    assert (ghyp.num_nodes, ghyp.num_edges) == tuple(ws["synCounts"]["ghyp"])
    assert list(g6_w.edges().ravel()) == ws["g6Edges"]

    return [{
        "case": "cross_target_rn_tsnet_declutter_dijkstra_py_vs_wasm",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "max_f32_diff": round(max_f32_diff, 12),
        "max_f64_diff": round(max_f64_diff, 12),
    }]
