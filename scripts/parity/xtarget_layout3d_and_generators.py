"""Cross-target Python <-> WASM parity for Layout3d, 3D solvers, EdgePolylines/FdebSolver, and LayoutGenerators."""

from __future__ import annotations

import gc
import time
from typing import Any

from parity.common import (
    check_array_close,
    make_asymmetric_coords,
    make_grid_edges,
    resolve_wasm_entries,
    run_node_json,
)


def _make_coords_3d(n: int, scale: float = 0.8) -> list[list[float]]:
    base2 = make_asymmetric_coords(n, scale=scale)
    return [[x, y, (((i * 7 + 3) % n) / float(max(n, 1)) - 0.5) * 0.4 * scale] for i, (x, y) in enumerate(base2)]


def run_cross_target_layout3d_generators_parity(
    eg: Any, wasm_pkg: Any
) -> list[dict[str, Any]]:
    """Verify Python <-> WASM parity for Layout3d, 3D solvers, 2D/3D EdgePolylines + FDEB, and LayoutGenerators."""
    import numpy as np

    if not hasattr(eg, "Layout3d") or not hasattr(eg, "PivotMds"):
        return []
    if resolve_wasm_entries(wasm_pkg) is None:
        return []

    t0 = time.perf_counter()
    g16_edges = make_grid_edges(4, 4)
    g16 = eg.CsrGraph.from_edges(16, g16_edges, directed=False, sanitize=True)

    init2_f32 = np.array(make_asymmetric_coords(16, scale=0.8), dtype=np.float32)
    init3_f32 = np.array(_make_coords_3d(16, scale=0.8), dtype=np.float32)
    xs_f32 = init3_f32[:, 0].copy()
    ys_f32 = init3_f32[:, 1].copy()
    zs_f32 = init3_f32[:, 2].copy()

    # 1. Layout3d lifecycle: zeros, init, from_coords, from_xyz, x/y/z/set_x/y/z, normalize_std, coords
    l3_zeros = eg.Layout3d.zeros(4, dtype="float32")
    assert l3_zeros.num_nodes == 4 and not l3_zeros.is_empty()
    l3_zeros.set_x(1, 0.25)
    l3_zeros.set_y(1, -0.5)
    l3_zeros.set_z(1, 0.75)
    assert abs(l3_zeros.x(1) - 0.25) < 1e-6
    assert abs(l3_zeros.y(1) + 0.5) < 1e-6
    assert abs(l3_zeros.z(1) - 0.75) < 1e-6

    l3_init = eg.Layout3d.init(16, seed=42, scale=0.9, dtype="float32")
    l3_init.normalize_std(1.0)
    assert l3_init.is_all_finite()
    l3_init_py = l3_init.coords(copy=True).ravel()

    l3_from_coords = eg.Layout3d.from_coords(init3_f32)
    l3_from_xyz = eg.Layout3d.from_xyz(xs_f32, ys_f32, zs_f32)
    assert np.array_equal(l3_from_coords.coords(copy=True), l3_from_xyz.coords(copy=True))

    # 2. 3D solvers with target="layout3d" (run vs run_steps + step equivalence)
    def run_3d_pair(mk: Any, chunk: int) -> Any:
        sr, ss = mk(), mk()
        assert sr.target == "layout3d"
        lr = eg.Layout3d.from_coords(init3_f32)
        ls = eg.Layout3d.from_coords(init3_f32)
        sr.run(lr)
        ss.run_steps(ls, chunk)
        while not ss.is_finished:
            ss.step(ls)
        cr = lr.coords(copy=True)
        assert np.array_equal(cr, ls.coords(copy=True))
        return cr.ravel()

    lm16 = eg.compute_regularized_dog_varadhan_landmarks(g16, degree=15, num_landmarks=8, seed=42)
    solvers_3d_py: dict[str, Any] = {
        "sgd": run_3d_pair(lambda: eg.SgdSolver.full(g16, target="layout3d", iterations=6, seed=42, dtype="float32"), 2),
        "sgd2": run_3d_pair(lambda: eg.Sgd2Solver(g16, target="layout3d", iterations=6, batch_size=8, seed=42, dtype="float32"), 2),
        "pacmap": run_3d_pair(lambda: eg.PacmapGraphSolver.with_landmarks(g16, lm16, target="layout3d", n_neighbors=4, phase_iterations=(2, 2, 2), seed=42, dtype="float32"), 2),
        "largevis": run_3d_pair(lambda: eg.LargeVisSolver(g16, target="layout3d", iterations=6, negative_samples=2, seed=42, dtype="float32"), 2),
        "drgraph": run_3d_pair(lambda: eg.DrGraphSolver(g16, target="layout3d", iterations=6, negative_samples=2, seed=42, dtype="float32"), 2),
        "graph_cne": run_3d_pair(lambda: eg.GraphCneSolver(g16, target="layout3d", iterations=6, negative_samples=2, seed=42, dtype="float32"), 2),
        "snap_t_fdp": run_3d_pair(lambda: eg.SnapTFdpSolver(g16, target="layout3d", iterations=6, negatives_per_edge=2, seed=42, dtype="float32"), 2),
        "tsnet_exact": run_3d_pair(lambda: eg.TsNetSolver.pure_2hop(g16, target="layout3d", backend="exact", stage_iterations=(2, 2, 2), seed=42, dtype="float32"), 2),
        "tsnet_bh": run_3d_pair(lambda: eg.TsNetSolver.pure_2hop(g16, target="layout3d", backend="barnes_hut", stage_iterations=(2, 2, 2), theta=0.5, seed=42, dtype="float32"), 2),
        "tfdp_exact": run_3d_pair(lambda: eg.TFdpSolver(g16, target="layout3d", backend="exact", iterations=6, seed=42, dtype="float32"), 2),
        "tfdp_bh": run_3d_pair(lambda: eg.TFdpSolver(g16, target="layout3d", backend="barnes_hut", iterations=6, theta=0.5, seed=42, dtype="float32"), 2),
        "gumap": run_3d_pair(lambda: eg.GuMapSolver.from_graph_dog(g16, target="layout3d", max_2hop_per_node=16, degree=15, num_landmarks=8, iterations=6, learning_rate=0.2, seed=42, dtype="float32"), 2),
        "fa2": run_3d_pair(lambda: eg.ForceAtlas2Solver(g16, target="layout3d", backend="barnes_hut", iterations=6, theta=0.5, seed=42, dtype="float32"), 2),
        "linlog": run_3d_pair(lambda: eg.LinLogSolver(g16, target="layout3d", backend="barnes_hut", iterations=6, theta=0.5, seed=42, dtype="float32"), 2),
        "yh": run_3d_pair(lambda: eg.YifanHuSolver(g16, target="layout3d", backend="barnes_hut", max_iterations_per_level=6, min_coarsen_size=6, theta=0.5, seed=42, dtype="float32"), 3),
        "fr_exact": run_3d_pair(lambda: eg.FruchtermanReingoldSolver(g16, target="layout3d", backend="exact", iterations=6, seed=42, dtype="float32"), 2),
        "fr_bh": run_3d_pair(lambda: eg.FruchtermanReingoldSolver(g16, target="layout3d", backend="barnes_hut", iterations=6, theta=0.5, seed=42, dtype="float32"), 2),
        "maxent": run_3d_pair(lambda: eg.MaxentStressSolver(g16, target="layout3d", backend="barnes_hut", stages=2, steps_per_stage=3, theta=0.5, seed=42, dtype="float32"), 2),
        "sm_pcg": run_3d_pair(lambda: eg.StressMajorizationSolver.from_graph(g16, target="layout3d", backend="euclidean_pcg", max_iterations=8, dtype="float32"), 3),
        "kk": run_3d_pair(lambda: eg.KamadaKawaiSolver.from_graph(g16, target="layout3d", max_iterations=8, eps=1e-3, max_inner_iters=25, dtype="float32"), 3),
    }

    # 3. EdgePolylines + FdebSolver (2D & 3D) with Python GC safety after dropping inputs
    def run_fdeb_py(layout_maker: Any) -> tuple[Any, Any]:
        g_tmp = eg.CsrGraph.from_edges(16, g16_edges, directed=False, sanitize=True)
        lay_tmp = layout_maker()
        poly_r = eg.EdgePolylines.straight_line_from_graph(g_tmp, lay_tmp)
        poly_s = eg.EdgePolylines.from_parts(poly_r.edge_offsets(copy=True), eg.Layout2d.from_coords(poly_r.points(copy=True)) if poly_r.dims == 2 else eg.Layout3d.from_coords(poly_r.points(copy=True)))
        del g_tmp, lay_tmp
        gc.collect()
        fd_r = eg.FdebSolver(poly_r, cycles=1, steps_per_cycle=4)
        fd_s = eg.FdebSolver(poly_s, cycles=1, steps_per_cycle=4)
        fd_r.run(poly_r)
        fd_s.step(poly_s)
        fd_s.run(poly_s)
        assert np.array_equal(poly_r.edge_offsets(copy=True), poly_s.edge_offsets(copy=True))
        assert np.array_equal(poly_r.points(copy=True), poly_s.points(copy=True))
        return poly_r.edge_offsets(copy=True), poly_r.points(copy=True).ravel()

    off2_py, pts2_py = run_fdeb_py(lambda: eg.Layout2d.from_coords(init2_f32))
    off3_py, pts3_py = run_fdeb_py(lambda: eg.Layout3d.from_coords(init3_f32))

    g16_bb_weights = [1.0 + 0.05 * i for i in range(len(g16_edges))]
    g16_bb = eg.CsrGraph.from_edges(16, g16_edges, weights=g16_bb_weights, directed=False, sanitize=True)

    # 4. LayoutGenerator classes (2D and 3D)
    gens_py: dict[str, Any] = {
        "pivot_2d": eg.PivotMds(g16, target="layout2d", num_pivots=6, seed=42, dtype="float32").generate().coords(copy=True).ravel(),
        "pivot_3d": eg.PivotMds(g16, target="layout3d", num_pivots=6, seed=42, dtype="float32").generate().coords(copy=True).ravel(),
        "classical_2d": eg.ClassicalMds(g16, target="layout2d", dtype="float32").generate().coords(copy=True).ravel(),
        "classical_3d": eg.ClassicalMds(g16, target="layout3d", dtype="float32").generate().coords(copy=True).ravel(),
        "hde_2d": eg.Hde(g16, target="layout2d", dimensions=6, seed=42, dtype="float32").generate().coords(copy=True).ravel(),
        "hde_3d": eg.Hde(g16, target="layout3d", dimensions=6, seed=42, dtype="float32").generate().coords(copy=True).ravel(),
        "rdmds_2d": eg.RdMds(g16, target="layout2d", seed=42, dtype="float32").generate().coords(copy=True).ravel(),
        "rdmds_3d": eg.RdMds(g16, target="layout3d", seed=42, dtype="float32").generate().coords(copy=True).ravel(),
        "koren_2d": eg.KorenSpectral(g16, target="layout2d", seed=42, dtype="float32").generate().coords(copy=True).ravel(),
        "koren_3d": eg.KorenSpectral(g16, target="layout3d", seed=42, dtype="float32").generate().coords(copy=True).ravel(),
        "coarse_backbone": eg.CoarseBackbone(
            g16_bb,
            max_coarse_nodes=4,
            edge_length_mode="resistance_curvature_flow",
            distance_backend="rhgh",
            coarse_t_max=20,
            seed=42,
            dtype="float32",
        ).generate().fine_layout.coords(copy=True).ravel(),
        "sfc_coords": eg.SfcLayout(g16, curve="hilbert", seed=42, dtype="float32").generate().coords(copy=True).ravel(),
    }
    sfc_g = eg.SfcLayout(g16, curve="hilbert", seed=42, dtype="float32")
    _ = sfc_g.generate()
    sfc_order_py = list(sfc_g.vertex_order())

    spec_payload = {
        "g16_edges": [int(v) for e in g16_edges for v in e],
        "g16_bb_weights": g16_bb_weights,
        "init2": [float(v) for v in init2_f32.ravel()],
        "init3": [float(v) for v in init3_f32.ravel()],
        "xs": [float(v) for v in xs_f32],
        "ys": [float(v) for v in ys_f32],
        "zs": [float(v) for v in zs_f32],
    }

    node_body = """
const g16 = mod.CsrGraph.fromEdgeList(16, new Uint32Array(spec.g16_edges));
const init2 = new Float32Array(spec.init2);
const init3 = new Float32Array(spec.init3);

const l3z = mod.Layout3d.zeros(4);
assert.equal(l3z.numNodes, 4);
assert.equal(l3z.isEmpty(), false);
l3z.setX(1, 0.25); l3z.setY(1, -0.5); l3z.setZ(1, 0.75);
assert.ok(Math.abs(l3z.x(1) - 0.25) < 1e-6);
assert.ok(Math.abs(l3z.y(1) + 0.5) < 1e-6);
assert.ok(Math.abs(l3z.z(1) - 0.75) < 1e-6);
l3z.free?.();

const l3Init = mod.Layout3d.init(16, { seed: 42, scale: 0.9 });
l3Init.normalizeStd(1.0);
assert.ok(l3Init.isAllFinite());
// Verify Layout3d.coordsView() survives memory.grow cleanly
const l3ViewPre = l3Init.coordsView();
const memPre = mod.wasmMemory().buffer;
mod.wasmMemory().grow(2);
assert.notEqual(mod.wasmMemory().buffer, memPre);
assert.equal(l3ViewPre.buffer.byteLength, 0);
const l3InitArr = Array.from(l3Init.coordsView());
l3Init.free?.();

const l3Coords = mod.Layout3d.fromCoords(init3);
const l3Xyz = mod.Layout3d.fromXyz(new Float32Array(spec.xs), new Float32Array(spec.ys), new Float32Array(spec.zs));
assert.deepEqual(Array.from(l3Coords.coordsView()), Array.from(l3Xyz.coordsView()));
l3Xyz.free?.();

const run3dPair = (mk, chunk) => {
  const sr = mk();
  const ss = mk();
  assert.equal(sr.target, 'layout3d');
  const lr = mod.Layout3d.fromCoords(init3);
  const ls = mod.Layout3d.fromCoords(init3);
  sr.run(lr);
  ss.runSteps(ls, chunk);
  while (!ss.isFinished) ss.step(ls);
  const cr = Array.from(lr.coordsView());
  assert.deepEqual(cr, Array.from(ls.coordsView()));
  lr.free?.(); ls.free?.(); sr.free?.(); ss.free?.();
  return cr;
};

const lm16 = mod.computeRegularizedDogVaradhanLandmarks(g16, { degree: 15, numLandmarks: 8, seed: 42 });
const solvers3d = {
  sgd: run3dPair(() => mod.SgdSolver.full(g16, { target: 'layout3d', iterations: 6, seed: 42 }), 2),
  sgd2: run3dPair(() => new mod.Sgd2Solver(g16, { target: 'layout3d', iterations: 6, batchSize: 8, seed: 42 }), 2),
  pacmap: run3dPair(() => mod.PacmapGraphSolver.withLandmarks(g16, lm16, { target: 'layout3d', nNeighbors: 4, phaseIterations: [2, 2, 2], seed: 42 }), 2),
  largevis: run3dPair(() => new mod.LargeVisSolver(g16, { target: 'layout3d', iterations: 6, negativeSamples: 2, seed: 42 }), 2),
  drgraph: run3dPair(() => new mod.DrGraphSolver(g16, { target: 'layout3d', iterations: 6, negativeSamples: 2, seed: 42 }), 2),
  graph_cne: run3dPair(() => new mod.GraphCneSolver(g16, { target: 'layout3d', iterations: 6, negativeSamples: 2, seed: 42 }), 2),
  snap_t_fdp: run3dPair(() => new mod.SnapTFdpSolver(g16, { target: 'layout3d', iterations: 6, negativesPerEdge: 2, seed: 42 }), 2),
  tsnet_exact: run3dPair(() => mod.TsNetSolver.pure2Hop(g16, { target: 'layout3d', backend: 'exact', stageIterations: [2, 2, 2], seed: 42 }), 2),
  tsnet_bh: run3dPair(() => mod.TsNetSolver.pure2Hop(g16, { target: 'layout3d', backend: 'barnesHut', stageIterations: [2, 2, 2], theta: 0.5, seed: 42 }), 2),
  tfdp_exact: run3dPair(() => new mod.TFdpSolver(g16, { target: 'layout3d', backend: 'exact', iterations: 6, seed: 42 }), 2),
  tfdp_bh: run3dPair(() => new mod.TFdpSolver(g16, { target: 'layout3d', backend: 'barnesHut', iterations: 6, theta: 0.5, seed: 42 }), 2),
  gumap: run3dPair(() => mod.GuMapSolver.fromGraphDog(g16, { target: 'layout3d', max2hopPerNode: 16, degree: 15, numLandmarks: 8, iterations: 6, learningRate: 0.2, seed: 42 }), 2),
  fa2: run3dPair(() => new mod.ForceAtlas2Solver(g16, { target: 'layout3d', backend: 'barnesHut', iterations: 6, theta: 0.5, seed: 42 }), 2),
  linlog: run3dPair(() => new mod.LinLogSolver(g16, { target: 'layout3d', backend: 'barnesHut', iterations: 6, theta: 0.5, seed: 42 }), 2),
  yh: run3dPair(() => new mod.YifanHuSolver(g16, { target: 'layout3d', backend: 'barnesHut', maxIterationsPerLevel: 6, minCoarsenSize: 6, theta: 0.5, seed: 42 }), 3),
  fr_exact: run3dPair(() => new mod.FruchtermanReingoldSolver(g16, { target: 'layout3d', backend: 'exact', iterations: 6, seed: 42 }), 2),
  fr_bh: run3dPair(() => new mod.FruchtermanReingoldSolver(g16, { target: 'layout3d', backend: 'barnesHut', iterations: 6, theta: 0.5, seed: 42 }), 2),
  maxent: run3dPair(() => new mod.MaxentStressSolver(g16, { target: 'layout3d', backend: 'barnesHut', stages: 2, stepsPerStage: 3, theta: 0.5, seed: 42 }), 2),
  sm_pcg: run3dPair(() => mod.StressMajorizationSolver.fromGraph(g16, { target: 'layout3d', backend: 'euclideanPcg', maxIterations: 8 }), 3),
  kk: run3dPair(() => mod.KamadaKawaiSolver.fromGraph(g16, { target: 'layout3d', maxIterations: 8, eps: 1e-3, maxInnerIters: 25 }), 3),
};

// EdgePolylines + FdebSolver (2D & 3D) across memory.grow
const runFdebGrow = (lay) => {
  const polyR = mod.EdgePolylines.straightLineFromGraph(g16, lay);
  const offPre = polyR.edgeOffsetsView();
  const ptsPre = polyR.pointsView();
  mod.wasmMemory().grow(2);
  assert.equal(offPre.buffer.byteLength, 0);
  assert.equal(ptsPre.buffer.byteLength, 0);
  assert.equal(polyR.edgeOffsetsView().length, g16.numEdges() + 1);
  assert.equal(polyR.pointsView().length, polyR.totalPoints * polyR.dims);
  const copyLay = polyR.dims === 2 ? mod.Layout2d.fromCoords(polyR.copyPoints()) : mod.Layout3d.fromCoords(polyR.copyPoints());
  const polyS = mod.EdgePolylines.fromParts(polyR.copyEdgeOffsets(), copyLay);
  const fdR = new mod.FdebSolver(polyR, { cycles: 1, stepsPerCycle: 4 });
  const fdS = new mod.FdebSolver(polyS, { cycles: 1, stepsPerCycle: 4 });
  fdR.run(polyR); fdS.step(polyS); fdS.run(polyS);
  const offArr = Array.from(polyR.edgeOffsetsView());
  const ptsArr = Array.from(polyR.pointsView());
  assert.deepEqual(ptsArr, Array.from(polyS.pointsView()));
  fdS.free?.(); fdR.free?.(); polyS.free?.(); copyLay.free?.(); polyR.free?.();
  return { offArr, ptsArr };
};
const l2Fdeb = mod.Layout2d.fromCoords(init2);
const fdeb2 = runFdebGrow(l2Fdeb);
const fdeb3 = runFdebGrow(l3Coords);
l2Fdeb.free?.(); l3Coords.free?.();

const genCoords = (gen) => {
  const out = gen.generate(g16);
  const arr = Array.from(out.coordsView());
  out.free?.(); gen.free?.();
  return arr;
};
const g16Bb = mod.CsrGraph.fromEdgeList(16, new Uint32Array(spec.g16_edges), new Float32Array(spec.g16_bb_weights));
const cbGen = new mod.CoarseBackbone(g16Bb, {
  maxCoarseNodes: 4,
  edgeLengthMode: 'resistanceCurvatureFlow',
  distanceBackend: 'rhgh',
  coarseTMax: 20,
  seed: 42,
});
const cbRes = cbGen.generateResult();
const cbCoords = Array.from(cbRes.fineLayout().coordsView());
cbRes.free?.(); cbGen.free?.(); g16Bb.free?.();

const sfcGen = new mod.SfcLayout(g16, { curve: 'hilbert', seed: 42 });
const sfcLay = sfcGen.generate();
const sfcCoords = Array.from(sfcLay.coordsView());
const sfcOrder = Array.from(sfcGen.vertexOrder());
sfcLay.free?.(); sfcGen.free?.();

const gens = {
  pivot_2d: genCoords(new mod.PivotMds({ target: 'layout2d', numPivots: 6, seed: 42 })),
  pivot_3d: genCoords(new mod.PivotMds({ target: 'layout3d', numPivots: 6, seed: 42 })),
  classical_2d: genCoords(new mod.ClassicalMds({ target: 'layout2d' })),
  classical_3d: genCoords(new mod.ClassicalMds({ target: 'layout3d' })),
  hde_2d: genCoords(new mod.Hde({ target: 'layout2d', dimensions: 6, seed: 42 })),
  hde_3d: genCoords(new mod.Hde({ target: 'layout3d', dimensions: 6, seed: 42 })),
  rdmds_2d: genCoords(new mod.RdMds({ target: 'layout2d', seed: 42 })),
  rdmds_3d: genCoords(new mod.RdMds({ target: 'layout3d', seed: 42 })),
  koren_2d: genCoords(new mod.KorenSpectral({ target: 'layout2d', seed: 42 })),
  koren_3d: genCoords(new mod.KorenSpectral({ target: 'layout3d', seed: 42 })),
  coarse_backbone: cbCoords,
  sfc_coords: sfcCoords,
};

lm16.free?.(); g16.free?.();
console.log(JSON.stringify({ l3InitArr, solvers3d, fdeb2, fdeb3, gens, sfcOrder }));
"""
    ws = run_node_json(wasm_pkg, node_body, spec_payload)

    max_f32_diff = 0.0
    max_f32_diff = max(max_f32_diff, check_array_close("layout3d/init", l3_init_py, ws["l3InitArr"], 1e-5))
    for k, arr in solvers_3d_py.items():
        max_f32_diff = max(max_f32_diff, check_array_close(f"layout3d/solver/{k}", arr, ws["solvers3d"][k], 1e-4))
    assert list(off2_py) == ws["fdeb2"]["offArr"]
    assert list(off3_py) == ws["fdeb3"]["offArr"]
    max_f32_diff = max(max_f32_diff, check_array_close("fdeb/2d_pts", pts2_py, ws["fdeb2"]["ptsArr"], 1e-4))
    max_f32_diff = max(max_f32_diff, check_array_close("fdeb/3d_pts", pts3_py, ws["fdeb3"]["ptsArr"], 1e-4))
    for k, arr in gens_py.items():
        max_f32_diff = max(max_f32_diff, check_array_close(f"gen/{k}", arr, ws["gens"][k], 1e-4))
    assert sfc_order_py == ws["sfcOrder"]

    return [{
        "case": "cross_target_layout3d_and_generators_py_vs_wasm",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "max_f32_diff": round(max_f32_diff, 12),
    }]
