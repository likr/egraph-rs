"""Node.js ESM parity checks for WASM G-CAF, community/overlap baselines, and entry runner."""

from __future__ import annotations

import pathlib
import time
from typing import Any

from parity.common import resolve_wasm_entries, run_node_script
from parity.wasm_node_core_and_linalg import WASM_NODE_CORE_AND_LINALG_JS
from parity.wasm_node_solvers import WASM_NODE_SOLVERS_JS

_WASM_NODE_METRICS_AND_COMMUNITY_JS = """
if (typeof mod.clusterAreaFaithfulness === 'function') {
  assert.equal(typeof mod.ClusterAreaReport, 'function', 'mod missing ClusterAreaReport');
  assert.equal(typeof mod.clusterAreaFaithfulnessDetail, 'function', 'mod missing clusterAreaFaithfulnessDetail');
  assert.equal(typeof indexMod.ClusterAreaReport, 'function', 'indexMod missing ClusterAreaReport');
  assert.equal(typeof indexMod.clusterAreaFaithfulness, 'function', 'indexMod missing clusterAreaFaithfulness');
  assert.equal(typeof indexMod.clusterAreaFaithfulnessDetail, 'function', 'indexMod missing clusterAreaFaithfulnessDetail');

  const imbCoords = new Float32Array(48);
  const imbLabels = new Uint32Array(24);
  for (let i = 0; i < 24; i++) {
    imbCoords[2 * i] = i < 20 ? -2.0 + 0.5 * (i / 19) : 1.5 + 0.5 * ((i - 20) / 3);
    imbCoords[2 * i + 1] = Math.sin(i * 0.7) * 0.2;
    imbLabels[i] = i < 20 ? 0 : 1;
  }
  const lImb = mod.Layout2d.fromCoords(imbCoords);
  const imbEdges = [];
  for (let i = 0; i < 23; i++) imbEdges.push(i, i + 1);
  const gImb = mod.CsrGraph.fromEdgeList(24, new Uint32Array(imbEdges));

  const cafDef = mod.clusterAreaFaithfulness(lImb, imbLabels);
  const detDef = mod.clusterAreaFaithfulnessDetail(lImb, imbLabels);
  assert.ok(detDef instanceof mod.ClusterAreaReport);
  assert.ok(Math.abs(detDef.caf - cafDef) < 1e-12);
  const recDef = detDef.toRecord();
  assert.ok(Math.abs(recDef.caf - cafDef) < 1e-12);
  for (const k of ['caf', 'purityAdj', 'propVis', 'resDensity', 'resDeconv', 'resCombined', 'spatialCoherence', 'inFrac']) {
    assert.ok(Number.isFinite(detDef[k]) && detDef[k] >= 0 && detDef[k] <= 1, `bad ${k}`);
    assert.ok(Math.abs(recDef[k] - detDef[k]) < 1e-12, `record mismatch ${k}`);
  }
  assert.ok(Number.isFinite(detDef.effectiveBoundaryR95) && detDef.effectiveBoundaryR95 > 0);
  assert.ok(Math.abs(recDef.effectiveBoundaryR95 - detDef.effectiveBoundaryR95) < 1e-12);
  assert.ok(Number.isInteger(detDef.gEff) && detDef.gEff >= 12);
  assert.equal(recDef.gEff, detDef.gEff);
  assert.equal(detDef.numClusters, 2);
  assert.equal(detDef.numNodes, 24);
  if (typeof detDef.clusterIds === 'function') {
    assert.equal(detDef.clusterIds().length, 2);
    assert.equal(detDef.counts().length, 2);
    const pSum = detDef.p().reduce((a, b) => a + b, 0);
    const qSum = detDef.q().reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(pSum - 1.0) < 1e-12);
    assert.ok(Math.abs(qSum - 1.0) < 1e-12);
    for (const m of ['purity', 'resolution', 'resDensityPerCluster', 'resDeconvPerCluster', 'cafPerCluster', 'coherence']) {
      assert.equal(detDef[m]().length, 2, `bad len ${m}`);
    }
  }
  detDef.free?.();

  const cafCustom = mod.clusterAreaFaithfulness(lImb, imbLabels, {
    gMax: 64, sigmaCells: 0.9, kappa: 8.0, tauFrac: 0.2, boundaryR95: 2.5, stevensGamma: 0.7,
  });
  const l5 = mod.Layout2d.fromCoords(new Float32Array([0, 0, 1, 0, 0, 1, 3, 3, 4, 3]));
  const l5Arr = [100, 5, 100, 4294967295, 5];
  assert.equal(mod.clusterAreaFaithfulness(l5, new Uint32Array(l5Arr)), mod.clusterAreaFaithfulness(l5, l5Arr));
  l5.free?.();
  for (const t of ['clusterAreaFaithfulness', 'gcaf']) {
    const rep = mod.evaluateMetrics(gImb, lImb, [t], {
      labels: imbLabels, gMax: 64, sigmaCells: 0.9, kappa: 8.0, tauFrac: 0.2, boundaryR95: 2.5, stevensGamma: 0.7,
    });
    assert.ok(Math.abs(rep.clusterAreaFaithfulness - cafCustom) < 1e-12);
    assert.ok(Math.abs(rep.toRecord().clusterAreaFaithfulness - cafCustom) < 1e-12);
    rep.free?.();
  }

  const detInf = mod.clusterAreaFaithfulnessDetail(lImb, imbLabels, { boundaryR95: Number.POSITIVE_INFINITY });
  assert.ok(Math.abs(detInf.inFrac - 1.0) < 1e-12);
  detInf.free?.();
  const detG05 = mod.clusterAreaFaithfulnessDetail(lImb, imbLabels, { stevensGamma: 0.5 });
  const detG20 = mod.clusterAreaFaithfulnessDetail(lImb, imbLabels, { stevensGamma: 2.0 });
  if (typeof detG05.p === 'function') assert.ok(detG05.p()[0] < detG20.p()[0]);
  detG05.free?.(); detG20.free?.();

  // Degenerate boundaries: N=0, N=1, K=1, collapsed
  const l0 = mod.Layout2d.zeros(0);
  const d0 = mod.clusterAreaFaithfulnessDetail(l0, []);
  assert.equal(d0.caf, 0); assert.equal(d0.effectiveBoundaryR95, 0); assert.equal(d0.gEff, 0);
  assert.equal(d0.numClusters, 0); assert.equal(d0.numNodes, 0);
  d0.free?.(); l0.free?.();
  const l1 = mod.Layout2d.fromCoords(new Float32Array([1, 2]));
  assert.equal(mod.clusterAreaFaithfulness(l1, [5]), 0);
  l1.free?.();
  const l4 = mod.Layout2d.fromCoords(new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]));
  assert.equal(mod.clusterAreaFaithfulness(l4, [3, 3, 3, 3]), 0);
  l4.free?.();
  const lCol = mod.Layout2d.fromCoords(new Float32Array([2, -1, 2, -1, 2, -1, 2, -1, 2, -1, 2, -1]));
  assert.equal(mod.clusterAreaFaithfulness(lCol, [0, 0, 0, 1, 1, 1]), 0);
  lCol.free?.();

  // Lazy gMax=4 passes without gcaf target
  const lazyOk = mod.evaluateMetrics(gImb, lImb, ['aspectRatio'], { gMax: 4 });
  assert.ok(Number.isFinite(lazyOk.aspectRatio));
  assert.equal(lazyOk.clusterAreaFaithfulness, undefined);
  lazyOk.free?.();

  assertNonPanicError('gcaf label len mismatch', () => mod.clusterAreaFaithfulness(lImb, [0, 1]));
  const restLabels = Array.from(imbLabels.slice(1));
  const invalidCommunityLabels = [
    ['neg', [-1, ...restLabels]],
    ['float', [1.5, ...restLabels]],
    ['nan', [Number.NaN, ...restLabels]],
    ['inf', [Number.POSITIVE_INFINITY, ...restLabels]],
    ['overflow', [4294967296, ...restLabels]],
    ['strElem', ['bad', ...restLabels]],
    ['nullElem', [null, ...restLabels]],
    ['undefElem', [undefined, ...restLabels]],
    ['int32Array', new Int32Array([-1, ...restLabels])],
    ['float64Array', new Float64Array([1.5, ...restLabels])],
    ['arrayLikeObj', { length: 24 }],
    ['string', '000000000000000000001111'],
  ];
  for (const [tag, badLabels] of invalidCommunityLabels) {
    assertNonPanicError(`gcaf bad labels (${tag})`, () => mod.clusterAreaFaithfulness(lImb, badLabels));
    assertNonPanicError(`gcafDetail bad labels (${tag})`, () => mod.clusterAreaFaithfulnessDetail(lImb, badLabels));
    assertNonPanicError(
      `gcaf eval bad labels (${tag})`,
      () => mod.evaluateMetrics(gImb, lImb, ['gcaf'], { labels: badLabels }),
    );
  }
  for (const [tag, nullish] of [['null', null], ['undefined', undefined]]) {
    assertNonPanicError(`gcaf nullish labels (${tag})`, () => mod.clusterAreaFaithfulness(lImb, nullish));
    assertNonPanicError(`gcafDetail nullish labels (${tag})`, () => mod.clusterAreaFaithfulnessDetail(lImb, nullish));
    assertNonPanicError(
      `gcaf eval nullish labels (${tag})`,
      () => mod.evaluateMetrics(gImb, lImb, ['gcaf'], { labels: nullish }),
    );
  }
  const nanCoords = new Float32Array(imbCoords); nanCoords[0] = Number.NaN;
  assertNonPanicError('gcaf Layout2d.fromCoords nan', () => mod.Layout2d.fromCoords(nanCoords));
  const lNan = mod.Layout2d.fromCoords(imbCoords);
  lNan.coordsView()[0] = Number.NaN;
  assertNonPanicError('gcaf nan layout', () => mod.clusterAreaFaithfulness(lNan, imbLabels));
  lNan.free?.();
  for (const bg of [11, -1, 1.5]) assertNonPanicError(`gcaf gMax=${bg}`, () => mod.clusterAreaFaithfulness(lImb, imbLabels, { gMax: bg }));
  for (const bs of [0, -0.5, Number.NaN, Number.POSITIVE_INFINITY, 400]) assertNonPanicError(`gcaf sigma=${bs}`, () => mod.clusterAreaFaithfulness(lImb, imbLabels, { sigmaCells: bs }));
  for (const bv of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
    assertNonPanicError(`gcaf kappa=${bv}`, () => mod.clusterAreaFaithfulness(lImb, imbLabels, { kappa: bv }));
    assertNonPanicError(`gcaf tau=${bv}`, () => mod.clusterAreaFaithfulness(lImb, imbLabels, { tauFrac: bv }));
    assertNonPanicError(`gcaf gamma=${bv}`, () => mod.clusterAreaFaithfulness(lImb, imbLabels, { stevensGamma: bv }));
  }
  for (const br of [0, -1, Number.NaN, Number.NEGATIVE_INFINITY]) assertNonPanicError(`gcaf r95=${br}`, () => mod.clusterAreaFaithfulness(lImb, imbLabels, { boundaryR95: br }));
  assertNonPanicError('gcaf eval missing labels', () => mod.evaluateMetrics(gImb, lImb, ['gcaf']));
  assertNonPanicError('gcaf eval lazy gMax=4', () => mod.evaluateMetrics(gImb, lImb, ['clusterAreaFaithfulness'], { labels: imbLabels, gMax: 4 }));

  // Target nodeWeights & clusterWeights parity + errors
  const imbNw = new Float64Array(24);
  for (let i = 0; i < 24; i++) imbNw[i] = i < 20 ? 1.5 : 3.0;
  const imbCw = new Float64Array([30.0, 12.0]);
  const cafNw = mod.clusterAreaFaithfulness(lImb, imbLabels, { stevensGamma: 2.0, nodeWeights: imbNw });
  const cafCw = mod.clusterAreaFaithfulness(lImb, imbLabels, { stevensGamma: 2.0, clusterWeights: imbCw });
  assert.ok(Math.abs(cafNw - cafCw) < 1e-12);
  const evalCw = mod.evaluateMetrics(gImb, lImb, ['gcaf'], { labels: imbLabels, stevensGamma: 2.0, clusterWeights: imbCw });
  assert.ok(Math.abs(evalCw.clusterAreaFaithfulness - cafCw) < 1e-12);
  evalCw.free?.();
  assertNonPanicError('gcaf both weights', () => mod.clusterAreaFaithfulness(lImb, imbLabels, { nodeWeights: imbNw, clusterWeights: imbCw }));
  assertNonPanicError('gcaf zero weight', () => mod.clusterAreaFaithfulness(lImb, imbLabels, { clusterWeights: [0.0, 1.0] }));
  assertNonPanicError('gcaf bool weight', () => mod.clusterAreaFaithfulness(lImb, imbLabels, { clusterWeights: [true, false] }));

  lImb.free?.();
  gImb.free?.();
}

if (typeof mod.leidenCommunities === 'function') {
  assert.equal(typeof indexMod.leidenCommunities, 'function', 'index.js missing leidenCommunities');
  assert.equal(typeof indexMod.LinLogSolver, 'function', 'index.js missing LinLogSolver');
  assert.equal(typeof indexMod.FruchtermanReingoldSolver, 'function', 'index.js missing FruchtermanReingoldSolver');
  assert.equal(typeof indexMod.BackboneSolver, 'function', 'index.js missing BackboneSolver');
  assert.equal(typeof indexMod.ClusterRefineSolver, 'function', 'index.js missing ClusterRefineSolver');
  assert.equal(typeof indexMod.GroupInABoxSolver, 'function', 'index.js missing GroupInABoxSolver');
  assert.equal(typeof indexMod.GTreeOverlapRemoval, 'function', 'index.js missing GTreeOverlapRemoval');
  assert.equal(typeof indexMod.ForbidOverlapRemoval, 'function', 'index.js missing ForbidOverlapRemoval');
  assert.equal(typeof indexMod.norMetrics, 'function', 'index.js missing norMetrics');

  const gBb8 = mod.CsrGraph.fromEdgeList(
    8,
    new Uint32Array([
      0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3,
      3, 4,
      4, 5, 4, 6, 4, 7, 5, 6, 5, 7, 6, 7,
    ]),
    null,
    false,
    true,
  );
  const bb8Init = new Float32Array([-2, 0, -1.5, 0.5, -2.5, 0.5, -2, 1, 2, 0, 2.5, 0.5, 1.5, 0.5, 2, 1]);

  const lRes = mod.leidenCommunities(gBb8, { objective: 'modularity', resolution: 1.0, seed: 42 });
  assert.equal(lRes.numCommunities, 2);
  assert.ok(lRes.labelsView() instanceof Uint32Array);
  assert.equal(lRes.toRecord().numCommunities, 2);
  lRes.free?.();

  const lRun = mod.Layout2d.fromCoords(bb8Init);
  const lStep = mod.Layout2d.fromCoords(bb8Init);
  const llRun = new mod.LinLogSolver(gBb8, { iterations: 6, seed: 42 });
  const llStep = new mod.LinLogSolver(gBb8, { iterations: 6, seed: 42 });
  llRun.run(lRun);
  llStep.step(lStep);
  llStep.run(lStep);
  assert.deepEqual(Array.from(lRun.coordsView()), Array.from(lStep.coordsView()));
  llRun.free?.(); llStep.free?.();

  const frRun = new mod.FruchtermanReingoldSolver(gBb8, { iterations: 6, backend: 'barnesHut', seed: 42 });
  frRun.run(lRun);
  assert.equal(frRun.currentTemperature, 0);
  frRun.free?.();

  const bbRes = mod.extractQuadrilateralBackbone(gBb8, { retainRatio: 0.4 });
  assert.ok(bbRes.backboneEdges().length >= bbRes.umstEdges().length);
  bbRes.free?.();

  const tm = mod.squarifiedTreemap([3, 2, 1], { width: 100, height: 50, x: 0, y: 0 });
  assert.equal(tm.length, 12);

  const lOv = mod.Layout2d.fromCoords(new Float32Array([0, 0, 0.05, 0.02, -0.04, 0.03, 0.02, -0.04, 0.08, 0.06, -0.07, -0.05, 0.01, 0.05, -0.02, 0.01]));
  const hw = new Float32Array(8).fill(0.4);
  const hh = new Float32Array(8).fill(0.3);
  const gt = new mod.GTreeOverlapRemoval(hw, hh, { maxIterations: 25, seed: 42 });
  gt.run(lOv);
  assert.equal(gt.lastOverlapCount, 0);
  gt.free?.();

  const fb = mod.ForbidOverlapRemoval.fromRadii(hw, { variant: 'forbid', maxScaleIters: 20, sgdIterations: 20, seed: 42 });
  const lOvFb = mod.Layout2d.fromCoords(new Float32Array([0, 0, 0.05, 0.02, -0.04, 0.03, 0.02, -0.04, 0.08, 0.06, -0.07, -0.05, 0.01, 0.05, -0.02, 0.01]));
  fb.run(lOvFb);
  assert.equal(fb.lastOverlapCount, 0);
  fb.free?.();

  assertNonPanicError('community leiden bad obj', () => mod.leidenCommunities(gBb8, { objective: 'bad' }));
  assertNonPanicError('community linlog bad rep', () => new mod.LinLogSolver(gBb8, { repulsionModel: 'bad' }));
  assertNonPanicError('community fr bad backend', () => new mod.FruchtermanReingoldSolver(gBb8, { backend: 'bad' }));
  assertNonPanicError('community bb bad ratio', () => mod.extractQuadrilateralBackbone(gBb8, { retainRatio: 2.0 }));
  assertNonPanicError('community treemap empty', () => mod.squarifiedTreemap([]));
  assertNonPanicError('community gtree len mismatch', () => new mod.GTreeOverlapRemoval([1, 1], [1]));
  assertNonPanicError('community forbid bad variant', () => new mod.ForbidOverlapRemoval([1], [1], { variant: 'bad' }));
  assertNonPanicError('community eval dsc missing labels', () => mod.evaluateMetrics(gBb8, lRun, ['dsc']));

  lOvFb.free?.(); lOv.free?.(); lStep.free?.(); lRun.free?.(); gBb8.free?.();
}

assert.equal(typeof mod.InfoMap, 'undefined');
console.log('WASM_NODE_PARITY_OK');
"""


def run_wasm_node_parity(wasm_pkg: pathlib.Path) -> list[dict[str, Any]]:
    """Verify Node.js ESM import, CsrGraph/Layout2d coordsView(), and JsError guard."""
    t0 = time.perf_counter()
    if resolve_wasm_entries(wasm_pkg) is None:
        return [{
            "case": "wasm_node_esm_parity",
            "status": "skipped_no_index_js",
            "elapsed_sec": 0.0,
        }]

    combined_script = (
        f"{WASM_NODE_CORE_AND_LINALG_JS}\n"
        f"{WASM_NODE_SOLVERS_JS}\n"
        f"{_WASM_NODE_METRICS_AND_COMMUNITY_JS}"
    )
    run_node_script(wasm_pkg, combined_script)
    return [{
        "case": "wasm_node_esm_parity",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
    }]
