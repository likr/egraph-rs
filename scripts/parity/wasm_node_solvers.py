"""Node.js ESM parity checks for WASM clustering, manifold, force, overlap, and layering solvers."""

from __future__ import annotations

WASM_NODE_SOLVERS_JS = """
if (typeof mod.TsNetSolver === 'function') {
  const solverSymbols = [
    'CoarsenedGraphResult',
    'GuMapSolver',
    'KMeansResult',
    'KamadaKawaiSolver',
    'LouvainResult',
    'ManifoldLayout2d',
    'SgdSolver',
    'SparseBarrierStats',
    'SpatialIntruderRefiner',
    'SpatialRefinerRoundStats',
    'SpatialRefinerStats',
    'StressMajorizationSolver',
    'TsNetSolver',
    'coarsenGraph',
    'kmeansPlusPlus',
    'labelPropagation',
    'louvainCommunities',
    'projectClusteredLayout',
    'spectralClustering',
  ];
  for (const sym of solverSymbols) {
    assert.equal(typeof indexMod[sym], 'function', `index.js must export ${sym}`);
    assert.equal(typeof mod[sym], 'function', `node entry must export ${sym}`);
  }

  const bbEdges = [];
  for (const off of [0, 6]) {
    for (let i = 0; i < 6; i += 1) {
      for (let j = i + 1; j < 6; j += 1) bbEdges.push(off + i, off + j);
    }
  }
  bbEdges.push(5, 6);
  const bb = CsrGraph.fromEdgeList(12, new Uint32Array(bbEdges));
  const km = mod.kmeansPlusPlus([10, 10, 10.2, 9.8, 9.8, 10.1, -10, -10, -9.9, -10.2, -10.1, -9.8], 6, 2, 2, 100, 1e-4, 3, 42);
  assert.deepEqual(Array.from(km.labels()), [0, 0, 0, 1, 1, 1]);
  assert.deepEqual(Array.from(mod.spectralClustering(bb, 2, 300, 1e-4, 100, 1e-4, 42)), [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]);
  const lv = mod.louvainCommunities(bb, 100, 1.0, 1e-7, 42);
  assert.equal(lv.numCommunities, 2);
  assert.deepEqual(Array.from(mod.labelPropagation(bb, 100, 42)), [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]);
  const coarse = mod.coarsenGraph(bb, [100, 100, 100, 100, 100, 100, 999, 999, 999, 999, 999, 999]);
  assert.equal(coarse.numClusters, 2);
  const cl = Layout2d.fromCoords(new Float32Array([-2, 0, 2, 0]));
  const proj = mod.projectClusteredLayout(cl, coarse.nodeToCoarse(), 0.2, 42);
  assert.equal(proj.len(), 12);
  proj.free?.();
  cl.free?.();
  coarse.free?.();
  lv.free?.();
  km.free?.();

  const g16Edges = [];
  for (let r = 0; r < 4; r += 1) {
    for (let c = 0; c < 4; c += 1) {
      const u = r * 4 + c;
      if (c + 1 < 4) g16Edges.push(u, u + 1);
      if (r + 1 < 4) g16Edges.push(u, u + 4);
    }
  }
  const g16 = CsrGraph.fromEdgeList(16, new Uint32Array(g16Edges));
  const initArr = new Float32Array(32);
  for (let i = 0; i < 16; i += 1) {
    initArr[2 * i] = 0.7 * Math.cos(0.83 * i + 0.15) + 0.11 * i / 16.0;
    initArr[2 * i + 1] = 0.6 * Math.sin(1.13 * i - 0.21) - 0.09 * i / 16.0;
  }

  const tsR = mod.TsNetSolver.pure2Hop(g16, { backend: 'fft', iterationsStage1: 4, iterationsStage2: 4, iterationsStage3: 4, seed: 42 });
  const tsS = mod.TsNetSolver.pure2Hop(g16, { backend: 'fft', iterationsStage1: 4, iterationsStage2: 4, iterationsStage3: 4, seed: 42 });
  const lR = Layout2d.fromCoords(initArr);
  const lS = Layout2d.fromCoords(initArr);
  assert.equal(tsR.run(lR), 'completed');
  tsS.runSteps(lS, 5);
  while (!tsS.isFinished) tsS.step(lS);
  assert.deepEqual(Array.from(lR.coordsView()), Array.from(lS.coordsView()));
  tsR.free?.();
  tsS.free?.();
  lR.free?.();
  lS.free?.();

  const ml = mod.ManifoldLayout2d.fromCoords('torus', new Float32Array([1.2, -0.1, 0.9, 0.2]));
  assert.equal(ml.kind, 'torus');
  ml.projectAll();
  assert.ok(ml.geodesicDistance(0, 1) >= 0);
  ml.free?.();

  const sgdR = mod.SgdSolver.sparseBarrierDog(g16, { degree: 15, numLandmarks: 8, tMax: 8, seed: 42 });
  const sgdL = Layout2d.fromCoords(initArr);
  assert.equal(sgdR.run(sgdL), 'completed');
  assert.ok(sgdL.isAllFinite());
  sgdR.free?.();
  sgdL.free?.();

  const dirG = CsrGraph.fromEdgeList(3, new Uint32Array([0, 1, 1, 2]), null, true, true);
  let threwDir = false;
  try {
    mod.spectralClustering(dirG, 2);
  } catch (err) {
    assert.ok(!(err instanceof WebAssembly.RuntimeError));
    threwDir = true;
  }
  assert.ok(threwDir, 'spectralClustering must reject directed graphs');
  dirG.free?.();
  g16.free?.();
  bb.free?.();
}

if (typeof mod.TFdpSolver === 'function') {
  const forcePostprocSymbols = [
    'FdebSolver',
    'ForceAtlas2Solver',
    'IbUmapSolver',
    'PrismOverlapRemoval',
    'SeparationConstraint',
    'TFdpSolver',
    'YifanHuSolver',
    'assignLayers',
    'cycleEdges',
    'generateLayeredConstraints',
    'project1d',
    'projectClusteredRectangleNoOverlap2d',
    'projectLayoutAxis',
    'projectRectangleNoOverlap2d',
    'removeCycles',
  ];
  for (const sym of forcePostprocSymbols) {
    assert.equal(typeof indexMod[sym], 'function', `index.js must export ${sym}`);
    assert.equal(typeof mod[sym], 'function', `node entry must export ${sym}`);
  }
  assert.equal(typeof indexMod.runProgressive, 'function', 'index.js must export runProgressive');
  assert.equal(typeof indexMod.LayoutNd, 'function', 'index.js must export LayoutNd');
  assert.equal(typeof indexMod.LayoutNd.prototype.coordsView, 'function', 'LayoutNd must expose coordsView()');

  const g16Edges = [];
  for (let r = 0; r < 4; r += 1) {
    for (let c = 0; c < 4; c += 1) {
      const u = r * 4 + c;
      if (c + 1 < 4) g16Edges.push(u, u + 1);
      if (r + 1 < 4) g16Edges.push(u, u + 4);
    }
  }
  const g16 = CsrGraph.fromEdgeList(16, new Uint32Array(g16Edges));
  const init16 = new Float32Array(32);
  for (let i = 0; i < 16; i += 1) {
    init16[2 * i] = 0.7 * Math.cos(0.83 * i + 0.15) + 0.11 * i / 16.0;
    init16[2 * i + 1] = 0.6 * Math.sin(1.13 * i - 0.21) - 0.09 * i / 16.0;
  }

  const checkStepEqRun = (mkSolver, initArr, stepChunk, label) => {
    const sr = mkSolver();
    const ss = mkSolver();
    const lr = Layout2d.fromCoords(initArr);
    const ls = Layout2d.fromCoords(initArr);
    const stR = sr.run(lr);
    assert.ok(stR === 'completed' || stR === 'converged', `${label} run status=${stR}`);
    ss.runSteps(ls, stepChunk);
    while (!ss.isFinished) ss.step(ls);
    assert.deepEqual(Array.from(lr.coordsView()), Array.from(ls.coordsView()), `${label} step != run`);
    assert.equal(lr.isAllFinite(), true, `${label} non-finite`);
    sr.free?.();
    ss.free?.();
    lr.free?.();
    ls.free?.();
  };

  for (const bk of ['ibfft', 'barnes_hut', 'exact']) {
    checkStepEqRun(() => new mod.TFdpSolver(g16, { backend: bk, totalIterations: 8, seed: 42 }), init16, 3, `TFdp/${bk}`);
    checkStepEqRun(() => new mod.IbUmapSolver(g16, { backend: bk, nEpochs: 6, seed: 42 }), init16, 2, `IbUmap/${bk}`);
  }
  checkStepEqRun(() => mod.SgdSolver.snapTFdp(g16, { iterations: 8, negativesPerEdge: 3, seed: 42 }), init16, 3, 'Sgd.snapTFdp');
  checkStepEqRun(() => mod.SgdSolver.pacmapGraph(g16, { nNeighbors: 5, phase1Iters: 3, phase2Iters: 3, phase3Iters: 4, seed: 42 }), init16, 4, 'Sgd.pacmapGraph');
  checkStepEqRun(() => mod.SgdSolver.sgd2(g16, { tMax: 8, batchSize: 16, seed: 42 }), init16, 3, 'Sgd.sgd2');
  checkStepEqRun(() => mod.SgdSolver.full(g16, { tMax: 8, kind: 'squared', seed: 42 }), init16, 3, 'Sgd.squared');
  checkStepEqRun(() => new mod.ForceAtlas2Solver(g16, { iterations: 8, useBarnesHut: true, seed: 42 }), init16, 3, 'FA2/bh');
  checkStepEqRun(() => new mod.ForceAtlas2Solver(g16, { iterations: 8, useBarnesHut: false, seed: 42 }), init16, 3, 'FA2/exact');
  checkStepEqRun(() => new mod.YifanHuSolver(g16, { maxIterationsPerLevel: 8, minCoarsenSize: 6, minCoarsenRatio: 0.9, seed: 42 }), init16, 3, 'YifanHu');

  const prismInit = init16.map((v) => v * 0.15);
  checkStepEqRun(() => mod.PrismOverlapRemoval.fromRadii(new Float32Array(16).fill(0.4), { maxIterations: 40, seed: 42 }), prismInit, 2, 'PRISM');

  // FdebSolver step() == run() and input layout reuse (no value-drop bug)
  const fdebInit = new Float32Array(32);
  for (let r = 0; r < 4; r += 1) {
    for (let c = 0; c < 4; c += 1) {
      fdebInit[2 * (r * 4 + c)] = c;
      fdebInit[2 * (r * 4 + c) + 1] = r;
    }
  }
  const lFdeb = Layout2d.fromCoords(fdebInit);
  const fdebR = new mod.FdebSolver(g16, lFdeb, { cycles: 2, i0: 6 });
  const fdebS = new mod.FdebSolver(g16, lFdeb, { cycles: 2, i0: 6 });
  assert.equal(lFdeb.len(), 16, 'FdebSolver must not drop input Layout2d');
  assert.equal(fdebR.run(), 'completed');
  fdebS.step();
  fdebS.runSteps(3);
  while (!fdebS.isFinished) fdebS.step();
  assert.deepEqual(Array.from(fdebR.edgeOffsets()), Array.from(fdebS.edgeOffsets()));
  assert.deepEqual(Array.from(fdebR.points()), Array.from(fdebS.points()));
  assert.deepEqual(Array.from(lFdeb.coordsView()), Array.from(fdebInit));
  fdebR.free?.();
  fdebS.free?.();
  lFdeb.free?.();

  // runProgressive parity vs run()
  const pRun = new mod.ForceAtlas2Solver(g16, { iterations: 6, useBarnesHut: false, seed: 42 });
  const pProg = new mod.ForceAtlas2Solver(g16, { iterations: 6, useBarnesHut: false, seed: 42 });
  const lpRun = Layout2d.fromCoords(init16);
  const lpProg = Layout2d.fromCoords(init16);
  pRun.run(lpRun);
  const frames = [];
  const pCtx = await indexMod.runProgressive(pProg, lpProg, (ctx) => frames.push(ctx.iteration), { frameBudgetMs: 10, stepsPerYield: 1 });
  assert.equal(pCtx.status, 'completed');
  assert.ok(frames.length >= 1);
  assert.deepEqual(Array.from(lpRun.coordsView()), Array.from(lpProg.coordsView()));
  pRun.free?.(); pProg.free?.(); lpRun.free?.(); lpProg.free?.();

  // Constraints & layering smoke + adversarial JsError checks
  const c0 = new mod.SeparationConstraint(0, 1, 2.0);
  const arr1d = new Float32Array([1.0, 0.0]);
  mod.project1d(arr1d, [c0]);
  assert.ok(arr1d[1] - arr1d[0] >= 2.0 - 1e-5);
  c0.free?.();

  const cyc = CsrGraph.fromEdgeList(4, new Uint32Array([0, 1, 1, 2, 2, 0, 1, 3]), null, true, true);
  assert.ok(mod.cycleEdges(cyc).length >= 2);
  const dag = mod.removeCycles(cyc);
  assert.equal(mod.cycleEdges(dag).length, 0);
  const layersNs = mod.assignLayers(dag, 'networkSimplex');
  assert.equal(layersNs.length, 4);
  const lCs = mod.generateLayeredConstraints(dag, layersNs, 2.0);
  const lDag = Layout2d.zeros(4);
  mod.projectLayoutAxis(lDag, 1, lCs);
  for (const c of lCs) c.free?.();
  lDag.free?.();
  dag.free?.();
  cyc.free?.();

  const badL = Layout2d.zeros(3);
  assertNonPanicError('force tfdp bad backend', () => new mod.TFdpSolver(g16, { backend: 'bad' }));
  assertNonPanicError('force tfdp dim mismatch', () => new mod.TFdpSolver(g16).step(badL));
  assertNonPanicError('force ibumap zero tau', () => new mod.IbUmapSolver(g16, { tauScale: 0 }));
  assertNonPanicError('force snapTFdp bad rhoPower', () => mod.SgdSolver.snapTFdp(g16, { rhoPower: 0 }));
  assertNonPanicError('force pacmapGraph bad nNeighbors', () => mod.SgdSolver.pacmapGraph(g16, { nNeighbors: 0 }));
  assertNonPanicError('force sgd2 bad batchSize', () => mod.SgdSolver.sgd2(g16, { batchSize: 0 }));
  assertNonPanicError('force fa2 bad jitter', () => new mod.ForceAtlas2Solver(g16, { jitterTolerance: 0 }));
  assertNonPanicError('force yh bad stepRatio', () => new mod.YifanHuSolver(g16, { stepRatio: 1.0 }));
  assertNonPanicError('force prism len mismatch', () => new mod.PrismOverlapRemoval(new Float32Array([1, 1]), new Float32Array([1])));
  assertNonPanicError('force constraint nan', () => new mod.SeparationConstraint(0, 1, Number.NaN));
  assertNonPanicError('force assignLayers undir', () => mod.assignLayers(g16, 'longestPath'));
  assertNonPanicError('force fdeb dim mismatch', () => new mod.FdebSolver(g16, badL));
  badL.free?.();
  g16.free?.();
}
"""
