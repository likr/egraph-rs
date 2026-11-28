"""Node.js ESM parity checks for WASM core CsrGraph/Layout2d, spatial metrics, and linalg/neighborhood."""

from __future__ import annotations

WASM_NODE_CORE_AND_LINALG_JS = """
assert.equal(typeof indexMod.CsrGraph, 'function', 'index.js must export CsrGraph');
assert.equal(typeof indexMod.Layout2d, 'function', 'index.js must export Layout2d');
assert.equal(typeof indexMod.Layout2d.prototype.coordsView, 'function', 'Layout2d must expose coordsView()');

const { CsrGraph, Layout2d } = mod;
const g = CsrGraph.fromCsr(new Uint32Array([0, 1, 3, 4]), new Uint32Array([1, 0, 2, 1]));
const numNodes = typeof g.numNodes === 'function' ? g.numNodes() : g.numNodes;
const numEdges = typeof g.numEdges === 'function' ? g.numEdges() : g.numEdges;
const isDirected = typeof g.isDirected === 'function' ? g.isDirected() : g.isDirected;
assert.equal(numNodes, 3);
assert.equal(numEdges, 2);
assert.equal(isDirected, false);
g.free?.();

const layout = Layout2d.zeros(4);
assert.equal(typeof layout.coordsView, 'function');
const view = layout.coordsView();
assert.ok(view instanceof Float32Array, 'coordsView() must return Float32Array');
assert.equal(view.length, 8);
view[2] = 3.25;
view[3] = -1.5;
assert.ok(Math.abs(layout.x(1) - 3.25) < 1e-6);
assert.ok(Math.abs(layout.y(1) - (-1.5)) < 1e-6);
layout.setX(0, 7.5);
assert.ok(Math.abs(layout.coordsView()[0] - 7.5) < 1e-6);
assert.equal(layout.isAllFinite(), true);

let threwCsr = false;
try {
  CsrGraph.fromCsr(new Uint32Array([0, 1]), new Uint32Array([99]));
} catch (err) {
  assert.ok(!(err instanceof WebAssembly.RuntimeError), 'CSR error must map to JsError, not RuntimeError');
  threwCsr = true;
}
assert.ok(threwCsr, 'CsrGraph.fromCsr must reject out-of-bounds vertex index');

let threwLayout = false;
try {
  layout.x(99);
} catch (err) {
  assert.ok(!(err instanceof WebAssembly.RuntimeError), 'Layout2d error must map to JsError, not RuntimeError');
  threwLayout = true;
}
assert.ok(threwLayout, 'Layout2d.x must reject out-of-bounds index');
layout.free?.();

if (typeof mod.scaleNormalizedStress === 'function') {
  assert.equal(typeof indexMod.scaleNormalizedStress, 'function');
  assert.equal(typeof indexMod.sweepCrossingNumber, 'function');
  assert.equal(typeof indexMod.neighborhoodPreservation1Hop, 'function');
  assert.equal(typeof indexMod.neighborhoodPreservation2Hop, 'function');
  assert.equal(typeof indexMod.prismDistortion, 'function');
  assert.equal(typeof indexMod.evaluateMetrics, 'function');
  assert.equal(typeof indexMod.MetricReport, 'function');
  assert.equal(typeof indexMod.PrismDistortion, 'function');

  const bbEdges = new Uint32Array([
    0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3, 3, 4, 4, 5,
    5, 6, 5, 7, 5, 8, 6, 7, 6, 8, 7, 8,
  ]);
  const bbCoords = new Float32Array([
    -1.9, 0.0, -2.5, 0.6, -3.1, 0.0, -2.5, -0.6,
    0.0, 0.25,
    3.063, 0.206, 2.294, 0.563, 1.937, -0.206, 2.706, -0.563,
  ]);
  const gBb = CsrGraph.fromEdgeList(9, bbEdges);
  const lBb = Layout2d.fromCoords(bbCoords);
  const lBbScaled = Layout2d.fromCoords(bbCoords.map((v) => v * 100.0));
  const lBbSmall = Layout2d.fromCoords(bbCoords.map((v) => v * 0.01));

  const sns1 = mod.scaleNormalizedStress(gBb, lBb);
  const sns100 = mod.scaleNormalizedStress(gBb, lBbScaled);
  const sns001 = mod.scaleNormalizedStress(gBb, lBbSmall);
  assert.ok(Number.isFinite(sns1) && sns1 >= 0.0 && sns1 <= 1.0);
  assert.ok(Math.abs(sns100 - sns1) < 1e-5, 'WASM SNS must be scale-invariant (100x)');
  assert.ok(Math.abs(sns001 - sns1) < 1e-5, 'WASM SNS must be scale-invariant (0.01x)');

  const cr = mod.sweepCrossingNumber(gBb, lBb);
  const np1 = mod.neighborhoodPreservation1Hop(gBb, lBb);
  const np2 = mod.neighborhoodPreservation2Hop(gBb, lBb);
  assert.ok(Number.isInteger(cr) && cr >= 0);
  assert.ok(Number.isFinite(np1) && np1 >= 0.0 && np1 <= 1.0);
  assert.ok(Number.isFinite(np2) && np2 >= 0.0 && np2 <= 1.0);

  const pd = mod.prismDistortion(lBb, lBbScaled);
  assert.ok(pd.sigmaDist < 1e-5 && pd.sigmaDisp < 1e-5);
  pd.free?.();

  const rep = mod.evaluateMetrics(gBb, lBb, ['sns', 'crossingNumber', 'np1', 'np2'], { numPivots: 4, numSamples: 200, seed: 42 });
  const rec = rep.toRecord();
  assert.deepEqual(
    Object.keys(rec).sort(),
    ['crossingNumber', 'neighborhoodPreservation1Hop', 'neighborhoodPreservation2Hop', 'scaleNormalizedStress'],
  );
  assert.ok(Math.abs(rec.scaleNormalizedStress - sns1) < 1e-12);
  assert.equal(rec.crossingNumber, cr);
  assert.ok(Math.abs(rec.neighborhoodPreservation1Hop - np1) < 1e-12);
  assert.ok(Math.abs(rec.neighborhoodPreservation2Hop - np2) < 1e-12);
  assert.equal(rep.aspectRatio, undefined);
  rep.free?.();

  if (typeof mod.delaunayTriangulation === 'function') {
    const dt = mod.delaunayTriangulation(lBb);
    assert.ok(dt.edges() instanceof Uint32Array && dt.edges().length > 0);
    assert.ok(dt.triangles() instanceof Uint32Array && dt.triangles().length > 0);
    dt.free?.();
  }

  const lShort = Layout2d.zeros(3);
  const badCoords = new Float32Array(bbCoords);
  badCoords[0] = Number.NaN;
  assertNonPanicError('Layout2d.fromCoords nan', () => Layout2d.fromCoords(badCoords));
  const lNan = Layout2d.fromCoords(bbCoords);
  lNan.coordsView()[0] = Number.NaN;
  assertNonPanicError('metrics sns dim mismatch', () => mod.scaleNormalizedStress(gBb, lShort));
  assertNonPanicError('metrics sweep nan', () => mod.sweepCrossingNumber(gBb, lNan));
  assertNonPanicError('metrics bad targets', () => mod.evaluateMetrics(gBb, lBb, 'sns'));
  assertNonPanicError('metrics zero pivots', () => mod.sampledScaleNormalizedStress(gBb, lBb, { numPivots: 0, seed: 42 }));
  assertNonPanicError('metrics neg seed', () => mod.sampledScaleNormalizedStress(gBb, lBb, { numPivots: 4, seed: -1 }));

  lShort.free?.();
  lNan.free?.();
  gBb.free?.();
  lBb.free?.();
  lBbScaled.free?.();
  lBbSmall.free?.();
}

if (typeof mod.selectMaxminSpPivots === 'function') {
  const linalgSymbols = [
    'KnnGraph',
    'Sparse12HopPairs',
    'SparseProbabilities',
    'DogVaradhanLandmarks',
    'EigenResult',
    'SpectralMomentStats',
    'TwoHopCsr',
    'TwoHopKnnResult',
    'HybridKnnResult',
    'selectMaxminSpPivots',
    'pivotMdsInit',
    'korenSpectralInit',
    'smallestLaplacianEigenpairs',
    'spectralMomentStats',
    'chebyshevHeatSpmv',
    'chebyshevDogJlEmbedding',
    'extract1hop2hopCsr',
    'extractSparse1hop2hopPairs',
    'buildPure2hopKnn',
    'compute2hopNeighborhoodSizes',
    'computeNodewise2hopPerplexities',
    'computeMedian2hopPerplexity',
    'buildHybridAdaptiveKnn',
    'partialBfsKnn',
    'computeKnnProbabilities',
    'computeFuzzySimplicialSet',
    'findAbParams',
    'computeHorvitzThompsonGamma',
    'computeRegularizedDogVaradhanLandmarks',
  ];
  for (const sym of linalgSymbols) {
    assert.equal(typeof indexMod[sym], 'function', `index.js must export ${sym}`);
    assert.equal(typeof mod[sym], 'function', `node entry must export ${sym}`);
  }

  const gEdges = [];
  for (let r = 0; r < 4; r += 1) {
    for (let c = 0; c < 4; c += 1) {
      const u = r * 4 + c;
      if (c + 1 < 4) gEdges.push(u, u + 1);
      if (r + 1 < 4) gEdges.push(u, u + 4);
    }
  }
  const g16 = CsrGraph.fromEdgeList(16, new Uint32Array(gEdges));

  const piv = mod.selectMaxminSpPivots(g16, { numPivots: 6, seed: 42, minDegree: 2 });
  assert.ok(piv instanceof Uint32Array && piv.length === 6);

  const lPmds = mod.pivotMdsInit(g16, { numPivots: 6, seed: 42, minDegree: 2, targetStd: 1.0 });
  const lPmdsUnnorm = mod.pivotMdsInit(g16, { numPivots: 6, seed: 42, minDegree: 2, targetStd: null });
  const lKoren = mod.korenSpectralInit(g16, { targetStd: 1.0, tol: 1e-4, maxIterations: 300, seed: 42 });
  assert.equal(lPmds.len(), 16);
  assert.equal(lPmdsUnnorm.len(), 16);
  assert.equal(lKoren.len(), 16);
  assert.equal(lPmds.isAllFinite(), true);
  assert.equal(lKoren.isAllFinite(), true);

  const eig = mod.smallestLaplacianEigenpairs(g16, { nModes: 4, tol: 1e-4, maxIterations: 300, seed: 42 });
  assert.equal(eig.numNodes, 16);
  assert.equal(eig.numModes, 4);
  const invD = eig.invSqrtDeg();
  assert.equal(eig.lambdas().length, 4);
  assert.equal(eig.vectors().length, 64);

  const mStats = mod.spectralMomentStats(g16);
  assert.ok(mStats.sigmaLambdaSq > 0 && Math.abs(mStats.meanDegree - 3.0) < 1e-12);

  const statVec = new Float64Array(16);
  for (let i = 0; i < 16; i += 1) statVec[i] = 1.0 / invD[i];
  const heat = mod.chebyshevHeatSpmv(g16, statVec, 1, 2.0);
  for (let i = 0; i < 16; i += 1) assert.ok(Math.abs(heat[i] - statVec[i]) < 1e-9);

  const jl = mod.chebyshevDogJlEmbedding(g16, { jlDim: 12, degree: 25, seed: 42, degreeNormalize: true, epsPsd: 1e-4 });
  assert.ok(jl instanceof Float64Array && jl.length === 16 * 12);

  const twoHop = mod.extract1hop2hopCsr(g16);
  const h1 = twoHop.hop1();
  const h2 = twoHop.hop2();
  assert.equal(h1.numEdges(), 24);
  assert.ok(h2.numEdges() > 0);

  const pureRes = mod.buildPure2hopKnn(g16);
  const pureKnn = pureRes.knn();
  assert.ok(pureKnn.indptr() instanceof Uint32Array, 'KnnGraph.indptr() must be Uint32Array');
  assert.deepEqual(Array.from(pureRes.k2Deg()), Array.from(mod.compute2hopNeighborhoodSizes(g16)));
  const knnRt = mod.KnnGraph.fromCsr(pureKnn.numNodes, pureKnn.indptr(), pureKnn.indices(), pureKnn.distances());
  assert.equal(knnRt.nnz, pureKnn.nnz);

  const perps = mod.computeNodewise2hopPerplexities(g16);
  const medPerp = mod.computeMedian2hopPerplexity(g16);
  assert.ok(perps.length === 16 && Number.isFinite(medPerp));

  const hybRes = mod.buildHybridAdaptiveKnn(g16, jl, 12, perps);
  const hybKnn = hybRes.knn();
  assert.equal(hybRes.nFilledResidual + hybRes.nPure2hopRows, 16);

  const bfsKnn = mod.partialBfsKnn(g16, 5, 42);
  assert.equal(bfsKnn.nnz, 16 * 5);

  const probs = mod.computeKnnProbabilities(hybKnn, perps);
  const pSum = probs.values().reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(pSum - 1.0) < 1e-10);

  const fuzzy = mod.computeFuzzySimplicialSet(hybKnn);
  assert.ok(fuzzy.values().every((v) => v > 1e-7 && v <= 1.0));

  const ab = mod.findAbParams(0.1, 1.0);
  assert.ok(Math.abs(ab[0] - 1.577) < 0.02 && Math.abs(ab[1] - 0.895) < 0.02);
  assert.ok(Math.abs(mod.computeHorvitzThompsonGamma(1.0, 1600, 16) - 10.0) < 1e-12);

  const lm = mod.computeRegularizedDogVaradhanLandmarks(g16, { degree: 25, numLandmarks: 16, seed: 42 });
  assert.equal(lm.numNodes, 16);
  assert.ok(lm.coarseTable() instanceof Float64Array);
  const alt = lm.altDistances(new Uint32Array([0, 15]), new Uint32Array([15, 0]));
  assert.ok(Math.abs(alt[0] - alt[1]) < 1e-12);

  assertNonPanicError('linalg bad seed', () => mod.selectMaxminSpPivots(g16, { numPivots: 4, seed: 1.5 }));
  assertNonPanicError('linalg neg targetStd', () => mod.pivotMdsInit(g16, { numPivots: 4, seed: 42, minDegree: 2, targetStd: -1 }));
  assertNonPanicError('linalg nan vec', () => mod.chebyshevHeatSpmv(g16, new Float64Array([Number.NaN]), 1, 1.0));
  assertNonPanicError('linalg neg dist', () => mod.KnnGraph.fromCsr(2, new Uint32Array([0, 1, 1]), new Uint32Array([1]), new Float64Array([-1.0])));
  assertNonPanicError('linalg alt oob', () => lm.altDistances(new Uint32Array([0]), new Uint32Array([99])));
  assertNonPanicError('linalg neg gamma', () => mod.computeHorvitzThompsonGamma(-1.0, 10, 4));
  assertNonPanicError('linalg neg minDist', () => mod.findAbParams(-0.1, 1.0));
  assertNonPanicError('linalg zero sigmaIters', () => mod.computeFuzzySimplicialSet(hybKnn, 0));

  lm.free?.();
  fuzzy.free?.();
  probs.free?.();
  bfsKnn.free?.();
  hybKnn.free?.();
  hybRes.free?.();
  knnRt.free?.();
  pureKnn.free?.();
  pureRes.free?.();
  h1.free?.();
  h2.free?.();
  twoHop.free?.();
  mStats.free?.();
  eig.free?.();
  lKoren.free?.();
  lPmdsUnnorm.free?.();
  lPmds.free?.();
  g16.free?.();
}
"""
