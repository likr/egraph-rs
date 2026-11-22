import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import init, * as eg from "../pkg/egraph_wasm.node.js";
import * as webPkg from "../pkg/index.js";

const {
  CsrGraph,
  DogVaradhanLandmarks,
  EigenResult,
  HybridKnnResult,
  KnnGraph,
  Layout2d,
  MetricReport,
  PrismDistortion,
  SparseProbabilities,
  SpectralMomentStats,
  TwoHopCsr,
  TwoHopKnnResult,
  angularResolutionV6,
  aspectRatioV6,
  buildHybridAdaptiveKnn,
  buildPure2hopKnn,
  chebyshevDogJlEmbedding,
  chebyshevHeatSpmv,
  compute2hopNeighborhoodSizes,
  computeFuzzySimplicialSet,
  computeHorvitzThompsonGamma,
  computeKnnProbabilities,
  computeMedian2hopPerplexity,
  computeNodewise2hopPerplexities,
  computeRegularizedDogVaradhanLandmarks,
  dggPreservation,
  drngPreservation,
  edgeAngle,
  evaluateMetrics,
  extract1hop2hopCsr,
  findAbParams,
  gabrielGraphPropertyV6,
  idealEdgeLengthsV6,
  korenSpectralInit,
  neighborhoodPreservation1Hop,
  neighborhoodPreservation2HopV6,
  nodeResolutionV6,
  partialBfsKnn,
  pivotMdsInit,
  prismDistortion,
  sampledCrossingNumber,
  sampledScaleNormalizedStress,
  scaleNormalizedStress,
  CoarsenedGraphResult,
  GuMapSolver,
  KMeansResult,
  KamadaKawaiSolver,
  LouvainResult,
  ManifoldLayout2d,
  SgdSolver,
  SparseBarrierStats,
  SpatialIntruderRefiner,
  SpatialRefinerRoundStats,
  SpatialRefinerStats,
  StressMajorizationSolver,
  TsNetSolver,
  FdebSolver,
  ForceAtlas2Solver,
  IbUmapSolver,
  PrismOverlapRemoval,
  SeparationConstraint,
  TFdpSolver,
  YifanHuSolver,
  assignLayers,
  coarsenGraph,
  cycleEdges,
  generateLayeredConstraints,
  kmeansPlusPlus,
  labelPropagation,
  louvainCommunities,
  project1d,
  projectClusteredLayout,
  projectClusteredRectangleNoOverlap2d,
  projectLayoutAxis,
  projectRectangleNoOverlap2d,
  removeCycles,
  selectMaxminSpPivots,
  setPanicHook,
  smallestLaplacianEigenpairs,
  spectralClustering,
  spectralMomentStats,
  sweepCrossingAngle,
  sweepCrossingNumber,
  torusCrossingNumber,
  wasmMemory,
} = eg;

// Cross-platform init() should resolve cleanly even when Node loader pre-initialized WASM.
await init();
setPanicHook();

assert.equal(typeof webPkg.Layout2d.prototype.coordsView, "function");
assert.equal(typeof Layout2d.prototype.coordsView, "function");

// 1. CsrGraph.fromCsr: strict valid undirected weighted graph
{
  const indptr = new Uint32Array([0, 1, 3, 4]);
  const indices = new Uint32Array([1, 0, 2, 1]);
  const weights = new Float32Array([1.5, 1.5, 2.5, 2.5]);

  const g = CsrGraph.fromCsr(indptr, indices, weights, {
    directed: false,
    sanitize: false,
  });
  assert.equal(g.numNodes(), 3);
  assert.equal(g.numEdges(), 2);
  assert.equal(g.isDirected(), false);
  assert.deepEqual(Array.from(g.indptr()), [0, 1, 3, 4]);
  assert.deepEqual(Array.from(g.indices()), [1, 0, 2, 1]);
  assert.deepEqual(Array.from(g.weights()), [1.5, 1.5, 2.5, 2.5]);
  g.free();
}

// 2. CsrGraph.fromCsr: default sanitization + boolean trailing params
{
  const dirtyIndptr = new Uint32Array([0, 4, 5, 5]);
  const dirtyIndices = new Uint32Array([0, 2, 1, 2, 0]);
  const dirtyWeights = new Float32Array([9.0, 2.0, 1.5, 4.0, 1.5]);

  assert.throws(
    () => CsrGraph.fromCsr(dirtyIndptr, dirtyIndices, dirtyWeights, false, false),
    Error,
  );

  const sanitized = CsrGraph.fromCsr(dirtyIndptr, dirtyIndices, dirtyWeights);
  assert.equal(sanitized.numNodes(), 3);
  assert.equal(sanitized.numEdges(), 2);
  assert.deepEqual(Array.from(sanitized.indptr()), [0, 2, 3, 4]);
  assert.deepEqual(Array.from(sanitized.indices()), [1, 2, 0, 0]);
  assert.deepEqual(Array.from(sanitized.weights()), [1.5, 4.0, 1.5, 4.0]);
  sanitized.free();
}

// 3. CsrGraph.fromEdgeList: directed and undirected
{
  const dg = CsrGraph.fromEdgeList(
    3,
    new Uint32Array([0, 1, 0, 2]),
    undefined,
    { directed: true, sanitize: false },
  );
  assert.equal(dg.isDirected(), true);
  assert.equal(dg.numNodes(), 3);
  assert.equal(dg.numEdges(), 2);
  assert.deepEqual(Array.from(dg.indptr()), [0, 2, 2, 2]);
  assert.deepEqual(Array.from(dg.indices()), [1, 2]);
  assert.equal(dg.weights(), undefined);
  dg.free();

  const ug = CsrGraph.fromEdgeList(
    3,
    new Uint32Array([0, 1, 1, 2]),
    new Float32Array([0.5, 1.25]),
  );
  assert.equal(ug.isDirected(), false);
  assert.equal(ug.numEdges(), 2);
  assert.deepEqual(Array.from(ug.indptr()), [0, 1, 3, 4]);
  assert.deepEqual(Array.from(ug.indices()), [1, 0, 2, 1]);
  assert.deepEqual(Array.from(ug.weights()), [0.5, 0.5, 1.25, 1.25]);
  ug.free();
}

// 4. Layout2d zero-copy coordsView() + memory.grow resilience
{
  const empty = Layout2d.zeros(0);
  assert.equal(empty.len(), 0);
  assert.equal(empty.isEmpty(), true);
  assert.equal(empty.coordsLen(), 0);
  assert.equal(empty.coordsView().length, 0);
  empty.free();

  const layout = Layout2d.zeros(4);
  assert.equal(layout.len(), 4);
  assert.equal(layout.isEmpty(), false);
  assert.equal(layout.coordsLen(), 8);
  assert.equal(layout.isAllFinite(), true);

  const view1 = layout.coordsView();
  const view2 = layout.coordsView();
  assert(view1 instanceof Float32Array);
  assert.equal(view1.length, 8);
  assert.equal(view1, view2, "coordsView() must cache Float32Array while buffer is unchanged");

  // Rust -> JS zero-copy visibility
  layout.setX(0, 1.5);
  layout.setY(0, 2.5);
  assert.equal(view1[0], 1.5);
  assert.equal(view1[1], 2.5);

  // JS -> Rust zero-copy visibility
  view1[2] = 7.5;
  view1[3] = -3.25;
  assert.equal(layout.x(1), 7.5);
  assert.equal(layout.y(1), -3.25);

  // Force WebAssembly linear memory growth to detach the old ArrayBuffer
  const oldBuf = view1.buffer;
  wasmMemory().grow(1);
  assert.equal(oldBuf.byteLength, 0, "expected old ArrayBuffer to be detached after memory.grow");

  const viewAfterGrow = layout.coordsView();
  assert.notEqual(viewAfterGrow, view1);
  assert.notEqual(viewAfterGrow.buffer, oldBuf);
  assert(viewAfterGrow.buffer.byteLength > 0);
  assert.equal(viewAfterGrow[0], 1.5);
  assert.equal(viewAfterGrow[1], 2.5);
  assert.equal(viewAfterGrow[2], 7.5);
  assert.equal(viewAfterGrow[3], -3.25);

  viewAfterGrow[4] = 9.0;
  viewAfterGrow[5] = -8.5;
  assert.equal(layout.x(2), 9.0);
  assert.equal(layout.y(2), -8.5);

  // Finiteness checks
  layout.setX(3, Number.NaN);
  assert.equal(layout.isAllFinite(), false);
  layout.setX(3, 0.0);
  assert.equal(layout.isAllFinite(), true);
  layout.setY(3, Number.POSITIVE_INFINITY);
  assert.equal(layout.isAllFinite(), false);

  layout.free();
}

// 5. Layout2d.fromCoords round-trip
{
  const layout = Layout2d.fromCoords(new Float32Array([1.0, -2.0, 3.5, 4.5]));
  assert.equal(layout.len(), 2);
  assert.equal(layout.x(0), 1.0);
  assert.equal(layout.y(0), -2.0);
  assert.equal(layout.x(1), 3.5);
  assert.equal(layout.y(1), 4.5);
  assert.deepEqual(Array.from(layout.coordsView()), [1.0, -2.0, 3.5, 4.5]);
  layout.free();
}

// 6. Adversarial inputs must throw Error (never crash/unreachable trap)
assert.throws(
  () => CsrGraph.fromCsr(new Uint32Array([]), new Uint32Array([])),
  Error,
);
assert.throws(
  () => CsrGraph.fromCsr(new Uint32Array([0, 1, 1]), new Uint32Array([5])),
  Error,
);
assert.throws(
  () =>
    CsrGraph.fromCsr(
      new Uint32Array([0, 1, 2]),
      new Uint32Array([1, 0]),
      new Float32Array([Number.NaN, 1.0]),
    ),
  Error,
);
assert.throws(
  () =>
    CsrGraph.fromCsr(
      new Uint32Array([0, 1, 2, 3]),
      new Uint32Array([1, 2, 0]),
      undefined,
      { directed: false, sanitize: false },
    ),
  Error,
);
assert.throws(
  () => CsrGraph.fromCsr(new Uint32Array([0, 0]), new Uint32Array([]), undefined, { directed: "yes" }),
  Error,
);
assert.throws(
  () => CsrGraph.fromEdgeList(3, new Uint32Array([0, 1, 2])),
  Error,
);
assert.throws(
  () => CsrGraph.fromEdgeList(-1, new Uint32Array([])),
  Error,
);
assert.throws(
  () => CsrGraph.fromEdgeList(1.5, new Uint32Array([])),
  Error,
);
assert.throws(
  () => Layout2d.fromCoords(new Float32Array([1.0, 2.0, 3.0])),
  Error,
);
assert.throws(() => Layout2d.zeros(-1), Error);
assert.throws(() => Layout2d.zeros(1.5), Error);
assert.throws(() => Layout2d.zeros(Number.NaN), Error);
{
  const l = Layout2d.zeros(2);
  assert.throws(() => l.x(2), Error);
  assert.throws(() => l.y(-1), Error);
  assert.throws(() => l.setX(1.5, 0), Error);
  assert.throws(() => l.setY(99, 0), Error);
  l.free();
}

// 7. Verify M1 v6 egraph-metrics and egraph-spatial APIs
{
  assert.equal(typeof webPkg.scaleNormalizedStress, "function");
  assert.equal(typeof webPkg.evaluateMetrics, "function");
  assert.equal(typeof webPkg.MetricReport, "function");
  assert.equal(typeof webPkg.PrismDistortion, "function");

  // 4x4 grid layout (16 nodes, 24 edges, planar crystalline)
  const gridEdges = [];
  const gridCoords = [];
  for (let r = 0; r < 4; r += 1) {
    for (let c = 0; c < 4; c += 1) {
      const id = r * 4 + c;
      gridCoords.push(c, r);
      if (c + 1 < 4) gridEdges.push(id, id + 1);
      if (r + 1 < 4) gridEdges.push(id, id + 4);
    }
  }
  const grid = CsrGraph.fromEdgeList(16, new Uint32Array(gridEdges));
  const gridLayout = Layout2d.fromCoords(new Float32Array(gridCoords));
  const scaledCoords = new Float32Array(gridCoords.length);
  for (let i = 0; i < gridCoords.length; i += 2) {
    scaledCoords[i] = 10.0 * gridCoords[i] + 5.0;
    scaledCoords[i + 1] = 10.0 * gridCoords[i + 1] - 3.0;
  }
  const gridScaled = Layout2d.fromCoords(scaledCoords);

  const sns1 = scaleNormalizedStress(grid, gridLayout);
  const sns10 = scaleNormalizedStress(grid, gridScaled);
  assert(Number.isFinite(sns1) && sns1 >= 0 && sns1 < 0.05);
  assert(Math.abs(sns1 - sns10) < 1e-5, "SNS must be scale-invariant");

  const sampledSnsDefault = sampledScaleNormalizedStress(grid, gridLayout);
  const sampledSns4 = sampledScaleNormalizedStress(grid, gridLayout, 16, 42);
  assert(Math.abs(sampledSnsDefault - sns1) < 1e-5);
  assert(Number.isFinite(sampledSns4) && sampledSns4 >= 0);

  assert.equal(sweepCrossingNumber(grid, gridLayout), 0);
  assert.equal(sweepCrossingAngle(grid, gridLayout), 0);
  assert.equal(sampledCrossingNumber(grid, gridLayout), 0);

  // K4 square with crossing diagonals
  const k4 = CsrGraph.fromEdgeList(4, new Uint32Array([0, 1, 1, 2, 2, 3, 3, 0, 0, 2, 1, 3]));
  const k4Layout = Layout2d.fromCoords(new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]));
  assert.equal(sweepCrossingNumber(k4, k4Layout), 1);
  assert(Math.abs(sweepCrossingAngle(k4, k4Layout)) < 1e-6);
  assert.equal(sampledCrossingNumber(k4, k4Layout, 100, 42), 1);
  k4.free();
  k4Layout.free();

  // Torus crossing on [0, 1)^2
  const torusGraph = CsrGraph.fromEdgeList(4, new Uint32Array([0, 1, 2, 3]));
  const torusLayout = Layout2d.fromCoords(new Float32Array([0.9, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1]));
  assert(torusCrossingNumber(torusGraph, torusLayout) >= 1);
  torusGraph.free();
  torusLayout.free();

  assert(Math.abs(neighborhoodPreservation1Hop(grid, gridLayout) - 1.0) < 1e-6);
  assert(Math.abs(neighborhoodPreservation2HopV6(grid, gridLayout) - 1.0) < 1e-6);
  assert(Math.abs(drngPreservation(grid, gridLayout) - 1.0) < 1e-6);
  assert(Math.abs(dggPreservation(grid, gridLayout) - 1.0) < 1e-6);
  assert.equal(gabrielGraphPropertyV6(grid, gridLayout), 0);
  assert(nodeResolutionV6(gridLayout) > 0);
  assert(Math.abs(aspectRatioV6(gridLayout) - 1.0) < 1e-6);
  assert(angularResolutionV6(grid, gridLayout) > 0);
  assert(Math.abs(idealEdgeLengthsV6(grid, gridLayout)) < 1e-6);

  assert(Math.abs(edgeAngle(1, 0, 0, 1) - Math.PI / 2) < 1e-12);
  assert.equal(edgeAngle(0, 0, 1, 0), undefined);
  assert.equal(edgeAngle(Number.NaN, 0, 1, 0), undefined);

  const distortion = prismDistortion(gridLayout, gridScaled);
  assert(distortion instanceof PrismDistortion);
  assert(distortion.sigmaDist < 1e-5);
  assert(distortion.sigmaDisp < 1e-5);
  distortion.free();

  // Selective metric evaluation without serde
  const report = evaluateMetrics(grid, gridLayout, ["sns", "np2", "aspectRatio", "drng"], 8, 64, 42);
  assert(report instanceof MetricReport);
  assert(Math.abs(report.scaleNormalizedStress - sns1) < 1e-12);
  assert(Math.abs(report.neighborhoodPreservation2Hop - 1.0) < 1e-6);
  assert(Math.abs(report.aspectRatio - 1.0) < 1e-6);
  assert(Math.abs(report.relativeNeighborhoodPreservation - 1.0) < 1e-6);
  assert.equal(report.crossingNumber, undefined);
  assert.equal(report.idealEdgeLengths, undefined);

  const rec = report.toRecord();
  assert.deepEqual(
    Object.keys(rec).sort(),
    [
      "aspectRatio",
      "neighborhoodPreservation2Hop",
      "relativeNeighborhoodPreservation",
      "scaleNormalizedStress",
    ],
  );
  assert(Math.abs(rec.scaleNormalizedStress - sns1) < 1e-12);
  report.free();

  // Delaunay triangulation when `delaunay` feature is enabled
  if (typeof eg.delaunayTriangulation === "function") {
    assert.equal(typeof webPkg.delaunayTriangulation, "function");
    const sq = Layout2d.fromCoords(new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]));
    const dt = eg.delaunayTriangulation(sq);
    const dtEdges = dt.edges();
    const dtTris = dt.triangles();
    assert(dtEdges instanceof Uint32Array);
    assert(dtTris instanceof Uint32Array);
    assert.equal(dt.numNodes, 4);
    assert.equal(dtEdges.length, 10);
    assert.equal(dtTris.length, 6);
    dt.free();
    sq.free();
  }

  // Adversarial inputs must throw Error (never WebAssembly.RuntimeError)
  const assertJsError = (fn, label) => {
    let threw = false;
    try {
      fn();
    } catch (err) {
      assert(!(err instanceof WebAssembly.RuntimeError), `${label} trapped with RuntimeError`);
      assert(err instanceof Error, `${label} must throw Error`);
      threw = true;
    }
    assert(threw, `${label} must throw`);
  };

  const shortLayout = Layout2d.zeros(3);
  const nanLayout = Layout2d.fromCoords(new Float32Array([Number.NaN, 0, ...gridCoords.slice(2)]));
  assertJsError(() => scaleNormalizedStress(grid, shortLayout), "sns dimension mismatch");
  assertJsError(() => sweepCrossingNumber(grid, nanLayout), "crossings non-finite");
  assertJsError(() => neighborhoodPreservation1Hop(grid, shortLayout), "np1 dimension mismatch");
  assertJsError(() => prismDistortion(gridLayout, shortLayout), "prism dimension mismatch");
  assertJsError(() => sampledScaleNormalizedStress(grid, gridLayout, 0, 42), "numPivots=0");
  assertJsError(() => sampledScaleNormalizedStress(grid, gridLayout, 4, -1), "seed=-1");
  assertJsError(() => sampledScaleNormalizedStress(grid, gridLayout, 4, 1.5), "seed=1.5");
  assertJsError(
    () => sampledScaleNormalizedStress(grid, gridLayout, 4, 9007199254740992),
    "seed > MAX_SAFE_INTEGER",
  );
  assertJsError(() => sampledCrossingNumber(grid, gridLayout, 0, 42), "numSamples=0");
  assertJsError(() => evaluateMetrics(grid, gridLayout, "sns"), "non-array targets");
  assertJsError(() => evaluateMetrics(grid, gridLayout, [123]), "non-string target element");
  assertJsError(() => evaluateMetrics(grid, gridLayout, ["bogusMetric"]), "unknown target name");
  if (typeof eg.delaunayTriangulation === "function") {
    assertJsError(() => eg.delaunayTriangulation(nanLayout), "delaunay non-finite");
  }
  shortLayout.free();
  nanLayout.free();
  grid.free();
  gridLayout.free();
  gridScaled.free();
}

// 8. Verify M2 v6 egraph-linalg and egraph-neighborhood APIs
{
  const m2Symbols = [
    "KnnGraph",
    "SparseProbabilities",
    "DogVaradhanLandmarks",
    "EigenResult",
    "SpectralMomentStats",
    "TwoHopCsr",
    "TwoHopKnnResult",
    "HybridKnnResult",
    "selectMaxminSpPivots",
    "pivotMdsInit",
    "korenSpectralInit",
    "smallestLaplacianEigenpairs",
    "spectralMomentStats",
    "chebyshevHeatSpmv",
    "chebyshevDogJlEmbedding",
    "extract1hop2hopCsr",
    "buildPure2hopKnn",
    "compute2hopNeighborhoodSizes",
    "computeNodewise2hopPerplexities",
    "computeMedian2hopPerplexity",
    "buildHybridAdaptiveKnn",
    "partialBfsKnn",
    "computeKnnProbabilities",
    "computeFuzzySimplicialSet",
    "findAbParams",
    "computeHorvitzThompsonGamma",
    "computeRegularizedDogVaradhanLandmarks",
  ];
  for (const sym of m2Symbols) {
    assert.equal(typeof webPkg[sym], "function", `missing webPkg.${sym}`);
    assert.equal(typeof eg[sym], "function", `missing eg.${sym}`);
  }

  // 4x4 grid graph (16 nodes)
  const gridEdges = [];
  for (let r = 0; r < 4; r += 1) {
    for (let c = 0; c < 4; c += 1) {
      const id = r * 4 + c;
      if (c + 1 < 4) gridEdges.push(id, id + 1);
      if (r + 1 < 4) gridEdges.push(id, id + 4);
    }
  }
  const grid = CsrGraph.fromEdgeList(16, new Uint32Array(gridEdges));

  const pivots = selectMaxminSpPivots(grid, 6);
  assert(pivots instanceof Uint32Array);
  assert.equal(pivots.length, 6);
  assert.deepEqual(Array.from(pivots), Array.from(selectMaxminSpPivots(grid, 6, 42, 2)));

  const mdsDef = pivotMdsInit(grid);
  assert.equal(mdsDef.len(), 16);
  assert.equal(mdsDef.isAllFinite(), true);
  const mdsCoords = mdsDef.coordsView();
  let sumSq = 0;
  let meanX = 0;
  let meanY = 0;
  for (let i = 0; i < 16; i += 1) {
    meanX += mdsCoords[2 * i];
    meanY += mdsCoords[2 * i + 1];
  }
  meanX /= 16;
  meanY /= 16;
  for (let i = 0; i < 16; i += 1) {
    const dx = mdsCoords[2 * i] - meanX;
    const dy = mdsCoords[2 * i + 1] - meanY;
    sumSq += dx * dx + dy * dy;
  }
  assert(Math.abs(Math.sqrt(sumSq / 32) - 1.0) < 1e-4);

  const mdsUnnorm = pivotMdsInit(grid, 6, 42, 2, null);
  assert.equal(mdsUnnorm.len(), 16);
  assert.equal(mdsUnnorm.isAllFinite(), true);
  mdsDef.free();
  mdsUnnorm.free();

  const korenDef = korenSpectralInit(grid);
  assert.equal(korenDef.len(), 16);
  assert.equal(korenDef.isAllFinite(), true);
  const korenUnnorm = korenSpectralInit(grid, null);
  assert.equal(korenUnnorm.len(), 16);
  korenDef.free();
  korenUnnorm.free();

  const eig = smallestLaplacianEigenpairs(grid, 4);
  assert(eig instanceof EigenResult);
  assert.equal(eig.numNodes, 16);
  assert.equal(eig.numModes, 4);
  assert.equal(eig.usedDenseFallback, true);
  assert(eig.lambdas() instanceof Float64Array);
  assert(eig.vectors() instanceof Float64Array);
  assert(eig.invSqrtDeg() instanceof Float64Array);
  assert.equal(eig.lambdas().length, 4);
  assert.equal(eig.vectors().length, 16 * 4);
  assert.equal(eig.invSqrtDeg().length, 16);

  const stats = spectralMomentStats(grid);
  assert(stats instanceof SpectralMomentStats);
  assert(stats.sigmaLambdaSq > 0);
  assert(Math.abs(stats.meanDegree - 3.0) < 1e-12);
  assert.equal(stats.leafFraction, 0);
  stats.free();

  const invD = eig.invSqrtDeg();
  const statVec = new Float64Array(16);
  for (let i = 0; i < 16; i += 1) {
    statVec[i] = 1.0 / invD[i];
  }
  const diffused = chebyshevHeatSpmv(grid, statVec, 1, 2.0);
  assert(diffused instanceof Float64Array);
  for (let i = 0; i < 16; i += 1) {
    assert(Math.abs(diffused[i] - statVec[i]) < 1e-10);
  }
  eig.free();

  const jlEmb = chebyshevDogJlEmbedding(grid, 8);
  assert(jlEmb instanceof Float64Array);
  assert.equal(jlEmb.length, 16 * 8);

  const twoHop = extract1hop2hopCsr(grid);
  assert(twoHop instanceof TwoHopCsr);
  const h1 = twoHop.hop1();
  const h2 = twoHop.hop2();
  assert(h1 instanceof CsrGraph);
  assert(h2 instanceof CsrGraph);
  assert.equal(h1.numEdges(), 24);
  assert(h2.numEdges() > 0);
  h1.free();
  h2.free();
  twoHop.free();

  const pureRes = buildPure2hopKnn(grid);
  assert(pureRes instanceof TwoHopKnnResult);
  const pureKnn = pureRes.knn();
  assert(pureKnn instanceof KnnGraph);
  assert.deepEqual(Array.from(pureRes.k2Deg()), Array.from(compute2hopNeighborhoodSizes(grid)));
  assert.equal(pureKnn.numNodes, 16);
  assert.equal(pureKnn.nnz, pureRes.nnz);
  assert(pureRes.avgDeg2 >= pureRes.avgDeg1);

  const knnRt = KnnGraph.fromCsr(
    pureKnn.numNodes,
    pureKnn.indptr(),
    pureKnn.indices(),
    pureKnn.distances(),
  );
  assert.equal(knnRt.numNodes, pureKnn.numNodes);
  assert.equal(knnRt.nnz, pureKnn.nnz);
  knnRt.free();
  pureKnn.free();
  pureRes.free();

  const nodePerps = computeNodewise2hopPerplexities(grid);
  assert(nodePerps instanceof Float64Array);
  assert.equal(nodePerps.length, 16);
  const medPerp = computeMedian2hopPerplexity(grid);
  assert(Number.isFinite(medPerp) && medPerp >= 1.0);

  const hybridRes = buildHybridAdaptiveKnn(grid, jlEmb, 8, nodePerps);
  assert(hybridRes instanceof HybridKnnResult);
  const hybridKnn = hybridRes.knn();
  assert(hybridKnn instanceof KnnGraph);
  assert.equal(hybridRes.nFilledResidual + hybridRes.nPure2hopRows, 16);

  const bfsKnn = partialBfsKnn(grid, 5, 42);
  assert(bfsKnn instanceof KnnGraph);
  assert.equal(bfsKnn.nnz, 16 * 5);

  const probs = computeKnnProbabilities(hybridKnn, nodePerps);
  assert(probs instanceof SparseProbabilities);
  assert.equal(probs.numNodes, 16);
  const pSum = probs.values().reduce((a, b) => a + b, 0);
  assert(Math.abs(pSum - 1.0) < 1e-10);
  assert(probs.indptr() instanceof Uint32Array);
  assert(probs.indices() instanceof Uint32Array);

  const fuzzy = computeFuzzySimplicialSet(hybridKnn);
  assert(fuzzy instanceof SparseProbabilities);
  assert(fuzzy.values().every((v) => v > 1e-7 && v <= 1.0));

  const ab = findAbParams();
  assert(ab instanceof Float64Array);
  assert.equal(ab.length, 2);
  assert(Math.abs(ab[0] - 1.577) < 0.02);
  assert(Math.abs(ab[1] - 0.895) < 0.02);

  assert(Math.abs(computeHorvitzThompsonGamma(1.0, 1600, 16) - 10.0) < 1e-12);

  const lm = computeRegularizedDogVaradhanLandmarks(grid);
  assert(lm instanceof DogVaradhanLandmarks);
  assert.equal(lm.numNodes, 16);
  assert.equal(lm.numLandmarks, 16);
  assert(lm.landmarks() instanceof Uint32Array);
  assert(lm.table() instanceof Float64Array);
  assert(lm.coarseTable() instanceof Float64Array);
  assert.equal(lm.usedCoarseScaffold, true);
  assert(lm.t1 > 0 && lm.t2 > lm.t1);
  assert(lm.alpha >= 0 && lm.deltaPsd >= 0 && lm.betaEff >= 1e-4 && lm.betaAuto >= 1e-4);
  const alt = lm.altDistances(new Uint32Array([0, 15]), new Uint32Array([15, 0]));
  assert(alt instanceof Float64Array);
  assert(Math.abs(alt[0] - alt[1]) < 1e-12);

  // Adversarial M2 JS inputs must throw Error (never RuntimeError)
  const assertM2Err = (fn, label) => {
    let threw = false;
    try {
      fn();
    } catch (err) {
      assert(!(err instanceof WebAssembly.RuntimeError), `${label} trapped`);
      assert(err instanceof Error, `${label} must throw Error`);
      threw = true;
    }
    assert(threw, `${label} must throw`);
  };
  assertM2Err(() => selectMaxminSpPivots(grid, 4, 1.5), "M2 seed fractional");
  assertM2Err(() => pivotMdsInit(grid, 4, 42, 2, -1), "M2 negative targetStd");
  assertM2Err(() => chebyshevHeatSpmv(grid, new Float64Array([Number.NaN]), 1, 1.0), "M2 NaN vec");
  assertM2Err(() => KnnGraph.fromCsr(2, new Uint32Array([0, 1, 1]), new Uint32Array([1]), new Float64Array([-1.0])), "M2 neg dist");
  assertM2Err(() => lm.altDistances(new Uint32Array([0]), new Uint32Array([99])), "M2 alt OOB");

  lm.free();
  fuzzy.free();
  probs.free();
  bfsKnn.free();
  hybridKnn.free();
  hybridRes.free();
  grid.free();

  // Verify index.d.ts carries clean optional camelCase M2 signatures
  const dtsText = await readFile(new URL("../pkg/index.d.ts", import.meta.url), "utf8");
  assert(dtsText.includes("pivotMdsInit(graph: CsrGraph, numPivots?: number | null, seed?: number | null, minDegree?: number | null, targetStd?: number | null): Layout2d;"));
  assert(dtsText.includes("korenSpectralInit(graph: CsrGraph, targetStd?: number | null, tol?: number | null, maxIters?: number | null, seed?: number | null): Layout2d;"));
  assert(!dtsText.includes("target_std: any"));
}

// 9. Verify M3 clustering, coarsening, ManifoldLayout2d, and cooperative f32 solvers
{
  const expectedM3 = [
    "CoarsenedGraphResult",
    "GuMapSolver",
    "KMeansResult",
    "KamadaKawaiSolver",
    "LouvainResult",
    "ManifoldLayout2d",
    "SgdSolver",
    "SparseBarrierStats",
    "SpatialIntruderRefiner",
    "SpatialRefinerRoundStats",
    "SpatialRefinerStats",
    "StressMajorizationSolver",
    "TsNetSolver",
    "coarsenGraph",
    "kmeansPlusPlus",
    "labelPropagation",
    "louvainCommunities",
    "projectClusteredLayout",
    "spectralClustering",
  ];
  for (const name of expectedM3) {
    assert.equal(typeof eg[name], "function", `missing M3 runtime export: ${name}`);
  }

  const indexJsText = await readFile(new URL("../pkg/index.js", import.meta.url), "utf8");
  for (const name of expectedM3) {
    assert(indexJsText.includes(name), `missing M3 export in pkg/index.js: ${name}`);
  }

  const dtsText = await readFile(new URL("../pkg/index.d.ts", import.meta.url), "utf8");
  assert(dtsText.includes("export interface TsNetOptions {"));
  assert(dtsText.includes("export interface SparseBarrierDogOptions extends SparseBarrierOptions {"));
  assert(dtsText.includes("static pure2Hop(graph: CsrGraph, backendOrOptions?: TsNetBackend | TsNetOptions | string | null"));
  assert(dtsText.includes("static sparseBarrierDog(graph: CsrGraph, max2hopOrOptions?: number | SparseBarrierDogOptions | null"));
  assert(dtsText.includes("static fromGraphDog(graph: CsrGraph, max2hopOrOptions?: number | GuMapDogOptions | null"));
  assert(dtsText.includes("constructor(graph: CsrGraph, roundsOrOptions?: number | SpatialRefinerOptions | null"));
  assert(dtsText.includes("static fromGraph(graph: CsrGraph, epsilonOrOptions?: number | StressMajorizationOptions | null"));
  assert(dtsText.includes("static fromGraph(graph: CsrGraph, epsOrOptions?: number | KamadaKawaiOptions | null"));
  assert(dtsText.includes("export function kmeansPlusPlus(points: Float64Array | number[], numNodes: number, dims: number"));
  assert(dtsText.includes("export function louvainCommunities(graph: CsrGraph, maxPasses?: number | null"));
  assert(!dtsText.includes("_or_options: any"));
  assert(!dtsText.includes("n_epochs_or_options"));
  assert(!dtsText.includes("max_2hop_or_options"));

  // Helper graphs: 12-node 2-clique barbell and 4x4 grid (16 nodes)
  const makeBarbell = (k) => {
    const edges = [];
    for (const off of [0, k]) {
      for (let i = 0; i < k; i++) {
        for (let j = i + 1; j < k; j++) {
          edges.push(off + i, off + j);
        }
      }
    }
    edges.push(k - 1, k);
    return eg.CsrGraph.fromEdgeList(k * 2, new Uint32Array(edges), null, false, true);
  };
  const makeGrid = (rows, cols) => {
    const edges = [];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const u = r * cols + c;
        if (c + 1 < cols) edges.push(u, u + 1);
        if (r + 1 < rows) edges.push(u, u + cols);
      }
    }
    return eg.CsrGraph.fromEdgeList(rows * cols, new Uint32Array(edges), null, false, true);
  };
  const makeCircleCoords = (n, radius = 1.0) => {
    const out = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      const ang = (2 * Math.PI * i) / Math.max(n, 1);
      out[2 * i] = radius * Math.cos(ang);
      out[2 * i + 1] = radius * Math.sin(ang);
    }
    return out;
  };

  // Clustering & coarsening smoke check
  const km = kmeansPlusPlus(
    new Float64Array([
      10, 10, 10.2, 9.8, 9.8, 10.1,
      -10, -10, -9.9, -10.2, -10.1, -9.8,
    ]),
    6,
    2,
    2,
    100,
    1e-4,
    2,
    42,
  );
  assert(km instanceof KMeansResult);
  assert.equal(km.k, 2);
  assert.equal(km.dims, 2);
  assert.equal(km.converged, true);
  const kmLabels = km.labels();
  assert.equal(kmLabels.length, 6);
  assert.equal(kmLabels[0], kmLabels[1]);
  assert.equal(kmLabels[3], kmLabels[4]);
  assert.notEqual(kmLabels[0], kmLabels[3]);
  assert.equal(km.centroids().length, 4);
  km.free();

  const barbell = makeBarbell(6);
  const specLabels = spectralClustering(barbell, 2, 300, 1e-4, 100, 1e-4, 42);
  assert.equal(specLabels.length, 12);
  assert.notEqual(specLabels[0], specLabels[11]);

  const louv = louvainCommunities(barbell, 50, 1.0, 1e-7, 42);
  assert(louv instanceof LouvainResult);
  assert(louv.numCommunities >= 2);
  assert(louv.modularity > 0.2);
  assert(louv.passes >= 1);
  const louvLabels = louv.labels();
  assert.equal(louvLabels.length, 12);
  louv.free();

  const lpaLabels = labelPropagation(barbell, 50, 42);
  assert.equal(lpaLabels.length, 12);

  const coarse = coarsenGraph(barbell, louvLabels);
  assert(coarse instanceof CoarsenedGraphResult);
  const coarseGraph = coarse.graph();
  assert.equal(coarseGraph.numNodes(), coarse.numClusters);
  assert.equal(coarse.nodeToCoarse().length, 12);
  assert.equal(coarse.coarseSizes().reduce((a, b) => a + b, 0), 12);

  const coarseLayout = eg.Layout2d.fromCoords(makeCircleCoords(coarse.numClusters, 2.0));
  const fineProjected = projectClusteredLayout(coarseLayout, coarse.nodeToCoarse(), 0.25, 42);
  assert.equal(fineProjected.len(), 12);
  assert(fineProjected.isAllFinite());
  coarseLayout.free();
  fineProjected.free();
  coarseGraph.free();
  coarse.free();
  barbell.free();

  // Layout solvers + in-place buffer mutation + step == run parity
  const grid = makeGrid(4, 4);

  // TsNetSolver.pure2Hop (also verifying totalFft2dCalls on linear_fft)
  const tsOpts = {
    backend: "linear_fft",
    iterationsStage1: 4,
    iterationsStage2: 4,
    iterationsStage3: 4,
    theta: 0.5,
    seed: 11,
  };
  const tsRunSolver = TsNetSolver.pure2Hop(grid, tsOpts);
  const tsStepSolver = TsNetSolver.pure2Hop(grid, tsOpts);
  const tsRunLayout = eg.Layout2d.fromCoords(makeCircleCoords(16, 1.0));
  const tsStepLayout = eg.Layout2d.fromCoords(makeCircleCoords(16, 1.0));
  const ptrBefore = tsStepLayout.coordsPtr();
  const lenBefore = tsStepLayout.coordsLen();
  assert.equal(tsRunSolver.run(tsRunLayout), "completed");
  while (!tsStepSolver.isFinished) {
    tsStepSolver.step(tsStepLayout);
  }
  assert.equal(tsStepSolver.status, "completed");
  assert.equal(tsStepLayout.coordsPtr(), ptrBefore);
  assert.equal(tsStepLayout.coordsLen(), lenBefore);
  assert.equal(typeof tsRunSolver.totalFft2dCalls, "number");
  assert(tsRunSolver.totalFft2dCalls > 0);
  assert.deepEqual(Array.from(tsStepLayout.coordsView()), Array.from(tsRunLayout.coordsView()));
  tsRunSolver.free();
  tsStepSolver.free();
  tsRunLayout.free();
  tsStepLayout.free();

  // SgdSolver.sparseBarrierDog + Typed SparseBarrierStats + in-place coordsView mutation
  const sgdSolver = SgdSolver.sparseBarrierDog(grid, {
    max2hopPerNode: 16,
    tMax: 8,
    degree: 6,
    numLandmarks: 8,
    seed: 19,
  });
  const sgdStats = sgdSolver.stats();
  assert(sgdStats instanceof SparseBarrierStats);
  assert(sgdStats.nPairs1hop > 0);
  assert.equal(typeof sgdStats.perplexitySource, "string");
  const sgdLayout = eg.Layout2d.fromCoords(makeCircleCoords(16, 1.5));
  const sgdViewBefore = sgdLayout.coordsView();
  const sgdPtrBefore = sgdLayout.coordsPtr();
  assert.equal(sgdSolver.run(sgdLayout), "completed");
  assert.equal(sgdLayout.coordsPtr(), sgdPtrBefore);
  assert(sgdLayout.isAllFinite());
  assert.equal(sgdViewBefore.length, 32);
  sgdStats.free();
  sgdSolver.free();
  sgdLayout.free();

  // ManifoldLayout2d across all 3 manifolds + SgdSolver.runManifold
  for (const kind of ["hyperbolic", "spherical", "torus"]) {
    const ml = ManifoldLayout2d.fromCoords(kind, makeCircleCoords(16, 0.35));
    assert.equal(ml.kind, kind);
    assert.equal(ml.len(), 16);
    assert(ml.isAllFinite());
    const manifoldSgd = SgdSolver.full(grid, { tMax: 4, seed: 23 });
    assert.equal(manifoldSgd.stats(), undefined);
    assert.equal(manifoldSgd.runManifold(ml), "completed");
    assert(ml.isAllFinite());
    assert(ml.geodesicDistance(0, 15) >= 0);
    manifoldSgd.free();
    ml.free();
  }

  // GuMapSolver.fromGraphDog
  const gumap = GuMapSolver.fromGraphDog(grid, {
    max2hopPerNode: 16,
    nEpochs: 8,
    degree: 6,
    numLandmarks: 8,
    seed: 31,
  });
  const gumapLayout = eg.Layout2d.fromCoords(makeCircleCoords(16, 1.0));
  assert.equal(gumap.run(gumapLayout), "completed");
  assert(gumap.numEdges > 0);
  assert(gumap.a > 0 && gumap.b > 0);
  assert(gumapLayout.isAllFinite());
  gumap.free();

  // SpatialIntruderRefiner + Typed SpatialRefinerStats
  const refiner = new SpatialIntruderRefiner(grid, { rounds: 2, tMax: 4, seed: 37 });
  assert.equal(refiner.run(gumapLayout), "completed");
  const refStats = refiner.stats();
  assert(refStats instanceof SpatialRefinerStats);
  assert.equal(refStats.roundsExecuted, 2);
  const rounds = refStats.roundStats();
  assert.equal(rounds.length, 2);
  assert(rounds[0] instanceof SpatialRefinerRoundStats);
  rounds.forEach((r) => r.free());
  refStats.free();
  refiner.free();
  gumapLayout.free();

  // StressMajorizationSolver.fromGraph
  const sm = StressMajorizationSolver.fromGraph(grid, { maxIterations: 12, epsilon: 1e-5 });
  const smLayout = eg.Layout2d.fromCoords(makeCircleCoords(16, 2.0));
  const stressBefore = sm.computeStress(smLayout);
  const smStatus = sm.run(smLayout);
  assert(smStatus === "converged" || smStatus === "completed");
  assert(sm.lastStress <= stressBefore + 1e-6);
  assert(smLayout.isAllFinite());
  sm.free();
  smLayout.free();

  // KamadaKawaiSolver.fromGraph
  const kk = KamadaKawaiSolver.fromGraph(grid, { eps: 1e-3, maxInnerIters: 40, maxOuterIters: 4 });
  const kkLayout = eg.Layout2d.fromCoords(makeCircleCoords(16, 2.0));
  const kkStatus = kk.run(kkLayout);
  assert(kkStatus === "converged" || kkStatus === "completed");
  assert(kkLayout.isAllFinite());
  kk.free();
  kkLayout.free();

  // Adversarial error checks raise recoverable JS Errors (no WASM traps)
  const dirGraph = eg.CsrGraph.fromEdgeList(3, new Uint32Array([0, 1, 1, 2]), null, true, true);
  assert.throws(() => louvainCommunities(dirGraph), Error);
  assert.throws(() => ManifoldLayout2d.zeros("hyperbolic_invalid", 4), Error);
  dirGraph.free();
  grid.free();
}

// 10. Verify M4 force models, post-processing, layering, constraints, and FDEB
{
  const expectedM4 = [
    "FdebSolver",
    "ForceAtlas2Solver",
    "IbUmapSolver",
    "PrismOverlapRemoval",
    "SeparationConstraint",
    "TFdpSolver",
    "YifanHuSolver",
    "assignLayers",
    "cycleEdges",
    "generateLayeredConstraints",
    "project1d",
    "projectClusteredRectangleNoOverlap2d",
    "projectLayoutAxis",
    "projectRectangleNoOverlap2d",
    "removeCycles",
  ];
  for (const name of expectedM4) {
    assert.equal(typeof eg[name], "function", `missing M4 runtime export: ${name}`);
    assert.equal(typeof webPkg[name], "function", `missing M4 webPkg export: ${name}`);
  }

  const dtsText = await readFile(new URL("../pkg/index.d.ts", import.meta.url), "utf8");
  assert(dtsText.includes("export interface TFdpOptions {"));
  assert(dtsText.includes("export interface IbUmapOptions {"));
  assert(dtsText.includes("export interface ForceAtlas2Options {"));
  assert(dtsText.includes("export interface YifanHuOptions {"));
  assert(dtsText.includes("export interface PrismOptions {"));
  assert(dtsText.includes("export interface FdebOptions {"));
  assert(dtsText.includes('"squared"'));
  assert(!dtsText.includes("options: any"));
  assert(!dtsText.includes("constraints: any"));

  const edges = [];
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      const u = r * 4 + c;
      if (c + 1 < 4) edges.push(u, u + 1);
      if (r + 1 < 4) edges.push(u, u + 4);
    }
  }
  const grid = CsrGraph.fromEdgeList(16, new Uint32Array(edges));
  const initCoords = new Float32Array(32);
  for (let i = 0; i < 16; i++) {
    const ang = (2 * Math.PI * i) / 16;
    initCoords[2 * i] = Math.cos(ang);
    initCoords[2 * i + 1] = Math.sin(ang);
  }

  // TFdpSolver (new + fromKnn)
  const tfdpRun = new TFdpSolver(grid, { backend: "ibfft", totalIterations: 8, seed: 42 });
  const tfdpStep = new TFdpSolver(grid, { backend: "ibfft", totalIterations: 8, seed: 42 });
  const lTfdpRun = Layout2d.fromCoords(initCoords);
  const lTfdpStep = Layout2d.fromCoords(initCoords);
  assert.equal(tfdpRun.run(lTfdpRun), "completed");
  while (!tfdpStep.isFinished) tfdpStep.step(lTfdpStep);
  assert.deepEqual(Array.from(lTfdpRun.coordsView()), Array.from(lTfdpStep.coordsView()));
  const tfdpKnn = TFdpSolver.fromKnn(
    new Uint32Array([0, 1, 3, 4]),
    new Uint32Array([1, 0, 2, 1]),
    new Float64Array([0.5, 0.5, 1.0, 1.0]),
    { backend: "exact", totalIterations: 4 },
  );
  const l3 = Layout2d.fromCoords(new Float32Array([0, 0, 1, 0, 2, 0]));
  assert.equal(tfdpKnn.run(l3), "completed");
  tfdpRun.free();
  tfdpStep.free();
  lTfdpRun.free();
  lTfdpStep.free();
  tfdpKnn.free();
  l3.free();

  // IbUmapSolver
  const ib = new IbUmapSolver(grid, { backend: "barnes_hut", nEpochs: 6, degree: 6, numLandmarks: 8 });
  const lIb = Layout2d.fromCoords(initCoords);
  assert.equal(ib.run(lIb), "completed");
  assert(ib.a > 0 && ib.b > 0);
  assert.equal(ib.currentRho, 1.0);
  ib.free();
  lIb.free();

  // SgdSolver.snapTFdp, pacmapGraph, sgd2, "squared"
  for (const solver of [
    SgdSolver.snapTFdp(grid, { iterations: 6, seed: 42 }),
    SgdSolver.pacmapGraph(grid, { nNeighbors: 5, phase1Iters: 3, phase2Iters: 3, phase3Iters: 3, seed: 42 }),
    SgdSolver.sgd2(grid, { tMax: 6, batchSize: 16, seed: 42 }),
    SgdSolver.full(grid, { tMax: 6, kind: "squared", seed: 42 }),
  ]) {
    const l = Layout2d.fromCoords(initCoords);
    assert.equal(solver.run(l), "completed");
    assert(l.isAllFinite());
    solver.free();
    l.free();
  }

  // ForceAtlas2Solver & YifanHuSolver
  const fa2 = new ForceAtlas2Solver(grid, { iterations: 8, useBarnesHut: true, seed: 42 });
  const lFa2 = Layout2d.fromCoords(initCoords);
  assert.equal(fa2.run(lFa2), "completed");
  assert(lFa2.isAllFinite());
  fa2.free();
  lFa2.free();

  const yh = new YifanHuSolver(grid, { maxIterationsPerLevel: 8, minCoarsenSize: 6, seed: 42 });
  assert(yh.numLevels >= 1);
  const lYh = Layout2d.fromCoords(initCoords);
  const yhStatus = yh.run(lYh);
  assert(yhStatus === "completed" || yhStatus === "converged");
  assert.equal(yh.currentLevel, 0);
  yh.free();
  lYh.free();

  // PrismOverlapRemoval
  const prism = PrismOverlapRemoval.fromRadii(new Float32Array([0.6, 0.6, 0.6, 0.6]), { maxIterations: 30 });
  const lPrism = Layout2d.fromCoords(new Float32Array([0, 0, 0.1, 0, 0, 0.1, 0.1, 0.1]));
  assert.equal(prism.run(lPrism), "converged");
  assert.equal(prism.lastOverlapCount, 0);
  prism.free();
  lPrism.free();

  // SeparationConstraint, project1d, projectLayoutAxis, generateLayeredConstraints, rectangle projections
  const c0 = new SeparationConstraint(0, 1, 2.0);
  const c1d = new Float32Array([1.0, 0.0]);
  const p1dOut = project1d(c1d, [c0]);
  assert(c1d[1] - c1d[0] >= 2.0 - 1e-5);
  assert.deepEqual(Array.from(p1dOut), Array.from(c1d));
  c0.free();

  const rectL = Layout2d.fromCoords(new Float32Array([0, 0, 0.1, 0.1, 0.2, 0.0, 0.3, 0.1]));
  projectRectangleNoOverlap2d(rectL, new Float32Array([0.5, 0.5, 0.5, 0.5]), new Float32Array([0.5, 0.5, 0.5, 0.5]));
  assert(rectL.isAllFinite());
  projectClusteredRectangleNoOverlap2d(
    rectL,
    new Uint32Array([0, 0, 1, 1]),
    new Float32Array([0.5, 0.5, 0.5, 0.5]),
    new Float32Array([0.5, 0.5, 0.5, 0.5]),
  );
  rectL.free();

  // Layering + generateLayeredConstraints + projectLayoutAxis
  const cycGraph = CsrGraph.fromEdgeList(4, new Uint32Array([0, 1, 1, 2, 2, 0, 2, 3]), null, true, true);
  assert.deepEqual(Array.from(cycleEdges(cycGraph)), [2, 0]);
  const dag = removeCycles(cycGraph);
  assert.equal(cycleEdges(dag).length, 0);
  const layersNs = assignLayers(dag, "networkSimplex");
  assert.equal(layersNs.length, 4);
  const layeredCs = generateLayeredConstraints(dag, layersNs, 2.5);
  const lDag = Layout2d.zeros(4);
  projectLayoutAxis(lDag, 1, layeredCs);
  for (const c of layeredCs) c.free();
  lDag.free();
  dag.free();
  cycGraph.free();

  // FdebSolver (no value-drop on layout)
  const lFdeb = Layout2d.fromCoords(initCoords);
  const fdeb1 = new FdebSolver(grid, lFdeb, { cycles: 2, i0: 6 });
  const fdeb2 = new FdebSolver(grid, lFdeb, { cycles: 2, i0: 6 });
  assert.equal(lFdeb.len(), 16);
  assert.equal(fdeb1.run(), "completed");
  while (!fdeb2.isFinished) fdeb2.step();
  assert.deepEqual(Array.from(fdeb1.edgeOffsets()), Array.from(fdeb2.edgeOffsets()));
  assert.deepEqual(Array.from(fdeb1.points()), Array.from(fdeb2.points()));
  fdeb1.free();
  fdeb2.free();
  lFdeb.free();

  assert.throws(() => assignLayers(grid, "longestPath"), Error);
  grid.free();
}

// 11. Verify legacy exports remain available when compiled with default `full` feature
if (typeof eg.Graph === "function") {
  const legacyGraph = new eg.Graph();
  const u = legacyGraph.addNode(0);
  const v = legacyGraph.addNode(1);
  legacyGraph.addEdge(u, v, 1);
  assert.equal(legacyGraph.nodeCount(), 2);
  assert.equal(legacyGraph.edgeCount(), 1);
  legacyGraph.free();
}

console.log("NODE_ESM_SMOKE_OK");
