import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import init, * as eg from "../pkg/egraph_wasm.node.js";
import * as webPkg from "../pkg/index.js";

const {
  AmgPcgBatchResult,
  AmgPcgSolveResult,
  AmgPcgSolver,
  AnisotropicLaplacianResult,
  ClusterAreaReport,
  CsrGraph,
  CsrMatrix,
  DogVaradhanLandmarks,
  EffectiveResistanceResult,
  EigenResult,
  HybridKnnResult,
  KnnGraph,
  Layout2d,
  LayoutDegeneracy,
  MetricReport,
  MinimumAngleSummary,
  PrismDistortion,
  ResistanceCurvatureFlowResult,
  ResistanceCurvatureStep,
  RtaEvaluation,
  Sparse12HopPairs,
  SparseProbabilities,
  SpectralMomentStats,
  TwoHopCsr,
  TwoHopKnnResult,
  adjustedPatchNig,
  adjustedPatchNigMultiscale,
  angularResolutionV6,
  anisotropicLaplacian,
  aspectRatioV6,
  buildHybridAdaptiveKnn,
  buildPure2hopKnn,
  chebyshevDogJlEmbedding,
  chebyshevDogJlEmbeddingMatrix,
  chebyshevHeatSpmv,
  chebyshevHeatSpmvMatrix,
  clusterAreaFaithfulness,
  clusterAreaFaithfulnessDetail,
  compute2hopNeighborhoodSizes,
  computeEffectiveResistances,
  computeFuzzySimplicialSet,
  computeHorvitzThompsonGamma,
  computeKnnProbabilities,
  computeMedian2hopPerplexity,
  computeNodewise2hopPerplexities,
  computeRegularizedDogVaradhanLandmarks,
  computeRegularizedDogVaradhanLandmarksFromMatrix,
  resistanceCurvatureFlow,
  crosslessness,
  dggPreservation,
  drngPreservation,
  edgeAngle,
  evaluateMetrics,
  extract1hop2hopCsr,
  extractSparse1hop2hopPairs,
  findAbParams,
  gabrielGraphPropertyV6,
  idealEdgeLengthsV6,
  korenSpectralInit,
  landmarkStratifiedRta,
  landmarkStratifiedRtaDetail,
  landmarkStratifiedRtaFromLandmarks,
  layoutDegeneracy,
  minimumAngle,
  minimumAngleSummary,
  neighborhoodPreservation1Hop,
  neighborhoodPreservation2HopV6,
  nodeResolutionV6,
  partialBfsKnn,
  pivotMdsInit,
  prismDistortion,
  sampledCrossingNumber,
  sampledCrosslessness,
  sampledNp1,
  sampledNp2,
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
  assert.equal(crosslessness(grid, gridLayout), 1.0);
  assert.equal(sampledCrosslessness(grid, gridLayout), 1.0);

  // K4 square with crossing diagonals
  const k4 = CsrGraph.fromEdgeList(4, new Uint32Array([0, 1, 1, 2, 2, 3, 3, 0, 0, 2, 1, 3]));
  const k4Layout = Layout2d.fromCoords(new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]));
  assert.equal(sweepCrossingNumber(k4, k4Layout), 1);
  assert(Math.abs(sweepCrossingAngle(k4, k4Layout)) < 1e-6);
  assert.equal(sampledCrossingNumber(k4, k4Layout, 100, 42), 1);
  assert(Math.abs(crosslessness(k4, k4Layout) - (1 - Math.sqrt(1 / 3))) < 1e-12);
  assert(Math.abs(sampledCrosslessness(k4, k4Layout, 100, 42) - (1 - Math.sqrt(1 / 3))) < 1e-12);
  k4.free();
  k4Layout.free();

  // Minimum angle, layout degeneracy, Adjusted Patch-NIG, Landmark-Stratified RTA
  const maSum = minimumAngleSummary(grid, gridLayout);
  assert(maSum instanceof MinimumAngleSummary);
  assert.equal(maSum.a2, minimumAngle(grid, gridLayout));
  maSum.free();

  const degInfo = layoutDegeneracy(grid, gridLayout);
  assert(degInfo instanceof LayoutDegeneracy);
  assert.equal(degInfo.numNodes, 16);
  assert.equal(degInfo.duplicateNodesF32, 0);
  assert.equal(degInfo.zeroLengthEdges, 0);
  assert.equal(degInfo.isBboxDegenerate, false);
  degInfo.free();

  const commLabels = new Uint32Array(Array.from({ length: 16 }, (_, i) => (i < 8 ? 0 : 1)));
  const nigVal = adjustedPatchNig(gridLayout, commLabels, 16, 2048, 0.04, 42);
  const nigMs = adjustedPatchNigMultiscale(gridLayout, commLabels, [16, 32], 2048, 0.04, 42);
  assert(nigMs instanceof Float64Array);
  assert(Math.abs(nigMs[0] - nigVal) < 1e-12);

  // G-CAF: separated 2-cluster 50-node grid, TypedArray/number[], invariance, boundaries, evaluateMetrics
  assert.equal(typeof webPkg.ClusterAreaReport, "function");
  assert.equal(typeof webPkg.clusterAreaFaithfulness, "function");
  assert.equal(typeof webPkg.clusterAreaFaithfulnessDetail, "function");
  const sepCoordsArr = [];
  const sepLabelsArr = [];
  for (let c = 0; c < 5; c++) {
    for (let r = 0; r < 5; r++) {
      sepCoordsArr.push(c, r);
      sepLabelsArr.push(0);
    }
  }
  for (let c = 0; c < 5; c++) {
    for (let r = 0; r < 5; r++) {
      sepCoordsArr.push(c + 10, r);
      sepLabelsArr.push(1);
    }
  }
  const sepCoords = new Float32Array(sepCoordsArr);
  const sepLayout = Layout2d.fromCoords(sepCoords);
  const sepLabelsU32 = new Uint32Array(sepLabelsArr);
  const cafValU32 = clusterAreaFaithfulness(sepLayout, sepLabelsU32);
  const cafValArr = clusterAreaFaithfulness(sepLayout, sepLabelsArr);
  assert.equal(cafValU32, cafValArr);
  const cafDet = clusterAreaFaithfulnessDetail(sepLayout, sepLabelsU32);
  assert(cafDet instanceof ClusterAreaReport);
  assert(Math.abs(cafDet.caf - cafValU32) < 1e-12);
  assert(cafValU32 > 0.5 && cafValU32 <= 1.0);
  assert.equal(cafDet.numClusters, 2);
  assert.equal(cafDet.numNodes, 50);
  for (const s of [
    cafDet.caf,
    cafDet.purityAdj,
    cafDet.propVis,
    cafDet.resDensity,
    cafDet.resDeconv,
    cafDet.resCombined,
    cafDet.spatialCoherence,
    cafDet.inFrac,
  ]) {
    assert(Number.isFinite(s) && s >= 0.0 && s <= 1.0);
  }
  assert(cafDet.clusterIds() instanceof Uint32Array);
  assert(cafDet.counts() instanceof Uint32Array);
  assert.deepEqual(Array.from(cafDet.clusterIds()), [0, 1]);
  assert.deepEqual(Array.from(cafDet.counts()), [25, 25]);
  for (const arr of [
    cafDet.p(),
    cafDet.q(),
    cafDet.purity(),
    cafDet.resolution(),
    cafDet.resDensityPerCluster(),
    cafDet.resDeconvPerCluster(),
    cafDet.cafPerCluster(),
    cafDet.coherence(),
  ]) {
    assert(arr instanceof Float64Array);
    assert.equal(arr.length, 2);
  }
  const cafRec = cafDet.toRecord();
  assert.equal(Object.keys(cafRec).length, 22);
  assert.equal(cafDet.effectiveBoundaryR95, 2.0);
  assert.equal(cafDet.gEff, 12);
  assert.equal(cafRec.effectiveBoundaryR95, 2.0);
  assert.equal(cafRec.gEff, 12);
  assert(Math.abs(cafRec.caf - cafValU32) < 1e-12);
  assert.deepEqual(Array.from(cafRec.clusterIds), [0, 1]);
  assert.deepEqual(Array.from(cafRec.counts), [25, 25]);
  cafDet.free();

  // Similarity/reflection/90-deg invariance
  const sepScaled = Layout2d.fromCoords(sepCoords.map((v, i) => v * 4 + (i % 2 === 0 ? 12 : -7)));
  assert(Math.abs(clusterAreaFaithfulness(sepScaled, sepLabelsU32) - cafValU32) < 1e-12);
  sepScaled.free();
  const rot90Coords = new Float32Array(sepCoords.length);
  const reflCoords = new Float32Array(sepCoords.length);
  for (let i = 0; i < 50; i++) {
    rot90Coords[2 * i] = -sepCoords[2 * i + 1];
    rot90Coords[2 * i + 1] = sepCoords[2 * i];
    reflCoords[2 * i] = sepCoords[2 * i];
    reflCoords[2 * i + 1] = -sepCoords[2 * i + 1];
  }
  const sepRot90 = Layout2d.fromCoords(rot90Coords);
  const sepRefl = Layout2d.fromCoords(reflCoords);
  assert(Math.abs(clusterAreaFaithfulness(sepRot90, sepLabelsU32) - cafValU32) < 1e-12);
  assert(Math.abs(clusterAreaFaithfulness(sepRefl, sepLabelsU32) - cafValU32) < 1e-12);
  sepRot90.free();
  sepRefl.free();

  // Imbalanced 24-node fixture: stevensGamma, boundaryR95=Infinity, evaluateMetrics
  const imbCoords = new Float32Array(48);
  const imbLabels = new Uint32Array(24);
  for (let i = 0; i < 24; i++) {
    imbCoords[2 * i] = i < 20 ? -2.0 + 0.5 * (i / 19) : 1.5 + 0.5 * ((i - 20) / 3);
    imbCoords[2 * i + 1] = Math.sin(i * 0.7) * 0.2;
    imbLabels[i] = i < 20 ? 0 : 1;
  }
  const imbLayout = Layout2d.fromCoords(imbCoords);
  const imbEdges = [];
  for (let i = 0; i < 23; i++) imbEdges.push(i, i + 1);
  const imbGraph = CsrGraph.fromEdgeList(24, new Uint32Array(imbEdges));
  const detG05 = clusterAreaFaithfulnessDetail(imbLayout, imbLabels, null, null, null, null, null, 0.5);
  const detG20 = clusterAreaFaithfulnessDetail(imbLayout, imbLabels, null, null, null, null, null, 2.0);
  assert(detG05.p()[0] < detG20.p()[0]);
  detG05.free();
  detG20.free();

  const detInf = clusterAreaFaithfulnessDetail(imbLayout, imbLabels, null, null, null, null, Number.POSITIVE_INFINITY, null);
  assert(Math.abs(detInf.inFrac - 1.0) < 1e-12);
  assert(detInf.caf >= 0.0 && detInf.caf <= 1.0);
  detInf.free();

  const expectedCustomCaf = clusterAreaFaithfulness(imbLayout, imbLabels, 64, 0.9, 8.0, 0.2, 2.5, 0.7);
  // Uint32Array vs number[] parity including u32::MAX
  const l5 = Layout2d.fromCoords(new Float32Array([0, 0, 1, 0, 0, 1, 3, 3, 4, 3]));
  const l5Arr = [100, 5, 100, 4294967295, 5];
  const l5U32 = new Uint32Array(l5Arr);
  assert.equal(clusterAreaFaithfulness(l5, l5U32), clusterAreaFaithfulness(l5, l5Arr));
  l5.free();
  const gcafEval = evaluateMetrics(
    imbGraph,
    imbLayout,
    ["clusterAreaFaithfulness", "sns"],
    8,
    64,
    42,
    null,
    imbLabels,
    null,
    null,
    null,
    null,
    null,
    null,
    64,
    0.9,
    8.0,
    0.2,
    2.5,
    0.7,
  );
  assert(Math.abs(gcafEval.clusterAreaFaithfulness - expectedCustomCaf) < 1e-12);
  const gcafEvalRec = gcafEval.toRecord();
  assert(Math.abs(gcafEvalRec.clusterAreaFaithfulness - expectedCustomCaf) < 1e-12);
  gcafEval.free();

  // G6 nodeWeights & clusterWeights parity + evaluateMetrics forwarding
  const imbNw = new Float64Array(24);
  for (let i = 0; i < 24; i++) imbNw[i] = i < 20 ? 1.5 : 3.0;
  const imbCw = new Float64Array([30.0, 12.0]);
  const cafNw = clusterAreaFaithfulness(imbLayout, imbLabels, null, null, null, null, null, null, imbNw, null);
  const cafCw = clusterAreaFaithfulness(imbLayout, imbLabels, null, null, null, null, null, null, null, imbCw);
  assert(Math.abs(cafNw - cafCw) < 1e-12);
  const evalCw = evaluateMetrics(
    imbGraph,
    imbLayout,
    ["gcaf"],
    null,
    null,
    null,
    null,
    imbLabels,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    imbCw,
  );
  assert(Math.abs(evalCw.clusterAreaFaithfulness - cafCw) < 1e-12);
  evalCw.free();
  assert.throws(() => clusterAreaFaithfulness(imbLayout, imbLabels, null, null, null, null, null, null, imbNw, imbCw), Error);
  assert.throws(() => clusterAreaFaithfulness(imbLayout, imbLabels, null, null, null, null, null, null, null, [0.0, 1.0]), Error);

  // Degenerate boundaries: N=0, N=1, K=1, collapsed
  const l0 = Layout2d.zeros(0);
  const d0 = clusterAreaFaithfulnessDetail(l0, []);
  assert.equal(d0.caf, 0);
  assert.equal(d0.effectiveBoundaryR95, 0);
  assert.equal(d0.gEff, 0);
  assert.equal(d0.numClusters, 0);
  assert.equal(d0.numNodes, 0);
  const d0Rec = d0.toRecord();
  const expectedRecKeys = [
    "caf",
    "purityAdj",
    "propVis",
    "resDensity",
    "resDeconv",
    "resCombined",
    "spatialCoherence",
    "inFrac",
    "effectiveBoundaryR95",
    "gEff",
    "numClusters",
    "numNodes",
    "clusterIds",
    "counts",
    "p",
    "q",
    "purity",
    "resolution",
    "resDensityPerCluster",
    "resDeconvPerCluster",
    "cafPerCluster",
    "coherence",
  ].sort();
  assert.deepEqual(Object.keys(d0Rec).sort(), expectedRecKeys);
  assert.equal(d0Rec.numClusters, 0);
  assert.equal(d0Rec.numNodes, 0);
  for (const arrKey of [
    "clusterIds",
    "counts",
    "p",
    "q",
    "purity",
    "resolution",
    "resDensityPerCluster",
    "resDeconvPerCluster",
    "cafPerCluster",
    "coherence",
  ]) {
    assert.equal(d0Rec[arrKey].length, 0);
  }
  d0.free();
  l0.free();

  // Gap 8: f64 vs f32 Layout2d parity, parse_u32_sequence container coercions & rejections,
  // evaluateMetrics G-CAF target aliases, and lazy invalid G-CAF config with targets=["sns"].
  const sepLayout64 = Layout2d.fromCoords(new Float64Array(sepCoordsArr), "float64");
  const cafVal64 = clusterAreaFaithfulness(sepLayout64, sepLabelsU32);
  assert(Math.abs(cafVal64 - cafValU32) < 1e-12);
  sepLayout64.free();

  const l4_64 = Layout2d.fromCoords(new Float64Array([0, 0, 0.1, 0, 4, 0, 4.1, 0]), "float64");
  const l4_32 = Layout2d.fromCoords(new Float32Array([0, 0, 0.1, 0, 4, 0, 4.1, 0]), "float32");
  const g4 = CsrGraph.fromEdgeList(4, new Uint32Array([0, 1, 1, 2, 2, 3, 3, 0]));
  const baseL4Caf = clusterAreaFaithfulness(l4_64, [0, 0, 1, 1]);
  assert(Math.abs(baseL4Caf - clusterAreaFaithfulness(l4_32, [0, 0, 1, 1])) < 1e-6);

  for (const validSeq of [
    new Uint16Array([0, 0, 1, 1]),
    new Int32Array([0, 0, 1, 1]),
    new Set([0, 1]),
    new Float64Array([0, 0, 1, 1]),
  ]) {
    if (validSeq instanceof Set) {
      const l2 = Layout2d.fromCoords(new Float64Array([0, 0, 4, 0]), "float64");
      assert.equal(clusterAreaFaithfulness(l2, validSeq), 0);
      l2.free();
    } else {
      assert.equal(clusterAreaFaithfulness(l4_64, validSeq), baseL4Caf);
    }
  }
  assert.throws(() => clusterAreaFaithfulness(l4_64, new Int32Array([0, -1, 1, 1])), Error);
  assert.throws(() => clusterAreaFaithfulness(l4_64, new Float64Array([0, 0, 1.5, 1])), Error);
  assert.throws(() => clusterAreaFaithfulness(l4_64, new BigInt64Array([0n, 0n, 1n, 1n])), Error);

  for (const alias of ["g-caf", "G-CAF", "cluster-area-faithfulness", "clusterArea"]) {
    const repAlias = evaluateMetrics(
      g4,
      l4_64,
      [alias],
      null,
      null,
      null,
      null,
      new Uint32Array([0, 0, 1, 1]),
    );
    assert(Math.abs(repAlias.clusterAreaFaithfulness - baseL4Caf) < 1e-12);
    repAlias.free();
  }

  const lazyInvalidGcaf = evaluateMetrics(
    g4,
    l4_64,
    ["sns"],
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    11,
    -1.0,
    null,
    null,
    null,
    Number.NaN,
  );
  assert(Number.isFinite(lazyInvalidGcaf.scaleNormalizedStress));
  assert.equal(lazyInvalidGcaf.clusterAreaFaithfulness, undefined);
  lazyInvalidGcaf.free();
  g4.free();
  l4_64.free();
  l4_32.free();

  const l1 = Layout2d.fromCoords(new Float32Array([1, 2]));
  assert.equal(clusterAreaFaithfulness(l1, [5]), 0);
  l1.free();

  const lCol = Layout2d.fromCoords(new Float32Array([2, -1, 2, -1, 2, -1, 2, -1, 2, -1, 2, -1]));
  assert.equal(clusterAreaFaithfulness(lCol, [0, 0, 0, 1, 1, 1]), 0);
  lCol.free();

  // Lazy gMax=4 allowed when G-CAF not requested
  const lazyOk = evaluateMetrics(
    imbGraph,
    imbLayout,
    ["aspectRatio"],
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    4,
  );
  assert(Number.isFinite(lazyOk.aspectRatio));
  assert.equal(lazyOk.clusterAreaFaithfulness, undefined);
  lazyOk.free();

  sepLayout.free();

  const rtaVal = landmarkStratifiedRta(grid, gridLayout, 4, 10000, 2, 42);
  const rtaDet = landmarkStratifiedRtaDetail(grid, gridLayout, 4, 10000, 2, 42);
  assert(rtaDet instanceof RtaEvaluation);
  assert(Math.abs(rtaDet.rta - rtaVal) < 1e-12);
  assert.equal(rtaDet.isExact, true);
  rtaDet.free();
  const rtaFromTbl = landmarkStratifiedRtaFromLandmarks(
    gridLayout,
    new Uint32Array([0, 15]),
    new Uint32Array([
      0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6,
      6, 5, 4, 3, 5, 4, 3, 2, 4, 3, 2, 1, 3, 2, 1, 0,
    ]),
    10000,
    42,
  );
  assert(rtaFromTbl instanceof RtaEvaluation);
  rtaFromTbl.free();

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
  const report = evaluateMetrics(
    grid,
    gridLayout,
    ["sns", "np2", "sampledNp1", "sampledNp2", "aspectRatio", "drng"],
    8,
    64,
    42,
  );
  assert(report instanceof MetricReport);
  assert(Math.abs(report.scaleNormalizedStress - sns1) < 1e-12);
  assert(Math.abs(report.neighborhoodPreservation2Hop - 1.0) < 1e-6);
  assert.equal(report.sampledNp1, sampledNp1(grid, gridLayout, 64, 42));
  assert.equal(report.sampledNp2, sampledNp2(grid, gridLayout, 64, 42));
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
      "sampledNp1",
      "sampledNp2",
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
  assertJsError(() => evaluateMetrics(imbGraph, imbLayout, ["gcaf"]), "gcaf missing communityLabels");
  assertJsError(
    () =>
      evaluateMetrics(
        imbGraph,
        imbLayout,
        ["clusterAreaFaithfulness"],
        null,
        null,
        null,
        null,
        imbLabels,
        null,
        null,
        null,
        null,
        null,
        null,
        4,
      ),
    "gcaf lazy gMax=4",
  );
  assertJsError(() => clusterAreaFaithfulness(imbLayout, [0, 1]), "gcaf label length mismatch");
  const restLabels = Array.from(imbLabels.slice(1));
  const invalidCommunityLabels = [
    ["neg", [-1, ...restLabels]],
    ["float", [1.5, ...restLabels]],
    ["nan", [Number.NaN, ...restLabels]],
    ["inf", [Number.POSITIVE_INFINITY, ...restLabels]],
    ["overflow", [4294967296, ...restLabels]],
    ["strElem", ["bad", ...restLabels]],
    ["nullElem", [null, ...restLabels]],
    ["undefElem", [undefined, ...restLabels]],
    ["int32Array", new Int32Array([-1, ...restLabels])],
    ["float64Array", new Float64Array([1.5, ...restLabels])],
    ["arrayLikeObj", { length: 24 }],
    ["string", "000000000000000000001111"],
  ];
  for (const [tag, badLabels] of invalidCommunityLabels) {
    assertJsError(() => clusterAreaFaithfulness(imbLayout, badLabels), `gcaf invalid labels (${tag})`);
    assertJsError(() => clusterAreaFaithfulnessDetail(imbLayout, badLabels), `gcafDetail invalid labels (${tag})`);
    assertJsError(
      () => evaluateMetrics(imbGraph, imbLayout, ["gcaf"], null, null, null, null, badLabels),
      `evaluateMetrics invalid labels (${tag})`,
    );
  }
  for (const [tag, nullish] of [["null", null], ["undefined", undefined]]) {
    assertJsError(() => clusterAreaFaithfulness(imbLayout, nullish), `gcaf nullish labels (${tag})`);
    assertJsError(() => clusterAreaFaithfulnessDetail(imbLayout, nullish), `gcafDetail nullish labels (${tag})`);
    assertJsError(
      () => evaluateMetrics(imbGraph, imbLayout, ["gcaf"], null, null, null, null, nullish),
      `evaluateMetrics nullish labels (${tag})`,
    );
  }
  const nanImbCoords = new Float32Array(imbCoords);
  nanImbCoords[0] = Number.NaN;
  const nanImbLayout = Layout2d.fromCoords(nanImbCoords);
  assertJsError(() => clusterAreaFaithfulness(nanImbLayout, imbLabels), "gcaf nan layout");
  nanImbLayout.free();
  for (const badG of [11, -1, 1.5]) {
    assertJsError(() => clusterAreaFaithfulness(imbLayout, imbLabels, badG), `gcaf bad gMax=${badG}`);
  }
  for (const badSigma of [0, -0.5, Number.NaN, Number.POSITIVE_INFINITY, 400]) {
    assertJsError(
      () => clusterAreaFaithfulness(imbLayout, imbLabels, null, badSigma),
      `gcaf bad sigmaCells=${badSigma}`,
    );
  }
  assertJsError(() => clusterAreaFaithfulness(imbLayout, imbLabels, null, null, 0), "gcaf bad kappa=0");
  assertJsError(() => clusterAreaFaithfulness(imbLayout, imbLabels, null, null, null, -1), "gcaf bad tauFrac=-1");
  for (const badR95 of [0, Number.NEGATIVE_INFINITY, Number.NaN]) {
    assertJsError(
      () => clusterAreaFaithfulness(imbLayout, imbLabels, null, null, null, null, badR95),
      `gcaf bad boundaryR95=${badR95}`,
    );
  }
  for (const badGamma of [0, Number.POSITIVE_INFINITY]) {
    assertJsError(
      () => clusterAreaFaithfulness(imbLayout, imbLabels, null, null, null, null, null, badGamma),
      `gcaf bad stevensGamma=${badGamma}`,
    );
  }
  imbGraph.free();
  imbLayout.free();
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
    "CsrMatrix",
    "AnisotropicLaplacianResult",
    "KnnGraph",
    "Sparse12HopPairs",
    "SparseProbabilities",
    "DogVaradhanLandmarks",
    "EigenResult",
    "SpectralMomentStats",
    "TwoHopCsr",
    "TwoHopKnnResult",
    "HybridKnnResult",
    "anisotropicLaplacian",
    "selectMaxminSpPivots",
    "pivotMdsInit",
    "korenSpectralInit",
    "smallestLaplacianEigenpairs",
    "spectralMomentStats",
    "chebyshevHeatSpmv",
    "chebyshevHeatSpmvMatrix",
    "chebyshevDogJlEmbedding",
    "chebyshevDogJlEmbeddingMatrix",
    "extract1hop2hopCsr",
    "extractSparse1hop2hopPairs",
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
    "computeRegularizedDogVaradhanLandmarksFromMatrix",
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
  assert(dtsText.includes("jacobiPrecondition?: boolean | null;"));
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
  assert(!dtsText.includes("w2_or_options"));
  assert(dtsText.includes("export interface TsNetRowNormalizedOptions extends TsNetOptions {"));
  assert(dtsText.includes("export interface SgdScheduleOptions {"));
  assert(dtsText.includes("export interface GuMapLandmarksOptions extends GuMapOptions {"));
  assert(dtsText.includes("wAnchor2?: number | null;"));
  assert(dtsText.includes("static fromGraphRowNormalized("));
  assert(dtsText.includes("static fromLandmarks("));
  assert(dtsText.includes('"sampledNp1"'));

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
  const tsRnSolver = TsNetSolver.fromGraphRowNormalized(grid, {
    w2: 0.5,
    max2hopPerNode: 8,
    backend: "fft",
    iterationsStage1: 2,
    iterationsStage2: 2,
    iterationsStage3: 2,
    seed: 11,
  });
  const tsRnLayout = eg.Layout2d.fromCoords(makeCircleCoords(16, 1.0));
  assert.equal(tsRnSolver.run(tsRnLayout), "completed");
  assert(tsRnLayout.isAllFinite());
  tsRnSolver.free();
  tsRnLayout.free();

  // TsNetSolver jacobiPrecondition option on star graph + non-bool JsError validation
  const starEdges = [];
  for (let v = 1; v < 20; v++) {
    starEdges.push(0, v);
  }
  const starGraph = eg.CsrGraph.fromEdgeList(20, new Uint32Array(starEdges), null, false, true);
  const starInit = makeCircleCoords(20, 0.5);
  starInit[0] = 1.0;
  starInit[1] = 0.0;
  for (const [label, mk] of [
    ["pure2Hop", (jp) => TsNetSolver.pure2Hop(starGraph, { backend: "exact", iterationsStage1: 4, iterationsStage2: 2, iterationsStage3: 2, jacobiPrecondition: jp, seed: 42 })],
    ["fromGraphRowNormalized", (jp) => TsNetSolver.fromGraphRowNormalized(starGraph, { w2: 0.0, backend: "exact", iterationsStage1: 4, iterationsStage2: 2, iterationsStage3: 2, jacobiPrecondition: jp, seed: 42 })],
  ]) {
    const sOn = mk(true);
    const sOff = mk(false);
    const lOn = eg.Layout2d.fromCoords(starInit);
    const lOff = eg.Layout2d.fromCoords(starInit);
    assert.equal(sOn.run(lOn), "completed", label);
    assert.equal(sOff.run(lOff), "completed", label);
    assert(lOn.isAllFinite(), label);
    assert(lOff.isAllFinite(), label);
    const vOn = Array.from(lOn.coordsView());
    const vOff = Array.from(lOff.coordsView());
    let maxDiff = 0;
    for (let i = 0; i < vOn.length; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(vOn[i] - vOff[i]));
    }
    assert(maxDiff > 1e-4, `${label}: expected jacobiPrecondition true/false to diverge, diff=${maxDiff}`);
    sOn.free();
    sOff.free();
    lOn.free();
    lOff.free();
  }
  assert.throws(
    () => TsNetSolver.pure2Hop(starGraph, { backend: "exact", jacobiPrecondition: 1 }),
    (err) => !(err instanceof WebAssembly.RuntimeError),
  );
  starGraph.free();
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
  const refiner = new SpatialIntruderRefiner(grid, {
    rounds: 2,
    kCap: 8,
    wAnchor2: 0.5,
    kCapDecay: 0.5,
    kCapMin: 2,
    tMax: 4,
    seed: 37,
  });
  assert.equal(refiner.currentKCap, 8);
  assert.equal(refiner.run(gumapLayout), "completed");
  refiner.setKCap(6);
  assert.equal(refiner.currentKCap, 6);
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
  assert(dtsText.includes("keepAll1hop?: boolean | null;"));
  assert(dtsText.includes("max1hopPerNode?: number | null;"));
  assert(dtsText.includes("degreeWeightBeta?: number | null;"));
  assert(dtsText.includes("maxDisplacement?: number | null;"));
  assert(dtsText.includes("degreeBeta?: number | null;"));
  assert(dtsText.includes("boundedAttraction?: boolean | null;"));
  assert(dtsText.includes("symmetricNegative?: boolean | null;"));
  assert(dtsText.includes("minSepRatio?: number | null;"));
  assert(dtsText.includes("wSep?: number | null;"));
  assert(dtsText.includes("pacmapGraphWithLandmarks("));
  assert(dtsText.includes("pacmapGraphWithLandmarkDistances("));
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

  // extractSparse1hop2hopPairs, GuMapSolver.fromLandmarks, IbUmapSolver.fromLandmarks
  const sparsePairs = extractSparse1hop2hopPairs(grid, 8, 42);
  assert(sparsePairs instanceof Sparse12HopPairs);
  assert.equal(sparsePairs.num1hop, grid.numEdges());
  assert(sparsePairs.u1 instanceof Uint32Array);
  assert(sparsePairs.v2 instanceof Uint32Array);
  sparsePairs.free();

  const lmSmoke = computeRegularizedDogVaradhanLandmarks(grid, 6, 8, 42);
  const gmLm = GuMapSolver.fromLandmarks(grid, lmSmoke, 32, { nEpochs: 4, seed: 42 });
  const lGmLm = Layout2d.fromCoords(initCoords);
  assert.equal(gmLm.run(lGmLm), "completed");
  gmLm.free();
  lGmLm.free();

  const gmLmAsym = GuMapSolver.fromLandmarks(grid, lmSmoke, 32, { nEpochs: 4, symmetricNegative: false, seed: 42 });
  const lGmLmAsym = Layout2d.fromCoords(initCoords);
  assert.equal(gmLmAsym.run(lGmLmAsym), "completed");
  gmLmAsym.free();
  lGmLmAsym.free();
  assert.throws(
    () => GuMapSolver.fromLandmarks(grid, lmSmoke, 32, { symmetricNegative: "nope" }),
    Error,
  );

  const ibLm = IbUmapSolver.fromLandmarks(grid, lmSmoke, { max2hopPerNode: 32, backend: "exact", nEpochs: 4, seed: 42 });
  const lIbLm = Layout2d.fromCoords(initCoords);
  assert.equal(ibLm.run(lIbLm), "completed");
  ibLm.free();
  lIbLm.free();
  lmSmoke.free();

  const lmPm = computeRegularizedDogVaradhanLandmarks(grid, 6, 8, 42);
  // SgdSolver.snapTFdp, pacmapGraph (including keepAll1hop/max1hopPerNode/degreeWeightBeta), pacmapGraphWithLandmarks, pacmapGraphWithLandmarkDistances, sgd2, "squared"
  for (const solver of [
    SgdSolver.snapTFdp(grid, { iterations: 6, seed: 42 }),
    SgdSolver.snapTFdp(grid, { iterations: 6, degreeBeta: 0.5, boundedAttraction: false, seed: 42 }),
    SgdSolver.pacmapGraph(grid, { nNeighbors: 5, phase1Iters: 3, phase2Iters: 3, phase3Iters: 3, keepAll1hop: true, max1hopPerNode: 3, degreeWeightBeta: 0.25, seed: 42 }),
    SgdSolver.pacmapGraphWithLandmarks(grid, lmPm, { nNeighbors: 5, phase1Iters: 3, phase2Iters: 3, phase3Iters: 3, seed: 42 }),
    SgdSolver.pacmapGraphWithLandmarkDistances(grid, lmPm.table(), lmPm.numLandmarks, { nNeighbors: 5, phase1Iters: 3, phase2Iters: 3, phase3Iters: 3, seed: 42 }),
    SgdSolver.sgd2(grid, { tMax: 6, batchSize: 16, seed: 42 }),
    SgdSolver.full(grid, { tMax: 6, kind: "squared", seed: 42 }),
  ]) {
    const l = Layout2d.fromCoords(initCoords);
    assert.equal(solver.run(l), "completed");
    assert(l.isAllFinite());
    solver.free();
    l.free();
  }
  lmPm.free();

  // ForceAtlas2Solver & YifanHuSolver
  const fa2 = new ForceAtlas2Solver(grid, { iterations: 8, useBarnesHut: true, seed: 42 });
  const lFa2 = Layout2d.fromCoords(initCoords);
  assert.equal(fa2.run(lFa2), "completed");
  assert(lFa2.isAllFinite());
  fa2.free();
  lFa2.free();

  const fa2Cap = new ForceAtlas2Solver(grid, { iterations: 8, maxDisplacement: 0.5, seed: 42 });
  const lFa2Cap = Layout2d.fromCoords(initCoords);
  assert.equal(fa2Cap.run(lFa2Cap), "completed");
  assert(lFa2Cap.isAllFinite());
  fa2Cap.free();
  lFa2Cap.free();
  assert.throws(
    () => new ForceAtlas2Solver(grid, { maxDisplacement: "bad" }),
    Error,
  );
  assert.throws(
    () => SgdSolver.snapTFdp(grid, { boundedAttraction: "bad" }),
    Error,
  );

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

// 11. Verify legacy exports are absent by default and InfoMap is omitted
assert.equal(typeof eg.InfoMap, "undefined");
if (typeof eg.Graph === "function") {
  const legacyGraph = new eg.Graph();
  const u = legacyGraph.addNode(0);
  const v = legacyGraph.addNode(1);
  legacyGraph.addEdge(u, v, 1);
  assert.equal(legacyGraph.nodeCount(), 2);
  assert.equal(legacyGraph.edgeCount(), 1);
  legacyGraph.free();
}

// 12. Verify M5 parity APIs, LayoutNd.coordsView(), runProgressive, and AbortSignal
{
  const {
    LayoutNd,
    runProgressive,
    parseCsrCsv,
    delaunayToCsr,
    rectangleOverlaps,
    idealEdgeLengthsWithUnit,
    crossingEdges,
    crossingEdgesTorus,
    crossingSummary,
    gabrielGraphEdges,
    relativeNeighborhoodEdges,
    allPairsBfs,
    bfsSingleSource,
    scaleNormalizedStressFromDistances,
    prismDistortionWithEdges,
    classicalMds2d,
    rdMds,
    normalizeLayout2d,
    robustNormalizeLayout2d,
    representativeMds,
    computeResistanceEmbedding,
    computeDiffusionEmbedding,
    buildRerankKnn,
    buildCapped12hopKnn,
    graphRowNormalizedProbabilities,
    graph12hopRowNormalizedProbabilities,
    symmetrizeRowNormalized,
    buildGumap12hopPriorityCsr,
    applyRelativeDistanceFloor,
    poincareDistance,
    poincareMobiusAdd,
    sphericalDistance,
    torusDistance,
    wrapTorusPoint,
    wasmMemory,
  } = eg;

  assert.equal(typeof runProgressive, "function");
  assert.equal(typeof webPkg.runProgressive, "function");
  assert.equal(typeof eg.loadDataset, "undefined");
  assert.equal(typeof webPkg.loadDataset, "undefined");

  // Dataset + CsrGraph M5 methods + LCC
  const karateCsv = await readFile(
    new URL("../../dataset/src/data/karate.csv", import.meta.url),
    "utf8",
  );
  const karate = parseCsrCsv(karateCsv, false, false);
  assert.equal(karate.numNodes(), 34);
  assert(karate.nnz() > 0);
  assert.equal(karate.isWeighted(), false);
  assert(karate.degree(0) >= 16);
  assert.equal(karate.neighbors(0).length, karate.degree(0));
  const sub = karate.inducedSubgraph(new Uint32Array([0, 1, 2, 3]));
  assert.equal(sub.numNodes(), 4);
  sub.free();

  const disc = parseCsrCsv("6,5\n0,1\n2,3\n3,4\n4,5\n5,2\n", false, true);
  const lcc = disc.largestConnectedComponent();
  const lccGraph = lcc.graph();
  assert.equal(lccGraph.numNodes(), 4);
  assert.deepEqual(Array.from(lcc.originalNodeIds()), [2, 3, 4, 5]);
  assert.deepEqual(Array.from(lcc.nodeIds()), [2, 3, 4, 5]);
  lccGraph.free();
  lcc.free();
  disc.free();

  // Layout2d.init + normalizeStd + LayoutNd.coordsView()
  const l2Init = Layout2d.init(10, 77);
  assert.equal(l2Init.numNodes, 10);
  assert(!l2Init.isEmpty());
  assert(l2Init.isAllFinite());
  l2Init.normalizeStd(1.5);
  assert.equal(l2Init.copyCoords().length, 20);
  l2Init.free();

  const lNd = LayoutNd.fromCoords(3, 4, new Float32Array(12).map((_, idx) => idx + 1));
  assert.equal(lNd.numNodes, 3);
  assert.equal(lNd.dim, 4);
  assert.equal(lNd.coordsView().length, 12);
  assert.equal(lNd.coordsView()[5], 6);
  // Force memory.grow to verify LayoutNd.coordsView refreshes automatically across detached buffers
  const prevBuf = lNd.coordsView().buffer;
  wasmMemory().grow(1);
  const refreshedNd = lNd.coordsView();
  assert.notEqual(refreshedNd.buffer, prevBuf);
  assert.equal(refreshedNd[5], 6);
  lNd.free();

  // ManifoldLayout2d constructors, tangentDelta, applyTangentStep & distance helpers
  const mHyp = ManifoldLayout2d.hyperbolicPoincare(2);
  assert.equal(mHyp.manifold, "hyperbolic");
  mHyp.setPoint(0, 0.1, 0.1);
  mHyp.setPoint(1, -0.1, 0.0);
  const delta = mHyp.tangentDelta(0, 1);
  assert.equal(delta.length, 2);
  mHyp.applyTangentStep(0, delta[0] * 0.05, delta[1] * 0.05);
  assert(poincareDistance(0, 0, 0.3, 0.4) > 0.5);
  assert(Math.abs(sphericalDistance(0, 0, 0, Math.PI / 2) - Math.PI / 2) < 1e-9);
  assert(Math.abs(torusDistance(0.05, 0.05, 0.95, 0.05) - 0.1) < 1e-9);
  const wrapped = wrapTorusPoint(1.25, -0.25);
  assert(Math.abs(wrapped[0] - 0.25) < 1e-9 && Math.abs(wrapped[1] - 0.75) < 1e-9);
  const mAdd = poincareMobiusAdd(-0.3, 0.4, 0.3, -0.4);
  assert(Math.hypot(mAdd[0], mAdd[1]) < 1e-12);
  const mRand = ManifoldLayout2d.randomInit("hyperbolic", 4, 0.25, 42);
  mRand.centerPoincareCentroid();
  const mc = mRand.poincareCentroid();
  assert(Math.hypot(mc[0], mc[1]) < 1e-4);
  mRand.free();
  mHyp.free();

  // Spatial + metrics parity
  const xGraph = CsrGraph.fromEdgeList(4, new Uint32Array([0, 3, 1, 2]), null, false, true);
  const xLayout = Layout2d.fromCoords(new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]));
  const cPairs = crossingEdges(xGraph, xLayout);
  assert.equal(cPairs.length, 1);
  cPairs[0].free();
  const cTorus = crossingEdgesTorus(xGraph, xLayout);
  assert(cTorus.length >= 1);
  for (const tc of cTorus) tc.free();
  const cSummary = crossingSummary(xGraph, xLayout);
  assert.equal(cSummary.count, 1);
  cSummary.free();
  xGraph.free();
  xLayout.free();

  const rectLayout = Layout2d.fromCoords(new Float32Array([0, 0, 0.5, 0, 5, 0]));
  assert.deepEqual(
    Array.from(rectangleOverlaps(rectLayout, new Float32Array([0.5, 0.5, 0.5]), new Float32Array([0.5, 0.5, 0.5]))),
    [0, 1],
  );
  rectLayout.free();

  // Linalg + neighborhood M5
  const cmds = classicalMds2d(karate);
  normalizeLayout2d(cmds, 1.0, 42);
  robustNormalizeLayout2d(cmds, 1.0, 0.95, 42);
  assert(cmds.isAllFinite());
  const delCsr = delaunayToCsr(cmds);
  assert.equal(delCsr.numNodes(), 34);
  assert(delCsr.numEdges() >= 33);
  delCsr.free();

  const rd = rdMds(karate, 3, 1e-10, 1e-5, 200, 42);
  assert.equal(rd.numNodes, 34);
  assert.equal(rd.dim, 3);
  rd.free();

  const rep = representativeMds(new Float64Array([0, 2, 1, 1, 2, 0]), 3, 2, 2);
  assert.equal(rep.numNodes, 3);
  rep.free();

  const resEmb = computeResistanceEmbedding(karate, 4);
  const diffEmb = computeDiffusionEmbedding(karate, 4, 0.5);
  const rerank = buildRerankKnn(diffEmb.coords(), 34, diffEmb.dim, 5, 2, 2);
  assert.equal(rerank.numNodes, 34);
  resEmb.free();
  diffEmb.free();

  const rowProbs = graphRowNormalizedProbabilities(karate);
  assert.equal(rowProbs.numNodes, 34);
  rowProbs.free();
  const rowProbs12 = graph12hopRowNormalizedProbabilities(karate, 0.5, 8, 0.35, 42);
  assert.equal(rowProbs12.numNodes, 34);
  rowProbs12.free();
  const cappedKnn = buildCapped12hopKnn(karate, 8, 42);
  assert.equal(cappedKnn.k2Deg().length, 34);
  cappedKnn.free();
  const symProbs = symmetrizeRowNormalized(3, new Uint32Array([0, 1, 2, 3]), new Uint32Array([1, 2, 0]), new Float64Array([1, 1, 1]));
  assert.equal(symProbs.numNodes, 3);
  symProbs.free();

  const prio = buildGumap12hopPriorityCsr(3, new Uint32Array([0]), new Uint32Array([1]), new Float64Array([1]), new Uint32Array([0, 1]), new Uint32Array([2, 2]), new Float64Array([2, 2]), 1.1);
  assert.equal(prio.numNodes, 3);
  prio.free();

  const d1 = new Float64Array([1e-6, 1.0]);
  const d2 = new Float64Array([2.0]);
  const d3 = new Float64Array([4.0]);
  const floorVal = applyRelativeDistanceFloor(d1, d2, d3, 1e-3, 0.1);
  assert(floorVal > 0);

  // Quality metrics M5 helpers on karate
  const gg = gabrielGraphEdges(cmds);
  const rng = relativeNeighborhoodEdges(cmds);
  assert(gg.length > 0 && rng.length > 0);
  const apsp = allPairsBfs(karate, 1.0);
  const sssp0 = bfsSingleSource(karate, 0, 1.0);
  assert.equal(sssp0.length, 34);
  assert(scaleNormalizedStressFromDistances(cmds, apsp) >= 0);
  assert(idealEdgeLengthsWithUnit(karate, cmds, 1.0) >= 0);
  const pd = prismDistortionWithEdges(cmds, cmds, gg);
  assert.equal(pd.sigmaDist, 0);
  pd.free();
  const repM5 = evaluateMetrics(karate, cmds, ["sns", "idealEdgeLengths"], 8, 64, 42, 1.0);
  assert(Number.isFinite(repM5.idealEdgeLengths));
  repM5.free();

  // Solvers M5 parity + runProgressive + AbortSignal
  const tsPartial = TsNetSolver.fromGraphPartialBfs(karate, 8, 6.0, {
    backend: "fft",
    iterationsStage1: 2,
    iterationsStage2: 2,
    iterationsStage3: 2,
    intervals: 1,
    interpolationOrder: 1,
    minGridDim: 16,
    maxGridDim: 64,
    seed: 42,
  });
  const frames = [];
  const lTs = Layout2d.fromCoords(cmds.copyCoords());
  const finalCtx = await runProgressive(
    tsPartial,
    lTs,
    (ctx) => frames.push(ctx.iteration),
    { frameBudgetMs: 10, stepsPerYield: 1 },
  );
  assert.equal(finalCtx.status, "completed");
  assert(frames.length >= 1);
  const fftStats = tsPartial.lastFftStats();
  assert(fftStats && fftStats.totalFft2dCalls > 0);
  fftStats.free();
  tsPartial.free();
  lTs.free();

  // AbortSignal test on runProgressive
  const ac = new AbortController();
  const tsAbort = TsNetSolver.pure2Hop(karate, {
    backend: "exact",
    iterationsStage1: 10,
    iterationsStage2: 10,
    iterationsStage3: 10,
    seed: 42,
  });
  const lAbort = Layout2d.fromCoords(cmds.copyCoords());
  const abortPromise = runProgressive(
    tsAbort,
    lAbort,
    () => ac.abort(),
    { frameBudgetMs: 10, maxStepsPerFrame: 1, signal: ac.signal },
  );
  await assert.rejects(abortPromise, (err) => err && err.name === "AbortError");
  tsAbort.free();
  lAbort.free();

  // TFdpSolver.fromKnn(KnnGraph) + SgdSolver.dogStats() + SgdSolver.fromPairs optional + GuMapSolver
  const tfdpFromKnn = TFdpSolver.fromKnn(rerank, { backend: "exact", totalIterations: 4, seed: 42 });
  const lTfdp = Layout2d.fromCoords(cmds.copyCoords());
  assert.equal(tfdpFromKnn.run(lTfdp), "completed");
  tfdpFromKnn.free();
  lTfdp.free();
  rerank.free();

  const sgdDog = SgdSolver.sparseBarrierDog(karate, { tMax: 4, seed: 42, minDegree: 1, jlDim: 8 });
  const dStats = sgdDog.dogStats();
  assert(dStats && dStats.degree > 0);
  dStats.free();
  sgdDog.free();

  const sgdPairs = SgdSolver.fromPairs(2, [0], [1], [1.0], null, null, null, { tMax: 3 });
  const lPair = Layout2d.fromCoords([0, 0, 2, 0]);
  assert.equal(sgdPairs.run(lPair), "completed");
  sgdPairs.free();
  lPair.free();

  const gmPos = GuMapSolver.fromGraphDog(karate, null, 4, null, null, null, null, null, null, null, null, 42, null, null, null, null, null, null, null, null, null, null, 1, 8);
  assert(gmPos.numEdges > 0);
  gmPos.free();

  // AmgPcgSolver smoke (regularized, CSR, cooperative stepping, batch, zero RHS)
  const amg = AmgPcgSolver.fromGraphRegularizedLaplacian(karate, 5.0, { relTol: 1e-9 });
  assert.equal(amg.numNodes, 34);
  assert(amg.numLevels >= 1 && amg.coarsestSize <= 64);
  const bKarate = Float64Array.from({ length: 34 }, (_, i) => Math.sin(0.3 * i + 0.1));
  const amgRes = amg.solve(bKarate);
  assert(amgRes instanceof AmgPcgSolveResult);
  assert.equal(amgRes.converged, true);
  assert(amgRes.relativeResidual <= 1e-8);
  const amgX = amgRes.x();
  assert.equal(amgX.length, 34);

  const stInit = amg.startSolve(bKarate);
  assert(["initialized", "converged"].includes(stInit));
  if (!amg.isFinished) {
    amg.step();
    amg.run();
  }
  assert.equal(amg.converged, true);
  assert.deepEqual(Array.from(amg.solution()), Array.from(amgX));

  const bBatch = new Float64Array(34 * 2);
  for (let i = 0; i < 34; i += 1) {
    bBatch[2 * i] = bKarate[i];
    bBatch[2 * i + 1] = -2.0 * bKarate[i];
  }
  const bRes = amg.solveBatch(bBatch, 2);
  assert(bRes instanceof AmgPcgBatchResult);
  assert.equal(bRes.allConverged, true);
  assert.deepEqual(Array.from(bRes.convergedPerCol()), [1, 1]);
  assert.equal(bRes.x().length, 68);
  bRes.free();
  amgRes.free();
  amg.free();

  const aniso = anisotropicLaplacian(karate, 0.0, 0.0, 0.0, null, false);
  assert(aniso instanceof AnisotropicLaplacianResult);
  const anisoMat = aniso.matrix();
  assert(anisoMat instanceof CsrMatrix);
  assert.equal(anisoMat.numNodes, 34);
  assert(anisoMat.estimateLambdaMaxLanczos(16, 1.05, 42) > 2.0);
  const heatMat = chebyshevHeatSpmvMatrix(anisoMat, bKarate, 1, 0.25);
  assert.equal(heatMat.length, 34);
  const jlMat = chebyshevDogJlEmbeddingMatrix(anisoMat, 8, null, 42);
  assert.equal(jlMat.length, 34 * 8);
  const lmMat = computeRegularizedDogVaradhanLandmarksFromMatrix(karate, anisoMat, null, null, 8, 2, 8, 42, true, 0.5, 1e-4, 0.0);
  assert.equal(lmMat.usedCoarseScaffold, false);
  const regAniso = anisoMat.scaleAndShift(1.0, 1.0);
  const amgMat = AmgPcgSolver.fromCsrMatrix(regAniso);
  const amgMatRes = amgMat.solve(bKarate);
  assert.equal(amgMatRes.converged, true);
  amgMatRes.free();
  amgMat.free();
  regAniso.free();
  lmMat.free();
  anisoMat.free();
  aniso.free();

  const amgCsr = AmgPcgSolver.fromCsr(3, [0, 2, 5, 7], [0, 1, 0, 1, 2, 1, 2], [2, -1, -1, 2, -1, -1, 2]);
  const csrRes = amgCsr.solve([1, 0, 1]);
  assert.equal(csrRes.converged, true);
  csrRes.free();
  amgCsr.free();

  // Effective resistance + Resistance curvature flow smoke
  const erKarate = computeEffectiveResistances(karate, { jlDim: 64, seed: 42 });
  assert(erKarate instanceof EffectiveResistanceResult);
  assert.equal(erKarate.numNodes, 34);
  assert.equal(erKarate.numComponents, 1);
  assert(erKarate.fosterRelErr < 1e-10);
  assert.equal(erKarate.effectiveResistances().length, erKarate.numEdges);
  erKarate.free();

  const rcfKarate = resistanceCurvatureFlow(karate, { steps: 2, eta: 0.2, seed: 42 });
  assert(rcfKarate instanceof ResistanceCurvatureFlowResult);
  assert.equal(rcfKarate.suggestedGeometry, "hyperbolic");
  assert.equal(rcfKarate.suggestedManifold, "hyperbolic");
  const rcfSteps = rcfKarate.stepHistory();
  assert.equal(rcfSteps.length, 3);
  assert(rcfSteps[0] instanceof ResistanceCurvatureStep);
  for (const s of rcfSteps) s.free();
  const rcfLenG = rcfKarate.lengthGraph();
  const rcfCondG = rcfKarate.conductanceGraph();
  assert.equal(rcfLenG.isWeighted(), true);
  assert.equal(rcfCondG.isWeighted(), true);
  rcfLenG.free();
  rcfCondG.free();
  rcfKarate.free();

  cmds.free();
  karate.free();
}

// 12b. M5.1 Louvain coarsen, pivoted RHGH landmarks, coarse backbone, centroid tether, and PaCMAP backbone
{
  const g16 = CsrGraph.fromEdgeList(
    16,
    new Uint32Array([
      0, 1, 1, 2, 2, 3, 3, 0,
      4, 5, 5, 6, 6, 7, 7, 4,
      8, 9, 9, 10, 10, 11, 11, 8,
      12, 13, 13, 14, 14, 15, 15, 12,
      3, 4, 7, 8, 11, 12,
    ]),
    undefined,
    { directed: false, sanitize: true },
  );
  const hier = webPkg.louvainCoarsen(g16, { maxCoarseNodes: 4, seed: 42 });
  assert(hier instanceof webPkg.CoarsenedHierarchyResult);
  assert(hier.numClusters <= 4);
  const hierCg = hier.coarseGraph();
  assert.equal(hierCg.numNodes(), hier.numClusters);
  hierCg.free();
  hier.free();

  const rhghRes = webPkg.computePivotedRhghLandmarks(g16, { numLandmarks: 8, minDegree: 1, numProbes: 16, seed: 42 });
  assert(rhghRes instanceof webPkg.RhghLandmarksResult);
  assert.equal(rhghRes.allConverged, true);
  const rhghLm = rhghRes.landmarks();
  assert(rhghLm instanceof DogVaradhanLandmarks);
  assert.equal(rhghLm.numNodes, 16);
  rhghLm.free();
  rhghRes.free();

  const bb = webPkg.coarseBackboneInit(g16, {
    maxCoarseNodes: 4,
    edgeLengthMode: "resistanceCurvatureFlow",
    distanceBackend: "rhgh",
    coarseTMax: 20,
    seed: 42,
  });
  assert(bb instanceof webPkg.CoarseBackboneResult);
  assert.equal(bb.numFineNodes, 16);
  const bbFine = bb.fineLayout();
  const bbCoarse = bb.coarseLayout();
  const bbN2c = bb.nodeToCoarse();
  webPkg.alignClusteredCentroids(bbFine, bbCoarse, bbN2c, 0.5);
  assert.equal(bbFine.isAllFinite(), true);
  const bbProj = webPkg.projectClusteredLayoutWithRadii(bbCoarse, bbN2c, bb.radii(), 0.5, 42);
  assert.equal(bbProj.len(), 16);
  bbProj.free();
  bbCoarse.free();
  bbFine.free();
  bb.free();

  const pmBb = SgdSolver.pacmapGraphWithBackbone(g16, {
    nNeighbors: 6,
    phase1Iters: 5,
    phase2Iters: 5,
    phase3Iters: 5,
    numLandmarks: 8,
    maxCoarseNodes: 4,
    coarseTMax: 10,
    seed: 42,
  });
  assert(pmBb instanceof webPkg.PacmapBackboneResult);
  const pmSolver = pmBb.solver();
  const pmLayout = pmBb.fineLayout();
  pmSolver.run(pmLayout);
  assert.equal(pmLayout.isAllFinite(), true);
  pmLayout.free();
  pmSolver.free();
  pmBb.free();
  g16.free();
}

// 12c. M5.2 ScreenedPoissonField2d & ScreenedPoissonRefiner
{
  const spField = new webPkg.ScreenedPoissonField2d({
    kernel: "screenedPoisson",
    gridSize: 64,
    paddingFactor: 2,
    rhoMax: 0.2,
  });
  assert(spField instanceof webPkg.ScreenedPoissonField2d);
  assert.equal(spField.gridSize, 64);
  assert.equal(spField.kernel, "screenedPoisson");
  assert.equal(spField.exactAdjoint, true);
  assert(spField.workspaceMemoryBytes() > 0);

  const lDense = Layout2d.fromCoords(new Float32Array([0.0, 0.0, 0.05, 0.02, -0.04, 0.03, 0.02, -0.05]));
  const spRes = spField.evaluate(lDense);
  assert(spRes instanceof webPkg.ScreenedPoissonResult);
  assert.equal(spRes.fft2dCalls, 6);
  assert.equal(spRes.earlyExitNoExcess, false);
  assert(spRes.energy > 0.0);
  assert.equal(spRes.forceX().length, 4);
  assert.equal(spRes.forceY().length, 4);

  const spRefiner = new webPkg.ScreenedPoissonRefiner(4, {
    gridSize: 64,
    iterations: 3,
    rhoMax: 0.2,
    seed: 42,
  });
  assert(spRefiner instanceof webPkg.ScreenedPoissonRefiner);
  spRefiner.run(lDense);
  assert.equal(spRefiner.isFinished, true);
  const lastSt = spRefiner.lastStats();
  assert(lastSt instanceof webPkg.ScreenedPoissonResult);
  lastSt.free();
  spRefiner.free();
  spRes.free();
  lDense.free();
  spField.free();
}

// 12d. W1 Baselines (Leiden, LinLog, FR, FA2::preventOverlap, Backbone, ClusterRefine, GIB, GTree, FORBID, and W1-6 metrics)
{
  const g8 = CsrGraph.fromEdgeList(
    8,
    new Uint32Array([
      0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3,
      4, 5, 4, 6, 4, 7, 5, 6, 5, 7, 6, 7,
      3, 4,
    ]),
    undefined,
    { directed: false, sanitize: true },
  );
  const labels = new Uint32Array([0, 0, 0, 0, 1, 1, 1, 1]);

  // W1-1: Leiden
  const leidenRes = webPkg.leidenCommunities(g8, { objective: "modularity", resolution: 1.0, seed: 42 });
  assert(leidenRes instanceof webPkg.LeidenResult);
  assert.equal(leidenRes.numCommunities, 2);
  assert.equal(leidenRes.labels().length, 8);
  const leidenRec = leidenRes.toRecord();
  assert.equal(leidenRec.numCommunities, 2);
  assert(leidenRec.labels instanceof Uint32Array);
  leidenRes.free();

  // W1-2: LinLog, FR, FA2 preventOverlap
  const lInit = Layout2d.fromCoords(
    new Float32Array([
      -2, 0, -1.5, 0.5, -2.5, 0.5, -2, 1,
      2, 0, 2.5, 0.5, 1.5, 0.5, 2, 1,
    ]),
  );
  const llSolver = new webPkg.LinLogSolver(g8, { iterations: 8, repulsionModel: "edge_repulsion", seed: 42 });
  assert(llSolver instanceof webPkg.LinLogSolver);
  assert.equal(llSolver.masses().length, 8);
  await webPkg.runProgressive(llSolver, lInit, null, { maxStepsPerFrame: 4 });
  assert.equal(llSolver.isFinished, true);
  assert(Number.isFinite(llSolver.energy(lInit)));
  llSolver.free();

  const frSolver = new webPkg.FruchtermanReingoldSolver(g8, { iterations: 8, backend: "barnes_hut", seed: 42 });
  assert(frSolver instanceof webPkg.FruchtermanReingoldSolver);
  assert(frSolver.optimalDistance > 0);
  frSolver.run(lInit);
  assert.equal(frSolver.currentTemperature, 0);
  frSolver.free();

  const fa2Overlap = new webPkg.ForceAtlas2Solver(g8, {
    iterations: 6,
    preventOverlap: true,
    defaultNodeRadius: 0.4,
    nodeRadii: [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
    seed: 42,
  });
  assert.equal(fa2Overlap.nodeRadii().length, 8);
  assert(fa2Overlap.maxDisplacement > 0);
  fa2Overlap.run(lInit);
  assert.equal(lInit.isAllFinite(), true);
  fa2Overlap.free();

  // W1-3: Backbone
  const bbRes = webPkg.extractQuadrilateralBackbone(g8, { retainRatio: 0.5 });
  assert(bbRes instanceof webPkg.BackboneResult);
  assert(bbRes.umstEdges().length > 0);
  assert(bbRes.backboneEdges().length > 0);
  assert(bbRes.simmelianWeights().length > 0);
  assert(bbRes.edgeWeights().length > 0);
  const bbCsr = bbRes.backboneCsr();
  assert.equal(bbCsr.numNodes(), 8);
  const bbRec = bbRes.toRecord();
  assert(bbRec.backboneEdges instanceof Uint32Array);
  bbCsr.free();
  bbRes.free();

  const bbSolver = new webPkg.BackboneSolver(g8, { layoutMode: "sgd", iterations: 6, seed: 42 });
  assert(bbSolver instanceof webPkg.BackboneSolver);
  const bbInner = bbSolver.backboneResult();
  assert(bbInner instanceof webPkg.BackboneResult);
  bbInner.free();
  bbSolver.run(lInit);
  assert.equal(bbSolver.isFinished, true);
  bbSolver.free();

  // W1-4: ClusterRefine, squarifiedTreemap, GroupInABox
  const lSwapped = Layout2d.fromCoords(
    new Float32Array([
      -5, 0, -4.5, 0.5, -5.5, 0.5, 5.2, 0.2,
      5, 0, 5.5, 0.5, 4.5, 0.5, -5.2, 0.2,
    ]),
  );
  const crSolver = webPkg.ClusterRefineSolver.fromGraph(g8, labels, { method: "kmeans", maxRounds: 5, seed: 42 });
  assert(crSolver instanceof webPkg.ClusterRefineSolver);
  assert.equal(crSolver.numClusters, 2);
  crSolver.run(lSwapped);
  assert(crSolver.lastMovedCount >= 0);
  crSolver.free();

  const movedFn = webPkg.refineClusterLayout(lSwapped, labels, { method: "hac", maxRounds: 3, seed: 42 });
  assert.equal(typeof movedFn, "number");

  const treemapRects = webPkg.squarifiedTreemap([3, 2, 1], 100, 50, 0, 0);
  assert.equal(treemapRects.length, 12);

  const gibSolver = new webPkg.GroupInABoxSolver(g8, labels, {
    boxMode: "treemap",
    intraMode: "fr",
    intraIterations: 6,
    clusterWeights: [2, 1],
    seed: 42,
  });
  assert(gibSolver instanceof webPkg.GroupInABoxSolver);
  const boxes = gibSolver.boxes();
  assert.equal(boxes.length, 2);
  assert.equal(boxes[0].nodeCount, 4);
  await webPkg.runProgressive(gibSolver, lInit, null, { maxStepsPerFrame: 3 });
  assert.equal(gibSolver.isFinished, true);
  gibSolver.free();

  // W1-5: GTree & FORBID
  const lOverlap = Layout2d.fromCoords(
    new Float32Array([
      0.0, 0.0, 0.1, 0.05, -0.08, 0.07, 0.05, -0.06,
      0.12, 0.1, -0.1, -0.08, 0.02, 0.09, -0.04, 0.02,
    ]),
  );
  const hw = new Float32Array(8).fill(0.4);
  const hh = new Float32Array(8).fill(0.3);
  const lGt = Layout2d.fromCoords(new Float32Array(lOverlap.coordsView()));
  const gtSolver = new webPkg.GTreeOverlapRemoval(hw, hh, { maxIterations: 25, padding: 0.02, seed: 42 });
  assert(gtSolver instanceof webPkg.GTreeOverlapRemoval);
  gtSolver.run(lGt);
  assert.equal(gtSolver.lastOverlapCount, 0);
  gtSolver.free();

  const lFb = Layout2d.fromCoords(new Float32Array(lOverlap.coordsView()));
  const fbSolver = webPkg.ForbidOverlapRemoval.fromRadii(hw, { variant: "sparse_forbid", maxScaleIters: 20, sgdIterations: 15, seed: 42 });
  assert(fbSolver instanceof webPkg.ForbidOverlapRemoval);
  fbSolver.run(lFb);
  assert.equal(fbSolver.lastOverlapCount, 0);
  fbSolver.free();

  // W1-6: standalone functions + evaluateMetrics targets + toRecord()
  const lSep = Layout2d.fromCoords(
    new Float32Array([
      -5, 0, -4.5, 0.5, -5.5, 0.5, -5, 1,
      5, 0, 5.5, 0.5, 4.5, 0.5, 5, 1,
    ]),
  );
  const paRes = webPkg.partitionAgreement(labels, labels);
  assert(paRes instanceof webPkg.PartitionAgreementResult);
  assert(Math.abs(paRes.toRecord().ari - 1.0) < 1e-12);
  paRes.free();

  const cqK = webPkg.cqKmeans(lSep, labels, 5, 50, 42);
  assert(Math.abs(cqK.ari - 1.0) < 1e-12);
  cqK.free();

  const cqH = webPkg.cqHac(lSep, labels);
  assert(Math.abs(cqH.homogeneity - 1.0) < 1e-12);
  cqH.free();

  assert(Math.abs(webPkg.distanceConsistency(lSep, labels) - 1.0) < 1e-12);
  assert(webPkg.distributionConsistency(lSep, labels, 16) > 0.9);
  const cdRes = webPkg.classDensityMeasure(lSep, labels, 32, null);
  assert(cdRes instanceof webPkg.ClassDensityResult);
  assert(cdRes.toRecord().cdm > 0.8);
  cdRes.free();

  assert(Math.abs(webPkg.sepmeKnng(lSep, labels, 2) - 1.0) < 1e-12);
  const rngScore = webPkg.sepmeRng(lSep, labels);
  const gabScore = webPkg.sepmeGab(lSep, labels);
  const gongScore = webPkg.sepmeGong(lSep, labels, 0.35);
  assert(rngScore > 0.5 && rngScore <= 1.0);
  assert(gabScore > 0.5 && gabScore <= 1.0);
  assert(gongScore > 0.5 && gongScore <= 1.0);
  assert(Math.abs(webPkg.neighborhoodHit(lSep, labels, 3) - 1.0) < 1e-12);
  assert(webPkg.silhouetteScore(lSep, labels) > 0.8);
  assert(webPkg.calinskiHarabaszScore(lSep, labels) > 50.0);
  assert(webPkg.daviesBouldinScore(lSep, labels) < 0.3);
  assert(webPkg.pairwiseDistanceOverlapError(lSep, labels, 32) < 0.1);
  assert(webPkg.sprawlterNn(lOverlap, hw, hh) > 0.0);
  assert(webPkg.sprawlterNnRadii(lOverlap, hw) > 0.0);

  const norG = webPkg.norMetrics(g8, lOverlap, lGt, hw, hh);
  assert(norG instanceof webPkg.NorMetricsResult);
  assert(Number.isFinite(norG.toRecord().elRsd));
  norG.free();

  const norNoG = webPkg.norMetricsWithoutGraph(lOverlap, lGt, hw, hh);
  assert.equal(norNoG.elRsd, 0.0);
  norNoG.free();

  const w1Rep = evaluateMetrics(
    g8,
    lSep,
    [
      "dsc",
      "dc",
      "cdm",
      "hdm",
      "sepme-knng2",
      "sepme_rng",
      "sepme_gab",
      "sepme_gong",
      "neighborhood_hit",
      "silhouette",
      "calinski_harabasz",
      "davies_bouldin",
      "cq_kmeans_ari",
      "cq_kmeans_ami",
      "cq_hac_hom",
      "cq_hac_ami",
      "pairwise_distance_overlap_error",
      "sprawlter_nn",
    ],
    {
      communityLabels: labels,
      sepmeKnngK: 2,
      neighborhoodHitK: 3,
      dcGridSize: 16,
      cdmGridSize: 32,
      numBins: 32,
      gongGamma: 0.35,
      kmeansNInit: 3,
      kmeansMaxIters: 50,
      sprawlterNodeRadius: 0.5,
      seed: 42,
    },
  );
  const w1Rec = w1Rep.toRecord();
  assert(Math.abs(w1Rec.sepmeKnng - 1.0) < 1e-12);
  assert(typeof w1Rec.sprawlterNn === "number");
  w1Rep.free();

  lSep.free();
  lFb.free();
  lGt.free();
  lOverlap.free();
  lSwapped.free();
  lInit.free();
  g8.free();
}

// 13. Run TypeScript declaration typechecking when `tsc` is installed locally
{
  const dtsPath = fileURLToPath(new URL("../pkg/index.d.ts", import.meta.url));
  const tscBinName = process.platform === "win32" ? "tsc.cmd" : "tsc";
  const tscCandidates = [
    fileURLToPath(new URL(`../node_modules/.bin/${tscBinName}`, import.meta.url)),
    fileURLToPath(new URL(`../../../node_modules/.bin/${tscBinName}`, import.meta.url)),
  ];
  const tscBin = tscCandidates.find((p) => existsSync(p));
  if (tscBin) {
    execFileSync(tscBin, ["--noEmit", dtsPath], { stdio: "inherit" });
  }
}

console.log("NODE_ESM_SMOKE_OK");
