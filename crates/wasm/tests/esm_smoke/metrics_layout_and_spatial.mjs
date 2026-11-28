import assert from "node:assert/strict";
import { assertJsError } from "./helpers.mjs";

// Verify egraph-metrics and egraph-spatial APIs
export function runMetricsLayoutAndSpatialSmoke({ eg, webPkg }) {
  const {
    CsrGraph,
    Layout2d,
    Layout3d,
    LayoutDegeneracy,
    ManifoldLayout2d,
    MetricReport,
    MinimumAngleSummary,
    PrismDistortion,
    RtaEvaluation,
    StressEvaluation,
    adjustedPatchNig,
    adjustedPatchNigMultiscale,
    angularResolution,
    aspectRatio,
    crosslessness,
    dggPreservation,
    drngPreservation,
    edgeAngle,
    evaluateMetrics,
    gabrielGraphEdges,
    gabrielGraphProperty,
    idealEdgeLengths,
    landmarkStratifiedRta,
    landmarkStratifiedRtaDetail,
    landmarkStratifiedRtaFromLandmarks,
    layoutDegeneracy,
    minimumAngle,
    minimumAngleSummary,
    neighborhoodPreservation1Hop,
    neighborhoodPreservation2Hop,
    nodeResolution,
    prismDistortion,
    relativeNeighborhoodEdges,
    sampledCrossingNumber,
    sampledCrosslessness,
    sampledNp1,
    sampledNp2,
    sampledScaleNormalizedStress,
    scaleNormalizedStress,
    scaleNormalizedStressDetail,
    sweepCrossingAngle,
    sweepCrossingNumber,
    torusCrossingNumber,
  } = eg;

  assert.equal(typeof webPkg.scaleNormalizedStress, "function");
  assert.equal(typeof webPkg.scaleNormalizedStressDetail, "function");
  assert.equal(typeof webPkg.StressEvaluation, "function");
  assert.equal(typeof webPkg.evaluateMetrics, "function");
  assert.equal(typeof webPkg.MetricReport, "function");
  assert.equal(typeof webPkg.PrismDistortion, "function");

  // 4x4 grid layout (16 nodes, 24 edges, planar crystalline)
  const gridEdges = [];
  const gridCoords = [];
  const gridCoords3d = [];
  for (let r = 0; r < 4; r += 1) {
    for (let c = 0; c < 4; c += 1) {
      const id = r * 4 + c;
      gridCoords.push(c, r);
      gridCoords3d.push(c, r, 0);
      if (c + 1 < 4) gridEdges.push(id, id + 1);
      if (r + 1 < 4) gridEdges.push(id, id + 4);
    }
  }
  const grid = CsrGraph.fromEdgeList(16, new Uint32Array(gridEdges));
  const gridLayout = Layout2d.fromCoords(new Float32Array(gridCoords));
  const gridLayout3d = Layout3d.fromCoords(new Float32Array(gridCoords3d));
  const scaledCoords = new Float32Array(gridCoords.length);
  for (let i = 0; i < gridCoords.length; i += 2) {
    scaledCoords[i] = 10.0 * gridCoords[i] + 5.0;
    scaledCoords[i + 1] = 10.0 * gridCoords[i + 1] - 3.0;
  }
  const gridScaled = Layout2d.fromCoords(scaledCoords);

  const sns1 = scaleNormalizedStress(grid, gridLayout);
  const sns10 = scaleNormalizedStress(grid, gridScaled);
  const sns3d = scaleNormalizedStress(grid, gridLayout3d);
  assert(Number.isFinite(sns1) && sns1 >= 0 && sns1 < 0.05);
  assert(Math.abs(sns1 - sns10) < 1e-5, "SNS must be scale-invariant");
  assert(Math.abs(sns1 - sns3d) < 1e-5, "3D SNS on z=0 grid must match 2D SNS");
  const snsDet = scaleNormalizedStressDetail(grid, gridLayout);
  assert(snsDet instanceof StressEvaluation);
  assert(Math.abs(snsDet.normalizedStress - sns1) < 1e-12);
  assert(Math.abs(snsDet.stress - sns1) < 1e-12);
  assert(Math.abs(snsDet.scale - snsDet.alphaStar) < 1e-12);
  assert(snsDet.pairCount > 0 && Number.isFinite(snsDet.rawStress));
  const snsRec = snsDet.toRecord();
  assert(Math.abs(snsRec.normalizedStress - sns1) < 1e-12);
  assert(Math.abs(snsRec.scale - snsDet.alphaStar) < 1e-12);
  snsDet.free();

  const sampledSnsDefault = sampledScaleNormalizedStress(grid, gridLayout);
  const sampledSns4 = sampledScaleNormalizedStress(grid, gridLayout, { numPivots: 16, seed: 42 });
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
  assert.equal(sampledCrossingNumber(k4, k4Layout, { numSamples: 100, seed: 42 }), 1);
  assert(Math.abs(crosslessness(k4, k4Layout) - (1 - Math.sqrt(1 / 3))) < 1e-12);
  assert(
    Math.abs(
      sampledCrosslessness(k4, k4Layout, { numSamples: 100, seed: 42 }) - (1 - Math.sqrt(1 / 3)),
    ) < 1e-12,
  );
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
  const nigVal = adjustedPatchNig(gridLayout, commLabels, {
    patchRadius: 16,
    canvasSize: 2048,
    padFrac: 0.04,
    seed: 42,
  });
  const nigMs = adjustedPatchNigMultiscale(gridLayout, commLabels, {
    radii: [16, 32],
    canvasSize: 2048,
    padFrac: 0.04,
    seed: 42,
  });
  assert(nigMs instanceof Float64Array);
  assert(Math.abs(nigMs[0] - nigVal) < 1e-12);

  const rtaOpts = { numLandmarks: 4, numTriplets: 10000, minDegree: 2, seed: 42 };
  const rtaVal = landmarkStratifiedRta(grid, gridLayout, rtaOpts);
  const rtaDet = landmarkStratifiedRtaDetail(grid, gridLayout, rtaOpts);
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
    { numTriplets: 10000, seed: 42 },
  );
  assert(rtaFromTbl instanceof RtaEvaluation);
  rtaFromTbl.free();

  // Torus crossing on [0, 1)^2
  const torusGraph = CsrGraph.fromEdgeList(4, new Uint32Array([0, 1, 2, 3]));
  const torusLayout = Layout2d.fromCoords(new Float32Array([0.9, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1]));
  assert(torusCrossingNumber(torusGraph, torusLayout) >= 1);
  torusGraph.free();
  torusLayout.free();

  // Gabriel & Relative Neighborhood edges across ManifoldLayout2d variants (canonical camelCase)
  for (const kind of ["hyperbolicPoincare", "hyperbolicLorentz", "spherical", "torus2d"]) {
    const ml = ManifoldLayout2d.init(kind, 8, { seed: 42, scale: 0.35 });
    const ggMani = gabrielGraphEdges(ml);
    const rngMani = relativeNeighborhoodEdges(ml);
    const ggWeb = webPkg.gabrielGraphEdges(ml);
    const rngWeb = webPkg.relativeNeighborhoodEdges(ml);
    assert(ggMani instanceof Uint32Array && ggMani.length > 0 && ggMani.length % 2 === 0);
    assert(rngMani instanceof Uint32Array && rngMani.length > 0 && rngMani.length % 2 === 0);
    assert.deepEqual(Array.from(ggWeb), Array.from(ggMani));
    assert.deepEqual(Array.from(rngWeb), Array.from(rngMani));
    assert(rngMani.length <= ggMani.length);
    ml.free();
  }

  assert(Math.abs(neighborhoodPreservation1Hop(grid, gridLayout) - 1.0) < 1e-6);
  assert(Math.abs(neighborhoodPreservation2Hop(grid, gridLayout) - 1.0) < 1e-6);
  assert(Math.abs(drngPreservation(grid, gridLayout) - 1.0) < 1e-6);
  assert(Math.abs(dggPreservation(grid, gridLayout) - 1.0) < 1e-6);
  assert.equal(gabrielGraphProperty(grid, gridLayout), 0);
  assert(nodeResolution(gridLayout) > 0);
  assert(Math.abs(aspectRatio(gridLayout) - 1.0) < 1e-6);
  assert(angularResolution(grid, gridLayout) > 0);
  assert(Math.abs(idealEdgeLengths(grid, gridLayout)) < 1e-6);

  assert(Math.abs(edgeAngle(1, 0, 0, 1) - Math.PI / 2) < 1e-12);
  assert.equal(edgeAngle(0, 0, 1, 0), undefined);
  assert.equal(edgeAngle(Number.NaN, 0, 1, 0), undefined);

  const distortion = prismDistortion(gridLayout, gridScaled);
  assert(distortion instanceof PrismDistortion);
  assert(distortion.sigmaDist < 1e-5);
  assert(distortion.sigmaDisp < 1e-5);
  distortion.free();

  // Selective metric evaluation across Layout2d & Layout3d
  const report = evaluateMetrics(
    grid,
    gridLayout,
    ["sns", "np2", "sampledNp1", "sampledNp2", "aspectRatio", "drng"],
    { numPivots: 8, numSamples: 64, seed: 42 },
  );
  assert(report instanceof MetricReport);
  assert(Math.abs(report.scaleNormalizedStress - sns1) < 1e-12);
  assert(Math.abs(report.neighborhoodPreservation2Hop - 1.0) < 1e-6);
  assert.equal(report.sampledNp1, sampledNp1(grid, gridLayout, { numSamples: 64, seed: 42 }));
  assert.equal(report.sampledNp2, sampledNp2(grid, gridLayout, { numSamples: 64, seed: 42 }));
  assert(Math.abs(report.aspectRatio - 1.0) < 1e-6);
  assert(Math.abs(report.relativeNeighborhoodPreservation - 1.0) < 1e-6);
  assert.equal(report.crossingNumber, undefined);
  assert.equal(report.idealEdgeLengths, undefined);

  const report3d = evaluateMetrics(grid, gridLayout3d, ["sns", "np2"]);
  assert(Math.abs(report3d.scaleNormalizedStress - sns3d) < 1e-12);
  assert(Math.abs(report3d.neighborhoodPreservation2Hop - 1.0) < 1e-6);
  report3d.free();

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

  // Delaunay triangulation (always-on)
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

  // Adversarial inputs must throw Error with .code (never WebAssembly.RuntimeError)
  const shortLayout = Layout2d.zeros(3);
  const nanLayout = Layout2d.fromCoords(new Float32Array([Number.NaN, 0, ...gridCoords.slice(2)]));
  assertJsError(() => scaleNormalizedStress(grid, shortLayout), "sns dimension mismatch", "DimensionMismatch");
  assertJsError(() => sweepCrossingNumber(grid, nanLayout), "crossings non-finite", "NonFiniteValue");
  assertJsError(() => neighborhoodPreservation1Hop(grid, shortLayout), "np1 dimension mismatch", "DimensionMismatch");
  assertJsError(() => prismDistortion(gridLayout, shortLayout), "prism dimension mismatch", "DimensionMismatch");
  assertJsError(() => sampledScaleNormalizedStress(grid, gridLayout, { numPivots: 0, seed: 42 }), "numPivots=0", "InvalidInput");
  assertJsError(() => sampledScaleNormalizedStress(grid, gridLayout, { numPivots: 4, seed: -1 }), "seed=-1", "InvalidInput");
  assertJsError(() => sampledScaleNormalizedStress(grid, gridLayout, { numPivots: 4, seed: 1.5 }), "seed=1.5", "InvalidInput");
  assertJsError(
    () => sampledScaleNormalizedStress(grid, gridLayout, { numPivots: 4, seed: 9007199254740992 }),
    "seed > MAX_SAFE_INTEGER",
    "InvalidInput",
  );
  assertJsError(() => sampledCrossingNumber(grid, gridLayout, { numSamples: 0, seed: 42 }), "numSamples=0", "InvalidInput");
  assertJsError(() => evaluateMetrics(grid, gridLayout, "sns"), "non-array targets", "InvalidInput");
  assertJsError(() => evaluateMetrics(grid, gridLayout, [123]), "non-string target element", "InvalidInput");
  assertJsError(() => evaluateMetrics(grid, gridLayout, ["bogusMetric"]), "unknown target name", "InvalidInput");
  assertJsError(() => eg.delaunayTriangulation(nanLayout), "delaunay non-finite", "NonFiniteValue");
  shortLayout.free();
  nanLayout.free();
  grid.free();
  gridLayout.free();
  gridLayout3d.free();
  gridScaled.free();
}
