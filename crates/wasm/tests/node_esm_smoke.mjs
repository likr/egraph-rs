import assert from "node:assert/strict";
import init, * as eg from "../pkg/egraph_wasm.node.js";
import * as webPkg from "../pkg/index.js";

const {
  CsrGraph,
  Layout2d,
  MetricReport,
  PrismDistortion,
  angularResolutionV6,
  aspectRatioV6,
  dggPreservation,
  drngPreservation,
  edgeAngle,
  evaluateMetrics,
  gabrielGraphPropertyV6,
  idealEdgeLengthsV6,
  neighborhoodPreservation1Hop,
  neighborhoodPreservation2HopV6,
  nodeResolutionV6,
  prismDistortion,
  sampledCrossingNumber,
  sampledScaleNormalizedStress,
  scaleNormalizedStress,
  setPanicHook,
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

// 8. Verify legacy exports remain available when compiled with default `full` feature
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
