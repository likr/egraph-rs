import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import init, * as eg from "../pkg/egraph_wasm.node.js";
import * as webPkg from "../pkg/index.js";

await init();
eg.setPanicHook();

function assertErrorWithCode(fn, label, expectedCode = null) {
  let caught = false;
  try {
    fn();
  } catch (err) {
    assert(!(err instanceof WebAssembly.RuntimeError), `${label} trapped with RuntimeError`);
    assert(err instanceof Error, `${label} must throw Error`);
    assert.equal(typeof err.code, "string", `${label} must expose string .code`);
    if (expectedCode !== null) {
      assert.equal(err.code, expectedCode, `${label} expected .code=${expectedCode}, got ${err.code}`);
    }
    caught = true;
  }
  assert(caught, `${label} must throw`);
}

function makeGridGraph(rows, cols) {
  const edges = [];
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const u = r * cols + c;
      if (c + 1 < cols) edges.push(u, u + 1);
      if (r + 1 < rows) edges.push(u, u + cols);
    }
  }
  return eg.CsrGraph.fromEdgeList(rows * cols, new Uint32Array(edges), null, {
    directed: false,
    sanitize: true,
  });
}

function makeCircleCoords(n, radius = 1.0) {
  const out = new Float32Array(n * 2);
  for (let i = 0; i < n; i += 1) {
    const angle = (2.0 * Math.PI * i) / Math.max(n, 1);
    out[2 * i] = radius * Math.cos(angle);
    out[2 * i + 1] = radius * Math.sin(angle);
  }
  return out;
}

// 1. Verify EgraphErrorCode in generated index.d.ts includes BackendUnavailable and DeviceLimitExceeded
{
  const dts = await readFile(new URL("../pkg/index.d.ts", import.meta.url), "utf8");
  assert(dts.includes('"BackendUnavailable"'), "missing BackendUnavailable in index.d.ts");
  assert(dts.includes('"DeviceLimitExceeded"'), "missing DeviceLimitExceeded in index.d.ts");
  assert(dts.includes("EgraphErrorCode"), "missing EgraphErrorCode in index.d.ts");
}

// 2. Verify IbUmapSolver across ibFft, barnesHut, exact and all constructors
{
  const grid = makeGridGraph(4, 4);
  const initCoords = makeCircleCoords(16, 1.2);
  const knnRes = webPkg.buildPure2hopKnn(grid);
  const pureKnn = knnRes.knn();
  const probs = webPkg.computeFuzzySimplicialSet(pureKnn, 64, 1e-5);
  const lmRes = webPkg.computeRegularizedDogVaradhanLandmarks(grid, { numLandmarks: 6, seed: 42 });

  for (const backend of ["ibFft", "barnesHut", "exact"]) {
    const opts = { backend, iterations: 8, numLandmarks: 6, seed: 42 };
    const sRun = new eg.IbUmapSolver(grid, opts);
    const sStep = new eg.IbUmapSolver(grid, opts);
    assert.equal(sRun.target, "layout2d");
    assert.equal(sRun.backend, backend);
    assert.equal(sRun.numNodes, 16);
    assert(sRun.numEdges > 0);
    assert(sRun.a > 0 && sRun.b > 0);
    assert.equal(sRun.ab.length, 2);

    const lRun = eg.Layout2d.fromCoords(initCoords);
    const lStep = eg.Layout2d.fromCoords(initCoords);
    const ptrBefore = lStep.coordsPtr();
    assert.equal(sRun.run(lRun), "completed");
    sStep.step(lStep);
    sStep.runSteps(lStep, 3);
    sStep.run(lStep);
    assert.equal(lStep.coordsPtr(), ptrBefore);
    assert.deepEqual(Array.from(lRun.coordsView()), Array.from(lStep.coordsView()));
    assert.equal(lRun.isAllFinite(), true);
    if (backend === "ibFft") {
      assert(sRun.totalFft2dCalls > 0);
    }

    sRun.free();
    sStep.free();
    lRun.free();
    lStep.free();
  }

  for (const [label, mk] of [
    ["fromLandmarks", () => eg.IbUmapSolver.fromLandmarks(grid, lmRes, 32, { backend: "exact", iterations: 6, seed: 42 })],
    ["fromKnn", () => eg.IbUmapSolver.fromKnn(pureKnn, { backend: "exact", iterations: 6, seed: 42 })],
    ["fromFuzzySet", () => eg.IbUmapSolver.fromFuzzySet(probs, { backend: "exact", iterations: 6, seed: 42 })],
    [
      "fromFuzzySetWithDegrees",
      () => eg.IbUmapSolver.fromFuzzySet(probs, new Uint32Array(16).fill(4), { backend: "exact", iterations: 6, seed: 42 }),
    ],
  ]) {
    const solver = mk();
    const layout = eg.Layout2d.fromCoords(initCoords);
    assert.equal(solver.run(layout), "completed", label);
    assert.equal(layout.isAllFinite(), true, label);
    solver.free();
    layout.free();
  }

  lmRes.free();
  probs.free();
  pureKnn.free();
  knnRes.free();
  grid.free();
}

// 3. Verify target/backend/config/layout validation and fail-fast error codes
{
  const grid = makeGridGraph(3, 3);
  const knnRes = webPkg.buildPure2hopKnn(grid);
  const pureKnn = knnRes.knn();
  const probs = webPkg.computeFuzzySimplicialSet(pureKnn, 64, 1e-5);
  const lmRes = webPkg.computeRegularizedDogVaradhanLandmarks(grid, { numLandmarks: 4, seed: 42 });

  for (const badTarget of ["layout3d", "layoutNd", "manifold2d"]) {
    assertErrorWithCode(
      () => new eg.IbUmapSolver(grid, { target: badTarget }),
      `IbUmapSolver target=${badTarget}`,
      "UnsupportedOperation",
    );
    assertErrorWithCode(
      () => eg.IbUmapSolver.fromKnn(pureKnn, { target: badTarget }),
      `IbUmapSolver.fromKnn target=${badTarget}`,
      "UnsupportedOperation",
    );
    assertErrorWithCode(
      () => eg.IbUmapSolver.fromFuzzySet(probs, { target: badTarget }),
      `IbUmapSolver.fromFuzzySet target=${badTarget}`,
      "UnsupportedOperation",
    );
    assertErrorWithCode(
      () => eg.IbUmapSolver.fromLandmarks(grid, lmRes, { target: badTarget }),
      `IbUmapSolver.fromLandmarks target=${badTarget}`,
      "UnsupportedOperation",
    );
  }

  assertErrorWithCode(
    () => new eg.IbUmapSolver(grid, { target: "unknown" }),
    "IbUmapSolver unknown target",
    "InvalidInput",
  );
  assertErrorWithCode(
    () => new eg.IbUmapSolver(grid, { backend: "unknownBackend" }),
    "IbUmapSolver unknown backend",
    "InvalidInput",
  );
  assertErrorWithCode(
    () => new eg.IbUmapSolver(grid, { backend: "barnesHut", theta: -0.1 }),
    "IbUmapSolver negative theta",
    "InvalidInput",
  );
  assertErrorWithCode(
    () => new eg.IbUmapSolver(grid, { backend: "barnesHut", theta: Number.NaN }),
    "IbUmapSolver NaN theta",
    "InvalidInput",
  );
  assertErrorWithCode(
    () => new eg.IbUmapSolver(grid, { cK: Number.NaN }),
    "IbUmapSolver NaN cK",
    "NonFiniteValue",
  );
  assertErrorWithCode(
    () => new eg.IbUmapSolver(grid, { cK: -1 }),
    "IbUmapSolver negative cK",
    "InvalidInput",
  );
  assertErrorWithCode(
    () => new eg.IbUmapSolver(grid, { learningRate: 0 }),
    "IbUmapSolver zero learningRate",
    "InvalidInput",
  );
  assertErrorWithCode(
    () => new eg.IbUmapSolver(grid, { seed: 1.5 }),
    "IbUmapSolver non-integer seed",
    "InvalidInput",
  );
  assertErrorWithCode(
    () => new eg.IbUmapSolver(grid, { backend: "ibFft", intervals: 0 }),
    "IbUmapSolver zero intervals",
    "InvalidInput",
  );
  assertErrorWithCode(
    () => eg.IbUmapSolver.fromFuzzySet(probs, [4, 4], { backend: "exact" }),
    "IbUmapSolver.fromFuzzySet mismatched degrees",
    "DimensionMismatch",
  );
  assertErrorWithCode(
    () => eg.CsrGraph.fromEdgeList(2, new Uint32Array([0, 1]), new Float32Array([1.0]), { weightDtype: "float64" }),
    "CsrGraph float64 weightDtype",
    "InvalidInput",
  );

  const solver = new eg.IbUmapSolver(grid, { backend: "exact", iterations: 4 });
  const badLen = eg.Layout2d.zeros(4);
  assertErrorWithCode(() => solver.runSteps(badLen, 0), "IbUmapSolver length mismatch", "DimensionMismatch");
  badLen.free();

  for (const badCoord of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
    const arr = new Float32Array(18);
    arr[0] = badCoord;
    assertErrorWithCode(() => eg.Layout2d.fromCoords(arr), `Layout2d.fromCoords(${badCoord})`, "NonFiniteValue");

    const nfLayout = eg.Layout2d.zeros(9);
    nfLayout.coordsView()[0] = badCoord;
    assertErrorWithCode(() => solver.step(nfLayout), `IbUmapSolver.step(${badCoord})`, "NonFiniteValue");
    assertErrorWithCode(() => solver.runSteps(nfLayout, 0), `IbUmapSolver.runSteps(${badCoord})`, "NonFiniteValue");
    nfLayout.free();
  }

  solver.free();
  lmRes.free();
  probs.free();
  pureKnn.free();
  knnRes.free();
  grid.free();
}

// 4. Verify boundary empty/singleton/zero-iter/zero-jitter and metrics workflow
{
  for (const n of [0, 1]) {
    const emptyGraph = eg.CsrGraph.fromEdgeList(n, new Uint32Array([]), null, { directed: false, sanitize: false });
    for (const backend of ["ibFft", "barnesHut", "exact"]) {
      const s = new eg.IbUmapSolver(emptyGraph, { backend, iterations: 4 });
      const l = eg.Layout2d.zeros(n);
      const status = s.run(l);
      assert(status === "converged" || status === "completed");
      assert.equal(l.isAllFinite(), true);
      s.free();
      l.free();
    }
    emptyGraph.free();
  }

  const grid = makeGridGraph(4, 4);
  const sZeroIter = new eg.IbUmapSolver(grid, { backend: "exact", iterations: 0 });
  assert.equal(sZeroIter.isFinished, true);
  const lZeroIter = eg.Layout2d.init(16, 17, 1.0);
  const snapZeroIter = Array.from(lZeroIter.coordsView());
  assert.equal(sZeroIter.step(lZeroIter), "completed");
  assert.deepEqual(Array.from(lZeroIter.coordsView()), snapZeroIter);
  lZeroIter.free();
  sZeroIter.free();

  const lZero = eg.Layout2d.zeros(16);
  const sGrid = new eg.IbUmapSolver(grid, { backend: "ibFft", iterations: 15, numLandmarks: 8, seed: 42 });
  assert.equal(sGrid.run(lZero), "completed");
  assert.equal(lZero.isAllFinite(), true);
  const report = webPkg.evaluateMetrics(grid, lZero, ["neighborhoodPreservation2Hop", "scaleNormalizedStress"]);
  assert(report.neighborhoodPreservation2Hop > 0.25);
  assert(Number.isFinite(report.scaleNormalizedStress));
  report.free();
  sGrid.free();
  lZero.free();
  grid.free();
}

console.log("WASM_SOLVER_SEAMS_E2E_OK");
