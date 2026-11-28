import assert from "node:assert/strict";
import { assertJsError, readPkgText } from "./helpers.mjs";

// Verify force models, post-processing, layering, constraints, and FDEB
export async function runForceAndConstraintsSmoke({ eg, webPkg }) {
  const {
    CsrGraph,
    EdgePolylines,
    FdebSolver,
    ForceAtlas2Solver,
    GuMapSolver,
    IbUmapSolver,
    Layout2d,
    PacmapGraphSolver,
    PrismOverlapRemoval,
    SeparationConstraint,
    Sgd2Solver,
    SgdSolver,
    SnapTFdpSolver,
    Sparse12HopPairs,
    TFdpSolver,
    YifanHuSolver,
    assignLayers,
    computeRegularizedDogVaradhanLandmarks,
    cycleEdges,
    extractSparse1hop2hopPairs,
    generateLayeredConstraints,
    project1d,
    projectClusteredRectangleNoOverlap2d,
    projectLayoutAxis,
    projectRectangleNoOverlap2d,
    removeCycles,
  } = eg;

  const forceConstraintExports = [
    "EdgePolylines",
    "FdebSolver",
    "ForceAtlas2Solver",
    "IbUmapSolver",
    "PacmapGraphSolver",
    "PrismOverlapRemoval",
    "SeparationConstraint",
    "Sgd2Solver",
    "SnapTFdpSolver",
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
  for (const name of forceConstraintExports) {
    assert.equal(typeof eg[name], "function", `missing force/constraint runtime export: ${name}`);
    assert.equal(typeof webPkg[name], "function", `missing force/constraint webPkg export: ${name}`);
  }

  const dtsText = await readPkgText("index.d.ts");
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
  assert(dtsText.includes("withLandmarks("));
  assert(dtsText.includes("withLandmarkDistances("));
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
  const tfdpRun = new TFdpSolver(grid, { backend: "ibFft", totalIterations: 8, seed: 42 });
  const tfdpStep = new TFdpSolver(grid, { backend: "ibFft", totalIterations: 8, seed: 42 });
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
  const ib = new IbUmapSolver(grid, { backend: "barnesHut", iterations: 6, degree: 6, numLandmarks: 8 });
  const lIb = Layout2d.fromCoords(initCoords);
  assert.equal(ib.run(lIb), "completed");
  assert(ib.a > 0 && ib.b > 0);
  assert.equal(ib.backend, "barnesHut");
  ib.free();
  lIb.free();

  // extractSparse1hop2hopPairs, GuMapSolver.fromLandmarks, IbUmapSolver.fromLandmarks
  const sparsePairs = extractSparse1hop2hopPairs(grid, 8, 42);
  assert(sparsePairs instanceof Sparse12HopPairs);
  assert.equal(sparsePairs.num1hop, grid.numEdges());
  assert(sparsePairs.u1 instanceof Uint32Array);
  assert(sparsePairs.v2 instanceof Uint32Array);
  sparsePairs.free();

  const lmSmoke = computeRegularizedDogVaradhanLandmarks(grid, { degree: 6, numLandmarks: 8, seed: 42 });
  const gmLm = GuMapSolver.fromLandmarks(grid, lmSmoke, { max2hopPerNode: 32, iterations: 4, seed: 42 });
  const lGmLm = Layout2d.fromCoords(initCoords);
  assert.equal(gmLm.run(lGmLm), "completed");
  gmLm.free();
  lGmLm.free();

  const gmLmAsym = GuMapSolver.fromLandmarks(grid, lmSmoke, { max2hopPerNode: 32, iterations: 4, symmetricNegative: false, seed: 42 });
  const lGmLmAsym = Layout2d.fromCoords(initCoords);
  assert.equal(gmLmAsym.run(lGmLmAsym), "completed");
  gmLmAsym.free();
  lGmLmAsym.free();
  assertJsError(
    () => GuMapSolver.fromLandmarks(grid, lmSmoke, { max2hopPerNode: 32, symmetricNegative: "nope" }),
    "GuMap invalid boolean",
    "InvalidInput",
  );

  const ibLm = IbUmapSolver.fromLandmarks(grid, lmSmoke, { max2hopPerNode: 32, backend: "exact", iterations: 4, seed: 42 });
  const lIbLm = Layout2d.fromCoords(initCoords);
  assert.equal(ibLm.run(lIbLm), "completed");
  ibLm.free();
  lIbLm.free();
  lmSmoke.free();

  const lmPm = computeRegularizedDogVaradhanLandmarks(grid, { degree: 6, numLandmarks: 8, seed: 42 });
  // SnapTFdpSolver, PacmapGraphSolver, Sgd2Solver, SgdSolver.full("squared")
  for (const solver of [
    new SnapTFdpSolver(grid, { iterations: 6, seed: 42 }),
    new SnapTFdpSolver(grid, { iterations: 6, degreeBeta: 0.5, boundedAttraction: false, seed: 42 }),
    new PacmapGraphSolver(grid, { nNeighbors: 5, phaseIterations: [3, 3, 3], keepAll1hop: true, max1hopPerNode: 3, degreeWeightBeta: 0.25, seed: 42 }),
    PacmapGraphSolver.withLandmarks(grid, lmPm, { nNeighbors: 5, phaseIterations: [3, 3, 3], seed: 42 }),
    PacmapGraphSolver.withLandmarkDistances(grid, lmPm.table(), lmPm.numLandmarks, { nNeighbors: 5, phaseIterations: [3, 3, 3], seed: 42 }),
    new Sgd2Solver(grid, { iterations: 6, batchSize: 16, seed: 42 }),
    SgdSolver.full(grid, { iterations: 6, kind: "squared", seed: 42 }),
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
  assertJsError(
    () => new ForceAtlas2Solver(grid, { maxDisplacement: "bad" }),
    "FA2 invalid maxDisplacement",
    "InvalidInput",
  );
  assertJsError(
    () => new SnapTFdpSolver(grid, { boundedAttraction: "bad" }),
    "SnapTFdp invalid boolean",
    "InvalidInput",
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
  assert.equal(p1dOut, undefined);
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
  const cycGraph = CsrGraph.fromEdgeList(4, new Uint32Array([0, 1, 1, 2, 2, 0, 2, 3]), null, { directed: true, sanitize: true });
  assert.deepEqual(Array.from(cycleEdges(cycGraph)), [2, 0]);
  const dag = removeCycles(cycGraph);
  assert.equal(cycleEdges(dag).length, 0);
  const layersNs = assignLayers(dag, "networkSimplex");
  const layersNsObj = assignLayers(dag, { method: "networkSimplex" });
  assert.deepEqual(Array.from(layersNsObj), Array.from(layersNs));
  assertJsError(() => assignLayers(dag, { method: "longest_path" }), "assignLayers snake_case obj", "InvalidInput");
  assertJsError(() => assignLayers(dag, 123), "assignLayers number", "InvalidInput");
  assertJsError(
    () => new ForceAtlas2Solver(grid, { backend: "exact", useBarnesHut: true }),
    "FA2 conflicting backend+useBarnesHut",
    "InvalidInput",
  );
  assert.equal(layersNs.length, 4);
  const layeredCs = generateLayeredConstraints(dag, layersNs, 2.5);
  const lDag = Layout2d.zeros(4);
  projectLayoutAxis(lDag, 1, layeredCs);
  for (const c of layeredCs) c.free();
  lDag.free();
  dag.free();
  cycGraph.free();

  // FdebSolver + EdgePolylines (straightLineFromGraph + step/run)
  const lFdeb = Layout2d.fromCoords(initCoords);
  const ep1 = EdgePolylines.straightLineFromGraph(grid, lFdeb);
  const ep2 = EdgePolylines.straightLineFromGraph(grid, lFdeb);
  const fdeb1 = new FdebSolver(ep1, { cycles: 2, stepsPerCycle: 6 });
  const fdeb2 = new FdebSolver(ep2, { cycles: 2, stepsPerCycle: 6 });
  assert.equal(lFdeb.len(), 16);
  assert.equal(fdeb1.run(ep1), "completed");
  while (!fdeb2.isFinished) fdeb2.step(ep2);
  assert.deepEqual(Array.from(ep1.edgeOffsetsView()), Array.from(ep2.edgeOffsetsView()));
  assert.deepEqual(Array.from(ep1.pointsView()), Array.from(ep2.pointsView()));
  fdeb1.free();
  fdeb2.free();
  ep1.free();
  ep2.free();
  lFdeb.free();

  assertJsError(() => assignLayers(grid, "longestPath"), "assignLayers undirected", "UnsupportedOperation");
  grid.free();
}
