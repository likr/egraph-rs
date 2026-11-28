import assert from "node:assert/strict";
import { assertJsError } from "./helpers.mjs";

export async function runSpatialTreesDatasetsAndNdSmoke({ eg }) {
  const {
    ArenaOctree3d,
    ChebyshevDogKernel,
    ClassicalMds,
    CoarseBackbone,
    DijkstraWorkspace,
    DynamicKdTree,
    EdgePolylines,
    FdebSolver,
    ForceAtlas2Solver,
    FruchtermanReingoldSolver,
    Hde,
    IbUmapSolver,
    KorenSpectral,
    Layout2d,
    Layout3d,
    LayoutNd,
    LinLogSolver,
    ManifoldLayout2d,
    MaxentStressSolver,
    PivotMds,
    RdMds,
    ScreenedPoissonField2d,
    SfcLayout,
    Sgd2Solver,
    SgdSolver,
    StressMajorizationSolver,
    TFdpSolver,
    TsNetSolver,
    TsNetStage,
    TwinDeclutterer,
    YifanHuSolver,
    allPairsDijkstra,
    datasetCatalog,
    datasetInfo,
    declutterTwins2d,
    dijkstraSingleSource,
    evaluateMetrics,
    generateHyperbolicTree,
    generateLayeredDag,
    generateSbmGraph,
    generateSphericalGrid,
    generateSphericalIcosphere,
    generateTorusGrid,
    generateWeightedGraph,
    landmarkStratifiedRtaFromLandmarks,
    listDatasets,
    runProgressive,
    torusEdgeSegments,
    torusLayoutEdgeSegments,
    torusSpectralPhaseInit,
    transitiveReduction,
  } = eg;

  // 1. ArenaOctree3d & DynamicKdTree
  const nd = LayoutNd.fromCoords(4, 3, new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1]));
  const oct = new ArenaOctree3d();
  oct.build(nd);
  assert.equal(oct.pointCount, 4);
  assert.equal(oct.computeCoulombRepulsion(0.5, 1e-6).length, 12);
  oct.buildWeighted(nd, new Float64Array([1, 2, 1, 2]));
  assert.equal(oct.computeMaxentRepulsion(0.5, 1.0, 1e-6).length, 12);
  oct.free();

  const kd = DynamicKdTree.fromLayoutNd(nd);
  assert.equal(kd.dims, 3);
  assert.equal(kd.kNearest(new Float32Array([0, 0, 0]), 2).length, 2);
  assert(kd.nearestExcludingSelf(0) !== undefined);
  kd.free();

  // 2. Datasets catalog & 7 synthetic generators
  const names = listDatasets();
  assert(names.length > 0);
  const catalog = datasetCatalog();
  assert.equal(catalog.length, names.length);
  const info = datasetInfo(names[0]);
  assert(info && info.numNodes > 0 && info.toRecord().name === names[0]);
  info.free();
  catalog.forEach((c) => c.free());

  const gW = generateWeightedGraph(8, { avgDegree: 3, minWeight: 0.5, maxWeight: 2.0, seed: 42 });
  assert.equal(gW.numNodes(), 8);
  const gDag = generateLayeredDag(3, 3, { edgeProb: 0.4, seed: 42 });
  const gRed = transitiveReduction(gDag);
  assert(gRed.numEdges() <= gDag.numEdges());
  const sbm = generateSbmGraph([4, 4], { pIntra: 0.7, pInter: 0.1, seed: 42 });
  assert.equal(sbm.labels.length, 8);
  const gTor = generateTorusGrid(3, 4, true);
  const gIco = generateSphericalIcosphere(1);
  const gSph = generateSphericalGrid(4, 6);
  const gHyp = generateHyperbolicTree(2, 2);
  [gW, gDag, gRed, sbm.graph, gTor, gIco, gSph, gHyp].forEach((g) => g.free());

  // 3. ScreenedPoissonField2d default gridSize=512 & C9 densityGrid / 4-arg evaluateInBounds
  const spDef = new ScreenedPoissonField2d();
  assert.equal(spDef.gridSize, 512);
  const emptyGrid = spDef.densityGrid();
  assert(emptyGrid instanceof Float32Array && emptyGrid.length === 0);
  const spSmall = new ScreenedPoissonField2d({ gridSize: 32, rhoMax: 0.05 });
  const lSmall = Layout2d.fromCoords(new Float32Array([0, 0, 0.02, 0.02, -0.02, 0.01, 0.01, -0.02]));
  const spRes = spSmall.evaluateInBounds(lSmall, -2, -2, 2, 2);
  assert.equal(spSmall.densityGridDim, 32);
  assert(spSmall.densityGrid().length === 32 * 32);
  spRes.free();
  lSmall.free();
  spSmall.free();
  spDef.free();

  // 4. Polymorphic runProgressive + ManifoldLayout2d.coordsView + Lorentz + Torus helpers + N-D Fdeb
  const g16 = generateTorusGrid(4, 4, false);
  const tor = torusSpectralPhaseInit(g16, 42);
  assert.equal(tor.kind, "torus");
  assert(tor.coordsView() instanceof Float32Array);
  const sgdT = SgdSolver.full(g16, { target: "manifold2d", iterations: 4, seed: 42 });
  await runProgressive(sgdT, tor, null, { maxStepsPerFrame: 2 });
  assert.equal(sgdT.isFinished, true);
  const panned = tor.torusAutoPan(g16);
  assert(torusEdgeSegments(0.9, 0.1, 0.1, 0.9).length >= 8);
  assert.equal(torusLayoutEdgeSegments(g16, panned).length % 4, 0);
  const repM = evaluateMetrics(g16, panned, ["sns", "geodesicDistortion"]);
  assert(Number.isFinite(repM.geodesicDistortion));
  repM.free();
  panned.free();
  tor.free();
  sgdT.free();

  const lor = ManifoldLayout2d.hyperbolicLorentz(16);
  const poinc = lor.toPoincareLayout();
  assert.equal(poinc.kind, "hyperbolicPoincare");
  poinc.free();
  lor.free();

  const nd16 = LayoutNd.init(16, 3, { seed: 42, scale: 1.0 });
  const epNd = EdgePolylines.straightLineFromGraph(g16, nd16);
  const fdebNd = new FdebSolver(epNd, { cycles: 1, stepsPerCycle: 2 });
  await runProgressive(fdebNd, epNd, null, { maxStepsPerFrame: 2 });
  assert.equal(epNd.pointsView().length % 3, 0);
  const repNd = evaluateMetrics(g16, nd16, ["sns", "np2"]);
  assert(Number.isFinite(repNd.scaleNormalizedStress));
  repNd.free();
  fdebNd.free();
  epNd.free();
  nd16.free();
  nd.free();

  // 4b. Layout3d + LayoutGenerators + Policy 1A 3D/N-D target solvers
  const l3 = Layout3d.init(16, { seed: 42, scale: 1.0 });
  const oct3 = new ArenaOctree3d();
  oct3.build(l3);
  assert.equal(oct3.pointCount, 16);
  oct3.free();
  for (const [label, solver] of [
    ["TsNet3d", TsNetSolver.pure2Hop(g16, { target: "layout3d", backend: "barnesHut", stageIterations: [2, 2, 2], seed: 42 })],
    ["TFdp3d", new TFdpSolver(g16, { target: "layout3d", backend: "exact", iterations: 3, seed: 42 })],
    ["Fa2_3d", new ForceAtlas2Solver(g16, { target: "layout3d", backend: "barnesHut", iterations: 3, seed: 42 })],
    ["Yh3d", new YifanHuSolver(g16, { target: "layout3d", backend: "barnesHut", maxIterationsPerLevel: 4, seed: 42 })],
    ["Fr3d", new FruchtermanReingoldSolver(g16, { target: "layout3d", backend: "exact", iterations: 3, seed: 42 })],
    ["LinLog3d", new LinLogSolver(g16, { target: "layout3d", backend: "barnesHut", iterations: 3, seed: 42 })],
    ["Maxent3d", new MaxentStressSolver(g16, { target: "layout3d", backend: "exact", stages: 1, stepsPerStage: 2, seed: 42 })],
    ["Sm3d", StressMajorizationSolver.fromGraph(g16, { target: "layout3d", maxIterations: 3 })],
    ["Sgd2_3d", new Sgd2Solver(g16, { target: "layout3d", iterations: 3, seed: 42 })],
  ]) {
    const st = solver.run(l3);
    assert(st === "completed" || st === "converged", `${label} unexpected status ${st}`);
    assert(l3.isAllFinite(), `${label} produced non-finite coords`);
    solver.free();
  }
  assertJsError(
    () => new IbUmapSolver(g16, { target: "layout3d" }),
    "IbUmapSolver rejects layout3d",
    "UnsupportedOperation",
  );
  l3.free();

  for (const gen of [
    new ClassicalMds({ target: "layout3d" }),
    new PivotMds({ target: "layout2d", numPivots: 6, seed: 42 }),
    new Hde({ target: "layoutNd", dimensions: 4, outputDims: 3, seed: 42 }),
    new RdMds({ target: "layoutNd", dims: 3, maxIterations: 30, seed: 42 }),
    new KorenSpectral({ target: "layout2d", maxIterations: 30, seed: 42 }),
    new CoarseBackbone(g16, { maxCoarseNodes: 4, seed: 42 }),
    new SfcLayout(g16, { curve: "hilbert", seed: 42 }),
  ]) {
    const generated =
      gen instanceof SfcLayout || gen instanceof CoarseBackbone
        ? gen.generate()
        : gen.generate(g16);
    assert(generated.isAllFinite());
    generated.free();
    gen.free();
  }

  // 5. Dijkstra, ChebyshevDogKernel, TwinDeclutterer, RTA Infinity, TsNet Auto & Stage, SgdSolver packed
  const ws = new DijkstraWorkspace(16);
  assert.equal(ws.dijkstraFrom(g16, 0).length, 16);
  assert.equal(dijkstraSingleSource(g16, 0).length, 16);
  assert.equal(allPairsDijkstra(g16).length, 256);
  ws.free();

  const kern = new ChebyshevDogKernel(g16, { jlDim: 8 });
  assert.equal(kern.numNodes, 16);
  kern.free();

  const dup = Layout2d.fromCoords(new Float32Array([0, 0, 0, 0, 2, 0, 0, 2]));
  assert(declutterTwins2d(dup) > 0);
  const dec = new TwinDeclutterer({ iterations: 2 });
  dec.apply(dup);
  dec.free();

  const rtaInf = landmarkStratifiedRtaFromLandmarks(
    dup,
    new Uint32Array([0, 3]),
    new Float64Array([0, 1, Number.POSITIVE_INFINITY, 2, 2, 1, Number.POSITIVE_INFINITY, 0]),
    { numTriplets: 32, seed: 42 },
  );
  assert(Number.isFinite(rtaInf.rta));
  rtaInf.free();
  dup.free();

  const tsAuto = TsNetSolver.pure2Hop(g16, { backend: "auto", learningRateDivisor: 2 });
  assert(typeof tsAuto.resolvedBackend === "string" && tsAuto.resolvedBackend.length > 0);
  tsAuto.free();
  const stage = new TsNetStage(2, { exaggeration: 2, lambdaKl: 1, lambdaC: 0.1, lambdaR: 0 });
  assert.equal(stage.iterations, 2);
  stage.free();

  const sgdPacked = SgdSolver.fromPackedSymmetric(4, [0, 1, 2], [1, 2, 3], [1, 1, 1], { iterations: 4, maxMu: 0.5 });
  assert.equal(sgdPacked.maxMu, 0.5);
  const l4 = Layout2d.init(4, 42);
  sgdPacked.runScheduleWindow(l4, 0, 2);
  sgdPacked.resetSchedule({ iterations: 3, seed: 42 });
  assert.equal(sgdPacked.run(l4), "completed");
  l4.free();
  sgdPacked.free();
  g16.free();
}
