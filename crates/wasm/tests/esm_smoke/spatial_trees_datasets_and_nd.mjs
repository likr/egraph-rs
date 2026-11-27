import assert from "node:assert/strict";

export async function runSpatialTreesDatasetsAndNdSmoke({ eg }) {
  const {
    ArenaOctree3d,
    ChebyshevDogKernel,
    DijkstraWorkspace,
    DynamicKdTree,
    FdebSolver,
    Layout2d,
    LayoutNd,
    ManifoldLayout2d,
    ScreenedPoissonField2d,
    SgdSolver,
    TsNetSolver,
    TsNetStage,
    TwinDeclutterer,
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

  const gW = generateWeightedGraph(8, 3, 0.5, 2.0, 42);
  assert.equal(gW.numNodes(), 8);
  const gDag = generateLayeredDag(3, 3, 0.4, 42);
  const gRed = transitiveReduction(gDag);
  assert(gRed.numEdges() <= gDag.numEdges());
  const sbm = generateSbmGraph([4, 4], 0.7, 0.1, 42);
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
  const sgdT = SgdSolver.full(g16, { iterations: 4, seed: 42 });
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
  assert.equal(poinc.kind, "hyperbolic");
  poinc.free();
  lor.free();

  const nd16 = LayoutNd.init(16, 3, 42, 1.0);
  const fdebNd = new FdebSolver(g16, { cycles: 1, stepsPerCycle: 2 });
  await runProgressive(fdebNd, nd16, null, { maxStepsPerFrame: 2 });
  assert.equal(fdebNd.points().length % 3, 0);
  const repNd = evaluateMetrics(g16, nd16, ["sns", "np2"]);
  assert(Number.isFinite(repNd.scaleNormalizedStress));
  repNd.free();
  fdebNd.free();
  nd16.free();
  nd.free();

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
  dec.run(dup);
  dec.free();

  const rtaInf = landmarkStratifiedRtaFromLandmarks(
    dup,
    new Uint32Array([0, 3]),
    new Float64Array([0, 1, Number.POSITIVE_INFINITY, 2, 2, 1, Number.POSITIVE_INFINITY, 0]),
    32,
    42,
  );
  assert(Number.isFinite(rtaInf.rta));
  rtaInf.free();
  dup.free();

  const tsAuto = TsNetSolver.pure2Hop(g16, { backend: "auto", learningRateDivisor: 2 });
  assert(typeof tsAuto.resolvedBackend === "string" && tsAuto.resolvedBackend.length > 0);
  tsAuto.free();
  const stage = new TsNetStage(2, 2, 1, 0.1, 0);
  assert.equal(stage.iterations, 2);
  stage.free();

  const sgdPacked = SgdSolver.fromPackedSymmetric(4, [0, 1, 2], [1, 2, 3], [1, 1, 1], null, null, { iterations: 4, maxMu: 0.5 });
  assert.equal(sgdPacked.maxMu, 0.5);
  const l4 = Layout2d.init(4, 42);
  sgdPacked.runScheduleWindow(l4, 0, 2);
  sgdPacked.resetSchedule(3, null, null, null, 42);
  assert.equal(sgdPacked.run(l4), "completed");
  l4.free();
  sgdPacked.free();
  g16.free();
}
