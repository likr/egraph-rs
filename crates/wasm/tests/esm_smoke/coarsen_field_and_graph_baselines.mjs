import assert from "node:assert/strict";

// Louvain coarsen, pivoted RHGH landmarks, coarse backbone, centroid tether, PaCMAP backbone, and Screened Poisson field
export function runCoarsenAndFieldSmoke({ eg, webPkg }) {
  const { CsrGraph, DogVaradhanLandmarks, Layout2d, SgdSolver } = eg;

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
      phaseIterations: [5, 5, 5],
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

  // ScreenedPoissonField2d & ScreenedPoissonRefiner
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
}

// Multilevel, stress, force-directed, overlap-removal, and SFC baseline layouts
export async function runGraphBaselinesSmoke({ eg, webPkg }) {
  const { CsrGraph, Layout2d, SgdSolver } = eg;

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

  // HDE
  const lHde = webPkg.hde(g8, { dimensions: 4, seed: 42, normalize: true });
  assert.equal(lHde.len(), 8);
  assert.equal(lHde.isAllFinite(), true);
  lHde.free();

  const lHdeNd = webPkg.hdeNd(g8, { dimensions: 4, outputDims: 3, pivots: [0, 3, 4, 7] });
  assert.equal(lHdeNd.numNodes, 8);
  assert.equal(lHdeNd.dims, 3);
  assert.equal(lHdeNd.isAllFinite(), true);
  lHdeNd.free();

  // SgdSolver baseline constructors
  const lWork = Layout2d.init(8, 42);
  for (const sgd of [
    SgdSolver.ortmannSparseStress(g8, { numPivots: 4, pivotStrategy: "kmeans_sp", iterations: 8, seed: 42 }),
    SgdSolver.largevis(g8, { iterations: 8, negativeSamples: 3, seed: 42 }),
    SgdSolver.drgraph(g8, { iterations: 8, negativeSamples: 3, seed: 42 }),
    SgdSolver.graphCne(g8, { objective: "infonce", iterations: 8, negativeSamples: 3, seed: 42 }),
  ]) {
    sgd.run(lWork);
    assert.equal(sgd.isFinished, true);
    assert.equal(lWork.isAllFinite(), true);
    sgd.free();
  }

  // MaxentStressSolver
  const maxent = new webPkg.MaxentStressSolver(g8, { stages: 2, stepsPerStage: 4, seed: 42 });
  assert.equal(maxent.numNodes, 8);
  assert(Number.isFinite(maxent.effectiveQ));
  assert(Number.isFinite(maxent.currentAlpha));
  await webPkg.runProgressive(maxent, lWork, null, { maxStepsPerFrame: 4 });
  assert.equal(maxent.isFinished, true);
  assert.equal(lWork.isAllFinite(), true);
  maxent.free();

  // GRIP, GEM, SEL, PFS', SFC
  const grip = new webPkg.GripSolver(g8, { roundsMin: 2, roundsMax: 4, seed: 42 });
  assert(grip.numLevels >= 1);
  grip.run(lWork);
  assert.equal(grip.isFinished, true);
  grip.free();

  const gem = new webPkg.GemSolver(g8, { maxIterations: 8, seed: 42 });
  gem.run(lWork);
  assert.equal(gem.isFinished, true);
  gem.free();

  const sel = new webPkg.SelLayoutSolver(g8, { mapping: "sigmoid", layoutBackend: "fr", iterations: 8, seed: 42 });
  assert(Number.isFinite(sel.seTarget));
  assert.equal(sel.modulatedWeights().length, 13);
  sel.run(lWork);
  assert.equal(sel.isFinished, true);
  sel.free();

  const lOverlap = Layout2d.fromCoords(
    new Float32Array([
      0.0, 0.0, 0.1, 0.05, -0.08, 0.07, 0.05, -0.06,
      0.12, 0.1, -0.1, -0.08, 0.02, 0.09, -0.04, 0.02,
    ]),
  );
  const pfs = webPkg.PfsPrimeOverlapRemoval.fromRadii(new Float32Array(8).fill(0.4), { padding: 0.02, seed: 42 });
  pfs.run(lOverlap);
  assert.equal(pfs.isFinished, true);
  assert.equal(pfs.lastOverlapCount, 0);
  pfs.free();
  lOverlap.free();

  const sfc = new webPkg.SfcLayoutSolver(g8, { curve: "hilbert", scale: 100.0, seed: 42 });
  assert.equal(sfc.vertexOrder().length, 8);
  sfc.run(lWork);
  assert.equal(sfc.isFinished, true);
  assert.equal(lWork.isAllFinite(), true);
  sfc.free();

  lWork.free();
  g8.free();
}
