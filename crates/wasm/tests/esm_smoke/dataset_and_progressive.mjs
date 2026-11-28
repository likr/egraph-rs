import assert from "node:assert/strict";
import { assertJsError, readKarateCsv } from "./helpers.mjs";

// Verify dataset, LayoutNd.coordsView(), runProgressive, AbortSignal, and AMG/resistance APIs
export async function runDatasetAndProgressiveSmoke({ eg, webPkg }) {
  const {
    AmgPcgBatchResult,
    AmgPcgSolveResult,
    AmgPcgSolver,
    AnisotropicLaplacianResult,
    CsrGraph,
    CsrMatrix,
    EffectiveResistanceResult,
    GuMapSolver,
    Layout2d,
    LayoutNd,
    ManifoldLayout2d,
    ResistanceCurvatureFlowResult,
    ResistanceCurvatureStep,
    SgdSolver,
    TFdpSolver,
    TsNetSolver,
    allPairsBfs,
    anisotropicLaplacian,
    applyRelativeDistanceFloor,
    bfsSingleSource,
    buildCapped12hopKnn,
    buildGumap12hopPriorityCsr,
    buildRerankKnn,
    chebyshevDogJlEmbeddingMatrix,
    chebyshevHeatSpmvMatrix,
    classicalMds2d,
    computeDiffusionEmbedding,
    computeEffectiveResistances,
    computeRegularizedDogVaradhanLandmarksFromMatrix,
    computeResistanceEmbedding,
    crossingEdges,
    crossingEdgesTorus,
    crossingSummary,
    delaunayToCsr,
    evaluateMetrics,
    gabrielGraphEdges,
    graph12hopRowNormalizedProbabilities,
    graphRowNormalizedProbabilities,
    idealEdgeLengthsWithUnit,
    normalizeLayout2d,
    parseCsrCsv,
    poincareDistance,
    poincareMobiusAdd,
    prismDistortionWithEdges,
    rdMds,
    rectangleOverlaps,
    relativeNeighborhoodEdges,
    representativeMds,
    resistanceCurvatureFlow,
    robustNormalizeLayout2d,
    runProgressive,
    scaleNormalizedStressFromDistances,
    sphericalDistance,
    symmetrizeRowNormalized,
    torusDistance,
    wasmMemory,
    wrapTorusPoint,
  } = eg;

  assert.equal(typeof runProgressive, "function");
  assert.equal(typeof webPkg.runProgressive, "function");
  assert.equal(typeof eg.loadDataset, "undefined");
  assert.equal(typeof webPkg.loadDataset, "undefined");

  // Dataset + CsrGraph queries + LCC
  const karateCsv = await readKarateCsv();
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
  assert.equal(lNd.dims, 4);
  assert.equal(lNd.coordsView().length, 12);
  assert.equal(lNd.coordsView()[5], 6);
  // Force memory.grow to verify LayoutNd.coordsView refreshes automatically across detached buffers
  const prevBuf = lNd.coordsView().buffer;
  wasmMemory().grow(1);
  const refreshedNd = lNd.coordsView();
  assert.notEqual(refreshedNd.buffer, prevBuf);
  assert.equal(refreshedNd[5], 6);
  lNd.free();

  // ManifoldLayout2d constructors, coordsView(), tangentDelta, applyTangentStep & distance helpers
  const mHyp = ManifoldLayout2d.hyperbolicPoincare(2);
  assert.equal(mHyp.kind, "hyperbolicPoincare");
  assert.equal(mHyp.manifold, undefined);
  assert(mHyp.coordsView() instanceof Float32Array);
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
  const mRand = ManifoldLayout2d.randomInit("hyperbolicPoincare", 4, { scale: 0.25, seed: 42 });
  mRand.centerPoincareCentroid();
  const mc = mRand.poincareCentroid();
  assert(Math.hypot(mc[0], mc[1]) < 1e-4);
  mRand.free();
  mHyp.free();

  // Spatial + metrics helpers
  const xGraph = CsrGraph.fromEdgeList(4, new Uint32Array([0, 3, 1, 2]), null, { directed: false, sanitize: true });
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

  // Linalg + neighborhood embedding & probability helpers
  const cmds = classicalMds2d(karate);
  normalizeLayout2d(cmds, 1.0, 42);
  robustNormalizeLayout2d(cmds, { targetStd: 1.0, quantile: 0.95, seed: 42 });
  assert(cmds.isAllFinite());
  const delCsr = delaunayToCsr(cmds);
  assert.equal(delCsr.numNodes(), 34);
  assert(delCsr.numEdges() >= 33);
  delCsr.free();

  const rd = rdMds(karate, { dims: 3, eigEps: 1e-10, tol: 1e-5, maxIterations: 200, seed: 42 });
  assert.equal(rd.numNodes, 34);
  assert.equal(rd.dims, 3);
  rd.free();

  const rep = representativeMds(new Float64Array([0, 2, 1, 1, 2, 0]), 3, 2, 2);
  assert.equal(rep.numNodes, 3);
  rep.free();

  const resEmb = computeResistanceEmbedding(karate, { dim: 4 });
  const diffEmb = computeDiffusionEmbedding(karate, { dim: 4, p: 0.5 });
  const rerank = buildRerankKnn(diffEmb.coords(), 34, diffEmb.dim, 5, { coarseDim: 2, rerankFactor: 2 });
  assert.equal(rerank.numNodes, 34);
  resEmb.free();
  diffEmb.free();

  const rowProbs = graphRowNormalizedProbabilities(karate);
  assert.equal(rowProbs.numNodes, 34);
  rowProbs.free();
  const rowProbs12 = graph12hopRowNormalizedProbabilities(karate, { w2: 0.5, max2hopPerNode: 8, degreeBeta: 0.35, seed: 42 });
  assert.equal(rowProbs12.numNodes, 34);
  rowProbs12.free();
  const cappedKnn = buildCapped12hopKnn(karate, 8, 42);
  assert.equal(cappedKnn.k2Deg().length, 34);
  cappedKnn.free();
  const symProbs = symmetrizeRowNormalized(3, new Uint32Array([0, 1, 2, 3]), new Uint32Array([1, 2, 0]), new Float64Array([1, 1, 1]));
  assert.equal(symProbs.numNodes, 3);
  symProbs.free();

  const prio = buildGumap12hopPriorityCsr(3, new Uint32Array([0]), new Uint32Array([1]), new Float64Array([1]), new Uint32Array([0, 1]), new Uint32Array([2, 2]), new Float64Array([2, 2]), { tier2Margin: 1.1 });
  assert.equal(prio.numNodes, 3);
  prio.free();

  const d1 = new Float64Array([1e-6, 1.0]);
  const d2 = new Float64Array([2.0]);
  const d3 = new Float64Array([4.0]);
  const floorVal = applyRelativeDistanceFloor(d1, d2, d3, { epsDist: 1e-3, minRatio: 0.1 });
  assert(floorVal > 0);

  // Quality metrics helpers on karate
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
  const karateMetricsReport = evaluateMetrics(karate, cmds, ["sns", "idealEdgeLengths"], {
    numPivots: 8,
    numSamples: 64,
    seed: 42,
    unitEdgeLength: 1.0,
  });
  assert(Number.isFinite(karateMetricsReport.idealEdgeLengths));
  karateMetricsReport.free();

  // Solver progressive stepping + AbortSignal
  const tsPartial = TsNetSolver.fromGraphPartialBfs(karate, {
    k: 8,
    uniformPerplexity: 6.0,
    backend: "fitFft",
    stageIterations: [2, 2, 2],
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
    stageIterations: [10, 10, 10],
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
  const tfdpFromKnn = TFdpSolver.fromKnn(rerank, { backend: "exact", iterations: 4, seed: 42 });
  const lTfdp = Layout2d.fromCoords(cmds.copyCoords());
  assert.equal(tfdpFromKnn.run(lTfdp), "completed");
  tfdpFromKnn.free();
  lTfdp.free();
  rerank.free();

  const sgdDog = SgdSolver.sparseBarrierDog(karate, { iterations: 4, seed: 42, minDegree: 1, jlDim: 8 });
  const dStats = sgdDog.dogStats();
  assert(dStats && dStats.degree > 0);
  dStats.free();
  sgdDog.free();

  const sgdPairs = SgdSolver.fromPairs(2, [0], [1], [1.0], { iterations: 3 });
  const lPair = Layout2d.fromCoords([0, 0, 2, 0]);
  assert.equal(sgdPairs.run(lPair), "completed");
  sgdPairs.free();
  lPair.free();

  const gmPos = GuMapSolver.fromGraphDog(karate, { iterations: 4, seed: 42, minDegree: 1, jlDim: 8 });
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

  const aniso = anisotropicLaplacian(karate, { alpha: 0.0, beta: 0.0, degreeReg: 0.0, reweightedDegreeNorm: false });
  assert(aniso instanceof AnisotropicLaplacianResult);
  const anisoMat = aniso.matrix();
  assert(anisoMat instanceof CsrMatrix);
  assert.equal(anisoMat.numNodes, 34);
  assert(anisoMat.estimateLambdaMaxLanczos(16, 1.05, 42) > 2.0);
  const heatMat = chebyshevHeatSpmvMatrix(anisoMat, bKarate, 1, 0.25);
  assert.equal(heatMat.length, 34);
  const jlMat = chebyshevDogJlEmbeddingMatrix(anisoMat, { jlDim: 8, seed: 42 });
  assert.equal(jlMat.length, 34 * 8);
  const lmMat = computeRegularizedDogVaradhanLandmarksFromMatrix(karate, anisoMat, {
    numLandmarks: 8,
    minDegree: 2,
    jlDim: 8,
    seed: 42,
    degreeNormalize: true,
    pExp: 0.5,
    epsPsd: 1e-4,
    tCoarse: 0.0,
  });
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
  assert.equal(rcfKarate.suggestedManifold, "hyperbolicPoincare");
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
