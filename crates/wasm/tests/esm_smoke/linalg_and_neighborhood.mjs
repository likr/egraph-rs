import assert from "node:assert/strict";
import { assertJsError, readPkgText } from "./helpers.mjs";

// Verify egraph-linalg and egraph-neighborhood APIs
export async function runLinalgAndNeighborhoodSmoke({ eg, webPkg }) {
  const {
    AnisotropicLaplacianResult,
    CsrGraph,
    CsrMatrix,
    DogVaradhanLandmarks,
    EigenResult,
    HybridKnnResult,
    KnnGraph,
    SparseProbabilities,
    SpectralMomentStats,
    TwoHopCsr,
    TwoHopKnnResult,
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
    extract1hop2hopCsr,
    findAbParams,
    korenSpectralInit,
    partialBfsKnn,
    pivotMdsInit,
    selectMaxminSpPivots,
    smallestLaplacianEigenpairs,
    spectralMomentStats,
  } = eg;

  const linalgNeighborhoodSymbols = [
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
  for (const sym of linalgNeighborhoodSymbols) {
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

  // Adversarial linalg/neighborhood JS inputs must throw Error (never RuntimeError)
  assertJsError(() => selectMaxminSpPivots(grid, 4, 1.5), "linalg seed fractional");
  assertJsError(() => pivotMdsInit(grid, 4, 42, 2, -1), "linalg negative targetStd");
  assertJsError(() => chebyshevHeatSpmv(grid, new Float64Array([Number.NaN]), 1, 1.0), "linalg NaN vec");
  assertJsError(() => KnnGraph.fromCsr(2, new Uint32Array([0, 1, 1]), new Uint32Array([1]), new Float64Array([-1.0])), "linalg neg dist");
  assertJsError(() => lm.altDistances(new Uint32Array([0]), new Uint32Array([99])), "linalg alt OOB");

  lm.free();
  fuzzy.free();
  probs.free();
  bfsKnn.free();
  hybridKnn.free();
  hybridRes.free();
  grid.free();

  // Verify index.d.ts carries clean optional camelCase linalg/neighborhood signatures
  const dtsText = await readPkgText("index.d.ts");
  assert(dtsText.includes("pivotMdsInit(graph: CsrGraph, numPivots?: number | null, seed?: number | null, minDegree?: number | null, targetStd?: number | null): Layout2d;"));
  assert(dtsText.includes("korenSpectralInit(graph: CsrGraph, targetStd?: number | null, tol?: number | null, maxIters?: number | null, seed?: number | null): Layout2d;"));
  assert(!dtsText.includes("target_std: any"));
}
