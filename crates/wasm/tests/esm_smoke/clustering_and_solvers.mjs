import assert from "node:assert/strict";
import { assertJsError, makeBarbellGraph, makeCircleCoords, makeGridGraph, readPkgText } from "./helpers.mjs";

// Verify clustering, coarsening, ManifoldLayout2d, and cooperative f32 solvers
export async function runClusteringAndSolversSmoke({ eg }) {
  const {
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
    coarsenGraph,
    kmeansPlusPlus,
    labelPropagation,
    louvainCommunities,
    projectClusteredLayout,
    spectralClustering,
  } = eg;

  const clusteringSolverExports = [
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
  for (const name of clusteringSolverExports) {
    assert.equal(typeof eg[name], "function", `missing clustering/solver runtime export: ${name}`);
  }

  const indexJsText = await readPkgText("index.js");
  for (const name of clusteringSolverExports) {
    assert(indexJsText.includes(name), `missing clustering/solver export in pkg/index.js: ${name}`);
  }

  const dtsText = await readPkgText("index.d.ts");
  assert(dtsText.includes("export interface TsNetOptions {"));
  assert(dtsText.includes("jacobiPrecondition?: boolean | null;"));
  assert(dtsText.includes("export interface SparseBarrierDogOptions extends SparseBarrierOptions {"));
  assert(dtsText.includes("static pure2Hop(graph: CsrGraph, options?: TsNetOptions | null): TsNetSolver;"));
  assert(dtsText.includes("static sparseBarrierDog(graph: CsrGraph, options?: SparseBarrierDogOptions | null): SgdSolver;"));
  assert(dtsText.includes("static fromGraphDog(graph: CsrGraph, options?: GuMapDogOptions | null): GuMapSolver;"));
  assert(dtsText.includes("constructor(graph: CsrGraph, options?: SpatialRefinerOptions | null);"));
  assert(dtsText.includes("static fromGraph(graph: CsrGraph, options?: StressMajorizationOptions | null): StressMajorizationSolver;"));
  assert(dtsText.includes("static fromGraph(graph: CsrGraph, options?: KamadaKawaiOptions | null): KamadaKawaiSolver;"));
  assert(dtsText.includes("export function kmeansPlusPlus(points: Float64Array | number[], numNodes: number, dims: number, options?: KMeansOptions | null): KMeansResult;"));
  assert(dtsText.includes("export function louvainCommunities(graph: CsrGraph, options?: LouvainOptions | null): LouvainResult;"));
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
  assert(dtsText.includes("export type EgraphErrorCode ="));
  assert(dtsText.includes('"BackendUnavailable"'));
  assert(dtsText.includes('"DeviceLimitExceeded"'));
  assert(dtsText.includes("export interface LayoutStorage {"));
  assert(dtsText.includes("export interface IterativeSolver<TTarget"));
  assert(dtsText.includes("export interface LayoutGenerator<TOutput"));
  assert(dtsText.includes("export interface LayoutTransform<TTarget"));

  // Clustering & coarsening smoke check
  const km = kmeansPlusPlus(
    new Float64Array([
      10, 10, 10.2, 9.8, 9.8, 10.1,
      -10, -10, -9.9, -10.2, -10.1, -9.8,
    ]),
    6,
    2,
    { k: 2, maxIterations: 100, tol: 1e-4, nInit: 2, seed: 42 },
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
  assert(km.labelsView() instanceof Uint32Array);
  assert(km.centroidsView() instanceof Float64Array);
  assert.equal(km.centroids().length, 4);
  km.free();

  const barbell = makeBarbellGraph(eg, 6);
  const specRes = spectralClustering(barbell, {
    k: 2,
    maxEigenIterations: 300,
    eigenTol: 1e-4,
    kmeansMaxIterations: 100,
    kmeansTol: 1e-4,
    seed: 42,
  });
  assert(specRes instanceof eg.SpectralClusteringResult);
  const specLabels = specRes.labelsView();
  assert(specLabels instanceof Uint32Array);
  assert.equal(specLabels.length, 12);
  assert.notEqual(specLabels[0], specLabels[11]);
  specRes.free();

  const louv = louvainCommunities(barbell, { maxPasses: 50, resolution: 1.0, minModularityGain: 1e-7, seed: 42 });
  assert(louv instanceof LouvainResult);
  assert(louv.numCommunities >= 2);
  assert(louv.modularity > 0.2);
  assert(louv.passes >= 1);
  assert(louv.labelsView() instanceof Uint32Array);
  const louvLabels = louv.labels();
  assert.equal(louvLabels.length, 12);
  louv.free();

  const lpaRes = labelPropagation(barbell, 50, 42);
  assert(lpaRes instanceof eg.LabelPropagationResult);
  assert(lpaRes.labelsView() instanceof Uint32Array);
  assert.equal(lpaRes.labels().length, 12);
  lpaRes.free();

  const leidenRes = eg.leidenCommunities(barbell, { seed: 42 });
  assert(leidenRes instanceof eg.LeidenResult);
  assert(leidenRes.labelsView() instanceof Uint32Array);
  leidenRes.free();

  const coarse = coarsenGraph(barbell, louvLabels);
  assert(coarse instanceof CoarsenedGraphResult);
  const coarseGraph = coarse.graph();
  assert.equal(coarseGraph.numNodes(), coarse.numClusters);
  assert.equal(coarse.nodeToCoarse().length, 12);
  assert.equal(coarse.coarseSizes().reduce((a, b) => a + b, 0), 12);

  const coarseLayout = eg.Layout2d.fromCoords(makeCircleCoords(coarse.numClusters, 2.0));
  const fineProjected = projectClusteredLayout(coarseLayout, coarse.nodeToCoarse(), { spreadRadius: 0.25, seed: 42 });
  assert.equal(fineProjected.len(), 12);
  assert(fineProjected.isAllFinite());
  coarseLayout.free();
  fineProjected.free();
  coarseGraph.free();
  coarse.free();
  barbell.free();

  // Layout solvers + in-place buffer mutation + step == run parity
  const grid = makeGridGraph(eg, 4, 4);

  // TsNetSolver.pure2Hop (also verifying totalFft2dCalls on linearFft)
  const tsOpts = {
    backend: "linearFft",
    stageIterations: [4, 4, 4],
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
    stageIterations: [2, 2, 2],
    seed: 11,
  });
  assert.equal(tsRnSolver.samplingMode, "twoStepRandomWalk");
  const tsRnLayout = eg.Layout2d.fromCoords(makeCircleCoords(16, 1.0));
  assert.equal(tsRnSolver.run(tsRnLayout), "completed");
  assert(tsRnLayout.isAllFinite());
  tsRnSolver.free();
  tsRnLayout.free();

  // TsNetSolver jacobiPrecondition option on star graph + non-bool error validation
  const starEdges = [];
  for (let v = 1; v < 20; v++) {
    starEdges.push(0, v);
  }
  const starGraph = eg.CsrGraph.fromEdgeList(20, new Uint32Array(starEdges), null, { directed: false, sanitize: true });
  const starInit = makeCircleCoords(20, 0.5);
  starInit[0] = 1.0;
  starInit[1] = 0.0;
  for (const [label, mk] of [
    ["pure2Hop", (jp) => TsNetSolver.pure2Hop(starGraph, { backend: "exact", stageIterations: [4, 2, 2], jacobiPrecondition: jp, seed: 42 })],
    ["fromGraphRowNormalized", (jp) => TsNetSolver.fromGraphRowNormalized(starGraph, { w2: 0.0, backend: "exact", stageIterations: [4, 2, 2], jacobiPrecondition: jp, seed: 42 })],
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
    iterations: 8,
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

  // ManifoldLayout2d across all 3 manifolds + Policy 1A target: "manifold2d" & coordsView()
  for (const kind of ["hyperbolicPoincare", "spherical", "torus"]) {
    const ml = ManifoldLayout2d.fromCoords(kind, makeCircleCoords(16, 0.35));
    assert.equal(ml.kind, kind);
    assert.equal(ml.len(), 16);
    assert(ml.isAllFinite());
    const manifoldSgd = SgdSolver.full(grid, { target: "manifold2d", iterations: 4, seed: 23 });
    assert.equal(manifoldSgd.stats(), undefined);
    assert.equal(manifoldSgd.target, "manifold2d");
    assert.equal(manifoldSgd.run(ml), "completed");
    assert(ml.isAllFinite());
    assert(ml.coordsView() instanceof Float32Array);
    assert(ml.geodesicDistance(0, 15) >= 0);
    manifoldSgd.free();
    ml.free();
  }

  // GuMapSolver.fromGraphDog
  const gumap = GuMapSolver.fromGraphDog(grid, {
    max2hopPerNode: 16,
    iterations: 8,
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
    iterations: 2,
    kCap: 8,
    wAnchor2: 0.5,
    kCapDecay: 0.5,
    kCapMin: 2,
    sgdIterations: 4,
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

  // Adversarial error checks raise recoverable JS Errors with .code (no WASM traps)
  const dirGraph = eg.CsrGraph.fromEdgeList(3, new Uint32Array([0, 1, 1, 2]), null, { directed: true, sanitize: true });
  assertJsError(() => louvainCommunities(dirGraph), "louvain on directed graph", "InvalidInput");
  assertJsError(() => ManifoldLayout2d.zeros("hyperbolic_invalid", 4), "invalid manifold kind", "InvalidInput");
  assertJsError(
    () => TsNetSolver.pure2Hop(grid, { target: "layoutNd", backend: "fitFft" }),
    "illegal (layoutNd, fitFft) pair",
    "UnsupportedOperation",
  );
  dirGraph.free();
  grid.free();
}
