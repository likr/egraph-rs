import assert from "node:assert/strict";

// Community detection, overlap removal, backbone, and cluster separation metrics
export async function runCommunityOverlapAndClusterMetricsSmoke({ eg, webPkg }) {
  const { CsrGraph, Layout2d, evaluateMetrics } = eg;

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
  const labels = new Uint32Array([0, 0, 0, 0, 1, 1, 1, 1]);

  // Leiden
  const leidenRes = webPkg.leidenCommunities(g8, { objective: "modularity", resolution: 1.0, seed: 42 });
  assert(leidenRes instanceof webPkg.LeidenResult);
  assert.equal(leidenRes.numCommunities, 2);
  assert.equal(leidenRes.labels().length, 8);
  const leidenRec = leidenRes.toRecord();
  assert.equal(leidenRec.numCommunities, 2);
  assert(leidenRec.labels instanceof Uint32Array);
  leidenRes.free();

  // LinLog, FR, FA2 preventOverlap
  const lInit = Layout2d.fromCoords(
    new Float32Array([
      -2, 0, -1.5, 0.5, -2.5, 0.5, -2, 1,
      2, 0, 2.5, 0.5, 1.5, 0.5, 2, 1,
    ]),
  );
  const llSolver = new webPkg.LinLogSolver(g8, { iterations: 8, repulsionModel: "edge_repulsion", seed: 42 });
  assert(llSolver instanceof webPkg.LinLogSolver);
  assert.equal(llSolver.masses().length, 8);
  await webPkg.runProgressive(llSolver, lInit, null, { maxStepsPerFrame: 4 });
  assert.equal(llSolver.isFinished, true);
  assert(Number.isFinite(llSolver.energy(lInit)));
  llSolver.free();

  const frSolver = new webPkg.FruchtermanReingoldSolver(g8, { iterations: 8, backend: "barnes_hut", seed: 42 });
  assert(frSolver instanceof webPkg.FruchtermanReingoldSolver);
  assert(frSolver.optimalDistance > 0);
  frSolver.run(lInit);
  assert.equal(frSolver.currentTemperature, 0);
  frSolver.free();

  const fa2Overlap = new webPkg.ForceAtlas2Solver(g8, {
    iterations: 6,
    preventOverlap: true,
    defaultNodeRadius: 0.4,
    nodeRadii: [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
    seed: 42,
  });
  assert.equal(fa2Overlap.nodeRadii().length, 8);
  assert(fa2Overlap.maxDisplacement > 0);
  fa2Overlap.run(lInit);
  assert.equal(lInit.isAllFinite(), true);
  fa2Overlap.free();

  // Backbone
  const bbRes = webPkg.extractQuadrilateralBackbone(g8, { retainRatio: 0.5 });
  assert(bbRes instanceof webPkg.BackboneResult);
  assert(bbRes.umstEdges().length > 0);
  assert(bbRes.backboneEdges().length > 0);
  assert(bbRes.simmelianWeights().length > 0);
  assert(bbRes.edgeWeights().length > 0);
  const bbCsr = bbRes.backboneCsr();
  assert.equal(bbCsr.numNodes(), 8);
  const bbRec = bbRes.toRecord();
  assert(bbRec.backboneEdges instanceof Uint32Array);
  bbCsr.free();
  bbRes.free();

  const bbSolver = new webPkg.BackboneSolver(g8, { layoutMode: "sgd", iterations: 6, seed: 42 });
  assert(bbSolver instanceof webPkg.BackboneSolver);
  const bbInner = bbSolver.backboneResult();
  assert(bbInner instanceof webPkg.BackboneResult);
  bbInner.free();
  bbSolver.run(lInit);
  assert.equal(bbSolver.isFinished, true);
  bbSolver.free();

  // ClusterRefine, squarifiedTreemap, GroupInABox
  const lSwapped = Layout2d.fromCoords(
    new Float32Array([
      -5, 0, -4.5, 0.5, -5.5, 0.5, 5.2, 0.2,
      5, 0, 5.5, 0.5, 4.5, 0.5, -5.2, 0.2,
    ]),
  );
  const crSolver = webPkg.ClusterRefineSolver.fromGraph(g8, labels, { method: "kmeans", maxRounds: 5, seed: 42 });
  assert(crSolver instanceof webPkg.ClusterRefineSolver);
  assert.equal(crSolver.numClusters, 2);
  crSolver.run(lSwapped);
  assert(crSolver.lastMovedCount >= 0);
  crSolver.free();

  const movedFn = webPkg.refineClusterLayout(lSwapped, labels, { method: "hac", maxRounds: 3, seed: 42 });
  assert.equal(typeof movedFn, "number");

  const treemapRects = webPkg.squarifiedTreemap([3, 2, 1], 100, 50, 0, 0);
  assert.equal(treemapRects.length, 12);

  const gibSolver = new webPkg.GroupInABoxSolver(g8, labels, {
    boxMode: "treemap",
    intraMode: "fr",
    intraIterations: 6,
    clusterWeights: [2, 1],
    seed: 42,
  });
  assert(gibSolver instanceof webPkg.GroupInABoxSolver);
  const boxes = gibSolver.boxes();
  assert.equal(boxes.length, 2);
  assert.equal(boxes[0].nodeCount, 4);
  await webPkg.runProgressive(gibSolver, lInit, null, { maxStepsPerFrame: 3 });
  assert.equal(gibSolver.isFinished, true);
  gibSolver.free();

  // GTree & FORBID
  const lOverlap = Layout2d.fromCoords(
    new Float32Array([
      0.0, 0.0, 0.1, 0.05, -0.08, 0.07, 0.05, -0.06,
      0.12, 0.1, -0.1, -0.08, 0.02, 0.09, -0.04, 0.02,
    ]),
  );
  const hw = new Float32Array(8).fill(0.4);
  const hh = new Float32Array(8).fill(0.3);
  const lGt = Layout2d.fromCoords(new Float32Array(lOverlap.coordsView()));
  const gtSolver = new webPkg.GTreeOverlapRemoval(hw, hh, { maxIterations: 25, padding: 0.02, seed: 42 });
  assert(gtSolver instanceof webPkg.GTreeOverlapRemoval);
  gtSolver.run(lGt);
  assert.equal(gtSolver.lastOverlapCount, 0);
  gtSolver.free();

  const lFb = Layout2d.fromCoords(new Float32Array(lOverlap.coordsView()));
  const fbSolver = webPkg.ForbidOverlapRemoval.fromRadii(hw, { variant: "sparse_forbid", maxScaleIters: 20, sgdIterations: 15, seed: 42 });
  assert(fbSolver instanceof webPkg.ForbidOverlapRemoval);
  fbSolver.run(lFb);
  assert.equal(fbSolver.lastOverlapCount, 0);
  fbSolver.free();

  // Standalone cluster separation, CVI, and overlap metrics + evaluateMetrics targets + toRecord()
  const lSep = Layout2d.fromCoords(
    new Float32Array([
      -5, 0, -4.5, 0.5, -5.5, 0.5, -5, 1,
      5, 0, 5.5, 0.5, 4.5, 0.5, 5, 1,
    ]),
  );
  const paRes = webPkg.partitionAgreement(labels, labels);
  assert(paRes instanceof webPkg.PartitionAgreementResult);
  assert(Math.abs(paRes.toRecord().ari - 1.0) < 1e-12);
  paRes.free();

  const cqK = webPkg.cqKmeans(lSep, labels, 5, 50, 42);
  assert(Math.abs(cqK.ari - 1.0) < 1e-12);
  cqK.free();

  const cqH = webPkg.cqHac(lSep, labels);
  assert(Math.abs(cqH.homogeneity - 1.0) < 1e-12);
  cqH.free();

  assert(Math.abs(webPkg.distanceConsistency(lSep, labels) - 1.0) < 1e-12);
  assert(webPkg.distributionConsistency(lSep, labels, 16) > 0.9);
  const cdRes = webPkg.classDensityMeasure(lSep, labels, 32, null);
  assert(cdRes instanceof webPkg.ClassDensityResult);
  assert(cdRes.toRecord().cdm > 0.8);
  cdRes.free();

  assert(Math.abs(webPkg.sepmeKnng(lSep, labels, 2) - 1.0) < 1e-12);
  const rngScore = webPkg.sepmeRng(lSep, labels);
  const gabScore = webPkg.sepmeGab(lSep, labels);
  const gongScore = webPkg.sepmeGong(lSep, labels, 0.35);
  assert(rngScore > 0.5 && rngScore <= 1.0);
  assert(gabScore > 0.5 && gabScore <= 1.0);
  assert(gongScore > 0.5 && gongScore <= 1.0);
  assert(Math.abs(webPkg.neighborhoodHit(lSep, labels, 3) - 1.0) < 1e-12);
  assert(webPkg.silhouetteScore(lSep, labels) > 0.8);
  assert(webPkg.calinskiHarabaszScore(lSep, labels) > 50.0);
  assert(webPkg.daviesBouldinScore(lSep, labels) < 0.3);
  assert(webPkg.pairwiseDistanceOverlapError(lSep, labels, 32) < 0.1);
  assert(webPkg.sprawlterNn(lOverlap, hw, hh) > 0.0);
  assert(webPkg.sprawlterNnRadii(lOverlap, hw) > 0.0);

  const norG = webPkg.norMetrics(g8, lOverlap, lGt, hw, hh);
  assert(norG instanceof webPkg.NorMetricsResult);
  assert(Number.isFinite(norG.toRecord().elRsd));
  norG.free();

  const norNoG = webPkg.norMetricsWithoutGraph(lOverlap, lGt, hw, hh);
  assert.equal(norNoG.elRsd, 0.0);
  norNoG.free();

  const clusterMetricsReport = evaluateMetrics(
    g8,
    lSep,
    [
      "dsc",
      "dc",
      "cdm",
      "hdm",
      "sepme-knng2",
      "sepme_rng",
      "sepme_gab",
      "sepme_gong",
      "neighborhood_hit",
      "silhouette",
      "calinski_harabasz",
      "davies_bouldin",
      "cq_kmeans_ari",
      "cq_kmeans_ami",
      "cq_hac_hom",
      "cq_hac_ami",
      "pairwise_distance_overlap_error",
      "sprawlter_nn",
    ],
    {
      communityLabels: labels,
      sepmeKnngK: 2,
      neighborhoodHitK: 3,
      dcGridSize: 16,
      cdmGridSize: 32,
      numBins: 32,
      gongGamma: 0.35,
      kmeansNInit: 3,
      kmeansMaxIters: 50,
      sprawlterNodeRadius: 0.5,
      seed: 42,
    },
  );
  const clusterMetricsRecord = clusterMetricsReport.toRecord();
  assert(Math.abs(clusterMetricsRecord.sepmeKnng - 1.0) < 1e-12);
  assert(typeof clusterMetricsRecord.sprawlterNn === "number");
  clusterMetricsReport.free();

  lSep.free();
  lFb.free();
  lGt.free();
  lOverlap.free();
  lSwapped.free();
  lInit.free();
  g8.free();
}
