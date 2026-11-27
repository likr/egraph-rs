import assert from "node:assert/strict";
import { assertJsError } from "./helpers.mjs";

// G-CAF: separated 2-cluster 50-node grid, TypedArray/number[], invariance, boundaries, evaluateMetrics
export function runMetricsClusterAreaSmoke({ eg, webPkg }) {
  const {
    ClusterAreaReport,
    CsrGraph,
    Layout2d,
    clusterAreaFaithfulness,
    clusterAreaFaithfulnessDetail,
    evaluateMetrics,
  } = eg;

  assert.equal(typeof webPkg.ClusterAreaReport, "function");
  assert.equal(typeof webPkg.clusterAreaFaithfulness, "function");
  assert.equal(typeof webPkg.clusterAreaFaithfulnessDetail, "function");
  const sepCoordsArr = [];
  const sepLabelsArr = [];
  for (let c = 0; c < 5; c++) {
    for (let r = 0; r < 5; r++) {
      sepCoordsArr.push(c, r);
      sepLabelsArr.push(0);
    }
  }
  for (let c = 0; c < 5; c++) {
    for (let r = 0; r < 5; r++) {
      sepCoordsArr.push(c + 10, r);
      sepLabelsArr.push(1);
    }
  }
  const sepCoords = new Float32Array(sepCoordsArr);
  const sepLayout = Layout2d.fromCoords(sepCoords);
  const sepLabelsU32 = new Uint32Array(sepLabelsArr);
  const cafValU32 = clusterAreaFaithfulness(sepLayout, sepLabelsU32);
  const cafValArr = clusterAreaFaithfulness(sepLayout, sepLabelsArr);
  assert.equal(cafValU32, cafValArr);
  const cafDet = clusterAreaFaithfulnessDetail(sepLayout, sepLabelsU32);
  assert(cafDet instanceof ClusterAreaReport);
  assert(Math.abs(cafDet.caf - cafValU32) < 1e-12);
  assert(cafValU32 > 0.5 && cafValU32 <= 1.0);
  assert.equal(cafDet.numClusters, 2);
  assert.equal(cafDet.numNodes, 50);
  for (const s of [
    cafDet.caf,
    cafDet.purityAdj,
    cafDet.propVis,
    cafDet.resDensity,
    cafDet.resDeconv,
    cafDet.resCombined,
    cafDet.spatialCoherence,
    cafDet.inFrac,
  ]) {
    assert(Number.isFinite(s) && s >= 0.0 && s <= 1.0);
  }
  assert(cafDet.clusterIds() instanceof Uint32Array);
  assert(cafDet.counts() instanceof Uint32Array);
  assert.deepEqual(Array.from(cafDet.clusterIds()), [0, 1]);
  assert.deepEqual(Array.from(cafDet.counts()), [25, 25]);
  for (const arr of [
    cafDet.p(),
    cafDet.q(),
    cafDet.purity(),
    cafDet.resolution(),
    cafDet.resDensityPerCluster(),
    cafDet.resDeconvPerCluster(),
    cafDet.cafPerCluster(),
    cafDet.coherence(),
  ]) {
    assert(arr instanceof Float64Array);
    assert.equal(arr.length, 2);
  }
  const cafRec = cafDet.toRecord();
  assert.equal(Object.keys(cafRec).length, 22);
  assert.equal(cafDet.effectiveBoundaryR95, 2.0);
  assert.equal(cafDet.gEff, 12);
  assert.equal(cafRec.effectiveBoundaryR95, 2.0);
  assert.equal(cafRec.gEff, 12);
  assert(Math.abs(cafRec.caf - cafValU32) < 1e-12);
  assert.deepEqual(Array.from(cafRec.clusterIds), [0, 1]);
  assert.deepEqual(Array.from(cafRec.counts), [25, 25]);
  cafDet.free();

  // Similarity/reflection/90-deg invariance
  const sepScaled = Layout2d.fromCoords(sepCoords.map((v, i) => v * 4 + (i % 2 === 0 ? 12 : -7)));
  assert(Math.abs(clusterAreaFaithfulness(sepScaled, sepLabelsU32) - cafValU32) < 1e-12);
  sepScaled.free();
  const rot90Coords = new Float32Array(sepCoords.length);
  const reflCoords = new Float32Array(sepCoords.length);
  for (let i = 0; i < 50; i++) {
    rot90Coords[2 * i] = -sepCoords[2 * i + 1];
    rot90Coords[2 * i + 1] = sepCoords[2 * i];
    reflCoords[2 * i] = sepCoords[2 * i];
    reflCoords[2 * i + 1] = -sepCoords[2 * i + 1];
  }
  const sepRot90 = Layout2d.fromCoords(rot90Coords);
  const sepRefl = Layout2d.fromCoords(reflCoords);
  assert(Math.abs(clusterAreaFaithfulness(sepRot90, sepLabelsU32) - cafValU32) < 1e-12);
  assert(Math.abs(clusterAreaFaithfulness(sepRefl, sepLabelsU32) - cafValU32) < 1e-12);
  sepRot90.free();
  sepRefl.free();

  // Imbalanced 24-node fixture: stevensGamma, boundaryR95=Infinity, evaluateMetrics
  const imbCoords = new Float32Array(48);
  const imbLabels = new Uint32Array(24);
  for (let i = 0; i < 24; i++) {
    imbCoords[2 * i] = i < 20 ? -2.0 + 0.5 * (i / 19) : 1.5 + 0.5 * ((i - 20) / 3);
    imbCoords[2 * i + 1] = Math.sin(i * 0.7) * 0.2;
    imbLabels[i] = i < 20 ? 0 : 1;
  }
  const imbLayout = Layout2d.fromCoords(imbCoords);
  const imbEdges = [];
  for (let i = 0; i < 23; i++) imbEdges.push(i, i + 1);
  const imbGraph = CsrGraph.fromEdgeList(24, new Uint32Array(imbEdges));
  const detG05 = clusterAreaFaithfulnessDetail(imbLayout, imbLabels, null, null, null, null, null, 0.5);
  const detG20 = clusterAreaFaithfulnessDetail(imbLayout, imbLabels, null, null, null, null, null, 2.0);
  assert(detG05.p()[0] < detG20.p()[0]);
  detG05.free();
  detG20.free();

  const detInf = clusterAreaFaithfulnessDetail(imbLayout, imbLabels, null, null, null, null, Number.POSITIVE_INFINITY, null);
  assert(Math.abs(detInf.inFrac - 1.0) < 1e-12);
  assert(detInf.caf >= 0.0 && detInf.caf <= 1.0);
  detInf.free();

  const expectedCustomCaf = clusterAreaFaithfulness(imbLayout, imbLabels, 64, 0.9, 8.0, 0.2, 2.5, 0.7);
  // Uint32Array vs number[] parity including u32::MAX
  const l5 = Layout2d.fromCoords(new Float32Array([0, 0, 1, 0, 0, 1, 3, 3, 4, 3]));
  const l5Arr = [100, 5, 100, 4294967295, 5];
  const l5U32 = new Uint32Array(l5Arr);
  assert.equal(clusterAreaFaithfulness(l5, l5U32), clusterAreaFaithfulness(l5, l5Arr));
  l5.free();
  const gcafEval = evaluateMetrics(
    imbGraph,
    imbLayout,
    ["clusterAreaFaithfulness", "sns"],
    8,
    64,
    42,
    null,
    imbLabels,
    null,
    null,
    null,
    null,
    null,
    null,
    64,
    0.9,
    8.0,
    0.2,
    2.5,
    0.7,
  );
  assert(Math.abs(gcafEval.clusterAreaFaithfulness - expectedCustomCaf) < 1e-12);
  const gcafEvalRec = gcafEval.toRecord();
  assert(Math.abs(gcafEvalRec.clusterAreaFaithfulness - expectedCustomCaf) < 1e-12);
  gcafEval.free();

  // nodeWeights & clusterWeights parity + evaluateMetrics forwarding
  const imbNw = new Float64Array(24);
  for (let i = 0; i < 24; i++) imbNw[i] = i < 20 ? 1.5 : 3.0;
  const imbCw = new Float64Array([30.0, 12.0]);
  const cafNw = clusterAreaFaithfulness(imbLayout, imbLabels, null, null, null, null, null, null, imbNw, null);
  const cafCw = clusterAreaFaithfulness(imbLayout, imbLabels, null, null, null, null, null, null, null, imbCw);
  assert(Math.abs(cafNw - cafCw) < 1e-12);
  const evalCw = evaluateMetrics(
    imbGraph,
    imbLayout,
    ["gcaf"],
    null,
    null,
    null,
    null,
    imbLabels,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    imbCw,
  );
  assert(Math.abs(evalCw.clusterAreaFaithfulness - cafCw) < 1e-12);
  evalCw.free();
  assert.throws(() => clusterAreaFaithfulness(imbLayout, imbLabels, null, null, null, null, null, null, imbNw, imbCw), Error);
  assert.throws(() => clusterAreaFaithfulness(imbLayout, imbLabels, null, null, null, null, null, null, null, [0.0, 1.0]), Error);

  // Degenerate boundaries: N=0, N=1, K=1, collapsed
  const l0 = Layout2d.zeros(0);
  const d0 = clusterAreaFaithfulnessDetail(l0, []);
  assert.equal(d0.caf, 0);
  assert.equal(d0.effectiveBoundaryR95, 0);
  assert.equal(d0.gEff, 0);
  assert.equal(d0.numClusters, 0);
  assert.equal(d0.numNodes, 0);
  const d0Rec = d0.toRecord();
  const expectedRecKeys = [
    "caf",
    "purityAdj",
    "propVis",
    "resDensity",
    "resDeconv",
    "resCombined",
    "spatialCoherence",
    "inFrac",
    "effectiveBoundaryR95",
    "gEff",
    "numClusters",
    "numNodes",
    "clusterIds",
    "counts",
    "p",
    "q",
    "purity",
    "resolution",
    "resDensityPerCluster",
    "resDeconvPerCluster",
    "cafPerCluster",
    "coherence",
  ].sort();
  assert.deepEqual(Object.keys(d0Rec).sort(), expectedRecKeys);
  assert.equal(d0Rec.numClusters, 0);
  assert.equal(d0Rec.numNodes, 0);
  for (const arrKey of [
    "clusterIds",
    "counts",
    "p",
    "q",
    "purity",
    "resolution",
    "resDensityPerCluster",
    "resDeconvPerCluster",
    "cafPerCluster",
    "coherence",
  ]) {
    assert.equal(d0Rec[arrKey].length, 0);
  }
  d0.free();
  l0.free();

  // f64 vs f32 Layout2d parity, parse_u32_sequence coercions & rejections, G-CAF aliases, lazy config
  const sepLayout64 = Layout2d.fromCoords(new Float64Array(sepCoordsArr), "float64");
  const cafVal64 = clusterAreaFaithfulness(sepLayout64, sepLabelsU32);
  assert(Math.abs(cafVal64 - cafValU32) < 1e-12);
  sepLayout64.free();

  const l4_64 = Layout2d.fromCoords(new Float64Array([0, 0, 0.1, 0, 4, 0, 4.1, 0]), "float64");
  const l4_32 = Layout2d.fromCoords(new Float32Array([0, 0, 0.1, 0, 4, 0, 4.1, 0]), "float32");
  const g4 = CsrGraph.fromEdgeList(4, new Uint32Array([0, 1, 1, 2, 2, 3, 3, 0]));
  const baseL4Caf = clusterAreaFaithfulness(l4_64, [0, 0, 1, 1]);
  assert(Math.abs(baseL4Caf - clusterAreaFaithfulness(l4_32, [0, 0, 1, 1])) < 1e-6);

  for (const validSeq of [
    new Uint16Array([0, 0, 1, 1]),
    new Int32Array([0, 0, 1, 1]),
    new Set([0, 1]),
    new Float64Array([0, 0, 1, 1]),
  ]) {
    if (validSeq instanceof Set) {
      const l2 = Layout2d.fromCoords(new Float64Array([0, 0, 4, 0]), "float64");
      assert.equal(clusterAreaFaithfulness(l2, validSeq), 0);
      l2.free();
    } else {
      assert.equal(clusterAreaFaithfulness(l4_64, validSeq), baseL4Caf);
    }
  }
  assert.throws(() => clusterAreaFaithfulness(l4_64, new Int32Array([0, -1, 1, 1])), Error);
  assert.throws(() => clusterAreaFaithfulness(l4_64, new Float64Array([0, 0, 1.5, 1])), Error);
  assert.throws(() => clusterAreaFaithfulness(l4_64, new BigInt64Array([0n, 0n, 1n, 1n])), Error);

  for (const alias of ["g-caf", "G-CAF", "cluster-area-faithfulness", "clusterArea"]) {
    const repAlias = evaluateMetrics(
      g4,
      l4_64,
      [alias],
      null,
      null,
      null,
      null,
      new Uint32Array([0, 0, 1, 1]),
    );
    assert(Math.abs(repAlias.clusterAreaFaithfulness - baseL4Caf) < 1e-12);
    repAlias.free();
  }

  const lazyInvalidGcaf = evaluateMetrics(
    g4,
    l4_64,
    ["sns"],
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    11,
    -1.0,
    null,
    null,
    null,
    Number.NaN,
  );
  assert(Number.isFinite(lazyInvalidGcaf.scaleNormalizedStress));
  assert.equal(lazyInvalidGcaf.clusterAreaFaithfulness, undefined);
  lazyInvalidGcaf.free();
  g4.free();
  l4_64.free();
  l4_32.free();

  const l1 = Layout2d.fromCoords(new Float32Array([1, 2]));
  assert.equal(clusterAreaFaithfulness(l1, [5]), 0);
  l1.free();

  const lCol = Layout2d.fromCoords(new Float32Array([2, -1, 2, -1, 2, -1, 2, -1, 2, -1, 2, -1]));
  assert.equal(clusterAreaFaithfulness(lCol, [0, 0, 0, 1, 1, 1]), 0);
  lCol.free();

  // Lazy gMax=4 allowed when G-CAF not requested
  const lazyOk = evaluateMetrics(
    imbGraph,
    imbLayout,
    ["aspectRatio"],
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    4,
  );
  assert(Number.isFinite(lazyOk.aspectRatio));
  assert.equal(lazyOk.clusterAreaFaithfulness, undefined);
  lazyOk.free();

  sepLayout.free();

  // Adversarial G-CAF input checks
  assertJsError(() => evaluateMetrics(imbGraph, imbLayout, ["gcaf"]), "gcaf missing labels");
  assertJsError(
    () =>
      evaluateMetrics(
        imbGraph,
        imbLayout,
        ["clusterAreaFaithfulness"],
        null,
        null,
        null,
        null,
        imbLabels,
        null,
        null,
        null,
        null,
        null,
        null,
        4,
      ),
    "gcaf lazy gMax=4",
  );
  assertJsError(() => clusterAreaFaithfulness(imbLayout, [0, 1]), "gcaf label length mismatch");
  const restLabels = Array.from(imbLabels.slice(1));
  const invalidCommunityLabels = [
    ["neg", [-1, ...restLabels]],
    ["float", [1.5, ...restLabels]],
    ["nan", [Number.NaN, ...restLabels]],
    ["inf", [Number.POSITIVE_INFINITY, ...restLabels]],
    ["overflow", [4294967296, ...restLabels]],
    ["strElem", ["bad", ...restLabels]],
    ["nullElem", [null, ...restLabels]],
    ["undefElem", [undefined, ...restLabels]],
    ["int32Array", new Int32Array([-1, ...restLabels])],
    ["float64Array", new Float64Array([1.5, ...restLabels])],
    ["arrayLikeObj", { length: 24 }],
    ["string", "000000000000000000001111"],
  ];
  for (const [tag, badLabels] of invalidCommunityLabels) {
    assertJsError(() => clusterAreaFaithfulness(imbLayout, badLabels), `gcaf invalid labels (${tag})`);
    assertJsError(() => clusterAreaFaithfulnessDetail(imbLayout, badLabels), `gcafDetail invalid labels (${tag})`);
    assertJsError(
      () => evaluateMetrics(imbGraph, imbLayout, ["gcaf"], null, null, null, null, badLabels),
      `evaluateMetrics invalid labels (${tag})`,
    );
  }
  for (const [tag, nullish] of [["null", null], ["undefined", undefined]]) {
    assertJsError(() => clusterAreaFaithfulness(imbLayout, nullish), `gcaf nullish labels (${tag})`);
    assertJsError(() => clusterAreaFaithfulnessDetail(imbLayout, nullish), `gcafDetail nullish labels (${tag})`);
    assertJsError(
      () => evaluateMetrics(imbGraph, imbLayout, ["gcaf"], null, null, null, null, nullish),
      `evaluateMetrics nullish labels (${tag})`,
    );
  }
  const nanImbCoords = new Float32Array(imbCoords);
  nanImbCoords[0] = Number.NaN;
  const nanImbLayout = Layout2d.fromCoords(nanImbCoords);
  assertJsError(() => clusterAreaFaithfulness(nanImbLayout, imbLabels), "gcaf nan layout");
  nanImbLayout.free();
  for (const badG of [11, -1, 1.5]) {
    assertJsError(() => clusterAreaFaithfulness(imbLayout, imbLabels, badG), `gcaf bad gMax=${badG}`);
  }
  for (const badSigma of [0, -0.5, Number.NaN, Number.POSITIVE_INFINITY, 400]) {
    assertJsError(
      () => clusterAreaFaithfulness(imbLayout, imbLabels, null, badSigma),
      `gcaf bad sigmaCells=${badSigma}`,
    );
  }
  assertJsError(() => clusterAreaFaithfulness(imbLayout, imbLabels, null, null, 0), "gcaf bad kappa=0");
  assertJsError(() => clusterAreaFaithfulness(imbLayout, imbLabels, null, null, null, -1), "gcaf bad tauFrac=-1");
  for (const badR95 of [0, Number.NEGATIVE_INFINITY, Number.NaN]) {
    assertJsError(
      () => clusterAreaFaithfulness(imbLayout, imbLabels, null, null, null, null, badR95),
      `gcaf bad boundaryR95=${badR95}`,
    );
  }
  for (const badGamma of [0, Number.POSITIVE_INFINITY]) {
    assertJsError(
      () => clusterAreaFaithfulness(imbLayout, imbLabels, null, null, null, null, null, badGamma),
      `gcaf bad stevensGamma=${badGamma}`,
    );
  }
  imbGraph.free();
  imbLayout.free();
}
