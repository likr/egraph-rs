export const polyWrapperRuntimeHelpers = `function attachCoordsView(LayoutClass) {
  if (typeof LayoutClass.prototype.coordsView !== "function") {
    LayoutClass.prototype.coordsView = function coordsView() {
      const buf = wasmMemory().buffer;
      const ptr = this.coordsPtr();
      const len = this.coordsLen();
      if (
        !this._coordsView ||
        this._coordsViewBuf !== buf ||
        this._coordsView.byteOffset !== ptr ||
        this._coordsView.length !== len
      ) {
        this._coordsView = new Float32Array(buf, ptr, len);
        this._coordsViewBuf = buf;
      }
      return this._coordsView;
    };
  }
  return LayoutClass;
}

function attachTypedView(ResultClass, methodName, ptrFn, lenFn, TypedCtor, cacheKey) {
  if (typeof ResultClass?.prototype?.[ptrFn] === "function") {
    ResultClass.prototype[methodName] = function () {
      const buf = wasmMemory().buffer;
      const ptr = this[ptrFn]();
      const len = this[lenFn]();
      const bufKey = cacheKey + "Buf";
      if (
        !this[cacheKey] ||
        this[bufKey] !== buf ||
        this[cacheKey].byteOffset !== ptr ||
        this[cacheKey].length !== len
      ) {
        this[cacheKey] = new TypedCtor(buf, ptr, len);
        this[bufKey] = buf;
      }
      return this[cacheKey];
    };
  }
  return ResultClass;
}

function attachSolverDispatch(SolverClass, { supportsNd = false, supportsManifold = false } = {}) {
  const proto = SolverClass.prototype;
  proto.step = function step(layout) {
    if (layout instanceof _Layout2d) return this._step2d(layout);
    if (supportsNd && layout instanceof _LayoutNd) return this._stepNd(layout);
    if (supportsManifold && layout instanceof _ManifoldLayout2d) return this._stepManifold(layout);
    throw new TypeError(\`\${SolverClass.name}.step: unsupported layout type\`);
  };
  proto.runSteps = function runSteps(layout, steps) {
    if (layout instanceof _Layout2d) return this._runSteps2d(layout, steps);
    if (supportsNd && layout instanceof _LayoutNd) return this._runStepsNd(layout, steps);
    if (supportsManifold && layout instanceof _ManifoldLayout2d) {
      return this._runStepsManifold(layout, steps);
    }
    throw new TypeError(\`\${SolverClass.name}.runSteps: unsupported layout type\`);
  };
  proto.run = function run(layout) {
    if (layout instanceof _Layout2d) return this._run2d(layout);
    if (supportsNd && layout instanceof _LayoutNd) return this._runNd(layout);
    if (supportsManifold && layout instanceof _ManifoldLayout2d) return this._runManifold(layout);
    throw new TypeError(\`\${SolverClass.name}.run: unsupported layout type\`);
  };
  return SolverClass;
}

function dispatchLayout3(name, fn2d, fnNd, fnManifold) {
  return function (layout, ...args) {
    if (layout instanceof _Layout2d) return fn2d(layout, ...args);
    if (fnNd && layout instanceof _LayoutNd) return fnNd(layout, ...args);
    if (fnManifold && layout instanceof _ManifoldLayout2d) return fnManifold(layout, ...args);
    throw new TypeError(\`\${name}: unsupported layout type\`);
  };
}

function dispatchGraphLayout3(name, fn2d, fnNd, fnManifold) {
  return function (graph, layout, ...args) {
    if (layout instanceof _Layout2d) return fn2d(graph, layout, ...args);
    if (fnNd && layout instanceof _LayoutNd) return fnNd(graph, layout, ...args);
    if (fnManifold && layout instanceof _ManifoldLayout2d) return fnManifold(graph, layout, ...args);
    throw new TypeError(\`\${name}: unsupported layout type\`);
  };
}
`;

export const polyExportDefinitions = `export const Layout2d = attachCoordsView(_Layout2d);
export const LayoutNd = attachCoordsView(_LayoutNd);
export const ManifoldLayout2d = attachCoordsView(_ManifoldLayout2d);

export const LouvainResult = attachTypedView(_LouvainResult, "labelsView", "labelsPtr", "labelsLen", Uint32Array, "_labelsView");
export const LeidenResult = attachTypedView(_LeidenResult, "labelsView", "labelsPtr", "labelsLen", Uint32Array, "_labelsView");
export const LabelPropagationResult = attachTypedView(_LabelPropagationResult, "labelsView", "labelsPtr", "labelsLen", Uint32Array, "_labelsView");
export const SpectralClusteringResult = attachTypedView(_SpectralClusteringResult, "labelsView", "labelsPtr", "labelsLen", Uint32Array, "_labelsView");
export const KMeansResult = attachTypedView(
  attachTypedView(_KMeansResult, "labelsView", "labelsPtr", "labelsLen", Uint32Array, "_labelsView"),
  "centroidsView",
  "centroidsPtr",
  "centroidsLen",
  Float64Array,
  "_centroidsView",
);

export const SgdSolver = attachSolverDispatch(_SgdSolver, { supportsNd: true, supportsManifold: true });
SgdSolver.prototype.runScheduleWindow = function runScheduleWindow(layout, startIter, count) {
  if (layout instanceof _Layout2d) return this._runScheduleWindow2d(layout, startIter, count);
  if (layout instanceof _ManifoldLayout2d) return this._runScheduleWindowManifold(layout, startIter, count);
  throw new TypeError("SgdSolver.runScheduleWindow: expected Layout2d or ManifoldLayout2d");
};

export const TsNetSolver = attachSolverDispatch(_TsNetSolver, { supportsNd: true, supportsManifold: true });
export const GuMapSolver = attachSolverDispatch(_GuMapSolver, { supportsNd: true, supportsManifold: true });
export const FruchtermanReingoldSolver = attachSolverDispatch(_FruchtermanReingoldSolver, { supportsNd: true, supportsManifold: true });

export const StressMajorizationSolver = attachSolverDispatch(_StressMajorizationSolver, { supportsNd: true, supportsManifold: true });
StressMajorizationSolver.prototype.computeStress = function computeStress(layout) {
  if (layout instanceof _Layout2d) return this._computeStress2d(layout);
  if (layout instanceof _LayoutNd) return this._computeStressNd(layout);
  if (layout instanceof _ManifoldLayout2d) return this._computeStressManifold(layout);
  throw new TypeError("StressMajorizationSolver.computeStress: unsupported layout type");
};

export const KamadaKawaiSolver = attachSolverDispatch(_KamadaKawaiSolver, { supportsNd: true, supportsManifold: true });
KamadaKawaiSolver.prototype.computeStress = function computeStress(layout) {
  if (layout instanceof _Layout2d) return this._computeStress2d(layout);
  if (layout instanceof _LayoutNd) return this._computeStressNd(layout);
  if (layout instanceof _ManifoldLayout2d) return this._computeStressManifold(layout);
  throw new TypeError("KamadaKawaiSolver.computeStress: unsupported layout type");
};

export const LinLogSolver = attachSolverDispatch(_LinLogSolver, { supportsNd: true });
LinLogSolver.prototype.energy = function energy(layout) {
  if (layout instanceof _Layout2d) return this._energy2d(layout);
  if (layout instanceof _LayoutNd) return this._energyNd(layout);
  throw new TypeError("LinLogSolver.energy: expected Layout2d or LayoutNd");
};

export const ForceAtlas2Solver = attachSolverDispatch(_ForceAtlas2Solver, { supportsNd: true });
export const YifanHuSolver = attachSolverDispatch(_YifanHuSolver, { supportsNd: true });
export const MaxentStressSolver = attachSolverDispatch(_MaxentStressSolver, { supportsNd: true });
export const FdebSolver = attachSolverDispatch(_FdebSolver, { supportsNd: true });

export const ArenaOctree3d = _ArenaOctree3d;
ArenaOctree3d.prototype.build = function build(coordsOrLayout) {
  if (coordsOrLayout instanceof _LayoutNd) return this.buildNd(coordsOrLayout);
  return this._buildFlat(coordsOrLayout);
};
ArenaOctree3d.prototype.buildWeighted = function buildWeighted(coordsOrLayout, masses) {
  if (coordsOrLayout instanceof _LayoutNd) return this.buildWeightedNd(coordsOrLayout, masses);
  return this._buildWeightedFlat(coordsOrLayout, masses);
};

export const DynamicKdTree = new Proxy(_DynamicKdTree, {
  construct(target, args) {
    if (args.length === 1 && args[0] instanceof _LayoutNd) {
      return target.fromLayoutNd(args[0]);
    }
    return Reflect.construct(target, args);
  },
});

export function evaluateMetrics(graph, layout, targets, ...args) {
  if (layout instanceof _LayoutNd) return _evaluateMetricsNd(graph, layout, targets, ...args);
  if (layout instanceof _ManifoldLayout2d) return _evaluateMetricsManifold(graph, layout, targets, ...args);
  if (layout instanceof _Layout2d) {
    const opt = args.length === 1 && args[0] && typeof args[0] === "object" && !Array.isArray(args[0]) ? args[0] : null;
    if (opt && opt.originalLayout != null) {
      return _evaluateMetrics2dWithOriginal(graph, layout, targets, opt, opt.originalLayout);
    }
    return _evaluateMetrics2d(graph, layout, targets, ...args);
  }
  throw new TypeError("evaluateMetrics: unsupported layout type");
}

export const scaleNormalizedStress = dispatchGraphLayout3("scaleNormalizedStress", _scaleNormalizedStress2d, _scaleNormalizedStressNd, _scaleNormalizedStressManifold);
export const scaleNormalizedStressDetail = dispatchGraphLayout3("scaleNormalizedStressDetail", _scaleNormalizedStressDetail2d, _scaleNormalizedStressDetailNd, _scaleNormalizedStressDetailManifold);
export const sampledScaleNormalizedStress = dispatchGraphLayout3("sampledScaleNormalizedStress", _sampledScaleNormalizedStress2d, _sampledScaleNormalizedStressNd, _sampledScaleNormalizedStressManifold);
export const scaleNormalizedStressFromDistances = dispatchLayout3("scaleNormalizedStressFromDistances", _scaleNormalizedStressFromDistances2d, _scaleNormalizedStressFromDistancesNd, _scaleNormalizedStressFromDistancesManifold);
export const neighborhoodPreservation1Hop = dispatchGraphLayout3("neighborhoodPreservation1Hop", _neighborhoodPreservation1Hop2d, _neighborhoodPreservation1HopNd, _neighborhoodPreservation1HopManifold);
export const neighborhoodPreservation2Hop = dispatchGraphLayout3("neighborhoodPreservation2Hop", _neighborhoodPreservation2Hop2d, _neighborhoodPreservation2HopNd, _neighborhoodPreservation2HopManifold);
export const sampledNp1 = dispatchGraphLayout3("sampledNp1", _sampledNp12d, _sampledNp1Nd, _sampledNp1Manifold);
export const sampledNp2 = dispatchGraphLayout3("sampledNp2", _sampledNp22d, _sampledNp2Nd, _sampledNp2Manifold);
export const gabrielGraphProperty = dispatchGraphLayout3("gabrielGraphProperty", _gabrielGraphProperty2d, _gabrielGraphPropertyNd, _gabrielGraphPropertyManifold);
export const nodeResolution = dispatchLayout3("nodeResolution", _nodeResolution2d, _nodeResolutionNd, _nodeResolutionManifold);
export const drngPreservation = dispatchGraphLayout3("drngPreservation", _drngPreservation2d, _drngPreservationNd, _drngPreservationManifold);
export const dggPreservation = dispatchGraphLayout3("dggPreservation", _dggPreservation2d, _dggPreservationNd, _dggPreservationManifold);
export const idealEdgeLengths = dispatchGraphLayout3("idealEdgeLengths", _idealEdgeLengths2d, _idealEdgeLengthsNd, _idealEdgeLengthsManifold);
export const idealEdgeLengthsWithUnit = dispatchGraphLayout3("idealEdgeLengthsWithUnit", _idealEdgeLengthsWithUnit2d, _idealEdgeLengthsWithUnitNd, _idealEdgeLengthsWithUnitManifold);
export const layoutDegeneracy = dispatchGraphLayout3("layoutDegeneracy", _layoutDegeneracy2d, _layoutDegeneracyNd, _layoutDegeneracyManifold);
export const gabrielGraphEdges = dispatchLayout3("gabrielGraphEdges", _gabrielGraphEdges2d, _gabrielGraphEdgesNd, null);
export const relativeNeighborhoodEdges = dispatchLayout3("relativeNeighborhoodEdges", _relativeNeighborhoodEdges2d, _relativeNeighborhoodEdgesNd, null);
export const landmarkStratifiedRta = dispatchGraphLayout3("landmarkStratifiedRta", _landmarkStratifiedRta2d, _landmarkStratifiedRtaNd, _landmarkStratifiedRtaManifold);
export const landmarkStratifiedRtaDetail = dispatchGraphLayout3("landmarkStratifiedRtaDetail", _landmarkStratifiedRtaDetail2d, _landmarkStratifiedRtaDetailNd, _landmarkStratifiedRtaDetailManifold);
export const landmarkStratifiedRtaFromLandmarks = dispatchLayout3("landmarkStratifiedRtaFromLandmarks", _landmarkStratifiedRtaFromLandmarks2d, _landmarkStratifiedRtaFromLandmarksNd, _landmarkStratifiedRtaFromLandmarksManifold);
export const torusCrossingNumber = dispatchGraphLayout3("torusCrossingNumber", _torusCrossingNumber2d, null, _torusCrossingNumberManifold);
export const crossingEdgesTorus = dispatchGraphLayout3("crossingEdgesTorus", _crossingEdgesTorus2d, null, _crossingEdgesTorusManifold);
export const torusLayoutEdgeSegments = dispatchGraphLayout3("torusLayoutEdgeSegments", _torusLayoutEdgeSegments2d, null, _torusLayoutEdgeSegmentsManifold);
export const cqKmeans = dispatchLayout3("cqKmeans", _cqKmeans2d, _cqKmeansNd, _cqKmeansManifold);
export const cqHac = dispatchLayout3("cqHac", _cqHac2d, _cqHacNd, _cqHacManifold);
export const distanceConsistency = dispatchLayout3("distanceConsistency", _distanceConsistency2d, _distanceConsistencyNd, _distanceConsistencyManifold);
export const distributionConsistency = dispatchLayout3("distributionConsistency", _distributionConsistency2d, _distributionConsistencyNd, _distributionConsistencyManifold);
export const sepmeKnng = dispatchLayout3("sepmeKnng", _sepmeKnng2d, _sepmeKnngNd, _sepmeKnngManifold);
export const sepmeRng = dispatchLayout3("sepmeRng", _sepmeRng2d, _sepmeRngNd, _sepmeRngManifold);
export const sepmeGab = dispatchLayout3("sepmeGab", _sepmeGab2d, _sepmeGabNd, _sepmeGabManifold);
export const sepmeGong = dispatchLayout3("sepmeGong", _sepmeGong2d, _sepmeGongNd, _sepmeGongManifold);
export const neighborhoodHit = dispatchLayout3("neighborhoodHit", _neighborhoodHit2d, _neighborhoodHitNd, _neighborhoodHitManifold);
export const silhouetteScore = dispatchLayout3("silhouetteScore", _silhouetteScore2d, _silhouetteScoreNd, _silhouetteScoreManifold);
export const calinskiHarabaszScore = dispatchLayout3("calinskiHarabaszScore", _calinskiHarabaszScore2d, _calinskiHarabaszScoreNd, _calinskiHarabaszScoreManifold);
export const daviesBouldinScore = dispatchLayout3("daviesBouldinScore", _daviesBouldinScore2d, _daviesBouldinScoreNd, _daviesBouldinScoreManifold);
export const pairwiseDistanceOverlapError = dispatchLayout3("pairwiseDistanceOverlapError", _pairwiseDistanceOverlapError2d, _pairwiseDistanceOverlapErrorNd, _pairwiseDistanceOverlapErrorManifold);
`;
