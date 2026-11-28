export const polyWrapperRuntimeHelpers = `function makeDispatchError(code, message) {
  const err = new TypeError(message);
  err.code = code;
  return err;
}

function attachCoordsView(LayoutClass) {
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

function attachSolverDispatch(
  SolverClass,
  { supports3d = false, supportsNd = false, supportsManifold = false } = {},
) {
  const proto = SolverClass.prototype;
  proto.step = function step(layout) {
    if (layout instanceof _Layout2d) return this._step2d(layout);
    if (supports3d && layout instanceof _Layout3d) return this._step3d(layout);
    if (supportsNd && layout instanceof _LayoutNd) return this._stepNd(layout);
    if (supportsManifold && layout instanceof _ManifoldLayout2d) return this._stepManifold(layout);
    throw makeDispatchError("UnsupportedOperation", \`\${SolverClass.name}.step: unsupported layout type\`);
  };
  proto.runSteps = function runSteps(layout, steps) {
    if (layout instanceof _Layout2d) return this._runSteps2d(layout, steps);
    if (supports3d && layout instanceof _Layout3d) return this._runSteps3d(layout, steps);
    if (supportsNd && layout instanceof _LayoutNd) return this._runStepsNd(layout, steps);
    if (supportsManifold && layout instanceof _ManifoldLayout2d) {
      return this._runStepsManifold(layout, steps);
    }
    throw makeDispatchError("UnsupportedOperation", \`\${SolverClass.name}.runSteps: unsupported layout type\`);
  };
  proto.run = function run(layout) {
    if (layout instanceof _Layout2d) return this._run2d(layout);
    if (supports3d && layout instanceof _Layout3d) return this._run3d(layout);
    if (supportsNd && layout instanceof _LayoutNd) return this._runNd(layout);
    if (supportsManifold && layout instanceof _ManifoldLayout2d) return this._runManifold(layout);
    throw makeDispatchError("UnsupportedOperation", \`\${SolverClass.name}.run: unsupported layout type\`);
  };
  return SolverClass;
}

function attachScheduleWindow(SolverClass) {
  SolverClass.prototype.runScheduleWindow = function runScheduleWindow(layout, startIter, count) {
    if (layout instanceof _Layout2d) return this._runScheduleWindow2d(layout, startIter, count);
    if (layout instanceof _Layout3d) return this._runScheduleWindow3d(layout, startIter, count);
    if (layout instanceof _LayoutNd) return this._runScheduleWindowNd(layout, startIter, count);
    if (layout instanceof _ManifoldLayout2d) {
      return this._runScheduleWindowManifold(layout, startIter, count);
    }
    throw makeDispatchError(
      "UnsupportedOperation",
      \`\${SolverClass.name}.runScheduleWindow: unsupported layout type\`,
    );
  };
  return SolverClass;
}

function attachEdgePolylines(PolyClass) {
  attachTypedView(PolyClass, "edgeOffsetsView", "edgeOffsetsPtr", "edgeOffsetsLen", Uint32Array, "_edgeOffsetsView");
  attachTypedView(PolyClass, "pointsView", "pointsPtr", "pointsLen", Float32Array, "_pointsView");
  PolyClass.straightLineFromGraph = function straightLineFromGraph(graph, layout) {
    if (layout instanceof _Layout2d) return PolyClass._straightLineFromGraph2d(graph, layout);
    if (layout instanceof _Layout3d) return PolyClass._straightLineFromGraph3d(graph, layout);
    if (layout instanceof _LayoutNd) return PolyClass._straightLineFromGraphNd(graph, layout);
    throw makeDispatchError(
      "UnsupportedOperation",
      "EdgePolylines.straightLineFromGraph: expected Layout2d, Layout3d, or LayoutNd",
    );
  };
  PolyClass.fromParts = function fromParts(edgeOffsets, points) {
    if (points instanceof _Layout2d) return PolyClass._fromParts2d(edgeOffsets, points);
    if (points instanceof _Layout3d) return PolyClass._fromParts3d(edgeOffsets, points);
    if (points instanceof _LayoutNd) return PolyClass._fromPartsNd(edgeOffsets, points);
    throw makeDispatchError(
      "UnsupportedOperation",
      "EdgePolylines.fromParts: expected points to be Layout2d, Layout3d, or LayoutNd",
    );
  };
  return PolyClass;
}

function dispatchLayout4(name, fn2d, fn3d, fnNd, fnManifold) {
  return function (layout, ...args) {
    if (layout instanceof _Layout2d) return fn2d(layout, ...args);
    if (fn3d && layout instanceof _Layout3d) return fn3d(layout, ...args);
    if (fnNd && layout instanceof _LayoutNd) return fnNd(layout, ...args);
    if (fnManifold && layout instanceof _ManifoldLayout2d) return fnManifold(layout, ...args);
    throw makeDispatchError("UnsupportedOperation", \`\${name}: unsupported layout type\`);
  };
}

function dispatchGraphLayout4(name, fn2d, fn3d, fnNd, fnManifold) {
  return function (graph, layout, ...args) {
    if (layout instanceof _Layout2d) return fn2d(graph, layout, ...args);
    if (fn3d && layout instanceof _Layout3d) return fn3d(graph, layout, ...args);
    if (fnNd && layout instanceof _LayoutNd) return fnNd(graph, layout, ...args);
    if (fnManifold && layout instanceof _ManifoldLayout2d) return fnManifold(graph, layout, ...args);
    throw makeDispatchError("UnsupportedOperation", \`\${name}: unsupported layout type\`);
  };
}
`;

export const polyExportDefinitions = `export const Layout2d = attachCoordsView(_Layout2d);
export const Layout3d = attachCoordsView(_Layout3d);
export const LayoutNd = attachCoordsView(_LayoutNd);
export const ManifoldLayout2d = attachCoordsView(_ManifoldLayout2d);
export const EdgePolylines = attachEdgePolylines(_EdgePolylines);

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

const allTargetFlags = { supports3d: true, supportsNd: true, supportsManifold: true };
const euclideanTargetFlags = { supports3d: true, supportsNd: true };

export const SgdSolver = attachScheduleWindow(attachSolverDispatch(_SgdSolver, allTargetFlags));
export const Sgd2Solver = attachScheduleWindow(attachSolverDispatch(_Sgd2Solver, allTargetFlags));
export const PacmapGraphSolver = attachSolverDispatch(_PacmapGraphSolver, allTargetFlags);
export const LargeVisSolver = attachSolverDispatch(_LargeVisSolver, allTargetFlags);
export const DrGraphSolver = attachSolverDispatch(_DrGraphSolver, allTargetFlags);
export const GraphCneSolver = attachSolverDispatch(_GraphCneSolver, allTargetFlags);
export const SnapTFdpSolver = attachSolverDispatch(_SnapTFdpSolver, allTargetFlags);

export const TsNetSolver = attachSolverDispatch(_TsNetSolver, allTargetFlags);
export const GuMapSolver = attachSolverDispatch(_GuMapSolver, allTargetFlags);
export const FruchtermanReingoldSolver = attachSolverDispatch(_FruchtermanReingoldSolver, allTargetFlags);
export const KamadaKawaiSolver = attachSolverDispatch(_KamadaKawaiSolver, allTargetFlags);

export const StressMajorizationSolver = attachSolverDispatch(_StressMajorizationSolver, allTargetFlags);
StressMajorizationSolver.prototype.computeStress = function computeStress(layout) {
  if (layout instanceof _Layout2d) return this._computeStress2d(layout);
  if (layout instanceof _Layout3d) return this._computeStress3d(layout);
  if (layout instanceof _LayoutNd) return this._computeStressNd(layout);
  if (layout instanceof _ManifoldLayout2d) return this._computeStressManifold(layout);
  throw makeDispatchError("UnsupportedOperation", "StressMajorizationSolver.computeStress: unsupported layout type");
};

export const LinLogSolver = attachSolverDispatch(_LinLogSolver, euclideanTargetFlags);
LinLogSolver.prototype.energy = function energy(layout) {
  if (layout instanceof _Layout2d) return this._energy2d(layout);
  if (layout instanceof _Layout3d) return this._energy3d(layout);
  if (layout instanceof _LayoutNd) return this._energyNd(layout);
  throw makeDispatchError("UnsupportedOperation", "LinLogSolver.energy: expected Layout2d, Layout3d, or LayoutNd");
};

export const ForceAtlas2Solver = attachSolverDispatch(_ForceAtlas2Solver, euclideanTargetFlags);
export const YifanHuSolver = attachSolverDispatch(_YifanHuSolver, euclideanTargetFlags);
export const MaxentStressSolver = attachSolverDispatch(_MaxentStressSolver, euclideanTargetFlags);
export const TFdpSolver = attachSolverDispatch(_TFdpSolver, euclideanTargetFlags);

export const ArenaOctree3d = _ArenaOctree3d;
ArenaOctree3d.prototype.build = function build(coordsOrLayout) {
  if (coordsOrLayout instanceof _Layout3d) return this.build3d(coordsOrLayout);
  if (coordsOrLayout instanceof _LayoutNd) return this.buildNd(coordsOrLayout);
  return this._buildFlat(coordsOrLayout);
};
ArenaOctree3d.prototype.buildWeighted = function buildWeighted(coordsOrLayout, masses) {
  if (coordsOrLayout instanceof _Layout3d) return this.buildWeighted3d(coordsOrLayout, masses);
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
  if (layout instanceof _Layout3d) return _evaluateMetrics3d(graph, layout, targets, ...args);
  if (layout instanceof _LayoutNd) return _evaluateMetricsNd(graph, layout, targets, ...args);
  if (layout instanceof _ManifoldLayout2d) return _evaluateMetricsManifold(graph, layout, targets, ...args);
  if (layout instanceof _Layout2d) {
    const opt = args.length === 1 && args[0] && typeof args[0] === "object" && !Array.isArray(args[0]) ? args[0] : null;
    if (opt && opt.originalLayout != null) {
      return _evaluateMetrics2dWithOriginal(graph, layout, targets, opt, opt.originalLayout);
    }
    return _evaluateMetrics2d(graph, layout, targets, ...args);
  }
  throw makeDispatchError("UnsupportedOperation", "evaluateMetrics: unsupported layout type");
}

export const scaleNormalizedStress = dispatchGraphLayout4("scaleNormalizedStress", _scaleNormalizedStress2d, _scaleNormalizedStress3d, _scaleNormalizedStressNd, _scaleNormalizedStressManifold);
export const scaleNormalizedStressDetail = dispatchGraphLayout4("scaleNormalizedStressDetail", _scaleNormalizedStressDetail2d, _scaleNormalizedStressDetail3d, _scaleNormalizedStressDetailNd, _scaleNormalizedStressDetailManifold);
export const sampledScaleNormalizedStress = dispatchGraphLayout4("sampledScaleNormalizedStress", _sampledScaleNormalizedStress2d, _sampledScaleNormalizedStress3d, _sampledScaleNormalizedStressNd, _sampledScaleNormalizedStressManifold);
export const scaleNormalizedStressFromDistances = dispatchLayout4("scaleNormalizedStressFromDistances", _scaleNormalizedStressFromDistances2d, _scaleNormalizedStressFromDistances3d, _scaleNormalizedStressFromDistancesNd, _scaleNormalizedStressFromDistancesManifold);
export const neighborhoodPreservation1Hop = dispatchGraphLayout4("neighborhoodPreservation1Hop", _neighborhoodPreservation1Hop2d, _neighborhoodPreservation1Hop3d, _neighborhoodPreservation1HopNd, _neighborhoodPreservation1HopManifold);
export const neighborhoodPreservation2Hop = dispatchGraphLayout4("neighborhoodPreservation2Hop", _neighborhoodPreservation2Hop2d, _neighborhoodPreservation2Hop3d, _neighborhoodPreservation2HopNd, _neighborhoodPreservation2HopManifold);
export const sampledNp1 = dispatchGraphLayout4("sampledNp1", _sampledNp12d, _sampledNp13d, _sampledNp1Nd, _sampledNp1Manifold);
export const sampledNp2 = dispatchGraphLayout4("sampledNp2", _sampledNp22d, _sampledNp23d, _sampledNp2Nd, _sampledNp2Manifold);
export const gabrielGraphProperty = dispatchGraphLayout4("gabrielGraphProperty", _gabrielGraphProperty2d, _gabrielGraphProperty3d, _gabrielGraphPropertyNd, _gabrielGraphPropertyManifold);
export const nodeResolution = dispatchLayout4("nodeResolution", _nodeResolution2d, _nodeResolution3d, _nodeResolutionNd, _nodeResolutionManifold);
export const drngPreservation = dispatchGraphLayout4("drngPreservation", _drngPreservation2d, _drngPreservation3d, _drngPreservationNd, _drngPreservationManifold);
export const dggPreservation = dispatchGraphLayout4("dggPreservation", _dggPreservation2d, _dggPreservation3d, _dggPreservationNd, _dggPreservationManifold);
export const idealEdgeLengths = dispatchGraphLayout4("idealEdgeLengths", _idealEdgeLengths2d, _idealEdgeLengths3d, _idealEdgeLengthsNd, _idealEdgeLengthsManifold);
export const idealEdgeLengthsWithUnit = dispatchGraphLayout4("idealEdgeLengthsWithUnit", _idealEdgeLengthsWithUnit2d, _idealEdgeLengthsWithUnit3d, _idealEdgeLengthsWithUnitNd, _idealEdgeLengthsWithUnitManifold);
export const layoutDegeneracy = dispatchGraphLayout4("layoutDegeneracy", _layoutDegeneracy2d, _layoutDegeneracy3d, _layoutDegeneracyNd, _layoutDegeneracyManifold);
export const gabrielGraphEdges = dispatchLayout4("gabrielGraphEdges", _gabrielGraphEdges2d, _gabrielGraphEdges3d, _gabrielGraphEdgesNd, _gabrielGraphEdgesManifold);
export const relativeNeighborhoodEdges = dispatchLayout4("relativeNeighborhoodEdges", _relativeNeighborhoodEdges2d, _relativeNeighborhoodEdges3d, _relativeNeighborhoodEdgesNd, _relativeNeighborhoodEdgesManifold);
export const landmarkStratifiedRta = dispatchGraphLayout4("landmarkStratifiedRta", _landmarkStratifiedRta2d, _landmarkStratifiedRta3d, _landmarkStratifiedRtaNd, _landmarkStratifiedRtaManifold);
export const landmarkStratifiedRtaDetail = dispatchGraphLayout4("landmarkStratifiedRtaDetail", _landmarkStratifiedRtaDetail2d, _landmarkStratifiedRtaDetail3d, _landmarkStratifiedRtaDetailNd, _landmarkStratifiedRtaDetailManifold);
export const landmarkStratifiedRtaFromLandmarks = dispatchLayout4("landmarkStratifiedRtaFromLandmarks", _landmarkStratifiedRtaFromLandmarks2d, _landmarkStratifiedRtaFromLandmarks3d, _landmarkStratifiedRtaFromLandmarksNd, _landmarkStratifiedRtaFromLandmarksManifold);
export const torusCrossingNumber = dispatchGraphLayout4("torusCrossingNumber", _torusCrossingNumber2d, null, null, _torusCrossingNumberManifold);
export const crossingEdgesTorus = dispatchGraphLayout4("crossingEdgesTorus", _crossingEdgesTorus2d, null, null, _crossingEdgesTorusManifold);
export const torusLayoutEdgeSegments = dispatchGraphLayout4("torusLayoutEdgeSegments", _torusLayoutEdgeSegments2d, null, null, _torusLayoutEdgeSegmentsManifold);
export const cqKmeans = dispatchLayout4("cqKmeans", _cqKmeans2d, _cqKmeans3d, _cqKmeansNd, _cqKmeansManifold);
export const cqHac = dispatchLayout4("cqHac", _cqHac2d, _cqHac3d, _cqHacNd, _cqHacManifold);
export const distanceConsistency = dispatchLayout4("distanceConsistency", _distanceConsistency2d, _distanceConsistency3d, _distanceConsistencyNd, _distanceConsistencyManifold);
export const distributionConsistency = dispatchLayout4("distributionConsistency", _distributionConsistency2d, _distributionConsistency3d, _distributionConsistencyNd, _distributionConsistencyManifold);
export const sepmeKnng = dispatchLayout4("sepmeKnng", _sepmeKnng2d, _sepmeKnng3d, _sepmeKnngNd, _sepmeKnngManifold);
export const sepmeRng = dispatchLayout4("sepmeRng", _sepmeRng2d, _sepmeRng3d, _sepmeRngNd, _sepmeRngManifold);
export const sepmeGab = dispatchLayout4("sepmeGab", _sepmeGab2d, _sepmeGab3d, _sepmeGabNd, _sepmeGabManifold);
export const sepmeGong = dispatchLayout4("sepmeGong", _sepmeGong2d, _sepmeGong3d, _sepmeGongNd, _sepmeGongManifold);
export const neighborhoodHit = dispatchLayout4("neighborhoodHit", _neighborhoodHit2d, _neighborhoodHit3d, _neighborhoodHitNd, _neighborhoodHitManifold);
export const silhouetteScore = dispatchLayout4("silhouetteScore", _silhouetteScore2d, _silhouetteScore3d, _silhouetteScoreNd, _silhouetteScoreManifold);
export const calinskiHarabaszScore = dispatchLayout4("calinskiHarabaszScore", _calinskiHarabaszScore2d, _calinskiHarabaszScore3d, _calinskiHarabaszScoreNd, _calinskiHarabaszScoreManifold);
export const daviesBouldinScore = dispatchLayout4("daviesBouldinScore", _daviesBouldinScore2d, _daviesBouldinScore3d, _daviesBouldinScoreNd, _daviesBouldinScoreManifold);
export const pairwiseDistanceOverlapError = dispatchLayout4("pairwiseDistanceOverlapError", _pairwiseDistanceOverlapError2d, _pairwiseDistanceOverlapError3d, _pairwiseDistanceOverlapErrorNd, _pairwiseDistanceOverlapErrorManifold);
`;
