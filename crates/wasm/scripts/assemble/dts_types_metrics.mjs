export const csrGraphOptionsDecl = `/**
 * Options controlling CSR / edge-list graph construction.
 */
export interface CsrGraphOptions {
    directed?: boolean;
    sanitize?: boolean;
    weightDtype?: "f32" | "f64" | string;
}

`;

export const metricsTypeDecls = `export type AnyLayout = Layout2d | LayoutNd | ManifoldLayout2d;

export interface KdNeighbor {
    index: number;
    dist: number;
}

export interface CrossingPairRecord {
    edgeA: number;
    edgeB: number;
    u1: number;
    v1: number;
    u2: number;
    v2: number;
    angleRad?: number;
    cosSq: number;
}

export interface TorusCrossingRecord extends CrossingPairRecord {
    segA: Float64Array;
    segB: Float64Array;
}

/**
 * Canonical target identifiers and normalized aliases accepted by \`evaluateMetrics\`.
 */
export type MetricTarget =
    | "scaleNormalizedStress"
    | "sampledScaleNormalizedStress"
    | "geodesicDistortion"
    | "crossingNumber"
    | "crossingEdges"
    | "crossingAngle"
    | "sampledCrossingNumber"
    | "crosslessness"
    | "sampledCrosslessness"
    | "torusCrossingNumber"
    | "crossingEdgesTorus"
    | "neighborhoodPreservation1Hop"
    | "neighborhoodPreservation2Hop"
    | "sampledNp1"
    | "sampledNp2"
    | "sampled_np1"
    | "sampled_np2"
    | "sampledNeighborhoodPreservation1Hop"
    | "sampledNeighborhoodPreservation2Hop"
    | "gabrielGraphProperty"
    | "nodeResolution"
    | "relativeNeighborhoodPreservation"
    | "gabrielGraphPreservation"
    | "aspectRatio"
    | "angularResolution"
    | "minimumAngle"
    | "minimumAngleA1"
    | "minimum_angle"
    | "minimum_angle_a1"
    | "layoutDegeneracy"
    | "layout_degeneracy"
    | "idealEdgeLengths"
    | "prismDistortion"
    | "adjustedPatchNig"
    | "adjusted_patch_nig"
    | "clusterAreaFaithfulness"
    | "cluster_area_faithfulness"
    | "cluster-area-faithfulness"
    | "clusterArea"
    | "cluster_area"
    | "gcaf"
    | "landmarkStratifiedRta"
    | "landmark_stratified_rta"
    | "rta"
    | "sns"
    | "sampled_sns"
    | "np1"
    | "np2"
    | "drng"
    | "dgg"
    | "distanceConsistency"
    | "dsc"
    | "distributionConsistency"
    | "dc"
    | "classDensityMeasure"
    | "cdm"
    | "histogramDensityMeasure"
    | "hdm"
    | "2d-hdm"
    | "sepmeKnng"
    | "sepme_knng"
    | "sepme-knng"
    | "sepmeKnng2"
    | "sepme_knng2"
    | "sepme-knng2"
    | "sepmeRng"
    | "sepme_rng"
    | "sepmeGab"
    | "sepme_gab"
    | "sepmeGong"
    | "sepme_gong"
    | "neighborhoodHit"
    | "neighborhood_hit"
    | "m_nh"
    | "silhouetteScore"
    | "silhouette"
    | "calinskiHarabaszScore"
    | "calinskiHarabasz"
    | "calinski_harabasz"
    | "ch"
    | "daviesBouldinScore"
    | "daviesBouldin"
    | "davies_bouldin"
    | "db"
    | "cqKmeansAri"
    | "cq_kmeans_ari"
    | "cqKmeansAmi"
    | "cq_kmeans_ami"
    | "cqHacHom"
    | "cq_hac_hom"
    | "cqHacAmi"
    | "cq_hac_ami"
    | "pairwiseDistanceOverlapError"
    | "pairwise_distance_overlap_error"
    | "pdoe"
    | "sprawlterNn"
    | "sprawlter_nn"
    | "norMetrics"
    | "nor";

/**
 * Plain JavaScript record returned by \`MetricReport.toRecord()\`.
 */
export interface MetricReportRecord {
    scaleNormalizedStress?: number;
    sampledScaleNormalizedStress?: number;
    geodesicDistortion?: number;
    crossingNumber?: number;
    crossingEdges?: CrossingPairRecord[];
    crossingAngle?: number;
    sampledCrossingNumber?: number;
    crosslessness?: number;
    sampledCrosslessness?: number;
    torusCrossingNumber?: number;
    crossingEdgesTorus?: TorusCrossingRecord[];
    neighborhoodPreservation1Hop?: number;
    neighborhoodPreservation2Hop?: number;
    sampledNp1?: number;
    sampledNp2?: number;
    gabrielGraphProperty?: number;
    nodeResolution?: number;
    relativeNeighborhoodPreservation?: number;
    gabrielGraphPreservation?: number;
    aspectRatio?: number;
    angularResolution?: number;
    minimumAngle?: number;
    minimumAngleA1?: number;
    layoutDegeneracy?: number;
    idealEdgeLengths?: number;
    prismSigmaDist?: number;
    prismSigmaDisp?: number;
    adjustedPatchNig?: number;
    clusterAreaFaithfulness?: number;
    landmarkStratifiedRta?: number;
    distanceConsistency?: number;
    distributionConsistency?: number;
    classDensityMeasure?: number;
    histogramDensityMeasure?: number;
    sepmeKnng?: number;
    sepmeRng?: number;
    sepmeGab?: number;
    sepmeGong?: number;
    neighborhoodHit?: number;
    silhouetteScore?: number;
    calinskiHarabaszScore?: number;
    daviesBouldinScore?: number;
    cqKmeansAri?: number;
    cqKmeansAmi?: number;
    cqHacHom?: number;
    cqHacAmi?: number;
    pairwiseDistanceOverlapError?: number;
    sprawlterNn?: number;
    norOoNni?: number;
    norSpChA?: number;
    norGsBbIar?: number;
    norNmDmImse?: number;
    norElRsd?: number;
}

/**
 * Configuration parameters for cluster-separation, CVI, and Sprawlter targets in \`evaluateMetrics\`.
 */
export interface ClusterBaselineOptions {
    sepmeKnngK?: number | null;
    neighborhoodHitK?: number | null;
    dcGridSize?: number | null;
    cdmGridSize?: number | null;
    cdmBandwidth?: number | null;
    numBins?: number | null;
    gongGamma?: number | null;
    kmeansNInit?: number | null;
    kmeansMaxIterations?: number | null;
    seed?: number | null;
    sprawlterNodeRadius?: number | null;
}

/**
 * Unified options object accepted as the 4th argument to \`evaluateMetrics\`.
 */
export interface MetricConfigOptions extends ClusterBaselineOptions {
    numPivots?: number | null;
    numSamples?: number | null;
    seed?: number | null;
    idealEdgeLength?: number | null;
    labels?: Uint32Array | number[] | null;
    originalLayout?: Layout2d | null;
    sprawlterHalfWidths?: Float32Array | number[] | null;
    sprawlterHalfHeights?: Float32Array | number[] | null;
    numLandmarks?: number | null;
    numTriplets?: number | null;
    minDegree?: number | null;
    patchRadius?: number | null;
    canvasSize?: number | null;
    padFrac?: number | null;
    gMax?: number | null;
    sigmaCells?: number | null;
    kappa?: number | null;
    tauFrac?: number | null;
    boundaryR95?: number | null;
    stevensGamma?: number | null;
    nodeWeights?: Float64Array | Float32Array | number[] | null;
    clusterWeights?: Float64Array | Float32Array | number[] | null;
    clusterBaselines?: ClusterBaselineOptions | null;
}

/**
 * Plain JavaScript record returned by \`StressEvaluation.toRecord()\`.
 */
export interface StressEvaluationRecord {
    normalizedStress: number;
    rawStress: number;
    alphaStar: number;
    pairCount: number;
    weightDistanceSqSum: number;
    stress: number;
    scale: number;
}

/**
 * Plain JavaScript record returned by \`ClusterAreaReport.toRecord()\`.
 */
export interface ClusterAreaReportRecord {
    caf: number;
    purityAdj: number;
    propVis: number;
    resDensity: number;
    resDeconv: number;
    resCombined: number;
    spatialCoherence: number;
    inFrac: number;
    effectiveBoundaryR95: number;
    gEff: number;
    numClusters: number;
    numNodes: number;
    uniqueLabels: Uint32Array;
    counts: Uint32Array;
    p: Float64Array;
    q: Float64Array;
    purity: Float64Array;
    resolution: Float64Array;
    resDensityPerCluster: Float64Array;
    resDeconvPerCluster: Float64Array;
    cafPerCluster: Float64Array;
    coherence: Float64Array;
}

`;
