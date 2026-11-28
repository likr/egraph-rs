export const csrGraphOptionsDecl = `/**
 * Strictly single-precision 32-bit float edge weights ("float32").
 */
export type WeightDtype = "float32";

/**
 * Structured error codes attached to \`Error.code\` across WASM API failures.
 */
export type EgraphErrorCode =
    | "InvalidInput"
    | "DimensionMismatch"
    | "IndexOutOfBounds"
    | "InvalidCsr"
    | "NonFiniteValue"
    | "UnsupportedOperation"
    | "NumericalBreakdown";

/**
 * Canonical target coordinate storage kind accepted by solvers and generators.
 */
export type TargetKind = "layout2d" | "layout3d" | "layoutNd" | "manifold2d";

/**
 * Options controlling CSR / edge-list graph construction.
 */
export interface CsrGraphOptions {
    directed?: boolean;
    sanitize?: boolean;
    weightDtype?: WeightDtype;
}

`;

export const metricsTypeDecls = `export type AnyLayout = Layout2d | Layout3d | LayoutNd | ManifoldLayout2d;

/**
 * Core linear-memory layout storage interface shared by \`Layout2d\`, \`Layout3d\`,
 * \`LayoutNd\`, and \`ManifoldLayout2d\`.
 */
export interface LayoutStorage {
    readonly numNodes: number;
    len(): number;
    isEmpty(): boolean;
    isAllFinite(): boolean;
    coordsPtr(): number;
    coordsLen(): number;
    coordsView(): Float32Array;
}

/**
 * Cooperative iterative layout solver interface supporting step-by-step execution.
 */
export interface IterativeSolver<TTarget = AnyLayout> {
    readonly status: SolverStatus | string;
    readonly iteration: number;
    readonly totalIterations: number | undefined;
    readonly isFinished: boolean;
    readonly numNodes: number;
    step(target: TTarget): string;
    runSteps(target: TTarget, steps: number): string;
    run(target: TTarget): string;
}

/**
 * One-shot deterministic layout generator interface.
 */
export interface LayoutGenerator<TOutput = Layout2d> {
    generate(graph?: CsrGraph): TOutput;
}

/**
 * In-place post-processing layout transform interface.
 */
export interface LayoutTransform<TTarget = Layout2d> {
    apply(target: TTarget): void;
}

export interface LayoutInitOptions {
    seed?: number | bigint | null;
    scale?: number | null;
}

export interface GenerateWeightedGraphOptions {
    avgDegree?: number | null;
    minWeight?: number | null;
    maxWeight?: number | null;
    seed?: number | bigint | null;
}

export interface GenerateLayeredDagOptions {
    edgeProb?: number | null;
    seed?: number | bigint | null;
}

export interface GenerateSbmGraphOptions {
    pIntra?: number | null;
    pInter?: number | null;
    seed?: number | bigint | null;
}

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
 * Canonical camelCase target identifiers accepted by \`evaluateMetrics\`.
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
    | "layoutDegeneracy"
    | "idealEdgeLengths"
    | "prismDistortion"
    | "adjustedPatchNig"
    | "clusterAreaFaithfulness"
    | "clusterArea"
    | "gcaf"
    | "landmarkStratifiedRta"
    | "rta"
    | "sns"
    | "sampledSns"
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
    | "sepmeKnng"
    | "sepmeKnng2"
    | "sepmeRng"
    | "sepmeGab"
    | "sepmeGong"
    | "neighborhoodHit"
    | "mnh"
    | "silhouetteScore"
    | "silhouette"
    | "calinskiHarabaszScore"
    | "calinskiHarabasz"
    | "ch"
    | "daviesBouldinScore"
    | "daviesBouldin"
    | "db"
    | "cqKmeansAri"
    | "cqKmeansAmi"
    | "cqHacHom"
    | "cqHacAmi"
    | "pairwiseDistanceOverlapError"
    | "pdoe"
    | "sprawlterNn"
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

export interface SamplingSeedOptions {
    numSamples?: number | null;
    seed?: number | bigint | null;
}

export interface SampledStressOptions {
    numPivots?: number | null;
    seed?: number | bigint | null;
}

export interface RtaOptions {
    numLandmarks?: number | null;
    numTriplets?: number | null;
    minDegree?: number | null;
    seed?: number | bigint | null;
}

export interface RtaFromLandmarksOptions {
    numTriplets?: number | null;
    seed?: number | bigint | null;
}

export interface AdjustedPatchNigOptions {
    patchRadius?: number | null;
    canvasSize?: number | null;
    padFrac?: number | null;
    seed?: number | bigint | null;
}

export interface AdjustedPatchNigMultiscaleOptions {
    radii?: Uint32Array | number[] | null;
    canvasSize?: number | null;
    padFrac?: number | null;
    seed?: number | bigint | null;
}

export interface ClusterAreaFaithfulnessOptions {
    gMax?: number | null;
    sigmaCells?: number | null;
    kappa?: number | null;
    tauFrac?: number | null;
    boundaryR95?: number | null;
    stevensGamma?: number | null;
    nodeWeights?: Float64Array | Float32Array | number[] | null;
    clusterWeights?: Float64Array | Float32Array | number[] | null;
}

export interface CqKmeansOptions {
    nInit?: number | null;
    maxIterations?: number | null;
    seed?: number | bigint | null;
}

export interface ClassDensityOptions {
    gridSize?: number | null;
    bandwidth?: number | null;
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
    seed?: number | bigint | null;
    sprawlterNodeRadius?: number | null;
}

/**
 * Unified options object accepted as the 4th argument to \`evaluateMetrics\`.
 */
export interface MetricConfigOptions extends ClusterBaselineOptions {
    numPivots?: number | null;
    numSamples?: number | null;
    seed?: number | bigint | null;
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

export interface DatasetInfoRecord {
    name: string;
    numNodes: number;
    numEdges: number;
    source: string;
    description: string;
}

export interface SbmGraphResult {
    graph: CsrGraph;
    labels: Uint32Array;
}

export interface ManifoldMetricRecord {
    scale: number;
    aspect: number;
}

export type SpectralFilterTerm = [number, number];

export type SpectralFilterTermsInput =
    | readonly (readonly [number, number])[]
    | Float64Array
    | Float32Array
    | readonly number[];

export interface PsdSpectralFilterProjection {
    termsPsd: SpectralFilterTerm[];
    deltaPsd: number;
    hMin: number;
}

export interface SymmetricNormalizedLaplacianResult {
    matrix: CsrMatrix;
    invSqrtDeg: Float64Array;
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
