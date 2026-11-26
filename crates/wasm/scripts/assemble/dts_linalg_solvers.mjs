// Refine linear-algebra and neighborhood TypeScript signatures (camelCase parameter names, optional targetStd?: number | null).
export const linalgNeighborhoodDtsReplacements = [
  [
    /static fromCsr\(num_nodes: number, indptr: Uint32Array \| number\[\], indices: Uint32Array \| number\[\], distances: Float64Array \| number\[\]\): KnnGraph;/,
    "static fromCsr(numNodes: number, indptr: Uint32Array | number[], indices: Uint32Array | number[], distances: Float64Array | number[]): KnnGraph;",
  ],
  [
    /export function selectMaxminSpPivots\([^;]+\): Uint32Array;/,
    "export function selectMaxminSpPivots(graph: CsrGraph, numPivots?: number | null, seed?: number | null, minDegree?: number | null): Uint32Array;",
  ],
  [
    /export function pivotMdsInit\([^;]+\): Layout2d;/,
    "export function pivotMdsInit(graph: CsrGraph, numPivots?: number | null, seed?: number | null, minDegree?: number | null, targetStd?: number | null): Layout2d;",
  ],
  [
    /export function korenSpectralInit\([^;]+\): Layout2d;/,
    "export function korenSpectralInit(graph: CsrGraph, targetStd?: number | null, tol?: number | null, maxIters?: number | null, seed?: number | null): Layout2d;",
  ],
  [
    /export function smallestLaplacianEigenpairs\([^;]+\): EigenResult;/,
    "export function smallestLaplacianEigenpairs(graph: CsrGraph, nModes?: number | null, tol?: number | null, maxIters?: number | null, seed?: number | null): EigenResult;",
  ],
  [
    /export function chebyshevHeatSpmv\([^;]+\): Float64Array;/,
    "export function chebyshevHeatSpmv(graph: CsrGraph, vectors: Float64Array | number[], numCols: number, t: number, degree?: number | null, lambdaMax?: number | null): Float64Array;",
  ],
  [
    /export function chebyshevHeatSpmvMatrix\([^;]+\): Float64Array;/,
    "export function chebyshevHeatSpmvMatrix(matrix: CsrMatrix, vectors: Float64Array | number[], numCols: number, t: number, degree?: number | null, lambdaMax?: number | null, lanczosSteps?: number | null, safetyFactor?: number | null, seed?: number | null): Float64Array;",
  ],
  [
    /export function chebyshevDogJlEmbedding\([^;]+\): Float64Array;/,
    "export function chebyshevDogJlEmbedding(graph: CsrGraph, jlDim?: number | null, degree?: number | null, seed?: number | null, degreeNormalize?: boolean | null, epsPsd?: number | null, t1?: number | null, t2?: number | null, alpha?: number | null): Float64Array;",
  ],
  [
    /export function chebyshevDogJlEmbeddingMatrix\([^;]+\): Float64Array;/,
    "export function chebyshevDogJlEmbeddingMatrix(matrix: CsrMatrix, jlDim?: number | null, degree?: number | null, seed?: number | null, epsPsd?: number | null, t1?: number | null, t2?: number | null, alpha?: number | null, lambdaMax?: number | null, lanczosSteps?: number | null, safetyFactor?: number | null, scaleVec?: Float64Array | number[] | null): Float64Array;",
  ],
  [
    /export function anisotropicLaplacian\([^;]+\): AnisotropicLaplacianResult;/,
    "export function anisotropicLaplacian(graph: CsrGraph, alpha?: number | null, beta?: number | null, degreeReg?: number | null, degreeCap?: number | null, reweightedDegreeNorm?: boolean | null): AnisotropicLaplacianResult;",
  ],
  [
    /(export class CsrMatrix \{[\s\S]*?)static fromCsr\([^;]+\): CsrMatrix;/,
    "$1static fromCsr(numNodes: number, indptr: Uint32Array | number[], indices: Uint32Array | number[], values: Float64Array | number[]): CsrMatrix;",
  ],
  [
    /export function extractSparse1hop2hopPairs\([^;]+\): Sparse12HopPairs;/,
    "export function extractSparse1hop2hopPairs(graph: CsrGraph, max2hopPerNode?: number | null, seed?: number | null): Sparse12HopPairs;",
  ],
  [
    /export function buildCapped12hopKnn\([^;]+\): TwoHopKnnResult;/,
    "export function buildCapped12hopKnn(graph: CsrGraph, max2hopPerNode?: number | null, seed?: number | null): TwoHopKnnResult;",
  ],
  [
    /export function robustNormalizeLayout2d\([^;]+\): void;/,
    "export function robustNormalizeLayout2d(layout: Layout2d, targetStd?: number | null, quantile?: number | null, seed?: number | null): void;",
  ],
  [
    /export function graph12hopRowNormalizedProbabilities\([^;]+\): SparseProbabilities;/,
    "export function graph12hopRowNormalizedProbabilities(graph: CsrGraph, w2?: number | null, max2hopPerNode?: number | null, degreeBeta?: number | null, seed?: number | null): SparseProbabilities;",
  ],
  [
    /export function computeNodewise2hopPerplexities\([^;]+\): Float64Array;/,
    "export function computeNodewise2hopPerplexities(graph: CsrGraph, c?: number | null, minPerplexity?: number | null, maxPerplexity?: number | null): Float64Array;",
  ],
  [
    /export function computeMedian2hopPerplexity\([^;]+\): number;/,
    "export function computeMedian2hopPerplexity(graph: CsrGraph, c?: number | null, minPerplexity?: number | null, maxPerplexity?: number | null): number;",
  ],
  [
    /export function buildHybridAdaptiveKnn\([^;]+\): HybridKnnResult;/,
    "export function buildHybridAdaptiveKnn(graph: CsrGraph, coords: Float64Array | number[], dim: number, perplexities: Float64Array | number[], kFactor?: number | null, marginGap?: number | null, coarseDim?: number | null, rerankFactor?: number | null, k?: number | null, maxRowK?: number | null): HybridKnnResult;",
  ],
  [
    /export function computeKnnProbabilities\([^;]+\): SparseProbabilities;/,
    "export function computeKnnProbabilities(knn: KnnGraph, perplexities: Float64Array | number[], power?: number | null, sigmaIters?: number | null, sigmaTolerance?: number | null, rowNormalize?: boolean | null): SparseProbabilities;",
  ],
  [
    /export function computeFuzzySimplicialSet\([^;]+\): SparseProbabilities;/,
    "export function computeFuzzySimplicialSet(knn: KnnGraph, sigmaIters?: number | null, sigmaTolerance?: number | null): SparseProbabilities;",
  ],
  [
    /export function findAbParams\([^;]+\): Float64Array;/,
    "export function findAbParams(minDist?: number | null, spread?: number | null): Float64Array;",
  ],
  [
    /export function computeHorvitzThompsonGamma\([^;]+\): number;/,
    "export function computeHorvitzThompsonGamma(gammaNon2: number, numNodes: number, negPerNode: number): number;",
  ],
  [
    /export function computeRegularizedDogVaradhanLandmarks\([^;]+\): DogVaradhanLandmarks;/,
    "export function computeRegularizedDogVaradhanLandmarks(graph: CsrGraph, degree?: number | null, numLandmarks?: number | null, seed?: number | null, degreeNormalize?: boolean | null, pExp?: number | null, epsPsd?: number | null, tCoarse?: number | null, t1?: number | null, t2?: number | null, alpha?: number | null, betaFloor?: number | null): DogVaradhanLandmarks;",
  ],
  [
    /export function computeRegularizedDogVaradhanLandmarksFromMatrix\([^;]+\): DogVaradhanLandmarks;/,
    "export function computeRegularizedDogVaradhanLandmarksFromMatrix(graph: CsrGraph, matrix: CsrMatrix, lambdaMax?: number | null, degree?: number | null, numLandmarks?: number | null, minDegree?: number | null, jlDim?: number | null, seed?: number | null, degreeNormalize?: boolean | null, pExp?: number | null, epsPsd?: number | null, tCoarse?: number | null, t1?: number | null, t2?: number | null, alpha?: number | null, betaFloor?: number | null): DogVaradhanLandmarks;",
  ],
  [
    /(export class AmgPcgSolver \{[\s\S]*?)static fromCsr\([^;]+\): AmgPcgSolver;/,
    "$1static fromCsr(numNodes: number, indptr: Uint32Array | number[], indices: Uint32Array | number[], values: Float64Array | number[], options?: AmgPcgOptions | null): AmgPcgSolver;",
  ],
  [
    /(export class AmgPcgSolver \{[\s\S]*?)static fromCsrMatrix\([^;]+\): AmgPcgSolver;/,
    "$1static fromCsrMatrix(matrix: CsrMatrix, options?: AmgPcgOptions | null): AmgPcgSolver;",
  ],
  [
    /(export class AmgPcgSolver \{[\s\S]*?)static fromGraphRegularizedLaplacian\([^;]+\): AmgPcgSolver;/,
    "$1static fromGraphRegularizedLaplacian(graph: CsrGraph, gamma?: number | null, options?: AmgPcgOptions | null): AmgPcgSolver;",
  ],
  [
    /(export class AmgPcgSolver \{[\s\S]*?)static fromGraphLaplacian\([^;]+\): AmgPcgSolver;/,
    "$1static fromGraphLaplacian(graph: CsrGraph, kind?: string | null, shift?: number | null, options?: AmgPcgOptions | null): AmgPcgSolver;",
  ],
  [
    /(export class AmgPcgSolver \{[\s\S]*?)solveBatch\([^;]+\): AmgPcgBatchResult;/,
    "$1solveBatch(bFlat: Float64Array | number[], ncols: number, x0Flat?: Float64Array | number[] | null): AmgPcgBatchResult;",
  ],
  [
    /export function computeEffectiveResistances\([^;]+\): EffectiveResistanceResult;/,
    "export function computeEffectiveResistances(graph: CsrGraph, options?: EffectiveResistanceOptions | null): EffectiveResistanceResult;",
  ],
  [
    /export function resistanceCurvatureFlow\([^;]+\): ResistanceCurvatureFlowResult;/,
    "export function resistanceCurvatureFlow(graph: CsrGraph, options?: ResistanceCurvatureFlowOptions | null): ResistanceCurvatureFlowResult;",
  ],
];

export const clusteringSolversDtsReplacements = [
  [
    /static fromFuzzySet\([^;]+\): GuMapSolver;/,
    "static fromFuzzySet(probs: SparseProbabilities, nEpochsOrOptions?: number | GuMapOptions | null, negativeSampleRate?: number | null, learningRate?: number | null, minDist?: number | null, spread?: number | null, a?: number | null, b?: number | null, seed?: number | null, sigmaIters?: number | null, sigmaTolerance?: number | null): GuMapSolver;",
  ],
  [
    /static fromGraphDog\([^;]+\): GuMapSolver;/,
    "static fromGraphDog(graph: CsrGraph, max2hopOrOptions?: number | GuMapDogOptions | null, nEpochsOrOptions?: number | GuMapDogOptions | null, negativeSampleRate?: number | null, learningRate?: number | null, minDist?: number | null, spread?: number | null, a?: number | null, b?: number | null, sigmaIters?: number | null, sigmaTolerance?: number | null, seed?: number | null, degree?: number | null, numLandmarks?: number | null, degreeNormalize?: boolean | null, pExp?: number | null, epsPsd?: number | null, tCoarse?: number | null, t1?: number | null, t2?: number | null, alpha?: number | null, betaFloor?: number | null, minDegree?: number | null, jlDim?: number | null): GuMapSolver;",
  ],
  [
    /static fromLandmarks\([^;]+\): GuMapSolver;/,
    "static fromLandmarks(graph: CsrGraph, landmarks: DogVaradhanLandmarks, options?: GuMapLandmarksOptions | null): GuMapSolver;\n    static fromLandmarks(graph: CsrGraph, landmarks: DogVaradhanLandmarks, max2hopPerNode?: number | null, options?: GuMapOptions | null): GuMapSolver;",
  ],
  [
    /static fromKnn\([^;]+\): GuMapSolver;/,
    "static fromKnn(knn: KnnGraph, nEpochsOrOptions?: number | GuMapOptions | null, negativeSampleRate?: number | null, learningRate?: number | null, minDist?: number | null, spread?: number | null, a?: number | null, b?: number | null, sigmaIters?: number | null, sigmaTolerance?: number | null, seed?: number | null): GuMapSolver;",
  ],
  [
    /static fromDistanceMatrix\(num_nodes: number, dists_flat: Float64Array \| number\[\], eps_or_options: any[^;]*\): KamadaKawaiSolver;/,
    "static fromDistanceMatrix(numNodes: number, distsFlat: Float64Array | number[], epsOrOptions?: number | KamadaKawaiOptions | null, maxInnerIters?: number | null, maxOuterIters?: number | null): KamadaKawaiSolver;",
  ],
  [
    /static fromGraph\(graph: CsrGraph, eps_or_options: any[^;]*\): KamadaKawaiSolver;/,
    "static fromGraph(graph: CsrGraph, epsOrOptions?: number | KamadaKawaiOptions | null, maxInnerIters?: number | null, maxOuterIters?: number | null): KamadaKawaiSolver;",
  ],
  [
    /static fromPairsWithHinges\([^;]+\): SgdSolver;/,
    "static fromPairsWithHinges(numNodes: number, springSrc: Uint32Array | number[], springTgt: Uint32Array | number[], springDij: Float32Array | number[], springDji: Float32Array | number[], springWij: Float32Array | number[], springWji: Float32Array | number[], hingeSrc: Uint32Array | number[], hingeTgt: Uint32Array | number[], hingeDij: Float32Array | number[], hingeDji: Float32Array | number[], hingeWij: Float32Array | number[], hingeWji: Float32Array | number[], tMaxOrOptions?: number | SgdScheduleOptions | null, eps?: number | null, kind?: SgdScheduleKind | string | null, stepScale?: number | null, seed?: number | null): SgdSolver;",
  ],
  [
    /static fromPairs\([^;]+\): SgdSolver;/,
    "static fromPairs(numNodes: number, src: Uint32Array | number[], tgt: Uint32Array | number[], dij: Float32Array | number[], dji?: Float32Array | number[] | null, wij?: Float32Array | number[] | null, wji?: Float32Array | number[] | null, tMaxOrOptions?: number | SgdScheduleOptions | null, eps?: number | null, kind?: SgdScheduleKind | string | null, stepScale?: number | null, seed?: number | null): SgdSolver;",
  ],
  [
    /static full\([^;]+\): SgdSolver;/,
    "static full(graph: CsrGraph, tMaxOrOptions?: number | SgdScheduleOptions | null, eps?: number | null, kind?: SgdScheduleKind | string | null, stepScale?: number | null, seed?: number | null): SgdSolver;",
  ],
  [
    /static omega\([^;]+\): SgdSolver;/,
    "static omega(graph: CsrGraph, embedding: Float64Array | number[], dims: number, kOrOptions?: number | OmegaSgdOptions | null, minDist?: number | null, tMaxOrOptions?: number | SgdScheduleOptions | null, eps?: number | null, kind?: SgdScheduleKind | string | null, stepScale?: number | null, seed?: number | null): SgdSolver;",
  ],
  [
    /static randomPair\([^;]+\): SgdSolver;/,
    "static randomPair(graph: CsrGraph, pairsPerNodeOrOptions?: number | RandomPairSgdOptions | null, tMaxOrOptions?: number | SgdScheduleOptions | null, eps?: number | null, kind?: SgdScheduleKind | string | null, stepScale?: number | null, seed?: number | null): SgdSolver;",
  ],
  [
    /static sparseBarrierDog\([^;]+\): SgdSolver;/,
    "static sparseBarrierDog(graph: CsrGraph, max2hopOrOptions?: number | SparseBarrierDogOptions | null, hingeRepulsion?: boolean | null, deltaRel?: number | null, barrierSym?: BarrierSymmetry | string | null, omega1?: number | null, omega2?: number | null, gammaNon2?: number | null, htRepulsion?: boolean | null, minRatio?: number | null, epsDist?: number | null, negPerNode?: number | null, numPivots?: number | null, corePivots?: boolean | null, tMax?: number | null, eps?: number | null, kind?: SgdScheduleKind | string | null, stepScale?: number | null, seed?: number | null, degree?: number | null, numLandmarks?: number | null, degreeNormalize?: boolean | null, pExp?: number | null, epsPsd?: number | null, tCoarse?: number | null, t1?: number | null, t2?: number | null, alpha?: number | null, betaFloor?: number | null, minDegree?: number | null, jlDim?: number | null): SgdSolver;",
  ],
  [
    /static sparseBarrierFromLandmarks\([^;]+\): SgdSolver;/,
    "static sparseBarrierFromLandmarks(graph: CsrGraph, landmarks: DogVaradhanLandmarks, pivots?: Uint32Array | number[] | null, max2hopOrOptions?: number | SparseBarrierOptions | null, hingeRepulsion?: boolean | null, deltaRel?: number | null, barrierSym?: BarrierSymmetry | string | null, omega1?: number | null, omega2?: number | null, gammaNon2?: number | null, htRepulsion?: boolean | null, minRatio?: number | null, epsDist?: number | null, negPerNode?: number | null, numPivots?: number | null, corePivots?: boolean | null, tMax?: number | null, eps?: number | null, kind?: SgdScheduleKind | string | null, stepScale?: number | null, seed?: number | null): SgdSolver;",
  ],
  [
    /static streamingNegatives\([^;]+\): SgdSolver;/,
    "static streamingNegatives(graph: CsrGraph, negativesOrOptions?: number | StreamingNegativeOptions | null, targetDistance?: number | null, weight?: number | null, hingeOnly?: boolean | null, tMax?: number | null, eps?: number | null, kind?: SgdScheduleKind | string | null, stepScale?: number | null, seed?: number | null): SgdSolver;",
  ],
  [
    /constructor\(graph: CsrGraph, rounds_or_options: any[^;]*\);/,
    "constructor(graph: CsrGraph, roundsOrOptions?: number | SpatialRefinerOptions | null, kCap?: number | null, marginScale?: number | null, wAnchor?: number | null, wFn?: number | null, wFp?: number | null, fnShrink?: number | null, roundDecay?: number | null, tMax?: number | null, eps?: number | null, hingeFp?: boolean | null, seed?: number | null, wAnchor2?: number | null, kCapDecay?: number | null, kCapMin?: number | null, minSepRatio?: number | null, wSep?: number | null);",
  ],
  [
    /setKCap\(k_cap: number\): void;/,
    "setKCap(kCap: number): void;",
  ],
  [
    /static fromDistanceMatrix\(num_nodes: number, dists_flat: Float64Array \| number\[\], epsilon_or_options: any[^;]*\): StressMajorizationSolver;/,
    "static fromDistanceMatrix(numNodes: number, distsFlat: Float64Array | number[], epsilonOrOptions?: number | StressMajorizationOptions | null, maxIterations?: number | null, pcgTolerance?: number | null, pcgMaxIterations?: number | null, minNorm?: number | null): StressMajorizationSolver;",
  ],
  [
    /static fromGraph\(graph: CsrGraph, epsilon_or_options: any[^;]*\): StressMajorizationSolver;/,
    "static fromGraph(graph: CsrGraph, epsilonOrOptions?: number | StressMajorizationOptions | null, maxIterations?: number | null, pcgTolerance?: number | null, pcgMaxIterations?: number | null, minNorm?: number | null): StressMajorizationSolver;",
  ],
  [
    /static fromGraphPartialBfs\([^;]+\): TsNetSolver;/,
    "static fromGraphPartialBfs(graph: CsrGraph, k?: number | null, uniformPerplexity?: number | null, backendOrOptions?: TsNetBackend | TsNetOptions | string | null, iterationsStage1?: number | null, iterationsStage2?: number | null, iterationsStage3?: number | null, learningRate?: number | null, exaggeration?: number | null, lambdaCStage1?: number | null, lambdaCStage2?: number | null, lambdaCStage3?: number | null, lambdaRStage3?: number | null, momentum?: number | null, epsilonR?: number | null, theta?: number | null, power?: number | null, sigmaIters?: number | null, sigmaTolerance?: number | null, rowNormalize?: boolean | null, seed?: number | null, intervals?: number | null, interpolationOrder?: \"bilinear\" | \"cubic\" | 1 | 3 | null, minGridDim?: number | null, maxGridDim?: number | null): TsNetSolver;",
  ],
  [
    /static fromLayoutScaled\([^;]+\): ManifoldLayout2d;/,
    "static fromLayoutScaled(kind: ManifoldKind | string, layout: Layout2d, maxRadius?: number | null): ManifoldLayout2d;",
  ],
  [
    /static randomInit\([^;]+\): ManifoldLayout2d;/,
    "static randomInit(kind: ManifoldKind | string, n: number, scale?: number | null, seed?: number | null): ManifoldLayout2d;",
  ],
  [
    /static fromGraphRowNormalized\([^;]+\): TsNetSolver;/,
    "static fromGraphRowNormalized(graph: CsrGraph, w2OrOptions?: number | TsNetRowNormalizedOptions | null, max2hopPerNode?: number | null, backendOrOptions?: TsNetBackend | TsNetOptions | string | null, iterationsStage1?: number | null, iterationsStage2?: number | null, iterationsStage3?: number | null, learningRate?: number | null, exaggeration?: number | null, lambdaCStage1?: number | null, lambdaCStage2?: number | null, lambdaCStage3?: number | null, lambdaRStage3?: number | null, momentum?: number | null, epsilonR?: number | null, theta?: number | null, seed?: number | null, intervals?: number | null, interpolationOrder?: \"bilinear\" | \"cubic\" | 1 | 3 | null, minGridDim?: number | null, maxGridDim?: number | null, degreeBeta?: number | null): TsNetSolver;",
  ],
  [
    /static fromKnn\([^;]+\): TsNetSolver;/,
    "static fromKnn(knn: KnnGraph, perplexities?: Float64Array | number[] | null, uniformPerplexity?: number | null, backendOrOptions?: TsNetBackend | TsNetOptions | string | null, iterationsStage1?: number | null, iterationsStage2?: number | null, iterationsStage3?: number | null, learningRate?: number | null, exaggeration?: number | null, lambdaCStage1?: number | null, lambdaCStage2?: number | null, lambdaCStage3?: number | null, lambdaRStage3?: number | null, momentum?: number | null, epsilonR?: number | null, theta?: number | null, power?: number | null, sigmaIters?: number | null, sigmaTolerance?: number | null, rowNormalize?: boolean | null, seed?: number | null, intervals?: number | null, interpolationOrder?: \"bilinear\" | \"cubic\" | 1 | 3 | null, minGridDim?: number | null, maxGridDim?: number | null): TsNetSolver;",
  ],
  [
    /static fromProbabilities\([^;]+\): TsNetSolver;/,
    "static fromProbabilities(probs: SparseProbabilities, backendOrOptions?: TsNetBackend | TsNetOptions | string | null, iterationsStage1?: number | null, iterationsStage2?: number | null, iterationsStage3?: number | null, learningRate?: number | null, exaggeration?: number | null, lambdaCStage1?: number | null, lambdaCStage2?: number | null, lambdaCStage3?: number | null, lambdaRStage3?: number | null, momentum?: number | null, epsilonR?: number | null, theta?: number | null, seed?: number | null, intervals?: number | null, interpolationOrder?: \"bilinear\" | \"cubic\" | 1 | 3 | null, minGridDim?: number | null, maxGridDim?: number | null): TsNetSolver;",
  ],
  [
    /static pure2Hop\([^;]+\): TsNetSolver;/,
    "static pure2Hop(graph: CsrGraph, backendOrOptions?: TsNetBackend | TsNetOptions | string | null, iterationsStage1?: number | null, iterationsStage2?: number | null, iterationsStage3?: number | null, learningRate?: number | null, perplexityC?: number | null, minPerplexity?: number | null, maxPerplexity?: number | null, exaggeration?: number | null, lambdaCStage1?: number | null, lambdaCStage2?: number | null, lambdaCStage3?: number | null, lambdaRStage3?: number | null, momentum?: number | null, epsilonR?: number | null, theta?: number | null, power?: number | null, sigmaIters?: number | null, sigmaTolerance?: number | null, rowNormalize?: boolean | null, seed?: number | null, intervals?: number | null, interpolationOrder?: \"bilinear\" | \"cubic\" | 1 | 3 | null, minGridDim?: number | null, maxGridDim?: number | null): TsNetSolver;",
  ],
  [
    /export function kmeansPlusPlus\([^;]+\): KMeansResult;/,
    "export function kmeansPlusPlus(points: Float64Array | number[], numNodes: number, dims: number, k?: number | null, maxIters?: number | null, tol?: number | null, nInit?: number | null, seed?: number | null): KMeansResult;",
  ],
  [
    /export function labelPropagation\([^;]+\): Uint32Array;/,
    "export function labelPropagation(graph: CsrGraph, maxIters?: number | null, seed?: number | null): Uint32Array;",
  ],
  [
    /export function louvainCommunities\([^;]+\): LouvainResult;/,
    "export function louvainCommunities(graph: CsrGraph, maxPasses?: number | null, resolution?: number | null, minModularityGain?: number | null, seed?: number | null): LouvainResult;",
  ],
  [
    /export function projectClusteredLayout\([^;]+\): Layout2d;/,
    "export function projectClusteredLayout(coarseLayout: Layout2d, nodeToCoarse: Uint32Array | number[], spreadRadius?: number | null, seed?: number | null): Layout2d;",
  ],
  [
    /export function spectralClustering\([^;]+\): Uint32Array;/,
    "export function spectralClustering(graph: CsrGraph, k?: number | null, maxEigenIters?: number | null, eigenTol?: number | null, kmeansMaxIters?: number | null, kmeansTol?: number | null, seed?: number | null): Uint32Array;",
  ],
];
