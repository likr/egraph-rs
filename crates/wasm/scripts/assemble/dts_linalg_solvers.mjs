export const linalgNeighborhoodDtsReplacements = [
  [
    /(export class ChebyshevDogKernel \{[\s\S]*?)static withLandmarks\([^;]+\): ChebyshevDogKernel;/,
    "$1static withLandmarks(graph: CsrGraph, landmarks: Uint32Array | number[], options?: ChebyshevDogKernelOptions | null): ChebyshevDogKernel;",
  ],
  [
    /(export class ChebyshevDogKernel \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: ChebyshevDogKernelOptions | null);",
  ],
  [
    /(export class ChebyshevDogKernel \{[\s\S]*?)termsPsd\(\): Array<any>;/,
    "$1termsPsd(): SpectralFilterTerm[];",
  ],
  [
    /(export class CsrMatrix \{[\s\S]*?)estimateLambdaMax\([^;]+\): number;/,
    "$1estimateLambdaMax(numIters?: number | null, safetyFactor?: number | null, seed?: number | bigint | null): number;",
  ],
  [
    /(export class CsrMatrix \{[\s\S]*?)estimateLambdaMaxLanczos\([^;]+\): number;/,
    "$1estimateLambdaMaxLanczos(numSteps?: number | null, safetyFactor?: number | null, seed?: number | bigint | null): number;",
  ],
  [
    /(export class CsrMatrix \{[\s\S]*?)static fromCsr\([^;]+\): CsrMatrix;/,
    "$1static fromCsr(numNodes: number, indptr: Uint32Array | number[], indices: Uint32Array | number[], values: Float64Array | number[]): CsrMatrix;",
  ],
  [
    /export function buildDogTerms\([^;]+\): Array<any>;/,
    "export function buildDogTerms(t1: number, t2: number, alpha: number, options?: { betaFloor?: number | null } | null): SpectralFilterTerm[];",
  ],
  [
    /export function buildRegularizedDogTerms\([^;]+\): Array<any>;/,
    "export function buildRegularizedDogTerms(sigmaLambda: number, betaFloor?: number | null): SpectralFilterTerm[];",
  ],
  [
    /export function computeChebyshevBandpassCoefficients\([^;]+\): Float64Array;/,
    "export function computeChebyshevBandpassCoefficients(terms: SpectralFilterTermsInput, lambdaMax?: number | null, degree?: number | null): Float64Array;",
  ],
  [
    /export function projectPsdSpectralFilter\([^;]+\): any;/,
    "export function projectPsdSpectralFilter(terms: SpectralFilterTermsInput, options?: { lambdaMax?: number | null; degree?: number | null; epsPsd?: number | null } | null): PsdSpectralFilterProjection;",
  ],
  [
    /export function symmetricNormalizedLaplacian\(graph: CsrGraph\): any;/,
    "export function symmetricNormalizedLaplacian(graph: CsrGraph): SymmetricNormalizedLaplacianResult;",
  ],
  [
    /static fromCsr\(num_nodes: number, indptr: Uint32Array \| number\[\], indices: Uint32Array \| number\[\], distances: Float64Array \| number\[\]\): KnnGraph;/,
    "static fromCsr(numNodes: number, indptr: Uint32Array | number[], indices: Uint32Array | number[], distances: Float64Array | number[]): KnnGraph;",
  ],
  [
    /export function selectMaxminSpPivots\([^;]+\): Uint32Array;/,
    "export function selectMaxminSpPivots(graph: CsrGraph, options?: PivotMdsGeneratorOptions | null): Uint32Array;",
  ],
  [
    /export function pivotMdsInit\([^;]+\): Layout2d;/,
    "export function pivotMdsInit(graph: CsrGraph, options?: PivotMdsGeneratorOptions | null): Layout2d;",
  ],
  [
    /export function pivotMds\([^;]+\): Layout2d;/,
    "export function pivotMds(graph: CsrGraph, options?: PivotMdsGeneratorOptions | null): Layout2d;",
  ],
  [
    /export function pivotMdsFromDistances\([^;]+\): Layout2d;/,
    "export function pivotMdsFromDistances(distancesNk: Float64Array | number[], numNodes: number, numPivots: number, options?: { targetStd?: number | null; seed?: number | bigint | null } | null): Layout2d;",
  ],
  [
    /export function pivotMdsWithPivots\([^;]+\): Layout2d;/,
    "export function pivotMdsWithPivots(graph: CsrGraph, pivots: Uint32Array | number[], options?: { targetStd?: number | null; seed?: number | bigint | null } | null): Layout2d;",
  ],
  [
    /export function korenSpectralInit\([^;]+\): Layout2d;/,
    "export function korenSpectralInit(graph: CsrGraph, options?: KorenSpectralGeneratorOptions | null): Layout2d;",
  ],
  [
    /export function smallestLaplacianEigenpairs\([^;]+\): EigenResult;/,
    "export function smallestLaplacianEigenpairs(graph: CsrGraph, options?: SpectralEigenOptions | null): EigenResult;",
  ],
  [
    /export function symmetricEigenJacobi\([^;]+\): EigenResult;/,
    "export function symmetricEigenJacobi(aRowMajor: Float64Array | number[], n: number, options?: SpectralEigenOptions | null): EigenResult;",
  ],
  [
    /export function rdMds\([^;]+\): LayoutNd;/,
    "export function rdMds(graph: CsrGraph, options?: RdMdsGeneratorOptions | null): LayoutNd;",
  ],
  [
    /export function computeResistanceEmbedding\([^;]+\): SpectralEmbeddingResult;/,
    "export function computeResistanceEmbedding(graph: CsrGraph, options?: SpectralEmbeddingOptions | null): SpectralEmbeddingResult;",
  ],
  [
    /export function computeDiffusionEmbedding\([^;]+\): SpectralEmbeddingResult;/,
    "export function computeDiffusionEmbedding(graph: CsrGraph, options?: SpectralEmbeddingOptions | null): SpectralEmbeddingResult;",
  ],
  [
    /export function chebyshevHeatSpmv\([^;]+\): Float64Array;/,
    "export function chebyshevHeatSpmv(graph: CsrGraph, vectors: Float64Array | number[], numCols: number, t: number, options?: ChebyshevHeatSpmvOptions | null): Float64Array;",
  ],
  [
    /export function chebyshevHeatSpmvMatrix\([^;]+\): Float64Array;/,
    "export function chebyshevHeatSpmvMatrix(matrix: CsrMatrix, vectors: Float64Array | number[], numCols: number, t: number, options?: ChebyshevHeatSpmvOptions | null): Float64Array;",
  ],
  [
    /export function chebyshevDogJlEmbedding\([^;]+\): Float64Array;/,
    "export function chebyshevDogJlEmbedding(graph: CsrGraph, options?: ChebyshevDogJlOptions | null): Float64Array;",
  ],
  [
    /export function chebyshevDogJlEmbeddingMatrix\([^;]+\): Float64Array;/,
    "export function chebyshevDogJlEmbeddingMatrix(matrix: CsrMatrix, options?: ChebyshevDogJlOptions | null): Float64Array;",
  ],
  [
    /export function anisotropicLaplacian\([^;]+\): AnisotropicLaplacianResult;/,
    "export function anisotropicLaplacian(graph: CsrGraph, options?: AnisotropicLaplacianOptions | null): AnisotropicLaplacianResult;",
  ],
  [
    /export function extractSparse1hop2hopPairs\([^;]+\): Sparse12HopPairs;/,
    "export function extractSparse1hop2hopPairs(graph: CsrGraph, max2hopPerNode?: number | null, seed?: number | bigint | null): Sparse12HopPairs;",
  ],
  [
    /export function extractSparse1hop2hopPairsWithMode\([^;]+\): Sparse12HopPairs;/,
    "export function extractSparse1hop2hopPairsWithMode(graph: CsrGraph, options?: TwoHopModeOptions | null): Sparse12HopPairs;",
  ],
  [
    /export function buildCapped12hopKnn\([^;]+\): TwoHopKnnResult;/,
    "export function buildCapped12hopKnn(graph: CsrGraph, max2hopPerNode?: number | null, seed?: number | bigint | null): TwoHopKnnResult;",
  ],
  [
    /export function partialBfsKnn\([^;]+\): KnnGraph;/,
    "export function partialBfsKnn(graph: CsrGraph, k?: number | null, seed?: number | bigint | null): KnnGraph;",
  ],
  [
    /export function normalizeLayout2d\([^;]+\): void;/,
    "export function normalizeLayout2d(layout: Layout2d, targetStd?: number | null, seed?: number | bigint | null): void;",
  ],
  [
    /export function robustNormalizeLayout2d\([^;]+\): void;/,
    "export function robustNormalizeLayout2d(layout: Layout2d, options?: { targetStd?: number | null; quantile?: number | null; seed?: number | bigint | null } | null): void;",
  ],
  [
    /export function graph12hopRowNormalizedProbabilities\([^;]+\): SparseProbabilities;/,
    "export function graph12hopRowNormalizedProbabilities(graph: CsrGraph, options?: TsNetRowNormalizedOptions | null): SparseProbabilities;",
  ],
  [
    /export function graph12hopRowNormalizedProbabilitiesWithOptions\([^;]+\): SparseProbabilities;/,
    "export function graph12hopRowNormalizedProbabilitiesWithOptions(graph: CsrGraph, options?: TsNetRowNormalizedOptions | null): SparseProbabilities;",
  ],
  [
    /export function computeNodewise2hopPerplexities\([^;]+\): Float64Array;/,
    "export function computeNodewise2hopPerplexities(graph: CsrGraph, options?: PerplexityOptions | null): Float64Array;",
  ],
  [
    /export function computeMedian2hopPerplexity\([^;]+\): number;/,
    "export function computeMedian2hopPerplexity(graph: CsrGraph, options?: PerplexityOptions | null): number;",
  ],
  [
    /export function buildHybridAdaptiveKnn\([^;]+\): HybridKnnResult;/,
    "export function buildHybridAdaptiveKnn(graph: CsrGraph, coords: Float64Array | number[], dim: number, perplexities: Float64Array | number[], options?: HybridAdaptiveKnnOptions | null): HybridKnnResult;",
  ],
  [
    /export function computeKnnProbabilities\([^;]+\): SparseProbabilities;/,
    "export function computeKnnProbabilities(knn: KnnGraph, perplexities: Float64Array | number[], options?: KnnProbabilitiesOptions | null): SparseProbabilities;",
  ],
  [
    /export function buildRerankKnn\([^;]+\): KnnGraph;/,
    "export function buildRerankKnn(coords: Float64Array | number[], numNodes: number, dim: number, k: number, options?: { coarseDim?: number | null; rerankFactor?: number | null } | null): KnnGraph;",
  ],
  [
    /export function buildGumap12hopPriorityCsr\([^;]+\): KnnGraph;/,
    "export function buildGumap12hopPriorityCsr(numNodes: number, u1: Uint32Array | number[], v1: Uint32Array | number[], d1: Float64Array | number[], u2: Uint32Array | number[], v2: Uint32Array | number[], d2: Float64Array | number[], options?: { tier2Margin?: number | null } | null): KnnGraph;",
  ],
  [
    /export function applyRelativeDistanceFloor\([^;]+\): number;/,
    "export function applyRelativeDistanceFloor(d1: Float64Array, d2: Float64Array, d3: Float64Array, options?: { epsDist?: number | null; minRatio?: number | null } | null): number;",
  ],
  [
    /export function computeRegularizedDogVaradhanLandmarks\([^;]+\): DogVaradhanLandmarks;/,
    "export function computeRegularizedDogVaradhanLandmarks(graph: CsrGraph, options?: DogVaradhanOptions | null): DogVaradhanLandmarks;",
  ],
  [
    /export function computeRegularizedDogVaradhanLandmarksFromMatrix\([^;]+\): DogVaradhanLandmarks;/,
    "export function computeRegularizedDogVaradhanLandmarksFromMatrix(graph: CsrGraph, matrix: CsrMatrix, options?: DogVaradhanOptions | null): DogVaradhanLandmarks;",
  ],
  [
    /(export class ClassicalMds \{[\s\S]*?)generate\(graph: CsrGraph\): any;/,
    "$1generate(graph: CsrGraph): Layout2d | Layout3d | LayoutNd;",
  ],
  [
    /(export class ClassicalMds \{[\s\S]*?)constructor\(options: any\);/,
    "$1constructor(options?: ClassicalMdsOptions | null);",
  ],
  [
    /(export class PivotMds \{[\s\S]*?)generate\(graph: CsrGraph\): any;/,
    "$1generate(graph: CsrGraph): Layout2d | Layout3d | LayoutNd;",
  ],
  [
    /(export class PivotMds \{[\s\S]*?)constructor\(options: any\);/,
    "$1constructor(options?: PivotMdsGeneratorOptions | null);",
  ],
  [
    /(export class RdMds \{[\s\S]*?)generate\(graph: CsrGraph\): any;/,
    "$1generate(graph: CsrGraph): Layout2d | Layout3d | LayoutNd;",
  ],
  [
    /(export class RdMds \{[\s\S]*?)constructor\(options: any\);/,
    "$1constructor(options?: RdMdsGeneratorOptions | null);",
  ],
  [
    /(export class KorenSpectral \{[\s\S]*?)generate\(graph: CsrGraph\): any;/,
    "$1generate(graph: CsrGraph): Layout2d | Layout3d | LayoutNd;",
  ],
  [
    /(export class KorenSpectral \{[\s\S]*?)constructor\(options: any\);/,
    "$1constructor(options?: KorenSpectralGeneratorOptions | null);",
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
    "$1static fromGraphLaplacian(graph: CsrGraph, kind?: AmgLaplacianKind | string | null, shift?: number | null, options?: AmgPcgOptions | null): AmgPcgSolver;",
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
    /(export class GuMapSolver \{[\s\S]*?)_step2d\(layout: Layout2d\): string;/,
    "$1step(layout: AnyLayout): string;\n    runSteps(layout: AnyLayout, steps: number): string;\n    run(layout: AnyLayout): string;\n    _step2d(layout: Layout2d): string;",
  ],
  [
    /(export class GuMapSolver \{[\s\S]*?)static fromFuzzySet\([^;]+\): GuMapSolver;/,
    "$1static fromFuzzySet(probs: SparseProbabilities, options?: GuMapOptions | null): GuMapSolver;",
  ],
  [
    /(export class GuMapSolver \{[\s\S]*?)static fromGraphDog\([^;]+\): GuMapSolver;/,
    "$1static fromGraphDog(graph: CsrGraph, options?: GuMapDogOptions | null): GuMapSolver;",
  ],
  [
    /(export class GuMapSolver \{[\s\S]*?)static fromLandmarks\([^;]+\): GuMapSolver;/,
    "$1static fromLandmarks(graph: CsrGraph, landmarks: DogVaradhanLandmarks, options?: GuMapLandmarksOptions | null): GuMapSolver;",
  ],
  [
    /(export class GuMapSolver \{[\s\S]*?)static fromKnn\([^;]+\): GuMapSolver;/,
    "$1static fromKnn(knn: KnnGraph, options?: GuMapOptions | null): GuMapSolver;",
  ],
  [
    /(export class GuMapSolver \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: GuMapDogOptions | null);",
  ],
  [
    /(export class KamadaKawaiSolver \{[\s\S]*?)_step2d\(layout: Layout2d\): string;/,
    "$1step(layout: AnyLayout): string;\n    runSteps(layout: AnyLayout, steps: number): string;\n    run(layout: AnyLayout): string;\n    _step2d(layout: Layout2d): string;",
  ],
  [
    /(export class KamadaKawaiSolver \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: KamadaKawaiOptions | null);",
  ],
  [
    /(export class KamadaKawaiSolver \{[\s\S]*?)static fromDistanceMatrix\([^;]+\): KamadaKawaiSolver;/,
    "$1static fromDistanceMatrix(numNodes: number, distsFlat: Float64Array | number[], options?: KamadaKawaiOptions | null): KamadaKawaiSolver;",
  ],
  [
    /(export class KamadaKawaiSolver \{[\s\S]*?)static fromGraph\([^;]+\): KamadaKawaiSolver;/,
    "$1static fromGraph(graph: CsrGraph, options?: KamadaKawaiOptions | null): KamadaKawaiSolver;",
  ],
  [
    /(export class SgdSolver \{[\s\S]*?)_step2d\(layout: Layout2d\): string;/,
    "$1step(layout: AnyLayout): string;\n    runSteps(layout: AnyLayout, steps: number): string;\n    run(layout: AnyLayout): string;\n    runScheduleWindow(layout: AnyLayout, startIter: number, count: number): string;\n    _step2d(layout: Layout2d): string;",
  ],
  [
    /(export class SgdSolver \{[\s\S]*?)static fromPairsWithHinges\([^;]+\): SgdSolver;/,
    "$1static fromPairsWithHinges(numNodes: number, springSrc: Uint32Array | number[], springTgt: Uint32Array | number[], springDij: Float32Array | number[], hingeSrc: Uint32Array | number[], hingeTgt: Uint32Array | number[], hingeDij: Float32Array | number[], options?: SgdPairsWithHingesOptions | null): SgdSolver;",
  ],
  [
    /(export class SgdSolver \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: SgdScheduleOptions | null);",
  ],
  [
    /(export class SgdSolver \{[\s\S]*?)static fromPairs\([^;]+\): SgdSolver;/,
    "$1static fromPairs(numNodes: number, src: Uint32Array | number[], tgt: Uint32Array | number[], dij: Float32Array | number[], options?: SgdPairsOptions | null): SgdSolver;",
  ],
  [
    /(export class SgdSolver \{[\s\S]*?)static full\([^;]+\): SgdSolver;/,
    "$1static full(graph: CsrGraph, options?: SgdScheduleOptions | null): SgdSolver;",
  ],
  [
    /(export class SgdSolver \{[\s\S]*?)static omega\([^;]+\): SgdSolver;/,
    "$1static omega(graph: CsrGraph, embedding: Float64Array | number[], dims: number, options?: OmegaSgdOptions | null): SgdSolver;",
  ],
  [
    /(export class SgdSolver \{[\s\S]*?)static randomPair\([^;]+\): SgdSolver;/,
    "$1static randomPair(graph: CsrGraph, options?: RandomPairSgdOptions | null): SgdSolver;",
  ],
  [
    /(export class SgdSolver \{[\s\S]*?)static sparseBarrierDog\([^;]+\): SgdSolver;/,
    "$1static sparseBarrierDog(graph: CsrGraph, options?: SparseBarrierDogOptions | null): SgdSolver;",
  ],
  [
    /(export class SgdSolver \{[\s\S]*?)static sparseBarrierFromLandmarks\([^;]+\): SgdSolver;/,
    "$1static sparseBarrierFromLandmarks(graph: CsrGraph, landmarks: DogVaradhanLandmarks, options?: SparseBarrierOptions | null): SgdSolver;",
  ],
  [
    /(export class SgdSolver \{[\s\S]*?)static streamingNegatives\([^;]+\): SgdSolver;/,
    "$1static streamingNegatives(graph: CsrGraph, options?: StreamingNegativeOptions | null): SgdSolver;",
  ],
  [
    /(export class SpatialIntruderRefiner \{[\s\S]*?)constructor\([^;]+\);/,
    "$1constructor(graph: CsrGraph, options?: SpatialRefinerOptions | null);",
  ],
  [
    /setKCap\(k_cap: number\): void;/,
    "setKCap(kCap: number): void;",
  ],
  [
    /(export class StressMajorizationSolver \{[\s\S]*?)_step2d\(layout: Layout2d\): string;/,
    "$1step(layout: AnyLayout): string;\n    runSteps(layout: AnyLayout, steps: number): string;\n    run(layout: AnyLayout): string;\n    computeStress(layout: AnyLayout): number;\n    _step2d(layout: Layout2d): string;",
  ],
  [
    /(export class StressMajorizationSolver \{[\s\S]*?)static fromDistanceMatrix\([^;]+\): StressMajorizationSolver;/,
    "$1static fromDistanceMatrix(numNodes: number, distsFlat: Float64Array | number[], options?: StressMajorizationOptions | null): StressMajorizationSolver;",
  ],
  [
    /(export class StressMajorizationSolver \{[\s\S]*?)static fromGraph\([^;]+\): StressMajorizationSolver;/,
    "$1static fromGraph(graph: CsrGraph, options?: StressMajorizationOptions | null): StressMajorizationSolver;",
  ],
  [
    /(export class StressMajorizationSolver \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: StressMajorizationOptions | null);",
  ],
  [
    /(export class TsNetSolver \{[\s\S]*?)_step2d\(layout: Layout2d\): string;/,
    "$1step(layout: AnyLayout): string;\n    runSteps(layout: AnyLayout, steps: number): string;\n    run(layout: AnyLayout): string;\n    _step2d(layout: Layout2d): string;",
  ],
  [
    /(export class TsNetSolver \{[\s\S]*?)static fromGraphRowNormalizedWithOptions\([^;]+\): TsNetSolver;/,
    "$1static fromGraphRowNormalizedWithOptions(graph: CsrGraph, options?: TsNetRowNormalizedOptions | null): TsNetSolver;",
  ],
  [
    /(export class TsNetSolver \{[\s\S]*?)static fromProbabilitiesWithStages\([^;]+\): TsNetSolver;/,
    "$1static fromProbabilitiesWithStages(probs: SparseProbabilities, stages: readonly (TsNetStage | TsNetStageOptions)[], options?: TsNetOptions | null): TsNetSolver;",
  ],
  [
    /(export class TsNetSolver \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: TsNetOptions | null);",
  ],
  [
    /(export class TsNetSolver \{[\s\S]*?)static fromGraphPartialBfs\([^;]+\): TsNetSolver;/,
    "$1static fromGraphPartialBfs(graph: CsrGraph, options?: TsNetPartialBfsOptions | null): TsNetSolver;",
  ],
  [
    /(export class TsNetSolver \{[\s\S]*?)static fromGraphRowNormalized\([^;]+\): TsNetSolver;/,
    "$1static fromGraphRowNormalized(graph: CsrGraph, options?: TsNetRowNormalizedOptions | null): TsNetSolver;",
  ],
  [
    /(export class TsNetSolver \{[\s\S]*?)static fromKnn\([^;]+\): TsNetSolver;/,
    "$1static fromKnn(knn: KnnGraph, options?: TsNetKnnOptions | null): TsNetSolver;",
  ],
  [
    /(export class TsNetSolver \{[\s\S]*?)static fromProbabilities\([^;]+\): TsNetSolver;/,
    "$1static fromProbabilities(probs: SparseProbabilities, options?: TsNetOptions | null): TsNetSolver;",
  ],
  [
    /(export class TsNetSolver \{[\s\S]*?)static pure2Hop\([^;]+\): TsNetSolver;/,
    "$1static pure2Hop(graph: CsrGraph, options?: TsNetOptions | null): TsNetSolver;",
  ],
  [
    /(export class TsNetStage \{[\s\S]*?)constructor\([^;]+\);/,
    "$1constructor(iterationsOrOptions: number | TsNetStageOptions, options?: TsNetStageOptions | null);",
  ],
  [
    /(export class ManifoldLayout2d \{[\s\S]*?)static init\([^;]+\): ManifoldLayout2d;/,
    "$1static init(kind: ManifoldKind | string, n: number, options?: LayoutInitOptions | null): ManifoldLayout2d;",
  ],
  [
    /(export class ManifoldLayout2d \{[\s\S]*?)static fromLayoutScaled\([^;]+\): ManifoldLayout2d;/,
    "$1static fromLayoutScaled(kind: ManifoldKind | string, layout: Layout2d, maxRadius?: number | null): ManifoldLayout2d;",
  ],
  [
    /(export class ManifoldLayout2d \{[\s\S]*?)static randomInit\([^;]+\): ManifoldLayout2d;/,
    "$1static randomInit(kind: ManifoldKind | string, n: number, options?: LayoutInitOptions | null): ManifoldLayout2d;",
  ],
  [
    /(export class ManifoldLayout2d \{[\s\S]*?)metric\(\): any;/,
    "$1metric(): ManifoldMetricRecord;",
  ],
  [
    /static torusSpectralPhaseInit\([^;]+\): ManifoldLayout2d;/,
    "static torusSpectralPhaseInit(graph: CsrGraph, seed?: number | bigint | null): ManifoldLayout2d;",
  ],
  [
    /export function torusSpectralPhaseInit\([^;]+\): ManifoldLayout2d;/,
    "export function torusSpectralPhaseInit(graph: CsrGraph, seed?: number | bigint | null): ManifoldLayout2d;",
  ],
  [
    /export function kmeansPlusPlus\([^;]+\): KMeansResult;/,
    "export function kmeansPlusPlus(points: Float64Array | number[], numNodes: number, dims: number, options?: KMeansOptions | null): KMeansResult;",
  ],
  [
    /export function labelPropagation\([^;]+\): LabelPropagationResult;/,
    "export function labelPropagation(graph: CsrGraph, maxIterations?: number | null, seed?: number | bigint | null): LabelPropagationResult;",
  ],
  [
    /export function louvainCommunities\([^;]+\): LouvainResult;/,
    "export function louvainCommunities(graph: CsrGraph, options?: LouvainOptions | null): LouvainResult;",
  ],
  [
    /export function projectClusteredLayout\([^;]+\): Layout2d;/,
    "export function projectClusteredLayout(coarseLayout: Layout2d, nodeToCoarse: Uint32Array | number[], options?: ProjectClusteredLayoutOptions | null): Layout2d;",
  ],
  [
    /export function spectralClustering\([^;]+\): SpectralClusteringResult;/,
    "export function spectralClustering(graph: CsrGraph, options?: SpectralClusteringOptions | null): SpectralClusteringResult;",
  ],
];
