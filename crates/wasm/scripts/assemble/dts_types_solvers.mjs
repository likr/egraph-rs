export const solverEnumAndPrimaryTypeDecls = `export type SolverStatus = "initialized" | "running" | "converged" | "completed" | "failed";
export type ManifoldKind = "euclidean" | "torus" | "spherical" | "hyperbolicPoincare" | "hyperbolicLorentz" | "hyperbolic";
export type SuggestedGeometry = "hyperbolic" | "spherical" | "torus" | "euclidean";
export type TsNetBackend = "auto2d" | "auto3d" | "exact" | "barnesHut" | "fitFft" | "linearFft";
export type TwoHopSamplingMode = "degreeOrderedHubSuppressed" | "twoStepRandomWalk";
export type SgdScheduleKind = "exponential" | "linear" | "quadratic" | "reciprocal" | "constant" | "squared";
export type BarrierSymmetry = "symMax" | "symAvg";
export type FieldRepulsionBackend = "ibFft" | "barnesHut" | "exact";
export type AmgLaplacianKind = "combinatorial" | "symmetricNormalized";
export type LayeringMethod = "longestPath" | "networkSimplex";
export type LayoutAxis = 0 | 1 | "x" | "y";

export interface AmgPcgOptions {
  relTol?: number | null;
  absTol?: number | null;
  maxIterations?: number | null;
  theta?: number | null;
  maxLevels?: number | null;
  maxCoarseSize?: number | null;
  preSweeps?: number | null;
  postSweeps?: number | null;
  maxAggSize?: number | null;
  diagEps?: number | null;
  eigEps?: number | null;
  symTol?: number | null;
  seed?: number | bigint | null;
  randomizeTies?: boolean | null;
  nearNullspace?: Float64Array | number[] | null;
}

export interface EffectiveResistanceOptions {
  jlDim?: number | null;
  seed?: number | bigint | null;
  weightsAreLengths?: boolean | null;
  exact?: boolean | null;
  exactMaxNodes?: number | null;
  eigEps?: number | null;
  relTol?: number | null;
  maxIterations?: number | null;
}

export interface ResistanceCurvatureFlowOptions extends EffectiveResistanceOptions {
  steps?: number | null;
  eta?: number | null;
  kappaMax?: number | null;
  minEdgeLength?: number | null;
  hyperbolicThreshold?: number | null;
  sphericalThreshold?: number | null;
  torusStdThreshold?: number | null;
}

export interface TsNetOptions {
  target?: TargetKind | string | null;
  backend?: TsNetBackend | string | null;
  stageIterations?: readonly [number, number, number] | Uint32Array | number[] | null;
  learningRate?: number | null;
  learningRateDivisor?: number | null;
  perplexityC?: number | null;
  minPerplexity?: number | null;
  maxPerplexity?: number | null;
  exaggeration?: number | null;
  lambdaCStage1?: number | null;
  lambdaCStage2?: number | null;
  lambdaCStage3?: number | null;
  lambdaRStage3?: number | null;
  momentum?: number | null;
  epsilonR?: number | null;
  theta?: number | null;
  power?: number | null;
  sigmaIters?: number | null;
  sigmaTolerance?: number | null;
  rowNormalize?: boolean | null;
  jacobiPrecondition?: boolean | null;
  intervals?: number | null;
  interpolationOrder?: "bilinear" | "cubic" | 1 | 3 | null;
  minGridDim?: number | null;
  maxGridDim?: number | null;
  seed?: number | bigint | null;
}

export interface TsNetRowNormalizedOptions extends TsNetOptions {
  w2?: number | null;
  max2hopPerNode?: number | null;
  degreeBeta?: number | null;
  mode?: TwoHopSamplingMode | string | null;
  samplingMode?: TwoHopSamplingMode | string | null;
  twoStepRandomWalk?: boolean | null;
  maxBridgeDegree?: number | null;
}

export interface TsNetKnnOptions extends TsNetOptions {
  perplexities?: Float64Array | number[] | null;
  uniformPerplexity?: number | null;
}

export interface TsNetPartialBfsOptions extends TsNetOptions {
  k?: number | null;
  uniformPerplexity?: number | null;
}

export interface TsNetStageOptions {
  iterations?: number | null;
  exaggeration?: number | null;
  lambdaKl?: number | null;
  lambdaC?: number | null;
  lambdaR?: number | null;
}

export interface SgdScheduleOptions {
  target?: TargetKind | string | null;
  iterations?: number | null;
  eps?: number | null;
  kind?: SgdScheduleKind | string | null;
  stepScale?: number | null;
  maxMu?: number | null;
  seed?: number | bigint | null;
}

export interface SgdPairsOptions extends SgdScheduleOptions {
  dji?: Float32Array | number[] | null;
  wij?: Float32Array | number[] | null;
  wji?: Float32Array | number[] | null;
}

export interface SgdPairsWithHingesOptions extends SgdScheduleOptions {
  springDji?: Float32Array | number[] | null;
  springWij?: Float32Array | number[] | null;
  springWji?: Float32Array | number[] | null;
  hingeDji?: Float32Array | number[] | null;
  hingeWij?: Float32Array | number[] | null;
  hingeWji?: Float32Array | number[] | null;
}

export interface SgdPackedSymmetricOptions extends SgdScheduleOptions {
  weights?: Float32Array | number[] | null;
  isHinge?: Uint8Array | number[] | null;
}

export interface RandomPairSgdOptions extends SgdScheduleOptions {
  pairsPerNode?: number | null;
}

export interface OmegaSgdOptions extends SgdScheduleOptions {
  k?: number | null;
  minDist?: number | null;
}

export interface StreamingNegativeOptions extends SgdScheduleOptions {
  negativesPerNode?: number | null;
  targetDistance?: number | null;
  weight?: number | null;
  hingeOnly?: boolean | null;
}

export interface SparseBarrierOptions extends SgdScheduleOptions {
  pivots?: Uint32Array | number[] | null;
  max2hopPerNode?: number | null;
  hingeRepulsion?: boolean | null;
  deltaRel?: number | null;
  barrierSym?: BarrierSymmetry | string | null;
  omega1?: number | null;
  omega2?: number | null;
  gammaNon2?: number | null;
  htRepulsion?: boolean | null;
  minRatio?: number | null;
  epsDist?: number | null;
  negPerNode?: number | null;
  numPivots?: number | null;
  corePivots?: boolean | null;
  degreeWeightBeta?: number | null;
}

export interface SparseBarrierDogOptions extends SparseBarrierOptions {
  degree?: number | null;
  numLandmarks?: number | null;
  minDegree?: number | null;
  jlDim?: number | null;
  degreeNormalize?: boolean | null;
  pExp?: number | null;
  epsPsd?: number | null;
  tCoarse?: number | null;
  t1?: number | null;
  t2?: number | null;
  alpha?: number | null;
  betaFloor?: number | null;
}

export interface GuMapOptions {
  target?: TargetKind | string | null;
  iterations?: number | null;
  negativeSampleRate?: number | null;
  learningRate?: number | null;
  minDist?: number | null;
  spread?: number | null;
  a?: number | null;
  b?: number | null;
  sigmaIters?: number | null;
  sigmaTolerance?: number | null;
  symmetricNegative?: boolean | null;
  seed?: number | bigint | null;
}

export interface GuMapDogOptions extends GuMapOptions {
  max2hopPerNode?: number | null;
  degree?: number | null;
  numLandmarks?: number | null;
  minDegree?: number | null;
  jlDim?: number | null;
  degreeNormalize?: boolean | null;
  pExp?: number | null;
  epsPsd?: number | null;
  tCoarse?: number | null;
  t1?: number | null;
  t2?: number | null;
  alpha?: number | null;
  betaFloor?: number | null;
}

export interface GuMapLandmarksOptions extends GuMapOptions {
  max2hopPerNode?: number | null;
}

export interface SpatialRefinerOptions {
  iterations?: number | null;
  kCap?: number | null;
  marginScale?: number | null;
  wAnchor?: number | null;
  wAnchor2?: number | null;
  wFn?: number | null;
  wFp?: number | null;
  fnShrink?: number | null;
  roundDecay?: number | null;
  kCapDecay?: number | null;
  kCapMin?: number | null;
  sgdIterations?: number | null;
  eps?: number | null;
  hingeFp?: boolean | null;
  minSepRatio?: number | null;
  wSep?: number | null;
  seed?: number | bigint | null;
}

export type StressMajorizationBackend = "euclideanPcg" | "geodesicGradient";

export interface StressMajorizationOptions {
  target?: TargetKind | string | null;
  backend?: StressMajorizationBackend | string | null;
  epsilon?: number | null;
  maxIterations?: number | null;
  pcgTolerance?: number | null;
  pcgMaxIterations?: number | null;
  minNorm?: number | null;
}

export interface KamadaKawaiOptions {
  target?: TargetKind | string | null;
  eps?: number | null;
  maxInnerIters?: number | null;
  maxIterations?: number | null;
}

export interface TFdpOptions {
  target?: "layout2d" | "layout3d" | "layoutNd" | string | null;
  backend?: FieldRepulsionBackend | string | null;
  iterations?: number | null;
  learningRate?: number | null;
  momentum?: number | null;
  theta?: number | null;
  alpha?: number | null;
  gamma?: number | null;
  rhoStart?: number | null;
  rhoEnd?: number | null;
  rhoPower?: number | null;
  interpolationOrder?: "bilinear" | "cubic" | 1 | 3 | null;
  intervals?: number | null;
  minGridDim?: number | null;
  maxGridDim?: number | null;
  seed?: number | bigint | null;
}

export interface IbUmapOptions {
  max2hopPerNode?: number | null;
  backend?: FieldRepulsionBackend | string | null;
  iterations?: number | null;
  learningRate?: number | null;
  minDist?: number | null;
  spread?: number | null;
  a?: number | null;
  b?: number | null;
  cK?: number | null;
  tauScale?: number | null;
  hubDamping?: boolean | null;
  theta?: number | null;
  sigmaIters?: number | null;
  sigmaTolerance?: number | null;
  degrees?: Uint32Array | number[] | null;
  interpolationOrder?: "bilinear" | "cubic" | 1 | 3 | null;
  intervals?: number | null;
  minGridDim?: number | null;
  maxGridDim?: number | null;
  degree?: number | null;
  numLandmarks?: number | null;
  minDegree?: number | null;
  jlDim?: number | null;
  degreeNormalize?: boolean | null;
  pExp?: number | null;
  epsPsd?: number | null;
  tCoarse?: number | null;
  t1?: number | null;
  t2?: number | null;
  alpha?: number | null;
  betaFloor?: number | null;
  seed?: number | bigint | null;
}

export interface SnapTFdpOptions {
  target?: TargetKind | string | null;
  iterations?: number | null;
  negativesPerEdge?: number | null;
  learningRate?: number | null;
  alpha?: number | null;
  repulsionScale?: number | null;
  rhoStart?: number | null;
  rhoEnd?: number | null;
  rhoPower?: number | null;
  degreeBeta?: number | null;
  boundedAttraction?: boolean | null;
  seed?: number | bigint | null;
}

export interface ChebyshevDogKernelOptions {
  t1?: number | null;
  t2?: number | null;
  alpha?: number | null;
  betaFloor?: number | null;
  degree?: number | null;
  numLandmarks?: number | null;
  minDegree?: number | null;
  jlDim?: number | null;
  epsPsd?: number | null;
  lambdaMax?: number | null;
  seed?: number | bigint | null;
}

export interface PacmapGraphOptions {
  target?: TargetKind | string | null;
  nNeighbors?: number | null;
  mnRatio?: number | null;
  fpRatio?: number | null;
  phaseIterations?: readonly [number, number, number] | Uint32Array | number[] | null;
  learningRate?: number | null;
  keepAll1hop?: boolean | null;
  max1hopPerNode?: number | null;
  degreeWeightBeta?: number | null;
  numLandmarks?: number | null;
  seed?: number | bigint | null;
}

export interface ClassicalMdsOptions {
  target?: "layout2d" | "layout3d" | "layoutNd" | string | null;
  dims?: number | null;
  eigEps?: number | null;
  tol?: number | null;
  maxSweeps?: number | null;
}

export interface PivotMdsGeneratorOptions {
  target?: "layout2d" | "layout3d" | "layoutNd" | string | null;
  dims?: number | null;
  numPivots?: number | null;
  minDegree?: number | null;
  targetStd?: number | null;
  pivots?: Uint32Array | number[] | null;
  seed?: number | bigint | null;
}

export interface RdMdsGeneratorOptions {
  target?: "layout2d" | "layout3d" | "layoutNd" | string | null;
  dims?: number | null;
  eigEps?: number | null;
  tol?: number | null;
  maxIterations?: number | null;
  degreeNormalize?: boolean | null;
  targetStd?: number | null;
  seed?: number | bigint | null;
}

export interface KorenSpectralGeneratorOptions {
  target?: "layout2d" | "layout3d" | "layoutNd" | string | null;
  dims?: number | null;
  targetStd?: number | null;
  tol?: number | null;
  maxIterations?: number | null;
  seed?: number | bigint | null;
}

export interface DogVaradhanOptions {
  degree?: number | null;
  numLandmarks?: number | null;
  minDegree?: number | null;
  jlDim?: number | null;
  lambdaMax?: number | null;
  degreeNormalize?: boolean | null;
  pExp?: number | null;
  epsPsd?: number | null;
  tCoarse?: number | null;
  t1?: number | null;
  t2?: number | null;
  alpha?: number | null;
  betaFloor?: number | null;
  seed?: number | bigint | null;
}

export interface ChebyshevHeatSpmvOptions {
  degree?: number | null;
  lambdaMax?: number | null;
  lanczosSteps?: number | null;
  safetyFactor?: number | null;
  seed?: number | bigint | null;
}

export interface ChebyshevDogJlOptions {
  jlDim?: number | null;
  degree?: number | null;
  degreeNormalize?: boolean | null;
  epsPsd?: number | null;
  t1?: number | null;
  t2?: number | null;
  alpha?: number | null;
  lambdaMax?: number | null;
  lanczosSteps?: number | null;
  safetyFactor?: number | null;
  scaleVec?: Float64Array | number[] | null;
  seed?: number | bigint | null;
}

export interface AnisotropicLaplacianOptions {
  alpha?: number | null;
  beta?: number | null;
  degreeReg?: number | null;
  degreeCap?: number | null;
  reweightedDegreeNorm?: boolean | null;
}

export interface SpectralEigenOptions {
  nModes?: number | null;
  tol?: number | null;
  maxIterations?: number | null;
  maxSweeps?: number | null;
  seed?: number | bigint | null;
}

export interface SpectralEmbeddingOptions {
  dim?: number | null;
  p?: number | null;
  t?: number | null;
  tol?: number | null;
  maxIterations?: number | null;
  eigEps?: number | null;
  seed?: number | bigint | null;
}

export interface TwoHopModeOptions {
  max2hopPerNode?: number | null;
  mode?: TwoHopSamplingMode | string | null;
  maxBridgeDegree?: number | null;
  seed?: number | bigint | null;
}

export interface PerplexityOptions {
  c?: number | null;
  minPerplexity?: number | null;
  maxPerplexity?: number | null;
}

export interface HybridAdaptiveKnnOptions {
  kFactor?: number | null;
  marginGap?: number | null;
  coarseDim?: number | null;
  rerankFactor?: number | null;
  k?: number | null;
  maxRowK?: number | null;
}

export interface KnnProbabilitiesOptions {
  power?: number | null;
  sigmaIters?: number | null;
  sigmaTolerance?: number | null;
  rowNormalize?: boolean | null;
}
`;
