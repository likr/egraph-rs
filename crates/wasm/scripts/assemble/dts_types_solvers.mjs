export const solverEnumAndPrimaryTypeDecls = `export type SolverStatus = "initialized" | "running" | "converged" | "completed";
export type ManifoldKind = "hyperbolic" | "hyperbolic_poincare" | "hyperbolic_lorentz" | "spherical" | "torus" | "torus_2d";
export type SuggestedGeometry = "hyperbolic" | "spherical" | "torus" | "euclidean";
export type TsNetBackend = "auto" | "exact" | "barnes_hut" | "fft" | "linear_fft";
export type TwoHopSamplingMode = "degree_ordered_hub_suppressed" | "two_step_random_walk";
export type SgdScheduleKind = "exponential" | "linear" | "quadratic" | "reciprocal" | "constant" | "squared";
export type BarrierSymmetry = "sym_max" | "sym_avg";
export type FieldRepulsionBackend = "ibfft" | "barnes_hut" | "exact";
export type LayeringMethod = "longestPath" | "longest_path" | "networkSimplex" | "network_simplex";
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
  seed?: number | null;
  randomizeTies?: boolean | null;
  nearNullspace?: Float64Array | number[] | null;
}

export interface EffectiveResistanceOptions {
  jlDim?: number | null;
  seed?: number | null;
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
  seed?: number | null;
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

export interface SgdScheduleOptions {
  iterations?: number | null;
  eps?: number | null;
  kind?: SgdScheduleKind | string | null;
  stepScale?: number | null;
  maxMu?: number | null;
  seed?: number | null;
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
  seed?: number | null;
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
  seed?: number | null;
}

export interface StressMajorizationOptions {
  epsilon?: number | null;
  maxIterations?: number | null;
  pcgTolerance?: number | null;
  pcgMaxIterations?: number | null;
  minNorm?: number | null;
}

export interface KamadaKawaiOptions {
  eps?: number | null;
  maxInnerIters?: number | null;
  maxIterations?: number | null;
}

export interface TFdpOptions {
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
  seed?: number | null;
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
  seed?: number | null;
}

export interface SnapTFdpOptions {
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
  seed?: number | null;
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
  seed?: number | null;
}

export interface PacmapGraphOptions {
  nNeighbors?: number | null;
  mnRatio?: number | null;
  fpRatio?: number | null;
  phaseIterations?: readonly [number, number, number] | Uint32Array | number[] | null;
  learningRate?: number | null;
  keepAll1hop?: boolean | null;
  max1hopPerNode?: number | null;
  degreeWeightBeta?: number | null;
  numLandmarks?: number | null;
  seed?: number | null;
}
`;
