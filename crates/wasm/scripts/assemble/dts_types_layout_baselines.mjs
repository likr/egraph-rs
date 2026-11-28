export const layoutBaselineAndProgressiveTypeDecls = `export interface HdeOptions {
  target?: "layout2d" | "layout3d" | "layoutNd" | string | null;
  dims?: number | null;
  dimensions?: number | null;
  seed?: number | bigint | null;
  tol?: number | null;
  maxSweeps?: number | null;
  normalize?: boolean | null;
  pivots?: Uint32Array | number[] | null;
}

export interface HdeNdOptions extends HdeOptions {
  outputDims?: number | null;
}

export type OrtmannPivotStrategy = "kmeansSp" | "maxminRandomSp" | "maxminSp";

export interface OrtmannSparseStressOptions {
  target?: TargetKind | string | null;
  numPivots?: number | null;
  pivotStrategy?: OrtmannPivotStrategy | string | null;
  iterations?: number | null;
  etaMax?: number | null;
  etaMin?: number | null;
  eps?: number | null;
  seed?: number | bigint | null;
}

export interface LargeVisOptions {
  target?: TargetKind | string | null;
  iterations?: number | null;
  negativeSamples?: number | null;
  gamma?: number | null;
  initialLr?: number | null;
  minLr?: number | null;
  degreeExponent?: number | null;
  seed?: number | bigint | null;
}

export interface DrGraphOptions {
  target?: TargetKind | string | null;
  kOrder?: number | null;
  negativeSamples?: number | null;
  b?: number | null;
  gammaCoarse?: number | null;
  gammaFine?: number | null;
  coarsenStopRatio?: number | null;
  iterations?: number | null;
  initialLr?: number | null;
  minLr?: number | null;
  seed?: number | bigint | null;
}

export type GraphCneObjective = "infoNce" | "negTSne";

export interface GraphCneOptions {
  target?: TargetKind | string | null;
  objective?: GraphCneObjective | string | null;
  negativeSamples?: number | null;
  iterations?: number | null;
  initialLr?: number | null;
  minLr?: number | null;
  repulsionWeight?: number | null;
  rowNormalize?: boolean | null;
  seed?: number | bigint | null;
}

export interface MaxentStressOptions {
  target?: "layout2d" | "layout3d" | "layoutNd" | string | null;
  backend?: "exact" | "barnesHut" | string | null;
  q?: number | null;
  alphaInit?: number | null;
  alphaDecay?: number | null;
  alphaMin?: number | null;
  stages?: number | null;
  stepsPerStage?: number | null;
  tol?: number | null;
  cgTol?: number | null;
  cgMaxIters?: number | null;
  theta?: number | null;
  useBarnesHut?: boolean | null;
  epsSq?: number | null;
  seed?: number | bigint | null;
}

export interface GripOptions {
  roundsMin?: number | null;
  roundsMax?: number | null;
  localNeighborhoodSize?: number | null;
  repulsionScale?: number | null;
  edgeLength?: number | null;
  tempR?: number | null;
  tempS?: number | null;
  seed?: number | bigint | null;
}

export interface GemOptions {
  edgeLength?: number | null;
  gravity?: number | null;
  tInit?: number | null;
  tMax?: number | null;
  tMin?: number | null;
  maxIterations?: number | null;
  oscillationAngle?: number | null;
  rotationAngle?: number | null;
  oscillationSensitivity?: number | null;
  rotationSensitivity?: number | null;
  perturbation?: number | null;
  seed?: number | bigint | null;
}

export type SelWeightMapping = "sigmoid" | "linear" | "tanh" | "customLogistic" | "binary";
export type SelLayoutBackend = "fr" | "tFdp";

export interface SelOptions {
  treeDepth?: number | null;
  pqMaxPops?: number | null;
  mapping?: SelWeightMapping | string | null;
  layoutBackend?: SelLayoutBackend | string | null;
  iterations?: number | null;
  seed?: number | bigint | null;
  frConfig?: FruchtermanReingoldOptions | null;
  tFdpConfig?: TFdpOptions | null;
}

export interface PfsPrimeOptions {
  padding?: number | null;
  seed?: number | bigint | null;
}

export type SfcCurveKind = "hilbert" | "morton";

export interface SfcOptions {
  curve?: SfcCurveKind | string | null;
  scale?: number | null;
  resolutionOrder?: number | null;
  seed?: number | bigint | null;
}

export interface YifanHuOptions {
  target?: "layout2d" | "layout3d" | "layoutNd" | string | null;
  backend?: "exact" | "barnesHut" | string | null;
  maxIterationsPerLevel?: number | null;
  stepSize?: number | null;
  stepRatio?: number | null;
  relativeStrength?: number | null;
  optimalDistance?: number | null;
  theta?: number | null;
  minCoarsenSize?: number | null;
  minCoarsenRatio?: number | null;
  convergenceTolerance?: number | null;
  seed?: number | bigint | null;
}

export interface PrismOptions {
  maxIterations?: number | null;
  maxScaleFactor?: number | null;
  cgMaxIters?: number | null;
  cgTolerance?: number | null;
  padding?: number | null;
  seed?: number | bigint | null;
}

export interface TwinDeclutterOptions {
  iterations?: number | null;
  kNeighbors?: number | null;
  minRelativeSeparation?: number | null;
  stiffness?: number | null;
  anchorWeight?: number | null;
  stepScale?: number | null;
  seed?: number | bigint | null;
}

export interface SeparationConstraintLike {
  left: number;
  right: number;
  gap: number;
}

export interface FdebOptions {
  cycles?: number | null;
  s0?: number | null;
  stepsPerCycle?: number | null;
  sStep?: number | null;
  iStep?: number | null;
  springK?: number | null;
  minimumEdgeCompatibility?: number | null;
}

export interface ProgressiveSolverLike<TLayout = AnyLayout | EdgePolylines> {
  readonly isFinished: boolean;
  readonly iteration: number;
  readonly totalIterations: number | undefined;
  readonly status: SolverStatus | string;
  step(layout: TLayout): SolverStatus | string;
  runSteps?(layout: TLayout, steps: number): SolverStatus | string;
}

export interface ProgressiveFrameContext<TLayout = AnyLayout | EdgePolylines> {
  iteration: number;
  totalIterations: number | undefined;
  status: SolverStatus | string;
  layout: TLayout;
}

export interface RunProgressiveOptions {
  frameBudgetMs?: number;
  maxStepsPerFrame?: number;
  stepsPerYield?: number;
  signal?: AbortSignal;
}

/**
 * Runs a cooperative iterative solver progressively within a per-frame time budget,
 * yielding back to the browser event loop via \`requestAnimationFrame\` (or \`setTimeout\`
 * in Node.js) between batches.
 */
export function runProgressive<TLayout = AnyLayout | EdgePolylines>(
  solver: ProgressiveSolverLike<TLayout>,
  layout: TLayout,
  onFrame?: ((ctx: ProgressiveFrameContext<TLayout>) => void) | null,
  options?: RunProgressiveOptions | null,
): Promise<ProgressiveFrameContext<TLayout>>;

`;
