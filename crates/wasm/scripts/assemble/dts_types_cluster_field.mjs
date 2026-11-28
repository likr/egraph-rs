export const clusterFieldTypeDecls = `export type CoarseEdgeLengthMode = "conductanceInverse" | "resistanceCurvatureFlow";

export type CoarseDistanceBackend = "rhgh" | "chebyshevDog" | "shortestPath";

export interface CoarseLouvainOptions {
  maxCoarseNodes?: number | null;
  resolution?: number | null;
  resolutionDecay?: number | null;
  maxPasses?: number | null;
  minModularityGain?: number | null;
  maxOuterRounds?: number | null;
  seed?: number | bigint | null;
}

export interface RhghLandmarkOptions {
  gamma?: number | null;
  pExp?: number | null;
  numLandmarks?: number | null;
  minDegree?: number | null;
  numProbes?: number | null;
  relTol?: number | null;
  maxIterations?: number | null;
  normalize1hop?: boolean | null;
  seed?: number | bigint | null;
}

export interface CoarseBackboneOptions extends CoarseLouvainOptions {
  edgeLengthMode?: CoarseEdgeLengthMode | string | null;
  distanceBackend?: CoarseDistanceBackend | string | null;
  alpha?: number | null;
  c0?: number | null;
  kappa?: number | null;
  gammaW?: number | null;
  gammaM?: number | null;
  muOv?: number | null;
  mGap?: number | null;
  coarsePivots?: number | null;
  coarseNegPerNode?: number | null;
  coarseTMax?: number | null;
  rhghGamma?: number | null;
  rhghP?: number | null;
  rcfSteps?: number | null;
  rcfEta?: number | null;
  jitterScale?: number | null;
  gammaTether?: number | null;
  targetStd?: number | null;
  robustQuantile?: number | null;
}

export interface PacmapBackboneOptions extends PacmapGraphOptions, CoarseBackboneOptions {}

export type DensityFieldKernel = "screenedPoisson" | "singleGaussian";

export interface ScreenedPoissonOptions {
  kernel?: DensityFieldKernel | string | null;
  gridSize?: number | null;
  paddingFactor?: number | null;
  marginRatio?: number | null;
  sigmaRatio?: number | null;
  sigma?: number | null;
  screeningLengthRatio?: number | null;
  kappa?: number | null;
  rhoMaxFactor?: number | null;
  rhoMax?: number | null;
  exactAdjoint?: boolean | null;
}

export interface ScreenedPoissonRefinerOptions extends ScreenedPoissonOptions {
  iterations?: number | null;
  stepSize?: number | null;
  momentum?: number | null;
  maxStepRatio?: number | null;
  anchorWeight?: number | null;
  gammaTether?: number | null;
  recenter?: boolean | null;
  energyTol?: number | null;
  seed?: number | bigint | null;
}

export interface Sgd2Options {
  target?: TargetKind | string | null;
  iterations?: number | null;
  eps?: number | null;
  stepScale?: number | null;
  maxMu?: number | null;
  batchSize?: number | null;
  damping?: number | null;
  seed?: number | bigint | null;
}

export interface ForceAtlas2Options {
  target?: "layout2d" | "layout3d" | "layoutNd" | string | null;
  backend?: "exact" | "barnesHut" | string | null;
  iterations?: number | null;
  theta?: number | null;
  useBarnesHut?: boolean | null;
  linlogMode?: boolean | null;
  strongGravity?: boolean | null;
  gravity?: number | null;
  scalingRatio?: number | null;
  edgeWeightInfluence?: number | null;
  outboundAttractionDistribution?: boolean | null;
  jitterTolerance?: number | null;
  maxDisplacement?: number | null;
  preventOverlap?: boolean | null;
  defaultNodeRadius?: number | null;
  nodeRadii?: Float64Array | Float32Array | number[] | null;
  seed?: number | bigint | null;
}

export type LeidenObjective = "modularity" | "cpm";

export interface LeidenOptions {
  objective?: LeidenObjective | string | null;
  resolution?: number | null;
  randomness?: number | null;
  maxPasses?: number | null;
  minQualityGain?: number | null;
  seed?: number | bigint | null;
}

export interface LeidenRecord {
  labels: Uint32Array;
  numCommunities: number;
  quality: number;
  modularity: number;
  passes: number;
}

export type LinLogRepulsion = "edgeRepulsion" | "nodeRepulsion";

export interface LinLogOptions {
  target?: "layout2d" | "layout3d" | "layoutNd" | string | null;
  backend?: "exact" | "barnesHut" | string | null;
  iterations?: number | null;
  repulsionModel?: LinLogRepulsion | string | null;
  attractionExponent?: number | null;
  repulsionExponent?: number | null;
  theta?: number | null;
  useBarnesHut?: boolean | null;
  gravity?: number | null;
  edgeWeightInfluence?: number | null;
  maxDisplacement?: number | null;
  seed?: number | bigint | null;
}

export type FrBackend = "exact" | "barnesHut" | "grid";

export interface FruchtermanReingoldOptions {
  target?: TargetKind | string | null;
  iterations?: number | null;
  backend?: FrBackend | string | null;
  k?: number | null;
  area?: number | null;
  initialTemperature?: number | null;
  gravity?: number | null;
  theta?: number | null;
  seed?: number | bigint | null;
}

export type BackboneLayoutMode = "stressMajorization" | "stress" | "sgd";

export interface BackboneOptions {
  retainRatio?: number | null;
  umstWeightScale?: number | null;
  nonBackboneWeightScale?: number | null;
  useWeights?: boolean | null;
  layoutMode?: BackboneLayoutMode | string | null;
  iterations?: number | null;
  seed?: number | bigint | null;
}

export interface BackboneRecord {
  umstEdges: Uint32Array;
  backboneEdges: Uint32Array;
  simmelianWeights: Float64Array;
  edgeWeights: Float64Array;
}

export type ClusterRefineMethod = "kmeans" | "hac";

export interface ClusterRefineOptions {
  method?: ClusterRefineMethod | string | null;
  maxIterations?: number | null;
  kmeansNInit?: number | null;
  kmeansMaxIterations?: number | null;
  shiftStep?: number | null;
  margin?: number | null;
  seed?: number | bigint | null;
}

export type GibBoxLayoutMode = "squarifiedTreemap" | "treemap" | "forceDirected";
export type GibIntraLayoutMode = "fr" | "sgdFull" | "yifanHu";

export interface GroupInABoxOptions {
  boxMode?: GibBoxLayoutMode | string | null;
  intraMode?: GibIntraLayoutMode | string | null;
  totalWidth?: number | null;
  totalHeight?: number | null;
  boxPadding?: number | null;
  intraIterations?: number | null;
  fdIterations?: number | null;
  fitBounds?: boolean | null;
  clusterWeights?: Float64Array | Float32Array | number[] | null;
  seed?: number | bigint | null;
}

export interface GibBoxRecord {
  clusterId: number;
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  nodeCount: number;
  weight: number;
}

export interface GTreeOptions {
  maxIterations?: number | null;
  padding?: number | null;
  compactTMax?: number | null;
  seed?: number | bigint | null;
}

export type ForbidVariant = "forbid" | "forbidPrime" | "sparseForbid";

export interface ForbidOptions {
  variant?: ForbidVariant | string | null;
  maxScaleIters?: number | null;
  sgdIterations?: number | null;
  sparseK?: number | null;
  overlapWeightK?: number | null;
  scaleStep?: number | null;
  padding?: number | null;
  seed?: number | bigint | null;
}

export interface KMeansOptions {
  k?: number | null;
  maxIterations?: number | null;
  tol?: number | null;
  nInit?: number | null;
  seed?: number | bigint | null;
}

export interface LouvainOptions {
  maxPasses?: number | null;
  resolution?: number | null;
  minModularityGain?: number | null;
  seed?: number | bigint | null;
}

export interface SpectralClusteringOptions {
  k?: number | null;
  maxEigenIterations?: number | null;
  eigenTol?: number | null;
  kmeansMaxIterations?: number | null;
  kmeansTol?: number | null;
  seed?: number | bigint | null;
}

export interface TreemapOptions {
  width?: number | null;
  height?: number | null;
  x?: number | null;
  y?: number | null;
}

export interface ProjectClusteredLayoutOptions {
  spreadRadius?: number | null;
  jitterScale?: number | null;
  seed?: number | bigint | null;
}

export interface PartitionAgreementRecord {
  ari: number;
  ami: number;
  homogeneity: number;
  completeness: number;
  vMeasure: number;
  fmi: number;
}

export interface ClassDensityRecord {
  cdm: number;
  hdm: number;
}

export interface NorMetricsRecord {
  ooNni: number;
  spChA: number;
  gsBbIar: number;
  nmDmImse: number;
  elRsd: number;
}
`;
