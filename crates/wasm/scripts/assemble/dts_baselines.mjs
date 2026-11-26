export const clusterBaselinesDtsReplacements = [
  [
    /(export class LeidenResult \{[\s\S]*?)toRecord\(\): object;/,
    "$1toRecord(): LeidenRecord;",
  ],
  [
    /export function leidenCommunities\([^;]+\): LeidenResult;/,
    "export function leidenCommunities(graph: CsrGraph, options?: LeidenOptions | null): LeidenResult;",
  ],
  [
    /(export class LinLogSolver \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: LinLogOptions | null);",
  ],
  [
    /(export class FruchtermanReingoldSolver \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: FruchtermanReingoldOptions | null);",
  ],
  [
    /(export class BackboneResult \{[\s\S]*?)toRecord\(\): object;/,
    "$1toRecord(): BackboneRecord;",
  ],
  [
    /export function extractQuadrilateralBackbone\([^;]+\): BackboneResult;/,
    "export function extractQuadrilateralBackbone(graph: CsrGraph, options?: BackboneOptions | null): BackboneResult;",
  ],
  [
    /(export class BackboneSolver \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: BackboneOptions | null);",
  ],
  [
    /(export class ClusterRefineSolver \{[\s\S]*?)static fromGraph\([^;]+\): ClusterRefineSolver;/,
    "$1static fromGraph(graph: CsrGraph, clusterLabels: Uint32Array | number[], options?: ClusterRefineOptions | null): ClusterRefineSolver;",
  ],
  [
    /(export class ClusterRefineSolver \{[\s\S]*?)constructor\([^;]+\);/,
    "$1constructor(clusterLabels: Uint32Array | number[], options?: ClusterRefineOptions | null);",
  ],
  [
    /export function refineClusterLayout\([^;]+\): number;/,
    "export function refineClusterLayout(layout: Layout2d, clusterLabels: Uint32Array | number[], options?: ClusterRefineOptions | null): number;",
  ],
  [
    /export function squarifiedTreemap\([^;]+\): Float64Array;/,
    "export function squarifiedTreemap(weights: Float64Array | Float32Array | number[], width?: number | null, height?: number | null, x?: number | null, y?: number | null): Float64Array;",
  ],
  [
    /(export class GroupInABoxSolver \{[\s\S]*?)boxes\(\): Array<any>;/,
    "$1boxes(): GibBoxRecord[];",
  ],
  [
    /(export class GroupInABoxSolver \{[\s\S]*?)constructor\([^;]+\);/,
    "$1constructor(graph: CsrGraph, clusterLabels: Uint32Array | number[], options?: GroupInABoxOptions | null);",
  ],
  [
    /(export class GTreeOverlapRemoval \{[\s\S]*?)static fromRadii\([^;]+\): GTreeOverlapRemoval;/,
    "$1static fromRadii(radii: Float32Array | number[], options?: GTreeOptions | null): GTreeOverlapRemoval;",
  ],
  [
    /(export class GTreeOverlapRemoval \{[\s\S]*?)constructor\([^;]+\);/,
    "$1constructor(halfWidths: Float32Array | number[], halfHeights: Float32Array | number[], options?: GTreeOptions | null);",
  ],
  [
    /(export class ForbidOverlapRemoval \{[\s\S]*?)static fromRadii\([^;]+\): ForbidOverlapRemoval;/,
    "$1static fromRadii(radii: Float32Array | number[], options?: ForbidOptions | null): ForbidOverlapRemoval;",
  ],
  [
    /(export class ForbidOverlapRemoval \{[\s\S]*?)constructor\([^;]+\);/,
    "$1constructor(halfWidths: Float32Array | number[], halfHeights: Float32Array | number[], options?: ForbidOptions | null);",
  ],
  [
    /(export class PartitionAgreementResult \{[\s\S]*?)toRecord\(\): object;/,
    "$1toRecord(): PartitionAgreementRecord;",
  ],
  [
    /(export class ClassDensityResult \{[\s\S]*?)toRecord\(\): object;/,
    "$1toRecord(): ClassDensityRecord;",
  ],
  [
    /(export class NorMetricsResult \{[\s\S]*?)toRecord\(\): object;/,
    "$1toRecord(): NorMetricsRecord;",
  ],
  [
    /export function partitionAgreement\([^;]+\): PartitionAgreementResult;/,
    "export function partitionAgreement(labelsTrue: Uint32Array | number[], labelsPred: Uint32Array | number[]): PartitionAgreementResult;",
  ],
  [
    /export function cqKmeans\([^;]+\): PartitionAgreementResult;/,
    "export function cqKmeans(layout: Layout2d, communityLabels: Uint32Array | number[], nInit?: number | null, maxIters?: number | null, seed?: number | null): PartitionAgreementResult;",
  ],
  [
    /export function cqHac\([^;]+\): PartitionAgreementResult;/,
    "export function cqHac(layout: Layout2d, communityLabels: Uint32Array | number[]): PartitionAgreementResult;",
  ],
  [
    /export function distanceConsistency\([^;]+\): number;/,
    "export function distanceConsistency(layout: Layout2d, labels: Uint32Array | number[]): number;",
  ],
  [
    /export function distributionConsistency\([^;]+\): number;/,
    "export function distributionConsistency(layout: Layout2d, labels: Uint32Array | number[], gridSize?: number | null): number;",
  ],
  [
    /export function classDensityMeasure\([^;]+\): ClassDensityResult;/,
    "export function classDensityMeasure(layout: Layout2d, labels: Uint32Array | number[], gridSize?: number | null, bandwidth?: number | null): ClassDensityResult;",
  ],
  [
    /export function sepmeKnng\([^;]+\): number;/,
    "export function sepmeKnng(layout: Layout2d, labels: Uint32Array | number[], k?: number | null): number;",
  ],
  [
    /export function sepmeRng\([^;]+\): number;/,
    "export function sepmeRng(layout: Layout2d, labels: Uint32Array | number[]): number;",
  ],
  [
    /export function sepmeGab\([^;]+\): number;/,
    "export function sepmeGab(layout: Layout2d, labels: Uint32Array | number[]): number;",
  ],
  [
    /export function sepmeGong\([^;]+\): number;/,
    "export function sepmeGong(layout: Layout2d, labels: Uint32Array | number[], gamma?: number | null): number;",
  ],
  [
    /export function neighborhoodHit\([^;]+\): number;/,
    "export function neighborhoodHit(layout: Layout2d, labels: Uint32Array | number[], k?: number | null): number;",
  ],
  [
    /export function silhouetteScore\([^;]+\): number;/,
    "export function silhouetteScore(layout: Layout2d, labels: Uint32Array | number[]): number;",
  ],
  [
    /export function calinskiHarabaszScore\([^;]+\): number;/,
    "export function calinskiHarabaszScore(layout: Layout2d, labels: Uint32Array | number[]): number;",
  ],
  [
    /export function daviesBouldinScore\([^;]+\): number;/,
    "export function daviesBouldinScore(layout: Layout2d, labels: Uint32Array | number[]): number;",
  ],
  [
    /export function pairwiseDistanceOverlapError\([^;]+\): number;/,
    "export function pairwiseDistanceOverlapError(layout: Layout2d, labels: Uint32Array | number[], numBins?: number | null): number;",
  ],
  [
    /export function sprawlterNn\([^;]+\): number;/,
    "export function sprawlterNn(layout: Layout2d, halfWidths: Float32Array | number[], halfHeights: Float32Array | number[]): number;",
  ],
  [
    /export function norMetrics\([^;]+\): NorMetricsResult;/,
    "export function norMetrics(graph: CsrGraph, origLayout: Layout2d, adjustedLayout: Layout2d, halfWidths: Float32Array | number[], halfHeights: Float32Array | number[]): NorMetricsResult;",
  ],
  [
    /export function norMetricsWithoutGraph\([^;]+\): NorMetricsResult;/,
    "export function norMetricsWithoutGraph(origLayout: Layout2d, adjustedLayout: Layout2d, halfWidths: Float32Array | number[], halfHeights: Float32Array | number[]): NorMetricsResult;",
  ],
];

export const multilevelStressBaselinesDtsReplacements = [
  [
    /export function hde\([^;]+\): Layout2d;/,
    "export function hde(graph: CsrGraph, options?: HdeOptions | null): Layout2d;",
  ],
  [
    /export function hdeNd\([^;]+\): LayoutNd;/,
    "export function hdeNd(graph: CsrGraph, options?: HdeNdOptions | null): LayoutNd;",
  ],
  [
    /static ortmannSparseStress\(graph: CsrGraph, options: any\): SgdSolver;/,
    "static ortmannSparseStress(graph: CsrGraph, options?: OrtmannSparseStressOptions | null): SgdSolver;",
  ],
  [
    /static largevis\(graph: CsrGraph, options: any\): SgdSolver;/,
    "static largevis(graph: CsrGraph, options?: LargeVisOptions | null): SgdSolver;",
  ],
  [
    /static drgraph\(graph: CsrGraph, options: any\): SgdSolver;/,
    "static drgraph(graph: CsrGraph, options?: DrGraphOptions | null): SgdSolver;",
  ],
  [
    /static graphCne\(graph: CsrGraph, options: any\): SgdSolver;/,
    "static graphCne(graph: CsrGraph, options?: GraphCneOptions | null): SgdSolver;",
  ],
  [
    /(export class MaxentStressSolver \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: MaxentStressOptions | null);",
  ],
  [
    /(export class GripSolver \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: GripOptions | null);",
  ],
  [
    /(export class GemSolver \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: GemOptions | null);",
  ],
  [
    /(export class SelLayoutSolver \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: SelOptions | null);",
  ],
  [
    /(export class PfsPrimeOverlapRemoval \{[\s\S]*?)static fromRadii\([^;]+\): PfsPrimeOverlapRemoval;/,
    "$1static fromRadii(radii: Float32Array | number[], options?: PfsPrimeOptions | null): PfsPrimeOverlapRemoval;",
  ],
  [
    /(export class PfsPrimeOverlapRemoval \{[\s\S]*?)constructor\([^;]+\);/,
    "$1constructor(halfWidths: Float32Array | number[], halfHeights: Float32Array | number[], options?: PfsPrimeOptions | null);",
  ],
  [
    /(export class SfcLayoutSolver \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: SfcOptions | null);",
  ],
];
