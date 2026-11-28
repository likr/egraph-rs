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
    /(export class LinLogSolver \{[\s\S]*?)_step2d\(layout: Layout2d\): string;/,
    "$1step(layout: Layout2d | Layout3d | LayoutNd): string;\n    runSteps(layout: Layout2d | Layout3d | LayoutNd, steps: number): string;\n    run(layout: Layout2d | Layout3d | LayoutNd): string;\n    energy(layout: Layout2d | Layout3d | LayoutNd): number;\n    _step2d(layout: Layout2d): string;",
  ],
  [
    /(export class LinLogSolver \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: LinLogOptions | null);",
  ],
  [
    /(export class FruchtermanReingoldSolver \{[\s\S]*?)_step2d\(layout: Layout2d\): string;/,
    "$1step(layout: AnyLayout): string;\n    runSteps(layout: AnyLayout, steps: number): string;\n    run(layout: AnyLayout): string;\n    _step2d(layout: Layout2d): string;",
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
    "$1static fromGraph(graph: CsrGraph, labels: Uint32Array | number[], options?: ClusterRefineOptions | null): ClusterRefineSolver;",
  ],
  [
    /(export class ClusterRefineSolver \{[\s\S]*?)constructor\([^;]+\);/,
    "$1constructor(labels: Uint32Array | number[], options?: ClusterRefineOptions | null);",
  ],
  [
    /export function refineClusterLayout\([^;]+\): number;/,
    "export function refineClusterLayout(layout: Layout2d, labels: Uint32Array | number[], options?: ClusterRefineOptions | null): number;",
  ],
  [
    /export function squarifiedTreemap\([^;]+\): Float64Array;/,
    "export function squarifiedTreemap(weights: Float64Array | Float32Array | number[], options?: TreemapOptions | null): Float64Array;",
  ],
  [
    /(export class GroupInABoxSolver \{[\s\S]*?)boxes\(\): Array<any>;/,
    "$1boxes(): GibBoxRecord[];",
  ],
  [
    /(export class GroupInABoxSolver \{[\s\S]*?)constructor\([^;]+\);/,
    "$1constructor(graph: CsrGraph, labels: Uint32Array | number[], options?: GroupInABoxOptions | null);",
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
    /export function classDensityMeasure\([^;]+\): ClassDensityResult;/,
    "export function classDensityMeasure(layout: Layout2d, labels: Uint32Array | number[], options?: ClassDensityOptions | null): ClassDensityResult;",
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
    /(export class Hde \{[\s\S]*?)generate\(graph: CsrGraph\): any;/,
    "$1generate(graph: CsrGraph): Layout2d | Layout3d | LayoutNd;",
  ],
  [
    /(export class Hde \{[\s\S]*?)constructor\(options: any\);/,
    "$1constructor(options?: HdeOptions | null);",
  ],
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
    /(export class LargeVisSolver \{[\s\S]*?)_step2d\(layout: Layout2d\): string;/,
    "$1step(layout: AnyLayout): string;\n    runSteps(layout: AnyLayout, steps: number): string;\n    run(layout: AnyLayout): string;\n    _step2d(layout: Layout2d): string;",
  ],
  [
    /(export class LargeVisSolver \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: LargeVisOptions | null);",
  ],
  [
    /(export class DrGraphSolver \{[\s\S]*?)_step2d\(layout: Layout2d\): string;/,
    "$1step(layout: AnyLayout): string;\n    runSteps(layout: AnyLayout, steps: number): string;\n    run(layout: AnyLayout): string;\n    _step2d(layout: Layout2d): string;",
  ],
  [
    /(export class DrGraphSolver \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: DrGraphOptions | null);",
  ],
  [
    /(export class GraphCneSolver \{[\s\S]*?)_step2d\(layout: Layout2d\): string;/,
    "$1step(layout: AnyLayout): string;\n    runSteps(layout: AnyLayout, steps: number): string;\n    run(layout: AnyLayout): string;\n    _step2d(layout: Layout2d): string;",
  ],
  [
    /(export class GraphCneSolver \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: GraphCneOptions | null);",
  ],
  [
    /(export class MaxentStressSolver \{[\s\S]*?)_step2d\(layout: Layout2d\): string;/,
    "$1step(layout: Layout2d | Layout3d | LayoutNd): string;\n    runSteps(layout: Layout2d | Layout3d | LayoutNd, steps: number): string;\n    run(layout: Layout2d | Layout3d | LayoutNd): string;\n    _step2d(layout: Layout2d): string;",
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
    /(export class SfcLayout \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: SfcOptions | null);",
  ],
];
