import { csrGraphOptionsDecl, metricsTypeDecls } from "./dts_types_metrics.mjs";

const coordsViewDecl = `    /**
     * Returns a zero-copy \`Float32Array\` view over the interleaved coordinates
     * in WebAssembly linear memory, refreshing automatically on \`memory.grow\`.
     */
    coordsView(): Float32Array;
`;

const labelsViewDecl = `    /**
     * Returns a zero-copy \`Uint32Array\` view over \`labels\` in WebAssembly linear memory.
     */
    labelsView(): Uint32Array;
`;

const centroidsViewDecl = `    /**
     * Returns a zero-copy \`Float64Array\` view over \`centroids\` in WebAssembly linear memory.
     */
    centroidsView(): Float64Array;
`;

const coreClassDtsReplacements = [
  [
    /(export class ClusterAreaReport \{[\s\S]*?)toRecord\(\): object;/,
    "$1toRecord(): ClusterAreaReportRecord;",
  ],
  [
    /(export class MetricReport \{[\s\S]*?)toRecord\(\): object;/,
    "$1toRecord(): MetricReportRecord;",
  ],
  [
    /(export class StressEvaluation \{[\s\S]*?)toRecord\(\): object;/,
    "$1toRecord(): StressEvaluationRecord;",
  ],
  [
    /(export class ArenaOctree3d \{[\s\S]*?)_buildFlat\([^;]+\): void;/,
    "$1build(coordsOrLayout: Float32Array | number[] | LayoutNd): void;\n    buildWeighted(coordsOrLayout: Float32Array | number[] | LayoutNd, masses: Float64Array | number[]): void;\n    _buildFlat(coords: Float32Array | number[]): void;",
  ],
  [
    /(export class DynamicKdTree \{[\s\S]*?)constructor\([^;]+\);/,
    "$1constructor(layout: LayoutNd);\n    constructor(coords: Float32Array | number[], n: number, dims: number);",
  ],
  [
    /(export class DynamicKdTree \{[\s\S]*?)kNearest\([^;]+\): Array<any>;/,
    "$1kNearest(query: Float32Array | number[], k: number): KdNeighbor[];",
  ],
  [
    /(export class DynamicKdTree \{[\s\S]*?)kNearestExcludingSelf\([^;]+\): Array<any>;/,
    "$1kNearestExcludingSelf(pointIdx: number, k: number): KdNeighbor[];",
  ],
  [
    /(export class DynamicKdTree \{[\s\S]*?)nearestExcludingSelf\([^;]+\): any;/,
    "$1nearestExcludingSelf(pointIdx: number): KdNeighbor | undefined;",
  ],
  [
    /(export class DynamicKdTree \{[\s\S]*?)withinRadius\([^;]+\): Array<any>;/,
    "$1withinRadius(query: Float32Array | number[], radius: number): KdNeighbor[];",
  ],
  [
    /(export class DynamicKdTree \{[\s\S]*?)withinRadiusSq\([^;]+\): Array<any>;/,
    "$1withinRadiusSq(query: Float32Array | number[], radiusSq: number): KdNeighbor[];",
  ],
];

const gcafFunctionDtsReplacements = [
  [
    /export function clusterAreaFaithfulness\([^;]+\): number;/,
    "export function clusterAreaFaithfulness(layout: Layout2d, labels: Uint32Array | number[], gMax?: number | null, sigmaCells?: number | null, kappa?: number | null, tauFrac?: number | null, boundaryR95?: number | null, stevensGamma?: number | null, nodeWeights?: Float64Array | Float32Array | number[] | null, clusterWeights?: Float64Array | Float32Array | number[] | null): number;",
  ],
  [
    /export function clusterAreaFaithfulnessDetail\([^;]+\): ClusterAreaReport;/,
    "export function clusterAreaFaithfulnessDetail(layout: Layout2d, labels: Uint32Array | number[], gMax?: number | null, sigmaCells?: number | null, kappa?: number | null, tauFrac?: number | null, boundaryR95?: number | null, stevensGamma?: number | null, nodeWeights?: Float64Array | Float32Array | number[] | null, clusterWeights?: Float64Array | Float32Array | number[] | null): ClusterAreaReport;",
  ],
];

export function applyCheckedReplacements(dts, replacements, domainLabel, wasmDtsPath) {
  let updatedDts = dts;
  const labelPrefix = domainLabel ? `${domainLabel} ` : "";
  for (const [pattern, replacement] of replacements) {
    const updated = updatedDts.replace(pattern, replacement);
    if (updated === updatedDts) {
      throw new Error(
        `Failed to refine ${labelPrefix}TypeScript declaration matching ${pattern} in ${wasmDtsPath}`,
      );
    }
    updatedDts = updated;
  }
  return updatedDts;
}

export function applyCoreAndMetricsDts(wasmDts, wasmDtsPath) {
  let indexDts = wasmDts;
  if (!indexDts.includes('/// <reference lib="esnext.disposable" />')) {
    indexDts = `/// <reference lib="esnext.disposable" />\n${indexDts}`;
  }

  if (!indexDts.includes("export interface CsrGraphOptions")) {
    indexDts = indexDts.replace(
      "export class CsrGraph {",
      `${csrGraphOptionsDecl}export class CsrGraph {`,
    );
  }

  indexDts = indexDts.replace(
    /static fromCsr\([^;]+\): CsrGraph;/,
    `static fromCsr(indptr: Uint32Array | number[], indices: Uint32Array | number[], weights?: Float32Array | number[] | null, options?: boolean | CsrGraphOptions | null, sanitize?: boolean | null): CsrGraph;`,
  );

  indexDts = indexDts.replace(
    /static fromEdgeList\([^;]+\): CsrGraph;/,
    `static fromEdgeList(numNodes: number, edges: Uint32Array | number[], weights?: Float32Array | number[] | null, options?: boolean | CsrGraphOptions | null, sanitize?: boolean | null): CsrGraph;`,
  );

  for (const clsName of ["Layout2d", "LayoutNd", "ManifoldLayout2d"]) {
    if (indexDts.includes(`export class ${clsName} {`)) {
      const pattern = new RegExp(`(export class ${clsName} \\{[\\s\\S]*?coordsPtr\\(\\): number;\\n)`);
      const updated = indexDts.replace(pattern, `$1${coordsViewDecl}`);
      if (updated === indexDts) {
        throw new Error(`Failed to inject coordsView() into ${clsName} declaration in ${wasmDtsPath}`);
      }
      indexDts = updated;
    }
  }

  for (const clsName of [
    "LouvainResult",
    "LeidenResult",
    "KMeansResult",
    "LabelPropagationResult",
    "SpectralClusteringResult",
  ]) {
    if (indexDts.includes(`export class ${clsName} {`)) {
      const pattern = new RegExp(`(export class ${clsName} \\{[\\s\\S]*?labelsPtr\\(\\): number;\\n)`);
      indexDts = indexDts.replace(pattern, `$1${labelsViewDecl}`);
    }
  }

  if (indexDts.includes("export class KMeansResult {")) {
    indexDts = indexDts.replace(
      /(export class KMeansResult \{[\s\S]*?centroidsPtr\(\): number;\n)/,
      `$1${centroidsViewDecl}`,
    );
  }

  if (indexDts.includes("export class MetricReport {") && !indexDts.includes("export interface MetricReportRecord")) {
    const metricDocMarker = "/**\n * Selective quality metric report";
    if (indexDts.includes(metricDocMarker)) {
      indexDts = indexDts.replace(metricDocMarker, `${metricsTypeDecls}${metricDocMarker}`);
    } else {
      indexDts = indexDts.replace(
        "export class MetricReport {",
        `${metricsTypeDecls}export class MetricReport {`,
      );
    }
  }

  indexDts = applyCheckedReplacements(indexDts, coreClassDtsReplacements, "core/spatial", wasmDtsPath);

  indexDts = indexDts.replace(
    /export function sampledCrosslessness\([^;]+\): number;/,
    "export function sampledCrosslessness(graph: CsrGraph, layout: Layout2d, numSamples?: number | null, seed?: number | null): number;",
  );
  indexDts = indexDts.replace(
    /export function adjustedPatchNig\([^;]+\): number;/,
    "export function adjustedPatchNig(layout: Layout2d, labels: Uint32Array | number[], patchRadius?: number | null, canvasSize?: number | null, padFrac?: number | null, seed?: number | null): number;",
  );
  indexDts = indexDts.replace(
    /export function adjustedPatchNigDetail\([^;]+\): AdjustedPatchNigDetail;/,
    "export function adjustedPatchNigDetail(layout: Layout2d, labels: Uint32Array | number[], patchRadius?: number | null, canvasSize?: number | null, padFrac?: number | null, seed?: number | null): AdjustedPatchNigDetail;",
  );
  indexDts = indexDts.replace(
    /export function adjustedPatchNigMultiscale\([^;]+\): Float64Array;/,
    "export function adjustedPatchNigMultiscale(layout: Layout2d, labels: Uint32Array | number[], radii?: Uint32Array | number[] | null, canvasSize?: number | null, padFrac?: number | null, seed?: number | null): Float64Array;",
  );

  indexDts = applyCheckedReplacements(indexDts, gcafFunctionDtsReplacements, "G-CAF", wasmDtsPath);

  indexDts = indexDts.replace(
    /export function prismDistortion\([^;]+\): PrismDistortion;/,
    "export function prismDistortion(originalLayout: Layout2d, adjustedLayout: Layout2d): PrismDistortion;",
  );
  indexDts = indexDts.replace(
    /export function sampledCrossingNumber\([^;]+\): number;/,
    "export function sampledCrossingNumber(graph: CsrGraph, layout: Layout2d, numSamples?: number | null, seed?: number | null): number;",
  );
  indexDts = indexDts.replace(
    /export function _evaluateMetrics2d\([^;]+\): MetricReport;/,
    "export function _evaluateMetrics2d(graph: CsrGraph, layout: Layout2d, targets: readonly (MetricTarget | string)[], options?: number | MetricConfigOptions | null, ...args: unknown[]): MetricReport;",
  );
  indexDts = indexDts.replace(
    /export function _evaluateMetrics2dWithOriginal\([^;]+\): MetricReport;/,
    "export function _evaluateMetrics2dWithOriginal(graph: CsrGraph, layout: Layout2d, targets: readonly (MetricTarget | string)[], options: number | MetricConfigOptions | null | undefined, originalLayout: Layout2d): MetricReport;",
  );
  indexDts = indexDts.replace(
    /export function _evaluateMetricsManifold\([^;]+\): MetricReport;/,
    "export function _evaluateMetricsManifold(graph: CsrGraph, layout: ManifoldLayout2d, targets: readonly (MetricTarget | string)[], options?: number | MetricConfigOptions | null, ...args: unknown[]): MetricReport;",
  );
  indexDts = indexDts.replace(
    /export function _evaluateMetricsNd\([^;]+\): MetricReport;/,
    "export function _evaluateMetricsNd(graph: CsrGraph, layout: LayoutNd, targets: readonly (MetricTarget | string)[], options?: number | MetricConfigOptions | null, ...args: unknown[]): MetricReport;",
  );

  return indexDts;
}
