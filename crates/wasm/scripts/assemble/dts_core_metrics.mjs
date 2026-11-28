import { csrGraphOptionsDecl, metricsTypeDecls } from "./dts_types_metrics.mjs";

const coordsViewDecl = `    /**
     * Returns a zero-copy \`Float32Array\` view over the interleaved coordinates
     * in WebAssembly linear memory, refreshing automatically on \`memory.grow\`.
     */
    coordsView(): Float32Array;
`;

const edgeOffsetsViewDecl = `    /**
     * Returns a zero-copy \`Uint32Array\` view over \`edgeOffsets\` in WebAssembly
     * linear memory, refreshing automatically on \`memory.grow\`.
     */
    edgeOffsetsView(): Uint32Array;
`;

const pointsViewDecl = `    /**
     * Returns a zero-copy \`Float32Array\` view over the interleaved control points
     * in WebAssembly linear memory, refreshing automatically on \`memory.grow\`.
     */
    pointsView(): Float32Array;
`;

const edgePolylinesStaticDecls = `    static straightLineFromGraph(graph: CsrGraph, layout: Layout2d | Layout3d | LayoutNd): EdgePolylines;
    static fromParts(edgeOffsets: Uint32Array | number[], points: Layout2d | Layout3d | LayoutNd): EdgePolylines;
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
    /(export class Layout2d \{[\s\S]*?)static init\([^;]+\): Layout2d;/,
    "$1static init(n: number, seedOrOptions?: number | bigint | LayoutInitOptions | null, scale?: number | null): Layout2d;",
  ],
  [
    /(export class Layout3d \{[\s\S]*?)static init\([^;]+\): Layout3d;/,
    "$1static init(n: number, options?: LayoutInitOptions | null): Layout3d;",
  ],
  [
    /(export class LayoutNd \{[\s\S]*?)static init\([^;]+\): LayoutNd;/,
    "$1static init(n: number, dims: number, options?: LayoutInitOptions | null): LayoutNd;",
  ],
  [
    /(export class ArenaOctree3d \{[\s\S]*?)_buildFlat\([^;]+\): void;/,
    "$1build(coordsOrLayout: Float32Array | number[] | Layout3d | LayoutNd): void;\n    buildWeighted(coordsOrLayout: Float32Array | number[] | Layout3d | LayoutNd, masses: Float64Array | number[]): void;\n    _buildFlat(coords: Float32Array | number[]): void;",
  ],
  [
    /(export class ArenaOctree3d \{[\s\S]*?)_buildWeightedFlat\([^;]+\): void;/,
    "$1_buildWeightedFlat(coords: Float32Array | number[], masses: Float64Array | number[]): void;",
  ],
  [
    /(export class DatasetInfo \{[\s\S]*?)toRecord\(\): any;/,
    "$1toRecord(): DatasetInfoRecord;",
  ],
  [
    /export function datasetCatalog\(\): Array<any>;/,
    "export function datasetCatalog(): DatasetInfo[];",
  ],
  [
    /export function generateSbmGraph\([^;]+\): any;/,
    "export function generateSbmGraph(blockSizes: Uint32Array | number[], options?: GenerateSbmGraphOptions | null): SbmGraphResult;",
  ],
  [
    /export function generateWeightedGraph\([^;]+\): CsrGraph;/,
    "export function generateWeightedGraph(numNodes: number, options?: GenerateWeightedGraphOptions | null): CsrGraph;",
  ],
  [
    /export function generateLayeredDag\([^;]+\): CsrGraph;/,
    "export function generateLayeredDag(numLayers: number, nodesPerLayer: number, options?: GenerateLayeredDagOptions | null): CsrGraph;",
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
    "export function clusterAreaFaithfulness(layout: Layout2d, labels: Uint32Array | number[], options?: ClusterAreaFaithfulnessOptions | null): number;",
  ],
  [
    /export function clusterAreaFaithfulnessDetail\([^;]+\): ClusterAreaReport;/,
    "export function clusterAreaFaithfulnessDetail(layout: Layout2d, labels: Uint32Array | number[], options?: ClusterAreaFaithfulnessOptions | null): ClusterAreaReport;",
  ],
];

const layoutTargets = [
  ["2d", "Layout2d"],
  ["3d", "Layout3d"],
  ["Manifold", "ManifoldLayout2d"],
  ["Nd", "LayoutNd"],
];

const clusterLabelMetricSpecs = [
  ["calinskiHarabaszScore", "", "number"],
  ["cqHac", "", "PartitionAgreementResult"],
  ["cqKmeans", ", options?: CqKmeansOptions | null", "PartitionAgreementResult"],
  ["daviesBouldinScore", "", "number"],
  ["distanceConsistency", "", "number"],
  ["distributionConsistency", ", options?: { gridSize?: number | null } | null", "number"],
  ["neighborhoodHit", ", options?: { k?: number | null } | null", "number"],
  ["pairwiseDistanceOverlapError", ", options?: { numBins?: number | null } | null", "number"],
  ["sepmeGab", "", "number"],
  ["sepmeGong", ", options?: { gamma?: number | null } | null", "number"],
  ["sepmeKnng", ", options?: { k?: number | null } | null", "number"],
  ["sepmeRng", "", "number"],
  ["silhouetteScore", "", "number"],
];

const internalHelperDtsReplacements = [];
for (const [fnBase, extraParams, retType] of clusterLabelMetricSpecs) {
  for (const [suffix, layoutType] of layoutTargets) {
    const fnName = `_${fnBase}${suffix}`;
    internalHelperDtsReplacements.push([
      new RegExp(`export function ${fnName}\\([^;]+\\): ${retType};`),
      `export function ${fnName}(layout: ${layoutType}, labels: Uint32Array | number[]${extraParams}): ${retType};`,
    ]);
  }
}
for (const [suffix, layoutType] of layoutTargets) {
  const fnName = `_landmarkStratifiedRtaFromLandmarks${suffix}`;
  internalHelperDtsReplacements.push([
    new RegExp(`export function ${fnName}\\([^;]+\\): RtaEvaluation;`),
    `export function ${fnName}(layout: ${layoutType}, landmarkIds: Uint32Array | number[], landmarkDists: Float64Array | Float32Array | Uint32Array | number[], options?: RtaFromLandmarksOptions | null): RtaEvaluation;`,
  ]);
  internalHelperDtsReplacements.push([
    new RegExp(`export function _landmarkStratifiedRta${suffix}\\([^;]+\\): number;`),
    `export function _landmarkStratifiedRta${suffix}(graph: CsrGraph, layout: ${layoutType}, options?: RtaOptions | null): number;`,
  ]);
  internalHelperDtsReplacements.push([
    new RegExp(`export function _landmarkStratifiedRtaDetail${suffix}\\([^;]+\\): RtaEvaluation;`),
    `export function _landmarkStratifiedRtaDetail${suffix}(graph: CsrGraph, layout: ${layoutType}, options?: RtaOptions | null): RtaEvaluation;`,
  ]);
  internalHelperDtsReplacements.push([
    new RegExp(`export function _sampledNp1${suffix}\\([^;]+\\): number;`),
    `export function _sampledNp1${suffix}(graph: CsrGraph, layout: ${layoutType}, options?: SamplingSeedOptions | null): number;`,
  ]);
  internalHelperDtsReplacements.push([
    new RegExp(`export function _sampledNp2${suffix}\\([^;]+\\): number;`),
    `export function _sampledNp2${suffix}(graph: CsrGraph, layout: ${layoutType}, options?: SamplingSeedOptions | null): number;`,
  ]);
  internalHelperDtsReplacements.push([
    new RegExp(`export function _sampledScaleNormalizedStress${suffix}\\([^;]+\\): number;`),
    `export function _sampledScaleNormalizedStress${suffix}(graph: CsrGraph, layout: ${layoutType}, options?: SampledStressOptions | null): number;`,
  ]);
}

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

  for (const clsName of ["Layout2d", "Layout3d", "LayoutNd", "ManifoldLayout2d"]) {
    if (indexDts.includes(`export class ${clsName} {`)) {
      const pattern = new RegExp(`(export class ${clsName} \\{[\\s\\S]*?coordsPtr\\(\\): number;\\n)`);
      const updated = indexDts.replace(pattern, `$1${coordsViewDecl}`);
      if (updated === indexDts) {
        throw new Error(`Failed to inject coordsView() into ${clsName} declaration in ${wasmDtsPath}`);
      }
      indexDts = updated;
    }
  }

  if (indexDts.includes("export class EdgePolylines {")) {
    indexDts = indexDts.replace(
      /(export class EdgePolylines \{[\s\S]*?)static _fromParts2d/,
      `$1${edgePolylinesStaticDecls}    static _fromParts2d`,
    );
    indexDts = indexDts.replace(
      /(export class EdgePolylines \{[\s\S]*?edgeOffsetsPtr\(\): number;\n)/,
      `$1${edgeOffsetsViewDecl}`,
    );
    indexDts = indexDts.replace(
      /(export class EdgePolylines \{[\s\S]*?pointsPtr\(\): number;\n)/,
      `$1${pointsViewDecl}`,
    );
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
    "export function sampledCrosslessness(graph: CsrGraph, layout: Layout2d, options?: SamplingSeedOptions | null): number;",
  );
  indexDts = indexDts.replace(
    /export function adjustedPatchNig\([^;]+\): number;/,
    "export function adjustedPatchNig(layout: Layout2d, labels: Uint32Array | number[], options?: AdjustedPatchNigOptions | null): number;",
  );
  indexDts = indexDts.replace(
    /export function adjustedPatchNigDetail\([^;]+\): AdjustedPatchNigDetail;/,
    "export function adjustedPatchNigDetail(layout: Layout2d, labels: Uint32Array | number[], options?: AdjustedPatchNigOptions | null): AdjustedPatchNigDetail;",
  );
  indexDts = indexDts.replace(
    /export function adjustedPatchNigMultiscale\([^;]+\): Float64Array;/,
    "export function adjustedPatchNigMultiscale(layout: Layout2d, labels: Uint32Array | number[], options?: AdjustedPatchNigMultiscaleOptions | null): Float64Array;",
  );

  indexDts = applyCheckedReplacements(indexDts, gcafFunctionDtsReplacements, "G-CAF", wasmDtsPath);
  indexDts = applyCheckedReplacements(indexDts, internalHelperDtsReplacements, "internal-metrics", wasmDtsPath);

  indexDts = indexDts.replace(
    /export function prismDistortion\([^;]+\): PrismDistortion;/,
    "export function prismDistortion(originalLayout: Layout2d, adjustedLayout: Layout2d): PrismDistortion;",
  );
  indexDts = indexDts.replace(
    /export function sampledCrossingNumber\([^;]+\): number;/,
    "export function sampledCrossingNumber(graph: CsrGraph, layout: Layout2d, options?: SamplingSeedOptions | null): number;",
  );
  indexDts = indexDts.replace(
    /export function _evaluateMetrics2d\([^;]+\): MetricReport;/,
    "export function _evaluateMetrics2d(graph: CsrGraph, layout: Layout2d, targets: readonly (MetricTarget | string)[], options?: MetricConfigOptions | null): MetricReport;",
  );
  indexDts = indexDts.replace(
    /export function _evaluateMetrics2dWithOriginal\([^;]+\): MetricReport;/,
    "export function _evaluateMetrics2dWithOriginal(graph: CsrGraph, layout: Layout2d, targets: readonly (MetricTarget | string)[], options: MetricConfigOptions | null | undefined, originalLayout: Layout2d): MetricReport;",
  );
  indexDts = indexDts.replace(
    /export function _evaluateMetrics3d\([^;]+\): MetricReport;/,
    "export function _evaluateMetrics3d(graph: CsrGraph, layout: Layout3d, targets: readonly (MetricTarget | string)[], options?: MetricConfigOptions | null): MetricReport;",
  );
  indexDts = indexDts.replace(
    /export function _evaluateMetricsManifold\([^;]+\): MetricReport;/,
    "export function _evaluateMetricsManifold(graph: CsrGraph, layout: ManifoldLayout2d, targets: readonly (MetricTarget | string)[], options?: MetricConfigOptions | null): MetricReport;",
  );
  indexDts = indexDts.replace(
    /export function _evaluateMetricsNd\([^;]+\): MetricReport;/,
    "export function _evaluateMetricsNd(graph: CsrGraph, layout: LayoutNd, targets: readonly (MetricTarget | string)[], options?: MetricConfigOptions | null): MetricReport;",
  );

  return indexDts;
}
