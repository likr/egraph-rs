import { csrGraphOptionsDecl, metricsTypeDecls } from "./dts_types_metrics.mjs";

const coordsViewDecl = `    /**
     * Returns a zero-copy \`Float32Array\` view over the interleaved \`[x0, y0, ...]\`
     * coordinates in WebAssembly linear memory, automatically refreshing the view
     * whenever \`memory.grow\` detaches the underlying \`ArrayBuffer\`.
     */
    coordsView(): Float32Array;
`;

const metricsReportDtsReplacements = [
  [
    /(export class ClusterAreaReport \{[\s\S]*?)toRecord\(\): object;/,
    "$1toRecord(): ClusterAreaReportRecord;",
  ],
  [
    /(export class MetricReport \{[\s\S]*?)toRecord\(\): object;/,
    "$1toRecord(): MetricReportRecord;",
  ],
  [
    /export function evaluateMetrics\([^;]+\): MetricReport;/,
    "export function evaluateMetrics(graph: CsrGraph, layout: Layout2d, targets: readonly (MetricTarget | string)[], options?: MetricConfigOptions | null): MetricReport;\nexport function evaluateMetrics(graph: CsrGraph, layout: Layout2d, targets: readonly (MetricTarget | string)[], numPivots?: number | null, numSamples?: number | null, seed?: number | null, idealEdgeLength?: number | null, communityLabels?: Uint32Array | number[] | null, numLandmarks?: number | null, numTriplets?: number | null, minDegree?: number | null, patchRadius?: number | null, canvasSize?: number | null, padFrac?: number | null, gMax?: number | null, sigmaCells?: number | null, kappa?: number | null, tauFrac?: number | null, boundaryR95?: number | null, stevensGamma?: number | null, nodeWeights?: Float64Array | Float32Array | number[] | null, clusterWeights?: Float64Array | Float32Array | number[] | null, clusterBaselineOptions?: ClusterBaselineOptions | null): MetricReport;",
  ],
];

const gcafFunctionDtsReplacements = [
  [
    /export function clusterAreaFaithfulness\([^;]+\): number;/,
    "export function clusterAreaFaithfulness(layout: Layout2d, communityLabels: Uint32Array | number[], gMax?: number | null, sigmaCells?: number | null, kappa?: number | null, tauFrac?: number | null, boundaryR95?: number | null, stevensGamma?: number | null, nodeWeights?: Float64Array | Float32Array | number[] | null, clusterWeights?: Float64Array | Float32Array | number[] | null): number;",
  ],
  [
    /export function clusterAreaFaithfulnessDetail\([^;]+\): ClusterAreaReport;/,
    "export function clusterAreaFaithfulnessDetail(layout: Layout2d, communityLabels: Uint32Array | number[], gMax?: number | null, sigmaCells?: number | null, kappa?: number | null, tauFrac?: number | null, boundaryR95?: number | null, stevensGamma?: number | null, nodeWeights?: Float64Array | Float32Array | number[] | null, clusterWeights?: Float64Array | Float32Array | number[] | null): ClusterAreaReport;",
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

// Build pkg/index.d.ts from egraph_wasm.d.ts, preserving all core declarations
// while adding CsrGraphOptions and Layout2d.prototype.coordsView().
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

  if (!indexDts.includes("coordsView(): Float32Array;")) {
    const updated = indexDts.replace(
      /(export class Layout2d \{[\s\S]*?coordsPtr\(\): number;\n)/,
      `$1${coordsViewDecl}`,
    );
    if (updated === indexDts) {
      throw new Error(`Failed to inject coordsView() into Layout2d declaration in ${wasmDtsPath}`);
    }
    indexDts = updated;
  }

  if (indexDts.includes("export class LayoutNd {")) {
    const updatedNd = indexDts.replace(
      /(export class LayoutNd \{[\s\S]*?coordsPtr\(\): number;\n)/,
      `$1${coordsViewDecl.replace("[x0, y0, ...]", "[c0, c1, ...]")}`,
    );
    if (updatedNd === indexDts) {
      throw new Error(`Failed to inject coordsView() into LayoutNd declaration in ${wasmDtsPath}`);
    }
    indexDts = updatedNd;
  }

  if (
    indexDts.includes("export class MetricReport {") &&
    !indexDts.includes("export interface MetricReportRecord")
  ) {
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

  indexDts = applyCheckedReplacements(indexDts, metricsReportDtsReplacements, "", wasmDtsPath);

  indexDts = indexDts.replace(
    /export function sampledCrosslessness\([^;]+\): number;/,
    "export function sampledCrosslessness(graph: CsrGraph, layout: Layout2d, numSamples?: number | null, seed?: number | null): number;",
  );

  indexDts = indexDts.replace(
    /export function adjustedPatchNig\([^;]+\): number;/,
    "export function adjustedPatchNig(layout: Layout2d, communityLabels: Uint32Array | number[], patchRadius?: number | null, canvasSize?: number | null, padFrac?: number | null, seed?: number | null): number;",
  );

  indexDts = indexDts.replace(
    /export function adjustedPatchNigMultiscale\([^;]+\): Float64Array;/,
    "export function adjustedPatchNigMultiscale(layout: Layout2d, communityLabels: Uint32Array | number[], radii?: Uint32Array | number[] | null, canvasSize?: number | null, padFrac?: number | null, seed?: number | null): Float64Array;",
  );

  indexDts = applyCheckedReplacements(
    indexDts,
    gcafFunctionDtsReplacements,
    "G-CAF",
    wasmDtsPath,
  );

  indexDts = indexDts.replace(
    /export function landmarkStratifiedRta\([^;]+\): number;/,
    "export function landmarkStratifiedRta(graph: CsrGraph, layout: Layout2d, numLandmarks?: number | null, numTriplets?: number | null, minDegree?: number | null, seed?: number | null): number;",
  );

  indexDts = indexDts.replace(
    /export function landmarkStratifiedRtaDetail\([^;]+\): RtaEvaluation;/,
    "export function landmarkStratifiedRtaDetail(graph: CsrGraph, layout: Layout2d, numLandmarks?: number | null, numTriplets?: number | null, minDegree?: number | null, seed?: number | null): RtaEvaluation;",
  );

  indexDts = indexDts.replace(
    /export function landmarkStratifiedRtaFromLandmarks\([^;]+\): RtaEvaluation;/,
    "export function landmarkStratifiedRtaFromLandmarks(layout: Layout2d, landmarkIds: Uint32Array | number[], landmarkDists: Uint32Array | number[], numTriplets?: number | null, seed?: number | null): RtaEvaluation;",
  );

  indexDts = indexDts.replace(
    /export function prismDistortion\([^;]+\): PrismDistortion;/,
    "export function prismDistortion(originalLayout: Layout2d, adjustedLayout: Layout2d): PrismDistortion;",
  );

  indexDts = indexDts.replace(
    /export function sampledCrossingNumber\([^;]+\): number;/,
    "export function sampledCrossingNumber(graph: CsrGraph, layout: Layout2d, numSamples?: number | null, seed?: number | null): number;",
  );

  indexDts = indexDts.replace(
    /export function sampledScaleNormalizedStress\([^;]+\): number;/,
    "export function sampledScaleNormalizedStress(graph: CsrGraph, layout: Layout2d, numPivots?: number | null, seed?: number | null): number;",
  );

  indexDts = indexDts.replace(
    /export function sampledNp1\([^;]+\): number;/,
    "export function sampledNp1(graph: CsrGraph, layout: Layout2d, numSamples?: number | null, seed?: number | null): number;",
  );

  indexDts = indexDts.replace(
    /export function sampledNp2\([^;]+\): number;/,
    "export function sampledNp2(graph: CsrGraph, layout: Layout2d, numSamples?: number | null, seed?: number | null): number;",
  );

  return indexDts;
}
