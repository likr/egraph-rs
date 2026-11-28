export const forceOverlapDtsReplacements = [
  [
    /(export class TFdpSolver \{[\s\S]*?)_step2d\(layout: Layout2d\): string;/,
    "$1step(layout: Layout2d | Layout3d | LayoutNd): string;\n    runSteps(layout: Layout2d | Layout3d | LayoutNd, steps: number): string;\n    run(layout: Layout2d | Layout3d | LayoutNd): string;\n    _step2d(layout: Layout2d): string;",
  ],
  [
    /(export class TFdpSolver \{[\s\S]*?)static fromKnn\([^;]+\): TFdpSolver;/,
    "$1static fromKnn(knn: KnnGraph, options?: TFdpOptions | null): TFdpSolver;\n    static fromKnn(indptr: Uint32Array | number[], indices: Uint32Array | number[], weightsOrOptions?: Float64Array | Float32Array | number[] | TFdpOptions | null, options?: TFdpOptions | null): TFdpSolver;",
  ],
  [
    /(export class TFdpSolver \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: TFdpOptions | null);",
  ],
  [
    /(export class IbUmapSolver \{[\s\S]*?)static fromFuzzySet\([^;]+\): IbUmapSolver;/,
    "$1static fromFuzzySet(probs: SparseProbabilities, degreesOrOptions?: Uint32Array | number[] | IbUmapOptions | null, options?: IbUmapOptions | null): IbUmapSolver;",
  ],
  [
    /(export class IbUmapSolver \{[\s\S]*?)static fromLandmarks\([^;]+\): IbUmapSolver;/,
    "$1static fromLandmarks(graph: CsrGraph, landmarks: DogVaradhanLandmarks, options?: IbUmapOptions | null): IbUmapSolver;\n    static fromLandmarks(graph: CsrGraph, landmarks: DogVaradhanLandmarks, max2hopPerNode?: number | null, options?: IbUmapOptions | null): IbUmapSolver;",
  ],
  [
    /(export class IbUmapSolver \{[\s\S]*?)static fromKnn\([^;]+\): IbUmapSolver;/,
    "$1static fromKnn(knn: KnnGraph, options?: IbUmapOptions | null): IbUmapSolver;",
  ],
  [
    /(export class IbUmapSolver \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: IbUmapOptions | null);",
  ],
  [
    /(export class Sgd2Solver \{[\s\S]*?)_step2d\(layout: Layout2d\): string;/,
    "$1step(layout: AnyLayout): string;\n    runSteps(layout: AnyLayout, steps: number): string;\n    run(layout: AnyLayout): string;\n    runScheduleWindow(layout: AnyLayout, startIter: number, count: number): string;\n    _step2d(layout: Layout2d): string;",
  ],
  [
    /(export class Sgd2Solver \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: Sgd2Options | null);",
  ],
  [
    /(export class Sgd2Solver \{[\s\S]*?)resetSchedule\(options: any\): void;/,
    "$1resetSchedule(options?: SgdScheduleOptions | null): void;",
  ],
  [
    /(export class PacmapGraphSolver \{[\s\S]*?)_step2d\(layout: Layout2d\): string;/,
    "$1step(layout: AnyLayout): string;\n    runSteps(layout: AnyLayout, steps: number): string;\n    run(layout: AnyLayout): string;\n    _step2d(layout: Layout2d): string;",
  ],
  [
    /(export class PacmapGraphSolver \{[\s\S]*?)static fromGraph\(graph: CsrGraph, options: any\): PacmapGraphSolver;/,
    "$1static fromGraph(graph: CsrGraph, options?: PacmapGraphOptions | null): PacmapGraphSolver;",
  ],
  [
    /(export class PacmapGraphSolver \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: PacmapGraphOptions | null);",
  ],
  [
    /(export class PacmapGraphSolver \{[\s\S]*?)static withBackbone\(graph: CsrGraph, options: any\): PacmapBackboneResult;/,
    "$1static withBackbone(graph: CsrGraph, options?: PacmapBackboneOptions | null): PacmapBackboneResult;",
  ],
  [
    /(export class PacmapGraphSolver \{[\s\S]*?)static withLandmarkDistances\([^;]+\): PacmapGraphSolver;/,
    "$1static withLandmarkDistances(graph: CsrGraph, landmarkDistances: Float64Array | number[], numLandmarks: number, options?: PacmapGraphOptions | null): PacmapGraphSolver;",
  ],
  [
    /(export class PacmapGraphSolver \{[\s\S]*?)static withLandmarks\([^;]+\): PacmapGraphSolver;/,
    "$1static withLandmarks(graph: CsrGraph, landmarks: DogVaradhanLandmarks, options?: PacmapGraphOptions | null): PacmapGraphSolver;",
  ],
  [
    /(export class SnapTFdpSolver \{[\s\S]*?)_step2d\(layout: Layout2d\): string;/,
    "$1step(layout: AnyLayout): string;\n    runSteps(layout: AnyLayout, steps: number): string;\n    run(layout: AnyLayout): string;\n    _step2d(layout: Layout2d): string;",
  ],
  [
    /(export class SnapTFdpSolver \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: SnapTFdpOptions | null);",
  ],
  [
    /(export class SgdSolver \{[\s\S]*?)static fromPackedSymmetric\([^;]+\): SgdSolver;/,
    "$1static fromPackedSymmetric(numNodes: number, src: Uint32Array | number[], tgt: Uint32Array | number[], dists: Float32Array | number[], options?: SgdPackedSymmetricOptions | null): SgdSolver;",
  ],
  [
    /(export class SgdSolver \{[\s\S]*?)resetSchedule\([^;]+\): void;/,
    "$1resetSchedule(options?: SgdScheduleOptions | null): void;",
  ],
  [
    /(export class ForceAtlas2Solver \{[\s\S]*?)_step2d\(layout: Layout2d\): string;/,
    "$1step(layout: Layout2d | Layout3d | LayoutNd): string;\n    runSteps(layout: Layout2d | Layout3d | LayoutNd, steps: number): string;\n    run(layout: Layout2d | Layout3d | LayoutNd): string;\n    _step2d(layout: Layout2d): string;",
  ],
  [
    /(export class ForceAtlas2Solver \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: ForceAtlas2Options | null);",
  ],
  [
    /(export class YifanHuSolver \{[\s\S]*?)_step2d\(layout: Layout2d\): string;/,
    "$1step(layout: Layout2d | Layout3d | LayoutNd): string;\n    runSteps(layout: Layout2d | Layout3d | LayoutNd, steps: number): string;\n    run(layout: Layout2d | Layout3d | LayoutNd): string;\n    _step2d(layout: Layout2d): string;",
  ],
  [
    /(export class YifanHuSolver \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: YifanHuOptions | null);",
  ],
  [
    /(export class PrismOverlapRemoval \{[\s\S]*?)static fromRadii\(radii: Float32Array \| number\[\], options: any\): PrismOverlapRemoval;/,
    "$1static fromRadii(radii: Float32Array | number[], options?: PrismOptions | null): PrismOverlapRemoval;",
  ],
  [
    /(export class PrismOverlapRemoval \{[\s\S]*?)constructor\([^;]+\);/,
    "$1constructor(halfWidths: Float32Array | number[], halfHeights: Float32Array | number[], options?: PrismOptions | null);",
  ],
  [
    /(export class TwinDeclutterer \{[\s\S]*?)constructor\(options: any\);/,
    "$1constructor(options?: TwinDeclutterOptions | null);",
  ],
  [
    /export function declutterTwins2d\(layout: Layout2d, options: any\): number;/,
    "export function declutterTwins2d(layout: Layout2d, options?: TwinDeclutterOptions | null): number;",
  ],
  [
    /export function project1d\(coords: Float32Array, constraints: any\): void;/,
    "export function project1d(coords: Float32Array, constraints: readonly (SeparationConstraint | SeparationConstraintLike)[]): void;",
  ],
  [
    /export function projectLayoutAxis\(layout: Layout2d, axis: any, constraints: any\): void;/,
    "export function projectLayoutAxis(layout: Layout2d, axis: LayoutAxis, constraints: readonly (SeparationConstraint | SeparationConstraintLike)[]): void;",
  ],
  [
    /export function generateLayeredConstraints\([^;]+\): SeparationConstraint\[\];/,
    "export function generateLayeredConstraints(graph: CsrGraph, layers: Uint32Array | number[], minLayerDistance?: number | null): SeparationConstraint[];",
  ],
  [
    /export function projectRectangleNoOverlap2d\([^;]+\): void;/,
    "export function projectRectangleNoOverlap2d(layout: Layout2d, halfWidths: Float32Array | number[], halfHeights: Float32Array | number[]): void;",
  ],
  [
    /export function projectClusteredRectangleNoOverlap2d\([^;]+\): void;/,
    "export function projectClusteredRectangleNoOverlap2d(layout: Layout2d, labels: Uint32Array | number[], halfWidths: Float32Array | number[], halfHeights: Float32Array | number[]): void;",
  ],
  [
    /export function assignLayers\([^;]+\): Uint32Array;/,
    "export function assignLayers(graph: CsrGraph, methodOrOptions?: LayeringMethod | { method?: LayeringMethod | null } | null): Uint32Array;",
  ],
  [
    /(export class FdebSolver \{[\s\S]*?)constructor\(polylines: EdgePolylines, options: any\);/,
    "$1constructor(polylines: EdgePolylines, options?: FdebOptions | null);",
  ],
];

export const coarseningFieldDtsReplacements = [
  [
    /(export class CoarseBackbone \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: CoarseBackboneOptions | null);",
  ],
  [
    /export function louvainCoarsen\(graph: CsrGraph, options: any\): CoarsenedHierarchyResult;/,
    "export function louvainCoarsen(graph: CsrGraph, options?: CoarseLouvainOptions | null): CoarsenedHierarchyResult;",
  ],
  [
    /export function projectClusteredLayoutWithRadii\([^;]+\): Layout2d;/,
    "export function projectClusteredLayoutWithRadii(coarseLayout: Layout2d, nodeToCoarse: Uint32Array | number[], clusterRadii: Float64Array | number[], options?: ProjectClusteredLayoutOptions | null): Layout2d;",
  ],
  [
    /export function alignClusteredCentroids\([^;]+\): void;/,
    "export function alignClusteredCentroids(fineLayout: Layout2d, coarseLayout: Layout2d, nodeToCoarse: Uint32Array | number[], options?: { gammaTether?: number | null } | null): void;",
  ],
  [
    /export function computePivotedRhghLandmarks\(graph: CsrGraph, options: any\): RhghLandmarksResult;/,
    "export function computePivotedRhghLandmarks(graph: CsrGraph, options?: RhghLandmarkOptions | null): RhghLandmarksResult;",
  ],
  [
    /export function computePivotedRhghLandmarksResult\(graph: CsrGraph, options: any\): RhghLandmarks;/,
    "export function computePivotedRhghLandmarksResult(graph: CsrGraph, options?: RhghLandmarkOptions | null): RhghLandmarks;",
  ],
  [
    /export function coarseBackboneInit\(graph: CsrGraph, options: any\): CoarseBackboneResult;/,
    "export function coarseBackboneInit(graph: CsrGraph, options?: CoarseBackboneOptions | null): CoarseBackboneResult;",
  ],
  [
    /(export class ScreenedPoissonField2d \{[\s\S]*?)constructor\(options: any\);/,
    "$1constructor(options?: ScreenedPoissonOptions | null);",
  ],
  [
    /(export class ScreenedPoissonRefiner \{[\s\S]*?)static fromGraph\(graph: CsrGraph, options: any\): ScreenedPoissonRefiner;/,
    "$1static fromGraph(graph: CsrGraph, options?: ScreenedPoissonRefinerOptions | null): ScreenedPoissonRefiner;",
  ],
  [
    /(export class ScreenedPoissonRefiner \{[\s\S]*?)constructor\(num_nodes: number, options: any\);/,
    "$1constructor(numNodes: number, options?: ScreenedPoissonRefinerOptions | null);",
  ],
  [
    /(export class ScreenedPoissonRefiner \{[\s\S]*?)setCoarseTether\([^;]+\): void;/,
    "$1setCoarseTether(coarseLayout: Layout2d, nodeToCoarse: Uint32Array | number[]): void;",
  ],
];
