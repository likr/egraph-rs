export const forceOverlapDtsReplacements = [
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
    /static snapTFdp\(graph: CsrGraph, options: any\): SgdSolver;/,
    "static snapTFdp(graph: CsrGraph, options?: SnapTFdpOptions | null): SgdSolver;",
  ],
  [
    /static pacmapGraph\(graph: CsrGraph, options: any\): SgdSolver;/,
    "static pacmapGraph(graph: CsrGraph, options?: PacmapGraphOptions | null): SgdSolver;",
  ],
  [
    /static pacmapGraphWithLandmarks\(graph: CsrGraph, landmarks: DogVaradhanLandmarks, options: any\): SgdSolver;/,
    "static pacmapGraphWithLandmarks(graph: CsrGraph, landmarks: DogVaradhanLandmarks, options?: PacmapGraphOptions | null): SgdSolver;",
  ],
  [
    /static pacmapGraphWithLandmarkDistances\(graph: CsrGraph, landmark_distances: Float64Array \| number\[\], num_landmarks: number, options: any\): SgdSolver;/,
    "static pacmapGraphWithLandmarkDistances(graph: CsrGraph, landmarkDistances: Float64Array | number[], numLandmarks: number, options?: PacmapGraphOptions | null): SgdSolver;",
  ],
  [
    /static sgd2\(graph: CsrGraph, options: any\): SgdSolver;/,
    "static sgd2(graph: CsrGraph, options?: Sgd2Options | null): SgdSolver;",
  ],
  [
    /static fromPackedSymmetric\([^;]+\): SgdSolver;/,
    "static fromPackedSymmetric(numNodes: number, src: Uint32Array | number[], tgt: Uint32Array | number[], dists: Float32Array | number[], weights?: Float32Array | number[] | null, isHinge?: Uint8Array | number[] | null, iterationsOrOptions?: number | SgdScheduleOptions | null, eps?: number | null, kind?: SgdScheduleKind | string | null, stepScale?: number | null, seed?: number | null): SgdSolver;",
  ],
  [
    /resetSchedule\([^;]+\): void;/,
    "resetSchedule(iterationsOrOptions?: number | SgdScheduleOptions | null, eps?: number | null, kind?: SgdScheduleKind | string | null, stepScale?: number | null, seed?: number | null): void;",
  ],
  [
    /(export class ForceAtlas2Solver \{[\s\S]*?)_step2d\(layout: Layout2d\): string;/,
    "$1step(layout: Layout2d | LayoutNd): string;\n    runSteps(layout: Layout2d | LayoutNd, steps: number): string;\n    run(layout: Layout2d | LayoutNd): string;\n    _step2d(layout: Layout2d): string;",
  ],
  [
    /(export class ForceAtlas2Solver \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: ForceAtlas2Options | null);",
  ],
  [
    /(export class YifanHuSolver \{[\s\S]*?)_step2d\(layout: Layout2d\): string;/,
    "$1step(layout: Layout2d | LayoutNd): string;\n    runSteps(layout: Layout2d | LayoutNd, steps: number): string;\n    run(layout: Layout2d | LayoutNd): string;\n    _step2d(layout: Layout2d): string;",
  ],
  [
    /(export class YifanHuSolver \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: YifanHuOptions | null);",
  ],
  [
    /static fromRadii\(radii: Float32Array \| number\[\], options: any\): PrismOverlapRemoval;/,
    "static fromRadii(radii: Float32Array | number[], options?: PrismOptions | null): PrismOverlapRemoval;",
  ],
  [
    /(export class PrismOverlapRemoval \{[\s\S]*?)constructor\(half_widths: Float32Array \| number\[\], half_heights: Float32Array \| number\[\], options: any\);/,
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
    /export function generateLayeredConstraints\(graph: CsrGraph, layers: Uint32Array \| number\[\], min_layer_distance\?: number \| null\): SeparationConstraint\[\];/,
    "export function generateLayeredConstraints(graph: CsrGraph, layers: Uint32Array | number[], minLayerDistance?: number | null): SeparationConstraint[];",
  ],
  [
    /export function projectRectangleNoOverlap2d\(layout: Layout2d, half_widths: Float32Array \| number\[\], half_heights: Float32Array \| number\[\]\): void;/,
    "export function projectRectangleNoOverlap2d(layout: Layout2d, halfWidths: Float32Array | number[], halfHeights: Float32Array | number[]): void;",
  ],
  [
    /export function projectClusteredRectangleNoOverlap2d\(layout: Layout2d, labels: Uint32Array \| number\[\], half_widths: Float32Array \| number\[\], half_heights: Float32Array \| number\[\]\): void;/,
    "export function projectClusteredRectangleNoOverlap2d(layout: Layout2d, labels: Uint32Array | number[], halfWidths: Float32Array | number[], halfHeights: Float32Array | number[]): void;",
  ],
  [
    /export function assignLayers\(graph: CsrGraph, method\?: string \| null\): Uint32Array;/,
    "export function assignLayers(graph: CsrGraph, method?: LayeringMethod | string | null): Uint32Array;",
  ],
  [
    /(export class FdebSolver \{[\s\S]*?)_step2d\(layout: Layout2d\): string;/,
    "$1step(layout: Layout2d | LayoutNd): string;\n    runSteps(layout: Layout2d | LayoutNd, steps: number): string;\n    run(layout: Layout2d | LayoutNd): string;\n    _step2d(layout: Layout2d): string;",
  ],
  [
    /(export class FdebSolver \{[\s\S]*?)constructor\(graph: CsrGraph, options: any\);/,
    "$1constructor(graph: CsrGraph, options?: FdebOptions | null);",
  ],
];

export const coarseningFieldDtsReplacements = [
  [
    /static pacmapGraphWithBackbone\(graph: CsrGraph, options: any\): PacmapBackboneResult;/,
    "static pacmapGraphWithBackbone(graph: CsrGraph, options?: PacmapBackboneOptions | null): PacmapBackboneResult;",
  ],
  [
    /export function louvainCoarsen\(graph: CsrGraph, options: any\): CoarsenedHierarchyResult;/,
    "export function louvainCoarsen(graph: CsrGraph, options?: CoarseLouvainOptions | null): CoarsenedHierarchyResult;",
  ],
  [
    /export function projectClusteredLayoutWithRadii\([^;]+\): Layout2d;/,
    "export function projectClusteredLayoutWithRadii(coarseLayout: Layout2d, nodeToCoarse: Uint32Array | number[], clusterRadii: Float64Array | number[], jitterScale?: number | null, seed?: number | null): Layout2d;",
  ],
  [
    /export function alignClusteredCentroids\([^;]+\): void;/,
    "export function alignClusteredCentroids(fineLayout: Layout2d, coarseLayout: Layout2d, nodeToCoarse: Uint32Array | number[], gammaTether?: number | null): void;",
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
    /(export class ScreenedPoissonRefiner \{[\s\S]*?)setCoarseTether\(coarse_layout: Layout2d, node_to_coarse: Uint32Array \| number\[\]\): void;/,
    "$1setCoarseTether(coarseLayout: Layout2d, nodeToCoarse: Uint32Array | number[]): void;",
  ],
];
