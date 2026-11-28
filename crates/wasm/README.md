# egraph

Fast, strictly single-threaded graph drawing and layout library implemented in pure Rust and WebAssembly (`wasm-bindgen`), distributed as a pure ESM package sharing a single `.wasm` binary across browsers, bundlers, and Node.js 18+.

## Install

```bash
npm install egraph
```

## Architecture Highlights (v6)

- **Pure ESM & Single `.wasm`**: Modern `"exports"` entrypoints for browsers/bundlers (`./pkg/index.js`) and Node.js 18+ (`./pkg/egraph_wasm.node.js`), sharing a single `wasm-opt -Oz` optimized `.wasm` binary.
- **Strictly Single-Threaded**: Pure CPU Rust with deterministic seeded execution (`seed = 42` default), zero `rayon`, zero `std::thread`, zero Web Workers, and zero `SharedArrayBuffer`.
- **Zero-Copy Linear Memory Views**: `coordsView()` on `Layout2d`, `Layout3d`, `LayoutNd`, and `ManifoldLayout2d` (plus `edgeOffsetsView()` / `pointsView()` on `EdgePolylines`, `labelsView()` on all 5 clustering result classes, and `centroidsView()` on `KMeansResult`) returns a cached typed-array view directly over WebAssembly linear memory and automatically refreshes if `memory.grow` detaches the `ArrayBuffer`.
- **Target-Checked Cooperative Execution & TS Contracts**: TypeScript interfaces (`LayoutStorage`, `IterativeSolver`, `LayoutGenerator`, `LayoutTransform`), explicit `target` (`"layout2d" | "layout3d" | "layoutNd" | "manifold2d"`), `step(layout)`, `runSteps(layout, k)`, `run(layout)`, and `runProgressive(solver, layout, onFrame, options?)` across `Layout2d`, `Layout3d`, `LayoutNd`, `ManifoldLayout2d`, and `EdgePolylines`.
- **Strict `camelCase`, `BigInt` Seeds, `.code` Errors & `f32` CSR**: Canonical `camelCase` option literals (`"barnesHut"`, `"fitFft"`, `"linearFft"`, `"ibFft"`, `"twoStepRandomWalk"`, `"hyperbolicLorentz"`), `number | bigint` 64-bit seeds, structured `Error` `.code` (`EgraphErrorCode`), and strict `weightDtype: "float32"` validation on `CsrGraph`.
- **Track A, Track B, Separated Solvers, 3D & Non-Euclidean Engines**: `TsNetSolver.pure2Hop({ backend: "auto2d" })` / `fromGraphRowNormalized`, `SgdSolver.sparseBarrierDog`, separated embedding classes (`Sgd2Solver`, `PacmapGraphSolver`, `LargeVisSolver`, `DrGraphSolver`, `GraphCneSolver`, `SnapTFdpSolver`), `TFdpSolver`, `IbUmapSolver`, `GuMapSolver`, `Layout3d` + `ArenaOctree3d`, `DynamicKdTree`, `EdgePolylines` + `FdebSolver(polylines)`, `ManifoldLayout2d` (`"hyperbolicLorentz"`, `toPoincareLayout`, `ManifoldMetric`, `torusSpectralPhaseInit`, `torusAutoPan`, `torusEdgeSegments`, geodesic crossings), `LayoutGenerator`s (`ClassicalMds`, `PivotMds`, `Hde`, `RdMds`, `KorenSpectral`, `CoarseBackbone`, `SfcLayout`), `LayoutTransform`s (`PfsPrimeOverlapRemoval`, `TwinDeclutterer` `.apply()`), `MetricReport` (`toRecord()`), `listDatasets` / synthetic generators, `DijkstraWorkspace`, Laplacian builders, and `ChebyshevDogKernel`.

## Quick Start (Browser / Bundler ESM)

```js
import init, {
  CsrGraph,
  Layout2d,
  TsNetSolver,
  evaluateMetrics,
  pivotMdsInit,
  runProgressive,
} from "egraph";

await init();

// Build an undirected CSR graph (0 - 1 - 2 - 3 - 0)
const indptr = new Uint32Array([0, 2, 4, 6, 8]);
const indices = new Uint32Array([1, 3, 0, 2, 1, 3, 2, 0]);
const graph = CsrGraph.fromCsr(indptr, indices);

// Initialize with normalized PivotMDS (f32 Layout2d)
const layout = pivotMdsInit(graph, { numPivots: 4, seed: 42 });

// Run Track A Pure 1+2-Hop Adaptive tsNET progressively in requestAnimationFrame
const solver = TsNetSolver.pure2Hop(graph, {
  backend: "auto2d",
  seed: 42,
});

await runProgressive(
  solver,
  layout,
  ({ iteration, totalIterations }) => {
    const xy = layout.coordsView(); // Zero-copy Float32Array [x0, y0, x1, y1, ...]
    console.log(`frame ${iteration}/${totalIterations}, node0=(${xy[0]}, ${xy[1]})`);
  },
  { frameBudgetMs: 10 },
);

// Selectively evaluate quality metrics -> MetricReport
const report = evaluateMetrics(graph, layout, ["sns", "np2", "crossingNumber"]);
console.log(report.toRecord());
```

## Node.js 18+ ESM

The same `"egraph"` import automatically resolves to the Node ESM loader via `"exports"`:

```js
import init, { CsrGraph, TFdpSolver, pivotMdsInit } from "egraph";

await init();

const graph = CsrGraph.fromEdgeList(4, new Uint32Array([0, 1, 1, 2, 2, 3, 3, 0]));
const layout = pivotMdsInit(graph, { numPivots: 4, seed: 42 });
const solver = new TFdpSolver(graph, { backend: "ibFft", iterations: 200, seed: 42 });
solver.run(layout);

const xy = layout.coordsView();
console.log(Array.from(xy));
```

## Solver & Engine Examples

### Track A: `TsNetSolver.pure2Hop` & `fromGraphRowNormalized`

```js
const solver = TsNetSolver.pure2Hop(graph, {
  backend: "auto2d", // "auto2d", "fitFft", "linearFft", "barnesHut", "exact"
  perplexityC: 0.9,
  seed: 42,
});
solver.run(layout);

const rnSolver = TsNetSolver.fromGraphRowNormalizedWithOptions(graph, {
  w2: 0.5,
  mode: "twoStepRandomWalk",
  backend: "auto2d",
  seed: 42,
});
rnSolver.run(layout);
```

### Track B, Separated Embedding & Force Models

```js
import {
  ForceAtlas2Solver,
  GuMapSolver,
  PacmapGraphSolver,
  SgdSolver,
  SnapTFdpSolver,
  TFdpSolver,
} from "egraph";

// t-FDP with ibFFT repulsion
const tfdp = new TFdpSolver(graph, {
  backend: "ibFft",
  iterations: 200,
  seed: 42,
});
tfdp.run(layout);

// ForceAtlas2 (Barnes-Hut)
const fa2 = new ForceAtlas2Solver(graph, {
  backend: "barnesHut",
  iterations: 200,
  theta: 0.5,
  seed: 42,
});
fa2.run(layout);

// PaCMAP-Graph & SNAP-tFDP (standalone separated classes)
new PacmapGraphSolver(graph, { phaseIterations: [20, 20, 40], seed: 42 }).run(layout);
new SnapTFdpSolver(graph, { iterations: 100, seed: 42 }).run(layout);

// GuMap from regularized Chebyshev DoG landmarks
const gumap = GuMapSolver.fromGraphDog(graph, {
  iterations: 200,
  minDist: 0.1,
  spread: 1.0,
  seed: 42,
});
gumap.run(layout);

// Regularized DoG 3-tier Sparse Barrier SGD
const barrierSgd = SgdSolver.sparseBarrierDog(graph, { seed: 42 });
barrierSgd.run(layout);
```

### `Layout3d`, `EdgePolylines` (`FdebSolver`), `BigInt` Seeds & Non-Euclidean Manifolds

```js
import {
  ArenaOctree3d,
  EdgePolylines,
  FdebSolver,
  ForceAtlas2Solver,
  Layout3d,
  ManifoldLayout2d,
  SgdSolver,
  TsNetSolver,
  geodesicDistortion,
  runProgressive,
} from "egraph";

// 3D Layout3d with 64-bit BigInt seed, target: "layout3d", and ArenaOctree3d
const layout3d = Layout3d.init(graph.numNodes(), { seed: 42n });
new ForceAtlas2Solver(graph, {
  target: "layout3d",
  backend: "barnesHut",
  iterations: 100,
  seed: 42n,
}).run(layout3d);

const octree = new ArenaOctree3d();
octree.build(layout3d);
const repulsion = octree.computeCoulombRepulsion(0.5);

// Edge bundling via EdgePolylines + FdebSolver
const polylines = EdgePolylines.straightLineFromGraph(graph, layout);
const fdeb = new FdebSolver(polylines, { cycles: 3, stepsPerCycle: 15 });
fdeb.run(polylines);
console.log("bundled points:", polylines.edgeOffsetsView().length, polylines.pointsView().length);

// Hyperbolic Lorentz hyperboloid -> Poincaré disk + explicit target: "manifold2d"
const lorentz = ManifoldLayout2d.init("hyperbolicLorentz", graph.numNodes(), { seed: 42 });
await runProgressive(
  SgdSolver.full(graph, { target: "manifold2d", iterations: 60, seed: 42 }),
  lorentz,
);
const poincare = lorentz.toPoincareLayout();
console.log("geodesicDistortion:", geodesicDistortion(graph, poincare));

// Torus spectral phase init & optimal boundary auto-pan
const torus = ManifoldLayout2d.torusSpectralPhaseInit(graph, 42);
const panned = torus.torusAutoPan(graph);
console.log(panned.coordsView().length);

// Structured error codes (.code: EgraphErrorCode) on invalid (target, backend) pairs
try {
  TsNetSolver.pure2Hop(graph, { target: "layout3d", backend: "fitFft" });
} catch (err) {
  console.error("Error code:", err.code); // "UnsupportedOperation"
}
```

### Clustering (`labelsView()`), Overlap Removal, `LayoutTransform` & Datasets

```js
import {
  Layout2d,
  PrismOverlapRemoval,
  TwinDeclutterer,
  generateSbmGraph,
  leidenCommunities,
  listDatasets,
  prismDistortion,
} from "egraph";

// Synthetic SBM graph ({ ...options }) + zero-copy LeidenResult.labelsView()
const { graph: sbmGraph, labels: groundTruth } = generateSbmGraph(
  new Uint32Array([12, 12]),
  { pIntra: 0.5, pInter: 0.05, seed: 42 },
);
const leiden = leidenCommunities(sbmGraph, { seed: 42 });
const labels = leiden.labelsView();
console.log("datasets:", listDatasets().length, "communities:", leiden.numCommunities, labels.length, groundTruth.length);

// Overlap removal (PRISM) + TwinDeclutterer LayoutTransform (.apply)
const original = Layout2d.fromCoords(layout.coordsView());
const radii = new Float32Array(graph.numNodes()).fill(0.25);
const prism = PrismOverlapRemoval.fromRadii(radii, {
  maxIterations: 50,
  padding: 0.05,
  seed: 42,
});
prism.run(layout);
new TwinDeclutterer({ iterations: 4 }).apply(layout);

const distortion = prismDistortion(original, layout);
console.log(prism.lastOverlapCount, distortion.sigmaDist, distortion.sigmaDisp);
```

### Linalg, `LayoutGenerator`s, Chebyshev & Shortest Paths

```js
import {
  ChebyshevDogKernel,
  ClassicalMds,
  DijkstraWorkspace,
  PivotMds,
  classicalMds,
  combinatorialLaplacian,
  pivotMds,
} from "egraph";

const ws = new DijkstraWorkspace(graph.numNodes());
const dists = ws.dijkstraFrom(graph, 0);
const mdsNd = classicalMds(graph, 3);
const mds3d = new ClassicalMds({ target: "layout3d" }).generate(graph);
const mds2d = new PivotMds({ numPivots: 4, seed: 42 }).generate(graph);
const mds2dFn = pivotMds(graph, { numPivots: 4, seed: 42 });
const lap = combinatorialLaplacian(graph);
const dog = new ChebyshevDogKernel(graph, { seed: 42 });
console.log(dists.length, mdsNd.dims, mds3d.len(), mds2d.numNodes, mds2dFn.numNodes, lap.nnz, dog.degree);
```

## Migration from v5 / Legacy Alpha

See [docs/MIGRATION_V6.md](https://github.com/likr/egraph-rs/blob/master/docs/MIGRATION_V6.md) for a complete v5, `alpha.8 -> alpha.9`, and v6 API freeze migration guide, including:

1. Replacing `Graph` host object weights with `CsrGraph.fromCsr` / `CsrGraph.fromEdgeList` (`weightDtype: "float32"`) and host-side arrays
2. Migrating overlap removal and post-processing to `PrismOverlapRemoval` / `ForbidOverlapRemoval` and `LayoutTransform.apply()` (`PfsPrimeOverlapRemoval`, `TwinDeclutterer`)
3. Consolidating `TsNet` / `BhTsNet` / `FitTsNet` / `LTsNet` into `TsNetSolver` (`pure2Hop`, `fromGraphRowNormalized`, `auto2d`/`auto3d`/`fitFft`/`linearFft`/`barnesHut`/`exact`) and `LayoutGenerator` structs (`ClassicalMds`, `PivotMds`, `Hde`, `RdMds`, `KorenSpectral`, `CoarseBackbone`, `SfcLayout`)
4. Adopting `Layout3d`, `EdgePolylines` + `FdebSolver(polylines)`, separated SGD classes (`Sgd2Solver`, `PacmapGraphSolver`, `LargeVisSolver`, `DrGraphSolver`, `GraphCneSolver`, `SnapTFdpSolver`), and explicit `target` (`"layout2d" | "layout3d" | "layoutNd" | "manifold2d"`)
5. Migrating from `egraph/dist/web/egraph_wasm` to pure ESM, `coordsView()`, `edgeOffsetsView()`, `pointsView()`, `labelsView()`, `runProgressive`, `{ ...options }`, `camelCase` literals, `number | bigint` seeds, and `.code` structured errors

## License

MIT
