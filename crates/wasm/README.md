# egraph

Fast, strictly single-threaded graph drawing and layout library implemented in pure Rust and WebAssembly (`wasm-bindgen`), distributed as a pure ESM package sharing a single `.wasm` binary across browsers, bundlers, and Node.js 18+.

## Install

```bash
npm install egraph
```

## Architecture Highlights (v6)

- **Pure ESM & Single `.wasm`**: Modern `"exports"` entrypoints for browsers/bundlers (`./pkg/index.js`) and Node.js 18+ (`./pkg/egraph_wasm.node.js`), sharing a single `wasm-opt -Oz` optimized `.wasm` binary.
- **Strictly Single-Threaded**: Pure CPU Rust with deterministic seeded execution (`seed = 42` default), zero `rayon`, zero `std::thread`, zero Web Workers, and zero `SharedArrayBuffer`.
- **Zero-Copy Linear Memory Views**: `coordsView()` on `Layout2d`, `LayoutNd`, and `ManifoldLayout2d` (plus `labelsView()` on all 5 clustering result classes and `centroidsView()` on `KMeansResult`) returns a cached typed-array view directly over WebAssembly linear memory and automatically refreshes if `memory.grow` detaches the `ArrayBuffer`.
- **Polymorphic Cooperative Execution**: `step(layout)`, `runSteps(layout, k)`, `run(layout)`, and `runProgressive(solver, layout, onFrame, options?)` work polymorphically across `Layout2d`, `LayoutNd` (11 N-D solvers), and `ManifoldLayout2d` (6 manifold solvers).
- **Track A, Track B, 3D/N-D & Non-Euclidean Engines**: `TsNetSolver.pure2Hop({ backend: "auto" })` / `fromGraphRowNormalized`, `SgdSolver.sparseBarrierDog`, `GuMapSolver`, `ArenaOctree3d`, `DynamicKdTree`, `ManifoldLayout2d` (`"hyperbolic_lorentz"`, `toPoincareLayout`, `ManifoldMetric`, `torusSpectralPhaseInit`, `torusAutoPan`, `torusEdgeSegments`, geodesic crossings), `TwinDeclutterer`, `transitiveReduction`, `MetricReport` (`toRecord()`), `listDatasets` / synthetic generators, `DijkstraWorkspace`, MDS, Laplacian builders, and `ChebyshevDogKernel`.

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
const layout = pivotMdsInit(graph, 4, 42);

// Run Track A Pure 1+2-Hop Adaptive tsNET progressively in requestAnimationFrame
const solver = TsNetSolver.pure2Hop(graph, {
  backend: "auto",
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
const layout = pivotMdsInit(graph, 4, 42);
const solver = new TFdpSolver(graph, { backend: "ibfft", iterations: 200, seed: 42 });
solver.run(layout);

const xy = layout.coordsView();
console.log(Array.from(xy));
```

## Solver & Engine Examples

### Track A: `TsNetSolver.pure2Hop` & `fromGraphRowNormalized`

```js
const solver = TsNetSolver.pure2Hop(graph, {
  backend: "auto", // "auto" (Exact <= 256, LinearFft > 256), "fft", "linear_fft", "barnes_hut", "exact"
  perplexityC: 0.9,
  seed: 42,
});
solver.run(layout);

const rnSolver = TsNetSolver.fromGraphRowNormalizedWithOptions(graph, {
  w2: 0.5,
  mode: "two_step_random_walk",
  backend: "auto",
  seed: 42,
});
rnSolver.run(layout);
```

### Track B & Unified Force Models: `TFdpSolver`, `ForceAtlas2Solver`, `GuMapSolver`

```js
import {
  ForceAtlas2Solver,
  GuMapSolver,
  SgdSolver,
  TFdpSolver,
} from "egraph";

// t-FDP with ibFFT repulsion
const tfdp = new TFdpSolver(graph, {
  backend: "ibfft",
  iterations: 200,
  seed: 42,
});
tfdp.run(layout);

// ForceAtlas2 (Barnes-Hut)
const fa2 = new ForceAtlas2Solver(graph, {
  iterations: 200,
  useBarnesHut: true,
  theta: 0.5,
  seed: 42,
});
fa2.run(layout);

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

### 3D / $N$-D (`LayoutNd` + `ArenaOctree3d`) & Non-Euclidean Manifolds (`ManifoldLayout2d`)

```js
import {
  ArenaOctree3d,
  LayoutNd,
  ManifoldLayout2d,
  SgdSolver,
  geodesicDistortion,
  runProgressive,
} from "egraph";

// 3D LayoutNd with ArenaOctree3d
const layout3d = LayoutNd.init(graph.numNodes(), 3, 42);
const octree = new ArenaOctree3d();
octree.build(layout3d);
const repulsion = octree.computeCoulombRepulsion(0.5);

// Hyperbolic Lorentz hyperboloid -> Poincaré disk + polymorphic runProgressive
const lorentz = ManifoldLayout2d.init("hyperbolic_lorentz", graph.numNodes(), 42);
await runProgressive(SgdSolver.full(graph, { iterations: 60, seed: 42 }), lorentz);
const poincare = lorentz.toPoincareLayout();
console.log("geodesicDistortion:", geodesicDistortion(graph, poincare));

// Torus spectral phase init & optimal boundary auto-pan
const torus = ManifoldLayout2d.torusSpectralPhaseInit(graph, 42);
const panned = torus.torusAutoPan(graph);
console.log(panned.coordsView().length);
```

### Clustering (`labelsView()`), Overlap Removal, Decluttering & Datasets

```js
import {
  Layout2d,
  PrismOverlapRemoval,
  TwinDeclutterer,
  generateSbmGraph,
  leidenCommunities,
  listDatasets,
  prismDistortion,
  transitiveReduction,
} from "egraph";

// Synthetic SBM graph + zero-copy LeidenResult.labelsView()
const { graph: sbmGraph, labels: groundTruth } = generateSbmGraph(new Uint32Array([12, 12]), 0.5, 0.05, 42);
const leiden = leidenCommunities(sbmGraph, { seed: 42 });
const labels = leiden.labelsView();
console.log("datasets:", listDatasets().length, "communities:", leiden.numCommunities, labels.length, groundTruth.length);

// Overlap removal (PRISM) + TwinDeclutterer
const original = Layout2d.fromCoords(layout.coordsView());
const radii = new Float32Array(graph.numNodes()).fill(0.25);
const prism = PrismOverlapRemoval.fromRadii(radii, {
  maxIterations: 50,
  padding: 0.05,
  seed: 42,
});
prism.run(layout);
new TwinDeclutterer({ iterations: 4 }).run(layout);

const distortion = prismDistortion(original, layout);
console.log(prism.lastOverlapCount, distortion.sigmaDist, distortion.sigmaDisp);
```

### Linalg, Chebyshev & Shortest Paths

```js
import {
  ChebyshevDogKernel,
  DijkstraWorkspace,
  classicalMds,
  combinatorialLaplacian,
  pivotMds,
} from "egraph";

const ws = new DijkstraWorkspace(graph.numNodes());
const dists = ws.dijkstraFrom(graph, 0);
const mds3d = classicalMds(graph, 3);
const mds2d = pivotMds(graph, 4, 42);
const lap = combinatorialLaplacian(graph);
const dog = new ChebyshevDogKernel(graph, { seed: 42 });
console.log(dists.length, mds3d.dims, mds2d.numNodes, lap.nnz, dog.degree);
```

## Migration from v5 / Legacy Alpha

See [docs/MIGRATION_V6.md](https://github.com/likr/egraph-rs/blob/master/docs/MIGRATION_V6.md) for a complete v5 and `alpha.8 -> alpha.9` migration guide, including:

1. Replacing `Graph` host object weights with `CsrGraph.fromCsr` / `CsrGraph.fromEdgeList` and host-side arrays
2. Migrating legacy v5 overlap removal classes to `PrismOverlapRemoval` / `ForbidOverlapRemoval` / `TwinDeclutterer`
3. Consolidating `TsNet` / `BhTsNet` / `FitTsNet` / `LTsNet` into `TsNetSolver` (`pure2Hop`, `fromGraphRowNormalized`, `"auto"`)
4. Replacing `stress` (`NS`) with Scale-Normalized Stress (`scaleNormalizedStress` / `evaluateMetrics` -> `MetricReport`)
5. Migrating from `egraph/dist/web/egraph_wasm` to pure ESM, `coordsView()`, `labelsView()`, and polymorphic `runProgressive`, plus `iterations`/`maxIterations` option unification

## License

MIT
