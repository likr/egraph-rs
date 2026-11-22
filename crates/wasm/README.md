# egraph

Fast, strictly single-threaded graph drawing and layout library implemented in pure Rust and WebAssembly (`wasm-bindgen`), distributed as a pure ESM package sharing a single `.wasm` binary across browsers, bundlers, and Node.js 18+.

## Install

```bash
npm install egraph
```

## Architecture Highlights (v6)

- **Pure ESM & Single `.wasm`**: Modern `"exports"` entrypoints for browsers/bundlers (`./pkg/index.js`) and Node.js 18+ (`./pkg/egraph_wasm.node.js`), sharing a single `wasm-opt -Oz` optimized `.wasm` binary.
- **Strictly Single-Threaded**: Pure CPU Rust with deterministic seeded execution (`seed = 42` default), zero `rayon`, zero `std::thread`, zero Web Workers, and zero `SharedArrayBuffer`.
- **Zero-Copy Linear Memory Views**: `Layout2d.coordsView()` returns a cached `Float32Array` view directly over WebAssembly linear memory and automatically refreshes when `memory.grow` detaches the underlying `ArrayBuffer`.
- **Cooperative `requestAnimationFrame` Execution**: All iterative solvers implement `step()`, `runSteps(layout, k)`, and `run(layout)`, plus `runProgressive(solver, layout, onFrame, options?)` for responsive 60fps browser rendering.

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
  backend: "fft",
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

// Selectively evaluate quality metrics
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
const solver = new TFdpSolver(graph, { backend: "ibfft", totalIterations: 200, seed: 42 });
solver.run(layout);

const xy = layout.coordsView();
console.log(Array.from(xy));
```

## Solver Examples

### Track A: `TsNetSolver.pure2Hop`

```js
const solver = TsNetSolver.pure2Hop(graph, {
  backend: "fft",
  perplexityC: 0.9,
  seed: 42,
});
solver.run(layout);
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
  totalIterations: 200,
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
  nEpochs: 200,
  minDist: 0.1,
  spread: 1.0,
  seed: 42,
});
gumap.run(layout);

// Regularized DoG 3-tier Sparse Barrier SGD
const barrierSgd = SgdSolver.sparseBarrierDog(graph, { seed: 42 });
barrierSgd.run(layout);
```

### Overlap Removal (`PrismOverlapRemoval`)

```js
import { PrismOverlapRemoval, prismDistortion } from "egraph";

const original = Layout2d.fromCoords(layout.coordsView());
const radii = new Float32Array(graph.numNodes()).fill(0.25);
const prism = PrismOverlapRemoval.fromRadii(radii, {
  maxIterations: 50,
  padding: 0.05,
  seed: 42,
});
prism.run(layout);

const distortion = prismDistortion(original, layout);
console.log(prism.lastOverlapCount, distortion.sigmaDist, distortion.sigmaDisp);
```

## Migration from v5 / Legacy Alpha

See [docs/MIGRATION_V6.md](https://github.com/likr/egraph-rs/blob/master/docs/MIGRATION_V6.md) for a complete v5/legacy-alpha to v6 migration guide, including:

1. Replacing `Graph` host object weights with `CsrGraph.fromCsr` / `CsrGraph.fromEdgeList` and host-side arrays
2. Migrating `OverwrapRemoval` to `PrismOverlapRemoval`
3. Consolidating `TsNet` / `BhTsNet` / `FitTsNet` / `LTsNet` into `TsNetSolver.pure2Hop`
4. Replacing `stress` (`NS`) with Scale-Normalized Stress (`scaleNormalizedStress` / `evaluateMetrics`)
5. Migrating from `egraph/dist/web/egraph_wasm` to pure ESM, `coordsView()`, and `runProgressive`

## License

MIT
