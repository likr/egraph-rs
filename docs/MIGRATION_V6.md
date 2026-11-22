# Migrating to `egraph` v6 (`v6.0.0-alpha.4`)

`egraph` v6 is a clean architectural redesign across Native Rust, Python (`egraph`), and WebAssembly/TypeScript (`egraph`), targeting strictly single-threaded, deterministic pure CPU execution with zero-copy CSR/SoA data buffers and 100% cross-target parity.

For background on the v6 architecture, see [the v6 release plan](V6_RELEASE_PLAN.md).

---

## Summary of Breaking Changes

| Area | Legacy (`v5` / `v6-alpha.0..3`) | `egraph` v6 (`v6.0.0-alpha.4`) |
|---|---|---|
| **Graph representation** | `Graph` / `DiGraph` storing `Py<PyAny>` or `JsValue` payloads | `CsrGraph` (`u32` CSR indices + numeric weights) + host-side metadata arrays + `egraph.interop` |
| **Coordinate buffers** | `DrawingEuclidean2d` / `DrawingHyperbolic2d` / `DrawingSpherical2d` / `DrawingTorus2d` | `Layout2d` (zero-copy `coords(copy=False)` / `coordsView()`), `LayoutNd`, `ManifoldLayout2d` |
| **Overlap removal** | `OverwrapRemoval` ($O(N^2)$) | `PrismOverlapRemoval` ($O(N \log N)$ Delaunay + bounded CG) |
| **tsNET family** | `TsNet`, `BhTsNet`, `FitTsNet`, `LTsNet` | Unified `TsNetSolver` (`pure_2hop` / `pure2Hop`, `from_knn`, `from_probabilities`, `from_graph_partial_bfs`) |
| **Stress metric** | Unscaled `stress(drawing, dm)` (`NS`) | Scale-Normalized Stress (`scale_normalized_stress` / `scaleNormalizedStress`, `evaluate_metrics` / `evaluateMetrics`) |
| **WASM packaging** | `egraph/dist/web/egraph_wasm`, UMD/Rollup, $2N$ `x(u)`/`y(u)` FFI loops | Pure ESM (`"type": "module"`), `await init()`, `CsrGraph.fromCsr`, `coordsView()`, `runProgressive()` |
| **Community detection** | `Louvain`, `LabelPropagation`, `SpectralClustering`, `InfoMap` | `louvain_communities`, `label_propagation`, `spectral_clustering`, `kmeans_plus_plus`, `coarsen_graph` (`InfoMap` removed) |
| **Python runtime** | Python `>= 3.6`, optional NumPy | Python `>= 3.10` (`abi3-py310`), required `numpy >= 1.26` |

---

## 1. `Graph` Host Object Weights Removed -> Host-Side Arrays + `CsrGraph` + `egraph.interop`

Legacy `Graph` and `DiGraph` stored arbitrary Python (`Py<PyAny>`) or JavaScript (`JsValue`) objects on every node and edge, causing per-element FFI allocations, GC pinning, and `petgraph` lookup overhead inside layout loops.

In v6, **`CsrGraph` stores topology (`u32` CSR indices) and optional numeric edge weights only**. Application metadata (labels, colors, node dictionaries, edge attributes) stays in host-side arrays/lists indexed by vertex ID `0..N-1`.

### Python: Before & After

**Before (legacy):**

```python
import networkx as nx
import egraph as eg

nx_graph = nx.karate_club_graph()
graph = eg.Graph()
node_map = {}
for node in nx_graph.nodes:
    node_map[node] = graph.add_node({"label": str(node)})
for u, v in nx_graph.edges:
    graph.add_edge(node_map[u], node_map[v], 1.0)
```

**After (v6):**

```python
import networkx as nx
import numpy as np
import scipy.sparse as sp
import egraph as eg
from egraph.interop import from_networkx, from_scipy_csr

# Option A: Direct NetworkX interop (keep node IDs in `nodes`)
nx_graph = nx.karate_club_graph()
graph, nodes = from_networkx(nx_graph)
node_labels = [str(n) for n in nodes]

# Option B: Direct SciPy CSR matrix interop
adj_csr = sp.csr_matrix(nx.to_scipy_sparse_array(nx_graph, format="csr", dtype=float))
graph_from_scipy = from_scipy_csr(adj_csr)

# Option C: Raw CSR or edge list arrays with zero host objects
indptr = np.array([0, 2, 4, 6], dtype=np.uint32)
indices = np.array([1, 2, 0, 2, 0, 1], dtype=np.uint32)
weights = np.array([1.0, 2.0, 1.0, 1.5, 2.0, 1.5], dtype=np.float64)
tri = eg.CsrGraph.from_csr(indptr, indices, weights)
```

### WASM / TypeScript: Before & After

**Before (legacy):**

```js
import { Graph } from "egraph";

const graph = new Graph();
const idToNode = new Map();
for (const item of data.nodes) {
  idToNode.set(item.id, graph.addNode(item));
}
for (const edge of data.links) {
  graph.addEdge(idToNode.get(edge.source), idToNode.get(edge.target), edge);
}
```

**After (v6):**

```ts
import init, { CsrGraph } from "egraph";

await init();

// Keep host metadata in parallel JS arrays indexed by 0..N-1
const nodeMetadata = data.nodes;
const idToIndex = new Map(data.nodes.map((n, idx) => [n.id, idx]));
const edges = new Uint32Array(data.links.length * 2);
const weights = new Float32Array(data.links.length);

data.links.forEach((link, e) => {
  edges[2 * e] = idToIndex.get(link.source)!;
  edges[2 * e + 1] = idToIndex.get(link.target)!;
  weights[e] = link.weight ?? 1.0;
});

const graph = CsrGraph.fromEdgeList(nodeMetadata.length, edges, weights);
```

---

## 2. `OverwrapRemoval` -> `PrismOverlapRemoval`

Legacy `OverwrapRemoval` (misspelled `Overwrap`) used an $O(N^2)$ heuristic pass. v6 replaces it with **`PrismOverlapRemoval`**, an $O(N \log N)$ Delaunay triangulation + bounded conjugate-gradient proximity stress solver supporting both uniform/per-node radii (`from_radii` / `fromRadii`) and rectangular bounding boxes (`half_widths` / `half_heights`). You can also measure layout preservation distortion via `prism_distortion` / `prismDistortion`.

### Python: Before & After

**Before (legacy):**

```python
import egraph as eg

or_algo = eg.OverwrapRemoval(graph, lambda i: radii[i])
or_algo.iterations = 10
or_algo.apply_with_drawing_euclidean_2d(drawing)
```

**After (v6):**

```python
import numpy as np
import egraph as eg

original = eg.Layout2d.from_coords(layout.coords(copy=True))
radii = np.full(graph.num_nodes, 0.25, dtype=np.float64)

prism = eg.PrismOverlapRemoval.from_radii(
    radii,
    max_iterations=50,
    padding=0.05,
    seed=42,
)
prism.run(layout)

sigma_dist, sigma_disp = eg.prism_distortion(original, layout)
print(f"overlaps={prism.last_overlap_count}, sigma_dist={sigma_dist:.4f}, sigma_disp={sigma_disp:.4f}")
```

For rectangular node boxes, pass `half_widths` and `half_heights` directly:

```python
prism_boxes = eg.PrismOverlapRemoval(half_widths, half_heights, padding=0.05, seed=42)
prism_boxes.run(layout)
```

### WASM / TypeScript: Before & After

**Before (legacy):**

```js
const removal = new OverwrapRemoval(graph, (i) => radii[i]);
removal.applyWithDrawingEuclidean2d(drawing);
```

**After (v6):**

```ts
import { Layout2d, PrismOverlapRemoval, prismDistortion } from "egraph";

const original = Layout2d.fromCoords(layout.coordsView());
const prism = PrismOverlapRemoval.fromRadii(radii, {
  maxIterations: 50,
  padding: 0.05,
  seed: 42,
});
prism.run(layout);

const { sigmaDist, sigmaDisp } = prismDistortion(original, layout);
console.log(prism.lastOverlapCount, sigmaDist, sigmaDisp);
```

---

## 3. `TsNet` / `BhTsNet` / `FitTsNet` / `LTsNet` -> `TsNetSolver`

Legacy v5/alpha split tsNET across four separate classes (`TsNet`, `BhTsNet`, `FitTsNet`, `LTsNet`) with fixed $0.05 N$ perplexity and separate builders. v6 unifies them into **`TsNetSolver`**, supporting `"fft"` (default `Fit-tsNET`), `"linear_fft"`, `"barnes_hut"`, and `"exact"` backends, cooperative `step()` / `run_steps()` / `run()` execution, and the validated Track A preset **`pure_2hop` / `pure2Hop`** (pure 1+2-hop CSR + node-wise adaptive perplexity).

### Python: Before & After

**Before (legacy):**

```python
import egraph as eg

drawing = eg.DrawingEuclidean2d.initial_placement(graph)
rng = eg.Rng.seed_from(42)
tsnet = (
    eg.FitTsNet.builder()
    .perplexity(15.0)
    .iterations_stage1(250)
    .iterations_stage2(250)
    .iterations_stage3(500)
    .build()
)
tsnet.run(drawing, graph, rng)
```

**After (v6):**

```python
import egraph as eg

layout = eg.pivot_mds_init(graph, num_pivots=50, seed=42)
solver = eg.TsNetSolver.pure_2hop(
    graph,
    backend="fft",        # "fft", "linear_fft", "barnes_hut", or "exact"
    perplexity_c=0.9,
    seed=42,
)
solver.run(layout)
```

Custom $k$-NN or precomputed probability matrices remain first-class via `TsNetSolver.from_knn(...)` and `TsNetSolver.from_probabilities(...)`.

### WASM / TypeScript: Before & After

**Before (legacy):**

`TsNet` / `FitTsNet` were unavailable in WASM (`crates/wasm`).

**After (v6):**

```ts
import { TsNetSolver, pivotMdsInit, runProgressive } from "egraph";

const layout = pivotMdsInit(graph, 50, 42);
const solver = TsNetSolver.pure2Hop(graph, {
  backend: "fft",
  perplexityC: 0.9,
  seed: 42,
});

await runProgressive(solver, layout, () => {
  draw(layout.coordsView());
}, { frameBudgetMs: 10 });
```

---

## 4. `stress` (`NS`) -> `SNS` (`scale_normalized_stress` / `evaluate_metrics`)

Legacy `stress(drawing, dm)` computed unscaled Normalized Stress (`NS`), which depended on the arbitrary drawing scale (for example, `PivotMds` or `tsNET` coordinate magnitudes compared against unit shortest-path distances) and triggered an $O(N^3)$ `warshall_floyd` pass on every call in WASM `js_stress`.

v6 introduces **Scale-Normalized Stress (`SNS`)**, which analytically solves for the optimal scalar scale $s^* = \frac{\sum w_{ij} d_{ij} \|x_i - x_j\|}{\sum w_{ij} \|x_i - x_j\|^2}$ in closed form, alongside $O(kN)$ pivot-sampled `SNS`, Bentley-Ottmann sweep-line crossings, and `StaticKdTree`-accelerated `NP1`/`NP2` via the selective `evaluate_metrics` / `evaluateMetrics` dispatcher.

### Python: Before & After

**Before (legacy):**

```python
import egraph as eg

dm = eg.all_sources_bfs(graph, 1.0)
raw_stress = eg.stress(drawing, dm)
np2 = eg.neighborhood_preservation_2hop(graph, drawing)
crossings = eg.crossing_number(graph, drawing)
```

**After (v6):**

```python
import egraph as eg

# Closed-form Scale-Normalized Stress (or use "sampled_sns" for large graphs)
sns = eg.scale_normalized_stress(graph, layout)

# Or evaluate multiple metrics in one selective pass (skipping unrequested metrics)
report = eg.evaluate_metrics(
    graph,
    layout,
    ["sns", "np1", "np2", "crossing_number"],
    num_pivots=64,
    num_samples=10_000,
    seed=42,
)
print(report["scale_normalized_stress"], report["neighborhood_preservation_2hop"])
```

### WASM / TypeScript: Before & After

**Before (legacy):**

```js
const rawStress = stress(graph, drawing); // Legacy O(N^3) all-pairs Warshall-Floyd + unscaled NS
```

**After (v6):**

```ts
import { evaluateMetrics, scaleNormalizedStress } from "egraph";

const sns = scaleNormalizedStress(graph, layout);

const report = evaluateMetrics(
  graph,
  layout,
  ["sns", "np2", "crossingNumber"],
  64,
  10_000,
  42,
);
const record = report.toRecord();
console.log(record.scaleNormalizedStress, record.neighborhoodPreservation2Hop, record.crossingNumber);
```

---

## 5. WASM `fromCsr` / `coordsView()` / `runProgressive()` (`rAF`) + Pure ESM Entrypoints

Legacy WASM builds shipped four duplicate `.wasm` bundles (`dist/web`, `dist/nodejs`, `dist/bundler`, `dist/no-modules`) plus UMD/Rollup output, required per-node `drawing.x(u)` / `drawing.y(u)` FFI calls ($2N$ boundary crossings per frame), and executed layouts synchronously.

v6 ships **a single pure ESM package (`"type": "module"`) with one shared `.wasm` binary** for browsers, bundlers, and Node.js 18+, zero-copy linear-memory views via `layout.coordsView()`, and non-blocking progressive `requestAnimationFrame` execution via `runProgressive()`.

### Before (legacy)

```js
import init, { Graph, DrawingEuclidean2d, FullSgd } from "egraph/dist/web/egraph_wasm";

await init();
const graph = new Graph();
// ... addNode / addEdge ...
const drawing = DrawingEuclidean2d.initialPlacement(graph);
const sgd = new FullSgd().build(graph, () => 1.0);
// Synchronous loop + 2N FFI calls per frame:
for (const u of graph.nodeIndices()) {
  const x = drawing.x(u);
  const y = drawing.y(u);
}
```

### After (v6)

```ts
import init, {
  CsrGraph,
  Layout2d,
  TFdpSolver,
  pivotMdsInit,
  runProgressive,
} from "egraph";

// 1. Initialize the single shared .wasm binary (works in Browser, Vite, and Node 18+)
await init();

// 2. Construct CSR graph in bulk (no per-node/per-edge FFI calls)
const graph = CsrGraph.fromCsr(indptr, indices, weights, { directed: false, sanitize: true });
const layout = pivotMdsInit(graph, 50, 42);
const solver = new TFdpSolver(graph, { backend: "ibfft", totalIterations: 200, seed: 42 });

// 3. Cooperative rAF loop with zero-copy linear-memory Float32Array view
await runProgressive(
  solver,
  layout,
  ({ iteration, totalIterations, frameTimeMs }) => {
    // Automatically checks wasmMemory().buffer identity in O(1) to prevent memory.grow detach bugs
    const xy = layout.coordsView();
    renderer.updatePositions(xy);
  },
  { frameBudgetMs: 10 },
);
```

---

## Additional v6 Solver & Space Upgrades

- **Track B Sparse Barrier SGD & GuMap**: Replace `SparseSgd.barrier(...).build_barrier_from_landmark_tables(...)` + manual scheduler loops with `SgdSolver.sparse_barrier_dog(graph, seed=42)` (or `SgdSolver.sparseBarrierDog(graph, { seed: 42 })` in WASM) and `SpatialIntruderRefiner(graph)` for 2D `StaticKdTree` post-refinement.
- **Unified M4 Solvers**: `TFdpSolver`, `IbUmapSolver`, `SgdSolver.snap_t_fdp` / `snapTFdp`, `SgdSolver.pacmap_graph` / `pacmapGraph`, `SgdSolver.sgd2`, `ForceAtlas2Solver`, `YifanHuSolver`, and reference-passing `FdebSolver` (`edge_offsets()`, `points()`).
- **Non-Euclidean Spaces**: Replace `DrawingHyperbolic2d`, `DrawingSpherical2d`, and `DrawingTorus2d` with `ManifoldLayout2d` (`"hyperbolic"`, `"spherical"`, `"torus"`) + `SgdSolver.step_manifold` / `run_manifold` (`stepManifold` / `runManifold`).
- **Further Reading**:
  - [Python README](../crates/python/README.md)
  - [WASM README](../crates/wasm/README.md)
  - [v6 Release Plan](V6_RELEASE_PLAN.md)
