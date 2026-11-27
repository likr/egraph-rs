# egraph

A fast, strictly single-threaded pure-CPU graph drawing and layout library for **Native Rust**, **Python** (`abi3-py310`), **WebAssembly/TypeScript** (pure ESM), and the **`egraph` CLI** (`6.0.0-alpha.8`).

## Highlights (v6)

- **Zero-Copy CSR & SoA Buffers**: `CsrGraph` (`u32` topology + optional `float64`/`float32` edge weights, no host-object payloads), `Layout2d`, `LayoutNd` (`dims`), and `ManifoldLayout2d`.
- **Track A (Near-Linear Adaptive tsNET)**: `TsNetSolver.pure_2hop` (exact CSR 1+2-hop + node-wise adaptive perplexity) and `TsNetSolver.from_graph_row_normalized` (`TwoStepRandomWalk` $A_1 + w_2 A_2$ affinity) with `TsNetBackend::Auto` (`N <= 256` Exact, `> 256` `LinearFft`) plus `"fft"`, `"linear_fft"`, `"barnes_hut"`, and `"exact"` backends.
- **Track B ($O(|E|)$ Chebyshev DoG, Sparse Barrier SGD & GuMap)**: Regularized Chebyshev DoG Varadhan landmarks (`ChebyshevDogKernel`, `DogVaradhanLandmarks`), 3-tier `SgdSolver.sparse_barrier_dog`, `GuMapSolver`, `IbUmapSolver`, `SgdSolver.pacmap_graph`, `SgdSolver.snap_t_fdp`, `TFdpSolver`, and 2D `StaticKdTree` `SpatialIntruderRefiner`.
- **3D & $N$-Dimensional Engine**: `ArenaOctree3d`, `DynamicKdTree`, `LayoutNd`, and **11 $N$-D solvers** (`SgdSolver`, `TsNetSolver`, `GuMapSolver`, `StressMajorizationSolver`, `KamadaKawaiSolver`, `MaxentStressSolver`, `FruchtermanReingoldSolver`, `LinLogSolver`, `ForceAtlas2Solver`, `YifanHuSolver`, `FdebSolver`) plus $N$-D quality metrics.
- **Non-Euclidean Manifold Engine**: `ManifoldLayout2d` across `"hyperbolic"` (Poincaré disk), `"hyperbolic_lorentz"` (Lorentz hyperboloid + `to_poincare_layout`), `"spherical"` ($S^2$), and `"torus"` ($T^2$ with `torus_spectral_phase_init` and `torus_auto_pan`), physical `ManifoldMetric` (`scale`, `aspect`, `rescale_geodesic`), **6 manifold solvers** (`SgdSolver`, `TsNetSolver`, `GuMapSolver`, `StressMajorizationSolver`, `KamadaKawaiSolver`, `FruchtermanReingoldSolver`), `geodesic_distortion`, and hyperbolic/spherical/torus geodesic crossings.
- **Unified Force, Multilevel, Overlap & Constraints**: `ForceAtlas2Solver`, `YifanHuSolver`, `MaxentStressSolver`, `GripSolver`, `GemSolver`, `SelLayoutSolver`, `SfcLayoutSolver`, `BackboneSolver`, `GroupInABoxSolver`, `ClusterRefineSolver`, `FdebSolver`, `PrismOverlapRemoval`, `GTreeOverlapRemoval`, `ForbidOverlapRemoval` (default `"sparse_forbid"`), `PfsPrimeOverlapRemoval`, `TwinDeclutterer` / `declutter_twins_2d`, `ScreenedPoissonField2d` / `ScreenedPoissonRefiner`, `transitive_reduction`, and 1D/2D `SeparationConstraint` projection.
- **Community Detection (5 Zero-Copy Result Classes)**: `louvain_communities` (`LouvainResult`), `leiden_communities` (`LeidenResult`), `kmeans_plus_plus` (`KMeansResult`), `label_propagation` (`LabelPropagationResult`), and `spectral_clustering` (`SpectralClusteringResult`), plus `coarsen_graph` and `coarse_backbone_init`.
- **Selective Quality Metrics (`MetricReport`)**: `evaluate_metrics` returning `MetricReport` across 2D, $N$-D, and manifold spaces—including Scale-Normalized Stress (`SNS` and pivot-sampled `SNS`), `geodesic_distortion`, Bentley-Ottmann sweep-line crossings, `NP1`/`NP2`, `S_RNG`/`S_GG`, `G-CAF`, `NOR`, `sprawlter_nn`, and cluster separation baselines.
- **Linear Algebra, Datasets & Interop**: `CsrMatrix`, `AmgPcgSolver`, `compute_effective_resistances`, `resistance_curvature_flow`, `DijkstraWorkspace`, `classical_mds` / `pivot_mds` / `hde` / `rd_mds`, `list_datasets` + 7 synthetic generators, and `egraph.interop` (`from_scipy_csr`/`to_scipy_csr`, `from_networkx`/`to_networkx`, `from_igraph`/`to_igraph`).
- **Strictly Single-Threaded & Deterministic**: Pure CPU Rust with explicit `seed=42` defaults, zero `rayon` / `std::thread` / Web Workers / `SharedArrayBuffer`, and cooperative `step()` / `run_steps()` / `run()` execution.

## Quick Start (Python)

**Version: `6.0.0-alpha.8` (`6.0.0a8`)**

```bash
pip install --pre "egraph[interop,viz]==6.0.0a8"
```

```python
import numpy as np
import egraph as eg
from egraph.interop import to_igraph, to_networkx

# Build a CsrGraph from CSR arrays (or from_edges / load_dataset / egraph.interop)
indptr = np.array([0, 2, 4, 6, 8, 10, 12], dtype=np.uint32)
indices = np.array([1, 5, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0], dtype=np.uint32)
graph = eg.CsrGraph.from_csr(indptr, indices)

# Initialize with normalized PivotMDS and run Track A Pure 1+2-Hop Adaptive tsNET
layout = eg.pivot_mds_init(graph, num_pivots=6, seed=42)
solver = eg.TsNetSolver.pure_2hop(
    graph,
    backend="auto",
    stage_iterations=(30, 30, 40),
    seed=42,
)
solver.run(layout)

# Zero-copy NumPy view (shape: (6, 2))
coords = layout.coords(copy=False)

# Selective MetricReport evaluation
report = eg.evaluate_metrics(graph, layout, ["sns", "np2", "crossing_number"])
print(report.scale_normalized_stress, report.neighborhood_preservation_2hop, report.crossing_number)

# Convert to NetworkX or python-igraph with attached coordinates
nx_g = to_networkx(graph, layout=layout)
ig_g = to_igraph(graph, layout=layout)
```

For $N$-D and non-Euclidean layouts, pass `LayoutNd` or `ManifoldLayout2d` directly into the same `run()` / `evaluate_metrics()` APIs:

```python
nd_layout = eg.LayoutNd.init(graph.num_nodes, dims=3, seed=42)
eg.ForceAtlas2Solver(graph, iterations=100, seed=42).run(nd_layout)

lorentz = eg.ManifoldLayout2d.init("hyperbolic_lorentz", graph.num_nodes, seed=42)
eg.SgdSolver.full(graph, iterations=60, seed=42).run(lorentz)
poincare = lorentz.to_poincare_layout()
```

See [crates/python/README.md](crates/python/README.md) and the [Python Sphinx Documentation](https://egraph-py-docs.netlify.app/) for full Python usage, and [notebooks/README.md](notebooks/README.md) for 9 interactive Google Colab tutorials.

## Quick Start (WebAssembly / TypeScript)

```bash
npm install egraph
```

```ts
import init, {
  CsrGraph,
  TsNetSolver,
  evaluateMetrics,
  pivotMdsInit,
  runProgressive,
} from "egraph";

await init();

const graph = CsrGraph.fromEdgeList(4, new Uint32Array([0, 1, 1, 2, 2, 3, 3, 0]));
const layout = pivotMdsInit(graph, 4, 42);
const solver = TsNetSolver.pure2Hop(graph, { backend: "auto", seed: 42 });

await runProgressive(solver, layout, () => {
  const xy = layout.coordsView(); // Zero-copy Float32Array over WASM linear memory
  console.log(xy[0], xy[1]);
}, { frameBudgetMs: 10 });

const report = evaluateMetrics(graph, layout, ["sns", "np2", "crossingNumber"]);
console.log(report.toRecord());
```

See [crates/wasm/README.md](crates/wasm/README.md) for browser, bundler, and Node.js 18+ examples.

## CLI (`egraph-cli`)

Build and invoke the single-threaded CLI (`layout`, `metrics`, `cluster`, `bench`):

```bash
cargo run --release -p egraph-cli -- layout --dataset karate --method tsnet-rn --output layout.json
cargo run --release -p egraph-cli -- layout --dataset lesmis --method sgd-barrier-dog --overlap-removal twin-declutter
cargo run --release -p egraph-cli -- layout --dataset karate --method sgd-full --manifold hyperbolic-lorentz
cargo run --release -p egraph-cli -- layout --dataset karate --method fa2 --dims 3
cargo run --release -p egraph-cli -- metrics --dataset karate --layout layout.json --targets sns,np2,crossings
cargo run --release -p egraph-cli -- cluster --dataset karate --method leiden-mod
cargo run --release -p egraph-cli -- bench --dataset 1138_bus --method tsnet-pure2hop
```

## Workspace Structure (27 Crates)

```text
egraph-rs/
├── crates/
│   ├── core/                    # CsrGraph, Layout2d, LayoutNd, ManifoldLayout2d, DeterministicRng
│   ├── ffi/                     # InterleavedLayout2d zero-copy FFI helpers
│   ├── petgraph/                # petgraph adapters
│   ├── dataset/                 # Embedded benchmark graphs & synthetic generators
│   ├── spatial/                 # Umbrella crate for spatial modules
│   ├── spatial-trees/           # StaticKdTree, DynamicKdTree, ArenaOctree3d, Delaunay
│   ├── spatial-fft/             # FFT, ibFFT, Screened Poisson 2D density field
│   ├── spatial-crossings/       # Sweep-line, torus, spherical, and hyperbolic crossings
│   ├── linalg/                  # CsrMatrix, Laplacians, DijkstraWorkspace, Chebyshev kernels
│   ├── linalg-amg/              # AMG-PCG, effective resistances, resistance curvature flow
│   ├── linalg-mds/              # Classical MDS, PivotMDS, HDE, RD-MDS, Koren spectral init
│   ├── neighborhood/            # 1+2-hop CSR, KnnGraph, SparseProbabilities, DoG/rHGH landmarks
│   ├── cluster/                 # Louvain, Leiden, Spectral, Label Propagation, K-Means++, coarsening
│   ├── metrics/                 # MetricReport & selective evaluate_metrics dispatcher
│   ├── metrics-geometry/        # SNS, geodesic distortion, crossings, NP1/NP2, S_RNG/S_GG, PRISM
│   ├── metrics-cluster/         # G-CAF, RTA, Adjusted Patch NIG, Sprawlter, NOR, separation baselines
│   ├── layout/                  # Unified layout solver umbrella crate
│   ├── layout-constraints/      # Cycle removal, transitive reduction, layering, VPSC, TwinDeclutterer
│   ├── layout-stress/           # Stress Majorization, Kamada-Kawai, Maxent-Stress, GRIP, SFC
│   ├── layout-force/            # t-FDP, ibUMAP, FA2, FR, LinLog, GEM, SEL, FDEB, ScreenedPoissonRefiner
│   ├── layout-neighbor/         # TsNetSolver, GuMapSolver, SpatialIntruderRefiner
│   ├── layout-sgd/              # SgdSolver (Full, SparseBarrierDoG, PaCMAP, SNAP-tFDP, Omega, etc.)
│   ├── layout-multilevel/       # Yifan Hu, Backbone, Group-in-a-Box, ClusterRefine, overlap removal
│   ├── egraph/                  # Public Rust facade crate
│   ├── cli/                     # egraph-cli (`layout`, `metrics`, `cluster`, `bench`)
│   ├── python/                  # PyO3 `abi3-py310` Python bindings & Sphinx docs
│   └── wasm/                    # wasm-bindgen pure ESM WebAssembly bindings & TypeDoc
├── docs/                        # Migration guide and architecture plans
├── js/examples/                 # Browser/Vite demos and Playwright verification suite
├── notebooks/                   # 9 Google Colab tutorial notebooks
└── scripts/                     # Unified verification, invariants, parity, and worktree tooling
```

See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for detailed crate descriptions and [docs/MIGRATION_V6.md](docs/MIGRATION_V6.md) for upgrading from v5 or earlier v6 alphas.

## Development & Verification

Provision the isolated Python environment once (`./scripts/setup_venv.sh` or `make venv-setup`), then use `./scripts/verify.sh` (or `Makefile` targets):

```bash
# Tier 0: Fast crate-level feedback (fmt, clippy -D warnings, nextest, thread scan)
./scripts/verify.sh --fast --crate egraph-layout

# Tier 1: Full Rust release matrix, Python (wheel + unittest + mypy --strict + stubtest), and WASM
./scripts/verify.sh --rust
./scripts/verify.sh --python
./scripts/verify.sh --wasm --build-wasm

# Tier 2: Strict single-thread/wasm32/metadata invariants, cross-target parity, and docs
./scripts/check_invariants.sh --mode strict
./scripts/verify.sh --parity
./scripts/verify.sh --docs --build-wasm

# Tier 3: Full integrated sign-off
./scripts/verify.sh --all --build-wasm --docs
```

Equivalent `Makefile` shortcuts include `make verify-fast`, `make verify-all`, `make invariants-strict`, `make parity`, `make python-build`, `make python-test`, `make python-docs`, `make python-doctest`, `make wasm-build`, and `make wasm-test`. See [CONTRIBUTING.md](CONTRIBUTING.md) for engineering invariants and contribution rules.

## Documentation & Resources

- **Python Docs (Sphinx)**: https://egraph-py-docs.netlify.app/
- **Migration Guide (v5 / alpha.0..8 -> alpha.9)**: [docs/MIGRATION_V6.md](docs/MIGRATION_V6.md)
- **Interactive Tutorials (Colab)**: [notebooks/README.md](notebooks/README.md)
- **Python Package README**: [crates/python/README.md](crates/python/README.md)
- **WebAssembly Package README**: [crates/wasm/README.md](crates/wasm/README.md)
- **Rust API Docs**: `cargo doc --workspace --no-deps --open`
- **Browser Examples**: `npm --prefix js/examples run dev`

## License

MIT

## Citation

If you use `egraph` in your research, please cite:

```bibtex
@software{egraph,
  title = {egraph: High-Performance Graph Drawing Library},
  author = {Yosuke Onoue},
  year = {2025},
  url = {https://github.com/likr/egraph-rs}
}
```
