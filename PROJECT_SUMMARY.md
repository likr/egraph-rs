# `egraph-rs` Project Overview

This document provides an overview of the `egraph-rs` v6 workspace architecture (`6.0.0-alpha.8`) across Native Rust, Python (`abi3-py310`), WebAssembly/TypeScript (pure ESM), and the `egraph-cli` binary.

## Project Structure

`egraph-rs` is organized as a strictly single-threaded pure-CPU Rust workspace of **27 crates**:

```text
egraph-rs/
├── crates/
│   ├── core/                    # egraph-core: CsrGraph, Layout2d, LayoutNd, ManifoldLayout2d, DeterministicRng, IterativeSolver
│   ├── ffi/                     # egraph-ffi: Zero-copy interleaved coordinate mirrors and C ABI helpers
│   ├── petgraph/                # egraph-petgraph: Graph adapter utilities for petgraph
│   ├── dataset/                 # egraph-dataset: Embedded benchmark graphs, catalog metadata, and synthetic generators
│   ├── spatial/                 # egraph-spatial: Umbrella crate re-exporting spatial primitives
│   ├── spatial-trees/           # egraph-spatial-trees: StaticKdTree, DynamicKdTree, ArenaOctree3d, Delaunay triangulation
│   ├── spatial-fft/             # egraph-spatial-fft: R2C/C2C FFT and ibFFT / Screened Poisson 2D density fields
│   ├── spatial-crossings/       # egraph-spatial-crossings: Bentley-Ottmann sweep-line, torus, spherical, and hyperbolic crossings
│   ├── linalg/                  # egraph-linalg: CsrMatrix, Laplacians, DijkstraWorkspace, spectral eigenpairs, Chebyshev kernels
│   ├── linalg-amg/              # egraph-linalg-amg: AMG-PCG solver, effective resistances, and resistance curvature flow
│   ├── linalg-mds/              # egraph-linalg-mds: Classical MDS, PivotMDS, HDE, RD-MDS, and Koren spectral init
│   ├── neighborhood/            # egraph-neighborhood: Exact 1+2-hop CSR, KnnGraph, SparseProbabilities, DoG & rHGH landmarks
│   ├── cluster/                 # egraph-clustering: Louvain, Leiden, Spectral Clustering, Label Propagation, K-Means++, coarsening
│   ├── metrics/                 # egraph-metrics: Umbrella crate and selective MetricReport / evaluate_metrics dispatcher
│   ├── metrics-geometry/        # egraph-metrics-geometry: SNS, geodesic distortion, crossings, NP1/NP2, S_RNG/S_GG, PRISM distortion
│   ├── metrics-cluster/         # egraph-metrics-cluster: G-CAF, RTA, Adjusted Patch NIG, Sprawlter, NOR, and separation baselines
│   ├── layout/                  # egraph-layout: Umbrella crate re-exporting unified v6 layout solvers
│   ├── layout-constraints/      # egraph-layout-constraints: DAG cycle removal, transitive reduction, layering, VPSC, TwinDeclutterer
│   ├── layout-stress/           # egraph-layout-stress: Stress Majorization, Kamada-Kawai, Maxent-Stress, GRIP, SFC
│   ├── layout-force/            # egraph-layout-force: t-FDP, ibUMAP, ForceAtlas2, Fruchterman-Reingold, LinLog, GEM, SEL, FDEB, ScreenedPoissonRefiner
│   ├── layout-neighbor/         # egraph-layout-neighbor: TsNetSolver (Track A), GuMapSolver (Track B), SpatialIntruderRefiner
│   ├── layout-sgd/              # egraph-layout-sgd: SgdSolver (Full, 3-tier Sparse Barrier DoG, PaCMAP-Graph, SNAP-tFDP, Omega, LargeVis, DRGraph, Graph-CNE, SGD2)
│   ├── layout-multilevel/       # egraph-layout-multilevel: Yifan Hu, Backbone, Group-in-a-Box, ClusterRefine, PRISM/GTree/Forbid/PFS' overlap removal
│   ├── egraph/                  # egraph: Unified public Rust facade crate with prelude
│   ├── cli/                     # egraph-cli: Standalone CLI (`layout`, `metrics`, `cluster`, `bench`)
│   ├── python/                  # egraph-python: PyO3 `abi3-py310` bindings, NumPy zero-copy views, `egraph.interop`, Sphinx docs
│   └── wasm/                    # egraph-wasm: `wasm-bindgen` pure ESM package, zero-copy `coordsView()`, `runProgressive()`, TypeDoc
├── docs/                        # Migration guide, remediation plan, and architecture documents
├── js/examples/                 # Browser/Vite demos and Playwright verification suite
├── notebooks/                   # 9 Google Colab-ready Python tutorial notebooks
└── scripts/                     # Unified verification (`verify.sh`), invariant checks, parity harness, and worktree tooling
```

## Main Components

### Core Graph & Coordinate Buffers (`egraph-core`)

- **`CsrGraph`**: Compact `u32`-indexed Compressed Sparse Row topology with optional `float64` or `float32` edge weights and zero host object payloads.
- **`Layout2d` & `LayoutNd`**: Zero-copy Structure-of-Arrays (SoA) coordinate containers for 2D and arbitrary $N$-D Euclidean embeddings (`float64` and `float32` in Rust/Python; `float32` linear-memory views in WASM).
- **`ManifoldLayout2d`**: Non-Euclidean coordinate container supporting Poincaré hyperbolic disk (`"hyperbolic"`), Lorentz hyperboloid (`"hyperbolic_lorentz"`), unit sphere $S^2$ (`"spherical"`), and periodic flat 2D torus $T^2$ (`"torus"`), with physical `ManifoldMetric` (`scale`, `aspect`, `rescale_geodesic`), `torus_spectral_phase_init`, and `torus_auto_pan`.
- **Cooperative & Deterministic Execution**: All iterative solvers implement `step()`, `run_steps(layout, steps)`, and `run(layout)` with explicit `u64` seeding (`seed=42` default) and zero threading (`rayon` / `std::thread` forbidden).

### Layout Engines (`egraph-layout*`)

- **Track A (`TsNetSolver`)**: Pure 1+2-hop adaptive perplexity (`pure_2hop`) and direct row-normalized $A_1 + w_2 A_2$ affinity (`from_graph_row_normalized` with `TwoStepRandomWalk`), supporting `"auto"`, `"fft"`, `"linear_fft"`, `"barnes_hut"`, and `"exact"` repulsion backends across 2D, N-D, and manifold spaces.
- **Track B (`SgdSolver.sparse_barrier_dog`, `GuMapSolver`, `SpatialIntruderRefiner`)**: $O(|E|)$ regularized Chebyshev Difference-of-Gaussians (DoG) Varadhan landmarks, 3-tier Sparse Barrier SGD, GuMap, and 2D `StaticKdTree` active-set spatial intruder refinement.
- **3D / N-D & Manifold Engines**: `ArenaOctree3d` and `DynamicKdTree` power **11 N-D solvers** (`LayoutNd`) and **6 manifold solvers** (`SgdSolver`, `TsNetSolver`, `GuMapSolver`, `StressMajorizationSolver`, `KamadaKawaiSolver`, `FruchtermanReingoldSolver`).
- **Force, Multilevel, Stress & Bundling Models**: `TFdpSolver`, `IbUmapSolver`, `SgdSolver.snap_t_fdp`, `SgdSolver.pacmap_graph`, `SgdSolver.sgd2`, `SgdSolver.ortmann_sparse_stress`, `SgdSolver.largevis`, `SgdSolver.drgraph`, `SgdSolver.graph_cne`, `ForceAtlas2Solver`, `YifanHuSolver`, `MaxentStressSolver`, `GripSolver`, `GemSolver`, `SelLayoutSolver`, `SfcLayoutSolver`, `BackboneSolver`, `GroupInABoxSolver`, `ClusterRefineSolver`, and decoupled `FdebSolver(graph)`.
- **Overlap Removal & Constraints**: `PrismOverlapRemoval`, `GTreeOverlapRemoval`, `ForbidOverlapRemoval` (default `"sparse_forbid"`), `PfsPrimeOverlapRemoval`, `TwinDeclutterer` / `declutter_twins_2d`, `ScreenedPoissonField2d` / `ScreenedPoissonRefiner`, `transitive_reduction`, DAG layering, and 1D/2D `SeparationConstraint` projection.

### Community Detection & Quality Metrics (`egraph-clustering`, `egraph-metrics*`)

- **5 Typed Zero-Copy Cluster Result Classes**: `louvain_communities` (`LouvainResult`), `leiden_communities` (`LeidenResult`), `kmeans_plus_plus` (`KMeansResult`), `label_propagation` (`LabelPropagationResult`), and `spectral_clustering` (`SpectralClusteringResult`), plus `coarsen_graph` and `coarse_backbone_init`.
- **Selective `evaluate_metrics` -> `MetricReport`**: Scale-Normalized Stress (`SNS` & pivot-sampled `SNS`), `geodesic_distortion`, Bentley-Ottmann sweep-line crossings, spherical/hyperbolic/torus geodesic crossings, `StaticKdTree` `NP1`/`NP2`, Eades et al. (2017) `S_RNG`/`S_GG` (`drng_preservation`, `dgg_preservation`), `G-CAF` (`cluster_area_faithfulness`), `landmark_stratified_rta`, `adjusted_patch_nig`, `NOR` (`nor_metrics`), `sprawlter_nn`, and cluster separation baselines.

## Language Interfaces & Tooling

- **Native Rust (`egraph`)**: Direct access to all 27 crates via `egraph::prelude::*`.
- **Command-Line Interface (`egraph-cli`)**: `layout`, `metrics`, `cluster`, and `bench` subcommands with 2D, `--dims`, and `--manifold` JSON I/O.
- **Python (`crates/python`)**: PyO3 `abi3-py310` wheels with zero-copy NumPy buffers, `egraph.interop` (`scipy.sparse`, `networkx`, `igraph`), `mypy --strict`, `stubtest`, and Sphinx documentation (`251` public symbols).
- **WebAssembly (`crates/wasm`)**: Pure ESM single-`.wasm` package with zero-copy `coordsView()` / `labelsView()`, polymorphic `runProgressive()`, TypeScript declarations, TypeDoc, and `js/examples/` Vite demos.
