# `egraph-rs` Project Overview

This document provides an overview of the `egraph-rs` v6 workspace architecture across Native Rust, Python (`abi3-py310`), WebAssembly/TypeScript (pure ESM), and the `egraph-cli` binary.

## Project Structure

`egraph-rs` is organized as a strictly single-threaded pure-CPU Rust workspace of **28 crates** (`24` publishable library crates + `4` non-publishable binding/CLI crates):

```text
egraph-rs/
├── crates/
│   ├── core/                    # egraph-core: CsrGraph, CsrGraphBuilder, Layout2d/3d/Nd, ManifoldLayout2d, EdgePolylines, NodeExtents2d, PinMask, DeterministicRng, 3 algorithm traits
│   ├── dataset/                 # egraph-dataset: Pure-CSR benchmark graphs (opt-in `embedded` feature), catalog metadata, and synthetic generators
│   ├── spatial-trees/           # egraph-spatial-trees: NearestNeighborIndex/SpatialIndex, StaticKdTree, DynamicKdTree, Quadtree2d, ArenaOctree3d, Delaunay2d (spade)
│   ├── spatial-fft/             # egraph-spatial-fft: R2C/C2C FFT and ibFFT / FitField / Screened Poisson 2D density fields
│   ├── spatial-crossings/       # egraph-spatial-crossings: Bentley-Ottmann sweep-line, sealed TorusView, torus, spherical, and hyperbolic crossings
│   ├── linalg/                  # egraph-linalg: LinearOperator, CsrMatrix, Laplacians, DijkstraWorkspace, spectral eigenpairs, Chebyshev kernels
│   ├── edge-routing/            # egraph-edge-routing: FdebSolver<L> operating in place on EdgePolylines<L>
│   ├── linalg-amg/              # egraph-linalg-amg: AMG-PCG solver, effective resistances, and resistance curvature flow
│   ├── linalg-mds/              # egraph-linalg-mds: LayoutGenerator structs (ClassicalMds, PivotMds, Hde, RdMds, KorenSpectral)
│   ├── neighborhood/            # egraph-neighborhood: Exact 1+2-hop CSR, encapsulated KnnGraph/SparseProbabilities/Sparse12HopPairs, DoG & rHGH landmarks
│   ├── clustering/              # egraph-clustering: CommunityDetector/Clusterer (Louvain, Leiden, Spectral, Label Propagation, K-Means++, coarsening)
│   ├── layout-sgd/              # egraph-layout-sgd: SgdSolver (Full, SgdPairBuffer, 3-tier Sparse Barrier DoG, Ortmann, Omega) and Sgd2Solver
│   ├── layout-embedding/        # egraph-layout-embedding: PacmapGraphSolver, LargeVisSolver, DrGraphSolver, GraphCneSolver, SnapTFdpSolver
│   ├── layout-neighbor/         # egraph-layout-neighbor: TsNetSolver, GuMapSolver, TFdpSolver, IbUmapSolver with typed repulsion backends
│   ├── layout-stress/           # egraph-layout-stress: StressMajorizationSolver (EuclideanPcg/GeodesicGradient), KamadaKawaiSolver, MaxentStressSolver
│   ├── layout-force/            # egraph-layout-force: ForceAtlas2Solver, YifanHuSolver, LinLogSolver, FruchtermanReingoldSolver, GemSolver
│   ├── layout-constraints/      # egraph-layout-constraints: DAG layering, VPSC/1D constraints, PRISM/GTree/Forbid, PfsPrimeOverlapRemoval & TwinDeclutterer transforms
│   ├── layout-multilevel/       # egraph-layout-multilevel: Leaf multilevel engines (GripSolver, CoarseBackbone, SfcLayout)
│   ├── layout-refine/           # egraph-layout-refine: Post-processing refiners (SpatialIntruderRefiner, ScreenedPoissonRefiner, ClusterRefineSolver)
│   ├── layout-composite/        # egraph-layout-composite: Composite multi-solver layouts (BackboneSolver, SelLayoutSolver, GroupInABoxSolver)
│   ├── metrics-geometry/        # egraph-metrics-geometry: Generic MetricSpace/EuclideanLayout SNS, geodesic distortion, crossings, NP1/NP2, S_RNG/S_GG, PRISM, RTA
│   ├── metrics-cluster/         # egraph-metrics-cluster: G-CAF, Adjusted Patch NIG, Sprawlter, NOR, and eval-baselines separation metrics
│   ├── metrics/                 # egraph-metrics: Selective MetricReport / MetricConfig / MetricInputs / evaluate_metrics* dispatcher
│   ├── egraph/                  # egraph-rs ([lib] name = "egraph"): Unified public Rust facade crate (`core`, `spatial`, `edge_routing`, `linalg`, `neighborhood`, `clustering`, `layout`, `metrics`, `dataset`, `prelude`)
│   ├── ffi/                     # egraph-ffi (publish=false): ErrorKind/FfiError, zero-copy InterleavedLayout2d/3d & EdgePolylines ABI helpers
│   ├── cli/                     # egraph-cli (publish=false): Standalone CLI (`layout`, `metrics`, `cluster`, `bench`)
│   ├── python/                  # egraph-python (publish=false): PyO3 `abi3-py310` bindings, NumPy zero-copy views, `EgraphError` hierarchy, `egraph.interop`, Sphinx docs
│   └── wasm/                    # egraph-wasm (publish=false): `wasm-bindgen` pure ESM package, zero-copy `coordsView()`, `.code` structured errors, `runProgressive()`, TypeDoc
├── docs/                        # API freeze specification (V6_API_FREEZE_SPEC.md), migration guide (MIGRATION_V6.md), and archived plans (archive/)
├── examples/web/                # Browser/Vite demos (`src/` + `dataset/`) and Playwright verification suite
├── notebooks/                   # 9 Google Colab-ready Python tutorial notebooks
└── scripts/                     # Unified verification (`verify.sh`), invariant checks (`check_invariants.sh`), parity harness, and worktree tooling
```

## Main Components

### Core Graph & Coordinate Buffers (`egraph-core`)

- **`CsrGraph` & `CsrGraphBuilder`**: Compact `u32`-indexed Compressed Sparse Row topology supporting unweighted `CsrGraph<()>` and weighted `float64` / `float32` edge weights with zero host object payloads.
- **`Layout2d`, `Layout3d` & `LayoutNd`**: Sealed `EuclideanLayout` Structure-of-Arrays (SoA) coordinate containers for 2D, 3D, and arbitrary $N$-D Euclidean embeddings (`float64` and `float32` in Rust/Python; `float32` linear-memory views in WASM), implementing `LayoutStorage`, `MetricSpace`, and `RiemannianLayout`.
- **`ManifoldLayout2d`**: Open `RiemannianLayout` non-Euclidean coordinate container supporting Poincaré hyperbolic disk (`"hyperbolicPoincare"` / `"hyperbolic_poincare"`), Lorentz hyperboloid (`"hyperbolicLorentz"` / `"hyperbolic_lorentz"`), unit sphere $S^2$ (`"spherical"`), and periodic flat 2D torus $T^2$ (`"torus"`), with Levi-Civita `parallel_transport_into` / `exp_map_and_transport`, physical `ManifoldMetric`, `torus_spectral_phase_init`, and `torus_auto_pan`.
- **`EdgePolylines<L>` & `NodeExtents2d` / `PinMask`**: Validated polyline storage (`edge_offsets` + `points: L`) for edge-routing solvers (`FdebSolver<L>`), plus shared 2D node bounding-shape views and fixed-node masks.
- **Cooperative & Deterministic 3-Trait Contract**: Iterative solvers implement `IterativeSolver<Target = ...>` (`step`, `run_steps`, `run`, `status` with `SolverStatus::Failed`, `iteration`, `total_iterations`), one-shot initializers implement `LayoutGenerator<Output = ...>`, and in-place post-processors implement `LayoutTransform<Target = ...>`, all using explicit `u64` seeds (`seed=42` default) and strictly single-threaded execution (`rayon` / `std::thread` forbidden).

### Layout Engines (`egraph-layout*` & `egraph-edge-routing`)

- **Compile-Time Typed Backends & Fail-Fast Space Selection**: Multi-backend solvers (`TsNetSolver<L, B>`, `TFdpSolver<L, B>`, `ForceAtlas2Solver<L, B>`, `LinLogSolver<L, B>`, `YifanHuSolver<L, B>`, `FruchtermanReingoldSolver<L, B>`, `MaxentStressSolver<L, B>`, `StressMajorizationSolver<L, S>`) promote the repulsion/solve backend to a type parameter with space-specific defaults (`Auto2d`/`Auto3d`, `BarnesHut`, `Exact`, `FitFft`, `LinearFft`, `IbFft`, `Grid`, `EuclideanPcg`, `GeodesicGradient`), catching incompatible layout/backend pairs at compile time in Rust and at constructor time in Python/WASM.
- **Track A (`TsNetSolver`)**: Pure 1+2-hop adaptive perplexity (`pure_2hop`) and direct row-normalized $A_1 + w_2 A_2$ affinity (`from_graph_row_normalized` with `TwoStepRandomWalk`), variable-length `TsNetStage` schedules, and `Auto2d` / `Auto3d` / `Exact` / `BarnesHut` / `FitFft` / `LinearFft` backends across 2D, 3D, $N$-D, and manifold spaces.
- **Track B (`SgdSolver::sparse_barrier_dog`, `GuMapSolver`, `SpatialIntruderRefiner`)**: $O(|E|)$ regularized Chebyshev Difference-of-Gaussians (DoG) Varadhan landmarks, 3-tier Sparse Barrier SGD (`SgdPairBuffer`), GuMap, and 2D `StaticKdTree` active-set spatial intruder refinement.
- **Separated Embedding, Neighbor, Force, Stress, Multilevel & Routing Engines**: Dedicated `Sgd2Solver`, `PacmapGraphSolver`, `LargeVisSolver`, `DrGraphSolver`, `GraphCneSolver`, `SnapTFdpSolver`, `TFdpSolver`, `IbUmapSolver`, `ForceAtlas2Solver`, `YifanHuSolver`, `MaxentStressSolver`, `KamadaKawaiSolver`, `StressMajorizationSolver`, `GripSolver`, `GemSolver`, `SelLayoutSolver`, `BackboneSolver`, `GroupInABoxSolver`, `ClusterRefineSolver`, `ScreenedPoissonRefiner`, `FdebSolver<L>` (`EdgePolylines<L>`), `LayoutGenerator` initializers (`ClassicalMds`, `PivotMds`, `Hde`, `RdMds`, `KorenSpectral`, `CoarseBackbone`, `SfcLayout`), and `LayoutTransform` overlap/declutter passes (`PfsPrimeOverlapRemoval`, `TwinDeclutterer`, alongside iterative `PrismOverlapRemoval`, `GTreeOverlapRemoval`, `ForbidOverlapRemoval`).

### Community Detection & Quality Metrics (`egraph-clustering`, `egraph-metrics*`)

- **Unified Clustering (`CommunityDetector` / `Clusterer`)**: `louvain_communities` (`LouvainResult`), `leiden_communities` (`LeidenResult`), `kmeans_plus_plus` (`KMeansResult`), `label_propagation` (`LabelPropagationResult`), and `spectral_clustering` (`SpectralClusteringResult`), plus `coarsen_graph`.
- **Unified `MetricSpace` / `EuclideanLayout` Metrics & `MetricReport`**: Additive `MetricReport` and `evaluate_metrics_with_inputs` covering Scale-Normalized Stress (`SNS` & pivot-sampled `SNS`), `geodesic_distortion`, Bentley-Ottmann sweep-line crossings, spherical/hyperbolic/torus geodesic crossings, `StaticKdTree` `NP1`/`NP2`, `S_RNG`/`S_GG`, `G-CAF`, `landmark_stratified_rta`, `adjusted_patch_nig`, `NOR`, `sprawlter_nn`, and opt-in `eval-baselines` cluster separation baselines.

## Language Interfaces & Tooling

- **Native Rust (`egraph-rs` / `use egraph::*`)**: Direct access to all 24 publishable crates via Cargo package `egraph-rs` (`[lib] name = "egraph"`) and curated `egraph::prelude::*` (`MSRV 1.88`, `edition = "2024"`, opt-in `serde` and `baselines` / `eval-baselines`).
- **Command-Line Interface (`egraph-cli`)**: `layout`, `metrics`, `cluster`, and `bench` subcommands with 2D, 3D (`Layout3d`), `--dims`, and `--manifold` JSON I/O.
- **Python (`crates/python`)**: PyO3 `abi3-py310` wheels with zero-copy NumPy buffers (`Layout2d`, `Layout3d`, `LayoutNd`, `ManifoldLayout2d`), UAF-safe `EdgePolylines`, `EgraphError` exception hierarchy, keyword-only `*` hyperparameters, `Protocol`/`TypedDict`/`Literal` `.pyi` stubs, `egraph.interop` (`scipy.sparse`, `networkx`, `igraph`), `mypy --strict`, `stubtest`, and Sphinx documentation.
- **WebAssembly (`crates/wasm`)**: Pure ESM single-`.wasm` package with zero-copy `coordsView()` / `edgeOffsetsView()` / `pointsView()` / `labelsView()`, polymorphic `runProgressive()`, `camelCase` options, `number | bigint` 64-bit seeds, strict `"float32"` `CsrGraph`, `.code` structured errors, TypeScript `LayoutStorage` / `IterativeSolver` / `LayoutGenerator` / `LayoutTransform` interfaces, TypeDoc, and `examples/web/` Vite demos.
