# Changelog

All notable changes to `egraph-rs` (`egraph` on PyPI and npm) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [6.0.0-beta.1] - 2026-11-28

### Added

- **4-Layer Geometry Traits & First-Class `Layout3d<S>` / `EdgePolylines<L>` (`egraph-core`, `egraph-edge-routing`)**:
  - Added open `LayoutStorage` -> `MetricSpace` -> `RiemannianLayout` (with `exp_map_and_transport` so `parallel_transport_into` executes before `exp_map`) and sealed `EuclideanLayout` (`Layout2d<S>`, `Layout3d<S>`, `LayoutNd<S>`) backed by hidden zero-cost `EuclideanStorageRef`/`EuclideanStorageMut` views.
  - Added `Layout3d<S>` (SoA `{x, y, z}` in Rust, zero-copy `(N, 3)` NumPy views in Python, interleaved `Float32Array` `coordsView()` in WASM) and `EdgePolylines<L>` with new `crates/edge-routing` (`egraph-edge-routing`) housing `FdebSolver<L>` (`IterativeSolver<Target = EdgePolylines<L>>`).
  - Added `NodeExtents2d<'a, S>`, `NodeShape2d<S>`, `PinMask`, `CsrGraphBuilder<W>`, `CsrGraph<()>` unweighted constructors, `NearestNeighborIndex`/`SpatialIndex`, `LinearOperator<S>`, `CommunityDetector`/`Clusterer`, `SgdPairBuffer<S>`, and opt-in `serde` feature on core/facade crates.
- **28-Crate Parallel Workspace & CI Acceleration**:
  - Split `layout-embedding` (`PacmapGraphSolver`, `LargeVisSolver`, `DrGraphSolver`, `GraphCneSolver`, `SnapTFdpSolver`), `layout-refine` (`SpatialIntruderRefiner`, `ScreenedPoissonRefiner`, `ClusterRefineSolver`), `layout-composite` (`BackboneSolver`, `SelLayoutSolver`, `GroupInABoxSolver`), and `edge-routing`; renamed `crates/cluster` -> `crates/clustering`; eliminated thin umbrellas `crates/spatial` and `crates/layout` (`egraph::spatial` and `egraph::layout` live on the `egraph` facade).
  - Consolidated 28 `crates/wasm/tests/` binaries into a single `wasm_suite` integration test binary, batched `wasm32` checks in `scripts/check_invariants.sh`, and parallelized `ci.yml` (`rust-core` + `rust-feature-matrix` + `semver-checks` + `uv`/npm/Playwright caching), cutting CI duration from ~12.5m to ~4.6m.

### Changed

- **Compile-Time Typed Backends & Open Repulsion Traits**:
  - Promoted repulsion/strategy backends to type parameter `B` (`TsNetSolver<L, B>`, `TFdpSolver<L, B>`, `ForceAtlas2Solver<L, B>`, `LinLogSolver<L, B>`, `YifanHuSolver<L, B>`, `FruchtermanReingoldSolver<L, B>`, `MaxentStressSolver<L, B>`, `StressMajorizationSolver<L, S>`) with open repulsion traits and associated `Default*Backend` types (`Auto2d` on `Layout2d`, `Auto3d` on `Layout3d`, `Exact` on `LayoutNd`), eliminating runtime `TypeId::of` dispatch, dead `theta`/`ibfft` config fields, and invalid `(L, B)` combinations at compile time.
- **Role-Separated Traits & One-Algorithm-Per-Type**:
  - Unified `IterativeSolver { type Target: ?Sized }` (with `run_steps(target, 0)` pre-validation, `SolverStatus::Failed`, `total_iterations() -> Option<usize>`, and `&mut T`/`Box<T>` blanket impls), `LayoutGenerator { type Output }` (`PivotMds`, `ClassicalMds`, `Hde`, `RdMds`, `KorenSpectral`, `CoarseBackbone`, `SfcLayout`), and `LayoutTransform { type Target: ?Sized }` (`PfsPrimeOverlapRemoval`, `TwinDeclutterer`).
  - Split `FieldLayoutSolver` into `TFdpSolver` and `IbUmapSolver`, separated `Sgd2Solver` + the 5 `layout-embedding` classes from `SgdSolver`, made `TsNetConfig` use variable-length `stages: Vec<TsNetStage>`, standardized `u32` layer/cluster IDs and `&[T]` inputs, and generalized `metrics*` over `MetricSpace`/`EuclideanLayout` with additive `eval-baselines` (`ClusterBaselineConfig`/`ClusterBaselineReport`) and separated `np_num_samples`/`np_seed`.
- **SemVer Hardening, MSRV `1.88` & Publishing Metadata**:
  - Declared workspace `rust-version = "1.88"` and metadata across all 24 publishable library crates (setting `publish = false` on `egraph-{ffi,python,wasm,cli}`), added `#[non_exhaustive]` across public enums/configs/reports/errors, removed `Copy` from evolvable config/report/stats/stage/result structs, encapsulated `SparseProbabilities` and `Sparse12HopPairs`, sealed `pub mod` internals behind flat crate-root re-exports, curated `egraph::prelude`, and switched Python classifiers to `Development Status :: 4 - Beta`.
- **Python, WASM/TS & Repo Layout Freeze**:
  - Python: Added explicit `target=` parameter (`"layout2d" | "layout3d" | "layout_nd" | "manifold2d"`) with fail-fast validation, UAF-safe `PyEdgePolylines`, `EgraphError(ValueError)` with 7 typed subclasses, keyword-only `*` hyperparameters, frozen Result structs, and modernized `.pyi` (`Protocol`/`TypedDict`/`Literal`) verified via `mypy --strict` and `stubtest`.
  - WASM/TS: Migrated all 620 error sites from `JsError` to `Result<T, JsValue>` with `.code: EgraphErrorCode`, converted 34+ multi-arg functions to `{ ...options }`, standardized strict `camelCase` tokens, added `number | bigint` 64-bit seeds, enforced `weightDtype: "float32"`, and updated `memory.grow`-safe `Layout3d`/`EdgePolylines` views.
  - Moved `js/` -> `examples/web/` (flat depth-2 Vite layout), moved historical plans into `docs/archive/`, separated `egraph-dataset` into lightweight `dataset` vs opt-in `dataset-embedded` (`9` CSVs excluded from `.wasm`), and updated `notebooks/01..09_*.ipynb` + Sphinx/TypeDoc docs.

### Fixed

- **Non-Euclidean Manifold Geometry & Mixed-Layout Soundness**:
  - Implemented closed-form Levi-Civita `parallel_transport_into` + `exp_map_and_transport` across `Torus2d`, `Spherical`, `HyperbolicPoincare`, and extreme-radius-safe `HyperbolicLorentz`; fixed `KamadaKawaiSolver` tangent Hessian positive-definiteness check, `TsNetSolver` velocity parallel transport, `PacmapGraphSolver` rotation-invariant Riemannian Adam, and `Sgd2Solver` `SO(2)`-invariant trace preconditioning; eradicated 2D/ND scratch aliasing bugs (`ActiveLayoutKind` removed, `LayoutNd` `dims == 3` branch removed, `ManifoldKind` locked per solver).

### Removed

- **`petgraph`, `ContinuousFdp`, `delaunay` Feature Fallback & Redundant Shims**:
  - Deleted `crates/petgraph` and all `petgraph` dependencies across the workspace, removed `ContinuousFdp` and the `delaunay` feature (making `spade` unconditional), eliminated `MetricLayout`/`SgdScheduleWindowLayout`/`NdSpatialIndex`/`*_nd`/`*_manifold` duplicates, sealed `TorusLayoutView` (removing implicit `Layout2d` impl), and encapsulated `ChaCha8Rng`/`RngCore` behind `DeterministicRng`.

## [6.0.0-alpha.9] - 2026-11-27

### Added

- **3D & $N$-Dimensional Layout Engine (`egraph-spatial-trees`, `egraph-layout*`, `egraph-metrics*`, Python, WASM, CLI)**:
  - Added `ArenaOctree3d` (3D Barnes–Hut), `DynamicKdTree`, `LayoutNd` (`.dims`, `copy()`, `get()`, `set()`, `normalize_std()`), `evaluate_metrics_nd`, and **11 $N$-D-capable solvers** (`SgdSolver`, `TsNetSolver`, `GuMapSolver`, `StressMajorizationSolver`, `KamadaKawaiSolver`, `MaxentStressSolver`, `FruchtermanReingoldSolver`, `LinLogSolver`, `ForceAtlas2Solver`, `YifanHuSolver`, `FdebSolver`).
- **Non-Euclidean Manifold Engine (`egraph-core`, `egraph-spatial-crossings`, `egraph-layout*`, `egraph-metrics*`, Python, WASM, CLI)**:
  - Added `ManifoldKind::HyperbolicLorentz` (`"hyperbolic_lorentz"` + `to_poincare_layout()`), physical `ManifoldMetric` (`scale`, `aspect`, `set_scale()`, `set_aspect()`, `set_torus_extents()`, `rescale_geodesic()`), **6 manifold solvers** (`SgdSolver`, `TsNetSolver`, `GuMapSolver`, `StressMajorizationSolver`, `KamadaKawaiSolver`, `FruchtermanReingoldSolver`), `geodesic_distortion`, exact spherical/hyperbolic/torus geodesic crossings (`crossing_edges_torus`, `crossing_edges_spherical`, `crossing_edges_hyperbolic`), `torus_spectral_phase_init`, `torus_auto_pan`, `torus_edge_segments`, `torus_layout_edge_segments`, and `SgdSolver.reset_schedule` / `run_schedule_window`.
- **Unified Metrics & Zero-Copy Cluster Result Classes (`egraph-metrics`, `egraph-clustering`, Python, WASM)**:
  - Added typed `MetricReport` returned by `evaluate_metrics` / `evaluateMetrics` across 2D, $N$-D, and manifold spaces (`to_dict(include_none=False)` / `toRecord()`).
  - Replaced raw tuple/dict clustering returns with 5 typed zero-copy Result classes: `LouvainResult`, `LeidenResult`, `KMeansResult`, `LabelPropagationResult`, and `SpectralClusteringResult`.
- **Neighborhood, Solvers, Decluttering, Interop & Datasets**:
  - Added `TsNetBackend::Auto` (`"auto"`, `.resolved_backend`), `TwoStepRandomWalk` mode in `TsNetSolver.from_graph_row_normalized`, `TsNetStage` + `TsNetSolver.from_probabilities_with_stages`, `SgdSolver.from_packed_symmetric`, `TwinDeclutterer` / `declutter_twins_2d`, `transitive_reduction`, `symmetric_eigen_jacobi`, `DijkstraWorkspace` (`dijkstra_single_source`, `all_pairs_dijkstra`), `CsrGraph` inspection helpers (`__len__`, `edges()`, `has_edge()`, `edge_weight()`, `is_connected()`, `weight_dtype`), `egraph.interop.to_igraph` & expanded `to_networkx(layout=...)`, `list_datasets` / `dataset_catalog` / `dataset_info` + 7 synthetic graph generators, and `egraph-cli` subcommands/flags (`tsnet-rn`, `twin-declutter`, `--dims`, `--manifold`).
- **Documentation, Tutorials & Release Build Profiles**:
  - Completed 251/251 Sphinx API coverage with doctests, updated `docs/MIGRATION_V6.md`, added 9 Google Colab tutorial notebooks (`notebooks/01..09_*.ipynb`), and added `[profile.dist-release]` (`lto = "thin"`, `codegen-units = 1`) in `Cargo.toml` for published release wheels.

### Changed

- **Cross-Target Clean-Break Unification**: Standardized fixed-iteration solvers on `iterations`, convergent solvers on `max_iterations`, multi-stage solvers on structured names (`stage_iterations`, `phase_iterations`, `steps_per_cycle`), clustering/metric arguments on `labels`, stochastic defaults on `seed = 42`, decoupled `FdebSolver(graph)`, and polymorphic `IterativeSolver::step` / `run_steps` / `run` and WASM `runProgressive` across `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`.
- **Safer & Smarter Defaults**: Set `ForbidOverlapRemoval` default to `"sparse_forbid"`, `GemSolver` default `max_iterations = min(4 * N, 400)`, `TsNetSolver` `"fft"` defaults to `interpolation_order = 3` and `learning_rate_divisor = 3.0`, `ScreenedPoissonField2d` default `grid_size = 512`, `EffectiveResistanceConfig` defaults to `exact_max_nodes = 256` and `jl_dim = 128`, and `SgdSolver` default `max_mu = None`.
- **Verification Performance**: Split `[profile.release]` to parallel `lto = false` for fast local/CI checks (reserving LTO + single CGU for `[profile.dist-release]` and `[profile.wasm-release]`) and batched `scripts/verify.sh --rust` feature matrices.

### Fixed

- **Numerical & Cross-Target Precision Hardening**: Stabilized Euclidean and torus sweep-line crossings across full finite `f64` exponents, hardened Lorentz distance/log/exp/midpoint across extreme scales and high ratios (`lorentz_scaled_cross` 2-FMA scaling, single-step subnormal `scale_pow2`, `[1, 2) * f64::MAX` multiplication order in `lorentz_exp_map`, subnormal continuity, and `wasm32` software-FMA parity), guarded `TwinDeclutterer` against large-coordinate overflow, locked layout kind/dims across `SgdSolver` and `GuMapSolver`, and ensured `npm ci --prefix crates/wasm` installs `tsc` before `node_esm_smoke.mjs` in CI and release pipelines.

### Removed

- **Legacy v5 Workspace & Compatibility Shims**: Removed all 18 legacy `petgraph-*` crates and `--features legacy` shims (`45 -> 27` clean v6 crates), `*_v6` suffixes, legacy `SgdSolver.{step,run_steps,run}_manifold` methods, and `remove_cycle`.

## [6.0.0-alpha.8] - 2026-11-26

### Added

- **G-CAF v2 Cluster Area Faithfulness (`egraph-metrics`, `egraph`, `egraph-cli`, Python, WASM)**:
  - Added `cluster_area_faithfulness`, `cluster_area_faithfulness_detail`, and `MetricTarget::ClusterAreaFaithfulness` (`gcaf`) with $O(N + K)$ CSR bucketing, `VisitStamp` sparse grid reuse, deterministic shuffled Welzl MEC normalization, Stevens $\gamma$ scaling, positive arbitrary `target_weights`, `effective_boundary_r95` / `g_eff` reporting, and 8-connected spatial coherence.
  - Added `egraph metrics` CLI `--gcaf-*` hyperparameter flags, `--gcaf-detail` JSON reporting, `adjusted-patch-nig` target registration, and automatic `gcaf` / `adjusted-patch-nig` expansion under `--targets all` when `--community-labels` is supplied.
- **Phase W1 Priority Baselines (`baselines` & `eval-baselines` features)**:
  - Implemented strictly single-threaded native `Leiden` community detection (Modularity & CPM) in `egraph-clustering`.
  - Implemented Noack (2007) `LinLogSolver` (`NodeRepulsion` / `EdgeRepulsion`), `FruchtermanReingoldSolver`, `ForceAtlas2Config::prevent_overlap`, Quadrilateral Simmelian `BackboneSolver`, Cai (2024) `ClusterRefineSolver`, `GroupInABoxSolver` (Squarified Treemap area allocation), `GTreeOverlapRemoval`, and `ForbidOverlapRemoval` in `egraph-layout`, exposed across Rust, Python, WASM, and CLI.
  - Added `cluster_baselines` quality metrics (`CQ-kmeans`, `CQ-HAC`, `ARI`, `AMI`, `HOM`, `V-Measure`, `FMI`, `DSC`, `DC`, `CDM`, `HDM`, `SepMe`, `M_NH`, `Silhouette`, `Calinski-Harabasz`, `Davies-Bouldin`, `Pairwise Distance-Overlap Error`, `Sprawlter-NN`, and Chen et al. 2020 NOR-5 metrics).
- **Phase W2 Layout & Spectral Diversity Baselines**:
  - Added Harel–Koren High-Dimensional Embedding (`HDE` in `egraph-linalg`), Ortmann Sparse Stress (`SgdSolver::ortmann_sparse_stress`), `MaxentStressSolver` (using `AmgPcgSolver` + general $q$-power Barnes–Hut `ArenaQuadtree2d`), deterministic alias-table negative-sampling DR presets (`SgdSolver::{largevis, drgraph, graph_cne}`), `GripSolver`, `GemSolver`, `SelLayoutSolver`, `PfsPrimeOverlapRemoval`, and `SfcLayoutSolver`, with cross-target parity checks (`scripts/parity_harness.py`) and WASM integration tests (`v6_w1`, `v6_w2`).

### Changed

- **WASM Packaging Policy**: Removed the `600 KiB` WASM gzip size limit enforcement gate from invariant scripts, CI/release workflows, and templates while retaining informational size reporting.
- **Specification Clarifications**: Documented exact spec distinctions for `ForceAtlas2Config::linlog_mode` (Jacomy 2014 attraction vs. Noack 2007 `LinLog`), `SgdSolver::sgd2` (diagonal-preconditioned stress SGD), `proximity.rs` (`S_RNG` / `S_GG`, Eades et al. 2017 unweighted shape metrics), and `project_clustered_rectangle_no_overlap_2d` (single-shot rigid cluster-AABB VPSC post-processing).

### Fixed

- **G-CAF v2 Boundary & Numerical Hardening**: Clamped `g_eff` rather than silently shrinking `boundary_r95` when `g_max > 510`, updated `spatial_coherence` to peak-protected single-pass active thresholding, aligned Step 3 deconvolution/density resolution targets with `p[k]` for Stevens $\gamma \ne 1$ and custom weights, fixed `proximity.rs` `-0.0` diameter and edge cases, ensured strict orthogonal-order preservation (`oo_nni == 0`) in `PfsPrimeOverlapRemoval`, and relaxed `mec_100k_circle_timing` wall-clock thresholds for shared CI runners.

## [6.0.0-alpha.7] - 2026-11-24

### Added

- **Python `egraph.__version__` Export**: Exposed `egraph.__version__` in PEP 440 form (`6.0.0a7`, derived from `CARGO_PKG_VERSION`) via `egraph._core` and re-exported it in `egraph/__init__.py` and `_core.pyi`.
- **Stage 1 W1-W3 Core, Initialization & Neighborhood Constructors**:
  - Added Poincare Mobius centroid centering (`poincare_mobius_add`, `ManifoldLayout2d` safe centroid initialization / projection) in `egraph-core`, Python, and WASM.
  - Added `robust_normalize_layout_2d`, streaming `PivotMds` low-memory projection, degree-decayed `graph_12hop_row_normalized_probabilities`, and capped `build_capped_12hop_knn` across Rust, Python, and WASM.
  - Added `keep_all_1hop` / `max_1hop_per_node`, `degree_weight_beta`, and external landmark distance support on `PacmapGraphConfig` / `SgdSolver.pacmap_graph`, plus opt-in short-range minimum-separation hinge (`min_sep_ratio`, `w_sep`) on `SpatialIntruderRefiner`.
- **Stage 1 W3 Geometric & Quality Metrics (`egraph-metrics`, `egraph-cli`, Python, WASM)**:
  - Added `crosslessness`, `sampled_crosslessness`, `minimum_angle`, `minimum_angle_summary`, `layout_degeneracy`, `adjusted_patch_nig`, `adjusted_patch_nig_multiscale`, `landmark_stratified_rta`, `landmark_stratified_rta_detail`, and `landmark_stratified_rta_from_landmarks`, along with `egraph-cli` targets (`crosslessness`, `minimum-angle`, `layout-degeneracy`, `rta`) and geometric oracle tests.
- **Stage 2A Non-SPD Ideal Distances & Resistance Curvature Flow (`egraph-linalg`, `egraph-neighborhood`)**:
  - Ported and hardened strictly single-threaded pure-Rust `AmgPcgSolver` on `CsrMatrix` (`egraph-linalg`), standalone `anisotropic_laplacian` ($L_\alpha$) with dynamic spectrum scaling, `chebyshev_heat_spmv_matrix`, `chebyshev_dog_jl_embedding_matrix`, `compute_effective_resistances`, and Devriendt–Lambiotte `resistance_curvature_flow` integrated with `ManifoldLayout2d`.
  - Added `compute_pivoted_rhgh_landmarks` (`RHGH` / `BFSR`) and `compute_regularized_dog_varadhan_landmarks_from_matrix`.
- **Stage 2B Coarse Louvain Area Reservation & Screened Poisson Density Field (`egraph-spatial`, `egraph-clustering`, `egraph-layout`)**:
  - Added multi-level Louvain coarsening (`louvain_coarsen`, `project_clustered_layout_with_radii`, `align_clustered_centroids`) and coarse backbone initialization (`coarse_backbone_init`) with edge-length-first area reservation.
  - Added `f32` R2C FFT `ScreenedPoissonField2d` and `ScreenedPoissonRefiner` unilateral upper-bound density refinement across Rust, Python, and WASM.
- **Cross-Target Documentation & Parity**: Documented Stage 1, 2A, and 2B APIs in Sphinx reference (`crates/python/docs/source/api/`), updated WASM TypeScript definitions (`assemble_pkg.mjs`), and extended `scripts/parity_harness.py` for full Rust / Python / WASM parity.

### Changed

- **WASM Bundle Budget Liberation (`W0`)**: Removed `egraph-dataset` embedded CSVs from `egraph-wasm`, enabled `[profile.wasm-release] lto = "fat"`, generalized `parse_csr_csv<W: Scalar>`, and switched `js/examples` embedded graph loaders to async JSON imports to keep `egraph_wasm_bg.wasm` well below the `600 KiB` gzipped budget.
- **Solver Memory & Parameter Improvements (`W2`)**: Reduced peak allocation in `GuMapSolver` / `IbUmapSolver` (staged drops, `f32` edge weights, zero `raw_edges`) and `FieldLayoutSolver::step()` (eliminated per-step `self.mode.clone()`), updated `ForceAtlas2Config` / `YifanHuConfig` default `theta` (`1.2`) and `YifanHuConfig` convergence parameters, clamped `FdebSolver` displacement (`(alpha * kp).min(0.45)`), and upgraded `sampled_scale_normalized_stress` to degree-stratified pivot selection (64–128 default pivots).

### Fixed

- **Small-Graph `TsNetSolver` Stage 3 Stability & Closed-Twin Separation**: Aligned `TsNetSolver` Stage 3 `learning_rate` cap ($2N$ for $N < 100$), fixed `YifanHuConfig` step-ratio convergence stagnation, corrected `PacmapGraphConfig` undirected 1-hop duplicate registration, and eliminated closed-twin zero-length edge collapse in `SpatialIntruderRefiner`.
- **`RHGH` / `BFSR` & `AmgPcgSolver` Correctness**: Resolved four `RHGH` / `BFSR` scaling/pivot bugs and six `AmgPcgSolver` solver defects under strictly single-threaded deterministic execution.

## [6.0.0-alpha.6] - 2026-11-23

### Added

- **Row-Normalized 1+2-Hop tsNET (`TsNetSolver.from_graph_row_normalized` / `TsNetSolver.fromGraphRowNormalized`)**: Added `graph_12hop_row_normalized_probabilities` and `TsNetSolver` constructor supporting `w2` weighting and optional `max_2hop_per_node` capping across Rust, Python, and WASM.
- **Sparse 1+2-Hop Pair Extraction (`extract_sparse_1hop_2hop_pairs` / `extractSparse1hop2hopPairs`)**: Exposed `Sparse12HopPairs` and `extract_sparse_1hop_2hop_pairs(graph, max_2hop_per_node=32, seed=42)` in `egraph::prelude`, Python, and WASM.
- **Precomputed-Landmark Graph UMAP Constructors (`GuMapSolver.from_landmarks` / `IbUmapSolver.from_landmarks`)**: Added landmark-reusing constructors (`from_landmarks` / `fromLandmarks`, default `max_2hop_per_node=32`) across Rust, Python, and WASM.
- **PaCMAP-Graph 1-Hop Full Retention (`keep_all_1hop` / `keepAll1hop`)**: Added `keep_all_1hop` option on `PacmapGraphConfig` and `SgdSolver.pacmap_graph` / `SgdSolver.pacmapGraph` to preserve all 1-hop edges beyond `n_neighbors`.
- **Sampled Neighborhood Preservation (`sampled_np1` / `sampled_np2`, `sampledNp1` / `sampledNp2`)**: Added deterministic vertex-sampled 1-hop and 2-hop neighborhood preservation metrics to `egraph-metrics`, `evaluate_metrics` / `evaluateMetrics`, `egraph-cli` (`sampled-np1`, `sampled-np2`), Python, and WASM.
- **Spatial Refiner 2-Hop Anchoring & Dynamic `k_cap`**: Added `w_anchor2`, `k_cap_decay`, `k_cap_min`, `current_k_cap` / `currentKCap`, and `set_k_cap` / `setKCap` on `SpatialIntruderRefiner`.

### Changed

- **Automatic $O(N)$ tsNET `learning_rate` Scaling**: Switched `TsNetConfig.learning_rate` and Python `TsNetSolver` constructor defaults to `None` (resolving to $\max(200.0, N / 3.0)$ with a Stage 3 stability cap of $2N$ for $N < 100$).
- **Streaming `SweepLineCrossings` & Halved `SpatialIntruderRefiner` Memory**: Eliminated global `PairSet` materialization in `SweepLineCrossings::{count, count_edges, summarize, summarize_edges}` and eliminated the double-buffered pair vectors in `SpatialIntruderRefiner`.

### Fixed

- **Row-Normalized Perplexity Symmetrization**: Fixed `compute_knn_probabilities(..., row_normalize=True)` (`SymmetrizationMode::RowNormalized`) so row normalization applies to the symmetric support matrix rather than collapsing back to `StandardTsne`.
- **`pacmap_graph` FP 2-Hop Exclusion**: Excluded 2-hop neighbors (`has_common_neighbor`) during `SgdSolver.pacmap_graph` further-pair (`FP`) sampling.
- **WASM `sampledCrossingNumber` Docstring**: Corrected default `numSamples` documentation from `10000` to `4096`.

### Removed

- **Python `InfoMap` Binding**: Removed the legacy `InfoMap` PyO3 placeholder class (`crates/python/src/clustering/infomap.rs`) and its Python exports/stubs, aligning Python with WASM and `v6.0.0-alpha.5` release notes.

## [6.0.0-alpha.5] - 2026-11-23

`v6.0.0-alpha.5` (`6.0.0a5` on PyPI, `6.0.0-alpha.5` with `--tag next` on npm) delivers a clean architectural redesign across Native Rust, Python (PyO3 `abi3-py310`), and WebAssembly (`wasm-bindgen` pure ESM), achieving 100% cross-target feature parity under strictly single-threaded, deterministic pure CPU Rust execution. Prior tag `v6.0.0-alpha.4` remained unpublished after a `check_invariants.sh` lockfile-stderr bug was fixed prior to `v6.0.0-alpha.5`.

### Added

- **Workspace & Core Architecture (`egraph-core`, `egraph-ffi`, `egraph-petgraph`, `egraph-dataset`, `egraph`, `egraph-cli`)**:
  - Zero-copy `u32`-indexed `CsrGraph` and Structure-of-Arrays `Layout2d`, `LayoutNd`, and `ManifoldLayout2d` (Poincare hyperbolic disk, spherical, 2D torus).
  - Cooperative `IterativeSolver` execution (`step()`, `run_steps(k)` / `runSteps(k)`, `run()`) for non-blocking `requestAnimationFrame` browser rendering via `runProgressive`.
  - Explicit deterministic `u64` seeded RNG (`DeterministicRng::from_seed_u64`), `CoreError` structured error handling, and unified feature-gated `egraph` facade + `clap`-based `egraph-cli`.
- **Spatial & Quality Metrics (`egraph-spatial`, `egraph-metrics`)**:
  - `IbFft2d` ($p=1$ default, `G_dual` non-periodic convolution) and fused `FitField2d` (halving particle forward 2D FFT calls), `ArenaQuadtree2d`, `StaticKdTree`, `SweepLineCrossings`, `RectangleOverlapSweep`, and `spade`-backed `Delaunay2d` (`delaunay` feature).
  - Selective `evaluate_metrics` / `evaluateMetrics`: closed-form and pivot-sampled Scale-Normalized Stress (`SNS`), sweep-line crossing numbers (`crossing_edges`, `crossing_edges_torus`), `StaticKdTree`-backed `NP1` / `NP2` / Gabriel / Node Resolution, `prism_distortion`, `drng_preservation`, and `dgg_preservation`.
- **Linear Algebra & Neighborhoods (`egraph-linalg`, `egraph-neighborhood`)**:
  - Zero-allocation 3-vector `ChebyshevWorkspace` with adaptive polynomial degree and PSD-projected regularized DoG (`R1`/`R2`) + JL embeddings, `SubspaceEigen`, built-in `std = 1.0` normalized `PivotMds`, Koren degree-normalized spectral initialization, and core-first `MaxMinSp`.
  - Pure 1+2-hop non-duplicated CSR extraction (`extract_1hop_2hop_csr`), node-wise adaptive perplexity, 2-stage spectral reranked $k$-NN & Hybrid Adaptive $k$-NN, `lambda`-rescaling, `fuzzy_simplicial_set`, and `GuMap` 1+2-hop priority CSR.
- **Track A, Track B & Unified Force Layouts (`egraph-layout`, `egraph-clustering`)**:
  - Track A: Unified `TsNetSolver` (`pure_2hop`, `from_knn`, `from_probabilities`, `from_graph_partial_bfs`) with `ibFFT`, Barnes-Hut, and Exact backends.
  - Track B: Regularized Chebyshev DoG + 3-tier `SparseBarrier` `SgdSolver`, `GuMapSolver`, and 2D `StaticKdTree` `SpatialIntruderRefiner` (`hinge_fp=True`, `w_fn=0.0` defaults).
  - Unified force & multilevel solvers: `TFdpSolver` (`ibFFT`), `IbUmapSolver`, `SgdSolver.snap_t_fdp` (edge-incident $k=3$ negative sampling), `SgdSolver.pacmap_graph`, `SgdSolver.sgd2`, `ForceAtlas2Solver`, `YifanHuSolver` (`EC+MIVS`), `StressMajorizationSolver`, `KamadaKawaiSolver`, `FdebSolver`, $O(N \log N)$ `PrismOverlapRemoval`, `SeparationConstraint`, and DAG layering (`cycle_edges`, `remove_cycles`, `assign_layers`, `generate_layered_constraints`).
  - Pure Rust community detection and coarsening: `kmeans_plus_plus`, `spectral_clustering`, `louvain_communities`, `label_propagation`, `coarsen_graph`, and `project_clustered_layout`.
- **Python & WebAssembly Tooling / Docs / Demos**:
  - Python `egraph.interop` (`from_scipy_csr`, `to_scipy_csr`, `from_networkx`, `to_networkx`, `from_igraph`), complete PEP 484/561 `_core.pyi` + `py.typed` (`mypy --strict` and `stubtest` verified), and Sphinx documentation + `doctest`.
  - WASM pure ESM single-`.wasm` package (`pkg/index.js`, `pkg/egraph_wasm.node.js`, `pkg/index.d.ts`) with `Layout2d.coordsView()` (`memory.grow` detach-safe) and `runProgressive`, TypeDoc API reference, and updated `js/examples` Vite + React demos.
  - Migration guide at `docs/MIGRATION_V6.md`.

### Changed

- **Strictly Single-Threaded Pure Rust**: Removed `rayon`, `linfa`, `serde-wasm-bindgen` (default WASM build), and host-object (`Py<PyAny>` / `JsValue`) storage from core solvers; aligned `ndarray` (`0.17`, `default-features = false`) across the workspace.
- **Python Packaging**: Upgraded to PyO3 `0.28` (`abi3-py310`, `requires-python = ">=3.10"`) with runtime `numpy >= 1.26` dependency.
- **WASM Packaging**: Consolidated duplicate `dist/*` / UMD / Rollup outputs into a single `wasm-opt -Oz` optimized ESM `.wasm` bundle (`<= 600 KiB` gzipped) shared across browsers, bundlers, and Node.js 18+.
- **Breaking API Renames**: `OverwrapRemoval` -> `PrismOverlapRemoval`; `TsNet` / `BhTsNet` / `FitTsNet` / `LTsNet` -> `TsNetSolver`; unscaled `stress` -> Scale-Normalized Stress (`SNS`).

### Removed

- Removed legacy `InfoMap` clustering placeholder and legacy dead workflow/crate scaffolding (`crates/layout/random`).

### Notes

- **Track A Interpolation Default**: `TsNetSolver.pure_2hop` defaults to `interpolation_order=1` (bilinear `ibFFT` for fast interactive/browser execution), while `interpolation_order=3` reproduces DKE Phase 4 cubic interpolation (`suite30` NP2 `0.6301`).
- **4M-Node Streaming Memory Profile**: `SgdSolver.snap_t_fdp` provides streaming `< 1.3 GiB` Peak RSS execution on 4M-node graphs (`com-LiveJournal`).
