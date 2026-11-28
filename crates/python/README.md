# egraph - High-Performance Graph Drawing Library

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://egraph-py-docs.netlify.app/)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Version: 6.0.0-beta.1**

**egraph** is a fast, strictly single-threaded graph drawing and layout library for Python powered by pure CPU Rust (`abi3-py310`). It provides zero-copy Compressed Sparse Row (`CsrGraph`) and Structure-of-Arrays (`Layout2d`, `Layout3d`, `LayoutNd`, `ManifoldLayout2d`, `EdgePolylines`) bindings, deterministic seeded execution (`seed=42` default), structured `EgraphError` exceptions, and SOTA neighborhood-preserving, force-directed, 3D/$N$-dimensional, and non-Euclidean layouts.

## Key Features

- **Zero-Copy CSR & SoA Buffers**: `CsrGraph.from_csr`, `CsrGraph.from_edges` (with `weight_dtype="float64" | "float32"`, `__len__`, `edges()`, `has_edge()`, `edge_weight()`, `is_connected()`), `Layout2d`, `Layout3d` (zero-copy `(N, 3)` NumPy views), `LayoutNd` (`.dims`), `ManifoldLayout2d`, and UAF-safe `EdgePolylines`.
- **3D & $N$-Dimensional Engine**: `Layout3d` (`target="layout3d"` with `ArenaOctree3d` Barnes-Hut), `LayoutNd` (`target="layout_nd"`, `DynamicKdTree`), and explicit `target`-selected solvers (`SgdSolver`, `Sgd2Solver`, `PacmapGraphSolver`, `TsNetSolver`, `GuMapSolver`, `StressMajorizationSolver`, `KamadaKawaiSolver`, `MaxentStressSolver`, `FruchtermanReingoldSolver`, `LinLogSolver`, `ForceAtlas2Solver`, `YifanHuSolver`, `TFdpSolver`, `LargeVisSolver`, `DrGraphSolver`, `GraphCneSolver`, `SnapTFdpSolver`, plus `FdebSolver` on `EdgePolylines`) with $N$-D evaluation metrics.
- **Non-Euclidean Manifold Engine**: `ManifoldLayout2d` across `"hyperbolic"` (Poincaré disk), `"hyperbolic_lorentz"` (Lorentz hyperboloid + `to_poincare_layout()`), `"spherical"` ($S^2$), and `"torus"` ($T^2$ with `torus_spectral_phase_init`, `torus_auto_pan`, `torus_edge_segments`, `torus_layout_edge_segments`), physical `ManifoldMetric` (`scale`, `aspect`, `set_scale`, `set_torus_extents`, `rescale_geodesic`), `target="manifold2d"` solvers (`SgdSolver`, `Sgd2Solver`, `PacmapGraphSolver`, `TsNetSolver`, `GuMapSolver`, `StressMajorizationSolver`, `KamadaKawaiSolver`, `FruchtermanReingoldSolver`), `geodesic_distortion`, and hyperbolic/spherical/torus geodesic crossings.
- **Ecosystem Interop (`egraph.interop`)**: Direct conversion helpers `from_scipy_csr`, `to_scipy_csr`, `from_networkx`, `to_networkx(layout=...)`, `from_igraph`, and `to_igraph(layout=...)` supporting `Layout2d`, `Layout3d`, `LayoutNd`, and `ManifoldLayout2d`.
- **Track A (Pure 1+2-Hop & Row-Normalized Adaptive tsNET)**: `TsNetSolver.pure_2hop`, `TsNetSolver.from_graph_row_normalized` (`TwoStepRandomWalk` $A_1 + w_2 A_2$), `TsNetSolver.from_knn`, `TsNetSolver.from_probabilities`, `TsNetSolver.from_probabilities_with_stages` (`TsNetStage`), and `TsNetSolver.from_graph_partial_bfs` with `backend="auto"` (`N <= 256` Exact, `> 256` `LinearFft` in 2D; `Exact`/`BarnesHut` in 3D) plus `"fit_fft"`, `"linear_fft"`, `"barnes_hut"`, and `"exact"` backends.
- **Track B (Regularized Chebyshev DoG + 3-Tier Sparse Barrier SGD & GuMap)**: `SgdSolver.sparse_barrier_dog`, `GuMapSolver.from_graph_dog`, `SgdSolver.from_packed_symmetric`, `SgdSolver.reset_schedule` / `run_schedule_window`, and 2D `StaticKdTree` `SpatialIntruderRefiner`.
- **Separated Force, Embedding, Multilevel & Edge-Bundling Models**: `TFdpSolver`, `IbUmapSolver`, separated SGD classes (`Sgd2Solver`, `PacmapGraphSolver`, `LargeVisSolver`, `DrGraphSolver`, `GraphCneSolver`, `SnapTFdpSolver`), `SgdSolver.ortmann_sparse_stress`, `MaxentStressSolver`, `GripSolver`, `GemSolver`, `SelLayoutSolver`, `BackboneSolver`, `GroupInABoxSolver`, `ClusterRefineSolver`, `ForceAtlas2Solver`, `YifanHuSolver`, `StressMajorizationSolver`, `KamadaKawaiSolver`, `LayoutGenerator` structs (`ClassicalMds`, `PivotMds`, `Hde`, `RdMds`, `KorenSpectral`, `CoarseBackbone`, `SfcLayout` `.generate()`), and `EdgePolylines` + `FdebSolver(polylines)` (`step(polylines)` / `run(polylines)`).
- **Overlap Removal, Decluttering & Constraints**: $O(N \log N)$ Delaunay `PrismOverlapRemoval`, `GTreeOverlapRemoval`, `ForbidOverlapRemoval` (default `"sparse_forbid"`), `LayoutTransform` passes (`PfsPrimeOverlapRemoval`, `TwinDeclutterer` `.apply(layout)`), `declutter_twins_2d`, `ScreenedPoissonField2d` (`grid_size=512`, `density_grid()`, 4-scalar `evaluate_in_bounds`), `ScreenedPoissonRefiner`, `transitive_reduction`, `SeparationConstraint` 1D/2D projection, and DAG layering (`cycle_edges`, `remove_cycles`, `assign_layers`, `generate_layered_constraints`).
- **Structured `EgraphError` Hierarchy & Keyword-Only Parameters**: `EgraphError(ValueError)` with 7 typed subclasses (`InvalidInputError`, `DimensionMismatchError`, `IndexOutOfBoundsError(EgraphError, IndexError)`, `InvalidCsrError`, `NonFiniteValueError`, `UnsupportedOperationError`, `NumericalBreakdownError`), keyword-only `*` hyperparameters, `.pyi` `Protocol`/`TypedDict`/`Literal` stubs (`LayoutStorage`, `IterativeSolver`, `LayoutGenerator`, `LayoutTransform`), and frozen Result classes (`KnnBuildResult`, `RhghLandmarksResult`, `AmgSolveResult`, `AmgBatchSolveResult`, `SpectralFilterProjection`, etc.).
- **Community Detection & Coarsening (5 Zero-Copy Result Classes)**: `louvain_communities` (`LouvainResult`), `leiden_communities` (`LeidenResult`), `kmeans_plus_plus` (`KMeansResult`), `label_propagation` (`LabelPropagationResult`), and `spectral_clustering` (`SpectralClusteringResult`), plus `coarsen_graph`, `louvain_coarsen`, `coarse_backbone_init`, `refine_cluster_layout`, and `squarified_treemap`.
- **Selective Quality Metrics (`MetricReport`)**: `evaluate_metrics` returning a `MetricReport` across 2D, $N$-D, and manifold spaces—covering Scale-Normalized Stress (`scale_normalized_stress`), `geodesic_distortion`, sweep-line crossings, spherical/hyperbolic/torus crossings, `NP1`/`NP2`, `prism_distortion`, `layout_degeneracy` (including `excess_duplicate_ratio` and `near_collision_ratio`), Eades et al. (2017) `S_RNG`/`S_GG` (`drng_preservation`, `dgg_preservation`), `cluster_area_faithfulness` (`G-CAF`), `landmark_stratified_rta`, `adjusted_patch_nig`, `nor_metrics`, `sprawlter_nn`, and cluster separation baselines.
- **Linear Algebra & Datasets**: `DijkstraWorkspace` (`dijkstra_single_source`, `all_pairs_dijkstra`), `classical_mds`, `pivot_mds`, `hde`, `rd_mds`, `laplacian` / `combinatorial_laplacian` / `symmetric_normalized_laplacian` / `random_walk_laplacian` / `shifted_normalized_laplacian`, `ChebyshevDogKernel`, `AmgPcgSolver`, `compute_effective_resistances`, `resistance_curvature_flow`, `list_datasets` / `dataset_catalog` / `dataset_info`, and 7 synthetic generators (`generate_weighted_graph`, `generate_layered_dag`, `generate_sbm_graph`, `generate_torus_grid`, `generate_spherical_icosphere`, `generate_spherical_grid`, `generate_hyperbolic_tree`).
- **Strictly Single-Threaded & Deterministic**: Pure CPU Rust with explicit `seed` parameters (`seed=42` default), zero `rayon`, zero `std::thread`, and zero C/Fortran FFI dependencies.

## Installation

### Requirements

- **Python**: `>= 3.10` (`abi3-py310`)
- **Required**: `numpy >= 1.26`
- **Optional extras**:
  - `interop`: `scipy >= 1.11`, `networkx >= 3.0`, `igraph >= 0.11`
  - `viz`: `matplotlib >= 3.7`, `networkx >= 3.0`
  - `docs`: `sphinx >= 8.0`, `sphinx-rtd-theme >= 2.0`

### Installing from PyPI (Pre-release)

```bash
pip install --pre "egraph==6.0.0b1"
# With SciPy/NetworkX/igraph/matplotlib support:
pip install --pre "egraph[interop,viz]==6.0.0b1"
```

### Building from Source

```bash
git clone https://github.com/likr/egraph-rs.git
cd egraph-rs/crates/python
pip install "maturin>=1.7,<2.0" "numpy>=1.26"
maturin develop --release
```

## Quick Start

### 1. CSR Construction, Track A (`TsNetSolver.pure_2hop`), and `MetricReport`

```python
import numpy as np
import egraph as eg

# Build an undirected cycle graph (0 - 1 - 2 - 3 - 4 - 5 - 0) via CSR arrays
indptr = np.array([0, 2, 4, 6, 8, 10, 12], dtype=np.uint32)
indices = np.array([1, 5, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0], dtype=np.uint32)
graph = eg.CsrGraph.from_csr(indptr, indices)

# Initialize layout with normalized PivotMDS
layout = eg.pivot_mds_init(graph, num_pivots=6, seed=42)

# Optimize with Track A Pure 1+2-Hop Adaptive tsNET (auto selects Exact/LinearFft)
solver = eg.TsNetSolver.pure_2hop(
    graph,
    backend="auto",
    stage_iterations=(30, 30, 40),
    seed=42,
)
solver.run(layout)

# Zero-copy NumPy view (shape: (N, 2))
coords = layout.coords(copy=False)
print("Coordinates shape:", coords.shape)

# Evaluate Scale-Normalized Stress (SNS), 2-hop NP2, and crossings -> MetricReport
report = eg.evaluate_metrics(graph, layout, ["sns", "np2", "crossing_number"])
print(report.scale_normalized_stress, report.neighborhood_preservation_2hop, report.crossing_number)
```

### 2. SciPy CSR, NetworkX & igraph Interoperability (`egraph.interop`)

```python
import networkx as nx
import scipy.sparse as sp
import egraph as eg
from egraph.interop import from_networkx, from_scipy_csr, to_igraph, to_networkx

# From NetworkX
nx_graph = nx.karate_club_graph()
g_nx, _nodes = from_networkx(nx_graph)

# From SciPy CSR matrix
adj_csr = nx.to_scipy_sparse_array(nx_graph, format="csr", dtype=float)
g_sp = from_scipy_csr(sp.csr_matrix(adj_csr))

layout = eg.pivot_mds_init(g_sp, num_pivots=16, seed=42)
eg.TsNetSolver.pure_2hop(g_sp, backend="auto", seed=42).run(layout)
print(eg.evaluate_metrics(g_sp, layout, ["sns", "np2"]).to_dict())

# Export back to NetworkX or python-igraph with coordinates attached
nx_out = to_networkx(g_sp, layout=layout)
ig_out = to_igraph(g_sp, layout=layout)
```

### 3. Track B, `Layout3d`, `EdgePolylines`, Non-Euclidean Manifolds & `EgraphError`

```python
import numpy as np
import egraph as eg

edges = [(i, (i + 1) % 12) for i in range(12)] + [(0, 6), (3, 9)]
graph = eg.CsrGraph.from_edges(12, edges)

# Track B: Regularized Chebyshev DoG + 3-tier Sparse Barrier SGD + TwinDeclutterer
layout_b = eg.PivotMds(graph, num_pivots=8, seed=42).generate()
eg.SgdSolver.sparse_barrier_dog(graph, iterations=60, seed=42).run(layout_b)
eg.TwinDeclutterer(iterations=4).apply(layout_b)

# 3D Layout3d optimization (explicit target="layout3d" + Barnes-Hut)
layout_3d = eg.Layout3d.init(graph.num_nodes, seed=42)
eg.ForceAtlas2Solver(graph, target="layout3d", iterations=100, seed=42).run(layout_3d)

# Edge bundling via EdgePolylines + FdebSolver
polylines = eg.EdgePolylines.straight_line_from_graph(graph, layout_b)
eg.FdebSolver(polylines, cycles=3, steps_per_cycle=20).run(polylines)
print("Bundled control points:", polylines.points(copy=False).shape)

# Non-Euclidean Lorentz hyperboloid -> Poincaré disk projection
lorentz = eg.ManifoldLayout2d.init("hyperbolic_lorentz", graph.num_nodes, seed=42)
eg.SgdSolver.full(graph, target="manifold2d", iterations=60, seed=42).run(lorentz)
poincare = lorentz.to_poincare_layout()
print("Geodesic distortion:", eg.geodesic_distortion(graph, poincare))

# Fail-fast typed error handling
try:
    eg.TsNetSolver.pure_2hop(graph, target="layout3d", backend="fit_fft")
except eg.UnsupportedOperationError as exc:
    print("Rejected incompatible (target, backend):", exc)
```

## Migration from v5 / Legacy Alpha

See [docs/MIGRATION_V6.md](https://github.com/likr/egraph-rs/blob/master/docs/MIGRATION_V6.md) for a complete guide covering v5, `alpha.8 -> alpha.9`, and the v6 API freeze updates:

1. Host object weights removal -> `CsrGraph` + `egraph.interop` + host-side arrays
2. Legacy overlap removal -> `PrismOverlapRemoval` / `ForbidOverlapRemoval` and `LayoutTransform.apply()` (`PfsPrimeOverlapRemoval`, `TwinDeclutterer`)
3. `TsNet` / `BhTsNet` / `FitTsNet` / `LTsNet` -> `TsNetSolver` (`pure_2hop`, `from_graph_row_normalized`, `"auto"`, `"fit_fft"`, `"linear_fft"`, `"barnes_hut"`, `"exact"`)
4. `LayoutGenerator` initializers (`PivotMds`, `ClassicalMds`, `Hde`, `RdMds`, `KorenSpectral`, `CoarseBackbone`, `SfcLayout`) and separated SGD classes (`Sgd2Solver`, `PacmapGraphSolver`, `LargeVisSolver`, `DrGraphSolver`, `GraphCneSolver`, `SnapTFdpSolver`)
5. Explicit `target=` (`"layout2d"`, `"layout3d"`, `"layout_nd"`, `"manifold2d"`), `Layout3d`, `EdgePolylines` + `FdebSolver(polylines)`, `EgraphError` hierarchy, `*` keyword-only hyperparameters, and frozen Result classes (`MetricReport`, `KnnBuildResult`, etc.)

## Documentation & Development

Full documentation is available at **https://egraph-py-docs.netlify.app/**, and 9 interactive Google Colab tutorials live under [notebooks/README.md](https://github.com/likr/egraph-rs/blob/master/notebooks/README.md).

```bash
# Run Python unit tests, mypy --strict, and stubtest from repo root
./scripts/verify.sh --python

# Build Sphinx HTML and doctests locally
python -m sphinx -b html crates/python/docs/source crates/python/docs/_build/html
python -m sphinx -b doctest crates/python/docs/source crates/python/docs/_build/doctest
```

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/likr/egraph-rs/blob/master/LICENSE) file for details.
