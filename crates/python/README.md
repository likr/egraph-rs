# egraph - High-Performance Graph Drawing Library

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://egraph-py-docs.netlify.app/)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Version: 6.0.0-alpha.4**

**egraph** is a fast, strictly single-threaded graph drawing and layout library for Python powered by pure CPU Rust (`abi3-py310`). It provides zero-copy Compressed Sparse Row (`CsrGraph`) and Structure-of-Arrays (`Layout2d`, `LayoutNd`, `ManifoldLayout2d`) bindings, deterministic seeded execution, and SOTA neighborhood-preserving and force-directed layouts.

## Key Features

- **Zero-Copy CSR & SoA Buffers**: `CsrGraph.from_csr`, `CsrGraph.from_edges`, `Layout2d` (`float64` default and first-class `dtype="float32"`), `LayoutNd`, and `ManifoldLayout2d` (Poincare hyperbolic disk, spherical, 2D torus).
- **Ecosystem Interop (`egraph.interop`)**: Direct conversion helpers `from_scipy_csr`, `to_scipy_csr`, `from_networkx`, `to_networkx`, and `from_igraph`.
- **Track A (Pure 1+2-Hop & Hybrid Adaptive tsNET)**: `TsNetSolver.pure_2hop`, `TsNetSolver.from_knn`, `TsNetSolver.from_probabilities`, and `TsNetSolver.from_graph_partial_bfs` with `ibFFT` / Barnes-Hut / Exact backends.
- **Track B (Regularized Chebyshev DoG + 3-Tier Sparse Barrier SGD & GuMap)**: `SgdSolver.sparse_barrier_dog`, `GuMapSolver.from_graph_dog`, and 2D `StaticKdTree` `SpatialIntruderRefiner`.
- **Unified Force & Multilevel Models**: `TFdpSolver`, `IbUmapSolver`, `SgdSolver.snap_t_fdp`, `SgdSolver.pacmap_graph`, `SgdSolver.sgd2`, `ForceAtlas2Solver`, `YifanHuSolver`, `StressMajorizationSolver`, `KamadaKawaiSolver`, and `FdebSolver`.
- **Overlap Removal & Constraints**: $O(N \log N)$ Delaunay `PrismOverlapRemoval`, `SeparationConstraint` 1D/2D projection, and DAG layering (`cycle_edges`, `remove_cycles`, `assign_layers`, `generate_layered_constraints`).
- **Community Detection & Coarsening**: `kmeans_plus_plus`, `spectral_clustering`, `louvain_communities`, `label_propagation`, `coarsen_graph`, and `project_clustered_layout`.
- **Selective Quality Metrics**: `evaluate_metrics` supporting closed-form & pivot-sampled Scale-Normalized Stress (`scale_normalized_stress`), sweep-line crossing counts (`sweep_crossing_number`), `StaticKdTree` `NP1` / `NP2`, `prism_distortion`, `drng_preservation`, and `dgg_preservation`.
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
pip install --pre "egraph==6.0.0a4"
# With SciPy/NetworkX/matplotlib support:
pip install --pre "egraph[interop,viz]==6.0.0a4"
```

### Building from Source

```bash
git clone https://github.com/likr/egraph-rs.git
cd egraph-rs/crates/python
pip install "maturin>=1.7,<2.0" "numpy>=1.26"
maturin develop --release
```

## Quick Start

### 1. CSR Construction, Track A (`TsNetSolver.pure_2hop`), and `evaluate_metrics`

```python
import numpy as np
import egraph as eg

# Build an undirected cycle graph (0 - 1 - 2 - 3 - 4 - 5 - 0) via CSR arrays
indptr = np.array([0, 2, 4, 6, 8, 10, 12], dtype=np.uint32)
indices = np.array([1, 5, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0], dtype=np.uint32)
graph = eg.CsrGraph.from_csr(indptr, indices)

# Initialize layout with normalized PivotMDS
layout = eg.pivot_mds_init(graph, num_pivots=6, seed=42)

# Optimize with Track A Pure 1+2-Hop Adaptive tsNET
solver = eg.TsNetSolver.pure_2hop(graph, backend="fft", seed=42)
solver.run(layout)

# Zero-copy NumPy view (shape: (N, 2))
coords = layout.coords(copy=False)
print("Coordinates shape:", coords.shape)

# Evaluate Scale-Normalized Stress (SNS), 2-hop NP2, and crossings
metrics = eg.evaluate_metrics(graph, layout, ["sns", "np2", "crossing_number"])
print(metrics)
```

### 2. SciPy CSR & NetworkX Interoperability (`egraph.interop`)

```python
import networkx as nx
import scipy.sparse as sp
import egraph as eg
from egraph.interop import from_networkx, from_scipy_csr

# From NetworkX
nx_graph = nx.karate_club_graph()
g_nx = from_networkx(nx_graph)

# From SciPy CSR matrix
adj_csr = nx.to_scipy_sparse_array(nx_graph, format="csr", dtype=float)
g_sp = from_scipy_csr(sp.csr_matrix(adj_csr))

layout = eg.pivot_mds_init(g_sp, num_pivots=16, seed=42)
eg.TsNetSolver.pure_2hop(g_sp, seed=42).run(layout)
print(eg.evaluate_metrics(g_sp, layout, ["sns", "np2"]))
```

### 3. Track B (`SgdSolver.sparse_barrier_dog`), `SgdSolver.snap_t_fdp`, and `PrismOverlapRemoval`

```python
import numpy as np
import egraph as eg

edges = [(i, (i + 1) % 12) for i in range(12)] + [(0, 6), (3, 9)]
graph = eg.CsrGraph.from_edges(12, edges)

# Track B: Regularized Chebyshev DoG + 3-tier Sparse Barrier SGD
layout_b = eg.pivot_mds_init(graph, num_pivots=8, seed=42)
barrier_sgd = eg.SgdSolver.sparse_barrier_dog(graph, seed=42)
barrier_sgd.run(layout_b)
print("DoG barrier stats:", barrier_sgd.dog_stats())

# M4: SNAP-tFDP (low-memory edge-incident negative sampling)
layout_snap = eg.pivot_mds_init(graph, num_pivots=8, seed=42)
snap = eg.SgdSolver.snap_t_fdp(graph, iterations=200, seed=42)
snap.run(layout_snap)

# Overlap removal with O(N log N) PRISM
radii = np.full(graph.num_nodes, 0.25, dtype=np.float64)
prism = eg.PrismOverlapRemoval.from_radii(radii, padding=0.05, seed=42)
prism.run(layout_snap)
print("Remaining overlaps:", prism.last_overlap_count)
```

## Migration from v5 / Legacy Alpha

See [docs/MIGRATION_V6.md](../../docs/MIGRATION_V6.md) for a complete guide covering:

1. Host object weights removal -> `CsrGraph` + `egraph.interop` + host-side arrays
2. `OverwrapRemoval` -> `PrismOverlapRemoval`
3. `TsNet` / `BhTsNet` / `FitTsNet` / `LTsNet` -> `TsNetSolver` (`pure_2hop`)
4. `stress` (`NS`) -> `scale_normalized_stress` (`SNS`) / `evaluate_metrics`
5. WASM pure ESM, `fromCsr`, `coordsView()`, and `runProgressive`

## Documentation & Development

Full documentation is available at **https://egraph-py-docs.netlify.app/**.

```bash
# Run Python unit tests, mypy --strict, and stubtest from repo root
./scripts/verify.sh --python

# Build Sphinx HTML and doctests locally
python -m sphinx -b html crates/python/docs/source crates/python/docs/_build/html
python -m sphinx -b doctest crates/python/docs/source crates/python/docs/_build/doctest
```

## License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.
