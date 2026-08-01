---
name: egraph-rs-system-patterns
description: Core architectural and implementation patterns of layouts, clustering, layering, and triangulation in egraph-rs.
---

# System and Architectural Patterns for egraph-rs

This skill details the architectural design and algorithms used across the `egraph-rs` library. Use this skill when modifying, maintaining, or implementing core algorithms (SGD, MDS, community detection, layering, triangulation, etc.) or cross-language bindings.

## Architecture & Design

- **Modular Crate Structure**
  - algorithm (connected-components, shortest-path, triangulation, layering)
  - clustering (community detection)
  - dataset (graph dataset loaders)
  - drawing (Euclidean, Spherical, Hyperbolic, Torus)
  - edge-bundling (force-directed edge bundling)
  - layout (SGD, MDS, Stress-Majorization, Kamada-Kawai, Omega, overlap-removal, random, separation-constraints)
  - linalg (RdMds - Resistance-distance MDS for spectral embeddings)
  - quality-metrics (layout evaluation)
  - language bindings (Python, WebAssembly)

- **Key Patterns**
  - Builder, Strategy, Adapter, Visitor, Factory Methods
  - Trait-based Interfaces (`CommunityDetection`, `LayeringAlgorithm`)
  - Composition over inheritance

## Layout Algorithms

### SGD (Stochastic Gradient Descent)
Force-directed layout with a unified concrete implementation.
- **Architectural Evolution**: Uses a concrete `Sgd<S>` struct rather than a trait-based design for efficiency.
- **Unified Framework**: Supports Full, Sparse, Distance-Adjusted, Omega, and Kernel-SGD variants through different node pair strategies.
- **Core Structure**:
  ```rust
  pub struct Sgd<S> {
      node_pairs: Vec<(usize, usize, S, S, S, S)>, // (i, j, dij, dji, wij, wji)
      epsilon: S,    // Stability parameter
      eta_min: S,    // Minimum learning rate (calculated from weights)
      eta_max: S,    // Maximum learning rate (calculated from weights)
  }
  ```
- **Learning Rate**: Automatic calculations based on weight distribution.
- **Scheduler Integration**: Custom scheduler trait:
  ```rust
  pub trait Scheduler<S> {
      fn run<F: FnMut(S)>(&mut self, callback: &mut F);
      fn step<F: FnMut(S)>(&mut self, callback: &mut F);
      fn is_finished(&self) -> bool;
  }
  ```

### RdMds (Resistance-distance MDS)
Computes spectral embeddings using the graph Laplacian.
- **Location**: `crates/linalg/rdmds/`
- **Core Structure**:
  ```rust
  pub struct RdMds<S> {
      pub d: usize,
      pub shift: S,
      pub eigenvalue_max_iterations: usize,
      pub cg_max_iterations: usize,
      pub eigenvalue_tolerance: S,
      pub cg_tolerance: S,
  }
  ```

### Omega
Generates node pairs for SGD from spectral embeddings.
- **Location**: `crates/layout/omega/`
- **Core Structure**:
  ```rust
  pub struct Omega<S> {
      pub k: usize,        // Random pairs per node
      pub min_dist: S,     // Min distance
  }
  ```

### Kernel-SGD
Diffusion kernel-based SGD using exp(-tL) kernel.
- **Location**: `crates/layout/kernel-sgd/` and `crates/linalg/diffusion-kernel/`
- **Architecture**:
  - `diffusion_kernel.rs`: Single-scale heat kernel `DiffusionKernel` ($e^{-tL}$), `DiffusionDistanceMatrix`, and `PivotDiffusionDistanceMatrix` (exact single-source heat diffusion distances from pivots $D_{ij}^2 = -4t \log(K_{ij} / \sqrt{K_{ii} K_{jj}})$)
  - `pivot_diffusion_sgd.rs`: `PivotDiffusionSgd` layout builder using incremental max-min random pivot sampling on exact single-source heat vectors, delegating layout optimization to `SparseSgd`.
  - `bicgstab.rs`: Batched BiCGSTAB linear solver ($(I - \alpha P) Y = B$) with internal `BicgstabSolverBuffers`
  - `multiscale.rs`: Eigenvalue-free `MultiscaleDiffusionKernel` engine using Batched BiCGSTAB and Hutchinson index, with `MultiscaleDiffusionDistanceMatrix`
  - `low_rank.rs`: Low-rank spectral heat kernel `LowRankDiffusionKernel` ($K_{ij}^{(r)} = \sum_{k=0}^{r} \exp(-t \lambda_k) u_{k,i} u_{k,j}$) and `LowRankDiffusionDistanceMatrix`, computing low-rank spectral embeddings via `petgraph-linalg-rdmds` eigensolver. Supports both Standard ($L = D - A$) and Symmetric Normalized ($L_{\text{sym}} = D^{-1/2} L D^{-1/2}$) Laplacians.
  - **API Surface Principle**: Keep submodules private or clean (`mod ...;`), export minimal symmetric API (`DiffusionKernel`, `LowRankDiffusionKernel`, & `MultiscaleDiffusionKernel`, `DiffusionDistanceMatrix` & `LowRankDiffusionDistanceMatrix`), and hide internal solver workspace buffers (`pub(crate)`).

## Community Detection

- **Unified Trait-Based Interface**:
  ```rust
  trait CommunityDetection<G> {
      fn detect_communities(&self, graph: G) -> HashMap<G::NodeId, usize>;
  }
  ```
- **Algorithms**: Louvain, Label Propagation, Spectral Clustering, InfoMap.
- **Graph Coarsening**: Custom merge functions for nodes and edges.

## Layering Algorithms

- **Unified Trait-Based Interface**:
  ```rust
  trait LayeringAlgorithm<N, E, Ix: IndexType> {
      fn assign_layers(&self, graph: &Graph<N, E, Directed, Ix>) -> HashMap<NodeIndex<Ix>, usize>;
  }
  ```
- **Cycle Handling**: Detection and removal functions to ensure directed graphs are acyclic before layering.

## Triangulation

- **Delaunay Triangulation**:
  - Uses the `spade` library to calculate 2D Euclidean triangulation.
