# System Patterns: egraph-rs

## Architecture & Design

- **Modular Crate Structure**

  - algorithm (connected-components, shortest-path, triangulation, layering)
  - clustering (community detection)
  - drawing (Euclidean, Spherical, Hyperbolic, Torus)
  - layout (SGD, MDS, Stress-Majorization, Kamada-Kawai)
  - quality-metrics, edge-bundling, separation-constraints
  - language bindings (Python, WebAssembly)

- **Key Patterns**
  - Builder, Strategy, Adapter, Visitor, Factory Methods
  - Trait-based Interfaces (`CommunityDetection`, `LayeringAlgorithm`)
  - Composition over inheritance

## Layout Algorithms

- **SGD**: Force-directed layout with multiple variants

  - Full, Sparse, Distance-Adjusted implementations
  - Various learning rate schedulers (Constant, Linear, etc.)

- **MDS**: Lower-dimensional space visualization

  - Classical (full distance matrix)
  - Pivot-based (efficient for large graphs)

- **Stress Majorization**: Iterative stress minimization

## Community Detection

- **Unified Trait-Based Interface**:

  ```rust
  trait CommunityDetection<G> {
      fn detect_communities(&self, graph: G) -> HashMap<G::NodeId, usize>;
  }
  ```

- **Implemented Algorithms**:

  - **Louvain**: Modularity optimization (fast for large networks)
  - **Label Propagation**: Fast majority-based diffusion
  - **Spectral**: Uses graph Laplacian eigenvectors
  - **InfoMap**: Information-theoretic random walk approach

- **Graph Coarsening**: Creates simplified graph representations

## Quality Metrics

- Graph-theoretical distance preservation
- Edge crossing minimization
- Angular resolution optimization
- Node distribution evaluation
