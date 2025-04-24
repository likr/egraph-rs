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

- **Python Bindings**:

  ```python
  # Common interface across algorithms
  algorithm = eg.Louvain()  # or eg.LabelPropagation(), eg.SpectralClustering(k), eg.InfoMap()
  communities = algorithm.detect_communities(graph)
  # communities is a dict mapping node indices to community IDs

  # Graph coarsening
  coarsened_graph, node_map = eg.py_coarsen(
      graph,
      lambda node: communities[node],  # Node grouping function
      lambda nodes: len(nodes),        # Node merge function
      lambda edges: len(edges)         # Edge merge function
  )
  ```

## Layering Algorithms

- **Unified Trait-Based Interface**:

  ```rust
  trait LayeringAlgorithm<N, E, Ix: IndexType> {
      fn assign_layers(&self, graph: &Graph<N, E, Directed, Ix>) -> HashMap<NodeIndex<Ix>, usize>;
  }
  ```

- **Implemented Algorithms**:

  - **LongestPath**: Assigns layers based on longest path from source nodes

- **Cycle Handling**:

  - Detection of cycles in directed graphs
  - Removal of minimum edge set to make graph acyclic

- **Python Bindings**:

  ```python
  # Layer assignment with LongestPath algorithm
  longest_path = eg.LongestPath()
  layers = longest_path.assign_layers(graph)
  # layers is a dict mapping node indices to layer numbers (0, 1, 2, ...)

  # Cycle detection and removal
  cycle_edges = eg.cycle_edges(graph)  # Returns list of (source, target) tuples
  eg.remove_cycle(graph)  # Modifies graph in-place to make it acyclic
  ```

## Quality Metrics

- Graph-theoretical distance preservation
- Edge crossing minimization
- Angular resolution optimization
- Node distribution evaluation
