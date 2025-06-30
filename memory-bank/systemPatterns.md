# System Patterns: egraph-rs

## Architecture & Design

- **Modular Crate Structure**

  - algorithm (connected-components, shortest-path, triangulation, layering)
  - clustering (community detection)
  - dataset (graph dataset loaders)
  - drawing (Euclidean, Spherical, Hyperbolic, Torus)
  - edge-bundling (force-directed edge bundling)
  - layout (SGD, MDS, Stress-Majorization, Kamada-Kawai, overlap-removal, random, separation-constraints)
  - quality-metrics (layout evaluation)
  - language bindings (Python, WebAssembly)

- **Key Patterns**
  - Builder, Strategy, Adapter, Visitor, Factory Methods
  - Trait-based Interfaces (`CommunityDetection`, `LayeringAlgorithm`)
  - Composition over inheritance

## Layout Algorithms

- **SGD**: Force-directed layout with unified concrete implementation

  - **Architectural Evolution**: Moved from trait-based approach to concrete `Sgd<S>` struct for better performance and simplicity
  - **Unified Framework**: Single implementation supports Full, Sparse, Distance-Adjusted, Omega variants through different node pair generation strategies
  - **Core Structure**:
    ```rust
    pub struct Sgd<S> {
        node_pairs: Vec<(usize, usize, S, S, S, S)>, // (i, j, dij, dji, wij, wji)
        epsilon: S,    // Numerical stability parameter
        eta_min: S,    // Minimum learning rate (calculated from weights)
        eta_max: S,    // Maximum learning rate (calculated from weights)
    }
    ```
  - **Learning Rate Management**: Automatic eta_min/eta_max calculation from weight distribution eliminates manual tuning
  - **Scheduler Integration**: Comprehensive trait-based scheduler system:
    ```rust
    pub trait Scheduler<S> {
        fn run<F: FnMut(S)>(&mut self, callback: &mut F);
        fn step<F: FnMut(S)>(&mut self, callback: &mut F);
        fn is_finished(&self) -> bool;
    }
    ```
  - **Five Scheduler Implementations**: Constant, Linear, Quadratic, Exponential, Reciprocal with customizable parameters
  - **Dynamic Updates**: Runtime distance and weight updates with automatic recalculation:
    ```rust
    sgd.update_distance(|i, j, dist, weight| new_distance);
    sgd.update_weight(|i, j, dist, weight| new_weight);
    ```
  - **Numerical Stability**: Proper epsilon handling and normalized learning rate calculation from [0,1] to [eta_min, eta_max]
  - **Omega**: Spectral coordinates-based SGD using graph Laplacian eigenvalues

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

## Triangulation

- **Delaunay Triangulation**:

  - Generates a graph based on node positions in a 2D Euclidean drawing
  - Uses the spade library for efficient triangulation
  - Handles general, triangular, and collinear point configurations

- **Python Bindings**:

  ```python
  # Triangulation of a 2D Euclidean drawing
  # Creates a new graph with edges representing the triangulation
  triangulated_graph = eg.triangulation(drawing)
  # triangulated_graph is a new Graph with edges representing the Delaunay triangulation
  ```

## Quality Metrics

- Graph-theoretical distance preservation
- Edge crossing minimization
- Angular resolution optimization
- Node distribution evaluation
