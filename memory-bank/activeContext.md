# Active Context: egraph-rs

## Current Work Focus

The current focus is on enhancing the clustering capabilities through community detection algorithms, while continuing to improve the WebAssembly bindings and layout algorithms in the graph visualization system:

1. **Community Detection Algorithms** for identifying clusters/communities in graphs:

   - Implemented a common `CommunityDetection` trait that all algorithms implement
   - Each algorithm returns `HashMap<G::NodeId, usize>` mapping nodes to community IDs
   - Maintained backward compatibility with existing code
   - Implemented four key algorithms:
     - **Louvain Method**: Modularity optimization approach that iteratively moves nodes between communities to maximize modularity
     - **Label Propagation**: Fast and simple algorithm that works by propagating labels through the network
     - **Spectral Clustering**: Uses eigenvectors of the graph Laplacian matrix to identify communities
     - **InfoMap**: Information-theoretic approach that minimizes description length

2. **WebAssembly Binding Tests**

   - Creating dedicated test files for each class and function
   - Implementing tests that can be run individually with `wasm-pack test --node --test <filename>`
   - Testing both basic functionality and integration with other components
   - Following the pattern: `tests/<component>.rs` and `tests/<component>.js`
   - Initial implementation for the `Rng` class (random number generation)

3. **Layout Algorithms** in the graph visualization system, particularly:

4. **Stochastic Gradient Descent (SGD)** implementations

   - Full SGD (all-pairs shortest path distances)
   - Sparse SGD (pivot-based sparse approximation)
   - Distance-adjusted SGD (dynamic distance adjustment)
   - Various learning rate schedulers:
     - Constant: Maintains a fixed learning rate
     - Linear: Linear decay of learning rate
     - Exponential: Exponential decay of learning rate
     - Quadratic: Quadratic decay of learning rate
     - Reciprocal: Reciprocal decay of learning rate
   - Based on research: Zheng, J. X., Pawar, S., & Goodman, D. F. (2018). "Graph drawing by stochastic gradient descent"

5. **Stress Majorization** algorithm

   - Iterative stress minimization
   - Conjugate gradient solver
   - Based on research: Gansner et al. (2004) "Graph drawing by stress majorization"

6. **Multidimensional Scaling (MDS)** implementations

   - Classical MDS: Standard implementation that computes a full distance matrix
   - Pivot MDS: Efficient implementation that uses a subset of nodes as pivots
   - Uses eigendecomposition and double centering to transform distance matrices
   - Based on research: Cox, T. F., & Cox, M. A. (2000). Multidimensional scaling. Chapman & Hall/CRC.

7. **Overlap Removal** techniques for improving readability

8. **Quality Metrics** for evaluating layout effectiveness
   - Stress: How well layout preserves graph-theoretical distances
   - Ideal Edge Lengths: How well edge lengths match their ideal lengths
   - Neighborhood Preservation: How well the layout preserves local neighborhoods
   - Crossing Number: Count of edge crossings in the layout
   - Edge Angle: Angles at which edges cross
   - Aspect Ratio: Balance between width and height of the drawing
   - Angular Resolution: Angles between edges connected to the same node
   - Node Resolution: How well nodes are distributed in the drawing space
   - Gabriel Graph Property: Adherence to the Gabriel graph condition

## Recent Changes

- Implemented comprehensive clustering (community detection) algorithms:

  - Created a new common interface for all algorithms with the `CommunityDetection` trait
  - Implemented four powerful algorithms with consistent API:
    - Louvain Method: Modularity optimization through iterative node movement
    - Label Propagation: Fast algorithm using majority-based label diffusion
    - Spectral Clustering: Using graph Laplacian eigenvectors for community identification
    - InfoMap: Information-theoretic method minimizing random walk description length
  - Updated the trait to take graphs by value instead of by reference
  - Maintained backward compatibility with the existing `louvain_step` function
  - Added comprehensive tests for all algorithms and edge cases
  - Each algorithm includes detailed documentation and examples
  - Properly organized code with separate modules for each algorithm
  - Algorithms follow a consistent pattern with similar constructors and configuration methods
  - Fixed all type annotation issues in the implementations
  - Suggested commit message: `feat(petgraph-clustering): implement comprehensive community detection algorithms`

- Enhanced cluster overlap removal in the separation-constraints module:

  - Updated the `project_clustered_rectangle_no_overlap_constraints` function to:
    - Use the more efficient `generate_rectangle_no_overlap_constraints_triangulated` function instead of the original function
    - Simplify the implementation by directly updating node positions rather than using a temporary map for displacements
    - Improve performance for large cluster graphs
  - Modified the louvain_step function in the clustering module to take graph by value rather than by reference
  - Added a new example `lesmis_cluster_overlap.rs` that demonstrates:
    - Using the Les Miserables dataset with Louvain community detection
    - Applying SGD layout with ClassicalMds initialization
    - Applying both node-level and cluster-level overlap removal
    - Visualizing the result with nodes colored by community
  - Added "lesmis" to the features of egraph-dataset dependency to support the new example
  - The improvements provide better results for visualizing graphs with community structure

- Created a new `petgraph-algorithm-layering` crate for layered graph algorithms:

  - Extracted layer assignment functionality from separation-constraints crate to a dedicated algorithm crate
  - Implemented a trait-based approach for greater extensibility:
    - Created a `LayeringAlgorithm` trait for different layering algorithms
    - Implemented the `LongestPath` algorithm based on the longest path method
  - Separated cycle detection and removal functionality:
    - Moved cycle detection/removal to a dedicated `cycle` module
    - Enhanced with comprehensive tests for various graph configurations
  - Updated the `generate_layered_constraints` function in separation-constraints crate to use the new API
  - Added comprehensive tests for all components:
    - Tests for cycle detection with multiple cycles
    - Tests for cycle removal in complex graphs
    - Tests for longest path layering algorithm
    - Integration tests for layering with the constraint generation
  - All tests pass successfully, verifying the correctness of the implementation
  - The new architecture allows for easier addition of new layering algorithms in the future
  - This refactoring follows the project pattern of moving algorithms to dedicated crates

- Enhanced the triangulation algorithm and implemented triangulation-based rectangle overlap constraints:

  - Updated the Delaunay triangulation function in `petgraph-algorithm-triangulation` to take only a drawing as input (without requiring a graph)
  - Made the triangulation function generic over scalar types with appropriate trait bounds
  - Updated documentation to reflect the new function signature and usage
  - Implemented a new function `generate_rectangle_no_overlap_constraints_triangulated` in the separation-constraints crate that:
    - Takes a DrawingEuclidean2d as input
    - Uses Delaunay triangulation to identify adjacent node pairs
    - Generates separation constraints only for those adjacent pairs
    - Works with generic scalar types through the DrawingValue trait
  - Added comprehensive tests for the new function covering:
    - Square formation (4 nodes)
    - Triangle formation (3 nodes)
    - Collinear points (3 nodes in a line)
    - Constant node sizes
  - Verified that all tests pass, including doc-tests
  - The implementation is more efficient than the original function for large graphs because it only checks node pairs that are adjacent in the triangulated graph, rather than checking all possible pairs (which would be O(n²))
  - The function is already being used in the example application (qh882_separation_constraints.rs)

- Implemented Delaunay triangulation algorithm:

  - Created a new crate `petgraph-algorithm-triangulation` in `crates/algorithm/triangulation`
  - Implemented a function that takes a 2D Euclidean drawing as input
  - Uses the spade library for efficient Delaunay triangulation computation
  - Returns a new graph with nodes corresponding to the drawing's nodes and edges representing the triangulation
  - Added comprehensive tests for various node configurations (square, triangle, collinear points)
  - Ensured proper integration with petgraph and petgraph-drawing crates
  - The implementation handles special cases like collinear points correctly
  - All tests pass successfully, verifying the correctness of the implementation

- Updated Python binding documentation format to follow Sphinx recommendations for the OverwrapRemoval module:

  - Updated documentation in `crates/python/src/layout/overwrap_removal.rs` to ensure consistent Sphinx format
  - Changed internal Rust documentation format (`# Parameters`, `# Returns`) to Sphinx format (`:param:`, `:type:`, `:return:`, `:rtype:`) for all Python-exposed functions and methods
  - Added `:raises:` directive to the constructor method to document potential exceptions
  - Enhanced descriptions for parameters and return values
  - Ensured consistent formatting across all methods
  - This change completes the standardization of Python bindings documentation to Sphinx format across the entire codebase

- Updated Python binding documentation format to follow Sphinx recommendations for additional modules:

  - Updated documentation in the following files to ensure consistent Sphinx format:
    - `crates/python/src/distance_matrix.rs`
    - `crates/python/src/rng.rs`
    - `crates/python/src/quality_metrics.rs`
  - Changed internal Rust documentation format (`# Parameters`, `# Returns`) to Sphinx format (`:param:`, `:type:`, `:return:`, `:rtype:`) for Python-exposed functions and methods
  - Added comprehensive docstrings to all Python-exposed functions in the quality metrics module
  - Added `:raises:` directives where methods might raise exceptions
  - Enhanced descriptions for parameters and return values
  - Ensured consistent formatting across all files
  - This change ensures the Python bindings documentation follows the Sphinx format consistently, making it more accessible to users and ensuring it can be properly processed by Sphinx's autodoc extension
