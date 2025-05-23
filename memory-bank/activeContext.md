# Active Context: egraph-rs

## Current Work Focus

Current work centers on these main areas:

1. **Separation Constraints**

   - **Added Python bindings** for the `petgraph-layout-separation-constraints` crate
   - Implemented `Constraint` class with properties for left, right, and gap
   - Added functions for separation constraint generation and application
   - Exposed rectangle overlap prevention methods and cluster overlap handling
   - Created comprehensive tests for constraint-based functionality
   - Improved rectangle overlap implementation by splitting into X and Y dimension algorithms
   - Replaced external RBTree implementation with Rust's built-in BTreeSet for better efficiency
   - Fixed infinite loop issues in rectangle overlap constraints with improved sweep line algorithm
   - Added extensive unit tests to validate algorithm correctness and performance

2. **Community Detection**

   - Implemented unified `CommunityDetection` trait with consistent interface
   - Four algorithms: Louvain, Label Propagation, Spectral Clustering, InfoMap
   - Each algorithm configurable (iterations, seed, etc.)
   - **Added Python bindings** for all community detection algorithms
   - Implemented graph coarsening functionality in Python

3. **Graph Layering**

   - **Added Python bindings** for the `petgraph-algorithm-layering` crate
   - Implemented `LongestPath` algorithm with consistent Python interface
   - Added cycle detection and removal functionality for directed graphs
   - Created comprehensive tests for layering algorithms

4. **Triangulation**

   - **Added Python bindings** for the `petgraph-algorithm-triangulation` crate
   - Exposed the Delaunay triangulation functionality to Python
   - Implemented function that creates a new graph with edges representing the triangulation
   - Created comprehensive tests verifying triangulation behavior for different configurations (square, triangle, collinear points)

5. **Cluster Visualization**

   - Enhanced overlap removal with direct constraint generation
   - Added Les Miserables dataset examples with community visualization
   - Improved performance through better caching and direct position updates

6. **Testing & Layout Improvements**
   - Comprehensive WebAssembly tests with individual test files
   - Optimized layout algorithms (SGD, MDS, Stress Majorization)
   - Fixed issues with high-dimensional embeddings and infinite loops
   - Added tests for Python bindings of clustering algorithms

## Recent Changes

- **Rectangle Overlap Algorithm Refactoring**

  - Completely refactored implementation with separate `rectangle_overlap_2d.rs` module
  - Split algorithm into distinct X and Y dimension-specific implementations following WebCola's approach
  - Replaced external RBTree with Rust's BTreeSet for better efficiency and maintainability
  - Improved event handling in sweep line algorithm to fix infinite loop issues
  - Enhanced neighbor finding logic to match WebCola's implementation more precisely
  - Added thorough unit tests for each component of the algorithm
  - Maintained backward compatibility through legacy function in original module

- **Triangulation Python Bindings**

  - Added Python bindings for the triangulation algorithm
  - Implemented as a single function that takes a DrawingEuclidean2d instance and returns a new graph
  - Consistent naming and documentation style with other Python modules
  - Added comprehensive tests for different geometrical configurations

- **Rectangle Overlap Constraints**

  - Direct method replacing triangulation-based approach
  - Added convenience function `project_rectangle_no_overlap_constraints_2d`
  - Enhanced Les Miserables example with community visualization

- **Community Detection**

  - Unified `CommunityDetection` trait for all algorithms
  - Value-based graph parameter with backward compatibility
  - Complete test coverage and documentation

- **Module Improvements**
  - New `petgraph-algorithm-layering` crate with extensible traits
  - Enhanced triangulation algorithm with scalar type generics
  - Improved documentation with Sphinx format for Python bindings
