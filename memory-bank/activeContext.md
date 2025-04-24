# Active Context: egraph-rs

## Current Work Focus

Current work centers on these main areas:

1. **Community Detection**

   - Implemented unified `CommunityDetection` trait with consistent interface
   - Four algorithms: Louvain, Label Propagation, Spectral Clustering, InfoMap
   - Each algorithm configurable (iterations, seed, etc.)
   - **Added Python bindings** for all community detection algorithms
   - Implemented graph coarsening functionality in Python

2. **Graph Layering**

   - **Added Python bindings** for the `petgraph-algorithm-layering` crate
   - Implemented `LongestPath` algorithm with consistent Python interface
   - Added cycle detection and removal functionality for directed graphs
   - Created comprehensive tests for layering algorithms

3. **Cluster Visualization**

   - Enhanced overlap removal with direct constraint generation
   - Added Les Miserables dataset examples with community visualization
   - Improved performance through better caching and direct position updates

4. **Testing & Layout Improvements**
   - Comprehensive WebAssembly tests with individual test files
   - Optimized layout algorithms (SGD, MDS, Stress Majorization)
   - Fixed issues with high-dimensional embeddings and infinite loops
   - Added tests for Python bindings of clustering algorithms

## Recent Changes

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
