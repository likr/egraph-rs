# Progress: egraph-rs

## What Works

### Graph Data Structures

- Base graph implementations (undirected and directed graphs)
- Node and edge management
- Generic data storage
- Common type definitions (`Node`, `Edge`, `IndexType`)

### Layout Algorithms

- Stochastic Gradient Descent (SGD)
  - Full implementation (all-pairs shortest path distances)
  - Sparse implementation (pivot-based sparse approximation)
  - Distance-adjusted implementation (dynamic distance adjustment)
  - Multiple scheduler strategies:
    - Constant, Linear, Exponential, Quadratic, Reciprocal
- Multidimensional Scaling (MDS)
  - Classical MDS (full distance matrix)
  - Pivot MDS (subset of nodes as pivots)
- Stress Majorization
  - Iterative stress minimization
  - Conjugate gradient solver
- Kamada-Kawai (spring model based layout)
- Overlap Removal (resolving node overlaps)
- Separation Constraints (layout constraint implementation)

### Drawing Implementations

- Euclidean 2D drawing (2D Cartesian coordinate drawings)
- Spherical 2D drawing (drawings on a spherical surface)
- Hyperbolic 2D drawing (drawings in hyperbolic space)
- Torus 2D drawing (drawings on a torus surface)
- N-dimensional Euclidean drawing (higher-dimensional Euclidean spaces)

### Quality Metrics

- Stress metric (how well layout preserves graph-theoretical distances)
- Ideal edge length metric (how well edge lengths match their ideal lengths)
- Neighborhood preservation (how well the layout preserves local neighborhoods)
- Crossing number (count of edge crossings in the layout)
- Edge angle metric (angles at which edges cross)
- Aspect ratio (balance between width and height of the drawing)
- Angular resolution (angles between edges connected to the same node)
- Node resolution (how well nodes are distributed in the drawing space)
- Gabriel graph property (adherence to the Gabriel graph condition)

### Additional Features

- Edge bundling (force-directed edge bundling for reducing visual clutter)
- Clustering (graph clustering and coarsening for simplifying complex graphs)
- Random number generation (with seed control for reproducible layouts)

### Language Bindings

- Python bindings (via PyO3)
  - Consistent Python-style API (following PEP 8)
  - Type hints in documentation
- WebAssembly bindings (via wasm-bindgen)
  - JavaScript-friendly interfaces (camelCase naming)
  - JSDoc-style comments
  - Transparent data handling
  - Memory safety
  - Method chaining
  - Error handling
  - Callback support

## What's Left to Build

### Layout Algorithms

- Further optimization of layout algorithms for large graphs (>10,000 nodes)
- Performance improvements for dense graphs in WebAssembly context
- Additional layout algorithms (e.g., additional force-directed variants)
- Fine-tuning of SGD schedulers for better convergence
- Optimization of stress majorization for large graphs
- Performance improvements for overlap removal
- ✅ Fix for ClassicalMds implementation to handle cases where a graph is embedded in a space with dimensions higher than what's needed
- ✅ Fix for PivotMds implementation to handle similar cases with high-dimensional embeddings

### Documentation

- More comprehensive examples showcasing different layout algorithms
- Detailed API documentation across all interfaces
- Tutorials for common use cases
- Best practices for selecting appropriate layout algorithms
- Usage examples for different geometric spaces

### Testing

- WebAssembly binding tests:
  - Implemented dedicated test files for the `Rng` class (`tests/rng.rs` and `tests/rng.js`)
  - Implemented dedicated test files for the `Graph` class (`tests/graph.rs` and `tests/graph.js`)
  - Implemented dedicated test files for the `DiGraph` class (`tests/digraph.rs` and `tests/digraph.js`)
  - Implemented dedicated test files for the `DrawingEuclidean2d` class (`tests/drawing_euclidean_2d.rs` and `tests/drawing_euclidean_2d.js`)
  - Implemented dedicated test files for the `DrawingSpherical2d` class (`tests/drawing_spherical_2d.rs` and `tests/drawing_spherical_2d.js`)
  - Implemented dedicated test files for the `DrawingHyperbolic2d` class (`tests/drawing_hyperbolic_2d.rs` and `tests/drawing_hyperbolic_2d.js`)
  - Implemented dedicated test files for the `DrawingTorus2d` class (`tests/drawing_torus_2d.rs` and `tests/drawing_torus_2d.js`)
  - Implemented dedicated test files for the `FullSgd` class (`tests/sgd_full.rs` and `tests/sgd_full.js`)
  - Implemented dedicated test files for the `ClassicalMds` class (`tests/classical_mds.rs` and `tests/classical_mds.js`)
  - Implemented dedicated test files for the `SparseSgd` class (`tests/sgd_sparse.rs` and `tests/sgd_sparse.js`)
  - Fixed an issue in the `DrawingSpherical2d` tests where nodes added to the graph after creating the drawing were not included in the drawing
  - Created a pattern for class/function-specific tests that can be run individually
  - Tests for basic functionality, node/edge operations, traversal, and integration with other components
  - Tests for directed graph functionality, including in/out neighbors and directed edge operations
  - Tests for drawing functionality, including node coordinate operations, drawing manipulation, edge segment representation, and integration with Graph class
  - Tests for spherical drawing functionality, including longitude/latitude coordinate operations and integration with Graph class
  - Tests for hyperbolic drawing functionality, including coordinate operations, Poincaré disc model constraints, and integration with Graph class
  - Tests for torus drawing functionality, including coordinate operations, torus wrapping behavior, edge segment representation, and integration with Graph class
  - Tests for FullSgd functionality, including instantiation, scheduler creation, applying SGD to different drawing types, updating distance and weight functions, shuffling node pairs, and integration with other components
  - Tests for SparseSgd functionality, including instantiation, pivot node configuration, scheduler creation, applying SGD to different drawing types, updating distance and weight functions, shuffling node pairs, and integration with other components
  - Tests for ClassicalMds functionality, including instantiation, 2D layout generation (run2d method), n-dimensional layout generation (run method), different graph structures (line, cycle, complete), custom length functions, high-dimensional embeddings, and integration with other components
  - Identified an issue with calling edgeWeight within callback functions, which needs to be addressed in a future task
  - Identified an issue with the ClassicalMds implementation when trying to embed a graph in a space with dimensions higher than what's needed, which causes NaN values in the coordinates
  - Identified an issue with the MetricSpherical2d implementation that outputs NaN values, causing the SparseSgd spherical drawing test to fail
  - Temporarily skipped the n-dimensional Euclidean drawing test with a clear comment explaining the issue, to be addressed in a future task
  - Temporarily skipped the spherical drawing test for SparseSgd with a clear comment explaining the issue, to be addressed in a future task
  - Verified test execution with `wasm-pack test --node --test <filename>`
- More comprehensive test suite with increased coverage needed for other components:
  - Other layout algorithms (SparseSgd, MDS, etc.)
  - Quality Metrics
  - Edge Bundling
  - Clustering
- Performance benchmarks for algorithm comparison
- Cross-platform consistency validation

## Current Status

- **Core Functionality**: ✅ Implemented and stable
- **Layout Algorithms**: 🔄 Functional but under active refinement
- **Drawing Implementations**: ✅ Complete
- **Quality Metrics**: ✅ Complete
- **Edge Bundling**: ✅ Functional
- **Clustering**: ✅ Functional
- **WebAssembly Bindings**: ✅ Functional
- **Python Bindings**: ✅ Functional
- **Documentation**: 🔄 In progress
- **Testing**: 🔄 In progress (WebAssembly binding tests for Rng, Graph, DiGraph, DrawingEuclidean2d, DrawingSpherical2d, DrawingHyperbolic2d, DrawingTorus2d, FullSgd, SparseSgd, and ClassicalMds classes completed)
- **Performance Optimization**: 🔄 Ongoing
- **Project Workflow**: ✅ Updated with new guidelines

## Development Workflow Improvements

### Git Command Usage

- Added standardized approach for git commands that might trigger pager views:
  - Always use `--no-pager` option with commands like `git diff`, `git log`, and `git show`
  - This prevents interactive pager (less) from requiring manual input
  - Documented in both `techContext.md` and `activeContext.md`
  - Examples added to quick reference and development setup sections

## Project Workflow Guidelines

New guidelines have been established for the project workflow:

1. **Test Execution from Project Root**:

   - All tests should be run from the project root directory using Cargo's workspace options
   - Use `cargo test --workspace` to run all tests
   - Use `cargo test -p <crate-name>` to run tests for a specific crate
   - Use `cargo test -p <crate-name> <test-name>` to run a specific test
   - For WebAssembly binding tests:
     - Run all WebAssembly tests: `wasm-pack test --node crates/wasm`
     - Run specific test files: `wasm-pack test --node crates/wasm --test <test-name>` (e.g., `wasm-pack test --node crates/wasm --test sgd_full`)
   - This approach ensures consistent test environment and better dependency resolution

2. **Commit Message Format**:

   - Follow the format: `<type>(<scope>): <description>`
   - For scope:
     - Use workspace crate names for changes specific to a crate
     - Omit scope for project-wide changes
   - Examples:
     - `feat(petgraph-layout-mds): add support for high-dimensional embeddings`
     - `fix(egraph-wasm): resolve NaN values in ClassicalMds implementation`
     - `test(petgraph-layout-sgd): add comprehensive tests for schedulers`
     - `docs: update project workflow guidelines` (project-wide change, no scope)

3. **Task Completion Process**:
   - When completing tasks, suggest appropriate commit messages following the format above
   - Ensure all tests are run from the project root before committing changes
   - **HIGHEST PRIORITY**: Always ask for final confirmation from the user before completing a task
     - Present a summary of all changes made
     - Include the proposed commit message
     - Wait for explicit approval before marking the task as complete
     - This confirmation step must never be skipped under any circumstances

## Known Issues

1. **Performance**:

   - Some layout algorithms may not scale well to very large graphs (>10,000 nodes)
   - SGD performance degrades with graph size, especially for full implementation
   - Stress majorization can be slow for large graphs

2. **Memory Usage**:

   - High memory consumption for dense graphs in WebAssembly context
   - Full distance matrices can exhaust memory for large graphs

3. **API Consistency**:

   - Some inconsistencies between language bindings (Rust, Python, JavaScript)
   - Naming conventions differ between platforms
   - Error handling approaches vary

4. **Documentation**:

   - Need more comprehensive examples and tutorials
   - Lack of detailed guidance on algorithm selection
   - Insufficient documentation on parameter tuning

5. **Cross-platform Issues**:

   - Some behaviors may differ slightly between Rust, Python, and JavaScript implementations
   - Performance characteristics vary across platforms

6. **Implementation Issues**:
   - ✅ Fixed: ClassicalMds implementation was producing NaN values when trying to embed a graph in a space with dimensions higher than what's needed for the graph
   - ✅ Fixed: PivotMds implementation was producing NaN values in high-dimensional embeddings due to issues in the power_iteration function
   - ✅ Fixed: MetricSpherical2d implementation had a bug that output NaN values, causing the SparseSgd spherical drawing test to fail. The issue was fixed by:
     - Adding safeguards against division by zero in vector normalization
     - Adding special handling for identical or very close points
     - Adding early returns for negligible movements
     - Ensuring proper clamping of values for trigonometric functions
     - Adding fallback strategies for edge cases near the poles
