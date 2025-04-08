# Active Context: egraph-rs

## Current Work Focus

The current focus is on enhancing the WebAssembly bindings with comprehensive tests, while continuing work on the layout algorithms in the graph visualization system:

1. **WebAssembly Binding Tests**

   - Creating dedicated test files for each class and function
   - Implementing tests that can be run individually with `wasm-pack test --node --test <filename>`
   - Testing both basic functionality and integration with other components
   - Following the pattern: `tests/<component>.rs` and `tests/<component>.js`
   - Initial implementation for the `Rng` class (random number generation)

2. **Layout Algorithms** in the graph visualization system, particularly:

3. **Stochastic Gradient Descent (SGD)** implementations

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

4. **Stress Majorization** algorithm

   - Iterative stress minimization
   - Conjugate gradient solver
   - Based on research: Gansner et al. (2004) "Graph drawing by stress majorization"

5. **Multidimensional Scaling (MDS)** implementations

   - Classical MDS: Standard implementation that computes a full distance matrix
   - Pivot MDS: Efficient implementation that uses a subset of nodes as pivots
   - Uses eigendecomposition and double centering to transform distance matrices
   - Based on research: Cox, T. F., & Cox, M. A. (2000). Multidimensional scaling. Chapman & Hall/CRC.

6. **Overlap Removal** techniques for improving readability

7. **Quality Metrics** for evaluating layout effectiveness
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

- Implemented Python binding tests for the DrawingSpherical2d class:

  - Created a comprehensive test file (`crates/python/tests/test_drawing_spherical_2d.py`) with tests for:
    - Constructor and basic instantiation
    - Node coordinate operations (get/set longitude/latitude)
    - Coordinate validation and range checking
    - Spherical coordinates and conversion to 3D points
    - Great circle distance calculations
    - Integration with Graph class
    - Testing with a large graph (Les Miserables)
  - Addressed floating-point precision issues by using appropriate delta values for comparisons
  - Modified tests for layout algorithms since KamadaKawai doesn't support spherical drawings directly
  - Verified that all tests pass, ensuring that the DrawingSpherical2d class works correctly in the Python bindings
  - Reused the test helpers module for spherical drawings, demonstrating the modularity of the test infrastructure

- Implemented Python binding tests for the DrawingEuclidean2d class:

  - Created a comprehensive test helpers module (`crates/python/tests/test_helpers.py`) with utility functions for:
    - Creating different graph structures (line, cycle, complete, star, grid)
    - Verifying drawing coordinates
    - Recording and comparing node positions
    - Calculating layout energy
    - Verifying layout quality
  - Created a comprehensive test file (`crates/python/tests/test_drawing_euclidean_2d.py`) with tests for:
    - Constructor and basic instantiation
    - Node coordinate operations (get/set x,y)
    - Drawing manipulation (centralize, clamp_region)
    - Edge segment representation
    - Integration with Graph class
    - Testing with a large graph (Les Miserables)
  - Verified that all tests pass, ensuring that the DrawingEuclidean2d class works correctly in the Python bindings
  - The test helpers module can be reused for implementing tests for other drawing implementations in the future

- Implemented Python binding tests for the KamadaKawai class:

  - Created a comprehensive test file (`crates/python/tests/test_kamada_kawai.py`) with tests for:
    - Basic constructor tests
    - Tests for the eps parameter getter and setter
    - Tests for node selection functionality
    - Tests for applying the algorithm to a single node
    - Tests for running the complete algorithm
    - Tests with a larger graph (Les Miserables)
    - Tests with custom distance functions
  - Fixed issues with the tests:
    - Fixed the floating-point comparison issue by using `assertAlmostEqual` with an appropriate delta
  - Verified that all tests pass, ensuring that the KamadaKawai algorithm works correctly in the Python bindings

- Implemented Python binding tests for the StressMajorization class:

  - Created a comprehensive test file (`crates/python/tests/test_stress_majorization.py`) with tests for:
    - Basic constructor tests
    - Tests for creating a StressMajorization instance from a distance matrix
    - Tests for applying a single iteration of the algorithm
    - Tests for running the complete algorithm until convergence
    - Tests for updating the weight matrix
    - Tests with a larger graph (Les Miserables)
    - Tests with custom distance functions
    - Tests for the epsilon and max_iterations parameters
  - Added getters and setters for the `epsilon` and `max_iterations` parameters to the Python bindings
  - Fixed issues with the tests:
    - Fixed the "Already borrowed" error by using a dictionary to store custom distances
    - Fixed the floating-point comparison issue by using `assertAlmostEqual` with a delta
  - Verified that all tests pass, ensuring that the StressMajorization algorithm works correctly in the Python bindings

- Fixed the StressMajorization implementation to prevent infinite loops:

  - Added a public `max_iterations` field with a default value of 100 to limit the number of iterations
  - Made the `epsilon` field public to allow external configuration of the convergence threshold
  - Simplified the `run` method to use `max_iterations` as a safety limit
  - Added WebAssembly bindings for `epsilon` and `max_iterations` parameters with getter and setter methods
  - Added comprehensive tests for parameter getters and setters
  - Updated test helpers to use the new parameters
  - This fix ensures that the StressMajorization algorithm will always terminate, even if convergence is not reached

- Removed the `createTestGraph` and `createTestDiGraph` functions from the WebAssembly binding test helpers and updated all affected test files:

  - Created separate specialized functions for each graph structure type:
    - `createLineGraph` and `createLineDiGraph` for path/line graphs
    - `createCycleGraph` and `createCycleDiGraph` for cycle/circular graphs
    - `createCompleteGraph` and `createCompleteDiGraph` for complete/fully-connected graphs
    - `createTriangleGraph` and `createTriangleDiGraph` for the simple triangle graph
    - Added `createStarDiGraph` as a directed version of the existing `createStarGraph`
    - Added `createGridDiGraph` as a directed version of the existing `createGridGraph`
  - Updated all test files that were using the removed functions to use the new specialized functions instead:
    - `sgd_full.js`
    - `sgd_sparse.js`
    - `classical_mds.js`
    - `kamada_kawai.js`
    - `stress_majorization.js`
    - `quality_metrics.js`
    - `drawing_euclidean_2d.js`
  - This refactoring makes the code more modular, easier to maintain, and provides more explicit functions for creating specific graph structures
  - All functions follow the same pattern, accepting appropriate parameters and returning an object with `{ graph, nodes }` structure
  - Added proper JSDoc documentation for all new functions

- Removed the `createDrawing` function from the WebAssembly binding test helpers:

  - Identified that the `createDrawing` function was not providing essential abstraction, similar to the previously removed `applyLayout` function
  - Updated all test files that were using this function (`quality_metrics.js`, `drawing_euclidean_2d.js`, `kamada_kawai.js`, and `stress_majorization.js`) to directly use the appropriate drawing initialization methods (e.g., `eg.DrawingEuclidean2d.initialPlacement(graph)`)
  - This change makes the tests more explicit about how each drawing type should be created
  - Improved the documentation value of the tests by showing direct API usage patterns
  - Verified that all tests still pass after the changes

- Fixed an issue in the WebAssembly binding tests where the `verifyNodePositions` function was failing:

  - Identified that when using object keys with computed property names like `[node1]`, JavaScript converts numeric node indices to strings
  - The drawing methods like `drawing.x()` expect numeric arguments, not strings
  - Fixed by converting the string node index back to a number using `Number(nodeIndexStr)` before passing it to the drawing methods
  - All tests are now passing (with one test intentionally ignored)

- Further refactored WebAssembly binding tests to improve maintainability and reduce code duplication:

  - Enhanced the test helpers module in `crates/wasm/tests/util/test_helpers.js` with additional functions:

    - Added `createStarGraph` and `createGridGraph` for more graph structure options
    - Added `createDrawing` to simplify drawing creation based on graph and drawing type
    - Added `verifyLayoutQuality` to check various quality aspects of layouts
    - Added `verifyLayoutImprovement` to compare layouts before and after algorithm application
    - Added `verifyNodePositions` to check if node positions match expected values

- Removed the `applyLayout` function from the WebAssembly binding test helpers:

  - Identified that the `applyLayout` function was not providing essential abstraction
  - Updated all test files that were using this function to directly instantiate and use the appropriate layout algorithm classes
  - This change makes the tests more explicit about how each layout algorithm should be used
  - Improved the documentation value of the tests by showing direct API usage patterns

  - Refactored additional test files to use the enhanced helper functions:

    - Updated `classical_mds.js` to use the helper functions
    - Updated `kamada_kawai.js` to use the helper functions
    - Updated `stress_majorization.js` to use the helper functions
    - Updated `drawing_euclidean_2d.js` to use the helper functions
    - Updated `quality_metrics.js` to use the helper functions

  - Benefits of the refactoring:
    - Reduced code duplication across test files
    - Improved test readability and maintainability
    - Standardized testing patterns for similar components
    - Made tests more concise while maintaining comprehensive coverage
    - Simplified the process of adding new tests in the future

- Skipped the StressMajorization run test to prevent infinite loop issues:

  - Added the `#[ignore]` attribute to the `stress_majorization_run` test function in `crates/wasm/tests/stress_majorization.rs`
  - Added a detailed comment explaining why the test is being skipped
  - Verified that the test is now being properly skipped while other tests continue to run successfully
  - This is a temporary solution until the underlying issue with the StressMajorization run method is fixed with proper convergence criteria or iteration limits

- Fixed the ClassicalMds implementation to handle cases where a graph is embedded in a space with dimensions higher than what's needed:

  - Added a threshold check for eigenvalues to prevent NaN values in the coordinates
  - Modified both `run()` and `run_2d()` methods to filter out negative or very small eigenvalues
  - Added comprehensive tests to verify the fix works for various dimensions
  - Updated the previously skipped WASM test for n-dimensional Euclidean drawings

- Fixed the PivotMds implementation to handle cases where a graph is embedded in a space with dimensions higher than what's needed:

  - Identified that the issue was in the power_iteration function, which was producing NaN values for certain edge cases
  - Added unit tests for the power_iteration function to reproduce and verify the issue with various matrix types
  - Fixed the power_iteration function by:
    - Initializing the vector with normalized values
    - Adding checks for zero or near-zero matrices
    - Handling cases where matrix multiplication results in very small values
    - Ensuring proper normalization of eigenvectors
    - Using the Rayleigh quotient for more stable eigenvalue calculation
    - Adding safeguards against division by very small numbers
  - Added comprehensive tests to verify the fix works for various dimensions

- Added comprehensive WebAssembly binding tests:

  - Created dedicated test files for the `Rng` class (`tests/rng.rs` and `tests/rng.js`)
  - Implemented tests for basic instantiation, seeded random number generation, and integration with layout algorithms
  - Created dedicated test files for the `Graph` class (`tests/graph.rs` and `tests/graph.js`)
  - Implemented tests for graph instantiation, node/edge operations, traversal, and integration with drawing components
  - Created dedicated test files for the `DiGraph` class (`tests/digraph.rs` and `tests/digraph.js`)
  - Implemented tests for directed graph functionality, including in/out neighbors, directed edge operations, and integration with drawing components
  - Created dedicated test files for the `DrawingEuclidean2d` class (`tests/drawing_euclidean_2d.rs` and `tests/drawing_euclidean_2d.js`)
  - Implemented tests for drawing instantiation, node coordinate operations, drawing manipulation (centralize, clamp_region), edge segment representation, and integration with Graph class
  - Created dedicated test files for the `DrawingSpherical2d` class (`tests/drawing_spherical_2d.rs` and `tests/drawing_spherical_2d.js`)
  - Implemented tests for spherical drawing instantiation, node coordinate operations (longitude/latitude), and integration with Graph class
  - Fixed an issue in the `DrawingSpherical2d` tests where nodes added to the graph after creating the drawing were not included in the drawing
  - Created dedicated test files for the `DrawingHyperbolic2d` class (`tests/drawing_hyperbolic_2d.rs` and `tests/drawing_hyperbolic_2d.js`)
  - Implemented tests for hyperbolic drawing instantiation, node coordinate operations, Poincaré disc model constraints, and integration with Graph class
  - Created dedicated test files for the `DrawingTorus2d` class (`tests/drawing_torus_2d.rs` and `tests/drawing_torus_2d.js`)
  - Implemented tests for torus drawing instantiation, node coordinate operations, torus wrapping behavior, edge segment representation, and integration with Graph class
  - Created dedicated test files for the `FullSgd` class (`tests/sgd_full.rs` and `tests/sgd_full.js`)
  - Implemented tests for FullSgd instantiation, scheduler creation, applying SGD to different drawing types (Euclidean 2D, Hyperbolic 2D, Spherical 2D, Torus 2D), updating distance and weight functions, shuffling node pairs, and integration with other components
  - Created dedicated test files for the `SparseSgd` class (`tests/sgd_sparse.rs` and `tests/sgd_sparse.js`)
  - Implemented tests for SparseSgd instantiation, pivot node configuration, scheduler creation, applying SGD to different drawing types (Euclidean 2D, Hyperbolic 2D, Torus 2D), updating distance and weight functions, shuffling node pairs, and integration with other components
  - Identified an issue with the MetricSpherical2d implementation that outputs NaN values, causing the SparseSgd spherical drawing test to fail
  - Temporarily skipped the spherical drawing test for SparseSgd with a clear comment explaining the issue, to be addressed in a future task
  - Created dedicated test files for the `ClassicalMds` class (`tests/classical_mds.rs` and `tests/classical_mds.js`)
  - Implemented tests for ClassicalMds instantiation, 2D layout generation (run2d method), n-dimensional layout generation (run method), different graph structures (line, cycle, complete), custom length functions, high-dimensional embeddings, and integration with other components
  - Identified an issue with calling edgeWeight within callback functions, which needs to be addressed in a future task
  - Identified an issue with the ClassicalMds implementation when trying to embed a graph in a space with dimensions higher than what's needed, which causes NaN values in the coordinates
  - Temporarily skipped the n-dimensional Euclidean drawing test with a clear comment explaining the issue, to be addressed in a future task
  - Created dedicated test files for the `KamadaKawai` class (`tests/kamada_kawai.rs` and `tests/kamada_kawai.js`)
  - Implemented tests for KamadaKawai instantiation, epsilon parameter getter/setter, node selection functionality, single-node application, complete algorithm run, and integration with other components
  - Fixed an issue in the epsilon parameter test by using approximate comparison for floating-point values
  - Created dedicated test files for the `StressMajorization` class (`tests/stress_majorization.rs` and `tests/stress_majorization.js`)
  - Implemented tests for StressMajorization instantiation, applying a single iteration, and integration with other components
  - Identified an issue with the StressMajorization run method that can cause infinite loops, and implemented a workaround using multiple apply calls
  - Established a pattern for class/function-specific tests that can be run individually
  - Verified test execution with `wasm-pack test --node --test <filename>`

- Refined various layout algorithm implementations
- Improved documentations of drawing implementations for different geometric spaces:
  - Euclidean 2D: 2D Cartesian coordinate drawings
  - Spherical 2D: Drawings on a spherical surface
  - Hyperbolic 2D: Drawings in hyperbolic space
  - Torus 2D: Drawings on a torus surface
  - N-dimensional Euclidean: Drawings in higher-dimensional Euclidean spaces
- Enhanced WebAssembly bindings with improved JavaScript API features:
  - Consistent naming conventions (camelCase for JavaScript)
  - JSDoc-style comments for better developer experience
  - Transparent data handling between Rust and JavaScript
  - Memory safety improvements
  - Method chaining support
  - Robust error handling
  - JavaScript callback support

## Next Steps

1. **Python Binding Tests**:

   - Plan comprehensive tests for all Python bindings, following the same approach used for WebAssembly binding tests
   - Create a test helpers module with common utility functions:
     - Graph creation helpers (line, cycle, complete, star, grid)
     - Position verification helpers
     - Layout quality verification helpers
   - Implement tests for Graph classes:
     - Basic graph operations
     - Node and edge management
     - Traversal methods
   - Implement tests for Drawing implementations:
     - `DrawingEuclidean2d`
     - `DrawingEuclidean` (n-dimensional)
     - `DrawingHyperbolic2d`
     - `DrawingSpherical2d`
     - `DrawingTorus2d`
   - Implement tests for Layout algorithms:
     - `KamadaKawai`
     - `StressMajorization`
     - `OverwrapRemoval`
   - Implement tests for Utility classes:
     - `Rng` (random number generation)
     - `DistanceMatrix`
   - Ensure consistent test coverage between Python and WebAssembly bindings
   - Account for Python-specific API differences

2. **WebAssembly Binding Tests**:

   - ✅ Completed: Implemented tests for the `QualityMetrics` module (`tests/quality_metrics.rs` and `tests/quality_metrics.js`)
   - ✅ Completed: Implemented tests for the Edge Bundling module (`tests/edge_bundling.rs` and `tests/edge_bundling.js`)
   - ✅ Completed: Implemented tests for the Clustering module (`tests/clustering.rs` and `tests/clustering.js`)
   - ✅ All WebAssembly binding tests are now implemented.
   - Ensure comprehensive coverage of all public API methods
   - Add edge cases and error handling tests
   - ✅ Fixed: ClassicalMds implementation for n-dimensional Euclidean drawings
   - ✅ Fixed: PivotMds implementation for high-dimensional embeddings
   - ✅ Fixed: MetricSpherical2d implementation that was causing NaN values
   - ✅ Fixed: StressMajorization run method to prevent infinite loops

3. **Layout Algorithm Refinement**:

   - Fine-tune SGD schedulers for better convergence
   - Optimize stress majorization for large graphs
   - Improve performance of overlap removal
   - Address performance issues with large graphs (>10,000 nodes)
   - Implement additional layout algorithms (e.g., additional force-directed variants)

4. **Integration Improvements**:

   - Ensure consistent behavior across drawing implementations
   - Improve interoperability between layout algorithms
   - Resolve inconsistencies between language bindings (Rust, Python, JavaScript)

5. **Documentation and Examples**:

   - Add examples showcasing different layout algorithms
   - Document best practices for selecting appropriate layout algorithms
   - Create comprehensive API documentation
   - Develop tutorials for common use cases
   - Create usage examples for different geometric spaces

6. **Testing Enhancements**:
   - Continue expanding test coverage for all components
   - Implement performance benchmarks
   - Validate cross-platform behavior consistency

## Project Workflow Guidelines

The following guidelines have been established for the project workflow:

1. **Test Execution from Project Root**:

   - All tests should be run from the project root directory using Cargo's workspace options
   - Use `cargo test --workspace` to run all tests
   - Use `cargo test -p <crate-name>` to run tests for a specific crate (e.g., `cargo test -p petgraph-layout-mds`)
   - Use `cargo test -p <crate-name> <test-name>` to run a specific test (e.g., `cargo test -p egraph-wasm sgd_full`)
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

## Active Decisions and Considerations

1. **Performance Trade-offs**:

   - Balance between layout quality and computation speed
   - Memory usage optimization for large graphs
   - Addressing high memory consumption for dense graphs in WebAssembly context

2. **API Design**:

   - Ensure consistent interfaces across layout implementations
   - Consider future extensibility for new layout algorithms
   - Maintain API stability while allowing for improvements

3. **Cross-platform Consistency**:

   - Validate behavior consistency across Rust, WebAssembly, and Python
   - Ensure uniform error handling across language boundaries
   - Maintain consistent naming conventions appropriate to each language

4. **Known Issues to Address**:
   - ✅ Fixed: PivotMds implementation was producing NaN values when trying to embed a graph in a space with dimensions higher than what's needed for the graph, similar to the fixed issue in ClassicalMds
   - ✅ Fixed: MetricSpherical2d implementation had a bug that output NaN values, causing the SparseSgd spherical drawing test to fail. Fixed by adding safeguards against division by zero, handling edge cases for identical or very close points, and ensuring proper clamping of values for trigonometric functions.
   - ✅ Fixed: StressMajorization run method could enter an infinite loop. Fixed by adding a max_iterations parameter and making epsilon configurable to control convergence criteria.
   - Need to address: Performance issues with large graphs (>10,000 nodes)
   - Need to address: High memory consumption for dense graphs in WebAssembly context
   - Need to address: Inconsistencies between language bindings (Rust, Python, JavaScript)

## Git Command Usage

When executing git commands that might trigger a pager view (such as `git diff`, `git log`, or `git show`), always use the `--no-pager` option to prevent interactive pager (less) from requiring manual input:

```bash
git --no-pager diff
git --no-pager log
git --no-pager show
```

This approach ensures that git commands can be executed without requiring interactive user operations to navigate through the output.
