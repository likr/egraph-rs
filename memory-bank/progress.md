# Progress: egraph-rs

## What Works

### Graph Data Structures

- Base graph implementations (undirected and directed graphs)
- Node and edge management
- Generic data storage
- Common type definitions (`Node`, `Edge`, `IndexType`)

### Graph Algorithms

- Connected Components (finding connected subgraphs)
- Shortest Path (finding shortest paths between nodes)
- Delaunay Triangulation (creating triangulation based on node positions)
- Layering (assigning layers to nodes in directed graphs)
  - Cycle removal to create acyclic graphs
  - Longest path algorithm for layer assignment
  - Extensible trait-based architecture for future layering algorithms

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
  - Configurable convergence criteria (epsilon)
  - Maximum iteration limit to prevent infinite loops
- Kamada-Kawai (spring model based layout)
- Overlap Removal (resolving node overlaps)
- Separation Constraints (layout constraint implementation)
  - Rectangle overlap constraints for all node pairs
  - Triangulation-based rectangle overlap constraints (more efficient for large graphs)
  - Layered constraints for hierarchical layouts (Sugiyama Framework)
  - Cluster overlap constraints for removing overlaps between node clusters

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
- ✅ Fixed: ClassicalMds implementation to handle cases where a graph is embedded in a space with dimensions higher than what's needed
- ✅ Fixed: PivotMds implementation to handle similar cases with high-dimensional embeddings
- ✅ Fixed: StressMajorization run method to prevent infinite loops by adding max_iterations parameter and making epsilon configurable
- 🔄 Planned: Additional layering algorithms for graph hierarchical layout
  - Network simplex algorithm
  - Coffman-Graham algorithm
  - Optimal layer assignment with integer programming

### Documentation

- ✅ Restructured Python API documentation with a hierarchical organization:

  - Created a hierarchical structure for the API documentation with folders for each main category
  - Organized documentation into logical groups:
    - Graph module (Graph, DiGraph, GraphAdapter)
    - Drawing module (DrawingEuclidean2d, DrawingEuclidean, DrawingSpherical2d, DrawingHyperbolic2d, DrawingTorus2d)
    - Layout module with submodules:
      - SGD (FullSgd, SparseSgd, Schedulers)
      - MDS (ClassicalMds, PivotMds)
      - Other layout algorithms (StressMajorization, KamadaKawai, OverwrapRemoval)
    - Algorithm module
    - Quality Metrics module
    - Distance Matrix module
    - Random Number Generation module
  - Added documentation for previously undocumented components:
    - GraphAdapter base class
    - SGD schedulers (SchedulerConstant, SchedulerLinear, SchedulerQuadratic, SchedulerExponential, SchedulerReciprocal)
  - This restructuring improves navigation, discoverability, and maintainability of the documentation

- ✅ Fixed Python binding documentation title underline warnings:

  - Fixed title underline length issues in all example files:
    - `crates/python/docs/source/examples/index.rst`
    - `crates/python/docs/source/examples/sgd.rst`
    - `crates/python/docs/source/examples/kamada_kawai.rst`
    - `crates/python/docs/source/examples/stress_majorization.rst`
    - `crates/python/docs/source/examples/sgd_3d.rst`
    - `crates/python/docs/source/examples/sgd_hyperbolic_2d.rst`
    - `crates/python/docs/source/examples/sgd_spherical_2d.rst`
    - `crates/python/docs/source/examples/sgd_torus.rst`
    - `crates/python/docs/source/examples/overwrap_removal.rst`
  - Fixed title underline length in the main index file:
    - `crates/python/docs/source/index.rst`
  - Made all title underlines sufficiently long to match or exceed their title text
  - Verified that documentation now builds without errors when using the `-W` flag to treat warnings as errors
  - These changes ensure that the Python bindings documentation renders correctly with consistent formatting

- ✅ Fixed Python binding documentation warnings:

  - Resolved duplicate object descriptions by removing redundant `automodule` directives from `index.rst`
  - Fixed title underline length issues in all API documentation files
  - Updated `algorithm.rst` to reference actual functions that exist (`all_sources_bfs`, `all_sources_dijkstra`, `warshall_floyd`) instead of non-existent functions
  - Removed reference to the non-existent `edge_angle` function from `quality_metrics.rst`
  - Fixed indentation error in StressMajorization docstring by adding a blank line before the numbered list
  - Changed the docstring example format in StressMajorization from doctest format to code block format
  - Verified that documentation now builds without errors when using the `-W` flag to treat warnings as errors
  - These changes ensure that the Python bindings documentation is properly formatted and free of critical warnings

- ✅ Updated Python binding documentation format to follow Sphinx recommendations for the OverwrapRemoval module:

  - Updated documentation in `crates/python/src/layout/overwrap_removal.rs` to ensure consistent Sphinx format
  - Changed internal Rust documentation format (`# Parameters`, `# Returns`) to Sphinx format (`:param:`, `:type:`, `:return:`, `:rtype:`) for all Python-exposed functions and methods
  - Added `:raises:` directive to the constructor method to document potential exceptions
  - Enhanced descriptions for parameters and return values
  - Ensured consistent formatting across all methods
  - This change completes the standardization of Python bindings documentation to Sphinx format across the entire codebase

- ✅ Updated Python binding documentation format to follow Sphinx recommendations for additional modules:

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

- ✅ Updated Python binding documentation format to follow Sphinx recommendations for SGD module:

  - Improved documentation in all SGD module files to ensure consistent Sphinx format:
    - `crates/python/src/layout/sgd/schedulers.rs`
    - `crates/python/src/layout/sgd/sparse.rs`
    - `crates/python/src/layout/sgd/full.rs`
    - `crates/python/src/layout/sgd/distance_adjusted_sparse.rs`
    - `crates/python/src/layout/sgd/distance_adjusted_full.rs`
  - Added class-level parameter documentation for all classes
  - Added `:raises:` directives where methods might raise exceptions
  - Enhanced descriptions for distance adjustment parameters
  - Provided more detailed explanations for complex parameters like `alpha`
  - Ensured consistent formatting across all files
  - This change ensures the Python bindings documentation follows the Sphinx format consistently, making it more accessible to users

- ✅ Updated Python binding documentation format to follow Sphinx recommendations for drawing module:

  - Modified documentation in all drawing module files to ensure only Python-exposed functions and classes use Sphinx format
  - Converted internal Rust utility functions from Sphinx format to standard Rust documentation format
  - Changed `:param name: description` to `# Parameters\n/// * `name` - description` for internal Rust functions
  - Changed `:return: description` to `# Returns\n/// description` for internal Rust functions
  - Removed `:type:` and `:rtype:` directives from internal Rust functions
  - Kept Sphinx format for all methods in `#[pymethods]` blocks and functions with `#[staticmethod]` attribute
  - Modified files include drawing module files (Euclidean2d, Spherical2d, Hyperbolic2d, Torus2d, Euclidean)
  - This change ensures documentation follows project standards with appropriate format for each language's ecosystem

- ✅ Implemented Python bindings documentation with Sphinx integration:

  - Created a comprehensive Sphinx documentation structure in `crates/python/docs/`
  - Set up configuration for doctest support to ensure examples are runnable and testable
  - Created detailed API reference documentation for all modules:
    - Graph module (Graph, DiGraph)
    - Drawing module (Euclidean2d, Euclidean, Spherical2d, Hyperbolic2d, Torus2d)
    - Layout module (StressMajorization, KamadaKawai, SGD, MDS, OverwrapRemoval)
    - Algorithm module (shortest_path, connected_components)
    - Quality metrics module (stress, crossing_number, etc.)
    - Distance matrix module
    - Random number generation module
  - Added example code for all layout algorithms and drawing spaces
  - Created build scripts (Makefile and make.bat) for building documentation on different platforms
  - Added requirements.txt for documentation dependencies
  - Configured for Netlify integration for automated documentation building
  - This documentation will make the library more accessible to users and ensure examples are always up-to-date

- 🔄 In progress: Move Python examples to doctests

  - ✅ Migrated stress_majorization.py example to a doctest in the StressMajorization module
  - Plan to migrate remaining examples from `crates/python/examples/` to Python doctests
  - Analyze each example file to extract core functionality
  - Create doctests in the corresponding Python module files
  - Simplify examples to focus on API usage rather than visualization
  - Ensure compatibility with the current API
  - Add appropriate documentation to explain purpose and usage
  - Exclude `nonconnected_sgd.py` as specified
  - Note that existing examples may not be maintained and might not work properly
  - This will improve the discoverability and maintainability of examples
  - Doctests will serve as both documentation and tests, ensuring they stay up-to-date

- ✅ Updated Netlify configuration for Python bindings documentation:

  - Changed the base directory from `crates/python/docs` to `crates/python`:
    - Moved `netlify.toml`, `requirements.txt`, and `runtime.txt` from `crates/python/docs/` to `crates/python/`
    - Updated the publish directory from `_build/html` to `docs/_build/html`
    - Modified the build command from `rustup toolchain install stable && pip install .. && make html` to `rustup toolchain install stable && pip install . && make -C docs html`
  - This change simplifies the configuration by setting the base at the Python crate level
  - The updated command uses `make -C docs html` to run make from the crates/python directory
  - This ensures that Netlify will build the documentation correctly with the new directory structure

- 🔄 Planned: Additional documentation improvements
  - More comprehensive examples showcasing different layout algorithms
  - Detailed API documentation across all interfaces
  - Tutorials for common use cases
  - Best practices for selecting appropriate layout algorithms
  - Usage examples for different geometric spaces

### Testing

- Python binding tests:

  - ✅ Created a comprehensive test helpers module in `crates/python/tests/test_helpers.py` with:
    - Graph creation helpers for different graph structures (line, cycle, complete, star, grid)
    - Position verification helpers for different geometric spaces
    - Layout quality verification helpers
    - Functions for recording and comparing node positions
    - Functions for calculating layout energy
  - ✅ Implemented tests for Graph classes:
    - Basic graph operations (constructor, node/edge addition/removal)
    - Node and edge management (weight access, contains/find operations)
    - Traversal methods (neighbors, edges, externals)
    - Map and filter_map operations
    - NetworkX conversion
    - Large graph handling (Les Miserables dataset)
  - 🔄 In progress: Implement tests for Drawing implementations:
    - ✅ `DrawingEuclidean2d` - Implemented comprehensive tests in `crates/python/tests/test_drawing_euclidean_2d.py`
    - ✅ `DrawingEuclidean` (n-dimensional) - Implemented comprehensive tests in `crates/python/tests/test_drawing_euclidean.py`
    - ✅ `DrawingHyperbolic2d` - Implemented comprehensive tests in `crates/python/tests/test_drawing_hyperbolic_2d.py`
    - ✅ `DrawingSpherical2d` - Implemented comprehensive tests in `crates/python/tests/test_drawing_spherical_2d.py`
    - ✅ `DrawingTorus2d` - Implemented comprehensive tests in `crates/python/tests/test_drawing_torus_2d.py`
  - 🔄 In progress: Implement tests for Layout algorithms:
    - ✅ `KamadaKawai` - Implemented comprehensive tests in `crates/python/tests/test_kamada_kawai.py`
    - ✅ `StressMajorization` - Implemented comprehensive tests in `crates/python/tests/test_stress_majorization.py`
    - ✅ `OverwrapRemoval` - Implemented comprehensive tests in `crates/python/tests/test_overwrap_removal.py`
  - ✅ Implemented tests for Utility classes:
    - ✅ `Rng` - Implemented comprehensive tests in `crates/python/tests/test_rng.py`
    - ✅ `DistanceMatrix` - Implemented comprehensive tests in `crates/python/tests/test_distance_matrix.py` with tests for:
      - Basic instantiation with different graph types (undirected, directed, empty, single-node)
      - Distance retrieval between connected and disconnected nodes
      - Distance modification and behavior differences between undirected and directed graphs
      - Integration with StressMajorization
      - Custom edge weights
      - Discovered that setting distances in one direction doesn't automatically set them in the reverse direction
  - 🔄 Planned: Ensure consistent test coverage between Python and WebAssembly bindings
  - 🔄 Planned: Account for Python-specific API differences

- WebAssembly binding tests:
  - ✅ Created a comprehensive test helpers module in `crates/wasm/tests/util/test_helpers.js` with:
    - Graph creation helpers for different graph structures (line, cycle, complete, etc.)
    - Position recording helpers for different geometric spaces (2D, spherical, n-dimensional)
    - Verification helpers for position changes, coordinate validity, and geometric constraints
    - RNG helpers for creating seeded random number generators
    - Layout quality helpers to verify that connected nodes are positioned closer together
    - Added additional helper functions:
      - `createStarGraph` and `createGridGraph` for more graph structure options
      - `createLineGraph`, `createLineDiGraph`, `createCycleGraph`, `createCycleDiGraph`, `createCompleteGraph`, `createCompleteDiGraph`, `createTriangleGraph`, `createTriangleDiGraph`, `createStarDiGraph`, and `createGridDiGraph` for specific graph structures
  - ✅ Removed the `createTestGraph` and `createTestDiGraph` functions and updated all affected test files to use the specialized functions instead:
    - Updated `sgd_full.js`, `sgd_sparse.js`, `classical_mds.js`, `kamada_kawai.js`, `stress_majorization.js`, `quality_metrics.js`, and `drawing_euclidean_2d.js` to use the specialized graph creation functions
    - This change makes the code more modular, easier to maintain, and provides more explicit functions for creating specific graph structures
      - `verifyLayoutQuality` to check various quality aspects of layouts
      - `verifyLayoutImprovement` to compare layouts before and after algorithm application
      - `verifyNodePositions` to check if node positions match expected values
  - ✅ Removed the `applyLayout` and `createDrawing` functions as they were not providing essential abstraction
  - ✅ Updated all test files that were using these functions to directly instantiate and use the appropriate classes
  - ✅ Updated all test files that were using this function to directly instantiate and use the appropriate layout algorithm classes
  - ✅ Fixed an issue in the `verifyNodePositions` function:
    - When using object keys with computed property names like `[node1]`, JavaScript converts numeric node indices to strings
    - The drawing methods like `drawing.x()` expect numeric arguments, not strings
    - Fixed by converting the string node index back to a number using `Number(nodeIndexStr)` before passing it to the drawing methods
    - All tests are now passing (with one test intentionally ignored)
  - ✅ Refactored tests to use the helper functions:
    - Updated `sgd_full.js` to use the helper functions for all tests
    - Updated `sgd_sparse.js` to use the helper functions for all tests
    - Updated `classical_mds.js` to use the helper functions
    - Updated `kamada_kawai.js` to use the helper functions
    - Updated `stress_majorization.js` to use the helper functions
    - Updated `drawing_euclidean_2d.js` to use the helper functions
    - Updated `quality_metrics.js` to use the helper functions
  - ✅ Fixed an issue where helper functions were assuming drawings had graph references
    - Modified helpers to take explicit graph parameters
  - ✅ Implemented dedicated test files for the `Rng` class (`tests/rng.rs` and `tests/rng.js`)
  - ✅ Implemented dedicated test files for the `Graph` class (`tests/graph.rs` and `tests/graph.js`)
  - ✅ Implemented dedicated test files for the `DiGraph` class (`tests/digraph.rs` and `tests/digraph.js`)
  - ✅ Implemented dedicated test files for the `DrawingEuclidean2d` class (`tests/drawing_euclidean_2d.rs` and `tests/drawing_euclidean_2d.js`)
  - ✅ Implemented dedicated test files for the `DrawingSpherical2d` class (`tests/drawing_spherical_2d.rs` and `tests/drawing_spherical_2d.js`)
  - ✅ Implemented dedicated test files for the `DrawingHyperbolic2d` class (`tests/drawing_hyperbolic_2d.rs` and `tests/drawing_hyperbolic_2d.js`)
  - ✅ Implemented dedicated test files for the `DrawingTorus2d` class (`tests/drawing_torus_2d.rs` and `tests/drawing_torus_2d.js`)
  - ✅ Implemented dedicated test files for the `FullSgd` class (`tests/sgd_full.rs` and `tests/sgd_full.js`)
  - ✅ Implemented dedicated test files for the `ClassicalMds` class (`tests/classical_mds.rs` and `tests/classical_mds.js`)
  - ✅ Implemented dedicated test files for the `SparseSgd` class (`tests/sgd_sparse.rs` and `tests/sgd_sparse.js`)
  - ✅ Fixed an issue in the `DrawingSpherical2d` tests where nodes added to the graph after creating the drawing were not included in the drawing
  - ✅ Created a pattern for class/function-specific tests that can be run individually
  - ✅ Tests for basic functionality, node/edge operations, traversal, and integration with other components
  - ✅ Tests for directed graph functionality, including in/out neighbors and directed edge operations
  - ✅ Tests for drawing functionality, including node coordinate operations, drawing manipulation, edge segment representation, and integration with Graph class
  - ✅ Tests for spherical drawing functionality, including longitude/latitude coordinate operations and integration with Graph class
  - ✅ Tests for hyperbolic drawing functionality, including coordinate operations, Poincaré disc model constraints, and integration with Graph class
  - ✅ Tests for torus drawing functionality, including coordinate operations, torus wrapping behavior, edge segment representation, and integration with Graph class
  - ✅ Tests for FullSgd functionality, including instantiation, scheduler creation, applying SGD to different drawing types, updating distance and weight functions, shuffling node pairs, and integration with other components
  - ✅ Tests for SparseSgd functionality, including instantiation, pivot node configuration, scheduler creation, applying SGD to different drawing types, updating distance and weight functions, shuffling node pairs, and integration with other components
  - ✅ Tests for ClassicalMds functionality, including instantiation, 2D layout generation (run2d method), n-dimensional layout generation (run method), different graph structures (line, cycle, complete), custom length functions, high-dimensional embeddings, and integration with other components
  - ✅ Created dedicated test files for the `KamadaKawai` class (`tests/kamada_kawai.rs` and `tests/kamada_kawai.js`)
  - ✅ Implemented tests for KamadaKawai instantiation, epsilon parameter getter/setter, node selection functionality, single-node application, complete algorithm run, and integration with other components
  - ✅ Fixed an issue in the epsilon parameter test by using approximate comparison for floating-point values
  - ✅ Created dedicated test files for the `StressMajorization` class (`tests/stress_majorization.rs` and `tests/stress_majorization.js`)
  - ✅ Implemented tests for StressMajorization instantiation, applying a single iteration, and integration with other components
  - ✅ Identified an issue with the StressMajorization run method that can cause infinite loops, and implemented a workaround using multiple apply calls
  - ✅ Identified an issue with calling edgeWeight within callback functions, which needs to be addressed in a future task
  - ✅ Identified an issue with the ClassicalMds implementation when trying to embed a graph in a space with dimensions higher than what's needed, which causes NaN values in the coordinates
  - ✅ Identified an issue with the MetricSpherical2d implementation that outputs NaN values, causing the SparseSgd spherical drawing test to fail
  - ✅ Fixed the n-dimensional Euclidean drawing test by adding threshold checks for eigenvalues
  - ✅ Fixed the spherical drawing test for SparseSgd by adding safeguards against division by zero and handling edge cases
  - ✅ Verified test execution with `wasm-pack test --node --test <filename>`
  - ✅ Completed: Created dedicated test files for the `QualityMetrics` module (`tests/quality_metrics.rs` and `tests/quality_metrics.js`)
  - ✅ Implemented tests for stress metric, crossing number in Euclidean and torus spaces, neighborhood preservation, and integration with layout algorithms
  - ✅ Completed: Created dedicated test files for the `EdgeBundling` module (`tests/edge_bundling.rs` and `tests/edge_bundling.js`)
  - ✅ Implemented tests for basic functionality, complex graphs, result structure verification, and integration with other components
  - ✅ Refactored tests to use helper functions for common verification tasks
  - ✅ Completed: Created dedicated test files for the `Clustering` module (`tests/clustering.rs` and `tests/clustering.js`)
  - ✅ Implemented tests for basic coarsening, complex graph coarsening, custom node and edge merging, and integration with other components
  - ✅ Addressed challenges with JavaScript Map objects and recursive borrowing issues
  - ✅ All WebAssembly binding tests are now implemented
- Performance benchmarks for algorithm comparison
- Cross-platform consistency validation

## Current Status

- **Core Functionality**: ✅ Implemented and stable
- **Layout Algorithms**: 🔄 Functional but under active refinement
  - ✅ Fixed: StressMajorization run method to prevent infinite loops
  - ✅ Fixed: ClassicalMds implementation for n-dimensional Euclidean drawings
  - ✅ Fixed: PivotMds implementation for high-dimensional embeddings
  - ✅ Fixed: MetricSpherical2d implementation that was causing NaN values
  - ✅ Added: Layering algorithm crate with trait-based extensible architecture
- **Drawing Implementations**: ✅ Complete
- **Quality Metrics**: ✅ Complete
- **Edge Bundling**: ✅ Functional
- **Clustering**: ✅ Functional
- **WebAssembly Bindings**: ✅ Functional
  - ✅ Added getter/setter methods for StressMajorization parameters
  - ✅ Improved error handling and parameter validation
- **Python Bindings**: ✅ Functional
- **Documentation**: 🔄 In progress
  - ✅ Completed: Updated all Python binding documentation to follow Sphinx format
- **Testing**: ✅ WebAssembly binding tests completed for all components
  - ✅ Comprehensive test suite for all WebAssembly classes and functions
  - ✅ Improved test helpers for creating graph structures and verifying layouts
  - 🔄 Performance benchmarks still needed
- **Performance Optimization**: 🔄 Ongoing
  - 🔄 Need to address performance issues with large graphs (>10,000 nodes)
  - 🔄 Need to optimize memory usage for dense graphs in WebAssembly context
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

1. **Plan Mode Workflow**:

   - Enhanced workflow for planning and executing tasks:
     - Task understanding and Memory Bank review
     - Solution plan development and presentation to user
     - User approval and transition to Act Mode
     - Implementation with task completion confirmation
     - Memory Bank updates after task completion
     - Commit message suggestion
   - Added comprehensive flowchart in `techContext.md` to visualize the workflow
   - This structured approach ensures consistent task execution and documentation
   - Improves project maintainability and knowledge transfer

2. **Test Execution from Project Root**:

   - All tests should be run from the project root directory using Cargo's workspace options
   - Use `cargo test --workspace` to run all tests
   - Use `cargo test -p <crate-name>` to run tests for a specific crate
   - Use `cargo test -p <crate-name> <test-name>` to run a specific test
   - For WebAssembly binding tests:
     - Run all WebAssembly tests: `wasm-pack test --node crates/wasm`
     - Run specific test files: `wasm-pack test --node crates/wasm --test <test-name>` (e.g., `wasm-pack test --node crates/wasm --test sgd_full`)
   - This approach ensures consistent test environment and better dependency resolution

3. **Commit Message Format**:

   - Follow the format: `<type>(<scope>): <description>`
   - For scope:
     - Use workspace crate names for changes specific to a crate
     - Omit scope for project-wide changes
   - Examples:
     - `feat(petgraph-layout-mds): add support for high-dimensional embeddings`
     - `fix(egraph-wasm): resolve NaN values in ClassicalMds implementation`
     - `test(petgraph-layout-sgd): add comprehensive tests for schedulers`
     - `docs: update project workflow guidelines` (project-wide change, no scope)

4. **Task Completion Process**:
   - When completing tasks, suggest appropriate commit messages following the format above
   - Ensure all tests are run from the project root before committing changes
   - **HIGHEST PRIORITY**: Always ask for final confirmation from the user before completing a task
     - Present a summary of all changes made
     - Include the proposed commit message
     - Wait for explicit approval before marking the task as complete
     - This confirmation step must never be skipped under any circumstances
   - After user approval:
     - Update Memory Bank to reflect changes
     - Suggest final commit message
     - Report task completion

## Known Issues

1. **Performance**:

   - Some layout algorithms may not scale well to very large graphs (>10,000 nodes)
   - SGD performance degrades with graph size, especially for full implementation
   - Stress majorization can be slow for large graphs

2. **Memory Usage**:

   - High memory consumption for dense graphs in WebAssembly context
   - Full distance matrices can exhaust memory for large graphs

3. **API Consistency**:

   - ✅ Fixed: DrawingEuclidean Python bindings API inconsistencies:
     - Renamed `set_x` method to `set` for better consistency with other Drawing classes
     - Added `len()` method to match other Drawing classes
     - Updated tests to use the new method names
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
   - ✅ Fixed: MetricSpherical2d implementation had a bug that output N
