# Active Context: egraph-rs

## Current Work Focus

The project has reached a mature state with comprehensive functionality across multiple domains:

1. **Complete Algorithm Suite**

   - **Graph Algorithms**: Connected components, shortest path, triangulation, layering
   - **Layout Algorithms**: SGD (Full/Sparse/Distance-Adjusted/Omega), MDS (Classical/Pivot), Stress Majorization, Kamada-Kawai
   - **Community Detection**: Louvain, Label Propagation, Spectral Clustering, InfoMap with unified trait interface
   - **Specialized Features**: Edge bundling, overlap removal, separation constraints, quality metrics

2. **Cross-Platform Language Bindings**

   - **Python Bindings**: Complete PyO3-based API with comprehensive coverage
     - All community detection algorithms with coarsening functionality
     - Layering algorithms with cycle detection and removal
     - Triangulation with Delaunay triangulation support
     - Separation constraints with rectangle overlap prevention
     - All layout algorithms with drawing space support
   - **WebAssembly Bindings**: JavaScript-friendly interfaces with comprehensive test coverage
     - Modular structure with clear separation of concerns
     - Memory-safe type conversions and error handling
     - Callback support for algorithm customization

3. **Robust Architecture**

   - **Modular Crate Design**: 15+ specialized crates for different functionality
   - **Trait-Based Interfaces**: Consistent APIs across algorithms (CommunityDetection, LayeringAlgorithm)
   - **Multiple Geometric Spaces**: Euclidean (2D/nD), Spherical, Hyperbolic, Torus drawings
   - **Quality Metrics**: Comprehensive evaluation suite for layout assessment

4. **Performance & Reliability**

   - **Algorithm Optimizations**: Fixed infinite loops, improved convergence, enhanced performance
   - **Memory Efficiency**: Replaced external dependencies with built-in Rust collections
   - **Comprehensive Testing**: Unit tests, integration tests, cross-language validation
   - **Documentation**: Sphinx-format documentation with examples and best practices

## Recent Changes

- **OmegaBuilder Parameter Renaming and Cleanup (2025-06-04)**

  - **Parameter Renaming for Better API Clarity**: Renamed OmegaBuilder parameters to be more descriptive and reflect their actual purpose in eigenvalue computation

  - **Parameter Name Changes**:

    - **`max_iterations`** → **`eigenvalue_max_iterations`**: More clearly indicates this controls the maximum iterations for eigenvalue computation using inverse power method
    - **`tolerance`** → **`eigenvalue_tolerance`**: Clarifies this is specifically the convergence tolerance for eigenvalue computation

  - **Removed Unused Parameter**:

    - **`vector_tolerance`**: Completely removed this parameter as it was not actually used in the eigenvalue computation implementation
    - Eliminated dead code that could confuse API users

  - **Files Updated**:

    - **eigenvalue.rs**: Removed `vector_tolerance` parameter from `compute_smallest_eigenvalues_with_laplacian` function signature and documentation
    - **omega.rs**: Updated OmegaBuilder struct with new parameter names, updated all builder methods, updated documentation and function calls
    - **omega.rs (CLI)**: Updated CLI usage to use new parameter method names

  - **Benefits Achieved**:

    - **Better API Clarity**: Parameter names now clearly indicate their purpose in eigenvalue computation
    - **Cleaner Code**: Removed unused parameter that could confuse users
    - **No Breaking Changes to Functionality**: Algorithm behavior remains exactly the same
    - **Improved Maintainability**: More descriptive parameter names make the code easier to understand

  - **Verification Results**:
    - **All Tests Pass**: 3 unit tests + 1 doc test successful
    - **Clean Compilation**: No warnings with cargo clippy
    - **No Regressions**: Algorithm behavior maintained with improved API clarity

- **Omega Algorithm ndarray Migration (2025-06-04)**

  - **Complete Vec to ndarray Migration**: Converted all vector and matrix operations from Vec to ndarray for better performance and consistency with other layout algorithms

  - **Core Type Changes**:

    - **eigenvalue.rs**:

      - Return types: `(Vec<S>, Vec<Vec<S>>)` → `(Array1<S>, Array2<S>)`
      - eigenvalues: `Vec<S>` → `Array1<S>` (1D array)
      - eigenvectors: `Vec<Vec<S>>` → `Array2<S>` (2D matrix where each column is an eigenvector)
      - found_eigenvalues and found_eigenvectors managed as ndarray from the start

    - **omega.rs**:
      - coordinates: `Vec<Vec<S>>` → `Array2<S>` (where each row is a node's d-dimensional coordinate)
      - Distance computation optimized with `Zip::from` for element-wise operations

  - **Implementation Improvements**:

    - **Zero Eigenvalue Handling**: Eliminated need for skip processing by directly computing n_target + 1 eigenvalues
    - **First Eigenvector**: Initialize with constant vector (1,1,...,1)^T/√n in first column
    - **Function Simplification**: Consolidated function names (removed `_array1` suffixes)
    - **Memory Efficiency**: Direct ndarray operations without intermediate Vec conversions

  - **API Refinements**:

    - **compute_smallest_eigenvalues_with_laplacian**: Returns (Array1<S>, Array2<S>) with n_target + 1 elements
    - **gram_schmidt_orthogonalize**: Works with ArrayView2<S> for efficient column operations
    - **euclidean_distance**: Optimized with Zip operations for element-wise computation
    - **solve_with_conjugate_gradient**: Direct Array1 input/output

  - **Files Updated**:

    - **Cargo.toml**: Added `ndarray = "0.15"`, removed unused `nalgebra`
    - **eigenvalue.rs**: Complete function rewrite for ndarray compatibility
    - **omega.rs**: Coordinate handling and distance computation updated
    - **lib.rs**: Test adjustments for ndarray types (n_target + 1 expectation)

  - **Benefits Achieved**:

    - **Performance**: ndarray's optimized numerical operations
    - **Memory Efficiency**: Better memory layout and cache efficiency
    - **Project Consistency**: Alignment with other layout algorithms (MDS, SGD, Stress Majorization)
    - **Type Safety**: Improved compile-time type checking
    - **Maintainability**: Cleaner code with fewer type conversions

  - **Verification Results**:
    - **All Tests Pass**: 3 unit tests + 1 doc test successful
    - **Workspace Build**: Complete workspace compilation successful
    - **No Regressions**: Algorithm behavior maintained with improved performance
    - **Memory Usage**: More efficient memory patterns with ndarray

- **OmegaBuilder Pattern Implementation (2025-06-04)**

  - [Previous implementation details maintained...]

- **EigenSolver Refactoring to Function-Based API (2025-06-04)**

  - **Architectural Transformation**: Completely refactored EigenSolver from struct-based OOP design to functional programming approach using plain functions

  - **Struct Elimination**:

    - **Removed**: `EigenSolver<S>` struct and all associated instance methods
    - **Converted**: All functionality to standalone public functions
    - **Maintained**: `LaplacianStructure<S>` as it represents cached data, not behavior

  - **New Function-Based API**:

    ```rust
    // Main eigenvalue computation functions
    pub fn compute_smallest_eigenvalues_with_laplacian<S, R>(
        laplacian: &LaplacianStructure<S>,
        n_target: usize,
        max_iterations: usize,
        cg_max_iterations: usize,
        tolerance: S,
        cg_tolerance: S,
        vector_tolerance: S,
        rng: &mut R,
    ) -> (Vec<S>, Vec<Vec<S>>)

    pub fn compute_smallest_eigenvalues<G, S>(
        graph: G,
        n_target: usize
    ) -> (Vec<S>, Vec<Vec<S>>)

    // Helper functions
    pub fn generate_random_vector<S, R>(n: usize, rng: &mut R) -> Vec<S>
    pub fn gram_schmidt_orthogonalize<S>(vector: &mut [S], known_vectors: &[Vec<S>])
    pub fn solve_with_conjugate_gradient<S>(...) -> Vec<S>
    pub fn dot_product<S>(a: &[S], b: &[S]) -> S
    pub fn normalize<S>(vector: &mut [S])
    ```

  - **Updated Integration Points**:

    - **omega.rs**: Modified `compute_spectral_coordinates_with_weights()` to call `compute_smallest_eigenvalues_with_laplacian()` directly
    - **lib.rs**: Updated public exports and tests to use new function-based API
    - **Type annotations**: Fixed generic type inference issues in tests

  - **Code Quality Improvements**:

    - **Clippy Compliance**: Fixed all assignment operation patterns (`a = a + b` → `a += b`)
    - **Function Signatures**: Improved parameter types (`&mut Vec<S>` → `&mut [S]`)
    - **Warning Suppression**: Added `#[allow(clippy::too_many_arguments)]` for unavoidable case
    - **Clean Compilation**: Zero warnings with `cargo clippy --all-targets --all-features -- -D warnings`

  - **Benefits Achieved**:

    - **Functional Programming Style**: Pure functions without state management
    - **Increased Flexibility**: No need to instantiate struct objects
    - **Easier Testing**: Each function can be independently tested
    - **Explicit Dependencies**: All parameters clearly visible in function signatures
    - **Simpler API**: Direct function calls instead of object creation and method invocation

  - **Verification Results**:
    - **Tests**: All 3 unit tests + 1 doc test continue to pass
    - **Functionality**: Identical eigenvalue computation behavior maintained
    - **Performance**: No regression in computational efficiency
    - **Compatibility**: All existing usage patterns continue to work

- **Omega Algorithm Complete Refactoring and Enhancement (2025-01-06)**

  - **Comprehensive Implementation Overhaul**: Completely redesigned the Omega algorithm implementation to address all identified issues

  - **Edge Length Integration**:

    - **Issue Fixed**: Edge lengths were previously ignored (`_length` parameter unused)
    - **Solution**: Implemented weighted Laplacian using `LaplacianStructure::new(graph, length)`
    - **Impact**: Proper edge weight support enables more accurate graph representation

  - **Efficient Laplacian Operations**:

    - **Issue Fixed**: Graph Laplacian rebuilt from scratch every time, causing redundant computations
    - **Solution**: Created `LaplacianStructure<S>` that pre-computes and caches:
      - Edge list with weights: `Vec<(usize, usize, S)>`
      - Node degrees: `Vec<S>`
      - Node count: `usize`
    - **Impact**: Eliminates repeated computations during eigenvalue iteration

  - **Optimized Quadratic Form Computation**:

    - **Issue Fixed**: Used inefficient full matrix-vector multiplication for Rayleigh quotient
    - **Solution**: Implemented `LaplacianStructure::quadratic_form()` with O(|E|) complexity
    - **Formula**: `x^T L x = Σ_{(i,j) ∈ E} weight[i,j] * (x[i] - x[j])^2`
    - **Impact**: Significant performance improvement from O(|V|²) to O(|E|)

  - **Configurable EigenSolver Parameters**:

    - **Issue Fixed**: EigenSolver parameters were hardcoded and not customizable
    - **Solution**: Implemented `OmegaOption<S>` with Builder pattern supporting:
      - `d`: Number of spectral dimensions
      - `k`: Number of random pairs per node
      - `min_dist`: Minimum distance between node pairs
      - `max_iterations`: Maximum iterations for inverse power method
      - `cg_max_iterations`: Maximum iterations for CG method
      - `tolerance`: Convergence tolerance for eigenvalues
      - `cg_tolerance`: Convergence tolerance for CG method
      - `vector_tolerance`: Convergence tolerance for eigenvectors
    - **Impact**: Full control over algorithm parameters and performance trade-offs

  - **Proper Random Vector Generation**:

    - **Issue Fixed**: Used deterministic sine function instead of actual randomness
    - **Solution**: Uses actual RNG: `rng.gen_range(-1.0..1.0)` while maintaining reproducibility
    - **Impact**: True randomness improves algorithm quality while preserving deterministic results with seeded RNG

  - **Static Function Refactoring**:

    - **Issue Fixed**: Utility functions unnecessarily implemented as instance methods
    - **Solution**: Converted to static associated functions:
      - `euclidean_distance()`, `generate_random_vector()`, `gram_schmidt_orthogonalize()`
      - `dot_product()`, `normalize()`
    - **Impact**: Better code organization and clearer intent

  - **New Builder Pattern API**:

    ```rust
    // Before: 6 individual parameters
    Omega::new(graph, length, d, k, min_dist, rng)

    // After: Clean options-based API
    let options = OmegaOption::new()
        .d(3).k(50).min_dist(1e-2)
        .max_iterations(2000).tolerance(1e-5);
    Omega::new(graph, length, options, rng)
    ```

  - **Enhanced Eigenvalue Implementation**:

    - **LaplacianStructure Integration**: Eigenvalue solver now uses precomputed Laplacian structure
    - **Optimized Methods**: `compute_smallest_eigenvalues_with_laplacian()` for direct structure usage
    - **Performance**: Maintains O(d(|V| + |E|) + k|V|) computational complexity with significant constant factor improvements

  - **Updated CLI and Testing**:

    - **CLI Binary**: Updated `crates/cli/src/bin/omega.rs` to use new OmegaOption API
    - **Comprehensive Testing**: All tests updated and passing with new API
    - **Documentation**: Complete API documentation with Builder pattern examples
    - **Exports**: Added `OmegaOption` and `LaplacianStructure` to public API

  - **Code Quality Improvements**:
    - **No Compilation Warnings**: Clean compilation with no dead code
    - **Type Safety**: Builder pattern prevents parameter confusion
    - **Extensibility**: Easy to add new configuration options
    - **Memory Efficiency**: More efficient storage of graph topology

- **Rectangle Overlap Algorithm Refactoring**

  - Completely refactored implementation with separate `rectangle_overlap_2d.rs` module
  - Split algorithm into distinct X and Y dimension-specific implementations following WebCola's approach
  - Replaced external RBTree with Rust's BTreeSet for better efficiency and maintainability
  - Improved event handling in sweep line algorithm to fix infinite loop issues
  - Enhanced neighbor finding logic to match WebCola's implementation more precisely
  - Added thorough unit tests for each component of the algorithm
  - Maintained backward compatibility through legacy function in original module

- **Complete Python Bindings Implementation**

  - Added Python bindings for triangulation algorithm with comprehensive tests
  - Implemented separation constraints with rectangle overlap prevention
  - Added layering algorithms with cycle detection and removal
  - Completed community detection bindings with graph coarsening
  - Consistent naming and documentation style across all Python modules

- **WebAssembly Module Enhancement**

  - Comprehensive test coverage with individual test files for each algorithm
  - Memory-safe type conversions between Rust and JavaScript
  - JSDoc-style documentation for better developer experience
  - Method chaining support where appropriate

- **Quality Improvements**

  - Fixed infinite loop issues in multiple algorithms
  - Enhanced convergence criteria for iterative algorithms
  - Improved error handling across language boundaries
  - Performance optimizations for large graph processing

## Next Steps

Given the mature state of the project, focus areas include:

1. **Documentation Enhancement**

   - Complete API documentation with examples
   - Tutorial content for new users
   - Best practice guides for algorithm selection

2. **Performance Benchmarking**

   - Systematic performance evaluation across algorithms
   - Comparison with other graph libraries
   - Optimization opportunities identification

3. **Community Engagement**
   - Example applications and use cases
   - Integration guides for different frameworks
   - User feedback incorporation

## Active Decisions and Considerations

- **API Stability**: Maintaining backward compatibility while allowing for improvements
- **Performance vs. Flexibility**: Balancing generic interfaces with performance requirements
- **Cross-Language Consistency**: Ensuring similar behavior across Rust, Python, and JavaScript
- **Memory Management**: Careful handling of large graphs, especially in WebAssembly context
- **Design Paradigms**: Choosing between object-oriented and functional programming approaches based on use case (data structures vs. algorithms)

## Important Patterns and Preferences

- **Trait-Based Design**: Unified interfaces for algorithm families (CommunityDetection, LayeringAlgorithm)
- **Builder Pattern**: Configurable construction of complex algorithms
- **Functional Programming**: Pure functions for stateless computations (eigenvalue algorithms)
- **Error Handling**: Explicit error handling with proper conversion across language boundaries
- **Modular Architecture**: Specialized crates for focused functionality
- **Testing Strategy**: Comprehensive coverage including cross-language validation

## Workflow Corrections and Guidelines

### Corrected Commit Message Rules (Updated 2025-01-06)

**CRITICAL**: The following workflow must be followed for ALL tasks without exception:

#### Conventional Commits Format

All commit messages must follow this exact format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**: feat, fix, docs, style, refactor, perf, test, chore

**Scoping Rules**:

- **Crate-specific changes**: Use crate name as scope (e.g., `petgraph-layout-omega`, `egraph-wasm`, `petgraph-layout-mds`)
- **Project-wide changes**: Omit scope entirely (e.g., root configuration files, memory-bank updates, workspace-level changes)

**Examples**:

- `feat(petgraph-layout-omega): add min_dist parameter for numerical stability`
- `fix(egraph-wasm): resolve NaN values in ClassicalMds implementation`
- `test(petgraph-layout-sgd): add comprehensive tests for schedulers`
- `docs(petgraph-drawing): improve API documentation`
- `refactor(petgraph-algorithm-shortest-path): optimize distance calculation`
- `docs: update project workflow guidelines` (no scope for project-wide)
- `chore: update workspace dependencies` (no scope for workspace-level)

#### Mandatory Final Confirmation Process

**This step must NEVER be skipped under any circumstances:**

1. **Run all required checks**:

   - `cargo fmt --all`
   - `cargo clippy --workspace --all-targets --all-features -- -D warnings`
   - `cargo test --workspace` (or appropriate tests for the changes)

2. **Prepare comprehensive summary**:

   - List all files modified
   - Describe all changes made
   - Explain the impact and benefits

3. **Create proper commit message**:

   - Follow Conventional Commits format exactly
   - Use correct scope based on crates affected
   - Provide clear, concise description

4. **Present to user for approval**:

   - Show complete summary of changes
   - Present the proposed commit message
   - Wait for explicit user approval
   - Do not proceed without confirmation

5. **Only after user approval**: Mark task as complete

#### Scope Determination Guide

- **Single crate affected**: Use that crate's name as scope
- **Multiple crates in same domain**: Use the primary crate name or most relevant scope
- **Cross-cutting changes**: Choose the most impacted crate or omit scope if truly project-wide
- **Documentation/configuration**: Use specific crate if crate-specific, omit scope if project-wide
- **Memory bank updates**: No scope (project-wide documentation)
- **Root-level files**: No scope (project-wide)

### Workflow Enforcement

This corrected workflow addresses previous issues where:

- Commit messages didn't consistently follow Conventional Commits format
- Scoping rules weren't properly applied
- Final confirmation step was sometimes skipped
- Commit message proposals weren't always provided

**All future tasks must follow this corrected workflow without exception.**

## Learnings and Project Insights

- **Rust-First Design**: Starting with Rust implementation ensures memory safety and performance
- **Language Binding Patterns**: PyO3 and wasm-bindgen provide excellent foundation for cross-language APIs
- **Algorithm Implementation**: Many graph algorithms benefit from trait-based generic implementations
- **Performance Considerations**: External dependencies should be carefully evaluated (RBTree → BTreeSet transition)
- **Testing Importance**: Cross-language testing reveals subtle implementation differences
- **Documentation Value**: Good documentation significantly improves adoption and usability
- **Workflow Discipline**: Consistent adherence to commit message conventions and confirmation processes is essential for project quality
- **Design Philosophy**: Data structures (LaplacianStructure) should be structs, while stateless computations (eigenvalue algorithms) benefit from functional programming approaches
