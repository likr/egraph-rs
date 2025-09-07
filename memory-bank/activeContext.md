# Active Context: egraph-rs

## Current Work Focus

The project has reached a mature state with comprehensive functionality across multiple domains. Current development appears focused on SGD algorithm refinements and cross-language binding optimizations:

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

- **WeightedEdgeLength Algorithm Implementation (2025-09-08)**

  - **Complete Rust Implementation**: Added WeightedEdgeLength algorithm to petgraph-algorithm-shortest-path crate with degree-based edge weight calculation

  - **Core Algorithm Features**:

    - **Weight Calculation Formula**: `degree(u) + degree(v) - 2 * common_neighbors` where u and v are edge endpoints
    - **Pre-computed Neighbor Sets**: Efficient HashSet-based neighbor storage for fast common neighbor calculation
    - **Edge Endpoints Caching**: Pre-computed edge endpoint pairs for efficient edge index-based weight lookup
    - **Optimization**: Orders nodes by degree size to minimize iteration in common neighbor counting

  - **Rust Implementation Details**:

    - **`crates/algorithm/shortest-path/src/weighted_edge_length.rs`**: Core implementation with WeightedEdgeLength struct
      - **`new(graph)`**: Constructor that pre-computes neighbor sets and edge endpoints from graph structure
      - **`edge_weight(edge_index)`**: Main method that calculates weight for given edge index
      - **Undirected Graph Support**: Treats all edges as undirected for neighbor relationship building
      - **Efficient Storage**: Vec<HashSet<usize>> for neighbors, Vec<(usize, usize)> for edge endpoints
    - **`crates/algorithm/shortest-path/src/lib.rs`**: Module integration and public exports
    - **Comprehensive Testing**: Unit tests for simple graphs, triangle graphs, and edge weight validation

  - **Python Bindings Implementation**:

    - **`crates/python/src/algorithm/shortest_path.rs`**: Complete Python wrapper with PyO3
      - **`PyWeightedEdgeLength` Class**: Python wrapper maintaining API compatibility with original Python implementation
      - **`__call__(edge_index)`**: Callable interface matching original Python WeightedEdgeLength class
      - **Constructor**: `WeightedEdgeLength(graph)` accepting PyGraphAdapter instances
      - **Graph Type Support**: Currently supports undirected graphs (Graph type) with DiGraph error handling
    - **Module Registration**: Integrated with existing shortest path module registration
    - **API Compatibility**: Drop-in replacement for original Python implementation

  - **Usage Pattern Compatibility**:

    ```python
    # Original Python usage pattern maintained
    import egraph as eg

    # Create graph and WeightedEdgeLength calculator
    weight_calc = eg.WeightedEdgeLength(graph)

    # Use with SGD exactly like original implementation
    sgd = eg.FullSgd().build(graph, weight_calc)
    ```

  - **Algorithm Optimization**:

    - **Pre-computation Strategy**: One-time neighbor set building during construction for O(1) lookups
    - **Memory Efficiency**: Edge endpoints stored only once (node_index < target_index) to avoid duplication
    - **Performance**: Common neighbor counting optimized by iterating over smaller neighbor set
    - **Complexity**: O(E + V) construction time, O(min(degree(u), degree(v))) per edge weight calculation

  - **Test Coverage**:

    - **Rust Tests**: `test_simple_graph()`, `test_triangle_graph()` with mathematical verification
    - **Python Tests**: `crates/python/tests/test_weighted_edge_length.py` with unittest framework
      - **Simple Graph Testing**: Validates degree calculation and weight formula
      - **Triangle Graph Testing**: Verifies common neighbor impact on weights
      - **SGD Integration Testing**: Confirms usage with FullSgd().build() pattern
      - **Callable Interface Testing**: Validates Python **call** method functionality

  - **Implementation Benefits**:

    - **High Performance**: Rust implementation with efficient data structures and algorithms
    - **Memory Safety**: Rust's ownership system prevents memory-related bugs
    - **API Consistency**: Maintains exact same interface as original Python implementation
    - **Zero Dependencies**: Uses only standard Rust collections (HashSet, Vec) for maximum compatibility
    - **Edge Case Handling**: Proper bounds checking and error handling for invalid edge indices

  - **Files Created/Modified**:

    - **`crates/algorithm/shortest-path/src/weighted_edge_length.rs`**: New core implementation file
    - **`crates/algorithm/shortest-path/src/lib.rs`**: Module integration and exports
    - **`crates/python/src/algorithm/shortest_path.rs`**: Python bindings addition to existing file
    - **`crates/python/tests/test_weighted_edge_length.py`**: New comprehensive test suite

  - **Quality Assurance Results**:

    - **Rust Tests**: All unit tests pass with correct mathematical validation
    - **Python Tests**: All unittest cases pass with comprehensive coverage
    - **Integration**: Successfully builds and integrates with existing SGD workflow
    - **API Compatibility**: Drop-in replacement confirmed with test usage patterns
    - **Performance**: Efficient implementation suitable for large graphs

  - **Mathematical Verification**:

    - **Simple Graph**: Path graph 0-1-2 produces weights [3, 3] (degrees [1,2,1], no common neighbors)
    - **Triangle Graph**: Complete triangle produces weights [2, 2, 2] (degrees [2,2,2], 1 common neighbor each)
    - **Formula Correctness**: `degree(u) + degree(v) - 2 * common_neighbors` properly implemented
    - **Edge Case Handling**: Proper behavior for isolated nodes and disconnected components

  - **Integration Status**:
    - ✅ **Core Algorithm**: Complete Rust implementation with efficient data structures
    - ✅ **Python Bindings**: Full PyO3 wrapper with callable interface
    - ✅ **SGD Integration**: Confirmed compatibility with FullSgd().build() usage pattern
    - ✅ **Testing**: Comprehensive test coverage for both Rust and Python
    - ✅ **Documentation**: Complete API documentation with usage examples
    - ✅ **Module Integration**: Properly exported and registered in both language bindings

- **Numpy Integration and PyO3 0.26 Upgrade (2025-09-07)**

  - **Complete Numpy Integration for Array Types**: Added comprehensive numpy support for PyArray1 and PyArray2 with bidirectional conversion capabilities

  - **New Array Constructor Features**:

    - **PyArray1 Constructor**: Optional numpy array parameter `PyArray1(array=None)` for seamless integration
    - **PyArray2 Constructor**: Optional numpy array parameter `PyArray2(array=None)` for matrix operations
    - **from_numpy() Class Methods**: Direct conversion from numpy arrays to custom array types
    - **to_numpy() Methods**: Convert custom arrays back to numpy arrays for external library compatibility
    - **Rust-numpy Integration**: Added `numpy = "0.26"` dependency with PyReadonlyArray1/PyReadonlyArray2 support

  - **DrawingEuclidean2d Enhancement**:

    - **from_array2 Static Method**: Create 2D Euclidean drawings directly from coordinate arrays
    - **Direct Integration**: `DrawingEuclidean2d.from_array2(graph, coordinates)` for streamlined workflow
    - **Type Safety**: Proper PyArray2 integration with coordinate validation

  - **PyO3 Framework Upgrade**:

    - **Version Upgrade**: PyO3 0.21 → 0.26 with breaking change migration
    - **Type Migration**: Fixed all `PyObject` → `Py<PyAny>` type changes across codebase
    - **API Migration**: Updated all `Python::with_gil` → `Python::attach` deprecation warnings
    - **Comprehensive Coverage**: Fixed deprecations in quality_metrics.rs, layout/sgd/sgd.rs, graph_base.rs, triangulation.rs
    - **Warning Resolution**: Eliminated all 11 compilation warnings for clean build

  - **Test Suite Enhancement**:

    - **Format Migration**: Converted from pytest to unittest format as requested
    - **Comprehensive Coverage**: test_numpy_integration.py with full feature testing
    - **Error Handling**: Proper exception testing for edge cases and invalid inputs
    - **Integration Testing**: Complete workflow validation from array creation to drawing generation

  - **Implementation Details**:

    - **`crates/python/src/array.rs`**: Enhanced with numpy imports and conversion methods

      ```rust
      use numpy::{PyArray1 as NumpyArray1, PyArray2 as NumpyArray2, PyReadonlyArray1, PyReadonlyArray2};

      #[new]
      #[pyo3(signature = (array=None))]
      fn new_py(array: Option<PyReadonlyArray1<FloatType>>) -> PyResult<Self>

      #[classmethod]
      fn from_numpy(_cls: &Bound<PyType>, array: PyReadonlyArray1<FloatType>) -> Self

      fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, NumpyArray1<FloatType>>
      ```

    - **`crates/python/src/drawing/drawing_euclidean_2d.rs`**: Added numpy integration

      ```rust
      #[staticmethod]
      pub fn from_array2(graph: &PyGraphAdapter, coordinates: &PyArray2) -> PyResult<Py<PyAny>>
      ```

    - **`crates/python/Cargo.toml`**: Updated dependencies

      ```toml
      pyo3 = "0.26"
      numpy = "0.26"
      ```

    - **`crates/python/tests/test_numpy_integration.py`**: Comprehensive unittest suite
      ```python
      class TestNumpyIntegration(unittest.TestCase):
          def test_array1_constructors(self)
          def test_array2_constructors(self)
          def test_numpy_conversion_methods(self)
          def test_drawing_from_array2(self)
          def test_error_handling(self)
      ```

  - **PyO3 Migration Scope**:

    - **Type Updates**: 16+ instances of PyObject → Py<PyAny> across drawing, layout, and quality modules
    - **API Updates**: 6+ instances of Python::with_gil → Python::attach deprecation fixes
    - **Parameter Updates**: Unused variable warnings fixed with underscore prefixes
    - **Return Type Updates**: All initial_placement methods updated for new PyO3 API

  - **Files Modified**:

    - **Core Implementation**: `crates/python/src/array.rs`, `crates/python/src/drawing/drawing_euclidean_2d.rs`
    - **PyO3 Migrations**: `crates/python/src/quality_metrics.rs`, `crates/python/src/layout/sgd/sgd.rs`, `crates/python/src/graph/graph_base.rs`, `crates/python/src/algorithm/triangulation.rs`
    - **Drawing Modules**: `crates/python/src/drawing/drawing_hyperbolic_2d.rs`, `crates/python/src/drawing/drawing_spherical_2d.rs`, `crates/python/src/drawing/drawing_torus_2d.rs`, `crates/python/src/drawing/drawing_base.rs`
    - **Layout Modules**: `crates/python/src/layout/mds.rs`
    - **Configuration**: `crates/python/Cargo.toml`
    - **Tests**: `crates/python/tests/test_numpy_integration.py`

  - **API Usage Examples**:

    ```python
    # Direct numpy integration
    import numpy as np
    import egraph as eg

    # Create arrays from numpy
    np_array = np.array([1.0, 2.0, 3.0])
    arr1 = eg.Array1(np_array)
    arr2 = eg.Array1.from_numpy(np_array)

    # Convert back to numpy
    result = arr1.to_numpy()

    # Create drawings from coordinate arrays
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    drawing = eg.DrawingEuclidean2d.from_array2(graph, eg.Array2(coords))
    ```

  - **Quality Assurance Results**:

    - **Clean Compilation**: `cargo check` completed with zero warnings
    - **Test Success**: All unittest cases pass with comprehensive coverage
    - **API Consistency**: Numpy integration follows established patterns
    - **Backward Compatibility**: Existing array functionality unchanged
    - **Type Safety**: Proper error handling for invalid inputs

  - **Benefits Achieved**:

    - **Seamless Integration**: Direct numpy array support for scientific computing workflows
    - **Modern PyO3**: Up-to-date Python bindings with latest performance improvements
    - **Clean Codebase**: Zero compilation warnings with modern Rust patterns
    - **Enhanced Usability**: Simplified coordinate handling for visualization applications
    - **Future-Proof**: Latest PyO3 version ensures compatibility with future Python releases

  - **Integration Status**:
    - ✅ **Array Constructors**: PyArray1/PyArray2 with optional numpy parameters
    - ✅ **Numpy Conversion**: Bidirectional conversion methods (from_numpy/to_numpy)
    - ✅ **Drawing Integration**: DrawingEuclidean2d.from_array2 static method
    - ✅ **PyO3 Migration**: Complete upgrade to 0.26 with all warnings resolved
    - ✅ **Test Coverage**: Comprehensive unittest suite with error handling
    - ✅ **Documentation**: Complete API documentation with usage examples

- **Omega Algorithm Enhancement with Custom Python Array Types (2025-09-03)**

  - **Complete Omega API Extension**: Added three new methods to Omega struct for enhanced spectral embedding functionality and improved Python bindings with custom array types

  - **New Omega Methods**:

    - **`embedding(graph, length, rng)`**: Computes spectral coordinates using `compute_spectral_coordinates`
    - **`embedding_and_eigenvalues(graph, length, rng)`**: Computes both embedding and eigenvalues using `compute_spectral_coordinates_and_eigenvalues`
    - **`build_with_embedding(graph, embedding, rng)`**: Creates SGD instance from precomputed embedding using `compute_omega_node_pairs`

  - **Core Function Refactoring**:

    - **`compute_omega_node_pairs`**: Now accepts embedding and individual parameters (min_dist, k) instead of full Omega struct
    - **`compute_spectral_coordinates`**: Now accepts individual parameters (shift, eigenvalue_max_iterations, cg_max_iterations, eigenvalue_tolerance, cg_tolerance) instead of SGD instance
    - **`compute_spectral_coordinates_and_eigenvalues`**: New function that returns both coordinates and eigenvalues
    - **Zero Eigenvalue Exclusion**: Modified to exclude zero eigenvalue and return only d non-zero eigenvalues and eigenvectors

  - **Custom Python Array Implementation**:

    - **`crates/python/src/array.rs`**: Created new module with custom wrapper classes
      - **`PyArray1`**: Wrapper for ndarray::Array1<FloatType> with Python indexing, iteration, and shape access
      - **`PyArray2`**: Wrapper for ndarray::Array2<FloatType> with Python indexing, row/column access, and shape methods
      - **Numpy Independence**: Complete elimination of numpy dependency in favor of custom types
    - **`crates/python/src/lib.rs`**: Integrated array module with `array::register(py, m)?`

  - **Updated Python Bindings**:

    - **`crates/python/src/layout/sgd/omega.rs`**: Complete overhaul of Python interface
      - **`embedding()`**: Returns `PyArray2` instead of numpy array with f32->f64 conversion
      - **`embedding_and_eigenvalues()`**: Returns `(PyArray2, PyArray1)` tuple instead of numpy arrays
      - **`build_with_embedding()`**: Accepts `PyArray2` instead of numpy array for embedding parameter
      - **Type Conversion**: Proper handling of f32 (Rust Omega) to f64 (Python FloatType) conversion using `mapv(|v| v as FloatType)`

  - **Enhanced Eigenvalue Computation**:

    - **`crates/layout/omega/src/eigenvalue.rs`**: Modified `compute_spectral_coordinates_and_eigenvalues`
      - **Zero Eigenvalue Exclusion**: Skip index 0 (zero eigenvalue) and extract only non-zero eigenvalues
      - **Proper Array Sizing**: Returns Array1<S> of size d and Array2<S> of size (n, d)
      - **Coordinate Calculation**: Divide eigenvectors by sqrt of eigenvalues for proper spectral coordinates
      - **API Consistency**: Both functions now work with the same underlying eigenvalue computation

  - **Implementation Benefits**:

    - **API Flexibility**: Users can now compute embeddings separately from SGD instance creation
    - **Performance**: Precomputed embeddings can be reused across multiple SGD instances
    - **Python Independence**: No external numpy dependency reduces installation complexity
    - **Type Safety**: Custom array types provide better Rust-Python type integration
    - **Mathematical Correctness**: Proper exclusion of zero eigenvalue aligns with spectral graph theory

  - **Files Modified**:

    - **`crates/layout/omega/src/omega.rs`**: Added three new methods and updated internal method calls
    - **`crates/layout/omega/src/eigenvalue.rs`**: Enhanced eigenvalue computation with zero exclusion
    - **`crates/python/src/array.rs`**: New custom array wrapper implementation
    - **`crates/python/src/layout/sgd/omega.rs`**: Complete Python binding overhaul
    - **`crates/python/src/lib.rs`**: Array module integration

  - **API Usage Examples**:

    ```python
    # Separate embedding computation
    omega = eg.Omega().d(2).k(30)
    embedding = omega.embedding(graph, lambda i: 1.0, rng)
    eigenvals = omega.embedding_and_eigenvalues(graph, lambda i: 1.0, rng)

    # Build SGD from precomputed embedding
    sgd = omega.build_with_embedding(graph, embedding, rng)

    # Custom array types
    coords = embedding  # PyArray2 with shape (n_nodes, d)
    values = eigenvals[1]  # PyArray1 with d eigenvalues
    ```

  - **Quality Assurance**:

    - **Compilation Success**: All workspace builds pass without warnings
    - **Type Safety**: Proper f32/f64 conversion and array bounds checking
    - **Memory Safety**: Custom arrays manage ndarray lifecycle properly
    - **API Consistency**: New methods follow existing Omega patterns
    - **Mathematical Correctness**: Eigenvalue computation excludes zero eigenvalue as required

  - **Breaking Changes**:
    - **Function Signatures**: `compute_omega_node_pairs` and `compute_spectral_coordinates` now accept individual parameters
    - **Return Types**: `compute_spectral_coordinates_and_eigenvalues` returns smaller arrays (d elements instead of d+1)
    - **Python Types**: Omega methods return custom PyArray types instead of numpy arrays

- **SGD Direct Constructor Implementation (2025-06-30)**

  - **Python SGD Constructor Addition**: Implemented direct constructor for `PySgd` class allowing users to create SGD instances with custom node pairs without requiring builder patterns

  - **Implementation Details**:

    - **`crates/python/src/layout/sgd/sgd.rs`**: Added `#[new]` constructor to `PySgd` class
      - **`#[new]` method**: Direct constructor accepting node pairs and epsilon parameter
      - **Parameter validation**: Simplified approach without extensive input validation (user's preference)
      - **Signature**: `fn new(node_pairs: Vec<(usize, usize, f32, f32, f32, f32)>, epsilon: f32) -> PyResult<Self>`
      - **Default epsilon**: 0.1 (matches other SGD implementations in the project)
      - **Error handling**: Returns `PyResult<Self>` for proper Python error propagation
    - **`crates/python/tests/test_sgd.py`**: Added comprehensive test case `test_sgd_constructor`
      - **Custom node pairs testing**: Creates simple 3-node triangle with equal distances and weights
      - **Default epsilon testing**: Verifies constructor works with default epsilon parameter
      - **Custom epsilon testing**: Verifies constructor works with user-specified epsilon
      - **Integration testing**: Full SGD workflow with schedulers and drawing updates
      - **Cross-scheduler validation**: Tests with all 5 scheduler types (Constant, Linear, Quadratic, Exponential, Reciprocal)

  - **API Features Implemented**:

    ```python
    # Basic usage with default epsilon
    node_pairs = [(0, 1, 1.0, 1.0, 1.0, 1.0), (0, 2, 1.0, 1.0, 1.0, 1.0), (1, 2, 1.0, 1.0, 1.0, 1.0)]
    sgd = eg.Sgd(node_pairs)

    # Custom epsilon
    sgd_custom = eg.Sgd(node_pairs, epsilon=0.05)

    # Full SGD workflow
    drawing = eg.DrawingEuclidean2d.initial_placement(graph)
    rng = eg.Rng.seed_from(42)
    scheduler = eg.SchedulerExponential(10)

    def step(eta):
        sgd.shuffle(rng)
        sgd.apply(drawing, eta)

    scheduler.run(step)
    ```

  - **Design Decisions**:

    - **Simplified validation**: User requested no extensive input validation for performance and simplicity
    - **Builder pattern avoided**: User specifically requested no builder pattern implementation
    - **Consistent API**: Follows same method signatures as existing SGD classes (shuffle, apply, update_distance, update_weight)
    - **Parameter format**: Node pairs use 6-tuple format (i, j, dij, dji, wij, wji) matching Rust SGD::new() signature

  - **Benefits Achieved**:

    - **Direct access**: Users can now create SGD instances directly without requiring FullSgd or SparseSgd builders
    - **Custom workflows**: Enables advanced users to define custom node pair configurations
    - **API completeness**: Fills gap in Python bindings where core SGD class was only accessible through builders
    - **Performance**: Direct constructor eliminates intermediate builder steps for custom use cases
    - **Flexibility**: Allows integration with external distance computation algorithms

  - **Integration Status**:
    - ✅ **Core constructor**: Direct SGD creation with custom node pairs
    - ✅ **Test coverage**: Comprehensive test with all scheduler types
    - ✅ **Error handling**: Proper PyResult error propagation
    - ✅ **Documentation**: Complete docstring with parameter descriptions and usage examples
    - ✅ **Backward compatibility**: Existing builder-based workflows continue to work unchanged

- **Omega Python Bindings Implementation (2025-06-05)**

  - **Complete Python Bindings for Omega Algorithm**: Implemented comprehensive PyO3-based Python wrapper for the Omega spectral coordinates SGD layout algorithm

  - **Implementation Details**:

    - **`crates/python/Cargo.toml`**: Added `petgraph-layout-omega` dependency to enable Omega algorithm access
    - **`crates/python/src/layout/sgd/omega.rs`**: Complete Python wrapper implementation with 400+ lines of code
      - **`PyOmega` class**: Main algorithm wrapper with full SGD interface compatibility
      - **`PyOmegaBuilder` class**: Configurable builder pattern with fluent API and method chaining
      - **Seven configurable parameters**: d, k, min_dist, eigenvalue_max_iterations, cg_max_iterations, eigenvalue_tolerance, cg_tolerance
      - **Complete SGD integration**: shuffle(), apply(), update_distance(), update_weight() methods
      - **All scheduler types**: scheduler(), scheduler_constant(), scheduler_linear(), scheduler_quadratic(), scheduler_exponential(), scheduler_reciprocal()
      - **Multi-space drawing support**: Works with all drawing spaces (Euclidean2d, Euclidean, Hyperbolic2d, Spherical2d, Torus2d)
    - **`crates/python/src/layout/sgd/mod.rs`**: Registered Omega and OmegaBuilder classes in layout.sgd module
    - **`crates/python/tests/test_omega.py`**: Comprehensive test suite with 7 test cases covering all functionality
      - Basic Omega functionality with default parameters
      - OmegaBuilder custom configuration testing
      - Method chaining validation
      - All scheduler types verification
      - Weighted edges support
      - Complete layout process with scheduler integration
      - Distance and weight update functions testing
    - **`crates/layout/omega/src/omega.rs`**: Fixed clippy warning for needless_borrow

  - **API Features Implemented**:

    ```python
    # Basic usage
    omega = eg.Omega(graph, lambda edge_idx: 1.0, rng)

    # Advanced configuration with builder pattern
    omega = (eg.OmegaBuilder()
             .d(3)  # Spectral dimensions
             .k(50)  # Random pairs per node
             .min_dist(1e-3)  # Minimum distance
             .eigenvalue_max_iterations(1000)
             .cg_max_iterations(100)
             .eigenvalue_tolerance(1e-4)
             .cg_tolerance(1e-4)
             .build(graph, lambda edge_idx: 1.0, rng))

    # Layout execution
    drawing = eg.DrawingEuclidean2d.initial_placement(graph)
    scheduler = omega.scheduler_exponential(100, 0.1)
    def step(eta):
        omega.shuffle(rng)
        omega.apply(drawing, eta)
    scheduler.run(step)
    ```

  - **Quality Assurance Results**:

    - **Build Success**: `maturin develop` completed successfully without warnings
    - **All Tests Pass**: 7/7 tests successful in 0.014 seconds
    - **Code Quality**: cargo fmt and clippy checks completed without issues
    - **API Consistency**: Follows same patterns as other SGD algorithms in Python bindings
    - **Documentation**: Comprehensive docstrings with parameter descriptions and usage examples

  - **Benefits Achieved**:

    - **Complete Algorithm Access**: Omega algorithm now fully available from Python with all advanced configuration options
    - **Consistent Interface**: Same SGD interface as FullSgd, SparseSgd, and DistanceAdjustedSgd for seamless algorithm switching
    - **Advanced Configuration**: All 7 parameters configurable via builder pattern for research and production use
    - **Professional Quality**: Comprehensive testing and documentation ensure production readiness
    - **Cross-Language Consistency**: Python API mirrors Rust API design and functionality

  - **Integration Status**:
    - ✅ **Omega Algorithm**: Complete Python bindings with comprehensive testing
    - ✅ **Other SGD Variants**: FullSgd, SparseSgd, DistanceAdjustedSgd already available
    - ✅ **All Drawing Spaces**: Euclidean2d, Euclidean, Hyperbolic2d, Spherical2d, Torus2d support
    - ✅ **Scheduler Integration**: All five scheduler types (Constant, Linear, Quadratic, Exponential, Reciprocal)

- **Omega CLI Command-Line Options Enhancement (2025-06-05)**

  - **Comprehensive CLI Parameter Support**: Added complete command-line interface for all Omega algorithm parameters in the omega binary

  - **New Command-Line Options Added**:

    - **`--d <value>`**: Number of spectral dimensions (default: 2)
    - **`--k <value>`**: Number of random pairs per node (default: 30)
    - **`--min-dist <value>`**: Minimum distance between node pairs (default: 1e-3)
    - **`--eigenvalue-max-iterations <value>`**: Max iterations for eigenvalue computation (default: 1000)
    - **`--cg-max-iterations <value>`**: Max iterations for CG method (default: 100)
    - **`--eigenvalue-tolerance <value>`**: Convergence tolerance for eigenvalue computation (default: 1e-4)
    - **`--cg-tolerance <value>`**: Convergence tolerance for CG method (default: 1e-4)
    - **`--unit-edge-length <value>`**: Length value for all edges (default: 30.0)
    - **`--sgd-iterations <value>`**: Number of SGD iterations (default: 100)
    - **`--sgd-eps <value>`**: Final learning rate for SGD scheduler (default: 0.1)

  - **Implementation Details**:

    - **Parameter Structure**: Created `OmegaParams` struct with Default trait implementation for organized parameter management
    - **Argument Parsing**: Extended `parse_args()` function using argparse library with proper Option<T> handling
    - **Borrowing Solution**: Used scoped parser creation to avoid Rust borrowing conflicts with argparse
    - **Parameter Integration**: All options properly wired through to `OmegaBuilder` and SGD scheduler
    - **Documentation**: Updated help text, usage examples, and code documentation

  - **Files Modified**:

    - **`crates/cli/src/bin/omega.rs`**: Complete rewrite of argument parsing and parameter handling
      - Added `OmegaParams` struct with sensible defaults
      - Extended `parse_args()` with all 10 new command-line options
      - Updated `layout()` function to accept and use parameter struct
      - Enhanced documentation with usage examples

  - **Usage Examples**:

    ```bash
    # Basic usage (uses defaults)
    cargo run --bin omega -- input.json output.json

    # Custom parameters
    cargo run --bin omega -- input.json output.json --d 3 --k 50 --sgd-iterations 200

    # Fine-tuned eigenvalue computation
    cargo run --bin omega -- input.json output.json --eigenvalue-tolerance 1e-6 --cg-tolerance 1e-6
    ```

  - **Benefits Achieved**:

    - **Full Parameter Control**: Users can now customize every aspect of the Omega algorithm from command line
    - **Backward Compatibility**: Existing usage patterns continue to work with sensible defaults
    - **Professional CLI**: Help output shows all options with descriptions and defaults
    - **Research Flexibility**: Enables parameter tuning and experimentation without code changes
    - **Production Ready**: Configurable performance/quality trade-offs for different use cases

  - **Verification Results**:
    - **Clean Compilation**: No warnings with `cargo clippy --bin omega`
    - **All Tests Pass**: omega-specific tests continue to pass
    - **Help Output**: Professional help display with all options documented
    - **Build Success**: Binary builds and runs successfully with all options

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

## Current Development Patterns (Based on Open Files)

The numerous SGD-related files currently open suggest active refinement and optimization work:

- **SGD Framework Evolution**: Recent architectural changes from trait-based to concrete implementation
- **Scheduler System**: Comprehensive development of learning rate scheduling with five different strategies
- **Cross-Language Integration**: Parallel development across Rust core, Python bindings, and WebAssembly interfaces
- **Algorithm Variants**: Unified approach supporting Full, Sparse, Distance-Adjusted, and Omega SGD variants
- **Testing and Validation**: Extensive test coverage across all language bindings to ensure behavioral consistency

This indicates ongoing efforts to:

1. **Simplify and optimize** the SGD framework architecture
2. **Enhance performance** through better learning rate management
3. **Maintain consistency** across all language bindings
4. **Improve usability** with better scheduler integration and documentation

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
