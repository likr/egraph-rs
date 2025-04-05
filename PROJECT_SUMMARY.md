# egraph-rs Project Summary

This document provides an overview of the egraph-rs project structure, components, and development processes.

## Project Configuration

- This project is a Rust workspace (`Cargo.toml`) comprising multiple crates (`crates/*`).
- JavaScript/WASM-related code is managed in the `js/` directory, `crates/wasm` crate, and the root `package.json` (using npm workspaces).

## Development Tools

### Rust

- **Edition:** `2021` (verify in each crate's `Cargo.toml`)
- **Commands:**
  - Check: `cargo check --workspace`
  - Test: `cargo test --workspace`
  - Format: `cargo fmt --all`
  - Lint: `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- **Adding Dependencies:** Add to `crates/<target-crate>/Cargo.toml` in the `[dependencies]` section.

### JavaScript/TypeScript/JSON

- **Format:** Run `npx prettier --write .` (using `.prettierrc.json` settings, currently default)
- **WASM Build:** `npm run wasm-build`
- **Run Examples:** `npm start` (internally runs `npm run dev -w examples`)
- **Adding Dependencies:** Use `npm install <package-name> -w <workspace-name>` (e.g., `-w examples`, `-w crates/wasm`) or run `npm install <package-name>` at the root.

## Architecture Overview

### Workspace Root

- Location: `/home/likr/src/likr/egraph-rs`
- Structure: Monorepo containing Rust workspace and JavaScript/TypeScript code

### Rust Workspace Members (from Cargo.toml)

- **crates/algorithm**: Graph algorithms
  - `connected-components`: Connected components
  - `shortest-path`: Shortest path algorithms (BFS, Dijkstra, Warshall-Floyd, DistanceMatrix)
- **crates/cli**: CLI tools
  - `src/lib.rs`: Shared library code (JSON I/O, etc.)
  - `src/bin/quality-metrics.rs`: Binary to calculate drawing quality metrics
  - `src/bin/sgd.rs`: Binary to apply SGD layout algorithm
- **crates/clustering**: Clustering (Louvain algorithm step functions, graph coarsening functions)
- **crates/dataset**: Provides functionality to load graph datasets such as SuiteSparse Matrix Collection
  - `src/lib.rs`: Dataset loading functions (`load_graph`)
  - `src/data/*.csv`: Data files (edge list format)
- **crates/drawing**: Drawing-related
  - `src/lib.rs`: Basic trait definitions (`DrawingIndex`, `DrawingValue`) and module exports
  - `src/drawing.rs`: Basic abstraction for drawing (`Drawing` trait)
  - `src/drawing/*.rs`: Specific drawing implementations (euclidean, spherical, hyperbolic, torus, etc.)
  - `src/metric.rs`: Distance space trait definitions (`Delta`, `Metric`, `MetricCartesian`)
  - `src/metric/*.rs`: Specific distance calculation implementations (distances in various spaces, vector difference calculations, etc.)
- **crates/edge-bundling**: Edge bundling algorithms for graph visualization
  - `fdeb`: Force Directed Edge Bundling - Implementation of Holten & Van Wijk's algorithm (2009) for reducing visual clutter in graph visualizations by bundling similar edges together. Provides:
    - `Point`: 2D point structure with position and velocity
    - `EdgeBundlingOptions`: Configuration for the bundling algorithm parameters
    - `fdeb`: Main function to apply edge bundling to a graph
- **crates/layout**: Layout algorithms
  - `kamada-kawai`: Implementation of Kamada-Kawai force-directed layout algorithm
    - `KamadaKawai`: Primary struct that implements the algorithm
    - Models a graph as a spring system where spring lengths are based on shortest path distances
    - Iteratively positions nodes to minimize the energy of the spring system
    - Provides methods for node selection based on energy gradient and node position optimization
    - Reference: Kamada, T., & Kawai, S. (1989). An algorithm for drawing general undirected graphs.
  - `mds`: Multidimensional Scaling (MDS) implementation
    - Provides algorithms to visualize graph structures in lower dimensional spaces
    - `ClassicalMds`: Standard implementation that computes a full distance matrix
    - `PivotMds`: Efficient implementation that uses a subset of nodes as pivots
    - Uses eigendecomposition and double centering to transform distance matrices
    - Reference: Cox, T. F., & Cox, M. A. (2000). Multidimensional scaling. Chapman & Hall/CRC.
  - `overwrap-removal`: Overlap removal algorithms for graph layouts
    - `OverwrapRemoval`: A graph layout post-processing algorithm that resolves node overlaps
    - Iteratively adjusts node positions based on their defined radii to ensure proper spacing
    - Distributes displacement forces between nodes based on their relative sizes
    - Reference: Similar to Prism force-directed overlap removal technique
  - `separation-constraints`: Separation constraints for layout algorithms
    - Implements one-dimensional separation constraints of the form `v_left + gap <= v_right`
    - Based on the Quadratic Programming Separation Constraints (QPSC) algorithm from IPSEP-COLA
    - `Constraint`: Public struct representing a separation constraint between two variables
    - `ConstraintGraph`: Main class managing constraint satisfaction via block operations
    - Uses a gradient projection approach that enforces constraints while minimizing deviation from desired positions
    - Useful for enforcing minimum distances, hierarchical relationships, or avoiding node overlaps
    - Can be applied to layouts generated by other algorithms like stress majorization
    - Reference: Dwyer, T., Koren, Y., & Marriott, K. (2006). "IPSEP-COLA: An incremental procedure for separation constraint layout of graphs."
  - `sgd`: Stochastic Gradient Descent layout implementation
    - Provides force-directed graph layout using stochastic gradient descent optimization
    - Multiple implementation variants for different graph sizes and requirements:
      - `FullSgd`: Uses all-pairs shortest path distances (accurate but slower for large graphs)
      - `SparseSgd`: Uses pivot-based sparse approximation (efficient for large graphs)
      - `DistanceAdjustedSgd`: Dynamically adjusts distances to improve aesthetics
    - Flexible learning rate control through various schedulers:
      - `SchedulerConstant`: Maintains a fixed learning rate
      - `SchedulerLinear`: Linear decay of learning rate
      - `SchedulerExponential`: Exponential decay of learning rate
      - `SchedulerQuadratic`: Quadratic decay of learning rate
      - `SchedulerReciprocal`: Reciprocal decay of learning rate
    - Modular design with the `Sgd` trait for common functionality
    - Reference: Zheng, J. X., Pawar, S., & Goodman, D. F. (2018). "Graph drawing by stochastic gradient descent"
  - `stress-majorization`: Stress Majorization graph layout method
    - Implements the Stress Majorization algorithm for force-directed graph layout
    - `StressMajorization`: Primary struct for the algorithm
    - Features initialization from a graph or pre-computed distance matrix
    - Iteratively minimizes the layout stress by solving a series of quadratic problems
    - Uses conjugate gradient method for efficient optimization
    - Supports customizable edge weights and convergence criteria
    - Based on Gansner et al. (2004) "Graph drawing by stress majorization"
- **crates/python**: Python bindings using PyO3
  - Provides Python interface to the core graph data structures and algorithms
  - Main components:
    - `Graph` and `DiGraph`: Undirected and directed graph data structures
    - `Drawing`: Base class for graph layouts with implementations for different geometric spaces:
      - `DrawingEuclidean2d`: 2D Euclidean space
      - `DrawingEuclidean`: N-dimensional Euclidean space
      - `DrawingHyperbolic2d`: 2D Hyperbolic space
      - `DrawingSpherical2d`: 2D Spherical space
      - `DrawingTorus2d`: 2D Torus space
    - `DistanceMatrix`: Matrix of distances between nodes
    - `Rng`: Random number generator for deterministic randomness
    - Layout algorithms:
      - `SparseSgd` and `FullSgd`: Stochastic gradient descent layout algorithms
      - `DistanceAdjustedSparseSgd` and `DistanceAdjustedFullSgd`: SGD with distance adjustment
      - Various learning rate schedulers (constant, linear, quadratic, exponential, reciprocal)
      - `MDS`, `KamadaKawai`, `StressMajorization` algorithms
    - Graph algorithms:
      - Shortest path functions: `all_sources_bfs`, `all_sources_dijkstra`, `warshall_floyd`
    - Quality metrics: Wrappers around Rust implementations for evaluating graph layouts
  - `examples/`: Example Python scripts demonstrating library usage
    - SGD layouts in various spaces (Euclidean, Spherical, Hyperbolic, Torus)
    - Other layout algorithms (Kamada-Kawai, Stress Majorization)
    - Overwrap removal techniques
- **crates/quality-metrics**: Drawing quality metrics for evaluating graph layouts
  - Collection of metrics to quantitatively assess the quality of graph layouts
  - Includes metrics for:
    - `Stress`: How well layout preserves graph-theoretical distances
    - `IdealEdgeLengths`: How well edge lengths match their ideal lengths
    - `NeighborhoodPreservation`: How well the layout preserves local neighborhoods
    - `CrossingNumber`: Count of edge crossings in the layout
    - `CrossingAngle`: Angles at which edges cross
    - `AspectRatio`: Balance between width and height of the drawing
    - `AngularResolution`: Angles between edges connected to the same node
    - `NodeResolution`: How well nodes are distributed in the drawing space
    - `GabrielGraphProperty`: Adherence to the Gabriel graph condition
  - Provides both individual metric functions and a combined quality evaluation
  - Metrics include sense (maximize/minimize) to indicate optimization direction
- **crates/wasm**: WebAssembly bindings using wasm-bindgen

### Other Directories

- **js/**: JS/TS code (npm workspaces)
  - `js/dataset/`: Dataset processing utilities
  - `js/examples/`: Sample code for JavaScript usage
- **.github/**: GitHub Actions workflows for CI/CD
- **.vscode/**: VS Code editor settings
- **docs/**: Documentation files
- **examples/**: Rust sample code
- **img/**: Image files for documentation
- **scripts/**: Development scripts
- **www/**: WebAssembly frontend examples

### Project Purpose

- A Rust library providing graph data structures, algorithms, quality metrics, and drawing functionality.
- Intended for use from Rust directly, via Python bindings, or via WebAssembly (JavaScript).
