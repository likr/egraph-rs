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
  - `mds`: Multidimensional Scaling implementation
  - `overwrap-removal`: Overlap removal algorithms for graph layouts
  - `separation-constraints`: Separation constraints for layout algorithms
  - `sgd`: Stochastic Gradient Descent layout implementation
  - `stress-majorization`: Stress Majorization graph layout method
- **crates/python**: Python bindings using PyO3
- **crates/quality-metrics**: Drawing quality metrics for evaluating graph layouts
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
