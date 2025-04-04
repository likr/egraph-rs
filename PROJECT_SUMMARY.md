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
- **crates/cli**: CLI (binary?)
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
- **crates/edge-bundling**: (Incomplete) Edge bundling
  - `fdeb`: (Incomplete) Force Directed Edge Bundling?
- **crates/layout**: Layout algorithms
  - `kamada-kawai`: (Incomplete) Kamada-Kawai method
  - `mds`: (Incomplete) Multidimensional Scaling
  - `overwrap-removal`: (Incomplete) Overlap removal
  - `separation-constraints`: (Incomplete) Separation constraints
  - `sgd`: (Incomplete) Stochastic Gradient Descent layout
  - `stress-majorization`: (Incomplete) Stress Majorization method
- **crates/python**: (Incomplete) Python bindings (PyO3?)
- **crates/quality-metrics**: (Incomplete) Drawing quality metrics
- **crates/wasm**: (Incomplete) WASM bindings (wasm-bindgen?)

### Other Directories

- **js/**: JS/TS code (npm workspaces)
  - `js/dataset/`: Dataset-related?
  - `js/examples/`: Sample code?
- **.github/**: (Incomplete) GitHub Actions workflows
- **.vscode/**: (Incomplete) VS Code settings
- **docs/**: (Incomplete) Documentation
- **examples/**: (Incomplete) Rust sample code
- **img/**: (Incomplete) Image files
- **scripts/**: (Incomplete) Development scripts
- **www/**: (Incomplete) WASM frontend?

### Project Purpose

- A Rust library providing graph data structures, algorithms, quality metrics, and drawing functionality.
- Intended for use from Python and WebAssembly (JavaScript).
