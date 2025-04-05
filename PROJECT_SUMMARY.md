# egraph-rs Project Summary

This document provides an overview of the egraph-rs project structure, components, and development processes.

## Table of Contents

- [Quick Reference](#quick-reference)
- [Project Overview](#project-overview)
- [Development Environment](#development-environment)
- [Project Structure](#project-structure)
- [Functional Components](#functional-components)
- [Interfaces](#interfaces)

## Quick Reference

### Frequently Used Commands

| Function            | Command                                                                |
| ------------------- | ---------------------------------------------------------------------- |
| **Rust: Run Tests** | `cargo test --workspace`                                               |
| **Rust: Format**    | `cargo fmt --all`                                                      |
| **Rust: Lint**      | `cargo clippy --workspace --all-targets --all-features -- -D warnings` |
| **JS/TS: Format**   | `npx prettier --write .`                                               |
| **WASM: Build**     | `npm run wasm-build`                                                   |
| **Examples: Run**   | `npm start`                                                            |

### Key Directories

| Directory                | Contents                        |
| ------------------------ | ------------------------------- |
| `crates/algorithm`       | Graph algorithm implementations |
| `crates/layout`          | Layout algorithms               |
| `crates/python`          | Python bindings                 |
| `crates/quality-metrics` | Drawing quality metrics         |
| `js/examples`            | JavaScript usage examples       |

## Project Overview

egraph-rs is a Rust library providing graph data structures, algorithms, quality metrics, and drawing functionality.
It can be used:

- Directly from Rust
- Via Python bindings
- Via WebAssembly (JavaScript)

### Project Configuration

- This project is a Rust workspace (`Cargo.toml`) comprising multiple crates (`crates/*`).
- JavaScript/WASM-related code is managed in the `js/` directory, `crates/wasm` crate, and the root `package.json` (using npm workspaces).

## Development Environment

### Rust Development

- **Edition:** `2021` (verify in each crate's `Cargo.toml`)
- **Commands:**
  - Check: `cargo check --workspace`
  - Test: `cargo test --workspace`
  - Format: `cargo fmt --all`
  - Lint: `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- **Adding Dependencies:** Add to `crates/<target-crate>/Cargo.toml` in the `[dependencies]` section.

### JavaScript/TypeScript Development

- **Format:** Run `npx prettier --write .` (using `.prettierrc.json` settings, currently default)
- **WASM Build:** `npm run wasm-build`
- **Run Examples:** `npm start` (internally runs `npm run dev -w examples`)
- **Adding Dependencies:** Use `npm install <package-name> -w <workspace-name>` (e.g., `-w examples`, `-w crates/wasm`) or run `npm install <package-name>` at the root.

## Project Structure

### Workspace Root

- Structure: Monorepo containing Rust workspace and JavaScript/TypeScript code

### Main Directory Structure

- **crates/**: Rust crates
  - **algorithm/**: Graph algorithms
  - **cli/**: CLI tools
  - **clustering/**: Clustering algorithms
  - **dataset/**: Graph dataset loaders
  - **drawing/**: Drawing-related implementations
  - **edge-bundling/**: Edge bundling algorithms
  - **layout/**: Layout algorithms
  - **python/**: Python bindings (using PyO3)
  - **quality-metrics/**: Drawing quality metrics
  - **wasm/**: WebAssembly bindings (using wasm-bindgen)
- **js/**: JS/TS code (npm workspaces)
  - **dataset/**: Dataset processing utilities
  - **examples/**: JavaScript usage examples
- **Other Directories**:
  - **.github/**: GitHub Actions workflows
  - **.vscode/**: VS Code editor settings
  - **docs/**: Documentation files
  - **examples/**: Rust sample code
  - **img/**: Image files for documentation
  - **scripts/**: Development scripts
  - **www/**: WebAssembly frontend examples

## Functional Components

### Graph Algorithms (`crates/algorithm`)

- **connected-components**: Connected components
- **shortest-path**: Shortest path algorithms (BFS, Dijkstra, Warshall-Floyd, DistanceMatrix)

### Layout Algorithms (`crates/layout`)

#### Stochastic Gradient Descent (SGD) (`crates/layout/sgd`)

Force-directed graph layout using stochastic gradient descent optimization:

- **Implementation Variants**:

  - `FullSgd`: Uses all-pairs shortest path distances (accurate but slower for large graphs)
  - `SparseSgd`: Uses pivot-based sparse approximation (efficient for large graphs)
  - `DistanceAdjustedSgd`: Dynamically adjusts distances to improve aesthetics

- **Learning Rate Schedulers**:

  - `SchedulerConstant`: Maintains a fixed learning rate
  - `SchedulerLinear`: Linear decay of learning rate
  - `SchedulerExponential`: Exponential decay of learning rate
  - `SchedulerQuadratic`: Quadratic decay of learning rate
  - `SchedulerReciprocal`: Reciprocal decay of learning rate

- **Reference**: Zheng, J. X., Pawar, S., & Goodman, D. F. (2018). "Graph drawing by stochastic gradient descent"

#### Multidimensional Scaling (MDS) (`crates/layout/mds`)

Algorithms to visualize graph structures in lower dimensional spaces:

- `ClassicalMds`: Standard implementation that computes a full distance matrix
- `PivotMds`: Efficient implementation that uses a subset of nodes as pivots
- Uses eigendecomposition and double centering to transform distance matrices
- **Reference**: Cox, T. F., & Cox, M. A. (2000). Multidimensional scaling. Chapman & Hall/CRC.

#### Stress Majorization (`crates/layout/stress-majorization`)

- Implements the Stress Majorization algorithm for force-directed graph layout
- Iteratively minimizes the layout stress by solving a series of quadratic problems
- Uses conjugate gradient method for efficient optimization
- **Reference**: Gansner et al. (2004) "Graph drawing by stress majorization"

#### Other Layout Algorithms

- **Kamada-Kawai** (`crates/layout/kamada-kawai`): Spring model based layout algorithm
- **Overlap Removal** (`crates/layout/overwrap-removal`): Algorithm to resolve node overlaps
- **Separation Constraints** (`crates/layout/separation-constraints`): Layout constraint implementation

### Drawing Quality Metrics (`crates/quality-metrics`)

Collection of metrics to quantitatively assess the quality of graph layouts:

- `Stress`: How well layout preserves graph-theoretical distances
- `IdealEdgeLengths`: How well edge lengths match their ideal lengths
- `NeighborhoodPreservation`: How well the layout preserves local neighborhoods
- `CrossingNumber`: Count of edge crossings in the layout
- `EdgeAngle`: Angles at which edges cross
- `AspectRatio`: Balance between width and height of the drawing
- `AngularResolution`: Angles between edges connected to the same node
- `NodeResolution`: How well nodes are distributed in the drawing space
- `GabrielGraphProperty`: Adherence to the Gabriel graph condition

## Interfaces

### Python Bindings (`crates/python`)

Python bindings structure using PyO3:

- **src/lib.rs**: Main entry point that registers all submodules
- **Data Structures**:

  - **src/graph/**: Graph data structures (`Graph`, `DiGraph`)
  - **src/drawing/**: Drawing implementations for various geometric spaces (Euclidean, Spherical, Hyperbolic, Torus, etc.)
  - **src/distance_matrix.rs**: Distance matrix implementation
  - **src/rng.rs**: Random number generation

- **Algorithms**:
  - **src/layout/**: Layout algorithms (SGD, MDS, Stress Majorization, etc.)
  - **src/algorithm/**: Graph algorithms (shortest path, etc.)
  - **src/quality_metrics.rs**: Layout quality evaluation metrics

### WebAssembly Bindings (`crates/wasm`)

WebAssembly bindings using wasm-bindgen to call Rust implementations from browser environments:

- **src/lib.rs**: Entry point that exports all WASM modules
- **src/graph.rs, src/graph/graph_impl.rs**: Graph data structures (`Graph`, `DiGraph`) for JavaScript
- **src/drawing.rs, src/drawing/\*.rs**: Drawing implementations for different geometries
- **src/layout/\*.rs**: Layout algorithms (Kamada-Kawai, SGD, etc.)
- **src/edge_bundling.rs**: Force-directed edge bundling for clearer visualizations
- **src/clustering.rs**: Graph clustering and coarsening functionality
- **src/quality_metrics.rs**: Layout quality evaluation metrics (stress, crossing number, etc.)
- **src/rng.rs**: Random number generation for stochastic algorithms

The WASM module provides JavaScript interfaces to the Rust implementations through clearly documented APIs, keeping the same general structure as the core Rust library but adapting function signatures and data structures to work well with JavaScript.
