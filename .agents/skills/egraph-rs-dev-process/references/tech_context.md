# Technical Context: egraph-rs

This reference document outlines the technical environment, folder architecture, bindings layout, and system constraints for `egraph-rs`.

## Technologies Used

### Core Technologies
- **Rust (2021 Edition)**: Primary implementation language
- **Cargo**: Build system and package manager
- **WebAssembly/wasm-bindgen**: For JavaScript integration
- **PyO3**: For Python bindings

### Development Tools
- **cargo fmt**: Code formatting
- **cargo clippy**: Linting and static analysis
- **cargo test**: Testing framework
- **npm/Node.js**: For JavaScript-related development
- **prettier**: For JavaScript/TypeScript formatting

## Key Directories

| Directory                | Contents                                                    |
| ------------------------ | ----------------------------------------------------------- |
| `crates/algorithm`       | Graph algorithm implementations                             |
| `crates/clustering`      | Community detection algorithms                              |
| `crates/dataset`         | Graph dataset loaders and utilities                         |
| `crates/drawing`         | Drawing implementations for various spaces                  |
| `crates/edge-bundling`   | Force-directed edge bundling                                |
| `crates/layout`          | Layout algorithms (SGD variants including Omega, MDS, etc.) |
| `crates/python`          | Python bindings (PyO3)                                      |
| `crates/quality-metrics` | Drawing quality evaluation metrics                          |
| `crates/wasm`            | WebAssembly bindings (wasm-bindgen)                         |
| `js/dataset`             | JavaScript dataset files                                    |
| `js/examples`            | JavaScript usage examples                                   |
| `.agents/skills`         | Workspace agent skills (guidelines and context)             |

## WebAssembly Bindings Structure

The WebAssembly bindings (`crates/wasm`) provide JavaScript-friendly interfaces:

### Module Structure
- **src/lib.rs**: Entry point that exports all WASM modules.
- **src/graph/**: Graph data structures and operations (directed and undirected).
- **src/drawing/**: Graph drawing implementations for various geometric spaces (Euclidean, Hyperbolic, Spherical, Torus).
- **src/layout/**: Layout algorithms (Kamada-Kawai, MDS, Stress Majorization, Overlap Removal, SGD).
- **src/edge_bundling.rs**: Force-directed edge bundling.
- **src/clustering.rs**: Graph clustering and coarsening.
- **src/quality_metrics.rs**: Layout quality metrics.
- **src/rng.rs**: Seed-controlled random number generation.

### JavaScript API Features
- **Naming**: Uses camelCase for JavaScript interface methods (e.g., `addNode`, `removeEdge`).
- **JSDoc-style comments**: Document parameters and return types.
- **Error Handling**: Failures return Rust `Result`s which translate to JS exceptions.
- **Callbacks**: Supporting JS functions passed as callbacks (e.g. edge length calculators).

## Technical Constraints

1. **Cross-Language Compatibility**: Must maintain identical algorithmic results across Rust, Python, and JavaScript.
2. **Memory Management**: Handle references carefully, especially within the WebAssembly boundary where large graphs can cause buffer bottlenecks.
3. **Performance**: Layout computations must run efficiently on large datasets.
4. **API Stability**: Minimize breaking changes to public APIs.
