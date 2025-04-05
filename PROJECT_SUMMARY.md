# egraph-rs Project Overview

This document provides an overview of the egraph-rs project structure and its main components.

## Project Structure

egraph-rs is organized as a workspace comprising multiple Rust crates:

```
egraph-rs/
├── crates/                      # Rust crates
│   ├── algorithm/               # Graph algorithm implementations
│   │   ├── connected-components/
│   │   ├── shortest-path/
│   ├── clustering/              # Clustering algorithms
│   ├── dataset/                 # Graph dataset loaders
│   ├── drawing/                 # Drawing implementations
│   ├── edge-bundling/           # Edge bundling algorithms
│   ├── layout/                  # Layout algorithms
│   │   ├── kamada-kawai/
│   │   ├── mds/
│   │   ├── overwrap-removal/
│   │   ├── sgd/
│   │   ├── stress-majorization/
│   ├── quality-metrics/         # Layout quality evaluation
│   ├── python/                  # Python bindings
│   ├── wasm/                    # WebAssembly bindings
├── js/                          # JavaScript-related code
│   ├── dataset/                 # Dataset processing utilities
│   ├── examples/                # JavaScript usage examples
```

## Main Components

### Graph Data Structures

Provides the foundation for the project:

- Undirected and directed graph implementations
- Node and edge management
- Generic data storage

### Layout Algorithms

Algorithms for positioning graph nodes for visual representation:

- **Stochastic Gradient Descent (SGD)**:

  - Full implementation (all-pairs shortest path distances)
  - Sparse implementation (pivot-based sparse approximation)
  - Distance-adjusted implementation (dynamic distance adjustment)

- **Multidimensional Scaling (MDS)**:

  - Classical MDS (full distance matrix)
  - Pivot MDS (uses a subset of nodes as pivots)

- **Stress Majorization**:

  - Iterative stress minimization
  - Conjugate gradient solver

- Other layout algorithms:
  - Kamada-Kawai (spring model based layout)
  - Overlap removal (resolving node overlaps)
  - Separation constraints (layout constraint implementation)

### Drawing Implementations

Graph drawing in various geometric spaces:

- Euclidean 2D drawing (2D Cartesian coordinates)
- Spherical 2D drawing (drawings on a spherical surface)
- Hyperbolic 2D drawing (drawings in hyperbolic space)
- Torus 2D drawing (drawings on a torus surface)
- N-dimensional Euclidean drawing (higher-dimensional Euclidean spaces)

### Quality Metrics

Metrics to quantitatively assess the quality of graph layouts:

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

## Language Interfaces

egraph-rs can be accessed from multiple languages:

### Rust Interface

- Directly usable as native Rust code
- Each crate can be used independently

### Python Bindings

- Python bindings using PyO3
- Python-style API following PEP 8
- Type hints in documentation

### WebAssembly Bindings

- JavaScript interfaces using wasm-bindgen
- JavaScript-friendly naming conventions (camelCase)
- JSDoc-style comments
- Transparent data handling
- Memory safety
- Method chaining
- Error handling
- Callback support

## Architecture Overview

The components of egraph-rs are related as follows:

- **Graph Structures** provide the foundation for all operations
- **Algorithms** operate on graph structures to compute properties
- **Layout Algorithms** position nodes in various geometric spaces
- **Drawing Implementations** render graphs in different coordinate systems
- **Quality Metrics** evaluate the effectiveness of layouts
- **Language Bindings** expose functionality to Python and JavaScript

## Implementation Characteristics

### Modular Structure

- **Modular Crate Structure**: Separating functionality into specialized crates for better maintainability and selective dependencies
- **Cross-Language Support**: Exposing functionality via WebAssembly and Python bindings
- **Multiple Geometric Spaces**: Supporting various drawing spaces (Euclidean, Spherical, Hyperbolic, Torus)
- **Generic Graph Implementation**: Using generic types for node and edge data

### Design Patterns

- **Builder Pattern**: For configurable construction of complex objects like layouts
- **Strategy Pattern**: For interchangeable layout and algorithm implementations
- **Adapter Pattern**: For bridging between different language interfaces
- **Visitor Pattern**: For operations over graph structures
- **Factory Methods**: For creating specialized graph instances
