# System Patterns: egraph-rs

## System Architecture

egraph-rs follows a modular architecture organized as a Rust workspace with multiple specialized crates:

```
egraph-rs/
├── crates/
│   ├── algorithm/           # Graph algorithms
│   │   ├── connected-components/
│   │   ├── shortest-path/
│   ├── clustering/          # Clustering algorithms
│   ├── dataset/             # Graph dataset loaders
│   ├── drawing/             # Drawing implementations
│   ├── edge-bundling/       # Edge bundling algorithms
│   ├── layout/              # Layout algorithms
│   │   ├── kamada-kawai/
│   │   ├── mds/
│   │   ├── overwrap-removal/
│   │   ├── sgd/
│   │   ├── stress-majorization/
│   ├── quality-metrics/     # Layout quality evaluation
│   ├── python/              # Python bindings
│   ├── wasm/                # WebAssembly bindings
```

## Key Technical Decisions

1. **Modular Crate Structure**: Separating functionality into specialized crates for better maintainability and selective dependencies
2. **Cross-Language Support**: Exposing functionality via WebAssembly and Python bindings
3. **Multiple Geometric Spaces**: Supporting various drawing spaces (Euclidean, Spherical, Hyperbolic, Torus)
4. **Generic Graph Implementation**: Using generic types for node and edge data

## Design Patterns in Use

1. **Builder Pattern**: For configurable construction of complex objects like layouts
2. **Strategy Pattern**: For interchangeable layout and algorithm implementations
3. **Adapter Pattern**: For bridging between different language interfaces
4. **Visitor Pattern**: For operations over graph structures
5. **Factory Methods**: For creating specialized graph instances

## Layout Algorithm Implementations

### Stochastic Gradient Descent (SGD) (`crates/layout/sgd`)

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

### Multidimensional Scaling (MDS) (`crates/layout/mds`)

Algorithms to visualize graph structures in lower dimensional spaces:

- `ClassicalMds`: Standard implementation that computes a full distance matrix
- `PivotMds`: Efficient implementation that uses a subset of nodes as pivots
- Uses eigendecomposition and double centering to transform distance matrices
- **Reference**: Cox, T. F., & Cox, M. A. (2000). Multidimensional scaling. Chapman & Hall/CRC.

### Stress Majorization (`crates/layout/stress-majorization`)

- Implements the Stress Majorization algorithm for force-directed graph layout
- Iteratively minimizes the layout stress by solving a series of quadratic problems
- Uses conjugate gradient method for efficient optimization
- **Reference**: Gansner et al. (2004) "Graph drawing by stress majorization"

### Other Layout Algorithms

- **Kamada-Kawai** (`crates/layout/kamada-kawai`): Spring model based layout algorithm
- **Overlap Removal** (`crates/layout/overwrap-removal`): Algorithm to resolve node overlaps
- **Separation Constraints** (`crates/layout/separation-constraints`): Layout constraint implementation

## Component Relationships

- **Graph Structures** provide the foundation for all operations
- **Algorithms** operate on graph structures to compute properties
- **Layout Algorithms** position nodes in various geometric spaces
- **Drawing Implementations** render graphs in different coordinate systems
- **Quality Metrics** evaluate the effectiveness of layouts
- **Language Bindings** expose functionality to Python and JavaScript

## Drawing Quality Metrics (`crates/quality-metrics`)

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
