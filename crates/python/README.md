# egraph - Python Graph Drawing Library

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://egraph-py-docs.netlify.app/)
[![Python Version](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Version: 6.0.0-alpha**

A high-performance graph drawing and layout library for Python, powered by Rust. egraph provides a comprehensive suite of graph algorithms, layout techniques, and visualization tools for network analysis and visualization.

## Features

### Graph Data Structures

- **Undirected Graphs** (`Graph`): Standard undirected graph with node and edge data
- **Directed Graphs** (`DiGraph`): Directed graph with full edge direction support
- **NetworkX Integration**: Seamless conversion to/from NetworkX graphs

### Layout Algorithms

- **SGD-based Layouts**: Stochastic Gradient Descent with multiple variants
  - Full SGD: Complete pairwise distance optimization
  - Sparse SGD: Efficient layout for large graphs
  - Distance-Adjusted SGD: Adaptive distance-based optimization
  - Omega: Spectral coordinates with SGD refinement
- **Classical Algorithms**:
  - MDS (Multidimensional Scaling): Classical and Pivot-based variants
  - Stress Majorization: Iterative stress minimization
  - Kamada-Kawai: Spring-based force-directed layout
- **Specialized Layouts**:
  - Overlap Removal: Eliminate node overlaps while preserving structure
  - Separation Constraints: Hierarchical layouts with constraints

### Drawing Spaces

- **Euclidean**: 2D and n-dimensional Cartesian coordinates
- **Spherical**: Layout on sphere surface
- **Hyperbolic**: Hyperbolic space visualization
- **Torus**: Toroidal surface layouts

### Community Detection

- **Louvain**: Modularity-based community detection
- **Label Propagation**: Fast community detection via label spreading
- **Spectral Clustering**: Eigenvalue-based clustering
- **InfoMap**: Information-theoretic community detection
- **Graph Coarsening**: Hierarchical graph simplification

### Graph Algorithms

- **Shortest Path**: Dijkstra's algorithm with custom edge weights
- **Triangulation**: Delaunay triangulation
- **Layering**: Hierarchical graph layering with cycle removal
- **Connected Components**: Component detection and analysis

### Quality Metrics

- **Stress**: Layout quality measurement
- **Edge Crossings**: Crossing number calculation
- **Angular Resolution**: Angle distribution analysis
- **Node Overlap**: Overlap detection and quantification

## Installation

### Alpha Release Installation

Since this is an alpha release, installation requires building from source:

```bash
# Clone the repository
git clone https://github.com/likr/egraph-rs.git
cd egraph-rs/crates/python

# Install with pip (requires Rust toolchain)
pip install maturin
maturin develop --release
```

### Requirements

- **Python**: 3.6 or higher
- **Rust**: Latest stable Rust toolchain (for building from source)
- **Optional**: NumPy for array operations

## Quick Start

```python
import egraph as eg
import networkx as nx

# Create a graph
graph = eg.Graph()
nodes = [graph.add_node() for _ in range(10)]
for i in range(9):
    graph.add_edge(nodes[i], nodes[i + 1])

# Apply a layout algorithm
drawing = eg.DrawingEuclidean2d.initial_placement(graph)
rng = eg.Rng.seed_from(42)

# Use SGD layout
sgd = eg.FullSgd().build(graph, lambda i: 1.0, rng)
scheduler = sgd.scheduler_exponential(100)

def step(eta):
    sgd.shuffle(rng)
    sgd.apply(drawing, eta)

scheduler.run(step)

# Get node positions
positions = {i: (drawing.x(i), drawing.y(i)) for i in range(graph.node_count())}
print(positions)

# Convert to NetworkX for visualization
nx_graph = nx.Graph()
indices = {}
for u in range(graph.node_count()):
    indices[u] = nx_graph.add_node(u)

for edge_idx in range(graph.edge_count()):
    u, v = graph.edge_endpoints(edge_idx)
    nx_graph.add_edge(indices[u], indices[v])

# Use with matplotlib
import matplotlib.pyplot as plt
nx.draw(nx_graph, pos=positions, with_labels=True)
plt.show()
```

## Documentation

Full documentation is available at: **https://egraph-py-docs.netlify.app/**

The documentation includes:

- **Getting Started Guide**: Installation and basic usage
- **Tutorials**: Step-by-step guides for common tasks
- **API Reference**: Complete API documentation with examples
- **Examples**: Real-world usage examples

## Key Capabilities

### Multiple Layout Algorithms

```python
# Classical MDS
mds = eg.ClassicalMds(graph, lambda i: 1.0)
drawing = mds.run(2)

# Stress Majorization
sm = eg.StressMajorization(graph, lambda i: 1.0)
drawing = eg.DrawingEuclidean2d.initial_placement(graph)
sm.run(drawing, 100)

# Kamada-Kawai
kk = eg.KamadaKawai(graph, lambda i: 1.0)
drawing = eg.DrawingEuclidean2d.initial_placement(graph)
kk.run(drawing)
```

### Community Detection

```python
# Louvain algorithm
louvain = eg.Louvain()
communities = louvain.detect(graph)

# Get community assignments
for node_id in range(graph.node_count()):
    community_id = communities.community(node_id)
    print(f"Node {node_id} -> Community {community_id}")
```

### Different Drawing Spaces

```python
# Spherical layout
drawing_spherical = eg.DrawingSpherical2d.initial_placement(graph)
sgd.apply(drawing_spherical, 0.1)

# Hyperbolic layout
drawing_hyperbolic = eg.DrawingHyperbolic2d.initial_placement(graph)
sgd.apply(drawing_hyperbolic, 0.1)
```

## Development

### Building from Source

```bash
# Install development dependencies
pip install maturin pytest

# Build in development mode
maturin develop

# Run tests
python -m pytest tests/

# Build documentation
cd docs
make html
```

### Running Tests

```bash
# Run all tests
make python-test

# Run specific test module
make python-test-module MODULE=test_sgd

# Run doctests
make python-doctest
```

## Performance

egraph is built on Rust, providing:

- **High Performance**: Native speed for graph algorithms
- **Memory Efficiency**: Optimized memory usage for large graphs
- **Parallel Processing**: Multi-threaded algorithms where applicable
- **Type Safety**: Rust's type system prevents common errors

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.

## Citation

If you use egraph in your research, please cite:

```bibtex
@software{egraph,
  title = {egraph: High-Performance Graph Drawing Library},
  author = {Yosuke Onoue},
  year = {2025},
  url = {https://github.com/likr/egraph-rs}
}
```

## Acknowledgments

egraph builds upon research in graph drawing and network visualization, incorporating algorithms from the graph drawing community.

## Links

- **Documentation**: https://egraph-py-docs.netlify.app/
- **GitHub Repository**: https://github.com/likr/egraph-rs
- **Issue Tracker**: https://github.com/likr/egraph-rs/issues
