# egraph

A high-performance graph drawing library for Rust, Python, and WebAssembly. egraph provides comprehensive graph algorithms, layout techniques, and visualization tools for network analysis and visualization.

## Features

- **Multiple Language Bindings**: Native Rust, Python (via PyO3), and JavaScript/WebAssembly
- **Rich Layout Algorithms**: SGD variants, MDS, Stress Majorization, Kamada-Kawai, and more
- **Multiple Drawing Spaces**: Euclidean (2D/nD), Spherical, Hyperbolic, and Torus
- **Community Detection**: Louvain, Label Propagation, Spectral Clustering, InfoMap
- **Graph Algorithms**: Shortest path, triangulation, layering, connected components
- **Quality Metrics**: Comprehensive layout evaluation tools

## Python Bindings (Alpha Release)

**Version: 6.0.0-alpha**

The Python bindings provide a complete interface to egraph's functionality with a Pythonic API.

### Installation

```bash
# Clone the repository
git clone https://github.com/likr/egraph-rs.git
cd egraph-rs/crates/python

# Install with pip (requires Rust toolchain)
pip install maturin
maturin develop --release
```

### Quick Start

```python
import egraph as eg

# Create a graph
graph = eg.Graph()
nodes = [graph.add_node() for _ in range(10)]
for i in range(9):
    graph.add_edge(nodes[i], nodes[i + 1])

# Apply a layout
drawing = eg.DrawingEuclidean2d.initial_placement(graph)
rng = eg.Rng.seed_from(42)

sgd = eg.FullSgd().build(graph, lambda i: 1.0, rng)
scheduler = sgd.scheduler_exponential(100)

def step(eta):
    sgd.shuffle(rng)
    sgd.apply(drawing, eta)

scheduler.run(step)

# Get positions
positions = {i: (drawing.x(i), drawing.y(i)) for i in range(graph.node_count())}
```

### Documentation

Full Python documentation: **https://egraph-py-docs.netlify.app/**

See [crates/python/README.md](crates/python/README.md) for detailed Python binding documentation.

## Development

### Rust (Native)

```bash
# Build all crates
cargo build --workspace

# Run tests
make test

# Run tests for specific crate
make test-crate CRATE=petgraph-layout-sgd

# Format and lint
make fmt
make lint
```

### Python

```bash
# Build Python bindings
make python-build

# Run Python tests
make python-test

# Build documentation
make python-docs

# Run doctests
make python-doctest
```

### WebAssembly

```bash
# Build WebAssembly module
npm run wasm-build

# Run WebAssembly tests
wasm-pack test --node crates/wasm

# Run examples
npm start
```

## Project Structure

```
egraph-rs/
├── crates/
│   ├── algorithm/          # Graph algorithms
│   ├── clustering/         # Community detection
│   ├── drawing/           # Drawing implementations
│   ├── layout/            # Layout algorithms
│   ├── python/            # Python bindings (PyO3)
│   ├── wasm/              # WebAssembly bindings
│   └── quality-metrics/   # Layout quality metrics
├── js/                    # JavaScript examples and datasets
└── memory-bank/           # Project documentation
```

## Key Algorithms

### Layout Algorithms

- **SGD Family**: Full, Sparse, Distance-Adjusted, Omega variants
- **Classical**: MDS (Classical/Pivot), Stress Majorization, Kamada-Kawai
- **Specialized**: Overlap Removal, Separation Constraints

### Community Detection

- Louvain (modularity-based)
- Label Propagation
- Spectral Clustering
- InfoMap (information-theoretic)

### Graph Algorithms

- Shortest Path (Dijkstra)
- Delaunay Triangulation
- Hierarchical Layering
- Connected Components

## Documentation

- **Python**: https://egraph-py-docs.netlify.app/
- **Rust**: Run `cargo doc --open` for API documentation
- **Examples**: See `js/examples/` and `crates/python/examples/`

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT

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
