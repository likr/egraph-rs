# `egraph-rs` (Rust library crate `egraph`)

Unified, strictly single-threaded pure-Rust graph drawing and layout facade crate.

> **Package vs. Library Name**: Install from crates.io as **`egraph-rs`** (`Cargo.toml`: `egraph-rs = "=6.0.0-beta.3"`) and import in Rust code as **`use egraph::prelude::*;`**.

## Quick Start

Add `egraph-rs` to your `Cargo.toml`:

```toml
[dependencies]
egraph-rs = "=6.0.0-beta.3"
```

Then construct a zero-copy `CsrGraph`, initialize with `PivotMds`, run a cooperative `IterativeSolver`, and evaluate selective layout metrics:

```rust
use egraph::prelude::*;

fn main() -> Result<(), CoreError> {
    let graph = CsrGraph::<f64>::from_unweighted_edges(
        6,
        &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
        false,
        false,
    )?;

    let mut layout: Layout2d<f64> = PivotMds::<Layout2d<f64>>::from_config(
        &graph,
        PivotMdsConfig::new().with_num_pivots(6).with_seed(42),
    )
    .generate()?;
    let mut solver = TsNetSolver::<Layout2d<f64>>::from_graph_pure_2hop(
        &graph,
        TsNetConfig::pure_2hop(),
    )?;
    solver.run(&mut layout)?;

    let config = MetricConfig::default();
    let targets = [
        MetricTarget::ScaleNormalizedStress,
        MetricTarget::NeighborhoodPreservation2Hop,
        MetricTarget::CrossingNumber,
    ];
    let report = evaluate_metrics(&graph, &layout, &targets, &config)?;
    println!("SNS={:?}, NP2={:?}", report.scale_normalized_stress, report.neighborhood_preservation_2hop);
    Ok(())
}
```

For 3D (`Layout3d<f64>`), N-dimensional (`LayoutNd<f64>`), non-Euclidean manifolds (`ManifoldLayout2d<f64>`), and edge bundling (`EdgePolylines`), select the target layout type directly in the solver type parameter:

```rust
use egraph::prelude::*;

fn run_3d_and_polylines(graph: &CsrGraph<f64>) -> Result<(), CoreError> {
    let mut layout_3d = Layout3d::<f64>::init(graph.num_nodes(), 42, 1.0)?;
    ForceAtlas2Solver::<Layout3d<f64>>::new(graph, ForceAtlas2Config::new())?
        .run(&mut layout_3d)?;

    let mut polylines = EdgePolylines::straight_line_from_graph(graph, &layout_3d)?;
    FdebSolver::<Layout3d<f64>>::new(&polylines, FdebConfig::new())?.run(&mut polylines)?;
    Ok(())
}
```

## Feature Flags

`default` enables `spatial`, `edge-routing`, `linalg`, `neighborhood`, `layout`, `clustering`, `metrics`, and `dataset`. (`dataset-embedded`, `baselines`, `eval-baselines`, and `serde` are opt-in.)

| Feature | Purpose |
|---|---|
| `spatial` | `StaticKdTree`, `DynamicKdTree`, `ArenaOctree3d`, `Delaunay2d`, FFT / ibFFT, sweep-line crossings |
| `edge-routing` | `EdgePolylines` force-directed edge bundling (`FdebSolver`) |
| `linalg` | `CsrMatrix`, Laplacians, Chebyshev diffusion kernels, `AmgPcgSolver`, `PivotMds` / `ClassicalMds` / `Hde` / `RdMds` / `KorenSpectral` |
| `neighborhood` | Exact 1+2-hop CSR, adaptive perplexity, DoG / rHGH Varadhan landmarks, `fuzzy_simplicial_set` |
| `layout` | `SgdSolver`, `TsNetSolver`, `GuMapSolver`, `TFdpSolver`, `IbUmapSolver`, `ForceAtlas2Solver`, `StressMajorizationSolver`, `PrismOverlapRemoval`, `SpatialIntruderRefiner`, etc. |
| `clustering` | Louvain, Leiden, Spectral Clustering, Label Propagation, K-Means++, graph coarsening |
| `metrics` | Selective `MetricReport` / `evaluate_metrics` (SNS, geodesic distortion, NP1/NP2, crossings, G-CAF, RTA, NOR) |
| `dataset` | Dataset loaders and synthetic graph generators |
| `dataset-embedded` | Opt-in embedded CSR benchmark fixtures (`karate`, `lesmis`, `1138_bus`, etc.) |
| `baselines` | Opt-in comparative layout algorithms (`LinLogSolver`, `FruchtermanReingoldSolver`, `GemSolver`, `GripSolver`, etc.) |
| `eval-baselines` | Opt-in cluster separation and baseline quality metrics (`ARI`, `AMI`, `Silhouette`, `Sprawlter-NN`, etc.) |
| `serde` | `serde` serialization support on `egraph-core` types |
| `full` | Enables all domain, baseline, dataset (including `dataset-embedded`), and `serde` features |

## Guarantees & Requirements

- **MSRV**: Rust `1.88` (2024 edition).
- **Strictly Single-Threaded Pure Rust**: Zero `rayon`, `std::thread`, Web Workers, GPU, or C/Fortran FFI dependencies.
- **Deterministic**: All stochastic initializers, solvers, and metrics accept explicit `u64` seeds (`DeterministicRng`).
- **Cooperative Execution**: Iterative solvers implement `IterativeSolver` (`step()`, `run_steps(k)`, `run()`, `status()`).
- **Documentation**: [API docs on docs.rs](https://docs.rs/egraph-rs) | [Repository](https://github.com/likr/egraph-rs/tree/master/crates/egraph)
- **License**: MIT ([LICENSE](https://github.com/likr/egraph-rs/blob/master/crates/egraph/LICENSE)).
