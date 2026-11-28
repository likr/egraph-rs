# `egraph-spatial-fft`

Single-threaded 2D FFT, interpolation-based FFT, and screened Poisson density-field kernels for `egraph-rs`.

> **Recommended Entry Point**: `egraph-spatial-fft` is a component crate of **`egraph-rs`**. Most Rust users should install **`egraph-rs`** (`cargo add egraph-rs@6.0.0-beta.3`) and import items via `use egraph::prelude::*;` (or `use egraph::spatial::...;`).

## Overview

Provides strictly single-threaded, pure-Rust frequency-domain spatial kernels used by FIt-SNE / linear-FFT force evaluators and screened Poisson field density refinement (`ScreenedPoissonField2d`, 2D FFT convolution, and Lagrange/Fourier interpolation grids).

## Direct Dependency (Minimal / Advanced Use)

```bash
cargo add egraph-spatial-fft@6.0.0-beta.3
```

## Links

- **API Documentation**: [docs.rs/egraph-spatial-fft](https://docs.rs/egraph-spatial-fft)
- **Unified Facade Crate**: [docs.rs/egraph-rs](https://docs.rs/egraph-rs)
- **Source**: [crates/spatial-fft on GitHub](https://github.com/likr/egraph-rs/tree/master/crates/spatial-fft)
- **License**: MIT ([LICENSE](https://github.com/likr/egraph-rs/blob/master/crates/spatial-fft/LICENSE))
