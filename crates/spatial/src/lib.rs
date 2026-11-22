//! Single-threaded spatial acceleration structures and geometric primitives for `egraph-rs`.
//!
//! `egraph-spatial` provides zero-allocation-friendly workspaces and flat arena data structures
//! targeting Native Rust, Python (PyO3), and WebAssembly (`wasm32-unknown-unknown`) equally:
//! - [`Fft2dWorkspace`]: Reusable in-place Radix-2 1D/2D FFT with call telemetry.
//! - [`IbFft2d`]: Bilinear ($p=1$) and cubic ($p=3$) interpolation-based FFT with `G_dual` binning.
//! - [`FitField2d`]: Fused C1 (KL Student-$t$) + C2 (entropy) repulsion sharing particle FFTs.
//! - [`ArenaQuadtree2d`]: Flat `Vec`-backed Barnes-Hut quadtree.
//! - [`StaticKdTree`]: Flat median-split static $D$-dimensional KD-tree.
//! - [`SweepLineCrossings`]: Exact $O((|E|+K)\log|E|)$ sweep-line crossing detector, torus segments,
//!   and deterministic sampled crossing estimation.
//! - [`RectangleOverlapSweep`]: $O(N \log N + K)$ axis-aligned rectangle overlap detector.
//! - [`Delaunay2d`] (`delaunay` feature): Deterministic `u32` Delaunay triangulation over `spade`.

pub mod crossings;
#[cfg(feature = "delaunay")]
pub mod delaunay;
pub mod fft;
pub mod fit_field;
pub mod ibfft;
pub mod kdtree;
pub mod overlap;
pub mod quadtree;

pub use crossings::{
    CrossingPair, CrossingSummary, SweepLineCrossings, TorusCrossing, sampled_crossing_number,
    sampled_crossing_number_edges, torus_crossing_edges, torus_crossing_edges_from_edges,
    torus_edge_segments,
};
#[cfg(feature = "delaunay")]
pub use delaunay::Delaunay2d;
pub use fft::{Complex64, Fft2dWorkspace};
pub use fit_field::{FitField2d, FitFieldConfig, FitFieldOutput, FitFieldStats};
pub use ibfft::{IbFft2d, IbFftConfig, IbFftGridSpec, InterpolationOrder};
pub use kdtree::StaticKdTree;
pub use overlap::{Rect2d, RectangleOverlapSweep};
pub use quadtree::ArenaQuadtree2d;
