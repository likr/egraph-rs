//! Single-threaded spatial acceleration structures and geometric primitives for `egraph-rs`.
//!
//! `egraph-spatial` provides zero-allocation-friendly workspaces and flat arena data structures
//! targeting Native Rust, Python (PyO3), and WebAssembly (`wasm32-unknown-unknown`) equally:
//! - [`Fft2dWorkspace`]: Reusable in-place Radix-2 1D/2D FFT with call telemetry.
//! - [`IbFft2d`]: Bilinear ($p=1$) and cubic ($p=3$) interpolation-based FFT with `G_dual` binning.
//! - [`FitField2d`]: Fused C1 (KL Student-$t$) + C2 (entropy) repulsion sharing particle FFTs.
//! - [`R2cFft2dWorkspaceF32`]: In-place `f32` 2D Real-to-Complex (`R2C`/`C2R`) Radix-2 FFT workspace.
//! - [`ScreenedPoissonField2d`]: Unilateral upper-bound Screened Poisson density field evaluator.
//! - [`ArenaQuadtree2d`]: Flat `Vec`-backed Barnes-Hut 2D quadtree.
//! - [`ArenaOctree3d`]: Flat `Vec`-backed Barnes-Hut 3D octree.
//! - [`StaticKdTree`]: Flat median-split static $D$-dimensional KD-tree.
//! - [`DynamicKdTree`]: Flat median-split runtime-dimension KD-tree.
//! - [`SweepLineCrossings`]: Exact $O((|E|+K)\log|E|)$ sweep-line crossing detector, torus segments,
//!   and deterministic sampled crossing estimation.
//! - [`RectangleOverlapSweep`]: $O(N \log N + K)$ axis-aligned rectangle overlap detector.
//! - [`Delaunay2d`] (`delaunay` feature): Deterministic `u32` Delaunay triangulation over `spade`.

#![warn(missing_docs)]

#[cfg(feature = "delaunay")]
pub use egraph_spatial_trees::Delaunay2d;
#[cfg(feature = "delaunay")]
pub use egraph_spatial_trees::delaunay;
pub use egraph_spatial_trees::{
    ArenaOctree3d, ArenaQuadtree2d, DynamicKdTree, KdQueryScratch, Rect2d, RectangleOverlapSweep,
    StaticKdTree, kdtree, octree, overlap, quadtree,
};

pub use egraph_spatial_fft::{
    Complex32, Complex64, DensityFieldKernel, Fft2dWorkspace, FitField2d, FitFieldConfig,
    FitFieldOutput, FitFieldStats, IbFft2d, IbFftConfig, IbFftGridSpec, InterpolationOrder,
    R2cFft2dWorkspaceF32, ScreenedPoissonConfig, ScreenedPoissonField2d, ScreenedPoissonOutput,
    ScreenedPoissonStats, fft, fft_r2c, fit_field, ibfft, screened_poisson,
};

pub use egraph_spatial_crossings::{
    CrossingPair, CrossingSummary, SweepLineCrossings, TorusCrossing, TorusLayoutView,
    crossing_edges_hyperbolic, crossing_edges_spherical, crossings, hyperbolic_crossing_number,
    sampled_crossing_number, sampled_crossing_number_edges, spherical_crossing_number,
    torus_crossing_edges, torus_crossing_edges_from_edges, torus_crossing_number,
    torus_edge_segments, torus_layout_edge_segments,
};

#[cfg(test)]
mod tests {
    use super::{
        ArenaOctree3d, ArenaQuadtree2d, Complex64, DynamicKdTree, Fft2dWorkspace, FitFieldConfig,
        Rect2d, RectangleOverlapSweep, ScreenedPoissonConfig, StaticKdTree, SweepLineCrossings,
        torus_edge_segments,
    };

    #[test]
    fn spatial_umbrella_reexports_work() {
        let _tree = ArenaQuadtree2d::<f64>::new();
        let _octree = ArenaOctree3d::<f64>::new();
        let _kd = StaticKdTree::<2, f64>::build(&[[0.0, 0.0], [1.0, 1.0]]).unwrap();
        let _dkd = DynamicKdTree::<f64>::from_flat(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 2, 3).unwrap();
        let _rect = Rect2d::new(0.0, 1.0, 0.0, 1.0).unwrap();
        assert!(!RectangleOverlapSweep::has_overlaps(&[_rect]).unwrap());

        let mut ws = Fft2dWorkspace::new();
        let mut buf = [Complex64::new(1.0, 0.0), Complex64::ZERO];
        ws.fft_1d(&mut buf, false).unwrap();
        assert!(FitFieldConfig::default().validate().is_ok());
        assert!(ScreenedPoissonConfig::default().validate().is_ok());

        let _ = std::any::type_name::<SweepLineCrossings>();
        assert_eq!(torus_edge_segments(0.1, 0.2, 0.3, 0.4).unwrap().len(), 1);
        #[cfg(feature = "delaunay")]
        {
            let _ = std::any::type_name::<super::Delaunay2d>();
        }
    }
}
