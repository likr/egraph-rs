//! Single-threaded spatial acceleration structures, density-field FFTs, and crossing detectors.

pub use egraph_spatial_crossings::{
    CrossingPair, CrossingSummary, SweepLineCrossings, TorusCrossing, TorusLayoutView, TorusView,
    crossing_edges_hyperbolic, crossing_edges_spherical, hyperbolic_crossing_number,
    sampled_crossing_number, sampled_crossing_number_edges, spherical_crossing_number,
    torus_crossing_edges, torus_crossing_edges_from_edges, torus_crossing_number,
    torus_edge_segments, torus_layout_edge_segments,
};
pub use egraph_spatial_fft::{
    Complex32, Complex64, DensityFieldKernel, Fft2dWorkspace, FitField2d, FitFieldConfig,
    FitFieldOutput, FitFieldStats, IbFft2d, IbFftConfig, IbFftGridSpec, InterpolationOrder,
    R2cFft2dWorkspaceF32, ScreenedPoissonConfig, ScreenedPoissonField2d, ScreenedPoissonOutput,
    ScreenedPoissonStats,
};
pub use egraph_spatial_trees::{
    ArenaOctree3d, ArenaQuadtree2d, Delaunay2d, DynamicKdTree, KdQueryScratch,
    NearestNeighborIndex, Rect2d, RectangleOverlapSweep, SpatialIndex, StaticKdTree,
    squarified_treemap,
};
