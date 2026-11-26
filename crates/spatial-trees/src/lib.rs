//! Spatial indexing trees, rectangle overlap sweeps, and optional Delaunay triangulation for `egraph-rs`.

#[cfg(feature = "delaunay")]
pub mod delaunay;
pub mod kdtree;
pub mod overlap;
pub mod quadtree;

#[cfg(feature = "delaunay")]
pub use delaunay::Delaunay2d;
pub use kdtree::{KdQueryScratch, StaticKdTree};
pub use overlap::{Rect2d, RectangleOverlapSweep};
pub use quadtree::ArenaQuadtree2d;

/// Returns the crate version.
pub fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::crate_version;

    #[test]
    fn spatial_trees_crate_version_non_empty() {
        assert!(!crate_version().is_empty());
    }
}
