//! Spatial indexing trees, rectangle overlap sweeps, squarified treemaps, and 2D Delaunay triangulation for `egraph-rs`.

mod delaunay;
mod kdtree;
mod octree;
mod overlap;
mod quadtree;

pub use delaunay::Delaunay2d;
pub use kdtree::{DynamicKdTree, KdQueryScratch, NearestNeighborIndex, SpatialIndex, StaticKdTree};
pub use octree::ArenaOctree3d;
pub use overlap::{Rect2d, RectangleOverlapSweep, squarified_treemap};
pub use quadtree::ArenaQuadtree2d;
