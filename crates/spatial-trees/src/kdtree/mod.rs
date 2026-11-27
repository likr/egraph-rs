//! Flat median-split static $D$-dimensional KD-tree (`StaticKdTree`).
//!
//! Stores permuted points in contiguous leaf slices with tight bounding boxes per internal
//! node, accumulating squared Euclidean distances in `f64` and breaking ties deterministically
//! by ascending original point index for exact parity with $O(N^2)$ brute-force search.

use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, EuclideanLayout, EuclideanStorageRef, Layout2d, Layout3d, LayoutNd,
    Scalar,
};

mod dynamic;
mod dynamic_query;
mod index;
mod index_impls;
mod query;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_dynamic;
#[cfg(test)]
mod tests_index;
mod types;

pub use dynamic::DynamicKdTree;
pub use index::{NearestNeighborIndex, SpatialIndex};
pub use types::KdQueryScratch;
use types::{KdEntry, KdNode, LEAF_SIZE};

/// Flat static median-split KD-tree in $D$ dimensions.
///
/// Distance computations and bounding-box pruning always execute in `f64` regardless of
/// whether the input coordinates are `f32` or `f64`.
#[derive(Debug, Clone)]
pub struct StaticKdTree<const D: usize, S: Scalar = f64> {
    nodes: Vec<KdNode<D>>,
    entries: Vec<KdEntry<D>>,
    coords_by_id: Vec<[f64; D]>,
    _marker: PhantomData<S>,
}

impl<const D: usize, S: Scalar> StaticKdTree<D, S> {
    /// Builds a static KD-tree from a slice of $D$-dimensional points.
    ///
    /// Returns [`CoreError::InvalidInput`] when `D == 0` and [`CoreError::NonFiniteValue`]
    /// if any coordinate is `NaN` or infinite.
    pub fn build(points: &[[S; D]]) -> CoreResult<Self> {
        if D == 0 {
            return Err(CoreError::invalid_input(
                "StaticKdTree dimension D must be >= 1",
            ));
        }
        let n = points.len();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_input(
                "StaticKdTree point count exceeds u32::MAX",
            ));
        }

        let mut coords_by_id = Vec::with_capacity(n);
        let mut entries = Vec::with_capacity(n);
        for (i, pt) in points.iter().enumerate() {
            let mut c = [0.0_f64; D];
            for d in 0..D {
                if !pt[d].is_finite() {
                    return Err(CoreError::non_finite("StaticKdTree::build"));
                }
                c[d] = pt[d].to_f64();
            }
            coords_by_id.push(c);
            entries.push(KdEntry {
                coords: c,
                index: i as u32,
            });
        }

        let mut tree = Self {
            nodes: Vec::new(),
            entries,
            coords_by_id,
            _marker: PhantomData,
        };
        if n > 0 {
            let est_nodes = (2 * n / LEAF_SIZE).max(8);
            tree.nodes.reserve(est_nodes);
            tree.build_node(0, n);
        }
        Ok(tree)
    }

    /// Builds a $D$-dimensional KD-tree from a flat [`LayoutNd<S>`].
    pub fn from_layout_nd(layout: &LayoutNd<S>) -> CoreResult<Self> {
        if D == 0 {
            return Err(CoreError::invalid_input(
                "StaticKdTree dimension D must be >= 1",
            ));
        }
        if layout.dims() != D {
            return Err(CoreError::dimension_mismatch(
                D,
                layout.dims(),
                "StaticKdTree::from_layout_nd",
            ));
        }
        if !layout.is_all_finite() {
            return Err(CoreError::non_finite("StaticKdTree::from_layout_nd"));
        }
        let n = layout.num_nodes();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_input(
                "StaticKdTree point count exceeds u32::MAX",
            ));
        }

        let flat = layout.coords();
        let mut coords_by_id = Vec::with_capacity(n);
        let mut entries = Vec::with_capacity(n);
        for i in 0..n {
            let mut c = [0.0_f64; D];
            let base = i * D;
            for d in 0..D {
                c[d] = flat[base + d].to_f64();
            }
            coords_by_id.push(c);
            entries.push(KdEntry {
                coords: c,
                index: i as u32,
            });
        }

        let mut tree = Self {
            nodes: Vec::new(),
            entries,
            coords_by_id,
            _marker: PhantomData,
        };
        if n > 0 {
            tree.nodes.reserve((2 * n / LEAF_SIZE).max(8));
            tree.build_node(0, n);
        }
        Ok(tree)
    }

    /// Returns the number of points stored in the tree.
    #[inline]
    pub fn len(&self) -> usize {
        self.coords_by_id.len()
    }

    /// Returns `true` if the tree stores zero points.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.coords_by_id.is_empty()
    }

    /// Returns the `f64` coordinates of original point `point_idx`.
    pub fn point(&self, point_idx: usize) -> CoreResult<[f64; D]> {
        self.coords_by_id.get(point_idx).copied().ok_or_else(|| {
            CoreError::index_out_of_bounds(point_idx, self.len(), "StaticKdTree::point")
        })
    }

    fn build_node(&mut self, start: usize, end: usize) -> u32 {
        let mut min_bounds = self.entries[start].coords;
        let mut max_bounds = min_bounds;
        for entry in &self.entries[start + 1..end] {
            for d in 0..D {
                min_bounds[d] = min_bounds[d].min(entry.coords[d]);
                max_bounds[d] = max_bounds[d].max(entry.coords[d]);
            }
        }

        let node_idx = self.nodes.len() as u32;
        if end - start <= LEAF_SIZE {
            self.nodes.push(KdNode {
                min_bounds,
                max_bounds,
                start,
                end,
                split_dim: 0,
                split_val: 0.0,
                left: u32::MAX,
                right: u32::MAX,
            });
            return node_idx;
        }

        let mut split_dim = 0usize;
        let mut max_spread = -1.0_f64;
        for d in 0..D {
            let spread = max_bounds[d] - min_bounds[d];
            if spread > max_spread {
                max_spread = spread;
                split_dim = d;
            }
        }

        // Collapse identical/duplicate clusters into a single leaf to avoid empty splits.
        if max_spread <= 0.0 {
            self.nodes.push(KdNode {
                min_bounds,
                max_bounds,
                start,
                end,
                split_dim: 0,
                split_val: 0.0,
                left: u32::MAX,
                right: u32::MAX,
            });
            return node_idx;
        }

        let mid = start + (end - start) / 2;
        let slice = &mut self.entries[start..end];
        let local_mid = mid - start;
        slice.select_nth_unstable_by(local_mid, |a, b| {
            a.coords[split_dim]
                .total_cmp(&b.coords[split_dim])
                .then_with(|| a.index.cmp(&b.index))
        });
        let split_val = self.entries[mid].coords[split_dim];

        self.nodes.push(KdNode {
            min_bounds,
            max_bounds,
            start,
            end,
            split_dim,
            split_val,
            left: u32::MAX,
            right: u32::MAX,
        });

        let left = self.build_node(start, mid);
        let right = self.build_node(mid, end);
        self.nodes[node_idx as usize].left = left;
        self.nodes[node_idx as usize].right = right;
        node_idx
    }

    /// Builds a $D$-dimensional KD-tree from any [`EuclideanLayout`] whose `dims() == D`.
    pub fn from_euclidean<L: EuclideanLayout<Scalar = S> + ?Sized>(layout: &L) -> CoreResult<Self> {
        if D == 0 {
            return Err(CoreError::invalid_input(
                "StaticKdTree dimension D must be >= 1",
            ));
        }
        if layout.dims() != D {
            return Err(CoreError::dimension_mismatch(
                D,
                layout.dims(),
                "StaticKdTree::from_euclidean",
            ));
        }
        let n = layout.len();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_input(
                "StaticKdTree point count exceeds u32::MAX",
            ));
        }
        layout.validate_finite()?;

        let mut coords_by_id = Vec::with_capacity(n);
        let mut entries = Vec::with_capacity(n);
        match layout.as_euclidean_storage() {
            EuclideanStorageRef::Soa2d { x, y } if D == 2 => {
                for i in 0..n {
                    let mut c = [0.0_f64; D];
                    c[0] = x[i].to_f64();
                    c[1] = y[i].to_f64();
                    coords_by_id.push(c);
                    entries.push(KdEntry {
                        coords: c,
                        index: i as u32,
                    });
                }
            }
            EuclideanStorageRef::Soa3d { x, y, z } if D == 3 => {
                for i in 0..n {
                    let mut c = [0.0_f64; D];
                    c[0] = x[i].to_f64();
                    c[1] = y[i].to_f64();
                    c[2] = z[i].to_f64();
                    coords_by_id.push(c);
                    entries.push(KdEntry {
                        coords: c,
                        index: i as u32,
                    });
                }
            }
            EuclideanStorageRef::FlatNd { coords, dims, .. } if dims == D => {
                for i in 0..n {
                    let mut c = [0.0_f64; D];
                    let base = i * D;
                    for d in 0..D {
                        c[d] = coords[base + d].to_f64();
                    }
                    coords_by_id.push(c);
                    entries.push(KdEntry {
                        coords: c,
                        index: i as u32,
                    });
                }
            }
            _ => {
                return Err(CoreError::dimension_mismatch(
                    D,
                    layout.dims(),
                    "StaticKdTree::from_euclidean",
                ));
            }
        }

        let mut tree = Self {
            nodes: Vec::new(),
            entries,
            coords_by_id,
            _marker: PhantomData,
        };
        if n > 0 {
            tree.nodes.reserve((2 * n / LEAF_SIZE).max(8));
            tree.build_node(0, n);
        }
        Ok(tree)
    }
}

impl<S: Scalar> StaticKdTree<2, S> {
    /// Builds a 2D KD-tree directly from an SoA [`Layout2d<S>`].
    pub fn from_layout(layout: &Layout2d<S>) -> CoreResult<Self> {
        let n = layout.len();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_input(
                "StaticKdTree point count exceeds u32::MAX",
            ));
        }
        layout.validate_finite()?;

        let xs = layout.x();
        let ys = layout.y();
        let mut coords_by_id = Vec::with_capacity(n);
        let mut entries = Vec::with_capacity(n);
        for i in 0..n {
            let c = [xs[i].to_f64(), ys[i].to_f64()];
            coords_by_id.push(c);
            entries.push(KdEntry {
                coords: c,
                index: i as u32,
            });
        }

        let mut tree = Self {
            nodes: Vec::new(),
            entries,
            coords_by_id,
            _marker: PhantomData,
        };
        if n > 0 {
            tree.nodes.reserve((2 * n / LEAF_SIZE).max(8));
            tree.build_node(0, n);
        }
        Ok(tree)
    }
}

impl<S: Scalar> StaticKdTree<3, S> {
    /// Builds a 3D KD-tree directly from an SoA [`Layout3d<S>`].
    pub fn from_layout_3d(layout: &Layout3d<S>) -> CoreResult<Self> {
        Self::from_euclidean(layout)
    }
}
