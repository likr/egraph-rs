use std::marker::PhantomData;

use egraph_core::{CoreError, CoreResult, EuclideanLayout, EuclideanStorageRef, LayoutNd, Scalar};

use super::types::LEAF_SIZE;

#[derive(Debug, Clone, Copy)]
pub(super) struct DynamicKdNode {
    pub(super) bounds_offset: usize,
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) split_dim: usize,
    pub(super) split_val: f64,
    pub(super) left: u32,
    pub(super) right: u32,
}

/// Flat median-split KD-tree with runtime coordinate dimension $D \ge 1$.
///
/// Mirrors [`super::StaticKdTree`] for arbitrary `LayoutNd<S>` dimensions without compile-time
/// const generics. Distance computations and bounding-box pruning always execute in `f64`,
/// breaking ties deterministically by ascending original point index for exact parity with
/// $O(N^2)$ brute-force search.
#[derive(Debug, Clone)]
pub struct DynamicKdTree<S: Scalar = f64> {
    pub(super) dims: usize,
    pub(super) num_nodes: usize,
    pub(super) nodes: Vec<DynamicKdNode>,
    /// Contiguous `[min_0..min_{D-1}, max_0..max_{D-1}]` bounds per node.
    pub(super) node_bounds: Vec<f64>,
    pub(super) entry_indices: Vec<u32>,
    pub(super) entry_coords: Vec<f64>,
    pub(super) coords_by_id: Vec<f64>,
    _marker: PhantomData<S>,
}

impl<S: Scalar> DynamicKdTree<S> {
    /// Builds a runtime-dimension KD-tree from a [`LayoutNd<S>`].
    pub fn from_layout_nd(layout: &LayoutNd<S>) -> CoreResult<Self> {
        Self::from_flat(layout.coords(), layout.num_nodes(), layout.dims())
    }

    /// Builds a runtime-dimension KD-tree from any [`EuclideanLayout`].
    pub fn from_euclidean<L: EuclideanLayout<Scalar = S> + ?Sized>(layout: &L) -> CoreResult<Self> {
        match layout.as_euclidean_storage() {
            EuclideanStorageRef::Soa2d { x, y } => {
                Self::from_soa_axes(&[x, y], "DynamicKdTree::from_euclidean")
            }
            EuclideanStorageRef::Soa3d { x, y, z } => {
                Self::from_soa_axes(&[x, y, z], "DynamicKdTree::from_euclidean")
            }
            EuclideanStorageRef::FlatNd {
                coords,
                num_nodes,
                dims,
            } => Self::from_flat(coords, num_nodes, dims),
        }
    }

    fn from_soa_axes(axes: &[&[S]], context: &'static str) -> CoreResult<Self> {
        let dims = axes.len();
        let num_nodes = axes[0].len();
        for axis in axes.iter().skip(1) {
            if axis.len() != num_nodes {
                return Err(CoreError::dimension_mismatch(
                    num_nodes,
                    axis.len(),
                    context,
                ));
            }
        }
        if u32::try_from(num_nodes).is_err() {
            return Err(CoreError::invalid_input(
                "DynamicKdTree point count exceeds u32::MAX",
            ));
        }
        let expected = num_nodes
            .checked_mul(dims)
            .ok_or_else(|| CoreError::invalid_input("num_nodes * dims overflows usize"))?;
        let mut coords_by_id = Vec::with_capacity(expected);
        for i in 0..num_nodes {
            for axis in axes {
                let v = axis[i];
                if !v.is_finite() {
                    return Err(CoreError::non_finite(context));
                }
                coords_by_id.push(v.to_f64());
            }
        }
        Self::build_from_coords_by_id(coords_by_id, num_nodes, dims)
    }

    /// Builds a runtime-dimension KD-tree from row-major flat coordinates `[num_nodes * dims]`.
    pub fn from_flat(coords: &[S], num_nodes: usize, dims: usize) -> CoreResult<Self> {
        if dims == 0 {
            return Err(CoreError::invalid_input(
                "DynamicKdTree dimension dims must be >= 1",
            ));
        }
        let expected = num_nodes
            .checked_mul(dims)
            .ok_or_else(|| CoreError::invalid_input("num_nodes * dims overflows usize"))?;
        if coords.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                coords.len(),
                "DynamicKdTree::from_flat",
            ));
        }
        if u32::try_from(num_nodes).is_err() {
            return Err(CoreError::invalid_input(
                "DynamicKdTree point count exceeds u32::MAX",
            ));
        }

        let mut coords_by_id = Vec::with_capacity(expected);
        for &v in coords {
            if !v.is_finite() {
                return Err(CoreError::non_finite("DynamicKdTree::from_flat"));
            }
            coords_by_id.push(v.to_f64());
        }
        Self::build_from_coords_by_id(coords_by_id, num_nodes, dims)
    }

    fn build_from_coords_by_id(
        coords_by_id: Vec<f64>,
        num_nodes: usize,
        dims: usize,
    ) -> CoreResult<Self> {
        let expected = num_nodes * dims;
        let entry_indices: Vec<u32> = (0..num_nodes).map(|i| i as u32).collect();
        let entry_coords = vec![0.0_f64; expected];

        let mut tree = Self {
            dims,
            num_nodes,
            nodes: Vec::new(),
            node_bounds: Vec::new(),
            entry_indices,
            entry_coords,
            coords_by_id,
            _marker: PhantomData,
        };

        if num_nodes > 0 {
            let est_nodes = (2 * num_nodes / LEAF_SIZE).max(8);
            tree.nodes.reserve(est_nodes);
            if let Some(est_bounds) = est_nodes.checked_mul(2 * dims) {
                tree.node_bounds.reserve(est_bounds);
            }
            tree.build_node(0, num_nodes);

            for (slot, &id) in tree.entry_indices.iter().enumerate() {
                let src = (id as usize) * dims;
                let dst = slot * dims;
                tree.entry_coords[dst..dst + dims]
                    .copy_from_slice(&tree.coords_by_id[src..src + dims]);
            }
        }

        Ok(tree)
    }

    /// Returns the coordinate dimensionality $D \ge 1$.
    #[inline]
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Returns the number of points stored in the tree.
    #[inline]
    pub fn len(&self) -> usize {
        self.num_nodes
    }

    /// Returns `true` if the tree stores zero points.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_nodes == 0
    }

    /// Returns the row-major `f64` coordinate slice `[num_nodes * dims]` in original point order.
    #[inline]
    pub fn coords_by_id(&self) -> &[f64] {
        &self.coords_by_id
    }

    /// Returns the `f64` coordinate slice of length `dims` for original point `point_idx`.
    pub fn point(&self, point_idx: usize) -> CoreResult<&[f64]> {
        if point_idx >= self.num_nodes {
            return Err(CoreError::index_out_of_bounds(
                point_idx,
                self.num_nodes,
                "DynamicKdTree::point",
            ));
        }
        let base = point_idx * self.dims;
        Ok(&self.coords_by_id[base..base + self.dims])
    }

    fn build_node(&mut self, start: usize, end: usize) -> u32 {
        let dims = self.dims;
        let bounds_offset = self.node_bounds.len();
        let first_id = self.entry_indices[start] as usize;
        let first_slice = &self.coords_by_id[first_id * dims..(first_id + 1) * dims];

        // Reserve [min_bounds (dims), max_bounds (dims)] contiguously for this node.
        self.node_bounds.extend_from_slice(first_slice);
        self.node_bounds.extend_from_slice(first_slice);

        let (mins, maxs) =
            self.node_bounds[bounds_offset..bounds_offset + 2 * dims].split_at_mut(dims);
        for &id in &self.entry_indices[start + 1..end] {
            let pt = &self.coords_by_id[(id as usize) * dims..(id as usize + 1) * dims];
            for d in 0..dims {
                mins[d] = mins[d].min(pt[d]);
                maxs[d] = maxs[d].max(pt[d]);
            }
        }

        let node_idx = self.nodes.len() as u32;
        if end - start <= LEAF_SIZE {
            self.nodes.push(DynamicKdNode {
                bounds_offset,
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
        for d in 0..dims {
            let spread = maxs[d] - mins[d];
            if spread > max_spread {
                max_spread = spread;
                split_dim = d;
            }
        }

        // Collapse identical/duplicate clusters into a single leaf to avoid empty splits.
        if max_spread <= 0.0 {
            self.nodes.push(DynamicKdNode {
                bounds_offset,
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
        let local_mid = mid - start;
        let coords = &self.coords_by_id;
        self.entry_indices[start..end].select_nth_unstable_by(local_mid, |&a, &b| {
            let va = coords[(a as usize) * dims + split_dim];
            let vb = coords[(b as usize) * dims + split_dim];
            va.total_cmp(&vb).then_with(|| a.cmp(&b))
        });
        let split_val = coords[(self.entry_indices[mid] as usize) * dims + split_dim];

        self.nodes.push(DynamicKdNode {
            bounds_offset,
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
}
