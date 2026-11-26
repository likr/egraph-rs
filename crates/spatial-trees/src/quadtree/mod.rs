//! Flat `Vec`-backed arena Barnes-Hut 2D quadtree for KL and entropy repulsion forces.
//!
//! Eliminates pointer-chasing boxed-node allocations by storing tree nodes and
//! coincident-point chains inside contiguous reusable `Vec` arenas. All geometry and moments
//! are accumulated in `f64` for numerical stability when `S = f32`.

use std::marker::PhantomData;

use egraph_core::Scalar;

mod build;
mod coulomb_maxent;
mod kl_entropy;
#[cfg(test)]
mod tests;

const MAX_DEPTH: usize = 32;
const COINCIDENT_DIST_SQ: f64 = 1e-12;
const ENTROPY_MIN_DIST: f64 = 1e-6;

#[derive(Debug, Clone, Copy)]
struct ArenaNode {
    center_x: f64,
    center_y: f64,
    width: f64,
    height: f64,
    count: usize,
    total_mass: f64,
    com_x: f64,
    com_y: f64,
    /// Head of intrusive `next_in_leaf` point list when leaf (`u32::MAX` if empty or internal).
    leaf_head: u32,
    /// Base index of the 4 child quadrants (`NW, NE, SW, SE`) or `u32::MAX` if leaf.
    first_child: u32,
}

impl ArenaNode {
    #[inline]
    fn new(center_x: f64, center_y: f64, width: f64, height: f64) -> Self {
        Self {
            center_x,
            center_y,
            width,
            height,
            count: 0,
            total_mass: 0.0,
            com_x: 0.0,
            com_y: 0.0,
            leaf_head: u32::MAX,
            first_child: u32::MAX,
        }
    }

    #[inline]
    fn quadrant(&self, x: f64, y: f64) -> usize {
        let right = usize::from(x >= self.center_x);
        let bottom = usize::from(y >= self.center_y) * 2;
        right + bottom
    }

    #[inline]
    fn is_unsplitable_at_depth(&self, depth: usize) -> bool {
        if depth >= MAX_DEPTH {
            return true;
        }
        let quarter_w = self.width * 0.25;
        let quarter_h = self.height * 0.25;
        quarter_w <= 0.0
            || quarter_h <= 0.0
            || (self.center_x - quarter_w) == self.center_x
            || (self.center_x + quarter_w) == self.center_x
            || (self.center_y - quarter_h) == self.center_y
            || (self.center_y + quarter_h) == self.center_y
    }
}

/// Flat arena-allocated Barnes-Hut quadtree in 2D Euclidean space.
///
/// Repeated calls to [`ArenaQuadtree2d::clear`] and [`ArenaQuadtree2d::build`] retain backing
/// `Vec` capacities so iterative layouts perform zero per-step heap allocations.
#[derive(Debug, Clone)]
pub struct ArenaQuadtree2d<S: Scalar = f64> {
    nodes: Vec<ArenaNode>,
    coords_x: Vec<f64>,
    coords_y: Vec<f64>,
    masses: Vec<f64>,
    leaf_of_point: Vec<u32>,
    next_in_leaf: Vec<u32>,
    _marker: PhantomData<S>,
}

impl<S: Scalar> Default for ArenaQuadtree2d<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Scalar> ArenaQuadtree2d<S> {
    /// Creates an empty reusable quadtree arena.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            coords_x: Vec::new(),
            coords_y: Vec::new(),
            masses: Vec::new(),
            leaf_of_point: Vec::new(),
            next_in_leaf: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Clears all nodes and points while retaining allocated buffer capacities.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.coords_x.clear();
        self.coords_y.clear();
        self.masses.clear();
        self.leaf_of_point.clear();
        self.next_in_leaf.clear();
    }

    /// Returns the number of points indexed in the tree.
    #[inline]
    pub fn point_count(&self) -> usize {
        self.coords_x.len()
    }

    /// Returns the number of allocated arena nodes (including internal and empty leaf slots).
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}
