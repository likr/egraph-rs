//! Flat `Vec`-backed arena Barnes-Hut 3D octree for Coulomb, MaxEnt, power-law, KL, and entropy forces.
//!
//! Eliminates pointer-chasing boxed-node allocations by storing tree nodes and
//! coincident-point chains inside contiguous reusable `Vec` arenas. All geometry, bounding
//! boxes, center-of-mass moments, and forces are accumulated in `f64` for numerical stability
//! when `S = f32`.

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

/// Explicit DFS stack capacity (`7 * MAX_DEPTH + 1 = 225 <= 256` for 8-child octrees).
const TRAVERSAL_STACK_CAP: usize = 256;

#[derive(Debug, Clone, Copy)]
struct ArenaNode3d {
    children: [u32; 8],
    center: [f64; 3],
    half_span: f64,
    com: [f64; 3],
    total_mass: f64,
    /// Head of intrusive `next_in_leaf` point list when leaf (`u32::MAX` if empty or internal).
    leaf_head: u32,
    point_count: u32,
    is_leaf: bool,
}

impl ArenaNode3d {
    #[inline]
    fn new(center: [f64; 3], half_span: f64) -> Self {
        Self {
            children: [u32::MAX; 8],
            center,
            half_span,
            com: [0.0; 3],
            total_mass: 0.0,
            leaf_head: u32::MAX,
            point_count: 0,
            is_leaf: true,
        }
    }

    #[inline]
    fn octant(&self, x: f64, y: f64, z: f64) -> usize {
        let bx = usize::from(x >= self.center[0]);
        let by = usize::from(y >= self.center[1]) << 1;
        let bz = usize::from(z >= self.center[2]) << 2;
        bx | by | bz
    }

    #[inline]
    fn is_unsplitable_at_depth(&self, depth: usize) -> bool {
        if depth >= MAX_DEPTH {
            return true;
        }
        // Child half-span equals the center offset `quarter = 0.5 * half_span`.
        let quarter = self.half_span * 0.5;
        quarter <= 0.0
            || !quarter.is_finite()
            || (self.center[0] - quarter) == self.center[0]
            || (self.center[0] + quarter) == self.center[0]
            || (self.center[1] - quarter) == self.center[1]
            || (self.center[1] + quarter) == self.center[1]
            || (self.center[2] - quarter) == self.center[2]
            || (self.center[2] + quarter) == self.center[2]
    }
}

/// Flat arena-allocated Barnes-Hut octree in 3D Euclidean space.
///
/// Repeated calls to [`ArenaOctree3d::clear`] and [`ArenaOctree3d::build`] retain backing
/// `Vec` capacities so iterative 3D layouts perform zero per-step heap allocations.
#[derive(Debug, Clone)]
pub struct ArenaOctree3d<S: Scalar = f64> {
    nodes: Vec<ArenaNode3d>,
    coords_x: Vec<f64>,
    coords_y: Vec<f64>,
    coords_z: Vec<f64>,
    masses: Vec<f64>,
    leaf_of_point: Vec<u32>,
    next_in_leaf: Vec<u32>,
    _marker: PhantomData<S>,
}

impl<S: Scalar> Default for ArenaOctree3d<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Scalar> ArenaOctree3d<S> {
    /// Creates an empty reusable 3D octree arena.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            coords_x: Vec::new(),
            coords_y: Vec::new(),
            coords_z: Vec::new(),
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
        self.coords_z.clear();
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
