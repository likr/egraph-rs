use std::cmp::Ordering;
use std::collections::BinaryHeap;

use egraph_core::{CoreError, CoreResult, Scalar};

pub(super) const LEAF_SIZE: usize = 16;

#[derive(Debug, Clone, Copy)]
pub(super) struct KdEntry<const D: usize> {
    pub(super) coords: [f64; D],
    pub(super) index: u32,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct KdNode<const D: usize> {
    pub(super) min_bounds: [f64; D],
    pub(super) max_bounds: [f64; D],
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) split_dim: usize,
    pub(super) split_val: f64,
    pub(super) left: u32,
    pub(super) right: u32,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Candidate {
    pub(super) dist_sq: f64,
    pub(super) index: u32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist_sq.total_cmp(&other.dist_sq) == Ordering::Equal && self.index == other.index
    }
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist_sq
            .total_cmp(&other.dist_sq)
            .then_with(|| self.index.cmp(&other.index))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Reusable scratch workspace for [`super::StaticKdTree`] and [`super::DynamicKdTree`] $k$-NN queries
/// without per-query allocations.
#[derive(Debug, Clone, Default)]
pub struct KdQueryScratch {
    pub(super) heap: BinaryHeap<Candidate>,
    pub(super) stack: Vec<(u32, f64)>,
    pub(super) query_buf: Vec<f64>,
}

impl KdQueryScratch {
    /// Allocates scratch buffers pre-sized for `k`-nearest queries.
    pub fn with_capacity(k: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(k),
            stack: Vec::with_capacity(64),
            query_buf: Vec::new(),
        }
    }
}

#[inline]
pub(super) fn convert_query<const D: usize, S: Scalar>(query: &[S; D]) -> CoreResult<[f64; D]> {
    let mut out = [0.0_f64; D];
    for d in 0..D {
        if !query[d].is_finite() {
            return Err(CoreError::non_finite("StaticKdTree query"));
        }
        out[d] = query[d].to_f64();
    }
    Ok(out)
}

#[inline]
pub(super) fn dist_sq<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    let mut sum = 0.0_f64;
    for d in 0..D {
        let diff = a[d] - b[d];
        sum += diff * diff;
    }
    sum
}

#[inline]
pub(super) fn box_dist_sq<const D: usize>(q: &[f64; D], node: &KdNode<D>) -> f64 {
    let mut sum = 0.0_f64;
    for d in 0..D {
        let v = q[d];
        let diff = if v < node.min_bounds[d] {
            node.min_bounds[d] - v
        } else if v > node.max_bounds[d] {
            v - node.max_bounds[d]
        } else {
            0.0
        };
        sum += diff * diff;
    }
    sum
}
