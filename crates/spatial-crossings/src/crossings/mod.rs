//! Exact $O((|E|+K)\log|E|)$ sweep-line edge-crossing analysis, non-Euclidean geodesic
//! crossings (torus, spherical, hyperbolic), and $O(M)$ deterministic edge-pair sampling.

use egraph_core::{CoreResult, CsrGraph, Layout2d, Scalar};

mod active_tree;
mod band;
mod band_reorder;
mod collector;
mod cross_pred;
mod double_double;
mod exact_limbs;
mod geometry;
mod hyperbolic;
mod pair_set;
mod sampling;
mod shifted_coords;
mod spherical;
mod sweep;
#[cfg(test)]
mod tests;
mod torus;
mod torus_clip;
mod torus_exact;
mod torus_repair;

use band_reorder::finalize_pairs;
use geometry::{build_segments, crossing_angle_metrics, extract_csr_edges};
pub use hyperbolic::{crossing_edges_hyperbolic, hyperbolic_crossing_number};
pub use sampling::{sampled_crossing_number, sampled_crossing_number_edges};
pub use spherical::{crossing_edges_spherical, spherical_crossing_number};
use sweep::{sweep_segments, sweep_segments_with};
pub use torus::{
    TorusLayoutView, torus_crossing_edges, torus_crossing_edges_from_edges, torus_crossing_number,
    torus_edge_segments, torus_layout_edge_segments,
};

/// Details of a single crossing pair between two non-adjacent edges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrossingPair {
    pub edge_a: usize,
    pub edge_b: usize,
    pub u1: u32,
    pub v1: u32,
    pub u2: u32,
    pub v2: u32,
    /// Acute crossing angle in `[0, pi/2]` radians, or `None` for zero-length segments.
    pub angle_rad: Option<f64>,
    /// $\cos^2(\theta)$ of the crossing angle (`0.0` when angle is undefined).
    pub cos_sq: f64,
}

/// Summary statistics of all edge crossings in a 2D Euclidean layout.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CrossingSummary {
    pub count: usize,
    /// Sum of $\cos^2(\theta)$ over crossing pairs (matches legacy `crossing_angle`).
    pub angle_cos_sq_sum: f64,
    /// Minimum acute crossing angle across all defined crossing pairs.
    pub min_angle_rad: Option<f64>,
}

/// Crossing pair of periodic `[0, 1)^2` torus edge sub-segments with physical aspect-aware angles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TorusCrossing {
    pub edge_a: usize,
    pub edge_b: usize,
    pub u1: u32,
    pub v1: u32,
    pub u2: u32,
    pub v2: u32,
    /// `[x1, y1, x2, y2]` coordinates of the first crossing sub-segment.
    pub seg_a: [f64; 4],
    /// `[x1, y1, x2, y2]` coordinates of the second crossing sub-segment.
    pub seg_b: [f64; 4],
    /// Acute crossing angle in `[0, pi/2]` radians in physical `(Lx, Ly)` coordinates.
    pub angle_rad: Option<f64>,
    /// $\cos^2(\theta)$ of the crossing angle (`0.0` when angle is undefined).
    pub cos_sq: f64,
}

/// Exact $O((|E|+K)\log|E|)$ x-sweep edge-crossing counter, angle evaluator, and pair finder.
pub struct SweepLineCrossings;

impl SweepLineCrossings {
    /// Counts all crossing edge pairs in `graph` under `layout`.
    pub fn count<W, S: Scalar>(graph: &CsrGraph<W>, layout: &Layout2d<S>) -> CoreResult<usize> {
        let edges = extract_csr_edges(graph);
        Self::count_edges(&edges, layout)
    }

    /// Computes crossing count and crossing angle statistics in `graph` under `layout`.
    pub fn summarize<W, S: Scalar>(
        graph: &CsrGraph<W>,
        layout: &Layout2d<S>,
    ) -> CoreResult<CrossingSummary> {
        let edges = extract_csr_edges(graph);
        Self::summarize_edges(&edges, layout)
    }

    /// Finds all crossing edge pairs in `graph` under `layout`, sorted deterministically.
    pub fn find_crossings<W, S: Scalar>(
        graph: &CsrGraph<W>,
        layout: &Layout2d<S>,
    ) -> CoreResult<Vec<CrossingPair>> {
        let edges = extract_csr_edges(graph);
        Self::find_edge_crossings(&edges, layout)
    }

    /// Counts all crossing pairs among `edges` under `layout`.
    pub fn count_edges<S: Scalar>(edges: &[(u32, u32)], layout: &Layout2d<S>) -> CoreResult<usize> {
        let segs = build_segments(edges, layout)?;
        let mut count = 0usize;
        sweep_segments_with(&segs, |_, _| {
            count += 1;
        });
        Ok(count)
    }

    /// Summarizes crossing count and crossing angles among `edges` under `layout`.
    pub fn summarize_edges<S: Scalar>(
        edges: &[(u32, u32)],
        layout: &Layout2d<S>,
    ) -> CoreResult<CrossingSummary> {
        let segs = build_segments(edges, layout)?;
        let mut count = 0usize;
        let mut cos_sq_sum = 0.0_f64;
        let mut cos_sq_comp = 0.0_f64;
        let mut min_angle_rad: Option<f64> = None;

        sweep_segments_with(&segs, |a, b| {
            count += 1;
            let (angle_rad, cos_sq) = crossing_angle_metrics(&segs[a], &segs[b]);
            // Neumaier compensated summation for deterministic O(eps) accumulation.
            let t = cos_sq_sum + cos_sq;
            cos_sq_comp += if cos_sq_sum.abs() >= cos_sq.abs() {
                (cos_sq_sum - t) + cos_sq
            } else {
                (cos_sq - t) + cos_sq_sum
            };
            cos_sq_sum = t;
            if let Some(ang) = angle_rad {
                min_angle_rad = Some(match min_angle_rad {
                    Some(prev) => prev.min(ang),
                    None => ang,
                });
            }
        });

        Ok(CrossingSummary {
            count,
            angle_cos_sq_sum: cos_sq_sum + cos_sq_comp,
            min_angle_rad,
        })
    }

    /// Finds all crossing pairs among `edges` under `layout`, sorted by `(edge_a, edge_b)`.
    pub fn find_edge_crossings<S: Scalar>(
        edges: &[(u32, u32)],
        layout: &Layout2d<S>,
    ) -> CoreResult<Vec<CrossingPair>> {
        let segs = build_segments(edges, layout)?;
        let reported = sweep_segments(&segs);
        Ok(finalize_pairs(&reported, &segs))
    }
}
