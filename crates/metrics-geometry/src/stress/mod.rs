//! BFS shortest-path utilities and closed-form / pivot-sampled Scale-Normalized Stress (`SNS`).
//!
//! Unscaled normalized stress `NS(X, D)` varies quadratically when coordinates are uniformly
//! rescaled, allowing tiny random layouts to outrank structured drawings. `SNS` eliminates
//! scale artifacts analytically via the closed-form optimal multiplier
//! $\alpha^* = \frac{\sum w_{ij} d_{ij} \|x_i - x_j\|}{\sum w_{ij} \|x_i - x_j\|^2}$
//! accumulated in `f64`.

mod bfs;
mod sampled;
mod sns;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_spaces;

pub use bfs::{BfsScratch, all_pairs_bfs, all_pairs_bfs_into, bfs_single_source};
pub use sampled::{sampled_scale_normalized_stress, sampled_scale_normalized_stress_detail};
pub use sns::{
    geodesic_distortion, scale_normalized_stress, scale_normalized_stress_detail,
    scale_normalized_stress_from_distances, scale_normalized_stress_from_distances_detail,
    scale_normalized_stress_from_pairs,
};

/// Detailed result from evaluating closed-form Scale-Normalized Stress (`SNS`).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StressEvaluation {
    /// Scale-normalized stress divided by $\sum w_{ij} d_{ij}^2$, lying in `[0.0, 1.0]`
    /// (equals mean relative squared error over evaluated pairs when $w_{ij} = d_{ij}^{-2}$).
    pub normalized_stress: f64,
    /// Unnormalized optimal-scale stress sum $\sum w_{ij} (\alpha^* \|x_i - x_j\| - d_{ij})^2$.
    pub raw_stress: f64,
    /// Closed-form optimal uniform coordinate scale factor $\alpha^*$.
    pub alpha_star: f64,
    /// Number of valid finite positive-distance pairs included in the evaluation.
    pub pair_count: usize,
    /// Sum of $\sum w_{ij} d_{ij}^2$ over evaluated pairs (equals `pair_count` for $w_{ij} = d_{ij}^{-2}$).
    pub weight_distance_sq_sum: f64,
}
