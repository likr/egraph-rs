use egraph_core::{CoreError, CoreResult};

/// Geometric clustering backend used by [`ClusterRefineSolver`](super::ClusterRefineSolver).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ClusterRefineMethod {
    /// Cai et al. (2024) Algorithm 1: $k$-means++ geometric partitioning + Voronoi exit shift.
    #[default]
    Kmeans,
    /// Cai et al. (2024) Algorithm 2: average-linkage (UPGMA) HAC + nearest-target-vertex shift.
    Hac,
}

/// Configuration for [`ClusterRefineSolver`](super::ClusterRefineSolver) and
/// [`refine_cluster_layout`](super::refine_cluster_layout).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClusterRefineConfig {
    /// Geometric clustering and vertex shift strategy (`default: ClusterRefineMethod::Kmeans`).
    pub method: ClusterRefineMethod,
    /// Maximum outer refinement rounds (`default: 10`).
    pub max_rounds: usize,
    /// Number of K-Means++ initializations per round (`default: 5`).
    pub kmeans_n_init: usize,
    /// Maximum Lloyd iterations per K-Means++ initialization (`default: 100`).
    pub kmeans_max_iters: usize,
    /// Interpolation factor toward the computed target coordinate (`default: 1.0`).
    pub shift_step: f64,
    /// Fractional clearance past the Voronoi boundary toward the target centroid (`default: 0.1`).
    pub margin: f64,
    /// Deterministic explicit RNG seed (`default: 42`).
    pub seed: u64,
}

impl Default for ClusterRefineConfig {
    fn default() -> Self {
        Self {
            method: ClusterRefineMethod::Kmeans,
            max_rounds: 10,
            kmeans_n_init: 5,
            kmeans_max_iters: 100,
            shift_step: 1.0,
            margin: 0.1,
            seed: 42,
        }
    }
}

impl ClusterRefineConfig {
    /// Validates all cluster refinement hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if self.kmeans_n_init == 0 {
            return Err(CoreError::invalid_input("kmeans_n_init must be >= 1"));
        }
        if self.kmeans_max_iters == 0 {
            return Err(CoreError::invalid_input("kmeans_max_iters must be >= 1"));
        }
        if !self.shift_step.is_finite() {
            return Err(CoreError::non_finite("ClusterRefineConfig::shift_step"));
        }
        if self.shift_step < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "shift_step must be >= 0.0, got {}",
                self.shift_step
            )));
        }
        if !self.margin.is_finite() {
            return Err(CoreError::non_finite("ClusterRefineConfig::margin"));
        }
        if self.margin < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "margin must be >= 0.0, got {}",
                self.margin
            )));
        }
        Ok(())
    }
}
