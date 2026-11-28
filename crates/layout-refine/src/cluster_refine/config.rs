use egraph_core::{CoreError, CoreResult};

/// Geometric clustering backend used by [`ClusterRefineSolver`](super::ClusterRefineSolver).
#[non_exhaustive]
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
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterRefineConfig {
    /// Geometric clustering and vertex shift strategy (`default: ClusterRefineMethod::Kmeans`).
    pub method: ClusterRefineMethod,
    /// Maximum outer refinement iterations (`default: 10`).
    pub max_iterations: usize,
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
            max_iterations: 10,
            kmeans_n_init: 5,
            kmeans_max_iters: 100,
            shift_step: 1.0,
            margin: 0.1,
            seed: 42,
        }
    }
}

impl ClusterRefineConfig {
    /// Creates a new [`ClusterRefineConfig`] initialized to [`Self::default`].
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the geometric clustering backend `method`.
    #[inline]
    pub fn with_method(mut self, method: ClusterRefineMethod) -> Self {
        self.method = method;
        self
    }

    /// Sets the maximum outer refinement `max_iterations`.
    #[inline]
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets the number of K-Means++ initializations `kmeans_n_init` per round.
    #[inline]
    pub fn with_kmeans_n_init(mut self, kmeans_n_init: usize) -> Self {
        self.kmeans_n_init = kmeans_n_init;
        self
    }

    /// Sets the maximum Lloyd iterations `kmeans_max_iters` per initialization.
    #[inline]
    pub fn with_kmeans_max_iters(mut self, kmeans_max_iters: usize) -> Self {
        self.kmeans_max_iters = kmeans_max_iters;
        self
    }

    /// Sets the interpolation factor `shift_step` toward the computed target coordinate.
    #[inline]
    pub fn with_shift_step(mut self, shift_step: f64) -> Self {
        self.shift_step = shift_step;
        self
    }

    /// Sets the fractional Voronoi boundary clearance `margin`.
    #[inline]
    pub fn with_margin(mut self, margin: f64) -> Self {
        self.margin = margin;
        self
    }

    /// Sets the deterministic explicit PRNG `seed`.
    #[inline]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

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
