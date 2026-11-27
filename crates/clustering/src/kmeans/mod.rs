//! Pure-Rust deterministic K-Means++ clustering over flat row-major feature matrices.

use egraph_core::{CoreError, CoreResult};

use crate::clusterer::{ClusterResult, Clusterer};

mod solver;

#[cfg(test)]
mod tests;

pub use solver::kmeans_plus_plus;

/// Configuration for [`KMeansPlusPlus`] and [`kmeans_plus_plus`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct KMeansConfig {
    /// Target number of clusters (`>= 1`, `<= n`).
    pub k: usize,
    /// Maximum number of Lloyd refinement iterations per initialization.
    pub max_iterations: usize,
    /// Convergence tolerance on maximum centroid Euclidean movement.
    pub tol: f64,
    /// Number of independent K-Means++ initializations; the lowest-inertia run is returned.
    pub n_init: usize,
    /// Deterministic PRNG seed for K-Means++ $D^2$ center sampling.
    pub seed: u64,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            k: 2,
            max_iterations: 100,
            tol: 1e-4,
            n_init: 1,
            seed: 42,
        }
    }
}

impl KMeansConfig {
    /// Creates a new [`KMeansConfig`] initialized to [`Self::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the target number of clusters `k`.
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Sets the maximum number of Lloyd refinement iterations `max_iterations`.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets the centroid movement convergence tolerance `tol`.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Sets the number of independent K-Means++ initializations `n_init`.
    pub fn with_n_init(mut self, n_init: usize) -> Self {
        self.n_init = n_init;
        self
    }

    /// Sets the deterministic PRNG seed `seed`.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validates all K-Means++ hyperparameters independent of input point count.
    pub fn validate(&self) -> CoreResult<()> {
        if self.k == 0 {
            return Err(CoreError::invalid_input("k must be >= 1"));
        }
        if self.max_iterations == 0 {
            return Err(CoreError::invalid_input("max_iterations must be >= 1"));
        }
        if self.n_init == 0 {
            return Err(CoreError::invalid_input("n_init must be >= 1"));
        }
        if !self.tol.is_finite() {
            return Err(CoreError::non_finite("KMeansConfig::tol"));
        }
        if self.tol < 0.0 {
            return Err(CoreError::invalid_input("tol must be >= 0.0"));
        }
        Ok(())
    }
}

/// Output of [`KMeansPlusPlus::run`] and [`kmeans_plus_plus`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct KMeansResult {
    /// Canonical cluster assignment in `0..k` for each input point.
    pub labels: Vec<u32>,
    /// Flat row-major centroids of shape `(k, dims)`.
    pub centroids: Vec<f64>,
    /// Number of target clusters `k`.
    pub k: usize,
    /// Feature dimensionality `dims` of each point and centroid row.
    pub dims: usize,
    /// Total within-cluster sum of squared Euclidean distances to assigned centroids.
    pub inertia: f64,
    /// Number of Lloyd iterations executed in the best initialization.
    pub iterations: usize,
    /// Whether the centroid shift converged within `config.tol`.
    pub converged: bool,
}

impl From<KMeansResult> for ClusterResult {
    fn from(res: KMeansResult) -> Self {
        Self::new(
            res.labels,
            res.k,
            Some(res.inertia),
            res.iterations,
            res.converged,
        )
    }
}

/// Deterministic K-Means++ feature-space clusterer implementing [`Clusterer`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct KMeansPlusPlus {
    config: KMeansConfig,
}

impl KMeansPlusPlus {
    /// Creates a new [`KMeansPlusPlus`] clusterer with the given configuration.
    pub fn new(config: KMeansConfig) -> Self {
        Self { config }
    }

    /// Returns a reference to the underlying [`KMeansConfig`].
    #[inline]
    pub fn config(&self) -> &KMeansConfig {
        &self.config
    }

    /// Replaces the configuration with `config`.
    pub fn with_config(mut self, config: KMeansConfig) -> Self {
        self.config = config;
        self
    }

    /// Sets the target number of clusters `k`.
    pub fn with_k(mut self, k: usize) -> Self {
        self.config.k = k;
        self
    }

    /// Sets the maximum number of Lloyd refinement iterations `max_iterations`.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.config.max_iterations = max_iterations;
        self
    }

    /// Sets the centroid movement convergence tolerance `tol`.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.config.tol = tol;
        self
    }

    /// Sets the number of independent K-Means++ initializations `n_init`.
    pub fn with_n_init(mut self, n_init: usize) -> Self {
        self.config.n_init = n_init;
        self
    }

    /// Sets the deterministic PRNG seed `seed`.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    /// Runs K-Means++ clustering and returns the full [`KMeansResult`] including centroids and inertia.
    pub fn run(&self, points_flat: &[f64], n: usize, dims: usize) -> CoreResult<KMeansResult> {
        kmeans_plus_plus(points_flat, n, dims, &self.config)
    }
}

impl Clusterer for KMeansPlusPlus {
    fn cluster(
        &self,
        points_flat: &[f64],
        num_points: usize,
        dims: usize,
    ) -> CoreResult<ClusterResult> {
        self.run(points_flat, num_points, dims).map(Into::into)
    }
}
