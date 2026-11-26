//! Pure-Rust deterministic K-Means++ clustering over flat row-major feature matrices.

mod solver;

#[cfg(test)]
mod tests;

pub use solver::kmeans_plus_plus;

/// Configuration for [`kmeans_plus_plus`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KMeansConfig {
    /// Target number of clusters (`>= 1`, `<= n`).
    pub k: usize,
    /// Maximum number of Lloyd refinement iterations per initialization.
    pub max_iters: usize,
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
            max_iters: 100,
            tol: 1e-4,
            n_init: 1,
            seed: 42,
        }
    }
}

/// Output of [`kmeans_plus_plus`].
#[derive(Debug, Clone, PartialEq)]
pub struct KMeansResult {
    /// Canonical cluster assignment in `0..k` for each input point.
    pub labels: Vec<u32>,
    /// Flat row-major centroids of shape `(k, dims)`.
    pub centroids: Vec<f64>,
    /// Total within-cluster sum of squared Euclidean distances to assigned centroids.
    pub inertia: f64,
    /// Number of Lloyd iterations executed in the best initialization.
    pub iterations: usize,
    /// Whether the centroid shift converged within `config.tol`.
    pub converged: bool,
}
