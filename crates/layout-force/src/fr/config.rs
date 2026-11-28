use egraph_core::{CoreError, CoreResult};
#[cfg(doc)]
use egraph_spatial_trees::{ArenaOctree3d, ArenaQuadtree2d};

/// Spatial repulsion backend for [`FruchtermanReingoldSolver`](super::FruchtermanReingoldSolver).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FrBackend {
    /// Exact $O(N^2)$ all-pairs repulsion (supported on all [`RiemannianLayout`](egraph_core::RiemannianLayout) targets).
    #[default]
    Exact,
    /// $O(N \log N)$ Barnes-Hut spatial tree approximation via [`ArenaQuadtree2d`] (`Layout2d`)
    /// or [`ArenaOctree3d`] (`Layout3d`).
    BarnesHut,
    /// Fruchterman–Reingold (1991) Fig. 2 $2k$ grid with strict $d < 2k$ cutoff (`Layout2d` and `Layout3d`).
    Grid,
}

/// Hyperparameters for [`FruchtermanReingoldSolver`](super::FruchtermanReingoldSolver).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct FruchtermanReingoldConfig {
    /// Total number of iterations (`default: 200`).
    pub iterations: usize,
    /// Spatial repulsion backend (`default: FrBackend::Exact`).
    pub backend: FrBackend,
    /// Optional optimal pairwise distance $k$ (`None` uses $\sqrt{\text{area} / N}$, `default: None`).
    pub k: Option<f64>,
    /// Optional canvas area (`None` uses $N$ or $k^2 N$ if `k` is specified, `default: None`).
    pub area: Option<f64>,
    /// Optional initial temperature (`None` uses $0.1 \sqrt{\text{area}}$, `default: None`).
    pub initial_temperature: Option<f64>,
    /// Optional central gravity coefficient pulling vertices toward origin (`default: 0.0`).
    pub gravity: f64,
    /// Barnes-Hut opening angle threshold $\theta \ge 0$ (`default: 1.2`).
    pub theta: f64,
    /// Deterministic explicit RNG seed for symmetry breaking (`default: 42`).
    pub seed: u64,
}

impl Default for FruchtermanReingoldConfig {
    fn default() -> Self {
        Self {
            iterations: 200,
            backend: FrBackend::Exact,
            k: None,
            area: None,
            initial_temperature: None,
            gravity: 0.0,
            theta: 1.2,
            seed: 42,
        }
    }
}

impl FruchtermanReingoldConfig {
    /// Creates a new [`FruchtermanReingoldConfig`] initialized with [`Self::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the total number of iterations.
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Sets the spatial repulsion backend.
    pub fn with_backend(mut self, backend: FrBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Sets the optional optimal pairwise distance `k`.
    pub fn with_k(mut self, k: Option<f64>) -> Self {
        self.k = k;
        self
    }

    /// Sets the optional canvas area.
    pub fn with_area(mut self, area: Option<f64>) -> Self {
        self.area = area;
        self
    }

    /// Sets the optional initial cooling temperature.
    pub fn with_initial_temperature(mut self, initial_temperature: Option<f64>) -> Self {
        self.initial_temperature = initial_temperature;
        self
    }

    /// Sets the central gravity coefficient.
    pub fn with_gravity(mut self, gravity: f64) -> Self {
        self.gravity = gravity;
        self
    }

    /// Sets the Barnes-Hut opening angle threshold `theta`.
    pub fn with_theta(mut self, theta: f64) -> Self {
        self.theta = theta;
        self
    }

    /// Sets the deterministic explicit RNG seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validates all Fruchterman–Reingold hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if let Some(k) = self.k
            && (!k.is_finite() || k <= 0.0)
        {
            return Err(CoreError::invalid_input(format!(
                "k must be finite and > 0.0, got {k}"
            )));
        }
        if let Some(area) = self.area
            && (!area.is_finite() || area <= 0.0)
        {
            return Err(CoreError::invalid_input(format!(
                "area must be finite and > 0.0, got {area}"
            )));
        }
        if let Some(t0) = self.initial_temperature
            && (!t0.is_finite() || t0 < 0.0)
        {
            return Err(CoreError::invalid_input(format!(
                "initial_temperature must be finite and >= 0.0, got {t0}"
            )));
        }
        if !self.gravity.is_finite() || self.gravity < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "gravity must be finite and >= 0.0, got {}",
                self.gravity
            )));
        }
        if !self.theta.is_finite() || self.theta < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "theta must be finite and >= 0.0, got {}",
                self.theta
            )));
        }
        Ok(())
    }
}
