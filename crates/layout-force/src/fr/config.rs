use egraph_core::{CoreError, CoreResult};
#[cfg(doc)]
use egraph_spatial_trees::ArenaQuadtree2d;

/// Spatial repulsion backend for [`FruchtermanReingoldSolver`](super::FruchtermanReingoldSolver).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FrBackend {
    /// Exact $O(N^2)$ all-pairs repulsion.
    #[default]
    Exact,
    /// $O(N \log N)$ Barnes-Hut quadtree approximation via [`ArenaQuadtree2d`].
    BarnesHut,
    /// Fruchterman–Reingold (1991) Fig. 2 $2k \times 2k$ grid with strict $d < 2k$ cutoff.
    Grid,
}

/// Hyperparameters for [`FruchtermanReingoldSolver`](super::FruchtermanReingoldSolver).
#[derive(Debug, Clone, Copy, PartialEq)]
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
    /// Optional central gravity coefficient pulling vertices toward `(0, 0)` (`default: 0.0`).
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
