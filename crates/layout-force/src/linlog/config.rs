use egraph_core::{CoreError, CoreResult};

/// Repulsion vertex-mass weighting model for [`LinLogSolver`](super::LinLogSolver).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LinLogRepulsion {
    /// Edge-repulsion LinLog ($m_u = \deg_w(u)$, Noack 2007/2009 modularity connection).
    #[default]
    EdgeRepulsion,
    /// Node-repulsion LinLog ($m_u = 1.0$ for all vertices).
    NodeRepulsion,
}

/// Hyperparameters for [`LinLogSolver`](super::LinLogSolver).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinLogConfig {
    /// Total number of optimization iterations (`default: 200`).
    pub iterations: usize,
    /// Repulsion mass weighting model (`default: LinLogRepulsion::EdgeRepulsion`).
    pub repulsion_model: LinLogRepulsion,
    /// Attraction force exponent $a$ in $\|d\|^a$ (Noack 2009 `a`; `a = 0.0` is unit pull, `r_polylog = a + 1`, `default: 0.0`).
    pub attraction_exponent: f64,
    /// Repulsion force exponent $r$ in $\|d\|^r$ (Noack 2009 `r`; `r = -1.0` is $1/d$ Coulomb, `default: -1.0`).
    pub repulsion_exponent: f64,
    /// Barnes-Hut quadtree opening angle threshold $\theta \ge 0$ (`default: 1.2`).
    pub theta: f64,
    /// Whether to use $O(N \log N)$ Barnes-Hut quadtree repulsion (`default: true`).
    pub use_barnes_hut: bool,
    /// Gravitational pull coefficient toward mass barycenter $b(p)$ (`default: 0.01`).
    pub gravity: f64,
    /// Exponent applied to edge weights (`default: 1.0`).
    pub edge_weight_influence: f64,
    /// Optional per-step vertex displacement cap (`None` adapts to `50.0 * max(1.0, sqrt(N / 1000.0))`, `default: None`).
    pub max_displacement: Option<f64>,
    /// Deterministic explicit RNG seed for collapsed-layout symmetry breaking (`default: 42`).
    pub seed: u64,
}

impl Default for LinLogConfig {
    fn default() -> Self {
        Self {
            iterations: 200,
            repulsion_model: LinLogRepulsion::EdgeRepulsion,
            attraction_exponent: 0.0,
            repulsion_exponent: -1.0,
            theta: 1.2,
            use_barnes_hut: true,
            gravity: 0.01,
            edge_weight_influence: 1.0,
            max_displacement: None,
            seed: 42,
        }
    }
}

impl LinLogConfig {
    /// Validates all LinLog hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.attraction_exponent.is_finite() {
            return Err(CoreError::invalid_input(format!(
                "attraction_exponent must be finite, got {}",
                self.attraction_exponent
            )));
        }
        if !self.repulsion_exponent.is_finite() {
            return Err(CoreError::invalid_input(format!(
                "repulsion_exponent must be finite, got {}",
                self.repulsion_exponent
            )));
        }
        if self.attraction_exponent <= self.repulsion_exponent {
            return Err(CoreError::invalid_input(format!(
                "attraction_exponent ({}) must be strictly greater than repulsion_exponent ({})",
                self.attraction_exponent, self.repulsion_exponent
            )));
        }
        if !self.theta.is_finite() || self.theta < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "theta must be finite and >= 0.0, got {}",
                self.theta
            )));
        }
        if !self.gravity.is_finite() || self.gravity < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "gravity must be finite and >= 0.0, got {}",
                self.gravity
            )));
        }
        if !self.edge_weight_influence.is_finite() || self.edge_weight_influence < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "edge_weight_influence must be finite and >= 0.0, got {}",
                self.edge_weight_influence
            )));
        }
        if let Some(max_disp) = self.max_displacement
            && (!max_disp.is_finite() || max_disp <= 0.0)
        {
            return Err(CoreError::invalid_input(format!(
                "max_displacement must be finite and > 0.0, got {max_disp}"
            )));
        }
        Ok(())
    }
}
