use egraph_core::{CoreError, CoreResult};

/// Hyperparameters for [`YifanHuSolver`](super::YifanHuSolver).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct YifanHuConfig {
    /// Maximum iterations executed at each multilevel scale (`default: 200`).
    pub max_iterations_per_level: usize,
    /// Initial step size multiplier relative to level optimal distance (`default: 1.0`).
    pub step_size: f64,
    /// Adaptive step cooling/expansion ratio in `(0.0, 1.0)` (`default: 0.9`).
    pub step_ratio: f64,
    /// Repulsion strength coefficient $C$ in $C K^2 / d$ (`default: 0.2`).
    pub relative_strength: f64,
    /// Optional explicit finest-level optimal edge length $K_0$ (`None` defaults to `1.0`).
    pub optimal_distance: Option<f64>,
    /// Barnes-Hut spatial tree opening angle threshold $\theta \ge 0$ (`default: 1.2`).
    pub theta: f64,
    /// Stop coarsening once vertex count drops to or below this size (`default: 30`).
    pub min_coarsen_size: usize,
    /// Stop coarsening if `coarse_nodes / fine_nodes >= min_coarsen_ratio` (`default: 0.75`).
    pub min_coarsen_ratio: f64,
    /// Convergence displacement tolerance relative to $K_{\text{level}}$ (`default: 1e-4`).
    pub convergence_tolerance: f64,
    /// Deterministic explicit RNG seed for prolongation jitter (`default: 42`).
    pub seed: u64,
}

impl Default for YifanHuConfig {
    fn default() -> Self {
        Self {
            max_iterations_per_level: 200,
            step_size: 1.0,
            step_ratio: 0.9,
            relative_strength: 0.2,
            optimal_distance: None,
            theta: 1.2,
            min_coarsen_size: 30,
            min_coarsen_ratio: 0.75,
            convergence_tolerance: 1e-4,
            seed: 42,
        }
    }
}

impl YifanHuConfig {
    /// Creates a new [`YifanHuConfig`] initialized with [`Self::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum iterations per multilevel scale.
    pub fn with_max_iterations_per_level(mut self, max_iterations_per_level: usize) -> Self {
        self.max_iterations_per_level = max_iterations_per_level;
        self
    }

    /// Sets the initial step size multiplier relative to level optimal distance.
    pub fn with_step_size(mut self, step_size: f64) -> Self {
        self.step_size = step_size;
        self
    }

    /// Sets the adaptive step cooling/expansion ratio in `(0.0, 1.0)`.
    pub fn with_step_ratio(mut self, step_ratio: f64) -> Self {
        self.step_ratio = step_ratio;
        self
    }

    /// Sets the repulsion strength coefficient `relative_strength`.
    pub fn with_relative_strength(mut self, relative_strength: f64) -> Self {
        self.relative_strength = relative_strength;
        self
    }

    /// Sets the optional explicit finest-level optimal distance $K_0$.
    pub fn with_optimal_distance(mut self, optimal_distance: Option<f64>) -> Self {
        self.optimal_distance = optimal_distance;
        self
    }

    /// Sets the Barnes-Hut spatial tree opening angle threshold `theta`.
    pub fn with_theta(mut self, theta: f64) -> Self {
        self.theta = theta;
        self
    }

    /// Sets the minimum graph size threshold at which coarsening stops.
    pub fn with_min_coarsen_size(mut self, min_coarsen_size: usize) -> Self {
        self.min_coarsen_size = min_coarsen_size;
        self
    }

    /// Sets the coarsening reduction ratio threshold in `(0.0, 1.0]`.
    pub fn with_min_coarsen_ratio(mut self, min_coarsen_ratio: f64) -> Self {
        self.min_coarsen_ratio = min_coarsen_ratio;
        self
    }

    /// Sets the convergence displacement tolerance relative to $K_{\text{level}}$.
    pub fn with_convergence_tolerance(mut self, convergence_tolerance: f64) -> Self {
        self.convergence_tolerance = convergence_tolerance;
        self
    }

    /// Sets the deterministic explicit RNG seed for prolongation jitter.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validates all Yifan Hu hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.step_size.is_finite() || self.step_size <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "step_size must be finite and > 0.0, got {}",
                self.step_size
            )));
        }
        if !self.step_ratio.is_finite() || self.step_ratio <= 0.0 || self.step_ratio >= 1.0 {
            return Err(CoreError::invalid_input(format!(
                "step_ratio must be finite and in (0.0, 1.0), got {}",
                self.step_ratio
            )));
        }
        if !self.relative_strength.is_finite() || self.relative_strength < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "relative_strength must be finite and >= 0.0, got {}",
                self.relative_strength
            )));
        }
        if let Some(k) = self.optimal_distance
            && (!k.is_finite() || k <= 0.0)
        {
            return Err(CoreError::invalid_input(format!(
                "optimal_distance must be finite and > 0.0, got {k}"
            )));
        }
        if !self.theta.is_finite() || self.theta < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "theta must be finite and >= 0.0, got {}",
                self.theta
            )));
        }
        if self.min_coarsen_size == 0 {
            return Err(CoreError::invalid_input("min_coarsen_size must be >= 1"));
        }
        if !self.min_coarsen_ratio.is_finite()
            || self.min_coarsen_ratio <= 0.0
            || self.min_coarsen_ratio > 1.0
        {
            return Err(CoreError::invalid_input(format!(
                "min_coarsen_ratio must be finite and in (0.0, 1.0], got {}",
                self.min_coarsen_ratio
            )));
        }
        if !self.convergence_tolerance.is_finite() || self.convergence_tolerance < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "convergence_tolerance must be finite and >= 0.0, got {}",
                self.convergence_tolerance
            )));
        }
        Ok(())
    }
}
