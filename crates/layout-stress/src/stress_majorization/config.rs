use egraph_core::{CoreError, CoreResult};

/// Hyperparameters for [`StressMajorizationSolver`](super::StressMajorizationSolver).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct StressMajorizationConfig {
    /// Relative stress convergence tolerance (`default: 1e-4`).
    pub epsilon: f64,
    /// Maximum outer stress majorization iterations (`default: 100`).
    pub max_iterations: usize,
    /// Inner Jacobi-PCG residual tolerance (`default: 1e-6`).
    pub pcg_tolerance: f64,
    /// Optional explicit inner PCG iteration cap (`None` uses `(2 * n).max(16)`).
    pub pcg_max_iterations: Option<usize>,
    /// Minimum pair separation below which majorizing forces are skipped (`default: 1e-4`).
    pub min_norm: f64,
}

impl Default for StressMajorizationConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-4,
            max_iterations: 100,
            pcg_tolerance: 1e-6,
            pcg_max_iterations: None,
            min_norm: 1e-4,
        }
    }
}

impl StressMajorizationConfig {
    /// Creates a new [`StressMajorizationConfig`] initialized to [`Self::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the relative stress convergence tolerance `epsilon`.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Sets the maximum outer stress majorization iterations `max_iterations`.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets the inner Jacobi-PCG residual tolerance `pcg_tolerance`.
    pub fn with_pcg_tolerance(mut self, pcg_tolerance: f64) -> Self {
        self.pcg_tolerance = pcg_tolerance;
        self
    }

    /// Sets the optional explicit inner PCG iteration cap `pcg_max_iterations`.
    pub fn with_pcg_max_iterations(mut self, pcg_max_iterations: Option<usize>) -> Self {
        self.pcg_max_iterations = pcg_max_iterations;
        self
    }

    /// Sets the minimum pair separation `min_norm`.
    pub fn with_min_norm(mut self, min_norm: f64) -> Self {
        self.min_norm = min_norm;
        self
    }

    /// Validates stress majorization parameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.epsilon.is_finite() || self.epsilon < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "epsilon must be finite and >= 0.0, got {}",
                self.epsilon
            )));
        }
        if !self.pcg_tolerance.is_finite() || self.pcg_tolerance < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "pcg_tolerance must be finite and >= 0.0, got {}",
                self.pcg_tolerance
            )));
        }
        if let Some(m) = self.pcg_max_iterations
            && m == 0
        {
            return Err(CoreError::invalid_input(
                "pcg_max_iterations must be >= 1 when specified",
            ));
        }
        if !self.min_norm.is_finite() || self.min_norm <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "min_norm must be finite and > 0.0, got {}",
                self.min_norm
            )));
        }
        Ok(())
    }
}
