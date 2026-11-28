use egraph_core::{CoreError, CoreResult};

/// Hyperparameters for [`GemSolver`](super::GemSolver).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct GemConfig {
    /// Desired ideal edge length $E_{\mathrm{des}}$ (`default: 128.0`).
    pub edge_length: f64,
    /// Gravitational attraction coefficient $\gamma$ toward graph barycenter (`default: 1.0 / 16.0`).
    pub gravity: f64,
    /// Initial per-vertex local temperature $T_{\mathrm{init}}$ (`default: 128.0`).
    pub t_init: f64,
    /// Maximum per-vertex local temperature $T_{\max}$ (`default: 256.0`).
    pub t_max: f64,
    /// Global mean-temperature convergence threshold $T_{\min}$ (`default: 3.0`).
    pub t_min: f64,
    /// Optional maximum iterations (`None` uses `min(4 * max(1, N), 400)`, `default: None`).
    pub max_iterations: Option<usize>,
    /// Oscillation detection opening angle $\alpha_o \in (0, \pi]$ (`default: PI`).
    pub oscillation_angle: f64,
    /// Rotation detection opening angle $\alpha_r \in (0, \pi]$ (`default: PI / 3.0`).
    pub rotation_angle: f64,
    /// Oscillation sensitivity $\sigma_o \ge 0$ (`default: 1.0 / 3.0`).
    pub oscillation_sensitivity: f64,
    /// Optional rotation sensitivity $\sigma_r \ge 0$ (`None` uses $1 / (2 \max(1, N))$, `default: None`).
    pub rotation_sensitivity: Option<f64>,
    /// Random impulse perturbation magnitude $\delta \in [-\text{perturbation}, \text{perturbation}]^2$ (`default: 32.0`).
    pub perturbation: f64,
    /// Explicit deterministic RNG seed (`default: 42`).
    pub seed: u64,
}

impl Default for GemConfig {
    fn default() -> Self {
        Self {
            edge_length: 128.0,
            gravity: 1.0 / 16.0,
            t_init: 128.0,
            t_max: 256.0,
            t_min: 3.0,
            max_iterations: None,
            oscillation_angle: std::f64::consts::PI,
            rotation_angle: std::f64::consts::PI / 3.0,
            oscillation_sensitivity: 1.0 / 3.0,
            rotation_sensitivity: None,
            perturbation: 32.0,
            seed: 42,
        }
    }
}

impl GemConfig {
    /// Creates a new [`GemConfig`] initialized with [`Self::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the desired ideal edge length.
    pub fn with_edge_length(mut self, edge_length: f64) -> Self {
        self.edge_length = edge_length;
        self
    }

    /// Sets the gravitational attraction coefficient toward the barycenter.
    pub fn with_gravity(mut self, gravity: f64) -> Self {
        self.gravity = gravity;
        self
    }

    /// Sets the initial per-vertex local temperature `t_init`.
    pub fn with_t_init(mut self, t_init: f64) -> Self {
        self.t_init = t_init;
        self
    }

    /// Sets the maximum per-vertex local temperature `t_max`.
    pub fn with_t_max(mut self, t_max: f64) -> Self {
        self.t_max = t_max;
        self
    }

    /// Sets the global mean-temperature convergence threshold `t_min`.
    pub fn with_t_min(mut self, t_min: f64) -> Self {
        self.t_min = t_min;
        self
    }

    /// Sets an optional maximum iteration cap.
    pub fn with_max_iterations(mut self, max_iterations: Option<usize>) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets the oscillation detection opening angle in `(0, PI]`.
    pub fn with_oscillation_angle(mut self, oscillation_angle: f64) -> Self {
        self.oscillation_angle = oscillation_angle;
        self
    }

    /// Sets the rotation detection opening angle in `(0, PI]`.
    pub fn with_rotation_angle(mut self, rotation_angle: f64) -> Self {
        self.rotation_angle = rotation_angle;
        self
    }

    /// Sets the oscillation sensitivity `sigma_o >= 0`.
    pub fn with_oscillation_sensitivity(mut self, oscillation_sensitivity: f64) -> Self {
        self.oscillation_sensitivity = oscillation_sensitivity;
        self
    }

    /// Sets the optional rotation sensitivity `sigma_r >= 0`.
    pub fn with_rotation_sensitivity(mut self, rotation_sensitivity: Option<f64>) -> Self {
        self.rotation_sensitivity = rotation_sensitivity;
        self
    }

    /// Sets the random impulse perturbation magnitude.
    pub fn with_perturbation(mut self, perturbation: f64) -> Self {
        self.perturbation = perturbation;
        self
    }

    /// Sets the explicit deterministic RNG seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validates all GEM hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.edge_length.is_finite() {
            return Err(CoreError::non_finite("GemConfig::edge_length"));
        }
        if self.edge_length <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "edge_length must be > 0.0, got {}",
                self.edge_length
            )));
        }
        if !self.gravity.is_finite() {
            return Err(CoreError::non_finite("GemConfig::gravity"));
        }
        if self.gravity < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "gravity must be >= 0.0, got {}",
                self.gravity
            )));
        }
        if !self.t_init.is_finite() {
            return Err(CoreError::non_finite("GemConfig::t_init"));
        }
        if self.t_init < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "t_init must be >= 0.0, got {}",
                self.t_init
            )));
        }
        if !self.t_max.is_finite() {
            return Err(CoreError::non_finite("GemConfig::t_max"));
        }
        if self.t_max <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "t_max must be > 0.0, got {}",
                self.t_max
            )));
        }
        if !self.t_min.is_finite() {
            return Err(CoreError::non_finite("GemConfig::t_min"));
        }
        if self.t_min < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "t_min must be >= 0.0, got {}",
                self.t_min
            )));
        }
        if !self.oscillation_angle.is_finite() {
            return Err(CoreError::non_finite("GemConfig::oscillation_angle"));
        }
        if self.oscillation_angle <= 0.0 || self.oscillation_angle > std::f64::consts::PI {
            return Err(CoreError::invalid_input(format!(
                "oscillation_angle must be in (0, PI], got {}",
                self.oscillation_angle
            )));
        }
        if !self.rotation_angle.is_finite() {
            return Err(CoreError::non_finite("GemConfig::rotation_angle"));
        }
        if self.rotation_angle <= 0.0 || self.rotation_angle > std::f64::consts::PI {
            return Err(CoreError::invalid_input(format!(
                "rotation_angle must be in (0, PI], got {}",
                self.rotation_angle
            )));
        }
        if !self.oscillation_sensitivity.is_finite() {
            return Err(CoreError::non_finite("GemConfig::oscillation_sensitivity"));
        }
        if self.oscillation_sensitivity < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "oscillation_sensitivity must be >= 0.0, got {}",
                self.oscillation_sensitivity
            )));
        }
        if let Some(sr) = self.rotation_sensitivity {
            if !sr.is_finite() {
                return Err(CoreError::non_finite("GemConfig::rotation_sensitivity"));
            }
            if sr < 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "rotation_sensitivity must be >= 0.0, got {sr}"
                )));
            }
        }
        if !self.perturbation.is_finite() {
            return Err(CoreError::non_finite("GemConfig::perturbation"));
        }
        if self.perturbation < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "perturbation must be >= 0.0, got {}",
                self.perturbation
            )));
        }
        Ok(())
    }
}
