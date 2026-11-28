use egraph_core::{CoreError, CoreResult};

/// Hyperparameters for [`MaxentStressSolver`](super::MaxentStressSolver).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct MaxentStressConfig {
    /// Optional explicit repulsion exponent $q > -2.0$ (`default: None`).
    ///
    /// When `None`, defaults to `0.0` (logarithmic entropy), or automatically switches to `0.8`
    /// when strictly more than `30%` of vertices have 1-hop degree `1` (Gansner et al. 2013 §4.2).
    pub q: Option<f64>,
    /// Initial annealing repulsion weight $\alpha_0 \ge 0.0$ (`default: 1.0`).
    pub alpha_init: f64,
    /// Multiplicative decay factor in `(0.0, 1.0]` applied between annealing stages (`default: 0.3`).
    pub alpha_decay: f64,
    /// Minimum floor for $\alpha$ across annealing stages (`default: 0.008`).
    pub alpha_min: f64,
    /// Number of outer annealing stages (`default: 5`).
    pub stages: usize,
    /// Maximum number of majorization steps per annealing stage (`default: 50`).
    pub steps_per_stage: usize,
    /// Relative position change convergence tolerance per stage (`default: 1e-3`).
    pub tol: f64,
    /// Relative residual tolerance passed to [`AmgPcgSolver`](egraph_linalg_amg::AmgPcgSolver) (`default: 0.1`).
    pub cg_tol: f64,
    /// Maximum PCG iterations per linear solve (`default: 10`).
    pub cg_max_iters: usize,
    /// Barnes-Hut opening angle $\theta \ge 0.0$ for `ArenaQuadtree2d` (`Layout2d`) or
    /// `ArenaOctree3d` (`Layout3d`) (`default: 1.2`).
    pub theta: f64,
    /// Whether to use Barnes-Hut spatial-tree approximation (`true` for `Layout2d` / `Layout3d`)
    /// or exact $O(N^2)$ repulsion (`false`, required for `LayoutNd`) (`default: true`).
    pub use_barnes_hut: bool,
    /// Softening squared distance $\varepsilon^2 > 0.0$ in repulsive denominator (`default: 1e-6`).
    pub eps_sq: f64,
    /// Deterministic PRNG seed for AMG tie-breaking and collapsed-layout initialization (`default: 42`).
    pub seed: u64,
}

impl Default for MaxentStressConfig {
    fn default() -> Self {
        Self {
            q: None,
            alpha_init: 1.0,
            alpha_decay: 0.3,
            alpha_min: 0.008,
            stages: 5,
            steps_per_stage: 50,
            tol: 1e-3,
            cg_tol: 0.1,
            cg_max_iters: 10,
            theta: 1.2,
            use_barnes_hut: true,
            eps_sq: 1e-6,
            seed: 42,
        }
    }
}

impl MaxentStressConfig {
    /// Creates a new [`MaxentStressConfig`] initialized to [`Self::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the optional explicit repulsion exponent `q`.
    pub fn with_q(mut self, q: Option<f64>) -> Self {
        self.q = q;
        self
    }

    /// Sets the initial annealing repulsion weight `alpha_init`.
    pub fn with_alpha_init(mut self, alpha_init: f64) -> Self {
        self.alpha_init = alpha_init;
        self
    }

    /// Sets the multiplicative annealing decay factor `alpha_decay`.
    pub fn with_alpha_decay(mut self, alpha_decay: f64) -> Self {
        self.alpha_decay = alpha_decay;
        self
    }

    /// Sets the minimum floor `alpha_min` for annealing repulsion weight.
    pub fn with_alpha_min(mut self, alpha_min: f64) -> Self {
        self.alpha_min = alpha_min;
        self
    }

    /// Sets the number of outer annealing stages `stages`.
    pub fn with_stages(mut self, stages: usize) -> Self {
        self.stages = stages;
        self
    }

    /// Sets the maximum majorization steps per stage `steps_per_stage`.
    pub fn with_steps_per_stage(mut self, steps_per_stage: usize) -> Self {
        self.steps_per_stage = steps_per_stage;
        self
    }

    /// Sets the relative position change convergence tolerance `tol`.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Sets the relative residual tolerance `cg_tol` passed to `AmgPcgSolver`.
    pub fn with_cg_tol(mut self, cg_tol: f64) -> Self {
        self.cg_tol = cg_tol;
        self
    }

    /// Sets the maximum PCG iterations per linear solve `cg_max_iters`.
    pub fn with_cg_max_iters(mut self, cg_max_iters: usize) -> Self {
        self.cg_max_iters = cg_max_iters;
        self
    }

    /// Sets the Barnes-Hut opening angle `theta`.
    pub fn with_theta(mut self, theta: f64) -> Self {
        self.theta = theta;
        self
    }

    /// Sets whether Barnes-Hut spatial-tree repulsion (`true`) or exact $O(N^2)$ repulsion
    /// (`false`) is used.
    pub fn with_use_barnes_hut(mut self, use_barnes_hut: bool) -> Self {
        self.use_barnes_hut = use_barnes_hut;
        self
    }

    /// Sets the softening squared distance `eps_sq`.
    pub fn with_eps_sq(mut self, eps_sq: f64) -> Self {
        self.eps_sq = eps_sq;
        self
    }

    /// Sets the deterministic PRNG `seed`.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validates all Maxent-Stress configuration parameters.
    pub fn validate(&self) -> CoreResult<()> {
        if let Some(qv) = self.q {
            if !qv.is_finite() {
                return Err(CoreError::non_finite("MaxentStressConfig::q"));
            }
            if qv <= -2.0 {
                return Err(CoreError::invalid_input(format!(
                    "MaxentStressConfig::q must be > -2.0, got {qv}"
                )));
            }
        }
        if !self.alpha_init.is_finite() {
            return Err(CoreError::non_finite("MaxentStressConfig::alpha_init"));
        }
        if self.alpha_init < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "MaxentStressConfig::alpha_init must be >= 0.0, got {}",
                self.alpha_init
            )));
        }
        if !self.alpha_decay.is_finite() {
            return Err(CoreError::non_finite("MaxentStressConfig::alpha_decay"));
        }
        if self.alpha_decay <= 0.0 || self.alpha_decay > 1.0 {
            return Err(CoreError::invalid_input(format!(
                "MaxentStressConfig::alpha_decay must be in (0.0, 1.0], got {}",
                self.alpha_decay
            )));
        }
        if !self.alpha_min.is_finite() {
            return Err(CoreError::non_finite("MaxentStressConfig::alpha_min"));
        }
        if self.alpha_min < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "MaxentStressConfig::alpha_min must be >= 0.0, got {}",
                self.alpha_min
            )));
        }
        if !self.tol.is_finite() {
            return Err(CoreError::non_finite("MaxentStressConfig::tol"));
        }
        if self.tol < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "MaxentStressConfig::tol must be >= 0.0, got {}",
                self.tol
            )));
        }
        if !self.cg_tol.is_finite() {
            return Err(CoreError::non_finite("MaxentStressConfig::cg_tol"));
        }
        if self.cg_tol < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "MaxentStressConfig::cg_tol must be >= 0.0, got {}",
                self.cg_tol
            )));
        }
        if self.cg_max_iters == 0 {
            return Err(CoreError::invalid_input(
                "MaxentStressConfig::cg_max_iters must be >= 1",
            ));
        }
        if !self.theta.is_finite() {
            return Err(CoreError::non_finite("MaxentStressConfig::theta"));
        }
        if self.theta < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "MaxentStressConfig::theta must be >= 0.0, got {}",
                self.theta
            )));
        }
        if !self.eps_sq.is_finite() {
            return Err(CoreError::non_finite("MaxentStressConfig::eps_sq"));
        }
        if self.eps_sq <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "MaxentStressConfig::eps_sq must be > 0.0, got {}",
                self.eps_sq
            )));
        }
        Ok(())
    }
}
