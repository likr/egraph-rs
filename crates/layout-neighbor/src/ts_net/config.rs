use egraph_core::{CoreError, CoreResult};
use egraph_neighborhood::{PerplexityConfig, SymmetrizationMode};

pub(super) const MIN_LAMBDA_SUM: f64 = 1e-12;

/// Hyperparameters governing a single optimization stage of [`TsNetSolver`](super::TsNetSolver).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct TsNetStage {
    /// Number of gradient descent iterations in this stage (`default: 250`).
    pub iterations: usize,
    /// Early exaggeration multiplier applied to high-dimensional probabilities $P_{ij}$ (`default: 1.0`).
    pub exaggeration: f64,
    /// Weight of the KL divergence gradient term (`default: 1.0`).
    pub lambda_kl: f64,
    /// Weight of the quadratic origin-compression gradient term (`default: 0.0`).
    pub lambda_c: f64,
    /// Weight of the repulsive entropy gradient term (`default: 0.0`).
    pub lambda_r: f64,
}

impl Default for TsNetStage {
    fn default() -> Self {
        Self {
            iterations: 250,
            exaggeration: 1.0,
            lambda_kl: 1.0,
            lambda_c: 0.0,
            lambda_r: 0.0,
        }
    }
}

impl TsNetStage {
    /// Creates a new [`TsNetStage`] initialized with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the number of gradient descent iterations in this stage.
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Sets the early exaggeration multiplier applied to high-dimensional probabilities $P_{ij}$.
    pub fn with_exaggeration(mut self, exaggeration: f64) -> Self {
        self.exaggeration = exaggeration;
        self
    }

    /// Sets the weight `lambda_kl` of the KL divergence gradient term.
    pub fn with_lambda_kl(mut self, lambda_kl: f64) -> Self {
        self.lambda_kl = lambda_kl;
        self
    }

    /// Sets the weight `lambda_c` of the origin-compression gradient term.
    pub fn with_lambda_c(mut self, lambda_c: f64) -> Self {
        self.lambda_c = lambda_c;
        self
    }

    /// Sets the weight `lambda_r` of the repulsive entropy gradient term.
    pub fn with_lambda_r(mut self, lambda_r: f64) -> Self {
        self.lambda_r = lambda_r;
        self
    }

    /// Validates stage weights and exaggeration multiplier.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.exaggeration.is_finite() || self.exaggeration < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "stage exaggeration must be finite and >= 0.0, got {}",
                self.exaggeration
            )));
        }
        for (name, val) in [
            ("lambda_kl", self.lambda_kl),
            ("lambda_c", self.lambda_c),
            ("lambda_r", self.lambda_r),
        ] {
            if !val.is_finite() || val < 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "stage {name} must be finite and >= 0.0, got {val}"
                )));
            }
        }
        Ok(())
    }
}

/// Space-independent hyperparameters for [`TsNetSolver`](super::TsNetSolver), defaulting to
/// the Pure 1+2-Hop adaptive preset.
///
/// Repulsion approximation settings (`theta`, FFT grid parameters, auto thresholds) live on
/// concrete backend types (`Exact`, `BarnesHut`, `FitFft`, `LinearFft`, `Auto2d`, `Auto3d`).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct TsNetConfig {
    /// Multiplier $c$ for node-wise 2-hop adaptive perplexity (`0.9`).
    pub perplexity_c: f64,
    /// Lower clamp for node-wise 2-hop adaptive perplexity (`5.0`).
    pub min_perplexity: f64,
    /// Upper clamp for node-wise 2-hop adaptive perplexity (`1200.0`).
    pub max_perplexity: f64,
    /// Exponent applied to normalized neighbor distances `(d / max_d)^power` (`2.0`).
    pub power: f64,
    /// Maximum binary-search iterations for conditional entropy matching (`50`).
    pub sigma_iters: usize,
    /// Convergence tolerance for perplexity binary search (`1e-5`).
    pub sigma_tolerance: f64,
    /// Symmetrization strategy for high-dimensional probabilities (`StandardTsne`).
    pub symmetrization: SymmetrizationMode,
    /// Ordered optimization stages (defaults to the 3 canonical Pure 1+2-Hop stages).
    pub stages: Vec<TsNetStage>,
    /// Optional explicit base gradient descent learning rate (`None` uses `max(10.0, N / learning_rate_divisor)`).
    pub learning_rate: Option<f64>,
    /// Divisor applied to $N$ in automatic learning-rate scaling (`3.0`).
    pub learning_rate_divisor: f64,
    /// Enables per-vertex Jacobi preconditioning of the learning rate
    /// `lr_i = stage_lr / max(1.0, N * P_{i.})` (`true` by default).
    pub jacobi_precondition: bool,
    /// Momentum factor in `[0.0, 1.0)` (`0.8`).
    pub momentum: f64,
    /// Entropy softening constant $\varepsilon_r > 0$ (`0.05`).
    pub epsilon_r: f64,
    /// Deterministic explicit RNG seed (`42`).
    pub seed: u64,
}

impl Default for TsNetConfig {
    fn default() -> Self {
        Self::pure_2hop()
    }
}

impl TsNetConfig {
    /// Creates a new [`TsNetConfig`] initialized with the [`pure_2hop`](Self::pure_2hop) defaults.
    pub fn new() -> Self {
        Self::pure_2hop()
    }

    /// Returns the 3 canonical optimization stages used by Pure 1+2-Hop Fit-tsNET.
    pub fn canonical_stages() -> Vec<TsNetStage> {
        vec![
            TsNetStage {
                iterations: 250,
                exaggeration: 4.0,
                lambda_kl: 1.0,
                lambda_c: 0.1,
                lambda_r: 0.0,
            },
            TsNetStage {
                iterations: 250,
                exaggeration: 1.0,
                lambda_kl: 1.0,
                lambda_c: 0.01,
                lambda_r: 0.0,
            },
            TsNetStage {
                iterations: 500,
                exaggeration: 1.0,
                lambda_kl: 1.0,
                lambda_c: 0.0,
                lambda_r: 0.0,
            },
        ]
    }

    /// Returns the Pure 1+2-Hop node-wise adaptive Fit-tsNET preset configuration.
    pub fn pure_2hop() -> Self {
        Self {
            perplexity_c: 0.9,
            min_perplexity: 5.0,
            max_perplexity: 1200.0,
            power: 2.0,
            sigma_iters: 50,
            sigma_tolerance: 1e-5,
            symmetrization: SymmetrizationMode::StandardTsne,
            stages: Self::canonical_stages(),
            learning_rate: None,
            learning_rate_divisor: 3.0,
            jacobi_precondition: true,
            momentum: 0.8,
            epsilon_r: 0.05,
            seed: 42,
        }
    }

    /// Sets the node-wise 2-hop adaptive perplexity multiplier `perplexity_c`.
    pub fn with_perplexity_c(mut self, perplexity_c: f64) -> Self {
        self.perplexity_c = perplexity_c;
        self
    }

    /// Sets the minimum adaptive perplexity clamp `min_perplexity`.
    pub fn with_min_perplexity(mut self, min_perplexity: f64) -> Self {
        self.min_perplexity = min_perplexity;
        self
    }

    /// Sets the maximum adaptive perplexity clamp `max_perplexity`.
    pub fn with_max_perplexity(mut self, max_perplexity: f64) -> Self {
        self.max_perplexity = max_perplexity;
        self
    }

    /// Sets the normalized distance exponent `power`.
    pub fn with_power(mut self, power: f64) -> Self {
        self.power = power;
        self
    }

    /// Sets the maximum binary-search iterations `sigma_iters`.
    pub fn with_sigma_iters(mut self, sigma_iters: usize) -> Self {
        self.sigma_iters = sigma_iters;
        self
    }

    /// Sets the perplexity binary-search tolerance `sigma_tolerance`.
    pub fn with_sigma_tolerance(mut self, sigma_tolerance: f64) -> Self {
        self.sigma_tolerance = sigma_tolerance;
        self
    }

    /// Sets the probability symmetrization strategy.
    pub fn with_symmetrization(mut self, symmetrization: SymmetrizationMode) -> Self {
        self.symmetrization = symmetrization;
        self
    }

    /// Replaces the optimization schedule stages.
    pub fn with_stages(mut self, stages: Vec<TsNetStage>) -> Self {
        self.stages = stages;
        self
    }

    /// Sets an explicit base learning rate (`None` enables automatic $O(N)$ scaling).
    pub fn with_learning_rate(mut self, learning_rate: Option<f64>) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Sets the automatic learning-rate divisor applied to $N$.
    pub fn with_learning_rate_divisor(mut self, learning_rate_divisor: f64) -> Self {
        self.learning_rate_divisor = learning_rate_divisor;
        self
    }

    /// Enables or disables per-vertex Jacobi preconditioning.
    pub fn with_jacobi_precondition(mut self, jacobi_precondition: bool) -> Self {
        self.jacobi_precondition = jacobi_precondition;
        self
    }

    /// Sets the velocity momentum factor in `[0.0, 1.0)`.
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Sets the entropy softening constant `epsilon_r`.
    pub fn with_epsilon_r(mut self, epsilon_r: f64) -> Self {
        self.epsilon_r = epsilon_r;
        self
    }

    /// Sets the deterministic RNG seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Resolves the effective base learning rate for a graph with `num_nodes` vertices:
    /// explicit `Some(lr)` or automatic $O(N)$ scaling `10.0_f64.max((num_nodes as f64) / self.learning_rate_divisor)`.
    #[inline]
    pub fn resolve_learning_rate(&self, num_nodes: usize) -> f64 {
        self.learning_rate
            .unwrap_or_else(|| 10.0_f64.max((num_nodes as f64) / self.learning_rate_divisor))
    }

    /// Returns the optimization stages defined by this configuration.
    #[inline]
    pub fn stages(&self) -> &[TsNetStage] {
        &self.stages
    }

    /// Validates all hyperparameters and stage definitions.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.perplexity_c.is_finite() || self.perplexity_c <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "perplexity_c must be finite and > 0.0, got {}",
                self.perplexity_c
            )));
        }
        if !self.min_perplexity.is_finite() || self.min_perplexity <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "min_perplexity must be finite and > 0.0, got {}",
                self.min_perplexity
            )));
        }
        if !self.max_perplexity.is_finite() || self.max_perplexity < self.min_perplexity {
            return Err(CoreError::invalid_input(format!(
                "max_perplexity ({}) must be finite and >= min_perplexity ({})",
                self.max_perplexity, self.min_perplexity
            )));
        }
        if !self.power.is_finite() || self.power <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "power must be finite and > 0.0, got {}",
                self.power
            )));
        }
        if self.sigma_iters == 0 {
            return Err(CoreError::invalid_input("sigma_iters must be >= 1"));
        }
        if !self.sigma_tolerance.is_finite() || self.sigma_tolerance <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "sigma_tolerance must be finite and > 0.0, got {}",
                self.sigma_tolerance
            )));
        }
        for stage in &self.stages {
            stage.validate()?;
        }
        if let Some(lr) = self.learning_rate
            && (!lr.is_finite() || lr <= 0.0)
        {
            return Err(CoreError::invalid_input(format!(
                "learning_rate must be finite and > 0.0, got {lr}"
            )));
        }
        if !self.learning_rate_divisor.is_finite() || self.learning_rate_divisor <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "learning_rate_divisor must be finite and > 0.0, got {}",
                self.learning_rate_divisor
            )));
        }
        if !self.momentum.is_finite() || self.momentum < 0.0 || self.momentum >= 1.0 {
            return Err(CoreError::invalid_input(format!(
                "momentum must be finite and in [0.0, 1.0), got {}",
                self.momentum
            )));
        }
        if !self.epsilon_r.is_finite() || self.epsilon_r <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "epsilon_r must be finite and > 0.0, got {}",
                self.epsilon_r
            )));
        }
        Ok(())
    }

    pub(super) fn perplexity_config(&self) -> PerplexityConfig {
        PerplexityConfig::new()
            .with_power(self.power)
            .with_sigma_iters(self.sigma_iters)
            .with_sigma_tolerance(self.sigma_tolerance)
            .with_symmetrization(self.symmetrization)
    }
}

#[inline]
pub(super) fn compute_stage_lr(
    stage: &TsNetStage,
    n: usize,
    learning_rate: f64,
    momentum: f64,
) -> f64 {
    let n_f64 = n as f64;
    let lambda_sum = (stage.lambda_kl + stage.lambda_c + stage.lambda_r).max(MIN_LAMBDA_SUM);
    let mut stage_lr = if stage.lambda_c > 0.0 {
        let max_stable_lr = (1.0 + momentum) * 0.85 * lambda_sum * n_f64 / stage.lambda_c;
        learning_rate.min(max_stable_lr)
    } else {
        learning_rate
    };
    if stage.lambda_r > 0.0 {
        let tsnet_lr_cap = 50.0_f64.min(2.0 * n_f64);
        stage_lr = stage_lr.min(tsnet_lr_cap);
    } else if stage.lambda_c == 0.0 || n < 100 {
        stage_lr = stage_lr.min(2.0 * n_f64);
    }
    stage_lr
}
