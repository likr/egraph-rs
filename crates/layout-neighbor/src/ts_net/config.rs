use egraph_core::{CoreError, CoreResult};
use egraph_neighborhood::{PerplexityConfig, SymmetrizationMode};
use egraph_spatial_fft::{IbFftConfig, InterpolationOrder};

pub(super) const MIN_LAMBDA_SUM: f64 = 1e-12;

/// Repulsion approximation backend used by [`TsNetSolver`](super::TsNetSolver).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TsNetBackend {
    /// Exact $O(N^2)$ pairwise Student-$t$ KL repulsion and entropy repulsion.
    Exact,
    /// $O(N \log N)$ Barnes-Hut quadtree approximation for KL and entropy repulsion.
    BarnesHut,
    /// Fused 2D FFT interpolation for KL and entropy repulsion via [`FitField2d`](egraph_spatial_fft::FitField2d).
    #[default]
    FftInterpolation,
    /// Linear-time fused 2D FFT interpolation enforcing [`InterpolationOrder::Bilinear`] ($p=1$).
    LinearFft,
}

/// Hyperparameters governing a single optimization stage of [`TsNetSolver`](super::TsNetSolver).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TsNetStage {
    /// Number of gradient descent iterations in this stage.
    pub iterations: usize,
    /// Early exaggeration multiplier applied to high-dimensional probabilities $P_{ij}$.
    pub exaggeration: f64,
    /// Weight of the KL divergence gradient term.
    pub lambda_kl: f64,
    /// Weight of the quadratic origin-compression gradient term.
    pub lambda_c: f64,
    /// Weight of the repulsive entropy gradient term.
    pub lambda_r: f64,
}

impl TsNetStage {
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

/// Configuration for [`TsNetSolver`](super::TsNetSolver), defaulting to the Pure 1+2-Hop adaptive preset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TsNetConfig {
    /// Repulsion computation backend (`FftInterpolation` by default).
    pub backend: TsNetBackend,
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
    /// Number of iterations in Stage 1 (early exaggeration, `250`).
    pub iterations_stage1: usize,
    /// Number of iterations in Stage 2 (compression relaxation, `250`).
    pub iterations_stage2: usize,
    /// Number of iterations in Stage 3 (fine-tuning / entropy repulsion, `500`).
    pub iterations_stage3: usize,
    /// Early exaggeration factor applied in Stage 1 (`4.0`).
    pub exaggeration: f64,
    /// Compression weight $\lambda_c$ in Stage 1 (`0.1`).
    pub lambda_c_stage1: f64,
    /// Compression weight $\lambda_c$ in Stage 2 (`0.01`).
    pub lambda_c_stage2: f64,
    /// Compression weight $\lambda_c$ in Stage 3 (`0.0`).
    pub lambda_c_stage3: f64,
    /// Repulsion weight $\lambda_r$ in Stage 3 (`0.0`).
    pub lambda_r_stage3: f64,
    /// Optional explicit base gradient descent learning rate (`None` uses $O(N)$ scaling `max(200.0, N / 3.0)`).
    pub learning_rate: Option<f64>,
    /// Enables per-vertex Jacobi preconditioning of the learning rate
    /// `lr_i = stage_lr / max(1.0, N * P_{i.})` (`true` by default).
    pub jacobi_precondition: bool,
    /// Momentum factor in `[0.0, 1.0)` (`0.8`).
    pub momentum: f64,
    /// Entropy softening constant $\varepsilon_r > 0$ (`0.05`).
    pub epsilon_r: f64,
    /// Barnes-Hut opening angle threshold $\theta \ge 0$ (`0.5`).
    pub theta: f64,
    /// Interpolation-based 2D FFT grid configuration (`Bilinear` $p=1$, `intervals=25`, `64..=512`).
    pub ibfft: IbFftConfig,
    /// Deterministic explicit RNG seed (`42`).
    pub seed: u64,
}

impl Default for TsNetConfig {
    fn default() -> Self {
        Self::pure_2hop()
    }
}

impl TsNetConfig {
    /// Returns the Pure 1+2-Hop node-wise adaptive Fit-tsNET preset configuration.
    pub fn pure_2hop() -> Self {
        Self {
            backend: TsNetBackend::FftInterpolation,
            perplexity_c: 0.9,
            min_perplexity: 5.0,
            max_perplexity: 1200.0,
            power: 2.0,
            sigma_iters: 50,
            sigma_tolerance: 1e-5,
            symmetrization: SymmetrizationMode::StandardTsne,
            iterations_stage1: 250,
            iterations_stage2: 250,
            iterations_stage3: 500,
            exaggeration: 4.0,
            lambda_c_stage1: 0.1,
            lambda_c_stage2: 0.01,
            lambda_c_stage3: 0.0,
            lambda_r_stage3: 0.0,
            learning_rate: None,
            jacobi_precondition: true,
            momentum: 0.8,
            epsilon_r: 0.05,
            theta: 0.5,
            ibfft: IbFftConfig::default(),
            seed: 42,
        }
    }

    /// Resolves the effective base learning rate for a graph with `num_nodes` vertices:
    /// explicit `Some(lr)` or automatic $O(N)$ scaling `200.0_f64.max((num_nodes as f64) / 3.0)`.
    #[inline]
    pub fn resolve_learning_rate(&self, num_nodes: usize) -> f64 {
        self.learning_rate
            .unwrap_or_else(|| 200.0_f64.max((num_nodes as f64) / 3.0))
    }

    /// Returns the 3 optimization stages defined by this configuration.
    pub fn stages(&self) -> [TsNetStage; 3] {
        [
            TsNetStage {
                iterations: self.iterations_stage1,
                exaggeration: self.exaggeration,
                lambda_kl: 1.0,
                lambda_c: self.lambda_c_stage1,
                lambda_r: 0.0,
            },
            TsNetStage {
                iterations: self.iterations_stage2,
                exaggeration: 1.0,
                lambda_kl: 1.0,
                lambda_c: self.lambda_c_stage2,
                lambda_r: 0.0,
            },
            TsNetStage {
                iterations: self.iterations_stage3,
                exaggeration: 1.0,
                lambda_kl: 1.0,
                lambda_c: self.lambda_c_stage3,
                lambda_r: self.lambda_r_stage3,
            },
        ]
    }

    /// Validates all hyperparameters and nested `ibfft` settings.
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
        if !self.exaggeration.is_finite() || self.exaggeration < 1.0 {
            return Err(CoreError::invalid_input(format!(
                "exaggeration must be finite and >= 1.0, got {}",
                self.exaggeration
            )));
        }
        for (name, val) in [
            ("lambda_c_stage1", self.lambda_c_stage1),
            ("lambda_c_stage2", self.lambda_c_stage2),
            ("lambda_c_stage3", self.lambda_c_stage3),
            ("lambda_r_stage3", self.lambda_r_stage3),
        ] {
            if !val.is_finite() || val < 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "{name} must be finite and >= 0.0, got {val}"
                )));
            }
        }
        if let Some(lr) = self.learning_rate
            && (!lr.is_finite() || lr <= 0.0)
        {
            return Err(CoreError::invalid_input(format!(
                "learning_rate must be finite and > 0.0, got {lr}"
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
        if !self.theta.is_finite() || self.theta < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "theta must be finite and >= 0.0, got {}",
                self.theta
            )));
        }
        self.ibfft.validate()
    }

    pub(super) fn effective_ibfft(&self) -> IbFftConfig {
        let mut cfg = self.ibfft;
        if self.backend == TsNetBackend::LinearFft {
            cfg.order = InterpolationOrder::Bilinear;
        }
        cfg
    }

    pub(super) fn perplexity_config(&self) -> PerplexityConfig {
        PerplexityConfig {
            power: self.power,
            sigma_iters: self.sigma_iters,
            sigma_tolerance: self.sigma_tolerance,
            symmetrization: self.symmetrization,
        }
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
