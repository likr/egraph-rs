//! Configuration and diagnostic summary types for regularized DoG Varadhan and pivoted
//! `RHGH` landmark table computation.

use egraph_core::{CoreError, CoreResult};
use egraph_linalg::SpectralMomentStats;

/// Configuration for [`compute_regularized_dog_varadhan_landmarks`](super::compute_regularized_dog_varadhan_landmarks).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct DogVaradhanConfig {
    /// Optional short-range diffusion time $t_1$ (`None` uses R1 $1 / (1 + \sigma_\lambda)$).
    pub t1: Option<f64>,
    /// Optional long-range diffusion time $t_2$ (`None` uses R1 $4 / (1 + \sigma_\lambda)$).
    pub t2: Option<f64>,
    /// Optional DoG attenuation $\alpha$ (`None` uses R1 $\max(0, 1 - \sigma_\lambda)$).
    pub alpha: Option<f64>,
    /// Optional explicit rank-1 positivity floor $\beta \in [0, 1)$ (`None` uses `beta_auto`).
    pub beta_floor: Option<f64>,
    /// Optional Chebyshev polynomial degree (`None` uses [`egraph_linalg::adaptive_chebyshev_degree`]).
    pub degree: Option<usize>,
    /// Number of Core-First MaxMin shortest-path landmarks (`default: 64`).
    pub n_landmarks: usize,
    /// Minimum degree threshold for Core-First MaxMin landmark selection (`default: 2`).
    pub min_degree: usize,
    /// Johnson-Lindenstrauss embedding dimension for positive kernel diagonal estimation (`default: 64`).
    pub jl_dim: usize,
    /// Minimum spectral floor $\epsilon_{\mathrm{psd}}$ enforced during R2 PSD projection (`default: 1e-4`).
    pub eps_psd: f64,
    /// Spectral upper bound $\lambda_{\max}$ for $L_{\mathrm{sym}}$ (`default: 2.0`).
    pub lambda_max: f64,
    /// Whether Varadhan similarities normalize by $\sqrt{K_{uu} K_{pp}}$ (`default: true`).
    pub degree_normalize: bool,
    /// Power exponent $p$ applied to $[-4 t \ln(\mathrm{sim})]^p$ (`default: 0.5`).
    pub p_exp: f64,
    /// Minimum similarity clamp $\epsilon_{\mathrm{sim}} \in (0, 1]$ (`default: 1e-12`).
    pub eps_sim: f64,
    /// Coarse diffusive scaffold time $t_{\mathrm{coarse}}$ evaluated when $\max(t_1, t_2) < 32$ (`default: 48.0`, `0.0` disables).
    pub t_coarse: f64,
    /// Deterministic explicit seed (`default: 42`).
    pub seed: u64,
}

impl Default for DogVaradhanConfig {
    fn default() -> Self {
        Self {
            t1: None,
            t2: None,
            alpha: None,
            beta_floor: None,
            degree: None,
            n_landmarks: 64,
            min_degree: 2,
            jl_dim: 64,
            eps_psd: 1e-4,
            lambda_max: 2.0,
            degree_normalize: true,
            p_exp: 0.5,
            eps_sim: 1e-12,
            t_coarse: 48.0,
            seed: 42,
        }
    }
}

impl DogVaradhanConfig {
    /// Creates a new [`DogVaradhanConfig`] initialized with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the optional short-range diffusion time `t1`.
    pub fn with_t1(mut self, t1: Option<f64>) -> Self {
        self.t1 = t1;
        self
    }

    /// Sets the optional long-range diffusion time `t2`.
    pub fn with_t2(mut self, t2: Option<f64>) -> Self {
        self.t2 = t2;
        self
    }

    /// Sets the optional DoG attenuation `alpha`.
    pub fn with_alpha(mut self, alpha: Option<f64>) -> Self {
        self.alpha = alpha;
        self
    }

    /// Sets the optional explicit rank-1 positivity floor `beta_floor`.
    pub fn with_beta_floor(mut self, beta_floor: Option<f64>) -> Self {
        self.beta_floor = beta_floor;
        self
    }

    /// Sets the optional Chebyshev polynomial degree `degree`.
    pub fn with_degree(mut self, degree: Option<usize>) -> Self {
        self.degree = degree;
        self
    }

    /// Sets the number of Core-First MaxMin shortest-path landmarks `n_landmarks`.
    pub fn with_n_landmarks(mut self, n_landmarks: usize) -> Self {
        self.n_landmarks = n_landmarks;
        self
    }

    /// Sets the minimum degree threshold for landmark selection `min_degree`.
    pub fn with_min_degree(mut self, min_degree: usize) -> Self {
        self.min_degree = min_degree;
        self
    }

    /// Sets the Johnson-Lindenstrauss embedding dimension `jl_dim`.
    pub fn with_jl_dim(mut self, jl_dim: usize) -> Self {
        self.jl_dim = jl_dim;
        self
    }

    /// Sets the minimum spectral floor `eps_psd` enforced during R2 PSD projection.
    pub fn with_eps_psd(mut self, eps_psd: f64) -> Self {
        self.eps_psd = eps_psd;
        self
    }

    /// Sets the spectral upper bound `lambda_max` for $L_{\mathrm{sym}}$.
    pub fn with_lambda_max(mut self, lambda_max: f64) -> Self {
        self.lambda_max = lambda_max;
        self
    }

    /// Sets whether Varadhan similarities normalize by $\sqrt{K_{uu} K_{pp}}$ (`degree_normalize`).
    pub fn with_degree_normalize(mut self, degree_normalize: bool) -> Self {
        self.degree_normalize = degree_normalize;
        self
    }

    /// Sets the power exponent `p_exp` applied to $[-4 t \ln(\mathrm{sim})]^p$.
    pub fn with_p_exp(mut self, p_exp: f64) -> Self {
        self.p_exp = p_exp;
        self
    }

    /// Sets the minimum similarity clamp `eps_sim`.
    pub fn with_eps_sim(mut self, eps_sim: f64) -> Self {
        self.eps_sim = eps_sim;
        self
    }

    /// Sets the coarse diffusive scaffold time `t_coarse`.
    pub fn with_t_coarse(mut self, t_coarse: f64) -> Self {
        self.t_coarse = t_coarse;
        self
    }

    /// Sets the deterministic explicit PRNG seed `seed`.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Summary diagnostics emitted by [`compute_regularized_dog_varadhan_landmarks`](super::compute_regularized_dog_varadhan_landmarks).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct DogVaradhanStats {
    pub moment_stats: SpectralMomentStats,
    pub t1: f64,
    pub t2: f64,
    pub alpha: f64,
    pub degree: usize,
    pub delta_psd: f64,
    pub h_min: f64,
    pub k_min: f64,
    pub k_med: f64,
    pub beta_auto: f64,
    pub beta_eff: f64,
    pub n_landmarks: usize,
    pub t_coarse: f64,
    pub used_coarse_scaffold: bool,
}

pub(super) fn validate_dog_varadhan_config(config: &DogVaradhanConfig) -> CoreResult<()> {
    if !config.t_coarse.is_finite() {
        return Err(CoreError::non_finite("DogVaradhanConfig::t_coarse"));
    }
    if config.t_coarse < 0.0 {
        return Err(CoreError::invalid_input(format!(
            "t_coarse must be >= 0, got {}",
            config.t_coarse
        )));
    }
    if !config.p_exp.is_finite() {
        return Err(CoreError::non_finite("DogVaradhanConfig::p_exp"));
    }
    if config.p_exp <= 0.0 {
        return Err(CoreError::invalid_input(format!(
            "p_exp must be > 0, got {}",
            config.p_exp
        )));
    }
    if !config.eps_sim.is_finite() {
        return Err(CoreError::non_finite("DogVaradhanConfig::eps_sim"));
    }
    if config.eps_sim <= 0.0 || config.eps_sim > 1.0 {
        return Err(CoreError::invalid_input(format!(
            "eps_sim must be in (0, 1], got {}",
            config.eps_sim
        )));
    }
    if config.n_landmarks == 0 {
        return Err(CoreError::invalid_input("n_landmarks must be >= 1"));
    }
    if config.jl_dim == 0 {
        return Err(CoreError::invalid_input("jl_dim must be >= 1"));
    }
    if !config.eps_psd.is_finite() || config.eps_psd < 0.0 {
        return Err(CoreError::invalid_input("eps_psd must be finite and >= 0"));
    }
    if let Some(bf) = config.beta_floor
        && (!bf.is_finite() || !(0.0..1.0).contains(&bf))
    {
        return Err(CoreError::invalid_input(
            "beta_floor must be finite and in [0.0, 1.0)",
        ));
    }
    if let Some(deg) = config.degree
        && deg == 0
    {
        return Err(CoreError::invalid_input("degree must be >= 1"));
    }
    Ok(())
}

/// Configuration for [`compute_pivoted_rhgh_landmarks`](super::compute_pivoted_rhgh_landmarks).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct RhghLandmarkConfig {
    /// Optional resolvent regularization $\gamma > 0$ (`None` uses $O(|E|)$ topology-adaptive auto-$\gamma$).
    pub gamma: Option<f64>,
    /// Optional power exponent $p > 0$ (`None` uses $O(|E|)$ topology-adaptive auto-$p$).
    pub p_exp: Option<f64>,
    /// Number of Core-First MaxMin shortest-path landmarks (`default: 64`).
    pub n_landmarks: usize,
    /// Minimum degree threshold for Core-First MaxMin landmark selection (`default: 2`).
    pub min_degree: usize,
    /// Number of Spielman-Srivastava JL probes for diagonal estimation (`default: 64`).
    pub num_probes: usize,
    /// Relative residual tolerance for `AmgPcgSolver` (`default: 1e-12`).
    pub rel_tol: f64,
    /// Maximum AMG-PCG iterations per RHS column (`default: 500`).
    pub max_iterations: usize,
    /// Whether to divide distances by the median 1-hop ALT landmark distance (`default: true`).
    pub normalize_1hop: bool,
    /// Deterministic explicit PRNG seed (`default: 42`).
    pub seed: u64,
}

impl Default for RhghLandmarkConfig {
    fn default() -> Self {
        Self {
            gamma: None,
            p_exp: None,
            n_landmarks: 64,
            min_degree: 2,
            num_probes: 64,
            rel_tol: 1e-12,
            max_iterations: 500,
            normalize_1hop: true,
            seed: 42,
        }
    }
}

impl RhghLandmarkConfig {
    /// Creates a new [`RhghLandmarkConfig`] initialized with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the optional resolvent regularization `gamma`.
    pub fn with_gamma(mut self, gamma: Option<f64>) -> Self {
        self.gamma = gamma;
        self
    }

    /// Sets the optional power exponent `p_exp`.
    pub fn with_p_exp(mut self, p_exp: Option<f64>) -> Self {
        self.p_exp = p_exp;
        self
    }

    /// Sets the number of Core-First MaxMin shortest-path landmarks `n_landmarks`.
    pub fn with_n_landmarks(mut self, n_landmarks: usize) -> Self {
        self.n_landmarks = n_landmarks;
        self
    }

    /// Sets the minimum degree threshold for landmark selection `min_degree`.
    pub fn with_min_degree(mut self, min_degree: usize) -> Self {
        self.min_degree = min_degree;
        self
    }

    /// Sets the number of Spielman-Srivastava JL probes `num_probes`.
    pub fn with_num_probes(mut self, num_probes: usize) -> Self {
        self.num_probes = num_probes;
        self
    }

    /// Sets the relative residual tolerance for `AmgPcgSolver` (`rel_tol`).
    pub fn with_rel_tol(mut self, rel_tol: f64) -> Self {
        self.rel_tol = rel_tol;
        self
    }

    /// Sets the maximum AMG-PCG iterations per RHS column (`max_iterations`).
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets whether to normalize distances by median 1-hop ALT distance (`normalize_1hop`).
    pub fn with_normalize_1hop(mut self, normalize_1hop: bool) -> Self {
        self.normalize_1hop = normalize_1hop;
        self
    }

    /// Sets the deterministic explicit PRNG seed `seed`.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validates all pivoted `RHGH` landmark configuration fields.
    pub fn validate(&self) -> CoreResult<()> {
        if let Some(g) = self.gamma {
            if !g.is_finite() {
                return Err(CoreError::non_finite("RhghLandmarkConfig::gamma"));
            }
            if g <= 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "RhghLandmarkConfig::gamma must be > 0, got {g}"
                )));
            }
        }
        if let Some(p) = self.p_exp {
            if !p.is_finite() {
                return Err(CoreError::non_finite("RhghLandmarkConfig::p_exp"));
            }
            if p <= 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "RhghLandmarkConfig::p_exp must be > 0, got {p}"
                )));
            }
        }
        if self.n_landmarks == 0 {
            return Err(CoreError::invalid_input(
                "RhghLandmarkConfig::n_landmarks must be >= 1",
            ));
        }
        if self.num_probes == 0 {
            return Err(CoreError::invalid_input(
                "RhghLandmarkConfig::num_probes must be >= 1",
            ));
        }
        if !self.rel_tol.is_finite() {
            return Err(CoreError::non_finite("RhghLandmarkConfig::rel_tol"));
        }
        if self.rel_tol < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "RhghLandmarkConfig::rel_tol must be >= 0, got {}",
                self.rel_tol
            )));
        }
        if self.max_iterations == 0 {
            return Err(CoreError::invalid_input(
                "RhghLandmarkConfig::max_iterations must be >= 1",
            ));
        }
        Ok(())
    }
}

/// Summary diagnostics emitted by [`compute_pivoted_rhgh_landmarks`](super::compute_pivoted_rhgh_landmarks).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct RhghLandmarkStats {
    /// Effective resolvent regularization $\gamma$ used.
    pub gamma: f64,
    /// Effective power exponent $p$ used.
    pub p_exp: f64,
    /// Coefficient of variation of node degrees.
    pub deg_cv: f64,
    /// Fraction of degree-1 leaf vertices.
    pub leaf_fraction: f64,
    /// Mean unweighted simple degree.
    pub mean_degree: f64,
    /// 2-hop expansion ratio $\mathrm{median}(k_2) / \max(\mathrm{mean\_degree}, 1)$.
    pub exp2: f64,
    /// Small-world / hub complexity score $s_{\mathrm{sw}} \in [0, 1]$.
    pub s_sw: f64,
    /// Actual number of selected landmarks.
    pub n_landmarks: usize,
    /// Number of Spielman-Srivastava JL probes used.
    pub num_probes: usize,
    /// Maximum AMG-PCG iterations across all RHS columns.
    pub max_pcg_iters: usize,
    /// Total AMG-PCG iterations across all RHS columns.
    pub total_pcg_iters: usize,
    /// Whether every AMG-PCG column converged.
    pub all_converged: bool,
    /// Median 1-hop ALT distance normalization factor applied (`1.0` if disabled).
    pub edge_med_scale: f64,
}
