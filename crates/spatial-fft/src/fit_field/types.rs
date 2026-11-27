use egraph_core::{CoreError, CoreResult, Scalar};

use crate::ibfft::{IbFftConfig, InterpolationOrder};

/// Configuration for [`super::FitField2d`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct FitFieldConfig {
    /// Grid & interpolation configuration (`Bilinear` $p=1$ by default).
    pub ibfft: IbFftConfig,
    /// Entropy softening constant $\varepsilon_r > 0$ in $K_e(d) = 1 / (\varepsilon_r + d^2)$.
    pub epsilon_r: f64,
    /// Whether to evaluate C2 entropy repulsion alongside C1 KL repulsion.
    pub compute_entropy: bool,
}

impl Default for FitFieldConfig {
    fn default() -> Self {
        Self {
            ibfft: IbFftConfig::default(),
            epsilon_r: 0.05,
            compute_entropy: true,
        }
    }
}

impl FitFieldConfig {
    /// Validates field parameters.
    pub fn validate(&self) -> CoreResult<()> {
        self.ibfft.validate()?;
        if !self.epsilon_r.is_finite() || self.epsilon_r <= 0.0 {
            return Err(CoreError::invalid_input(
                "epsilon_r must be finite and strictly positive",
            ));
        }
        Ok(())
    }

    /// Sets the underlying [`IbFftConfig`].
    pub fn with_ibfft(mut self, ibfft: IbFftConfig) -> Self {
        self.ibfft = ibfft;
        self
    }

    /// Sets the interpolation order inside `ibfft`.
    pub fn with_order(mut self, order: InterpolationOrder) -> Self {
        self.ibfft.order = order;
        self
    }

    /// Sets the entropy softening constant $\varepsilon_r > 0$.
    pub fn with_epsilon_r(mut self, epsilon_r: f64) -> Self {
        self.epsilon_r = epsilon_r;
        self
    }

    /// Sets whether to evaluate C2 entropy repulsion alongside C1 KL repulsion.
    pub fn with_compute_entropy(mut self, compute_entropy: bool) -> Self {
        self.compute_entropy = compute_entropy;
        self
    }
}

/// Telemetry counters recorded for each [`super::FitField2d::evaluate`] invocation.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FitFieldStats {
    /// Number of forward 2D FFTs executed on particle charge grids (`2` when `n >= 2`).
    pub particle_forward_ffts: u64,
    /// Number of forward 2D FFTs executed on kernel grids (`2` when fused, `1` for KL-only).
    pub kernel_forward_ffts: u64,
    /// Number of inverse 2D FFTs executed (`4` when fused, `2` for KL-only).
    pub inverse_ffts: u64,
    /// Total `fft_2d` calls (`particle_forward_ffts + kernel_forward_ffts + inverse_ffts <= 8`).
    pub total_fft_2d_calls: u64,
}

/// Reusable output buffers filled by [`super::FitField2d::evaluate`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct FitFieldOutput<S: Scalar = f64> {
    /// Normalized C1 KL repulsive force $x$-components: $(1/Z) \sum_{j \neq i} (y_{i,x} - y_{j,x})/(1 + d_{ij}^2)^2$.
    pub kl_repulsion_x: Vec<S>,
    /// Normalized C1 KL repulsive force $y$-components.
    pub kl_repulsion_y: Vec<S>,
    /// Estimated global Student-$t$ partition sum $Z = \sum_{i \neq j} (1 + d_{ij}^2)^{-1}$.
    pub total_z: S,
    /// Unnormalized C2 entropy repulsive force $x$-components: $\sum_{j \neq i} (y_{i,x} - y_{j,x})/(\varepsilon_r + d_{ij}^2)$.
    pub entropy_repulsion_x: Vec<S>,
    /// Unnormalized C2 entropy repulsive force $y$-components.
    pub entropy_repulsion_y: Vec<S>,
    /// Call-count telemetry for the completed evaluation.
    pub stats: FitFieldStats,
}

impl<S: Scalar> Default for FitFieldOutput<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Scalar> FitFieldOutput<S> {
    /// Allocates an empty reusable output buffer.
    pub fn new() -> Self {
        Self {
            kl_repulsion_x: Vec::new(),
            kl_repulsion_y: Vec::new(),
            total_z: S::ONE,
            entropy_repulsion_x: Vec::new(),
            entropy_repulsion_y: Vec::new(),
            stats: FitFieldStats::default(),
        }
    }

    pub(super) fn resize_and_zero(&mut self, n: usize, compute_entropy: bool) {
        self.kl_repulsion_x.resize(n, S::ZERO);
        self.kl_repulsion_x.fill(S::ZERO);
        self.kl_repulsion_y.resize(n, S::ZERO);
        self.kl_repulsion_y.fill(S::ZERO);
        if compute_entropy {
            self.entropy_repulsion_x.resize(n, S::ZERO);
            self.entropy_repulsion_x.fill(S::ZERO);
            self.entropy_repulsion_y.resize(n, S::ZERO);
            self.entropy_repulsion_y.fill(S::ZERO);
        } else {
            self.entropy_repulsion_x.clear();
            self.entropy_repulsion_y.clear();
        }
        self.total_z = S::ONE;
        self.stats = FitFieldStats::default();
    }
}
