use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar};

use crate::clusterer::{ClusterResult, CommunityDetector};

use super::solver::leiden;

/// Objective function optimized by [`Leiden`] and [`leiden`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LeidenObjective {
    /// Standard Modularity $Q_\gamma$ (Traag et al., 2019, Eq. (1) / Eq. (A.5)).
    #[default]
    Modularity,
    /// Constant Potts Model (CPM) (Traag et al., 2019, Eq. (2) / Eq. (A.4)).
    Cpm,
}

/// Configuration for [`Leiden`] and [`leiden`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct LeidenConfig {
    /// Quality objective (`Modularity` or `Cpm`).
    pub objective: LeidenObjective,
    /// Resolution parameter $\gamma \ge 0.0$ (`1.0` for standard modularity).
    pub resolution: f64,
    /// Temperature/randomness parameter $\theta > 0.0$ controlling stochastic refinement merges.
    pub randomness: f64,
    /// Maximum number of multi-level Leiden passes (local moving + refinement + aggregation).
    pub max_passes: usize,
    /// Minimum quality gain required to commit a pass and continue.
    pub min_quality_gain: f64,
    /// Deterministic PRNG seed controlling queue order and refinement sampling.
    pub seed: u64,
}

impl Default for LeidenConfig {
    fn default() -> Self {
        Self {
            objective: LeidenObjective::Modularity,
            resolution: 1.0,
            randomness: 0.01,
            max_passes: 100,
            min_quality_gain: 1e-7,
            seed: 42,
        }
    }
}

impl LeidenConfig {
    /// Creates a new [`LeidenConfig`] initialized to [`Self::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the quality objective (`Modularity` or `Cpm`).
    pub fn with_objective(mut self, objective: LeidenObjective) -> Self {
        self.objective = objective;
        self
    }

    /// Sets the resolution parameter `resolution`.
    pub fn with_resolution(mut self, resolution: f64) -> Self {
        self.resolution = resolution;
        self
    }

    /// Sets the refinement randomness temperature `randomness`.
    pub fn with_randomness(mut self, randomness: f64) -> Self {
        self.randomness = randomness;
        self
    }

    /// Sets the maximum number of multi-level Leiden passes `max_passes`.
    pub fn with_max_passes(mut self, max_passes: usize) -> Self {
        self.max_passes = max_passes;
        self
    }

    /// Sets the minimum quality gain threshold `min_quality_gain`.
    pub fn with_min_quality_gain(mut self, min_quality_gain: f64) -> Self {
        self.min_quality_gain = min_quality_gain;
        self
    }

    /// Sets the deterministic PRNG seed `seed`.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validates all Leiden hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if self.max_passes == 0 {
            return Err(CoreError::invalid_input("max_passes must be >= 1"));
        }
        if !self.resolution.is_finite() {
            return Err(CoreError::non_finite("LeidenConfig::resolution"));
        }
        if self.resolution < 0.0 {
            return Err(CoreError::invalid_input("resolution must be >= 0.0"));
        }
        if !self.randomness.is_finite() {
            return Err(CoreError::non_finite("LeidenConfig::randomness"));
        }
        if self.randomness <= 0.0 {
            return Err(CoreError::invalid_input("randomness must be > 0.0"));
        }
        if !self.min_quality_gain.is_finite() {
            return Err(CoreError::non_finite("LeidenConfig::min_quality_gain"));
        }
        if self.min_quality_gain < 0.0 {
            return Err(CoreError::invalid_input("min_quality_gain must be >= 0.0"));
        }
        Ok(())
    }
}

/// Output of [`Leiden::run`] and [`leiden`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct LeidenResult {
    /// Canonical community assignment in `0..num_communities` in first-appearance vertex order.
    pub labels: Vec<u32>,
    /// Number of distinct non-empty communities detected.
    pub num_communities: usize,
    /// Objective quality value: normalized $Q_\gamma$ for `Modularity`, or unnormalized
    /// $\sum_C [E(C, C) - \gamma \binom{\|C\|}{2}]$ (Traag Eq. (A.4)) for `Cpm`.
    pub quality: f64,
    /// Standard normalized modularity $Q$ evaluated at $\gamma = 1.0$ (`0.0` if `2m == 0`).
    pub modularity: f64,
    /// Number of multi-level passes committed.
    pub passes: usize,
    /// Whether Leiden converged within `min_quality_gain` before exhausting `max_passes`.
    pub converged: bool,
}

impl From<LeidenResult> for ClusterResult {
    fn from(res: LeidenResult) -> Self {
        Self::new(
            res.labels,
            res.num_communities,
            Some(res.quality),
            res.passes,
            res.converged,
        )
    }
}

/// Multi-level Leiden community detector implementing [`CommunityDetector`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Leiden {
    config: LeidenConfig,
}

impl Leiden {
    /// Creates a new [`Leiden`] community detector with the given configuration.
    pub fn new(config: LeidenConfig) -> Self {
        Self { config }
    }

    /// Returns a reference to the underlying [`LeidenConfig`].
    #[inline]
    pub fn config(&self) -> &LeidenConfig {
        &self.config
    }

    /// Replaces the configuration with `config`.
    pub fn with_config(mut self, config: LeidenConfig) -> Self {
        self.config = config;
        self
    }

    /// Sets the quality objective (`Modularity` or `Cpm`).
    pub fn with_objective(mut self, objective: LeidenObjective) -> Self {
        self.config.objective = objective;
        self
    }

    /// Sets the resolution parameter `resolution`.
    pub fn with_resolution(mut self, resolution: f64) -> Self {
        self.config.resolution = resolution;
        self
    }

    /// Sets the refinement randomness temperature `randomness`.
    pub fn with_randomness(mut self, randomness: f64) -> Self {
        self.config.randomness = randomness;
        self
    }

    /// Sets the maximum number of multi-level Leiden passes `max_passes`.
    pub fn with_max_passes(mut self, max_passes: usize) -> Self {
        self.config.max_passes = max_passes;
        self
    }

    /// Sets the minimum quality gain threshold `min_quality_gain`.
    pub fn with_min_quality_gain(mut self, min_quality_gain: f64) -> Self {
        self.config.min_quality_gain = min_quality_gain;
        self
    }

    /// Sets the deterministic PRNG seed `seed`.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    /// Runs Leiden community detection and returns the specialized [`LeidenResult`].
    pub fn run<W: Scalar>(&self, graph: &CsrGraph<W>) -> CoreResult<LeidenResult> {
        leiden(graph, &self.config)
    }
}

impl<W: Scalar> CommunityDetector<W> for Leiden {
    fn detect(&self, graph: &CsrGraph<W>) -> CoreResult<ClusterResult> {
        self.run(graph).map(Into::into)
    }
}
