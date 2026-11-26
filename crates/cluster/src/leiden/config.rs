/// Objective function optimized by [`leiden`](super::leiden).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LeidenObjective {
    /// Standard Modularity $Q_\gamma$ (Traag et al., 2019, Eq. (1) / Eq. (A.5)).
    #[default]
    Modularity,
    /// Constant Potts Model (CPM) (Traag et al., 2019, Eq. (2) / Eq. (A.4)).
    Cpm,
}

/// Configuration for [`leiden`](super::leiden).
#[derive(Debug, Clone, Copy, PartialEq)]
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

/// Output of [`leiden`](super::leiden).
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
}
