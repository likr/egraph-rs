use egraph_core::{CoreError, CoreResult};

/// Hyperparameters for [`SpatialIntruderRefiner`](super::SpatialIntruderRefiner).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialRefinerConfig {
    /// Maximum number of outer active-set refinement rounds (`default: 4`).
    pub rounds: usize,
    /// Per-node spatial $k$-NN cap (`default: 32`).
    pub k_cap: usize,
    /// Repulsive hinge target multiplier relative to local empirical $k_u$-NN radius (`default: 1.2`).
    pub margin_scale: f64,
    /// Tier 1 (1-hop anchor) spring weight multiplier (`default: 1.0`).
    pub w_anchor: f64,
    /// Tier 1/2 (2-hop anchor) spring weight multiplier (`default: 0.0`).
    pub w_anchor2: f64,
    /// Tier 2 (2-hop / elongated 1-hop false negative) spring weight multiplier (`default: 0.0`).
    pub w_fn: f64,
    /// Tier 3 (>2-hop spatial intruder false positive) weight multiplier (`default: 0.2`).
    pub w_fp: f64,
    /// Target shrinkage factor applied to active false negatives (`default: 0.88`).
    pub fn_shrink: f64,
    /// Multiplicative step-scale decay per outer round (`default: 0.75`).
    pub round_decay: f64,
    /// Per-round multiplicative decay applied to `k_cap` after each active round (`default: 1.0`).
    pub k_cap_decay: f64,
    /// Minimum lower bound for decayed `k_cap` (`default: 1`).
    pub k_cap_min: usize,
    /// Inner `SgdSolver` iterations executed per active round (`default: 20`).
    pub t_max: usize,
    /// Exponential scheduler minimum learning rate ratio (`default: 0.05`).
    pub eps: f64,
    /// Whether Tier 3 FPs use one-sided hinge repulsion instead of two-sided springs (`default: true`).
    pub hinge_fp: bool,
    /// Opt-in minimum separation threshold as a fraction of `initial_scale` (`default: 0.0`, disabled).
    pub min_sep_ratio: f64,
    /// Opt-in short-range separation hinge weight multiplier (`default: 0.0`, disabled).
    pub w_sep: f64,
    /// Explicit deterministic seed for jittering and SGD shuffling (`default: 42`).
    pub seed: u64,
}

impl Default for SpatialRefinerConfig {
    fn default() -> Self {
        Self {
            rounds: 4,
            k_cap: 32,
            margin_scale: 1.2,
            w_anchor: 1.0,
            w_anchor2: 0.0,
            w_fn: 0.0,
            w_fp: 0.2,
            fn_shrink: 0.88,
            round_decay: 0.75,
            k_cap_decay: 1.0,
            k_cap_min: 1,
            t_max: 20,
            eps: 0.05,
            hinge_fp: true,
            min_sep_ratio: 0.0,
            w_sep: 0.0,
            seed: 42,
        }
    }
}

impl SpatialRefinerConfig {
    /// Validates all refinement hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if self.k_cap == 0 {
            return Err(CoreError::invalid_input("k_cap must be >= 1"));
        }
        if !self.margin_scale.is_finite() || self.margin_scale <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "margin_scale must be finite and > 0.0, got {}",
                self.margin_scale
            )));
        }
        for (name, val) in [
            ("w_anchor", self.w_anchor),
            ("w_anchor2", self.w_anchor2),
            ("w_fn", self.w_fn),
            ("w_fp", self.w_fp),
            ("min_sep_ratio", self.min_sep_ratio),
            ("w_sep", self.w_sep),
        ] {
            if !val.is_finite() || val < 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "{name} must be finite and >= 0.0, got {val}"
                )));
            }
        }
        if !self.fn_shrink.is_finite() || self.fn_shrink <= 0.0 || self.fn_shrink > 1.0 {
            return Err(CoreError::invalid_input(format!(
                "fn_shrink must be finite and in (0.0, 1.0], got {}",
                self.fn_shrink
            )));
        }
        if !self.round_decay.is_finite() || self.round_decay <= 0.0 || self.round_decay > 1.0 {
            return Err(CoreError::invalid_input(format!(
                "round_decay must be finite and in (0.0, 1.0], got {}",
                self.round_decay
            )));
        }
        if !self.k_cap_decay.is_finite() || self.k_cap_decay <= 0.0 || self.k_cap_decay > 1.0 {
            return Err(CoreError::invalid_input(format!(
                "k_cap_decay must be finite and in (0.0, 1.0], got {}",
                self.k_cap_decay
            )));
        }
        if self.k_cap_min == 0 {
            return Err(CoreError::invalid_input("k_cap_min must be >= 1"));
        }
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "eps must be finite and > 0.0, got {}",
                self.eps
            )));
        }
        Ok(())
    }
}

/// Per-round diagnostic counters emitted by [`SpatialIntruderRefiner`](super::SpatialIntruderRefiner).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpatialRefinerRoundStats {
    /// 1-indexed refinement round number.
    pub round: usize,
    /// Number of active Tier 1 1-hop anchor pairs.
    pub n_anchors: usize,
    /// Number of active Tier 2 false-negative pairs (including missing 1-hop boosts).
    pub n_fn: usize,
    /// Number of active Tier 3 spatial false-positive intruder pairs.
    pub n_fp: usize,
    /// Number of active short-range separation hinge pairs.
    pub n_sep: usize,
    /// Total active pairs supplied to the inner `SgdSolver`.
    pub n_pairs_total: usize,
    /// `true` when zero active intruders/stragglers were found and SGD was skipped.
    pub skipped: bool,
}

/// Cumulative diagnostic summary across rounds of [`SpatialIntruderRefiner`](super::SpatialIntruderRefiner).
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialRefinerStats {
    pub rounds_executed: usize,
    pub n_anchors_total: usize,
    pub n_fn_total: usize,
    pub n_fp_total: usize,
    pub n_sep_total: usize,
    pub n_pairs_total: usize,
    pub initial_scale: f64,
    pub final_scale: f64,
    pub d_floor: f64,
    pub scale_ratio: f64,
    pub used_hub_fallback: bool,
    pub hinge_fp: bool,
    pub round_stats: Vec<SpatialRefinerRoundStats>,
}
