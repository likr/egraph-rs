use egraph_core::{CoreError, CoreResult};

/// Hyperparameters for [`SpatialIntruderRefiner`](super::SpatialIntruderRefiner).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialRefinerConfig {
    /// Maximum number of outer active-set refinement iterations (`default: 4`).
    pub iterations: usize,
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
    pub sgd_iterations: usize,
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
            iterations: 4,
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
            sgd_iterations: 20,
            eps: 0.05,
            hinge_fp: true,
            min_sep_ratio: 0.0,
            w_sep: 0.0,
            seed: 42,
        }
    }
}

impl SpatialRefinerConfig {
    /// Creates a new [`SpatialRefinerConfig`] initialized to [`Self::default`].
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of outer active-set refinement `iterations`.
    #[inline]
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Sets the per-node spatial $k$-NN cap `k_cap`.
    #[inline]
    pub fn with_k_cap(mut self, k_cap: usize) -> Self {
        self.k_cap = k_cap;
        self
    }

    /// Sets the repulsive hinge target multiplier `margin_scale`.
    #[inline]
    pub fn with_margin_scale(mut self, margin_scale: f64) -> Self {
        self.margin_scale = margin_scale;
        self
    }

    /// Sets the Tier 1 (1-hop anchor) spring weight multiplier `w_anchor`.
    #[inline]
    pub fn with_w_anchor(mut self, w_anchor: f64) -> Self {
        self.w_anchor = w_anchor;
        self
    }

    /// Sets the Tier 1/2 (2-hop anchor) spring weight multiplier `w_anchor2`.
    #[inline]
    pub fn with_w_anchor2(mut self, w_anchor2: f64) -> Self {
        self.w_anchor2 = w_anchor2;
        self
    }

    /// Sets the Tier 2 (false negative) spring weight multiplier `w_fn`.
    #[inline]
    pub fn with_w_fn(mut self, w_fn: f64) -> Self {
        self.w_fn = w_fn;
        self
    }

    /// Sets the Tier 3 (>2-hop spatial false positive) weight multiplier `w_fp`.
    #[inline]
    pub fn with_w_fp(mut self, w_fp: f64) -> Self {
        self.w_fp = w_fp;
        self
    }

    /// Sets the target shrinkage factor `fn_shrink` applied to active false negatives.
    #[inline]
    pub fn with_fn_shrink(mut self, fn_shrink: f64) -> Self {
        self.fn_shrink = fn_shrink;
        self
    }

    /// Sets the multiplicative step-scale decay `round_decay` per outer round.
    #[inline]
    pub fn with_round_decay(mut self, round_decay: f64) -> Self {
        self.round_decay = round_decay;
        self
    }

    /// Sets the per-round multiplicative decay `k_cap_decay` applied to `k_cap`.
    #[inline]
    pub fn with_k_cap_decay(mut self, k_cap_decay: f64) -> Self {
        self.k_cap_decay = k_cap_decay;
        self
    }

    /// Sets the minimum lower bound `k_cap_min` for decayed `k_cap`.
    #[inline]
    pub fn with_k_cap_min(mut self, k_cap_min: usize) -> Self {
        self.k_cap_min = k_cap_min;
        self
    }

    /// Sets the inner `SgdSolver` iterations `sgd_iterations` per active round.
    #[inline]
    pub fn with_sgd_iterations(mut self, sgd_iterations: usize) -> Self {
        self.sgd_iterations = sgd_iterations;
        self
    }

    /// Sets the exponential scheduler minimum learning rate ratio `eps`.
    #[inline]
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Sets whether Tier 3 FPs use one-sided hinge repulsion (`hinge_fp`).
    #[inline]
    pub fn with_hinge_fp(mut self, hinge_fp: bool) -> Self {
        self.hinge_fp = hinge_fp;
        self
    }

    /// Sets the opt-in minimum separation threshold `min_sep_ratio`.
    #[inline]
    pub fn with_min_sep_ratio(mut self, min_sep_ratio: f64) -> Self {
        self.min_sep_ratio = min_sep_ratio;
        self
    }

    /// Sets the opt-in short-range separation hinge weight multiplier `w_sep`.
    #[inline]
    pub fn with_w_sep(mut self, w_sep: f64) -> Self {
        self.w_sep = w_sep;
        self
    }

    /// Sets the explicit deterministic PRNG `seed`.
    #[inline]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validates all refinement hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if self.k_cap == 0 {
            return Err(CoreError::invalid_input("k_cap must be >= 1"));
        }
        if !self.margin_scale.is_finite() {
            return Err(CoreError::non_finite("SpatialRefinerConfig::margin_scale"));
        }
        if self.margin_scale <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "margin_scale must be > 0.0, got {}",
                self.margin_scale
            )));
        }
        for (name, ctx, val) in [
            ("w_anchor", "SpatialRefinerConfig::w_anchor", self.w_anchor),
            (
                "w_anchor2",
                "SpatialRefinerConfig::w_anchor2",
                self.w_anchor2,
            ),
            ("w_fn", "SpatialRefinerConfig::w_fn", self.w_fn),
            ("w_fp", "SpatialRefinerConfig::w_fp", self.w_fp),
            (
                "min_sep_ratio",
                "SpatialRefinerConfig::min_sep_ratio",
                self.min_sep_ratio,
            ),
            ("w_sep", "SpatialRefinerConfig::w_sep", self.w_sep),
        ] {
            if !val.is_finite() {
                return Err(CoreError::non_finite(ctx));
            }
            if val < 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "{name} must be >= 0.0, got {val}"
                )));
            }
        }
        if !self.fn_shrink.is_finite() {
            return Err(CoreError::non_finite("SpatialRefinerConfig::fn_shrink"));
        }
        if self.fn_shrink <= 0.0 || self.fn_shrink > 1.0 {
            return Err(CoreError::invalid_input(format!(
                "fn_shrink must be in (0.0, 1.0], got {}",
                self.fn_shrink
            )));
        }
        if !self.round_decay.is_finite() {
            return Err(CoreError::non_finite("SpatialRefinerConfig::round_decay"));
        }
        if self.round_decay <= 0.0 || self.round_decay > 1.0 {
            return Err(CoreError::invalid_input(format!(
                "round_decay must be in (0.0, 1.0], got {}",
                self.round_decay
            )));
        }
        if !self.k_cap_decay.is_finite() {
            return Err(CoreError::non_finite("SpatialRefinerConfig::k_cap_decay"));
        }
        if self.k_cap_decay <= 0.0 || self.k_cap_decay > 1.0 {
            return Err(CoreError::invalid_input(format!(
                "k_cap_decay must be in (0.0, 1.0], got {}",
                self.k_cap_decay
            )));
        }
        if self.k_cap_min == 0 {
            return Err(CoreError::invalid_input("k_cap_min must be >= 1"));
        }
        if !self.eps.is_finite() {
            return Err(CoreError::non_finite("SpatialRefinerConfig::eps"));
        }
        if self.eps <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "eps must be > 0.0, got {}",
                self.eps
            )));
        }
        Ok(())
    }
}

/// Per-round diagnostic counters emitted by [`SpatialIntruderRefiner`](super::SpatialIntruderRefiner).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
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
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SpatialRefinerStats {
    /// Number of active SGD refinement rounds executed.
    pub rounds_executed: usize,
    /// Cumulative number of active Tier 1/2 anchor pairs across rounds.
    pub n_anchors_total: usize,
    /// Cumulative number of active Tier 2 false-negative pairs across rounds.
    pub n_fn_total: usize,
    /// Cumulative number of active Tier 3 spatial false-positive intruder pairs across rounds.
    pub n_fp_total: usize,
    /// Cumulative number of active short-range separation hinge pairs across rounds.
    pub n_sep_total: usize,
    /// Cumulative total active pairs supplied to inner `SgdSolver` instances across rounds.
    pub n_pairs_total: usize,
    /// Initial layout scale (median 1-hop edge length, or coordinate std fallback).
    pub initial_scale: f64,
    /// Final layout scale after executed refinement rounds.
    pub final_scale: f64,
    /// Minimum positive target distance floor derived from `initial_scale`.
    pub d_floor: f64,
    /// Ratio of final coordinate standard deviation to initial coordinate standard deviation.
    pub scale_ratio: f64,
    /// `true` when high-degree hub fallback avoided materializing the full 2-hop CSR graph.
    pub used_hub_fallback: bool,
    /// Whether Tier 3 FPs were configured as one-sided hinges.
    pub hinge_fp: bool,
    /// Per-round diagnostic records in execution order.
    pub round_stats: Vec<SpatialRefinerRoundStats>,
}
