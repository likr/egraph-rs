use egraph_core::{CoreError, CoreResult};

pub(super) const MIN_EDGE_DIST: f64 = 1e-9;
pub(super) const MIN_TARGET_DIST: f64 = 1e-6;
pub(super) const SGD2_EPS: f64 = 0.01;

/// Algorithm variant for [`super::ForbidOverlapRemoval`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ForbidVariant {
    /// Original FORBID: scales current adjusted layout `X'` across binary-search passes.
    Forbid,
    /// `FORBID'`: resets each binary-search pass by scaling the initial layout `X`.
    ForbidPrime,
    /// Sparse FORBID: combines `StaticKdTree` `sparse_k`-NN pairs with all active overlaps $\mathcal{O}(X')$.
    #[default]
    SparseForbid,
}

/// Hyperparameters for [`super::ForbidOverlapRemoval`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForbidConfig {
    /// FORBID variant (`default: ForbidVariant::SparseForbid`).
    pub variant: ForbidVariant,
    /// Maximum number of binary-search scaling iterations (`default: 20`).
    pub max_scale_iters: usize,
    /// Maximum SGD2 iterations per scaling pass (`default: 30`).
    pub sgd_iterations: usize,
    /// Number of nearest neighbors per node in [`ForbidVariant::SparseForbid`] (`default: 30`).
    pub sparse_k: usize,
    /// Exponent multiplier $k$ in overlap pair weights $\delta_{ij}^{k \alpha}$ ($\alpha = -2$, `default: 1.0`).
    pub overlap_weight_k: f64,
    /// Binary-search stopping width `up_scale - low_scale <= scale_step` (`default: 0.1`).
    pub scale_step: f64,
    /// Extra minimum clearance added between rectangle borders (`default: 0.0`).
    pub padding: f64,
    /// Explicit deterministic seed for coincident perturbation and SGD shuffling (`default: 42`).
    pub seed: u64,
}

impl Default for ForbidConfig {
    fn default() -> Self {
        Self {
            variant: ForbidVariant::SparseForbid,
            max_scale_iters: 20,
            sgd_iterations: 30,
            sparse_k: 30,
            overlap_weight_k: 1.0,
            scale_step: 0.1,
            padding: 0.0,
            seed: 42,
        }
    }
}

impl ForbidConfig {
    /// Validates all FORBID configuration parameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.overlap_weight_k.is_finite() {
            return Err(CoreError::non_finite("ForbidConfig::overlap_weight_k"));
        }
        if !self.scale_step.is_finite() {
            return Err(CoreError::non_finite("ForbidConfig::scale_step"));
        }
        if self.scale_step <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "ForbidConfig::scale_step must be > 0.0, got {}",
                self.scale_step
            )));
        }
        if !self.padding.is_finite() {
            return Err(CoreError::non_finite("ForbidConfig::padding"));
        }
        if self.padding < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "ForbidConfig::padding must be >= 0.0, got {}",
                self.padding
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct SgdPairItem {
    pub(super) u: u32,
    pub(super) v: u32,
    pub(super) d: f64,
    pub(super) w: f64,
    pub(super) is_overlap: bool,
}
