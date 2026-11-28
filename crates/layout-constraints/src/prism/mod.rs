//! PRISM proximity-stress overlap removal using Delaunay triangulation.
//!
//! Implements Gansner & Hu (2010), "Efficient Node Overlap Removal Using a Proximity Stress
//! Model", using [`Delaunay2d`](egraph_spatial_trees::Delaunay2d) combined with an
//! allocation-free Sweep-and-Prune (SAP) overlap detector to guarantee `0` overlaps at
//! convergence, and a zero-dense-allocation split `L+D+U` sparse CSR fused 2-RHS
//! Symmetric SSOR-PCG solver.

mod pcg;
mod pcg_ssor;
mod proximity;
mod sap;
mod solver;

use egraph_core::{CoreError, CoreResult};

pub use pcg::SparsePcgWorkspace;
pub use solver::PrismOverlapRemoval;

pub(super) const MIN_EDGE_DIST: f64 = 1e-9;
pub(super) const MIN_WEIGHT_DIST: f64 = 1e-6;
pub(super) const INNER_MAJORIZATION_ITERS: usize = 1;
pub(super) const OVERLAP_WEIGHT_BOOST: f64 = 16.0;
pub(super) const SSOR_OMEGA: f64 = 1.2;
pub(super) const OVERLAP_MICRO_DIST_RATIO: f64 = 1e-4;
pub(super) const STALL_NO_IMPROVE_STREAK: usize = 8;

/// Hyperparameters for [`PrismOverlapRemoval`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct PrismConfig {
    /// Maximum number of PRISM proximity-stress rounds (`default: 50`).
    pub max_iterations: usize,
    /// Maximum per-round edge expansion factor $s_{\max} > 1.0$ (`default: 1.5`).
    pub max_scale_factor: f64,
    /// Maximum inner PCG iterations per linear solve (`default: 80`).
    pub cg_max_iters: usize,
    /// Relative residual tolerance for inner PCG (`default: 1e-3`).
    pub cg_tolerance: f64,
    /// Extra minimum clearance added between rectangle borders (`default: 0.0`).
    pub padding: f64,
    /// Explicit deterministic seed used for coincident-point jitter (`default: 42`).
    pub seed: u64,
}

impl Default for PrismConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            max_scale_factor: 1.5,
            cg_max_iters: 80,
            cg_tolerance: 1e-3,
            padding: 0.0,
            seed: 42,
        }
    }
}

impl PrismConfig {
    /// Creates a new [`PrismConfig`] initialized with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of PRISM proximity-stress rounds.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets the maximum per-round edge expansion factor $s_{\max} > 1.0$.
    pub fn with_max_scale_factor(mut self, max_scale_factor: f64) -> Self {
        self.max_scale_factor = max_scale_factor;
        self
    }

    /// Sets the maximum inner PCG iterations per linear solve.
    pub fn with_cg_max_iters(mut self, cg_max_iters: usize) -> Self {
        self.cg_max_iters = cg_max_iters;
        self
    }

    /// Sets the relative residual tolerance for inner PCG.
    pub fn with_cg_tolerance(mut self, cg_tolerance: f64) -> Self {
        self.cg_tolerance = cg_tolerance;
        self
    }

    /// Sets the extra minimum clearance added between rectangle borders.
    pub fn with_padding(mut self, padding: f64) -> Self {
        self.padding = padding;
        self
    }

    /// Sets the explicit deterministic seed used for coincident-point jitter.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validates all PRISM configuration parameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.max_scale_factor.is_finite() {
            return Err(CoreError::non_finite("PrismConfig::max_scale_factor"));
        }
        if self.max_scale_factor <= 1.0 {
            return Err(CoreError::invalid_input(format!(
                "max_scale_factor must be > 1.0, got {}",
                self.max_scale_factor
            )));
        }
        if self.cg_max_iters == 0 {
            return Err(CoreError::invalid_input("cg_max_iters must be >= 1"));
        }
        if !self.cg_tolerance.is_finite() {
            return Err(CoreError::non_finite("PrismConfig::cg_tolerance"));
        }
        if self.cg_tolerance < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "cg_tolerance must be >= 0.0, got {}",
                self.cg_tolerance
            )));
        }
        if !self.padding.is_finite() {
            return Err(CoreError::non_finite("PrismConfig::padding"));
        }
        if self.padding < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "padding must be >= 0.0, got {}",
                self.padding
            )));
        }
        Ok(())
    }
}
