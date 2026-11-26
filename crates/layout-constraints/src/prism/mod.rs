//! PRISM proximity-stress overlap removal using Delaunay triangulation or KD-tree fallback.
//!
//! Implements Gansner & Hu (2010), "Efficient Node Overlap Removal Using a Proximity Stress
//! Model", using `Delaunay2d` (under `feature = "delaunay"`) or `StaticKdTree` $k$-NN fallback
//! (under `--no-default-features`), combined with an allocation-free Sweep-and-Prune (SAP)
//! overlap detector to guarantee `0` overlaps at convergence, and a zero-dense-allocation split
//! `L+D+U` sparse CSR fused 2-RHS Symmetric SSOR-PCG solver.

mod pcg;
mod pcg_ssor;
mod proximity;
mod sap;
mod solver;

use egraph_core::{CoreError, CoreResult};

pub use pcg::SparsePcgWorkspace;
pub use solver::PrismOverlapRemoval;

#[cfg(not(feature = "delaunay"))]
pub(super) const K_FALLBACK_NEIGHBORS: usize = 12;

pub(super) const MIN_EDGE_DIST: f64 = 1e-9;
pub(super) const MIN_WEIGHT_DIST: f64 = 1e-6;
pub(super) const INNER_MAJORIZATION_ITERS: usize = 1;
pub(super) const OVERLAP_WEIGHT_BOOST: f64 = 16.0;
pub(super) const SSOR_OMEGA: f64 = 1.2;
pub(super) const OVERLAP_MICRO_DIST_RATIO: f64 = 1e-4;
pub(super) const STALL_NO_IMPROVE_STREAK: usize = 8;

/// Hyperparameters for [`PrismOverlapRemoval`].
#[derive(Debug, Clone, Copy, PartialEq)]
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
