//! Hayashi et al. (1998) `PFS'` two-pass scan-line node overlap removal transform.
//!
//! Implements `PFS'` (*"Layout Adjustment and the Mental Map"*, Misue, Eades, Lai & Sugiyama;
//! *"A Fast Algorithm for Overlap Removal"*, Hayashi, Inoue, Masuzawa & Fujiwara, GD '98)
//! following the authoritative `agorajs/agora-pfsp` two-pass `horizontalScan` + `verticalScan`
//! equations, augmented with:
//! 1. Deterministic exact-coincident-point jitter (`dx == 0.0 && dy == 0.0`) bounded strictly
//!    below `0.25 * min_positive_gap` so zero-separation pairs do not stall the scan-line force
//!    function while preserving all strict orthogonal order relations.
//! 2. Symmetric border clearance (`0.5 * padding` added to each half-width and half-height).
//! 3. Direct displacement accumulation without mean-centering or extent round-trip cancellation,
//!    magnitude-aware `S`-precision separation slack, `f64`-exact $O(N \log N)$ sweep verification,
//!    `RectangleOverlapSweep` verification, an $O(N \log N)$ orthogonal-order-inversion check,
//!    and a strictly order-preserving monotonic X/Y safety sweep so converged outputs satisfy
//!    `last_overlap_count() == 0` and `oo_nni == 0` for both `f64` and `f32`.

mod config;
mod order_sweep;
mod scan;
mod solver;
#[cfg(test)]
mod tests;
mod verify;

pub use config::PfsPrimeConfig;
pub use solver::PfsPrimeOverlapRemoval;
