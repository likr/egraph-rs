use egraph_core::{
    CoreError, CoreResult, DeterministicRng, IterativeSolver, Layout2d, Scalar, SolverStatus,
};
use egraph_spatial_fft::{Fft2dWorkspace, IbFft2d};
use egraph_spatial_trees::ArenaQuadtree2d;

use super::config::{FieldRepulsionBackend, IbUmapConfig, TFdpConfig};
use super::quadtree::FieldQuadtree;

/// Telemetry counters from the most recent `IbFft` step on [`FieldLayoutSolver`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FieldFftStats {
    pub particle_forward_ffts: u64,
    pub kernel_forward_ffts: u64,
    pub inverse_ffts: u64,
    pub total_fft_2d_calls: u64,
    pub nx: usize,
    pub ny: usize,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum FieldSolverMode {
    TFdp {
        config: TFdpConfig,
        rep_scale: f64,
    },
    IbUmap {
        config: IbUmapConfig,
        a: f64,
        b: f64,
    },
}

/// Cooperative, zero-per-step-allocation field-based layout solver supporting `t-FDP` and `ibUMAP`.
///
/// Implements [`IterativeSolver`] with bit-exact equivalence between `step()`, `run_steps(k)`,
/// and `run()` across both `f64` and `f32`.
#[derive(Debug, Clone)]
pub struct FieldLayoutSolver<S: Scalar = f64> {
    pub(super) mode: FieldSolverMode,
    pub(super) backend: FieldRepulsionBackend,
    pub(super) num_nodes: usize,
    pub(super) indptr: Vec<u32>,
    pub(super) indices: Vec<u32>,
    pub(super) edge_weights: Vec<f32>,
    pub(super) hub_eta: Vec<f64>,
    pub(super) attr_x: Vec<f64>,
    pub(super) attr_y: Vec<f64>,
    pub(super) vel_x: Vec<f64>,
    pub(super) vel_y: Vec<f64>,
    pub(super) next_x: Vec<f64>,
    pub(super) next_y: Vec<f64>,
    pub(super) scaled_x: Vec<S>,
    pub(super) scaled_y: Vec<S>,
    pub(super) phi_1: Vec<S>,
    pub(super) rep_x: Vec<S>,
    pub(super) rep_y: Vec<S>,
    pub(super) fft_ws: Fft2dWorkspace,
    pub(super) ibfft: Option<IbFft2d<S>>,
    pub(super) last_fft_stats: Option<FieldFftStats>,
    pub(super) quadtree: ArenaQuadtree2d<S>,
    pub(super) field_quadtree: FieldQuadtree,
    pub(super) rng: DeterministicRng,
    pub(super) iteration: usize,
    pub(super) total_iterations: usize,
    pub(super) status: SolverStatus,
}

/// Convenient alias for [`FieldLayoutSolver`].
pub type FieldLayout<S = f64> = FieldLayoutSolver<S>;

impl<S: Scalar> FieldLayoutSolver<S> {
    /// Returns the number of vertices expected in [`Layout2d`].
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the active repulsion evaluation backend.
    #[inline]
    pub fn backend(&self) -> FieldRepulsionBackend {
        self.backend
    }

    /// Returns the number of stored directed positive attraction entries.
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.indices.len()
    }

    /// Returns the cumulative number of `fft_2d` calls executed across all steps.
    #[inline]
    pub fn total_fft_2d_calls(&self) -> u64 {
        self.fft_ws.fft_2d_call_count()
    }

    /// Returns telemetry counters from the most recent `IbFft` step, if any.
    #[inline]
    pub fn last_fft_stats(&self) -> Option<FieldFftStats> {
        self.last_fft_stats
    }

    /// Returns the resolved UMAP `(a, b)` parameters when in `ibUMAP` mode, or `None` for `t-FDP`.
    #[inline]
    pub fn ab(&self) -> Option<(f64, f64)> {
        match self.mode {
            FieldSolverMode::TFdp { .. } => None,
            FieldSolverMode::IbUmap { a, b, .. } => Some((a, b)),
        }
    }

    /// Returns the resolved UMAP curve parameter `a` in `ibUMAP` mode, or `None` for `t-FDP`.
    #[inline]
    pub fn a(&self) -> Option<f64> {
        match self.mode {
            FieldSolverMode::TFdp { .. } => None,
            FieldSolverMode::IbUmap { a, .. } => Some(a),
        }
    }

    /// Returns the resolved UMAP curve parameter `b` in `ibUMAP` mode, or `None` for `t-FDP`.
    #[inline]
    pub fn b(&self) -> Option<f64> {
        match self.mode {
            FieldSolverMode::TFdp { .. } => None,
            FieldSolverMode::IbUmap { b, .. } => Some(b),
        }
    }

    /// Returns the current $\rho(t)$ repulsion weight in `t-FDP` mode (or `1.0` in `ibUMAP` mode).
    #[inline]
    pub fn current_rho(&self) -> f64 {
        match self.mode {
            FieldSolverMode::TFdp { config, .. } => {
                let progress = if self.total_iterations <= 1 {
                    if self.iteration >= self.total_iterations {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    (self.iteration as f64) / ((self.total_iterations - 1) as f64)
                };
                config.rho.evaluate(progress)
            }
            FieldSolverMode::IbUmap { .. } => 1.0,
        }
    }
}

impl<S: Scalar> IterativeSolver<S> for FieldLayoutSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "FieldLayoutSolver::step",
            ));
        }
        layout.validate_finite()?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        if self.num_nodes < 2 {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        match self.mode {
            FieldSolverMode::TFdp { config, rep_scale } => {
                self.step_t_fdp(layout, config, rep_scale)?;
            }
            FieldSolverMode::IbUmap { config, a, b } => {
                self.step_ib_umap(layout, config, a, b)?;
            }
        }

        self.iteration += 1;
        self.status = if self.iteration >= self.total_iterations {
            SolverStatus::Completed
        } else {
            SolverStatus::Running
        };
        Ok(self.status)
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.iteration
    }

    #[inline]
    fn total_iterations(&self) -> usize {
        self.total_iterations
    }
}
