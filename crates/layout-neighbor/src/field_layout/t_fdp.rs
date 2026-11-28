use std::any::TypeId;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, EuclideanLayout, IterativeSolver, Layout2d,
    Layout3d, LayoutNd, Scalar, SolverStatus,
};
use egraph_neighborhood::KnnGraph;
use egraph_spatial_fft::{Fft2dWorkspace, IbFft2d};
use egraph_spatial_trees::{ArenaOctree3d, ArenaQuadtree2d};

use super::config::{FieldFftStats, FieldRepulsionBackend, TFdpConfig};
use super::t_fdp_dispatch::{
    TFdpStepFn, TFdpValidateFn, extract_symmetric_csr_edges, step_any_2d, step_any_3d, step_any_nd,
    symmetrize_pair_list, validate_2d, validate_3d, validate_nd,
};
use super::t_fdp_step::TFdpNdScratch;

#[derive(Debug, Clone)]
pub(super) struct TFdpState<S: Scalar> {
    pub(super) config: TFdpConfig,
    pub(super) rep_scale: f64,
    pub(super) num_nodes: usize,
    pub(super) indptr: Vec<u32>,
    pub(super) indices: Vec<u32>,
    pub(super) edge_weights: Vec<f32>,
    pub(super) attr_x: Vec<f64>,
    pub(super) attr_y: Vec<f64>,
    pub(super) attr_z: Vec<f64>,
    pub(super) vel_x: Vec<f64>,
    pub(super) vel_y: Vec<f64>,
    pub(super) vel_z: Vec<f64>,
    pub(super) next_x: Vec<f64>,
    pub(super) next_y: Vec<f64>,
    pub(super) next_z: Vec<f64>,
    pub(super) scaled_x: Vec<S>,
    pub(super) scaled_y: Vec<S>,
    pub(super) phi_1: Vec<S>,
    pub(super) rep_x: Vec<S>,
    pub(super) rep_y: Vec<S>,
    pub(super) rep_z: Vec<S>,
    pub(super) fft_ws: Fft2dWorkspace,
    pub(super) ibfft: Option<IbFft2d<S>>,
    pub(super) last_fft_stats: Option<FieldFftStats>,
    pub(super) quadtree: ArenaQuadtree2d<S>,
    pub(super) octree: ArenaOctree3d<S>,
    pub(super) nd_scratch: Option<TFdpNdScratch<S>>,
    pub(super) locked_nd_dims: Option<usize>,
    pub(super) rng: DeterministicRng,
    pub(super) iteration: usize,
    pub(super) total_iterations: usize,
    pub(super) status: SolverStatus,
}

impl<S: Scalar> TFdpState<S> {
    #[inline]
    pub(super) fn current_rho(&self) -> f64 {
        let progress = if self.total_iterations <= 1 {
            if self.iteration >= self.total_iterations {
                1.0
            } else {
                0.0
            }
        } else {
            (self.iteration as f64) / ((self.total_iterations - 1) as f64)
        };
        self.config.rho.evaluate(progress)
    }
}

/// Cooperative, zero-per-step-allocation `t-FDP` (bounded Student-$t$ force-directed
/// placement) solver.
///
/// Supports:
/// - [`Layout2d`]: `Exact`, `BarnesHut` (`ArenaQuadtree2d`), and `IbFft` (`IbFft2d`).
/// - [`Layout3d`]: `Exact` and `BarnesHut` (`ArenaOctree3d`).
/// - [`LayoutNd`]: `Exact` only (requests for `BarnesHut` or `IbFft` fail fast at construction).
#[derive(Debug, Clone)]
pub struct TFdpSolver<L: EuclideanLayout = Layout2d<f64>> {
    state: TFdpState<L::Scalar>,
    validate_fn: TFdpValidateFn<L>,
    step_fn: TFdpStepFn<L>,
}

impl<L: EuclideanLayout + 'static> TFdpSolver<L> {
    /// Constructs a `t-FDP` [`TFdpSolver`] directly from a [`CsrGraph`].
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: TFdpConfig) -> CoreResult<Self> {
        config.validate()?;
        let n = graph.num_nodes();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_csr("n_nodes exceeds u32::MAX"));
        }
        if let Some(ws) = graph.weights() {
            for &w in ws {
                let wf = w.to_f64();
                if !wf.is_finite() {
                    return Err(CoreError::non_finite("TFdpSolver::new::weights"));
                }
                if wf < 0.0 {
                    return Err(CoreError::invalid_input(format!(
                        "graph edge weights must be >= 0.0, got {wf}"
                    )));
                }
            }
        }

        let (indptr, indices, edge_weights) = extract_symmetric_csr_edges(graph)?;
        Self::build_inner(n, indptr, indices, edge_weights, config)
    }

    /// Constructs a `t-FDP` [`TFdpSolver`] from a validated [`KnnGraph`], converting
    /// neighbor distances $d_{ij} \ge 0$ to symmetric weights $1 / (1 + d_{ij})$.
    pub fn from_knn(knn: &KnnGraph, config: TFdpConfig) -> CoreResult<Self> {
        config.validate()?;
        let n = knn.num_nodes();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_csr("n_nodes exceeds u32::MAX"));
        }

        let mut pairs: Vec<(u32, u32, f32)> = Vec::with_capacity(knn.nnz() * 2);
        for u in 0..n {
            let u_u32 = u as u32;
            let (nbrs, dists) = knn.row(u);
            for (&v, &d) in nbrs.iter().zip(dists.iter()) {
                let v_usize = v as usize;
                if v_usize == u || v_usize >= n {
                    continue;
                }
                if !d.is_finite() {
                    return Err(CoreError::non_finite("TFdpSolver::from_knn::distances"));
                }
                if d < 0.0 {
                    return Err(CoreError::invalid_input(format!(
                        "KnnGraph distances must be >= 0.0, got {d}"
                    )));
                }
                let w = (1.0 / (1.0 + d)) as f32;
                pairs.push((u_u32, v, w));
                pairs.push((v, u_u32, w));
            }
        }

        let (indptr, indices, edge_weights) = symmetrize_pair_list(n, pairs)?;
        Self::build_inner(n, indptr, indices, edge_weights, config)
    }

    fn build_inner(
        n: usize,
        indptr: Vec<u32>,
        indices: Vec<u32>,
        edge_weights: Vec<f32>,
        config: TFdpConfig,
    ) -> CoreResult<Self> {
        let tid = TypeId::of::<L>();
        let is_2d = tid == TypeId::of::<Layout2d<L::Scalar>>();
        let is_3d = tid == TypeId::of::<Layout3d<L::Scalar>>();
        let is_nd = tid == TypeId::of::<LayoutNd<L::Scalar>>();

        if is_3d && config.backend == FieldRepulsionBackend::IbFft {
            return Err(CoreError::unsupported(
                "FieldRepulsionBackend::IbFft is only supported on Layout2d; use Exact or BarnesHut on Layout3d",
            ));
        }
        if is_nd {
            match config.backend {
                FieldRepulsionBackend::Exact => {}
                FieldRepulsionBackend::BarnesHut => {
                    return Err(CoreError::unsupported(
                        "FieldRepulsionBackend::BarnesHut is not supported on LayoutNd; use Layout3d for 3D Barnes-Hut, Layout2d for 2D, or Exact for arbitrary N-D",
                    ));
                }
                FieldRepulsionBackend::IbFft => {
                    return Err(CoreError::unsupported(
                        "FieldRepulsionBackend::IbFft is only supported on Layout2d; use Exact on LayoutNd",
                    ));
                }
            }
        }
        if !is_2d && !is_3d && !is_nd {
            return Err(CoreError::unsupported(
                "TFdpSolver supports Layout2d, Layout3d, and LayoutNd",
            ));
        }

        let (validate_fn, step_fn): (TFdpValidateFn<L>, TFdpStepFn<L>) = if is_2d {
            (validate_2d::<L>, step_any_2d::<L>)
        } else if is_3d {
            (validate_3d::<L>, step_any_3d::<L>)
        } else {
            (validate_nd::<L>, step_any_nd::<L>)
        };

        let ibfft = if is_2d && config.backend == FieldRepulsionBackend::IbFft {
            Some(IbFft2d::new(config.ibfft.clone())?)
        } else {
            None
        };
        let total_iterations = config.iterations;
        let status = if total_iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };
        let fft_buf_len = if ibfft.is_some() { n } else { 0 };
        let len_2d3d = if is_2d || is_3d { n } else { 0 };
        let len_3d = if is_3d { n } else { 0 };

        let total_w: f64 = edge_weights.iter().map(|&w| w as f64).sum();
        let avg_deg = if n > 0 {
            (total_w / (n as f64)).max(1.0)
        } else {
            1.0
        };
        let rep_scale = if n > 1 {
            avg_deg / ((n - 1) as f64)
        } else {
            1.0
        };
        let seed = config.seed;

        let state = TFdpState {
            config,
            rep_scale,
            num_nodes: n,
            indptr,
            indices,
            edge_weights,
            attr_x: vec![0.0; len_2d3d],
            attr_y: vec![0.0; len_2d3d],
            attr_z: vec![0.0; len_3d],
            vel_x: vec![0.0; len_2d3d],
            vel_y: vec![0.0; len_2d3d],
            vel_z: vec![0.0; len_3d],
            next_x: vec![0.0; len_2d3d],
            next_y: vec![0.0; len_2d3d],
            next_z: vec![0.0; len_3d],
            scaled_x: vec![L::Scalar::ZERO; fft_buf_len],
            scaled_y: vec![L::Scalar::ZERO; fft_buf_len],
            phi_1: vec![L::Scalar::ZERO; fft_buf_len],
            rep_x: vec![L::Scalar::ZERO; len_2d3d],
            rep_y: vec![L::Scalar::ZERO; len_2d3d],
            rep_z: vec![L::Scalar::ZERO; len_3d],
            fft_ws: Fft2dWorkspace::new(),
            ibfft,
            last_fft_stats: None,
            quadtree: ArenaQuadtree2d::new(),
            octree: ArenaOctree3d::new(),
            nd_scratch: None,
            locked_nd_dims: None,
            rng: DeterministicRng::from_seed_u64(seed),
            iteration: 0,
            total_iterations,
            status,
        };

        Ok(Self {
            state,
            validate_fn,
            step_fn,
        })
    }
}

impl<L: EuclideanLayout> TFdpSolver<L> {
    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &TFdpConfig {
        &self.state.config
    }

    /// Returns the number of vertices expected in the target layout.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.state.num_nodes
    }

    /// Returns the active repulsion evaluation backend.
    #[inline]
    pub fn backend(&self) -> FieldRepulsionBackend {
        self.state.config.backend
    }

    /// Returns the number of stored directed positive attraction entries.
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.state.indices.len()
    }

    /// Returns the cumulative number of `fft_2d` calls executed across all steps.
    #[inline]
    pub fn total_fft_2d_calls(&self) -> u64 {
        self.state.fft_ws.fft_2d_call_count()
    }

    /// Returns telemetry counters from the most recent `IbFft` step, if any.
    #[inline]
    pub fn last_fft_stats(&self) -> Option<&FieldFftStats> {
        self.state.last_fft_stats.as_ref()
    }

    /// Returns the current $\rho(t)$ repulsion weight scheduled at the active iteration.
    #[inline]
    pub fn current_rho(&self) -> f64 {
        self.state.current_rho()
    }

    /// Returns the current lifecycle status.
    #[inline]
    pub fn status(&self) -> SolverStatus {
        self.state.status
    }

    /// Returns the number of completed iterations.
    #[inline]
    pub fn iteration(&self) -> usize {
        self.state.iteration
    }

    /// Returns the total scheduled iteration budget.
    #[inline]
    pub fn total_iterations(&self) -> Option<usize> {
        Some(self.state.total_iterations)
    }

    #[cfg(test)]
    pub(super) fn state_capacities(&self) -> (usize, usize, usize, usize, usize) {
        (
            self.state.attr_x.capacity(),
            self.state.vel_x.capacity(),
            self.state.next_x.capacity(),
            self.state.rep_x.capacity(),
            self.state.phi_1.capacity(),
        )
    }
}

impl<L: EuclideanLayout> IterativeSolver for TFdpSolver<L> {
    type Target = L;

    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        (self.validate_fn)(&self.state, target)
    }

    fn step(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        self.validate_target(target)?;

        if self.state.status.is_finished() {
            return Ok(self.state.status);
        }

        if self.state.num_nodes < 2 {
            self.state.status = SolverStatus::Converged;
            return Ok(self.state.status);
        }

        if let Err(err) = (self.step_fn)(&mut self.state, target) {
            if matches!(
                err,
                CoreError::NumericalBreakdown(_) | CoreError::NonFiniteValue { .. }
            ) {
                self.state.status = SolverStatus::Failed;
            }
            return Err(err);
        }

        self.state.iteration += 1;
        self.state.status = if self.state.iteration >= self.state.total_iterations {
            SolverStatus::Completed
        } else {
            SolverStatus::Running
        };
        Ok(self.state.status)
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.state.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.state.iteration
    }

    #[inline]
    fn total_iterations(&self) -> Option<usize> {
        Some(self.state.total_iterations)
    }
}
