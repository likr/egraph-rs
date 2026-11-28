use std::fmt;
use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, EuclideanLayout, IterativeSolver, Layout2d,
    Scalar, SolverStatus,
};
use egraph_neighborhood::KnnGraph;

use super::config::{FieldFftStats, TFdpConfig};
use super::repulsion::{DefaultTFdpBackend, TFdpCore, TFdpRepulsion, TFdpWorkspace};
use super::t_fdp_edges::{extract_symmetric_csr_edges, symmetrize_pair_list};

/// Cooperative, zero-per-step-allocation `t-FDP` (bounded Student-$t$ force-directed
/// placement) solver.
///
/// The repulsion backend `B` is bound at the type level via [`TFdpRepulsion<L>`], rejecting
/// unsupported `(layout, backend)` combinations at compile time:
/// - [`Layout2d`](egraph_core::Layout2d): [`IbFft`](crate::IbFft) (default),
///   [`BarnesHut`](crate::BarnesHut), [`Exact`](crate::Exact), and
///   [`TFdpBackend2d`](crate::TFdpBackend2d).
/// - [`Layout3d`](egraph_core::Layout3d): [`BarnesHut`](crate::BarnesHut) (default),
///   [`Exact`](crate::Exact), and [`TFdpBackend3d`](crate::TFdpBackend3d).
/// - [`LayoutNd`](egraph_core::LayoutNd): [`Exact`](crate::Exact) (default).
///
/// # Compile-time Space/Backend Guards
///
/// ```compile_fail
/// use egraph_core::{CsrGraph, Layout3d};
/// use egraph_layout_neighbor::{IbFft, TFdpConfig, TFdpSolver};
///
/// let g = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
/// let _ = TFdpSolver::<Layout3d<f64>, IbFft>::new(&g, TFdpConfig::new());
/// ```
///
/// ```compile_fail
/// use egraph_core::{CsrGraph, LayoutNd};
/// use egraph_layout_neighbor::{BarnesHut, TFdpConfig, TFdpSolver};
///
/// let g = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
/// let _ = TFdpSolver::<LayoutNd<f64>, BarnesHut>::new(&g, TFdpConfig::new());
/// ```
///
/// ```compile_fail
/// use egraph_core::{CsrGraph, LayoutNd};
/// use egraph_layout_neighbor::{IbFft, TFdpConfig, TFdpSolver};
///
/// let g = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
/// let _ = TFdpSolver::<LayoutNd<f64>, IbFft>::new(&g, TFdpConfig::new());
/// ```
pub struct TFdpSolver<
    L: EuclideanLayout + DefaultTFdpBackend = Layout2d<f64>,
    B: TFdpRepulsion<L> = <L as DefaultTFdpBackend>::Backend,
> {
    core: TFdpCore,
    space: TFdpWorkspace,
    backend: B,
    backend_state: B::State,
    _target: PhantomData<fn() -> L>,
}

impl<L, B> Clone for TFdpSolver<L, B>
where
    L: EuclideanLayout + DefaultTFdpBackend,
    B: TFdpRepulsion<L> + Clone,
{
    fn clone(&self) -> Self {
        Self {
            core: self.core.clone(),
            space: self.space.clone(),
            backend: self.backend.clone(),
            backend_state: self.backend_state.clone(),
            _target: PhantomData,
        }
    }
}

impl<L, B> fmt::Debug for TFdpSolver<L, B>
where
    L: EuclideanLayout + DefaultTFdpBackend,
    B: TFdpRepulsion<L> + fmt::Debug,
    B::State: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TFdpSolver")
            .field("core", &self.core)
            .field("space", &self.space)
            .field("backend", &self.backend)
            .field("backend_state", &self.backend_state)
            .finish()
    }
}

impl<L, B> TFdpSolver<L, B>
where
    L: EuclideanLayout + DefaultTFdpBackend,
    B: TFdpRepulsion<L> + Default,
{
    /// Constructs a `t-FDP` [`TFdpSolver`] directly from a [`CsrGraph`] using `B::default()`.
    #[inline]
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: TFdpConfig) -> CoreResult<Self> {
        Self::new_with_backend(graph, config, B::default())
    }

    /// Constructs a `t-FDP` [`TFdpSolver`] from a validated [`KnnGraph`] using `B::default()`,
    /// converting neighbor distances $d_{ij} \ge 0$ to symmetric weights $1 / (1 + d_{ij})$.
    #[inline]
    pub fn from_knn(knn: &KnnGraph, config: TFdpConfig) -> CoreResult<Self> {
        Self::from_knn_with_backend(knn, config, B::default())
    }
}

impl<L, B> TFdpSolver<L, B>
where
    L: EuclideanLayout + DefaultTFdpBackend,
    B: TFdpRepulsion<L>,
{
    /// Constructs a `t-FDP` [`TFdpSolver`] directly from a [`CsrGraph`] with an explicit
    /// repulsion `backend`.
    pub fn new_with_backend<W: Scalar>(
        graph: &CsrGraph<W>,
        config: TFdpConfig,
        backend: B,
    ) -> CoreResult<Self> {
        config.validate()?;
        backend.validate_config()?;
        let n = graph.num_nodes();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_csr("n_nodes exceeds u32::MAX"));
        }
        if let Some(ws) = graph.weights() {
            for &w in ws {
                let wf = w.to_f64();
                if !wf.is_finite() {
                    return Err(CoreError::non_finite(
                        "TFdpSolver::new_with_backend::weights",
                    ));
                }
                if wf < 0.0 {
                    return Err(CoreError::invalid_input(format!(
                        "graph edge weights must be >= 0.0, got {wf}"
                    )));
                }
            }
        }

        let (indptr, indices, edge_weights) = extract_symmetric_csr_edges(graph)?;
        Self::build_inner(n, indptr, indices, edge_weights, config, backend)
    }

    /// Constructs a `t-FDP` [`TFdpSolver`] from a validated [`KnnGraph`] with an explicit
    /// repulsion `backend`, converting neighbor distances $d_{ij} \ge 0$ to symmetric weights
    /// $1 / (1 + d_{ij})$.
    pub fn from_knn_with_backend(
        knn: &KnnGraph,
        config: TFdpConfig,
        backend: B,
    ) -> CoreResult<Self> {
        config.validate()?;
        backend.validate_config()?;
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
                    return Err(CoreError::non_finite(
                        "TFdpSolver::from_knn_with_backend::distances",
                    ));
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
        Self::build_inner(n, indptr, indices, edge_weights, config, backend)
    }

    fn build_inner(
        n: usize,
        indptr: Vec<u32>,
        indices: Vec<u32>,
        edge_weights: Vec<f32>,
        config: TFdpConfig,
        backend: B,
    ) -> CoreResult<Self> {
        let backend_state = backend.init_state(n)?;
        let space = L::init_tfdp_workspace(n);

        let total_iterations = config.iterations;
        let status = if total_iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };

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

        let core = TFdpCore {
            config,
            rep_scale,
            num_nodes: n,
            indptr,
            indices,
            edge_weights,
            rng: DeterministicRng::from_seed_u64(seed),
            iteration: 0,
            total_iterations,
            status,
        };

        Ok(Self {
            core,
            space,
            backend,
            backend_state,
            _target: PhantomData,
        })
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &TFdpConfig {
        &self.core.config
    }

    /// Returns the repulsion backend configuration.
    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Returns the number of vertices expected in the target layout.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.core.num_nodes
    }

    /// Returns the number of stored directed positive attraction entries.
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.core.indices.len()
    }

    /// Returns the symmetrized positive attraction edges `(indptr, indices, edge_weights)` in CSR format.
    #[inline]
    pub fn edge_csr(&self) -> (&[u32], &[u32], &[f32]) {
        (
            &self.core.indptr,
            &self.core.indices,
            &self.core.edge_weights,
        )
    }

    /// Returns the cumulative number of `fft_2d` calls executed across all steps.
    #[inline]
    pub fn total_fft_2d_calls(&self) -> u64 {
        self.backend.total_fft_2d_calls(&self.backend_state)
    }

    /// Returns telemetry counters from the most recent `IbFft` step, if any.
    #[inline]
    pub fn last_fft_stats(&self) -> Option<&FieldFftStats> {
        self.backend.last_fft_stats(&self.backend_state)
    }

    /// Returns the current $\rho(t)$ repulsion weight scheduled at the active iteration.
    #[inline]
    pub fn current_rho(&self) -> f64 {
        self.core.current_rho()
    }

    /// Returns the current lifecycle status.
    #[inline]
    pub fn status(&self) -> SolverStatus {
        self.core.status
    }

    /// Returns the number of completed iterations.
    #[inline]
    pub fn iteration(&self) -> usize {
        self.core.iteration
    }

    /// Returns the total scheduled iteration budget.
    #[inline]
    pub fn total_iterations(&self) -> Option<usize> {
        Some(self.core.total_iterations)
    }

    #[cfg(test)]
    pub(super) fn state_capacities(&self) -> (usize, usize, usize, usize) {
        self.space.capacities()
    }
}

impl<L, B> IterativeSolver for TFdpSolver<L, B>
where
    L: EuclideanLayout + DefaultTFdpBackend,
    B: TFdpRepulsion<L>,
{
    type Target = L;

    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        target.validate_tfdp_target(&self.core, &self.space)
    }

    fn step(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        self.validate_target(target)?;

        if self.core.status.is_finished() {
            return Ok(self.core.status);
        }

        if self.core.num_nodes < 2 {
            self.core.status = SolverStatus::Converged;
            return Ok(self.core.status);
        }

        if let Err(err) = target.step_tfdp_target(
            &mut self.core,
            &mut self.space,
            &self.backend,
            &mut self.backend_state,
        ) {
            if matches!(
                err,
                CoreError::NumericalBreakdown(_)
                    | CoreError::NonFiniteValue { .. }
                    | CoreError::BackendUnavailable(_)
                    | CoreError::DeviceLimitExceeded(_)
            ) {
                self.core.status = SolverStatus::Failed;
            }
            return Err(err);
        }

        self.core.iteration += 1;
        self.core.status = if self.core.iteration >= self.core.total_iterations {
            SolverStatus::Completed
        } else {
            SolverStatus::Running
        };
        Ok(self.core.status)
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.core.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.core.iteration
    }

    #[inline]
    fn total_iterations(&self) -> Option<usize> {
        Some(self.core.total_iterations)
    }
}
