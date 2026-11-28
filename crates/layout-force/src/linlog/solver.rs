use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, EuclideanLayout, EuclideanStorageRef, IterativeSolver,
    Layout2d, Scalar, SolverStatus,
};

use crate::backend::DefaultForceBackend;
use crate::graph_edges::{
    extract_canonical_undirected_edges, extract_canonical_undirected_unweighted_edges,
};

use super::ISOLATED_NODE_MASS;
use super::config::{LinLogConfig, LinLogRepulsion};
use super::dim3::{LinLogBuf3d, step_3d};
use super::energy::compute_energy;
use super::nd::{LinLogNdScratch, step_nd};
use super::repulsion::LinLogRepulsionBackend;
use super::step2d::{LinLogBuf2d, step_2d};

const MAX_LINLOG_DISPLACEMENT: f64 = 50.0;
pub(super) const COULOMB_EPS_SQ: f64 = 1e-6;
pub(super) const MIN_LINLOG_SPEED: f64 = 1e-4;
pub(super) const MAX_LINLOG_SPEED: f64 = 10.0;

#[derive(Debug, Clone)]
pub(super) struct LinLogState {
    pub(super) config: LinLogConfig,
    pub(super) num_nodes: usize,
    pub(super) edges_u: Vec<u32>,
    pub(super) edges_v: Vec<u32>,
    pub(super) edges_w: Vec<f64>,
    pub(super) masses: Vec<f64>,
    pub(super) total_mass: f64,
    pub(super) swing_buf: Vec<f64>,
    pub(super) global_speed: f64,
    pub(super) max_displacement: f64,
    pub(super) iter: usize,
    pub(super) status: SolverStatus,
}

#[derive(Debug, Clone)]
enum LinLogSpaceBuf {
    Uninit,
    Dim2(LinLogBuf2d),
    Dim3(LinLogBuf3d),
    Nd(LinLogNdScratch),
}

/// Cooperative true Noack (2007/2009) LinLog / $(a, r)$-PolyLog solver implementing
/// [`IterativeSolver`].
///
/// Parameterized at compile time by target layout `L` and repulsion backend `B`:
/// - [`Layout2d`](egraph_core::Layout2d): defaults to [`BarnesHut`](crate::BarnesHut);
///   supports [`BarnesHut`](crate::BarnesHut), [`Exact`](crate::Exact), and [`ForceBackend`](crate::ForceBackend).
/// - [`Layout3d`](egraph_core::Layout3d): defaults to [`BarnesHut`](crate::BarnesHut);
///   supports [`BarnesHut`](crate::BarnesHut), [`Exact`](crate::Exact), and [`ForceBackend`](crate::ForceBackend).
/// - [`LayoutNd`](egraph_core::LayoutNd): defaults to [`Exact`](crate::Exact);
///   rejects [`BarnesHut`](crate::BarnesHut) and [`ForceBackend`](crate::ForceBackend) at compile time:
///
/// ```compile_fail
/// use egraph_core::{CsrGraph, LayoutNd};
/// use egraph_layout_force::{BarnesHut, LinLogConfig, LinLogSolver};
///
/// let g = CsrGraph::<()>::from_unweighted_edges(3, &[(0, 1), (1, 2)], false, false).unwrap();
/// let _solver = LinLogSolver::<LayoutNd<f64>, BarnesHut>::new(&g, LinLogConfig::new());
/// ```
#[derive(Debug, Clone)]
pub struct LinLogSolver<
    L: EuclideanLayout + DefaultForceBackend = Layout2d<f64>,
    B: LinLogRepulsionBackend<L> = <L as DefaultForceBackend>::Backend,
> {
    state: LinLogState,
    space: LinLogSpaceBuf,
    backend: B,
    repulsion_state: B::State,
    _layout: PhantomData<fn() -> L>,
}

impl<L, B> LinLogSolver<L, B>
where
    L: EuclideanLayout + DefaultForceBackend,
    B: LinLogRepulsionBackend<L> + Default,
{
    /// Constructs a [`LinLogSolver`] with the default backend `B::default()`.
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: LinLogConfig) -> CoreResult<Self> {
        Self::with_backend(graph, config, B::default())
    }

    /// Constructs a [`LinLogSolver`] from an unweighted [`CsrGraph<W>`] (or ignoring stored edge
    /// weights) with `B::default()`, without requiring `W: Scalar`.
    pub fn new_unweighted<W>(graph: &CsrGraph<W>, config: LinLogConfig) -> CoreResult<Self> {
        Self::with_backend_unweighted(graph, config, B::default())
    }
}

impl<L, B> LinLogSolver<L, B>
where
    L: EuclideanLayout + DefaultForceBackend,
    B: LinLogRepulsionBackend<L>,
{
    /// Constructs a [`LinLogSolver`] from a [`CsrGraph`], [`LinLogConfig`], and explicit `backend`.
    pub fn with_backend<W: Scalar>(
        graph: &CsrGraph<W>,
        config: LinLogConfig,
        backend: B,
    ) -> CoreResult<Self> {
        config.validate()?;
        backend.validate()?;
        let (n, edges_u, edges_v, raw_weights, _) =
            extract_canonical_undirected_edges(graph, "LinLogSolver::graph_weights")?;
        Self::from_canonical_edges(n, edges_u, edges_v, raw_weights, config, backend)
    }

    /// Constructs a [`LinLogSolver`] from an unweighted [`CsrGraph<W>`] (or ignoring stored edge
    /// weights) and explicit `backend`, without requiring `W: Scalar`.
    pub fn with_backend_unweighted<W>(
        graph: &CsrGraph<W>,
        config: LinLogConfig,
        backend: B,
    ) -> CoreResult<Self> {
        config.validate()?;
        backend.validate()?;
        let (n, edges_u, edges_v, raw_weights, _) =
            extract_canonical_undirected_unweighted_edges(graph)?;
        Self::from_canonical_edges(n, edges_u, edges_v, raw_weights, config, backend)
    }

    fn from_canonical_edges(
        n: usize,
        edges_u: Vec<u32>,
        edges_v: Vec<u32>,
        raw_weights: Vec<f64>,
        config: LinLogConfig,
        backend: B,
    ) -> CoreResult<Self> {
        let repulsion_state = backend.init_state(n)?;
        let max_displacement = config
            .max_displacement
            .unwrap_or_else(|| MAX_LINLOG_DISPLACEMENT * ((n as f64) / 1000.0).sqrt().max(1.0));
        let influence = config.edge_weight_influence;
        let mut edges_w = Vec::with_capacity(edges_u.len());
        let mut weighted_deg = vec![0.0_f64; n];
        for idx in 0..edges_u.len() {
            let raw_w = raw_weights[idx];
            let w_eff = if influence == 0.0 {
                1.0
            } else if (influence - 1.0).abs() <= 1e-15 {
                raw_w
            } else if raw_w == 0.0 {
                0.0
            } else {
                raw_w.powf(influence)
            };
            if !w_eff.is_finite() {
                return Err(CoreError::non_finite(
                    "LinLogSolver::edge_weight_influence::powf",
                ));
            }
            edges_w.push(w_eff);
            let u = edges_u[idx] as usize;
            let v = edges_v[idx] as usize;
            weighted_deg[u] += w_eff;
            weighted_deg[v] += w_eff;
            if !weighted_deg[u].is_finite() || !weighted_deg[v].is_finite() {
                return Err(CoreError::non_finite("LinLogSolver::weighted_degree"));
            }
        }

        let mut masses = Vec::with_capacity(n);
        let mut total_mass = 0.0_f64;
        for &d in &weighted_deg {
            let m = match config.repulsion_model {
                LinLogRepulsion::NodeRepulsion => 1.0,
                LinLogRepulsion::EdgeRepulsion => {
                    if d > 0.0 {
                        d
                    } else {
                        ISOLATED_NODE_MASS
                    }
                }
            };
            masses.push(m);
            total_mass += m;
            if !total_mass.is_finite() {
                return Err(CoreError::non_finite("LinLogSolver::total_mass"));
            }
        }

        let status = if config.iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };

        let state = LinLogState {
            config,
            num_nodes: n,
            edges_u,
            edges_v,
            edges_w,
            masses,
            total_mass,
            swing_buf: vec![0.0; n],
            global_speed: 1.0,
            max_displacement,
            iter: 0,
            status,
        };

        Ok(Self {
            state,
            space: LinLogSpaceBuf::Uninit,
            backend,
            repulsion_state,
            _layout: PhantomData,
        })
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &LinLogConfig {
        &self.state.config
    }

    /// Returns the configured repulsion backend.
    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Returns the expected number of vertices in the target layout.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.state.num_nodes
    }

    /// Returns the current lifecycle status.
    #[inline]
    pub fn status(&self) -> SolverStatus {
        self.state.status
    }

    /// Returns the number of iterations executed so far.
    #[inline]
    pub fn iteration(&self) -> usize {
        self.state.iter
    }

    /// Returns the scheduled maximum number of iterations.
    #[inline]
    pub fn total_iterations(&self) -> Option<usize> {
        Some(self.state.config.iterations)
    }

    /// Returns the per-node repulsion masses ($1$ for node-repulsion, $\deg_w(u)$ for edge-repulsion).
    #[inline]
    pub fn masses(&self) -> &[f64] {
        &self.state.masses
    }

    /// Returns the effective per-step vertex displacement cap.
    #[inline]
    pub fn max_displacement(&self) -> f64 {
        self.state.max_displacement
    }

    /// Evaluates the exact Noack (2009) $(a, r)$-LinLog energy $U_{(a,r)}(p)$ of `layout`.
    pub fn energy(&self, layout: &L) -> CoreResult<f64> {
        self.validate_target(layout)?;
        compute_energy(&self.state, layout)
    }
}

impl<L, B> IterativeSolver for LinLogSolver<L, B>
where
    L: EuclideanLayout + DefaultForceBackend,
    B: LinLogRepulsionBackend<L>,
{
    type Target = L;

    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        if target.len() != self.state.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.state.num_nodes,
                target.len(),
                "LinLogSolver::validate_target",
            ));
        }
        if let EuclideanStorageRef::FlatNd { dims, .. } = target.as_euclidean_storage() {
            if dims == 0 {
                return Err(CoreError::invalid_input("LayoutNd dims must be >= 1"));
            }
            if let LinLogSpaceBuf::Nd(scratch) = &self.space
                && scratch.dims != dims
            {
                return Err(CoreError::dimension_mismatch(
                    scratch.dims,
                    dims,
                    "LinLogSolver::validate_target::dims",
                ));
            }
        }
        target.validate_finite()
    }

    fn step(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        self.validate_target(target)?;

        if self.state.status.is_finished() {
            return Ok(self.state.status);
        }

        if self.state.num_nodes <= 1 {
            self.state.status = SolverStatus::Converged;
            return Ok(self.state.status);
        }

        if matches!(self.space, LinLogSpaceBuf::Uninit) {
            self.space = match target.as_euclidean_storage() {
                EuclideanStorageRef::Soa2d { .. } => {
                    LinLogSpaceBuf::Dim2(LinLogBuf2d::new(self.state.num_nodes))
                }
                EuclideanStorageRef::Soa3d { .. } => {
                    LinLogSpaceBuf::Dim3(LinLogBuf3d::new(self.state.num_nodes))
                }
                EuclideanStorageRef::FlatNd { dims, .. } => {
                    LinLogSpaceBuf::Nd(LinLogNdScratch::new(self.state.num_nodes, dims)?)
                }
            };
        }

        let step_res = match &mut self.space {
            LinLogSpaceBuf::Dim2(buf) => step_2d(
                &mut self.state,
                buf,
                &self.backend,
                &mut self.repulsion_state,
                target,
            ),
            LinLogSpaceBuf::Dim3(buf) => step_3d(
                &mut self.state,
                buf,
                &self.backend,
                &mut self.repulsion_state,
                target,
            ),
            LinLogSpaceBuf::Nd(scratch) => step_nd(
                &mut self.state,
                scratch,
                &self.backend,
                &mut self.repulsion_state,
                target,
            ),
            LinLogSpaceBuf::Uninit => unreachable!("space buffer initialized above"),
        };

        if let Err(err) = step_res {
            if matches!(
                err,
                CoreError::NumericalBreakdown(_)
                    | CoreError::NonFiniteValue { .. }
                    | CoreError::BackendUnavailable(_)
                    | CoreError::DeviceLimitExceeded(_)
            ) {
                self.state.status = SolverStatus::Failed;
            }
            return Err(err);
        }

        self.state.iter += 1;
        self.state.status = if self.state.iter >= self.state.config.iterations {
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
        self.state.iter
    }

    #[inline]
    fn total_iterations(&self) -> Option<usize> {
        Some(self.state.config.iterations)
    }
}
