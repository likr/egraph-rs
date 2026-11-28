use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, EuclideanLayout, EuclideanStorageRef, IterativeSolver,
    Layout2d, Scalar, SolverStatus,
};

use crate::backend::DefaultForceBackend;
use crate::graph_edges::{
    extract_canonical_undirected_edges, extract_canonical_undirected_unweighted_edges,
};

use super::config::{ForceAtlas2Config, validate_node_radii};
use super::dim3::{Fa2Buf3d, step_3d};
use super::nd::{ForceAtlas2NdScratch, step_nd};
use super::repulsion::ForceRepulsion;
use super::step2d::{Fa2Buf2d, step_2d};

pub(crate) const MAX_FA2_DISPLACEMENT: f64 = 50.0;
pub(super) const COULOMB_EPS_SQ: f64 = 1e-6;
pub(super) const MIN_FA2_SPEED: f64 = 1e-4;

#[derive(Debug, Clone)]
pub(super) struct ForceAtlas2State {
    pub(super) config: ForceAtlas2Config,
    pub(super) num_nodes: usize,
    pub(super) edges_u: Vec<u32>,
    pub(super) edges_v: Vec<u32>,
    pub(super) edges_w: Vec<f64>,
    pub(super) masses: Vec<f64>,
    pub(super) node_radii: Vec<f64>,
    pub(super) swing_buf: Vec<f64>,
    pub(super) global_speed: f64,
    pub(super) max_displacement: f64,
    pub(super) iter: usize,
    pub(super) status: SolverStatus,
}

#[derive(Debug, Clone)]
enum Fa2SpaceBuf {
    Uninit,
    Dim2(Fa2Buf2d),
    Dim3(Fa2Buf3d),
    Nd(ForceAtlas2NdScratch),
}

/// Cooperative ForceAtlas2 layout solver implementing [`IterativeSolver`].
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
/// use egraph_layout_force::{BarnesHut, ForceAtlas2Config, ForceAtlas2Solver};
///
/// let g = CsrGraph::<()>::from_unweighted_edges(3, &[(0, 1), (1, 2)], false, false).unwrap();
/// let _solver = ForceAtlas2Solver::<LayoutNd<f64>, BarnesHut>::new(&g, ForceAtlas2Config::new());
/// ```
#[derive(Debug)]
pub struct ForceAtlas2Solver<
    L: EuclideanLayout + DefaultForceBackend = Layout2d<f64>,
    B: ForceRepulsion<L> = <L as DefaultForceBackend>::Backend,
> {
    state: ForceAtlas2State,
    space: Fa2SpaceBuf,
    backend: B,
    repulsion_state: B::State,
    _layout: PhantomData<fn() -> L>,
}

impl<L, B> Clone for ForceAtlas2Solver<L, B>
where
    L: EuclideanLayout + DefaultForceBackend,
    B: ForceRepulsion<L> + Clone,
    B::State: Clone,
{
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            space: self.space.clone(),
            backend: self.backend.clone(),
            repulsion_state: self.repulsion_state.clone(),
            _layout: PhantomData,
        }
    }
}

impl<L, B> ForceAtlas2Solver<L, B>
where
    L: EuclideanLayout + DefaultForceBackend,
    B: ForceRepulsion<L> + Default,
{
    /// Constructs a [`ForceAtlas2Solver`] with the default backend `B::default()` using
    /// `config.default_node_radius`.
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: ForceAtlas2Config) -> CoreResult<Self> {
        Self::with_backend(graph, config, B::default())
    }

    /// Constructs a [`ForceAtlas2Solver`] from an unweighted [`CsrGraph<W>`] (or ignoring stored
    /// edge weights) with `B::default()` and `config.default_node_radius`, without requiring `W: Scalar`.
    pub fn new_unweighted<W>(graph: &CsrGraph<W>, config: ForceAtlas2Config) -> CoreResult<Self> {
        Self::with_backend_unweighted(graph, config, B::default())
    }

    /// Constructs a [`ForceAtlas2Solver`] with explicit per-node radii and `B::default()`.
    pub fn with_node_radii<W: Scalar>(
        graph: &CsrGraph<W>,
        config: ForceAtlas2Config,
        node_radii: &[f64],
    ) -> CoreResult<Self> {
        Self::with_node_radii_and_backend(graph, config, node_radii, B::default())
    }

    /// Constructs a [`ForceAtlas2Solver`] from an unweighted [`CsrGraph<W>`] with explicit
    /// per-node radii and `B::default()`, without requiring `W: Scalar`.
    pub fn with_node_radii_unweighted<W>(
        graph: &CsrGraph<W>,
        config: ForceAtlas2Config,
        node_radii: &[f64],
    ) -> CoreResult<Self> {
        Self::with_node_radii_and_backend_unweighted(graph, config, node_radii, B::default())
    }
}

impl<L, B> ForceAtlas2Solver<L, B>
where
    L: EuclideanLayout + DefaultForceBackend,
    B: ForceRepulsion<L>,
{
    /// Constructs a [`ForceAtlas2Solver`] with an explicit repulsion `backend` using
    /// `config.default_node_radius`.
    pub fn with_backend<W: Scalar>(
        graph: &CsrGraph<W>,
        config: ForceAtlas2Config,
        backend: B,
    ) -> CoreResult<Self> {
        config.validate()?;
        backend.validate()?;
        let n = graph.num_nodes();
        let radii = vec![config.default_node_radius; n];
        Self::with_node_radii_and_backend(graph, config, &radii, backend)
    }

    /// Constructs a [`ForceAtlas2Solver`] from an unweighted [`CsrGraph<W>`] (or ignoring stored
    /// edge weights) with an explicit repulsion `backend`, without requiring `W: Scalar`.
    pub fn with_backend_unweighted<W>(
        graph: &CsrGraph<W>,
        config: ForceAtlas2Config,
        backend: B,
    ) -> CoreResult<Self> {
        config.validate()?;
        backend.validate()?;
        let n = graph.num_nodes();
        let radii = vec![config.default_node_radius; n];
        Self::with_node_radii_and_backend_unweighted(graph, config, &radii, backend)
    }

    /// Constructs a [`ForceAtlas2Solver`] with explicit per-node radii and an explicit `backend`.
    pub fn with_node_radii_and_backend<W: Scalar>(
        graph: &CsrGraph<W>,
        config: ForceAtlas2Config,
        node_radii: &[f64],
        backend: B,
    ) -> CoreResult<Self> {
        config.validate()?;
        backend.validate()?;
        validate_node_radii(graph.num_nodes(), node_radii)?;
        let (n, edges_u, edges_v, raw_weights, deg) =
            extract_canonical_undirected_edges(graph, "ForceAtlas2Solver::graph_weights")?;
        Self::from_canonical_edges(
            n,
            edges_u,
            edges_v,
            raw_weights,
            deg,
            config,
            node_radii,
            backend,
        )
    }

    /// Constructs a [`ForceAtlas2Solver`] from an unweighted [`CsrGraph<W>`] with explicit
    /// per-node radii and an explicit `backend`, without requiring `W: Scalar`.
    pub fn with_node_radii_and_backend_unweighted<W>(
        graph: &CsrGraph<W>,
        config: ForceAtlas2Config,
        node_radii: &[f64],
        backend: B,
    ) -> CoreResult<Self> {
        config.validate()?;
        backend.validate()?;
        validate_node_radii(graph.num_nodes(), node_radii)?;
        let (n, edges_u, edges_v, raw_weights, deg) =
            extract_canonical_undirected_unweighted_edges(graph)?;
        Self::from_canonical_edges(
            n,
            edges_u,
            edges_v,
            raw_weights,
            deg,
            config,
            node_radii,
            backend,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_canonical_edges(
        n: usize,
        edges_u: Vec<u32>,
        edges_v: Vec<u32>,
        raw_weights: Vec<f64>,
        deg: Vec<usize>,
        config: ForceAtlas2Config,
        node_radii: &[f64],
        backend: B,
    ) -> CoreResult<Self> {
        let repulsion_state = backend.init_state(n)?;
        let max_displacement = config
            .max_displacement
            .unwrap_or_else(|| MAX_FA2_DISPLACEMENT * ((n as f64) / 1000.0).sqrt().max(1.0));

        let mut masses = Vec::with_capacity(n);
        let mut total_mass = 0.0_f64;
        for &d in &deg {
            let m = (d as f64) + 1.0;
            masses.push(m);
            total_mass += m;
        }
        let mean_mass = if n > 0 { total_mass / (n as f64) } else { 1.0 };

        let influence = config.edge_weight_influence;
        let mut edges_w = Vec::with_capacity(edges_u.len());
        for idx in 0..edges_u.len() {
            let raw_w = raw_weights[idx];
            let w_pow = if influence == 0.0 {
                1.0
            } else if (influence - 1.0).abs() <= 1e-15 {
                raw_w
            } else if raw_w == 0.0 {
                0.0
            } else {
                raw_w.powf(influence)
            };
            if !w_pow.is_finite() {
                return Err(CoreError::non_finite(
                    "ForceAtlas2Solver::edge_weight_influence::powf",
                ));
            }
            let eff_w = if config.outbound_attraction_distribution {
                let mu = masses[edges_u[idx] as usize];
                let mv = masses[edges_v[idx] as usize];
                w_pow * mean_mass * 0.5 * (1.0 / mu + 1.0 / mv)
            } else {
                w_pow
            };
            if !eff_w.is_finite() {
                return Err(CoreError::non_finite(
                    "ForceAtlas2Solver::outbound_attraction_distribution::eff_w",
                ));
            }
            edges_w.push(eff_w);
        }

        let status = if config.iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };

        let state = ForceAtlas2State {
            config,
            num_nodes: n,
            edges_u,
            edges_v,
            edges_w,
            masses,
            node_radii: node_radii.to_vec(),
            swing_buf: vec![0.0; n],
            global_speed: 1.0,
            max_displacement,
            iter: 0,
            status,
        };

        Ok(Self {
            state,
            space: Fa2SpaceBuf::Uninit,
            backend,
            repulsion_state,
            _layout: PhantomData,
        })
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &ForceAtlas2Config {
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

    /// Returns the number of canonical undirected edges.
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.state.edges_u.len()
    }

    /// Returns the current global adaptive speed scalar.
    #[inline]
    pub fn global_speed(&self) -> f64 {
        self.state.global_speed
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

    /// Returns the canonical undirected edges `(u, v, weights)` with effective weights applied.
    #[inline]
    pub fn edges(&self) -> (&[u32], &[u32], &[f64]) {
        (
            &self.state.edges_u,
            &self.state.edges_v,
            &self.state.edges_w,
        )
    }

    /// Returns the per-node ForceAtlas2 masses `deg + 1`.
    #[inline]
    pub fn masses(&self) -> &[f64] {
        &self.state.masses
    }

    /// Returns the per-node collision radii.
    #[inline]
    pub fn node_radii(&self) -> &[f64] {
        &self.state.node_radii
    }

    /// Returns the effective per-step vertex displacement cap.
    #[inline]
    pub fn max_displacement(&self) -> f64 {
        self.state.max_displacement
    }

    /// Returns `true` if ForceAtlas2 logarithmic attraction mode is enabled.
    #[inline]
    pub fn linlog_mode(&self) -> bool {
        self.state.config.linlog_mode
    }

    /// Returns `true` if strong gravity mode is enabled.
    #[inline]
    pub fn strong_gravity(&self) -> bool {
        self.state.config.strong_gravity
    }
}

impl<L, B> IterativeSolver for ForceAtlas2Solver<L, B>
where
    L: EuclideanLayout + DefaultForceBackend,
    B: ForceRepulsion<L>,
{
    type Target = L;

    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        if target.len() != self.state.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.state.num_nodes,
                target.len(),
                "ForceAtlas2Solver::validate_target",
            ));
        }
        if let EuclideanStorageRef::FlatNd { dims, .. } = target.as_euclidean_storage() {
            if dims == 0 {
                return Err(CoreError::invalid_input("LayoutNd dims must be >= 1"));
            }
            if let Fa2SpaceBuf::Nd(scratch) = &self.space
                && scratch.dims != dims
            {
                return Err(CoreError::dimension_mismatch(
                    scratch.dims,
                    dims,
                    "ForceAtlas2Solver::validate_target::dims",
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

        if matches!(self.space, Fa2SpaceBuf::Uninit) {
            self.space = match target.as_euclidean_storage() {
                EuclideanStorageRef::Soa2d { .. } => {
                    Fa2SpaceBuf::Dim2(Fa2Buf2d::new(self.state.num_nodes))
                }
                EuclideanStorageRef::Soa3d { .. } => {
                    Fa2SpaceBuf::Dim3(Fa2Buf3d::new(self.state.num_nodes))
                }
                EuclideanStorageRef::FlatNd { dims, .. } => {
                    Fa2SpaceBuf::Nd(ForceAtlas2NdScratch::new(self.state.num_nodes, dims)?)
                }
            };
        }

        let step_res = match &mut self.space {
            Fa2SpaceBuf::Dim2(buf) => step_2d(
                &mut self.state,
                buf,
                &self.backend,
                &mut self.repulsion_state,
                target,
            ),
            Fa2SpaceBuf::Dim3(buf) => step_3d(
                &mut self.state,
                buf,
                &self.backend,
                &mut self.repulsion_state,
                target,
            ),
            Fa2SpaceBuf::Nd(scratch) => step_nd(
                &mut self.state,
                scratch,
                &self.backend,
                &mut self.repulsion_state,
                target,
            ),
            Fa2SpaceBuf::Uninit => unreachable!("space buffer initialized above"),
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
