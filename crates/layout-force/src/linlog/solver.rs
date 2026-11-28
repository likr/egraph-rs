use std::any::{Any, TypeId};
use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, EuclideanLayout, IterativeSolver, Layout2d, Layout3d,
    LayoutNd, Scalar, SolverStatus,
};
use egraph_spatial_trees::{ArenaOctree3d, ArenaQuadtree2d};

use crate::graph_edges::{
    extract_canonical_undirected_edges, extract_canonical_undirected_unweighted_edges,
};

use super::ISOLATED_NODE_MASS;
use super::config::{LinLogConfig, LinLogRepulsion};
use super::dim3::step_3d;
use super::energy::compute_energy;
use super::nd::{LinLogNdScratch, step_nd};
use super::quadtree::GenRepulsionQuadtree;
use super::step2d::step_2d;

const MAX_LINLOG_DISPLACEMENT: f64 = 50.0;
pub(super) const COULOMB_EPS_SQ: f64 = 1e-6;
pub(super) const MIN_LINLOG_SPEED: f64 = 1e-4;
pub(super) const MAX_LINLOG_SPEED: f64 = 10.0;

pub(super) type LinLogValidateFn<L> = fn(&LinLogState, &L) -> CoreResult<()>;
pub(super) type LinLogStepFn<L> = fn(&mut LinLogState, &mut L) -> CoreResult<()>;

#[derive(Debug, Clone)]
pub(super) struct LinLogState {
    pub(super) config: LinLogConfig,
    pub(super) num_nodes: usize,
    pub(super) edges_u: Vec<u32>,
    pub(super) edges_v: Vec<u32>,
    pub(super) edges_w: Vec<f64>,
    pub(super) masses: Vec<f64>,
    pub(super) total_mass: f64,
    pub(super) x_buf: Vec<f64>,
    pub(super) y_buf: Vec<f64>,
    pub(super) z_buf: Vec<f64>,
    pub(super) fx: Vec<f64>,
    pub(super) fy: Vec<f64>,
    pub(super) fz: Vec<f64>,
    pub(super) prev_fx: Vec<f64>,
    pub(super) prev_fy: Vec<f64>,
    pub(super) prev_fz: Vec<f64>,
    pub(super) swing_buf: Vec<f64>,
    pub(super) att_hess: Vec<f64>,
    pub(super) rep_x: Vec<f64>,
    pub(super) rep_y: Vec<f64>,
    pub(super) rep_3d_flat: Vec<f64>,
    pub(super) quadtree: ArenaQuadtree2d<f64>,
    pub(super) gen_quadtree: GenRepulsionQuadtree,
    pub(super) octree: ArenaOctree3d<f64>,
    pub(super) global_speed: f64,
    pub(super) max_displacement: f64,
    pub(super) locked_nd_dims: Option<usize>,
    pub(super) nd_scratch: Option<LinLogNdScratch>,
    pub(super) iter: usize,
    pub(super) status: SolverStatus,
}

/// Cooperative true Noack (2007/2009) LinLog / $(a, r)$-PolyLog solver implementing
/// [`IterativeSolver`].
///
/// Supports:
/// - [`Layout2d`]: `use_barnes_hut` (`ArenaQuadtree2d` / `GenRepulsionQuadtree`) and exact.
/// - [`Layout3d`]: `use_barnes_hut` (`ArenaOctree3d`) and exact.
/// - [`LayoutNd`]: exact only (`use_barnes_hut` must be `false`).
#[derive(Debug, Clone)]
pub struct LinLogSolver<L: EuclideanLayout = Layout2d<f64>> {
    state: LinLogState,
    validate_fn: LinLogValidateFn<L>,
    step_fn: LinLogStepFn<L>,
    _layout: PhantomData<fn() -> L>,
}

impl<L: EuclideanLayout + 'static> LinLogSolver<L> {
    /// Constructs a [`LinLogSolver`] from a [`CsrGraph`] and [`LinLogConfig`].
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: LinLogConfig) -> CoreResult<Self> {
        config.validate()?;
        let (n, edges_u, edges_v, raw_weights, _) =
            extract_canonical_undirected_edges(graph, "LinLogSolver::graph_weights")?;
        Self::from_canonical_edges(n, edges_u, edges_v, raw_weights, config)
    }

    /// Constructs a [`LinLogSolver`] from an unweighted [`CsrGraph<W>`] (or ignoring stored edge
    /// weights) without requiring `W: Scalar`.
    pub fn new_unweighted<W>(graph: &CsrGraph<W>, config: LinLogConfig) -> CoreResult<Self> {
        config.validate()?;
        let (n, edges_u, edges_v, raw_weights, _) =
            extract_canonical_undirected_unweighted_edges(graph)?;
        Self::from_canonical_edges(n, edges_u, edges_v, raw_weights, config)
    }

    fn from_canonical_edges(
        n: usize,
        edges_u: Vec<u32>,
        edges_v: Vec<u32>,
        raw_weights: Vec<f64>,
        config: LinLogConfig,
    ) -> CoreResult<Self> {
        let tid = TypeId::of::<L>();
        let is_2d = tid == TypeId::of::<Layout2d<L::Scalar>>();
        let is_3d = tid == TypeId::of::<Layout3d<L::Scalar>>();
        let is_nd = tid == TypeId::of::<LayoutNd<L::Scalar>>();

        if is_nd && config.use_barnes_hut {
            return Err(CoreError::unsupported(
                "LinLog Barnes-Hut repulsion is not supported on LayoutNd; use Layout3d for 3D Barnes-Hut, Layout2d for 2D, or set use_barnes_hut = false (Exact) for arbitrary N-D",
            ));
        }
        if !is_2d && !is_3d && !is_nd {
            return Err(CoreError::unsupported(
                "LinLogSolver supports Layout2d, Layout3d, and LayoutNd",
            ));
        }

        let (validate_fn, step_fn): (LinLogValidateFn<L>, LinLogStepFn<L>) = if is_2d {
            (validate_basic::<L>, step_any_2d::<L>)
        } else if is_3d {
            (validate_basic::<L>, step_any_3d::<L>)
        } else {
            (validate_nd::<L>, step_any_nd::<L>)
        };

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
            edges_w.push(w_eff);
            weighted_deg[edges_u[idx] as usize] += w_eff;
            weighted_deg[edges_v[idx] as usize] += w_eff;
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
        }

        let len_2d3d = if is_2d || is_3d { n } else { 0 };
        let len_3d = if is_3d { n } else { 0 };
        let rep_2d_len = if is_2d && config.use_barnes_hut { n } else { 0 };
        let rep_3d_len = if is_3d && config.use_barnes_hut {
            n * 3
        } else {
            0
        };
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
            x_buf: vec![0.0; len_2d3d],
            y_buf: vec![0.0; len_2d3d],
            z_buf: vec![0.0; len_3d],
            fx: vec![0.0; len_2d3d],
            fy: vec![0.0; len_2d3d],
            fz: vec![0.0; len_3d],
            prev_fx: vec![0.0; len_2d3d],
            prev_fy: vec![0.0; len_2d3d],
            prev_fz: vec![0.0; len_3d],
            swing_buf: vec![0.0; n],
            att_hess: vec![0.0; len_2d3d],
            rep_x: vec![0.0; rep_2d_len],
            rep_y: vec![0.0; rep_2d_len],
            rep_3d_flat: vec![0.0; rep_3d_len],
            quadtree: ArenaQuadtree2d::new(),
            gen_quadtree: GenRepulsionQuadtree::default(),
            octree: ArenaOctree3d::new(),
            global_speed: 1.0,
            max_displacement,
            locked_nd_dims: None,
            nd_scratch: None,
            iter: 0,
            status,
        };

        Ok(Self {
            state,
            validate_fn,
            step_fn,
            _layout: PhantomData,
        })
    }
}

impl<L: EuclideanLayout> LinLogSolver<L> {
    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &LinLogConfig {
        &self.state.config
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

impl<L: EuclideanLayout> IterativeSolver for LinLogSolver<L> {
    type Target = L;

    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        (self.validate_fn)(&self.state, target)
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

        if let Err(err) = (self.step_fn)(&mut self.state, target) {
            if matches!(
                err,
                CoreError::NumericalBreakdown(_) | CoreError::NonFiniteValue { .. }
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

fn validate_basic<L: EuclideanLayout>(state: &LinLogState, target: &L) -> CoreResult<()> {
    if target.len() != state.num_nodes {
        return Err(CoreError::dimension_mismatch(
            state.num_nodes,
            target.len(),
            "LinLogSolver::validate_target",
        ));
    }
    target.validate_finite()
}

fn validate_nd<L: EuclideanLayout>(state: &LinLogState, target: &L) -> CoreResult<()> {
    if target.len() != state.num_nodes {
        return Err(CoreError::dimension_mismatch(
            state.num_nodes,
            target.len(),
            "LinLogSolver::validate_target",
        ));
    }
    let dims = target.dims();
    if dims == 0 {
        return Err(CoreError::invalid_input("LayoutNd dims must be >= 1"));
    }
    if let Some(prev) = state.locked_nd_dims
        && prev != dims
    {
        return Err(CoreError::dimension_mismatch(
            prev,
            dims,
            "LinLogSolver::validate_target::dims",
        ));
    }
    target.validate_finite()
}

fn step_any_2d<L: EuclideanLayout + 'static>(
    state: &mut LinLogState,
    target: &mut L,
) -> CoreResult<()> {
    let layout = (target as &mut dyn Any)
        .downcast_mut::<Layout2d<L::Scalar>>()
        .ok_or_else(|| CoreError::numerical_breakdown("LinLogSolver target type mismatch"))?;
    step_2d(state, layout)
}

fn step_any_3d<L: EuclideanLayout + 'static>(
    state: &mut LinLogState,
    target: &mut L,
) -> CoreResult<()> {
    let layout = (target as &mut dyn Any)
        .downcast_mut::<Layout3d<L::Scalar>>()
        .ok_or_else(|| CoreError::numerical_breakdown("LinLogSolver target type mismatch"))?;
    step_3d(state, layout)
}

fn step_any_nd<L: EuclideanLayout + 'static>(
    state: &mut LinLogState,
    target: &mut L,
) -> CoreResult<()> {
    let layout = (target as &mut dyn Any)
        .downcast_mut::<LayoutNd<L::Scalar>>()
        .ok_or_else(|| CoreError::numerical_breakdown("LinLogSolver target type mismatch"))?;
    step_nd(state, layout)
}
