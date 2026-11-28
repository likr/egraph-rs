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

use super::config::ForceAtlas2Config;
use super::dim3::step_3d;
use super::nd::{ForceAtlas2NdScratch, step_nd};
use super::overlap_quadtree::OverlapQuadtree;
use super::step2d::step_2d;

pub(crate) const MAX_FA2_DISPLACEMENT: f64 = 50.0;
pub(super) const COULOMB_EPS_SQ: f64 = 1e-6;
pub(super) const MIN_FA2_SPEED: f64 = 1e-4;

pub(super) type Fa2ValidateFn<L> = fn(&ForceAtlas2State, &L) -> CoreResult<()>;
pub(super) type Fa2StepFn<L> = fn(&mut ForceAtlas2State, &mut L) -> CoreResult<()>;

#[derive(Debug, Clone)]
pub(super) struct ForceAtlas2State {
    pub(super) config: ForceAtlas2Config,
    pub(super) num_nodes: usize,
    pub(super) edges_u: Vec<u32>,
    pub(super) edges_v: Vec<u32>,
    pub(super) edges_w: Vec<f64>,
    pub(super) masses: Vec<f64>,
    pub(super) node_radii: Vec<f64>,
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
    pub(super) rep_x: Vec<f64>,
    pub(super) rep_y: Vec<f64>,
    pub(super) rep_z: Vec<f64>,
    pub(super) quadtree: ArenaQuadtree2d<f64>,
    pub(super) octree: ArenaOctree3d<f64>,
    pub(super) overlap_tree: OverlapQuadtree,
    pub(super) global_speed: f64,
    pub(super) max_displacement: f64,
    pub(super) locked_nd_dims: Option<usize>,
    pub(super) nd_scratch: Option<ForceAtlas2NdScratch>,
    pub(super) iter: usize,
    pub(super) status: SolverStatus,
}

/// Cooperative ForceAtlas2 layout solver implementing [`IterativeSolver`].
///
/// Bound at compile time to a Euclidean layout type `L`:
/// - [`Layout2d`]: supports both $O(N \log N)$ [`ArenaQuadtree2d`] Barnes-Hut (or radius-aware
///   overlap quadtree) and $O(N^2)$ exact repulsion.
/// - [`Layout3d`]: supports $O(N \log N)$ [`ArenaOctree3d`] Barnes-Hut and $O(N^2)$ exact
///   repulsion (including 3D overlap repulsion).
/// - [`LayoutNd`]: supports $O(N^2)$ exact repulsion for arbitrary dimensions (`use_barnes_hut`
///   must be `false`).
#[derive(Debug, Clone)]
pub struct ForceAtlas2Solver<L: EuclideanLayout = Layout2d<f64>> {
    state: ForceAtlas2State,
    validate_fn: Fa2ValidateFn<L>,
    step_fn: Fa2StepFn<L>,
    _layout: PhantomData<fn() -> L>,
}

impl<L: EuclideanLayout + 'static> ForceAtlas2Solver<L> {
    /// Constructs a [`ForceAtlas2Solver`] from a [`CsrGraph`] using `config.default_node_radius`.
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: ForceAtlas2Config) -> CoreResult<Self> {
        config.validate()?;
        let n = graph.num_nodes();
        let radii = vec![config.default_node_radius; n];
        Self::with_node_radii(graph, config, &radii)
    }

    /// Constructs a [`ForceAtlas2Solver`] from an unweighted [`CsrGraph<W>`] (or ignoring stored
    /// edge weights) using `config.default_node_radius`, without requiring `W: Scalar`.
    pub fn new_unweighted<W>(graph: &CsrGraph<W>, config: ForceAtlas2Config) -> CoreResult<Self> {
        config.validate()?;
        let n = graph.num_nodes();
        let radii = vec![config.default_node_radius; n];
        Self::with_node_radii_unweighted(graph, config, &radii)
    }

    /// Constructs a [`ForceAtlas2Solver`] from a [`CsrGraph`] with explicit per-node radii.
    pub fn with_node_radii<W: Scalar>(
        graph: &CsrGraph<W>,
        config: ForceAtlas2Config,
        node_radii: &[f64],
    ) -> CoreResult<Self> {
        config.validate()?;
        validate_node_radii(graph.num_nodes(), node_radii)?;
        let (n, edges_u, edges_v, raw_weights, deg) =
            extract_canonical_undirected_edges(graph, "ForceAtlas2Solver::graph_weights")?;
        Self::from_canonical_edges(n, edges_u, edges_v, raw_weights, deg, config, node_radii)
    }

    /// Constructs a [`ForceAtlas2Solver`] from an unweighted [`CsrGraph<W>`] (or ignoring stored
    /// edge weights) with explicit per-node radii, without requiring `W: Scalar`.
    pub fn with_node_radii_unweighted<W>(
        graph: &CsrGraph<W>,
        config: ForceAtlas2Config,
        node_radii: &[f64],
    ) -> CoreResult<Self> {
        config.validate()?;
        validate_node_radii(graph.num_nodes(), node_radii)?;
        let (n, edges_u, edges_v, raw_weights, deg) =
            extract_canonical_undirected_unweighted_edges(graph)?;
        Self::from_canonical_edges(n, edges_u, edges_v, raw_weights, deg, config, node_radii)
    }

    fn from_canonical_edges(
        n: usize,
        edges_u: Vec<u32>,
        edges_v: Vec<u32>,
        raw_weights: Vec<f64>,
        deg: Vec<usize>,
        config: ForceAtlas2Config,
        node_radii: &[f64],
    ) -> CoreResult<Self> {
        let tid = TypeId::of::<L>();
        let is_2d = tid == TypeId::of::<Layout2d<L::Scalar>>();
        let is_3d = tid == TypeId::of::<Layout3d<L::Scalar>>();
        let is_nd = tid == TypeId::of::<LayoutNd<L::Scalar>>();

        if is_nd && config.use_barnes_hut {
            return Err(CoreError::unsupported(
                "ForceAtlas2 Barnes-Hut repulsion is not supported on LayoutNd; use Layout3d for 3D Barnes-Hut, Layout2d for 2D, or set use_barnes_hut = false (Exact) for arbitrary N-D",
            ));
        }
        if !is_2d && !is_3d && !is_nd {
            return Err(CoreError::unsupported(
                "ForceAtlas2Solver supports Layout2d, Layout3d, and LayoutNd",
            ));
        }

        let (validate_fn, step_fn): (Fa2ValidateFn<L>, Fa2StepFn<L>) = if is_2d {
            (validate_basic::<L>, step_any_2d::<L>)
        } else if is_3d {
            (validate_basic::<L>, step_any_3d::<L>)
        } else {
            (validate_nd::<L>, step_any_nd::<L>)
        };

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
            let eff_w = if config.outbound_attraction_distribution {
                let mu = masses[edges_u[idx] as usize];
                let mv = masses[edges_v[idx] as usize];
                w_pow * mean_mass * 0.5 * (1.0 / mu + 1.0 / mv)
            } else {
                w_pow
            };
            edges_w.push(eff_w);
        }

        let len_2d3d = if is_2d || is_3d { n } else { 0 };
        let len_3d = if is_3d { n } else { 0 };
        let rep_len = if (is_2d || is_3d) && config.use_barnes_hut && !config.prevent_overlap {
            n
        } else {
            0
        };
        let rep_3d_len = if is_3d && config.use_barnes_hut && !config.prevent_overlap {
            n
        } else {
            0
        };
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
            rep_x: vec![0.0; rep_len],
            rep_y: vec![0.0; rep_len],
            rep_z: vec![0.0; rep_3d_len],
            quadtree: ArenaQuadtree2d::new(),
            octree: ArenaOctree3d::new(),
            overlap_tree: OverlapQuadtree::default(),
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

impl<L: EuclideanLayout> ForceAtlas2Solver<L> {
    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &ForceAtlas2Config {
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

    /// Returns `true` if Barnes-Hut spatial tree repulsion is enabled.
    #[inline]
    pub fn use_barnes_hut(&self) -> bool {
        self.state.config.use_barnes_hut
    }
}

impl<L: EuclideanLayout> IterativeSolver for ForceAtlas2Solver<L> {
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

fn validate_node_radii(n_graph: usize, node_radii: &[f64]) -> CoreResult<()> {
    if node_radii.len() != n_graph {
        return Err(CoreError::dimension_mismatch(
            n_graph,
            node_radii.len(),
            "ForceAtlas2Solver::with_node_radii::node_radii",
        ));
    }
    for &r in node_radii {
        if !r.is_finite() {
            return Err(CoreError::non_finite(
                "ForceAtlas2Solver::with_node_radii::node_radii",
            ));
        }
        if r < 0.0 {
            return Err(CoreError::invalid_input(
                "node_radii must be finite and >= 0.0",
            ));
        }
    }
    Ok(())
}

fn validate_basic<L: EuclideanLayout>(state: &ForceAtlas2State, target: &L) -> CoreResult<()> {
    if target.len() != state.num_nodes {
        return Err(CoreError::dimension_mismatch(
            state.num_nodes,
            target.len(),
            "ForceAtlas2Solver::validate_target",
        ));
    }
    target.validate_finite()
}

fn validate_nd<L: EuclideanLayout>(state: &ForceAtlas2State, target: &L) -> CoreResult<()> {
    if target.len() != state.num_nodes {
        return Err(CoreError::dimension_mismatch(
            state.num_nodes,
            target.len(),
            "ForceAtlas2Solver::validate_target",
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
            "ForceAtlas2Solver::validate_target::dims",
        ));
    }
    target.validate_finite()
}

fn step_any_2d<L: EuclideanLayout + 'static>(
    state: &mut ForceAtlas2State,
    target: &mut L,
) -> CoreResult<()> {
    let layout = (target as &mut dyn Any)
        .downcast_mut::<Layout2d<L::Scalar>>()
        .ok_or_else(|| CoreError::numerical_breakdown("ForceAtlas2Solver target type mismatch"))?;
    step_2d(state, layout)
}

fn step_any_3d<L: EuclideanLayout + 'static>(
    state: &mut ForceAtlas2State,
    target: &mut L,
) -> CoreResult<()> {
    let layout = (target as &mut dyn Any)
        .downcast_mut::<Layout3d<L::Scalar>>()
        .ok_or_else(|| CoreError::numerical_breakdown("ForceAtlas2Solver target type mismatch"))?;
    step_3d(state, layout)
}

fn step_any_nd<L: EuclideanLayout + 'static>(
    state: &mut ForceAtlas2State,
    target: &mut L,
) -> CoreResult<()> {
    let layout = (target as &mut dyn Any)
        .downcast_mut::<LayoutNd<L::Scalar>>()
        .ok_or_else(|| CoreError::numerical_breakdown("ForceAtlas2Solver target type mismatch"))?;
    step_nd(state, layout)
}
