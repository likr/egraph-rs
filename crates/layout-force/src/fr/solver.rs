use std::any::{Any, TypeId};
use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Layout3d,
    LayoutNd, ManifoldKind, ManifoldLayout2d, RiemannianLayout, Scalar, SolverStatus,
};
use egraph_spatial_trees::{ArenaOctree3d, ArenaQuadtree2d};

use crate::graph_edges::{
    extract_canonical_undirected_edges, extract_canonical_undirected_unweighted_edges,
};

use super::config::{FrBackend, FruchtermanReingoldConfig};
use super::dim3::step_3d;
use super::helpers::{Grid3Cell, Grid3Entry, GridCell, GridEntry};
use super::manifold::step_manifold;
use super::nd::{FrNdScratch, step_nd};
use super::riemannian::{FrRiemannianScratch, step_riemannian};
use super::step2d::step_2d;

pub(super) const COINCIDENT_DIST_SQ: f64 = 1e-24;
pub(super) const COINCIDENT_PERTURB_DIST: f64 = 1e-6;
pub(super) const BH_EPS_SQ: f64 = 1e-12;

pub(super) type FrValidateFn<L> = fn(&FrState, &L) -> CoreResult<()>;
pub(super) type FrStepFn<L> = fn(&mut FrState, &mut L) -> CoreResult<()>;

#[derive(Debug, Clone)]
pub(super) struct FrState {
    pub(super) config: FruchtermanReingoldConfig,
    pub(super) num_nodes: usize,
    pub(super) edges_u: Vec<u32>,
    pub(super) edges_v: Vec<u32>,
    pub(super) edges_w: Vec<f64>,
    pub(super) k: f64,
    pub(super) initial_temperature: f64,
    pub(super) x_buf: Vec<f64>,
    pub(super) y_buf: Vec<f64>,
    pub(super) z_buf: Vec<f64>,
    pub(super) disp_x: Vec<f64>,
    pub(super) disp_y: Vec<f64>,
    pub(super) disp_z: Vec<f64>,
    pub(super) rep_x: Vec<f64>,
    pub(super) rep_y: Vec<f64>,
    pub(super) rep_z: Vec<f64>,
    pub(super) quadtree: ArenaQuadtree2d<f64>,
    pub(super) octree: ArenaOctree3d<f64>,
    pub(super) grid_entries: Vec<GridEntry>,
    pub(super) grid_cells: Vec<GridCell>,
    pub(super) grid3_entries: Vec<Grid3Entry>,
    pub(super) grid3_cells: Vec<Grid3Cell>,
    pub(super) nd_scratch: Option<FrNdScratch>,
    pub(super) riemannian_scratch: Option<FrRiemannianScratch>,
    pub(super) locked_nd_dims: Option<usize>,
    pub(super) locked_manifold: Option<ManifoldKind>,
    pub(super) rng: DeterministicRng,
    pub(super) iter: usize,
    pub(super) status: SolverStatus,
}

impl FrState {
    #[inline]
    pub(super) fn current_temperature(&self) -> f64 {
        if self.config.iterations == 0 || self.iter >= self.config.iterations {
            return 0.0;
        }
        let rem = (self.config.iterations - self.iter) as f64;
        self.initial_temperature * (rem / (self.config.iterations as f64))
    }

    #[inline]
    pub(super) fn next_unit_vector(&mut self) -> (f64, f64) {
        let angle = self.rng.next_f64() * std::f64::consts::TAU;
        (angle.cos(), angle.sin())
    }
}

/// Cooperative Fruchterman–Reingold (1991) layout solver implementing [`IterativeSolver`].
///
/// Supports:
/// - [`Layout2d`]: `Exact`, `BarnesHut` (`ArenaQuadtree2d`), and `Grid`.
/// - [`Layout3d`]: `Exact`, `BarnesHut` (`ArenaOctree3d`), and `Grid`.
/// - [`LayoutNd`], [`ManifoldLayout2d`], and any [`RiemannianLayout`]: `Exact` only
///   (`BarnesHut` and `Grid` fail fast at construction).
#[derive(Debug, Clone)]
pub struct FruchtermanReingoldSolver<L: RiemannianLayout = Layout2d<f64>> {
    state: FrState,
    validate_fn: FrValidateFn<L>,
    step_fn: FrStepFn<L>,
    _layout: PhantomData<fn() -> L>,
}

impl<L: RiemannianLayout + 'static> FruchtermanReingoldSolver<L> {
    /// Constructs a [`FruchtermanReingoldSolver`] from a [`CsrGraph`] and [`FruchtermanReingoldConfig`].
    pub fn new<W: Scalar>(
        graph: &CsrGraph<W>,
        config: FruchtermanReingoldConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let (n, edges_u, edges_v, edges_w, _) =
            extract_canonical_undirected_edges(graph, "FruchtermanReingoldSolver::graph_weights")?;
        Self::from_canonical_edges(n, edges_u, edges_v, edges_w, config)
    }

    /// Constructs a [`FruchtermanReingoldSolver`] from an unweighted [`CsrGraph<W>`] (or ignoring
    /// stored edge weights) without requiring `W: Scalar`.
    pub fn new_unweighted<W>(
        graph: &CsrGraph<W>,
        config: FruchtermanReingoldConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let (n, edges_u, edges_v, edges_w, _) =
            extract_canonical_undirected_unweighted_edges(graph)?;
        Self::from_canonical_edges(n, edges_u, edges_v, edges_w, config)
    }

    fn from_canonical_edges(
        n: usize,
        edges_u: Vec<u32>,
        edges_v: Vec<u32>,
        edges_w: Vec<f64>,
        config: FruchtermanReingoldConfig,
    ) -> CoreResult<Self> {
        let tid = TypeId::of::<L>();
        let is_2d = tid == TypeId::of::<Layout2d<L::Scalar>>();
        let is_3d = tid == TypeId::of::<Layout3d<L::Scalar>>();
        let is_nd = tid == TypeId::of::<LayoutNd<L::Scalar>>();
        let is_manifold = tid == TypeId::of::<ManifoldLayout2d<L::Scalar>>();

        if is_nd && config.backend != FrBackend::Exact {
            return Err(CoreError::unsupported(format!(
                "Fruchterman-Reingold {:?} backend is not supported on LayoutNd; use Layout3d for 3D Barnes-Hut/Grid, Layout2d for 2D, or FrBackend::Exact for arbitrary N-D",
                config.backend
            )));
        }
        if is_manifold && config.backend != FrBackend::Exact {
            return Err(CoreError::unsupported(format!(
                "Fruchterman-Reingold {:?} backend is only supported on Euclidean layouts; use FrBackend::Exact on ManifoldLayout2d",
                config.backend
            )));
        }
        if !is_2d && !is_3d && !is_nd && !is_manifold && config.backend != FrBackend::Exact {
            return Err(CoreError::unsupported(format!(
                "Fruchterman-Reingold {:?} backend is only supported on Layout2d and Layout3d; use FrBackend::Exact on general RiemannianLayout targets",
                config.backend
            )));
        }

        let (validate_fn, step_fn): (FrValidateFn<L>, FrStepFn<L>) = if is_2d {
            (validate_basic::<L>, step_any_2d::<L>)
        } else if is_3d {
            (validate_basic::<L>, step_any_3d::<L>)
        } else if is_nd {
            (validate_nd::<L>, step_any_nd::<L>)
        } else if is_manifold {
            (validate_manifold::<L>, step_any_manifold::<L>)
        } else {
            (validate_riemannian::<L>, step_riemannian::<L>)
        };

        let n_eff = n.max(1) as f64;
        let effective_area = config.area.unwrap_or(if let Some(kv) = config.k {
            kv * kv * n_eff
        } else {
            n_eff
        });
        let k = config.k.unwrap_or_else(|| (effective_area / n_eff).sqrt());
        let initial_temperature = config
            .initial_temperature
            .unwrap_or_else(|| 0.1 * effective_area.sqrt());

        let len_2d3d = if is_2d || is_3d || is_manifold { n } else { 0 };
        let len_2d3d_pos = if is_2d || is_3d { n } else { 0 };
        let len_3d = if is_3d { n } else { 0 };
        let rep_2d3d = if (is_2d || is_3d) && config.backend == FrBackend::BarnesHut {
            n
        } else {
            0
        };
        let rep_3d = if is_3d && config.backend == FrBackend::BarnesHut {
            n
        } else {
            0
        };
        let status = if config.iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };
        let seed = config.seed;

        let state = FrState {
            config,
            num_nodes: n,
            edges_u,
            edges_v,
            edges_w,
            k,
            initial_temperature,
            x_buf: vec![0.0; len_2d3d_pos],
            y_buf: vec![0.0; len_2d3d_pos],
            z_buf: vec![0.0; len_3d],
            disp_x: vec![0.0; len_2d3d],
            disp_y: vec![0.0; len_2d3d],
            disp_z: vec![0.0; len_3d],
            rep_x: vec![0.0; rep_2d3d],
            rep_y: vec![0.0; rep_2d3d],
            rep_z: vec![0.0; rep_3d],
            quadtree: ArenaQuadtree2d::new(),
            octree: ArenaOctree3d::new(),
            grid_entries: Vec::new(),
            grid_cells: Vec::new(),
            grid3_entries: Vec::new(),
            grid3_cells: Vec::new(),
            nd_scratch: None,
            riemannian_scratch: None,
            locked_nd_dims: None,
            locked_manifold: None,
            rng: DeterministicRng::from_seed_u64(seed),
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

impl<L: RiemannianLayout> FruchtermanReingoldSolver<L> {
    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &FruchtermanReingoldConfig {
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

    /// Returns the effective optimal pairwise distance $k$.
    #[inline]
    pub fn optimal_distance(&self) -> f64 {
        self.state.k
    }

    /// Returns the temperature used for the upcoming (or final) iteration.
    #[inline]
    pub fn current_temperature(&self) -> f64 {
        self.state.current_temperature()
    }
}

impl<L: RiemannianLayout> IterativeSolver for FruchtermanReingoldSolver<L> {
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

fn validate_basic<L: RiemannianLayout>(state: &FrState, target: &L) -> CoreResult<()> {
    if target.len() != state.num_nodes {
        return Err(CoreError::dimension_mismatch(
            state.num_nodes,
            target.len(),
            "FruchtermanReingoldSolver::validate_target",
        ));
    }
    target.validate_finite()
}

fn validate_nd<L: RiemannianLayout + 'static>(state: &FrState, target: &L) -> CoreResult<()> {
    if target.len() != state.num_nodes {
        return Err(CoreError::dimension_mismatch(
            state.num_nodes,
            target.len(),
            "FruchtermanReingoldSolver::validate_target",
        ));
    }
    let layout = (target as &dyn Any)
        .downcast_ref::<LayoutNd<L::Scalar>>()
        .ok_or_else(|| {
            CoreError::numerical_breakdown("FruchtermanReingoldSolver target type mismatch")
        })?;
    let dims = layout.dims();
    if dims == 0 {
        return Err(CoreError::invalid_input("LayoutNd dims must be >= 1"));
    }
    if let Some(prev) = state.locked_nd_dims
        && prev != dims
    {
        return Err(CoreError::dimension_mismatch(
            prev,
            dims,
            "FruchtermanReingoldSolver::validate_target::dims",
        ));
    }
    layout.validate_finite()
}

fn validate_manifold<L: RiemannianLayout + 'static>(state: &FrState, target: &L) -> CoreResult<()> {
    if target.len() != state.num_nodes {
        return Err(CoreError::dimension_mismatch(
            state.num_nodes,
            target.len(),
            "FruchtermanReingoldSolver::validate_target",
        ));
    }
    let layout = (target as &dyn Any)
        .downcast_ref::<ManifoldLayout2d<L::Scalar>>()
        .ok_or_else(|| {
            CoreError::numerical_breakdown("FruchtermanReingoldSolver target type mismatch")
        })?;
    if let Some(prev) = state.locked_manifold
        && prev != layout.kind()
    {
        return Err(CoreError::invalid_input(
            "FruchtermanReingoldSolver::validate_target: cannot switch ManifoldKind on stateful solver",
        ));
    }
    layout.validate_finite()
}

fn validate_riemannian<L: RiemannianLayout>(state: &FrState, target: &L) -> CoreResult<()> {
    if target.len() != state.num_nodes {
        return Err(CoreError::dimension_mismatch(
            state.num_nodes,
            target.len(),
            "FruchtermanReingoldSolver::validate_target",
        ));
    }
    if target.tangent_dim() == 0 && state.num_nodes > 0 {
        return Err(CoreError::invalid_input("tangent_dim must be >= 1"));
    }
    target.validate_finite()
}

fn step_any_2d<L: RiemannianLayout + 'static>(
    state: &mut FrState,
    target: &mut L,
) -> CoreResult<()> {
    let layout = (target as &mut dyn Any)
        .downcast_mut::<Layout2d<L::Scalar>>()
        .ok_or_else(|| {
            CoreError::numerical_breakdown("FruchtermanReingoldSolver target type mismatch")
        })?;
    step_2d(state, layout)
}

fn step_any_3d<L: RiemannianLayout + 'static>(
    state: &mut FrState,
    target: &mut L,
) -> CoreResult<()> {
    let layout = (target as &mut dyn Any)
        .downcast_mut::<Layout3d<L::Scalar>>()
        .ok_or_else(|| {
            CoreError::numerical_breakdown("FruchtermanReingoldSolver target type mismatch")
        })?;
    step_3d(state, layout)
}

fn step_any_nd<L: RiemannianLayout + 'static>(
    state: &mut FrState,
    target: &mut L,
) -> CoreResult<()> {
    let layout = (target as &mut dyn Any)
        .downcast_mut::<LayoutNd<L::Scalar>>()
        .ok_or_else(|| {
            CoreError::numerical_breakdown("FruchtermanReingoldSolver target type mismatch")
        })?;
    step_nd(state, layout)
}

fn step_any_manifold<L: RiemannianLayout + 'static>(
    state: &mut FrState,
    target: &mut L,
) -> CoreResult<()> {
    let layout = (target as &mut dyn Any)
        .downcast_mut::<ManifoldLayout2d<L::Scalar>>()
        .ok_or_else(|| {
            CoreError::numerical_breakdown("FruchtermanReingoldSolver target type mismatch")
        })?;
    step_manifold(state, layout)
}
