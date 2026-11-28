use std::any::{Any, TypeId};
use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, EuclideanLayout, IterativeSolver, Layout2d,
    Layout3d, LayoutNd, Scalar, SolverStatus, VisitStamp,
};
use egraph_spatial_trees::{ArenaOctree3d, ArenaQuadtree2d};

use crate::coarsening::{build_coarse_csr, ec_coarsen, mivs_coarsen};
use crate::graph_edges::{
    extract_canonical_undirected_edges, extract_canonical_undirected_unweighted_edges,
};

use super::config::YifanHuConfig;
use super::dim3::step_3d;
use super::nd::{YifanHuNdScratch, step_nd};
use super::step2d::step_2d;

const MAX_COARSEN_LEVELS: usize = 64;
pub(super) const PROLONGATION_JITTER_FACTOR: f64 = 0.05;

pub(super) type YhValidateFn<L> = fn(&YifanHuState, &L) -> CoreResult<()>;
pub(super) type YhStepFn<L> = fn(&mut YifanHuState, &mut L) -> CoreResult<SolverStatus>;

#[derive(Debug, Clone)]
pub(super) struct MultilevelGraphLevel {
    pub(super) graph: CsrGraph<f64>,
    pub(super) optimal_distance: f64,
    pub(super) coarse_of_finer: Option<Vec<u32>>,
    pub(super) cluster_counts_of_finer: Option<Vec<u32>>,
}

#[derive(Debug, Clone)]
pub(super) struct YifanHuState {
    pub(super) config: YifanHuConfig,
    pub(super) num_nodes: usize,
    pub(super) has_explicit_weights: bool,
    pub(super) levels: Vec<MultilevelGraphLevel>,
    pub(super) fine_to_level: Vec<Vec<u32>>,
    pub(super) current_level: usize,
    pub(super) level_iter: usize,
    pub(super) total_iter_executed: usize,
    pub(super) max_total_iterations: usize,
    pub(super) initialized_coords: bool,
    pub(super) level_step_len: f64,
    pub(super) level_progress: usize,
    pub(super) prev_energy: f64,
    pub(super) active_x: Vec<f64>,
    pub(super) active_y: Vec<f64>,
    pub(super) active_z: Vec<f64>,
    pub(super) fx: Vec<f64>,
    pub(super) fy: Vec<f64>,
    pub(super) fz: Vec<f64>,
    pub(super) rep_x: Vec<f64>,
    pub(super) rep_y: Vec<f64>,
    pub(super) rep_z: Vec<f64>,
    pub(super) unit_masses: Vec<f64>,
    pub(super) quadtree: ArenaQuadtree2d<f64>,
    pub(super) octree: ArenaOctree3d<f64>,
    pub(super) locked_nd_dims: Option<usize>,
    pub(super) nd_scratch: Option<YifanHuNdScratch>,
    pub(super) rng: DeterministicRng,
    pub(super) status: SolverStatus,
}

/// Cooperative multilevel Yifan Hu (2005) layout solver implementing [`IterativeSolver`].
///
/// Bound at compile time to a Euclidean layout type `L`:
/// - [`Layout2d`]: uses [`ArenaQuadtree2d`] Barnes-Hut repulsion across levels.
/// - [`Layout3d`]: uses [`ArenaOctree3d`] Barnes-Hut repulsion across levels.
/// - [`LayoutNd`]: uses exact $O(N^2)$ Coulomb repulsion across arbitrary dimensions.
#[derive(Debug, Clone)]
pub struct YifanHuSolver<L: EuclideanLayout = Layout2d<f64>> {
    state: YifanHuState,
    validate_fn: YhValidateFn<L>,
    step_fn: YhStepFn<L>,
    _layout: PhantomData<fn() -> L>,
}

impl<L: EuclideanLayout + 'static> YifanHuSolver<L> {
    /// Constructs a [`YifanHuSolver`] and builds its `EC + MIVS` coarsening hierarchy.
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: YifanHuConfig) -> CoreResult<Self> {
        config.validate()?;
        let has_explicit_weights = graph.is_weighted();
        let (n, edges_u, edges_v, edge_weights, _) =
            extract_canonical_undirected_edges(graph, "YifanHuSolver::graph_weights")?;
        Self::from_canonical_edges(
            n,
            edges_u,
            edges_v,
            edge_weights,
            has_explicit_weights,
            config,
        )
    }

    /// Constructs a [`YifanHuSolver`] from an unweighted [`CsrGraph<W>`] (or ignoring stored
    /// edge weights) without requiring `W: Scalar`.
    pub fn new_unweighted<W>(graph: &CsrGraph<W>, config: YifanHuConfig) -> CoreResult<Self> {
        config.validate()?;
        let (n, edges_u, edges_v, edge_weights, _) =
            extract_canonical_undirected_unweighted_edges(graph)?;
        Self::from_canonical_edges(n, edges_u, edges_v, edge_weights, false, config)
    }

    fn from_canonical_edges(
        n: usize,
        edges_u: Vec<u32>,
        edges_v: Vec<u32>,
        edge_weights: Vec<f64>,
        has_explicit_weights: bool,
        config: YifanHuConfig,
    ) -> CoreResult<Self> {
        let tid = TypeId::of::<L>();
        let is_2d = tid == TypeId::of::<Layout2d<L::Scalar>>();
        let is_3d = tid == TypeId::of::<Layout3d<L::Scalar>>();
        let is_nd = tid == TypeId::of::<LayoutNd<L::Scalar>>();

        if !is_2d && !is_3d && !is_nd {
            return Err(CoreError::unsupported(
                "YifanHuSolver supports Layout2d, Layout3d, and LayoutNd",
            ));
        }

        let (validate_fn, step_fn): (YhValidateFn<L>, YhStepFn<L>) = if is_2d {
            (validate_basic::<L>, step_any_2d::<L>)
        } else if is_3d {
            (validate_basic::<L>, step_any_3d::<L>)
        } else {
            (validate_nd::<L>, step_any_nd::<L>)
        };

        let edge_pairs: Vec<(u32, u32)> = edges_u.into_iter().zip(edges_v).collect();
        let level0_graph = CsrGraph::from_edges(n, &edge_pairs, Some(&edge_weights), false, false)?;

        let k0 = config.optimal_distance.unwrap_or(1.0);
        let mut levels = vec![MultilevelGraphLevel {
            graph: level0_graph,
            optimal_distance: k0,
            coarse_of_finer: None,
            cluster_counts_of_finer: None,
        }];

        let mut coarsen_step = 0usize;
        let mut stamp = VisitStamp::new(n);
        while levels.len() < MAX_COARSEN_LEVELS {
            let cur_len = levels.len();
            let cur_graph = &levels[cur_len - 1].graph;
            let cur_n = cur_graph.num_nodes();
            if cur_n <= config.min_coarsen_size || cur_graph.num_edges() == 0 {
                break;
            }

            let primary_is_ec = coarsen_step.is_multiple_of(2);
            let mut candidate = if primary_is_ec {
                ec_coarsen(cur_graph)
            } else {
                mivs_coarsen(cur_graph)
            };

            let mut ratio_ok = (candidate.coarse_n as f64)
                < config.min_coarsen_ratio * (cur_n as f64)
                && candidate.coarse_n < cur_n;
            if !ratio_ok {
                let alt = if primary_is_ec {
                    mivs_coarsen(cur_graph)
                } else {
                    ec_coarsen(cur_graph)
                };
                if alt.coarse_n < cur_n
                    && (alt.coarse_n as f64) < config.min_coarsen_ratio * (cur_n as f64)
                {
                    candidate = alt;
                    ratio_ok = true;
                }
            }
            if !ratio_ok || candidate.coarse_n == 0 {
                break;
            }

            let coarse_graph = build_coarse_csr(
                cur_graph,
                candidate.coarse_n,
                &candidate.coarse_of,
                &candidate.cluster_counts,
                &mut stamp,
            )?;
            let k_next = k0 * ((n as f64) / (candidate.coarse_n as f64)).sqrt();

            levels[cur_len - 1].coarse_of_finer = Some(candidate.coarse_of);
            levels[cur_len - 1].cluster_counts_of_finer = Some(candidate.cluster_counts);
            levels.push(MultilevelGraphLevel {
                graph: coarse_graph,
                optimal_distance: k_next,
                coarse_of_finer: None,
                cluster_counts_of_finer: None,
            });
            coarsen_step += 1;
        }

        let num_levels = levels.len();
        let mut fine_to_level = Vec::with_capacity(num_levels);
        fine_to_level.push(Vec::new());
        for l in 1..num_levels {
            let mut map = Vec::with_capacity(n);
            let step_map = levels[l - 1]
                .coarse_of_finer
                .as_ref()
                .ok_or_else(|| CoreError::invalid_csr("missing coarse_of_finer mapping"))?;
            if l == 1 {
                map.extend_from_slice(step_map);
            } else {
                let prev = &fine_to_level[l - 1];
                for &idx in prev {
                    map.push(step_map[idx as usize]);
                }
            }
            fine_to_level.push(map);
        }

        let coarsest_level = num_levels - 1;
        let max_total_iterations = num_levels
            .checked_mul(config.max_iterations_per_level)
            .ok_or_else(|| CoreError::invalid_input("YifanHu total_iterations overflow"))?;
        let status = if max_total_iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };
        let init_step_len = config.step_size * levels[coarsest_level].optimal_distance;
        let len_2d3d = if is_2d || is_3d { n } else { 0 };
        let len_3d = if is_3d { n } else { 0 };
        let seed = config.seed;

        let state = YifanHuState {
            config,
            num_nodes: n,
            has_explicit_weights,
            levels,
            fine_to_level,
            current_level: coarsest_level,
            level_iter: 0,
            total_iter_executed: 0,
            max_total_iterations,
            initialized_coords: false,
            level_step_len: init_step_len,
            level_progress: 0,
            prev_energy: f64::INFINITY,
            active_x: vec![0.0; len_2d3d],
            active_y: vec![0.0; len_2d3d],
            active_z: vec![0.0; len_3d],
            fx: vec![0.0; len_2d3d],
            fy: vec![0.0; len_2d3d],
            fz: vec![0.0; len_3d],
            rep_x: vec![0.0; len_2d3d],
            rep_y: vec![0.0; len_2d3d],
            rep_z: vec![0.0; len_3d],
            unit_masses: vec![1.0; if is_2d || is_3d { n } else { 0 }],
            quadtree: ArenaQuadtree2d::new(),
            octree: ArenaOctree3d::new(),
            locked_nd_dims: None,
            nd_scratch: None,
            rng: DeterministicRng::from_seed_u64(seed),
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

impl<L: EuclideanLayout> YifanHuSolver<L> {
    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &YifanHuConfig {
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
        self.state.total_iter_executed
    }

    /// Returns the scheduled maximum number of iterations across all levels.
    #[inline]
    pub fn total_iterations(&self) -> Option<usize> {
        Some(self.state.max_total_iterations)
    }

    /// Returns the number of levels in the `EC + MIVS` hierarchy (`>= 1`).
    #[inline]
    pub fn num_levels(&self) -> usize {
        self.state.levels.len()
    }

    /// Returns the currently active level index (`num_levels - 1` coarsest down to `0` finest).
    #[inline]
    pub fn current_level(&self) -> usize {
        self.state.current_level
    }

    /// Returns the finest-level optimal distance parameter $K_0$.
    #[inline]
    pub fn optimal_distance(&self) -> f64 {
        self.state.levels[0].optimal_distance
    }
}

impl<L: EuclideanLayout> IterativeSolver for YifanHuSolver<L> {
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

        match (self.step_fn)(&mut self.state, target) {
            Ok(status) => Ok(status),
            Err(err) => {
                if matches!(
                    err,
                    CoreError::NumericalBreakdown(_) | CoreError::NonFiniteValue { .. }
                ) {
                    self.state.status = SolverStatus::Failed;
                }
                Err(err)
            }
        }
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.state.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.state.total_iter_executed
    }

    #[inline]
    fn total_iterations(&self) -> Option<usize> {
        Some(self.state.max_total_iterations)
    }
}

fn validate_basic<L: EuclideanLayout>(state: &YifanHuState, target: &L) -> CoreResult<()> {
    if target.len() != state.num_nodes {
        return Err(CoreError::dimension_mismatch(
            state.num_nodes,
            target.len(),
            "YifanHuSolver::validate_target",
        ));
    }
    target.validate_finite()
}

fn validate_nd<L: EuclideanLayout>(state: &YifanHuState, target: &L) -> CoreResult<()> {
    if target.len() != state.num_nodes {
        return Err(CoreError::dimension_mismatch(
            state.num_nodes,
            target.len(),
            "YifanHuSolver::validate_target",
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
            "YifanHuSolver::validate_target::dims",
        ));
    }
    target.validate_finite()
}

fn step_any_2d<L: EuclideanLayout + 'static>(
    state: &mut YifanHuState,
    target: &mut L,
) -> CoreResult<SolverStatus> {
    let layout = (target as &mut dyn Any)
        .downcast_mut::<Layout2d<L::Scalar>>()
        .ok_or_else(|| CoreError::numerical_breakdown("YifanHuSolver target type mismatch"))?;
    step_2d(state, layout)
}

fn step_any_3d<L: EuclideanLayout + 'static>(
    state: &mut YifanHuState,
    target: &mut L,
) -> CoreResult<SolverStatus> {
    let layout = (target as &mut dyn Any)
        .downcast_mut::<Layout3d<L::Scalar>>()
        .ok_or_else(|| CoreError::numerical_breakdown("YifanHuSolver target type mismatch"))?;
    step_3d(state, layout)
}

fn step_any_nd<L: EuclideanLayout + 'static>(
    state: &mut YifanHuState,
    target: &mut L,
) -> CoreResult<SolverStatus> {
    let layout = (target as &mut dyn Any)
        .downcast_mut::<LayoutNd<L::Scalar>>()
        .ok_or_else(|| CoreError::numerical_breakdown("YifanHuSolver target type mismatch"))?;
    step_nd(state, layout)
}
