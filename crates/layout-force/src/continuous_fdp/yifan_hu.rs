use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, Scalar, SolverStatus, VisitStamp,
};
#[cfg(doc)]
use egraph_core::{IterativeSolver, Layout2d};
use egraph_spatial_trees::ArenaQuadtree2d;

use super::coarsening::{
    build_coarse_csr, ec_coarsen, extract_canonical_undirected_edges, mivs_coarsen,
};
use super::config::YifanHuConfig;
use super::yifan_hu_nd::YifanHuNdScratch;

const MAX_COARSEN_LEVELS: usize = 64;

#[derive(Debug, Clone)]
pub(super) struct MultilevelGraphLevel {
    pub(super) graph: CsrGraph<f64>,
    pub(super) optimal_distance: f64,
    /// Maps each vertex at this level to its coarse parent in level `l + 1`.
    pub(super) coarse_of_finer: Option<Vec<u32>>,
    /// Number of vertices at this level collapsed into each coarse vertex in level `l + 1`.
    pub(super) cluster_counts_of_finer: Option<Vec<u32>>,
}

/// Cooperative multilevel Yifan Hu (2005) layout solver implementing [`IterativeSolver`].
///
/// Constructs an alternating Edge Collapsing (EC) + Maximal Independent Vertex Set (MIVS)
/// coarsening hierarchy directly on [`CsrGraph`], evaluates forces using [`ArenaQuadtree2d`]
/// Coulomb repulsion (`C K^2 / d`) and quadratic spring attraction (`d^2 / K`) with adaptive
/// step length control, and prolongates coarse layouts down to the finest level.
#[derive(Debug, Clone)]
pub struct YifanHuSolver<S: Scalar = f64> {
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
    pub(super) fx: Vec<f64>,
    pub(super) fy: Vec<f64>,
    pub(super) rep_x: Vec<f64>,
    pub(super) rep_y: Vec<f64>,
    pub(super) unit_masses: Vec<f64>,
    pub(super) quadtree: ArenaQuadtree2d<f64>,
    pub(super) active_dims: usize,
    pub(super) nd_scratch: Option<YifanHuNdScratch>,
    pub(super) rng: DeterministicRng,
    pub(super) status: SolverStatus,
    _marker: std::marker::PhantomData<S>,
}

impl<S: Scalar> YifanHuSolver<S> {
    /// Constructs a [`YifanHuSolver`] and builds its `EC + MIVS` coarsening hierarchy.
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: YifanHuConfig) -> CoreResult<Self> {
        config.validate()?;
        let has_explicit_weights = graph.is_weighted();
        let (n, edges_u, edges_v, edge_weights, _) = extract_canonical_undirected_edges(graph)?;
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
                // Fall back once to the complementary coarsening strategy before terminating.
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

        Ok(Self {
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
            active_x: vec![0.0; n],
            active_y: vec![0.0; n],
            fx: vec![0.0; n],
            fy: vec![0.0; n],
            rep_x: vec![0.0; n],
            rep_y: vec![0.0; n],
            unit_masses: vec![1.0; n],
            quadtree: ArenaQuadtree2d::new(),
            active_dims: 0,
            nd_scratch: None,
            rng: DeterministicRng::from_seed_u64(config.seed),
            status,
            _marker: std::marker::PhantomData,
        })
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &YifanHuConfig {
        &self.config
    }

    /// Returns the expected number of vertices in [`Layout2d`] or [`egraph_core::LayoutNd`].
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the current lifecycle status.
    #[inline]
    pub fn status(&self) -> SolverStatus {
        self.status
    }

    /// Returns the number of iterations executed so far.
    #[inline]
    pub fn iteration(&self) -> usize {
        self.total_iter_executed
    }

    /// Returns the scheduled maximum number of iterations.
    #[inline]
    pub fn total_iterations(&self) -> usize {
        self.max_total_iterations
    }

    /// Returns the number of levels in the `EC + MIVS` hierarchy (`>= 1`).
    #[inline]
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Returns the currently active level index (`num_levels - 1` coarsest down to `0` finest).
    #[inline]
    pub fn current_level(&self) -> usize {
        self.current_level
    }

    /// Returns the finest-level optimal distance parameter $K_0$.
    #[inline]
    pub fn optimal_distance(&self) -> f64 {
        self.levels[0].optimal_distance
    }
}
