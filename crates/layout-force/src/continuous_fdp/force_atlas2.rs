use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar, SolverStatus};
#[cfg(doc)]
use egraph_core::{IterativeSolver, Layout2d};
use egraph_spatial_trees::ArenaQuadtree2d;

use super::coarsening::extract_canonical_undirected_edges;
use super::config::ForceAtlas2Config;
use super::force_atlas2_nd::ForceAtlas2NdScratch;
use super::overlap_quadtree::OverlapQuadtree;

pub(super) const MAX_FA2_DISPLACEMENT: f64 = 50.0;

/// Cooperative ForceAtlas2 layout solver implementing [`IterativeSolver`].
///
/// Uses degree-weighted masses (`m_i = deg(i) + 1`) with either $O(N \log N)$ Barnes-Hut
/// [`ArenaQuadtree2d`] Coulomb repulsion or exact $O(N^2)$ repulsion, linear/LinLog attraction,
/// standard/strong gravity, optional Jacomy (2014) border-to-border `prevent_overlap`, and
/// adaptive swing-traction step sizing without per-step allocations.
#[derive(Debug, Clone)]
pub struct ForceAtlas2Solver<S: Scalar = f64> {
    pub(super) config: ForceAtlas2Config,
    pub(super) num_nodes: usize,
    pub(super) edges_u: Vec<u32>,
    pub(super) edges_v: Vec<u32>,
    pub(super) edges_w: Vec<f64>,
    pub(super) masses: Vec<f64>,
    pub(super) node_radii: Vec<f64>,
    pub(super) x_buf: Vec<f64>,
    pub(super) y_buf: Vec<f64>,
    pub(super) fx: Vec<f64>,
    pub(super) fy: Vec<f64>,
    pub(super) prev_fx: Vec<f64>,
    pub(super) prev_fy: Vec<f64>,
    pub(super) swing_buf: Vec<f64>,
    pub(super) rep_x: Vec<f64>,
    pub(super) rep_y: Vec<f64>,
    pub(super) quadtree: ArenaQuadtree2d<f64>,
    pub(super) overlap_tree: OverlapQuadtree,
    pub(super) global_speed: f64,
    pub(super) max_displacement: f64,
    pub(super) active_dims: usize,
    pub(super) nd_scratch: Option<ForceAtlas2NdScratch>,
    pub(super) iter: usize,
    pub(super) status: SolverStatus,
    _marker: std::marker::PhantomData<S>,
}

impl<S: Scalar> ForceAtlas2Solver<S> {
    /// Constructs a [`ForceAtlas2Solver`] from a [`CsrGraph`] using `config.default_node_radius`.
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: ForceAtlas2Config) -> CoreResult<Self> {
        config.validate()?;
        let n = graph.num_nodes();
        let radii = vec![config.default_node_radius; n];
        Self::with_node_radii(graph, config, radii)
    }

    /// Constructs a [`ForceAtlas2Solver`] from a [`CsrGraph`] with explicit per-node radii.
    pub fn with_node_radii<W: Scalar>(
        graph: &CsrGraph<W>,
        config: ForceAtlas2Config,
        node_radii: Vec<f64>,
    ) -> CoreResult<Self> {
        config.validate()?;
        let n_graph = graph.num_nodes();
        if node_radii.len() != n_graph {
            return Err(CoreError::dimension_mismatch(
                n_graph,
                node_radii.len(),
                "ForceAtlas2Solver::with_node_radii::node_radii",
            ));
        }
        for &r in &node_radii {
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

        let (n, edges_u, edges_v, raw_weights, deg) = extract_canonical_undirected_edges(graph)?;
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

        let rep_len = if config.use_barnes_hut && !config.prevent_overlap {
            n
        } else {
            0
        };
        let status = if config.iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };

        Ok(Self {
            config,
            num_nodes: n,
            edges_u,
            edges_v,
            edges_w,
            masses,
            node_radii,
            x_buf: vec![0.0; n],
            y_buf: vec![0.0; n],
            fx: vec![0.0; n],
            fy: vec![0.0; n],
            prev_fx: vec![0.0; n],
            prev_fy: vec![0.0; n],
            swing_buf: vec![0.0; n],
            rep_x: vec![0.0; rep_len],
            rep_y: vec![0.0; rep_len],
            quadtree: ArenaQuadtree2d::new(),
            overlap_tree: OverlapQuadtree::default(),
            global_speed: 1.0,
            max_displacement,
            active_dims: 0,
            nd_scratch: None,
            iter: 0,
            status,
            _marker: std::marker::PhantomData,
        })
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &ForceAtlas2Config {
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
        self.iter
    }

    /// Returns the scheduled maximum number of iterations.
    #[inline]
    pub fn total_iterations(&self) -> usize {
        self.config.iterations
    }

    /// Returns the per-node collision radii.
    #[inline]
    pub fn node_radii(&self) -> &[f64] {
        &self.node_radii
    }

    /// Returns the effective per-step vertex displacement cap.
    #[inline]
    pub fn max_displacement(&self) -> f64 {
        self.max_displacement
    }

    /// Returns `true` if ForceAtlas2 logarithmic attraction mode is enabled.
    #[inline]
    pub fn linlog_mode(&self) -> bool {
        self.config.linlog_mode
    }

    /// Returns `true` if strong gravity mode is enabled.
    #[inline]
    pub fn strong_gravity(&self) -> bool {
        self.config.strong_gravity
    }

    /// Returns `true` if Barnes-Hut quadtree repulsion is enabled.
    #[inline]
    pub fn use_barnes_hut(&self) -> bool {
        self.config.use_barnes_hut
    }
}
