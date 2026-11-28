//! Cooperative Screened Poisson unilateral upper-bound density field refiner.
//!
//! Provides [`ScreenedPoissonRefiner`], implementing [`IterativeSolver`] and
//! [`ScreenedPoissonRefiner::apply_epoch_step`] for epoch-boundary Eulerian repulsion,
//! optional 1-hop edge-length anchor springs, and rigid cluster centroid tethering
//! (`align_clustered_centroids`).

mod config;
#[cfg(test)]
mod tests;

pub use config::ScreenedPoissonRefinerConfig;

use egraph_clustering::align_clustered_centroids;
use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Scalar,
    SolverStatus,
};
use egraph_spatial_fft::{
    DensityFieldKernel, ScreenedPoissonField2d, ScreenedPoissonOutput, ScreenedPoissonStats,
};

/// Cooperative Screened Poisson unilateral upper-bound density field refiner implementing
/// [`IterativeSolver`].
#[derive(Debug, Clone)]
pub struct ScreenedPoissonRefiner<S: Scalar = f64> {
    config: ScreenedPoissonRefinerConfig,
    num_nodes: usize,
    field: ScreenedPoissonField2d<S>,
    field_out: ScreenedPoissonOutput<S>,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    anchor_targets: Vec<f64>,
    anchor_targets_initialized: bool,
    coarse_tether: Option<(Layout2d<S>, Vec<u32>)>,
    vel_x: Vec<f64>,
    vel_y: Vec<f64>,
    disp_x: Vec<f64>,
    disp_y: Vec<f64>,
    last_stats: ScreenedPoissonStats,
    iter: usize,
    status: SolverStatus,
}

impl<S: Scalar> ScreenedPoissonRefiner<S> {
    /// Creates a standalone [`ScreenedPoissonRefiner`] for `num_nodes` vertices without graph anchors.
    pub fn new(num_nodes: usize, config: ScreenedPoissonRefinerConfig) -> CoreResult<Self> {
        config.validate()?;
        let field = ScreenedPoissonField2d::<S>::new(config.field.clone())?;
        let status = if config.iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };
        Ok(Self {
            config,
            num_nodes,
            field,
            field_out: ScreenedPoissonOutput::new(),
            edges_u: Vec::new(),
            edges_v: Vec::new(),
            anchor_targets: Vec::new(),
            anchor_targets_initialized: false,
            coarse_tether: None,
            vel_x: vec![0.0; num_nodes],
            vel_y: vec![0.0; num_nodes],
            disp_x: vec![0.0; num_nodes],
            disp_y: vec![0.0; num_nodes],
            last_stats: ScreenedPoissonStats::default(),
            iter: 0,
            status,
        })
    }

    /// Creates a [`ScreenedPoissonRefiner`] storing canonical undirected 1-hop edges from `graph`
    /// for optional anchor-spring preservation (`anchor_weight > 0`).
    pub fn with_graph<W>(
        graph: &CsrGraph<W>,
        config: ScreenedPoissonRefinerConfig,
    ) -> CoreResult<Self> {
        let mut refiner = Self::new(graph.num_nodes(), config)?;
        let n = graph.num_nodes();
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        for u in 0..n {
            let u_u32 = u as u32;
            for &v_u32 in graph.neighbors(u) {
                if u_u32 != v_u32 {
                    pairs.push(if u_u32 < v_u32 {
                        (u_u32, v_u32)
                    } else {
                        (v_u32, u_u32)
                    });
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        refiner.edges_u.reserve(pairs.len());
        refiner.edges_v.reserve(pairs.len());
        for (u, v) in pairs {
            refiner.edges_u.push(u);
            refiner.edges_v.push(v);
        }
        Ok(refiner)
    }

    /// Attaches a coarse-level cluster centroid tether (`align_clustered_centroids`) in builder style.
    pub fn with_coarse_tether(
        mut self,
        coarse_layout: &Layout2d<S>,
        node_to_coarse: &[u32],
    ) -> CoreResult<Self> {
        self.set_coarse_tether(coarse_layout, node_to_coarse)?;
        Ok(self)
    }

    /// Attaches or replaces the coarse-level cluster centroid tether (`align_clustered_centroids`).
    pub fn set_coarse_tether(
        &mut self,
        coarse_layout: &Layout2d<S>,
        node_to_coarse: &[u32],
    ) -> CoreResult<()> {
        coarse_layout.validate_finite()?;
        if node_to_coarse.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                node_to_coarse.len(),
                "ScreenedPoissonRefiner::set_coarse_tether",
            ));
        }
        let num_clusters = coarse_layout.len();
        for &cid in node_to_coarse {
            let idx = cid as usize;
            if idx >= num_clusters {
                return Err(CoreError::index_out_of_bounds(
                    idx,
                    num_clusters,
                    "ScreenedPoissonRefiner::set_coarse_tether",
                ));
            }
        }
        self.coarse_tether = Some((coarse_layout.clone(), node_to_coarse.to_vec()));
        Ok(())
    }

    /// Executes a single decoupled Eulerian density step in place on `layout` and returns telemetry.
    pub fn apply_epoch_step(
        &mut self,
        layout: &mut Layout2d<S>,
    ) -> CoreResult<ScreenedPoissonStats> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "ScreenedPoissonRefiner::apply_epoch_step",
            ));
        }
        layout.validate_finite()?;
        let n = self.num_nodes;
        if n <= 1 {
            self.field.evaluate_euclidean(layout, &mut self.field_out)?;
            self.last_stats = self.field_out.stats.clone();
            return Ok(self.last_stats.clone());
        }

        // Break exact coincident degeneracy deterministically using the explicit u64 seed.
        let (mut min_x, mut max_x, mut min_y, mut max_y) = bbox_of(layout);
        if (max_x - min_x) < 1e-9 && (max_y - min_y) < 1e-9 {
            let step_seed = self
                .config
                .seed
                .wrapping_add((self.iter as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut rng = DeterministicRng::from_seed_u64(step_seed);
            let (xs, ys) = layout.xy_mut();
            for i in 0..n {
                let dx = (rng.next_f64() - 0.5) * 1e-4;
                let dy = (rng.next_f64() - 0.5) * 1e-4;
                xs[i] = S::from_f64(xs[i].to_f64() + dx);
                ys[i] = S::from_f64(ys[i].to_f64() + dy);
            }
            (min_x, max_x, min_y, max_y) = bbox_of(layout);
        }

        let w_span = (max_x - min_x).max(max_y - min_y).max(1e-4);

        if !self.anchor_targets_initialized
            && self.config.anchor_weight > 0.0
            && !self.edges_u.is_empty()
        {
            self.anchor_targets.clear();
            self.anchor_targets.reserve(self.edges_u.len());
            let d_floor = 1e-6 * w_span;
            let xs = layout.x();
            let ys = layout.y();
            for (&u, &v) in self.edges_u.iter().zip(self.edges_v.iter()) {
                let ui = u as usize;
                let vi = v as usize;
                let d = (xs[vi].to_f64() - xs[ui].to_f64())
                    .hypot(ys[vi].to_f64() - ys[ui].to_f64())
                    .max(d_floor);
                self.anchor_targets.push(d);
            }
            self.anchor_targets_initialized = true;
        }

        self.field.evaluate_euclidean(layout, &mut self.field_out)?;
        let stats = self.field_out.stats.clone();
        self.last_stats = stats.clone();

        let has_anchors = self.config.anchor_weight > 0.0 && !self.edges_u.is_empty();
        let has_tether = self.config.gamma_tether > 0.0 && self.coarse_tether.is_some();
        if stats.early_exit_no_excess && !has_anchors && !has_tether {
            self.vel_x.fill(0.0);
            self.vel_y.fill(0.0);
            return Ok(stats);
        }

        self.disp_x.fill(0.0);
        self.disp_y.fill(0.0);

        if !stats.early_exit_no_excess {
            let base_scale = self.config.step_size * (n as f64) * w_span * w_span;
            let disp_scale = match self.config.field.kernel {
                DensityFieldKernel::ScreenedPoisson => base_scale,
                DensityFieldKernel::SingleGaussian => {
                    let k2 = stats.kappa_used * stats.kappa_used;
                    if k2 > 0.0 {
                        base_scale / k2
                    } else {
                        base_scale
                    }
                }
                _ => base_scale,
            };
            for i in 0..n {
                self.disp_x[i] = disp_scale * self.field_out.force_x[i].to_f64();
                self.disp_y[i] = disp_scale * self.field_out.force_y[i].to_f64();
            }
        }

        if has_anchors {
            let xs = layout.x();
            let ys = layout.y();
            let anc_scale = self.config.step_size * self.config.anchor_weight;
            for ((&u, &v), &d0) in self
                .edges_u
                .iter()
                .zip(self.edges_v.iter())
                .zip(self.anchor_targets.iter())
            {
                let ui = u as usize;
                let vi = v as usize;
                let dx = xs[vi].to_f64() - xs[ui].to_f64();
                let dy = ys[vi].to_f64() - ys[ui].to_f64();
                let d = dx.hypot(dy);
                if d > 1e-12 {
                    let coeff = anc_scale * (d - d0) / d;
                    self.disp_x[ui] += coeff * dx;
                    self.disp_y[ui] += coeff * dy;
                    self.disp_x[vi] -= coeff * dx;
                    self.disp_y[vi] -= coeff * dy;
                }
            }
        }

        let max_step = self.config.max_step_ratio * w_span;
        let mom = self.config.momentum;
        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        {
            let (xs, ys) = layout.xy_mut();
            for i in 0..n {
                let mut vx = mom * self.vel_x[i] + self.disp_x[i];
                let mut vy = mom * self.vel_y[i] + self.disp_y[i];
                let norm = vx.hypot(vy);
                if norm > max_step && norm > 0.0 {
                    let s = max_step / norm;
                    vx *= s;
                    vy *= s;
                }
                self.vel_x[i] = vx;
                self.vel_y[i] = vy;
                let nx = xs[i].to_f64() + vx;
                let ny = ys[i].to_f64() + vy;
                xs[i] = S::from_f64(nx);
                ys[i] = S::from_f64(ny);
                sum_x += nx;
                sum_y += ny;
            }

            if self.config.recenter {
                let inv_n = 1.0 / (n as f64);
                let mx = sum_x * inv_n;
                let my = sum_y * inv_n;
                for i in 0..n {
                    xs[i] = S::from_f64(xs[i].to_f64() - mx);
                    ys[i] = S::from_f64(ys[i].to_f64() - my);
                }
            }
        }

        if layout.validate_finite().is_err() {
            return Err(CoreError::numerical_breakdown(
                "ScreenedPoissonRefiner produced non-finite coordinates",
            ));
        }

        if has_tether && let Some((coarse_layout, node_to_coarse)) = &self.coarse_tether {
            align_clustered_centroids(
                layout,
                coarse_layout,
                node_to_coarse,
                self.config.gamma_tether,
            )?;
            if layout.validate_finite().is_err() {
                return Err(CoreError::numerical_breakdown(
                    "ScreenedPoissonRefiner produced non-finite coordinates after centroid tether",
                ));
            }
        }

        Ok(stats)
    }

    /// Returns the expected number of vertices.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns telemetry from the most recent density field evaluation.
    #[inline]
    pub fn last_stats(&self) -> &ScreenedPoissonStats {
        &self.last_stats
    }

    /// Returns the most recently evaluated variational density energy.
    #[inline]
    pub fn last_energy(&self) -> f64 {
        self.last_stats.energy
    }

    /// Returns the refiner configuration.
    #[inline]
    pub fn config(&self) -> &ScreenedPoissonRefinerConfig {
        &self.config
    }

    /// Returns approximate heap memory in bytes held by the underlying density field workspace.
    #[inline]
    pub fn workspace_memory_bytes(&self) -> usize {
        self.field.workspace_memory_bytes()
    }
}

impl<S: Scalar> IterativeSolver for ScreenedPoissonRefiner<S> {
    type Target = Layout2d<S>;

    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        if target.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                target.len(),
                "ScreenedPoissonRefiner::validate_target",
            ));
        }
        target.validate_finite()
    }

    fn step(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        self.validate_target(target)?;
        if self.status.is_finished() {
            return Ok(self.status);
        }
        if self.config.iterations == 0 {
            self.status = SolverStatus::Completed;
            return Ok(self.status);
        }
        let stats = match self.apply_epoch_step(target) {
            Ok(stats) => stats,
            Err(err) => {
                self.status = SolverStatus::Failed;
                return Err(err);
            }
        };
        self.iter += 1;
        self.status = if stats.early_exit_no_excess || stats.energy <= self.config.energy_tol {
            SolverStatus::Converged
        } else if self.iter >= self.config.iterations {
            SolverStatus::Completed
        } else {
            SolverStatus::Running
        };
        Ok(self.status)
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.iter
    }

    #[inline]
    fn total_iterations(&self) -> Option<usize> {
        Some(self.config.iterations)
    }
}

pub(super) fn bbox_of<S: Scalar>(layout: &Layout2d<S>) -> (f64, f64, f64, f64) {
    let xs = layout.x();
    let ys = layout.y();
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for (&x_s, &y_s) in xs.iter().zip(ys.iter()) {
        let x = x_s.to_f64();
        let y = y_s.to_f64();
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    (min_x, max_x, min_y, max_y)
}
