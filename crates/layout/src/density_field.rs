//! Cooperative Screened Poisson unilateral upper-bound density field refiner.
//!
//! Provides [`ScreenedPoissonRefiner`], implementing [`IterativeSolver`] and
//! [`ScreenedPoissonRefiner::apply_epoch_step`] for epoch-boundary Eulerian repulsion,
//! optional 1-hop edge-length anchor springs, and Pitfall #7 rigid cluster centroid tethering
//! (`align_clustered_centroids`).

use egraph_clustering::align_clustered_centroids;
use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Scalar,
    SolverStatus,
};
use egraph_spatial::{
    DensityFieldKernel, ScreenedPoissonConfig, ScreenedPoissonField2d, ScreenedPoissonOutput,
    ScreenedPoissonStats,
};

/// Configuration for [`ScreenedPoissonRefiner`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenedPoissonRefinerConfig {
    /// Underlying Eulerian density field configuration (`default: ScreenedPoissonConfig::default()`).
    pub field: ScreenedPoissonConfig,
    /// Maximum number of refinement iterations (`default: 20`).
    pub iterations: usize,
    /// Dimensionless gradient step multiplier (`default: 0.25`, must be `> 0.0`).
    pub step_size: f64,
    /// Heavy-ball momentum coefficient in `[0.0, 1.0)` (`default: 0.5`).
    pub momentum: f64,
    /// Per-vertex displacement clamp as a fraction of domain span (`default: 0.05`, must be `> 0.0`).
    pub max_step_ratio: f64,
    /// Optional 1-hop edge-length anchor spring weight (`default: 0.0`).
    pub anchor_weight: f64,
    /// Optional Pitfall #7 cluster centroid tether strength in `[0.0, 1.0]` (`default: 0.0`).
    pub gamma_tether: f64,
    /// Whether to recenter layout mean to the origin after each active step (`default: true`).
    pub recenter: bool,
    /// Early-convergence energy tolerance (`default: 1e-12`).
    pub energy_tol: f64,
    /// Deterministic RNG seed for breaking exact coincident degeneracy (`default: 42`).
    pub seed: u64,
}

impl Default for ScreenedPoissonRefinerConfig {
    fn default() -> Self {
        Self {
            field: ScreenedPoissonConfig::default(),
            iterations: 20,
            step_size: 0.25,
            momentum: 0.5,
            max_step_ratio: 0.05,
            anchor_weight: 0.0,
            gamma_tether: 0.0,
            recenter: true,
            energy_tol: 1e-12,
            seed: 42,
        }
    }
}

impl ScreenedPoissonRefinerConfig {
    /// Validates all refiner hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        self.field.validate()?;
        if !self.step_size.is_finite() || self.step_size <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "step_size must be finite and > 0.0, got {}",
                self.step_size
            )));
        }
        if !self.momentum.is_finite() || !(0.0..1.0).contains(&self.momentum) {
            return Err(CoreError::invalid_input(format!(
                "momentum must be finite and in [0.0, 1.0), got {}",
                self.momentum
            )));
        }
        if !self.max_step_ratio.is_finite() || self.max_step_ratio <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "max_step_ratio must be finite and > 0.0, got {}",
                self.max_step_ratio
            )));
        }
        if !self.anchor_weight.is_finite() || self.anchor_weight < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "anchor_weight must be finite and >= 0.0, got {}",
                self.anchor_weight
            )));
        }
        if !self.gamma_tether.is_finite() || !(0.0..=1.0).contains(&self.gamma_tether) {
            return Err(CoreError::invalid_input(format!(
                "gamma_tether must be finite and in [0.0, 1.0], got {}",
                self.gamma_tether
            )));
        }
        if !self.energy_tol.is_finite() || self.energy_tol < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "energy_tol must be finite and >= 0.0, got {}",
                self.energy_tol
            )));
        }
        Ok(())
    }
}

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
        let field = ScreenedPoissonField2d::<S>::new(config.field)?;
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
    pub fn with_graph<W: Scalar>(
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
        coarse_layout: Layout2d<S>,
        node_to_coarse: Vec<u32>,
    ) -> CoreResult<Self> {
        self.set_coarse_tether(coarse_layout, node_to_coarse)?;
        Ok(self)
    }

    /// Attaches or replaces the coarse-level cluster centroid tether (`align_clustered_centroids`).
    pub fn set_coarse_tether(
        &mut self,
        coarse_layout: Layout2d<S>,
        node_to_coarse: Vec<u32>,
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
        for &cid in &node_to_coarse {
            let idx = cid as usize;
            if idx >= num_clusters {
                return Err(CoreError::index_out_of_bounds(
                    idx,
                    num_clusters,
                    "ScreenedPoissonRefiner::set_coarse_tether",
                ));
            }
        }
        self.coarse_tether = Some((coarse_layout, node_to_coarse));
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
            self.field.evaluate(layout, &mut self.field_out)?;
            self.last_stats = self.field_out.stats;
            return Ok(self.last_stats);
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

        self.field.evaluate(layout, &mut self.field_out)?;
        let stats = self.field_out.stats;
        self.last_stats = stats;

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

        if has_tether && let Some((coarse_layout, node_to_coarse)) = &self.coarse_tether {
            align_clustered_centroids(
                layout,
                coarse_layout,
                node_to_coarse,
                self.config.gamma_tether,
            )?;
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

impl<S: Scalar> IterativeSolver<S> for ScreenedPoissonRefiner<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if self.status.is_finished() {
            return Ok(self.status);
        }
        if self.config.iterations == 0 {
            self.status = SolverStatus::Completed;
            return Ok(self.status);
        }
        let stats = self.apply_epoch_step(layout)?;
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
    fn total_iterations(&self) -> usize {
        self.config.iterations
    }
}

fn bbox_of<S: Scalar>(layout: &Layout2d<S>) -> (f64, f64, f64, f64) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_disk_layout(side: usize) -> Layout2d<f64> {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for r in 0..side {
            for c in 0..side {
                let x = -0.8 + 1.6 * (c as f64) / ((side - 1) as f64);
                let y = -0.8 + 1.6 * (r as f64) / ((side - 1) as f64);
                if x * x + y * y <= 0.64 {
                    xs.push(x);
                    ys.push(y);
                }
            }
        }
        Layout2d::from_xy(xs, ys).unwrap()
    }

    #[test]
    fn test_refiner_step_run_equivalence_and_excess_reduction() {
        let layout_init = dense_disk_layout(24);
        let n = layout_init.len();
        let cfg = ScreenedPoissonRefinerConfig {
            field: ScreenedPoissonConfig {
                grid_size: 128,
                padding_factor: 2,
                rho_max: Some(0.15),
                sigma: Some(0.10),
                kappa: Some(1.0),
                ..ScreenedPoissonConfig::default()
            },
            iterations: 8,
            step_size: 0.3,
            momentum: 0.4,
            ..ScreenedPoissonRefinerConfig::default()
        };

        let mut r_run = ScreenedPoissonRefiner::<f64>::new(n, cfg).unwrap();
        let mut r_step = ScreenedPoissonRefiner::<f64>::new(n, cfg).unwrap();
        let mut l_run = layout_init.clone();
        let mut l_step = layout_init;

        r_step.step(&mut l_step).unwrap();
        let initial_excess_mass = r_step.last_stats().excess_mass;
        assert!(initial_excess_mass > 0.1);
        r_step.run_steps(&mut l_step, 2).unwrap();
        assert_eq!(r_step.iteration(), 3);
        r_step.run(&mut l_step).unwrap();

        r_run.run(&mut l_run).unwrap();

        assert_eq!(r_run.iteration(), r_step.iteration());
        assert_eq!(r_run.status(), r_step.status());
        assert_eq!(l_run, l_step);
        assert!(r_run.last_energy() >= 0.0);
        assert!(r_run.last_stats().excess_mass < initial_excess_mass);
        // Overdense plateau must expand: final bbox span larger than initial 1.6.
        let (min_x, max_x, min_y, max_y) = bbox_of(&l_run);
        assert!(
            max_x - min_x > 1.9,
            "expected expansion > 1.9, got {}",
            max_x - min_x
        );
        assert!(
            max_y - min_y > 1.9,
            "expected expansion > 1.9, got {}",
            max_y - min_y
        );
    }

    #[test]
    fn test_refiner_deterministic_seed_on_coincident_zero_layout() {
        let n = 32usize;
        let l0 = Layout2d::<f64>::zeros(n);
        let cfg = ScreenedPoissonRefinerConfig {
            field: ScreenedPoissonConfig {
                grid_size: 64,
                padding_factor: 2,
                ..ScreenedPoissonConfig::default()
            },
            iterations: 4,
            seed: 99,
            ..ScreenedPoissonRefinerConfig::default()
        };

        let mut r1 = ScreenedPoissonRefiner::<f64>::new(n, cfg).unwrap();
        let mut r2 = ScreenedPoissonRefiner::<f64>::new(n, cfg).unwrap();
        let mut l1 = l0.clone();
        let mut l2 = l0;
        r1.run(&mut l1).unwrap();
        r2.run(&mut l2).unwrap();
        assert_eq!(l1, l2);
        let (min_x, max_x, min_y, max_y) = bbox_of(&l1);
        assert!(max_x - min_x > 0.0 || max_y - min_y > 0.0);
    }

    #[test]
    fn test_refiner_with_graph_anchors_and_coarse_tether() {
        let g =
            CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)], None, false, true)
                .unwrap();
        let mut layout =
            Layout2d::from_xy(vec![0.0, 0.1, 0.1, 0.0], vec![0.0, 0.0, 0.1, 0.1]).unwrap();
        let coarse = Layout2d::from_xy(vec![5.0, -5.0], vec![2.0, -2.0]).unwrap();
        let n2c = vec![0u32, 0u32, 1u32, 1u32];

        let cfg = ScreenedPoissonRefinerConfig {
            field: ScreenedPoissonConfig {
                grid_size: 64,
                padding_factor: 2,
                rho_max: Some(0.1),
                ..ScreenedPoissonConfig::default()
            },
            iterations: 3,
            anchor_weight: 0.2,
            gamma_tether: 1.0,
            recenter: false,
            ..ScreenedPoissonRefinerConfig::default()
        };
        let mut refiner = ScreenedPoissonRefiner::<f64>::with_graph(&g, cfg)
            .unwrap()
            .with_coarse_tether(coarse, n2c)
            .unwrap();
        refiner.run(&mut layout).unwrap();

        let cx0 = 0.5 * (layout.x()[0] + layout.x()[1]);
        let cy0 = 0.5 * (layout.y()[0] + layout.y()[1]);
        let cx1 = 0.5 * (layout.x()[2] + layout.x()[3]);
        let cy1 = 0.5 * (layout.y()[2] + layout.y()[3]);
        assert!((cx0 - 5.0).abs() < 1e-9);
        assert!((cy0 - 2.0).abs() < 1e-9);
        assert!((cx1 + 5.0).abs() < 1e-9);
        assert!((cy1 + 2.0).abs() < 1e-9);
    }
}
