use std::f64::consts::TAU;
use std::marker::PhantomData;

use egraph_core::{
    CoreResult, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, RiemannianLayout, Scalar,
    SolverStatus,
};

use crate::config_negative::DrGraphConfig;
use crate::drgraph_coarsen::{DrGraphLevel, build_drgraph_hierarchy};
use crate::sampling_utils::{AliasTable, is_collapsed_riemannian, validate_target_riemannian};

/// Multilevel negative-sampling `DRGraph` layout solver (`Zhu et al., 2020`).
#[derive(Debug, Clone)]
pub struct DrGraphSolver<L: RiemannianLayout = Layout2d<f64>> {
    config: DrGraphConfig,
    num_nodes: usize,
    pos_u: Vec<u32>,
    pos_v: Vec<u32>,
    edge_alias: Option<AliasTable>,
    levels: Vec<DrGraphLevel>,
    fine_to_level: Vec<Vec<u32>>,
    level_reps: Vec<Vec<u32>>,
    level_iters: Vec<usize>,
    current_level: usize,
    level_iter: usize,
    initialized_coords: bool,
    log_uv: Vec<f64>,
    log_vu: Vec<f64>,
    delta_u: Vec<f64>,
    delta_v: Vec<f64>,
    centroid_acc: Vec<f64>,
    centroid_cnt: Vec<usize>,
    iter: usize,
    status: SolverStatus,
    rng: DeterministicRng,
    locked_tangent_dim: Option<usize>,
    _target: PhantomData<fn() -> L>,
}

impl<L: RiemannianLayout> DrGraphSolver<L> {
    /// Constructs a multilevel negative-sampling `DRGraph` solver with random-merge coarsening,
    /// $b$-parameterized Cauchy proximity, and cooperative coarse-to-fine stepping.
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: DrGraphConfig) -> CoreResult<Self> {
        let built = build_drgraph_hierarchy(graph, &config)?;
        Ok(Self {
            config,
            num_nodes: built.num_nodes,
            pos_u: built.pos_u,
            pos_v: built.pos_v,
            edge_alias: built.edge_alias,
            levels: built.levels,
            fine_to_level: built.fine_to_level,
            level_reps: built.level_reps,
            level_iters: built.level_iters,
            current_level: built.current_level,
            level_iter: 0,
            initialized_coords: false,
            log_uv: Vec::new(),
            log_vu: Vec::new(),
            delta_u: Vec::new(),
            delta_v: Vec::new(),
            centroid_acc: Vec::new(),
            centroid_cnt: Vec::new(),
            iter: 0,
            status: built.status,
            rng: built.rng,
            locked_tangent_dim: None,
            _target: PhantomData,
        })
    }

    /// Returns the expected number of vertices in the target layout.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of positive similarity pairs stored.
    #[inline]
    pub fn num_springs(&self) -> usize {
        self.pos_u.len()
    }

    /// Returns the number of levels in the multilevel coarsening hierarchy (`>= 1`).
    #[inline]
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Returns a reference to the active [`DrGraphConfig`].
    #[inline]
    pub fn config(&self) -> &DrGraphConfig {
        &self.config
    }

    /// Returns the learning rate for the upcoming iteration (`0.0` if finished).
    #[inline]
    pub fn current_lr(&self) -> f64 {
        let total = self.config.iterations;
        if total == 0 || self.iter >= total {
            return 0.0;
        }
        (self.config.initial_lr * (1.0 - (self.iter as f64) / (total as f64)))
            .max(self.config.min_lr)
    }

    fn init_coarse_coords(&mut self, layout: &mut L, dim: usize) -> CoreResult<()> {
        let collapsed = is_collapsed_riemannian(layout, 1e-12)?;
        let l_init = self.current_level;
        let ln = self.levels[l_init].num_nodes;
        if collapsed {
            for c in 0..ln {
                let rep = self.level_reps[l_init][c] as usize;
                for slot in self.delta_u.iter_mut().take(dim) {
                    *slot = self.rng.next_f64() - 0.5;
                }
                layout.exp_map(rep, &self.delta_u[..dim])?;
            }
        } else if l_init > 0 {
            let map = &self.fine_to_level[l_init];
            self.centroid_acc.resize(ln * dim, 0.0);
            self.centroid_acc.fill(0.0);
            self.centroid_cnt.resize(ln, 0usize);
            self.centroid_cnt.fill(0);
            for (u, &c_u32) in map.iter().enumerate().take(self.num_nodes) {
                let c = c_u32 as usize;
                self.centroid_cnt[c] += 1;
                let rep = self.level_reps[l_init][c] as usize;
                if u != rep {
                    layout.log_map_into(rep, u, &mut self.log_uv)?;
                    let base = c * dim;
                    for a in 0..dim {
                        self.centroid_acc[base + a] += self.log_uv[a];
                    }
                }
            }
            for c in 0..ln {
                let cnt = self.centroid_cnt[c];
                if cnt > 1 {
                    let inv = 1.0 / (cnt as f64);
                    let base = c * dim;
                    for a in 0..dim {
                        self.delta_u[a] = self.centroid_acc[base + a] * inv;
                    }
                    let rep = self.level_reps[l_init][c] as usize;
                    layout.exp_map(rep, &self.delta_u[..dim])?;
                }
            }
        }
        self.initialized_coords = true;
        Ok(())
    }

    fn uncoarsen_step(&mut self, layout: &mut L, dim: usize) -> CoreResult<()> {
        self.level_iter += 1;
        while self.current_level > 0 && self.level_iter >= self.level_iters[self.current_level] {
            let finer = self.current_level - 1;
            let fine_n = self.levels[finer].num_nodes;
            if let (Some(coarse_of), Some(counts)) = (
                self.levels[finer].coarse_of_finer.as_ref(),
                self.levels[finer].cluster_counts_of_finer.as_ref(),
            ) {
                for (f, &c_u32) in coarse_of.iter().enumerate().take(fine_n) {
                    let rf = self.level_reps[finer][f] as usize;
                    let rc = self.level_reps[self.current_level][c_u32 as usize] as usize;
                    if rf != rc {
                        layout.log_map_into(rf, rc, &mut self.log_uv)?;
                        layout.exp_map(rf, &self.log_uv[..dim])?;
                    }
                }
                for (f, &c_u32) in coarse_of.iter().enumerate().take(fine_n) {
                    let c = c_u32 as usize;
                    if counts[c] > 1 {
                        let rf = self.level_reps[finer][f] as usize;
                        if dim == 2 {
                            let angle = TAU * self.rng.next_f64();
                            let radius = 1e-4 * (0.25 + 0.75 * self.rng.next_f64());
                            let (s, cs) = angle.sin_cos();
                            self.delta_u[0] = radius * cs;
                            self.delta_u[1] = radius * s;
                        } else {
                            let mut norm_sq = 0.0_f64;
                            for a in 0..dim {
                                let comp = self.rng.next_f64() - 0.5;
                                self.delta_u[a] = comp;
                                norm_sq += comp * comp;
                            }
                            let radius = 1e-4 * (0.25 + 0.75 * self.rng.next_f64());
                            let norm = norm_sq.sqrt();
                            if norm <= 1e-12 || !norm.is_finite() {
                                self.delta_u[..dim].fill(0.0);
                                self.delta_u[0] = radius;
                            } else {
                                let scale = radius / norm;
                                for d in self.delta_u.iter_mut().take(dim) {
                                    *d *= scale;
                                }
                            }
                        }
                        layout.exp_map(rf, &self.delta_u[..dim])?;
                    }
                }
            }
            self.current_level = finer;
            self.level_iter = 0;
        }
        Ok(())
    }

    fn step_inner(&mut self, layout: &mut L, dim: usize) -> CoreResult<()> {
        let total = self.config.iterations;
        if self.num_nodes <= 1 || total == 0 || self.iter >= total {
            return Ok(());
        }
        if self.log_uv.len() != dim {
            self.log_uv.resize(dim, 0.0);
            self.log_vu.resize(dim, 0.0);
            self.delta_u.resize(dim, 0.0);
            self.delta_v.resize(dim, 0.0);
        }

        if !self.initialized_coords {
            self.init_coarse_coords(layout, dim)?;
        }

        let l = self.current_level;
        let ln = self.levels[l].num_nodes;
        let lr = self.current_lr();
        let gamma = if l == self.levels.len() - 1 && self.levels.len() > 1 {
            self.config.gamma_coarse
        } else {
            self.config.gamma_fine
        };
        let b = self.config.b;
        let neg_samples = self.config.negative_samples;

        if let Some(edge_alias) = &self.edge_alias
            && lr.is_finite()
            && lr > 0.0
            && ln > 1
        {
            let m = self.pos_u.len();
            for _ in 0..m {
                let e_idx = edge_alias.sample(&mut self.rng);
                let u0 = self.pos_u[e_idx] as usize;
                let v0 = self.pos_v[e_idx] as usize;
                let (su, sv) = if l > 0 {
                    let map = &self.fine_to_level[l];
                    (map[u0] as usize, map[v0] as usize)
                } else {
                    (u0, v0)
                };
                let (uc, vc) = if (self.rng.next_u32() & 1) == 0 {
                    (su, sv)
                } else {
                    (sv, su)
                };
                let ru = self.level_reps[l][uc] as usize;
                let rv = self.level_reps[l][vc] as usize;

                if uc != vc {
                    layout.log_map_into(ru, rv, &mut self.log_uv)?;
                    layout.log_map_into(rv, ru, &mut self.log_vu)?;
                    let mut d2 = 0.0_f64;
                    for &val in &self.log_uv[..dim] {
                        d2 += val * val;
                    }
                    if d2.is_finite() {
                        let d2_safe = if b < 1.0 { d2.max(1e-12) } else { d2 };
                        let (pow_bm1, pow_b) = if (b - 2.0).abs() == 0.0 {
                            (d2_safe, d2_safe * d2_safe)
                        } else if (b - 1.0).abs() == 0.0 {
                            (1.0, d2_safe)
                        } else {
                            (d2_safe.powf(b - 1.0), d2_safe.powf(b))
                        };
                        let attr_coeff = (2.0 * b * pow_bm1) / (1.0 + pow_b);
                        if attr_coeff.is_finite() {
                            let mut ok = true;
                            for a in 0..dim {
                                let su_a = (lr * attr_coeff * self.log_uv[a]).clamp(-4.0, 4.0);
                                let sv_a = (lr * attr_coeff * self.log_vu[a]).clamp(-4.0, 4.0);
                                if !su_a.is_finite() || !sv_a.is_finite() {
                                    ok = false;
                                    break;
                                }
                                self.delta_u[a] = su_a;
                                self.delta_v[a] = sv_a;
                            }
                            if ok {
                                layout.exp_map(ru, &self.delta_u[..dim])?;
                                layout.exp_map(rv, &self.delta_v[..dim])?;
                            }
                        }
                    }
                }

                if gamma > 0.0 {
                    for _ in 0..neg_samples {
                        let kc = self.rng.gen_range_usize(ln);
                        if kc == uc {
                            continue;
                        }
                        let rk = self.level_reps[l][kc] as usize;
                        layout.log_map_into(ru, rk, &mut self.log_uv)?;
                        let mut rd2 = 0.0_f64;
                        for &val in &self.log_uv[..dim] {
                            rd2 += val * val;
                        }
                        if rd2.is_finite() && rd2 > 0.0 {
                            let pow_b = if (b - 2.0).abs() == 0.0 {
                                rd2 * rd2
                            } else if (b - 1.0).abs() == 0.0 {
                                rd2
                            } else {
                                rd2.powf(b)
                            };
                            let rep_coeff = (2.0 * b * gamma) / ((rd2 + 1e-6) * (1.0 + pow_b));
                            if rep_coeff.is_finite() {
                                let mut ok = true;
                                for a in 0..dim {
                                    let su_a = (-lr * rep_coeff * self.log_uv[a]).clamp(-4.0, 4.0);
                                    if !su_a.is_finite() {
                                        ok = false;
                                        break;
                                    }
                                    self.delta_u[a] = su_a;
                                }
                                if ok {
                                    layout.exp_map(ru, &self.delta_u[..dim])?;
                                }
                            }
                        }
                    }
                }
            }
        }

        self.uncoarsen_step(layout, dim)?;

        if self.current_level > 0 {
            let map = &self.fine_to_level[self.current_level];
            let reps = &self.level_reps[self.current_level];
            for (u, &c_u32) in map.iter().enumerate().take(self.num_nodes) {
                let rc = reps[c_u32 as usize] as usize;
                if u != rc {
                    layout.log_map_into(u, rc, &mut self.log_uv)?;
                    layout.exp_map(u, &self.log_uv[..dim])?;
                }
            }
        }
        Ok(())
    }
}

impl<L: RiemannianLayout> IterativeSolver for DrGraphSolver<L> {
    type Target = L;

    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        validate_target_riemannian(
            self.num_nodes,
            self.locked_tangent_dim,
            target,
            "DrGraphSolver::validate_target",
        )?;
        Ok(())
    }

    fn step(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        let dim = validate_target_riemannian(
            self.num_nodes,
            self.locked_tangent_dim,
            target,
            "DrGraphSolver::step",
        )?;
        if self.status.is_finished() {
            return Ok(self.status);
        }
        self.locked_tangent_dim.get_or_insert(dim);
        if let Err(err) = self.step_inner(target, dim) {
            self.status = SolverStatus::Failed;
            return Err(err);
        }
        self.iter += 1;
        self.status = if self.iter >= self.config.iterations {
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
