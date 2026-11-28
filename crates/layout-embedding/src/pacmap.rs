use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, IterativeSolver, Layout2d, RiemannianLayout, Scalar,
    SolverStatus,
};
use egraph_neighborhood::DogVaradhanLandmarks;

use crate::config_pacmap_snap::PacmapGraphConfig;
use crate::pacmap_build::{PacmapExternalLandmarks, build_pacmap_data};
use crate::sampling_utils::validate_target_riemannian;

/// 3-tier `PaCMAP-Graph` layout solver (`NB` 1+2-hop, `MN` mid-near, `FP` further) with
/// 3-phase dynamic loss weights and rotation-invariant Riemannian Adam updates.
#[derive(Debug, Clone)]
pub struct PacmapGraphSolver<L: RiemannianLayout = Layout2d<f64>> {
    config: PacmapGraphConfig,
    num_nodes: usize,
    nb_pairs: Vec<(u32, u32)>,
    nb_weights: Vec<f64>,
    mn_pairs: Vec<(u32, u32)>,
    fp_pairs: Vec<(u32, u32)>,
    grad: Vec<f64>,
    m: Vec<f64>,
    v: Vec<f64>,
    log_uv: Vec<f64>,
    log_vu: Vec<f64>,
    delta: Vec<f64>,
    iter: usize,
    status: SolverStatus,
    locked_tangent_dim: Option<usize>,
    _target: PhantomData<fn() -> L>,
}

impl<L: RiemannianLayout> PacmapGraphSolver<L> {
    /// Constructs a 3-tier `PaCMAP-Graph` solver (`NB` 1+2-hop, `MN` mid-near, `FP` further)
    /// using internal Core-First MaxMin BFS landmarks for Mid-Near (`MN`) ranking.
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: PacmapGraphConfig) -> CoreResult<Self> {
        Self::new_inner(graph, config, None)
    }

    /// Constructs a 3-tier `PaCMAP-Graph` solver using precomputed [`DogVaradhanLandmarks`]
    /// for Mid-Near (`MN`) proximity ranking instead of internal BFS landmarks.
    pub fn with_landmarks<W: Scalar>(
        graph: &CsrGraph<W>,
        landmarks: &DogVaradhanLandmarks,
        config: PacmapGraphConfig,
    ) -> CoreResult<Self> {
        Self::new_inner(graph, config, Some(PacmapExternalLandmarks::Dog(landmarks)))
    }

    /// Constructs a 3-tier `PaCMAP-Graph` solver using an externally supplied row-major
    /// `(num_nodes, num_landmarks)` landmark distance table for Mid-Near (`MN`) ranking.
    pub fn with_landmark_distances<W: Scalar>(
        graph: &CsrGraph<W>,
        landmark_distances: &[f64],
        num_landmarks: usize,
        config: PacmapGraphConfig,
    ) -> CoreResult<Self> {
        Self::new_inner(
            graph,
            config,
            Some(PacmapExternalLandmarks::Distances {
                data: landmark_distances,
                num_landmarks,
            }),
        )
    }

    fn new_inner<W: Scalar>(
        graph: &CsrGraph<W>,
        config: PacmapGraphConfig,
        ext_landmarks: Option<PacmapExternalLandmarks<'_>>,
    ) -> CoreResult<Self> {
        let built = build_pacmap_data(graph, &config, ext_landmarks)?;
        let status = if config.total_iterations() == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };
        let n = built.num_nodes;
        Ok(Self {
            config,
            num_nodes: n,
            nb_pairs: built.nb_pairs,
            nb_weights: built.nb_weights,
            mn_pairs: built.mn_pairs,
            fp_pairs: built.fp_pairs,
            grad: Vec::new(),
            m: Vec::new(),
            v: if n > 1 { vec![0.0; n] } else { Vec::new() },
            log_uv: Vec::new(),
            log_vu: Vec::new(),
            delta: Vec::new(),
            iter: 0,
            status,
            locked_tangent_dim: None,
            _target: PhantomData,
        })
    }

    /// Returns the expected number of vertices in the target layout.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of attractive (`NB` + `MN`) pairs stored.
    #[inline]
    pub fn num_springs(&self) -> usize {
        self.nb_pairs.len() + self.mn_pairs.len()
    }

    /// Returns the number of repulsive (`FP`) pairs stored.
    #[inline]
    pub fn num_hinges(&self) -> usize {
        self.fp_pairs.len()
    }

    /// Returns a reference to the active [`PacmapGraphConfig`].
    #[inline]
    pub fn config(&self) -> &PacmapGraphConfig {
        &self.config
    }

    /// Returns the canonical Near (`NB`) undirected pairs (`u < v`).
    #[inline]
    pub fn nb_pairs(&self) -> &[(u32, u32)] {
        &self.nb_pairs
    }

    /// Returns optional degree-attenuated weights aligned with [`Self::nb_pairs`].
    #[inline]
    pub fn nb_weights(&self) -> &[f64] {
        &self.nb_weights
    }

    /// Returns the Mid-Near (`MN`) pairs `(u, v)`.
    #[inline]
    pub fn mn_pairs(&self) -> &[(u32, u32)] {
        &self.mn_pairs
    }

    /// Returns the Further (`FP`) repulsive pairs `(u, v)`.
    #[inline]
    pub fn fp_pairs(&self) -> &[(u32, u32)] {
        &self.fp_pairs
    }

    fn accumulate_tier(
        layout: &L,
        dim: usize,
        pairs: &[(u32, u32)],
        weights: Option<&[f64]>,
        (scale, denom_offset): (f64, f64),
        grad: &mut [f64],
        scratch: (&mut [f64], &mut [f64]),
    ) -> CoreResult<()> {
        let (log_uv, log_vu) = scratch;
        for (idx, &(u, v)) in pairs.iter().enumerate() {
            let ui = u as usize;
            let vi = v as usize;
            if ui == vi {
                continue;
            }
            layout.log_map_into(ui, vi, log_uv)?;
            layout.log_map_into(vi, ui, log_vu)?;
            let mut d2 = 0.0_f64;
            for &val in log_uv.iter().take(dim) {
                d2 += val * val;
            }
            if d2.is_finite() {
                let denom = denom_offset + d2;
                let pair_w = weights.map_or(1.0, |ws| ws[idx]);
                let coeff = (scale * pair_w) / (denom * denom);
                if coeff.is_finite() {
                    let bu = ui * dim;
                    let bv = vi * dim;
                    for a in 0..dim {
                        grad[bu + a] -= coeff * log_uv[a];
                        grad[bv + a] -= coeff * log_vu[a];
                    }
                }
            }
        }
        Ok(())
    }

    fn step_inner(&mut self, layout: &mut L, dim: usize) -> CoreResult<()> {
        let total = self.config.total_iterations();
        if self.num_nodes <= 1 || total == 0 || self.iter >= total {
            return Ok(());
        }

        let total_len = self
            .num_nodes
            .checked_mul(dim)
            .ok_or_else(|| CoreError::invalid_input("PacmapGraphSolver gradient size overflow"))?;
        if self.grad.len() != total_len {
            self.grad.resize(total_len, 0.0);
            self.m.resize(total_len, 0.0);
        }
        if self.v.len() != self.num_nodes {
            self.v.resize(self.num_nodes, 0.0);
        }
        if self.log_uv.len() != dim {
            self.log_uv.resize(dim, 0.0);
            self.log_vu.resize(dim, 0.0);
            self.delta.resize(dim, 0.0);
        }

        let [p1_iters, p2_iters, _] = self.config.phase_iterations;
        let (w_nb, w_mn, w_fp) = if self.iter < p1_iters {
            let p1 = if p1_iters <= 1 {
                0.0
            } else {
                (self.iter as f64) / ((p1_iters - 1) as f64)
            };
            (2.0, 1000.0 * (1.0 - p1) + 3.0 * p1, 1.0)
        } else if self.iter < p1_iters + p2_iters {
            (3.0, 3.0, 1.0)
        } else {
            (1.0, 0.0, 1.0)
        };

        self.grad.fill(0.0);
        if w_nb > 0.0 {
            let nb_w = (!self.nb_weights.is_empty()).then_some(self.nb_weights.as_slice());
            Self::accumulate_tier(
                layout,
                dim,
                &self.nb_pairs,
                nb_w,
                (20.0 * w_nb, 10.0),
                &mut self.grad,
                (&mut self.log_uv, &mut self.log_vu),
            )?;
        }
        if w_mn > 0.0 {
            Self::accumulate_tier(
                layout,
                dim,
                &self.mn_pairs,
                None,
                (20_000.0 * w_mn, 10_000.0),
                &mut self.grad,
                (&mut self.log_uv, &mut self.log_vu),
            )?;
        }
        if w_fp > 0.0 {
            Self::accumulate_tier(
                layout,
                dim,
                &self.fp_pairs,
                None,
                (-2.0 * w_fp, 1.0),
                &mut self.grad,
                (&mut self.log_uv, &mut self.log_vu),
            )?;
        }

        // Rotation-invariant Riemannian Adam: scalar second moment `v[u]` tracks squared
        // gradient norm, and first-moment vector `m_u` is parallel-transported atomically.
        const BETA1: f64 = 0.9;
        const BETA2: f64 = 0.999;
        const ADAM_EPS: f64 = 1e-7;
        let t_step = (self.iter + 1) as i32;
        let bc1 = (1.0 - BETA1.powi(t_step)).max(1e-12);
        let bc2 = (1.0 - BETA2.powi(t_step)).max(1e-12);
        let lr_t = self.config.learning_rate * bc2.sqrt() / bc1;

        for u in 0..self.num_nodes {
            let base = u * dim;
            let mut g_sq = 0.0_f64;
            for a in 0..dim {
                let ga = self.grad[base + a];
                self.m[base + a] = BETA1 * self.m[base + a] + (1.0 - BETA1) * ga;
                g_sq += ga * ga;
            }
            if !g_sq.is_finite() {
                return Err(CoreError::numerical_breakdown(
                    "PacmapGraphSolver gradient norm overflowed",
                ));
            }
            self.v[u] = BETA2 * self.v[u] + (1.0 - BETA2) * g_sq;
            let denom = self.v[u].sqrt() + ADAM_EPS;
            if !denom.is_finite() || denom <= 0.0 {
                return Err(CoreError::numerical_breakdown(
                    "PacmapGraphSolver Adam denominator became non-finite",
                ));
            }
            for a in 0..dim {
                let d = -lr_t * self.m[base + a] / denom;
                if !d.is_finite() {
                    return Err(CoreError::numerical_breakdown(
                        "PacmapGraphSolver Adam step became non-finite",
                    ));
                }
                self.delta[a] = d;
            }
            layout.exp_map_and_transport(u, &self.delta[..dim], &mut self.m[base..base + dim])?;
        }
        Ok(())
    }
}

impl<L: RiemannianLayout> IterativeSolver for PacmapGraphSolver<L> {
    type Target = L;

    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        validate_target_riemannian(
            self.num_nodes,
            self.locked_tangent_dim,
            target,
            "PacmapGraphSolver::validate_target",
        )?;
        Ok(())
    }

    fn step(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        let dim = validate_target_riemannian(
            self.num_nodes,
            self.locked_tangent_dim,
            target,
            "PacmapGraphSolver::step",
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
        self.status = if self.iter >= self.config.total_iterations() {
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
        Some(self.config.total_iterations())
    }
}
