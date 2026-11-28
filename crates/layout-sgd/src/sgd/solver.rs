use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, DeterministicRng, Layout2d, ManifoldKind, RiemannianLayout, Scalar,
    SolverStatus,
};

use super::config::{SgdScheduleConfig, SgdScheduleKind};
use super::pair_buffer::SgdPairBuffer;
use super::pairs::{
    INDEX_MASK, MAX_PACKED_NODES, PackedSymPair, PairStorage, compute_eta_bounds,
    eta_bounds_from_extrema, init_pair_blocking, validate_pair_endpoints, validate_schedule_config,
};

#[derive(Debug, Clone)]
pub(super) struct StreamingState<S: Scalar> {
    pub(super) negatives_per_node: usize,
    pub(super) target_distance: S,
    pub(super) weight: S,
    pub(super) hinge_only: bool,
    pub(super) adj_indptr: Vec<u32>,
    pub(super) adj_indices: Vec<u32>,
    pub(super) node_order: Vec<u32>,
}

/// Contiguous packed Stochastic Gradient Descent (SGD) layout solver parameterized over any
/// [`RiemannianLayout`] target (`Layout2d`, `Layout3d`, `LayoutNd`, `ManifoldLayout2d`).
///
/// Supports symmetric and general two-sided spring pairs, one-sided hinge repulsion pairs,
/// Ortmann (2016) sparse stress (`ortmann_sparse_stress`), 3-tier sparse barrier schedules
/// (`sparse_barrier_dog`, `sparse_barrier_from_landmarks`), cache-blocked L2-resident shuffling,
/// streaming negative sampling, and zero-copy [`SgdPairBuffer`] construction.
#[derive(Debug, Clone)]
pub struct SgdSolver<L: RiemannianLayout = Layout2d<f64>> {
    pub(super) num_nodes: usize,
    pub(super) pairs: PairStorage<L::Scalar>,
    pub(super) num_springs: usize,
    pub(super) num_hinges: usize,
    pub(super) block_order: Vec<u32>,
    pub(super) streaming: Option<StreamingState<L::Scalar>>,
    pub(super) schedule_cfg: SgdScheduleConfig,
    pub(super) eta_min: f64,
    pub(super) eta_max: f64,
    pub(super) iter: usize,
    pub(super) status: SolverStatus,
    pub(super) locked_dim: Option<usize>,
    pub(super) locked_manifold: Option<ManifoldKind>,
    pub(super) rng: DeterministicRng,
    pub(super) _layout: PhantomData<fn() -> L>,
}

impl<L: RiemannianLayout> SgdSolver<L> {
    /// Constructs an [`SgdSolver`] directly from a pre-populated [`SgdPairBuffer`] without
    /// intermediate pair allocations.
    pub fn from_pair_buffer(
        num_nodes: usize,
        buffer: SgdPairBuffer<L::Scalar>,
        schedule_cfg: SgdScheduleConfig,
        seed: u64,
    ) -> CoreResult<Self> {
        let (pairs, num_springs, num_hinges) = buffer.into_parts();
        Self::from_packed_symmetric(
            num_nodes,
            pairs,
            num_springs,
            num_hinges,
            schedule_cfg,
            seed,
        )
    }

    pub(super) fn from_packed_symmetric(
        num_nodes: usize,
        mut pairs: Vec<PackedSymPair<L::Scalar>>,
        num_springs: usize,
        num_hinges: usize,
        schedule_cfg: SgdScheduleConfig,
        seed: u64,
    ) -> CoreResult<Self> {
        if num_nodes > MAX_PACKED_NODES {
            return Err(CoreError::invalid_input(format!(
                "num_nodes ({num_nodes}) exceeds packed u32 index limit {MAX_PACKED_NODES}"
            )));
        }
        validate_schedule_config(&schedule_cfg)?;
        if num_springs + num_hinges != pairs.len() {
            return Err(CoreError::dimension_mismatch(
                pairs.len(),
                num_springs + num_hinges,
                "SgdSolver::from_packed_symmetric",
            ));
        }

        let mut w_min = f64::INFINITY;
        let mut w_max = 0.0_f64;
        for (idx, p) in pairs.iter().enumerate() {
            let u = p.i as usize;
            let v = (p.j_hinge & INDEX_MASK) as usize;
            validate_pair_endpoints(u, v, num_nodes, idx, "SgdSolver::from_packed_symmetric")?;
            if !p.d.is_finite() {
                return Err(CoreError::non_finite("SgdSolver::from_packed_symmetric::d"));
            }
            if p.d <= L::Scalar::ZERO {
                return Err(CoreError::invalid_input(format!(
                    "SgdSolver::from_packed_symmetric: target distances must be > 0 at index {idx}"
                )));
            }
            if !p.w.is_finite() {
                return Err(CoreError::non_finite("SgdSolver::from_packed_symmetric::w"));
            }
            if p.w < L::Scalar::ZERO {
                return Err(CoreError::invalid_input(format!(
                    "SgdSolver::from_packed_symmetric: weights must be >= 0 at index {idx}"
                )));
            }
            let wf = p.w.to_f64();
            if wf > 0.0 {
                w_min = w_min.min(wf);
                w_max = w_max.max(wf);
            }
        }

        let (eta_min, eta_max) = eta_bounds_from_extrema(w_min, w_max, None, schedule_cfg.eps);
        let status = if schedule_cfg.iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };
        let mut rng = DeterministicRng::from_seed_u64(seed);
        let block_order = init_pair_blocking(&mut pairs, &mut rng);

        Ok(Self {
            num_nodes,
            pairs: PairStorage::Symmetric(pairs),
            num_springs,
            num_hinges,
            block_order,
            streaming: None,
            schedule_cfg,
            eta_min,
            eta_max,
            iter: 0,
            status,
            locked_dim: None,
            locked_manifold: None,
            rng,
            _layout: PhantomData,
        })
    }

    pub(super) fn from_pair_storage(
        num_nodes: usize,
        mut pairs: PairStorage<L::Scalar>,
        num_springs: usize,
        num_hinges: usize,
        schedule_cfg: SgdScheduleConfig,
        seed: u64,
    ) -> CoreResult<Self> {
        if num_nodes > MAX_PACKED_NODES {
            return Err(CoreError::invalid_input(format!(
                "num_nodes ({num_nodes}) exceeds packed u32 index limit {MAX_PACKED_NODES}"
            )));
        }
        validate_schedule_config(&schedule_cfg)?;
        let (eta_min, eta_max) = compute_eta_bounds(&pairs, None, schedule_cfg.eps);
        let status = if schedule_cfg.iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };
        let mut rng = DeterministicRng::from_seed_u64(seed);
        let block_order = match &mut pairs {
            PairStorage::Symmetric(v) => init_pair_blocking(v, &mut rng),
            PairStorage::General(v) => init_pair_blocking(v, &mut rng),
        };
        Ok(Self {
            num_nodes,
            pairs,
            num_springs,
            num_hinges,
            block_order,
            streaming: None,
            schedule_cfg,
            eta_min,
            eta_max,
            iter: 0,
            status,
            locked_dim: None,
            locked_manifold: None,
            rng,
            _layout: PhantomData,
        })
    }

    pub(super) fn empty(
        num_nodes: usize,
        schedule: SgdScheduleConfig,
        seed: u64,
    ) -> CoreResult<Self> {
        Self::from_packed_symmetric(num_nodes, Vec::new(), 0, 0, schedule, seed)
    }

    /// Returns the expected number of vertices in the layout.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of two-sided spring pairs stored.
    #[inline]
    pub fn num_springs(&self) -> usize {
        self.num_springs
    }

    /// Returns the number of one-sided hinge repulsion pairs stored.
    #[inline]
    pub fn num_hinges(&self) -> usize {
        self.num_hinges
    }

    /// Returns the current solver execution status.
    #[inline]
    pub fn status(&self) -> SolverStatus {
        self.status
    }

    /// Returns the number of completed iterations.
    #[inline]
    pub fn iteration(&self) -> usize {
        self.iter
    }

    /// Returns the total scheduled iteration count.
    #[inline]
    pub fn total_iterations(&self) -> Option<usize> {
        Some(self.schedule_cfg.iterations)
    }

    #[inline]
    pub(super) fn max_mu_f64(&self) -> f64 {
        self.schedule_cfg.max_mu.unwrap_or(1.0)
    }

    /// Resets the iteration schedule and deterministic RNG without reallocating stored pair buffers.
    pub fn reset_schedule(&mut self, schedule_cfg: SgdScheduleConfig, seed: u64) -> CoreResult<()> {
        validate_schedule_config(&schedule_cfg)?;
        let extra = self.streaming.as_ref().map(|s| s.weight);
        let (eta_min, eta_max) = compute_eta_bounds(&self.pairs, extra, schedule_cfg.eps);
        self.eta_min = eta_min;
        self.eta_max = eta_max;
        self.schedule_cfg = schedule_cfg;
        self.rng = DeterministicRng::from_seed_u64(seed);
        self.iter = 0;
        self.locked_dim = None;
        self.locked_manifold = None;
        self.status = if self.schedule_cfg.iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };
        Ok(())
    }

    /// Returns the learning rate $\eta$ for the upcoming iteration (`0.0` if finished).
    pub fn current_eta(&self) -> f64 {
        self.eta_at(self.iter)
    }

    pub(super) fn eta_at(&self, t: usize) -> f64 {
        let t_max = self.schedule_cfg.iterations;
        if t_max == 0 || t >= t_max {
            return 0.0;
        }
        eval_schedule_eta(t, t_max, self.eta_min, self.eta_max, &self.schedule_cfg)
    }

    pub(super) fn run_streaming_negatives<F>(&mut self, mut apply_sym: F) -> CoreResult<()>
    where
        F: FnMut(PackedSymPair<L::Scalar>) -> CoreResult<()>,
    {
        if let Some(stream) = &mut self.streaming
            && stream.negatives_per_node > 0
            && stream.weight > L::Scalar::ZERO
        {
            self.rng.shuffle(&mut stream.node_order);
            let max_trials = stream.negatives_per_node.saturating_mul(4).max(8);
            for &u_u32 in &stream.node_order {
                let row_start = stream.adj_indptr[u_u32 as usize] as usize;
                let row_end = stream.adj_indptr[(u_u32 as usize) + 1] as usize;
                let adj_u = &stream.adj_indices[row_start..row_end];
                if adj_u.len() + 1 >= self.num_nodes {
                    continue;
                }
                let mut accepted = 0usize;
                let mut trials = 0usize;
                while accepted < stream.negatives_per_node && trials < max_trials {
                    trials += 1;
                    let v_u32 = self.rng.gen_range_usize(self.num_nodes) as u32;
                    if v_u32 == u_u32 || adj_u.binary_search(&v_u32).is_ok() {
                        continue;
                    }
                    apply_sym(PackedSymPair::new(
                        u_u32,
                        v_u32,
                        stream.hinge_only,
                        stream.target_distance,
                        stream.weight,
                    ))?;
                    accepted += 1;
                }
            }
        }
        Ok(())
    }

    pub(super) fn finish_iteration(&mut self) -> SolverStatus {
        self.iter += 1;
        self.status = if self.iter >= self.schedule_cfg.iterations {
            SolverStatus::Completed
        } else {
            SolverStatus::Running
        };
        self.status
    }
}

pub(super) fn eval_schedule_eta(
    t: usize,
    t_max: usize,
    eta_min: f64,
    eta_max: f64,
    cfg: &SgdScheduleConfig,
) -> f64 {
    let scale = cfg.step_scale;
    match cfg.kind {
        SgdScheduleKind::Constant => scale,
        SgdScheduleKind::Linear => {
            if eta_max <= 0.0 {
                return 0.0;
            }
            let b = if t_max <= 1 {
                0.0
            } else {
                (eta_max - eta_min) / ((t_max - 1) as f64)
            };
            scale * (eta_max - b * (t as f64)).max(0.0)
        }
        SgdScheduleKind::Quadratic => {
            if eta_min <= 0.0 || eta_max <= 0.0 {
                return 0.0;
            }
            let b = if t_max <= 1 {
                0.0
            } else {
                (1.0 - (eta_min / eta_max).sqrt()) / ((t_max - 1) as f64)
            };
            let factor = 1.0 - b * (t as f64);
            scale * eta_max * factor * factor
        }
        SgdScheduleKind::Exponential => {
            if eta_min <= 0.0 || eta_max <= 0.0 {
                return 0.0;
            }
            let b = if t_max <= 1 {
                0.0
            } else {
                (eta_max / eta_min).ln() / ((t_max - 1) as f64)
            };
            scale * eta_max * (-b * (t as f64)).exp()
        }
        SgdScheduleKind::Reciprocal => {
            if eta_min <= 0.0 || eta_max <= 0.0 {
                return 0.0;
            }
            let b = if t_max <= 1 {
                0.0
            } else {
                (eta_max / eta_min - 1.0) / ((t_max - 1) as f64)
            };
            scale * eta_max / (1.0 + b * (t as f64))
        }
        SgdScheduleKind::Squared => {
            if eta_max <= 0.0 {
                return 0.0;
            }
            let progress = if t_max <= 1 {
                0.0
            } else {
                (t as f64) / ((t_max - 1) as f64)
            };
            let factor = 1.0 - progress;
            scale * (eta_max * factor * factor).max(eta_min)
        }
    }
}
