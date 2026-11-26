use egraph_core::{CoreError, CoreResult, DeterministicRng, Scalar, SolverStatus};

#[allow(unused_imports)]
use egraph_core::{IterativeSolver, Layout2d, ManifoldLayout2d};

use super::config::{SgdScheduleConfig, SgdScheduleKind};
use super::drgraph_coarsen::DrGraphState;
use super::graph_cne::GraphCneState;
use super::largevis::LargeVisState;
use super::pacmap::PacmapState;
use super::pairs::{
    INDEX_MASK, MAX_PACKED_NODES, PackedSymPair, PairStorage, compute_eta_bounds,
    eta_bounds_from_extrema, init_pair_blocking, validate_pair_endpoints, validate_schedule_config,
};
use super::sgd2::Sgd2State;
use super::snap_t_fdp::SnapTFdpState;

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

/// Contiguous packed Stochastic Gradient Descent (SGD) layout solver.
///
/// Supports symmetric and general two-sided spring pairs, one-sided hinge repulsion pairs,
/// Ortmann (2016) sparse stress (`ortmann_sparse_stress`), Track B 3-tier sparse barrier
/// schedules, cache-blocked L2-resident shuffling, streaming negative sampling, compact-memory
/// `SNAP-tFDP`, 3-tier `PaCMAP-Graph`, `LargeVis`, `DRGraph`, `Graph-CNE`, doubly-stochastic
/// diagonal-preconditioned stress SGD (`sgd2`, distinct from Ahmed et al. 2021 multi-criteria
/// `(SGD)^2`), Euclidean [`Layout2d<S>`] via [`IterativeSolver<S>`], and non-Euclidean
/// [`ManifoldLayout2d<S>`] stepping.
#[derive(Debug, Clone)]
pub struct SgdSolver<S: Scalar = f64> {
    pub(super) num_nodes: usize,
    pub(super) pairs: PairStorage<S>,
    pub(super) num_springs: usize,
    pub(super) num_hinges: usize,
    pub(super) block_order: Vec<u32>,
    pub(super) streaming: Option<StreamingState<S>>,
    pub(super) snap_t_fdp: Option<SnapTFdpState>,
    pub(super) pacmap: Option<PacmapState>,
    pub(super) sgd2: Option<Sgd2State>,
    pub(super) largevis: Option<LargeVisState>,
    pub(super) drgraph: Option<DrGraphState>,
    pub(super) graph_cne: Option<GraphCneState>,
    pub(super) schedule_cfg: SgdScheduleConfig,
    pub(super) eta_min: f64,
    pub(super) eta_max: f64,
    pub(super) iter: usize,
    pub(super) status: SolverStatus,
    pub(super) rng: DeterministicRng,
}

impl<S: Scalar> SgdSolver<S> {
    /// Constructs an [`SgdSolver`] directly from pre-packed symmetric spring/hinge pairs.
    pub fn from_packed_symmetric(
        num_nodes: usize,
        mut pairs: Vec<PackedSymPair<S>>,
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
            if p.d <= S::ZERO {
                return Err(CoreError::invalid_input(format!(
                    "SgdSolver::from_packed_symmetric: target distances must be > 0 at index {idx}"
                )));
            }
            if !p.w.is_finite() {
                return Err(CoreError::non_finite("SgdSolver::from_packed_symmetric::w"));
            }
            if p.w < S::ZERO {
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
            snap_t_fdp: None,
            pacmap: None,
            sgd2: None,
            largevis: None,
            drgraph: None,
            graph_cne: None,
            schedule_cfg,
            eta_min,
            eta_max,
            iter: 0,
            status,
            rng,
        })
    }

    pub(super) fn from_pair_storage(
        num_nodes: usize,
        mut pairs: PairStorage<S>,
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
            snap_t_fdp: None,
            pacmap: None,
            sgd2: None,
            largevis: None,
            drgraph: None,
            graph_cne: None,
            schedule_cfg,
            eta_min,
            eta_max,
            iter: 0,
            status,
            rng,
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

    /// Resets the iteration schedule and deterministic RNG without reallocating stored pair buffers.
    pub fn reset_schedule(&mut self, schedule_cfg: SgdScheduleConfig, seed: u64) -> CoreResult<()> {
        if self.snap_t_fdp.is_some()
            || self.pacmap.is_some()
            || self.sgd2.is_some()
            || self.largevis.is_some()
            || self.drgraph.is_some()
            || self.graph_cne.is_some()
        {
            return Err(CoreError::invalid_input(
                "SgdSolver::reset_schedule is only supported for pair-based SGD modes",
            ));
        }
        validate_schedule_config(&schedule_cfg)?;
        let (eta_min, eta_max) = compute_eta_bounds(&self.pairs, None, schedule_cfg.eps);
        self.schedule_cfg = schedule_cfg;
        self.eta_min = eta_min;
        self.eta_max = eta_max;
        self.rng = DeterministicRng::from_seed_u64(seed);
        self.iter = 0;
        self.status = if schedule_cfg.iterations == 0 {
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
        if let Some(st) = &self.snap_t_fdp {
            let total = st.config.iterations;
            if total == 0 || t >= total {
                return 0.0;
            }
            return st.config.learning_rate * (1.0 - (t as f64) / (total as f64));
        }
        if let Some(st) = &self.pacmap {
            let total = st.config.total_iterations();
            if total == 0 || t >= total {
                return 0.0;
            }
            return st.config.learning_rate;
        }
        if let Some(st) = &self.largevis {
            let total = st.config.iterations;
            if total == 0 || t >= total {
                return 0.0;
            }
            return (st.config.initial_lr * (1.0 - (t as f64) / (total as f64)))
                .max(st.config.min_lr);
        }
        if let Some(st) = &self.drgraph {
            let total = st.config.iterations;
            if total == 0 || t >= total {
                return 0.0;
            }
            return (st.config.initial_lr * (1.0 - (t as f64) / (total as f64)))
                .max(st.config.min_lr);
        }
        if let Some(st) = &self.graph_cne {
            let total = st.config.iterations;
            if total == 0 || t >= total {
                return 0.0;
            }
            return (st.config.initial_lr * (1.0 - (t as f64) / (total as f64)))
                .max(st.config.min_lr);
        }
        let t_max = self.schedule_cfg.iterations;
        if t_max == 0 || t >= t_max {
            return 0.0;
        }
        let scale = self.schedule_cfg.step_scale;
        match self.schedule_cfg.kind {
            SgdScheduleKind::Constant => scale,
            SgdScheduleKind::Linear => {
                if self.eta_max <= 0.0 {
                    return 0.0;
                }
                let b = if t_max <= 1 {
                    0.0
                } else {
                    (self.eta_max - self.eta_min) / ((t_max - 1) as f64)
                };
                scale * (self.eta_max - b * (t as f64)).max(0.0)
            }
            SgdScheduleKind::Quadratic => {
                if self.eta_min <= 0.0 || self.eta_max <= 0.0 {
                    return 0.0;
                }
                let b = if t_max <= 1 {
                    0.0
                } else {
                    (1.0 - (self.eta_min / self.eta_max).sqrt()) / ((t_max - 1) as f64)
                };
                let factor = 1.0 - b * (t as f64);
                scale * self.eta_max * factor * factor
            }
            SgdScheduleKind::Exponential => {
                if self.eta_min <= 0.0 || self.eta_max <= 0.0 {
                    return 0.0;
                }
                let b = if t_max <= 1 {
                    0.0
                } else {
                    (self.eta_max / self.eta_min).ln() / ((t_max - 1) as f64)
                };
                scale * self.eta_max * (-b * (t as f64)).exp()
            }
            SgdScheduleKind::Reciprocal => {
                if self.eta_min <= 0.0 || self.eta_max <= 0.0 {
                    return 0.0;
                }
                let b = if t_max <= 1 {
                    0.0
                } else {
                    (self.eta_max / self.eta_min - 1.0) / ((t_max - 1) as f64)
                };
                scale * self.eta_max / (1.0 + b * (t as f64))
            }
            SgdScheduleKind::Squared => {
                if self.eta_max <= 0.0 {
                    return 0.0;
                }
                let progress = if t_max <= 1 {
                    0.0
                } else {
                    (t as f64) / ((t_max - 1) as f64)
                };
                let factor = 1.0 - progress;
                scale * (self.eta_max * factor * factor).max(self.eta_min)
            }
        }
    }

    pub(super) fn run_streaming_negatives<F>(&mut self, eta: S, mut apply_sym: F) -> CoreResult<()>
    where
        F: FnMut(S, PackedSymPair<S>) -> CoreResult<()>,
    {
        if let Some(stream) = &mut self.streaming
            && stream.negatives_per_node > 0
            && stream.weight > S::ZERO
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
                    apply_sym(
                        eta,
                        PackedSymPair::new(
                            u_u32,
                            v_u32,
                            stream.hinge_only,
                            stream.target_distance,
                            stream.weight,
                        ),
                    )?;
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
