use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, ManifoldKind,
    RiemannianLayout, Scalar, SolverStatus, VisitStamp,
};

use super::config::{SgdScheduleConfig, SgdScheduleKind};
use super::pair_constructors::{build_full_pair_storage, build_random_pair_storage};
use super::pairs::{
    MAX_PACKED_NODES, PairStorage, compute_eta_bounds, init_pair_blocking,
    try_for_each_shuffled_pair, validate_graph_weights, validate_schedule_config,
};
use super::sgd2_kernels::{ScratchAccum, accumulate_gen, accumulate_sym, flush_batch};
use super::stepping::{lock_target_layout, validate_target_layout};
use super::variant_config::Sgd2Config;

/// Doubly-stochastic mini-batch preconditioned stress SGD solver parameterized over any
/// [`RiemannianLayout`] target (`Layout2d`, `Layout3d`, `LayoutNd`, `ManifoldLayout2d`).
///
/// Uses an $\mathrm{SO}(2)$-invariant full $2 \times 2$ Gauss-Newton tangent Hessian
/// preconditioner $(H + \lambda I)^{-1} g$ on 2D tangent spaces (and $\mathrm{SO}(d)$-invariant
/// isotropic trace curvature preconditioning for $d \ne 2$).
///
/// Note: This single-criterion preconditioned stress SGD solver (using
/// [`SgdScheduleKind::Squared`]) is distinct from Ahmed et al. (2021) multi-criteria `(SGD)^2`.
#[derive(Debug, Clone)]
pub struct Sgd2Solver<L: RiemannianLayout = Layout2d<f64>> {
    num_nodes: usize,
    pairs: PairStorage<L::Scalar>,
    num_springs: usize,
    block_order: Vec<u32>,
    batch_size: usize,
    damping: f64,
    schedule_cfg: SgdScheduleConfig,
    eta_min: f64,
    eta_max: f64,
    iter: usize,
    status: SolverStatus,
    locked_dim: Option<usize>,
    locked_manifold: Option<ManifoldKind>,
    rng: DeterministicRng,
    grad: Vec<f64>,
    hess_diag: Vec<f64>,
    hess_yy: Vec<f64>,
    hess_xy: Vec<f64>,
    tangent_scratch: Vec<f64>,
    touched_stamp: VisitStamp,
    touched_nodes: Vec<u32>,
    _layout: PhantomData<fn() -> L>,
}

impl<L: RiemannianLayout + 'static> Sgd2Solver<L> {
    /// Constructs a preconditioned mini-batch stress SGD solver from `graph` and `config`.
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: Sgd2Config) -> CoreResult<Self> {
        config.validate()?;
        validate_graph_weights(graph, "Sgd2Solver::new")?;
        let n = graph.num_nodes();
        if n > MAX_PACKED_NODES {
            return Err(CoreError::invalid_input(format!(
                "num_nodes ({n}) exceeds packed u32 index limit {MAX_PACKED_NODES}"
            )));
        }

        let schedule_cfg = SgdScheduleConfig::new(config.iterations)
            .with_eps(config.eps)
            .with_kind(SgdScheduleKind::Squared)
            .with_step_scale(config.step_scale);
        let mut pairs = if n <= 256 {
            build_full_pair_storage(graph)?
        } else {
            build_random_pair_storage(graph, 16, config.seed)?
        };
        let num_springs = pairs.len();
        let (eta_min, eta_max) = compute_eta_bounds(&pairs, None, schedule_cfg.eps);
        let mut rng = DeterministicRng::from_seed_u64(config.seed);
        let block_order = match &mut pairs {
            PairStorage::Symmetric(v) => init_pair_blocking(v, &mut rng),
            PairStorage::General(v) => init_pair_blocking(v, &mut rng),
        };
        let status = if config.iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };
        let touched_cap = (config.batch_size.saturating_mul(2)).min(n);

        Ok(Self {
            num_nodes: n,
            pairs,
            num_springs,
            block_order,
            batch_size: config.batch_size,
            damping: config.damping,
            schedule_cfg,
            eta_min,
            eta_max,
            iter: 0,
            status,
            locked_dim: None,
            locked_manifold: None,
            rng,
            grad: Vec::new(),
            hess_diag: Vec::new(),
            hess_yy: Vec::new(),
            hess_xy: Vec::new(),
            tangent_scratch: Vec::new(),
            touched_stamp: VisitStamp::new(n),
            touched_nodes: Vec::with_capacity(touched_cap),
            _layout: PhantomData,
        })
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

    /// Returns the learning rate $\eta$ for the upcoming iteration (`0.0` if finished).
    #[inline]
    pub fn current_eta(&self) -> f64 {
        self.eta_at(self.iter)
    }

    #[inline]
    fn eta_at(&self, t: usize) -> f64 {
        self.schedule_cfg.evaluate(t, self.eta_min, self.eta_max)
    }

    /// Resets the iteration schedule and deterministic RNG without reallocating stored pair buffers.
    pub fn reset_schedule(&mut self, schedule_cfg: SgdScheduleConfig, seed: u64) -> CoreResult<()> {
        validate_schedule_config(&schedule_cfg)?;
        let (eta_min, eta_max) = compute_eta_bounds(&self.pairs, None, schedule_cfg.eps);
        self.eta_min = eta_min;
        self.eta_max = eta_max;
        self.schedule_cfg = schedule_cfg;
        self.rng = DeterministicRng::from_seed_u64(seed);
        self.iter = 0;
        self.locked_dim = None;
        self.locked_manifold = None;
        self.grad.fill(0.0);
        self.hess_diag.fill(0.0);
        self.hess_yy.fill(0.0);
        self.hess_xy.fill(0.0);
        self.touched_nodes.clear();
        self.touched_stamp.clear();
        self.status = if self.schedule_cfg.iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };
        Ok(())
    }

    fn ensure_scratch(&mut self, dim: usize) {
        let total = self.num_nodes * dim;
        if self.grad.len() != total {
            self.grad.resize(total, 0.0);
        } else {
            self.grad.fill(0.0);
        }
        if self.hess_diag.len() != self.num_nodes {
            self.hess_diag.resize(self.num_nodes, 0.0);
        } else {
            self.hess_diag.fill(0.0);
        }
        if dim == 2 {
            if self.hess_yy.len() != self.num_nodes {
                self.hess_yy.resize(self.num_nodes, 0.0);
            } else {
                self.hess_yy.fill(0.0);
            }
            if self.hess_xy.len() != self.num_nodes {
                self.hess_xy.resize(self.num_nodes, 0.0);
            } else {
                self.hess_xy.fill(0.0);
            }
        }
        if self.tangent_scratch.len() < dim {
            self.tangent_scratch.resize(dim, 0.0);
        }
        self.touched_nodes.clear();
        self.touched_stamp.clear();
    }

    fn step_inner(&mut self, layout: &mut L) -> CoreResult<()> {
        let dim = layout.tangent_dim();
        self.ensure_scratch(dim);
        let eta_f64 = self.eta_at(self.iter);
        if self.num_nodes <= 1 || !eta_f64.is_finite() || eta_f64 <= 0.0 || self.eta_max <= 0.0 {
            return Ok(());
        }
        let alpha_t = (eta_f64 / self.eta_max).clamp(0.0, self.schedule_cfg.step_scale.max(1.0));
        let batch_size = self.batch_size;
        let damping = self.damping;
        let mut batch_count = 0usize;

        let mut scratch = ScratchAccum {
            dim,
            grad: &mut self.grad,
            hess_diag: &mut self.hess_diag,
            hess_yy: &mut self.hess_yy,
            hess_xy: &mut self.hess_xy,
            tan_buf: &mut self.tangent_scratch,
            touched_stamp: &mut self.touched_stamp,
            touched_nodes: &mut self.touched_nodes,
        };

        match &mut self.pairs {
            PairStorage::Symmetric(sym_pairs) => {
                try_for_each_shuffled_pair(
                    sym_pairs,
                    &mut self.block_order,
                    &mut self.rng,
                    |&p| {
                        accumulate_sym(layout, &mut scratch, p)?;
                        batch_count += 1;
                        if batch_count >= batch_size {
                            flush_batch(layout, &mut scratch, alpha_t, damping)?;
                            batch_count = 0;
                        }
                        Ok(())
                    },
                )?;
            }
            PairStorage::General(gen_pairs) => {
                try_for_each_shuffled_pair(gen_pairs, &mut self.block_order, &mut self.rng, |p| {
                    accumulate_gen(layout, &mut scratch, p)?;
                    batch_count += 1;
                    if batch_count >= batch_size {
                        flush_batch(layout, &mut scratch, alpha_t, damping)?;
                        batch_count = 0;
                    }
                    Ok(())
                })?;
            }
        }
        if batch_count > 0 {
            flush_batch(layout, &mut scratch, alpha_t, damping)?;
        }
        Ok(())
    }

    /// Executes `count` SGD2 iterations over schedule indices `start_iter..start_iter + count`
    /// on `layout` (stopping early if `schedule.iterations` is reached).
    pub fn run_schedule_window(
        &mut self,
        layout: &mut L,
        start_iter: usize,
        count: usize,
    ) -> CoreResult<SolverStatus> {
        self.validate_target(layout)?;
        if count == 0 {
            return Ok(self.status);
        }
        let total = self.schedule_cfg.iterations;
        if start_iter >= total {
            self.iter = total;
            self.status = SolverStatus::Completed;
            return Ok(self.status);
        }
        self.iter = start_iter;
        self.status = if start_iter == 0 {
            SolverStatus::Initialized
        } else {
            SolverStatus::Running
        };
        for _ in 0..count {
            if self.status.is_finished() {
                break;
            }
            let st = self.step(layout)?;
            if st.is_finished() {
                break;
            }
        }
        Ok(self.status)
    }
}

impl<L: RiemannianLayout + 'static> IterativeSolver for Sgd2Solver<L> {
    type Target = L;

    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        validate_target_layout(
            self.num_nodes,
            self.locked_dim,
            self.locked_manifold,
            target,
            "Sgd2Solver::validate_target",
        )
    }

    fn step(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        self.validate_target(target)?;
        if self.status.is_finished() {
            return Ok(self.status);
        }
        lock_target_layout(&mut self.locked_dim, &mut self.locked_manifold, target);

        if let Err(err) = self.step_inner(target) {
            self.status = SolverStatus::Failed;
            return Err(err);
        }
        if let Err(err) = target.validate_finite() {
            self.status = SolverStatus::Failed;
            return Err(err);
        }

        self.iter += 1;
        self.status = if self.iter >= self.schedule_cfg.iterations {
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
        Some(self.schedule_cfg.iterations)
    }
}
