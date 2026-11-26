use egraph_core::{
    CoreError, CoreResult, IterativeSolver, Layout2d, ManifoldLayout2d, Scalar, SolverStatus,
};

use super::drgraph::step_drgraph_state;
use super::graph_cne::step_graph_cne_state;
use super::largevis::step_largevis_state;
use super::manifold_drgraph::{step_drgraph_state_manifold, sync_drgraph_schedule_iter};
use super::manifold_sampling_variants::{
    step_graph_cne_state_manifold, step_largevis_state_manifold, step_pacmap_state_manifold,
};
use super::manifold_variants::{step_sgd2_state_manifold, step_snap_t_fdp_state_manifold};
use super::pacmap::step_pacmap_state;
use super::pairs::{PairStorage, try_for_each_shuffled_pair};
use super::sgd2::step_sgd2_state;
use super::snap_t_fdp::step_snap_t_fdp_state;
use super::solver::{ActiveLayoutKind, SgdSolver};
use super::update::{apply_euclidean_gen_pair, apply_euclidean_sym_pair, apply_manifold_pair};

mod sealed {
    use super::*;

    pub trait Sealed<S: Scalar> {
        fn validate_window_input(
            &self,
            solver: &mut SgdSolver<S>,
            ctx: &'static str,
        ) -> CoreResult<()>;

        fn step_window(&mut self, solver: &mut SgdSolver<S>) -> CoreResult<SolverStatus>;
    }
}

/// Layout types accepted by [`SgdSolver::run_schedule_window`]: [`Layout2d<S>`] and [`ManifoldLayout2d<S>`].
pub trait SgdScheduleWindowLayout<S: Scalar>: sealed::Sealed<S> {}

impl<S: Scalar> sealed::Sealed<S> for Layout2d<S> {
    fn validate_window_input(
        &self,
        solver: &mut SgdSolver<S>,
        ctx: &'static str,
    ) -> CoreResult<()> {
        validate_euclidean_step_preconditions(solver, self, ctx)
    }

    fn step_window(&mut self, solver: &mut SgdSolver<S>) -> CoreResult<SolverStatus> {
        IterativeSolver::<S, Layout2d<S>>::step(solver, self)
    }
}

impl<S: Scalar> SgdScheduleWindowLayout<S> for Layout2d<S> {}

impl<S: Scalar> sealed::Sealed<S> for ManifoldLayout2d<S> {
    fn validate_window_input(
        &self,
        solver: &mut SgdSolver<S>,
        ctx: &'static str,
    ) -> CoreResult<()> {
        validate_manifold_step_preconditions(solver, self, ctx)
    }

    fn step_window(&mut self, solver: &mut SgdSolver<S>) -> CoreResult<SolverStatus> {
        solver.step_manifold(self)
    }
}

impl<S: Scalar> SgdScheduleWindowLayout<S> for ManifoldLayout2d<S> {}

fn validate_euclidean_step_preconditions<S: Scalar>(
    solver: &mut SgdSolver<S>,
    layout: &Layout2d<S>,
    ctx: &'static str,
) -> CoreResult<()> {
    if layout.len() != solver.num_nodes {
        return Err(CoreError::dimension_mismatch(
            solver.num_nodes,
            layout.len(),
            ctx,
        ));
    }
    layout.validate_finite()?;
    match solver.active_layout {
        ActiveLayoutKind::Unset => solver.active_layout = ActiveLayoutKind::Euclidean2d,
        ActiveLayoutKind::Euclidean2d => {}
        ActiveLayoutKind::Nd(_) => {
            return Err(CoreError::invalid_input(format!(
                "{ctx}: cannot switch from LayoutNd to Layout2d on stateful solver"
            )));
        }
        ActiveLayoutKind::Manifold(_) => {
            return Err(CoreError::invalid_input(format!(
                "{ctx}: cannot switch from ManifoldLayout2d to Layout2d on stateful solver"
            )));
        }
    }
    Ok(())
}

fn validate_manifold_step_preconditions<S: Scalar>(
    solver: &mut SgdSolver<S>,
    layout: &ManifoldLayout2d<S>,
    ctx: &'static str,
) -> CoreResult<()> {
    if layout.len() != solver.num_nodes {
        return Err(CoreError::dimension_mismatch(
            solver.num_nodes,
            layout.len(),
            ctx,
        ));
    }
    layout.as_layout().validate_finite()?;
    let kind = layout.kind();
    match solver.active_layout {
        ActiveLayoutKind::Unset => solver.active_layout = ActiveLayoutKind::Manifold(kind),
        ActiveLayoutKind::Manifold(prev) if prev == kind => {}
        ActiveLayoutKind::Manifold(_) => {
            return Err(CoreError::invalid_input(format!(
                "{ctx}: cannot switch ManifoldKind on stateful solver"
            )));
        }
        ActiveLayoutKind::Euclidean2d => {
            return Err(CoreError::invalid_input(format!(
                "{ctx}: cannot switch from Layout2d to ManifoldLayout2d on stateful solver"
            )));
        }
        ActiveLayoutKind::Nd(_) => {
            return Err(CoreError::invalid_input(format!(
                "{ctx}: cannot switch from LayoutNd to ManifoldLayout2d on stateful solver"
            )));
        }
    }
    Ok(())
}

impl<S: Scalar> SgdSolver<S> {
    /// Executes one SGD iteration on a non-Euclidean [`ManifoldLayout2d<S>`] across all
    /// [`ManifoldLayout2d<S>`] geometries (`HyperbolicPoincare`, `HyperbolicLorentz`,
    /// `Spherical`, and `Torus2d`).
    pub fn step_manifold(&mut self, layout: &mut ManifoldLayout2d<S>) -> CoreResult<SolverStatus> {
        validate_manifold_step_preconditions(self, layout, "SgdSolver::step_manifold")?;
        if self.status.is_finished() {
            return Ok(self.status);
        }

        if let Some(st) = self.snap_t_fdp.as_mut() {
            step_snap_t_fdp_state_manifold(
                st,
                &mut self.block_order,
                &mut self.rng,
                self.num_nodes,
                self.iter,
                layout,
            )?;
            return Ok(self.finish_iteration());
        }
        if let Some(st) = self.pacmap.as_mut() {
            step_pacmap_state_manifold(st, self.num_nodes, self.iter, layout)?;
            return Ok(self.finish_iteration());
        }
        if self.sgd2.is_some() {
            let eta_f64 = self.eta_at(self.iter);
            if let Some(st) = self.sgd2.as_mut() {
                step_sgd2_state_manifold(
                    st,
                    &mut self.pairs,
                    &mut self.block_order,
                    &mut self.rng,
                    self.num_nodes,
                    eta_f64,
                    self.eta_max,
                    self.schedule_cfg.step_scale,
                    layout,
                )?;
            }
            return Ok(self.finish_iteration());
        }
        if let Some(st) = self.largevis.as_ref() {
            step_largevis_state_manifold(st, &mut self.rng, self.num_nodes, self.iter, layout)?;
            return Ok(self.finish_iteration());
        }
        if let Some(st) = self.drgraph.as_mut() {
            step_drgraph_state_manifold(st, &mut self.rng, self.num_nodes, self.iter, layout)?;
            return Ok(self.finish_iteration());
        }
        if let Some(st) = self.graph_cne.as_mut() {
            step_graph_cne_state_manifold(st, &mut self.rng, self.num_nodes, self.iter, layout)?;
            return Ok(self.finish_iteration());
        }

        let eta_f64 = self.eta_at(self.iter);
        if self.num_nodes > 1 && eta_f64.is_finite() && eta_f64 > 0.0 {
            let eta = S::from_f64(eta_f64);
            let max_mu = self.max_mu_scalar();
            match &mut self.pairs {
                PairStorage::Symmetric(pairs) => {
                    try_for_each_shuffled_pair(pairs, &mut self.block_order, &mut self.rng, |p| {
                        apply_manifold_pair(layout, eta, max_mu, p.to_pair_spec())
                    })?;
                }
                PairStorage::General(pairs) => {
                    try_for_each_shuffled_pair(pairs, &mut self.block_order, &mut self.rng, |p| {
                        apply_manifold_pair(layout, eta, max_mu, p.to_pair_spec())
                    })?;
                }
            }
            self.run_streaming_negatives(eta, |e, p| {
                apply_manifold_pair(layout, e, max_mu, p.to_pair_spec())
            })?;
        }

        Ok(self.finish_iteration())
    }

    /// Executes up to `steps` manifold SGD iterations or stops when finished.
    pub fn run_steps_manifold(
        &mut self,
        layout: &mut ManifoldLayout2d<S>,
        steps: usize,
    ) -> CoreResult<SolverStatus> {
        for _ in 0..steps {
            if self.status.is_finished() {
                break;
            }
            let status = self.step_manifold(layout)?;
            if status.is_finished() {
                break;
            }
        }
        Ok(self.status)
    }

    /// Runs manifold SGD optimization to completion.
    pub fn run_manifold(&mut self, layout: &mut ManifoldLayout2d<S>) -> CoreResult<SolverStatus> {
        while !self.status.is_finished() {
            let status = self.step_manifold(layout)?;
            if status.is_finished() {
                break;
            }
        }
        Ok(self.status)
    }

    fn run_schedule_window_inner<L: SgdScheduleWindowLayout<S> + ?Sized>(
        &mut self,
        layout: &mut L,
        start_iter: usize,
        count: usize,
        ctx: &'static str,
    ) -> CoreResult<SolverStatus> {
        layout.validate_window_input(self, ctx)?;
        if count == 0 {
            return Ok(self.status);
        }
        let total = self.schedule_cfg.iterations;
        if start_iter >= total {
            self.iter = total;
            self.status = SolverStatus::Completed;
            return Ok(self.status);
        }
        if start_iter != self.iter
            && let Some(st) = self.drgraph.as_mut()
        {
            sync_drgraph_schedule_iter(st, start_iter);
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
            let st = layout.step_window(self)?;
            if st.is_finished() {
                break;
            }
        }
        Ok(self.status)
    }

    /// Executes `count` SGD iterations over schedule indices `start_iter..start_iter + count`
    /// on either [`Layout2d<S>`] or [`ManifoldLayout2d<S>`] (stopping early if `schedule.iterations`
    /// is reached).
    pub fn run_schedule_window<L: SgdScheduleWindowLayout<S> + ?Sized>(
        &mut self,
        layout: &mut L,
        start_iter: usize,
        count: usize,
    ) -> CoreResult<SolverStatus> {
        self.run_schedule_window_inner(layout, start_iter, count, "SgdSolver::run_schedule_window")
    }

    /// Executes `count` non-Euclidean manifold SGD iterations over schedule indices
    /// `start_iter..start_iter + count` (stopping early if `schedule.iterations` is reached).
    pub fn run_schedule_window_manifold(
        &mut self,
        layout: &mut ManifoldLayout2d<S>,
        start_iter: usize,
        count: usize,
    ) -> CoreResult<SolverStatus> {
        self.run_schedule_window_inner(
            layout,
            start_iter,
            count,
            "SgdSolver::run_schedule_window_manifold",
        )
    }
}

impl<S: Scalar> IterativeSolver<S> for SgdSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        validate_euclidean_step_preconditions(self, layout, "SgdSolver::step")?;
        if self.status.is_finished() {
            return Ok(self.status);
        }

        if let Some(st) = self.snap_t_fdp.as_mut() {
            step_snap_t_fdp_state(
                st,
                &mut self.block_order,
                &mut self.rng,
                self.num_nodes,
                self.iter,
                layout,
            );
            return Ok(self.finish_iteration());
        }
        if let Some(st) = self.pacmap.as_mut() {
            step_pacmap_state(st, self.num_nodes, self.iter, layout);
            return Ok(self.finish_iteration());
        }
        if self.sgd2.is_some() {
            let eta_f64 = self.eta_at(self.iter);
            if let Some(st) = self.sgd2.as_mut() {
                step_sgd2_state(
                    st,
                    &mut self.pairs,
                    &mut self.block_order,
                    &mut self.rng,
                    self.num_nodes,
                    eta_f64,
                    self.eta_max,
                    self.schedule_cfg.step_scale,
                    layout,
                )?;
            }
            return Ok(self.finish_iteration());
        }
        if let Some(st) = self.largevis.as_ref() {
            step_largevis_state(st, &mut self.rng, self.num_nodes, self.iter, layout);
            return Ok(self.finish_iteration());
        }
        if let Some(st) = self.drgraph.as_mut() {
            step_drgraph_state(st, &mut self.rng, self.num_nodes, self.iter, layout);
            return Ok(self.finish_iteration());
        }
        if let Some(st) = self.graph_cne.as_mut() {
            step_graph_cne_state(st, &mut self.rng, self.num_nodes, self.iter, layout);
            return Ok(self.finish_iteration());
        }

        let eta_f64 = self.eta_at(self.iter);
        if self.num_nodes > 1 && eta_f64.is_finite() && eta_f64 > 0.0 {
            let eta = S::from_f64(eta_f64);
            let max_mu = self.max_mu_scalar();
            let (xs, ys) = layout.xy_mut();
            match &mut self.pairs {
                PairStorage::Symmetric(pairs) => {
                    try_for_each_shuffled_pair(
                        pairs,
                        &mut self.block_order,
                        &mut self.rng,
                        |&p| {
                            apply_euclidean_sym_pair(xs, ys, eta, max_mu, p);
                            Ok(())
                        },
                    )?;
                }
                PairStorage::General(pairs) => {
                    try_for_each_shuffled_pair(pairs, &mut self.block_order, &mut self.rng, |p| {
                        apply_euclidean_gen_pair(xs, ys, eta, max_mu, p);
                        Ok(())
                    })?;
                }
            }
            let (xs, ys) = layout.xy_mut();
            self.run_streaming_negatives(eta, |e, p| {
                apply_euclidean_sym_pair(xs, ys, e, max_mu, p);
                Ok(())
            })?;
        }

        Ok(self.finish_iteration())
    }

    fn status(&self) -> SolverStatus {
        self.status
    }

    fn iteration(&self) -> usize {
        self.iter
    }

    fn total_iterations(&self) -> usize {
        self.schedule_cfg.iterations
    }
}

impl<S: Scalar> IterativeSolver<S, ManifoldLayout2d<S>> for SgdSolver<S> {
    fn step(&mut self, layout: &mut ManifoldLayout2d<S>) -> CoreResult<SolverStatus> {
        self.step_manifold(layout)
    }

    fn run_steps(
        &mut self,
        layout: &mut ManifoldLayout2d<S>,
        steps: usize,
    ) -> CoreResult<SolverStatus> {
        self.run_steps_manifold(layout, steps)
    }

    fn run(&mut self, layout: &mut ManifoldLayout2d<S>) -> CoreResult<SolverStatus> {
        self.run_manifold(layout)
    }

    fn status(&self) -> SolverStatus {
        self.status
    }

    fn iteration(&self) -> usize {
        self.iter
    }

    fn total_iterations(&self) -> usize {
        self.schedule_cfg.iterations
    }
}
