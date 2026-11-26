use egraph_core::{
    CoreError, CoreResult, IterativeSolver, Layout2d, ManifoldLayout2d, Scalar, SolverStatus,
};

use super::drgraph::step_drgraph_state;
use super::graph_cne::step_graph_cne_state;
use super::largevis::step_largevis_state;
use super::pacmap::step_pacmap_state;
use super::pairs::{PairStorage, try_for_each_shuffled_pair};
use super::sgd2::step_sgd2_state;
use super::snap_t_fdp::step_snap_t_fdp_state;
use super::solver::SgdSolver;
use super::update::{apply_euclidean_gen_pair, apply_euclidean_sym_pair, apply_manifold_pair};

impl<S: Scalar> SgdSolver<S> {
    /// Executes one SGD iteration on a non-Euclidean [`ManifoldLayout2d<S>`] across
    /// `HyperbolicPoincare`, `Spherical`, and `Torus2d` geometries.
    pub fn step_manifold(&mut self, layout: &mut ManifoldLayout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "SgdSolver::step_manifold",
            ));
        }
        layout.as_layout().validate_finite()?;
        if self.snap_t_fdp.is_some()
            || self.pacmap.is_some()
            || self.sgd2.is_some()
            || self.largevis.is_some()
            || self.drgraph.is_some()
            || self.graph_cne.is_some()
        {
            return Err(CoreError::invalid_input(
                "SgdSolver::step_manifold is not supported for SnapTFdp, PacmapGraph, Sgd2, LargeVis, DrGraph, or GraphCne modes",
            ));
        }
        if self.status.is_finished() {
            return Ok(self.status);
        }

        let eta_f64 = self.eta_at(self.iter);
        if self.num_nodes > 1 && eta_f64.is_finite() && eta_f64 > 0.0 {
            let eta = S::from_f64(eta_f64);
            match &mut self.pairs {
                PairStorage::Symmetric(pairs) => {
                    try_for_each_shuffled_pair(pairs, &mut self.block_order, &mut self.rng, |p| {
                        apply_manifold_pair(layout, eta, p.to_pair_spec())
                    })?;
                }
                PairStorage::General(pairs) => {
                    try_for_each_shuffled_pair(pairs, &mut self.block_order, &mut self.rng, |p| {
                        apply_manifold_pair(layout, eta, p.to_pair_spec())
                    })?;
                }
            }
            self.run_streaming_negatives(eta, |e, p| {
                apply_manifold_pair(layout, e, p.to_pair_spec())
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

    /// Executes `count` Euclidean SGD iterations over schedule indices `start_iter..start_iter + count`
    /// (stopping early if `schedule.iterations` is reached).
    pub fn run_schedule_window(
        &mut self,
        layout: &mut Layout2d<S>,
        start_iter: usize,
        count: usize,
    ) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "SgdSolver::run_schedule_window",
            ));
        }
        layout.validate_finite()?;
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

    /// Executes `count` non-Euclidean manifold SGD iterations over schedule indices
    /// `start_iter..start_iter + count` (stopping early if `schedule.iterations` is reached).
    pub fn run_schedule_window_manifold(
        &mut self,
        layout: &mut ManifoldLayout2d<S>,
        start_iter: usize,
        count: usize,
    ) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "SgdSolver::run_schedule_window_manifold",
            ));
        }
        layout.as_layout().validate_finite()?;
        if self.snap_t_fdp.is_some()
            || self.pacmap.is_some()
            || self.sgd2.is_some()
            || self.largevis.is_some()
            || self.drgraph.is_some()
            || self.graph_cne.is_some()
        {
            return Err(CoreError::invalid_input(
                "SgdSolver::run_schedule_window_manifold is not supported for SnapTFdp, PacmapGraph, Sgd2, LargeVis, DrGraph, or GraphCne modes",
            ));
        }
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
            let st = self.step_manifold(layout)?;
            if st.is_finished() {
                break;
            }
        }
        Ok(self.status)
    }
}

impl<S: Scalar> IterativeSolver<S> for SgdSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "SgdSolver::step",
            ));
        }
        layout.validate_finite()?;
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
            let (xs, ys) = layout.xy_mut();
            match &mut self.pairs {
                PairStorage::Symmetric(pairs) => {
                    try_for_each_shuffled_pair(
                        pairs,
                        &mut self.block_order,
                        &mut self.rng,
                        |&p| {
                            apply_euclidean_sym_pair(xs, ys, eta, p);
                            Ok(())
                        },
                    )?;
                }
                PairStorage::General(pairs) => {
                    try_for_each_shuffled_pair(pairs, &mut self.block_order, &mut self.rng, |p| {
                        apply_euclidean_gen_pair(xs, ys, eta, p);
                        Ok(())
                    })?;
                }
            }
            let (xs, ys) = layout.xy_mut();
            self.run_streaming_negatives(eta, |e, p| {
                apply_euclidean_sym_pair(xs, ys, e, p);
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
