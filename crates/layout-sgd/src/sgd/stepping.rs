use std::any::Any;

use egraph_core::{
    CoreError, CoreResult, IterativeSolver, ManifoldKind, ManifoldLayout2d, RiemannianLayout,
    Scalar, SolverStatus,
};

use super::pairs::{
    HINGE_BIT, INDEX_MASK, PackedGenPair, PackedSymPair, PairStorage, try_for_each_shuffled_pair,
};
use super::solver::SgdSolver;

pub(super) fn validate_target_layout<L: RiemannianLayout + 'static>(
    expected_nodes: usize,
    locked_dim: Option<usize>,
    locked_manifold: Option<ManifoldKind>,
    target: &L,
    ctx: &'static str,
) -> CoreResult<()> {
    if target.len() != expected_nodes {
        return Err(CoreError::dimension_mismatch(
            expected_nodes,
            target.len(),
            ctx,
        ));
    }
    let dim = target.tangent_dim();
    if dim == 0 {
        return Err(CoreError::invalid_input(format!(
            "{ctx}: tangent_dim must be >= 1"
        )));
    }
    if let Some(prev_dim) = locked_dim
        && dim != prev_dim
    {
        return Err(CoreError::dimension_mismatch(prev_dim, dim, ctx));
    }
    if let Some(prev_kind) = locked_manifold
        && let Some(manifold) = (target as &dyn Any).downcast_ref::<ManifoldLayout2d<L::Scalar>>()
        && manifold.kind() != prev_kind
    {
        return Err(CoreError::invalid_input(format!(
            "{ctx}: cannot switch ManifoldKind on stateful solver"
        )));
    }
    target.validate_finite()
}

pub(super) fn lock_target_layout<L: RiemannianLayout + 'static>(
    locked_dim: &mut Option<usize>,
    locked_manifold: &mut Option<ManifoldKind>,
    target: &L,
) {
    if locked_dim.is_none() {
        *locked_dim = Some(target.tangent_dim());
    }
    if locked_manifold.is_none()
        && let Some(manifold) = (target as &dyn Any).downcast_ref::<ManifoldLayout2d<L::Scalar>>()
    {
        *locked_manifold = Some(manifold.kind());
    }
}

#[inline]
fn capped_mu(eta: f64, w: f64, max_mu: f64) -> f64 {
    if w.is_finite() && w > 0.0 {
        let raw = eta * w;
        if max_mu.is_finite() {
            raw.min(max_mu)
        } else {
            raw
        }
    } else {
        0.0
    }
}

#[inline]
fn apply_riemannian_sym_pair<L: RiemannianLayout>(
    layout: &mut L,
    eta: f64,
    max_mu: f64,
    p: PackedSymPair<L::Scalar>,
) -> CoreResult<()> {
    let d = p.d.to_f64();
    let mu = capped_mu(eta, p.w.to_f64(), max_mu);
    if !d.is_finite() || d <= 0.0 || !mu.is_finite() || mu <= 0.0 {
        return Ok(());
    }
    let i = p.i as usize;
    let j = (p.j_hinge & INDEX_MASK) as usize;
    let is_hinge = (p.j_hinge & HINGE_BIT) != 0;
    if is_hinge {
        let dist = layout.distance(i, j);
        if !dist.is_finite() {
            return Err(CoreError::numerical_breakdown(
                "non-finite distance during SGD step",
            ));
        }
        if dist >= d {
            return Ok(());
        }
    }
    let half_mu = 0.5 * mu;
    layout.move_pair_toward(i, j, d, half_mu, half_mu)
}

#[inline]
fn apply_riemannian_gen_pair<L: RiemannianLayout>(
    layout: &mut L,
    eta: f64,
    max_mu: f64,
    p: &PackedGenPair<L::Scalar>,
    tan_buf: &mut Vec<f64>,
) -> CoreResult<()> {
    let dij = p.dij.to_f64();
    let dji = p.dji.to_f64();
    if !dij.is_finite() || !dji.is_finite() || dij <= 0.0 || dji <= 0.0 {
        return Ok(());
    }
    let mu_i = capped_mu(eta, p.wij.to_f64(), max_mu);
    let mu_j = capped_mu(eta, p.wji.to_f64(), max_mu);
    if mu_i <= 0.0 && mu_j <= 0.0 {
        return Ok(());
    }
    let i = p.i as usize;
    let j = (p.j_hinge & INDEX_MASK) as usize;
    let is_hinge = (p.j_hinge & HINGE_BIT) != 0;

    if dij == dji {
        let dist = if is_hinge {
            let d = layout.distance(i, j);
            if !d.is_finite() {
                return Err(CoreError::numerical_breakdown(
                    "non-finite distance during SGD step",
                ));
            }
            d
        } else {
            0.0
        };
        let wi = if mu_i > 0.0 && (!is_hinge || dist < dij) {
            0.5 * mu_i
        } else {
            0.0
        };
        let wj = if mu_j > 0.0 && (!is_hinge || dist < dji) {
            0.5 * mu_j
        } else {
            0.0
        };
        if wi == 0.0 && wj == 0.0 {
            return Ok(());
        }
        return layout.move_pair_toward(i, j, dij, wi, wj);
    }

    let dist = layout.distance(i, j);
    if !dist.is_finite() {
        return Err(CoreError::numerical_breakdown(
            "non-finite distance during SGD step",
        ));
    }
    if dist <= 1e-15 {
        return Ok(());
    }
    let update_i = mu_i > 0.0 && (!is_hinge || dist < dij);
    let update_j = mu_j > 0.0 && (!is_hinge || dist < dji);
    if !update_i && !update_j {
        return Ok(());
    }

    let dim = layout.tangent_dim();
    if tan_buf.len() < 2 * dim {
        tan_buf.resize(2 * dim, 0.0);
    }
    let (ti, tj) = tan_buf.split_at_mut(dim);
    if update_i {
        layout.log_map_into(i, j, &mut ti[..dim])?;
    }
    if update_j {
        layout.log_map_into(j, i, &mut tj[..dim])?;
    }
    if update_i {
        let si = 0.5 * mu_i * (dist - dij) / dist;
        for v in &mut ti[..dim] {
            *v *= si;
        }
        layout.exp_map(i, &ti[..dim])?;
    }
    if update_j {
        let sj = 0.5 * mu_j * (dist - dji) / dist;
        for v in &mut tj[..dim] {
            *v *= sj;
        }
        layout.exp_map(j, &tj[..dim])?;
    }
    Ok(())
}

impl<L: RiemannianLayout + 'static> SgdSolver<L> {
    fn step_pairs_inner(&mut self, layout: &mut L) -> CoreResult<()> {
        let eta = self.eta_at(self.iter);
        if self.num_nodes > 1 && eta.is_finite() && eta > 0.0 {
            let max_mu = self.max_mu_f64();
            match &mut self.pairs {
                PairStorage::Symmetric(pairs) => {
                    try_for_each_shuffled_pair(
                        pairs,
                        &mut self.block_order,
                        &mut self.rng,
                        |&p| apply_riemannian_sym_pair(layout, eta, max_mu, p),
                    )?;
                }
                PairStorage::General(pairs) => {
                    let mut tan_buf = Vec::new();
                    try_for_each_shuffled_pair(pairs, &mut self.block_order, &mut self.rng, |p| {
                        apply_riemannian_gen_pair(layout, eta, max_mu, p, &mut tan_buf)
                    })?;
                }
            }
            self.run_streaming_negatives(|p| apply_riemannian_sym_pair(layout, eta, max_mu, p))?;
        }
        Ok(())
    }

    /// Executes `count` SGD iterations over schedule indices `start_iter..start_iter + count`
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

impl<L: RiemannianLayout + 'static> IterativeSolver for SgdSolver<L> {
    type Target = L;

    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        validate_target_layout(
            self.num_nodes,
            self.locked_dim,
            self.locked_manifold,
            target,
            "SgdSolver::validate_target",
        )
    }

    fn step(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        self.validate_target(target)?;
        if self.status.is_finished() {
            return Ok(self.status);
        }
        lock_target_layout(&mut self.locked_dim, &mut self.locked_manifold, target);

        if let Err(err) = self.step_pairs_inner(target) {
            self.status = SolverStatus::Failed;
            return Err(err);
        }
        if let Err(err) = target.validate_finite() {
            self.status = SolverStatus::Failed;
            return Err(err);
        }

        Ok(self.finish_iteration())
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
        self.total_iterations()
    }
}
