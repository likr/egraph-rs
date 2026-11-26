use egraph_core::{
    CoreError, CoreResult, DeterministicRng, IterativeSolver, LayoutNd, Scalar, SolverStatus,
};

use super::drgraph::step_drgraph_state_nd;
use super::graph_cne::step_graph_cne_state_nd;
use super::largevis::step_largevis_state_nd;
use super::nd_variants::{step_pacmap_state_nd, step_snap_t_fdp_state_nd};
use super::pairs::{
    HINGE_BIT, INDEX_MASK, PackedGenPair, PackedSymPair, PairStorage, try_for_each_shuffled_pair,
};
use super::sgd2::step_sgd2_state_nd;
use super::solver::SgdSolver;

#[inline]
pub(super) fn nd_norm<S: Scalar>(
    coords: &[S],
    base_i: usize,
    base_j: usize,
    dims: usize,
    d2: S,
) -> S {
    if d2 > S::ZERO && d2.is_finite() {
        return d2.sqrt();
    }
    if dims == 2 {
        return (coords[base_i] - coords[base_j]).hypot(coords[base_i + 1] - coords[base_j + 1]);
    }
    let mut max_abs = 0.0_f64;
    for a in 0..dims {
        let abs_d = (coords[base_i + a] - coords[base_j + a]).to_f64().abs();
        if !abs_d.is_finite() {
            return S::from_f64(f64::INFINITY);
        }
        if abs_d > max_abs {
            max_abs = abs_d;
        }
    }
    if max_abs <= 0.0 {
        return S::ZERO;
    }
    let mut sum_sq = 0.0_f64;
    for a in 0..dims {
        let ratio = (coords[base_i + a] - coords[base_j + a]).to_f64() / max_abs;
        sum_sq += ratio * ratio;
    }
    S::from_f64(max_abs * sum_sq.sqrt())
}

#[inline]
pub(super) fn apply_euclidean_sym_pair_nd<S: Scalar>(
    coords: &mut [S],
    dims: usize,
    eta: S,
    p: PackedSymPair<S>,
) {
    let i = p.i as usize;
    let j = (p.j_hinge & INDEX_MASK) as usize;
    let is_hinge = (p.j_hinge & HINGE_BIT) != 0;
    let mu = if p.w.is_finite() && p.w > S::ZERO {
        (eta * p.w).min(S::ONE)
    } else {
        S::ZERO
    };
    if !p.d.is_finite() || p.d <= S::ZERO || !mu.is_finite() || mu <= S::ZERO {
        return;
    }
    let base_i = i * dims;
    let base_j = j * dims;
    let mut d2 = S::ZERO;
    for a in 0..dims {
        let diff = coords[base_i + a] - coords[base_j + a];
        d2 += diff * diff;
    }
    if is_hinge && d2.is_finite() && d2 >= p.d * p.d {
        return;
    }
    let norm = nd_norm(coords, base_i, base_j, dims, d2);
    if !norm.is_finite() || norm <= S::ZERO {
        return;
    }
    if is_hinge && norm >= p.d {
        return;
    }
    let coeff = S::from_f64(0.5) * (norm - p.d) / norm * mu;
    for a in 0..dims {
        let diff = coords[base_i + a] - coords[base_j + a];
        let step = diff * coeff;
        coords[base_i + a] -= step;
        coords[base_j + a] += step;
    }
}

#[inline]
pub(super) fn apply_euclidean_gen_pair_nd<S: Scalar>(
    coords: &mut [S],
    dims: usize,
    eta: S,
    p: &PackedGenPair<S>,
) {
    if !p.dij.is_finite() || !p.dji.is_finite() || p.dij <= S::ZERO || p.dji <= S::ZERO {
        return;
    }
    let mu_i = if p.wij.is_finite() && p.wij > S::ZERO {
        (eta * p.wij).min(S::ONE)
    } else {
        S::ZERO
    };
    let mu_j = if p.wji.is_finite() && p.wji > S::ZERO {
        (eta * p.wji).min(S::ONE)
    } else {
        S::ZERO
    };
    if mu_i <= S::ZERO && mu_j <= S::ZERO {
        return;
    }
    let i = p.i as usize;
    let j = (p.j_hinge & INDEX_MASK) as usize;
    let is_hinge = (p.j_hinge & HINGE_BIT) != 0;
    let base_i = i * dims;
    let base_j = j * dims;
    let mut d2 = S::ZERO;
    for a in 0..dims {
        let diff = coords[base_i + a] - coords[base_j + a];
        d2 += diff * diff;
    }
    if is_hinge && d2.is_finite() && d2 >= p.dij * p.dij && d2 >= p.dji * p.dji {
        return;
    }
    let norm = nd_norm(coords, base_i, base_j, dims, d2);
    if !norm.is_finite() || norm <= S::ZERO {
        return;
    }
    let update_i = mu_i > S::ZERO && (!is_hinge || norm < p.dij);
    let update_j = mu_j > S::ZERO && (!is_hinge || norm < p.dji);
    if !update_i && !update_j {
        return;
    }
    let half = S::from_f64(0.5);
    let coeff_i = half * (norm - p.dij) / norm * mu_i;
    let coeff_j = half * (norm - p.dji) / norm * mu_j;
    for a in 0..dims {
        let diff = coords[base_i + a] - coords[base_j + a];
        if update_i {
            coords[base_i + a] -= diff * coeff_i;
        }
        if update_j {
            coords[base_j + a] += diff * coeff_j;
        }
    }
}

pub(super) fn scatter_collapsed_layout_nd_if_needed<S: Scalar>(
    layout: &mut LayoutNd<S>,
    span: f64,
    rng: &mut DeterministicRng,
) {
    let n = layout.num_nodes();
    if n <= 1 {
        return;
    }
    let dims = layout.dims();
    let coords = layout.coords();
    for a in 0..dims {
        let mut min_a = f64::INFINITY;
        let mut max_a = f64::NEG_INFINITY;
        for i in 0..n {
            let v = coords[i * dims + a].to_f64();
            min_a = min_a.min(v);
            max_a = max_a.max(v);
        }
        if (max_a - min_a) >= 1e-12 {
            return;
        }
    }
    let coords_mut = layout.coords_mut();
    for val in coords_mut.iter_mut() {
        *val = S::from_f64(span * (rng.next_f64() - 0.5));
    }
}

fn validate_nd_step_preconditions<S: Scalar>(
    solver: &SgdSolver<S>,
    layout: &LayoutNd<S>,
) -> CoreResult<usize> {
    let dims = layout.dims();
    if dims == 0 {
        return Err(CoreError::invalid_input("LayoutNd dims must be >= 1"));
    }
    if layout.num_nodes() != solver.num_nodes {
        return Err(CoreError::dimension_mismatch(
            solver.num_nodes,
            layout.num_nodes(),
            "SgdSolver::step",
        ));
    }
    layout.validate_finite()?;

    if let Some(st) = solver.pacmap.as_ref() {
        if let Some(sc) = st.nd_scratch.as_ref() {
            if sc.dims != dims {
                return Err(CoreError::dimension_mismatch(
                    sc.dims,
                    dims,
                    "SgdSolver::step::pacmap::dims",
                ));
            }
        } else if solver.iter > 0 {
            return Err(CoreError::invalid_input(
                "SgdSolver::step: cannot switch from Layout2d to LayoutNd on stateful PaCMAP solver",
            ));
        }
    }
    if let Some(st) = solver.sgd2.as_ref() {
        if let Some(sc) = st.nd_scratch.as_ref() {
            if sc.dims != dims {
                return Err(CoreError::dimension_mismatch(
                    sc.dims,
                    dims,
                    "SgdSolver::step::sgd2::dims",
                ));
            }
        } else if solver.iter > 0 {
            return Err(CoreError::invalid_input(
                "SgdSolver::step: cannot switch from Layout2d to LayoutNd on stateful Sgd2 solver",
            ));
        }
    }
    if let Some(st) = solver.drgraph.as_ref() {
        if let Some(sc) = st.nd_scratch.as_ref() {
            if sc.dims != dims {
                return Err(CoreError::dimension_mismatch(
                    sc.dims,
                    dims,
                    "SgdSolver::step::drgraph::dims",
                ));
            }
        } else if st.initialized_coords {
            return Err(CoreError::invalid_input(
                "SgdSolver::step: cannot switch from Layout2d to LayoutNd on stateful DRGraph solver",
            ));
        }
    }
    Ok(dims)
}

impl<S: Scalar> IterativeSolver<S, LayoutNd<S>> for SgdSolver<S> {
    fn step(&mut self, layout: &mut LayoutNd<S>) -> CoreResult<SolverStatus> {
        let dims = validate_nd_step_preconditions(self, layout)?;
        if self.status.is_finished() {
            return Ok(self.status);
        }

        if let Some(st) = self.snap_t_fdp.as_mut() {
            step_snap_t_fdp_state_nd(
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
            step_pacmap_state_nd(st, self.num_nodes, self.iter, layout);
            return Ok(self.finish_iteration());
        }
        if self.sgd2.is_some() {
            let eta_f64 = self.eta_at(self.iter);
            if let Some(st) = self.sgd2.as_mut() {
                step_sgd2_state_nd(
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
            step_largevis_state_nd(st, &mut self.rng, self.num_nodes, self.iter, layout);
            return Ok(self.finish_iteration());
        }
        if let Some(st) = self.drgraph.as_mut() {
            step_drgraph_state_nd(st, &mut self.rng, self.num_nodes, self.iter, layout);
            return Ok(self.finish_iteration());
        }
        if let Some(st) = self.graph_cne.as_mut() {
            step_graph_cne_state_nd(st, &mut self.rng, self.num_nodes, self.iter, layout);
            return Ok(self.finish_iteration());
        }

        let eta_f64 = self.eta_at(self.iter);
        if self.num_nodes > 1 && eta_f64.is_finite() && eta_f64 > 0.0 {
            let eta = S::from_f64(eta_f64);
            let coords = layout.coords_mut();
            match &mut self.pairs {
                PairStorage::Symmetric(pairs) => {
                    try_for_each_shuffled_pair(
                        pairs,
                        &mut self.block_order,
                        &mut self.rng,
                        |&p| {
                            apply_euclidean_sym_pair_nd(coords, dims, eta, p);
                            Ok(())
                        },
                    )?;
                }
                PairStorage::General(pairs) => {
                    try_for_each_shuffled_pair(pairs, &mut self.block_order, &mut self.rng, |p| {
                        apply_euclidean_gen_pair_nd(coords, dims, eta, p);
                        Ok(())
                    })?;
                }
            }
            let coords = layout.coords_mut();
            self.run_streaming_negatives(eta, |e, p| {
                apply_euclidean_sym_pair_nd(coords, dims, e, p);
                Ok(())
            })?;
        }

        Ok(self.finish_iteration())
    }

    fn status(&self) -> SolverStatus {
        self.status()
    }

    fn iteration(&self) -> usize {
        self.iteration()
    }

    fn total_iterations(&self) -> usize {
        self.total_iterations()
    }
}
