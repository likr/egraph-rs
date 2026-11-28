use egraph_core::{
    CoreError, CoreResult, EuclideanLayout, EuclideanStorageMut, EuclideanStorageRef,
    RepulsionBackend, Scalar, SolverStatus,
};

use super::graph_prep::{is_layout_collapsed_euclidean, norm2_slice};
use super::init::initialize_collapsed_euclidean;
use super::solver::MaxentCoreState;

const MIN_NORM_SQ_SUM: f64 = 1e-12;

pub(super) fn validate_euclidean_target<L: EuclideanLayout>(
    core: &MaxentCoreState,
    target: &L,
) -> CoreResult<()> {
    if target.len() != core.num_nodes {
        return Err(CoreError::dimension_mismatch(
            core.num_nodes,
            target.len(),
            "MaxentStressSolver::validate_target",
        ));
    }
    let d = target.dims();
    if d == 0 {
        return Err(CoreError::invalid_input("Layout dims must be >= 1"));
    }
    if let Some(prev) = core.locked_dims
        && prev != d
    {
        return Err(CoreError::dimension_mismatch(
            prev,
            d,
            "MaxentStressSolver::validate_target::dims",
        ));
    }
    target.validate_finite()
}

fn ensure_buffers(core: &mut MaxentCoreState, n: usize, d: usize) -> CoreResult<()> {
    let needed = n
        .checked_mul(d)
        .ok_or_else(|| CoreError::invalid_input("Maxent coordinate buffer size overflow"))?;
    let comp_needed = core
        .comp_sizes
        .len()
        .checked_mul(d)
        .ok_or_else(|| CoreError::invalid_input("Maxent component sum buffer size overflow"))?;
    if core.cur.len() != needed {
        core.cur.resize(needed, 0.0);
        core.next.resize(needed, 0.0);
        core.stress_rhs.resize(needed, 0.0);
        core.rep.resize(needed, 0.0);
    }
    if core.comp_sum.len() != comp_needed {
        core.comp_sum.resize(comp_needed, 0.0);
    }
    if core.rhs_axis.len() != n {
        core.rhs_axis.resize(n, 0.0);
        core.next_axis.resize(n, 0.0);
    }
    Ok(())
}

#[inline]
fn euclid_sq_diff_flat(coords: &[f64], d: usize, u: usize, v: usize) -> f64 {
    let base_u = u * d;
    let base_v = v * d;
    let mut sum = 0.0_f64;
    for a in 0..d {
        let diff = coords[base_u + a] - coords[base_v + a];
        sum += diff * diff;
    }
    sum
}

#[inline]
fn euclid_norm_diff_flat(coords: &[f64], d: usize, u: usize, v: usize) -> f64 {
    let sum_sq = euclid_sq_diff_flat(coords, d, u, v);
    if sum_sq.is_finite() {
        return sum_sq.sqrt();
    }
    let base_u = u * d;
    let base_v = v * d;
    let mut max_abs = 0.0_f64;
    for a in 0..d {
        let av = (coords[base_u + a] - coords[base_v + a]).abs();
        if av > max_abs {
            max_abs = av;
        }
    }
    if !max_abs.is_finite() || max_abs == 0.0 {
        return max_abs;
    }
    let mut scaled_sq = 0.0_f64;
    for a in 0..d {
        let r = (coords[base_u + a] - coords[base_v + a]) / max_abs;
        scaled_sq += r * r;
    }
    max_abs * scaled_sq.sqrt()
}

fn run_maxent_step<L: EuclideanLayout, E: RepulsionBackend<L>>(
    core: &mut MaxentCoreState,
    evaluator: &mut E,
    staged: Option<&L>,
    layout: &mut L,
) -> CoreResult<()> {
    let n = core.num_nodes;
    let src_layout = staged.unwrap_or(&*layout);
    let d = src_layout.dims();
    ensure_buffers(core, n, d)?;

    match src_layout.as_euclidean_storage() {
        EuclideanStorageRef::Soa2d { x, y } => {
            for i in 0..n {
                core.cur[i * 2] = x[i].to_f64();
                core.cur[i * 2 + 1] = y[i].to_f64();
            }
        }
        EuclideanStorageRef::Soa3d { x, y, z } => {
            for i in 0..n {
                core.cur[i * 3] = x[i].to_f64();
                core.cur[i * 3 + 1] = y[i].to_f64();
                core.cur[i * 3 + 2] = z[i].to_f64();
            }
        }
        EuclideanStorageRef::FlatNd { coords, .. } => {
            for (dst, &src) in core.cur.iter_mut().zip(coords.iter()) {
                *dst = src.to_f64();
            }
        }
    }

    // 1. Majorizing stress RHS: L_{w,d}(x^{(t)}) x^{(t)}.
    core.stress_rhs.fill(0.0);
    for edge in &core.edges {
        let u = edge.u as usize;
        let v = edge.v as usize;
        let norm = euclid_norm_diff_flat(&core.cur, d, u, v);
        if norm > 1e-12 && norm.is_finite() {
            let f = edge.w * edge.d / norm;
            let base_u = u * d;
            let base_v = v * d;
            for a in 0..d {
                let fa = f * (core.cur[base_u + a] - core.cur[base_v + a]);
                core.stress_rhs[base_u + a] += fa;
                core.stress_rhs[base_v + a] -= fa;
            }
        }
    }

    // 2. Repulsive force b(x) over non-adjacent vertex pairs {i, j} \notin S.
    let mut rep_coeff = 0.0_f64;
    if core.alpha > 0.0 {
        evaluator.accumulate_repulsion(staged.unwrap_or(&*layout), &mut core.rep)?;

        let eps_sq = core.config.eps_sq;
        let q = core.effective_q;
        let is_q_zero = q == 0.0;
        let exp = 0.5 * (q + 2.0);

        // Subtract 1-hop S-edge pairs so b(x) sums strictly over {i, j} \notin S.
        for edge in &core.edges {
            let u = edge.u as usize;
            let v = edge.v as usize;
            let dist_sq = euclid_sq_diff_flat(&core.cur, d, u, v);
            let d2 = dist_sq + eps_sq;
            let mult = if is_q_zero {
                1.0 / d2
            } else {
                1.0 / d2.powf(exp)
            };
            let base_u = u * d;
            let base_v = v * d;
            for a in 0..d {
                let fa = mult * (core.cur[base_u + a] - core.cur[base_v + a]);
                core.rep[base_u + a] -= fa;
                core.rep[base_v + a] += fa;
            }
        }

        // Zero isolated vertices and center b(x) per connected component so
        // sum_{i in C_k} b_i == 0 lies strictly in range(L_w).
        core.comp_sum.fill(0.0);
        for i in 0..n {
            let cid = core.comp_id[i] as usize;
            let base_i = i * d;
            if core.comp_sizes[cid] <= 1 {
                core.rep[base_i..base_i + d].fill(0.0);
            } else {
                let base_c = cid * d;
                for a in 0..d {
                    core.comp_sum[base_c + a] += core.rep[base_i + a];
                }
            }
        }
        for i in 0..n {
            let cid = core.comp_id[i] as usize;
            let sz = core.comp_sizes[cid];
            if sz > 1 {
                let inv_sz = 1.0 / (sz as f64);
                let base_i = i * d;
                let base_c = cid * d;
                for a in 0..d {
                    core.rep[base_i + a] -= core.comp_sum[base_c + a] * inv_sz;
                }
            }
        }

        let norm_stress = norm2_slice(&core.stress_rhs);
        let norm_rep = norm2_slice(&core.rep);
        if norm_rep > 1e-30 {
            let scale = if norm_stress > 0.0 {
                norm_stress / norm_rep
            } else {
                1.0
            };
            rep_coeff = core.alpha * scale;
        }
    }

    // 3. Solve L_w x_a^{(t+1)} = rhs_a warm-started from x_a^{(t)} per coordinate axis.
    for a in 0..d {
        if rep_coeff != 0.0 {
            for i in 0..n {
                let idx = i * d + a;
                core.rhs_axis[i] = core.stress_rhs[idx] + rep_coeff * core.rep[idx];
                core.next_axis[i] = core.cur[idx];
            }
        } else {
            for i in 0..n {
                let idx = i * d + a;
                core.rhs_axis[i] = core.stress_rhs[idx];
                core.next_axis[i] = core.cur[idx];
            }
        }

        core.amg
            .solve(&core.rhs_axis, &mut core.next_axis, &mut core.amg_ws)?;

        for i in 0..n {
            core.next[i * d + a] = core.next_axis[i];
        }
    }

    for i in 0..n {
        let base = i * d;
        let is_isolated = core.comp_sizes[core.comp_id[i] as usize] <= 1;
        let all_finite = !is_isolated && core.next[base..base + d].iter().all(|v| v.is_finite());
        if !all_finite {
            core.next[base..base + d].copy_from_slice(&core.cur[base..base + d]);
        }
    }

    let mut diff_sq = 0.0_f64;
    let mut cur_sq = 0.0_f64;
    for (&next_v, &cur_v) in core.next.iter().zip(core.cur.iter()) {
        let diff = next_v - cur_v;
        diff_sq += diff * diff;
        cur_sq += cur_v * cur_v;
    }
    let rel_change = diff_sq.sqrt() / cur_sq.sqrt().max(MIN_NORM_SQ_SUM);

    match layout.as_euclidean_storage_mut() {
        EuclideanStorageMut::Soa2d { x, y } => {
            for i in 0..n {
                let nx = L::Scalar::from_f64(core.next[i * 2]);
                let ny = L::Scalar::from_f64(core.next[i * 2 + 1]);
                if nx.is_finite() {
                    x[i] = nx;
                }
                if ny.is_finite() {
                    y[i] = ny;
                }
            }
        }
        EuclideanStorageMut::Soa3d { x, y, z } => {
            for i in 0..n {
                let nx = L::Scalar::from_f64(core.next[i * 3]);
                let ny = L::Scalar::from_f64(core.next[i * 3 + 1]);
                let nz = L::Scalar::from_f64(core.next[i * 3 + 2]);
                if nx.is_finite() {
                    x[i] = nx;
                }
                if ny.is_finite() {
                    y[i] = ny;
                }
                if nz.is_finite() {
                    z[i] = nz;
                }
            }
        }
        EuclideanStorageMut::FlatNd { coords, .. } => {
            for (dst, &val) in coords.iter_mut().zip(core.next.iter()) {
                let sv = L::Scalar::from_f64(val);
                if sv.is_finite() {
                    *dst = sv;
                }
            }
        }
    }

    core.stage_step += 1;
    core.total_iter_executed += 1;

    let stage_converged = rel_change < core.config.tol;
    if stage_converged || core.stage_step >= core.config.steps_per_stage {
        core.stage_idx += 1;
        core.stage_step = 0;
        if core.stage_idx >= core.config.stages {
            core.status = if stage_converged {
                SolverStatus::Converged
            } else {
                SolverStatus::Completed
            };
        } else {
            core.alpha = (core.alpha * core.config.alpha_decay).max(core.config.alpha_min);
            core.status = SolverStatus::Running;
        }
    } else {
        core.status = SolverStatus::Running;
    }

    Ok(())
}

pub(super) fn step_euclidean<L: EuclideanLayout, E: RepulsionBackend<L>>(
    core: &mut MaxentCoreState,
    evaluator: &mut E,
    target: &mut L,
) -> CoreResult<()> {
    let d = target.dims();
    if core.num_nodes <= 1 {
        if core.locked_dims.is_none() {
            core.locked_dims = Some(d);
        }
        core.status = SolverStatus::Converged;
        return Ok(());
    }
    if core.total_iter_executed == 0 && is_layout_collapsed_euclidean(target) {
        let mut staged = target.zeros_like(core.num_nodes)?;
        initialize_collapsed_euclidean(core, &mut staged)?;
        run_maxent_step(core, evaluator, Some(&staged), target)?;
    } else {
        run_maxent_step(core, evaluator, None, target)?;
    }
    if core.locked_dims.is_none() {
        core.locked_dims = Some(d);
    }
    Ok(())
}
