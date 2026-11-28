use std::any::Any;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, EuclideanLayout, Layout2d, Layout3d, LayoutNd, Scalar,
};

use super::t_fdp::TFdpState;
use super::t_fdp_step::{step_2d, step_nd};
use super::t_fdp_step3d::step_3d;

pub(super) type TFdpValidateFn<L> =
    fn(&TFdpState<<L as egraph_core::LayoutStorage>::Scalar>, &L) -> CoreResult<()>;
pub(super) type TFdpStepFn<L> =
    fn(&mut TFdpState<<L as egraph_core::LayoutStorage>::Scalar>, &mut L) -> CoreResult<()>;

pub(super) fn validate_2d<L: EuclideanLayout>(
    state: &TFdpState<L::Scalar>,
    target: &L,
) -> CoreResult<()> {
    if target.len() != state.num_nodes {
        return Err(CoreError::dimension_mismatch(
            state.num_nodes,
            target.len(),
            "TFdpSolver::validate_target",
        ));
    }
    target.validate_finite()
}

pub(super) fn validate_3d<L: EuclideanLayout>(
    state: &TFdpState<L::Scalar>,
    target: &L,
) -> CoreResult<()> {
    if target.len() != state.num_nodes {
        return Err(CoreError::dimension_mismatch(
            state.num_nodes,
            target.len(),
            "TFdpSolver::validate_target",
        ));
    }
    target.validate_finite()
}

pub(super) fn validate_nd<L: EuclideanLayout>(
    state: &TFdpState<L::Scalar>,
    target: &L,
) -> CoreResult<()> {
    if target.len() != state.num_nodes {
        return Err(CoreError::dimension_mismatch(
            state.num_nodes,
            target.len(),
            "TFdpSolver::validate_target",
        ));
    }
    let dims = target.dims();
    if dims == 0 {
        return Err(CoreError::invalid_input("LayoutNd dims must be >= 1"));
    }
    if let Some(prev) = state.locked_nd_dims
        && prev != dims
    {
        return Err(CoreError::dimension_mismatch(
            prev,
            dims,
            "TFdpSolver::validate_target::dims",
        ));
    }
    target.validate_finite()
}

pub(super) fn step_any_2d<L: EuclideanLayout + 'static>(
    state: &mut TFdpState<L::Scalar>,
    target: &mut L,
) -> CoreResult<()> {
    let layout = (target as &mut dyn Any)
        .downcast_mut::<Layout2d<L::Scalar>>()
        .ok_or_else(|| CoreError::numerical_breakdown("TFdpSolver target type mismatch"))?;
    step_2d(state, layout)
}

pub(super) fn step_any_3d<L: EuclideanLayout + 'static>(
    state: &mut TFdpState<L::Scalar>,
    target: &mut L,
) -> CoreResult<()> {
    let layout = (target as &mut dyn Any)
        .downcast_mut::<Layout3d<L::Scalar>>()
        .ok_or_else(|| CoreError::numerical_breakdown("TFdpSolver target type mismatch"))?;
    step_3d(state, layout)
}

pub(super) fn step_any_nd<L: EuclideanLayout + 'static>(
    state: &mut TFdpState<L::Scalar>,
    target: &mut L,
) -> CoreResult<()> {
    let layout = (target as &mut dyn Any)
        .downcast_mut::<LayoutNd<L::Scalar>>()
        .ok_or_else(|| CoreError::numerical_breakdown("TFdpSolver target type mismatch"))?;
    step_nd(state, layout)
}

pub(super) fn extract_symmetric_csr_edges<W: Scalar>(
    graph: &CsrGraph<W>,
) -> CoreResult<(Vec<u32>, Vec<u32>, Vec<f32>)> {
    let n = graph.num_nodes();
    if n == 0 {
        return Ok((vec![0], Vec::new(), Vec::new()));
    }
    let mut pairs: Vec<(u32, u32, f32)> = Vec::with_capacity(if graph.is_directed() {
        graph.nnz() * 2
    } else {
        graph.nnz()
    });
    for u in 0..n {
        let u_u32 = u as u32;
        let nbrs = graph.neighbors(u);
        let w_opt = graph.neighbor_weights(u);
        for (idx, &v) in nbrs.iter().enumerate() {
            let v_usize = v as usize;
            if v_usize == u || v_usize >= n {
                continue;
            }
            let w = w_opt.map_or(1.0, |ws| ws[idx].to_f64());
            if w <= 0.0 {
                continue;
            }
            let w_f32 = w as f32;
            pairs.push((u_u32, v, w_f32));
            if graph.is_directed() {
                pairs.push((v, u_u32, w_f32));
            }
        }
    }
    symmetrize_pair_list(n, pairs)
}

pub(super) fn symmetrize_pair_list(
    n: usize,
    mut pairs: Vec<(u32, u32, f32)>,
) -> CoreResult<(Vec<u32>, Vec<u32>, Vec<f32>)> {
    if pairs.is_empty() {
        return Ok((vec![0; n + 1], Vec::new(), Vec::new()));
    }
    pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let mut deg = vec![0usize; n];
    let mut write_idx = 0usize;
    let mut pos = 0usize;
    let len = pairs.len();
    while pos < len {
        let (u, v, first_w) = pairs[pos];
        let mut sum_w = first_w as f64;
        let mut count = 1usize;
        pos += 1;
        while pos < len && pairs[pos].0 == u && pairs[pos].1 == v {
            sum_w += pairs[pos].2 as f64;
            count += 1;
            pos += 1;
        }
        let avg_w = (sum_w / (count as f64)) as f32;
        if avg_w > 0.0 {
            pairs[write_idx] = (u, v, avg_w);
            write_idx += 1;
            deg[u as usize] += 1;
        }
    }
    pairs.truncate(write_idx);

    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0u32);
    let mut total = 0usize;
    for &d in &deg {
        total += d;
        let ptr = u32::try_from(total)
            .map_err(|_| CoreError::invalid_csr("TFdpSolver edge nnz exceeds u32::MAX"))?;
        indptr.push(ptr);
    }
    drop(deg);
    let mut cursor: Vec<usize> = indptr[..n].iter().map(|&p| p as usize).collect();
    let mut indices = vec![0u32; total];
    let mut weights = vec![0.0_f32; total];
    for (u, v, w) in pairs {
        let p = cursor[u as usize];
        indices[p] = v;
        weights[p] = w;
        cursor[u as usize] += 1;
    }
    Ok((indptr, indices, weights))
}
