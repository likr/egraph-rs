use egraph_core::{CoreError, CoreResult, LayoutNd, Scalar};
use egraph_spatial_trees::{DynamicKdTree, KdQueryScratch};

use super::validate_layout_nd_labels;

#[derive(Debug, Clone, Copy)]
pub(super) enum ProximityModeNd {
    Gabriel,
    RelativeNeighborhood,
}

#[inline]
pub(super) fn dist_sq_slice(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    for axis in 0..a.len() {
        let diff = a[axis] - b[axis];
        sum += diff * diff;
    }
    sum
}

#[inline]
pub(super) fn euclidean_dist_nd(a: &[f64], b: &[f64]) -> f64 {
    if a.len() == 2 {
        return (a[0] - b[0]).hypot(a[1] - b[1]);
    }
    let mut max_abs = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for axis in 0..a.len() {
        let diff = (a[axis] - b[axis]).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        sum_sq += diff * diff;
    }
    if max_abs == 0.0 {
        return 0.0;
    }
    // Rescale by max_abs when squaring overflows or underflows into zero (subnormal layouts).
    if sum_sq == 0.0 || !sum_sq.is_finite() {
        let inv = 1.0 / max_abs;
        let mut scaled_sq = 0.0_f64;
        for axis in 0..a.len() {
            let s = (a[axis] - b[axis]) * inv;
            scaled_sq += s * s;
        }
        return max_abs * scaled_sq.sqrt();
    }
    sum_sq.sqrt()
}

#[inline]
pub(super) fn validate_tree_layout_nd<S: Scalar>(
    tree: &DynamicKdTree<f64>,
    layout: &LayoutNd<S>,
    context: &'static str,
) -> CoreResult<()> {
    if tree.len() != layout.num_nodes() {
        return Err(CoreError::dimension_mismatch(
            layout.num_nodes(),
            tree.len(),
            context,
        ));
    }
    if tree.dims() != layout.dims() {
        return Err(CoreError::dimension_mismatch(
            layout.dims(),
            tree.dims(),
            context,
        ));
    }
    Ok(())
}

/// Builds a `f64` [`DynamicKdTree`] from an arbitrary-dimensional [`LayoutNd`].
#[doc(hidden)]
pub fn build_dynamic_kdtree_f64<S: Scalar>(layout: &LayoutNd<S>) -> CoreResult<DynamicKdTree<f64>> {
    layout.validate_finite()?;
    let coords: Vec<f64> = layout.coords().iter().map(|v| v.to_f64()).collect();
    DynamicKdTree::<f64>::from_flat(&coords, layout.num_nodes(), layout.dims())
}

#[inline]
pub(super) fn validate_nd_labels_and_tree<S: Scalar>(
    layout: &LayoutNd<S>,
    labels: &[u32],
    tree: &DynamicKdTree<f64>,
    context: &'static str,
) -> CoreResult<()> {
    validate_layout_nd_labels(layout, labels, context)?;
    validate_tree_layout_nd(tree, layout, context)
}

fn blocked_by_local_witnesses(
    witnesses: &[(usize, f64)],
    other_idx: usize,
    other_pt: &[f64],
    coords: &[f64],
    dims: usize,
    d_ij_sq: f64,
    mode: ProximityModeNd,
) -> bool {
    let thresh = match mode {
        ProximityModeNd::Gabriel => d_ij_sq * (1.0 + 1e-12),
        ProximityModeNd::RelativeNeighborhood => d_ij_sq * (1.0 - 1e-12),
    };
    for &(r, d_self_r_sq) in witnesses {
        match mode {
            ProximityModeNd::Gabriel => {
                if d_self_r_sq > thresh {
                    break;
                }
                if r == other_idx {
                    continue;
                }
                let pr = &coords[r * dims..(r + 1) * dims];
                if d_self_r_sq + dist_sq_slice(other_pt, pr) <= thresh {
                    return true;
                }
            }
            ProximityModeNd::RelativeNeighborhood => {
                if d_self_r_sq >= thresh {
                    break;
                }
                if r == other_idx {
                    continue;
                }
                let pr = &coords[r * dims..(r + 1) * dims];
                if dist_sq_slice(other_pt, pr) < thresh {
                    return true;
                }
            }
        }
    }
    false
}

pub(super) fn proximity_edges_nd_with_tree(
    tree: &DynamicKdTree<f64>,
    mode: ProximityModeNd,
) -> CoreResult<Vec<(u32, u32)>> {
    let n = tree.len();
    let d = tree.dims();
    if n < 2 {
        return Ok(Vec::new());
    }

    let coords = tree.coords_by_id();
    let w_cap = 16usize.min(n - 1);
    let mut scratch = KdQueryScratch::with_capacity(w_cap);
    let mut local_witnesses = vec![(0usize, 0.0_f64); n * w_cap];
    let mut knn_buf = Vec::with_capacity(w_cap);
    for i in 0..n {
        tree.k_nearest_excluding_self_with_scratch(i, w_cap, &mut scratch, &mut knn_buf)?;
        local_witnesses[i * w_cap..(i + 1) * w_cap].copy_from_slice(&knn_buf[..w_cap]);
    }

    let mut edges = Vec::new();
    let mut mid = vec![0.0_f64; d];
    let mut witness_buf = Vec::new();
    let d_sqrt = (d as f64).sqrt();

    for i in 0..n {
        let pi = &coords[i * d..(i + 1) * d];
        let wi = &local_witnesses[i * w_cap..(i + 1) * w_cap];
        for j in (i + 1)..n {
            let pj = &coords[j * d..(j + 1) * d];
            let d_ij_sq = dist_sq_slice(pi, pj);
            if d_ij_sq <= 0.0 {
                continue;
            }
            if blocked_by_local_witnesses(wi, j, pj, coords, d, d_ij_sq, mode) {
                continue;
            }
            let wj = &local_witnesses[j * w_cap..(j + 1) * w_cap];
            if blocked_by_local_witnesses(wj, i, pi, coords, d, d_ij_sq, mode) {
                continue;
            }

            let mut max_coord_abs = 0.0_f64;
            for axis in 0..d {
                mid[axis] = pi[axis] + 0.5 * (pj[axis] - pi[axis]);
                max_coord_abs = max_coord_abs.max(pi[axis].abs()).max(pj[axis].abs());
            }
            let base_r_sq = match mode {
                ProximityModeNd::Gabriel => 0.25 * d_ij_sq * (1.0 + 1e-11),
                ProximityModeNd::RelativeNeighborhood => 0.75 * d_ij_sq * (1.0 + 1e-11),
            };
            let eps_mid = 4.0 * d_sqrt * f64::EPSILON * max_coord_abs;
            let padded_r = base_r_sq.sqrt() + eps_mid;
            let query_r_sq = (padded_r * padded_r).min(f64::MAX);
            tree.within_radius_sq_with_scratch(&mid, query_r_sq, &mut scratch, &mut witness_buf)?;

            let mut blocked = false;
            for &(r, _) in &witness_buf {
                if r == i || r == j {
                    continue;
                }
                let pr = &coords[r * d..(r + 1) * d];
                let d_ir_sq = dist_sq_slice(pi, pr);
                let d_jr_sq = dist_sq_slice(pj, pr);
                blocked = match mode {
                    ProximityModeNd::Gabriel => d_ir_sq + d_jr_sq <= d_ij_sq * (1.0 + 1e-12),
                    ProximityModeNd::RelativeNeighborhood => {
                        let thresh = d_ij_sq * (1.0 - 1e-12);
                        d_ir_sq < thresh && d_jr_sq < thresh
                    }
                };
                if blocked {
                    break;
                }
            }
            if !blocked {
                edges.push((i as u32, j as u32));
            }
        }
    }
    Ok(edges)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn is_gong_directed_edge_nd(
    i: usize,
    j: usize,
    gamma: f64,
    coords: &[f64],
    dims: usize,
    tree: &DynamicKdTree<f64>,
    mid_buf: &mut [f64],
    scratch: &mut KdQueryScratch,
    witness_buf: &mut Vec<(usize, f64)>,
) -> CoreResult<bool> {
    let pi = &coords[i * dims..(i + 1) * dims];
    let pj = &coords[j * dims..(j + 1) * dims];
    let d_ij_sq = dist_sq_slice(pi, pj);
    if d_ij_sq <= 0.0 {
        return Ok(false);
    }
    let one_minus_g = 1.0 - gamma;
    for axis in 0..dims {
        mid_buf[axis] = gamma * pj[axis] + one_minus_g * pi[axis];
    }
    let r_sq = one_minus_g * one_minus_g * d_ij_sq;
    let query_r_sq = (r_sq * (1.0 + 1e-11)).min(f64::MAX);
    tree.within_radius_sq_with_scratch(mid_buf, query_r_sq, scratch, witness_buf)?;

    let block_thresh = r_sq * (1.0 + 1e-12);
    for &(q, d_mq_sq) in witness_buf.iter() {
        if q == i || q == j {
            continue;
        }
        if d_mq_sq <= block_thresh {
            return Ok(false);
        }
    }
    Ok(true)
}
