use egraph_core::{DeterministicRng, LayoutNd, Scalar};

use super::pacmap::PacmapState;
use super::pairs::{BLOCK_SHUFFLE_THRESHOLD, BLOCK_SIZE};
use super::snap_t_fdp::SnapTFdpState;

#[derive(Debug, Clone)]
pub(super) struct PacmapNdScratch {
    grad: Vec<f64>,
    m: Vec<f64>,
    v: Vec<f64>,
}

#[derive(Debug, Clone)]
pub(super) struct Sgd2NdScratch {
    pub(super) grad: Vec<f64>,
    pub(super) h: Vec<f64>,
}

impl Sgd2NdScratch {
    pub(super) fn new(total_len: usize) -> Self {
        Self {
            grad: vec![0.0; total_len],
            h: vec![0.0; total_len],
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct DrGraphNdScratch {
    pub(super) active: Vec<f64>,
    pub(super) scratch: Vec<f64>,
    pub(super) scratch_cnt: Vec<f64>,
}

impl DrGraphNdScratch {
    pub(super) fn new(num_nodes: usize, total_len: usize) -> Self {
        Self {
            active: vec![0.0; total_len],
            scratch: vec![0.0; total_len],
            scratch_cnt: vec![0.0; num_nodes],
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct GraphCneNdScratch {
    pub(super) neg_w: Vec<f64>,
    pub(super) acc_rep: Vec<f64>,
}

impl GraphCneNdScratch {
    pub(super) fn new(neg_samples: usize, dims: usize) -> Self {
        Self {
            neg_w: Vec::with_capacity(neg_samples),
            acc_rep: vec![0.0; dims],
        }
    }
}

#[inline]
pub(super) fn nd_dist_sq_s<S: Scalar>(
    coords: &[S],
    base_a: usize,
    base_b: usize,
    dims: usize,
) -> S {
    let mut d2 = S::ZERO;
    for a in 0..dims {
        let diff = coords[base_a + a] - coords[base_b + a];
        d2 += diff * diff;
    }
    d2
}

#[inline]
pub(super) fn nd_dist_sq_f64<S: Scalar>(
    coords: &[S],
    base_a: usize,
    base_b: usize,
    dims: usize,
) -> f64 {
    let mut d2 = 0.0_f64;
    for a in 0..dims {
        let diff = coords[base_a + a].to_f64() - coords[base_b + a].to_f64();
        d2 += diff * diff;
    }
    d2
}

#[inline]
pub(super) fn nd_norm_f64<S: Scalar>(
    coords: &[S],
    base_i: usize,
    base_j: usize,
    dims: usize,
) -> f64 {
    if dims == 2 {
        let dx = coords[base_i].to_f64() - coords[base_j].to_f64();
        let dy = coords[base_i + 1].to_f64() - coords[base_j + 1].to_f64();
        return dx.hypot(dy);
    }
    let mut max_abs = 0.0_f64;
    for a in 0..dims {
        let abs_d = (coords[base_i + a].to_f64() - coords[base_j + a].to_f64()).abs();
        if !abs_d.is_finite() {
            return f64::INFINITY;
        }
        if abs_d > max_abs {
            max_abs = abs_d;
        }
    }
    if max_abs <= 0.0 {
        return 0.0;
    }
    let mut sum_sq = 0.0_f64;
    for a in 0..dims {
        let ratio = (coords[base_i + a].to_f64() - coords[base_j + a].to_f64()) / max_abs;
        sum_sq += ratio * ratio;
    }
    max_abs * sum_sq.sqrt()
}

pub(super) fn step_snap_t_fdp_state_nd<S: Scalar>(
    st: &mut SnapTFdpState,
    block_order: &mut [u32],
    rng: &mut DeterministicRng,
    num_nodes: usize,
    iter: usize,
    layout: &mut LayoutNd<S>,
) {
    let total = st.config.iterations;
    if num_nodes <= 1 || st.edges_u.is_empty() || total == 0 || iter >= total {
        return;
    }
    let eta_f64 = st.config.learning_rate * (1.0 - (iter as f64) / (total as f64));
    if !eta_f64.is_finite() || eta_f64 <= 0.0 {
        return;
    }
    let progress = if total <= 1 {
        0.0
    } else {
        (iter as f64) / ((total - 1) as f64)
    };
    let rho = st.config.rho_start
        + (st.config.rho_end - st.config.rho_start) * progress.powf(st.config.rho_power);
    let base_attr = S::from_f64(eta_f64 * 0.1 * st.config.alpha);
    let base_rep = S::from_f64(eta_f64 * rho * st.config.repulsion_scale);
    let k_neg = st.config.negatives_per_edge;
    let bounded_attraction = st.config.bounded_attraction;
    let dims = layout.dims();
    let coords = layout.coords_mut();
    let len = st.edge_order.len();

    if len <= BLOCK_SHUFFLE_THRESHOLD {
        rng.shuffle(&mut st.edge_order);
        apply_snap_edge_slice_nd(
            coords,
            dims,
            &st.edge_order,
            &st.edges_u,
            &st.edges_v,
            st.edge_weights.as_deref(),
            st.deg_scales.as_deref(),
            bounded_attraction,
            num_nodes,
            k_neg,
            base_attr,
            base_rep,
            rng,
        );
    } else {
        rng.shuffle(block_order);
        for &b in block_order.iter() {
            let start = (b as usize) * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(len);
            let chunk = &mut st.edge_order[start..end];
            rng.shuffle(chunk);
            apply_snap_edge_slice_nd(
                coords,
                dims,
                chunk,
                &st.edges_u,
                &st.edges_v,
                st.edge_weights.as_deref(),
                st.deg_scales.as_deref(),
                bounded_attraction,
                num_nodes,
                k_neg,
                base_attr,
                base_rep,
                rng,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn apply_snap_edge_slice_nd<S: Scalar>(
    coords: &mut [S],
    dims: usize,
    order_slice: &[u32],
    edges_u: &[u32],
    edges_v: &[u32],
    edge_weights: Option<&[f32]>,
    deg_scales: Option<&[f32]>,
    bounded_attraction: bool,
    num_nodes: usize,
    negatives_per_edge: usize,
    base_attr: S,
    base_rep: S,
    rng: &mut DeterministicRng,
) {
    let eight = S::from_f64(8.0);
    let max_attr = S::from_f64(0.49);
    let max_rep = S::from_f64(0.5);
    for (pos, &edge_idx) in order_slice.iter().enumerate() {
        let idx = edge_idx as usize;
        let u = edges_u[idx] as usize;
        let v = edges_v[idx] as usize;
        let w_e = edge_weights.map_or(S::ONE, |ws| S::from_f32(ws[idx]));
        let deg_factor = deg_scales.map_or(S::ONE, |ds| S::from_f32(ds[u]) * S::from_f32(ds[v]));

        if w_e > S::ZERO && base_attr > S::ZERO && deg_factor > S::ZERO {
            let (bu, bv) = (u * dims, v * dims);
            let d2 = nd_dist_sq_s(coords, bu, bv, dims);
            let inv_1_d2 = if d2.is_finite() {
                S::ONE / (S::ONE + d2)
            } else {
                S::ZERO
            };
            let kernel = if bounded_attraction {
                eight * inv_1_d2
            } else {
                S::ONE + eight * inv_1_d2
            };
            let attr = base_attr * w_e * deg_factor * kernel;
            if attr.is_finite() && attr > S::ZERO {
                let coeff = attr.min(max_attr);
                let ok = (0..dims).all(|a| ((coords[bu + a] - coords[bv + a]) * coeff).is_finite());
                if ok {
                    for a in 0..dims {
                        let step = (coords[bu + a] - coords[bv + a]) * coeff;
                        coords[bu + a] -= step;
                        coords[bv + a] += step;
                    }
                }
            }
        }

        if base_rep > S::ZERO {
            for neg_idx in 0..negatives_per_edge {
                let anchor = if ((pos + neg_idx) & 1) == 0 { u } else { v };
                let w = rng.gen_range_usize(num_nodes);
                if w == anchor {
                    continue;
                }
                let (ba, bw) = (anchor * dims, w * dims);
                let rd2 = nd_dist_sq_s(coords, ba, bw, dims);
                if rd2.is_finite() {
                    let denom = (S::ONE + rd2) * (S::ONE + rd2);
                    if denom.is_finite() && denom > S::ZERO {
                        let rep = (base_rep / denom).min(max_rep);
                        if rep.is_finite() && rep > S::ZERO {
                            let ok = (0..dims)
                                .all(|a| ((coords[ba + a] - coords[bw + a]) * rep).is_finite());
                            if ok {
                                for a in 0..dims {
                                    let step = (coords[ba + a] - coords[bw + a]) * rep;
                                    coords[ba + a] += step;
                                    coords[bw + a] -= step;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[inline]
fn accumulate_pacmap_tier_nd<S: Scalar>(
    coords: &[S],
    dims: usize,
    pairs: &[(u32, u32)],
    weights: Option<&[f64]>,
    scale: f64,
    denom_offset: f64,
    grad: &mut [f64],
) {
    for (idx, &(u, v)) in pairs.iter().enumerate() {
        let (bu, bv) = ((u as usize) * dims, (v as usize) * dims);
        let d2 = nd_dist_sq_f64(coords, bu, bv, dims);
        if d2.is_finite() {
            let denom = denom_offset + d2;
            let pair_w = weights.map_or(1.0, |ws| ws[idx]);
            let coeff = (scale * pair_w) / (denom * denom);
            if coeff.is_finite() {
                for a in 0..dims {
                    let g = coeff * (coords[bu + a].to_f64() - coords[bv + a].to_f64());
                    grad[bu + a] += g;
                    grad[bv + a] -= g;
                }
            }
        }
    }
}

pub(super) fn step_pacmap_state_nd<S: Scalar>(
    st: &mut PacmapState,
    num_nodes: usize,
    iter: usize,
    layout: &mut LayoutNd<S>,
) {
    let dims = layout.dims();
    let total_len = num_nodes * dims;
    let sc = st.nd_scratch.get_or_insert_with(|| PacmapNdScratch {
        grad: vec![0.0; total_len],
        m: vec![0.0; total_len],
        v: vec![0.0; total_len],
    });
    let total = st.config.total_iterations();
    if num_nodes <= 1 || total == 0 || iter >= total {
        return;
    }

    let [p1_iters, p2_iters, _] = st.config.phase_iterations;
    let (w_nb, w_mn, w_fp) = if iter < p1_iters {
        let p1 = if p1_iters <= 1 {
            0.0
        } else {
            (iter as f64) / ((p1_iters - 1) as f64)
        };
        (2.0, 1000.0 * (1.0 - p1) + 3.0 * p1, 1.0)
    } else if iter < p1_iters + p2_iters {
        (3.0, 3.0, 1.0)
    } else {
        (1.0, 0.0, 1.0)
    };

    sc.grad.fill(0.0);
    let coords = layout.coords_mut();
    if w_nb > 0.0 {
        let nb_weights = (!st.nb_weights.is_empty()).then_some(st.nb_weights.as_slice());
        accumulate_pacmap_tier_nd(
            coords,
            dims,
            &st.nb_pairs,
            nb_weights,
            20.0 * w_nb,
            10.0,
            &mut sc.grad,
        );
    }
    if w_mn > 0.0 {
        accumulate_pacmap_tier_nd(
            coords,
            dims,
            &st.mn_pairs,
            None,
            20_000.0 * w_mn,
            10_000.0,
            &mut sc.grad,
        );
    }
    if w_fp > 0.0 {
        accumulate_pacmap_tier_nd(
            coords,
            dims,
            &st.fp_pairs,
            None,
            -2.0 * w_fp,
            1.0,
            &mut sc.grad,
        );
    }

    const BETA1: f64 = 0.9;
    const BETA2: f64 = 0.999;
    const ADAM_EPS: f64 = 1e-7;
    let t_step = (iter + 1) as i32;
    let bc1 = (1.0 - BETA1.powi(t_step)).max(1e-12);
    let bc2 = (1.0 - BETA2.powi(t_step)).max(1e-12);
    let lr_t = st.config.learning_rate * bc2.sqrt() / bc1;

    for idx in 0..total_len {
        let g = sc.grad[idx];
        sc.m[idx] = BETA1 * sc.m[idx] + (1.0 - BETA1) * g;
        sc.v[idx] = BETA2 * sc.v[idx] + (1.0 - BETA2) * g * g;
        let step = lr_t * sc.m[idx] / (sc.v[idx].sqrt() + ADAM_EPS);
        if step.is_finite() {
            let next_c = coords[idx] - S::from_f64(step);
            if next_c.is_finite() {
                coords[idx] = next_c;
            }
        }
    }
}
