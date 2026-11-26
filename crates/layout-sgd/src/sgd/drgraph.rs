use egraph_core::{DeterministicRng, Layout2d, LayoutNd, Scalar};

use super::drgraph_coarsen::DrGraphState;
use super::nd_variants::DrGraphNdScratch;

pub(super) fn step_drgraph_state<S: Scalar>(
    st: &mut DrGraphState,
    rng: &mut DeterministicRng,
    num_nodes: usize,
    iter: usize,
    layout: &mut Layout2d<S>,
) {
    let total = st.config.iterations;
    if num_nodes <= 1 || total == 0 || iter >= total {
        return;
    }

    if !st.initialized_coords {
        let xs = layout.x();
        let ys = layout.y();
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for i in 0..num_nodes {
            let x = xs[i].to_f64();
            let y = ys[i].to_f64();
            st.active_x[i] = x;
            st.active_y[i] = y;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
        let l_init = st.current_level;
        let ln = st.levels[l_init].num_nodes;
        if (max_x - min_x) < 1e-12 && (max_y - min_y) < 1e-12 {
            for i in 0..ln {
                st.active_x[i] = rng.next_f64() - 0.5;
                st.active_y[i] = rng.next_f64() - 0.5;
            }
        } else if l_init > 0 {
            let map = &st.fine_to_level[l_init];
            st.scratch_x[..ln].fill(0.0);
            st.scratch_y[..ln].fill(0.0);
            st.scratch_cnt[..ln].fill(0.0);
            for (u, &c_u32) in map.iter().enumerate().take(num_nodes) {
                let c = c_u32 as usize;
                st.scratch_x[c] += st.active_x[u];
                st.scratch_y[c] += st.active_y[u];
                st.scratch_cnt[c] += 1.0;
            }
            for c in 0..ln {
                let cnt = st.scratch_cnt[c].max(1.0);
                st.active_x[c] = st.scratch_x[c] / cnt;
                st.active_y[c] = st.scratch_y[c] / cnt;
            }
        }
        st.initialized_coords = true;
    } else if st.current_level == 0 {
        let xs = layout.x();
        let ys = layout.y();
        for i in 0..num_nodes {
            st.active_x[i] = xs[i].to_f64();
            st.active_y[i] = ys[i].to_f64();
        }
    }

    let l = st.current_level;
    let ln = st.levels[l].num_nodes;
    let lr = (st.config.initial_lr * (1.0 - (iter as f64) / (total as f64))).max(st.config.min_lr);
    let gamma = if l == st.levels.len() - 1 && st.levels.len() > 1 {
        st.config.gamma_coarse
    } else {
        st.config.gamma_fine
    };
    let b = st.config.b;
    let neg_samples = st.config.negative_samples;

    if let Some(edge_alias) = &st.edge_alias
        && lr.is_finite()
        && lr > 0.0
        && ln > 1
    {
        let m = st.pos_u.len();
        for _ in 0..m {
            let e_idx = edge_alias.sample(rng);
            let u0 = st.pos_u[e_idx] as usize;
            let v0 = st.pos_v[e_idx] as usize;
            let (su, sv) = if l > 0 {
                let map = &st.fine_to_level[l];
                (map[u0] as usize, map[v0] as usize)
            } else {
                (u0, v0)
            };
            let (u, v) = if (rng.next_u32() & 1) == 0 {
                (su, sv)
            } else {
                (sv, su)
            };

            if u != v {
                let dx = st.active_x[u] - st.active_x[v];
                let dy = st.active_y[u] - st.active_y[v];
                let d2 = dx * dx + dy * dy;
                if d2.is_finite() {
                    let d2_safe = if b < 1.0 { d2.max(1e-12) } else { d2 };
                    let (pow_bm1, pow_b) = if (b - 2.0).abs() == 0.0 {
                        (d2_safe, d2_safe * d2_safe)
                    } else if (b - 1.0).abs() == 0.0 {
                        (1.0, d2_safe)
                    } else {
                        (d2_safe.powf(b - 1.0), d2_safe.powf(b))
                    };
                    let attr_coeff = (2.0 * b * pow_bm1) / (1.0 + pow_b);
                    if attr_coeff.is_finite() {
                        let sx = (lr * attr_coeff * dx).clamp(-4.0, 4.0);
                        let sy = (lr * attr_coeff * dy).clamp(-4.0, 4.0);
                        if sx.is_finite() && sy.is_finite() {
                            st.active_x[u] -= sx;
                            st.active_y[u] -= sy;
                            st.active_x[v] += sx;
                            st.active_y[v] += sy;
                        }
                    }
                }
            }

            if gamma > 0.0 {
                for _ in 0..neg_samples {
                    let k = rng.gen_range_usize(ln);
                    if k == u {
                        continue;
                    }
                    let rdx = st.active_x[u] - st.active_x[k];
                    let rdy = st.active_y[u] - st.active_y[k];
                    let rd2 = rdx * rdx + rdy * rdy;
                    if rd2.is_finite() && rd2 > 0.0 {
                        let pow_b = if (b - 2.0).abs() == 0.0 {
                            rd2 * rd2
                        } else if (b - 1.0).abs() == 0.0 {
                            rd2
                        } else {
                            rd2.powf(b)
                        };
                        let rep_coeff = (2.0 * b * gamma) / ((rd2 + 1e-6) * (1.0 + pow_b));
                        if rep_coeff.is_finite() {
                            let sx = (lr * rep_coeff * rdx).clamp(-4.0, 4.0);
                            let sy = (lr * rep_coeff * rdy).clamp(-4.0, 4.0);
                            if sx.is_finite() && sy.is_finite() {
                                st.active_x[u] += sx;
                                st.active_y[u] += sy;
                            }
                        }
                    }
                }
            }
        }
    }

    st.level_iter += 1;
    while st.current_level > 0 && st.level_iter >= st.level_iters[st.current_level] {
        let finer = st.current_level - 1;
        let fine_n = st.levels[finer].num_nodes;
        if let (Some(coarse_of), Some(counts)) = (
            st.levels[finer].coarse_of_finer.as_ref(),
            st.levels[finer].cluster_counts_of_finer.as_ref(),
        ) {
            for (u, &c_u32) in coarse_of.iter().enumerate().take(fine_n) {
                let c = c_u32 as usize;
                let cx = st.active_x[c];
                let cy = st.active_y[c];
                if counts[c] <= 1 {
                    st.scratch_x[u] = cx;
                    st.scratch_y[u] = cy;
                } else {
                    let angle = std::f64::consts::TAU * rng.next_f64();
                    let radius = 1e-4 * (0.25 + 0.75 * rng.next_f64());
                    let (s, cs) = angle.sin_cos();
                    st.scratch_x[u] = cx + radius * cs;
                    st.scratch_y[u] = cy + radius * s;
                }
            }
            st.active_x[..fine_n].copy_from_slice(&st.scratch_x[..fine_n]);
            st.active_y[..fine_n].copy_from_slice(&st.scratch_y[..fine_n]);
        }
        st.current_level = finer;
        st.level_iter = 0;
    }

    let (xs_mut, ys_mut) = layout.xy_mut();
    if st.current_level == 0 {
        for i in 0..num_nodes {
            let nx = S::from_f64(st.active_x[i]);
            let ny = S::from_f64(st.active_y[i]);
            if nx.is_finite() {
                xs_mut[i] = nx;
            }
            if ny.is_finite() {
                ys_mut[i] = ny;
            }
        }
    } else {
        let map = &st.fine_to_level[st.current_level];
        for (u, &c) in map.iter().enumerate() {
            let nx = S::from_f64(st.active_x[c as usize]);
            let ny = S::from_f64(st.active_y[c as usize]);
            if nx.is_finite() {
                xs_mut[u] = nx;
            }
            if ny.is_finite() {
                ys_mut[u] = ny;
            }
        }
    }
}

pub(super) fn step_drgraph_state_nd<S: Scalar>(
    st: &mut DrGraphState,
    rng: &mut DeterministicRng,
    num_nodes: usize,
    iter: usize,
    layout: &mut LayoutNd<S>,
) {
    let dims = layout.dims();
    let total_len = num_nodes * dims;
    let sc = st
        .nd_scratch
        .get_or_insert_with(|| DrGraphNdScratch::new(num_nodes, total_len));
    let total = st.config.iterations;
    if num_nodes <= 1 || total == 0 || iter >= total {
        return;
    }

    if !st.initialized_coords {
        for (dst, src) in sc.active.iter_mut().zip(layout.coords().iter()) {
            *dst = src.to_f64();
        }
        let all_collapsed = (0..dims).all(|a| {
            let mut min_a = f64::INFINITY;
            let mut max_a = f64::NEG_INFINITY;
            for i in 0..num_nodes {
                let v = sc.active[i * dims + a];
                min_a = min_a.min(v);
                max_a = max_a.max(v);
            }
            (max_a - min_a) < 1e-12
        });
        let l_init = st.current_level;
        let ln = st.levels[l_init].num_nodes;
        if all_collapsed {
            for val in sc.active[..ln * dims].iter_mut() {
                *val = rng.next_f64() - 0.5;
            }
        } else if l_init > 0 {
            let map = &st.fine_to_level[l_init];
            sc.scratch[..ln * dims].fill(0.0);
            sc.scratch_cnt[..ln].fill(0.0);
            for (u, &c_u32) in map.iter().enumerate().take(num_nodes) {
                let c = c_u32 as usize;
                let (su, dc) = (u * dims, c * dims);
                for a in 0..dims {
                    sc.scratch[dc + a] += sc.active[su + a];
                }
                sc.scratch_cnt[c] += 1.0;
            }
            for c in 0..ln {
                let cnt = sc.scratch_cnt[c].max(1.0);
                let base = c * dims;
                for a in 0..dims {
                    sc.active[base + a] = sc.scratch[base + a] / cnt;
                }
            }
        }
        st.initialized_coords = true;
    } else if st.current_level == 0 {
        for (dst, src) in sc.active.iter_mut().zip(layout.coords().iter()) {
            *dst = src.to_f64();
        }
    }

    let l = st.current_level;
    let ln = st.levels[l].num_nodes;
    let lr = (st.config.initial_lr * (1.0 - (iter as f64) / (total as f64))).max(st.config.min_lr);
    let gamma = if l == st.levels.len() - 1 && st.levels.len() > 1 {
        st.config.gamma_coarse
    } else {
        st.config.gamma_fine
    };
    let b = st.config.b;
    let neg_samples = st.config.negative_samples;

    if let Some(edge_alias) = &st.edge_alias
        && lr.is_finite()
        && lr > 0.0
        && ln > 1
    {
        for _ in 0..st.pos_u.len() {
            let e_idx = edge_alias.sample(rng);
            let u0 = st.pos_u[e_idx] as usize;
            let v0 = st.pos_v[e_idx] as usize;
            let (su, sv) = if l > 0 {
                let map = &st.fine_to_level[l];
                (map[u0] as usize, map[v0] as usize)
            } else {
                (u0, v0)
            };
            let (u, v) = if (rng.next_u32() & 1) == 0 {
                (su, sv)
            } else {
                (sv, su)
            };

            if u != v {
                let (bu, bv) = (u * dims, v * dims);
                let d2: f64 = (0..dims)
                    .map(|a| {
                        let d = sc.active[bu + a] - sc.active[bv + a];
                        d * d
                    })
                    .sum();
                if d2.is_finite() {
                    let d2_safe = if b < 1.0 { d2.max(1e-12) } else { d2 };
                    let (pow_bm1, pow_b) = if (b - 2.0).abs() == 0.0 {
                        (d2_safe, d2_safe * d2_safe)
                    } else if (b - 1.0).abs() == 0.0 {
                        (1.0, d2_safe)
                    } else {
                        (d2_safe.powf(b - 1.0), d2_safe.powf(b))
                    };
                    let coeff = (2.0 * b * pow_bm1) / (1.0 + pow_b);
                    if coeff.is_finite()
                        && (0..dims).all(|a| {
                            (lr * coeff * (sc.active[bu + a] - sc.active[bv + a]))
                                .clamp(-4.0, 4.0)
                                .is_finite()
                        })
                    {
                        for a in 0..dims {
                            let s = (lr * coeff * (sc.active[bu + a] - sc.active[bv + a]))
                                .clamp(-4.0, 4.0);
                            sc.active[bu + a] -= s;
                            sc.active[bv + a] += s;
                        }
                    }
                }
            }

            if gamma > 0.0 {
                let bu = u * dims;
                for _ in 0..neg_samples {
                    let k = rng.gen_range_usize(ln);
                    if k == u {
                        continue;
                    }
                    let bk = k * dims;
                    let rd2: f64 = (0..dims)
                        .map(|a| {
                            let d = sc.active[bu + a] - sc.active[bk + a];
                            d * d
                        })
                        .sum();
                    if rd2.is_finite() && rd2 > 0.0 {
                        let pow_b = if (b - 2.0).abs() == 0.0 {
                            rd2 * rd2
                        } else if (b - 1.0).abs() == 0.0 {
                            rd2
                        } else {
                            rd2.powf(b)
                        };
                        let coeff = (2.0 * b * gamma) / ((rd2 + 1e-6) * (1.0 + pow_b));
                        if coeff.is_finite()
                            && (0..dims).all(|a| {
                                (lr * coeff * (sc.active[bu + a] - sc.active[bk + a]))
                                    .clamp(-4.0, 4.0)
                                    .is_finite()
                            })
                        {
                            for a in 0..dims {
                                let s = (lr * coeff * (sc.active[bu + a] - sc.active[bk + a]))
                                    .clamp(-4.0, 4.0);
                                sc.active[bu + a] += s;
                            }
                        }
                    }
                }
            }
        }
    }

    st.level_iter += 1;
    while st.current_level > 0 && st.level_iter >= st.level_iters[st.current_level] {
        let finer = st.current_level - 1;
        let fine_n = st.levels[finer].num_nodes;
        if let (Some(coarse_of), Some(counts)) = (
            st.levels[finer].coarse_of_finer.as_ref(),
            st.levels[finer].cluster_counts_of_finer.as_ref(),
        ) {
            for (u, &c_u32) in coarse_of.iter().enumerate().take(fine_n) {
                let c = c_u32 as usize;
                let (dst, src) = (u * dims, c * dims);
                if counts[c] <= 1 {
                    sc.scratch[dst..dst + dims].copy_from_slice(&sc.active[src..src + dims]);
                } else {
                    let mut norm_sq = 0.0_f64;
                    for a in 0..dims {
                        let comp = rng.next_f64() - 0.5;
                        sc.scratch[dst + a] = comp;
                        norm_sq += comp * comp;
                    }
                    let radius = 1e-4 * (0.25 + 0.75 * rng.next_f64());
                    let norm = norm_sq.sqrt();
                    if norm <= 1e-12 || !norm.is_finite() {
                        sc.scratch[dst..dst + dims].copy_from_slice(&sc.active[src..src + dims]);
                        sc.scratch[dst] += radius;
                    } else {
                        let scale = radius / norm;
                        for a in 0..dims {
                            sc.scratch[dst + a] = sc.active[src + a] + sc.scratch[dst + a] * scale;
                        }
                    }
                }
            }
            sc.active[..fine_n * dims].copy_from_slice(&sc.scratch[..fine_n * dims]);
        }
        st.current_level = finer;
        st.level_iter = 0;
    }

    let coords_mut = layout.coords_mut();
    if st.current_level == 0 {
        for (dst, &val) in coords_mut.iter_mut().zip(sc.active.iter()) {
            let next_c = S::from_f64(val);
            if next_c.is_finite() {
                *dst = next_c;
            }
        }
    } else {
        let map = &st.fine_to_level[st.current_level];
        for (u, &c) in map.iter().enumerate() {
            let (src, dst) = ((c as usize) * dims, u * dims);
            for a in 0..dims {
                let next_c = S::from_f64(sc.active[src + a]);
                if next_c.is_finite() {
                    coords_mut[dst + a] = next_c;
                }
            }
        }
    }
}
