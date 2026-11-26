use std::f64::consts::TAU;

use egraph_core::{
    CoreResult, DeterministicRng, ManifoldKind, ManifoldLayout2d, ManifoldMetric, Scalar,
    cartesian_to_spherical, lorentz_einstein_midpoint, poincare_einstein_midpoint,
    spherical_to_cartesian, wrap_torus_point,
};

use super::drgraph_coarsen::DrGraphState;
use super::update::apply_clamped_manifold_vertex_step;

pub(super) fn sync_drgraph_schedule_iter(st: &mut DrGraphState, target_iter: usize) {
    let start_level = st.level_iters.iter().rposition(|&c| c > 0).unwrap_or(0);
    let mut rem = target_iter;
    let mut lvl = start_level;
    while lvl > 0 && rem >= st.level_iters[lvl] {
        rem -= st.level_iters[lvl];
        lvl -= 1;
    }
    st.current_level = lvl;
    st.level_iter = rem;
}

fn sync_manifold_metric(dst: &mut ManifoldLayout2d<f64>, target: ManifoldMetric) -> CoreResult<()> {
    if dst.metric() == target {
        return Ok(());
    }
    if dst.kind() == ManifoldKind::Torus2d {
        dst.set_torus_extents(target.scale * target.aspect, target.scale)
    } else {
        dst.set_scale(target.scale)
    }
}

fn init_drgraph_coarse_manifold<S: Scalar>(
    st: &mut DrGraphState,
    rng: &mut DeterministicRng,
    num_nodes: usize,
    layout: &ManifoldLayout2d<S>,
) -> CoreResult<ManifoldLayout2d<f64>> {
    let kind = layout.kind();
    let metric = layout.metric();
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
    let mut ml = if (max_x - min_x) < 1e-12 && (max_y - min_y) < 1e-12 {
        let mut init = ManifoldLayout2d::<f64>::zeros(kind, ln);
        sync_manifold_metric(&mut init, metric)?;
        let span = metric.scale;
        for i in 0..ln {
            let vx = span * (rng.next_f64() - 0.5);
            let vy = span * (rng.next_f64() - 0.5);
            apply_clamped_manifold_vertex_step(&mut init, i, vx, vy)?;
        }
        init
    } else if l_init == 0 {
        let mut init = ManifoldLayout2d::<f64>::from_xy(
            kind,
            st.active_x[..ln].to_vec(),
            st.active_y[..ln].to_vec(),
        )?;
        sync_manifold_metric(&mut init, metric)?;
        init
    } else {
        let map = &st.fine_to_level[l_init];
        let mut out_x = vec![0.0_f64; ln];
        let mut out_y = vec![0.0_f64; ln];
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); ln];
        for (u, &c_u32) in map.iter().enumerate().take(num_nodes) {
            buckets[c_u32 as usize].push(u);
        }
        for (c, members) in buckets.iter().enumerate() {
            if members.is_empty() {
                continue;
            }
            if members.len() == 1 {
                let u = members[0];
                out_x[c] = st.active_x[u];
                out_y[c] = st.active_y[u];
                continue;
            }
            match kind {
                ManifoldKind::HyperbolicPoincare => {
                    let cx: Vec<f64> = members.iter().map(|&u| st.active_x[u]).collect();
                    let cy: Vec<f64> = members.iter().map(|&u| st.active_y[u]).collect();
                    let (mx, my) = poincare_einstein_midpoint(&cx, &cy)?;
                    out_x[c] = mx;
                    out_y[c] = my;
                }
                ManifoldKind::HyperbolicLorentz => {
                    let cx: Vec<f64> = members.iter().map(|&u| st.active_x[u]).collect();
                    let cy: Vec<f64> = members.iter().map(|&u| st.active_y[u]).collect();
                    let (mx, my) = lorentz_einstein_midpoint(&cx, &cy)?;
                    out_x[c] = mx;
                    out_y[c] = my;
                }
                ManifoldKind::Spherical => {
                    let mut sum = [0.0_f64; 3];
                    for &u in members {
                        let cart = spherical_to_cartesian(st.active_x[u], st.active_y[u]);
                        sum[0] += cart[0];
                        sum[1] += cart[1];
                        sum[2] += cart[2];
                    }
                    if sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2] <= 1e-24 {
                        let u = members[0];
                        out_x[c] = st.active_x[u];
                        out_y[c] = st.active_y[u];
                    } else {
                        let (lon, colat) = cartesian_to_spherical(sum[0], sum[1], sum[2]);
                        out_x[c] = lon;
                        out_y[c] = colat;
                    }
                }
                ManifoldKind::Torus2d => {
                    let (mut cx, mut sx, mut cy, mut sy) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
                    for &u in members {
                        let (s_x, c_x) = (TAU * st.active_x[u]).sin_cos();
                        let (s_y, c_y) = (TAU * st.active_y[u]).sin_cos();
                        cx += c_x;
                        sx += s_x;
                        cy += c_y;
                        sy += s_y;
                    }
                    let ax = if cx * cx + sx * sx <= 1e-24 {
                        st.active_x[members[0]]
                    } else {
                        (sx.atan2(cx) / TAU).rem_euclid(1.0)
                    };
                    let ay = if cy * cy + sy * sy <= 1e-24 {
                        st.active_y[members[0]]
                    } else {
                        (sy.atan2(cy) / TAU).rem_euclid(1.0)
                    };
                    let (wx, wy) = wrap_torus_point(ax, ay);
                    out_x[c] = wx;
                    out_y[c] = wy;
                }
                _ => {
                    let inv = 1.0 / (members.len() as f64);
                    out_x[c] = members.iter().map(|&u| st.active_x[u]).sum::<f64>() * inv;
                    out_y[c] = members.iter().map(|&u| st.active_y[u]).sum::<f64>() * inv;
                }
            }
        }
        let mut init = ManifoldLayout2d::<f64>::from_xy(kind, out_x, out_y)?;
        sync_manifold_metric(&mut init, metric)?;
        init
    };
    sync_manifold_metric(&mut ml, metric)?;
    Ok(ml)
}

pub(super) fn step_drgraph_state_manifold<S: Scalar>(
    st: &mut DrGraphState,
    rng: &mut DeterministicRng,
    num_nodes: usize,
    iter: usize,
    layout: &mut ManifoldLayout2d<S>,
) -> CoreResult<()> {
    let total = st.config.iterations;
    if num_nodes <= 1 || total == 0 || iter >= total {
        return Ok(());
    }

    let kind = layout.kind();
    let metric = layout.metric();

    if !st.initialized_coords || st.active_manifold.is_none() {
        let ml = init_drgraph_coarse_manifold(st, rng, num_nodes, layout)?;
        st.active_manifold = Some(ml);
        st.initialized_coords = true;
    } else if st.current_level == 0 {
        let mut fine_x = Vec::with_capacity(num_nodes);
        let mut fine_y = Vec::with_capacity(num_nodes);
        for i in 0..num_nodes {
            fine_x.push(layout.x()[i].to_f64());
            fine_y.push(layout.y()[i].to_f64());
        }
        let mut ml = ManifoldLayout2d::<f64>::from_xy(kind, fine_x, fine_y)?;
        sync_manifold_metric(&mut ml, metric)?;
        st.active_manifold = Some(ml);
    } else if let Some(ml) = st.active_manifold.as_mut() {
        sync_manifold_metric(ml, metric)?;
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
    let grad_cap = 4.0 * metric.scale;

    if let (Some(edge_alias), Some(active_ml)) = (&st.edge_alias, st.active_manifold.as_mut())
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
                let (tux, tuy) = active_ml.tangent_delta(u, v)?;
                let (tvx, tvy) = active_ml.tangent_delta(v, u)?;
                let d2 = tux * tux + tuy * tuy;
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
                        let sux = (lr * attr_coeff * tux).clamp(-grad_cap, grad_cap);
                        let suy = (lr * attr_coeff * tuy).clamp(-grad_cap, grad_cap);
                        let svx = (lr * attr_coeff * tvx).clamp(-grad_cap, grad_cap);
                        let svy = (lr * attr_coeff * tvy).clamp(-grad_cap, grad_cap);
                        if sux.is_finite() && suy.is_finite() && svx.is_finite() && svy.is_finite()
                        {
                            apply_clamped_manifold_vertex_step(active_ml, u, sux, suy)?;
                            apply_clamped_manifold_vertex_step(active_ml, v, svx, svy)?;
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
                    let (tukx, tuky) = active_ml.tangent_delta(u, k)?;
                    let rd2 = tukx * tukx + tuky * tuky;
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
                            let sx = (-lr * rep_coeff * tukx).clamp(-grad_cap, grad_cap);
                            let sy = (-lr * rep_coeff * tuky).clamp(-grad_cap, grad_cap);
                            if sx.is_finite() && sy.is_finite() {
                                apply_clamped_manifold_vertex_step(active_ml, u, sx, sy)?;
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
        if let (Some(coarse_of), Some(counts), Some(active_ml)) = (
            st.levels[finer].coarse_of_finer.as_ref(),
            st.levels[finer].cluster_counts_of_finer.as_ref(),
            st.active_manifold.as_ref(),
        ) {
            let mut fx = Vec::with_capacity(fine_n);
            let mut fy = Vec::with_capacity(fine_n);
            for &c_u32 in coarse_of.iter().take(fine_n) {
                let c = c_u32 as usize;
                fx.push(active_ml.x()[c]);
                fy.push(active_ml.y()[c]);
            }
            let mut finer_ml = ManifoldLayout2d::<f64>::from_xy(kind, fx, fy)?;
            sync_manifold_metric(&mut finer_ml, metric)?;
            for (u, &c_u32) in coarse_of.iter().enumerate().take(fine_n) {
                let c = c_u32 as usize;
                if counts[c] > 1 {
                    let angle = TAU * rng.next_f64();
                    let radius = 1e-4 * metric.scale * (0.25 + 0.75 * rng.next_f64());
                    let (s, cs) = angle.sin_cos();
                    apply_clamped_manifold_vertex_step(&mut finer_ml, u, radius * cs, radius * s)?;
                }
            }
            st.active_manifold = Some(finer_ml);
        }
        st.current_level = finer;
        st.level_iter = 0;
    }

    if let Some(active_ml) = st.active_manifold.as_ref() {
        if st.current_level == 0 {
            for i in 0..num_nodes {
                let nx = S::from_f64(active_ml.x()[i]);
                let ny = S::from_f64(active_ml.y()[i]);
                if nx.is_finite() && ny.is_finite() {
                    layout.set_point(i, nx, ny)?;
                }
            }
        } else {
            let map = &st.fine_to_level[st.current_level];
            for (u, &c) in map.iter().enumerate() {
                let nx = S::from_f64(active_ml.x()[c as usize]);
                let ny = S::from_f64(active_ml.y()[c as usize]);
                if nx.is_finite() && ny.is_finite() {
                    layout.set_point(u, nx, ny)?;
                }
            }
        }
    }
    Ok(())
}
