use std::f64::consts::PI;

use egraph_core::VisitStamp;

use super::config::{ClusterAreaConfig, H_MAX};
use super::mec::NormalizedFrame;

/// Fills `buf` (`len == 2 * r_stencil + 1`) with normalized 1D Gaussian stencil
/// weights centered at fractional offset `frac` (`|frac| <= 0.5`).
///
/// Shifting squared exponents by `min_dx2 = frac * frac` prevents catastrophic
/// underflow and `0.0 * (-inf) = NaN` when `sigma_cells` is arbitrarily small,
/// smoothly converging to the exact Kronecker delta (or `[0.5, 0.5]` at half-cell
/// ties) as `sigma_cells -> 0+`.
#[inline]
pub(super) fn fill_1d_gaussian(
    buf: &mut [f64],
    r_stencil: usize,
    frac: f64,
    sigma_cells: f64,
    inv_two_sig2: f64,
) {
    let _ = sigma_cells;
    let min_dx2 = frac * frac;
    let r_stencil_i = r_stencil as isize;
    let mut sum = 0.0_f64;
    for (t_idx, slot) in buf.iter_mut().enumerate() {
        let off = (t_idx as isize) - r_stencil_i;
        let dx = (off as f64) - frac;
        let delta2 = (dx * dx - min_dx2).max(0.0);
        let val = if delta2 == 0.0 {
            // Avoid 0.0 * (-inf) = NaN when inv_two_sig2 overflows for tiny sigma_cells.
            1.0
        } else if !inv_two_sig2.is_finite() {
            0.0
        } else {
            (inv_two_sig2 * delta2).exp()
        };
        *slot = val;
        sum += val;
    }
    let inv_sum = 1.0 / sum;
    for slot in buf.iter_mut() {
        *slot *= inv_sum;
    }
}

pub(super) struct GridContext {
    pub(super) g_eff: usize,
    pub(super) g_eff_f: f64,
    pub(super) effective_b: f64,
    pub(super) in_frac: f64,
    pub(super) h_f: f64,
    pub(super) s: usize,
    pub(super) s_sq: usize,
    pub(super) r_stencil: usize,
    pub(super) stencil_len: usize,
    pub(super) inv_two_sig2: f64,
    pub(super) m0_disc: f64,
    pub(super) a_disk: f64,
    pub(super) rho_canv: f64,
    pub(super) tau: f64,
    pub(super) offsets: Vec<usize>,
    pub(super) bucket_x: Vec<f64>,
    pub(super) bucket_y: Vec<f64>,
    pub(super) w_tot: Vec<f64>,
    pub(super) q_tot: Vec<f64>,
    pub(super) phi_all: Vec<f64>,
}

pub(super) fn build_global_density_grid(
    frame: &NormalizedFrame,
    inv: &[u32],
    k_count: usize,
    config: &ClusterAreaConfig,
) -> GridContext {
    let n = inv.len();
    let n_f = n as f64;
    let r_stencil = (3.5 * config.sigma_cells).ceil() as usize;
    let stencil_len = 2 * r_stencil + 1;
    let avail_h = (H_MAX - r_stencil - 1) as f64;
    let denom_g = 0.55 * (PI / n_f).sqrt();
    let raw_g = (config.sigma_cells / denom_g).floor();
    let g_eff_raw = if raw_g.is_finite() {
        (raw_g.min(config.g_max as f64) as usize).clamp(12, config.g_max)
    } else {
        config.g_max
    };
    let (g_eff, effective_b) = if config.boundary_r95.is_finite() {
        let g_max_fit_f = (avail_h / config.boundary_r95).floor();
        if g_max_fit_f >= 12.0 {
            let g = if g_max_fit_f >= (g_eff_raw as f64) {
                g_eff_raw
            } else {
                (g_max_fit_f as usize).max(12)
            };
            (g, config.boundary_r95)
        } else {
            let g = 12usize;
            let max_b_grid = avail_h / (g as f64);
            (g, config.boundary_r95.min(max_b_grid))
        }
    } else {
        // Dense-grid resolution-protection mode for boundary_r95 = +inf:
        // preserve g_eff >= 12 instead of crushing main-body resolution for far outliers.
        let g = g_eff_raw;
        let max_b_grid = avail_h / (g as f64);
        (g, max_b_grid)
    };
    let g_eff_f = g_eff as f64;
    let max_b_grid = avail_h / g_eff_f;
    let r_cut = effective_b * frame.r95 * (1.0 + 1e-12);

    let mut in_counts = vec![0usize; k_count];
    let mut n_in = 0usize;
    let mut max_abs_in = 0.0_f64;
    for i in 0..n {
        if frame.d_buf[i] <= r_cut {
            in_counts[inv[i] as usize] += 1;
            n_in += 1;
            let xn = (frame.x_pre[i] - frame.mu.0) / frame.r95;
            let yn = (frame.y_pre[i] - frame.mu.1) / frame.r95;
            max_abs_in = max_abs_in.max(xn.abs()).max(yn.abs());
        }
    }
    let in_frac = ((n_in as f64) / n_f).clamp(0.0, 1.0);

    let h = if config.boundary_r95.is_finite() && config.boundary_r95 <= max_b_grid {
        ((effective_b * g_eff_f).ceil() as usize + r_stencil + 1).min(H_MAX)
    } else {
        ((max_abs_in * g_eff_f).ceil() as usize + r_stencil + 1).min(H_MAX)
    };
    let h_f = h as f64;
    let s = 2 * h + 1;
    let s_sq = s * s;

    // O(N + K) CSR bucket grouping of in-bounds normalized coordinates.
    let mut offsets = vec![0usize; k_count + 1];
    for k in 0..k_count {
        offsets[k + 1] = offsets[k] + in_counts[k];
    }
    let mut cursor = offsets[..k_count].to_vec();
    let mut bucket_x = vec![0.0_f64; n_in];
    let mut bucket_y = vec![0.0_f64; n_in];
    for i in 0..n {
        if frame.d_buf[i] <= r_cut {
            let k = inv[i] as usize;
            let pos = cursor[k];
            cursor[k] = pos + 1;
            bucket_x[pos] = (frame.x_pre[i] - frame.mu.0) / frame.r95;
            bucket_y[pos] = (frame.y_pre[i] - frame.mu.1) / frame.r95;
        }
    }

    // Reference constants.
    let mut g0 = vec![0.0_f64; stencil_len];
    let inv_two_sig2 = -0.5 / (config.sigma_cells * config.sigma_cells);
    fill_1d_gaussian(&mut g0, r_stencil, 0.0, config.sigma_cells, inv_two_sig2);
    let mut m0_entropy = 0.0_f64;
    for &val in &g0 {
        if val > 0.0 {
            m0_entropy -= val * val.max(1e-30).ln();
        }
    }
    let m0_disc = (2.0 * m0_entropy).exp();
    let a_disk = (PI / 0.95) * g_eff_f * g_eff_f;
    let rho_canv = n_f / a_disk;
    let tau = config.tau_frac * rho_canv;

    // Global density accumulation across all in-bounds points.
    // Track subtraction-free pairwise weight sum Q_m = 2 * sum_{i < j} w_i * w_j
    // via Q_{m+1} = Q_m + 2 * S_m * w_{m+1} to avoid catastrophic cancellation in
    // S^2 - sum(w^2) when one dominant Gaussian weight dwarfs a tail contribution.
    let mut w_tot = vec![0.0_f64; s_sq];
    let mut q_tot = vec![0.0_f64; s_sq];
    let mut wx = vec![0.0_f64; stencil_len];
    let mut wy = vec![0.0_f64; stencil_len];
    let r_stencil_i = r_stencil as isize;
    let s_i = s as isize;

    for m in 0..n_in {
        let u = bucket_x[m] * g_eff_f + h_f;
        let v = bucket_y[m] * g_eff_f + h_f;
        // round_ties_even matches np.rint() bit-for-bit.
        let iu = u.round_ties_even() as isize;
        let iv = v.round_ties_even() as isize;
        let fu = u - (iu as f64);
        let fv = v - (iv as f64);

        fill_1d_gaussian(&mut wx, r_stencil, fu, config.sigma_cells, inv_two_sig2);
        fill_1d_gaussian(&mut wy, r_stencil, fv, config.sigma_cells, inv_two_sig2);

        for (dy_idx, &wy_val) in wy.iter().enumerate() {
            let gy = iv + (dy_idx as isize - r_stencil_i);
            if gy < 0 || gy >= s_i {
                continue;
            }
            let row_off = (gy as usize) * s;
            for (dx_idx, &wx_val) in wx.iter().enumerate() {
                let gx = iu + (dx_idx as isize - r_stencil_i);
                if gx < 0 || gx >= s_i {
                    continue;
                }
                let w = wx_val * wy_val;
                let cell = row_off + (gx as usize);
                q_tot[cell] += 2.0 * w_tot[cell] * w;
                w_tot[cell] += w;
            }
        }
    }

    let mut phi_all = vec![0.0_f64; s_sq];
    for c in 0..s_sq {
        let wt = w_tot[c];
        if wt > 0.0 {
            // Compute (wt / tau)^2 before squaring and use exp_m1 to avoid both
            // 1 - exp(-x) subtraction cancellation and 0.0 * inf = NaN when wt and tau underflow.
            let wt_tau = wt / tau;
            phi_all[c] = -(-(wt_tau * wt_tau)).exp_m1();
        }
    }

    GridContext {
        g_eff,
        g_eff_f,
        effective_b,
        in_frac,
        h_f,
        s,
        s_sq,
        r_stencil,
        stencil_len,
        inv_two_sig2,
        m0_disc,
        a_disk,
        rho_canv,
        tau,
        offsets,
        bucket_x,
        bucket_y,
        w_tot,
        q_tot,
        phi_all,
    }
}

pub(super) struct ClusterGridScratch {
    pub(super) wk_grid: Vec<f64>,
    pub(super) qk_grid: Vec<f64>,
    pub(super) splat_stamp: VisitStamp,
    pub(super) touched: Vec<usize>,
    pub(super) wx: Vec<f64>,
    pub(super) wy: Vec<f64>,
}

impl ClusterGridScratch {
    pub(super) fn new(ctx: &GridContext) -> Self {
        Self {
            wk_grid: vec![0.0_f64; ctx.s_sq],
            qk_grid: vec![0.0_f64; ctx.s_sq],
            splat_stamp: VisitStamp::new(ctx.s_sq),
            touched: Vec::with_capacity(1024),
            wx: vec![0.0_f64; ctx.stencil_len],
            wy: vec![0.0_f64; ctx.stencil_len],
        }
    }
}

pub(super) struct ClusterDensityMetrics {
    pub(super) e_k: f64,
    pub(super) d_k: f64,
    pub(super) r_k: f64,
    pub(super) purity: f64,
    pub(super) a_vis: f64,
    pub(super) max_wk: f64,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn splat_cluster_density(
    ctx: &GridContext,
    scratch: &mut ClusterGridScratch,
    st: usize,
    en: usize,
    nk: usize,
    n: usize,
    pk: f64,
    uniform_gamma_one: bool,
    config: &ClusterAreaConfig,
) -> ClusterDensityMetrics {
    let r_stencil_i = ctx.r_stencil as isize;
    let s_i = ctx.s as isize;

    scratch.splat_stamp.clear();
    scratch.touched.clear();
    for m in st..en {
        let u = ctx.bucket_x[m] * ctx.g_eff_f + ctx.h_f;
        let v = ctx.bucket_y[m] * ctx.g_eff_f + ctx.h_f;
        let iu = u.round_ties_even() as isize;
        let iv = v.round_ties_even() as isize;
        let fu = u - (iu as f64);
        let fv = v - (iv as f64);

        fill_1d_gaussian(
            &mut scratch.wx,
            ctx.r_stencil,
            fu,
            config.sigma_cells,
            ctx.inv_two_sig2,
        );
        fill_1d_gaussian(
            &mut scratch.wy,
            ctx.r_stencil,
            fv,
            config.sigma_cells,
            ctx.inv_two_sig2,
        );

        for (dy_idx, &wy_val) in scratch.wy.iter().enumerate() {
            let gy = iv + (dy_idx as isize - r_stencil_i);
            if gy < 0 || gy >= s_i {
                continue;
            }
            let row_off = (gy as usize) * ctx.s;
            for (dx_idx, &wx_val) in scratch.wx.iter().enumerate() {
                let gx = iu + (dx_idx as isize - r_stencil_i);
                if gx < 0 || gx >= s_i {
                    continue;
                }
                let w = wx_val * wy_val;
                let cell = row_off + (gx as usize);
                if scratch.splat_stamp.visit(cell) {
                    // First touch in this cluster overwrites stale values from earlier clusters.
                    scratch.wk_grid[cell] = w;
                    scratch.qk_grid[cell] = 0.0;
                    scratch.touched.push(cell);
                } else {
                    scratch.qk_grid[cell] += 2.0 * scratch.wk_grid[cell] * w;
                    scratch.wk_grid[cell] += w;
                }
            }
        }
    }

    let cap_density_k = if uniform_gamma_one {
        config.kappa * ctx.rho_canv
    } else if pk > 0.0 {
        let target_area_k = pk * ctx.a_disk;
        if target_area_k > 0.0 {
            config.kappa * ((nk as f64) / target_area_k)
        } else {
            f64::INFINITY
        }
    } else {
        f64::INFINITY
    };

    let mut wk_sum = 0.0_f64;
    let mut e_sum = 0.0_f64;
    let mut a_vis_k = 0.0_f64;
    let mut num_k = 0.0_f64;
    let mut b_k = 0.0_f64;
    let mut max_wk = 0.0_f64;

    for &cell in &scratch.touched {
        let wk = scratch.wk_grid[cell];
        let wt = ctx.w_tot[cell];
        if wk <= 1e-15 || wt <= 1e-15 {
            continue;
        }
        if wk > max_wk {
            max_wk = wk;
        }
        let qk = scratch.qk_grid[cell];
        let qt = ctx.q_tot[cell];
        let ph = ctx.phi_all[cell];
        let frac = wk / wt;
        let wt2 = wt * wt;
        wk_sum += wk;
        e_sum += frac * wt.min(cap_density_k);
        a_vis_k += frac * ph;
        num_k += (qk / wt2) * ph;
        b_k += frac * (qt / wt2) * ph;
    }

    let e_k = (e_sum / (nk as f64)).clamp(0.0, 1.0);
    let d_k = if nk <= 1 {
        1.0
    } else {
        let mut m_shan = 0.0_f64;
        if wk_sum > 0.0 {
            let mut h_shan = 0.0_f64;
            for &cell in &scratch.touched {
                let wk = scratch.wk_grid[cell];
                if wk > 1e-15 {
                    let uk = wk / wk_sum;
                    h_shan -= uk * uk.max(1e-30).ln();
                }
            }
            m_shan = h_shan.exp();
        }
        let denom_d = pk * ctx.a_disk / config.kappa;
        if denom_d > 0.0 {
            ((m_shan - ctx.m0_disc).max(0.0) / denom_d).clamp(0.0, 1.0)
        } else {
            0.0
        }
    };
    let r_k = e_k.min(d_k).clamp(0.0, 1.0);

    let p_pair = ((nk as f64) - 1.0) / ((n as f64) - 1.0);
    let purity = if b_k > 1e-15 && nk < n {
        ((num_k / b_k - p_pair) / (1.0 - p_pair)).clamp(0.0, 1.0)
    } else {
        0.0
    };

    ClusterDensityMetrics {
        e_k,
        d_k,
        r_k,
        purity,
        a_vis: a_vis_k,
        max_wk,
    }
}
