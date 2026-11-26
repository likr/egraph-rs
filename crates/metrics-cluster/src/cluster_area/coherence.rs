use egraph_core::{CoreResult, VisitStamp};

use super::config::{
    COH_DOMINANCE_MIN, ClusterAreaConfig, ClusterAreaDetail, ClusterAreaReport,
    compute_target_weights, degenerate_detail,
};
use super::grid::{
    ClusterGridScratch, GridContext, build_global_density_grid, splat_cluster_density,
};
use super::mec::normalize_and_center_mec;

fn cluster_active_coherence(
    ctx: &GridContext,
    scratch: &ClusterGridScratch,
    bfs_stamp: &mut VisitStamp,
    stack: &mut Vec<usize>,
    max_wk_k: f64,
) -> f64 {
    let s = ctx.s;
    let s_i = s as isize;
    let wk_thresh_k = ctx.tau.min(0.10 * max_wk_k);
    let is_active_cell = |cell: usize| -> bool {
        let wk = scratch.wk_grid[cell];
        let wt = ctx.w_tot[cell];
        wk > 1e-15 && wt > 1e-15 && wk >= wk_thresh_k && (wk / wt) >= COH_DOMINANCE_MIN
    };

    let mut active_mass = 0.0_f64;
    for &cell in &scratch.touched {
        if is_active_cell(cell) {
            active_mass += (scratch.wk_grid[cell] / ctx.w_tot[cell]) * ctx.phi_all[cell];
        }
    }
    if active_mass <= 1e-15 {
        return 0.0;
    }

    bfs_stamp.clear();
    let mut max_comp_mass = 0.0_f64;
    for &start_cell in &scratch.touched {
        if !is_active_cell(start_cell) || !bfs_stamp.visit(start_cell) {
            continue;
        }
        stack.clear();
        stack.push(start_cell);
        let mut comp_mass = 0.0_f64;
        while let Some(cell) = stack.pop() {
            comp_mass += (scratch.wk_grid[cell] / ctx.w_tot[cell]) * ctx.phi_all[cell];
            let iy = (cell / s) as isize;
            let ix = (cell % s) as isize;
            for dy_nb in -1isize..=1 {
                let ny = iy + dy_nb;
                if ny < 0 || ny >= s_i {
                    continue;
                }
                let row_off = (ny as usize) * s;
                for dx_nb in -1isize..=1 {
                    if dx_nb == 0 && dy_nb == 0 {
                        continue;
                    }
                    let nx = ix + dx_nb;
                    if nx < 0 || nx >= s_i {
                        continue;
                    }
                    let ncell = row_off + (nx as usize);
                    // Check splat_stamp before wk_grid to skip stale values from prior clusters.
                    if scratch.splat_stamp.is_visited(ncell)
                        && is_active_cell(ncell)
                        && bfs_stamp.visit(ncell)
                    {
                        stack.push(ncell);
                    }
                }
            }
        }
        max_comp_mass = max_comp_mass.max(comp_mass);
    }
    (max_comp_mass / active_mass).clamp(0.0, 1.0)
}

#[allow(clippy::too_many_arguments)]
fn aggregate_caf_detail(
    ctx: &GridContext,
    cluster_ids: Vec<u32>,
    counts: Vec<usize>,
    p: Vec<f64>,
    e_k: Vec<f64>,
    d_k: Vec<f64>,
    r_k: Vec<f64>,
    purity: Vec<f64>,
    a_vis: Vec<f64>,
    coherence: Vec<f64>,
    n: usize,
    include_detail: bool,
) -> ClusterAreaDetail {
    let k_count = cluster_ids.len();
    // Clamp all [0.0, 1.0] quantities so 1-3 ULP floating-point summation drift
    // never violates the closed unit-interval output contract.
    let res_density: f64 = p
        .iter()
        .zip(e_k.iter())
        .map(|(&pk, &ek)| pk * ek)
        .sum::<f64>()
        .clamp(0.0, 1.0);
    let res_deconv: f64 = p
        .iter()
        .zip(d_k.iter())
        .map(|(&pk, &dk)| pk * dk)
        .sum::<f64>()
        .clamp(0.0, 1.0);
    let res_combined: f64 = p
        .iter()
        .zip(r_k.iter())
        .map(|(&pk, &rk)| pk * rk)
        .sum::<f64>()
        .clamp(0.0, 1.0);
    let a_tot: f64 = a_vis.iter().sum();

    if a_tot <= 1e-15 {
        let report = ClusterAreaReport {
            caf: 0.0,
            purity_adj: 0.0,
            prop_vis: 0.0,
            res_density,
            res_deconv,
            res_combined,
            spatial_coherence: 0.0,
            in_frac: ctx.in_frac,
            effective_boundary_r95: ctx.effective_b,
            g_eff: ctx.g_eff,
            num_clusters: k_count,
            num_nodes: n,
        };
        if !include_detail {
            return ClusterAreaDetail {
                report,
                ..ClusterAreaDetail::default()
            };
        }
        return ClusterAreaDetail {
            report,
            cluster_ids,
            counts,
            p,
            q: vec![0.0; k_count],
            purity,
            resolution: r_k,
            res_density: e_k,
            res_deconv: d_k,
            caf_per_cluster: vec![0.0; k_count],
            coherence: vec![0.0; k_count],
        };
    }

    let q: Vec<f64> = a_vis
        .iter()
        .map(|&ak| (ak / a_tot).clamp(0.0, 1.0))
        .collect();
    let purity_adj: f64 = purity
        .iter()
        .zip(q.iter())
        .map(|(&pk, &qk)| pk * qk)
        .sum::<f64>()
        .clamp(0.0, 1.0);
    let spatial_coherence: f64 = coherence
        .iter()
        .zip(q.iter())
        .map(|(&ck, &qk)| ck * qk)
        .sum::<f64>()
        .clamp(0.0, 1.0);
    let prop_vis = (1.0
        - 0.5
            * p.iter()
                .zip(q.iter())
                .map(|(&pk, &qk)| (pk - qk).abs())
                .sum::<f64>())
    .clamp(0.0, 1.0);

    let mut sp = 0.0_f64;
    let mut sq = 0.0_f64;
    for k in 0..k_count {
        let sk = r_k[k] * purity[k];
        sp += sk * p[k];
        sq += sk * q[k];
    }

    let mut caf = 0.0_f64;
    let mut caf_per_cluster = if include_detail {
        vec![0.0_f64; k_count]
    } else {
        Vec::new()
    };
    if sp > 1e-15 && sq > 1e-15 {
        let s_min = sp.min(sq);
        let scale_p = s_min / sp;
        let scale_q = s_min / sq;
        for k in 0..k_count {
            let sk = r_k[k] * purity[k];
            let term = sk * (scale_p * p[k]).min(scale_q * q[k]);
            caf += term;
            if include_detail && p[k] > 0.0 {
                caf_per_cluster[k] = (term / p[k]).clamp(0.0, 1.0);
            }
        }
    }
    let caf = caf.clamp(0.0, 1.0);

    let report = ClusterAreaReport {
        caf,
        purity_adj,
        prop_vis,
        res_density,
        res_deconv,
        res_combined,
        spatial_coherence,
        in_frac: ctx.in_frac,
        effective_boundary_r95: ctx.effective_b,
        g_eff: ctx.g_eff,
        num_clusters: k_count,
        num_nodes: n,
    };
    if !include_detail {
        return ClusterAreaDetail {
            report,
            ..ClusterAreaDetail::default()
        };
    }

    ClusterAreaDetail {
        report,
        cluster_ids,
        counts,
        p,
        q,
        purity,
        resolution: r_k,
        res_density: e_k,
        res_deconv: d_k,
        caf_per_cluster,
        coherence,
    }
}

/// Non-generic `f64` core implementation keeping WASM binary footprint minimal.
#[allow(clippy::too_many_arguments)]
pub(super) fn cluster_area_core(
    xs: &[f64],
    ys: &[f64],
    cluster_ids: Vec<u32>,
    inv: Vec<u32>,
    config: &ClusterAreaConfig,
    node_weights: Option<&[f64]>,
    cluster_weights: Option<&[f64]>,
    include_detail: bool,
) -> CoreResult<ClusterAreaDetail> {
    let n = xs.len();
    let k_count = cluster_ids.len();

    let (counts, p, has_custom_weights) = compute_target_weights(
        &inv,
        k_count,
        config.stevens_gamma,
        node_weights,
        cluster_weights,
    );

    if n < 2 || k_count < 2 {
        return Ok(degenerate_detail(cluster_ids, counts, p, n, include_detail));
    }

    let Some(frame) = normalize_and_center_mec(xs, ys) else {
        return Ok(degenerate_detail(cluster_ids, counts, p, n, include_detail));
    };

    let ctx = build_global_density_grid(&frame, &inv, k_count, config);
    let uniform_gamma_one = !has_custom_weights && (config.stevens_gamma - 1.0).abs() == 0.0;

    let mut scratch = ClusterGridScratch::new(&ctx);
    let mut bfs_stamp = VisitStamp::new(ctx.s_sq);
    let mut stack = Vec::with_capacity(256);

    let mut e_k = vec![0.0_f64; k_count];
    let mut d_k = vec![0.0_f64; k_count];
    let mut r_k = vec![0.0_f64; k_count];
    let mut purity = vec![0.0_f64; k_count];
    let mut a_vis = vec![0.0_f64; k_count];
    let mut coherence = vec![0.0_f64; k_count];

    for k in 0..k_count {
        let nk = counts[k];
        let st = ctx.offsets[k];
        let en = ctx.offsets[k + 1];
        if nk == 0 || st == en {
            if nk <= 1 {
                d_k[k] = 1.0;
            }
            continue;
        }

        let metrics = splat_cluster_density(
            &ctx,
            &mut scratch,
            st,
            en,
            nk,
            n,
            p[k],
            uniform_gamma_one,
            config,
        );
        e_k[k] = metrics.e_k;
        d_k[k] = metrics.d_k;
        r_k[k] = metrics.r_k;
        purity[k] = metrics.purity;
        a_vis[k] = metrics.a_vis;

        if metrics.a_vis > 1e-15 && metrics.max_wk > 1e-15 {
            coherence[k] = cluster_active_coherence(
                &ctx,
                &scratch,
                &mut bfs_stamp,
                &mut stack,
                metrics.max_wk,
            );
        }
    }

    Ok(aggregate_caf_detail(
        &ctx,
        cluster_ids,
        counts,
        p,
        e_k,
        d_k,
        r_k,
        purity,
        a_vis,
        coherence,
        n,
        include_detail,
    ))
}
