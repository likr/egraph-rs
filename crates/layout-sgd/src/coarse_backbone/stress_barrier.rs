use egraph_core::{
    CoreResult, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Scalar, VisitStamp,
};
use egraph_neighborhood::DogVaradhanLandmarks;
#[cfg(not(feature = "delaunay"))]
use egraph_spatial_trees::KdQueryScratch;

use crate::sgd::{SgdScheduleConfig, SgdScheduleKind, SgdSolver};

use super::config::CoarseBackboneConfig;
use super::geodesics::CoarseMetaEdges;
#[cfg(not(feature = "delaunay"))]
use super::overlap::K_PROX_FALLBACK;
use super::overlap::{
    collect_disk_hinge_pairs, count_disk_overlaps, resolve_residual_disk_overlaps,
};

fn coarse_connected_components<S: Scalar>(
    cg: &CsrGraph<S>,
    n_l: usize,
) -> (Vec<usize>, Vec<Vec<usize>>) {
    let mut comp_id = vec![usize::MAX; n_l];
    let mut comp_nodes: Vec<Vec<usize>> = Vec::new();
    let cg_indptr = cg.indptr();
    let cg_indices = cg.indices();
    let mut queue = Vec::new();

    for start in 0..n_l {
        if comp_id[start] != usize::MAX {
            continue;
        }
        let cid = comp_nodes.len();
        comp_id[start] = cid;
        queue.clear();
        queue.push(start);
        let mut head = 0usize;
        while head < queue.len() {
            let u = queue[head];
            head += 1;
            let s = cg_indptr[u] as usize;
            let e = cg_indptr[u + 1] as usize;
            for &nbr in &cg_indices[s..e] {
                let v = nbr as usize;
                if comp_id[v] == usize::MAX {
                    comp_id[v] = cid;
                    queue.push(v);
                }
            }
        }
        queue.sort_unstable();
        comp_nodes.push(queue.clone());
    }

    (comp_id, comp_nodes)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn optimize_coarse_stress_and_barrier<W: Scalar, S: Scalar>(
    cg: &CsrGraph<W>,
    coarse_sizes: &[u32],
    n_l: usize,
    meta: &CoarseMetaEdges,
    landmarks: &DogVaradhanLandmarks,
    radii: &[f64],
    config: &CoarseBackboneConfig,
    coarse_layout: &mut Layout2d<S>,
) -> CoreResult<(usize, usize)> {
    // Optimize mass-weighted stress and one-sided disk-barrier hinges via SGD.
    let m_l = meta.edge_u.len();
    let half_boxes: Vec<S> = radii
        .iter()
        .map(|&r| S::from_f64(r + 0.5 * config.m_gap))
        .collect();
    let coarse_overlap_pairs_initial =
        count_disk_overlaps(coarse_layout, radii, config.m_gap, &half_boxes)?;

    if config.coarse_t_max > 0 && (m_l > 0 || config.mu_ov > 0.0 || config.coarse_neg_per_node > 0)
    {
        let mut rng = DeterministicRng::from_seed_u64(config.seed);
        let mut spring_src = Vec::with_capacity(m_l + n_l * config.coarse_neg_per_node);
        let mut spring_tgt = Vec::with_capacity(m_l + n_l * config.coarse_neg_per_node);
        let mut spring_dist: Vec<S> = Vec::with_capacity(m_l + n_l * config.coarse_neg_per_node);
        let mut spring_wt: Vec<S> = Vec::with_capacity(m_l + n_l * config.coarse_neg_per_node);

        let edge_alts = if m_l > 0 {
            landmarks.alt_pair_distances(&meta.edge_u, &meta.edge_v)?
        } else {
            Vec::new()
        };
        for e in 0..m_l {
            let u = meta.edge_u[e] as usize;
            let v = meta.edge_v[e] as usize;
            let d = edge_alts[e].max(meta.l_edge[e]).max(1e-6);
            let sz_u = coarse_sizes[u] as f64;
            let sz_v = coarse_sizes[v] as f64;
            let mass = meta.edge_w[e].max(1e-12).powf(config.gamma_w)
                * (sz_u * sz_v).powf(0.5 * config.gamma_m);
            let w = mass / (d * d);
            let d_s = S::from_f64(d);
            let w_s = S::from_f64(w);
            if d_s.is_finite() && d_s > S::ZERO && w_s.is_finite() && w_s > S::ZERO {
                spring_src.push(meta.edge_u[e]);
                spring_tgt.push(meta.edge_v[e]);
                spring_dist.push(d_s);
                spring_wt.push(w_s);
            }
        }

        // Sample canonical non-edge coarse pairs only within the same connected component.
        if m_l > 0 && config.coarse_neg_per_node > 0 && n_l > 2 {
            let (comp_id, comp_nodes) = coarse_connected_components(cg, n_l);
            let mut neg_pairs: Vec<(u32, u32)> =
                Vec::with_capacity(n_l * config.coarse_neg_per_node);
            let mut stamp = VisitStamp::new(n_l);
            let cg_indptr = cg.indptr();
            let cg_indices = cg.indices();
            for u in 0..n_l {
                let members = &comp_nodes[comp_id[u]];
                if members.len() <= 2 {
                    continue;
                }
                stamp.clear();
                stamp.mark(u);
                for &nbr in &cg_indices[cg_indptr[u] as usize..cg_indptr[u + 1] as usize] {
                    stamp.mark(nbr as usize);
                }
                let mut collected = 0usize;
                let max_trials = config.coarse_neg_per_node * 6;
                for _ in 0..max_trials {
                    if collected >= config.coarse_neg_per_node {
                        break;
                    }
                    let v = members[rng.gen_range_usize(members.len())];
                    if comp_id[v] != comp_id[u] || !stamp.visit(v) {
                        continue;
                    }
                    let (a, b) = if u < v { (u, v) } else { (v, u) };
                    neg_pairs.push((a as u32, b as u32));
                    collected += 1;
                }
            }
            neg_pairs.sort_unstable();
            neg_pairs.dedup();
            if !neg_pairs.is_empty() {
                let neg_u: Vec<u32> = neg_pairs.iter().map(|&(a, _)| a).collect();
                let neg_v: Vec<u32> = neg_pairs.iter().map(|&(_, b)| b).collect();
                let neg_alts = landmarks.alt_pair_distances(&neg_u, &neg_v)?;
                let floor_d = (1e-3 * meta.median_l_edge.max(1.0)).max(1e-6);
                for idx in 0..neg_pairs.len() {
                    let u = neg_u[idx] as usize;
                    let v = neg_v[idx] as usize;
                    let min_clear = radii[u] + radii[v] + config.m_gap;
                    let d = neg_alts[idx].max(floor_d).max(min_clear);
                    let sz_u = coarse_sizes[u] as f64;
                    let sz_v = coarse_sizes[v] as f64;
                    let mass = (sz_u * sz_v).powf(0.5 * config.gamma_m);
                    let w = mass / (d * d);
                    let d_s = S::from_f64(d);
                    let w_s = S::from_f64(w);
                    if d_s.is_finite() && d_s > S::ZERO && w_s.is_finite() && w_s > S::ZERO {
                        spring_src.push(neg_u[idx]);
                        spring_tgt.push(neg_v[idx]);
                        spring_dist.push(d_s);
                        spring_wt.push(w_s);
                    }
                }
            }
        }

        let n_rounds = if config.mu_ov > 0.0 && config.coarse_t_max >= 4 {
            4usize
        } else {
            1usize
        };
        let base_iters = config.coarse_t_max / n_rounds;
        let rem_iters = config.coarse_t_max % n_rounds;

        #[cfg(not(feature = "delaunay"))]
        let mut kd_scratch = KdQueryScratch::with_capacity(K_PROX_FALLBACK);
        #[cfg(not(feature = "delaunay"))]
        let mut knn_out = Vec::new();

        for round in 0..n_rounds {
            let round_iters = base_iters + usize::from(round < rem_iters);
            if round_iters == 0 {
                continue;
            }
            let mut hinge_src = Vec::new();
            let mut hinge_tgt = Vec::new();
            let mut hinge_dist: Vec<S> = Vec::new();
            let mut hinge_wt: Vec<S> = Vec::new();

            if config.mu_ov > 0.0 {
                let eff_mu = config.mu_ov * (1.0 + 2.0 * (round as f64));
                let h_pairs = collect_disk_hinge_pairs(
                    coarse_layout,
                    radii,
                    config.m_gap,
                    &half_boxes,
                    #[cfg(not(feature = "delaunay"))]
                    &mut kd_scratch,
                    #[cfg(not(feature = "delaunay"))]
                    &mut knn_out,
                )?;
                for (u, v) in h_pairs {
                    let ui = u as usize;
                    let vi = v as usize;
                    let d_bar = radii[ui] + radii[vi] + config.m_gap;
                    if !d_bar.is_finite() || d_bar <= 0.0 {
                        continue;
                    }
                    let sz_u = coarse_sizes[ui] as f64;
                    let sz_v = coarse_sizes[vi] as f64;
                    let mass = (sz_u * sz_v).powf(0.5 * config.gamma_m);
                    let denom = d_bar.max(1e-6);
                    let w = eff_mu * mass / (denom * denom);
                    let d_bar_s = S::from_f64(d_bar);
                    let w_s = S::from_f64(w);
                    if d_bar_s.is_finite() && d_bar_s > S::ZERO && w_s.is_finite() && w_s > S::ZERO
                    {
                        hinge_src.push(u);
                        hinge_tgt.push(v);
                        hinge_dist.push(d_bar_s);
                        hinge_wt.push(w_s);
                    }
                }
            }

            let step_scale = 1.0 - 0.7 * (round as f64) / (n_rounds.max(1) as f64);
            let sched = SgdScheduleConfig {
                t_max: round_iters,
                eps: 0.01,
                kind: SgdScheduleKind::Exponential,
                step_scale,
            };
            let mut solver = SgdSolver::<S>::from_pairs_with_hinges(
                n_l,
                spring_src.clone(),
                spring_tgt.clone(),
                spring_dist.clone(),
                spring_dist.clone(),
                spring_wt.clone(),
                spring_wt.clone(),
                hinge_src,
                hinge_tgt,
                hinge_dist.clone(),
                hinge_dist,
                hinge_wt.clone(),
                hinge_wt,
                sched,
                config.seed.wrapping_add(round as u64),
            )?;
            solver.run(coarse_layout)?;
        }

        if config.mu_ov > 0.0
            && n_l > 1
            && count_disk_overlaps(coarse_layout, radii, config.m_gap, &half_boxes)? > 0
        {
            resolve_residual_disk_overlaps(
                coarse_layout,
                radii,
                config.m_gap,
                &half_boxes,
                coarse_sizes,
            )?;
        }
    }

    let coarse_overlap_pairs_final =
        count_disk_overlaps(coarse_layout, radii, config.m_gap, &half_boxes)?;
    Ok((coarse_overlap_pairs_initial, coarse_overlap_pairs_final))
}
