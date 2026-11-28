use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, Layout2d, MetricSpace, RiemannianLayout,
    Scalar, VisitStamp,
};
use egraph_neighborhood::DogVaradhanLandmarks;

use super::config::CoarseBackboneConfig;
use super::geodesics::CoarseMetaEdges;
use super::overlap::{
    collect_disk_hinge_pairs, count_disk_overlaps, resolve_residual_disk_overlaps,
};

#[derive(Debug, Clone, Copy)]
struct CoarsePair {
    u: u32,
    v: u32,
    d: f64,
    w: f64,
    is_hinge: bool,
}

/// Self-contained 2D pairwise stress + one-sided hinge SGD runner over an exponential schedule.
fn run_coarse_pairwise_sgd<S: Scalar>(
    layout: &mut Layout2d<S>,
    pairs: &mut [CoarsePair],
    iterations: usize,
    step_scale: f64,
    seed: u64,
) -> CoreResult<()> {
    if layout.len() <= 1 || iterations == 0 || pairs.is_empty() {
        return Ok(());
    }
    let mut w_min = f64::INFINITY;
    let mut w_max = 0.0_f64;
    for p in pairs.iter() {
        if p.w.is_finite() && p.w > 0.0 {
            w_min = w_min.min(p.w);
            w_max = w_max.max(p.w);
        }
    }
    if !w_min.is_finite() || w_max <= 0.0 {
        return Ok(());
    }

    let eps = 0.01_f64;
    let eta_max = 1.0 / w_min;
    let eta_min = eps / w_max;
    if !eta_min.is_finite() || !eta_max.is_finite() || eta_min <= 0.0 || eta_max <= 0.0 {
        return Ok(());
    }
    let decay = if iterations <= 1 {
        0.0
    } else {
        (eta_max / eta_min).ln() / ((iterations - 1) as f64)
    };
    let mut rng = DeterministicRng::from_seed_u64(seed);

    for t in 0..iterations {
        let eta = step_scale * eta_max * (-decay * (t as f64)).exp();
        if !eta.is_finite() || eta <= 0.0 {
            continue;
        }
        rng.shuffle(pairs);
        for p in pairs.iter() {
            let mu = (eta * p.w).min(1.0);
            if !mu.is_finite() || mu <= 0.0 {
                continue;
            }
            let u = p.u as usize;
            let v = p.v as usize;
            if p.is_hinge {
                let dist = layout.distance(u, v);
                if !dist.is_finite() {
                    return Err(CoreError::numerical_breakdown(
                        "non-finite distance during coarse backbone SGD step",
                    ));
                }
                if dist >= p.d {
                    continue;
                }
            }
            let half_mu = 0.5 * mu;
            layout.move_pair_toward(u, v, p.d, half_mu, half_mu)?;
        }
    }
    layout.validate_finite()
}

fn coarse_connected_components<W>(cg: &CsrGraph<W>, n_l: usize) -> (Vec<usize>, Vec<Vec<usize>>) {
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
pub(super) fn optimize_coarse_stress_and_barrier<W, S: Scalar>(
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
        let mut spring_pairs = Vec::with_capacity(m_l + n_l * config.coarse_neg_per_node);

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
                spring_pairs.push(CoarsePair {
                    u: meta.edge_u[e],
                    v: meta.edge_v[e],
                    d: d_s.to_f64(),
                    w: w_s.to_f64(),
                    is_hinge: false,
                });
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
                        spring_pairs.push(CoarsePair {
                            u: neg_u[idx],
                            v: neg_v[idx],
                            d: d_s.to_f64(),
                            w: w_s.to_f64(),
                            is_hinge: false,
                        });
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

        for round in 0..n_rounds {
            let round_iters = base_iters + usize::from(round < rem_iters);
            if round_iters == 0 {
                continue;
            }
            let mut round_pairs = spring_pairs.clone();

            if config.mu_ov > 0.0 {
                let eff_mu = config.mu_ov * (1.0 + 2.0 * (round as f64));
                let h_pairs =
                    collect_disk_hinge_pairs(coarse_layout, radii, config.m_gap, &half_boxes)?;
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
                        round_pairs.push(CoarsePair {
                            u,
                            v,
                            d: d_bar_s.to_f64(),
                            w: w_s.to_f64(),
                            is_hinge: true,
                        });
                    }
                }
            }

            let step_scale = 1.0 - 0.7 * (round as f64) / (n_rounds.max(1) as f64);
            run_coarse_pairwise_sgd(
                coarse_layout,
                &mut round_pairs,
                round_iters,
                step_scale,
                config.seed.wrapping_add(round as u64),
            )?;
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
