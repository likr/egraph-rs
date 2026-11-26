use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng};
use egraph_linalg::{BfsWorkspace, DijkstraWorkspace};

use super::graph_sampling::compute_mssp_dists;

pub(super) fn select_weighted_maxmin_sp_pivots(
    graph: &CsrGraph<f64>,
    num_pivots: usize,
    seed: u64,
    min_degree: usize,
) -> CoreResult<Vec<u32>> {
    let n = graph.num_nodes();
    if n == 0 || num_pivots == 0 {
        return Ok(Vec::new());
    }
    let target = num_pivots.min(n);
    if target == n {
        return Ok((0..n as u32).collect());
    }

    let mut core_count = 0usize;
    let mut non_iso_count = 0usize;
    for u in 0..n {
        let deg = graph.degree(u);
        if deg >= min_degree {
            core_count += 1;
        }
        if deg > 0 {
            non_iso_count += 1;
        }
    }

    let mut rng = DeterministicRng::from_seed_u64(seed);
    let first_pivot = if core_count > 0 {
        let pick = rng.gen_range_usize(core_count);
        (0..n)
            .filter(|&u| graph.degree(u) >= min_degree)
            .nth(pick)
            .ok_or_else(|| CoreError::numerical_breakdown("failed to pick first core pivot"))?
            as u32
    } else if non_iso_count > 0 {
        let pick = rng.gen_range_usize(non_iso_count);
        (0..n)
            .filter(|&u| graph.degree(u) > 0)
            .nth(pick)
            .ok_or_else(|| CoreError::numerical_breakdown("failed to pick first non-iso pivot"))?
            as u32
    } else {
        rng.gen_range_usize(n) as u32
    };

    let mut pivots = Vec::with_capacity(target);
    let mut chosen = vec![false; n];
    let mut min_dist = vec![f64::INFINITY; n];
    let mut dijk_dist = vec![f64::INFINITY; n];
    let mut dijk = DijkstraWorkspace::new(n);

    pivots.push(first_pivot);
    chosen[first_pivot as usize] = true;
    let mut last = first_pivot as usize;

    for _ in 1..target {
        dijk.dijkstra_from(graph, last, &mut dijk_dist)?;

        let mut best_core: Option<(usize, f64)> = None;
        let mut best_non_iso: Option<(usize, f64)> = None;
        let mut best_any: Option<(usize, f64)> = None;

        for u in 0..n {
            let d = min_dist[u].min(dijk_dist[u]);
            min_dist[u] = d;
            if chosen[u] {
                continue;
            }
            let deg = graph.degree(u);
            if deg >= min_degree && best_core.is_none_or(|(_, bd)| d.total_cmp(&bd).is_gt()) {
                best_core = Some((u, d));
            }
            if deg > 0 && best_non_iso.is_none_or(|(_, bd)| d.total_cmp(&bd).is_gt()) {
                best_non_iso = Some((u, d));
            }
            if best_any.is_none_or(|(_, bd)| d.total_cmp(&bd).is_gt()) {
                best_any = Some((u, d));
            }
        }

        let next_pivot = best_core
            .or(best_non_iso)
            .or(best_any)
            .ok_or_else(|| {
                CoreError::numerical_breakdown("no unchosen vertex available for MaxMin pivot")
            })?
            .0 as u32;
        chosen[next_pivot as usize] = true;
        pivots.push(next_pivot);
        last = next_pivot as usize;
    }

    Ok(pivots)
}

pub(super) fn select_maxmin_random_sp_pivots(
    graph: &CsrGraph<f64>,
    target_k: usize,
    seed: u64,
) -> CoreResult<(Vec<u32>, Vec<f64>)> {
    let n = graph.num_nodes();
    if target_k == 0 || n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    if target_k >= n {
        let pivots: Vec<u32> = (0..n as u32).collect();
        let dists = compute_mssp_dists(graph, &pivots)?;
        return Ok((pivots, dists));
    }

    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut pivots = Vec::with_capacity(target_k);
    let mut chosen = vec![false; n];
    let mut piv_dists = vec![f64::INFINITY; target_k * n];
    let mut min_d = vec![f64::INFINITY; n];
    let weighted = graph.is_weighted();
    let mut bfs = (!weighted).then(|| BfsWorkspace::new(n));
    let mut dijk = weighted.then(|| DijkstraWorkspace::new(n));

    let first = rng.gen_range_usize(n);
    pivots.push(first as u32);
    chosen[first] = true;
    if let Some(ws) = bfs.as_mut() {
        ws.bfs_from(graph, first, &mut piv_dists[0..n])?;
    } else if let Some(ws) = dijk.as_mut() {
        ws.dijkstra_from(graph, first, &mut piv_dists[0..n])?;
    }
    min_d.copy_from_slice(&piv_dists[0..n]);

    for k in 1..target_k {
        let mut inf_count = 0usize;
        let mut finite_sum = 0.0_f64;
        let mut last_unchosen = None;
        for u in 0..n {
            if chosen[u] {
                continue;
            }
            last_unchosen = Some(u);
            if !min_d[u].is_finite() {
                inf_count += 1;
            } else if min_d[u] > 0.0 {
                finite_sum += min_d[u];
            }
        }
        let next_u = if inf_count > 0 {
            let pick = rng.gen_range_usize(inf_count);
            let mut seen = 0usize;
            let mut chosen_u = last_unchosen.unwrap_or(0);
            for u in 0..n {
                if !chosen[u] && !min_d[u].is_finite() {
                    if seen == pick {
                        chosen_u = u;
                        break;
                    }
                    seen += 1;
                }
            }
            chosen_u
        } else if finite_sum.is_finite() && finite_sum > 0.0 {
            let threshold = rng.next_f64() * finite_sum;
            let mut acc = 0.0_f64;
            let mut picked = last_unchosen.unwrap_or(0);
            for u in 0..n {
                if !chosen[u] && min_d[u].is_finite() && min_d[u] > 0.0 {
                    acc += min_d[u];
                    picked = u;
                    if acc >= threshold {
                        break;
                    }
                }
            }
            picked
        } else {
            let unchosen_total = n - k;
            let pick = rng.gen_range_usize(unchosen_total);
            let mut seen = 0usize;
            let mut chosen_u = last_unchosen.unwrap_or(0);
            for (u, &is_chosen) in chosen.iter().enumerate().take(n) {
                if !is_chosen {
                    if seen == pick {
                        chosen_u = u;
                        break;
                    }
                    seen += 1;
                }
            }
            chosen_u
        };

        chosen[next_u] = true;
        pivots.push(next_u as u32);
        let row = &mut piv_dists[k * n..(k + 1) * n];
        if let Some(ws) = bfs.as_mut() {
            ws.bfs_from(graph, next_u, row)?;
        } else if let Some(ws) = dijk.as_mut() {
            ws.dijkstra_from(graph, next_u, row)?;
        }
        for u in 0..n {
            min_d[u] = min_d[u].min(row[u]);
        }
    }

    Ok((pivots, piv_dists))
}
