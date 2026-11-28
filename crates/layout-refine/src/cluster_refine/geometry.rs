use egraph_core::{CoreResult, CsrGraph};
use egraph_spatial_trees::StaticKdTree;

pub(super) const GEOM_EPS: f64 = 1e-12;

pub(super) fn extract_unique_undirected_edges<W>(graph: &CsrGraph<W>) -> Vec<(u32, u32)> {
    let n = graph.num_nodes();
    let mut edges = Vec::new();
    if !graph.is_directed() {
        for u in 0..n {
            let u32_u = u as u32;
            for &v_u32 in graph.neighbors(u) {
                if u32_u < v_u32 {
                    edges.push((u32_u, v_u32));
                }
            }
        }
    } else {
        for u in 0..n {
            let u32_u = u as u32;
            for &v_u32 in graph.neighbors(u) {
                if u32_u != v_u32 {
                    edges.push(if u32_u < v_u32 {
                        (u32_u, v_u32)
                    } else {
                        (v_u32, u32_u)
                    });
                }
            }
        }
        edges.sort_unstable();
        edges.dedup();
    }
    edges
}

/// Maps arbitrary `u32` cluster labels into contiguous `0..k` IDs in first-appearance order.
pub(super) fn canonicalize_labels(labels: &[u32]) -> (Vec<u32>, usize) {
    let n = labels.len();
    if n == 0 {
        return (Vec::new(), 0);
    }
    let max_label = labels.iter().copied().max().unwrap_or(0) as usize;
    let mut out = vec![0u32; n];
    let fast_limit = n.max(64).saturating_mul(4).min(1_000_000);
    if max_label < fast_limit {
        let mut remap = vec![u32::MAX; max_label + 1];
        let mut next_id = 0u32;
        for (i, &lbl) in labels.iter().enumerate() {
            let idx = lbl as usize;
            if remap[idx] == u32::MAX {
                remap[idx] = next_id;
                next_id += 1;
            }
            out[i] = remap[idx];
        }
        return (out, next_id as usize);
    }

    let mut pairs: Vec<(u32, u32)> = labels
        .iter()
        .enumerate()
        .map(|(i, &lbl)| (lbl, i as u32))
        .collect();
    pairs.sort_unstable();

    let mut groups: Vec<(u32, usize, usize)> = Vec::new();
    let mut start = 0usize;
    while start < n {
        let key = pairs[start].0;
        let mut end = start + 1;
        while end < n && pairs[end].0 == key {
            end += 1;
        }
        groups.push((pairs[start].1, start, end));
        start = end;
    }
    groups.sort_unstable_by_key(|&(first_node, _, _)| first_node);

    let k = groups.len();
    for (cid, &(_, g_start, g_end)) in groups.iter().enumerate() {
        let cid_u32 = cid as u32;
        for &(_, node_idx) in &pairs[g_start..g_end] {
            out[node_idx as usize] = cid_u32;
        }
    }
    (out, k)
}

pub(super) fn compute_cluster_centroids(
    xs: &[f64],
    ys: &[f64],
    labels: &[u32],
    k: usize,
) -> Vec<(f64, f64)> {
    let mut sum_x = vec![0.0_f64; k];
    let mut sum_y = vec![0.0_f64; k];
    let mut counts = vec![0usize; k];
    for (i, &lbl) in labels.iter().enumerate() {
        let c = lbl as usize;
        sum_x[c] += xs[i];
        sum_y[c] += ys[i];
        counts[c] += 1;
    }
    let mut centroids = Vec::with_capacity(k);
    for c in 0..k {
        if counts[c] > 0 {
            let inv = 1.0 / (counts[c] as f64);
            centroids.push((sum_x[c] * inv, sum_y[c] * inv));
        } else {
            centroids.push((0.0, 0.0));
        }
    }
    centroids
}

/// Computes the exact point where segment $[\mu_m, \mu_j]$ exits the Voronoi cell of $\mu_m$,
/// falling back to the midpoint $\frac{1}{2}(\mu_m + \mu_j)$ if degenerate.
pub(super) fn voronoi_cell_exit(centroids: &[(f64, f64)], m: usize, j: usize) -> (f64, f64) {
    let mu_m = centroids[m];
    let mu_j = centroids[j];
    let vx = mu_j.0 - mu_m.0;
    let vy = mu_j.1 - mu_m.1;
    let v_sq = vx * vx + vy * vy;
    if v_sq <= GEOM_EPS * GEOM_EPS {
        return (0.5 * (mu_m.0 + mu_j.0), 0.5 * (mu_m.1 + mu_j.1));
    }

    // Half-plane for site k != m: <(mu_m + t*v) - (mu_m + mu_k)/2, mu_k - mu_m> <= 0
    // <=> t * <v, d_k> <= 0.5 * ||d_k||^2, where d_k = mu_k - mu_m.
    let mut t_exit = 0.5_f64;
    for (c_idx, &mu_k) in centroids.iter().enumerate() {
        if c_idx == m {
            continue;
        }
        let dkx = mu_k.0 - mu_m.0;
        let dky = mu_k.1 - mu_m.1;
        let dk_sq = dkx * dkx + dky * dky;
        let dot = vx * dkx + vy * dky;
        if dot > GEOM_EPS && dk_sq > 0.0 {
            let t_k = (0.5 * dk_sq) / dot;
            if t_k >= 0.0 && t_k < t_exit {
                t_exit = t_k;
            }
        }
    }

    if !t_exit.is_finite() || t_exit <= 0.0 {
        t_exit = 0.5;
    }
    (mu_m.0 + t_exit * vx, mu_m.1 + t_exit * vy)
}

/// Computes $\ell$ for `Cluster-HAC`: mean graph edge length in the current layout when edges
/// are present, otherwise mean 1-NN Euclidean distance (falling back to layout scale).
pub(super) fn compute_hac_step_scale(
    xs: &[f64],
    ys: &[f64],
    edges: &[(u32, u32)],
) -> CoreResult<f64> {
    if !edges.is_empty() {
        let mut sum = 0.0_f64;
        for &(u, v) in edges {
            let ui = u as usize;
            let vi = v as usize;
            sum += (xs[ui] - xs[vi]).hypot(ys[ui] - ys[vi]);
        }
        let mean_edge = sum / (edges.len() as f64);
        if mean_edge > GEOM_EPS && mean_edge.is_finite() {
            return Ok(mean_edge);
        }
    }

    let n = xs.len();
    if n >= 2 {
        let mut pts = Vec::with_capacity(n);
        for i in 0..n {
            pts.push([xs[i], ys[i]]);
        }
        let tree = StaticKdTree::<2, f64>::build(&pts)?;
        let mut nn_sum = 0.0_f64;
        let mut nn_cnt = 0usize;
        for i in 0..n {
            if let Some((_, d2)) = tree.nearest_excluding_self(i)? {
                nn_sum += d2.sqrt();
                nn_cnt += 1;
            }
        }
        if nn_cnt > 0 {
            let mean_nn = nn_sum / (nn_cnt as f64);
            if mean_nn > GEOM_EPS && mean_nn.is_finite() {
                return Ok(mean_nn);
            }
        }
    }

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for i in 0..n {
        min_x = min_x.min(xs[i]);
        max_x = max_x.max(xs[i]);
        min_y = min_y.min(ys[i]);
        max_y = max_y.max(ys[i]);
    }
    let diag = (max_x - min_x).hypot(max_y - min_y);
    Ok(if diag > GEOM_EPS { diag * 0.1 } else { 1.0 })
}

#[inline]
pub(super) fn deterministic_unit_vector(seed: u64, iter: usize, u: usize, w: usize) -> (f64, f64) {
    let mut z = seed
        .wrapping_add((iter as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add((u as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add((w as u64).wrapping_mul(0x94D0_49BB_1331_11EB));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let frac = ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
    let angle = frac * std::f64::consts::TAU;
    (angle.cos(), angle.sin())
}
