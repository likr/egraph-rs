use egraph_core::{CoreError, CoreResult};

use super::geometry::canonicalize_labels;

/// Exact $O(N^2)$ nearest-neighbor-chain UPGMA (average-linkage) clustering cut at `k` clusters.
pub(super) fn upgma_hac_2d(xs: &[f64], ys: &[f64], k: usize) -> CoreResult<Vec<u32>> {
    let n = xs.len();
    if k == 0 || k > n {
        return Err(CoreError::invalid_input(format!(
            "upgma_hac_2d requires 1 <= k <= n, got k={k}, n={n}"
        )));
    }
    if k == n {
        return Ok((0..n as u32).collect());
    }
    if k == 1 {
        return Ok(vec![0u32; n]);
    }

    let mat_len = n
        .checked_mul(n)
        .ok_or_else(|| CoreError::invalid_input("HAC distance matrix size overflow"))?;
    let mut dist = vec![0.0_f64; mat_len];
    for i in 0..n {
        let xi = xs[i];
        let yi = ys[i];
        for j in (i + 1)..n {
            let d = (xi - xs[j]).hypot(ys[j] - yi);
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }

    let mut active = vec![true; n];
    let mut sizes = vec![1usize; n];
    let mut min_node: Vec<usize> = (0..n).collect();
    let member_head = (0..n).collect::<Vec<usize>>();
    let mut member_tail = (0..n).collect::<Vec<usize>>();
    let mut member_next = vec![usize::MAX; n];
    let mut chain: Vec<usize> = Vec::with_capacity(n);
    let mut remaining = n;

    while remaining > k {
        if chain.is_empty() {
            let first = active
                .iter()
                .position(|&a| a)
                .ok_or_else(|| CoreError::invalid_input("HAC internal state exhausted"))?;
            chain.push(first);
        }

        let u = match chain.last() {
            Some(&top) => top,
            None => break,
        };
        let prev = if chain.len() >= 2 {
            Some(chain[chain.len() - 2])
        } else {
            None
        };

        // Find nearest active cluster to `u` with deterministic tie-breaking by `min_node`.
        let mut best_v = usize::MAX;
        let mut best_d = f64::INFINITY;
        let mut best_min = usize::MAX;
        let row_off = u * n;
        for v in 0..n {
            if !active[v] || v == u {
                continue;
            }
            let d = dist[row_off + v];
            let mv = min_node[v];
            if d < best_d || (d == best_d && mv < best_min) {
                best_d = d;
                best_v = v;
                best_min = mv;
            }
        }

        if best_v == usize::MAX {
            break;
        }

        // Check reciprocal nearest neighbor: either `best_v == prev`, or `prev` is active and
        // tied in distance with `best_v` (exact equilateral tie check).
        let is_reciprocal =
            prev == Some(best_v) || prev.is_some_and(|p| active[p] && dist[row_off + p] == best_d);

        if is_reciprocal {
            let Some(p) = prev else {
                break;
            };
            chain.pop();
            chain.pop();

            // Merge `u` and `p` into the slot with smaller index `a` (deactivating `b`).
            let (a, b) = if u < p { (u, p) } else { (p, u) };
            let sa = sizes[a] as f64;
            let sb = sizes[b] as f64;
            let sab = sa + sb;

            // Lance-Williams average-linkage update:
            // d(A U B, C) = (|A| * d(A, C) + |B| * d(B, C)) / (|A| + |B|).
            for c in 0..n {
                if !active[c] || c == a || c == b {
                    continue;
                }
                let d_new = (sa * dist[a * n + c] + sb * dist[b * n + c]) / sab;
                dist[a * n + c] = d_new;
                dist[c * n + a] = d_new;
            }

            sizes[a] += sizes[b];
            min_node[a] = min_node[a].min(min_node[b]);
            member_next[member_tail[a]] = member_head[b];
            member_tail[a] = member_tail[b];
            active[b] = false;
            remaining -= 1;
        } else {
            chain.push(best_v);
        }
    }

    let mut raw_labels = vec![0u32; n];
    let mut cid = 0u32;
    for c in 0..n {
        if !active[c] {
            continue;
        }
        let mut cur = member_head[c];
        while cur != usize::MAX {
            raw_labels[cur] = cid;
            cur = member_next[cur];
        }
        cid += 1;
    }

    let (canon, _) = canonicalize_labels(&raw_labels);
    Ok(canon)
}

/// Computes exact maximum-weight bipartite matching over the $K \times K$ Jaccard matrix
/// between ground-truth clusters (`gt_labels`) and geometric clusters (`geom_labels`).
pub(super) fn match_clusters_hungarian(
    gt_labels: &[u32],
    geom_labels: &[u32],
    k: usize,
) -> Vec<usize> {
    let n = gt_labels.len();
    let mut inter = vec![0usize; k * k];
    let mut gt_sizes = vec![0usize; k];
    let mut geom_sizes = vec![0usize; k];

    for idx in 0..n {
        let i = gt_labels[idx] as usize;
        let j = geom_labels[idx] as usize;
        gt_sizes[i] += 1;
        geom_sizes[j] += 1;
        inter[i * k + j] += 1;
    }

    let mut cost = vec![1.0_f64; k * k];
    for i in 0..k {
        for j in 0..k {
            let ov = inter[i * k + j];
            let uni = gt_sizes[i] + geom_sizes[j] - ov;
            let js = if uni > 0 {
                (ov as f64) / (uni as f64)
            } else {
                0.0
            };
            cost[i * k + j] = 1.0 - js;
        }
    }

    // 1-indexed O(K^3) Hungarian algorithm minimizing sum_i cost[i, match[i]].
    let mut u = vec![0.0_f64; k + 1];
    let mut v = vec![0.0_f64; k + 1];
    let mut p = vec![0usize; k + 1];
    let mut way = vec![0usize; k + 1];
    let mut minv = vec![0.0_f64; k + 1];
    let mut used = vec![false; k + 1];

    for i in 1..=k {
        p[0] = i;
        let mut j0 = 0usize;
        minv.fill(f64::INFINITY);
        used.fill(false);

        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;
            for j in 1..=k {
                if !used[j] {
                    let cur = cost[(i0 - 1) * k + (j - 1)] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=k {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }

        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut row_to_col = vec![0usize; k];
    for j in 1..=k {
        if p[j] > 0 {
            row_to_col[p[j] - 1] = j - 1;
        }
    }
    row_to_col
}
