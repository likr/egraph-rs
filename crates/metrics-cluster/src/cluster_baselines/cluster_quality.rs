use std::cmp::Ordering;

use egraph_clustering::{KMeansConfig, kmeans_plus_plus};
use egraph_core::{CoreError, CoreResult, Layout2d, Scalar};

use super::agreement::partition_agreement;
use super::{PartitionAgreement, compact_labels, validate_layout_labels};

fn normalized_coords_for_cq<S: Scalar>(layout: &Layout2d<S>) -> Option<Vec<f64>> {
    let n = layout.len();
    if n < 2 {
        return None;
    }
    let xs = layout.x();
    let ys = layout.y();
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    for i in 0..n {
        let x = xs[i].to_f64();
        let y = ys[i].to_f64();
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
        sum_x += x;
        sum_y += y;
    }
    let span = (max_x - min_x).max(max_y - min_y);
    if span <= 1e-15 {
        return None;
    }
    let mx = sum_x / n as f64;
    let my = sum_y / n as f64;
    let mut sum_sq = 0.0_f64;
    for i in 0..n {
        let dx = xs[i].to_f64() - mx;
        let dy = ys[i].to_f64() - my;
        sum_sq += dx * dx + dy * dy;
    }
    let rms = (sum_sq / n as f64).sqrt();
    if rms <= 1e-15 {
        return None;
    }
    let inv_rms = 1.0 / rms;
    let mut flat = Vec::with_capacity(n * 2);
    for i in 0..n {
        flat.push((xs[i].to_f64() - mx) * inv_rms);
        flat.push((ys[i].to_f64() - my) * inv_rms);
    }
    Some(flat)
}

/// Evaluates Cluster Quality via $k$-means++ clustering on centered, RMS-normalized 2D
/// layout coordinates (`CQ-kmeans`), returning [`PartitionAgreement`] against `labels`.
///
/// Returns [`PartitionAgreement::default()`] (all zeros) on degenerate cases (`n < 2`, `k < 2`,
/// `k >= n`, or collapsed coordinates with `span <= 1e-15` or `rms <= 1e-15`).
pub fn cq_kmeans<S: Scalar>(
    layout: &Layout2d<S>,
    labels: &[u32],
    n_init: usize,
    max_iters: usize,
    seed: u64,
) -> CoreResult<PartitionAgreement> {
    if n_init == 0 {
        return Err(CoreError::invalid_input("cq_kmeans requires n_init >= 1"));
    }
    if max_iters == 0 {
        return Err(CoreError::invalid_input(
            "cq_kmeans requires max_iters >= 1",
        ));
    }
    validate_layout_labels(layout, labels, "cq_kmeans")?;

    let n = layout.len();
    let (_, _, k) = compact_labels(labels);
    if n < 2 || k < 2 || k >= n {
        return Ok(PartitionAgreement::default());
    }
    let Some(points_flat) = normalized_coords_for_cq(layout) else {
        return Ok(PartitionAgreement::default());
    };

    let cfg = KMeansConfig {
        k,
        max_iters,
        tol: 1e-4,
        n_init,
        seed,
    };
    let res = kmeans_plus_plus(&points_flat, n, 2, &cfg)?;
    partition_agreement(labels, &res.labels)
}

struct DisjointSet {
    parent: Vec<usize>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            let p = self.parent[x];
            self.parent[x] = self.parent[p];
            x = p;
        }
        x
    }

    fn union_into_parent(&mut self, a: usize, b: usize, parent: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        self.parent[ra] = parent;
        self.parent[rb] = parent;
    }
}

/// Evaluates Cluster Quality via exact $O(N^2)$ Murtagh NN-chain average-linkage (UPGMA)
/// hierarchical agglomerative clustering on centered, RMS-normalized 2D layout coordinates
/// (`CQ-HAC`), returning [`PartitionAgreement`] against `labels`.
///
/// Collects all $N - 1$ dendrogram merges, sorts by `(height, step)`, and cuts at `K` clusters
/// to match `scipy.cluster.hierarchy.linkage(method='average')` + `fcluster(criterion='maxclust')`.
/// Allocates an $O(N^2)$ symmetric distance matrix. Returns [`PartitionAgreement::default()`] on
/// degenerate cases (`n < 2`, `k < 2`, `k >= n`, or collapsed coordinates).
pub fn cq_hac<S: Scalar>(layout: &Layout2d<S>, labels: &[u32]) -> CoreResult<PartitionAgreement> {
    validate_layout_labels(layout, labels, "cq_hac")?;

    let n = layout.len();
    let (_, _, k) = compact_labels(labels);
    if n < 2 || k < 2 || k >= n {
        return Ok(PartitionAgreement::default());
    }
    let Some(points_flat) = normalized_coords_for_cq(layout) else {
        return Ok(PartitionAgreement::default());
    };

    let matrix_len = n
        .checked_mul(n)
        .ok_or_else(|| CoreError::invalid_input("cq_hac N x N matrix size overflow"))?;
    let mut dist = vec![0.0_f64; matrix_len];
    for i in 0..n {
        let xi = points_flat[2 * i];
        let yi = points_flat[2 * i + 1];
        for j in (i + 1)..n {
            let d = (xi - points_flat[2 * j]).hypot(yi - points_flat[2 * j + 1]);
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }

    let mut active_slots: Vec<usize> = (0..n).collect();
    let mut size = vec![1usize; n];
    let mut node_id: Vec<usize> = (0..n).collect();
    let mut slot_height = vec![0.0_f64; n];
    let mut chain: Vec<usize> = Vec::with_capacity(n);
    let mut merges: Vec<(f64, usize, usize, usize)> = Vec::with_capacity(n - 1);

    while merges.len() < n - 1 {
        if chain.is_empty() {
            chain.push(active_slots[0]);
        }
        let cur = *chain.last().expect("chain is non-empty");
        let prev = if chain.len() >= 2 {
            Some(chain[chain.len() - 2])
        } else {
            None
        };

        let mut best_slot = usize::MAX;
        let mut best_dist = f64::INFINITY;
        let mut best_nid = usize::MAX;
        for &cand in &active_slots {
            if cand == cur {
                continue;
            }
            let d = dist[cur * n + cand];
            let cmp = d.total_cmp(&best_dist);
            if cmp == Ordering::Less || (cmp == Ordering::Equal && node_id[cand] < best_nid) {
                best_dist = d;
                best_slot = cand;
                best_nid = node_id[cand];
            }
        }

        // Prefer `prev` on exact distance tie to resolve reciprocal nearest neighbors cleanly.
        let nn = if let Some(p) = prev
            && dist[cur * n + p].total_cmp(&best_dist) == Ordering::Equal
        {
            p
        } else {
            best_slot
        };

        if Some(nn) == prev {
            chain.pop();
            chain.pop();
            let p = nn;
            let h = dist[cur * n + p].max(slot_height[p]).max(slot_height[cur]);
            let id_a = node_id[p];
            let id_b = node_id[cur];
            let (left, right) = if id_a < id_b {
                (id_a, id_b)
            } else {
                (id_b, id_a)
            };
            let new_id = n + merges.len();
            merges.push((h, left, right, new_id));
            slot_height[p] = h;

            let sa = size[p] as f64;
            let sb = size[cur] as f64;
            let denom = sa + sb;
            for &other in &active_slots {
                if other != p && other != cur {
                    let d_new = (sa * dist[p * n + other] + sb * dist[cur * n + other]) / denom;
                    dist[p * n + other] = d_new;
                    dist[other * n + p] = d_new;
                }
            }
            size[p] += size[cur];
            node_id[p] = new_id;
            if let Some(pos) = active_slots.iter().position(|&x| x == cur) {
                active_slots.swap_remove(pos);
            }
        } else {
            chain.push(nn);
        }
    }

    merges.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.3.cmp(&b.3)));

    let mut dsu = DisjointSet::new(2 * n - 1);
    for &(_, left, right, parent) in &merges[..n - k] {
        dsu.union_into_parent(left, right, parent);
    }

    let mut root_to_label = vec![u32::MAX; 2 * n - 1];
    let mut next_label = 0u32;
    let mut pred_labels = vec![0u32; n];
    for (i, slot) in pred_labels.iter_mut().enumerate() {
        let r = dsu.find(i);
        if root_to_label[r] == u32::MAX {
            root_to_label[r] = next_label;
            next_label += 1;
        }
        *slot = root_to_label[r];
    }

    partition_agreement(labels, &pred_labels)
}
