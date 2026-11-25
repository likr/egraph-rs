//! Cluster-aware, visual separation, Sprawlter-NN, and NOR baseline evaluation metrics.
//!
//! Gated behind the `eval-baselines` Cargo feature to keep the minimal `egraph-metrics`
//! dependency graph free of `egraph-clustering` when baseline clustering metrics are not needed.
//!
//! All coordinates and extents are accumulated in `f64` internally regardless of layout
//! scalar type `S: Scalar`, all algorithms are strictly single-threaded and deterministic,
//! and all normalized score families (`DSC`, `DC`, `CDM`, `HDM`, `SepMe`, `M_NH`, `PDOE`)
//! are returned on `[0.0, 1.0]` (`paper_percent = 100.0 * rust_score`).

use std::cmp::Ordering;

use egraph_clustering::{KMeansConfig, kmeans_plus_plus};
use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, Scalar, VisitStamp};
use egraph_spatial::{KdQueryScratch, RectangleOverlapSweep, StaticKdTree};

use crate::{
    build_kdtree_f64, extract_unique_edges,
    proximity::{gabriel_graph_edges_with_tree, relative_neighborhood_edges_with_tree},
    validate_graph_layout, validate_tree_layout,
};

/// Partition agreement metrics comparing two discrete node labelings.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PartitionAgreement {
    /// Adjusted Rand Index (`sklearn.metrics.adjusted_rand_score`).
    pub ari: f64,
    /// Adjusted Mutual Information with arithmetic normalization (`nats`, `sklearn` default).
    pub ami: f64,
    /// Cluster homogeneity in `[0.0, 1.0]`.
    pub homogeneity: f64,
    /// Cluster completeness in `[0.0, 1.0]`.
    pub completeness: f64,
    /// Harmonic mean V-measure ($\beta = 1.0$) in `[0.0, 1.0]`.
    pub v_measure: f64,
    /// Fowlkes-Mallows Index in `[0.0, 1.0]`.
    pub fmi: f64,
}

/// Chen et al. (2020) Node Overlap Removal (NOR) 5-metric evaluation summary.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct NorMetrics {
    /// Orthogonal Ordering: Normalized Number of Inversions in `[0.0, 1.0]` (lower is better).
    pub oo_nni: f64,
    /// Spread Minimization: Convex Hull Area Ratio using $4N$ box corners (optimal `1.0`).
    pub sp_ch_a: f64,
    /// Global Shape: Bounding Box Improved Aspect Ratio $\ge 1.0$ using $4N$ box corners (optimal `1.0`).
    pub gs_bb_iar: f64,
    /// Node Movement: Translation-invariant Distance Moved Improved MSE (lower is better).
    pub nm_dm_imse: f64,
    /// Edge Length Preservation: Relative Standard Deviation of edge length ratios (lower is better).
    pub el_rsd: f64,
}

#[inline]
fn canonical_zero(v: f64) -> f64 {
    if v == 0.0 { 0.0 } else { v }
}

#[inline]
fn validate_layout_labels<S: Scalar>(
    layout: &Layout2d<S>,
    labels: &[u32],
    context: &'static str,
) -> CoreResult<()> {
    if layout.len() != labels.len() {
        return Err(CoreError::dimension_mismatch(
            layout.len(),
            labels.len(),
            context,
        ));
    }
    layout.validate_finite()?;
    Ok(())
}

/// Compacts arbitrary `u32` labels into canonical `0..num_clusters` in ascending order
/// of first appearance without hash maps.
fn compact_labels(labels: &[u32]) -> (Vec<u32>, Vec<usize>, usize) {
    let n = labels.len();
    if n == 0 {
        return (Vec::new(), Vec::new(), 0);
    }

    let max_label = labels.iter().copied().max().unwrap_or(0) as usize;
    let mut compacted = vec![0u32; n];
    let fast_limit = n.max(64).saturating_mul(4).min(1_000_000);
    if max_label < fast_limit {
        let mut remap = vec![u32::MAX; max_label + 1];
        let mut next_id = 0u32;
        for (i, &l) in labels.iter().enumerate() {
            let idx = l as usize;
            if remap[idx] == u32::MAX {
                remap[idx] = next_id;
                next_id += 1;
            }
            compacted[i] = remap[idx];
        }
        let num_clusters = next_id as usize;
        let mut sizes = vec![0usize; num_clusters];
        for &c in &compacted {
            sizes[c as usize] += 1;
        }
        return (compacted, sizes, num_clusters);
    }

    let mut pairs: Vec<(u32, u32)> = labels
        .iter()
        .enumerate()
        .map(|(i, &l)| (l, i as u32))
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

    let num_clusters = groups.len();
    let mut sizes = Vec::with_capacity(num_clusters);
    for (cid, &(_, g_start, g_end)) in groups.iter().enumerate() {
        let cid_u32 = cid as u32;
        sizes.push(g_end - g_start);
        for &(_, orig_node) in &pairs[g_start..g_end] {
            compacted[orig_node as usize] = cid_u32;
        }
    }
    (compacted, sizes, num_clusters)
}

#[inline]
fn sign_preserving_clamp(x: f64) -> f64 {
    if x < 0.0 {
        x.min(-f64::EPSILON)
    } else {
        x.max(f64::EPSILON)
    }
}

fn group_marginal_multiplicities(counts: &[usize]) -> Vec<(usize, usize)> {
    let mut sorted = counts.to_vec();
    sorted.sort_unstable();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < sorted.len() {
        let val = sorted[i];
        let mut j = i + 1;
        while j < sorted.len() && sorted[j] == val {
            j += 1;
        }
        if val > 0 {
            out.push((val, j - i));
        }
        i = j;
    }
    out
}

/// Computes exact partition agreement metrics (`ARI`, `AMI`, `Homogeneity`, `Completeness`,
/// `V-Measure`, `FMI`) matching `scikit-learn` conventions and edge cases.
pub fn partition_agreement(
    labels_true: &[u32],
    labels_pred: &[u32],
) -> CoreResult<PartitionAgreement> {
    if labels_true.len() != labels_pred.len() {
        return Err(CoreError::dimension_mismatch(
            labels_true.len(),
            labels_pred.len(),
            "partition_agreement",
        ));
    }

    let n = labels_true.len();
    if n == 0 {
        return Ok(PartitionAgreement {
            ari: 1.0,
            ami: 1.0,
            homogeneity: 1.0,
            completeness: 1.0,
            v_measure: 1.0,
            fmi: 0.0,
        });
    }

    let (u_comp, a_counts, r_classes) = compact_labels(labels_true);
    let (v_comp, b_counts, c_classes) = compact_labels(labels_pred);

    let mut packed = Vec::with_capacity(n);
    for i in 0..n {
        packed.push(((u_comp[i] as u64) << 32) | (v_comp[i] as u64));
    }
    packed.sort_unstable();

    let mut contingency: Vec<(usize, usize, usize)> = Vec::new();
    let mut idx = 0usize;
    while idx < n {
        let key = packed[idx];
        let mut end = idx + 1;
        while end < n && packed[end] == key {
            end += 1;
        }
        let r = (key >> 32) as usize;
        let c = (key & 0xFFFF_FFFF) as usize;
        contingency.push((r, c, end - idx));
        idx = end;
    }

    let n_u128 = n as u128;
    let sum_nij_sq: u128 = contingency
        .iter()
        .map(|&(_, _, c)| (c as u128) * (c as u128))
        .sum();
    let sum_a_sq: u128 = a_counts.iter().map(|&c| (c as u128) * (c as u128)).sum();
    let sum_b_sq: u128 = b_counts.iter().map(|&c| (c as u128) * (c as u128)).sum();

    let tp = sum_nij_sq - n_u128;
    let fp = sum_b_sq - n_u128 - tp;
    let fn_ = sum_a_sq - n_u128 - tp;
    let tn = n_u128 * (n_u128 - 1) - tp - fp - fn_;

    let ari = if fn_ == 0 && fp == 0 {
        1.0
    } else {
        let num = match (tp.checked_mul(tn), fn_.checked_mul(fp)) {
            (Some(lhs), Some(rhs)) => {
                if lhs >= rhs {
                    2.0 * ((lhs - rhs) as f64)
                } else {
                    -2.0 * ((rhs - lhs) as f64)
                }
            }
            _ => 2.0 * ((tp as f64) * (tn as f64) - (fn_ as f64) * (fp as f64)),
        };
        let den =
            ((tp + fn_) as f64) * ((fn_ + tn) as f64) + ((tp + fp) as f64) * ((fp + tn) as f64);
        if den == 0.0 { 1.0 } else { num / den }
    };

    let fmi = if tp == 0 {
        0.0
    } else {
        let pk = (sum_b_sq - n_u128) as f64;
        let qk = (sum_a_sq - n_u128) as f64;
        let tk = tp as f64;
        (tk / pk).sqrt() * (tk / qk).sqrt()
    };

    let n_f = n as f64;
    let mut h_true = 0.0_f64;
    for &a_i in &a_counts {
        if a_i > 0 {
            let p = a_i as f64 / n_f;
            h_true -= p * p.ln();
        }
    }
    let mut h_pred = 0.0_f64;
    for &b_j in &b_counts {
        if b_j > 0 {
            let p = b_j as f64 / n_f;
            h_pred -= p * p.ln();
        }
    }

    let mut mi = 0.0_f64;
    let ln_n = n_f.ln();
    for &(r, c, nij) in &contingency {
        let p = nij as f64 / n_f;
        let log_term =
            (nij as f64).ln() + ln_n - (a_counts[r] as f64).ln() - (b_counts[c] as f64).ln();
        mi += p * log_term;
    }
    let mi = mi.max(0.0);

    let homogeneity = if h_true > 0.0 { mi / h_true } else { 1.0 };
    let completeness = if h_pred > 0.0 { mi / h_pred } else { 1.0 };
    let v_measure = if homogeneity + completeness == 0.0 {
        0.0
    } else {
        2.0 * homogeneity * completeness / (homogeneity + completeness)
    };

    let ami = if r_classes == 1 && c_classes == 1 {
        1.0
    } else if r_classes == 1 || c_classes == 1 {
        0.0
    } else {
        let mut ln_fact = vec![0.0_f64; n + 1];
        for k in 1..=n {
            ln_fact[k] = ln_fact[k - 1] + (k as f64).ln();
        }
        let a_groups = group_marginal_multiplicities(&a_counts);
        let b_groups = group_marginal_multiplicities(&b_counts);

        let mut emi = 0.0_f64;
        for &(a_i, mult_a) in &a_groups {
            for &(b_j, mult_b) in &b_groups {
                let mult = (mult_a as f64) * (mult_b as f64);
                let n_min = 1usize.max((a_i + b_j).saturating_sub(n));
                let n_max = a_i.min(b_j);
                let base_log =
                    ln_fact[a_i] + ln_fact[b_j] + ln_fact[n - a_i] + ln_fact[n - b_j] - ln_fact[n];
                let ln_ab = (a_i as f64 * b_j as f64).ln();
                for nij in n_min..=n_max {
                    let log_p = base_log
                        - ln_fact[nij]
                        - ln_fact[a_i - nij]
                        - ln_fact[b_j - nij]
                        - ln_fact[(n - a_i) + nij - b_j];
                    let term_log = (n_f * nij as f64).ln() - ln_ab;
                    emi += mult * (nij as f64 / n_f) * term_log * log_p.exp();
                }
            }
        }
        let normalizer = 0.5 * (h_true + h_pred);
        let num = sign_preserving_clamp(mi - emi);
        let den = sign_preserving_clamp(normalizer - emi);
        num / den
    };

    Ok(PartitionAgreement {
        ari,
        ami,
        homogeneity,
        completeness,
        v_measure,
        fmi,
    })
}

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
/// layout coordinates (`CQ-kmeans`), returning [`PartitionAgreement`] against `community_labels`.
///
/// Returns [`PartitionAgreement::default()`] (all zeros) on degenerate cases (`n < 2`, `k < 2`,
/// `k >= n`, or collapsed coordinates with `span <= 1e-15` or `rms <= 1e-15`).
pub fn cq_kmeans<S: Scalar>(
    layout: &Layout2d<S>,
    community_labels: &[u32],
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
    validate_layout_labels(layout, community_labels, "cq_kmeans")?;

    let n = layout.len();
    let (_, _, k) = compact_labels(community_labels);
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
    partition_agreement(community_labels, &res.labels)
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
/// (`CQ-HAC`), returning [`PartitionAgreement`] against `community_labels`.
///
/// Collects all $N - 1$ dendrogram merges, sorts by `(height, step)`, and cuts at `K` clusters
/// to match `scipy.cluster.hierarchy.linkage(method='average')` + `fcluster(criterion='maxclust')`.
/// Allocates an $O(N^2)$ symmetric distance matrix. Returns [`PartitionAgreement::default()`] on
/// degenerate cases (`n < 2`, `k < 2`, `k >= n`, or collapsed coordinates).
pub fn cq_hac<S: Scalar>(
    layout: &Layout2d<S>,
    community_labels: &[u32],
) -> CoreResult<PartitionAgreement> {
    validate_layout_labels(layout, community_labels, "cq_hac")?;

    let n = layout.len();
    let (_, _, k) = compact_labels(community_labels);
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

    partition_agreement(community_labels, &pred_labels)
}

/// Computes Sips et al. (2009) Distance Consistency (`DSC`) in `[0.0, 1.0]` (higher is better).
///
/// Counts the fraction of points whose squared Euclidean distance to their own class centroid
/// is **strictly smaller** than to every other class centroid (`N < 2 || K < 2` returns `0.0`).
pub fn distance_consistency<S: Scalar>(layout: &Layout2d<S>, labels: &[u32]) -> CoreResult<f64> {
    validate_layout_labels(layout, labels, "distance_consistency")?;
    let n = layout.len();
    let (comp, sizes, k) = compact_labels(labels);
    if n < 2 || k < 2 {
        return Ok(0.0);
    }

    let xs = layout.x();
    let ys = layout.y();
    let mut cx = vec![0.0_f64; k];
    let mut cy = vec![0.0_f64; k];
    for i in 0..n {
        let c = comp[i] as usize;
        cx[c] += xs[i].to_f64();
        cy[c] += ys[i].to_f64();
    }
    for c in 0..k {
        let inv = 1.0 / sizes[c] as f64;
        cx[c] *= inv;
        cy[c] *= inv;
    }

    let mut consistent = 0usize;
    for i in 0..n {
        let own = comp[i] as usize;
        let x = xs[i].to_f64();
        let y = ys[i].to_f64();
        let dx_own = x - cx[own];
        let dy_own = y - cy[own];
        let own_d2 = dx_own * dx_own + dy_own * dy_own;

        let mut is_strictly_closest = true;
        for c in 0..k {
            if c == own {
                continue;
            }
            let dx = x - cx[c];
            let dy = y - cy[c];
            if dx * dx + dy * dy <= own_d2 {
                is_strictly_closest = false;
                break;
            }
        }
        if is_strictly_closest {
            consistent += 1;
        }
    }
    Ok(consistent as f64 / n as f64)
}

fn normalized_unit_coords<S: Scalar>(layout: &Layout2d<S>) -> (Vec<f64>, Vec<f64>) {
    let n = layout.len();
    let xs = layout.x();
    let ys = layout.y();
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for i in 0..n {
        let x = xs[i].to_f64();
        let y = ys[i].to_f64();
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    let span_x = max_x - min_x;
    let span_y = max_y - min_y;
    let mut us = Vec::with_capacity(n);
    let mut vs = Vec::with_capacity(n);
    for i in 0..n {
        let u = if span_x > 1e-15 {
            (xs[i].to_f64() - min_x) / span_x
        } else {
            0.5
        };
        let v = if span_y > 1e-15 {
            (ys[i].to_f64() - min_y) / span_y
        } else {
            0.5
        };
        us.push(u);
        vs.push(v);
    }
    (us, vs)
}

/// Computes Sips et al. (2009) Distribution Consistency (`DC`) scaled to `[0.0, 1.0]`
/// (`DC_paper = 100.0 * dc`), measuring weighted grid-cell class entropy purity over a
/// `grid_size x grid_size` partition of the normalized `[0, 1]^2` layout box.
///
/// Returns `0.0` when `N == 0` or `K < 2`. Errors when `grid_size == 0`.
pub fn distribution_consistency<S: Scalar>(
    layout: &Layout2d<S>,
    labels: &[u32],
    grid_size: usize,
) -> CoreResult<f64> {
    if grid_size == 0 {
        return Err(CoreError::invalid_input(
            "distribution_consistency requires grid_size >= 1",
        ));
    }
    validate_layout_labels(layout, labels, "distribution_consistency")?;

    let n = layout.len();
    let (comp, _, k) = compact_labels(labels);
    if n == 0 || k < 2 {
        return Ok(0.0);
    }

    let (us, vs) = normalized_unit_coords(layout);
    let g_f = grid_size as f64;
    let mut pairs: Vec<(usize, u32)> = Vec::with_capacity(n);
    for i in 0..n {
        let gx = ((us[i] * g_f).floor() as usize).min(grid_size - 1);
        let gy = ((vs[i] * g_f).floor() as usize).min(grid_size - 1);
        pairs.push((gy * grid_size + gx, comp[i]));
    }
    pairs.sort_unstable();

    let mut weighted_entropy = 0.0_f64;
    let mut start = 0usize;
    while start < n {
        let cell = pairs[start].0;
        let mut end = start + 1;
        while end < n && pairs[end].0 == cell {
            end += 1;
        }
        let p_total = (end - start) as f64;
        let mut cell_h = 0.0_f64;
        let mut c_start = start;
        while c_start < end {
            let cls = pairs[c_start].1;
            let mut c_end = c_start + 1;
            while c_end < end && pairs[c_end].1 == cls {
                c_end += 1;
            }
            let prob = (c_end - c_start) as f64 / p_total;
            cell_h -= prob * prob.log2();
            c_start = c_end;
        }
        weighted_entropy += p_total * cell_h;
        start = end;
    }

    let denom = (k as f64).log2() * n as f64;
    Ok((1.0 - weighted_entropy / denom).clamp(0.0, 1.0))
}

/// Computes Tatu et al. (2009) Class Density Measure (`CDM`) and `3x3` Moore-smoothed
/// Histogram Density Measure (`2D-HDM`), both normalized to `[0.0, 1.0]` (`1.0` = maximally
/// separated, `0.0` = identical overlapping distributions).
///
/// - `CDM` evaluates unit-normalized Gaussian KDE densities per class (default `bandwidth =
///   1.5 / grid_size as f64`) and averages pairwise L1 separation $\frac{1}{2\binom{K}{2}}
///   \sum_{c < d} \sum_g |\tilde{\rho}_g^c - \tilde{\rho}_g^d|$.
/// - `HDM` applies $3\times 3$ Moore-neighborhood smoothing to per-class grid counts and computes
///   `1.0 - sum(U * H) / (log2(K) * sum(U))` (`hdm_paper = 100.0 * hdm`).
///
/// Returns `Ok((0.0, 0.0))` when `N == 0` or `K < 2`.
pub fn class_density_measure<S: Scalar>(
    layout: &Layout2d<S>,
    labels: &[u32],
    grid_size: usize,
    bandwidth: Option<f64>,
) -> CoreResult<(f64, f64)> {
    if grid_size == 0 {
        return Err(CoreError::invalid_input(
            "class_density_measure requires grid_size >= 1",
        ));
    }
    let bw = match bandwidth {
        Some(h) => {
            if !h.is_finite() {
                return Err(CoreError::non_finite("class_density_measure::bandwidth"));
            }
            if h <= 0.0 {
                return Err(CoreError::invalid_input(
                    "class_density_measure requires bandwidth > 0.0",
                ));
            }
            h
        }
        None => 1.5 / grid_size as f64,
    };
    validate_layout_labels(layout, labels, "class_density_measure")?;

    let n = layout.len();
    let (comp, sizes, k) = compact_labels(labels);
    if n == 0 || k < 2 {
        return Ok((0.0, 0.0));
    }

    let num_cells = grid_size
        .checked_mul(grid_size)
        .ok_or_else(|| CoreError::invalid_input("class_density_measure grid_size^2 overflow"))?;
    let (us, vs) = normalized_unit_coords(layout);
    let g_f = grid_size as f64;

    // 1. 2D-HDM via 3x3 Moore-smoothed sparse `(cell, class)` entries (O(N log N) time & O(N) RAM).
    let mut hdm_entries: Vec<(usize, u32)> = Vec::with_capacity(n.saturating_mul(9));
    for i in 0..n {
        let gx = ((us[i] * g_f).floor() as isize).clamp(0, grid_size as isize - 1);
        let gy = ((vs[i] * g_f).floor() as isize).clamp(0, grid_size as isize - 1);
        let cls = comp[i];
        for dy in -1isize..=1 {
            let ny = gy + dy;
            if ny < 0 || ny >= grid_size as isize {
                continue;
            }
            for dx in -1isize..=1 {
                let nx = gx + dx;
                if nx < 0 || nx >= grid_size as isize {
                    continue;
                }
                hdm_entries.push(((ny as usize) * grid_size + (nx as usize), cls));
            }
        }
    }
    hdm_entries.sort_unstable();

    let mut hdm_weighted_entropy = 0.0_f64;
    let mut hdm_total_u = 0.0_f64;
    let mut idx = 0usize;
    while idx < hdm_entries.len() {
        let cell = hdm_entries[idx].0;
        let mut end = idx + 1;
        while end < hdm_entries.len() && hdm_entries[end].0 == cell {
            end += 1;
        }
        let u_total = (end - idx) as f64;
        let mut cell_h = 0.0_f64;
        let mut c_start = idx;
        while c_start < end {
            let cls = hdm_entries[c_start].1;
            let mut c_end = c_start + 1;
            while c_end < end && hdm_entries[c_end].1 == cls {
                c_end += 1;
            }
            let prob = (c_end - c_start) as f64 / u_total;
            cell_h -= prob * prob.log2();
            c_start = c_end;
        }
        hdm_weighted_entropy += u_total * cell_h;
        hdm_total_u += u_total;
        idx = end;
    }
    let hdm = if hdm_total_u <= 0.0 {
        0.0
    } else {
        (1.0 - hdm_weighted_entropy / ((k as f64).log2() * hdm_total_u)).clamp(0.0, 1.0)
    };

    // 2. 2D-CDM via per-class CSR point grouping + single reusable grid with VisitStamp.
    let mut class_offsets = vec![0usize; k + 1];
    for c in 0..k {
        class_offsets[c + 1] = class_offsets[c] + sizes[c];
    }
    let mut class_cursor = class_offsets[..k].to_vec();
    let mut class_points = vec![0usize; n];
    for (i, &c) in comp.iter().enumerate() {
        let pos = class_cursor[c as usize];
        class_points[pos] = i;
        class_cursor[c as usize] = pos + 1;
    }

    let inv_two_bw_sq = 1.0 / (2.0 * bw * bw);
    let cutoff = 4.0 * bw;
    let cutoff_sq = cutoff * cutoff;
    let r_cells = (cutoff * g_f).ceil().min(g_f) as isize;

    let mut cell_acc = vec![0.0_f64; num_cells];
    let mut stamp = VisitStamp::new(num_cells);
    let mut touched = Vec::new();
    let mut sparse_kde: Vec<(usize, f64)> = Vec::new();

    for c in 0..k {
        stamp.clear();
        touched.clear();
        let mut sum_w = 0.0_f64;

        for &pt_idx in &class_points[class_offsets[c]..class_offsets[c + 1]] {
            let u = us[pt_idx];
            let v = vs[pt_idx];
            let gx = ((u * g_f).floor() as isize).clamp(0, grid_size as isize - 1);
            let gy = ((v * g_f).floor() as isize).clamp(0, grid_size as isize - 1);

            let x_lo = (gx - r_cells).max(0);
            let x_hi = (gx + r_cells).min(grid_size as isize - 1);
            let y_lo = (gy - r_cells).max(0);
            let y_hi = (gy + r_cells).min(grid_size as isize - 1);

            for cy in y_lo..=y_hi {
                let cy_center = (cy as f64 + 0.5) / g_f;
                let dy = cy_center - v;
                let dy_sq = dy * dy;
                if dy_sq > cutoff_sq {
                    continue;
                }
                let row_off = (cy as usize) * grid_size;
                for cx in x_lo..=x_hi {
                    let cx_center = (cx as f64 + 0.5) / g_f;
                    let dx = cx_center - u;
                    let d_sq = dx * dx + dy_sq;
                    if d_sq > cutoff_sq {
                        continue;
                    }
                    let w = (-d_sq * inv_two_bw_sq).exp();
                    if w > 0.0 {
                        let cell = row_off + (cx as usize);
                        if stamp.visit(cell) {
                            cell_acc[cell] = w;
                            touched.push(cell);
                        } else {
                            cell_acc[cell] += w;
                        }
                        sum_w += w;
                    }
                }
            }
        }

        if sum_w <= 0.0 {
            // Extreme sub-cell bandwidth fallback: Dirac impulse onto owning grid cell.
            stamp.clear();
            touched.clear();
            sum_w = 0.0;
            for &pt_idx in &class_points[class_offsets[c]..class_offsets[c + 1]] {
                let gx = ((us[pt_idx] * g_f).floor() as usize).min(grid_size - 1);
                let gy = ((vs[pt_idx] * g_f).floor() as usize).min(grid_size - 1);
                let cell = gy * grid_size + gx;
                if stamp.visit(cell) {
                    cell_acc[cell] = 1.0;
                    touched.push(cell);
                } else {
                    cell_acc[cell] += 1.0;
                }
                sum_w += 1.0;
            }
        }

        let inv_sum = 1.0 / sum_w;
        for &cell in &touched {
            sparse_kde.push((cell, cell_acc[cell] * inv_sum));
        }
    }

    sparse_kde.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));

    let mut total_min_sum = 0.0_f64;
    let mut s_idx = 0usize;
    while s_idx < sparse_kde.len() {
        let cell = sparse_kde[s_idx].0;
        let mut s_end = s_idx + 1;
        while s_end < sparse_kde.len() && sparse_kde[s_end].0 == cell {
            s_end += 1;
        }
        let m = s_end - s_idx;
        if m >= 2 {
            // Because slice is sorted ascending by density, v_p is the minimum in (m - 1 - p) pairs.
            for (p, &(_, val)) in sparse_kde[s_idx..s_end].iter().enumerate() {
                total_min_sum += val * ((m - 1 - p) as f64);
            }
        }
        s_idx = s_end;
    }

    let num_pairs = (k * (k - 1) / 2) as f64;
    let cdm = (1.0 - total_min_sum / num_pairs).clamp(0.0, 1.0);
    Ok((cdm, hdm))
}

fn dir_cpt_from_degree_counts(
    comp: &[u32],
    class_sizes: &[usize],
    k_classes: usize,
    deg_out: &[usize],
    same_out: &[usize],
) -> f64 {
    let mut class_sum = vec![0.0_f64; k_classes];
    for (i, &c) in comp.iter().enumerate() {
        let p_i = if deg_out[i] == 0 {
            1.0
        } else {
            same_out[i] as f64 / deg_out[i] as f64
        };
        class_sum[c as usize] += p_i;
    }
    let macro_sum: f64 = class_sum
        .iter()
        .zip(class_sizes.iter())
        .map(|(&s, &sz)| s / sz as f64)
        .sum();
    macro_sum / k_classes as f64
}

fn dir_cpt_from_undirected_edges(
    n: usize,
    comp: &[u32],
    class_sizes: &[usize],
    k_classes: usize,
    edges: &[(u32, u32)],
) -> f64 {
    let mut deg = vec![0usize; n];
    let mut same = vec![0usize; n];
    for &(u, v) in edges {
        let ui = u as usize;
        let vi = v as usize;
        deg[ui] += 1;
        deg[vi] += 1;
        if comp[ui] == comp[vi] {
            same[ui] += 1;
            same[vi] += 1;
        }
    }
    dir_cpt_from_degree_counts(comp, class_sizes, k_classes, &deg, &same)
}

/// Computes Aupetit & Sedlmair (2016) `SepMe` $k$-Nearest Neighbor Graph (`KNNG DIR CPT`)
/// in `[0.0, 1.0]`.
pub fn sepme_knng<S: Scalar>(layout: &Layout2d<S>, labels: &[u32], k: usize) -> CoreResult<f64> {
    if k == 0 {
        return Err(CoreError::invalid_input("sepme_knng requires k >= 1"));
    }
    validate_layout_labels(layout, labels, "sepme_knng")?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    if layout.len() == 1 {
        return Ok(1.0);
    }
    let tree = build_kdtree_f64(layout)?;
    sepme_knng_with_tree(layout, labels, k, &tree)
}

pub(crate) fn sepme_knng_with_tree<S: Scalar>(
    layout: &Layout2d<S>,
    labels: &[u32],
    k: usize,
    tree: &StaticKdTree<2, f64>,
) -> CoreResult<f64> {
    if k == 0 {
        return Err(CoreError::invalid_input("sepme_knng requires k >= 1"));
    }
    validate_layout_labels(layout, labels, "sepme_knng_with_tree")?;
    validate_tree_layout(tree, layout, "sepme_knng_with_tree")?;

    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        return Ok(1.0);
    }

    let (comp, class_sizes, k_classes) = compact_labels(labels);
    let k_eff = k.min(n - 1);
    let mut scratch = KdQueryScratch::with_capacity(k_eff);
    let mut nbrs = Vec::with_capacity(k_eff);
    let mut deg = vec![0usize; n];
    let mut same = vec![0usize; n];

    for i in 0..n {
        tree.k_nearest_excluding_self_with_scratch(i, k_eff, &mut scratch, &mut nbrs)?;
        deg[i] = nbrs.len();
        let ci = comp[i];
        let mut s = 0usize;
        for &(j, _) in &nbrs {
            if comp[j] == ci {
                s += 1;
            }
        }
        same[i] = s;
    }

    Ok(dir_cpt_from_degree_counts(
        &comp,
        &class_sizes,
        k_classes,
        &deg,
        &same,
    ))
}

/// Computes Aupetit & Sedlmair (2016) `SepMe` Relative Neighborhood Graph (`RNG DIR CPT`)
/// in `[0.0, 1.0]`.
pub fn sepme_rng<S: Scalar>(layout: &Layout2d<S>, labels: &[u32]) -> CoreResult<f64> {
    validate_layout_labels(layout, labels, "sepme_rng")?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    if layout.len() == 1 {
        return Ok(1.0);
    }
    let tree = build_kdtree_f64(layout)?;
    sepme_rng_with_tree(layout, labels, &tree)
}

pub(crate) fn sepme_rng_with_tree<S: Scalar>(
    layout: &Layout2d<S>,
    labels: &[u32],
    tree: &StaticKdTree<2, f64>,
) -> CoreResult<f64> {
    validate_layout_labels(layout, labels, "sepme_rng_with_tree")?;
    validate_tree_layout(tree, layout, "sepme_rng_with_tree")?;
    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        return Ok(1.0);
    }
    let (comp, class_sizes, k_classes) = compact_labels(labels);
    let edges = relative_neighborhood_edges_with_tree(layout, tree)?;
    Ok(dir_cpt_from_undirected_edges(
        n,
        &comp,
        &class_sizes,
        k_classes,
        &edges,
    ))
}

/// Computes Aupetit & Sedlmair (2016) `SepMe` Gabriel Graph (`GAB DIR CPT`) in `[0.0, 1.0]`.
pub fn sepme_gab<S: Scalar>(layout: &Layout2d<S>, labels: &[u32]) -> CoreResult<f64> {
    validate_layout_labels(layout, labels, "sepme_gab")?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    if layout.len() == 1 {
        return Ok(1.0);
    }
    let tree = build_kdtree_f64(layout)?;
    sepme_gab_with_tree(layout, labels, &tree)
}

pub(crate) fn sepme_gab_with_tree<S: Scalar>(
    layout: &Layout2d<S>,
    labels: &[u32],
    tree: &StaticKdTree<2, f64>,
) -> CoreResult<f64> {
    validate_layout_labels(layout, labels, "sepme_gab_with_tree")?;
    validate_tree_layout(tree, layout, "sepme_gab_with_tree")?;
    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        return Ok(1.0);
    }
    let (comp, class_sizes, k_classes) = compact_labels(labels);
    let edges = gabriel_graph_edges_with_tree(layout, tree)?;
    Ok(dir_cpt_from_undirected_edges(
        n,
        &comp,
        &class_sizes,
        k_classes,
        &edges,
    ))
}

#[inline]
fn is_gong_directed_edge(
    i: usize,
    j: usize,
    gamma: f64,
    pts: &[[f64; 2]],
    tree: &StaticKdTree<2, f64>,
    witness_buf: &mut Vec<(usize, f64)>,
) -> CoreResult<bool> {
    let dx = pts[j][0] - pts[i][0];
    let dy = pts[j][1] - pts[i][1];
    let d_ij_sq = dx * dx + dy * dy;
    if d_ij_sq <= 0.0 {
        return Ok(false);
    }
    let one_minus_g = 1.0 - gamma;
    let m = [
        gamma * pts[j][0] + one_minus_g * pts[i][0],
        gamma * pts[j][1] + one_minus_g * pts[i][1],
    ];
    let r_sq = one_minus_g * one_minus_g * d_ij_sq;
    let query_r_sq = r_sq * (1.0 + 1e-11);
    tree.within_radius_sq_into(&m, query_r_sq, witness_buf)?;

    let block_thresh = r_sq * (1.0 + 1e-12);
    for &(q, d_mq_sq) in witness_buf.iter() {
        if q == i || q == j {
            continue;
        }
        if d_mq_sq <= block_thresh {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Computes Aupetit & Sedlmair (2016) `SepMe` $\gamma$-Observable Neighbor Graph (`GONG DIR CPT`,
/// default `gamma = 0.35`) in `[0.0, 1.0]`.
///
/// For `0.0 <= gamma <= 0.5`, prunes candidates via `gabriel_graph_edges_with_tree` since
/// $\mathrm{GONG}_\gamma \subseteq \mathrm{GG}$ holds strictly on $[0, 0.5]$.
pub fn sepme_gong<S: Scalar>(layout: &Layout2d<S>, labels: &[u32], gamma: f64) -> CoreResult<f64> {
    if !gamma.is_finite() {
        return Err(CoreError::non_finite("sepme_gong::gamma"));
    }
    if !(0.0..=1.0).contains(&gamma) {
        return Err(CoreError::invalid_input(
            "sepme_gong requires gamma in [0.0, 1.0]",
        ));
    }
    validate_layout_labels(layout, labels, "sepme_gong")?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    if layout.len() == 1 {
        return Ok(1.0);
    }
    let tree = build_kdtree_f64(layout)?;
    sepme_gong_with_tree(layout, labels, gamma, &tree)
}

pub(crate) fn sepme_gong_with_tree<S: Scalar>(
    layout: &Layout2d<S>,
    labels: &[u32],
    gamma: f64,
    tree: &StaticKdTree<2, f64>,
) -> CoreResult<f64> {
    if !gamma.is_finite() {
        return Err(CoreError::non_finite("sepme_gong::gamma"));
    }
    if !(0.0..=1.0).contains(&gamma) {
        return Err(CoreError::invalid_input(
            "sepme_gong requires gamma in [0.0, 1.0]",
        ));
    }
    validate_layout_labels(layout, labels, "sepme_gong_with_tree")?;
    validate_tree_layout(tree, layout, "sepme_gong_with_tree")?;

    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        return Ok(1.0);
    }

    let (comp, class_sizes, k_classes) = compact_labels(labels);
    let xs = layout.x();
    let ys = layout.y();
    let pts: Vec<[f64; 2]> = (0..n).map(|i| [xs[i].to_f64(), ys[i].to_f64()]).collect();

    let mut deg_out = vec![0usize; n];
    let mut same_out = vec![0usize; n];
    let mut witness_buf = Vec::new();

    if gamma <= 0.5 {
        let gg_edges = gabriel_graph_edges_with_tree(layout, tree)?;
        for (u, v) in gg_edges {
            let ui = u as usize;
            let vi = v as usize;
            let same_class = comp[ui] == comp[vi];
            if is_gong_directed_edge(ui, vi, gamma, &pts, tree, &mut witness_buf)? {
                deg_out[ui] += 1;
                if same_class {
                    same_out[ui] += 1;
                }
            }
            if is_gong_directed_edge(vi, ui, gamma, &pts, tree, &mut witness_buf)? {
                deg_out[vi] += 1;
                if same_class {
                    same_out[vi] += 1;
                }
            }
        }
    } else {
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                if is_gong_directed_edge(i, j, gamma, &pts, tree, &mut witness_buf)? {
                    deg_out[i] += 1;
                    if comp[i] == comp[j] {
                        same_out[i] += 1;
                    }
                }
            }
        }
    }

    Ok(dir_cpt_from_degree_counts(
        &comp,
        &class_sizes,
        k_classes,
        &deg_out,
        &same_out,
    ))
}

/// Computes Paulovich et al. (2008) / Espadoto et al. (2021) Neighborhood Hit (`M_NH`, default
/// `k = 7`) in `[0.0, 1.0]`, micro-averaged over all `N` vertices (`N < 2` returns `0.0`).
pub fn neighborhood_hit<S: Scalar>(
    layout: &Layout2d<S>,
    labels: &[u32],
    k: usize,
) -> CoreResult<f64> {
    if k == 0 {
        return Err(CoreError::invalid_input("neighborhood_hit requires k >= 1"));
    }
    validate_layout_labels(layout, labels, "neighborhood_hit")?;
    if layout.len() < 2 {
        return Ok(0.0);
    }
    let tree = build_kdtree_f64(layout)?;
    neighborhood_hit_with_tree(layout, labels, k, &tree)
}

pub(crate) fn neighborhood_hit_with_tree<S: Scalar>(
    layout: &Layout2d<S>,
    labels: &[u32],
    k: usize,
    tree: &StaticKdTree<2, f64>,
) -> CoreResult<f64> {
    if k == 0 {
        return Err(CoreError::invalid_input("neighborhood_hit requires k >= 1"));
    }
    validate_layout_labels(layout, labels, "neighborhood_hit_with_tree")?;
    validate_tree_layout(tree, layout, "neighborhood_hit_with_tree")?;

    let n = layout.len();
    if n < 2 {
        return Ok(0.0);
    }
    let k_eff = k.min(n - 1);
    let mut scratch = KdQueryScratch::with_capacity(k_eff);
    let mut nbrs = Vec::with_capacity(k_eff);
    let mut total_same = 0usize;

    for i in 0..n {
        tree.k_nearest_excluding_self_with_scratch(i, k_eff, &mut scratch, &mut nbrs)?;
        let li = labels[i];
        for &(j, _) in &nbrs {
            if labels[j] == li {
                total_same += 1;
            }
        }
    }

    Ok(total_same as f64 / (k_eff * n) as f64)
}

/// Computes mean Silhouette coefficient matching `sklearn.metrics.silhouette_score` in
/// `[-1.0, 1.0]` using $O(N + K)$ memory (singleton clusters score `0.0`).
///
/// Under the non-panic evaluation contract, returns `Ok(0.0)` when `N < 3`, `K < 2`, or `K >= N`.
pub fn silhouette_score<S: Scalar>(layout: &Layout2d<S>, labels: &[u32]) -> CoreResult<f64> {
    validate_layout_labels(layout, labels, "silhouette_score")?;
    let n = layout.len();
    let (comp, sizes, k) = compact_labels(labels);
    if n < 3 || k < 2 || k >= n {
        return Ok(0.0);
    }

    let xs = layout.x();
    let ys = layout.y();
    let mut sum_dists = vec![0.0_f64; k];
    let mut total_s = 0.0_f64;

    for i in 0..n {
        let ci = comp[i] as usize;
        if sizes[ci] == 1 {
            continue;
        }
        sum_dists.fill(0.0);
        let xi = xs[i].to_f64();
        let yi = ys[i].to_f64();
        for j in 0..n {
            let d = (xi - xs[j].to_f64()).hypot(yi - ys[j].to_f64());
            sum_dists[comp[j] as usize] += d;
        }
        let a_i = sum_dists[ci] / (sizes[ci] - 1) as f64;
        let mut b_i = f64::INFINITY;
        for c in 0..k {
            if c != ci {
                b_i = b_i.min(sum_dists[c] / sizes[c] as f64);
            }
        }
        let denom = a_i.max(b_i);
        if denom > 0.0 {
            total_s += (b_i - a_i) / denom;
        }
    }

    Ok(total_s / n as f64)
}

/// Computes Calinski-Harabasz Index matching `sklearn.metrics.calinski_harabasz_score`.
///
/// Returns `Ok(1.0)` when `intra_disp == 0.0` (exact `sklearn` convention) and `Ok(0.0)` when
/// `N < 3`, `K < 2`, or `K >= N`.
pub fn calinski_harabasz_score<S: Scalar>(layout: &Layout2d<S>, labels: &[u32]) -> CoreResult<f64> {
    validate_layout_labels(layout, labels, "calinski_harabasz_score")?;
    let n = layout.len();
    let (comp, sizes, k) = compact_labels(labels);
    if n < 3 || k < 2 || k >= n {
        return Ok(0.0);
    }

    let xs = layout.x();
    let ys = layout.y();
    let mut mean_x = 0.0_f64;
    let mut mean_y = 0.0_f64;
    let mut cx = vec![0.0_f64; k];
    let mut cy = vec![0.0_f64; k];
    for i in 0..n {
        let x = xs[i].to_f64();
        let y = ys[i].to_f64();
        let c = comp[i] as usize;
        mean_x += x;
        mean_y += y;
        cx[c] += x;
        cy[c] += y;
    }
    mean_x /= n as f64;
    mean_y /= n as f64;
    for c in 0..k {
        let inv = 1.0 / sizes[c] as f64;
        cx[c] *= inv;
        cy[c] *= inv;
    }

    let mut extra_disp = 0.0_f64;
    for c in 0..k {
        let dx = cx[c] - mean_x;
        let dy = cy[c] - mean_y;
        extra_disp += (sizes[c] as f64) * (dx * dx + dy * dy);
    }

    let mut intra_disp = 0.0_f64;
    for i in 0..n {
        let c = comp[i] as usize;
        let dx = xs[i].to_f64() - cx[c];
        let dy = ys[i].to_f64() - cy[c];
        intra_disp += dx * dx + dy * dy;
    }

    if intra_disp == 0.0 {
        return Ok(1.0);
    }
    Ok(extra_disp * ((n - k) as f64) / (intra_disp * ((k - 1) as f64)))
}

/// Computes Davies-Bouldin Index matching `sklearn.metrics.davies_bouldin_score` (lower is better).
///
/// Returns `Ok(0.0)` when `N < 3`, `K < 2`, `K >= N`, all intra-cluster spreads `<= 1e-8`, or all
/// inter-centroid distances `<= 1e-8`.
pub fn davies_bouldin_score<S: Scalar>(layout: &Layout2d<S>, labels: &[u32]) -> CoreResult<f64> {
    validate_layout_labels(layout, labels, "davies_bouldin_score")?;
    let n = layout.len();
    let (comp, sizes, k) = compact_labels(labels);
    if n < 3 || k < 2 || k >= n {
        return Ok(0.0);
    }

    let xs = layout.x();
    let ys = layout.y();
    let mut cx = vec![0.0_f64; k];
    let mut cy = vec![0.0_f64; k];
    for i in 0..n {
        let c = comp[i] as usize;
        cx[c] += xs[i].to_f64();
        cy[c] += ys[i].to_f64();
    }
    for c in 0..k {
        let inv = 1.0 / sizes[c] as f64;
        cx[c] *= inv;
        cy[c] *= inv;
    }

    let mut s = vec![0.0_f64; k];
    for i in 0..n {
        let c = comp[i] as usize;
        s[c] += (xs[i].to_f64() - cx[c]).hypot(ys[i].to_f64() - cy[c]);
    }
    for c in 0..k {
        s[c] /= sizes[c] as f64;
    }

    let all_s_zero = s.iter().all(|&v| v <= 1e-8);
    let mut all_d_zero = true;
    for i in 0..k {
        for j in (i + 1)..k {
            if (cx[i] - cx[j]).hypot(cy[i] - cy[j]) > 1e-8 {
                all_d_zero = false;
                break;
            }
        }
        if !all_d_zero {
            break;
        }
    }
    if all_s_zero || all_d_zero {
        return Ok(0.0);
    }

    let mut sum_max = 0.0_f64;
    for i in 0..k {
        let mut max_r = 0.0_f64;
        for j in 0..k {
            if i == j {
                continue;
            }
            let d_ij = (cx[i] - cx[j]).hypot(cy[i] - cy[j]);
            if d_ij > 0.0 {
                let r_ij = (s[i] + s[j]) / d_ij;
                if r_ij > max_r {
                    max_r = r_ij;
                }
            }
        }
        sum_max += max_r;
    }
    Ok(sum_max / k as f64)
}

#[inline]
fn cross_2d(o: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

#[inline]
fn dist_sq_2d(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

fn convex_hull_ccw(mut pts: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    for p in &mut pts {
        p[0] = canonical_zero(p[0]);
        p[1] = canonical_zero(p[1]);
    }
    pts.sort_unstable_by(|a, b| a[0].total_cmp(&b[0]).then_with(|| a[1].total_cmp(&b[1])));
    pts.dedup_by(|b, a| a[0] == b[0] && a[1] == b[1]);

    if pts.len() <= 2 {
        return pts;
    }

    let mut hull: Vec<[f64; 2]> = Vec::with_capacity(pts.len() + 1);
    for &p in &pts {
        while hull.len() >= 2 && cross_2d(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    let lower_len = hull.len();
    for &p in pts.iter().rev().skip(1) {
        while hull.len() > lower_len
            && cross_2d(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
        {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();
    hull
}

fn euclidean_diameter_2d_core(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    if n <= 1 {
        return 0.0;
    }
    if n <= 64 {
        let mut max_sq = 0.0_f64;
        for i in 0..n {
            let xi = xs[i];
            let yi = ys[i];
            for j in (i + 1)..n {
                let dx = xi - xs[j];
                let dy = yi - ys[j];
                max_sq = max_sq.max(dx * dx + dy * dy);
            }
        }
        return max_sq.sqrt();
    }

    let pts: Vec<[f64; 2]> = (0..n).map(|i| [xs[i], ys[i]]).collect();
    let hull = convex_hull_ccw(pts);
    let m = hull.len();
    if m <= 1 {
        return 0.0;
    }
    if m == 2 {
        return dist_sq_2d(hull[0], hull[1]).sqrt();
    }

    let mut max_sq = 0.0_f64;
    let mut j = 1usize;
    for i in 0..m {
        let next_i = (i + 1) % m;
        loop {
            let next_j = (j + 1) % m;
            let cur_area = cross_2d(hull[i], hull[next_i], hull[j]).abs();
            let next_area = cross_2d(hull[i], hull[next_i], hull[next_j]).abs();
            match next_area.total_cmp(&cur_area) {
                Ordering::Greater => j = next_j,
                Ordering::Equal => {
                    max_sq = max_sq.max(dist_sq_2d(hull[i], hull[next_j]));
                    max_sq = max_sq.max(dist_sq_2d(hull[next_i], hull[next_j]));
                    break;
                }
                Ordering::Less => break,
            }
        }
        max_sq = max_sq.max(dist_sq_2d(hull[i], hull[j]));
        max_sq = max_sq.max(dist_sq_2d(hull[next_i], hull[j]));
    }
    max_sq.sqrt()
}

fn euclidean_diameter_2d<S: Scalar>(layout: &Layout2d<S>) -> f64 {
    let n = layout.len();
    if n <= 1 {
        return 0.0;
    }
    let xs: Vec<f64> = layout.x().iter().map(|v| v.to_f64()).collect();
    let ys: Vec<f64> = layout.y().iter().map(|v| v.to_f64()).collect();
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    let span = (max_x - min_x).max(max_y - min_y);
    if span == 0.0 {
        return 0.0;
    }
    if span < 1e-100 || !span.is_finite() || span > 1e100 {
        let (s_xs, s_ys, back_scale) = if !span.is_finite() || span > 1e100 {
            let half_span = (max_x * 0.5 - min_x * 0.5).max(max_y * 0.5 - min_y * 0.5);
            let s_xs: Vec<f64> = xs
                .iter()
                .map(|&x| (x * 0.5 - min_x * 0.5) / half_span)
                .collect();
            let s_ys: Vec<f64> = ys
                .iter()
                .map(|&y| (y * 0.5 - min_y * 0.5) / half_span)
                .collect();
            (s_xs, s_ys, 2.0 * half_span)
        } else {
            let s_xs: Vec<f64> = xs.iter().map(|&x| (x - min_x) / span).collect();
            let s_ys: Vec<f64> = ys.iter().map(|&y| (y - min_y) / span).collect();
            (s_xs, s_ys, span)
        };
        return euclidean_diameter_2d_core(&s_xs, &s_ys) * back_scale;
    }
    euclidean_diameter_2d_core(&xs, &ys)
}

/// Computes Nocaj et al. (2015) §4.3 Pairwise Distance-Overlap Error (`PDOE`) in `[0.0, 1.0]`
/// (lower is better).
///
/// Partitions all $\binom{N}{2}$ pairs into intra-cluster and inter-cluster sets, constructs
/// normalized empirical distance histograms over `[0, d_max]` with `num_bins` uniform bins,
/// and returns $\sum_b \min(h_{\mathrm{intra}}(b), h_{\mathrm{inter}}(b))$.
pub fn pairwise_distance_overlap_error<S: Scalar>(
    layout: &Layout2d<S>,
    labels: &[u32],
    num_bins: usize,
) -> CoreResult<f64> {
    if num_bins == 0 {
        return Err(CoreError::invalid_input(
            "pairwise_distance_overlap_error requires num_bins >= 1",
        ));
    }
    validate_layout_labels(layout, labels, "pairwise_distance_overlap_error")?;

    let n = layout.len();
    let (comp, _, k) = compact_labels(labels);
    if n < 2 || k <= 1 || k == n {
        return Ok(0.0);
    }

    let d_max = euclidean_diameter_2d(layout);
    if d_max == 0.0 {
        return Ok(1.0);
    }

    let xs = layout.x();
    let ys = layout.y();
    let mut hist_intra = vec![0u64; num_bins];
    let mut hist_inter = vec![0u64; num_bins];
    let mut count_intra = 0u64;
    let mut count_inter = 0u64;
    let num_bins_f = num_bins as f64;

    for i in 0..n {
        let xi = xs[i].to_f64();
        let yi = ys[i].to_f64();
        let ci = comp[i];
        for j in (i + 1)..n {
            let d = (xi - xs[j].to_f64()).hypot(yi - ys[j].to_f64());
            let bin = (((d / d_max) * num_bins_f).floor() as usize).min(num_bins - 1);
            if ci == comp[j] {
                hist_intra[bin] += 1;
                count_intra += 1;
            } else {
                hist_inter[bin] += 1;
                count_inter += 1;
            }
        }
    }

    if count_intra == 0 || count_inter == 0 {
        return Ok(0.0);
    }

    let inv_intra = 1.0 / count_intra as f64;
    let inv_inter = 1.0 / count_inter as f64;
    let mut overlap = 0.0_f64;
    for b in 0..num_bins {
        let hi = hist_intra[b] as f64 * inv_intra;
        let hb = hist_inter[b] as f64 * inv_inter;
        overlap += hi.min(hb);
    }
    Ok(overlap.clamp(0.0, 1.0))
}

struct UnionAreaSegTree {
    ys: Vec<f64>,
    count: Vec<i32>,
    len: Vec<f64>,
}

impl UnionAreaSegTree {
    fn new(ys: Vec<f64>) -> Self {
        let m = ys.len().saturating_sub(1);
        let size = (m * 4).max(4);
        Self {
            ys,
            count: vec![0; size],
            len: vec![0.0; size],
        }
    }

    fn update(&mut self, node: usize, l: usize, r: usize, ql: usize, qr: usize, delta: i32) {
        if ql <= l && r <= qr {
            self.count[node] += delta;
        } else {
            let mid = l + (r - l) / 2;
            if ql < mid {
                self.update(node * 2, l, mid, ql, qr, delta);
            }
            if qr > mid {
                self.update(node * 2 + 1, mid, r, ql, qr, delta);
            }
        }
        if self.count[node] > 0 {
            self.len[node] = self.ys[r] - self.ys[l];
        } else if r - l == 1 {
            self.len[node] = 0.0;
        } else {
            self.len[node] = self.len[node * 2] + self.len[node * 2 + 1];
        }
    }
}

fn rect_union_area(rects: &[[f64; 4]]) -> f64 {
    if rects.is_empty() {
        return 0.0;
    }
    let mut ys = Vec::with_capacity(rects.len() * 2);
    for r in rects {
        ys.push(canonical_zero(r[2]));
        ys.push(canonical_zero(r[3]));
    }
    ys.sort_unstable_by(f64::total_cmp);
    ys.dedup();
    if ys.len() < 2 {
        return 0.0;
    }
    let m = ys.len() - 1;

    let mut events: Vec<(f64, i32, usize, usize)> = Vec::with_capacity(rects.len() * 2);
    for r in rects {
        let y0 = canonical_zero(r[2]);
        let y1 = canonical_zero(r[3]);
        let y_lo = ys.binary_search_by(|v| v.total_cmp(&y0)).unwrap_or(0);
        let y_hi = ys.binary_search_by(|v| v.total_cmp(&y1)).unwrap_or(0);
        if y_lo < y_hi {
            events.push((canonical_zero(r[0]), 1, y_lo, y_hi));
            events.push((canonical_zero(r[1]), -1, y_lo, y_hi));
        }
    }
    if events.is_empty() {
        return 0.0;
    }
    events.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let mut seg = UnionAreaSegTree::new(ys);
    let mut area = 0.0_f64;
    let mut prev_x = events[0].0;
    for &(x, delta, y_lo, y_hi) in &events {
        if x > prev_x {
            area += (x - prev_x) * seg.len[1];
            prev_x = x;
        }
        seg.update(1, 0, m, y_lo, y_hi, delta);
    }
    area
}

/// Computes Liu et al. (2020) `Sprawlter-NN` composite node-node sprawl and area-aware
/// clutter measure ($T_{\mathrm{NN}}(G) = \sqrt{S(G) \cdot \max(A_{\mathrm{NN}}(G), 1)}$)
/// for axis-aligned rectangles $[x_i - hw_i, x_i + hw_i] \times [y_i - hh_i, y_i + hh_i]$.
///
/// Uses $\alpha = 0.20, \gamma = 0.7$, exact $O(N \log N)$ Klee sweep-line union area for $S(G)$,
/// and [`RectangleOverlapSweep::find_layout_overlaps`] for strict interior overlaps (`ox > 0 && oy > 0`).
/// Returns `0.0` for `N == 0` and `1.0` for `N == 1`. Requires strictly positive finite extents.
pub fn sprawlter_nn<S: Scalar>(
    layout: &Layout2d<S>,
    half_widths: &[S],
    half_heights: &[S],
) -> CoreResult<f64> {
    let n = layout.len();
    if half_widths.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            half_widths.len(),
            "sprawlter_nn::half_widths",
        ));
    }
    if half_heights.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            half_heights.len(),
            "sprawlter_nn::half_heights",
        ));
    }
    layout.validate_finite()?;
    for (&hw, &hh) in half_widths.iter().zip(half_heights.iter()) {
        if !hw.is_finite() || !hh.is_finite() {
            return Err(CoreError::non_finite("sprawlter_nn::extents"));
        }
        if hw <= S::ZERO || hh <= S::ZERO {
            return Err(CoreError::invalid_input(
                "sprawlter_nn requires strictly positive half_widths and half_heights",
            ));
        }
    }

    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        return Ok(1.0);
    }

    let xs = layout.x();
    let ys = layout.y();
    let mut rects = Vec::with_capacity(n);
    let mut areas = Vec::with_capacity(n);
    let mut a_min = f64::INFINITY;
    let mut bb_xmin = f64::INFINITY;
    let mut bb_xmax = f64::NEG_INFINITY;
    let mut bb_ymin = f64::INFINITY;
    let mut bb_ymax = f64::NEG_INFINITY;

    for i in 0..n {
        let cx = xs[i].to_f64();
        let cy = ys[i].to_f64();
        let hw = half_widths[i].to_f64();
        let hh = half_heights[i].to_f64();
        let x0 = cx - hw;
        let x1 = cx + hw;
        let y0 = cy - hh;
        let y1 = cy + hh;
        bb_xmin = bb_xmin.min(x0);
        bb_xmax = bb_xmax.max(x1);
        bb_ymin = bb_ymin.min(y0);
        bb_ymax = bb_ymax.max(y1);
        let area_i = 4.0 * hw * hh;
        a_min = a_min.min(area_i);
        areas.push(area_i);
        rects.push([x0, x1, y0, y1]);
    }

    let bbox_area = (bb_xmax - bb_xmin) * (bb_ymax - bb_ymin);
    let union_area = rect_union_area(&rects);
    let sprawl = if union_area <= 0.0 {
        1.0
    } else {
        (bbox_area / union_area).max(1.0)
    };

    let overlaps = RectangleOverlapSweep::find_layout_overlaps(layout, half_widths, half_heights)?;
    let mut a_nn = 0.0_f64;
    if a_min > 0.0 {
        for (i, j) in overlaps {
            let ox = rects[i][1].min(rects[j][1]) - rects[i][0].max(rects[j][0]);
            let oy = rects[i][3].min(rects[j][3]) - rects[i][2].max(rects[j][2]);
            if ox > 0.0 && oy > 0.0 {
                let x = (ox * oy) / a_min;
                let m_ij = areas[i].min(areas[j]) / a_min;
                a_nn += 0.80 * (2.0 * x).powf(0.7) + 0.20 * m_ij.powf(0.7);
            }
        }
    }

    Ok((sprawl * a_nn.max(1.0)).sqrt())
}

/// Uniform-radius square-box convenience wrapper calling [`sprawlter_nn`]`(layout, radii, radii)`.
pub fn sprawlter_nn_radii<S: Scalar>(layout: &Layout2d<S>, radii: &[S]) -> CoreResult<f64> {
    sprawlter_nn(layout, radii, radii)
}

struct Fenwick {
    tree: Vec<u64>,
}

impl Fenwick {
    fn new(n: usize) -> Self {
        Self {
            tree: vec![0; n + 1],
        }
    }

    fn add(&mut self, mut idx: usize, delta: u64) {
        idx += 1;
        while idx < self.tree.len() {
            self.tree[idx] += delta;
            idx += idx.isolate_lowest_one();
        }
    }

    fn prefix_sum(&self, mut idx: usize) -> u64 {
        let mut sum = 0u64;
        idx += 1;
        while idx > 0 {
            sum += self.tree[idx];
            idx -= idx.isolate_lowest_one();
        }
        sum
    }
}

fn axis_inversions_f64(a: &[f64], b: &[f64]) -> u64 {
    let n = a.len();
    if n < 2 {
        return 0;
    }

    let mut order_b: Vec<usize> = (0..n).collect();
    order_b.sort_unstable_by(|&i, &j| canonical_zero(b[i]).total_cmp(&canonical_zero(b[j])));
    let mut rank_b = vec![0usize; n];
    let mut cur_rank = 0usize;
    rank_b[order_b[0]] = 0;
    for pos in 1..n {
        if canonical_zero(b[order_b[pos]]).total_cmp(&canonical_zero(b[order_b[pos - 1]]))
            != Ordering::Equal
        {
            cur_rank += 1;
        }
        rank_b[order_b[pos]] = cur_rank;
    }

    let mut order_a: Vec<usize> = (0..n).collect();
    order_a.sort_unstable_by(|&i, &j| canonical_zero(a[i]).total_cmp(&canonical_zero(a[j])));

    let mut bit = Fenwick::new(cur_rank + 1);
    let mut inserted = 0u64;
    let mut inversions = 0u64;
    let mut start = 0usize;
    while start < n {
        let val = canonical_zero(a[order_a[start]]);
        let mut end = start + 1;
        while end < n && canonical_zero(a[order_a[end]]).total_cmp(&val) == Ordering::Equal {
            end += 1;
        }
        for &idx in &order_a[start..end] {
            let le_count = bit.prefix_sum(rank_b[idx]);
            inversions += inserted - le_count;
        }
        for &idx in &order_a[start..end] {
            bit.add(rank_b[idx], 1);
            inserted += 1;
        }
        start = end;
    }

    inversions
}

fn polygon_area(poly: &[[f64; 2]]) -> f64 {
    let m = poly.len();
    if m < 3 {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    for i in 0..m {
        let j = (i + 1) % m;
        sum += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1];
    }
    0.5 * sum.abs()
}

fn corners_convex_hull_area<S: Scalar>(
    layout: &Layout2d<S>,
    half_widths: &[S],
    half_heights: &[S],
) -> f64 {
    let n = layout.len();
    if n == 0 {
        return 0.0;
    }
    let xs = layout.x();
    let ys = layout.y();
    let mut corners = Vec::with_capacity(n * 4);
    for i in 0..n {
        let x = xs[i].to_f64();
        let y = ys[i].to_f64();
        let hw = half_widths[i].to_f64();
        let hh = half_heights[i].to_f64();
        corners.push([x - hw, y - hh]);
        corners.push([x + hw, y - hh]);
        corners.push([x + hw, y + hh]);
        corners.push([x - hw, y + hh]);
    }
    let hull = convex_hull_ccw(corners);
    polygon_area(&hull)
}

fn box_inclusive_bbox<S: Scalar>(
    layout: &Layout2d<S>,
    half_widths: &[S],
    half_heights: &[S],
) -> (f64, f64) {
    let xs = layout.x();
    let ys = layout.y();
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    for i in 0..layout.len() {
        let x = xs[i].to_f64();
        let y = ys[i].to_f64();
        let hw = half_widths[i].to_f64();
        let hh = half_heights[i].to_f64();
        xmin = xmin.min(x - hw);
        xmax = xmax.max(x + hw);
        ymin = ymin.min(y - hh);
        ymax = ymax.max(y + hh);
    }
    (xmax - xmin, ymax - ymin)
}

fn center_bbox(xs: &[f64], ys: &[f64]) -> (f64, f64, f64, f64) {
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        xmin = xmin.min(x);
        xmax = xmax.max(x);
        ymin = ymin.min(y);
        ymax = ymax.max(y);
    }
    (
        0.5 * (xmin + xmax),
        0.5 * (ymin + ymax),
        xmax - xmin,
        ymax - ymin,
    )
}

fn nor_metrics_core<S: Scalar>(
    orig_layout: &Layout2d<S>,
    adjusted_layout: &Layout2d<S>,
    half_widths: &[S],
    half_heights: &[S],
    edges: Option<&[(u32, u32)]>,
) -> CoreResult<NorMetrics> {
    let n = orig_layout.len();
    if adjusted_layout.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            adjusted_layout.len(),
            "nor_metrics::adjusted_layout",
        ));
    }
    if half_widths.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            half_widths.len(),
            "nor_metrics::half_widths",
        ));
    }
    if half_heights.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            half_heights.len(),
            "nor_metrics::half_heights",
        ));
    }
    orig_layout.validate_finite()?;
    adjusted_layout.validate_finite()?;
    for (&hw, &hh) in half_widths.iter().zip(half_heights.iter()) {
        if !hw.is_finite() || !hh.is_finite() {
            return Err(CoreError::non_finite("nor_metrics::extents"));
        }
        if hw < S::ZERO || hh < S::ZERO {
            return Err(CoreError::invalid_input(
                "nor_metrics requires non-negative half_widths and half_heights",
            ));
        }
    }

    if n == 0 {
        return Ok(NorMetrics::default());
    }
    if n == 1 {
        return Ok(NorMetrics {
            oo_nni: 0.0,
            sp_ch_a: 1.0,
            gs_bb_iar: 1.0,
            nm_dm_imse: 0.0,
            el_rsd: 0.0,
        });
    }

    let orig_x: Vec<f64> = orig_layout.x().iter().map(|v| v.to_f64()).collect();
    let orig_y: Vec<f64> = orig_layout.y().iter().map(|v| v.to_f64()).collect();
    let adj_x: Vec<f64> = adjusted_layout.x().iter().map(|v| v.to_f64()).collect();
    let adj_y: Vec<f64> = adjusted_layout.y().iter().map(|v| v.to_f64()).collect();

    let inv_x = axis_inversions_f64(&orig_x, &adj_x);
    let inv_y = axis_inversions_f64(&orig_y, &adj_y);
    let oo_nni = (inv_x + inv_y) as f64 / (n * (n - 1)) as f64;

    let orig_ch_area = corners_convex_hull_area(orig_layout, half_widths, half_heights);
    let adj_ch_area = corners_convex_hull_area(adjusted_layout, half_widths, half_heights);
    let sp_ch_a = if orig_ch_area <= 1e-18 {
        1.0
    } else {
        adj_ch_area / orig_ch_area
    };

    let (w_orig, h_orig) = box_inclusive_bbox(orig_layout, half_widths, half_heights);
    let (w_adj, h_adj) = box_inclusive_bbox(adjusted_layout, half_widths, half_heights);
    let gs_bb_iar = if w_orig <= 1e-18 || h_orig <= 1e-18 || w_adj <= 1e-18 || h_adj <= 1e-18 {
        1.0
    } else {
        let r1 = (w_adj * h_orig) / (h_adj * w_orig);
        let r2 = (h_adj * w_orig) / (w_adj * h_orig);
        r1.max(r2)
    };

    let (cx_o, cy_o, w_o, h_o) = center_bbox(&orig_x, &orig_y);
    let (cx_a, cy_a, w_a, h_a) = center_bbox(&adj_x, &adj_y);
    let sx = if w_o > 1e-18 { w_a / w_o } else { 1.0 };
    let sy = if h_o > 1e-18 { h_a / h_o } else { 1.0 };
    let mut sum_sq_err = 0.0_f64;
    for i in 0..n {
        let tx = cx_a + (orig_x[i] - cx_o) * sx;
        let ty = cy_a + (orig_y[i] - cy_o) * sy;
        let dx = adj_x[i] - tx;
        let dy = adj_y[i] - ty;
        sum_sq_err += dx * dx + dy * dy;
    }
    let nm_dm_imse = sum_sq_err / n as f64;

    let el_rsd = if let Some(edge_list) = edges {
        let mut ratios = Vec::with_capacity(edge_list.len());
        let mut sum_r = 0.0_f64;
        for &(u, v) in edge_list {
            let ui = u as usize;
            let vi = v as usize;
            let len_o = (orig_x[ui] - orig_x[vi]).hypot(orig_y[ui] - orig_y[vi]);
            if len_o > 1e-18 {
                let len_a = (adj_x[ui] - adj_x[vi]).hypot(adj_y[ui] - adj_y[vi]);
                let r = len_a / len_o;
                ratios.push(r);
                sum_r += r;
            }
        }
        if ratios.is_empty() {
            0.0
        } else {
            let mean_r = sum_r / ratios.len() as f64;
            if mean_r <= 1e-18 {
                0.0
            } else {
                let var = ratios
                    .iter()
                    .map(|&r| {
                        let d = r - mean_r;
                        d * d
                    })
                    .sum::<f64>()
                    / ratios.len() as f64;
                var.sqrt() / mean_r
            }
        }
    } else {
        0.0
    };

    Ok(NorMetrics {
        oo_nni,
        sp_ch_a,
        gs_bb_iar,
        nm_dm_imse,
        el_rsd,
    })
}

/// Computes Chen et al. (2020) 5-metric Node Overlap Removal (`NOR`) evaluation including
/// edge length preservation (`el_rsd`) over unique non-self-loop edges of `graph`.
pub fn nor_metrics<W, S: Scalar>(
    graph: &CsrGraph<W>,
    orig_layout: &Layout2d<S>,
    adjusted_layout: &Layout2d<S>,
    half_widths: &[S],
    half_heights: &[S],
) -> CoreResult<NorMetrics> {
    validate_graph_layout(graph, orig_layout, "nor_metrics::orig_layout")?;
    validate_graph_layout(graph, adjusted_layout, "nor_metrics::adjusted_layout")?;
    let edges = extract_unique_edges(graph);
    nor_metrics_core(
        orig_layout,
        adjusted_layout,
        half_widths,
        half_heights,
        Some(&edges),
    )
}

/// Computes Chen et al. (2020) 5-metric Node Overlap Removal (`NOR`) evaluation without a graph
/// (setting `el_rsd = 0.0`).
pub fn nor_metrics_without_graph<S: Scalar>(
    orig_layout: &Layout2d<S>,
    adjusted_layout: &Layout2d<S>,
    half_widths: &[S],
    half_heights: &[S],
) -> CoreResult<NorMetrics> {
    nor_metrics_core(
        orig_layout,
        adjusted_layout,
        half_widths,
        half_heights,
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use egraph_core::DeterministicRng;

    fn assert_close(got: f64, expected: f64, tol: f64, name: &str) {
        assert!(
            (got - expected).abs() < tol,
            "{name}: got {got}, expected {expected} (diff = {})",
            (got - expected).abs()
        );
    }

    fn layout_from_pairs(pts: &[(f64, f64)]) -> Layout2d<f64> {
        let xs: Vec<f64> = pts.iter().map(|p| p.0).collect();
        let ys: Vec<f64> = pts.iter().map(|p| p.1).collect();
        Layout2d::from_xy(xs, ys).unwrap()
    }

    fn brute_oo_nni(orig: &Layout2d<f64>, adj: &Layout2d<f64>) -> f64 {
        let n = orig.len();
        if n <= 1 {
            return 0.0;
        }
        let mut inv = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                if (orig.x()[i] - orig.x()[j]) * (adj.x()[i] - adj.x()[j]) < 0.0 {
                    inv += 1;
                }
                if (orig.y()[i] - orig.y()[j]) * (adj.y()[i] - adj.y()[j]) < 0.0 {
                    inv += 1;
                }
            }
        }
        inv as f64 / (n * (n - 1)) as f64
    }

    fn brute_gong_edges(layout: &Layout2d<f64>, gamma: f64) -> Vec<(usize, usize)> {
        let n = layout.len();
        let xs = layout.x();
        let ys = layout.y();
        let mut edges = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dx = xs[j] - xs[i];
                let dy = ys[j] - ys[i];
                let d_ij_sq = dx * dx + dy * dy;
                if d_ij_sq <= 0.0 {
                    continue;
                }
                let mx = gamma * xs[j] + (1.0 - gamma) * xs[i];
                let my = gamma * ys[j] + (1.0 - gamma) * ys[i];
                let r_sq = (1.0 - gamma).powi(2) * d_ij_sq;
                let mut ok = true;
                for q in 0..n {
                    if q == i || q == j {
                        continue;
                    }
                    let d_mq_sq = (xs[q] - mx).powi(2) + (ys[q] - my).powi(2);
                    if d_mq_sq <= r_sq * (1.0 + 1e-12) {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    edges.push((i, j));
                }
            }
        }
        edges
    }

    #[test]
    fn test_fixture_a_b_c_sklearn_scipy_exact_parity() {
        // Fixture A: N = 12, K = 3 uneven [6, 4, 2].
        let coords_a = [
            (0.0, 0.0),
            (0.4, 0.1),
            (0.1, 0.5),
            (-0.3, 0.2),
            (0.2, -0.3),
            (0.5, 0.4),
            (5.0, 0.2),
            (5.4, 0.0),
            (5.1, 0.6),
            (4.8, -0.2),
            (2.5, 4.5),
            (2.8, 4.9),
        ];
        let layout_a = layout_from_pairs(&coords_a);
        let labels_a = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2];
        let pred_a = [0, 0, 0, 0, 1, 0, 1, 1, 1, 2, 2, 2];

        let pa_a = partition_agreement(&labels_a, &pred_a).unwrap();
        assert_close(pa_a.ari, 0.541_176_470_588_235_3, 1e-10, "A.ari");
        assert_close(pa_a.ami, 0.534_654_132_249_696_5, 1e-10, "A.ami");
        assert_close(pa_a.homogeneity, 0.657_334_260_519_206_7, 1e-10, "A.hom");
        assert_close(pa_a.completeness, 0.616_979_973_786_710_7, 1e-10, "A.cmp");
        assert_close(pa_a.v_measure, 0.636_518_158_404_069_2, 1e-10, "A.vm");
        assert_close(pa_a.fmi, 0.684_762_383_262_325_9, 1e-10, "A.fmi");

        let hac_a = cq_hac(&layout_a, &labels_a).unwrap();
        assert_close(hac_a.ari, 1.0, 1e-10, "A.hac.ari");
        assert_close(hac_a.ami, 1.0, 1e-10, "A.hac.ami");
        assert_close(hac_a.homogeneity, 1.0, 1e-10, "A.hac.hom");
        assert_close(hac_a.completeness, 1.0, 1e-10, "A.hac.cmp");
        assert_close(hac_a.v_measure, 1.0, 1e-10, "A.hac.vm");
        assert_close(hac_a.fmi, 1.0, 1e-10, "A.hac.fmi");

        let km_a = cq_kmeans(&layout_a, &labels_a, 5, 100, 42).unwrap();
        assert_close(km_a.ari, 1.0, 1e-10, "A.kmeans.ari");

        assert_close(
            silhouette_score(&layout_a, &labels_a).unwrap(),
            0.888_186_961_713_074_7,
            1e-10,
            "A.silhouette",
        );
        assert_close(
            calinski_harabasz_score(&layout_a, &labels_a).unwrap(),
            280.962_311_557_788_95,
            1e-10,
            "A.ch",
        );
        assert_close(
            davies_bouldin_score(&layout_a, &labels_a).unwrap(),
            0.133_252_117_690_975_3,
            1e-10,
            "A.db",
        );

        // Fixture B: N = 20, K = 4 overlapping [7, 6, 4, 3].
        let x_b = vec![
            -1.1897, -2.8445, 0.9574, -0.4679, -0.4015, 1.0801, -0.0962, 0.2414, -0.6056, -0.1957,
            -0.7159, -0.3484, 0.9628, -1.0698, -1.845, -1.7543, 1.2509, -1.3283, -0.7785, 0.7598,
        ];
        let y_b = vec![
            0.3609, 2.0937, -0.4381, 0.4558, -0.3389, 0.7721, -0.1282, -0.921, 0.8224, -2.0616,
            0.9849, -0.2231, 2.7369, 2.0223, 0.2625, 2.0272, 1.7066, 1.0268, -0.6861, 1.3151,
        ];
        let layout_b = Layout2d::from_xy(x_b, y_b).unwrap();
        let labels_b = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3];
        let pred_b = [0, 0, 1, 0, 2, 0, 0, 1, 1, 0, 1, 3, 1, 2, 2, 1, 2, 3, 3, 2];

        let pa_b = partition_agreement(&labels_b, &pred_b).unwrap();
        assert_close(pa_b.ari, 0.290_272_373_540_856_05, 1e-10, "B.ari");
        assert_close(pa_b.ami, 0.297_072_950_635_337_3, 1e-10, "B.ami");
        assert_close(pa_b.homogeneity, 0.454_355_339_790_888_9, 1e-10, "B.hom");
        assert_close(pa_b.completeness, 0.448_165_310_990_085, 1e-10, "B.cmp");
        assert_close(pa_b.v_measure, 0.451_239_097_922_438_6, 1e-10, "B.vm");
        assert_close(pa_b.fmi, 0.454_662_892_980_315_65, 1e-10, "B.fmi");

        let hac_b = cq_hac(&layout_b, &labels_b).unwrap();
        assert_close(hac_b.ari, 0.005_607_476_635_514_018, 1e-10, "B.hac.ari");
        assert_close(hac_b.ami, -0.046_981_511_733_698_73, 1e-10, "B.hac.ami");
        assert_close(
            hac_b.homogeneity,
            0.165_036_847_012_176_2,
            1e-10,
            "B.hac.hom",
        );
        assert_close(
            hac_b.completeness,
            0.207_330_815_002_267_4,
            1e-10,
            "B.hac.cmp",
        );
        assert_close(hac_b.v_measure, 0.183_781_931_069_577_02, 1e-10, "B.hac.vm");
        assert_close(hac_b.fmi, 0.309_838_667_696_593_35, 1e-10, "B.hac.fmi");

        assert_close(
            silhouette_score(&layout_b, &labels_b).unwrap(),
            -0.155_875_228_260_980_2,
            1e-10,
            "B.silhouette",
        );
        assert_close(
            calinski_harabasz_score(&layout_b, &labels_b).unwrap(),
            0.692_974_573_303_605_2,
            1e-10,
            "B.ch",
        );
        assert_close(
            davies_bouldin_score(&layout_b, &labels_b).unwrap(),
            10.141_407_555_771_2,
            1e-10,
            "B.db",
        );

        assert_close(
            pairwise_distance_overlap_error(&layout_b, &labels_b, 10).unwrap(),
            0.880_459_770_114_942_7,
            1e-10,
            "B.pdoe",
        );
        assert_close(
            neighborhood_hit(&layout_b, &labels_b, 7).unwrap(),
            0.242_857_142_857_142_85,
            1e-10,
            "B.m_nh",
        );
        assert_close(
            sepme_knng(&layout_b, &labels_b, 2).unwrap(),
            0.178_571_428_571_428_58,
            1e-10,
            "B.sepme_knng2",
        );
        assert_close(
            sepme_rng(&layout_b, &labels_b).unwrap(),
            0.212_301_587_301_587_3,
            1e-10,
            "B.sepme_rng",
        );
        assert_close(
            sepme_gab(&layout_b, &labels_b).unwrap(),
            0.189_037_698_412_698_4,
            1e-10,
            "B.sepme_gab",
        );
        assert_close(
            sepme_gong(&layout_b, &labels_b, 0.35).unwrap(),
            0.223_214_285_714_285_7,
            1e-10,
            "B.sepme_gong",
        );
        assert_close(
            distance_consistency(&layout_b, &labels_b).unwrap(),
            0.3,
            1e-10,
            "B.dsc",
        );

        // Fixture C: Degenerate partition_agreement & CVI edge cases.
        let c_cases: &[(&[u32], &[u32], [f64; 6])] = &[
            (&[], &[], [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
            (&[0], &[0], [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
            (&[0, 0, 0, 0], &[1, 1, 1, 1], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            (
                &[0, 0, 0, 0],
                &[0, 1, 0, 1],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.5773502691896257],
            ),
            (&[0, 1, 2, 3], &[0, 0, 0, 0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            (&[0, 1, 2, 3], &[3, 2, 1, 0], [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
            (
                &[0, 0, 1, 1],
                &[0, 1, 0, 1],
                [-0.5, -0.49999999999999944, 0.0, 0.0, 0.0, 0.0],
            ),
        ];
        for (idx, &(lt, lp, exp)) in c_cases.iter().enumerate() {
            let got = partition_agreement(lt, lp).unwrap();
            assert_close(got.ari, exp[0], 1e-10, &format!("C[{idx}].ari"));
            assert_close(got.ami, exp[1], 1e-10, &format!("C[{idx}].ami"));
            assert_close(got.homogeneity, exp[2], 1e-10, &format!("C[{idx}].hom"));
            assert_close(got.completeness, exp[3], 1e-10, &format!("C[{idx}].cmp"));
            assert_close(got.v_measure, exp[4], 1e-10, &format!("C[{idx}].vm"));
            assert_close(got.fmi, exp[5], 1e-10, &format!("C[{idx}].fmi"));
        }

        let l_ident = layout_from_pairs(&[(0.0, 0.0); 4]);
        let y_ident = [0, 0, 1, 1];
        assert_close(
            silhouette_score(&l_ident, &y_ident).unwrap(),
            0.0,
            1e-10,
            "C.ident.sil",
        );
        assert_close(
            calinski_harabasz_score(&l_ident, &y_ident).unwrap(),
            1.0,
            1e-10,
            "C.ident.ch",
        );
        assert_close(
            davies_bouldin_score(&l_ident, &y_ident).unwrap(),
            0.0,
            1e-10,
            "C.ident.db",
        );

        let l_icent = layout_from_pairs(&[(0.0, 0.0), (2.0, 0.0), (0.0, 0.0), (2.0, 0.0)]);
        assert_close(
            silhouette_score(&l_icent, &y_ident).unwrap(),
            -0.5,
            1e-10,
            "C.icent.sil",
        );
        assert_close(
            calinski_harabasz_score(&l_icent, &y_ident).unwrap(),
            0.0,
            1e-10,
            "C.icent.ch",
        );
        assert_close(
            davies_bouldin_score(&l_icent, &y_ident).unwrap(),
            0.0,
            1e-10,
            "C.icent.db",
        );

        let l_col = layout_from_pairs(&[(0.0, 0.0), (1.0, 0.0), (10.0, 0.0), (11.0, 0.0)]);
        assert_close(
            silhouette_score(&l_col, &y_ident).unwrap(),
            0.899749373433584,
            1e-10,
            "C.col.sil",
        );
        assert_close(
            calinski_harabasz_score(&l_col, &y_ident).unwrap(),
            200.0,
            1e-10,
            "C.col.ch",
        );
        assert_close(
            davies_bouldin_score(&l_col, &y_ident).unwrap(),
            0.1,
            1e-10,
            "C.col.db",
        );

        let l_sing = layout_from_pairs(&[(0.0, 0.0), (0.1, 0.0), (10.0, 0.0)]);
        let y_sing = [0, 0, 1];
        assert_close(
            silhouette_score(&l_sing, &y_sing).unwrap(),
            0.65996632996633,
            1e-10,
            "C.sing.sil",
        );
        assert_close(
            calinski_harabasz_score(&l_sing, &y_sing).unwrap(),
            13200.33333333333,
            1e-10,
            "C.sing.ch",
        );
        assert_close(
            davies_bouldin_score(&l_sing, &y_sing).unwrap(),
            0.0050251256281407045,
            1e-10,
            "C.sing.db",
        );
    }

    #[test]
    fn test_fixture_d_nor_and_sprawlter_nn_exact_parity() {
        let orig_d = layout_from_pairs(&[(0.0, 0.0), (0.5, 0.2), (2.0, 0.0), (2.2, 1.5)]);
        let adj_d = layout_from_pairs(&[(0.0, 0.0), (1.2, 0.4), (3.0, 0.1), (3.2, 2.0)]);
        let hw_d = [0.5, 0.5, 0.4, 0.6];
        let hh_d = [0.4, 0.4, 0.3, 0.5];
        let g_d =
            CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3), (0, 2)], None, false, false)
                .unwrap();

        let nor = nor_metrics(&g_d, &orig_d, &adj_d, &hw_d, &hh_d).unwrap();
        assert_close(nor.oo_nni, 0.0, 1e-10, "D.oo_nni");
        assert_close(nor.sp_ch_a, 1.455_089_820_359_281_7, 1e-10, "D.sp_ch_a");
        assert_close(nor.gs_bb_iar, 1.078_369_905_956_113, 1e-10, "D.gs_bb_iar");
        assert_close(
            nor.nm_dm_imse,
            0.064_878_328_741_965_13,
            1e-10,
            "D.nm_dm_imse",
        );
        assert_close(nor.el_rsd, 0.289_816_821_614_026_05, 1e-10, "D.el_rsd");

        let nor_ng = nor_metrics_without_graph(&orig_d, &adj_d, &hw_d, &hh_d).unwrap();
        assert_close(nor_ng.oo_nni, nor.oo_nni, 1e-15, "D.ng.oo_nni");
        assert_close(nor_ng.sp_ch_a, nor.sp_ch_a, 1e-15, "D.ng.sp_ch_a");
        assert_close(nor_ng.gs_bb_iar, nor.gs_bb_iar, 1e-15, "D.ng.gs_bb_iar");
        assert_close(nor_ng.nm_dm_imse, nor.nm_dm_imse, 1e-15, "D.ng.nm_dm_imse");
        assert_eq!(nor_ng.el_rsd, 0.0);

        assert_close(
            sprawlter_nn(&orig_d, &hw_d, &hh_d).unwrap(),
            1.8015717518758179,
            1e-10,
            "D.sprawlter.orig",
        );
        assert_close(
            sprawlter_nn(&adj_d, &hw_d, &hh_d).unwrap(),
            1.9498280099261789,
            1e-10,
            "D.sprawlter.adj",
        );

        // Translation and uniform scaling invariance for nm_dm_imse.
        let scaled_translated = Layout2d::from_xy(
            orig_d.x().iter().map(|&x| 3.5 * x + 12.0).collect(),
            orig_d.y().iter().map(|&y| 3.5 * y - 7.0).collect(),
        )
        .unwrap();
        let inv_nor = nor_metrics(&g_d, &orig_d, &scaled_translated, &hw_d, &hh_d).unwrap();
        assert!(inv_nor.nm_dm_imse < 1e-24);
        assert!(inv_nor.el_rsd < 1e-14);
    }

    #[test]
    fn test_dc_cdm_hdm_dsc_separation_and_properties() {
        // Well-separated clusters vs fully mixed clusters.
        let well_sep = layout_from_pairs(&[
            (0.0, 0.0),
            (0.1, 0.0),
            (0.0, 0.1),
            (0.1, 0.1),
            (10.0, 10.0),
            (10.1, 10.0),
            (10.0, 10.1),
            (10.1, 10.1),
        ]);
        let labels = [10_u32, 10, 10, 10, 42, 42, 42, 42]; // sparse non-0..K labels

        assert_close(
            distance_consistency(&well_sep, &labels).unwrap(),
            1.0,
            1e-12,
            "DSC.sep",
        );
        assert_close(
            distribution_consistency(&well_sep, &labels, 20).unwrap(),
            1.0,
            1e-12,
            "DC.sep",
        );
        let (cdm_sep, hdm_sep) = class_density_measure(&well_sep, &labels, 64, None).unwrap();
        assert!(cdm_sep > 0.99, "cdm_sep = {cdm_sep}");
        assert_close(hdm_sep, 1.0, 1e-12, "HDM.sep");

        // Equidistant / identical centroids -> DSC strict `<` gives 0.0 regardless of label order.
        let tied = layout_from_pairs(&[(0.0, 0.0), (2.0, 0.0), (0.0, 0.0), (2.0, 0.0)]);
        let tied_labels = [0, 0, 1, 1];
        assert_eq!(distance_consistency(&tied, &tied_labels).unwrap(), 0.0);
        assert_eq!(
            distribution_consistency(&tied, &tied_labels, 20).unwrap(),
            0.0
        );
        let (cdm_tied, hdm_tied) = class_density_measure(&tied, &tied_labels, 32, None).unwrap();
        assert!(cdm_tied < 1e-12);
        assert!(hdm_tied < 1e-12);

        // Collapsed single-axis layout must not divide by zero or panic.
        let col = layout_from_pairs(&[(0.0, 5.0), (0.0, 5.0), (1.0, 5.0), (1.0, 5.0)]);
        let dc_col = distribution_consistency(&col, &tied_labels, 16).unwrap();
        assert!((0.0..=1.0).contains(&dc_col));
        let (cdm_col, hdm_col) =
            class_density_measure(&col, &tied_labels, 16, Some(0.001)).unwrap();
        assert!((0.0..=1.0).contains(&cdm_col));
        assert!((0.0..=1.0).contains(&hdm_col));
    }

    #[test]
    fn test_gong_and_oo_nni_brute_force_oracles() {
        for seed in [7_u64, 19] {
            let mut rng = DeterministicRng::from_seed_u64(seed);
            let n = 45usize;
            let xs: Vec<f64> = (0..n).map(|_| rng.next_f64() * 10.0 - 5.0).collect();
            let ys: Vec<f64> = (0..n).map(|_| rng.next_f64() * 10.0 - 5.0).collect();
            let orig = Layout2d::from_xy(xs, ys).unwrap();
            let adj_x: Vec<f64> = orig
                .x()
                .iter()
                .map(|&v| v + (rng.next_f64() - 0.5) * 3.0)
                .collect();
            let adj_y: Vec<f64> = orig
                .y()
                .iter()
                .map(|&v| v + (rng.next_f64() - 0.5) * 3.0)
                .collect();
            let adj = Layout2d::from_xy(adj_x, adj_y).unwrap();
            let hw = vec![0.25_f64; n];
            let hh = vec![0.25_f64; n];

            let nor = nor_metrics_without_graph(&orig, &adj, &hw, &hh).unwrap();
            assert_close(
                nor.oo_nni,
                brute_oo_nni(&orig, &adj),
                1e-12,
                "oo_nni brute parity",
            );

            let labels: Vec<u32> = (0..n).map(|i| (i % 3) as u32).collect();
            let (comp, sizes, k_classes) = compact_labels(&labels);
            for gamma in [0.0, 0.35, 0.5, 0.75] {
                let b_edges = brute_gong_edges(&orig, gamma);
                let mut deg = vec![0usize; n];
                let mut same = vec![0usize; n];
                for &(u, v) in &b_edges {
                    deg[u] += 1;
                    if comp[u] == comp[v] {
                        same[u] += 1;
                    }
                }
                let expected = dir_cpt_from_degree_counts(&comp, &sizes, k_classes, &deg, &same);
                let got = sepme_gong(&orig, &labels, gamma).unwrap();
                assert_close(got, expected, 1e-12, &format!("gong({gamma})"));
            }
        }

        // Ties in coordinates (including -0.0 vs +0.0) for oo_nni.
        let tie_orig = layout_from_pairs(&[(-0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, -0.0)]);
        let tie_adj = layout_from_pairs(&[(1.0, 0.0), (0.0, 1.0), (0.5, 1.0), (0.2, 0.0)]);
        let hw = [0.1; 4];
        let nor_tie = nor_metrics_without_graph(&tie_orig, &tie_adj, &hw, &hw).unwrap();
        assert_close(
            nor_tie.oo_nni,
            brute_oo_nni(&tie_orig, &tie_adj),
            1e-12,
            "oo_nni tie parity",
        );
    }

    #[test]
    fn test_degenerate_f32_and_error_cases() {
        let empty = Layout2d::<f64>::zeros(0);
        let empty_l: [u32; 0] = [];
        let empty_s: [f64; 0] = [];
        let empty_g = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();

        assert_eq!(
            cq_kmeans(&empty, &empty_l, 5, 50, 42).unwrap(),
            PartitionAgreement::default()
        );
        assert_eq!(
            cq_hac(&empty, &empty_l).unwrap(),
            PartitionAgreement::default()
        );
        assert_eq!(distance_consistency(&empty, &empty_l).unwrap(), 0.0);
        assert_eq!(distribution_consistency(&empty, &empty_l, 10).unwrap(), 0.0);
        assert_eq!(
            class_density_measure(&empty, &empty_l, 10, None).unwrap(),
            (0.0, 0.0)
        );
        assert_eq!(sepme_knng(&empty, &empty_l, 2).unwrap(), 0.0);
        assert_eq!(sepme_rng(&empty, &empty_l).unwrap(), 0.0);
        assert_eq!(sepme_gab(&empty, &empty_l).unwrap(), 0.0);
        assert_eq!(sepme_gong(&empty, &empty_l, 0.35).unwrap(), 0.0);
        assert_eq!(neighborhood_hit(&empty, &empty_l, 7).unwrap(), 0.0);
        assert_eq!(silhouette_score(&empty, &empty_l).unwrap(), 0.0);
        assert_eq!(calinski_harabasz_score(&empty, &empty_l).unwrap(), 0.0);
        assert_eq!(davies_bouldin_score(&empty, &empty_l).unwrap(), 0.0);
        assert_eq!(
            pairwise_distance_overlap_error(&empty, &empty_l, 10).unwrap(),
            0.0
        );
        assert_eq!(sprawlter_nn(&empty, &empty_s, &empty_s).unwrap(), 0.0);
        assert_eq!(sprawlter_nn_radii(&empty, &empty_s).unwrap(), 0.0);
        assert_eq!(
            nor_metrics(&empty_g, &empty, &empty, &empty_s, &empty_s).unwrap(),
            NorMetrics::default()
        );

        let single = Layout2d::from_xy(vec![1.0_f32], vec![2.0_f32]).unwrap();
        let single_l = [0_u32];
        let single_r = [0.5_f32];
        assert_eq!(
            cq_kmeans(&single, &single_l, 3, 10, 42).unwrap(),
            PartitionAgreement::default()
        );
        assert_eq!(
            cq_hac(&single, &single_l).unwrap(),
            PartitionAgreement::default()
        );
        assert_eq!(distance_consistency(&single, &single_l).unwrap(), 0.0);
        assert_eq!(
            distribution_consistency(&single, &single_l, 10).unwrap(),
            0.0
        );
        assert_eq!(
            class_density_measure(&single, &single_l, 10, None).unwrap(),
            (0.0, 0.0)
        );
        assert_eq!(sepme_knng(&single, &single_l, 2).unwrap(), 1.0);
        assert_eq!(sepme_rng(&single, &single_l).unwrap(), 1.0);
        assert_eq!(sepme_gab(&single, &single_l).unwrap(), 1.0);
        assert_eq!(sepme_gong(&single, &single_l, 0.35).unwrap(), 1.0);
        assert_eq!(neighborhood_hit(&single, &single_l, 7).unwrap(), 0.0);
        assert_eq!(silhouette_score(&single, &single_l).unwrap(), 0.0);
        assert_eq!(calinski_harabasz_score(&single, &single_l).unwrap(), 0.0);
        assert_eq!(davies_bouldin_score(&single, &single_l).unwrap(), 0.0);
        assert_eq!(
            pairwise_distance_overlap_error(&single, &single_l, 10).unwrap(),
            0.0
        );
        assert_eq!(sprawlter_nn_radii(&single, &single_r).unwrap(), 1.0);
        let nor1 = nor_metrics_without_graph(&single, &single, &single_r, &single_r).unwrap();
        assert_eq!(nor1.sp_ch_a, 1.0);
        assert_eq!(nor1.gs_bb_iar, 1.0);

        // f32 multi-cluster smoke check across all metrics.
        let l32 = Layout2d::from_xy(
            vec![0.0_f32, 0.2, 0.1, 5.0, 5.2, 5.1],
            vec![0.0_f32, 0.1, -0.1, 5.0, 5.1, 4.9],
        )
        .unwrap();
        let lbl32 = [0_u32, 0, 0, 1, 1, 1];
        assert_close(
            cq_kmeans(&l32, &lbl32, 3, 50, 42).unwrap().ari,
            1.0,
            1e-10,
            "f32.km",
        );
        assert_close(cq_hac(&l32, &lbl32).unwrap().ari, 1.0, 1e-10, "f32.hac");
        assert_close(
            distance_consistency(&l32, &lbl32).unwrap(),
            1.0,
            1e-10,
            "f32.dsc",
        );
        assert!(silhouette_score(&l32, &lbl32).unwrap() > 0.9);
        assert_eq!(
            pairwise_distance_overlap_error(&l32, &lbl32, 20).unwrap(),
            0.0
        );
        let r32 = [0.25_f32; 6];
        assert!(sprawlter_nn_radii(&l32, &r32).unwrap().is_finite());

        // Validation errors.
        let l2 = Layout2d::from_xy(vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();
        let bad_nan = Layout2d::from_xy(vec![f64::NAN, 1.0], vec![0.0, 1.0]).unwrap();
        assert!(matches!(
            partition_agreement(&[0], &[0, 1]),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            cq_kmeans(&l2, &[0, 1], 0, 10, 42),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            cq_kmeans(&l2, &[0, 1], 1, 0, 42),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            distribution_consistency(&l2, &[0, 1], 0),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            class_density_measure(&l2, &[0, 1], 0, None),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            class_density_measure(&l2, &[0, 1], 10, Some(-0.5)),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            class_density_measure(&l2, &[0, 1], 10, Some(f64::NAN)),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            sepme_knng(&l2, &[0, 1], 0),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            sepme_gong(&l2, &[0, 1], 1.5),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            sepme_gong(&l2, &[0, 1], f64::NAN),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            neighborhood_hit(&l2, &[0, 1], 0),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            pairwise_distance_overlap_error(&l2, &[0, 1], 0),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            sprawlter_nn(&l2, &[0.0, 1.0], &[1.0, 1.0]),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            sprawlter_nn(&bad_nan, &[1.0, 1.0], &[1.0, 1.0]),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            nor_metrics_without_graph(&l2, &l2, &[-0.1, 1.0], &[1.0, 1.0]),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            distance_consistency(&l2, &[0]),
            Err(CoreError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_klee_union_area_and_large_n_diameter_oracles() {
        let mut rng = DeterministicRng::from_seed_u64(31415);
        let mut rects = Vec::new();
        for _ in 0..35 {
            let x0 = (rng.gen_range_usize(12) as f64) * 0.5 - 2.0;
            let y0 = (rng.gen_range_usize(12) as f64) * 0.5 - 2.0;
            let w = (rng.gen_range_usize(6) + 1) as f64 * 0.5;
            let h = (rng.gen_range_usize(6) + 1) as f64 * 0.5;
            rects.push([x0, x0 + w, y0, y0 + h]);
        }

        let got_union = rect_union_area(&rects);
        let mut xs: Vec<f64> = rects.iter().flat_map(|r| [r[0], r[1]]).collect();
        let mut ys: Vec<f64> = rects.iter().flat_map(|r| [r[2], r[3]]).collect();
        xs.sort_unstable_by(f64::total_cmp);
        xs.dedup();
        ys.sort_unstable_by(f64::total_cmp);
        ys.dedup();
        let mut brute_union = 0.0_f64;
        for xi in 0..(xs.len() - 1) {
            let mx = 0.5 * (xs[xi] + xs[xi + 1]);
            let dx = xs[xi + 1] - xs[xi];
            for yi in 0..(ys.len() - 1) {
                let my = 0.5 * (ys[yi] + ys[yi + 1]);
                let dy = ys[yi + 1] - ys[yi];
                if rects
                    .iter()
                    .any(|r| mx > r[0] && mx < r[1] && my > r[2] && my < r[3])
                {
                    brute_union += dx * dy;
                }
            }
        }
        assert_close(got_union, brute_union, 1e-12, "Klee union area");

        // N > 64 rotating-calipers branch of euclidean_diameter_2d + tie-heavy cq_hac.
        let mut lxs = Vec::with_capacity(80);
        let mut lys = Vec::with_capacity(80);
        let mut llabels = Vec::with_capacity(80);
        for i in 0..80usize {
            let c = (i / 20) as u32;
            let ox = (c % 2) as f64 * 20.0;
            let oy = (c / 2) as f64 * 20.0;
            lxs.push(ox + (rng.next_f64() - 0.5) * 0.5);
            lys.push(oy + (rng.next_f64() - 0.5) * 0.5);
            llabels.push(c);
        }
        let big_layout = Layout2d::from_xy(lxs, lys).unwrap();
        let got_diam = euclidean_diameter_2d(&big_layout);
        let mut max_sq = 0.0_f64;
        for i in 0..80 {
            for j in (i + 1)..80 {
                let dx = big_layout.x()[i] - big_layout.x()[j];
                let dy = big_layout.y()[i] - big_layout.y()[j];
                max_sq = max_sq.max(dx * dx + dy * dy);
            }
        }
        assert_close(got_diam, max_sq.sqrt(), 1e-10, "large N rotating calipers");
        assert_close(
            cq_hac(&big_layout, &llabels).unwrap().ari,
            1.0,
            1e-10,
            "large N separated cq_hac",
        );
        assert_eq!(
            pairwise_distance_overlap_error(&big_layout, &llabels, 50).unwrap(),
            0.0
        );
    }

    #[test]
    fn test_partition_agreement_dominant_cluster_no_underflow() {
        // a_0 = 3, b_0 = 2, n = 4 -> a_0 + b_0 = 5 > n (must not underflow in debug builds).
        let small = partition_agreement(&[0, 0, 0, 1], &[0, 0, 1, 1]).unwrap();
        assert_close(small.ari, 0.0, 1e-12, "skewed4.ari");
        assert!(
            small.ami.abs() < 1e-12,
            "skewed4.ami = {} (expected ~0.0)",
            small.ami
        );

        // Larger skewed partition: n = 16, a_0 = 11, b_0 = 10 -> a_0 + b_0 = 21 > 16.
        let lt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2];
        let lp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2];
        let skewed = partition_agreement(&lt, &lp).unwrap();
        assert_close(skewed.ari, 0.782_850_779_510_022_2, 1e-10, "skewed16.ari");
        assert_close(skewed.ami, 0.754_922_259_280_672, 1e-10, "skewed16.ami");
        assert_close(
            skewed.homogeneity,
            0.830_907_673_853_462_1,
            1e-10,
            "skewed16.hom",
        );
        assert_close(
            skewed.completeness,
            0.767_358_057_041_120_3,
            1e-10,
            "skewed16.cmp",
        );
        assert_close(
            skewed.v_measure,
            0.797_869_447_944_516_2,
            1e-10,
            "skewed16.vm",
        );
        assert_close(skewed.fmi, 0.884_643_727_225_795_5, 1e-10, "skewed16.fmi");
    }

    #[test]
    fn test_cq_hac_ulp_height_inversion_regression() {
        // Challenger N=5, K=3 counterexample where UPGMA (2*h + 1*h)/3 rounds 1 ULP below h.
        let xs64 = vec![0.0, 0.0, 1.0, 0.5, 5.550275137568784];
        let ys64 = vec![0.0, 0.0, 0.0, 3.0_f64.sqrt() / 2.0, 0.0];
        let labels = [0_u32, 0, 0, 1, 2];

        let layout64 = Layout2d::from_xy(xs64.clone(), ys64.clone()).unwrap();
        let res64 = cq_hac(&layout64, &labels).unwrap();
        assert!(
            (res64.ari - 0.4117647058823529).abs() > 1e-6,
            "f64 cq_hac produced 4-cluster inverted cut (ari = {})",
            res64.ari
        );
        assert!(
            (res64.ari - 1.0).abs() < 1e-10 || (res64.ari - 6.0 / 11.0).abs() < 1e-10,
            "f64 cq_hac unexpected 3-cluster ari = {}",
            res64.ari
        );

        let xs32: Vec<f32> = xs64.iter().map(|&v| v as f32).collect();
        let ys32: Vec<f32> = ys64.iter().map(|&v| v as f32).collect();
        let layout32 = Layout2d::from_xy(xs32, ys32).unwrap();
        let res32 = cq_hac(&layout32, &labels).unwrap();
        // In f32, (sqrt(3)/2) as f32 rounds slightly down, so d(0, 3) < d(0, 2), yielding the
        // valid 3-cluster UPGMA cut [{0, 1, 3}, {2}, {4}] with ARI = 1/21 (never 4 clusters).
        assert!(
            (res32.ari - 0.411_764_705_882_352_9).abs() > 1e-6,
            "f32 cq_hac produced 4-cluster inverted cut (ari = {})",
            res32.ari
        );
        assert_close(res32.ari, 1.0 / 21.0, 1e-10, "f32 cq_hac 3-cluster ari");
    }

    #[test]
    fn test_pdoe_subnormal_scale_no_overflow() {
        // Subnormal coordinates (~1e-307) with coincident intra pairs so d = 0.0 is also exercised.
        let sep_subnormal = layout_from_pairs(&[
            (0.0, 0.0),
            (0.0, 0.0),
            (1e-308, 0.0),
            (1e-307, 0.0),
            (1e-307, 0.0),
            (1.01e-307, 0.0),
        ]);
        let labels = [0_u32, 0, 0, 1, 1, 1];
        let pdoe_sep = pairwise_distance_overlap_error(&sep_subnormal, &labels, 1000).unwrap();
        assert!(pdoe_sep.is_finite());
        assert_close(pdoe_sep, 0.0, 1e-12, "subnormal separated pdoe");

        // Intra pairs are both at d = 1e-307 (hist_intra[last] = 1.0), while inter pairs split
        // evenly between d = 0.0 and d = 1e-307 (hist_inter[0] = 0.5, hist_inter[last] = 0.5).
        let mix_subnormal =
            layout_from_pairs(&[(0.0, 0.0), (1e-307, 0.0), (0.0, 0.0), (1e-307, 0.0)]);
        let pdoe_mix =
            pairwise_distance_overlap_error(&mix_subnormal, &[0, 0, 1, 1], 1000).unwrap();
        assert!(pdoe_mix.is_finite());
        assert_close(pdoe_mix, 0.5, 1e-12, "subnormal overlapping pdoe");

        // N > 64 rotating-calipers branch at subnormal scale.
        let mut xs = Vec::with_capacity(70);
        let mut ys = Vec::with_capacity(70);
        let mut big_labels = Vec::with_capacity(70);
        for i in 0..70usize {
            let c = (i / 35) as u32;
            xs.push(if c == 0 {
                (i as f64) * 1e-309
            } else {
                1e-307 + ((i - 35) as f64) * 1e-309
            });
            ys.push(((i % 5) as f64) * 1e-309);
            big_labels.push(c);
        }
        let big_subnormal = Layout2d::from_xy(xs, ys).unwrap();
        let pdoe_big = pairwise_distance_overlap_error(&big_subnormal, &big_labels, 100).unwrap();
        assert!(pdoe_big.is_finite());
        assert_close(pdoe_big, 0.0, 1e-12, "subnormal N>64 separated pdoe");
    }

    #[test]
    fn test_nor_metrics_zero_extent() {
        let orig = layout_from_pairs(&[(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (2.0, 2.0)]);
        let adj = layout_from_pairs(&[(0.0, 0.0), (4.0, 0.0), (0.0, 4.0), (4.0, 4.0)]);
        let zero_hw = [0.0_f64; 4];
        let zero_hh = [0.0_f64; 4];
        let g =
            CsrGraph::<f64>::from_edges(4, &[(0, 1), (0, 2), (1, 3)], None, false, false).unwrap();

        let m = nor_metrics(&g, &orig, &adj, &zero_hw, &zero_hh).unwrap();
        assert_close(m.oo_nni, 0.0, 1e-12, "zero_ext.oo_nni");
        assert_close(m.sp_ch_a, 4.0, 1e-12, "zero_ext.sp_ch_a");
        assert_close(m.gs_bb_iar, 1.0, 1e-12, "zero_ext.gs_bb_iar");
        assert_close(m.nm_dm_imse, 0.0, 1e-12, "zero_ext.nm_dm_imse");
        assert_close(m.el_rsd, 0.0, 1e-12, "zero_ext.el_rsd");

        let m_ng = nor_metrics_without_graph(&orig, &adj, &zero_hw, &zero_hh).unwrap();
        assert_close(m_ng.sp_ch_a, 4.0, 1e-12, "zero_ext_ng.sp_ch_a");
        assert_eq!(m_ng.el_rsd, 0.0);

        // Collinear zero-extent layout falls back cleanly to 1.0 for sp_ch_a and gs_bb_iar.
        let col = layout_from_pairs(&[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]);
        let m_col = nor_metrics_without_graph(&col, &col, &zero_hw, &zero_hh).unwrap();
        assert_close(m_col.oo_nni, 0.0, 1e-12, "zero_ext_col.oo_nni");
        assert_close(m_col.sp_ch_a, 1.0, 1e-12, "zero_ext_col.sp_ch_a");
        assert_close(m_col.gs_bb_iar, 1.0, 1e-12, "zero_ext_col.gs_bb_iar");
        assert_close(m_col.nm_dm_imse, 0.0, 1e-12, "zero_ext_col.nm_dm_imse");
    }
}
