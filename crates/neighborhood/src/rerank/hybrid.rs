//! Hybrid 1+2-hop + 2-stage spectral rerank $k$-NN builder with strict margin separation.

use egraph_core::{CoreError, CoreResult, CsrGraph, VisitStamp};

use crate::knn::{KnnGraph, KnnStats, extract_1hop_2hop_csr};

use super::kdtree_rerank::{
    CoarseKdTree, euclidean_dist, select_top_k_pairs, validate_and_center_coords,
};

/// Configuration for [`build_hybrid_adaptive_knn`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HybridKnnConfig {
    /// Target budget multiplier `round(k_factor * perplexity[i])` when `k_override` is `None` (`default: 3.0`).
    pub k_factor: f64,
    /// Optional explicit uniform target neighbor budget overriding `k_factor * perplexity[i]`.
    pub k_override: Option<usize>,
    /// Relative distance gap enforced between 1+2-hop neighbors and non-2-hop residual fills (`default: 0.1`).
    pub margin_gap: f64,
    /// Coarse KD-tree dimension in `1..=12` for Stage 1 candidate filtering (`default: 10`).
    pub coarse_dim: usize,
    /// Oversampling multiplier for Stage 2 full-dimensional reranking (`default: 16`).
    pub rerank_factor: usize,
    /// Optional hard cap per row (`None` preserves every 1-hop and 2-hop neighbor without truncation).
    pub max_row_k: Option<usize>,
}

impl Default for HybridKnnConfig {
    fn default() -> Self {
        Self {
            k_factor: 3.0,
            k_override: None,
            margin_gap: 0.1,
            coarse_dim: 10,
            rerank_factor: 16,
            max_row_k: None,
        }
    }
}

#[inline]
fn apply_margin_gap(d_raw: &mut [f64], d_2hop_max: f64, margin_gap: f64) {
    if margin_gap <= 0.0 || d_raw.is_empty() {
        return;
    }
    let floor = if d_2hop_max > 0.0 {
        ((1.0 + margin_gap) * d_2hop_max).max(d_2hop_max.next_up())
    } else {
        let min_pos = d_raw
            .iter()
            .copied()
            .filter(|&d| d > 0.0)
            .fold(f64::INFINITY, f64::min);
        if min_pos.is_finite() {
            min_pos.max(1e-9)
        } else {
            1e-6 * (1.0 + margin_gap)
        }
    };
    for d in d_raw {
        if *d < floor {
            *d = floor;
        }
    }
}

/// Constructs a variable-length CSR hybrid 1+2-hop + 2-stage spectral rerank [`KnnGraph`].
///
/// Preserves every 1-hop and pure 2-hop neighbor without truncation unless `config.max_row_k` is
/// set, fills residual slots up to `max(kept_12, min_k_i)` via 2-stage coarse
/// [`egraph_spatial::StaticKdTree`] + full-$D$ reranking, and enforces
/// $\max(d_{1,2\text{-hop}}) < \min(d_{\text{non-2}})$ when `config.margin_gap > 0.0`.
pub fn build_hybrid_adaptive_knn<W>(
    graph: &CsrGraph<W>,
    coords_row_major: &[f64],
    dim: usize,
    perplexities: &[f64],
    config: &HybridKnnConfig,
) -> CoreResult<(KnnGraph, KnnStats)> {
    let n = graph.num_nodes();
    let centered = validate_and_center_coords(
        coords_row_major,
        n,
        dim,
        "build_hybrid_adaptive_knn::coords",
    )?;
    if perplexities.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            perplexities.len(),
            "build_hybrid_adaptive_knn::perplexities",
        ));
    }
    for &p in perplexities {
        if !p.is_finite() {
            return Err(CoreError::non_finite(
                "build_hybrid_adaptive_knn::perplexities",
            ));
        }
        if p <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "perplexities must be > 0, got {p}"
            )));
        }
    }
    if !config.k_factor.is_finite() {
        return Err(CoreError::non_finite("HybridKnnConfig::k_factor"));
    }
    if config.k_factor <= 0.0 {
        return Err(CoreError::invalid_input(format!(
            "k_factor must be > 0, got {}",
            config.k_factor
        )));
    }
    if let Some(k_ov) = config.k_override
        && k_ov == 0
    {
        return Err(CoreError::invalid_input("k_override must be >= 1"));
    }
    if !config.margin_gap.is_finite() {
        return Err(CoreError::non_finite("HybridKnnConfig::margin_gap"));
    }
    if config.margin_gap < 0.0 {
        return Err(CoreError::invalid_input(format!(
            "margin_gap must be >= 0, got {}",
            config.margin_gap
        )));
    }
    if config.coarse_dim == 0 {
        return Err(CoreError::invalid_input("coarse_dim must be >= 1"));
    }
    if config.rerank_factor == 0 {
        return Err(CoreError::invalid_input("rerank_factor must be >= 1"));
    }
    if let Some(cap) = config.max_row_k
        && cap == 0
    {
        return Err(CoreError::invalid_input("max_row_k must be >= 1"));
    }

    if n <= 1 {
        let knn = KnnGraph::empty(n)?;
        let stats = KnnStats {
            n_pure_2hop_rows: n,
            ..KnnStats::default()
        };
        return Ok((knn, stats));
    }

    let cap = config.max_row_k.map(|c| c.min(n - 1));
    let min_k_arr: Vec<usize> = perplexities
        .iter()
        .map(|&p| {
            let base = match config.k_override {
                Some(k_ov) => k_ov.min(n - 1),
                None => (config.k_factor * p)
                    .max(1.0)
                    .min((n - 1) as f64)
                    .round_ties_even() as usize,
            };
            match cap {
                Some(c) => base.min(c),
                None => base,
            }
        })
        .collect();

    let (adj1, adj2) = extract_1hop_2hop_csr(graph)?;
    let n_1hop = adj1.nnz();
    let n_2hop = adj2.nnz();

    let mut row_12_entries: Vec<Vec<(u32, f64)>> = Vec::with_capacity(n);
    let mut d2_max_arr = vec![0.0_f64; n];
    let mut total_deg1 = 0usize;
    let mut total_deg2 = 0usize;
    let mut n_truncated_1hop = 0usize;
    let mut n_truncated_2hop = 0usize;
    let mut need_fill: Vec<usize> = Vec::new();
    let mut any_kdtree_row = false;

    for i in 0..n {
        let n1 = adj1.neighbors(i);
        let n2 = adj2.neighbors(i);
        let deg1 = n1.len();
        let deg2_total = deg1 + n2.len();
        total_deg1 += deg1;
        total_deg2 += deg2_total;

        let mut tier1: Vec<(f64, u32)> = Vec::with_capacity(deg1);
        for &v in n1 {
            let d = if dim > 0 {
                euclidean_dist(
                    &centered[i * dim..(i + 1) * dim],
                    &centered[(v as usize) * dim..((v as usize) + 1) * dim],
                )
            } else {
                0.0
            };
            tier1.push((d, v));
        }
        tier1.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        if let Some(c) = cap
            && deg1 >= c
        {
            if deg1 > c {
                n_truncated_1hop += 1;
            }
            if deg2_total > c {
                n_truncated_2hop += 1;
            }
            let mut row = Vec::with_capacity(c);
            for &(d, v) in tier1.iter().take(c) {
                row.push((v, d));
            }
            if row.len() < min_k_arr[i] {
                d2_max_arr[i] = row.iter().map(|&(_, d)| d).fold(0.0_f64, f64::max);
                need_fill.push(i);
                let cand_k = config
                    .rerank_factor
                    .saturating_mul(min_k_arr[i])
                    .max(min_k_arr[i])
                    .min(n - 1);
                if dim > 0 && cand_k < n - 1 {
                    any_kdtree_row = true;
                }
            }
            row_12_entries.push(row);
            continue;
        }

        let mut tier2: Vec<(f64, u32)> = Vec::with_capacity(n2.len());
        for &v in n2 {
            let d = if dim > 0 {
                euclidean_dist(
                    &centered[i * dim..(i + 1) * dim],
                    &centered[(v as usize) * dim..((v as usize) + 1) * dim],
                )
            } else {
                0.0
            };
            tier2.push((d, v));
        }
        tier2.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        let keep2 = match cap {
            Some(c) if deg2_total > c => {
                n_truncated_2hop += 1;
                c - deg1
            }
            _ => tier2.len(),
        };

        let mut row = Vec::with_capacity(deg1 + keep2);
        for &(d, v) in &tier1 {
            row.push((v, d));
        }
        for &(d, v) in tier2.iter().take(keep2) {
            row.push((v, d));
        }
        if row.len() < min_k_arr[i] {
            d2_max_arr[i] = row.iter().map(|&(_, d)| d).fold(0.0_f64, f64::max);
            need_fill.push(i);
            let cand_k = config
                .rerank_factor
                .saturating_mul(min_k_arr[i])
                .max(min_k_arr[i])
                .min(n - 1);
            if dim > 0 && cand_k < n - 1 {
                any_kdtree_row = true;
            }
        }
        row_12_entries.push(row);
    }

    if !need_fill.is_empty() {
        let tree_opt = if any_kdtree_row {
            Some(CoarseKdTree::build(&centered, n, dim, config.coarse_dim)?)
        } else {
            None
        };
        let mut stamp = VisitStamp::new(n);
        let mut cand_buf: Vec<(usize, f64)> = Vec::new();
        let mut scored: Vec<(f64, u32)> = Vec::new();
        let mut res_dists: Vec<f64> = Vec::new();

        for &i in &need_fill {
            let kept_i = row_12_entries[i].len();
            let target_k = min_k_arr[i];
            if kept_i >= target_k {
                continue;
            }
            let rem = target_k - kept_i;

            stamp.clear();
            stamp.mark(i);
            for &(v, _) in &row_12_entries[i] {
                stamp.mark(v as usize);
            }

            if dim == 0 {
                let mut added = 0usize;
                res_dists.clear();
                res_dists.resize(rem, 0.0);
                if kept_i > 0 {
                    apply_margin_gap(&mut res_dists, d2_max_arr[i], config.margin_gap);
                }
                for j in 0..n {
                    if stamp.visit(j) {
                        row_12_entries[i].push((j as u32, res_dists[added]));
                        added += 1;
                        if added == rem {
                            break;
                        }
                    }
                }
                continue;
            }

            let cand_k = config
                .rerank_factor
                .saturating_mul(target_k)
                .max(target_k)
                .min(n - 1);
            let row_i = &centered[i * dim..(i + 1) * dim];
            scored.clear();

            if let Some(tree) = &tree_opt
                && cand_k < n - 1
            {
                tree.k_nearest_excluding_self_into(i, cand_k, &mut cand_buf)?;
                for &(j, _) in &cand_buf {
                    if stamp.visit(j) {
                        let row_j = &centered[j * dim..(j + 1) * dim];
                        scored.push((euclidean_dist(row_i, row_j), j as u32));
                    }
                }
                if scored.len() < rem {
                    for j in 0..n {
                        if stamp.visit(j) {
                            let row_j = &centered[j * dim..(j + 1) * dim];
                            scored.push((euclidean_dist(row_i, row_j), j as u32));
                        }
                    }
                }
            } else {
                for j in 0..n {
                    if stamp.visit(j) {
                        let row_j = &centered[j * dim..(j + 1) * dim];
                        scored.push((euclidean_dist(row_i, row_j), j as u32));
                    }
                }
            }

            select_top_k_pairs(&mut scored, rem);
            res_dists.clear();
            res_dists.extend(scored.iter().map(|&(d, _)| d));
            if kept_i > 0 {
                apply_margin_gap(&mut res_dists, d2_max_arr[i], config.margin_gap);
            }
            for ((_, idx), d_adj) in scored.iter().zip(res_dists.iter()) {
                row_12_entries[i].push((*idx, *d_adj));
            }
        }
    }

    let total_nnz: usize = row_12_entries.iter().map(Vec::len).sum();
    if u32::try_from(total_nnz).is_err() {
        return Err(CoreError::invalid_csr(
            "hybrid adaptive knn nnz exceeds u32::MAX",
        ));
    }
    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0u32);
    let mut indices = Vec::with_capacity(total_nnz);
    let mut distances = Vec::with_capacity(total_nnz);
    let mut min_row_k = usize::MAX;
    let mut max_row_k = 0usize;

    for row in &row_12_entries {
        let rk = row.len();
        min_row_k = min_row_k.min(rk);
        max_row_k = max_row_k.max(rk);
        for &(v, d) in row {
            indices.push(v);
            distances.push(d);
        }
        let ptr = u32::try_from(indices.len())
            .map_err(|_| CoreError::invalid_csr("hybrid adaptive knn nnz exceeds u32::MAX"))?;
        indptr.push(ptr);
    }

    let knn = KnnGraph::new_validated(n, indptr, indices, distances)?;
    let n_f64 = n as f64;
    let n_filled = need_fill.len();
    let stats = KnnStats {
        avg_deg1: (total_deg1 as f64) / n_f64,
        avg_deg2: (total_deg2 as f64) / n_f64,
        avg_row_k: (total_nnz as f64) / n_f64,
        min_row_k,
        max_row_k,
        nnz: total_nnz,
        n_1hop,
        n_2hop,
        n_truncated_1hop,
        n_truncated_2hop,
        n_filled_residual: n_filled,
        n_pure_2hop_rows: n - n_filled,
    };

    Ok((knn, stats))
}
