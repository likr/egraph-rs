//! Spectral resistance/diffusion embeddings, 2-stage coarse `StaticKdTree` + full-$D$ Euclidean
//! reranking $k$-NN, and hybrid 1+2-hop + adaptive residual $k$-NN construction.

use std::cmp::Ordering;

use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar, VisitStamp};
use egraph_linalg::smallest_sym_laplacian_eigenpairs;
use egraph_spatial::StaticKdTree;

use crate::knn::{KnnGraph, KnnStats, extract_1hop_2hop_csr};

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

/// Computes effective-resistance spectral embedding $X_{\mathrm{res}} = D^{-1/2} U \Lambda^{-1/2}$
/// from the smallest nontrivial eigenpairs of $L_{\mathrm{sym}}$, filtering eigenvalues
/// $\lambda \le \mathrm{eig\_eps}$.
///
/// Returns `(coords_row_major, lambdas_kept)` where `coords_row_major` has shape
/// `(graph.num_nodes(), lambdas_kept.len())`.
pub fn compute_resistance_embedding<W: Scalar>(
    graph: &CsrGraph<W>,
    dim: usize,
    tol: f64,
    max_iters: usize,
    seed: u64,
    eig_eps: f64,
) -> CoreResult<(Vec<f64>, Vec<f64>)> {
    if dim == 0 {
        return Err(CoreError::invalid_input("dim must be >= 1"));
    }
    if !eig_eps.is_finite() {
        return Err(CoreError::non_finite(
            "compute_resistance_embedding::eig_eps",
        ));
    }
    if eig_eps < 0.0 {
        return Err(CoreError::invalid_input(format!(
            "eig_eps must be >= 0, got {eig_eps}"
        )));
    }

    let n = graph.num_nodes();
    if n <= 1 {
        return Ok((Vec::new(), Vec::new()));
    }

    let eig = smallest_sym_laplacian_eigenpairs(graph, dim, tol, max_iters, seed)?;
    let kept_modes: Vec<usize> = eig
        .lambdas
        .iter()
        .enumerate()
        .filter_map(|(m, &lam)| (lam > eig_eps).then_some(m))
        .collect();
    let k_kept = kept_modes.len();
    if k_kept == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let lambdas_kept: Vec<f64> = kept_modes.iter().map(|&m| eig.lambdas[m]).collect();
    let inv_sqrt_lam: Vec<f64> = lambdas_kept.iter().map(|&lam| lam.sqrt().recip()).collect();

    let mut coords = vec![0.0_f64; n * k_kept];
    for i in 0..n {
        let d_inv = eig.inv_sqrt_deg[i];
        let out_row = &mut coords[i * k_kept..(i + 1) * k_kept];
        for (col, &m) in kept_modes.iter().enumerate() {
            out_row[col] = d_inv * eig.vectors[i * eig.num_modes + m] * inv_sqrt_lam[col];
        }
    }
    Ok((coords, lambdas_kept))
}

/// Computes heat-diffusion spectral embedding $X_{\mathrm{diff}} = D^{-1/2} U \exp(-0.5 t \Lambda)$
/// from the smallest nontrivial eigenpairs of $L_{\mathrm{sym}}$.
///
/// When `t` is `None`, automatically sets $t = -\ln(p) / \lambda_k$ (falling back to $-\ln(p)$
/// when $\lambda_k \le 10^{-12}$). Returns `(coords_row_major, lambdas, t_used)`.
pub fn compute_diffusion_embedding<W: Scalar>(
    graph: &CsrGraph<W>,
    dim: usize,
    p: f64,
    t: Option<f64>,
    tol: f64,
    max_iters: usize,
    seed: u64,
) -> CoreResult<(Vec<f64>, Vec<f64>, f64)> {
    if dim == 0 {
        return Err(CoreError::invalid_input("dim must be >= 1"));
    }
    if !p.is_finite() {
        return Err(CoreError::non_finite("compute_diffusion_embedding::p"));
    }
    if !(0.0 < p && p < 1.0) {
        return Err(CoreError::invalid_input(format!(
            "diffusion parameter p must satisfy 0 < p < 1, got {p}"
        )));
    }
    if let Some(t_val) = t {
        if !t_val.is_finite() {
            return Err(CoreError::non_finite("compute_diffusion_embedding::t"));
        }
        if t_val < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "diffusion time t must be >= 0, got {t_val}"
            )));
        }
    }

    let n = graph.num_nodes();
    if n <= 1 {
        let t_used = t.unwrap_or_else(|| -p.ln());
        return Ok((Vec::new(), Vec::new(), t_used));
    }

    let eig = smallest_sym_laplacian_eigenpairs(graph, dim, tol, max_iters, seed)?;
    let lambdas: Vec<f64> = eig.lambdas.iter().map(|&lam| lam.max(0.0)).collect();
    let t_used = match t {
        Some(tv) => tv,
        None => match lambdas.last() {
            Some(&lk) if lk > 1e-12 => -p.ln() / lk,
            _ => -p.ln(),
        },
    };
    let weights: Vec<f64> = lambdas
        .iter()
        .map(|&lam| (-0.5 * t_used * lam).exp())
        .collect();

    let k_modes = eig.num_modes;
    let mut coords = vec![0.0_f64; n * k_modes];
    for i in 0..n {
        let d_inv = eig.inv_sqrt_deg[i];
        let out_row = &mut coords[i * k_modes..(i + 1) * k_modes];
        for m in 0..k_modes {
            out_row[m] = d_inv * eig.vectors[i * k_modes + m] * weights[m];
        }
    }
    Ok((coords, lambdas, t_used))
}

#[inline]
fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    let mut sum_sq = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let diff = x - y;
        sum_sq += diff * diff;
    }
    sum_sq.sqrt()
}

fn validate_and_center_coords(
    coords_row_major: &[f64],
    n_nodes: usize,
    dim: usize,
    context: &'static str,
) -> CoreResult<Vec<f64>> {
    if u32::try_from(n_nodes).is_err() {
        return Err(CoreError::invalid_csr("n_nodes exceeds u32::MAX"));
    }
    let expected = n_nodes
        .checked_mul(dim)
        .ok_or_else(|| CoreError::invalid_input("coordinate buffer size overflow"))?;
    if coords_row_major.len() != expected {
        return Err(CoreError::dimension_mismatch(
            expected,
            coords_row_major.len(),
            context,
        ));
    }
    if coords_row_major.iter().any(|v| !v.is_finite()) {
        return Err(CoreError::non_finite(context));
    }
    if n_nodes == 0 || dim == 0 {
        return Ok(Vec::new());
    }

    let mut means = vec![0.0_f64; dim];
    for i in 0..n_nodes {
        let row = &coords_row_major[i * dim..(i + 1) * dim];
        for d in 0..dim {
            means[d] += row[d];
        }
    }
    let inv_n = 1.0 / (n_nodes as f64);
    for m in &mut means {
        *m *= inv_n;
    }

    let mut centered = vec![0.0_f64; expected];
    for i in 0..n_nodes {
        let src = &coords_row_major[i * dim..(i + 1) * dim];
        let dst = &mut centered[i * dim..(i + 1) * dim];
        for d in 0..dim {
            dst[d] = src[d] - means[d];
        }
    }
    Ok(centered)
}

#[inline]
fn select_top_k_pairs(scored: &mut Vec<(f64, u32)>, k: usize) {
    let cmp = |a: &(f64, u32), b: &(f64, u32)| -> Ordering {
        a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1))
    };
    if scored.len() > k && k > 0 {
        scored.select_nth_unstable_by(k - 1, cmp);
        scored.truncate(k);
    }
    scored.sort_unstable_by(cmp);
}

enum CoarseKdTree {
    D1(StaticKdTree<1, f64>),
    D2(StaticKdTree<2, f64>),
    D3(StaticKdTree<3, f64>),
    D4(StaticKdTree<4, f64>),
    D5(StaticKdTree<5, f64>),
    D6(StaticKdTree<6, f64>),
    D7(StaticKdTree<7, f64>),
    D8(StaticKdTree<8, f64>),
    D9(StaticKdTree<9, f64>),
    D10(StaticKdTree<10, f64>),
    D11(StaticKdTree<11, f64>),
    D12(StaticKdTree<12, f64>),
}

impl CoarseKdTree {
    fn build(centered: &[f64], n_nodes: usize, dim: usize, coarse_dim: usize) -> CoreResult<Self> {
        let cd = coarse_dim.min(dim).clamp(1, 12);
        match cd {
            1 => Ok(Self::D1(build_typed_tree::<1>(centered, n_nodes, dim)?)),
            2 => Ok(Self::D2(build_typed_tree::<2>(centered, n_nodes, dim)?)),
            3 => Ok(Self::D3(build_typed_tree::<3>(centered, n_nodes, dim)?)),
            4 => Ok(Self::D4(build_typed_tree::<4>(centered, n_nodes, dim)?)),
            5 => Ok(Self::D5(build_typed_tree::<5>(centered, n_nodes, dim)?)),
            6 => Ok(Self::D6(build_typed_tree::<6>(centered, n_nodes, dim)?)),
            7 => Ok(Self::D7(build_typed_tree::<7>(centered, n_nodes, dim)?)),
            8 => Ok(Self::D8(build_typed_tree::<8>(centered, n_nodes, dim)?)),
            9 => Ok(Self::D9(build_typed_tree::<9>(centered, n_nodes, dim)?)),
            10 => Ok(Self::D10(build_typed_tree::<10>(centered, n_nodes, dim)?)),
            11 => Ok(Self::D11(build_typed_tree::<11>(centered, n_nodes, dim)?)),
            _ => Ok(Self::D12(build_typed_tree::<12>(centered, n_nodes, dim)?)),
        }
    }

    fn k_nearest_excluding_self_into(
        &self,
        point_idx: usize,
        k: usize,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        match self {
            Self::D1(t) => t.k_nearest_excluding_self_into(point_idx, k, out),
            Self::D2(t) => t.k_nearest_excluding_self_into(point_idx, k, out),
            Self::D3(t) => t.k_nearest_excluding_self_into(point_idx, k, out),
            Self::D4(t) => t.k_nearest_excluding_self_into(point_idx, k, out),
            Self::D5(t) => t.k_nearest_excluding_self_into(point_idx, k, out),
            Self::D6(t) => t.k_nearest_excluding_self_into(point_idx, k, out),
            Self::D7(t) => t.k_nearest_excluding_self_into(point_idx, k, out),
            Self::D8(t) => t.k_nearest_excluding_self_into(point_idx, k, out),
            Self::D9(t) => t.k_nearest_excluding_self_into(point_idx, k, out),
            Self::D10(t) => t.k_nearest_excluding_self_into(point_idx, k, out),
            Self::D11(t) => t.k_nearest_excluding_self_into(point_idx, k, out),
            Self::D12(t) => t.k_nearest_excluding_self_into(point_idx, k, out),
        }
    }
}

fn build_typed_tree<const D: usize>(
    centered: &[f64],
    n_nodes: usize,
    dim: usize,
) -> CoreResult<StaticKdTree<D, f64>> {
    let mut pts = Vec::with_capacity(n_nodes);
    for i in 0..n_nodes {
        let mut p = [0.0_f64; D];
        let row = &centered[i * dim..(i + 1) * dim];
        p.copy_from_slice(&row[..D]);
        pts.push(p);
    }
    StaticKdTree::<D, f64>::build(&pts)
}

/// Constructs an exact/2-stage reranked $k$-NN [`KnnGraph`] from row-major `(n_nodes, dim)`
/// coordinates using a coarse [`StaticKdTree`] over `1..=12` dimensions followed by exact
/// full-`dim` Euclidean reranking.
///
/// Automatically falls back to deterministic exact full-`dim` search when `cand_k >= n_nodes - 1`.
pub fn build_rerank_knn(
    coords_row_major: &[f64],
    n_nodes: usize,
    dim: usize,
    k: usize,
    coarse_dim: usize,
    rerank_factor: usize,
) -> CoreResult<KnnGraph> {
    if coarse_dim == 0 {
        return Err(CoreError::invalid_input("coarse_dim must be >= 1"));
    }
    if rerank_factor == 0 {
        return Err(CoreError::invalid_input("rerank_factor must be >= 1"));
    }
    let centered = validate_and_center_coords(coords_row_major, n_nodes, dim, "build_rerank_knn")?;
    if n_nodes <= 1 || k == 0 {
        return KnnGraph::empty(n_nodes);
    }

    let k_eff = k.min(n_nodes - 1);
    let mut indptr = Vec::with_capacity(n_nodes + 1);
    indptr.push(0usize);
    let mut indices = Vec::with_capacity(n_nodes * k_eff);
    let mut distances = Vec::with_capacity(n_nodes * k_eff);

    if dim == 0 {
        for i in 0..n_nodes {
            let mut added = 0usize;
            for j in 0..n_nodes {
                if j == i {
                    continue;
                }
                indices.push(j as u32);
                distances.push(0.0);
                added += 1;
                if added == k_eff {
                    break;
                }
            }
            indptr.push(indices.len());
        }
        return KnnGraph::new_validated(n_nodes, indptr, indices, distances);
    }

    let cand_k = rerank_factor
        .saturating_mul(k_eff)
        .max(k_eff)
        .min(n_nodes - 1);
    let mut scored: Vec<(f64, u32)> = Vec::with_capacity(cand_k);

    if cand_k >= n_nodes - 1 {
        for i in 0..n_nodes {
            scored.clear();
            let row_i = &centered[i * dim..(i + 1) * dim];
            for j in 0..n_nodes {
                if j == i {
                    continue;
                }
                let row_j = &centered[j * dim..(j + 1) * dim];
                scored.push((euclidean_dist(row_i, row_j), j as u32));
            }
            select_top_k_pairs(&mut scored, k_eff);
            for &(dist, idx) in &scored {
                indices.push(idx);
                distances.push(dist);
            }
            indptr.push(indices.len());
        }
    } else {
        let tree = CoarseKdTree::build(&centered, n_nodes, dim, coarse_dim)?;
        let mut cand_buf: Vec<(usize, f64)> = Vec::with_capacity(cand_k);
        for i in 0..n_nodes {
            tree.k_nearest_excluding_self_into(i, cand_k, &mut cand_buf)?;
            scored.clear();
            let row_i = &centered[i * dim..(i + 1) * dim];
            for &(j, _) in &cand_buf {
                let row_j = &centered[j * dim..(j + 1) * dim];
                scored.push((euclidean_dist(row_i, row_j), j as u32));
            }
            select_top_k_pairs(&mut scored, k_eff);
            for &(dist, idx) in &scored {
                indices.push(idx);
                distances.push(dist);
            }
            indptr.push(indices.len());
        }
    }

    KnnGraph::new_validated(n_nodes, indptr, indices, distances)
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
/// set, fills residual slots up to `max(kept_12, min_k_i)` via 2-stage coarse `StaticKdTree`
/// + full-$D$ reranking, and enforces $\max(d_{1,2\text{-hop}}) < \min(d_{\text{non-2}})$ when
///   `config.margin_gap > 0.0`.
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
    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0usize);
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
        indptr.push(indices.len());
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

#[cfg(test)]
mod tests {
    use super::*;
    use egraph_core::DeterministicRng;

    #[test]
    fn test_resistance_and_diffusion_embeddings() {
        let edges: Vec<(u32, u32)> = (0..20).map(|i| (i, (i + 1) % 20)).collect();
        let cycle = CsrGraph::<f64>::from_edges(20, &edges, None, false, false).unwrap();

        let (res_coords, res_lams) =
            compute_resistance_embedding(&cycle, 8, 1e-6, 300, 42, 1e-10).unwrap();
        assert_eq!(res_lams.len(), 8);
        assert_eq!(res_coords.len(), 20 * 8);
        assert!(res_coords.iter().all(|v| v.is_finite()));

        let (diff_coords, diff_lams, t_used) =
            compute_diffusion_embedding(&cycle, 8, 0.1, None, 1e-6, 300, 42).unwrap();
        assert_eq!(diff_lams.len(), 8);
        assert_eq!(diff_coords.len(), 20 * 8);
        assert!(t_used > 0.0 && t_used.is_finite());
        assert!(diff_coords.iter().all(|v| v.is_finite()));

        // Disconnected graph: zero eigenvalue mode filtered cleanly in resistance embedding.
        let disc = CsrGraph::<f64>::from_edges(
            6,
            &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)],
            None,
            false,
            false,
        )
        .unwrap();
        let (rc_disc, rl_disc) =
            compute_resistance_embedding(&disc, 4, 1e-6, 200, 42, 1e-10).unwrap();
        assert_eq!(rl_disc.len(), 3);
        assert_eq!(rc_disc.len(), 6 * 3);

        // Edgeless graph -> zero nonzero eigenvalues -> empty coords.
        let iso = CsrGraph::<f64>::from_edges(5, &[], None, false, false).unwrap();
        let (rc_iso, rl_iso) = compute_resistance_embedding(&iso, 3, 1e-6, 100, 42, 1.5).unwrap();
        assert!(rl_iso.is_empty());
        assert!(rc_iso.is_empty());
    }

    #[test]
    fn test_rerank_knn_exact_and_two_stage_kdtree() {
        let n = 150;
        let dim = 14;
        let mut rng = DeterministicRng::from_seed_u64(123);
        let mut coords = Vec::with_capacity(n * dim);
        for _ in 0..n {
            for d in 0..dim {
                // First few dimensions dominate slightly so coarse_dim=10 recovers exact k-NN well.
                let scale = if d < 10 { 2.0 } else { 0.05 };
                coords.push((rng.next_f64() - 0.5) * scale);
            }
        }

        // Exact fallback when cand_k >= n - 1
        let exact = build_rerank_knn(&coords, n, dim, 12, 10, 16).unwrap();
        // 2-stage KD-tree when cand_k < n - 1 (k=5, rerank=4 => cand_k=20 < 149)
        let two_stage = build_rerank_knn(&coords, n, dim, 5, 10, 4).unwrap();
        assert_eq!(two_stage.nnz(), n * 5);

        for i in 0..n {
            let (e_idx, e_d) = exact.row(i);
            let (t_idx, t_d) = two_stage.row(i);
            assert_eq!(t_idx, &e_idx[..5]);
            for (&a, &b) in t_d.iter().zip(e_d[..5].iter()) {
                assert!((a - b).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_hybrid_adaptive_knn_preserves_12hop_and_margin_gap() {
        // Hub 0 connected to 1..=12; chain 12-13-...-24; isolated 25 (N=26).
        let mut edges: Vec<(u32, u32)> = (1..=12).map(|i| (0, i)).collect();
        edges.extend((12..24).map(|i| (i, i + 1)));
        let g = CsrGraph::<f64>::from_edges(26, &edges, None, false, false).unwrap();

        let (coords, _) = compute_resistance_embedding(&g, 12, 1e-6, 300, 42, 1e-10).unwrap();
        let dim = coords.len() / 26;
        let mut perps = vec![4.0; 26];
        // Force small target k=6 on hub 0 (which has 13 1+2-hop neighbors) -> must NOT truncate!
        perps[0] = 2.0;

        let cfg = HybridKnnConfig {
            k_factor: 3.0,
            k_override: None,
            margin_gap: 0.15,
            coarse_dim: 8,
            rerank_factor: 4,
            max_row_k: None,
        };
        let (knn, stats) = build_hybrid_adaptive_knn(&g, &coords, dim, &perps, &cfg).unwrap();
        assert_eq!(stats.n_truncated_1hop, 0);
        assert_eq!(stats.n_truncated_2hop, 0);
        assert!(stats.n_pure_2hop_rows >= 1);
        assert_eq!(knn.row(0).0.len(), 13);

        // Verify strict max(d_{1,2-hop}) < min(d_{non-2}) on rows with residual fills.
        let (adj1, adj2) = extract_1hop_2hop_csr(&g).unwrap();
        for i in 0..26 {
            let deg12 = adj1.degree(i) + adj2.degree(i);
            let (row_idx, row_d) = knn.row(i);
            assert!(row_idx.len() >= deg12);
            if deg12 > 0 && row_idx.len() > deg12 {
                let max_12 = row_d[..deg12].iter().copied().fold(0.0_f64, f64::max);
                let min_non2 = row_d[deg12..].iter().copied().fold(f64::INFINITY, f64::min);
                assert!(max_12 < min_non2, "node {i}: {max_12} vs {min_non2}");
                assert!(min_non2 >= max_12 * 1.15 - 1e-12);
            }
        }
        // Isolated node 25 gets round(3 * 4) = 12 neighbors.
        assert_eq!(knn.row(25).0.len(), 12);

        // Twin nodes with d_2hop_max == 0.0 still enforce min_non2 > 0.
        let tri_chain = CsrGraph::<f64>::from_edges(
            6,
            &[(0, 1), (1, 2), (0, 2), (3, 4), (4, 5)],
            None,
            false,
            false,
        )
        .unwrap();
        let twin_coords = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
        let (knn_t, _) = build_hybrid_adaptive_knn(
            &tri_chain,
            &twin_coords,
            2,
            &[1.5; 6],
            &HybridKnnConfig::default(),
        )
        .unwrap();
        for i in 0..3 {
            let (_ri, rd) = knn_t.row(i);
            assert_eq!(rd[0], 0.0);
            assert_eq!(rd[1], 0.0);
            assert!(rd[2] > 0.0);
        }
    }

    #[test]
    fn test_rerank_and_hybrid_empty_singleton_and_nan_rejection() {
        let empty_knn = build_rerank_knn(&[], 0, 2, 4, 2, 2).unwrap();
        assert_eq!(empty_knn.num_nodes(), 0);
        assert_eq!(empty_knn.nnz(), 0);

        let single_knn = build_rerank_knn(&[1.0, 2.0], 1, 2, 4, 2, 2).unwrap();
        assert_eq!(single_knn.num_nodes(), 1);
        assert_eq!(single_knn.nnz(), 0);

        assert!(build_rerank_knn(&[0.0, f64::NAN, 1.0, 2.0], 2, 2, 1, 2, 2).is_err());

        let g = CsrGraph::<f64>::from_edges(4, &[(0, 1), (2, 3)], None, false, false).unwrap();
        let bad_coords = vec![0.0, 0.0, 1.0, f64::NAN, 2.0, 0.0, 3.0, 0.0];
        assert!(
            build_hybrid_adaptive_knn(&g, &bad_coords, 2, &[2.0; 4], &HybridKnnConfig::default())
                .is_err()
        );
        assert!(
            build_hybrid_adaptive_knn(
                &g,
                &[0.0; 8],
                2,
                &[f64::NAN; 4],
                &HybridKnnConfig::default()
            )
            .is_err()
        );
    }
}
