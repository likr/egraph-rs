//! 2-hop adaptive perplexity sizing, `f64` $\lambda$-rescaling binary search for sparse joint
//! probabilities, row-normalized symmetrization, and `VisitStamp`-backed Partial BFS.

use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, Scalar, VisitStamp};

use crate::knn::{KnnGraph, canonical_unweighted_1hop};

/// Default multiplier $c$ for node-wise 2-hop adaptive perplexity (`c = 0.9`).
pub const DEFAULT_NODEWISE_PERPLEXITY_C: f64 = 0.9;
/// Default lower clamp for node-wise 2-hop adaptive perplexity (`5.0`).
pub const DEFAULT_NODEWISE_MIN_PERPLEXITY: f64 = 5.0;
/// Default multiplier $c$ for median 2-hop adaptive perplexity (`c = 1.5`).
pub const DEFAULT_MEDIAN_PERPLEXITY_C: f64 = 1.5;
/// Default lower clamp for median 2-hop adaptive perplexity (`15.0`).
pub const DEFAULT_MEDIAN_MIN_PERPLEXITY: f64 = 15.0;
/// Default upper clamp for 2-hop adaptive perplexity (`1200.0`).
pub const DEFAULT_MAX_PERPLEXITY: f64 = 1200.0;

/// Symmetrization strategy applied to directed conditional probabilities or graph affinities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SymmetrizationMode {
    /// Standard t-SNE joint normalization: $P_{ij} = (p_{j|i} + p_{i|j}) / (2 N)$.
    #[default]
    StandardTsne,
    /// Explicit row-normalized symmetrization: $P = (D_{\mathrm{row}}^{-1} A + A^T D_{\mathrm{row}}^{-1}) / (2 N)$.
    RowNormalized,
}

/// Configuration for perplexity-based sparse probability computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerplexityConfig {
    /// Power exponent applied to normalized distances `(d / max_d)^power` (`default: 2.0`).
    pub power: f64,
    /// Maximum binary-search iterations for conditional entropy matching (`default: 50`).
    pub sigma_iters: usize,
    /// Entropy convergence tolerance (`default: 1e-5`).
    pub sigma_tolerance: f64,
    /// Symmetrization strategy (`default: SymmetrizationMode::StandardTsne`).
    pub symmetrization: SymmetrizationMode,
}

impl Default for PerplexityConfig {
    fn default() -> Self {
        Self {
            power: 2.0,
            sigma_iters: 50,
            sigma_tolerance: 1e-5,
            symmetrization: SymmetrizationMode::StandardTsne,
        }
    }
}

/// Canonical sorted symmetric Compressed Sparse Row joint probability matrix $P$.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseProbabilities {
    pub n_nodes: usize,
    pub indptr: Vec<usize>,
    pub indices: Vec<u32>,
    pub values: Vec<f64>,
}

impl SparseProbabilities {
    /// Creates an empty [`SparseProbabilities`] with `n_nodes` zero-degree rows.
    pub fn empty(n_nodes: usize) -> Self {
        Self {
            n_nodes,
            indptr: vec![0; n_nodes + 1],
            indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Returns the number of vertices.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Returns the total number of stored probability entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Returns `(neighbor_indices, probabilities)` for row `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.n_nodes`.
    #[inline]
    pub fn row(&self, i: usize) -> (&[u32], &[f64]) {
        let start = self.indptr[i];
        let end = self.indptr[i + 1];
        (&self.indices[start..end], &self.values[start..end])
    }
}

/// Computes exact distinct 1-hop + 2-hop neighborhood sizes $|B_2(u) \setminus \{u\}|$ for every
/// vertex in `graph` using a single reusable `VisitStamp` (`O(1)` clear per vertex).
///
/// Directed inputs are treated as undirected simple adjacency.
pub fn compute_2hop_neighborhood_sizes<W>(graph: &CsrGraph<W>) -> Vec<u32> {
    let n = graph.num_nodes();
    if n == 0 {
        return Vec::new();
    }

    if !graph.is_directed() {
        let mut stamp = VisitStamp::new(n);
        let mut sizes = vec![0u32; n];
        for (u, size_out) in sizes.iter_mut().enumerate() {
            stamp.clear();
            stamp.mark(u);
            let mut count = 0u32;
            let n1 = graph.neighbors(u);
            for &w in n1 {
                let wi = w as usize;
                if wi < n && stamp.visit(wi) {
                    count += 1;
                }
            }
            for &w in n1 {
                let wi = w as usize;
                if wi >= n || wi == u {
                    continue;
                }
                for &v in graph.neighbors(wi) {
                    let vi = v as usize;
                    if vi < n && stamp.visit(vi) {
                        count += 1;
                    }
                }
            }
            *size_out = count;
        }
        return sizes;
    }

    let Ok(adj1) = canonical_unweighted_1hop(graph) else {
        return vec![0; n];
    };
    compute_2hop_neighborhood_sizes(&adj1)
}

fn validate_adaptive_perplexity_params(
    c: f64,
    min_perplexity: f64,
    max_perplexity: f64,
    context: &'static str,
) -> CoreResult<()> {
    if !c.is_finite() {
        return Err(CoreError::non_finite(context));
    }
    if c <= 0.0 {
        return Err(CoreError::invalid_input(format!("c must be > 0, got {c}")));
    }
    if !min_perplexity.is_finite() {
        return Err(CoreError::non_finite(context));
    }
    if min_perplexity <= 0.0 {
        return Err(CoreError::invalid_input(format!(
            "min_perplexity must be > 0, got {min_perplexity}"
        )));
    }
    if !max_perplexity.is_finite() {
        return Err(CoreError::non_finite(context));
    }
    if max_perplexity < min_perplexity {
        return Err(CoreError::invalid_input(format!(
            "max_perplexity ({max_perplexity}) must be >= min_perplexity ({min_perplexity})"
        )));
    }
    Ok(())
}

/// Computes per-node 2-hop adaptive perplexities
/// $u_i = \mathrm{clip}(c \cdot k_i^{(2)},\; \min(\mathrm{min\_perplexity}, U),\; U)$
/// where $U = \min(\mathrm{max\_perplexity}, (N - 1) / 3.0)$.
///
/// Recommended defaults: `c = 0.9`, `min_perplexity = 5.0`, `max_perplexity = 1200.0`.
pub fn compute_nodewise_2hop_perplexities<W>(
    graph: &CsrGraph<W>,
    c: f64,
    min_perplexity: f64,
    max_perplexity: f64,
) -> CoreResult<Vec<f64>> {
    validate_adaptive_perplexity_params(
        c,
        min_perplexity,
        max_perplexity,
        "compute_nodewise_2hop_perplexities",
    )?;
    let n = graph.num_nodes();
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        return Ok(vec![1.0]);
    }
    let upper = max_perplexity.min((n - 1) as f64 / 3.0);
    if upper < 1.0 {
        return Ok(vec![1.0; n]);
    }
    let lower = min_perplexity.min(upper);
    let sizes = compute_2hop_neighborhood_sizes(graph);
    Ok(sizes
        .into_iter()
        .map(|k| (c * (k as f64)).clamp(lower, upper))
        .collect())
}

/// Computes scalar median 2-hop adaptive perplexity
/// $u = \mathrm{clip}(c \cdot \mathrm{median}_i(k_i^{(2)}),\; \min(\mathrm{min\_perplexity}, U),\; U)$
/// where $U = \min(\mathrm{max\_perplexity}, (N - 1) / 3.0)$.
///
/// Recommended defaults: `c = 1.5`, `min_perplexity = 15.0`, `max_perplexity = 1200.0`.
pub fn compute_median_2hop_perplexity<W>(
    graph: &CsrGraph<W>,
    c: f64,
    min_perplexity: f64,
    max_perplexity: f64,
) -> CoreResult<f64> {
    validate_adaptive_perplexity_params(
        c,
        min_perplexity,
        max_perplexity,
        "compute_median_2hop_perplexity",
    )?;
    let n = graph.num_nodes();
    if n <= 1 {
        return Ok(1.0);
    }
    let upper = max_perplexity.min((n - 1) as f64 / 3.0);
    if upper < 1.0 {
        return Ok(1.0);
    }
    let lower = min_perplexity.min(upper);
    let mut sizes = compute_2hop_neighborhood_sizes(graph);
    sizes.sort_unstable();
    let mid = n / 2;
    let med = if n % 2 == 1 {
        sizes[mid] as f64
    } else {
        0.5 * ((sizes[mid - 1] as f64) + (sizes[mid] as f64))
    };
    Ok((c * med).clamp(lower, upper))
}

/// Symmetrizes undirected pairs `(u, v, p_val)` with `u < v` into canonical sorted symmetric CSR
/// [`SparseProbabilities`] without per-row `HashMap`s.
fn scatter_symmetric_pairs(
    n_nodes: usize,
    mut pairs: Vec<(u32, u32, f64)>,
    scale_by_two_n: bool,
) -> SparseProbabilities {
    if n_nodes < 2 || pairs.is_empty() {
        return SparseProbabilities::empty(n_nodes);
    }

    pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let inv_two_n = if scale_by_two_n {
        1.0 / (2.0 * (n_nodes as f64))
    } else {
        1.0
    };

    let mut merged: Vec<(u32, u32, f64)> = Vec::with_capacity(pairs.len());
    let mut deg = vec![0usize; n_nodes];

    let mut i = 0usize;
    while i < pairs.len() {
        let (u, v, mut acc) = pairs[i];
        i += 1;
        while i < pairs.len() && pairs[i].0 == u && pairs[i].1 == v {
            acc += pairs[i].2;
            i += 1;
        }
        let p_sym = acc * inv_two_n;
        if p_sym > 0.0 {
            merged.push((u, v, p_sym));
            deg[u as usize] += 1;
            deg[v as usize] += 1;
        }
    }

    if merged.is_empty() {
        return SparseProbabilities::empty(n_nodes);
    }

    let mut indptr = Vec::with_capacity(n_nodes + 1);
    indptr.push(0usize);
    let mut total = 0usize;
    for d in &deg {
        total += *d;
        indptr.push(total);
    }
    let mut cursor = indptr[..n_nodes].to_vec();
    let mut indices = vec![0u32; total];
    let mut values = vec![0.0_f64; total];

    // Iterating pairs sorted by (u, v) with u < v inserts neighbors < x before neighbors > x
    // in strictly ascending column order without per-row sorting.
    for (u, v, p_sym) in merged {
        let pu = cursor[u as usize];
        indices[pu] = v;
        values[pu] = p_sym;
        cursor[u as usize] += 1;

        let pv = cursor[v as usize];
        indices[pv] = u;
        values[pv] = p_sym;
        cursor[v as usize] += 1;
    }

    SparseProbabilities {
        n_nodes,
        indptr,
        indices,
        values,
    }
}

/// Computes row-normalized symmetrized joint probabilities
/// $P = (D_{\mathrm{row}}^{-1} A + A^T D_{\mathrm{row}}^{-1}) / (2 N)$ from arbitrary CSR
/// directed affinities `(indptr, indices, weights)`.
///
/// Self-loops and non-positive weights are dropped before row normalization; duplicate entries in
/// the same row are summed via a reusable `VisitStamp` without per-row `HashMap`s.
pub fn symmetrize_row_normalized(
    n_nodes: usize,
    indptr: &[usize],
    indices: &[u32],
    weights: &[f64],
) -> CoreResult<SparseProbabilities> {
    if indptr.len() != n_nodes + 1 {
        return Err(CoreError::dimension_mismatch(
            n_nodes + 1,
            indptr.len(),
            "symmetrize_row_normalized::indptr",
        ));
    }
    if indptr[0] != 0 {
        return Err(CoreError::invalid_csr(format!(
            "indptr[0] must be 0, got {}",
            indptr[0]
        )));
    }
    for row in 0..n_nodes {
        if indptr[row + 1] < indptr[row] {
            return Err(CoreError::invalid_csr(format!(
                "indptr must be non-decreasing at row {row}: {} > {}",
                indptr[row],
                indptr[row + 1]
            )));
        }
    }
    let nnz = indptr[n_nodes];
    if indices.len() != nnz {
        return Err(CoreError::dimension_mismatch(
            nnz,
            indices.len(),
            "symmetrize_row_normalized::indices",
        ));
    }
    if weights.len() != nnz {
        return Err(CoreError::dimension_mismatch(
            nnz,
            weights.len(),
            "symmetrize_row_normalized::weights",
        ));
    }
    for (&idx, &w) in indices.iter().zip(weights.iter()) {
        if (idx as usize) >= n_nodes {
            return Err(CoreError::index_out_of_bounds(
                idx as usize,
                n_nodes,
                "symmetrize_row_normalized::indices",
            ));
        }
        if !w.is_finite() {
            return Err(CoreError::non_finite("symmetrize_row_normalized::weights"));
        }
        if w < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "weights must be non-negative, got {w}"
            )));
        }
    }

    if n_nodes < 2 {
        return Ok(SparseProbabilities::empty(n_nodes));
    }

    let mut seen = VisitStamp::new(n_nodes);
    let mut seen_pos = vec![0usize; n_nodes];
    let mut row_entries: Vec<(u32, f64)> = Vec::new();
    let mut pairs: Vec<(u32, u32, f64)> = Vec::with_capacity(nnz);

    for i in 0..n_nodes {
        seen.clear();
        row_entries.clear();
        let start = indptr[i];
        let end = indptr[i + 1];
        for pos in start..end {
            let j = indices[pos];
            let j_usize = j as usize;
            let w = weights[pos];
            if j_usize == i || w <= 0.0 {
                continue;
            }
            if seen.visit(j_usize) {
                seen_pos[j_usize] = row_entries.len();
                row_entries.push((j, w));
            } else {
                row_entries[seen_pos[j_usize]].1 += w;
            }
        }
        let row_sum: f64 = row_entries.iter().map(|&(_, w)| w).sum();
        if row_sum > 0.0 {
            let inv = 1.0 / row_sum;
            let i_u32 = i as u32;
            for &(j, w) in &row_entries {
                let a_tilde = w * inv;
                let (u, v) = if i_u32 < j { (i_u32, j) } else { (j, i_u32) };
                pairs.push((u, v, a_tilde));
            }
        }
    }

    Ok(scatter_symmetric_pairs(n_nodes, pairs, true))
}

/// Computes row-normalized symmetrized graph probabilities
/// $P = (D_{\mathrm{row}}^{-1} A + A^T D_{\mathrm{row}}^{-1}) / (2 N)$ directly from a
/// [`CsrGraph<W>`] (Böhm et al. 2025 `graph t-SNE` normalization).
pub fn graph_row_normalized_probabilities<W: Scalar>(
    graph: &CsrGraph<W>,
) -> CoreResult<SparseProbabilities> {
    let n = graph.num_nodes();
    if n < 2 {
        return Ok(SparseProbabilities::empty(n));
    }
    let indptr_usize: Vec<usize> = graph.indptr().iter().map(|&p| p as usize).collect();
    let weights: Vec<f64> = match graph.weights() {
        Some(ws) => ws.iter().map(|w| w.to_f64()).collect(),
        None => vec![1.0; graph.nnz()],
    };
    symmetrize_row_normalized(n, &indptr_usize, graph.indices(), &weights)
}

/// Computes sparse symmetric high-dimensional joint probabilities from a validated [`KnnGraph`]
/// and per-node target `perplexities` using $\lambda$-rescaling `(d / max_d)^power`, `f64`
/// entropy binary search, reusable `VisitStamp` deduplication, and canonical sorted CSR output.
pub fn compute_knn_probabilities(
    knn: &KnnGraph,
    perplexities: &[f64],
    config: &PerplexityConfig,
) -> CoreResult<SparseProbabilities> {
    let n = knn.num_nodes();
    if perplexities.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            perplexities.len(),
            "compute_knn_probabilities::perplexities",
        ));
    }
    for &p in perplexities {
        if !p.is_finite() {
            return Err(CoreError::non_finite(
                "compute_knn_probabilities::perplexities",
            ));
        }
        if p <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "perplexities must be > 0, got {p}"
            )));
        }
    }
    if !config.power.is_finite() {
        return Err(CoreError::non_finite("PerplexityConfig::power"));
    }
    if config.power <= 0.0 {
        return Err(CoreError::invalid_input(format!(
            "power must be > 0, got {}",
            config.power
        )));
    }
    if config.sigma_iters == 0 {
        return Err(CoreError::invalid_input("sigma_iters must be >= 1"));
    }
    if !config.sigma_tolerance.is_finite() {
        return Err(CoreError::non_finite("PerplexityConfig::sigma_tolerance"));
    }
    if config.sigma_tolerance <= 0.0 {
        return Err(CoreError::invalid_input(format!(
            "sigma_tolerance must be > 0, got {}",
            config.sigma_tolerance
        )));
    }

    if n < 2 {
        return Ok(SparseProbabilities::empty(n));
    }

    const MIN_SUM: f64 = 1e-12;
    const PERP_MARGIN: f64 = 0.01;

    let mut seen = VisitStamp::new(n);
    let mut seen_pos = vec![0usize; n];
    let mut collected: Vec<(u32, f64)> = Vec::new();
    let mut dp_values: Vec<f64> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();

    let mut dir_indptr = Vec::with_capacity(n + 1);
    dir_indptr.push(0usize);
    let mut dir_indices: Vec<u32> = Vec::with_capacity(knn.nnz());
    let mut dir_values: Vec<f64> = Vec::with_capacity(knn.nnz());

    for (i, &target_perp) in perplexities.iter().enumerate() {
        seen.clear();
        collected.clear();

        let (row_idx, row_dist) = knn.row(i);
        for (&j, &d) in row_idx.iter().zip(row_dist.iter()) {
            let j_usize = j as usize;
            if j_usize == i || j_usize >= n || !d.is_finite() {
                continue;
            }
            let d_nonneg = d.max(0.0);
            if seen.visit(j_usize) {
                seen_pos[j_usize] = collected.len();
                collected.push((j, d_nonneg));
            } else {
                let pos = seen_pos[j_usize];
                if d_nonneg < collected[pos].1 {
                    collected[pos].1 = d_nonneg;
                }
            }
        }

        let m = collected.len();
        if m == 0 {
            dir_indptr.push(dir_indices.len());
            continue;
        }
        if m == 1 {
            dir_indices.push(collected[0].0);
            dir_values.push(1.0);
            dir_indptr.push(dir_indices.len());
            continue;
        }

        // Normalize distances before applying power and shifting by min_dp so binary search
        // operates over [0, 1] without overflow or underflow across arbitrary distance scales.
        let mut max_d = 0.0_f64;
        for &(_, d) in &collected {
            if d > max_d {
                max_d = d;
            }
        }

        if max_d <= 0.0 {
            let p_uni = 1.0 / (m as f64);
            for &(j, _) in &collected {
                dir_indices.push(j);
                dir_values.push(p_uni);
            }
            dir_indptr.push(dir_indices.len());
            continue;
        }

        dp_values.clear();
        let mut min_dp = f64::INFINITY;
        let mut max_dp = f64::NEG_INFINITY;
        for &(_, d) in &collected {
            let dp = (d / max_d).powf(config.power);
            let dp_safe = if dp.is_finite() { dp } else { 1.0 };
            if dp_safe < min_dp {
                min_dp = dp_safe;
            }
            if dp_safe > max_dp {
                max_dp = dp_safe;
            }
            dp_values.push(dp_safe);
        }

        let range = max_dp - min_dp;
        if range <= 0.0 {
            let p_uni = 1.0 / (m as f64);
            for &(j, _) in &collected {
                dir_indices.push(j);
                dir_values.push(p_uni);
            }
            dir_indptr.push(dir_indices.len());
            continue;
        }

        for dp in &mut dp_values {
            *dp = (*dp - min_dp) / range;
        }

        let max_perp = ((m as f64) - PERP_MARGIN).max(1.0);
        let perplexity_clamped = target_perp.min(max_perp).max(1.0);
        let target_entropy = perplexity_clamped.ln();

        let mut beta = 1.0_f64;
        let mut beta_min = 0.0_f64;
        let mut beta_max: Option<f64> = None;

        weights.resize(m, 0.0);
        for _ in 0..config.sigma_iters {
            let mut sum_w = 0.0_f64;
            for (idx, &dp) in dp_values.iter().enumerate() {
                let w = (-beta * dp).exp();
                weights[idx] = w;
                sum_w += w;
            }
            sum_w = sum_w.max(MIN_SUM);

            let mut sum_dp_p = 0.0_f64;
            for (idx, &dp) in dp_values.iter().enumerate() {
                sum_dp_p += (weights[idx] / sum_w) * dp;
            }

            let h = beta * sum_dp_p + sum_w.ln();
            let h_diff = h - target_entropy;
            if h_diff.abs() < config.sigma_tolerance {
                break;
            }

            if h_diff > 0.0 {
                beta_min = beta;
                match beta_max {
                    None => {
                        let next = beta * 2.0;
                        if !next.is_finite() {
                            break;
                        }
                        beta = next;
                    }
                    Some(bmax) => {
                        beta = 0.5 * (beta_min + bmax);
                    }
                }
            } else {
                beta_max = Some(beta);
                beta = 0.5 * (beta_min + beta);
            }
        }

        let mut sum_w = 0.0_f64;
        for (idx, &dp) in dp_values.iter().enumerate() {
            let w = (-beta * dp).exp();
            weights[idx] = w;
            sum_w += w;
        }
        sum_w = sum_w.max(MIN_SUM);

        for (idx, &(j, _)) in collected.iter().enumerate() {
            dir_indices.push(j);
            dir_values.push(weights[idx] / sum_w);
        }
        dir_indptr.push(dir_indices.len());
    }

    match config.symmetrization {
        SymmetrizationMode::StandardTsne => {
            let mut pairs = Vec::with_capacity(dir_indices.len());
            for i in 0..n {
                let i_u32 = i as u32;
                let start = dir_indptr[i];
                let end = dir_indptr[i + 1];
                for pos in start..end {
                    let j = dir_indices[pos];
                    let p_ji = dir_values[pos];
                    if p_ji > 0.0 {
                        let (u, v) = if i_u32 < j { (i_u32, j) } else { (j, i_u32) };
                        pairs.push((u, v, p_ji));
                    }
                }
            }
            Ok(scatter_symmetric_pairs(n, pairs, true))
        }
        SymmetrizationMode::RowNormalized => {
            symmetrize_row_normalized(n, &dir_indptr, &dir_indices, &dir_values)
        }
    }
}

/// Computes a $k$-NN graph via Partial BFS in $O(N \cdot k \cdot \bar{d})$ time using a single
/// reusable `VisitStamp` and deterministic Fisher-Yates tie-breaking at boundary hop shells.
pub fn partial_bfs_knn<W>(graph: &CsrGraph<W>, k: usize, seed: u64) -> CoreResult<KnnGraph> {
    let n = graph.num_nodes();
    if u32::try_from(n).is_err() {
        return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
    }
    if n < 2 || k == 0 {
        return KnnGraph::empty(n);
    }

    let k_eff = k.min(n - 1);
    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut visited = VisitStamp::new(n);
    let mut current_layer: Vec<u32> = Vec::new();
    let mut next_layer: Vec<u32> = Vec::new();

    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0usize);
    let mut indices = Vec::with_capacity(n * k_eff);
    let mut distances = Vec::with_capacity(n * k_eff);

    for i in 0..n {
        visited.clear();
        visited.mark(i);
        current_layer.clear();
        current_layer.push(i as u32);

        let mut collected = 0usize;
        let mut current_dist = 0.0_f64;

        while !current_layer.is_empty() && collected < k_eff {
            current_dist += 1.0;
            next_layer.clear();

            for &u in &current_layer {
                for &v in graph.neighbors(u as usize) {
                    let vi = v as usize;
                    if vi < n && visited.visit(vi) {
                        next_layer.push(v);
                    }
                }
            }

            if next_layer.is_empty() {
                break;
            }

            let remaining = k_eff - collected;
            if next_layer.len() <= remaining {
                for &v in &next_layer {
                    indices.push(v);
                    distances.push(current_dist);
                }
                collected += next_layer.len();
                std::mem::swap(&mut current_layer, &mut next_layer);
            } else {
                rng.shuffle(&mut next_layer);
                for &v in next_layer.iter().take(remaining) {
                    indices.push(v);
                    distances.push(current_dist);
                }
                break;
            }
        }

        indptr.push(indices.len());
    }

    KnnGraph::new_validated(n, indptr, indices, distances)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2hop_adaptive_perplexity_nodewise_and_median() {
        // Hub 0 connected to 1..=15; chain 15-16-17; isolated 18, 19 (N=20).
        let mut edges: Vec<(u32, u32)> = (1..=15).map(|i| (0, i)).collect();
        edges.extend([(15, 16), (16, 17)]);
        let g = CsrGraph::<f64>::from_edges(20, &edges, None, false, false).unwrap();

        let sizes = compute_2hop_neighborhood_sizes(&g);
        assert_eq!(sizes[0], 16); // 1..=15 (1-hop) + 16 (2-hop)
        assert_eq!(sizes[18], 0);
        assert_eq!(sizes[19], 0);

        let nodewise = compute_nodewise_2hop_perplexities(
            &g,
            DEFAULT_NODEWISE_PERPLEXITY_C,
            DEFAULT_NODEWISE_MIN_PERPLEXITY,
            DEFAULT_MAX_PERPLEXITY,
        )
        .unwrap();
        let upper = 19.0 / 3.0;
        assert!((nodewise[0] - upper).abs() < 1e-12); // 0.9*16=14.4 clamped to 19/3
        assert!((nodewise[18] - 5.0).abs() < 1e-12); // isolated clamped to lower=5.0

        let med = compute_median_2hop_perplexity(
            &g,
            DEFAULT_MEDIAN_PERPLEXITY_C,
            DEFAULT_MEDIAN_MIN_PERPLEXITY,
            DEFAULT_MAX_PERPLEXITY,
        )
        .unwrap();
        assert!((med - upper).abs() < 1e-12);

        // Small N=0..=3 safety
        let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        assert!(
            compute_nodewise_2hop_perplexities(&empty, 0.9, 5.0, 1200.0)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            compute_median_2hop_perplexity(&empty, 1.5, 15.0, 1200.0).unwrap(),
            1.0
        );
        let tiny3 = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, false, false).unwrap();
        assert_eq!(
            compute_nodewise_2hop_perplexities(&tiny3, 0.9, 5.0, 1200.0).unwrap(),
            vec![1.0, 1.0, 1.0]
        );
    }

    #[test]
    fn test_compute_knn_probabilities_entropy_symmetry_and_row_normalized() {
        // Construct non-uniform k-NN with duplicates and self-loops to verify filtering and entropy.
        let knn = KnnGraph::new_validated(
            4,
            vec![0, 5, 7, 9, 11],
            vec![
                0, 1, 1, 2, 3, // row 0: self-loop 0 + dup 1
                0, 2, // row 1
                1, 3, // row 2
                0, 2, // row 3
            ],
            vec![
                0.0, 5.0, 0.2, 1.0, 3.0, // row 0 min for 1 is 0.2
                0.2, 0.8, // row 1
                0.8, 1.2, // row 2
                3.0, 1.2, // row 3
            ],
        )
        .unwrap();
        let perps = vec![2.0, 1.5, 1.5, 1.8];
        let cfg = PerplexityConfig::default();
        let sp = compute_knn_probabilities(&knn, &perps, &cfg).unwrap();

        let mut total = 0.0;
        for i in 0..4 {
            let (cols, vals) = sp.row(i);
            assert!(cols.windows(2).all(|w| w[0] < w[1]));
            assert!(!cols.contains(&(i as u32)));
            for (&j, &p_ij) in cols.iter().zip(vals.iter()) {
                assert!(p_ij.is_finite() && p_ij > 0.0);
                let (rcols, rvals) = sp.row(j as usize);
                let pos = rcols.binary_search(&(i as u32)).unwrap();
                assert_eq!(p_ij.to_bits(), rvals[pos].to_bits());
                total += p_ij;
            }
        }
        assert!((total - 1.0).abs() < 1e-12);

        // Verify RowNormalized graph probabilities P = (D^{-1} A + A^T D^{-1}) / (2N)
        let star =
            CsrGraph::<f64>::from_edges(5, &[(0, 1), (0, 2), (0, 3), (0, 4)], None, false, false)
                .unwrap();
        let p_graph = graph_row_normalized_probabilities(&star).unwrap();
        // For leaf 1 -> hub 0: a_{1->0} = 1.0, a_{0->1} = 0.25 => P_{01} = (1 + 0.25) / 10 = 0.125.
        let (r0, v0) = p_graph.row(0);
        assert_eq!(r0, &[1, 2, 3, 4]);
        for &val in v0 {
            assert!((val - 0.125).abs() < 1e-12);
        }
        let p_total: f64 = p_graph.values.iter().sum();
        assert!((p_total - 1.0).abs() < 1e-12);

        // Verify RowNormalized mode inside compute_knn_probabilities matches symmetrize_row_normalized.
        let cfg_rn = PerplexityConfig {
            symmetrization: SymmetrizationMode::RowNormalized,
            ..Default::default()
        };
        let sp_rn = compute_knn_probabilities(&knn, &perps, &cfg_rn).unwrap();
        let sp_rn_total: f64 = sp_rn.values.iter().sum();
        assert!((sp_rn_total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_compute_knn_probabilities_entropy_accuracy_on_distinct_row() {
        // Single node with 5 distinct neighbors: verify conditional probability perplexity matches target.
        let knn = KnnGraph::new_validated(
            6,
            vec![0, 5, 6, 7, 8, 9, 10],
            vec![1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
            vec![0.1, 0.4, 0.9, 1.6, 2.5, 0.1, 0.4, 0.9, 1.6, 2.5],
        )
        .unwrap();
        // Verify node 0 conditional probabilities via StandardTsne vs non-zero entries:
        // Note: rows 1..=5 only have node 0 so p_{0|j} = 1.0, while p_{j|0} = 2*6*P_{0j} - 1.0.
        let target_perp = 2.5;
        let perps = vec![target_perp, 1.0, 1.0, 1.0, 1.0, 1.0];
        let sp = compute_knn_probabilities(
            &knn,
            &perps,
            &PerplexityConfig {
                sigma_tolerance: 1e-9,
                sigma_iters: 80,
                ..Default::default()
            },
        )
        .unwrap();
        let (_cols, vals) = sp.row(0);
        let p_cond: Vec<f64> = vals.iter().map(|&p0j| 12.0 * p0j - 1.0).collect();
        let sum_p: f64 = p_cond.iter().sum();
        assert!((sum_p - 1.0).abs() < 1e-12);
        let entropy: f64 = p_cond
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();
        assert!((entropy.exp() - target_perp).abs() < 1e-7);
    }

    #[test]
    fn test_partial_bfs_knn_determinism_and_tiers() {
        let mut edges = Vec::new();
        for r in 0..5 {
            for c in 0..5 {
                let u = (r * 5 + c) as u32;
                if c + 1 < 5 {
                    edges.push((u, u + 1));
                }
                if r + 1 < 5 {
                    edges.push((u, u + 5));
                }
            }
        }
        let g = CsrGraph::<f64>::from_edges(25, &edges, None, false, false).unwrap();

        let k1 = partial_bfs_knn(&g, 6, 42).unwrap();
        let k2 = partial_bfs_knn(&g, 6, 42).unwrap();
        assert_eq!(k1, k2);
        for i in 0..25 {
            let (nbrs, dists) = k1.row(i);
            assert_eq!(nbrs.len(), 6);
            assert!(dists.windows(2).all(|w| w[0] <= w[1]));
            assert!(!nbrs.contains(&(i as u32)));
        }

        // k == 0 returns empty rows
        let k0 = partial_bfs_knn(&g, 0, 42).unwrap();
        assert_eq!(k0.nnz(), 0);
    }

    #[test]
    fn test_perplexity_disconnected_singleton_and_nan_rejection() {
        // Two disconnected components (0-1-2 and 3-4) + isolated node 5
        let g =
            CsrGraph::<f64>::from_edges(6, &[(0, 1), (1, 2), (3, 4)], None, false, false).unwrap();
        let knn = partial_bfs_knn(&g, 4, 7).unwrap();
        // Isolated node 5 has 0 reachable neighbors; component {3,4} reaches only 1 neighbor.
        assert_eq!(knn.row(5).0.len(), 0);
        assert_eq!(knn.row(3).0, &[4]);
        assert_eq!(knn.row(0).0.len(), 2);

        let probs =
            compute_knn_probabilities(&knn, &[1.5; 6], &PerplexityConfig::default()).unwrap();
        assert_eq!(probs.n_nodes, 6);
        assert!(probs.values.iter().all(|v| v.is_finite() && *v > 0.0));

        let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        assert_eq!(partial_bfs_knn(&single, 5, 1).unwrap().nnz(), 0);
        assert!(
            compute_knn_probabilities(
                &knn,
                &[1.5, f64::NAN, 1.5, 1.5, 1.5, 1.5],
                &PerplexityConfig::default()
            )
            .is_err()
        );
        assert!(compute_nodewise_2hop_perplexities(&g, f64::NAN, 5.0, 1200.0).is_err());
        assert!(compute_median_2hop_perplexity(&g, 1.5, 15.0, -1.0).is_err());
    }
}
