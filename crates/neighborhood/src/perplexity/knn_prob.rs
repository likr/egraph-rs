//! Conditional entropy binary search (`compute_knn_probabilities`) and Partial BFS $k$-NN
//! graph construction (`partial_bfs_knn`).

use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, VisitStamp};

use crate::knn::KnnGraph;

use super::config::{PerplexityConfig, SparseProbabilities, SymmetrizationMode};
use super::symmetrize::{scatter_symmetric_pairs, symmetrize_row_normalized};

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

    match config.symmetrization {
        SymmetrizationMode::StandardTsne => scatter_symmetric_pairs(n, pairs, true),
        SymmetrizationMode::RowNormalized => {
            let sym = scatter_symmetric_pairs(n, pairs, false)?;
            symmetrize_row_normalized(n, &sym.indptr, &sym.indices, &sym.values)
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
    indptr.push(0u32);
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

        let ptr = u32::try_from(indices.len())
            .map_err(|_| CoreError::invalid_csr("partial_bfs_knn nnz exceeds u32::MAX"))?;
        indptr.push(ptr);
    }

    KnnGraph::new_validated(n, indptr, indices, distances)
}
