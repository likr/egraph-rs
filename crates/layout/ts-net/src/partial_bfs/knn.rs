use super::{validate_csr_knn, validate_perplexities, SparseP};
use num_traits::Float;
use petgraph_drawing::DrawingValue;
use std::collections::HashMap;

/// Computes sparse high-dimensional symmetric joint probabilities
/// `P_{ij} = (P_{j|i} + P_{i|j}) / (2N)` from precomputed k-NN neighbor lists.
///
/// Each row `knn[i]` contains `(neighbor_idx, distance)` pairs for node `i`. Self-loops (`j == i`),
/// out-of-range indices (`j >= n`), and non-finite distances are filtered out, duplicate neighbors in
/// a row are deduplicated by retaining the minimum distance, and empty rows are handled safely.
pub fn compute_knn_probabilities<S>(n: usize, knn: &[Vec<(usize, S)>], perplexity: S) -> SparseP<S>
where
    S: DrawingValue + Float + Default,
{
    compute_knn_probabilities_with_params(
        n,
        knn,
        perplexity,
        S::from_f32(2.0).unwrap(),
        50,
        S::from_f32(1e-5).unwrap(),
    )
}

/// Computes sparse high-dimensional symmetric joint probabilities from precomputed k-NN neighbor
/// lists using explicit distance exponent and perplexity binary search parameters.
pub fn compute_knn_probabilities_with_params<S>(
    n: usize,
    knn: &[Vec<(usize, S)>],
    perplexity: S,
    power: S,
    sigma_iters: usize,
    sigma_tolerance: S,
) -> SparseP<S>
where
    S: DrawingValue + Float + Default,
{
    compute_knn_probabilities_core(
        n,
        |i| knn.get(i).into_iter().flat_map(|row| row.iter().copied()),
        |_| perplexity,
        power,
        sigma_iters,
        sigma_tolerance,
    )
}

/// Computes sparse high-dimensional symmetric joint probabilities from precomputed k-NN neighbor
/// lists using per-node target perplexities.
pub fn compute_knn_probabilities_with_perplexities<S>(
    n: usize,
    knn: &[Vec<(usize, S)>],
    perplexities: &[S],
    power: S,
    sigma_iters: usize,
    sigma_tolerance: S,
) -> Result<SparseP<S>, String>
where
    S: DrawingValue + Float + Default,
{
    if knn.len() != n {
        return Err(format!(
            "knn row count ({}) must match node count ({n})",
            knn.len()
        ));
    }
    validate_perplexities(n, perplexities)?;
    for (i, row) in knn.iter().enumerate() {
        for &(j, d) in row {
            if j >= n {
                return Err(format!(
                    "knn[{i}] neighbor index {j} is out of range for {n} nodes"
                ));
            }
            if !d.is_finite() || d < S::zero() {
                return Err(format!(
                    "knn[{i}] distance to neighbor {j} must be finite and non-negative"
                ));
            }
        }
    }
    if n < 2 {
        return Ok(vec![Vec::new(); n]);
    }
    Ok(compute_knn_probabilities_core(
        n,
        |i| knn[i].iter().copied(),
        |i| perplexities[i],
        power,
        sigma_iters,
        sigma_tolerance,
    ))
}

/// Computes sparse high-dimensional symmetric joint probabilities directly from CSR k-NN arrays
/// `(indptr, indices, distances)` and per-node target perplexities without intermediate row copies.
#[allow(clippy::too_many_arguments)]
pub fn compute_knn_probabilities_from_csr<S>(
    n: usize,
    indptr: &[usize],
    indices: &[usize],
    distances: &[S],
    perplexities: &[S],
    power: S,
    sigma_iters: usize,
    sigma_tolerance: S,
) -> Result<SparseP<S>, String>
where
    S: DrawingValue + Float + Default,
{
    validate_csr_knn(n, indptr, indices, distances)?;
    validate_perplexities(n, perplexities)?;
    if n < 2 {
        return Ok(vec![Vec::new(); n]);
    }
    Ok(compute_knn_probabilities_core(
        n,
        |i| {
            let start = indptr[i];
            let end = indptr[i + 1];
            indices[start..end]
                .iter()
                .copied()
                .zip(distances[start..end].iter().copied())
        },
        |i| perplexities[i],
        power,
        sigma_iters,
        sigma_tolerance,
    ))
}

fn compute_knn_probabilities_core<S, I, FRow, FPerp>(
    n: usize,
    mut row_iter: FRow,
    mut row_perplexity: FPerp,
    power: S,
    sigma_iters: usize,
    sigma_tolerance: S,
) -> SparseP<S>
where
    S: DrawingValue + Float + Default,
    I: IntoIterator<Item = (usize, S)>,
    FRow: FnMut(usize) -> I,
    FPerp: FnMut(usize) -> S,
{
    if n < 2 {
        return vec![Vec::new(); n];
    }

    let two = S::from_f32(2.0).unwrap();
    let min_sum = S::from_f32(1e-12).unwrap();
    let perp_margin = S::from_f32(0.01).unwrap();

    let mut row_probs: Vec<Vec<(usize, S)>> = vec![Vec::new(); n];
    let mut seen_pos: HashMap<usize, usize> = HashMap::new();
    let mut collected: Vec<(usize, S)> = Vec::new();
    let mut dp_values: Vec<S> = Vec::new();
    let mut weights: Vec<S> = Vec::new();

    for (i, row_prob) in row_probs.iter_mut().enumerate() {
        seen_pos.clear();
        collected.clear();

        for (j, d) in row_iter(i) {
            if j == i || j >= n || !d.is_finite() {
                continue;
            }
            let d_nonneg = if d < S::zero() { S::zero() } else { d };
            if let Some(&pos) = seen_pos.get(&j) {
                if d_nonneg < collected[pos].1 {
                    collected[pos].1 = d_nonneg;
                }
            } else {
                seen_pos.insert(j, collected.len());
                collected.push((j, d_nonneg));
            }
        }

        let m = collected.len();
        if m == 0 {
            continue;
        }
        if m == 1 {
            row_prob.push((collected[0].0, S::one()));
            continue;
        }

        // Normalize distances before applying power and shifting by min_dp so binary search
        // operates over [0, 1] without overflow/underflow across arbitrary distance scales.
        let mut max_d = S::zero();
        for &(_, d) in &collected {
            if d > max_d {
                max_d = d;
            }
        }

        if max_d <= S::zero() {
            let p_uni = S::one() / S::from_usize(m).unwrap();
            for &(j, _) in &collected {
                row_prob.push((j, p_uni));
            }
            row_prob.sort_unstable_by_key(|&(j, _)| j);
            continue;
        }

        dp_values.clear();
        let mut min_dp = S::infinity();
        let mut max_dp = S::neg_infinity();
        for &(_, d) in &collected {
            let dp = (d / max_d).powf(power);
            let dp_safe = if dp.is_finite() { dp } else { S::one() };
            if dp_safe < min_dp {
                min_dp = dp_safe;
            }
            if dp_safe > max_dp {
                max_dp = dp_safe;
            }
            dp_values.push(dp_safe);
        }

        let range = max_dp - min_dp;
        if range <= S::zero() {
            let p_uni = S::one() / S::from_usize(m).unwrap();
            for &(j, _) in &collected {
                row_prob.push((j, p_uni));
            }
            row_prob.sort_unstable_by_key(|&(j, _)| j);
            continue;
        }

        for dp in &mut dp_values {
            *dp = (*dp - min_dp) / range;
        }

        let m_s = S::from_usize(m).unwrap();
        let max_perp = (m_s - perp_margin).max(S::one());
        let perp = row_perplexity(i);
        // Guard against NaN because Float::min returns the non-NaN argument, which would
        // silently turn NaN into max_perp on unvalidated scalar paths.
        let perp_safe = if perp.is_finite() { perp } else { S::one() };
        let perplexity_clamped = perp_safe.min(max_perp).max(S::one());
        let target_entropy = perplexity_clamped.ln();

        let mut beta = S::one();
        let mut beta_min = S::zero();
        let mut beta_max: Option<S> = None;

        weights.resize(m, S::zero());
        for _ in 0..sigma_iters {
            let mut sum_w = S::zero();
            for (idx, &dp) in dp_values.iter().enumerate() {
                let w = (-beta * dp).exp();
                weights[idx] = w;
                sum_w += w;
            }
            sum_w = sum_w.max(min_sum);

            let mut sum_dp_p = S::zero();
            for (idx, &dp) in dp_values.iter().enumerate() {
                sum_dp_p += (weights[idx] / sum_w) * dp;
            }

            let h = beta * sum_dp_p + sum_w.ln();
            let h_diff = h - target_entropy;

            if h_diff.abs() < sigma_tolerance {
                break;
            }

            if h_diff > S::zero() {
                beta_min = beta;
                match beta_max {
                    None => {
                        let next = beta * two;
                        if !next.is_finite() {
                            break;
                        }
                        beta = next;
                    }
                    Some(bmax) => {
                        beta = (beta_min + bmax) / two;
                    }
                }
            } else {
                beta_max = Some(beta);
                beta = (beta_min + beta) / two;
            }
        }

        // Compute final conditional probabilities p_{j|i}
        let mut sum_w = S::zero();
        for (idx, &dp) in dp_values.iter().enumerate() {
            let w = (-beta * dp).exp();
            weights[idx] = w;
            sum_w += w;
        }
        sum_w = sum_w.max(min_sum);

        for (idx, &(j, _)) in collected.iter().enumerate() {
            row_prob.push((j, weights[idx] / sum_w));
        }
        row_prob.sort_unstable_by_key(|&(j, _)| j);
    }

    // Symmetrize joint probabilities: p_{ij} = p_{ji} = (p_{j|i} + p_{i|j}) / (2 * N)
    let two_n = S::from_usize(2 * n).unwrap();
    let mut sparse_p: SparseP<S> = vec![Vec::new(); n];

    for i in 0..n {
        for &(j, p_ji) in &row_probs[i] {
            match row_probs[j].binary_search_by_key(&i, |&(col, _)| col) {
                Ok(pos) => {
                    if i < j {
                        let p_sym = (p_ji + row_probs[j][pos].1) / two_n;
                        if p_sym > S::zero() {
                            sparse_p[i].push((j, p_sym));
                            sparse_p[j].push((i, p_sym));
                        }
                    }
                }
                Err(_) => {
                    let p_sym = p_ji / two_n;
                    if p_sym > S::zero() {
                        sparse_p[i].push((j, p_sym));
                        sparse_p[j].push((i, p_sym));
                    }
                }
            }
        }
    }

    for row in &mut sparse_p {
        row.sort_unstable_by_key(|&(j, _)| j);
    }

    sparse_p
}
