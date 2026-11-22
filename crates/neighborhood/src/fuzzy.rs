//! UMAP / GuMap fuzzy simplicial set probability computation and `(a, b)` curve fitting
//! strictly in `f64` without per-row `HashMap` allocations.

use egraph_core::{CoreError, CoreResult, VisitStamp};

use crate::knn::KnnGraph;
use crate::perplexity::SparseProbabilities;

/// Computes the UMAP / GuMap fuzzy simplicial set probabilities from a [`KnnGraph`].
///
/// Uses `VisitStamp` deduplication (retaining minimum non-negative distance and stripping
/// self-loops), `f64` $\sigma_i$ binary search targeting $\sum_j \exp(-(d_{ij} - \rho_i) / \sigma_i) = \log_2(m_i)$,
/// fuzzy union symmetrization $p_{ij} = p_{j|i} + p_{i|j} - p_{j|i} p_{i|j}$, and `1e-7` edge
/// thresholding into canonical sorted symmetric CSR [`SparseProbabilities`].
pub fn compute_fuzzy_simplicial_set(
    knn: &KnnGraph,
    sigma_iters: usize,
    sigma_tolerance: f64,
) -> CoreResult<SparseProbabilities> {
    if sigma_iters == 0 {
        return Err(CoreError::invalid_input("sigma_iters must be >= 1"));
    }
    if !sigma_tolerance.is_finite() {
        return Err(CoreError::non_finite(
            "compute_fuzzy_simplicial_set::sigma_tolerance",
        ));
    }
    if sigma_tolerance <= 0.0 {
        return Err(CoreError::invalid_input(format!(
            "sigma_tolerance must be > 0, got {sigma_tolerance}"
        )));
    }

    let n = knn.num_nodes();
    if n < 2 {
        return Ok(SparseProbabilities::empty(n));
    }

    const THRESHOLD: f64 = 1e-7;

    let mut seen = VisitStamp::new(n);
    let mut seen_pos = vec![0usize; n];
    let mut collected: Vec<(u32, f64)> = Vec::new();
    let mut directed_pairs: Vec<(u32, u32, f64)> = Vec::with_capacity(knn.nnz());

    for i in 0..n {
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

        let m_i = collected.len();
        if m_i == 0 {
            continue;
        }

        let mut rho_i = collected[0].1;
        let mut max_d = collected[0].1;
        for &(_, d) in &collected[1..] {
            if d < rho_i {
                rho_i = d;
            }
            if d > max_d {
                max_d = d;
            }
        }

        let i_u32 = i as u32;
        if m_i == 1 || max_d <= rho_i {
            for &(j, _) in &collected {
                let (u, v) = if i_u32 < j { (i_u32, j) } else { (j, i_u32) };
                directed_pairs.push((u, v, 1.0));
            }
            continue;
        }

        let target_sum = (m_i as f64).log2().max(1e-5);
        let mut sigma_min = 0.0_f64;
        let mut sigma_max = f64::INFINITY;
        let mut sigma_i = 1.0_f64;

        for _ in 0..sigma_iters {
            let mut p_sum = 0.0_f64;
            for &(_, d_ij) in &collected {
                let dist_diff = (d_ij - rho_i).max(0.0);
                p_sum += (-dist_diff / sigma_i).exp();
            }

            let diff = p_sum - target_sum;
            if diff.abs() < sigma_tolerance {
                break;
            }

            if diff > 0.0 {
                sigma_max = sigma_i;
                sigma_i = 0.5 * (sigma_min + sigma_max);
            } else {
                sigma_min = sigma_i;
                if sigma_max.is_infinite() {
                    let next = sigma_i * 2.0;
                    if !next.is_finite() {
                        break;
                    }
                    sigma_i = next;
                } else {
                    sigma_i = 0.5 * (sigma_min + sigma_max);
                }
            }
        }

        for &(j, d_ij) in &collected {
            let dist_diff = (d_ij - rho_i).max(0.0);
            let p_val = (-dist_diff / sigma_i).exp();
            if p_val > 0.0 {
                let (u, v) = if i_u32 < j { (i_u32, j) } else { (j, i_u32) };
                directed_pairs.push((u, v, p_val));
            }
        }
    }

    if directed_pairs.is_empty() {
        return Ok(SparseProbabilities::empty(n));
    }

    directed_pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let mut merged: Vec<(u32, u32, f64)> = Vec::with_capacity(directed_pairs.len());
    let mut deg = vec![0usize; n];

    let mut pos = 0usize;
    while pos < directed_pairs.len() {
        let (u, v, mut w) = directed_pairs[pos];
        pos += 1;
        while pos < directed_pairs.len() && directed_pairs[pos].0 == u && directed_pairs[pos].1 == v
        {
            let p = directed_pairs[pos].2;
            w = w + p - w * p;
            pos += 1;
        }
        if w > THRESHOLD {
            merged.push((u, v, w));
            deg[u as usize] += 1;
            deg[v as usize] += 1;
        }
    }

    if merged.is_empty() {
        return Ok(SparseProbabilities::empty(n));
    }

    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0usize);
    let mut total = 0usize;
    for d in &deg {
        total += *d;
        indptr.push(total);
    }
    let mut cursor = indptr[..n].to_vec();
    let mut indices = vec![0u32; total];
    let mut values = vec![0.0_f64; total];

    for (u, v, w) in merged {
        let pu = cursor[u as usize];
        indices[pu] = v;
        values[pu] = w;
        cursor[u as usize] += 1;

        let pv = cursor[v as usize];
        indices[pv] = u;
        values[pv] = w;
        cursor[v as usize] += 1;
    }

    Ok(SparseProbabilities {
        n_nodes: n,
        indptr,
        indices,
        values,
    })
}

/// Fits low-dimensional UMAP parameters `(a, b)` from `min_dist` and `spread` strictly in `f64`
/// via Levenberg-Marquardt non-linear least squares.
pub fn find_ab_params(min_dist: f64, spread: f64) -> CoreResult<(f64, f64)> {
    if !min_dist.is_finite() {
        return Err(CoreError::non_finite("find_ab_params::min_dist"));
    }
    if min_dist < 0.0 {
        return Err(CoreError::invalid_input(format!(
            "min_dist must be >= 0, got {min_dist}"
        )));
    }
    if !spread.is_finite() {
        return Err(CoreError::non_finite("find_ab_params::spread"));
    }
    if spread <= 0.0 {
        return Err(CoreError::invalid_input(format!(
            "spread must be > 0, got {spread}"
        )));
    }

    const NUM_SAMPLES: usize = 300;
    let max_dist = spread * 3.0;
    let step = max_dist / (NUM_SAMPLES as f64);

    let mut d_vals = Vec::with_capacity(NUM_SAMPLES);
    let mut target_vals = Vec::with_capacity(NUM_SAMPLES);
    for i in 1..=NUM_SAMPLES {
        let d = (i as f64) * step;
        let target = if d <= min_dist {
            1.0
        } else {
            (-(d - min_dist) / spread).exp()
        };
        d_vals.push(d);
        target_vals.push(target);
    }

    let mut a = 1.0_f64;
    let mut b = 1.0_f64;
    let mut lambda = 1e-3_f64;

    for _ in 0..100 {
        let mut jtj00 = 0.0_f64;
        let mut jtj01 = 0.0_f64;
        let mut jtj11 = 0.0_f64;
        let mut g0 = 0.0_f64;
        let mut g1 = 0.0_f64;
        let mut total_error = 0.0_f64;

        for i in 0..NUM_SAMPLES {
            let d = d_vals[i];
            let target = target_vals[i];

            let d_2b = d.powf(2.0 * b);
            let denom = 1.0 + a * d_2b;
            let denom_sq = denom * denom;
            let f_val = 1.0 / denom;
            let res = f_val - target;
            total_error += res * res;

            let j0 = -d_2b / denom_sq;
            let j1 = -(2.0 * a * d_2b * d.ln()) / denom_sq;

            jtj00 += j0 * j0;
            jtj01 += j0 * j1;
            jtj11 += j1 * j1;

            g0 -= j0 * res;
            g1 -= j1 * res;
        }

        let a00 = jtj00 + lambda;
        let a01 = jtj01;
        let a11 = jtj11 + lambda;

        let det = a00 * a11 - a01 * a01;
        if det.abs() < 1e-12 {
            lambda *= 10.0;
            continue;
        }

        let delta_a = (a11 * g0 - a01 * g1) / det;
        let delta_b = (-a01 * g0 + a00 * g1) / det;

        let new_a = (a + delta_a).max(1e-4);
        let new_b = (b + delta_b).max(1e-4);

        let mut new_error = 0.0_f64;
        for i in 0..NUM_SAMPLES {
            let d = d_vals[i];
            let target = target_vals[i];
            let d_2b = d.powf(2.0 * new_b);
            let f_val = 1.0 / (1.0 + new_a * d_2b);
            let res = f_val - target;
            new_error += res * res;
        }

        if new_error < total_error {
            a = new_a;
            b = new_b;
            lambda = (lambda * 0.8).max(1e-7);
            if delta_a.abs() < 1e-6 && delta_b.abs() < 1e-6 {
                break;
            }
        } else {
            lambda *= 2.0;
        }
    }

    Ok((a, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuzzy_simplicial_set_symmetry_dedup_and_determinism() {
        // Asymmetric k-NN with duplicate entry and self-loop on row 0.
        let knn = KnnGraph::new_validated(
            4,
            vec![0, 5, 8, 10, 11],
            vec![0, 1, 1, 2, 3, 0, 2, 3, 0, 1, 2],
            vec![0.0, 2.0, 0.2, 0.8, 1.5, 0.2, 0.5, 1.0, 0.8, 0.5, 1.0],
        )
        .unwrap();

        let p1 = compute_fuzzy_simplicial_set(&knn, 64, 1e-5).unwrap();
        let p2 = compute_fuzzy_simplicial_set(&knn, 64, 1e-5).unwrap();
        assert_eq!(p1, p2);

        for i in 0..4 {
            let (cols, vals) = p1.row(i);
            assert!(cols.windows(2).all(|w| w[0] < w[1]));
            assert!(!cols.contains(&(i as u32)));
            for (&j, &w_ij) in cols.iter().zip(vals.iter()) {
                assert!(w_ij > 1e-7 && w_ij <= 1.0);
                let (rcols, rvals) = p1.row(j as usize);
                let pos = rcols.binary_search(&(i as u32)).unwrap();
                assert_eq!(w_ij.to_bits(), rvals[pos].to_bits());
            }
        }

        // Closest neighbor gets weight 1.0 (since exp(-(rho_i - rho_i)/sigma_i) = 1.0).
        let (r0_cols, r0_vals) = p1.row(0);
        assert_eq!(r0_cols[0], 1);
        assert!((r0_vals[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_find_ab_params_matches_umap_reference_and_rejects_bad_inputs() {
        let (a, b) = find_ab_params(0.1, 1.0).unwrap();
        assert!((a - 1.577).abs() < 0.01, "expected a ~ 1.577, got {a}");
        assert!((b - 0.895).abs() < 0.01, "expected b ~ 0.895, got {b}");

        assert!(find_ab_params(-0.1, 1.0).is_err());
        assert!(find_ab_params(f64::NAN, 1.0).is_err());
        assert!(find_ab_params(0.1, 0.0).is_err());
        assert!(find_ab_params(0.1, f64::INFINITY).is_err());
    }

    #[test]
    fn test_fuzzy_edge_cases() {
        let empty = KnnGraph::empty(0).unwrap();
        let pe = compute_fuzzy_simplicial_set(&empty, 64, 1e-5).unwrap();
        assert_eq!(pe.num_nodes(), 0);
        assert_eq!(pe.nnz(), 0);

        let single = KnnGraph::empty(1).unwrap();
        let ps = compute_fuzzy_simplicial_set(&single, 64, 1e-5).unwrap();
        assert_eq!(ps.num_nodes(), 1);
        assert_eq!(ps.nnz(), 0);

        assert!(compute_fuzzy_simplicial_set(&single, 0, 1e-5).is_err());
        assert!(compute_fuzzy_simplicial_set(&single, 64, f64::NAN).is_err());
    }
}
