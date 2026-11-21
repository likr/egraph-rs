use num_traits::Float;
use petgraph_drawing::{DrawingIndex, DrawingValue};
use petgraph_linalg_kernel::Distance;
use std::collections::HashMap;

/// Sparse directed probability edge
#[derive(Debug, Clone)]
pub struct SparseEdge<S> {
    pub source: usize,
    pub target: usize,
    pub weight: S,
}

/// Validates CSR k-NN arrays (`indptr`, `indices`, `distances`) for `n` nodes without allocating.
pub fn validate_csr_knn<S: Float>(
    n: usize,
    indptr: &[usize],
    indices: &[usize],
    distances: &[S],
) -> Result<(), String> {
    if indptr.len() != n + 1 {
        return Err(format!(
            "indptr length ({}) must equal node count + 1 ({})",
            indptr.len(),
            n + 1
        ));
    }
    if indptr[0] != 0 {
        return Err(format!("indptr[0] must be 0, got {}", indptr[0]));
    }
    for i in 0..n {
        if indptr[i + 1] < indptr[i] {
            return Err(format!(
                "indptr must be non-decreasing: indptr[{i}]={} > indptr[{}]={}",
                indptr[i],
                i + 1,
                indptr[i + 1]
            ));
        }
    }
    let nnz = indptr[n];
    if indices.len() != nnz {
        return Err(format!(
            "indices length ({}) must equal indptr[n] ({nnz})",
            indices.len()
        ));
    }
    if distances.len() != nnz {
        return Err(format!(
            "distances length ({}) must equal indptr[n] ({nnz})",
            distances.len()
        ));
    }
    for (k, (&idx, &dist)) in indices.iter().zip(distances.iter()).enumerate() {
        if idx >= n {
            return Err(format!(
                "indices[{k}] = {idx} is out of range for {n} nodes"
            ));
        }
        if !dist.is_finite() || dist < S::zero() {
            return Err(format!("distances[{k}] must be finite and non-negative"));
        }
    }
    Ok(())
}

fn compute_directed_row_probs<S>(
    collected: &[(usize, S)],
    sigma_iters: usize,
    sigma_tolerance: S,
    out: &mut Vec<(usize, S)>,
) where
    S: DrawingValue + Float + std::iter::Sum + std::ops::AddAssign + Default,
{
    let m_i = collected.len();
    if m_i == 0 {
        return;
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

    // When only one neighbor exists or all neighbors share the minimum distance,
    // exp(-(d - rho_i) / sigma) == 1.0 identically, avoiding 0/0 NaN under sigma underflow.
    if m_i == 1 || max_d <= rho_i {
        for &(j, _) in collected {
            out.push((j, S::one()));
        }
        return;
    }

    let target_sum = S::from_f64((m_i as f64).log2().max(1e-5)).unwrap();
    let half = S::from_f32(0.5).unwrap();
    let two = S::from_f32(2.0).unwrap();

    let mut sigma_min = S::zero();
    let mut sigma_max = S::from_f64(f64::INFINITY).unwrap();
    let mut sigma_i = S::one();

    for _ in 0..sigma_iters {
        let mut p_sum = S::zero();
        for &(_, d_ij) in collected {
            let dist_diff = (d_ij - rho_i).max(S::zero());
            let p_val = (-dist_diff / sigma_i).exp();
            p_sum += p_val;
        }

        let diff = p_sum - target_sum;
        if diff.abs() < sigma_tolerance {
            break;
        }

        if diff > S::zero() {
            sigma_max = sigma_i;
            sigma_i = (sigma_min + sigma_max) * half;
        } else {
            sigma_min = sigma_i;
            if sigma_max.is_infinite() {
                let next = sigma_i * two;
                if !next.is_finite() {
                    break;
                }
                sigma_i = next;
            } else {
                sigma_i = (sigma_min + sigma_max) * half;
            }
        }
    }

    for &(j, d_ij) in collected {
        let dist_diff = (d_ij - rho_i).max(S::zero());
        let p_val = (-dist_diff / sigma_i).exp();
        if p_val > S::zero() {
            out.push((j, p_val));
        }
    }
}

fn symmetrize_directed_probs<S>(directed_p: &[Vec<(usize, S)>]) -> Vec<SparseEdge<S>>
where
    S: DrawingValue + Float + std::iter::Sum + std::ops::AddAssign + Default,
{
    // Fuzzy set union: p_{ij} = p_{j|i} + p_{i|j} - p_{j|i} * p_{i|j}
    let mut edge_map: HashMap<(usize, usize), S> = HashMap::new();
    for (i, neighbors) in directed_p.iter().enumerate() {
        for &(j, p_ji) in neighbors {
            let key = if i < j { (i, j) } else { (j, i) };
            edge_map
                .entry(key)
                .and_modify(|existing| {
                    *existing = *existing + p_ji - (*existing * p_ji);
                })
                .or_insert(p_ji);
        }
    }

    let threshold = S::from_f32(1e-7).unwrap();
    let mut sorted_edges: Vec<((usize, usize), S)> = edge_map
        .into_iter()
        .filter(|&(_, weight)| weight > threshold)
        .collect();
    // Canonical (u, v) ordering guarantees deterministic StdRng SGD streams and run/run_with_knn parity.
    sorted_edges.sort_unstable_by_key(|&(uv, _)| uv);

    let mut edges = Vec::with_capacity(sorted_edges.len() * 2);
    for ((u, v), weight) in sorted_edges {
        edges.push(SparseEdge {
            source: u,
            target: v,
            weight,
        });
        edges.push(SparseEdge {
            source: v,
            target: u,
            weight,
        });
    }
    edges
}

fn compute_fuzzy_simplicial_set_from_rows<S, I, FRow>(
    n: usize,
    mut row_iter: FRow,
    sigma_iters: usize,
    sigma_tolerance: S,
) -> Vec<SparseEdge<S>>
where
    S: DrawingValue + Float + std::iter::Sum + std::ops::AddAssign + Default,
    I: IntoIterator<Item = (usize, S)>,
    FRow: FnMut(usize) -> I,
{
    if n < 2 {
        return Vec::new();
    }

    let mut directed_p: Vec<Vec<(usize, S)>> = vec![Vec::new(); n];
    let mut seen_pos: HashMap<usize, usize> = HashMap::new();
    let mut collected: Vec<(usize, S)> = Vec::new();

    for (i, out_row) in directed_p.iter_mut().enumerate() {
        seen_pos.clear();
        collected.clear();

        // Filter self-loops and deduplicate by minimum distance BEFORE computing rho_i
        // so that self-distance 0 does not collapse rho_i or skew fuzzy union.
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

        compute_directed_row_probs(&collected, sigma_iters, sigma_tolerance, out_row);
    }

    symmetrize_directed_probs(&directed_p)
}

/// Computes the fuzzy simplicial set from row-wise precomputed k-NN lists `(neighbor_idx, distance)`.
///
/// Filters self-loops (`j == i`), out-of-range indices (`j >= n`), and non-finite distances,
/// clamps negative distances to zero, and deduplicates duplicate neighbors by minimum distance.
pub fn compute_fuzzy_simplicial_set_from_knn<S>(
    n: usize,
    knn: &[Vec<(usize, S)>],
    sigma_iters: usize,
    sigma_tolerance: S,
) -> Vec<SparseEdge<S>>
where
    S: DrawingValue + Float + std::iter::Sum + std::ops::AddAssign + Default,
{
    compute_fuzzy_simplicial_set_from_rows(
        n,
        |i| knn.get(i).into_iter().flat_map(|row| row.iter().copied()),
        sigma_iters,
        sigma_tolerance,
    )
}

/// Validating variant of [`compute_fuzzy_simplicial_set_from_knn`] requiring `knn.len() == n`,
/// `j < n`, and finite non-negative distances (self-loops and duplicates are still handled).
pub fn try_compute_fuzzy_simplicial_set_from_knn<S>(
    n: usize,
    knn: &[Vec<(usize, S)>],
    sigma_iters: usize,
    sigma_tolerance: S,
) -> Result<Vec<SparseEdge<S>>, String>
where
    S: DrawingValue + Float + std::iter::Sum + std::ops::AddAssign + Default,
{
    if knn.len() != n {
        return Err(format!(
            "knn row count ({}) must match node count ({n})",
            knn.len()
        ));
    }
    for (i, row) in knn.iter().enumerate() {
        for (pos, &(j, d)) in row.iter().enumerate() {
            if j >= n {
                return Err(format!(
                    "knn[{i}][{pos}] neighbor index {j} is out of range for {n} nodes"
                ));
            }
            if !d.is_finite() || d < S::zero() {
                return Err(format!(
                    "knn[{i}][{pos}] distance must be finite and non-negative"
                ));
            }
        }
    }
    Ok(compute_fuzzy_simplicial_set_from_rows(
        n,
        |i| knn[i].iter().copied(),
        sigma_iters,
        sigma_tolerance,
    ))
}

/// Computes the fuzzy simplicial set directly from validated CSR k-NN arrays `(indptr, indices, distances)`.
pub fn compute_fuzzy_simplicial_set_from_csr<S>(
    n: usize,
    indptr: &[usize],
    indices: &[usize],
    distances: &[S],
    sigma_iters: usize,
    sigma_tolerance: S,
) -> Result<Vec<SparseEdge<S>>, String>
where
    S: DrawingValue + Float + std::iter::Sum + std::ops::AddAssign + Default,
{
    validate_csr_knn(n, indptr, indices, distances)?;
    Ok(compute_fuzzy_simplicial_set_from_rows(
        n,
        |i| {
            let start = indptr[i];
            let end = indptr[i + 1];
            indices[start..end]
                .iter()
                .copied()
                .zip(distances[start..end].iter().copied())
        },
        sigma_iters,
        sigma_tolerance,
    ))
}

/// Computes the fuzzy simplicial set (symmetrized high-dimensional joint probabilities P)
/// from a distance matrix according to the GUMAP / UMAP specification.
#[allow(clippy::needless_range_loop)]
pub fn compute_fuzzy_simplicial_set<N, S, D>(
    distance_matrix: &D,
    n: usize,
    k: usize,
    sigma_iters: usize,
    sigma_tolerance: S,
) -> Vec<SparseEdge<S>>
where
    N: Copy + Eq + std::hash::Hash + DrawingIndex,
    S: DrawingValue + Float + std::iter::Sum + std::ops::AddAssign + Default,
    D: Distance<N, S> + ?Sized,
{
    if n < 2 {
        return Vec::new();
    }

    let k_eff = (k).min(n - 1).max(1);
    let mut directed_p: Vec<Vec<(usize, S)>> = vec![Vec::with_capacity(k_eff); n];

    for i in 0..n {
        let mut dists: Vec<(usize, S)> = Vec::with_capacity(n - 1);
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = distance_matrix.get_by_index(i, j);
            if d.is_finite() && d >= S::zero() {
                dists.push((j, d));
            } else {
                dists.push((j, S::from_f32(1e6).unwrap()));
            }
        }

        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let knn: Vec<(usize, S)> = dists.into_iter().take(k_eff).collect();
        compute_directed_row_probs(&knn, sigma_iters, sigma_tolerance, &mut directed_p[i]);
    }

    symmetrize_directed_probs(&directed_p)
}
