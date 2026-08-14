use num_traits::Float;
use petgraph_drawing::{DrawingIndex, DrawingValue};
use petgraph_linalg_kernel::Distance;

/// Sparse directed probability edge
#[derive(Debug, Clone)]
pub struct SparseEdge<S> {
    pub source: usize,
    pub target: usize,
    pub weight: S,
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
    let target_sum = S::from_f64((k_eff as f64).log2().max(1e-5)).unwrap();

    // directed_p[i] stores (j, p_j|i)
    let mut directed_p: Vec<Vec<(usize, S)>> = vec![Vec::with_capacity(k_eff); n];

    for i in 0..n {
        // Find all pairwise distances from i
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

        // Sort by ascending distance and pick top k_eff neighbors
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let knn: Vec<(usize, S)> = dists.into_iter().take(k_eff).collect();

        if knn.is_empty() {
            continue;
        }

        // rho_i = distance to the nearest neighbor
        let rho_i = knn[0].1;

        // Binary search for sigma_i such that sum exp(-max(0, d_ij - rho_i) / sigma_i) = log2(k)
        let mut sigma_min = S::zero();
        let mut sigma_max = S::from_f64(f64::INFINITY).unwrap();
        let mut sigma_i = S::one();

        for _ in 0..sigma_iters {
            let mut p_sum = S::zero();
            for &(_, d_ij) in &knn {
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
                sigma_i = (sigma_min + sigma_max) * S::from_f32(0.5).unwrap();
            } else {
                sigma_min = sigma_i;
                if sigma_max.is_infinite() {
                    sigma_i *= S::from_f32(2.0).unwrap();
                } else {
                    sigma_i = (sigma_min + sigma_max) * S::from_f32(0.5).unwrap();
                }
            }
        }

        // Compute directed probabilities p_{j|i}
        for &(j, d_ij) in &knn {
            let dist_diff = (d_ij - rho_i).max(S::zero());
            let p_val = (-dist_diff / sigma_i).exp();
            if p_val > S::zero() {
                directed_p[i].push((j, p_val));
            }
        }
    }

    // Step 4: Fuzzy Set Union Symmetrization: p_{ij} = p_{j|i} + p_{i|j} - p_{j|i} * p_{i|j}
    // We combine all directed probabilities into a symmetrical sparse edge list
    use std::collections::HashMap;
    let mut edge_map: HashMap<(usize, usize), S> = HashMap::new();

    for (i, neighbors) in directed_p.iter().enumerate() {
        for &(j, p_ji) in neighbors {
            let key = if i < j { (i, j) } else { (j, i) };
            // Retrieve reciprocal directed probability if already partially stored or calculate
            edge_map
                .entry(key)
                .and_modify(|existing| {
                    // existing is p_{i|j} (or p_{j|i}), combine via fuzzy union
                    *existing = *existing + p_ji - (*existing * p_ji);
                })
                .or_insert(p_ji);
        }
    }

    let mut edges = Vec::with_capacity(edge_map.len() * 2);
    for ((u, v), weight) in edge_map {
        if weight > S::from_f32(1e-7).unwrap() {
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
    }

    edges
}
