mod bfs;
mod knn;

pub use bfs::compute_partial_bfs_probabilities;
pub use knn::{
    compute_knn_probabilities, compute_knn_probabilities_from_csr,
    compute_knn_probabilities_with_params, compute_knn_probabilities_with_perplexities,
};

use num_traits::Float;
use petgraph_drawing::{Drawing, DrawingValue, MetricEuclidean2d};

/// Sparse symmetric joint probability matrix P stored as adjacency rows `(neighbor_idx, p_ij)`.
pub type SparseP<S> = Vec<Vec<(usize, S)>>;

pub(crate) fn validate_finite_drawing<D, S>(drawing: &D) -> Result<(), String>
where
    D: Drawing<Item = MetricEuclidean2d<S>>,
    S: Float,
{
    for i in 0..drawing.len() {
        let entry = drawing.raw_entry(i);
        if !entry.0.is_finite() || !entry.1.is_finite() {
            return Err(format!(
                "drawing coordinate at node index {i} must be finite"
            ));
        }
    }
    Ok(())
}

/// Validates that `perplexities` has length `n` and contains only finite, strictly positive values.
pub(super) fn validate_perplexities<S: Float>(n: usize, perplexities: &[S]) -> Result<(), String> {
    if perplexities.len() != n {
        return Err(format!(
            "perplexities length ({}) must match node count ({n})",
            perplexities.len()
        ));
    }
    for (i, &p) in perplexities.iter().enumerate() {
        if !p.is_finite() || p <= S::zero() {
            return Err(format!(
                "perplexities[{i}] must be finite and strictly positive"
            ));
        }
    }
    Ok(())
}

/// Validates CSR k-NN arrays (`indptr`, `indices`, `distances`) for `n` nodes without allocating.
pub(super) fn validate_csr_knn<S: Float>(
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

/// Converts validated CSR k-NN arrays into row-wise neighbor lists `Vec<Vec<(usize, S)>>`.
pub fn knn_from_csr<S>(
    n: usize,
    indptr: &[usize],
    indices: &[usize],
    distances: &[S],
) -> Result<Vec<Vec<(usize, S)>>, String>
where
    S: DrawingValue + Float,
{
    validate_csr_knn(n, indptr, indices, distances)?;
    let mut knn = Vec::with_capacity(n);
    for i in 0..n {
        let start = indptr[i];
        let end = indptr[i + 1];
        let mut row = Vec::with_capacity(end - start);
        for (&j, &d) in indices[start..end].iter().zip(distances[start..end].iter()) {
            row.push((j, d));
        }
        knn.push(row);
    }
    Ok(knn)
}
