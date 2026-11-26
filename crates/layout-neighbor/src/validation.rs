use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar};
use egraph_neighborhood::{
    DogVaradhanLandmarks, Sparse12HopPairs, SparseProbabilities, extract_sparse_1hop_2hop_pairs,
};

const DEFAULT_EPS_DIST: f64 = 1e-6;

pub(crate) fn validate_sparse_probabilities(probs: &SparseProbabilities) -> CoreResult<()> {
    let n = probs.n_nodes;
    if u32::try_from(n).is_err() {
        return Err(CoreError::invalid_csr("n_nodes exceeds u32::MAX"));
    }
    if probs.indptr.len() != n + 1 {
        return Err(CoreError::dimension_mismatch(
            n + 1,
            probs.indptr.len(),
            "SparseProbabilities::indptr",
        ));
    }
    if probs.indptr[0] != 0 {
        return Err(CoreError::invalid_csr(format!(
            "SparseProbabilities indptr[0] must be 0, got {}",
            probs.indptr[0]
        )));
    }
    for row in 0..n {
        if probs.indptr[row + 1] < probs.indptr[row] {
            return Err(CoreError::invalid_csr(format!(
                "SparseProbabilities indptr must be non-decreasing at row {row}: {} > {}",
                probs.indptr[row],
                probs.indptr[row + 1]
            )));
        }
    }
    let nnz = probs.indptr[n] as usize;
    if probs.indices.len() != nnz {
        return Err(CoreError::dimension_mismatch(
            nnz,
            probs.indices.len(),
            "SparseProbabilities::indices",
        ));
    }
    if probs.values.len() != nnz {
        return Err(CoreError::dimension_mismatch(
            nnz,
            probs.values.len(),
            "SparseProbabilities::values",
        ));
    }
    for (&idx, &val) in probs.indices.iter().zip(probs.values.iter()) {
        if (idx as usize) >= n {
            return Err(CoreError::index_out_of_bounds(
                idx as usize,
                n,
                "SparseProbabilities::indices",
            ));
        }
        if !val.is_finite() {
            return Err(CoreError::non_finite("SparseProbabilities::values"));
        }
        if val < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "SparseProbabilities values must be >= 0.0, got {val}"
            )));
        }
    }
    Ok(())
}

pub(crate) fn compute_12hop_alt_distances<W: Scalar>(
    graph: &CsrGraph<W>,
    landmarks: &DogVaradhanLandmarks,
    max_2hop_per_node: usize,
    seed: u64,
    mismatch_context: &'static str,
) -> CoreResult<(Sparse12HopPairs, Vec<f64>, Vec<f64>)> {
    let n = graph.num_nodes();
    if u32::try_from(n).is_err() {
        return Err(CoreError::invalid_csr("n_nodes exceeds u32::MAX"));
    }
    if landmarks.num_nodes() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            landmarks.num_nodes(),
            mismatch_context,
        ));
    }
    let k_lm = landmarks.num_landmarks();
    if n > 0 && k_lm == 0 {
        return Err(CoreError::invalid_input(
            "landmarks must have at least 1 landmark when num_nodes > 0",
        ));
    }
    let expected = n
        .checked_mul(k_lm)
        .ok_or_else(|| CoreError::invalid_input("landmark table size overflow"))?;
    if landmarks.d_dog().len() != expected {
        return Err(CoreError::dimension_mismatch(
            expected,
            landmarks.d_dog().len(),
            "DogVaradhanLandmarks::d_dog",
        ));
    }
    if landmarks.d_dog().iter().any(|v| !v.is_finite()) {
        return Err(CoreError::non_finite("DogVaradhanLandmarks::d_dog"));
    }
    if let Some(dc) = landmarks.d_coarse() {
        if dc.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                dc.len(),
                "DogVaradhanLandmarks::d_coarse",
            ));
        }
        if dc.iter().any(|v| !v.is_finite()) {
            return Err(CoreError::non_finite("DogVaradhanLandmarks::d_coarse"));
        }
    }
    for &lm in landmarks.landmarks() {
        if (lm as usize) >= n {
            return Err(CoreError::index_out_of_bounds(
                lm as usize,
                n,
                "DogVaradhanLandmarks::landmarks",
            ));
        }
    }

    let pairs = extract_sparse_1hop_2hop_pairs(graph, max_2hop_per_node, seed)?;
    let mut d1 = landmarks.alt_pair_distances(&pairs.u1, &pairs.v1)?;
    let mut d2 = landmarks.alt_pair_distances(&pairs.u2, &pairs.v2)?;
    for d in &mut d1 {
        *d = d.max(DEFAULT_EPS_DIST);
    }
    for d in &mut d2 {
        *d = d.max(DEFAULT_EPS_DIST);
    }
    Ok((pairs, d1, d2))
}
