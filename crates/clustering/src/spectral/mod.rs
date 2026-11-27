//! Pure-Rust Ng-Jordan-Weiss spectral clustering on undirected `CsrGraph` instances.

use std::collections::VecDeque;

use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar, VisitStamp};
use egraph_linalg::smallest_sym_laplacian_eigenpairs;

use crate::kmeans::{KMeansConfig, kmeans_plus_plus};

#[cfg(test)]
mod tests;

/// Configuration for [`spectral_clustering`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralClusteringConfig {
    /// Target number of clusters (`>= 1`). Clamped to `graph.num_nodes()` when `k > n`.
    pub k: usize,
    /// Maximum iterations for the Lanczos/Jacobi Laplacian eigensolver.
    pub max_eigen_iterations: usize,
    /// Convergence tolerance for Laplacian eigenpairs (`> 0.0`).
    pub eigen_tol: f64,
    /// Maximum Lloyd iterations for the post-normalization K-Means++ step.
    pub kmeans_max_iterations: usize,
    /// Convergence tolerance for K-Means++.
    pub kmeans_tol: f64,
    /// Deterministic PRNG seed controlling eigensolver initialization and K-Means++.
    pub seed: u64,
}

impl Default for SpectralClusteringConfig {
    fn default() -> Self {
        Self {
            k: 2,
            max_eigen_iterations: 300,
            eigen_tol: 1e-4,
            kmeans_max_iterations: 100,
            kmeans_tol: 1e-4,
            seed: 42,
        }
    }
}

/// Output of [`spectral_clustering`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpectralClusteringResult {
    /// Canonical cluster assignment in `0..num_clusters` for each vertex.
    pub labels: Vec<u32>,
    /// Number of distinct non-empty clusters detected.
    pub num_clusters: usize,
    /// Number of Lanczos thick-restart iterations executed (`0` when using dense fallback or trivial graphs).
    pub eigen_iterations: usize,
}

/// Renumbers labels into canonical `0..c` in ascending order of first appearance and
/// returns the number of distinct clusters.
fn canonicalize_labels(labels: &mut [u32]) -> usize {
    let mut max_label = 0usize;
    for &l in labels.iter() {
        if (l as usize) > max_label {
            max_label = l as usize;
        }
    }
    let mut remap = vec![u32::MAX; max_label + 1];
    let mut next_id = 0u32;
    for label in labels.iter_mut() {
        let idx = *label as usize;
        if remap[idx] == u32::MAX {
            remap[idx] = next_id;
            next_id += 1;
        }
        *label = remap[idx];
    }
    next_id as usize
}

/// Orthogonalizes `candidate` against the `num_cols` columns stored in row-major `basis`
/// using two-pass Modified Gram-Schmidt and returns the residual L2 norm.
fn orthogonalize_against_basis(
    candidate: &mut [f64],
    basis: &[f64],
    n: usize,
    num_cols: usize,
) -> f64 {
    for _pass in 0..2 {
        for c in 0..num_cols {
            let mut dot = 0.0_f64;
            for i in 0..n {
                dot += candidate[i] * basis[i * num_cols + c];
            }
            for i in 0..n {
                candidate[i] -= dot * basis[i * num_cols + c];
            }
        }
    }
    let norm_sq: f64 = candidate.iter().map(|&v| v * v).sum();
    norm_sq.sqrt()
}

/// Recovers the missing $\lambda_0 = 0$ eigenvector orthogonal to the nontrivial modes
/// returned by [`smallest_sym_laplacian_eigenpairs`].
///
/// For connected graphs with edges, $D^{1/2}\mathbf{1}$ is strictly orthogonal to all
/// nontrivial eigenvectors. When a graph has multiple connected components and the
/// eigensolver dropped a non-$D^{1/2}\mathbf{1}$ nullspace vector, we scan connected
/// component indicators $D^{1/2}\mathbf{1}_{C}$ to find an orthogonal complement so no
/// component collapses to the origin.
fn find_missing_nullspace_mode<W: Scalar>(
    graph: &CsrGraph<W>,
    inv_sqrt_deg: &[f64],
    vectors: &[f64],
    num_modes: usize,
) -> Option<Vec<f64>> {
    let n = graph.num_nodes();
    let mut u0 = vec![0.0_f64; n];
    let mut norm_sq = 0.0_f64;
    for i in 0..n {
        let inv = inv_sqrt_deg[i];
        if inv > 0.0 {
            let deg_sqrt = 1.0 / inv;
            u0[i] = deg_sqrt;
            norm_sq += deg_sqrt * deg_sqrt;
        }
    }
    if norm_sq <= 1e-24 {
        // Edgeless graph: L_sym = I has no zero-eigenvalue subspace.
        return None;
    }

    let inv_norm = norm_sq.sqrt().recip();
    for val in &mut u0 {
        *val *= inv_norm;
    }

    let res_norm = orthogonalize_against_basis(&mut u0, vectors, n, num_modes);
    if res_norm > 1e-6 {
        let inv_r = res_norm.recip();
        for val in &mut u0 {
            *val *= inv_r;
        }
        return Some(u0);
    }

    // Disconnected fallback: find a component indicator D^{1/2} 1_C orthogonal to `vectors`.
    let mut visited = VisitStamp::new(n);
    let mut queue = VecDeque::new();
    let mut comp = Vec::new();
    let mut candidate = vec![0.0_f64; n];

    for start in 0..n {
        if !visited.visit(start) || inv_sqrt_deg[start] <= 0.0 {
            continue;
        }
        comp.clear();
        queue.push_back(start as u32);
        while let Some(u) = queue.pop_front() {
            comp.push(u as usize);
            for &v in graph.neighbors(u as usize) {
                if visited.visit(v as usize) {
                    queue.push_back(v);
                }
            }
        }

        candidate.fill(0.0);
        let mut c_norm_sq = 0.0_f64;
        for &idx in &comp {
            let inv = inv_sqrt_deg[idx];
            if inv > 0.0 {
                let d_sqrt = 1.0 / inv;
                candidate[idx] = d_sqrt;
                c_norm_sq += d_sqrt * d_sqrt;
            }
        }
        if c_norm_sq <= 1e-24 {
            continue;
        }
        let inv_c = c_norm_sq.sqrt().recip();
        for &idx in &comp {
            candidate[idx] *= inv_c;
        }

        let r_norm = orthogonalize_against_basis(&mut candidate, vectors, n, num_modes);
        if r_norm > 1e-6 {
            let inv_r = r_norm.recip();
            for val in &mut candidate {
                *val *= inv_r;
            }
            return Some(candidate);
        }
    }

    None
}

/// Performs Ng-Jordan-Weiss spectral clustering on an undirected [`CsrGraph`].
///
/// Computes the smallest eigenpairs of the symmetric normalized Laplacian
/// $L_{\mathrm{sym}} = I - D^{-1/2} A D^{-1/2}$ via
/// [`smallest_sym_laplacian_eigenpairs`], restores the leading $\lambda_0 = 0$ nullspace
/// mode when present, L2-normalizes each vertex row, clusters the normalized embedding with
/// [`kmeans_plus_plus`], and returns a [`SpectralClusteringResult`] with canonical `0..c`
/// cluster labels.
pub fn spectral_clustering<W: Scalar>(
    graph: &CsrGraph<W>,
    config: &SpectralClusteringConfig,
) -> CoreResult<SpectralClusteringResult> {
    if graph.is_directed() {
        return Err(CoreError::invalid_input(
            "spectral_clustering requires an undirected graph",
        ));
    }
    if config.k == 0 {
        return Err(CoreError::invalid_input("k must be >= 1"));
    }
    if config.max_eigen_iterations == 0 {
        return Err(CoreError::invalid_input(
            "max_eigen_iterations must be >= 1",
        ));
    }
    if !config.eigen_tol.is_finite() {
        return Err(CoreError::non_finite("SpectralClusteringConfig::eigen_tol"));
    }
    if config.eigen_tol <= 0.0 {
        return Err(CoreError::invalid_input(
            "eigen_tol must be strictly positive",
        ));
    }
    if config.kmeans_max_iterations == 0 {
        return Err(CoreError::invalid_input(
            "kmeans_max_iterations must be >= 1",
        ));
    }
    if !config.kmeans_tol.is_finite() {
        return Err(CoreError::non_finite(
            "SpectralClusteringConfig::kmeans_tol",
        ));
    }
    if config.kmeans_tol < 0.0 {
        return Err(CoreError::invalid_input("kmeans_tol must be >= 0.0"));
    }

    let n = graph.num_nodes();
    if n == 0 {
        return Ok(SpectralClusteringResult {
            labels: Vec::new(),
            num_clusters: 0,
            eigen_iterations: 0,
        });
    }
    let effective_k = config.k.min(n);
    if n == 1 || effective_k == 1 {
        return Ok(SpectralClusteringResult {
            labels: vec![0; n],
            num_clusters: 1,
            eigen_iterations: 0,
        });
    }

    let req_modes = (effective_k - 1).max(1);
    let eig = smallest_sym_laplacian_eigenpairs(
        graph,
        req_modes,
        config.eigen_tol,
        config.max_eigen_iterations,
        config.seed,
    )?;
    let eigen_iterations = eig.iterations;

    let null_mode =
        find_missing_nullspace_mode(graph, &eig.inv_sqrt_deg, &eig.vectors, eig.num_modes);

    // Build the Ng-Jordan-Weiss row-normalized spectral matrix T in row-major order.
    let n_cols = eig.num_modes + usize::from(null_mode.is_some());
    let mut row_norm_embed = vec![0.0_f64; n * n_cols];
    for i in 0..n {
        let dst = &mut row_norm_embed[i * n_cols..(i + 1) * n_cols];
        let mut col = 0usize;
        if let Some(ref u0) = null_mode {
            dst[0] = u0[i];
            col = 1;
        }
        let src_off = i * eig.num_modes;
        dst[col..(col + eig.num_modes)]
            .copy_from_slice(&eig.vectors[src_off..(src_off + eig.num_modes)]);

        let row_norm = dst.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if row_norm > 1e-12 {
            let inv = row_norm.recip();
            for val in dst.iter_mut() {
                *val *= inv;
            }
        } else {
            dst.fill(0.0);
        }
    }

    let kmeans_cfg = KMeansConfig {
        k: effective_k,
        max_iterations: config.kmeans_max_iterations,
        tol: config.kmeans_tol,
        n_init: if effective_k <= 32 { 10 } else { 3 },
        seed: config.seed,
    };
    let mut labels = kmeans_plus_plus(&row_norm_embed, n, n_cols, &kmeans_cfg)?.labels;
    let num_clusters = canonicalize_labels(&mut labels);
    Ok(SpectralClusteringResult {
        labels,
        num_clusters,
        eigen_iterations,
    })
}
