//! Pure-Rust Ng-Jordan-Weiss spectral clustering on undirected `CsrGraph` instances.

use std::collections::VecDeque;

use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar, VisitStamp};
use egraph_linalg::smallest_sym_laplacian_eigenpairs;

use crate::kmeans::{KMeansConfig, kmeans_plus_plus};

/// Configuration for [`spectral_clustering`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralClusteringConfig {
    /// Target number of clusters (`>= 1`). Clamped to `graph.num_nodes()` when `k > n`.
    pub k: usize,
    /// Maximum iterations for the Lanczos/Jacobi Laplacian eigensolver.
    pub max_eigen_iters: usize,
    /// Convergence tolerance for Laplacian eigenpairs (`> 0.0`).
    pub eigen_tol: f64,
    /// Maximum Lloyd iterations for the post-normalization K-Means++ step.
    pub kmeans_max_iters: usize,
    /// Convergence tolerance for K-Means++.
    pub kmeans_tol: f64,
    /// Deterministic PRNG seed controlling eigensolver initialization and K-Means++.
    pub seed: u64,
}

impl Default for SpectralClusteringConfig {
    fn default() -> Self {
        Self {
            k: 2,
            max_eigen_iters: 300,
            eigen_tol: 1e-4,
            kmeans_max_iters: 100,
            kmeans_tol: 1e-4,
            seed: 42,
        }
    }
}

/// Renumbers labels into canonical `0..c` in ascending order of first appearance.
fn canonicalize_labels(labels: &mut [u32]) {
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
/// [`kmeans_plus_plus`], and returns canonical `0..c` cluster labels.
pub fn spectral_clustering<W: Scalar>(
    graph: &CsrGraph<W>,
    config: &SpectralClusteringConfig,
) -> CoreResult<Vec<u32>> {
    if graph.is_directed() {
        return Err(CoreError::invalid_input(
            "spectral_clustering requires an undirected graph",
        ));
    }
    if config.k == 0 {
        return Err(CoreError::invalid_input("k must be >= 1"));
    }
    if config.max_eigen_iters == 0 {
        return Err(CoreError::invalid_input("max_eigen_iters must be >= 1"));
    }
    if !config.eigen_tol.is_finite() {
        return Err(CoreError::non_finite("SpectralClusteringConfig::eigen_tol"));
    }
    if config.eigen_tol <= 0.0 {
        return Err(CoreError::invalid_input(
            "eigen_tol must be strictly positive",
        ));
    }
    if config.kmeans_max_iters == 0 {
        return Err(CoreError::invalid_input("kmeans_max_iters must be >= 1"));
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
        return Ok(Vec::new());
    }
    let effective_k = config.k.min(n);
    if n == 1 || effective_k == 1 {
        return Ok(vec![0; n]);
    }

    let req_modes = (effective_k - 1).max(1);
    let eig = smallest_sym_laplacian_eigenpairs(
        graph,
        req_modes,
        config.eigen_tol,
        config.max_eigen_iters,
        config.seed,
    )?;

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
        max_iters: config.kmeans_max_iters,
        tol: config.kmeans_tol,
        n_init: if effective_k <= 32 { 10 } else { 3 },
        seed: config.seed,
    };
    let mut labels = kmeans_plus_plus(&row_norm_embed, n, n_cols, &kmeans_cfg)?.labels;
    canonicalize_labels(&mut labels);
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_barbell(clique_size: usize) -> CsrGraph<f64> {
        let n = clique_size * 2;
        let mut edges = Vec::new();
        for i in 0..clique_size {
            for j in (i + 1)..clique_size {
                edges.push((i as u32, j as u32));
                edges.push(((clique_size + i) as u32, (clique_size + j) as u32));
            }
        }
        edges.push(((clique_size - 1) as u32, clique_size as u32));
        CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
    }

    #[test]
    fn test_spectral_recovers_barbell_and_three_cliques() {
        let barbell = make_barbell(5);
        let labels = spectral_clustering(&barbell, &SpectralClusteringConfig::default()).unwrap();
        assert_eq!(labels.len(), 10);
        assert_eq!(labels[..5], [0, 0, 0, 0, 0]);
        assert_eq!(labels[5..], [1, 1, 1, 1, 1]);

        // Three 5-cliques connected as a ring of 3 weak bridges.
        let mut edges = Vec::new();
        for c in 0..3u32 {
            let base = c * 5;
            for i in 0..5 {
                for j in (i + 1)..5 {
                    edges.push((base + i, base + j));
                }
            }
        }
        edges.push((0, 5));
        edges.push((6, 10));
        edges.push((11, 1));
        let tri_clique = CsrGraph::<f64>::from_edges(15, &edges, None, false, false).unwrap();
        let cfg3 = SpectralClusteringConfig {
            k: 3,
            ..Default::default()
        };
        let l3 = spectral_clustering(&tri_clique, &cfg3).unwrap();
        assert_eq!(l3[0..5], [0, 0, 0, 0, 0]);
        assert_eq!(l3[5..10], [1, 1, 1, 1, 1]);
        assert_eq!(l3[10..15], [2, 2, 2, 2, 2]);
    }

    #[test]
    fn test_spectral_disconnected_cliques_and_seed_determinism() {
        let mut edges = Vec::new();
        for i in 0..5u32 {
            for j in (i + 1)..5 {
                edges.push((i, j));
                edges.push((5 + i, 5 + j));
            }
        }
        let disc = CsrGraph::<f64>::from_edges(10, &edges, None, false, false).unwrap();
        let cfg = SpectralClusteringConfig {
            k: 2,
            seed: 77,
            ..Default::default()
        };
        let l1 = spectral_clustering(&disc, &cfg).unwrap();
        let l2 = spectral_clustering(&disc, &cfg).unwrap();
        assert_eq!(l1, l2);
        assert_eq!(l1[..5], [0, 0, 0, 0, 0]);
        assert_eq!(l1[5..], [1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_spectral_lanczos_large_barbell_and_weighted_f32() {
        // n = 60 (> 48 dense threshold) exercises the Lanczos eigensolver path.
        let large_barbell = make_barbell(30);
        let labels =
            spectral_clustering(&large_barbell, &SpectralClusteringConfig::default()).unwrap();
        assert_eq!(labels.len(), 60);
        assert_eq!(labels[..30], [0; 30]);
        assert_eq!(labels[30..], [1; 30]);

        // Weighted f32 graph: strong cliques (10.0) joined by weak bridge (0.05).
        let mut edges = Vec::new();
        let mut weights = Vec::new();
        for i in 0..4u32 {
            for j in (i + 1)..4 {
                edges.push((i, j));
                weights.push(10.0_f32);
                edges.push((4 + i, 4 + j));
                weights.push(10.0_f32);
            }
        }
        edges.push((3, 4));
        weights.push(0.05_f32);
        let wg = CsrGraph::from_edges(8, &edges, Some(&weights), false, false).unwrap();
        let wl = spectral_clustering(&wg, &SpectralClusteringConfig::default()).unwrap();
        assert_eq!(wl[..4], [0, 0, 0, 0]);
        assert_eq!(wl[4..], [1, 1, 1, 1]);
    }

    #[test]
    fn test_spectral_degenerate_and_error_cases() {
        let cfg = SpectralClusteringConfig::default();

        let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        assert!(spectral_clustering(&empty, &cfg).unwrap().is_empty());

        let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        assert_eq!(spectral_clustering(&single, &cfg).unwrap(), vec![0]);

        let edgeless = CsrGraph::<f64>::from_edges(6, &[], None, false, false).unwrap();
        let le = spectral_clustering(&edgeless, &cfg).unwrap();
        assert_eq!(le.len(), 6);
        assert!(le.iter().all(|&l| l < 2));

        let directed =
            CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, true, false).unwrap();
        assert!(matches!(
            spectral_clustering(&directed, &cfg),
            Err(CoreError::InvalidInput(_))
        ));

        let barbell = make_barbell(3);
        assert!(matches!(
            spectral_clustering(
                &barbell,
                &SpectralClusteringConfig {
                    k: 0,
                    ..Default::default()
                }
            ),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            spectral_clustering(
                &barbell,
                &SpectralClusteringConfig {
                    eigen_tol: 0.0,
                    ..Default::default()
                }
            ),
            Err(CoreError::InvalidInput(_))
        ));
    }
}
