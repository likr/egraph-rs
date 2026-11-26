use egraph_core::{CsrGraph, Scalar};

use super::laplacians::{canonical_undirected_adjacency, is_fast_undirected_clean};

/// Exact $O(|E|)$ spectral variance and degree summary statistics of the unweighted
/// simple-graph symmetric normalized Laplacian $L_{\mathrm{sym}}$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralMomentStats {
    /// Exact spectral variance $\sigma_\lambda^2 = \frac{1}{N}\|L_{\mathrm{sym}}\|_F^2 - 1 = \frac{2}{N}\sum_{(u,v)\in E}\frac{1}{d_u d_v}$.
    pub sigma_lambda_sq: f64,
    /// Clamped spectral standard deviation $\sigma_\lambda = \sqrt{\max(\sigma_\lambda^2, 10^{-6})}$.
    pub sigma_lambda: f64,
    /// Coefficient of variation of node degrees ($\mathrm{std}(d) / \mathrm{mean}(d)$).
    pub deg_cv: f64,
    /// Mean unweighted simple degree across all vertices.
    pub mean_degree: f64,
    /// Fraction of vertices with degree equal to 1.
    pub leaf_fraction: f64,
}

/// Computes exact $O(|E|)$ spectral variance and degree summary statistics of the unweighted
/// simple-graph symmetric normalized Laplacian $L_{\mathrm{sym}}$ without eigensolvers.
pub fn spectral_moment_stats<W: Scalar>(graph: &CsrGraph<W>) -> SpectralMomentStats {
    let n = graph.num_nodes();
    if n == 0 {
        return SpectralMomentStats {
            sigma_lambda_sq: 0.0,
            sigma_lambda: 1e-6_f64.sqrt(),
            deg_cv: 0.0,
            mean_degree: 0.0,
            leaf_fraction: 0.0,
        };
    }

    if is_fast_undirected_clean(graph) {
        let mut deg = vec![0usize; n];
        let mut deg_sum = 0usize;
        let mut leaf_count = 0usize;
        for (u, d_u) in deg.iter_mut().enumerate() {
            let d = graph.degree(u);
            *d_u = d;
            deg_sum += d;
            if d == 1 {
                leaf_count += 1;
            }
        }
        let n_f64 = n as f64;
        let mean_degree = (deg_sum as f64) / n_f64;
        let deg_cv = if mean_degree > 0.0 {
            let var_deg = deg
                .iter()
                .map(|&d| {
                    let diff = (d as f64) - mean_degree;
                    diff * diff
                })
                .sum::<f64>()
                / n_f64;
            var_deg.sqrt() / mean_degree
        } else {
            0.0
        };
        let leaf_fraction = (leaf_count as f64) / n_f64;

        let sigma_lambda_sq = if n <= 1 || graph.nnz() == 0 {
            0.0
        } else {
            let inv_deg: Vec<f64> = deg
                .iter()
                .map(|&d| if d > 0 { 1.0 / (d as f64) } else { 0.0 })
                .collect();
            let mut sum_inv = 0.0_f64;
            for (u, &inv_u) in inv_deg.iter().enumerate() {
                if inv_u == 0.0 {
                    continue;
                }
                for &v in graph.neighbors(u) {
                    sum_inv += inv_u * inv_deg[v as usize];
                }
            }
            sum_inv / n_f64
        };

        return SpectralMomentStats {
            sigma_lambda_sq,
            sigma_lambda: sigma_lambda_sq.max(1e-6).sqrt(),
            deg_cv,
            mean_degree,
            leaf_fraction,
        };
    }

    let (rows, deg_f64) = canonical_undirected_adjacency(graph, true);
    let n_f64 = n as f64;
    let deg_sum: f64 = deg_f64.iter().sum();
    let mean_degree = deg_sum / n_f64;
    let deg_cv = if mean_degree > 0.0 {
        let var_deg = deg_f64
            .iter()
            .map(|&d| {
                let diff = d - mean_degree;
                diff * diff
            })
            .sum::<f64>()
            / n_f64;
        var_deg.sqrt() / mean_degree
    } else {
        0.0
    };
    let leaf_count = deg_f64.iter().filter(|&&d| (d - 1.0).abs() < 1e-12).count();
    let leaf_fraction = (leaf_count as f64) / n_f64;

    let nnz: usize = rows.iter().map(Vec::len).sum();
    let sigma_lambda_sq = if n <= 1 || nnz == 0 {
        0.0
    } else {
        let inv_deg: Vec<f64> = deg_f64
            .iter()
            .map(|&d| if d > 0.0 { 1.0 / d } else { 0.0 })
            .collect();
        let mut sum_inv = 0.0_f64;
        for (u, row) in rows.iter().enumerate() {
            let inv_u = inv_deg[u];
            if inv_u == 0.0 {
                continue;
            }
            for &(v, _) in row {
                sum_inv += inv_u * inv_deg[v as usize];
            }
        }
        sum_inv / n_f64
    };

    SpectralMomentStats {
        sigma_lambda_sq,
        sigma_lambda: sigma_lambda_sq.max(1e-6).sqrt(),
        deg_cv,
        mean_degree,
        leaf_fraction,
    }
}
