use egraph_core::CsrGraph;

/// Exact $O(|E|)$ spectral variance and degree summary statistics of the unweighted
/// simple-graph symmetric normalized Laplacian $L_{\mathrm{sym}}$.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
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

fn is_fast_undirected_unweighted_clean<W>(graph: &CsrGraph<W>) -> bool {
    if graph.is_directed() {
        return false;
    }
    let n = graph.num_nodes();
    for u in 0..n {
        let u32_u = u as u32;
        let row = graph.neighbors(u);
        if row.iter().any(|&v| v == u32_u || (v as usize) >= n) {
            return false;
        }
        if !row.windows(2).all(|win| win[0] < win[1]) {
            return false;
        }
    }
    true
}

fn degree_summary(deg: &[usize]) -> (f64, f64, f64) {
    let n = deg.len();
    let n_f64 = n as f64;
    let mut deg_sum = 0usize;
    let mut leaf_count = 0usize;
    for &d in deg {
        deg_sum += d;
        if d == 1 {
            leaf_count += 1;
        }
    }
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
    (mean_degree, deg_cv, leaf_fraction)
}

/// Computes exact $O(|E|)$ spectral variance and degree summary statistics of the unweighted
/// simple-graph symmetric normalized Laplacian $L_{\mathrm{sym}}$ without eigensolvers.
///
/// Works on unweighted [`CsrGraph<()>`](CsrGraph) as well as any arbitrary weight type `<W>`
/// using purely topological adjacency.
pub fn spectral_moment_stats<W>(graph: &CsrGraph<W>) -> SpectralMomentStats {
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

    let n_f64 = n as f64;
    if is_fast_undirected_unweighted_clean(graph) {
        let deg: Vec<usize> = (0..n).map(|u| graph.degree(u)).collect();
        let (mean_degree, deg_cv, leaf_fraction) = degree_summary(&deg);

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

    let edges = graph.canonical_undirected_edges();
    let mut deg = vec![0usize; n];
    for &(u, v) in &edges {
        deg[u as usize] += 1;
        deg[v as usize] += 1;
    }
    let (mean_degree, deg_cv, leaf_fraction) = degree_summary(&deg);

    let sigma_lambda_sq = if n <= 1 || edges.is_empty() {
        0.0
    } else {
        let inv_deg: Vec<f64> = deg
            .iter()
            .map(|&d| if d > 0 { 1.0 / (d as f64) } else { 0.0 })
            .collect();
        let mut sum_inv = 0.0_f64;
        for &(u, v) in &edges {
            sum_inv += 2.0 * inv_deg[u as usize] * inv_deg[v as usize];
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
