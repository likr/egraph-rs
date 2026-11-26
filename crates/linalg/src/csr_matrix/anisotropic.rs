use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar};

use super::laplacians::{canonical_undirected_adjacency, is_fast_undirected_clean};
use super::matrix::CsrMatrix;

/// Configuration for Coifman-Lafon anisotropic graph Laplacian construction ($L_\alpha$).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnisotropicLaplacianConfig {
    /// Density reweighting exponent `alpha` in $A^{(\alpha)}_{ij} = \tilde{d}_i^{-\alpha} A_{ij} \tilde{d}_j^{-\alpha}$.
    /// `0.0` recovers standard adjacency; `0.5` gives Fokker-Planck; `1.0` gives Laplace-Beltrami.
    pub alpha: f64,
    /// Outer symmetric degree normalization exponent `beta` in $S (D^{(\alpha)} - A^{(\alpha)}) S$.
    /// `0.0` gives unnormalized $D^{(\alpha)} - A^{(\alpha)}$; `-0.5` gives symmetric normalized form.
    pub beta: f64,
    /// Additive degree regularization $c_{\mathrm{deg}} \ge 0.0$ in $\tilde{d}_i = \min(d_i, d_{\mathrm{cap}}) + c_{\mathrm{deg}}$.
    pub degree_reg: f64,
    /// Optional degree saturation cap $d_{\mathrm{cap}} > 0.0$ applied before `degree_reg`.
    pub degree_cap: Option<f64>,
    /// When `true` (default), outer scaling $S$ uses reweighted degrees $(d^{(\alpha)}_i)^\beta$,
    /// guaranteeing spectrum $\subseteq [0, 2]$ whenever `beta == -0.5`.
    /// When `false`, outer scaling $S$ uses saturated raw degrees $\tilde{d}_i^\beta$.
    pub reweighted_degree_norm: bool,
}

impl Default for AnisotropicLaplacianConfig {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            beta: 0.0,
            degree_reg: 0.0,
            degree_cap: None,
            reweighted_degree_norm: true,
        }
    }
}

/// Output of [`anisotropic_laplacian`].
#[derive(Debug, Clone, PartialEq)]
pub struct AnisotropicLaplacianResult {
    /// Symmetric PSD sparse matrix $S (D^{(\alpha)} - A^{(\alpha)}) S$.
    pub matrix: CsrMatrix,
    /// Near-nullspace vector $u$ satisfying `matrix * u == 0` on every non-isolated component
    /// (`1.0 / s_i` when `s_i > 0`, `0.0` for isolated unit-diagonal rows, `1.0` for isolated zero rows).
    pub nullspace: Vec<f64>,
    /// Outer symmetric scaling factors $s_i$ (`1.0` when `beta == 0.0`).
    pub outer_scale: Vec<f64>,
}

fn is_fast_undirected_clean_loopless<W: Scalar>(graph: &CsrGraph<W>) -> bool {
    if !is_fast_undirected_clean(graph) {
        return false;
    }
    for u in 0..graph.num_nodes() {
        if graph.neighbors(u).binary_search(&(u as u32)).is_ok() {
            return false;
        }
    }
    true
}

/// Constructs a symmetric positive semidefinite Coifman-Lafon anisotropic graph Laplacian
/// $L_{\mathrm{aniso}} = S (D^{(\alpha)} - A^{(\alpha)}) S$ from `graph`.
///
/// Special cases:
/// - `(alpha=0.0, beta=0.0, degree_reg=0.0, degree_cap=None)`: exact combinatorial Laplacian $L = D - A$.
/// - `(alpha=0.0, beta=-0.5, degree_reg=0.0, degree_cap=None, reweighted_degree_norm=true)`:
///   exact symmetric normalized Laplacian $L_{\mathrm{sym}} = I - D^{-1/2} A D^{-1/2}$.
/// - `(alpha=0.0, beta=-p, degree_reg=0.0, degree_cap=None, reweighted_degree_norm=false)`:
///   degree-power interpolation $L_{\mathrm{pow}}(p) = D^{-p}(D - A)D^{-p}$.
/// - `(alpha, beta=-0.5, reweighted_degree_norm=true)`: Coifman-Lafon symmetric normalized
///   operator with spectrum strictly bounded in $[0, 2]$.
pub fn anisotropic_laplacian<W: Scalar>(
    graph: &CsrGraph<W>,
    config: AnisotropicLaplacianConfig,
) -> CoreResult<AnisotropicLaplacianResult> {
    if !config.alpha.is_finite() {
        return Err(CoreError::non_finite("anisotropic_laplacian::alpha"));
    }
    if !config.beta.is_finite() {
        return Err(CoreError::non_finite("anisotropic_laplacian::beta"));
    }
    if !config.degree_reg.is_finite() {
        return Err(CoreError::non_finite("anisotropic_laplacian::degree_reg"));
    }
    if config.degree_reg < 0.0 {
        return Err(CoreError::invalid_input(format!(
            "degree_reg must be >= 0.0, got {}",
            config.degree_reg
        )));
    }
    if let Some(cap) = config.degree_cap {
        if !cap.is_finite() {
            return Err(CoreError::non_finite("anisotropic_laplacian::degree_cap"));
        }
        if cap <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "degree_cap must be > 0.0, got {cap}"
            )));
        }
    }

    let n = graph.num_nodes();
    if n == 0 {
        return Ok(AnisotropicLaplacianResult {
            matrix: CsrMatrix::from_validated_parts(0, vec![0], Vec::new(), Vec::new(), Vec::new()),
            nullspace: Vec::new(),
            outer_scale: Vec::new(),
        });
    }

    let is_fast = is_fast_undirected_clean_loopless(graph);
    let (rows_opt, raw_degrees) = if is_fast {
        let mut deg = vec![0.0_f64; n];
        let weights_opt = graph.weights();
        for (u, d_u) in deg.iter_mut().enumerate() {
            let start = graph.indptr()[u] as usize;
            let end = graph.indptr()[u + 1] as usize;
            *d_u = match weights_opt {
                Some(ws) => ws[start..end].iter().map(|w| w.to_f64()).sum(),
                None => (end - start) as f64,
            };
        }
        (None, deg)
    } else {
        let (rows, deg) = canonical_undirected_adjacency(graph, false);
        (Some(rows), deg)
    };

    let mut d_eff = vec![0.0_f64; n];
    let mut d_neg_alpha = vec![0.0_f64; n];
    for i in 0..n {
        let d_sat = match config.degree_cap {
            Some(cap) => raw_degrees[i].min(cap),
            None => raw_degrees[i],
        };
        let eff = d_sat + config.degree_reg;
        d_eff[i] = eff;
        d_neg_alpha[i] = if config.alpha == 0.0 {
            1.0
        } else if eff > 0.0 {
            eff.powf(-config.alpha)
        } else {
            0.0
        };
    }

    let mut d_alpha = vec![0.0_f64; n];
    if config.alpha == 0.0 {
        d_alpha.copy_from_slice(&raw_degrees);
    } else if let Some(ref rows) = rows_opt {
        for (u, row) in rows.iter().enumerate() {
            let du = d_neg_alpha[u];
            if du == 0.0 {
                continue;
            }
            let mut sum = 0.0_f64;
            for &(v, w) in row {
                sum += du * w * d_neg_alpha[v as usize];
            }
            d_alpha[u] = sum;
        }
    } else {
        let weights_opt = graph.weights();
        for (u, d_a_u) in d_alpha.iter_mut().enumerate() {
            let du = d_neg_alpha[u];
            if du == 0.0 {
                continue;
            }
            let start = graph.indptr()[u] as usize;
            let end = graph.indptr()[u + 1] as usize;
            let mut sum = 0.0_f64;
            for pos in start..end {
                let v = graph.indices()[pos] as usize;
                let w = match weights_opt {
                    Some(ws) => ws[pos].to_f64(),
                    None => 1.0,
                };
                sum += du * w * d_neg_alpha[v];
            }
            *d_a_u = sum;
        }
    }

    let is_sym_norm_half = (config.beta + 0.5).abs() <= 1e-12;
    let use_unit_diag_isolated = config.reweighted_degree_norm && is_sym_norm_half;

    let mut outer_scale = vec![0.0_f64; n];
    let mut nullspace = vec![0.0_f64; n];
    for i in 0..n {
        let base_deg = if config.reweighted_degree_norm {
            d_alpha[i]
        } else {
            d_eff[i]
        };
        let s_i = if config.beta == 0.0 {
            1.0
        } else if base_deg > 0.0 {
            if is_sym_norm_half {
                base_deg.sqrt().recip()
            } else {
                base_deg.powf(config.beta)
            }
        } else {
            0.0
        };
        outer_scale[i] = s_i;
        nullspace[i] = if s_i > 0.0 {
            if config.beta == 0.0 {
                1.0
            } else if is_sym_norm_half {
                base_deg.sqrt()
            } else {
                1.0 / s_i
            }
        } else if use_unit_diag_isolated {
            0.0
        } else {
            1.0
        };
    }

    let est_edges = match &rows_opt {
        Some(rows) => rows.iter().map(Vec::len).sum(),
        None => graph.nnz(),
    };
    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0u32);
    let mut indices = Vec::with_capacity(est_edges + n);
    let mut values = Vec::with_capacity(est_edges + n);
    let mut diagonal = vec![0.0_f64; n];

    if let Some(ref rows) = rows_opt {
        for (u, row) in rows.iter().enumerate() {
            let u_u32 = u as u32;
            let scale_u = outer_scale[u] * d_neg_alpha[u];
            let diag_u = if use_unit_diag_isolated {
                1.0
            } else {
                outer_scale[u] * d_alpha[u] * outer_scale[u]
            };
            diagonal[u] = diag_u;
            let mut diag_inserted = !use_unit_diag_isolated && diag_u == 0.0;

            for &(v, w) in row {
                if !diag_inserted && v > u_u32 {
                    indices.push(u_u32);
                    values.push(diag_u);
                    diag_inserted = true;
                }
                let entry = -(scale_u * w * d_neg_alpha[v as usize] * outer_scale[v as usize]);
                indices.push(v);
                values.push(if entry == 0.0 { 0.0 } else { entry });
            }
            if !diag_inserted {
                indices.push(u_u32);
                values.push(diag_u);
            }
            let ptr = u32::try_from(indices.len()).map_err(|_| {
                CoreError::invalid_csr("anisotropic_laplacian nnz exceeds u32::MAX")
            })?;
            indptr.push(ptr);
        }
    } else {
        let weights_opt = graph.weights();
        for u in 0..n {
            let u_u32 = u as u32;
            let scale_u = outer_scale[u] * d_neg_alpha[u];
            let diag_u = if use_unit_diag_isolated {
                1.0
            } else {
                outer_scale[u] * d_alpha[u] * outer_scale[u]
            };
            diagonal[u] = diag_u;
            let mut diag_inserted = !use_unit_diag_isolated && diag_u == 0.0;

            let start = graph.indptr()[u] as usize;
            let end = graph.indptr()[u + 1] as usize;
            for pos in start..end {
                let v = graph.indices()[pos];
                if !diag_inserted && v > u_u32 {
                    indices.push(u_u32);
                    values.push(diag_u);
                    diag_inserted = true;
                }
                let w = match weights_opt {
                    Some(ws) => ws[pos].to_f64(),
                    None => 1.0,
                };
                let entry = -(scale_u * w * d_neg_alpha[v as usize] * outer_scale[v as usize]);
                indices.push(v);
                values.push(if entry == 0.0 { 0.0 } else { entry });
            }
            if !diag_inserted {
                indices.push(u_u32);
                values.push(diag_u);
            }
            let ptr = u32::try_from(indices.len()).map_err(|_| {
                CoreError::invalid_csr("anisotropic_laplacian nnz exceeds u32::MAX")
            })?;
            indptr.push(ptr);
        }
    }

    Ok(AnisotropicLaplacianResult {
        matrix: CsrMatrix::from_validated_parts(n, indptr, indices, values, diagonal),
        nullspace,
        outer_scale,
    })
}
