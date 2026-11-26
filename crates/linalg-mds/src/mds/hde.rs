use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, LayoutNd, Scalar, VisitStamp};
use egraph_linalg::shortest_path::{BfsWorkspace, DijkstraWorkspace, select_maxmin_sp_pivots};

use super::hde_pca::{hde_pca_project, prepare_unweighted_row_stats, validate_positive_weights};
use super::normalize::{normalize_layout_2d, normalize_layout_nd};

/// Configuration for Harel-Koren High-Dimensional Embedding ([`hde`], [`hde_nd`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HdeConfig {
    /// Maximum number of MaxMin shortest-path pivot dimensions $m$ (`default: 50`).
    pub dimensions: usize,
    /// Target PCA output dimensionality $k$ (`default: 2`).
    pub output_dims: usize,
    /// Deterministic PRNG seed for pivot selection and collapsed normalization fallback (`default: 42`).
    pub seed: u64,
    /// Convergence tolerance passed to [`symmetric_eigen_jacobi`](egraph_linalg::symmetric_eigen_jacobi) (`default: 1e-3`).
    pub tol: f64,
    /// Maximum Jacobi sweeps passed to [`symmetric_eigen_jacobi`](egraph_linalg::symmetric_eigen_jacobi) (`default: 100`).
    pub max_sweeps: usize,
    /// Whether to center and isotropically normalize the embedding to unit std (`default: true`).
    pub normalize: bool,
}

impl Default for HdeConfig {
    fn default() -> Self {
        Self {
            dimensions: 50,
            output_dims: 2,
            seed: 42,
            tol: 1e-3,
            max_sweeps: 100,
            normalize: true,
        }
    }
}

fn validate_hde_config(config: &HdeConfig, require_2d: bool) -> CoreResult<()> {
    if config.dimensions == 0 {
        return Err(CoreError::invalid_input(
            "HdeConfig::dimensions must be >= 1",
        ));
    }
    if config.output_dims == 0 {
        return Err(CoreError::invalid_input(
            "HdeConfig::output_dims must be >= 1",
        ));
    }
    if require_2d && config.output_dims != 2 {
        return Err(CoreError::invalid_input(
            "hde requires HdeConfig::output_dims == 2; use hde_nd for arbitrary dimensions",
        ));
    }
    if !config.tol.is_finite() {
        return Err(CoreError::non_finite("HdeConfig::tol"));
    }
    if config.tol <= 0.0 {
        return Err(CoreError::invalid_input(
            "HdeConfig::tol must be strictly positive",
        ));
    }
    if config.max_sweeps == 0 {
        return Err(CoreError::invalid_input(
            "HdeConfig::max_sweeps must be >= 1",
        ));
    }
    Ok(())
}

fn harel2004_hde_nd<W: Scalar>(
    graph: &CsrGraph<W>,
    pivots: &[u32],
    config: &HdeConfig,
    pivot_context: &'static str,
) -> CoreResult<LayoutNd<f64>> {
    let n = graph.num_nodes();
    for &p in pivots {
        if (p as usize) >= n {
            return Err(CoreError::index_out_of_bounds(p as usize, n, pivot_context));
        }
    }
    if n == 0 {
        return LayoutNd::zeros(0, config.output_dims);
    }
    if pivots.is_empty() {
        return Err(CoreError::invalid_input("pivots must not be empty"));
    }

    let mut seen = VisitStamp::new(n);
    let mut unique_pivots = Vec::with_capacity(pivots.len().min(n));
    for &p in pivots {
        if seen.visit(p as usize) {
            unique_pivots.push(p);
        }
    }
    let m_eff = unique_pivots.len();

    if n == 1 {
        return LayoutNd::zeros(1, config.output_dims);
    }

    let total = m_eff
        .checked_mul(n)
        .ok_or_else(|| CoreError::invalid_input("hde distance matrix size overflow"))?;
    m_eff
        .checked_mul(m_eff)
        .ok_or_else(|| CoreError::invalid_input("hde covariance matrix size overflow"))?;

    let mut layout = LayoutNd::<f64>::zeros(n, config.output_dims)?;
    let out_coords = layout.coords_mut();

    if graph.is_weighted() {
        let mut x_hat = vec![f64::INFINITY; total];
        let mut ws = DijkstraWorkspace::new(n);
        let mut max_abs_centered = 0.0_f64;
        let inv_n = 1.0 / (n as f64);

        for (i, &p) in unique_pivots.iter().enumerate() {
            let row = &mut x_hat[i * n..(i + 1) * n];
            ws.dijkstra_from(graph, p as usize, row)?;

            let mut max_finite = 0.0_f64;
            let mut has_finite = false;
            let mut has_nonfinite = false;
            for &d in row.iter() {
                if d.is_finite() {
                    has_finite = true;
                    if d > max_finite {
                        max_finite = d;
                    }
                } else {
                    has_nonfinite = true;
                }
            }
            if has_nonfinite {
                let fill = if has_finite { max_finite + 1.0 } else { 0.0 };
                if !fill.is_finite() {
                    return Err(CoreError::numerical_breakdown(
                        "HDE finite distance fallback overflowed f64",
                    ));
                }
                for d in row.iter_mut() {
                    if !d.is_finite() {
                        *d = fill;
                    }
                }
            }

            let mut mean = row.iter().sum::<f64>() * inv_n;
            if !mean.is_finite() {
                mean = row.iter().map(|&d| d * inv_n).sum::<f64>();
                if !mean.is_finite() {
                    return Err(CoreError::numerical_breakdown(
                        "HDE column mean overflowed f64",
                    ));
                }
            }
            for d in row.iter_mut() {
                *d -= mean;
                let abs_v = d.abs();
                if abs_v > max_abs_centered {
                    max_abs_centered = abs_v;
                }
            }
        }

        let zeros = vec![0.0_f64; m_eff];
        hde_pca_project(
            &x_hat,
            n,
            m_eff,
            &zeros,
            &zeros,
            max_abs_centered,
            config,
            out_coords,
        )?;
    } else {
        let mut ws = BfsWorkspace::new(n);
        let mut dist_col = vec![u32::MAX; n];
        let mut fills = vec![0.0_f64; m_eff];
        let mut means = vec![0.0_f64; m_eff];
        let mut max_abs_centered = 0.0_f64;

        if n <= u16::MAX as usize {
            let mut dists = vec![u16::MAX; total];
            for (i, &p) in unique_pivots.iter().enumerate() {
                ws.bfs_from_u32(graph, p as usize, &mut dist_col)?;
                let row = &mut dists[i * n..(i + 1) * n];
                for (dst, &d) in row.iter_mut().zip(dist_col.iter()) {
                    *dst = if d == u32::MAX { u16::MAX } else { d as u16 };
                }
                let (fill, mean, row_max_abs) =
                    prepare_unweighted_row_stats(row, u16::MAX, f64::from, n)?;
                fills[i] = fill;
                means[i] = mean;
                if row_max_abs > max_abs_centered {
                    max_abs_centered = row_max_abs;
                }
            }
            hde_pca_project(
                &dists,
                n,
                m_eff,
                &fills,
                &means,
                max_abs_centered,
                config,
                out_coords,
            )?;
        } else {
            let mut dists = vec![u32::MAX; total];
            for (i, &p) in unique_pivots.iter().enumerate() {
                ws.bfs_from_u32(graph, p as usize, &mut dist_col)?;
                let row = &mut dists[i * n..(i + 1) * n];
                row.copy_from_slice(&dist_col);
                let (fill, mean, row_max_abs) =
                    prepare_unweighted_row_stats(row, u32::MAX, f64::from, n)?;
                fills[i] = fill;
                means[i] = mean;
                if row_max_abs > max_abs_centered {
                    max_abs_centered = row_max_abs;
                }
            }
            hde_pca_project(
                &dists,
                n,
                m_eff,
                &fills,
                &means,
                max_abs_centered,
                config,
                out_coords,
            )?;
        }
    }

    Ok(layout)
}

/// Computes a 2D [`Layout2d`] using Harel-Koren High-Dimensional Embedding (`HDE`) with
/// MaxMin shortest-path pivot selection and PCA projection.
pub fn hde<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    config: &HdeConfig,
) -> CoreResult<Layout2d<S>> {
    validate_hde_config(config, true)?;
    validate_positive_weights(graph)?;
    let n = graph.num_nodes();
    if n == 0 {
        return Ok(Layout2d::zeros(0));
    }
    if n == 1 {
        return Ok(Layout2d::zeros(1));
    }
    let pivots = select_maxmin_sp_pivots(graph, config.dimensions, config.seed, 1)?;
    hde_with_pivots(graph, &pivots, config)
}

/// Computes a 2D [`Layout2d`] using Harel-Koren High-Dimensional Embedding (`HDE`) from an
/// explicitly supplied slice of pivot vertex indices.
pub fn hde_with_pivots<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    pivots: &[u32],
    config: &HdeConfig,
) -> CoreResult<Layout2d<S>> {
    validate_hde_config(config, true)?;
    validate_positive_weights(graph)?;
    let nd = harel2004_hde_nd(graph, pivots, config, "hde_with_pivots::pivots")?;
    let n = graph.num_nodes();
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let coords = nd.coords();
    for i in 0..n {
        x.push(coords[i * 2]);
        y.push(coords[i * 2 + 1]);
    }
    let mut layout_f64 = Layout2d::from_xy(x, y)?;
    if config.normalize {
        normalize_layout_2d(&mut layout_f64, 1.0, config.seed)?;
    }
    let out_x = layout_f64.x().iter().copied().map(S::from_f64).collect();
    let out_y = layout_f64.y().iter().copied().map(S::from_f64).collect();
    let out = Layout2d::from_xy(out_x, out_y)?;
    if !out.is_all_finite() {
        return Err(CoreError::numerical_breakdown(
            "HDE projection overflowed output scalar",
        ));
    }
    Ok(out)
}

/// Computes an $N$-dimensional [`LayoutNd`] using Harel-Koren High-Dimensional Embedding (`HDE`)
/// with MaxMin shortest-path pivot selection and PCA projection.
pub fn hde_nd<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    config: &HdeConfig,
) -> CoreResult<LayoutNd<S>> {
    validate_hde_config(config, false)?;
    validate_positive_weights(graph)?;
    let n = graph.num_nodes();
    if n == 0 {
        return LayoutNd::zeros(0, config.output_dims);
    }
    if n == 1 {
        return LayoutNd::zeros(1, config.output_dims);
    }
    let pivots = select_maxmin_sp_pivots(graph, config.dimensions, config.seed, 1)?;
    hde_nd_with_pivots(graph, &pivots, config)
}

/// Computes an $N$-dimensional [`LayoutNd`] using Harel-Koren High-Dimensional Embedding (`HDE`)
/// from an explicitly supplied slice of pivot vertex indices.
pub fn hde_nd_with_pivots<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    pivots: &[u32],
    config: &HdeConfig,
) -> CoreResult<LayoutNd<S>> {
    validate_hde_config(config, false)?;
    validate_positive_weights(graph)?;
    let mut nd_f64 = harel2004_hde_nd(graph, pivots, config, "hde_nd_with_pivots::pivots")?;
    if config.normalize {
        normalize_layout_nd(&mut nd_f64, 1.0, config.seed)?;
    }
    let n = graph.num_nodes();
    let d = config.output_dims;
    let mut out = LayoutNd::<S>::zeros(n, d)?;
    for (dst, &src) in out.coords_mut().iter_mut().zip(nd_f64.coords().iter()) {
        *dst = S::from_f64(src);
    }
    if !out.is_all_finite() {
        return Err(CoreError::numerical_breakdown(
            "HDE projection overflowed output scalar",
        ));
    }
    Ok(out)
}
