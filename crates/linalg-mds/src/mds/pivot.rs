use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, LayoutNd, Scalar};
use egraph_linalg::eigen::symmetric_eigen_jacobi;
use egraph_linalg::shortest_path::{BfsWorkspace, DijkstraWorkspace, select_maxmin_sp_pivots};

use super::normalize::normalize_layout_2d;

/// Configuration for [`pivot_mds`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PivotMdsConfig {
    /// Maximum number of Core-First MaxMin shortest-path pivots (`default: 50`).
    pub num_pivots: usize,
    /// Minimum degree prioritized during Core-First MaxMin pivot selection (`default: 2`).
    pub min_degree: usize,
    /// Deterministic PRNG seed controlling pivot selection and degenerate fallback (`default: 42`).
    pub seed: u64,
    /// Optional target global standard deviation enforced via [`normalize_layout_2d`] (`default: Some(1.0)`).
    pub target_std: Option<f64>,
}

impl Default for PivotMdsConfig {
    fn default() -> Self {
        Self {
            num_pivots: 50,
            min_degree: 2,
            seed: 42,
            target_std: Some(1.0),
        }
    }
}

/// Streams a `(num_nodes, num_reps)` row-major distance table in 3 passes with zero `(N, k)`
/// temporary `f64` buffers, computing $(n / k)^{1/4} C V \mathrm{diag}(e^{-1/4})$.
fn mds_from_row_major_distances<F, W>(
    num_nodes: usize,
    num_reps: usize,
    dims: usize,
    fill_row: F,
    mut write_row: W,
) -> CoreResult<()>
where
    F: Fn(usize, &mut [f64]),
    W: FnMut(usize, &[f64]),
{
    if num_nodes <= 1 || num_reps == 0 {
        return Ok(());
    }

    let mut row_means = vec![0.0_f64; num_nodes];
    let mut col_means = vec![0.0_f64; num_reps];
    let mut row_inf_cnt = vec![0u32; num_nodes];
    let mut col_inf_cnt = vec![0usize; num_reps];
    let mut row_buf = vec![0.0_f64; num_reps];
    let mut max_finite_sq = 0.0_f64;
    let mut has_nonfinite = false;

    for i in 0..num_nodes {
        fill_row(i, &mut row_buf);
        let mut r_sum = 0.0_f64;
        let mut r_inf = 0u32;
        for (j, &d) in row_buf.iter().enumerate() {
            if d.is_finite() {
                let sq = d * d;
                if sq.is_finite() {
                    r_sum += sq;
                    col_means[j] += sq;
                    if sq > max_finite_sq {
                        max_finite_sq = sq;
                    }
                    continue;
                }
            }
            r_inf = r_inf.saturating_add(1);
            col_inf_cnt[j] += 1;
            has_nonfinite = true;
        }
        row_means[i] = r_sum;
        row_inf_cnt[i] = r_inf;
    }

    let total = num_nodes.saturating_mul(num_reps);
    let fallback_sq = if has_nonfinite {
        let denom = (total.max(1) * 4) as f64;
        let safe_cap = f64::MAX / denom;
        if max_finite_sq > 0.0 {
            (max_finite_sq * 4.0).min(safe_cap)
        } else {
            1.0
        }
    } else {
        0.0
    };

    let inv_cols = 1.0 / (num_reps as f64);
    let inv_rows = 1.0 / (num_nodes as f64);
    for (rm, &cnt) in row_means.iter_mut().zip(row_inf_cnt.iter()) {
        if cnt > 0 {
            *rm += (cnt as f64) * fallback_sq;
        }
        *rm *= inv_cols;
    }
    for (cm, &cnt) in col_means.iter_mut().zip(col_inf_cnt.iter()) {
        if cnt > 0 {
            *cm += (cnt as f64) * fallback_sq;
        }
        *cm *= inv_rows;
    }
    let grand_mean = row_means.iter().sum::<f64>() * inv_rows;

    let mut ct_c = vec![0.0_f64; num_reps * num_reps];
    for (i, &rm) in row_means.iter().enumerate() {
        fill_row(i, &mut row_buf);
        for (j, val) in row_buf.iter_mut().enumerate() {
            let d = *val;
            let delta_ij = if d.is_finite() {
                let sq = d * d;
                if sq.is_finite() { sq } else { fallback_sq }
            } else {
                fallback_sq
            };
            *val = 0.5 * (rm + col_means[j] - delta_ij - grand_mean);
        }
        for a in 0..num_reps {
            let ca = row_buf[a];
            for b in a..num_reps {
                ct_c[a * num_reps + b] += ca * row_buf[b];
            }
        }
    }
    for a in 0..num_reps {
        for b in 0..a {
            ct_c[a * num_reps + b] = ct_c[b * num_reps + a];
        }
    }

    let (evals, evecs) = symmetric_eigen_jacobi(&ct_c, num_reps, 120, 1e-12)?;
    let used_dims = dims.min(num_reps);
    let scale = ((num_nodes as f64) / (num_reps as f64)).powf(0.25);

    let mut proj = vec![0.0_f64; num_reps * used_dims];
    for m in 0..used_dims {
        let eig_idx = num_reps - 1 - m;
        let eig_val = evals[eig_idx];
        let weight = if eig_val > 1e-10 {
            scale / eig_val.sqrt().sqrt()
        } else {
            0.0
        };
        for j in 0..num_reps {
            proj[j * used_dims + m] = weight * evecs[j * num_reps + eig_idx];
        }
    }

    let mut out_acc = vec![0.0_f64; used_dims];
    for (i, &rm) in row_means.iter().enumerate() {
        fill_row(i, &mut row_buf);
        out_acc.fill(0.0);
        for (j, &d) in row_buf.iter().enumerate() {
            let delta_ij = if d.is_finite() {
                let sq = d * d;
                if sq.is_finite() { sq } else { fallback_sq }
            } else {
                fallback_sq
            };
            let cij = 0.5 * (rm + col_means[j] - delta_ij - grand_mean);
            let p_row = &proj[j * used_dims..(j + 1) * used_dims];
            for m in 0..used_dims {
                out_acc[m] += cij * p_row[m];
            }
        }
        write_row(i, &out_acc);
    }
    Ok(())
}

/// Computes $d$-dimensional Representative / Landmark MDS coordinates
/// $(n / k)^{1/4} C V \mathrm{diag}(e^{-1/4})$ from a precomputed row-major `(num_nodes, num_reps)`
/// distance matrix using [`symmetric_eigen_jacobi`] with `e > 1e-10` zero-eigenvalue guards.
pub fn representative_mds<S: Scalar>(
    distances_to_reps: &[f64],
    num_nodes: usize,
    num_reps: usize,
    dims: usize,
) -> CoreResult<LayoutNd<S>> {
    if dims == 0 {
        return Err(CoreError::invalid_input("dims must be >= 1"));
    }
    let expected = num_nodes
        .checked_mul(num_reps)
        .ok_or_else(|| CoreError::invalid_input("representative_mds matrix size overflow"))?;
    if distances_to_reps.len() != expected {
        return Err(CoreError::dimension_mismatch(
            expected,
            distances_to_reps.len(),
            "representative_mds::distances_to_reps",
        ));
    }

    let mut layout = LayoutNd::zeros(num_nodes, dims)?;
    if num_nodes <= 1 || num_reps == 0 {
        return Ok(layout);
    }

    let coords = layout.coords_mut();
    mds_from_row_major_distances(
        num_nodes,
        num_reps,
        dims,
        |i, row_buf| {
            row_buf.copy_from_slice(&distances_to_reps[i * num_reps..(i + 1) * num_reps]);
        },
        |i, vals| {
            let out_row = &mut coords[i * dims..i * dims + vals.len()];
            for (dst, &val) in out_row.iter_mut().zip(vals.iter()) {
                *dst = S::from_f64(val);
            }
        },
    )?;
    Ok(layout)
}

/// Computes a 2D [`Layout2d`] via PivotMDS from a precomputed row-major `(num_nodes, num_pivots)`
/// distance matrix, optionally normalizing isotropic standard deviation to `target_std`.
pub fn pivot_mds_from_distances<S: Scalar>(
    distances_nk: &[f64],
    num_nodes: usize,
    num_pivots: usize,
    target_std: Option<f64>,
    seed: u64,
) -> CoreResult<Layout2d<S>> {
    if let Some(t_std) = target_std {
        if !t_std.is_finite() {
            return Err(CoreError::non_finite("pivot_mds::target_std"));
        }
        if t_std <= 0.0 {
            return Err(CoreError::invalid_input(
                "target_std must be strictly positive",
            ));
        }
    }
    let expected = num_nodes
        .checked_mul(num_pivots)
        .ok_or_else(|| CoreError::invalid_input("pivot_mds distance matrix size overflow"))?;
    if distances_nk.len() != expected {
        return Err(CoreError::dimension_mismatch(
            expected,
            distances_nk.len(),
            "pivot_mds_from_distances::distances_nk",
        ));
    }

    let mut layout = Layout2d::zeros(num_nodes);
    if num_nodes > 1 && num_pivots > 0 {
        let (xs, ys) = layout.xy_mut();
        mds_from_row_major_distances(
            num_nodes,
            num_pivots,
            2,
            |i, row_buf| {
                row_buf.copy_from_slice(&distances_nk[i * num_pivots..(i + 1) * num_pivots]);
            },
            |i, vals| {
                xs[i] = S::from_f64(vals[0]);
                if vals.len() > 1 {
                    ys[i] = S::from_f64(vals[1]);
                }
            },
        )?;
    }
    if let Some(t_std) = target_std {
        normalize_layout_2d(&mut layout, t_std, seed)?;
    }
    Ok(layout)
}

/// Computes a 2D [`Layout2d`] using Core-First MaxMin PivotMDS with `(n/k)^{1/4} C V diag(e^{-1/4})`
/// scaling, disconnected finite distance fallback, and optional isotropic std normalization.
pub fn pivot_mds<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    config: &PivotMdsConfig,
) -> CoreResult<Layout2d<S>> {
    if config.num_pivots == 0 {
        return Err(CoreError::invalid_input("num_pivots must be >= 1"));
    }
    if let Some(t_std) = config.target_std {
        if !t_std.is_finite() {
            return Err(CoreError::non_finite("PivotMdsConfig::target_std"));
        }
        if t_std <= 0.0 {
            return Err(CoreError::invalid_input(
                "target_std must be strictly positive",
            ));
        }
    }

    let n = graph.num_nodes();
    if n == 0 {
        return Ok(Layout2d::zeros(0));
    }
    if n == 1 {
        return Ok(Layout2d::zeros(1));
    }

    let pivots = select_maxmin_sp_pivots(graph, config.num_pivots, config.seed, config.min_degree)?;
    pivot_mds_with_pivots(graph, &pivots, config.target_std, config.seed)
}

/// Computes PivotMDS using an explicitly supplied slice of pivot vertex indices.
pub fn pivot_mds_with_pivots<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    pivots: &[u32],
    target_std: Option<f64>,
    seed: u64,
) -> CoreResult<Layout2d<S>> {
    if let Some(t_std) = target_std {
        if !t_std.is_finite() {
            return Err(CoreError::non_finite("pivot_mds_with_pivots::target_std"));
        }
        if t_std <= 0.0 {
            return Err(CoreError::invalid_input(
                "target_std must be strictly positive",
            ));
        }
    }
    let n = graph.num_nodes();
    if n == 0 {
        return Ok(Layout2d::zeros(0));
    }
    if n == 1 {
        return Ok(Layout2d::zeros(1));
    }
    let k = pivots.len();
    if k < 2 {
        let mut layout = Layout2d::zeros(n);
        if let Some(t_std) = target_std {
            normalize_layout_2d(&mut layout, t_std, seed)?;
        }
        return Ok(layout);
    }
    for &p in pivots {
        if (p as usize) >= n {
            return Err(CoreError::index_out_of_bounds(
                p as usize,
                n,
                "pivot_mds_with_pivots::pivots",
            ));
        }
    }

    let total = n
        .checked_mul(k)
        .ok_or_else(|| CoreError::invalid_input("pivot_mds distance matrix size overflow"))?;

    if graph.is_weighted() {
        let mut distances_nk = vec![f64::INFINITY; total];
        let mut dist_col = vec![f64::INFINITY; n];
        let mut ws = DijkstraWorkspace::new(n);
        for (col, &p) in pivots.iter().enumerate() {
            ws.dijkstra_from(graph, p as usize, &mut dist_col)?;
            for i in 0..n {
                distances_nk[i * k + col] = dist_col[i];
            }
        }
        return pivot_mds_from_distances(&distances_nk, n, k, target_std, seed);
    }

    let mut layout = Layout2d::zeros(n);
    let mut dist_col = vec![u32::MAX; n];
    let mut ws = BfsWorkspace::new(n);

    if n <= u16::MAX as usize {
        let mut distances_nk = vec![u16::MAX; total];
        for (col, &p) in pivots.iter().enumerate() {
            ws.bfs_from_u32(graph, p as usize, &mut dist_col)?;
            for i in 0..n {
                let d = dist_col[i];
                distances_nk[i * k + col] = if d == u32::MAX { u16::MAX } else { d as u16 };
            }
        }
        let (xs, ys) = layout.xy_mut();
        mds_from_row_major_distances(
            n,
            k,
            2,
            |i, row_buf| {
                let row = &distances_nk[i * k..(i + 1) * k];
                for (dst, &d) in row_buf.iter_mut().zip(row.iter()) {
                    *dst = if d == u16::MAX {
                        f64::INFINITY
                    } else {
                        f64::from(d)
                    };
                }
            },
            |i, vals| {
                xs[i] = S::from_f64(vals[0]);
                if vals.len() > 1 {
                    ys[i] = S::from_f64(vals[1]);
                }
            },
        )?;
    } else {
        let mut distances_nk = vec![u32::MAX; total];
        for (col, &p) in pivots.iter().enumerate() {
            ws.bfs_from_u32(graph, p as usize, &mut dist_col)?;
            for i in 0..n {
                distances_nk[i * k + col] = dist_col[i];
            }
        }
        let (xs, ys) = layout.xy_mut();
        mds_from_row_major_distances(
            n,
            k,
            2,
            |i, row_buf| {
                let row = &distances_nk[i * k..(i + 1) * k];
                for (dst, &d) in row_buf.iter_mut().zip(row.iter()) {
                    *dst = if d == u32::MAX {
                        f64::INFINITY
                    } else {
                        f64::from(d)
                    };
                }
            },
            |i, vals| {
                xs[i] = S::from_f64(vals[0]);
                if vals.len() > 1 {
                    ys[i] = S::from_f64(vals[1]);
                }
            },
        )?;
    }

    if let Some(t_std) = target_std {
        normalize_layout_2d(&mut layout, t_std, seed)?;
    }
    Ok(layout)
}
