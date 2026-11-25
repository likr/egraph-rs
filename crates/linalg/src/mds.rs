//! Multidimensional Scaling (`pivot_mds`, `representative_mds`, `classical_mds`, `RdMds`),
//! Harel-Koren High-Dimensional Embedding (`hde`, `hde_nd`), Koren degree-normalized spectral
//! initialization, and isotropic 2D coordinate normalization.

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, Layout2d, LayoutNd, Scalar, VisitStamp,
};

use crate::eigen::{smallest_sym_laplacian_eigenpairs, symmetric_eigen_jacobi};
use crate::shortest_path::{BfsWorkspace, DijkstraWorkspace, select_maxmin_sp_pivots};

#[inline]
fn next_gaussian_pair(rng: &mut DeterministicRng) -> (f64, f64) {
    let u1 = rng.next_f64().max(f64::MIN_POSITIVE);
    let u2 = rng.next_f64();
    let r = (-2.0 * u1.ln()).sqrt();
    let (s, c) = (std::f64::consts::TAU * u2).sin_cos();
    (r * c, r * s)
}

#[inline]
fn mean_and_iso_std<S: Scalar>(xs: &[S], ys: &[S]) -> (f64, f64, f64) {
    let n_f64 = xs.len() as f64;
    let mx = xs.iter().map(|v| v.to_f64()).sum::<f64>() / n_f64;
    let my = ys.iter().map(|v| v.to_f64()).sum::<f64>() / n_f64;
    let mut sum_sq = 0.0_f64;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let dx = x.to_f64() - mx;
        let dy = y.to_f64() - my;
        sum_sq += dx * dx + dy * dy;
    }
    (mx, my, (sum_sq / (2.0 * n_f64)).sqrt())
}

#[inline]
fn median_of_slice(buf: &mut [f64]) -> f64 {
    let len = buf.len();
    debug_assert!(len > 0);
    if len % 2 == 1 {
        *buf.select_nth_unstable_by(len / 2, f64::total_cmp).1
    } else {
        let (left, &mut upper, _) = buf.select_nth_unstable_by(len / 2, f64::total_cmp);
        let lower = left.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        0.5 * (lower + upper)
    }
}

#[inline]
fn quantile_of_slice(buf: &mut [f64], q: f64) -> f64 {
    let len = buf.len();
    debug_assert!(len >= 2);
    let pos = q * ((len - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - (lo as f64);
    let (left, &mut val_hi, _) = buf.select_nth_unstable_by(hi, f64::total_cmp);
    if lo == hi {
        val_hi
    } else {
        let val_lo = left.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        val_lo + frac * (val_hi - val_lo)
    }
}

/// Sanitizes non-finite entries, centers coordinates per axis, and scales isotropically so
/// $\sqrt{\frac{1}{2N} \sum_i ((x_i - \bar{x})^2 + (y_i - \bar{y})^2)} = \mathrm{target\_std}$,
/// matching `numpy.std(coords)` on `(N, 2)` centered arrays.
///
/// Collapsed or degenerate layouts (`n >= 2`, `std < 1e-12`) fall back deterministically to
/// standard Gaussian coordinates drawn from `DeterministicRng::from_seed_u64(seed)`.
pub fn normalize_layout_2d<S: Scalar>(
    layout: &mut Layout2d<S>,
    target_std: f64,
    seed: u64,
) -> CoreResult<()> {
    if !target_std.is_finite() {
        return Err(CoreError::non_finite("normalize_layout_2d::target_std"));
    }
    if target_std <= 0.0 {
        return Err(CoreError::invalid_input(
            "target_std must be strictly positive",
        ));
    }
    let n = layout.len();
    if n == 0 {
        return Ok(());
    }

    let mut rng = DeterministicRng::from_seed_u64(seed);
    let (xs, ys) = layout.xy_mut();
    for i in 0..n {
        if !xs[i].is_finite() {
            xs[i] = S::from_f64(next_gaussian_pair(&mut rng).0);
        }
        if !ys[i].is_finite() {
            ys[i] = S::from_f64(next_gaussian_pair(&mut rng).0);
        }
    }

    if n == 1 {
        xs[0] = S::ZERO;
        ys[0] = S::ZERO;
        return Ok(());
    }

    let (mut mx, mut my, mut std_dev) = mean_and_iso_std(xs, ys);
    if !std_dev.is_finite() || std_dev < 1e-12 {
        for i in 0..n {
            let (gx, gy) = next_gaussian_pair(&mut rng);
            xs[i] = S::from_f64(gx);
            ys[i] = S::from_f64(gy);
        }
        (mx, my, std_dev) = mean_and_iso_std(xs, ys);
        if !std_dev.is_finite() || std_dev < 1e-12 {
            xs.fill(S::ZERO);
            ys.fill(S::ZERO);
            return Ok(());
        }
    }

    let scale = target_std / std_dev;
    for i in 0..n {
        xs[i] = S::from_f64((xs[i].to_f64() - mx) * scale);
        ys[i] = S::from_f64((ys[i].to_f64() - my) * scale);
    }
    Ok(())
}

fn mean_and_iso_std_nd(coords: &[f64], n: usize, d: usize, means_out: &mut [f64]) -> f64 {
    let n_f64 = n as f64;
    means_out.fill(0.0);
    for i in 0..n {
        let row = &coords[i * d..(i + 1) * d];
        for (m, &val) in row.iter().enumerate() {
            means_out[m] += val;
        }
    }
    for mean in means_out.iter_mut() {
        *mean /= n_f64;
    }
    let mut sum_sq = 0.0_f64;
    for i in 0..n {
        let row = &coords[i * d..(i + 1) * d];
        for (m, &val) in row.iter().enumerate() {
            let diff = val - means_out[m];
            sum_sq += diff * diff;
        }
    }
    (sum_sq / ((d as f64) * n_f64)).sqrt()
}

/// Normalizes an N-dimensional layout identically to [`normalize_layout_2d`] (matching bit-for-bit
/// when `d == 2`).
fn normalize_layout_nd<S: Scalar>(
    layout: &mut LayoutNd<S>,
    target_std: f64,
    seed: u64,
) -> CoreResult<()> {
    if !target_std.is_finite() {
        return Err(CoreError::non_finite("normalize_layout_nd::target_std"));
    }
    if target_std <= 0.0 {
        return Err(CoreError::invalid_input(
            "target_std must be strictly positive",
        ));
    }
    let n = layout.num_nodes();
    let d = layout.dims();
    if n == 0 {
        return Ok(());
    }

    let mut rng = DeterministicRng::from_seed_u64(seed);
    let coords = layout.coords_mut();
    for val in coords.iter_mut() {
        if !val.is_finite() {
            *val = S::from_f64(next_gaussian_pair(&mut rng).0);
        }
    }

    if n == 1 {
        coords.fill(S::ZERO);
        return Ok(());
    }

    let mut work = Vec::with_capacity(coords.len());
    work.extend(coords.iter().map(|v| v.to_f64()));
    let mut means = vec![0.0_f64; d];
    let mut std_dev = mean_and_iso_std_nd(&work, n, d, &mut means);
    if !std_dev.is_finite() || std_dev < 1e-12 {
        // Fill pairs with next_gaussian_pair so d == 2 consumes the RNG identically to normalize_layout_2d.
        for i in 0..n {
            let row = &mut work[i * d..(i + 1) * d];
            let mut m = 0usize;
            while m < d {
                let (gx, gy) = next_gaussian_pair(&mut rng);
                row[m] = gx;
                if m + 1 < d {
                    row[m + 1] = gy;
                }
                m += 2;
            }
        }
        std_dev = mean_and_iso_std_nd(&work, n, d, &mut means);
        if !std_dev.is_finite() || std_dev < 1e-12 {
            coords.fill(S::ZERO);
            return Ok(());
        }
    }

    let scale = target_std / std_dev;
    for i in 0..n {
        let src = &work[i * d..(i + 1) * d];
        let dst = &mut coords[i * d..(i + 1) * d];
        for m in 0..d {
            dst[m] = S::from_f64((src[m] - means[m]) * scale);
        }
    }
    Ok(())
}

/// Sanitizes non-finite entries, centers coordinates at per-axis medians `(mx, my)`, and scales
/// isotropically using the Gaussian-calibrated radial quantile $r_q / \sqrt{-2 \ln(1 - q)}$ so
/// core spread matches `target_std` even when extreme outliers inflate the bounding box.
///
/// Falls back to isotropic standard deviation around `(mx, my)` if the quantile scale collapses
/// (`< 1e-12`), and deterministically to Gaussian coordinates from `seed` if all points collapse.
pub fn robust_normalize_layout_2d<S: Scalar>(
    layout: &mut Layout2d<S>,
    target_std: f64,
    quantile: f64,
    seed: u64,
) -> CoreResult<()> {
    if !target_std.is_finite() {
        return Err(CoreError::non_finite(
            "robust_normalize_layout_2d::target_std",
        ));
    }
    if target_std <= 0.0 {
        return Err(CoreError::invalid_input(
            "target_std must be strictly positive",
        ));
    }
    if !quantile.is_finite() {
        return Err(CoreError::non_finite(
            "robust_normalize_layout_2d::quantile",
        ));
    }
    if quantile <= 0.0 || quantile >= 1.0 {
        return Err(CoreError::invalid_input(
            "quantile must be strictly in (0, 1)",
        ));
    }

    let n = layout.len();
    if n == 0 {
        return Ok(());
    }

    let mut rng = DeterministicRng::from_seed_u64(seed);
    let (xs, ys) = layout.xy_mut();
    for i in 0..n {
        if !xs[i].is_finite() {
            xs[i] = S::from_f64(next_gaussian_pair(&mut rng).0);
        }
        if !ys[i].is_finite() {
            ys[i] = S::from_f64(next_gaussian_pair(&mut rng).0);
        }
    }

    if n == 1 {
        xs[0] = S::ZERO;
        ys[0] = S::ZERO;
        return Ok(());
    }

    let mut scratch = Vec::with_capacity(n);
    scratch.extend(xs.iter().map(|v| v.to_f64()));
    let mx = median_of_slice(&mut scratch);

    scratch.clear();
    scratch.extend(ys.iter().map(|v| v.to_f64()));
    let my = median_of_slice(&mut scratch);

    scratch.clear();
    for i in 0..n {
        scratch.push((xs[i].to_f64() - mx).hypot(ys[i].to_f64() - my));
    }
    let r_q = quantile_of_slice(&mut scratch, quantile);
    let gauss_factor = (-2.0 * (1.0 - quantile).ln()).sqrt();
    let sigma_hat = r_q / gauss_factor;

    let mut center_x = mx;
    let mut center_y = my;
    let mut sigma = sigma_hat;
    if !sigma.is_finite() || sigma < 1e-12 {
        let mut sum_sq = 0.0_f64;
        for i in 0..n {
            let dx = xs[i].to_f64() - mx;
            let dy = ys[i].to_f64() - my;
            sum_sq += dx * dx + dy * dy;
        }
        let iso_std = (sum_sq / (2.0 * (n as f64))).sqrt();
        if iso_std.is_finite() && iso_std >= 1e-12 {
            sigma = iso_std;
        } else {
            for i in 0..n {
                let (gx, gy) = next_gaussian_pair(&mut rng);
                xs[i] = S::from_f64(gx);
                ys[i] = S::from_f64(gy);
            }
            let (gmx, gmy, g_std) = mean_and_iso_std(xs, ys);
            if !g_std.is_finite() || g_std < 1e-12 {
                xs.fill(S::ZERO);
                ys.fill(S::ZERO);
                return Ok(());
            }
            center_x = gmx;
            center_y = gmy;
            sigma = g_std;
        }
    }

    let scale = target_std / sigma;
    for i in 0..n {
        xs[i] = S::from_f64((xs[i].to_f64() - center_x) * scale);
        ys[i] = S::from_f64((ys[i].to_f64() - center_y) * scale);
    }
    Ok(())
}

/// Replaces infinite/non-finite entries in a squared-distance buffer with a bounded finite
/// fallback and applies double centering in-place:
/// $C = -\frac{1}{2}(I - \frac{1}{n}\mathbf{1}\mathbf{1}^T)\Delta(I - \frac{1}{k}\mathbf{1}\mathbf{1}^T)$.
fn double_center_distances(distances: &[f64], nrows: usize, ncols: usize) -> Vec<f64> {
    if nrows == 0 || ncols == 0 {
        return Vec::new();
    }
    let total = nrows * ncols;
    let mut delta = Vec::with_capacity(total);
    let mut max_finite_sq = 0.0_f64;
    let mut has_nonfinite = false;

    for &d in distances {
        if d.is_finite() {
            let sq = d * d;
            if sq.is_finite() {
                delta.push(sq);
                if sq > max_finite_sq {
                    max_finite_sq = sq;
                }
                continue;
            }
        }
        delta.push(f64::INFINITY);
        has_nonfinite = true;
    }

    if has_nonfinite {
        let denom = (total.max(1) * 4) as f64;
        let safe_cap = f64::MAX / denom;
        let fallback_sq = if max_finite_sq > 0.0 {
            (max_finite_sq * 4.0).min(safe_cap)
        } else {
            1.0
        };
        for val in &mut delta {
            if !val.is_finite() {
                *val = fallback_sq;
            }
        }
    }

    let mut row_means = vec![0.0_f64; nrows];
    let mut col_means = vec![0.0_f64; ncols];
    let inv_cols = 1.0 / (ncols as f64);
    let inv_rows = 1.0 / (nrows as f64);

    for i in 0..nrows {
        let row = &delta[i * ncols..(i + 1) * ncols];
        let mut r_sum = 0.0_f64;
        for (j, &val) in row.iter().enumerate() {
            r_sum += val;
            col_means[j] += val;
        }
        row_means[i] = r_sum * inv_cols;
    }
    for cm in &mut col_means {
        *cm *= inv_rows;
    }
    let grand_mean = row_means.iter().sum::<f64>() * inv_rows;

    for (i, &rm) in row_means.iter().enumerate() {
        let row_off = i * ncols;
        for j in 0..ncols {
            delta[row_off + j] = 0.5 * (rm + col_means[j] - delta[row_off + j] - grand_mean);
        }
    }
    delta
}

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

/// Configuration for Harel-Koren High-Dimensional Embedding ([`hde`], [`hde_nd`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HdeConfig {
    /// Maximum number of MaxMin shortest-path pivot dimensions $m$ (`default: 50`).
    pub dimensions: usize,
    /// Target PCA output dimensionality $k$ (`default: 2`).
    pub output_dims: usize,
    /// Deterministic PRNG seed for pivot selection and collapsed normalization fallback (`default: 42`).
    pub seed: u64,
    /// Convergence tolerance passed to [`symmetric_eigen_jacobi`] (`default: 1e-3`).
    pub tol: f64,
    /// Maximum Jacobi sweeps passed to [`symmetric_eigen_jacobi`] (`default: 100`).
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

fn validate_positive_weights<W: Scalar>(graph: &CsrGraph<W>) -> CoreResult<()> {
    if let Some(ws) = graph.weights() {
        for &w in ws {
            let wf = w.to_f64();
            if !wf.is_finite() || wf <= 0.0 {
                return Err(CoreError::invalid_input(
                    "HDE edge weights must be finite and strictly positive",
                ));
            }
        }
    }
    Ok(())
}

/// Canonicalizes eigenvector orientation deterministically without depending on near-tie ordering.
fn canonicalize_eigenvector_sign(u: &mut [f64]) {
    let max_abs = u.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    if max_abs > 0.0
        && let Some(&first_dom) = u.iter().find(|v| v.abs() > 0.5 * max_abs)
        && first_dom < 0.0
    {
        for v in u.iter_mut() {
            *v = -*v;
        }
    }
}

trait PivotRowAccess {
    fn centered_val(&self, idx: usize, fill: f64, mean: f64) -> f64;
}

impl PivotRowAccess for [u16] {
    #[inline]
    fn centered_val(&self, idx: usize, fill: f64, mean: f64) -> f64 {
        let d = self[idx];
        let raw = if d == u16::MAX { fill } else { f64::from(d) };
        raw - mean
    }
}

impl PivotRowAccess for [u32] {
    #[inline]
    fn centered_val(&self, idx: usize, fill: f64, mean: f64) -> f64 {
        let d = self[idx];
        let raw = if d == u32::MAX { fill } else { f64::from(d) };
        raw - mean
    }
}

impl PivotRowAccess for [f64] {
    #[inline]
    fn centered_val(&self, idx: usize, _fill: f64, _mean: f64) -> f64 {
        self[idx]
    }
}

fn prepare_unweighted_row_stats<T: Copy + PartialEq>(
    row: &[T],
    sentinel: T,
    to_f64: fn(T) -> f64,
    n: usize,
) -> CoreResult<(f64, f64, f64)> {
    let mut max_finite = 0.0_f64;
    let mut has_finite = false;
    let mut has_nonfinite = false;
    for &d in row {
        if d != sentinel {
            has_finite = true;
            let df = to_f64(d);
            if df > max_finite {
                max_finite = df;
            }
        } else {
            has_nonfinite = true;
        }
    }
    let fill = if has_nonfinite {
        let f = if has_finite { max_finite + 1.0 } else { 0.0 };
        if !f.is_finite() {
            return Err(CoreError::numerical_breakdown(
                "HDE finite distance fallback overflowed f64",
            ));
        }
        f
    } else {
        0.0
    };

    let inv_n = 1.0 / (n as f64);
    let mut sum = 0.0_f64;
    for &d in row {
        sum += if d == sentinel { fill } else { to_f64(d) };
    }
    let mut mean = sum * inv_n;
    if !mean.is_finite() {
        mean = row
            .iter()
            .map(|&d| (if d == sentinel { fill } else { to_f64(d) }) * inv_n)
            .sum::<f64>();
        if !mean.is_finite() {
            return Err(CoreError::numerical_breakdown(
                "HDE column mean overflowed f64",
            ));
        }
    }

    let mut max_abs = 0.0_f64;
    for &d in row {
        let val = if d == sentinel { fill } else { to_f64(d) };
        let abs_v = (val - mean).abs();
        if abs_v > max_abs {
            max_abs = abs_v;
        }
    }
    Ok((fill, mean, max_abs))
}

#[allow(clippy::too_many_arguments)]
fn hde_pca_project<T>(
    dists: &[T],
    n: usize,
    m_eff: usize,
    fills: &[f64],
    means: &[f64],
    max_abs_centered: f64,
    config: &HdeConfig,
    out_coords: &mut [f64],
) -> CoreResult<()>
where
    [T]: PivotRowAccess,
{
    if !max_abs_centered.is_finite() {
        return Err(CoreError::numerical_breakdown(
            "HDE centered distance overflowed f64",
        ));
    }
    if max_abs_centered == 0.0 {
        return Ok(());
    }

    // Unconditionally normalize centered entries into [-1, 1] via a two-step scale so
    // S = (1/n) X_hat X_hat^T and its Frobenius norm never overflow or underflow across all
    // positive finite f64 scales (including subnormals < 1e-300 and huge weights > 1e300).
    let (pre_scale, cov_scale) = if max_abs_centered < 1e-300 {
        let pre = 1e150_f64;
        (pre, 1.0 / (max_abs_centered * pre))
    } else if max_abs_centered > 1e300 {
        let pre = 1e-150_f64;
        (pre, 1.0 / (max_abs_centered * pre))
    } else {
        (1.0_f64, 1.0 / max_abs_centered)
    };
    if !cov_scale.is_finite() {
        return Err(CoreError::numerical_breakdown(
            "HDE covariance scale overflowed f64",
        ));
    }
    let scale_centered = |val: f64| -> f64 { (val * pre_scale) * cov_scale };

    let mut cov = vec![0.0_f64; m_eff * m_eff];
    let inv_n = 1.0 / (n as f64);
    for a in 0..m_eff {
        let row_a = &dists[a * n..(a + 1) * n];
        let fill_a = fills[a];
        let mean_a = means[a];
        for b in a..m_eff {
            let row_b = &dists[b * n..(b + 1) * n];
            let fill_b = fills[b];
            let mean_b = means[b];
            let mut dot = 0.0_f64;
            for v in 0..n {
                let va = scale_centered(row_a.centered_val(v, fill_a, mean_a));
                let vb = scale_centered(row_b.centered_val(v, fill_b, mean_b));
                dot += va * vb;
            }
            let s_ab = dot * inv_n;
            if !s_ab.is_finite() {
                return Err(CoreError::numerical_breakdown(
                    "HDE covariance entry overflowed f64",
                ));
            }
            cov[a * m_eff + b] = s_ab;
            cov[b * m_eff + a] = s_ab;
        }
    }

    let (evals, evecs) = symmetric_eigen_jacobi(&cov, m_eff, config.max_sweeps, config.tol)?;
    let output_dims = config.output_dims;
    let used_dims = output_dims.min(m_eff);
    let lambda_0 = evals[m_eff - 1];
    if !lambda_0.is_finite() || lambda_0 <= 1e-300 {
        return Ok(());
    }

    let mut u_k = vec![0.0_f64; m_eff * used_dims];
    let mut mode_active = vec![false; used_dims];
    let mut col_buf = vec![0.0_f64; m_eff];
    for m in 0..used_dims {
        let eig_idx = m_eff - 1 - m;
        let lambda_m = evals[eig_idx];
        if m >= 1 && (!lambda_m.is_finite() || lambda_m <= 1e-12 * lambda_0) {
            continue;
        }
        for p in 0..m_eff {
            col_buf[p] = evecs[p * m_eff + eig_idx];
        }
        canonicalize_eigenvector_sign(&mut col_buf);
        for p in 0..m_eff {
            u_k[p * used_dims + m] = col_buf[p];
        }
        mode_active[m] = true;
    }

    // When config.normalize is enabled, project using the scaled O(1) matrix so downstream
    // isotropic normalization neither falls back to Gaussian noise on tiny weights (<= 1e-13)
    // nor overflows sum_sq on huge weights (>= 1e155).
    for p in 0..m_eff {
        let row_p = &dists[p * n..(p + 1) * n];
        let fill_p = fills[p];
        let mean_p = means[p];
        let u_row = &u_k[p * used_dims..(p + 1) * used_dims];
        for v in 0..n {
            let raw_x_pv = row_p.centered_val(v, fill_p, mean_p);
            let x_pv = if config.normalize {
                scale_centered(raw_x_pv)
            } else {
                raw_x_pv
            };
            let out_row = &mut out_coords[v * output_dims..v * output_dims + used_dims];
            for m in 0..used_dims {
                if mode_active[m] {
                    out_row[m] += x_pv * u_row[m];
                }
            }
        }
    }
    if !config.normalize && !out_coords.iter().all(|v| v.is_finite()) {
        return Err(CoreError::numerical_breakdown(
            "HDE unnormalized projection overflowed f64",
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

/// Computes Koren's degree-normalized 2D spectral layout $(D^{-1/2} v_2, D^{-1/2} v_3)$
/// from the 2nd and 3rd smallest eigenvectors of $L_{\mathrm{sym}}$, with optional isotropic
/// standard-deviation normalization.
pub fn koren_spectral_init<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    target_std: Option<f64>,
    tol: f64,
    max_iters: usize,
    seed: u64,
) -> CoreResult<Layout2d<S>> {
    if let Some(t_std) = target_std {
        if !t_std.is_finite() {
            return Err(CoreError::non_finite("koren_spectral_init::target_std"));
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

    let eig = smallest_sym_laplacian_eigenpairs(graph, 2, tol, max_iters, seed)?;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let m = eig.num_modes;
    for i in 0..n {
        let d_inv = eig.inv_sqrt_deg[i];
        let vx = if m >= 1 {
            d_inv * eig.vectors[i * m]
        } else {
            0.0
        };
        let vy = if m >= 2 {
            d_inv * eig.vectors[i * m + 1]
        } else {
            0.0
        };
        x.push(S::from_f64(vx));
        y.push(S::from_f64(vy));
    }

    let mut layout = Layout2d::from_xy(x, y)?;
    if let Some(t_std) = target_std {
        normalize_layout_2d(&mut layout, t_std, seed)?;
    }
    Ok(layout)
}

/// Computes zero-guarded Classical MDS from a flat row-major `(num_nodes, num_nodes)`
/// distance matrix via [`symmetric_eigen_jacobi`].
pub fn classical_mds_from_distances<S: Scalar>(
    distances_nn: &[f64],
    num_nodes: usize,
    dims: usize,
) -> CoreResult<LayoutNd<S>> {
    if dims == 0 {
        return Err(CoreError::invalid_input("dims must be >= 1"));
    }
    let expected = num_nodes
        .checked_mul(num_nodes)
        .ok_or_else(|| CoreError::invalid_input("classical_mds matrix size overflow"))?;
    if distances_nn.len() != expected {
        return Err(CoreError::dimension_mismatch(
            expected,
            distances_nn.len(),
            "classical_mds_from_distances::distances_nn",
        ));
    }

    let mut layout = LayoutNd::zeros(num_nodes, dims)?;
    if num_nodes <= 1 {
        return Ok(layout);
    }

    let b = double_center_distances(distances_nn, num_nodes, num_nodes);
    let (evals, evecs) = symmetric_eigen_jacobi(&b, num_nodes, 120, 1e-12)?;
    let used_dims = dims.min(num_nodes);

    let mut weights = vec![0.0_f64; used_dims];
    for m in 0..used_dims {
        let eig_val = evals[num_nodes - 1 - m];
        weights[m] = if eig_val > 1e-10 { eig_val.sqrt() } else { 0.0 };
    }

    let coords = layout.coords_mut();
    for i in 0..num_nodes {
        let out_row = &mut coords[i * dims..i * dims + used_dims];
        for m in 0..used_dims {
            let eig_idx = num_nodes - 1 - m;
            out_row[m] = S::from_f64(evecs[i * num_nodes + eig_idx] * weights[m]);
        }
    }
    Ok(layout)
}

/// Computes zero-guarded Classical MDS on `graph` via all-pairs shortest paths and
/// [`symmetric_eigen_jacobi`].
pub fn classical_mds<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    dims: usize,
) -> CoreResult<LayoutNd<S>> {
    if dims == 0 {
        return Err(CoreError::invalid_input("dims must be >= 1"));
    }
    let n = graph.num_nodes();
    if n <= 1 {
        return LayoutNd::zeros(n, dims);
    }

    let total = n
        .checked_mul(n)
        .ok_or_else(|| CoreError::invalid_input("classical_mds APSP size overflow"))?;
    let mut dists = vec![f64::INFINITY; total];
    if graph.is_weighted() {
        let mut ws = DijkstraWorkspace::new(n);
        for src in 0..n {
            ws.dijkstra_from(graph, src, &mut dists[src * n..(src + 1) * n])?;
        }
    } else {
        let mut ws = BfsWorkspace::new(n);
        for src in 0..n {
            ws.bfs_from(graph, src, &mut dists[src * n..(src + 1) * n])?;
        }
    }
    classical_mds_from_distances(&dists, n, dims)
}

/// Computes a 2D [`Layout2d`] using zero-guarded Classical MDS via [`symmetric_eigen_jacobi`].
pub fn classical_mds_2d<W: Scalar, S: Scalar>(graph: &CsrGraph<W>) -> CoreResult<Layout2d<S>> {
    let nd = classical_mds::<W, f64>(graph, 2)?;
    let n = graph.num_nodes();
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let row = nd.node(i)?;
        x.push(S::from_f64(row[0]));
        y.push(S::from_f64(row[1]));
    }
    Layout2d::from_xy(x, y)
}

/// Zero-guarded Resistance-distance MDS (`RdMds`) spectral embedding builder.
///
/// Computes $d$-dimensional coordinates $D^{-1/2} U \Lambda^{-1/2}$ (or $U \Lambda^{-1/2}$
/// when `degree_normalize == false`) from the smallest nontrivial eigenpairs of $L_{\mathrm{sym}}$,
/// zeroing out modes with $\lambda \le \mathrm{eig\_eps}$ to prevent zero-division blowups on
/// disconnected or edgeless graphs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RdMds {
    /// Target spectral embedding dimension $d$.
    pub dims: usize,
    /// Minimum eigenvalue threshold; modes with $\lambda \le \mathrm{eig\_eps}$ map to zero.
    pub eig_eps: f64,
    /// Eigensolver convergence tolerance.
    pub tol: f64,
    /// Maximum number of thick-restart Lanczos cycles.
    pub max_iters: usize,
    /// Deterministic PRNG seed.
    pub seed: u64,
    /// Whether to apply $D^{-1/2}$ degree normalization (`true` for effective resistance).
    pub degree_normalize: bool,
    /// Optional target global standard deviation when producing 2D layouts via [`RdMds::embedding_2d`].
    pub target_std: Option<f64>,
}

impl Default for RdMds {
    fn default() -> Self {
        Self {
            dims: 2,
            eig_eps: 1e-10,
            tol: 1e-6,
            max_iters: 1000,
            seed: 42,
            degree_normalize: true,
            target_std: None,
        }
    }
}

impl RdMds {
    /// Creates a new [`RdMds`] configuration with default parameters (`dims = 2`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new [`RdMds`] configuration targeting `dims` spectral dimensions.
    pub fn with_dims(dims: usize) -> Self {
        Self {
            dims,
            ..Self::default()
        }
    }

    /// Sets the number of embedding dimensions.
    pub fn dims(mut self, dims: usize) -> Self {
        self.dims = dims;
        self
    }

    /// Sets the zero-eigenvalue guard threshold `eig_eps`.
    pub fn eig_eps(mut self, eig_eps: f64) -> Self {
        self.eig_eps = eig_eps;
        self
    }

    /// Sets the eigensolver convergence tolerance.
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Sets the maximum number of thick-restart Lanczos cycles.
    pub fn max_iters(mut self, max_iters: usize) -> Self {
        self.max_iters = max_iters;
        self
    }

    /// Sets the deterministic PRNG seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Configures whether $D^{-1/2}$ degree normalization is applied.
    pub fn degree_normalize(mut self, degree_normalize: bool) -> Self {
        self.degree_normalize = degree_normalize;
        self
    }

    /// Sets an optional target standard deviation for 2D layouts.
    pub fn target_std(mut self, target_std: Option<f64>) -> Self {
        self.target_std = target_std;
        self
    }

    /// Computes a $d$-dimensional zero-guarded resistance-distance spectral embedding.
    pub fn embedding<W: Scalar, S: Scalar>(&self, graph: &CsrGraph<W>) -> CoreResult<LayoutNd<S>> {
        if self.dims == 0 {
            return Err(CoreError::invalid_input("RdMds dims must be >= 1"));
        }
        if !self.eig_eps.is_finite() || self.eig_eps < 0.0 {
            return Err(CoreError::invalid_input(
                "RdMds eig_eps must be finite and >= 0",
            ));
        }

        let n = graph.num_nodes();
        let mut layout = LayoutNd::zeros(n, self.dims)?;
        if n <= 1 {
            return Ok(layout);
        }

        let eig = smallest_sym_laplacian_eigenpairs(
            graph,
            self.dims,
            self.tol,
            self.max_iters,
            self.seed,
        )?;
        let m_count = eig.num_modes;
        let mut inv_sqrt_lambda = vec![0.0_f64; m_count];
        for m in 0..m_count {
            let lam = eig.lambdas[m];
            inv_sqrt_lambda[m] = if lam > self.eig_eps {
                lam.sqrt().recip()
            } else {
                0.0
            };
        }

        let coords = layout.coords_mut();
        for i in 0..n {
            let row_scale = if self.degree_normalize {
                eig.inv_sqrt_deg[i]
            } else {
                1.0
            };
            let out_row = &mut coords[i * self.dims..i * self.dims + m_count];
            for m in 0..m_count {
                let val = row_scale * eig.vectors[i * m_count + m] * inv_sqrt_lambda[m];
                out_row[m] = S::from_f64(val);
            }
        }
        Ok(layout)
    }

    /// Computes a 2D [`Layout2d`] resistance-distance embedding with optional `target_std` normalization.
    pub fn embedding_2d<W: Scalar, S: Scalar>(
        &self,
        graph: &CsrGraph<W>,
    ) -> CoreResult<Layout2d<S>> {
        if let Some(t_std) = self.target_std {
            if !t_std.is_finite() {
                return Err(CoreError::non_finite("RdMds::target_std"));
            }
            if t_std <= 0.0 {
                return Err(CoreError::invalid_input(
                    "target_std must be strictly positive",
                ));
            }
        }
        let cfg = Self { dims: 2, ..*self };
        let nd = cfg.embedding::<W, f64>(graph)?;
        let n = graph.num_nodes();
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let row = nd.node(i)?;
            x.push(S::from_f64(row[0]));
            y.push(S::from_f64(row[1]));
        }
        let mut layout = Layout2d::from_xy(x, y)?;
        if let Some(t_std) = self.target_std {
            normalize_layout_2d(&mut layout, t_std, self.seed)?;
        }
        Ok(layout)
    }
}

/// Functional shorthand for computing a zero-guarded [`RdMds`] spectral embedding.
pub fn rd_mds<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    dims: usize,
    eig_eps: f64,
    tol: f64,
    max_iters: usize,
    seed: u64,
) -> CoreResult<LayoutNd<S>> {
    RdMds::with_dims(dims)
        .eig_eps(eig_eps)
        .tol(tol)
        .max_iters(max_iters)
        .seed(seed)
        .embedding(graph)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn isotropic_std(layout: &Layout2d<f64>) -> f64 {
        let n = layout.len() as f64;
        let mx = layout.x().iter().sum::<f64>() / n;
        let my = layout.y().iter().sum::<f64>() / n;
        let sum_sq: f64 = layout
            .x()
            .iter()
            .zip(layout.y().iter())
            .map(|(&x, &y)| (x - mx).powi(2) + (y - my).powi(2))
            .sum();
        (sum_sq / (2.0 * n)).sqrt()
    }

    #[test]
    fn test_normalize_layout_2d_nan_collapsed_singleton_empty() {
        // NaN / Inf sanitization + exact isotropic std == 1.0
        let mut layout = Layout2d::from_xy(
            vec![f64::NAN, 10.0, 20.0, f64::INFINITY],
            vec![-5.0, f64::NEG_INFINITY, 15.0, 25.0],
        )
        .unwrap();
        normalize_layout_2d(&mut layout, 1.0, 42).unwrap();
        assert!(layout.is_all_finite());
        assert!((isotropic_std(&layout) - 1.0).abs() < 1e-12);

        // Collapsed layout falls back deterministically to Gaussian with target_std
        let mut collapsed = Layout2d::from_xy(vec![3.0; 8], vec![7.0; 8]).unwrap();
        normalize_layout_2d(&mut collapsed, 2.5, 7).unwrap();
        assert!(collapsed.is_all_finite());
        assert!((isotropic_std(&collapsed) - 2.5).abs() < 1e-12);

        // Singleton and empty
        let mut single = Layout2d::from_xy(vec![f64::NAN], vec![5.0]).unwrap();
        normalize_layout_2d(&mut single, 1.0, 42).unwrap();
        assert_eq!(single.x(), &[0.0]);
        assert_eq!(single.y(), &[0.0]);

        let mut empty = Layout2d::<f64>::zeros(0);
        normalize_layout_2d(&mut empty, 1.0, 42).unwrap();
    }

    #[test]
    fn test_pivot_mds_std_and_disconnected_safety() {
        // 8x8 grid (64 nodes)
        let mut edges = Vec::new();
        for r in 0..8 {
            for c in 0..8 {
                let u = (r * 8 + c) as u32;
                if c + 1 < 8 {
                    edges.push((u, u + 1));
                }
                if r + 1 < 8 {
                    edges.push((u, u + 8));
                }
            }
        }
        let grid = CsrGraph::<f64>::from_edges(64, &edges, None, false, false).unwrap();
        let layout: Layout2d<f64> = pivot_mds(&grid, &PivotMdsConfig::default()).unwrap();
        assert_eq!(layout.len(), 64);
        assert!(layout.is_all_finite());
        let std_val = isotropic_std(&layout);
        assert!(
            (std_val - 1.0).abs() < 1e-5,
            "PivotMds std expected 1.0, got {std_val}"
        );

        // Disconnected graph: two 4-cycles + 2 isolated vertices
        let disc = CsrGraph::<f64>::from_edges(
            10,
            &[
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
            ],
            None,
            false,
            false,
        )
        .unwrap();
        let disc_layout: Layout2d<f64> = pivot_mds(&disc, &PivotMdsConfig::default()).unwrap();
        assert!(disc_layout.is_all_finite());
        assert!((isotropic_std(&disc_layout) - 1.0).abs() < 1e-5);

        // Empty and singleton
        let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        assert_eq!(
            pivot_mds::<f64, f64>(&empty, &PivotMdsConfig::default())
                .unwrap()
                .len(),
            0
        );
        let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        let s_lay: Layout2d<f64> = pivot_mds(&single, &PivotMdsConfig::default()).unwrap();
        assert_eq!(s_lay.x(), &[0.0]);
        assert_eq!(s_lay.y(), &[0.0]);
    }

    #[test]
    fn test_koren_spectral_init_and_rd_and_classical_mds() {
        // Cycle C_20 embeds cleanly on a circle under Koren spectral init
        let edges: Vec<(u32, u32)> = (0..20).map(|i| (i, (i + 1) % 20)).collect();
        let cycle = CsrGraph::<f64>::from_edges(20, &edges, None, false, false).unwrap();

        let koren: Layout2d<f64> = koren_spectral_init(&cycle, Some(1.0), 1e-6, 300, 42).unwrap();
        assert!(koren.is_all_finite());
        assert!((isotropic_std(&koren) - 1.0).abs() < 1e-5);
        // Adjacent cycle vertices should have uniform chord length in 2D spectral init
        let mut step_dists = Vec::new();
        for i in 0..20 {
            let j = (i + 1) % 20;
            let dx = koren.x()[i] - koren.x()[j];
            let dy = koren.y()[i] - koren.y()[j];
            step_dists.push(dx.hypot(dy));
        }
        let mean_d = step_dists.iter().sum::<f64>() / 20.0;
        for &d in &step_dists {
            assert!((d - mean_d).abs() < 1e-4);
        }

        // RdMds on disconnected graph must remain finite (zero-eigenvalue guard)
        let disc = CsrGraph::<f64>::from_edges(
            6,
            &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)],
            None,
            false,
            false,
        )
        .unwrap();
        let rd: LayoutNd<f64> = rd_mds(&disc, 3, 1e-10, 1e-6, 200, 42).unwrap();
        assert!(rd.is_all_finite());
        // First mode corresponds to lambda_1 == 0 across components -> safely zeroed out
        for i in 0..6 {
            assert_eq!(rd.node(i).unwrap()[0], 0.0);
        }

        // Classical MDS recovers exact Euclidean coordinates of a 4-vertex unit square
        let mut d_mat = vec![0.0; 16];
        let pts = [(0.0_f64, 0.0_f64), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        for i in 0..4 {
            for j in 0..4 {
                d_mat[i * 4 + j] = (pts[i].0 - pts[j].0).hypot(pts[i].1 - pts[j].1);
            }
        }
        let cmds: LayoutNd<f64> = classical_mds_from_distances(&d_mat, 4, 2).unwrap();
        assert!(cmds.is_all_finite());
        for i in 0..4 {
            for j in (i + 1)..4 {
                let p_i = cmds.node(i).unwrap();
                let p_j = cmds.node(j).unwrap();
                let emb_d = (p_i[0] - p_j[0]).hypot(p_i[1] - p_j[1]);
                assert!(
                    (emb_d - d_mat[i * 4 + j]).abs() < 1e-10,
                    "Classical MDS distance mismatch ({i}, {j}): {emb_d} vs {}",
                    d_mat[i * 4 + j]
                );
            }
        }

        // Graph-based classical_mds on disconnected graph stays finite
        let cmds_disc: LayoutNd<f64> = classical_mds(&disc, 2).unwrap();
        assert!(cmds_disc.is_all_finite());

        // Weighted PivotMds + f32 Layout2d output + RdMds builder ergonomics
        let wg = CsrGraph::<f32>::from_edges(
            4,
            &[(0, 1), (1, 2), (2, 3), (3, 0)],
            Some(&[1.0_f32, 2.0, 1.0, 2.0]),
            false,
            false,
        )
        .unwrap();
        let lay_f32: Layout2d<f32> = pivot_mds(&wg, &PivotMdsConfig::default()).unwrap();
        assert!(lay_f32.is_all_finite());
        let rd_2d: Layout2d<f32> = RdMds::new()
            .dims(2)
            .eig_eps(1e-10)
            .tol(1e-6)
            .max_iters(200)
            .seed(42)
            .degree_normalize(true)
            .target_std(Some(1.0))
            .embedding_2d(&wg)
            .unwrap();
        assert!(rd_2d.is_all_finite());

        // Edgeless graph under koren_spectral_init with target_std=Some(1.0) hits finite fallback
        let iso = CsrGraph::<f64>::from_edges(8, &[], None, false, false).unwrap();
        let k_iso: Layout2d<f64> = koren_spectral_init(&iso, Some(1.0), 1e-6, 100, 42).unwrap();
        assert!(k_iso.is_all_finite());
        assert!((isotropic_std(&k_iso) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_robust_normalize_layout_2d_outliers_fallbacks_and_edge_cases() {
        // 1. 96 core points + 4 extreme outliers at (500, 500): core std remains ~1.0
        let mut xs = Vec::with_capacity(100);
        let mut ys = Vec::with_capacity(100);
        let mut rng = DeterministicRng::from_seed_u64(123);
        for _ in 0..96 {
            let (gx, gy) = next_gaussian_pair(&mut rng);
            xs.push(gx);
            ys.push(gy);
        }
        for _ in 0..4 {
            xs.push(500.0);
            ys.push(500.0);
        }
        let mut layout = Layout2d::from_xy(xs, ys).unwrap();
        robust_normalize_layout_2d(&mut layout, 1.0, 0.95, 42).unwrap();
        assert!(layout.is_all_finite());
        let core_sum_sq: f64 = layout.x()[..96]
            .iter()
            .zip(layout.y()[..96].iter())
            .map(|(&x, &y)| x * x + y * y)
            .sum();
        let core_std = (core_sum_sq / (2.0 * 96.0)).sqrt();
        assert!(
            (core_std - 1.0).abs() < 0.35,
            "expected core_std ~ 1.0 under 4% outliers, got {core_std}"
        );

        // 2. >95% duplicate hub core + 2 distinct outer nodes falls back cleanly to isotropic_std
        let mut hub_xs = vec![10.0_f64; 98];
        let mut hub_ys = vec![-5.0_f64; 98];
        hub_xs.extend([12.0, 8.0]);
        hub_ys.extend([-5.0, -5.0]);
        let mut hub_layout = Layout2d::from_xy(hub_xs, hub_ys).unwrap();
        robust_normalize_layout_2d(&mut hub_layout, 1.5, 0.95, 42).unwrap();
        assert!(hub_layout.is_all_finite());
        assert!((isotropic_std(&hub_layout) - 1.5).abs() < 1e-12);

        // 3. All collapsed -> deterministic Gaussian fallback
        let mut collapsed = Layout2d::from_xy(vec![3.0_f32; 8], vec![7.0_f32; 8]).unwrap();
        robust_normalize_layout_2d(&mut collapsed, 2.0, 0.95, 7).unwrap();
        assert!(collapsed.is_all_finite());
        let mut collapsed2 = Layout2d::from_xy(vec![3.0_f32; 8], vec![7.0_f32; 8]).unwrap();
        robust_normalize_layout_2d(&mut collapsed2, 2.0, 0.95, 7).unwrap();
        assert_eq!(collapsed, collapsed2);

        // 4. n = 0, 1, 2, NaN/Inf
        let mut empty = Layout2d::<f64>::zeros(0);
        robust_normalize_layout_2d(&mut empty, 1.0, 0.95, 42).unwrap();

        let mut single = Layout2d::from_xy(vec![f64::NAN], vec![5.0]).unwrap();
        robust_normalize_layout_2d(&mut single, 1.0, 0.95, 42).unwrap();
        assert_eq!(single.x(), &[0.0]);
        assert_eq!(single.y(), &[0.0]);

        let mut two = Layout2d::from_xy(vec![f64::INFINITY, 2.0], vec![0.0, f64::NAN]).unwrap();
        robust_normalize_layout_2d(&mut two, 1.0, 0.95, 42).unwrap();
        assert!(two.is_all_finite());

        // 5. Validation errors
        assert!(matches!(
            robust_normalize_layout_2d(&mut two, 0.0, 0.95, 42),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            robust_normalize_layout_2d(&mut two, f64::NAN, 0.95, 42),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            robust_normalize_layout_2d(&mut two, 1.0, 1.0, 42),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            robust_normalize_layout_2d(&mut two, 1.0, 0.0, 42),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            robust_normalize_layout_2d(&mut two, 1.0, f64::NAN, 42),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }

    #[test]
    fn test_pivot_mds_streaming_matches_all_pivots_classical_mds() {
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)];
        let g = CsrGraph::<f64>::from_edges(5, &edges, None, false, false).unwrap();
        let all_pivots: Vec<u32> = (0..5).collect();
        let pmds: Layout2d<f64> = pivot_mds_with_pivots(&g, &all_pivots, None, 42).unwrap();
        let cmds: Layout2d<f64> = classical_mds_2d(&g).unwrap();
        for i in 0..5 {
            for j in (i + 1)..5 {
                let dp = (pmds.x()[i] - pmds.x()[j]).hypot(pmds.y()[i] - pmds.y()[j]);
                let dc = (cmds.x()[i] - cmds.x()[j]).hypot(cmds.y()[i] - cmds.y()[j]);
                assert!((dp - dc).abs() < 1e-10, "pair ({i},{j}): {dp} vs {dc}");
            }
        }
    }

    fn isotropic_std_nd(layout: &LayoutNd<f64>) -> f64 {
        let n = layout.num_nodes();
        let d = layout.dims();
        let coords = layout.coords();
        let mut means = vec![0.0_f64; d];
        for i in 0..n {
            for m in 0..d {
                means[m] += coords[i * d + m];
            }
        }
        for mean in &mut means {
            *mean /= n as f64;
        }
        let mut sum_sq = 0.0_f64;
        for i in 0..n {
            for m in 0..d {
                let diff = coords[i * d + m] - means[m];
                sum_sq += diff * diff;
            }
        }
        (sum_sq / ((d * n) as f64)).sqrt()
    }

    /// Independent brute-force Floyd-Warshall + O(m^2 n) covariance PCA oracle for HDE.
    fn hde_naive_oracle(
        graph: &CsrGraph<f64>,
        pivots: &[u32],
        output_dims: usize,
    ) -> Vec<Vec<f64>> {
        let n = graph.num_nodes();
        let mut apsp = vec![vec![f64::INFINITY; n]; n];
        for (u, apsp_row) in apsp.iter_mut().enumerate() {
            apsp_row[u] = 0.0;
            if let Some(ws) = graph.neighbor_weights(u) {
                for (&v, &w) in graph.neighbors(u).iter().zip(ws.iter()) {
                    let vi = v as usize;
                    if w < apsp_row[vi] {
                        apsp_row[vi] = w;
                    }
                }
            } else {
                for &v in graph.neighbors(u) {
                    let vi = v as usize;
                    if 1.0 < apsp_row[vi] {
                        apsp_row[vi] = 1.0;
                    }
                }
            }
        }
        for k in 0..n {
            for i in 0..n {
                let dik = apsp[i][k];
                if !dik.is_finite() {
                    continue;
                }
                for j in 0..n {
                    let cand = dik + apsp[k][j];
                    if cand < apsp[i][j] {
                        apsp[i][j] = cand;
                    }
                }
            }
        }

        // Deduplicate pivots preserving order.
        let mut uniq = Vec::new();
        for &p in pivots {
            if !uniq.contains(&p) {
                uniq.push(p);
            }
        }
        let m = uniq.len();
        let mut x_hat = vec![vec![0.0_f64; n]; m];
        for (i, &p) in uniq.iter().enumerate() {
            let row = &apsp[p as usize];
            let mut max_fin = 0.0_f64;
            let mut has_fin = false;
            let mut has_inf = false;
            for &d in row {
                if d.is_finite() {
                    has_fin = true;
                    max_fin = max_fin.max(d);
                } else {
                    has_inf = true;
                }
            }
            let fill = if has_fin { max_fin + 1.0 } else { 0.0 };
            let mut sum = 0.0;
            for v in 0..n {
                let val = if has_inf && !row[v].is_finite() {
                    fill
                } else {
                    row[v]
                };
                x_hat[i][v] = val;
                sum += val;
            }
            let mean = sum / (n as f64);
            for val in &mut x_hat[i] {
                *val -= mean;
            }
        }

        let mut cov = vec![0.0_f64; m * m];
        for a in 0..m {
            for b in 0..m {
                let mut s = 0.0;
                for v in 0..n {
                    s += x_hat[a][v] * x_hat[b][v];
                }
                cov[a * m + b] = s / (n as f64);
            }
        }
        let (evals, evecs) = symmetric_eigen_jacobi(&cov, m, 200, 1e-14).unwrap();
        let used = output_dims.min(m);
        let lam0 = evals[m - 1];
        let mut y = vec![vec![0.0_f64; output_dims]; n];
        if lam0 > 1e-300 {
            for d in 0..used {
                let idx = m - 1 - d;
                if d >= 1 && evals[idx] <= 1e-12 * lam0 {
                    continue;
                }
                for (v, y_row) in y.iter_mut().enumerate() {
                    let mut acc = 0.0;
                    for p in 0..m {
                        acc += x_hat[p][v] * evecs[p * m + idx];
                    }
                    y_row[d] = acc;
                }
            }
        }
        y
    }

    fn axis_cos_abs(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let na: f64 = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            (dot / (na * nb)).abs()
        }
    }

    fn pairwise_euclidean_nd(coords: &[Vec<f64>]) -> Vec<f64> {
        let n = coords.len();
        let mut out = Vec::with_capacity(n * (n.saturating_sub(1)) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let mut max_abs = 0.0_f64;
                for (&a, &b) in coords[i].iter().zip(coords[j].iter()) {
                    let d = (a - b).abs();
                    if d > max_abs {
                        max_abs = d;
                    }
                }
                if max_abs == 0.0 || !max_abs.is_finite() {
                    out.push(max_abs);
                    continue;
                }
                let (pre, inv) = if max_abs < 1e-300 {
                    (1e150_f64, 1.0 / (max_abs * 1e150_f64))
                } else if max_abs > 1e300 {
                    (1e-150_f64, 1.0 / (max_abs * 1e-150_f64))
                } else {
                    (1.0_f64, 1.0 / max_abs)
                };
                let d2: f64 = coords[i]
                    .iter()
                    .zip(coords[j].iter())
                    .map(|(&a, &b)| {
                        let d = ((a - b) * pre) * inv;
                        d * d
                    })
                    .sum();
                out.push(max_abs * d2.sqrt());
            }
        }
        out
    }

    #[test]
    fn test_hde_dense_naive_oracle() {
        // 1. Asymmetric unweighted 10-node graph with irregular chords
        let asym_edges = vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (0, 3),
            (1, 5),
            (2, 8),
            (4, 9),
            (0, 7),
        ];
        let g_unw = CsrGraph::<f64>::from_edges(10, &asym_edges, None, false, false).unwrap();
        let pivots = vec![0, 4, 7, 2];
        let cfg = HdeConfig {
            dimensions: 4,
            output_dims: 2,
            seed: 42,
            tol: 1e-12,
            max_sweeps: 200,
            normalize: false,
        };
        let lay = hde_with_pivots::<f64, f64>(&g_unw, &pivots, &cfg).unwrap();
        let oracle = hde_naive_oracle(&g_unw, &pivots, 2);
        let ox: Vec<f64> = oracle.iter().map(|r| r[0]).collect();
        let oy: Vec<f64> = oracle.iter().map(|r| r[1]).collect();
        assert!(axis_cos_abs(lay.x(), &ox) > 1.0 - 1e-12);
        assert!(axis_cos_abs(lay.y(), &oy) > 1.0 - 1e-12);
        let lay_coords: Vec<Vec<f64>> = (0..10).map(|i| vec![lay.x()[i], lay.y()[i]]).collect();
        for (a, b) in pairwise_euclidean_nd(&lay_coords)
            .iter()
            .zip(pairwise_euclidean_nd(&oracle).iter())
        {
            assert!((a - b).abs() < 1e-10, "pairwise mismatch: {a} vs {b}");
        }

        // 2. Asymmetric positive weighted graph
        let w_edges = vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (0, 4),
            (1, 6),
            (2, 5),
            (3, 7),
        ];
        let weights = vec![0.5, 1.3, 2.7, 0.9, 3.4, 1.1, 2.2, 4.0, 1.7, 0.8, 2.9];
        let g_w = CsrGraph::<f64>::from_edges(8, &w_edges, Some(&weights), false, false).unwrap();
        let w_pivots = select_maxmin_sp_pivots(&g_w, 5, 11, 1).unwrap();
        let cfg_w = HdeConfig {
            dimensions: 5,
            seed: 11,
            tol: 1e-12,
            max_sweeps: 200,
            normalize: false,
            ..HdeConfig::default()
        };
        let lay_w = hde::<f64, f64>(&g_w, &cfg_w).unwrap();
        let oracle_w = hde_naive_oracle(&g_w, &w_pivots, 2);
        let lay_w_coords: Vec<Vec<f64>> =
            (0..8).map(|i| vec![lay_w.x()[i], lay_w.y()[i]]).collect();
        for (a, b) in pairwise_euclidean_nd(&lay_w_coords)
            .iter()
            .zip(pairwise_euclidean_nd(&oracle_w).iter())
        {
            assert!((a - b).abs() < 1e-10, "weighted mismatch: {a} vs {b}");
        }

        // Normalized variant retains finite std=1.0
        let lay_w_norm = hde::<f64, f64>(
            &g_w,
            &HdeConfig {
                normalize: true,
                ..cfg_w
            },
        )
        .unwrap();
        assert!((isotropic_std(&lay_w_norm) - 1.0).abs() < 1e-12);

        // 3. Symmetric 4x5 grid (pairwise distance invariant under possible basis rotation)
        let mut grid_edges = Vec::new();
        for r in 0..4 {
            for c in 0..5 {
                let u = (r * 5 + c) as u32;
                if c + 1 < 5 {
                    grid_edges.push((u, u + 1));
                }
                if r + 1 < 4 {
                    grid_edges.push((u, u + 5));
                }
            }
        }
        let grid = CsrGraph::<f64>::from_edges(20, &grid_edges, None, false, false).unwrap();
        let gp = select_maxmin_sp_pivots(&grid, 8, 42, 1).unwrap();
        let grid_cfg = HdeConfig {
            dimensions: 8,
            tol: 1e-12,
            max_sweeps: 200,
            normalize: false,
            ..HdeConfig::default()
        };
        let grid_lay = hde::<f64, f64>(&grid, &grid_cfg).unwrap();
        let grid_oracle = hde_naive_oracle(&grid, &gp, 2);
        let grid_coords: Vec<Vec<f64>> = (0..20)
            .map(|i| vec![grid_lay.x()[i], grid_lay.y()[i]])
            .collect();
        for (a, b) in pairwise_euclidean_nd(&grid_coords)
            .iter()
            .zip(pairwise_euclidean_nd(&grid_oracle).iter())
        {
            assert!((a - b).abs() < 1e-10, "grid mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_hde_topologies_path_tree_grid_cycle() {
        // 1. Path P_8: rank-1 -> y identically 0.0 for both normalize=false and normalize=true
        let p8_edges: Vec<(u32, u32)> = (0..7).map(|i| (i, i + 1)).collect();
        let p8 = CsrGraph::<f64>::from_edges(8, &p8_edges, None, false, false).unwrap();
        for normalize in [false, true] {
            let lay = hde_with_pivots::<f64, f64>(
                &p8,
                &[0, 7],
                &HdeConfig {
                    normalize,
                    ..HdeConfig::default()
                },
            )
            .unwrap();
            assert!(lay.y().iter().all(|&y| y == 0.0));
            // Strictly monotonic along X
            let increasing = lay.x().windows(2).all(|w| w[0] < w[1]);
            let decreasing = lay.x().windows(2).all(|w| w[0] > w[1]);
            assert!(increasing || decreasing);
            if normalize {
                assert!((isotropic_std(&lay) - 1.0).abs() < 1e-12);
            }
        }
        let p8_auto = hde::<f64, f64>(&p8, &HdeConfig::default()).unwrap();
        assert!(p8_auto.is_all_finite());
        assert!((isotropic_std(&p8_auto) - 1.0).abs() < 1e-12);
        // With interior pivots (dimensions >= 8), |v - p| creates a U-arch along X; verify X ordering.
        let inc_auto = p8_auto.x().windows(2).all(|w| w[0] < w[1]);
        let dec_auto = p8_auto.x().windows(2).all(|w| w[0] > w[1]);
        assert!(inc_auto || dec_auto);

        // 2. Tree with pivot-free subtree (Harel & Koren §4 Fig. 8):
        // Edges: 0-1, 1-2, 1-3, 0-4, 4-5; pivots=[0, 4] -> leaves 2 and 3 collapse identically.
        let tree = CsrGraph::<f64>::from_edges(
            6,
            &[(0, 1), (1, 2), (1, 3), (0, 4), (4, 5)],
            None,
            false,
            false,
        )
        .unwrap();
        let tree_lay = hde_with_pivots::<f64, f64>(&tree, &[0, 4], &HdeConfig::default()).unwrap();
        assert!((tree_lay.x()[2] - tree_lay.x()[3]).abs() < 1e-12);
        assert!((tree_lay.y()[2] - tree_lay.y()[3]).abs() < 1e-12);

        // 3. Cycle C_12
        let c12_edges: Vec<(u32, u32)> = (0..12).map(|i| (i, (i + 1) % 12)).collect();
        let c12 = CsrGraph::<f64>::from_edges(12, &c12_edges, None, false, false).unwrap();
        let c12_lay = hde::<f64, f64>(&c12, &HdeConfig::default()).unwrap();
        assert!(c12_lay.is_all_finite());
        assert!((isotropic_std(&c12_lay) - 1.0).abs() < 1e-12);
        let c12_unnorm = hde::<f64, f64>(
            &c12,
            &HdeConfig {
                tol: 1e-12,
                max_sweeps: 200,
                normalize: false,
                ..HdeConfig::default()
            },
        )
        .unwrap();
        let c12_pivots = select_maxmin_sp_pivots(&c12, 50, 42, 1).unwrap();
        let c12_oracle = hde_naive_oracle(&c12, &c12_pivots, 2);
        let c12_coords: Vec<Vec<f64>> = (0..12)
            .map(|i| vec![c12_unnorm.x()[i], c12_unnorm.y()[i]])
            .collect();
        for (a, b) in pairwise_euclidean_nd(&c12_coords)
            .iter()
            .zip(pairwise_euclidean_nd(&c12_oracle).iter())
        {
            assert!((a - b).abs() < 1e-10);
        }

        // 4. 8x8 regular grid
        let mut grid_edges = Vec::new();
        for r in 0..8 {
            for c in 0..8 {
                let u = (r * 8 + c) as u32;
                if c + 1 < 8 {
                    grid_edges.push((u, u + 1));
                }
                if r + 1 < 8 {
                    grid_edges.push((u, u + 8));
                }
            }
        }
        let grid8 = CsrGraph::<f64>::from_edges(64, &grid_edges, None, false, false).unwrap();
        let g8_lay = hde::<f64, f64>(&grid8, &HdeConfig::default()).unwrap();
        assert_eq!(g8_lay.len(), 64);
        assert!(g8_lay.is_all_finite());
        assert!((isotropic_std(&g8_lay) - 1.0).abs() < 1e-12);
        let mean_edge_len: f64 = grid_edges
            .iter()
            .map(|&(u, v)| {
                let ui = u as usize;
                let vi = v as usize;
                (g8_lay.x()[ui] - g8_lay.x()[vi]).hypot(g8_lay.y()[ui] - g8_lay.y()[vi])
            })
            .sum::<f64>()
            / (grid_edges.len() as f64);
        assert!(
            mean_edge_len < 0.7,
            "expected short grid edges, got {mean_edge_len}"
        );
    }

    #[test]
    fn test_hde_disconnected_and_isolated_nodes() {
        // Mixed components: triangle {0,1,2} + edge {3,4} + isolated {5,6}
        let mixed =
            CsrGraph::<f64>::from_edges(7, &[(0, 1), (1, 2), (2, 0), (3, 4)], None, false, false)
                .unwrap();
        let pivots = vec![0, 3, 5];
        let cfg_unnorm = HdeConfig {
            tol: 1e-12,
            max_sweeps: 200,
            normalize: false,
            ..HdeConfig::default()
        };
        let lay_unnorm = hde_with_pivots::<f64, f64>(&mixed, &pivots, &cfg_unnorm).unwrap();
        let oracle = hde_naive_oracle(&mixed, &pivots, 2);
        let coords: Vec<Vec<f64>> = (0..7)
            .map(|i| vec![lay_unnorm.x()[i], lay_unnorm.y()[i]])
            .collect();
        for (a, b) in pairwise_euclidean_nd(&coords)
            .iter()
            .zip(pairwise_euclidean_nd(&oracle).iter())
        {
            assert!((a - b).abs() < 1e-10);
        }

        let lay_norm = hde::<f64, f64>(&mixed, &HdeConfig::default()).unwrap();
        assert!(lay_norm.is_all_finite());
        assert!((isotropic_std(&lay_norm) - 1.0).abs() < 1e-12);

        // All-isolated edgeless graph (n = 5)
        let edgeless = CsrGraph::<f64>::from_edges(5, &[], None, false, false).unwrap();
        let edge_unnorm = hde_with_pivots::<f64, f64>(&edgeless, &[0, 2, 4], &cfg_unnorm).unwrap();
        let edge_oracle = hde_naive_oracle(&edgeless, &[0, 2, 4], 2);
        let edge_coords: Vec<Vec<f64>> = (0..5)
            .map(|i| vec![edge_unnorm.x()[i], edge_unnorm.y()[i]])
            .collect();
        for (a, b) in pairwise_euclidean_nd(&edge_coords)
            .iter()
            .zip(pairwise_euclidean_nd(&edge_oracle).iter())
        {
            assert!((a - b).abs() < 1e-10);
        }
        let edge_norm = hde::<f64, f64>(&edgeless, &HdeConfig::default()).unwrap();
        assert!(edge_norm.is_all_finite());
        assert!((isotropic_std(&edge_norm) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_hde_directed_and_reciprocal_edges() {
        let undir_edges = vec![(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (3, 4)];
        let mut recip_edges = Vec::new();
        let mut recip_weights = Vec::new();
        let undir_weights = vec![1.0, 2.5, 1.5, 3.0, 4.0, 0.75];
        for (&(u, v), &w) in undir_edges.iter().zip(undir_weights.iter()) {
            recip_edges.push((u, v));
            recip_edges.push((v, u));
            recip_weights.push(w);
            recip_weights.push(w);
        }

        // Unweighted reciprocal vs undirected: bit-exact
        let gu = CsrGraph::<f64>::from_edges(5, &undir_edges, None, false, false).unwrap();
        let gr = CsrGraph::<f64>::from_edges(5, &recip_edges, None, true, true).unwrap();
        let cfg = HdeConfig::default();
        let lu = hde::<f64, f64>(&gu, &cfg).unwrap();
        let lr = hde::<f64, f64>(&gr, &cfg).unwrap();
        assert_eq!(
            lu.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            lr.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            lu.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            lr.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        // Weighted reciprocal vs undirected: bit-exact
        let gwu = CsrGraph::<f64>::from_edges(5, &undir_edges, Some(&undir_weights), false, false)
            .unwrap();
        let gwr =
            CsrGraph::<f64>::from_edges(5, &recip_edges, Some(&recip_weights), true, true).unwrap();
        let lwu = hde_with_pivots::<f64, f64>(&gwu, &[0, 3, 4], &cfg).unwrap();
        let lwr = hde_with_pivots::<f64, f64>(&gwr, &[0, 3, 4], &cfg).unwrap();
        assert_eq!(
            lwu.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            lwr.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            lwu.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            lwr.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        // One-way DAG 0 -> 1 -> 2 -> 3 with pivots = &[3, 0]
        let dag =
            CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3)], None, true, false).unwrap();
        let dag_cfg = HdeConfig {
            tol: 1e-12,
            max_sweeps: 200,
            normalize: false,
            ..HdeConfig::default()
        };
        let dag_lay = hde_with_pivots::<f64, f64>(&dag, &[3, 0], &dag_cfg).unwrap();
        assert!(dag_lay.is_all_finite());
        let dag_oracle = hde_naive_oracle(&dag, &[3, 0], 2);
        let dag_coords: Vec<Vec<f64>> = (0..4)
            .map(|i| vec![dag_lay.x()[i], dag_lay.y()[i]])
            .collect();
        for (a, b) in pairwise_euclidean_nd(&dag_coords)
            .iter()
            .zip(pairwise_euclidean_nd(&dag_oracle).iter())
        {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_hde_weighted_positive_and_invalid() {
        // Shortcut 0-2 (w=10.0) longer than 0-1-2 (w=1.0, 2.0) -> Dijkstra distance = 3.0
        let wg = CsrGraph::<f64>::from_edges(
            4,
            &[(0, 1), (1, 2), (0, 2), (2, 3)],
            Some(&[1.0, 2.0, 10.0, 1.5]),
            false,
            false,
        )
        .unwrap();
        let cfg_unnorm = HdeConfig {
            tol: 1e-12,
            max_sweeps: 200,
            normalize: false,
            ..HdeConfig::default()
        };
        let lay = hde_with_pivots::<f64, f64>(&wg, &[0, 1, 3], &cfg_unnorm).unwrap();
        let oracle = hde_naive_oracle(&wg, &[0, 1, 3], 2);
        let coords: Vec<Vec<f64>> = (0..4).map(|i| vec![lay.x()[i], lay.y()[i]]).collect();
        for (a, b) in pairwise_euclidean_nd(&coords)
            .iter()
            .zip(pairwise_euclidean_nd(&oracle).iter())
        {
            assert!((a - b).abs() < 1e-10);
        }

        // Tiny weights (1e-8) with normalize=false preserve rank-2 axes via relative threshold
        let tiny = CsrGraph::<f64>::from_edges(
            4,
            &[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
            Some(&[1e-8, 2e-8, 1.5e-8, 2.5e-8, 3e-8]),
            false,
            false,
        )
        .unwrap();
        let tiny_lay = hde_with_pivots::<f64, f64>(&tiny, &[0, 1, 2], &cfg_unnorm).unwrap();
        assert!(tiny_lay.is_all_finite());
        assert!(tiny_lay.x().iter().any(|&v| v.abs() > 1e-12));
        assert!(tiny_lay.y().iter().any(|&v| v.abs() > 1e-12));

        // Huge weights (1e150) test covariance scaling overflow guard
        let huge = CsrGraph::<f64>::from_edges(
            4,
            &[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
            Some(&[1e150, 2e150, 1.5e150, 2.5e150, 3e150]),
            false,
            false,
        )
        .unwrap();
        let huge_norm = hde::<f64, f64>(&huge, &HdeConfig::default()).unwrap();
        assert!(huge_norm.is_all_finite());
        assert!((isotropic_std(&huge_norm) - 1.0).abs() < 1e-12);
        let huge_unnorm = hde_with_pivots::<f64, f64>(&huge, &[0, 1, 2], &cfg_unnorm).unwrap();
        assert!(huge_unnorm.is_all_finite());
        assert!(huge_unnorm.x().iter().any(|&v| v.abs() > 1e140));

        // Zero and negative weights rejected with InvalidInput
        for bad_w in [0.0_f64, -1.0_f64] {
            let bad =
                CsrGraph::<f64>::from_edges(3, &[(0, 1)], Some(&[bad_w]), false, false).unwrap();
            assert!(matches!(
                hde::<f64, f64>(&bad, &HdeConfig::default()),
                Err(CoreError::InvalidInput(_))
            ));
            assert!(matches!(
                hde_with_pivots::<f64, f64>(&bad, &[0, 1], &HdeConfig::default()),
                Err(CoreError::InvalidInput(_))
            ));
            assert!(matches!(
                hde_nd::<f64, f64>(&bad, &HdeConfig::default()),
                Err(CoreError::InvalidInput(_))
            ));
            assert!(matches!(
                hde_nd_with_pivots::<f64, f64>(&bad, &[0, 1], &HdeConfig::default()),
                Err(CoreError::InvalidInput(_))
            ));
        }

        // CsrGraph rejects NaN/Inf weights at construction
        assert!(matches!(
            CsrGraph::<f64>::from_edges(2, &[(0, 1)], Some(&[f64::NAN]), false, false),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            CsrGraph::<f64>::from_edges(2, &[(0, 1)], Some(&[f64::INFINITY]), false, false),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }

    #[test]
    fn test_hde_degenerate_and_validation_errors() {
        let cfg = HdeConfig::default();
        let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        assert_eq!(hde::<f64, f64>(&empty, &cfg).unwrap().len(), 0);
        assert_eq!(
            hde_with_pivots::<f64, f64>(&empty, &[], &cfg)
                .unwrap()
                .len(),
            0
        );
        assert!(matches!(
            hde_with_pivots::<f64, f64>(&empty, &[0], &cfg),
            Err(CoreError::IndexOutOfBounds { .. })
        ));

        let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        for normalize in [true, false] {
            let s_cfg = HdeConfig { normalize, ..cfg };
            let s1 = hde::<f64, f64>(&single, &s_cfg).unwrap();
            assert_eq!(s1.x(), &[0.0]);
            assert_eq!(s1.y(), &[0.0]);
            let s2 = hde_with_pivots::<f64, f64>(&single, &[0], &s_cfg).unwrap();
            assert_eq!(s2.x(), &[0.0]);
            assert_eq!(s2.y(), &[0.0]);
        }

        // N = 2 edge 0-1
        let two = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
        let l2 = hde::<f64, f64>(&two, &cfg).unwrap();
        assert_eq!(l2.y(), &[0.0, 0.0]);
        assert!((l2.x()[0] + l2.x()[1]).abs() < 1e-12);
        assert!(l2.x()[0].abs() > 0.5);

        // dimensions > n
        let g5 = CsrGraph::<f64>::from_edges(
            5,
            &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
            None,
            false,
            false,
        )
        .unwrap();
        let l5 = hde::<f64, f64>(&g5, &cfg).unwrap();
        assert!(l5.is_all_finite());
        assert!((isotropic_std(&l5) - 1.0).abs() < 1e-12);

        // Duplicate pivots produce bit-exact results to deduplicated pivots
        let l_dup = hde_with_pivots::<f64, f64>(&g5, &[0, 2, 0, 4, 2], &cfg).unwrap();
        let l_uniq = hde_with_pivots::<f64, f64>(&g5, &[0, 2, 4], &cfg).unwrap();
        assert_eq!(
            l_dup.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l_uniq.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            l_dup.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l_uniq.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        // Single pivot m_eff = 1 -> y identically 0.0
        for normalize in [false, true] {
            let s_cfg = HdeConfig { normalize, ..cfg };
            let l1a = hde_with_pivots::<f64, f64>(&g5, &[1], &s_cfg).unwrap();
            let l1b = hde_with_pivots::<f64, f64>(&g5, &[1, 1, 1], &s_cfg).unwrap();
            assert!(l1a.y().iter().all(|&y| y == 0.0));
            assert_eq!(l1a.x(), l1b.x());
            assert_eq!(l1a.y(), l1b.y());
        }

        // Empty pivots on n >= 1 and out-of-range pivot
        assert!(matches!(
            hde_with_pivots::<f64, f64>(&g5, &[], &cfg),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            hde_with_pivots::<f64, f64>(&g5, &[0, 5], &cfg),
            Err(CoreError::IndexOutOfBounds { .. })
        ));

        // Config validation errors even when n == 0
        for graph in [&empty, &g5] {
            assert!(matches!(
                hde::<f64, f64>(
                    graph,
                    &HdeConfig {
                        dimensions: 0,
                        ..cfg
                    }
                ),
                Err(CoreError::InvalidInput(_))
            ));
            for bad_out in [0usize, 1, 3] {
                assert!(matches!(
                    hde::<f64, f64>(
                        graph,
                        &HdeConfig {
                            output_dims: bad_out,
                            ..cfg
                        }
                    ),
                    Err(CoreError::InvalidInput(_))
                ));
                assert!(matches!(
                    hde_with_pivots::<f64, f64>(
                        graph,
                        &[0],
                        &HdeConfig {
                            output_dims: bad_out,
                            ..cfg
                        }
                    ),
                    Err(CoreError::InvalidInput(_))
                ));
            }
            assert!(matches!(
                hde_nd::<f64, f64>(
                    graph,
                    &HdeConfig {
                        output_dims: 0,
                        ..cfg
                    }
                ),
                Err(CoreError::InvalidInput(_))
            ));
            for bad_tol in [0.0, -1.0] {
                assert!(matches!(
                    hde::<f64, f64>(
                        graph,
                        &HdeConfig {
                            tol: bad_tol,
                            ..cfg
                        }
                    ),
                    Err(CoreError::InvalidInput(_))
                ));
            }
            for nf_tol in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                assert!(matches!(
                    hde::<f64, f64>(graph, &HdeConfig { tol: nf_tol, ..cfg }),
                    Err(CoreError::NonFiniteValue { .. })
                ));
            }
            assert!(matches!(
                hde::<f64, f64>(
                    graph,
                    &HdeConfig {
                        max_sweeps: 0,
                        ..cfg
                    }
                ),
                Err(CoreError::InvalidInput(_))
            ));
        }
    }

    #[test]
    fn test_hde_f32_f64_and_determinism() {
        let edges = vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0),
            (0, 3),
            (1, 4),
        ];
        let w64 = vec![1.0_f64, 1.5, 2.0, 1.2, 0.8, 2.5, 1.7, 1.9];
        let w32: Vec<f32> = w64.iter().map(|&v| v as f32).collect();

        let g64 = CsrGraph::<f64>::from_edges(6, &edges, Some(&w64), false, false).unwrap();
        let g32 = CsrGraph::<f32>::from_edges(6, &edges, Some(&w32), false, false).unwrap();
        let cfg = HdeConfig {
            dimensions: 4,
            seed: 99,
            ..HdeConfig::default()
        };

        let l64_64_a = hde::<f64, f64>(&g64, &cfg).unwrap();
        let l64_64_b = hde::<f64, f64>(&g64, &cfg).unwrap();
        assert_eq!(
            l64_64_a.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l64_64_b.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            l64_64_a.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l64_64_b.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert!(l64_64_a.is_all_finite());
        assert!((isotropic_std(&l64_64_a) - 1.0).abs() < 1e-12);

        let l64_32 = hde::<f64, f32>(&g64, &cfg).unwrap();
        let l32_64 = hde::<f32, f64>(&g32, &cfg).unwrap();
        let l32_32 = hde::<f32, f32>(&g32, &cfg).unwrap();
        assert!(l64_32.is_all_finite());
        assert!(l32_64.is_all_finite());
        assert!(l32_32.is_all_finite());
        assert!((isotropic_std(&l32_64) - 1.0).abs() < 1e-6);

        for i in 0..6 {
            assert!((f64::from(l64_32.x()[i]) - l64_64_a.x()[i]).abs() < 1e-5);
            assert!((f64::from(l64_32.y()[i]) - l64_64_a.y()[i]).abs() < 1e-5);
            assert!((f64::from(l32_32.x()[i]) - l32_64.x()[i]).abs() < 1e-5);
            assert!((f64::from(l32_32.y()[i]) - l32_64.y()[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_hde_nd_and_large_u32_branch() {
        let edges = vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (0, 3),
            (1, 6),
            (2, 7),
            (4, 8),
        ];
        let g = CsrGraph::<f64>::from_edges(9, &edges, None, false, false).unwrap();

        // 1. output_dims == 2 matches hde bit-for-bit
        for normalize in [false, true] {
            let cfg2 = HdeConfig {
                dimensions: 5,
                output_dims: 2,
                seed: 7,
                tol: 1e-12,
                max_sweeps: 200,
                normalize,
            };
            let lay2 = hde::<f64, f64>(&g, &cfg2).unwrap();
            let lay_nd2 = hde_nd::<f64, f64>(&g, &cfg2).unwrap();
            assert_eq!(lay_nd2.dims(), 2);
            for i in 0..9 {
                let row = lay_nd2.node(i).unwrap();
                assert_eq!(row[0].to_bits(), lay2.x()[i].to_bits());
                assert_eq!(row[1].to_bits(), lay2.y()[i].to_bits());
            }
        }

        // 2. output_dims == 3 matches dense 3D oracle
        let cfg3_unnorm = HdeConfig {
            dimensions: 5,
            output_dims: 3,
            seed: 7,
            tol: 1e-12,
            max_sweeps: 200,
            normalize: false,
        };
        let pivots = select_maxmin_sp_pivots(&g, 5, 7, 1).unwrap();
        let lay3 = hde_nd::<f64, f64>(&g, &cfg3_unnorm).unwrap();
        let oracle3 = hde_naive_oracle(&g, &pivots, 3);
        let lay3_coords: Vec<Vec<f64>> = (0..9).map(|i| lay3.node(i).unwrap().to_vec()).collect();
        for (a, b) in pairwise_euclidean_nd(&lay3_coords)
            .iter()
            .zip(pairwise_euclidean_nd(&oracle3).iter())
        {
            assert!((a - b).abs() < 1e-10);
        }
        let lay3_norm = hde_nd::<f64, f64>(
            &g,
            &HdeConfig {
                normalize: true,
                ..cfg3_unnorm
            },
        )
        .unwrap();
        assert!((isotropic_std_nd(&lay3_norm) - 1.0).abs() < 1e-12);

        // 3. m_eff < output_dims zero-pads remaining dimensions when normalize=false
        let lay_pad = hde_nd_with_pivots::<f64, f64>(
            &g,
            &[0, 4],
            &HdeConfig {
                output_dims: 4,
                normalize: false,
                ..HdeConfig::default()
            },
        )
        .unwrap();
        for i in 0..9 {
            let row = lay_pad.node(i).unwrap();
            assert_eq!(row[2], 0.0);
            assert_eq!(row[3], 0.0);
        }

        // 4. n > u16::MAX exercises the u32 BFS storage path cleanly
        let n_big = (u16::MAX as usize) + 5;
        let big_edges: Vec<(u32, u32)> = (0..(n_big as u32 - 1)).map(|i| (i, i + 1)).collect();
        let g_big = CsrGraph::<f64>::from_edges(n_big, &big_edges, None, false, false).unwrap();
        let big_lay =
            hde_with_pivots::<f64, f64>(&g_big, &[0, (n_big - 1) as u32], &HdeConfig::default())
                .unwrap();
        assert_eq!(big_lay.len(), n_big);
        assert!(big_lay.is_all_finite());
        assert!(big_lay.y().iter().all(|&y| y == 0.0));
        assert!((isotropic_std(&big_lay) - 1.0).abs() < 1e-10);
    }

    fn max_rel_pairwise_err(a: &[f64], b: &[f64], b_scale: f64) -> f64 {
        let mut max_err = 0.0_f64;
        for (&da, &db) in a.iter().zip(b.iter()) {
            let expected = db * b_scale;
            let denom = expected.abs().max(1e-300);
            let rel = (da - expected).abs() / denom;
            if rel > max_err {
                max_err = rel;
            }
        }
        max_err
    }

    #[test]
    fn test_hde_scale_invariance_and_extreme_weights() {
        let edges = vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0),
            (0, 3),
            (1, 4),
        ];
        let base_w = vec![1.0_f64, 1.5, 2.2, 0.8, 3.1, 1.7, 2.6, 1.3];
        let pivots = vec![0u32, 2, 4, 1];

        let base_g = CsrGraph::<f64>::from_edges(6, &edges, Some(&base_w), false, false).unwrap();
        let cfg2_norm = HdeConfig {
            dimensions: 4,
            output_dims: 2,
            seed: 42,
            tol: 1e-12,
            max_sweeps: 200,
            normalize: true,
        };
        let cfg2_unnorm = HdeConfig {
            normalize: false,
            ..cfg2_norm
        };
        let cfg3_norm = HdeConfig {
            output_dims: 3,
            ..cfg2_norm
        };
        let cfg3_unnorm = HdeConfig {
            output_dims: 3,
            normalize: false,
            ..cfg2_norm
        };

        let ref2_norm = hde_with_pivots::<f64, f64>(&base_g, &pivots, &cfg2_norm).unwrap();
        let ref2_unnorm = hde_with_pivots::<f64, f64>(&base_g, &pivots, &cfg2_unnorm).unwrap();
        let ref3_norm = hde_nd_with_pivots::<f64, f64>(&base_g, &pivots, &cfg3_norm).unwrap();
        let ref3_unnorm = hde_nd_with_pivots::<f64, f64>(&base_g, &pivots, &cfg3_unnorm).unwrap();

        let ref2_norm_pw = pairwise_euclidean_nd(
            &(0..6)
                .map(|i| vec![ref2_norm.x()[i], ref2_norm.y()[i]])
                .collect::<Vec<_>>(),
        );
        let ref2_unnorm_pw = pairwise_euclidean_nd(
            &(0..6)
                .map(|i| vec![ref2_unnorm.x()[i], ref2_unnorm.y()[i]])
                .collect::<Vec<_>>(),
        );
        let ref3_norm_pw = pairwise_euclidean_nd(
            &(0..6)
                .map(|i| ref3_norm.node(i).unwrap().to_vec())
                .collect::<Vec<_>>(),
        );
        let ref3_unnorm_pw = pairwise_euclidean_nd(
            &(0..6)
                .map(|i| ref3_unnorm.node(i).unwrap().to_vec())
                .collect::<Vec<_>>(),
        );

        let scales = [
            1e-200, 1e-150, 1e-100, 1e-20, 1e-13, 1.0, 1e80, 1e150, 1e200,
        ];
        for &scale in &scales {
            let scaled_w: Vec<f64> = base_w.iter().map(|&w| w * scale).collect();
            let g = CsrGraph::<f64>::from_edges(6, &edges, Some(&scaled_w), false, false).unwrap();

            // 2D normalized: scale-invariant vs unit scale
            let l2n = hde_with_pivots::<f64, f64>(&g, &pivots, &cfg2_norm).unwrap();
            assert!(l2n.is_all_finite());
            assert!((isotropic_std(&l2n) - 1.0).abs() < 1e-12);
            let l2n_pw = pairwise_euclidean_nd(
                &(0..6)
                    .map(|i| vec![l2n.x()[i], l2n.y()[i]])
                    .collect::<Vec<_>>(),
            );
            let err_2n = max_rel_pairwise_err(&l2n_pw, &ref2_norm_pw, 1.0);
            assert!(
                err_2n < 1e-10,
                "2D normalized rel err at scale {scale:e}: {err_2n}"
            );

            // 2D unnormalized: scales linearly by `scale`
            let l2u = hde_with_pivots::<f64, f64>(&g, &pivots, &cfg2_unnorm).unwrap();
            assert!(l2u.is_all_finite());
            let l2u_pw = pairwise_euclidean_nd(
                &(0..6)
                    .map(|i| vec![l2u.x()[i], l2u.y()[i]])
                    .collect::<Vec<_>>(),
            );
            let err_2u = max_rel_pairwise_err(&l2u_pw, &ref2_unnorm_pw, scale);
            assert!(
                err_2u < 1e-10,
                "2D unnormalized rel err at scale {scale:e}: {err_2u}"
            );

            // 3D normalized: scale-invariant vs unit scale
            let l3n = hde_nd_with_pivots::<f64, f64>(&g, &pivots, &cfg3_norm).unwrap();
            assert!((isotropic_std_nd(&l3n) - 1.0).abs() < 1e-12);
            let l3n_pw = pairwise_euclidean_nd(
                &(0..6)
                    .map(|i| l3n.node(i).unwrap().to_vec())
                    .collect::<Vec<_>>(),
            );
            let err_3n = max_rel_pairwise_err(&l3n_pw, &ref3_norm_pw, 1.0);
            assert!(
                err_3n < 1e-10,
                "3D normalized rel err at scale {scale:e}: {err_3n}"
            );

            // 3D unnormalized: scales linearly by `scale`
            let l3u = hde_nd_with_pivots::<f64, f64>(&g, &pivots, &cfg3_unnorm).unwrap();
            assert!(l3u.coords().iter().all(|v| v.is_finite()));
            let l3u_pw = pairwise_euclidean_nd(
                &(0..6)
                    .map(|i| l3u.node(i).unwrap().to_vec())
                    .collect::<Vec<_>>(),
            );
            let err_3u = max_rel_pairwise_err(&l3u_pw, &ref3_unnorm_pw, scale);
            assert!(
                err_3u < 1e-10,
                "3D unnormalized rel err at scale {scale:e}: {err_3u}"
            );
        }

        // Subnormal positive weights (e.g., 1e-315): two-step covariance scaling handles subnormals
        let subnormal_w: Vec<f64> = base_w.iter().map(|&w| w * 1e-315).collect();
        let g_sub =
            CsrGraph::<f64>::from_edges(6, &edges, Some(&subnormal_w), false, false).unwrap();
        let l2_sub = hde_with_pivots::<f64, f64>(&g_sub, &pivots, &cfg2_norm).unwrap();
        assert!(l2_sub.is_all_finite());
        assert!((isotropic_std(&l2_sub) - 1.0).abs() < 1e-12);
        let l2_sub_pw = pairwise_euclidean_nd(
            &(0..6)
                .map(|i| vec![l2_sub.x()[i], l2_sub.y()[i]])
                .collect::<Vec<_>>(),
        );
        let err_sub = max_rel_pairwise_err(&l2_sub_pw, &ref2_norm_pw, 1.0);
        assert!(
            err_sub < 1e-5,
            "2D subnormal normalized rel err too large: {err_sub}"
        );

        // Disconnected N=4 with huge weights = 0.5 * f64::MAX:
        // normalize=true must match normalize=false rescaled to unit isotropic std (not Gaussian fallback).
        let disc_edges = vec![(0, 1), (2, 3)];
        let huge = 0.5 * f64::MAX;
        let disc_huge =
            CsrGraph::<f64>::from_edges(4, &disc_edges, Some(&[huge, huge]), false, false).unwrap();
        let disc_pivots = vec![0u32, 1, 2];

        let d2_unnorm =
            hde_with_pivots::<f64, f64>(&disc_huge, &disc_pivots, &cfg2_unnorm).unwrap();
        assert!(d2_unnorm.is_all_finite());
        let mut d2_ref = Layout2d::from_xy(
            d2_unnorm.x().iter().map(|&v| v / huge).collect(),
            d2_unnorm.y().iter().map(|&v| v / huge).collect(),
        )
        .unwrap();
        normalize_layout_2d(&mut d2_ref, 1.0, cfg2_norm.seed).unwrap();
        let d2_huge = hde_with_pivots::<f64, f64>(&disc_huge, &disc_pivots, &cfg2_norm).unwrap();
        assert!(d2_huge.is_all_finite());
        assert!((isotropic_std(&d2_huge) - 1.0).abs() < 1e-12);
        let d2_ref_pw = pairwise_euclidean_nd(
            &(0..4)
                .map(|i| vec![d2_ref.x()[i], d2_ref.y()[i]])
                .collect::<Vec<_>>(),
        );
        let d2_huge_pw = pairwise_euclidean_nd(
            &(0..4)
                .map(|i| vec![d2_huge.x()[i], d2_huge.y()[i]])
                .collect::<Vec<_>>(),
        );
        assert!(max_rel_pairwise_err(&d2_huge_pw, &d2_ref_pw, 1.0) < 1e-10);

        let d3_unnorm =
            hde_nd_with_pivots::<f64, f64>(&disc_huge, &disc_pivots, &cfg3_unnorm).unwrap();
        assert!(d3_unnorm.coords().iter().all(|v| v.is_finite()));
        let mut d3_ref = LayoutNd::<f64>::zeros(4, 3).unwrap();
        for (dst, &src) in d3_ref
            .coords_mut()
            .iter_mut()
            .zip(d3_unnorm.coords().iter())
        {
            *dst = src / huge;
        }
        normalize_layout_nd(&mut d3_ref, 1.0, cfg3_norm.seed).unwrap();
        let d3_huge = hde_nd_with_pivots::<f64, f64>(&disc_huge, &disc_pivots, &cfg3_norm).unwrap();
        assert!((isotropic_std_nd(&d3_huge) - 1.0).abs() < 1e-12);
        let d3_ref_pw = pairwise_euclidean_nd(
            &(0..4)
                .map(|i| d3_ref.node(i).unwrap().to_vec())
                .collect::<Vec<_>>(),
        );
        let d3_huge_pw = pairwise_euclidean_nd(
            &(0..4)
                .map(|i| d3_huge.node(i).unwrap().to_vec())
                .collect::<Vec<_>>(),
        );
        assert!(max_rel_pairwise_err(&d3_huge_pw, &d3_ref_pw, 1.0) < 1e-10);

        // Unnormalized projection overflow near f64::MAX returns NumericalBreakdown instead of Inf/NaN
        let ovf_edges = vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (0, 4)];
        let ovf_w = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.95 * f64::MAX];
        let ovf_g = CsrGraph::<f64>::from_edges(5, &ovf_edges, Some(&ovf_w), false, false).unwrap();
        let ovf_pivots = vec![0u32, 1, 2, 3];
        assert!(matches!(
            hde_with_pivots::<f64, f64>(&ovf_g, &ovf_pivots, &cfg2_unnorm),
            Err(CoreError::NumericalBreakdown { .. })
        ));
        let ovf_norm = hde_with_pivots::<f64, f64>(&ovf_g, &ovf_pivots, &cfg2_norm).unwrap();
        assert!(ovf_norm.is_all_finite());
        assert!((isotropic_std(&ovf_norm) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_hde_f32_output_overflow_returns_numerical_breakdown() {
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        let pivots = [0u32, 3];

        // 1. Valid positive finite f32 weights whose unnormalized projection exceeds f32::MAX
        let g_f32 =
            CsrGraph::<f32>::from_edges(4, &edges, Some(&[3e38_f32; 3]), false, false).unwrap();

        let cfg2_unnorm = HdeConfig {
            normalize: false,
            ..HdeConfig::default()
        };
        let cfg2_norm = HdeConfig::default();
        let cfg3_unnorm = HdeConfig {
            output_dims: 3,
            normalize: false,
            ..HdeConfig::default()
        };
        let cfg3_norm = HdeConfig {
            output_dims: 3,
            normalize: true,
            ..HdeConfig::default()
        };

        assert!(matches!(
            hde::<f32, f32>(&g_f32, &cfg2_unnorm),
            Err(CoreError::NumericalBreakdown { .. })
        ));
        assert!(matches!(
            hde_with_pivots::<f32, f32>(&g_f32, &pivots, &cfg2_unnorm),
            Err(CoreError::NumericalBreakdown { .. })
        ));
        assert!(matches!(
            hde_nd::<f32, f32>(&g_f32, &cfg3_unnorm),
            Err(CoreError::NumericalBreakdown { .. })
        ));
        assert!(matches!(
            hde_nd_with_pivots::<f32, f32>(&g_f32, &pivots, &cfg3_unnorm),
            Err(CoreError::NumericalBreakdown { .. })
        ));

        let ok_2d = hde::<f32, f32>(&g_f32, &cfg2_norm).unwrap();
        assert!(ok_2d.is_all_finite());
        let ok_2d_piv = hde_with_pivots::<f32, f32>(&g_f32, &pivots, &cfg2_norm).unwrap();
        assert!(ok_2d_piv.is_all_finite());
        let ok_3d = hde_nd::<f32, f32>(&g_f32, &cfg3_norm).unwrap();
        assert!(ok_3d.is_all_finite());
        let ok_3d_piv = hde_nd_with_pivots::<f32, f32>(&g_f32, &pivots, &cfg3_norm).unwrap();
        assert!(ok_3d_piv.is_all_finite());

        // 2. f64 graph weights overflowing S = f32 on unnormalized cast
        let g_f64 =
            CsrGraph::<f64>::from_edges(4, &edges, Some(&[1e40_f64; 3]), false, false).unwrap();
        assert!(matches!(
            hde::<f64, f32>(&g_f64, &cfg2_unnorm),
            Err(CoreError::NumericalBreakdown { .. })
        ));
        let ok_f64_to_f32 = hde::<f64, f32>(&g_f64, &cfg2_norm).unwrap();
        assert!(ok_f64_to_f32.is_all_finite());
    }
}
