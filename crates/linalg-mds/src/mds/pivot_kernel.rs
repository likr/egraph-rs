use egraph_core::{CoreError, CoreResult};
use egraph_linalg::symmetric_eigen_jacobi;

/// Streams a `(num_nodes, num_reps)` row-major distance table in 3 passes with zero `(N, k)`
/// temporary `f64` buffers, computing $(n / k)^{1/4} C V \mathrm{diag}(e^{-1/4})$.
pub(super) fn mds_from_row_major_distances<F, W>(
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
            if d.is_nan() || d == f64::NEG_INFINITY {
                return Err(CoreError::non_finite(
                    "mds_from_row_major_distances::distances",
                ));
            }
            if d < 0.0 {
                return Err(CoreError::invalid_input(
                    "mds_from_row_major_distances::distances must be >= 0",
                ));
            }
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
