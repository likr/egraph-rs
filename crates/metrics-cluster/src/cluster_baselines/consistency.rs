use egraph_core::{CoreError, CoreResult, Layout2d, Scalar, VisitStamp};

use super::{compact_labels, validate_layout_labels};

/// Computes Sips et al. (2009) Distance Consistency (`DSC`) in `[0.0, 1.0]` (higher is better).
///
/// Counts the fraction of points whose squared Euclidean distance to their own class centroid
/// is **strictly smaller** than to every other class centroid (`N < 2 || K < 2` returns `0.0`).
pub fn distance_consistency<S: Scalar>(layout: &Layout2d<S>, labels: &[u32]) -> CoreResult<f64> {
    validate_layout_labels(layout, labels, "distance_consistency")?;
    let n = layout.len();
    let (comp, sizes, k) = compact_labels(labels);
    if n < 2 || k < 2 {
        return Ok(0.0);
    }

    let xs = layout.x();
    let ys = layout.y();
    let mut cx = vec![0.0_f64; k];
    let mut cy = vec![0.0_f64; k];
    for i in 0..n {
        let c = comp[i] as usize;
        cx[c] += xs[i].to_f64();
        cy[c] += ys[i].to_f64();
    }
    for c in 0..k {
        let inv = 1.0 / sizes[c] as f64;
        cx[c] *= inv;
        cy[c] *= inv;
    }

    let mut consistent = 0usize;
    for i in 0..n {
        let own = comp[i] as usize;
        let x = xs[i].to_f64();
        let y = ys[i].to_f64();
        let dx_own = x - cx[own];
        let dy_own = y - cy[own];
        let own_d2 = dx_own * dx_own + dy_own * dy_own;

        let mut is_strictly_closest = true;
        for c in 0..k {
            if c == own {
                continue;
            }
            let dx = x - cx[c];
            let dy = y - cy[c];
            if dx * dx + dy * dy <= own_d2 {
                is_strictly_closest = false;
                break;
            }
        }
        if is_strictly_closest {
            consistent += 1;
        }
    }
    Ok(consistent as f64 / n as f64)
}

fn normalized_unit_coords<S: Scalar>(layout: &Layout2d<S>) -> (Vec<f64>, Vec<f64>) {
    let n = layout.len();
    let xs = layout.x();
    let ys = layout.y();
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for i in 0..n {
        let x = xs[i].to_f64();
        let y = ys[i].to_f64();
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    let span_x = max_x - min_x;
    let span_y = max_y - min_y;
    let mut us = Vec::with_capacity(n);
    let mut vs = Vec::with_capacity(n);
    for i in 0..n {
        let u = if span_x > 1e-15 {
            (xs[i].to_f64() - min_x) / span_x
        } else {
            0.5
        };
        let v = if span_y > 1e-15 {
            (ys[i].to_f64() - min_y) / span_y
        } else {
            0.5
        };
        us.push(u);
        vs.push(v);
    }
    (us, vs)
}

/// Computes Sips et al. (2009) Distribution Consistency (`DC`) scaled to `[0.0, 1.0]`
/// (`DC_paper = 100.0 * dc`), measuring weighted grid-cell class entropy purity over a
/// `grid_size x grid_size` partition of the normalized `[0, 1]^2` layout box.
///
/// Returns `0.0` when `N == 0` or `K < 2`. Errors when `grid_size == 0`.
pub fn distribution_consistency<S: Scalar>(
    layout: &Layout2d<S>,
    labels: &[u32],
    grid_size: usize,
) -> CoreResult<f64> {
    if grid_size == 0 {
        return Err(CoreError::invalid_input(
            "distribution_consistency requires grid_size >= 1",
        ));
    }
    validate_layout_labels(layout, labels, "distribution_consistency")?;

    let n = layout.len();
    let (comp, _, k) = compact_labels(labels);
    if n == 0 || k < 2 {
        return Ok(0.0);
    }

    let (us, vs) = normalized_unit_coords(layout);
    let g_f = grid_size as f64;
    let mut pairs: Vec<(usize, u32)> = Vec::with_capacity(n);
    for i in 0..n {
        let gx = ((us[i] * g_f).floor() as usize).min(grid_size - 1);
        let gy = ((vs[i] * g_f).floor() as usize).min(grid_size - 1);
        pairs.push((gy * grid_size + gx, comp[i]));
    }
    pairs.sort_unstable();

    let mut weighted_entropy = 0.0_f64;
    let mut start = 0usize;
    while start < n {
        let cell = pairs[start].0;
        let mut end = start + 1;
        while end < n && pairs[end].0 == cell {
            end += 1;
        }
        let p_total = (end - start) as f64;
        let mut cell_h = 0.0_f64;
        let mut c_start = start;
        while c_start < end {
            let cls = pairs[c_start].1;
            let mut c_end = c_start + 1;
            while c_end < end && pairs[c_end].1 == cls {
                c_end += 1;
            }
            let prob = (c_end - c_start) as f64 / p_total;
            cell_h -= prob * prob.log2();
            c_start = c_end;
        }
        weighted_entropy += p_total * cell_h;
        start = end;
    }

    let denom = (k as f64).log2() * n as f64;
    Ok((1.0 - weighted_entropy / denom).clamp(0.0, 1.0))
}

/// Computes Tatu et al. (2009) Class Density Measure (`CDM`) and `3x3` Moore-smoothed
/// Histogram Density Measure (`2D-HDM`), both normalized to `[0.0, 1.0]` (`1.0` = maximally
/// separated, `0.0` = identical overlapping distributions).
///
/// - `CDM` evaluates unit-normalized Gaussian KDE densities per class (default `bandwidth =
///   1.5 / grid_size as f64`) and averages pairwise L1 separation $\frac{1}{2\binom{K}{2}}
///   \sum_{c < d} \sum_g |\tilde{\rho}_g^c - \tilde{\rho}_g^d|$.
/// - `HDM` applies $3\times 3$ Moore-neighborhood smoothing to per-class grid counts and computes
///   `1.0 - sum(U * H) / (log2(K) * sum(U))` (`hdm_paper = 100.0 * hdm`).
///
/// Returns `Ok((0.0, 0.0))` when `N == 0` or `K < 2`.
pub fn class_density_measure<S: Scalar>(
    layout: &Layout2d<S>,
    labels: &[u32],
    grid_size: usize,
    bandwidth: Option<f64>,
) -> CoreResult<(f64, f64)> {
    if grid_size == 0 {
        return Err(CoreError::invalid_input(
            "class_density_measure requires grid_size >= 1",
        ));
    }
    let bw = match bandwidth {
        Some(h) => {
            if !h.is_finite() {
                return Err(CoreError::non_finite("class_density_measure::bandwidth"));
            }
            if h <= 0.0 {
                return Err(CoreError::invalid_input(
                    "class_density_measure requires bandwidth > 0.0",
                ));
            }
            h
        }
        None => 1.5 / grid_size as f64,
    };
    validate_layout_labels(layout, labels, "class_density_measure")?;

    let n = layout.len();
    let (comp, sizes, k) = compact_labels(labels);
    if n == 0 || k < 2 {
        return Ok((0.0, 0.0));
    }

    let num_cells = grid_size
        .checked_mul(grid_size)
        .ok_or_else(|| CoreError::invalid_input("class_density_measure grid_size^2 overflow"))?;
    let (us, vs) = normalized_unit_coords(layout);
    let g_f = grid_size as f64;

    // 1. 2D-HDM via 3x3 Moore-smoothed sparse `(cell, class)` entries (O(N log N) time & O(N) RAM).
    let mut hdm_entries: Vec<(usize, u32)> = Vec::with_capacity(n.saturating_mul(9));
    for i in 0..n {
        let gx = ((us[i] * g_f).floor() as isize).clamp(0, grid_size as isize - 1);
        let gy = ((vs[i] * g_f).floor() as isize).clamp(0, grid_size as isize - 1);
        let cls = comp[i];
        for dy in -1isize..=1 {
            let ny = gy + dy;
            if ny < 0 || ny >= grid_size as isize {
                continue;
            }
            for dx in -1isize..=1 {
                let nx = gx + dx;
                if nx < 0 || nx >= grid_size as isize {
                    continue;
                }
                hdm_entries.push(((ny as usize) * grid_size + (nx as usize), cls));
            }
        }
    }
    hdm_entries.sort_unstable();

    let mut hdm_weighted_entropy = 0.0_f64;
    let mut hdm_total_u = 0.0_f64;
    let mut idx = 0usize;
    while idx < hdm_entries.len() {
        let cell = hdm_entries[idx].0;
        let mut end = idx + 1;
        while end < hdm_entries.len() && hdm_entries[end].0 == cell {
            end += 1;
        }
        let u_total = (end - idx) as f64;
        let mut cell_h = 0.0_f64;
        let mut c_start = idx;
        while c_start < end {
            let cls = hdm_entries[c_start].1;
            let mut c_end = c_start + 1;
            while c_end < end && hdm_entries[c_end].1 == cls {
                c_end += 1;
            }
            let prob = (c_end - c_start) as f64 / u_total;
            cell_h -= prob * prob.log2();
            c_start = c_end;
        }
        hdm_weighted_entropy += u_total * cell_h;
        hdm_total_u += u_total;
        idx = end;
    }
    let hdm = if hdm_total_u <= 0.0 {
        0.0
    } else {
        (1.0 - hdm_weighted_entropy / ((k as f64).log2() * hdm_total_u)).clamp(0.0, 1.0)
    };

    // 2. 2D-CDM via per-class CSR point grouping + single reusable grid with VisitStamp.
    let mut class_offsets = vec![0usize; k + 1];
    for c in 0..k {
        class_offsets[c + 1] = class_offsets[c] + sizes[c];
    }
    let mut class_cursor = class_offsets[..k].to_vec();
    let mut class_points = vec![0usize; n];
    for (i, &c) in comp.iter().enumerate() {
        let pos = class_cursor[c as usize];
        class_points[pos] = i;
        class_cursor[c as usize] = pos + 1;
    }

    let inv_two_bw_sq = 1.0 / (2.0 * bw * bw);
    let cutoff = 4.0 * bw;
    let cutoff_sq = cutoff * cutoff;
    let r_cells = (cutoff * g_f).ceil().min(g_f) as isize;

    let mut cell_acc = vec![0.0_f64; num_cells];
    let mut stamp = VisitStamp::new(num_cells);
    let mut touched = Vec::new();
    let mut sparse_kde: Vec<(usize, f64)> = Vec::new();

    for c in 0..k {
        stamp.clear();
        touched.clear();
        let mut sum_w = 0.0_f64;

        for &pt_idx in &class_points[class_offsets[c]..class_offsets[c + 1]] {
            let u = us[pt_idx];
            let v = vs[pt_idx];
            let gx = ((u * g_f).floor() as isize).clamp(0, grid_size as isize - 1);
            let gy = ((v * g_f).floor() as isize).clamp(0, grid_size as isize - 1);

            let x_lo = (gx - r_cells).max(0);
            let x_hi = (gx + r_cells).min(grid_size as isize - 1);
            let y_lo = (gy - r_cells).max(0);
            let y_hi = (gy + r_cells).min(grid_size as isize - 1);

            for cy in y_lo..=y_hi {
                let cy_center = (cy as f64 + 0.5) / g_f;
                let dy = cy_center - v;
                let dy_sq = dy * dy;
                if dy_sq > cutoff_sq {
                    continue;
                }
                let row_off = (cy as usize) * grid_size;
                for cx in x_lo..=x_hi {
                    let cx_center = (cx as f64 + 0.5) / g_f;
                    let dx = cx_center - u;
                    let d_sq = dx * dx + dy_sq;
                    if d_sq > cutoff_sq {
                        continue;
                    }
                    let w = (-d_sq * inv_two_bw_sq).exp();
                    if w > 0.0 {
                        let cell = row_off + (cx as usize);
                        if stamp.visit(cell) {
                            cell_acc[cell] = w;
                            touched.push(cell);
                        } else {
                            cell_acc[cell] += w;
                        }
                        sum_w += w;
                    }
                }
            }
        }

        if sum_w <= 0.0 {
            // Extreme sub-cell bandwidth fallback: Dirac impulse onto owning grid cell.
            stamp.clear();
            touched.clear();
            sum_w = 0.0;
            for &pt_idx in &class_points[class_offsets[c]..class_offsets[c + 1]] {
                let gx = ((us[pt_idx] * g_f).floor() as usize).min(grid_size - 1);
                let gy = ((vs[pt_idx] * g_f).floor() as usize).min(grid_size - 1);
                let cell = gy * grid_size + gx;
                if stamp.visit(cell) {
                    cell_acc[cell] = 1.0;
                    touched.push(cell);
                } else {
                    cell_acc[cell] += 1.0;
                }
                sum_w += 1.0;
            }
        }

        let inv_sum = 1.0 / sum_w;
        for &cell in &touched {
            sparse_kde.push((cell, cell_acc[cell] * inv_sum));
        }
    }

    sparse_kde.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));

    let mut total_min_sum = 0.0_f64;
    let mut s_idx = 0usize;
    while s_idx < sparse_kde.len() {
        let cell = sparse_kde[s_idx].0;
        let mut s_end = s_idx + 1;
        while s_end < sparse_kde.len() && sparse_kde[s_end].0 == cell {
            s_end += 1;
        }
        let m = s_end - s_idx;
        if m >= 2 {
            // Because slice is sorted ascending by density, v_p is the minimum in (m - 1 - p) pairs.
            for (p, &(_, val)) in sparse_kde[s_idx..s_end].iter().enumerate() {
                total_min_sum += val * ((m - 1 - p) as f64);
            }
        }
        s_idx = s_end;
    }

    let num_pairs = (k * (k - 1) / 2) as f64;
    let cdm = (1.0 - total_min_sum / num_pairs).clamp(0.0, 1.0);
    Ok((cdm, hdm))
}
