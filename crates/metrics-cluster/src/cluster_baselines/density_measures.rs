use egraph_core::{CoreError, CoreResult, Layout2d, Scalar, VisitStamp};

use super::compact_labels;
use super::space_dispatch::validate_metric_labels;

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

fn histogram_density_measure_core(
    us: &[f64],
    vs: &[f64],
    comp: &[u32],
    k: usize,
    grid_size: usize,
) -> f64 {
    let n = us.len();
    let g_f = grid_size as f64;
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
    if hdm_total_u <= 0.0 {
        0.0
    } else {
        (1.0 - hdm_weighted_entropy / ((k as f64).log2() * hdm_total_u)).clamp(0.0, 1.0)
    }
}

/// Computes Tatu et al. (2009) `3x3` Moore-smoothed 2D Histogram Density Measure (`2D-HDM`)
/// normalized to `[0.0, 1.0]` (`1.0` = maximally separated, `0.0` = overlapping or `N == 0` / `K < 2`).
pub fn histogram_density_measure<S: Scalar>(
    layout: &Layout2d<S>,
    labels: &[u32],
    grid_size: usize,
) -> CoreResult<f64> {
    if grid_size == 0 {
        return Err(CoreError::invalid_input(
            "histogram_density_measure requires grid_size >= 1",
        ));
    }
    grid_size.checked_mul(grid_size).ok_or_else(|| {
        CoreError::invalid_input("histogram_density_measure grid_size^2 overflow")
    })?;
    validate_metric_labels(layout, labels, "histogram_density_measure")?;

    let n = layout.len();
    let (comp, _, k) = compact_labels(labels);
    if n == 0 || k < 2 {
        return Ok(0.0);
    }

    let (us, vs) = normalized_unit_coords(layout);
    Ok(histogram_density_measure_core(
        &us, &vs, &comp, k, grid_size,
    ))
}

/// Computes Tatu et al. (2009) Class Density Measure (`CDM`) and `3x3` Moore-smoothed
/// Histogram Density Measure (`2D-HDM`), both normalized to `[0.0, 1.0]` (`1.0` = maximally
/// separated, `0.0` = identical overlapping distributions).
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
    validate_metric_labels(layout, labels, "class_density_measure")?;

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
    let hdm = histogram_density_measure_core(&us, &vs, &comp, k, grid_size);

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
