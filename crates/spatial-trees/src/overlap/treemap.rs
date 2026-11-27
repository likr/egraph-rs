use egraph_core::{CoreError, CoreResult};

/// Computes a Bruls–Huizing–van Wijk (2000) squarified treemap partitioning of rectangle
/// `[x, x + width] x [y, y + height]` proportional to `weights`.
///
/// Each returned rectangle `[x_min, x_max, y_min, y_max]` corresponds to the same index as
/// `weights` (even though items are internally sorted descending by weight during squarification).
///
/// Returns [`CoreError`] if `weights` is empty, any weight is non-finite or `<= 0.0`, or any
/// rectangle parameter is non-finite or `width <= 0.0` / `height <= 0.0`.
pub fn squarified_treemap(
    weights: &[f64],
    x: f64,
    y: f64,
    width: f64,
    height: f64,
) -> CoreResult<Vec<[f64; 4]>> {
    if !x.is_finite() || !y.is_finite() || !width.is_finite() || !height.is_finite() {
        return Err(CoreError::non_finite("squarified_treemap::bounds"));
    }
    if width <= 0.0 || height <= 0.0 {
        return Err(CoreError::invalid_input(format!(
            "squarified_treemap requires width > 0.0 and height > 0.0, got ({width}, {height})"
        )));
    }
    if weights.is_empty() {
        return Err(CoreError::invalid_input(
            "squarified_treemap requires non-empty weights",
        ));
    }

    let mut total_weight = 0.0_f64;
    for (idx, &w) in weights.iter().enumerate() {
        if !w.is_finite() {
            return Err(CoreError::non_finite("squarified_treemap::weights"));
        }
        if w <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "squarified_treemap requires all weights > 0.0, got {w} at index {idx}"
            )));
        }
        total_weight += w;
    }
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return Err(CoreError::invalid_input(
            "squarified_treemap total weight must be positive and finite",
        ));
    }

    let total_area = width * height;
    if !total_area.is_finite() || total_area <= 0.0 {
        return Err(CoreError::invalid_input(
            "squarified_treemap rectangle area must be positive and finite",
        ));
    }

    let k = weights.len();
    let mut items: Vec<(f64, usize)> = weights
        .iter()
        .enumerate()
        .map(|(i, &w)| (total_area * (w / total_weight), i))
        .collect();
    items.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

    let mut out = vec![[0.0_f64; 4]; k];
    let mut rx = x;
    let mut ry = y;
    let mut rw = width;
    let mut rh = height;

    let mut row: Vec<(f64, usize)> = Vec::new();
    let mut row_sum = 0.0_f64;

    for &(area, orig_idx) in &items {
        let short_side = rw.min(rh);
        if row.is_empty() {
            row.push((area, orig_idx));
            row_sum = area;
        } else {
            let cur_worst = treemap_worst(&row, row_sum, short_side);
            let next_worst = treemap_worst_with(&row, row_sum, area, short_side);
            if cur_worst >= next_worst {
                row.push((area, orig_idx));
                row_sum += area;
            } else {
                layout_treemap_row(
                    &row, row_sum, false, &mut rx, &mut ry, &mut rw, &mut rh, &mut out,
                );
                row.clear();
                row.push((area, orig_idx));
                row_sum = area;
            }
        }
    }

    if !row.is_empty() {
        layout_treemap_row(
            &row, row_sum, true, &mut rx, &mut ry, &mut rw, &mut rh, &mut out,
        );
    }

    Ok(out)
}

#[inline]
fn treemap_worst(row: &[(f64, usize)], sum: f64, side_len: f64) -> f64 {
    if row.is_empty() || sum <= 0.0 || side_len <= 0.0 {
        return f64::INFINITY;
    }
    let r_max = row[0].0;
    let r_min = row[row.len() - 1].0;
    let w2 = side_len * side_len;
    let s2 = sum * sum;
    ((w2 * r_max) / s2).max(s2 / (w2 * r_min))
}

#[inline]
fn treemap_worst_with(row: &[(f64, usize)], sum: f64, candidate_area: f64, side_len: f64) -> f64 {
    if side_len <= 0.0 {
        return f64::INFINITY;
    }
    let new_sum = sum + candidate_area;
    let r_max = if row.is_empty() {
        candidate_area
    } else {
        row[0].0.max(candidate_area)
    };
    let r_min = if row.is_empty() {
        candidate_area
    } else {
        row[row.len() - 1].0.min(candidate_area)
    };
    let w2 = side_len * side_len;
    let s2 = new_sum * new_sum;
    ((w2 * r_max) / s2).max(s2 / (w2 * r_min))
}

#[allow(clippy::too_many_arguments)]
fn layout_treemap_row(
    row: &[(f64, usize)],
    row_sum: f64,
    is_final_row: bool,
    rx: &mut f64,
    ry: &mut f64,
    rw: &mut f64,
    rh: &mut f64,
    out: &mut [[f64; 4]],
) {
    if *rw <= *rh {
        let strip_h = if is_final_row {
            *rh
        } else {
            let max_h = *rh * (1.0 - 1e-12);
            (row_sum / *rw).min(max_h).max(1e-15 * *rh)
        };
        let x_end = *rx + *rw;
        let mut x_cursor = *rx;
        for (idx, &(area, orig_idx)) in row.iter().enumerate() {
            let cell_x_max = if idx + 1 == row.len() {
                x_end
            } else {
                (x_cursor + area / strip_h).min(x_end)
            };
            out[orig_idx] = [x_cursor, cell_x_max, *ry, *ry + strip_h];
            x_cursor = cell_x_max;
        }
        *ry += strip_h;
        *rh = (*rh - strip_h).max(0.0);
    } else {
        let strip_w = if is_final_row {
            *rw
        } else {
            let max_w = *rw * (1.0 - 1e-12);
            (row_sum / *rh).min(max_w).max(1e-15 * *rw)
        };
        let y_end = *ry + *rh;
        let mut y_cursor = *ry;
        for (idx, &(area, orig_idx)) in row.iter().enumerate() {
            let cell_y_max = if idx + 1 == row.len() {
                y_end
            } else {
                (y_cursor + area / strip_w).min(y_end)
            };
            out[orig_idx] = [*rx, *rx + strip_w, y_cursor, cell_y_max];
            y_cursor = cell_y_max;
        }
        *rx += strip_w;
        *rw = (*rw - strip_w).max(0.0);
    }
}
