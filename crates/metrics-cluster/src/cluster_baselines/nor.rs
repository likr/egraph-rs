use std::cmp::Ordering;

use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, Scalar};
use egraph_metrics_geometry::{extract_unique_edges, validate_graph_layout};

use super::{NorMetrics, canonical_zero, convex_hull_ccw};

struct Fenwick {
    tree: Vec<u64>,
}

impl Fenwick {
    fn new(n: usize) -> Self {
        Self {
            tree: vec![0; n + 1],
        }
    }

    fn add(&mut self, mut idx: usize, delta: u64) {
        idx += 1;
        while idx < self.tree.len() {
            self.tree[idx] += delta;
            idx += idx.isolate_lowest_one();
        }
    }

    fn prefix_sum(&self, mut idx: usize) -> u64 {
        let mut sum = 0u64;
        idx += 1;
        while idx > 0 {
            sum += self.tree[idx];
            idx -= idx.isolate_lowest_one();
        }
        sum
    }
}

fn axis_inversions_f64(a: &[f64], b: &[f64]) -> u64 {
    let n = a.len();
    if n < 2 {
        return 0;
    }

    let mut order_b: Vec<usize> = (0..n).collect();
    order_b.sort_unstable_by(|&i, &j| canonical_zero(b[i]).total_cmp(&canonical_zero(b[j])));
    let mut rank_b = vec![0usize; n];
    let mut cur_rank = 0usize;
    rank_b[order_b[0]] = 0;
    for pos in 1..n {
        if canonical_zero(b[order_b[pos]]).total_cmp(&canonical_zero(b[order_b[pos - 1]]))
            != Ordering::Equal
        {
            cur_rank += 1;
        }
        rank_b[order_b[pos]] = cur_rank;
    }

    let mut order_a: Vec<usize> = (0..n).collect();
    order_a.sort_unstable_by(|&i, &j| canonical_zero(a[i]).total_cmp(&canonical_zero(a[j])));

    let mut bit = Fenwick::new(cur_rank + 1);
    let mut inserted = 0u64;
    let mut inversions = 0u64;
    let mut start = 0usize;
    while start < n {
        let val = canonical_zero(a[order_a[start]]);
        let mut end = start + 1;
        while end < n && canonical_zero(a[order_a[end]]).total_cmp(&val) == Ordering::Equal {
            end += 1;
        }
        for &idx in &order_a[start..end] {
            let le_count = bit.prefix_sum(rank_b[idx]);
            inversions += inserted - le_count;
        }
        for &idx in &order_a[start..end] {
            bit.add(rank_b[idx], 1);
            inserted += 1;
        }
        start = end;
    }

    inversions
}

fn polygon_area(poly: &[[f64; 2]]) -> f64 {
    let m = poly.len();
    if m < 3 {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    for i in 0..m {
        let j = (i + 1) % m;
        sum += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1];
    }
    0.5 * sum.abs()
}

fn corners_convex_hull_area<S: Scalar>(
    layout: &Layout2d<S>,
    half_widths: &[S],
    half_heights: &[S],
) -> f64 {
    let n = layout.len();
    if n == 0 {
        return 0.0;
    }
    let xs = layout.x();
    let ys = layout.y();
    let mut corners = Vec::with_capacity(n * 4);
    for i in 0..n {
        let x = xs[i].to_f64();
        let y = ys[i].to_f64();
        let hw = half_widths[i].to_f64();
        let hh = half_heights[i].to_f64();
        corners.push([x - hw, y - hh]);
        corners.push([x + hw, y - hh]);
        corners.push([x + hw, y + hh]);
        corners.push([x - hw, y + hh]);
    }
    let hull = convex_hull_ccw(corners);
    polygon_area(&hull)
}

fn box_inclusive_bbox<S: Scalar>(
    layout: &Layout2d<S>,
    half_widths: &[S],
    half_heights: &[S],
) -> (f64, f64) {
    let xs = layout.x();
    let ys = layout.y();
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    for i in 0..layout.len() {
        let x = xs[i].to_f64();
        let y = ys[i].to_f64();
        let hw = half_widths[i].to_f64();
        let hh = half_heights[i].to_f64();
        xmin = xmin.min(x - hw);
        xmax = xmax.max(x + hw);
        ymin = ymin.min(y - hh);
        ymax = ymax.max(y + hh);
    }
    (xmax - xmin, ymax - ymin)
}

fn center_bbox(xs: &[f64], ys: &[f64]) -> (f64, f64, f64, f64) {
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        xmin = xmin.min(x);
        xmax = xmax.max(x);
        ymin = ymin.min(y);
        ymax = ymax.max(y);
    }
    (
        0.5 * (xmin + xmax),
        0.5 * (ymin + ymax),
        xmax - xmin,
        ymax - ymin,
    )
}

fn nor_metrics_core<S: Scalar>(
    orig_layout: &Layout2d<S>,
    adjusted_layout: &Layout2d<S>,
    half_widths: &[S],
    half_heights: &[S],
    edges: Option<&[(u32, u32)]>,
) -> CoreResult<NorMetrics> {
    let n = orig_layout.len();
    if adjusted_layout.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            adjusted_layout.len(),
            "nor_metrics::adjusted_layout",
        ));
    }
    if half_widths.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            half_widths.len(),
            "nor_metrics::half_widths",
        ));
    }
    if half_heights.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            half_heights.len(),
            "nor_metrics::half_heights",
        ));
    }
    orig_layout.validate_finite()?;
    adjusted_layout.validate_finite()?;
    for (&hw, &hh) in half_widths.iter().zip(half_heights.iter()) {
        if !hw.is_finite() || !hh.is_finite() {
            return Err(CoreError::non_finite("nor_metrics::extents"));
        }
        if hw < S::ZERO || hh < S::ZERO {
            return Err(CoreError::invalid_input(
                "nor_metrics requires non-negative half_widths and half_heights",
            ));
        }
    }

    if n == 0 {
        return Ok(NorMetrics::default());
    }
    if n == 1 {
        return Ok(NorMetrics {
            oo_nni: 0.0,
            sp_ch_a: 1.0,
            gs_bb_iar: 1.0,
            nm_dm_imse: 0.0,
            el_rsd: 0.0,
        });
    }

    let orig_x: Vec<f64> = orig_layout.x().iter().map(|v| v.to_f64()).collect();
    let orig_y: Vec<f64> = orig_layout.y().iter().map(|v| v.to_f64()).collect();
    let adj_x: Vec<f64> = adjusted_layout.x().iter().map(|v| v.to_f64()).collect();
    let adj_y: Vec<f64> = adjusted_layout.y().iter().map(|v| v.to_f64()).collect();

    let inv_x = axis_inversions_f64(&orig_x, &adj_x);
    let inv_y = axis_inversions_f64(&orig_y, &adj_y);
    let oo_nni = (inv_x + inv_y) as f64 / (n * (n - 1)) as f64;

    let orig_ch_area = corners_convex_hull_area(orig_layout, half_widths, half_heights);
    let adj_ch_area = corners_convex_hull_area(adjusted_layout, half_widths, half_heights);
    let sp_ch_a = if orig_ch_area <= 1e-18 {
        1.0
    } else {
        adj_ch_area / orig_ch_area
    };

    let (w_orig, h_orig) = box_inclusive_bbox(orig_layout, half_widths, half_heights);
    let (w_adj, h_adj) = box_inclusive_bbox(adjusted_layout, half_widths, half_heights);
    let gs_bb_iar = if w_orig <= 1e-18 || h_orig <= 1e-18 || w_adj <= 1e-18 || h_adj <= 1e-18 {
        1.0
    } else {
        let r1 = (w_adj * h_orig) / (h_adj * w_orig);
        let r2 = (h_adj * w_orig) / (w_adj * h_orig);
        r1.max(r2)
    };

    let (cx_o, cy_o, w_o, h_o) = center_bbox(&orig_x, &orig_y);
    let (cx_a, cy_a, w_a, h_a) = center_bbox(&adj_x, &adj_y);
    let sx = if w_o > 1e-18 { w_a / w_o } else { 1.0 };
    let sy = if h_o > 1e-18 { h_a / h_o } else { 1.0 };
    let mut sum_sq_err = 0.0_f64;
    for i in 0..n {
        let tx = cx_a + (orig_x[i] - cx_o) * sx;
        let ty = cy_a + (orig_y[i] - cy_o) * sy;
        let dx = adj_x[i] - tx;
        let dy = adj_y[i] - ty;
        sum_sq_err += dx * dx + dy * dy;
    }
    let nm_dm_imse = sum_sq_err / n as f64;

    let el_rsd = if let Some(edge_list) = edges {
        let mut ratios = Vec::with_capacity(edge_list.len());
        let mut sum_r = 0.0_f64;
        for &(u, v) in edge_list {
            let ui = u as usize;
            let vi = v as usize;
            let len_o = (orig_x[ui] - orig_x[vi]).hypot(orig_y[ui] - orig_y[vi]);
            if len_o > 1e-18 {
                let len_a = (adj_x[ui] - adj_x[vi]).hypot(adj_y[ui] - adj_y[vi]);
                let r = len_a / len_o;
                ratios.push(r);
                sum_r += r;
            }
        }
        if ratios.is_empty() {
            0.0
        } else {
            let mean_r = sum_r / ratios.len() as f64;
            if mean_r <= 1e-18 {
                0.0
            } else {
                let var = ratios
                    .iter()
                    .map(|&r| {
                        let d = r - mean_r;
                        d * d
                    })
                    .sum::<f64>()
                    / ratios.len() as f64;
                var.sqrt() / mean_r
            }
        }
    } else {
        0.0
    };

    Ok(NorMetrics {
        oo_nni,
        sp_ch_a,
        gs_bb_iar,
        nm_dm_imse,
        el_rsd,
    })
}

/// Computes Chen et al. (2020) 5-metric Node Overlap Removal (`NOR`) evaluation including
/// edge length preservation (`el_rsd`) over unique non-self-loop edges of `graph`.
pub fn nor_metrics<W, S: Scalar>(
    graph: &CsrGraph<W>,
    orig_layout: &Layout2d<S>,
    adjusted_layout: &Layout2d<S>,
    half_widths: &[S],
    half_heights: &[S],
) -> CoreResult<NorMetrics> {
    validate_graph_layout(graph, orig_layout, "nor_metrics::orig_layout")?;
    validate_graph_layout(graph, adjusted_layout, "nor_metrics::adjusted_layout")?;
    let edges = extract_unique_edges(graph);
    nor_metrics_core(
        orig_layout,
        adjusted_layout,
        half_widths,
        half_heights,
        Some(&edges),
    )
}

/// Computes Chen et al. (2020) 5-metric Node Overlap Removal (`NOR`) evaluation without a graph
/// (setting `el_rsd = 0.0`).
pub fn nor_metrics_without_graph<S: Scalar>(
    orig_layout: &Layout2d<S>,
    adjusted_layout: &Layout2d<S>,
    half_widths: &[S],
    half_heights: &[S],
) -> CoreResult<NorMetrics> {
    nor_metrics_core(
        orig_layout,
        adjusted_layout,
        half_widths,
        half_heights,
        None,
    )
}
