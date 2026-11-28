use egraph_core::{CoreError, CoreResult, Layout2d, NodeExtents2d, Scalar};
use egraph_spatial_trees::RectangleOverlapSweep;

use super::canonical_zero;

struct UnionAreaSegTree {
    ys: Vec<f64>,
    count: Vec<i32>,
    len: Vec<f64>,
}

impl UnionAreaSegTree {
    fn new(ys: Vec<f64>) -> Self {
        let m = ys.len().saturating_sub(1);
        let size = (m * 4).max(4);
        Self {
            ys,
            count: vec![0; size],
            len: vec![0.0; size],
        }
    }

    fn update(&mut self, node: usize, l: usize, r: usize, ql: usize, qr: usize, delta: i32) {
        if ql <= l && r <= qr {
            self.count[node] += delta;
        } else {
            let mid = l + (r - l) / 2;
            if ql < mid {
                self.update(node * 2, l, mid, ql, qr, delta);
            }
            if qr > mid {
                self.update(node * 2 + 1, mid, r, ql, qr, delta);
            }
        }
        if self.count[node] > 0 {
            self.len[node] = self.ys[r] - self.ys[l];
        } else if r - l == 1 {
            self.len[node] = 0.0;
        } else {
            self.len[node] = self.len[node * 2] + self.len[node * 2 + 1];
        }
    }
}

pub(super) fn rect_union_area(rects: &[[f64; 4]]) -> f64 {
    if rects.is_empty() {
        return 0.0;
    }
    let mut ys = Vec::with_capacity(rects.len() * 2);
    for r in rects {
        ys.push(canonical_zero(r[2]));
        ys.push(canonical_zero(r[3]));
    }
    ys.sort_unstable_by(f64::total_cmp);
    ys.dedup();
    if ys.len() < 2 {
        return 0.0;
    }
    let m = ys.len() - 1;

    let mut events: Vec<(f64, i32, usize, usize)> = Vec::with_capacity(rects.len() * 2);
    for r in rects {
        let y0 = canonical_zero(r[2]);
        let y1 = canonical_zero(r[3]);
        let y_lo = ys.binary_search_by(|v| v.total_cmp(&y0)).unwrap_or(0);
        let y_hi = ys.binary_search_by(|v| v.total_cmp(&y1)).unwrap_or(0);
        if y_lo < y_hi {
            events.push((canonical_zero(r[0]), 1, y_lo, y_hi));
            events.push((canonical_zero(r[1]), -1, y_lo, y_hi));
        }
    }
    if events.is_empty() {
        return 0.0;
    }
    events.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let mut seg = UnionAreaSegTree::new(ys);
    let mut area = 0.0_f64;
    let mut prev_x = events[0].0;
    for &(x, delta, y_lo, y_hi) in &events {
        if x > prev_x {
            area += (x - prev_x) * seg.len[1];
            prev_x = x;
        }
        seg.update(1, 0, m, y_lo, y_hi, delta);
    }
    area
}

/// Computes Liu et al. (2020) `Sprawlter-NN` composite node-node sprawl and area-aware
/// clutter measure ($T_{\mathrm{NN}}(G) = \sqrt{S(G) \cdot \max(A_{\mathrm{NN}}(G), 1)}$)
/// for axis-aligned rectangles $[x_i - hw_i, x_i + hw_i] \times [y_i - hh_i, y_i + hh_i]$.
///
/// Uses $\alpha = 0.20, \gamma = 0.7$, exact $O(N \log N)$ Klee sweep-line union area for $S(G)$,
/// and [`RectangleOverlapSweep::find_layout_extents_overlaps`] for strict interior overlaps
/// (`ox > 0 && oy > 0`). Returns `0.0` for `N == 0` and `1.0` for `N == 1`. Requires strictly
/// positive finite extents.
pub fn sprawlter_nn<S: Scalar>(
    layout: &Layout2d<S>,
    extents: NodeExtents2d<'_, S>,
) -> CoreResult<f64> {
    let n = layout.len();
    if extents.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            extents.len(),
            "sprawlter_nn::extents",
        ));
    }
    layout.validate_finite()?;
    extents.validate()?;
    for i in 0..n {
        let (hw, hh) = extents.extents(i)?;
        if hw <= S::ZERO || hh <= S::ZERO {
            return Err(CoreError::invalid_input(
                "sprawlter_nn requires strictly positive half_widths and half_heights",
            ));
        }
    }

    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        return Ok(1.0);
    }

    let xs = layout.x();
    let ys = layout.y();
    let mut rects = Vec::with_capacity(n);
    let mut areas = Vec::with_capacity(n);
    let mut a_min = f64::INFINITY;
    let mut bb_xmin = f64::INFINITY;
    let mut bb_xmax = f64::NEG_INFINITY;
    let mut bb_ymin = f64::INFINITY;
    let mut bb_ymax = f64::NEG_INFINITY;

    for i in 0..n {
        let cx = xs[i].to_f64();
        let cy = ys[i].to_f64();
        let (hw_s, hh_s) = extents.extents(i)?;
        let hw = hw_s.to_f64();
        let hh = hh_s.to_f64();
        let x0 = cx - hw;
        let x1 = cx + hw;
        let y0 = cy - hh;
        let y1 = cy + hh;
        bb_xmin = bb_xmin.min(x0);
        bb_xmax = bb_xmax.max(x1);
        bb_ymin = bb_ymin.min(y0);
        bb_ymax = bb_ymax.max(y1);
        let area_i = 4.0 * hw * hh;
        a_min = a_min.min(area_i);
        areas.push(area_i);
        rects.push([x0, x1, y0, y1]);
    }

    let bbox_area = (bb_xmax - bb_xmin) * (bb_ymax - bb_ymin);
    let union_area = rect_union_area(&rects);
    let sprawl = if union_area <= 0.0 {
        1.0
    } else {
        (bbox_area / union_area).max(1.0)
    };

    let overlaps = RectangleOverlapSweep::find_layout_extents_overlaps(layout, &extents)?;
    let mut a_nn = 0.0_f64;
    if a_min > 0.0 {
        for (i, j) in overlaps {
            let ox = rects[i][1].min(rects[j][1]) - rects[i][0].max(rects[j][0]);
            let oy = rects[i][3].min(rects[j][3]) - rects[i][2].max(rects[j][2]);
            if ox > 0.0 && oy > 0.0 {
                let x = (ox * oy) / a_min;
                let m_ij = areas[i].min(areas[j]) / a_min;
                a_nn += 0.80 * (2.0 * x).powf(0.7) + 0.20 * m_ij.powf(0.7);
            }
        }
    }

    Ok((sprawl * a_nn.max(1.0)).sqrt())
}

/// Uniform-radius square-box convenience wrapper calling
/// [`sprawlter_nn`]`(layout, NodeExtents2d::from_radii(radii)?)`.
pub fn sprawlter_nn_radii<S: Scalar>(layout: &Layout2d<S>, radii: &[S]) -> CoreResult<f64> {
    sprawlter_nn(layout, NodeExtents2d::from_radii(radii)?)
}
