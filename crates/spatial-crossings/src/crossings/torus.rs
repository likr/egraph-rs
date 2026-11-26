use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, Scalar, wrap_torus_coord};

use super::TorusCrossing;
use super::geometry::{Seg, extract_csr_edges};
use super::pair_set::unpack_pair;
use super::sweep::sweep_segments;

#[inline]
fn wrap_torus_pos<S: Scalar>(value: S) -> f64 {
    let w = wrap_torus_coord(value).to_f64();
    if w == 0.0 { 0.0 } else { w }
}

/// Splits a 2D torus edge `(x1, y1) -> (x2, y2)` into 1, 2, or 3 Euclidean segments on `[0, 1]^2`
/// matching legacy `DrawingTorus2d::edge_segments` wrap-around conventions.
pub fn torus_edge_segments<S: Scalar>(x1: S, y1: S, x2: S, y2: S) -> CoreResult<Vec<[f64; 4]>> {
    if !x1.is_finite() || !y1.is_finite() || !x2.is_finite() || !y2.is_finite() {
        return Err(CoreError::non_finite("torus_edge_segments"));
    }
    let px = wrap_torus_pos(x1);
    let py = wrap_torus_pos(y1);
    let qx = wrap_torus_pos(x2);
    let qy = wrap_torus_pos(y2);

    // Find periodic shift (dx, dy) in {-1, 0, +1}^2 minimizing Euclidean distance (px + dx - qx, py + dy - qy).
    let mut best_d = f64::INFINITY;
    let mut best_dx = 0_i32;
    let mut best_dy = 0_i32;
    for dy in -1..=1 {
        let y_shift = py + dy as f64;
        for dx in -1..=1 {
            let x_shift = px + dx as f64;
            let d = (x_shift - qx).hypot(y_shift - qy);
            if d < best_d {
                best_d = d;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }

    if best_dx == 0 && best_dy == 0 {
        return Ok(vec![[px, py, qx, qy]]);
    }

    let torus_max = 1.0 - f64::EPSILON;

    if best_dx == 0 {
        let (x0, y0, x1_w, y1_w) = if py < qy {
            (px, py, qx, qy)
        } else {
            (qx, qy, px, py)
        };
        let denom = y0 - y1_w + 1.0;
        let x2_cross = if denom.abs() <= 1e-15 {
            0.5 * (x0 + x1_w)
        } else {
            (y0 * x1_w - y1_w * x0 + x0) / denom
        };
        let xw = wrap_torus_pos(x2_cross);
        return Ok(vec![[x0, y0, xw, 0.0], [xw, torus_max, x1_w, y1_w]]);
    }

    if best_dy == 0 {
        let (x0, y0, x1_w, y1_w) = if px < qx {
            (px, py, qx, qy)
        } else {
            (qx, qy, px, py)
        };
        let denom = x0 - x1_w + 1.0;
        let y2_cross = if denom.abs() <= 1e-15 {
            0.5 * (y0 + y1_w)
        } else {
            (x0 * y1_w - x1_w * y0 + y0) / denom
        };
        let yw = wrap_torus_pos(y2_cross);
        return Ok(vec![[x0, y0, 0.0, yw], [torus_max, yw, x1_w, y1_w]]);
    }

    let (x0, y0, x1_w, y1_w) = if px < qx {
        (px, py, qx, qy)
    } else {
        (qx, qy, px, py)
    };
    let cx = x0 - x1_w + 1.0;
    let opp = best_dx * best_dy < 0;
    let cy = if opp {
        y0 - y1_w - 1.0
    } else {
        y0 - y1_w + 1.0
    };
    let x2_cross = if opp {
        (cy * x0 - cx * y0 + cx) / cy
    } else {
        (cy * x0 - cx * y0) / cy
    };
    let y2_cross = (cx * y0 - cy * x0) / cx;

    if opp {
        if x2_cross < 0.0 {
            let yw = wrap_torus_pos(y2_cross);
            let xw = wrap_torus_pos(x2_cross + 1.0);
            Ok(vec![
                [x0, y0, 0.0, yw],
                [torus_max, yw, xw, torus_max],
                [xw, 0.0, x1_w, y1_w],
            ])
        } else {
            let xw = wrap_torus_pos(x2_cross);
            let yw = wrap_torus_pos(y2_cross + 1.0);
            Ok(vec![
                [x0, y0, xw, torus_max],
                [xw, 0.0, 0.0, yw],
                [torus_max, yw, x1_w, y1_w],
            ])
        }
    } else if y2_cross < 0.0 {
        let xw = wrap_torus_pos(x2_cross);
        let yw = wrap_torus_pos(y2_cross + 1.0);
        Ok(vec![
            [x0, y0, xw, 0.0],
            [xw, torus_max, 0.0, yw],
            [torus_max, yw, x1_w, y1_w],
        ])
    } else {
        let yw = wrap_torus_pos(y2_cross);
        let xw = wrap_torus_pos(x2_cross + 1.0);
        Ok(vec![
            [x0, y0, 0.0, yw],
            [torus_max, yw, xw, 0.0],
            [xw, torus_max, x1_w, y1_w],
        ])
    }
}

/// Finds all crossing edge segment pairs for `graph` laid out on the flat unit torus `[0, 1)^2`.
pub fn torus_crossing_edges<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> CoreResult<Vec<TorusCrossing>> {
    let edges = extract_csr_edges(graph);
    torus_crossing_edges_from_edges(&edges, layout)
}

/// Finds all crossing edge segment pairs among `edges` on the flat unit torus `[0, 1)^2`.
pub fn torus_crossing_edges_from_edges<S: Scalar>(
    edges: &[(u32, u32)],
    layout: &Layout2d<S>,
) -> CoreResult<Vec<TorusCrossing>> {
    layout.validate_finite()?;
    let n = layout.len();
    let xs = layout.x();
    let ys = layout.y();

    let mut sub_segs: Vec<Seg> = Vec::with_capacity(edges.len());
    let mut orig_edge_of_sub: Vec<usize> = Vec::with_capacity(edges.len());

    for (edge_id, &(u, v)) in edges.iter().enumerate() {
        let ui = u as usize;
        let vi = v as usize;
        if ui >= n {
            return Err(CoreError::index_out_of_bounds(
                ui,
                n,
                "torus_crossing_edges_from_edges",
            ));
        }
        if vi >= n {
            return Err(CoreError::index_out_of_bounds(
                vi,
                n,
                "torus_crossing_edges_from_edges",
            ));
        }
        let pieces = torus_edge_segments(xs[ui], ys[ui], xs[vi], ys[vi])?;
        for piece in pieces {
            let sub_id = sub_segs.len();
            sub_segs.push(Seg::from_endpoints(
                sub_id, u, v, piece[0], piece[1], piece[2], piece[3],
            ));
            orig_edge_of_sub.push(edge_id);
        }
    }

    if sub_segs.len() > u32::MAX as usize {
        return Err(CoreError::invalid_input("edge count exceeds u32::MAX"));
    }
    let pairs = sweep_segments(&sub_segs);
    let mut out = Vec::with_capacity(pairs.len());
    for key in pairs {
        let (sub_a, sub_b) = unpack_pair(key);
        let sa = &sub_segs[sub_a];
        let sb = &sub_segs[sub_b];
        let ea = orig_edge_of_sub[sub_a];
        let eb = orig_edge_of_sub[sub_b];
        if ea <= eb {
            out.push(TorusCrossing {
                edge_a: ea,
                edge_b: eb,
                seg_a: [sa.x1, sa.y1, sa.x2, sa.y2],
                seg_b: [sb.x1, sb.y1, sb.x2, sb.y2],
            });
        } else {
            out.push(TorusCrossing {
                edge_a: eb,
                edge_b: ea,
                seg_a: [sb.x1, sb.y1, sb.x2, sb.y2],
                seg_b: [sa.x1, sa.y1, sa.x2, sa.y2],
            });
        }
    }
    out.sort_unstable_by(|a, b| {
        a.edge_a
            .cmp(&b.edge_a)
            .then_with(|| a.edge_b.cmp(&b.edge_b))
            .then_with(|| a.seg_a[0].total_cmp(&b.seg_a[0]))
            .then_with(|| a.seg_a[1].total_cmp(&b.seg_a[1]))
            .then_with(|| a.seg_a[2].total_cmp(&b.seg_a[2]))
            .then_with(|| a.seg_a[3].total_cmp(&b.seg_a[3]))
            .then_with(|| a.seg_b[0].total_cmp(&b.seg_b[0]))
            .then_with(|| a.seg_b[1].total_cmp(&b.seg_b[1]))
            .then_with(|| a.seg_b[2].total_cmp(&b.seg_b[2]))
            .then_with(|| a.seg_b[3].total_cmp(&b.seg_b[3]))
    });
    Ok(out)
}
