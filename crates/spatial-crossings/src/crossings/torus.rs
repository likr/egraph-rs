use std::f64::consts::FRAC_PI_2;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, Layout2d, ManifoldKind, ManifoldLayout2d, Scalar,
    wrap_torus_coord,
};

use super::TorusCrossing;
use super::geometry::{Seg, extract_csr_edges};
use super::pair_set::unpack_pair;
use super::sweep::{sweep_segments, sweep_segments_with};

/// Read-only view of a 2D torus layout providing normalized coordinates and physical extents `(Lx, Ly)`.
pub trait TorusLayoutView<S: Scalar> {
    /// Returns the physical periodic box extents `(lx, ly)` (`> 0`).
    fn torus_extents(&self) -> CoreResult<(f64, f64)>;
    /// Returns the underlying normalized `[0, 1)^2` coordinate buffer.
    fn torus_layout(&self) -> CoreResult<&Layout2d<S>>;
}

impl<S: Scalar> TorusLayoutView<S> for Layout2d<S> {
    #[inline]
    fn torus_extents(&self) -> CoreResult<(f64, f64)> {
        Ok((1.0, 1.0))
    }

    #[inline]
    fn torus_layout(&self) -> CoreResult<&Layout2d<S>> {
        Ok(self)
    }
}

impl<S: Scalar> TorusLayoutView<S> for ManifoldLayout2d<S> {
    #[inline]
    fn torus_extents(&self) -> CoreResult<(f64, f64)> {
        if self.kind() != ManifoldKind::Torus2d {
            return Err(CoreError::unsupported(
                "torus crossing operations require ManifoldKind::Torus2d",
            ));
        }
        let ly = self.scale();
        let lx = ly * self.aspect();
        if !lx.is_finite() || !ly.is_finite() {
            return Err(CoreError::non_finite("TorusLayoutView::torus_extents"));
        }
        if lx <= 0.0 || ly <= 0.0 {
            return Err(CoreError::invalid_input(
                "torus extents lx and ly must be strictly positive",
            ));
        }
        Ok((lx, ly))
    }

    #[inline]
    fn torus_layout(&self) -> CoreResult<&Layout2d<S>> {
        if self.kind() != ManifoldKind::Torus2d {
            return Err(CoreError::unsupported(
                "torus crossing operations require ManifoldKind::Torus2d",
            ));
        }
        Ok(self.as_layout())
    }
}

#[inline]
fn wrap_torus_pos<S: Scalar>(value: S) -> f64 {
    let w = wrap_torus_coord(value).to_f64();
    if w == 0.0 { 0.0 } else { w }
}

/// Splits a 2D torus edge `(x1, y1) -> (x2, y2)` into 1, 2, or 3 Euclidean segments on `[0, 1]^2`
/// under unit `[0, 1)^2` aspect ratio.
pub fn torus_edge_segments<S: Scalar>(x1: S, y1: S, x2: S, y2: S) -> CoreResult<Vec<[f64; 4]>> {
    torus_edge_segments_with_extents(x1, y1, x2, y2, 1.0, 1.0)
}

fn torus_edge_segments_with_extents<S: Scalar>(
    x1: S,
    y1: S,
    x2: S,
    y2: S,
    lx: f64,
    ly: f64,
) -> CoreResult<Vec<[f64; 4]>> {
    if !x1.is_finite() || !y1.is_finite() || !x2.is_finite() || !y2.is_finite() {
        return Err(CoreError::non_finite("torus_edge_segments"));
    }
    if !lx.is_finite() || !ly.is_finite() {
        return Err(CoreError::non_finite("torus_edge_segments::extents"));
    }
    if lx <= 0.0 || ly <= 0.0 {
        return Err(CoreError::invalid_input(
            "torus extents lx and ly must be strictly positive",
        ));
    }

    let px = wrap_torus_pos(x1);
    let py = wrap_torus_pos(y1);
    let qx = wrap_torus_pos(x2);
    let qy = wrap_torus_pos(y2);

    // Find periodic shift (dx, dy) in {-1, 0, +1}^2 minimizing physical distance.
    let mut best_d = f64::INFINITY;
    let mut best_dx = 0_i32;
    let mut best_dy = 0_i32;
    for dy in -1..=1 {
        let y_shift = py + dy as f64;
        for dx in -1..=1 {
            let x_shift = px + dx as f64;
            let d = (lx * (x_shift - qx)).hypot(ly * (y_shift - qy));
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

#[inline]
fn torus_crossing_angle_metrics(a: &Seg, b: &Seg, lx: f64, ly: f64) -> (Option<f64>, f64) {
    let dx1 = lx * (a.x2 - a.x1);
    let dy1 = ly * (a.y2 - a.y1);
    let dx2 = lx * (b.x2 - b.x1);
    let dy2 = ly * (b.y2 - b.y1);
    let len1 = dx1.hypot(dy1);
    let len2 = dx2.hypot(dy2);
    if len1 == 0.0 || len2 == 0.0 {
        return (None, 0.0);
    }
    let ux1 = dx1 / len1;
    let uy1 = dy1 / len1;
    let ux2 = dx2 / len2;
    let uy2 = dy2 / len2;
    let cos_val = (ux1 * ux2 + uy1 * uy2).clamp(-1.0, 1.0);
    let acute = cos_val.abs().acos().min(FRAC_PI_2);
    let cos_sq = cos_val * cos_val;
    (
        Some(if acute == 0.0 { 0.0 } else { acute }),
        if cos_sq == 0.0 { 0.0 } else { cos_sq },
    )
}

fn build_torus_subsegments<S: Scalar>(
    edges: &[(u32, u32)],
    layout: &Layout2d<S>,
    lx: f64,
    ly: f64,
    context: &'static str,
) -> CoreResult<(Vec<Seg>, Vec<usize>)> {
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
            return Err(CoreError::index_out_of_bounds(ui, n, context));
        }
        if vi >= n {
            return Err(CoreError::index_out_of_bounds(vi, n, context));
        }
        let pieces = torus_edge_segments_with_extents(xs[ui], ys[ui], xs[vi], ys[vi], lx, ly)?;
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
    Ok((sub_segs, orig_edge_of_sub))
}

/// Returns all piecewise Euclidean edge sub-segments in `[0, 1)^2` for `graph` under `layout`,
/// respecting periodic wrap-around under physical torus extents `(Lx, Ly)`.
pub fn torus_layout_edge_segments<W, S: Scalar, L: TorusLayoutView<S> + ?Sized>(
    graph: &CsrGraph<W>,
    layout: &L,
) -> CoreResult<Vec<[f64; 4]>> {
    let (lx, ly) = layout.torus_extents()?;
    let coords = layout.torus_layout()?;
    coords.validate_finite()?;
    let n = coords.len();
    let xs = coords.x();
    let ys = coords.y();
    let edges = extract_csr_edges(graph);
    let mut out = Vec::with_capacity(edges.len());
    for &(u, v) in &edges {
        let ui = u as usize;
        let vi = v as usize;
        if ui >= n {
            return Err(CoreError::index_out_of_bounds(
                ui,
                n,
                "torus_layout_edge_segments",
            ));
        }
        if vi >= n {
            return Err(CoreError::index_out_of_bounds(
                vi,
                n,
                "torus_layout_edge_segments",
            ));
        }
        out.extend(torus_edge_segments_with_extents(
            xs[ui], ys[ui], xs[vi], ys[vi], lx, ly,
        )?);
    }
    Ok(out)
}

/// Counts all crossing edge segment pairs for `graph` on a flat 2D torus layout
/// (accepting `&Layout2d<S>` or `&ManifoldLayout2d<S>` of kind [`ManifoldKind::Torus2d`]).
pub fn torus_crossing_number<W, S: Scalar, L: TorusLayoutView<S> + ?Sized>(
    graph: &CsrGraph<W>,
    layout: &L,
) -> CoreResult<usize> {
    let (lx, ly) = layout.torus_extents()?;
    let coords = layout.torus_layout()?;
    let edges = extract_csr_edges(graph);
    let (sub_segs, _) = build_torus_subsegments(&edges, coords, lx, ly, "torus_crossing_number")?;
    let mut count = 0usize;
    sweep_segments_with(&sub_segs, |_, _| {
        count += 1;
    });
    Ok(count)
}

/// Finds all crossing edge segment pairs for `graph` laid out on a flat 2D torus
/// (accepting `&Layout2d<S>` or `&ManifoldLayout2d<S>` of kind [`ManifoldKind::Torus2d`]).
pub fn torus_crossing_edges<W, S: Scalar, L: TorusLayoutView<S> + ?Sized>(
    graph: &CsrGraph<W>,
    layout: &L,
) -> CoreResult<Vec<TorusCrossing>> {
    let edges = extract_csr_edges(graph);
    torus_crossing_edges_from_edges(&edges, layout)
}

/// Finds all crossing edge segment pairs among `edges` on a flat 2D torus
/// (accepting `&Layout2d<S>` or `&ManifoldLayout2d<S>` of kind [`ManifoldKind::Torus2d`]).
pub fn torus_crossing_edges_from_edges<S: Scalar, L: TorusLayoutView<S> + ?Sized>(
    edges: &[(u32, u32)],
    layout: &L,
) -> CoreResult<Vec<TorusCrossing>> {
    let (lx, ly) = layout.torus_extents()?;
    let coords = layout.torus_layout()?;
    let (sub_segs, orig_edge_of_sub) =
        build_torus_subsegments(edges, coords, lx, ly, "torus_crossing_edges_from_edges")?;

    let pairs = sweep_segments(&sub_segs);
    let mut out = Vec::with_capacity(pairs.len());
    for key in pairs {
        let (sub_a, sub_b) = unpack_pair(key);
        let sa = &sub_segs[sub_a];
        let sb = &sub_segs[sub_b];
        let ea = orig_edge_of_sub[sub_a];
        let eb = orig_edge_of_sub[sub_b];
        let (angle_rad, cos_sq) = torus_crossing_angle_metrics(sa, sb, lx, ly);
        if ea <= eb {
            out.push(TorusCrossing {
                edge_a: ea,
                edge_b: eb,
                u1: sa.u,
                v1: sa.v,
                u2: sb.u,
                v2: sb.v,
                seg_a: [sa.x1, sa.y1, sa.x2, sa.y2],
                seg_b: [sb.x1, sb.y1, sb.x2, sb.y2],
                angle_rad,
                cos_sq,
            });
        } else {
            out.push(TorusCrossing {
                edge_a: eb,
                edge_b: ea,
                u1: sb.u,
                v1: sb.v,
                u2: sa.u,
                v2: sa.v,
                seg_a: [sb.x1, sb.y1, sb.x2, sb.y2],
                seg_b: [sa.x1, sa.y1, sa.x2, sa.y2],
                angle_rad,
                cos_sq,
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
