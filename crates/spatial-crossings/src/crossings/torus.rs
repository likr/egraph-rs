use std::f64::consts::FRAC_PI_2;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, Layout2d, ManifoldKind, ManifoldLayout2d, Scalar,
    wrap_torus_coord,
};

use super::TorusCrossing;
use super::geometry::{Seg, extract_csr_edges, segments_geometrically_intersect};
use super::pair_set::{PairSet, pack_pair, unpack_pair};
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
    canon_zero(wrap_torus_coord(value).to_f64())
}

/// Largest IEEE-754 f64 strictly below 1.0 so clipped endpoints stay in [0, 1) with zero ULP gap.
const TORUS_MAX: f64 = 1.0 - 0.5 * f64::EPSILON;

#[inline]
fn canon_zero(v: f64) -> f64 {
    if v == 0.0 { 0.0 } else { v }
}

#[inline]
fn clamp_unit(v: f64) -> f64 {
    canon_zero(v.clamp(0.0, TORUS_MAX))
}

#[inline]
fn best_axis_shift(p0: f64, p1: f64) -> i32 {
    let d = p1 - p0;
    if d > 0.5 {
        -1
    } else if d < -0.5 {
        1
    } else {
        0
    }
}

#[inline]
fn unwrapped_delta(p0: f64, p1: f64, shift: i32) -> f64 {
    match shift {
        0 => p1 - p0,
        1 => (1.0 - p0) + p1,
        _ => (p1 - 1.0) - p0,
    }
}

#[inline]
fn axis_cut(p0: f64, d: f64, shift: i32) -> Option<f64> {
    match shift {
        -1 => Some(canon_zero((-p0 / d).clamp(0.0, 1.0))),
        1 => Some(canon_zero(((1.0 - p0) / d).clamp(0.0, 1.0))),
        _ => None,
    }
}

/// Splits a 2D torus edge `(x1, y1) -> (x2, y2)` into 1, 2, or 3 Euclidean segments in `[0, 1)^2`
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

    let x0 = wrap_torus_pos(x1);
    let y0 = wrap_torus_pos(y1);
    let x1_w = wrap_torus_pos(x2);
    let y1_w = wrap_torus_pos(y2);

    // Since the torus box is axis-aligned, minimizing physical distance decouples per axis.
    let best_dx = best_axis_shift(x0, x1_w);
    let best_dy = best_axis_shift(y0, y1_w);

    if best_dx == 0 && best_dy == 0 {
        return Ok(vec![[x0, y0, x1_w, y1_w]]);
    }

    let dx = unwrapped_delta(x0, x1_w, best_dx);
    let dy = unwrapped_delta(y0, y1_w, best_dy);

    let mut cuts = [0.0_f64, 1.0, 0.0, 0.0];
    let mut n_cuts = 2usize;
    let tx = axis_cut(x0, dx, best_dx);
    let ty = axis_cut(y0, dy, best_dy);
    for t in [tx, ty].into_iter().flatten() {
        cuts[n_cuts] = t;
        n_cuts += 1;
    }
    cuts[..n_cuts].sort_unstable_by(f64::total_cmp);

    let eval_axis = |t: f64, p0: f64, p1_w: f64, d: f64, shift: i32, cell: i32, tc: Option<f64>| {
        if tc == Some(t) {
            return match (shift, cell) {
                (1, 0) | (-1, -1) => TORUS_MAX,
                (1, 1) | (-1, 0) => 0.0,
                _ => clamp_unit(p0 - cell as f64 + t * d),
            };
        }
        let raw = if t == 0.0 {
            p0 - cell as f64
        } else if t == 1.0 {
            p1_w + (shift - cell) as f64
        } else {
            (p0 - cell as f64) + t * d
        };
        clamp_unit(raw)
    };

    let mut pieces = Vec::with_capacity(n_cuts - 1);
    for w in cuts[..n_cuts].windows(2) {
        let (t_a, t_b) = (w[0], w[1]);
        if t_b <= t_a {
            continue;
        }
        let (cx, cy) = (
            tx.map_or(0, |tc| if t_a >= tc { best_dx } else { 0 }),
            ty.map_or(0, |tc| if t_a >= tc { best_dy } else { 0 }),
        );
        pieces.push([
            eval_axis(t_a, x0, x1_w, dx, best_dx, cx, tx),
            eval_axis(t_a, y0, y1_w, dy, best_dy, cy, ty),
            eval_axis(t_b, x0, x1_w, dx, best_dx, cx, tx),
            eval_axis(t_b, y0, y1_w, dy, best_dy, cy, ty),
        ]);
    }
    Ok(pieces)
}

#[inline]
fn clip_seg_to_unit_torus(ax: f64, ay: f64, bx: f64, by: f64) -> Option<[f64; 4]> {
    let (dx, dy) = (bx - ax, by - ay);
    let (mut t0, mut t1) = (0.0_f64, 1.0_f64);
    for (p, q) in [(-dx, ax), (dx, 1.0 - ax), (-dy, ay), (dy, 1.0 - ay)] {
        if p == 0.0 {
            if q < -1e-12 {
                return None;
            }
        } else if p < 0.0 {
            t0 = t0.max(q / p);
        } else {
            t1 = t1.min(q / p);
        }
    }
    if t0 > t1 + 1e-12 {
        return None;
    }
    let ta = t0.clamp(0.0, 1.0);
    let tb = t1.max(t0).clamp(0.0, 1.0);
    let eval = |a: f64, b: f64, d: f64, t: f64| {
        let v = if t == 0.0 {
            a
        } else if t == 1.0 {
            b
        } else {
            a + t * d
        };
        clamp_unit(v)
    };
    Some([
        eval(ax, bx, dx, ta),
        eval(ay, by, dy, ta),
        eval(ax, bx, dx, tb),
        eval(ay, by, dy, tb),
    ])
}

#[inline]
fn crossing_point(a: &Seg, b: &Seg) -> (f64, f64) {
    let (dx1, dy1, dx2, dy2) = (a.x2 - a.x1, a.y2 - a.y1, b.x2 - b.x1, b.y2 - b.y1);
    let det = dx1 * dy2 - dy1 * dx2;
    if det != 0.0 {
        let t = ((b.x1 - a.x1) * dy2 - (b.y1 - a.y1) * dx2) / det;
        (
            (a.x1 + t * dx1).clamp(a.x_min.max(b.x_min), a.x_max.min(b.x_max)),
            (a.y1 + t * dy1).clamp(a.y_min.max(b.y_min), a.y_max.min(b.y_max)),
        )
    } else {
        (
            0.5 * (a.x_min.max(b.x_min) + a.x_max.min(b.x_max)),
            0.5 * (a.y_min.max(b.y_min) + a.y_max.min(b.y_max)),
        )
    }
}

#[inline]
fn torus_crossing_angle_metrics(
    (dx1, dy1): (f64, f64),
    (dx2, dy2): (f64, f64),
) -> (Option<f64>, f64) {
    let (len1, len2) = (dx1.hypot(dy1), dx2.hypot(dy2));
    if len1 == 0.0 || len2 == 0.0 {
        return (None, 0.0);
    }
    let cos_val = ((dx1 / len1) * (dx2 / len2) + (dy1 / len1) * (dy2 / len2)).clamp(-1.0, 1.0);
    let acute = cos_val.abs().acos().min(FRAC_PI_2);
    let cos_sq = cos_val * cos_val;
    (Some(canon_zero(acute)), canon_zero(cos_sq))
}

struct TorusSubsegments {
    sub_segs: Vec<Seg>,
    orig_edge_of_sub: Vec<usize>,
    orig_disps: Vec<(f64, f64)>,
}

fn build_torus_subsegments<S: Scalar>(
    edges: &[(u32, u32)],
    layout: &Layout2d<S>,
    lx: f64,
    ly: f64,
    context: &'static str,
) -> CoreResult<TorusSubsegments> {
    layout.validate_finite()?;
    let (n, xs, ys) = (layout.len(), layout.x(), layout.y());

    let mut sub_segs: Vec<Seg> = Vec::with_capacity(edges.len());
    let mut orig_edge_of_sub: Vec<usize> = Vec::with_capacity(edges.len());
    let mut orig_disps: Vec<(f64, f64)> = Vec::with_capacity(edges.len());

    for (edge_id, &(u, v)) in edges.iter().enumerate() {
        let (ui, vi) = (u as usize, v as usize);
        if ui >= n || vi >= n {
            return Err(CoreError::index_out_of_bounds(ui.max(vi), n, context));
        }
        let (x0, y0) = (wrap_torus_pos(xs[ui]), wrap_torus_pos(ys[ui]));
        let (x1_w, y1_w) = (wrap_torus_pos(xs[vi]), wrap_torus_pos(ys[vi]));
        let best_dx = best_axis_shift(x0, x1_w);
        let best_dy = best_axis_shift(y0, y1_w);
        orig_disps.push((
            lx * unwrapped_delta(x0, x1_w, best_dx),
            ly * unwrapped_delta(y0, y1_w, best_dy),
        ));

        let cx_shifts = [0, best_dx];
        let cx_len = if best_dx == 0 { 1 } else { 2 };
        let cy_shifts = [0, best_dy];
        let cy_len = if best_dy == 0 { 1 } else { 2 };

        // Emit all <= 4 integer lattice copies without pre-clipping to [0, 1]^2 so Sterbenz-exact
        // (0, 0)-centered copies are preserved even when an intersection near a wrap edge lies slightly outside.
        for &cy in &cy_shifts[..cy_len] {
            let sy0 = canon_zero(if cy == 0 { y0 } else { y0 - cy as f64 });
            let sy1 = canon_zero(y1_w + (best_dy - cy) as f64);
            for &cx in &cx_shifts[..cx_len] {
                let sx0 = canon_zero(if cx == 0 { x0 } else { x0 - cx as f64 });
                let sx1 = canon_zero(x1_w + (best_dx - cx) as f64);
                sub_segs.push(Seg::from_endpoints(
                    sub_segs.len(),
                    u,
                    v,
                    sx0,
                    sy0,
                    sx1,
                    sy1,
                ));
                orig_edge_of_sub.push(edge_id);
            }
        }
    }

    if sub_segs.len() > u32::MAX as usize {
        return Err(CoreError::invalid_input("edge count exceeds u32::MAX"));
    }
    Ok(TorusSubsegments {
        sub_segs,
        orig_edge_of_sub,
        orig_disps,
    })
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
    let (n, xs, ys) = (coords.len(), coords.x(), coords.y());
    let edges = extract_csr_edges(graph);
    let mut out = Vec::with_capacity(edges.len());
    for &(u, v) in &edges {
        let (ui, vi) = (u as usize, v as usize);
        if ui >= n || vi >= n {
            return Err(CoreError::index_out_of_bounds(
                ui.max(vi),
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
    let data = build_torus_subsegments(&edges, coords, lx, ly, "torus_crossing_number")?;
    let mut seen_edges = PairSet::default();
    sweep_segments_with(&data.sub_segs, |sub_a, sub_b| {
        let ea = data.orig_edge_of_sub[sub_a];
        let eb = data.orig_edge_of_sub[sub_b];
        if ea != eb {
            seen_edges.insert(pack_pair(ea, eb));
        }
    });
    Ok(seen_edges.len())
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
    let data = build_torus_subsegments(edges, coords, lx, ly, "torus_crossing_edges_from_edges")?;

    let pairs = sweep_segments(&data.sub_segs);
    let mut candidates: Vec<((bool, bool, bool), TorusCrossing)> = Vec::with_capacity(pairs.len());
    for key in pairs {
        let (sub_a, sub_b) = unpack_pair(key);
        let sa = &data.sub_segs[sub_a];
        let sb = &data.sub_segs[sub_b];
        let ea = data.orig_edge_of_sub[sub_a];
        let eb = data.orig_edge_of_sub[sub_b];
        if ea == eb {
            continue;
        }
        let (edge_a, edge_b, sa_ord, sb_ord) = if ea <= eb {
            (ea, eb, sa, sb)
        } else {
            (eb, ea, sb, sa)
        };
        let (angle_rad, cos_sq) =
            torus_crossing_angle_metrics(data.orig_disps[edge_a], data.orig_disps[edge_b]);
        let (xi, yi) = crossing_point(sa_ord, sb_ord);
        let in_box = (0.0..1.0).contains(&xi) && (0.0..1.0).contains(&yi);
        let clipped_a = clip_seg_to_unit_torus(sa_ord.x1, sa_ord.y1, sa_ord.x2, sa_ord.y2);
        let clipped_b = clip_seg_to_unit_torus(sb_ord.x1, sb_ord.y1, sb_ord.x2, sb_ord.y2);
        let clipped_intersect = match (clipped_a, clipped_b) {
            (Some(ca), Some(cb)) => segments_geometrically_intersect(
                &Seg::from_endpoints(0, 0, 1, ca[0], ca[1], ca[2], ca[3]),
                &Seg::from_endpoints(1, 2, 3, cb[0], cb[1], cb[2], cb[3]),
            ),
            _ => false,
        };
        let (seg_a, seg_b) = match (clipped_a, clipped_b) {
            (Some(ca), Some(cb)) if in_box || clipped_intersect => (ca, cb),
            _ => {
                let (kx, ky) = (canon_zero(-xi.floor()), canon_zero(-yi.floor()));
                let clip_shifted = |s: &Seg| {
                    let (x1, y1, x2, y2) = (s.x1 + kx, s.y1 + ky, s.x2 + kx, s.y2 + ky);
                    clip_seg_to_unit_torus(x1, y1, x2, y2).unwrap_or([
                        clamp_unit(x1),
                        clamp_unit(y1),
                        clamp_unit(x2),
                        clamp_unit(y2),
                    ])
                };
                (clip_shifted(sa_ord), clip_shifted(sb_ord))
            }
        };
        candidates.push((
            (in_box && clipped_intersect, clipped_intersect, in_box),
            TorusCrossing {
                edge_a,
                edge_b,
                u1: sa_ord.u,
                v1: sa_ord.v,
                u2: sb_ord.u,
                v2: sb_ord.v,
                seg_a,
                seg_b,
                angle_rad,
                cos_sq,
            },
        ));
    }
    let cmp_seg = |p: &[f64; 4], q: &[f64; 4]| {
        p[0].total_cmp(&q[0])
            .then_with(|| p[1].total_cmp(&q[1]))
            .then_with(|| p[2].total_cmp(&q[2]))
            .then_with(|| p[3].total_cmp(&q[3]))
    };
    candidates.sort_unstable_by(|(ra, a), (rb, b)| {
        a.edge_a
            .cmp(&b.edge_a)
            .then_with(|| a.edge_b.cmp(&b.edge_b))
            .then_with(|| rb.cmp(ra))
            .then_with(|| a.angle_rad.is_none().cmp(&b.angle_rad.is_none()))
            .then_with(|| cmp_seg(&a.seg_a, &b.seg_a))
            .then_with(|| cmp_seg(&a.seg_b, &b.seg_b))
    });
    let mut seen_edges = PairSet::default();
    let mut out = Vec::with_capacity(candidates.len());
    for (_, item) in candidates {
        if seen_edges.insert(pack_pair(item.edge_a, item.edge_b)) {
            out.push(item);
        }
    }
    Ok(out)
}
