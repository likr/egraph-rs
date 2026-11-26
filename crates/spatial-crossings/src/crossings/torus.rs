use std::cmp::Ordering;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, Layout2d, ManifoldKind, ManifoldLayout2d, Scalar,
};

use super::TorusCrossing;
use super::cross_pred::TorusExactPred;
use super::double_double::{Dd, dd_mul_f64};
use super::geometry::{Seg, extract_csr_edges};
use super::pair_set::{PairSet, pack_pair, unpack_pair};
use super::sweep::{sweep_segments_pred, sweep_segments_with_pred};
pub use super::torus_clip::torus_edge_segments;
use super::torus_clip::{
    ClippedTorusSeg, TORUS_MAX, best_axis_shift, clamp_unit, clip_shifted_seg_to_torus,
    dd_unwrapped_delta, torus_crossing_angle_metrics, torus_edge_segments_with_extents,
    wrap_torus_pos,
};
use super::torus_exact::{
    ShiftedF64, ShiftedPt, ShiftedSeg, dd_intersection_xy, exact_intersection_in_torus_max_box,
    exact_intersection_in_unit_box, exact_xi_cmp, exact_yi_cmp,
};
use super::torus_repair::{both_intersect, ensure_intersecting_pair};

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

struct TorusSubsegments {
    sub_segs: Vec<Seg>,
    shifted_segs: Vec<ShiftedSeg>,
    orig_edge_of_sub: Vec<usize>,
    orig_disps: Vec<(Dd, Dd)>,
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
    let mut shifted_segs: Vec<ShiftedSeg> = Vec::with_capacity(edges.len());
    let mut orig_edge_of_sub: Vec<usize> = Vec::with_capacity(edges.len());
    let mut orig_disps: Vec<(Dd, Dd)> = Vec::with_capacity(edges.len());

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
            dd_mul_f64(dd_unwrapped_delta(x0, x1_w, best_dx), lx),
            dd_mul_f64(dd_unwrapped_delta(y0, y1_w, best_dy), ly),
        ));

        let (cx_shifts, cx_len) = if best_dx != 0 {
            ([0, best_dx], 2usize)
        } else {
            ([0, 0], 1usize)
        };
        let (cy_shifts, cy_len) = if best_dy != 0 {
            ([0, best_dy], 2usize)
        } else {
            ([0, 0], 1usize)
        };

        // Emit only canonical lattice copies {0, best_dx} x {0, best_dy} whose exact
        // shifted bounding box overlaps `[0, 1)^2`, preserving exact `(base, shift)` coordinates.
        for &cy in &cy_shifts[..cy_len] {
            let sy0 = ShiftedF64::new(y0, -cy as i8);
            let sy1 = ShiftedF64::new(y1_w, (best_dy - cy) as i8);
            for &cx in &cx_shifts[..cx_len] {
                let sx0 = ShiftedF64::new(x0, -cx as i8);
                let sx1 = ShiftedF64::new(x1_w, (best_dx - cx) as i8);
                let p0 = ShiftedPt::new(sx0, sy0);
                let p1 = ShiftedPt::new(sx1, sy1);
                let meta = ShiftedSeg::from_shifted(p0, p1);
                if meta.x_max.cmp_exact(ShiftedF64::ZERO) == Ordering::Less
                    || meta.x_min.cmp_exact(ShiftedF64::ONE) != Ordering::Less
                    || meta.y_max.cmp_exact(ShiftedF64::ZERO) == Ordering::Less
                    || meta.y_min.cmp_exact(ShiftedF64::ONE) != Ordering::Less
                {
                    continue;
                }
                sub_segs.push(Seg::from_shifted(sub_segs.len(), u, v, p0, p1));
                shifted_segs.push(meta);
                orig_edge_of_sub.push(edge_id);
            }
        }
    }

    if sub_segs.len() > u32::MAX as usize {
        return Err(CoreError::invalid_input("edge count exceeds u32::MAX"));
    }
    Ok(TorusSubsegments {
        sub_segs,
        shifted_segs,
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
    let pred = TorusExactPred(&data.shifted_segs);
    let mut seen_edges = PairSet::default();
    sweep_segments_with_pred(&data.sub_segs, &pred, |sub_a, sub_b| {
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
    let pred = TorusExactPred(&data.shifted_segs);

    let pairs = sweep_segments_pred(&data.sub_segs, &pred);
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
        let ma = &data.shifted_segs[sa_ord.id];
        let mb = &data.shifted_segs[sb_ord.id];
        let (angle_rad, cos_sq) =
            torus_crossing_angle_metrics(data.orig_disps[edge_a], data.orig_disps[edge_b]);

        let in_box = exact_intersection_in_unit_box(ma.p1, ma.p2, mb.p1, mb.p2);
        let (kx, ky) = if in_box {
            (0_i8, 0_i8)
        } else {
            let kx = match exact_xi_cmp(ma.p1, ma.p2, mb.p1, mb.p2, 0.0) {
                Some(Ordering::Less) => 1,
                Some(_)
                    if exact_xi_cmp(ma.p1, ma.p2, mb.p1, mb.p2, 1.0) != Some(Ordering::Less) =>
                {
                    -1
                }
                _ => 0,
            };
            let ky = match exact_yi_cmp(ma.p1, ma.p2, mb.p1, mb.p2, 0.0) {
                Some(Ordering::Less) => 1,
                Some(_)
                    if exact_yi_cmp(ma.p1, ma.p2, mb.p1, mb.p2, 1.0) != Some(Ordering::Less) =>
                {
                    -1
                }
                _ => 0,
            };
            (kx, ky)
        };

        let shift_pt = |p: ShiftedPt| ShiftedPt::new(p.x.shifted_by(kx), p.y.shifted_by(ky));
        let non_gap = exact_intersection_in_torus_max_box(
            shift_pt(ma.p1),
            shift_pt(ma.p2),
            shift_pt(mb.p1),
            shift_pt(mb.p2),
            TORUS_MAX,
        );

        let clipped_a = clip_shifted_seg_to_torus(ma.p1, ma.p2, kx, ky);
        let clipped_b = clip_shifted_seg_to_torus(mb.p1, mb.p2, kx, ky);
        let clamp_shifted = |m: &ShiftedSeg| ClippedTorusSeg {
            coords: [
                clamp_unit(m.p1.x.shifted_by(kx).to_f64()),
                clamp_unit(m.p1.y.shifted_by(ky).to_f64()),
                clamp_unit(m.p2.x.shifted_by(kx).to_f64()),
                clamp_unit(m.p2.y.shifted_by(ky).to_f64()),
            ],
            free_dd: [None; 4],
        };
        let ca = clipped_a.unwrap_or_else(|| clamp_shifted(ma));
        let cb = clipped_b.unwrap_or_else(|| clamp_shifted(mb));

        let (seg_a, seg_b, final_intersect) = if non_gap {
            let (xi_s, yi_s) = dd_intersection_xy(
                shift_pt(ma.p1),
                shift_pt(ma.p2),
                shift_pt(mb.p1),
                shift_pt(mb.p2),
            )
            .unwrap_or((Dd::from_f64(ca.coords[0]), Dd::from_f64(ca.coords[1])));
            let (ra, rb) = ensure_intersecting_pair(&ca, &cb, xi_s, yi_s);
            (ra, rb, true)
        } else {
            let isect = both_intersect(ca.coords, cb.coords);
            (ca.coords, cb.coords, isect)
        };

        candidates.push((
            (in_box && final_intersect, final_intersect, in_box),
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
