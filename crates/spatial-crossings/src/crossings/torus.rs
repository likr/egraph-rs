use egraph_core::{
    CoreError, CoreResult, CsrGraph, Layout2d, ManifoldKind, ManifoldLayout2d, Scalar,
};

use super::TorusCrossing;
use super::geometry::{Seg, extract_csr_edges, segments_geometrically_intersect};
use super::pair_set::{PairSet, pack_pair, unpack_pair};
use super::sweep::{sweep_segments, sweep_segments_with};
pub use super::torus_clip::torus_edge_segments;
use super::torus_clip::{
    best_axis_shift, canon_zero, clamp_unit, clip_seg_to_shifted_box, clip_seg_to_unit_torus,
    crossing_point, torus_crossing_angle_metrics, torus_edge_segments_with_extents,
    unwrapped_delta, wrap_torus_pos,
};

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

        // For non-wrapping edges near `1.0`, also emit the Sterbenz-exact `shift = 1` copy
        // (`x - 1.0` in `[-0.6, 0)`) so sub-ULP endpoints near `0.0` do not lose precision via `1.0 + tiny`.
        let (cx_shifts, cx_len) = if best_dx != 0 {
            ([0, best_dx], 2usize)
        } else if x0 >= 0.9 || x1_w >= 0.9 {
            ([0, 1], 2usize)
        } else {
            ([0, 0], 1usize)
        };
        let (cy_shifts, cy_len) = if best_dy != 0 {
            ([0, best_dy], 2usize)
        } else if y0 >= 0.9 || y1_w >= 0.9 {
            ([0, 1], 2usize)
        } else {
            ([0, 0], 1usize)
        };

        // Emit all <= 4 integer lattice copies without pre-clipping to [0, 1]^2 so Sterbenz-exact
        // copies are preserved even when an intersection near a wrap edge lies slightly outside.
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
        let (seg_a, seg_b, final_intersect) = match (clipped_a, clipped_b) {
            (Some(ca), Some(cb)) if in_box && clipped_intersect => (ca, cb, true),
            (Some(ca), Some(cb)) if clipped_intersect => (ca, cb, true),
            _ => {
                let (kx, ky) = (canon_zero(-xi.floor()), canon_zero(-yi.floor()));
                let clip_shifted = |s: &Seg| {
                    clip_seg_to_shifted_box(s.x1, s.y1, s.x2, s.y2, kx, ky).unwrap_or([
                        clamp_unit(s.x1 + kx),
                        clamp_unit(s.y1 + ky),
                        clamp_unit(s.x2 + kx),
                        clamp_unit(s.y2 + ky),
                    ])
                };
                let ca = clip_shifted(sa_ord);
                let cb = clip_shifted(sb_ord);
                let shifted_intersect = segments_geometrically_intersect(
                    &Seg::from_endpoints(0, 0, 1, ca[0], ca[1], ca[2], ca[3]),
                    &Seg::from_endpoints(1, 2, 3, cb[0], cb[1], cb[2], cb[3]),
                );
                if shifted_intersect {
                    (ca, cb, true)
                } else if let (Some(ua), Some(ub)) = (clipped_a, clipped_b) {
                    if in_box {
                        (ua, ub, false)
                    } else {
                        (ca, cb, false)
                    }
                } else {
                    (ca, cb, false)
                }
            }
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
