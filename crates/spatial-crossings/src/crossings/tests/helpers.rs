use egraph_core::{Layout2d, Scalar, wrap_torus_coord};

use super::super::SweepLineCrossings;
use super::super::TorusCrossing;
use super::super::TorusLayoutView;
use super::super::double_double::dd_mul_f64;
use super::super::euclidean_geom::segments_geometrically_intersect;
use super::super::geometry::{Seg, build_segments};
use super::super::torus::torus_crossing_edges_from_edges;
use super::super::torus_clip::{dd_unwrapped_delta, torus_crossing_angle_metrics};
use super::super::torus_exact::{
    ShiftedF64, ShiftedPt, ShiftedSeg, segs_intersect_exact_f64, torus_segments_intersect,
};

pub(super) fn brute_force_pairs<S: Scalar>(
    edges: &[(u32, u32)],
    layout: &Layout2d<S>,
) -> Vec<(usize, usize)> {
    let segs = build_segments(edges, layout).unwrap();
    let mut out = Vec::new();
    for i in 0..segs.len() {
        for j in (i + 1)..segs.len() {
            if !segs[i].shares_endpoint(&segs[j])
                && segs_intersect_exact_f64(
                    [segs[i].x1, segs[i].y1, segs[i].x2, segs[i].y2],
                    [segs[j].x1, segs[j].y1, segs[j].x2, segs[j].y2],
                )
            {
                out.push((i, j));
            }
        }
    }
    out
}

pub(super) fn assert_matches_brute_force<S: Scalar>(
    label: &str,
    edges: &[(u32, u32)],
    layout: &Layout2d<S>,
) {
    let expected = brute_force_pairs(edges, layout);
    let got_pairs = SweepLineCrossings::find_edge_crossings(edges, layout).unwrap();
    let got_indices: Vec<(usize, usize)> = got_pairs.iter().map(|p| (p.edge_a, p.edge_b)).collect();
    assert_eq!(got_indices, expected, "{label}: crossing mismatch");
    assert_eq!(
        SweepLineCrossings::count_edges(edges, layout).unwrap(),
        expected.len(),
        "{label}: count_edges mismatch"
    );
    let summary = SweepLineCrossings::summarize_edges(edges, layout).unwrap();
    assert_eq!(summary.count, expected.len());
    let expected_cos_sq: f64 = got_pairs.iter().map(|p| p.cos_sq).sum();
    assert!(
        (summary.angle_cos_sq_sum - expected_cos_sq).abs()
            <= 1e-12 * (expected.len().max(1) as f64),
        "{label}: angle_cos_sq_sum mismatch"
    );
    let expected_min_ang = got_pairs
        .iter()
        .filter_map(|p| p.angle_rad)
        .reduce(f64::min);
    assert_eq!(summary.min_angle_rad, expected_min_ang);
}

pub(super) fn assert_streaming_matches_find(
    label: &str,
    edges: &[(u32, u32)],
    layout: &Layout2d<f64>,
) {
    let pairs = SweepLineCrossings::find_edge_crossings(edges, layout).unwrap();
    let count = SweepLineCrossings::count_edges(edges, layout).unwrap();
    let summary = SweepLineCrossings::summarize_edges(edges, layout).unwrap();
    assert_eq!(
        count,
        pairs.len(),
        "{label}: count_edges vs find_edge_crossings"
    );
    assert_eq!(
        summary.count,
        pairs.len(),
        "{label}: summary.count vs find_edge_crossings"
    );
    let expected_min_ang = pairs.iter().filter_map(|p| p.angle_rad).reduce(f64::min);
    assert_eq!(
        summary.min_angle_rad, expected_min_ang,
        "{label}: min_angle_rad mismatch"
    );
    let expected_cos_sq: f64 = pairs.iter().map(|p| p.cos_sq).sum();
    assert!(
        (summary.angle_cos_sq_sum - expected_cos_sq).abs() <= 1e-12 * (pairs.len().max(1) as f64),
        "{label}: angle_cos_sq_sum mismatch (got={}, expected={})",
        summary.angle_cos_sq_sum,
        expected_cos_sq
    );
}

#[derive(Debug, Clone, Copy)]
pub(super) struct OracleCrossing {
    pub(super) edge_a: usize,
    pub(super) edge_b: usize,
    pub(super) angle_rad: Option<f64>,
    pub(super) cos_sq: f64,
}

struct OracleBaseSeg {
    u: u32,
    v: u32,
    x0: f64,
    y0: f64,
    x1_w: f64,
    y1_w: f64,
    sx: i32,
    sy: i32,
    dx_phys: super::super::double_double::Dd,
    dy_phys: super::super::double_double::Dd,
}

/// Independent 9-shift O(M^2) torus crossing oracle using exact shifted minimum-image segments.
pub(super) fn torus_oracle_with_extents<S: Scalar>(
    edges: &[(u32, u32)],
    layout: &Layout2d<S>,
    lx: f64,
    ly: f64,
) -> Vec<OracleCrossing> {
    let mut base_segs: Vec<OracleBaseSeg> = Vec::with_capacity(edges.len());
    for &(u, v) in edges {
        let x0 = wrap_torus_coord(layout.x()[u as usize]).to_f64();
        let y0 = wrap_torus_coord(layout.y()[u as usize]).to_f64();
        let x1_w = wrap_torus_coord(layout.x()[v as usize]).to_f64();
        let y1_w = wrap_torus_coord(layout.y()[v as usize]).to_f64();
        let sx = if x1_w - x0 > 0.5 {
            -1
        } else if x1_w - x0 < -0.5 {
            1
        } else {
            0
        };
        let sy = if y1_w - y0 > 0.5 {
            -1
        } else if y1_w - y0 < -0.5 {
            1
        } else {
            0
        };
        base_segs.push(OracleBaseSeg {
            u,
            v,
            x0,
            y0,
            x1_w,
            y1_w,
            sx,
            sy,
            dx_phys: dd_mul_f64(dd_unwrapped_delta(x0, x1_w, sx), lx),
            dy_phys: dd_mul_f64(dd_unwrapped_delta(y0, y1_w, sy), ly),
        });
    }

    let mut out = Vec::new();
    for (i, a) in base_segs.iter().enumerate() {
        let ax_list = [0, a.sx];
        let ax_len = if a.sx == 0 { 1 } else { 2 };
        let ay_list = [0, a.sy];
        let ay_len = if a.sy == 0 { 1 } else { 2 };
        for (j, b) in base_segs.iter().enumerate().skip(i + 1) {
            if a.u == a.v || b.u == b.v || a.u == b.u || a.u == b.v || a.v == b.u || a.v == b.v {
                continue;
            }
            let bx_list = [0, b.sx];
            let bx_len = if b.sx == 0 { 1 } else { 2 };
            let by_list = [0, b.sy];
            let by_len = if b.sy == 0 { 1 } else { 2 };
            let mut crossed = false;
            'shift_loop: for &ay in &ay_list[..ay_len] {
                for &ax in &ax_list[..ax_len] {
                    let sa = ShiftedSeg::from_shifted(
                        ShiftedPt::new(
                            ShiftedF64::new(a.x0, -ax as i8),
                            ShiftedF64::new(a.y0, -ay as i8),
                        ),
                        ShiftedPt::new(
                            ShiftedF64::new(a.x1_w, (a.sx - ax) as i8),
                            ShiftedF64::new(a.y1_w, (a.sy - ay) as i8),
                        ),
                    );
                    for &by in &by_list[..by_len] {
                        for &bx in &bx_list[..bx_len] {
                            let sb = ShiftedSeg::from_shifted(
                                ShiftedPt::new(
                                    ShiftedF64::new(b.x0, -bx as i8),
                                    ShiftedF64::new(b.y0, -by as i8),
                                ),
                                ShiftedPt::new(
                                    ShiftedF64::new(b.x1_w, (b.sx - bx) as i8),
                                    ShiftedF64::new(b.y1_w, (b.sy - by) as i8),
                                ),
                            );
                            if torus_segments_intersect(&sa, &sb) {
                                crossed = true;
                                break 'shift_loop;
                            }
                        }
                    }
                }
            }
            if crossed {
                let (angle_rad, cos_sq) =
                    torus_crossing_angle_metrics((a.dx_phys, a.dy_phys), (b.dx_phys, b.dy_phys));
                out.push(OracleCrossing {
                    edge_a: i,
                    edge_b: j,
                    angle_rad,
                    cos_sq,
                });
            }
        }
    }
    out
}

pub(super) fn assert_torus_matches_oracle<S: Scalar, L: TorusLayoutView<S> + ?Sized>(
    label: &str,
    edges: &[(u32, u32)],
    view: &L,
) -> Vec<TorusCrossing> {
    let (lx, ly) = view.torus_extents().unwrap();
    let coords = view.torus_layout().unwrap();
    let expected = torus_oracle_with_extents(edges, coords, lx, ly);
    let got = torus_crossing_edges_from_edges(edges, view).unwrap();
    assert_eq!(
        got.len(),
        expected.len(),
        "{label}: count mismatch (got={}, expected={})",
        got.len(),
        expected.len()
    );
    for (g, e) in got.iter().zip(&expected) {
        assert_eq!(g.edge_a, e.edge_a, "{label}: edge_a mismatch");
        assert_eq!(g.edge_b, e.edge_b, "{label}: edge_b mismatch");
        assert_eq!((g.u1, g.v1), edges[g.edge_a], "{label}: edge_a endpoints");
        assert_eq!((g.u2, g.v2), edges[g.edge_b], "{label}: edge_b endpoints");
        match (g.angle_rad, e.angle_rad) {
            (Some(ga), Some(ea)) => {
                assert!(
                    (ga - ea).abs() <= 1e-11,
                    "{label}: angle mismatch for ({}, {}): got {ga}, expected {ea}",
                    g.edge_a,
                    g.edge_b
                );
            }
            (None, None) => {}
            _ => panic!(
                "{label}: angle Option mismatch for ({}, {}): got {:?}, expected {:?}",
                g.edge_a, g.edge_b, g.angle_rad, e.angle_rad
            ),
        }
        assert!(
            (g.cos_sq - e.cos_sq).abs() <= 1e-11,
            "{label}: cos_sq mismatch for ({}, {}): got {}, expected {}",
            g.edge_a,
            g.edge_b,
            g.cos_sq,
            e.cos_sq
        );
        for c in g.seg_a.into_iter().chain(g.seg_b) {
            assert!(
                c.is_finite() && (0.0..1.0).contains(&c),
                "{label}: TorusCrossing seg coord out of [0, 1): {c}"
            );
            assert_ne!(
                c.to_bits(),
                (-0.0_f64).to_bits(),
                "{label}: TorusCrossing seg coord has -0.0"
            );
        }
    }
    got
}

pub(super) fn assert_torus_pair_intersects(label: &str, c: &TorusCrossing) {
    for coord in c.seg_a.into_iter().chain(c.seg_b) {
        assert!(
            coord.is_finite() && (0.0..1.0).contains(&coord),
            "{label}: TorusCrossing coord out of [0, 1): {coord}"
        );
        assert_ne!(
            coord.to_bits(),
            (-0.0_f64).to_bits(),
            "{label}: TorusCrossing coord has -0.0"
        );
    }
    assert!(
        segs_intersect_exact_f64(c.seg_a, c.seg_b),
        "{label}: clipped TorusCrossing segments must exact-intersect: seg_a={:?}, seg_b={:?}",
        c.seg_a,
        c.seg_b
    );
    assert!(
        segments_geometrically_intersect(
            &Seg::from_endpoints(
                0, c.u1, c.v1, c.seg_a[0], c.seg_a[1], c.seg_a[2], c.seg_a[3]
            ),
            &Seg::from_endpoints(
                1, c.u2, c.v2, c.seg_b[0], c.seg_b[1], c.seg_b[2], c.seg_b[3]
            ),
        ),
        "{label}: clipped TorusCrossing segments must intersect: seg_a={:?}, seg_b={:?}",
        c.seg_a,
        c.seg_b
    );
}
