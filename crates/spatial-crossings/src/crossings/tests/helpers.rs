use egraph_core::{Layout2d, Scalar};

use super::super::SweepLineCrossings;
use super::super::TorusCrossing;
use super::super::geometry::{Seg, build_segments, segments_cross};
use super::super::torus::torus_edge_segments;

pub(super) fn brute_force_pairs<S: Scalar>(
    edges: &[(u32, u32)],
    layout: &Layout2d<S>,
) -> Vec<(usize, usize)> {
    let segs = build_segments(edges, layout).unwrap();
    let mut out = Vec::new();
    for i in 0..segs.len() {
        for j in (i + 1)..segs.len() {
            if segments_cross(&segs[i], &segs[j]) {
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

pub(super) fn torus_brute_force<S: Scalar>(
    edges: &[(u32, u32)],
    layout: &Layout2d<S>,
) -> Vec<TorusCrossing> {
    let mut sub_segs = Vec::new();
    let mut orig_edge = Vec::new();
    for (eid, &(u, v)) in edges.iter().enumerate() {
        let pieces = torus_edge_segments(
            layout.x()[u as usize],
            layout.y()[u as usize],
            layout.x()[v as usize],
            layout.y()[v as usize],
        )
        .unwrap();
        for p in pieces {
            let id = sub_segs.len();
            sub_segs.push(Seg::from_endpoints(id, u, v, p[0], p[1], p[2], p[3]));
            orig_edge.push(eid);
        }
    }
    let mut expected = Vec::new();
    for i in 0..sub_segs.len() {
        for j in (i + 1)..sub_segs.len() {
            if segments_cross(&sub_segs[i], &sub_segs[j]) {
                let ea = orig_edge[i];
                let eb = orig_edge[j];
                let sa = &sub_segs[i];
                let sb = &sub_segs[j];
                if ea <= eb {
                    expected.push(TorusCrossing {
                        edge_a: ea,
                        edge_b: eb,
                        seg_a: [sa.x1, sa.y1, sa.x2, sa.y2],
                        seg_b: [sb.x1, sb.y1, sb.x2, sb.y2],
                    });
                } else {
                    expected.push(TorusCrossing {
                        edge_a: eb,
                        edge_b: ea,
                        seg_a: [sb.x1, sb.y1, sb.x2, sb.y2],
                        seg_b: [sa.x1, sa.y1, sa.x2, sa.y2],
                    });
                }
            }
        }
    }
    expected.sort_unstable_by(|a, b| {
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
    expected
}
