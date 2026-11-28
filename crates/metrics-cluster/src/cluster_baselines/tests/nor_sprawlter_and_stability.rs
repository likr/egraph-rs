use super::super::space_dispatch::metric_diameter;
use super::super::sprawlter::rect_union_area;
use super::*;
use egraph_core::{CsrGraph, DeterministicRng, NodeExtents2d, NodeShape2d};

#[test]
fn test_fixture_d_nor_and_sprawlter_nn_exact_parity() {
    let orig_d = layout_from_pairs(&[(0.0, 0.0), (0.5, 0.2), (2.0, 0.0), (2.2, 1.5)]);
    let adj_d = layout_from_pairs(&[(0.0, 0.0), (1.2, 0.4), (3.0, 0.1), (3.2, 2.0)]);
    let hw_d = [0.5, 0.5, 0.4, 0.6];
    let hh_d = [0.4, 0.4, 0.3, 0.5];
    let g_d = CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3), (0, 2)], None, false, false)
        .unwrap();

    let ext_d = NodeExtents2d::new(&hw_d, &hh_d).unwrap();
    let nor = nor_metrics(&g_d, &orig_d, &adj_d, ext_d).unwrap();
    assert_close(nor.oo_nni, 0.0, 1e-10, "D.oo_nni");
    assert_close(nor.sp_ch_a, 1.455_089_820_359_281_7, 1e-10, "D.sp_ch_a");
    assert_close(nor.gs_bb_iar, 1.078_369_905_956_113, 1e-10, "D.gs_bb_iar");
    assert_close(
        nor.nm_dm_imse,
        0.064_878_328_741_965_13,
        1e-10,
        "D.nm_dm_imse",
    );
    assert_close(nor.el_rsd, 0.289_816_821_614_026_05, 1e-10, "D.el_rsd");

    let nor_ng = nor_metrics_without_graph(&orig_d, &adj_d, ext_d).unwrap();
    assert_close(nor_ng.oo_nni, nor.oo_nni, 1e-15, "D.ng.oo_nni");
    assert_close(nor_ng.sp_ch_a, nor.sp_ch_a, 1e-15, "D.ng.sp_ch_a");
    assert_close(nor_ng.gs_bb_iar, nor.gs_bb_iar, 1e-15, "D.ng.gs_bb_iar");
    assert_close(nor_ng.nm_dm_imse, nor.nm_dm_imse, 1e-15, "D.ng.nm_dm_imse");
    assert_eq!(nor_ng.el_rsd, 0.0);

    assert_close(
        sprawlter_nn(&orig_d, ext_d).unwrap(),
        1.8015717518758179,
        1e-10,
        "D.sprawlter.orig",
    );
    assert_close(
        sprawlter_nn(&adj_d, ext_d).unwrap(),
        1.9498280099261789,
        1e-10,
        "D.sprawlter.adj",
    );

    // Translation and uniform scaling invariance for nm_dm_imse.
    let scaled_translated = Layout2d::from_xy(
        orig_d.x().iter().map(|&x| 3.5 * x + 12.0).collect(),
        orig_d.y().iter().map(|&y| 3.5 * y - 7.0).collect(),
    )
    .unwrap();
    let inv_nor = nor_metrics(&g_d, &orig_d, &scaled_translated, ext_d).unwrap();
    assert!(inv_nor.nm_dm_imse < 1e-24);
    assert!(inv_nor.el_rsd < 1e-14);

    // Verify NodeExtents2d::from_shapes and NodeExtents2d::uniform parity.
    let shapes_d = [
        NodeShape2d::rectangle(0.5, 0.4).unwrap(),
        NodeShape2d::rectangle(0.5, 0.4).unwrap(),
        NodeShape2d::rectangle(0.4, 0.3).unwrap(),
        NodeShape2d::rectangle(0.6, 0.5).unwrap(),
    ];
    let ext_shapes = NodeExtents2d::from_shapes(&shapes_d).unwrap();
    assert_eq!(nor_metrics(&g_d, &orig_d, &adj_d, ext_shapes).unwrap(), nor);
    assert_close(
        sprawlter_nn(&orig_d, ext_shapes).unwrap(),
        sprawlter_nn(&orig_d, ext_d).unwrap(),
        1e-15,
        "D.sprawlter.shapes",
    );
    let ext_uni = NodeExtents2d::uniform(4, 0.3_f64, 0.3).unwrap();
    let ext_rad = NodeExtents2d::from_radii(&[0.3_f64; 4]).unwrap();
    assert_eq!(
        nor_metrics_without_graph(&orig_d, &adj_d, ext_uni).unwrap(),
        nor_metrics_without_graph(&orig_d, &adj_d, ext_rad).unwrap()
    );
    assert_close(
        sprawlter_nn(&orig_d, ext_uni).unwrap(),
        sprawlter_nn_radii(&orig_d, &[0.3_f64; 4]).unwrap(),
        1e-15,
        "D.sprawlter.uniform_vs_radii",
    );
}

#[test]
fn test_klee_union_area_and_large_n_diameter_oracles() {
    let mut rng = DeterministicRng::from_seed_u64(31415);
    let mut rects = Vec::new();
    for _ in 0..35 {
        let x0 = (rng.gen_range_usize(12) as f64) * 0.5 - 2.0;
        let y0 = (rng.gen_range_usize(12) as f64) * 0.5 - 2.0;
        let w = (rng.gen_range_usize(6) + 1) as f64 * 0.5;
        let h = (rng.gen_range_usize(6) + 1) as f64 * 0.5;
        rects.push([x0, x0 + w, y0, y0 + h]);
    }

    let got_union = rect_union_area(&rects);
    let mut xs: Vec<f64> = rects.iter().flat_map(|r| [r[0], r[1]]).collect();
    let mut ys: Vec<f64> = rects.iter().flat_map(|r| [r[2], r[3]]).collect();
    xs.sort_unstable_by(f64::total_cmp);
    xs.dedup();
    ys.sort_unstable_by(f64::total_cmp);
    ys.dedup();
    let mut brute_union = 0.0_f64;
    for xi in 0..(xs.len() - 1) {
        let mx = 0.5 * (xs[xi] + xs[xi + 1]);
        let dx = xs[xi + 1] - xs[xi];
        for yi in 0..(ys.len() - 1) {
            let my = 0.5 * (ys[yi] + ys[yi + 1]);
            let dy = ys[yi + 1] - ys[yi];
            if rects
                .iter()
                .any(|r| mx > r[0] && mx < r[1] && my > r[2] && my < r[3])
            {
                brute_union += dx * dy;
            }
        }
    }
    assert_close(got_union, brute_union, 1e-12, "Klee union area");

    // N > 64 rotating-calipers branch of euclidean_diameter_2d + tie-heavy cq_hac.
    let mut lxs = Vec::with_capacity(80);
    let mut lys = Vec::with_capacity(80);
    let mut llabels = Vec::with_capacity(80);
    for i in 0..80usize {
        let c = (i / 20) as u32;
        let ox = (c % 2) as f64 * 20.0;
        let oy = (c / 2) as f64 * 20.0;
        lxs.push(ox + (rng.next_f64() - 0.5) * 0.5);
        lys.push(oy + (rng.next_f64() - 0.5) * 0.5);
        llabels.push(c);
    }
    let big_layout = Layout2d::from_xy(lxs, lys).unwrap();
    let got_diam = metric_diameter(&big_layout);
    let mut max_sq = 0.0_f64;
    for i in 0..80 {
        for j in (i + 1)..80 {
            let dx = big_layout.x()[i] - big_layout.x()[j];
            let dy = big_layout.y()[i] - big_layout.y()[j];
            max_sq = max_sq.max(dx * dx + dy * dy);
        }
    }
    assert_close(got_diam, max_sq.sqrt(), 1e-10, "large N rotating calipers");
    assert_close(
        cq_hac(&big_layout, &llabels).unwrap().ari,
        1.0,
        1e-10,
        "large N separated cq_hac",
    );
    assert_eq!(
        pairwise_distance_overlap_error(&big_layout, &llabels, 50).unwrap(),
        0.0
    );
}

#[test]
fn test_pdoe_subnormal_scale_no_overflow() {
    // Subnormal coordinates (~1e-307) with coincident intra pairs so d = 0.0 is also exercised.
    let sep_subnormal = layout_from_pairs(&[
        (0.0, 0.0),
        (0.0, 0.0),
        (1e-308, 0.0),
        (1e-307, 0.0),
        (1e-307, 0.0),
        (1.01e-307, 0.0),
    ]);
    let labels = [0_u32, 0, 0, 1, 1, 1];
    let pdoe_sep = pairwise_distance_overlap_error(&sep_subnormal, &labels, 1000).unwrap();
    assert!(pdoe_sep.is_finite());
    assert_close(pdoe_sep, 0.0, 1e-12, "subnormal separated pdoe");

    // Intra pairs are both at d = 1e-307 (hist_intra[last] = 1.0), while inter pairs split
    // evenly between d = 0.0 and d = 1e-307 (hist_inter[0] = 0.5, hist_inter[last] = 0.5).
    let mix_subnormal = layout_from_pairs(&[(0.0, 0.0), (1e-307, 0.0), (0.0, 0.0), (1e-307, 0.0)]);
    let pdoe_mix = pairwise_distance_overlap_error(&mix_subnormal, &[0, 0, 1, 1], 1000).unwrap();
    assert!(pdoe_mix.is_finite());
    assert_close(pdoe_mix, 0.5, 1e-12, "subnormal overlapping pdoe");

    // N > 64 rotating-calipers branch at subnormal scale.
    let mut xs = Vec::with_capacity(70);
    let mut ys = Vec::with_capacity(70);
    let mut big_labels = Vec::with_capacity(70);
    for i in 0..70usize {
        let c = (i / 35) as u32;
        xs.push(if c == 0 {
            (i as f64) * 1e-309
        } else {
            1e-307 + ((i - 35) as f64) * 1e-309
        });
        ys.push(((i % 5) as f64) * 1e-309);
        big_labels.push(c);
    }
    let big_subnormal = Layout2d::from_xy(xs, ys).unwrap();
    let pdoe_big = pairwise_distance_overlap_error(&big_subnormal, &big_labels, 100).unwrap();
    assert!(pdoe_big.is_finite());
    assert_close(pdoe_big, 0.0, 1e-12, "subnormal N>64 separated pdoe");
}

#[test]
fn test_nor_metrics_zero_extent() {
    let orig = layout_from_pairs(&[(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (2.0, 2.0)]);
    let adj = layout_from_pairs(&[(0.0, 0.0), (4.0, 0.0), (0.0, 4.0), (4.0, 4.0)]);
    let zero_ext = NodeExtents2d::uniform(4, 0.0_f64, 0.0).unwrap();
    let g = CsrGraph::<f64>::from_edges(4, &[(0, 1), (0, 2), (1, 3)], None, false, false).unwrap();

    let m = nor_metrics(&g, &orig, &adj, zero_ext).unwrap();
    assert_close(m.oo_nni, 0.0, 1e-12, "zero_ext.oo_nni");
    assert_close(m.sp_ch_a, 4.0, 1e-12, "zero_ext.sp_ch_a");
    assert_close(m.gs_bb_iar, 1.0, 1e-12, "zero_ext.gs_bb_iar");
    assert_close(m.nm_dm_imse, 0.0, 1e-12, "zero_ext.nm_dm_imse");
    assert_close(m.el_rsd, 0.0, 1e-12, "zero_ext.el_rsd");

    let m_ng = nor_metrics_without_graph(&orig, &adj, zero_ext).unwrap();
    assert_close(m_ng.sp_ch_a, 4.0, 1e-12, "zero_ext_ng.sp_ch_a");
    assert_eq!(m_ng.el_rsd, 0.0);

    // Collinear zero-extent layout falls back cleanly to 1.0 for sp_ch_a and gs_bb_iar.
    let col = layout_from_pairs(&[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]);
    let m_col = nor_metrics_without_graph(&col, &col, zero_ext).unwrap();
    assert_close(m_col.oo_nni, 0.0, 1e-12, "zero_ext_col.oo_nni");
    assert_close(m_col.sp_ch_a, 1.0, 1e-12, "zero_ext_col.sp_ch_a");
    assert_close(m_col.gs_bb_iar, 1.0, 1e-12, "zero_ext_col.gs_bb_iar");
    assert_close(m_col.nm_dm_imse, 0.0, 1e-12, "zero_ext_col.nm_dm_imse");
}
