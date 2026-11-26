use super::super::separation::dir_cpt_from_degree_counts;
use super::*;
use egraph_core::{CsrGraph, DeterministicRng};

#[test]
fn test_dc_cdm_hdm_dsc_separation_and_properties() {
    // Well-separated clusters vs fully mixed clusters.
    let well_sep = layout_from_pairs(&[
        (0.0, 0.0),
        (0.1, 0.0),
        (0.0, 0.1),
        (0.1, 0.1),
        (10.0, 10.0),
        (10.1, 10.0),
        (10.0, 10.1),
        (10.1, 10.1),
    ]);
    let labels = [10_u32, 10, 10, 10, 42, 42, 42, 42]; // sparse non-0..K labels

    assert_close(
        distance_consistency(&well_sep, &labels).unwrap(),
        1.0,
        1e-12,
        "DSC.sep",
    );
    assert_close(
        distribution_consistency(&well_sep, &labels, 20).unwrap(),
        1.0,
        1e-12,
        "DC.sep",
    );
    let (cdm_sep, hdm_sep) = class_density_measure(&well_sep, &labels, 64, None).unwrap();
    assert!(cdm_sep > 0.99, "cdm_sep = {cdm_sep}");
    assert_close(hdm_sep, 1.0, 1e-12, "HDM.sep");

    // Equidistant / identical centroids -> DSC strict `<` gives 0.0 regardless of label order.
    let tied = layout_from_pairs(&[(0.0, 0.0), (2.0, 0.0), (0.0, 0.0), (2.0, 0.0)]);
    let tied_labels = [0, 0, 1, 1];
    assert_eq!(distance_consistency(&tied, &tied_labels).unwrap(), 0.0);
    assert_eq!(
        distribution_consistency(&tied, &tied_labels, 20).unwrap(),
        0.0
    );
    let (cdm_tied, hdm_tied) = class_density_measure(&tied, &tied_labels, 32, None).unwrap();
    assert!(cdm_tied < 1e-12);
    assert!(hdm_tied < 1e-12);

    // Collapsed single-axis layout must not divide by zero or panic.
    let col = layout_from_pairs(&[(0.0, 5.0), (0.0, 5.0), (1.0, 5.0), (1.0, 5.0)]);
    let dc_col = distribution_consistency(&col, &tied_labels, 16).unwrap();
    assert!((0.0..=1.0).contains(&dc_col));
    let (cdm_col, hdm_col) = class_density_measure(&col, &tied_labels, 16, Some(0.001)).unwrap();
    assert!((0.0..=1.0).contains(&cdm_col));
    assert!((0.0..=1.0).contains(&hdm_col));
}

#[test]
fn test_gong_and_oo_nni_brute_force_oracles() {
    for seed in [7_u64, 19] {
        let mut rng = DeterministicRng::from_seed_u64(seed);
        let n = 45usize;
        let xs: Vec<f64> = (0..n).map(|_| rng.next_f64() * 10.0 - 5.0).collect();
        let ys: Vec<f64> = (0..n).map(|_| rng.next_f64() * 10.0 - 5.0).collect();
        let orig = Layout2d::from_xy(xs, ys).unwrap();
        let adj_x: Vec<f64> = orig
            .x()
            .iter()
            .map(|&v| v + (rng.next_f64() - 0.5) * 3.0)
            .collect();
        let adj_y: Vec<f64> = orig
            .y()
            .iter()
            .map(|&v| v + (rng.next_f64() - 0.5) * 3.0)
            .collect();
        let adj = Layout2d::from_xy(adj_x, adj_y).unwrap();
        let hw = vec![0.25_f64; n];
        let hh = vec![0.25_f64; n];

        let nor = nor_metrics_without_graph(&orig, &adj, &hw, &hh).unwrap();
        assert_close(
            nor.oo_nni,
            brute_oo_nni(&orig, &adj),
            1e-12,
            "oo_nni brute parity",
        );

        let labels: Vec<u32> = (0..n).map(|i| (i % 3) as u32).collect();
        let (comp, sizes, k_classes) = compact_labels(&labels);
        for gamma in [0.0, 0.35, 0.5, 0.75] {
            let b_edges = brute_gong_edges(&orig, gamma);
            let mut deg = vec![0usize; n];
            let mut same = vec![0usize; n];
            for &(u, v) in &b_edges {
                deg[u] += 1;
                if comp[u] == comp[v] {
                    same[u] += 1;
                }
            }
            let expected = dir_cpt_from_degree_counts(&comp, &sizes, k_classes, &deg, &same);
            let got = sepme_gong(&orig, &labels, gamma).unwrap();
            assert_close(got, expected, 1e-12, &format!("gong({gamma})"));
        }
    }

    // Ties in coordinates (including -0.0 vs +0.0) for oo_nni.
    let tie_orig = layout_from_pairs(&[(-0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, -0.0)]);
    let tie_adj = layout_from_pairs(&[(1.0, 0.0), (0.0, 1.0), (0.5, 1.0), (0.2, 0.0)]);
    let hw = [0.1; 4];
    let nor_tie = nor_metrics_without_graph(&tie_orig, &tie_adj, &hw, &hw).unwrap();
    assert_close(
        nor_tie.oo_nni,
        brute_oo_nni(&tie_orig, &tie_adj),
        1e-12,
        "oo_nni tie parity",
    );
}

#[test]
fn test_degenerate_f32_and_error_cases() {
    let empty = Layout2d::<f64>::zeros(0);
    let empty_l: [u32; 0] = [];
    let empty_s: [f64; 0] = [];
    let empty_g = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();

    assert_eq!(
        cq_kmeans(&empty, &empty_l, 5, 50, 42).unwrap(),
        PartitionAgreement::default()
    );
    assert_eq!(
        cq_hac(&empty, &empty_l).unwrap(),
        PartitionAgreement::default()
    );
    assert_eq!(distance_consistency(&empty, &empty_l).unwrap(), 0.0);
    assert_eq!(distribution_consistency(&empty, &empty_l, 10).unwrap(), 0.0);
    assert_eq!(
        class_density_measure(&empty, &empty_l, 10, None).unwrap(),
        (0.0, 0.0)
    );
    assert_eq!(sepme_knng(&empty, &empty_l, 2).unwrap(), 0.0);
    assert_eq!(sepme_rng(&empty, &empty_l).unwrap(), 0.0);
    assert_eq!(sepme_gab(&empty, &empty_l).unwrap(), 0.0);
    assert_eq!(sepme_gong(&empty, &empty_l, 0.35).unwrap(), 0.0);
    assert_eq!(neighborhood_hit(&empty, &empty_l, 7).unwrap(), 0.0);
    assert_eq!(silhouette_score(&empty, &empty_l).unwrap(), 0.0);
    assert_eq!(calinski_harabasz_score(&empty, &empty_l).unwrap(), 0.0);
    assert_eq!(davies_bouldin_score(&empty, &empty_l).unwrap(), 0.0);
    assert_eq!(
        pairwise_distance_overlap_error(&empty, &empty_l, 10).unwrap(),
        0.0
    );
    assert_eq!(sprawlter_nn(&empty, &empty_s, &empty_s).unwrap(), 0.0);
    assert_eq!(sprawlter_nn_radii(&empty, &empty_s).unwrap(), 0.0);
    assert_eq!(
        nor_metrics(&empty_g, &empty, &empty, &empty_s, &empty_s).unwrap(),
        NorMetrics::default()
    );

    let single = Layout2d::from_xy(vec![1.0_f32], vec![2.0_f32]).unwrap();
    let single_l = [0_u32];
    let single_r = [0.5_f32];
    assert_eq!(
        cq_kmeans(&single, &single_l, 3, 10, 42).unwrap(),
        PartitionAgreement::default()
    );
    assert_eq!(
        cq_hac(&single, &single_l).unwrap(),
        PartitionAgreement::default()
    );
    assert_eq!(distance_consistency(&single, &single_l).unwrap(), 0.0);
    assert_eq!(
        distribution_consistency(&single, &single_l, 10).unwrap(),
        0.0
    );
    assert_eq!(
        class_density_measure(&single, &single_l, 10, None).unwrap(),
        (0.0, 0.0)
    );
    assert_eq!(sepme_knng(&single, &single_l, 2).unwrap(), 1.0);
    assert_eq!(sepme_rng(&single, &single_l).unwrap(), 1.0);
    assert_eq!(sepme_gab(&single, &single_l).unwrap(), 1.0);
    assert_eq!(sepme_gong(&single, &single_l, 0.35).unwrap(), 1.0);
    assert_eq!(neighborhood_hit(&single, &single_l, 7).unwrap(), 0.0);
    assert_eq!(silhouette_score(&single, &single_l).unwrap(), 0.0);
    assert_eq!(calinski_harabasz_score(&single, &single_l).unwrap(), 0.0);
    assert_eq!(davies_bouldin_score(&single, &single_l).unwrap(), 0.0);
    assert_eq!(
        pairwise_distance_overlap_error(&single, &single_l, 10).unwrap(),
        0.0
    );
    assert_eq!(sprawlter_nn_radii(&single, &single_r).unwrap(), 1.0);
    let nor1 = nor_metrics_without_graph(&single, &single, &single_r, &single_r).unwrap();
    assert_eq!(nor1.sp_ch_a, 1.0);
    assert_eq!(nor1.gs_bb_iar, 1.0);

    // f32 multi-cluster smoke check across all metrics.
    let l32 = Layout2d::from_xy(
        vec![0.0_f32, 0.2, 0.1, 5.0, 5.2, 5.1],
        vec![0.0_f32, 0.1, -0.1, 5.0, 5.1, 4.9],
    )
    .unwrap();
    let lbl32 = [0_u32, 0, 0, 1, 1, 1];
    assert_close(
        cq_kmeans(&l32, &lbl32, 3, 50, 42).unwrap().ari,
        1.0,
        1e-10,
        "f32.km",
    );
    assert_close(cq_hac(&l32, &lbl32).unwrap().ari, 1.0, 1e-10, "f32.hac");
    assert_close(
        distance_consistency(&l32, &lbl32).unwrap(),
        1.0,
        1e-10,
        "f32.dsc",
    );
    assert!(silhouette_score(&l32, &lbl32).unwrap() > 0.9);
    assert_eq!(
        pairwise_distance_overlap_error(&l32, &lbl32, 20).unwrap(),
        0.0
    );
    let r32 = [0.25_f32; 6];
    assert!(sprawlter_nn_radii(&l32, &r32).unwrap().is_finite());

    // Validation errors.
    let l2 = Layout2d::from_xy(vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();
    let bad_nan = Layout2d::from_xy(vec![f64::NAN, 1.0], vec![0.0, 1.0]).unwrap();
    assert!(matches!(
        partition_agreement(&[0], &[0, 1]),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        cq_kmeans(&l2, &[0, 1], 0, 10, 42),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        cq_kmeans(&l2, &[0, 1], 1, 0, 42),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        distribution_consistency(&l2, &[0, 1], 0),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        class_density_measure(&l2, &[0, 1], 0, None),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        class_density_measure(&l2, &[0, 1], 10, Some(-0.5)),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        class_density_measure(&l2, &[0, 1], 10, Some(f64::NAN)),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        sepme_knng(&l2, &[0, 1], 0),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        sepme_gong(&l2, &[0, 1], 1.5),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        sepme_gong(&l2, &[0, 1], f64::NAN),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        neighborhood_hit(&l2, &[0, 1], 0),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        pairwise_distance_overlap_error(&l2, &[0, 1], 0),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        sprawlter_nn(&l2, &[0.0, 1.0], &[1.0, 1.0]),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        sprawlter_nn(&bad_nan, &[1.0, 1.0], &[1.0, 1.0]),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        nor_metrics_without_graph(&l2, &l2, &[-0.1, 1.0], &[1.0, 1.0]),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        distance_consistency(&l2, &[0]),
        Err(CoreError::DimensionMismatch { .. })
    ));
}
