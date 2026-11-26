use super::super::nd_proximity::build_dynamic_kdtree_f64;
use super::super::separation::dir_cpt_from_degree_counts;
use super::*;
use egraph_core::{DeterministicRng, LayoutNd};

fn layout_nd_from_2d(l2: &Layout2d<f64>) -> LayoutNd<f64> {
    let n = l2.len();
    let mut flat = Vec::with_capacity(n * 2);
    for i in 0..n {
        flat.push(l2.x()[i]);
        flat.push(l2.y()[i]);
    }
    LayoutNd::from_flat(n, 2, flat).unwrap()
}

fn brute_sepme_nd(
    layout: &LayoutNd<f64>,
    labels: &[u32],
    mode: u8, // 0 = RNG, 1 = GAB, 2 = GONG
    gamma: f64,
) -> f64 {
    let n = layout.num_nodes();
    let d = layout.dims();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return 1.0;
    }
    let coords = layout.coords();
    let (comp, sizes, k_classes) = compact_labels(labels);
    let mut deg = vec![0usize; n];
    let mut same = vec![0usize; n];
    let dist_sq = |a: &[f64], b: &[f64]| -> f64 { (0..d).map(|ax| (a[ax] - b[ax]).powi(2)).sum() };

    for i in 0..n {
        let pi = &coords[i * d..(i + 1) * d];
        for j in 0..n {
            if i == j {
                continue;
            }
            if mode != 2 && j < i {
                continue;
            }
            let pj = &coords[j * d..(j + 1) * d];
            let d_ij_sq = dist_sq(pi, pj);
            if d_ij_sq <= 0.0 {
                continue;
            }
            let mut ok = true;
            if mode == 2 {
                let mut m = vec![0.0_f64; d];
                for ax in 0..d {
                    m[ax] = gamma * pj[ax] + (1.0 - gamma) * pi[ax];
                }
                let block_thresh = (1.0 - gamma).powi(2) * d_ij_sq * (1.0 + 1e-12);
                for q in 0..n {
                    if q != i && q != j && dist_sq(&coords[q * d..(q + 1) * d], &m) <= block_thresh
                    {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    deg[i] += 1;
                    if comp[i] == comp[j] {
                        same[i] += 1;
                    }
                }
            } else {
                for r in 0..n {
                    if r == i || r == j {
                        continue;
                    }
                    let pr = &coords[r * d..(r + 1) * d];
                    let d_ir_sq = dist_sq(pi, pr);
                    let d_jr_sq = dist_sq(pj, pr);
                    let blocked = if mode == 1 {
                        d_ir_sq + d_jr_sq <= d_ij_sq * (1.0 + 1e-12)
                    } else {
                        let t = d_ij_sq * (1.0 - 1e-12);
                        d_ir_sq < t && d_jr_sq < t
                    };
                    if blocked {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    deg[i] += 1;
                    deg[j] += 1;
                    if comp[i] == comp[j] {
                        same[i] += 1;
                        same[j] += 1;
                    }
                }
            }
        }
    }
    dir_cpt_from_degree_counts(&comp, &sizes, k_classes, &deg, &same)
}

#[test]
fn test_nd_dim2_exact_parity_with_layout2d_all_13_metrics() {
    let x_b = vec![
        -1.1897, -2.8445, 0.9574, -0.4679, -0.4015, 1.0801, -0.0962, 0.2414, -0.6056, -0.1957,
        -0.7159, -0.3484, 0.9628, -1.0698, -1.845, -1.7543, 1.2509, -1.3283, -0.7785, 0.7598,
    ];
    let y_b = vec![
        0.3609, 2.0937, -0.4381, 0.4558, -0.3389, 0.7721, -0.1282, -0.921, 0.8224, -2.0616, 0.9849,
        -0.2231, 2.7369, 2.0223, 0.2625, 2.0272, 1.7066, 1.0268, -0.6861, 1.3151,
    ];
    let l2 = Layout2d::from_xy(x_b, y_b).unwrap();
    let lnd = layout_nd_from_2d(&l2);
    let labels = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3];

    assert_close(
        distance_consistency_nd(&lnd, &labels).unwrap(),
        distance_consistency(&l2, &labels).unwrap(),
        1e-12,
        "dim2.dsc",
    );
    assert_close(
        distribution_consistency_nd(&lnd, &labels, 16).unwrap(),
        distribution_consistency(&l2, &labels, 16).unwrap(),
        1e-12,
        "dim2.dc",
    );
    assert_close(
        silhouette_score_nd(&lnd, &labels).unwrap(),
        silhouette_score(&l2, &labels).unwrap(),
        1e-12,
        "dim2.sil",
    );
    assert_close(
        calinski_harabasz_score_nd(&lnd, &labels).unwrap(),
        calinski_harabasz_score(&l2, &labels).unwrap(),
        1e-12,
        "dim2.ch",
    );
    assert_close(
        davies_bouldin_score_nd(&lnd, &labels).unwrap(),
        davies_bouldin_score(&l2, &labels).unwrap(),
        1e-12,
        "dim2.db",
    );
    assert_close(
        pairwise_distance_overlap_error_nd(&lnd, &labels, 10).unwrap(),
        pairwise_distance_overlap_error(&l2, &labels, 10).unwrap(),
        1e-12,
        "dim2.pdoe",
    );

    let km2 = cq_kmeans(&l2, &labels, 5, 100, 42).unwrap();
    let km_nd = cq_kmeans_nd(&lnd, &labels, 5, 100, 42).unwrap();
    assert_close(km_nd.ari, km2.ari, 1e-12, "dim2.cq_kmeans.ari");
    assert_close(km_nd.ami, km2.ami, 1e-12, "dim2.cq_kmeans.ami");

    let hac2 = cq_hac(&l2, &labels).unwrap();
    let hac_nd = cq_hac_nd(&lnd, &labels).unwrap();
    assert_close(hac_nd.ari, hac2.ari, 1e-12, "dim2.cq_hac.ari");
    assert_close(hac_nd.ami, hac2.ami, 1e-12, "dim2.cq_hac.ami");

    assert_close(
        sepme_knng_nd(&lnd, &labels, 2).unwrap(),
        sepme_knng(&l2, &labels, 2).unwrap(),
        1e-12,
        "dim2.knng",
    );
    assert_close(
        sepme_rng_nd(&lnd, &labels).unwrap(),
        sepme_rng(&l2, &labels).unwrap(),
        1e-12,
        "dim2.rng",
    );
    assert_close(
        sepme_gab_nd(&lnd, &labels).unwrap(),
        sepme_gab(&l2, &labels).unwrap(),
        1e-12,
        "dim2.gab",
    );
    for gamma in [0.0, 0.35, 0.75] {
        assert_close(
            sepme_gong_nd(&lnd, &labels, gamma).unwrap(),
            sepme_gong(&l2, &labels, gamma).unwrap(),
            1e-12,
            &format!("dim2.gong({gamma})"),
        );
    }
    assert_close(
        neighborhood_hit_nd(&lnd, &labels, 7).unwrap(),
        neighborhood_hit(&l2, &labels, 7).unwrap(),
        1e-12,
        "dim2.m_nh",
    );

    // Subnormal PDOE parity (no underflow to zero in Euclidean distance).
    let sub2 = layout_from_pairs(&[(0.0, 0.0), (1e-307, 0.0), (0.0, 0.0), (1e-307, 0.0)]);
    let sub_nd = layout_nd_from_2d(&sub2);
    assert_close(
        pairwise_distance_overlap_error_nd(&sub_nd, &[0, 0, 1, 1], 1000).unwrap(),
        pairwise_distance_overlap_error(&sub2, &[0, 0, 1, 1], 1000).unwrap(),
        1e-12,
        "dim2.subnormal.pdoe",
    );
}

#[test]
fn test_nd_across_dimensions_1_2_3_4_10_f64_and_f32() {
    for dims in [1usize, 2, 3, 4, 10] {
        // 3 well-separated clusters of 4 points each (N = 12, K = 3) in `dims` dimensions.
        let mut rng = DeterministicRng::from_seed_u64(100 + dims as u64);
        let n = 12usize;
        let mut flat64 = Vec::with_capacity(n * dims);
        let mut labels = Vec::with_capacity(n);
        for c in 0..3u32 {
            let offset = (c as f64) * 20.0;
            for _ in 0..4 {
                for ax in 0..dims {
                    let jitter = (rng.next_f64() - 0.5) * 0.2;
                    let base = if ax == 0 { offset } else { 0.0 };
                    flat64.push(base + jitter);
                }
                labels.push(c * 5 + 3); // non-contiguous label IDs
            }
        }
        let l64 = LayoutNd::from_flat(n, dims, flat64.clone()).unwrap();
        let l32 = LayoutNd::from_flat(
            n,
            dims,
            flat64.iter().map(|&v| v as f32).collect::<Vec<f32>>(),
        )
        .unwrap();

        assert_close(
            distance_consistency_nd(&l64, &labels).unwrap(),
            1.0,
            1e-12,
            &format!("d{dims}.dsc"),
        );
        assert_close(
            distribution_consistency_nd(&l64, &labels, 100).unwrap(),
            1.0,
            1e-12,
            &format!("d{dims}.dc_sparse_100"),
        );
        assert!(silhouette_score_nd(&l64, &labels).unwrap() > 0.95);
        assert!(calinski_harabasz_score_nd(&l64, &labels).unwrap() > 100.0);
        assert!(davies_bouldin_score_nd(&l64, &labels).unwrap() < 0.1);
        assert_eq!(
            pairwise_distance_overlap_error_nd(&l64, &labels, 20).unwrap(),
            0.0
        );
        assert_close(
            cq_kmeans_nd(&l64, &labels, 5, 50, 42).unwrap().ari,
            1.0,
            1e-12,
            &format!("d{dims}.km"),
        );
        assert_close(
            cq_hac_nd(&l64, &labels).unwrap().ari,
            1.0,
            1e-12,
            &format!("d{dims}.hac"),
        );
        assert_close(
            sepme_knng_nd(&l64, &labels, 3).unwrap(),
            1.0,
            1e-12,
            &format!("d{dims}.knng"),
        );
        assert_close(
            neighborhood_hit_nd(&l64, &labels, 3).unwrap(),
            1.0,
            1e-12,
            &format!("d{dims}.nh"),
        );
        assert_close(
            sepme_rng_nd(&l64, &labels).unwrap(),
            brute_sepme_nd(&l64, &labels, 0, 0.0),
            1e-12,
            &format!("d{dims}.rng"),
        );
        assert_close(
            sepme_gab_nd(&l64, &labels).unwrap(),
            brute_sepme_nd(&l64, &labels, 1, 0.0),
            1e-12,
            &format!("d{dims}.gab"),
        );
        assert_close(
            sepme_gong_nd(&l64, &labels, 0.35).unwrap(),
            brute_sepme_nd(&l64, &labels, 2, 0.35),
            1e-12,
            &format!("d{dims}.gong"),
        );

        // Verify f32 matches f64 closely on well-separated clusters.
        assert_close(
            distance_consistency_nd(&l32, &labels).unwrap(),
            1.0,
            1e-12,
            &format!("d{dims}.f32.dsc"),
        );
        assert!(silhouette_score_nd(&l32, &labels).unwrap() > 0.95);
        assert_close(
            cq_hac_nd(&l32, &labels).unwrap().ari,
            1.0,
            1e-12,
            &format!("d{dims}.f32.hac"),
        );
        assert_close(
            sepme_knng_nd(&l32, &labels, 3).unwrap(),
            1.0,
            1e-12,
            &format!("d{dims}.f32.knng"),
        );
    }
}

#[test]
fn test_nd_brute_force_separation_and_coincident_ties() {
    for dims in [1usize, 3, 4, 10] {
        let mut rng = DeterministicRng::from_seed_u64(777 + dims as u64);
        let n = 36usize;
        let mut coords = Vec::with_capacity(n * dims);
        for i in 0..n {
            if i == 5 {
                // Coincident duplicate of vertex 0 to exercise tie & duplicate handling.
                for ax in 0..dims {
                    coords.push(coords[ax]);
                }
            } else {
                for _ in 0..dims {
                    coords.push((rng.next_f64() - 0.5) * 10.0);
                }
            }
        }
        let layout = LayoutNd::from_flat(n, dims, coords).unwrap();
        let labels: Vec<u32> = (0..n).map(|i| (i % 4) as u32).collect();
        let tree = build_dynamic_kdtree_f64(&layout).unwrap();

        assert_eq!(
            sepme_knng_nd(&layout, &labels, 4).unwrap().to_bits(),
            sepme_knng_nd_with_tree(&layout, &labels, 4, &tree)
                .unwrap()
                .to_bits()
        );
        assert_eq!(
            neighborhood_hit_nd(&layout, &labels, 7).unwrap().to_bits(),
            neighborhood_hit_nd_with_tree(&layout, &labels, 7, &tree)
                .unwrap()
                .to_bits()
        );
        assert_close(
            sepme_rng_nd(&layout, &labels).unwrap(),
            brute_sepme_nd(&layout, &labels, 0, 0.0),
            1e-12,
            &format!("rand.d{dims}.rng"),
        );
        assert_close(
            sepme_gab_nd(&layout, &labels).unwrap(),
            brute_sepme_nd(&layout, &labels, 1, 0.0),
            1e-12,
            &format!("rand.d{dims}.gab"),
        );
        for gamma in [0.0, 0.35, 0.5, 0.75] {
            assert_close(
                sepme_gong_nd(&layout, &labels, gamma).unwrap(),
                brute_sepme_nd(&layout, &labels, 2, gamma),
                1e-12,
                &format!("rand.d{dims}.gong({gamma})"),
            );
        }
    }
}

fn assert_nd_degenerate<S: Scalar>(layout: &LayoutNd<S>, labels: &[u32], expected_sepme: f64) {
    assert_eq!(distance_consistency_nd(layout, labels).unwrap(), 0.0);
    assert_eq!(
        distribution_consistency_nd(layout, labels, 10).unwrap(),
        0.0
    );
    assert_eq!(silhouette_score_nd(layout, labels).unwrap(), 0.0);
    assert_eq!(calinski_harabasz_score_nd(layout, labels).unwrap(), 0.0);
    assert_eq!(davies_bouldin_score_nd(layout, labels).unwrap(), 0.0);
    assert_eq!(
        pairwise_distance_overlap_error_nd(layout, labels, 10).unwrap(),
        0.0
    );
    assert_eq!(
        cq_kmeans_nd(layout, labels, 3, 20, 42).unwrap(),
        PartitionAgreement::default()
    );
    assert_eq!(
        cq_hac_nd(layout, labels).unwrap(),
        PartitionAgreement::default()
    );
    assert_eq!(sepme_knng_nd(layout, labels, 3).unwrap(), expected_sepme);
    assert_eq!(sepme_rng_nd(layout, labels).unwrap(), expected_sepme);
    assert_eq!(sepme_gab_nd(layout, labels).unwrap(), expected_sepme);
    assert_eq!(sepme_gong_nd(layout, labels, 0.35).unwrap(), expected_sepme);
    assert_eq!(neighborhood_hit_nd(layout, labels, 7).unwrap(), 0.0);
}

#[test]
fn test_nd_degenerate_and_validation_errors() {
    let empty64 = LayoutNd::<f64>::zeros(0, 3).unwrap();
    assert_nd_degenerate(&empty64, &[], 0.0);

    let single32 = LayoutNd::from_flat(1, 4, vec![1.0_f32, 2.0, -3.0, 0.5]).unwrap();
    assert_nd_degenerate(&single32, &[0], 1.0);

    // Collapsed 3D layout returns 1.0 for PDOE and 0.0 for CQ.
    let collapsed = LayoutNd::<f64>::zeros(4, 3).unwrap();
    let col_l = [0_u32, 0, 1, 1];
    assert_eq!(
        pairwise_distance_overlap_error_nd(&collapsed, &col_l, 10).unwrap(),
        1.0
    );
    assert_eq!(
        cq_kmeans_nd(&collapsed, &col_l, 2, 10, 42).unwrap(),
        PartitionAgreement::default()
    );
    assert_eq!(
        cq_hac_nd(&collapsed, &col_l).unwrap(),
        PartitionAgreement::default()
    );

    // Validation errors: NaN / Inf / label mismatch / invalid scalar args / tree mismatch.
    let l3 = LayoutNd::from_flat(2, 3, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let bad_nan = LayoutNd::from_flat(2, 3, vec![f64::NAN, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let bad_inf = LayoutNd::from_flat(2, 3, vec![f64::INFINITY, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    for bad in [&bad_nan, &bad_inf] {
        let f64_checks: [(&str, CoreResult<f64>); 11] = [
            ("dsc", distance_consistency_nd(bad, &[0, 1])),
            ("dc", distribution_consistency_nd(bad, &[0, 1], 4)),
            ("sil", silhouette_score_nd(bad, &[0, 1])),
            ("ch", calinski_harabasz_score_nd(bad, &[0, 1])),
            ("db", davies_bouldin_score_nd(bad, &[0, 1])),
            ("pdoe", pairwise_distance_overlap_error_nd(bad, &[0, 1], 4)),
            ("knng", sepme_knng_nd(bad, &[0, 1], 1)),
            ("rng", sepme_rng_nd(bad, &[0, 1])),
            ("gab", sepme_gab_nd(bad, &[0, 1])),
            ("gong", sepme_gong_nd(bad, &[0, 1], 0.35)),
            ("nh", neighborhood_hit_nd(bad, &[0, 1], 1)),
        ];
        for (name, res) in f64_checks {
            assert!(
                matches!(res, Err(CoreError::NonFiniteValue { .. })),
                "{name} did not reject non-finite layout"
            );
        }
        assert!(matches!(
            cq_kmeans_nd(bad, &[0, 1], 2, 10, 42),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            cq_hac_nd(bad, &[0, 1]),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }

    let l_dim4 = LayoutNd::from_flat(2, 4, vec![0.0; 8]).unwrap();
    let tree_dim4 = build_dynamic_kdtree_f64(&l_dim4).unwrap();
    for res in [
        distance_consistency_nd(&l3, &[0]),
        neighborhood_hit_nd(&l3, &[0], 1),
        sepme_knng_nd_with_tree(&l3, &[0, 1], 1, &tree_dim4),
    ] {
        assert!(matches!(res, Err(CoreError::DimensionMismatch { .. })));
    }
    assert!(matches!(
        cq_hac_nd(&l3, &[0]),
        Err(CoreError::DimensionMismatch { .. })
    ));
    for res in [
        distribution_consistency_nd(&l3, &[0, 1], 0),
        pairwise_distance_overlap_error_nd(&l3, &[0, 1], 0),
        sepme_knng_nd(&l3, &[0, 1], 0),
        sepme_gong_nd(&l3, &[0, 1], 1.2),
        neighborhood_hit_nd(&l3, &[0, 1], 0),
    ] {
        assert!(matches!(res, Err(CoreError::InvalidInput(_))));
    }
    assert!(matches!(
        cq_kmeans_nd(&l3, &[0, 1], 0, 10, 42),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        cq_kmeans_nd(&l3, &[0, 1], 1, 0, 42),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        sepme_gong_nd(&l3, &[0, 1], f64::NAN),
        Err(CoreError::NonFiniteValue { .. })
    ));
}
