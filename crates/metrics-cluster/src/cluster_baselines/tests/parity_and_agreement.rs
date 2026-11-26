use super::*;

#[test]
fn test_fixture_a_b_c_sklearn_scipy_exact_parity() {
    // Fixture A: N = 12, K = 3 uneven [6, 4, 2].
    let coords_a = [
        (0.0, 0.0),
        (0.4, 0.1),
        (0.1, 0.5),
        (-0.3, 0.2),
        (0.2, -0.3),
        (0.5, 0.4),
        (5.0, 0.2),
        (5.4, 0.0),
        (5.1, 0.6),
        (4.8, -0.2),
        (2.5, 4.5),
        (2.8, 4.9),
    ];
    let layout_a = layout_from_pairs(&coords_a);
    let labels_a = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2];
    let pred_a = [0, 0, 0, 0, 1, 0, 1, 1, 1, 2, 2, 2];

    let pa_a = partition_agreement(&labels_a, &pred_a).unwrap();
    assert_close(pa_a.ari, 0.541_176_470_588_235_3, 1e-10, "A.ari");
    assert_close(pa_a.ami, 0.534_654_132_249_696_5, 1e-10, "A.ami");
    assert_close(pa_a.homogeneity, 0.657_334_260_519_206_7, 1e-10, "A.hom");
    assert_close(pa_a.completeness, 0.616_979_973_786_710_7, 1e-10, "A.cmp");
    assert_close(pa_a.v_measure, 0.636_518_158_404_069_2, 1e-10, "A.vm");
    assert_close(pa_a.fmi, 0.684_762_383_262_325_9, 1e-10, "A.fmi");

    let hac_a = cq_hac(&layout_a, &labels_a).unwrap();
    assert_close(hac_a.ari, 1.0, 1e-10, "A.hac.ari");
    assert_close(hac_a.ami, 1.0, 1e-10, "A.hac.ami");
    assert_close(hac_a.homogeneity, 1.0, 1e-10, "A.hac.hom");
    assert_close(hac_a.completeness, 1.0, 1e-10, "A.hac.cmp");
    assert_close(hac_a.v_measure, 1.0, 1e-10, "A.hac.vm");
    assert_close(hac_a.fmi, 1.0, 1e-10, "A.hac.fmi");

    let km_a = cq_kmeans(&layout_a, &labels_a, 5, 100, 42).unwrap();
    assert_close(km_a.ari, 1.0, 1e-10, "A.kmeans.ari");

    assert_close(
        silhouette_score(&layout_a, &labels_a).unwrap(),
        0.888_186_961_713_074_7,
        1e-10,
        "A.silhouette",
    );
    assert_close(
        calinski_harabasz_score(&layout_a, &labels_a).unwrap(),
        280.962_311_557_788_95,
        1e-10,
        "A.ch",
    );
    assert_close(
        davies_bouldin_score(&layout_a, &labels_a).unwrap(),
        0.133_252_117_690_975_3,
        1e-10,
        "A.db",
    );

    // Fixture B: N = 20, K = 4 overlapping [7, 6, 4, 3].
    let x_b = vec![
        -1.1897, -2.8445, 0.9574, -0.4679, -0.4015, 1.0801, -0.0962, 0.2414, -0.6056, -0.1957,
        -0.7159, -0.3484, 0.9628, -1.0698, -1.845, -1.7543, 1.2509, -1.3283, -0.7785, 0.7598,
    ];
    let y_b = vec![
        0.3609, 2.0937, -0.4381, 0.4558, -0.3389, 0.7721, -0.1282, -0.921, 0.8224, -2.0616, 0.9849,
        -0.2231, 2.7369, 2.0223, 0.2625, 2.0272, 1.7066, 1.0268, -0.6861, 1.3151,
    ];
    let layout_b = Layout2d::from_xy(x_b, y_b).unwrap();
    let labels_b = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3];
    let pred_b = [0, 0, 1, 0, 2, 0, 0, 1, 1, 0, 1, 3, 1, 2, 2, 1, 2, 3, 3, 2];

    let pa_b = partition_agreement(&labels_b, &pred_b).unwrap();
    assert_close(pa_b.ari, 0.290_272_373_540_856_05, 1e-10, "B.ari");
    assert_close(pa_b.ami, 0.297_072_950_635_337_3, 1e-10, "B.ami");
    assert_close(pa_b.homogeneity, 0.454_355_339_790_888_9, 1e-10, "B.hom");
    assert_close(pa_b.completeness, 0.448_165_310_990_085, 1e-10, "B.cmp");
    assert_close(pa_b.v_measure, 0.451_239_097_922_438_6, 1e-10, "B.vm");
    assert_close(pa_b.fmi, 0.454_662_892_980_315_65, 1e-10, "B.fmi");

    let hac_b = cq_hac(&layout_b, &labels_b).unwrap();
    assert_close(hac_b.ari, 0.005_607_476_635_514_018, 1e-10, "B.hac.ari");
    assert_close(hac_b.ami, -0.046_981_511_733_698_73, 1e-10, "B.hac.ami");
    assert_close(
        hac_b.homogeneity,
        0.165_036_847_012_176_2,
        1e-10,
        "B.hac.hom",
    );
    assert_close(
        hac_b.completeness,
        0.207_330_815_002_267_4,
        1e-10,
        "B.hac.cmp",
    );
    assert_close(hac_b.v_measure, 0.183_781_931_069_577_02, 1e-10, "B.hac.vm");
    assert_close(hac_b.fmi, 0.309_838_667_696_593_35, 1e-10, "B.hac.fmi");

    assert_close(
        silhouette_score(&layout_b, &labels_b).unwrap(),
        -0.155_875_228_260_980_2,
        1e-10,
        "B.silhouette",
    );
    assert_close(
        calinski_harabasz_score(&layout_b, &labels_b).unwrap(),
        0.692_974_573_303_605_2,
        1e-10,
        "B.ch",
    );
    assert_close(
        davies_bouldin_score(&layout_b, &labels_b).unwrap(),
        10.141_407_555_771_2,
        1e-10,
        "B.db",
    );

    assert_close(
        pairwise_distance_overlap_error(&layout_b, &labels_b, 10).unwrap(),
        0.880_459_770_114_942_7,
        1e-10,
        "B.pdoe",
    );
    assert_close(
        neighborhood_hit(&layout_b, &labels_b, 7).unwrap(),
        0.242_857_142_857_142_85,
        1e-10,
        "B.m_nh",
    );
    assert_close(
        sepme_knng(&layout_b, &labels_b, 2).unwrap(),
        0.178_571_428_571_428_58,
        1e-10,
        "B.sepme_knng2",
    );
    assert_close(
        sepme_rng(&layout_b, &labels_b).unwrap(),
        0.212_301_587_301_587_3,
        1e-10,
        "B.sepme_rng",
    );
    assert_close(
        sepme_gab(&layout_b, &labels_b).unwrap(),
        0.189_037_698_412_698_4,
        1e-10,
        "B.sepme_gab",
    );
    assert_close(
        sepme_gong(&layout_b, &labels_b, 0.35).unwrap(),
        0.223_214_285_714_285_7,
        1e-10,
        "B.sepme_gong",
    );
    assert_close(
        distance_consistency(&layout_b, &labels_b).unwrap(),
        0.3,
        1e-10,
        "B.dsc",
    );

    // Fixture C: Degenerate partition_agreement & CVI edge cases.
    let c_cases: &[(&[u32], &[u32], [f64; 6])] = &[
        (&[], &[], [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
        (&[0], &[0], [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
        (&[0, 0, 0, 0], &[1, 1, 1, 1], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        (
            &[0, 0, 0, 0],
            &[0, 1, 0, 1],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.5773502691896257],
        ),
        (&[0, 1, 2, 3], &[0, 0, 0, 0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        (&[0, 1, 2, 3], &[3, 2, 1, 0], [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
        (
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            [-0.5, -0.49999999999999944, 0.0, 0.0, 0.0, 0.0],
        ),
    ];
    for (idx, &(lt, lp, exp)) in c_cases.iter().enumerate() {
        let got = partition_agreement(lt, lp).unwrap();
        assert_close(got.ari, exp[0], 1e-10, &format!("C[{idx}].ari"));
        assert_close(got.ami, exp[1], 1e-10, &format!("C[{idx}].ami"));
        assert_close(got.homogeneity, exp[2], 1e-10, &format!("C[{idx}].hom"));
        assert_close(got.completeness, exp[3], 1e-10, &format!("C[{idx}].cmp"));
        assert_close(got.v_measure, exp[4], 1e-10, &format!("C[{idx}].vm"));
        assert_close(got.fmi, exp[5], 1e-10, &format!("C[{idx}].fmi"));
    }

    let l_ident = layout_from_pairs(&[(0.0, 0.0); 4]);
    let y_ident = [0, 0, 1, 1];
    assert_close(
        silhouette_score(&l_ident, &y_ident).unwrap(),
        0.0,
        1e-10,
        "C.ident.sil",
    );
    assert_close(
        calinski_harabasz_score(&l_ident, &y_ident).unwrap(),
        1.0,
        1e-10,
        "C.ident.ch",
    );
    assert_close(
        davies_bouldin_score(&l_ident, &y_ident).unwrap(),
        0.0,
        1e-10,
        "C.ident.db",
    );

    let l_icent = layout_from_pairs(&[(0.0, 0.0), (2.0, 0.0), (0.0, 0.0), (2.0, 0.0)]);
    assert_close(
        silhouette_score(&l_icent, &y_ident).unwrap(),
        -0.5,
        1e-10,
        "C.icent.sil",
    );
    assert_close(
        calinski_harabasz_score(&l_icent, &y_ident).unwrap(),
        0.0,
        1e-10,
        "C.icent.ch",
    );
    assert_close(
        davies_bouldin_score(&l_icent, &y_ident).unwrap(),
        0.0,
        1e-10,
        "C.icent.db",
    );

    let l_col = layout_from_pairs(&[(0.0, 0.0), (1.0, 0.0), (10.0, 0.0), (11.0, 0.0)]);
    assert_close(
        silhouette_score(&l_col, &y_ident).unwrap(),
        0.899749373433584,
        1e-10,
        "C.col.sil",
    );
    assert_close(
        calinski_harabasz_score(&l_col, &y_ident).unwrap(),
        200.0,
        1e-10,
        "C.col.ch",
    );
    assert_close(
        davies_bouldin_score(&l_col, &y_ident).unwrap(),
        0.1,
        1e-10,
        "C.col.db",
    );

    let l_sing = layout_from_pairs(&[(0.0, 0.0), (0.1, 0.0), (10.0, 0.0)]);
    let y_sing = [0, 0, 1];
    assert_close(
        silhouette_score(&l_sing, &y_sing).unwrap(),
        0.65996632996633,
        1e-10,
        "C.sing.sil",
    );
    assert_close(
        calinski_harabasz_score(&l_sing, &y_sing).unwrap(),
        13200.33333333333,
        1e-10,
        "C.sing.ch",
    );
    assert_close(
        davies_bouldin_score(&l_sing, &y_sing).unwrap(),
        0.0050251256281407045,
        1e-10,
        "C.sing.db",
    );
}

#[test]
fn test_partition_agreement_dominant_cluster_no_underflow() {
    // a_0 = 3, b_0 = 2, n = 4 -> a_0 + b_0 = 5 > n (must not underflow in debug builds).
    let small = partition_agreement(&[0, 0, 0, 1], &[0, 0, 1, 1]).unwrap();
    assert_close(small.ari, 0.0, 1e-12, "skewed4.ari");
    assert!(
        small.ami.abs() < 1e-12,
        "skewed4.ami = {} (expected ~0.0)",
        small.ami
    );

    // Larger skewed partition: n = 16, a_0 = 11, b_0 = 10 -> a_0 + b_0 = 21 > 16.
    let lt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2];
    let lp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2];
    let skewed = partition_agreement(&lt, &lp).unwrap();
    assert_close(skewed.ari, 0.782_850_779_510_022_2, 1e-10, "skewed16.ari");
    assert_close(skewed.ami, 0.754_922_259_280_672, 1e-10, "skewed16.ami");
    assert_close(
        skewed.homogeneity,
        0.830_907_673_853_462_1,
        1e-10,
        "skewed16.hom",
    );
    assert_close(
        skewed.completeness,
        0.767_358_057_041_120_3,
        1e-10,
        "skewed16.cmp",
    );
    assert_close(
        skewed.v_measure,
        0.797_869_447_944_516_2,
        1e-10,
        "skewed16.vm",
    );
    assert_close(skewed.fmi, 0.884_643_727_225_795_5, 1e-10, "skewed16.fmi");
}

#[test]
fn test_cq_hac_ulp_height_inversion_regression() {
    // Challenger N=5, K=3 counterexample where UPGMA (2*h + 1*h)/3 rounds 1 ULP below h.
    let xs64 = vec![0.0, 0.0, 1.0, 0.5, 5.550275137568784];
    let ys64 = vec![0.0, 0.0, 0.0, 3.0_f64.sqrt() / 2.0, 0.0];
    let labels = [0_u32, 0, 0, 1, 2];

    let layout64 = Layout2d::from_xy(xs64.clone(), ys64.clone()).unwrap();
    let res64 = cq_hac(&layout64, &labels).unwrap();
    assert!(
        (res64.ari - 0.4117647058823529).abs() > 1e-6,
        "f64 cq_hac produced 4-cluster inverted cut (ari = {})",
        res64.ari
    );
    assert!(
        (res64.ari - 1.0).abs() < 1e-10 || (res64.ari - 6.0 / 11.0).abs() < 1e-10,
        "f64 cq_hac unexpected 3-cluster ari = {}",
        res64.ari
    );

    let xs32: Vec<f32> = xs64.iter().map(|&v| v as f32).collect();
    let ys32: Vec<f32> = ys64.iter().map(|&v| v as f32).collect();
    let layout32 = Layout2d::from_xy(xs32, ys32).unwrap();
    let res32 = cq_hac(&layout32, &labels).unwrap();
    // In f32, (sqrt(3)/2) as f32 rounds slightly down, so d(0, 3) < d(0, 2), yielding the
    // valid 3-cluster UPGMA cut [{0, 1, 3}, {2}, {4}] with ARI = 1/21 (never 4 clusters).
    assert!(
        (res32.ari - 0.411_764_705_882_352_9).abs() > 1e-6,
        "f32 cq_hac produced 4-cluster inverted cut (ari = {})",
        res32.ari
    );
    assert_close(res32.ari, 1.0 / 21.0, 1e-10, "f32 cq_hac 3-cluster ari");
}
