use std::f64::consts::FRAC_PI_2;

use egraph_core::{CoreError, ManifoldKind, ManifoldLayout2d};

use super::assert_close;
use crate::cluster_baselines::*;

fn make_separated_manifold_fixture(kind: ManifoldKind) -> (ManifoldLayout2d<f64>, Vec<u32>) {
    let (xs, ys) = match kind {
        ManifoldKind::HyperbolicPoincare => (
            vec![-0.62, -0.58, -0.60, 0.62, 0.58, 0.60],
            vec![0.02, -0.02, 0.0, 0.02, -0.02, 0.0],
        ),
        ManifoldKind::HyperbolicLorentz => (
            vec![-5.0, -5.2, -4.8, 5.0, 5.2, 4.8],
            vec![0.1, -0.1, 0.0, 0.1, -0.1, 0.0],
        ),
        ManifoldKind::Spherical => (
            vec![0.0, 0.05, -0.05, 2.0, 2.05, 1.95],
            vec![
                FRAC_PI_2,
                FRAC_PI_2 + 0.04,
                FRAC_PI_2 - 0.04,
                FRAC_PI_2,
                FRAC_PI_2 + 0.04,
                FRAC_PI_2 - 0.04,
            ],
        ),
        ManifoldKind::Torus2d => (
            // Cluster 0 straddles the periodic boundary x = 0.0/1.0!
            vec![0.99, 0.01, 0.0, 0.40, 0.41, 0.39],
            vec![0.20, 0.21, 0.19, 0.55, 0.56, 0.54],
        ),
        _ => unreachable!(),
    };
    let ml = ManifoldLayout2d::from_xy(kind, xs, ys).unwrap();
    let labels = vec![0, 0, 0, 1, 1, 1];
    (ml, labels)
}

#[test]
fn manifold_cluster_metrics_separated_all_kinds_and_scale_invariance() {
    for kind in [
        ManifoldKind::HyperbolicPoincare,
        ManifoldKind::HyperbolicLorentz,
        ManifoldKind::Spherical,
        ManifoldKind::Torus2d,
    ] {
        let (mut ml, labels) = make_separated_manifold_fixture(kind);
        let dsc = distance_consistency_manifold(&ml, &labels).unwrap();
        let dc = distribution_consistency_manifold(&ml, &labels, 8).unwrap();
        let nh = neighborhood_hit_manifold(&ml, &labels, 2).unwrap();
        let sil = silhouette_score_manifold(&ml, &labels).unwrap();
        let ch = calinski_harabasz_score_manifold(&ml, &labels).unwrap();
        let db = davies_bouldin_score_manifold(&ml, &labels).unwrap();
        let pdoe = pairwise_distance_overlap_error_manifold(&ml, &labels, 10).unwrap();
        let cq_km = cq_kmeans_manifold(&ml, &labels, 5, 50, 42).unwrap();
        let cq_hc = cq_hac_manifold(&ml, &labels).unwrap();
        let knng = sepme_knng_manifold(&ml, &labels, 2).unwrap();
        let rng = sepme_rng_manifold(&ml, &labels).unwrap();
        let gab = sepme_gab_manifold(&ml, &labels).unwrap();
        let gong = sepme_gong_manifold(&ml, &labels, 0.35).unwrap();

        assert_close(dsc, 1.0, 1e-12, "dsc");
        assert_close(dc, 1.0, 1e-12, "dc");
        assert_close(nh, 1.0, 1e-12, "nh");
        assert!(sil > 0.8, "{kind:?}: expected sil > 0.8, got {sil}");
        assert!(ch > 10.0, "{kind:?}: expected ch > 10.0, got {ch}");
        assert!(db < 0.2, "{kind:?}: expected db < 0.2, got {db}");
        assert_close(pdoe, 0.0, 1e-12, "pdoe");
        assert_close(cq_km.ari, 1.0, 1e-12, "cq_kmeans ari");
        assert_close(cq_hc.ari, 1.0, 1e-12, "cq_hac ari");
        assert_close(knng, 1.0, 1e-12, "sepme_knng");
        assert!(rng > 0.5 && rng <= 1.0, "{kind:?}: rng = {rng}");
        assert!(gab > 0.5 && gab <= 1.0, "{kind:?}: gab = {gab}");
        assert!(gong > 0.5 && gong <= 1.0, "{kind:?}: gong = {gong}");
        let mixed = vec![0, 1, 0, 1, 0, 1];
        assert!(rng > sepme_rng_manifold(&ml, &mixed).unwrap());
        assert!(gab > sepme_gab_manifold(&ml, &mixed).unwrap());
        assert!(gong > sepme_gong_manifold(&ml, &mixed, 0.35).unwrap());

        // Conformal metric scaling invariance across all 13 scores
        ml.set_scale(3.7).unwrap();
        assert_close(
            distance_consistency_manifold(&ml, &labels).unwrap(),
            dsc,
            1e-12,
            "dsc scale",
        );
        assert_close(
            distribution_consistency_manifold(&ml, &labels, 8).unwrap(),
            dc,
            1e-12,
            "dc scale",
        );
        assert_close(
            neighborhood_hit_manifold(&ml, &labels, 2).unwrap(),
            nh,
            1e-12,
            "nh scale",
        );
        assert_close(
            sepme_knng_manifold(&ml, &labels, 2).unwrap(),
            knng,
            1e-12,
            "knng scale",
        );
        assert_close(
            sepme_rng_manifold(&ml, &labels).unwrap(),
            rng,
            1e-12,
            "rng scale",
        );
        assert_close(
            sepme_gab_manifold(&ml, &labels).unwrap(),
            gab,
            1e-12,
            "gab scale",
        );
        assert_close(
            silhouette_score_manifold(&ml, &labels).unwrap(),
            sil,
            1e-12,
            "sil scale",
        );
        assert_close(
            calinski_harabasz_score_manifold(&ml, &labels).unwrap(),
            ch,
            1e-10,
            "ch scale",
        );
        assert_close(
            davies_bouldin_score_manifold(&ml, &labels).unwrap(),
            db,
            1e-10,
            "db scale",
        );
        assert_close(
            pairwise_distance_overlap_error_manifold(&ml, &labels, 10).unwrap(),
            pdoe,
            1e-12,
            "pdoe scale",
        );
        assert_close(
            cq_kmeans_manifold(&ml, &labels, 5, 50, 42).unwrap().ari,
            cq_km.ari,
            1e-12,
            "km scale",
        );
        assert_close(
            cq_hac_manifold(&ml, &labels).unwrap().ari,
            cq_hc.ari,
            1e-12,
            "hac scale",
        );
        assert_close(
            sepme_gong_manifold(&ml, &labels, 0.35).unwrap(),
            gong,
            1e-12,
            "gong scale",
        );
    }
}

#[test]
fn manifold_cluster_torus_aspect_and_spherical_antipodal_fallback() {
    // Rectangular torus aspect ratio changes nearest neighbors in neighborhood_hit & sepme_knng.
    let mut torus = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.1, 0.1, 0.25, 0.25],
        vec![0.1, 0.35, 0.1, 0.35],
    )
    .unwrap();
    let labels = vec![0, 0, 1, 1];
    assert_close(
        neighborhood_hit_manifold(&torus, &labels, 1).unwrap(),
        0.0,
        1e-12,
        "nh square",
    );
    torus.set_torus_extents(4.0, 1.0).unwrap();
    assert_close(
        neighborhood_hit_manifold(&torus, &labels, 1).unwrap(),
        1.0,
        1e-12,
        "nh rect",
    );
    assert_close(
        sepme_knng_manifold(&torus, &labels, 1).unwrap(),
        1.0,
        1e-12,
        "knng rect",
    );

    // Antipodal cluster on S^2 (zero 3D mean) must not NaN or panic.
    let sph_anti = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Spherical,
        vec![0.0, std::f64::consts::PI, 1.0, 1.0 + std::f64::consts::PI],
        vec![FRAC_PI_2, FRAC_PI_2, 0.5, std::f64::consts::PI - 0.5],
    )
    .unwrap();
    let anti_labels = vec![0, 0, 1, 1];
    assert!(
        distance_consistency_manifold(&sph_anti, &anti_labels)
            .unwrap()
            .is_finite()
    );
    assert!(
        calinski_harabasz_score_manifold(&sph_anti, &anti_labels)
            .unwrap()
            .is_finite()
    );
    assert!(
        davies_bouldin_score_manifold(&sph_anti, &anti_labels)
            .unwrap()
            .is_finite()
    );
    assert!(
        cq_kmeans_manifold(&sph_anti, &anti_labels, 2, 20, 7)
            .unwrap()
            .ari
            .is_finite()
    );

    // GONG across gamma <= 0.5 and gamma > 0.5
    let lor = ManifoldLayout2d::<f32>::init(ManifoldKind::HyperbolicLorentz, 8, 99, 1.2).unwrap();
    let l8 = vec![0, 0, 0, 0, 1, 1, 1, 1];
    for gamma in [0.0, 0.35, 0.5, 0.75] {
        let g_score = sepme_gong_manifold(&lor, &l8, gamma).unwrap();
        assert!(g_score.is_finite() && (0.0..=1.0).contains(&g_score));
    }
}

#[test]
fn manifold_cluster_degeneracies_and_validation_errors() {
    let empty = ManifoldLayout2d::<f64>::zeros(ManifoldKind::HyperbolicPoincare, 0);
    assert_eq!(distance_consistency_manifold(&empty, &[]).unwrap(), 0.0);
    assert_eq!(
        distribution_consistency_manifold(&empty, &[], 4).unwrap(),
        0.0
    );
    assert_eq!(neighborhood_hit_manifold(&empty, &[], 3).unwrap(), 0.0);
    assert_eq!(silhouette_score_manifold(&empty, &[]).unwrap(), 0.0);
    assert_eq!(calinski_harabasz_score_manifold(&empty, &[]).unwrap(), 0.0);
    assert_eq!(davies_bouldin_score_manifold(&empty, &[]).unwrap(), 0.0);
    assert_eq!(
        pairwise_distance_overlap_error_manifold(&empty, &[], 5).unwrap(),
        0.0
    );
    assert_eq!(cq_kmeans_manifold(&empty, &[], 1, 5, 1).unwrap().ari, 0.0);
    assert_eq!(cq_hac_manifold(&empty, &[]).unwrap().ari, 0.0);
    assert_eq!(sepme_knng_manifold(&empty, &[], 2).unwrap(), 0.0);
    assert_eq!(sepme_rng_manifold(&empty, &[]).unwrap(), 0.0);
    assert_eq!(sepme_gab_manifold(&empty, &[]).unwrap(), 0.0);
    assert_eq!(sepme_gong_manifold(&empty, &[], 0.35).unwrap(), 0.0);

    let single = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Torus2d, 1);
    assert_eq!(sepme_knng_manifold(&single, &[0], 2).unwrap(), 1.0);
    assert_eq!(sepme_rng_manifold(&single, &[0]).unwrap(), 1.0);
    assert_eq!(sepme_gab_manifold(&single, &[0]).unwrap(), 1.0);
    assert_eq!(sepme_gong_manifold(&single, &[0], 0.35).unwrap(), 1.0);

    // Collapsed 4-node layout
    let collapsed = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Spherical,
        vec![0.1, 0.5, 1.0, 2.0],
        vec![0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    let l4 = vec![0, 0, 1, 1];
    assert_eq!(
        calinski_harabasz_score_manifold(&collapsed, &l4).unwrap(),
        1.0
    );
    assert_eq!(davies_bouldin_score_manifold(&collapsed, &l4).unwrap(), 0.0);
    assert_eq!(
        pairwise_distance_overlap_error_manifold(&collapsed, &l4, 5).unwrap(),
        1.0
    );
    assert_eq!(
        cq_kmeans_manifold(&collapsed, &l4, 2, 10, 1).unwrap().ari,
        0.0
    );
    assert_eq!(cq_hac_manifold(&collapsed, &l4).unwrap().ari, 0.0);

    // Error validation
    assert!(matches!(
        neighborhood_hit_manifold(&collapsed, &l4, 0),
        Err(CoreError::InvalidInput { .. })
    ));
    assert!(matches!(
        sepme_gong_manifold(&collapsed, &l4, 1.5),
        Err(CoreError::InvalidInput { .. })
    ));
    assert!(matches!(
        sepme_gong_manifold(&collapsed, &l4, f64::NAN),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        distance_consistency_manifold(&collapsed, &[0, 1]),
        Err(CoreError::DimensionMismatch { .. })
    ));
}
