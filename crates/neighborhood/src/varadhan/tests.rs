use egraph_core::CsrGraph;

use super::*;
use crate::knn::extract_sparse_1hop_2hop_pairs;

#[test]
fn test_horvitz_thompson_gamma_r3() {
    let g = compute_horvitz_thompson_gamma(1.0, 1600, 16).unwrap();
    assert!((g - 10.0).abs() < 1e-12);
    assert_eq!(compute_horvitz_thompson_gamma(1.0, 0, 16).unwrap(), 0.0);
    assert_eq!(compute_horvitz_thompson_gamma(0.5, 9, 0).unwrap(), 1.5);
    assert!(compute_horvitz_thompson_gamma(-0.1, 10, 4).is_err());
    assert!(compute_horvitz_thompson_gamma(f64::NAN, 10, 4).is_err());
}

#[test]
fn test_regularized_dog_varadhan_landmarks_alt_and_floor() {
    let edges: Vec<(u32, u32)> = (0..16)
        .map(|i| (i, (i + 1) % 16))
        .chain([(0, 8), (4, 12)])
        .collect();
    let g = CsrGraph::<f64>::from_edges(16, &edges, None, false, false).unwrap();

    let cfg = DogVaradhanConfig::default();
    let (lm1, st1) = compute_regularized_dog_varadhan_landmarks(&g, &cfg).unwrap();
    let (lm2, st2) = compute_regularized_dog_varadhan_landmarks(&g, &cfg).unwrap();
    assert_eq!(lm1, lm2);
    assert_eq!(st1, st2);
    assert!(st1.used_coarse_scaffold);
    assert!(st1.beta_eff >= 1e-4 && st1.beta_eff <= 0.5);
    assert!(lm1.d_dog().iter().all(|&v| v.is_finite() && v >= 0.0));
    assert!(
        lm1.d_coarse()
            .unwrap()
            .iter()
            .all(|&v| v.is_finite() && v >= 0.0)
    );

    let pairs = extract_sparse_1hop_2hop_pairs(&g, 8, 42).unwrap();
    let mut d1 = lm1.alt_pair_distances(&pairs.u1, &pairs.v1).unwrap();
    let mut d2 = lm1.alt_pair_distances(&pairs.u2, &pairs.v2).unwrap();
    let mut d3 = lm1.alt_pair_distances(&[0, 1], &[8, 9]).unwrap();
    let floor = apply_relative_distance_floor(&mut d1, &mut d2, &mut d3, 1e-6, 0.5).unwrap();
    assert!(floor >= 1e-6);
    assert!(
        d1.iter()
            .chain(d2.iter())
            .chain(d3.iter())
            .all(|&d| d >= floor)
    );

    // Empty and singleton graphs
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let (le, se) = compute_regularized_dog_varadhan_landmarks(&empty, &cfg).unwrap();
    assert_eq!(le.num_nodes(), 0);
    assert!(!se.used_coarse_scaffold);
    assert!(le.alt_pair_distances(&[], &[]).unwrap().is_empty());
    assert!(le.alt_pair_distances(&[0], &[0]).is_err());

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let (ls, ss) = compute_regularized_dog_varadhan_landmarks(&single, &cfg).unwrap();
    assert_eq!(ls.num_nodes(), 1);
    assert_eq!(ss.n_landmarks, 1);
    assert_eq!(ls.alt_pair_distances(&[0], &[0]).unwrap(), vec![0.0]);

    // Disconnected graph and NaN config rejection
    let disc =
        CsrGraph::<f64>::from_edges(6, &[(0, 1), (1, 2), (3, 4), (4, 5)], None, false, false)
            .unwrap();
    let (ld, _sd) = compute_regularized_dog_varadhan_landmarks(&disc, &cfg).unwrap();
    let d_disc = ld.alt_pair_distances(&[0, 3], &[2, 5]).unwrap();
    assert!(d_disc.iter().all(|v| v.is_finite() && *v >= 0.0));

    let mut bad_cfg = cfg;
    bad_cfg.t1 = Some(f64::NAN);
    assert!(compute_regularized_dog_varadhan_landmarks(&g, &bad_cfg).is_err());

    let mut bad_cfg2 = cfg;
    bad_cfg2.t_coarse = f64::NAN;
    assert!(compute_regularized_dog_varadhan_landmarks(&g, &bad_cfg2).is_err());

    // Fine DoG table and beta_auto must match identically whether coarse scaffold is enabled or disabled.
    let cfg_fine_only = DogVaradhanConfig {
        degree: Some(25),
        t_coarse: 0.0,
        ..cfg
    };
    let cfg_dual = DogVaradhanConfig {
        degree: Some(25),
        t_coarse: 48.0,
        ..cfg
    };
    let (lm_f, st_f) = compute_regularized_dog_varadhan_landmarks(&g, &cfg_fine_only).unwrap();
    let (lm_d, st_d) = compute_regularized_dog_varadhan_landmarks(&g, &cfg_dual).unwrap();
    assert_eq!(lm_f.landmarks(), lm_d.landmarks());
    assert!((st_f.beta_auto - st_d.beta_auto).abs() < 1e-14);
    for (&a, &b) in lm_f.d_dog().iter().zip(lm_d.d_dog().iter()) {
        assert!((a - b).abs() < 1e-13);
    }
    assert!(lm_f.d_coarse().is_none());
    assert!(lm_d.d_coarse().is_some());

    // Custom matrix via anisotropic_laplacian + dynamic lambda_max + t_coarse=0.0
    let aniso = egraph_linalg::anisotropic_laplacian(
        &g,
        egraph_linalg::AnisotropicLaplacianConfig {
            alpha: 0.0,
            beta: -0.5,
            degree_reg: 0.0,
            degree_cap: None,
            reweighted_degree_norm: false,
        },
    )
    .unwrap();
    let (lm_from_mat, st_from_mat) = compute_regularized_dog_varadhan_landmarks_from_matrix(
        &g,
        &aniso.matrix,
        None,
        &cfg_fine_only,
    )
    .unwrap();
    assert!(!st_from_mat.used_coarse_scaffold);
    assert!(lm_from_mat.d_coarse().is_none());
    assert!(
        lm_from_mat
            .d_dog()
            .iter()
            .all(|&v| v.is_finite() && v >= 0.0)
    );

    // Combinatorial Laplacian with large lambda_max > 2.0 and dynamic estimation
    let l_comb = egraph_linalg::combinatorial_laplacian(&g);
    let cfg_auto = DogVaradhanConfig {
        degree: None,
        t_coarse: 0.0,
        ..cfg
    };
    let (lm_comb, st_comb) =
        compute_regularized_dog_varadhan_landmarks_from_matrix(&g, &l_comb, None, &cfg_auto)
            .unwrap();
    assert!(st_comb.degree >= 40);
    assert!(lm_comb.d_dog().iter().all(|&v| v.is_finite() && v >= 0.0));

    // Zero matrix fast path
    let edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    let l_zero = egraph_linalg::combinatorial_laplacian(&edgeless);
    let (lm_zero, _) =
        compute_regularized_dog_varadhan_landmarks_from_matrix(&edgeless, &l_zero, None, &cfg_auto)
            .unwrap();
    assert!(lm_zero.d_dog().iter().all(|&v| v.is_finite() && v >= 0.0));
}

#[test]
fn test_pivoted_rhgh_landmarks_determinism_and_monotonicity() {
    // Path of 10 nodes: ALT distances along the path must increase monotonically from 0.
    let edges: Vec<(u32, u32)> = (0..9).map(|i| (i, i + 1)).collect();
    let g = CsrGraph::<f64>::from_edges(10, &edges, None, false, false).unwrap();

    let cfg = RhghLandmarkConfig {
        n_landmarks: 10,
        num_probes: 32,
        min_degree: 1,
        seed: 42,
        ..RhghLandmarkConfig::default()
    };
    let (lm1, st1) = compute_pivoted_rhgh_landmarks(&g, &cfg).unwrap();
    let (lm2, st2) = compute_pivoted_rhgh_landmarks(&g, &cfg).unwrap();
    assert_eq!(lm1, lm2);
    assert_eq!(st1, st2);
    assert!(st1.all_converged);
    assert!(st1.gamma >= 0.5 && st1.gamma <= 32.0);
    assert!(st1.p_exp >= 0.6 && st1.p_exp <= 0.9);
    assert!(lm1.d_dog().iter().all(|&v| v.is_finite() && v >= 0.0));

    let (rhgh_res, rhgh_st) = compute_pivoted_rhgh_landmarks_result(&g, &cfg).unwrap();
    assert_eq!(rhgh_st, st1);
    assert_eq!(rhgh_res.num_nodes(), 10);
    assert_eq!(rhgh_res.num_landmarks(), 10);
    assert_eq!(rhgh_res.d_rhgh(), lm1.d_dog());
    assert_eq!(
        rhgh_res.alt_pair_distances(&[0, 0], &[1, 9]).unwrap(),
        lm1.alt_pair_distances(&[0, 0], &[1, 9]).unwrap()
    );
    let converted: DogVaradhanLandmarks = rhgh_res.into();
    assert_eq!(converted, lm1);

    let d01 = lm1.alt_pair_distances(&[0], &[1]).unwrap()[0];
    let d05 = lm1.alt_pair_distances(&[0], &[5]).unwrap()[0];
    let d09 = lm1.alt_pair_distances(&[0], &[9]).unwrap()[0];
    assert!(d01 > 0.0 && d05 > d01 && d09 > d05);

    // Empty, singleton, edgeless, and invalid configs.
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    assert_eq!(
        compute_pivoted_rhgh_landmarks(&empty, &cfg)
            .unwrap()
            .0
            .num_nodes(),
        0
    );
    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    assert_eq!(
        compute_pivoted_rhgh_landmarks(&single, &cfg)
            .unwrap()
            .0
            .num_landmarks(),
        1
    );
    let edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    let (le, se) = compute_pivoted_rhgh_landmarks(&edgeless, &cfg).unwrap();
    assert!(se.all_converged);
    assert!(le.d_dog().iter().all(|&v| v.is_finite() && v >= 0.0));

    let bad_gamma = RhghLandmarkConfig {
        gamma: Some(-1.0),
        ..cfg
    };
    assert!(compute_pivoted_rhgh_landmarks(&g, &bad_gamma).is_err());
    let bad_p = RhghLandmarkConfig {
        p_exp: Some(f64::NAN),
        ..cfg
    };
    assert!(compute_pivoted_rhgh_landmarks(&g, &bad_p).is_err());
}
