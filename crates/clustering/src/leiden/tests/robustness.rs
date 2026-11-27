use egraph_core::{CoreError, CsrGraph};

use super::super::*;
use super::fixtures::{
    assert_canonical_and_connected, make_two_cliques_barbell, oracle_quality_and_modularity,
};

#[test]
fn test_leiden_adversarial_bridge_connectivity_across_seeds_and_single_pass() {
    // Bridge vertex 0 connects two K_4 subgraphs (1..=4, 5..=8) and a stronger K_5 (9..=13)
    // that pulls vertex 0 away after Pass 0, testing Leiden refinement and BFS guard.
    let mut edges = Vec::new();
    for i in 1..=4u32 {
        for j in (i + 1)..=4 {
            edges.push((i, j));
        }
    }
    for i in 5..=8u32 {
        for j in (i + 1)..=8 {
            edges.push((i, j));
        }
    }
    for i in 9..=13u32 {
        for j in (i + 1)..=13 {
            edges.push((i, j));
        }
    }
    edges.extend_from_slice(&[(0, 1), (0, 2), (0, 5), (0, 6), (0, 9), (0, 10), (0, 11)]);
    let mut weights = vec![1.0_f64; edges.len()];
    let len = weights.len();
    weights[len - 3] = 2.0;
    weights[len - 2] = 2.0;
    weights[len - 1] = 2.0;
    let bridge_graph = CsrGraph::from_edges(14, &edges, Some(&weights), false, false).unwrap();

    for seed in 0..20u64 {
        for max_passes in [1usize, 100] {
            for &(objective, resolution) in &[
                (LeidenObjective::Modularity, 1.0),
                (LeidenObjective::Cpm, 0.25),
                (LeidenObjective::Cpm, 0.0),
            ] {
                let res = leiden(
                    &bridge_graph,
                    &LeidenConfig {
                        objective,
                        resolution,
                        max_passes,
                        seed,
                        ..Default::default()
                    },
                )
                .unwrap();
                assert_canonical_and_connected(&bridge_graph, &res);
                let (oq, om) = oracle_quality_and_modularity(
                    &bridge_graph,
                    &res.labels,
                    res.num_communities,
                    objective,
                    resolution,
                );
                assert!((res.quality - oq).abs() < 1e-12);
                assert!((res.modularity - om).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn test_leiden_log_sum_exp_extreme_weights_and_scales() {
    let mut edges = Vec::new();
    let mut weights_large = Vec::new();
    let mut weights_small = Vec::new();
    for i in 0..4u32 {
        for j in (i + 1)..4 {
            edges.push((i, j));
            weights_large.push(1e9_f64);
            weights_small.push(1e-9_f64);
            edges.push((4 + i, 4 + j));
            weights_large.push(1e9_f64);
            weights_small.push(1e-9_f64);
        }
    }
    edges.push((3, 4));
    weights_large.push(1e6_f64);
    weights_small.push(1e-12_f64);

    let g_large = CsrGraph::from_edges(8, &edges, Some(&weights_large), false, false).unwrap();
    let g_small = CsrGraph::from_edges(8, &edges, Some(&weights_small), false, false).unwrap();

    for (g, cpm_gamma, cpm_theta, min_gain) in [
        (&g_large, 1e8_f64, 1e-6_f64, 1e-7_f64),
        (&g_small, 1e-10_f64, 1e-15_f64, 0.0_f64),
    ] {
        for objective in [LeidenObjective::Modularity, LeidenObjective::Cpm] {
            let (resolution, randomness, min_quality_gain) = match objective {
                LeidenObjective::Modularity => (1.0, 1e-6, 1e-7),
                LeidenObjective::Cpm => (cpm_gamma, cpm_theta, min_gain),
            };
            let res = leiden(
                g,
                &LeidenConfig {
                    objective,
                    resolution,
                    randomness,
                    min_quality_gain,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(res.quality.is_finite());
            assert!(res.modularity.is_finite());
            assert_canonical_and_connected(g, &res);
            assert_eq!(res.num_communities, 2);
            let (oq, om) = oracle_quality_and_modularity(
                g,
                &res.labels,
                res.num_communities,
                objective,
                resolution,
            );
            assert!((res.quality - oq).abs() <= 1e-12 * (1.0 + oq.abs()));
            assert!((res.modularity - om).abs() <= 1e-12);
        }
    }
}

#[test]
fn test_leiden_determinism_degenerate_and_disconnected_cases() {
    let barbell = make_two_cliques_barbell(5);
    let cfg = LeidenConfig {
        seed: 1234,
        ..Default::default()
    };
    assert_eq!(
        leiden(&barbell, &cfg).unwrap(),
        leiden(&barbell, &cfg).unwrap()
    );

    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let re = leiden(&empty, &cfg).unwrap();
    assert!(re.labels.is_empty());
    assert_eq!(re.num_communities, 0);
    assert_eq!(re.quality, 0.0);
    assert_eq!(re.modularity, 0.0);
    assert_eq!(re.passes, 0);

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let rs = leiden(&single, &cfg).unwrap();
    assert_eq!(rs.labels, vec![0]);
    assert_eq!(rs.num_communities, 1);
    assert_eq!(rs.quality, 0.0);
    assert_eq!(rs.modularity, 0.0);

    let edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    let ri = leiden(&edgeless, &cfg).unwrap();
    assert_eq!(ri.labels, vec![0, 1, 2, 3]);
    assert_eq!(ri.num_communities, 4);
    assert_eq!(ri.quality, 0.0);
    assert_eq!(ri.modularity, 0.0);

    // Zero-weight edges behave identically to edgeless.
    let zero_w =
        CsrGraph::from_edges(3, &[(0, 1), (1, 2)], Some(&[0.0, 0.0]), false, false).unwrap();
    let rz = leiden(&zero_w, &cfg).unwrap();
    assert_eq!(rz.labels, vec![0, 1, 2]);
    assert_eq!(rz.num_communities, 3);
    assert_eq!(rz.modularity, 0.0);

    // Disconnected two triangles + isolated node: never merge even at resolution = 0.0.
    let disc = CsrGraph::<f64>::from_edges(
        7,
        &[(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)],
        None,
        false,
        false,
    )
    .unwrap();
    for objective in [LeidenObjective::Modularity, LeidenObjective::Cpm] {
        for resolution in [0.0, 0.5, 1.0] {
            let rd = leiden(
                &disc,
                &LeidenConfig {
                    objective,
                    resolution,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_canonical_and_connected(&disc, &rd);
            assert!(rd.num_communities >= 3);
            if resolution <= 0.5 {
                assert_eq!(rd.labels, vec![0, 0, 0, 1, 1, 1, 2]);
            }
        }
    }
}

#[test]
fn test_leiden_validation_errors() {
    let cfg = LeidenConfig::default();
    let directed = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, true, false).unwrap();
    assert!(matches!(
        leiden(&directed, &cfg),
        Err(CoreError::InvalidInput(_))
    ));

    let g = make_two_cliques_barbell(3);
    for bad_cfg in [
        LeidenConfig {
            max_passes: 0,
            ..Default::default()
        },
        LeidenConfig {
            resolution: -0.5,
            ..Default::default()
        },
        LeidenConfig {
            randomness: 0.0,
            ..Default::default()
        },
        LeidenConfig {
            randomness: -0.01,
            ..Default::default()
        },
        LeidenConfig {
            min_quality_gain: -1e-6,
            ..Default::default()
        },
    ] {
        assert!(matches!(
            leiden(&g, &bad_cfg),
            Err(CoreError::InvalidInput(_))
        ));
    }

    for non_finite_cfg in [
        LeidenConfig {
            resolution: f64::NAN,
            ..Default::default()
        },
        LeidenConfig {
            resolution: f64::INFINITY,
            ..Default::default()
        },
        LeidenConfig {
            randomness: f64::NAN,
            ..Default::default()
        },
        LeidenConfig {
            randomness: f64::INFINITY,
            ..Default::default()
        },
        LeidenConfig {
            min_quality_gain: f64::NAN,
            ..Default::default()
        },
        LeidenConfig {
            min_quality_gain: f64::INFINITY,
            ..Default::default()
        },
    ] {
        assert!(matches!(
            leiden(&g, &non_finite_cfg),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }

    let neg_w = CsrGraph::from_edges(2, &[(0, 1)], Some(&[-1.0_f64]), false, false).unwrap();
    assert!(matches!(
        leiden(&neg_w, &cfg),
        Err(CoreError::InvalidInput(_))
    ));
}

#[test]
fn test_leiden_extreme_weights_never_return_nan_or_inf() {
    let g_subnormal = CsrGraph::from_edges(
        4,
        &[(0, 1), (1, 2), (2, 3)],
        Some(&[1e-312_f64; 3]),
        false,
        false,
    )
    .unwrap();
    let res_sub_mod = leiden(&g_subnormal, &LeidenConfig::default());
    assert!(matches!(res_sub_mod, Err(CoreError::NumericalBreakdown(_))));

    for (resolution, min_quality_gain, expected_comms) in
        [(1.0_f64, 1e-7_f64, 4usize), (0.0_f64, 0.0_f64, 1usize)]
    {
        let res_sub_cpm = leiden(
            &g_subnormal,
            &LeidenConfig {
                objective: LeidenObjective::Cpm,
                resolution,
                min_quality_gain,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(res_sub_cpm.quality.is_finite());
        assert!(res_sub_cpm.modularity.is_finite());
        assert_canonical_and_connected(&g_subnormal, &res_sub_cpm);
        assert_eq!(res_sub_cpm.num_communities, expected_comms);
    }

    let g_overflow = CsrGraph::from_edges(
        4,
        &[(0, 1), (1, 2), (2, 3)],
        Some(&[1e308_f64; 3]),
        false,
        false,
    )
    .unwrap();
    for objective in [LeidenObjective::Modularity, LeidenObjective::Cpm] {
        let res = leiden(
            &g_overflow,
            &LeidenConfig {
                objective,
                ..Default::default()
            },
        );
        assert!(matches!(res, Err(CoreError::NumericalBreakdown(_))));
    }

    // Single edge of 1e308 overflows total_weight_2m = 2e308 even when each degree is 1e308.
    let g_pair_overflow =
        CsrGraph::from_edges(2, &[(0, 1)], Some(&[1e308_f64]), false, false).unwrap();
    for objective in [LeidenObjective::Modularity, LeidenObjective::Cpm] {
        let res = leiden(
            &g_pair_overflow,
            &LeidenConfig {
                objective,
                ..Default::default()
            },
        );
        assert!(matches!(res, Err(CoreError::NumericalBreakdown(_))));
    }

    // Huge resolution on small total_weight_2m overflows gamma_eff cleanly without NaN.
    let g_tiny_normal = CsrGraph::from_edges(
        4,
        &[(0, 1), (1, 2), (2, 3)],
        Some(&[1e-305_f64; 3]),
        false,
        false,
    )
    .unwrap();
    let res_gamma_overflow = leiden(
        &g_tiny_normal,
        &LeidenConfig {
            objective: LeidenObjective::Modularity,
            resolution: 1e10,
            ..Default::default()
        },
    );
    assert!(matches!(
        res_gamma_overflow,
        Err(CoreError::NumericalBreakdown(_))
    ));

    // Huge f64::MAX resolution must never return Ok with NaN or Inf.
    let g_unit = make_two_cliques_barbell(3);
    for objective in [LeidenObjective::Modularity, LeidenObjective::Cpm] {
        match leiden(
            &g_unit,
            &LeidenConfig {
                objective,
                resolution: f64::MAX,
                ..Default::default()
            },
        ) {
            Ok(res) => {
                assert!(res.quality.is_finite());
                assert!(res.modularity.is_finite());
            }
            Err(CoreError::NumericalBreakdown(_)) => {}
            Err(other) => panic!("unexpected error variant: {other:?}"),
        }
    }
}
