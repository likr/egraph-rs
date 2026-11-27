use super::*;
use egraph_core::DeterministicRng;

fn seeded_layout<S: Scalar>(n: usize, scale: f64, seed: u64) -> Layout2d<S> {
    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        x.push(S::from_f64((rng.next_f64() * 2.0 - 1.0) * scale));
        y.push(S::from_f64((rng.next_f64() * 2.0 - 1.0) * scale));
    }
    Layout2d::from_xy(x, y).unwrap()
}

#[test]
fn test_backbone_solver_step_vs_run_bit_exact_modes_and_dtypes() {
    let mut edges = Vec::new();
    for u in 0..5u32 {
        for v in (u + 1)..5u32 {
            edges.push((u, v));
        }
    }
    for u in 5..10u32 {
        for v in (u + 1)..10u32 {
            edges.push((u, v));
        }
    }
    edges.push((4, 5));
    let g = CsrGraph::<f64>::from_edges(10, &edges, None, false, false).unwrap();

    for mode in [
        BackboneLayoutMode::StressMajorization,
        BackboneLayoutMode::Sgd,
    ] {
        let cfg = BackboneConfig {
            retain_ratio: 0.25,
            umst_weight_scale: 1.5,
            non_backbone_weight_scale: 0.2,
            use_weights: true,
            layout_mode: mode,
            iterations: 25,
            seed: 77,
        };

        // f64 bit-exact check
        let mut s_run = BackboneSolver::<f64>::new(&g, cfg).unwrap();
        let mut s_step = BackboneSolver::<f64>::new(&g, cfg).unwrap();
        let mut l_run = seeded_layout::<f64>(10, 1.0, 99);
        let mut l_step = l_run.clone();

        assert_eq!(s_run.total_iterations(), 25);
        assert!(s_run.run(&mut l_run).unwrap().is_finished());
        assert!(l_run.is_all_finite());
        let _ = s_step.run_steps(&mut l_step, 5).unwrap();
        while !s_step.status().is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        assert_eq!(s_run.status(), s_step.status());
        assert_eq!(s_run.iteration(), s_step.iteration());
        for i in 0..10 {
            assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
            assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
        }

        // f32 bit-exact check
        let mut s32_run = BackboneSolver::<f32>::new(&g, cfg).unwrap();
        let mut s32_step = BackboneSolver::<f32>::new(&g, cfg).unwrap();
        let mut l32_run = seeded_layout::<f32>(10, 1.0, 99);
        let mut l32_step = l32_run.clone();
        s32_run.run(&mut l32_run).unwrap();
        while !s32_step.status().is_finished() {
            s32_step.step(&mut l32_step).unwrap();
        }
        assert!(l32_run.is_all_finite());
        for i in 0..10 {
            assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
            assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
        }
    }
}

#[test]
fn test_backbone_degenerate_directed_and_validation_errors() {
    let cfg = BackboneConfig::default();

    // Empty graph
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let res0 = extract_quadrilateral_backbone(&empty, &cfg).unwrap();
    assert!(res0.simmelian_weights.is_empty());
    assert!(res0.backbone_edges.is_empty());
    let mut s0 = BackboneSolver::<f64>::new(&empty, cfg).unwrap();
    let mut l0 = Layout2d::<f64>::zeros(0);
    assert!(s0.run(&mut l0).unwrap().is_finished());

    // Singleton graph
    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let mut s1 = BackboneSolver::<f64>::new(&single, cfg).unwrap();
    let mut l1 = Layout2d::from_xy(vec![2.0], vec![-3.0]).unwrap();
    assert!(s1.run(&mut l1).unwrap().is_finished());
    assert_eq!(l1.x(), &[2.0]);
    assert_eq!(l1.y(), &[-3.0]);

    // Edgeless + disconnected + isolated vertices
    let disc = CsrGraph::<f64>::from_edges(5, &[(0, 1), (2, 3)], None, false, false).unwrap();
    let mut s_disc = BackboneSolver::<f64>::new(&disc, cfg).unwrap();
    let mut l_disc = seeded_layout::<f64>(5, 1.0, 5);
    assert!(s_disc.run(&mut l_disc).unwrap().is_finished());
    assert!(l_disc.is_all_finite());

    // Directed graphs are rejected, while undirected self-loops and parallel edges are deduplicated.
    let undir = CsrGraph::<f64>::from_edges(
        4,
        &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        None,
        false,
        false,
    )
    .unwrap();
    let undir_dup = CsrGraph::<f64>::from_edges(
        4,
        &[
            (0, 0),
            (0, 1),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (3, 3),
        ],
        None,
        false,
        true,
    )
    .unwrap();
    let dir = CsrGraph::<f64>::from_edges(
        4,
        &[
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 2),
            (0, 3),
            (1, 2),
            (2, 1),
            (1, 3),
            (2, 3),
            (3, 3),
        ],
        None,
        true,
        true,
    )
    .unwrap();
    assert!(matches!(
        extract_quadrilateral_backbone(&dir, &cfg),
        Err(CoreError::InvalidInput { .. })
    ));
    assert!(matches!(
        BackboneSolver::<f64>::new(&dir, cfg),
        Err(CoreError::InvalidInput { .. })
    ));
    let r_u = extract_quadrilateral_backbone(&undir, &cfg).unwrap();
    let r_dup = extract_quadrilateral_backbone(&undir_dup, &cfg).unwrap();
    assert_eq!(r_u.simmelian_weights, r_dup.simmelian_weights);
    assert_eq!(r_u.umst_edges, r_dup.umst_edges);
    assert_eq!(r_u.backbone_edges, r_dup.backbone_edges);

    // iterations == 0 completes immediately without touching layout.
    let mut s_zero = BackboneSolver::<f64>::new(
        &undir,
        BackboneConfig {
            iterations: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(s_zero.status(), SolverStatus::Completed);
    let mut l_zero = seeded_layout::<f64>(4, 1.0, 1);
    let before = l_zero.clone();
    assert_eq!(s_zero.run(&mut l_zero).unwrap(), SolverStatus::Completed);
    assert_eq!(l_zero, before);

    // Invalid config checks
    for bad in [
        BackboneConfig {
            retain_ratio: f64::NAN,
            ..Default::default()
        },
        BackboneConfig {
            retain_ratio: -0.1,
            ..Default::default()
        },
        BackboneConfig {
            retain_ratio: 1.1,
            ..Default::default()
        },
        BackboneConfig {
            umst_weight_scale: -1.0,
            ..Default::default()
        },
        BackboneConfig {
            umst_weight_scale: f64::INFINITY,
            ..Default::default()
        },
        BackboneConfig {
            non_backbone_weight_scale: -0.1,
            ..Default::default()
        },
        BackboneConfig {
            non_backbone_weight_scale: f64::NAN,
            ..Default::default()
        },
    ] {
        assert!(extract_quadrilateral_backbone(&undir, &bad).is_err());
    }

    // Negative input weights rejected
    let neg_w = CsrGraph::from_csr(&[0, 1, 2], &[1, 0], Some(&[-1.0, -1.0]), false, false).unwrap();
    assert!(extract_quadrilateral_backbone(&neg_w, &cfg).is_err());

    // Layout dimension mismatch rejected
    let mut wrong_l = Layout2d::<f64>::zeros(2);
    assert!(matches!(
        s_disc.step(&mut wrong_l),
        Err(CoreError::DimensionMismatch { .. })
    ));

    // Weighted K_4 with one weak edge: `use_weights: true` prioritizes heavier edges while
    // `use_weights: false` preserves unweighted symmetry.
    let k4_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let k4_w = CsrGraph::from_edges(
        4,
        &k4_edges,
        Some(&[0.1, 10.0, 10.0, 10.0, 10.0, 10.0]),
        false,
        false,
    )
    .unwrap();
    let res_unweighted = extract_quadrilateral_backbone(
        &k4_w,
        &BackboneConfig {
            retain_ratio: 0.0,
            use_weights: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(res_unweighted.umst_edges.len(), 6);
    assert_eq!(res_unweighted.edge_weights, vec![1.0; 6]);

    let res_weighted = extract_quadrilateral_backbone(
        &k4_w,
        &BackboneConfig {
            retain_ratio: 0.0,
            use_weights: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!res_weighted.umst_edges.contains(&(0, 1)));
    assert_eq!(res_weighted.umst_edges.len(), 5);
    assert!(res_weighted.backbone_csr.is_weighted());
}
