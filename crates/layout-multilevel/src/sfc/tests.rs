use super::*;

fn assert_strictly_distinct<S: Scalar>(layout: &Layout2d<S>) {
    let n = layout.len();
    let mut pts: Vec<(u64, u64)> = (0..n)
        .map(|i| {
            (
                layout.x()[i].to_f64().to_bits(),
                layout.y()[i].to_f64().to_bits(),
            )
        })
        .collect();
    pts.sort_unstable();
    for w in pts.windows(2) {
        assert_ne!(w[0], w[1], "co-located points found in SFC layout");
    }
}

#[test]
fn test_sfc_config_validation_and_degenerate_inputs() {
    assert!(SfcConfig::default().validate().is_ok());
    assert!(matches!(
        SfcConfig {
            scale: 0.0,
            ..Default::default()
        }
        .validate(),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        SfcConfig {
            scale: f64::NAN,
            ..Default::default()
        }
        .validate(),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        SfcConfig {
            resolution_order: Some(0),
            ..Default::default()
        }
        .validate(),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        SfcConfig {
            resolution_order: Some(16),
            ..Default::default()
        }
        .validate(),
        Err(CoreError::InvalidInput(_))
    ));

    // Explicit resolution_order too small for N -> error.
    let g5 = CsrGraph::<f64>::from_edges(5, &[(0, 1)], None, false, false).unwrap();
    assert!(matches!(
        SfcLayoutSolver::<f64>::new(
            &g5,
            SfcConfig {
                resolution_order: Some(1), // 4 cells < 5 nodes
                ..Default::default()
            },
        ),
        Err(CoreError::InvalidInput(_))
    ));

    // N = 0 and N = 1.
    let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let mut s0 = SfcLayoutSolver::<f64>::new(&g0, SfcConfig::default()).unwrap();
    let mut l0 = Layout2d::zeros(0);
    assert_eq!(s0.step(&mut l0).unwrap(), SolverStatus::Converged);

    let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let mut s1 = SfcLayoutSolver::<f64>::new(&g1, SfcConfig::default()).unwrap();
    let mut l1 = Layout2d::zeros(1);
    assert_eq!(s1.step(&mut l1).unwrap(), SolverStatus::Converged);
    assert_eq!(s1.vertex_order(), &[0]);

    // Edgeless graph (N = 10).
    let g_edgeless = CsrGraph::<f64>::from_edges(10, &[], None, false, false).unwrap();
    let mut s_edgeless = SfcLayoutSolver::<f64>::new(&g_edgeless, SfcConfig::default()).unwrap();
    let mut l_edgeless = Layout2d::zeros(10);
    assert_eq!(
        s_edgeless.run(&mut l_edgeless).unwrap(),
        SolverStatus::Converged
    );
    assert!(l_edgeless.is_all_finite());
    assert_strictly_distinct(&l_edgeless);
}

#[test]
fn test_sfc_hilbert_morton_zero_colocation_and_step_run_parity() {
    // Two cliques connected by a weak bridge + disconnected isolated node.
    let mut edges = Vec::new();
    let mut weights = Vec::new();
    for i in 0..5 {
        for j in (i + 1)..5 {
            edges.push((i as u32, j as u32));
            weights.push(3.0);
        }
    }
    for i in 5..10 {
        for j in (i + 1)..10 {
            edges.push((i as u32, j as u32));
            weights.push(2.5);
        }
    }
    edges.push((4, 5));
    weights.push(0.1);
    // N = 12 (nodes 10, 11 isolated).
    let g = CsrGraph::from_edges(12, &edges, Some(&weights), false, false).unwrap();

    for &curve in &[SfcCurveKind::Hilbert, SfcCurveKind::Morton] {
        let cfg = SfcConfig {
            curve,
            scale: 50.0,
            resolution_order: None,
            seed: 42,
        };
        // f64 step() == run()
        let mut s1 = SfcLayoutSolver::<f64>::new(&g, cfg).unwrap();
        let mut s2 = SfcLayoutSolver::<f64>::new(&g, cfg).unwrap();
        let mut l1 = Layout2d::zeros(12);
        let mut l2 = Layout2d::zeros(12);
        assert_eq!(s1.step(&mut l1).unwrap(), SolverStatus::Converged);
        assert_eq!(s2.run(&mut l2).unwrap(), SolverStatus::Converged);
        assert_strictly_distinct(&l1);
        assert_eq!(
            l1.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l2.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            l1.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l2.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        // Permutation check on vertex_order.
        let mut sorted_order = s1.vertex_order().to_vec();
        sorted_order.sort_unstable();
        assert_eq!(sorted_order, (0..12).collect::<Vec<u32>>());

        // f32 step() == run() & zero co-location.
        let mut s32_a = SfcLayoutSolver::<f32>::new(&g, cfg).unwrap();
        let mut s32_b = SfcLayoutSolver::<f32>::new(&g, cfg).unwrap();
        let mut l32_a = Layout2d::<f32>::zeros(12);
        let mut l32_b = Layout2d::<f32>::zeros(12);
        assert_eq!(s32_a.step(&mut l32_a).unwrap(), SolverStatus::Converged);
        assert_eq!(s32_b.run(&mut l32_b).unwrap(), SolverStatus::Converged);
        assert_strictly_distinct(&l32_a);
        assert_eq!(
            l32_a.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l32_b.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            l32_a.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l32_b.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    // Directed graph with reciprocal edges and zero-weight edge.
    let g_dir = CsrGraph::from_edges(
        6,
        &[(0, 1), (1, 0), (1, 2), (2, 0), (3, 4), (4, 5), (0, 3)],
        Some(&[1.0, 2.0, 1.5, 1.0, 2.0, 2.0, 0.0]),
        true,
        false,
    )
    .unwrap();
    let mut s_dir = SfcLayoutSolver::<f64>::new(&g_dir, SfcConfig::default()).unwrap();
    let mut l_dir = Layout2d::zeros(6);
    assert_eq!(s_dir.run(&mut l_dir).unwrap(), SolverStatus::Converged);
    assert!(l_dir.is_all_finite());
    assert_strictly_distinct(&l_dir);
}
