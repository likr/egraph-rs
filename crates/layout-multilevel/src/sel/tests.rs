use super::*;

fn two_cliques_bridge() -> CsrGraph<f64> {
    // Two weighted K_6 cliques (intra-clique w=10.0) joined by a weak bridge (5, 6) (w=1.0).
    let mut edges = Vec::new();
    let mut weights = Vec::new();
    for i in 0..6u32 {
        for j in (i + 1)..6u32 {
            edges.push((i, j));
            weights.push(10.0);
        }
    }
    for i in 6..12u32 {
        for j in (i + 1)..12u32 {
            edges.push((i, j));
            weights.push(10.0);
        }
    }
    edges.push((5, 6));
    weights.push(1.0);
    CsrGraph::from_edges(12, &edges, Some(&weights), false, false).unwrap()
}

#[test]
fn test_sel_config_validation() {
    let def = SelConfig::default();
    assert!(def.validate().is_ok());
    assert!(
        SelConfig {
            tree_depth: 0,
            ..def
        }
        .validate()
        .is_err()
    );
    assert!(
        SelConfig {
            pq_max_pops: 0,
            ..def
        }
        .validate()
        .is_err()
    );
}

#[test]
fn test_sel_five_mappings_and_entropy_attenuation() {
    let g = two_cliques_bridge();
    for mapping in [
        SelWeightMapping::Sigmoid,
        SelWeightMapping::Linear,
        SelWeightMapping::Tanh,
        SelWeightMapping::CustomLogistic,
        SelWeightMapping::Binary,
    ] {
        let cfg = SelConfig {
            mapping,
            iterations: 40,
            ..SelConfig::default()
        };
        let mut solver = SelLayoutSolver::<f64>::new(&g, cfg).unwrap();
        assert_eq!(solver.modulated_weights().len(), 31);
        assert!(solver.se_target().is_finite() && solver.se_target() >= 0.0);
        assert!(
            solver
                .modulated_weights()
                .iter()
                .all(|&w| w.is_finite() && w >= 0.0)
        );
        // Weak inter-clique bridge edge (5, 6) [idx 15] is cut at root (0.0 <= se_target)
        // so it must be attenuated to 0.0 while dense intra-clique edges stay > 0.
        assert_eq!(solver.modulated_weights()[15], 0.0);
        assert!(solver.modulated_weights().iter().any(|&w| w > 0.0));
        if mapping != SelWeightMapping::CustomLogistic || solver.se_target() > 0.0 {
            assert!(solver.modulated_weights()[0] > 0.0);
        }

        let mut layout = Layout2d::<f64>::zeros(g.num_nodes());
        assert_eq!(solver.run(&mut layout).unwrap(), SolverStatus::Completed);
        assert!(layout.is_all_finite());
    }
}

#[test]
fn test_sel_backends_and_step_run_bit_exact_f64_f32() {
    let g = two_cliques_bridge();
    for backend in [SelLayoutBackend::Fr, SelLayoutBackend::TFdp] {
        let cfg = SelConfig {
            layout_backend: backend,
            iterations: 30,
            ..SelConfig::default()
        };

        // f64 parity
        let mut s_run = SelLayoutSolver::<f64>::new(&g, cfg).unwrap();
        let mut s_step = SelLayoutSolver::<f64>::new(&g, cfg).unwrap();
        let mut l_run = Layout2d::<f64>::zeros(g.num_nodes());
        let mut l_step = Layout2d::<f64>::zeros(g.num_nodes());
        assert_eq!(s_run.run(&mut l_run).unwrap(), SolverStatus::Completed);
        while !s_step.status().is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        assert_eq!(s_step.iteration(), s_run.iteration());
        assert_eq!(
            l_run.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l_step.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            l_run.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l_step.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        // f32 parity
        let mut s32_run = SelLayoutSolver::<f32>::new(&g, cfg).unwrap();
        let mut s32_step = SelLayoutSolver::<f32>::new(&g, cfg).unwrap();
        let mut l32_run = Layout2d::<f32>::zeros(g.num_nodes());
        let mut l32_step = Layout2d::<f32>::zeros(g.num_nodes());
        assert_eq!(s32_run.run(&mut l32_run).unwrap(), SolverStatus::Completed);
        while !s32_step.status().is_finished() {
            s32_step.step(&mut l32_step).unwrap();
        }
        assert_eq!(
            l32_run.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l32_step.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            l32_run.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l32_step.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}

#[test]
fn test_sel_edge_cases_n0_n1_edgeless_disconnected_directed_depth1() {
    let cfg = SelConfig {
        iterations: 20,
        ..SelConfig::default()
    };

    // N = 0
    let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let mut s0 = SelLayoutSolver::<f64>::new(&g0, cfg).unwrap();
    assert!(s0.modulated_weights().is_empty());
    let mut l0 = Layout2d::<f64>::zeros(0);
    assert_eq!(s0.run(&mut l0).unwrap(), SolverStatus::Converged);

    // N = 1
    let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let mut s1 = SelLayoutSolver::<f64>::new(&g1, cfg).unwrap();
    let mut l1 = Layout2d::<f64>::from_xy(vec![1.5], vec![-2.5]).unwrap();
    assert_eq!(s1.run(&mut l1).unwrap(), SolverStatus::Converged);
    assert_eq!(l1.x()[0], 1.5);

    // Edgeless graph
    let ge = CsrGraph::<f64>::from_edges(5, &[], None, false, false).unwrap();
    let mut se = SelLayoutSolver::<f64>::new(&ge, cfg).unwrap();
    let mut le = Layout2d::<f64>::zeros(5);
    assert_eq!(se.run(&mut le).unwrap(), SolverStatus::Completed);
    assert!(le.is_all_finite());

    // Disconnected graph (two triangles + 2 isolated nodes)
    let gd = CsrGraph::<f64>::from_edges(
        8,
        &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)],
        None,
        false,
        false,
    )
    .unwrap();
    let mut sd = SelLayoutSolver::<f64>::new(&gd, cfg).unwrap();
    let mut ld = Layout2d::<f64>::zeros(8);
    assert_eq!(sd.run(&mut ld).unwrap(), SolverStatus::Completed);
    assert!(ld.is_all_finite());

    // Directed graph with reciprocal weights & tree_depth = 1
    let gdir = CsrGraph::<f64>::from_edges(
        4,
        &[(0, 1), (1, 0), (1, 2), (2, 3), (3, 0)],
        Some(&[1.0, 2.0, 1.5, 1.0, 1.0]),
        true,
        false,
    )
    .unwrap();
    let mut sdir = SelLayoutSolver::<f64>::new(
        &gdir,
        SelConfig {
            tree_depth: 1,
            ..cfg
        },
    )
    .unwrap();
    let mut ldir = Layout2d::<f64>::zeros(4);
    assert_eq!(sdir.run(&mut ldir).unwrap(), SolverStatus::Completed);
    assert!(ldir.is_all_finite());

    // Nested backend config overrides
    let fr_override = SelConfig {
        layout_backend: SelLayoutBackend::Fr,
        fr_config: Some(FruchtermanReingoldConfig {
            iterations: 12,
            seed: 9,
            ..Default::default()
        }),
        ..SelConfig::default()
    };
    let mut sfr = SelLayoutSolver::<f64>::new(&gdir, fr_override).unwrap();
    assert_eq!(sfr.total_iterations(), 12);
    let mut lfr = Layout2d::<f64>::zeros(4);
    assert_eq!(sfr.run(&mut lfr).unwrap(), SolverStatus::Completed);

    let tfdp_override = SelConfig {
        layout_backend: SelLayoutBackend::TFdp,
        t_fdp_config: Some(TFdpConfig {
            iterations: 8,
            seed: 9,
            ..Default::default()
        }),
        ..SelConfig::default()
    };
    let mut stf = SelLayoutSolver::<f64>::new(&gdir, tfdp_override).unwrap();
    assert_eq!(stf.total_iterations(), 8);
    let mut ltf = Layout2d::<f64>::zeros(4);
    assert_eq!(stf.run(&mut ltf).unwrap(), SolverStatus::Completed);

    // Non-finite and dimension mismatch guards
    let mut bad = Layout2d::<f64>::from_xy(vec![f64::NAN; 4], vec![0.0; 4]).unwrap();
    let mut sdir2 = SelLayoutSolver::<f64>::new(&gdir, cfg).unwrap();
    assert!(sdir2.step(&mut bad).is_err());
    let mut wrong = Layout2d::<f64>::zeros(3);
    assert!(sdir2.step(&mut wrong).is_err());
}
