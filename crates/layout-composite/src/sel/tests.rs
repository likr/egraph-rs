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
    assert!(def.clone().with_tree_depth(0).validate().is_err());
    assert!(def.clone().with_pq_max_pops(0).validate().is_err());
    assert!(
        def.clone()
            .with_fr_backend(FrBackend::barnes_hut(-0.5))
            .validate()
            .is_err()
    );
    assert!(
        def.with_t_fdp_backend(TFdpBackend2d::BarnesHut(
            egraph_layout_neighbor::BarnesHut::new(-1.0)
        ))
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
        let cfg = SelConfig::new()
            .with_layout_backend(backend)
            .with_iterations(30);

        // f64 parity
        let mut s_run = SelLayoutSolver::<f64>::new(&g, cfg.clone()).unwrap();
        let mut s_step = SelLayoutSolver::<f64>::new(&g, cfg.clone()).unwrap();
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
        let mut s32_run = SelLayoutSolver::<f32>::new(&g, cfg.clone()).unwrap();
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
    let cfg = SelConfig::new().with_iterations(20);

    // N = 0
    let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let mut s0 = SelLayoutSolver::<f64>::new(&g0, cfg.clone()).unwrap();
    assert!(s0.modulated_weights().is_empty());
    let mut l0 = Layout2d::<f64>::zeros(0);
    assert_eq!(s0.run(&mut l0).unwrap(), SolverStatus::Converged);

    // N = 1
    let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let mut s1 = SelLayoutSolver::<f64>::new(&g1, cfg.clone()).unwrap();
    let mut l1 = Layout2d::<f64>::from_xy(vec![1.5], vec![-2.5]).unwrap();
    assert_eq!(s1.run(&mut l1).unwrap(), SolverStatus::Converged);
    assert_eq!(l1.x()[0], 1.5);

    // Edgeless graph
    let ge = CsrGraph::<f64>::from_edges(5, &[], None, false, false).unwrap();
    let mut se = SelLayoutSolver::<f64>::new(&ge, cfg.clone()).unwrap();
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
    let mut sd = SelLayoutSolver::<f64>::new(&gd, cfg.clone()).unwrap();
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
    let mut sdir = SelLayoutSolver::<f64>::new(&gdir, cfg.clone().with_tree_depth(1)).unwrap();
    let mut ldir = Layout2d::<f64>::zeros(4);
    assert_eq!(sdir.run(&mut ldir).unwrap(), SolverStatus::Completed);
    assert!(ldir.is_all_finite());

    // Nested backend config and repulsion backend overrides
    let fr_override = SelConfig::new()
        .with_layout_backend(SelLayoutBackend::Fr)
        .with_fr_config(Some(
            FruchtermanReingoldConfig::new()
                .with_iterations(12)
                .with_seed(9),
        ))
        .with_fr_backend(FrBackend::grid());
    let mut sfr = SelLayoutSolver::<f64>::new(&gdir, fr_override).unwrap();
    assert_eq!(sfr.total_iterations(), Some(12));
    let mut lfr = Layout2d::<f64>::zeros(4);
    assert_eq!(sfr.run(&mut lfr).unwrap(), SolverStatus::Completed);

    let tfdp_override = SelConfig::new()
        .with_layout_backend(SelLayoutBackend::TFdp)
        .with_t_fdp_config(Some(TFdpConfig::new().with_iterations(8).with_seed(9)))
        .with_t_fdp_backend(TFdpBackend2d::BarnesHut(
            egraph_layout_neighbor::BarnesHut::new(0.6),
        ));
    let mut stf = SelLayoutSolver::<f64>::new(&gdir, tfdp_override).unwrap();
    assert_eq!(stf.total_iterations(), Some(8));
    let mut ltf = Layout2d::<f64>::zeros(4);
    assert_eq!(stf.run(&mut ltf).unwrap(), SolverStatus::Completed);

    // Unweighted CsrGraph<()> + zero-step validation
    let g_unw =
        CsrGraph::<()>::from_unweighted_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)], false, false)
            .unwrap();
    let mut s_unw = SelLayoutSolver::<f64>::new_unweighted(&g_unw, cfg.clone()).unwrap();
    let mut l_unw = Layout2d::<f64>::zeros(4);
    assert_eq!(s_unw.run(&mut l_unw).unwrap(), SolverStatus::Completed);
    assert!(l_unw.is_all_finite());

    // Non-finite and dimension mismatch guards (including run_steps(..., 0))
    let mut bad = Layout2d::<f64>::from_xy(vec![f64::NAN; 4], vec![0.0; 4]).unwrap();
    let mut sdir2 = SelLayoutSolver::<f64>::new(&gdir, cfg).unwrap();
    assert!(sdir2.run_steps(&mut bad, 0).is_err());
    assert!(sdir2.step(&mut bad).is_err());
    let mut wrong = Layout2d::<f64>::zeros(3);
    assert!(sdir2.run_steps(&mut wrong, 0).is_err());
    assert!(sdir2.step(&mut wrong).is_err());
}
