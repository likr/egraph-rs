use egraph_core::{CoreError, CsrGraph, IterativeSolver, Layout2d, SolverStatus};

use super::*;

fn cycle_graph(n: usize) -> CsrGraph<f64> {
    let mut edges = Vec::with_capacity(n);
    for i in 0..n {
        edges.push((i as u32, ((i + 1) % n) as u32));
    }
    CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
}

#[test]
fn test_gem_config_validation_and_builders() {
    let def = GemConfig::new();
    assert!(def.validate().is_ok());

    assert!(def.clone().with_edge_length(0.0).validate().is_err());
    assert!(def.clone().with_gravity(-0.1).validate().is_err());
    assert!(def.clone().with_t_init(-1.0).validate().is_err());
    assert!(def.clone().with_t_max(0.0).validate().is_err());
    assert!(def.clone().with_t_min(f64::NAN).validate().is_err());
    assert!(def.clone().with_oscillation_angle(0.0).validate().is_err());
    assert!(def.clone().with_rotation_angle(4.0).validate().is_err());
    assert!(
        def.clone()
            .with_oscillation_sensitivity(-0.5)
            .validate()
            .is_err()
    );
    assert!(
        def.clone()
            .with_rotation_sensitivity(Some(-0.1))
            .validate()
            .is_err()
    );
    assert!(def.with_perturbation(f64::INFINITY).validate().is_err());
}

#[test]
fn test_gem_step_run_bit_exact_f64_f32_and_unweighted() {
    let g = cycle_graph(12);
    let default_solver = GemSolver::<f64>::new(&g, GemConfig::new()).unwrap();
    assert_eq!(default_solver.total_iterations(), Some(48));
    let large_g = cycle_graph(150);
    let large_solver = GemSolver::<f64>::new(&large_g, GemConfig::new()).unwrap();
    assert_eq!(large_solver.total_iterations(), Some(400));

    let cfg = GemConfig::new().with_max_iterations(Some(25));

    // f64 parity
    let mut s_run = GemSolver::<f64>::new(&g, cfg.clone()).unwrap();
    let mut s_step = GemSolver::<f64>::new(&g, cfg.clone()).unwrap();
    let mut l_run = Layout2d::<f64>::zeros(g.num_nodes());
    let mut l_step = Layout2d::<f64>::zeros(g.num_nodes());

    let st_run = s_run.run(&mut l_run).unwrap();
    assert!(st_run.is_finished());
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
    assert!(l_run.is_all_finite());

    let mut min_dist = f64::INFINITY;
    for i in 0..g.num_nodes() {
        for j in (i + 1)..g.num_nodes() {
            let d = (l_run.x()[i] - l_run.x()[j]).hypot(l_run.y()[i] - l_run.y()[j]);
            min_dist = min_dist.min(d);
        }
    }
    assert!(min_dist > 1.0);

    // f32 parity + unweighted CsrGraph<()>
    let unit_g =
        CsrGraph::<()>::from_unweighted_csr(g.indptr(), g.indices(), false, false).unwrap();
    let mut s32_run = GemSolver::<f32>::new_unweighted(&unit_g, cfg.clone()).unwrap();
    let mut s32_step = GemSolver::<f32>::new_unweighted(&unit_g, cfg).unwrap();
    let mut l32_run = Layout2d::<f32>::zeros(g.num_nodes());
    let mut l32_step = Layout2d::<f32>::zeros(g.num_nodes());
    s32_run.run(&mut l32_run).unwrap();
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
    assert!(l32_run.is_all_finite());
}

#[test]
fn test_gem_edge_cases_n0_n1_edgeless_disconnected_weighted_directed_and_v6_contract() {
    let cfg = GemConfig::new().with_max_iterations(Some(15));

    let g0 = CsrGraph::<()>::from_unweighted_edges(0, &[], false, false).unwrap();
    let mut s0 = GemSolver::<f64>::new_unweighted(&g0, cfg.clone()).unwrap();
    let mut l0 = Layout2d::<f64>::zeros(0);
    assert_eq!(s0.run(&mut l0).unwrap(), SolverStatus::Converged);

    let g1 = CsrGraph::<()>::from_unweighted_edges(1, &[], false, false).unwrap();
    let mut s1 = GemSolver::<f64>::new_unweighted(&g1, cfg.clone()).unwrap();
    let mut l1 = Layout2d::<f64>::from_xy(vec![2.0], vec![-3.0]).unwrap();
    assert_eq!(s1.run(&mut l1).unwrap(), SolverStatus::Converged);
    assert_eq!(l1.x()[0], 2.0);
    assert_eq!(l1.y()[0], -3.0);

    let g2 = cycle_graph(4);
    let s_zero = GemSolver::<f64>::new(&g2, cfg.clone().with_max_iterations(Some(0))).unwrap();
    assert_eq!(s_zero.status(), SolverStatus::Completed);
    assert_eq!(s_zero.total_iterations(), Some(0));

    let gd = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)],
        None,
        false,
        false,
    )
    .unwrap();
    let mut sr = GemSolver::<f64>::new(
        &gd,
        GemConfig::new()
            .with_rotation_sensitivity(Some(5.0))
            .with_oscillation_sensitivity(2.0)
            .with_max_iterations(Some(30)),
    )
    .unwrap();
    let mut lr = Layout2d::<f64>::zeros(6);
    sr.run(&mut lr).unwrap();
    assert!(lr.is_all_finite());
    assert!(sr.temp.iter().all(|&t| t.is_finite() && t >= 0.0));

    let gw = CsrGraph::from_edges(
        4,
        &[(0, 1), (1, 0), (1, 2), (2, 3), (3, 0)],
        Some(&[1.0, 3.0, 2.0, 0.5, 1.5]),
        true,
        false,
    )
    .unwrap();
    let mut bad = Layout2d::<f64>::from_xy(vec![f64::NAN; 4], vec![0.0; 4]).unwrap();
    let mut sw2 = GemSolver::<f64>::new(&gw, cfg).unwrap();
    assert!(matches!(
        sw2.run_steps(&mut bad, 0),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert_eq!(sw2.status(), SolverStatus::Initialized);
    let mut wrong = Layout2d::<f64>::zeros(3);
    assert!(matches!(
        sw2.step(&mut wrong),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert_eq!(sw2.status(), SolverStatus::Initialized);
}
