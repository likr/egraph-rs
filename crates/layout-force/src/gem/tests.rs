use egraph_core::{CsrGraph, IterativeSolver, Layout2d, SolverStatus};

use super::*;

fn cycle_graph(n: usize) -> CsrGraph<f64> {
    let mut edges = Vec::with_capacity(n);
    for i in 0..n {
        edges.push((i as u32, ((i + 1) % n) as u32));
    }
    CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
}

#[test]
fn test_gem_config_validation() {
    let def = GemConfig::default();
    assert!(def.validate().is_ok());

    assert!(
        GemConfig {
            edge_length: 0.0,
            ..def
        }
        .validate()
        .is_err()
    );
    assert!(
        GemConfig {
            gravity: -0.1,
            ..def
        }
        .validate()
        .is_err()
    );
    assert!(
        GemConfig {
            t_init: -1.0,
            ..def
        }
        .validate()
        .is_err()
    );
    assert!(GemConfig { t_max: 0.0, ..def }.validate().is_err());
    assert!(
        GemConfig {
            t_min: f64::NAN,
            ..def
        }
        .validate()
        .is_err()
    );
    assert!(
        GemConfig {
            oscillation_angle: 0.0,
            ..def
        }
        .validate()
        .is_err()
    );
    assert!(
        GemConfig {
            rotation_angle: 4.0,
            ..def
        }
        .validate()
        .is_err()
    );
    assert!(
        GemConfig {
            oscillation_sensitivity: -0.5,
            ..def
        }
        .validate()
        .is_err()
    );
    assert!(
        GemConfig {
            rotation_sensitivity: Some(-0.1),
            ..def
        }
        .validate()
        .is_err()
    );
    assert!(
        GemConfig {
            perturbation: f64::INFINITY,
            ..def
        }
        .validate()
        .is_err()
    );
}

#[test]
fn test_gem_step_run_bit_exact_f64_and_f32() {
    let g = cycle_graph(12);
    let default_solver = GemSolver::<f64>::new(&g, GemConfig::default()).unwrap();
    assert_eq!(default_solver.total_iterations(), 48);
    let large_g = cycle_graph(150);
    let large_solver = GemSolver::<f64>::new(&large_g, GemConfig::default()).unwrap();
    assert_eq!(large_solver.total_iterations(), 400);

    let cfg = GemConfig {
        max_iterations: Some(25),
        ..GemConfig::default()
    };

    // f64 parity
    let mut s_run = GemSolver::<f64>::new(&g, cfg).unwrap();
    let mut s_step = GemSolver::<f64>::new(&g, cfg).unwrap();
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

    // Ensure collapsed zeros layout was scattered and separated.
    let mut min_dist = f64::INFINITY;
    for i in 0..g.num_nodes() {
        for j in (i + 1)..g.num_nodes() {
            let d = (l_run.x()[i] - l_run.x()[j]).hypot(l_run.y()[i] - l_run.y()[j]);
            min_dist = min_dist.min(d);
        }
    }
    assert!(min_dist > 1.0);

    // f32 parity
    let mut s32_run = GemSolver::<f32>::new(&g, cfg).unwrap();
    let mut s32_step = GemSolver::<f32>::new(&g, cfg).unwrap();
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
fn test_gem_edge_cases_n0_n1_edgeless_disconnected_weighted_directed_and_rotation_clamp() {
    let cfg = GemConfig {
        max_iterations: Some(15),
        ..GemConfig::default()
    };

    // N = 0
    let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let mut s0 = GemSolver::<f64>::new(&g0, cfg).unwrap();
    let mut l0 = Layout2d::<f64>::zeros(0);
    assert_eq!(s0.run(&mut l0).unwrap(), SolverStatus::Converged);

    // N = 1
    let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let mut s1 = GemSolver::<f64>::new(&g1, cfg).unwrap();
    let mut l1 = Layout2d::<f64>::from_xy(vec![2.0], vec![-3.0]).unwrap();
    assert_eq!(s1.run(&mut l1).unwrap(), SolverStatus::Converged);
    assert_eq!(l1.x()[0], 2.0);
    assert_eq!(l1.y()[0], -3.0);

    // max_iterations = 0 -> Completed immediately
    let g2 = cycle_graph(4);
    let s_zero = GemSolver::<f64>::new(
        &g2,
        GemConfig {
            max_iterations: Some(0),
            ..cfg
        },
    )
    .unwrap();
    assert_eq!(s_zero.status(), SolverStatus::Completed);

    // Edgeless graph
    let ge = CsrGraph::<f64>::from_edges(6, &[], None, false, false).unwrap();
    let mut se = GemSolver::<f64>::new(&ge, cfg).unwrap();
    let mut le = Layout2d::<f64>::zeros(6);
    se.run(&mut le).unwrap();
    assert!(le.is_all_finite());

    // Disconnected graph
    let gd = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)],
        None,
        false,
        false,
    )
    .unwrap();
    let mut sd = GemSolver::<f64>::new(&gd, cfg).unwrap();
    let mut ld = Layout2d::<f64>::zeros(6);
    sd.run(&mut ld).unwrap();
    assert!(ld.is_all_finite());

    // Weighted directed graph with reciprocal edges
    let gw = CsrGraph::<f64>::from_edges(
        4,
        &[(0, 1), (1, 0), (1, 2), (2, 3), (3, 0)],
        Some(&[1.0, 3.0, 2.0, 0.5, 1.5]),
        true,
        false,
    )
    .unwrap();
    let mut sw = GemSolver::<f64>::new(&gw, cfg).unwrap();
    let mut lw = Layout2d::<f64>::zeros(4);
    sw.run(&mut lw).unwrap();
    assert!(lw.is_all_finite());

    // Aggressive rotation sensitivity (tests |v.d| >= 1 clamp keeping temperature >= 0)
    let mut sr = GemSolver::<f64>::new(
        &gd,
        GemConfig {
            rotation_sensitivity: Some(5.0),
            oscillation_sensitivity: 2.0,
            max_iterations: Some(30),
            ..GemConfig::default()
        },
    )
    .unwrap();
    let mut lr = Layout2d::<f64>::zeros(6);
    sr.run(&mut lr).unwrap();
    assert!(lr.is_all_finite());
    assert!(sr.temp.iter().all(|&t| t.is_finite() && t >= 0.0));

    // Non-finite and dimension mismatch guards
    let mut bad = Layout2d::<f64>::from_xy(vec![f64::NAN; 4], vec![0.0; 4]).unwrap();
    let mut sw2 = GemSolver::<f64>::new(&gw, cfg).unwrap();
    assert!(sw2.step(&mut bad).is_err());
    let mut wrong = Layout2d::<f64>::zeros(3);
    assert!(sw2.step(&mut wrong).is_err());
}
