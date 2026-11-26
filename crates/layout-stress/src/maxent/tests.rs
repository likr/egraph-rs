use egraph_core::{CoreError, CsrGraph, IterativeSolver, Layout2d, SolverStatus};

use super::{MaxentStressConfig, MaxentStressSolver};

fn grid_graph(rows: usize, cols: usize) -> CsrGraph<f64> {
    let n = rows * cols;
    let mut edges = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let u = (r * cols + c) as u32;
            if c + 1 < cols {
                edges.push((u, u + 1));
            }
            if r + 1 < rows {
                edges.push((u, u + cols as u32));
            }
        }
    }
    CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
}

fn path_graph(n: usize) -> CsrGraph<f64> {
    let edges: Vec<(u32, u32)> = (0..n.saturating_sub(1))
        .map(|i| (i as u32, (i + 1) as u32))
        .collect();
    CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
}

fn star_graph(leaves: usize) -> CsrGraph<f64> {
    let n = leaves + 1;
    let edges: Vec<(u32, u32)> = (1..n).map(|i| (0u32, i as u32)).collect();
    CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
}

#[test]
fn test_maxent_config_validation() {
    assert!(MaxentStressConfig::default().validate().is_ok());
    assert!(matches!(
        MaxentStressConfig {
            q: Some(-2.0),
            ..Default::default()
        }
        .validate(),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        MaxentStressConfig {
            q: Some(f64::NAN),
            ..Default::default()
        }
        .validate(),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        MaxentStressConfig {
            alpha_decay: 0.0,
            ..Default::default()
        }
        .validate(),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        MaxentStressConfig {
            alpha_decay: 1.5,
            ..Default::default()
        }
        .validate(),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        MaxentStressConfig {
            cg_max_iters: 0,
            ..Default::default()
        }
        .validate(),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        MaxentStressConfig {
            eps_sq: 0.0,
            ..Default::default()
        }
        .validate(),
        Err(CoreError::InvalidInput(_))
    ));
}

#[test]
fn test_maxent_adaptive_q_and_annealing() {
    // Grid 4x4 has 4 degree-2 corners and 0 degree-1 vertices -> q = 0.0.
    let grid = grid_graph(4, 4);
    let solver_grid = MaxentStressSolver::<f64>::new(&grid, MaxentStressConfig::default()).unwrap();
    assert_eq!(solver_grid.effective_q(), 0.0);

    // Star graph has > 30% degree-1 leaves -> q = 0.8.
    let star = star_graph(6);
    let solver_star = MaxentStressSolver::<f64>::new(&star, MaxentStressConfig::default()).unwrap();
    assert_eq!(solver_star.effective_q(), 0.8);

    // Explicit q overrides automatic selection.
    let solver_override = MaxentStressSolver::<f64>::new(
        &star,
        MaxentStressConfig {
            q: Some(-0.5),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(solver_override.effective_q(), -0.5);

    // Annealing alpha decays cleanly from 1.0 -> 0.3 -> 0.09 -> 0.027 -> 0.0081.
    let mut s = MaxentStressSolver::<f64>::new(
        &grid,
        MaxentStressConfig {
            stages: 5,
            steps_per_stage: 2,
            tol: 0.0,
            ..Default::default()
        },
    )
    .unwrap();
    let mut layout = Layout2d::zeros(grid.num_nodes());
    assert_eq!(s.current_alpha(), 1.0);
    s.run_steps(&mut layout, 2).unwrap();
    assert!((s.current_alpha() - 0.3).abs() < 1e-12);
    s.run(&mut layout).unwrap();
    assert!(s.current_alpha() >= 0.008);
    assert!(layout.is_all_finite());
}

#[test]
fn test_maxent_step_run_bit_exact_f64_and_f32() {
    let g = grid_graph(3, 4);

    // f64 bit-exact step() == run(), both Barnes-Hut and exact O(N^2).
    for &use_bh in &[true, false] {
        let cfg = MaxentStressConfig {
            use_barnes_hut: use_bh,
            stages: 3,
            steps_per_stage: 10,
            ..Default::default()
        };
        let mut s1 = MaxentStressSolver::<f64>::new(&g, cfg).unwrap();
        let mut s2 = MaxentStressSolver::<f64>::new(&g, cfg).unwrap();
        let mut l1 = Layout2d::zeros(g.num_nodes());
        let mut l2 = Layout2d::zeros(g.num_nodes());

        let st1 = s1.run(&mut l1).unwrap();
        let mut st2 = SolverStatus::Initialized;
        while !st2.is_finished() {
            st2 = s2.step(&mut l2).unwrap();
        }
        assert_eq!(st1, st2);
        assert_eq!(s1.iteration(), s2.iteration());
        assert_eq!(
            l1.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l2.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            l1.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l2.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    // f32 bit-exact step() == run().
    let cfg32 = MaxentStressConfig {
        stages: 3,
        steps_per_stage: 8,
        ..Default::default()
    };
    let mut s1 = MaxentStressSolver::<f32>::new(&g, cfg32).unwrap();
    let mut s2 = MaxentStressSolver::<f32>::new(&g, cfg32).unwrap();
    let mut l1 = Layout2d::<f32>::zeros(g.num_nodes());
    let mut l2 = Layout2d::<f32>::zeros(g.num_nodes());
    let st1 = s1.run(&mut l1).unwrap();
    let mut st2 = SolverStatus::Initialized;
    while !st2.is_finished() {
        st2 = s2.step(&mut l2).unwrap();
    }
    assert_eq!(st1, st2);
    assert!(l1.is_all_finite());
    assert_eq!(
        l1.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        l2.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        l1.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        l2.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn test_maxent_edge_cases_disconnected_isolated_weighted_directed() {
    // N = 0 and N = 1.
    let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let mut s0 = MaxentStressSolver::<f64>::new(&g0, MaxentStressConfig::default()).unwrap();
    let mut l0 = Layout2d::zeros(0);
    assert_eq!(s0.step(&mut l0).unwrap(), SolverStatus::Converged);

    let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let mut s1 = MaxentStressSolver::<f64>::new(&g1, MaxentStressConfig::default()).unwrap();
    let mut l1 = Layout2d::zeros(1);
    assert_eq!(s1.step(&mut l1).unwrap(), SolverStatus::Converged);

    // Edgeless graph (N = 4).
    let g_edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    let mut s_edgeless =
        MaxentStressSolver::<f64>::new(&g_edgeless, MaxentStressConfig::default()).unwrap();
    let mut l_edgeless = Layout2d::zeros(4);
    s_edgeless.run(&mut l_edgeless).unwrap();
    assert!(l_edgeless.is_all_finite());

    // Disconnected graph: two 3-paths + 2 isolated vertices (N = 8).
    let g_disc =
        CsrGraph::<f64>::from_edges(8, &[(0, 1), (1, 2), (3, 4), (4, 5)], None, false, false)
            .unwrap();
    let mut s_disc =
        MaxentStressSolver::<f64>::new(&g_disc, MaxentStressConfig::default()).unwrap();
    let mut l_disc = Layout2d::from_xy(
        vec![0.0, 1.0, 2.0, 5.0, 6.0, 7.0, -3.0, -4.0],
        vec![0.0, 0.5, 0.0, 2.0, 2.5, 2.0, 8.0, -8.0],
    )
    .unwrap();
    let st_disc = s_disc.run(&mut l_disc).unwrap();
    assert!(st_disc.is_finished());
    assert!(l_disc.is_all_finite());
    // Isolated vertices (6, 7) must remain untouched.
    assert_eq!(l_disc.x()[6], -3.0);
    assert_eq!(l_disc.y()[6], 8.0);
    assert_eq!(l_disc.x()[7], -4.0);
    assert_eq!(l_disc.y()[7], -8.0);

    // Weighted + directed graph with reciprocal edges.
    let g_dir = CsrGraph::from_edges(
        4,
        &[(0, 1), (1, 0), (1, 2), (2, 3), (3, 0)],
        Some(&[2.0, 1.5, 1.0, 3.0, 2.0]),
        true,
        false,
    )
    .unwrap();
    let mut s_dir = MaxentStressSolver::<f64>::new(&g_dir, MaxentStressConfig::default()).unwrap();
    let mut l_dir = Layout2d::zeros(4);
    s_dir.run(&mut l_dir).unwrap();
    assert!(l_dir.is_all_finite());

    // Caller-provided non-collapsed layout is refined rather than overwritten by PivotMDS.
    let g_path = path_graph(5);
    let mut s_custom = MaxentStressSolver::<f64>::new(
        &g_path,
        MaxentStressConfig {
            stages: 1,
            steps_per_stage: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let mut l_custom = Layout2d::from_xy(
        vec![10.0, 20.0, 30.0, 40.0, 50.0],
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
    )
    .unwrap();
    assert_eq!(s_custom.total_iterations(), 0);
    s_custom.run(&mut l_custom).unwrap();
    assert_eq!(l_custom.x(), &[10.0, 20.0, 30.0, 40.0, 50.0]);
}
