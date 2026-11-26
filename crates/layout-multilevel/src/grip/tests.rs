use super::*;

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

#[test]
fn test_grip_config_validation() {
    let default_cfg = GripConfig::default();
    assert!(default_cfg.validate().is_ok());

    assert!(
        GripConfig {
            rounds_min: 10,
            rounds_max: 5,
            ..default_cfg
        }
        .validate()
        .is_err()
    );
    assert!(
        GripConfig {
            local_neighborhood_size: 0,
            ..default_cfg
        }
        .validate()
        .is_err()
    );
    assert!(
        GripConfig {
            edge_length: 0.0,
            ..default_cfg
        }
        .validate()
        .is_err()
    );
    assert!(
        GripConfig {
            repulsion_scale: f64::NAN,
            ..default_cfg
        }
        .validate()
        .is_err()
    );
    assert!(
        GripConfig {
            temp_r: -0.1,
            ..default_cfg
        }
        .validate()
        .is_err()
    );
    assert!(
        GripConfig {
            temp_s: f64::INFINITY,
            ..default_cfg
        }
        .validate()
        .is_err()
    );
}

#[test]
fn test_grip_step_run_bit_exact_f64_and_f32() {
    let g = grid_graph(5, 5);
    let cfg = GripConfig::default();

    // f64 parity
    let mut s_run = GripSolver::<f64>::new(&g, cfg).unwrap();
    let mut s_step = GripSolver::<f64>::new(&g, cfg).unwrap();
    assert!(s_run.num_levels() >= 2);
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
    assert!(l_run.is_all_finite());

    // f32 parity
    let mut s32_run = GripSolver::<f32>::new(&g, cfg).unwrap();
    let mut s32_step = GripSolver::<f32>::new(&g, cfg).unwrap();
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
    assert!(l32_run.is_all_finite());
}

#[test]
fn test_grip_edge_cases_n0_n1_n2_edgeless_disconnected_directed() {
    let cfg = GripConfig::default();

    // N = 0
    let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let mut s0 = GripSolver::<f64>::new(&g0, cfg).unwrap();
    let mut l0 = Layout2d::<f64>::zeros(0);
    assert_eq!(s0.run(&mut l0).unwrap(), SolverStatus::Converged);

    // N = 1
    let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let mut s1 = GripSolver::<f64>::new(&g1, cfg).unwrap();
    let mut l1 = Layout2d::<f64>::from_xy(vec![3.5], vec![-1.2]).unwrap();
    assert_eq!(s1.run(&mut l1).unwrap(), SolverStatus::Converged);
    assert_eq!(l1.x()[0], 3.5);

    // N = 2 connected
    let g2 = path_graph(2);
    let mut s2 = GripSolver::<f64>::new(&g2, cfg).unwrap();
    let mut l2 = Layout2d::<f64>::zeros(2);
    assert_eq!(s2.run(&mut l2).unwrap(), SolverStatus::Completed);
    let d2 = (l2.x()[0] - l2.x()[1]).hypot(l2.y()[0] - l2.y()[1]);
    assert!((d2 - cfg.edge_length).abs() < 1e-12);

    // N = 3 path (collinear triangle inequality test)
    let g3 = path_graph(3);
    let mut s3 = GripSolver::<f64>::new(
        &g3,
        GripConfig {
            local_neighborhood_size: 1,
            ..cfg
        },
    )
    .unwrap();
    let mut l3 = Layout2d::<f64>::zeros(3);
    assert_eq!(s3.run(&mut l3).unwrap(), SolverStatus::Completed);
    assert!(l3.is_all_finite());

    // Edgeless graph (N = 5) packed without overlap
    let ge = CsrGraph::<f64>::from_edges(5, &[], None, false, false).unwrap();
    let mut se = GripSolver::<f64>::new(&ge, cfg).unwrap();
    let mut le = Layout2d::<f64>::zeros(5);
    assert_eq!(se.run(&mut le).unwrap(), SolverStatus::Completed);
    assert!(le.is_all_finite());
    for i in 0..5 {
        for j in (i + 1)..5 {
            let d = (le.x()[i] - le.x()[j]).hypot(le.y()[i] - le.y()[j]);
            assert!(d > 0.1, "isolated nodes {i} and {j} collapsed: d={d}");
        }
    }

    // Disconnected multi-component (2 paths of size 6 + 1 isolated node)
    let mut edges = Vec::new();
    for i in 0..5 {
        edges.push((i, i + 1));
    }
    for i in 6..11 {
        edges.push((i, i + 1));
    }
    let gd = CsrGraph::<f64>::from_edges(13, &edges, None, false, false).unwrap();
    let mut sd = GripSolver::<f64>::new(&gd, cfg).unwrap();
    let mut ld = Layout2d::<f64>::zeros(13);
    assert_eq!(sd.run(&mut ld).unwrap(), SolverStatus::Completed);
    assert!(ld.is_all_finite());

    // Directed graph with reciprocal edges
    let gdir = CsrGraph::<f64>::from_edges(
        4,
        &[(0, 1), (1, 0), (1, 2), (2, 3), (3, 0)],
        Some(&[1.0, 2.0, 1.5, 1.0, 1.0]),
        true,
        false,
    )
    .unwrap();
    let mut sdir = GripSolver::<f64>::new(&gdir, cfg).unwrap();
    let mut ldir = Layout2d::<f64>::zeros(4);
    assert_eq!(sdir.run(&mut ldir).unwrap(), SolverStatus::Completed);
    assert!(ldir.is_all_finite());

    // Non-finite input and dimension mismatch guards (including empty and singleton graphs)
    let mut s0_bad = GripSolver::<f64>::new(&g0, cfg).unwrap();
    let mut l0_wrong = Layout2d::<f64>::zeros(1);
    assert!(s0_bad.step(&mut l0_wrong).is_err());

    let mut l1_nan = Layout2d::from_xy(vec![f64::NAN], vec![0.0]).unwrap();
    let mut s1_bad = GripSolver::<f64>::new(&g1, cfg).unwrap();
    assert!(s1_bad.step(&mut l1_nan).is_err());

    let mut l_bad = Layout2d::<f64>::from_xy(vec![f64::NAN; 4], vec![0.0; 4]).unwrap();
    let mut sdir2 = GripSolver::<f64>::new(&gdir, cfg).unwrap();
    assert!(sdir2.step(&mut l_bad).is_err());
    let mut l_wrong = Layout2d::<f64>::zeros(3);
    assert!(sdir2.step(&mut l_wrong).is_err());
}
