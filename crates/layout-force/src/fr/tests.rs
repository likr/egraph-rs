use egraph_core::{
    CoreError, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Scalar, SolverStatus,
};

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

fn seeded_layout<S: Scalar>(n: usize, seed: u64) -> Layout2d<S> {
    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        x.push(S::from_f64(4.0 * (rng.next_f64() - 0.5)));
        y.push(S::from_f64(4.0 * (rng.next_f64() - 0.5)));
    }
    Layout2d::from_xy(x, y).unwrap()
}

#[test]
fn test_fr_config_defaults_and_validation() {
    let cfg = FruchtermanReingoldConfig::default();
    assert_eq!(cfg.iterations, 200);
    assert_eq!(cfg.backend, FrBackend::Exact);
    assert_eq!(cfg.k, None);
    assert_eq!(cfg.area, None);
    assert_eq!(cfg.initial_temperature, None);
    assert_eq!(cfg.gravity, 0.0);
    assert_eq!(cfg.theta, 1.2);
    assert_eq!(cfg.seed, 42);
    cfg.validate().unwrap();

    for bad in [
        FruchtermanReingoldConfig {
            k: Some(0.0),
            ..cfg
        },
        FruchtermanReingoldConfig {
            k: Some(-1.0),
            ..cfg
        },
        FruchtermanReingoldConfig {
            k: Some(f64::NAN),
            ..cfg
        },
        FruchtermanReingoldConfig {
            area: Some(0.0),
            ..cfg
        },
        FruchtermanReingoldConfig {
            area: Some(f64::INFINITY),
            ..cfg
        },
        FruchtermanReingoldConfig {
            initial_temperature: Some(-0.1),
            ..cfg
        },
        FruchtermanReingoldConfig {
            initial_temperature: Some(f64::NAN),
            ..cfg
        },
        FruchtermanReingoldConfig {
            gravity: -0.1,
            ..cfg
        },
        FruchtermanReingoldConfig { theta: -0.1, ..cfg },
    ] {
        assert!(bad.validate().is_err());
    }
}

#[test]
fn test_fr_backends_exact_barnes_hut_and_grid_cutoff() {
    let graph = grid_graph(4, 4);
    let init = seeded_layout::<f64>(16, 17);

    // 1. BarnesHut with theta=0.0 matches Exact to machine precision.
    let mut l_exact = init.clone();
    let mut l_bh0 = init.clone();
    FruchtermanReingoldSolver::new(
        &graph,
        FruchtermanReingoldConfig {
            iterations: 25,
            backend: FrBackend::Exact,
            ..Default::default()
        },
    )
    .unwrap()
    .run(&mut l_exact)
    .unwrap();
    FruchtermanReingoldSolver::new(
        &graph,
        FruchtermanReingoldConfig {
            iterations: 25,
            backend: FrBackend::BarnesHut,
            theta: 0.0,
            ..Default::default()
        },
    )
    .unwrap()
    .run(&mut l_bh0)
    .unwrap();
    for i in 0..16 {
        assert!((l_exact.x()[i] - l_bh0.x()[i]).abs() < 1e-8);
        assert!((l_exact.y()[i] - l_bh0.y()[i]).abs() < 1e-8);
    }

    // 2. Grid backend enforces strict Euclidean d < 2k cutoff (edgeless graph at d >= 2k stays unmoved,
    //    whereas d < 2k repels).
    let edgeless = CsrGraph::<f64>::from_edges(3, &[], None, false, false).unwrap();
    // Node 0 at (0, 0), Node 1 at (1.5, 0) (d = 1.5 < 2k=2.0), Node 2 at (2.1, 0) (d = 2.1 >= 2k=2.0 from 0, d = 0.6 < 2k from 1).
    // Put Node 2 far away at (10.0, 0) so it is >= 2k from both 0 and 1.
    let mut l_grid_cutoff = Layout2d::from_xy(vec![0.0, 1.5, 10.0], vec![0.0, 0.0, 0.0]).unwrap();
    let mut s_grid = FruchtermanReingoldSolver::new(
        &edgeless,
        FruchtermanReingoldConfig {
            iterations: 1,
            backend: FrBackend::Grid,
            k: Some(1.0),
            initial_temperature: Some(10.0),
            ..Default::default()
        },
    )
    .unwrap();
    s_grid.step(&mut l_grid_cutoff).unwrap();
    // Node 0 and Node 1 repel each other (d = 1.5 < 2k = 2.0), Node 2 experiences 0 repulsion (d >= 2k).
    assert!(l_grid_cutoff.x()[0] < 0.0);
    assert!(l_grid_cutoff.x()[1] > 1.5);
    assert_eq!(l_grid_cutoff.x()[2], 10.0);
    assert_eq!(l_grid_cutoff.y()[2], 0.0);

    // Also check diagonal corner inside adjacent grid cell with d >= 2k is strictly ignored:
    // k=1.0 -> 2k=2.0; (0.1, 0.1) in cell (0,0) and (1.9, 1.9) in cell (0,0) or (2.05, 0.05) -> d = sqrt(1.95^2 + 0.05^2) < 2.0 vs (2.1, 2.1) -> d = sqrt(4+4) >= 2.0.
    let mut l_diag = Layout2d::from_xy(vec![0.1, 1.9], vec![0.1, 1.9]).unwrap();
    let edgeless2 = CsrGraph::<f64>::from_edges(2, &[], None, false, false).unwrap();
    let mut s_diag = FruchtermanReingoldSolver::new(
        &edgeless2,
        FruchtermanReingoldConfig {
            iterations: 1,
            backend: FrBackend::Grid,
            k: Some(1.0),
            initial_temperature: Some(10.0),
            ..Default::default()
        },
    )
    .unwrap();
    s_diag.step(&mut l_diag).unwrap();
    // Same cell (0,0), but Euclidean distance = 1.8 * sqrt(2) ~= 2.5456 >= 2.0 -> zero force!
    assert_eq!(l_diag.x()[0], 0.1);
    assert_eq!(l_diag.x()[1], 1.9);

    // 3. All 3 backends produce clean finite layouts on a 4x4 grid.
    for backend in [FrBackend::Exact, FrBackend::BarnesHut, FrBackend::Grid] {
        let mut l = init.clone();
        let mut s = FruchtermanReingoldSolver::new(
            &graph,
            FruchtermanReingoldConfig {
                iterations: 100,
                backend,
                gravity: 0.05,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(s.run(&mut l).unwrap(), SolverStatus::Completed);
        assert!(l.is_all_finite());
    }
}

#[test]
fn test_fr_step_vs_run_bit_exact_f64_f32_and_edge_cases() {
    let graph = grid_graph(3, 4);
    for backend in [FrBackend::Exact, FrBackend::BarnesHut, FrBackend::Grid] {
        let cfg = FruchtermanReingoldConfig {
            iterations: 20,
            backend,
            gravity: 0.02,
            ..Default::default()
        };

        // f64 bit-exactness
        let mut l64_r = seeded_layout::<f64>(12, 33);
        let mut l64_s = l64_r.clone();
        let mut s64_r = FruchtermanReingoldSolver::<f64>::new(&graph, cfg).unwrap();
        let mut s64_s = FruchtermanReingoldSolver::<f64>::new(&graph, cfg).unwrap();
        s64_r.run(&mut l64_r).unwrap();
        s64_s.run_steps(&mut l64_s, 7).unwrap();
        while !s64_s.status().is_finished() {
            s64_s.step(&mut l64_s).unwrap();
        }
        for i in 0..12 {
            assert_eq!(l64_r.x()[i].to_bits(), l64_s.x()[i].to_bits());
            assert_eq!(l64_r.y()[i].to_bits(), l64_s.y()[i].to_bits());
        }

        // f32 bit-exactness
        let mut l32_r = seeded_layout::<f32>(12, 33);
        let mut l32_s = l32_r.clone();
        let mut s32_r = FruchtermanReingoldSolver::<f32>::new(&graph, cfg).unwrap();
        let mut s32_s = FruchtermanReingoldSolver::<f32>::new(&graph, cfg).unwrap();
        s32_r.run(&mut l32_r).unwrap();
        while !s32_s.status().is_finished() {
            s32_s.step(&mut l32_s).unwrap();
        }
        for i in 0..12 {
            assert_eq!(l32_r.x()[i].to_bits(), l32_s.x()[i].to_bits());
            assert_eq!(l32_r.y()[i].to_bits(), l32_s.y()[i].to_bits());
        }
    }

    // Degenerate graphs: N=0, N=1, collapsed start, extreme coordinate span on Grid (no OOM!),
    // directed reciprocal edges, negative weight rejection, dimension/NaN errors.
    for n in [0usize, 1, 4] {
        let g = CsrGraph::<f64>::from_edges(n, &[], None, false, false).unwrap();
        let mut l = Layout2d::<f64>::zeros(n);
        let mut s =
            FruchtermanReingoldSolver::new(&g, FruchtermanReingoldConfig::default()).unwrap();
        assert!(s.run(&mut l).unwrap().is_finished());
        assert!(l.is_all_finite());
    }

    // Extreme coordinate span with Grid backend must not OOM or panic.
    let far_g = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
    let mut far_l = Layout2d::from_xy(vec![-1e9, 1e9], vec![-1e9, 1e9]).unwrap();
    let mut far_s = FruchtermanReingoldSolver::new(
        &far_g,
        FruchtermanReingoldConfig {
            iterations: 5,
            backend: FrBackend::Grid,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(far_s.run(&mut far_l).unwrap(), SolverStatus::Completed);
    assert!(far_l.is_all_finite());

    let dir_g = CsrGraph::from_edges(
        4,
        &[(0, 1), (1, 0), (1, 2), (2, 3)],
        Some(&[2.0, 4.0, 1.5, 3.0]),
        true,
        false,
    )
    .unwrap();
    let mut l_dir = seeded_layout::<f64>(4, 8);
    assert!(
        FruchtermanReingoldSolver::new(&dir_g, FruchtermanReingoldConfig::default())
            .unwrap()
            .run(&mut l_dir)
            .unwrap()
            .is_finished()
    );
    assert!(l_dir.is_all_finite());

    let s_zero = FruchtermanReingoldSolver::<f64>::new(
        &far_g,
        FruchtermanReingoldConfig {
            iterations: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(s_zero.status(), SolverStatus::Completed);
    assert_eq!(s_zero.current_temperature(), 0.0);

    let neg_g = CsrGraph::from_edges(2, &[(0, 1)], Some(&[-1.0]), false, false).unwrap();
    assert!(matches!(
        FruchtermanReingoldSolver::<f64>::new(&neg_g, FruchtermanReingoldConfig::default()),
        Err(CoreError::InvalidInput(_))
    ));

    let mut s_err =
        FruchtermanReingoldSolver::<f64>::new(&far_g, FruchtermanReingoldConfig::default())
            .unwrap();
    let mut wrong = Layout2d::<f64>::zeros(5);
    assert!(matches!(
        s_err.step(&mut wrong),
        Err(CoreError::DimensionMismatch { .. })
    ));
    let mut bad = Layout2d::from_xy(vec![0.0, f64::NAN], vec![0.0, 0.0]).unwrap();
    assert!(matches!(
        s_err.step(&mut bad),
        Err(CoreError::NonFiniteValue { .. })
    ));
}
