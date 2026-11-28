use egraph_core::{
    CoreError, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Layout3d, Scalar,
    SolverStatus,
};

use crate::backend::{BarnesHut, Exact, FrBackend, Grid};

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
fn test_fr_config_defaults_builders_and_validation() {
    let cfg = FruchtermanReingoldConfig::new();
    assert_eq!(cfg.iterations, 200);
    assert_eq!(cfg.k, None);
    assert_eq!(cfg.area, None);
    assert_eq!(cfg.initial_temperature, None);
    assert_eq!(cfg.gravity, 0.0);
    assert_eq!(cfg.seed, 42);
    cfg.validate().unwrap();

    for bad in [
        cfg.clone().with_k(Some(0.0)),
        cfg.clone().with_k(Some(-1.0)),
        cfg.clone().with_k(Some(f64::NAN)),
        cfg.clone().with_area(Some(0.0)),
        cfg.clone().with_area(Some(f64::INFINITY)),
        cfg.clone().with_initial_temperature(Some(-0.1)),
        cfg.clone().with_initial_temperature(Some(f64::NAN)),
        cfg.clone().with_gravity(-0.1),
    ] {
        assert!(bad.validate().is_err());
    }
}

#[test]
fn test_fr_backends_exact_barnes_hut_and_grid_cutoff() {
    let graph = grid_graph(4, 4);
    let init = seeded_layout::<f64>(16, 17);

    let mut l_exact = init.clone();
    let mut l_bh0 = init.clone();
    FruchtermanReingoldSolver::<Layout2d<f64>>::new(
        &graph,
        FruchtermanReingoldConfig::new().with_iterations(25),
    )
    .unwrap()
    .run(&mut l_exact)
    .unwrap();
    FruchtermanReingoldSolver::<Layout2d<f64>, BarnesHut>::with_backend(
        &graph,
        FruchtermanReingoldConfig::new().with_iterations(25),
        BarnesHut::new(0.0),
    )
    .unwrap()
    .run(&mut l_bh0)
    .unwrap();
    for i in 0..16 {
        assert!((l_exact.x()[i] - l_bh0.x()[i]).abs() < 1e-8);
        assert!((l_exact.y()[i] - l_bh0.y()[i]).abs() < 1e-8);
    }

    let edgeless = CsrGraph::<()>::from_unweighted_edges(3, &[], false, false).unwrap();
    let mut l_grid_cutoff = Layout2d::from_xy(vec![0.0, 1.5, 10.0], vec![0.0, 0.0, 0.0]).unwrap();
    let mut s_grid = FruchtermanReingoldSolver::<Layout2d<f64>, Grid>::new_unweighted(
        &edgeless,
        FruchtermanReingoldConfig::new()
            .with_iterations(1)
            .with_k(Some(1.0))
            .with_initial_temperature(Some(10.0)),
    )
    .unwrap();
    s_grid.step(&mut l_grid_cutoff).unwrap();
    assert!(l_grid_cutoff.x()[0] < 0.0);
    assert!(l_grid_cutoff.x()[1] > 1.5);
    assert_eq!(l_grid_cutoff.x()[2], 10.0);
    assert_eq!(l_grid_cutoff.y()[2], 0.0);

    let mut l_diag = Layout2d::from_xy(vec![0.1, 1.9], vec![0.1, 1.9]).unwrap();
    let edgeless2 = CsrGraph::<()>::from_unweighted_edges(2, &[], false, false).unwrap();
    let mut s_diag = FruchtermanReingoldSolver::<Layout2d<f64>, Grid>::new_unweighted(
        &edgeless2,
        FruchtermanReingoldConfig::new()
            .with_iterations(1)
            .with_k(Some(1.0))
            .with_initial_temperature(Some(10.0)),
    )
    .unwrap();
    s_diag.step(&mut l_diag).unwrap();
    assert_eq!(l_diag.x()[0], 0.1);
    assert_eq!(l_diag.x()[1], 1.9);

    for backend in [
        FrBackend::exact(),
        FrBackend::barnes_hut(1.2),
        FrBackend::grid(),
    ] {
        let mut l = init.clone();
        let mut s = FruchtermanReingoldSolver::<Layout2d<f64>, FrBackend>::with_backend(
            &graph,
            FruchtermanReingoldConfig::new()
                .with_iterations(100)
                .with_gravity(0.05),
            backend,
        )
        .unwrap();
        assert_eq!(s.run(&mut l).unwrap(), SolverStatus::Completed);
        assert!(l.is_all_finite());
    }
}

#[test]
fn test_fr_step_vs_run_bit_exact_3d_and_v6_contract() {
    let graph = grid_graph(3, 4);
    for backend in [
        FrBackend::exact(),
        FrBackend::barnes_hut(1.2),
        FrBackend::grid(),
    ] {
        let cfg = FruchtermanReingoldConfig::new()
            .with_iterations(20)
            .with_gravity(0.02);

        let mut l64_r = seeded_layout::<f64>(12, 33);
        let mut l64_s = l64_r.clone();
        let mut s64_r = FruchtermanReingoldSolver::<Layout2d<f64>, FrBackend>::with_backend(
            &graph,
            cfg.clone(),
            backend.clone(),
        )
        .unwrap();
        let mut s64_s = FruchtermanReingoldSolver::<Layout2d<f64>, FrBackend>::with_backend(
            &graph,
            cfg.clone(),
            backend.clone(),
        )
        .unwrap();
        s64_r.run(&mut l64_r).unwrap();
        s64_s.run_steps(&mut l64_s, 7).unwrap();
        while !s64_s.status().is_finished() {
            s64_s.step(&mut l64_s).unwrap();
        }
        for i in 0..12 {
            assert_eq!(l64_r.x()[i].to_bits(), l64_s.x()[i].to_bits());
            assert_eq!(l64_r.y()[i].to_bits(), l64_s.y()[i].to_bits());
        }

        let mut l32_r = Layout3d::<f32>::init(12, 33, 2.0).unwrap();
        let mut l32_s = l32_r.clone();
        let mut s32_r = FruchtermanReingoldSolver::<Layout3d<f32>, FrBackend>::with_backend(
            &graph,
            cfg.clone(),
            backend.clone(),
        )
        .unwrap();
        let mut s32_s = FruchtermanReingoldSolver::<Layout3d<f32>, FrBackend>::with_backend(
            &graph, cfg, backend,
        )
        .unwrap();
        s32_r.run(&mut l32_r).unwrap();
        while !s32_s.status().is_finished() {
            s32_s.step(&mut l32_s).unwrap();
        }
        for i in 0..12 {
            assert_eq!(l32_r.x()[i].to_bits(), l32_s.x()[i].to_bits());
            assert_eq!(l32_r.y()[i].to_bits(), l32_s.y()[i].to_bits());
            assert_eq!(l32_r.z()[i].to_bits(), l32_s.z()[i].to_bits());
        }
    }

    let far_g = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
    let mut far_l = Layout2d::from_xy(vec![-1e9, 1e9], vec![-1e9, 1e9]).unwrap();
    let mut far_s = FruchtermanReingoldSolver::<Layout2d<f64>, Grid>::new(
        &far_g,
        FruchtermanReingoldConfig::new().with_iterations(5),
    )
    .unwrap();
    assert_eq!(far_s.run(&mut far_l).unwrap(), SolverStatus::Completed);
    assert!(far_l.is_all_finite());

    let mut s_err = FruchtermanReingoldSolver::<Layout2d<f64>, Exact>::new(
        &far_g,
        FruchtermanReingoldConfig::new(),
    )
    .unwrap();
    assert_eq!(s_err.backend(), &Exact::new());
    assert_eq!(s_err.total_iterations(), Some(200));
    let mut wrong = Layout2d::<f64>::zeros(5);
    assert!(matches!(
        s_err.run_steps(&mut wrong, 0),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert_eq!(s_err.status(), SolverStatus::Initialized);
    let mut bad = Layout2d::from_xy(vec![0.0, f64::NAN], vec![0.0, 0.0]).unwrap();
    assert!(matches!(
        s_err.step(&mut bad),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert_eq!(s_err.status(), SolverStatus::Initialized);

    // Non-finite f32 overflow during step transitions to SolverStatus::Failed.
    let edgeless = CsrGraph::<f64>::from_edges(2, &[], None, false, false).unwrap();
    let mut s_fail = FruchtermanReingoldSolver::<Layout2d<f32>>::new(
        &edgeless,
        FruchtermanReingoldConfig::new()
            .with_iterations(5)
            .with_initial_temperature(Some(1e38))
            .with_k(Some(1e58)),
    )
    .unwrap();
    let mut l_fail = Layout2d::<f32>::from_xy(vec![-3.4e38, -2.4e38], vec![0.0, 0.0]).unwrap();
    assert!(matches!(
        s_fail.step(&mut l_fail),
        Err(CoreError::NumericalBreakdown(_))
    ));
    assert_eq!(s_fail.status(), SolverStatus::Failed);
}
