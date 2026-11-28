use egraph_core::{CoreError, CsrGraph, IterativeSolver, Layout3d, LayoutNd, Scalar, SolverStatus};

use super::*;

fn grid_3d(nx: usize, ny: usize, nz: usize) -> CsrGraph<f64> {
    let n = nx * ny * nz;
    let mut edges = Vec::new();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let u = ((z * ny + y) * nx + x) as u32;
                if x + 1 < nx {
                    edges.push((u, u + 1));
                }
                if y + 1 < ny {
                    edges.push((u, u + nx as u32));
                }
                if z + 1 < nz {
                    edges.push((u, u + (nx * ny) as u32));
                }
            }
        }
    }
    CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
}

fn assert_step_eq_run_3d<S: Scalar>(
    cfg: FruchtermanReingoldConfig,
    graph: &CsrGraph<f64>,
    init: &Layout3d<S>,
) {
    let mut s_run = FruchtermanReingoldSolver::<Layout3d<S>>::new(graph, cfg.clone()).unwrap();
    let mut s_step = FruchtermanReingoldSolver::<Layout3d<S>>::new(graph, cfg).unwrap();
    let mut l_run = init.clone();
    let mut l_step = init.clone();

    assert_eq!(s_run.run(&mut l_run).unwrap(), SolverStatus::Completed);
    s_step.run_steps(&mut l_step, 6).unwrap();
    while !s_step.status().is_finished() {
        s_step.step(&mut l_step).unwrap();
    }
    assert_eq!(s_run.status(), s_step.status());
    assert_eq!(s_run.iteration(), s_step.iteration());
    assert_eq!(s_run.total_iterations(), s_step.total_iterations());
    for i in 0..init.len() {
        assert_eq!(
            l_run.x()[i].to_f64().to_bits(),
            l_step.x()[i].to_f64().to_bits()
        );
        assert_eq!(
            l_run.y()[i].to_f64().to_bits(),
            l_step.y()[i].to_f64().to_bits()
        );
        assert_eq!(
            l_run.z()[i].to_f64().to_bits(),
            l_step.z()[i].to_f64().to_bits()
        );
    }
}

fn assert_step_eq_run_nd<S: Scalar>(
    cfg: FruchtermanReingoldConfig,
    graph: &CsrGraph<f64>,
    init: &LayoutNd<S>,
) {
    let mut s_run = FruchtermanReingoldSolver::<LayoutNd<S>>::new(graph, cfg.clone()).unwrap();
    let mut s_step = FruchtermanReingoldSolver::<LayoutNd<S>>::new(graph, cfg).unwrap();
    let mut l_run = init.clone();
    let mut l_step = init.clone();

    assert_eq!(s_run.run(&mut l_run).unwrap(), SolverStatus::Completed);
    s_step.run_steps(&mut l_step, 6).unwrap();
    while !s_step.status().is_finished() {
        s_step.step(&mut l_step).unwrap();
    }
    assert_eq!(s_run.status(), s_step.status());
    assert_eq!(s_run.iteration(), s_step.iteration());
    assert_eq!(s_run.total_iterations(), s_step.total_iterations());
    for (&a, &b) in l_run.coords().iter().zip(l_step.coords().iter()) {
        assert_eq!(a.to_f64().to_bits(), b.to_f64().to_bits());
        assert!(a.is_finite());
    }
}

#[test]
fn test_fr_3d_backends_and_nd_exact() {
    let graph = grid_3d(3, 2, 2);
    let n = graph.num_nodes();

    for backend in [FrBackend::Exact, FrBackend::BarnesHut, FrBackend::Grid] {
        let cfg = FruchtermanReingoldConfig::new()
            .with_iterations(20)
            .with_backend(backend)
            .with_gravity(0.03);
        let init64 = Layout3d::<f64>::init(n, 27, 4.0).unwrap();
        assert_step_eq_run_3d(cfg.clone(), &graph, &init64);
        let init32 = Layout3d::<f32>::init(n, 27, 4.0).unwrap();
        assert_step_eq_run_3d(cfg, &graph, &init32);
    }

    // 4D Exact backend bit-exactness across f64 & f32.
    let cfg_exact = FruchtermanReingoldConfig::new()
        .with_iterations(15)
        .with_backend(FrBackend::Exact);
    assert_step_eq_run_nd(
        cfg_exact.clone(),
        &graph,
        &LayoutNd::<f64>::init(n, 4, 88, 3.0).unwrap(),
    );
    assert_step_eq_run_nd(
        cfg_exact,
        &graph,
        &LayoutNd::<f32>::init(n, 4, 88, 3.0).unwrap(),
    );

    // 3D Layout3d BarnesHut theta=0.0 matches Exact to machine precision.
    let init = Layout3d::<f64>::init(n, 41, 4.0).unwrap();
    let mut l_exact = init.clone();
    let mut l_bh0 = init.clone();
    FruchtermanReingoldSolver::<Layout3d<f64>>::new(
        &graph,
        FruchtermanReingoldConfig::new()
            .with_iterations(15)
            .with_backend(FrBackend::Exact),
    )
    .unwrap()
    .run(&mut l_exact)
    .unwrap();
    FruchtermanReingoldSolver::<Layout3d<f64>>::new(
        &graph,
        FruchtermanReingoldConfig::new()
            .with_iterations(15)
            .with_backend(FrBackend::BarnesHut)
            .with_theta(0.0),
    )
    .unwrap()
    .run(&mut l_bh0)
    .unwrap();
    for i in 0..n {
        assert!((l_exact.x()[i] - l_bh0.x()[i]).abs() < 1e-8);
        assert!((l_exact.y()[i] - l_bh0.y()[i]).abs() < 1e-8);
        assert!((l_exact.z()[i] - l_bh0.z()[i]).abs() < 1e-8);
    }

    // 3D Grid backend strictly enforces d < 2k across forward 3D cell neighbors.
    let edgeless = CsrGraph::<()>::from_unweighted_edges(4, &[], false, false).unwrap();
    let mut l_grid3 = Layout3d::from_xyz(
        vec![0.1, 0.1, 1.9, 10.0],
        vec![0.1, 0.1, 1.9, 10.0],
        vec![0.1, 1.5, 1.9, 10.0],
    )
    .unwrap();
    let mut s_grid3 = FruchtermanReingoldSolver::<Layout3d<f64>>::new_unweighted(
        &edgeless,
        FruchtermanReingoldConfig::new()
            .with_iterations(1)
            .with_backend(FrBackend::Grid)
            .with_k(Some(1.0))
            .with_initial_temperature(Some(10.0)),
    )
    .unwrap();
    s_grid3.step(&mut l_grid3).unwrap();
    assert!(l_grid3.z()[0] < 0.1);
    assert!(l_grid3.z()[1] > 1.5);
    assert_eq!(l_grid3.x()[2], 1.9);
    assert_eq!(l_grid3.x()[3], 10.0);
    assert_eq!(s_grid3.status(), SolverStatus::Completed);
    assert_eq!(s_grid3.iteration(), 1);
    assert_eq!(s_grid3.total_iterations(), Some(1));
}

#[test]
fn test_fr_nd_fail_fast_construction_and_dimension_lock() {
    let disc = CsrGraph::<f64>::from_edges(6, &[(0, 1), (2, 3)], None, false, false).unwrap();

    // BarnesHut and Grid fail fast at construction on LayoutNd.
    for backend in [FrBackend::BarnesHut, FrBackend::Grid] {
        let err = FruchtermanReingoldSolver::<LayoutNd<f64>>::new(
            &disc,
            FruchtermanReingoldConfig::new().with_backend(backend),
        );
        assert!(matches!(err, Err(CoreError::UnsupportedOperation(_))));
    }

    let mut s_exact = FruchtermanReingoldSolver::<LayoutNd<f64>>::new(
        &disc,
        FruchtermanReingoldConfig::new()
            .with_iterations(5)
            .with_backend(FrBackend::Exact),
    )
    .unwrap();

    let mut l_bad_len = LayoutNd::<f64>::zeros(2, 3).unwrap();
    assert!(matches!(
        s_exact.run_steps(&mut l_bad_len, 0),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let mut l_nan = LayoutNd::from_flat(6, 3, &{
        let mut v = vec![0.0; 18];
        v[5] = f64::NAN;
        v
    })
    .unwrap();
    assert!(matches!(
        s_exact.step(&mut l_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));

    let mut l3 = LayoutNd::<f64>::init(6, 3, 9, 1.0).unwrap();
    let mut l4 = LayoutNd::<f64>::init(6, 4, 1, 1.0).unwrap();
    s_exact.step(&mut l3).unwrap();
    assert!(matches!(
        s_exact.step(&mut l4),
        Err(CoreError::DimensionMismatch { .. })
    ));
}
