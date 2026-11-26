use egraph_core::{CoreError, CsrGraph, IterativeSolver, Layout2d, LayoutNd, Scalar, SolverStatus};

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

fn assert_step_eq_run_nd<S: Scalar>(
    cfg: FruchtermanReingoldConfig,
    graph: &CsrGraph<f64>,
    init: &LayoutNd<S>,
) {
    let mut s_run = FruchtermanReingoldSolver::<S>::new(graph, cfg).unwrap();
    let mut s_step = FruchtermanReingoldSolver::<S>::new(graph, cfg).unwrap();
    let mut l_run = init.copy();
    let mut l_step = init.copy();

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
        assert_eq!(a.to_f32().to_bits(), b.to_f32().to_bits());
        assert!(a.is_finite());
    }
}

#[test]
fn test_fr_nd_backends_step_run_bit_exact_and_3d_grid_cutoff() {
    let graph = grid_3d(3, 2, 2);
    let n = graph.num_nodes();

    for backend in [FrBackend::Exact, FrBackend::BarnesHut, FrBackend::Grid] {
        let cfg = FruchtermanReingoldConfig {
            iterations: 20,
            backend,
            gravity: 0.03,
            ..Default::default()
        };
        let init64 = LayoutNd::<f64>::init(n, 3, 27, 4.0).unwrap();
        assert_step_eq_run_nd(cfg, &graph, &init64);
        let init32 = LayoutNd::<f32>::init(n, 3, 27, 4.0).unwrap();
        assert_step_eq_run_nd(cfg, &graph, &init32);
    }

    // 4D Exact backend bit-exactness across f64 & f32.
    let cfg_exact = FruchtermanReingoldConfig {
        iterations: 15,
        backend: FrBackend::Exact,
        ..Default::default()
    };
    assert_step_eq_run_nd(
        cfg_exact,
        &graph,
        &LayoutNd::<f64>::init(n, 4, 88, 3.0).unwrap(),
    );
    assert_step_eq_run_nd(
        cfg_exact,
        &graph,
        &LayoutNd::<f32>::init(n, 4, 88, 3.0).unwrap(),
    );

    // 3D BarnesHut theta=0.0 matches Exact to machine precision.
    let init = LayoutNd::<f64>::init(n, 3, 41, 4.0).unwrap();
    let mut l_exact = init.copy();
    let mut l_bh0 = init.copy();
    FruchtermanReingoldSolver::<f64>::new(
        &graph,
        FruchtermanReingoldConfig {
            iterations: 15,
            backend: FrBackend::Exact,
            ..Default::default()
        },
    )
    .unwrap()
    .run(&mut l_exact)
    .unwrap();
    FruchtermanReingoldSolver::<f64>::new(
        &graph,
        FruchtermanReingoldConfig {
            iterations: 15,
            backend: FrBackend::BarnesHut,
            theta: 0.0,
            ..Default::default()
        },
    )
    .unwrap()
    .run(&mut l_bh0)
    .unwrap();
    for (&a, &b) in l_exact.coords().iter().zip(l_bh0.coords().iter()) {
        assert!((a - b).abs() < 1e-8);
    }

    // 3D Grid backend strictly enforces d < 2k across forward 3D cell neighbors.
    let edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    // k=1.0 -> 2k=2.0.
    // 0 at (0.1, 0.1, 0.1), 1 at (0.1, 0.1, 1.5) (dz=1.4 < 2.0 -> repels in z),
    // 2 at (1.9, 1.9, 1.9) (same cell (0,0,0) but d > 2.0 from 0 -> zero force from 0; d > 2.0 from 1),
    // 3 at (10.0, 10.0, 10.0) (far -> unmoved).
    let mut l_grid3 = LayoutNd::from_flat(
        4,
        3,
        vec![
            0.1, 0.1, 0.1, //
            0.1, 0.1, 1.5, //
            1.9, 1.9, 1.9, //
            10.0, 10.0, 10.0,
        ],
    )
    .unwrap();
    let mut s_grid3 = FruchtermanReingoldSolver::<f64>::new(
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
    s_grid3.step(&mut l_grid3).unwrap();
    let c = l_grid3.coords();
    assert!(c[2] < 0.1);
    assert!(c[5] > 1.5);
    assert_eq!(c[6], 1.9);
    assert_eq!(c[9], 10.0);

    // 2D LayoutNd matches Layout2d.
    let g2 = grid_3d(3, 3, 1);
    let mut l2 = Layout2d::<f64>::init(9, 12, 3.0).unwrap();
    let flat2: Vec<f64> = l2
        .x()
        .iter()
        .zip(l2.y().iter())
        .flat_map(|(&x, &y)| [x, y])
        .collect();
    let mut lnd2 = LayoutNd::from_flat(9, 2, flat2).unwrap();
    let cfg_g = FruchtermanReingoldConfig {
        iterations: 10,
        backend: FrBackend::Grid,
        ..Default::default()
    };
    FruchtermanReingoldSolver::<f64>::new(&g2, cfg_g)
        .unwrap()
        .run(&mut l2)
        .unwrap();
    FruchtermanReingoldSolver::<f64>::new(&g2, cfg_g)
        .unwrap()
        .run(&mut lnd2)
        .unwrap();
    for i in 0..9 {
        assert!((l2.x()[i] - lnd2.coords()[i * 2]).abs() < 1e-12);
        assert!((l2.y()[i] - lnd2.coords()[i * 2 + 1]).abs() < 1e-12);
    }

    // Inherent status methods remain unambiguous.
    assert_eq!(s_grid3.status(), SolverStatus::Completed);
    assert_eq!(s_grid3.iteration(), 1);
    assert_eq!(s_grid3.total_iterations(), 1);
}

#[test]
fn test_fr_nd_degenerates_and_error_guards() {
    for n in [0usize, 1, 5] {
        let g = CsrGraph::<f64>::from_edges(n, &[], None, false, false).unwrap();
        let mut l = LayoutNd::<f64>::zeros(n, 3).unwrap();
        let mut s = FruchtermanReingoldSolver::<f64>::new(
            &g,
            FruchtermanReingoldConfig {
                iterations: 5,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(s.run(&mut l).unwrap().is_finished());
        assert!(l.is_all_finite());
    }

    let disc = CsrGraph::<f64>::from_edges(6, &[(0, 1), (2, 3)], None, false, false).unwrap();
    let l_collinear = LayoutNd::from_flat(
        6,
        3,
        (0..6)
            .flat_map(|i| [i as f64, 2.0 * i as f64, 3.0 * i as f64])
            .collect(),
    )
    .unwrap();
    let l_planar = LayoutNd::from_flat(
        6,
        3,
        vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.5, 0.0, 2.0, 1.0,
            0.0,
        ],
    )
    .unwrap();
    for init in [&l_collinear, &l_planar] {
        let mut l = init.copy();
        FruchtermanReingoldSolver::<f64>::new(
            &disc,
            FruchtermanReingoldConfig {
                iterations: 5,
                backend: FrBackend::BarnesHut,
                ..Default::default()
            },
        )
        .unwrap()
        .run(&mut l)
        .unwrap();
        assert!(l.is_all_finite());
    }

    let mut s_bh = FruchtermanReingoldSolver::<f64>::new(
        &disc,
        FruchtermanReingoldConfig {
            backend: FrBackend::BarnesHut,
            ..Default::default()
        },
    )
    .unwrap();
    let mut s_grid = FruchtermanReingoldSolver::<f64>::new(
        &disc,
        FruchtermanReingoldConfig {
            backend: FrBackend::Grid,
            ..Default::default()
        },
    )
    .unwrap();
    let mut l4 = LayoutNd::<f64>::init(6, 4, 1, 1.0).unwrap();
    assert!(matches!(
        s_bh.step(&mut l4),
        Err(CoreError::UnsupportedOperation(_))
    ));
    assert!(matches!(
        s_grid.step(&mut l4),
        Err(CoreError::UnsupportedOperation(_))
    ));

    let mut s_exact = FruchtermanReingoldSolver::<f64>::new(
        &disc,
        FruchtermanReingoldConfig {
            iterations: 5,
            backend: FrBackend::Exact,
            ..Default::default()
        },
    )
    .unwrap();
    let mut l_bad_len = LayoutNd::<f64>::zeros(2, 3).unwrap();
    assert!(matches!(
        s_exact.step(&mut l_bad_len),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let mut l_nan = LayoutNd::from_flat(6, 3, {
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
    s_exact.step(&mut l3).unwrap();
    assert!(matches!(
        s_exact.step(&mut l4),
        Err(CoreError::DimensionMismatch { .. })
    ));
}
