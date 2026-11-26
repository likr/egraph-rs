use egraph_core::{
    CoreError, CsrGraph, IterativeSolver, Layout2d, LayoutNd, ManifoldKind, ManifoldLayout2d,
    Scalar, SolverStatus,
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

fn init_manifold<S: Scalar>(kind: ManifoldKind, n: usize, seed: u64) -> ManifoldLayout2d<S> {
    let init_scale = match kind {
        ManifoldKind::HyperbolicPoincare => 0.45,
        ManifoldKind::HyperbolicLorentz => 0.8,
        ManifoldKind::Spherical | ManifoldKind::Torus2d => 1.0,
        _ => 0.5,
    };
    ManifoldLayout2d::<S>::init(kind, n, seed, init_scale).unwrap()
}

#[test]
fn test_fr_manifold_all_kinds_bit_exact_f64_f32_and_metric() {
    let graph = grid_graph(3, 4);
    let n = graph.num_nodes();
    let cfg = FruchtermanReingoldConfig {
        iterations: 20,
        backend: FrBackend::Exact,
        gravity: 0.03,
        ..Default::default()
    };

    for kind in [
        ManifoldKind::HyperbolicPoincare,
        ManifoldKind::HyperbolicLorentz,
        ManifoldKind::Spherical,
        ManifoldKind::Torus2d,
    ] {
        // f64 bit-exact step == run.
        let mut l_run = init_manifold::<f64>(kind, n, 33);
        if kind == ManifoldKind::Torus2d {
            l_run.set_torus_extents(2.5, 1.5).unwrap();
        } else {
            l_run.set_scale(2.0).unwrap();
        }
        let mut l_step = l_run.clone();
        let mut s_run = FruchtermanReingoldSolver::<f64>::new(&graph, cfg).unwrap();
        let mut s_step = FruchtermanReingoldSolver::<f64>::new(&graph, cfg).unwrap();

        assert_eq!(s_run.run(&mut l_run).unwrap(), SolverStatus::Completed);
        assert!(l_run.validate_finite().is_ok());

        s_step.run_steps(&mut l_step, 7).unwrap();
        while !s_step.status().is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        assert_eq!(s_run.status(), s_step.status());
        assert_eq!(s_run.iteration(), s_step.iteration());
        for i in 0..n {
            assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
            assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
        }

        // f32 bit-exact step == run.
        let mut l32_run = init_manifold::<f32>(kind, n, 33);
        let mut l32_step = l32_run.clone();
        let mut s32_run = FruchtermanReingoldSolver::<f32>::new(&graph, cfg).unwrap();
        let mut s32_step = FruchtermanReingoldSolver::<f32>::new(&graph, cfg).unwrap();
        assert_eq!(s32_run.run(&mut l32_run).unwrap(), SolverStatus::Completed);
        assert!(l32_run.validate_finite().is_ok());
        while !s32_step.status().is_finished() {
            s32_step.step(&mut l32_step).unwrap();
        }
        for i in 0..n {
            assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
            assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
        }
    }
}

#[test]
fn test_fr_manifold_edge_cases_backends_and_layout_kind_lock() {
    let graph = grid_graph(2, 2);

    // Non-Exact backends return UnsupportedOperation on ManifoldLayout2d.
    for backend in [FrBackend::BarnesHut, FrBackend::Grid] {
        let mut s = FruchtermanReingoldSolver::<f64>::new(
            &graph,
            FruchtermanReingoldConfig {
                backend,
                ..Default::default()
            },
        )
        .unwrap();
        let mut l = init_manifold::<f64>(ManifoldKind::Spherical, 4, 1);
        assert!(matches!(
            s.step(&mut l),
            Err(CoreError::UnsupportedOperation(_))
        ));
    }

    // Degenerates: n=0, n=1, collapsed zero start, edgeless.
    for n in [0usize, 1, 4] {
        let g = CsrGraph::<f64>::from_edges(n, &[], None, false, false).unwrap();
        for kind in [
            ManifoldKind::HyperbolicPoincare,
            ManifoldKind::HyperbolicLorentz,
            ManifoldKind::Spherical,
            ManifoldKind::Torus2d,
        ] {
            let mut l = ManifoldLayout2d::<f64>::zeros(kind, n);
            let mut s = FruchtermanReingoldSolver::<f64>::new(
                &g,
                FruchtermanReingoldConfig {
                    iterations: 5,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(s.run(&mut l).unwrap().is_finished());
            assert!(l.validate_finite().is_ok());
        }
    }

    // Node mismatch and ActiveLayoutKind switching rejections.
    let cfg = FruchtermanReingoldConfig {
        iterations: 5,
        ..Default::default()
    };
    let mut s_err = FruchtermanReingoldSolver::<f64>::new(&graph, cfg).unwrap();
    let mut l_wrong = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Torus2d, 2);
    assert!(matches!(
        s_err.step(&mut l_wrong),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let mut l2 = Layout2d::<f64>::init(4, 1, 1.0).unwrap();
    let mut lnd = LayoutNd::<f64>::init(4, 3, 1, 1.0).unwrap();
    let mut l_tor = init_manifold::<f64>(ManifoldKind::Torus2d, 4, 2);
    let mut l_sph = init_manifold::<f64>(ManifoldKind::Spherical, 4, 2);

    let mut s_from_2d = FruchtermanReingoldSolver::<f64>::new(&graph, cfg).unwrap();
    s_from_2d.step(&mut l2).unwrap();
    assert!(matches!(
        s_from_2d.step(&mut lnd),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        s_from_2d.step(&mut l_tor),
        Err(CoreError::InvalidInput(_))
    ));

    let mut s_from_nd = FruchtermanReingoldSolver::<f64>::new(&graph, cfg).unwrap();
    s_from_nd.step(&mut lnd).unwrap();
    assert!(matches!(
        s_from_nd.step(&mut l2),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        s_from_nd.step(&mut l_tor),
        Err(CoreError::InvalidInput(_))
    ));

    let mut s_from_m = FruchtermanReingoldSolver::<f64>::new(&graph, cfg).unwrap();
    s_from_m.step(&mut l_tor).unwrap();
    assert!(matches!(
        s_from_m.step(&mut l2),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        s_from_m.step(&mut lnd),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        s_from_m.step(&mut l_sph),
        Err(CoreError::InvalidInput(_))
    ));
}
