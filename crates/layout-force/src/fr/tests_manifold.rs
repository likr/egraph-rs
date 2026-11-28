use egraph_core::{
    CoreError, CoreResult, CsrGraph, IterativeSolver, LayoutStorage, ManifoldKind,
    ManifoldLayout2d, MetricSpace, RiemannianLayout, Scalar, SolverStatus,
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
    let cfg = FruchtermanReingoldConfig::new()
        .with_iterations(20)
        .with_backend(FrBackend::Exact)
        .with_gravity(0.03);

    for kind in [
        ManifoldKind::HyperbolicPoincare,
        ManifoldKind::HyperbolicLorentz,
        ManifoldKind::Spherical,
        ManifoldKind::Torus2d,
    ] {
        let mut l_run = init_manifold::<f64>(kind, n, 33);
        if kind == ManifoldKind::Torus2d {
            l_run.set_torus_extents(2.5, 1.5).unwrap();
        } else {
            l_run.set_scale(2.0).unwrap();
        }
        let mut l_step = l_run.clone();
        let mut s_run =
            FruchtermanReingoldSolver::<ManifoldLayout2d<f64>>::new(&graph, cfg.clone()).unwrap();
        let mut s_step =
            FruchtermanReingoldSolver::<ManifoldLayout2d<f64>>::new(&graph, cfg.clone()).unwrap();

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

        let mut l32_run = init_manifold::<f32>(kind, n, 33);
        let mut l32_step = l32_run.clone();
        let mut s32_run =
            FruchtermanReingoldSolver::<ManifoldLayout2d<f32>>::new(&graph, cfg.clone()).unwrap();
        let mut s32_step =
            FruchtermanReingoldSolver::<ManifoldLayout2d<f32>>::new(&graph, cfg.clone()).unwrap();
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
fn test_fr_manifold_fail_fast_backends_and_kind_lock() {
    let graph = grid_graph(2, 2);

    // Non-Exact backends fail fast at construction on ManifoldLayout2d.
    for backend in [FrBackend::BarnesHut, FrBackend::Grid] {
        let err = FruchtermanReingoldSolver::<ManifoldLayout2d<f64>>::new(
            &graph,
            FruchtermanReingoldConfig::new().with_backend(backend),
        );
        assert!(matches!(err, Err(CoreError::UnsupportedOperation(_))));
    }

    for n in [0usize, 1, 4] {
        let g = CsrGraph::<f64>::from_edges(n, &[], None, false, false).unwrap();
        for kind in [
            ManifoldKind::HyperbolicPoincare,
            ManifoldKind::HyperbolicLorentz,
            ManifoldKind::Spherical,
            ManifoldKind::Torus2d,
        ] {
            let mut l = ManifoldLayout2d::<f64>::zeros(kind, n);
            let mut s = FruchtermanReingoldSolver::<ManifoldLayout2d<f64>>::new(
                &g,
                FruchtermanReingoldConfig::new().with_iterations(5),
            )
            .unwrap();
            assert!(s.run(&mut l).unwrap().is_finished());
            assert!(l.validate_finite().is_ok());
        }
    }

    let cfg = FruchtermanReingoldConfig::new().with_iterations(5);
    let mut s_err =
        FruchtermanReingoldSolver::<ManifoldLayout2d<f64>>::new(&graph, cfg.clone()).unwrap();
    let mut l_wrong = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Torus2d, 2);
    assert!(matches!(
        s_err.step(&mut l_wrong),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let mut l_tor = init_manifold::<f64>(ManifoldKind::Torus2d, 4, 2);
    let mut l_sph = init_manifold::<f64>(ManifoldKind::Spherical, 4, 2);
    let mut s_from_m =
        FruchtermanReingoldSolver::<ManifoldLayout2d<f64>>::new(&graph, cfg).unwrap();
    s_from_m.step(&mut l_tor).unwrap();
    assert!(matches!(
        s_from_m.step(&mut l_sph),
        Err(CoreError::InvalidInput(_))
    ));
}

#[derive(Debug, Clone)]
struct CustomRiemannianRing {
    angles: Vec<f64>,
}

impl LayoutStorage for CustomRiemannianRing {
    type Scalar = f64;

    fn len(&self) -> usize {
        self.angles.len()
    }

    fn validate_finite(&self) -> CoreResult<()> {
        if self.angles.iter().all(|a| a.is_finite()) {
            Ok(())
        } else {
            Err(CoreError::non_finite("CustomRiemannianRing"))
        }
    }
}

impl MetricSpace for CustomRiemannianRing {
    fn distance(&self, i: usize, j: usize) -> f64 {
        let diff = (self.angles[j] - self.angles[i]).rem_euclid(std::f64::consts::TAU);
        let signed = if diff > std::f64::consts::PI {
            diff - std::f64::consts::TAU
        } else {
            diff
        };
        signed.abs()
    }
}

impl RiemannianLayout for CustomRiemannianRing {
    fn tangent_dim(&self) -> usize {
        1
    }

    fn log_map_into(&self, i: usize, j: usize, out: &mut [f64]) -> CoreResult<()> {
        let diff = (self.angles[j] - self.angles[i]).rem_euclid(std::f64::consts::TAU);
        out[0] = if diff > std::f64::consts::PI {
            diff - std::f64::consts::TAU
        } else {
            diff
        };
        Ok(())
    }

    fn exp_map(&mut self, i: usize, delta: &[f64]) -> CoreResult<()> {
        self.angles[i] = (self.angles[i] + delta[0]).rem_euclid(std::f64::consts::TAU);
        Ok(())
    }

    fn parallel_transport_into(
        &self,
        _i: usize,
        _delta: &[f64],
        _vec: &mut [f64],
    ) -> CoreResult<()> {
        Ok(())
    }
}

#[test]
fn test_fr_custom_riemannian_layout_exact_works_and_bh_fails() {
    let g = CsrGraph::<()>::from_unweighted_edges(3, &[(0, 1), (1, 2)], false, false).unwrap();
    assert!(matches!(
        FruchtermanReingoldSolver::<CustomRiemannianRing>::new_unweighted(
            &g,
            FruchtermanReingoldConfig::new().with_backend(FrBackend::BarnesHut),
        ),
        Err(CoreError::UnsupportedOperation(_))
    ));

    let mut ring = CustomRiemannianRing {
        angles: vec![0.1, 0.2, 2.5],
    };
    let mut solver = FruchtermanReingoldSolver::<CustomRiemannianRing>::new_unweighted(
        &g,
        FruchtermanReingoldConfig::new()
            .with_iterations(10)
            .with_backend(FrBackend::Exact),
    )
    .unwrap();
    assert_eq!(solver.run(&mut ring).unwrap(), SolverStatus::Completed);
    assert!(ring.validate_finite().is_ok());
}
