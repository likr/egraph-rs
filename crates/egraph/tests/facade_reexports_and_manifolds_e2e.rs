//! Opaque-box E2E tests through the `egraph` facade covering non-Euclidean
//! manifold radial pipelines, `RepulsionBackend` direct execution, and seam re-exports.

#[cfg(feature = "layout")]
use egraph::prelude::*;

#[cfg(feature = "layout")]
fn make_grid_csr(rows: usize, cols: usize) -> CsrGraph<f64> {
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
    CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap()
}

#[cfg(all(feature = "layout", feature = "metrics", feature = "spatial"))]
#[test]
fn workflow_non_euclidean_multi_manifold_radial_pipeline() {
    use egraph::layout::sgd::{SgdPairBuffer, SgdScheduleConfig, SgdSolver};
    use egraph::metrics::geodesic_distortion;
    use egraph::spatial::{hyperbolic_crossing_number, torus_crossing_number};

    let g = make_grid_csr(3, 3);
    let n = g.num_nodes();

    for kind in [
        ManifoldKind::HyperbolicPoincare,
        ManifoldKind::HyperbolicLorentz,
        ManifoldKind::Spherical,
        ManifoldKind::Torus2d,
    ] {
        let mut buf = SgdPairBuffer::<f64>::new();
        for (u, v) in g.edges() {
            buf.push_spring(u, v, 0.25, 1.0).unwrap();
        }
        buf.push_hinge(0, 8, 0.5, 0.5).unwrap();
        let sched = SgdScheduleConfig::new(8);
        let mut solver =
            SgdSolver::<ManifoldLayout2d<f64>>::from_pair_buffer(n, buf, sched, 19).unwrap();

        let mut ml = ManifoldLayout2d::<f64>::init(kind, n, 29, 0.3).unwrap();
        let mut ml_radial = ml.clone();
        assert_eq!(solver.run(&mut ml).unwrap(), SolverStatus::Completed);

        let d0 = ml_radial.distance(0, 1);
        ml_radial.move_pair_radial(0, 1, 0.02, 0.02).unwrap();
        let mut ml_toward = ManifoldLayout2d::<f64>::init(kind, n, 29, 0.3).unwrap();
        ml_toward
            .move_pair_toward(0, 1, d0 - 0.04, 0.5, 0.5)
            .unwrap();
        for idx in 0..n {
            assert!((ml_radial.x()[idx] - ml_toward.x()[idx]).abs() < 1e-11);
            assert!((ml_radial.y()[idx] - ml_toward.y()[idx]).abs() < 1e-11);
        }

        let gd = geodesic_distortion(&g, &ml).unwrap();
        assert!(gd.is_finite());
        if kind == ManifoldKind::Torus2d {
            let _ = torus_crossing_number(&g, &ml).unwrap();
        } else if kind == ManifoldKind::HyperbolicPoincare
            || kind == ManifoldKind::HyperbolicLorentz
        {
            let _ = hyperbolic_crossing_number(&g, &ml).unwrap();
        }
    }
}

#[cfg(feature = "layout")]
#[test]
fn facade_reexports_seam_types_and_repulsion_backend() {
    use egraph::core::RepulsionBackend;
    use egraph::layout::neighbor::Exact;
    use egraph::layout::{
        DefaultIbUmapBackend, IbUmapBackend2d, IbUmapConfig, IbUmapRepulsion, IbUmapSolver,
        OrtmannSparseProblem, OrtmannSparseStressConfig, SgdPairKind, SgdPairRecord,
        build_ortmann_sparse_problem,
    };

    struct FacadeScalarRepulsion;
    impl RepulsionBackend<Layout2d<f64>> for FacadeScalarRepulsion {
        fn accumulate_repulsion(
            &mut self,
            layout: &Layout2d<f64>,
            forces: &mut [f64],
        ) -> CoreResult<Option<f64>> {
            assert_eq!(forces.len(), layout.len() * 2);
            forces.fill(0.5);
            Ok(Some(3.0))
        }
    }

    fn assert_default_backend<L: DefaultIbUmapBackend>() {}
    fn assert_ib_repulsion<B: IbUmapRepulsion<Layout2d<f64>>>() {}
    assert_default_backend::<Layout2d<f64>>();
    assert_ib_repulsion::<IbUmapBackend2d>();

    let l2 = Layout2d::<f64>::zeros(3);
    let mut rep: Box<dyn RepulsionBackend<Layout2d<f64>>> = Box::new(FacadeScalarRepulsion);
    let mut forces = vec![0.0_f64; 6];
    assert_eq!(
        rep.accumulate_repulsion(&l2, &mut forces).unwrap(),
        Some(3.0)
    );
    assert!(forces.iter().all(|&v| v == 0.5));

    let g = make_grid_csr(2, 3);
    let prob: OrtmannSparseProblem<f64> = build_ortmann_sparse_problem::<f64, f64>(
        &g,
        OrtmannSparseStressConfig::new().with_num_pivots(2),
    )
    .unwrap();
    let first: &SgdPairRecord<f64> = &prob.two_sided[0];
    assert_eq!(first.kind, SgdPairKind::Spring);

    let mut ib = IbUmapSolver::<Layout2d<f64>, IbUmapBackend2d>::new_with_backend(
        &g,
        IbUmapConfig::new().with_iterations(2),
        IbUmapBackend2d::Exact(Exact::new()),
    )
    .unwrap();
    let mut l_ib = Layout2d::<f64>::init(6, 1, 1.0).unwrap();
    assert_eq!(ib.run(&mut l_ib).unwrap(), SolverStatus::Completed);
}

#[cfg(all(feature = "layout", feature = "baselines"))]
#[test]
fn workflow_maxent_repulsion_backend_and_baselines_smoke() {
    use egraph::core::RepulsionBackend;
    use egraph::layout::{MaxentStressConfig, MaxentStressSolver};

    let g = make_grid_csr(3, 3);
    let mut solver =
        MaxentStressSolver::<Layout2d<f64>>::new(&g, MaxentStressConfig::default()).unwrap();
    let mut layout = Layout2d::<f64>::init(9, 5, 1.0).unwrap();
    assert!(solver.run(&mut layout).unwrap().is_finished());
    assert!(layout.is_all_finite());

    struct CountingRepulsion;
    impl RepulsionBackend<Layout2d<f64>> for CountingRepulsion {
        fn accumulate_repulsion(
            &mut self,
            _layout: &Layout2d<f64>,
            forces: &mut [f64],
        ) -> CoreResult<Option<f64>> {
            forces.fill(1.0);
            Ok(None)
        }
    }
    let mut cr = CountingRepulsion;
    let mut buf = vec![0.0; 18];
    assert_eq!(cr.accumulate_repulsion(&layout, &mut buf).unwrap(), None);
    assert!(buf.iter().all(|&f| f == 1.0));
}
