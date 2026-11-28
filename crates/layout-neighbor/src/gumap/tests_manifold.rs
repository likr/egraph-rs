use egraph_core::{
    CoreError, CsrGraph, IterativeSolver, ManifoldKind, ManifoldLayout2d, Scalar, SolverStatus,
};
use egraph_neighborhood::DogVaradhanConfig;

use super::{GuMapConfig, GuMapSolver};

fn barbell_graph(clique_size: usize) -> CsrGraph<f64> {
    let n = 2 * clique_size;
    let mut edges = Vec::new();
    for c in 0..2 {
        let off = (c * clique_size) as u32;
        for i in 0..clique_size as u32 {
            for j in (i + 1)..clique_size as u32 {
                edges.push((off + i, off + j));
            }
        }
    }
    edges.push((0, clique_size as u32));
    CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
}

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
        ManifoldKind::HyperbolicPoincare => 0.4,
        ManifoldKind::HyperbolicLorentz => 0.7,
        ManifoldKind::Spherical | ManifoldKind::Torus2d => 1.0,
        _ => 0.5,
    };
    ManifoldLayout2d::<S>::init(kind, n, seed, init_scale).unwrap()
}

#[test]
fn test_gumap_manifold_all_kinds_bit_exact_and_separation() {
    let grid = grid_graph(4, 4);
    let n = grid.num_nodes();
    let cfg = GuMapConfig::new().with_iterations(15).with_seed(73);
    let dog_cfg = DogVaradhanConfig::default();

    for kind in [
        ManifoldKind::HyperbolicPoincare,
        ManifoldKind::HyperbolicLorentz,
        ManifoldKind::Spherical,
        ManifoldKind::Torus2d,
    ] {
        // f64 bit-exact step == run.
        let mut s_run =
            GuMapSolver::<ManifoldLayout2d<f64>>::from_graph_dog(&grid, 32, &dog_cfg, cfg.clone())
                .unwrap();
        let mut s_step =
            GuMapSolver::<ManifoldLayout2d<f64>>::from_graph_dog(&grid, 32, &dog_cfg, cfg.clone())
                .unwrap();
        let mut l_run = init_manifold::<f64>(kind, n, 17);
        if kind == ManifoldKind::Torus2d {
            l_run.set_torus_extents(2.5, 1.5).unwrap();
        } else {
            l_run.set_scale(1.5).unwrap();
        }
        let mut l_step = l_run.clone();

        assert_eq!(s_run.run(&mut l_run).unwrap(), SolverStatus::Completed);
        assert!(l_run.validate_finite().is_ok());

        assert_eq!(
            s_step.run_steps(&mut l_step, 6).unwrap(),
            SolverStatus::Running
        );
        while !s_step.status().is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        assert_eq!(s_step.status(), SolverStatus::Completed);
        assert_eq!(s_run.iteration(), s_step.iteration());
        for i in 0..n {
            assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
            assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
        }

        // f32 bit-exact step == run.
        let mut s32_run =
            GuMapSolver::<ManifoldLayout2d<f32>>::from_graph_dog(&grid, 32, &dog_cfg, cfg.clone())
                .unwrap();
        let mut s32_step =
            GuMapSolver::<ManifoldLayout2d<f32>>::from_graph_dog(&grid, 32, &dog_cfg, cfg.clone())
                .unwrap();
        let mut l32_run = init_manifold::<f32>(kind, n, 17);
        let mut l32_step = l32_run.clone();
        assert_eq!(s32_run.run(&mut l32_run).unwrap(), SolverStatus::Completed);
        assert!(l32_run.validate_finite().is_ok());
        for _ in 0..15 {
            s32_step.step(&mut l32_step).unwrap();
        }
        for i in 0..n {
            assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
            assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
        }
    }

    // Barbell cluster separation from all-zero manifold layout on HyperbolicLorentz and Torus2d.
    let barbell = barbell_graph(6);
    let nb = barbell.num_nodes();
    let cfg_bb = GuMapConfig::new()
        .with_iterations(100)
        .with_negative_sample_rate(5)
        .with_learning_rate(0.5)
        .with_seed(42);
    for kind in [ManifoldKind::HyperbolicLorentz, ManifoldKind::Torus2d] {
        let mut solver =
            GuMapSolver::<ManifoldLayout2d<f64>>::new(&barbell, cfg_bb.clone()).unwrap();
        let mut layout = ManifoldLayout2d::<f64>::zeros(kind, nb);
        assert_eq!(solver.run(&mut layout).unwrap(), SolverStatus::Completed);
        assert!(layout.validate_finite().is_ok());

        let mut intra = 0.0;
        let mut intra_cnt = 0usize;
        for c in 0..2 {
            let off = c * 6;
            for i in 0..6 {
                for j in (i + 1)..6 {
                    intra += layout.geodesic_distance(off + i, off + j).unwrap();
                    intra_cnt += 1;
                }
            }
        }
        let mut inter = 0.0;
        let mut inter_cnt = 0usize;
        for i in 0..6 {
            for j in 6..12 {
                inter += layout.geodesic_distance(i, j).unwrap();
                inter_cnt += 1;
            }
        }
        let mean_intra = intra / (intra_cnt as f64);
        let mean_inter = inter / (inter_cnt as f64);
        assert!(
            mean_inter > mean_intra * 1.2,
            "{kind:?}: mean_inter={mean_inter}, mean_intra={mean_intra}"
        );
    }
}

#[test]
fn test_gumap_manifold_edge_cases_and_kind_lock() {
    let cfg = GuMapConfig::new().with_iterations(10);

    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let mut s0 = GuMapSolver::<ManifoldLayout2d<f64>>::new(&empty, cfg.clone()).unwrap();
    let mut l0 = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Spherical, 0);
    assert_eq!(s0.run(&mut l0).unwrap(), SolverStatus::Converged);

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let mut s1 = GuMapSolver::<ManifoldLayout2d<f64>>::new(&single, cfg.clone()).unwrap();
    let mut l1 = ManifoldLayout2d::from_xy(ManifoldKind::Torus2d, vec![0.2], vec![0.8]).unwrap();
    assert_eq!(s1.run(&mut l1).unwrap(), SolverStatus::Converged);
    assert_eq!(l1.x(), &[0.2]);
    assert_eq!(l1.y(), &[0.8]);

    let edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    let mut se = GuMapSolver::<ManifoldLayout2d<f64>>::new(&edgeless, cfg.clone()).unwrap();
    let mut le = init_manifold::<f64>(ManifoldKind::HyperbolicPoincare, 4, 9);
    let before = le.clone();
    assert_eq!(se.run(&mut le).unwrap(), SolverStatus::Converged);
    assert_eq!(le, before);

    // Non-symmetric negative sampling mode.
    let grid = grid_graph(3, 3);
    let mut s_asym = GuMapSolver::<ManifoldLayout2d<f64>>::new(
        &grid,
        cfg.clone()
            .with_symmetric_negative(false)
            .with_iterations(8),
    )
    .unwrap();
    let mut l_asym = init_manifold::<f64>(ManifoldKind::Spherical, 9, 3);
    assert_eq!(s_asym.run(&mut l_asym).unwrap(), SolverStatus::Completed);
    assert!(l_asym.validate_finite().is_ok());

    // Node mismatch & ManifoldKind lock rejections.
    let mut l_wrong = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Spherical, 4);
    assert!(matches!(
        s_asym.step(&mut l_wrong),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        s_asym.run_steps(&mut l_wrong, 0),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let mut l_poinc = init_manifold::<f64>(ManifoldKind::HyperbolicPoincare, 9, 2);
    let mut l_lor = init_manifold::<f64>(ManifoldKind::HyperbolicLorentz, 9, 2);
    let mut s_from_m = GuMapSolver::<ManifoldLayout2d<f64>>::new(&grid, cfg).unwrap();
    s_from_m.step(&mut l_poinc).unwrap();
    assert!(matches!(
        s_from_m.step(&mut l_lor),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        s_from_m.run_steps(&mut l_lor, 0),
        Err(CoreError::InvalidInput(_))
    ));
}
