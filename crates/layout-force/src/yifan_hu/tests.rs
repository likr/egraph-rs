use egraph_core::{
    CoreError, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Layout3d, LayoutNd, Scalar,
    SolverStatus,
};

use super::{YifanHuConfig, YifanHuSolver};

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
        x.push(S::from_f64(2.0 * (rng.next_f64() - 0.5)));
        y.push(S::from_f64(2.0 * (rng.next_f64() - 0.5)));
    }
    Layout2d::from_xy(x, y).unwrap()
}

#[test]
fn test_yifan_hu_config_builders_and_validation() {
    let yh = YifanHuConfig::new();
    assert_eq!(yh.max_iterations_per_level, 200);
    assert_eq!(yh.step_size, 1.0);
    assert_eq!(yh.step_ratio, 0.9);
    assert_eq!(yh.relative_strength, 0.2);
    assert_eq!(yh.optimal_distance, None);
    assert_eq!(yh.theta, 1.2);
    assert_eq!(yh.min_coarsen_size, 30);
    assert_eq!(yh.min_coarsen_ratio, 0.75);
    assert_eq!(yh.convergence_tolerance, 1e-4);
    assert_eq!(yh.seed, 42);
    yh.validate().unwrap();

    for bad in [
        yh.clone().with_step_size(0.0),
        yh.clone().with_step_ratio(1.0),
        yh.clone().with_step_ratio(0.0),
        yh.clone().with_relative_strength(-0.1),
        yh.clone().with_optimal_distance(Some(0.0)),
        yh.clone().with_theta(-0.1),
        yh.clone().with_min_coarsen_size(0),
        yh.clone().with_min_coarsen_ratio(0.0),
        yh.clone().with_min_coarsen_ratio(1.1),
        yh.clone().with_convergence_tolerance(-1e-4),
    ] {
        assert!(bad.validate().is_err());
    }
}

#[test]
fn test_yifan_hu_2d_3d_nd_and_unweighted_csr() {
    let grid = grid_graph(15, 15);
    let cfg = YifanHuConfig::new()
        .with_max_iterations_per_level(30)
        .with_min_coarsen_size(20);

    // 2D multilevel bit-exact step vs run.
    let mut yh_run = YifanHuSolver::<Layout2d<f64>>::new(&grid, cfg.clone()).unwrap();
    let mut yh_step = YifanHuSolver::<Layout2d<f64>>::new(&grid, cfg.clone()).unwrap();
    assert!(yh_run.num_levels() >= 3);
    assert_eq!(yh_run.total_iterations(), Some(yh_run.num_levels() * 30));

    let mut l2_run = Layout2d::<f64>::zeros(225);
    let mut l2_step = l2_run.clone();
    assert!(yh_run.run(&mut l2_run).unwrap().is_finished());
    while !yh_step.status().is_finished() {
        yh_step.step(&mut l2_step).unwrap();
    }
    for i in 0..225 {
        assert_eq!(l2_run.x()[i].to_bits(), l2_step.x()[i].to_bits());
        assert_eq!(l2_run.y()[i].to_bits(), l2_step.y()[i].to_bits());
    }

    // 3D Layout3d multilevel octree bit-exactness across f64 & f32.
    let g_small = grid_graph(6, 6);
    let cfg3 = YifanHuConfig::new()
        .with_max_iterations_per_level(15)
        .with_min_coarsen_size(10);
    let mut s3_run = YifanHuSolver::<Layout3d<f32>>::new(&g_small, cfg3.clone()).unwrap();
    let mut s3_step = YifanHuSolver::<Layout3d<f32>>::new(&g_small, cfg3).unwrap();
    let mut l3_run = Layout3d::<f32>::init(36, 404, 3.0).unwrap();
    let mut l3_step = l3_run.clone();
    assert!(s3_run.run(&mut l3_run).unwrap().is_finished());
    s3_step.run_steps(&mut l3_step, 5).unwrap();
    while !s3_step.status().is_finished() {
        s3_step.step(&mut l3_step).unwrap();
    }
    assert!(l3_run.is_all_finite());
    for i in 0..36 {
        assert_eq!(l3_run.x()[i].to_bits(), l3_step.x()[i].to_bits());
        assert_eq!(l3_run.y()[i].to_bits(), l3_step.y()[i].to_bits());
        assert_eq!(l3_run.z()[i].to_bits(), l3_step.z()[i].to_bits());
    }

    // N-D exact repulsion + unweighted CsrGraph<()> + zero-step validation.
    let unit_g =
        CsrGraph::<()>::from_unweighted_csr(g_small.indptr(), g_small.indices(), false, false)
            .unwrap();
    let mut s_nd = YifanHuSolver::<LayoutNd<f64>>::new_unweighted(
        &unit_g,
        YifanHuConfig::new()
            .with_max_iterations_per_level(10)
            .with_min_coarsen_size(10),
    )
    .unwrap();
    let mut wrong = LayoutNd::<f64>::zeros(4, 4).unwrap();
    assert!(matches!(
        s_nd.run_steps(&mut wrong, 0),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert_eq!(s_nd.status(), SolverStatus::Initialized);

    let mut l_nd = LayoutNd::<f64>::init(36, 4, 17, 2.0).unwrap();
    assert_eq!(
        s_nd.run_steps(&mut l_nd, 0).unwrap(),
        SolverStatus::Initialized
    );
    assert!(s_nd.run(&mut l_nd).unwrap().is_finished());
    assert!(l_nd.is_all_finite());

    let mut l_2d_check = seeded_layout::<f64>(36, 99);
    let mut s_unit_2d = YifanHuSolver::<Layout2d<f64>>::new_unweighted(
        &unit_g,
        YifanHuConfig::new().with_max_iterations_per_level(0),
    )
    .unwrap();
    assert_eq!(s_unit_2d.status(), SolverStatus::Completed);
    assert_eq!(s_unit_2d.total_iterations(), Some(0));
    assert_eq!(
        s_unit_2d.run(&mut l_2d_check).unwrap(),
        SolverStatus::Completed
    );
}
