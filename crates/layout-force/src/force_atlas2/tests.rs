use egraph_core::{
    CoreError, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Layout3d, LayoutNd, Scalar,
    SolverStatus,
};

use super::solver::MAX_FA2_DISPLACEMENT;
use super::{ForceAtlas2Config, ForceAtlas2Solver};

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

fn barbell_graph(k: usize) -> CsrGraph<f64> {
    let n = 2 * k;
    let mut edges = Vec::new();
    for i in 0..k {
        for j in (i + 1)..k {
            edges.push((i as u32, j as u32));
            edges.push(((k + i) as u32, (k + j) as u32));
        }
    }
    edges.push(((k - 1) as u32, k as u32));
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

fn cluster_separation_ratio<S: Scalar>(layout: &Layout2d<S>, k: usize) -> f64 {
    let mut c0x = 0.0;
    let mut c0y = 0.0;
    let mut c1x = 0.0;
    let mut c1y = 0.0;
    for i in 0..k {
        c0x += layout.x()[i].to_f64();
        c0y += layout.y()[i].to_f64();
        c1x += layout.x()[k + i].to_f64();
        c1y += layout.y()[k + i].to_f64();
    }
    c0x /= k as f64;
    c0y /= k as f64;
    c1x /= k as f64;
    c1y /= k as f64;
    let inter = (c0x - c1x).hypot(c0y - c1y);

    let mut intra = 0.0;
    for i in 0..k {
        intra += (layout.x()[i].to_f64() - c0x).hypot(layout.y()[i].to_f64() - c0y);
        intra += (layout.x()[k + i].to_f64() - c1x).hypot(layout.y()[k + i].to_f64() - c1y);
    }
    intra /= (2 * k) as f64;
    inter / intra.max(1e-12)
}

fn count_disk_overlaps<S: Scalar>(layout: &Layout2d<S>, radii: &[f64]) -> usize {
    let n = layout.len();
    let mut overlaps = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = (layout.x()[i].to_f64() - layout.x()[j].to_f64())
                .hypot(layout.y()[i].to_f64() - layout.y()[j].to_f64());
            if d < radii[i] + radii[j] - 1e-6 {
                overlaps += 1;
            }
        }
    }
    overlaps
}

#[test]
fn test_force_atlas2_config_defaults_builders_and_validation() {
    let cfg = ForceAtlas2Config::new();
    assert_eq!(cfg.iterations, 200);
    assert_eq!(cfg.theta, 1.2);
    assert!(cfg.use_barnes_hut);
    assert!(!cfg.linlog_mode);
    assert!(!cfg.strong_gravity);
    assert_eq!(cfg.gravity, 1.0);
    assert_eq!(cfg.scaling_ratio, 2.0);
    assert_eq!(cfg.edge_weight_influence, 1.0);
    assert!(!cfg.outbound_attraction_distribution);
    assert_eq!(cfg.jitter_tolerance, 1.0);
    assert_eq!(cfg.max_displacement, None);
    assert!(!cfg.prevent_overlap);
    assert_eq!(cfg.default_node_radius, 1.0);
    assert_eq!(cfg.seed, 42);
    cfg.validate().unwrap();

    for bad in [
        cfg.clone().with_theta(-0.1),
        cfg.clone().with_gravity(f64::NAN),
        cfg.clone().with_scaling_ratio(-1.0),
        cfg.clone().with_edge_weight_influence(-0.5),
        cfg.clone().with_jitter_tolerance(0.0),
        cfg.clone().with_max_displacement(Some(0.0)),
        cfg.clone().with_max_displacement(Some(-5.0)),
        cfg.clone().with_max_displacement(Some(f64::NAN)),
        cfg.clone().with_max_displacement(Some(f64::INFINITY)),
        cfg.clone().with_default_node_radius(-0.1),
        cfg.clone().with_default_node_radius(f64::INFINITY),
    ] {
        assert!(bad.validate().is_err());
    }
}

#[test]
fn test_force_atlas2_2d_barnes_hut_vs_exact_and_modes() {
    let graph = barbell_graph(6);
    let init = seeded_layout::<f64>(graph.num_nodes(), 7);

    let mut l_exact = init.clone();
    let mut l_bh_zero = init.clone();
    ForceAtlas2Solver::<Layout2d<f64>>::new(
        &graph,
        ForceAtlas2Config::new()
            .with_iterations(20)
            .with_use_barnes_hut(false),
    )
    .unwrap()
    .run(&mut l_exact)
    .unwrap();
    ForceAtlas2Solver::<Layout2d<f64>>::new(
        &graph,
        ForceAtlas2Config::new()
            .with_iterations(20)
            .with_theta(0.0)
            .with_use_barnes_hut(true),
    )
    .unwrap()
    .run(&mut l_bh_zero)
    .unwrap();
    for i in 0..graph.num_nodes() {
        assert!((l_exact.x()[i] - l_bh_zero.x()[i]).abs() < 1e-8);
        assert!((l_exact.y()[i] - l_bh_zero.y()[i]).abs() < 1e-8);
    }

    for cfg in [
        ForceAtlas2Config::new()
            .with_iterations(150)
            .with_use_barnes_hut(true),
        ForceAtlas2Config::new()
            .with_iterations(150)
            .with_linlog_mode(true)
            .with_strong_gravity(true)
            .with_gravity(0.5)
            .with_outbound_attraction_distribution(true),
    ] {
        let mut layout = init.clone();
        let mut solver = ForceAtlas2Solver::<Layout2d<f64>>::new(&graph, cfg).unwrap();
        assert_eq!(solver.run(&mut layout).unwrap(), SolverStatus::Completed);
        assert!(layout.is_all_finite());
        assert!(cluster_separation_ratio(&layout, 6) > 1.5);
    }
}

#[test]
fn test_force_atlas2_3d_octree_vs_exact_and_nd_fail_fast() {
    let graph = grid_graph(3, 3);
    let init3 = Layout3d::<f64>::init(9, 77, 4.0).unwrap();
    let mut l_exact = init3.clone();
    let mut l_bh0 = init3.clone();
    ForceAtlas2Solver::<Layout3d<f64>>::new(
        &graph,
        ForceAtlas2Config::new()
            .with_iterations(15)
            .with_use_barnes_hut(false),
    )
    .unwrap()
    .run(&mut l_exact)
    .unwrap();
    ForceAtlas2Solver::<Layout3d<f64>>::new(
        &graph,
        ForceAtlas2Config::new()
            .with_iterations(15)
            .with_use_barnes_hut(true)
            .with_theta(0.0),
    )
    .unwrap()
    .run(&mut l_bh0)
    .unwrap();
    for i in 0..9 {
        assert!((l_exact.x()[i] - l_bh0.x()[i]).abs() < 1e-8);
        assert!((l_exact.y()[i] - l_bh0.y()[i]).abs() < 1e-8);
        assert!((l_exact.z()[i] - l_bh0.z()[i]).abs() < 1e-8);
    }

    // LayoutNd fails fast at construction when Barnes-Hut is enabled.
    let err = ForceAtlas2Solver::<LayoutNd<f64>>::new(&graph, ForceAtlas2Config::default());
    assert!(matches!(err, Err(CoreError::UnsupportedOperation(_))));

    // LayoutNd succeeds with Exact (use_barnes_hut = false) and unweighted CsrGraph<()>.
    let unit_graph =
        CsrGraph::<()>::from_unweighted_csr(graph.indptr(), graph.indices(), false, false).unwrap();
    let mut l_nd = LayoutNd::<f32>::init(9, 4, 19, 2.0).unwrap();
    let mut s_nd = ForceAtlas2Solver::<LayoutNd<f32>>::new_unweighted(
        &unit_graph,
        ForceAtlas2Config::new()
            .with_iterations(10)
            .with_use_barnes_hut(false),
    )
    .unwrap();
    assert_eq!(s_nd.total_iterations(), Some(10));
    assert_eq!(s_nd.run(&mut l_nd).unwrap(), SolverStatus::Completed);
    assert!(l_nd.is_all_finite());
}

#[test]
fn test_force_atlas2_step_vs_run_overlap_and_v6_contract() {
    let grid = grid_graph(4, 4);
    let radii: Vec<f64> = (0..16).map(|i| 0.8 + 0.1 * ((i % 4) as f64)).collect();
    let mut init_overlap = seeded_layout::<f64>(16, 42);
    for i in 0..16 {
        init_overlap.x_mut()[i] *= 0.25;
        init_overlap.y_mut()[i] *= 0.25;
    }

    for use_bh in [false, true] {
        let mut layout = init_overlap.clone();
        let cfg = ForceAtlas2Config::new()
            .with_iterations(300)
            .with_use_barnes_hut(use_bh)
            .with_prevent_overlap(true)
            .with_linlog_mode(use_bh);
        let mut solver =
            ForceAtlas2Solver::<Layout2d<f64>>::with_node_radii(&grid, cfg, &radii).unwrap();
        assert_eq!(solver.node_radii(), radii.as_slice());
        assert_eq!(solver.run(&mut layout).unwrap(), SolverStatus::Completed);
        assert_eq!(count_disk_overlaps(&layout, &radii), 0);
    }

    // Scale-adaptive max_displacement check.
    let small = ForceAtlas2Solver::<Layout2d<f64>>::new(&grid, ForceAtlas2Config::new()).unwrap();
    assert_eq!(small.max_displacement(), MAX_FA2_DISPLACEMENT);

    // Zero-step target validation leaves solver state untouched.
    let mut solver =
        ForceAtlas2Solver::<Layout2d<f64>>::new(&grid, ForceAtlas2Config::new().with_iterations(5))
            .unwrap();
    let mut wrong = Layout2d::<f64>::zeros(4);
    assert!(matches!(
        solver.run_steps(&mut wrong, 0),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert_eq!(solver.status(), SolverStatus::Initialized);
    assert_eq!(solver.iteration(), 0);

    let mut nan_l = Layout2d::from_xy(vec![f64::NAN; 16], vec![0.0; 16]).unwrap();
    assert!(matches!(
        solver.run_steps(&mut nan_l, 0),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert_eq!(solver.status(), SolverStatus::Initialized);

    // Overflow to non-finite f32 transitions solver to Failed.
    let edgeless = CsrGraph::<()>::from_unweighted_edges(2, &[], false, false).unwrap();
    let mut overflow_solver = ForceAtlas2Solver::<Layout2d<f32>>::new_unweighted(
        &edgeless,
        ForceAtlas2Config::new()
            .with_iterations(5)
            .with_use_barnes_hut(false)
            .with_gravity(0.0)
            .with_scaling_ratio(1e120)
            .with_max_displacement(Some(1e38)),
    )
    .unwrap();
    let mut huge_l = Layout2d::<f32>::from_xy(vec![-3.4e38, -2.4e38], vec![0.0, 0.0]).unwrap();
    let res = overflow_solver.step(&mut huge_l);
    assert!(matches!(res, Err(CoreError::NumericalBreakdown(_))));
    assert_eq!(overflow_solver.status(), SolverStatus::Failed);
}
