use std::f64::consts::PI;

use egraph_core::{
    CoreError, CsrGraph, IterativeSolver, ManifoldKind, ManifoldLayout2d, SolverStatus,
};

use super::super::sgd::{Sgd2Config, Sgd2Solver, SgdScheduleConfig, SgdScheduleKind, SgdSolver};
use super::test_support::{grid_graph, manifold_pair_stress};

fn subdivided_icosahedron_csr() -> CsrGraph<f64> {
    let base_edges = [
        (0usize, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 1),
        (1, 6),
        (2, 6),
        (2, 7),
        (3, 7),
        (3, 8),
        (4, 8),
        (4, 9),
        (5, 9),
        (5, 10),
        (1, 10),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 6),
        (6, 11),
        (7, 11),
        (8, 11),
        (9, 11),
        (10, 11),
    ];
    let faces = [
        (0, 1, 2),
        (0, 2, 3),
        (0, 3, 4),
        (0, 4, 5),
        (0, 5, 1),
        (1, 6, 2),
        (2, 6, 7),
        (2, 7, 3),
        (3, 7, 8),
        (3, 8, 4),
        (4, 8, 9),
        (4, 9, 5),
        (5, 9, 10),
        (5, 10, 1),
        (1, 10, 6),
        (11, 7, 6),
        (11, 8, 7),
        (11, 9, 8),
        (11, 10, 9),
        (11, 6, 10),
    ];
    let mut edge_mid = std::collections::BTreeMap::new();
    let mut next_id = 12usize;
    for &(a, b) in &base_edges {
        let key = (a.min(b), a.max(b));
        edge_mid.entry(key).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
    }
    let mut edges = Vec::new();
    for &(a, b, c) in &faces {
        let mab = edge_mid[&(a.min(b), a.max(b))];
        let mbc = edge_mid[&(b.min(c), b.max(c))];
        let mca = edge_mid[&(c.min(a), c.max(a))];
        for (u, v) in [
            (a, mab),
            (mab, b),
            (b, mbc),
            (mbc, c),
            (c, mca),
            (mca, a),
            (mab, mbc),
            (mbc, mca),
            (mca, mab),
        ] {
            edges.push((u.min(v) as u32, u.max(v) as u32));
        }
    }
    edges.sort_unstable();
    edges.dedup();
    CsrGraph::<f64>::from_edges(next_id, &edges, None, false, false).unwrap()
}

#[test]
fn test_spherical_mu_1_icosahedral_convergence_and_max_mu_guard() {
    let g = subdivided_icosahedron_csr();
    let n = g.num_nodes();

    let default_sched = SgdScheduleConfig::new(20);
    assert_eq!(default_sched.max_mu, None);
    let sched_with_mu = default_sched.clone().with_max_mu(Some(0.5));
    assert_eq!(sched_with_mu.max_mu, Some(0.5));

    assert!(matches!(
        SgdSolver::<ManifoldLayout2d<f64>>::full(
            &g,
            default_sched.clone().with_max_mu(Some(f64::NAN)),
            42
        ),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        SgdSolver::<ManifoldLayout2d<f64>>::full(
            &g,
            default_sched.clone().with_max_mu(Some(0.0)),
            42
        ),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        SgdSolver::<ManifoldLayout2d<f64>>::full(&g, default_sched.with_max_mu(Some(-0.2)), 42),
        Err(CoreError::InvalidInput(_))
    ));

    let fast_sched = SgdScheduleConfig::new(2);
    let mut s_full = SgdSolver::<ManifoldLayout2d<f64>>::full(&g, fast_sched.clone(), 42).unwrap();
    let mut s_clamped =
        SgdSolver::<ManifoldLayout2d<f64>>::full(&g, fast_sched.with_max_mu(Some(0.1)), 42)
            .unwrap();
    assert!(s_full.current_eta() >= 1.0);

    let lons: Vec<f64> = (0..n)
        .map(|i| (i as f64) * std::f64::consts::TAU / (n as f64))
        .collect();
    let colats: Vec<f64> = (0..n)
        .map(|i| 0.05 + 0.25 * ((i as f64) / (n as f64)))
        .collect();
    let mut l_full =
        ManifoldLayout2d::<f64>::from_xy(ManifoldKind::Spherical, lons, colats).unwrap();
    l_full.set_scale(6.0 / PI).unwrap();
    let mut l_clamped = l_full.clone();
    s_full.run(&mut l_full).unwrap();
    s_clamped.run(&mut l_clamped).unwrap();

    let mut pairs = Vec::new();
    let mut bfs = egraph_linalg::BfsWorkspace::new(n);
    let mut row = vec![0.0_f64; n];
    for i in 0..n {
        bfs.bfs_from(&g, i, &mut row).unwrap();
        for (j, &d) in row.iter().enumerate().skip(i + 1) {
            pairs.push((i, j, d));
        }
    }
    let stress_full = manifold_pair_stress(&l_full, &pairs);
    let stress_clamped = manifold_pair_stress(&l_clamped, &pairs);
    assert!(
        stress_full < stress_clamped * 0.5,
        "expected un-clamped Spherical mu=1.0 stress ({stress_full}) << max_mu=0.1 ({stress_clamped})"
    );
}

#[test]
fn test_hyperbolic_lorentz_large_radii_without_disk_clamp() {
    let mut lorentz = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::HyperbolicLorentz,
        vec![-50.0, 50.0],
        vec![0.0, 0.0],
    )
    .unwrap();
    let sched = SgdScheduleConfig::new(15).with_kind(SgdScheduleKind::Constant);
    let mut solver = SgdSolver::<ManifoldLayout2d<f64>>::from_pairs(
        2,
        &[0],
        &[1],
        &[14.0],
        &[14.0],
        &[1.0],
        &[1.0],
        sched,
        42,
    )
    .unwrap();
    let st = solver.run(&mut lorentz).unwrap();
    assert_eq!(st, SolverStatus::Completed);
    assert!(lorentz.as_layout().is_all_finite());
    let d_lorentz = lorentz.geodesic_distance(0, 1).unwrap();
    assert!(
        (d_lorentz - 14.0).abs() < 1e-2,
        "expected Lorentz target distance ~14.0, got {d_lorentz}"
    );
    assert!(lorentz.x()[0].abs() > 1.0 && lorentz.x()[1].abs() > 1.0);
}

#[test]
fn test_manifold_metric_scaling_and_torus_rectangular_aspect_clamping() {
    let sched = SgdScheduleConfig::new(1).with_kind(SgdScheduleKind::Constant);
    let mut s1 = SgdSolver::<ManifoldLayout2d<f64>>::from_pairs(
        2,
        &[0],
        &[1],
        &[2000.0],
        &[2000.0],
        &[1.0],
        &[1.0],
        sched.clone(),
        42,
    )
    .unwrap();
    let mut s4 = s1.clone();
    let mut ml_s1 = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::HyperbolicLorentz,
        vec![-0.1, 0.1],
        vec![0.0, 0.0],
    )
    .unwrap();
    let mut ml_s4 = ml_s1.clone();
    ml_s4.set_scale(4.0).unwrap();
    s1.run(&mut ml_s1).unwrap();
    s4.run(&mut ml_s4).unwrap();
    let d_s1 = ml_s1.geodesic_distance(0, 1).unwrap();
    let d_s4 = ml_s4.geodesic_distance(0, 1).unwrap();
    assert!(
        d_s4 > d_s1,
        "expected scaled step d_s4 ({d_s4}) > d_s1 ({d_s1})"
    );

    let mut sp_s1 =
        ManifoldLayout2d::<f64>::from_xy(ManifoldKind::Spherical, vec![0.0, 0.2], vec![1.5, 1.5])
            .unwrap();
    let mut sp_s3 = sp_s1.clone();
    sp_s3.set_scale(3.0).unwrap();
    let mut sp_a = SgdSolver::<ManifoldLayout2d<f64>>::from_pairs(
        2,
        &[0],
        &[1],
        &[100.0],
        &[100.0],
        &[1.0],
        &[1.0],
        sched,
        42,
    )
    .unwrap();
    let mut sp_b = sp_a.clone();
    sp_a.step(&mut sp_s1).unwrap();
    sp_b.step(&mut sp_s3).unwrap();
    assert!(sp_s1.as_layout().is_all_finite() && sp_s3.as_layout().is_all_finite());
    assert!(sp_s3.geodesic_distance(0, 1).unwrap() <= 3.0 * PI + 1e-9);

    let mut torus =
        ManifoldLayout2d::<f64>::from_xy(ManifoldKind::Torus2d, vec![0.1, 0.2], vec![0.1, 0.1])
            .unwrap();
    torus.set_torus_extents(4.0, 1.0).unwrap();
    let mut tor_solver = SgdSolver::<ManifoldLayout2d<f64>>::from_pairs(
        2,
        &[0],
        &[1],
        &[1.8],
        &[1.8],
        &[1.0],
        &[1.0],
        SgdScheduleConfig::new(1).with_kind(SgdScheduleKind::Constant),
        42,
    )
    .unwrap();
    tor_solver.step(&mut torus).unwrap();
    let d_tor = torus.geodesic_distance(0, 1).unwrap();
    assert!(
        (d_tor - 1.8).abs() < 1e-10,
        "expected unclipped rectangular torus displacement reaching 1.8, got {d_tor}"
    );
}

#[test]
fn test_manifold_kind_lock_enforcement() {
    let g = grid_graph(2, 2);
    let sched = SgdScheduleConfig::new(5);

    let mut s_m = SgdSolver::<ManifoldLayout2d<f64>>::full(&g, sched, 42).unwrap();
    let mut ml_tor = ManifoldLayout2d::<f64>::init(ManifoldKind::Torus2d, 4, 42, 1.0).unwrap();
    let mut ml_sph = ManifoldLayout2d::<f64>::init(ManifoldKind::Spherical, 4, 42, 1.0).unwrap();

    s_m.step(&mut ml_tor).unwrap();
    assert!(matches!(
        s_m.step(&mut ml_sph),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        s_m.run_steps(&mut ml_sph, 0),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        s_m.run_schedule_window(&mut ml_sph, 0, 1),
        Err(CoreError::InvalidInput(_))
    ));

    let mut s2_m =
        Sgd2Solver::<ManifoldLayout2d<f64>>::new(&g, Sgd2Config::new().with_iterations(5)).unwrap();
    s2_m.step(&mut ml_tor).unwrap();
    assert!(matches!(
        s2_m.step(&mut ml_sph),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        s2_m.run_steps(&mut ml_sph, 0),
        Err(CoreError::InvalidInput(_))
    ));
}

#[test]
fn test_run_schedule_window_manifold_determinism() {
    let g = grid_graph(3, 3);
    let cfg = SgdScheduleConfig::new(10);
    let mut s1 = SgdSolver::<ManifoldLayout2d<f64>>::full(&g, cfg.clone(), 42).unwrap();
    let mut s2 = SgdSolver::<ManifoldLayout2d<f64>>::full(&g, cfg.clone(), 42).unwrap();

    let mut ml1 = ManifoldLayout2d::<f64>::init(ManifoldKind::Torus2d, 9, 42, 1.0).unwrap();
    let mut ml2 = ml1.clone();
    s1.run_schedule_window(&mut ml1, 0, 0).unwrap();
    assert_eq!(ml1, ml2);
    s1.run_schedule_window(&mut ml1, 20, 5).unwrap();
    assert_eq!(ml1, ml2);

    s1.reset_schedule(cfg.clone(), 99).unwrap();
    s2.reset_schedule(cfg, 99).unwrap();
    s1.run_schedule_window(&mut ml1, 0, 4).unwrap();
    s1.run_schedule_window(&mut ml1, 4, 6).unwrap();
    s2.run_schedule_window(&mut ml2, 0, 10).unwrap();
    assert_eq!(ml1, ml2);
}
