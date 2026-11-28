//! Opaque-box E2E tests for `ForceAtlas2Solver::{edges, masses}`, `new_unweighted`,
//! `with_node_radii`, multi-space (`Layout2d`/`Layout3d`/`LayoutNd`, `f64`/`f32`),
//! and fail-fast validation.

use egraph_core::{
    CoreError, CsrGraph, IterativeSolver, Layout2d, Layout3d, LayoutNd, SolverStatus,
};
use egraph_layout_force::{BarnesHut, Exact, ForceAtlas2Config, ForceAtlas2Solver};

fn make_grid(rows: usize, cols: usize) -> CsrGraph<f64> {
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

#[test]
fn force_atlas2_edges_masses_and_outbound_attraction_formula() {
    let g = make_grid(3, 3);
    let n = g.num_nodes();

    let cfg_std = ForceAtlas2Config::new()
        .with_outbound_attraction_distribution(false)
        .with_iterations(4);
    let cfg_hub = ForceAtlas2Config::new()
        .with_outbound_attraction_distribution(true)
        .with_iterations(4);

    let s_default = ForceAtlas2Solver::<Layout2d<f64>>::new(&g, cfg_std).unwrap();
    let s_exact =
        ForceAtlas2Solver::<Layout2d<f64>, Exact>::with_backend(&g, cfg_hub, Exact::new()).unwrap();

    let (u1, v1, w1) = s_default.edges();
    assert_eq!(u1.len(), g.num_edges());
    assert_eq!(v1.len(), g.num_edges());
    assert_eq!(w1.len(), g.num_edges());
    assert!(u1.iter().all(|&x| (x as usize) < n));
    assert!(v1.iter().all(|&x| (x as usize) < n));
    assert!(w1.iter().all(|&x| (x - 1.0).abs() < 1e-12));

    let masses = s_default.masses();
    assert_eq!(masses.len(), n);
    for (node, &m) in masses.iter().enumerate() {
        assert!((m - ((g.degree(node) as f64) + 1.0)).abs() < 1e-12);
        assert!(m >= 1.0);
    }

    let (u2, v2, w2) = s_exact.edges();
    assert_eq!(u1, u2);
    assert_eq!(v1, v2);
    assert_eq!(s_exact.masses(), masses);

    let mean_mass = masses.iter().sum::<f64>() / (n as f64);
    for idx in 0..u2.len() {
        let m_u = masses[u2[idx] as usize];
        let m_v = masses[v2[idx] as usize];
        let expected = w1[idx] * mean_mass * 0.5 * (1.0 / m_u + 1.0 / m_v);
        assert!(
            (w2[idx] - expected).abs() < 1e-12,
            "edge {idx}: expected {expected}, got {}",
            w2[idx]
        );
    }

    // Weighted graph with non-unit edge_weight_influence + outbound_attraction_distribution.
    let gw = CsrGraph::<f64>::from_edges(
        4,
        &[(0, 1), (1, 2), (2, 3), (0, 2)],
        Some(&[2.0, 3.0, 4.0, 1.5]),
        false,
        false,
    )
    .unwrap();
    let influence = 1.5_f64;
    let s_w_out = ForceAtlas2Solver::<Layout2d<f64>, Exact>::with_backend(
        &gw,
        ForceAtlas2Config::new()
            .with_edge_weight_influence(influence)
            .with_outbound_attraction_distribution(true),
        Exact::new(),
    )
    .unwrap();
    let (uw, vw, ww) = s_w_out.edges();
    let mw = s_w_out.masses();
    let mean_mw = mw.iter().sum::<f64>() / 4.0;
    for idx in 0..uw.len() {
        let raw_w = gw.edge_weight(uw[idx] as usize, vw[idx] as usize).unwrap();
        let w_pow = raw_w.powf(influence);
        let expected =
            w_pow * mean_mw * 0.5 * (1.0 / mw[uw[idx] as usize] + 1.0 / mw[vw[idx] as usize]);
        assert!((ww[idx] - expected).abs() < 1e-12);
    }
}

#[test]
fn force_atlas2_edge_weight_influence_zero_vs_two() {
    let g = CsrGraph::<f64>::from_edges(
        4,
        &[(0, 1), (1, 2), (2, 3)],
        Some(&[2.0, 3.0, 4.0]),
        false,
        false,
    )
    .unwrap();

    let s_zero = ForceAtlas2Solver::<Layout2d<f64>, BarnesHut>::new(
        &g,
        ForceAtlas2Config::new().with_edge_weight_influence(0.0),
    )
    .unwrap();
    let s_two = ForceAtlas2Solver::<Layout2d<f64>, BarnesHut>::new(
        &g,
        ForceAtlas2Config::new().with_edge_weight_influence(2.0),
    )
    .unwrap();

    assert_eq!(s_zero.edges().2, &[1.0, 1.0, 1.0]);
    let w_two = s_two.edges().2;
    assert!((w_two[0] - 4.0).abs() < 1e-12);
    assert!((w_two[1] - 9.0).abs() < 1e-12);
    assert!((w_two[2] - 16.0).abs() < 1e-12);
}

#[test]
fn force_atlas2_new_unweighted_with_node_radii_and_spaces_f64_f32() {
    let gw = CsrGraph::<f64>::from_edges(
        4,
        &[(0, 1), (1, 2), (2, 3), (3, 0)],
        Some(&[2.5, 5.0, 7.5, 10.0]),
        false,
        false,
    )
    .unwrap();
    let gu =
        CsrGraph::<()>::from_unweighted_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)], false, false)
            .unwrap();
    let radii = [0.5, 1.0, 1.5, 2.0];
    let cfg = ForceAtlas2Config::new()
        .with_iterations(4)
        .with_prevent_overlap(true);

    let s_unw_from_w =
        ForceAtlas2Solver::<Layout2d<f64>>::new_unweighted(&gw, cfg.clone()).unwrap();
    let s_unw_unit = ForceAtlas2Solver::<Layout2d<f64>>::new_unweighted(&gu, cfg.clone()).unwrap();
    assert_eq!(s_unw_from_w.edges(), s_unw_unit.edges());
    assert_eq!(s_unw_from_w.masses(), s_unw_unit.masses());
    assert_eq!(s_unw_from_w.edges().2, &[1.0, 1.0, 1.0, 1.0]);

    let s_radii =
        ForceAtlas2Solver::<Layout2d<f64>>::with_node_radii(&gw, cfg.clone(), &radii).unwrap();
    assert_eq!(s_radii.node_radii(), &radii);

    let mut s3_64 = ForceAtlas2Solver::<Layout3d<f64>>::new(&gw, cfg.clone()).unwrap();
    let mut s3_32 = ForceAtlas2Solver::<Layout3d<f32>>::new(&gw, cfg.clone()).unwrap();
    let mut snd_64 = ForceAtlas2Solver::<LayoutNd<f64>>::new(&gw, cfg.clone()).unwrap();
    let mut snd_32 = ForceAtlas2Solver::<LayoutNd<f32>>::new(&gw, cfg).unwrap();

    assert_eq!(s3_64.edges(), s_radii.edges());
    assert_eq!(s3_32.edges(), s_radii.edges());
    assert_eq!(snd_64.edges(), s_radii.edges());
    assert_eq!(snd_32.edges(), s_radii.edges());
    assert_eq!(s3_64.masses(), s_radii.masses());
    assert_eq!(snd_64.masses(), s_radii.masses());

    let mut l3_64 = Layout3d::<f64>::init(4, 1, 2.0).unwrap();
    let mut l3_32 = Layout3d::<f32>::init(4, 2, 2.0).unwrap();
    let mut lnd_64 = LayoutNd::<f64>::init(4, 5, 3, 2.0).unwrap();
    let mut lnd_32 = LayoutNd::<f32>::init(4, 5, 4, 2.0).unwrap();
    assert_eq!(s3_64.run(&mut l3_64).unwrap(), SolverStatus::Completed);
    assert_eq!(s3_32.run(&mut l3_32).unwrap(), SolverStatus::Completed);
    assert_eq!(snd_64.run(&mut lnd_64).unwrap(), SolverStatus::Completed);
    assert_eq!(snd_32.run(&mut lnd_32).unwrap(), SolverStatus::Completed);
    assert!(l3_64.is_all_finite() && l3_32.is_all_finite());
    assert!(lnd_64.is_all_finite() && lnd_32.is_all_finite());
}

#[test]
fn force_atlas2_empty_singleton_and_isolated_node_ir() {
    let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let s0 = ForceAtlas2Solver::<Layout2d<f64>>::new(&g0, ForceAtlas2Config::new()).unwrap();
    assert!(s0.edges().0.is_empty());
    assert!(s0.edges().1.is_empty());
    assert!(s0.edges().2.is_empty());
    assert!(s0.masses().is_empty());

    let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let s1 = ForceAtlas2Solver::<Layout2d<f64>>::new(&g1, ForceAtlas2Config::new()).unwrap();
    assert!(s1.edges().0.is_empty());
    assert_eq!(s1.masses(), &[1.0]);

    // Node 2 is isolated.
    let g_iso = CsrGraph::<f64>::from_edges(3, &[(0, 1)], None, false, false).unwrap();
    let mut s_iso = ForceAtlas2Solver::<Layout2d<f64>, Exact>::with_backend(
        &g_iso,
        ForceAtlas2Config::new().with_iterations(3),
        Exact::new(),
    )
    .unwrap();
    assert_eq!(s_iso.masses(), &[2.0, 2.0, 1.0]);
    let mut layout = Layout2d::<f64>::init(3, 5, 1.0).unwrap();
    assert_eq!(s_iso.run(&mut layout).unwrap(), SolverStatus::Completed);
    assert!(layout.is_all_finite());
}

#[test]
fn force_atlas2_ir_pointers_remain_stable_across_steps() {
    let g = make_grid(3, 3);
    let mut solver =
        ForceAtlas2Solver::<Layout2d<f64>>::new(&g, ForceAtlas2Config::new().with_iterations(4))
            .unwrap();
    let ptr_u = solver.edges().0.as_ptr();
    let ptr_m = solver.masses().as_ptr();
    let mut layout = Layout2d::<f64>::init(9, 17, 2.0).unwrap();
    assert_eq!(solver.run(&mut layout).unwrap(), SolverStatus::Completed);
    assert_eq!(solver.edges().0.as_ptr(), ptr_u);
    assert_eq!(solver.masses().as_ptr(), ptr_m);
}

#[test]
fn force_atlas2_invalid_configs_radii_weights_and_targets_fail_fast() {
    let g = make_grid(2, 2);

    for bad_cfg in [
        ForceAtlas2Config::new().with_gravity(-0.1),
        ForceAtlas2Config::new().with_gravity(f64::NAN),
        ForceAtlas2Config::new().with_scaling_ratio(-1.0),
        ForceAtlas2Config::new().with_edge_weight_influence(-0.5),
        ForceAtlas2Config::new().with_jitter_tolerance(0.0),
        ForceAtlas2Config::new().with_max_displacement(Some(0.0)),
        ForceAtlas2Config::new().with_default_node_radius(-1.0),
    ] {
        assert!(matches!(
            ForceAtlas2Solver::<Layout2d<f64>>::new(&g, bad_cfg),
            Err(CoreError::InvalidInput(_))
        ));
    }

    assert!(matches!(
        ForceAtlas2Solver::<Layout2d<f64>, BarnesHut>::with_backend(
            &g,
            ForceAtlas2Config::new(),
            BarnesHut::new(-0.1),
        ),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        ForceAtlas2Solver::<Layout2d<f64>, BarnesHut>::with_backend(
            &g,
            ForceAtlas2Config::new(),
            BarnesHut::new(f64::NAN),
        ),
        Err(CoreError::InvalidInput(_))
    ));

    assert!(matches!(
        ForceAtlas2Solver::<Layout2d<f64>>::with_node_radii(
            &g,
            ForceAtlas2Config::new(),
            &[1.0, 2.0],
        ),
        Err(CoreError::DimensionMismatch {
            expected: 4,
            actual: 2,
            ..
        })
    ));
    assert!(matches!(
        ForceAtlas2Solver::<Layout2d<f64>>::with_node_radii(
            &g,
            ForceAtlas2Config::new(),
            &[1.0, -0.5, 1.0, 1.0],
        ),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        ForceAtlas2Solver::<Layout2d<f64>>::with_node_radii(
            &g,
            ForceAtlas2Config::new(),
            &[1.0, f64::NAN, 1.0, 1.0],
        ),
        Err(CoreError::NonFiniteValue { .. })
    ));

    let g_neg = CsrGraph::<f64>::from_edges(2, &[(0, 1)], Some(&[-1.0]), false, false).unwrap();
    assert!(matches!(
        ForceAtlas2Solver::<Layout2d<f64>>::new(&g_neg, ForceAtlas2Config::new()),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        CsrGraph::<f64>::from_edges(2, &[(0, 1)], Some(&[f64::NAN]), false, false),
        Err(CoreError::NonFiniteValue { .. })
    ));

    let mut solver =
        ForceAtlas2Solver::<Layout2d<f64>>::new(&g, ForceAtlas2Config::new().with_iterations(4))
            .unwrap();
    let mut wrong_len = Layout2d::<f64>::init(2, 5, 1.0).unwrap();
    let wrong_snap = wrong_len.clone();
    assert!(matches!(
        solver.run_steps(&mut wrong_len, 0),
        Err(CoreError::DimensionMismatch {
            expected: 4,
            actual: 2,
            ..
        })
    ));
    assert_eq!(wrong_len, wrong_snap);

    let mut nan_lay =
        Layout2d::<f64>::from_xy(vec![0.0, f64::NAN, 1.0, 2.0], vec![0.0; 4]).unwrap();
    let nan_x_bits: Vec<u64> = nan_lay.x().iter().map(|v| v.to_bits()).collect();
    assert!(matches!(
        solver.step(&mut nan_lay),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert_eq!(
        nan_lay.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        nan_x_bits
    );
    assert_eq!(solver.iteration(), 0);
}
