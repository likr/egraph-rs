use egraph_core::{CoreError, CsrGraph, IterativeSolver, Layout3d, LayoutNd, SolverStatus};

use super::{MaxentStressConfig, MaxentStressSolver};
use crate::{StressMajorizationConfig, StressMajorizationSolver};

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

#[test]
fn test_maxent_nd_step_vs_run_bit_exact_3d_5d() {
    let g = grid_graph(3, 4);

    // f64 dims=3 with LayoutNd (Exact repulsion)
    let cfg_exact = MaxentStressConfig::new()
        .with_use_barnes_hut(false)
        .with_stages(3)
        .with_steps_per_stage(10);
    let mut s1 = MaxentStressSolver::<LayoutNd<f64>>::new(&g, cfg_exact.clone()).unwrap();
    let mut s2 = MaxentStressSolver::<LayoutNd<f64>>::new(&g, cfg_exact.clone()).unwrap();
    let mut l1 = LayoutNd::zeros(g.num_nodes(), 3).unwrap();
    let mut l2 = LayoutNd::zeros(g.num_nodes(), 3).unwrap();

    let st1 = s1.run(&mut l1).unwrap();
    assert!(st1.is_finished());
    assert!(l1.is_all_finite());
    while !s2.status().is_finished() {
        s2.step(&mut l2).unwrap();
    }
    assert_eq!(st1, s2.status());
    assert_eq!(s1.iteration(), s2.iteration());
    assert_eq!(s1.total_iterations(), Some(30));
    for (a, b) in l1.coords().iter().zip(l2.coords().iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    // f32 dims=3 with LayoutNd (Exact repulsion)
    let cfg32 = MaxentStressConfig::new()
        .with_use_barnes_hut(false)
        .with_stages(3)
        .with_steps_per_stage(8);
    let mut s1 = MaxentStressSolver::<LayoutNd<f32>>::new(&g, cfg32.clone()).unwrap();
    let mut s2 = MaxentStressSolver::<LayoutNd<f32>>::new(&g, cfg32).unwrap();
    let mut l1 = LayoutNd::<f32>::zeros(g.num_nodes(), 3).unwrap();
    let mut l2 = LayoutNd::<f32>::zeros(g.num_nodes(), 3).unwrap();
    let st1 = s1.run(&mut l1).unwrap();
    while !s2.status().is_finished() {
        s2.step(&mut l2).unwrap();
    }
    assert_eq!(st1, s2.status());
    assert!(l1.is_all_finite());
    for (a, b) in l1.coords().iter().zip(l2.coords().iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    // f64 dims=5 with LayoutNd (Exact repulsion)
    let cfg5 = MaxentStressConfig::new()
        .with_use_barnes_hut(false)
        .with_stages(3)
        .with_steps_per_stage(8);
    let mut s5_1 = MaxentStressSolver::<LayoutNd<f64>>::new(&g, cfg5.clone()).unwrap();
    let mut s5_2 = MaxentStressSolver::<LayoutNd<f64>>::new(&g, cfg5).unwrap();
    let mut l5_1 = LayoutNd::zeros(g.num_nodes(), 5).unwrap();
    let mut l5_2 = LayoutNd::zeros(g.num_nodes(), 5).unwrap();
    let st5 = s5_1.run(&mut l5_1).unwrap();
    while !s5_2.status().is_finished() {
        s5_2.step(&mut l5_2).unwrap();
    }
    assert_eq!(st5, s5_2.status());
    for (a, b) in l5_1.coords().iter().zip(l5_2.coords().iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn test_maxent_3d_barnes_hut_stress_decrease_q_variants_and_exact_parity() {
    let g = grid_graph(3, 3);
    let stress_eval = StressMajorizationSolver::<Layout3d<f64>>::from_graph(
        &g,
        StressMajorizationConfig::default(),
    )
    .unwrap();
    let l_init = Layout3d::init(g.num_nodes(), 77, 10.0).unwrap();
    let stress_before = stress_eval.compute_stress(&l_init).unwrap();

    // Test q=0.0 and q=0.8 in 3D Barnes-Hut on Layout3d
    for &q_val in &[0.0, 0.8] {
        let mut l = l_init.clone();
        let mut solver = MaxentStressSolver::<Layout3d<f64>>::new(
            &g,
            MaxentStressConfig::new()
                .with_q(Some(q_val))
                .with_use_barnes_hut(true)
                .with_stages(4)
                .with_steps_per_stage(20),
        )
        .unwrap();
        solver.run(&mut l).unwrap();
        let stress_after = stress_eval.compute_stress(&l).unwrap();
        assert!(
            stress_after < stress_before * 0.25,
            "q={q_val}: before={stress_before}, after={stress_after}"
        );
    }

    // Exact vs Barnes-Hut theta=0.0 parity on Layout3d
    let mut l_bh = l_init.clone();
    let mut l_ex = l_init;
    let mut s_bh = MaxentStressSolver::<Layout3d<f64>>::new(
        &g,
        MaxentStressConfig::new()
            .with_use_barnes_hut(true)
            .with_theta(0.0)
            .with_stages(2)
            .with_steps_per_stage(5),
    )
    .unwrap();
    let mut s_ex = MaxentStressSolver::<Layout3d<f64>>::new(
        &g,
        MaxentStressConfig::new()
            .with_use_barnes_hut(false)
            .with_stages(2)
            .with_steps_per_stage(5),
    )
    .unwrap();
    s_bh.run(&mut l_bh).unwrap();
    s_ex.run(&mut l_ex).unwrap();
    for i in 0..g.num_nodes() {
        assert!((l_bh.x()[i] - l_ex.x()[i]).abs() < 1e-10);
        assert!((l_bh.y()[i] - l_ex.y()[i]).abs() < 1e-10);
        assert!((l_bh.z()[i] - l_ex.z()[i]).abs() < 1e-10);
    }
}

#[test]
fn test_maxent_nd_edge_cases_and_degenerate_geometries() {
    let cfg_exact = MaxentStressConfig::new().with_use_barnes_hut(false);

    // n=0 and n=1
    let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let mut s0 = MaxentStressSolver::<LayoutNd<f64>>::new(&g0, cfg_exact.clone()).unwrap();
    let mut l0 = LayoutNd::zeros(0, 3).unwrap();
    assert_eq!(s0.step(&mut l0).unwrap(), SolverStatus::Converged);

    let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let mut s1 = MaxentStressSolver::<LayoutNd<f64>>::new(&g1, cfg_exact.clone()).unwrap();
    let mut l1 = LayoutNd::zeros(1, 3).unwrap();
    assert_eq!(s1.step(&mut l1).unwrap(), SolverStatus::Converged);

    // Edgeless graph (n=4, dims=3)
    let g_edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    let mut s_edgeless =
        MaxentStressSolver::<LayoutNd<f64>>::new(&g_edgeless, cfg_exact.clone()).unwrap();
    let mut l_edgeless = LayoutNd::zeros(4, 3).unwrap();
    s_edgeless.run(&mut l_edgeless).unwrap();
    assert!(l_edgeless.is_all_finite());

    // Disconnected graph + isolated vertices
    let g_disc =
        CsrGraph::<f64>::from_edges(8, &[(0, 1), (1, 2), (3, 4), (4, 5)], None, false, false)
            .unwrap();
    let mut s_disc = MaxentStressSolver::<LayoutNd<f64>>::new(&g_disc, cfg_exact.clone()).unwrap();
    let mut l_disc = LayoutNd::from_flat(
        8,
        3,
        &[
            0.0, 0.0, 0.0, //
            1.0, 0.5, 0.2, //
            2.0, 0.0, -0.2, //
            5.0, 2.0, 1.0, //
            6.0, 2.5, 1.5, //
            7.0, 2.0, 1.0, //
            -3.0, 8.0, 4.0, //
            -4.0, -8.0, -2.0,
        ],
    )
    .unwrap();
    assert!(s_disc.run(&mut l_disc).unwrap().is_finished());
    assert!(l_disc.is_all_finite());
    assert_eq!(l_disc.node(6).unwrap(), &[-3.0, 8.0, 4.0]);
    assert_eq!(l_disc.node(7).unwrap(), &[-4.0, -8.0, -2.0]);

    // Coincident, collinear-in-3D, planar-in-3D, and dims=1 / dims=2 layouts
    let g3 = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, false, false).unwrap();
    let mut s_coin = MaxentStressSolver::<LayoutNd<f64>>::new(&g3, cfg_exact.clone()).unwrap();
    let mut l_coin = LayoutNd::zeros(3, 3).unwrap();
    assert!(s_coin.run(&mut l_coin).unwrap().is_finished());
    assert!(l_coin.is_all_finite());

    let mut s_col = MaxentStressSolver::<LayoutNd<f64>>::new(&g3, cfg_exact.clone()).unwrap();
    let mut l_col =
        LayoutNd::from_flat(3, 3, &[0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.8, 0.0, 0.0]).unwrap();
    assert!(s_col.run(&mut l_col).unwrap().is_finished());
    assert!(l_col.is_all_finite());

    let g4 = grid_graph(2, 2);
    let mut s_pln = MaxentStressSolver::<LayoutNd<f64>>::new(&g4, cfg_exact.clone()).unwrap();
    let mut l_pln = LayoutNd::from_flat(
        4,
        3,
        &[0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0],
    )
    .unwrap();
    assert!(s_pln.run(&mut l_pln).unwrap().is_finished());
    assert!(l_pln.is_all_finite());

    let mut s_1d = MaxentStressSolver::<LayoutNd<f64>>::new(&g4, cfg_exact).unwrap();
    let mut l_1d = LayoutNd::zeros(4, 1).unwrap();
    assert!(s_1d.run(&mut l_1d).unwrap().is_finished());
    assert!(l_1d.is_all_finite());
}

#[test]
fn test_maxent_nd_validation_errors_and_fail_fast() {
    let g = grid_graph(2, 2);

    assert!(matches!(
        LayoutNd::<f64>::zeros(4, 0),
        Err(CoreError::InvalidInput(_))
    ));

    // use_barnes_hut == true on LayoutNd fails fast at construction.
    assert!(matches!(
        MaxentStressSolver::<LayoutNd<f64>>::new(&g, MaxentStressConfig::default()),
        Err(CoreError::UnsupportedOperation(_))
    ));

    let cfg_exact = MaxentStressConfig::new().with_use_barnes_hut(false);
    let mut s_nd = MaxentStressSolver::<LayoutNd<f64>>::new(&g, cfg_exact.clone()).unwrap();

    // NaN / Inf rejected via both step and run_steps(target, 0).
    let mut l_nan = LayoutNd::from_flat(4, 3, &[f64::NAN; 12]).unwrap();
    assert!(matches!(
        s_nd.run_steps(&mut l_nan, 0),
        Err(CoreError::NonFiniteValue { .. })
    ));
    let mut l_inf = LayoutNd::from_flat(4, 3, &[f64::INFINITY; 12]).unwrap();
    assert!(matches!(
        s_nd.step(&mut l_inf),
        Err(CoreError::NonFiniteValue { .. })
    ));

    // Node count mismatch & mid-run dims mismatch (including run_steps(0)).
    let mut l_wrong_n = LayoutNd::<f64>::zeros(5, 3).unwrap();
    assert!(matches!(
        s_nd.run_steps(&mut l_wrong_n, 0),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let mut l3 = LayoutNd::<f64>::init(4, 3, 5, 1.0).unwrap();
    s_nd.step(&mut l3).unwrap();
    let mut l2 = LayoutNd::<f64>::init(4, 2, 5, 1.0).unwrap();
    assert!(matches!(
        s_nd.validate_target(&l2),
        Err(CoreError::DimensionMismatch {
            expected: 3,
            actual: 2,
            ..
        })
    ));
    assert!(matches!(
        s_nd.run_steps(&mut l2, 0),
        Err(CoreError::DimensionMismatch {
            expected: 3,
            actual: 2,
            ..
        })
    ));

    // Ensure f32 LayoutNd non-finite check also fires on run_steps(0).
    let mut s_nd32 = MaxentStressSolver::<LayoutNd<f32>>::new(&g, cfg_exact).unwrap();
    let mut l32_nan = LayoutNd::from_flat(4, 3, &[f32::NAN; 12]).unwrap();
    assert!(matches!(
        s_nd32.run_steps(&mut l32_nan, 0),
        Err(CoreError::NonFiniteValue { .. })
    ));
}
