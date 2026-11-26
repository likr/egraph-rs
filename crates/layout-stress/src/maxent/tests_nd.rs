use egraph_core::{CoreError, CsrGraph, IterativeSolver, LayoutNd, SolverStatus};

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

    // f64 dims=3 (both Barnes-Hut via ArenaOctree3d and exact O(N^2))
    for &use_bh in &[true, false] {
        let cfg = MaxentStressConfig {
            use_barnes_hut: use_bh,
            stages: 3,
            steps_per_stage: 10,
            ..Default::default()
        };
        let mut s1 = MaxentStressSolver::<f64>::new(&g, cfg).unwrap();
        let mut s2 = MaxentStressSolver::<f64>::new(&g, cfg).unwrap();
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
        assert_eq!(s1.total_iterations(), 30);
        for (a, b) in l1.coords().iter().zip(l2.coords().iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    // f32 dims=3 (Barnes-Hut)
    let cfg32 = MaxentStressConfig {
        use_barnes_hut: true,
        stages: 3,
        steps_per_stage: 8,
        ..Default::default()
    };
    let mut s1 = MaxentStressSolver::<f32>::new(&g, cfg32).unwrap();
    let mut s2 = MaxentStressSolver::<f32>::new(&g, cfg32).unwrap();
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

    // f64 dims=5 (exact O(N^2))
    let cfg5 = MaxentStressConfig {
        use_barnes_hut: false,
        stages: 3,
        steps_per_stage: 8,
        ..Default::default()
    };
    let mut s5_1 = MaxentStressSolver::<f64>::new(&g, cfg5).unwrap();
    let mut s5_2 = MaxentStressSolver::<f64>::new(&g, cfg5).unwrap();
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
fn test_maxent_nd_stress_decrease_and_q_variants() {
    let g = grid_graph(3, 3);
    let stress_eval =
        StressMajorizationSolver::<f64>::from_graph(&g, StressMajorizationConfig::default())
            .unwrap();
    let mut l_init = LayoutNd::init(g.num_nodes(), 3, 77, 10.0).unwrap();
    let stress_before = stress_eval.compute_stress_nd(&l_init).unwrap();

    // Test q=0.0 and q=0.8 in 3D Barnes-Hut
    for &q_val in &[0.0, 0.8] {
        let mut l = l_init.copy();
        let mut solver = MaxentStressSolver::<f64>::new(
            &g,
            MaxentStressConfig {
                q: Some(q_val),
                use_barnes_hut: true,
                stages: 4,
                steps_per_stage: 20,
                ..Default::default()
            },
        )
        .unwrap();
        solver.run(&mut l).unwrap();
        let stress_after = stress_eval.compute_stress_nd(&l).unwrap();
        assert!(
            stress_after < stress_before * 0.25,
            "q={q_val}: before={stress_before}, after={stress_after}"
        );
    }

    // Exact vs Barnes-Hut theta=0.0 parity in 3D
    let mut l_bh = l_init.copy();
    let mut l_ex = l_init.copy();
    let mut s_bh = MaxentStressSolver::<f64>::new(
        &g,
        MaxentStressConfig {
            use_barnes_hut: true,
            theta: 0.0,
            stages: 2,
            steps_per_stage: 5,
            ..Default::default()
        },
    )
    .unwrap();
    let mut s_ex = MaxentStressSolver::<f64>::new(
        &g,
        MaxentStressConfig {
            use_barnes_hut: false,
            stages: 2,
            steps_per_stage: 5,
            ..Default::default()
        },
    )
    .unwrap();
    s_bh.run(&mut l_bh).unwrap();
    s_ex.run(&mut l_ex).unwrap();
    for (a, b) in l_bh.coords().iter().zip(l_ex.coords().iter()) {
        assert!((a - b).abs() < 1e-10, "a={a}, b={b}");
    }
    let _ = &mut l_init;
}

#[test]
fn test_maxent_nd_edge_cases_and_degenerate_geometries() {
    let cfg3 = MaxentStressConfig::default();

    // n=0 and n=1
    let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let mut s0 = MaxentStressSolver::<f64>::new(&g0, cfg3).unwrap();
    let mut l0 = LayoutNd::zeros(0, 3).unwrap();
    assert_eq!(s0.step(&mut l0).unwrap(), SolverStatus::Converged);

    let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let mut s1 = MaxentStressSolver::<f64>::new(&g1, cfg3).unwrap();
    let mut l1 = LayoutNd::zeros(1, 3).unwrap();
    assert_eq!(s1.step(&mut l1).unwrap(), SolverStatus::Converged);

    // Edgeless graph (n=4, dims=3)
    let g_edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    let mut s_edgeless = MaxentStressSolver::<f64>::new(&g_edgeless, cfg3).unwrap();
    let mut l_edgeless = LayoutNd::zeros(4, 3).unwrap();
    s_edgeless.run(&mut l_edgeless).unwrap();
    assert!(l_edgeless.is_all_finite());

    // Disconnected graph + isolated vertices
    let g_disc =
        CsrGraph::<f64>::from_edges(8, &[(0, 1), (1, 2), (3, 4), (4, 5)], None, false, false)
            .unwrap();
    let mut s_disc = MaxentStressSolver::<f64>::new(&g_disc, cfg3).unwrap();
    let mut l_disc = LayoutNd::from_flat(
        8,
        3,
        vec![
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

    // Coincident, collinear-in-3D, and planar-in-3D layouts
    let g3 = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, false, false).unwrap();
    let mut s_coin = MaxentStressSolver::<f64>::new(&g3, cfg3).unwrap();
    let mut l_coin = LayoutNd::zeros(3, 3).unwrap();
    assert!(s_coin.run(&mut l_coin).unwrap().is_finished());
    assert!(l_coin.is_all_finite());

    let mut s_col = MaxentStressSolver::<f64>::new(&g3, cfg3).unwrap();
    let mut l_col =
        LayoutNd::from_flat(3, 3, vec![0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.8, 0.0, 0.0]).unwrap();
    assert!(s_col.run(&mut l_col).unwrap().is_finished());
    assert!(l_col.is_all_finite());

    let g4 = grid_graph(2, 2);
    let mut s_pln = MaxentStressSolver::<f64>::new(&g4, cfg3).unwrap();
    let mut l_pln = LayoutNd::from_flat(
        4,
        3,
        vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0],
    )
    .unwrap();
    assert!(s_pln.run(&mut l_pln).unwrap().is_finished());
    assert!(l_pln.is_all_finite());

    // dims=2 via LayoutNd with Barnes-Hut also supported
    let mut s_2d_nd = MaxentStressSolver::<f64>::new(&g4, cfg3).unwrap();
    let mut l_2d_nd = LayoutNd::zeros(4, 2).unwrap();
    assert!(s_2d_nd.run(&mut l_2d_nd).unwrap().is_finished());
    assert!(l_2d_nd.is_all_finite());

    // dims=1 with exact repulsion
    let mut s_1d = MaxentStressSolver::<f64>::new(
        &g4,
        MaxentStressConfig {
            use_barnes_hut: false,
            ..Default::default()
        },
    )
    .unwrap();
    let mut l_1d = LayoutNd::zeros(4, 1).unwrap();
    assert!(s_1d.run(&mut l_1d).unwrap().is_finished());
    assert!(l_1d.is_all_finite());
}

#[test]
fn test_maxent_nd_validation_errors() {
    let g = grid_graph(2, 2);

    // dims=0 rejected by LayoutNd construction
    assert!(matches!(
        LayoutNd::<f64>::zeros(4, 0),
        Err(CoreError::InvalidInput(_))
    ));

    // use_barnes_hut == true with dims=4 rejected with UnsupportedOperation
    let mut s_bh = MaxentStressSolver::<f64>::new(&g, MaxentStressConfig::default()).unwrap();
    let mut l4 = LayoutNd::<f64>::zeros(4, 4).unwrap();
    assert!(matches!(
        s_bh.step(&mut l4),
        Err(CoreError::UnsupportedOperation(_))
    ));

    // NaN / Inf rejected
    let mut l_nan = LayoutNd::from_flat(4, 3, vec![f64::NAN; 12]).unwrap();
    assert!(matches!(
        s_bh.step(&mut l_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));
    let mut l_inf = LayoutNd::from_flat(4, 3, vec![f64::INFINITY; 12]).unwrap();
    assert!(matches!(
        s_bh.step(&mut l_inf),
        Err(CoreError::NonFiniteValue { .. })
    ));

    // Node count mismatch & mid-run dims mismatch
    let mut l_wrong_n = LayoutNd::<f64>::zeros(5, 3).unwrap();
    assert!(matches!(
        s_bh.step(&mut l_wrong_n),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let mut l3 = LayoutNd::<f64>::init(4, 3, 5, 1.0).unwrap();
    s_bh.step(&mut l3).unwrap();
    let mut l2 = LayoutNd::<f64>::init(4, 2, 5, 1.0).unwrap();
    assert!(matches!(
        s_bh.step(&mut l2),
        Err(CoreError::DimensionMismatch {
            expected: 3,
            actual: 2,
            ..
        })
    ));

    // Ensure f32 LayoutNd non-finite check also fires
    let mut s_bh32 = MaxentStressSolver::<f32>::new(&g, MaxentStressConfig::default()).unwrap();
    let mut l32_nan = LayoutNd::from_flat(4, 3, vec![f32::NAN; 12]).unwrap();
    assert!(matches!(
        s_bh32.step(&mut l32_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));
}
