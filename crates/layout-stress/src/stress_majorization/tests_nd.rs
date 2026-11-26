use egraph_core::{
    CoreError, CsrGraph, DeterministicRng, IterativeSolver, LayoutNd, Scalar, SolverStatus,
};

use super::{StressMajorizationConfig, StressMajorizationSolver};

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

fn seeded_nd_layout<S: Scalar>(n: usize, dims: usize, scale: f64, seed: u64) -> LayoutNd<S> {
    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut coords = Vec::with_capacity(n * dims);
    for _ in 0..(n * dims) {
        coords.push(S::from_f64((rng.next_f64() * 2.0 - 1.0) * scale));
    }
    LayoutNd::from_flat(n, dims, coords).unwrap()
}

#[test]
fn test_stress_majorization_nd_step_vs_run_bit_exact_3d_5d() {
    let grid = grid_graph(3, 3);
    let cfg = StressMajorizationConfig {
        max_iterations: 30,
        epsilon: 1e-6,
        ..Default::default()
    };

    // f64 in dims=3
    let mut s_run = StressMajorizationSolver::<f64>::from_graph(&grid, cfg).unwrap();
    let mut s_step = StressMajorizationSolver::<f64>::from_graph(&grid, cfg).unwrap();
    let mut l_run = seeded_nd_layout::<f64>(9, 3, 1.5, 101);
    let mut l_step = l_run.copy();

    let stress_before = s_run.compute_stress_nd(&l_run).unwrap();
    let st_run = s_run.run(&mut l_run).unwrap();
    assert!(st_run.is_finished());
    let stress_after = s_run.compute_stress_nd(&l_run).unwrap();
    assert!(
        stress_after < stress_before * 0.2,
        "before={stress_before}, after={stress_after}"
    );

    let _ = s_step.run_steps(&mut l_step, 4).unwrap();
    while !s_step.status().is_finished() {
        s_step.step(&mut l_step).unwrap();
    }
    assert_eq!(s_run.status(), s_step.status());
    assert_eq!(s_run.iteration(), s_step.iteration());
    assert_eq!(s_run.total_iterations(), 30);
    for (a, b) in l_run.coords().iter().zip(l_step.coords().iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    // f32 in dims=3
    let mut s32_run = StressMajorizationSolver::<f32>::from_graph(&grid, cfg).unwrap();
    let mut s32_step = StressMajorizationSolver::<f32>::from_graph(&grid, cfg).unwrap();
    let mut l32_run = seeded_nd_layout::<f32>(9, 3, 1.5, 101);
    let mut l32_step = l32_run.copy();

    let stress32_before = s32_run.compute_stress_nd(&l32_run).unwrap();
    s32_run.run(&mut l32_run).unwrap();
    let stress32_after = s32_run.compute_stress_nd(&l32_run).unwrap();
    assert!(stress32_after < stress32_before * 0.2);

    while !s32_step.status().is_finished() {
        s32_step.step(&mut l32_step).unwrap();
    }
    assert_eq!(s32_run.iteration(), s32_step.iteration());
    for (a, b) in l32_run.coords().iter().zip(l32_step.coords().iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    // f64 in dims=5
    let mut s5_run = StressMajorizationSolver::<f64>::from_graph(&grid, cfg).unwrap();
    let mut s5_step = StressMajorizationSolver::<f64>::from_graph(&grid, cfg).unwrap();
    let mut l5_run = seeded_nd_layout::<f64>(9, 5, 1.5, 202);
    let mut l5_step = l5_run.copy();

    let stress5_before = s5_run.compute_stress_nd(&l5_run).unwrap();
    s5_run.run(&mut l5_run).unwrap();
    let stress5_after = s5_run.compute_stress_nd(&l5_run).unwrap();
    assert!(stress5_after < stress5_before * 0.2);

    while !s5_step.status().is_finished() {
        s5_step.step(&mut l5_step).unwrap();
    }
    for (a, b) in l5_run.coords().iter().zip(l5_step.coords().iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    // f64 in dims=1 (grid stress decreases monotonically; path graph embeds nearly exact)
    let mut s1_run = StressMajorizationSolver::<f64>::from_graph(&grid, cfg).unwrap();
    let mut l1_run = seeded_nd_layout::<f64>(9, 1, 3.0, 303);
    let stress1_before = s1_run.compute_stress_nd(&l1_run).unwrap();
    s1_run.run(&mut l1_run).unwrap();
    let stress1_after = s1_run.compute_stress_nd(&l1_run).unwrap();
    assert!(
        stress1_after < stress1_before,
        "before={stress1_before}, after={stress1_after}"
    );
}

#[test]
fn test_stress_majorization_nd_edge_cases_and_degenerate_geometries() {
    let cfg = StressMajorizationConfig::default();

    // n=0 and n=1
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let mut s0 = StressMajorizationSolver::<f64>::from_graph(&empty, cfg).unwrap();
    let mut l0 = LayoutNd::<f64>::zeros(0, 3).unwrap();
    assert_eq!(s0.run(&mut l0).unwrap(), SolverStatus::Converged);
    assert_eq!(s0.compute_stress_nd(&l0).unwrap(), 0.0);

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let mut s1 = StressMajorizationSolver::<f64>::from_graph(&single, cfg).unwrap();
    let mut l1 = LayoutNd::from_flat(1, 3, vec![3.0, -4.0, 5.0]).unwrap();
    assert_eq!(s1.run(&mut l1).unwrap(), SolverStatus::Converged);
    assert_eq!(l1.coords(), &[3.0, -4.0, 5.0]);

    // Edgeless graph (n=4, dims=3)
    let edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    let mut se = StressMajorizationSolver::<f64>::from_graph(&edgeless, cfg).unwrap();
    let mut le = seeded_nd_layout::<f64>(4, 3, 1.0, 7);
    let le_orig = le.copy();
    assert_eq!(se.run(&mut le).unwrap(), SolverStatus::Converged);
    assert_eq!(le.coords(), le_orig.coords());

    // Disconnected graph + isolated vertex
    let disc = CsrGraph::<f64>::from_edges(5, &[(0, 1), (2, 3)], None, false, false).unwrap();
    let mut s_disc = StressMajorizationSolver::<f64>::from_graph(&disc, cfg).unwrap();
    let mut l_disc = LayoutNd::from_flat(
        5,
        3,
        vec![
            0.0, 0.0, 0.0, //
            0.2, 0.1, 0.2, //
            5.0, 1.0, -1.0, //
            8.0, 1.5, 2.0, //
            -3.0, 2.0, 7.0,
        ],
    )
    .unwrap();
    assert!(s_disc.run(&mut l_disc).unwrap().is_finished());
    assert!(l_disc.is_all_finite());
    assert_eq!(l_disc.node(4).unwrap(), &[-3.0, 2.0, 7.0]);

    // Coincident points in 3D remain finite
    let g3 = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, false, false).unwrap();
    let mut s_coin = StressMajorizationSolver::<f64>::from_graph(&g3, cfg).unwrap();
    let mut l_coin = LayoutNd::<f64>::zeros(3, 3).unwrap();
    assert!(s_coin.run(&mut l_coin).unwrap().is_finished());
    assert!(l_coin.is_all_finite());

    // Collinear in 3D (x-axis) & planar in 3D (z=0)
    let mut s_col = StressMajorizationSolver::<f64>::from_graph(&g3, cfg).unwrap();
    let mut l_col =
        LayoutNd::from_flat(3, 3, vec![0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.6, 0.0, 0.0]).unwrap();
    let col_before = s_col.compute_stress_nd(&l_col).unwrap();
    assert!(s_col.run(&mut l_col).unwrap().is_finished());
    assert!(s_col.compute_stress_nd(&l_col).unwrap() < col_before);

    let g4 = grid_graph(2, 2);
    let mut s_pln = StressMajorizationSolver::<f64>::from_graph(&g4, cfg).unwrap();
    let mut l_pln = LayoutNd::from_flat(
        4,
        3,
        vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0],
    )
    .unwrap();
    let pln_before = s_pln.compute_stress_nd(&l_pln).unwrap();
    assert!(s_pln.run(&mut l_pln).unwrap().is_finished());
    assert!(s_pln.compute_stress_nd(&l_pln).unwrap() < pln_before);
}

#[test]
fn test_stress_majorization_nd_validation_errors() {
    let g = grid_graph(2, 2);
    let cfg = StressMajorizationConfig::default();

    // dims=0 rejected by LayoutNd construction
    assert!(matches!(
        LayoutNd::<f64>::zeros(4, 0),
        Err(CoreError::InvalidInput(_))
    ));

    // NaN and Inf rejected
    let mut s = StressMajorizationSolver::<f64>::from_graph(&g, cfg).unwrap();
    let mut l_nan = LayoutNd::from_flat(4, 3, vec![f64::NAN; 12]).unwrap();
    assert!(matches!(
        s.step(&mut l_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        s.compute_stress_nd(&l_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));

    let mut l_inf = LayoutNd::from_flat(4, 3, vec![f64::INFINITY; 12]).unwrap();
    assert!(matches!(
        s.step(&mut l_inf),
        Err(CoreError::NonFiniteValue { .. })
    ));

    // Node count mismatch
    let mut l_wrong_n = LayoutNd::<f64>::zeros(5, 3).unwrap();
    assert!(matches!(
        s.step(&mut l_wrong_n),
        Err(CoreError::DimensionMismatch { .. })
    ));

    // Mid-run dims consistency violation
    let mut l3 = seeded_nd_layout::<f64>(4, 3, 1.0, 9);
    s.step(&mut l3).unwrap();
    let mut l4 = seeded_nd_layout::<f64>(4, 4, 1.0, 9);
    assert!(matches!(
        s.step(&mut l4),
        Err(CoreError::DimensionMismatch {
            expected: 3,
            actual: 4,
            ..
        })
    ));
}
