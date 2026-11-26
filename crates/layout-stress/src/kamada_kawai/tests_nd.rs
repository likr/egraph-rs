use egraph_core::{
    CoreError, CsrGraph, DeterministicRng, IterativeSolver, LayoutNd, Scalar, SolverStatus,
};

use super::{KamadaKawaiConfig, KamadaKawaiSolver};

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

fn compute_kk_energy(solver: &KamadaKawaiSolver<f64>, layout: &LayoutNd<f64>) -> f64 {
    let n = layout.num_nodes();
    let d = layout.dims();
    let coords = layout.coords();
    let l = &solver.l;
    let k = &solver.k;
    let mut total = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let kij = k[i * n + j];
            if kij <= 0.0 {
                continue;
            }
            let mut dist_sq = 0.0_f64;
            for a in 0..d {
                let diff = coords[i * d + a] - coords[j * d + a];
                dist_sq += diff * diff;
            }
            let diff = dist_sq.sqrt() - l[i * n + j];
            total += 0.5 * kij * diff * diff;
        }
    }
    total
}

#[test]
fn test_kamada_kawai_nd_step_vs_run_bit_exact_3d_5d() {
    let grid = grid_graph(3, 3);
    let cfg = KamadaKawaiConfig::default();

    // f64 in dims=3
    let mut s_run = KamadaKawaiSolver::<f64>::from_graph(&grid, cfg).unwrap();
    let mut s_step = KamadaKawaiSolver::<f64>::from_graph(&grid, cfg).unwrap();
    let mut l_run = seeded_nd_layout::<f64>(9, 3, 1.5, 31);
    let mut l_step = l_run.copy();

    let energy_before = compute_kk_energy(&s_run, &l_run);
    assert_eq!(s_run.total_iterations(), 100);
    let st_run = s_run.run(&mut l_run).unwrap();
    assert!(st_run.is_finished());
    assert!(l_run.is_all_finite());
    let energy_after = compute_kk_energy(&s_run, &l_run);
    assert!(
        energy_after < energy_before * 0.25,
        "before={energy_before}, after={energy_after}"
    );

    let _ = s_step.run_steps(&mut l_step, 5).unwrap();
    while !s_step.status().is_finished() {
        s_step.step(&mut l_step).unwrap();
    }
    assert_eq!(s_run.status(), s_step.status());
    assert_eq!(s_run.iteration(), s_step.iteration());
    for (a, b) in l_run.coords().iter().zip(l_step.coords().iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    // f32 in dims=3
    let mut s32_run = KamadaKawaiSolver::<f32>::from_graph(&grid, cfg).unwrap();
    let mut s32_step = KamadaKawaiSolver::<f32>::from_graph(&grid, cfg).unwrap();
    let mut l32_run = seeded_nd_layout::<f32>(9, 3, 1.5, 31);
    let mut l32_step = l32_run.copy();
    s32_run.run(&mut l32_run).unwrap();
    while !s32_step.status().is_finished() {
        s32_step.step(&mut l32_step).unwrap();
    }
    assert_eq!(s32_run.iteration(), s32_step.iteration());
    for (a, b) in l32_run.coords().iter().zip(l32_step.coords().iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    // f64 in dims=5
    let mut s5_run = KamadaKawaiSolver::<f64>::from_graph(&grid, cfg).unwrap();
    let mut s5_step = KamadaKawaiSolver::<f64>::from_graph(&grid, cfg).unwrap();
    let mut l5_run = seeded_nd_layout::<f64>(9, 5, 1.5, 41);
    let mut l5_step = l5_run.copy();
    let e5_before = compute_kk_energy(&s5_run, &l5_run);
    s5_run.run(&mut l5_run).unwrap();
    let e5_after = compute_kk_energy(&s5_run, &l5_run);
    assert!(e5_after < e5_before * 0.25);

    while !s5_step.status().is_finished() {
        s5_step.step(&mut l5_step).unwrap();
    }
    for (a, b) in l5_run.coords().iter().zip(l5_step.coords().iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    // f64 in dims=1
    let mut s1_run = KamadaKawaiSolver::<f64>::from_graph(&grid, cfg).unwrap();
    let mut l1_run = seeded_nd_layout::<f64>(9, 1, 1.5, 51);
    let e1_before = compute_kk_energy(&s1_run, &l1_run);
    s1_run.run(&mut l1_run).unwrap();
    assert!(compute_kk_energy(&s1_run, &l1_run) < e1_before * 0.7);
}

#[test]
fn test_kamada_kawai_nd_edge_cases_and_singular_fallbacks() {
    let cfg = KamadaKawaiConfig::default();

    // n=0, n=1, edgeless
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let mut s0 = KamadaKawaiSolver::<f64>::from_graph(&empty, cfg).unwrap();
    let mut l0 = LayoutNd::<f64>::zeros(0, 3).unwrap();
    assert_eq!(s0.run(&mut l0).unwrap(), SolverStatus::Converged);

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let mut s1 = KamadaKawaiSolver::<f64>::from_graph(&single, cfg).unwrap();
    let mut l1 = LayoutNd::from_flat(1, 3, vec![1.0, 2.0, 3.0]).unwrap();
    assert_eq!(s1.run(&mut l1).unwrap(), SolverStatus::Converged);

    let edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    let mut se = KamadaKawaiSolver::<f64>::from_graph(&edgeless, cfg).unwrap();
    let mut le = seeded_nd_layout::<f64>(4, 3, 1.0, 5);
    assert_eq!(se.run(&mut le).unwrap(), SolverStatus::Converged);

    // Disconnected graph
    let disc = CsrGraph::<f64>::from_edges(4, &[(0, 1), (2, 3)], None, false, false).unwrap();
    let mut s_disc = KamadaKawaiSolver::<f64>::from_graph(&disc, cfg).unwrap();
    let mut l_disc = LayoutNd::from_flat(
        4,
        3,
        vec![0.0, 0.0, 0.0, 0.3, 0.2, 0.1, 4.0, 1.0, 0.0, 6.0, 1.3, 0.5],
    )
    .unwrap();
    assert!(s_disc.run(&mut l_disc).unwrap().is_finished());
    assert!(l_disc.is_all_finite());

    // Coincident points in 3D
    let g3 = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, false, false).unwrap();
    let mut s_coin = KamadaKawaiSolver::<f64>::from_graph(&g3, cfg).unwrap();
    let mut l_coin = LayoutNd::<f64>::zeros(3, 3).unwrap();
    assert!(s_coin.run(&mut l_coin).unwrap().is_finished());
    assert!(l_coin.is_all_finite());

    // Collinear in 3D with exact Hessian singularity (H_yy = H_zz = 0 at node 0)
    let phi_inv = (5.0_f64.sqrt() - 1.0) * 0.5;
    let dists = [
        0.0, 1.0, phi_inv, //
        1.0, 0.0, 1.0, //
        phi_inv, 1.0, 0.0,
    ];
    let mut s_col = KamadaKawaiSolver::<f64>::from_distance_matrix(
        3,
        &dists,
        KamadaKawaiConfig {
            eps: 1e-3,
            max_inner_iters: 50,
            max_iterations: Some(60),
        },
    )
    .unwrap();
    let mut l_col =
        LayoutNd::from_flat(3, 3, vec![0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
    let col_before = compute_kk_energy(&s_col, &l_col);
    assert!(s_col.run(&mut l_col).unwrap().is_finished());
    assert!(l_col.is_all_finite());
    assert!(compute_kk_energy(&s_col, &l_col) < col_before);

    // Planar in 3D
    let g4 = grid_graph(2, 2);
    let mut s_pln = KamadaKawaiSolver::<f64>::from_graph(&g4, cfg).unwrap();
    let mut l_pln = LayoutNd::from_flat(
        4,
        3,
        vec![0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 2.5, 0.0, 2.5, 2.5, 0.0],
    )
    .unwrap();
    let pln_before = compute_kk_energy(&s_pln, &l_pln);
    assert!(s_pln.run(&mut l_pln).unwrap().is_finished());
    assert!(compute_kk_energy(&s_pln, &l_pln) < pln_before);
}

#[test]
fn test_kamada_kawai_nd_validation_errors() {
    let g = grid_graph(2, 2);
    let cfg = KamadaKawaiConfig::default();

    assert!(matches!(
        LayoutNd::<f64>::zeros(4, 0),
        Err(CoreError::InvalidInput(_))
    ));

    let mut s = KamadaKawaiSolver::<f64>::from_graph(&g, cfg).unwrap();
    let mut l_nan = LayoutNd::from_flat(4, 3, vec![f64::NAN; 12]).unwrap();
    assert!(matches!(
        s.step(&mut l_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));

    let mut l_inf = LayoutNd::from_flat(4, 3, vec![f64::NEG_INFINITY; 12]).unwrap();
    assert!(matches!(
        s.step(&mut l_inf),
        Err(CoreError::NonFiniteValue { .. })
    ));

    let mut l_wrong_n = LayoutNd::<f64>::zeros(3, 3).unwrap();
    assert!(matches!(
        s.step(&mut l_wrong_n),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let mut l3 = seeded_nd_layout::<f64>(4, 3, 1.0, 9);
    s.step(&mut l3).unwrap();
    let mut l5 = seeded_nd_layout::<f64>(4, 5, 1.0, 9);
    assert!(matches!(
        s.step(&mut l5),
        Err(CoreError::DimensionMismatch {
            expected: 3,
            actual: 5,
            ..
        })
    ));
}
