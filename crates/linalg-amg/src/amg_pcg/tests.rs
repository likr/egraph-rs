use egraph_core::{CsrGraph, SolverStatus};
use egraph_linalg::{CsrMatrix, FnOperator, LanczosEigen, LaplacianKind, LinearOperator};

use super::*;

fn make_grid_graph(rows: usize, cols: usize) -> CsrGraph<f64> {
    let mut edges = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let u = (r * cols + c) as u32;
            if c + 1 < cols {
                edges.push((u, u + 1));
            }
            if r + 1 < rows {
                edges.push((u, u + (cols as u32)));
            }
        }
    }
    CsrGraph::from_edges(rows * cols, &edges, None, false, true).unwrap()
}

fn make_power_law_hub_graph(num_hubs: usize, hub_deg: usize) -> CsrGraph<f64> {
    // Construct a synthetic web-Stanford-like multi-hub graph with thousands of degree-1
    // leaves and interconnected high-degree hubs.
    let mut edges = Vec::new();
    for h in 0..num_hubs {
        for h2 in (h + 1)..num_hubs {
            edges.push((h as u32, h2 as u32));
        }
    }
    let mut next_node = num_hubs;
    for h in 0..num_hubs {
        for _ in 0..hub_deg {
            let leaf = next_node as u32;
            edges.push((h as u32, leaf));
            next_node += 1;
        }
        // Add a chain bridge between hubs through degree-2 nodes.
        if h + 1 < num_hubs {
            let b1 = next_node as u32;
            let b2 = (next_node + 1) as u32;
            edges.push((h as u32, b1));
            edges.push((b1, b2));
            edges.push((b2, (h + 1) as u32));
            next_node += 2;
        }
    }
    CsrGraph::from_edges(next_node, &edges, None, false, true).unwrap()
}

#[test]
fn test_spd_regularized_laplacians_and_shifted_combinatorial() {
    let grid = make_grid_graph(12, 12);
    let n = grid.num_nodes();
    let b: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) as f64).sin()).collect();

    for &gamma in &[0.01, 1.0, 100.0, 10_000.0] {
        let solver =
            AmgPcgSolver::from_regularized_sym_laplacian(&grid, gamma, AmgPcgConfig::default())
                .unwrap();
        assert!(solver.num_levels() >= 2);
        assert!(solver.coarsest_size() <= solver.config().max_coarse_size);
        assert!(solver.max_level_nnz() <= solver.finest_matrix().nnz());

        let (x, res) = solver.solve_vec(&b).unwrap();
        assert!(res.converged, "gamma={gamma} failed: {res:?}");
        assert!(res.relative_residual <= 1e-8);
        assert!(x.iter().all(|v| v.is_finite()));
    }

    let comb_solver = AmgPcgSolver::from_laplacian(
        &grid,
        LaplacianKind::Combinatorial,
        1e-3,
        AmgPcgConfig::default(),
    )
    .unwrap();
    let (x_c, res_c) = comb_solver.solve_vec(&b).unwrap();
    assert!(res_c.converged, "shifted combinatorial failed: {res_c:?}");
    assert!(res_c.relative_residual <= 1e-8);
    assert!(x_c.iter().all(|v| v.is_finite()));
}

#[test]
fn test_singular_disconnected_isolated_empty_singleton_and_zero_rhs() {
    // 1. Empty graph (n = 0)
    let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let s0 = AmgPcgSolver::from_laplacian(
        &g0,
        LaplacianKind::Combinatorial,
        0.0,
        AmgPcgConfig::default(),
    )
    .unwrap();
    assert_eq!(s0.num_nodes(), 0);
    let (x0, r0) = s0.solve_vec(&[]).unwrap();
    assert!(x0.is_empty());
    assert!(r0.converged);
    assert_eq!(r0.iterations, 0);

    // 2. Singleton graph (n = 1) unshifted L = [0]
    let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let s1 = AmgPcgSolver::from_laplacian(
        &g1,
        LaplacianKind::Combinatorial,
        0.0,
        AmgPcgConfig::default(),
    )
    .unwrap();
    let (x1_zero, r1_zero) = s1.solve_vec(&[0.0]).unwrap();
    assert_eq!(x1_zero, vec![0.0]);
    assert!(r1_zero.converged);
    // Inconsistent RHS on zero row must not NaN or panic.
    let (x1_bad, r1_bad) = s1.solve_vec(&[3.0]).unwrap();
    assert!(x1_bad.iter().all(|v| v.is_finite()));
    assert!(!r1_bad.converged);
    assert!((r1_bad.residual_norm - 3.0).abs() < 1e-12);

    // 3. Edgeless graph (n = 32) unshifted L = 0
    let g_edgeless = CsrGraph::<f64>::from_edges(32, &[], None, false, false).unwrap();
    let s_edgeless = AmgPcgSolver::from_laplacian(
        &g_edgeless,
        LaplacianKind::Combinatorial,
        0.0,
        AmgPcgConfig::default(),
    )
    .unwrap();
    let (xe, re) = s_edgeless.solve_vec(&vec![0.0; 32]).unwrap();
    assert_eq!(xe, vec![0.0; 32]);
    assert!(re.converged);
    let (xe_inc, re_inc) = s_edgeless.solve_vec(&vec![1.0; 32]).unwrap();
    assert!(xe_inc.iter().all(|v| v.is_finite()));
    assert!(!re_inc.converged);

    // 4. Multi-component graph + isolated zero-row vertices with unshifted combinatorial L
    let mut edges = Vec::new();
    // Component 1: 0..30 cycle
    for i in 0..30u32 {
        edges.push((i, (i + 1) % 30));
    }
    // Component 2: 30..60 cycle
    for i in 30..60u32 {
        edges.push((i, 30 + (i - 30 + 1) % 30));
    }
    // Component 3: 60..90 path
    for i in 60..89u32 {
        edges.push((i, i + 1));
    }
    // Nodes 90, 91, 92 are isolated (zero rows in combinatorial L)
    let g_multi = CsrGraph::<f64>::from_edges(93, &edges, None, false, true).unwrap();
    let s_multi = AmgPcgSolver::from_laplacian(
        &g_multi,
        LaplacianKind::Combinatorial,
        0.0,
        AmgPcgConfig::default(),
    )
    .unwrap();
    // Construct consistent RHS orthogonal to each component's constant vector and 0 on isolated nodes.
    let mut b_cons = vec![0.0_f64; 93];
    for (i, v) in b_cons.iter_mut().enumerate().take(30) {
        *v = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    for (i, v) in b_cons.iter_mut().enumerate().take(60).skip(30) {
        *v = if i % 2 == 0 { 2.0 } else { -2.0 };
    }
    for (i, v) in b_cons.iter_mut().enumerate().take(90).skip(60) {
        *v = if i % 2 == 0 { 0.5 } else { -0.5 };
    }
    let (x_m, r_m) = s_multi.solve_vec(&b_cons).unwrap();
    assert!(r_m.converged, "multi-CC singular L failed: {r_m:?}");
    assert!(r_m.relative_residual <= 1e-8);
    assert!(x_m.iter().all(|v| v.is_finite()));
    assert_eq!(x_m[90], 0.0);
    assert_eq!(x_m[91], 0.0);
    assert_eq!(x_m[92], 0.0);

    // Zero RHS fast path
    let (x_z, r_z) = s_multi.solve_vec(&vec![0.0; 93]).unwrap();
    assert_eq!(x_z, vec![0.0; 93]);
    assert!(r_z.converged);
    assert_eq!(r_z.iterations, 0);
}

#[test]
fn test_power_law_hub_web_stanford_like_no_fill_explosion() {
    let g = make_power_law_hub_graph(3, 2_000);
    let n = g.num_nodes();
    let cfg = AmgPcgConfig::default();
    let solver = AmgPcgSolver::from_laplacian(&g, LaplacianKind::Combinatorial, 1e-3, cfg)
        .expect("hub graph solver construction");
    let input_nnz = solver.finest_matrix().nnz();
    assert!(
        solver.max_level_nnz() <= input_nnz,
        "fill-in explosion: max_level_nnz={} > input_nnz={}",
        solver.max_level_nnz(),
        input_nnz
    );
    assert!(
        solver.operator_complexity() < 2.0,
        "operator_complexity={} >= 2.0",
        solver.operator_complexity()
    );

    let b: Vec<f64> = (0..n).map(|i| ((i * 13 + 5) as f64).cos()).collect();
    let (x, res) = solver.solve_vec(&b).unwrap();
    assert!(res.converged, "hub graph solve failed: {res:?}");
    assert!(
        res.iterations < 40,
        "too many iterations: {}",
        res.iterations
    );
    assert!(x.iter().all(|v| v.is_finite()));
}

#[test]
fn test_step_run_equivalence_batch_determinism_and_scale_invariance() {
    let g = make_grid_graph(10, 10);
    let n = g.num_nodes();
    let cfg = AmgPcgConfig::default()
        .with_randomize_ties(true)
        .with_seed(123);
    let solver1 = AmgPcgSolver::from_regularized_sym_laplacian(&g, 10.0, cfg.clone()).unwrap();
    let solver2 = AmgPcgSolver::from_regularized_sym_laplacian(&g, 10.0, cfg).unwrap();

    let b: Vec<f64> = (0..n).map(|i| ((i * 11 + 1) as f64).sin()).collect();
    let mut ws_sync = solver1.new_workspace();
    let mut x_sync = vec![0.0; n];
    let res_sync = solver1.solve(&b, &mut x_sync, &mut ws_sync).unwrap();

    let mut ws_step = AmgPcgWorkspace::new();
    solver2.init_solve(&b, None, &mut ws_step).unwrap();
    assert_eq!(ws_step.status(), SolverStatus::Initialized);
    solver2.run_steps(&mut ws_step, 2).unwrap();
    while !ws_step.is_finished() {
        solver2.step(&mut ws_step).unwrap();
    }
    assert_eq!(ws_step.x(), x_sync.as_slice());
    assert_eq!(ws_step.iteration(), res_sync.iterations);
    assert_eq!(ws_step.converged(), res_sync.converged);

    // Batched solve equivalence with single-RHS solves
    let ncols = 3;
    let mut b_batch = vec![0.0; n * ncols];
    let mut x_batch = vec![0.0; n * ncols];
    let mut cols = vec![vec![0.0; n]; ncols];
    for c in 0..ncols {
        for r in 0..n {
            let v = (((r * 17 + c * 31) % 97) as f64) / 50.0 - 1.0;
            b_batch[r * ncols + c] = v;
            cols[c][r] = v;
        }
    }
    let batch_res = solver1
        .solve_batch(&b_batch, &mut x_batch, ncols, &mut ws_sync)
        .unwrap();
    assert!(batch_res.all_converged);
    for c in 0..ncols {
        let (x_col, _) = solver1.solve_vec(&cols[c]).unwrap();
        for r in 0..n {
            assert_eq!(x_batch[r * ncols + c], x_col[r]);
        }
    }

    // Scale invariance: b * 1e-12 vs b * 1e10 must NOT exit at iter 0 for tiny b
    let b_tiny: Vec<f64> = b.iter().map(|&v| v * 1e-12).collect();
    let b_huge: Vec<f64> = b.iter().map(|&v| v * 1e10).collect();
    let (x_tiny, r_tiny) = solver1.solve_vec(&b_tiny).unwrap();
    let (x_huge, r_huge) = solver1.solve_vec(&b_huge).unwrap();
    assert!(r_tiny.converged && r_tiny.iterations > 0);
    assert!(r_huge.converged && r_huge.iterations > 0);
    assert_eq!(r_tiny.iterations, r_huge.iterations);
    for i in 0..n {
        assert!((x_tiny[i] * 1e22 - x_huge[i]).abs() <= 1e-6 * x_huge[i].abs().max(1.0));
    }

    // Extreme scale invariance across 1e-290 .. 1e+290 (no overflow/underflow)
    let b_unit: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.37).cos()).collect();
    let (x_ref, r_ref) = solver1.solve_vec(&b_unit).unwrap();
    assert!(r_ref.converged);
    for &scale in &[
        1e-155_f64, 1e155_f64, 1e-250_f64, 1e250_f64, 1e-290_f64, 1e290_f64,
    ] {
        let b_scaled: Vec<f64> = b_unit.iter().map(|&v| v * scale).collect();
        let (x_scaled, r_scaled) = solver1.solve_vec(&b_scaled).unwrap();
        assert!(
            r_scaled.converged,
            "failed to converge at scale={scale:.1e}: rel_res={:.3e}",
            r_scaled.relative_residual
        );
        assert_eq!(r_scaled.iterations, r_ref.iterations);
        for i in 0..n {
            let rel_err = (x_scaled[i] / scale - x_ref[i]).abs();
            assert!(
                rel_err <= 1e-7 * x_ref[i].abs().max(1.0),
                "scale={scale:.1e} mismatch at {i}: rel_err={rel_err}"
            );
        }
    }
}

#[test]
fn test_adversarial_validation_rejects_bad_inputs() {
    // Asymmetric matrix
    let asym = CsrMatrix::from_csr(
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![2.0, -1.0, -2.0, 2.0],
    )
    .unwrap();
    assert!(AmgPcgSolver::new(&asym, AmgPcgConfig::default()).is_err());

    // Negative diagonal
    let neg_diag = CsrMatrix::from_csr(
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![-1.0, 0.0, 0.0, 1.0],
    )
    .unwrap();
    assert!(AmgPcgSolver::new(&neg_diag, AmgPcgConfig::default()).is_err());

    // Zero diagonal with non-zero off-diagonal
    let zero_diag_off = CsrMatrix::from_csr(
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![0.0, -1.0, -1.0, 2.0],
    )
    .unwrap();
    assert!(AmgPcgSolver::new(&zero_diag_off, AmgPcgConfig::default()).is_err());

    // Non-finite RHS / dimension mismatch
    let g = make_grid_graph(3, 3);
    let solver =
        AmgPcgSolver::from_regularized_sym_laplacian(&g, 1.0, AmgPcgConfig::default()).unwrap();
    assert!(solver.solve_vec(&[f64::NAN; 9]).is_err());
    assert!(solver.solve_vec(&[1.0; 8]).is_err());
    assert!(
        AmgPcgSolver::from_laplacian(&g, LaplacianKind::RandomWalk, 0.0, AmgPcgConfig::default())
            .is_err()
    );
}

#[test]
fn test_linear_operator_impl_on_amg_pcg_solver() {
    let grid = make_grid_graph(8, 8);
    let n = grid.num_nodes();
    let mut solver = AmgPcgSolver::from_regularized_sym_laplacian(
        &grid,
        5.0,
        AmgPcgConfig::default()
            .with_pre_sweeps(1)
            .with_post_sweeps(1),
    )
    .unwrap();
    assert_eq!(solver.dim(), n);

    let x1: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.31).sin()).collect();
    let x2: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.73).cos()).collect();
    let alpha = 2.25_f64;
    let beta = -1.5_f64;
    let x_comb: Vec<f64> = x1
        .iter()
        .zip(x2.iter())
        .map(|(&a, &b)| alpha * a + beta * b)
        .collect();

    let mut y1 = vec![0.0_f64; n];
    let mut y2 = vec![0.0_f64; n];
    let mut y_comb = vec![0.0_f64; n];
    solver.apply(&x1, &mut y1).unwrap();
    solver.apply(&x2, &mut y2).unwrap();
    solver.apply(&x_comb, &mut y_comb).unwrap();

    // Linearity: M^-1 (alpha x1 + beta x2) == alpha M^-1 x1 + beta M^-1 x2.
    for i in 0..n {
        let expected = alpha * y1[i] + beta * y2[i];
        assert!(
            (y_comb[i] - expected).abs() <= 1e-12,
            "linearity mismatch at {i}: got {}, expected {}",
            y_comb[i],
            expected
        );
    }

    // Equivalence with explicit apply_vcycle(&self, ..., &mut ws).
    let mut ws = solver.new_workspace();
    let mut y_vcycle = vec![0.0_f64; n];
    solver.apply_vcycle(&x1, &mut y_vcycle, &mut ws).unwrap();
    assert_eq!(y_vcycle, y1);

    // Symmetry: <x1, M^-1 x2> == <M^-1 x1, x2> when pre_sweeps == post_sweeps.
    let dot_12: f64 = x1.iter().zip(y2.iter()).map(|(a, b)| a * b).sum();
    let dot_21: f64 = y1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum();
    assert!(
        (dot_12 - dot_21).abs() <= 1e-11,
        "V-cycle symmetry violated: {dot_12} vs {dot_21}"
    );

    // Usable directly as a LinearOperator<f64> inside LanczosEigen.
    let lanczos = LanczosEigen::default().with_tol(1e-6).with_max_iters(200);
    let (lambdas, _) = lanczos.largest_eigenpairs_operator(&mut solver, 2).unwrap();
    assert_eq!(lambdas.len(), 2);
    assert!(lambdas[0] >= lambdas[1] && lambdas[1] > 0.0);

    // Input validation on LinearOperator::apply.
    let mut bad_y = vec![0.0_f64; n - 1];
    assert!(solver.apply(&x1, &mut bad_y).is_err());
    assert!(solver.apply(&x1[..n - 1], &mut y1).is_err());
    let mut bad_x = x1.clone();
    bad_x[0] = f64::NAN;
    assert!(solver.apply(&bad_x, &mut y1).is_err());
}

#[test]
fn test_matrix_free_operator_solve_step_and_batch_equivalence() {
    let grid = make_grid_graph(8, 8);
    let n = grid.num_nodes();
    let cfg = AmgPcgConfig::default().with_rel_tol(1e-10).with_seed(77);
    let solver = AmgPcgSolver::from_regularized_sym_laplacian(&grid, 4.0, cfg).unwrap();
    let csr = solver.finest_matrix().clone();

    let b: Vec<f64> = (0..n).map(|i| ((i * 19 + 3) as f64).sin()).collect();
    let (x_csr, res_csr) = solver.solve_vec(&b).unwrap();
    assert!(res_csr.converged);

    // Wrap the exact same operator as a matrix-free FnOperator.
    let mut fn_op = FnOperator::new(n, |vx: &[f64], vy: &mut [f64]| csr.spmv(vx, vy));
    let (x_fn, res_fn) = solver.solve_vec_with_operator(&mut fn_op, &b).unwrap();
    assert_eq!(x_fn, x_csr);
    assert_eq!(res_fn, res_csr);

    // Cooperative step_with_operator + run_steps_with_operator via &mut dyn LinearOperator.
    let mut ws_op = AmgPcgWorkspace::new();
    let mut dyn_op: &mut dyn LinearOperator<f64> = &mut fn_op;
    solver
        .init_solve_with_operator(&mut dyn_op, &b, None, &mut ws_op)
        .unwrap();
    assert_eq!(ws_op.status(), SolverStatus::Initialized);
    solver
        .run_steps_with_operator(&mut dyn_op, &mut ws_op, 0)
        .unwrap();
    assert_eq!(ws_op.iteration(), 0);
    solver
        .run_steps_with_operator(&mut dyn_op, &mut ws_op, 2)
        .unwrap();
    while !ws_op.is_finished() {
        solver.step_with_operator(&mut dyn_op, &mut ws_op).unwrap();
    }
    assert_eq!(ws_op.x(), x_csr.as_slice());
    assert_eq!(ws_op.result(), res_csr);

    // Batched solve_batch_with_operator equivalence.
    let ncols = 2;
    let b_batch: Vec<f64> = (0..n * ncols).map(|i| ((i * 5 + 1) as f64).cos()).collect();
    let mut x_batch_csr = vec![0.0_f64; n * ncols];
    let mut x_batch_op = vec![0.0_f64; n * ncols];
    let mut ws_csr = solver.new_workspace();
    let mut ws_op_b = solver.new_workspace();
    let batch_csr = solver
        .solve_batch(&b_batch, &mut x_batch_csr, ncols, &mut ws_csr)
        .unwrap();
    let batch_op = solver
        .solve_batch_with_operator(&mut dyn_op, &b_batch, &mut x_batch_op, ncols, &mut ws_op_b)
        .unwrap();
    assert_eq!(x_batch_op, x_batch_csr);
    assert_eq!(batch_op, batch_csr);

    // Dimension mismatch validation on operator paths (including steps == 0).
    let mut wrong_op = FnOperator::new(n + 1, |vx: &[f64], vy: &mut [f64]| {
        vy.copy_from_slice(vx);
        Ok(())
    });
    assert!(solver.solve_vec_with_operator(&mut wrong_op, &b).is_err());
    assert!(
        solver
            .run_steps_with_operator(&mut wrong_op, &mut ws_op, 0)
            .is_err()
    );
}
