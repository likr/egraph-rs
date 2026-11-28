use super::*;
use crate::csr_matrix::symmetric_normalized_laplacian;
use egraph_core::CsrGraph;

fn make_grid_2d(rows: usize, cols: usize, periodic: bool) -> CsrGraph<f64> {
    let n = rows * cols;
    let mut edges = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let u = (r * cols + c) as u32;
            if c + 1 < cols {
                edges.push((u, u + 1));
            } else if periodic && cols > 2 {
                edges.push((u, (r * cols) as u32));
            }
            if r + 1 < rows {
                edges.push((u, u + cols as u32));
            } else if periodic && rows > 2 {
                edges.push((u, c as u32));
            }
        }
    }
    CsrGraph::from_edges(n, &edges, None, false, true).unwrap()
}

/// Computes minimum principal-angle cosine between column spaces of two (n x k) row-major bases.
fn min_principal_angle_cosine(u: &[f64], v: &[f64], n: usize, k: usize) -> f64 {
    let mut c = vec![0.0_f64; k * k];
    for r in 0..n {
        for a in 0..k {
            let ua = u[r * k + a];
            for b in 0..k {
                c[a * k + b] += ua * v[r * k + b];
            }
        }
    }
    let mut ctc = vec![0.0_f64; k * k];
    for a in 0..k {
        for b in 0..k {
            let mut sum = 0.0_f64;
            for p in 0..k {
                sum += c[p * k + a] * c[p * k + b];
            }
            ctc[a * k + b] = sum;
        }
    }
    let (evals, _) = symmetric_eigen_jacobi(&ctc, k, 120, 1e-13).unwrap();
    evals[0].max(0.0).sqrt().min(1.0)
}

#[test]
fn test_jacobi_zero_degenerate_and_orthonormal() {
    let zero = vec![0.0; 9];
    let (evals, evecs) = symmetric_eigen_jacobi(&zero, 3, 50, 1e-12).unwrap();
    assert_eq!(evals, vec![0.0, 0.0, 0.0]);
    assert_eq!(evecs, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

    // Matrix with repeated eigenvalues: diag(3, 1, 3) rotated
    let a = vec![2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 2.0];
    let (ev, vecs) = symmetric_eigen_jacobi(&a, 3, 80, 1e-13).unwrap();
    assert!((ev[0] - 1.0).abs() < 1e-12);
    assert!((ev[1] - 3.0).abs() < 1e-12);
    assert!((ev[2] - 3.0).abs() < 1e-12);

    // Verify A * V == V * diag(ev) and V^T * V == I
    for m in 0..3 {
        for i in 0..3 {
            let mut av = 0.0;
            for j in 0..3 {
                av += a[i * 3 + j] * vecs[j * 3 + m];
            }
            assert!((av - ev[m] * vecs[i * 3 + m]).abs() < 1e-11);
        }
        for m2 in 0..3 {
            let dot: f64 = (0..3).map(|i| vecs[i * 3 + m] * vecs[i * 3 + m2]).sum();
            let expected = if m == m2 { 1.0 } else { 0.0 };
            assert!((dot - expected).abs() < 1e-12);
        }
    }
}

#[test]
fn test_lanczos_matches_jacobi_on_16x16_grid_principal_angle() {
    // 16x16 grid (n=256) exercises iterative Lanczos and allows fast exact dense Jacobi comparison.
    let grid = make_grid_2d(16, 16, false);
    let n = grid.num_nodes();
    let n_modes = 8; // Avoid splitting degenerate (p, q)/(q, p) pair at boundary.
    let res = smallest_sym_laplacian_eigenpairs(&grid, n_modes, 1e-5, 500, 42).unwrap();
    assert!(!res.used_dense_fallback);
    assert!(res.iterations >= 1);
    assert_eq!(res.num_nodes, n);
    assert_eq!(res.num_modes, n_modes);

    let (l_sym, _) = symmetric_normalized_laplacian(&grid);
    let dense = l_sym.to_dense().unwrap();
    let (evals_ref, evecs_ref) = symmetric_eigen_jacobi(&dense, n, 120, 1e-12).unwrap();

    for m in 0..n_modes {
        assert!(
            (res.lambdas[m] - evals_ref[1 + m]).abs() < 1e-7,
            "mode {m} eigenvalue mismatch: {} vs {}",
            res.lambdas[m],
            evals_ref[1 + m]
        );
    }

    let mut u_ref = vec![0.0_f64; n * n_modes];
    for row in 0..n {
        for m in 0..n_modes {
            u_ref[row * n_modes + m] = evecs_ref[row * n + (1 + m)];
        }
    }
    let cos_min = min_principal_angle_cosine(&res.vectors, &u_ref, n, n_modes);
    assert!(
        cos_min > 0.9999,
        "16x16 grid principal-angle cosine {cos_min:.6} <= 0.9999"
    );
}

#[test]
fn test_lanczos_40x40_grid_tight_spectral_gaps() {
    // 40x40 2D grid (n=1600): validate residual accuracy, orthogonality, and subspace stability.
    let grid = make_grid_2d(40, 40, false);
    let n = grid.num_nodes();
    let n_modes = 16; // Modes 1..=16 end cleanly after (4,0)/(0,4) pair.
    let res = smallest_sym_laplacian_eigenpairs(&grid, n_modes, 1e-4, 500, 42).unwrap();
    let res_tight = smallest_sym_laplacian_eigenpairs(&grid, n_modes, 1e-8, 800, 99).unwrap();
    assert!(!res.used_dense_fallback);

    let (l_sym, _) = symmetric_normalized_laplacian(&grid);
    let mut x = vec![0.0_f64; n];
    let mut lx = vec![0.0_f64; n];
    for m in 0..n_modes {
        for i in 0..n {
            x[i] = res.vectors[i * n_modes + m];
        }
        l_sym.spmv(&x, &mut lx).unwrap();
        let res_norm: f64 = lx
            .iter()
            .zip(x.iter())
            .map(|(&lv, &xv)| {
                let diff = lv - res.lambdas[m] * xv;
                diff * diff
            })
            .sum::<f64>()
            .sqrt();
        assert!(
            res_norm < 1e-5,
            "40x40 mode {m} residual norm too large: {res_norm:.3e}"
        );
        assert!(
            (res.lambdas[m] - res_tight.lambdas[m]).abs() < 1e-7,
            "40x40 mode {m} eigenvalue mismatch vs tight ref"
        );
    }

    let cos_min = min_principal_angle_cosine(&res.vectors, &res_tight.vectors, n, n_modes);
    assert!(
        cos_min > 0.9999,
        "40x40 grid principal-angle cosine {cos_min:.6} <= 0.9999"
    );

    // Also validate exact closed-form eigenvalues with multiplicity 4 on a 40x40 periodic torus.
    let torus = make_grid_2d(40, 40, true);
    let torus_modes = 8; // (1,0),(0,1),(1,1) Fourier shells (4 + 4 = 8 modes cleanly separated).
    let res_torus = smallest_sym_laplacian_eigenpairs(&torus, torus_modes, 1e-4, 500, 7).unwrap();
    let expected_lam1 = 0.5 * (1.0 - (std::f64::consts::TAU / 40.0).cos());
    let expected_lam2 = 1.0 - (std::f64::consts::TAU / 40.0).cos();
    for (idx, &lam) in res_torus.lambdas[..4].iter().enumerate() {
        assert!(
            (lam - expected_lam1).abs() < 1e-7,
            "torus mode {idx}: got {lam}, expected {expected_lam1}, all={:?}",
            res_torus.lambdas
        );
    }
    for (idx, &lam) in res_torus.lambdas[4..8].iter().enumerate() {
        assert!(
            (lam - expected_lam2).abs() < 1e-7,
            "torus mode {}: got {lam}, expected {expected_lam2}, all={:?}",
            idx + 4,
            res_torus.lambdas
        );
    }
}

#[test]
fn test_eigen_edge_cases_empty_singleton_edgeless_disconnected() {
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let re = smallest_sym_laplacian_eigenpairs(&empty, 4, 1e-6, 100, 42).unwrap();
    assert_eq!(re.num_modes, 0);
    assert!(re.used_dense_fallback);
    assert_eq!(re.iterations, 0);

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let rs = smallest_sym_laplacian_eigenpairs(&single, 4, 1e-6, 100, 42).unwrap();
    assert_eq!(rs.num_modes, 0);
    assert_eq!(rs.iterations, 0);

    // Small grid hits dense Jacobi fallback with 0 Lanczos iterations.
    let small_grid = make_grid_2d(4, 4, false);
    let r_small = smallest_sym_laplacian_eigenpairs(&small_grid, 2, 1e-6, 100, 42).unwrap();
    assert!(r_small.used_dense_fallback);
    assert_eq!(r_small.iterations, 0);

    // Large edgeless graph (triggers Lanczos invariant-subspace recovery for L_sym = I)
    let edgeless = CsrGraph::<f64>::from_edges(60, &[], None, false, false).unwrap();
    let r_iso = smallest_sym_laplacian_eigenpairs(&edgeless, 3, 1e-6, 100, 42).unwrap();
    assert!(!r_iso.used_dense_fallback);
    assert!(r_iso.iterations >= 1);
    assert_eq!(r_iso.num_modes, 3);
    for &lam in &r_iso.lambdas {
        assert!((lam - 1.0).abs() < 1e-10);
    }

    // Disconnected graph: two 30-cycles (two zero eigenvalues)
    let mut edges = Vec::new();
    for i in 0..30 {
        edges.push((i, (i + 1) % 30));
        edges.push((30 + i, 30 + ((i + 1) % 30)));
    }
    let disc = CsrGraph::<f64>::from_edges(60, &edges, None, false, false).unwrap();
    let rd = smallest_sym_laplacian_eigenpairs(&disc, 3, 1e-6, 300, 42).unwrap();
    assert!(rd.lambdas[0] < 1e-10);
    assert!(rd.lambdas[1] > 1e-3);
}

#[test]
fn test_lanczos_operator_largest_smallest_and_builders() {
    use crate::operator::{FnOperator, LinearOperator};

    let solver = LanczosEigen::default()
        .with_tol(1e-8)
        .with_max_iters(400)
        .with_subspace_dim(Some(32))
        .with_seed(99);
    assert_eq!(solver.tol, 1e-8);
    assert_eq!(solver.max_iters, 400);
    assert_eq!(solver.subspace_dim, Some(32));
    assert_eq!(solver.seed, 99);

    let grid = make_grid_2d(10, 10, false);
    let (mut l_sym, _) = symmetric_normalized_laplacian(&grid);
    let n = l_sym.num_nodes();

    let (evals_op, evecs_op) = solver.largest_eigenpairs_operator(&mut l_sym, 3).unwrap();
    let (evals_fn, evecs_fn) = solver
        .largest_eigenpairs(n, 3, |x, y| l_sym.spmv(x, y))
        .unwrap();
    assert_eq!(evals_op, evals_fn);
    assert_eq!(evecs_op, evecs_fn);

    // Smallest eigenpairs with explicit shift and auto-estimated shift (None)
    let (small_explicit, vecs_explicit) = solver
        .smallest_eigenpairs_operator(&mut l_sym, 4, Some(2.5))
        .unwrap();
    let (small_auto, vecs_auto) = solver
        .smallest_eigenpairs_operator(&mut l_sym, 4, None)
        .unwrap();
    assert!(small_explicit[0].abs() < 1e-8);
    for m in 0..4 {
        assert!((small_explicit[m] - small_auto[m]).abs() < 1e-7);
    }
    let cos_min = min_principal_angle_cosine(&vecs_explicit, &vecs_auto, n, 4);
    assert!(cos_min > 0.9999);

    // Negative-definite diagonal matrix-free operator via dyn LinearOperator<f64>
    let diag: Vec<f64> = (0..80).map(|i| -((i + 1) as f64)).collect();
    let mut neg_op: Box<dyn LinearOperator<f64>> =
        Box::new(FnOperator::new(80, move |x: &[f64], y: &mut [f64]| {
            for i in 0..80 {
                y[i] = diag[i] * x[i];
            }
            Ok(())
        }));
    let (neg_small, _) = solver
        .smallest_eigenpairs_operator(&mut neg_op, 3, None)
        .unwrap();
    assert!((neg_small[0] - (-80.0)).abs() < 1e-6);
    assert!((neg_small[1] - (-79.0)).abs() < 1e-6);
    assert!((neg_small[2] - (-78.0)).abs() < 1e-6);

    assert!(
        solver
            .smallest_eigenpairs_operator(&mut neg_op, 3, Some(f64::NAN))
            .is_err()
    );
}

#[test]
fn test_smallest_sym_laplacian_with_config_unit_graph_and_shift_invert() {
    use crate::operator::FnOperator;

    // CsrGraph<()> works directly for Laplacians and smallest_sym_laplacian_eigenpairs_with_config
    let mut edges = Vec::new();
    let rows = 10usize;
    let cols = 10usize;
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
    let g_unit = CsrGraph::<()>::from_unweighted_edges(100, &edges, false, false).unwrap();
    let g_f64 = CsrGraph::<f64>::from_edges(100, &edges, None, false, false).unwrap();

    let cfg = LanczosEigen::default()
        .with_tol(1e-7)
        .with_max_iters(400)
        .with_subspace_dim(Some(24))
        .with_seed(42);
    // Request n_modes = 3 ((1,0), (0,1), (1,1)) to avoid splitting the degenerate (2,0)/(0,2) pair.
    let n_modes = 3;
    let res_unit = smallest_sym_laplacian_eigenpairs_with_config(&g_unit, n_modes, &cfg).unwrap();
    let res_f64 = smallest_sym_laplacian_eigenpairs_with_config(&g_f64, n_modes, &cfg).unwrap();
    assert_eq!(res_unit, res_f64);
    assert!(!res_unit.used_dense_fallback);

    // Exact (L_sym + sigma * I)^{-1} operator via spectral decomposition of L_sym
    let (l_sym, inv_sqrt_deg) = symmetric_normalized_laplacian(&g_unit);
    let n = g_unit.num_nodes();
    let dense = l_sym.to_dense().unwrap();
    let (evals, evecs) = symmetric_eigen_jacobi(&dense, n, 120, 1e-13).unwrap();
    let sigma = 0.05;
    let mut tmp_proj = vec![0.0_f64; n];
    let mut inv_op = FnOperator::new(n, |x: &[f64], y: &mut [f64]| {
        tmp_proj.fill(0.0);
        for row in 0..n {
            let xr = x[row];
            for k in 0..n {
                tmp_proj[k] += evecs[row * n + k] * xr;
            }
        }
        for k in 0..n {
            tmp_proj[k] /= evals[k] + sigma;
        }
        for row in 0..n {
            let mut acc = 0.0_f64;
            for k in 0..n {
                acc += evecs[row * n + k] * tmp_proj[k];
            }
            y[row] = acc;
        }
        Ok(())
    });

    let res_si =
        smallest_eigenpairs_from_inverse_operator(&mut inv_op, inv_sqrt_deg, n_modes, sigma, &cfg)
            .unwrap();
    assert!(!res_si.used_dense_fallback);
    assert_eq!(res_si.num_nodes, n);
    assert_eq!(res_si.num_modes, n_modes);
    for m in 0..n_modes {
        assert!(
            (res_si.lambdas[m] - res_unit.lambdas[m]).abs() < 1e-6,
            "shift-invert lambda[{m}] mismatch: {} vs {}",
            res_si.lambdas[m],
            res_unit.lambdas[m]
        );
    }
    let cos_min = min_principal_angle_cosine(&res_si.vectors, &res_unit.vectors, n, n_modes);
    assert!(cos_min > 0.9999, "shift-invert principal cosine {cos_min}");

    // Validation errors on smallest_eigenpairs_from_inverse_operator
    assert!(
        smallest_eigenpairs_from_inverse_operator(&mut inv_op, vec![1.0; n], 0, sigma, &cfg)
            .is_err()
    );
    assert!(
        smallest_eigenpairs_from_inverse_operator(&mut inv_op, vec![1.0; n], 4, 0.0, &cfg).is_err()
    );
    assert!(
        smallest_eigenpairs_from_inverse_operator(&mut inv_op, vec![1.0; n], 4, f64::NAN, &cfg)
            .is_err()
    );
    assert!(
        smallest_eigenpairs_from_inverse_operator(&mut inv_op, vec![1.0; n - 1], 4, sigma, &cfg)
            .is_err()
    );
    let mut bad_inv = vec![1.0; n];
    bad_inv[0] = f64::NAN;
    assert!(
        smallest_eigenpairs_from_inverse_operator(&mut inv_op, bad_inv, 4, sigma, &cfg).is_err()
    );
}
