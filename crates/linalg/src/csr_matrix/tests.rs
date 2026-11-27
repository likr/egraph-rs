use super::*;
use crate::eigen::symmetric_eigen_jacobi;
use egraph_core::CsrGraph;

fn path_graph(n: usize) -> CsrGraph<f64> {
    let edges: Vec<(u32, u32)> = (0..n.saturating_sub(1))
        .map(|i| (i as u32, (i + 1) as u32))
        .collect();
    CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
}

#[test]
fn test_laplacians_and_spmv_spmm() {
    // 4-node graph with isolated node 3: triangle 0-1-2 + isolated 3
    let g = CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 0)], None, false, false).unwrap();
    let (l_sym, inv_sqrt) = symmetric_normalized_laplacian(&g);
    assert_eq!(l_sym.num_nodes(), 4);
    assert_eq!(l_sym.diagonal(), &[1.0, 1.0, 1.0, 1.0]);
    assert!((inv_sqrt[0] - (0.5_f64).sqrt()).abs() < 1e-12);
    assert_eq!(inv_sqrt[3], 0.0);

    // Stationary vector [sqrt(2), sqrt(2), sqrt(2), 0] is in nullspace of L_sym
    let x = vec![2.0_f64.sqrt(), 2.0_f64.sqrt(), 2.0_f64.sqrt(), 0.0];
    let mut y = vec![0.0; 4];
    l_sym.spmv(&x, &mut y).unwrap();
    for &val in &y {
        assert!(val.abs() < 1e-12);
    }

    // Isolated node unit vector maps to 1.0 under L_sym
    let e3 = vec![0.0, 0.0, 0.0, 1.0];
    l_sym.spmv(&e3, &mut y).unwrap();
    assert_eq!(y, vec![0.0, 0.0, 0.0, 1.0]);

    // Test SpMM with 2 columns
    let x_batch = vec![
        x[0], e3[0], // row 0
        x[1], e3[1], // row 1
        x[2], e3[2], // row 2
        x[3], e3[3], // row 3
    ];
    let mut y_batch = vec![0.0; 8];
    l_sym.spmm(&x_batch, &mut y_batch, 2).unwrap();
    for i in 0..4 {
        assert!(y_batch[2 * i].abs() < 1e-12);
        assert!((y_batch[2 * i + 1] - e3[i]).abs() < 1e-12);
    }

    // Shifted Laplacian L_tilde = L_sym - I has 0 diagonal
    let l_tilde = shifted_normalized_laplacian(&l_sym, 2.0).unwrap();
    assert_eq!(l_tilde.diagonal(), &[0.0, 0.0, 0.0, 0.0]);
    let mut y_tilde = vec![0.0; 4];
    l_tilde.spmv(&e3, &mut y_tilde).unwrap();
    assert_eq!(y_tilde, vec![0.0, 0.0, 0.0, 0.0]);

    // Combinatorial and random-walk Laplacians
    let l_comb = laplacian(&g, LaplacianKind::Combinatorial);
    assert_eq!(l_comb.diagonal(), &[2.0, 2.0, 2.0, 0.0]);
    l_comb.spmv(&[1.0, 1.0, 1.0, 5.0], &mut y).unwrap();
    assert_eq!(y, vec![0.0, 0.0, 0.0, 0.0]);

    let l_rw = laplacian(&g, LaplacianKind::RandomWalk);
    assert_eq!(l_rw.diagonal(), &[1.0, 1.0, 1.0, 0.0]);
    l_rw.spmv(&[1.0, 1.0, 1.0, 5.0], &mut y).unwrap();
    assert_eq!(y, vec![0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_weighted_and_filtered_laplacian() {
    // Negative / zero / non-positive weights are ignored cleanly
    let g = CsrGraph::<f64>::from_edges(
        3,
        &[(0, 1), (1, 2), (0, 2)],
        Some(&[4.0, 0.0, -2.0]),
        false,
        false,
    )
    .unwrap();
    let (l_sym, inv_sqrt) = symmetric_normalized_laplacian(&g);
    assert!((inv_sqrt[0] - 0.5).abs() < 1e-12);
    assert!((inv_sqrt[1] - 0.5).abs() < 1e-12);
    assert_eq!(inv_sqrt[2], 0.0);
    let dense = l_sym.to_dense().unwrap();
    assert!((dense[1] - (-1.0)).abs() < 1e-12);
    assert!((dense[3] - (-1.0)).abs() < 1e-12);
    assert_eq!(dense[8], 1.0);
}

#[test]
fn test_spectral_moment_stats_exact_frobenius_identity() {
    let g = path_graph(6);
    let stats = spectral_moment_stats(&g);
    let (l_sym, _) = symmetric_normalized_laplacian(&g);
    let dense = l_sym.to_dense().unwrap();
    let n = g.num_nodes() as f64;
    let frob_sq: f64 = dense.iter().map(|&v| v * v).sum();
    let trace: f64 = (0..g.num_nodes()).map(|i| dense[i * 6 + i]).sum();
    let mu1 = trace / n;
    let expected_var = frob_sq / n - mu1 * mu1;
    assert!(
        (stats.sigma_lambda_sq - expected_var).abs() < 1e-12,
        "got {} vs expected {expected_var}",
        stats.sigma_lambda_sq
    );
    assert!((stats.sigma_lambda - expected_var.sqrt()).abs() < 1e-12);
    assert!((stats.mean_degree - (10.0 / 6.0)).abs() < 1e-12);
    assert!((stats.leaf_fraction - (2.0 / 6.0)).abs() < 1e-12);
}

#[test]
fn test_spectral_moment_stats_unit_weight_and_directed_topology() {
    let edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)];
    let g_unit = CsrGraph::<()>::from_unweighted_edges(6, &edges, false, false).unwrap();
    let g_f64 = path_graph(6);
    assert_eq!(
        spectral_moment_stats(&g_unit),
        spectral_moment_stats(&g_f64)
    );

    // Directed graph with reverse edges matches simple undirected stats.
    let dir_edges = [(0, 1), (1, 0), (1, 2), (2, 3), (3, 4), (4, 5)];
    let g_dir = CsrGraph::<()>::from_unweighted_edges(6, &dir_edges, true, false).unwrap();
    assert_eq!(
        spectral_moment_stats(&g_dir),
        spectral_moment_stats(&g_unit)
    );

    // Directed graph constructed with sanitize=true (self-loops/duplicates stripped) matches too.
    let dir_dirty = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 5),
    ];
    let g_dir_sanitized = CsrGraph::<()>::from_unweighted_edges(6, &dir_dirty, true, true).unwrap();
    assert_eq!(
        spectral_moment_stats(&g_dir_sanitized),
        spectral_moment_stats(&g_unit)
    );

    // Empty and singleton unit-weight graphs
    let empty_unit = CsrGraph::<()>::from_unweighted_edges(0, &[], false, false).unwrap();
    assert_eq!(spectral_moment_stats(&empty_unit).sigma_lambda_sq, 0.0);
    let single_unit = CsrGraph::<()>::from_unweighted_edges(1, &[], false, false).unwrap();
    assert_eq!(spectral_moment_stats(&single_unit).mean_degree, 0.0);
    assert_eq!(spectral_moment_stats(&single_unit).sigma_lambda_sq, 0.0);
}

#[test]
fn test_empty_and_singleton_csr_matrix() {
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let (l_empty, inv_empty) = symmetric_normalized_laplacian(&empty);
    assert_eq!(l_empty.num_nodes(), 0);
    assert_eq!(l_empty.nnz(), 0);
    assert!(inv_empty.is_empty());
    let empty_stats = spectral_moment_stats(&empty);
    assert_eq!(empty_stats.sigma_lambda_sq, 0.0);
    assert!((empty_stats.sigma_lambda - 1e-3).abs() < 1e-12);

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let (l_single, inv_single) = symmetric_normalized_laplacian(&single);
    assert_eq!(l_single.num_nodes(), 1);
    assert_eq!(l_single.diagonal(), &[1.0]);
    assert_eq!(inv_single, vec![0.0]);
    let single_stats = spectral_moment_stats(&single);
    assert_eq!(single_stats.sigma_lambda_sq, 0.0);
    assert!((single_stats.sigma_lambda - 1e-3).abs() < 1e-12);

    assert!(shifted_normalized_laplacian(&l_single, 0.0).is_err());
    assert!(shifted_normalized_laplacian(&l_single, f64::NAN).is_err());
}

#[test]
fn test_is_symmetric_and_symmetrize() {
    // Asymmetric 3x3 matrix:
    // [ 2.0, -1.0,  0.0]
    // [-3.0,  4.0,  2.0]
    // [ 0.0, -2.0,  1.0]
    let asym = CsrMatrix::from_csr(
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![2.0, -1.0, -3.0, 4.0, 2.0, -2.0, 1.0],
    )
    .unwrap();
    assert!(!asym.is_symmetric(1e-12));
    assert!(!asym.is_symmetric(1.5));
    assert!(asym.is_symmetric(4.0));
    assert!(!asym.is_symmetric(-1.0));
    assert!(!asym.is_symmetric(f64::NAN));

    let sym = asym.symmetrize().unwrap();
    assert!(sym.is_symmetric(0.0));
    assert_eq!(sym.diagonal(), &[2.0, 4.0, 1.0]);
    // (1, 2) and (2, 1) averaged to (2 + (-2)) * 0.5 = 0.0 and should be pruned!
    let dense = sym.to_dense().unwrap();
    assert_eq!(dense, vec![2.0, -2.0, 0.0, -2.0, 4.0, 0.0, 0.0, 0.0, 1.0]);
    assert_eq!(sym.nnz(), 5);

    let empty = CsrMatrix::from_csr(0, vec![0], vec![], vec![]).unwrap();
    assert!(empty.is_symmetric(0.0));
    assert_eq!(empty.symmetrize().unwrap().num_nodes(), 0);
}

#[test]
fn test_anisotropic_laplacian_special_cases_and_psd() {
    // Weighted graph with hub and isolated vertex 4
    let g = CsrGraph::<f64>::from_edges(
        5,
        &[(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)],
        Some(&[2.0, 1.0, 3.0, 1.5, 0.5]),
        false,
        false,
    )
    .unwrap();

    // 1. (0, 0) recovers combinatorial_laplacian bit-for-bit
    let l_comb = combinatorial_laplacian(&g);
    let res_comb = anisotropic_laplacian(
        &g,
        AnisotropicLaplacianConfig {
            alpha: 0.0,
            beta: 0.0,
            degree_reg: 0.0,
            degree_cap: None,
            reweighted_degree_norm: true,
        },
    )
    .unwrap();
    assert_eq!(res_comb.matrix, l_comb);
    assert_eq!(res_comb.nullspace, vec![1.0; 5]);

    // 2. (0, -0.5, reweighted=true) recovers symmetric_normalized_laplacian
    let (l_sym, _) = symmetric_normalized_laplacian(&g);
    let res_sym = anisotropic_laplacian(
        &g,
        AnisotropicLaplacianConfig {
            alpha: 0.0,
            beta: -0.5,
            degree_reg: 0.0,
            degree_cap: None,
            reweighted_degree_norm: true,
        },
    )
    .unwrap();
    assert_eq!(res_sym.matrix, l_sym);
    let mut y_ns = vec![0.0; 5];
    res_sym.matrix.spmv(&res_sym.nullspace, &mut y_ns).unwrap();
    for &v in &y_ns {
        assert!(v.abs() < 1e-13);
    }

    // 3. Connected graph: (0, -0.5, reweighted=false) also matches L_sym
    let g_conn = path_graph(6);
    let (l_sym_conn, _) = symmetric_normalized_laplacian(&g_conn);
    let res_sym_unrew = anisotropic_laplacian(
        &g_conn,
        AnisotropicLaplacianConfig {
            alpha: 0.0,
            beta: -0.5,
            degree_reg: 0.0,
            degree_cap: None,
            reweighted_degree_norm: false,
        },
    )
    .unwrap();
    let diff_sym: f64 = l_sym_conn
        .to_dense()
        .unwrap()
        .iter()
        .zip(res_sym_unrew.matrix.to_dense().unwrap().iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(diff_sym < 1e-14);

    // 4. L_pow(p) = D^{-p}(D - A)D^{-p} at (0.0, -p, reweighted=false)
    let p = 0.6;
    let res_pow = anisotropic_laplacian(
        &g_conn,
        AnisotropicLaplacianConfig {
            alpha: 0.0,
            beta: -p,
            degree_reg: 0.0,
            degree_cap: None,
            reweighted_degree_norm: false,
        },
    )
    .unwrap();
    let l_comb_conn = combinatorial_laplacian(&g_conn).to_dense().unwrap();
    let d_pow: Vec<f64> = (0..6).map(|i| (g_conn.degree(i) as f64).powf(-p)).collect();
    let dense_pow = res_pow.matrix.to_dense().unwrap();
    for i in 0..6 {
        for j in 0..6 {
            let expected = d_pow[i] * l_comb_conn[i * 6 + j] * d_pow[j];
            assert!((dense_pow[i * 6 + j] - expected).abs() < 1e-13);
        }
    }

    // 5. CL_sym at (alpha, -0.5, reweighted=true) is symmetric, PSD, and spectrum in [0, 2]
    for &alpha in &[0.0, 0.25, 0.5, 0.75, 1.0] {
        let cl_sym = anisotropic_laplacian(
            &g,
            AnisotropicLaplacianConfig {
                alpha,
                beta: -0.5,
                degree_reg: 0.0,
                degree_cap: None,
                reweighted_degree_norm: true,
            },
        )
        .unwrap();
        assert!(cl_sym.matrix.is_symmetric(1e-13));
        let dense = cl_sym.matrix.to_dense().unwrap();
        let (evals, _) = symmetric_eigen_jacobi(&dense, 5, 64, 1e-12).unwrap();
        assert!(evals.iter().all(|&ev| (-1e-12..=2.0 + 1e-12).contains(&ev)));
        cl_sym.matrix.spmv(&cl_sym.nullspace, &mut y_ns).unwrap();
        assert!(y_ns.iter().all(|&v| v.abs() < 1e-12));

        // Also verify unnormalized (beta = 0.0) with degree_reg & degree_cap is symmetric PSD
        let cl_reg = anisotropic_laplacian(
            &g,
            AnisotropicLaplacianConfig {
                alpha,
                beta: 0.0,
                degree_reg: 0.5,
                degree_cap: Some(3.0),
                reweighted_degree_norm: true,
            },
        )
        .unwrap();
        assert!(cl_reg.matrix.is_symmetric(1e-13));
        let dense_reg = cl_reg.matrix.to_dense().unwrap();
        let (evals_reg, _) = symmetric_eigen_jacobi(&dense_reg, 5, 64, 1e-12).unwrap();
        assert!(evals_reg.iter().all(|&ev| ev >= -1e-12));
    }

    // 6. Validation errors
    assert!(
        anisotropic_laplacian(
            &g,
            AnisotropicLaplacianConfig {
                alpha: f64::NAN,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        anisotropic_laplacian(
            &g,
            AnisotropicLaplacianConfig {
                degree_reg: -0.1,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        anisotropic_laplacian(
            &g,
            AnisotropicLaplacianConfig {
                degree_cap: Some(0.0),
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn test_spectrum_estimation_gershgorin_power_lanczos_and_trace() {
    // Empty and zero matrices
    let empty = CsrMatrix::from_csr(0, vec![0], vec![], vec![]).unwrap();
    assert_eq!(empty.trace(), 0.0);
    assert_eq!(empty.gershgorin_spectral_bound(), 0.0);
    assert_eq!(empty.estimate_lambda_max(10, 1.02, 42).unwrap(), 0.0);
    assert_eq!(
        empty.estimate_lambda_max_lanczos(16, 1.05, 42).unwrap(),
        0.0
    );

    let edgeless = CsrGraph::<f64>::from_edges(5, &[], None, false, false).unwrap();
    let zero_lap = combinatorial_laplacian(&edgeless);
    assert_eq!(zero_lap.trace(), 0.0);
    assert_eq!(zero_lap.gershgorin_spectral_bound(), 0.0);
    assert_eq!(zero_lap.estimate_lambda_max(10, 1.02, 42).unwrap(), 0.0);
    assert_eq!(
        zero_lap.estimate_lambda_max_lanczos(16, 1.05, 42).unwrap(),
        0.0
    );

    // Star graph K_{1, 50}: L_comb has exact lambda_max = 51.0, trace = 100.0
    let n_star = 51usize;
    let star_edges: Vec<(u32, u32)> = (1..n_star).map(|i| (0, i as u32)).collect();
    let star = CsrGraph::<f64>::from_edges(n_star, &star_edges, None, false, false).unwrap();
    let l_star = combinatorial_laplacian(&star);
    assert!((l_star.trace() - 100.0).abs() < 1e-12);
    let gersh = l_star.gershgorin_spectral_bound();
    assert!((gersh - 100.0).abs() < 1e-12);
    let pi = l_star.estimate_lambda_max(10, 1.02, 42).unwrap();
    let lanc = l_star.estimate_lambda_max_lanczos(16, 1.05, 42).unwrap();
    assert!(pi >= 50.0 && pi <= gersh);
    assert!(lanc >= 51.0 && lanc <= (1.05_f64 * 51.0 + 1e-9).min(gersh));

    // 2-node graph (tests constant-vector Rademacher edge case!)
    let g2 = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
    let l2 = combinatorial_laplacian(&g2);
    assert!((l2.estimate_lambda_max_lanczos(16, 1.05, 42).unwrap() - 2.0).abs() < 1e-12);
    assert!(l2.estimate_lambda_max(10, 1.02, 42).unwrap() >= 2.0 - 1e-12);

    // Validation
    assert!(l_star.estimate_lambda_max(10, 0.99, 42).is_err());
    assert!(l_star.estimate_lambda_max_lanczos(16, 0.99, 42).is_err());
    assert!((l_star.estimate_lambda_max(0, 1.0, 42).unwrap() - gersh).abs() < 1e-12);
    assert!((l_star.estimate_lambda_max_lanczos(0, 1.0, 42).unwrap() - gersh).abs() < 1e-12);

    // Operator spectral bound matches CsrMatrix estimate and handles zero/singleton operators
    let mut l_star_op = &l_star;
    let op_lanc = estimate_lambda_max_operator(&mut l_star_op, 16, 1.05, 42).unwrap();
    assert!((op_lanc - lanc).abs() < 1e-12);
    let mut zero_op = &zero_lap;
    assert_eq!(
        estimate_lambda_max_operator(&mut zero_op, 16, 1.05, 42).unwrap(),
        0.0
    );
    let mut one_op = CsrMatrix::identity(1).unwrap();
    assert!((estimate_lambda_max_operator(&mut one_op, 4, 1.05, 42).unwrap() - 1.05).abs() < 1e-12);
    assert!(estimate_lambda_max_operator(&mut one_op, 0, 1.05, 42).is_err());
    assert!(estimate_lambda_max_operator(&mut one_op, 4, 0.99, 42).is_err());
}

#[test]
fn test_anisotropic_laplacian_config_builders() {
    let cfg = AnisotropicLaplacianConfig::new()
        .with_alpha(1.0)
        .with_beta(-0.5)
        .with_degree_reg(0.25)
        .with_degree_cap(Some(4.0))
        .with_reweighted_degree_norm(false);
    assert_eq!(cfg.alpha, 1.0);
    assert_eq!(cfg.beta, -0.5);
    assert_eq!(cfg.degree_reg, 0.25);
    assert_eq!(cfg.degree_cap, Some(4.0));
    assert!(!cfg.reweighted_degree_norm);
}
