use super::*;
use crate::estimate_lambda_max;
use ndarray::Array2;
use petgraph::graph::UnGraph;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn sparse_to_dense(matrix: &SparseSymmetricMatrix<f64>) -> Array2<f64> {
    let n = matrix.dim();
    let mut dense = Array2::zeros((n, n));
    for i in 0..n {
        dense[[i, i]] = matrix.diagonal()[i];
    }
    for &(i, j, w) in matrix.edges() {
        dense[[i, j]] += w;
        dense[[j, i]] += w;
    }
    dense
}

fn dense_expm_neg(l_dense: &Array2<f64>, t: f64) -> Array2<f64> {
    let n = l_dense.nrows();
    let mut a = l_dense * (-t);
    let max_row_norm = (0..n)
        .map(|i| (0..n).map(|j| a[[i, j]].abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    let s = if max_row_norm > 0.5 {
        max_row_norm.log2().ceil().max(0.0) as u32 + 1
    } else {
        0
    };
    let scale = 2.0_f64.powi(-(s as i32));
    a *= scale;

    let mut res = Array2::eye(n);
    let mut term = Array2::eye(n);
    for k in 1..=30 {
        term = term.dot(&a) / (k as f64);
        res = &res + &term;
    }
    for _ in 0..s {
        res = res.dot(&res);
    }
    res
}

#[test]
fn test_chebyshev_approximation_identity() {
    let mut matrix: SparseSymmetricMatrix<f64> = SparseSymmetricMatrix::new(3);
    matrix.set_diagonal(0, 1.0);
    matrix.set_diagonal(1, 1.0);
    matrix.set_diagonal(2, 1.0);

    let vectors = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let result = chebyshev_approximation(&matrix, 0.0, 10, 1.0, &vectors);

    for i in 0..3 {
        for j in 0..2 {
            assert!((result[[i, j]] - vectors[[i, j]]).abs() < 1e-6);
        }
    }
}

#[test]
fn test_bandpass_coefficients_zero_time_exact() {
    let coeffs = compute_chebyshev_bandpass_coefficients(&[(0.25, 0.0), (-0.1, 0.0)], 2.0, 10);
    assert!((coeffs[0] - 0.15).abs() < 1e-12);
    for &c in &coeffs[1..] {
        assert_eq!(c, 0.0);
    }
}

#[test]
fn test_chebyshev_bandpass_vs_dense_expm() {
    let mut g = UnGraph::<(), ()>::new_undirected();
    let nodes: Vec<_> = (0..5).map(|_| g.add_node(())).collect();
    for &(u, v) in &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)] {
        g.add_edge(nodes[u], nodes[v], ());
    }

    let lap_std = SparseSymmetricMatrix::standard_laplacian(&g, |_| 1.0);
    let lap_sym = SparseSymmetricMatrix::symmetric_normalized_laplacian(&g, |_| 1.0);

    let terms = vec![(1.5, 0.4), (-0.75, 1.8), (0.25, 0.0)];
    let mut rng = StdRng::seed_from_u64(42);

    for lap in [&lap_std, &lap_sym] {
        let lmax = estimate_lambda_max(lap, &mut rng, 100, 1e-6) * 1.05;
        let l_dense = sparse_to_dense(lap);
        let mut k_dense = Array2::<f64>::zeros((5, 5));
        for &(c, t) in &terms {
            k_dense = k_dense + dense_expm_neg(&l_dense, t) * c;
        }

        let eye = Array2::eye(5);
        let k_cheb = chebyshev_bandpass_approximation(lap, &terms, 25, lmax, &eye);
        for i in 0..5 {
            for j in 0..5 {
                assert!(
                    (k_cheb[[i, j]] - k_dense[[i, j]]).abs() < 1e-7,
                    "mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    k_cheb[[i, j]],
                    k_dense[[i, j]]
                );
            }
        }

        let v = Array1::from_vec(vec![1.0, -0.5, 2.0, 0.3, -1.2]);
        let vec_cheb = chebyshev_bandpass_approximation_vec(lap, &terms, 25, lmax, &v);
        let vec_dense = k_dense.dot(&v);
        for i in 0..5 {
            assert!(
                (vec_cheb[i] - vec_dense[i]).abs() < 1e-7,
                "vec mismatch at {}: {} vs {}",
                i,
                vec_cheb[i],
                vec_dense[i]
            );
        }

        // Verify baseline stationary response g_hat(0) = sum c_m = 1.0
        let u1 = lap.stationary_vector();
        let u1_col = u1.clone().insert_axis(ndarray::Axis(1));
        let (kv, w_mat, _) =
            chebyshev_bandpass_approximation_with_baseline(lap, &terms, 25, lmax, &u1_col, &u1);
        let g_hat_0: f64 = terms.iter().map(|&(c, _)| c).sum();
        for i in 0..5 {
            assert!(w_mat[[i, 0]].abs() < 1e-10);
            assert!((kv[[i, 0]] - g_hat_0 * u1[i]).abs() < 1e-10);
        }
    }
}
