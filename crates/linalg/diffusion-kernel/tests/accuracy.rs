#![allow(clippy::needless_range_loop)]

use ndarray::Array1;
use petgraph::graph::UnGraph;
use petgraph_distance::{Laplacian, StandardLaplacian};
use petgraph_linalg_diffusion_kernel::{DiffusionKernel, MultiscaleDiffusionKernel};
use petgraph_linalg_spmv::SparseSymmetricMatrix;
use rand::SeedableRng;

/// Computes exact matrix exponential exp(-t * L) using Taylor series expansion.
/// Designed for small matrices (n <= 30) where high accuracy is required.
fn compute_exact_exp_neg_tl(
    laplacian: &SparseSymmetricMatrix<f64>,
    t: f64,
    num_terms: usize,
) -> Vec<Vec<f64>> {
    let n = laplacian.dim();
    // Initialize result with Identity matrix I
    let mut exp_matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        exp_matrix[i][i] = 1.0;
    }

    // Current term T_k = (-t)^k L^k / k!
    let mut current_term = vec![vec![0.0; n]; n];
    for i in 0..n {
        current_term[i][i] = 1.0;
    }

    for k in 1..num_terms {
        // Compute next_term = (-t / k) * (L @ current_term)
        let mut next_term = vec![vec![0.0; n]; n];
        let factor = -t / (k as f64);

        for j in 0..n {
            // Extract column j of current_term as Array1
            let mut col_j = vec![0.0; n];
            for i in 0..n {
                col_j[i] = current_term[i][j];
            }
            let x_arr = Array1::from_vec(col_j);

            // SpMV: L @ col_j
            let l_col_j = laplacian.multiply(&x_arr);
            for i in 0..n {
                next_term[i][j] = factor * l_col_j[i];
            }
        }

        // Add to exp_matrix
        for i in 0..n {
            for j in 0..n {
                exp_matrix[i][j] += next_term[i][j];
            }
        }

        current_term = next_term;
    }

    exp_matrix
}

/// Computes exact multiscale diffusion vector representation:
/// s_x = sqrt(alpha) * (I - alpha * P)^(-1) e_x * sqrt(deg)
/// and exact distance d(x, y) = || s_x - s_y ||_2.
fn compute_exact_multiscale_distances(
    matrix: &SparseSymmetricMatrix<f64>,
    alpha: f64,
) -> Vec<Vec<f64>> {
    let n = matrix.dim();

    // Compute degrees from matrix diagonal and edges
    let mut degrees = matrix.diagonal().to_vec();
    for &(i, j, w) in matrix.edges() {
        degrees[i] += w;
        degrees[j] += w;
    }

    // Build matrix A = (I - alpha * P) where P_ij = W_ij / deg(i)
    let mut mat_sys = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        mat_sys[i][i] = 1.0;
        if degrees[i] > 0.0 {
            for &(row, col, w) in matrix.edges() {
                if row == i {
                    mat_sys[i][col] -= alpha * w / degrees[i];
                } else if col == i {
                    mat_sys[i][row] -= alpha * w / degrees[i];
                }
            }
        }
    }

    // Invert mat_sys using Gaussian elimination to get (I - alpha * P)^(-1)
    let mut aug = vec![vec![0.0f64; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = mat_sys[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    for i in 0..n {
        // Pivot selection
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[k][i].abs() > aug[max_row][i].abs() {
                max_row = k;
            }
        }
        aug.swap(i, max_row);

        let pivot: f64 = aug[i][i];
        assert!(pivot.abs() > 1e-12, "Matrix system is singular");

        for j in i..(2 * n) {
            aug[i][j] /= pivot;
        }

        for k in 0..n {
            if k != i {
                let factor: f64 = aug[k][i];
                for j in i..(2 * n) {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
    }

    let mut inv_m = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv_m[i][j] = aug[i][n + j];
        }
    }

    // Compute exact representation vector Y_x for each node x:
    // Y_x = ((I - alpha * P)^(-1) - I)_x
    let mut y_vecs = vec![vec![0.0f64; n]; n];
    for x in 0..n {
        for z in 0..n {
            let delta = if x == z { 1.0 } else { 0.0 };
            y_vecs[x][z] = inv_m[x][z] - delta;
        }
    }

    // Compute pairwise distances || Y_x - Y_y ||_2
    let mut distances = vec![vec![0.0f64; n]; n];
    for x in 0..n {
        for y in 0..n {
            let mut sum_sq = 0.0f64;
            for z in 0..n {
                let diff = y_vecs[x][z] - y_vecs[y][z];
                sum_sq += diff * diff;
            }
            distances[x][y] = sum_sq.sqrt();
        }
    }

    distances
}

#[test]
#[ignore = "Performance and accuracy verification test for DiffusionKernel"]
fn test_diffusion_kernel_accuracy() {
    // Construct a path graph with 12 nodes
    let n = 12;
    let mut graph = UnGraph::<(), ()>::new_undirected();
    let nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();
    for i in 0..(n - 1) {
        graph.add_edge(nodes[i], nodes[i + 1], ());
    }

    // Build standard Laplacian L
    let laplacian = StandardLaplacian.build(&graph, &mut |_| 1.0);

    // Compute ground truth matrix exp(-t L) with t = 1.0
    let t = 1.0;
    let exact_exp = compute_exact_exp_neg_tl(&laplacian, t, 50);

    // Compute ground truth diffusion distance matrix
    let mut exact_dist = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let d_sq = (exact_exp[i][i] + exact_exp[j][j] - 2.0 * exact_exp[i][j]).max(0.0);
            exact_dist[i][j] = d_sq.sqrt();
        }
    }

    // Run DiffusionKernel with degree=30, num_vectors=3000
    let degree = 30;
    let num_vectors = 3000;
    let mut rng = rand::rngs::StdRng::seed_from_u64(12345);

    let kernel = DiffusionKernel::new(
        &graph,
        |_| 1.0,
        t,
        degree,
        num_vectors,
        StandardLaplacian,
        &mut rng,
    );

    // Evaluate error statistics
    let mut total_abs_err = 0.0;
    let mut total_sq_err = 0.0;
    let mut total_rel_err = 0.0;
    let mut max_abs_err = 0.0f64;
    let count = (n * (n - 1)) / 2;

    for i in 0..n {
        for j in (i + 1)..n {
            let approx = kernel.distance(i, j);
            let exact = exact_dist[i][j];

            let abs_err = (approx - exact).abs();
            let rel_err = if exact > 1e-8 { abs_err / exact } else { 0.0 };

            total_abs_err += abs_err;
            total_sq_err += abs_err * abs_err;
            total_rel_err += rel_err;
            max_abs_err = max_abs_err.max(abs_err);
        }
    }

    let mean_abs_err = total_abs_err / (count as f64);
    let rmse = (total_sq_err / (count as f64)).sqrt();
    let mean_rel_err = total_rel_err / (count as f64);

    println!("\n=== DiffusionKernel Accuracy Verification Report ===");
    println!("Graph: Path Graph (N = {})", n);
    println!(
        "Parameters: t = {}, degree = {}, num_vectors = {}",
        t, degree, num_vectors
    );
    println!("Mean Absolute Error (MAE) : {:.6}", mean_abs_err);
    println!("Root Mean Square Error    : {:.6}", rmse);
    println!("Max Absolute Error        : {:.6}", max_abs_err);
    println!("Mean Relative Error       : {:.2}%", mean_rel_err * 100.0);
    println!("=====================================================\n");

    // Statistical assertions for accuracy
    assert!(
        mean_abs_err < 0.05,
        "MAE ({:.6}) exceeded acceptable limit 0.05",
        mean_abs_err
    );
    assert!(
        max_abs_err < 0.15,
        "MaxAE ({:.6}) exceeded acceptable limit 0.15",
        max_abs_err
    );
}

#[test]
#[ignore = "Performance and accuracy verification test for MultiscaleDiffusionKernel"]
fn test_multiscale_diffusion_kernel_accuracy() {
    // Construct a cycle graph with 12 nodes
    let n = 12;
    let mut matrix = SparseSymmetricMatrix::new(n);
    for i in 0..n {
        let j = (i + 1) % n;
        let (u, v) = if i < j { (i, j) } else { (j, i) };
        matrix.add_edge(u, v, 1.0);
    }

    let alpha = 0.85;
    let exact_dist = compute_exact_multiscale_distances(&matrix, alpha);

    let num_samples = 4000;
    let tol = 1e-8;
    let max_iter = 200;

    let mut kernel = MultiscaleDiffusionKernel::new(matrix, alpha, num_samples, tol, max_iter);
    let mut rng = rand::rngs::StdRng::seed_from_u64(54321);
    kernel.build_index_with_rng(&mut rng);

    let mut total_abs_err = 0.0;
    let mut total_sq_err = 0.0;
    let mut total_rel_err = 0.0;
    let mut max_abs_err = 0.0f64;
    let count = (n * (n - 1)) / 2;

    for i in 0..n {
        for j in (i + 1)..n {
            let approx = kernel.sample_distance(i, j).unwrap();
            let exact = exact_dist[i][j];

            let abs_err = (approx - exact).abs();
            let rel_err = if exact > 1e-8 { abs_err / exact } else { 0.0 };

            total_abs_err += abs_err;
            total_sq_err += abs_err * abs_err;
            total_rel_err += rel_err;
            max_abs_err = max_abs_err.max(abs_err);
        }
    }

    let mean_abs_err = total_abs_err / (count as f64);
    let rmse = (total_sq_err / (count as f64)).sqrt();
    let mean_rel_err = total_rel_err / (count as f64);

    println!("\n=== MultiscaleDiffusionKernel Accuracy Verification Report ===");
    println!("Graph: Cycle Graph (N = {})", n);
    println!(
        "Parameters: alpha = {}, num_samples = {}, tol = {:e}",
        alpha, num_samples, tol
    );
    println!("Mean Absolute Error (MAE) : {:.6}", mean_abs_err);
    println!("Root Mean Square Error    : {:.6}", rmse);
    println!("Max Absolute Error        : {:.6}", max_abs_err);
    println!("Mean Relative Error       : {:.2}%", mean_rel_err * 100.0);
    println!("==============================================================\n");

    assert!(
        mean_abs_err < 0.05,
        "Multiscale MAE ({:.6}) exceeded acceptable limit 0.05",
        mean_abs_err
    );
    assert!(
        max_abs_err < 0.15,
        "Multiscale MaxAE ({:.6}) exceeded acceptable limit 0.15",
        max_abs_err
    );
}
