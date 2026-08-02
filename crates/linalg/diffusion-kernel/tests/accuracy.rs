#![allow(clippy::needless_range_loop)]

use petgraph::graph::UnGraph;
use petgraph_distance::{Laplacian, SparseSymmetricMatrix, StandardLaplacian};
use petgraph_linalg_diffusion_kernel::{
    DiffusionKernel, ExactDiffusionKernel, MultiscaleDiffusionKernel,
};
use rand::SeedableRng;

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
    struct GraphTestCase {
        name: &'static str,
        laplacian: SparseSymmetricMatrix<f64>,
        n: usize,
    }

    let mut test_cases = Vec::new();

    // 1. Path Graph (Normalized Laplacian, N = 16)
    {
        let n = 16;
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();
        for i in 0..(n - 1) {
            graph.add_edge(nodes[i], nodes[i + 1], ());
        }
        let laplacian = petgraph_distance::SymmetricNormalizedLaplacian.build(&graph, &mut |_| 1.0);
        test_cases.push(GraphTestCase {
            name: "Path Graph (Symmetric Normalized Laplacian, N = 16)",
            laplacian,
            n,
        });
    }

    // 2. Cycle Graph (Normalized Laplacian, N = 16)
    {
        let n = 16;
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();
        for i in 0..n {
            graph.add_edge(nodes[i], nodes[(i + 1) % n], ());
        }
        let laplacian = petgraph_distance::SymmetricNormalizedLaplacian.build(&graph, &mut |_| 1.0);
        test_cases.push(GraphTestCase {
            name: "Cycle Graph (Symmetric Normalized Laplacian, N = 16)",
            laplacian,
            n,
        });
    }

    // 3. 2D Grid Graph (Normalized Laplacian, 4x4, N = 16)
    {
        let rows = 4;
        let cols = 4;
        let n = rows * cols;
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();
        for r in 0..rows {
            for c in 0..cols {
                let u = r * cols + c;
                if c + 1 < cols {
                    graph.add_edge(nodes[u], nodes[u + 1], ());
                }
                if r + 1 < rows {
                    graph.add_edge(nodes[u], nodes[u + cols], ());
                }
            }
        }
        let laplacian = petgraph_distance::SymmetricNormalizedLaplacian.build(&graph, &mut |_| 1.0);
        test_cases.push(GraphTestCase {
            name: "2D Grid Graph 4x4 (Symmetric Normalized Laplacian, N = 16)",
            laplacian,
            n,
        });
    }

    // 4. Path Graph (Standard Laplacian, N = 16)
    {
        let n = 16;
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();
        for i in 0..(n - 1) {
            graph.add_edge(nodes[i], nodes[i + 1], ());
        }
        let laplacian = StandardLaplacian.build(&graph, &mut |_| 1.0);
        test_cases.push(GraphTestCase {
            name: "Path Graph (Standard Laplacian, N = 16)",
            laplacian,
            n,
        });
    }

    let t = 1.0;
    let degree = 30;
    let num_vectors = 200;

    println!("\n==========================================================================================");
    println!("             DIFFUSION KERNEL HUTCHINSON ACCURACY IMPROVEMENT COMPARISON REPORT");
    println!("==========================================================================================");
    println!(
        "Parameters: t = {}, Chebyshev Degree = {}, Hutchinson Samples = {}",
        t, degree, num_vectors
    );

    for tc in test_cases {
        let n = tc.n;

        // Ground truth exp(-t L) via Chebyshev polynomial expansion
        let exact_kernel = ExactDiffusionKernel::new(&tc.laplacian, t, 60);

        // Ground truth pairwise diffusion distances
        let mut exact_dist = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let d_sq = (exact_kernel.get(i, i) + exact_kernel.get(j, j)
                    - 2.0 * exact_kernel.get(i, j))
                .max(0.0);
                exact_dist[i][j] = d_sq.sqrt();
            }
        }

        // --- AFTER: DiffusionKernel with Baseline Subtraction ---
        let mut rng_after = rand::rngs::StdRng::seed_from_u64(42);
        let kernel_after =
            DiffusionKernel::new(&tc.laplacian, t, degree, num_vectors, &mut rng_after);

        // --- BEFORE: Naive Hutchinson (without Baseline Subtraction) ---
        let mut rng_before = rand::rngs::StdRng::seed_from_u64(42);
        // Compute naive diagonal manually using raw V * KV
        let v_rand = petgraph_linalg_diffusion_kernel::hutchinson::generate_rademacher_vectors(
            n,
            num_vectors,
            &mut rng_before,
        );
        let lambda_max = petgraph_linalg_diffusion_kernel::power_method::estimate_lambda_max(
            &tc.laplacian,
            &mut rng_before,
            100,
            1e-6,
        );
        let kv_rand = petgraph_linalg_diffusion_kernel::chebyshev::chebyshev_approximation(
            &tc.laplacian,
            t,
            degree,
            lambda_max,
            &v_rand,
        );
        let estimator_before =
            petgraph_linalg_diffusion_kernel::HutchinsonEstimator::new(v_rand, kv_rand);

        // 1. Evaluate Diagonal K_ii Error
        let mut diag_naive_mae = 0.0;
        let mut diag_naive_sq = 0.0;
        let mut diag_sub_mae = 0.0;
        let mut diag_sub_sq = 0.0;

        for i in 0..n {
            let err_diag_sub = (kernel_after.get(i, i) - exact_kernel.get(i, i)).abs();
            diag_sub_mae += err_diag_sub;
            diag_sub_sq += err_diag_sub * err_diag_sub;

            let err_diag_naive =
                (estimator_before.query_diagonal(i) - exact_kernel.get(i, i)).abs();
            diag_naive_mae += err_diag_naive;
            diag_naive_sq += err_diag_naive * err_diag_naive;
        }

        diag_naive_mae /= n as f64;
        let diag_naive_rmse = (diag_naive_sq / n as f64).sqrt();

        diag_sub_mae /= n as f64;
        let diag_sub_rmse = (diag_sub_sq / n as f64).sqrt();

        // 2. Evaluate Distance Error
        let count = (n * (n - 1)) / 2;
        let mut dist_naive_mae = 0.0;
        let mut dist_naive_sq = 0.0;
        let mut dist_sub_mae = 0.0;
        let mut dist_sub_sq = 0.0;

        for i in 0..n {
            for j in (i + 1)..n {
                let exact_d = exact_dist[i][j];

                let approx_d_naive = estimator_before.distance(i, j);
                let approx_d_sub = kernel_after.distance(i, j);

                let err_d_naive = (approx_d_naive - exact_d).abs();
                let err_d_sub = (approx_d_sub - exact_d).abs();

                dist_naive_mae += err_d_naive;
                dist_naive_sq += err_d_naive * err_d_naive;

                dist_sub_mae += err_d_sub;
                dist_sub_sq += err_d_sub * err_d_sub;
            }
        }

        dist_naive_mae /= count as f64;
        let dist_naive_rmse = (dist_naive_sq / count as f64).sqrt();

        dist_sub_mae /= count as f64;
        let dist_sub_rmse = (dist_sub_sq / count as f64).sqrt();

        let diag_mae_reduction = (1.0 - (diag_sub_mae / diag_naive_mae)) * 100.0;
        let dist_mae_reduction = (1.0 - (dist_sub_mae / dist_naive_mae)) * 100.0;

        println!("\nTopology: {}", tc.name);
        println!("------------------------------------------------------------------------------------------");
        println!(
            " Metric                   | Before (Naive) | After (Baseline Sub) | Error Reduction"
        );
        println!("------------------------------------------------------------------------------------------");
        println!(
            " Diagonal MAE (K_ii)      | {:.6}       | {:.6}             | {:.2}% reduction ({:.2}x better)",
            diag_naive_mae,
            diag_sub_mae,
            diag_mae_reduction,
            diag_naive_mae / diag_sub_mae
        );
        println!(
            " Diagonal RMSE            | {:.6}       | {:.6}             | {:.2}% reduction",
            diag_naive_rmse,
            diag_sub_rmse,
            (1.0 - (diag_sub_rmse / diag_naive_rmse)) * 100.0
        );
        println!(
            " Distance MAE d(i,j)      | {:.6}       | {:.6}             | {:.2}% reduction ({:.2}x better)",
            dist_naive_mae,
            dist_sub_mae,
            dist_mae_reduction,
            dist_naive_mae / dist_sub_mae
        );
        println!(
            " Distance RMSE            | {:.6}       | {:.6}             | {:.2}% reduction",
            dist_naive_rmse,
            dist_sub_rmse,
            (1.0 - (dist_sub_rmse / dist_naive_rmse)) * 100.0
        );
        println!("------------------------------------------------------------------------------------------");

        // Assert that Baseline Subtraction strictly improves accuracy
        assert!(
            diag_sub_mae < diag_naive_mae,
            "Baseline Subtraction should reduce diagonal MAE"
        );
        assert!(
            dist_sub_mae < dist_naive_mae,
            "Baseline Subtraction should reduce distance MAE"
        );
    }
    println!("==========================================================================================\n");
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

#[test]
fn test_single_source_heat_vector_and_pivot_distance() {
    let n = 6;
    let mut graph = UnGraph::<(), ()>::new_undirected();
    let nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();
    for i in 0..(n - 1) {
        graph.add_edge(nodes[i], nodes[i + 1], ());
    }

    let laplacian = StandardLaplacian.build(&graph, &mut |_| 1.0);
    let t = 0.5;
    let degree = 20;

    let pivot = 0;
    let heat_vec = DiffusionKernel::single_source_heat_vector(&laplacian, t, degree, pivot);

    assert_eq!(heat_vec.len(), n);
    assert!(heat_vec[0] > 0.0, "Pivot heat should be positive");
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let kernel = DiffusionKernel::new(&laplacian, t, degree, 50, &mut rng);
    use petgraph_linalg_diffusion_kernel::PivotDiffusionDistanceMatrix;
    let dist_vec =
        PivotDiffusionDistanceMatrix::pivot_distance_vector(&laplacian, &kernel, t, degree, pivot);

    assert_eq!(dist_vec[0], 0.0, "Distance to self should be 0");
    assert!(dist_vec[1] > 0.0, "Distance to neighbor should be positive");
    assert!(
        dist_vec[2] > dist_vec[1],
        "Distance should increase with path length"
    );
}
