#![allow(clippy::needless_range_loop)]

use petgraph::graph::UnGraph;
use petgraph_distance::{
    Laplacian, SparseSymmetricMatrix, StandardLaplacian, SymmetricNormalizedLaplacian,
};
use petgraph_linalg_diffusion_kernel::{
    DiffusionKernel, ExactDiffusionKernel, LowRankDiffusionKernel,
};
use rand::SeedableRng;
use std::time::Instant;

struct TestTopology {
    name: String,
    n: usize,
    laplacian_std: SparseSymmetricMatrix<f64>,
    laplacian_norm: SparseSymmetricMatrix<f64>,
}

fn create_path_graph(n: usize) -> TestTopology {
    let mut g = UnGraph::<(), ()>::new_undirected();
    let nodes: Vec<_> = (0..n).map(|_| g.add_node(())).collect();
    for i in 0..(n - 1) {
        g.add_edge(nodes[i], nodes[i + 1], ());
    }
    let laplacian_std = StandardLaplacian.build(&g, &mut |_| 1.0);
    let laplacian_norm = SymmetricNormalizedLaplacian.build(&g, &mut |_| 1.0);
    TestTopology {
        name: format!("Path Graph (N={})", n),
        n,
        laplacian_std,
        laplacian_norm,
    }
}

fn create_cycle_graph(n: usize) -> TestTopology {
    let mut g = UnGraph::<(), ()>::new_undirected();
    let nodes: Vec<_> = (0..n).map(|_| g.add_node(())).collect();
    for i in 0..n {
        g.add_edge(nodes[i], nodes[(i + 1) % n], ());
    }
    let laplacian_std = StandardLaplacian.build(&g, &mut |_| 1.0);
    let laplacian_norm = SymmetricNormalizedLaplacian.build(&g, &mut |_| 1.0);
    TestTopology {
        name: format!("Cycle Graph (N={})", n),
        n,
        laplacian_std,
        laplacian_norm,
    }
}

fn create_grid_graph(rows: usize, cols: usize) -> TestTopology {
    let n = rows * cols;
    let mut g = UnGraph::<(), ()>::new_undirected();
    let nodes: Vec<_> = (0..n).map(|_| g.add_node(())).collect();
    for r in 0..rows {
        for c in 0..cols {
            let u = r * cols + c;
            if c + 1 < cols {
                g.add_edge(nodes[u], nodes[u + 1], ());
            }
            if r + 1 < rows {
                g.add_edge(nodes[u], nodes[u + cols], ());
            }
        }
    }
    let laplacian_std = StandardLaplacian.build(&g, &mut |_| 1.0);
    let laplacian_norm = SymmetricNormalizedLaplacian.build(&g, &mut |_| 1.0);
    TestTopology {
        name: format!("Grid Graph ({}x{}, N={})", rows, cols, n),
        n,
        laplacian_std,
        laplacian_norm,
    }
}

#[test]
fn test_low_rank_kernel_basic_properties() {
    let topology = create_cycle_graph(10);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let t = 1.0;
    let rank = 4;

    let kernel = LowRankDiffusionKernel::new(&topology.laplacian_std, t, rank, &mut rng);

    assert_eq!(kernel.n(), 10);
    assert_eq!(kernel.t(), t);
    assert_eq!(kernel.rank(), rank);

    // Diagonal elements K_ii should be positive
    for i in 0..10 {
        assert!(
            kernel.get(i, i) > 0.0,
            "Diagonal heat kernel should be positive"
        );
    }

    // Distance to self should be 0.0
    for i in 0..10 {
        assert_eq!(kernel.distance(i, i), 0.0);
    }

    // Single source pivot distance vector
    let dist_vec = kernel.pivot_distance_vector(0);
    assert_eq!(dist_vec.len(), 10);
    assert_eq!(dist_vec[0], 0.0);
    assert!(dist_vec[1] > 0.0);
}

#[test]
#[ignore = "Comprehensive benchmark report comparing Hutchinson vs Low-Rank (rDMDS) against exact exp(-tL)"]
fn benchmark_hutchinson_vs_low_rank_accuracy_and_speed() {
    let t = 1.0;
    let cheby_degree = 30;
    let num_vectors = 32;

    let test_topologies = vec![
        create_path_graph(20),
        create_cycle_graph(20),
        create_grid_graph(5, 5),
    ];

    println!("\n==========================================================================================");
    println!(
        "        DIFFUSION KERNEL: HUTCHINSON ESTIMATION vs LOW-RANK (rDMDS) BENCHMARK REPORT"
    );
    println!("==========================================================================================");
    println!(
        "Parameters: t = {}, Chebyshev Degree = {}, Hutchinson Samples = {}",
        t, cheby_degree, num_vectors
    );

    for (lap_type_name, is_norm) in [
        ("Standard Laplacian (L)", false),
        ("Normalized Laplacian (L_sym)", true),
    ] {
        println!("\n>>> LAPLACIAN TYPE: {} <<<", lap_type_name);

        for tc in &test_topologies {
            let n = tc.n;
            let laplacian = if is_norm {
                &tc.laplacian_norm
            } else {
                &tc.laplacian_std
            };

            // 1. Ground truth exact matrix exponential exp(-t L) via Chebyshev polynomial expansion
            let exact_kernel = ExactDiffusionKernel::new(laplacian, t, 60);

            // Ground truth pairwise exact heat distances
            let mut exact_dist = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    exact_dist[i][j] = exact_kernel.distance(i, j);
                }
            }

            // 2. Hutchinson Estimation Kernel
            let mut rng_hutch = rand::rngs::StdRng::seed_from_u64(42);
            let start_hutch = Instant::now();
            let kernel_hutch =
                DiffusionKernel::new(laplacian, t, cheby_degree, num_vectors, &mut rng_hutch);
            let duration_hutch = start_hutch.elapsed();

            // Evaluate Hutchinson Accuracy against exact exp(-t L)
            let mut hutch_diag_mae = 0.0;
            let mut hutch_diag_sq = 0.0;
            let mut hutch_offdiag_mae = 0.0;
            let mut hutch_offdiag_sq = 0.0;
            let mut hutch_dist_mae = 0.0;
            let mut hutch_dist_sq = 0.0;
            let mut hutch_dist_maxae = 0.0f64;

            let pair_count = (n * (n - 1)) / 2;

            for i in 0..n {
                let err_diag = (kernel_hutch.get(i, i) - exact_kernel.get(i, i)).abs();
                hutch_diag_mae += err_diag;
                hutch_diag_sq += err_diag * err_diag;

                for j in (i + 1)..n {
                    let err_offdiag = (kernel_hutch.get(i, j) - exact_kernel.get(i, j)).abs();
                    hutch_offdiag_mae += err_offdiag;
                    hutch_offdiag_sq += err_offdiag * err_offdiag;

                    let approx_d = kernel_hutch.distance(i, j);
                    let err_d = (approx_d - exact_dist[i][j]).abs();
                    hutch_dist_mae += err_d;
                    hutch_dist_sq += err_d * err_d;
                    hutch_dist_maxae = hutch_dist_maxae.max(err_d);
                }
            }

            hutch_diag_mae /= n as f64;
            let _hutch_diag_rmse = (hutch_diag_sq / n as f64).sqrt();
            hutch_offdiag_mae /= pair_count as f64;
            let _hutch_offdiag_rmse = (hutch_offdiag_sq / pair_count as f64).sqrt();
            hutch_dist_mae /= pair_count as f64;
            let hutch_dist_rmse = (hutch_dist_sq / pair_count as f64).sqrt();

            println!("\nTopology: {}", tc.name);
            println!("---------------------------------------------------------------------------------------------------------------");
            println!(" Method               | Build Time  | Diag MAE (Kii) | Off-Diag MAE | Dist MAE (d_ij) | Dist RMSE  | Dist MaxAE ");
            println!("---------------------------------------------------------------------------------------------------------------");
            println!(
                " Hutchinson (R={:<2})    | {:>7.3?} | {:.6}       | {:.6}     | {:.6}        | {:.6}   | {:.6}",
                num_vectors, duration_hutch, hutch_diag_mae, hutch_offdiag_mae, hutch_dist_mae, hutch_dist_rmse, hutch_dist_maxae
            );

            // 3. Low-Rank Spectral Approximation for varying ranks r
            let ranks = vec![2, 5, 10, n.saturating_sub(1)];

            for &r in &ranks {
                let mut rng_lr = rand::rngs::StdRng::seed_from_u64(42);
                let start_lr = Instant::now();
                let kernel_lr = LowRankDiffusionKernel::new(laplacian, t, r, &mut rng_lr);
                let duration_lr = start_lr.elapsed();

                let mut lr_diag_mae = 0.0;
                let mut lr_diag_sq = 0.0;
                let mut lr_offdiag_mae = 0.0;
                let mut lr_offdiag_sq = 0.0;
                let mut lr_dist_mae = 0.0;
                let mut lr_dist_sq = 0.0;
                let mut lr_dist_maxae = 0.0f64;

                for i in 0..n {
                    let err_diag = (kernel_lr.get(i, i) - exact_kernel.get(i, i)).abs();
                    lr_diag_mae += err_diag;
                    lr_diag_sq += err_diag * err_diag;

                    for j in (i + 1)..n {
                        let err_offdiag = (kernel_lr.get(i, j) - exact_kernel.get(i, j)).abs();
                        lr_offdiag_mae += err_offdiag;
                        lr_offdiag_sq += err_offdiag * err_offdiag;

                        let approx_d = kernel_lr.distance(i, j);
                        let err_d = (approx_d - exact_dist[i][j]).abs();
                        lr_dist_mae += err_d;
                        lr_dist_sq += err_d * err_d;
                        lr_dist_maxae = lr_dist_maxae.max(err_d);
                    }
                }

                lr_diag_mae /= n as f64;
                let _lr_diag_rmse = (lr_diag_sq / n as f64).sqrt();
                lr_offdiag_mae /= pair_count as f64;
                let _lr_offdiag_rmse = (lr_offdiag_sq / pair_count as f64).sqrt();
                lr_dist_mae /= pair_count as f64;
                let lr_dist_rmse = (lr_dist_sq / pair_count as f64).sqrt();

                println!(
                    " Low-Rank (r={:<2})     | {:>7.3?} | {:.6}       | {:.6}     | {:.6}        | {:.6}   | {:.6}",
                    r, duration_lr, lr_diag_mae, lr_offdiag_mae, lr_dist_mae, lr_dist_rmse, lr_dist_maxae
                );
            }
            println!("---------------------------------------------------------------------------------------------------------------");
        }
    }
    println!("==========================================================================================\n");
}
