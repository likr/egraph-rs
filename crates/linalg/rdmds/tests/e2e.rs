use egraph_dataset::dataset_1138_bus;
use ndarray::{Array1, Array2};
use petgraph::Graph;
use petgraph::Undirected;
use petgraph_linalg_rdmds::RdMds;
use petgraph_linalg_rdmds::solvers::{AmgCgSolver, CgSolver, Ic0CgSolver, JacobiCgSolver};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

// Computes the expected eigenvalues and eigenvectors using nalgebra.
// Returns (eigenvalues, eigenvectors) where eigenvalues are sorted in ascending order.
// d is the number of smallest eigenvalues to return (excluding the 0 eigenvalue).
fn compute_expected(
    graph: &Graph<(), (), Undirected>,
    d: usize,
    laplacian_type: &str,
) -> (Array1<f64>, Array2<f64>) {
    let n = graph.node_count();
    let mut a = nalgebra::DMatrix::<f64>::zeros(n, n);
    let mut d_mat = nalgebra::DMatrix::<f64>::zeros(n, n);

    for edge in graph.edge_indices() {
        let (u, v) = graph.edge_endpoints(edge).unwrap();
        let u_idx = u.index();
        let v_idx = v.index();
        a[(u_idx, v_idx)] = 1.0;
        a[(v_idx, u_idx)] = 1.0;
    }

    for i in 0..n {
        let mut degree = 0.0;
        for j in 0..n {
            degree += a[(i, j)];
        }
        d_mat[(i, i)] = degree;
    }

    let l_std = &d_mat - &a;

    let mut d_inv_sqrt = nalgebra::DMatrix::<f64>::zeros(n, n);
    let mut d_inv = nalgebra::DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        if d_mat[(i, i)] > 0.0 {
            d_inv_sqrt[(i, i)] = 1.0 / d_mat[(i, i)].sqrt();
            d_inv[(i, i)] = 1.0 / d_mat[(i, i)];
        }
    }

    let l_sym = &d_inv_sqrt * &l_std * &d_inv_sqrt;

    let (evals, evecs) = match laplacian_type {
        "standard" => {
            let eig = nalgebra::SymmetricEigen::new(l_std);
            (eig.eigenvalues, eig.eigenvectors)
        }
        "symmetric_normalized" => {
            let eig = nalgebra::SymmetricEigen::new(l_sym);
            (eig.eigenvalues, eig.eigenvectors)
        }
        "random_walk_normalized" => {
            let eig = nalgebra::SymmetricEigen::new(l_sym);
            // v = D^{-1/2} u
            let mut vecs = eig.eigenvectors.clone();
            for j in 0..n {
                for i in 0..n {
                    vecs[(i, j)] = d_inv_sqrt[(i, i)] * eig.eigenvectors[(i, j)];
                }
            }
            (eig.eigenvalues, vecs)
        }
        _ => panic!("Unknown laplacian type"),
    };

    // Sort eigenvalues and corresponding eigenvectors
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| evals[i].partial_cmp(&evals[j]).unwrap());

    // We want `d` eigenvalues after the first one (which should be 0)
    let start_idx = 1;

    let mut out_evals = Array1::zeros(d);
    let mut out_evecs = Array2::zeros((n, d));

    for (k, i) in indices.iter().skip(start_idx).take(d).enumerate() {
        out_evals[k] = evals[*i];
        for j in 0..n {
            out_evecs[(j, k)] = evecs[(j, *i)];
        }
    }

    // Normalize eigenvectors sign: make the first non-zero element positive
    for k in 0..d {
        for j in 0..n {
            if out_evecs[(j, k)].abs() > 1e-5 {
                if out_evecs[(j, k)] < 0.0 {
                    for i in 0..n {
                        out_evecs[(i, k)] = -out_evecs[(i, k)];
                    }
                }
                break;
            }
        }
    }

    (out_evals, out_evecs)
}

fn assert_eigendecomposition_match(
    expected_evals: &Array1<f64>,
    expected_evecs: &Array2<f64>,
    actual_evals: &Array1<f64>,
    actual_evecs: &Array2<f64>,
) {
    let d = expected_evals.len();
    let n = expected_evecs.shape()[0];

    for i in 0..d {
        // Compare eigenvalue
        assert!(
            (expected_evals[i] - actual_evals[i]).abs() < 1e-2,
            "Eigenvalue mismatch at dim {}: expected {}, got {}",
            i,
            expected_evals[i],
            actual_evals[i]
        );

        // Normalize actual eigenvector sign
        let mut actual_vec = actual_evecs.column(i).to_owned();
        for j in 0..n {
            if actual_vec[j].abs() > 1e-5 {
                if actual_vec[j] < 0.0 {
                    actual_vec *= -1.0;
                }
                break;
            }
        }

        // Compare eigenvector (using a looser tolerance since iterative methods might not be perfectly exact)
        for j in 0..n {
            assert!(
                (expected_evecs[(j, i)] - actual_vec[j]).abs() < 5e-2,
                "Eigenvector mismatch at node {}, dim {}: expected {}, got {}",
                j,
                i,
                expected_evecs[(j, i)],
                actual_vec[j]
            );
        }
    }
}

macro_rules! generate_solver_tests {
    ($solver_name:ident, $solver_expr:expr) => {
        mod $solver_name {
            use super::*;

            #[test]
            fn test_standard() {
                let graph = dataset_1138_bus();
                let d = 5;
                let (expected_evals, expected_evecs) = compute_expected(&graph, d, "standard");

                let mut rng = ChaCha8Rng::seed_from_u64(0);
                let solver = $solver_expr;
                let mut rdmds = RdMds::new();
                rdmds
                    .d(d)
                    .eigenvalue_tolerance(1e-6)
                    .eigenvalue_max_iterations(500);

                let result = rdmds.eigendecomposition(&graph, |_| 1.0, &solver, &mut rng);
                println!("cg_iterations: {:?}", result.cg_iterations);
                println!("power_iterations: {:?}", result.power_iterations);
                assert_eigendecomposition_match(
                    &expected_evals,
                    &expected_evecs,
                    &result.eigenvalues,
                    &result.eigenvectors,
                );
            }

            #[test]
            fn test_symmetric_normalized() {
                let graph = dataset_1138_bus();
                let d = 5;
                let (expected_evals, expected_evecs) =
                    compute_expected(&graph, d, "symmetric_normalized");

                let mut rng = ChaCha8Rng::seed_from_u64(0);
                let solver = $solver_expr;
                let mut rdmds = RdMds::new();
                rdmds
                    .d(d)
                    .eigenvalue_tolerance(1e-6)
                    .eigenvalue_max_iterations(500);

                let result = rdmds.eigendecomposition_symmetric_normalized(
                    &graph,
                    |_| 1.0,
                    &solver,
                    &mut rng,
                );
                println!("cg_iterations: {:?}", result.cg_iterations);
                println!("power_iterations: {:?}", result.power_iterations);
                assert_eigendecomposition_match(
                    &expected_evals,
                    &expected_evecs,
                    &result.eigenvalues,
                    &result.eigenvectors,
                );
            }

            #[test]
            fn test_random_walk_normalized() {
                let graph = dataset_1138_bus();
                let d = 5;
                let (expected_evals, expected_evecs) =
                    compute_expected(&graph, d, "random_walk_normalized");

                let mut rng = ChaCha8Rng::seed_from_u64(0);
                let solver = $solver_expr;
                let mut rdmds = RdMds::new();
                rdmds
                    .d(d)
                    .eigenvalue_tolerance(1e-6)
                    .eigenvalue_max_iterations(500);

                let result = rdmds.eigendecomposition_random_walk_normalized(
                    &graph,
                    |_| 1.0,
                    &solver,
                    &mut rng,
                );
                println!("cg_iterations: {:?}", result.cg_iterations);
                println!("power_iterations: {:?}", result.power_iterations);
                assert_eigendecomposition_match(
                    &expected_evals,
                    &expected_evecs,
                    &result.eigenvalues,
                    &result.eigenvectors,
                );
            }
        }
    };
}

generate_solver_tests!(
    cg_solver,
    CgSolver {
        max_iterations: 100,
        tolerance: 1e-6
    }
);
generate_solver_tests!(
    jacobi_cg_solver,
    JacobiCgSolver {
        max_iterations: 100,
        tolerance: 1e-6
    }
);
generate_solver_tests!(
    ic0_cg_solver,
    Ic0CgSolver {
        max_iterations: 100,
        tolerance: 1e-6
    }
);
generate_solver_tests!(
    amg_cg_solver,
    AmgCgSolver {
        max_iterations: 100,
        tolerance: 1e-6
    }
);
