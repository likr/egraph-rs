use crate::util;

use egraph_wasm::linalg_amg::JsAmgPcgSolver;
use egraph_wasm::linalg_chebyshev::{
    js_chebyshev_dog_jl_embedding, js_chebyshev_dog_jl_embedding_matrix, js_chebyshev_heat_spmv,
    js_chebyshev_heat_spmv_matrix,
};
use egraph_wasm::linalg_laplacian::js_laplacian;
use egraph_wasm::linalg_matrix::{js_anisotropic_laplacian, js_spectral_moment_stats};
use egraph_wasm::linalg_resistance::{
    js_compute_effective_resistances, js_resistance_curvature_flow,
};
use egraph_wasm::linalg_spectral::{
    js_koren_spectral_init, js_pivot_mds_init, js_select_maxmin_sp_pivots,
    js_smallest_laplacian_eigenpairs,
};
use util::fixtures::{layout_global_std, make_grid, make_hub_chain_graph, make_opts};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn linalg_pivots_mds_koren_eigen_and_moments() {
    let grid = make_grid(4, 4);

    // 1. selectMaxminSpPivots
    let p1 =
        js_select_maxmin_sp_pivots(&grid, &make_opts(&[("numPivots", JsValue::from_f64(6.0))]))
            .expect("pivots default");
    let p2 = js_select_maxmin_sp_pivots(
        &grid,
        &make_opts(&[
            ("numPivots", JsValue::from_f64(6.0)),
            ("seed", JsValue::from_f64(42.0)),
            ("minDegree", JsValue::from_f64(2.0)),
        ]),
    )
    .expect("pivots seeded");
    assert_eq!(p1, p2);
    assert_eq!(p1.len(), 6);
    assert_eq!(
        js_select_maxmin_sp_pivots(&grid, &make_opts(&[("numPivots", JsValue::from_f64(0.0))]))
            .unwrap()
            .len(),
        0
    );

    // 2. pivotMdsInit: default targetStd=1.0 vs explicit vs null (unnormalized)
    let mds_default =
        js_pivot_mds_init(&grid, &make_opts(&[("numPivots", JsValue::from_f64(8.0))]))
            .expect("mds default");
    assert_eq!(mds_default.len(), 16);
    assert!(mds_default.is_all_finite());
    assert!((layout_global_std(&mds_default) - 1.0).abs() < 1e-4);

    let mds_scaled = js_pivot_mds_init(
        &grid,
        &make_opts(&[
            ("numPivots", JsValue::from_f64(8.0)),
            ("seed", JsValue::from_f64(42.0)),
            ("minDegree", JsValue::from_f64(2.0)),
            ("targetStd", JsValue::from_f64(2.5)),
        ]),
    )
    .expect("mds 2.5");
    assert!((layout_global_std(&mds_scaled) - 2.5).abs() < 1e-4);

    let mds_unnorm = js_pivot_mds_init(
        &grid,
        &make_opts(&[
            ("numPivots", JsValue::from_f64(8.0)),
            ("seed", JsValue::from_f64(42.0)),
            ("minDegree", JsValue::from_f64(2.0)),
            ("targetStd", JsValue::NULL),
        ]),
    )
    .expect("mds unnormalized");
    assert!(mds_unnorm.is_all_finite());
    assert!((layout_global_std(&mds_unnorm) - 1.0).abs() > 0.05);

    // 3. korenSpectralInit: default targetStd=1.0 vs null
    let koren_default = js_koren_spectral_init(&grid, &JsValue::UNDEFINED).expect("koren");
    assert_eq!(koren_default.len(), 16);
    assert!(koren_default.is_all_finite());
    assert!((layout_global_std(&koren_default) - 1.0).abs() < 1e-4);

    let koren_unnorm = js_koren_spectral_init(
        &grid,
        &make_opts(&[
            ("targetStd", JsValue::NULL),
            ("tol", JsValue::from_f64(1e-5)),
            ("maxIterations", JsValue::from_f64(300.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .expect("koren unnorm");
    assert!(koren_unnorm.is_all_finite());

    // 4. smallestLaplacianEigenpairs: dense fallback (16 nodes) and Lanczos (60 nodes)
    let eig_small =
        js_smallest_laplacian_eigenpairs(&grid, &make_opts(&[("nModes", JsValue::from_f64(6.0))]))
            .expect("eig small");
    assert_eq!(eig_small.num_nodes(), 16);
    assert_eq!(eig_small.num_modes(), 6);
    assert!(eig_small.used_dense_fallback());
    let lams = eig_small.lambdas();
    let vecs = eig_small.vectors();
    let inv_d = eig_small.inv_sqrt_deg();
    assert_eq!(lams.len(), 6);
    assert_eq!(vecs.len(), 16 * 6);
    assert_eq!(inv_d.len(), 16);
    assert!(lams.windows(2).all(|w| w[0] <= w[1] + 1e-12));
    assert!(lams[0] > 0.0);

    let grid_large = make_grid(6, 10);
    let eig_large = js_smallest_laplacian_eigenpairs(
        &grid_large,
        &make_opts(&[("nModes", JsValue::from_f64(4.0))]),
    )
    .expect("eig lanczos");
    assert_eq!(eig_large.num_nodes(), 60);
    assert_eq!(eig_large.num_modes(), 4);
    assert!(!eig_large.used_dense_fallback());
    assert_eq!(eig_large.lambdas().len(), 4);
    assert_eq!(eig_large.vectors().len(), 60 * 4);

    // 5. spectralMomentStats
    let stats = js_spectral_moment_stats(&grid);
    assert!(stats.sigma_lambda_sq() > 0.0);
    assert!((stats.sigma_lambda() - stats.sigma_lambda_sq().sqrt()).abs() < 1e-12);
    assert!(stats.deg_cv() >= 0.0);
    assert!((stats.mean_degree() - 3.0).abs() < 1e-12);
    assert_eq!(stats.leaf_fraction(), 0.0);
}

#[wasm_bindgen_test]
fn chebyshev_heat_spmv_and_dog_jl_embedding() {
    let grid = make_grid(4, 4);
    let eig =
        js_smallest_laplacian_eigenpairs(&grid, &make_opts(&[("nModes", JsValue::from_f64(2.0))]))
            .unwrap();
    let inv_d = eig.inv_sqrt_deg();

    // Stationary vector v = D^{1/2} * 1 is in nullspace of L_sym => exp(-t L_sym) v == v
    let mut vecs = Vec::with_capacity(32);
    for &id in &inv_d {
        let deg_sqrt = if id > 0.0 { 1.0 / id } else { 0.0 };
        vecs.push(deg_sqrt);
        vecs.push(1.0);
    }

    let diffused =
        js_chebyshev_heat_spmv(&grid, &vecs, 2.0, 2.0, &JsValue::UNDEFINED).expect("heat spmv");
    assert_eq!(diffused.len(), 32);
    for i in 0..16 {
        let deg_sqrt = vecs[i * 2];
        assert!((diffused[i * 2] - deg_sqrt).abs() < 1e-10);
    }

    // t = 0.0 acts as identity
    let ident = js_chebyshev_heat_spmv(
        &grid,
        &vecs,
        2.0,
        0.0,
        &make_opts(&[
            ("degree", JsValue::from_f64(10.0)),
            ("lambdaMax", JsValue::from_f64(2.0)),
        ]),
    )
    .expect("heat t=0");
    for (&a, &b) in ident.iter().zip(vecs.iter()) {
        assert!((a - b).abs() < 1e-12);
    }

    // chebyshevDogJlEmbedding determinism and degreeNormalize difference
    let emb1 =
        js_chebyshev_dog_jl_embedding(&grid, &make_opts(&[("jlDim", JsValue::from_f64(12.0))]))
            .expect("jl default");
    let emb2 = js_chebyshev_dog_jl_embedding(
        &grid,
        &make_opts(&[
            ("jlDim", JsValue::from_f64(12.0)),
            ("degree", JsValue::from_f64(25.0)),
            ("seed", JsValue::from_f64(42.0)),
            ("degreeNormalize", JsValue::from_bool(true)),
            ("epsPsd", JsValue::from_f64(1e-4)),
        ]),
    )
    .expect("jl explicit");
    assert_eq!(emb1.len(), 16 * 12);
    assert_eq!(emb1, emb2);
    assert!(emb1.iter().all(|v| v.is_finite()));

    let emb_unnorm = js_chebyshev_dog_jl_embedding(
        &grid,
        &make_opts(&[
            ("jlDim", JsValue::from_f64(12.0)),
            ("degree", JsValue::from_f64(25.0)),
            ("seed", JsValue::from_f64(42.0)),
            ("degreeNormalize", JsValue::from_bool(false)),
            ("epsPsd", JsValue::from_f64(1e-4)),
            ("t1", JsValue::from_f64(0.5)),
            ("t2", JsValue::from_f64(2.0)),
            ("alpha", JsValue::from_f64(0.4)),
        ]),
    )
    .expect("jl unnorm");
    assert_ne!(emb1, emb_unnorm);
}

#[wasm_bindgen_test]
fn amg_pcg_solver_hardened_and_cooperative() {
    use egraph_wasm::dog_varadhan::js_compute_regularized_dog_varadhan_landmarks_from_matrix;

    let grid = make_grid(6, 6);
    let n = grid.num_nodes();
    let mut solver =
        JsAmgPcgSolver::from_graph_regularized_laplacian(&grid, Some(10.0), &JsValue::UNDEFINED)
            .expect("amg solver");
    assert_eq!(solver.num_nodes(), n);
    assert!(solver.num_levels() >= 1);
    assert!(solver.coarsest_size() <= 64);

    let b: Vec<f64> = (0..n).map(|i| ((i * 7 + 1) as f64).sin()).collect();
    let res = solver.solve(&b, None).expect("solve");
    assert!(res.converged());
    assert!(res.relative_residual() <= 1e-8);
    assert_eq!(res.x().len(), n);

    // Cooperative stepping equivalence
    let mut step_solver =
        JsAmgPcgSolver::from_graph_regularized_laplacian(&grid, Some(10.0), &JsValue::UNDEFINED)
            .expect("step solver");
    step_solver.start_solve(&b, None).expect("startSolve");
    if !step_solver.is_finished() {
        step_solver.step().expect("step");
        step_solver.run_steps(2.0).expect("runSteps");
        step_solver.run().expect("run");
    }
    assert!(step_solver.converged());
    assert_eq!(step_solver.solution(), res.x());

    // Batch solve
    let mut b_flat = Vec::with_capacity(n * 2);
    for &bi in &b {
        b_flat.push(bi);
        b_flat.push(2.0 * bi);
    }
    let batch = solver.solve_batch(&b_flat, 2.0, None).expect("solveBatch");
    assert!(batch.all_converged());
    assert_eq!(batch.converged_per_col(), vec![1u8, 1u8]);
    assert_eq!(batch.num_nodes(), n);
    assert_eq!(batch.ncols(), 2);
    assert_eq!(batch.x().len(), n * 2);

    // Anisotropic Laplacian + dynamic spectrum scaling + fromCsrMatrix
    let aniso = js_anisotropic_laplacian(
        &grid,
        &make_opts(&[
            ("alpha", JsValue::from_f64(0.0)),
            ("beta", JsValue::from_f64(0.0)),
            ("reweightedDegreeNorm", JsValue::from_bool(false)),
        ]),
    )
    .expect("anisotropicLaplacian");
    let l_comb = aniso.matrix();
    assert_eq!(l_comb.num_nodes(), n);
    assert!(l_comb.is_symmetric(None));
    assert!(l_comb.trace() > 0.0);
    assert!(l_comb.gershgorin_spectral_bound() > 2.0);
    let lmax_lanc = l_comb
        .estimate_lambda_max_lanczos(Some(16.0), Some(1.05), &JsValue::from_f64(42.0))
        .expect("lanczos");
    assert!(lmax_lanc > 2.0);
    let heat_mat = js_chebyshev_heat_spmv_matrix(
        &l_comb,
        &b,
        1.0,
        0.25,
        &make_opts(&[("seed", JsValue::from_f64(42.0))]),
    )
    .expect("heat matrix");
    assert_eq!(heat_mat.len(), n);
    assert!(heat_mat.iter().all(|v| v.is_finite()));
    let jl_mat = js_chebyshev_dog_jl_embedding_matrix(
        &l_comb,
        &make_opts(&[
            ("jlDim", JsValue::from_f64(8.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .expect("dog jl matrix");
    assert_eq!(jl_mat.len(), n * 8);
    let lm_mat = js_compute_regularized_dog_varadhan_landmarks_from_matrix(
        &grid,
        &l_comb,
        &make_opts(&[
            ("numLandmarks", JsValue::from_f64(8.0)),
            ("seed", JsValue::from_f64(42.0)),
            ("tCoarse", JsValue::from_f64(0.0)),
        ]),
    )
    .expect("dog landmarks from matrix");
    assert!(!lm_mat.used_coarse_scaffold());
    assert!(lm_mat.degree() >= 40);
    let reg_mat = l_comb.scale_and_shift(1.0, 1.0).expect("scaleAndShift");
    let mut mat_solver =
        JsAmgPcgSolver::from_csr_matrix(&reg_mat, &JsValue::UNDEFINED).expect("fromCsrMatrix");
    let mat_res = mat_solver.solve(&b, None).expect("solve mat");
    assert!(mat_res.converged());

    // Singular disconnected + isolated node + zero RHS + error handling
    let hub = make_hub_chain_graph();
    let mut hub_solver = JsAmgPcgSolver::from_graph_laplacian(
        &hub,
        &JsValue::from_str("combinatorial"),
        Some(0.0),
        &JsValue::UNDEFINED,
    )
    .expect("hub singular solver");
    let zero_res = hub_solver.solve(&[0.0; 13], None).expect("zero rhs");
    assert!(zero_res.converged());
    assert_eq!(zero_res.iterations(), 0);
    assert!(hub_solver.solve(&[1.0; 5], None).is_err());
    assert!(hub_solver.solve(&[f64::NAN; 13], None).is_err());
    assert!(
        JsAmgPcgSolver::from_graph_laplacian(
            &hub,
            &JsValue::from_str("random_walk"),
            Some(0.0),
            &JsValue::UNDEFINED,
        )
        .is_err()
    );
    assert!(
        JsAmgPcgSolver::from_graph_laplacian(
            &hub,
            &JsValue::from_str("rw"),
            Some(0.0),
            &JsValue::UNDEFINED,
        )
        .is_err()
    );
    let obj_kind_err = JsAmgPcgSolver::from_graph_laplacian(
        &hub,
        &make_opts(&[("kind", JsValue::from_str("combinatorial"))]),
        Some(0.0),
        &JsValue::UNDEFINED,
    )
    .unwrap_err();
    assert_eq!(
        js_sys::Reflect::get(&obj_kind_err, &"code".into())
            .unwrap()
            .as_string(),
        Some("InvalidInput".to_owned())
    );
    assert!(
        js_laplacian(
            &hub,
            &make_opts(&[("kind", JsValue::from_str("combinatorial"))])
        )
        .is_ok()
    );
    assert!(js_laplacian(&hub, &JsValue::from_f64(1.0)).is_err());
}

#[wasm_bindgen_test]
fn effective_resistance_and_curvature_flow_wasm() {
    let grid10 = make_grid(10, 10);
    let er_opts = make_opts(&[("exact", JsValue::from_bool(false))]);
    let er = js_compute_effective_resistances(&grid10, &er_opts).expect("er jl");
    assert!(!er.used_exact());
    assert!(er.amg_all_converged());
    assert_eq!(er.num_nodes(), 100);
    assert_eq!(er.num_edges(), 180);
    assert_eq!(er.num_components(), 1);
    assert!((er.expected_foster() - 99.0).abs() < 1e-12);
    assert!(er.foster_rel_err() < 0.05);
    assert_eq!(er.edge_u().len(), 180);
    assert_eq!(er.edge_v().len(), 180);
    assert_eq!(er.effective_resistances().len(), 180);

    let hub = make_hub_chain_graph();
    let flow = js_resistance_curvature_flow(&hub, &JsValue::UNDEFINED).expect("rcf hub");
    assert!(flow.used_exact());
    assert_eq!(flow.num_components(), 2);
    assert!((flow.expected_foster() - 11.0).abs() < 1e-12);
    assert!(flow.foster_rel_err() < 1e-11);
    assert_eq!(flow.suggested_geometry(), "hyperbolic");
    assert_eq!(
        flow.suggested_manifold().as_deref(),
        Some("hyperbolicPoincare")
    );
    assert_eq!(flow.step_history().len(), 3);
    let lg = flow.length_graph().expect("lengthGraph");
    let cg = flow.conductance_graph().expect("conductanceGraph");
    assert!(lg.is_weighted());
    assert!(cg.is_weighted());
    assert_eq!(lg.num_nodes(), 13);
}
