use egraph_wasm::v6::{
    JsCsrGraph, JsGuMapSolver, JsKamadaKawaiSolver, JsKnnGraph, JsLayout2d, JsManifoldLayout2d,
    JsSgdSolver, JsSpatialIntruderRefiner, JsStressMajorizationSolver, JsTsNetSolver,
    js_build_pure_2hop_knn, js_chebyshev_dog_jl_embedding, js_coarsen_graph,
    js_compute_fuzzy_simplicial_set, js_compute_knn_probabilities,
    js_compute_nodewise_2hop_perplexities, js_compute_regularized_dog_varadhan_landmarks,
    js_graph_row_normalized_probabilities, js_kmeans_plus_plus, js_label_propagation,
    js_louvain_communities, js_pivot_mds_init, js_project_clustered_layout, js_spectral_clustering,
};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

fn make_grid(rows: u32, cols: u32) -> JsCsrGraph {
    let mut edges = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let u = r * cols + c;
            if c + 1 < cols {
                edges.push(u);
                edges.push(u + 1);
            }
            if r + 1 < rows {
                edges.push(u);
                edges.push(u + cols);
            }
        }
    }
    JsCsrGraph::from_edge_list(
        (rows * cols) as f64,
        &edges,
        None,
        &JsValue::UNDEFINED,
        None,
    )
    .expect("grid graph")
}

fn make_barbell(clique_size: u32) -> JsCsrGraph {
    let mut edges = Vec::new();
    for off in [0u32, clique_size] {
        for i in 0..clique_size {
            for j in (i + 1)..clique_size {
                edges.push(off + i);
                edges.push(off + j);
            }
        }
    }
    edges.push(clique_size - 1);
    edges.push(clique_size);
    JsCsrGraph::from_edge_list(
        (clique_size * 2) as f64,
        &edges,
        None,
        &JsValue::UNDEFINED,
        None,
    )
    .expect("barbell graph")
}

fn make_circle_layout(n: usize, radius: f32) -> JsLayout2d {
    let mut coords = Vec::with_capacity(n * 2);
    for i in 0..n {
        let ang = std::f32::consts::TAU * (i as f32) / (n.max(1) as f32);
        let (s, c) = ang.sin_cos();
        coords.push(radius * c);
        coords.push(radius * s);
    }
    JsLayout2d::from_coords(&coords).expect("circle layout")
}

fn assert_layouts_bit_exact(a: &JsLayout2d, b: &JsLayout2d) {
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        let idx = i as f64;
        assert_eq!(a.x(idx).unwrap().to_bits(), b.x(idx).unwrap().to_bits());
        assert_eq!(a.y(idx).unwrap().to_bits(), b.y(idx).unwrap().to_bits());
    }
}

fn make_opts(entries: &[(&str, JsValue)]) -> js_sys::Object {
    let obj = js_sys::Object::new();
    for (k, v) in entries {
        js_sys::Reflect::set(&obj, &JsValue::from_str(k), v).unwrap();
    }
    obj
}

#[wasm_bindgen_test]
fn clustering_kmeans_spectral_louvain_lpa_coarsen_and_project() {
    // 1. kmeansPlusPlus
    let pts = [
        10.0, 10.0, 10.2, 9.8, 9.8, 10.1, // cluster 0
        -10.0, -10.0, -9.9, -10.2, -10.1, -9.8, // cluster 1
    ];
    let km = js_kmeans_plus_plus(
        &pts,
        6.0,
        2.0,
        Some(2.0),
        Some(100.0),
        Some(1e-4),
        Some(3.0),
        Some(42.0),
    )
    .expect("kmeans");
    assert_eq!(km.labels(), vec![0, 0, 0, 1, 1, 1]);
    assert!(km.converged());
    assert!(km.inertia() < 1.0);
    assert!(km.iterations() >= 1);
    assert_eq!(km.k(), 2);
    assert_eq!(km.dims(), 2);
    assert_eq!(km.centroids().len(), 4);

    // 2. spectralClustering on barbell
    let barbell = make_barbell(5);
    let sp = js_spectral_clustering(&barbell, Some(2.0), None, None, None, None, Some(42.0))
        .expect("spectral");
    assert_eq!(sp, vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);

    // 3. louvainCommunities
    let lv = js_louvain_communities(&barbell, None, None, None, Some(42.0)).expect("louvain");
    assert_eq!(lv.num_communities(), 2);
    assert_eq!(lv.labels(), vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);
    assert!(lv.modularity() > 0.3);
    assert!(lv.passes() >= 1);

    // 4. labelPropagation
    let lpa = js_label_propagation(&barbell, Some(100.0), Some(42.0)).expect("lpa");
    assert_eq!(lpa, vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);

    // 5. coarsenGraph & projectClusteredLayout
    let coarse = js_coarsen_graph(
        &barbell,
        &[100, 100, 100, 100, 100, 999, 999, 999, 999, 999],
    )
    .expect("coarsen");
    assert_eq!(coarse.num_clusters(), 2);
    assert_eq!(coarse.node_to_coarse(), vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);
    assert_eq!(coarse.coarse_sizes(), vec![5, 5]);
    let cg = coarse.graph();
    assert_eq!(cg.num_nodes(), 2);
    assert_eq!(cg.num_edges(), 1);
    assert_eq!(cg.weights(), Some(vec![1.0, 1.0]));

    let coarse_layout = JsLayout2d::from_coords(&[-2.0, 0.0, 2.0, 0.0]).unwrap();
    let fine1 = js_project_clustered_layout(
        &coarse_layout,
        &coarse.node_to_coarse(),
        Some(0.2),
        Some(42.0),
    )
    .expect("project 1");
    let fine2 = js_project_clustered_layout(
        &coarse_layout,
        &coarse.node_to_coarse(),
        Some(0.2),
        Some(42.0),
    )
    .expect("project 2");
    assert_eq!(fine1.len(), 10);
    assert!(fine1.is_all_finite());
    assert_layouts_bit_exact(&fine1, &fine2);

    // Directed graph must be rejected by clustering & coarsening
    let directed =
        JsCsrGraph::from_edge_list(3.0, &[0, 1, 1, 2], None, &JsValue::from_bool(true), None)
            .unwrap();
    assert!(js_spectral_clustering(&directed, Some(2.0), None, None, None, None, None).is_err());
    assert!(js_louvain_communities(&directed, None, None, None, None).is_err());
    assert!(js_label_propagation(&directed, None, None).is_err());
    assert!(js_coarsen_graph(&directed, &[0, 1, 1]).is_err());

    // Adversarial clustering errors
    assert!(js_kmeans_plus_plus(&[0.0, 1.0], 1.0, 2.0, Some(2.0), None, None, None, None).is_err());
    assert!(
        js_kmeans_plus_plus(
            &[0.0, f64::NAN],
            1.0,
            2.0,
            Some(1.0),
            None,
            None,
            None,
            None
        )
        .is_err()
    );
    assert!(js_coarsen_graph(&barbell, &[0, 1]).is_err());
    assert!(js_project_clustered_layout(&coarse_layout, &[0, 2], Some(0.1), None).is_err());
    assert!(js_project_clustered_layout(&coarse_layout, &[0, 1], Some(-0.1), None).is_err());
}

#[wasm_bindgen_test]
fn manifold_layout_and_sgd_manifold_all_three_geometries() {
    // 1. ManifoldLayout2d constructors, aliases, projections, geodesic distances, mutators
    let hyp_z = JsManifoldLayout2d::zeros("POINCARE", 3.0).expect("zeros poincare");
    assert_eq!(hyp_z.kind(), "hyperbolic");
    assert_eq!(hyp_z.len(), 3);
    assert!(!hyp_z.is_empty());

    let mut hyp = JsManifoldLayout2d::from_coords("hyperbolic_poincare", &[3.0, 4.0, 0.0, 0.0])
        .expect("fromCoords hyp");
    assert_eq!(hyp.kind(), "hyperbolic");
    let r0 = hyp.x(0.0).unwrap().hypot(hyp.y(0.0).unwrap());
    assert!(r0 < 1.0 && r0 > 0.99);
    assert!(hyp.geodesic_distance(0.0, 1.0).unwrap() > 1.0);
    hyp.set_point(1.0, 0.25, -0.25).unwrap();
    assert!((hyp.x(1.0).unwrap() - 0.25).abs() < 1e-6);
    let euc = hyp.to_layout();
    assert_eq!(euc.len(), 2);
    let hyp_from_l = JsManifoldLayout2d::from_layout("hyperbolic", &euc).unwrap();
    assert_eq!(hyp_from_l.coords(), hyp.coords());

    let sph = JsManifoldLayout2d::from_coords("sphere", &[-0.5, -0.25, 1.0, 1.5]).unwrap();
    assert_eq!(sph.kind(), "spherical");
    assert!((0.0..std::f32::consts::TAU).contains(&sph.x(0.0).unwrap()));
    assert!((0.0..=std::f32::consts::PI).contains(&sph.y(0.0).unwrap()));

    let tor = JsManifoldLayout2d::from_coords("torus2d", &[1.2, -0.1, 0.9, 0.2]).unwrap();
    let mut tor_mut = tor.clone();
    assert_eq!(tor_mut.kind(), "torus");
    assert!((tor_mut.x(0.0).unwrap() - 0.2).abs() < 1e-6);
    assert!((tor_mut.y(0.0).unwrap() - 0.9).abs() < 1e-6);
    tor_mut.project_all().unwrap();
    assert!(tor_mut.is_all_finite());

    // Poincare Mobius centroid centering & safe initialization
    let m_rand =
        JsManifoldLayout2d::random_init("hyperbolic", 8.0, Some(0.25), Some(42.0)).unwrap();
    assert_eq!(m_rand.len(), 8);
    assert!(m_rand.is_all_finite());
    let big_euc =
        egraph_wasm::v6::JsLayout2d::from_coords(&[10.0, -5.0, 20.0, 5.0, 30.0, 15.0]).unwrap();
    let mut m_scaled =
        JsManifoldLayout2d::from_layout_scaled("hyperbolic", &big_euc, Some(0.7)).unwrap();
    let d_before = m_scaled.geodesic_distance(0.0, 1.0).unwrap();
    m_scaled.center_poincare_centroid().unwrap();
    let c = m_scaled.poincare_centroid().unwrap();
    assert!(c[0].hypot(c[1]) < 1e-4);
    assert!((m_scaled.geodesic_distance(0.0, 1.0).unwrap() - d_before).abs() < 1e-3);
    m_scaled.recenter_poincare_at(2.0).unwrap();
    assert!(m_scaled.x(2.0).unwrap().hypot(m_scaled.y(2.0).unwrap()) < 1e-4);
    let m_add = egraph_wasm::v6::js_poincare_mobius_add(-0.3, 0.4, 0.3, -0.4);
    assert!(m_add[0].hypot(m_add[1]) < 1e-12);
    assert!(tor_mut.poincare_centroid().is_err());
    assert!(tor_mut.recenter_poincare(0.1, 0.1).is_err());
    assert!(tor_mut.recenter_poincare_at(0.0).is_err());
    assert!(tor_mut.center_poincare_centroid().is_err());

    // Invalid manifold inputs
    assert!(JsManifoldLayout2d::zeros("klein_bottle", 2.0).is_err());
    assert!(JsManifoldLayout2d::from_coords("torus", &[0.0, 1.0, 2.0]).is_err());
    assert!(JsManifoldLayout2d::from_coords("spherical", &[f32::NAN, 0.0]).is_err());
    assert!(tor_mut.geodesic_distance(0.0, 9.0).is_err());
    assert!(tor_mut.set_point(0.0, f32::INFINITY, 0.0).is_err());
    assert!(tor_mut.x(5.0).is_err());

    // 2. SgdSolver stepManifold == runManifold across all 3 manifolds
    let cyc = JsCsrGraph::from_edge_list(
        6.0,
        &[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
        None,
        &JsValue::UNDEFINED,
        None,
    )
    .unwrap();
    let sched_opts = make_opts(&[
        ("tMax", JsValue::from_f64(20.0)),
        ("eps", JsValue::from_f64(0.02)),
        ("kind", JsValue::from_str("exponential")),
        ("seed", JsValue::from_f64(19.0)),
    ]);

    for (kind, init_coords) in [
        (
            "hyperbolic",
            vec![
                0.1f32, 0.0, 0.0, 0.1, -0.1, 0.0, 0.0, -0.1, 0.15, 0.15, -0.15, -0.15,
            ],
        ),
        (
            "spherical",
            vec![
                0.1f32, 1.2, 0.2, 1.3, 0.3, 1.1, 0.4, 1.4, 0.5, 1.25, 0.6, 1.35,
            ],
        ),
        (
            "torus",
            vec![
                0.05f32, 0.92, 0.85, 0.10, 0.45, 0.55, 0.12, 0.48, 0.3, 0.7, 0.6, 0.2,
            ],
        ),
    ] {
        let mut s_run = JsSgdSolver::full(&cyc, &sched_opts, None, None, None, None).unwrap();
        let mut s_step = JsSgdSolver::full(&cyc, &sched_opts, None, None, None, None).unwrap();
        let mut m_run = JsManifoldLayout2d::from_coords(kind, &init_coords).unwrap();
        let mut m_step = JsManifoldLayout2d::from_coords(kind, &init_coords).unwrap();

        assert_eq!(s_run.run_manifold(&mut m_run).unwrap(), "completed");
        assert_eq!(
            s_step.run_steps_manifold(&mut m_step, 5.0).unwrap(),
            "running"
        );
        while !s_step.is_finished() {
            s_step.step_manifold(&mut m_step).unwrap();
        }
        assert_eq!(s_step.status(), "completed");
        assert_eq!(m_run.coords(), m_step.coords());
    }
}

#[wasm_bindgen_test]
fn tsnet_solver_all_backends_constructors_and_step_eq_run() {
    let grid = make_grid(4, 4);
    let init = make_circle_layout(16, 0.5);

    for backend in ["exact", "BARNES-HUT", "fft_interpolation", "L"] {
        let opts = make_opts(&[
            ("backend", JsValue::from_str(backend)),
            ("iterationsStage1", JsValue::from_f64(4.0)),
            ("iterationsStage2", JsValue::from_f64(4.0)),
            ("iterationsStage3", JsValue::from_f64(4.0)),
            ("lambdaRStage3", JsValue::from_f64(0.5)),
            ("learningRate", JsValue::from_f64(25.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]);

        let mut s_run = JsTsNetSolver::pure_2hop(
            &grid,
            &opts,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &JsValue::UNDEFINED,
            None,
            None,
        )
        .expect("tsnet pure2Hop");
        let mut s_step = JsTsNetSolver::pure_2hop(
            &grid,
            &opts,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &JsValue::UNDEFINED,
            None,
            None,
        )
        .expect("tsnet pure2Hop step");

        let mut l_run = init.clone();
        let mut l_step = init.clone();
        let ptr_before = l_step.coords_ptr();

        assert_eq!(s_run.status(), "initialized");
        assert_eq!(s_run.total_iterations(), 12);
        assert_eq!(s_run.run(&mut l_run).unwrap(), "completed");
        assert!(s_run.is_finished());

        assert_eq!(s_step.run_steps(&mut l_step, 5.0).unwrap(), "running");
        assert_eq!(s_step.iteration(), 5);
        assert_eq!(s_step.stage_index(), 1);
        while !s_step.is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        assert_eq!(l_step.coords_ptr(), ptr_before);
        assert_eq!(s_step.status(), "completed");
        assert_eq!(s_step.total_fft_2d_calls(), s_run.total_fft_2d_calls());
        assert_layouts_bit_exact(&l_run, &l_step);
        if matches!(backend, "fft_interpolation" | "L") {
            assert!(s_run.total_fft_2d_calls() > 0.0);
        } else {
            assert_eq!(s_run.total_fft_2d_calls(), 0.0);
        }
    }

    // fromKnn and fromProbabilities
    let pure = js_build_pure_2hop_knn(&grid).unwrap();
    let knn = pure.knn();
    let perps = js_compute_nodewise_2hop_perplexities(&grid, None, None, None).unwrap();
    let probs = js_compute_knn_probabilities(&knn, &perps, None, None, None, None).unwrap();

    let s_knn = JsTsNetSolver::from_knn(
        &knn,
        Some(perps),
        None,
        &JsValue::from_str("fit"),
        Some(2.0),
        Some(2.0),
        Some(2.0),
        Some(20.0),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(42.0),
        None,
        &JsValue::UNDEFINED,
        None,
        None,
    )
    .unwrap();
    let s_probs = JsTsNetSolver::from_probabilities(
        &probs,
        &JsValue::from_str("fft"),
        Some(2.0),
        Some(2.0),
        Some(2.0),
        Some(20.0),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(42.0),
        None,
        &JsValue::UNDEFINED,
        None,
        None,
    )
    .unwrap();
    assert_eq!(s_knn.num_nodes(), 16);
    assert_eq!(s_probs.total_iterations(), 6);

    // Optional learningRate auto-scaling & Stage 3 LR cap (2 * N for N < 100) + fromGraphRowNormalized
    let rn_opts = make_opts(&[
        ("w2", JsValue::from_f64(0.5)),
        ("max2hopPerNode", JsValue::from_f64(8.0)),
        ("degreeBeta", JsValue::from_f64(0.35)),
        ("backend", JsValue::from_str("fft")),
        ("iterationsStage1", JsValue::from_f64(2.0)),
        ("iterationsStage2", JsValue::from_f64(2.0)),
        ("iterationsStage3", JsValue::from_f64(2.0)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut rn_run = JsTsNetSolver::from_graph_row_normalized(
        &grid,
        &rn_opts,
        None,
        &JsValue::UNDEFINED,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        &JsValue::UNDEFINED,
        None,
        None,
        None,
    )
    .expect("fromGraphRowNormalized opts");
    let mut rn_step = JsTsNetSolver::from_graph_row_normalized(
        &grid,
        &JsValue::from_f64(0.5),
        Some(8.0),
        &JsValue::from_str("fft"),
        Some(2.0),
        Some(2.0),
        Some(2.0),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(42.0),
        None,
        &JsValue::UNDEFINED,
        None,
        None,
        Some(0.35),
    )
    .expect("fromGraphRowNormalized pos");
    let mut lrn_run = init.clone();
    let mut lrn_step = init.clone();
    assert_eq!(rn_run.run(&mut lrn_run).unwrap(), "completed");
    while !rn_step.is_finished() {
        rn_step.step(&mut lrn_step).unwrap();
    }
    assert_layouts_bit_exact(&lrn_run, &lrn_step);
    assert!(rn_run.current_stage_lr() <= 32.0);

    // w2 = 0.0 parity with graphRowNormalizedProbabilities + fromProbabilities
    let probs_1hop = js_graph_row_normalized_probabilities(&grid).unwrap();
    let mut s_rn0 = JsTsNetSolver::from_graph_row_normalized(
        &grid,
        &JsValue::from_f64(0.0),
        None,
        &JsValue::from_str("fft"),
        Some(2.0),
        Some(2.0),
        Some(2.0),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(42.0),
        None,
        &JsValue::UNDEFINED,
        None,
        None,
        None,
    )
    .unwrap();
    let mut s_p0 = JsTsNetSolver::from_probabilities(
        &probs_1hop,
        &JsValue::from_str("fft"),
        Some(2.0),
        Some(2.0),
        Some(2.0),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(42.0),
        None,
        &JsValue::UNDEFINED,
        None,
        None,
    )
    .unwrap();
    let mut lp0 = init.clone();
    let mut lr0 = init;
    s_rn0.run(&mut lr0).unwrap();
    s_p0.run(&mut lp0).unwrap();
    assert_layouts_bit_exact(&lr0, &lp0);
    assert!(
        JsTsNetSolver::from_graph_row_normalized(
            &grid,
            &JsValue::from_f64(-0.1),
            None,
            &JsValue::UNDEFINED,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &JsValue::UNDEFINED,
            None,
            None,
            None,
        )
        .is_err()
    );

    // Invalid backend or extra positional arg with options rejected
    assert!(
        JsTsNetSolver::pure_2hop(
            &grid,
            &JsValue::from_str("bogus"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &JsValue::UNDEFINED,
            None,
            None,
        )
        .is_err()
    );
    let opts = make_opts(&[("backend", JsValue::from_str("exact"))]);
    assert!(
        JsTsNetSolver::pure_2hop(
            &grid,
            &opts,
            Some(10.0),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &JsValue::UNDEFINED,
            None,
            None,
        )
        .is_err()
    );
}

#[wasm_bindgen_test]
fn sgd_solver_all_constructors_and_step_eq_run() {
    let grid = make_grid(4, 4);
    let init = js_pivot_mds_init(&grid, None, None, None, &JsValue::UNDEFINED).unwrap();

    // 1. sparseBarrierDog + stats() + step() == run() + in-place coordsPtr stability
    let dog_opts = make_opts(&[
        ("tMax", JsValue::from_f64(25.0)),
        ("hingeRepulsion", JsValue::from_bool(true)),
        ("barrierSym", JsValue::from_str("sym_avg")),
        ("kind", JsValue::from_str("exponential")),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut s_run = JsSgdSolver::sparse_barrier_dog(
        &grid, &dog_opts, None, None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
        None, None,
    )
    .expect("sparseBarrierDog");
    let mut s_step = JsSgdSolver::sparse_barrier_dog(
        &grid, &dog_opts, None, None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
        None, None,
    )
    .expect("sparseBarrierDog step");
    let st = s_run.stats().expect("barrier stats");
    assert!(st.n_pairs_1hop() > 0);
    assert!(st.n_pairs_2hop() > 0);
    assert!(st.n_pairs_hinge() > 0);
    assert_eq!(st.perplexity_source(), "adaptive_2hop");
    assert!(s_run.num_springs() > 0);
    assert!(s_run.num_hinges() > 0);
    assert!(s_run.current_eta() > 0.0);

    let mut l_run = init.clone();
    let mut l_step = init.clone();
    let ptr_before = l_step.coords_ptr();
    assert_eq!(s_run.run(&mut l_run).unwrap(), "completed");
    assert_eq!(s_run.current_eta(), 0.0);
    assert_eq!(s_step.run_steps(&mut l_step, 7.0).unwrap(), "running");
    while !s_step.is_finished() {
        s_step.step(&mut l_step).unwrap();
    }
    assert_eq!(l_step.coords_ptr(), ptr_before);
    assert_layouts_bit_exact(&l_run, &l_step);

    // 2. sparseBarrierFromLandmarks
    let lm = js_compute_regularized_dog_varadhan_landmarks(
        &grid, None, None, None, None, None, None, None, None, None, None, None,
    )
    .unwrap();
    let s_lm = JsSgdSolver::sparse_barrier_from_landmarks(
        &grid,
        &lm,
        Some(vec![0, 5, 10, 15]),
        &JsValue::from_f64(16.0),
        None,
        None,
        Some("symMax".to_owned()),
        None,
        None,
        None,
        None,
        None,
        None,
        Some(4.0),
        None,
        None,
        Some(10.0),
        None,
        Some("linear".to_owned()),
        None,
        Some(42.0),
    )
    .unwrap();
    assert!(s_lm.stats().is_some());
    assert_eq!(s_lm.stats().unwrap().perplexity_source(), "explicit");

    // 3. randomPair, omega, streamingNegatives, fromPairs, fromPairsWithHinges
    let s_rp = JsSgdSolver::random_pair(
        &grid,
        &JsValue::from_f64(6.0),
        &JsValue::from_f64(10.0),
        None,
        Some("quadratic".to_owned()),
        None,
        Some(42.0),
    )
    .unwrap();
    assert!(s_rp.stats().is_none());

    let emb =
        js_chebyshev_dog_jl_embedding(&grid, Some(8.0), None, None, None, None, None, None, None)
            .unwrap();
    let mut s_om = JsSgdSolver::omega(
        &grid,
        &emb,
        8.0,
        &JsValue::from_f64(5.0),
        None,
        &JsValue::from_f64(10.0),
        None,
        Some("reciprocal".to_owned()),
        None,
        Some(42.0),
    )
    .unwrap();
    let mut l_om = init.clone();
    assert_eq!(s_om.run(&mut l_om).unwrap(), "completed");

    let mut s_sn = JsSgdSolver::streaming_negatives(
        &grid,
        &JsValue::from_f64(4.0),
        Some(2.5),
        Some(0.25),
        Some(true),
        Some(10.0),
        None,
        Some("constant".to_owned()),
        Some(0.5),
        Some(42.0),
    )
    .unwrap();
    let mut l_sn = init.clone();
    assert_eq!(s_sn.run(&mut l_sn).unwrap(), "completed");

    let mut s_pairs = JsSgdSolver::from_pairs(
        2.0,
        &[0],
        &[1],
        &[1.0],
        Some(vec![1.0]),
        Some(vec![1.0]),
        Some(vec![1.0]),
        &JsValue::from_f64(5.0),
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let mut l2 = JsLayout2d::from_coords(&[0.0, 0.0, 3.0, 0.0]).unwrap();
    assert_eq!(s_pairs.run(&mut l2).unwrap(), "completed");

    let mut s_hinges = JsSgdSolver::from_pairs_with_hinges(
        2.0,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[0],
        &[1],
        &[2.0],
        &[2.0],
        &[1.0],
        &[1.0],
        &JsValue::from_f64(5.0),
        None,
        Some("constant".to_owned()),
        Some(0.5),
        None,
    )
    .unwrap();
    let mut l_hinge = JsLayout2d::from_coords(&[0.0, 0.0, 0.5, 0.0]).unwrap();
    s_hinges.step(&mut l_hinge).unwrap();
    assert!((l_hinge.x(1.0).unwrap() - l_hinge.x(0.0).unwrap()) > 0.5);

    // Invalid schedule / dimensions / NaN errors
    assert!(
        JsSgdSolver::full(
            &grid,
            &JsValue::from_f64(10.0),
            None,
            Some("bad_kind".to_owned()),
            None,
            None,
        )
        .is_err()
    );
    let mut bad_layout = JsLayout2d::zeros(3.0).unwrap();
    assert!(s_om.step(&mut bad_layout).is_err());
    assert!(s_om.run(&mut bad_layout).is_err());
}

#[wasm_bindgen_test]
fn gumap_refiner_stress_majorization_and_kamada_kawai_solvers() {
    let grid = make_grid(4, 4);
    let init = make_circle_layout(16, 1.0);

    // 1. GuMapSolver (fromGraphDog, fromKnn, fromFuzzySet, step == run)
    let gm_opts = make_opts(&[
        ("nEpochs", JsValue::from_f64(20.0)),
        ("negativeSampleRate", JsValue::from_f64(4.0)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut g_run = JsGuMapSolver::from_graph_dog(
        &grid,
        &gm_opts,
        &JsValue::UNDEFINED,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .expect("GuMap.fromGraphDog");
    let mut g_step = JsGuMapSolver::from_graph_dog(
        &grid,
        &gm_opts,
        &JsValue::UNDEFINED,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .expect("GuMap.fromGraphDog step");
    assert!(g_run.num_edges() > 0);
    assert!(g_run.a() > 0.0 && g_run.b() > 0.0);

    let mut gl_run = init.clone();
    let mut gl_step = init.clone();
    let ptr_before = gl_step.coords_ptr();
    assert_eq!(g_run.run(&mut gl_run).unwrap(), "completed");
    assert_eq!(g_step.run_steps(&mut gl_step, 5.0).unwrap(), "running");
    while !g_step.is_finished() {
        g_step.step(&mut gl_step).unwrap();
    }
    assert_eq!(gl_step.coords_ptr(), ptr_before);
    assert_layouts_bit_exact(&gl_run, &gl_step);

    let knn = js_build_pure_2hop_knn(&grid).unwrap().knn();
    let fuzzy = js_compute_fuzzy_simplicial_set(&knn, None, None).unwrap();
    let g_knn = JsGuMapSolver::from_knn(
        &knn,
        &JsValue::from_f64(5.0),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let g_fuz = JsGuMapSolver::from_fuzzy_set(
        &fuzzy,
        &JsValue::from_f64(5.0),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    assert_eq!(g_knn.num_edges(), g_fuz.num_edges());

    // GuMapSolver.fromLandmarks (both options-object and positional max2hopPerNode)
    let lm_barbell = js_compute_regularized_dog_varadhan_landmarks(
        &grid,
        Some(12.0),
        Some(6.0),
        Some(42.0),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let g_lm_opts = make_opts(&[
        ("max2hopPerNode", JsValue::from_f64(32.0)),
        ("nEpochs", JsValue::from_f64(10.0)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut g_lm_run =
        JsGuMapSolver::from_landmarks(&grid, &lm_barbell, &g_lm_opts, &JsValue::UNDEFINED)
            .expect("GuMap.fromLandmarks opts");
    let mut g_lm_step = JsGuMapSolver::from_landmarks(
        &grid,
        &lm_barbell,
        &JsValue::from_f64(32.0),
        &make_opts(&[
            ("nEpochs", JsValue::from_f64(10.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .expect("GuMap.fromLandmarks pos");
    let mut glm_l1 = init.clone();
    let mut glm_l2 = init;
    assert_eq!(g_lm_run.run(&mut glm_l1).unwrap(), "completed");
    while !g_lm_step.is_finished() {
        g_lm_step.step(&mut glm_l2).unwrap();
    }
    assert_layouts_bit_exact(&glm_l1, &glm_l2);
    let lm_small = js_compute_regularized_dog_varadhan_landmarks(
        &make_grid(2, 2),
        None,
        Some(2.0),
        Some(42.0),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    assert!(
        JsGuMapSolver::from_landmarks(&grid, &lm_small, &JsValue::UNDEFINED, &JsValue::UNDEFINED)
            .is_err()
    );

    // 2. SpatialIntruderRefiner: clean grid converges early untouched; intruder pushed apart + step == run
    let small_grid = make_grid(3, 3);
    let mut clean_coords = Vec::new();
    for r in 0..3 {
        for c in 0..3 {
            clean_coords.push(c as f32);
            clean_coords.push(r as f32);
        }
    }
    let clean_orig = JsLayout2d::from_coords(&clean_coords).unwrap();
    let mut clean_layout = clean_orig.clone();
    let mut ref_clean = JsSpatialIntruderRefiner::new(
        &small_grid,
        &JsValue::from_f64(4.0),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    assert_eq!(ref_clean.run(&mut clean_layout).unwrap(), "converged");
    assert_eq!(ref_clean.stats().rounds_executed(), 0);
    assert!(ref_clean.stats().round_stats()[0].skipped());
    assert_layouts_bit_exact(&clean_orig, &clean_layout);

    let mut intr_coords = clean_coords;
    intr_coords[16] = 0.08;
    intr_coords[17] = 0.08;
    let mut intr_run = JsLayout2d::from_coords(&intr_coords).unwrap();
    let mut intr_step = intr_run.clone();
    let dist_before = (intr_run.x(0.0).unwrap() - intr_run.x(8.0).unwrap())
        .hypot(intr_run.y(0.0).unwrap() - intr_run.y(8.0).unwrap());

    let mut r_run = JsSpatialIntruderRefiner::new(
        &small_grid,
        &JsValue::from_f64(3.0),
        None,
        None,
        None,
        Some(0.2),
        None,
        None,
        None,
        None,
        None,
        Some(true),
        Some(42.0),
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let mut r_step = JsSpatialIntruderRefiner::new(
        &small_grid,
        &JsValue::from_f64(3.0),
        None,
        None,
        None,
        Some(0.2),
        None,
        None,
        None,
        None,
        None,
        Some(true),
        Some(42.0),
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    assert!(r_run.run(&mut intr_run).is_ok());
    r_step.step(&mut intr_step).unwrap();
    r_step.run(&mut intr_step).unwrap();
    assert_layouts_bit_exact(&intr_run, &intr_step);
    let r_stats = r_run.stats();
    assert!(r_stats.rounds_executed() >= 1);
    assert!(r_stats.n_fp_total() > 0);
    assert_eq!(r_stats.n_sep_total(), 0);
    assert!(!r_stats.round_stats().is_empty());
    let dist_after = (intr_run.x(0.0).unwrap() - intr_run.x(8.0).unwrap())
        .hypot(intr_run.y(0.0).unwrap() - intr_run.y(8.0).unwrap());
    assert!(dist_after > dist_before * 2.0);

    // SpatialIntruderRefiner dynamic kCap + wAnchor2 + setKCap
    let dyn_opts = make_opts(&[
        ("rounds", JsValue::from_f64(4.0)),
        ("kCap", JsValue::from_f64(8.0)),
        ("wAnchor2", JsValue::from_f64(0.5)),
        ("kCapDecay", JsValue::from_f64(0.5)),
        ("kCapMin", JsValue::from_f64(2.0)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut r_dyn = JsSpatialIntruderRefiner::new(
        &small_grid,
        &dyn_opts,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    assert_eq!(r_dyn.current_k_cap(), 8);
    let mut l_dyn = JsLayout2d::from_coords(&intr_coords).unwrap();
    r_dyn.step(&mut l_dyn).unwrap();
    assert_eq!(r_dyn.current_k_cap(), 4);
    r_dyn.set_k_cap(6.0).unwrap();
    assert_eq!(r_dyn.current_k_cap(), 6);
    r_dyn.run(&mut l_dyn).unwrap();
    assert!(l_dyn.is_all_finite());
    assert!(r_dyn.set_k_cap(0.0).is_err());

    // SpatialIntruderRefiner opt-in separation hinge (minSepRatio + wSep) cures closed-twin collapse
    let mut twin_edges: Vec<u32> = Vec::new();
    for r in 0..3u32 {
        for c in 0..3u32 {
            let u = r * 3 + c;
            if c + 1 < 3 {
                twin_edges.extend([u, u + 1]);
            }
            if r + 1 < 3 {
                twin_edges.extend([u, u + 3]);
            }
        }
    }
    twin_edges.extend([0, 9, 0, 10, 9, 10]);
    let twin_graph =
        JsCsrGraph::from_edge_list(11.0, &twin_edges, None, &JsValue::from_bool(false), None)
            .unwrap();
    let mut twin_coords = Vec::with_capacity(22);
    for r in 0..3 {
        for c in 0..3 {
            twin_coords.push(c as f32);
            twin_coords.push(r as f32);
        }
    }
    twin_coords.extend([0.0_f32, 0.0, 0.0, 0.0]);
    let mut twin_layout = JsLayout2d::from_coords(&twin_coords).unwrap();
    let sep_opts = make_opts(&[
        ("minSepRatio", JsValue::from_f64(0.15)),
        ("wSep", JsValue::from_f64(1.0)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut r_sep = JsSpatialIntruderRefiner::new(
        &twin_graph,
        &sep_opts,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    r_sep.run(&mut twin_layout).unwrap();
    let sep_stats = r_sep.stats();
    assert!(sep_stats.n_sep_total() > 0);
    assert!(sep_stats.round_stats()[0].n_sep() > 0);
    let d9_10 = (twin_layout.x(9.0).unwrap() - twin_layout.x(10.0).unwrap())
        .hypot(twin_layout.y(9.0).unwrap() - twin_layout.y(10.0).unwrap());
    assert!(d9_10 > 0.05);

    // 3. StressMajorizationSolver (fromGraph, fromDistanceMatrix, computeStress, lastStress, step == run)
    let sm_opts = make_opts(&[
        ("epsilon", JsValue::from_f64(1e-6)),
        ("maxIterations", JsValue::from_f64(30.0)),
    ]);
    let mut sm_run =
        JsStressMajorizationSolver::from_graph(&small_grid, &sm_opts, None, None, None, None)
            .expect("SM fromGraph");
    let mut sm_step =
        JsStressMajorizationSolver::from_graph(&small_grid, &sm_opts, None, None, None, None)
            .expect("SM fromGraph step");
    let mut sm_l_run = make_circle_layout(9, 0.2);
    let mut sm_l_step = sm_l_run.clone();
    let stress_before = sm_run.compute_stress(&sm_l_run).unwrap();
    assert!(sm_run.last_stress().is_none());
    sm_run.run(&mut sm_l_run).unwrap();
    let stress_after = sm_run.last_stress().unwrap();
    assert!(stress_after < stress_before * 0.5);
    assert!((stress_after - sm_run.compute_stress(&sm_l_run).unwrap()).abs() < 1e-5);

    sm_step.run_steps(&mut sm_l_step, 4.0).unwrap();
    while !sm_step.is_finished() {
        sm_step.step(&mut sm_l_step).unwrap();
    }
    assert_layouts_bit_exact(&sm_l_run, &sm_l_step);

    let sm_dm = JsStressMajorizationSolver::from_distance_matrix(
        2.0,
        &[0.0, 2.0, 2.0, 0.0],
        &JsValue::UNDEFINED,
        Some(15.0),
        None,
        None,
        None,
    )
    .unwrap();
    assert_eq!(sm_dm.num_nodes(), 2);

    // 4. KamadaKawaiSolver (fromGraph, fromDistanceMatrix, step == run)
    let kk_opts = make_opts(&[
        ("eps", JsValue::from_f64(1e-2)),
        ("maxInnerIters", JsValue::from_f64(50.0)),
        ("maxOuterIters", JsValue::from_f64(100.0)),
    ]);
    let mut kk_run = JsKamadaKawaiSolver::from_graph(&small_grid, &kk_opts, None, None).unwrap();
    let mut kk_step = JsKamadaKawaiSolver::from_graph(&small_grid, &kk_opts, None, None).unwrap();
    let mut kk_l_run = make_circle_layout(9, 1.5);
    let mut kk_l_step = kk_l_run.clone();
    kk_run.run(&mut kk_l_run).unwrap();
    kk_step.run_steps(&mut kk_l_step, 5.0).unwrap();
    while !kk_step.is_finished() {
        kk_step.step(&mut kk_l_step).unwrap();
    }
    assert_layouts_bit_exact(&kk_l_run, &kk_l_step);

    let mut kk_dm = JsKamadaKawaiSolver::from_distance_matrix(
        2.0,
        &[0.0, 3.0, 3.0, 0.0],
        &JsValue::UNDEFINED,
        None,
        Some(20.0),
    )
    .unwrap();
    let mut kk_l2 = JsLayout2d::from_coords(&[0.0, 0.0, 0.5, 0.0]).unwrap();
    kk_dm.run(&mut kk_l2).unwrap();
    let sep = (kk_l2.x(0.0).unwrap() - kk_l2.x(1.0).unwrap())
        .hypot(kk_l2.y(0.0).unwrap() - kk_l2.y(1.0).unwrap());
    assert!((sep - 3.0).abs() < 0.05);

    // Adversarial checks
    assert!(
        JsStressMajorizationSolver::from_distance_matrix(
            2.0,
            &[0.0, f64::NAN, 1.0, 0.0],
            &JsValue::UNDEFINED,
            None,
            None,
            None,
            None,
        )
        .is_err()
    );
    assert!(
        JsKamadaKawaiSolver::from_distance_matrix(
            2.0,
            &[0.0, -1.0, 1.0, 0.0],
            &JsValue::UNDEFINED,
            None,
            None,
        )
        .is_err()
    );
    assert!(
        JsSpatialIntruderRefiner::new(
            &small_grid,
            &JsValue::from_f64(2.0),
            Some(0.0),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .is_err()
    );
    assert!(
        JsGuMapSolver::from_knn(
            &knn,
            &JsValue::from_f64(5.0),
            None,
            Some(0.0),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .is_err()
    );
    assert!(JsKnnGraph::from_csr(2.0, &[0, 0, 0], &[], &[]).is_ok());
}
