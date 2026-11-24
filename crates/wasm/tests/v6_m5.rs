//! Integration tests verifying M5 WASM vs Native/Python API parity.

#![cfg(target_arch = "wasm32")]

use egraph_wasm::v6::{
    JsCsrGraph, JsGuMapSolver, JsIbUmapSolver, JsLayout2d, JsLayoutNd, JsManifoldLayout2d,
    JsScreenedPoissonField2d, JsScreenedPoissonRefiner, JsSgdSolver, JsTFdpSolver, JsTsNetSolver,
    js_all_pairs_bfs, js_apply_relative_distance_floor, js_bfs_single_source,
    js_build_capped_12hop_knn, js_build_gumap_12hop_priority_csr, js_build_pure_2hop_knn,
    js_build_rerank_knn, js_classical_mds_2d, js_compute_diffusion_embedding,
    js_compute_regularized_dog_varadhan_landmarks, js_compute_resistance_embedding,
    js_crossing_edges, js_crossing_edges_torus, js_crossing_summary, js_evaluate_metrics,
    js_gabriel_graph_edges, js_graph_12hop_row_normalized_probabilities,
    js_graph_row_normalized_probabilities, js_ideal_edge_lengths_with_unit, js_normalize_layout_2d,
    js_parse_csr_csv, js_poincare_distance, js_poincare_mobius_add, js_prism_distortion_with_edges,
    js_rd_mds, js_rectangle_overlaps, js_relative_neighborhood_edges, js_representative_mds,
    js_robust_normalize_layout_2d, js_scale_normalized_stress_from_distances,
    js_spherical_distance, js_symmetrize_row_normalized, js_torus_distance, js_wrap_torus_point,
};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

fn make_opts(entries: &[(&str, JsValue)]) -> JsValue {
    let obj = js_sys::Object::new();
    for (k, v) in entries {
        js_sys::Reflect::set(&obj, &JsValue::from_str(k), v).unwrap();
    }
    obj.into()
}

fn make_path_graph(n: usize) -> JsCsrGraph {
    let mut edges = Vec::with_capacity((n - 1) * 2);
    for u in 0..(n as u32 - 1) {
        edges.push(u);
        edges.push(u + 1);
    }
    JsCsrGraph::from_edge_list(
        n as f64,
        &edges,
        None,
        &JsValue::from_bool(false),
        Some(true),
    )
    .unwrap()
}

fn make_grid_graph(rows: usize, cols: usize) -> JsCsrGraph {
    let n = rows * cols;
    let mut edges = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let u = (r * cols + c) as u32;
            if c + 1 < cols {
                edges.push(u);
                edges.push(u + 1);
            }
            if r + 1 < rows {
                edges.push(u);
                edges.push(u + cols as u32);
            }
        }
    }
    JsCsrGraph::from_edge_list(
        n as f64,
        &edges,
        None,
        &JsValue::from_bool(false),
        Some(true),
    )
    .unwrap()
}

#[wasm_bindgen_test]
fn csr_graph_m5_methods_and_lcc() {
    // Disconnected graph: component {0,1} and larger component {2,3,4,5}
    let edges = [0u32, 1, 2, 3, 3, 4, 4, 5, 5, 2];
    let weights = [1.5f32, 2.0, 2.0, 2.0, 2.0];
    let g = JsCsrGraph::from_edge_list(
        6.0,
        &edges,
        Some(weights.to_vec()),
        &JsValue::from_bool(false),
        Some(true),
    )
    .unwrap();

    assert_eq!(g.nnz(), 10);
    assert!(g.is_weighted());
    assert_eq!(g.degree(2.0).unwrap(), 2);
    assert_eq!(g.neighbors(2.0).unwrap(), vec![3, 5]);
    assert!(g.degree(99.0).is_err());

    let sub = g.induced_subgraph(&[2, 3, 4]).unwrap();
    assert_eq!(sub.num_nodes(), 3);
    assert_eq!(sub.num_edges(), 2);

    let lcc = g.largest_connected_component();
    assert_eq!(lcc.graph().num_nodes(), 4);
    assert_eq!(lcc.original_node_ids(), vec![2, 3, 4, 5]);
    assert_eq!(lcc.node_ids(), vec![2, 3, 4, 5]);
}

#[wasm_bindgen_test]
fn layout_2d_and_nd_m5_methods() {
    let mut l2 = JsLayout2d::init(8.0, Some(123.0)).unwrap();
    assert_eq!(l2.num_nodes(), 8);
    assert!(!l2.is_empty());
    assert!(l2.is_all_finite());
    assert_eq!(l2.copy_coords().len(), 16);

    l2.normalize_std(Some(2.0)).unwrap();
    let coords = l2.copy_coords();
    let mut mean_x = 0.0f64;
    let mut mean_y = 0.0f64;
    for i in 0..8 {
        mean_x += coords[2 * i] as f64 / 8.0;
        mean_y += coords[2 * i + 1] as f64 / 8.0;
    }
    assert!(mean_x.abs() < 1e-4 && mean_y.abs() < 1e-4);

    let nd = JsLayoutNd::zeros(4.0, 3.0).unwrap();
    assert_eq!(nd.num_nodes(), 4);
    assert_eq!(nd.dim(), 3);
    assert_eq!(nd.len(), 4);
    assert_eq!(nd.coords_len(), 12);
    assert!(nd.coords_ptr() > 0);
    assert!(!nd.is_empty());
    assert!(nd.is_all_finite());

    let nd2 = JsLayoutNd::from_coords(2.0, 3.0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert_eq!(nd2.copy_coords(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert!(JsLayoutNd::from_coords(2.0, 3.0, &[1.0]).is_err());
}

#[wasm_bindgen_test]
fn manifold_layout_and_distances() {
    let mut hyp = JsManifoldLayout2d::hyperbolic_poincare(3.0).unwrap();
    assert_eq!(hyp.manifold(), "hyperbolic");
    assert_eq!(hyp.num_nodes(), 3);
    hyp.set_point(0.0, 0.1, 0.2).unwrap();
    hyp.set_point(1.0, -0.1, 0.05).unwrap();
    let delta = hyp.tangent_delta(0.0, 1.0).unwrap();
    assert_eq!(delta.len(), 2);
    hyp.apply_tangent_step(0.0, delta[0] * 0.1, delta[1] * 0.1)
        .unwrap();
    assert!(js_poincare_distance(0.0, 0.0, 0.3, 0.4) > 0.5);
    let m_add = js_poincare_mobius_add(-0.3, 0.4, 0.3, -0.4);
    assert!(m_add[0].hypot(m_add[1]) < 1e-12);
    let mut m_rand =
        JsManifoldLayout2d::random_init("hyperbolic", 4.0, Some(0.25), Some(42.0)).unwrap();
    m_rand.center_poincare_centroid().unwrap();
    let mc = m_rand.poincare_centroid().unwrap();
    assert!(mc[0].hypot(mc[1]) < 1e-4);

    let sph = JsManifoldLayout2d::spherical(2.0).unwrap();
    assert_eq!(sph.manifold(), "spherical");
    assert!(
        (js_spherical_distance(0.0, 0.0, 0.0, std::f64::consts::FRAC_PI_2)
            - std::f64::consts::FRAC_PI_2)
            .abs()
            < 1e-9
    );

    let tor = JsManifoldLayout2d::torus_2d(2.0).unwrap();
    assert_eq!(tor.manifold(), "torus");
    assert!((js_torus_distance(0.05, 0.05, 0.95, 0.05) - 0.1).abs() < 1e-9);
    let wrapped = js_wrap_torus_point(1.25, -0.25);
    assert!((wrapped[0] - 0.25).abs() < 1e-9 && (wrapped[1] - 0.75).abs() < 1e-9);
}

#[wasm_bindgen_test]
fn dataset_spatial_and_metrics_m5_apis() {
    let parsed = js_parse_csr_csv("3,2\n0,1\n1,2\n", Some(false), Some(true)).unwrap();
    assert_eq!(parsed.num_nodes(), 3);
    assert_eq!(parsed.num_edges(), 2);

    let rects = JsLayout2d::from_coords(&[0.0, 0.0, 0.5, 0.0, 5.0, 0.0]).unwrap();
    let overlaps = js_rectangle_overlaps(&rects, &[0.5, 0.5, 0.5], &[0.5, 0.5, 0.5]).unwrap();
    assert_eq!(overlaps, vec![0, 1]);

    // X-shaped graph for crossings
    let cross_graph = JsCsrGraph::from_edge_list(
        4.0,
        &[0, 3, 1, 2],
        None,
        &JsValue::from_bool(false),
        Some(true),
    )
    .unwrap();
    let cross_layout = JsLayout2d::from_coords(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let pairs = js_crossing_edges(&cross_graph, &cross_layout).unwrap();
    assert_eq!(pairs.len(), 1);
    let summary = js_crossing_summary(&cross_graph, &cross_layout).unwrap();
    assert_eq!(summary.count(), 1);
    assert!(summary.min_angle_rad().unwrap_or(0.0) > 0.0);
    assert!(summary.angle_cos_sq_sum() >= 0.0);

    let torus_pairs = js_crossing_edges_torus(&cross_graph, &cross_layout).unwrap();
    assert!(!torus_pairs.is_empty());

    let grid = make_grid_graph(3, 3);
    let mut grid_coords = Vec::new();
    for r in 0..3 {
        for c in 0..3 {
            grid_coords.push(c as f32);
            grid_coords.push(r as f32);
        }
    }
    let grid_layout = JsLayout2d::from_coords(&grid_coords).unwrap();
    let iel_unit = js_ideal_edge_lengths_with_unit(&grid, &grid_layout, 1.0).unwrap();
    assert!(iel_unit.is_finite());

    let targets = js_sys::Array::of2(&"sns".into(), &"idealEdgeLengths".into());
    let report = js_evaluate_metrics(
        &grid,
        &grid_layout,
        &targets,
        None,
        None,
        None,
        Some(1.0),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    assert!(report.scale_normalized_stress().is_some());
    assert!(report.ideal_edge_lengths().is_some());

    let gg_edges = js_gabriel_graph_edges(&grid_layout).unwrap();
    let rng_edges = js_relative_neighborhood_edges(&grid_layout).unwrap();
    assert!(!gg_edges.is_empty() && !rng_edges.is_empty());

    let apsp = js_all_pairs_bfs(&grid, Some(1.0)).unwrap();
    assert_eq!(apsp.len(), 81);
    let sssp = js_bfs_single_source(&grid, 0.0, Some(2.0)).unwrap();
    assert_eq!(sssp.len(), 9);
    assert!((sssp[8] - 8.0).abs() < 1e-9);
    let sns = js_scale_normalized_stress_from_distances(&grid_layout, &apsp, None).unwrap();
    assert!(sns.is_finite() && sns < 0.25);

    let mut adjusted = grid_layout.clone();
    adjusted.set_x(0.0, 0.05).unwrap();
    let p_dist = js_prism_distortion_with_edges(&grid_layout, &adjusted, &gg_edges).unwrap();
    assert!(p_dist.sigma_dist().is_finite() && p_dist.sigma_disp().is_finite());

    #[cfg(feature = "delaunay")]
    {
        let del_csr = egraph_wasm::v6::js_delaunay_to_csr(&grid_layout).unwrap();
        assert_eq!(del_csr.num_nodes(), 9);
        assert!(del_csr.num_edges() >= 12);
    }
}

#[wasm_bindgen_test]
fn linalg_and_neighborhood_m5_apis() {
    let grid = make_grid_graph(4, 4);
    let mut cmds = js_classical_mds_2d(&grid).unwrap();
    assert_eq!(cmds.num_nodes(), 16);
    assert!(cmds.is_all_finite());
    js_normalize_layout_2d(&mut cmds, Some(1.0), Some(42.0)).unwrap();
    js_robust_normalize_layout_2d(&mut cmds, Some(1.0), Some(0.95), Some(42.0)).unwrap();

    let rd = js_rd_mds(&grid, Some(3.0), None, None, Some(200.0), Some(42.0)).unwrap();
    assert_eq!(rd.num_nodes(), 16);
    assert_eq!(rd.dim(), 3);
    assert!(rd.is_all_finite());

    // 4 nodes x 2 representatives
    let d_reps = [0.0, 3.0, 1.0, 2.0, 2.0, 1.0, 3.0, 0.0];
    let rep_nd = js_representative_mds(&d_reps, 4.0, 2.0, 2.0).unwrap();
    assert_eq!(rep_nd.num_nodes(), 4);
    assert_eq!(rep_nd.dim(), 2);
    assert!(rep_nd.is_all_finite());

    let res_emb =
        js_compute_resistance_embedding(&grid, Some(3.0), None, None, Some(42.0), None).unwrap();
    assert_eq!(res_emb.num_nodes(), 16);
    assert!(!res_emb.lambdas().is_empty());

    let diff_emb =
        js_compute_diffusion_embedding(&grid, Some(3.0), Some(0.5), None, None, None, Some(42.0))
            .unwrap();
    assert_eq!(diff_emb.num_nodes(), 16);
    assert!(diff_emb.t_used().is_some());

    let rerank = js_build_rerank_knn(
        &diff_emb.coords(),
        16.0,
        diff_emb.dim() as f64,
        4.0,
        Some(2.0),
        Some(2.0),
    )
    .unwrap();
    assert_eq!(rerank.num_nodes(), 16);
    assert!(rerank.nnz() > 0);

    let row_probs = js_graph_row_normalized_probabilities(&grid).unwrap();
    assert_eq!(row_probs.num_nodes(), 16);
    let row_probs_12 = js_graph_12hop_row_normalized_probabilities(
        &grid,
        Some(0.5),
        Some(4.0),
        Some(0.35),
        Some(42.0),
    )
    .unwrap();
    assert_eq!(row_probs_12.num_nodes(), 16);
    let capped_knn = js_build_capped_12hop_knn(&grid, Some(4.0), Some(42.0)).unwrap();
    assert_eq!(capped_knn.knn().num_nodes(), 16);
    let sym =
        js_symmetrize_row_normalized(3.0, &[0, 1, 2, 3], &[1, 2, 0], &[1.0, 1.0, 1.0]).unwrap();
    assert_eq!(sym.num_nodes(), 3);

    let prio = js_build_gumap_12hop_priority_csr(
        3.0,
        &[0],
        &[1],
        &[1.0],
        &[0, 1],
        &[2, 2],
        &[2.0, 2.0],
        Some(1.1),
    )
    .unwrap();
    assert_eq!(prio.num_nodes(), 3);
    assert!(prio.nnz() > 0);

    let mut d1 = [1e-6_f64, 1.0];
    let mut d2 = [2.0_f64];
    let mut d3 = [5.0_f64];
    let floor =
        js_apply_relative_distance_floor(&mut d1, &mut d2, &mut d3, Some(1e-3), Some(0.1)).unwrap();
    assert!(floor >= 0.04 && floor.is_finite());
    assert!(d1[0] >= floor);
}

#[wasm_bindgen_test]
fn solvers_m5_parity_apis() {
    let grid = make_grid_graph(4, 4);
    let init = js_classical_mds_2d(&grid).unwrap();

    // TsNetSolver.fromGraphPartialBfs + lastFftStats + short FFT options
    let ts_opts = make_opts(&[
        ("backend", JsValue::from_str("fit")),
        ("iterationsStage1", JsValue::from_f64(2.0)),
        ("iterationsStage2", JsValue::from_f64(2.0)),
        ("iterationsStage3", JsValue::from_f64(2.0)),
        ("intervals", JsValue::from_f64(1.0)),
        ("interpolationOrder", JsValue::from_f64(1.0)),
        ("minGridDim", JsValue::from_f64(16.0)),
        ("maxGridDim", JsValue::from_f64(64.0)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut ts = JsTsNetSolver::from_graph_partial_bfs(
        &grid,
        Some(8.0),
        Some(6.0),
        &ts_opts,
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
    .unwrap();
    let mut l_ts = init.clone();
    assert!(ts.last_fft_stats().is_none());
    ts.run(&mut l_ts).unwrap();
    let fft_stats = ts.last_fft_stats().expect("lastFftStats after FFT run");
    assert!(fft_stats.total_fft_2d_calls() > 0.0);
    assert!(fft_stats.particle_forward_ffts() > 0.0);

    // TFdpSolver.fromKnn accepting a KnnGraph instance directly
    let knn = js_build_pure_2hop_knn(&grid).unwrap().knn();
    let tfdp_opts = make_opts(&[
        ("backend", JsValue::from_str("exact")),
        ("totalIterations", JsValue::from_f64(4.0)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut tfdp = JsTFdpSolver::from_knn(
        &JsValue::from(knn.clone()),
        &tfdp_opts,
        &JsValue::UNDEFINED,
        &JsValue::UNDEFINED,
    )
    .unwrap();
    let mut l_tfdp = init.clone();
    tfdp.run(&mut l_tfdp).unwrap();
    assert!(l_tfdp.is_all_finite());

    // SgdSolver.dogStats + optional fromPairs args
    let mut sgd = JsSgdSolver::sparse_barrier_dog(
        &grid,
        &make_opts(&[
            ("tMax", JsValue::from_f64(5.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
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
    let d_stats = sgd.dog_stats().expect("dogStats from sparseBarrierDog");
    assert!(d_stats.degree() > 0);
    assert!(d_stats.t2() > d_stats.t1() && d_stats.t1() > 0.0);
    let mut l_sgd = init.clone();
    sgd.run(&mut l_sgd).unwrap();

    let mut pairs_opt = JsSgdSolver::from_pairs(
        2.0,
        &[0],
        &[1],
        &[1.5],
        None,
        None,
        None,
        &JsValue::from_f64(4.0),
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let mut l2 = JsLayout2d::from_coords(&[0.0, 0.0, 0.5, 0.0]).unwrap();
    pairs_opt.run(&mut l2).unwrap();
    assert!(l2.is_all_finite());

    // GuMapSolver.fromGraphDog positional minDegree/jlDim + fromFuzzySet sigmaIters/sigmaTolerance
    let path = make_path_graph(6);
    let gm_pos = JsGuMapSolver::from_graph_dog(
        &path,
        &JsValue::UNDEFINED,
        &JsValue::from_f64(5.0),
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
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(2.0),
        Some(8.0),
    )
    .unwrap();
    assert!(gm_pos.num_edges() > 0);
    let gm_ab = gm_pos.ab();
    assert_eq!(gm_ab.len(), 2);
    assert!((gm_ab[0] - gm_pos.a()).abs() < 1e-12);
    assert!((gm_ab[1] - gm_pos.b()).abs() < 1e-12);

    let ib = JsIbUmapSolver::new(
        &grid,
        &make_opts(&[
            ("backend", JsValue::from_str("exact")),
            ("nEpochs", JsValue::from_f64(3.0)),
            ("minDegree", JsValue::from_f64(1.0)),
            ("jlDim", JsValue::from_f64(8.0)),
        ]),
    )
    .unwrap();
    let ib_ab = ib.ab();
    assert_eq!(ib_ab.len(), 2);
    assert!((ib_ab[0] - ib.a()).abs() < 1e-12);
    assert!((ib_ab[1] - ib.b()).abs() < 1e-12);

    // Default degree=None in computeRegularizedDogVaradhanLandmarks matches adaptive degree >= 40
    let lm_def = js_compute_regularized_dog_varadhan_landmarks(
        &grid, None, None, None, None, None, None, None, None, None, None, None,
    )
    .unwrap();
    assert_eq!(lm_def.num_nodes(), 16);
}

#[wasm_bindgen_test]
fn m5_1_louvain_coarsen_rhgh_and_coarse_backbone_parity() {
    use egraph_wasm::v6::{
        js_align_clustered_centroids, js_coarse_backbone_init, js_compute_pivoted_rhgh_landmarks,
        js_louvain_coarsen, js_project_clustered_layout_with_radii,
    };

    // 1. Three-cluster chain A(6) - B(16) - C(6)
    let small = 6usize;
    let big = 16usize;
    let n = 2 * small + big;
    let mut edges = Vec::new();
    for i in 0..small {
        for j in (i + 1)..small {
            edges.push(i as u32);
            edges.push(j as u32);
        }
    }
    for i in 0..big {
        for j in (i + 1)..big {
            edges.push((small + i) as u32);
            edges.push((small + j) as u32);
        }
    }
    let c0 = small + big;
    for i in 0..small {
        for j in (i + 1)..small {
            edges.push((c0 + i) as u32);
            edges.push((c0 + j) as u32);
        }
    }
    edges.push((small - 1) as u32);
    edges.push(small as u32);
    edges.push((small + big - 1) as u32);
    edges.push(c0 as u32);
    let chain = JsCsrGraph::from_edge_list(
        n as f64,
        &edges,
        None,
        &JsValue::from_bool(false),
        Some(true),
    )
    .unwrap();

    let h = js_louvain_coarsen(
        &chain,
        &make_opts(&[
            ("maxCoarseNodes", JsValue::from_f64(3.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    assert!(h.num_clusters() <= 3);
    assert!(h.num_levels() >= 1);
    assert_eq!(h.node_to_coarse().len(), n);

    // 2. Pivoted RHGH landmarks on path(10)
    let path = make_path_graph(10);
    let rhgh = js_compute_pivoted_rhgh_landmarks(
        &path,
        &make_opts(&[
            ("numLandmarks", JsValue::from_f64(10.0)),
            ("minDegree", JsValue::from_f64(1.0)),
            ("numProbes", JsValue::from_f64(16.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    assert!(rhgh.all_converged());
    let d_alt = rhgh.landmarks().alt_distances(&[0, 0], &[1, 9]).unwrap();
    assert!(d_alt[1] > d_alt[0]);

    // 3. Pitfall #5 coarse backbone init (works under BOTH default and --no-default-features!)
    let bb = js_coarse_backbone_init(
        &chain,
        &make_opts(&[
            ("maxCoarseNodes", JsValue::from_f64(3.0)),
            ("c0", JsValue::from_f64(0.8)),
            ("kappa", JsValue::from_f64(1.0)),
            ("muOv", JsValue::from_f64(2.0)),
            ("mGap", JsValue::from_f64(0.1)),
            (
                "edgeLengthMode",
                JsValue::from_str("resistanceCurvatureFlow"),
            ),
            ("distanceBackend", JsValue::from_str("rhgh")),
            ("coarseTMax", JsValue::from_f64(30.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    assert_eq!(bb.num_fine_nodes(), n);
    assert_eq!(bb.num_coarse_nodes(), 3);
    assert_eq!(bb.coarse_overlap_pairs_final(), 0);
    assert!(bb.fine_layout().is_all_finite());

    // 4. Pitfall #7 centroid alignment + radii projection
    let coarse = JsLayout2d::from_coords(&[5.0, -3.0, -5.0, 2.0]).unwrap();
    let mut fine = JsLayout2d::from_coords(&[
        0.0, 0.0, 2.5, 0.25, 5.0, 0.0, 7.5, -0.25, 10.0, 0.0, 0.0, 0.0,
    ])
    .unwrap();
    let n2c = [0u32, 0, 0, 0, 0, 1];
    js_align_clustered_centroids(&mut fine, &coarse, &n2c, Some(1.0)).unwrap();
    assert!(fine.is_all_finite());
    let proj =
        js_project_clustered_layout_with_radii(&coarse, &n2c, &[2.0, 4.0], Some(0.5), Some(7.0))
            .unwrap();
    assert_eq!(proj.num_nodes(), 6);

    // 5. SparseBarrierDog with degreeWeightBeta + pacmapGraphWithBackbone
    let grid = make_grid_graph(6, 6);
    let mut sgd_beta = JsSgdSolver::sparse_barrier_dog(
        &grid,
        &make_opts(&[
            ("degreeWeightBeta", JsValue::from_f64(0.5)),
            ("minDegree", JsValue::from_f64(1.0)),
            ("jlDim", JsValue::from_f64(8.0)),
            ("tMax", JsValue::from_f64(5.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
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
    let mut l_beta = JsLayout2d::init(36.0, Some(42.0)).unwrap();
    sgd_beta.run(&mut l_beta).unwrap();
    assert!(l_beta.is_all_finite());

    let pm_bb = JsSgdSolver::pacmap_graph_with_backbone(
        &grid,
        &make_opts(&[
            ("nNeighbors", JsValue::from_f64(8.0)),
            ("phase1Iters", JsValue::from_f64(10.0)),
            ("phase2Iters", JsValue::from_f64(10.0)),
            ("phase3Iters", JsValue::from_f64(10.0)),
            ("numLandmarks", JsValue::from_f64(12.0)),
            ("maxCoarseNodes", JsValue::from_f64(9.0)),
            ("coarseTMax", JsValue::from_f64(20.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    let mut pm_solver = pm_bb.solver();
    let mut pm_layout = pm_bb.fine_layout();
    pm_solver.step(&mut pm_layout).unwrap();
    pm_solver.run(&mut pm_layout).unwrap();
    assert!(pm_layout.is_all_finite());

    // 6. Challenger reproducer: 20-node two-10-clique bridge with jitterScale=0.0, targetStd=1.0
    let mut two_clique_edges = Vec::new();
    for i in 0..10u32 {
        for j in (i + 1)..10u32 {
            two_clique_edges.push(i);
            two_clique_edges.push(j);
            two_clique_edges.push(10 + i);
            two_clique_edges.push(10 + j);
        }
    }
    two_clique_edges.push(9);
    two_clique_edges.push(10);
    let g_two = JsCsrGraph::from_edge_list(
        20.0,
        &two_clique_edges,
        None,
        &JsValue::from_bool(false),
        Some(true),
    )
    .unwrap();
    let bb_two = js_coarse_backbone_init(
        &g_two,
        &make_opts(&[
            ("maxCoarseNodes", JsValue::from_f64(2.0)),
            ("jitterScale", JsValue::from_f64(0.0)),
            ("targetStd", JsValue::from_f64(1.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    assert_eq!(bb_two.num_coarse_nodes(), 2);
    assert!((bb_two.scale_factor_applied() - 1.0).abs() > 1e-3);
    let fl = bb_two.fine_layout().copy_coords();
    let cl = bb_two.coarse_layout().copy_coords();
    let n2c = bb_two.node_to_coarse();
    for cid in 0..2u32 {
        let mut sx = 0.0_f64;
        let mut sy = 0.0_f64;
        let mut cnt = 0usize;
        for (idx, &c) in n2c.iter().enumerate() {
            if c == cid {
                sx += fl[2 * idx] as f64;
                sy += fl[2 * idx + 1] as f64;
                cnt += 1;
            }
        }
        let mx = sx / (cnt as f64);
        let my = sy / (cnt as f64);
        let diff =
            (mx - cl[2 * (cid as usize)] as f64).hypot(my - cl[2 * (cid as usize) + 1] as f64);
        assert!(
            diff < 1e-5,
            "WASM cluster {cid} centroid mismatch: diff={diff}"
        );
    }
}

#[wasm_bindgen_test]
fn m5_2_screened_poisson_density_field_and_refiner_parity() {
    // 1. Finite-difference gradient oracle & Pitfall #1 outward sign check.
    let mut coords = Vec::new();
    let h = 12.0_f64 / 128.0_f64;
    for i in 0..10 {
        let ang = 0.31 + (i as f64) * std::f64::consts::TAU / 10.0;
        let mut x = 0.17 + 0.25 * ang.cos();
        let mut y = -0.13 + 0.25 * ang.sin();
        let ux = ((x + 6.0) / h).fract();
        let uy = ((y + 6.0) / h).fract();
        if ux.min(1.0 - ux) < 0.1 {
            x += 0.25 * h;
        }
        if uy.min(1.0 - uy) < 0.1 {
            y += 0.25 * h;
        }
        coords.push(x as f32);
        coords.push(y as f32);
    }
    for i in 0..4 {
        let ang = 0.43 + (i as f64) * std::f64::consts::TAU / 4.0;
        coords.push((0.17 + 0.70 * ang.cos()) as f32);
        coords.push((-0.13 + 0.70 * ang.sin()) as f32);
    }
    coords.extend_from_slice(&[3.23_f32, -2.91, -3.13, 2.83]);

    let bounds = [-5.0_f64, 5.0, -5.0, 5.0];
    for (exact_adjoint, rho_max) in [(true, 0.05_f64), (false, 0.0_f64)] {
        let mut field = JsScreenedPoissonField2d::new(&make_opts(&[
            ("kernel", JsValue::from_str("screenedPoisson")),
            ("gridSize", JsValue::from_f64(128.0)),
            ("paddingFactor", JsValue::from_f64(2.0)),
            ("sigma", JsValue::from_f64(0.35)),
            ("kappa", JsValue::from_f64(0.5)),
            ("rhoMax", JsValue::from_f64(rho_max)),
            ("exactAdjoint", JsValue::from_bool(exact_adjoint)),
        ]))
        .unwrap();

        let l_base = JsLayout2d::from_coords(&coords).unwrap();
        let res = field.evaluate_in_bounds(&l_base, &bounds).unwrap();
        assert!(!res.early_exit_no_excess());
        assert_eq!(res.fft_2d_calls() as u64, if exact_adjoint { 6 } else { 4 });

        let fx = res.force_x();
        let fy = res.force_y();
        let eps = 2e-3_f32;
        let mut max_err = 0.0_f64;
        let mut max_grad = 0.0_f64;
        for i in 0..16 {
            for (axis, anal_f) in [(0usize, fx[i] as f64), (1usize, fy[i] as f64)] {
                let mut cp = coords.clone();
                let mut cm = coords.clone();
                cp[2 * i + axis] += eps;
                cm[2 * i + axis] -= eps;
                let lp = field
                    .evaluate_in_bounds(&JsLayout2d::from_coords(&cp).unwrap(), &bounds)
                    .unwrap()
                    .energy();
                let lm = field
                    .evaluate_in_bounds(&JsLayout2d::from_coords(&cm).unwrap(), &bounds)
                    .unwrap()
                    .energy();
                let fd = (lp - lm) / (2.0 * (eps as f64));
                max_grad = max_grad.max(anal_f.abs());
                max_err = max_err.max(((-anal_f) - fd).abs());
            }
        }
        assert!(max_grad > 1e-4);
        assert!(max_err / max_grad < 1e-2 && max_err < 2e-4);
    }

    // 2. Unilateral early-exit zero force.
    let mut grid_coords = Vec::with_capacity(128);
    for r in 0..8 {
        for c in 0..8 {
            grid_coords.push(c as f32);
            grid_coords.push(r as f32);
        }
    }
    let mut field_ee = JsScreenedPoissonField2d::new(&make_opts(&[
        ("gridSize", JsValue::from_f64(128.0)),
        ("rhoMax", JsValue::from_f64(10.0)),
    ]))
    .unwrap();
    let res_ee = field_ee
        .evaluate(&JsLayout2d::from_coords(&grid_coords).unwrap())
        .unwrap();
    assert!(res_ee.early_exit_no_excess());
    assert_eq!(res_ee.energy(), 0.0);
    assert_eq!(res_ee.fft_2d_calls() as u64, 2);

    // 3. Refiner step == run, fromGraph, setCoarseTether, applyEpochStep.
    let mut disk_coords = Vec::new();
    for r in 0..16 {
        for c in 0..16 {
            let x = -0.8_f32 + 1.6_f32 * (c as f32) / 15.0;
            let y = -0.8_f32 + 1.6_f32 * (r as f32) / 15.0;
            if x * x + y * y <= 0.64 {
                disk_coords.push(x);
                disk_coords.push(y);
            }
        }
    }
    let n_disk = disk_coords.len() / 2;
    let ref_opts = make_opts(&[
        ("gridSize", JsValue::from_f64(64.0)),
        ("sigma", JsValue::from_f64(0.10)),
        ("kappa", JsValue::from_f64(1.0)),
        ("rhoMax", JsValue::from_f64(0.15)),
        ("iterations", JsValue::from_f64(4.0)),
        ("stepSize", JsValue::from_f64(0.3)),
        ("momentum", JsValue::from_f64(0.4)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut r_run = JsScreenedPoissonRefiner::new(n_disk as f64, &ref_opts).unwrap();
    let mut r_step = JsScreenedPoissonRefiner::new(n_disk as f64, &ref_opts).unwrap();
    let mut l_run = JsLayout2d::from_coords(&disk_coords).unwrap();
    let mut l_step = JsLayout2d::from_coords(&disk_coords).unwrap();
    r_run.run(&mut l_run).unwrap();
    r_step.run_steps(&mut l_step, 2.0).unwrap();
    assert_eq!(r_step.iteration(), 2);
    r_step.run(&mut l_step).unwrap();
    assert_eq!(l_run.copy_coords(), l_step.copy_coords());

    let g4 = make_path_graph(4);
    let mut r_g = JsScreenedPoissonRefiner::from_graph(
        &g4,
        &make_opts(&[
            ("gridSize", JsValue::from_f64(64.0)),
            ("rhoMax", JsValue::from_f64(0.1)),
            ("anchorWeight", JsValue::from_f64(0.2)),
            ("gammaTether", JsValue::from_f64(1.0)),
            ("recenter", JsValue::from_bool(false)),
        ]),
    )
    .unwrap();
    let coarse = JsLayout2d::from_coords(&[5.0_f32, 2.0, -5.0, -2.0]).unwrap();
    r_g.set_coarse_tether(&coarse, &[0, 0, 1, 1]).unwrap();
    let mut l4 = JsLayout2d::from_coords(&[0.0_f32, 0.0, 0.1, 0.0, 0.1, 0.1, 0.0, 0.1]).unwrap();
    let ep_st = r_g.apply_epoch_step(&mut l4).unwrap();
    assert!(!ep_st.early_exit_no_excess());
    let c4 = l4.copy_coords();
    assert!((0.5 * (c4[0] + c4[2]) - 5.0).abs() < 1e-4);

    // 4. Adversarial error checks.
    assert!(
        JsScreenedPoissonField2d::new(&make_opts(&[("kernel", JsValue::from_str("badKernel"))]))
            .is_err()
    );
    assert!(
        JsScreenedPoissonField2d::new(&make_opts(&[("gridSize", JsValue::from_f64(100.0))]))
            .is_err()
    );
}
