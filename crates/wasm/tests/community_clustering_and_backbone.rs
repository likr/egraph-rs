mod util;

use egraph_wasm::backbone::{JsBackboneSolver, js_extract_quadrilateral_backbone};
use egraph_wasm::cluster_layout::{
    JsClusterRefineSolver, JsGroupInABoxSolver, js_refine_cluster_layout, js_squarified_treemap,
};
use egraph_wasm::community::{
    js_coarsen_graph, js_kmeans_plus_plus, js_louvain_communities, js_project_clustered_layout,
};
use egraph_wasm::community_leiden::js_leiden_communities;
use egraph_wasm::community_spectral::{js_label_propagation, js_spectral_clustering};
use egraph_wasm::csr_graph::JsCsrGraph;
use egraph_wasm::layout_buffers::JsLayout2d;
use util::fixtures::{
    assert_layouts_bit_exact, f64_js, make_barbell, make_circle_layout, make_opts,
    make_two_cluster_graph, u32_array_js,
};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

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
    assert_eq!(sp.labels(), vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);

    // 3. louvainCommunities
    let lv = js_louvain_communities(&barbell, None, None, None, Some(42.0)).expect("louvain");
    assert_eq!(lv.num_communities(), 2);
    assert_eq!(lv.labels(), vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);
    assert!(lv.modularity() > 0.3);
    assert!(lv.passes() >= 1);

    // 4. labelPropagation
    let lpa = js_label_propagation(&barbell, Some(100.0), Some(42.0)).expect("lpa");
    assert_eq!(lpa.labels(), vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);

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
fn leiden_modularity_cpm_and_invalid_inputs() {
    let (graph, _) = make_two_cluster_graph();

    let res_mod = js_leiden_communities(&graph, &JsValue::UNDEFINED).unwrap();
    assert_eq!(res_mod.labels().len(), 8);
    assert_eq!(res_mod.num_communities(), 2);
    assert!(res_mod.quality().is_finite());
    assert!(res_mod.modularity().is_finite());
    assert!(res_mod.passes() >= 1);

    let rec = res_mod.to_record().unwrap();
    assert_eq!(
        js_sys::Reflect::get(&rec, &JsValue::from_str("numCommunities"))
            .unwrap()
            .as_f64(),
        Some(2.0)
    );

    let cpm_opts = make_opts(&[
        ("objective", JsValue::from_str("cpm")),
        ("resolution", JsValue::from_f64(0.2)),
        ("randomness", JsValue::from_f64(0.01)),
        ("maxPasses", JsValue::from_f64(20.0)),
        ("minQualityGain", JsValue::from_f64(1e-7)),
        ("seed", JsValue::from_f64(7.0)),
    ]);
    let res_cpm = js_leiden_communities(&graph, &cpm_opts).unwrap();
    assert_eq!(res_cpm.labels().len(), 8);
    assert!(res_cpm.num_communities() >= 1);

    let empty = JsCsrGraph::from_edge_list(0.0, &[], None, &JsValue::UNDEFINED, None).unwrap();
    let res_empty = js_leiden_communities(&empty, &JsValue::UNDEFINED).unwrap();
    assert_eq!(res_empty.num_communities(), 0);
    assert!(res_empty.labels().is_empty());

    let directed =
        JsCsrGraph::from_edge_list(2.0, &[0, 1], None, &JsValue::from_bool(true), None).unwrap();
    assert!(js_leiden_communities(&directed, &JsValue::UNDEFINED).is_err());

    let bad_obj = make_opts(&[("objective", JsValue::from_str("bogus"))]);
    assert!(js_leiden_communities(&graph, &bad_obj).is_err());
}

#[wasm_bindgen_test]
fn backbone_extraction_and_solver_step_eq_run() {
    let (graph, _) = make_two_cluster_graph();
    let opts = make_opts(&[
        ("retainRatio", JsValue::from_f64(0.5)),
        ("umstWeightScale", JsValue::from_f64(1.0)),
        ("nonBackboneWeightScale", JsValue::from_f64(0.1)),
        ("layoutMode", JsValue::from_str("sgd")),
        ("iterations", JsValue::from_f64(12.0)),
        ("seed", JsValue::from_f64(42.0)),
    ]);

    let bb = js_extract_quadrilateral_backbone(&graph, &opts).unwrap();
    assert!(!bb.umst_edges().is_empty());
    assert_eq!(bb.umst_edges().len() % 2, 0);
    assert!(!bb.backbone_edges().is_empty());
    assert_eq!(bb.backbone_edges().len() % 2, 0);
    assert_eq!(bb.edge_weights().len(), 13);
    assert!(!bb.simmelian_weights().is_empty());
    let csr = bb.backbone_csr();
    assert_eq!(csr.num_nodes(), 8);
    assert!(csr.num_edges() > 0);

    let rec = bb.to_record().unwrap();
    assert!(
        js_sys::Reflect::get(&rec, &JsValue::from_str("backboneEdges"))
            .unwrap()
            .is_object()
    );

    let mut s_step = JsBackboneSolver::new(&graph, &opts).unwrap();
    let mut s_run = JsBackboneSolver::new(&graph, &opts).unwrap();
    assert_eq!(
        s_step.backbone_result().unwrap().backbone_edges().len(),
        bb.backbone_edges().len()
    );
    let init = make_circle_layout(8, 1.0);
    let mut l_step = init.clone();
    let mut l_run = init;
    while !s_step.is_finished() {
        s_step.step(&mut l_step).unwrap();
    }
    s_run.run(&mut l_run).unwrap();
    assert_layouts_bit_exact(&l_step, &l_run);
}

#[wasm_bindgen_test]
fn cluster_refine_treemap_and_group_in_a_box() {
    let (graph, labels) = make_two_cluster_graph();
    let labels_js = u32_array_js(&labels);

    // Swap two points so ClusterRefineSolver moves misplaced nodes
    let swapped = JsLayout2d::from_coords(&[
        -5.0, 0.0, -4.5, 0.5, -5.5, 0.5, 5.2, 0.2, // node 3 misplaced into cluster 1
        5.0, 0.0, 5.5, 0.5, 4.5, 0.5, -5.2, 0.2, // node 7 misplaced into cluster 0
    ])
    .unwrap();

    let cr_opts = make_opts(&[
        ("method", JsValue::from_str("kmeans")),
        ("maxIterations", JsValue::from_f64(5.0)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut cr_step = JsClusterRefineSolver::new(&labels_js, &cr_opts).unwrap();
    let mut cr_run = JsClusterRefineSolver::from_graph(&graph, &labels_js, &cr_opts).unwrap();
    assert_eq!(cr_step.num_clusters(), 2);
    let mut l_step = swapped.clone();
    let mut l_run = swapped.clone();
    while !cr_step.is_finished() {
        cr_step.step(&mut l_step).unwrap();
    }
    cr_run.run(&mut l_run).unwrap();
    assert_layouts_bit_exact(&l_step, &l_run);

    let mut l_fn = swapped;
    let moved = js_refine_cluster_layout(&mut l_fn, &labels_js, &cr_opts).unwrap();
    assert!(moved > 0);
    assert_layouts_bit_exact(&l_fn, &l_run);

    // Squarified treemap
    let weights = f64_js(&[6.0, 4.0, 2.0]);
    let rects =
        js_squarified_treemap(&weights, Some(100.0), Some(50.0), Some(0.0), Some(0.0)).unwrap();
    assert_eq!(rects.len(), 12);
    let mut area_sum = 0.0;
    for chunk in rects.as_chunks::<4>().0 {
        assert!(chunk[1] > chunk[0]);
        assert!(chunk[3] > chunk[2]);
        area_sum += (chunk[1] - chunk[0]) * (chunk[3] - chunk[2]);
    }
    assert!((area_sum - 5000.0).abs() < 1e-9);
    assert!(js_squarified_treemap(&f64_js(&[0.0, 1.0]), None, None, None, None).is_err());

    // GroupInABoxSolver with clusterWeights
    for box_mode in ["treemap", "force_directed"] {
        let gib_opts = make_opts(&[
            ("boxMode", JsValue::from_str(box_mode)),
            ("intraMode", JsValue::from_str("fr")),
            ("intraIterations", JsValue::from_f64(10.0)),
            ("fdIterations", JsValue::from_f64(10.0)),
            ("clusterWeights", f64_js(&[3.0, 1.0])),
            ("seed", JsValue::from_f64(42.0)),
        ]);
        let mut g_step = JsGroupInABoxSolver::new(&graph, &labels_js, &gib_opts).unwrap();
        let mut g_run = JsGroupInABoxSolver::new(&graph, &labels_js, &gib_opts).unwrap();
        let boxes = g_step.boxes().unwrap();
        assert_eq!(boxes.length(), 2);
        let first_box = boxes.get(0);
        assert_eq!(
            js_sys::Reflect::get(&first_box, &JsValue::from_str("nodeCount"))
                .unwrap()
                .as_f64(),
            Some(4.0)
        );
        let init = make_circle_layout(8, 1.0);
        let mut lg_step = init.clone();
        let mut lg_run = init;
        while !g_step.is_finished() {
            g_step.step(&mut lg_step).unwrap();
        }
        g_run.run(&mut lg_run).unwrap();
        assert_layouts_bit_exact(&lg_step, &lg_run);
    }
}
