#![cfg(target_arch = "wasm32")]

use egraph_wasm::v6::{
    JsBackboneSolver, JsClusterRefineSolver, JsCsrGraph, JsForbidOverlapRemoval,
    JsForceAtlas2Solver, JsFruchtermanReingoldSolver, JsGTreeOverlapRemoval, JsGroupInABoxSolver,
    JsLayout2d, JsLinLogSolver, js_calinski_harabasz_score, js_class_density_measure, js_cq_hac,
    js_cq_kmeans, js_davies_bouldin_score, js_distance_consistency, js_distribution_consistency,
    js_evaluate_metrics_entry, js_extract_quadrilateral_backbone, js_leiden_communities,
    js_neighborhood_hit, js_nor_metrics, js_nor_metrics_without_graph,
    js_pairwise_distance_overlap_error, js_partition_agreement, js_refine_cluster_layout,
    js_sepme_gab, js_sepme_gong, js_sepme_knng, js_sepme_rng, js_silhouette_score, js_sprawlter_nn,
    js_sprawlter_nn_radii, js_squarified_treemap,
};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

fn make_two_cluster_graph() -> (JsCsrGraph, Vec<u32>) {
    // Two 4-cliques (0..4 and 4..8) connected by a single bridge (3, 4).
    let mut edges = Vec::new();
    for off in [0u32, 4u32] {
        for i in 0..4 {
            for j in (i + 1)..4 {
                edges.push(off + i);
                edges.push(off + j);
            }
        }
    }
    edges.push(3);
    edges.push(4);
    let graph = JsCsrGraph::from_edge_list(8.0, &edges, None, &JsValue::UNDEFINED, None)
        .expect("two-cluster graph");
    let labels = vec![0, 0, 0, 0, 1, 1, 1, 1];
    (graph, labels)
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

fn make_separated_cluster_layout() -> JsLayout2d {
    JsLayout2d::from_coords(&[
        -5.0, 0.0, -4.5, 0.5, -5.5, 0.5, -5.0, 1.0, 5.0, 0.0, 5.5, 0.5, 4.5, 0.5, 5.0, 1.0,
    ])
    .expect("separated layout")
}

fn make_overlapping_layout() -> JsLayout2d {
    JsLayout2d::from_coords(&[
        0.0, 0.0, 0.1, 0.05, -0.08, 0.07, 0.05, -0.06, 0.12, 0.1, -0.1, -0.08, 0.02, 0.09, -0.04,
        0.02,
    ])
    .expect("overlapping layout")
}

fn make_opts(entries: &[(&str, JsValue)]) -> js_sys::Object {
    let obj = js_sys::Object::new();
    for (k, v) in entries {
        js_sys::Reflect::set(&obj, &JsValue::from_str(k), v).unwrap();
    }
    obj
}

fn u32_js(vals: &[u32]) -> JsValue {
    let arr = js_sys::Array::new();
    for &v in vals {
        arr.push(&JsValue::from_f64(v as f64));
    }
    arr.into()
}

fn f64_js(vals: &[f64]) -> JsValue {
    let arr = js_sys::Array::new();
    for &v in vals {
        arr.push(&JsValue::from_f64(v));
    }
    arr.into()
}

fn assert_layouts_bit_exact(a: &JsLayout2d, b: &JsLayout2d) {
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        let idx = i as f64;
        assert_eq!(a.x(idx).unwrap().to_bits(), b.x(idx).unwrap().to_bits());
        assert_eq!(a.y(idx).unwrap().to_bits(), b.y(idx).unwrap().to_bits());
    }
}

fn count_rect_overlaps(layout: &JsLayout2d, hw: &[f32], hh: &[f32]) -> usize {
    let n = layout.len();
    let mut count = 0usize;
    for i in 0..n {
        let xi = layout.x(i as f64).unwrap();
        let yi = layout.y(i as f64).unwrap();
        for j in (i + 1)..n {
            let xj = layout.x(j as f64).unwrap();
            let yj = layout.y(j as f64).unwrap();
            if (xi - xj).abs() < hw[i] + hw[j] && (yi - yj).abs() < hh[i] + hh[j] {
                count += 1;
            }
        }
    }
    count
}

#[wasm_bindgen_test]
fn w1_1_leiden_modularity_cpm_and_invalid_inputs() {
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
fn w1_2_linlog_fr_and_fa2_overlap_step_eq_run() {
    let (graph, _) = make_two_cluster_graph();
    let init = make_circle_layout(8, 2.0);

    // LinLogSolver (edge_repulsion & node_repulsion)
    for rep in ["edge_repulsion", "node_repulsion"] {
        let opts = make_opts(&[
            ("iterations", JsValue::from_f64(10.0)),
            ("repulsionModel", JsValue::from_str(rep)),
            ("useBarnesHut", JsValue::from_bool(true)),
            ("seed", JsValue::from_f64(42.0)),
        ]);
        let mut s_step = JsLinLogSolver::new(&graph, &opts).unwrap();
        let mut s_run = JsLinLogSolver::new(&graph, &opts).unwrap();
        assert_eq!(s_step.num_nodes(), 8);
        assert_eq!(s_step.total_iterations(), 10);
        assert_eq!(s_step.masses().len(), 8);
        assert!(s_step.max_displacement() > 0.0);

        let mut l_step = init.clone();
        let mut l_run = init.clone();
        while !s_step.is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        let status = s_run.run(&mut l_run).unwrap();
        assert!(matches!(status.as_str(), "completed" | "converged"));
        assert_layouts_bit_exact(&l_step, &l_run);
        assert!(s_run.energy(&l_run).unwrap().is_finite());
    }
    let bad_ll = make_opts(&[
        ("attractionExponent", JsValue::from_f64(-1.0)),
        ("repulsionExponent", JsValue::from_f64(0.0)),
    ]);
    assert!(JsLinLogSolver::new(&graph, &bad_ll).is_err());

    // FruchtermanReingoldSolver (exact, barnes_hut, grid)
    for backend in ["exact", "barnes_hut", "grid"] {
        let opts = make_opts(&[
            ("iterations", JsValue::from_f64(10.0)),
            ("backend", JsValue::from_str(backend)),
            ("gravity", JsValue::from_f64(0.1)),
            ("seed", JsValue::from_f64(42.0)),
        ]);
        let mut s_step = JsFruchtermanReingoldSolver::new(&graph, &opts).unwrap();
        let mut s_run = JsFruchtermanReingoldSolver::new(&graph, &opts).unwrap();
        assert!(s_step.optimal_distance() > 0.0);
        assert!(s_step.current_temperature() > 0.0);
        let mut l_step = init.clone();
        let mut l_run = init.clone();
        while !s_step.is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        s_run.run(&mut l_run).unwrap();
        assert_layouts_bit_exact(&l_step, &l_run);
        assert_eq!(s_run.current_temperature(), 0.0);
    }

    // ForceAtlas2Solver with preventOverlap + nodeRadii
    let radii = vec![0.4; 8];
    let fa2_opts = make_opts(&[
        ("iterations", JsValue::from_f64(10.0)),
        ("preventOverlap", JsValue::from_bool(true)),
        ("defaultNodeRadius", JsValue::from_f64(0.5)),
        ("nodeRadii", f64_js(&radii)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut fa_step = JsForceAtlas2Solver::new(&graph, &fa2_opts).unwrap();
    let mut fa_run = JsForceAtlas2Solver::new(&graph, &fa2_opts).unwrap();
    assert_eq!(fa_step.node_radii(), radii);
    assert!(fa_step.max_displacement() > 0.0);
    let mut l_fa_step = init.clone();
    let mut l_fa_run = init;
    while !fa_step.is_finished() {
        fa_step.step(&mut l_fa_step).unwrap();
    }
    fa_run.run(&mut l_fa_run).unwrap();
    assert_layouts_bit_exact(&l_fa_step, &l_fa_run);

    let bad_fa2 = make_opts(&[
        ("preventOverlap", JsValue::from_bool(true)),
        ("nodeRadii", f64_js(&[1.0, 2.0])),
    ]);
    assert!(JsForceAtlas2Solver::new(&graph, &bad_fa2).is_err());
}

#[wasm_bindgen_test]
fn w1_3_backbone_extraction_and_solver_step_eq_run() {
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
fn w1_4_cluster_refine_treemap_and_group_in_a_box() {
    let (graph, labels) = make_two_cluster_graph();
    let labels_js = u32_js(&labels);

    // Swap two points so ClusterRefineSolver moves misplaced nodes
    let swapped = JsLayout2d::from_coords(&[
        -5.0, 0.0, -4.5, 0.5, -5.5, 0.5, 5.2, 0.2, // node 3 misplaced into cluster 1
        5.0, 0.0, 5.5, 0.5, 4.5, 0.5, -5.2, 0.2, // node 7 misplaced into cluster 0
    ])
    .unwrap();

    let cr_opts = make_opts(&[
        ("method", JsValue::from_str("kmeans")),
        ("maxRounds", JsValue::from_f64(5.0)),
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
    for chunk in rects.chunks_exact(4) {
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

#[wasm_bindgen_test]
fn w1_5_gtree_and_forbid_overlap_removal_step_eq_run() {
    let init = make_overlapping_layout();
    let radii = vec![0.4_f32; 8];
    let hw = vec![0.4_f32; 8];
    let hh = vec![0.3_f32; 8];

    let gtree_opts = make_opts(&[
        ("maxIterations", JsValue::from_f64(25.0)),
        ("padding", JsValue::from_f64(0.02)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut gt_step = JsGTreeOverlapRemoval::new(&hw, &hh, &gtree_opts).unwrap();
    let mut gt_run = JsGTreeOverlapRemoval::new(&hw, &hh, &gtree_opts).unwrap();
    let mut l_gt_step = init.clone();
    let mut l_gt_run = init.clone();
    while !gt_step.is_finished() {
        gt_step.step(&mut l_gt_step).unwrap();
    }
    gt_run.run(&mut l_gt_run).unwrap();
    assert_layouts_bit_exact(&l_gt_step, &l_gt_run);
    assert_eq!(gt_run.last_overlap_count(), 0);
    assert_eq!(count_rect_overlaps(&l_gt_run, &hw, &hh), 0);

    let mut gt_radii = JsGTreeOverlapRemoval::from_radii(&radii, &gtree_opts).unwrap();
    let mut l_gt_rad = init.clone();
    gt_radii.run(&mut l_gt_rad).unwrap();
    assert_eq!(count_rect_overlaps(&l_gt_rad, &radii, &radii), 0);

    for variant in ["forbid", "forbid_prime", "sparse_forbid"] {
        let f_opts = make_opts(&[
            ("variant", JsValue::from_str(variant)),
            ("maxScaleIters", JsValue::from_f64(20.0)),
            ("sgdIterations", JsValue::from_f64(15.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]);
        let mut fb_step = JsForbidOverlapRemoval::from_radii(&radii, &f_opts).unwrap();
        let mut fb_run = JsForbidOverlapRemoval::from_radii(&radii, &f_opts).unwrap();
        let mut l_fb_step = init.clone();
        let mut l_fb_run = init.clone();
        while !fb_step.is_finished() {
            fb_step.step(&mut l_fb_step).unwrap();
        }
        fb_run.run(&mut l_fb_run).unwrap();
        assert_layouts_bit_exact(&l_fb_step, &l_fb_run);
        assert_eq!(fb_run.last_overlap_count(), 0);
        assert_eq!(count_rect_overlaps(&l_fb_run, &radii, &radii), 0);
    }
}

#[wasm_bindgen_test]
fn w1_6_cluster_baseline_functions_and_evaluate_metrics_targets() {
    let (graph, labels) = make_two_cluster_graph();
    let labels_js = u32_js(&labels);
    let sep = make_separated_cluster_layout();
    let overlap = make_overlapping_layout();

    let pa = js_partition_agreement(&labels_js, &labels_js).unwrap();
    assert!((pa.ari() - 1.0).abs() < 1e-12);
    assert!((pa.ami() - 1.0).abs() < 1e-12);
    assert!((pa.v_measure() - 1.0).abs() < 1e-12);
    assert!((pa.fmi() - 1.0).abs() < 1e-12);
    let pa_rec = pa.to_record().unwrap();
    assert_eq!(
        js_sys::Reflect::get(&pa_rec, &JsValue::from_str("vMeasure"))
            .unwrap()
            .as_f64(),
        Some(pa.v_measure())
    );

    let cqk = js_cq_kmeans(&sep, &labels_js, Some(5.0), Some(50.0), Some(42.0)).unwrap();
    assert!((cqk.ari() - 1.0).abs() < 1e-12);
    let cqh = js_cq_hac(&sep, &labels_js).unwrap();
    assert!((cqh.homogeneity() - 1.0).abs() < 1e-12);

    assert!((js_distance_consistency(&sep, &labels_js).unwrap() - 1.0).abs() < 1e-12);
    assert!(js_distribution_consistency(&sep, &labels_js, Some(16.0)).unwrap() > 0.9);
    let cd = js_class_density_measure(&sep, &labels_js, Some(32.0), None).unwrap();
    assert!(cd.cdm() > 0.8 && cd.hdm() > 0.8);
    let cd_rec = cd.to_record().unwrap();
    assert_eq!(
        js_sys::Reflect::get(&cd_rec, &JsValue::from_str("cdm"))
            .unwrap()
            .as_f64(),
        Some(cd.cdm())
    );

    assert!((js_sepme_knng(&sep, &labels_js, Some(2.0)).unwrap() - 1.0).abs() < 1e-12);
    let rng_score = js_sepme_rng(&sep, &labels_js).unwrap();
    let gab_score = js_sepme_gab(&sep, &labels_js).unwrap();
    let gong_score = js_sepme_gong(&sep, &labels_js, Some(0.35)).unwrap();
    assert!(rng_score > 0.5 && rng_score <= 1.0);
    assert!(gab_score > 0.5 && gab_score <= 1.0);
    assert!(gong_score > 0.5 && gong_score <= 1.0);
    assert!((js_neighborhood_hit(&sep, &labels_js, Some(3.0)).unwrap() - 1.0).abs() < 1e-12);
    assert!(js_silhouette_score(&sep, &labels_js).unwrap() > 0.8);
    assert!(js_calinski_harabasz_score(&sep, &labels_js).unwrap() > 50.0);
    assert!(js_davies_bouldin_score(&sep, &labels_js).unwrap() < 0.3);
    assert!(js_pairwise_distance_overlap_error(&sep, &labels_js, Some(32.0)).unwrap() < 0.1);

    let hw = vec![0.4_f32; 8];
    let hh = vec![0.4_f32; 8];
    let sp1 = js_sprawlter_nn(&overlap, &hw, &hh).unwrap();
    let sp2 = js_sprawlter_nn_radii(&overlap, &hw).unwrap();
    assert!((sp1 - sp2).abs() < 1e-12);
    assert!(sp1 > 0.0);

    let nor = js_nor_metrics(&graph, &overlap, &sep, &hw, &hh).unwrap();
    assert!(nor.oo_nni().is_finite() && nor.el_rsd().is_finite());
    let nor_no_g = js_nor_metrics_without_graph(&overlap, &sep, &hw, &hh).unwrap();
    assert_eq!(nor_no_g.el_rsd(), 0.0);
    let nor_rec = nor.to_record().unwrap();
    assert_eq!(
        js_sys::Reflect::get(&nor_rec, &JsValue::from_str("ooNni"))
            .unwrap()
            .as_f64(),
        Some(nor.oo_nni())
    );

    // evaluateMetrics with all 18 W1-6 targets via options object & sepme-knng2 alias
    let targets = js_sys::Array::new();
    for t in [
        "dsc",
        "dc",
        "cdm",
        "hdm",
        "sepme-knng2",
        "sepme_rng",
        "sepme_gab",
        "sepme_gong",
        "neighborhood_hit",
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
        "cq_kmeans_ari",
        "cq_kmeans_ami",
        "cq_hac_hom",
        "cq_hac_ami",
        "pairwise_distance_overlap_error",
        "sprawlter_nn",
    ] {
        targets.push(&JsValue::from_str(t));
    }
    let eval_opts = make_opts(&[
        ("communityLabels", labels_js.clone()),
        ("sepmeKnngK", JsValue::from_f64(2.0)),
        ("neighborhoodHitK", JsValue::from_f64(3.0)),
        ("dcGridSize", JsValue::from_f64(16.0)),
        ("cdmGridSize", JsValue::from_f64(32.0)),
        ("numBins", JsValue::from_f64(32.0)),
        ("gongGamma", JsValue::from_f64(0.35)),
        ("kmeansNInit", JsValue::from_f64(3.0)),
        ("kmeansMaxIters", JsValue::from_f64(50.0)),
        ("sprawlterNodeRadius", JsValue::from_f64(0.5)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let rep = js_evaluate_metrics_entry(
        &graph,
        &sep,
        &targets,
        &eval_opts,
        None,
        None,
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
        &JsValue::UNDEFINED,
        &JsValue::UNDEFINED,
        &JsValue::UNDEFINED,
    )
    .unwrap();
    assert!(rep.dsc().is_some());
    assert!(rep.dc().is_some());
    assert!(rep.cdm().is_some());
    assert!(rep.hdm().is_some());
    assert!(rep.sepme_knng().is_some());
    assert!(rep.sepme_rng().is_some());
    assert!(rep.sepme_gab().is_some());
    assert!(rep.sepme_gong().is_some());
    assert!(rep.neighborhood_hit().is_some());
    assert!(rep.silhouette().is_some());
    assert!(rep.calinski_harabasz().is_some());
    assert!(rep.davies_bouldin().is_some());
    assert!(rep.cq_kmeans_ari().is_some());
    assert!(rep.cq_kmeans_ami().is_some());
    assert!(rep.cq_hac_hom().is_some());
    assert!(rep.cq_hac_ami().is_some());
    assert!(rep.pairwise_distance_overlap_error().is_some());
    assert!(rep.sprawlter_nn().is_some());

    let rep_rec = rep.to_record().unwrap();
    assert_eq!(
        js_sys::Reflect::get(&rep_rec, &JsValue::from_str("sepmeKnng"))
            .unwrap()
            .as_f64(),
        rep.sepme_knng()
    );

    // 23rd-arg positional clusterBaselineOptions compatibility check
    let cb_opts = make_opts(&[("sepmeKnngK", JsValue::from_f64(3.0))]);
    let knng_target = js_sys::Array::of1(&JsValue::from_str("sepme_knng2"));
    let rep_pos = js_evaluate_metrics_entry(
        &graph,
        &sep,
        &knng_target,
        &JsValue::UNDEFINED,
        None,
        Some(42.0),
        None,
        &labels_js,
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
        &JsValue::UNDEFINED,
        &cb_opts,
    )
    .unwrap();
    assert!(
        (rep_pos.sepme_knng().unwrap() - js_sepme_knng(&sep, &labels_js, Some(3.0)).unwrap()).abs()
            < 1e-12
    );

    // Lazy validation & error path: `sns` without `communityLabels` succeeds; `dsc` without `communityLabels` fails.
    let sns_only = js_sys::Array::of1(&JsValue::from_str("sns"));
    let sns_rep = js_evaluate_metrics_entry(
        &graph,
        &sep,
        &sns_only,
        &JsValue::UNDEFINED,
        None,
        None,
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
        &JsValue::UNDEFINED,
        &JsValue::UNDEFINED,
        &JsValue::UNDEFINED,
    )
    .unwrap();
    assert!(sns_rep.scale_normalized_stress().is_some());

    let dsc_only = js_sys::Array::of1(&JsValue::from_str("dsc"));
    assert!(
        js_evaluate_metrics_entry(
            &graph,
            &sep,
            &dsc_only,
            &JsValue::UNDEFINED,
            None,
            None,
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
            &JsValue::UNDEFINED,
            &JsValue::UNDEFINED,
            &JsValue::UNDEFINED,
        )
        .is_err()
    );

    // Degenerate N=0 / N=1 non-panic checks
    let l0 = JsLayout2d::from_coords(&[]).unwrap();
    let empty_labels = u32_js(&[]);
    assert_eq!(js_distance_consistency(&l0, &empty_labels).unwrap(), 0.0);
    assert_eq!(js_silhouette_score(&l0, &empty_labels).unwrap(), 0.0);
    let mut gt0 = JsGTreeOverlapRemoval::new(&[], &[], &JsValue::UNDEFINED).unwrap();
    let st0 = gt0.run(&mut l0.clone()).unwrap();
    assert!(matches!(st0.as_str(), "converged" | "completed"));
}
