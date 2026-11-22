use egraph_wasm::v6::{
    JsCsrGraph, JsFdebSolver, JsForceAtlas2Solver, JsIbUmapSolver, JsLayout2d,
    JsPrismOverlapRemoval, JsSeparationConstraint, JsSgdSolver, JsTFdpSolver, JsYifanHuSolver,
    js_assign_layers, js_build_pure_2hop_knn, js_compute_fuzzy_simplicial_set, js_cycle_edges,
    js_generate_layered_constraints, js_pivot_mds_init, js_project_1d,
    js_project_clustered_rectangle_no_overlap_2d, js_project_layout_axis,
    js_project_rectangle_no_overlap_2d, js_remove_cycles,
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
fn t_fdp_solver_all_backends_from_knn_and_step_eq_run() {
    let grid = make_grid(4, 4);
    let init = make_circle_layout(16, 1.0);

    for backend in ["ibfft", "barnes_hut", "exact"] {
        let opts = make_opts(&[
            ("backend", JsValue::from_str(backend)),
            ("totalIterations", JsValue::from_f64(12.0)),
            ("learningRate", JsValue::from_f64(0.5)),
            ("momentum", JsValue::from_f64(0.75)),
            ("rhoStart", JsValue::from_f64(1.2)),
            ("rhoEnd", JsValue::from_f64(0.2)),
            ("rhoPower", JsValue::from_f64(1.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]);

        let mut s_run = JsTFdpSolver::new(&grid, &opts).expect("TFdpSolver::new");
        let mut s_step = JsTFdpSolver::new(&grid, &opts).expect("TFdpSolver::new step");
        assert_eq!(s_run.status(), "initialized");
        assert_eq!(s_run.num_nodes(), 16);
        assert_eq!(s_run.total_iterations(), 12);
        assert!(s_run.num_edges() > 0);
        assert!((s_run.current_rho() - 1.2).abs() < 1e-12);

        let mut l_run = init.clone();
        let mut l_step = init.clone();
        let ptr_before = l_step.coords_ptr();

        assert_eq!(s_run.run(&mut l_run).unwrap(), "completed");
        assert!(s_run.is_finished());
        assert!((s_run.current_rho() - 0.2).abs() < 1e-12);
        assert!(l_run.is_all_finite());

        assert_eq!(s_step.run_steps(&mut l_step, 5.0).unwrap(), "running");
        assert_eq!(s_step.iteration(), 5);
        while !s_step.is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        assert_eq!(l_step.coords_ptr(), ptr_before);
        assert_eq!(s_step.status(), "completed");
        assert_layouts_bit_exact(&l_run, &l_step);
        if backend == "ibfft" {
            assert!(s_run.total_fft_2d_calls() > 0.0);
        } else {
            assert_eq!(s_run.total_fft_2d_calls(), 0.0);
        }
    }

    // TFdpSolver.fromKnn with raw CSR arrays (both omitted weights and explicit Float64Array)
    let indptr = [0u32, 1, 3, 4];
    let indices = [1u32, 0, 2, 1];
    let knn_opts = make_opts(&[
        ("backend", JsValue::from_str("exact")),
        ("totalIterations", JsValue::from_f64(8.0)),
    ]);
    let indptr_js: JsValue = js_sys::Uint32Array::from(&indptr[..]).into();
    let indices_js: JsValue = js_sys::Uint32Array::from(&indices[..]).into();
    let mut k_default =
        JsTFdpSolver::from_knn(&indptr_js, &indices_js, &JsValue::UNDEFINED, &knn_opts).unwrap();
    let mut k_opts_short =
        JsTFdpSolver::from_knn(&indptr_js, &indices_js, &knn_opts, &JsValue::UNDEFINED).unwrap();
    let weights_arr = js_sys::Float64Array::from(&[0.5_f64, 0.5, 1.0, 1.0][..]);
    let mut k_weighted =
        JsTFdpSolver::from_knn(&indptr_js, &indices_js, &weights_arr, &knn_opts).unwrap();
    let mut l3_a = JsLayout2d::from_coords(&[0.0, 0.0, 1.0, 0.0, 2.0, 0.0]).unwrap();
    let mut l3_b = l3_a.clone();
    let mut l3_c = l3_a.clone();
    assert_eq!(k_default.run(&mut l3_a).unwrap(), "completed");
    assert_eq!(k_opts_short.run(&mut l3_b).unwrap(), "completed");
    assert_layouts_bit_exact(&l3_a, &l3_b);
    assert_eq!(k_weighted.run(&mut l3_c).unwrap(), "completed");
    assert!(l3_c.is_all_finite());

    // Adversarial checks
    assert!(
        JsTFdpSolver::new(
            &grid,
            &make_opts(&[("backend", JsValue::from_str("unknown"))])
        )
        .is_err()
    );
    assert!(JsTFdpSolver::new(&grid, &JsValue::from_str("bad")).is_err());
    assert!(JsTFdpSolver::new(&grid, &make_opts(&[("alpha", JsValue::from_f64(0.0))])).is_err());
    let empty_u32: JsValue = js_sys::Uint32Array::new_with_length(0).into();
    assert!(
        JsTFdpSolver::from_knn(
            &empty_u32,
            &empty_u32,
            &JsValue::UNDEFINED,
            &JsValue::UNDEFINED
        )
        .is_err()
    );
    let mut bad_layout = JsLayout2d::zeros(2.0).unwrap();
    assert!(k_default.step(&mut bad_layout).is_err());
}

#[wasm_bindgen_test]
fn ib_umap_solver_all_constructors_and_step_eq_run() {
    let grid = make_grid(4, 4);
    let init = make_circle_layout(16, 1.0);

    for backend in ["ibfft", "barnes_hut", "exact"] {
        let opts = make_opts(&[
            ("backend", JsValue::from_str(backend)),
            ("max2hopPerNode", JsValue::from_f64(16.0)),
            ("nEpochs", JsValue::from_f64(10.0)),
            ("cK", JsValue::from_f64(4.0)),
            ("tauScale", JsValue::from_f64(4.0)),
            ("hubDamping", JsValue::from_bool(true)),
            ("degree", JsValue::from_f64(6.0)),
            ("numLandmarks", JsValue::from_f64(8.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]);

        let mut s_run = JsIbUmapSolver::new(&grid, &opts).expect("IbUmapSolver::new");
        let mut s_step = JsIbUmapSolver::new(&grid, &opts).expect("IbUmapSolver::new step");
        assert_eq!(s_run.num_nodes(), 16);
        assert_eq!(s_run.total_iterations(), 10);
        assert_eq!(s_run.current_rho(), 1.0);
        assert!(s_run.a() > 0.0 && s_run.b() > 0.0);
        assert!(s_run.num_edges() > 0);

        let mut l_run = init.clone();
        let mut l_step = init.clone();
        let ptr_before = l_step.coords_ptr();

        assert_eq!(s_run.run(&mut l_run).unwrap(), "completed");
        assert!(l_run.is_all_finite());
        assert_eq!(s_step.run_steps(&mut l_step, 4.0).unwrap(), "running");
        while !s_step.is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        assert_eq!(l_step.coords_ptr(), ptr_before);
        assert_layouts_bit_exact(&l_run, &l_step);
    }

    // fromKnn and fromFuzzySet (with and without explicit degrees)
    let knn = js_build_pure_2hop_knn(&grid).unwrap().knn();
    let fuzzy = js_compute_fuzzy_simplicial_set(&knn, None, None).unwrap();
    let opts = make_opts(&[
        ("backend", JsValue::from_str("barnes_hut")),
        ("nEpochs", JsValue::from_f64(6.0)),
    ]);
    let mut s_knn = JsIbUmapSolver::from_knn(&knn, &opts).unwrap();
    let mut s_fuz_a = JsIbUmapSolver::from_fuzzy_set(&fuzzy, &opts, &JsValue::UNDEFINED).unwrap();
    let degs = js_sys::Uint32Array::from(&vec![3u32; 16][..]);
    let mut s_fuz_b = JsIbUmapSolver::from_fuzzy_set(&fuzzy, &degs, &opts).unwrap();

    let mut l_a = init.clone();
    let mut l_b = init.clone();
    let mut l_c = init;
    assert_eq!(s_knn.run(&mut l_a).unwrap(), "completed");
    assert_eq!(s_fuz_a.run(&mut l_b).unwrap(), "completed");
    assert_eq!(s_fuz_b.run(&mut l_c).unwrap(), "completed");
    assert_layouts_bit_exact(&l_a, &l_b);
    assert!(l_c.is_all_finite());

    // Adversarial checks
    assert!(JsIbUmapSolver::new(&grid, &make_opts(&[("cK", JsValue::from_f64(-1.0))])).is_err());
    assert!(JsIbUmapSolver::from_fuzzy_set(&fuzzy, &opts, &opts).is_err());
}

#[wasm_bindgen_test]
fn sgd_m4_constructors_snap_t_fdp_pacmap_sgd2_and_squared_schedule() {
    let grid = make_grid(4, 4);
    let init = js_pivot_mds_init(&grid, None, None, None, &JsValue::UNDEFINED).unwrap();

    // 1. SgdSolver.snapTFdp
    let snap_opts = make_opts(&[
        ("iterations", JsValue::from_f64(15.0)),
        ("negativesPerEdge", JsValue::from_f64(3.0)),
        ("learningRate", JsValue::from_f64(0.5)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut snap_run = JsSgdSolver::snap_t_fdp(&grid, &snap_opts).unwrap();
    let mut snap_step = JsSgdSolver::snap_t_fdp(&grid, &snap_opts).unwrap();
    assert_eq!(snap_run.num_springs(), grid.num_edges());
    assert_eq!(snap_run.total_iterations(), 15);
    assert!(snap_run.current_eta() > 0.0);

    let mut l_snap_run = init.clone();
    let mut l_snap_step = init.clone();
    assert_eq!(snap_run.run(&mut l_snap_run).unwrap(), "completed");
    assert_eq!(snap_run.current_eta(), 0.0);
    assert_eq!(
        snap_step.run_steps(&mut l_snap_step, 5.0).unwrap(),
        "running"
    );
    while !snap_step.is_finished() {
        snap_step.step(&mut l_snap_step).unwrap();
    }
    assert_layouts_bit_exact(&l_snap_run, &l_snap_step);

    // 2. SgdSolver.pacmapGraph
    let pacmap_opts = make_opts(&[
        ("nNeighbors", JsValue::from_f64(6.0)),
        ("mnRatio", JsValue::from_f64(0.5)),
        ("fpRatio", JsValue::from_f64(1.5)),
        ("phase1Iters", JsValue::from_f64(5.0)),
        ("phase2Iters", JsValue::from_f64(5.0)),
        ("phase3Iters", JsValue::from_f64(5.0)),
        ("learningRate", JsValue::from_f64(0.2)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut pm_run = JsSgdSolver::pacmap_graph(&grid, &pacmap_opts).unwrap();
    let mut pm_step = JsSgdSolver::pacmap_graph(&grid, &pacmap_opts).unwrap();
    assert_eq!(pm_run.total_iterations(), 15);
    assert!(pm_run.num_springs() > 0);
    assert!(pm_run.num_hinges() > 0);

    let mut l_pm_run = init.clone();
    let mut l_pm_step = init.clone();
    assert_eq!(pm_run.run(&mut l_pm_run).unwrap(), "completed");
    assert_eq!(pm_step.run_steps(&mut l_pm_step, 6.0).unwrap(), "running");
    while !pm_step.is_finished() {
        pm_step.step(&mut l_pm_step).unwrap();
    }
    assert_layouts_bit_exact(&l_pm_run, &l_pm_step);

    // 3. SgdSolver.sgd2 + "squared" schedule kind on SgdSolver.full
    let sgd2_opts = make_opts(&[
        ("tMax", JsValue::from_f64(12.0)),
        ("batchSize", JsValue::from_f64(16.0)),
        ("damping", JsValue::from_f64(1e-4)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut s2_run = JsSgdSolver::sgd2(&grid, &sgd2_opts).unwrap();
    let mut s2_step = JsSgdSolver::sgd2(&grid, &sgd2_opts).unwrap();
    let mut l_s2_run = init.clone();
    let mut l_s2_step = init.clone();
    assert_eq!(s2_run.run(&mut l_s2_run).unwrap(), "completed");
    assert_eq!(s2_step.run_steps(&mut l_s2_step, 4.0).unwrap(), "running");
    while !s2_step.is_finished() {
        s2_step.step(&mut l_s2_step).unwrap();
    }
    assert_layouts_bit_exact(&l_s2_run, &l_s2_step);

    let sq_opts = make_opts(&[
        ("tMax", JsValue::from_f64(10.0)),
        ("kind", JsValue::from_str("squared")),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut sq_run = JsSgdSolver::full(&grid, &sq_opts, None, None, None, None).unwrap();
    let mut sq_step = JsSgdSolver::full(&grid, &sq_opts, None, None, None, None).unwrap();
    let mut l_sq_run = init.clone();
    let mut l_sq_step = init;
    assert_eq!(sq_run.run(&mut l_sq_run).unwrap(), "completed");
    sq_step.run_steps(&mut l_sq_step, 3.0).unwrap();
    while !sq_step.is_finished() {
        sq_step.step(&mut l_sq_step).unwrap();
    }
    assert_layouts_bit_exact(&l_sq_run, &l_sq_step);

    // Adversarial checks
    assert!(
        JsSgdSolver::pacmap_graph(&grid, &make_opts(&[("nNeighbors", JsValue::from_f64(0.0))]))
            .is_err()
    );
    assert!(
        JsSgdSolver::sgd2(&grid, &make_opts(&[("batchSize", JsValue::from_f64(0.0))])).is_err()
    );
    assert!(JsSgdSolver::snap_t_fdp(&grid, &JsValue::from_f64(123.0)).is_err());
}

#[wasm_bindgen_test]
fn force_atlas2_and_yifan_hu_solvers() {
    let barbell = make_barbell(6);
    let init_b = make_circle_layout(12, 1.0);

    // 1. ForceAtlas2Solver (Barnes-Hut + Exact, LinLog + StrongGravity)
    for use_bh in [true, false] {
        let fa2_opts = make_opts(&[
            ("iterations", JsValue::from_f64(15.0)),
            ("useBarnesHut", JsValue::from_bool(use_bh)),
            ("linlogMode", JsValue::from_bool(!use_bh)),
            ("strongGravity", JsValue::from_bool(!use_bh)),
            ("outboundAttractionDistribution", JsValue::from_bool(use_bh)),
            ("seed", JsValue::from_f64(42.0)),
        ]);
        let mut fa2_run = JsForceAtlas2Solver::new(&barbell, &fa2_opts).unwrap();
        let mut fa2_step = JsForceAtlas2Solver::new(&barbell, &fa2_opts).unwrap();
        let mut l_run = init_b.clone();
        let mut l_step = init_b.clone();
        let ptr_before = l_step.coords_ptr();

        assert_eq!(fa2_run.run(&mut l_run).unwrap(), "completed");
        assert!(l_run.is_all_finite());
        assert_eq!(fa2_step.run_steps(&mut l_step, 5.0).unwrap(), "running");
        while !fa2_step.is_finished() {
            fa2_step.step(&mut l_step).unwrap();
        }
        assert_eq!(l_step.coords_ptr(), ptr_before);
        assert_layouts_bit_exact(&l_run, &l_step);
    }

    // 2. YifanHuSolver (multilevel EC + MIVS hierarchy, numLevels >= 2, currentLevel -> 0)
    let grid6 = make_grid(6, 6);
    let init36 = make_circle_layout(36, 1.0);
    let yh_opts = make_opts(&[
        ("maxIterationsPerLevel", JsValue::from_f64(12.0)),
        ("minCoarsenSize", JsValue::from_f64(8.0)),
        ("minCoarsenRatio", JsValue::from_f64(0.85)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut yh_run = JsYifanHuSolver::new(&grid6, &yh_opts).unwrap();
    let mut yh_step = JsYifanHuSolver::new(&grid6, &yh_opts).unwrap();
    assert!(yh_run.num_levels() >= 2);
    assert_eq!(yh_run.current_level(), yh_run.num_levels() - 1);

    let mut l_yh_run = init36.clone();
    let mut l_yh_step = init36;
    let ptr_before = l_yh_step.coords_ptr();

    let st = yh_run.run(&mut l_yh_run).unwrap();
    assert!(st == "completed" || st == "converged");
    assert_eq!(yh_run.current_level(), 0);
    assert!(l_yh_run.is_all_finite());

    yh_step.run_steps(&mut l_yh_step, 4.0).unwrap();
    while !yh_step.is_finished() {
        yh_step.step(&mut l_yh_step).unwrap();
    }
    assert_eq!(l_yh_step.coords_ptr(), ptr_before);
    assert_eq!(yh_step.current_level(), 0);
    assert_layouts_bit_exact(&l_yh_run, &l_yh_step);

    // Adversarial checks
    assert!(
        JsForceAtlas2Solver::new(&barbell, &make_opts(&[("theta", JsValue::from_f64(-0.1))]))
            .is_err()
    );
    assert!(
        JsYifanHuSolver::new(&grid6, &make_opts(&[("stepRatio", JsValue::from_f64(1.0))])).is_err()
    );
    assert!(
        JsYifanHuSolver::new(
            &grid6,
            &make_opts(&[("minCoarsenSize", JsValue::from_f64(0.0))])
        )
        .is_err()
    );
}

#[wasm_bindgen_test]
fn prism_overlap_removal_zero_overlaps_and_step_eq_run() {
    let hw = vec![0.6f32; 9];
    let hh = vec![0.4f32; 9];
    let mut init_coords = Vec::new();
    for r in 0..3 {
        for c in 0..3 {
            init_coords.push((c as f32) * 0.3);
            init_coords.push((r as f32) * 0.25);
        }
    }
    let init = JsLayout2d::from_coords(&init_coords).unwrap();
    assert!(count_rect_overlaps(&init, &hw, &hh) > 0);

    let opts = make_opts(&[
        ("maxIterations", JsValue::from_f64(50.0)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut p_run = JsPrismOverlapRemoval::new(&hw, &hh, &opts).unwrap();
    let mut p_step = JsPrismOverlapRemoval::new(&hw, &hh, &opts).unwrap();

    let mut l_run = init.clone();
    let mut l_step = init;
    assert_eq!(p_run.run(&mut l_run).unwrap(), "converged");
    assert_eq!(p_run.last_overlap_count(), 0);
    assert_eq!(count_rect_overlaps(&l_run, &hw, &hh), 0);

    p_step.step(&mut l_step).unwrap();
    while !p_step.is_finished() {
        p_step.step(&mut l_step).unwrap();
    }
    assert_eq!(p_step.last_overlap_count(), 0);
    assert_layouts_bit_exact(&l_run, &l_step);

    // fromRadii on already separated boxes converges immediately
    let radii = vec![0.25f32; 4];
    let mut sep_layout =
        JsLayout2d::from_coords(&[0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0]).unwrap();
    let mut p_rad = JsPrismOverlapRemoval::from_radii(&radii, &JsValue::UNDEFINED).unwrap();
    assert_eq!(p_rad.run(&mut sep_layout).unwrap(), "converged");
    assert_eq!(p_rad.last_overlap_count(), 0);

    // Adversarial checks
    assert!(JsPrismOverlapRemoval::new(&[0.5, 0.5], &[0.5], &JsValue::UNDEFINED).is_err());
    assert!(JsPrismOverlapRemoval::from_radii(&[-0.1, 0.5], &JsValue::UNDEFINED).is_err());
    assert!(
        JsPrismOverlapRemoval::from_radii(
            &radii,
            &make_opts(&[("maxScaleFactor", JsValue::from_f64(1.0))])
        )
        .is_err()
    );
}

#[wasm_bindgen_test]
fn separation_constraints_and_rectangle_projection() {
    // 1. SeparationConstraint + project1d
    let c0 = JsSeparationConstraint::new(0.0, 1.0, 2.0).unwrap();
    let c1 = JsSeparationConstraint::new(1.0, 2.0, 1.5).unwrap();
    assert_eq!(c0.left(), 0);
    assert_eq!(c0.right(), 1);
    assert_eq!(c0.gap(), 2.0);

    let arr = js_sys::Array::new();
    arr.push(&JsValue::from(c0));
    arr.push(&JsValue::from(c1));

    let mut coords = [3.0f32, 1.0, 0.0];
    let projected = js_project_1d(&mut coords, &arr).unwrap();
    assert_eq!(projected, coords.to_vec());
    assert!(coords[1] - coords[0] >= 2.0 - 1e-5);
    assert!(coords[2] - coords[1] >= 1.5 - 1e-5);

    // 2. projectLayoutAxis + generateLayeredConstraints
    let dag = JsCsrGraph::from_edge_list(
        4.0,
        &[0, 1, 0, 2, 1, 3, 2, 3],
        None,
        &JsValue::from_bool(true),
        None,
    )
    .unwrap();
    let layers = js_assign_layers(&dag, Some("longestPath".to_owned())).unwrap();
    let layered_cs = js_generate_layered_constraints(&dag, &layers, Some(3.0)).unwrap();
    let js_cs = js_sys::Array::new();
    for c in layered_cs {
        js_cs.push(&JsValue::from(c));
    }
    let mut layout = JsLayout2d::zeros(4.0).unwrap();
    js_project_layout_axis(&mut layout, &JsValue::from_str("y"), &js_cs).unwrap();
    assert!(layout.y(1.0).unwrap() - layout.y(0.0).unwrap() >= 3.0 - 1e-5);
    assert!(layout.y(3.0).unwrap() - layout.y(1.0).unwrap() >= 3.0 - 1e-5);

    // 3. projectRectangleNoOverlap2d
    let hw = [0.6f32; 4];
    let hh = [0.6f32; 4];
    let mut rect_l = JsLayout2d::from_coords(&[0.0, 0.0, 0.2, 0.1, 0.1, 0.2, -0.1, 0.0]).unwrap();
    js_project_rectangle_no_overlap_2d(&mut rect_l, &hw, &hh).unwrap();
    assert_eq!(count_rect_overlaps(&rect_l, &hw, &hh), 0);

    // 4. projectClusteredRectangleNoOverlap2d rigidly preserves intra-cluster deltas
    let mut cl_l = JsLayout2d::from_coords(&[0.0, 0.0, 0.2, 0.1, 0.1, 0.0, 0.3, 0.1]).unwrap();
    let dx0_before = cl_l.x(1.0).unwrap() - cl_l.x(0.0).unwrap();
    let dy0_before = cl_l.y(1.0).unwrap() - cl_l.y(0.0).unwrap();
    let cluster_ids = [0u32, 0, 1, 1];
    let c_hw = [0.5f32; 4];
    let c_hh = [0.5f32; 4];
    js_project_clustered_rectangle_no_overlap_2d(&mut cl_l, &cluster_ids, &c_hw, &c_hh).unwrap();
    let dx0_after = cl_l.x(1.0).unwrap() - cl_l.x(0.0).unwrap();
    let dy0_after = cl_l.y(1.0).unwrap() - cl_l.y(0.0).unwrap();
    assert!((dx0_after - dx0_before).abs() < 1e-5);
    assert!((dy0_after - dy0_before).abs() < 1e-5);

    // Adversarial checks
    let cyc_cs = js_sys::Array::new();
    cyc_cs.push(&JsValue::from(
        JsSeparationConstraint::new(0.0, 1.0, 1.0).unwrap(),
    ));
    cyc_cs.push(&JsValue::from(
        JsSeparationConstraint::new(1.0, 0.0, 1.0).unwrap(),
    ));
    let mut bad_c = [0.0f32, 0.0];
    assert!(js_project_1d(&mut bad_c, &cyc_cs).is_err());
    assert!(js_project_layout_axis(&mut layout, &JsValue::from_f64(2.0), &js_cs).is_err());
    assert!(JsSeparationConstraint::new(0.0, 1.0, f32::NAN).is_err());
}

#[wasm_bindgen_test]
fn layering_cycle_edges_remove_cycles_and_assign_layers() {
    // Cyclic graph: 0 -> 1 -> 2 -> 0, plus 2 -> 3
    let cyc = JsCsrGraph::from_edge_list(
        4.0,
        &[0, 1, 1, 2, 2, 0, 2, 3],
        None,
        &JsValue::from_bool(true),
        None,
    )
    .unwrap();
    let back = js_cycle_edges(&cyc).unwrap();
    assert_eq!(back, vec![2, 0]);

    let dag = js_remove_cycles(&cyc).unwrap();
    assert!(dag.is_directed());
    assert!(js_cycle_edges(&dag).unwrap().is_empty());

    // Multi-span DAG: 0 -> 1 -> 2 -> 3 and 0 -> 3; plus 4 -> 3 (network simplex pulls 4 to layer 2)
    let span_dag = JsCsrGraph::from_edge_list(
        5.0,
        &[0, 1, 1, 2, 2, 3, 0, 3, 4, 3],
        None,
        &JsValue::from_bool(true),
        None,
    )
    .unwrap();
    let lp = js_assign_layers(&span_dag, Some("longestPath".to_owned())).unwrap();
    let ns = js_assign_layers(&span_dag, Some("network_simplex".to_owned())).unwrap();
    assert_eq!(lp.len(), 5);
    assert_eq!(ns.len(), 5);
    assert_eq!(*ns.iter().min().unwrap(), 0);
    assert_eq!(lp[4], 0);
    assert_eq!(ns[4], 2);

    // Adversarial: undirected graph rejected and bogus method rejected
    let undir = make_grid(2, 2);
    assert!(js_cycle_edges(&undir).is_err());
    assert!(js_remove_cycles(&undir).is_err());
    assert!(js_assign_layers(&undir, None).is_err());
    assert!(js_assign_layers(&span_dag, Some("badMethod".to_owned())).is_err());
}

#[wasm_bindgen_test]
fn fdeb_solver_reference_passing_and_step_eq_run() {
    let graph =
        JsCsrGraph::from_edge_list(4.0, &[0, 1, 2, 3], None, &JsValue::from_bool(false), None)
            .unwrap();
    // Two parallel horizontal edges separated by y=1.0 -> high compatibility
    let layout = JsLayout2d::from_coords(&[0.0, 0.0, 10.0, 0.0, 0.0, 1.0, 10.0, 1.0]).unwrap();
    let opts = make_opts(&[
        ("cycles", JsValue::from_f64(3.0)),
        ("i0", JsValue::from_f64(12.0)),
        ("s0", JsValue::from_f64(0.1)),
    ]);

    // Construct twice on the same `layout` handle to verify no value-drop bug
    let mut f_run = JsFdebSolver::new(&graph, &layout, &opts).unwrap();
    let mut f_step = JsFdebSolver::new(&graph, &layout, &opts).unwrap();
    assert_eq!(layout.len(), 4);
    assert_eq!(f_run.num_edges(), 2);

    assert_eq!(f_run.run().unwrap(), "completed");
    assert!(f_run.is_finished());

    assert_eq!(f_step.run_steps(5.0).unwrap(), "running");
    while !f_step.is_finished() {
        f_step.step().unwrap();
    }
    assert_eq!(f_step.status(), "completed");
    assert_eq!(f_run.edge_offsets(), f_step.edge_offsets());
    let pts_run = f_run.points();
    let pts_step = f_step.points();
    assert_eq!(pts_run.len(), pts_step.len());
    for (a, b) in pts_run.iter().zip(pts_step.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    let offsets = f_run.edge_offsets();
    assert_eq!(offsets.len(), 3);
    assert_eq!(pts_run.len(), (offsets[2] as usize) * 2);
    assert!(pts_run.iter().all(|v| v.is_finite()));

    // Midpoint y-coordinates of the two parallel edges should attract toward 0.5
    let mid0 = (offsets[0] as usize + offsets[1] as usize) / 2;
    let mid1 = (offsets[1] as usize + offsets[2] as usize) / 2;
    let mid_gap = (pts_run[mid1 * 2 + 1] - pts_run[mid0 * 2 + 1]).abs();
    assert!(mid_gap < 1.0);

    // Adversarial checks
    assert!(
        JsFdebSolver::new(
            &graph,
            &layout,
            &make_opts(&[("cycles", JsValue::from_f64(30.0))])
        )
        .is_err()
    );
    let short_l = JsLayout2d::zeros(2.0).unwrap();
    assert!(JsFdebSolver::new(&graph, &short_l, &JsValue::UNDEFINED).is_err());
}
