use super::helpers::MAX_WEIGHT_RATIO;
use super::tests_basic::grid_graph;
use super::*;
use egraph_core::IterativeSolver;
use egraph_layout_sgd::{SgdPairBuffer, SgdScheduleConfig, SgdScheduleKind, SgdSolver};

#[test]
fn test_refiner_w_anchor2_and_hub_fallback() {
    // 1. Exact 2-hop path: 3x3 grid with intruder (0, 8).
    let grid = grid_graph(3, 3);
    let mut clean_x = Vec::with_capacity(9);
    let mut clean_y = Vec::with_capacity(9);
    for r in 0..3 {
        for c in 0..3 {
            clean_x.push(c as f64);
            clean_y.push(r as f64);
        }
    }
    let clean_layout = Layout2d::from_xy(clean_x.clone(), clean_y.clone()).unwrap();
    let cfg_a2 = SpatialRefinerConfig::default()
        .with_iterations(2)
        .with_w_anchor2(0.25)
        .with_seed(42);

    // Clean layout must remain untouched even with w_anchor2 > 0.0.
    let mut l_clean = clean_layout.clone();
    let mut r_clean = SpatialIntruderRefiner::<f64>::new(&grid, cfg_a2.clone()).unwrap();
    assert_eq!(r_clean.run(&mut l_clean).unwrap(), SolverStatus::Converged);
    assert_eq!(r_clean.stats().rounds_executed, 0);

    // Intruder layout: w_anchor2 > 0 increases n_anchors relative to w_anchor2 == 0,
    // and step == run is bit-exact in f64 and f32.
    let mut intr_x = clean_x;
    let mut intr_y = clean_y;
    intr_x[8] = 0.1;
    intr_y[8] = 0.1;
    let intr_init = Layout2d::from_xy(intr_x.clone(), intr_y.clone()).unwrap();

    let cfg_base = cfg_a2.clone().with_w_anchor2(0.0);
    let mut r_base = SpatialIntruderRefiner::<f64>::new(&grid, cfg_base).unwrap();
    let mut l_base = intr_init.clone();
    r_base.run(&mut l_base).unwrap();

    let mut r_run = SpatialIntruderRefiner::<f64>::new(&grid, cfg_a2.clone()).unwrap();
    let mut r_step = SpatialIntruderRefiner::<f64>::new(&grid, cfg_a2.clone()).unwrap();
    let mut l_run = intr_init.clone();
    let mut l_step = intr_init;
    r_run.run(&mut l_run).unwrap();
    while !r_step.status().is_finished() {
        r_step.step(&mut l_step).unwrap();
    }
    assert!(r_run.stats().n_anchors_total > r_base.stats().n_anchors_total);
    for i in 0..9 {
        assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
        assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
    }

    // f32 bit-exact parity
    let intr32 = Layout2d::from_xy(
        intr_x.iter().map(|&v| v as f32).collect(),
        intr_y.iter().map(|&v| v as f32).collect(),
    )
    .unwrap();
    let mut r32_run = SpatialIntruderRefiner::<f32>::new(&grid, cfg_a2.clone()).unwrap();
    let mut r32_step = SpatialIntruderRefiner::<f32>::new(&grid, cfg_a2).unwrap();
    let mut l32_run = intr32.clone();
    let mut l32_step = intr32;
    r32_run.run(&mut l32_run).unwrap();
    while !r32_step.status().is_finished() {
        r32_step.step(&mut l32_step).unwrap();
    }
    for i in 0..9 {
        assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
        assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
    }

    // 2. Hub fallback path with w_anchor2 > 0.0 and w_fn == 0.0 must populate u2_hub/v2_hub.
    let n_star = 400usize;
    let mut edges: Vec<(u32, u32)> = (1..n_star as u32).map(|v| (0, v)).collect();
    edges.extend([(399, 400), (400, 401), (401, 402)]);
    let n_hub = 403usize;
    let g_hub = CsrGraph::<()>::from_unweighted_edges(n_hub, &edges, false, false).unwrap();
    let mut hxs = vec![0.0_f64; n_hub];
    let mut hys = vec![0.0_f64; n_hub];
    for i in 1..n_star {
        let ang = std::f64::consts::TAU * (i as f64) / ((n_star - 1) as f64);
        let (s, c) = ang.sin_cos();
        hxs[i] = 2.0 * c;
        hys[i] = 2.0 * s;
    }
    hxs[400] = hxs[399] + 1.0;
    hys[400] = hys[399];
    hxs[401] = hxs[400] + 1.0;
    hys[401] = hys[400];
    hxs[402] = hxs[1];
    hys[402] = hys[1];

    let cfg_hub_a2 = SpatialRefinerConfig::default()
        .with_iterations(1)
        .with_k_cap(16)
        .with_w_fn(0.0)
        .with_w_anchor2(0.25);
    let mut r_hub = SpatialIntruderRefiner::<f64>::new(&g_hub, cfg_hub_a2).unwrap();
    assert!(r_hub.stats().used_hub_fallback);
    assert!(!r_hub.u2_hub.is_empty());
    let mut l_hub = Layout2d::from_xy(hxs, hys).unwrap();
    r_hub.run(&mut l_hub).unwrap();
    assert!(l_hub.is_all_finite());
    assert!(r_hub.stats().n_anchors_total > g_hub.num_edges());
}

#[test]
fn test_refiner_dynamic_k_cap_decay_and_set_k_cap() {
    let grid = grid_graph(4, 4);
    let n = grid.num_nodes();
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for r in 0..4 {
        for c in 0..4 {
            xs.push(c as f64);
            ys.push(r as f64);
        }
    }
    // Place corner node 15 next to node 0 to ensure active rounds.
    xs[15] = 0.05;
    ys[15] = 0.05;
    let init = Layout2d::from_xy(xs, ys).unwrap();

    let cfg = SpatialRefinerConfig::default()
        .with_iterations(4)
        .with_k_cap(12)
        .with_k_cap_decay(0.5)
        .with_k_cap_min(4);
    let mut r_run = SpatialIntruderRefiner::<f64>::new(&grid, cfg.clone()).unwrap();
    let mut r_step = SpatialIntruderRefiner::<f64>::new(&grid, cfg).unwrap();
    assert_eq!(r_step.current_k_cap(), 12);
    assert!(matches!(
        r_step.set_k_cap(0),
        Err(CoreError::InvalidInput(_))
    ));

    let mut l_run = init.clone();
    let mut l_step = init.clone();
    r_run.run(&mut l_run).unwrap();

    r_step.step(&mut l_step).unwrap();
    assert_eq!(r_step.current_k_cap(), 6);
    r_step.step(&mut l_step).unwrap();
    assert_eq!(r_step.current_k_cap(), 4);
    while !r_step.status().is_finished() {
        r_step.step(&mut l_step).unwrap();
        assert!(r_step.current_k_cap() >= 4);
    }
    for i in 0..n {
        assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
        assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
    }

    // Test explicit set_k_cap shrinking and growing past initial capacity.
    let mut r_dyn = SpatialIntruderRefiner::<f64>::new(
        &grid,
        SpatialRefinerConfig::default()
            .with_iterations(3)
            .with_k_cap(6),
    )
    .unwrap();
    let mut l_dyn = init;
    r_dyn.set_k_cap(3).unwrap();
    assert_eq!(r_dyn.current_k_cap(), 3);
    r_dyn.step(&mut l_dyn).unwrap();
    r_dyn.set_k_cap(14).unwrap();
    assert_eq!(r_dyn.current_k_cap(), 14);
    r_dyn.run(&mut l_dyn).unwrap();
    assert!(l_dyn.is_all_finite());
}

#[test]
fn test_refiner_sgd_pair_buffer_zero_alloc_path() {
    let mut buf = SgdPairBuffer::<f64>::with_capacity(6);
    buf.push_spring(0, 1, 1.0, 1e-6).unwrap();
    buf.push_spring(1, 2, 1.0, 1.0).unwrap();
    buf.push_spring(2, 3, 1.0, 2.0).unwrap();
    buf.push_spring(3, 4, 1.0, 4.0).unwrap();
    buf.push_hinge(0, 4, 2.0, 1e6).unwrap();
    assert_eq!(buf.len(), 5);
    assert_eq!(buf.num_springs(), 4);
    assert_eq!(buf.num_hinges(), 1);

    buf.condition_weights(MAX_WEIGHT_RATIO).unwrap();
    assert_eq!(buf.len(), 5);

    let sched = SgdScheduleConfig::new(10)
        .with_eps(0.05)
        .with_kind(SgdScheduleKind::Exponential);
    let mut sgd = SgdSolver::<Layout2d<f64>>::from_pair_buffer(5, buf, sched, 42).unwrap();
    let mut layout = Layout2d::from_xy(
        vec![0.0, 1.2, 2.1, 3.0, 0.1],
        vec![0.0, 0.2, -0.1, 0.0, 0.1],
    )
    .unwrap();
    let status = sgd.run(&mut layout).unwrap();
    assert!(status.is_finished());
    assert!(layout.is_all_finite());
    let sep_0_4 = (layout.x()[0] - layout.x()[4]).hypot(layout.y()[0] - layout.y()[4]);
    assert!(
        sep_0_4 > 0.5,
        "expected SgdPairBuffer hinge to separate (0, 4), got {sep_0_4}"
    );
}

#[test]
fn test_refiner_folded_grid_strip_f64_and_f32() {
    // 2x8 grid strip folded back onto itself at column 4 so distant >2-hop layers
    // overlap while 1-hop edges remain near unit scale.
    let grid = grid_graph(2, 8);
    let mut x64 = Vec::with_capacity(16);
    let mut y64 = Vec::with_capacity(16);
    for r in 0..2 {
        for c in 0..8 {
            let folded_x = if c < 4 {
                c as f64
            } else {
                (7 - c) as f64 + 0.05
            };
            let folded_y = (r as f64) + if c >= 4 { 0.05 } else { 0.0 };
            x64.push(folded_x);
            y64.push(folded_y);
        }
    }

    let cfg = SpatialRefinerConfig::default()
        .with_iterations(4)
        .with_k_cap(8)
        .with_w_fp(0.25)
        .with_seed(42);

    // Overlap distance between distant pair (0, 7) (7 hops apart) before refinement.
    let orig_0_7 = (x64[0] - x64[7]).hypot(y64[0] - y64[7]);
    assert!(orig_0_7 < 0.15);

    let mut l64_run = Layout2d::from_xy(x64.clone(), y64.clone()).unwrap();
    let mut l64_step = l64_run.clone();
    let mut r64_run = SpatialIntruderRefiner::<f64>::new(&grid, cfg.clone()).unwrap();
    let mut r64_step = SpatialIntruderRefiner::<f64>::new(&grid, cfg.clone()).unwrap();
    assert!(r64_run.run(&mut l64_run).unwrap().is_finished());
    while !r64_step.status().is_finished() {
        r64_step.step(&mut l64_step).unwrap();
    }
    assert!(r64_run.stats().n_fp_total > 0);
    assert_eq!(r64_run.stats(), r64_step.stats());
    for i in 0..16 {
        assert_eq!(l64_run.x()[i].to_bits(), l64_step.x()[i].to_bits());
        assert_eq!(l64_run.y()[i].to_bits(), l64_step.y()[i].to_bits());
    }
    let sep64_0_7 = (l64_run.x()[0] - l64_run.x()[7]).hypot(l64_run.y()[0] - l64_run.y()[7]);
    assert!(
        sep64_0_7 > orig_0_7 * 2.0,
        "f64 folded 2x8 strip should separate distant pair (0, 7): {orig_0_7} -> {sep64_0_7}"
    );

    // f32 parity and separation on the same folded 2x8 strip.
    let x32: Vec<f32> = x64.iter().map(|&v| v as f32).collect();
    let y32: Vec<f32> = y64.iter().map(|&v| v as f32).collect();
    let mut l32_run = Layout2d::from_xy(x32, y32).unwrap();
    let mut l32_step = l32_run.clone();
    let mut r32_run = SpatialIntruderRefiner::<f32>::new(&grid, cfg.clone()).unwrap();
    let mut r32_step = SpatialIntruderRefiner::<f32>::new(&grid, cfg).unwrap();
    assert!(r32_run.run(&mut l32_run).unwrap().is_finished());
    while !r32_step.status().is_finished() {
        r32_step.step(&mut l32_step).unwrap();
    }
    assert!(r32_run.stats().n_fp_total > 0);
    for i in 0..16 {
        assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
        assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
    }
    let sep32_0_7 =
        ((l32_run.x()[0] - l32_run.x()[7]) as f64).hypot((l32_run.y()[0] - l32_run.y()[7]) as f64);
    assert!(
        sep32_0_7 > orig_0_7 * 2.0,
        "f32 folded 2x8 strip should separate distant pair (0, 7): {orig_0_7} -> {sep32_0_7}"
    );
}
