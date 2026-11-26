use super::helpers::{MAX_WEIGHT_RATIO, condition_weights_packed};
use super::tests_basic::grid_graph;
use super::*;
use egraph_core::IterativeSolver;
use egraph_layout_sgd::sgd::PackedSymPair;

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
    let cfg_a2 = SpatialRefinerConfig {
        rounds: 2,
        w_anchor2: 0.25,
        seed: 42,
        ..Default::default()
    };

    // Clean layout must remain untouched even with w_anchor2 > 0.0.
    let mut l_clean = clean_layout.clone();
    let mut r_clean = SpatialIntruderRefiner::<f64>::new(&grid, cfg_a2).unwrap();
    assert_eq!(r_clean.run(&mut l_clean).unwrap(), SolverStatus::Converged);
    assert_eq!(r_clean.stats().rounds_executed, 0);

    // Intruder layout: w_anchor2 > 0 increases n_anchors relative to w_anchor2 == 0,
    // and step == run is bit-exact in f64 and f32.
    let mut intr_x = clean_x;
    let mut intr_y = clean_y;
    intr_x[8] = 0.1;
    intr_y[8] = 0.1;
    let intr_init = Layout2d::from_xy(intr_x.clone(), intr_y.clone()).unwrap();

    let cfg_base = SpatialRefinerConfig {
        w_anchor2: 0.0,
        ..cfg_a2
    };
    let mut r_base = SpatialIntruderRefiner::<f64>::new(&grid, cfg_base).unwrap();
    let mut l_base = intr_init.clone();
    r_base.run(&mut l_base).unwrap();

    let mut r_run = SpatialIntruderRefiner::<f64>::new(&grid, cfg_a2).unwrap();
    let mut r_step = SpatialIntruderRefiner::<f64>::new(&grid, cfg_a2).unwrap();
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
    let mut r32_run = SpatialIntruderRefiner::<f32>::new(&grid, cfg_a2).unwrap();
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
    let g_hub = CsrGraph::<f64>::from_edges(n_hub, &edges, None, false, false).unwrap();
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

    let cfg_hub_a2 = SpatialRefinerConfig {
        rounds: 1,
        k_cap: 16,
        w_fn: 0.0,
        w_anchor2: 0.25,
        ..Default::default()
    };
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

    let cfg = SpatialRefinerConfig {
        rounds: 4,
        k_cap: 12,
        k_cap_decay: 0.5,
        k_cap_min: 4,
        ..Default::default()
    };
    let mut r_run = SpatialIntruderRefiner::<f64>::new(&grid, cfg).unwrap();
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
        SpatialRefinerConfig {
            rounds: 3,
            k_cap: 6,
            ..Default::default()
        },
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
fn test_condition_weights_packed_zero_alloc_equivalence() {
    for len in [5usize, 6] {
        let raw_weights: Vec<f64> = if len == 5 {
            vec![1e-6, 1.0, 2.0, 4.0, 1e6]
        } else {
            vec![1e-6, 1.0, 2.0, 4.0, 8.0, 1e6]
        };
        let mut pairs: Vec<PackedSymPair<f64>> = raw_weights
            .iter()
            .enumerate()
            .map(|(idx, &w)| PackedSymPair::new(idx as u32, (idx + 1) as u32, false, 1.0, w))
            .collect();

        condition_weights_packed(&mut pairs, MAX_WEIGHT_RATIO);

        let half_span = MAX_WEIGHT_RATIO.sqrt();
        let w_med = if len == 5 { 2.0 } else { 3.0 };
        let lo = w_med / half_span;
        let hi = w_med * half_span;
        let mut got: Vec<f64> = pairs.iter().map(|p| p.w).collect();
        got.sort_by(f64::total_cmp);
        let mut expected: Vec<f64> = raw_weights.iter().map(|&w| w.clamp(lo, hi)).collect();
        expected.sort_by(f64::total_cmp);
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-12);
        }
    }
}
