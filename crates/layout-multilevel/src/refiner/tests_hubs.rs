use super::tests_basic::grid_graph;
use super::*;
use egraph_core::IterativeSolver;

#[test]
fn test_refiner_hub_fallback_path_and_coincident_jitter() {
    // Hub star of 400 nodes (deg[0]=399 => two_hop_work > 159_000 > max(100_000, 64*403))
    // plus path 399-400-401-402 so node 402 is 4 hops from node 1.
    let n_star = 400usize;
    let mut edges: Vec<(u32, u32)> = (1..n_star as u32).map(|v| (0, v)).collect();
    edges.push((399, 400));
    edges.push((400, 401));
    edges.push((401, 402));
    let n = 403usize;
    let g = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();

    let mut xs = vec![0.0_f64; n];
    let mut ys = vec![0.0_f64; n];
    for i in 1..n_star {
        let ang = std::f64::consts::TAU * (i as f64) / ((n_star - 1) as f64);
        let (s, c) = ang.sin_cos();
        xs[i] = 2.0 * c;
        ys[i] = 2.0 * s;
    }
    xs[400] = xs[399] + 1.0;
    ys[400] = ys[399];
    xs[401] = xs[400] + 1.0;
    ys[401] = ys[400];
    // Place >2-hop node 402 coincident with leaf node 1!
    xs[402] = xs[1];
    ys[402] = ys[1];

    let cfg = SpatialRefinerConfig {
        iterations: 2,
        k_cap: 16,
        ..Default::default()
    };
    let mut refiner = SpatialIntruderRefiner::<f64>::new(&g, cfg).unwrap();
    assert!(refiner.stats().used_hub_fallback);

    let mut layout = Layout2d::from_xy(xs, ys).unwrap();
    let status = refiner.run(&mut layout).unwrap();
    assert!(status.is_finished());
    assert!(layout.is_all_finite());
    assert!(refiner.stats().n_fp_total > 0);
    let sep_1_402 = (layout.x()[1] - layout.x()[402]).hypot(layout.y()[1] - layout.y()[402]);
    assert!(
        sep_1_402 > 0.1,
        "expected hub fallback to separate coincident intruder (1, 402), got {sep_1_402}"
    );
}

#[test]
fn test_refiner_opt_in_separation_hinge_separates_closed_twins() {
    // 3x3 grid (nodes 0..9) + closed-twin pair (9, 10) attached to node 0 with triangle (0, 9), (0, 10), (9, 10)
    // so N[9] = {0, 9, 10} = N[10]. Nodes 9 and 10 are placed collapsed at (0.0, 0.0).
    let mut edges = Vec::new();
    for r in 0..3 {
        for c in 0..3 {
            let u = (r * 3 + c) as u32;
            if c + 1 < 3 {
                edges.push((u, u + 1));
            }
            if r + 1 < 3 {
                edges.push((u, u + 3));
            }
        }
    }
    edges.extend([(0, 9), (0, 10), (9, 10)]);
    let g = CsrGraph::<f64>::from_edges(11, &edges, None, false, false).unwrap();

    let mut base_x = Vec::with_capacity(11);
    let mut base_y = Vec::with_capacity(11);
    for r in 0..3 {
        for c in 0..3 {
            base_x.push(c as f64);
            base_y.push(r as f64);
        }
    }
    base_x.extend([0.0, 0.0]);
    base_y.extend([0.0, 0.0]);

    // 1. Legacy defaults (min_sep_ratio = 0.0, w_sep = 0.0) preserve backward-compat: twins stay collapsed.
    let legacy_cfg = SpatialRefinerConfig::default();
    let mut l_legacy = Layout2d::from_xy(base_x.clone(), base_y.clone()).unwrap();
    let mut r_legacy = SpatialIntruderRefiner::<f64>::new(&g, legacy_cfg).unwrap();
    r_legacy.run(&mut l_legacy).unwrap();
    assert_eq!(r_legacy.stats().n_sep_total, 0);
    let d_legacy_9_10 =
        (l_legacy.x()[9] - l_legacy.x()[10]).hypot(l_legacy.y()[9] - l_legacy.y()[10]);
    assert!(
        d_legacy_9_10 < 1e-2,
        "expected legacy config to leave closed twins collapsed, got d={d_legacy_9_10}"
    );

    // 2. Opt-in separation hinge (min_sep_ratio = 0.15, w_sep = 1.0) eliminates closed-twin
    // collapse and zero-length edges in both f64 and f32, with bit-exact step == run.
    let sep_cfg = SpatialRefinerConfig {
        min_sep_ratio: 0.15,
        w_sep: 1.0,
        ..Default::default()
    };
    let mut l64_run = Layout2d::from_xy(base_x.clone(), base_y.clone()).unwrap();
    let mut l64_step = l64_run.clone();
    let mut r64_run = SpatialIntruderRefiner::<f64>::new(&g, sep_cfg).unwrap();
    let mut r64_step = SpatialIntruderRefiner::<f64>::new(&g, sep_cfg).unwrap();
    r64_run.run(&mut l64_run).unwrap();
    r64_step.step(&mut l64_step).unwrap();
    r64_step.run(&mut l64_step).unwrap();
    assert_eq!(r64_run.stats(), r64_step.stats());
    assert!(r64_run.stats().n_sep_total > 0);
    for i in 0..11 {
        assert_eq!(l64_run.x()[i].to_bits(), l64_step.x()[i].to_bits());
        assert_eq!(l64_run.y()[i].to_bits(), l64_step.y()[i].to_bits());
    }
    for i in 0..11 {
        for j in (i + 1)..11 {
            let d = (l64_run.x()[i] - l64_run.x()[j]).hypot(l64_run.y()[i] - l64_run.y()[j]);
            assert!(d > 0.05, "pair ({i}, {j}) too close after sep hinge: d={d}");
        }
    }
    for &(u, v) in &edges {
        let d = (l64_run.x()[u as usize] - l64_run.x()[v as usize])
            .hypot(l64_run.y()[u as usize] - l64_run.y()[v as usize]);
        assert!(d > 0.05, "edge ({u}, {v}) near zero length: d={d}");
    }

    // f32 parity
    let bx32: Vec<f32> = base_x.iter().map(|&v| v as f32).collect();
    let by32: Vec<f32> = base_y.iter().map(|&v| v as f32).collect();
    let mut l32_run = Layout2d::from_xy(bx32, by32).unwrap();
    let mut l32_step = l32_run.clone();
    let mut r32_run = SpatialIntruderRefiner::<f32>::new(&g, sep_cfg).unwrap();
    let mut r32_step = SpatialIntruderRefiner::<f32>::new(&g, sep_cfg).unwrap();
    r32_run.run(&mut l32_run).unwrap();
    r32_step.step(&mut l32_step).unwrap();
    r32_step.run(&mut l32_step).unwrap();
    assert!(r32_run.stats().n_sep_total > 0);
    for i in 0..11 {
        assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
        assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
        for j in (i + 1)..11 {
            let d = (l32_run.x()[i] - l32_run.x()[j]).hypot(l32_run.y()[i] - l32_run.y()[j]);
            assert!(d > 0.05, "f32 pair ({i}, {j}) too close: d={d}");
        }
    }

    // 3. Clean 3x3 grid without collapsed vertices remains completely untouched even with sep hinge enabled.
    let clean_grid = grid_graph(3, 3);
    let clean_orig = Layout2d::from_xy(base_x[..9].to_vec(), base_y[..9].to_vec()).unwrap();
    let mut clean_work = clean_orig.clone();
    let mut r_clean = SpatialIntruderRefiner::<f64>::new(&clean_grid, sep_cfg).unwrap();
    assert_eq!(
        r_clean.run(&mut clean_work).unwrap(),
        SolverStatus::Converged
    );
    assert_eq!(r_clean.stats().n_sep_total, 0);
    assert_eq!(r_clean.stats().rounds_executed, 0);
    for i in 0..9 {
        assert_eq!(clean_orig.x()[i].to_bits(), clean_work.x()[i].to_bits());
        assert_eq!(clean_orig.y()[i].to_bits(), clean_work.y()[i].to_bits());
    }
}

#[test]
fn test_refiner_medium_hub_non_hub_2hop_and_super_hub_fallback() {
    // 1. Medium-hub graph (N = 400, non_hub_thresh = 20):
    //    - Hub 0 connects to ring leaves 1..=30 (deg(0) = 30 > 20, 30^2 = 900 << 100_000 -> use_exact_2hop == true).
    //    - Ring edges (1,2)..(29,30),(30,1) give leaves deg = 3 <= 20 so distance-2 ring neighbors
    //      (e.g. 1 and 3 via 2) are valid non-hub 2-hop pairs while opposite leaves (e.g. 1 and 15)
    //      share ONLY medium hub 0.
    //    - Path 30-31-...-399 gives low-degree background nodes (deg = 2 <= 20).
    let n = 400usize;
    let mut edges: Vec<(u32, u32)> = (1..=30u32).map(|v| (0, v)).collect();
    for i in 1..=30u32 {
        let next = if i == 30 { 1 } else { i + 1 };
        edges.push((i, next));
    }
    for v in 30..399u32 {
        edges.push((v, v + 1));
    }
    let g_med = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();

    let mut base_x = vec![0.0_f64; n];
    let mut base_y = vec![0.0_f64; n];
    for i in 1..=30 {
        let ang = std::f64::consts::TAU * ((i - 1) as f64) / 30.0;
        base_x[i] = 5.0 * ang.cos();
        base_y[i] = 5.0 * ang.sin();
    }
    for v in 31..n {
        base_x[v] = base_x[30] + ((v - 30) as f64);
        base_y[v] = base_y[30];
    }
    let clean_orig = Layout2d::from_xy(base_x.clone(), base_y.clone()).unwrap();

    // Clean medium-hub layout with k_cap = 4 remains completely undisturbed.
    let clean_cfg = SpatialRefinerConfig {
        iterations: 2,
        k_cap: 4,
        ..Default::default()
    };
    let mut l_clean = clean_orig.clone();
    let mut r_clean = SpatialIntruderRefiner::<f64>::new(&g_med, clean_cfg).unwrap();
    assert!(!r_clean.stats().used_hub_fallback);
    assert_eq!(r_clean.run(&mut l_clean).unwrap(), SolverStatus::Converged);
    assert_eq!(r_clean.stats().n_fp_total, 0);
    assert_eq!(r_clean.stats().rounds_executed, 0);
    for i in 0..n {
        assert_eq!(l_clean.x()[i].to_bits(), clean_orig.x()[i].to_bits());
        assert_eq!(l_clean.y()[i].to_bits(), clean_orig.y()[i].to_bits());
    }

    // Pair sharing ONLY non-hub node 2 (deg = 3 <= 20): collapse (1, 3) slightly closer to 2
    // with k_cap = 2 -> NOT flagged as F+ (n_fp_total == 0).
    let mut nonhub_x = base_x.clone();
    let mut nonhub_y = base_y.clone();
    nonhub_x[1] = base_x[2] - 0.05;
    nonhub_y[1] = base_y[2];
    nonhub_x[3] = base_x[2] + 0.05;
    nonhub_y[3] = base_y[2];
    let mut l_nonhub = Layout2d::from_xy(nonhub_x, nonhub_y).unwrap();
    let mut r_nonhub = SpatialIntruderRefiner::<f64>::new(
        &g_med,
        SpatialRefinerConfig {
            iterations: 1,
            k_cap: 2,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        r_nonhub.run(&mut l_nonhub).unwrap(),
        SolverStatus::Converged
    );
    assert_eq!(r_nonhub.stats().n_fp_total, 0);

    // Collapse opposite hub leaves (1, 15) together (share ONLY medium hub 0, deg=30 > 20):
    // medium-hub adj2 filter must detect F+ and push (1, 15) apart, with f64/f32 step == run
    // parity and set_k_cap growth.
    let mut intr_x = base_x;
    let mut intr_y = base_y;
    intr_x[15] = intr_x[1] + 0.01;
    intr_y[15] = intr_y[1];
    let intr_init = Layout2d::from_xy(intr_x.clone(), intr_y.clone()).unwrap();

    let intr_cfg = SpatialRefinerConfig {
        iterations: 2,
        k_cap: 8,
        seed: 19,
        ..Default::default()
    };
    let mut r64_run = SpatialIntruderRefiner::<f64>::new(&g_med, intr_cfg).unwrap();
    let mut r64_step = SpatialIntruderRefiner::<f64>::new(&g_med, intr_cfg).unwrap();
    assert!(!r64_run.stats().used_hub_fallback);
    let mut l64_run = intr_init.clone();
    let mut l64_step = intr_init.clone();
    r64_run.run(&mut l64_run).unwrap();
    r64_step.step(&mut l64_step).unwrap();
    r64_step.run(&mut l64_step).unwrap();
    assert!(r64_run.stats().n_fp_total > 0);
    assert_eq!(r64_run.stats(), r64_step.stats());
    for i in 0..n {
        assert_eq!(l64_run.x()[i].to_bits(), l64_step.x()[i].to_bits());
        assert_eq!(l64_run.y()[i].to_bits(), l64_step.y()[i].to_bits());
    }
    let sep_1_15 = (l64_run.x()[1] - l64_run.x()[15]).hypot(l64_run.y()[1] - l64_run.y()[15]);
    assert!(
        sep_1_15 > 0.1,
        "expected medium-hub adj2 filter to separate (1, 15), got {sep_1_15}"
    );

    // f32 step == run parity on medium-hub intruder layout.
    let intr32 = Layout2d::from_xy(
        intr_x.iter().map(|&v| v as f32).collect(),
        intr_y.iter().map(|&v| v as f32).collect(),
    )
    .unwrap();
    let mut r32_run = SpatialIntruderRefiner::<f32>::new(&g_med, intr_cfg).unwrap();
    let mut r32_step = SpatialIntruderRefiner::<f32>::new(&g_med, intr_cfg).unwrap();
    let mut l32_run = intr32.clone();
    let mut l32_step = intr32;
    r32_run.run(&mut l32_run).unwrap();
    while !r32_step.status().is_finished() {
        r32_step.step(&mut l32_step).unwrap();
    }
    assert!(r32_run.stats().n_fp_total > 0);
    for i in 0..n {
        assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
        assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
    }

    // set_k_cap growth beyond initial capacity on medium-hub graph.
    let mut r_grow = SpatialIntruderRefiner::<f64>::new(&g_med, intr_cfg).unwrap();
    let mut l_grow = intr_init;
    r_grow.step(&mut l_grow).unwrap();
    r_grow.set_k_cap(40).unwrap();
    assert_eq!(r_grow.current_k_cap(), 40);
    r_grow.run(&mut l_grow).unwrap();
    assert!(l_grow.is_all_finite());

    // 2. Super-hub adj1 fallback path (use_exact_2hop == false):
    //    star(400) where hub 0 has deg >= 399 > 20, plus node 399 connected to 300..379
    //    (deg(399) = 80 > 64) so (u=1, v=399) exercises the binary_search branch and
    //    (u=1, v=2) exercises the stamp branch in has_common_non_hub_neighbor.
    let mut hub_edges: Vec<(u32, u32)> = (1..n as u32).map(|v| (0, v)).collect();
    for v in 300..379u32 {
        hub_edges.push((399, v));
    }
    let g_super = CsrGraph::<f64>::from_edges(n, &hub_edges, None, false, false).unwrap();

    let mut sxs = vec![0.0_f64; n];
    let mut sys = vec![0.0_f64; n];
    for i in 1..n {
        let ang = std::f64::consts::TAU * ((i - 1) as f64) / ((n - 1) as f64);
        sxs[i] = 3.0 * ang.cos();
        sys[i] = 3.0 * ang.sin();
    }
    // Collapse (1, 2) and (1, 399) together while sharing ONLY super-hub 0.
    sxs[2] = sxs[1] + 0.005;
    sys[2] = sys[1];
    sxs[399] = sxs[1] - 0.005;
    sys[399] = sys[1] + 0.005;

    let mut r_super = SpatialIntruderRefiner::<f64>::new(
        &g_super,
        SpatialRefinerConfig {
            iterations: 2,
            k_cap: 16,
            seed: 23,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(r_super.stats().used_hub_fallback);
    let mut l_super = Layout2d::from_xy(sxs, sys).unwrap();
    r_super.run(&mut l_super).unwrap();
    assert!(l_super.is_all_finite());
    assert!(r_super.stats().n_fp_total > 0);
    let sep_1_2 = (l_super.x()[1] - l_super.x()[2]).hypot(l_super.y()[1] - l_super.y()[2]);
    let sep_1_399 = (l_super.x()[1] - l_super.x()[399]).hypot(l_super.y()[1] - l_super.y()[399]);
    assert!(
        sep_1_2 > 0.05 && sep_1_399 > 0.05,
        "expected super-hub adj1 fallback to separate (1,2)={sep_1_2} and (1,399)={sep_1_399}"
    );
}
