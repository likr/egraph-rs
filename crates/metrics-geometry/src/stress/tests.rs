use egraph_core::{CoreError, CsrGraph, DeterministicRng, Layout2d};

use super::sampled::select_degree_stratified_pivots;
use super::*;

fn scale_layout(layout: &Layout2d<f64>, factor: f64) -> Layout2d<f64> {
    let x: Vec<f64> = layout.x().iter().map(|&v| v * factor).collect();
    let y: Vec<f64> = layout.y().iter().map(|&v| v * factor).collect();
    Layout2d::from_xy(x, y).unwrap()
}

fn random_layout(n: usize, seed: u64, scale: f64) -> Layout2d<f64> {
    let mut rng = DeterministicRng::from_seed_u64(seed);
    let x: Vec<f64> = (0..n).map(|_| (rng.next_f64() - 0.5) * scale).collect();
    let y: Vec<f64> = (0..n).map(|_| (rng.next_f64() - 0.5) * scale).collect();
    Layout2d::from_xy(x, y).unwrap()
}

fn make_path(n: usize) -> (CsrGraph<f64>, Layout2d<f64>) {
    let edges: Vec<(u32, u32)> = (0..n - 1).map(|i| (i as u32, (i + 1) as u32)).collect();
    let g = CsrGraph::from_edges(n, &edges, None, false, false).unwrap();
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y = vec![0.0; n];
    (g, Layout2d::from_xy(x, y).unwrap())
}

fn make_cycle(n: usize) -> (CsrGraph<f64>, Layout2d<f64>) {
    let edges: Vec<(u32, u32)> = (0..n).map(|i| (i as u32, ((i + 1) % n) as u32)).collect();
    let g = CsrGraph::from_edges(n, &edges, None, false, false).unwrap();
    let x: Vec<f64> = (0..n)
        .map(|i| (i as f64 * std::f64::consts::TAU / n as f64).cos())
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| (i as f64 * std::f64::consts::TAU / n as f64).sin())
        .collect();
    (g, Layout2d::from_xy(x, y).unwrap())
}

fn make_grid(rows: usize, cols: usize) -> (CsrGraph<f64>, Layout2d<f64>) {
    let n = rows * cols;
    let mut edges = Vec::new();
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for r in 0..rows {
        for c in 0..cols {
            let id = (r * cols + c) as u32;
            x.push(c as f64);
            y.push(r as f64);
            if c + 1 < cols {
                edges.push((id, id + 1));
            }
            if r + 1 < rows {
                edges.push((id, id + cols as u32));
            }
        }
    }
    let g = CsrGraph::from_edges(n, &edges, None, false, false).unwrap();
    (g, Layout2d::from_xy(x, y).unwrap())
}

fn make_barbell(clique_size: usize, bridge_len: usize) -> (CsrGraph<f64>, Layout2d<f64>) {
    let n = 2 * clique_size + bridge_len;
    let mut edges = Vec::new();
    for i in 0..clique_size {
        for j in (i + 1)..clique_size {
            edges.push((i as u32, j as u32));
        }
    }
    let offset = clique_size + bridge_len;
    for i in 0..clique_size {
        for j in (i + 1)..clique_size {
            edges.push(((offset + i) as u32, (offset + j) as u32));
        }
    }
    let mut prev = (clique_size - 1) as u32;
    for b in 0..bridge_len {
        let cur = (clique_size + b) as u32;
        edges.push((prev, cur));
        prev = cur;
    }
    edges.push((prev, offset as u32));

    let g = CsrGraph::from_edges(n, &edges, None, false, false).unwrap();
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let radius = 0.6;
    for i in 0..clique_size {
        let ang = i as f64 * std::f64::consts::TAU / clique_size as f64;
        x.push(-3.0 + radius * ang.cos());
        y.push(radius * ang.sin());
    }
    for b in 0..bridge_len {
        let t = (b + 1) as f64 / (bridge_len + 1) as f64;
        x.push(-2.0 + 4.0 * t);
        y.push(0.0);
    }
    for i in 0..clique_size {
        let ang = i as f64 * std::f64::consts::TAU / clique_size as f64;
        x.push(3.0 + radius * ang.cos());
        y.push(radius * ang.sin());
    }
    (g, Layout2d::from_xy(x, y).unwrap())
}

fn make_karate() -> (CsrGraph<f64>, Layout2d<f64>) {
    const KARATE_EDGES: &[(u32, u32)] = &[
        (1, 0),
        (1, 2),
        (1, 3),
        (1, 7),
        (1, 13),
        (1, 17),
        (1, 19),
        (1, 21),
        (1, 30),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 7),
        (0, 8),
        (0, 10),
        (0, 11),
        (0, 12),
        (0, 13),
        (0, 17),
        (0, 19),
        (0, 21),
        (0, 31),
        (2, 3),
        (2, 7),
        (2, 8),
        (2, 9),
        (2, 13),
        (2, 27),
        (2, 28),
        (2, 32),
        (3, 7),
        (3, 12),
        (3, 13),
        (4, 6),
        (4, 10),
        (5, 6),
        (5, 10),
        (5, 16),
        (6, 16),
        (8, 30),
        (8, 32),
        (8, 33),
        (9, 33),
        (13, 33),
        (19, 33),
        (25, 23),
        (25, 24),
        (25, 31),
        (23, 27),
        (23, 29),
        (23, 32),
        (23, 33),
        (24, 27),
        (24, 31),
        (27, 33),
        (28, 31),
        (28, 33),
        (29, 26),
        (29, 32),
        (29, 33),
        (26, 33),
        (30, 32),
        (30, 33),
        (31, 32),
        (31, 33),
        (32, 14),
        (32, 15),
        (32, 18),
        (32, 20),
        (32, 22),
        (32, 33),
        (14, 33),
        (15, 33),
        (18, 33),
        (20, 33),
        (22, 33),
    ];
    let n = 34usize;
    let g = CsrGraph::from_edges(n, KARATE_EDGES, None, false, true).unwrap();
    let apsp = all_pairs_bfs(&g, 1.0).unwrap();
    // Deterministic stress-SGD embedding from circular initialization.
    let mut xs: Vec<f64> = (0..n)
        .map(|i| (i as f64 * std::f64::consts::TAU / n as f64).cos())
        .collect();
    let mut ys: Vec<f64> = (0..n)
        .map(|i| (i as f64 * std::f64::consts::TAU / n as f64).sin())
        .collect();
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            pairs.push((i, j, apsp[i * n + j]));
        }
    }
    let mut rng = DeterministicRng::from_seed_u64(7);
    for iter in 0..60 {
        rng.shuffle(&mut pairs);
        let eta = 0.6 * (0.05_f64).powf(iter as f64 / 59.0);
        for &(i, j, d) in &pairs {
            let w = 1.0 / (d * d);
            let mu = (eta * w).min(1.0);
            let dx = xs[i] - xs[j];
            let dy = ys[i] - ys[j];
            let r = dx.hypot(dy).max(1e-9);
            let step = 0.5 * mu * (r - d) / r;
            xs[i] -= step * dx;
            ys[i] -= step * dy;
            xs[j] += step * dx;
            ys[j] += step * dy;
        }
    }
    (g, Layout2d::from_xy(xs, ys).unwrap())
}

#[test]
fn test_bfs_scratch_reuse_and_apsp() {
    let g = CsrGraph::<f64>::from_edges(6, &[(0, 1), (1, 2), (2, 3), (4, 5)], None, false, false)
        .unwrap();
    let mut scratch = BfsScratch::new(6);
    let d0 = bfs_single_source(&g, 0, 2.0, &mut scratch)
        .unwrap()
        .to_vec();
    assert_eq!(d0[0], 0.0);
    assert_eq!(d0[1], 2.0);
    assert_eq!(d0[2], 4.0);
    assert_eq!(d0[3], 6.0);
    assert!(d0[4].is_infinite());
    assert!(d0[5].is_infinite());

    let d4 = bfs_single_source(&g, 4, 1.0, &mut scratch).unwrap();
    assert!(d4[0].is_infinite());
    assert_eq!(d4[4], 0.0);
    assert_eq!(d4[5], 1.0);

    let apsp = all_pairs_bfs(&g, 1.0).unwrap();
    assert_eq!(apsp.len(), 36);
    assert_eq!(apsp[3], 3.0);
    assert!(apsp[4].is_infinite());

    assert!(matches!(
        bfs_single_source(&g, 6, 1.0, &mut scratch),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        bfs_single_source(&g, 0, 0.0, &mut scratch),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        bfs_single_source(&g, 0, f64::NAN, &mut scratch),
        Err(CoreError::NonFiniteValue { .. })
    ));
}

#[test]
fn test_scale_invariance_and_zero_inversion_against_random() {
    let fixtures: Vec<(&str, CsrGraph<f64>, Layout2d<f64>)> = vec![
        {
            let (g, l) = make_path(16);
            ("path", g, l)
        },
        {
            let (g, l) = make_cycle(20);
            ("cycle", g, l)
        },
        {
            let (g, l) = make_grid(5, 5);
            ("grid", g, l)
        },
        {
            let (g, l) = make_barbell(5, 3);
            ("barbell", g, l)
        },
        {
            let (g, l) = make_karate();
            ("karate", g, l)
        },
    ];

    let mut total_comparisons = 0usize;
    let mut inversions = 0usize;

    for (name, g, layout) in &fixtures {
        let s1 = scale_normalized_stress(g, layout).unwrap();
        let s001 = scale_normalized_stress(g, &scale_layout(layout, 0.01)).unwrap();
        let s100 = scale_normalized_stress(g, &scale_layout(layout, 100.0)).unwrap();
        assert!(
            (s001 - s1).abs() < 1e-12 && (s100 - s1).abs() < 1e-12,
            "{name}: scale invariance failed: s001={s001}, s1={s1}, s100={s100}"
        );

        // Also verify precomputed APSP variant matches graph-computing SNS identically.
        let apsp = all_pairs_bfs(g, 1.0).unwrap();
        let s_pre = scale_normalized_stress_from_distances(layout, &apsp, None).unwrap();
        assert!((s_pre - s1).abs() < 1e-12, "{name}: precomputed mismatch");

        // Check sampled SNS exact parity when all nodes are pivots, and scale invariance.
        let s_sampled_all = sampled_scale_normalized_stress(g, layout, g.num_nodes(), 42).unwrap();
        assert!((s_sampled_all - s1).abs() < 1e-12);
        let s_sampled = sampled_scale_normalized_stress(g, layout, 8, 42).unwrap();
        let s_sampled_100 =
            sampled_scale_normalized_stress(g, &scale_layout(layout, 100.0), 8, 42).unwrap();
        assert!((s_sampled - s_sampled_100).abs() < 1e-12);

        // Test across 20 random layouts at adversarial scales (0.01, 1.0, 100.0).
        for seed in 1..=20u64 {
            for &rand_scale in &[0.01, 1.0, 100.0] {
                let r_layout = random_layout(g.num_nodes(), seed, rand_scale);
                let r_sns = scale_normalized_stress(g, &r_layout).unwrap();
                total_comparisons += 1;
                if s1 >= r_sns {
                    inversions += 1;
                }
                assert!(
                    s1 < r_sns,
                    "{name} seed={seed} scale={rand_scale}: SNS inverted ({s1} >= {r_sns})"
                );
            }
        }
    }

    assert_eq!(inversions, 0);
    assert_eq!(total_comparisons, 5 * 20 * 3);
}

#[test]
fn test_sns_degeneracies_and_error_cases() {
    // Collapsed layout -> alpha_star = 0, normalized_stress = 1.0, no NaN.
    let (g, _) = make_path(4);
    let collapsed = Layout2d::<f64>::zeros(4);
    let eval = scale_normalized_stress_detail(&g, &collapsed).unwrap();
    assert_eq!(eval.alpha_star, 0.0);
    assert_eq!(eval.normalized_stress, 1.0);
    assert!(eval.raw_stress.is_finite());

    // Edgeless graph -> no positive-distance pairs -> 0.0.
    let edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    assert_eq!(
        scale_normalized_stress(&edgeless, &Layout2d::<f64>::zeros(4)).unwrap(),
        0.0
    );

    // Precomputed zero / NaN / Inf distances skipped cleanly.
    let dm_zero = vec![0.0; 16];
    assert_eq!(
        scale_normalized_stress_from_distances(&collapsed, &dm_zero, None).unwrap(),
        0.0
    );
    let dm_nan = vec![f64::NAN; 16];
    assert_eq!(
        scale_normalized_stress_from_distances(&collapsed, &dm_nan, None).unwrap(),
        0.0
    );

    // Validation errors.
    let bad = Layout2d::from_xy(vec![0.0, f64::NAN, 1.0, 2.0], vec![0.0; 4]).unwrap();
    assert!(matches!(
        scale_normalized_stress(&g, &bad),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        scale_normalized_stress(&g, &Layout2d::<f64>::zeros(3)),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        sampled_scale_normalized_stress(&g, &collapsed, 0, 1),
        Err(CoreError::InvalidInput(_))
    ));
}

#[test]
fn test_degree_stratified_pivot_coverage_and_exact_parity() {
    // Construct a heterogeneous graph with 40 isolated nodes, 60 deg-1 leaves,
    // 30 mid-degree nodes (deg >= 4), and 10 high-degree hubs (deg >= 12).
    let mut edges = Vec::new();
    // 10 hubs (0..10) forming a clique (deg >= 9), each with 6 leaves (10..70).
    for i in 0..10u32 {
        for j in (i + 1)..10u32 {
            edges.push((i, j));
        }
    }
    for h in 0..10u32 {
        for l in 0..6u32 {
            edges.push((h, 10 + h * 6 + l));
        }
    }
    // 30 mid-degree nodes (70..100) forming a 4-regular-ish ring + chord.
    for k in 0..30u32 {
        let u = 70 + k;
        let v1 = 70 + ((k + 1) % 30);
        let v2 = 70 + ((k + 2) % 30);
        edges.push((u, v1));
        edges.push((u, v2));
    }
    // Connect mid-degree ring to hub 0 via two bridges so component is mostly connected.
    edges.push((0, 70));
    edges.push((5, 85));
    // Nodes 100..140 remain isolated (deg == 0).
    let n = 140usize;
    let g = CsrGraph::<f64>::from_edges(n, &edges, None, false, true).unwrap();

    let pivots = select_degree_stratified_pivots(&g, 20, 42);
    assert_eq!(pivots.len(), 20);
    // Zero isolated nodes selected while non_iso (100) >= 20.
    assert!(pivots.iter().all(|&u| g.degree(u) > 0));
    // All selected pivots are distinct.
    let mut uniq = pivots.clone();
    uniq.sort_unstable();
    uniq.dedup();
    assert_eq!(uniq.len(), 20);

    // Proportional coverage across leaves (60% -> ~12), mid (30% -> ~6), hubs (10% -> ~2).
    let leaf_cnt = pivots.iter().filter(|&&u| g.degree(u) == 1).count();
    let mid_cnt = pivots
        .iter()
        .filter(|&&u| (2..=8).contains(&g.degree(u)))
        .count();
    let hub_cnt = pivots.iter().filter(|&&u| g.degree(u) >= 9).count();
    assert_eq!(leaf_cnt + mid_cnt + hub_cnt, 20);
    assert!((11..=13).contains(&leaf_cnt), "leaf_cnt={leaf_cnt}");
    assert!((5..=7).contains(&mid_cnt), "mid_cnt={mid_cnt}");
    assert!((1..=3).contains(&hub_cnt), "hub_cnt={hub_cnt}");

    // Exact parity with scale_normalized_stress when k >= non_iso (100) even if k < n (140).
    let layout64 = random_layout(n, 77, 10.0);
    let exact_sns = scale_normalized_stress(&g, &layout64).unwrap();
    let non_iso_sns = sampled_scale_normalized_stress(&g, &layout64, 100, 42).unwrap();
    let all_sns = sampled_scale_normalized_stress(&g, &layout64, 200, 42).unwrap();
    assert!((non_iso_sns - exact_sns).abs() < 1e-12);
    assert!((all_sns - exact_sns).abs() < 1e-12);

    // f32 scale invariance on sampled_scale_normalized_stress.
    let lx32: Vec<f32> = layout64.x().iter().map(|&v| v as f32).collect();
    let ly32: Vec<f32> = layout64.y().iter().map(|&v| v as f32).collect();
    let l32 = Layout2d::from_xy(lx32.clone(), ly32.clone()).unwrap();
    let l32_scaled = Layout2d::from_xy(
        lx32.iter().map(|&v| v * 12.5).collect(),
        ly32.iter().map(|&v| v * 12.5).collect(),
    )
    .unwrap();
    let s32_a = sampled_scale_normalized_stress(&g, &l32, 32, 99).unwrap();
    let s32_b = sampled_scale_normalized_stress(&g, &l32_scaled, 32, 99).unwrap();
    assert!((s32_a - s32_b).abs() < 1e-6);
}
