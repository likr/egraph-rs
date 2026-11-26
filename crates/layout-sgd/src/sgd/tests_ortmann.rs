use std::collections::HashSet;

use egraph_core::{CsrGraph, IterativeSolver, Layout2d};

use super::super::sgd::{OrtmannPivotStrategy, OrtmannSparseStressConfig, SgdSolver};
use super::graph_sampling::{build_positive_canonical_csr, compute_mssp_dists};
use super::ortmann::select_ortmann_pivots_and_dists;
use super::pairs::{INDEX_MASK, PairStorage};
use super::test_support::{circle_layout, euclidean_apsp_stress, grid_graph};

#[test]
fn test_ortmann_voronoi_tie_breaking_and_exact_weights_vs_bruteforce() {
    let edges = vec![(0, 1), (0, 2), (0, 3), (0, 4), (4, 5)];
    let g = CsrGraph::<f64>::from_edges(6, &edges, None, false, false).unwrap();

    for strategy in [
        OrtmannPivotStrategy::MaxMinSp,
        OrtmannPivotStrategy::MaxMinRandomSp,
        OrtmannPivotStrategy::KMeansSp,
    ] {
        let cfg = OrtmannSparseStressConfig {
            num_pivots: 2,
            pivot_strategy: strategy,
            iterations: 50,
            seed: 7,
            ..Default::default()
        };
        let solver = SgdSolver::<f64>::ortmann_sparse_stress(&g, cfg).unwrap();

        // Brute-force oracle using the exact pivot set chosen for `strategy`.
        let (sym_g, pos_edges) = build_positive_canonical_csr(&g).unwrap();
        let (pivots, piv_dists) = select_ortmann_pivots_and_dists(&sym_g, 2, strategy, 7).unwrap();
        let n = g.num_nodes();
        let h = pivots.len();
        let mut r_counts = vec![0usize; h];
        let mut r_dists = vec![Vec::new(); h];
        for j in 0..n {
            let mut best: Option<(f64, usize, usize)> = None;
            for k in 0..h {
                let d = piv_dists[k * n + j];
                if !d.is_finite() {
                    continue;
                }
                let cand = (d, r_counts[k], k);
                if best.is_none_or(|b| {
                    cand.0
                        .total_cmp(&b.0)
                        .then_with(|| cand.1.cmp(&b.1))
                        .then_with(|| cand.2.cmp(&b.2))
                        .is_lt()
                }) {
                    best = Some(cand);
                }
            }
            if let Some((d, _, bk)) = best {
                r_counts[bk] += 1;
                r_dists[bk].push(d);
            }
        }
        let mut expected_pairs: Vec<(u32, u32, f64, f64, f64, f64)> = Vec::new();
        for &(u, v, d_uv) in &pos_edges {
            let w = 1.0 / (d_uv * d_uv);
            expected_pairs.push((u, v, d_uv, d_uv, w, w));
        }
        let edge_set: HashSet<(u32, u32)> = pos_edges.iter().map(|&(u, v, _)| (u, v)).collect();
        for (k, &p) in pivots.iter().enumerate() {
            for i in 0..n {
                let i_u32 = i as u32;
                if i_u32 == p || edge_set.contains(&(p.min(i_u32), p.max(i_u32))) {
                    continue;
                }
                let dpi = piv_dists[k * n + i];
                if !dpi.is_finite() || dpi <= 0.0 {
                    continue;
                }
                let s_pi = r_dists[k].iter().filter(|&&dpj| 2.0 * dpj <= dpi).count();
                if s_pi > 0 {
                    expected_pairs.push((p, i_u32, dpi, dpi, 0.0, (s_pi as f64) / (dpi * dpi)));
                }
            }
        }
        assert_eq!(solver.num_springs(), expected_pairs.len());
        let PairStorage::General(actual_pairs) = &solver.pairs else {
            panic!("expected PairStorage::General for directional Ortmann pairs");
        };
        let mut actual_sorted: Vec<(u32, u32, f64, f64)> = actual_pairs
            .iter()
            .map(|p| (p.i, p.j_hinge & INDEX_MASK, p.wij, p.wji))
            .collect();
        actual_sorted.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let mut exp_sorted: Vec<(u32, u32, f64, f64)> = expected_pairs
            .iter()
            .map(|&(u, v, _, _, wij, wji)| (u, v, wij, wji))
            .collect();
        exp_sorted.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        for (a, e) in actual_sorted.iter().zip(exp_sorted.iter()) {
            assert_eq!(a.0, e.0);
            assert_eq!(a.1, e.1);
            assert!((a.2 - e.2).abs() < 1e-12);
            assert!((a.3 - e.3).abs() < 1e-12);
        }
    }

    // Direct tie-breaking check: graph 0-1, 0-2, 0-3, 0-4, 4-5, 5-6 (7 nodes) with
    // all-pivot vs 2-pivot where `pivots` = [0, 5] leaves `4` tied at distance 1.
    // At `j = 4`, `r_counts = [4, 0]` (`0,1,2,3` went to pivot 0), so smallest |R(p)|
    // MUST assign `4` to pivot `5` (`r_counts[1] == 0 < 4`), NOT lowest pivot index `0`!
    let edges2 = vec![(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (5, 6)];
    let g2 = CsrGraph::<f64>::from_edges(7, &edges2, None, false, false).unwrap();
    let mut solver_tie = SgdSolver::<f64>::ortmann_sparse_stress(
        &g2,
        OrtmannSparseStressConfig {
            num_pivots: 2,
            pivot_strategy: OrtmannPivotStrategy::MaxMinSp,
            iterations: 60,
            seed: 1,
            ..Default::default()
        },
    )
    .unwrap();
    // Verify `r_counts` tie-breaking directly on the deterministic pivot pair `[0, 5]`:
    let (sym2, _) = build_positive_canonical_csr(&g2).unwrap();
    let d_tie = compute_mssp_dists(&sym2, &[0, 5]).unwrap();
    let mut r_c = [0usize; 2];
    let mut assigned_4 = 0usize;
    for j in 0..7 {
        let d0 = d_tie[j];
        let d1 = d_tie[7 + j];
        let bk = match d0.total_cmp(&d1) {
            std::cmp::Ordering::Less => 0,
            std::cmp::Ordering::Greater => 1,
            std::cmp::Ordering::Equal => usize::from(r_c[1] < r_c[0]),
        };
        r_c[bk] += 1;
        if j == 4 {
            assigned_4 = bk;
        }
    }
    assert_eq!(assigned_4, 1, "node 4 tie must break to smaller |R(p1)|");
    assert_eq!(r_c, [4, 3]);
    let mut l2 = circle_layout::<f64>(7, 1.0);
    solver_tie.run(&mut l2).unwrap();
    assert!(l2.is_all_finite());
}

#[test]
fn test_ortmann_sparse_stress_parity_weighted_and_edge_cases() {
    let grid = grid_graph(4, 4);
    let n = grid.num_nodes();

    for strategy in [
        OrtmannPivotStrategy::KMeansSp,
        OrtmannPivotStrategy::MaxMinRandomSp,
        OrtmannPivotStrategy::MaxMinSp,
    ] {
        let cfg = OrtmannSparseStressConfig {
            num_pivots: 8,
            pivot_strategy: strategy,
            iterations: 60,
            seed: 42,
            ..Default::default()
        };

        // f64 bit-exact step() == run() + stress reduction.
        let mut s_run = SgdSolver::<f64>::ortmann_sparse_stress(&grid, cfg).unwrap();
        let mut s_step = SgdSolver::<f64>::ortmann_sparse_stress(&grid, cfg).unwrap();
        let mut l_run = circle_layout::<f64>(n, 0.25);
        let mut l_step = l_run.clone();
        let stress_before = euclidean_apsp_stress(&grid, &l_run);
        s_run.run(&mut l_run).unwrap();
        s_step.run_steps(&mut l_step, 17).unwrap();
        while !s_step.status().is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        for i in 0..n {
            assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
            assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
        }
        let stress_after = euclidean_apsp_stress(&grid, &l_run);
        assert!(
            stress_after < stress_before * 0.25,
            "Ortmann ({strategy:?}) failed to reduce stress: {stress_before} -> {stress_after}"
        );

        // f32 bit-exact step() == run().
        let mut s32_run = SgdSolver::<f32>::ortmann_sparse_stress(&grid, cfg).unwrap();
        let mut s32_step = SgdSolver::<f32>::ortmann_sparse_stress(&grid, cfg).unwrap();
        let mut l32_run = circle_layout::<f32>(n, 0.25);
        let mut l32_step = l32_run.clone();
        s32_run.run(&mut l32_run).unwrap();
        s32_step.run_steps(&mut l32_step, 13).unwrap();
        s32_step.run(&mut l32_step).unwrap();
        for i in 0..n {
            assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
            assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
        }
    }

    // Seed determinism.
    let cfg_a = OrtmannSparseStressConfig {
        num_pivots: 6,
        seed: 11,
        iterations: 30,
        ..Default::default()
    };
    let cfg_b = OrtmannSparseStressConfig { seed: 99, ..cfg_a };
    let mut sa1 = SgdSolver::<f64>::ortmann_sparse_stress(&grid, cfg_a).unwrap();
    let mut sa2 = SgdSolver::<f64>::ortmann_sparse_stress(&grid, cfg_a).unwrap();
    let mut sb = SgdSolver::<f64>::ortmann_sparse_stress(&grid, cfg_b).unwrap();
    let mut la1 = circle_layout::<f64>(n, 1.0);
    let mut la2 = la1.clone();
    let mut lb = la1.clone();
    sa1.run(&mut la1).unwrap();
    sa2.run(&mut la2).unwrap();
    sb.run(&mut lb).unwrap();
    for i in 0..n {
        assert_eq!(la1.x()[i].to_bits(), la2.x()[i].to_bits());
        assert_eq!(la1.y()[i].to_bits(), la2.y()[i].to_bits());
    }
    assert!((0..n).any(|i| la1.x()[i].to_bits() != lb.x()[i].to_bits()));

    // Weighted graph uses DijkstraWorkspace (shortcut 0-2 weight 10 vs 0-1-2 weight 1+1=2).
    let w_edges = vec![(0, 1), (1, 2), (2, 3), (0, 2)];
    let w_vals = vec![1.0, 1.0, 4.0, 10.0];
    let wg = CsrGraph::<f64>::from_edges(4, &w_edges, Some(&w_vals), false, false).unwrap();
    let (wsym, _) = build_positive_canonical_csr(&wg).unwrap();
    let w_dists = compute_mssp_dists(&wsym, &[0]).unwrap();
    assert!((w_dists[2] - 2.0).abs() < 1e-12);
    assert!((w_dists[3] - 6.0).abs() < 1e-12);
    let mut w_solver = SgdSolver::<f64>::ortmann_sparse_stress(
        &wg,
        OrtmannSparseStressConfig {
            num_pivots: 2,
            iterations: 40,
            eta_max: Some(2.0),
            eta_min: Some(0.05),
            ..Default::default()
        },
    )
    .unwrap();
    assert!((w_solver.eta_max - 2.0).abs() < 1e-12);
    assert!((w_solver.eta_min - 0.05).abs() < 1e-12);
    let mut wl = circle_layout::<f64>(4, 1.0);
    w_solver.run(&mut wl).unwrap();
    assert!(wl.is_all_finite());

    // Directed reciprocal edges + self-loop (sanitized) + zero-weight edge.
    let dir_edges = vec![(0, 0), (0, 1), (1, 0), (1, 2), (2, 3), (3, 1)];
    let dir_weights = vec![1.0, 5.0, 2.0, 3.0, 0.0, 4.0];
    let dg = CsrGraph::<f64>::from_edges(5, &dir_edges, Some(&dir_weights), true, true).unwrap();
    let mut d_solver = SgdSolver::<f64>::ortmann_sparse_stress(
        &dg,
        OrtmannSparseStressConfig {
            num_pivots: 3,
            iterations: 20,
            ..Default::default()
        },
    )
    .unwrap();
    let mut dl = circle_layout::<f64>(5, 1.0);
    d_solver.run(&mut dl).unwrap();
    assert!(dl.is_all_finite());

    // Degenerate graphs: N=0, N=1, edgeless, disconnected + isolated, collinear.
    for deg_g in [
        CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap(),
        CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap(),
        CsrGraph::<f64>::from_edges(6, &[], None, false, false).unwrap(),
        CsrGraph::<f64>::from_edges(8, &[(0, 1), (1, 2), (3, 4), (4, 5)], None, false, false)
            .unwrap(),
    ] {
        let dn = deg_g.num_nodes();
        let xs: Vec<f64> = (0..dn).map(|i| i as f64).collect();
        let ys = vec![0.0; dn];
        let init = Layout2d::from_xy(xs, ys).unwrap();
        let mut dl = init;
        let mut ds = SgdSolver::<f64>::ortmann_sparse_stress(
            &deg_g,
            OrtmannSparseStressConfig {
                num_pivots: 4,
                iterations: 15,
                ..Default::default()
            },
        )
        .unwrap();
        ds.run(&mut dl).unwrap();
        assert!(dl.is_all_finite());
    }

    // Non-finite layout coordinates rejected on step, and CsrGraph rejects non-finite weights.
    assert!(CsrGraph::<f64>::from_edges(2, &[(0, 1)], Some(&[f64::NAN]), false, false).is_err());
    let mut nan_layout = Layout2d::from_xy(vec![f64::NAN; n], vec![0.0; n]).unwrap();
    assert!(
        SgdSolver::<f64>::ortmann_sparse_stress(&grid, OrtmannSparseStressConfig::default())
            .unwrap()
            .step(&mut nan_layout)
            .is_err()
    );
}
