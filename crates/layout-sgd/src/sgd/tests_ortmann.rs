use std::collections::{HashMap, HashSet};

use egraph_core::{CoreError, CsrGraph, IterativeSolver, Layout2d, Scalar, SolverStatus};

use super::super::sgd::{
    OrtmannPivotStrategy, OrtmannSparseProblem, OrtmannSparseStressConfig, SgdPairKind,
    SgdPairRecord, SgdSolver, build_ortmann_sparse_problem,
};
use super::graph_helpers::{build_positive_canonical_csr, compute_mssp_dists};
use super::ortmann::select_ortmann_pivots_and_dists;
use super::test_support::{circle_layout, euclidean_apsp_stress, grid_graph};

fn verify_pivot_read_only<S: Scalar>(problem: &OrtmannSparseProblem<S>) {
    let pivot_set: HashSet<u32> = problem.pivots.iter().copied().collect();
    assert_eq!(pivot_set.len(), problem.pivots.len());
    for &p in &problem.pivots {
        assert!((p as usize) < problem.num_nodes);
    }
    for rec in &problem.one_sided {
        assert_eq!(rec.kind, SgdPairKind::Spring);
        assert!(pivot_set.contains(&rec.u), "one_sided u must be a pivot");
        assert!(
            !pivot_set.contains(&rec.v),
            "one_sided v must be a non-pivot (pivots are read-only)"
        );
        assert_eq!(rec.w_uv, S::ZERO, "pivot weight w_uv must be zero");
        assert!(rec.w_vu > S::ZERO, "non-pivot weight w_vu must be > 0");
    }
    let mut seen = HashSet::new();
    for rec in &problem.two_sided {
        assert_eq!(rec.kind, SgdPairKind::Spring);
        assert!(rec.u < rec.v, "two_sided pairs must be canonical u < v");
        assert!(seen.insert((rec.u, rec.v)), "duplicate two_sided pair");
        assert!(rec.w_uv > S::ZERO || rec.w_vu > S::ZERO);
    }

    if !problem.one_sided.is_empty() {
        let src: Vec<u32> = problem.one_sided.iter().map(|r| r.u).collect();
        let tgt: Vec<u32> = problem.one_sided.iter().map(|r| r.v).collect();
        let dij: Vec<f64> = problem.one_sided.iter().map(|r| r.d_uv.to_f64()).collect();
        let dji: Vec<f64> = problem.one_sided.iter().map(|r| r.d_vu.to_f64()).collect();
        let wij: Vec<f64> = problem.one_sided.iter().map(|r| r.w_uv.to_f64()).collect();
        let wji: Vec<f64> = problem.one_sided.iter().map(|r| r.w_vu.to_f64()).collect();
        let mut solver = SgdSolver::<Layout2d<f64>>::from_pairs(
            problem.num_nodes,
            &src,
            &tgt,
            &dij,
            &dji,
            &wij,
            &wji,
            problem.schedule_cfg.clone(),
            19,
        )
        .unwrap();
        let mut layout = circle_layout::<f64>(problem.num_nodes, 1.0);
        let init_x = layout.x().to_vec();
        let init_y = layout.y().to_vec();
        solver.step(&mut layout).unwrap();
        for &p in &problem.pivots {
            let idx = p as usize;
            assert_eq!(layout.x()[idx].to_bits(), init_x[idx].to_bits());
            assert_eq!(layout.y()[idx].to_bits(), init_y[idx].to_bits());
        }
        assert!((0..problem.num_nodes).any(|i| {
            !pivot_set.contains(&(i as u32))
                && (layout.x()[i].to_bits() != init_x[i].to_bits()
                    || layout.y()[i].to_bits() != init_y[i].to_bits())
        }));
    }
}

#[test]
fn test_ortmann_voronoi_tie_breaking_and_exact_weights_vs_bruteforce() {
    let edges = vec![(0, 1), (0, 2), (0, 3), (0, 4), (4, 5)];
    let g = CsrGraph::<f64>::from_edges(6, &edges, None, false, false).unwrap();

    for strategy in [
        OrtmannPivotStrategy::MaxMinSp,
        OrtmannPivotStrategy::MaxMinRandomSp,
        OrtmannPivotStrategy::KMeansSp,
    ] {
        let cfg = OrtmannSparseStressConfig::default()
            .with_num_pivots(2)
            .with_pivot_strategy(strategy)
            .with_iterations(50)
            .with_seed(7);
        let problem = build_ortmann_sparse_problem::<f64, f64>(&g, cfg.clone()).unwrap();
        verify_pivot_read_only(&problem);
        let solver = SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress(&g, cfg).unwrap();

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
        let mut expected_two_sided: Vec<(u32, u32, f64, f64)> = Vec::new();
        for &(u, v, d_uv) in &pos_edges {
            let w = 1.0 / (d_uv * d_uv);
            expected_two_sided.push((u, v, w, w));
        }
        let edge_set: HashSet<(u32, u32)> = pos_edges.iter().map(|&(u, v, _)| (u, v)).collect();
        let pivot_set: HashSet<u32> = pivots.iter().copied().collect();
        for k1 in 0..h {
            for k2 in (k1 + 1)..h {
                let (u, v, ku, kv) = if pivots[k1] < pivots[k2] {
                    (pivots[k1], pivots[k2], k1, k2)
                } else {
                    (pivots[k2], pivots[k1], k2, k1)
                };
                if edge_set.contains(&(u, v)) {
                    continue;
                }
                let d = piv_dists[ku * n + (v as usize)];
                if !d.is_finite() || d <= 0.0 {
                    continue;
                }
                let count_v = r_dists[kv].iter().filter(|&&dpj| 2.0 * dpj <= d).count();
                let count_u = r_dists[ku].iter().filter(|&&dpj| 2.0 * dpj <= d).count();
                if count_v > 0 || count_u > 0 {
                    expected_two_sided.push((
                        u,
                        v,
                        (count_v as f64) / (d * d),
                        (count_u as f64) / (d * d),
                    ));
                }
            }
        }
        let mut expected_one_sided: Vec<(u32, u32, f64, f64)> = Vec::new();
        for (k, &p) in pivots.iter().enumerate() {
            for i in 0..n {
                let i_u32 = i as u32;
                if pivot_set.contains(&i_u32) || edge_set.contains(&(p.min(i_u32), p.max(i_u32))) {
                    continue;
                }
                let dpi = piv_dists[k * n + i];
                if !dpi.is_finite() || dpi <= 0.0 {
                    continue;
                }
                let s_pi = r_dists[k].iter().filter(|&&dpj| 2.0 * dpj <= dpi).count();
                if s_pi > 0 {
                    expected_one_sided.push((p, i_u32, 0.0, (s_pi as f64) / (dpi * dpi)));
                }
            }
        }
        let mut expected_pairs = expected_two_sided.clone();
        expected_pairs.extend_from_slice(&expected_one_sided);

        assert_eq!(problem.two_sided.len(), expected_two_sided.len());
        assert_eq!(problem.one_sided.len(), expected_one_sided.len());
        assert_eq!(solver.num_springs(), expected_pairs.len());

        let mut actual_sorted: Vec<(u32, u32, f64, f64)> = solver
            .iter_pairs()
            .map(|p| (p.u, p.v, p.w_uv, p.w_vu))
            .collect();
        actual_sorted.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        expected_pairs.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        for (a, e) in actual_sorted.iter().zip(expected_pairs.iter()) {
            assert_eq!((a.0, a.1), (e.0, e.1));
            assert!((a.2 - e.2).abs() < 1e-12);
            assert!((a.3 - e.3).abs() < 1e-12);
        }
    }

    let edges2 = vec![(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (5, 6)];
    let g2 = CsrGraph::<f64>::from_edges(7, &edges2, None, false, false).unwrap();
    let mut solver_tie = SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress(
        &g2,
        OrtmannSparseStressConfig::default()
            .with_num_pivots(2)
            .with_pivot_strategy(OrtmannPivotStrategy::MaxMinSp)
            .with_iterations(60)
            .with_seed(1),
    )
    .unwrap();
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
        let cfg = OrtmannSparseStressConfig::default()
            .with_num_pivots(8)
            .with_pivot_strategy(strategy)
            .with_iterations(60)
            .with_seed(42);

        let mut s_run =
            SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress(&grid, cfg.clone()).unwrap();
        let mut s_step =
            SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress(&grid, cfg.clone()).unwrap();
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

        let mut s32_run =
            SgdSolver::<Layout2d<f32>>::ortmann_sparse_stress(&grid, cfg.clone()).unwrap();
        let mut s32_step = SgdSolver::<Layout2d<f32>>::ortmann_sparse_stress(&grid, cfg).unwrap();
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

    let cfg_a = OrtmannSparseStressConfig::default()
        .with_num_pivots(6)
        .with_seed(11)
        .with_iterations(30);
    let cfg_b = cfg_a.clone().with_seed(99);
    let mut sa1 = SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress(&grid, cfg_a.clone()).unwrap();
    let mut sa2 = SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress(&grid, cfg_a).unwrap();
    let mut sb = SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress(&grid, cfg_b).unwrap();
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

    let w_edges = vec![(0, 1), (1, 2), (2, 3), (0, 2)];
    let w_vals = vec![1.0, 1.0, 4.0, 10.0];
    let wg = CsrGraph::<f64>::from_edges(4, &w_edges, Some(&w_vals), false, false).unwrap();
    let (wsym, _) = build_positive_canonical_csr(&wg).unwrap();
    let w_dists = compute_mssp_dists(&wsym, &[0]).unwrap();
    assert!((w_dists[2] - 2.0).abs() < 1e-12);
    assert!((w_dists[3] - 6.0).abs() < 1e-12);
    let mut w_solver = SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress(
        &wg,
        OrtmannSparseStressConfig::default()
            .with_num_pivots(2)
            .with_iterations(40)
            .with_eta_max(Some(2.0))
            .with_eta_min(Some(0.05)),
    )
    .unwrap();
    assert!((w_solver.eta_max - 2.0).abs() < 1e-12);
    assert!((w_solver.eta_min - 0.05).abs() < 1e-12);
    let mut wl = circle_layout::<f64>(4, 1.0);
    w_solver.run(&mut wl).unwrap();
    assert!(wl.is_all_finite());

    let dir_edges = vec![(0, 0), (0, 1), (1, 0), (1, 2), (2, 3), (3, 1)];
    let dir_weights = vec![1.0, 5.0, 2.0, 3.0, 0.0, 4.0];
    let dg = CsrGraph::<f64>::from_edges(5, &dir_edges, Some(&dir_weights), true, true).unwrap();
    let mut d_solver = SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress(
        &dg,
        OrtmannSparseStressConfig::default()
            .with_num_pivots(3)
            .with_iterations(20),
    )
    .unwrap();
    let mut dl = circle_layout::<f64>(5, 1.0);
    d_solver.run(&mut dl).unwrap();
    assert!(dl.is_all_finite());

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
        let mut dl = Layout2d::from_xy(xs, ys).unwrap();
        let mut ds = SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress(
            &deg_g,
            OrtmannSparseStressConfig::default()
                .with_num_pivots(4)
                .with_iterations(15),
        )
        .unwrap();
        ds.run(&mut dl).unwrap();
        assert!(dl.is_all_finite());
    }

    assert!(CsrGraph::<f64>::from_edges(2, &[(0, 1)], Some(&[f64::NAN]), false, false).is_err());
    let mut nan_layout = Layout2d::from_xy(vec![f64::NAN; n], vec![0.0; n]).unwrap();
    assert!(
        SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress(
            &grid,
            OrtmannSparseStressConfig::default()
        )
        .unwrap()
        .step(&mut nan_layout)
        .is_err()
    );
}

#[test]
fn test_ortmann_sparse_problem_ir_contract_and_edge_cases() {
    let path_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)];
    let path_weights = [1.0_f64, 2.0, 0.5, 3.0, 1.5];
    let path =
        CsrGraph::<f64>::from_edges(6, &path_edges, Some(&path_weights), false, false).unwrap();
    let cfg = OrtmannSparseStressConfig::new()
        .with_num_pivots(3)
        .with_pivot_strategy(OrtmannPivotStrategy::MaxMinSp)
        .with_iterations(20)
        .with_eps(0.1)
        .with_seed(11);
    let prob = build_ortmann_sparse_problem::<f64, f64>(&path, cfg.clone()).unwrap();
    verify_pivot_read_only(&prob);

    let two_map: HashMap<(u32, u32), &SgdPairRecord<f64>> =
        prob.two_sided.iter().map(|r| ((r.u, r.v), r)).collect();
    for (&(u, v), &d) in path_edges.iter().zip(path_weights.iter()) {
        let rec = two_map.get(&(u.min(v), u.max(v))).unwrap();
        assert!((rec.d_uv - d).abs() < 1e-12 && (rec.d_vu - d).abs() < 1e-12);
        let ew = 1.0 / (d * d);
        assert!((rec.w_uv - ew).abs() < 1e-12 && (rec.w_vu - ew).abs() < 1e-12);
    }

    let pivot_set: HashSet<u32> = prob.pivots.iter().copied().collect();
    assert!(prob.two_sided.iter().any(|r| {
        pivot_set.contains(&r.u)
            && pivot_set.contains(&r.v)
            && !path.has_edge(r.u as usize, r.v as usize)
    }));

    let mut solver = SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress(&path, cfg.clone()).unwrap();
    let (emin, emax) = solver.eta_bounds();
    assert!((emax - 1.0 / prob.w_min).abs() < 1e-10);
    assert!((emin - prob.schedule_cfg.eps / prob.w_max).abs() < 1e-10);
    assert_eq!(
        solver.iter_pairs().count(),
        prob.two_sided.len() + prob.one_sided.len()
    );
    let mut l2 = circle_layout::<f64>(6, 1.0);
    assert_eq!(solver.run(&mut l2).unwrap(), SolverStatus::Completed);

    let prob_f32_a = build_ortmann_sparse_problem::<f64, f32>(&path, cfg.clone()).unwrap();
    let prob_f32_b = build_ortmann_sparse_problem::<f64, f32>(&path, cfg).unwrap();
    assert_eq!(prob_f32_a, prob_f32_b);
    verify_pivot_read_only(&prob_f32_a);

    let full_piv = build_ortmann_sparse_problem::<f64, f64>(
        &path,
        OrtmannSparseStressConfig::new().with_num_pivots(100),
    )
    .unwrap();
    assert_eq!(full_piv.pivots.len(), 6);
    assert!(full_piv.one_sided.is_empty());

    let disc = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)],
        None,
        false,
        false,
    )
    .unwrap();
    let disc_prob = build_ortmann_sparse_problem::<f64, f64>(
        &disc,
        OrtmannSparseStressConfig::new().with_num_pivots(2),
    )
    .unwrap();
    for r in disc_prob.one_sided.iter().chain(disc_prob.two_sided.iter()) {
        assert_eq!(r.u < 3, r.v < 3);
    }

    for n in [0usize, 1, 4] {
        let empty = CsrGraph::<f64>::from_edges(n, &[], None, false, false).unwrap();
        let ep = build_ortmann_sparse_problem::<f64, f64>(&empty, OrtmannSparseStressConfig::new())
            .unwrap();
        assert_eq!(ep.num_nodes, n);
        assert!(ep.one_sided.is_empty() && ep.two_sided.is_empty());
        assert_eq!((ep.w_min, ep.w_max), (0.0, 0.0));
    }

    assert!(matches!(
        build_ortmann_sparse_problem::<f64, f64>(
            &path,
            OrtmannSparseStressConfig::new().with_num_pivots(0),
        ),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        build_ortmann_sparse_problem::<f64, f64>(
            &path,
            OrtmannSparseStressConfig::new()
                .with_eta_min(Some(2.0))
                .with_eta_max(Some(1.0)),
        ),
        Err(CoreError::InvalidInput(_))
    ));
}
