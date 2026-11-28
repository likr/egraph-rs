//! Opaque-box E2E tests for `OrtmannSparseProblem`, `build_ortmann_sparse_problem`,
//! and Ortmann `SgdSolver` Problem IR accessors.

use std::collections::{HashMap, HashSet};

use egraph_core::{CoreError, CsrGraph, IterativeSolver, Layout2d, Scalar, SolverStatus};
use egraph_layout_sgd::{
    OrtmannPivotStrategy, OrtmannSparseProblem, OrtmannSparseStressConfig, SgdPairKind,
    SgdPairRecord, SgdSolver, build_ortmann_sparse_problem,
};

fn make_grid(rows: usize, cols: usize) -> CsrGraph<f64> {
    let n = rows * cols;
    let mut edges = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let u = (r * cols + c) as u32;
            if c + 1 < cols {
                edges.push((u, u + 1));
            }
            if r + 1 < rows {
                edges.push((u, u + cols as u32));
            }
        }
    }
    CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap()
}

fn make_barbell(clique: usize) -> CsrGraph<f64> {
    let n = 2 * clique;
    let mut edges = Vec::new();
    for i in 0..clique {
        for j in (i + 1)..clique {
            edges.push((i as u32, j as u32));
            edges.push(((clique + i) as u32, (clique + j) as u32));
        }
    }
    edges.push(((clique - 1) as u32, clique as u32));
    CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap()
}

fn make_star(leaves: usize) -> CsrGraph<f64> {
    let edges: Vec<(u32, u32)> = (1..=leaves).map(|v| (0, v as u32)).collect();
    CsrGraph::<f64>::from_edges(leaves + 1, &edges, None, false, false).unwrap()
}

fn canonical_pair_keys<S: Scalar>(
    records: &[SgdPairRecord<S>],
) -> Vec<(u32, u32, u8, u64, u64, u64, u64)> {
    let mut out: Vec<_> = records
        .iter()
        .map(|r| {
            let kind_tag = match r.kind {
                SgdPairKind::Spring => 0_u8,
                SgdPairKind::Hinge => 1_u8,
                _ => 2_u8,
            };
            let is_sym = r.d_uv.to_f64().to_bits() == r.d_vu.to_f64().to_bits()
                && r.w_uv.to_f64().to_bits() == r.w_vu.to_f64().to_bits();
            let (u, v) = if is_sym {
                (r.u.min(r.v), r.u.max(r.v))
            } else {
                (r.u, r.v)
            };
            (
                u,
                v,
                kind_tag,
                r.d_uv.to_f64().to_bits(),
                r.d_vu.to_f64().to_bits(),
                r.w_uv.to_f64().to_bits(),
                r.w_vu.to_f64().to_bits(),
            )
        })
        .collect();
    out.sort_unstable();
    out
}

fn verify_pivot_read_only(problem: &OrtmannSparseProblem<f64>) {
    let pivot_set: HashSet<u32> = problem.pivots.iter().copied().collect();
    assert_eq!(pivot_set.len(), problem.pivots.len());
    for pair in &problem.one_sided {
        assert_eq!(pair.kind, SgdPairKind::Spring);
        assert!(
            pivot_set.contains(&pair.u),
            "one_sided source {} must be a pivot",
            pair.u
        );
        assert!(
            !pivot_set.contains(&pair.v),
            "one_sided target {} must be a non-pivot (pivots are read-only)",
            pair.v
        );
        assert_eq!(pair.w_uv, 0.0, "pivot weight w_uv must be 0.0");
        assert!(pair.w_vu > 0.0);
    }
}

#[test]
fn ortmann_pivot_read_only_invariant_all_strategies() {
    let graphs = [make_grid(4, 4), make_barbell(5), make_star(10)];
    for g in &graphs {
        for strategy in [
            OrtmannPivotStrategy::MaxMinSp,
            OrtmannPivotStrategy::MaxMinRandomSp,
            OrtmannPivotStrategy::KMeansSp,
        ] {
            let cfg = OrtmannSparseStressConfig::new()
                .with_num_pivots(4)
                .with_pivot_strategy(strategy)
                .with_iterations(12)
                .with_seed(17);
            let problem = build_ortmann_sparse_problem::<f64, f64>(g, cfg).unwrap();
            assert_eq!(problem.num_nodes, g.num_nodes());
            verify_pivot_read_only(&problem);
        }
    }
}

#[test]
fn ortmann_pivot_pivot_pairs_merged_into_two_sided() {
    let g = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        None,
        false,
        false,
    )
    .unwrap();
    let cfg = OrtmannSparseStressConfig::new()
        .with_num_pivots(3)
        .with_pivot_strategy(OrtmannPivotStrategy::MaxMinSp)
        .with_seed(7);
    let problem = build_ortmann_sparse_problem::<f64, f64>(&g, cfg).unwrap();
    verify_pivot_read_only(&problem);

    let pivot_set: HashSet<u32> = problem.pivots.iter().copied().collect();
    let mut seen_pairs = HashSet::new();
    let mut pivot_pivot_non_edge_count = 0usize;
    for p in &problem.two_sided {
        let key = (p.u.min(p.v), p.u.max(p.v));
        assert!(
            seen_pairs.insert(key),
            "duplicate pair {key:?} in two_sided"
        );
        if pivot_set.contains(&p.u)
            && pivot_set.contains(&p.v)
            && !g.has_edge(p.u as usize, p.v as usize)
        {
            pivot_pivot_non_edge_count += 1;
            assert!(p.w_uv > 0.0 || p.w_vu > 0.0);
        }
    }
    assert!(
        pivot_pivot_non_edge_count >= 1,
        "expected merged non-edge pivot-pivot pair in two_sided"
    );
}

#[test]
fn ortmann_one_hop_edges_precedence_and_weights() {
    let edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)];
    let weights = [1.0_f64, 2.0, 0.5, 3.0, 1.5];
    let gw = CsrGraph::<f64>::from_edges(6, &edges, Some(&weights), false, false).unwrap();
    let cfg = OrtmannSparseStressConfig::new()
        .with_num_pivots(3)
        .with_pivot_strategy(OrtmannPivotStrategy::MaxMinSp)
        .with_seed(11);
    let problem = build_ortmann_sparse_problem::<f64, f64>(&gw, cfg).unwrap();

    let two_sided_map: HashMap<(u32, u32), &SgdPairRecord<f64>> = problem
        .two_sided
        .iter()
        .map(|p| ((p.u.min(p.v), p.u.max(p.v)), p))
        .collect();
    for (&(u, v), &d) in edges.iter().zip(weights.iter()) {
        let rec = two_sided_map
            .get(&(u.min(v), u.max(v)))
            .expect("1-hop edge missing from two_sided");
        assert!((rec.d_uv - d).abs() < 1e-12);
        assert!((rec.d_vu - d).abs() < 1e-12);
        let expected_w = 1.0 / (d * d);
        assert!((rec.w_uv - expected_w).abs() < 1e-12);
        assert!((rec.w_vu - expected_w).abs() < 1e-12);
    }
}

#[test]
fn ortmann_solver_consumes_ir_and_matches_exact_pair_records_f64_f32() {
    let g = make_grid(3, 4);
    let cfg = OrtmannSparseStressConfig::new()
        .with_num_pivots(4)
        .with_pivot_strategy(OrtmannPivotStrategy::KMeansSp)
        .with_iterations(25)
        .with_eps(0.1)
        .with_seed(42);
    let problem64 = build_ortmann_sparse_problem::<f64, f64>(&g, cfg.clone()).unwrap();
    let mut solver64 = SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress(&g, cfg.clone()).unwrap();

    let (emin, emax) = solver64.eta_bounds();
    assert!((emax - 1.0 / problem64.w_min).abs() < 1e-10);
    assert!((emin - problem64.schedule_cfg.eps / problem64.w_max).abs() < 1e-10);

    let solver_pairs64: Vec<_> = solver64.iter_pairs().collect();
    let mut expected_pairs64 = problem64.two_sided.clone();
    expected_pairs64.extend_from_slice(&problem64.one_sided);
    assert_eq!(
        canonical_pair_keys(&solver_pairs64),
        canonical_pair_keys(&expected_pairs64)
    );

    let mut layout64 = Layout2d::<f64>::init(g.num_nodes(), 5, 2.0).unwrap();
    assert_eq!(
        solver64.run(&mut layout64).unwrap(),
        SolverStatus::Completed
    );
    assert!(layout64.is_all_finite());

    let problem32 = build_ortmann_sparse_problem::<f64, f32>(&g, cfg.clone()).unwrap();
    let mut solver32 = SgdSolver::<Layout2d<f32>>::ortmann_sparse_stress(&g, cfg).unwrap();
    let solver_pairs32: Vec<_> = solver32.iter_pairs().collect();
    let mut expected_pairs32 = problem32.two_sided.clone();
    expected_pairs32.extend_from_slice(&problem32.one_sided);
    assert_eq!(
        canonical_pair_keys(&solver_pairs32),
        canonical_pair_keys(&expected_pairs32)
    );
    let mut layout32 = Layout2d::<f32>::init(g.num_nodes(), 5, 2.0).unwrap();
    assert_eq!(
        solver32.run(&mut layout32).unwrap(),
        SolverStatus::Completed
    );
    assert!(layout32.is_all_finite());
}

#[test]
fn ortmann_deterministic_reproducibility_f32_and_f64() {
    let g = make_grid(3, 3);
    let cfg = OrtmannSparseStressConfig::new()
        .with_num_pivots(4)
        .with_pivot_strategy(OrtmannPivotStrategy::MaxMinRandomSp)
        .with_seed(123);

    let p64_a = build_ortmann_sparse_problem::<f64, f64>(&g, cfg.clone()).unwrap();
    let p64_b = build_ortmann_sparse_problem::<f64, f64>(&g, cfg.clone()).unwrap();
    assert_eq!(p64_a.pivots, p64_b.pivots);
    assert_eq!(p64_a.one_sided, p64_b.one_sided);
    assert_eq!(p64_a.two_sided, p64_b.two_sided);
    assert_eq!(p64_a.w_min.to_bits(), p64_b.w_min.to_bits());
    assert_eq!(p64_a.w_max.to_bits(), p64_b.w_max.to_bits());

    let p32_a = build_ortmann_sparse_problem::<f64, f32>(&g, cfg.clone()).unwrap();
    let p32_b = build_ortmann_sparse_problem::<f64, f32>(&g, cfg).unwrap();
    assert_eq!(p32_a.pivots, p32_b.pivots);
    assert_eq!(p32_a.one_sided, p32_b.one_sided);
    assert_eq!(p32_a.two_sided, p32_b.two_sided);
}

#[test]
fn ortmann_empty_singleton_edgeless_and_zero_weight_graphs() {
    let cfg = OrtmannSparseStressConfig::new().with_num_pivots(4);
    for n in [0usize, 1, 5] {
        let g = CsrGraph::<f64>::from_edges(n, &[], None, false, false).unwrap();
        let prob = build_ortmann_sparse_problem::<f64, f64>(&g, cfg.clone()).unwrap();
        assert_eq!(prob.num_nodes, n);
        assert!(prob.one_sided.is_empty());
        assert!(prob.two_sided.is_empty());

        let solver = SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress(&g, cfg.clone()).unwrap();
        assert_eq!(solver.eta_bounds(), (0.0, 0.0));
        assert_eq!(solver.iter_pairs().count(), 0);
    }

    // All-zero-weight edges produce an empty positive-edge problem without division by zero.
    let g_zero =
        CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], Some(&[0.0, 0.0]), false, false).unwrap();
    let prob_zero = build_ortmann_sparse_problem::<f64, f64>(&g_zero, cfg.clone()).unwrap();
    assert!(prob_zero.one_sided.is_empty());
    assert!(prob_zero.two_sided.is_empty());
    let solver_zero =
        SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress(&g_zero, cfg.clone()).unwrap();
    assert_eq!(solver_zero.eta_bounds(), (0.0, 0.0));

    // Mixed zero-weight and positive-weight edges skip only zero-weight edges.
    let g_mixed =
        CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], Some(&[0.0, 2.0]), false, false).unwrap();
    let prob_mixed = build_ortmann_sparse_problem::<f64, f64>(&g_mixed, cfg).unwrap();
    assert_eq!(prob_mixed.two_sided.len(), 1);
    assert_eq!(
        (prob_mixed.two_sided[0].u, prob_mixed.two_sided[0].v),
        (1, 2)
    );
}

#[test]
fn ortmann_directed_graph_canonicalizes_and_converges() {
    let dg = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 0), (1, 2), (2, 3), (3, 4), (4, 5)],
        Some(&[3.0, 1.5, 2.0, 1.0, 2.5, 1.0]),
        true,
        false,
    )
    .unwrap();
    let cfg = OrtmannSparseStressConfig::new()
        .with_num_pivots(3)
        .with_pivot_strategy(OrtmannPivotStrategy::MaxMinSp)
        .with_iterations(15)
        .with_seed(19);
    let prob = build_ortmann_sparse_problem::<f64, f64>(&dg, cfg.clone()).unwrap();
    verify_pivot_read_only(&prob);
    let edge_01 = prob
        .two_sided
        .iter()
        .find(|p| (p.u.min(p.v), p.u.max(p.v)) == (0, 1))
        .expect("canonical edge (0, 1) present");
    assert!((edge_01.d_uv - 1.5).abs() < 1e-12);
    assert!((edge_01.w_uv - 1.0 / (1.5 * 1.5)).abs() < 1e-12);

    let mut solver = SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress(&dg, cfg).unwrap();
    let mut layout = Layout2d::<f64>::init(6, 7, 2.0).unwrap();
    assert_eq!(solver.run(&mut layout).unwrap(), SolverStatus::Completed);
    assert!(layout.is_all_finite());
}

#[test]
fn ortmann_disconnected_components_skip_unreachable_pairs() {
    // Two disjoint triangles: 0-1-2 and 3-4-5.
    let g = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)],
        None,
        false,
        false,
    )
    .unwrap();
    let cfg = OrtmannSparseStressConfig::new()
        .with_num_pivots(2)
        .with_seed(9);
    let prob = build_ortmann_sparse_problem::<f64, f64>(&g, cfg).unwrap();
    for p in prob.one_sided.iter().chain(prob.two_sided.iter()) {
        assert!(p.d_uv.is_finite() && p.d_uv > 0.0);
        assert!(p.w_uv.is_finite() && p.w_vu.is_finite());
        assert_eq!(p.u < 3, p.v < 3, "cross-component pair should not exist");
    }
}

#[test]
fn ortmann_num_pivots_ge_num_nodes_leaves_one_sided_empty() {
    let g = make_grid(2, 3);
    let cfg = OrtmannSparseStressConfig::new()
        .with_num_pivots(100)
        .with_pivot_strategy(OrtmannPivotStrategy::MaxMinSp)
        .with_seed(5);
    let prob = build_ortmann_sparse_problem::<f64, f64>(&g, cfg).unwrap();
    assert!(prob.pivots.len() <= g.num_nodes());
    if prob.pivots.len() == g.num_nodes() {
        assert!(prob.one_sided.is_empty());
    }
    verify_pivot_read_only(&prob);
}

#[test]
fn ortmann_invalid_config_and_bad_graph_weights_fail_fast() {
    let g = make_grid(2, 2);
    for bad_cfg in [
        OrtmannSparseStressConfig::new().with_num_pivots(0),
        OrtmannSparseStressConfig::new().with_eps(0.0),
        OrtmannSparseStressConfig::new().with_eta_max(Some(0.0)),
        OrtmannSparseStressConfig::new().with_eta_min(Some(0.0)),
        OrtmannSparseStressConfig::new()
            .with_eta_min(Some(2.0))
            .with_eta_max(Some(1.0)),
    ] {
        assert!(matches!(
            build_ortmann_sparse_problem::<f64, f64>(&g, bad_cfg.clone()),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress(&g, bad_cfg),
            Err(CoreError::InvalidInput(_))
        ));
    }

    assert!(matches!(
        build_ortmann_sparse_problem::<f64, f64>(
            &g,
            OrtmannSparseStressConfig::new().with_eps(f64::NAN)
        ),
        Err(CoreError::NonFiniteValue { .. })
    ));

    let g_neg = CsrGraph::<f64>::from_edges(2, &[(0, 1)], Some(&[-1.0]), false, false).unwrap();
    assert!(matches!(
        build_ortmann_sparse_problem::<f64, f64>(&g_neg, OrtmannSparseStressConfig::new()),
        Err(CoreError::InvalidInput(_))
    ));

    let g_nan = CsrGraph::<f64>::from_edges(2, &[(0, 1)], Some(&[f64::NAN]), false, false).unwrap();
    assert!(matches!(
        build_ortmann_sparse_problem::<f64, f64>(&g_nan, OrtmannSparseStressConfig::new()),
        Err(CoreError::NonFiniteValue { .. })
    ));
}

#[test]
fn ortmann_solver_eta_overrides_reflect_in_eta_bounds() {
    let g = make_grid(3, 3);
    let cfg = OrtmannSparseStressConfig::new()
        .with_num_pivots(3)
        .with_iterations(10)
        .with_eta_max(Some(4.5))
        .with_eta_min(Some(0.15))
        .with_seed(42);
    let solver = SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress(&g, cfg).unwrap();
    let (emin, emax) = solver.eta_bounds();
    assert!((emin - 0.15).abs() < 1e-12);
    assert!((emax - 4.5).abs() < 1e-12);
}
