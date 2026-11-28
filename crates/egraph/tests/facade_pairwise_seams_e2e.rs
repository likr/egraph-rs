//! Pairwise cross-feature integration E2E tests through the `egraph` facade.

#[cfg(all(feature = "layout", feature = "metrics"))]
use egraph::core::DeterministicRng;
#[cfg(feature = "layout")]
use egraph::prelude::*;

#[cfg(feature = "layout")]
fn make_grid_csr(rows: usize, cols: usize) -> CsrGraph<f64> {
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

#[cfg(feature = "layout")]
#[test]
fn pairwise_ib_umap_ir_invariant_across_backends_and_execution() {
    use egraph::layout::neighbor::{BarnesHut, Exact, IbFft, IbUmapConfig, IbUmapSolver};

    let g = make_grid_csr(4, 4);
    let cfg = IbUmapConfig::new().with_iterations(4).with_seed(42);

    let mut s_fft =
        IbUmapSolver::<Layout2d<f64>, IbFft>::new_with_backend(&g, cfg.clone(), IbFft::new())
            .unwrap();
    let mut s_bh = IbUmapSolver::<Layout2d<f64>, BarnesHut>::new_with_backend(
        &g,
        cfg.clone(),
        BarnesHut::new(0.5),
    )
    .unwrap();
    let mut s_ex =
        IbUmapSolver::<Layout2d<f64>, Exact>::new_with_backend(&g, cfg, Exact::new()).unwrap();

    let (ip_f, idx_f, w_f) = s_fft.edge_csr();
    let (ip_b, idx_b, w_b) = s_bh.edge_csr();
    let (ip_e, idx_e, w_e) = s_ex.edge_csr();
    assert_eq!(ip_f, ip_b);
    assert_eq!(ip_b, ip_e);
    assert_eq!(idx_f, idx_b);
    assert_eq!(idx_b, idx_e);
    assert_eq!(w_f, w_b);
    assert_eq!(w_b, w_e);
    assert_eq!(s_fft.hub_eta(), s_bh.hub_eta());
    assert_eq!(s_bh.hub_eta(), s_ex.hub_eta());
    assert_eq!(s_fft.ab_params(), s_bh.ab_params());
    assert_eq!(s_bh.ab_params(), s_ex.ab_params());

    let snapshot_w = w_e.to_vec();
    let snapshot_eta = s_ex.hub_eta().to_vec();
    let mut layout = Layout2d::<f64>::init(16, 3, 1.0).unwrap();
    s_fft.run(&mut layout.clone()).unwrap();
    s_bh.run(&mut layout.clone()).unwrap();
    s_ex.run(&mut layout).unwrap();

    assert_eq!(s_ex.edge_csr().2, snapshot_w.as_slice());
    assert_eq!(s_ex.hub_eta(), snapshot_eta.as_slice());
}

#[cfg(feature = "layout")]
#[test]
fn pairwise_pair_buffer_condition_weights_roundtrip_into_solver_ir() {
    use egraph::layout::sgd::{SgdPairBuffer, SgdScheduleConfig, SgdSolver};

    let mut buf = SgdPairBuffer::<f64>::new();
    buf.push_spring(0, 1, 1.0, 500.0).unwrap();
    buf.push_spring(1, 2, 1.0, 5.0).unwrap();
    buf.push_hinge(0, 2, 1.8, 1.0).unwrap();
    buf.condition_weights(10.0).unwrap();
    let captured: Vec<_> = buf.iter().collect();

    let solver =
        SgdSolver::<Layout2d<f64>>::from_pair_buffer(3, buf, SgdScheduleConfig::new(5), 42)
            .unwrap();
    assert_eq!(solver.iter_pairs().collect::<Vec<_>>(), captured);
}

#[cfg(feature = "layout")]
#[test]
fn pairwise_ortmann_problem_ir_schedule_and_solver_parity() {
    use egraph::layout::sgd::{
        OrtmannPivotStrategy, OrtmannSparseStressConfig, SgdPairKind, SgdPairRecord, SgdSolver,
        build_ortmann_sparse_problem,
    };
    use std::collections::HashSet;

    fn canonical_keys(records: &[SgdPairRecord<f64>]) -> Vec<(u32, u32, u8, u64, u64, u64, u64)> {
        let mut out: Vec<_> = records
            .iter()
            .map(|r| {
                let tag = u8::from(r.kind != SgdPairKind::Spring);
                let sym =
                    r.d_uv.to_bits() == r.d_vu.to_bits() && r.w_uv.to_bits() == r.w_vu.to_bits();
                let (u, v) = if sym {
                    (r.u.min(r.v), r.u.max(r.v))
                } else {
                    (r.u, r.v)
                };
                (
                    u,
                    v,
                    tag,
                    r.d_uv.to_bits(),
                    r.d_vu.to_bits(),
                    r.w_uv.to_bits(),
                    r.w_vu.to_bits(),
                )
            })
            .collect();
        out.sort_unstable();
        out
    }

    let g = make_grid_csr(3, 4);
    let cfg = OrtmannSparseStressConfig::new()
        .with_num_pivots(4)
        .with_pivot_strategy(OrtmannPivotStrategy::MaxMinSp)
        .with_iterations(12)
        .with_seed(17);
    let problem = build_ortmann_sparse_problem::<f64, f64>(&g, cfg.clone()).unwrap();
    let mut solver = SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress(&g, cfg).unwrap();

    let pivot_set: HashSet<u32> = problem.pivots.iter().copied().collect();
    for p in &problem.one_sided {
        assert!(!pivot_set.contains(&p.v));
        assert_eq!(p.w_uv, 0.0);
    }

    let solver_pairs: Vec<_> = solver.iter_pairs().collect();
    let mut expected_pairs = problem.two_sided.clone();
    expected_pairs.extend_from_slice(&problem.one_sided);
    assert_eq!(
        canonical_keys(&solver_pairs),
        canonical_keys(&expected_pairs)
    );

    let (emin, emax) = solver.eta_bounds();
    let mut layout = Layout2d::<f64>::init(g.num_nodes(), 8, 2.0).unwrap();
    for t in 0..solver.schedule_config().iterations {
        assert_eq!(
            solver.schedule_config().evaluate(t, emin, emax).to_bits(),
            solver.current_eta().to_bits()
        );
        solver.step(&mut layout).unwrap();
    }
}

#[cfg(all(feature = "layout", feature = "metrics"))]
#[test]
fn pairwise_iter_pairs_drives_move_pair_radial_across_euclidean_and_manifolds() {
    use egraph::layout::sgd::{SgdPairBuffer, SgdPairKind, SgdScheduleConfig, SgdSolver};
    use egraph::metrics::{geodesic_distortion, scale_normalized_stress};

    let g = make_grid_csr(3, 3);
    let n = g.num_nodes();
    let mut buf = SgdPairBuffer::<f64>::new();
    for (u, v) in g.edges() {
        buf.push_spring(u, v, 0.4, 1.0).unwrap();
    }
    buf.push_hinge(0, 8, 0.8, 0.5).unwrap();
    let solver =
        SgdSolver::<Layout2d<f64>>::from_pair_buffer(n, buf, SgdScheduleConfig::new(20), 11)
            .unwrap();
    let pairs: Vec<_> = solver.iter_pairs().collect();
    let (emin, emax) = solver.eta_bounds();
    let sched = solver.schedule_config().clone();

    let mut l2 = Layout2d::<f64>::init(n, 21, 3.0).unwrap();
    let s_before = scale_normalized_stress(&g, &l2).unwrap();
    for t in 0..sched.iterations {
        let eta = sched.evaluate(t, emin, emax);
        for rec in &pairs {
            let d = l2.distance(rec.u as usize, rec.v as usize);
            if rec.kind == SgdPairKind::Hinge && d >= rec.d_uv {
                continue;
            }
            let mu = (eta * rec.w_uv).min(1.0);
            let dr = 0.5 * mu * (d - rec.d_uv);
            l2.move_pair_radial(rec.u as usize, rec.v as usize, dr, dr)
                .unwrap();
        }
    }
    assert!(scale_normalized_stress(&g, &l2).unwrap() < s_before);

    for kind in [ManifoldKind::HyperbolicPoincare, ManifoldKind::Torus2d] {
        let mut ml = ManifoldLayout2d::<f64>::init(kind, n, 31, 0.3).unwrap();
        for t in 0..10 {
            let eta = sched.evaluate(t, emin, emax);
            for rec in &pairs {
                let d = ml.distance(rec.u as usize, rec.v as usize);
                if rec.kind == SgdPairKind::Hinge && d >= rec.d_uv {
                    continue;
                }
                let mu = (eta * rec.w_uv).min(1.0);
                let dr = 0.25 * mu * (d - rec.d_uv);
                ml.move_pair_radial(rec.u as usize, rec.v as usize, dr, dr)
                    .unwrap();
            }
        }
        assert!(geodesic_distortion(&g, &ml).unwrap().is_finite());
    }
}

#[cfg(all(feature = "layout", feature = "metrics"))]
#[test]
fn pairwise_permute_before_ortmann_and_unpermute_preserves_stress() {
    use egraph::layout::sgd::{OrtmannSparseStressConfig, SgdSolver};
    use egraph::metrics::scale_normalized_stress;

    let g = make_grid_csr(3, 4);
    let n = g.num_nodes();
    let mut perm: Vec<u32> = (0..n as u32).collect();
    DeterministicRng::from_seed_u64(55).shuffle(&mut perm);

    let gp = g.permute_nodes(&perm).unwrap();
    let mut lp = Layout2d::<f64>::init(n, 7, 3.0)
        .unwrap()
        .permute(&perm)
        .unwrap();

    let cfg = OrtmannSparseStressConfig::new()
        .with_num_pivots(4)
        .with_iterations(25)
        .with_seed(42);
    let mut solver = SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress(&gp, cfg).unwrap();
    assert_eq!(solver.run(&mut lp).unwrap(), SolverStatus::Completed);
    let stress_perm = scale_normalized_stress(&gp, &lp).unwrap();
    let l_out = lp.unpermute(&perm).unwrap();
    let stress_orig = scale_normalized_stress(&g, &l_out).unwrap();
    assert!((stress_perm - stress_orig).abs() < 1e-11);
}

#[cfg(feature = "layout")]
#[test]
fn pairwise_permutation_preserves_ir_correspondence_across_solvers() {
    use egraph::layout::force::{ForceAtlas2Config, ForceAtlas2Solver};
    use egraph::layout::neighbor::{
        GuMapConfig, GuMapSolver, IbUmapConfig, IbUmapSolver, TFdpConfig, TFdpSolver, TsNetConfig,
        TsNetSolver, TsNetStage,
    };

    let g = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 2)],
        Some(&[1.0, 2.0, 1.5, 2.5, 3.0, 1.2]),
        false,
        false,
    )
    .unwrap();
    let perm = [3_u32, 5, 0, 4, 1, 2];
    let gp = g.permute_nodes(&perm).unwrap();

    let ib =
        IbUmapSolver::<Layout2d<f64>>::new(&g, IbUmapConfig::new().with_iterations(2)).unwrap();
    let ib_p =
        IbUmapSolver::<Layout2d<f64>>::new(&gp, IbUmapConfig::new().with_iterations(2)).unwrap();
    for old_i in 0..6 {
        let new_i = perm[old_i] as usize;
        assert!((ib.hub_eta()[old_i] - ib_p.hub_eta()[new_i]).abs() < 1e-12);
    }

    let fa = ForceAtlas2Solver::<Layout2d<f64>>::new(&g, ForceAtlas2Config::new()).unwrap();
    let fa_p = ForceAtlas2Solver::<Layout2d<f64>>::new(&gp, ForceAtlas2Config::new()).unwrap();
    for old_i in 0..6 {
        let new_i = perm[old_i] as usize;
        assert!((fa.masses()[old_i] - fa_p.masses()[new_i]).abs() < 1e-12);
    }

    let ts = TsNetSolver::<Layout2d<f64>>::pure_2hop(
        &g,
        TsNetConfig::pure_2hop().with_stages(vec![TsNetStage::new().with_iterations(1)]),
    )
    .unwrap();
    let ts_p = TsNetSolver::<Layout2d<f64>>::pure_2hop(
        &gp,
        TsNetConfig::pure_2hop().with_stages(vec![TsNetStage::new().with_iterations(1)]),
    )
    .unwrap();
    assert_eq!(
        ts.probabilities_csr().2.len(),
        ts_p.probabilities_csr().2.len()
    );

    let tf = TFdpSolver::<Layout2d<f64>>::new(&g, TFdpConfig::new()).unwrap();
    let tf_p = TFdpSolver::<Layout2d<f64>>::new(&gp, TFdpConfig::new()).unwrap();
    assert_eq!(tf.edge_csr().2.len(), tf_p.edge_csr().2.len());

    let gu = GuMapSolver::<Layout2d<f64>>::new(&g, GuMapConfig::new()).unwrap();
    let gu_p = GuMapSolver::<Layout2d<f64>>::new(&gp, GuMapConfig::new()).unwrap();
    assert_eq!(gu.edges().2.len(), gu_p.edges().2.len());
}

#[cfg(feature = "layout")]
#[test]
fn pairwise_backend_error_variants_distinct_from_ib_umap_validation_errors() {
    use egraph::layout::neighbor::{BarnesHut, IbUmapConfig, IbUmapSolver};

    let g = make_grid_csr(2, 3);
    let err_invalid = IbUmapSolver::<Layout2d<f64>, BarnesHut>::new_with_backend(
        &g,
        IbUmapConfig::new(),
        BarnesHut::new(-0.25),
    )
    .unwrap_err();
    let err_backend = CoreError::backend_unavailable("no device");
    let err_device = CoreError::device_limit_exceeded("limit");

    assert_ne!(err_invalid, err_backend);
    assert_ne!(err_invalid, err_device);
    assert_ne!(err_backend, err_device);
    assert!(matches!(err_invalid, CoreError::InvalidInput(_)));
}
