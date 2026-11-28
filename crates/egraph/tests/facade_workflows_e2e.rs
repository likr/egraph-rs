//! Opaque-box E2E workflows through the `egraph` facade covering core-only
//! minimal-feature compatibility, graph permutation, external IR stepping,
//! and zero-copy solver inspection.

#[cfg(all(feature = "layout", feature = "metrics"))]
use egraph::core::DeterministicRng;
use egraph::prelude::*;

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

#[cfg(all(feature = "layout", feature = "metrics"))]
fn make_clustered_sbm(communities: usize, per_comm: usize, seed: u64) -> CsrGraph<f64> {
    let n = communities * per_comm;
    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut edges = Vec::new();
    for c in 0..communities {
        let base = c * per_comm;
        for i in 0..per_comm {
            let u = (base + i) as u32;
            let v = (base + (i + 1) % per_comm) as u32;
            edges.push((u, v));
            for j in (i + 2)..per_comm {
                if rng.next_f64() < 0.6 {
                    edges.push((u, (base + j) as u32));
                }
            }
        }
    }
    for c in 0..communities {
        let next_c = (c + 1) % communities;
        edges.push(((c * per_comm) as u32, (next_c * per_comm) as u32));
    }
    CsrGraph::<f64>::from_edges(n, &edges, None, false, true).unwrap()
}

#[cfg(all(feature = "layout", feature = "metrics"))]
fn deterministic_permutation(n: usize, seed: u64) -> Vec<u32> {
    let mut perm: Vec<u32> = (0..n as u32).collect();
    let mut rng = DeterministicRng::from_seed_u64(seed);
    rng.shuffle(&mut perm);
    perm
}

#[test]
fn minimal_feature_core_facade_radial_permutation_and_errors() {
    let g = make_grid_csr(2, 3);
    let perm = [4_u32, 1, 5, 0, 3, 2];
    let gp = g.permute_nodes(&perm).unwrap();
    assert_eq!(gp.num_edges(), g.num_edges());

    let mut l2 = Layout2d::<f64>::init(6, 17, 2.0).unwrap();
    let d_before = l2.distance(0, 1);
    l2.move_pair_radial(0, 1, 0.1, 0.1).unwrap();
    assert!(l2.distance(0, 1) < d_before || d_before <= 1e-15);

    let l2_rt = l2.permute(&perm).unwrap().unpermute(&perm).unwrap();
    assert_eq!(l2_rt, l2);

    let l3 = Layout3d::<f64>::init(6, 19, 2.0).unwrap();
    assert_eq!(l3.permute(&perm).unwrap().unpermute(&perm).unwrap(), l3);

    let lnd = LayoutNd::<f64>::init(6, 4, 23, 2.0).unwrap();
    assert_eq!(lnd.permute(&perm).unwrap().unpermute(&perm).unwrap(), lnd);

    let err_b = CoreError::backend_unavailable("no device");
    let err_d = CoreError::device_limit_exceeded("limit");
    assert!(matches!(err_b, CoreError::BackendUnavailable(_)));
    assert!(matches!(err_d, CoreError::DeviceLimitExceeded(_)));
}

#[cfg(all(feature = "layout", feature = "metrics"))]
#[test]
fn workflow_clustered_permutation_external_ir_radial_step_and_metrics() {
    use egraph::layout::sgd::{
        OrtmannPivotStrategy, OrtmannSparseStressConfig, SgdPairKind, SgdPairRecord, SgdSolver,
        build_ortmann_sparse_problem,
    };
    use egraph::metrics::{neighborhood_preservation_2hop, scale_normalized_stress};

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

    let g = make_clustered_sbm(4, 10, 101);
    let n = g.num_nodes();
    let perm = deterministic_permutation(n, 202);
    let g_perm = g.permute_nodes(&perm).unwrap();

    let l_init = Layout2d::<f64>::init(n, 303, 5.0).unwrap();
    let stress_before = scale_normalized_stress(&g, &l_init).unwrap();

    let mut l_perm = l_init.permute(&perm).unwrap();
    let ort_cfg = OrtmannSparseStressConfig::new()
        .with_num_pivots(10)
        .with_pivot_strategy(OrtmannPivotStrategy::MaxMinSp)
        .with_iterations(40)
        .with_seed(42);
    let problem = build_ortmann_sparse_problem::<f64, f64>(&g_perm, ort_cfg.clone()).unwrap();
    let solver = SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress(&g_perm, ort_cfg).unwrap();
    let (emin, emax) = solver.eta_bounds();
    let sched = solver.schedule_config().clone();
    let pairs: Vec<_> = solver.iter_pairs().collect();
    let mut expected = problem.two_sided.clone();
    expected.extend_from_slice(&problem.one_sided);
    assert_eq!(canonical_keys(&pairs), canonical_keys(&expected));

    for t in 0..sched.iterations {
        let eta = sched.evaluate(t, emin, emax);
        for rec in &pairs {
            assert_eq!(rec.kind, SgdPairKind::Spring);
            let d_cur = l_perm.distance(rec.u as usize, rec.v as usize);
            if d_cur <= 1e-12 {
                continue;
            }
            let mu_u = (eta * rec.w_uv).min(1.0);
            let mu_v = (eta * rec.w_vu).min(1.0);
            let dr_u = 0.5 * mu_u * (d_cur - rec.d_uv);
            let dr_v = 0.5 * mu_v * (d_cur - rec.d_vu);
            l_perm
                .move_pair_radial(rec.u as usize, rec.v as usize, dr_u, dr_v)
                .unwrap();
        }
    }

    let l_final = l_perm.unpermute(&perm).unwrap();
    assert!(l_final.is_all_finite());
    let stress_after = scale_normalized_stress(&g, &l_final).unwrap();
    let np2_after = neighborhood_preservation_2hop(&g, &l_final).unwrap();
    assert!(
        stress_after < stress_before,
        "expected stress improvement: before={stress_before}, after={stress_after}"
    );
    assert!(np2_after > 0.35, "expected NP2 > 0.35, got {np2_after}");
}

#[cfg(all(feature = "layout", feature = "metrics"))]
#[test]
fn workflow_multi_solver_zero_copy_ir_and_custom_ib_umap_plugin() {
    use egraph::layout::force::{ForceAtlas2Config, ForceAtlas2Solver};
    use egraph::layout::neighbor::{
        BarnesHut, GuMapConfig, GuMapSolver, IbUmapBackend2d, IbUmapConfig, IbUmapRepulsion,
        IbUmapSolver, TFdpConfig, TFdpSolver, TsNetConfig, TsNetSolver, TsNetStage,
    };
    use egraph::metrics::neighborhood_preservation_2hop;

    #[derive(Clone)]
    struct InverseSquarePlugin;

    impl IbUmapRepulsion<Layout2d<f64>> for InverseSquarePlugin {
        type State = usize;

        fn validate_config(&self) -> CoreResult<()> {
            Ok(())
        }

        fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
            Ok(num_nodes)
        }

        fn compute_repulsion(
            &self,
            _state: &mut Self::State,
            layout: &Layout2d<f64>,
            a: f64,
            b: f64,
            c_k: f64,
            rep_forces: &mut [f64],
        ) -> CoreResult<()> {
            let n = layout.len();
            let two_b_ck = 2.0 * b * c_k;
            for i in 0..n {
                let mut fx = 0.0;
                let mut fy = 0.0;
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let dx = layout.x()[i] - layout.x()[j];
                    let dy = layout.y()[i] - layout.y()[j];
                    let d2 = dx * dx + dy * dy;
                    let k = two_b_ck / ((0.05 + d2) * (1.0 + a * d2.powf(b)));
                    fx += k * dx;
                    fy += k * dy;
                }
                rep_forces[2 * i] = fx;
                rep_forces[2 * i + 1] = fy;
            }
            Ok(())
        }
    }

    let g = make_grid_csr(5, 5);
    let n = g.num_nodes();
    let base = Layout2d::<f64>::init(n, 42, 2.0).unwrap();

    let mut ts = TsNetSolver::<Layout2d<f64>>::pure_2hop(
        &g,
        TsNetConfig::pure_2hop().with_stages(vec![
            TsNetStage::new().with_iterations(25).with_exaggeration(2.0),
            TsNetStage::new().with_iterations(25).with_exaggeration(1.0),
        ]),
    )
    .unwrap();
    assert_eq!(ts.probabilities_csr().0.len(), n + 1);
    let mut l_ts = base.clone();
    assert_eq!(ts.run(&mut l_ts).unwrap(), SolverStatus::Completed);
    assert!(neighborhood_preservation_2hop(&g, &l_ts).unwrap() > 0.5);

    let mut tf =
        TFdpSolver::<Layout2d<f64>>::new(&g, TFdpConfig::new().with_iterations(40)).unwrap();
    assert_eq!(tf.edge_csr().0.len(), n + 1);
    let mut l_tf = base.clone();
    assert_eq!(tf.run(&mut l_tf).unwrap(), SolverStatus::Completed);
    assert!(l_tf.is_all_finite());

    let mut ib = IbUmapSolver::<Layout2d<f64>, IbUmapBackend2d>::new_with_backend(
        &g,
        IbUmapConfig::new().with_iterations(40).with_seed(42),
        IbUmapBackend2d::BarnesHut(BarnesHut::new(0.5)),
    )
    .unwrap();
    assert_eq!(ib.edge_csr().0.len(), n + 1);
    assert_eq!(ib.hub_eta().len(), n);
    assert_eq!(ib.ab_params(), ib.ab());
    let mut l_ib = base.clone();
    assert_eq!(ib.run(&mut l_ib).unwrap(), SolverStatus::Completed);
    assert!(l_ib.is_all_finite());

    let mut ib_plugin = IbUmapSolver::<Layout2d<f64>, InverseSquarePlugin>::new_with_backend(
        &g,
        IbUmapConfig::new().with_iterations(15).with_seed(42),
        InverseSquarePlugin,
    )
    .unwrap();
    let mut l_plugin = base.clone();
    assert_eq!(
        ib_plugin.run(&mut l_plugin).unwrap(),
        SolverStatus::Completed
    );
    assert!(l_plugin.is_all_finite());

    let mut fa2 =
        ForceAtlas2Solver::<Layout2d<f64>>::new(&g, ForceAtlas2Config::new().with_iterations(40))
            .unwrap();
    assert_eq!(fa2.edges().0.len(), g.num_edges());
    assert_eq!(fa2.masses().len(), n);
    let mut l_fa2 = base.clone();
    assert_eq!(fa2.run(&mut l_fa2).unwrap(), SolverStatus::Completed);
    assert!(l_fa2.is_all_finite());

    let mut gu =
        GuMapSolver::<Layout2d<f64>>::new(&g, GuMapConfig::new().with_iterations(40)).unwrap();
    assert_eq!(gu.edges().0.len(), gu.num_edges());
    assert_eq!(gu.ab_params(), gu.ab());
    let mut l_gu = base;
    assert_eq!(gu.run(&mut l_gu).unwrap(), SolverStatus::Completed);
    assert!(l_gu.is_all_finite());
}

#[cfg(all(feature = "layout", feature = "metrics"))]
#[test]
fn workflow_multi_dimension_permutation_layout_equivalence() {
    use egraph::layout::force::{ForceAtlas2Config, ForceAtlas2Solver};
    use egraph::layout::neighbor::{Exact, TFdpConfig, TFdpSolver};
    use egraph::layout::sgd::{SgdScheduleConfig, SgdSolver};
    use egraph::metrics::{neighborhood_preservation_2hop, scale_normalized_stress};

    let g = make_grid_csr(3, 3);
    let n = g.num_nodes();
    let perm = deterministic_permutation(n, 777);
    let gp = g.permute_nodes(&perm).unwrap();

    // 2D via TFdp Exact
    let l2_base = Layout2d::<f64>::init(n, 11, 2.0).unwrap();
    let mut l2 = l2_base.clone();
    let mut l2_p = l2_base.permute(&perm).unwrap();
    TFdpSolver::<Layout2d<f64>, Exact>::new(&g, TFdpConfig::new().with_iterations(8))
        .unwrap()
        .run(&mut l2)
        .unwrap();
    TFdpSolver::<Layout2d<f64>, Exact>::new(&gp, TFdpConfig::new().with_iterations(8))
        .unwrap()
        .run(&mut l2_p)
        .unwrap();
    let l2_unp = l2_p.unpermute(&perm).unwrap();
    assert!(
        (scale_normalized_stress(&g, &l2).unwrap() - scale_normalized_stress(&g, &l2_unp).unwrap())
            .abs()
            < 1e-9
    );

    // 3D via ForceAtlas2 Exact
    let l3_base = Layout3d::<f64>::init(n, 13, 2.0).unwrap();
    let mut l3 = l3_base.clone();
    let mut l3_p = l3_base.permute(&perm).unwrap();
    ForceAtlas2Solver::<Layout3d<f64>, Exact>::with_backend(
        &g,
        ForceAtlas2Config::new().with_iterations(8),
        Exact::new(),
    )
    .unwrap()
    .run(&mut l3)
    .unwrap();
    ForceAtlas2Solver::<Layout3d<f64>, Exact>::with_backend(
        &gp,
        ForceAtlas2Config::new().with_iterations(8),
        Exact::new(),
    )
    .unwrap()
    .run(&mut l3_p)
    .unwrap();
    let l3_unp = l3_p.unpermute(&perm).unwrap();
    assert!(
        (neighborhood_preservation_2hop(&g, &l3).unwrap()
            - neighborhood_preservation_2hop(&g, &l3_unp).unwrap())
        .abs()
            < 1e-12
    );

    // ND (dims = 5) via SgdSolver::full
    let lnd_base = LayoutNd::<f64>::init(n, 5, 19, 2.0).unwrap();
    let mut lnd_p = lnd_base.permute(&perm).unwrap();
    SgdSolver::<LayoutNd<f64>>::full(&gp, SgdScheduleConfig::new(10), 42)
        .unwrap()
        .run(&mut lnd_p)
        .unwrap();
    let lnd_unp = lnd_p.unpermute(&perm).unwrap();
    assert!(scale_normalized_stress(&g, &lnd_unp).unwrap().is_finite());
}
