use egraph_core::{CsrGraph, ManifoldKind};

use super::*;

fn make_complete_graph(n: usize) -> CsrGraph<f64> {
    let mut edges = Vec::new();
    for u in 0..n {
        for v in (u + 1)..n {
            edges.push((u as u32, v as u32));
        }
    }
    CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
}

fn make_cycle_graph(n: usize) -> CsrGraph<f64> {
    let edges: Vec<(u32, u32)> = (0..n).map(|i| (i as u32, ((i + 1) % n) as u32)).collect();
    CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
}

fn make_torus_graph(rows: usize, cols: usize) -> CsrGraph<f64> {
    let mut edges = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let u = (r * cols + c) as u32;
            let r_next = (r * cols + ((c + 1) % cols)) as u32;
            let d_next = (((r + 1) % rows) * cols + c) as u32;
            edges.push((u, r_next));
            edges.push((u, d_next));
        }
    }
    CsrGraph::from_edges(rows * cols, &edges, None, false, true).unwrap()
}

fn make_star_graph(n: usize) -> CsrGraph<f64> {
    let edges: Vec<(u32, u32)> = (1..n).map(|i| (0, i as u32)).collect();
    CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
}

fn make_barbell_graph(clique_size: usize) -> CsrGraph<f64> {
    let mut edges = Vec::new();
    for u in 0..clique_size {
        for v in (u + 1)..clique_size {
            edges.push((u as u32, v as u32));
        }
    }
    let off = clique_size;
    for u in 0..clique_size {
        for v in (u + 1)..clique_size {
            edges.push(((off + u) as u32, (off + v) as u32));
        }
    }
    edges.push(((clique_size - 1) as u32, off as u32));
    CsrGraph::from_edges(2 * clique_size, &edges, None, false, false).unwrap()
}

#[test]
fn test_exact_analytical_resistances_and_curvatures() {
    // Complete graph K_8: R_e = 2/n, p_i = 1/n, kappa = 2
    let k8 = make_complete_graph(8);
    let res_k8 = resistance_curvature_flow(
        &k8,
        &ResistanceCurvatureConfig {
            steps: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(res_k8.used_exact);
    assert_eq!(res_k8.num_components, 1);
    assert!(res_k8.foster_rel_err < 1e-12);
    for &r in &res_k8.effective_resistances {
        assert!((r - 0.25).abs() < 1e-11);
    }
    for &p in &res_k8.vertex_curvatures {
        assert!((p - 0.125).abs() < 1e-11);
    }
    for &k in &res_k8.link_curvatures {
        assert!((k - 2.0).abs() < 1e-10);
    }
    assert_eq!(res_k8.suggested_geometry, SuggestedGeometry::Spherical);
    assert_eq!(
        res_k8.suggested_manifold_kind(),
        Some(ManifoldKind::Spherical)
    );

    // Cycle C_10: R_e = (n-1)/n, p_i = 1/n, kappa = 4/(n-1)
    let c10 = make_cycle_graph(10);
    let res_c10 = resistance_curvature_flow(
        &c10,
        &ResistanceCurvatureConfig {
            steps: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(res_c10.used_exact);
    assert!(res_c10.foster_rel_err < 1e-12);
    for &r in &res_c10.effective_resistances {
        assert!((r - 0.9).abs() < 1e-11);
    }
    for &k in &res_c10.link_curvatures {
        assert!((k - 4.0 / 9.0).abs() < 1e-10);
    }

    // Torus T_{8x8}: N=64, uniform p_i = 1/64, kappa = 8/63, std = 0 => Torus2d
    let t8 = make_torus_graph(8, 8);
    let res_t8 = resistance_curvature_flow(
        &t8,
        &ResistanceCurvatureConfig {
            steps: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(res_t8.used_exact);
    assert!(res_t8.foster_rel_err < 1e-11);
    assert!((res_t8.mean_link_curvature - 8.0 / 63.0).abs() < 1e-9);
    assert!(res_t8.std_link_curvature < 1e-9);
    assert_eq!(res_t8.suggested_geometry, SuggestedGeometry::Torus2d);
    assert_eq!(
        res_t8.suggested_manifold_kind(),
        Some(ManifoldKind::Torus2d)
    );

    // Star K_{1,12}: R_e = 1.0, p_center = 1 - 12/2 = -5, p_leaf = 0.5, kappa = 2*(-4.5)/1 = -9
    let star = make_star_graph(13);
    let res_star = resistance_curvature_flow(
        &star,
        &ResistanceCurvatureConfig {
            steps: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(res_star.used_exact);
    assert!(res_star.foster_rel_err < 1e-12);
    for &r in &res_star.effective_resistances {
        assert!((r - 1.0).abs() < 1e-11);
    }
    assert!((res_star.mean_link_curvature - (-9.0)).abs() < 1e-10);
    assert_eq!(
        res_star.suggested_geometry,
        SuggestedGeometry::HyperbolicPoincare
    );
    assert_eq!(
        res_star.suggested_manifold_kind(),
        Some(ManifoldKind::HyperbolicPoincare)
    );

    // Weighted tree: R_e = l_e, w_e * R_e = 1
    let tree_edges = vec![(0, 1), (1, 2), (1, 3), (3, 4)];
    let tree_lengths = vec![2.0_f64, 0.5, 4.0, 1.5];
    let tree = CsrGraph::from_edges(5, &tree_edges, Some(&tree_lengths), false, false).unwrap();
    let er_tree = compute_effective_resistances(
        &tree,
        &EffectiveResistanceConfig {
            weights_are_lengths: true,
            exact: Some(true),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(er_tree.foster_rel_err < 1e-12);
    for (e, &r) in er_tree.effective_resistances.iter().enumerate() {
        assert!((r - er_tree.edge_lengths[e]).abs() < 1e-11);
    }
}

#[test]
fn test_jl_amg_pcg_foster_and_multicc_isolated() {
    // Connected 10x10 grid (N=100) forced through JL + AmgPcgSolver
    let mut grid_edges = Vec::new();
    for r in 0..10u32 {
        for c in 0..10u32 {
            let u = r * 10 + c;
            if c + 1 < 10 {
                grid_edges.push((u, u + 1));
            }
            if r + 1 < 10 {
                grid_edges.push((u, u + 10));
            }
        }
    }
    let g100 = CsrGraph::<f64>::from_edges(100, &grid_edges, None, false, false).unwrap();
    let cfg = EffectiveResistanceConfig {
        jl_dim: 64,
        seed: 42,
        exact: Some(false),
        ..Default::default()
    };
    let r1 = compute_effective_resistances(&g100, &cfg).unwrap();
    let r2 = compute_effective_resistances(&g100, &cfg).unwrap();
    assert!(!r1.used_exact);
    assert!(r1.amg_all_converged);
    assert!(r1.amg_total_iterations > 0);
    assert_eq!(r1.num_components, 1);
    assert!((r1.expected_foster - 99.0).abs() < 1e-12);
    assert!(
        r1.foster_rel_err < 0.05,
        "Foster relative error too large: {}",
        r1.foster_rel_err
    );
    assert_eq!(r1.effective_resistances, r2.effective_resistances);

    // Multi-component + isolated nodes + weighted conductances
    let mut multi_edges = Vec::new();
    let mut multi_weights = Vec::new();
    // Component 1: 6x6 grid (0..36)
    for r in 0..6u32 {
        for c in 0..6u32 {
            let u = r * 6 + c;
            if c + 1 < 6 {
                multi_edges.push((u, u + 1));
                multi_weights.push(2.5);
            }
            if r + 1 < 6 {
                multi_edges.push((u, u + 6));
                multi_weights.push(1.25);
            }
        }
    }
    // Component 2: K_40 (36..76)
    for u in 36..76u32 {
        for v in (u + 1)..76u32 {
            multi_edges.push((u, v));
            multi_weights.push(0.5);
        }
    }
    // Isolated vertices: 76..80 (4 isolated nodes) => c_cc = 1 + 1 + 4 = 6
    let g_multi =
        CsrGraph::from_edges(80, &multi_edges, Some(&multi_weights), false, false).unwrap();
    let flow_multi = resistance_curvature_flow(
        &g_multi,
        &ResistanceCurvatureConfig {
            steps: 1,
            resistance: EffectiveResistanceConfig {
                jl_dim: 96,
                seed: 7,
                weights_are_lengths: false,
                exact: Some(false),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(flow_multi.num_components, 6);
    assert!((flow_multi.expected_foster - 74.0).abs() < 1e-12);
    assert!(flow_multi.amg_all_converged);
    assert!(flow_multi.foster_rel_err < 0.05);
    for &p in &flow_multi.vertex_curvatures[76..80] {
        assert!((p - 1.0).abs() < 1e-12);
    }
    let sum_p: f64 = flow_multi.vertex_curvatures.iter().sum();
    assert!((sum_p - (80.0 - flow_multi.foster_sum)).abs() < 1e-10);
}

#[test]
fn test_barbell_curvature_flow_separates_bridge_and_cliques() {
    let barbell = make_barbell_graph(6);
    let flow = resistance_curvature_flow(
        &barbell,
        &ResistanceCurvatureConfig {
            steps: 3,
            eta: 0.2,
            kappa_max: 2.0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(flow.step_history.len(), 4);
    assert_eq!(flow.step_history[0].step, 0);
    assert_eq!(flow.step_history[3].step, 3);
    let mean_len = flow.edge_lengths.iter().sum::<f64>() / (flow.num_edges as f64);
    assert!((mean_len - 1.0).abs() < 1e-12);

    let mut bridge_len = 0.0_f64;
    let mut intra_lens = Vec::new();
    for e in 0..flow.num_edges {
        if flow.edge_u[e] == 5 && flow.edge_v[e] == 6 {
            bridge_len = flow.edge_lengths[e];
        } else {
            intra_lens.push(flow.edge_lengths[e]);
        }
    }
    assert!(
        bridge_len > 1.0,
        "Bridge edge should lengthen (>1.0), got {bridge_len}"
    );
    let mean_intra = intra_lens.iter().sum::<f64>() / (intra_lens.len() as f64);
    assert!(
        mean_intra < 1.0,
        "Intra-clique edges should shorten (<1.0), got {mean_intra}"
    );

    let lg = flow.to_length_graph().unwrap();
    let cg = flow.to_conductance_graph().unwrap();
    let lg32 = flow.to_length_graph_f32().unwrap();
    let cg32 = flow.to_conductance_graph_f32().unwrap();
    assert_eq!(lg.num_nodes(), barbell.num_nodes());
    assert_eq!(cg.num_edges(), barbell.num_edges());
    assert_eq!(lg32.num_edges(), barbell.num_edges());
    assert_eq!(cg32.num_edges(), barbell.num_edges());
    assert!(lg.is_weighted());
    assert!(cg.is_weighted());
}

#[test]
fn test_adversarial_edge_cases_and_validation() {
    let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let r0 = resistance_curvature_flow(&g0, &ResistanceCurvatureConfig::default()).unwrap();
    assert_eq!(r0.num_nodes, 0);
    assert_eq!(r0.num_edges, 0);
    assert_eq!(r0.num_components, 0);
    assert_eq!(r0.foster_sum, 0.0);
    assert_eq!(r0.suggested_geometry, SuggestedGeometry::Euclidean);

    let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let r1 = resistance_curvature_flow(&g1, &ResistanceCurvatureConfig::default()).unwrap();
    assert_eq!(r1.num_nodes, 1);
    assert_eq!(r1.num_components, 1);
    assert!((r1.vertex_curvatures[0] - 1.0).abs() < 1e-12);
    assert_eq!(r1.suggested_geometry, SuggestedGeometry::Euclidean);

    let g_edgeless = CsrGraph::<f64>::from_edges(5, &[], None, false, false).unwrap();
    let r_edgeless =
        resistance_curvature_flow(&g_edgeless, &ResistanceCurvatureConfig::default()).unwrap();
    assert_eq!(r_edgeless.num_components, 5);
    assert_eq!(r_edgeless.expected_foster, 0.0);
    assert_eq!(r_edgeless.foster_sum, 0.0);

    // Directed input graph with self-loop and parallel edge should sanitize cleanly
    let g_dir = CsrGraph::from_edges(
        3,
        &[(0, 0), (0, 1), (1, 0), (1, 2), (1, 2)],
        Some(&[1.0, 2.0, 1.0, 3.0, 4.0]),
        true,
        true,
    )
    .unwrap();
    let r_dir =
        compute_effective_resistances(&g_dir, &EffectiveResistanceConfig::default()).unwrap();
    assert_eq!(r_dir.num_edges, 2);
    assert!(r_dir.foster_rel_err < 1e-12);

    // Invalid weights: zero, negative (and NaN rejected by CsrGraph)
    for bad_w in [0.0, -1.0] {
        let g_bad = CsrGraph::from_edges(2, &[(0, 1)], Some(&[bad_w]), false, false).unwrap();
        assert!(
            compute_effective_resistances(&g_bad, &EffectiveResistanceConfig::default()).is_err()
        );
    }
    assert!(CsrGraph::from_edges(2, &[(0, 1)], Some(&[f64::NAN]), false, false).is_err());

    // Invalid config parameters
    assert!(
        compute_effective_resistances(
            &g1,
            &EffectiveResistanceConfig {
                jl_dim: 0,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        resistance_curvature_flow(
            &g1,
            &ResistanceCurvatureConfig {
                eta: -0.1,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        resistance_curvature_flow(
            &g1,
            &ResistanceCurvatureConfig {
                kappa_max: 0.0,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        resistance_curvature_flow(
            &g1,
            &ResistanceCurvatureConfig {
                min_edge_length: 0.0,
                ..Default::default()
            }
        )
        .is_err()
    );

    // exact = Some(true) cap at 512 nodes
    let g_513 = CsrGraph::<f64>::from_edges(513, &[(0, 1)], None, false, false).unwrap();
    assert!(
        compute_effective_resistances(
            &g_513,
            &EffectiveResistanceConfig {
                exact: Some(true),
                ..Default::default()
            }
        )
        .is_err()
    );
}
