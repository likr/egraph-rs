use egraph_core::CsrGraph;

use super::super::*;
use super::fixtures::{
    BUS_1138_CSV, KARATE_CSV, LESMIS_CSV, assert_canonical_and_connected, make_ring_of_cliques,
    make_two_cliques_barbell, oracle_quality_and_modularity, parse_test_csv,
};

#[test]
fn test_leiden_barbell_ring_and_weighted_f32() {
    let barbell = make_two_cliques_barbell(6);
    let res = leiden(&barbell, &LeidenConfig::default()).unwrap();
    assert_canonical_and_connected(&barbell, &res);
    assert_eq!(res.num_communities, 2);
    assert_eq!(res.labels[..6], [0; 6]);
    assert_eq!(res.labels[6..], [1; 6]);
    assert!((res.quality - res.modularity).abs() < 1e-12);
    assert!((res.modularity - 0.467_741_935_483_870_94).abs() < 1e-10);
    assert!(res.passes >= 1);

    let res_cpm = leiden(
        &barbell,
        &LeidenConfig {
            objective: LeidenObjective::Cpm,
            resolution: 0.5,
            ..Default::default()
        },
    )
    .unwrap();
    assert_canonical_and_connected(&barbell, &res_cpm);
    assert_eq!(res_cpm.num_communities, 2);
    assert!((res_cpm.quality - 15.0).abs() < 1e-12);

    // Ring of 8 cliques of size 5.
    let ring = make_ring_of_cliques(8, 5);
    let res_ring = leiden(&ring, &LeidenConfig::default()).unwrap();
    assert_canonical_and_connected(&ring, &res_ring);
    assert_eq!(res_ring.num_communities, 8);
    assert!((res_ring.modularity - 0.784_090_909_090_909_1).abs() < 1e-10);
    for c in 0..8 {
        let first = res_ring.labels[c * 5];
        assert!(
            res_ring.labels[c * 5..(c + 1) * 5]
                .iter()
                .all(|&l| l == first)
        );
    }

    let res_ring_cpm = leiden(
        &ring,
        &LeidenConfig {
            objective: LeidenObjective::Cpm,
            resolution: 0.5,
            ..Default::default()
        },
    )
    .unwrap();
    assert_canonical_and_connected(&ring, &res_ring_cpm);
    assert_eq!(res_ring_cpm.num_communities, 8);

    // Weighted f32 barbell + low-resolution merge.
    let mut w_edges = Vec::new();
    let mut w_vals = Vec::new();
    for i in 0..4u32 {
        for j in (i + 1)..4 {
            w_edges.push((i, j));
            w_vals.push(8.0_f32);
            w_edges.push((4 + i, 4 + j));
            w_vals.push(8.0_f32);
        }
    }
    w_edges.push((3, 4));
    w_vals.push(0.1_f32);
    let wg = CsrGraph::from_edges(8, &w_edges, Some(&w_vals), false, false).unwrap();

    let w_res = leiden(&wg, &LeidenConfig::default()).unwrap();
    assert_canonical_and_connected(&wg, &w_res);
    assert_eq!(w_res.num_communities, 2);
    assert_eq!(w_res.labels[..4], [0; 4]);
    assert_eq!(w_res.labels[4..], [1; 4]);
    let (oq, om) = oracle_quality_and_modularity(
        &wg,
        &w_res.labels,
        w_res.num_communities,
        LeidenObjective::Modularity,
        1.0,
    );
    assert!((w_res.quality - oq).abs() < 1e-12);
    assert!((w_res.modularity - om).abs() < 1e-12);

    let low_gamma = leiden(
        &wg,
        &LeidenConfig {
            resolution: 0.001,
            ..Default::default()
        },
    )
    .unwrap();
    assert_canonical_and_connected(&wg, &low_gamma);
    assert_eq!(low_gamma.num_communities, 1);
}

#[test]
fn test_leiden_real_datasets_connectivity_and_oracle() {
    let datasets = [
        ("karate", parse_test_csv(KARATE_CSV)),
        ("lesmis", parse_test_csv(LESMIS_CSV)),
        ("1138_bus", parse_test_csv(BUS_1138_CSV)),
    ];

    for (name, graph) in &datasets {
        for &(objective, resolution, max_passes) in &[
            (LeidenObjective::Modularity, 1.0, 100),
            (LeidenObjective::Modularity, 0.5, 100),
            (LeidenObjective::Modularity, 2.0, 100),
            (LeidenObjective::Modularity, 1.0, 1),
            (LeidenObjective::Cpm, 0.0, 100),
            (LeidenObjective::Cpm, 0.05, 100),
            (LeidenObjective::Cpm, 0.25, 100),
            (LeidenObjective::Cpm, 0.5, 100),
            (LeidenObjective::Cpm, 1.0, 100),
            (LeidenObjective::Cpm, 0.1, 1),
        ] {
            let cfg = LeidenConfig {
                objective,
                resolution,
                max_passes,
                ..Default::default()
            };
            let res = leiden(graph, &cfg).unwrap();
            assert_canonical_and_connected(graph, &res);
            let (oq, om) = oracle_quality_and_modularity(
                graph,
                &res.labels,
                res.num_communities,
                objective,
                resolution,
            );
            assert!(
                (res.quality - oq).abs() < 1e-12,
                "{name} quality mismatch: {} vs {oq}",
                res.quality
            );
            assert!(
                (res.modularity - om).abs() < 1e-12,
                "{name} modularity mismatch: {} vs {om}",
                res.modularity
            );
            assert!(res.passes <= max_passes);
        }

        let karate_mod = leiden(graph, &LeidenConfig::default()).unwrap();
        if *name == "karate" {
            assert!(karate_mod.modularity > 0.41);
            assert!(karate_mod.num_communities >= 2);
        }
    }
}

#[test]
fn test_leiden_cpm_resolution_monotonicity() {
    let graphs = [
        parse_test_csv(KARATE_CSV),
        parse_test_csv(LESMIS_CSV),
        make_ring_of_cliques(8, 5),
    ];
    let gammas = [0.0, 0.05, 0.2, 0.5, 1.0];

    for g in &graphs {
        let mut prev_k = 0usize;
        for &gamma in &gammas {
            let res = leiden(
                g,
                &LeidenConfig {
                    objective: LeidenObjective::Cpm,
                    resolution: gamma,
                    seed: 7,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_canonical_and_connected(g, &res);
            assert!(
                res.num_communities >= prev_k,
                "CPM community count decreased from {prev_k} to {} at gamma={gamma}",
                res.num_communities
            );
            prev_k = res.num_communities;
        }
    }
}

#[test]
fn test_leiden_outer_restarts_convergence_and_max_passes_bound() {
    let karate = parse_test_csv(KARATE_CSV);
    let lesmis = parse_test_csv(LESMIS_CSV);
    let bus = parse_test_csv(BUS_1138_CSV);

    // Converged outer-restart Modularity matches igraph(n_iterations=-1) on karate and lesmis.
    for seed in 0..10u64 {
        let rk = leiden(
            &karate,
            &LeidenConfig {
                seed,
                ..Default::default()
            },
        )
        .unwrap();
        assert_canonical_and_connected(&karate, &rk);
        assert!(
            rk.modularity > 0.4195,
            "karate seed={seed} Q={} below converged threshold",
            rk.modularity
        );
        assert_eq!(rk.num_communities, 4);

        let rl = leiden(
            &lesmis,
            &LeidenConfig {
                seed,
                ..Default::default()
            },
        )
        .unwrap();
        assert_canonical_and_connected(&lesmis, &rl);
        assert!(
            rl.modularity > 0.5595,
            "lesmis seed={seed} Q={} below converged threshold",
            rl.modularity
        );
        assert_eq!(rl.num_communities, 6);
    }

    // Converged 1138_bus Modularity and CPM outperform single-pass bounds and match igraph(-1).
    let mut bus_q_sum = 0.0_f64;
    let mut bus_h_sum = 0.0_f64;
    for seed in 0..5u64 {
        let rb_mod = leiden(
            &bus,
            &LeidenConfig {
                seed,
                ..Default::default()
            },
        )
        .unwrap();
        assert_canonical_and_connected(&bus, &rb_mod);
        assert!(rb_mod.passes <= 100);
        bus_q_sum += rb_mod.modularity;

        let rb_cpm = leiden(
            &bus,
            &LeidenConfig {
                objective: LeidenObjective::Cpm,
                resolution: 0.25,
                seed,
                ..Default::default()
            },
        )
        .unwrap();
        assert_canonical_and_connected(&bus, &rb_cpm);
        assert!(rb_cpm.passes <= 100);
        bus_h_sum += rb_cpm.quality;
    }
    let mean_bus_q = bus_q_sum / 5.0;
    let mean_bus_h = bus_h_sum / 5.0;
    assert!(
        mean_bus_q > 0.8985,
        "1138_bus mean Modularity Q={mean_bus_q} did not converge above 0.8985"
    );
    assert!(
        mean_bus_h > 503.0,
        "1138_bus mean CPM(0.25) H={mean_bus_h} did not converge above 503.0"
    );

    // Verify max_passes strictly bounds committed passes across tight budgets.
    for &max_passes in &[1usize, 2, 5] {
        let r1 = leiden(
            &bus,
            &LeidenConfig {
                max_passes,
                seed: 42,
                ..Default::default()
            },
        )
        .unwrap();
        assert_canonical_and_connected(&bus, &r1);
        assert!(r1.passes <= max_passes);
    }
}
