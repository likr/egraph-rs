use egraph_core::{CoreError, CsrGraph};

use super::*;

fn make_two_cliques_barbell(clique_size: usize) -> CsrGraph<f64> {
    let n = clique_size * 2;
    let mut edges = Vec::new();
    for i in 0..clique_size {
        for j in (i + 1)..clique_size {
            edges.push((i as u32, j as u32));
            edges.push(((clique_size + i) as u32, (clique_size + j) as u32));
        }
    }
    edges.push(((clique_size - 1) as u32, clique_size as u32));
    CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
}

#[test]
fn test_louvain_recovers_two_cliques_and_multilevel_ring() {
    let barbell = make_two_cliques_barbell(6);
    let res = louvain(&barbell, &LouvainConfig::default()).unwrap();
    assert_eq!(res.num_communities, 2);
    assert_eq!(res.labels[..6], [0; 6]);
    assert_eq!(res.labels[6..], [1; 6]);
    assert!(res.modularity > 0.4);
    assert!(res.passes >= 1);

    // Ring of 8 cliques of size 5: multi-level aggregation merges pairs cleanly into 8 cliques.
    let mut edges = Vec::new();
    for c in 0..8u32 {
        let base = c * 5;
        for i in 0..5 {
            for j in (i + 1)..5 {
                edges.push((base + i, base + j));
            }
        }
        let next_base = ((c + 1) % 8) * 5;
        edges.push((base + 4, next_base));
    }
    let ring = CsrGraph::<f64>::from_edges(40, &edges, None, false, false).unwrap();
    let res_ring = louvain(&ring, &LouvainConfig::default()).unwrap();
    assert_eq!(res_ring.num_communities, 8);
    assert!(res_ring.modularity > 0.7);
    for c in 0..8 {
        let first = res_ring.labels[c * 5];
        assert!(
            res_ring.labels[c * 5..(c + 1) * 5]
                .iter()
                .all(|&l| l == first)
        );
    }

    // Weighted f32 graph + resolution variation.
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
    let w_res = louvain(&wg, &LouvainConfig::default()).unwrap();
    assert_eq!(w_res.num_communities, 2);
    assert_eq!(w_res.labels[..4], [0; 4]);
    assert_eq!(w_res.labels[4..], [1; 4]);

    let low_gamma = louvain(
        &wg,
        &LouvainConfig {
            resolution: 0.001,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(low_gamma.num_communities, 1);

    let coarse_ring = louvain(
        &ring,
        &LouvainConfig {
            resolution: 0.2,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(coarse_ring.num_communities < 8);
}

#[test]
fn test_louvain_determinism_and_degenerate_cases() {
    let barbell = make_two_cliques_barbell(5);
    let cfg = LouvainConfig {
        seed: 1234,
        ..Default::default()
    };
    assert_eq!(
        louvain(&barbell, &cfg).unwrap(),
        louvain(&barbell, &cfg).unwrap()
    );

    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let re = louvain(&empty, &cfg).unwrap();
    assert!(re.labels.is_empty());
    assert_eq!(re.num_communities, 0);
    assert_eq!(re.modularity, 0.0);

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let rs = louvain(&single, &cfg).unwrap();
    assert_eq!(rs.labels, vec![0]);
    assert_eq!(rs.num_communities, 1);

    let edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    let ri = louvain(&edgeless, &cfg).unwrap();
    assert_eq!(ri.labels, vec![0, 1, 2, 3]);
    assert_eq!(ri.num_communities, 4);
    assert_eq!(ri.modularity, 0.0);
}

#[test]
fn test_louvain_directed_and_invalid_config_errors() {
    let cfg = LouvainConfig::default();
    let directed = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, true, false).unwrap();
    assert!(matches!(
        louvain(&directed, &cfg),
        Err(CoreError::InvalidInput(_))
    ));

    let g = make_two_cliques_barbell(3);
    assert!(matches!(
        louvain(
            &g,
            &LouvainConfig {
                max_passes: 0,
                ..Default::default()
            }
        ),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        louvain(
            &g,
            &LouvainConfig {
                resolution: -0.5,
                ..Default::default()
            }
        ),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        louvain(
            &g,
            &LouvainConfig {
                min_modularity_gain: f64::NAN,
                ..Default::default()
            }
        ),
        Err(CoreError::NonFiniteValue { .. })
    ));
}

#[test]
fn test_louvain_struct_builders_trait_and_convergence() {
    use crate::clusterer::{ClusterResult, CommunityDetector};

    let barbell = make_two_cliques_barbell(6);
    let cfg = LouvainConfig::new()
        .with_max_passes(50)
        .with_resolution(1.0)
        .with_min_modularity_gain(1e-7)
        .with_seed(7);
    let alg = Louvain::new(cfg).with_seed(42);
    assert_eq!(alg.config().max_passes, 50);
    assert_eq!(alg.config().seed, 42);

    let full = alg.run(&barbell).unwrap();
    assert!(full.converged);
    let unified = alg.detect(&barbell).unwrap();
    let converted: ClusterResult = full.clone().into();
    assert_eq!(unified, converted);
    assert_eq!(unified.quality(), Some(full.modularity));
    assert!(unified.converged());

    // A ring of 8 cliques requires >1 pass when resolution is low, so max_passes = 1 does not converge.
    let mut edges = Vec::new();
    for c in 0..8u32 {
        let base = c * 5;
        for i in 0..5 {
            for j in (i + 1)..5 {
                edges.push((base + i, base + j));
            }
        }
        let next_base = ((c + 1) % 8) * 5;
        edges.push((base + 4, next_base));
    }
    let ring = CsrGraph::<f64>::from_edges(40, &edges, None, false, false).unwrap();
    let limited = Louvain::default()
        .with_max_passes(1)
        .with_resolution(0.2)
        .run(&ring)
        .unwrap();
    assert!(!limited.converged);
}
