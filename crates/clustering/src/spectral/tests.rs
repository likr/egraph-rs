use egraph_core::{CoreError, CsrGraph};

use super::*;

fn make_barbell(clique_size: usize) -> CsrGraph<f64> {
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
fn test_spectral_recovers_barbell_and_three_cliques() {
    let barbell = make_barbell(5);
    let res = spectral_clustering(&barbell, &SpectralClusteringConfig::default()).unwrap();
    assert_eq!(res.num_clusters, 2);
    assert_eq!(res.eigen_iterations, 0);
    assert_eq!(res.labels.len(), 10);
    assert_eq!(res.labels[..5], [0, 0, 0, 0, 0]);
    assert_eq!(res.labels[5..], [1, 1, 1, 1, 1]);

    // Three 5-cliques connected as a ring of 3 weak bridges.
    let mut edges = Vec::new();
    for c in 0..3u32 {
        let base = c * 5;
        for i in 0..5 {
            for j in (i + 1)..5 {
                edges.push((base + i, base + j));
            }
        }
    }
    edges.push((0, 5));
    edges.push((6, 10));
    edges.push((11, 1));
    let tri_clique = CsrGraph::<f64>::from_edges(15, &edges, None, false, false).unwrap();
    let cfg3 = SpectralClusteringConfig {
        k: 3,
        ..Default::default()
    };
    let r3 = spectral_clustering(&tri_clique, &cfg3).unwrap();
    assert_eq!(r3.num_clusters, 3);
    assert_eq!(r3.eigen_iterations, 0);
    assert_eq!(r3.labels[0..5], [0, 0, 0, 0, 0]);
    assert_eq!(r3.labels[5..10], [1, 1, 1, 1, 1]);
    assert_eq!(r3.labels[10..15], [2, 2, 2, 2, 2]);
}

#[test]
fn test_spectral_disconnected_cliques_and_seed_determinism() {
    let mut edges = Vec::new();
    for i in 0..5u32 {
        for j in (i + 1)..5 {
            edges.push((i, j));
            edges.push((5 + i, 5 + j));
        }
    }
    let disc = CsrGraph::<f64>::from_edges(10, &edges, None, false, false).unwrap();
    let cfg = SpectralClusteringConfig {
        k: 2,
        seed: 77,
        ..Default::default()
    };
    let r1 = spectral_clustering(&disc, &cfg).unwrap();
    let r2 = spectral_clustering(&disc, &cfg).unwrap();
    assert_eq!(r1, r2);
    assert_eq!(r1.num_clusters, 2);
    assert_eq!(r1.labels[..5], [0, 0, 0, 0, 0]);
    assert_eq!(r1.labels[5..], [1, 1, 1, 1, 1]);

    // Large disconnected cliques (n = 60) exercise Lanczos with disconnected nullspace.
    let mut large_edges = Vec::new();
    for i in 0..30u32 {
        for j in (i + 1)..30 {
            large_edges.push((i, j));
            large_edges.push((30 + i, 30 + j));
        }
    }
    let large_disc = CsrGraph::<f64>::from_edges(60, &large_edges, None, false, false).unwrap();
    let r_large = spectral_clustering(&large_disc, &cfg).unwrap();
    assert_eq!(r_large.num_clusters, 2);
    assert!(r_large.eigen_iterations >= 1);
    assert_eq!(r_large.labels[..30], [0; 30]);
    assert_eq!(r_large.labels[30..], [1; 30]);
}

#[test]
fn test_spectral_lanczos_large_barbell_and_weighted_f32() {
    // n = 60 (> 48 dense threshold) exercises the Lanczos eigensolver path.
    let large_barbell = make_barbell(30);
    let cfg = SpectralClusteringConfig::default();
    let res = spectral_clustering(&large_barbell, &cfg).unwrap();
    assert_eq!(res.num_clusters, 2);
    assert!(res.eigen_iterations >= 1 && res.eigen_iterations <= cfg.max_eigen_iterations);
    assert_eq!(res.labels.len(), 60);
    assert_eq!(res.labels[..30], [0; 30]);
    assert_eq!(res.labels[30..], [1; 30]);

    // Weighted f32 graph: strong cliques (10.0) joined by weak bridge (0.05).
    let mut edges = Vec::new();
    let mut weights = Vec::new();
    for i in 0..4u32 {
        for j in (i + 1)..4 {
            edges.push((i, j));
            weights.push(10.0_f32);
            edges.push((4 + i, 4 + j));
            weights.push(10.0_f32);
        }
    }
    edges.push((3, 4));
    weights.push(0.05_f32);
    let wg = CsrGraph::from_edges(8, &edges, Some(&weights), false, false).unwrap();
    let wr = spectral_clustering(&wg, &SpectralClusteringConfig::default()).unwrap();
    assert_eq!(wr.num_clusters, 2);
    assert_eq!(wr.labels[..4], [0, 0, 0, 0]);
    assert_eq!(wr.labels[4..], [1, 1, 1, 1]);
}

#[test]
fn test_spectral_degenerate_and_error_cases() {
    let cfg = SpectralClusteringConfig::default();

    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let r_empty = spectral_clustering(&empty, &cfg).unwrap();
    assert!(r_empty.labels.is_empty());
    assert_eq!(r_empty.num_clusters, 0);
    assert_eq!(r_empty.eigen_iterations, 0);

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let r_single = spectral_clustering(&single, &cfg).unwrap();
    assert_eq!(r_single.labels, vec![0]);
    assert_eq!(r_single.num_clusters, 1);
    assert_eq!(r_single.eigen_iterations, 0);

    let edgeless = CsrGraph::<f64>::from_edges(6, &[], None, false, false).unwrap();
    let r_edgeless = spectral_clustering(&edgeless, &cfg).unwrap();
    assert_eq!(r_edgeless.labels.len(), 6);
    assert!(r_edgeless.num_clusters >= 1 && r_edgeless.num_clusters <= 2);
    assert!(
        r_edgeless
            .labels
            .iter()
            .all(|&l| (l as usize) < r_edgeless.num_clusters)
    );

    let directed = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, true, false).unwrap();
    assert!(matches!(
        spectral_clustering(&directed, &cfg),
        Err(CoreError::InvalidInput(_))
    ));

    let barbell = make_barbell(3);
    assert!(matches!(
        spectral_clustering(
            &barbell,
            &SpectralClusteringConfig {
                k: 0,
                ..Default::default()
            }
        ),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        spectral_clustering(
            &barbell,
            &SpectralClusteringConfig {
                max_eigen_iterations: 0,
                ..Default::default()
            }
        ),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        spectral_clustering(
            &barbell,
            &SpectralClusteringConfig {
                kmeans_max_iterations: 0,
                ..Default::default()
            }
        ),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        spectral_clustering(
            &barbell,
            &SpectralClusteringConfig {
                eigen_tol: 0.0,
                ..Default::default()
            }
        ),
        Err(CoreError::InvalidInput(_))
    ));
}

#[test]
fn test_spectral_struct_builders_and_trait() {
    use crate::clusterer::{ClusterResult, CommunityDetector};

    let barbell = make_barbell(5);
    let cfg = SpectralClusteringConfig::new()
        .with_k(2)
        .with_max_eigen_iterations(250)
        .with_eigen_tol(1e-5)
        .with_kmeans_max_iterations(80)
        .with_kmeans_tol(1e-5)
        .with_seed(42);
    let alg = SpectralClustering::new(cfg).with_max_eigen_iterations(300);
    assert_eq!(alg.config().max_eigen_iterations, 300);

    let full = alg.run(&barbell).unwrap();
    assert!(full.converged);
    let unified = alg.detect(&barbell).unwrap();
    let converted: ClusterResult = full.clone().into();
    assert_eq!(unified, converted);
    assert_eq!(unified.num_clusters(), 2);
    assert_eq!(unified.quality(), None);
    assert_eq!(unified.iterations(), full.eigen_iterations);
    assert!(unified.converged());
}
