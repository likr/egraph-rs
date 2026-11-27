use egraph_core::CoreError;

use super::*;

#[test]
fn test_kmeans_well_separated_clusters_and_canonical_labels() {
    // Two well-separated clusters placed so cluster 0 starts at index 0.
    let pts = vec![
        10.0, 10.0, 10.2, 9.8, 9.8, 10.1, // cluster 0
        -10.0, -10.0, -9.9, -10.2, -10.1, -9.8, // cluster 1
    ];
    let cfg = KMeansConfig {
        k: 2,
        n_init: 3,
        seed: 123,
        ..Default::default()
    };
    let res = kmeans_plus_plus(&pts, 6, 2, &cfg).unwrap();
    assert_eq!(res.labels, vec![0, 0, 0, 1, 1, 1]);
    assert_eq!(res.k, 2);
    assert_eq!(res.dims, 2);
    assert!(res.converged);
    assert!(res.inertia < 1.0);
    assert_eq!(res.centroids.len(), res.k * res.dims);
    assert!((res.centroids[0] - 10.0).abs() < 1e-12);
    assert!((res.centroids[1] - 9.966666666666667).abs() < 1e-12);
    assert!((res.centroids[2] + 10.0).abs() < 1e-12);
    assert!((res.centroids[3] + 10.0).abs() < 1e-12);
}

#[test]
fn test_kmeans_seed_determinism_and_duplicate_points() {
    let pts = vec![0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0];
    let cfg = KMeansConfig {
        k: 3,
        seed: 99,
        ..Default::default()
    };
    let r1 = kmeans_plus_plus(&pts, 5, 2, &cfg).unwrap();
    let r2 = kmeans_plus_plus(&pts, 5, 2, &cfg).unwrap();
    assert_eq!(r1, r2);
    assert_eq!(r1.k, 3);
    assert_eq!(r1.dims, 2);
    assert_eq!(r1.centroids.len(), 6);

    // Identical points should still return finite centroids and valid 0..k labels.
    let identical = vec![1.5, -2.5, 1.5, -2.5, 1.5, -2.5, 1.5, -2.5];
    let r_id = kmeans_plus_plus(
        &identical,
        4,
        2,
        &KMeansConfig {
            k: 2,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(r_id.k, 2);
    assert_eq!(r_id.dims, 2);
    assert!(r_id.centroids.iter().all(|v| v.is_finite()));
    assert!(r_id.labels.iter().all(|&l| l < 2));
    assert_eq!(r_id.inertia, 0.0);

    // Two distinct locations with k=3 forces empty-cluster farthest-point reseed by pigeonhole.
    let two_loc = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 10.0, 0.0];
    let r_reseed = kmeans_plus_plus(
        &two_loc,
        6,
        2,
        &KMeansConfig {
            k: 3,
            max_iterations: 10,
            seed: 7,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(r_reseed.k, 3);
    assert_eq!(r_reseed.dims, 2);
    assert!(r_reseed.centroids.iter().all(|v| v.is_finite()));
    assert!(r_reseed.labels.iter().all(|&l| l < 3));
    assert!(r_reseed.inertia >= 0.0 && r_reseed.inertia < 1e-12);

    // k == n boundary yields zero inertia.
    let exact = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let r_k_eq_n = kmeans_plus_plus(
        &exact,
        4,
        2,
        &KMeansConfig {
            k: 4,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(r_k_eq_n.k, 4);
    assert_eq!(r_k_eq_n.dims, 2);
    assert_eq!(r_k_eq_n.labels, vec![0, 1, 2, 3]);
    assert!(r_k_eq_n.inertia < 1e-12);

    // n == 1, k == 1, dims == 3 singleton boundary.
    let single = kmeans_plus_plus(
        &[2.0, -3.0, 4.5],
        1,
        3,
        &KMeansConfig {
            k: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(single.labels, vec![0]);
    assert_eq!(single.k, 1);
    assert_eq!(single.dims, 3);
    assert_eq!(single.centroids, vec![2.0, -3.0, 4.5]);
    assert_eq!(single.inertia, 0.0);
    assert!(single.converged);
}

#[test]
fn test_kmeans_validation_errors() {
    let cfg = KMeansConfig::default();
    assert!(matches!(
        kmeans_plus_plus(&[], 0, 2, &cfg),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        kmeans_plus_plus(&[0.0, 1.0], 2, 0, &cfg),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        kmeans_plus_plus(
            &[0.0, 1.0],
            1,
            2,
            &KMeansConfig {
                k: 0,
                ..Default::default()
            }
        ),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        kmeans_plus_plus(&[0.0, 1.0], 1, 2, &cfg),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        kmeans_plus_plus(&[0.0, 1.0, 2.0], 2, 2, &cfg),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        kmeans_plus_plus(&[0.0, f64::NAN, 1.0, 2.0], 2, 2, &cfg),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        kmeans_plus_plus(
            &[0.0, 0.0, 1.0, 1.0],
            2,
            2,
            &KMeansConfig {
                tol: f64::INFINITY,
                ..Default::default()
            }
        ),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        kmeans_plus_plus(
            &[0.0, 0.0, 1.0, 1.0],
            2,
            2,
            &KMeansConfig {
                max_iterations: 0,
                ..Default::default()
            }
        ),
        Err(CoreError::InvalidInput(_))
    ));
}
