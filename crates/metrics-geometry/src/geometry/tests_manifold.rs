use egraph_core::{CoreError, CsrGraph, ManifoldKind, ManifoldLayout2d};

use super::*;

#[test]
fn geometry_manifold_ideal_edge_lengths_variants() {
    let g_unw = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, false, false).unwrap();
    let mut torus = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.0, 0.25, 0.5],
        vec![0.0, 0.0, 0.0],
    )
    .unwrap();
    torus.set_torus_extents(4.0, 1.0).unwrap();
    // Physical edge lengths are 4.0 * 0.25 = 1.0 each!
    assert!(ideal_edge_lengths_manifold(&g_unw, &torus).unwrap() < 1e-12);
    assert!(ideal_edge_lengths_with_unit_manifold(&g_unw, &torus, 1.0).unwrap() < 1e-12);
    assert!(ideal_edge_lengths_scaled_manifold(&g_unw, &torus, 1.0).unwrap() < 1e-12);

    // Weighted graph with target lengths 2.0 -> (1.0 - 2.0)/2.0 = -0.5 => err^2 = 0.25 * 2 = 0.5
    let g_w =
        CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], Some(&[2.0, 2.0]), false, false).unwrap();
    assert!((ideal_edge_lengths_manifold(&g_w, &torus).unwrap() - 0.5).abs() < 1e-12);
    assert!((ideal_edge_lengths_scaled_manifold(&g_w, &torus, 0.5).unwrap()).abs() < 1e-12);

    let dists = vec![0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0];
    assert!(ideal_edge_lengths_from_distances_manifold(&g_unw, &torus, &dists).unwrap() < 1e-12);

    // Invalid inputs
    assert!(matches!(
        ideal_edge_lengths_with_unit_manifold(&g_unw, &torus, 0.0),
        Err(CoreError::InvalidInput { .. })
    ));
    assert!(matches!(
        ideal_edge_lengths_scaled_manifold(&g_unw, &torus, f64::NAN),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        ideal_edge_lengths_from_distances_manifold(&g_unw, &torus, &[0.0]),
        Err(CoreError::DimensionMismatch { .. })
    ));
}

#[test]
fn geometry_manifold_degeneracy_duplicates_and_geodesic_collisions() {
    // Spherical north-pole collapse on edge (0, 1): distinct longitudes, colat = 0 -> zero geodesic length!
    let g = CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3)], None, false, false).unwrap();
    let sph = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Spherical,
        vec![0.1, 1.7, 0.5, 0.5],
        vec![0.0, 0.0, 1.0, 1.0],
    )
    .unwrap();
    let deg = layout_degeneracy_manifold(&g, &sph).unwrap();
    assert_eq!(deg.num_nodes, 4);
    assert_eq!(deg.num_edges, 3);
    // Nodes 2 & 3 share exact coordinates; nodes 0 & 1 share north pole geodesically (zero-length edge (0,1) + (2,3))
    assert_eq!(deg.duplicate_nodes_f64, 2);
    assert!((deg.excess_duplicate_ratio - 0.25).abs() < 1e-12);
    assert_eq!(deg.zero_length_edges, 2);
    assert!((deg.zero_length_edge_ratio - 2.0 / 3.0).abs() < 1e-12);
    assert!((deg.near_collision_ratio - 1.0).abs() < 1e-12);

    // Torus wrap near-collision: (0.0, 0.5) and (1.0 - 1e-9, 0.5) are near-colliding across wrap boundary
    let g2 = CsrGraph::<f64>::from_edges(3, &[(0, 2)], None, false, false).unwrap();
    let torus = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.0, 1.0 - 1e-9, 0.5],
        vec![0.5, 0.5, 0.5],
    )
    .unwrap();
    let deg_t = layout_degeneracy_manifold(&g2, &torus).unwrap();
    assert_eq!(deg_t.duplicate_nodes_f64, 0);
    assert!((deg_t.near_collision_ratio - 2.0 / 3.0).abs() < 1e-12);

    // f32 truncation duplicate while f64 coordinates remain distinct
    let torus_f32_dup = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.0, 1e-50, 0.5],
        vec![0.5, 0.5, 0.5],
    )
    .unwrap();
    let deg_f32 = layout_degeneracy_manifold(&g2, &torus_f32_dup).unwrap();
    assert_eq!(deg_f32.duplicate_nodes_f64, 0);
    assert_eq!(deg_f32.duplicate_nodes_f32, 2);

    // HyperbolicLorentz non-degenerate layout
    let lor = ManifoldLayout2d::<f32>::init(ManifoldKind::HyperbolicLorentz, 4, 42, 1.0).unwrap();
    let deg_l = layout_degeneracy_manifold(&g, &lor).unwrap();
    assert_eq!(deg_l.duplicate_nodes_f64, 0);
    assert_eq!(deg_l.zero_length_edges, 0);
    assert_eq!(deg_l.near_collision_ratio, 0.0);

    // Empty layout
    let empty_g = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let empty_l = ManifoldLayout2d::<f64>::zeros(ManifoldKind::HyperbolicPoincare, 0);
    let deg_e = layout_degeneracy_manifold(&empty_g, &empty_l).unwrap();
    assert_eq!(deg_e.num_nodes, 0);
    assert_eq!(deg_e.near_collision_ratio, 0.0);
}
