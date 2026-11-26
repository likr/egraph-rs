use egraph_core::{CsrGraph, ManifoldKind, ManifoldLayout2d};

use super::*;

fn brute_rng_edges_manifold(layout: &ManifoldLayout2d<f64>) -> Vec<(u32, u32)> {
    let n = layout.len();
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let d_ij = layout.geodesic_distance(i, j).unwrap();
            if d_ij <= 0.0 {
                continue;
            }
            let thresh = d_ij * (1.0 - 1e-12);
            let mut ok = true;
            for r in 0..n {
                if r == i || r == j {
                    continue;
                }
                if layout.geodesic_distance(i, r).unwrap() < thresh
                    && layout.geodesic_distance(j, r).unwrap() < thresh
                {
                    ok = false;
                    break;
                }
            }
            if ok {
                edges.push((i as u32, j as u32));
            }
        }
    }
    edges
}

fn brute_gg_edges_manifold(layout: &ManifoldLayout2d<f64>) -> Vec<(u32, u32)> {
    let n = layout.len();
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let d_ij = layout.geodesic_distance(i, j).unwrap();
            if d_ij <= 0.0 {
                continue;
            }
            let mid = manifold_geodesic_interpolate_point(layout, i, j, 0.5).unwrap();
            let thresh = 0.5 * d_ij * (1.0 + 1e-12);
            let mut ok = true;
            for r in 0..n {
                if r == i || r == j {
                    continue;
                }
                if manifold_point_distance_to_node(layout, mid, r).unwrap() <= thresh {
                    ok = false;
                    break;
                }
            }
            if ok {
                edges.push((i as u32, j as u32));
            }
        }
    }
    edges
}

#[test]
fn proximity_manifold_rng_gg_oracle_all_kinds() {
    for kind in [
        ManifoldKind::HyperbolicPoincare,
        ManifoldKind::HyperbolicLorentz,
        ManifoldKind::Spherical,
        ManifoldKind::Torus2d,
    ] {
        let mut ml = ManifoldLayout2d::<f64>::init(kind, 14, 77, 0.7).unwrap();
        if kind == ManifoldKind::Torus2d {
            ml.set_torus_extents(2.5, 0.8).unwrap();
        } else {
            ml.set_scale(2.2).unwrap();
        }
        let rng = relative_neighborhood_edges_manifold(&ml).unwrap();
        let rng_oracle = brute_rng_edges_manifold(&ml);
        assert_eq!(rng, rng_oracle, "RNG mismatch for {kind:?}");

        let gg = gabriel_graph_edges_manifold(&ml).unwrap();
        let gg_oracle = brute_gg_edges_manifold(&ml);
        assert_eq!(gg, gg_oracle, "GG mismatch for {kind:?}");
        // RNG is a subset of GG on Euclidean/flat regions and close on smooth manifolds;
        // verify non-empty and valid preservation scores.
        let g = CsrGraph::<f64>::from_edges(14, &gg, None, false, false).unwrap();
        assert!((gabriel_graph_preservation_manifold(&g, &ml).unwrap() - 1.0).abs() < 1e-12);
        assert!(gabriel_graph_property_manifold(&g, &ml).unwrap() < 1e-12);
        let srng = relative_neighborhood_preservation_manifold(&g, &ml).unwrap();
        assert!((0.0..=1.0).contains(&srng));
    }
}

#[test]
fn proximity_manifold_node_resolution_scale_and_collapse() {
    let mut ml =
        ManifoldLayout2d::<f64>::init(ManifoldKind::HyperbolicLorentz, 8, 19, 1.5).unwrap();
    let nr1 = node_resolution_manifold(&ml).unwrap();
    ml.set_scale(4.0).unwrap();
    let nr2 = node_resolution_manifold(&ml).unwrap();
    assert!((nr1 - nr2).abs() < 1e-12);

    // Completely collapsed layout on Sphere (all at north pole with varying longitudes)
    let collapsed = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Spherical,
        vec![0.1, 0.5, 1.0, 2.0],
        vec![0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    assert_eq!(node_resolution_manifold(&collapsed).unwrap(), 6.0);

    // Gabriel violation check on a collinear triple (0 - 2 with 1 at midpoint)
    let g = CsrGraph::<f64>::from_edges(3, &[(0, 2)], None, false, false).unwrap();
    let torus = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.1, 0.25, 0.4],
        vec![0.5, 0.5, 0.5],
    )
    .unwrap();
    let viol = gabriel_graph_property_manifold(&g, &torus).unwrap();
    // radius = 0.15, midpoint at 0.25 -> vertex 1 is at distance 0 -> violation = 0.15^2
    assert!((viol - 0.15_f64.powi(2)).abs() < 1e-12, "got {viol}");
}

#[test]
fn proximity_manifold_empty_and_f32() {
    let empty_g = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let empty_l = ManifoldLayout2d::<f32>::zeros(ManifoldKind::Torus2d, 0);
    assert_eq!(node_resolution_manifold(&empty_l).unwrap(), 0.0);
    assert_eq!(
        gabriel_graph_property_manifold(&empty_g, &empty_l).unwrap(),
        0.0
    );
    assert_eq!(
        relative_neighborhood_preservation_manifold(&empty_g, &empty_l).unwrap(),
        0.0
    );
    assert_eq!(
        gabriel_graph_preservation_manifold(&empty_g, &empty_l).unwrap(),
        0.0
    );
}
