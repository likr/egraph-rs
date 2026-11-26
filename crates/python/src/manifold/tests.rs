use egraph::core as egraph_core;
use egraph_core::ManifoldKind;

use super::geodesic::{manifold_kind_str, parse_manifold_kind};

#[test]
fn test_manifold_kind_parsers() {
    assert_eq!(
        parse_manifold_kind("Hyperbolic_Poincare").unwrap(),
        ManifoldKind::HyperbolicPoincare
    );
    assert_eq!(
        parse_manifold_kind("poincare").unwrap(),
        ManifoldKind::HyperbolicPoincare
    );
    assert_eq!(
        parse_manifold_kind("SPHERE").unwrap(),
        ManifoldKind::Spherical
    );
    assert_eq!(
        parse_manifold_kind("torus-2d").unwrap(),
        ManifoldKind::Torus2d
    );
    assert_eq!(
        manifold_kind_str(ManifoldKind::HyperbolicPoincare),
        "hyperbolic"
    );
    assert_eq!(manifold_kind_str(ManifoldKind::Spherical), "spherical");
    assert_eq!(manifold_kind_str(ManifoldKind::Torus2d), "torus");
    assert!(parse_manifold_kind("euclidean").is_err());
}
