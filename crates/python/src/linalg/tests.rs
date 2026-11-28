//! Pure-Rust unit tests for linear algebra, spectral, Chebyshev, and AMG helpers.

use egraph::core::CsrGraph;
use egraph::linalg::{LaplacianKind, spectral_moment_stats};

use super::amg_helpers::parse_amg_laplacian_kind;
use super::chebyshev::{resolve_dog_r1_defaults, validate_dog_jl_params};
use super::spectral::validate_koren_params;

#[test]
fn test_linalg_spectral_and_amg_helpers() {
    let g = CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3)], None, false, false).unwrap();
    let stats = spectral_moment_stats(&g);
    assert!(stats.sigma_lambda_sq >= 0.0);
    assert!(stats.sigma_lambda >= 0.0);
    assert!(stats.leaf_fraction > 0.0);

    let (t1, t2, alpha) = resolve_dog_r1_defaults(&stats, None, Some(2.0), None);
    assert!((t1 - 1.0 / (1.0 + stats.sigma_lambda)).abs() < 1e-12);
    assert!((t2 - 2.0).abs() < 1e-12);
    assert!((alpha - (1.0_f64 - stats.sigma_lambda).max(0.1)).abs() < 1e-12);

    assert!(validate_koren_params(1e-4, 300, Some(1.0)).is_ok());
    assert!(validate_koren_params(0.0, 300, None).is_err());
    assert!(validate_koren_params(f64::NAN, 300, None).is_err());
    assert!(validate_koren_params(1e-4, 0, None).is_err());
    assert!(validate_koren_params(1e-4, 300, Some(0.0)).is_err());

    assert!(validate_dog_jl_params(25, 64, 1e-4).is_ok());
    assert!(validate_dog_jl_params(0, 64, 1e-4).is_err());
    assert!(validate_dog_jl_params(25, 0, 1e-4).is_err());
    assert!(validate_dog_jl_params(25, 64, -1e-4).is_err());
    assert!(validate_dog_jl_params(25, 64, f64::NAN).is_err());

    assert_eq!(
        parse_amg_laplacian_kind("combinatorial").unwrap(),
        LaplacianKind::Combinatorial
    );
    assert_eq!(
        parse_amg_laplacian_kind("symmetric_normalized").unwrap(),
        LaplacianKind::SymmetricNormalized
    );
    assert!(parse_amg_laplacian_kind("random_walk").is_err());
    assert!(parse_amg_laplacian_kind("rw").is_err());
    assert!(parse_amg_laplacian_kind("unknown").is_err());
}
