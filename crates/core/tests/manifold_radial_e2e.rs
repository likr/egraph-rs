//! Opaque-box E2E tests for `RiemannianLayout::move_pair_radial` on
//! `ManifoldLayout2d` (`f64` and `f32`, all 4 manifold kinds, boundary and error guards).

use std::f64::consts::{PI, TAU};

use egraph_core::{CoreError, ManifoldKind, ManifoldLayout2d, MetricSpace, RiemannianLayout};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn manifold_radial_move_matches_move_pair_toward_all_kinds_f64_and_f32() {
    for kind in [
        ManifoldKind::HyperbolicPoincare,
        ManifoldKind::HyperbolicLorentz,
        ManifoldKind::Spherical,
        ManifoldKind::Torus2d,
    ] {
        let mut base64 = ManifoldLayout2d::<f64>::init(kind, 4, 31, 0.35).unwrap();
        if kind == ManifoldKind::Torus2d {
            base64.set_torus_extents(2.2, 1.4).unwrap();
        } else {
            base64.set_scale(1.5).unwrap();
        }
        let d64 = base64.distance(0, 1);
        let mut t64 = base64.clone();
        let mut r64 = base64.clone();
        t64.move_pair_toward(0, 1, 0.6 * d64, 0.3, 0.2).unwrap();
        r64.move_pair_radial(0, 1, 0.3 * 0.4 * d64, 0.2 * 0.4 * d64)
            .unwrap();
        for idx in 0..base64.len() {
            assert!(approx_eq(t64.x()[idx], r64.x()[idx], 1e-12));
            assert!(approx_eq(t64.y()[idx], r64.y()[idx], 1e-12));
        }
        for untouched in 2..base64.len() {
            assert_eq!(
                r64.x()[untouched].to_bits(),
                base64.x()[untouched].to_bits()
            );
            assert_eq!(
                r64.y()[untouched].to_bits(),
                base64.y()[untouched].to_bits()
            );
        }
        assert!(r64.validate_finite().is_ok());

        let mut base32 = ManifoldLayout2d::<f32>::init(kind, 4, 37, 0.35).unwrap();
        if kind == ManifoldKind::Torus2d {
            base32.set_torus_extents(2.0, 1.5).unwrap();
        } else {
            base32.set_scale(1.25).unwrap();
        }
        let d32 = base32.distance(0, 1);
        let mut t32 = base32.clone();
        let mut r32 = base32.clone();
        t32.move_pair_toward(0, 1, 0.6 * d32, 0.25, 0.25).unwrap();
        r32.move_pair_radial(0, 1, 0.25 * 0.4 * d32, 0.25 * 0.4 * d32)
            .unwrap();
        for idx in 0..base32.len() {
            assert!(approx_eq(t32.x()[idx] as f64, r32.x()[idx] as f64, 1e-5));
            assert!(approx_eq(t32.y()[idx] as f64, r32.y()[idx] as f64, 1e-5));
            if kind == ManifoldKind::HyperbolicPoincare {
                assert!(r32.x()[idx].hypot(r32.y()[idx]) < 1.0);
            } else if kind == ManifoldKind::Torus2d {
                assert!((0.0..1.0).contains(&r32.x()[idx]));
                assert!((0.0..1.0).contains(&r32.y()[idx]));
            } else if kind == ManifoldKind::Spherical {
                assert!(
                    (0.0..TAU as f32).contains(&r32.x()[idx])
                        && (0.0..=PI as f32).contains(&r32.y()[idx])
                );
            }
        }
        assert!(r32.validate_finite().is_ok());
    }
}

#[test]
fn manifold_radial_move_torus_wrap_and_spherical_exact_delta() {
    let mut torus =
        ManifoldLayout2d::<f64>::from_xy(ManifoldKind::Torus2d, vec![0.05, 0.95], vec![0.5, 0.5])
            .unwrap();
    let d_before = torus.distance(0, 1);
    assert!(approx_eq(d_before, 0.1, 1e-12));
    torus.move_pair_radial(0, 1, 0.02, 0.02).unwrap();
    assert!(approx_eq(torus.distance(0, 1), 0.06, 1e-12));
    assert!((0.0..1.0).contains(&torus.x()[0]) && (0.0..1.0).contains(&torus.x()[1]));

    let mut sphere =
        ManifoldLayout2d::<f64>::from_xy(ManifoldKind::Spherical, vec![0.1, 3.0], vec![0.05, 0.08])
            .unwrap();
    let ds_before = sphere.distance(0, 1);
    sphere.move_pair_radial(0, 1, 0.01, 0.01).unwrap();
    assert!(sphere.validate_finite().is_ok());
    assert!(approx_eq(sphere.distance(0, 1), ds_before - 0.02, 1e-12));
}

#[test]
fn manifold_radial_move_empty_singleton_oob_and_non_finite_guards() {
    let mut ml_0 = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Torus2d, 0);
    assert!(matches!(
        ml_0.move_pair_radial(0, 0, 0.1, 0.1),
        Err(CoreError::IndexOutOfBounds {
            index: 0,
            len: 0,
            ..
        })
    ));

    let mut ml_1 = ManifoldLayout2d::<f64>::init(ManifoldKind::Spherical, 1, 5, 0.3).unwrap();
    let ml_1_snap = ml_1.clone();
    ml_1.move_pair_radial(0, 0, 1.0, 1.0).unwrap();
    assert_eq!(ml_1, ml_1_snap);
    assert!(matches!(
        ml_1.move_pair_radial(1, 0, 0.1, 0.1),
        Err(CoreError::IndexOutOfBounds {
            index: 1,
            len: 1,
            ..
        })
    ));

    let mut ml =
        ManifoldLayout2d::<f64>::init(ManifoldKind::HyperbolicPoincare, 3, 4, 0.3).unwrap();
    let snap = ml.clone();
    ml.move_pair_radial(0, 1, 0.0, 0.0).unwrap();
    assert_eq!(ml, snap);

    for (i, j) in [(3, 0), (0, 3), (3, 3)] {
        assert!(matches!(
            ml.move_pair_radial(i, j, 0.1, 0.1),
            Err(CoreError::IndexOutOfBounds {
                index: 3,
                len: 3,
                ..
            })
        ));
        assert_eq!(ml, snap);
    }

    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            ml.move_pair_radial(0, 1, bad, 0.1),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            ml.move_pair_radial(0, 1, 0.1, bad),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert_eq!(ml, snap);
        assert!(matches!(
            ManifoldLayout2d::<f64>::from_xy(ManifoldKind::Torus2d, vec![bad, 0.5], vec![0.5, 0.5]),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }
}
