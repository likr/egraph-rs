//! Opaque-box E2E tests for `RiemannianLayout::move_pair_radial` and
//! `move_pair_toward` parity across Euclidean spaces and custom Riemannian layouts.

use std::f64::consts::{PI, TAU};

use egraph_core::{
    CoreError, CoreResult, Layout2d, Layout3d, LayoutNd, LayoutStorage, MetricSpace,
    RiemannianLayout, Scalar,
};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

struct Circle1dLayout {
    theta: Vec<f64>,
}

impl Circle1dLayout {
    fn new(theta: Vec<f64>) -> Self {
        Self { theta }
    }
}

impl LayoutStorage for Circle1dLayout {
    type Scalar = f64;

    fn len(&self) -> usize {
        self.theta.len()
    }

    fn validate_finite(&self) -> CoreResult<()> {
        if self.theta.iter().all(|v| v.is_finite()) {
            Ok(())
        } else {
            Err(CoreError::non_finite("Circle1dLayout"))
        }
    }
}

impl MetricSpace for Circle1dLayout {
    fn distance(&self, i: usize, j: usize) -> f64 {
        let d = (self.theta[j] - self.theta[i]).rem_euclid(TAU);
        if d > PI { TAU - d } else { d }
    }
}

impl RiemannianLayout for Circle1dLayout {
    fn tangent_dim(&self) -> usize {
        1
    }

    fn log_map_into(&self, i: usize, j: usize, out: &mut [f64]) -> CoreResult<()> {
        if i >= self.len() || j >= self.len() {
            return Err(CoreError::index_out_of_bounds(
                i.max(j),
                self.len(),
                "Circle1dLayout::log_map_into",
            ));
        }
        if out.is_empty() {
            return Err(CoreError::dimension_mismatch(
                1,
                0,
                "Circle1dLayout::log_map_into",
            ));
        }
        let mut d = (self.theta[j] - self.theta[i]).rem_euclid(TAU);
        if d > PI {
            d -= TAU;
        }
        out[0] = d;
        Ok(())
    }

    fn exp_map(&mut self, i: usize, delta: &[f64]) -> CoreResult<()> {
        if i >= self.len() {
            return Err(CoreError::index_out_of_bounds(
                i,
                self.len(),
                "Circle1dLayout::exp_map",
            ));
        }
        if delta.len() != 1 {
            return Err(CoreError::dimension_mismatch(
                1,
                delta.len(),
                "Circle1dLayout::exp_map",
            ));
        }
        if !delta[0].is_finite() {
            return Err(CoreError::non_finite("Circle1dLayout::exp_map"));
        }
        self.theta[i] = (self.theta[i] + delta[0]).rem_euclid(TAU);
        Ok(())
    }

    fn parallel_transport_into(&self, i: usize, delta: &[f64], vec: &mut [f64]) -> CoreResult<()> {
        if i >= self.len() {
            return Err(CoreError::index_out_of_bounds(
                i,
                self.len(),
                "Circle1dLayout::parallel_transport_into",
            ));
        }
        if delta.len() != 1 || vec.len() != 1 {
            return Err(CoreError::dimension_mismatch(
                1,
                delta.len().min(vec.len()),
                "Circle1dLayout::parallel_transport_into",
            ));
        }
        if !delta[0].is_finite() || !vec[0].is_finite() {
            return Err(CoreError::non_finite(
                "Circle1dLayout::parallel_transport_into",
            ));
        }
        Ok(())
    }
}

fn verify_radial_sign_and_asymmetry<L: RiemannianLayout + Clone>(base: &L, tol: f64, label: &str) {
    let d0 = base.distance(0, 1);
    let mut attract = base.clone();
    attract.move_pair_radial(0, 1, 0.25, 0.15).unwrap();
    let d_attract = attract.distance(0, 1);
    assert!(
        approx_eq(d_attract, d0 - 0.40, tol),
        "{label}: expected attract distance {}, got {d_attract}",
        d0 - 0.40
    );

    let mut repel = base.clone();
    repel.move_pair_radial(0, 1, -0.30, -0.20).unwrap();
    let d_repel = repel.distance(0, 1);
    assert!(
        approx_eq(d_repel, d0 + 0.50, tol),
        "{label}: expected repel distance {}, got {d_repel}",
        d0 + 0.50
    );

    let mut one_sided = base.clone();
    one_sided.move_pair_radial(0, 1, 0.35, 0.0).unwrap();
    assert!(approx_eq(one_sided.distance(0, 1), d0 - 0.35, tol));
}

fn check_parity_2d<S: Scalar>(base: &Layout2d<S>, target_d: f64, wi: f64, wj: f64, tol: f64) {
    let d = base.distance(0, 1);
    let mut via_toward = base.clone();
    let mut via_radial = base.clone();
    via_toward.move_pair_toward(0, 1, target_d, wi, wj).unwrap();
    via_radial
        .move_pair_radial(0, 1, wi * (d - target_d), wj * (d - target_d))
        .unwrap();
    for idx in 0..base.len() {
        assert!(approx_eq(
            via_toward.x()[idx].to_f64(),
            via_radial.x()[idx].to_f64(),
            tol
        ));
        assert!(approx_eq(
            via_toward.y()[idx].to_f64(),
            via_radial.y()[idx].to_f64(),
            tol
        ));
    }
    for untouched in 2..base.len() {
        assert_eq!(via_radial.x()[untouched], base.x()[untouched]);
        assert_eq!(via_radial.y()[untouched], base.y()[untouched]);
    }
    assert!(approx_eq(
        via_toward.distance(0, 1),
        via_radial.distance(0, 1),
        tol
    ));
}

fn check_parity_3d<S: Scalar>(base: &Layout3d<S>, target_d: f64, wi: f64, wj: f64, tol: f64) {
    let d = base.distance(0, 1);
    let mut via_toward = base.clone();
    let mut via_radial = base.clone();
    via_toward.move_pair_toward(0, 1, target_d, wi, wj).unwrap();
    via_radial
        .move_pair_radial(0, 1, wi * (d - target_d), wj * (d - target_d))
        .unwrap();
    for idx in 0..base.len() {
        assert!(approx_eq(
            via_toward.x()[idx].to_f64(),
            via_radial.x()[idx].to_f64(),
            tol
        ));
        assert!(approx_eq(
            via_toward.y()[idx].to_f64(),
            via_radial.y()[idx].to_f64(),
            tol
        ));
        assert!(approx_eq(
            via_toward.z()[idx].to_f64(),
            via_radial.z()[idx].to_f64(),
            tol
        ));
    }
    for untouched in 2..base.len() {
        assert_eq!(via_radial.x()[untouched], base.x()[untouched]);
        assert_eq!(via_radial.y()[untouched], base.y()[untouched]);
        assert_eq!(via_radial.z()[untouched], base.z()[untouched]);
    }
    assert!(approx_eq(
        via_toward.distance(0, 1),
        via_radial.distance(0, 1),
        tol
    ));
}

fn check_parity_nd<S: Scalar>(base: &LayoutNd<S>, target_d: f64, wi: f64, wj: f64, tol: f64) {
    let d = base.distance(0, 1);
    let dims = base.dims();
    let mut via_toward = base.clone();
    let mut via_radial = base.clone();
    via_toward.move_pair_toward(0, 1, target_d, wi, wj).unwrap();
    via_radial
        .move_pair_radial(0, 1, wi * (d - target_d), wj * (d - target_d))
        .unwrap();
    for (&a, &b) in via_toward.coords().iter().zip(via_radial.coords().iter()) {
        assert!(approx_eq(a.to_f64(), b.to_f64(), tol));
    }
    let start_untouched = 2 * dims;
    assert_eq!(
        &via_radial.coords()[start_untouched..],
        &base.coords()[start_untouched..]
    );
    assert!(approx_eq(
        via_toward.distance(0, 1),
        via_radial.distance(0, 1),
        tol
    ));
}

#[test]
fn radial_move_positive_and_negative_euclidean_2d_3d_nd() {
    let l2_f64 = Layout2d::<f64>::from_xy(vec![0.0, 3.0], vec![0.0, 4.0]).unwrap();
    verify_radial_sign_and_asymmetry(&l2_f64, 1e-12, "Layout2d<f64>");

    let mut l2_asym = l2_f64.clone();
    l2_asym.move_pair_radial(0, 1, 0.5, 0.0).unwrap();
    assert!(l2_asym.x()[0] > 0.0 && l2_asym.y()[0] > 0.0);
    assert_eq!(l2_asym.x()[1].to_bits(), 3.0_f64.to_bits());
    assert_eq!(l2_asym.y()[1].to_bits(), 4.0_f64.to_bits());

    let l2_f32 = Layout2d::<f32>::from_xy(vec![0.0, 3.0], vec![0.0, 4.0]).unwrap();
    verify_radial_sign_and_asymmetry(&l2_f32, 1e-5, "Layout2d<f32>");

    let l3_f64 = Layout3d::<f64>::from_xyz(vec![0.0, 2.0], vec![0.0, 3.0], vec![0.0, 6.0]).unwrap();
    verify_radial_sign_and_asymmetry(&l3_f64, 1e-12, "Layout3d<f64>");

    let l3_f32 = Layout3d::<f32>::from_xyz(vec![0.0, 2.0], vec![0.0, 3.0], vec![0.0, 6.0]).unwrap();
    verify_radial_sign_and_asymmetry(&l3_f32, 1e-5, "Layout3d<f32>");

    let lnd_f64 =
        LayoutNd::<f64>::from_flat(2, 5, &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 0.0, 0.0])
            .unwrap();
    verify_radial_sign_and_asymmetry(&lnd_f64, 1e-12, "LayoutNd<f64>");

    let lnd_f32 =
        LayoutNd::<f32>::from_flat(2, 5, &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 0.0, 0.0])
            .unwrap();
    verify_radial_sign_and_asymmetry(&lnd_f32, 1e-5, "LayoutNd<f32>");
}

#[test]
fn radial_move_matches_move_pair_toward_euclidean() {
    let l2 = Layout2d::<f64>::init(4, 11, 4.0).unwrap();
    check_parity_2d(&l2, 1.25, 0.3, 0.45, 1e-13);
    let l2_32 = Layout2d::<f32>::init(4, 11, 4.0).unwrap();
    check_parity_2d(&l2_32, 1.25, 0.3, 0.45, 1e-5);

    let l3 = Layout3d::<f64>::init(4, 17, 4.0).unwrap();
    check_parity_3d(&l3, 1.5, 0.25, 0.35, 1e-13);
    let l3_32 = Layout3d::<f32>::init(4, 17, 4.0).unwrap();
    check_parity_3d(&l3_32, 1.5, 0.25, 0.35, 1e-5);

    let lnd = LayoutNd::<f64>::init(4, 5, 23, 4.0).unwrap();
    check_parity_nd(&lnd, 2.0, 0.2, 0.4, 1e-13);
    let lnd_32 = LayoutNd::<f32>::init(4, 5, 23, 4.0).unwrap();
    check_parity_nd(&lnd_32, 2.0, 0.2, 0.4, 1e-5);
}

#[test]
fn radial_move_default_trait_impl_custom_riemannian_space() {
    let mut c1 = Circle1dLayout::new(vec![0.2, 1.4, 5.9]);
    let mut c2 = Circle1dLayout::new(vec![0.2, 1.4, 5.9]);
    let d01 = c1.distance(0, 1);
    c1.move_pair_toward(0, 1, 0.5, 0.25, 0.25).unwrap();
    c2.move_pair_radial(0, 1, 0.25 * (d01 - 0.5), 0.25 * (d01 - 0.5))
        .unwrap();
    assert!(approx_eq(c1.theta[0], c2.theta[0], 1e-14));
    assert!(approx_eq(c1.theta[1], c2.theta[1], 1e-14));
    assert!(c2.distance(0, 1) < d01);
}

#[test]
fn radial_move_zero_deltas_empty_singleton_and_self_pair_euclidean() {
    let mut l2_0 = Layout2d::<f64>::zeros(0);
    let mut l3_0 = Layout3d::<f64>::zeros(0);
    let mut lnd_0 = LayoutNd::<f64>::zeros(0, 3).unwrap();
    for res in [
        l2_0.move_pair_radial(0, 0, 0.1, 0.1),
        l3_0.move_pair_radial(0, 0, 0.1, 0.1),
        lnd_0.move_pair_radial(0, 0, 0.1, 0.1),
    ] {
        assert!(matches!(
            res,
            Err(CoreError::IndexOutOfBounds {
                index: 0,
                len: 0,
                ..
            })
        ));
    }

    let mut l2_1 = Layout2d::<f64>::from_xy(vec![1.25], vec![-2.5]).unwrap();
    let l2_1_snap = l2_1.clone();
    l2_1.move_pair_radial(0, 0, 3.0, -2.0).unwrap();
    assert_eq!(l2_1, l2_1_snap);
    assert!(matches!(
        l2_1.move_pair_radial(0, 1, 0.1, 0.1),
        Err(CoreError::IndexOutOfBounds {
            index: 1,
            len: 1,
            ..
        })
    ));
    assert_eq!(l2_1, l2_1_snap);

    let mut l2 = Layout2d::<f64>::init(3, 5, 2.0).unwrap();
    let l2_snap = l2.clone();
    l2.move_pair_radial(0, 1, 0.0, 0.0).unwrap();
    assert_eq!(l2, l2_snap);
    l2.move_pair_radial(0, 0, 10.0, -5.0).unwrap();
    assert_eq!(l2, l2_snap);

    let mut l3 = Layout3d::<f64>::init(3, 6, 2.0).unwrap();
    let l3_snap = l3.clone();
    l3.move_pair_radial(0, 1, 0.0, 0.0).unwrap();
    assert_eq!(l3, l3_snap);

    let mut lnd = LayoutNd::<f64>::init(3, 4, 7, 2.0).unwrap();
    let lnd_snap = lnd.clone();
    lnd.move_pair_radial(0, 1, 0.0, 0.0).unwrap();
    assert_eq!(lnd, lnd_snap);
}

#[test]
fn radial_move_out_of_bounds_and_non_finite_euclidean_without_mutation() {
    fn check_oob<L: RiemannianLayout + Clone + PartialEq + std::fmt::Debug>(
        mut layout: L,
        n: usize,
    ) {
        let snap = layout.clone();
        for (i, j, exp_idx) in [(n, 0, n), (0, n, n), (n, n, n)] {
            assert!(matches!(
                layout.move_pair_radial(i, j, 0.1, 0.1),
                Err(CoreError::IndexOutOfBounds { index, len, .. }) if index == exp_idx && len == n
            ));
            assert_eq!(layout, snap);
        }
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                layout.move_pair_radial(0, 1, bad, 0.1),
                Err(CoreError::NonFiniteValue { .. })
            ));
            assert!(matches!(
                layout.move_pair_radial(0, 1, 0.1, bad),
                Err(CoreError::NonFiniteValue { .. })
            ));
            assert_eq!(layout, snap);
        }
    }

    check_oob(Layout2d::<f64>::init(3, 1, 1.0).unwrap(), 3);
    check_oob(Layout3d::<f64>::init(3, 2, 1.0).unwrap(), 3);
    check_oob(LayoutNd::<f64>::init(3, 4, 3, 1.0).unwrap(), 3);

    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut l2 = Layout2d::<f64>::from_xy(vec![bad, 2.0], vec![0.0, 3.0]).unwrap();
        assert!(matches!(
            l2.move_pair_radial(0, 1, 0.1, 0.1),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert_eq!(l2.x()[1].to_bits(), 2.0_f64.to_bits());
        assert_eq!(l2.y()[1].to_bits(), 3.0_f64.to_bits());

        let mut l3 =
            Layout3d::<f64>::from_xyz(vec![0.0, 2.0], vec![bad, 3.0], vec![0.0, 4.0]).unwrap();
        assert!(matches!(
            l3.move_pair_radial(0, 1, 0.1, 0.1),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert_eq!(l3.z()[1].to_bits(), 4.0_f64.to_bits());

        let mut lnd = LayoutNd::<f64>::from_flat(2, 3, &[0.0, bad, 0.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(matches!(
            lnd.move_pair_radial(0, 1, 0.1, 0.1),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert_eq!(lnd.coords()[3..], [2.0, 3.0, 4.0]);
    }
}

#[test]
fn radial_move_coincident_and_f32_overflow_atomic_no_mutation() {
    let mut l2_co = Layout2d::<f64>::from_xy(vec![1.0, 1.0], vec![2.0, 2.0]).unwrap();
    let snap_co = l2_co.clone();
    l2_co.move_pair_radial(0, 1, 0.5, 0.5).unwrap();
    assert_eq!(l2_co, snap_co);

    let mut l2 = Layout2d::<f32>::from_xy(vec![0.0, 1.0], vec![0.0, 0.0]).unwrap();
    let snap = l2.clone();
    assert!(matches!(
        l2.move_pair_radial(0, 1, 1e39_f64, 0.1),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert_eq!(l2, snap);

    let mut l3 = Layout3d::<f32>::from_xyz(vec![0.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0]).unwrap();
    let snap3 = l3.clone();
    assert!(matches!(
        l3.move_pair_radial(0, 1, 0.1, 1e39_f64),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert_eq!(l3, snap3);

    let mut lnd = LayoutNd::<f32>::from_flat(2, 2, &[1.0, 0.0, 2.0, 3.0e38]).unwrap();
    let snap_nd = lnd.clone();
    assert!(matches!(
        lnd.move_pair_radial(0, 1, -1e39_f64, -1e39_f64),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert_eq!(lnd, snap_nd);
}
