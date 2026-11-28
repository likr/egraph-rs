use super::*;
use crate::error::CoreError;
use crate::layout::{Layout2d, Layout3d, LayoutNd};

fn check_riemannian_contract<L: RiemannianLayout>(layout: &mut L) {
    let dim = layout.tangent_dim();
    let d01 = layout.distance(0, 1);
    let mut log01 = vec![0.0; dim];
    layout.log_map_into(0, 1, &mut log01).unwrap();
    let norm01 = log01.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!((norm01 - d01).abs() < 1e-12);

    let mut vec = vec![1.25; dim];
    let vec_before = vec.clone();
    layout.parallel_transport_into(0, &log01, &mut vec).unwrap();
    assert_eq!(vec, vec_before);

    let mut vec_combined = vec_before.clone();
    layout
        .exp_map_and_transport(0, &log01, &mut vec_combined)
        .unwrap();
    assert_eq!(vec_combined, vec_before);
    assert!(layout.distance(0, 1) < 1e-12);
}

#[test]
fn test_euclidean_traits_2d_3d_nd() {
    let mut l2 = Layout2d::from_xy(vec![0.0_f64, 3.0], vec![0.0, 4.0]).unwrap();
    assert_eq!(LayoutStorage::len(&l2), 2);
    assert!((l2.distance(0, 1) - 5.0).abs() < 1e-12);
    assert_eq!(l2.dims(), 2);
    assert_eq!(l2.coord(1, 1).unwrap(), 4.0);
    l2.set_coord(1, 1, 4.0).unwrap();
    check_riemannian_contract(&mut l2);

    let mut l3 = Layout3d::from_xyz(vec![0.0_f64, 2.0], vec![0.0, 3.0], vec![0.0, 6.0]).unwrap();
    assert_eq!(LayoutStorage::len(&l3), 2);
    assert!((l3.distance(0, 1) - 7.0).abs() < 1e-12);
    assert_eq!(l3.dims(), 3);
    assert_eq!(l3.coord(1, 2).unwrap(), 6.0);
    l3.set_coord(1, 2, 6.0).unwrap();
    check_riemannian_contract(&mut l3);

    let mut lnd = LayoutNd::from_flat(2, 4, &[0.0_f64, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 4.0]).unwrap();
    assert_eq!(LayoutStorage::len(&lnd), 2);
    assert!((lnd.distance(0, 1) - 5.0).abs() < 1e-12);
    assert_eq!(lnd.dims(), 4);
    assert_eq!(lnd.coord(1, 3).unwrap(), 4.0);
    check_riemannian_contract(&mut lnd);
}

#[test]
fn test_move_pair_toward_and_error_validation() {
    let mut l2 = Layout2d::from_xy(vec![0.0_f64, 10.0], vec![0.0, 0.0]).unwrap();
    l2.move_pair_toward(0, 1, 4.0, 0.25, 0.25).unwrap();
    // Distance moved from 10 to 10 - (0.25 + 0.25) * (10 - 4) = 7.0.
    assert!((l2.distance(0, 1) - 7.0).abs() < 1e-12);
    assert!((l2.x()[0] - 1.5).abs() < 1e-12);
    assert!((l2.x()[1] - 8.5).abs() < 1e-12);

    let mut l3 = Layout3d::from_xyz(vec![0.0_f64, 10.0], vec![0.0, 0.0], vec![0.0, 0.0]).unwrap();
    l3.move_pair_toward(0, 1, 4.0, 0.5, 0.0).unwrap();
    assert!((l3.distance(0, 1) - 7.0).abs() < 1e-12);
    assert!((l3.x()[0] - 3.0).abs() < 1e-12);
    assert_eq!(l3.x()[1], 10.0);

    let mut lnd = LayoutNd::from_flat(2, 3, &[0.0_f64, 0.0, 0.0, 10.0, 0.0, 0.0]).unwrap();
    lnd.move_pair_toward(0, 1, 4.0, 0.25, 0.25).unwrap();
    assert!((lnd.distance(0, 1) - 7.0).abs() < 1e-12);

    // Self-pair is a clean no-op after validation.
    l2.move_pair_toward(0, 0, 1.0, 0.5, 0.5).unwrap();

    // Error paths: out of bounds, wrong buffer dims, non-finite inputs.
    assert!(matches!(
        l2.coord(2, 0),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        l2.coord(0, 2),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        l2.set_coord(0, 0, f64::NAN),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        l3.exp_map(0, &[1.0, 2.0]),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        lnd.parallel_transport_into(0, &[0.0, 0.0, 0.0], &mut [f64::NAN, 0.0, 0.0]),
        Err(CoreError::NonFiniteValue { .. })
    ));
    let mut bad32 = Layout2d::<f32>::zeros(2);
    assert!(matches!(
        bad32.set_coord(0, 0, 1e40),
        Err(CoreError::NonFiniteValue { .. })
    ));
}

#[test]
fn test_euclidean_storage_views_and_repulsion_backend_blanket() {
    let mut l3 = Layout3d::from_xyz(vec![1.0_f64, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]).unwrap();
    match l3.as_euclidean_storage() {
        EuclideanStorageRef::Soa3d { x, y, z } => {
            assert_eq!(x, &[1.0, 2.0]);
            assert_eq!(y, &[3.0, 4.0]);
            assert_eq!(z, &[5.0, 6.0]);
        }
        _ => panic!("expected Soa3d"),
    }
    if let EuclideanStorageMut::Soa3d { x, .. } = l3.as_euclidean_storage_mut() {
        x[0] = 9.0;
    }
    assert_eq!(l3.x()[0], 9.0);

    struct ZeroBackend;
    impl<L: LayoutStorage + ?Sized> RepulsionBackend<L> for ZeroBackend {
        fn accumulate_repulsion(
            &mut self,
            layout: &L,
            forces: &mut [f64],
        ) -> CoreResult<Option<f64>> {
            layout.validate_finite()?;
            forces.fill(1.0);
            Ok(Some(layout.len() as f64))
        }
    }

    let mut raw = ZeroBackend;
    let mut r: &mut dyn RepulsionBackend<Layout3d<f64>> = &mut raw;
    let mut forces = vec![0.0_f64; 6];
    assert_eq!(
        (&mut r).accumulate_repulsion(&l3, &mut forces).unwrap(),
        Some(2.0)
    );
    assert_eq!(forces, vec![1.0; 6]);

    let mut boxed: Box<dyn RepulsionBackend<Layout3d<f64>>> = Box::new(ZeroBackend);
    assert_eq!(
        boxed.accumulate_repulsion(&l3, &mut forces).unwrap(),
        Some(2.0)
    );
}

#[test]
fn test_riemannian_default_move_pair_radial_and_delegation() {
    struct Wrapper2d(Layout2d<f64>);

    impl LayoutStorage for Wrapper2d {
        type Scalar = f64;

        fn len(&self) -> usize {
            LayoutStorage::len(&self.0)
        }

        fn validate_finite(&self) -> CoreResult<()> {
            LayoutStorage::validate_finite(&self.0)
        }
    }

    impl MetricSpace for Wrapper2d {
        fn distance(&self, i: usize, j: usize) -> f64 {
            MetricSpace::distance(&self.0, i, j)
        }
    }

    impl RiemannianLayout for Wrapper2d {
        fn tangent_dim(&self) -> usize {
            2
        }

        fn log_map_into(&self, i: usize, j: usize, out: &mut [f64]) -> CoreResult<()> {
            RiemannianLayout::log_map_into(&self.0, i, j, out)
        }

        fn exp_map(&mut self, i: usize, delta: &[f64]) -> CoreResult<()> {
            RiemannianLayout::exp_map(&mut self.0, i, delta)
        }

        fn parallel_transport_into(
            &self,
            i: usize,
            delta: &[f64],
            vec: &mut [f64],
        ) -> CoreResult<()> {
            RiemannianLayout::parallel_transport_into(&self.0, i, delta, vec)
        }
    }

    // Positive, negative, asymmetric, zero, self-pair, and near-coincident steps.
    let base = Layout2d::from_xy(vec![1.0, 4.0], vec![2.0, 6.0]).unwrap();
    let mut wrapped = Wrapper2d(base.clone());
    let mut direct = base.clone();
    wrapped.move_pair_radial(0, 1, 1.25, 0.75).unwrap();
    direct.move_pair_radial(0, 1, 1.25, 0.75).unwrap();
    assert!((wrapped.distance(0, 1) - 3.0).abs() < 1e-12);
    assert!((wrapped.0.x()[0] - direct.x()[0]).abs() < 1e-12);
    assert!((wrapped.0.y()[1] - direct.y()[1]).abs() < 1e-12);

    // Negative radial step pushes endpoints apart.
    wrapped.move_pair_radial(0, 1, -0.5, -1.5).unwrap();
    assert!((wrapped.distance(0, 1) - 5.0).abs() < 1e-12);

    // Asymmetric one-sided step leaves endpoint 0 bit-exact untouched.
    let before_x0 = wrapped.0.x()[0].to_bits();
    let before_y0 = wrapped.0.y()[0].to_bits();
    wrapped.move_pair_radial(0, 1, 0.0, 1.0).unwrap();
    assert_eq!(wrapped.0.x()[0].to_bits(), before_x0);
    assert_eq!(wrapped.0.y()[0].to_bits(), before_y0);
    assert!((wrapped.distance(0, 1) - 4.0).abs() < 1e-12);

    // Zero delta and self-pair are clean no-ops.
    let snap = wrapped.0.clone();
    wrapped.move_pair_radial(0, 1, 0.0, 0.0).unwrap();
    wrapped.move_pair_radial(0, 0, 1.0, 1.0).unwrap();
    assert_eq!(wrapped.0, snap);

    // Near-coincident d <= 1e-15 is a clean no-op.
    let mut tiny = Wrapper2d(Layout2d::from_xy(vec![1.0, 1.0 + 1e-16], vec![2.0, 2.0]).unwrap());
    let tiny_snap = tiny.0.clone();
    tiny.move_pair_radial(0, 1, 0.5, 0.5).unwrap();
    assert_eq!(tiny.0, tiny_snap);

    // Default move_pair_toward delegates to move_pair_radial and matches Layout2d.
    let mut w_toward = Wrapper2d(base.clone());
    let mut d_toward = base;
    w_toward.move_pair_toward(0, 1, 2.0, 0.3, 0.2).unwrap();
    d_toward.move_pair_toward(0, 1, 2.0, 0.3, 0.2).unwrap();
    assert!((w_toward.0.x()[0] - d_toward.x()[0]).abs() < 1e-12);
    assert!((w_toward.0.y()[0] - d_toward.y()[0]).abs() < 1e-12);
    assert!((w_toward.0.x()[1] - d_toward.x()[1]).abs() < 1e-12);
    assert!((w_toward.0.y()[1] - d_toward.y()[1]).abs() < 1e-12);

    // Validation errors on default impl.
    assert!(matches!(
        w_toward.move_pair_radial(2, 0, 0.1, 0.1),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        w_toward.move_pair_radial(0, 1, f64::NAN, 0.1),
        Err(CoreError::NonFiniteValue { .. })
    ));
    let mut short = Wrapper2d(Layout2d::from_xy(vec![0.0, 0.5], vec![0.0, 0.0]).unwrap());
    assert!(matches!(
        short.move_pair_radial(0, 1, 1e308, 0.0),
        Err(CoreError::NonFiniteValue { .. })
    ));
}

#[test]
fn test_euclidean_move_pair_radial_parity_and_atomic_guards() {
    // Equivalence between move_pair_toward and move_pair_radial on 2D, 3D, and ND (f64 & f32).
    let mut l2_t = Layout2d::from_xy(vec![1.0_f64, 4.0], vec![-2.0, 2.0]).unwrap();
    let mut l2_r = l2_t.clone();
    let d2 = l2_t.distance(0, 1);
    l2_t.move_pair_toward(0, 1, 2.0, 0.3, 0.2).unwrap();
    l2_r.move_pair_radial(0, 1, 0.3 * (d2 - 2.0), 0.2 * (d2 - 2.0))
        .unwrap();
    assert!((l2_t.x()[0] - l2_r.x()[0]).abs() <= 1e-13);
    assert!((l2_t.y()[1] - l2_r.y()[1]).abs() <= 1e-13);

    let mut l3_t = Layout3d::from_xyz(vec![0.0_f32, 2.0], vec![1.0, 4.0], vec![-1.0, 5.0]).unwrap();
    let mut l3_r = l3_t.clone();
    let d3 = l3_t.distance(0, 1);
    l3_t.move_pair_toward(0, 1, 3.0, 0.4, 0.1).unwrap();
    l3_r.move_pair_radial(0, 1, 0.4 * (d3 - 3.0), 0.1 * (d3 - 3.0))
        .unwrap();
    assert!((l3_t.x()[0] - l3_r.x()[0]).abs() <= 1e-5);
    assert!((l3_t.z()[1] - l3_r.z()[1]).abs() <= 1e-5);

    let mut lnd_t =
        LayoutNd::from_flat(2, 4, &[0.0_f64, 1.0, -1.0, 2.0, 3.0, 5.0, -1.0, 2.0]).unwrap();
    let mut lnd_r = lnd_t.clone();
    let dnd = lnd_t.distance(0, 1);
    lnd_t.move_pair_toward(0, 1, 2.5, 0.25, 0.35).unwrap();
    lnd_r
        .move_pair_radial(0, 1, 0.25 * (dnd - 2.5), 0.35 * (dnd - 2.5))
        .unwrap();
    for axis in 0..4 {
        assert!((lnd_t.get(0, axis).unwrap() - lnd_r.get(0, axis).unwrap()).abs() <= 1e-13);
        assert!((lnd_t.get(1, axis).unwrap() - lnd_r.get(1, axis).unwrap()).abs() <= 1e-13);
    }

    // One-sided update preserves -0.0 bit-exact on stationary endpoint.
    let mut neg_zero = Layout2d::from_xy(vec![-0.0_f64, 3.0], vec![-0.0, 4.0]).unwrap();
    let x0_bits = neg_zero.x()[0].to_bits();
    let y0_bits = neg_zero.y()[0].to_bits();
    neg_zero.move_pair_radial(0, 1, 0.0, 1.0).unwrap();
    assert_eq!(neg_zero.x()[0].to_bits(), x0_bits);
    assert_eq!(neg_zero.y()[0].to_bits(), y0_bits);
    assert!((neg_zero.distance(0, 1) - 4.0).abs() < 1e-12);

    // Pre-i==j coordinate finiteness checks.
    let mut nan_2d = Layout2d::from_xy(vec![f64::NAN, 1.0], vec![0.0, 1.0]).unwrap();
    assert!(matches!(
        nan_2d.move_pair_radial(0, 0, 0.1, 0.1),
        Err(CoreError::NonFiniteValue { .. })
    ));
    let mut nan_3d =
        Layout3d::from_xyz(vec![0.0, 1.0], vec![f64::INFINITY, 1.0], vec![0.0, 1.0]).unwrap();
    assert!(matches!(
        nan_3d.move_pair_radial(0, 0, 0.1, 0.1),
        Err(CoreError::NonFiniteValue { .. })
    ));
    let mut nan_nd = LayoutNd::from_flat(2, 2, &[0.0, f64::NAN, 1.0, 1.0]).unwrap();
    assert!(matches!(
        nan_nd.move_pair_radial(0, 0, 0.1, 0.1),
        Err(CoreError::NonFiniteValue { .. })
    ));

    // Empty and OOB guards.
    let mut empty2 = Layout2d::<f64>::zeros(0);
    assert!(matches!(
        empty2.move_pair_radial(0, 0, 0.1, 0.1),
        Err(CoreError::IndexOutOfBounds { .. })
    ));

    // Atomic no-mutation when LayoutNd axis 1 overflows f32 (axis 0 must remain untouched).
    let mut nd32 = LayoutNd::from_flat(2, 2, &[1.0_f32, 0.0_f32, 2.0_f32, 3.0e38_f32]).unwrap();
    let nd32_snap = nd32.clone();
    assert!(matches!(
        nd32.move_pair_radial(0, 1, 0.0, -3.0e38),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert_eq!(nd32, nd32_snap);

    let mut l2_32 = Layout2d::from_xy(vec![0.0_f32, 1.0], vec![0.0, 3.0e38]).unwrap();
    let l2_32_snap = l2_32.clone();
    assert!(matches!(
        l2_32.move_pair_radial(0, 1, 0.0, -3.0e38),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert_eq!(l2_32, l2_32_snap);
}
