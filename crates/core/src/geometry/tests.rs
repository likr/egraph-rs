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
