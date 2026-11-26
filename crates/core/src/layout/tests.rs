use super::*;
use crate::error::CoreError;

#[test]
fn test_layout2d_interleaved_roundtrip_and_mutators() {
    let mut layout = Layout2d::from_interleaved(&[1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(layout.len(), 2);
    assert!(!layout.is_empty());
    assert_eq!(layout.x(), &[1.0, 3.0]);
    assert_eq!(layout.y(), &[2.0, 4.0]);

    layout.x_mut()[0] = 5.0;
    layout.y_mut()[1] = 6.0;
    let mut out = vec![0.0_f64; 4];
    layout.write_interleaved(&mut out).unwrap();
    assert_eq!(out, vec![5.0, 2.0, 3.0, 6.0]);

    assert!(matches!(
        Layout2d::<f64>::from_interleaved(&[1.0, 2.0, 3.0]),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        layout.write_interleaved(&mut [0.0; 2]),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        Layout2d::from_xy(vec![1.0], vec![1.0, 2.0]),
        Err(CoreError::DimensionMismatch { .. })
    ));
}

#[test]
fn test_layout2d_init_and_copy() {
    let l1 = Layout2d::<f64>::init(8, 42, 2.0).unwrap();
    let l2 = Layout2d::<f64>::init(8, 42, 2.0).unwrap();
    assert_eq!(l1, l2);
    for (&x, &y) in l1.x().iter().zip(l1.y().iter()) {
        assert!((-1.0..1.0).contains(&x));
        assert!((-1.0..1.0).contains(&y));
    }
    let mut cloned = l1.copy();
    assert_eq!(cloned, l1);
    cloned.x_mut()[0] += 1.0;
    assert_ne!(cloned, l1);

    let l32 = Layout2d::<f32>::init(4, 7, 1.0).unwrap();
    assert_eq!(l32.len(), 4);
    assert!(l32.is_all_finite());

    assert!(matches!(
        Layout2d::<f64>::init(4, 42, f64::NAN),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        Layout2d::<f64>::init(4, 42, 0.0),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        Layout2d::<f32>::init(4, 42, 1e40),
        Err(CoreError::NonFiniteValue { .. })
    ));
}

#[test]
fn test_layout2d_center_and_normalize_std() {
    let mut layout = Layout2d::from_xy(
        vec![10.0_f64, 12.0, 14.0, 10.0],
        vec![-5.0_f64, -3.0, -5.0, -7.0],
    )
    .unwrap();
    layout.center_and_normalize_std().unwrap();
    let n = layout.len() as f64;
    let mean_x = layout.x().iter().sum::<f64>() / n;
    let mean_y = layout.y().iter().sum::<f64>() / n;
    assert!(mean_x.abs() < 1e-12);
    assert!(mean_y.abs() < 1e-12);
    let var: f64 = layout
        .x()
        .iter()
        .zip(layout.y().iter())
        .map(|(&x, &y)| x * x + y * y)
        .sum::<f64>()
        / (2.0 * n);
    assert!((var.sqrt() - 1.0).abs() < 1e-12);

    // Collapsed / 1-node / empty / non-finite edge cases.
    let mut collapsed = Layout2d::from_xy(vec![3.0, 3.0], vec![7.0, 7.0]).unwrap();
    collapsed.center_and_normalize_std().unwrap();
    assert_eq!(collapsed.x(), &[0.0, 0.0]);
    assert_eq!(collapsed.y(), &[0.0, 0.0]);

    let mut single = Layout2d::from_xy(vec![9.0], vec![-4.0]).unwrap();
    single.center_and_normalize_std().unwrap();
    assert_eq!(single.x(), &[0.0]);
    assert_eq!(single.y(), &[0.0]);

    let mut empty = Layout2d::<f32>::zeros(0);
    assert!(empty.is_empty());
    empty.center_and_normalize_std().unwrap();

    let mut bad = Layout2d::from_xy(vec![0.0, f64::NAN], vec![0.0, 1.0]).unwrap();
    assert!(!bad.is_all_finite());
    assert!(matches!(
        bad.center_and_normalize_std(),
        Err(CoreError::NonFiniteValue { .. })
    ));
}

#[test]
fn test_layout_nd_parity_methods() {
    let mut nd = LayoutNd::<f32>::zeros(2, 3).unwrap();
    assert_eq!(nd.dims(), 3);
    assert_eq!(nd.num_nodes(), 2);
    assert_eq!(nd.len(), 2);
    assert!(nd.is_all_finite());
    nd.validate_finite().unwrap();

    nd.node_mut(1).unwrap().copy_from_slice(&[1.0, 2.0, 3.0]);
    assert_eq!(nd.node(1).unwrap(), &[1.0, 2.0, 3.0]);
    assert_eq!(nd.get(1, 2).unwrap(), 3.0);
    nd.set(0, 1, -4.5).unwrap();
    assert_eq!(nd.get(0, 1).unwrap(), -4.5);

    assert!(matches!(
        LayoutNd::<f64>::zeros(2, 0),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        nd.node(2),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        nd.get(2, 0),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        nd.get(0, 3),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        nd.set(2, 0, 1.0),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        nd.set(0, 3, 1.0),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        nd.set(0, 0, f32::NAN),
        Err(CoreError::NonFiniteValue { .. })
    ));

    // init + copy + 2D parity with Layout2d.
    let mut nd2 = LayoutNd::<f64>::init(6, 2, 42, 1.0).unwrap();
    let mut l2 = Layout2d::<f64>::init(6, 42, 1.0).unwrap();
    assert_eq!(nd2.coords(), l2.to_interleaved().as_slice());

    let mut nd2_copy = nd2.copy();
    assert_eq!(nd2_copy, nd2);
    nd2_copy.set(0, 0, 99.0).unwrap();
    assert_ne!(nd2_copy, nd2);

    nd2.center_and_normalize_to_std(2.5).unwrap();
    l2.center_and_normalize_to_std(2.5).unwrap();
    for i in 0..6 {
        assert!((nd2.get(i, 0).unwrap() - l2.x()[i]).abs() < 1e-12);
        assert!((nd2.get(i, 1).unwrap() - l2.y()[i]).abs() < 1e-12);
    }

    // Collapsed / singleton / empty / non-finite LayoutNd normalization.
    let mut nd_collapsed =
        LayoutNd::<f64>::from_flat(2, 3, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap();
    nd_collapsed.center_and_normalize_std().unwrap();
    assert_eq!(nd_collapsed.coords(), &[0.0; 6]);

    let mut nd_single = LayoutNd::<f64>::from_flat(1, 3, vec![5.0, -2.0, 9.0]).unwrap();
    nd_single.center_and_normalize_std().unwrap();
    assert_eq!(nd_single.coords(), &[0.0; 3]);

    let mut nd_empty = LayoutNd::<f64>::zeros(0, 3).unwrap();
    nd_empty.center_and_normalize_std().unwrap();

    let mut nd_bad = LayoutNd::<f64>::from_flat(1, 2, vec![f64::INFINITY, 1.0]).unwrap();
    assert!(matches!(
        nd_bad.validate_finite(),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        nd_bad.center_and_normalize_std(),
        Err(CoreError::NonFiniteValue { .. })
    ));
}
