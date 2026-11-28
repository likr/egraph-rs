use super::*;
use crate::error::CoreError;

#[test]
fn test_layout2d_interleaved_roundtrip_and_mutators() {
    let mut layout = Layout2d::from_interleaved(&[1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(layout.len(), 2);
    assert_eq!(layout.num_nodes(), 2);
    assert!(!layout.is_empty());
    assert_eq!(layout.x(), &[1.0, 3.0]);
    assert_eq!(layout.y(), &[2.0, 4.0]);

    layout.x_mut()[0] = 5.0;
    layout.y_mut()[1] = 6.0;
    let mut out = vec![0.0_f64; 4];
    layout.write_interleaved(&mut out).unwrap();
    assert_eq!(out, vec![5.0, 2.0, 3.0, 6.0]);

    assert_eq!(layout.get(1).unwrap(), [3.0, 6.0]);
    layout.set(0, 7.0, 8.0).unwrap();
    assert_eq!(layout.get(0).unwrap(), [7.0, 8.0]);
    assert!(matches!(
        layout.get(2),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        layout.set(0, f64::NAN, 0.0),
        Err(CoreError::NonFiniteValue { .. })
    ));

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
fn test_layout2d_init_and_clone() {
    let l1 = Layout2d::<f64>::init(8, 42, 2.0).unwrap();
    let l2 = Layout2d::<f64>::init(8, 42, 2.0).unwrap();
    assert_eq!(l1, l2);
    for (&x, &y) in l1.x().iter().zip(l1.y().iter()) {
        assert!((-1.0..1.0).contains(&x));
        assert!((-1.0..1.0).contains(&y));
    }
    let mut cloned = l1.clone();
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
fn test_layout3d_api_and_normalization() {
    let mut l3 = Layout3d::from_interleaved(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert_eq!(l3.len(), 2);
    assert_eq!(l3.num_nodes(), 2);
    assert!(!l3.is_empty());
    assert_eq!(l3.x(), &[1.0, 4.0]);
    assert_eq!(l3.y(), &[2.0, 5.0]);
    assert_eq!(l3.z(), &[3.0, 6.0]);
    assert_eq!(l3.get(1).unwrap(), [4.0, 5.0, 6.0]);

    l3.set(0, -1.0, 0.0, 1.0).unwrap();
    assert_eq!(l3.get(0).unwrap(), [-1.0, 0.0, 1.0]);
    let (xm, ym, zm) = l3.xyz_mut();
    xm[0] = 2.0;
    ym[0] = 3.0;
    zm[0] = 4.0;
    assert_eq!(l3.to_interleaved(), vec![2.0, 3.0, 4.0, 4.0, 5.0, 6.0]);

    let mut buf = vec![0.0_f64; 6];
    l3.write_interleaved(&mut buf).unwrap();
    assert_eq!(buf, vec![2.0, 3.0, 4.0, 4.0, 5.0, 6.0]);

    assert!(matches!(
        Layout3d::<f64>::from_interleaved(&[1.0, 2.0]),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        l3.set(0, f64::INFINITY, 0.0, 0.0),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(l3.get(2), Err(CoreError::IndexOutOfBounds { .. })));

    let mut rand3 = Layout3d::<f64>::init(10, 42, 3.0).unwrap();
    assert_eq!(rand3, Layout3d::<f64>::init(10, 42, 3.0).unwrap());
    rand3.center_and_normalize_to_std(2.5).unwrap();
    let n = rand3.len() as f64;
    let mx = rand3.x().iter().sum::<f64>() / n;
    let my = rand3.y().iter().sum::<f64>() / n;
    let mz = rand3.z().iter().sum::<f64>() / n;
    assert!(mx.abs() < 1e-12 && my.abs() < 1e-12 && mz.abs() < 1e-12);
    let sq: f64 = rand3
        .x()
        .iter()
        .zip(rand3.y().iter())
        .zip(rand3.z().iter())
        .map(|((&x, &y), &z)| x * x + y * y + z * z)
        .sum();
    assert!(((sq / (3.0 * n)).sqrt() - 2.5).abs() < 1e-12);

    // Bit-exact 3D parity with LayoutNd(n, 3).
    let mut nd3 = LayoutNd::<f64>::init(10, 3, 42, 3.0).unwrap();
    nd3.center_and_normalize_to_std(2.5).unwrap();
    let l3_from_init = Layout3d::<f64>::init(10, 42, 3.0).unwrap();
    let mut l3_cmp = l3_from_init;
    l3_cmp.center_and_normalize_to_std(2.5).unwrap();
    assert_eq!(l3_cmp.to_interleaved(), nd3.coords());
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

    let mut nd2 = LayoutNd::<f64>::init(6, 2, 42, 1.0).unwrap();
    let mut l2 = Layout2d::<f64>::init(6, 42, 1.0).unwrap();
    assert_eq!(nd2.coords(), l2.to_interleaved().as_slice());

    let mut nd2_copy = nd2.clone();
    assert_eq!(nd2_copy, nd2);
    nd2_copy.set(0, 0, 99.0).unwrap();
    assert_ne!(nd2_copy, nd2);

    nd2.center_and_normalize_to_std(2.5).unwrap();
    l2.center_and_normalize_to_std(2.5).unwrap();
    for i in 0..6 {
        assert!((nd2.get(i, 0).unwrap() - l2.x()[i]).abs() < 1e-12);
        assert!((nd2.get(i, 1).unwrap() - l2.y()[i]).abs() < 1e-12);
    }

    let mut nd_collapsed =
        LayoutNd::<f64>::from_flat(2, 3, &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap();
    nd_collapsed.center_and_normalize_std().unwrap();
    assert_eq!(nd_collapsed.coords(), &[0.0; 6]);

    let mut nd_single = LayoutNd::<f64>::from_flat_vec(1, 3, vec![5.0, -2.0, 9.0]).unwrap();
    nd_single.center_and_normalize_std().unwrap();
    assert_eq!(nd_single.coords(), &[0.0; 3]);

    let mut nd_empty = LayoutNd::<f64>::zeros(0, 3).unwrap();
    nd_empty.center_and_normalize_std().unwrap();

    let mut nd_bad = LayoutNd::<f64>::from_flat(1, 2, &[f64::INFINITY, 1.0]).unwrap();
    assert!(matches!(
        nd_bad.validate_finite(),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        nd_bad.center_and_normalize_std(),
        Err(CoreError::NonFiniteValue { .. })
    ));
}

#[test]
fn test_layout_nd_2d_bit_exact_parity_f64_and_f32() {
    for &n in &[3usize, 5, 7, 11, 13] {
        let mut l2_64 = Layout2d::<f64>::init(n, 100 + n as u64, 7.3).unwrap();
        for i in 0..n {
            l2_64.x_mut()[i] += (i as f64) * 1.37 + 0.19;
            l2_64.y_mut()[i] -= (i as f64) * 0.83 - 2.41;
        }
        let mut nd_64 = LayoutNd::<f64>::from_flat_vec(n, 2, l2_64.to_interleaved()).unwrap();

        l2_64.center_and_normalize_std().unwrap();
        nd_64.center_and_normalize_std().unwrap();
        for i in 0..n {
            assert_eq!(
                nd_64.get(i, 0).unwrap().to_bits(),
                l2_64.x()[i].to_bits(),
                "f64 x bit-exact mismatch at n={n}, i={i}"
            );
            assert_eq!(
                nd_64.get(i, 1).unwrap().to_bits(),
                l2_64.y()[i].to_bits(),
                "f64 y bit-exact mismatch at n={n}, i={i}"
            );
        }

        let mut l2_32 = Layout2d::<f32>::init(n, 200 + n as u64, 5.1).unwrap();
        for i in 0..n {
            l2_32.x_mut()[i] += (i as f32) * 1.13 + 0.31;
            l2_32.y_mut()[i] -= (i as f32) * 0.71 - 1.79;
        }
        let mut nd_32 = LayoutNd::<f32>::from_flat_vec(n, 2, l2_32.to_interleaved()).unwrap();

        l2_32.center_and_normalize_to_std(2.75).unwrap();
        nd_32.center_and_normalize_to_std(2.75).unwrap();
        for i in 0..n {
            assert_eq!(
                nd_32.get(i, 0).unwrap().to_bits(),
                l2_32.x()[i].to_bits(),
                "f32 x bit-exact mismatch at n={n}, i={i}"
            );
            assert_eq!(
                nd_32.get(i, 1).unwrap().to_bits(),
                l2_32.y()[i].to_bits(),
                "f32 y bit-exact mismatch at n={n}, i={i}"
            );
        }
    }
}

#[test]
fn test_layout_center_and_normalize_to_std_f32_overflow_no_mutation() {
    let mut l2 = Layout2d::<f32>::from_xy(vec![1.0, -3.0], vec![0.0, 0.0]).unwrap();
    let orig_l2 = l2.clone();
    assert!(matches!(
        l2.center_and_normalize_to_std(3.0e38),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert_eq!(l2, orig_l2);
    assert!(l2.is_all_finite());

    let mut l3 =
        Layout3d::<f32>::from_xyz(vec![1.0, -3.0], vec![0.0, 0.0], vec![0.0, 0.0]).unwrap();
    let orig_l3 = l3.clone();
    assert!(matches!(
        l3.center_and_normalize_to_std(3.0e38),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert_eq!(l3, orig_l3);
    assert!(l3.is_all_finite());

    let mut nd =
        LayoutNd::<f32>::from_flat(2, 4, &[1.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0]).unwrap();
    let orig_nd = nd.clone();
    assert!(matches!(
        nd.center_and_normalize_to_std(3.0e38),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert_eq!(nd, orig_nd);
    assert!(nd.is_all_finite());
}

#[test]
fn test_layout_permute_unpermute_2d_3d_nd_and_errors() {
    let perm = [2u32, 0, 3, 1];

    // Layout2d (f64 & f32)
    let l2 = Layout2d::from_xy(vec![10.0_f64, 20.0, 30.0, 40.0], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let l2_p = l2.permute(&perm).unwrap();
    for (old_i, &new_i) in perm.iter().enumerate() {
        assert_eq!(l2_p.get(new_i as usize).unwrap(), l2.get(old_i).unwrap());
    }
    assert_eq!(l2_p.unpermute(&perm).unwrap(), l2);
    assert_eq!(l2.unpermute(&perm).unwrap().permute(&perm).unwrap(), l2);

    let l2_32 = Layout2d::from_xy(vec![1.0_f32, 2.0], vec![3.0, 4.0]).unwrap();
    assert_eq!(
        l2_32.permute(&[1, 0]).unwrap().unpermute(&[1, 0]).unwrap(),
        l2_32
    );

    // Layout3d (f64 & f32)
    let l3 = Layout3d::from_xyz(
        vec![1.0_f64, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0, 8.0],
        vec![9.0, 10.0, 11.0, 12.0],
    )
    .unwrap();
    let l3_p = l3.permute(&perm).unwrap();
    for (old_i, &new_i) in perm.iter().enumerate() {
        assert_eq!(l3_p.get(new_i as usize).unwrap(), l3.get(old_i).unwrap());
    }
    assert_eq!(l3_p.unpermute(&perm).unwrap(), l3);
    assert_eq!(l3.unpermute(&perm).unwrap().permute(&perm).unwrap(), l3);

    let l3_32 = Layout3d::from_xyz(vec![1.0_f32], vec![2.0], vec![3.0]).unwrap();
    assert_eq!(l3_32.permute(&[0]).unwrap(), l3_32);
    assert_eq!(l3_32.unpermute(&[0]).unwrap(), l3_32);

    // LayoutNd (f64 & f32) preserving dims (including n == 0).
    let lnd = LayoutNd::from_flat(4, 5, &(0..20).map(|v| v as f64).collect::<Vec<f64>>()).unwrap();
    let lnd_p = lnd.permute(&perm).unwrap();
    assert_eq!(lnd_p.dims(), 5);
    for (old_i, &new_i) in perm.iter().enumerate() {
        assert_eq!(
            lnd_p.node(new_i as usize).unwrap(),
            lnd.node(old_i).unwrap()
        );
    }
    assert_eq!(lnd_p.unpermute(&perm).unwrap(), lnd);
    assert_eq!(lnd.unpermute(&perm).unwrap().permute(&perm).unwrap(), lnd);

    let lnd_empty = LayoutNd::<f32>::zeros(0, 5).unwrap();
    let lnd_empty_p = lnd_empty.permute(&[]).unwrap();
    assert_eq!(lnd_empty_p.dims(), 5);
    assert_eq!(lnd_empty_p, lnd_empty);
    assert_eq!(lnd_empty.unpermute(&[]).unwrap(), lnd_empty);

    // Error contracts across all 3 layout types (both permute & unpermute).
    let empty2 = Layout2d::<f64>::zeros(0);
    let empty3 = Layout3d::<f64>::zeros(0);
    assert!(matches!(
        empty2.permute(&[0]),
        Err(CoreError::DimensionMismatch {
            expected: 0,
            actual: 1,
            ..
        })
    ));
    assert!(matches!(
        empty3.unpermute(&[0]),
        Err(CoreError::DimensionMismatch {
            expected: 0,
            actual: 1,
            ..
        })
    ));
    assert!(matches!(
        lnd_empty.permute(&[0]),
        Err(CoreError::DimensionMismatch {
            expected: 0,
            actual: 1,
            ..
        })
    ));

    for bad_len in [&[][..], &[0, 1, 2][..]] {
        assert!(matches!(
            l2.permute(bad_len),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            l2.unpermute(bad_len),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            l3.permute(bad_len),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            lnd.unpermute(bad_len),
            Err(CoreError::DimensionMismatch { .. })
        ));
    }

    for bad_perm in [&[0, 0, 2, 3][..], &[0, 1, 2, 4][..], &[0, 1, 2, 99][..]] {
        assert!(matches!(
            l2.permute(bad_perm),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            l2.unpermute(bad_perm),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            l3.permute(bad_perm),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            l3.unpermute(bad_perm),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            lnd.permute(bad_perm),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            lnd.unpermute(bad_perm),
            Err(CoreError::InvalidInput(_))
        ));
    }
}
