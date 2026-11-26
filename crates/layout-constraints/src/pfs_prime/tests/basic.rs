use egraph_core::{CoreError, IterativeSolver, Layout2d, SolverStatus};

use super::super::verify::has_orthogonal_order_inversion;
use super::super::{PfsPrimeConfig, PfsPrimeOverlapRemoval};
use super::{
    assert_no_overlaps, assert_no_overlaps_all, count_orthogonal_inversions,
    count_orthogonal_inversions_scalar,
};

#[test]
fn test_pfs_prime_validation_and_degenerate_inputs() {
    assert!(matches!(
        PfsPrimeConfig {
            padding: -0.1,
            ..Default::default()
        }
        .validate(),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        PfsPrimeConfig {
            padding: f64::NAN,
            ..Default::default()
        }
        .validate(),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        PfsPrimeOverlapRemoval::new(vec![1.0, 2.0], vec![1.0], PfsPrimeConfig::default()),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        PfsPrimeOverlapRemoval::new(vec![-1.0], vec![1.0], PfsPrimeConfig::default()),
        Err(CoreError::InvalidInput(_))
    ));

    // N = 0
    let mut s0 =
        PfsPrimeOverlapRemoval::<f64>::from_radii(vec![], PfsPrimeConfig::default()).unwrap();
    let mut l0 = Layout2d::zeros(0);
    assert_eq!(s0.step(&mut l0).unwrap(), SolverStatus::Converged);
    assert_eq!(s0.last_overlap_count(), 0);

    // N = 1
    let mut s1 =
        PfsPrimeOverlapRemoval::<f64>::from_radii(vec![2.0], PfsPrimeConfig::default()).unwrap();
    let mut l1 = Layout2d::from_xy(vec![3.0], vec![4.0]).unwrap();
    assert_eq!(s1.step(&mut l1).unwrap(), SolverStatus::Converged);
    assert_eq!(s1.last_overlap_count(), 0);
    assert_eq!(l1.x(), &[3.0]);
    assert_eq!(l1.y(), &[4.0]);
}

#[test]
fn test_pfs_prime_zero_overlap_and_oo_nni_zero_f64_f32() {
    // Non-coincident overlapping boxes.
    let orig_x = vec![0.0_f64, 0.4, 0.9, 1.3, 0.2, 0.7, 1.5, 2.0];
    let orig_y = vec![0.0_f64, 0.2, 0.5, 0.1, 0.8, 0.6, 0.9, 0.3];
    let hw = vec![0.8_f64; 8];
    let hh = vec![0.6_f64; 8];

    let mut solver =
        PfsPrimeOverlapRemoval::<f64>::new(hw.clone(), hh.clone(), PfsPrimeConfig::default())
            .unwrap();
    let mut layout = Layout2d::from_xy(orig_x.clone(), orig_y.clone()).unwrap();
    let status = solver.run(&mut layout).unwrap();
    assert_eq!(status, SolverStatus::Converged);
    assert_eq!(solver.last_overlap_count(), 0);
    assert_no_overlaps(&layout, &hw, &hh);
    assert_eq!(
        count_orthogonal_inversions(&orig_x, &orig_y, layout.x(), layout.y()),
        0
    );

    // With positive padding.
    let pad = 0.5_f64;
    let cfg_pad = PfsPrimeConfig {
        padding: pad,
        seed: 7,
    };
    let mut solver_pad =
        PfsPrimeOverlapRemoval::<f64>::new(hw.clone(), hh.clone(), cfg_pad).unwrap();
    let mut layout_pad = Layout2d::from_xy(orig_x.clone(), orig_y.clone()).unwrap();
    assert_eq!(
        solver_pad.run(&mut layout_pad).unwrap(),
        SolverStatus::Converged
    );
    assert_eq!(solver_pad.last_overlap_count(), 0);
    assert_no_overlaps_all(&layout_pad, &hw, &hh, pad);
    assert_eq!(
        count_orthogonal_inversions(&orig_x, &orig_y, layout_pad.x(), layout_pad.y()),
        0
    );

    // f32 precision at large coordinate offset.
    let hw32 = vec![1.0_f32; 6];
    let hh32 = vec![0.75_f32; 6];
    let mut s32_a =
        PfsPrimeOverlapRemoval::<f32>::new(hw32.clone(), hh32.clone(), PfsPrimeConfig::default())
            .unwrap();
    let mut s32_b =
        PfsPrimeOverlapRemoval::<f32>::new(hw32.clone(), hh32.clone(), PfsPrimeConfig::default())
            .unwrap();
    let orig_x32 = vec![100.0_f32, 100.2, 100.5, 100.8, 100.1, 100.6];
    let orig_y32 = vec![-50.0_f32, -50.1, -49.8, -50.2, -49.9, -50.0];
    let mut l32_a = Layout2d::from_xy(orig_x32.clone(), orig_y32.clone()).unwrap();
    let mut l32_b = l32_a.clone();
    assert_eq!(s32_a.step(&mut l32_a).unwrap(), SolverStatus::Converged);
    assert_eq!(s32_b.run(&mut l32_b).unwrap(), SolverStatus::Converged);
    assert_no_overlaps(&l32_a, &hw32, &hh32);
    assert_eq!(
        count_orthogonal_inversions_scalar(&orig_x32, &orig_y32, l32_a.x(), l32_a.y()),
        0
    );
    assert_eq!(
        l32_a.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        l32_b.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        l32_a.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        l32_b.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn test_pfs_prime_collinear_and_coincident() {
    // Collinear nodes along X and Y.
    let radii = vec![1.0_f64; 5];
    let mut s_col =
        PfsPrimeOverlapRemoval::<f64>::from_radii(radii.clone(), PfsPrimeConfig::default())
            .unwrap();
    let orig_x_col = vec![0.0, 0.2, 0.4, 0.6, 0.8];
    let orig_y_col = vec![0.0; 5];
    let mut l_col = Layout2d::from_xy(orig_x_col.clone(), orig_y_col.clone()).unwrap();
    assert_eq!(s_col.run(&mut l_col).unwrap(), SolverStatus::Converged);
    assert_eq!(s_col.last_overlap_count(), 0);
    assert_no_overlaps(&l_col, &radii, &radii);
    assert_eq!(
        count_orthogonal_inversions(&orig_x_col, &orig_y_col, l_col.x(), l_col.y()),
        0
    );

    // All coincident nodes at (0, 0).
    let mut s_coin =
        PfsPrimeOverlapRemoval::<f64>::from_radii(radii.clone(), PfsPrimeConfig::default())
            .unwrap();
    let mut l_coin = Layout2d::zeros(5);
    assert_eq!(s_coin.run(&mut l_coin).unwrap(), SolverStatus::Converged);
    assert_eq!(s_coin.last_overlap_count(), 0);
    assert_no_overlaps(&l_coin, &radii, &radii);
}

#[test]
fn test_pfs_prime_has_orthogonal_order_inversion_helper() {
    let orig_x = [0.0, 0.0, 1.0, 2.0];
    let orig_y = [-0.0, 0.0, 5.0, 10.0];
    let l_ok = Layout2d::from_xy(vec![1.0, 0.5, 1.0, 3.0], vec![0.0, 0.0, 5.0, 10.0]).unwrap();
    assert!(!has_orthogonal_order_inversion(&orig_x, &orig_y, &l_ok));

    let l_x_inv = Layout2d::from_xy(
        vec![1.0, 0.5, 1.0, 0.999999999999],
        vec![0.0, 0.0, 5.0, 10.0],
    )
    .unwrap();
    assert!(has_orthogonal_order_inversion(&orig_x, &orig_y, &l_x_inv));

    let l_y_inv = Layout2d::from_xy(vec![0.5, 1.0, 1.0, 3.0], vec![0.0, 0.0, 10.0, 9.999]).unwrap();
    assert!(has_orthogonal_order_inversion(&orig_x, &orig_y, &l_y_inv));

    // 1-ULP inversion at 50096.94 matching 3-node X-sweep ULP reproducer.
    let orig_3x = [10000.0 - 1e-6, 10000.0, 10000.0 + 8e-10];
    let orig_3y = [0.0, 0.0, -1e8];
    let l_ulp_inv = Layout2d::from_xy(
        vec![9999.999998999996, 50096.94009590009, 50096.94009590008],
        vec![0.0, 0.0, -1e8],
    )
    .unwrap();
    assert!(has_orthogonal_order_inversion(
        &orig_3x, &orig_3y, &l_ulp_inv
    ));
}
