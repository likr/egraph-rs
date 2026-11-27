use egraph_core::{CoreError, CsrGraph, Layout2d, Layout3d, LayoutGenerator, LayoutNd};
use egraph_linalg::select_maxmin_sp_pivots;

use super::tests_hde_oracle::{hde_naive_oracle, isotropic_std_nd, pairwise_euclidean_nd};
use super::tests_mds::isotropic_std;
use super::*;

#[test]
fn test_hde_nd_and_large_u32_branch() {
    let edges = vec![
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 3),
        (1, 6),
        (2, 7),
        (4, 8),
    ];
    let g = CsrGraph::<f64>::from_edges(9, &edges, None, false, false).unwrap();

    // 1. output_dims == 2 matches Layout2d bit-for-bit
    for normalize in [false, true] {
        let cfg2 = HdeConfig::default()
            .with_dimensions(5)
            .with_output_dims(Some(2))
            .with_seed(7)
            .with_tol(1e-12)
            .with_max_sweeps(200)
            .with_normalize(normalize);
        let lay2: Layout2d<f64> = Hde::from_config(&g, cfg2.clone()).generate().unwrap();
        let lay_nd2: LayoutNd<f64> = Hde::from_config(&g, cfg2).generate().unwrap();
        assert_eq!(lay_nd2.dims(), 2);
        for i in 0..9 {
            let row = lay_nd2.node(i).unwrap();
            assert_eq!(row[0].to_bits(), lay2.x()[i].to_bits());
            assert_eq!(row[1].to_bits(), lay2.y()[i].to_bits());
        }
    }

    // 2. output_dims == 3 matches dense 3D oracle and Layout3d bit-for-bit
    let cfg3_unnorm = HdeConfig::default()
        .with_dimensions(5)
        .with_output_dims(Some(3))
        .with_seed(7)
        .with_tol(1e-12)
        .with_max_sweeps(200)
        .with_normalize(false);
    let pivots = select_maxmin_sp_pivots(&g, 5, 7, 1).unwrap();
    let lay3: LayoutNd<f64> = Hde::from_config(&g, cfg3_unnorm.clone())
        .generate()
        .unwrap();
    let lay3_soa: Layout3d<f64> = Hde::from_config(&g, cfg3_unnorm.clone())
        .generate()
        .unwrap();
    for i in 0..9 {
        let row = lay3.node(i).unwrap();
        assert_eq!(row[0].to_bits(), lay3_soa.x()[i].to_bits());
        assert_eq!(row[1].to_bits(), lay3_soa.y()[i].to_bits());
        assert_eq!(row[2].to_bits(), lay3_soa.z()[i].to_bits());
    }
    let oracle3 = hde_naive_oracle(&g, &pivots, 3);
    let lay3_coords: Vec<Vec<f64>> = (0..9).map(|i| lay3.node(i).unwrap().to_vec()).collect();
    for (a, b) in pairwise_euclidean_nd(&lay3_coords)
        .iter()
        .zip(pairwise_euclidean_nd(&oracle3).iter())
    {
        assert!((a - b).abs() < 1e-10);
    }
    let lay3_norm: LayoutNd<f64> = Hde::from_config(&g, cfg3_unnorm.with_normalize(true))
        .generate()
        .unwrap();
    assert!((isotropic_std_nd(&lay3_norm) - 1.0).abs() < 1e-12);

    // 3. m_eff < output_dims zero-pads remaining dimensions when normalize=false
    let lay_pad: LayoutNd<f64> = Hde::new(&g)
        .with_pivots(&[0, 4])
        .with_output_dims(4)
        .with_normalize(false)
        .generate()
        .unwrap();
    for i in 0..9 {
        let row = lay_pad.node(i).unwrap();
        assert_eq!(row[2], 0.0);
        assert_eq!(row[3], 0.0);
    }

    // 4. n > u16::MAX exercises the u32 BFS storage path cleanly
    let n_big = (u16::MAX as usize) + 5;
    let big_edges: Vec<(u32, u32)> = (0..(n_big as u32 - 1)).map(|i| (i, i + 1)).collect();
    let g_big = CsrGraph::<f64>::from_edges(n_big, &big_edges, None, false, false).unwrap();
    let big_lay: Layout2d<f64> = Hde::new(&g_big)
        .with_pivots(&[0, (n_big - 1) as u32])
        .generate()
        .unwrap();
    assert_eq!(big_lay.len(), n_big);
    assert!(big_lay.is_all_finite());
    assert!(big_lay.y().iter().all(|&y| y == 0.0));
    assert!((isotropic_std(&big_lay) - 1.0).abs() < 1e-10);
}

fn max_rel_pairwise_err(a: &[f64], b: &[f64], b_scale: f64) -> f64 {
    let mut max_err = 0.0_f64;
    for (&da, &db) in a.iter().zip(b.iter()) {
        let expected = db * b_scale;
        let denom = expected.abs().max(1e-300);
        let rel = (da - expected).abs() / denom;
        if rel > max_err {
            max_err = rel;
        }
    }
    max_err
}

#[test]
fn test_hde_scale_invariance_and_extreme_weights() {
    let edges = vec![
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 0),
        (0, 3),
        (1, 4),
    ];
    let base_w = vec![1.0_f64, 1.5, 2.2, 0.8, 3.1, 1.7, 2.6, 1.3];
    let pivots = vec![0u32, 2, 4, 1];

    let base_g = CsrGraph::<f64>::from_edges(6, &edges, Some(&base_w), false, false).unwrap();
    let cfg2_norm = HdeConfig::default()
        .with_dimensions(4)
        .with_output_dims(Some(2))
        .with_seed(42)
        .with_tol(1e-12)
        .with_max_sweeps(200)
        .with_normalize(true);
    let cfg2_unnorm = cfg2_norm.clone().with_normalize(false);
    let cfg3_norm = cfg2_norm.clone().with_output_dims(Some(3));
    let cfg3_unnorm = cfg3_norm.clone().with_normalize(false);

    let ref2_norm: Layout2d<f64> = Hde::from_config(&base_g, cfg2_norm.clone())
        .with_pivots(&pivots)
        .generate()
        .unwrap();
    let ref2_unnorm: Layout2d<f64> = Hde::from_config(&base_g, cfg2_unnorm.clone())
        .with_pivots(&pivots)
        .generate()
        .unwrap();
    let ref3_norm: LayoutNd<f64> = Hde::from_config(&base_g, cfg3_norm.clone())
        .with_pivots(&pivots)
        .generate()
        .unwrap();
    let ref3_unnorm: LayoutNd<f64> = Hde::from_config(&base_g, cfg3_unnorm.clone())
        .with_pivots(&pivots)
        .generate()
        .unwrap();

    let ref2_norm_pw = pairwise_euclidean_nd(
        &(0..6)
            .map(|i| vec![ref2_norm.x()[i], ref2_norm.y()[i]])
            .collect::<Vec<_>>(),
    );
    let ref2_unnorm_pw = pairwise_euclidean_nd(
        &(0..6)
            .map(|i| vec![ref2_unnorm.x()[i], ref2_unnorm.y()[i]])
            .collect::<Vec<_>>(),
    );
    let ref3_norm_pw = pairwise_euclidean_nd(
        &(0..6)
            .map(|i| ref3_norm.node(i).unwrap().to_vec())
            .collect::<Vec<_>>(),
    );
    let ref3_unnorm_pw = pairwise_euclidean_nd(
        &(0..6)
            .map(|i| ref3_unnorm.node(i).unwrap().to_vec())
            .collect::<Vec<_>>(),
    );

    let scales = [
        1e-200, 1e-150, 1e-100, 1e-20, 1e-13, 1.0, 1e80, 1e150, 1e200,
    ];
    for &scale in &scales {
        let scaled_w: Vec<f64> = base_w.iter().map(|&w| w * scale).collect();
        let g = CsrGraph::<f64>::from_edges(6, &edges, Some(&scaled_w), false, false).unwrap();

        // 2D normalized: scale-invariant vs unit scale
        let l2n: Layout2d<f64> = Hde::from_config(&g, cfg2_norm.clone())
            .with_pivots(&pivots)
            .generate()
            .unwrap();
        assert!(l2n.is_all_finite());
        assert!((isotropic_std(&l2n) - 1.0).abs() < 1e-12);
        let l2n_pw = pairwise_euclidean_nd(
            &(0..6)
                .map(|i| vec![l2n.x()[i], l2n.y()[i]])
                .collect::<Vec<_>>(),
        );
        let err_2n = max_rel_pairwise_err(&l2n_pw, &ref2_norm_pw, 1.0);
        assert!(
            err_2n < 1e-10,
            "2D normalized rel err at scale {scale:e}: {err_2n}"
        );

        // 2D unnormalized: scales linearly by `scale`
        let l2u: Layout2d<f64> = Hde::from_config(&g, cfg2_unnorm.clone())
            .with_pivots(&pivots)
            .generate()
            .unwrap();
        assert!(l2u.is_all_finite());
        let l2u_pw = pairwise_euclidean_nd(
            &(0..6)
                .map(|i| vec![l2u.x()[i], l2u.y()[i]])
                .collect::<Vec<_>>(),
        );
        let err_2u = max_rel_pairwise_err(&l2u_pw, &ref2_unnorm_pw, scale);
        assert!(
            err_2u < 1e-10,
            "2D unnormalized rel err at scale {scale:e}: {err_2u}"
        );

        // 3D normalized: scale-invariant vs unit scale
        let l3n: LayoutNd<f64> = Hde::from_config(&g, cfg3_norm.clone())
            .with_pivots(&pivots)
            .generate()
            .unwrap();
        assert!((isotropic_std_nd(&l3n) - 1.0).abs() < 1e-12);
        let l3n_pw = pairwise_euclidean_nd(
            &(0..6)
                .map(|i| l3n.node(i).unwrap().to_vec())
                .collect::<Vec<_>>(),
        );
        let err_3n = max_rel_pairwise_err(&l3n_pw, &ref3_norm_pw, 1.0);
        assert!(
            err_3n < 1e-10,
            "3D normalized rel err at scale {scale:e}: {err_3n}"
        );

        // 3D unnormalized: scales linearly by `scale`
        let l3u: LayoutNd<f64> = Hde::from_config(&g, cfg3_unnorm.clone())
            .with_pivots(&pivots)
            .generate()
            .unwrap();
        assert!(l3u.coords().iter().all(|v| v.is_finite()));
        let l3u_pw = pairwise_euclidean_nd(
            &(0..6)
                .map(|i| l3u.node(i).unwrap().to_vec())
                .collect::<Vec<_>>(),
        );
        let err_3u = max_rel_pairwise_err(&l3u_pw, &ref3_unnorm_pw, scale);
        assert!(
            err_3u < 1e-10,
            "3D unnormalized rel err at scale {scale:e}: {err_3u}"
        );
    }

    // Subnormal positive weights (e.g., 1e-315): two-step covariance scaling handles subnormals
    let subnormal_w: Vec<f64> = base_w.iter().map(|&w| w * 1e-315).collect();
    let g_sub = CsrGraph::<f64>::from_edges(6, &edges, Some(&subnormal_w), false, false).unwrap();
    let l2_sub: Layout2d<f64> = Hde::from_config(&g_sub, cfg2_norm.clone())
        .with_pivots(&pivots)
        .generate()
        .unwrap();
    assert!(l2_sub.is_all_finite());
    assert!((isotropic_std(&l2_sub) - 1.0).abs() < 1e-12);
    let l2_sub_pw = pairwise_euclidean_nd(
        &(0..6)
            .map(|i| vec![l2_sub.x()[i], l2_sub.y()[i]])
            .collect::<Vec<_>>(),
    );
    let err_sub = max_rel_pairwise_err(&l2_sub_pw, &ref2_norm_pw, 1.0);
    assert!(
        err_sub < 1e-5,
        "2D subnormal normalized rel err too large: {err_sub}"
    );

    // Disconnected N=4 with huge weights = 0.5 * f64::MAX:
    // normalize=true must match normalize=false rescaled to unit isotropic std (not Gaussian fallback).
    let disc_edges = vec![(0, 1), (2, 3)];
    let huge = 0.5 * f64::MAX;
    let disc_huge =
        CsrGraph::<f64>::from_edges(4, &disc_edges, Some(&[huge, huge]), false, false).unwrap();
    let disc_pivots = vec![0u32, 1, 2];

    let d2_unnorm: Layout2d<f64> = Hde::from_config(&disc_huge, cfg2_unnorm.clone())
        .with_pivots(&disc_pivots)
        .generate()
        .unwrap();
    assert!(d2_unnorm.is_all_finite());
    let mut d2_ref = Layout2d::from_xy(
        d2_unnorm.x().iter().map(|&v| v / huge).collect(),
        d2_unnorm.y().iter().map(|&v| v / huge).collect(),
    )
    .unwrap();
    normalize_layout_2d(&mut d2_ref, 1.0, cfg2_norm.seed).unwrap();
    let d2_huge: Layout2d<f64> = Hde::from_config(&disc_huge, cfg2_norm.clone())
        .with_pivots(&disc_pivots)
        .generate()
        .unwrap();
    assert!(d2_huge.is_all_finite());
    assert!((isotropic_std(&d2_huge) - 1.0).abs() < 1e-12);
    let d2_ref_pw = pairwise_euclidean_nd(
        &(0..4)
            .map(|i| vec![d2_ref.x()[i], d2_ref.y()[i]])
            .collect::<Vec<_>>(),
    );
    let d2_huge_pw = pairwise_euclidean_nd(
        &(0..4)
            .map(|i| vec![d2_huge.x()[i], d2_huge.y()[i]])
            .collect::<Vec<_>>(),
    );
    assert!(max_rel_pairwise_err(&d2_huge_pw, &d2_ref_pw, 1.0) < 1e-10);

    let d3_unnorm: LayoutNd<f64> = Hde::from_config(&disc_huge, cfg3_unnorm.clone())
        .with_pivots(&disc_pivots)
        .generate()
        .unwrap();
    assert!(d3_unnorm.coords().iter().all(|v| v.is_finite()));
    let mut d3_ref = LayoutNd::<f64>::zeros(4, 3).unwrap();
    for (dst, &src) in d3_ref
        .coords_mut()
        .iter_mut()
        .zip(d3_unnorm.coords().iter())
    {
        *dst = src / huge;
    }
    normalize_euclidean_layout(&mut d3_ref, 1.0, cfg3_norm.seed).unwrap();
    let d3_huge: LayoutNd<f64> = Hde::from_config(&disc_huge, cfg3_norm)
        .with_pivots(&disc_pivots)
        .generate()
        .unwrap();
    assert!((isotropic_std_nd(&d3_huge) - 1.0).abs() < 1e-12);
    let d3_ref_pw = pairwise_euclidean_nd(
        &(0..4)
            .map(|i| d3_ref.node(i).unwrap().to_vec())
            .collect::<Vec<_>>(),
    );
    let d3_huge_pw = pairwise_euclidean_nd(
        &(0..4)
            .map(|i| d3_huge.node(i).unwrap().to_vec())
            .collect::<Vec<_>>(),
    );
    assert!(max_rel_pairwise_err(&d3_huge_pw, &d3_ref_pw, 1.0) < 1e-10);

    // Unnormalized projection overflow near f64::MAX returns NumericalBreakdown instead of Inf/NaN
    let ovf_edges = vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (0, 4)];
    let ovf_w = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.95 * f64::MAX];
    let ovf_g = CsrGraph::<f64>::from_edges(5, &ovf_edges, Some(&ovf_w), false, false).unwrap();
    let ovf_pivots = vec![0u32, 1, 2, 3];
    assert!(matches!(
        Hde::<Layout2d<f64>, f64>::from_config(&ovf_g, cfg2_unnorm)
            .with_pivots(&ovf_pivots)
            .generate(),
        Err(CoreError::NumericalBreakdown { .. })
    ));
    let ovf_norm: Layout2d<f64> = Hde::from_config(&ovf_g, cfg2_norm)
        .with_pivots(&ovf_pivots)
        .generate()
        .unwrap();
    assert!(ovf_norm.is_all_finite());
    assert!((isotropic_std(&ovf_norm) - 1.0).abs() < 1e-12);
}

#[test]
fn test_hde_f32_output_overflow_returns_numerical_breakdown() {
    let edges = vec![(0, 1), (1, 2), (2, 3)];
    let pivots = [0u32, 3];

    // 1. Valid positive finite f32 weights whose unnormalized projection exceeds f32::MAX
    let g_f32 = CsrGraph::<f32>::from_edges(4, &edges, Some(&[3e38_f32; 3]), false, false).unwrap();

    let cfg2_unnorm = HdeConfig::default().with_normalize(false);
    let cfg2_norm = HdeConfig::default();
    let cfg3_unnorm = HdeConfig::default()
        .with_output_dims(Some(3))
        .with_normalize(false);
    let cfg3_norm = HdeConfig::default()
        .with_output_dims(Some(3))
        .with_normalize(true);

    assert!(matches!(
        Hde::<Layout2d<f32>, f32>::from_config(&g_f32, cfg2_unnorm.clone()).generate(),
        Err(CoreError::NumericalBreakdown { .. })
    ));
    assert!(matches!(
        Hde::<Layout2d<f32>, f32>::from_config(&g_f32, cfg2_unnorm)
            .with_pivots(&pivots)
            .generate(),
        Err(CoreError::NumericalBreakdown { .. })
    ));
    assert!(matches!(
        Hde::<LayoutNd<f32>, f32>::from_config(&g_f32, cfg3_unnorm.clone()).generate(),
        Err(CoreError::NumericalBreakdown { .. })
    ));
    assert!(matches!(
        Hde::<LayoutNd<f32>, f32>::from_config(&g_f32, cfg3_unnorm)
            .with_pivots(&pivots)
            .generate(),
        Err(CoreError::NumericalBreakdown { .. })
    ));

    let ok_2d: Layout2d<f32> = Hde::from_config(&g_f32, cfg2_norm.clone())
        .generate()
        .unwrap();
    assert!(ok_2d.is_all_finite());
    let ok_2d_piv: Layout2d<f32> = Hde::from_config(&g_f32, cfg2_norm)
        .with_pivots(&pivots)
        .generate()
        .unwrap();
    assert!(ok_2d_piv.is_all_finite());
    let ok_3d: LayoutNd<f32> = Hde::from_config(&g_f32, cfg3_norm.clone())
        .generate()
        .unwrap();
    assert!(ok_3d.is_all_finite());
    let ok_3d_piv: LayoutNd<f32> = Hde::from_config(&g_f32, cfg3_norm)
        .with_pivots(&pivots)
        .generate()
        .unwrap();
    assert!(ok_3d_piv.is_all_finite());

    // 2. f64 graph weights overflowing S = f32 on unnormalized cast
    let g_f64 = CsrGraph::<f64>::from_edges(4, &edges, Some(&[1e40_f64; 3]), false, false).unwrap();
    assert!(matches!(
        Hde::<Layout2d<f32>, f64>::new(&g_f64)
            .with_normalize(false)
            .generate(),
        Err(CoreError::NumericalBreakdown { .. })
    ));
    let ok_f64_to_f32: Layout2d<f32> = Hde::new(&g_f64).generate().unwrap();
    assert!(ok_f64_to_f32.is_all_finite());
}
