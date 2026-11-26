use egraph_core::{CoreError, CsrGraph, Layout2d, LayoutNd};
use egraph_linalg::shortest_path::select_maxmin_sp_pivots;

use super::normalize::normalize_layout_nd;
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

    // 1. output_dims == 2 matches hde bit-for-bit
    for normalize in [false, true] {
        let cfg2 = HdeConfig {
            dimensions: 5,
            output_dims: 2,
            seed: 7,
            tol: 1e-12,
            max_sweeps: 200,
            normalize,
        };
        let lay2 = hde::<f64, f64>(&g, &cfg2).unwrap();
        let lay_nd2 = hde_nd::<f64, f64>(&g, &cfg2).unwrap();
        assert_eq!(lay_nd2.dims(), 2);
        for i in 0..9 {
            let row = lay_nd2.node(i).unwrap();
            assert_eq!(row[0].to_bits(), lay2.x()[i].to_bits());
            assert_eq!(row[1].to_bits(), lay2.y()[i].to_bits());
        }
    }

    // 2. output_dims == 3 matches dense 3D oracle
    let cfg3_unnorm = HdeConfig {
        dimensions: 5,
        output_dims: 3,
        seed: 7,
        tol: 1e-12,
        max_sweeps: 200,
        normalize: false,
    };
    let pivots = select_maxmin_sp_pivots(&g, 5, 7, 1).unwrap();
    let lay3 = hde_nd::<f64, f64>(&g, &cfg3_unnorm).unwrap();
    let oracle3 = hde_naive_oracle(&g, &pivots, 3);
    let lay3_coords: Vec<Vec<f64>> = (0..9).map(|i| lay3.node(i).unwrap().to_vec()).collect();
    for (a, b) in pairwise_euclidean_nd(&lay3_coords)
        .iter()
        .zip(pairwise_euclidean_nd(&oracle3).iter())
    {
        assert!((a - b).abs() < 1e-10);
    }
    let lay3_norm = hde_nd::<f64, f64>(
        &g,
        &HdeConfig {
            normalize: true,
            ..cfg3_unnorm
        },
    )
    .unwrap();
    assert!((isotropic_std_nd(&lay3_norm) - 1.0).abs() < 1e-12);

    // 3. m_eff < output_dims zero-pads remaining dimensions when normalize=false
    let lay_pad = hde_nd_with_pivots::<f64, f64>(
        &g,
        &[0, 4],
        &HdeConfig {
            output_dims: 4,
            normalize: false,
            ..HdeConfig::default()
        },
    )
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
    let big_lay =
        hde_with_pivots::<f64, f64>(&g_big, &[0, (n_big - 1) as u32], &HdeConfig::default())
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
    let cfg2_norm = HdeConfig {
        dimensions: 4,
        output_dims: 2,
        seed: 42,
        tol: 1e-12,
        max_sweeps: 200,
        normalize: true,
    };
    let cfg2_unnorm = HdeConfig {
        normalize: false,
        ..cfg2_norm
    };
    let cfg3_norm = HdeConfig {
        output_dims: 3,
        ..cfg2_norm
    };
    let cfg3_unnorm = HdeConfig {
        output_dims: 3,
        normalize: false,
        ..cfg2_norm
    };

    let ref2_norm = hde_with_pivots::<f64, f64>(&base_g, &pivots, &cfg2_norm).unwrap();
    let ref2_unnorm = hde_with_pivots::<f64, f64>(&base_g, &pivots, &cfg2_unnorm).unwrap();
    let ref3_norm = hde_nd_with_pivots::<f64, f64>(&base_g, &pivots, &cfg3_norm).unwrap();
    let ref3_unnorm = hde_nd_with_pivots::<f64, f64>(&base_g, &pivots, &cfg3_unnorm).unwrap();

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
        let l2n = hde_with_pivots::<f64, f64>(&g, &pivots, &cfg2_norm).unwrap();
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
        let l2u = hde_with_pivots::<f64, f64>(&g, &pivots, &cfg2_unnorm).unwrap();
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
        let l3n = hde_nd_with_pivots::<f64, f64>(&g, &pivots, &cfg3_norm).unwrap();
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
        let l3u = hde_nd_with_pivots::<f64, f64>(&g, &pivots, &cfg3_unnorm).unwrap();
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
    let l2_sub = hde_with_pivots::<f64, f64>(&g_sub, &pivots, &cfg2_norm).unwrap();
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

    let d2_unnorm = hde_with_pivots::<f64, f64>(&disc_huge, &disc_pivots, &cfg2_unnorm).unwrap();
    assert!(d2_unnorm.is_all_finite());
    let mut d2_ref = Layout2d::from_xy(
        d2_unnorm.x().iter().map(|&v| v / huge).collect(),
        d2_unnorm.y().iter().map(|&v| v / huge).collect(),
    )
    .unwrap();
    normalize_layout_2d(&mut d2_ref, 1.0, cfg2_norm.seed).unwrap();
    let d2_huge = hde_with_pivots::<f64, f64>(&disc_huge, &disc_pivots, &cfg2_norm).unwrap();
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

    let d3_unnorm = hde_nd_with_pivots::<f64, f64>(&disc_huge, &disc_pivots, &cfg3_unnorm).unwrap();
    assert!(d3_unnorm.coords().iter().all(|v| v.is_finite()));
    let mut d3_ref = LayoutNd::<f64>::zeros(4, 3).unwrap();
    for (dst, &src) in d3_ref
        .coords_mut()
        .iter_mut()
        .zip(d3_unnorm.coords().iter())
    {
        *dst = src / huge;
    }
    normalize_layout_nd(&mut d3_ref, 1.0, cfg3_norm.seed).unwrap();
    let d3_huge = hde_nd_with_pivots::<f64, f64>(&disc_huge, &disc_pivots, &cfg3_norm).unwrap();
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
        hde_with_pivots::<f64, f64>(&ovf_g, &ovf_pivots, &cfg2_unnorm),
        Err(CoreError::NumericalBreakdown { .. })
    ));
    let ovf_norm = hde_with_pivots::<f64, f64>(&ovf_g, &ovf_pivots, &cfg2_norm).unwrap();
    assert!(ovf_norm.is_all_finite());
    assert!((isotropic_std(&ovf_norm) - 1.0).abs() < 1e-12);
}

#[test]
fn test_hde_f32_output_overflow_returns_numerical_breakdown() {
    let edges = vec![(0, 1), (1, 2), (2, 3)];
    let pivots = [0u32, 3];

    // 1. Valid positive finite f32 weights whose unnormalized projection exceeds f32::MAX
    let g_f32 = CsrGraph::<f32>::from_edges(4, &edges, Some(&[3e38_f32; 3]), false, false).unwrap();

    let cfg2_unnorm = HdeConfig {
        normalize: false,
        ..HdeConfig::default()
    };
    let cfg2_norm = HdeConfig::default();
    let cfg3_unnorm = HdeConfig {
        output_dims: 3,
        normalize: false,
        ..HdeConfig::default()
    };
    let cfg3_norm = HdeConfig {
        output_dims: 3,
        normalize: true,
        ..HdeConfig::default()
    };

    assert!(matches!(
        hde::<f32, f32>(&g_f32, &cfg2_unnorm),
        Err(CoreError::NumericalBreakdown { .. })
    ));
    assert!(matches!(
        hde_with_pivots::<f32, f32>(&g_f32, &pivots, &cfg2_unnorm),
        Err(CoreError::NumericalBreakdown { .. })
    ));
    assert!(matches!(
        hde_nd::<f32, f32>(&g_f32, &cfg3_unnorm),
        Err(CoreError::NumericalBreakdown { .. })
    ));
    assert!(matches!(
        hde_nd_with_pivots::<f32, f32>(&g_f32, &pivots, &cfg3_unnorm),
        Err(CoreError::NumericalBreakdown { .. })
    ));

    let ok_2d = hde::<f32, f32>(&g_f32, &cfg2_norm).unwrap();
    assert!(ok_2d.is_all_finite());
    let ok_2d_piv = hde_with_pivots::<f32, f32>(&g_f32, &pivots, &cfg2_norm).unwrap();
    assert!(ok_2d_piv.is_all_finite());
    let ok_3d = hde_nd::<f32, f32>(&g_f32, &cfg3_norm).unwrap();
    assert!(ok_3d.is_all_finite());
    let ok_3d_piv = hde_nd_with_pivots::<f32, f32>(&g_f32, &pivots, &cfg3_norm).unwrap();
    assert!(ok_3d_piv.is_all_finite());

    // 2. f64 graph weights overflowing S = f32 on unnormalized cast
    let g_f64 = CsrGraph::<f64>::from_edges(4, &edges, Some(&[1e40_f64; 3]), false, false).unwrap();
    assert!(matches!(
        hde::<f64, f32>(&g_f64, &cfg2_unnorm),
        Err(CoreError::NumericalBreakdown { .. })
    ));
    let ok_f64_to_f32 = hde::<f64, f32>(&g_f64, &cfg2_norm).unwrap();
    assert!(ok_f64_to_f32.is_all_finite());
}
