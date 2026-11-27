use super::*;

#[test]
fn test_ibfft_default_is_bilinear_and_matches_exact_cauchy() {
    let mut ws = Fft2dWorkspace::new();
    let mut ibfft = IbFft2d::<f64>::default();
    assert_eq!(ibfft.config().order, InterpolationOrder::Bilinear);

    let layout = Layout2d::from_xy(
        vec![-1.2, -0.4, 0.2, 0.9, 1.5, -0.8, 0.6, 1.1],
        vec![0.7, -0.9, 0.3, -0.2, 0.8, 1.2, -1.1, 0.1],
    )
    .unwrap();
    let n = layout.len();

    let fwd = ibfft.prepare(&mut ws, &layout).unwrap();
    assert_eq!(fwd, 2);

    let mut phi_1 = vec![0.0; n];
    let mut fx = vec![0.0; n];
    let mut fy = vec![0.0; n];
    ibfft
        .evaluate_isotropic_kernel(
            &mut ws,
            |d_sq| 1.0 / (1.0 + d_sq),
            &mut phi_1,
            &mut fx,
            &mut fy,
        )
        .unwrap();
    assert_eq!(ws.fft_2d_call_count(), 5);

    // Compare against exact O(N^2) sum (including K(0) = 1.0 self term in phi_1).
    for i in 0..n {
        let mut exact_phi = 0.0;
        let mut exact_fx = 0.0;
        let mut exact_fy = 0.0;
        for j in 0..n {
            let dx = layout.x()[i] - layout.x()[j];
            let dy = layout.y()[i] - layout.y()[j];
            let k = 1.0 / (1.0 + dx * dx + dy * dy);
            exact_phi += k;
            exact_fx += dx * k;
            exact_fy += dy * k;
        }
        assert!(
            (phi_1[i] - exact_phi).abs() < 5e-3,
            "phi_1[{i}]={} vs {exact_phi}",
            phi_1[i]
        );
        assert!((fx[i] - exact_fx).abs() < 5e-3);
        assert!((fy[i] - exact_fy).abs() < 5e-3);
    }
}

#[test]
fn test_ibfft_cubic_and_bounded_adaptive_grid_size() {
    let mut ws = Fft2dWorkspace::new();
    let mut ibfft = IbFft2d::<f32>::with_order(InterpolationOrder::Cubic);

    // Extreme coordinate span (> 512) must clamp grid to max_grid_dim = 512 and stay finite.
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for k in 0..12 {
        let v = -1200.0_f32 + (k as f32) * 240.0;
        xs.push(v);
        ys.push(v * 0.5);
    }
    let layout = Layout2d::from_xy(xs, ys).unwrap();
    ibfft.prepare(&mut ws, &layout).unwrap();
    let spec = ibfft.grid_spec().unwrap().clone();
    assert_eq!(spec.nx, 512);
    assert_eq!(spec.ny, 512);
    assert_eq!(spec.effective_order, InterpolationOrder::Bilinear);
    assert!(spec.softening_sq > 0.0);

    let mut phi_1 = vec![0.0_f32; layout.len()];
    let mut fx = vec![0.0_f32; layout.len()];
    let mut fy = vec![0.0_f32; layout.len()];
    ibfft
        .evaluate_isotropic_kernel(
            &mut ws,
            |d_sq| 1.0 / (1.0 + spec.softening_sq + d_sq),
            &mut phi_1,
            &mut fx,
            &mut fy,
        )
        .unwrap();
    assert!(phi_1.iter().all(|v| v.is_finite() && *v > 0.0));
    assert!(fx.iter().all(|v| v.is_finite()));
    assert!(fy.iter().all(|v| v.is_finite()));

    assert!(matches!(
        InterpolationOrder::from_order(2),
        Err(CoreError::InvalidInput(_))
    ));
}

#[test]
fn test_ibfft_euclidean_layout_entrypoints_and_config_builders() {
    use egraph_core::{Layout3d, LayoutNd};

    let cfg = IbFftConfig::default()
        .with_order(InterpolationOrder::Cubic)
        .with_intervals(30)
        .with_min_grid_dim(32)
        .with_max_grid_dim(256);
    assert_eq!(cfg.order, InterpolationOrder::Cubic);
    assert_eq!(cfg.intervals, 30);
    assert_eq!(cfg.min_grid_dim, 32);
    assert_eq!(cfg.max_grid_dim, 256);

    let xs = vec![-1.2, -0.4, 0.2, 0.9, 1.5, -0.8];
    let ys = vec![0.7, -0.9, 0.3, -0.2, 0.8, 1.2];
    let l2 = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
    let mut flat = Vec::with_capacity(xs.len() * 2);
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        flat.push(x);
        flat.push(y);
    }
    let lnd2 = LayoutNd::from_flat(xs.len(), 2, &flat).unwrap();

    let mut ws1 = Fft2dWorkspace::new();
    let mut ib1 = IbFft2d::<f64>::new(cfg.clone()).unwrap();
    let fwd1 = ib1.prepare(&mut ws1, &l2).unwrap();

    let mut ws2 = Fft2dWorkspace::new();
    let mut ib2 = IbFft2d::<f64>::new(cfg.clone()).unwrap();
    let fwd2 = ib2.prepare_euclidean(&mut ws2, &l2).unwrap();

    let mut ws3 = Fft2dWorkspace::new();
    let mut ib3 = IbFft2d::<f64>::new(cfg).unwrap();
    let fwd3 = ib3.prepare_euclidean(&mut ws3, &lnd2).unwrap();

    assert_eq!(fwd1, fwd2);
    assert_eq!(fwd1, fwd3);
    assert_eq!(ib1.grid_spec(), ib2.grid_spec());
    assert_eq!(ib1.grid_spec(), ib3.grid_spec());

    let mut phi1 = vec![0.0; xs.len()];
    let mut fx1 = vec![0.0; xs.len()];
    let mut fy1 = vec![0.0; xs.len()];
    ib1.evaluate_isotropic_kernel(
        &mut ws1,
        |d2| 1.0 / (1.0 + d2),
        &mut phi1,
        &mut fx1,
        &mut fy1,
    )
    .unwrap();

    let mut phi3 = vec![0.0; xs.len()];
    let mut fx3 = vec![0.0; xs.len()];
    let mut fy3 = vec![0.0; xs.len()];
    ib3.evaluate_isotropic_kernel(
        &mut ws3,
        |d2| 1.0 / (1.0 + d2),
        &mut phi3,
        &mut fx3,
        &mut fy3,
    )
    .unwrap();
    assert_eq!(phi1, phi3);
    assert_eq!(fx1, fx3);
    assert_eq!(fy1, fy3);

    // Non-2D Euclidean layouts fail fast with DimensionMismatch.
    let l3 = Layout3d::<f64>::zeros(4);
    assert!(matches!(
        ib3.prepare_euclidean(&mut ws3, &l3),
        Err(CoreError::DimensionMismatch {
            expected: 2,
            actual: 3,
            ..
        })
    ));
    let lnd4 = LayoutNd::<f64>::zeros(4, 4).unwrap();
    assert!(matches!(
        ib3.prepare_euclidean(&mut ws3, &lnd4),
        Err(CoreError::DimensionMismatch {
            expected: 2,
            actual: 4,
            ..
        })
    ));
}
