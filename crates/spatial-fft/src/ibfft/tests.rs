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
    let spec = ibfft.grid_spec().unwrap();
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
