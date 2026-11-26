use super::*;
use egraph_core::DeterministicRng;

fn run_fd_gradient_check(exact_adjoint: bool, rho_max: Option<f64>, sigma: f64, kappa: f64) {
    let mut rng = DeterministicRng::from_seed_u64(77);
    let n = 16usize;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    // Tight overdense core (10 pts) + boundary (4 pts) + outer whitespace (2 pts),
    // offset from exact grid lines where bilinear splatting has C^0 derivative kinks.
    for i in 0..10 {
        let angle = 0.31 + (i as f64) * std::f64::consts::TAU / 10.0;
        xs.push(0.17 + 0.25 * angle.cos() + 0.05 * (rng.next_f64() - 0.5));
        ys.push(-0.13 + 0.25 * angle.sin() + 0.05 * (rng.next_f64() - 0.5));
    }
    for i in 0..4 {
        let angle = 0.43 + (i as f64) * std::f64::consts::TAU / 4.0;
        xs.push(0.17 + 0.70 * angle.cos());
        ys.push(-0.13 + 0.70 * angle.sin());
    }
    xs.push(3.23);
    ys.push(-2.91);
    xs.push(-3.13);
    ys.push(2.83);

    let cfg = ScreenedPoissonConfig {
        kernel: DensityFieldKernel::ScreenedPoisson,
        grid_size: 128,
        padding_factor: 2,
        margin_ratio: 0.1,
        sigma: Some(sigma),
        kappa: Some(kappa),
        rho_max,
        exact_adjoint,
        ..ScreenedPoissonConfig::default()
    };
    let mut field = ScreenedPoissonField2d::<f64>::new(cfg).unwrap();
    let bounds = [-5.0, -5.0, 5.0, 5.0];
    let mut base_out = ScreenedPoissonOutput::new();
    field
        .evaluate_xy_in_bounds(&xs, &ys, bounds, &mut base_out)
        .unwrap();
    assert!(!base_out.stats.early_exit_no_excess);
    assert_eq!(
        base_out.stats.fft_2d_calls,
        if exact_adjoint { 6 } else { 4 }
    );

    let eps = 2e-3_f64;
    let mut probe_out = ScreenedPoissonOutput::new();
    let mut max_err = 0.0_f64;
    let mut max_grad = 0.0_f64;
    for i in 0..n {
        let mut xp = xs.clone();
        let mut xm = xs.clone();
        xp[i] += eps;
        xm[i] -= eps;
        field
            .evaluate_xy_in_bounds(&xp, &ys, bounds, &mut probe_out)
            .unwrap();
        let lp = probe_out.energy;
        field
            .evaluate_xy_in_bounds(&xm, &ys, bounds, &mut probe_out)
            .unwrap();
        let lm = probe_out.energy;
        let fd_x = (lp - lm) / (2.0 * eps);
        let anal_grad_x = -base_out.force_x[i];
        max_grad = max_grad.max(anal_grad_x.abs());
        max_err = max_err.max((anal_grad_x - fd_x).abs());

        let mut yp = ys.clone();
        let mut ym = ys.clone();
        yp[i] += eps;
        ym[i] -= eps;
        field
            .evaluate_xy_in_bounds(&xs, &yp, bounds, &mut probe_out)
            .unwrap();
        let lpy = probe_out.energy;
        field
            .evaluate_xy_in_bounds(&xs, &ym, bounds, &mut probe_out)
            .unwrap();
        let lmy = probe_out.energy;
        let fd_y = (lpy - lmy) / (2.0 * eps);
        let anal_grad_y = -base_out.force_y[i];
        max_grad = max_grad.max(anal_grad_y.abs());
        max_err = max_err.max((anal_grad_y - fd_y).abs());
    }

    assert!(
        max_grad > 1e-4,
        "expected non-trivial gradient, got {max_grad}"
    );
    let rel_err = max_err / max_grad.max(1e-12);
    assert!(
        rel_err < 1e-2 && max_err < 2e-4,
        "FD mismatch (exact_adjoint={exact_adjoint}): max_err={max_err}, rel_err={rel_err}, max_grad={max_grad}"
    );

    // Core particles must be pushed outward from core center, never imploding.
    let cx: f64 = xs[..10].iter().sum::<f64>() / 10.0;
    let cy: f64 = ys[..10].iter().sum::<f64>() / 10.0;
    let radial_dot: f64 = (0..10)
        .map(|i| (xs[i] - cx) * base_out.force_x[i] + (ys[i] - cy) * base_out.force_y[i])
        .sum();
    assert!(
        radial_dot > 0.0,
        "Radial outward push check failed (exact_adjoint={exact_adjoint}): radial_dot={radial_dot}"
    );
}

#[test]
fn test_pitfall1_fd_exact_adjoint_and_unmasked_rho_zero() {
    run_fd_gradient_check(true, Some(0.05), 0.35, 0.5);
    run_fd_gradient_check(false, Some(0.0), 0.35, 0.5);
}

#[test]
fn test_unilateral_whitespace_invariance_and_early_exit() {
    // Uniform 8x8 grid with rho_max above peak density -> early exit, zero force.
    let mut xs = Vec::with_capacity(64);
    let mut ys = Vec::with_capacity(64);
    for r in 0..8 {
        for c in 0..8 {
            xs.push(c as f64);
            ys.push(r as f64);
        }
    }
    let cfg = ScreenedPoissonConfig {
        grid_size: 128,
        padding_factor: 2,
        rho_max: Some(10.0),
        ..ScreenedPoissonConfig::default()
    };
    let mut field = ScreenedPoissonField2d::<f64>::new(cfg).unwrap();
    let mut out = ScreenedPoissonOutput::new();
    field.evaluate_xy(&xs, &ys, &mut out).unwrap();
    assert!(out.stats.early_exit_no_excess);
    assert_eq!(out.energy, 0.0);
    assert_eq!(out.stats.fft_2d_calls, 2);
    assert!(out.force_x.iter().all(|&f| f == 0.0));
    assert!(out.force_y.iter().all(|&f| f == 0.0));

    // Dense cluster near (-4, -4) + distant outlier in whitespace at (+4, +4).
    let mut cxs = Vec::new();
    let mut cys = Vec::new();
    for i in 0..32 {
        let ang = (i as f64) * std::f64::consts::TAU / 32.0;
        cxs.push(-4.0 + 0.1 * ang.cos());
        cys.push(-4.0 + 0.1 * ang.sin());
    }
    cxs.push(4.0);
    cys.push(4.0);

    let cfg2 = ScreenedPoissonConfig {
        grid_size: 128,
        padding_factor: 2,
        sigma: Some(0.25),
        rho_max: Some(0.1),
        exact_adjoint: true,
        ..ScreenedPoissonConfig::default()
    };
    let mut field2 = ScreenedPoissonField2d::<f64>::new(cfg2).unwrap();
    let mut out2 = ScreenedPoissonOutput::new();
    field2
        .evaluate_xy_in_bounds(&cxs, &cys, [-5.0, -5.0, 5.0, 5.0], &mut out2)
        .unwrap();
    assert!(!out2.stats.early_exit_no_excess);
    let outlier_fx = out2.force_x[32];
    let outlier_fy = out2.force_y[32];
    let core_force_max = out2.force_x[..32]
        .iter()
        .zip(out2.force_y[..32].iter())
        .map(|(&fx, &fy)| fx.hypot(fy))
        .fold(0.0_f64, f64::max);
    assert!(core_force_max > 1e-3);
    assert!(
        outlier_fx.abs() < 1e-8 && outlier_fy.abs() < 1e-8,
        "whitespace outlier force should be ~0 within f32 FFT noise: ({outlier_fx}, {outlier_fy})"
    );
}

#[test]
fn test_pitfall2_plateau_expansion_screened_poisson_vs_single_gaussian() {
    // Dense uniform 40x40 plateau inside [-1, 1]^2 (with rho_max low enough that plateau is 3x overdense).
    let side = 40usize;
    let mut xs = Vec::with_capacity(side * side);
    let mut ys = Vec::with_capacity(side * side);
    for r in 0..side {
        for c in 0..side {
            let x = -1.0 + 2.0 * (c as f64) / ((side - 1) as f64);
            let y = -1.0 + 2.0 * (r as f64) / ((side - 1) as f64);
            if x * x + y * y <= 1.0 {
                xs.push(x);
                ys.push(y);
            }
        }
    }
    let n = xs.len();
    let bounds = [-3.0, -3.0, 3.0, 3.0];

    let sp_cfg = ScreenedPoissonConfig {
        kernel: DensityFieldKernel::ScreenedPoisson,
        grid_size: 256,
        padding_factor: 2,
        sigma: Some(0.08),
        kappa: Some(0.8),
        rho_max: Some(0.08),
        exact_adjoint: true,
        ..ScreenedPoissonConfig::default()
    };
    let sg_cfg = ScreenedPoissonConfig {
        kernel: DensityFieldKernel::SingleGaussian,
        ..sp_cfg
    };

    let mut sp_field = ScreenedPoissonField2d::<f64>::new(sp_cfg).unwrap();
    let mut sg_field = ScreenedPoissonField2d::<f64>::new(sg_cfg).unwrap();
    let mut sp_out = ScreenedPoissonOutput::new();
    let mut sg_out = ScreenedPoissonOutput::new();
    sp_field
        .evaluate_xy_in_bounds(&xs, &ys, bounds, &mut sp_out)
        .unwrap();
    sg_field
        .evaluate_xy_in_bounds(&xs, &ys, bounds, &mut sg_out)
        .unwrap();

    let mut sp_interior_radial = 0.0_f64;
    let mut sg_interior_radial = 0.0_f64;
    let mut interior_count = 0usize;
    for i in 0..n {
        let r = (xs[i] * xs[i] + ys[i] * ys[i]).sqrt();
        if r > 0.1 && r < 0.55 {
            let urx = xs[i] / r;
            let ury = ys[i] / r;
            sp_interior_radial += sp_out.force_x[i] * urx + sp_out.force_y[i] * ury;
            sg_interior_radial += sg_out.force_x[i] * urx + sg_out.force_y[i] * ury;
            interior_count += 1;
        }
    }
    sp_interior_radial /= interior_count as f64;
    sg_interior_radial /= interior_count as f64;

    assert!(
        sp_interior_radial > 1e-6,
        "ScreenedPoisson must exert positive outward radial force in plateau interior, got {sp_interior_radial}"
    );
    assert!(
        sg_interior_radial.abs() < 0.05 * sp_interior_radial,
        "SingleGaussian interior plateau force should be near zero relative to ScreenedPoisson: sg={sg_interior_radial}, sp={sp_interior_radial}"
    );
}

#[test]
fn test_default_512_memory_budget_and_finite_eval() {
    let mut field = ScreenedPoissonField2d::<f64>::new(ScreenedPoissonConfig::default()).unwrap();
    assert_eq!(field.config().grid_size, 512);
    assert!(field.density_grid().is_none());
    assert_eq!(field.density_grid_dim(), 0);
    let mem = field.workspace_memory_bytes();
    assert!(
        mem <= 6_000_000,
        "512x512 padded f32 workspace exceeded 6 MB budget: {mem} bytes"
    );

    let mut rng = DeterministicRng::from_seed_u64(2048);
    let n = 1024usize;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        xs.push(rng.next_f64() * 2.0 - 1.0);
        ys.push(rng.next_f64() * 2.0 - 1.0);
    }
    let mut out = ScreenedPoissonOutput::new();
    field.evaluate_xy(&xs, &ys, &mut out).unwrap();
    assert_eq!(out.stats.grid_dim, 512);
    assert_eq!(out.stats.pad_dim, 1024);
    assert!(out.energy.is_finite());
    assert!(out.force_x.iter().all(|v| v.is_finite()));
    assert!(out.force_y.iter().all(|v| v.is_finite()));

    let fg = field.density_grid().expect("density_grid populated");
    let og = out.density_grid().expect("output density_grid populated");
    assert_eq!(field.density_grid_dim(), 512);
    assert_eq!(out.density_grid_dim(), 512);
    assert_eq!(fg.len(), 512 * 512);
    assert_eq!(fg, og);
    assert!(fg.iter().all(|v| v.is_finite() && *v >= 0.0));
}

#[test]
fn test_density_grid_early_exit_f32_and_asymmetric_bounds_order() {
    let cfg = ScreenedPoissonConfig {
        grid_size: 64,
        padding_factor: 2,
        rho_max: Some(100.0),
        ..ScreenedPoissonConfig::default()
    };
    let mut field = ScreenedPoissonField2d::<f32>::new(cfg).unwrap();
    let mut out = ScreenedPoissonOutput::new();
    let xs = vec![-2.0_f32, 4.0, 0.5, 1.0];
    let ys = vec![-10.0_f32, 20.0, 2.0, -3.0];

    // Bounds must follow [min_x, min_y, max_x, max_y].
    assert!(
        field
            .evaluate_xy_in_bounds(&xs, &ys, [-2.0, 4.0, -10.0, 20.0], &mut out)
            .is_err()
    );
    assert!(field.density_grid().is_none());

    field.evaluate_xy(&xs, &ys, &mut out).unwrap();
    assert!(out.stats.early_exit_no_excess);
    let auto_grid = field
        .density_grid()
        .expect("early exit still populates density_grid")
        .to_vec();
    assert_eq!(auto_grid.len(), 64 * 64);

    let mut out_bounds = ScreenedPoissonOutput::new();
    field
        .evaluate_xy_in_bounds(&xs, &ys, [-2.0, -10.0, 4.0, 20.0], &mut out_bounds)
        .unwrap();
    assert_eq!(out_bounds.density_grid().unwrap(), auto_grid.as_slice());

    let empty: [f32; 0] = [];
    field.evaluate_xy(&empty, &empty, &mut out_bounds).unwrap();
    assert!(field.density_grid().is_none());
    assert_eq!(field.density_grid_dim(), 0);
    assert!(out_bounds.density_grid().is_none());
}
