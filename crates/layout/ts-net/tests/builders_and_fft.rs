use petgraph_layout_ts_net::fft::{fft_1d, fft_2d, Complex};
use petgraph_layout_ts_net::{BhTsNetBuilder, FitTsNetBuilder, LTsNetBuilder, TsNetBuilder};

#[test]
fn test_ts_net_builder_defaults() {
    let builder = TsNetBuilder::<f32>::new();
    let ts_net = builder.build().expect("default builder should be valid");
    assert_eq!(ts_net.perplexity, 40.0);
    assert_eq!(ts_net.n_epochs, 5000);
    assert_eq!(ts_net.initial_lr, 50.0);
    assert_eq!(ts_net.final_lr, 50.0);
    assert_eq!(ts_net.lr_switch, None);
    assert_eq!(ts_net.initial_momentum, 0.5);
    assert_eq!(ts_net.final_momentum, 0.5);
    assert_eq!(ts_net.momentum_switch, None);
    assert_eq!(ts_net.initial_l_kl, Some(1.0));
    assert_eq!(ts_net.final_l_kl, Some(1.0));
    assert_eq!(ts_net.l_kl_switch, None);
    assert_eq!(ts_net.initial_l_c, Some(1.2));
    assert_eq!(ts_net.final_l_c, Some(0.01));
    assert_eq!(ts_net.l_c_switch, None);
    assert_eq!(ts_net.initial_l_r, Some(0.0));
    assert_eq!(ts_net.final_l_r, Some(0.6));
    assert_eq!(ts_net.l_r_switch, None);
    assert_eq!(ts_net.r_eps, 0.05);
    assert_eq!(ts_net.sigma_iters, 50);
    assert_eq!(ts_net.autostop, 1e-7);
    assert_eq!(ts_net.window_size, 40);
}

#[test]
fn test_ts_net_builder_new_star() {
    let builder = TsNetBuilder::<f32>::new_star();
    let ts_net = builder.build().expect("star builder should be valid");
    assert_eq!(ts_net.initial_l_c, Some(0.1));
    assert_eq!(ts_net.perplexity, 40.0);
    assert_eq!(ts_net.n_epochs, 5000);
    assert_eq!(ts_net.initial_l_r, Some(0.0));
    assert_eq!(ts_net.final_l_r, Some(0.6));
}

#[test]
fn test_ts_net_builder_validation() {
    assert!(TsNetBuilder::<f32>::new().perplexity(-1.0).build().is_err());
    assert!(TsNetBuilder::<f32>::new().n_epochs(0).build().is_err());
    assert!(TsNetBuilder::<f32>::new().initial_lr(0.0).build().is_err());
    assert!(TsNetBuilder::<f32>::new().final_lr(0.0).build().is_err());
    assert!(TsNetBuilder::<f32>::new()
        .learning_rate(0.0)
        .build()
        .is_err());
    assert!(TsNetBuilder::<f32>::new()
        .initial_momentum(1.0)
        .build()
        .is_err());
    assert!(TsNetBuilder::<f32>::new()
        .initial_momentum(-0.1)
        .build()
        .is_err());
    assert!(TsNetBuilder::<f32>::new()
        .final_momentum(1.0)
        .build()
        .is_err());
    assert!(TsNetBuilder::<f32>::new().momentum(1.0).build().is_err());
    assert!(TsNetBuilder::<f32>::new()
        .initial_l_kl(Some(-0.1))
        .build()
        .is_err());
    assert!(TsNetBuilder::<f32>::new()
        .final_l_kl(Some(-0.1))
        .build()
        .is_err());
    assert!(TsNetBuilder::<f32>::new()
        .initial_l_c(Some(-0.1))
        .build()
        .is_err());
    assert!(TsNetBuilder::<f32>::new()
        .final_l_c(Some(-0.1))
        .build()
        .is_err());
    assert!(TsNetBuilder::<f32>::new()
        .initial_l_r(Some(-0.1))
        .build()
        .is_err());
    assert!(TsNetBuilder::<f32>::new()
        .final_l_r(Some(-0.1))
        .build()
        .is_err());
    assert!(TsNetBuilder::<f32>::new().r_eps(0.0).build().is_err());
    assert!(TsNetBuilder::<f32>::new().sigma_iters(0).build().is_err());
    assert!(TsNetBuilder::<f32>::new().autostop(-1.0).build().is_err());
    assert!(TsNetBuilder::<f32>::new().window_size(0).build().is_err());
}

#[test]
fn test_bh_ts_net_builder_defaults() {
    let builder = BhTsNetBuilder::<f32>::new();
    let bh_ts_net = builder.build().expect("default builder should be valid");
    assert_eq!(bh_ts_net.perplexity, 40.0);
    assert_eq!(bh_ts_net.theta, 0.5);
    assert_eq!(bh_ts_net.k, 120);
    assert_eq!(bh_ts_net.iterations_stage1, 250);
    assert_eq!(bh_ts_net.exaggeration, 4.0);
    assert_eq!(bh_ts_net.iterations_stage2, 250);
    assert_eq!(bh_ts_net.lambda_c_stage2, 1.2);
    assert_eq!(bh_ts_net.iterations_stage3, 250);
    assert_eq!(bh_ts_net.lambda_c_stage3, 0.01);
    assert_eq!(bh_ts_net.lambda_r_stage3, 0.6);
    assert_eq!(bh_ts_net.learning_rate, 200.0);
    assert_eq!(bh_ts_net.momentum, 0.8);
    assert_eq!(bh_ts_net.power, 2.0);
    assert_eq!(bh_ts_net.epsilon_r, 0.05);
}

#[test]
fn test_bh_ts_net_builder_validation() {
    assert!(BhTsNetBuilder::<f32>::new()
        .perplexity(-1.0)
        .build()
        .is_err());
    assert!(BhTsNetBuilder::<f32>::new().theta(0.0).build().is_err());
    assert!(BhTsNetBuilder::<f32>::new().k(0).build().is_err());
    assert!(BhTsNetBuilder::<f32>::new()
        .learning_rate(0.0)
        .build()
        .is_err());
    assert!(BhTsNetBuilder::<f32>::new().momentum(1.0).build().is_err());
}

#[test]
fn test_fit_ts_net_builder_defaults() {
    let builder = FitTsNetBuilder::<f32>::new();
    let fit_ts_net = builder.build().expect("default builder should be valid");
    assert_eq!(fit_ts_net.intervals, 25);
    assert_eq!(fit_ts_net.interpolation_points, 3);
    assert_eq!(fit_ts_net.perplexity, 40.0);
    assert_eq!(fit_ts_net.theta, 0.5);
    assert_eq!(fit_ts_net.k, 120);
    assert_eq!(fit_ts_net.iterations_stage1, 250);
    assert_eq!(fit_ts_net.exaggeration, 4.0);
    assert_eq!(fit_ts_net.iterations_stage2, 250);
    assert_eq!(fit_ts_net.lambda_c_stage2, 1.2);
    assert_eq!(fit_ts_net.iterations_stage3, 250);
    assert_eq!(fit_ts_net.lambda_c_stage3, 0.01);
    assert_eq!(fit_ts_net.lambda_r_stage3, 0.6);
    assert_eq!(fit_ts_net.learning_rate, 200.0);
    assert_eq!(fit_ts_net.momentum, 0.8);
    assert_eq!(fit_ts_net.power, 2.0);
    assert_eq!(fit_ts_net.epsilon_r, 0.05);
}

#[test]
fn test_fit_ts_net_builder_validation() {
    assert!(FitTsNetBuilder::<f32>::new().intervals(0).build().is_err());
    assert!(FitTsNetBuilder::<f32>::new()
        .interpolation_points(0)
        .build()
        .is_err());
    assert!(FitTsNetBuilder::<f32>::new()
        .perplexity(-1.0)
        .build()
        .is_err());
    assert!(FitTsNetBuilder::<f32>::new().theta(0.0).build().is_err());
    assert!(FitTsNetBuilder::<f32>::new().k(0).build().is_err());
    assert!(FitTsNetBuilder::<f32>::new()
        .learning_rate(0.0)
        .build()
        .is_err());
    assert!(FitTsNetBuilder::<f32>::new().momentum(1.0).build().is_err());
}

#[test]
fn test_l_ts_net_builder_defaults() {
    let builder = LTsNetBuilder::<f32>::new();
    let l_ts_net = builder.build().expect("default builder should be valid");
    assert_eq!(l_ts_net.intervals, 25);
    assert_eq!(l_ts_net.interpolation_points, 3);
    assert_eq!(l_ts_net.perplexity, 40.0);
    assert_eq!(l_ts_net.k, 120);
    assert_eq!(l_ts_net.iterations_stage1, 250);
    assert_eq!(l_ts_net.exaggeration, 4.0);
    assert_eq!(l_ts_net.iterations_stage2, 250);
    assert_eq!(l_ts_net.lambda_c_stage2, 1.2);
    assert_eq!(l_ts_net.iterations_stage3, 250);
    assert_eq!(l_ts_net.lambda_c_stage3, 0.01);
    assert_eq!(l_ts_net.lambda_r_stage3, 0.6);
    assert_eq!(l_ts_net.learning_rate, 200.0);
    assert_eq!(l_ts_net.momentum, 0.8);
    assert_eq!(l_ts_net.power, 2.0);
    assert_eq!(l_ts_net.epsilon_r, 0.05);
}

#[test]
fn test_l_ts_net_builder_validation() {
    assert!(LTsNetBuilder::<f32>::new().intervals(0).build().is_err());
    assert!(LTsNetBuilder::<f32>::new()
        .interpolation_points(0)
        .build()
        .is_err());
    assert!(LTsNetBuilder::<f32>::new()
        .perplexity(-1.0)
        .build()
        .is_err());
    assert!(LTsNetBuilder::<f32>::new().k(0).build().is_err());
    assert!(LTsNetBuilder::<f32>::new()
        .learning_rate(0.0)
        .build()
        .is_err());
    assert!(LTsNetBuilder::<f32>::new().momentum(1.0).build().is_err());
}

#[test]
fn test_fft_roundtrip() {
    // 1D test
    let mut data = vec![
        Complex::new(1.0f32, 0.0),
        Complex::new(2.0, 0.0),
        Complex::new(3.0, 0.0),
        Complex::new(4.0, 0.0),
    ];
    let original = data.clone();
    fft_1d(&mut data, false);
    fft_1d(&mut data, true);
    for (orig, res) in original.iter().zip(data.iter()) {
        assert!((orig.re - res.re).abs() < 1e-5);
        assert!((orig.im - res.im).abs() < 1e-5);
    }

    // 2D test
    let mut data_2d = vec![Complex::new(0.0f32, 0.0); 16];
    data_2d[0] = Complex::new(1.0, 0.0);
    data_2d[5] = Complex::new(2.5, 0.0);
    let orig_2d = data_2d.clone();
    fft_2d(&mut data_2d, 4, 4, false);
    fft_2d(&mut data_2d, 4, 4, true);
    for (orig, res) in orig_2d.iter().zip(data_2d.iter()) {
        assert!((orig.re - res.re).abs() < 1e-5);
        assert!((orig.im - res.im).abs() < 1e-5);
    }
}

#[test]
fn test_tsnet_builders_reject_nan_and_inf() {
    assert!(BhTsNetBuilder::<f64>::new()
        .perplexity(f64::NAN)
        .build()
        .is_err());
    assert!(BhTsNetBuilder::<f64>::new()
        .learning_rate(f64::INFINITY)
        .build()
        .is_err());
    assert!(FitTsNetBuilder::<f64>::new()
        .momentum(f64::NAN)
        .build()
        .is_err());
    assert!(LTsNetBuilder::<f64>::new()
        .perplexity(f64::NAN)
        .build()
        .is_err());
    assert!(TsNetBuilder::<f64>::new()
        .perplexity(f64::NAN)
        .build()
        .is_err());
}
