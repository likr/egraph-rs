use egraph_core::{CoreError, Layout2d, Layout3d, LayoutNd};
use egraph_spatial_fft::{
    Complex32, Complex64, DensityFieldKernel, Fft2dWorkspace, FitField2d, FitFieldConfig,
    FitFieldOutput, FitFieldStats, IbFft2d, IbFftConfig, IbFftGridSpec, InterpolationOrder,
    R2cFft2dWorkspaceF32, ScreenedPoissonConfig, ScreenedPoissonField2d, ScreenedPoissonOutput,
    ScreenedPoissonStats,
};

fn assert_copy<T: Copy>() {}

#[test]
fn test_spatial_fft_public_api_contract() {
    assert_copy::<Complex32>();
    assert_copy::<Complex64>();
    assert_copy::<InterpolationOrder>();
    assert_copy::<DensityFieldKernel>();

    let ib_cfg = IbFftConfig::default()
        .with_order(InterpolationOrder::Bilinear)
        .with_intervals(24)
        .with_min_grid_dim(64)
        .with_max_grid_dim(256);
    let fit_cfg = FitFieldConfig::default()
        .with_ibfft(ib_cfg.clone())
        .with_epsilon_r(0.05)
        .with_compute_entropy(false);
    let sp_cfg = ScreenedPoissonConfig::default()
        .with_kernel(DensityFieldKernel::ScreenedPoisson)
        .with_grid_size(64)
        .with_padding_factor(2)
        .with_exact_adjoint(true);

    let mut ib = IbFft2d::<f64>::new(ib_cfg).unwrap();
    let mut fit = FitField2d::<f64>::new(fit_cfg).unwrap();
    let mut sp = ScreenedPoissonField2d::<f64>::new(sp_cfg).unwrap();
    let mut ws = Fft2dWorkspace::new();
    let _r2c = R2cFft2dWorkspaceF32::new();

    let l2 = Layout2d::from_xy(vec![-0.5, 0.5, 0.0], vec![0.1, -0.2, 0.4]).unwrap();
    ib.prepare_euclidean(&mut ws, &l2).unwrap();
    let _spec: Option<&IbFftGridSpec> = ib.grid_spec();
    let _ib_cfg_ref: &IbFftConfig = ib.config();

    let mut fit_out = FitFieldOutput::new();
    fit.evaluate_euclidean(&l2, &mut fit_out).unwrap();
    let _fit_cfg_ref: &FitFieldConfig = fit.config();
    let _fit_stats_ref: &FitFieldStats = fit.last_stats();
    let _fit_spec_ref: Option<&IbFftGridSpec> = fit.grid_spec();

    let mut sp_out = ScreenedPoissonOutput::new();
    sp.evaluate_euclidean(&l2, &mut sp_out).unwrap();
    let _sp_cfg_ref: &ScreenedPoissonConfig = sp.config();
    let _sp_stats_ref: &ScreenedPoissonStats = &sp_out.stats;

    let l3 = Layout3d::<f64>::zeros(3);
    assert!(matches!(
        fit.evaluate_euclidean(&l3, &mut fit_out),
        Err(CoreError::DimensionMismatch {
            expected: 2,
            actual: 3,
            ..
        })
    ));
    let lnd4 = LayoutNd::<f64>::zeros(3, 4).unwrap();
    assert!(matches!(
        sp.evaluate_euclidean(&lnd4, &mut sp_out),
        Err(CoreError::DimensionMismatch {
            expected: 2,
            actual: 4,
            ..
        })
    ));
}
