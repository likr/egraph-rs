use super::*;
use crate::ibfft::InterpolationOrder;

fn brute_force_kl_and_entropy(
    layout: &Layout2d<f64>,
    epsilon_r: f64,
) -> (Vec<f64>, Vec<f64>, f64, Vec<f64>, Vec<f64>) {
    let n = layout.len();
    let mut kl_x = vec![0.0; n];
    let mut kl_y = vec![0.0; n];
    let mut ent_x = vec![0.0; n];
    let mut ent_y = vec![0.0; n];
    let mut total_z = 0.0;

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = layout.x()[i] - layout.x()[j];
            let dy = layout.y()[i] - layout.y()[j];
            let d_sq = dx * dx + dy * dy;
            let w = 1.0 / (1.0 + d_sq);
            total_z += w;
            let w2 = w * w;
            kl_x[i] += w2 * dx;
            kl_y[i] += w2 * dy;

            let ke = 1.0 / (epsilon_r + d_sq);
            ent_x[i] += ke * dx;
            ent_y[i] += ke * dy;
        }
    }

    let denom = total_z.max(1e-12);
    for i in 0..n {
        kl_x[i] /= denom;
        kl_y[i] /= denom;
    }
    (kl_x, kl_y, denom, ent_x, ent_y)
}

#[test]
fn test_fit_field_fused_fft_call_counts_and_accuracy_bilinear_and_cubic() {
    let layout = Layout2d::from_xy(
        vec![-3.1, -1.8, -0.4, 0.2, 1.3, 2.7, -2.5, 2.1, 0.7, -1.0],
        vec![-2.2, 1.5, -0.7, 2.4, -1.9, 0.8, 2.6, -2.8, 0.3, -0.1],
    )
    .unwrap();
    let eps_r = 0.05;
    let (ex_kl_x, ex_kl_y, ex_z, ex_ent_x, ex_ent_y) = brute_force_kl_and_entropy(&layout, eps_r);

    for (order, max_kl_err, max_ent_rel_err, max_z_rel_err) in [
        (InterpolationOrder::Bilinear, 5e-3, 1e-2, 0.02),
        (InterpolationOrder::Cubic, 1e-3, 1e-3, 0.01),
    ] {
        let mut field =
            FitField2d::<f64>::new(FitFieldConfig::default().with_order(order)).unwrap();
        let mut out = FitFieldOutput::new();

        field.reset_fft_2d_call_count();
        field.evaluate(&layout, &mut out).unwrap();

        assert_eq!(out.stats.particle_forward_ffts, 2);
        assert!(out.stats.total_fft_2d_calls <= 8);
        assert_eq!(out.stats.total_fft_2d_calls, 8);
        assert_eq!(field.fft_2d_call_count(), 8);

        let z_rel = (out.total_z - ex_z).abs() / ex_z;
        assert!(z_rel < max_z_rel_err, "{order:?}: z_rel={z_rel}");

        for i in 0..layout.len() {
            let kl_err =
                (out.kl_repulsion_x[i] - ex_kl_x[i]).hypot(out.kl_repulsion_y[i] - ex_kl_y[i]);
            assert!(kl_err < max_kl_err, "{order:?}[{i}]: kl_err={kl_err}");

            let ent_norm = ex_ent_x[i].hypot(ex_ent_y[i]).max(1e-6);
            let ent_rel = (out.entropy_repulsion_x[i] - ex_ent_x[i])
                .hypot(out.entropy_repulsion_y[i] - ex_ent_y[i])
                / ent_norm;
            assert!(
                ent_rel < max_ent_rel_err,
                "{order:?}[{i}]: ent_rel={ent_rel}"
            );
        }
    }
}

#[test]
fn test_fit_field_kl_only_uses_five_ffts_and_f32_supported() {
    let layout = Layout2d::from_xy(vec![0.0_f32, 1.0, -1.0], vec![0.5, -0.5, 0.2]).unwrap();
    let mut field = FitField2d::<f32>::new(FitFieldConfig {
        compute_entropy: false,
        ..FitFieldConfig::default()
    })
    .unwrap();
    let mut out = FitFieldOutput::new();

    field.evaluate(&layout, &mut out).unwrap();
    assert_eq!(out.stats.particle_forward_ffts, 2);
    assert_eq!(out.stats.total_fft_2d_calls, 5);
    assert!(out.entropy_repulsion_x.is_empty());
    assert!(out.total_z.is_finite() && out.total_z > 0.0);
}

#[test]
fn test_fit_field_validation_errors() {
    assert!(matches!(
        FitField2d::<f64>::new(FitFieldConfig {
            epsilon_r: 0.0,
            ..FitFieldConfig::default()
        }),
        Err(CoreError::InvalidInput(_))
    ));

    let mut field = FitField2d::<f64>::default();
    let bad = Layout2d::from_xy(vec![0.0, f64::NAN], vec![0.0, 1.0]).unwrap();
    let mut out = FitFieldOutput::new();
    assert!(matches!(
        field.evaluate(&bad, &mut out),
        Err(CoreError::NonFiniteValue { .. })
    ));
}
