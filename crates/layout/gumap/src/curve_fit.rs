use num_traits::Float;
use petgraph_drawing::DrawingValue;

/// Fits low-dimensional UMAP parameters `(a, b)` from `min_dist` and `spread`
/// by non-linear least squares (Levenberg-Marquardt / Gauss-Newton).
pub fn find_ab_params<S>(min_dist: S, spread: S) -> (S, S)
where
    S: DrawingValue + Float + Default,
{
    let num_samples = 300;
    let max_dist = spread * S::from_f32(3.0).unwrap();
    let step = max_dist / S::from_usize(num_samples).unwrap();

    let mut d_vals = Vec::with_capacity(num_samples);
    let mut target_vals = Vec::with_capacity(num_samples);

    for i in 1..=num_samples {
        let d = S::from_usize(i).unwrap() * step;
        let target = if d <= min_dist {
            S::one()
        } else {
            (-(d - min_dist) / spread).exp()
        };
        d_vals.push(d);
        target_vals.push(target);
    }

    let mut a = S::from_f32(1.0).unwrap();
    let mut b = S::from_f32(1.0).unwrap();
    let mut lambda = S::from_f32(1e-3).unwrap();

    for _ in 0..100 {
        let mut jtj00 = S::zero();
        let mut jtj01 = S::zero();
        let mut jtj11 = S::zero();
        let mut g0 = S::zero();
        let mut g1 = S::zero();
        let mut total_error = S::zero();

        for i in 0..num_samples {
            let d = d_vals[i];
            let target = target_vals[i];

            let d_2b = d.powf(b * S::from_f32(2.0).unwrap());
            let denom = S::one() + a * d_2b;
            let denom_sq = denom * denom;

            let f_val = S::one() / denom;
            let res = f_val - target;
            total_error += res * res;

            let j0 = -d_2b / denom_sq;
            let ln_d = d.ln();
            let j1 = -(S::from_f32(2.0).unwrap() * a * d_2b * ln_d) / denom_sq;

            jtj00 += j0 * j0;
            jtj01 += j0 * j1;
            jtj11 += j1 * j1;

            g0 -= j0 * res;
            g1 -= j1 * res;
        }

        // Solve (J^T J + lambda * I) delta = g
        let a00 = jtj00 + lambda;
        let a01 = jtj01;
        let a11 = jtj11 + lambda;

        let det = a00 * a11 - a01 * a01;
        if det.abs() < S::from_f32(1e-12).unwrap() {
            lambda *= S::from_f32(10.0).unwrap();
            continue;
        }

        let delta_a = (a11 * g0 - a01 * g1) / det;
        let delta_b = (-a01 * g0 + a00 * g1) / det;

        let new_a = (a + delta_a).max(S::from_f32(1e-4).unwrap());
        let new_b = (b + delta_b).max(S::from_f32(1e-4).unwrap());

        // Check new error
        let mut new_error = S::zero();
        for i in 0..num_samples {
            let d = d_vals[i];
            let target = target_vals[i];
            let d_2b = d.powf(new_b * S::from_f32(2.0).unwrap());
            let f_val = S::one() / (S::one() + new_a * d_2b);
            let res = f_val - target;
            new_error += res * res;
        }

        if new_error < total_error {
            a = new_a;
            b = new_b;
            lambda = (lambda * S::from_f32(0.8).unwrap()).max(S::from_f32(1e-7).unwrap());
            if delta_a.abs() < S::from_f32(1e-6).unwrap()
                && delta_b.abs() < S::from_f32(1e-6).unwrap()
            {
                break;
            }
        } else {
            lambda *= S::from_f32(2.0).unwrap();
        }
    }

    (a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_ab_params_defaults() {
        let (a, b) = find_ab_params(0.1f32, 1.0f32);
        // Default UMAP parameters for min_dist=0.1, spread=1.0 are approx a=1.577, b=0.895
        assert!((a - 1.577).abs() < 0.05);
        assert!((b - 0.895).abs() < 0.05);
    }
}
