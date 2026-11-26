use egraph_core::{CoreError, CoreResult, DeterministicRng, Layout2d, LayoutNd, Scalar};

#[inline]
pub(super) fn next_gaussian_pair(rng: &mut DeterministicRng) -> (f64, f64) {
    let u1 = rng.next_f64().max(f64::MIN_POSITIVE);
    let u2 = rng.next_f64();
    let r = (-2.0 * u1.ln()).sqrt();
    let (s, c) = (std::f64::consts::TAU * u2).sin_cos();
    (r * c, r * s)
}

#[inline]
fn mean_and_iso_std<S: Scalar>(xs: &[S], ys: &[S]) -> (f64, f64, f64) {
    let n_f64 = xs.len() as f64;
    let mx = xs.iter().map(|v| v.to_f64()).sum::<f64>() / n_f64;
    let my = ys.iter().map(|v| v.to_f64()).sum::<f64>() / n_f64;
    let mut sum_sq = 0.0_f64;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let dx = x.to_f64() - mx;
        let dy = y.to_f64() - my;
        sum_sq += dx * dx + dy * dy;
    }
    (mx, my, (sum_sq / (2.0 * n_f64)).sqrt())
}

#[inline]
fn median_of_slice(buf: &mut [f64]) -> f64 {
    let len = buf.len();
    debug_assert!(len > 0);
    if len % 2 == 1 {
        *buf.select_nth_unstable_by(len / 2, f64::total_cmp).1
    } else {
        let (left, &mut upper, _) = buf.select_nth_unstable_by(len / 2, f64::total_cmp);
        let lower = left.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        0.5 * (lower + upper)
    }
}

#[inline]
fn quantile_of_slice(buf: &mut [f64], q: f64) -> f64 {
    let len = buf.len();
    debug_assert!(len >= 2);
    let pos = q * ((len - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - (lo as f64);
    let (left, &mut val_hi, _) = buf.select_nth_unstable_by(hi, f64::total_cmp);
    if lo == hi {
        val_hi
    } else {
        let val_lo = left.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        val_lo + frac * (val_hi - val_lo)
    }
}

/// Sanitizes non-finite entries, centers coordinates per axis, and scales isotropically so
/// $\sqrt{\frac{1}{2N} \sum_i ((x_i - \bar{x})^2 + (y_i - \bar{y})^2)} = \mathrm{target\_std}$,
/// matching `numpy.std(coords)` on `(N, 2)` centered arrays.
///
/// Collapsed or degenerate layouts (`n >= 2`, `std < 1e-12`) fall back deterministically to
/// standard Gaussian coordinates drawn from `DeterministicRng::from_seed_u64(seed)`.
pub fn normalize_layout_2d<S: Scalar>(
    layout: &mut Layout2d<S>,
    target_std: f64,
    seed: u64,
) -> CoreResult<()> {
    if !target_std.is_finite() {
        return Err(CoreError::non_finite("normalize_layout_2d::target_std"));
    }
    if target_std <= 0.0 {
        return Err(CoreError::invalid_input(
            "target_std must be strictly positive",
        ));
    }
    let n = layout.len();
    if n == 0 {
        return Ok(());
    }

    let mut rng = DeterministicRng::from_seed_u64(seed);
    let (xs, ys) = layout.xy_mut();
    for i in 0..n {
        if !xs[i].is_finite() {
            xs[i] = S::from_f64(next_gaussian_pair(&mut rng).0);
        }
        if !ys[i].is_finite() {
            ys[i] = S::from_f64(next_gaussian_pair(&mut rng).0);
        }
    }

    if n == 1 {
        xs[0] = S::ZERO;
        ys[0] = S::ZERO;
        return Ok(());
    }

    let (mut mx, mut my, mut std_dev) = mean_and_iso_std(xs, ys);
    if !std_dev.is_finite() || std_dev < 1e-12 {
        for i in 0..n {
            let (gx, gy) = next_gaussian_pair(&mut rng);
            xs[i] = S::from_f64(gx);
            ys[i] = S::from_f64(gy);
        }
        (mx, my, std_dev) = mean_and_iso_std(xs, ys);
        if !std_dev.is_finite() || std_dev < 1e-12 {
            xs.fill(S::ZERO);
            ys.fill(S::ZERO);
            return Ok(());
        }
    }

    let scale = target_std / std_dev;
    for i in 0..n {
        xs[i] = S::from_f64((xs[i].to_f64() - mx) * scale);
        ys[i] = S::from_f64((ys[i].to_f64() - my) * scale);
    }
    Ok(())
}

fn mean_and_iso_std_nd(coords: &[f64], n: usize, d: usize, means_out: &mut [f64]) -> f64 {
    let n_f64 = n as f64;
    means_out.fill(0.0);
    for i in 0..n {
        let row = &coords[i * d..(i + 1) * d];
        for (m, &val) in row.iter().enumerate() {
            means_out[m] += val;
        }
    }
    for mean in means_out.iter_mut() {
        *mean /= n_f64;
    }
    let mut sum_sq = 0.0_f64;
    for i in 0..n {
        let row = &coords[i * d..(i + 1) * d];
        for (m, &val) in row.iter().enumerate() {
            let diff = val - means_out[m];
            sum_sq += diff * diff;
        }
    }
    (sum_sq / ((d as f64) * n_f64)).sqrt()
}

/// Normalizes an N-dimensional layout identically to [`normalize_layout_2d`] (matching bit-for-bit
/// when `d == 2`).
pub(super) fn normalize_layout_nd<S: Scalar>(
    layout: &mut LayoutNd<S>,
    target_std: f64,
    seed: u64,
) -> CoreResult<()> {
    if !target_std.is_finite() {
        return Err(CoreError::non_finite("normalize_layout_nd::target_std"));
    }
    if target_std <= 0.0 {
        return Err(CoreError::invalid_input(
            "target_std must be strictly positive",
        ));
    }
    let n = layout.num_nodes();
    let d = layout.dims();
    if n == 0 {
        return Ok(());
    }

    let mut rng = DeterministicRng::from_seed_u64(seed);
    let coords = layout.coords_mut();
    for val in coords.iter_mut() {
        if !val.is_finite() {
            *val = S::from_f64(next_gaussian_pair(&mut rng).0);
        }
    }

    if n == 1 {
        coords.fill(S::ZERO);
        return Ok(());
    }

    let mut work = Vec::with_capacity(coords.len());
    work.extend(coords.iter().map(|v| v.to_f64()));
    let mut means = vec![0.0_f64; d];
    let mut std_dev = mean_and_iso_std_nd(&work, n, d, &mut means);
    if !std_dev.is_finite() || std_dev < 1e-12 {
        // Fill pairs with next_gaussian_pair so d == 2 consumes the RNG identically to normalize_layout_2d.
        for i in 0..n {
            let row = &mut work[i * d..(i + 1) * d];
            let mut m = 0usize;
            while m < d {
                let (gx, gy) = next_gaussian_pair(&mut rng);
                row[m] = gx;
                if m + 1 < d {
                    row[m + 1] = gy;
                }
                m += 2;
            }
        }
        std_dev = mean_and_iso_std_nd(&work, n, d, &mut means);
        if !std_dev.is_finite() || std_dev < 1e-12 {
            coords.fill(S::ZERO);
            return Ok(());
        }
    }

    let scale = target_std / std_dev;
    for i in 0..n {
        let src = &work[i * d..(i + 1) * d];
        let dst = &mut coords[i * d..(i + 1) * d];
        for m in 0..d {
            dst[m] = S::from_f64((src[m] - means[m]) * scale);
        }
    }
    Ok(())
}

/// Sanitizes non-finite entries, centers coordinates at per-axis medians `(mx, my)`, and scales
/// isotropically using the Gaussian-calibrated radial quantile $r_q / \sqrt{-2 \ln(1 - q)}$ so
/// core spread matches `target_std` even when extreme outliers inflate the bounding box.
///
/// Falls back to isotropic standard deviation around `(mx, my)` if the quantile scale collapses
/// (`< 1e-12`), and deterministically to Gaussian coordinates from `seed` if all points collapse.
pub fn robust_normalize_layout_2d<S: Scalar>(
    layout: &mut Layout2d<S>,
    target_std: f64,
    quantile: f64,
    seed: u64,
) -> CoreResult<()> {
    if !target_std.is_finite() {
        return Err(CoreError::non_finite(
            "robust_normalize_layout_2d::target_std",
        ));
    }
    if target_std <= 0.0 {
        return Err(CoreError::invalid_input(
            "target_std must be strictly positive",
        ));
    }
    if !quantile.is_finite() {
        return Err(CoreError::non_finite(
            "robust_normalize_layout_2d::quantile",
        ));
    }
    if quantile <= 0.0 || quantile >= 1.0 {
        return Err(CoreError::invalid_input(
            "quantile must be strictly in (0, 1)",
        ));
    }

    let n = layout.len();
    if n == 0 {
        return Ok(());
    }

    let mut rng = DeterministicRng::from_seed_u64(seed);
    let (xs, ys) = layout.xy_mut();
    for i in 0..n {
        if !xs[i].is_finite() {
            xs[i] = S::from_f64(next_gaussian_pair(&mut rng).0);
        }
        if !ys[i].is_finite() {
            ys[i] = S::from_f64(next_gaussian_pair(&mut rng).0);
        }
    }

    if n == 1 {
        xs[0] = S::ZERO;
        ys[0] = S::ZERO;
        return Ok(());
    }

    let mut scratch = Vec::with_capacity(n);
    scratch.extend(xs.iter().map(|v| v.to_f64()));
    let mx = median_of_slice(&mut scratch);

    scratch.clear();
    scratch.extend(ys.iter().map(|v| v.to_f64()));
    let my = median_of_slice(&mut scratch);

    scratch.clear();
    for i in 0..n {
        scratch.push((xs[i].to_f64() - mx).hypot(ys[i].to_f64() - my));
    }
    let r_q = quantile_of_slice(&mut scratch, quantile);
    let gauss_factor = (-2.0 * (1.0 - quantile).ln()).sqrt();
    let sigma_hat = r_q / gauss_factor;

    let mut center_x = mx;
    let mut center_y = my;
    let mut sigma = sigma_hat;
    if !sigma.is_finite() || sigma < 1e-12 {
        let mut sum_sq = 0.0_f64;
        for i in 0..n {
            let dx = xs[i].to_f64() - mx;
            let dy = ys[i].to_f64() - my;
            sum_sq += dx * dx + dy * dy;
        }
        let iso_std = (sum_sq / (2.0 * (n as f64))).sqrt();
        if iso_std.is_finite() && iso_std >= 1e-12 {
            sigma = iso_std;
        } else {
            for i in 0..n {
                let (gx, gy) = next_gaussian_pair(&mut rng);
                xs[i] = S::from_f64(gx);
                ys[i] = S::from_f64(gy);
            }
            let (gmx, gmy, g_std) = mean_and_iso_std(xs, ys);
            if !g_std.is_finite() || g_std < 1e-12 {
                xs.fill(S::ZERO);
                ys.fill(S::ZERO);
                return Ok(());
            }
            center_x = gmx;
            center_y = gmy;
            sigma = g_std;
        }
    }

    let scale = target_std / sigma;
    for i in 0..n {
        xs[i] = S::from_f64((xs[i].to_f64() - center_x) * scale);
        ys[i] = S::from_f64((ys[i].to_f64() - center_y) * scale);
    }
    Ok(())
}
