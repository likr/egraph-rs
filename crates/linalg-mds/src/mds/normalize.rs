use egraph_core::{
    CoreError, CoreResult, DeterministicRng, EuclideanLayout, EuclideanStorageMut, Layout2d, Scalar,
};

#[inline]
pub(super) fn next_gaussian_pair(rng: &mut DeterministicRng) -> (f64, f64) {
    let u1 = rng.next_f64().max(f64::MIN_POSITIVE);
    let u2 = rng.next_f64();
    let r = (-2.0 * u1.ln()).sqrt();
    let (s, c) = (std::f64::consts::TAU * u2).sin_cos();
    (r * c, r * s)
}

#[inline]
pub(super) fn mean_and_iso_std_2d<S: Scalar>(xs: &[S], ys: &[S]) -> (f64, f64, f64) {
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
pub(super) fn mean_and_iso_std_3d<S: Scalar>(xs: &[S], ys: &[S], zs: &[S]) -> (f64, f64, f64, f64) {
    let n_f64 = xs.len() as f64;
    let mx = xs.iter().map(|v| v.to_f64()).sum::<f64>() / n_f64;
    let my = ys.iter().map(|v| v.to_f64()).sum::<f64>() / n_f64;
    let mz = zs.iter().map(|v| v.to_f64()).sum::<f64>() / n_f64;
    let mut sum_sq = 0.0_f64;
    for i in 0..xs.len() {
        let dx = xs[i].to_f64() - mx;
        let dy = ys[i].to_f64() - my;
        let dz = zs[i].to_f64() - mz;
        sum_sq += dx * dx;
        sum_sq += dy * dy;
        sum_sq += dz * dz;
    }
    (mx, my, mz, (sum_sq / (3.0 * n_f64)).sqrt())
}

pub(super) fn mean_and_iso_std_nd(
    coords: &[f64],
    n: usize,
    d: usize,
    means_out: &mut [f64],
) -> f64 {
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

/// Normalizes a row-major `f64` coordinate buffer of shape `(n, d)` in place,
/// matching [`normalize_euclidean_layout`] bit-for-bit.
pub(super) fn normalize_f64_buffer(
    coords: &mut [f64],
    n: usize,
    d: usize,
    target_std: f64,
    seed: u64,
) -> CoreResult<()> {
    if !target_std.is_finite() {
        return Err(CoreError::non_finite("normalize_f64_buffer::target_std"));
    }
    if target_std <= 0.0 {
        return Err(CoreError::invalid_input(
            "target_std must be strictly positive",
        ));
    }
    if n == 0 || d == 0 {
        return Ok(());
    }

    let mut rng = DeterministicRng::from_seed_u64(seed);
    for val in coords.iter_mut() {
        if !val.is_finite() {
            *val = next_gaussian_pair(&mut rng).0;
        }
    }

    if n == 1 {
        coords.fill(0.0);
        return Ok(());
    }

    let mut means = vec![0.0_f64; d];
    let mut std_dev = mean_and_iso_std_nd(coords, n, d, &mut means);
    if !std_dev.is_finite() || std_dev < 1e-12 {
        for i in 0..n {
            let row = &mut coords[i * d..(i + 1) * d];
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
        std_dev = mean_and_iso_std_nd(coords, n, d, &mut means);
        if !std_dev.is_finite() || std_dev < 1e-12 {
            coords.fill(0.0);
            return Ok(());
        }
    }

    let scale = target_std / std_dev;
    for i in 0..n {
        let row = &mut coords[i * d..(i + 1) * d];
        for m in 0..d {
            row[m] = (row[m] - means[m]) * scale;
        }
    }
    Ok(())
}

/// Sanitizes non-finite entries, centers coordinates per axis, and scales isotropically so
/// $\sqrt{\frac{1}{DN} \sum_{i,a} (x_{i,a} - \bar{x}_a)^2} = \mathrm{target\_std}$.
///
/// Collapsed or degenerate layouts (`n >= 2`, `std < 1e-12`) fall back deterministically to
/// standard Gaussian coordinates drawn from `DeterministicRng::from_seed_u64(seed)`.
pub fn normalize_euclidean_layout<L: EuclideanLayout>(
    layout: &mut L,
    target_std: f64,
    seed: u64,
) -> CoreResult<()> {
    if !target_std.is_finite() || !L::Scalar::from_f64(target_std).is_finite() {
        return Err(CoreError::non_finite(
            "normalize_euclidean_layout::target_std",
        ));
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
    match layout.as_euclidean_storage_mut() {
        EuclideanStorageMut::Soa2d { x, y } => {
            for i in 0..n {
                if !x[i].is_finite() {
                    x[i] = L::Scalar::from_f64(next_gaussian_pair(&mut rng).0);
                }
                if !y[i].is_finite() {
                    y[i] = L::Scalar::from_f64(next_gaussian_pair(&mut rng).0);
                }
            }
            if n == 1 {
                x[0] = L::Scalar::ZERO;
                y[0] = L::Scalar::ZERO;
                return Ok(());
            }
            let (mut mx, mut my, mut std_dev) = mean_and_iso_std_2d(x, y);
            if !std_dev.is_finite() || std_dev < 1e-12 {
                for i in 0..n {
                    let (gx, gy) = next_gaussian_pair(&mut rng);
                    x[i] = L::Scalar::from_f64(gx);
                    y[i] = L::Scalar::from_f64(gy);
                }
                (mx, my, std_dev) = mean_and_iso_std_2d(x, y);
                if !std_dev.is_finite() || std_dev < 1e-12 {
                    x.fill(L::Scalar::ZERO);
                    y.fill(L::Scalar::ZERO);
                    return Ok(());
                }
            }
            let scale = target_std / std_dev;
            for i in 0..n {
                x[i] = L::Scalar::from_f64((x[i].to_f64() - mx) * scale);
                y[i] = L::Scalar::from_f64((y[i].to_f64() - my) * scale);
            }
        }
        EuclideanStorageMut::Soa3d { x, y, z } => {
            for i in 0..n {
                if !x[i].is_finite() {
                    x[i] = L::Scalar::from_f64(next_gaussian_pair(&mut rng).0);
                }
                if !y[i].is_finite() {
                    y[i] = L::Scalar::from_f64(next_gaussian_pair(&mut rng).0);
                }
                if !z[i].is_finite() {
                    z[i] = L::Scalar::from_f64(next_gaussian_pair(&mut rng).0);
                }
            }
            if n == 1 {
                x[0] = L::Scalar::ZERO;
                y[0] = L::Scalar::ZERO;
                z[0] = L::Scalar::ZERO;
                return Ok(());
            }
            let (mut mx, mut my, mut mz, mut std_dev) = mean_and_iso_std_3d(x, y, z);
            if !std_dev.is_finite() || std_dev < 1e-12 {
                for i in 0..n {
                    let (gx, gy) = next_gaussian_pair(&mut rng);
                    let (gz, _) = next_gaussian_pair(&mut rng);
                    x[i] = L::Scalar::from_f64(gx);
                    y[i] = L::Scalar::from_f64(gy);
                    z[i] = L::Scalar::from_f64(gz);
                }
                (mx, my, mz, std_dev) = mean_and_iso_std_3d(x, y, z);
                if !std_dev.is_finite() || std_dev < 1e-12 {
                    x.fill(L::Scalar::ZERO);
                    y.fill(L::Scalar::ZERO);
                    z.fill(L::Scalar::ZERO);
                    return Ok(());
                }
            }
            let scale = target_std / std_dev;
            for i in 0..n {
                x[i] = L::Scalar::from_f64((x[i].to_f64() - mx) * scale);
                y[i] = L::Scalar::from_f64((y[i].to_f64() - my) * scale);
                z[i] = L::Scalar::from_f64((z[i].to_f64() - mz) * scale);
            }
        }
        EuclideanStorageMut::FlatNd {
            coords,
            num_nodes,
            dims,
        } => {
            let mut work = Vec::with_capacity(coords.len());
            work.extend(coords.iter().map(|v| v.to_f64()));
            normalize_f64_buffer(&mut work, num_nodes, dims, target_std, seed)?;
            for (dst, &src) in coords.iter_mut().zip(work.iter()) {
                *dst = L::Scalar::from_f64(src);
            }
        }
    }
    Ok(())
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
    normalize_euclidean_layout(layout, target_std, seed)
}
