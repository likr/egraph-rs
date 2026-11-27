use egraph_core::{
    CoreError, CoreResult, DeterministicRng, EuclideanLayout, EuclideanStorageMut, Layout2d, Scalar,
};

use super::normalize::{
    mean_and_iso_std_2d, mean_and_iso_std_3d, mean_and_iso_std_nd, next_gaussian_pair,
};

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

fn ln_gamma(z: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let z = z - 1.0;
    let mut x = COEFFS[0];
    for (i, &c) in COEFFS.iter().enumerate().skip(1) {
        x += c / (z + (i as f64));
    }
    let t = z + 7.5;
    0.5 * std::f64::consts::TAU.ln() + (z + 0.5) * t.ln() - t + x.ln()
}

fn regularized_lower_gamma(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let log_prefix = a * x.ln() - x - ln_gamma(a);
    if x < a + 1.0 {
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..200 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() <= sum.abs() * 1e-14 {
                break;
            }
        }
        (sum * log_prefix.exp()).clamp(0.0, 1.0)
    } else {
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / 1e-30;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..=200 {
            let i_f64 = i as f64;
            let an = -i_f64 * (i_f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < 1e-30 {
                d = 1e-30;
            }
            c = b + an / c;
            if c.abs() < 1e-30 {
                c = 1e-30;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() <= 1e-14 {
                break;
            }
        }
        (1.0 - h * log_prefix.exp()).clamp(0.0, 1.0)
    }
}

#[inline]
pub(super) fn gaussian_radial_quantile(dims: usize, quantile: f64) -> f64 {
    if dims == 2 {
        return (-2.0 * (1.0 - quantile).ln()).sqrt();
    }
    let a = 0.5 * (dims as f64);
    let mut lo = 0.0_f64;
    let mut hi = (dims as f64).max(1.0);
    while regularized_lower_gamma(a, 0.5 * hi) < quantile {
        hi *= 2.0;
        if hi > 1e12 {
            break;
        }
    }
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        if regularized_lower_gamma(a, 0.5 * mid) < quantile {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (0.5 * (lo + hi)).sqrt()
}

fn robust_normalize_nd_storage<S: Scalar>(
    coords: &mut [S],
    n: usize,
    d: usize,
    target_std: f64,
    quantile: f64,
    rng: &mut DeterministicRng,
) {
    for val in coords.iter_mut() {
        if !val.is_finite() {
            *val = S::from_f64(next_gaussian_pair(rng).0);
        }
    }
    if n == 1 {
        coords.fill(S::ZERO);
        return;
    }

    let mut work = Vec::with_capacity(coords.len());
    work.extend(coords.iter().map(|v| v.to_f64()));

    let mut scratch = Vec::with_capacity(n);
    let mut centers = vec![0.0_f64; d];
    for axis in 0..d {
        scratch.clear();
        for i in 0..n {
            scratch.push(work[i * d + axis]);
        }
        centers[axis] = median_of_slice(&mut scratch);
    }

    scratch.clear();
    for i in 0..n {
        let row = &work[i * d..(i + 1) * d];
        let mut sum_sq = 0.0_f64;
        for (axis, &val) in row.iter().enumerate() {
            let diff = val - centers[axis];
            sum_sq += diff * diff;
        }
        scratch.push(sum_sq.sqrt());
    }
    let r_q = quantile_of_slice(&mut scratch, quantile);
    let gauss_factor = gaussian_radial_quantile(d, quantile);
    let mut sigma = r_q / gauss_factor;

    if !sigma.is_finite() || sigma < 1e-12 {
        let mut sum_sq = 0.0_f64;
        for i in 0..n {
            let row = &work[i * d..(i + 1) * d];
            for (axis, &val) in row.iter().enumerate() {
                let diff = val - centers[axis];
                sum_sq += diff * diff;
            }
        }
        let iso_std = (sum_sq / ((d * n) as f64)).sqrt();
        if iso_std.is_finite() && iso_std >= 1e-12 {
            sigma = iso_std;
        } else {
            for i in 0..n {
                let row = &mut work[i * d..(i + 1) * d];
                let mut m = 0usize;
                while m < d {
                    let (gx, gy) = next_gaussian_pair(rng);
                    row[m] = gx;
                    if m + 1 < d {
                        row[m + 1] = gy;
                    }
                    m += 2;
                }
            }
            let g_std = mean_and_iso_std_nd(&work, n, d, &mut centers);
            if !g_std.is_finite() || g_std < 1e-12 {
                coords.fill(S::ZERO);
                return;
            }
            sigma = g_std;
        }
    }

    let scale = target_std / sigma;
    for i in 0..n {
        let src = &work[i * d..(i + 1) * d];
        let dst = &mut coords[i * d..(i + 1) * d];
        for axis in 0..d {
            dst[axis] = S::from_f64((src[axis] - centers[axis]) * scale);
        }
    }
}

/// Sanitizes non-finite entries, centers coordinates at per-axis medians, and scales
/// isotropically using the Gaussian-calibrated radial quantile so core spread matches
/// `target_std` even when extreme outliers inflate the bounding box.
///
/// Falls back to isotropic standard deviation around medians if the quantile scale collapses
/// (`< 1e-12`), and deterministically to Gaussian coordinates from `seed` if all points collapse.
pub fn robust_normalize_euclidean_layout<L: EuclideanLayout>(
    layout: &mut L,
    target_std: f64,
    quantile: f64,
    seed: u64,
) -> CoreResult<()> {
    if !target_std.is_finite() || !L::Scalar::from_f64(target_std).is_finite() {
        return Err(CoreError::non_finite(
            "robust_normalize_euclidean_layout::target_std",
        ));
    }
    if target_std <= 0.0 {
        return Err(CoreError::invalid_input(
            "target_std must be strictly positive",
        ));
    }
    if !quantile.is_finite() {
        return Err(CoreError::non_finite(
            "robust_normalize_euclidean_layout::quantile",
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

            let mut scratch = Vec::with_capacity(n);
            scratch.extend(x.iter().map(|v| v.to_f64()));
            let mut center_x = median_of_slice(&mut scratch);

            scratch.clear();
            scratch.extend(y.iter().map(|v| v.to_f64()));
            let mut center_y = median_of_slice(&mut scratch);

            scratch.clear();
            for i in 0..n {
                scratch.push((x[i].to_f64() - center_x).hypot(y[i].to_f64() - center_y));
            }
            let r_q = quantile_of_slice(&mut scratch, quantile);
            let gauss_factor = gaussian_radial_quantile(2, quantile);
            let mut sigma = r_q / gauss_factor;
            if !sigma.is_finite() || sigma < 1e-12 {
                let mut sum_sq = 0.0_f64;
                for i in 0..n {
                    let dx = x[i].to_f64() - center_x;
                    let dy = y[i].to_f64() - center_y;
                    sum_sq += dx * dx + dy * dy;
                }
                let iso_std = (sum_sq / (2.0 * (n as f64))).sqrt();
                if iso_std.is_finite() && iso_std >= 1e-12 {
                    sigma = iso_std;
                } else {
                    for i in 0..n {
                        let (gx, gy) = next_gaussian_pair(&mut rng);
                        x[i] = L::Scalar::from_f64(gx);
                        y[i] = L::Scalar::from_f64(gy);
                    }
                    let (gmx, gmy, g_std) = mean_and_iso_std_2d(x, y);
                    if !g_std.is_finite() || g_std < 1e-12 {
                        x.fill(L::Scalar::ZERO);
                        y.fill(L::Scalar::ZERO);
                        return Ok(());
                    }
                    center_x = gmx;
                    center_y = gmy;
                    sigma = g_std;
                }
            }
            let scale = target_std / sigma;
            for i in 0..n {
                x[i] = L::Scalar::from_f64((x[i].to_f64() - center_x) * scale);
                y[i] = L::Scalar::from_f64((y[i].to_f64() - center_y) * scale);
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

            let mut scratch = Vec::with_capacity(n);
            scratch.extend(x.iter().map(|v| v.to_f64()));
            let mut center_x = median_of_slice(&mut scratch);
            scratch.clear();
            scratch.extend(y.iter().map(|v| v.to_f64()));
            let mut center_y = median_of_slice(&mut scratch);
            scratch.clear();
            scratch.extend(z.iter().map(|v| v.to_f64()));
            let mut center_z = median_of_slice(&mut scratch);

            scratch.clear();
            for i in 0..n {
                let dx = x[i].to_f64() - center_x;
                let dy = y[i].to_f64() - center_y;
                let dz = z[i].to_f64() - center_z;
                scratch.push((dx * dx + dy * dy + dz * dz).sqrt());
            }
            let r_q = quantile_of_slice(&mut scratch, quantile);
            let gauss_factor = gaussian_radial_quantile(3, quantile);
            let mut sigma = r_q / gauss_factor;
            if !sigma.is_finite() || sigma < 1e-12 {
                let mut sum_sq = 0.0_f64;
                for i in 0..n {
                    let dx = x[i].to_f64() - center_x;
                    let dy = y[i].to_f64() - center_y;
                    let dz = z[i].to_f64() - center_z;
                    sum_sq += dx * dx + dy * dy + dz * dz;
                }
                let iso_std = (sum_sq / (3.0 * (n as f64))).sqrt();
                if iso_std.is_finite() && iso_std >= 1e-12 {
                    sigma = iso_std;
                } else {
                    for i in 0..n {
                        let (gx, gy) = next_gaussian_pair(&mut rng);
                        let (gz, _) = next_gaussian_pair(&mut rng);
                        x[i] = L::Scalar::from_f64(gx);
                        y[i] = L::Scalar::from_f64(gy);
                        z[i] = L::Scalar::from_f64(gz);
                    }
                    let (gmx, gmy, gmz, g_std) = mean_and_iso_std_3d(x, y, z);
                    if !g_std.is_finite() || g_std < 1e-12 {
                        x.fill(L::Scalar::ZERO);
                        y.fill(L::Scalar::ZERO);
                        z.fill(L::Scalar::ZERO);
                        return Ok(());
                    }
                    center_x = gmx;
                    center_y = gmy;
                    center_z = gmz;
                    sigma = g_std;
                }
            }
            let scale = target_std / sigma;
            for i in 0..n {
                x[i] = L::Scalar::from_f64((x[i].to_f64() - center_x) * scale);
                y[i] = L::Scalar::from_f64((y[i].to_f64() - center_y) * scale);
                z[i] = L::Scalar::from_f64((z[i].to_f64() - center_z) * scale);
            }
        }
        EuclideanStorageMut::FlatNd {
            coords,
            num_nodes,
            dims,
        } => {
            robust_normalize_nd_storage(coords, num_nodes, dims, target_std, quantile, &mut rng);
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
    robust_normalize_euclidean_layout(layout, target_std, quantile, seed)
}
