use egraph_core::{CoreResult, Scalar};

use crate::fft_r2c::{Complex32, R2cFft2dWorkspaceF32};

use super::config::{DensityFieldKernel, ScreenedPoissonConfig, ScreenedPoissonStats};

#[inline]
pub(super) fn add_real_cell(
    buf: &mut [Complex32],
    stride: usize,
    row: usize,
    col: usize,
    delta: f32,
) {
    let slot = &mut buf[row * stride + (col >> 1)];
    if (col & 1) == 0 {
        slot.re += delta;
    } else {
        slot.im += delta;
    }
}

#[inline]
pub(super) fn read_real_cell(buf: &[Complex32], stride: usize, row: usize, col: usize) -> f32 {
    let slot = buf[row * stride + (col >> 1)];
    if (col & 1) == 0 { slot.re } else { slot.im }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn solve_density_pipeline_f32<S: Scalar>(
    config: &ScreenedPoissonConfig,
    h: f64,
    sigma_eff: f64,
    kappa_eff: f64,
    rho_max_eff: f64,
    ws: &mut R2cFft2dWorkspaceF32,
    grid_buf: &mut [Complex32],
    density_grid: &mut Vec<S>,
    excess_mask: &mut [u64],
    kx_sq: &mut [f32],
    ky_sq: &mut [f32],
    gauss_x: &mut [f32],
    gauss_y: &mut [f32],
) -> CoreResult<ScreenedPoissonStats> {
    let g = config.grid_size;
    let pad_dim = g * config.padding_factor;
    let stride = pad_dim / 2 + 1;
    let count_before = ws.fft_2d_call_count();

    // Populate 1D Fourier wavenumber and Gaussian tables.
    let domain_len = (pad_dim as f64) * h;
    let d_omega = std::f64::consts::TAU / domain_len;
    let half_sigma_sq = 0.5 * sigma_eff * sigma_eff;
    for (kx, (k2_slot, gx_slot)) in kx_sq.iter_mut().zip(gauss_x.iter_mut()).enumerate() {
        let wx = (kx as f64) * d_omega;
        let wx2 = wx * wx;
        *k2_slot = wx2 as f32;
        *gx_slot = (-half_sigma_sq * wx2).exp() as f32;
    }
    let half_pad = pad_dim / 2;
    for (ky, (k2_slot, gy_slot)) in ky_sq.iter_mut().zip(gauss_y.iter_mut()).enumerate() {
        let signed = if ky <= half_pad {
            ky as isize
        } else {
            (ky as isize) - (pad_dim as isize)
        };
        let wy = (signed as f64) * d_omega;
        let wy2 = wy * wy;
        *k2_slot = wy2 as f32;
        *gy_slot = (-half_sigma_sq * wy2).exp() as f32;
    }

    // Smooth density: rho_V = K_sigma * rho_hat.
    ws.r2c_2d(grid_buf, pad_dim, pad_dim)?;
    for ky in 0..pad_dim {
        let gy = gauss_y[ky];
        let row = &mut grid_buf[ky * stride..ky * stride + stride];
        for kx in 0..stride {
            let k_gauss = gy * gauss_x[kx];
            row[kx] = row[kx].scale(k_gauss);
        }
    }
    ws.c2r_2d(grid_buf, pad_dim, pad_dim)?;

    // Capture the interior G x G smoothed empirical density rho_V before delta_rho overwrites grid_buf.
    let half_g = g / 2;
    density_grid.clear();
    density_grid.reserve(g * g);
    for v in 0..g {
        let row_off = v * stride;
        for pair_idx in 0..half_g {
            let slot = grid_buf[row_off + pair_idx];
            density_grid.push(S::from_f64((slot.re as f64).max(0.0)));
            density_grid.push(S::from_f64((slot.im as f64).max(0.0)));
        }
    }

    // Unilateral excess source: delta_rho = max(0, rho_V - rho_max), mu_ent === 0.
    excess_mask.fill(0u64);
    let rho_max_f32 = rho_max_eff as f32;
    let half_width = pad_dim / 2;
    let mut max_density = 0.0_f64;
    let mut excess_sum = 0.0_f64;
    let mut total_excess_count = 0usize;
    let mut active_excess_count = 0usize;

    for v in 0..pad_dim {
        let row_off = v * stride;
        let cell_row_base = v * pad_dim;
        let in_active_y = v < g;
        for pair_idx in 0..half_width {
            let slot = &mut grid_buf[row_off + pair_idx];
            let u0 = pair_idx << 1;

            let r0 = slot.re;
            if (r0 as f64) > max_density {
                max_density = r0 as f64;
            }
            if r0 > rho_max_f32 {
                let d0 = r0 - rho_max_f32;
                slot.re = d0;
                excess_sum += d0 as f64;
                total_excess_count += 1;
                if in_active_y && u0 < g {
                    active_excess_count += 1;
                }
                let bit_idx = cell_row_base + u0;
                excess_mask[bit_idx >> 6] |= 1u64 << (bit_idx & 63);
            } else {
                slot.re = 0.0;
            }

            let r1 = slot.im;
            if (r1 as f64) > max_density {
                max_density = r1 as f64;
            }
            if r1 > rho_max_f32 {
                let d1 = r1 - rho_max_f32;
                slot.im = d1;
                excess_sum += d1 as f64;
                total_excess_count += 1;
                if in_active_y && (u0 + 1) < g {
                    active_excess_count += 1;
                }
                let bit_idx = cell_row_base + u0 + 1;
                excess_mask[bit_idx >> 6] |= 1u64 << (bit_idx & 63);
            } else {
                slot.im = 0.0;
            }
        }
    }

    let cell_area = h * h;
    let excess_mass = excess_sum * cell_area;
    let excess_fraction = (active_excess_count as f64) / ((g * g) as f64);

    if total_excess_count == 0 {
        return Ok(ScreenedPoissonStats {
            energy: 0.0,
            max_density,
            excess_fraction: 0.0,
            excess_mass: 0.0,
            sigma_used: sigma_eff,
            kappa_used: kappa_eff,
            rho_max_used: rho_max_eff,
            grid_dim: g,
            pad_dim,
            fft_2d_calls: ws.fft_2d_call_count() - count_before,
            early_exit_no_excess: true,
        });
    }

    // Forward FFT of delta_rho, apply Green operator, and accumulate discrete Parseval energy.
    ws.r2c_2d(grid_buf, pad_dim, pad_dim)?;
    let kappa_sq = (kappa_eff * kappa_eff) as f32;
    let is_screened = matches!(config.kernel, DensityFieldKernel::ScreenedPoisson);
    let exact_adjoint = config.exact_adjoint;
    let mut energy_accum = 0.0_f64;

    for ky in 0..pad_dim {
        let wy2 = ky_sq[ky];
        let gy = gauss_y[ky];
        let row = &mut grid_buf[ky * stride..ky * stride + stride];
        for kx in 0..stride {
            let g_hat = if is_screened {
                1.0_f32 / (kx_sq[kx] + wy2 + kappa_sq)
            } else {
                1.0_f32
            };
            let w_kx = if kx == 0 || kx == half_width {
                1.0_f64
            } else {
                2.0_f64
            };
            let c = row[kx];
            let norm_sq = (c.re as f64) * (c.re as f64) + (c.im as f64) * (c.im as f64);
            energy_accum += w_kx * (g_hat as f64) * norm_sq;

            let scale = if exact_adjoint {
                g_hat
            } else {
                g_hat * gy * gauss_x[kx]
            };
            row[kx] = c.scale(scale);
        }
    }

    let energy = 0.5 * cell_area * energy_accum / ((pad_dim * pad_dim) as f64);
    ws.c2r_2d(grid_buf, pad_dim, pad_dim)?;

    if exact_adjoint {
        // Multiply u = G * delta_rho by indicator 1_{rho_V > rho_max}, then convolve with K_sigma.
        for v in 0..pad_dim {
            let row_off = v * stride;
            let cell_row_base = v * pad_dim;
            for pair_idx in 0..half_width {
                let bit_base = cell_row_base + (pair_idx << 1);
                let word = excess_mask[bit_base >> 6];
                let shift0 = bit_base & 63;
                let slot = &mut grid_buf[row_off + pair_idx];
                if ((word >> shift0) & 1) == 0 {
                    slot.re = 0.0;
                }
                if ((word >> (shift0 + 1)) & 1) == 0 {
                    slot.im = 0.0;
                }
            }
        }

        ws.r2c_2d(grid_buf, pad_dim, pad_dim)?;
        for ky in 0..pad_dim {
            let gy = gauss_y[ky];
            let row = &mut grid_buf[ky * stride..ky * stride + stride];
            for kx in 0..stride {
                row[kx] = row[kx].scale(gy * gauss_x[kx]);
            }
        }
        ws.c2r_2d(grid_buf, pad_dim, pad_dim)?;
    }

    Ok(ScreenedPoissonStats {
        energy,
        max_density,
        excess_fraction,
        excess_mass,
        sigma_used: sigma_eff,
        kappa_used: kappa_eff,
        rho_max_used: rho_max_eff,
        grid_dim: g,
        pad_dim,
        fft_2d_calls: ws.fft_2d_call_count() - count_before,
        early_exit_no_excess: false,
    })
}
