use super::*;
use egraph_core::DeterministicRng;

fn pack_real_grid(real: &[f32], width: usize, height: usize) -> Vec<Complex32> {
    let stride = width / 2 + 1;
    let mut buf = vec![Complex32::ZERO; height * stride];
    for v in 0..height {
        for u in 0..width {
            let val = real[v * width + u];
            let slot = &mut buf[v * stride + (u >> 1)];
            if (u & 1) == 0 {
                slot.re = val;
            } else {
                slot.im = val;
            }
        }
    }
    buf
}

fn unpack_real_grid(buf: &[Complex32], width: usize, height: usize) -> Vec<f32> {
    let stride = width / 2 + 1;
    let mut out = vec![0.0_f32; height * width];
    for v in 0..height {
        for u in 0..width {
            let slot = buf[v * stride + (u >> 1)];
            out[v * width + u] = if (u & 1) == 0 { slot.re } else { slot.im };
        }
    }
    out
}

fn naive_2d_dft_half(real: &[f32], width: usize, height: usize) -> Vec<Complex32> {
    let stride = width / 2 + 1;
    let mut out = vec![Complex32::ZERO; height * stride];
    let w_f = width as f64;
    let h_f = height as f64;
    for ky in 0..height {
        for kx in 0..stride {
            let mut sum_re = 0.0_f64;
            let mut sum_im = 0.0_f64;
            for v in 0..height {
                for u in 0..width {
                    let angle =
                        -std::f64::consts::TAU * ((kx * u) as f64 / w_f + (ky * v) as f64 / h_f);
                    let (s, c) = angle.sin_cos();
                    let x = real[v * width + u] as f64;
                    sum_re += x * c;
                    sum_im += x * s;
                }
            }
            out[ky * stride + kx] = Complex32::new(sum_re as f32, sum_im as f32);
        }
    }
    out
}

#[test]
fn test_r2c_2d_matches_naive_dft_and_parseval() {
    let mut rng = DeterministicRng::from_seed_u64(12345);
    for (width, height) in [(4usize, 2usize), (16, 8), (32, 32)] {
        let mut real = Vec::with_capacity(width * height);
        for _ in 0..(width * height) {
            real.push((rng.next_f64() * 2.0 - 1.0) as f32);
        }
        let expected = naive_2d_dft_half(&real, width, height);
        let mut buf = pack_real_grid(&real, width, height);
        let mut ws = R2cFft2dWorkspaceF32::new();
        ws.r2c_2d(&mut buf, width, height).unwrap();
        assert_eq!(ws.fft_2d_call_count(), 1);

        let max_err = buf
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a.re - b.re).abs().max((a.im - b.im).abs()))
            .fold(0.0_f32, f32::max);
        assert!(
            max_err < 1e-4,
            "R2C mismatch for {width}x{height}: max_err={max_err}"
        );

        // Parseval energy identity.
        let spatial_energy: f64 = real.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let stride = width / 2 + 1;
        let mut freq_energy = 0.0_f64;
        for ky in 0..height {
            for kx in 0..stride {
                let w = if kx == 0 || kx == width / 2 { 1.0 } else { 2.0 };
                let c = buf[ky * stride + kx];
                freq_energy += w * ((c.re as f64) * (c.re as f64) + (c.im as f64) * (c.im as f64));
            }
        }
        freq_energy /= (width * height) as f64;
        let rel_err = (spatial_energy - freq_energy).abs() / spatial_energy.max(1e-12);
        assert!(
            rel_err < 1e-5,
            "Parseval mismatch for {width}x{height}: rel_err={rel_err}"
        );
    }
}

#[test]
fn test_r2c_c2r_roundtrip() {
    let mut rng = DeterministicRng::from_seed_u64(9876);
    for (width, height) in [(16usize, 16usize), (64, 32), (128, 128)] {
        let mut real = Vec::with_capacity(width * height);
        for _ in 0..(width * height) {
            real.push((rng.next_f64() * 4.0 - 2.0) as f32);
        }
        let mut buf = pack_real_grid(&real, width, height);
        let mut ws = R2cFft2dWorkspaceF32::new();
        ws.r2c_2d(&mut buf, width, height).unwrap();
        ws.c2r_2d(&mut buf, width, height).unwrap();
        assert_eq!(ws.fft_2d_call_count(), 2);

        let recovered = unpack_real_grid(&buf, width, height);
        let max_err = recovered
            .iter()
            .zip(real.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_err < 1e-5,
            "Roundtrip mismatch for {width}x{height}: max_err={max_err}"
        );
        let stride = width / 2 + 1;
        for row in 0..height {
            assert_eq!(buf[row * stride + width / 2], Complex32::ZERO);
        }
    }
}

#[test]
fn test_r2c_validation_rejects_invalid_inputs() {
    let mut ws = R2cFft2dWorkspaceF32::new();
    let mut buf = vec![Complex32::ZERO; 16 * 9];
    assert!(ws.r2c_2d(&mut buf, 2, 16).is_err());
    assert!(ws.r2c_2d(&mut buf, 12, 16).is_err());
    assert!(ws.r2c_2d(&mut buf, 16, 1).is_err());
    assert!(ws.r2c_2d(&mut buf, 16, 12).is_err());
    assert!(ws.r2c_2d(&mut buf[..10], 16, 16).is_err());
    assert!(ws.c2r_2d(&mut buf[..10], 16, 16).is_err());
    assert_eq!(ws.fft_2d_call_count(), 0);
}
