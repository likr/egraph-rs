use super::*;

fn brute_force_dft_1d(input: &[Complex64], inverse: bool) -> Vec<Complex64> {
    let n = input.len();
    let sign = if inverse { 1.0 } else { -1.0 };
    let scale = if inverse { 1.0 / n as f64 } else { 1.0 };
    let mut out = vec![Complex64::ZERO; n];
    for (k, slot) in out.iter_mut().enumerate() {
        let mut sum = Complex64::ZERO;
        for (j, &val) in input.iter().enumerate() {
            let angle = sign * TAU * (k * j) as f64 / n as f64;
            let (s, c) = angle.sin_cos();
            sum += val * Complex64::new(c, s);
        }
        *slot = sum.scale(scale);
    }
    out
}

#[test]
fn test_fft_1d_matches_dft_and_roundtrips() {
    let mut ws = Fft2dWorkspace::new();
    for &n in &[1usize, 2, 4, 8, 16, 64] {
        let orig: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.37).sin(), (i as f64 * 0.21).cos()))
            .collect();
        let expected_fwd = brute_force_dft_1d(&orig, false);

        let mut buf = orig.clone();
        ws.fft_1d(&mut buf, false).unwrap();
        for (a, b) in buf.iter().zip(expected_fwd.iter()) {
            assert!((a.re - b.re).abs() < 1e-11);
            assert!((a.im - b.im).abs() < 1e-11);
        }

        ws.fft_1d(&mut buf, true).unwrap();
        for (a, b) in buf.iter().zip(orig.iter()) {
            assert!((a.re - b.re).abs() < 1e-12);
            assert!((a.im - b.im).abs() < 1e-12);
        }
    }
}

#[test]
fn test_fft_2d_convolve_and_telemetry_zero_alloc_reuse() {
    let mut ws = Fft2dWorkspace::new();
    let w = 8;
    let h = 16;

    let signal: Vec<Complex64> = (0..w * h)
        .map(|i| Complex64::new(((i * 7 % 19) as f64) * 0.1, ((i * 3 % 11) as f64) * -0.05))
        .collect();
    let kernel: Vec<Complex64> = (0..w * h)
        .map(|i| Complex64::new(1.0 / (1.0 + (i % 9) as f64), 0.0))
        .collect();

    // Brute-force 2D circular convolution.
    let mut expected = vec![Complex64::ZERO; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = Complex64::ZERO;
            for ky in 0..h {
                let sy = (y + h - ky) % h;
                for kx in 0..w {
                    let sx = (x + w - kx) % w;
                    acc += signal[sy * w + sx] * kernel[ky * w + kx];
                }
            }
            expected[y * w + x] = acc;
        }
    }

    let mut kernel_fft = kernel.clone();
    ws.fft_2d(&mut kernel_fft, w, h, false).unwrap();
    let mut conv = signal.clone();
    ws.convolve_2d_in_place(&mut conv, &kernel_fft, w, h)
        .unwrap();
    assert_eq!(ws.fft_2d_call_count(), 3);

    for (a, b) in conv.iter().zip(expected.iter()) {
        assert!((a.re - b.re).abs() < 1e-10);
        assert!((a.im - b.im).abs() < 1e-10);
    }

    // Subsequent calls at the same dimensions must reuse buffers without reallocation.
    let col_cap_before = ws.col_buf.capacity();
    let plan0_cap_before = ws.plans[0].twiddles_fwd.capacity();
    let plan1_cap_before = ws.plans[1].twiddles_fwd.capacity();
    ws.reset_fft_2d_call_count();
    assert_eq!(ws.fft_2d_call_count(), 0);

    let mut conv2 = signal;
    ws.convolve_2d_in_place(&mut conv2, &kernel_fft, w, h)
        .unwrap();
    assert_eq!(ws.fft_2d_call_count(), 2);
    assert_eq!(ws.col_buf.capacity(), col_cap_before);
    assert_eq!(ws.plans[0].twiddles_fwd.capacity(), plan0_cap_before);
    assert_eq!(ws.plans[1].twiddles_fwd.capacity(), plan1_cap_before);
}

#[test]
fn test_fft_rejects_invalid_dimensions() {
    let mut ws = Fft2dWorkspace::new();
    assert!(matches!(
        ws.fft_1d(&mut [], false),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        ws.fft_1d(&mut [Complex64::ZERO; 6], false),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        ws.fft_2d(&mut [Complex64::ZERO; 12], 3, 4, false),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        ws.fft_2d(&mut [Complex64::ZERO; 8], 4, 4, false),
        Err(CoreError::DimensionMismatch { .. })
    ));
}

#[test]
fn test_fft_2d_plan_slot_shrink_and_growth_roundtrip() {
    let mut ws = Fft2dWorkspace::new();

    let run_roundtrip = |ws: &mut Fft2dWorkspace, w: usize, h: usize| {
        let n = w * h;
        let input: Vec<Complex64> = (0..n)
            .map(|i| {
                Complex64::new(
                    ((i * 13 % 29) as f64 - 14.0) * 0.05,
                    ((i * 7 % 23) as f64 - 11.0) * 0.03,
                )
            })
            .collect();
        let mut buf = input.clone();
        ws.fft_2d(&mut buf, w, h, false).unwrap();
        ws.fft_2d(&mut buf, w, h, true).unwrap();
        for (idx, (a, b)) in buf.iter().zip(input.iter()).enumerate() {
            assert!(
                (a.re - b.re).abs() < 1e-11 && (a.im - b.im).abs() < 1e-11,
                "roundtrip mismatch at ({w}x{h}) idx={idx}: got=({}, {}), expected=({}, {})",
                a.re,
                a.im,
                b.re,
                b.im
            );
        }
    };

    // Repros for plan-slot eviction when width hits slot 0 and height misses into slot 0.
    run_roundtrip(&mut ws, 16, 8);
    run_roundtrip(&mut ws, 16, 4);
    run_roundtrip(&mut ws, 8, 16);
    run_roundtrip(&mut ws, 8, 32);
    run_roundtrip(&mut ws, 32, 64);
    run_roundtrip(&mut ws, 16, 64);

    // Large FitField2d shrink sequence (512x256 -> 512x128).
    let mut big = vec![Complex64::new(0.25, -0.5); 512 * 256];
    ws.fft_2d(&mut big, 512, 256, false).unwrap();
    let mut small = vec![Complex64::new(0.125, 0.25); 512 * 128];
    let small_orig = small.clone();
    ws.fft_2d(&mut small, 512, 128, false).unwrap();
    ws.fft_2d(&mut small, 512, 128, true).unwrap();
    for (a, b) in small.iter().zip(small_orig.iter()) {
        assert!((a.re - b.re).abs() < 1e-10 && (a.im - b.im).abs() < 1e-10);
    }
}
