//! Reusable in-place Radix-2 Cooley-Tukey 1D/2D FFT workspace in `f64` with call telemetry.

use std::f64::consts::TAU;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use egraph_core::{CoreError, CoreResult};

/// Double-precision complex scalar for 1D and 2D Fast Fourier Transforms.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    /// Additive identity `(0.0, 0.0)`.
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };

    /// Constructs a complex scalar `re + i * im`.
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Multiplies both components by a real factor.
    #[inline]
    pub fn scale(self, factor: f64) -> Self {
        Self {
            re: self.re * factor,
            im: self.im * factor,
        }
    }

    /// Returns the complex conjugate `re - i * im`.
    #[inline]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Returns `re^2 + im^2`.
    #[inline]
    pub fn norm_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

impl Add for Complex64 {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

impl AddAssign for Complex64 {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.re += other.re;
        self.im += other.im;
    }
}

impl Sub for Complex64 {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

impl SubAssign for Complex64 {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.re -= other.re;
        self.im -= other.im;
    }
}

impl Mul for Complex64 {
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

impl MulAssign for Complex64 {
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

#[derive(Debug, Clone, Default)]
struct Plan1d {
    len: usize,
    /// Precomputed forward unit roots `exp(-i * 2*pi * k / len)` for `k in 0..len/2`.
    twiddles_fwd: Vec<Complex64>,
}

impl Plan1d {
    fn configure(&mut self, len: usize) {
        if self.len == len {
            return;
        }
        self.len = len;
        let half = len / 2;
        self.twiddles_fwd.clear();
        self.twiddles_fwd.reserve(half);
        let step = -TAU / (len as f64);
        for k in 0..half {
            let (s, c) = ((k as f64) * step).sin_cos();
            self.twiddles_fwd.push(Complex64::new(c, s));
        }
    }
}

/// Reusable workspace for zero-allocation repeated 1D and 2D Radix-2 FFTs in `f64`.
///
/// Caches 1D twiddle tables for up to two power-of-two axes (`width` and `height`) and a
/// reusable column scratch buffer, so repeated iterations at fixed grid dimensions perform
/// zero heap allocations and zero trigonometric calls inside butterflies.
#[derive(Debug, Clone, Default)]
pub struct Fft2dWorkspace {
    col_buf: Vec<Complex64>,
    plans: [Plan1d; 2],
    next_plan_slot: usize,
    fft_2d_calls: u64,
}

impl Fft2dWorkspace {
    /// Creates an empty FFT workspace with zero telemetry count.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the total number of [`Fft2dWorkspace::fft_2d`] calls executed since creation or
    /// the last [`Fft2dWorkspace::reset_fft_2d_call_count`].
    #[inline]
    pub fn fft_2d_call_count(&self) -> u64 {
        self.fft_2d_calls
    }

    /// Resets the `fft_2d` call counter to `0`.
    #[inline]
    pub fn reset_fft_2d_call_count(&mut self) {
        self.fft_2d_calls = 0;
    }

    /// Computes an in-place 1D Radix-2 FFT (or inverse FFT when `inverse == true`).
    ///
    /// Returns [`CoreError::InvalidInput`] if `data.len()` is zero or not a power of two.
    pub fn fft_1d(&mut self, data: &mut [Complex64], inverse: bool) -> CoreResult<()> {
        let n = data.len();
        if n == 0 || !n.is_power_of_two() {
            return Err(CoreError::invalid_input(format!(
                "FFT length must be a non-zero power of two, got {n}"
            )));
        }
        let slot = self.ensure_plan(n);
        fft_1d_with_twiddles(data, &self.plans[slot].twiddles_fwd, inverse);
        Ok(())
    }

    /// Computes an in-place 2D Radix-2 FFT over a row-major `width x height` complex grid.
    ///
    /// Increments [`Fft2dWorkspace::fft_2d_call_count`] by `1` on valid inputs.
    pub fn fft_2d(
        &mut self,
        data: &mut [Complex64],
        width: usize,
        height: usize,
        inverse: bool,
    ) -> CoreResult<()> {
        if width == 0 || !width.is_power_of_two() {
            return Err(CoreError::invalid_input(format!(
                "2D FFT width must be a non-zero power of two, got {width}"
            )));
        }
        if height == 0 || !height.is_power_of_two() {
            return Err(CoreError::invalid_input(format!(
                "2D FFT height must be a non-zero power of two, got {height}"
            )));
        }
        let expected = width
            .checked_mul(height)
            .ok_or_else(|| CoreError::invalid_input("2D FFT dimensions overflow usize"))?;
        if data.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                data.len(),
                "Fft2dWorkspace::fft_2d",
            ));
        }

        self.fft_2d_calls += 1;
        let (row_slot, col_slot) = self.ensure_plan_pair(width, height);

        // Transform rows using cached row twiddles.
        if width > 1 {
            let row_twiddles = &self.plans[row_slot].twiddles_fwd;
            for row in 0..height {
                let start = row * width;
                fft_1d_with_twiddles(&mut data[start..start + width], row_twiddles, inverse);
            }
        }

        // Transform columns into reusable scratch buffer to avoid strided butterfly cache misses.
        if height > 1 {
            if self.col_buf.len() < height {
                self.col_buf.resize(height, Complex64::ZERO);
            }
            let col_twiddles = &self.plans[col_slot].twiddles_fwd;
            let col_slice = &mut self.col_buf[..height];
            for col in 0..width {
                for row in 0..height {
                    col_slice[row] = data[row * width + col];
                }
                fft_1d_with_twiddles(col_slice, col_twiddles, inverse);
                for row in 0..height {
                    data[row * width + col] = col_slice[row];
                }
            }
        }

        Ok(())
    }

    /// Performs in-place 2D frequency-domain convolution `IFFT(FFT(signal) * kernel_fft)`.
    pub fn convolve_2d_in_place(
        &mut self,
        signal: &mut [Complex64],
        kernel_fft: &[Complex64],
        width: usize,
        height: usize,
    ) -> CoreResult<()> {
        let expected = width
            .checked_mul(height)
            .ok_or_else(|| CoreError::invalid_input("2D convolution dimensions overflow usize"))?;
        if kernel_fft.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                kernel_fft.len(),
                "Fft2dWorkspace::convolve_2d_in_place",
            ));
        }
        self.fft_2d(signal, width, height, false)?;
        for (s, &k) in signal.iter_mut().zip(kernel_fft.iter()) {
            *s *= k;
        }
        self.fft_2d(signal, width, height, true)
    }

    fn ensure_plan_pair(&mut self, width: usize, height: usize) -> (usize, usize) {
        if width == height {
            let slot = self.ensure_plan(width);
            return (slot, slot);
        }
        if self.plans[0].len == width {
            if self.plans[1].len != height {
                self.plans[1].configure(height);
            }
            self.next_plan_slot = 0;
            return (0, 1);
        }
        if self.plans[1].len == width {
            if self.plans[0].len != height {
                self.plans[0].configure(height);
            }
            self.next_plan_slot = 1;
            return (1, 0);
        }
        if self.plans[0].len == height {
            self.plans[1].configure(width);
            self.next_plan_slot = 0;
            return (1, 0);
        }
        if self.plans[1].len == height {
            self.plans[0].configure(width);
            self.next_plan_slot = 1;
            return (0, 1);
        }
        self.plans[0].configure(width);
        self.plans[1].configure(height);
        self.next_plan_slot = 0;
        (0, 1)
    }

    fn ensure_plan(&mut self, len: usize) -> usize {
        if self.plans[0].len == len {
            return 0;
        }
        if self.plans[1].len == len {
            return 1;
        }
        let slot = self.next_plan_slot;
        self.next_plan_slot = 1 - slot;
        self.plans[slot].configure(len);
        slot
    }
}

fn fft_1d_with_twiddles(data: &mut [Complex64], twiddles_fwd: &[Complex64], inverse: bool) {
    let n = data.len();
    if n <= 1 {
        return;
    }

    // In-place bit-reversal permutation.
    let mut j = 0usize;
    for i in 0..n {
        if i < j {
            data.swap(i, j);
        }
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
    }

    // Iterative Cooley-Tukey butterflies using cached unit roots to prevent phase drift.
    let half_n = n >> 1;
    let mut len = 2usize;
    while len <= n {
        let half_len = len >> 1;
        let stride = half_n / half_len;
        let mut base = 0usize;
        while base < n {
            for k in 0..half_len {
                let tw = twiddles_fwd[k * stride];
                let w = if inverse { tw.conj() } else { tw };
                let u = data[base + k];
                let v = data[base + k + half_len] * w;
                data[base + k] = u + v;
                data[base + k + half_len] = u - v;
            }
            base += len;
        }
        len <<= 1;
    }

    if inverse {
        let inv_n = 1.0 / (n as f64);
        for val in data.iter_mut() {
            *val = val.scale(inv_n);
        }
    }
}

#[cfg(test)]
mod tests {
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
}
