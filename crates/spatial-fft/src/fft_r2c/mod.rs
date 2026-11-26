//! Pure-Rust, zero-`unsafe`, strictly single-threaded in-place 2D Real-to-Complex (`R2C`)
//! and Complex-to-Real (`C2R`) Radix-2 FFT workspace in `f32`.
//!
//! Stores a `width x height` real spatial grid inside a single contiguous `Complex32` buffer of
//! length `height * (width / 2 + 1)`, reducing memory by ~2x vs complex `f32` grids and ~4x vs
//! `Complex64` grids (`~67.1 MB` for a `4096 x 4096` padded field).

use std::f32::consts::TAU;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use egraph_core::{CoreError, CoreResult};

#[cfg(test)]
mod tests;

/// Single-precision complex scalar for 1D and 2D Fast Fourier Transforms.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex32 {
    /// Real component.
    pub re: f32,
    /// Imaginary component.
    pub im: f32,
}

impl Complex32 {
    /// Additive identity `(0.0, 0.0)`.
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };

    /// Constructs a complex scalar `re + i * im`.
    #[inline]
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// Multiplies both components by a real `f32` factor.
    #[inline]
    pub fn scale(self, factor: f32) -> Self {
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
    pub fn norm_sq(self) -> f32 {
        self.re * self.re + self.im * self.im
    }
}

impl Add for Complex32 {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

impl AddAssign for Complex32 {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.re += other.re;
        self.im += other.im;
    }
}

impl Sub for Complex32 {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

impl SubAssign for Complex32 {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.re -= other.re;
        self.im -= other.im;
    }
}

impl Mul for Complex32 {
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

impl MulAssign for Complex32 {
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

#[derive(Debug, Clone, Default)]
struct R2cRowPlanF32 {
    /// Full real row width `N = 2 * M` (power of two >= 4).
    width: usize,
    /// Half-width complex FFT twiddles `exp(-i * 2*pi * k / M)` for `k in 0..M/2`.
    half_twiddles_fwd: Vec<Complex32>,
    /// Unpack twiddles `exp(-i * 2*pi * k / N)` for `k in 0..=M/2`.
    unpack_twiddles_fwd: Vec<Complex32>,
}

impl R2cRowPlanF32 {
    fn configure(&mut self, width: usize) {
        if self.width == width {
            return;
        }
        self.width = width;
        let m = width / 2;
        let half_m = m / 2;

        self.half_twiddles_fwd.clear();
        self.half_twiddles_fwd.reserve(half_m);
        let step_m = -TAU / (m as f32);
        for k in 0..half_m {
            let (s, c) = ((k as f32) * step_m).sin_cos();
            self.half_twiddles_fwd.push(Complex32::new(c, s));
        }

        self.unpack_twiddles_fwd.clear();
        self.unpack_twiddles_fwd.reserve(half_m + 1);
        let step_n = -TAU / (width as f32);
        for k in 0..=half_m {
            let (s, c) = ((k as f32) * step_n).sin_cos();
            self.unpack_twiddles_fwd.push(Complex32::new(c, s));
        }
    }
}

#[derive(Debug, Clone, Default)]
struct ColPlanF32 {
    height: usize,
    twiddles_fwd: Vec<Complex32>,
}

impl ColPlanF32 {
    fn configure(&mut self, height: usize) {
        if self.height == height {
            return;
        }
        self.height = height;
        let half = height / 2;
        self.twiddles_fwd.clear();
        self.twiddles_fwd.reserve(half);
        let step = -TAU / (height as f32);
        for k in 0..half {
            let (s, c) = ((k as f32) * step).sin_cos();
            self.twiddles_fwd.push(Complex32::new(c, s));
        }
    }
}

/// Reusable workspace for in-place `f32` 2D Real-to-Complex (`R2C`) and Complex-to-Real (`C2R`)
/// Radix-2 FFTs.
///
/// # Buffer Layout
/// Given `width` (power of two `>= 4`) and `height` (power of two `>= 2`), let
/// `stride = width / 2 + 1`. The slice `buf` must have length `height * stride`:
/// - **Spatial domain** (`r2c_2d` input / `c2r_2d` output): row `v in 0..height` stores real cell
///   `u in 0..width` at `buf[v * stride + (u >> 1)].re` when `u` is even, or `.im` when `u` is
///   odd. The trailing `buf[v * stride + width / 2]` element is ignored by `r2c_2d` and zeroed by
///   `c2r_2d`.
/// - **Frequency domain** (`r2c_2d` output / `c2r_2d` input): row `k_y in 0..height`, column
///   `k_x in 0..=width / 2` stores the half-complex DFT coefficient at `buf[k_y * stride + k_x]`.
#[derive(Debug, Clone, Default)]
pub struct R2cFft2dWorkspaceF32 {
    row_plan: R2cRowPlanF32,
    col_plan: ColPlanF32,
    col_buf: Vec<Complex32>,
    fft_2d_calls: u64,
}

impl R2cFft2dWorkspaceF32 {
    /// Creates an empty 2D `R2C`/`C2R` FFT workspace with zero call telemetry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the total number of [`Self::r2c_2d`] and [`Self::c2r_2d`] calls executed.
    #[inline]
    pub fn fft_2d_call_count(&self) -> u64 {
        self.fft_2d_calls
    }

    /// Resets the 2D FFT call counter to `0`.
    #[inline]
    pub fn reset_fft_2d_call_count(&mut self) {
        self.fft_2d_calls = 0;
    }

    /// Returns approximate heap bytes retained by cached plans and scratch column buffer.
    pub fn workspace_memory_bytes(&self) -> usize {
        use std::mem::size_of;
        self.row_plan.half_twiddles_fwd.capacity() * size_of::<Complex32>()
            + self.row_plan.unpack_twiddles_fwd.capacity() * size_of::<Complex32>()
            + self.col_plan.twiddles_fwd.capacity() * size_of::<Complex32>()
            + self.col_buf.capacity() * size_of::<Complex32>()
    }

    /// Computes an in-place unnormalized forward 2D `R2C` FFT over `buf`.
    ///
    /// Increments [`Self::fft_2d_call_count`] by `1` on valid inputs.
    pub fn r2c_2d(&mut self, buf: &mut [Complex32], width: usize, height: usize) -> CoreResult<()> {
        let stride = validate_r2c_dims(buf.len(), width, height, "R2cFft2dWorkspaceF32::r2c_2d")?;
        self.fft_2d_calls += 1;
        self.row_plan.configure(width);
        self.col_plan.configure(height);

        let m = width / 2;
        let half_m = m / 2;
        let half_tw = &self.row_plan.half_twiddles_fwd;
        let unpack_tw = &self.row_plan.unpack_twiddles_fwd;

        // 1. Transform each real row of length 2*M via an M-point complex FFT + Hermitian unpack.
        for row_idx in 0..height {
            let row_start = row_idx * stride;
            let row = &mut buf[row_start..row_start + stride];
            fft_1d_with_twiddles(&mut row[..m], half_tw, false);

            let z0 = row[0];
            row[0] = Complex32::new(z0.re + z0.im, 0.0);
            row[m] = Complex32::new(z0.re - z0.im, 0.0);

            for k in 1..half_m {
                let kp = m - k;
                let a = row[k];
                let b = row[kp].conj();
                let even = (a + b).scale(0.5);
                let diff = a - b;
                // O_k = (a - b) / (2i) = (0.5 * diff.im, -0.5 * diff.re)
                let odd = Complex32::new(0.5 * diff.im, -0.5 * diff.re);
                let t = unpack_tw[k] * odd;
                row[k] = even + t;
                row[kp] = (even - t).conj();
            }

            if half_m > 0 {
                row[half_m] = row[half_m].conj();
            }
        }

        // 2. Transform all k_x in 0..=M columns in a contiguous scratch buffer.
        if height > 1 {
            if self.col_buf.len() < height {
                self.col_buf.resize(height, Complex32::ZERO);
            }
            let col_tw = &self.col_plan.twiddles_fwd;
            let col_slice = &mut self.col_buf[..height];
            for col in 0..stride {
                for r in 0..height {
                    col_slice[r] = buf[r * stride + col];
                }
                fft_1d_with_twiddles(col_slice, col_tw, false);
                for r in 0..height {
                    buf[r * stride + col] = col_slice[r];
                }
            }
        }

        Ok(())
    }

    /// Computes an in-place normalized (`1 / (width * height)`) inverse 2D `C2R` FFT over `buf`.
    ///
    /// Increments [`Self::fft_2d_call_count`] by `1` on valid inputs.
    pub fn c2r_2d(&mut self, buf: &mut [Complex32], width: usize, height: usize) -> CoreResult<()> {
        let stride = validate_r2c_dims(buf.len(), width, height, "R2cFft2dWorkspaceF32::c2r_2d")?;
        self.fft_2d_calls += 1;
        self.row_plan.configure(width);
        self.col_plan.configure(height);

        // 1. Inverse-transform all k_x in 0..=M columns (scaled by 1 / height).
        if height > 1 {
            if self.col_buf.len() < height {
                self.col_buf.resize(height, Complex32::ZERO);
            }
            let col_tw = &self.col_plan.twiddles_fwd;
            let col_slice = &mut self.col_buf[..height];
            for col in 0..stride {
                for r in 0..height {
                    col_slice[r] = buf[r * stride + col];
                }
                fft_1d_with_twiddles(col_slice, col_tw, true);
                for r in 0..height {
                    buf[r * stride + col] = col_slice[r];
                }
            }
        }

        // 2. Repack Hermitian spectrum and run M-point inverse complex FFT (scaled by 1 / M = 2 / width).
        // Because E_k and O_k incorporate a 0.5 factor relative to X[k], the combined inverse scaling
        // across repack + M-point IFFT equals 1 / (2 * M) = 1 / width, matching 1 / (width * height).
        let m = width / 2;
        let half_m = m / 2;
        let half_tw = &self.row_plan.half_twiddles_fwd;
        let unpack_tw = &self.row_plan.unpack_twiddles_fwd;

        for row_idx in 0..height {
            let row_start = row_idx * stride;
            let row = &mut buf[row_start..row_start + stride];

            let x0 = row[0].re;
            let xm = row[m].re;
            row[0] = Complex32::new(0.5 * (x0 + xm), 0.5 * (x0 - xm));
            row[m] = Complex32::ZERO;

            for k in 1..half_m {
                let kp = m - k;
                let a = row[k];
                let b = row[kp].conj();
                let even = (a + b).scale(0.5);
                let odd = (unpack_tw[k].conj() * (a - b)).scale(0.5);
                let i_odd = Complex32::new(-odd.im, odd.re);
                row[k] = even + i_odd;
                row[kp] = (even - i_odd).conj();
            }

            if half_m > 0 {
                row[half_m] = row[half_m].conj();
            }

            fft_1d_with_twiddles(&mut row[..m], half_tw, true);
        }

        Ok(())
    }
}

fn validate_r2c_dims(
    buf_len: usize,
    width: usize,
    height: usize,
    context: &'static str,
) -> CoreResult<usize> {
    if width < 4 || !width.is_power_of_two() {
        return Err(CoreError::invalid_input(format!(
            "2D R2C FFT width must be a power of two >= 4, got {width}"
        )));
    }
    if height < 2 || !height.is_power_of_two() {
        return Err(CoreError::invalid_input(format!(
            "2D R2C FFT height must be a power of two >= 2, got {height}"
        )));
    }
    let stride = width / 2 + 1;
    let expected = height
        .checked_mul(stride)
        .ok_or_else(|| CoreError::invalid_input("2D R2C FFT dimensions overflow usize"))?;
    if buf_len != expected {
        return Err(CoreError::dimension_mismatch(expected, buf_len, context));
    }
    Ok(stride)
}

fn fft_1d_with_twiddles(data: &mut [Complex32], twiddles_fwd: &[Complex32], inverse: bool) {
    let n = data.len();
    if n <= 1 {
        return;
    }

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
        let inv_n = 1.0 / (n as f32);
        for val in data.iter_mut() {
            *val = val.scale(inv_n);
        }
    }
}
