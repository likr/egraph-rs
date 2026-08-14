use num_traits::Float;
use petgraph_drawing::DrawingValue;
use std::ops::{Add, Mul, Sub};

/// Lightweight Complex number implementation for FFT operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex<S> {
    pub re: S,
    pub im: S,
}

impl<S> Complex<S>
where
    S: DrawingValue + Float + Default,
{
    #[inline]
    pub fn new(re: S, im: S) -> Self {
        Self { re, im }
    }

    #[inline]
    pub fn zero() -> Self {
        Self {
            re: S::zero(),
            im: S::zero(),
        }
    }

    #[inline]
    pub fn scale(self, factor: S) -> Self {
        Self {
            re: self.re * factor,
            im: self.im * factor,
        }
    }
}

impl<S> Add for Complex<S>
where
    S: DrawingValue + Float + Default,
{
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

impl<S> Sub for Complex<S>
where
    S: DrawingValue + Float + Default,
{
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

impl<S> Mul for Complex<S>
where
    S: DrawingValue + Float + Default,
{
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

/// Computes in-place 1D Radix-2 Cooley-Tukey Fast Fourier Transform.
/// Length `n` of `data` MUST be a power of 2.
pub fn fft_1d<S>(data: &mut [Complex<S>], inverse: bool)
where
    S: DrawingValue + Float + Default,
{
    let n = data.len();
    if n <= 1 {
        return;
    }
    assert!(n.is_power_of_two(), "FFT length must be a power of 2");

    // Bit reversal permutation
    let mut j = 0;
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

    // Cooley-Tukey iterative butterflies
    let sign = if inverse {
        S::from_f64(1.0).unwrap()
    } else {
        S::from_f64(-1.0).unwrap()
    };
    let two_pi = S::from_f64(std::f64::consts::TAU).unwrap();

    let mut len = 2;
    while len <= n {
        let half_len = len / 2;
        let angle = sign * two_pi / S::from_usize(len).unwrap();
        let w_unit = Complex::new(angle.cos(), angle.sin());

        let mut i = 0;
        while i < n {
            let mut w = Complex::new(S::one(), S::zero());
            for k in 0..half_len {
                let u = data[i + k];
                let v = data[i + k + half_len] * w;
                data[i + k] = u + v;
                data[i + k + half_len] = u - v;
                w = w * w_unit;
            }
            i += len;
        }
        len <<= 1;
    }

    if inverse {
        let n_inv = S::one() / S::from_usize(n).unwrap();
        for val in data.iter_mut() {
            *val = val.scale(n_inv);
        }
    }
}

/// Computes in-place 2D Radix-2 Fast Fourier Transform of size `width x height`.
pub fn fft_2d<S>(data: &mut [Complex<S>], width: usize, height: usize, inverse: bool)
where
    S: DrawingValue + Float + Default,
{
    assert_eq!(data.len(), width * height);
    assert!(width.is_power_of_two() && height.is_power_of_two());

    // Row-wise 1D FFT
    for row in 0..height {
        let start = row * width;
        let end = start + width;
        fft_1d(&mut data[start..end], inverse);
    }

    // Column-wise 1D FFT
    let mut col_buf = vec![Complex::zero(); height];
    for col in 0..width {
        for row in 0..height {
            col_buf[row] = data[row * width + col];
        }
        fft_1d(&mut col_buf, inverse);
        for row in 0..height {
            data[row * width + col] = col_buf[row];
        }
    }
}

/// Performs 2D linear/circular convolution in frequency domain using 2D FFT.
pub fn convolve_2d<S>(
    signal: &[Complex<S>],
    kernel_fft: &[Complex<S>],
    width: usize,
    height: usize,
) -> Vec<Complex<S>>
where
    S: DrawingValue + Float + Default,
{
    let mut signal_fft = signal.to_vec();
    fft_2d(&mut signal_fft, width, height, false);

    for (s, k) in signal_fft.iter_mut().zip(kernel_fft.iter()) {
        *s = *s * *k;
    }

    fft_2d(&mut signal_fft, width, height, true);
    signal_fft
}
