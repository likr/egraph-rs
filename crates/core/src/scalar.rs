//! Floating-point scalar abstraction (`f32` and `f64`) without external trait crates.

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}

    pub trait WeightSealed {}
    impl WeightSealed for () {}
    impl WeightSealed for f32 {}
    impl WeightSealed for f64 {}
}

/// Sealed edge-weight trait implemented strictly for `()`, `f32`, and `f64`.
///
/// Accepting `W: EdgeWeight` on CSR graph algorithms allows unweighted [`crate::CsrGraph<()>`]
/// inputs alongside weighted [`crate::CsrGraph<f32>`] and [`crate::CsrGraph<f64>`] without
/// duplicating `*_unweighted` entry points. Method names avoid collision with [`Scalar`].
pub trait EdgeWeight: private::WeightSealed + Copy + Debug + Send + Sync + 'static {
    /// Converts the edge weight to `f64` (`1.0` for `()`).
    fn weight_to_f64(self) -> f64;

    /// Returns `true` if the edge weight is finite (`true` for `()`).
    fn weight_is_finite(self) -> bool;
}

impl EdgeWeight for () {
    #[inline]
    fn weight_to_f64(self) -> f64 {
        1.0
    }

    #[inline]
    fn weight_is_finite(self) -> bool {
        true
    }
}

impl EdgeWeight for f32 {
    #[inline]
    fn weight_to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn weight_is_finite(self) -> bool {
        f32::is_finite(self)
    }
}

impl EdgeWeight for f64 {
    #[inline]
    fn weight_to_f64(self) -> f64 {
        self
    }

    #[inline]
    fn weight_is_finite(self) -> bool {
        f64::is_finite(self)
    }
}

/// Sealed floating-point scalar trait implemented strictly for `f32` and `f64`.
///
/// Keeping this trait dependency-free avoids pulling `num-traits` or `ndarray` into
/// `egraph-core`, while giving solvers uniform access to IEEE-754 operations and `f64`
/// accumulation helpers.
pub trait Scalar:
    private::Sealed
    + EdgeWeight
    + Copy
    + Clone
    + Debug
    + Default
    + PartialEq
    + PartialOrd
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const PI: Self;
    const TAU: Self;
    const EPSILON: Self;
    const INFINITY: Self;
    const NEG_INFINITY: Self;
    const MIN_POSITIVE: Self;

    fn from_f64(v: f64) -> Self;
    fn from_f32(v: f32) -> Self;
    fn from_usize(v: usize) -> Self;
    fn to_f64(self) -> f64;
    fn to_f32(self) -> f32;

    fn is_finite(self) -> bool;
    fn is_nan(self) -> bool;
    fn abs(self) -> Self;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn clamp(self, min: Self, max: Self) -> Self;
    fn sqrt(self) -> Self;
    fn hypot(self, other: Self) -> Self;
    fn ln(self) -> Self;
    fn exp(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn sin_cos(self) -> (Self, Self);
    fn tan(self) -> Self;
    fn atan2(self, other: Self) -> Self;
    fn acos(self) -> Self;
    fn asin(self) -> Self;
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;
    fn acosh(self) -> Self;
    fn atanh(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn fract(self) -> Self;
    fn signum(self) -> Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn rem_euclid(self, rhs: Self) -> Self;
}

impl Scalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const PI: Self = std::f32::consts::PI;
    const TAU: Self = std::f32::consts::TAU;
    const EPSILON: Self = f32::EPSILON;
    const INFINITY: Self = f32::INFINITY;
    const NEG_INFINITY: Self = f32::NEG_INFINITY;
    const MIN_POSITIVE: Self = f32::MIN_POSITIVE;

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn from_f32(v: f32) -> Self {
        v
    }
    #[inline]
    fn from_usize(v: usize) -> Self {
        v as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline]
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
    #[inline]
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }
    #[inline]
    fn abs(self) -> Self {
        f32::abs(self)
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        f32::min(self, other)
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        f32::max(self, other)
    }
    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        f32::clamp(self, min, max)
    }
    #[inline]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        f32::hypot(self, other)
    }
    #[inline]
    fn ln(self) -> Self {
        f32::ln(self)
    }
    #[inline]
    fn exp(self) -> Self {
        f32::exp(self)
    }
    #[inline]
    fn powf(self, n: Self) -> Self {
        f32::powf(self, n)
    }
    #[inline]
    fn sin(self) -> Self {
        f32::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        f32::cos(self)
    }
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        f32::sin_cos(self)
    }
    #[inline]
    fn tan(self) -> Self {
        f32::tan(self)
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        f32::atan2(self, other)
    }
    #[inline]
    fn acos(self) -> Self {
        f32::acos(self)
    }
    #[inline]
    fn asin(self) -> Self {
        f32::asin(self)
    }
    #[inline]
    fn sinh(self) -> Self {
        f32::sinh(self)
    }
    #[inline]
    fn cosh(self) -> Self {
        f32::cosh(self)
    }
    #[inline]
    fn tanh(self) -> Self {
        f32::tanh(self)
    }
    #[inline]
    fn acosh(self) -> Self {
        f32::acosh(self)
    }
    #[inline]
    fn atanh(self) -> Self {
        f32::atanh(self)
    }
    #[inline]
    fn floor(self) -> Self {
        f32::floor(self)
    }
    #[inline]
    fn ceil(self) -> Self {
        f32::ceil(self)
    }
    #[inline]
    fn round(self) -> Self {
        f32::round(self)
    }
    #[inline]
    fn fract(self) -> Self {
        f32::fract(self)
    }
    #[inline]
    fn signum(self) -> Self {
        f32::signum(self)
    }
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f32::mul_add(self, a, b)
    }
    #[inline]
    fn rem_euclid(self, rhs: Self) -> Self {
        f32::rem_euclid(self, rhs)
    }
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const PI: Self = std::f64::consts::PI;
    const TAU: Self = std::f64::consts::TAU;
    const EPSILON: Self = f64::EPSILON;
    const INFINITY: Self = f64::INFINITY;
    const NEG_INFINITY: Self = f64::NEG_INFINITY;
    const MIN_POSITIVE: Self = f64::MIN_POSITIVE;

    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn from_f32(v: f32) -> Self {
        v as f64
    }
    #[inline]
    fn from_usize(v: usize) -> Self {
        v as f64
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
    #[inline]
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }
    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        f64::clamp(self, min, max)
    }
    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        f64::hypot(self, other)
    }
    #[inline]
    fn ln(self) -> Self {
        f64::ln(self)
    }
    #[inline]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline]
    fn powf(self, n: Self) -> Self {
        f64::powf(self, n)
    }
    #[inline]
    fn sin(self) -> Self {
        f64::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        f64::cos(self)
    }
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        f64::sin_cos(self)
    }
    #[inline]
    fn tan(self) -> Self {
        f64::tan(self)
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        f64::atan2(self, other)
    }
    #[inline]
    fn acos(self) -> Self {
        f64::acos(self)
    }
    #[inline]
    fn asin(self) -> Self {
        f64::asin(self)
    }
    #[inline]
    fn sinh(self) -> Self {
        f64::sinh(self)
    }
    #[inline]
    fn cosh(self) -> Self {
        f64::cosh(self)
    }
    #[inline]
    fn tanh(self) -> Self {
        f64::tanh(self)
    }
    #[inline]
    fn acosh(self) -> Self {
        f64::acosh(self)
    }
    #[inline]
    fn atanh(self) -> Self {
        f64::atanh(self)
    }
    #[inline]
    fn floor(self) -> Self {
        f64::floor(self)
    }
    #[inline]
    fn ceil(self) -> Self {
        f64::ceil(self)
    }
    #[inline]
    fn round(self) -> Self {
        f64::round(self)
    }
    #[inline]
    fn fract(self) -> Self {
        f64::fract(self)
    }
    #[inline]
    fn signum(self) -> Self {
        f64::signum(self)
    }
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f64::mul_add(self, a, b)
    }
    #[inline]
    fn rem_euclid(self, rhs: Self) -> Self {
        f64::rem_euclid(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sum_and_hypot<S: Scalar>(a: S, b: S) -> (S, S) {
        (a + b, a.hypot(b))
    }

    #[test]
    fn test_scalar_f32_and_f64() {
        let (s32, h32) = sum_and_hypot(3.0_f32, 4.0_f32);
        assert_eq!(s32, 7.0_f32);
        assert!((h32 - 5.0_f32).abs() < 1e-6);
        assert_eq!(f32::from_f64(2.5).to_f64(), 2.5);

        let (s64, h64) = sum_and_hypot(3.0_f64, 4.0_f64);
        assert_eq!(s64, 7.0_f64);
        assert!((h64 - 5.0_f64).abs() < 1e-12);
        assert_eq!(f64::from_f32(1.25).to_f32(), 1.25);
    }

    fn eval_weight<W: EdgeWeight>(w: W) -> (f64, bool) {
        (w.weight_to_f64(), w.weight_is_finite())
    }

    fn eval_scalar_as_weight<W: Scalar>(w: W) -> f64 {
        eval_weight(w).0
    }

    #[test]
    fn test_edge_weight_unit_f32_f64_and_scalar_subtrait() {
        assert_eq!(eval_weight(()), (1.0, true));
        assert_eq!(eval_weight(2.5_f32), (2.5, true));
        assert!(!eval_weight(f32::NAN).1);
        assert_eq!(eval_weight(3.25_f64), (3.25, true));
        assert!(!eval_weight(f64::INFINITY).1);
        assert_eq!(eval_scalar_as_weight(1.5_f32), 1.5);
        assert_eq!(eval_scalar_as_weight(4.5_f64), 4.5);
    }
}
