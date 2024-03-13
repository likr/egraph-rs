pub mod euclidean;
pub mod euclidean2d;
pub mod torus2d;

use crate::DrawingValue;
use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};
pub trait Difference:
    Sized + Add<Self> + Sub<Self> + Mul<Self::S, Output = Self> + Div<Self::S>
{
    type S: DrawingValue;

    fn norm(&self) -> Self::S;
}

pub trait Metric:
    Sized
    + Sub<Output = Self::D>
    + Add<Self::D>
    + Sub<Self::D>
    + AddAssign<Self::D>
    + SubAssign<Self::D>
{
    type D: Difference;
}
