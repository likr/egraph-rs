use crate::DrawingValue;
use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};
pub trait Difference: Sized + Add<Self> + Sub<Self> + Mul<Self::S> + Div<Self::S> {
    type S: DrawingValue;

    fn norm(&self) -> Self::S;
}

pub trait Metric: Add<Self::D> + Sub<Self::D> + AddAssign<Self::D> + SubAssign<Self::D> {
    type D: Difference;
}
