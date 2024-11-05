pub mod metric_euclidean;
pub mod metric_euclidean_2d;
pub mod metric_hyperbolic_2d;
pub mod metric_spherical_2d;
pub mod metric_torus2d;

use crate::DrawingValue;
use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};
pub trait Delta:
    Sized + Add<Self> + Sub<Self> + Mul<Self::S, Output = Self> + Div<Self::S> + Clone
{
    type S: DrawingValue;

    fn norm(&self) -> Self::S;
}

pub trait Metric: Sized + AddAssign<Self::D> + SubAssign<Self::D> {
    type D: Delta;
}
