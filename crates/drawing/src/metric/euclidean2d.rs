use crate::{
    metric::{Difference, Metric},
    DrawingValue,
};
use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

#[derive(Copy, Clone, Debug, Default)]
pub struct MetricEuclidean2d<S>(pub S, pub S);

impl<S> Add for MetricEuclidean2d<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        MetricEuclidean2d(self.0 + other.0, self.1 + other.1)
    }
}

impl<S> AddAssign for MetricEuclidean2d<S>
where
    S: DrawingValue,
{
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
        self.1 += other.1;
    }
}

impl<S> Sub for MetricEuclidean2d<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        MetricEuclidean2d(self.0 - other.0, self.1 - other.1)
    }
}

impl<S> SubAssign for MetricEuclidean2d<S>
where
    S: DrawingValue,
{
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
        self.1 -= other.1;
    }
}

impl<S> Mul<S> for MetricEuclidean2d<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn mul(self, other: S) -> Self {
        MetricEuclidean2d(self.0 * other, self.1 * other)
    }
}

impl<S> Div<S> for MetricEuclidean2d<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn div(self, other: S) -> Self {
        MetricEuclidean2d(self.0 / other, self.1 / other)
    }
}

impl<S> Difference for MetricEuclidean2d<S>
where
    S: DrawingValue,
{
    type S = S;
    fn norm(&self) -> Self::S {
        self.0.hypot(self.1)
    }
}

impl<S> Metric for MetricEuclidean2d<S>
where
    S: DrawingValue,
{
    type D = MetricEuclidean2d<S>;
}
