use crate::{Delta, DrawingValue, Metric};
use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

use super::MetricCartesian;

/// Represents the difference vector between two points in 2D Euclidean space.
///
/// This struct implements the `Delta` trait for 2D Euclidean space.
/// It stores the x and y components of the vector.
///
/// # Type Parameters
///
/// * `S`: The scalar type used for coordinate values (must implement `DrawingValue`).
#[derive(Copy, Clone, Debug, Default)]
pub struct DeltaEuclidean2d<S>(pub S, pub S);

impl<S> Add for DeltaEuclidean2d<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        DeltaEuclidean2d(self.0 + other.0, self.1 + other.1)
    }
}

impl<S> Sub for DeltaEuclidean2d<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        DeltaEuclidean2d(self.0 - other.0, self.1 - other.1)
    }
}

impl<S> Mul<S> for DeltaEuclidean2d<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn mul(self, other: S) -> Self {
        DeltaEuclidean2d(self.0 * other, self.1 * other)
    }
}

impl<S> Div<S> for DeltaEuclidean2d<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn div(self, other: S) -> Self {
        DeltaEuclidean2d(self.0 / other, self.1 / other)
    }
}

impl<S> Delta for DeltaEuclidean2d<S>
where
    S: DrawingValue,
{
    type S = S;
    fn norm(&self) -> Self::S {
        self.0.hypot(self.1)
    }
}

/// Represents a point in 2D Euclidean space.
///
/// This struct implements the `Metric` trait for 2D Euclidean space.
/// It stores the x and y coordinates of the point.
///
/// # Type Parameters
///
/// * `S`: The scalar type used for coordinate values (must implement `DrawingValue`).
#[derive(Copy, Clone, Debug, Default)]
pub struct MetricEuclidean2d<S>(pub S, pub S);

impl<S> AddAssign<DeltaEuclidean2d<S>> for MetricEuclidean2d<S>
where
    S: DrawingValue,
{
    fn add_assign(&mut self, other: DeltaEuclidean2d<S>) {
        self.0 += other.0;
        self.1 += other.1;
    }
}

impl<S> SubAssign<DeltaEuclidean2d<S>> for MetricEuclidean2d<S>
where
    S: DrawingValue,
{
    fn sub_assign(&mut self, other: DeltaEuclidean2d<S>) {
        self.0 -= other.0;
        self.1 -= other.1;
    }
}

impl<S> Metric for MetricEuclidean2d<S>
where
    S: DrawingValue,
{
    type D = DeltaEuclidean2d<S>;
}

impl<'b, S> Sub<&'b MetricEuclidean2d<S>> for &MetricEuclidean2d<S>
where
    S: DrawingValue,
{
    type Output = DeltaEuclidean2d<S>;

    fn sub(self, other: &'b MetricEuclidean2d<S>) -> DeltaEuclidean2d<S> {
        DeltaEuclidean2d(self.0 - other.0, self.1 - other.1)
    }
}

impl<S> MetricCartesian for MetricEuclidean2d<S>
where
    S: DrawingValue,
{
    fn nth(&self, n: usize) -> &S {
        if n == 0 {
            &self.0
        } else if n == 1 {
            &self.1
        } else {
            unreachable!("index error");
        }
    }

    fn nth_mut(&mut self, n: usize) -> &mut S {
        if n == 0 {
            &mut self.0
        } else if n == 1 {
            &mut self.1
        } else {
            unreachable!("index error");
        }
    }
}
