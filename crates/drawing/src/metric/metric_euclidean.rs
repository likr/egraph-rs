use crate::{Delta, DrawingValue, Metric};
use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

use super::MetricCartesian;

/// Represents the difference vector between two points in Euclidean space.
///
/// This struct implements the `Delta` trait and represents the vector from one point to another
/// in a Euclidean space of arbitrary dimension.
///
/// # Type Parameters
///
/// * `S`: The scalar type used for coordinate values (must implement `DrawingValue`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeltaEuclidean<S>(pub Vec<S>);

impl<S> Add for DeltaEuclidean<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        DeltaEuclidean(
            self.0
                .iter()
                .zip(other.0.iter())
                .map(|(a, b)| *a + *b)
                .collect::<Vec<_>>(),
        )
    }
}

impl<S> Sub for DeltaEuclidean<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        DeltaEuclidean(
            self.0
                .iter()
                .zip(other.0.iter())
                .map(|(a, b)| *a - *b)
                .collect::<Vec<_>>(),
        )
    }
}

impl<S> Mul<S> for DeltaEuclidean<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn mul(self, other: S) -> Self {
        DeltaEuclidean(self.0.iter().map(|a| *a * other).collect::<Vec<_>>())
    }
}

impl<S> Div<S> for DeltaEuclidean<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn div(self, other: S) -> Self {
        DeltaEuclidean(self.0.iter().map(|a| *a / other).collect::<Vec<_>>())
    }
}

impl<S> Delta for DeltaEuclidean<S>
where
    S: DrawingValue,
{
    type S = S;
    fn norm(&self) -> Self::S {
        let mut s = S::zero();
        for a in self.0.iter() {
            s += (*a) * (*a);
        }
        s.sqrt()
    }
}

impl<S> DeltaEuclidean<S>
where
    S: DrawingValue + Default,
{
    /// Creates a new `DeltaEuclidean` instance with the specified dimension.
    ///
    /// All coordinates are initialized to their default values.
    ///
    /// # Parameters
    ///
    /// * `dimension`: The dimensionality of the Euclidean space.
    ///
    /// # Returns
    ///
    /// A new `DeltaEuclidean` instance.
    pub fn new(dimension: usize) -> Self {
        DeltaEuclidean(vec![S::default(); dimension])
    }
}

/// Represents a point in Euclidean space of arbitrary dimension.
///
/// This struct implements the `Metric` trait and represents a point in
/// a Euclidean space with coordinates stored as a vector.
///
/// # Type Parameters
///
/// * `S`: The scalar type used for coordinate values (must implement `DrawingValue`).
#[derive(Clone, Debug, Default)]
pub struct MetricEuclidean<S>(pub Vec<S>);

impl<S> MetricEuclidean<S>
where
    S: DrawingValue + Default,
{
    /// Creates a new `MetricEuclidean` instance with the specified dimension.
    ///
    /// All coordinates are initialized to their default values.
    ///
    /// # Parameters
    ///
    /// * `dimension`: The dimensionality of the Euclidean space.
    ///
    /// # Returns
    ///
    /// A new `MetricEuclidean` instance.
    pub fn new(dimension: usize) -> Self {
        MetricEuclidean(vec![S::default(); dimension])
    }
}

impl<S> AddAssign<DeltaEuclidean<S>> for MetricEuclidean<S>
where
    S: DrawingValue,
{
    fn add_assign(&mut self, other: DeltaEuclidean<S>) {
        for (a, b) in self.0.iter_mut().zip(other.0.iter()) {
            *a += *b
        }
    }
}

impl<S> SubAssign<DeltaEuclidean<S>> for MetricEuclidean<S>
where
    S: DrawingValue,
{
    fn sub_assign(&mut self, other: DeltaEuclidean<S>) {
        for (a, b) in self.0.iter_mut().zip(other.0.iter()) {
            *a -= *b
        }
    }
}

impl<S> Metric for MetricEuclidean<S>
where
    S: DrawingValue,
{
    type D = DeltaEuclidean<S>;
}

impl<'b, S> Sub<&'b MetricEuclidean<S>> for &MetricEuclidean<S>
where
    S: DrawingValue,
{
    type Output = DeltaEuclidean<S>;

    fn sub(self, other: &'b MetricEuclidean<S>) -> DeltaEuclidean<S> {
        DeltaEuclidean(
            self.0
                .iter()
                .zip(other.0.iter())
                .map(|(a, b)| *a - *b)
                .collect::<Vec<_>>(),
        )
    }
}

impl<S> MetricCartesian for MetricEuclidean<S>
where
    S: DrawingValue,
{
    fn nth(&self, n: usize) -> &S {
        &self.0[n]
    }

    fn nth_mut(&mut self, n: usize) -> &mut S {
        &mut self.0[n]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_metric_euclidean_sub() {
        let x = MetricEuclidean(vec![1., 2., 3.]);
        let y = MetricEuclidean(vec![4., 5., 6.]);
        let z = DeltaEuclidean(vec![-3., -3., -3.]);
        assert_eq!(&x - &y, z);
    }

    #[test]
    fn test_delta_euclidean_norm() {
        let x = DeltaEuclidean(vec![2., 2., 2., 2.]);
        assert_eq!(x.norm(), 4.);
    }
}
