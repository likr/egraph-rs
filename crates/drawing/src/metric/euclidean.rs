use crate::{
    metric::{Difference, Metric},
    DrawingValue,
};
use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

#[derive(Clone, Debug, Default)]
pub struct MetricEuclidean<S>(pub Vec<S>);

impl<S> Add for MetricEuclidean<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        MetricEuclidean(
            self.0
                .iter()
                .zip(other.0.iter())
                .map(|(a, b)| *a + *b)
                .collect::<Vec<_>>(),
        )
    }
}

impl<S> AddAssign for MetricEuclidean<S>
where
    S: DrawingValue,
{
    fn add_assign(&mut self, other: Self) {
        for (a, b) in self.0.iter_mut().zip(other.0.iter()) {
            *a += *b
        }
    }
}

impl<S> Sub for MetricEuclidean<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        MetricEuclidean(
            self.0
                .iter()
                .zip(other.0.iter())
                .map(|(a, b)| *a - *b)
                .collect::<Vec<_>>(),
        )
    }
}

impl<S> SubAssign for MetricEuclidean<S>
where
    S: DrawingValue,
{
    fn sub_assign(&mut self, other: Self) {
        for (a, b) in self.0.iter_mut().zip(other.0.iter()) {
            *a -= *b
        }
    }
}

impl<S> Mul<S> for MetricEuclidean<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn mul(self, other: S) -> Self {
        MetricEuclidean(self.0.iter().map(|a| *a * other).collect::<Vec<_>>())
    }
}

impl<S> Div<S> for MetricEuclidean<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn div(self, other: S) -> Self {
        MetricEuclidean(self.0.iter().map(|a| *a / other).collect::<Vec<_>>())
    }
}

impl<S> Difference for MetricEuclidean<S>
where
    S: DrawingValue,
{
    type S = S;
    fn norm(&self) -> Self::S {
        let mut s = S::zero();
        for a in self.0.iter() {
            s += *a * *a;
        }
        s.sqrt()
    }
}

impl<S> MetricEuclidean<S>
where
    S: DrawingValue + Default,
{
    pub fn new(dimension: usize) -> Self {
        MetricEuclidean(vec![S::default(); dimension])
    }
}

impl<S> Metric for MetricEuclidean<S>
where
    S: DrawingValue,
{
    type D = MetricEuclidean<S>;
}
