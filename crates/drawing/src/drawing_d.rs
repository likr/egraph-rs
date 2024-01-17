use crate::{Difference, Drawing, DrawingIndex, DrawingValue, Metric};
use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

#[derive(Clone, Debug, Default)]
pub struct CoordinatesD<S>(pub Vec<S>);

impl<S> Add for CoordinatesD<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        CoordinatesD(
            self.0
                .iter()
                .zip(other.0.iter())
                .map(|(a, b)| *a + *b)
                .collect::<Vec<_>>(),
        )
    }
}

impl<S> AddAssign for CoordinatesD<S>
where
    S: DrawingValue,
{
    fn add_assign(&mut self, other: Self) {
        for (a, b) in self.0.iter_mut().zip(other.0.iter()) {
            *a += *b
        }
    }
}

impl<S> Sub for CoordinatesD<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        CoordinatesD(
            self.0
                .iter()
                .zip(other.0.iter())
                .map(|(a, b)| *a - *b)
                .collect::<Vec<_>>(),
        )
    }
}

impl<S> SubAssign for CoordinatesD<S>
where
    S: DrawingValue,
{
    fn sub_assign(&mut self, other: Self) {
        for (a, b) in self.0.iter_mut().zip(other.0.iter()) {
            *a -= *b
        }
    }
}

impl<S> Mul<S> for CoordinatesD<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn mul(self, other: S) -> Self {
        CoordinatesD(self.0.iter().map(|a| *a * other).collect::<Vec<_>>())
    }
}

impl<S> Div<S> for CoordinatesD<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn div(self, other: S) -> Self {
        CoordinatesD(self.0.iter().map(|a| *a / other).collect::<Vec<_>>())
    }
}

impl<S> Difference for CoordinatesD<S>
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

impl<S> Metric for CoordinatesD<S>
where
    S: DrawingValue,
{
    type D = CoordinatesD<S>;
}

pub type DrawingD<N, S> = Drawing<N, CoordinatesD<S>>;

impl<N, S> DrawingD<N, S>
where
    N: DrawingIndex,
    S: DrawingValue,
{
    pub fn set_dimension(&mut self, d: usize) {
        for i in 0..self.len() {
            self.coordinates[i].0.resize(d, S::zero());
        }
    }
}
