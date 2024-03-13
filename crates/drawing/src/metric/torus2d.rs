use crate::{
    metric::{Difference, Metric},
    DrawingValue,
};
use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

fn torus_value<S>(value: S) -> S
where
    S: DrawingValue,
{
    if value < S::zero() {
        value.fract() + S::one()
    } else {
        value.fract()
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct TorusValue<S>(pub S);

impl<S> TorusValue<S>
where
    S: DrawingValue,
{
    pub fn new(value: S) -> Self {
        TorusValue(torus_value(value))
    }

    pub fn min() -> Self {
        TorusValue(S::zero())
    }

    pub fn max() -> Self {
        TorusValue(S::one() - S::epsilon())
    }
}

impl<S> Add<S> for TorusValue<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn add(self, other: S) -> Self {
        Self::new(self.0 + other)
    }
}

impl<S> AddAssign<S> for TorusValue<S>
where
    S: DrawingValue,
{
    fn add_assign(&mut self, other: S) {
        self.0 = torus_value(self.0 + other);
    }
}

impl<S> Sub<S> for TorusValue<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn sub(self, other: S) -> Self {
        Self::new(self.0 - other)
    }
}

impl<S> SubAssign<S> for TorusValue<S>
where
    S: DrawingValue,
{
    fn sub_assign(&mut self, other: S) {
        self.0 = torus_value(self.0 - other);
    }
}

impl<S> Mul<S> for TorusValue<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn mul(self, other: S) -> Self {
        Self::new(self.0 * other)
    }
}

impl<S> Div<S> for TorusValue<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn div(self, other: S) -> Self {
        Self::new(self.0 / other)
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DifferenceTorus2d<S>(pub S, pub S);

impl<S> Add for DifferenceTorus2d<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0, self.1 + other.1)
    }
}

impl<S> Sub for DifferenceTorus2d<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0, self.1 - other.1)
    }
}

impl<S> Mul<S> for DifferenceTorus2d<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn mul(self, other: S) -> Self {
        Self(self.0 * other, self.1 * other)
    }
}

impl<S> Div<S> for DifferenceTorus2d<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn div(self, other: S) -> Self {
        Self(self.0 / other, self.1 / other)
    }
}

impl<S> Difference for DifferenceTorus2d<S>
where
    S: DrawingValue,
{
    type S = S;

    fn norm(&self) -> Self::S {
        self.0.hypot(self.1)
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct MetricTorus2d<S>(pub TorusValue<S>, pub TorusValue<S>);

impl<S> MetricTorus2d<S>
where
    S: DrawingValue,
{
    pub fn new() -> Self
    where
        S: Default,
    {
        Self(TorusValue::new(S::default()), TorusValue::new(S::default()))
    }
    pub fn nearest_dxdy(self, other: Self) -> (S, S)
    where
        S: DrawingValue,
    {
        let x0 = other.0 .0;
        let y0 = other.1 .0;
        let mut d = S::infinity();
        let mut min_dxdy = (S::zero(), S::zero());
        for dy in -1..=1 {
            let dy = S::from_i32(dy).unwrap();
            let y1 = self.1 .0 + dy;
            for dx in -1..=1 {
                let dx = S::from_i32(dx).unwrap();
                let x1 = self.0 .0 + dx;
                let new_d = (x1 - x0).hypot(y1 - y0);
                if new_d < d {
                    d = new_d;
                    min_dxdy = (dx, dy);
                }
            }
        }
        min_dxdy
    }
}

impl<S> Sub for MetricTorus2d<S>
where
    S: DrawingValue,
{
    type Output = DifferenceTorus2d<S>;

    fn sub(self, other: Self) -> Self::Output {
        let (dx, dy) = self.nearest_dxdy(other);
        let x0 = other.0 .0;
        let y0 = other.1 .0;
        let x1 = self.0 .0 + dx;
        let y1 = self.1 .0 + dy;
        DifferenceTorus2d(x1 - x0, y1 - y0)
    }
}

impl<S> Add<DifferenceTorus2d<S>> for MetricTorus2d<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn add(self, other: DifferenceTorus2d<S>) -> Self {
        Self(self.0 + other.0, self.1 + other.1)
    }
}

impl<S> Sub<DifferenceTorus2d<S>> for MetricTorus2d<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn sub(self, other: DifferenceTorus2d<S>) -> Self {
        Self(self.0 - other.0, self.1 - other.1)
    }
}

impl<S> AddAssign<DifferenceTorus2d<S>> for MetricTorus2d<S>
where
    S: DrawingValue,
{
    fn add_assign(&mut self, other: DifferenceTorus2d<S>) {
        self.0 += other.0;
        self.1 += other.1;
    }
}

impl<S> SubAssign<DifferenceTorus2d<S>> for MetricTorus2d<S>
where
    S: DrawingValue,
{
    fn sub_assign(&mut self, other: DifferenceTorus2d<S>) {
        self.0 -= other.0;
        self.1 -= other.1;
    }
}

impl<S> Metric for MetricTorus2d<S>
where
    S: DrawingValue,
{
    type D = DifferenceTorus2d<S>;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_torus_value() {
        let a = 8.625;
        assert_eq!(torus_value(a), 0.625);
    }

    #[test]
    fn test_torus_add() {
        let a = TorusValue::new(0.5);
        let b = 1.625;
        assert_eq!(a + b, TorusValue::new(0.125));
    }

    #[test]
    fn test_torus_sub() {
        let a = TorusValue::new(0.5);
        let b = 1.625;
        assert_eq!(a - b, TorusValue::new(0.875));
    }

    #[test]
    fn test_metric_2d_torus_sub() {
        let x = MetricTorus2d(TorusValue::new(0.), TorusValue::new(0.75));
        let y = MetricTorus2d(TorusValue::new(0.5), TorusValue::new(0.5));
        let z = DifferenceTorus2d(-0.5, 0.25);
        assert_eq!(x - y, z);
    }
}
