use num_traits::clamp;

use crate::{Delta, DrawingValue, Metric};
use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

#[derive(Copy, Clone, Debug, Default)]
pub struct DeltaSpherical2d<S>(pub S, pub S);

impl<S> Add for DeltaSpherical2d<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        DeltaSpherical2d(self.0 + other.0, self.1 + other.1)
    }
}

impl<S> Sub for DeltaSpherical2d<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        DeltaSpherical2d(self.0 - other.0, self.1 - other.1)
    }
}

impl<S> Mul<S> for DeltaSpherical2d<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn mul(self, other: S) -> Self {
        DeltaSpherical2d(self.0 * other, self.1 * other)
    }
}

impl<S> Div<S> for DeltaSpherical2d<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn div(self, other: S) -> Self {
        DeltaSpherical2d(self.0 / other, self.1 / other)
    }
}

impl<S> Delta for DeltaSpherical2d<S>
where
    S: DrawingValue,
{
    type S = S;
    fn norm(&self) -> Self::S {
        self.0.hypot(self.1)
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct MetricSpherical2d<S>(pub S, pub S);

impl<S> AddAssign<DeltaSpherical2d<S>> for MetricSpherical2d<S>
where
    S: DrawingValue,
{
    fn add_assign(&mut self, other: DeltaSpherical2d<S>) {
        let x = (self.0, self.1);
        let y = (-other.0, -other.1);
        let z = from_tangent_space(x, y);
        self.0 = z.0;
        self.1 = z.1;
    }
}

impl<S> SubAssign<DeltaSpherical2d<S>> for MetricSpherical2d<S>
where
    S: DrawingValue,
{
    fn sub_assign(&mut self, other: DeltaSpherical2d<S>) {
        let x = (self.0, self.1);
        let y = (other.0, other.1);
        let z = from_tangent_space(x, y);
        self.0 = z.0;
        self.1 = z.1;
    }
}

impl<S> Metric for MetricSpherical2d<S>
where
    S: DrawingValue,
{
    type D = DeltaSpherical2d<S>;
}

impl<'a, 'b, S> Sub<&'b MetricSpherical2d<S>> for &'a MetricSpherical2d<S>
where
    S: DrawingValue,
{
    type Output = DeltaSpherical2d<S>;

    fn sub(self, other: &'b MetricSpherical2d<S>) -> DeltaSpherical2d<S> {
        let x = (self.0, self.1);
        let y = (other.0, other.1);
        let z = to_tangent_space(x, y);
        DeltaSpherical2d(z.0, z.1)
    }
}

fn to_tangent_space<S>(x: (S, S), y: (S, S)) -> (S, S)
where
    S: DrawingValue,
{
    let ux = (-x.0.sin() * x.1.sin(), S::zero(), x.0.cos() * x.1.sin());
    let vx = (x.0.cos() * x.1.cos(), -x.1.sin(), x.0.sin() * x.1.cos());
    let ey = (y.0.cos() * y.1.sin(), y.1.cos(), y.0.sin() * y.1.sin());
    let d = clamp(
        x.1.sin() * y.1.sin() * (y.0 - x.0).cos() + x.1.cos() * y.1.cos(),
        -S::one(),
        S::one(),
    )
    .acos();
    (
        d * (ux.0 * ey.0 + ux.1 * ey.1 + ux.2 * ey.2),
        d * (vx.0 * ey.0 + vx.1 * ey.1 + vx.2 * ey.2),
    )
}

fn from_tangent_space<S>(x: (S, S), z: (S, S)) -> (S, S)
where
    S: DrawingValue,
{
    let ux = (-x.0.sin() * x.1.sin(), S::zero(), x.0.cos() * x.1.sin());
    let vx = (x.0.cos() * x.1.cos(), -x.1.sin(), x.0.sin() * x.1.cos());
    let p = (z.1, -z.0);
    let n = {
        let n = (
            p.0 * ux.0 + p.1 * vx.0,
            p.0 * ux.1 + p.1 * vx.1,
            p.0 * ux.2 + p.1 * vx.2,
        );
        let d = (n.0 * n.0 + n.1 * n.1 + n.2 * n.2).sqrt();
        (n.0 / d, n.1 / d, n.2 / d)
    };
    let ex = (x.0.cos() * x.1.sin(), x.1.cos(), x.0.sin() * x.1.sin());
    let t = -(z.0 * z.0 + z.1 * z.1).sqrt();
    let ey = (
        (n.0 * n.0 * (S::one() - t.cos()) + t.cos()) * ex.0
            + (n.0 * n.1 * (S::one() - t.cos()) - n.2 * t.sin()) * ex.1
            + (n.2 * n.0 * (S::one() - t.cos()) + n.1 * t.sin()) * ex.2,
        (n.0 * n.1 * (S::one() - t.cos()) + n.2 * t.sin()) * ex.0
            + (n.1 * n.1 * (S::one() - t.cos()) + t.cos()) * ex.1
            + (n.1 * n.2 * (S::one() - t.cos()) - n.0 * t.sin()) * ex.2,
        (n.2 * n.0 * (S::one() - t.cos()) - n.1 * t.sin()) * ex.0
            + (n.1 * n.2 * (S::one() - t.cos()) + n.0 * t.sin()) * ex.1
            + (n.2 * n.2 * (S::one() - t.cos()) + t.cos()) * ex.2,
    );
    (ey.2.atan2(ey.0), ey.1.acos())
}
