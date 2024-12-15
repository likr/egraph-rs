use crate::{Delta, DrawingValue, Metric};
use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

#[derive(Copy, Clone, Debug, Default)]
pub struct DeltaHyperbolic2d<S>(pub S, pub S);

impl<S> Add for DeltaHyperbolic2d<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        DeltaHyperbolic2d(self.0 + other.0, self.1 + other.1)
    }
}

impl<S> Sub for DeltaHyperbolic2d<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        DeltaHyperbolic2d(self.0 - other.0, self.1 - other.1)
    }
}

impl<S> Mul<S> for DeltaHyperbolic2d<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn mul(self, other: S) -> Self {
        DeltaHyperbolic2d(self.0 * other, self.1 * other)
    }
}

impl<S> Div<S> for DeltaHyperbolic2d<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn div(self, other: S) -> Self {
        DeltaHyperbolic2d(self.0 / other, self.1 / other)
    }
}

impl<S> Delta for DeltaHyperbolic2d<S>
where
    S: DrawingValue,
{
    type S = S;
    fn norm(&self) -> Self::S {
        self.0.hypot(self.1)
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct MetricHyperbolic2d<S>(pub S, pub S);

impl<S> AddAssign<DeltaHyperbolic2d<S>> for MetricHyperbolic2d<S>
where
    S: DrawingValue,
{
    fn add_assign(&mut self, other: DeltaHyperbolic2d<S>) {
        let x = (self.0, self.1);
        let y = (-other.0, -other.1);
        let z = from_tangent_space(x, y);
        self.0 = z.0;
        self.1 = z.1;
    }
}

impl<S> SubAssign<DeltaHyperbolic2d<S>> for MetricHyperbolic2d<S>
where
    S: DrawingValue,
{
    fn sub_assign(&mut self, other: DeltaHyperbolic2d<S>) {
        let x = (self.0, self.1);
        let y = (other.0, other.1);
        let z = from_tangent_space(x, y);
        self.0 = z.0;
        self.1 = z.1;
    }
}

impl<S> Metric for MetricHyperbolic2d<S>
where
    S: DrawingValue,
{
    type D = DeltaHyperbolic2d<S>;
}

impl<'a, 'b, S> Sub<&'b MetricHyperbolic2d<S>> for &'a MetricHyperbolic2d<S>
where
    S: DrawingValue,
{
    type Output = DeltaHyperbolic2d<S>;

    fn sub(self, other: &'b MetricHyperbolic2d<S>) -> DeltaHyperbolic2d<S> {
        let x = (self.0, self.1);
        let y = (other.0, other.1);
        let z = to_tangent_space(x, y);
        DeltaHyperbolic2d(z.0, z.1)
    }
}

fn to_tangent_space<S>(x: (S, S), y: (S, S)) -> (S, S)
where
    S: DrawingValue,
{
    let dx = y.0 - x.0;
    let dy = y.1 - x.1;
    let dr = S::one() - x.0 * y.0 - x.1 * y.1;
    let di = x.1 * y.0 - x.0 * y.1;
    let d = dr * dr + di * di;
    let z = ((dr * dx + di * dy) / d, (dr * dy - di * dx) / d);
    let z_norm = (z.0 * z.0 + z.1 * z.1).sqrt();
    if z_norm < S::from_f32(1e-4).unwrap() {
        return (S::zero(), S::zero());
    }
    let e = ((S::one() + z_norm) / (S::one() - z_norm)).ln();
    if e.is_finite() {
        (z.0 / z_norm * e, z.1 / z_norm * e)
    } else {
        (z.0 / z_norm, z.1 / z_norm)
    }
}

fn from_tangent_space<S>(x: (S, S), z: (S, S)) -> (S, S)
where
    S: DrawingValue,
{
    let z_norm = (z.0 * z.0 + z.1 * z.1).sqrt();
    let y = if z_norm < S::from_f32(1e-4).unwrap() {
        (S::zero(), S::zero())
    } else if z_norm.exp().is_infinite() {
        (z.0 / z_norm, z.1 / z_norm)
    } else {
        let e = ((S::one() - z_norm.exp()) / (S::one() + z_norm.exp())).abs();
        (z.0 / z_norm * e, z.1 / z_norm * e)
    };
    let dx = -y.0 - x.0;
    let dy = -y.1 - x.1;
    let dr = -S::one() - x.0 * y.0 - x.1 * y.1;
    let di = x.1 * y.0 - x.0 * y.1;
    let d = dr * dr + di * di;
    let yx = (dr * dx + di * dy) / d;
    let yy = (dr * dy - di * dx) / d;
    let t = S::from_f32(0.99).unwrap();
    if (yx * yx + yy * yy).sqrt() < t {
        (yx, yy)
    } else {
        (yx * t, yy * t)
    }
}
