use num_traits::{FloatConst, FromPrimitive};
use petgraph::visit::IntoNodeIdentifiers;

use crate::{Difference, Drawing, DrawingIndex, DrawingValue, Metric};
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
pub struct Difference2DTorus<S>(pub S, pub S);

impl<S> Add for Difference2DTorus<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0, self.1 + other.1)
    }
}

impl<S> Sub for Difference2DTorus<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0, self.1 - other.1)
    }
}

impl<S> Mul<S> for Difference2DTorus<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn mul(self, other: S) -> Self {
        Self(self.0 * other, self.1 * other)
    }
}

impl<S> Div<S> for Difference2DTorus<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn div(self, other: S) -> Self {
        Self(self.0 / other, self.1 / other)
    }
}

impl<S> Difference for Difference2DTorus<S>
where
    S: DrawingValue,
{
    type S = S;

    fn norm(&self) -> Self::S {
        self.0.hypot(self.1)
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Metric2DTorus<S>(pub TorusValue<S>, pub TorusValue<S>);

impl<S> Sub for Metric2DTorus<S>
where
    S: DrawingValue,
{
    type Output = Difference2DTorus<S>;

    fn sub(self, other: Self) -> Self::Output {
        let x0 = other.0 .0;
        let y0 = other.1 .0;
        let mut d = Difference2DTorus(self.0 .0 - x0, self.1 .0 - y0);
        for dy in -1..=1 {
            for dx in -1..=1 {
                let x1 = self.0 .0 + S::from_i32(dx).unwrap();
                let y1 = self.1 .0 + S::from_i32(dy).unwrap();
                let new_d = Difference2DTorus(x1 - x0, y1 - y0);
                if new_d.norm() < d.norm() {
                    d = new_d;
                }
            }
        }
        d
    }
}

impl<S> Add<Difference2DTorus<S>> for Metric2DTorus<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn add(self, other: Difference2DTorus<S>) -> Self {
        Self(self.0 + other.0, self.1 + other.1)
    }
}

impl<S> Sub<Difference2DTorus<S>> for Metric2DTorus<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn sub(self, other: Difference2DTorus<S>) -> Self {
        Self(self.0 - other.0, self.1 - other.1)
    }
}

impl<S> AddAssign<Difference2DTorus<S>> for Metric2DTorus<S>
where
    S: DrawingValue,
{
    fn add_assign(&mut self, other: Difference2DTorus<S>) {
        self.0 += other.0;
        self.1 += other.1;
    }
}

impl<S> SubAssign<Difference2DTorus<S>> for Metric2DTorus<S>
where
    S: DrawingValue,
{
    fn sub_assign(&mut self, other: Difference2DTorus<S>) {
        self.0 -= other.0;
        self.1 -= other.1;
    }
}

impl<S> Metric for Metric2DTorus<S>
where
    S: DrawingValue,
{
    type D = Difference2DTorus<S>;
}

pub type DrawingTorus<N, S> = Drawing<N, Metric2DTorus<S>>;

impl<N, S> DrawingTorus<N, S>
where
    N: DrawingIndex,
    S: DrawingValue,
{
    pub fn x(&self, u: N) -> Option<S> {
        self.position(u).map(|p| p.0 .0)
    }

    pub fn y(&self, u: N) -> Option<S> {
        self.position(u).map(|p| p.1 .0)
    }

    pub fn set_x(&mut self, u: N, value: S) -> Option<()> {
        self.position_mut(u).map(|p| p.0 = TorusValue::new(value))
    }

    pub fn set_y(&mut self, u: N, value: S) -> Option<()> {
        self.position_mut(u).map(|p| p.1 = TorusValue::new(value))
    }

    pub fn initial_placement<G>(graph: G) -> DrawingTorus<N, S>
    where
        G: IntoNodeIdentifiers,
        G::NodeId: DrawingIndex + Into<N>,
        N: Copy,
        S: FloatConst + FromPrimitive + Default,
    {
        let nodes = graph.node_identifiers().collect::<Vec<_>>();
        let n = nodes.len();
        let dt = S::from(2.).unwrap() * S::PI() / S::from_usize(n).unwrap();
        let r = S::from(0.4).unwrap();
        let cx = S::from(0.5).unwrap();
        let cy = S::from(0.5).unwrap();
        let mut drawing = Self::new(graph);
        for i in 0..n {
            let t = dt * S::from_usize(i).unwrap();
            if let Some(p) = drawing.position_mut(nodes[i].into()) {
                *p = Metric2DTorus(
                    TorusValue::new(r * t.cos() + cx),
                    TorusValue::new(r * t.sin() + cy),
                );
            }
        }
        drawing
    }
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
        let x = Metric2DTorus(TorusValue::new(0.), TorusValue::new(0.75));
        let y = Metric2DTorus(TorusValue::new(0.5), TorusValue::new(0.5));
        let z = Difference2DTorus(-0.5, 0.25);
        assert_eq!(x - y, z);
    }
}
