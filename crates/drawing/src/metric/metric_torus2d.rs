use crate::{Delta, DrawingValue, Metric};
use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

/// Normalizes a value to the range [0,1) for use in torus coordinate calculations.
///
/// This function takes any value and maps it to the range [0,1) by taking the fractional
/// part and handling negative values appropriately.
///
/// # Parameters
///
/// * `value`: The value to be normalized.
///
/// # Returns
///
/// The normalized value in the range [0,1).
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

/// Represents a value in the torus space, constrained to the range [0,1).
///
/// This type ensures that coordinate values remain within the proper range
/// for torus computations, where values wrap around at 0 and 1.
///
/// # Type Parameters
///
/// * `S`: The scalar type used for the value (must implement `DrawingValue`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct TorusValue<S>(pub S);

impl<S> TorusValue<S>
where
    S: DrawingValue,
{
    /// Creates a new `TorusValue` by normalizing the input value to the range [0,1).
    ///
    /// # Parameters
    ///
    /// * `value`: The value to be wrapped into the torus coordinate range.
    ///
    /// # Returns
    ///
    /// A new `TorusValue` instance with the normalized value.
    pub fn new(value: S) -> Self {
        TorusValue(torus_value(value))
    }

    /// Returns the minimum value in the torus space (0).
    ///
    /// # Returns
    ///
    /// A `TorusValue` representing the minimum value (0).
    pub fn min() -> Self {
        TorusValue(S::zero())
    }

    /// Returns the maximum value in the torus space (just under 1).
    ///
    /// # Returns
    ///
    /// A `TorusValue` representing the maximum value (1-ε).
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

/// Represents the difference vector between two points in 2D Torus space.
///
/// This struct implements the `Delta` trait for 2D Torus space.
/// It stores the x and y components of the vector, accounting for
/// the wrapping property of the torus.
///
/// # Type Parameters
///
/// * `S`: The scalar type used for coordinate values (must implement `DrawingValue`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DeltaTorus2d<S>(pub S, pub S);

impl<S> Add for DeltaTorus2d<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0, self.1 + other.1)
    }
}

impl<S> Sub for DeltaTorus2d<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0, self.1 - other.1)
    }
}

impl<S> Mul<S> for DeltaTorus2d<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn mul(self, other: S) -> Self {
        Self(self.0 * other, self.1 * other)
    }
}

impl<S> Div<S> for DeltaTorus2d<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn div(self, other: S) -> Self {
        Self(self.0 / other, self.1 / other)
    }
}

impl<S> Delta for DeltaTorus2d<S>
where
    S: DrawingValue,
{
    type S = S;

    fn norm(&self) -> Self::S {
        self.0.hypot(self.1)
    }
}

/// Represents a point in 2D Torus space.
///
/// This struct implements the `Metric` trait for 2D Torus space.
/// It stores x and y coordinates as `TorusValue` instances to ensure they
/// remain in the proper range [0,1) with wrap-around semantics.
///
/// # Type Parameters
///
/// * `S`: The scalar type used for coordinate values (must implement `DrawingValue`).
#[derive(Copy, Clone, Debug, Default)]
pub struct MetricTorus2d<S>(pub TorusValue<S>, pub TorusValue<S>);

impl<S> MetricTorus2d<S>
where
    S: DrawingValue,
{
    /// Creates a new `MetricTorus2d` instance with default coordinates.
    ///
    /// The coordinates are initialized to the default value of `S` (typically 0).
    ///
    /// # Returns
    ///
    /// A new `MetricTorus2d` instance.
    pub fn new() -> Self
    where
        S: Default,
    {
        Self(TorusValue::new(S::default()), TorusValue::new(S::default()))
    }

    /// Computes the shortest displacement vector (dx, dy) between two points on the torus.
    ///
    /// Due to the wrap-around nature of the torus, the shortest path between two points
    /// may cross the boundary. This function determines the optimal direction and distance.
    ///
    /// # Parameters
    ///
    /// * `other`: The target point to compute the displacement to.
    ///
    /// # Returns
    ///
    /// A tuple (dx, dy) representing the x and y components of the displacement.
    pub fn nearest_dxdy(self, other: &Self) -> (S, S)
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

impl<S> AddAssign<DeltaTorus2d<S>> for MetricTorus2d<S>
where
    S: DrawingValue,
{
    fn add_assign(&mut self, other: DeltaTorus2d<S>) {
        self.0 += other.0;
        self.1 += other.1;
    }
}

impl<S> SubAssign<DeltaTorus2d<S>> for MetricTorus2d<S>
where
    S: DrawingValue,
{
    fn sub_assign(&mut self, other: DeltaTorus2d<S>) {
        self.0 -= other.0;
        self.1 -= other.1;
    }
}

impl<S> Metric for MetricTorus2d<S>
where
    S: DrawingValue,
{
    type D = DeltaTorus2d<S>;
}

impl<'b, S> Sub<&'b MetricTorus2d<S>> for &MetricTorus2d<S>
where
    S: DrawingValue,
{
    type Output = DeltaTorus2d<S>;

    fn sub(self, other: &'b MetricTorus2d<S>) -> DeltaTorus2d<S> {
        let (dx, dy) = self.nearest_dxdy(other);
        let x0 = other.0 .0;
        let y0 = other.1 .0;
        let x1 = self.0 .0 + dx;
        let y1 = self.1 .0 + dy;
        DeltaTorus2d(x1 - x0, y1 - y0)
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
        let x = MetricTorus2d(TorusValue::new(0.), TorusValue::new(0.75));
        let y = MetricTorus2d(TorusValue::new(0.5), TorusValue::new(0.5));
        let z = DeltaTorus2d(-0.5, 0.25);
        assert_eq!(&x - &y, z);
    }
}
