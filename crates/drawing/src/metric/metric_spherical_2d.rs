use num_traits::clamp;

use crate::{Delta, DrawingValue, Metric};
use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

/// Represents the difference vector between two points in 2D Spherical space.
///
/// This struct implements the `Delta` trait for 2D Spherical space.
/// It stores the components of the vector in the tangent space using longitudinal
/// and latitudinal displacements.
///
/// # Type Parameters
///
/// * `S`: The scalar type used for coordinate values (must implement `DrawingValue`).
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

/// Represents a point in 2D Spherical space.
///
/// This struct implements the `Metric` trait for 2D Spherical space.
/// Points are represented using spherical coordinates: longitude (0) and latitude (1).
/// Longitude represents the angular coordinate running east-west (0 to 2π),
/// while latitude represents the angular coordinate running north-south (-π/2 to π/2).
///
/// # Type Parameters
///
/// * `S`: The scalar type used for coordinate values (must implement `DrawingValue`).
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

impl<'b, S> Sub<&'b MetricSpherical2d<S>> for &MetricSpherical2d<S>
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

/// Converts a vector from spherical space to the tangent space at point x.
///
/// This function computes the exponential map from the spherical coordinates to
/// the tangent space, allowing for calculations in the flat tangent space.
///
/// # Parameters
///
/// * `x`: The coordinates of the reference point in spherical space (longitude, latitude).
/// * `y`: The coordinates of the target point in spherical space (longitude, latitude).
///
/// # Returns
///
/// The coordinates of the vector in the tangent space at point x.
fn to_tangent_space<S>(x: (S, S), y: (S, S)) -> (S, S)
where
    S: DrawingValue,
{
    // Check if the points are identical or very close
    let lon_diff = (y.0 - x.0).abs();
    let lat_diff = (y.1 - x.1).abs();

    if lon_diff <= S::from_f32(1e-10).unwrap() && lat_diff <= S::from_f32(1e-10).unwrap() {
        return (S::zero(), S::zero()); // Return zero vector for identical points
    }

    let ux = (-x.0.sin() * x.1.sin(), S::zero(), x.0.cos() * x.1.sin());
    let vx = (x.0.cos() * x.1.cos(), -x.1.sin(), x.0.sin() * x.1.cos());
    let ey = (y.0.cos() * y.1.sin(), y.1.cos(), y.0.sin() * y.1.sin());

    // Calculate the dot product with safeguards
    let dot_product = x.1.sin() * y.1.sin() * (y.0 - x.0).cos() + x.1.cos() * y.1.cos();
    let d = clamp(dot_product, -S::one(), S::one()).acos();

    // If the angle is very small, return a small vector in the appropriate direction
    if d <= S::from_f32(1e-10).unwrap() {
        // For very small angles, use a simplified approach
        return (
            S::from_f32(1e-10).unwrap() * (y.0 - x.0).signum(),
            S::from_f32(1e-10).unwrap() * (y.1 - x.1).signum(),
        );
    }

    (
        d * (ux.0 * ey.0 + ux.1 * ey.1 + ux.2 * ey.2),
        d * (vx.0 * ey.0 + vx.1 * ey.1 + vx.2 * ey.2),
    )
}

/// Converts a vector from the tangent space at point x to spherical space.
///
/// This function computes the logarithmic map from the tangent space back to
/// spherical coordinates, effectively implementing great circle navigation.
///
/// # Parameters
///
/// * `x`: The coordinates of the reference point in spherical space (longitude, latitude).
/// * `z`: The coordinates of the vector in the tangent space at point x.
///
/// # Returns
///
/// The coordinates of the point in spherical space (longitude, latitude).
fn from_tangent_space<S>(x: (S, S), z: (S, S)) -> (S, S)
where
    S: DrawingValue,
{
    // Handle the case where the tangent vector is zero (no movement)
    let z_squared_sum = z.0 * z.0 + z.1 * z.1;
    if z_squared_sum <= S::from_f32(1e-10).unwrap() {
        return x; // Return the original point if the movement is negligible
    }

    let ux = (-x.0.sin() * x.1.sin(), S::zero(), x.0.cos() * x.1.sin());
    let vx = (x.0.cos() * x.1.cos(), -x.1.sin(), x.0.sin() * x.1.cos());
    let p = (z.1, -z.0);

    // Calculate the normal vector with safeguards against division by zero
    let n = {
        let n = (
            p.0 * ux.0 + p.1 * vx.0,
            p.0 * ux.1 + p.1 * vx.1,
            p.0 * ux.2 + p.1 * vx.2,
        );
        let d_squared = n.0 * n.0 + n.1 * n.1 + n.2 * n.2;

        // Check if the magnitude is too small
        if d_squared <= S::from_f32(1e-10).unwrap() {
            // Use a default normal vector if the calculated one is too small
            (S::zero(), S::one(), S::zero())
        } else {
            let d = d_squared.sqrt();
            (n.0 / d, n.1 / d, n.2 / d)
        }
    };

    let ex = (x.0.cos() * x.1.sin(), x.1.cos(), x.0.sin() * x.1.sin());

    // Calculate the magnitude of the tangent vector with a safety check
    let t = -(z_squared_sum.sqrt());

    // Calculate the new point with safeguards
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

    // Handle potential NaN in the final calculation
    let ey_xy_squared = ey.0 * ey.0 + ey.2 * ey.2;
    let lon = if ey_xy_squared <= S::from_f32(1e-10).unwrap() {
        // If the point is at or very near a pole, longitude doesn't matter much
        // but we need to avoid NaN, so use the original longitude
        x.0
    } else {
        ey.2.atan2(ey.0)
    };

    // Ensure the y component is within valid range for acos
    let y_clamped = clamp(ey.1, S::from_f32(-1.0).unwrap(), S::from_f32(1.0).unwrap());
    let lat = y_clamped.acos();

    (lon, lat)
}
