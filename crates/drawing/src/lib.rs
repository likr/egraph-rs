mod drawing;
mod metric;

use ndarray::prelude::*;
use num_traits::{FloatConst, FromPrimitive, Signed};
use std::hash::Hash;

/// A trait constraint for types that can be used as indices in a drawing (typically node identifiers).
///
/// Requires the type to implement `Eq` and `Hash`.
pub trait DrawingIndex: Eq + Hash {}
impl<T> DrawingIndex for T where T: Eq + Hash {}

/// A trait constraint for types that can be used as coordinate values in a drawing.
///
/// Requires the type to implement `NdFloat` (from `ndarray`) and `FromPrimitive` (from `num_traits`).
pub trait DrawingValue:
    NdFloat + FromPrimitive + FloatConst + Signed + Into<f64> + From<f32>
{
}
impl<T> DrawingValue for T where
    T: NdFloat + FromPrimitive + FloatConst + Signed + Into<f64> + From<f32>
{
}

/// Represents a drawing, mapping indices (nodes) to coordinate arrays.
pub use drawing::{
    drawing_euclidean::DrawingEuclidean, drawing_euclidean_2d::DrawingEuclidean2d,
    drawing_hyperbolic_2d::DrawingHyperbolic2d, drawing_spherical_2d::DrawingSpherical2d,
    drawing_torus2d::DrawingTorus2d, Drawing,
};

pub use metric::{
    metric_euclidean::{DeltaEuclidean, MetricEuclidean},
    metric_euclidean_2d::{DeltaEuclidean2d, MetricEuclidean2d},
    metric_hyperbolic_2d::{DeltaHyperbolic2d, MetricHyperbolic2d},
    metric_spherical_2d::{DeltaSpherical2d, MetricSpherical2d},
    metric_torus2d::{DeltaTorus2d, MetricTorus2d, TorusValue},
    Delta, Metric, MetricCartesian,
};
