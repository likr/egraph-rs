mod drawing;
mod metric;

use ndarray::prelude::*;
use num_traits::FromPrimitive;
use std::hash::Hash;

pub trait DrawingIndex: Eq + Hash {}
impl<T> DrawingIndex for T where T: Eq + Hash {}
pub trait DrawingValue: NdFloat + FromPrimitive {}
impl<T> DrawingValue for T where T: NdFloat + FromPrimitive {}

pub use drawing::{
    drawing_euclidean::DrawingEuclidean, drawing_euclidean_2d::DrawingEuclidean2d,
    drawing_spherical_2d::DrawingSpherical2d, drawing_torus2d::DrawingTorus2d, Drawing,
};
pub use metric::{
    metric_euclidean::{DeltaEuclidean, MetricEuclidean},
    metric_euclidean_2d::{DeltaEuclidean2d, MetricEuclidean2d},
    metric_spherical_2d::{DeltaSpherical2d, MetricSpherical2d},
    metric_torus2d::{DeltaTorus2d, MetricTorus2d, TorusValue},
    Delta, Metric,
};
