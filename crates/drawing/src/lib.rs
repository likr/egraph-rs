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
    euclidean::DrawingEuclidean, euclidean2d::DrawingEuclidean2d, torus2d::DrawingTorus2d, Drawing,
};
pub use metric::{
    euclidean::MetricEuclidean, euclidean2d::MetricEuclidean2d, torus2d::MetricTorus2d, Difference,
    Metric,
};
