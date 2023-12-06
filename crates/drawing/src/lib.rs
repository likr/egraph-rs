mod drawing;
mod drawing2d;
mod drawing_torus;
mod metric;

pub use drawing::*;
pub use drawing2d::*;
pub use drawing_torus::*;
pub use metric::*;

use ndarray::prelude::*;
use num_traits::FromPrimitive;
use std::hash::Hash;

pub trait DrawingIndex: Eq + Hash {}
impl<T> DrawingIndex for T where T: Eq + Hash {}
pub trait DrawingValue: NdFloat + FromPrimitive {}
impl<T> DrawingValue for T where T: NdFloat + FromPrimitive {}
