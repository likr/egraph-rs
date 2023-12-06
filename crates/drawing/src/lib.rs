mod drawing;
mod drawing2d;
mod metric;

pub use drawing::*;
pub use drawing2d::*;
pub use metric::*;

use ndarray::prelude::*;
use std::hash::Hash;

pub trait DrawingIndex: Eq + Hash {}
impl<T> DrawingIndex for T where T: Eq + Hash {}
pub trait DrawingValue: NdFloat {}
impl<T> DrawingValue for T where T: NdFloat {}
