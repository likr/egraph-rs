pub mod drawing_euclidean;
pub mod drawing_euclidean_2d;
pub mod drawing_hyperbolic_2d;
pub mod drawing_spherical_2d;
pub mod drawing_torus2d;

use crate::{metric::Metric, DrawingIndex};

pub trait Drawing {
    type Index: DrawingIndex;
    type Item: Metric;

    fn len(&self) -> usize;

    fn dimension(&self) -> usize;

    fn position(&self, u: Self::Index) -> Option<&Self::Item>;

    fn position_mut(&mut self, u: Self::Index) -> Option<&mut Self::Item>;

    fn node_id(&self, i: usize) -> &Self::Index;

    fn index(&self, u: Self::Index) -> usize;

    fn raw_entry(&self, i: usize) -> &Self::Item;

    fn raw_entry_mut(&mut self, i: usize) -> &mut Self::Item;

    fn delta(&self, i: usize, j: usize) -> <Self::Item as Metric>::D;
}
