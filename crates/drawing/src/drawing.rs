pub mod drawing_euclidean;
pub mod drawing_euclidean_2d;
pub mod drawing_hyperbolic_2d;
pub mod drawing_spherical_2d;
pub mod drawing_torus2d;

use crate::{metric::Metric, DrawingIndex};

/// A generic trait representing a drawing or layout of items (nodes) in a specific metric space.
///
/// This trait provides methods to access item positions, dimensions, and other properties.
/// It acts as an abstraction over concrete drawing implementations like Euclidean, Spherical, etc.
pub trait Drawing {
    /// The type used to index items (nodes) in the drawing. Must implement `DrawingIndex`.
    type Index: DrawingIndex;
    /// The type representing the position of an item in the metric space. Must implement `Metric`.
    type Item: Metric;

    /// Returns the total number of items (nodes) in the drawing.
    fn len(&self) -> usize;
    /// Returns `true` if the drawing contains no items.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the dimensionality of the space in which the items are drawn.
    fn dimension(&self) -> usize;

    /// Returns an immutable reference to the position of the item identified by `u`.
    /// Returns `None` if the item `u` is not found.
    fn position(&self, u: Self::Index) -> Option<&Self::Item>;

    /// Returns a mutable reference to the position of the item identified by `u`.
    /// Returns `None` if the item `u` is not found.
    fn position_mut(&mut self, u: Self::Index) -> Option<&mut Self::Item>;

    /// Returns a reference to the item identifier (`Index`) at the given raw numerical index `i`.
    /// Panics if `i` is out of bounds.
    fn node_id(&self, i: usize) -> &Self::Index;

    /// Returns the raw numerical index corresponding to the item identifier `u`.
    /// Panics if `u` is not found.
    fn index(&self, u: Self::Index) -> usize;

    /// Returns an immutable reference to the position (`Item`) at the given raw numerical index `i`.
    /// Panics if `i` is out of bounds.
    fn raw_entry(&self, i: usize) -> &Self::Item;

    /// Returns a mutable reference to the position (`Item`) at the given raw numerical index `i`.
    /// Panics if `i` is out of bounds.
    fn raw_entry_mut(&mut self, i: usize) -> &mut Self::Item;

    /// Calculates the difference vector (delta) between the items at raw numerical indices `i` and `j`.
    /// Panics if `i` or `j` are out of bounds.
    fn delta(&self, i: usize, j: usize) -> <Self::Item as Metric>::D;
}
