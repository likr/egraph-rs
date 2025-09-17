use crate::{DeltaHyperbolic2d, Drawing, DrawingIndex, DrawingValue, MetricHyperbolic2d};
use num_traits::{FloatConst, FromPrimitive};
use petgraph::visit::IntoNodeIdentifiers;
use std::collections::HashMap;

/// Represents a drawing of items (nodes) in 2-dimensional Hyperbolic space.
///
/// This drawing uses the Poincaré disk model of hyperbolic geometry, where points are
/// represented as coordinates within a unit disk.
///
/// # Type Parameters
///
/// * `N`: The type used for indexing items (must implement `DrawingIndex`).
/// * `S`: The scalar type used for coordinates (must implement `DrawingValue`).
pub struct DrawingHyperbolic2d<N, S> {
    /// A vector containing the unique identifiers (indices) of the items.
    indices: Vec<N>,
    /// A vector storing the hyperbolic coordinates (`MetricHyperbolic2d`) of each item.
    /// The order corresponds to the `indices` vector.
    coordinates: Vec<MetricHyperbolic2d<S>>,
    /// A map from item identifiers (`N`) to their numerical index (position in `indices` and `coordinates`).
    index_map: HashMap<N, usize>,
}

impl<N, S> DrawingHyperbolic2d<N, S>
where
    N: DrawingIndex,
    S: DrawingValue,
{
    /// Creates a new `DrawingHyperbolic2d` instance from a graph-like structure.
    ///
    /// It extracts the node identifiers from the graph, assigns default coordinates
    /// to each item, and sets up the internal mapping.
    ///
    /// - `graph`: An object implementing `IntoNodeIdentifiers` (like `petgraph::Graph`).
    ///
    /// Returns a new `DrawingHyperbolic2d` instance.
    pub fn new<G>(graph: G) -> Self
    where
        G: IntoNodeIdentifiers,
        G::NodeId: DrawingIndex + Into<N>,
        N: Copy,
        S: Default,
    {
        let indices = graph
            .node_identifiers()
            .map(|u| u.into())
            .collect::<Vec<N>>();
        Self::from_node_indices(&indices)
    }

    /// Creates a new `DrawingHyperbolic2d` instance from a slice of node indices.
    ///
    /// This is a lower-level constructor. It initializes coordinates to default values.
    ///
    /// - `indices`: A slice containing the unique identifiers (`N`) for the items.
    ///
    /// Returns a new `DrawingHyperbolic2d` instance.
    pub fn from_node_indices(indices: &[N]) -> Self
    where
        N: Copy,
        S: Default,
    {
        let indices = indices.to_vec();
        let index_map = indices
            .iter()
            .enumerate()
            .map(|(i, &u)| (u, i))
            .collect::<HashMap<_, _>>();
        let coordinates = vec![MetricHyperbolic2d::default(); indices.len()];
        Self {
            indices,
            coordinates,
            index_map,
        }
    }

    /// Gets the x-coordinate for the item `u` in the Poincaré disk model.
    ///
    /// Returns `None` if the item `u` is not found.
    pub fn x(&self, u: N) -> Option<S> {
        self.position(u).map(|p| p.0)
    }

    /// Gets the y-coordinate for the item `u` in the Poincaré disk model.
    ///
    /// Returns `None` if the item `u` is not found.
    pub fn y(&self, u: N) -> Option<S> {
        self.position(u).map(|p| p.1)
    }

    /// Sets the x-coordinate for the item `u` in the Poincaré disk model.
    ///
    /// Returns `None` if the item `u` is not found, otherwise returns `Some(())`.
    pub fn set_x(&mut self, u: N, value: S) -> Option<()> {
        self.position_mut(u).map(|p| p.0 = value)
    }

    /// Sets the y-coordinate for the item `u` in the Poincaré disk model.
    ///
    /// Returns `None` if the item `u` is not found, otherwise returns `Some(())`.
    pub fn set_y(&mut self, u: N, value: S) -> Option<()> {
        self.position_mut(u).map(|p| p.1 = value)
    }

    /// Creates a new drawing with nodes placed in a circular pattern within the Poincaré disk.
    ///
    /// This is useful for creating an initial layout before applying layout algorithms.
    ///
    /// - `graph`: An object implementing `IntoNodeIdentifiers` (like `petgraph::Graph`).
    ///
    /// Returns a new `DrawingHyperbolic2d` instance with nodes placed in a circular pattern.
    pub fn initial_placement<G>(graph: G) -> Self
    where
        G: IntoNodeIdentifiers,
        G::NodeId: DrawingIndex + Into<N>,
        N: Copy,
        S: FloatConst + FromPrimitive + Default,
    {
        let mut drawing = Self::new(graph);
        let n = drawing.len();
        let d = S::PI() * S::from_usize(2).unwrap() / S::from_usize(n).unwrap();
        for i in 0..n {
            drawing.coordinates[i].0 =
                S::from_f32(0.5).unwrap() * (S::from_usize(i).unwrap() * d).cos();
            drawing.coordinates[i].1 =
                S::from_f32(0.5).unwrap() * (S::from_usize(i).unwrap() * d).sin();
        }
        drawing
    }

    pub fn recenter(&mut self, u: N) {
        if let Some(center) = self.position(u) {
            let delta = &MetricHyperbolic2d::zero() - center;
            for pos in self.coordinates.iter_mut() {
                *pos += delta;
            }
        }
    }
}

impl<N, S> Drawing for DrawingHyperbolic2d<N, S>
where
    N: DrawingIndex,
    S: DrawingValue,
{
    type Index = N;
    type Item = MetricHyperbolic2d<S>;

    fn len(&self) -> usize {
        self.indices.len()
    }

    fn dimension(&self) -> usize {
        2
    }

    fn position(&self, u: Self::Index) -> Option<&Self::Item> {
        self.index_map.get(&u).map(|&i| &self.coordinates[i])
    }

    fn position_mut(&mut self, u: Self::Index) -> Option<&mut Self::Item> {
        self.index_map.get(&u).map(|&i| &mut self.coordinates[i])
    }

    fn node_id(&self, i: usize) -> &Self::Index {
        &self.indices[i]
    }

    fn index(&self, u: Self::Index) -> usize {
        self.index_map[&u]
    }

    fn raw_entry(&self, i: usize) -> &Self::Item {
        &self.coordinates[i]
    }

    fn raw_entry_mut(&mut self, i: usize) -> &mut Self::Item {
        &mut self.coordinates[i]
    }

    fn delta(&self, i: usize, j: usize) -> DeltaHyperbolic2d<S> {
        self.raw_entry(i) - self.raw_entry(j)
    }
}
