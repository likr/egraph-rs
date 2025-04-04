use crate::{DeltaEuclidean, Drawing, DrawingIndex, DrawingValue, MetricEuclidean};
use petgraph::visit::IntoNodeIdentifiers;
use std::collections::HashMap;

/// Represents a drawing of items (nodes) in Euclidean space of arbitrary dimension.
///
/// It stores the positions of items and provides methods to access and modify them.
/// This struct implements the `Drawing` trait.
///
/// Generic Parameters:
/// - `N`: The type used for indexing items (must implement `DrawingIndex`).
/// - `S`: The scalar type used for coordinates (must implement `DrawingValue`).
pub struct DrawingEuclidean<N, S> {
    /// A vector containing the unique identifiers (indices) of the items.
    indices: Vec<N>,
    /// A vector storing the coordinates (`MetricEuclidean`) of each item.
    /// The order corresponds to the `indices` vector.
    coordinates: Vec<MetricEuclidean<S>>,
    /// A map from item identifiers (`N`) to their numerical index (position in `indices` and `coordinates`).
    index_map: HashMap<N, usize>,
    /// The dimensionality of the Euclidean space.
    dimension: usize,
}

impl<N, S> DrawingEuclidean<N, S>
where
    N: DrawingIndex,
    S: DrawingValue,
{
    /// Creates a new `DrawingEuclidean` instance from a graph-like structure and a dimension.
    ///
    /// It extracts the node identifiers from the graph, assigns default coordinates (origin)
    /// to each item, and sets up the internal mapping.
    ///
    /// - `graph`: An object implementing `IntoNodeIdentifiers` (like `petgraph::Graph`).
    /// - `dimension`: The desired dimensionality of the Euclidean space.
    ///
    /// Returns a new `DrawingEuclidean` instance.
    pub fn new<G>(graph: G, dimension: usize) -> Self
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
        Self::from_node_indices(&indices, dimension)
    }

    /// Creates a new `DrawingEuclidean` instance from a slice of node indices and a dimension.
    ///
    /// This is a lower-level constructor used by `new`. It initializes coordinates to the origin.
    ///
    /// - `indices`: A slice containing the unique identifiers (`N`) for the items.
    /// - `dimension`: The desired dimensionality of the Euclidean space.
    ///
    /// Returns a new `DrawingEuclidean` instance.
    pub fn from_node_indices(indices: &[N], dimension: usize) -> Self
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
        let coordinates = vec![MetricEuclidean::new(dimension); indices.len()];
        Self {
            indices,
            coordinates,
            index_map,
            dimension,
        }
    }

    /// Gets the value of the `d`-th dimension for the item `u`.
    ///
    /// Returns `None` if the item `u` is not found or if `d` is out of bounds for the dimension.
    pub fn get(&self, u: N, d: usize) -> Option<S> {
        self.position(u).and_then(|p| p.0.get(d)).copied()
    }

    /// Sets the value of the `d`-th dimension for the item `u` to `value`.
    ///
    /// Returns `None` if the item `u` is not found or if `d` is out of bounds for the dimension.
    /// Otherwise, returns `Some(())`.
    pub fn set(&mut self, u: N, d: usize, value: S) -> Option<()> {
        self.position_mut(u)
            .and_then(|p| p.0.get_mut(d))
            .map(|p| *p = value)
    }
}

impl<N, S> Drawing for DrawingEuclidean<N, S>
where
    N: DrawingIndex,
    S: DrawingValue,
{
    type Index = N;
    type Item = MetricEuclidean<S>;

    fn len(&self) -> usize {
        self.indices.len()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn position(&self, u: N) -> Option<&Self::Item> {
        self.index_map.get(&u).map(|&i| &self.coordinates[i])
    }

    fn position_mut(&mut self, u: N) -> Option<&mut Self::Item> {
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

    fn delta(&self, i: usize, j: usize) -> DeltaEuclidean<S> {
        self.raw_entry(i) - self.raw_entry(j)
    }
}
