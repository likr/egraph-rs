use crate::{DeltaTorus2d, Drawing, DrawingIndex, DrawingValue, MetricTorus2d, TorusValue};
use num_traits::{FloatConst, FromPrimitive};
use petgraph::visit::IntoNodeIdentifiers;
use std::collections::HashMap;

/// Represents a drawing of items (nodes) in a 2-dimensional Torus space.
///
/// A torus is a surface of revolution generated by revolving a circle in three-dimensional
/// space about an axis that is coplanar with the circle. In this implementation, we use a
/// flat representation where opposite edges are connected.
///
/// # Type Parameters
///
/// * `N`: The type used for indexing items (must implement `DrawingIndex`).
/// * `S`: The scalar type used for coordinates (must implement `DrawingValue`).
pub struct DrawingTorus2d<N, S> {
    /// A vector containing the unique identifiers (indices) of the items.
    indices: Vec<N>,
    /// A vector storing the toroidal coordinates (`MetricTorus2d`) of each item.
    /// The order corresponds to the `indices` vector.
    coordinates: Vec<MetricTorus2d<S>>,
    /// A map from item identifiers (`N`) to their numerical index (position in `indices` and `coordinates`).
    index_map: HashMap<N, usize>,
}

impl<N, S> DrawingTorus2d<N, S>
where
    N: DrawingIndex,
    S: DrawingValue,
{
    /// Creates a new `DrawingTorus2d` instance from a graph-like structure.
    ///
    /// It extracts the node identifiers from the graph, assigns default coordinates
    /// to each item, and sets up the internal mapping.
    ///
    /// - `graph`: An object implementing `IntoNodeIdentifiers` (like `petgraph::Graph`).
    ///
    /// Returns a new `DrawingTorus2d` instance.
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

    /// Creates a new `DrawingTorus2d` instance from a slice of node indices.
    ///
    /// This is a lower-level constructor. It initializes coordinates to default values.
    ///
    /// - `indices`: A slice containing the unique identifiers (`N`) for the items.
    ///
    /// Returns a new `DrawingTorus2d` instance.
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
        let coordinates = vec![MetricTorus2d::new(); indices.len()];
        Self {
            indices,
            coordinates,
            index_map,
        }
    }
    /// Gets the x-coordinate for the item `u` in the torus space.
    ///
    /// Returns `None` if the item `u` is not found.
    pub fn x(&self, u: N) -> Option<S> {
        self.position(u).map(|p| p.0 .0)
    }

    /// Gets the y-coordinate for the item `u` in the torus space.
    ///
    /// Returns `None` if the item `u` is not found.
    pub fn y(&self, u: N) -> Option<S> {
        self.position(u).map(|p| p.1 .0)
    }

    /// Sets the x-coordinate for the item `u` in the torus space.
    ///
    /// Returns `None` if the item `u` is not found, otherwise returns `Some(())`.
    pub fn set_x(&mut self, u: N, value: S) -> Option<()> {
        self.position_mut(u).map(|p| p.0 = TorusValue::new(value))
    }

    /// Sets the y-coordinate for the item `u` in the torus space.
    ///
    /// Returns `None` if the item `u` is not found, otherwise returns `Some(())`.
    pub fn set_y(&mut self, u: N, value: S) -> Option<()> {
        self.position_mut(u).map(|p| p.1 = TorusValue::new(value))
    }

    /// Creates a new drawing with nodes placed in a circular pattern on the torus.
    ///
    /// This is useful for creating an initial layout before applying layout algorithms.
    /// Nodes are placed in a circular pattern around the center of the torus.
    ///
    /// - `graph`: An object implementing `IntoNodeIdentifiers` (like `petgraph::Graph`).
    ///
    /// Returns a new `DrawingTorus2d` instance with nodes placed in a circular pattern.
    pub fn initial_placement<G>(graph: G) -> Self
    where
        G: IntoNodeIdentifiers,
        G::NodeId: DrawingIndex + Into<N>,
        N: Copy,
        S: FloatConst + FromPrimitive + Default,
    {
        let nodes = graph.node_identifiers().collect::<Vec<_>>();
        let n = nodes.len();
        let dt = S::from(2.).unwrap() * S::PI() / S::from_usize(n).unwrap();
        let r = S::from(0.4).unwrap();
        let cx = S::from(0.5).unwrap();
        let cy = S::from(0.5).unwrap();
        let mut drawing = Self::new(graph);
        for (i, &node) in nodes.iter().enumerate().take(n) {
            let t = dt * S::from_usize(i).unwrap();
            if let Some(p) = drawing.position_mut(node.into()) {
                *p = MetricTorus2d(
                    TorusValue::new(r * t.cos() + cx),
                    TorusValue::new(r * t.sin() + cy),
                );
            }
        }
        drawing
    }

    /// Computes the segments needed to draw an edge between two nodes on the torus.
    ///
    /// Since the torus wraps around at the edges, drawing a straight line between
    /// two points may require multiple segments that cross the boundaries.
    ///
    /// - `u`: The identifier of the first node.
    /// - `v`: The identifier of the second node.
    ///
    /// Returns a vector of line segments represented as pairs of points (start, end).
    /// Returns `None` if either node is not found.
    pub fn edge_segments(&self, u: N, v: N) -> Option<Vec<(MetricTorus2d<S>, MetricTorus2d<S>)>> {
        self.position(u).zip(self.position(v)).map(|(&p, &q)| {
            let (dx, dy) = p.nearest_dxdy(&q);
            if dx == S::zero() && dy == S::zero() {
                vec![(p, q)]
            } else if dx == S::zero() {
                let (x0, y0, x1, y1) = if p.1 .0 < q.1 .0 {
                    (p.0 .0, p.1 .0, q.0 .0, q.1 .0)
                } else {
                    (q.0 .0, q.1 .0, p.0 .0, p.1 .0)
                };
                let x2 = (y0 * x1 - y1 * x0 + x0) / (y0 - y1 + S::one());
                vec![
                    (
                        MetricTorus2d(TorusValue::new(x0), TorusValue::new(y0)),
                        MetricTorus2d(TorusValue::new(x2), TorusValue::min()),
                    ),
                    (
                        MetricTorus2d(TorusValue::new(x2), TorusValue::max()),
                        MetricTorus2d(TorusValue::new(x1), TorusValue::new(y1)),
                    ),
                ]
            } else if dy == S::zero() {
                let (x0, y0, x1, y1) = if p.0 .0 < q.0 .0 {
                    (p.0 .0, p.1 .0, q.0 .0, q.1 .0)
                } else {
                    (q.0 .0, q.1 .0, p.0 .0, p.1 .0)
                };
                let y2 = (x0 * y1 - x1 * y0 + y0) / (x0 - x1 + S::one());
                vec![
                    (
                        MetricTorus2d(TorusValue::new(x0), TorusValue::new(y0)),
                        MetricTorus2d(TorusValue::min(), TorusValue::new(y2)),
                    ),
                    (
                        MetricTorus2d(TorusValue::max(), TorusValue::new(y2)),
                        MetricTorus2d(TorusValue::new(x1), TorusValue::new(y1)),
                    ),
                ]
            } else {
                let (x0, y0, x1, y1) = if p.0 .0 < q.0 .0 {
                    (p.0 .0, p.1 .0, q.0 .0, q.1 .0)
                } else {
                    (q.0 .0, q.1 .0, p.0 .0, p.1 .0)
                };
                let cx = x0 - x1 + S::one();
                let cy = if dx * dy < S::zero() {
                    y0 - y1 - S::one()
                } else {
                    y0 - y1 + S::one()
                };
                let x2 = if dx * dy < S::zero() {
                    (cy * x0 - cx * y0 + cx) / cy
                } else {
                    (cy * x0 - cx * y0) / cy
                };
                let y2 = (cx * y0 - cy * x0) / cx;
                if dx * dy < S::zero() {
                    if x2 < S::zero() {
                        vec![
                            (
                                MetricTorus2d(TorusValue::new(x0), TorusValue::new(y0)),
                                MetricTorus2d(TorusValue::min(), TorusValue::new(y2)),
                            ),
                            (
                                MetricTorus2d(TorusValue::max(), TorusValue::new(y2)),
                                MetricTorus2d(TorusValue::new(x2 + S::one()), TorusValue::max()),
                            ),
                            (
                                MetricTorus2d(TorusValue::new(x2 + S::one()), TorusValue::min()),
                                MetricTorus2d(TorusValue::new(x1), TorusValue::new(y1)),
                            ),
                        ]
                    } else {
                        vec![
                            (
                                MetricTorus2d(TorusValue::new(x0), TorusValue::new(y0)),
                                MetricTorus2d(TorusValue::new(x2), TorusValue::max()),
                            ),
                            (
                                MetricTorus2d(TorusValue::new(x2), TorusValue::min()),
                                MetricTorus2d(TorusValue::min(), TorusValue::new(y2 + S::one())),
                            ),
                            (
                                MetricTorus2d(TorusValue::max(), TorusValue::new(y2 + S::one())),
                                MetricTorus2d(TorusValue::new(x1), TorusValue::new(y1)),
                            ),
                        ]
                    }
                } else if y2 < S::zero() {
                    vec![
                        (
                            MetricTorus2d(TorusValue::new(x0), TorusValue::new(y0)),
                            MetricTorus2d(TorusValue::new(x2), TorusValue::min()),
                        ),
                        (
                            MetricTorus2d(TorusValue::new(x2), TorusValue::max()),
                            MetricTorus2d(TorusValue::min(), TorusValue::new(y2 + S::one())),
                        ),
                        (
                            MetricTorus2d(TorusValue::max(), TorusValue::new(y2 + S::one())),
                            MetricTorus2d(TorusValue::new(x1), TorusValue::new(y1)),
                        ),
                    ]
                } else {
                    vec![
                        (
                            MetricTorus2d(TorusValue::new(x0), TorusValue::new(y0)),
                            MetricTorus2d(TorusValue::min(), TorusValue::new(y2)),
                        ),
                        (
                            MetricTorus2d(TorusValue::max(), TorusValue::new(y2)),
                            MetricTorus2d(TorusValue::new(x2 + S::one()), TorusValue::min()),
                        ),
                        (
                            MetricTorus2d(TorusValue::new(x2 + S::one()), TorusValue::max()),
                            MetricTorus2d(TorusValue::new(x1), TorusValue::new(y1)),
                        ),
                    ]
                }
            }
        })
    }
}

impl<N, S> Drawing for DrawingTorus2d<N, S>
where
    N: DrawingIndex,
    S: DrawingValue,
{
    type Index = N;
    type Item = MetricTorus2d<S>;

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

    fn delta(&self, i: usize, j: usize) -> DeltaTorus2d<S> {
        self.raw_entry(i) - self.raw_entry(j)
    }
}
