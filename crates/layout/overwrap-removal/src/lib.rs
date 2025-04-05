use petgraph::visit::IntoNodeIdentifiers;
use petgraph_drawing::{Delta, Drawing, DrawingValue, Metric};

/// An algorithm for resolving node overlaps in graph drawings.
///
/// `OverwrapRemoval` iteratively adjusts node positions to ensure that nodes do not
/// overlap beyond what is expected based on their defined radii. Each node has an
/// associated radius, and when two nodes overlap (distance between centers is less than
/// the sum of their radii), they are pushed apart.
///
/// The algorithm distributes the displacement between nodes based on their relative sizes,
/// with smaller nodes moving more than larger ones.
///
/// # Type Parameters
///
/// * `S` - A numeric type implementing `DrawingValue` for scalar calculations
pub struct OverwrapRemoval<S> {
    /// Radii of nodes
    radius: Vec<S>,
    /// Strength of the repulsion force applied to overlapping nodes.
    /// Higher values result in stronger separation forces.
    /// Default value is 1.0.
    pub strength: S,
    /// Number of iterations to perform during the overlap removal process.
    /// More iterations can lead to better results but increase computation time.
    /// Default value is 1.
    pub iterations: usize,
    /// Minimum distance used to prevent division by zero when normalizing displacement vectors.
    /// Default value is 0.001.
    pub min_distance: S,
}

impl<S> OverwrapRemoval<S>
where
    S: DrawingValue,
{
    /// Creates a new `OverwrapRemoval` instance with the specified node radii.
    ///
    /// # Type Parameters
    ///
    /// * `G` - Graph type that implements `IntoNodeIdentifiers`
    /// * `F` - Function type that maps node IDs to their radii
    ///
    /// # Parameters
    ///
    /// * `graph` - The graph whose node overlaps will be removed
    /// * `radius` - A function that assigns a radius to each node
    ///
    /// # Returns
    ///
    /// A new `OverwrapRemoval` instance with default settings:
    /// * `strength` = 1.0
    /// * `iterations` = 1
    /// * `min_distance` = 0.001
    pub fn new<G, F>(graph: G, radius: F) -> OverwrapRemoval<S>
    where
        G: IntoNodeIdentifiers,
        F: FnMut(G::NodeId) -> S,
    {
        let radius = radius;
        OverwrapRemoval {
            radius: graph.node_identifiers().map(radius).collect::<Vec<_>>(),
            strength: S::one(),
            iterations: 1,
            min_distance: S::from_f32(1e-3).unwrap(),
        }
    }

    /// Applies the overlap removal algorithm to a graph drawing.
    ///
    /// Iteratively adjusts node positions to reduce overlaps between nodes.
    /// When two nodes overlap (their centers are closer than the sum of their radii),
    /// they are pushed apart with a force proportional to the amount of overlap and
    /// the `strength` parameter. The force is distributed between the nodes based on
    /// their relative sizes.
    ///
    /// # Type Parameters
    ///
    /// * `DR` - Drawing type implementing the `Drawing` trait
    /// * `M` - Metric type implementing the `Metric` trait
    /// * `D` - Delta type representing the difference between points
    ///
    /// # Parameters
    ///
    /// * `drawing` - The drawing to modify, with node positions that will be adjusted
    pub fn apply<DR, M, D>(&self, drawing: &mut DR)
    where
        DR: Drawing<Item = M>,
        M: Metric<D = D>,
        D: Delta<S = S>,
    {
        let n = drawing.len();
        for _ in 0..self.iterations {
            for i in 0..n {
                let ri = self.radius[i];
                for j in (i + 1)..n {
                    let rj = self.radius[j];
                    let delta1 = drawing.delta(i, j);
                    let delta2 = drawing.delta(i, j);
                    let r = ri + rj;
                    let l = delta1.norm().max(self.min_distance);
                    if l < r {
                        let d = (r - l) / l * self.strength;
                        let rr = (rj * rj) / (ri * ri + rj * rj);
                        *drawing.raw_entry_mut(i) += delta1 * (d * rr);
                        *drawing.raw_entry_mut(j) -= delta2 * (d * (S::one() - rr));
                    }
                }
            }
        }
    }
}
